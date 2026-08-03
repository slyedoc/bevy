// GPU per-base-triangle classification for the adaptive tessellation path.
//
// One thread per base triangle of every tessellated instance. For each triangle
// it computes a per-EDGE tessellation factor (screen-space edge length → segment
// count), classifies full / part / split, canonicalizes the edge factors to a table
// config (rotate the max factor to the front + a mirror `flip` bit, per
// `vk_tessellated_clusters` `tess_getConfig`), and appends a `TessTriangleInfo` to
// the work list. The one-thread `finalize` entry then turns the emitted part count
// into the gen passes' indirect dispatch args.
//
// Crack-free guarantee: an edge factor depends ONLY on its two world-space
// endpoints (symmetric `distance`), so two triangles sharing an edge derives the
// SAME factor → watertight tessellation.

// Mirrors `tess_classify.rs::ClassifyParams` (96 B). `clip_from_world` rides as
// 4 explicit float4 COLUMNS (`bevy_math::Mat4`'s column-major bytes), applied in
// `to_clip` — no matrix type, so no host/Slang matrix-layout convention to match.
struct ClassifyParams {
    float4 clip_c0;
    float4 clip_c1;
    float4 clip_c2;
    float4 clip_c3;
    float2 viewport;        // framebuffer size in pixels
    float px_per_segment;   // target screen pixels per edge segment
    uint work_cluster_count;
    uint max_size;          // max segments per edge (11)
    uint max_size_configs;  // lookup stride per axis (16)
    uint part_capacity;
    uint _pad;
}

[[vk::push_constant]] ClassifyParams params;

// One base cluster of a tessellated instance.
struct WorkCluster {
    uint instance_idx;
    uint index_base;     // global index-pool offset (u32 units) of triangle 0
    uint triangle_count;
    uint vertex_base;    // global vertex-pool base; index VALUES are cluster-local
    uint part_base;      // deterministic part-index base (Σ prior clusters' tri counts)
}

// Emitted per classified base triangle (32 bytes).
struct TessTriangleInfo {
    uint instance_index;
    uint config_lookup;  // table slot: canonical (y>=z) or its mirror (z>y = flipped template)
    uint flip;           // 1 = mirror config; gen/attrs swap barycentrics `wuv.yxz`
    uint i0;             // the 3 global vertex indices, ROTATED to canonical (max factor first)
    uint i1;
    uint i2;
    uint _pad0;
    uint _pad1;
}

// A `mat3x4` affine: `r[k]` is row k of the 4x4 (the gathered transform
// column's 48-B packing).
struct Affine3x4 {
    float4 r0;
    float4 r1;
    float4 r2;
}

[[vk::binding(0, 0)]] StructuredBuffer<WorkCluster> work_clusters;
[[vk::binding(1, 0)]] StructuredBuffer<float> vertex_positions; // stride 3 floats
[[vk::binding(2, 0)]] StructuredBuffer<uint> indices;
// counts[0] = emitted part count (the indirect INSTANTIATE's src_infos_count).
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> counts;
[[vk::binding(4, 0)]] RWStructuredBuffer<TessTriangleInfo> part_triangles;
// `DispatchIndirectCommand` (x, y, z) for the vertex-gen pass: one workgroup per
// emitted part triangle. Written by `finalize` after `classify`.
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> gen_dispatch;
// Gathered cluster-indexed origin-relative world column (mat3x4 per instance slot) —
// the same `transforms` the closest-hit reads — plus each tess instance's cluster
// slot. Screen-space edge factors MUST project origin-relative world (the params'
// clip-from-world is itself origin-relative), else a moving camera re-derives wrong
// densities. `slots[instance_idx]` is the cluster slot, so it indexes `transforms`.
[[vk::binding(6, 0)]] StructuredBuffer<Affine3x4> transforms;
[[vk::binding(7, 0)]] StructuredBuffer<uint> slots;

float3 fetch_pos(uint i) {
    let b = i * 3;
    return float3(vertex_positions[b], vertex_positions[b + 1], vertex_positions[b + 2]);
}

float3 to_world(Affine3x4 m, float3 p) {
    let h = float4(p, 1.0);
    return float3(dot(m.r0, h), dot(m.r1, h), dot(m.r2, h));
}

float4 to_clip(float3 p) {
    return params.clip_c0 * p.x + params.clip_c1 * p.y + params.clip_c2 * p.z
        + params.clip_c3;
}

// Screen-space pixel position (offset-free: only the scale matters for edge length,
// and the offset cancels in a difference). Returns xy in half-pixel units.
float2 clip_to_px(float4 c) {
    return (c.xy / c.w) * 0.5 * params.viewport;
}

// Per-edge segment count from the two endpoints' world positions. Symmetric in
// (a, b), so a shared edge yields the same factor for both triangles (crack-free).
uint edge_segments(float3 a, float3 b) {
    let ca = to_clip(a);
    let cb = to_clip(b);
    // Either endpoint at/behind the near plane: the edge straddles the camera, so its
    // projected length is undefined (a point behind the camera projects to infinity).
    // That's the closest geometry ever gets → max out the tessellation.
    if (ca.w <= 1e-5 || cb.w <= 1e-5) {
        return params.max_size;
    }
    let len = distance(clip_to_px(ca), clip_to_px(cb));
    let s = uint(round(len / max(params.px_per_segment, 1.0)));
    return clamp(s, 1u, params.max_size);
}

uint lookup_index(uint x, uint y, uint z) {
    let s = params.max_size_configs;
    return x + y * s + z * s * s - (1 + s + s * s);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void classify(uint3 wg: SV_GroupID, uint lid: SV_GroupIndex) {
    let wc_idx = wg.x;
    if (wc_idx >= params.work_cluster_count) {
        return;
    }
    let wc = work_clusters[wc_idx];
    let inst = transforms[slots[wc.instance_idx]];

    // Stride loop over the cluster's base triangles (handles any cluster size).
    for (uint t = lid; t < wc.triangle_count; t += 64) {
        let base = wc.index_base + t * 3;
        // Stored index values are SOURCE-CLUSTER-LOCAL → rebase onto the global pool.
        let i0 = wc.vertex_base + indices[base];
        let i1 = wc.vertex_base + indices[base + 1];
        let i2 = wc.vertex_base + indices[base + 2];

        let p0 = to_world(inst, fetch_pos(i0));
        let p1 = to_world(inst, fetch_pos(i1));
        let p2 = to_world(inst, fetch_pos(i2));

        // Per-edge factors. Edge k spans vertex k → (k+1)%3.
        uint e[3] = { edge_segments(p0, p1), edge_segments(p1, p2), edge_segments(p2, p0) };
        uint iv[3] = { i0, i1, i2 };

        // Canonicalize like `vk_tessellated_clusters` `tess_getConfig`: bring the max
        // factor to slot 0 by a winding-preserving ROTATION (rotating the base vertices
        // with it), then set a single flip bit if the remaining two are ascending
        // (z > y = the mirror of a canonical config). The rotation resolves ties without
        // a reflection, so `y == z` never needs a flip (which the table's mirror slots
        // don't cover). `config_lookup` then indexes the canonical slot (y>=z) or its
        // mirror slot (z>y → the flipped template).
        let maxf = max(max(e[0], e[1]), e[2]);
        if (maxf == e[1]) {
            uint re[3] = { e[1], e[2], e[0] };      // .yzx
            uint ri[3] = { iv[1], iv[2], iv[0] };
            e = re;
            iv = ri;
        } else if (maxf == e[2]) {
            uint re[3] = { e[2], e[0], e[1] };      // .zxy
            uint ri[3] = { iv[2], iv[0], iv[1] };
            e = re;
            iv = ri;
        }
        let cfg = lookup_index(e[0], e[1], e[2]);
        let flip = e[2] > e[1] ? 1u : 0u;

        // DETERMINISTIC part slot (every base triangle → exactly one part), so a
        // part's cluster_id is stable frame-to-frame. The atomic only tallies the
        // total into counts[0] (the indirect INSTANTIATE's src_infos_count); the
        // write target is fixed. A part past capacity is dropped AND uncounted —
        // counts[0] must never exceed what `instantiate_infos` / the gen and CLAS
        // pools were sized for, or the raw build reads uninitialized descriptors.
        let idx = wc.part_base + t;
        if (idx < params.part_capacity) {
            InterlockedAdd(counts[0], 1);
            TessTriangleInfo info;
            info.instance_index = wc.instance_idx;
            info.config_lookup = cfg;
            info.flip = flip;
            info.i0 = iv[0];
            info.i1 = iv[1];
            info.i2 = iv[2];
            info._pad0 = 0;
            info._pad1 = 0;
            part_triangles[idx] = info;
        }
    }
}

// One-thread pass after `classify`: turn the emitted part count into the
// vertex-gen pass's indirect dispatch — one workgroup per part triangle. A scene
// can emit >65535 parts (the per-dimension grid limit), so tile across x AND y:
// `x = min(n, 65535)`, `y = ceil(n / 65535)`. The gen pass recovers the flat part
// index as `wg.x + wg.y * 65535` and bails past the real count / pool capacity.
[shader("compute")]
[numthreads(1, 1, 1)]
void finalize() {
    let n = min(counts[0], params.part_capacity);
    gen_dispatch[0] = min(n, 65535u);
    gen_dispatch[1] = (n + 65534) / 65535;
    gen_dispatch[2] = 1;
}
