// GPU per-base-triangle classification for the adaptive tessellation path.
//
// One thread per base triangle of every tessellated instance. For each triangle
// it computes a per-EDGE tessellation factor (screen-space edge length → segment
// count), classifies full / part / split, looks up the matching table config
// (sorted edges → `lookup_index`, with a permutation so the gen pass can map the
// pattern's barycentrics back onto the actual edges), and appends a
// `TessTriangleInfo` to the work list.
//
// Crack-free guarantee: an edge factor depends ONLY on its two world-space
// endpoints (symmetric `distance`), so two triangles sharing an edge derive the
// SAME factor → watertight tessellation.

struct Params {
    clip_from_world: mat4x4<f32>,
    viewport: vec2<f32>,        // framebuffer size in pixels
    px_per_segment: f32,        // target screen pixels per edge segment
    work_cluster_count: u32,
    max_size: u32,              // max segments per edge (11)
    max_size_configs: u32,      // lookup stride per axis (16)
    part_capacity: u32,
    _pad: u32,
}

// One base cluster of a tessellated instance.
struct WorkCluster {
    instance_idx: u32,
    index_base: u32,     // global index-pool offset (u32 units) of triangle 0
    triangle_count: u32,
    vertex_base: u32,    // global vertex-pool base; index VALUES are cluster-local
    part_base: u32,      // deterministic part-index base (Σ prior clusters' tri counts)
}

// Emitted per classified base triangle (32 bytes).
struct TessTriangleInfo {
    instance_index: u32,
    config_lookup: u32,   // index into the table's template_addresses / configs
    edge_perm: u32,       // original edge indices in sorted (x>=y>=z) order, 2 bits each
    i0: u32,              // the 3 global vertex indices (gen pass fetches pos/normal/uv)
    i1: u32,
    i2: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> work_clusters: array<WorkCluster>;
@group(0) @binding(2) var<storage, read> vertex_positions: array<f32>; // stride 3 floats
@group(0) @binding(3) var<storage, read> indices: array<u32>;
// counts[0] = emitted part count (the indirect INSTANTIATE's src_infos_count).
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> part_triangles: array<TessTriangleInfo>;
// `DispatchIndirectCommand` (x, y, z) for the vertex-gen pass: one workgroup per
// emitted part triangle. Written by `finalize` after `classify`.
@group(0) @binding(6) var<storage, read_write> gen_dispatch: array<u32>;
// Gathered cluster-indexed origin-relative world column (mat3x4 per instance slot) —
// the same `transforms` the closest-hit reads — plus each tess instance's cluster
// slot. Screen-space edge factors MUST project origin-relative world (the params'
// `clip_from_world` is itself origin-relative), else a moving camera re-derives wrong
// densities. `slots[instance_idx]` is the cluster slot, so it indexes `transforms`.
@group(0) @binding(7) var<storage, read> transforms: array<mat3x4<f32>>;
@group(0) @binding(8) var<storage, read> slots: array<u32>;

fn fetch_pos(i: u32) -> vec3<f32> {
    let b = i * 3u;
    return vec3<f32>(vertex_positions[b], vertex_positions[b + 1u], vertex_positions[b + 2u]);
}

fn to_world(m: mat3x4<f32>, p: vec3<f32>) -> vec3<f32> {
    let h = vec4<f32>(p, 1.0);
    return vec3<f32>(dot(m[0], h), dot(m[1], h), dot(m[2], h));
}

// Screen-space pixel position (offset-free: only the scale matters for edge length,
// and the offset cancels in a difference). Returns xy in half-pixel units.
fn clip_to_px(c: vec4<f32>) -> vec2<f32> {
    return (c.xy / c.w) * 0.5 * params.viewport;
}

// Per-edge segment count from the two endpoints' world positions. Symmetric in
// (a, b), so a shared edge yields the same factor for both triangles (crack-free).
fn edge_segments(a: vec3<f32>, b: vec3<f32>) -> u32 {
    let ca = params.clip_from_world * vec4<f32>(a, 1.0);
    let cb = params.clip_from_world * vec4<f32>(b, 1.0);
    // Either endpoint at/behind the near plane: the edge straddles the camera, so its
    // projected length is undefined (a point behind the camera projects to infinity).
    // That's the closest geometry ever gets → max out the tessellation.
    if ca.w <= 1e-5 || cb.w <= 1e-5 {
        return params.max_size;
    }
    let len = distance(clip_to_px(ca), clip_to_px(cb));
    let s = u32(round(len / max(params.px_per_segment, 1.0)));
    return clamp(s, 1u, params.max_size);
}

fn lookup_index(x: u32, y: u32, z: u32) -> u32 {
    let s = params.max_size_configs;
    return x + y * s + z * s * s - (1u + s + s * s);
}

@compute @workgroup_size(64)
fn classify(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let wc_idx = wg.x;
    if wc_idx >= params.work_cluster_count {
        return;
    }
    let wc = work_clusters[wc_idx];
    let inst = transforms[slots[wc.instance_idx]];

    // Stride loop over the cluster's base triangles (handles any cluster size).
    var t = lid;
    loop {
        if t >= wc.triangle_count {
            break;
        }
        let base = wc.index_base + t * 3u;
        // Stored index values are SOURCE-CLUSTER-LOCAL → rebase onto the global pool.
        let i0 = wc.vertex_base + indices[base];
        let i1 = wc.vertex_base + indices[base + 1u];
        let i2 = wc.vertex_base + indices[base + 2u];

        let p0 = to_world(inst, fetch_pos(i0));
        let p1 = to_world(inst, fetch_pos(i1));
        let p2 = to_world(inst, fetch_pos(i2));

        // Per-edge factors. Edge k spans vertex k → (k+1)%3.
        var e = array<u32, 3>(
            edge_segments(p0, p1),
            edge_segments(p1, p2),
            edge_segments(p2, p0),
        );

        // Sort edges descending → canonical (x>=y>=z) config; track the original
        // edge index at each rank for the gen pass's barycentric remap.
        var pi = array<u32, 3>(0u, 1u, 2u);
        if e[0] < e[1] {
            let te = e[0]; e[0] = e[1]; e[1] = te;
            let tp = pi[0]; pi[0] = pi[1]; pi[1] = tp;
        }
        if e[1] < e[2] {
            let te = e[1]; e[1] = e[2]; e[2] = te;
            let tp = pi[1]; pi[1] = pi[2]; pi[2] = tp;
        }
        if e[0] < e[1] {
            let te = e[0]; e[0] = e[1]; e[1] = te;
            let tp = pi[0]; pi[0] = pi[1]; pi[1] = tp;
        }
        let cfg = lookup_index(e[0], e[1], e[2]);
        let perm = pi[0] | (pi[1] << 2u) | (pi[2] << 4u);

        // DETERMINISTIC part slot (every base triangle → exactly one part), so a
        // part's cluster_id is stable frame-to-frame. The atomic only tallies the
        // total into counts[0] (the indirect INSTANTIATE's src_infos_count); the
        // write target is fixed. A part past capacity is dropped AND uncounted —
        // counts[0] must never exceed what `instantiate_infos` / the gen and CLAS
        // pools were sized for, or the raw build reads uninitialized descriptors.
        let idx = wc.part_base + t;
        if idx < params.part_capacity {
            atomicAdd(&counts[0], 1u);
            part_triangles[idx] = TessTriangleInfo(
                wc.instance_idx, cfg, perm, i0, i1, i2, 0u, 0u,
            );
        }

        t += 64u;
    }
}

// One-thread pass after `classify`: turn the emitted part count into the
// vertex-gen pass's indirect dispatch — one workgroup per part triangle. A scene
// can emit >65535 parts (the per-dimension grid limit), so tile across x AND y:
// `x = min(n, 65535)`, `y = ceil(n / 65535)`. The gen pass recovers the flat part
// index as `wg.x + wg.y * 65535` and bails past the real count / pool capacity.
@compute @workgroup_size(1)
fn finalize() {
    let n = min(atomicLoad(&counts[0]), params.part_capacity);
    gen_dispatch[0] = min(n, 65535u);
    gen_dispatch[1] = (n + 65534u) / 65535u;
    gen_dispatch[2] = 1u;
}
