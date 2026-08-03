// GPU micro-vertex generation for the adaptive tessellation path.
//
// One workgroup per emitted part triangle (indirect, sized by `classify`'s
// `finalize`), one thread per config micro-vertex. Each thread reads the part's
// table config, fetches the matching barycentric (UV-packed) from the table, and —
// for a mirror config (`flip`) — swaps `wuv.yxz` (mirroring `vk_tessellated_clusters`'
// `tess_getConfigVertexBarycentrics`). classify already rotated the base vertices to
// the canonical order, so the barycentric interpolates straight over `i0/i1/i2`. It
// then displaces along the interpolated normal and writes the OBJECT-space position
// into `gen_vertices` at a fixed `max_verts` stride per part. The instantiate pass
// builds each part's config template against its slice of `gen_vertices`; the PTLAS
// instance transform (`world_rel[slot]`) places the BLAS in the world.
//
// Binding split: the per-dispatch buffers sit in SET 1 (push-index-mapped slot
// array, `HeapKernel::new_with_mappings`); the displacement-map array + sampler
// sit in SET 0 as constant-offset heap rows over the block
// `tess_classify.rs` maintains (an arrayed mapping reads consecutive heap
// descriptors from the block base).

import octahedral;

struct TessTriangleInfo {
    uint instance_index;
    uint config_lookup;
    uint flip;          // 1 = mirror config; swap barycentrics `wuv.yxz`
    uint i0;
    uint i1;
    uint i2;
    uint _pad0;
    uint _pad1;
}

// Mirrors `tess_classify.rs::GenParams` (32 B).
struct GenParams {
    float displacement_scale;
    float displacement_bias;
    uint max_verts;         // fixed gen_vertices stride per part
    uint has_displacement;
    uint part_capacity;     // gen_vertices slot bound (2D dispatch over-covers)
    uint _pad0;
    uint _pad1;
    uint _pad2;
}

[[vk::push_constant]] GenParams params;

[[vk::binding(0, 1)]] StructuredBuffer<TessTriangleInfo> part_triangles;
// Config entries: 2 u32 per entry. [0] = first_triangle | (first_vertex << 16),
// [1] = num_triangles | (num_vertices << 16).
[[vk::binding(1, 1)]] StructuredBuffer<uint> configs;
// Table barycentrics, UV-packed (1<<15 == 1.0).
[[vk::binding(2, 1)]] StructuredBuffer<uint> table_vertices;
// Base mesh positions (stride 3 f32) and packed attrs (4 u32/vertex: normal[0], uv[2..3]).
[[vk::binding(3, 1)]] StructuredBuffer<float> base_positions;
[[vk::binding(4, 1)]] StructuredBuffer<uint> base_packed;
// Output object-space micro-vertices, stride 3 f32, slot = part*max_verts + v.
[[vk::binding(5, 1)]] RWStructuredBuffer<float> gen_vertices;
// Per-instance displacement maps, indexed by the part's `instance_index` (one part
// per workgroup ⇒ the index is uniform across the workgroup). The size MUST match
// `tess_classify.rs::MAX_TESS_DISPLACEMENT_MAPS` (the heap block); unused slots
// hold fallback descriptors — a heap array has no "partially bound".
[[vk::binding(6, 0)]] Texture2D displacement_maps[256];
[[vk::binding(7, 0)]] SamplerState displacement_sampler;

float3 load_pos(uint i) {
    let b = i * 3;
    return float3(base_positions[b], base_positions[b + 1], base_positions[b + 2]);
}
float3 load_normal(uint i) {
    return octahedral_decode_signed(unpack_snorm2x16(base_packed[i * 4]));
}
float2 load_uv(uint i) {
    let b = i * 4;
    return float2(asfloat(base_packed[b + 2]), asfloat(base_packed[b + 3]));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void gen_verts(uint3 wg: SV_GroupID, uint lid: SV_GroupIndex) {
    // Flat part index from the 2D over-covering grid (finalize: x≤65535, y=ceil(n/65535)).
    let p = wg.x + wg.y * 65535;
    if (p >= params.part_capacity) {
        return;
    }
    let part = part_triangles[p];
    let c0 = configs[part.config_lookup * 2];
    let c1 = configs[part.config_lookup * 2 + 1];
    let first_vert = c0 >> 16;
    let num_verts = c1 >> 16;
    if (lid >= num_verts) {
        return;
    }

    let bp = table_vertices[first_vert + lid];
    let uu = float(bp & 0xFFFF) / 32768.0;
    let vv = float(bp >> 16) / 32768.0;
    // Canonical weights (w, u, v): corner0 @ (0,0), corner1 @ (1,0), corner2 @ (0,1).
    var b = float3(1.0 - uu - vv, uu, vv);
    // Mirror config: swap `wuv.yxz` (matches the flipped template's reversed winding
    // so the net micro-triangle stays consistently wound). base verts pre-rotated.
    if (part.flip == 1) {
        b = b.yxz;
    }

    let p0 = load_pos(part.i0);
    let p1 = load_pos(part.i1);
    let p2 = load_pos(part.i2);
    let n0 = load_normal(part.i0);
    let n1 = load_normal(part.i1);
    let n2 = load_normal(part.i2);
    let t0 = load_uv(part.i0);
    let t1 = load_uv(part.i1);
    let t2 = load_uv(part.i2);

    var pos = b.x * p0 + b.y * p1 + b.z * p2;
    let nrm = normalize(b.x * n0 + b.y * n1 + b.z * n2);
    let uv = b.x * t0 + b.y * t1 + b.z * t2;

    var height = 0.0;
    if (params.has_displacement == 1) {
        height = saturate(displacement_maps[part.instance_index]
            .SampleLevel(displacement_sampler, uv, 0.0).r);
    }
    // Brighter texel = deeper (bevy `depth_map`): recede along -normal.
    pos += nrm * (params.displacement_bias - height * params.displacement_scale);

    // Object-space micro-vertex — the PTLAS instance transform places it in the
    // floating-origin world at TLAS build, and the closest-hit re-applies that same
    // ObjectToWorld to the position-fetched vertices (so shading lands in world too).
    let o = (p * params.max_verts + lid) * 3;
    gen_vertices[o + 0] = pos.x;
    gen_vertices[o + 1] = pos.y;
    gen_vertices[o + 2] = pos.z;
}
