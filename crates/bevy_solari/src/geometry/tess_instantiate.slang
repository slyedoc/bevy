// Build one `VkClusterAccelerationStructureInstantiateClusterInfoNV` per emitted
// part triangle, so the raw-VK indirect INSTANTIATE turns each part's config
// template + its `gen_vertices` slice into a CLAS.
//
// The descriptor is 32 bytes / 8 u32 (matches the ash struct):
//   [0] cluster_id_offset
//   [1] geometry_index_offset_and_reserved (Packed24_8, 0)
//   [2..3] cluster_template_address (u64)
//   [4..5] vertex_buffer.start_address (u64) = gen_base + part*max_verts*12
//   [6..7] vertex_buffer.stride_in_bytes (u64) = 12

struct TessTriangleInfo {
    uint instance_index;
    uint config_lookup; // canonical or mirror slot → normal / flipped template
    uint flip;
    uint i0;
    uint i1;
    uint i2;
    uint _pad0;
    uint _pad1;
}

struct InstParams {
    uint gen_base_lo;
    uint gen_base_hi;
    uint max_verts;
    uint part_capacity;
    // ClusterIDNV base baked into each tess CLAS (sentinel above the real cluster
    // pool so the closest-hit detects tess hits).
    uint cluster_id_base;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}

[[vk::push_constant]] InstParams params;

[[vk::binding(0, 0)]] StructuredBuffer<TessTriangleInfo> part_triangles;
// counts[0] = emitted part count.
[[vk::binding(1, 0)]] StructuredBuffer<uint> counts;
// Table template addresses, u64 as 2 u32 per lookup slot.
[[vk::binding(2, 0)]] StructuredBuffer<uint> template_addresses;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> instantiate_infos;

// 64-bit add of a u64 (lo,hi) and a u32.
uint2 add_u64_u32(uint lo, uint hi, uint add) {
    let nlo = lo + add;
    let carry = nlo < lo ? 1u : 0u;
    return uint2(nlo, hi + carry);
}

// 64-bit a*b for u32 operands (returns lo,hi).
uint2 mul_u32(uint a, uint b) {
    let a_lo = a & 0xFFFF;
    let a_hi = a >> 16;
    let b_lo = b & 0xFFFF;
    let b_hi = b >> 16;
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    let mid = (ll >> 16) + (lh & 0xFFFF) + (hl & 0xFFFF);
    let lo = (ll & 0xFFFF) | (mid << 16);
    let hi = hh + (lh >> 16) + (hl >> 16) + (mid >> 16);
    return uint2(lo, hi);
}

uint2 add_u64(uint2 a, uint2 b) {
    let lo = a.x + b.x;
    let carry = lo < a.x ? 1u : 0u;
    return uint2(lo, a.y + b.y + carry);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_infos(uint3 gid: SV_DispatchThreadID) {
    let p = gid.x;
    let count = min(counts[0], params.part_capacity);
    if (p >= count) {
        return;
    }
    let part = part_triangles[p];
    let tpl_lo = template_addresses[part.config_lookup * 2];
    let tpl_hi = template_addresses[part.config_lookup * 2 + 1];

    // vertex start = gen_base + part * max_verts * 12.
    let off = mul_u32(p, params.max_verts * 12);
    let vstart = add_u64(uint2(params.gen_base_lo, params.gen_base_hi), off);

    let o = p * 8;
    instantiate_infos[o + 0] = params.cluster_id_base + p; // unique ClusterIDNV
    instantiate_infos[o + 1] = 0;                          // geometry_index_offset
    instantiate_infos[o + 2] = tpl_lo;
    instantiate_infos[o + 3] = tpl_hi;
    instantiate_infos[o + 4] = vstart.x;
    instantiate_infos[o + 5] = vstart.y;
    instantiate_infos[o + 6] = 12;                         // stride lo
    instantiate_infos[o + 7] = 0;                          // stride hi
}
