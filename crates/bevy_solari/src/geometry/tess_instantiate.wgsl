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
    instance_index: u32,
    config_lookup: u32,
    edge_perm: u32,
    i0: u32,
    i1: u32,
    i2: u32,
    _pad0: u32,
    _pad1: u32,
}

struct InstParams {
    gen_base_lo: u32,
    gen_base_hi: u32,
    max_verts: u32,
    part_capacity: u32,
    // ClusterIDNV base baked into each tess CLAS (sentinel above the real cluster
    // pool so the closest-hit detects tess hits).
    cluster_id_base: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: InstParams;
@group(0) @binding(1) var<storage, read> part_triangles: array<TessTriangleInfo>;
// counts[0] = emitted part count.
@group(0) @binding(2) var<storage, read> counts: array<u32>;
// Table template addresses, u64 as 2 u32 per lookup slot.
@group(0) @binding(3) var<storage, read> template_addresses: array<u32>;
@group(0) @binding(4) var<storage, read_write> instantiate_infos: array<u32>;

// 64-bit add of a u64 (lo,hi) and a u32.
fn add_u64_u32(lo: u32, hi: u32, add: u32) -> vec2<u32> {
    let nlo = lo + add;
    let carry = select(0u, 1u, nlo < lo);
    return vec2<u32>(nlo, hi + carry);
}

// 64-bit a*b for u32 operands (returns lo,hi).
fn mul_u32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    let mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    let lo = (ll & 0xFFFFu) | (mid << 16u);
    let hi = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

fn add_u64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    return vec2<u32>(lo, a.y + b.y + carry);
}

@compute @workgroup_size(64)
fn build_infos(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    let count = min(counts[0], params.part_capacity);
    if p >= count {
        return;
    }
    let part = part_triangles[p];
    let tpl_lo = template_addresses[part.config_lookup * 2u];
    let tpl_hi = template_addresses[part.config_lookup * 2u + 1u];

    // vertex start = gen_base + part * max_verts * 12.
    let off = mul_u32(p, params.max_verts * 12u);
    let vstart = add_u64(vec2<u32>(params.gen_base_lo, params.gen_base_hi), off);

    let o = p * 8u;
    instantiate_infos[o + 0u] = params.cluster_id_base + p; // unique ClusterIDNV
    instantiate_infos[o + 1u] = 0u;                          // geometry_index_offset
    instantiate_infos[o + 2u] = tpl_lo;
    instantiate_infos[o + 3u] = tpl_hi;
    instantiate_infos[o + 4u] = vstart.x;
    instantiate_infos[o + 5u] = vstart.y;
    instantiate_infos[o + 6u] = 12u;                         // stride lo
    instantiate_infos[o + 7u] = 0u;                          // stride hi
}
