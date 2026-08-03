// Per-micro-triangle smooth normals + UVs for the GPU tessellation path, in the
// DENORMALIZED layout the closest-hit's smooth-tess branch reads
// (`resolve_triangle_data_full_mat_fetch`, the
// `geometry_addresses.tess_clusters != 0` path).
//
// One workgroup per emitted part (same indirect grid as `tess_gen_verts`), one
// thread per micro-triangle of the part's table config. For each micro-triangle it
// fetches the config's 3 local micro-vertex indices (table topology), applies the
// same mirror swap (`wuv.yxz` on `flip`) as `tess_gen_verts` so attrs land on the
// generated positions, interpolates the base triangle's smooth normal + UV, and
// writes them denormalized: 3 × (packed-octahedral normal @0, uv @4) = 36 B per
// micro-triangle, contiguous at `(part * max_tris + micro_tri) * 9` u32. Thread 0
// also writes the part's metadata record (attr-buffer address + `primitive_base`).
//
// Normals are the interpolated base normals (no displacement-gradient bump).

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

// Mirrors `tess_classify.rs::AttrParams` (16 B).
struct AttrParams {
    // Device address of `gen_attrs` (lo/hi) — baked into each part's metadata record
    // so the closest-hit can `physical_load` the denormalized attrs.
    uint attr_addr_lo;
    uint attr_addr_hi;
    // Fixed per-part micro-triangle stride (= table max_triangles); part p's attrs
    // occupy `[p*max_tris, p*max_tris + num_tris)`.
    uint max_tris;
    // gen_attrs slot bound (2D dispatch over-covers); threads past it bail.
    uint part_capacity;
}

[[vk::push_constant]] AttrParams params;

[[vk::binding(0, 0)]] StructuredBuffer<TessTriangleInfo> part_triangles;
// Config entries: 2 u32 each. [0] = first_triangle | (first_vertex << 16),
// [1] = num_triangles | (num_vertices << 16).
[[vk::binding(1, 0)]] StructuredBuffer<uint> configs;
// Table topology: per micro-triangle, 3 local micro-vertex indices packed 3×8-bit.
[[vk::binding(2, 0)]] StructuredBuffer<uint> table_indices;
// Table barycentrics, UV-packed (1<<15 == 1.0).
[[vk::binding(3, 0)]] StructuredBuffer<uint> table_vertices;
// Base mesh packed attrs, 4 u32 / vertex: normal[0], tangent[1], uv[2..3].
[[vk::binding(4, 0)]] StructuredBuffer<uint> base_packed;
// Denormalized output: 3 × (packed normal + uv) per micro-triangle.
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> gen_attrs;
// Per-part metadata (16 B): attr address (lo/hi) + primitive_base + pad.
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> part_meta;

float3 load_normal(uint i) {
    return octahedral_decode_signed(unpack_snorm2x16(base_packed[i * 4]));
}
float2 load_uv(uint i) {
    let b = i * 4;
    return float2(asfloat(base_packed[b + 2]), asfloat(base_packed[b + 3]));
}
uint pack_normal(float3 n) {
    return pack_snorm2x16(octahedral_encode(n) * 2.0 - 1.0);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void gen_attrs_main(uint3 wg: SV_GroupID, uint lid: SV_GroupIndex) {
    let p = wg.x + wg.y * 65535;
    if (p >= params.part_capacity) {
        return;
    }
    let part = part_triangles[p];
    let c0 = configs[part.config_lookup * 2];
    let c1 = configs[part.config_lookup * 2 + 1];
    let first_tri = c0 & 0xFFFF;
    let first_vert = c0 >> 16;
    let num_tris = c1 & 0xFFFF;

    // Thread 0 writes the part's metadata record (read by the closest-hit at
    // `(cluster_id - TESS_CLUSTER_ID_BASE) * 16`).
    if (lid == 0) {
        let mo = p * 4;
        part_meta[mo + 0] = params.attr_addr_lo;
        part_meta[mo + 1] = params.attr_addr_hi;
        part_meta[mo + 2] = p * params.max_tris; // primitive_base
        part_meta[mo + 3] = 0;
    }
    if (lid >= num_tris) {
        return;
    }
    let micro_tri = lid;

    let n0 = load_normal(part.i0);
    let n1 = load_normal(part.i1);
    let n2 = load_normal(part.i2);
    let t0 = load_uv(part.i0);
    let t1 = load_uv(part.i1);
    let t2 = load_uv(part.i2);

    // The micro-triangle's 3 local micro-vertex indices (packed 3×8-bit).
    let tri = table_indices[first_tri + micro_tri];
    uint lv[3] = { tri & 0xFF, (tri >> 8) & 0xFF, (tri >> 16) & 0xFF };
    // Mirror config: the flipped template swaps each micro-triangle's last two
    // vertices (`.xzy`), so bake the attrs in that order to align with the CLAS's
    // position-fetched vertices the closest-hit interpolates against.
    if (part.flip == 1) {
        uint rv[3] = { lv[0], lv[2], lv[1] };
        lv = rv;
    }

    let o = (p * params.max_tris + micro_tri) * 9;
    for (uint k = 0; k < 3; k++) {
        let bp = table_vertices[first_vert + lv[k]];
        let uu = float(bp & 0xFFFF) / 32768.0;
        let vv = float(bp >> 16) / 32768.0;
        var b = float3(1.0 - uu - vv, uu, vv);
        // Same barycentric mirror as `tess_gen_verts` so attrs land on the positions.
        if (part.flip == 1) {
            b = b.yxz;
        }
        // Object-space normal: gen bakes object-space positions, and the closest-hit
        // applies the instance's ObjectToWorld (the PTLAS transform = `world_rel[slot]`)
        // to BOTH the position-fetched vertices and this normal — so baking world here
        // would double-rotate. Leave it in object space.
        let nrm = normalize(b.x * n0 + b.y * n1 + b.z * n2);
        let uv = b.x * t0 + b.y * t1 + b.z * t2;
        gen_attrs[o + k * 3 + 0] = pack_normal(nrm);
        gen_attrs[o + k * 3 + 1] = asuint(uv.x);
        gen_attrs[o + k * 3 + 2] = asuint(uv.y);
    }
}
