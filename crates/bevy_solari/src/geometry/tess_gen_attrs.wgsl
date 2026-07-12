// Per-micro-triangle smooth normals + UVs for the GPU tessellation path, in the
// DENORMALIZED layout the closest-hit's smooth-tess branch reads
// (`resolve_triangle_data_full_mat_fetch`, the
// `geometry_addresses.tess_clusters != 0` path).
//
// One workgroup per emitted part (same indirect grid as `tess_gen_verts`), one
// thread per micro-triangle of the part's table config. For each micro-triangle it
// fetches the config's 3 local micro-vertex indices (table topology), remaps each
// onto the base triangle's edges (the classify permutation, identical to
// `tess_gen_verts` so attrs stay consistent with the generated positions),
// interpolates the base triangle's smooth normal + UV, and writes them
// denormalized: 3 × (packed-octahedral normal @0, uv @4) = 36 B per micro-triangle,
// contiguous at `(part * max_tris + micro_tri) * 9` u32. Thread 0 also writes the
// part's metadata record (attr-buffer address + `primitive_base = part * max_tris`).
//
// Normals are the interpolated base normals (no displacement-gradient bump).

#import bevy_render::utils::{octahedral_decode_signed, octahedral_encode}

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

struct Instance {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

struct AttrParams {
    // Device address of `gen_attrs` (lo/hi) — baked into each part's metadata record
    // so the closest-hit can `physical_load` the denormalized attrs.
    attr_addr_lo: u32,
    attr_addr_hi: u32,
    // Fixed per-part micro-triangle stride (= table max_triangles); part p's attrs
    // occupy `[p*max_tris, p*max_tris + num_tris)`.
    max_tris: u32,
    // gen_attrs slot bound (2D dispatch over-covers); threads past it bail.
    part_capacity: u32,
}

@group(0) @binding(0) var<uniform> params: AttrParams;
@group(0) @binding(1) var<storage, read> part_triangles: array<TessTriangleInfo>;
// Config entries: 2 u32 each. [0] = first_triangle | (first_vertex << 16),
// [1] = num_triangles | (num_vertices << 16).
@group(0) @binding(2) var<storage, read> configs: array<u32>;
// Table topology: per micro-triangle, 3 local micro-vertex indices packed 3×8-bit.
@group(0) @binding(3) var<storage, read> table_indices: array<u32>;
// Table barycentrics, UV-packed (1<<15 == 1.0).
@group(0) @binding(4) var<storage, read> table_vertices: array<u32>;
// Base mesh packed attrs, 4 u32 / vertex: normal[0], tangent[1], uv[2..3].
@group(0) @binding(5) var<storage, read> base_packed: array<u32>;
// Denormalized output: 3 × (packed normal + uv) per micro-triangle.
@group(0) @binding(6) var<storage, read_write> gen_attrs: array<u32>;
// Per-part metadata (16 B): attr address (lo/hi) + primitive_base + pad.
@group(0) @binding(7) var<storage, read_write> part_meta: array<u32>;
// Instance object→world affine (row-major mat3x4); rotates the object normal to
// world, since gen bakes WORLD positions and the PTLAS inject is identity.
@group(0) @binding(8) var<storage, read> instances: array<Instance>;

// Rotate an object-space normal to world by the instance's linear part. Assumes
// rotation + (near-)uniform scale (true for the showcase instances), so the basis
// rotates the normal directly; re-normalized after.
fn normal_to_world(inst: Instance, n: vec3<f32>) -> vec3<f32> {
    let m0 = vec3<f32>(inst.r0.x, inst.r0.y, inst.r0.z);
    let m1 = vec3<f32>(inst.r1.x, inst.r1.y, inst.r1.z);
    let m2 = vec3<f32>(inst.r2.x, inst.r2.y, inst.r2.z);
    return normalize(vec3<f32>(dot(m0, n), dot(m1, n), dot(m2, n)));
}

fn load_normal(i: u32) -> vec3<f32> {
    return octahedral_decode_signed(unpack2x16snorm(base_packed[i * 4u]));
}
fn load_uv(i: u32) -> vec2<f32> {
    let b = i * 4u;
    return vec2<f32>(bitcast<f32>(base_packed[b + 2u]), bitcast<f32>(base_packed[b + 3u]));
}
fn pack_normal(n: vec3<f32>) -> u32 {
    return pack2x16snorm(octahedral_encode(n) * 2.0 - 1.0);
}

// Reflection-aware barycentric remap — MUST match `tess_gen_verts.wgsl` exactly so
// attrs land on the same micro-vertices as the generated positions. See that file for
// the derivation (vertex permutation σ from the edge permutation).
fn edge_common(a: u32, b: u32) -> u32 {
    let a1 = (a + 1u) % 3u;
    let b1 = (b + 1u) % 3u;
    if a == b || a == b1 {
        return a;
    }
    return a1;
}
fn remap_bary(w: vec3<f32>, perm: u32) -> vec3<f32> {
    let pi0 = perm & 3u;
    let pi1 = (perm >> 2u) & 3u;
    let pi2 = (perm >> 4u) & 3u;
    let s0 = edge_common(pi2, pi0);
    let s1 = edge_common(pi0, pi1);
    let s2 = edge_common(pi1, pi2);
    let ox = select(select(w.z, w.y, s1 == 0u), w.x, s0 == 0u);
    let oy = select(select(w.z, w.y, s1 == 1u), w.x, s0 == 1u);
    let oz = select(select(w.z, w.y, s1 == 2u), w.x, s0 == 2u);
    return vec3<f32>(ox, oy, oz);
}

@compute @workgroup_size(128)
fn gen_attrs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let p = wg.x + wg.y * 65535u;
    if p >= params.part_capacity {
        return;
    }
    let part = part_triangles[p];
    let c0 = configs[part.config_lookup * 2u];
    let c1 = configs[part.config_lookup * 2u + 1u];
    let first_tri = c0 & 0xFFFFu;
    let first_vert = c0 >> 16u;
    let num_tris = c1 & 0xFFFFu;

    // Thread 0 writes the part's metadata record (read by the closest-hit at
    // `(cluster_id - TESS_CLUSTER_ID_BASE) * 16`).
    if lid == 0u {
        let mo = p * 4u;
        part_meta[mo + 0u] = params.attr_addr_lo;
        part_meta[mo + 1u] = params.attr_addr_hi;
        part_meta[mo + 2u] = p * params.max_tris; // primitive_base
        part_meta[mo + 3u] = 0u;
    }
    if lid >= num_tris {
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
    var lv = array<u32, 3>(tri & 0xFFu, (tri >> 8u) & 0xFFu, (tri >> 16u) & 0xFFu);

    let o = (p * params.max_tris + micro_tri) * 9u;
    for (var k = 0u; k < 3u; k = k + 1u) {
        let bp = table_vertices[first_vert + lv[k]];
        let uu = f32(bp & 0xFFFFu) / 32768.0;
        let vv = f32(bp >> 16u) / 32768.0;
        var b = vec3<f32>(1.0 - uu - vv, uu, vv);
        b = remap_bary(b, part.edge_perm);
        let n_obj = normalize(b.x * n0 + b.y * n1 + b.z * n2);
        let nrm = normal_to_world(instances[part.instance_index], n_obj);
        let uv = b.x * t0 + b.y * t1 + b.z * t2;
        gen_attrs[o + k * 3u + 0u] = pack_normal(nrm);
        gen_attrs[o + k * 3u + 1u] = bitcast<u32>(uv.x);
        gen_attrs[o + k * 3u + 2u] = bitcast<u32>(uv.y);
    }
}
