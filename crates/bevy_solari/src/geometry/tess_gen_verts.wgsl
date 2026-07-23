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

#import bevy_render::utils::octahedral_decode_signed

struct TessTriangleInfo {
    instance_index: u32,
    config_lookup: u32,
    flip: u32,          // 1 = mirror config; swap barycentrics `wuv.yxz`
    i0: u32,
    i1: u32,
    i2: u32,
    _pad0: u32,
    _pad1: u32,
}

struct GenParams {
    displacement_scale: f32,
    displacement_bias: f32,
    max_verts: u32,         // fixed gen_vertices stride per part
    has_displacement: u32,
    part_capacity: u32,     // gen_vertices slot bound (2D dispatch over-covers)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Storage (not uniform): a bind group may not mix a uniform buffer with the
// displacement binding array, so params ride a storage buffer instead.
@group(0) @binding(0) var<storage, read> params: GenParams;
@group(0) @binding(1) var<storage, read> part_triangles: array<TessTriangleInfo>;
// Config entries: 2 u32 per entry. [0] = first_triangle | (first_vertex << 16),
// [1] = num_triangles | (num_vertices << 16).
@group(0) @binding(2) var<storage, read> configs: array<u32>;
// Table barycentrics, UV-packed (1<<15 == 1.0).
@group(0) @binding(3) var<storage, read> table_vertices: array<u32>;
// Base mesh positions (stride 3 f32) and packed attrs (4 u32/vertex: normal[0], uv[2..3]).
@group(0) @binding(4) var<storage, read> base_positions: array<f32>;
@group(0) @binding(5) var<storage, read> base_packed: array<u32>;
// Per-instance displacement maps, indexed by the part's `instance_index` (one part
// per workgroup ⇒ the index is uniform across the workgroup). SIZED (not unsized): an
// unsized binding_array compiles to an OpTypeRuntimeArray that a plain compute
// pipeline can't instantiate without RuntimeDescriptorArray; the fixed size matches
// the layout's `.count()` (MAX_TESS_DISPLACEMENT_MAPS) and is partially bound. The
// bevy bindless features (TEXTURE_BINDING_ARRAY + non-uniform indexing) back it.
@group(0) @binding(6) var displacement_textures: binding_array<texture_2d<f32>, 256>;
@group(0) @binding(7) var displacement_sampler: sampler;
// Output object-space micro-vertices, stride 3 f32, slot = part*max_verts + v.
@group(0) @binding(8) var<storage, read_write> gen_vertices: array<f32>;

fn load_pos(i: u32) -> vec3<f32> {
    let b = i * 3u;
    return vec3<f32>(base_positions[b], base_positions[b + 1u], base_positions[b + 2u]);
}
fn load_normal(i: u32) -> vec3<f32> {
    return octahedral_decode_signed(unpack2x16snorm(base_packed[i * 4u]));
}
fn load_uv(i: u32) -> vec2<f32> {
    let b = i * 4u;
    return vec2<f32>(bitcast<f32>(base_packed[b + 2u]), bitcast<f32>(base_packed[b + 3u]));
}
@compute @workgroup_size(128)
fn gen_verts(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // Flat part index from the 2D over-covering grid (finalize: x≤65535, y=ceil(n/65535)).
    let p = wg.x + wg.y * 65535u;
    if p >= params.part_capacity {
        return;
    }
    let part = part_triangles[p];
    let c0 = configs[part.config_lookup * 2u];
    let c1 = configs[part.config_lookup * 2u + 1u];
    let first_vert = c0 >> 16u;
    let num_verts = c1 >> 16u;
    if lid >= num_verts {
        return;
    }

    let bp = table_vertices[first_vert + lid];
    let uu = f32(bp & 0xFFFFu) / 32768.0;
    let vv = f32(bp >> 16u) / 32768.0;
    // Canonical weights (w, u, v): corner0 @ (0,0), corner1 @ (1,0), corner2 @ (0,1).
    var b = vec3<f32>(1.0 - uu - vv, uu, vv);
    // Mirror config: swap `wuv.yxz` (matches the flipped template's reversed winding
    // so the net micro-triangle stays consistently wound). base verts pre-rotated.
    if part.flip == 1u {
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
    if params.has_displacement == 1u {
        height = saturate(textureSampleLevel(
            displacement_textures[part.instance_index], displacement_sampler, uv, 0.0).r);
    }
    // Brighter texel = deeper (bevy `depth_map`): recede along -normal.
    pos += nrm * (params.displacement_bias - height * params.displacement_scale);

    // Object-space micro-vertex — the PTLAS instance transform places it in the
    // floating-origin world at TLAS build, and the closest-hit re-applies that same
    // ObjectToWorld to the position-fetched vertices (so shading lands in world too).
    let o = (p * params.max_verts + lid) * 3u;
    gen_vertices[o + 0u] = pos.x;
    gen_vertices[o + 1u] = pos.y;
    gen_vertices[o + 2u] = pos.z;
}
