// Phase 2c — the displacement compute pass.
//
// One invocation per generated micro-vertex of one cluster's base triangles:
// it barycentrically interpolates the base triangle's position / normal / UV,
// samples the displacement height map at the interpolated UV, pushes the
// position along the interpolated (crack-free, shared-vertex) normal by
// `height*scale + bias`, and writes the displaced position into the buffer
// that `INSTANTIATE_TRIANGLE_CLUSTER` reads (`out_positions`). Clean-room WGSL
// analog of the reference's `triangle_tess_template_instantiate.comp.glsl`
// height-map branch (347-371).
//
// Deliberately self-contained: it does NOT reuse the raytracing scene bind
// group (that path is raw-VK descriptor sets + the fork's `physical_load`
// intrinsic, neither of which a plain wgpu compute pipeline can bind). Instead
// it binds the cluster pools as ordinary wgpu storage and the single
// displacement texture directly. Per-cluster dispatch, so the cluster's
// `vertex_offset` / `index_offset` ride in the params UBO rather than a
// `clusters[]` binding.
//
// STRIDE TRAP: `array<vec3<f32>>` storage has a 16-byte stride, but the vertex
// position pool and the INSTANTIATE vertex buffer are tightly packed at stride
// 12. So positions in/out and barycentrics are `array<f32>` indexed `*3`, never
// `array<vec3<f32>>`. The 16-byte packed attribute pool is read as `array<u32>`
// (4 u32 / vertex): normal at [0] (octahedral), UV at [2..3] (raw f32 bits).

#import bevy_render::utils::octahedral_decode_signed

struct TessDisplaceParams {
    // Base mesh pool offsets for the cluster being displaced (global indices),
    // matching the chit's rebasing (`vertex_offset + cluster_indices[...]`).
    vertex_offset: u32,
    index_offset: u32,
    // Base triangles in this cluster; each expands to `split_count` sub-triangles,
    // each of `micro_vertex_count` verts.
    base_triangle_count: u32,
    // Micro-vertices per sub-triangle = (level+1)(level+2)/2.
    micro_vertex_count: u32,
    displacement_scale: f32,
    displacement_bias: f32,
    // First output slot (micro-vertex units) for this cluster's region.
    out_vertex_base: u32,
    // 1 if a real displacement texture is bound; 0 leaves height at 0.
    has_displacement: u32,
    // RECURSIVE SPLIT: each base triangle is split into `split_count` (= K²)
    // sub-triangles (each its own CLAS), so effective resolution is (K·level)²
    // beyond the per-CLAS budget. 1 = no split. `split_corner_barys` holds each
    // sub-triangle's 3 corner barycentrics (within the base triangle).
    split_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: TessDisplaceParams;
// Base mesh cluster-local triangle indices (rebased by `vertex_offset`).
@group(0) @binding(1) var<storage, read> cluster_indices: array<u32>;
// Base vertex positions, flat f32 (stride 12).
@group(0) @binding(2) var<storage, read> vertex_positions: array<f32>;
// Base vertex packed attrs, 4 u32 / vertex (stride 16): normal[0], uv[2..3].
@group(0) @binding(3) var<storage, read> vertex_packed: array<u32>;
// Per-level micro-vertex barycentrics, flat f32 (stride 12); matches
// `SubdividedTriangle::barycentrics`.
@group(0) @binding(4) var<storage, read> tess_barycentrics: array<f32>;
// Displaced micro-vertex positions for INSTANTIATE, flat f32 (stride 12).
@group(0) @binding(5) var<storage, read_write> out_positions: array<f32>;
@group(0) @binding(6) var displacement_texture: texture_2d<f32>;
@group(0) @binding(7) var displacement_sampler: sampler;
// Recursive split: per sub-triangle, its 3 corner barycentrics (within the base
// triangle), flat f32 — 9 per sub-triangle, `split_count` sub-triangles. For
// `split_count == 1` the lone sub-triangle's corners are the base corners
// (identity composition ⇒ no split).
@group(0) @binding(8) var<storage, read> split_corner_barys: array<f32>;

fn load_position(global_vertex: u32) -> vec3<f32> {
    let b = global_vertex * 3u;
    return vec3<f32>(vertex_positions[b], vertex_positions[b + 1u], vertex_positions[b + 2u]);
}

fn load_normal(global_vertex: u32) -> vec3<f32> {
    return octahedral_decode_signed(unpack2x16snorm(vertex_packed[global_vertex * 4u]));
}

fn load_uv(global_vertex: u32) -> vec2<f32> {
    let base = global_vertex * 4u;
    return vec2<f32>(bitcast<f32>(vertex_packed[base + 2u]), bitcast<f32>(vertex_packed[base + 3u]));
}

@compute @workgroup_size(64)
fn displace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread = gid.x;
    let subs = params.split_count;
    let per_sub = params.micro_vertex_count;
    // Thread layout: (base_tri, sub ∈ 0..split_count, micro ∈ 0..per_sub).
    let total = params.base_triangle_count * subs * per_sub;
    if thread >= total {
        return;
    }
    let base_tri = thread / (subs * per_sub);
    let rem = thread % (subs * per_sub);
    let sub = rem / per_sub;
    let micro = rem % per_sub;

    let idx_base = params.index_offset + base_tri * 3u;
    let i0 = params.vertex_offset + cluster_indices[idx_base + 0u];
    let i1 = params.vertex_offset + cluster_indices[idx_base + 1u];
    let i2 = params.vertex_offset + cluster_indices[idx_base + 2u];

    let p0 = load_position(i0);
    let p1 = load_position(i1);
    let p2 = load_position(i2);
    let n0 = load_normal(i0);
    let n1 = load_normal(i1);
    let n2 = load_normal(i2);
    let t0 = load_uv(i0);
    let t1 = load_uv(i1);
    let t2 = load_uv(i2);

    // This sub-triangle's 3 corner barycentrics (within the base triangle).
    let sc = sub * 9u;
    let sb0 = vec3<f32>(split_corner_barys[sc + 0u], split_corner_barys[sc + 1u], split_corner_barys[sc + 2u]);
    let sb1 = vec3<f32>(split_corner_barys[sc + 3u], split_corner_barys[sc + 4u], split_corner_barys[sc + 5u]);
    let sb2 = vec3<f32>(split_corner_barys[sc + 6u], split_corner_barys[sc + 7u], split_corner_barys[sc + 8u]);

    // Micro-vertex barycentric within the sub-triangle, composed into the base
    // triangle's barycentric space (nested subdivision).
    let mb_i = micro * 3u;
    let mb = vec3<f32>(tess_barycentrics[mb_i], tess_barycentrics[mb_i + 1u], tess_barycentrics[mb_i + 2u]);
    let bary = sb0 * mb.x + sb1 * mb.y + sb2 * mb.z;

    var pos = bary.x * p0 + bary.y * p1 + bary.z * p2;
    let nrm = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
    let uv = bary.x * t0 + bary.y * t1 + bary.z * t2;

    var height = 0.0;
    if params.has_displacement == 1u {
        // Clamp to [0,1]: the displaced offset stays within ±scale, so the placed
        // world AABB (mesh AABB ± scale) is an exact bound — a height-map texel > 1
        // would otherwise push geometry outside the AABB and fault the traversal.
        height = saturate(textureSampleLevel(displacement_texture, displacement_sampler, uv, 0.0).r);
    }
    // Match bevy's `depth_map` convention: a brighter texel is DEEPER, so the surface
    // recedes ALONG -normal (white = recessed). The displaced offset stays within
    // [-scale, 0], covered by the +/-scale world AABB.
    pos += nrm * (params.displacement_bias - height * params.displacement_scale);

    // One CLAS per (base_tri, sub); its region is contiguous by global sub index.
    let o = (params.out_vertex_base + (base_tri * subs + sub) * per_sub + micro) * 3u;
    out_positions[o + 0u] = pos.x;
    out_positions[o + 1u] = pos.y;
    out_positions[o + 2u] = pos.z;
}
