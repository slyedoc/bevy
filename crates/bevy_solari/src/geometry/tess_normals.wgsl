// Smooth-normal pass for the displacement tessellation showcase.
//
// One invocation per micro-triangle of one cluster's base triangles: it computes,
// for each of the micro-triangle's three micro-vertices, a SMOOTH, displacement-aware
// surface normal and writes it (packed octahedral) into the per-instance normal
// buffer the closest-hit reads. The normal is the interpolated base-surface normal
// perturbed by the displacement height-map gradient (standard bump formula) — it uses
// the base mesh's shared, crack-free normal/tangent so it stays continuous across base
// triangle edges (no seams), unlike a per-micro-facet geometric normal.
//
// DENORMALIZED layout: 3 normals per micro-triangle, contiguous by global primitive
// index (`(out_clas_base + base_tri*split + sub) * micro_triangle_count + micro_tri`),
// so the closest-hit indexes them by `primitive_base + triangle_id` with no template
// index buffer of its own. Packed octahedral u32 (matches the vertex pool / the chit's
// `octahedral_decode_signed(unpack2x16snorm(...))`).

#import bevy_render::utils::{octahedral_decode_signed, octahedral_encode}

struct TessNormalsParams {
    // Base mesh pool offsets for the cluster being processed (global indices).
    vertex_offset: u32,
    index_offset: u32,
    // Base triangles in this cluster dispatch.
    base_triangle_count: u32,
    // Micro-triangles per CLAS (= level²); also the per-CLAS primitive stride.
    micro_triangle_count: u32,
    // Bump strength: the gradient term's scale (folds displacement height + UV density).
    normal_strength: f32,
    has_displacement: u32,
    // Running CLAS index for this cluster (= prior base-triangles × split_count).
    out_clas_base: u32,
    // Recursive split factor K² (sub-triangles per base triangle).
    split_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: TessNormalsParams;
@group(0) @binding(1) var<storage, read> cluster_indices: array<u32>;
// Base vertex packed attrs, 4 u32 / vertex (stride 16): normal[0], tangent[1], uv[2..3].
@group(0) @binding(2) var<storage, read> vertex_packed: array<u32>;
// Per-level micro-vertex barycentrics, flat f32 (stride 12).
@group(0) @binding(3) var<storage, read> tess_barycentrics: array<f32>;
// Template topology: per micro-triangle, 3 local micro-vertex indices.
@group(0) @binding(4) var<storage, read> micro_indices: array<u32>;
// Per sub-triangle, its 3 corner barycentrics within the base triangle (9 floats).
@group(0) @binding(5) var<storage, read> split_corner_barys: array<f32>;
@group(0) @binding(6) var displacement_texture: texture_2d<f32>;
@group(0) @binding(7) var displacement_sampler: sampler;
// Denormalized output: 3 packed-octahedral normals per micro-triangle.
@group(0) @binding(8) var<storage, read_write> normals_out: array<u32>;

fn load_normal(global_vertex: u32) -> vec3<f32> {
    return octahedral_decode_signed(unpack2x16snorm(vertex_packed[global_vertex * 4u]));
}

// 10-bit packed tangent (xyz unorm of [-1,1] + bitangent sign in bit 30) — matches
// `scene_bindings::unpack_tangent`.
fn load_tangent(global_vertex: u32) -> vec4<f32> {
    let p = vertex_packed[global_vertex * 4u + 1u];
    let x = f32(p & 0x3ffu) / 1023.0 * 2.0 - 1.0;
    let y = f32((p >> 10u) & 0x3ffu) / 1023.0 * 2.0 - 1.0;
    let z = f32((p >> 20u) & 0x3ffu) / 1023.0 * 2.0 - 1.0;
    let w = select(1.0, -1.0, ((p >> 30u) & 1u) == 1u);
    return vec4<f32>(normalize(vec3<f32>(x, y, z)), w);
}

fn load_uv(global_vertex: u32) -> vec2<f32> {
    let base = global_vertex * 4u;
    return vec2<f32>(bitcast<f32>(vertex_packed[base + 2u]), bitcast<f32>(vertex_packed[base + 3u]));
}

fn pack_normal(n: vec3<f32>) -> u32 {
    // Signed octahedral (the form `octahedral_decode_signed` inverts) → snorm u32.
    return pack2x16snorm(octahedral_encode(n) * 2.0 - 1.0);
}

@compute @workgroup_size(64)
fn compute_normals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread = gid.x;
    let subs = params.split_count;
    let tris = params.micro_triangle_count;
    let total = params.base_triangle_count * subs * tris;
    if thread >= total {
        return;
    }
    let base_tri = thread / (subs * tris);
    let rem = thread % (subs * tris);
    let sub = rem / tris;
    let micro_tri = rem % tris;

    let idx_base = params.index_offset + base_tri * 3u;
    let i0 = params.vertex_offset + cluster_indices[idx_base + 0u];
    let i1 = params.vertex_offset + cluster_indices[idx_base + 1u];
    let i2 = params.vertex_offset + cluster_indices[idx_base + 2u];
    let n0 = load_normal(i0);
    let n1 = load_normal(i1);
    let n2 = load_normal(i2);
    let tg0 = load_tangent(i0);
    let tg1 = load_tangent(i1);
    let tg2 = load_tangent(i2);
    let uv0 = load_uv(i0);
    let uv1 = load_uv(i1);
    let uv2 = load_uv(i2);

    // This sub-triangle's 3 corner barycentrics (within the base triangle).
    let sc = sub * 9u;
    let sb0 = vec3<f32>(split_corner_barys[sc + 0u], split_corner_barys[sc + 1u], split_corner_barys[sc + 2u]);
    let sb1 = vec3<f32>(split_corner_barys[sc + 3u], split_corner_barys[sc + 4u], split_corner_barys[sc + 5u]);
    let sb2 = vec3<f32>(split_corner_barys[sc + 6u], split_corner_barys[sc + 7u], split_corner_barys[sc + 8u]);

    let clas = params.out_clas_base + base_tri * subs + sub;
    let global_prim = clas * tris + micro_tri;

    var dims = vec2<f32>(1.0, 1.0);
    if params.has_displacement == 1u {
        dims = vec2<f32>(textureDimensions(displacement_texture, 0));
    }
    let texel = 1.0 / dims;
    // Bandlimit the height gradient to THIS micro-triangle's UV footprint instead of a
    // single texel: a per-texel gradient injects sub-geometry detail into the normal
    // that the geometry can't represent, and it aliases into DLSS flicker under jitter
    // (diffuse + specular both shimmer). Sampling one texel apart at the matching mip
    // (≈ the micro-triangle's size) keeps the normal at the geometry's own frequency.
    let subdiv = sqrt(f32(tris) * f32(subs));
    let micro_uv = max(length(uv1 - uv0), length(uv2 - uv0)) / max(subdiv, 1.0);
    let grad_eps = max(micro_uv, max(texel.x, texel.y));
    let grad_mip = max(0.0, log2(grad_eps * max(dims.x, dims.y)));

    for (var k = 0u; k < 3u; k = k + 1u) {
        let lv = micro_indices[micro_tri * 3u + k];
        let mb_i = lv * 3u;
        let mb = vec3<f32>(tess_barycentrics[mb_i], tess_barycentrics[mb_i + 1u], tess_barycentrics[mb_i + 2u]);
        // Compose into the base triangle's barycentric space (nested subdivision).
        let bary = sb0 * mb.x + sb1 * mb.y + sb2 * mb.z;

        let nrm = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
        let tan = normalize(bary.x * tg0.xyz + bary.y * tg1.xyz + bary.z * tg2.xyz);
        let sign_w = select(1.0, -1.0, (bary.x * tg0.w + bary.y * tg1.w + bary.z * tg2.w) < 0.0);
        let bitan = cross(nrm, tan) * sign_w;
        let uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

        var disp_n = nrm;
        if params.has_displacement == 1u {
            // Central-difference height gradient at the GEOMETRY scale (`grad_eps` /
            // `grad_mip`), not per texel — bandlimited so the normal doesn't alias.
            let hl = textureSampleLevel(displacement_texture, displacement_sampler, uv - vec2<f32>(grad_eps, 0.0), grad_mip).r;
            let hr = textureSampleLevel(displacement_texture, displacement_sampler, uv + vec2<f32>(grad_eps, 0.0), grad_mip).r;
            let hd = textureSampleLevel(displacement_texture, displacement_sampler, uv - vec2<f32>(0.0, grad_eps), grad_mip).r;
            let hu = textureSampleLevel(displacement_texture, displacement_sampler, uv + vec2<f32>(0.0, grad_eps), grad_mip).r;
            let dhdu = (hr - hl) * 0.5;
            let dhdv = (hu - hd) * 0.5;
            // `+` (not `-`): the surface recedes along -normal (bevy depth convention),
            // so the slope tilts opposite the height-up case.
            disp_n = normalize(nrm + params.normal_strength * (dhdu * tan + dhdv * bitan));
        }
        // Per micro-triangle vertex: packed normal + base UV (3 u32, 12 B), so the
        // closest-hit interpolates a real UV and the base-color texture comes back.
        let o = (global_prim * 3u + k) * 3u;
        normals_out[o + 0u] = pack_normal(disp_n);
        normals_out[o + 1u] = bitcast<u32>(uv.x);
        normals_out[o + 2u] = bitcast<u32>(uv.y);
    }
}
