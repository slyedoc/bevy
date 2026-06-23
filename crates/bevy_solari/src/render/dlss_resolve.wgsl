// DLSS Ray Reconstruction guide resolve: unpacks the rt_pipeline's per-pixel guide
// STORAGE BUFFERS (written chit-direct by the trace) into the TEXTURES the
// `dlss_wgpu` ray-reconstruction API consumes. A plain wgpu compute pass — the
// trace's trailing SHADER_WRITE→SHADER_READ barrier makes the buffer writes visible
// here, and the textures stay fully wgpu-layout-tracked.
//
// Packing produced by the trace (each `vec4<f32>` per pixel, row-major y*width+x):
//   normal_roughness : world normal.xyz + linear roughness (.w)   [Packed]
//   diffuse          : diffuse albedo.xyz + linear view depth (.w)
//   specular         : specular albedo (F0).xyz + spec hit dist (.w)
//   motion           : screen-space motion.xy

@group(0) @binding(0) var<storage, read> nr_buf: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> diffuse_buf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> specular_buf: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> motion_buf: array<vec4<f32>>;
@group(0) @binding(4) var out_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var out_normal_roughness: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var out_diffuse: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var out_specular: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(8) var out_motion: texture_storage_2d<rg16float, write>;
@group(0) @binding(9) var out_spec_hit_distance: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_depth);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let p = vec2<i32>(gid.xy);
    let idx = gid.y * dims.x + gid.x;

    let nr = nr_buf[idx];
    let diffuse = diffuse_buf[idx];
    let specular = specular_buf[idx];
    let motion = motion_buf[idx];

    textureStore(out_normal_roughness, p, nr);
    textureStore(out_diffuse, p, vec4<f32>(diffuse.xyz, 1.0));
    textureStore(out_depth, p, vec4<f32>(diffuse.w, 0.0, 0.0, 0.0));
    textureStore(out_specular, p, vec4<f32>(specular.xyz, 1.0));
    textureStore(out_spec_hit_distance, p, vec4<f32>(specular.w, 0.0, 0.0, 0.0));
    textureStore(out_motion, p, vec4<f32>(motion.xy, 0.0, 0.0));
}
