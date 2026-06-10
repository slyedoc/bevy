// DLSS Ray Reconstruction guide-buffer resolve.
//
// Reads the restir G-buffer and writes the guide buffers DLSS-RR consumes:
//   depth            : linear view-space depth (R32Float) — `DepthMode::Linear`
//   normal_roughness : world normal (xyz) + perceptual roughness (w) — `Packed`
//   diffuse_albedo   : demodulated diffuse color
//   specular_albedo  : F0 (specular reflectance at normal incidence)
//   specular_motion  : reuse the regular motion vectors for now
//
// Self-contained group(1) (the G-buffer subset + outputs) so it doesn't depend
// on the main restir bind group. Scene materials come from group(0).

#import bevy_render::view::View
#import bevy_solari::pbr::{calculate_diffuse_color, calculate_F0}
#import bevy_solari::scene_bindings::{resolve_material, materials}

@group(1) @binding(0) var gbuffer_position: texture_storage_2d<rgba32float, read>;
@group(1) @binding(1) var gbuffer_normal: texture_storage_2d<rgba16float, read>;
@group(1) @binding(2) var gbuffer_uv: texture_storage_2d<rgba32float, read>;
@group(1) @binding(3) var motion_vectors: texture_storage_2d<rgba16float, read>;
@group(1) @binding(4) var<uniform> view: View;
@group(1) @binding(5) var out_depth: texture_storage_2d<r32float, write>;
@group(1) @binding(6) var out_normal_roughness: texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var out_diffuse_albedo: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(8) var out_specular_albedo: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(9) var out_specular_motion: texture_storage_2d<rg16float, write>;

@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }
    let pixel = global_id.xy;

    // Specular motion guide — regular motion vectors for now (proper specular
    // / PSR motion is a later refinement).
    let motion = textureLoad(motion_vectors, pixel).xy;
    textureStore(out_specular_motion, pixel, vec4(motion, 0.0, 0.0));

    let gpos = textureLoad(gbuffer_position, pixel);
    if gpos.w < 0.0 {
        // Sky / miss: push depth far, clear material guides.
        textureStore(out_depth, pixel, vec4(1.0e9, 0.0, 0.0, 0.0));
        textureStore(out_normal_roughness, pixel, vec4(0.0, 0.0, 1.0, 1.0));
        textureStore(out_diffuse_albedo, pixel, vec4(0.0));
        textureStore(out_specular_albedo, pixel, vec4(0.0));
        return;
    }

    let normal = textureLoad(gbuffer_normal, pixel).xyz;
    let uv = textureLoad(gbuffer_uv, pixel).xy;
    let material = resolve_material(materials[u32(gpos.w)], uv);

    // Linear view-space depth (positive distance along the view forward axis).
    let view_position = view.view_from_world * vec4(gpos.xyz, 1.0);
    textureStore(out_depth, pixel, vec4(-view_position.z, 0.0, 0.0, 0.0));

    textureStore(out_normal_roughness, pixel, vec4(normal, material.perceptual_roughness));
    textureStore(out_diffuse_albedo, pixel, vec4(calculate_diffuse_color(material.base_color, material.metallic, 0.0, 0.0), 1.0));
    textureStore(out_specular_albedo, pixel, vec4(calculate_F0(material.base_color, material.metallic, vec3(material.reflectance)), 1.0));
}
