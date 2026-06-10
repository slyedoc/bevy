// DLSS Ray Reconstruction guide-buffer resolve.
//
// Reads the restir G-buffer and writes the guide buffers DLSS-RR consumes:
//   depth            : linear view-space depth (R32Float) — `DepthMode::Linear`
//   normal_roughness : world normal (xyz) + perceptual roughness (w) — `Packed`
//   diffuse_albedo   : demodulated diffuse color
//   specular_albedo  : F0 (specular reflectance at normal incidence)
//   specular_motion  : screen-space motion of the VIRTUAL reflected image
//
// Self-contained group(1) (the G-buffer subset + outputs) so it doesn't depend
// on the main restir bind group. Scene materials come from group(0).

#import bevy_render::view::View
#import bevy_solari::pbr::{calculate_diffuse_color, calculate_F0}
#import bevy_solari::scene_bindings::{resolve_material, materials, RAY_T_MAX}

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
// Specular reflection first-hit distance (specular-GI pass; RAY_T_MAX = env miss).
@group(1) @binding(10) var specular_hit_distance: texture_storage_2d<r32float, read>;
// Current/previous unjittered clip_from_world, frame-parity ping-ponged by the
// visibility pass (slot [frame&1] = current).
@group(1) @binding(11) var<storage, read> view_clip_from_world: array<mat4x4<f32>, 2>;

@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }
    let pixel = global_id.xy;

    let gpos = textureLoad(gbuffer_position, pixel);
    if gpos.w < 0.0 {
        // Sky / miss: push depth far, clear material guides; the reflected
        // image IS the environment, so its motion is the surface motion (the
        // visibility pass wrote camera-rotation reprojection for the sky).
        textureStore(out_depth, pixel, vec4(1.0e9, 0.0, 0.0, 0.0));
        textureStore(out_normal_roughness, pixel, vec4(0.0, 0.0, 1.0, 1.0));
        textureStore(out_diffuse_albedo, pixel, vec4(0.0));
        textureStore(out_specular_albedo, pixel, vec4(0.0));
        let motion = textureLoad(motion_vectors, pixel).xy;
        textureStore(out_specular_motion, pixel, vec4(motion, 0.0, 0.0));
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

    // Specular motion: where did the REFLECTED image move on screen? A mirror
    // image of a point hit at distance `t` behind the reflector appears at
    // depth `primary_t + t` along the camera ray (the virtual image), so
    // reproject that virtual world point with the previous frame's clip
    // matrix. Static-world assumption — this captures camera motion over
    // static reflectors, the case surface motion gets badly wrong (a mirror's
    // reflection moves opposite/faster than the mirror surface itself).
    // Zero hit distance = no reflection data this pixel → surface motion.
    let spec_t = textureLoad(specular_hit_distance, pixel).x;
    var spec_motion = textureLoad(motion_vectors, pixel).xy;
    if spec_t > 0.0 {
        let camera_to_hit = gpos.xyz - view.world_position;
        let primary_t = length(camera_to_hit);
        let primary_dir = camera_to_hit / primary_t;
        // Clamp the env-miss sentinel so the virtual point stays finite; at
        // this range the reprojection is rotation-dominated anyway.
        let virtual_point = view.world_position
            + primary_dir * (primary_t + min(spec_t, RAY_T_MAX));

        let parity = view.frame_count & 1u;
        let current_clip = view_clip_from_world[parity] * vec4(virtual_point, 1.0);
        let previous_clip = view_clip_from_world[parity ^ 1u] * vec4(virtual_point, 1.0);
        if current_clip.w > 0.0 && previous_clip.w > 0.0 {
            let current_ndc = current_clip.xy / current_clip.w;
            let previous_ndc = previous_clip.xy / previous_clip.w;
            spec_motion = (current_ndc - previous_ndc) * vec2(0.5, -0.5);
        }
    }
    textureStore(out_specular_motion, pixel, vec4(spec_motion, 0.0, 0.0));
}
