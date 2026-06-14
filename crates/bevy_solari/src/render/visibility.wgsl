enable wgpu_ray_query;

// Pass 1 — primary visibility.
//
// Casts the camera ray per pixel and writes the lean PT G-buffer:
//   world_position : xyz = hit world position, w = material slot id as f32
//                    (w < 0 = miss / no geometry hit)
//   world_normal   : xyz = shading normal
//   motion_vectors : xy = previous→current screen displacement (UV-space),
//                    from the unjittered clip matrices + the hit's
//                    previous-frame world position (covers camera + object
//                    motion). Consumed by DLSS / temporal reprojection.
//
// Material is intentionally NOT baked into the G-buffer (lean): only the
// material slot id is stored, to be refetched from the scene bindings at
// shade time.

#import bevy_solari::restir_bindings::{view, view_output, gbuffer_position, gbuffer_normal, motion_vectors, gbuffer_uv, view_clip_from_world, solari_view}
#import bevy_solari::scene_bindings::{trace_ray, trace_ray_traversal, set_view_cull_mask, resolve_ray_hit_full, resolve_hair_hit, is_hair_instance, material_ids, RAY_T_MAX}
#import bevy_solari::hair_shade::shade_hair_path

/// Hair strand-strand bounces for the realtime path (the pathtracer uses its own
/// budget). Few bounces — temporal/DLSS denoises the rest.
const HAIR_MAX_BOUNCES = 4u;

@compute @workgroup_size(8, 8, 1)
fn visibility(@builtin(global_invocation_id) global_id: vec3<u32>) {
    set_view_cull_mask(solari_view.cull_mask.x);
    // One thread persists this frame's UNJITTERED clip_from_world into the
    // current parity slot; next frame reads the other slot as the previous-frame
    // matrix (used for motion vectors + ReSTIR temporal reprojection).
    if all(global_id.xy == vec2u(0u)) {
        view_clip_from_world[view.frame_count & 1u] = view.unjittered_clip_from_world;
    }

    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }

    // Primary camera ray. `world_from_clip` carries the TemporalJitter offset,
    // so the sub-pixel sample position varies each frame (AA / upscaler input);
    // motion vectors below use the unjittered matrices to stay jitter-free.
    let pixel_center = vec2<f32>(global_id.xy) + 0.5;
    let pixel_uv = pixel_center / view.main_pass_viewport.zw;
    let pixel_ndc = pixel_uv * 2.0 - 1.0;
    let ray_target = view.world_from_clip * vec4(pixel_ndc.x, -pixel_ndc.y, 1.0, 1.0);
    var ray_origin = view.world_position;
    var ray_direction = normalize((ray_target.xyz / ray_target.w) - ray_origin);

    // Portal surfaces teleport the primary ray and black holes bend it: the
    // G-buffer holds the DESTINATION surface, so lighting, NEE, and temporal
    // reprojection all operate on what the pixel actually SHOWS.
    // (View-dependent specular at those pixels uses the camera direction
    // rather than the redirected ray — a known approximation; the pathtracer
    // is exact. Accretion-disk emission is dropped on the primary here — the
    // realtime view gets the lensing and the black horizon; the disk's glow
    // is pathtracer + reflections.)
    var traversal_emitted = vec3(0.0);
    var traversal_captured = 0u;
    let ray = trace_ray_traversal(&ray_origin, &ray_direction, 0.0, RAY_FLAG_NONE, &traversal_emitted, &traversal_captured);

    if traversal_captured != 0u {
        // Horizon-captured: `w = -2` — the diffuse pass paints BLACK, not sky.
        textureStore(gbuffer_position, global_id.xy, vec4(0.0, 0.0, 0.0, -2.0));
        textureStore(gbuffer_normal, global_id.xy, vec4(0.0));
        textureStore(motion_vectors, global_id.xy, vec4(0.0));
        textureStore(gbuffer_uv, global_id.xy, vec4(0.0));
        return;
    }

    if ray.kind == RAY_QUERY_INTERSECTION_NONE {
        // Miss: negative w marks "no geometry"; later passes skip it.
        textureStore(gbuffer_position, global_id.xy, vec4(0.0, 0.0, 0.0, -1.0));
        textureStore(gbuffer_normal, global_id.xy, vec4(0.0));
        textureStore(motion_vectors, global_id.xy, vec4(0.0));
        textureStore(gbuffer_uv, global_id.xy, vec4(0.0));
        return;
    }

    // Hair: shade it inline (the surface-reservoir path can't), tag the G-buffer
    // with the `w = -3` hair sentinel so the diffuse shade pass leaves it alone,
    // and write motion + the fiber tangent (as the normal) for the temporal/DLSS
    // resolve. `compose` exposes + fogs it like any pixel.
    if is_hair_instance(ray.instance_index) {
        var rng = (global_id.x + global_id.y * u32(view.main_pass_viewport.z))
            + view.frame_count * 5782582u;
        let world_position = ray_origin + ray_direction * ray.t;
        let hair = resolve_hair_hit(ray.instance_index, ray.primitive_index, world_position);
        let radiance = shade_hair_path(
            ray_origin, ray_direction,
            ray.t, ray.instance_index, ray.primitive_index,
            HAIR_MAX_BOUNCES, &rng,
        );
        textureStore(view_output, global_id.xy, vec4(radiance, 1.0));
        textureStore(gbuffer_position, global_id.xy, vec4(world_position, -3.0));
        textureStore(gbuffer_normal, global_id.xy, vec4(hair.tangent, 0.0));
        textureStore(gbuffer_uv, global_id.xy, vec4(0.0));
        let prev_parity = (view.frame_count & 1u) ^ 1u;
        let current_clip = view.unjittered_clip_from_world * vec4(world_position, 1.0);
        let previous_clip = view_clip_from_world[prev_parity] * vec4(world_position, 1.0);
        var motion = vec2(0.0);
        if current_clip.w > 0.0 && previous_clip.w > 0.0 {
            motion = (current_clip.xy / current_clip.w - previous_clip.xy / previous_clip.w)
                * vec2(0.5, -0.5);
        }
        textureStore(motion_vectors, global_id.xy, vec4(motion, 0.0, 0.0));
        return;
    }

    let hit = resolve_ray_hit_full(ray);
    let material_id = material_ids[ray.instance_index];

    textureStore(
        gbuffer_position,
        global_id.xy,
        vec4(hit.world_position, f32(material_id)),
    );
    textureStore(gbuffer_normal, global_id.xy, vec4(hit.world_normal, 0.0));
    textureStore(gbuffer_uv, global_id.xy, vec4(hit.uv, 0.0, 0.0));

    // Motion vector: project this hit with the current unjittered clip and its
    // previous-frame world position with last frame's clip, then take the
    // UV-space delta (bevy convention: subtracted from current UV gives the
    // previous UV; *0.5 maps NDC→UV, y flipped).
    let prev_parity = (view.frame_count & 1u) ^ 1u;
    let current_clip = view.unjittered_clip_from_world * vec4(hit.world_position, 1.0);
    let previous_clip = view_clip_from_world[prev_parity] * vec4(hit.previous_frame_world_position, 1.0);
    var motion = vec2(0.0);
    if current_clip.w > 0.0 && previous_clip.w > 0.0 {
        let current_ndc = current_clip.xy / current_clip.w;
        let previous_ndc = previous_clip.xy / previous_clip.w;
        motion = (current_ndc - previous_ndc) * vec2(0.5, -0.5);
    }
    textureStore(motion_vectors, global_id.xy, vec4(motion, 0.0, 0.0));
}
