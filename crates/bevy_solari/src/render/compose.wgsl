enable wgpu_ray_query;

// Pass 5 — compose.
//
// Reads the raw linear radiance written to `view_output` by the ReSTIR +
// specular passes, applies aerial perspective (running after specular GI so
// reflections fog like everything else) and camera exposure, and writes it
// back. Downstream bloom + tonemapping (bevy's post chain) take it from there.
//
// Debug visualization deliberately does NOT live here (no baked view-mode
// switch) — it belongs in a separate render-debug-style overlay that reads the
// G-buffer / reservoir buffers.

#import bevy_solari::pbr::rand_f
#import bevy_solari::atmosphere::{atmosphere_fog_extinction, atmosphere_mie_phase, atmosphere_sun_optical_depth}
#import bevy_solari::scene_bindings::{trace_ray, set_view_cull_mask, RAY_T_MIN, RAY_T_MAX}
#import bevy_solari::restir_bindings::{view, view_output, gbuffer_position, solari_view, environment_map, environment_map_sampler, atmosphere}

const AERIAL_STEPS = 8u;

@compute @workgroup_size(8, 8, 1)
fn compose(@builtin(global_invocation_id) global_id: vec3<u32>) {
    set_view_cull_mask(solari_view.cull_mask.x);
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }

    var radiance = textureLoad(view_output, global_id.xy).rgb;

    // Aerial perspective: in-scatter single-scattered sunlight + ambient
    // skylight along the camera ray through the height fog, attenuating the
    // already-shaded radiance by the fog transmittance (Koschmieder). Same
    // model as the pathtracer's march: each step's sunlight is shadowed by a
    // ray toward the sun, so buildings cast real shafts (god rays) and
    // shadowed fog stops glowing. Per-pixel jittered steps — DLSS-RR (or any
    // temporal pass downstream) integrates the noise.
    if atmosphere.aerial_enabled > 0.0 {
        let pixel_center = vec2<f32>(global_id.xy) + 0.5;
        let pixel_uv = pixel_center / view.main_pass_viewport.zw;
        let pixel_ndc = pixel_uv * 2.0 - 1.0;
        let ray_target = view.world_from_clip * vec4(pixel_ndc.x, -pixel_ndc.y, 1.0, 1.0);
        let ray_direction = normalize((ray_target.xyz / ray_target.w) - view.world_position);

        // March camera→hit, or out to the fog's visibility range for a sky
        // pixel (`gbuffer_position.w < 0` = primary miss).
        let gpos = textureLoad(gbuffer_position, global_id.xy);
        var aerial_distance = atmosphere.aerial_visibility;
        if gpos.w >= 0.0 {
            aerial_distance = min(length(gpos.xyz - view.world_position), aerial_distance);
        }

        let ds = aerial_distance / f32(AERIAL_STEPS);
        let cos_theta = dot(atmosphere.sun_direction, ray_direction);
        let sun_phase = atmosphere_mie_phase(atmosphere.aerial_phase_g, cos_theta);
        // Sunlight reaching the fog, attenuated by the atmosphere toward the
        // sun (Beer-Lambert over the same optical depth the sky bake uses).
        // Evaluated once at the camera — the fog is a thin near-ground layer.
        let planet_camera = vec3(0.0, atmosphere.bottom_radius + atmosphere.camera_altitude, 0.0);
        let sun_od = atmosphere_sun_optical_depth(atmosphere, planet_camera);
        let sun_radiance = atmosphere.sun_illuminance * exp(-(
            atmosphere.rayleigh_scattering * sun_od.x + vec3(atmosphere.mie_extinction) * sun_od.y));
        let sky_ambient = textureSampleLevel(
            environment_map, environment_map_sampler, vec3(0.0, 1.0, 0.0), 0.0,
        ).rgb * solari_view.environment_brightness;

        var rng = (global_id.x + global_id.y * u32(view.main_pass_viewport.z))
            + view.frame_count * 5782582u;
        let jitter = rand_f(&rng);
        var fog_transmittance = 1.0;
        var inscatter = vec3(0.0);
        for (var i = 0u; i < AERIAL_STEPS; i = i + 1u) {
            if fog_transmittance < 0.003 { break; } // fog is opaque — nothing more shows through
            let p = view.world_position + ray_direction * ((f32(i) + jitter) * ds);
            let sigma = atmosphere_fog_extinction(atmosphere, p.y);
            if sigma < 1e-7 { continue; } // above the fog layer — no scattering, skip the shadow ray
            let sun_ray = trace_ray(p, atmosphere.sun_direction, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
            let sun_vis = f32(sun_ray.kind == RAY_QUERY_INTERSECTION_NONE);
            let in_scatter = sun_radiance * (sun_phase * sun_vis) + sky_ambient;
            inscatter += fog_transmittance * sigma * in_scatter * ds;
            fog_transmittance *= exp(-sigma * ds);
        }
        radiance = radiance * fog_transmittance + inscatter;
    }

    textureStore(view_output, global_id.xy, vec4(radiance * view.exposure, 1.0));
}
