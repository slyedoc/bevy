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
#import bevy_solari::scene_bindings::{trace_ray, set_view_cull_mask, fog_volumes_sample, fog_volumes_range, RAY_T_MIN, RAY_T_MAX}
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
    // skylight along the camera ray through the medium (the global height fog
    // plus any local fog volumes), attenuating the already-shaded radiance by
    // the medium transmittance (Koschmieder). Same model as the pathtracer's
    // march: each step's sunlight is shadowed by a ray toward the sun, so
    // buildings cast real shafts (god rays) and shadowed fog stops glowing.
    // Per-pixel jittered steps — DLSS-RR (or any temporal pass downstream)
    // integrates the noise.
    let pixel_center = vec2<f32>(global_id.xy) + 0.5;
    let pixel_uv = pixel_center / view.main_pass_viewport.zw;
    let pixel_ndc = pixel_uv * 2.0 - 1.0;
    let ray_target = view.world_from_clip * vec4(pixel_ndc.x, -pixel_ndc.y, 1.0, 1.0);
    let ray_direction = normalize((ray_target.xyz / ray_target.w) - view.world_position);

    // March camera→hit, or out for a sky pixel (`gbuffer_position.w < 0` =
    // primary miss).
    let gpos = textureLoad(gbuffer_position, global_id.xy);
    var hit_t = RAY_T_MAX;
    if gpos.w >= 0.0 {
        hit_t = length(gpos.xyz - view.world_position);
    }

    // March span: the global height fog covers out to its visibility range
    // (past which it's opaque anyway); fog volumes extend it — or, with the
    // global fog off, define it — so a volume-only view marches just the
    // occupied segment and medium-free pixels skip the march entirely.
    var t_start = 0.0;
    var t_end = 0.0;
    if atmosphere.aerial_enabled > 0.0 {
        t_end = min(hit_t, atmosphere.aerial_visibility);
    }
    let vol_span = fog_volumes_range(view.world_position, ray_direction, hit_t);
    if vol_span.y > vol_span.x {
        if t_end > 0.0 {
            t_end = max(t_end, vol_span.y);
        } else {
            t_start = vol_span.x;
            t_end = vol_span.y;
        }
    }

    if t_end > t_start {
        // Strata: when the volumes occupy a small slice of a long global span,
        // a uniform march undersamples them (a few-metre mist pool gets <1 of
        // the steps over a hundreds-of-metres sky ray) — low-frequency noise
        // until the temporal pass settles. Bracket the march at the volume
        // span and give that segment half the steps.
        var seg_bounds: array<vec2<f32>, 3>;
        var seg_steps: array<u32, 3>;
        var seg_count = 0u;
        if atmosphere.aerial_enabled > 0.0 && vol_span.y > vol_span.x {
            let a = clamp(vol_span.x, t_start, t_end);
            let b = clamp(vol_span.y, t_start, t_end);
            if a > t_start {
                seg_bounds[seg_count] = vec2(t_start, a);
                seg_steps[seg_count] = AERIAL_STEPS / 4u;
                seg_count += 1u;
            }
            seg_bounds[seg_count] = vec2(a, b);
            seg_steps[seg_count] = AERIAL_STEPS / 2u;
            seg_count += 1u;
            if b < t_end {
                seg_bounds[seg_count] = vec2(b, t_end);
                seg_steps[seg_count] = AERIAL_STEPS / 4u;
                seg_count += 1u;
            }
        } else {
            seg_bounds[0] = vec2(t_start, t_end);
            seg_steps[0] = AERIAL_STEPS;
            seg_count = 1u;
        }

        let cos_theta = dot(atmosphere.sun_direction, ray_direction);
        let sun_phase = atmosphere_mie_phase(atmosphere.aerial_phase_g, cos_theta);
        // Sunlight reaching the fog, attenuated by the atmosphere toward the
        // sun (Beer-Lambert over the same optical depth the sky bake uses).
        // Evaluated once at the camera — the fog is a thin near-ground layer.
        // Zero without an atmosphere view: fog volumes are then ambient-lit.
        let planet_camera = vec3(0.0, atmosphere.bottom_radius + atmosphere.camera_altitude, 0.0);
        let sun_od = atmosphere_sun_optical_depth(atmosphere, planet_camera);
        let sun_radiance = atmosphere.sun_illuminance * exp(-(
            atmosphere.rayleigh_scattering * sun_od.x + vec3(atmosphere.mie_extinction) * sun_od.y));
        let sun_lit = any(sun_radiance > vec3(0.0));
        let sky_ambient = textureSampleLevel(
            environment_map, environment_map_sampler, vec3(0.0, 1.0, 0.0), 0.0,
        ).rgb * solari_view.environment_brightness;

        var rng = (global_id.x + global_id.y * u32(view.main_pass_viewport.z))
            + view.frame_count * 5782582u;
        var fog_transmittance = 1.0;
        var inscatter = vec3(0.0);
        for (var si = 0u; si < seg_count; si += 1u) {
            let ds = (seg_bounds[si].y - seg_bounds[si].x) / f32(seg_steps[si]);
            for (var i = 0u; i < seg_steps[si]; i = i + 1u) {
                if fog_transmittance < 0.003 { break; } // fog is opaque — nothing more shows through
                // Independent per-step jitter: a shared offset makes the whole
                // march's error coherent per pixel (blotches); independent
                // strata read as fine grain and settle faster.
                let p = view.world_position + ray_direction
                    * (seg_bounds[si].x + (f32(i) + rand_f(&rng)) * ds);
                var s = fog_volumes_sample(p, cos_theta);
                if atmosphere.aerial_enabled > 0.0 {
                    let sigma = atmosphere_fog_extinction(atmosphere, p.y);
                    s.sigma_t += sigma;
                    s.sigma_s += vec3(sigma);
                    s.sun_scatter += vec3(sigma * sun_phase);
                }
                if s.sigma_t < 1e-7 { continue; } // empty step — no scattering, skip the shadow ray
                var sun_vis = 0.0;
                if sun_lit {
                    let sun_ray = trace_ray(p, atmosphere.sun_direction, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
                    sun_vis = f32(sun_ray.kind == RAY_QUERY_INTERSECTION_NONE);
                }
                let in_scatter = sun_radiance * (sun_vis * s.sun_scatter) + sky_ambient * s.sigma_s;
                inscatter += fog_transmittance * in_scatter * ds;
                fog_transmittance *= exp(-s.sigma_t * ds);
            }
        }
        radiance = radiance * fog_transmittance + inscatter;
    }

    textureStore(view_output, global_id.xy, vec4(radiance * view.exposure, 1.0));
}
