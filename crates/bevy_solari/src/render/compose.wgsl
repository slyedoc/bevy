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

#import bevy_solari::pbr::{rand_f, rand_u, rand_range_u}
#import bevy_solari::atmosphere::{atmosphere_fog_extinction, atmosphere_mie_phase, atmosphere_sun_optical_depth}
#import bevy_solari::scene_bindings::{trace_ray, set_view_cull_mask, fog_volumes_sample, fog_volumes_range, light_sources, RAY_T_MIN, RAY_T_MAX}
#import bevy_solari::sampling::{resolve_light_sample, calculate_resolved_light_contribution, trace_light_visibility, emissive_light_count, LightSample, ResolvedLightSample, NULL_LIGHT_ID}
#import bevy_solari::restir_bindings::{view, view_output, gbuffer_position, solari_view, environment_map, environment_map_sampler, atmosphere, light_tiles, unpack_light_tile_sample, regir_query, regir_samples, caustic_gather_volume, REGIR_CELL_NONE, REGIR_ENTRIES_PER_CELL}

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
        var total_steps = 0u;
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
        for (var si = 0u; si < seg_count; si += 1u) {
            total_steps += seg_steps[si];
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
        // Local-light sampling runs at ONE randomly chosen step per pixel,
        // weighted by the step count (unbiased) — per-step sampling multiplies
        // the march's whole cost (query + resolve + shadow ray) by the step
        // count, which is a fps cliff under city-wide global fog. The temporal
        // pass owns the variance, same contract as the march jitter.
        let light_step = rand_range_u(total_steps, &rng);
        var step_index = 0u;
        var fog_transmittance = 1.0;
        var inscatter = vec3(0.0);
        for (var si = 0u; si < seg_count; si += 1u) {
            let ds = (seg_bounds[si].y - seg_bounds[si].x) / f32(seg_steps[si]);
            for (var i = 0u; i < seg_steps[si]; i = i + 1u) {
                if fog_transmittance < 0.003 { break; } // fog is opaque — nothing more shows through
                let fog_step = step_index;
                step_index += 1u;
                // Independent per-step jitter: a shared offset makes the whole
                // march's error coherent per pixel (blotches); independent
                // strata read as fine grain and settle faster.
                let p = view.world_position + ray_direction
                    * (seg_bounds[si].x + (f32(i) + rand_f(&rng)) * ds);
                var s = fog_volumes_sample(p, cos_theta);
                // Local-light NEE is gated on FOG-VOLUME scattering — with
                // only the global height fog (a whole city of it), paying a
                // grid query + resolve + shadow ray per pixel buys emissive
                // glow daylight drowns out. Volumes are the opt-in.
                let volume_light_weight = s.phase_weight;
                if atmosphere.aerial_enabled > 0.0 {
                    let sigma = atmosphere_fog_extinction(atmosphere, p.y);
                    s.sigma_t += sigma;
                    s.sigma_s += vec3(sigma);
                    s.sun_scatter += vec3(sigma * sun_phase);
                    s.phase_g_sum += atmosphere.aerial_phase_g * (3.0 * sigma);
                    s.phase_weight += 3.0 * sigma;
                }
                if s.sigma_t < 1e-7 { continue; } // empty step — no scattering, skip the shadow ray
                var sun_vis = 0.0;
                if sun_lit {
                    let sun_ray = trace_ray(p, atmosphere.sun_direction, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
                    sun_vis = f32(sun_ray.kind == RAY_QUERY_INTERSECTION_NONE);
                }
                let in_scatter = sun_radiance * (sun_vis * s.sun_scatter) + sky_ambient * s.sigma_s;
                inscatter += fog_transmittance * in_scatter * ds;

                // Volumetric caustics: in-scatter the photon pass deposited
                // mid-air (post-glass beam segments crossing fog volumes) —
                // the dispersed beams glow in the dust as colored shafts.
                // σ_s and phase are folded at deposit time.
                if volume_light_weight > 1e-7 {
                    inscatter += fog_transmittance * caustic_gather_volume(p, &rng) * ds;
                }

                // Local (emissive) lights: one ReGIR-guided NEE sample for the
                // step. A live cell hands us a light its RIS already vetted
                // for this region (fresh point on it, chained weight — the
                // same consumption as the per-pixel DI candidates); a cold or
                // missing cell falls back to a uniform light-tile sample, and
                // the query itself inserts the cell so it's warm next frame.
                // The grid and tiles are emissive-only, so the sun term above
                // is never double-counted. The "normal" handed to the query is
                // the view ray — its tangent-plane jitter then decorrelates
                // the two directions the march doesn't already jitter along.
                if fog_step == light_step && volume_light_weight > 1e-7 && emissive_light_count() > 0u {
                    var resolved = ResolvedLightSample(vec4(0.0), vec3(0.0), vec3(0.0), 0.0);
                    var light_valid = false;
                    let cell = regir_query(p, ray_direction, view.world_position, &rng);
                    if cell != REGIR_CELL_NONE {
                        let entry = regir_samples[cell * REGIR_ENTRIES_PER_CELL + rand_range_u(REGIR_ENTRIES_PER_CELL, &rng)];
                        if entry.light_id != NULL_LIGHT_ID {
                            let slot = entry.light_id >> 16u;
                            let light_source = light_sources[slot];
                            let triangle_id = rand_range_u(light_source.kind >> 1u, &rng);
                            resolved = resolve_light_sample(LightSample((slot << 16u) | triangle_id, rand_u(&rng)), light_source);
                            // light weight × fresh point's area inverse-pdf
                            resolved.inverse_pdf *= entry.inverse_pdf;
                            light_valid = true;
                        }
                    } else {
                        let entry = light_tiles[rand_range_u(arrayLength(&light_tiles), &rng)];
                        if entry.light_id != NULL_LIGHT_ID {
                            resolved = unpack_light_tile_sample(entry);
                            light_valid = true;
                        }
                    }
                    if light_valid {
                        let light = calculate_resolved_light_contribution(resolved, p, ray_direction);
                        if any(light.radiance > vec3(0.0)) {
                            let g_eff = s.phase_g_sum / s.phase_weight;
                            let phase = atmosphere_mie_phase(g_eff, dot(light.wi, ray_direction));
                            let vis = trace_light_visibility(p, resolved.world_position);
                            inscatter += fog_transmittance * s.sigma_s
                                * (phase * light.inverse_pdf * vis * f32(total_steps))
                                * light.radiance * ds;
                        }
                    }
                }

                fog_transmittance *= exp(-s.sigma_t * ds);
            }
        }
        radiance = radiance * fog_transmittance + inscatter;
    }

    textureStore(view_output, global_id.xy, vec4(radiance * view.exposure, 1.0));
}
