enable wgpu_ray_query;

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_solari::pbr::{rand_f, rand_vec2f, rand_range_u}
#import bevy_render::view::View
#import bevy_solari::brdf::{evaluate_brdf, evaluate_and_sample_brdf, brdf_pdf, F_AB, bend_shading_normal, sample_glass_bsdf, dispersive_ior, spectral_rgb_weight, sample_hero_wavelength}
#import bevy_solari::sampling::{sample_random_light, random_emissive_light_pdf, power_heuristic, generate_random_emissive_light_sample, calculate_resolved_light_contribution, trace_light_visibility, trace_light_transmittance, emissive_light_count, NULL_LIGHT_ID}
#import bevy_solari::scene_bindings::{trace_ray, trace_ray_traversal, set_view_cull_mask, resolve_ray_hit_full, resolve_ray_hit_full_lod, offset_ray_origin, directional_lights, light_sources, active_light_list, fog_volumes_sample, fog_volumes_range, is_hair_instance, resolve_hair_hit, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD}
#import bevy_solari::hair::sample_hair_bsdf
#import bevy_solari::hair_shade::hair_direct_lighting
#import bevy_solari::atmosphere::{Atmosphere, atmosphere_fog_extinction, atmosphere_mie_phase, atmosphere_sun_optical_depth}

@group(1) @binding(0) var accumulation_texture: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(1) var view_output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var<uniform> view: View;
// Per-view RT cull mask (camera `RenderLayers` → low 8 bits, `.x`), the no-skybox
// primary-miss background (`clear_color`, linear RGB), and the sky brightness in
// raw cd/m² (`0.0` ⇒ no skybox).
struct SolariView {
    cull_mask: vec4<u32>,
    clear_color: vec3<f32>,
    environment_brightness: f32,
    debug_mode: u32,
    // Thin-lens DoF (`SolariLens`); aperture 0 = pinhole.
    focus_distance: f32,
    aperture_radius: f32,
}
@group(1) @binding(3) var<uniform> solari_view: SolariView;
// Environment map (sky), sampled in the ray direction on a miss. Bound to the
// view's `Skybox.image` cube (or a fallback when absent).
@group(1) @binding(4) var environment_map: texture_cube<f32>;
@group(1) @binding(5) var environment_map_sampler: sampler;
// Atmosphere params + sun for primary-ray aerial perspective (distance haze).
// Disabled (`aerial_enabled == 0`) when the view has no `SolariAtmosphere`.
@group(1) @binding(6) var<uniform> atmosphere: Atmosphere;

const MAX_BOUNCES = 64u;
// Max simultaneously-nested transmissive volumes a path tracks (air → bottle
// glass → wine is 2; 4 leaves headroom).
const MEDIUM_STACK_SIZE = 4u;
const MEDIUM_NOT_FOUND = 0xFFFFFFFFu;

@compute @workgroup_size(8, 8, 1)
fn pathtrace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    set_view_cull_mask(solari_view.cull_mask.x);
    if any(global_id.xy >= vec2u(view.viewport.zw)) {
        return;
    }

    let old_color = textureLoad(accumulation_texture, global_id.xy);

    // Setup RNG, seeded by (pixel, accumulated-sample index): sample N always
    // draws the same numbers, so a converged accumulation is bit-reproducible.
    // Deliberate side effect: while the camera moves (reset every frame, so
    // the sample index stays 0) the grain pattern holds still on screen —
    // calmer to look at than per-frame varying noise boiling during motion.
    let pixel_index = global_id.x + global_id.y * u32(view.viewport.z);
    let frame_index = u32(old_color.a) * 5782582u;
    var rng = pixel_index + frame_index;

    // Shoot the first ray from the camera
    let pixel_center = vec2<f32>(global_id.xy) + 0.5;
    let jitter = rand_vec2f(&rng) - 0.5;
    let pixel_uv = (pixel_center + jitter) / view.viewport.zw;
    let pixel_ndc = (pixel_uv * 2.0) - 1.0;
    let primary_ray_target = view.world_from_clip * vec4(pixel_ndc.x, -pixel_ndc.y, 1.0, 1.0);
    var ray_origin = view.world_position;
    var ray_direction = normalize((primary_ray_target.xyz / primary_ray_target.w) - ray_origin);
    var ray_t_min = 0.0;

    // Thin-lens depth of field (`SolariLens`): jitter the origin across the
    // aperture disk and re-aim at this ray's focal-plane point — the plane at
    // `focus_distance` along the VIEW axis stays sharp, everything off it
    // blurs, and the accumulation converges the blur into true bokeh.
    if solari_view.aperture_radius > 0.0 {
        let camera_right = view.world_from_view[0].xyz;
        let camera_up = view.world_from_view[1].xyz;
        let camera_forward = -view.world_from_view[2].xyz;
        let focus_t = solari_view.focus_distance / max(dot(ray_direction, camera_forward), 1e-4);
        let focus_point = ray_origin + ray_direction * focus_t;
        // Uniform disk sample (r = sqrt(u) for uniform area density).
        let lens_rand = rand_vec2f(&rng);
        let lens_r = solari_view.aperture_radius * sqrt(lens_rand.x);
        let lens_theta = 6.283185307 * lens_rand.y;
        ray_origin += camera_right * (lens_r * cos(lens_theta))
            + camera_up * (lens_r * sin(lens_theta));
        ray_direction = normalize(focus_point - ray_origin);
    }

    // Aerial perspective (primary ray only): remember the camera ray and capture
    // the primary hit distance, to apply distance haze once the path completes.
    let camera_position = ray_origin;
    let primary_ray_direction = ray_direction;
    var primary_distance = -1.0;

    // Ray-cone texture LOD: the cone's diameter `cone_width` grows by the pixel's
    // angular footprint (`2·tan(fovY/2)/height`, from `clip_from_view[1][1] =
    // 1/tan(fovY/2)`) over each segment's length, so distant and indirect hits
    // sample coarser mips. `cone_spread` is a loop-invariant constant (no
    // per-bounce roughness widening) to keep the path loop's live state minimal —
    // distance alone already coarsens deep bounces. Only `cone_width` persists.
    let cone_spread = 2.0 / (view.clip_from_view[1].y * view.viewport.w);
    var cone_width = 0.0;

    var radiance = vec3(0.0);
    var throughput = vec3(1.0);
    var p_bounce = 0.0;
    // Nested-dielectric medium stack (Schmidt & Budge): every transmissive
    // volume the path is currently inside, identified by material slot. The
    // highest-priority entry is the ACTIVE medium (air when empty) — it
    // provides the segment absorption and the incident IOR, and boundaries of
    // lower-priority volumes inside it are false interfaces (NVIDIA models
    // the wine interpenetrating its glass; priority resolves the overlap).
    var medium_id: array<u32, MEDIUM_STACK_SIZE>;
    var medium_priority: array<u32, MEDIUM_STACK_SIZE>;
    var medium_ior: array<f32, MEDIUM_STACK_SIZE>;
    var medium_dispersion: array<f32, MEDIUM_STACK_SIZE>;
    var medium_extinction_entry: array<vec3<f32>, MEDIUM_STACK_SIZE>;
    var medium_count = 0u;
    // Active-medium absorption, recomputed from the stack on every crossing.
    var medium_extinction = vec3(0.0);
    // Hero wavelength (nm). 0 = the path is still RGB; the first dispersive
    // interface collapses it to a single sampled λ (throughput takes the
    // spectral→RGB weight once), and every dispersive eta after that uses
    // n(λ). Paths that never touch dispersive glass pay nothing.
    var lambda = 0.0;
    // Hard path-length cap: russian roulette terminates almost every path
    // long before this; the cap is GPU-timeout insurance.
    var bounces = 0u;
    loop {
        // Portal surfaces teleport the ray and black holes bend it mid-trace
        // (origin/direction updated in place); accretion-disk emission rides
        // back in `traversal_emitted`, captured rays end the path BLACK.
        var traversal_emitted = vec3(0.0);
        var traversal_captured = 0u;
        let ray = trace_ray_traversal(&ray_origin, &ray_direction, ray_t_min, RAY_FLAG_NONE, &traversal_emitted, &traversal_captured);
        radiance += throughput * traversal_emitted;
        if traversal_captured != 0u {
            break;
        }
        if ray.kind != RAY_QUERY_INTERSECTION_NONE {
            // Hair (linear swept spheres): a separate primitive type. The ray
            // query exposes no NV LSS getters here, so reconstruct the fiber
            // tangent from the hit segment's endpoints and shade with the Chiang
            // fiber BSDF (no surface normal, no textures, no cone LOD).
            if is_hair_instance(ray.instance_index) {
                let world_position = ray_origin + ray_direction * ray.t;
                throughput *= exp(-medium_extinction * ray.t);
                if p_bounce == 0.0 {
                    primary_distance = length(world_position - camera_position);
                }
                let hair = resolve_hair_hit(ray.instance_index, ray.primitive_index, world_position);
                let wo = -ray_direction;

                // Direct lighting (NEE) — shared with the realtime path.
                radiance += throughput * hair_direct_lighting(world_position, wo, hair, &rng);

                // Next bounce from the fiber BSDF.
                let next = sample_hair_bsdf(wo, hair.tangent, hair.sigma_a,
                    hair.beta_m, hair.beta_n, hair.alpha, hair.ior, &rng);
                if next.pdf == 0.0 { break; }
                ray_direction = next.wi;
                // Hair has no surface normal — nudge the origin along the new
                // direction to avoid re-hitting the same segment.
                ray_origin = world_position + next.wi * 1e-3;
                ray_t_min = 0.0;
                p_bounce = next.pdf;
                throughput *= next.throughput;

                let p = min(luminance(throughput), 0.95);
                if rand_f(&rng) > p { break; }
                throughput /= p;
                bounces += 1u;
                if bounces >= MAX_BOUNCES { break; }
                continue;
            }
            // Grow the cone over the segment just travelled, then resolve this
            // hit's textures at the cone's mip.
            cone_width += cone_spread * ray.t;
            let ray_hit = resolve_ray_hit_full_lod(ray, cone_width, ray_direction);
            throughput *= exp(-medium_extinction * length(ray_hit.world_position - ray_origin));
            if p_bounce == 0.0 { // Primary hit — distance for aerial perspective.
                primary_distance = length(ray_hit.world_position - camera_position);
            }
            let wo = -ray_direction;
            // Bend the smooth shading normal into the view hemisphere so silhouette
            // edges (where the interpolated normal dips past 90°) don't black out.
            let world_normal = bend_shading_normal(ray_hit.world_normal, wo);
            let NdotV = max(dot(world_normal, wo), 0.0001);
            let F_ab = F_AB(ray_hit.material.perceptual_roughness, NdotV);

            // Emissive contribution
            var mis_weight = 1.0;
            if p_bounce != 0.0 { // Not first bounce
                let p_light = random_emissive_light_pdf(ray_hit);
                mis_weight = power_heuristic(p_bounce, p_light);
            }
            radiance += mis_weight * throughput * ray_hit.material.emissive;

            if rand_f(&rng) < ray_hit.material.specular_transmission {
                // Transmissive surface: reflect or refract (delta lobes — no
                // next-event estimation), crossing in/out of a volume.
                // Whether the ray is entering: tested against the TRUE
                // (planar, vertex-normal-sign-matched) triangle normal — not
                // `front_face` (these assets' winding is inconsistent, which
                // alternates eta per triangle) and not the interpolated
                // normal (which crosses the horizon mid-facet at grazing) —
                // either mismatch reads as bands of total-internal-reflection
                // mirror. `bend_shading_normal` then handles the residual
                // past-horizon interpolation of the shading normal.
                let entering = dot(ray_direction, ray_hit.geometric_world_normal) < 0.0;
                let oriented_geometric_normal = select(
                    -ray_hit.geometric_world_normal,
                    ray_hit.geometric_world_normal,
                    entering,
                );
                let oriented_normal = bend_shading_normal(
                    select(-ray_hit.world_normal, ray_hit.world_normal, entering),
                    wo);

                // Stack lookup: this material's entry (for exits), and the
                // dominant medium among the OTHER entries — the medium on the
                // far side of this boundary (air when none).
                var self_index = MEDIUM_NOT_FOUND;
                var other_priority = 0u;
                var other_ior = 1.0;
                var other_dispersion = 0.0;
                for (var i = 0u; i < medium_count; i += 1u) {
                    if medium_id[i] == ray_hit.material_id {
                        self_index = i;
                        continue;
                    }
                    if medium_priority[i] >= other_priority {
                        other_priority = medium_priority[i];
                        other_ior = medium_ior[i];
                        other_dispersion = medium_dispersion[i];
                    }
                }
                // A boundary inside a strictly higher-priority volume is a
                // false interface: the overlap belongs to the other medium,
                // so the ray crosses with no optical event. Stack membership
                // still updates so the real exit is recognized later.
                let true_interface = ray_hit.material.nested_priority >= other_priority;

                var crossed = true;
                if true_interface {
                    var self_ior = ray_hit.material.ior;
                    var far_ior = other_ior;
                    if ray_hit.material.dispersion > 0.0 || other_dispersion > 0.0 {
                        // Hero-wavelength collapse at the first dispersive
                        // interface; both sides of the boundary then refract
                        // at their λ-dependent indices.
                        if lambda == 0.0 {
                            lambda = sample_hero_wavelength(&rng);
                            throughput *= spectral_rgb_weight(lambda);
                        }
                        self_ior = dispersive_ior(self_ior, ray_hit.material.dispersion, lambda);
                        far_ior = dispersive_ior(far_ior, other_dispersion, lambda);
                    }
                    let eta = select(
                        self_ior / far_ior, // exiting: M → far side
                        far_ior / self_ior, // entering: far side → M
                        entering,
                    );
                    let glass = sample_glass_bsdf(wo, oriented_normal, eta, &rng);
                    ray_direction = glass.wi;
                    crossed = glass.refracted;
                    p_bounce = bitcast<f32>(0x7F800000u); // INF: delta lobe
                }

                if crossed {
                    if entering {
                        if self_index == MEDIUM_NOT_FOUND && medium_count < MEDIUM_STACK_SIZE {
                            medium_id[medium_count] = ray_hit.material_id;
                            medium_priority[medium_count] = ray_hit.material.nested_priority;
                            medium_ior[medium_count] = ray_hit.material.ior;
                            medium_dispersion[medium_count] = ray_hit.material.dispersion;
                            medium_extinction_entry[medium_count] = ray_hit.material.extinction;
                            medium_count += 1u;
                        }
                    } else if self_index != MEDIUM_NOT_FOUND {
                        medium_count -= 1u;
                        medium_id[self_index] = medium_id[medium_count];
                        medium_priority[self_index] = medium_priority[medium_count];
                        medium_ior[self_index] = medium_ior[medium_count];
                        medium_dispersion[self_index] = medium_dispersion[medium_count];
                        medium_extinction_entry[self_index] = medium_extinction_entry[medium_count];
                    }
                    // Active medium = highest-priority remaining entry.
                    medium_extinction = vec3(0.0);
                    var active_priority = 0u;
                    var found = false;
                    for (var i = 0u; i < medium_count; i += 1u) {
                        if !found || medium_priority[i] >= active_priority {
                            active_priority = medium_priority[i];
                            medium_extinction = medium_extinction_entry[i];
                            found = true;
                        }
                    }
                }

                // Continue on the side of the surface the path is now on,
                // from a ULP-offset origin: any world-space epsilon either
                // skips interfaces (a wine glass wall is ~1 mm and its liquid
                // sits closer still — skipped boundaries also corrupt the
                // medium stack) or self-intersects at large coordinates.
                let offset_normal = select(oriented_geometric_normal, -oriented_geometric_normal, crossed);
                ray_origin = offset_ray_origin(ray_hit.world_position, offset_normal);
                ray_t_min = 0.0;
            } else {
                // Sample direct lighting, but only if the surface is not mirror-like
                let is_perfectly_specular = ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && ray_hit.material.metallic > 0.9999;
                if !is_perfectly_specular {
                    let direct_lighting = sample_random_light(ray_hit.world_position, world_normal, &rng);

                    mis_weight = 1.0;
                    if direct_lighting.brdf_rays_can_hit {
                        let pdf_of_bounce = brdf_pdf(wo, direct_lighting.wi, world_normal, ray_hit.material, F_ab);
                        mis_weight = power_heuristic(1.0 / direct_lighting.inverse_pdf, pdf_of_bounce);
                    }

                    let direct_lighting_brdf = evaluate_brdf(wo, direct_lighting.wi, world_normal, ray_hit.material, F_ab);
                    radiance += mis_weight * throughput * direct_lighting.radiance * direct_lighting.inverse_pdf * direct_lighting_brdf;
                }

                // Sample new ray direction from the material BRDF for next bounce and apply BRDF
                let next_bounce = evaluate_and_sample_brdf(wo, world_normal, ray_hit.material, F_ab, &rng);
                if next_bounce.pdf == 0.0 { break; }
                ray_direction = next_bounce.wi;
                ray_origin = offset_ray_origin(ray_hit.world_position, ray_hit.geometric_world_normal);
                ray_t_min = 0.0;
                p_bounce = next_bounce.pdf;
                throughput *= next_bounce.throughput;
            }

            // Russian roulette for early termination. Survival is capped
            // below 1 (unbiased — the ÷p compensates) so even lossless paths
            // terminate: clear glass keeps throughput at exactly 1, and a ray
            // in total internal reflection inside a window pane would
            // otherwise bounce forever.
            let p = min(luminance(throughput), 0.95);
            if rand_f(&rng) > p { break; }
            throughput /= p;

            bounces += 1u;
            if bounces >= MAX_BOUNCES { break; }
        } else {
            // Ray escaped the scene. With a skybox: add it (cube × brightness) in
            // the ray direction — raw radiance, exposed once at the end; the sky
            // both lights surfaces (bounce misses) and is the background (primary
            // miss). No MIS weight (the sky isn't in next-event estimation).
            if solari_view.environment_brightness > 0.0 {
                let sky = textureSampleLevel(
                    environment_map, environment_map_sampler, ray_direction, 0.0,
                ).rgb;
                radiance += throughput * solari_view.environment_brightness * sky;
            } else if p_bounce == 0.0 {
                // No skybox + primary ray → the camera clear color as a flat
                // background, shown as-is: ÷ exposure cancels the end-of-loop
                // × exposure (the clear is a fixed framebuffer value, not
                // exposure-scaled), then tonemapping applies. A *bounce* miss with
                // no skybox stays black — a flat clear color shouldn't add light.
                radiance += throughput * solari_view.clear_color / max(view.exposure, 1e-6);
            }
            // Sun disk(s): on a primary miss (the visible sky), draw each
            // directional light as a disk of its angular radius (`cos_theta_max`) at
            // its `luminance`. Primary-only — surfaces already receive the sun via
            // next-event estimation, so adding the disk on bounce misses would
            // double-count it. The atmosphere's Mie glow halo comes from the sky
            // cube; this is the bright disk itself, which the bake doesn't draw.
            if p_bounce == 0.0 {
                // Walked via the active list: the slot-indexed column keeps
                // stale luminance in freed slots (would draw a ghost sun).
                let emissive_count = active_light_list[0];
                let num_directional = active_light_list[1];
                for (var i = 0u; i < num_directional; i = i + 1u) {
                    let source = light_sources[active_light_list[2u + emissive_count + i]];
                    let sun = directional_lights[source.id];
                    if dot(ray_direction, sun.direction_to_light) >= sun.cos_theta_max {
                        radiance += throughput * sun.luminance;
                    }
                }
            }
            break;
        }
    }

    // Aerial perspective: fade the primary surface toward the sky it occludes by
    // a distance-based haze transmittance (Koschmieder). Blending toward the baked
    // sky cube — rather than a separately-calibrated inscatter — makes distant
    // geometry converge to exactly the sky behind it, so it's seamless with rays
    // that fall through to sky. Primary rays only (NVIDIA-style); reflections/
    // bounces stay unhazed.
    //
    // Volumetric aerial perspective (god rays), applied to BOTH primary hits and
    // primary misses (the sky) — otherwise the near fog hazes geometry but not
    // the sky behind it, leaving a hard seam at the silhouette. The medium is the
    // global height fog plus any local fog volumes. March the camera→hit segment
    // for a hit, or out to the medium's extent for a miss. Accumulate
    // single-scattered sunlight — shadowed per step by a ray toward the sun, so
    // buildings cast real shafts and shadowed fog stops glowing — plus isotropic
    // ambient skylight (a soft sky tint, no sun-ward glare). In-scatter is
    // weighted by running fog transmittance; the surface/sky already in
    // `radiance` is attenuated by the total. Few steps + per-pixel jitter + the
    // path tracer's temporal accumulation keep it smooth. Primary rays only.
    {
        var hit_t = RAY_T_MAX;
        if primary_distance > 0.0 {
            hit_t = primary_distance;
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
        let vol_span = fog_volumes_range(camera_position, primary_ray_direction, hit_t);
        if vol_span.y > vol_span.x {
            if t_end > 0.0 {
                t_end = max(t_end, vol_span.y);
            } else {
                t_start = vol_span.x;
                t_end = vol_span.y;
            }
        }
        if t_end > t_start {
            let AERIAL_STEPS = 16u;
            // Strata: when the volumes occupy a small slice of a long global
            // span, a uniform march undersamples them (a few-metre mist pool
            // gets <1 of the steps over a hundreds-of-metres sky ray) —
            // low-frequency noise until accumulation settles. Bracket the
            // march at the volume span and give that segment half the steps.
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
            var total_steps = 0u;
            for (var si = 0u; si < seg_count; si += 1u) {
                total_steps += seg_steps[si];
            }
            let cos_theta = dot(atmosphere.sun_direction, primary_ray_direction);
            let sun_phase = atmosphere_mie_phase(atmosphere.aerial_phase_g, cos_theta);
            // Sunlight reaching the fog, attenuated by the atmosphere toward the sun
            // (Beer-Lambert over the same optical depth the sky bake uses). For a low
            // sun this reddens and dims the shafts so they match the sky instead of
            // blowing out to white. Evaluated once at the camera — the fog is a thin
            // near-ground layer, so the sun transmittance is ~constant across it.
            // Zero without an atmosphere view: fog volumes are then ambient-lit.
            let planet_camera = vec3(0.0, atmosphere.bottom_radius + atmosphere.camera_altitude, 0.0);
            let sun_od = atmosphere_sun_optical_depth(atmosphere, planet_camera);
            let sun_radiance = atmosphere.sun_illuminance * exp(-(
                atmosphere.rayleigh_scattering * sun_od.x + vec3(atmosphere.mie_extinction) * sun_od.y));
            let sun_lit = any(sun_radiance > vec3(0.0));
            let sky_ambient = textureSampleLevel(
                environment_map, environment_map_sampler, vec3(0.0, 1.0, 0.0), 0.0,
            ).rgb * solari_view.environment_brightness;
            // Local-light sampling runs at ONE randomly chosen step per
            // sample, weighted by the step count (unbiased) — accumulation
            // converges it, and per-step sampling multiplies the march's cost
            // by the step count.
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
                    // Independent per-step jitter: a shared offset makes the
                    // whole march's error coherent per pixel (blotches);
                    // independent strata read as fine grain and settle faster.
                    let p = camera_position + primary_ray_direction
                        * (seg_bounds[si].x + (f32(i) + rand_f(&rng)) * ds);
                    var s = fog_volumes_sample(p, cos_theta);
                    // Local-light NEE is gated on FOG-VOLUME scattering (see
                    // the compose pass) — volumes are the opt-in; the global
                    // height fog alone takes only sun + sky.
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
                    // Tinted sun visibility: the shadow ray toward the sun
                    // accumulates colored Beer-Lambert through any stained glass
                    // it crosses, so the in-scattered shafts take the window's
                    // color instead of being hard-cut at the glass.
                    var sun_vis = vec3(0.0);
                    if sun_lit {
                        sun_vis = trace_light_transmittance(p, vec4(atmosphere.sun_direction, 0.0));
                    }
                    let in_scatter = sun_radiance * (sun_vis * s.sun_scatter) + sky_ambient * s.sigma_s;
                    inscatter += fog_transmittance * in_scatter * ds;

                    // Local (emissive) lights: one uniform NEE sample for the
                    // step — emissive-only (the sun term above is the
                    // directional stratum), phase-weighted at the media's
                    // scattering-averaged `g`, shadowed by a real visibility
                    // ray. Accumulation converges the single-sample variance;
                    // the realtime compose pass uses the ReGIR grid instead.
                    if fog_step == light_step && volume_light_weight > 1e-7 && emissive_light_count() > 0u {
                        let light = generate_random_emissive_light_sample(&rng);
                        if light.light_sample.light_id != NULL_LIGHT_ID {
                            let contribution = calculate_resolved_light_contribution(
                                light.resolved_light_sample, p, primary_ray_direction);
                            if any(contribution.radiance > vec3(0.0)) {
                                let g_eff = s.phase_g_sum / s.phase_weight;
                                let phase = atmosphere_mie_phase(g_eff, dot(contribution.wi, primary_ray_direction));
                                let vis = trace_light_visibility(p, light.resolved_light_sample.world_position);
                                inscatter += fog_transmittance * s.sigma_s
                                    * (phase * contribution.inverse_pdf * vis * f32(total_steps))
                                    * contribution.radiance * ds;
                            }
                        }
                    }

                    fog_transmittance *= exp(-s.sigma_t * ds);
                }
            }
            radiance = radiance * fog_transmittance + inscatter;
        }
    }

    // Camera exposure
    radiance *= view.exposure;

    // Safety net: a single non-finite sample (NaN/inf that slipped past the
    // sampling guards) would poison the running average permanently — drop
    // this pixel's sample for the frame instead.
    if any((bitcast<vec3<u32>>(radiance) & vec3(0x7fffffffu)) >= vec3(0x7f800000u)) {
        textureStore(view_output, global_id.xy, vec4(old_color.rgb, 1.0));
        return;
    }

    // Accumulation over time via running average
    let new_color = mix(old_color.rgb, radiance, 1.0 / (old_color.a + 1.0));
    textureStore(accumulation_texture, global_id.xy, vec4(new_color, old_color.a + 1.0));
    textureStore(view_output, global_id.xy, vec4(new_color, 1.0));
}
