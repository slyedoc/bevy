enable wgpu_ray_query;

// Passes 3 & 4 — ReSTIR DI (direct) + ReSTIR GI (one-bounce indirect).
//
// `initial_and_temporal`: per primary-hit pixel, build + temporally reuse both
//   a DI reservoir (a light sample) and a GI reservoir (a hemisphere
//   reconnection vertex with its incoming radiance). Reprojection is shared:
//   the current world position is projected with the previous-frame clip matrix
//   and validated against the previous-frame G-buffer. Results → the `_b`
//   (intermediate) reservoir buffers.
// `spatial_and_shade`: merge one validated spatial neighbour into both
//   reservoirs, shade direct (DI reservoir, one final visibility ray) +
//   indirect (GI reservoir, one visibility ray to the reconnection vertex),
//   sum → view_output, and persist the result as next frame's history (`_a`).
//
// DI: light samples are world-space → no shift/Jacobian. GI: the reconnection
// vertex needs a geometry Jacobian on reuse. Both use confidence-weighted
// (M-based) MIS. No world cache → GI is a single indirect bounce (direct
// lighting at the reconnection vertex); multi-bounce returns with the unified
// PT reservoir later.

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_solari::pbr::{rand_f, rand_u, rand_range_u, sample_uniform_hemisphere, uniform_hemisphere_inverse_pdf, sample_disk}
#import bevy_render::maths::{PI, orthonormalize}
#import bevy_solari::brdf::{evaluate_brdf, evaluate_diffuse_brdf, evaluate_specular_brdf, F_AB, bend_shading_normal}
#import bevy_solari::sampling::{LightSample, ResolvedLightSample, generate_random_light_sample, resolve_light_sample, calculate_resolved_light_contribution, trace_light_visibility, trace_point_visibility, sample_random_light, sample_ggx_vndf, ggx_vndf_pdf, ggx_vndf_sample_invalid, random_emissive_light_pdf, power_heuristic, isnan, NULL_LIGHT_ID}
#import bevy_solari::scene_bindings::{trace_ray, set_view_cull_mask, resolve_ray_hit_full, resolve_material, materials, light_sources, active_light_list, directional_lights, ResolvedMaterial, LIGHT_SOURCE_KIND_NONE, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD}
#import bevy_solari::restir_bindings::{view, view_output, gbuffer_position, gbuffer_normal, previous_gbuffer_position, previous_gbuffer_normal, gbuffer_uv, motion_vectors, reservoir_a, reservoir_b, gi_reservoir_a, gi_reservoir_b, GiReservoir, solari_view, light_tiles, unpack_light_tile_sample, LightTileSample, LIGHT_TILE_BLOCKS, LIGHT_TILE_SAMPLES_PER_BLOCK, environment_map, environment_map_sampler, specular_hit_distance, regir_query, regir_find, regir_samples, REGIR_CELL_NONE, REGIR_ENTRIES_PER_CELL}

const INITIAL_SAMPLES = 8u;
const DI_CONFIDENCE_WEIGHT_CAP = 20.0;
const GI_CONFIDENCE_WEIGHT_CAP = 8.0;
const MAX_JACOBIAN = 1.2;
const SPATIAL_ATTEMPTS = 5u;
const SPATIAL_RADIUS_PIXELS = 30.0;
// Below this roughness the DI reservoir stops shading the specular lobe and
// hands it to the BRDF-sampled specular GI pass (which then owns first-bounce
// emissive). Above it, DI does specular NEE (needed for directional/analytic
// lights a reflection ray can't hit) and specular GI zeroes first-bounce
// emissive to avoid double-counting reflected emissive geometry.
const SPECULAR_GI_FOR_DI_ROUGHNESS_THRESHOLD = 0.0225;
// Within this range of a candidate light, the initial-candidate target
// includes a real visibility trace. At point-blank range an area light
// self-occludes most of its own surface (a receiver 2 radii from a bulb sees
// ~25% of it), so an unshadowed target keeps electing occluded points and the
// light's contribution gates on/off with each frame's pick. The rays are
// short and only contact-range pixels pay for them.
const SHADOWED_TARGET_DISTANCE_SQUARED = 1.0;

struct Reservoir {
    light_id: u32,
    seed: u32,
    confidence_weight: f32,            // M
    unbiased_contribution_weight: f32, // W (unshadowed)
}

struct Surface {
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    material: ResolvedMaterial,
    valid: bool,
}

struct Reprojection {
    valid: bool,
    pixel: vec2<u32>,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
}

struct ReservoirContribution {
    radiance: vec3<f32>,
    target_function: f32,
    wi: vec3<f32>,
    world_position: vec4<f32>,
}

struct MergeResult {
    reservoir: Reservoir,
    radiance: vec3<f32>,
    wi: vec3<f32>,
    world_position: vec4<f32>,
}

struct GiNeighbor {
    reservoir: GiReservoir,
    world_position: vec3<f32>,
}

struct GiMergeResult {
    reservoir: GiReservoir,
    radiance: vec3<f32>,
    wi: vec3<f32>,
}

// ---------------------------------------------------------------- entry points

// Boiling filter (cf. RTXDI): a rare candidate caught with a huge
// compensation weight survives merges on raw intensity and glows for its
// whole confidence lifetime — the "spot that appears then slowly fades".
// Kill any reservoir whose weight exceeds this multiple of its 8×8
// workgroup's average. Slightly biased (drops legitimate extreme energy),
// hugely stabilizing.
const BOILING_FILTER_STRENGTH = 20.0;

var<workgroup> boiling_energy: array<f32, 64>;

@compute @workgroup_size(8, 8, 1)
fn initial_and_temporal(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    set_view_cull_mask(solari_view.cull_mask.x);
    let pixel = global_id.xy;
    let index = reservoir_index(pixel);
    // No early returns: the boiling filter's workgroup barriers below need
    // uniform control flow, so inactive threads carry empty reservoirs
    // through instead of bailing.
    let in_viewport = all(pixel < vec2u(view.main_pass_viewport.zw));

    // Workgroup-uniform tile pick: all 64 threads draw their initial
    // candidates from the same 1024-entry tile, so the pool reads stay
    // coherent across the workgroup.
    var workgroup_rng = (workgroup_id.x * 7919u + workgroup_id.y) + view.frame_count * 5782582u;
    let tile_start = rand_range_u(LIGHT_TILE_BLOCKS, &workgroup_rng) * LIGHT_TILE_SAMPLES_PER_BLOCK;

    var surface: Surface;
    surface.valid = false;
    if in_viewport {
        surface = load_surface(pixel);
    }

    var di_reservoir = empty_reservoir();
    var gi_reservoir = gi_empty();
    if surface.valid {
        var rng = index + view.frame_count * 5782582u;
        let diffuse_brdf = surface.material.base_color / PI;

        // ReGIR cell for this surface point (inserting + marking it). Cold cell
        // (just inserted / probe failed) → the initial candidates fall back to
        // the workgroup's uniform tile.
        let regir_cell = regir_query(surface.world_position, surface.world_normal, view.world_position, &rng);

        // Direct (DI).
        let di_initial = generate_initial_reservoir(tile_start, regir_cell, surface.world_position, surface.world_normal, diffuse_brdf, &rng);
        let di_temporal = load_temporal_reservoir(pixel, surface.world_position, surface.world_normal);
        di_reservoir = merge_reservoirs(di_initial, di_temporal, surface.world_position, surface.world_normal, diffuse_brdf, &rng).reservoir;

        // Indirect (GI).
        let gi_initial = gi_generate_initial(surface.world_position, surface.world_normal, &rng);
        let gi_temporal = gi_load_temporal(pixel, surface.world_position, surface.world_normal);
        gi_reservoir = gi_merge(gi_initial, gi_temporal.reservoir, gi_temporal.world_position, surface.world_position, surface.world_normal, diffuse_brdf, &rng).reservoir;
    }

    // DI boiling filter.
    var energy = 0.0;
    if reservoir_valid(di_reservoir) {
        energy = di_reservoir.unbiased_contribution_weight;
    }
    boiling_energy[local_index] = energy;
    workgroupBarrier();
    var sum = 0.0;
    var live = 0u;
    for (var k = 0u; k < 64u; k += 1u) {
        let v = boiling_energy[k];
        sum += v;
        live += u32(v > 0.0);
    }
    if live > 0u && energy > BOILING_FILTER_STRENGTH * (sum / f32(live)) {
        di_reservoir = empty_reservoir();
    }

    // GI boiling filter (energy = expected contribution scale).
    workgroupBarrier();
    energy = 0.0;
    if gi_reservoir.confidence_weight > 0.0 {
        energy = luminance(gi_reservoir.radiance) * gi_reservoir.unbiased_contribution_weight;
    }
    boiling_energy[local_index] = energy;
    workgroupBarrier();
    sum = 0.0;
    live = 0u;
    for (var k = 0u; k < 64u; k += 1u) {
        let v = boiling_energy[k];
        sum += v;
        live += u32(v > 0.0);
    }
    if live > 0u && energy > BOILING_FILTER_STRENGTH * (sum / f32(live)) {
        gi_reservoir = gi_empty();
    }

    if in_viewport {
        store_reservoir_b(pixel, di_reservoir);
        gi_reservoir_b[index] = gi_reservoir;
    }
}

@compute @workgroup_size(8, 8, 1)
fn spatial_and_shade(@builtin(global_invocation_id) global_id: vec3<u32>) {
    set_view_cull_mask(solari_view.cull_mask.x);
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }
    let pixel = global_id.xy;
    let index = reservoir_index(pixel);

    let surface = load_surface(pixel);
    if !surface.valid {
        store_reservoir_a(pixel, empty_reservoir());
        gi_reservoir_a[index] = gi_empty();
        // Primary miss: the sky (or the camera clear color) is the background.
        // Raw radiance — compose applies aerial perspective + exposure; the
        // clear color is a fixed framebuffer value, so ÷ exposure pre-cancels.
        let ray_direction = primary_ray_direction(pixel);
        var background = vec3(0.0);
        if solari_view.environment_brightness > 0.0 {
            background = sky_radiance(ray_direction);
        } else {
            background = solari_view.clear_color / max(view.exposure, 1e-6);
        }
        // Each ACTIVE directional light as a disk of its angular radius — the
        // sky bake's Mie halo doesn't draw the disk itself. Walked via the
        // active list: the slot-indexed column keeps stale luminance in freed
        // slots, which would draw a ghost sun.
        let emissive_count = active_light_list[0];
        let num_directional = active_light_list[1];
        for (var i = 0u; i < num_directional; i = i + 1u) {
            let source = light_sources[active_light_list[2u + emissive_count + i]];
            let sun = directional_lights[source.id];
            if dot(ray_direction, sun.direction_to_light) >= sun.cos_theta_max {
                background += sun.luminance;
            }
        }
        textureStore(view_output, pixel, vec4(background, 1.0));
        return;
    }

    var rng = index + view.frame_count * 5782582u + 0x9e3779b9u;
    let diffuse_brdf = surface.material.base_color / PI;

    // Spatial reuse: merge one validated neighbour's reservoir (this frame's
    // intermediate `_b` buffers), then persist the combined result as next
    // frame's temporal history (`_a`).
    let neighbor = find_spatial_neighbor(pixel, surface, &rng);

    var di_reservoir = load_reservoir_b(pixel);
    if neighbor.valid {
        let neighbor_reservoir = load_reservoir_b(neighbor.pixel);
        di_reservoir = merge_reservoirs(di_reservoir, neighbor_reservoir, surface.world_position, surface.world_normal, diffuse_brdf, &rng).reservoir;
    }
    store_reservoir_a(pixel, di_reservoir);

    var gi_reservoir = gi_reservoir_b[index];
    if neighbor.valid {
        let neighbor_gi_reservoir = gi_reservoir_b[reservoir_index(neighbor.pixel)];
        gi_reservoir = gi_merge(gi_reservoir, neighbor_gi_reservoir, neighbor.world_position, surface.world_position, surface.world_normal, diffuse_brdf, &rng).reservoir;
    }
    gi_reservoir_a[index] = gi_reservoir;

    let wo = normalize(view.world_position - surface.world_position);
    // Bend the smooth shading normal into the view hemisphere so silhouette
    // edges (where the interpolated normal dips past 90°) don't black out.
    // The raw G-buffer normal stays in `surface` — reuse validation compares
    // stored normals across pixels/frames, which must stay un-bent.
    let shading_normal = bend_shading_normal(surface.world_normal, wo);
    let NdotV = max(dot(shading_normal, wo), 0.0001);
    let F_ab = F_AB(surface.material.perceptual_roughness, NdotV);

    var radiance = surface.material.emissive;

    // Directional lights (the sun) are shaded deterministically every frame —
    // one cone sample + shadow ray each — OUTSIDE the reservoir. A reservoir
    // holds one light sample per pixel; making the sun compete with a nearby
    // emissive for that slot patchworks the screen into per-light winners,
    // which reuse correlates into morphing light blobs. The light tiles feed
    // the reservoir emissive samples only.
    let emissive_count = active_light_list[0];
    let directional_count = active_light_list[1];
    for (var i = 0u; i < directional_count; i = i + 1u) {
        let slot = active_light_list[2u + emissive_count + i];
        let light_sample = LightSample(slot << 16u, rand_u(&rng));
        let resolved = resolve_light_sample(light_sample, light_sources[slot]);
        let contribution = calculate_resolved_light_contribution(resolved, surface.world_position, surface.world_normal);
        let visibility = trace_light_visibility(surface.world_position + surface.world_normal * RAY_T_MIN, resolved.world_position);
        var brdf = evaluate_diffuse_brdf(wo, contribution.wi, shading_normal, surface.material, F_ab);
        if surface.material.roughness > SPECULAR_GI_FOR_DI_ROUGHNESS_THRESHOLD {
            brdf += evaluate_specular_brdf(wo, contribution.wi, shading_normal, surface.material, F_ab);
        }
        radiance += contribution.radiance * resolved.inverse_pdf * visibility * brdf;
    }

    // Direct lighting from the DI reservoir (final visibility traced once).
    // Diffuse always; specular NEE too unless the surface is near-mirror, where
    // the specular GI pass takes over via BRDF sampling.
    if reservoir_valid(di_reservoir) {
        let contribution = reservoir_contribution(di_reservoir, surface.world_position, surface.world_normal, diffuse_brdf);
        let visibility = trace_light_visibility(surface.world_position + surface.world_normal * RAY_T_MIN, contribution.world_position);
        var brdf = evaluate_diffuse_brdf(wo, contribution.wi, shading_normal, surface.material, F_ab);
        if surface.material.roughness > SPECULAR_GI_FOR_DI_ROUGHNESS_THRESHOLD {
            brdf += evaluate_specular_brdf(wo, contribution.wi, shading_normal, surface.material, F_ab);
        }
        radiance += contribution.radiance * di_reservoir.unbiased_contribution_weight * visibility * brdf;
    }

    // Indirect lighting from the GI reservoir (visibility to the reconnection
    // vertex traced once).
    if gi_reservoir.confidence_weight > 0.0 {
        let wi = normalize(gi_reservoir.sample_point_world_position - surface.world_position);
        let visibility = trace_point_visibility(surface.world_position + surface.world_normal * RAY_T_MIN, gi_reservoir.sample_point_world_position);
        let brdf = evaluate_diffuse_brdf(wo, wi, shading_normal, surface.material, F_ab);
        radiance += gi_reservoir.radiance * gi_reservoir.unbiased_contribution_weight * visibility * brdf;
    }

    // Raw linear radiance — `compose` applies exposure.
    textureStore(view_output, pixel, vec4(radiance, 1.0));
}

// Pass 4b — specular GI. Trace a fresh GGX-sampled specular path each frame and
// add it on top of the diffuse passes' output. Not ReSTIR'd (single noisy
// sample, meant to be denoised downstream). Owns ALL specular, including
// direct-light highlights (reflected emissives), since DI shades diffuse only.
@compute @workgroup_size(8, 8, 1)
fn specular_gi(@builtin(global_invocation_id) global_id: vec3<u32>) {
    set_view_cull_mask(solari_view.cull_mask.x);
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }
    let pixel = global_id.xy;

    let surface = load_surface(pixel);
    if !surface.valid {
        // Sky: the reflection IS the environment — virtual point at infinity
        // for the DLSS specular-motion guide.
        textureStore(specular_hit_distance, pixel, vec4(RAY_T_MAX, 0.0, 0.0, 0.0));
        return; // diffuse passes already wrote view_output (black on miss).
    }
    // Zero distance = "no reflection data" — the guide resolve falls back to
    // the surface motion. Overwritten below once the reflection ray reports.
    textureStore(specular_hit_distance, pixel, vec4(0.0));

    var rng = reservoir_index(pixel) + view.frame_count * 5782582u + 0x68bc21ebu;

    let wo = normalize(view.world_position - surface.world_position);
    let shading_normal = bend_shading_normal(surface.world_normal, wo);

    // Sample the GGX specular lobe in tangent space.
    let TBN = orthonormalize(shading_normal);
    let T = TBN[0];
    let B = TBN[1];
    let N = TBN[2];
    let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));
    let wi_tangent = sample_ggx_vndf(wo_tangent, surface.material.roughness, &rng);
    if ggx_vndf_sample_invalid(wi_tangent) {
        return;
    }
    let wi = wi_tangent.x * T + wi_tangent.y * B + wi_tangent.z * N;
    let pdf = ggx_vndf_pdf(wo_tangent, wi_tangent, surface.material.roughness);

    var first_hit_t = RAY_T_MAX; // environment miss = reflection at infinity
    var radiance = trace_specular_path(surface, wi, pdf, &first_hit_t, &rng);
    textureStore(specular_hit_distance, pixel, vec4(first_hit_t, 0.0, 0.0, 0.0));
    if surface.material.roughness > MIRROR_ROUGHNESS_THRESHOLD {
        radiance /= pdf;
    }

    let NdotV = max(dot(shading_normal, wo), 0.0001);
    let F_ab = F_AB(surface.material.perceptual_roughness, NdotV);
    radiance *= evaluate_specular_brdf(wo, wi, shading_normal, surface.material, F_ab);

    let existing = textureLoad(view_output, pixel).rgb;
    textureStore(view_output, pixel, vec4(existing + radiance, 1.0));
}

// Up to 3 GGX-sampled specular bounces with NEE. No world cache, so a glossy
// chain just terminates (no diffuse-cache fallback at the end).
fn trace_specular_path(primary_surface: Surface, initial_wi: vec3<f32>, initial_pdf: f32, first_hit_t: ptr<function, f32>, rng: ptr<function, u32>) -> vec3<f32> {
    var radiance = vec3(0.0);
    var throughput = vec3(1.0);

    var ray_origin = primary_surface.world_position + primary_surface.world_normal * RAY_T_MIN;
    var wi = initial_wi;
    var p_bounce = initial_pdf;

    for (var i = 0u; i < 3u; i++) {
        let ray = trace_ray(ray_origin, wi, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
        if ray.kind == RAY_QUERY_INTERSECTION_NONE {
            // Sky reflection — full weight, the sky isn't in next-event estimation.
            radiance += throughput * sky_radiance(wi);
            break;
        }
        let hit = resolve_ray_hit_full(ray);
        if i == 0u {
            *first_hit_t = length(hit.world_position - primary_surface.world_position);
        }

        let wo = -wi;
        let hit_normal = bend_shading_normal(hit.world_normal, wo);
        let TBN = orthonormalize(hit_normal);
        let T = TBN[0];
        let B = TBN[1];
        let N = TBN[2];
        let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));
        let NdotV = max(dot(hit_normal, wo), 0.0001);
        let F_ab = F_AB(hit.material.perceptual_roughness, NdotV);

        // Emission MIS. Later bounces weight against the BSDF sample that found
        // the hit. First bounce: full weight only if DI did NOT do specular NEE
        // (near-mirror primary); otherwise 0 to avoid double-counting reflected
        // emissive geometry that DI's light sampling already caught.
        var emissive_mis = 0.0;
        if i != 0u {
            emissive_mis = power_heuristic(p_bounce, random_emissive_light_pdf(hit));
        } else if primary_surface.material.roughness <= SPECULAR_GI_FOR_DI_ROUGHNESS_THRESHOLD {
            emissive_mis = 1.0;
        }
        radiance += throughput * emissive_mis * hit.material.emissive;

        let is_mirror = hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && hit.material.metallic > 0.9999;
        if !is_mirror {
            let light = sample_random_light(hit.world_position, hit_normal, rng);
            let mis = specular_nee_mis_weight(light.inverse_pdf, light.brdf_rays_can_hit, light.wi, wo_tangent, hit.material.roughness, TBN);
            radiance += throughput * mis * light.radiance * light.inverse_pdf * evaluate_brdf(wo, light.wi, hit_normal, hit.material, F_ab);
        }

        let next_tangent = sample_ggx_vndf(wo_tangent, hit.material.roughness, rng);
        if ggx_vndf_sample_invalid(next_tangent) {
            break;
        }
        wi = next_tangent.x * T + next_tangent.y * B + next_tangent.z * N;
        ray_origin = hit.world_position + hit.geometric_world_normal * RAY_T_MIN;
        p_bounce = ggx_vndf_pdf(wo_tangent, next_tangent, hit.material.roughness);
        throughput *= evaluate_brdf(wo, wi, N, hit.material, F_ab);
        if hit.material.roughness > MIRROR_ROUGHNESS_THRESHOLD {
            throughput /= p_bounce;
        }

        let p = luminance(throughput);
        if rand_f(rng) > p {
            break;
        }
        throughput /= p;
    }

    return radiance;
}

fn specular_nee_mis_weight(inverse_p_light: f32, brdf_rays_can_hit: bool, light_wi: vec3<f32>, wo_tangent: vec3<f32>, roughness: f32, TBN: mat3x3<f32>) -> f32 {
    if !brdf_rays_can_hit {
        return 1.0;
    }
    let wi_tangent = vec3(dot(light_wi, TBN[0]), dot(light_wi, TBN[1]), dot(light_wi, TBN[2]));
    let p_light = 1.0 / inverse_p_light;
    let p_bounce = ggx_vndf_pdf(wo_tangent, wi_tangent, roughness);
    return power_heuristic(p_light, p_bounce);
}

// --------------------------------------------------------------- shared reuse

// Reproject the current world position to the previous frame and validate the
// previous-frame surface. The reservoir load is left to the caller (DI/GI use
// different buffers).
fn reproject_previous(pixel: vec2<u32>, world_position: vec3<f32>, world_normal: vec3<f32>) -> Reprojection {
    var result: Reprojection;
    result.valid = false;

    // Follow the motion vector back to the previous-frame pixel.
    let motion = textureLoad(motion_vectors, pixel).xy;
    let previous_pixel_f = vec2<f32>(pixel) - motion * view.main_pass_viewport.zw;
    if any(previous_pixel_f < vec2(0.0)) || any(previous_pixel_f >= view.main_pass_viewport.zw) {
        return result;
    }
    let previous_pixel = vec2<u32>(previous_pixel_f);

    let prev_gpos = textureLoad(previous_gbuffer_position, previous_pixel);
    if prev_gpos.w < 0.0 {
        return result;
    }
    let prev_normal = textureLoad(previous_gbuffer_normal, previous_pixel).xyz;
    if !surface_similar(world_position, world_normal, prev_gpos.xyz, prev_normal) {
        return result;
    }

    result.valid = true;
    result.pixel = previous_pixel;
    result.world_position = prev_gpos.xyz;
    result.world_normal = prev_normal;
    return result;
}

// Pick one current-frame neighbour pixel (disk around the center) whose surface
// matches the center — up to SPATIAL_ATTEMPTS tries. Used by both DI and GI
// spatial reuse.
fn find_spatial_neighbor(center_pixel: vec2<u32>, surface: Surface, rng: ptr<function, u32>) -> Reprojection {
    var result: Reprojection;
    result.valid = false;

    for (var i = 0u; i < SPATIAL_ATTEMPTS; i++) {
        let offset = sample_disk(SPATIAL_RADIUS_PIXELS, rng);
        let neighbor_f = clamp(vec2<f32>(center_pixel) + offset, vec2(0.0), view.main_pass_viewport.zw - 1.0);
        let pixel = vec2<u32>(neighbor_f);

        let neighbor_gpos = textureLoad(gbuffer_position, pixel);
        if neighbor_gpos.w < 0.0 {
            continue;
        }
        let neighbor_normal = textureLoad(gbuffer_normal, pixel).xyz;
        if !surface_similar(surface.world_position, surface.world_normal, neighbor_gpos.xyz, neighbor_normal) {
            continue;
        }

        result.valid = true;
        result.pixel = pixel;
        result.world_position = neighbor_gpos.xyz;
        result.world_normal = neighbor_normal;
        return result;
    }
    return result;
}

// --------------------------------------------------------------------- DI

// WRS over INITIAL_SAMPLES candidates drawn from the surface point's ReGIR
// cell (LIGHTS pre-selected for proximity; a fresh point is sampled per
// candidate — see the cell branch), or from one presampled light tile while
// the cell is cold (tile entries are points, used as-is). The selected
// `LightSample` identity is what the reservoir stores for exact re-resolution
// at merge/shade time. Note the cell path re-resolves the light per candidate
// (a cluster walk for the picked triangle) — cheap for lamp-sized emissive
// meshes; a scene with huge emissive meshes would want this bounded.
fn generate_initial_reservoir(tile_start: u32, regir_cell: u32, world_position: vec3<f32>, world_normal: vec3<f32>, diffuse_brdf: vec3<f32>, rng: ptr<function, u32>) -> Reservoir {
    var reservoir = empty_reservoir();
    var weight_sum = 0.0;
    var reservoir_target_function = 0.0;
    let mis_weight = 1.0 / f32(INITIAL_SAMPLES);

    for (var i = 0u; i < INITIAL_SAMPLES; i++) {
        var candidate_light_id: u32;
        var candidate_seed: u32;
        var resolved: ResolvedLightSample;
        if regir_cell != REGIR_CELL_NONE {
            // The cell entry selects a LIGHT (light-space contribution weight);
            // sample a FRESH point on it for this pixel. Pinning the entry's
            // stored point would correlate every receiver in the cell onto one
            // spot of the emitter — the lamp's illumination (and the shadows
            // it casts) would swing direction with each frame's entry roll
            // instead of resolving the penumbra spatially across pixels.
            let entry = regir_samples[regir_cell * REGIR_ENTRIES_PER_CELL + rand_range_u(REGIR_ENTRIES_PER_CELL, rng)];
            if entry.light_id == NULL_LIGHT_ID {
                continue;
            }
            let slot = entry.light_id >> 16u;
            let light_source = light_sources[slot];
            let triangle_id = rand_range_u(light_source.kind >> 1u, rng);
            candidate_light_id = (slot << 16u) | triangle_id;
            candidate_seed = rand_u(rng);
            resolved = resolve_light_sample(LightSample(candidate_light_id, candidate_seed), light_source);
            // light weight × fresh point's area inverse-pdf
            resolved.inverse_pdf *= entry.inverse_pdf;
        } else {
            let entry = light_tiles[tile_start + rand_range_u(LIGHT_TILE_SAMPLES_PER_BLOCK, rng)];
            if entry.light_id == NULL_LIGHT_ID {
                continue;
            }
            candidate_light_id = entry.light_id;
            candidate_seed = entry.seed;
            resolved = unpack_light_tile_sample(entry);
        }
        let contribution = calculate_resolved_light_contribution(resolved, world_position, world_normal);
        var target_function = luminance(contribution.radiance * diffuse_brdf * saturate(dot(contribution.wi, world_normal)));
        // Shadowed target at contact range (see SHADOWED_TARGET_DISTANCE_SQUARED).
        // Valid RIS: the contribution weight divides by this same target.
        let to_light = resolved.world_position.xyz - world_position;
        if target_function > 0.0 && dot(to_light, to_light) < SHADOWED_TARGET_DISTANCE_SQUARED {
            target_function *= trace_light_visibility(world_position + world_normal * RAY_T_MIN, resolved.world_position);
        }
        let resampling_weight = mis_weight * (target_function * contribution.inverse_pdf);

        weight_sum += resampling_weight;
        if rand_f(rng) < resampling_weight / weight_sum {
            reservoir.light_id = candidate_light_id;
            reservoir.seed = candidate_seed;
            reservoir_target_function = target_function;
        }
    }

    if reservoir_target_function > 0.0 {
        reservoir.unbiased_contribution_weight = weight_sum / reservoir_target_function;
    }
    reservoir.confidence_weight = 1.0;
    return reservoir;
}

fn load_temporal_reservoir(pixel: vec2<u32>, world_position: vec3<f32>, world_normal: vec3<f32>) -> Reservoir {
    let reprojection = reproject_previous(pixel, world_position, world_normal);
    if !reprojection.valid {
        return empty_reservoir();
    }

    var reservoir = load_reservoir_a(reprojection.pixel);
    // `light_id` packs the light's STABLE slot, so history survives lights
    // being added/removed; a dead light's slot resolves to a `NONE` hole and
    // the reservoir is rejected in `reservoir_contribution`. (A freed slot's
    // reuse can briefly misattribute history to the new occupant — bounded by
    // the confidence cap.)
    reservoir.confidence_weight = min(reservoir.confidence_weight, DI_CONFIDENCE_WEIGHT_CAP);

    // Visibility recheck (one shadow ray): the merge's target function is
    // unshadowed, so an occluded history sample would keep winning merges on
    // pure intensity and smear a bright spot for its whole confidence
    // lifetime. Discarding it here lets this frame's initial candidates take
    // the pixel immediately.
    if reservoir_valid(reservoir) {
        let light_source = light_sources[reservoir.light_id >> 16u];
        if light_source.kind == LIGHT_SOURCE_KIND_NONE {
            return empty_reservoir();
        }
        let sample = LightSample(reservoir.light_id, reservoir.seed);
        let resolved = resolve_light_sample(sample, light_source);
        let visibility = trace_light_visibility(world_position + world_normal * RAY_T_MIN, resolved.world_position);
        if visibility == 0.0 {
            return empty_reservoir();
        }
    }
    return reservoir;
}

fn merge_reservoirs(
    canonical: Reservoir,
    other: Reservoir,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    diffuse_brdf: vec3<f32>,
    rng: ptr<function, u32>,
) -> MergeResult {
    let canonical_contribution = reservoir_contribution(canonical, world_position, world_normal, diffuse_brdf);
    let other_contribution = reservoir_contribution(other, world_position, world_normal, diffuse_brdf);

    var combined = empty_reservoir();
    combined.confidence_weight = canonical.confidence_weight + other.confidence_weight;

    let m_total = combined.confidence_weight;
    let m_canonical = select(0.0, canonical.confidence_weight / m_total, m_total > 0.0);
    let m_other = select(0.0, other.confidence_weight / m_total, m_total > 0.0);
    let canonical_weight = m_canonical * canonical_contribution.target_function * canonical.unbiased_contribution_weight;
    let other_weight = m_other * other_contribution.target_function * other.unbiased_contribution_weight;
    let weight_sum = canonical_weight + other_weight;

    if rand_f(rng) < other_weight / weight_sum {
        combined.light_id = other.light_id;
        combined.seed = other.seed;
        combined.unbiased_contribution_weight = select(0.0, weight_sum / other_contribution.target_function, other_contribution.target_function > 0.0);
        return MergeResult(combined, other_contribution.radiance, other_contribution.wi, other_contribution.world_position);
    }
    combined.light_id = canonical.light_id;
    combined.seed = canonical.seed;
    combined.unbiased_contribution_weight = select(0.0, weight_sum / canonical_contribution.target_function, canonical_contribution.target_function > 0.0);
    return MergeResult(combined, canonical_contribution.radiance, canonical_contribution.wi, canonical_contribution.world_position);
}

fn reservoir_contribution(reservoir: Reservoir, world_position: vec3<f32>, world_normal: vec3<f32>, diffuse_brdf: vec3<f32>) -> ReservoirContribution {
    if !reservoir_valid(reservoir) {
        return ReservoirContribution(vec3(0.0), 0.0, vec3(0.0), vec4(0.0));
    }
    // The stored slot's light died (slot holed out as `NONE`): resolving it
    // would read another instance's geometry — reject the reservoir instead.
    let light_source = light_sources[reservoir.light_id >> 16u];
    if light_source.kind == LIGHT_SOURCE_KIND_NONE {
        return ReservoirContribution(vec3(0.0), 0.0, vec3(0.0), vec4(0.0));
    }
    let sample = LightSample(reservoir.light_id, reservoir.seed);
    let resolved = resolve_light_sample(sample, light_source);
    let contribution = calculate_resolved_light_contribution(resolved, world_position, world_normal);
    let target_function = luminance(contribution.radiance * diffuse_brdf * saturate(dot(contribution.wi, world_normal)));
    return ReservoirContribution(contribution.radiance, target_function, contribution.wi, resolved.world_position);
}

// --------------------------------------------------------------------- GI

fn gi_generate_initial(world_position: vec3<f32>, world_normal: vec3<f32>, rng: ptr<function, u32>) -> GiReservoir {
    var reservoir = gi_empty();

    let ray_direction = sample_uniform_hemisphere(world_normal, rng);
    let ray = trace_ray(world_position + world_normal * RAY_T_MIN, ray_direction, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
    if ray.kind == RAY_QUERY_INTERSECTION_NONE {
        // Sky miss → skylight as a reconnection vertex at a far virtual
        // distance. Reuse stays consistent: the shade visibility ray re-traces
        // toward (effectively) the same direction, and the spatial/temporal
        // jacobian degenerates to ~1 at this range.
        if solari_view.environment_brightness > 0.0 {
            reservoir.sample_point_world_position = world_position + ray_direction * SKY_VERTEX_DISTANCE;
            reservoir.sample_point_world_normal = -ray_direction;
            reservoir.confidence_weight = 1.0;
            reservoir.radiance = sky_radiance(ray_direction);
            reservoir.unbiased_contribution_weight = uniform_hemisphere_inverse_pdf();
        }
        return reservoir;
    }

    let sample_point = resolve_ray_hit_full(ray);
    // Emissive reconnection vertices are covered by direct lighting; skip them
    // here to avoid double counting.
    if any(sample_point.material.emissive != vec3(0.0)) {
        return reservoir;
    }

    reservoir.sample_point_world_position = sample_point.world_position;
    reservoir.sample_point_world_normal = sample_point.world_normal;
    reservoir.confidence_weight = 1.0;

    // One bounce of indirect: direct lighting at the reconnection vertex.
    let direct_lighting = sample_random_light(sample_point.world_position, sample_point.world_normal, rng);
    reservoir.radiance = direct_lighting.radiance * saturate(dot(direct_lighting.wi, sample_point.world_normal));
    reservoir.unbiased_contribution_weight = direct_lighting.inverse_pdf * uniform_hemisphere_inverse_pdf();
    reservoir.radiance *= sample_point.material.base_color / PI;
    return reservoir;
}

fn gi_load_temporal(pixel: vec2<u32>, world_position: vec3<f32>, world_normal: vec3<f32>) -> GiNeighbor {
    var neighbor: GiNeighbor;
    neighbor.reservoir = gi_empty();
    neighbor.world_position = vec3(0.0);

    let reprojection = reproject_previous(pixel, world_position, world_normal);
    if !reprojection.valid {
        return neighbor;
    }

    var reservoir = gi_reservoir_a[reservoir_index(reprojection.pixel)];
    reservoir.confidence_weight = min(reservoir.confidence_weight, GI_CONFIDENCE_WEIGHT_CAP);
    neighbor.reservoir = reservoir;
    neighbor.world_position = reprojection.world_position;
    return neighbor;
}

fn gi_merge(
    canonical: GiReservoir,
    other: GiReservoir,
    other_world_position: vec3<f32>,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    diffuse_brdf: vec3<f32>,
    rng: ptr<function, u32>,
) -> GiMergeResult {
    let canonical_wi = normalize(canonical.sample_point_world_position - world_position);
    let other_wi = normalize(other.sample_point_world_position - world_position);

    let canonical_tf = luminance(canonical.radiance * saturate(dot(canonical_wi, world_normal)) * diffuse_brdf);
    let other_tf = luminance(other.radiance * saturate(dot(other_wi, world_normal)) * diffuse_brdf);

    // Reconnect the other sample's vertex to this pixel (change of measure).
    let jacobian = gi_jacobian(world_position, other_world_position, other.sample_point_world_position, other.sample_point_world_normal);
    // Huge jacobians explode the variance — keep the canonical sample instead.
    if jacobian > MAX_JACOBIAN {
        return GiMergeResult(canonical, canonical.radiance, canonical_wi);
    }

    var combined = gi_empty();
    combined.confidence_weight = canonical.confidence_weight + other.confidence_weight;

    let m_total = combined.confidence_weight;
    let m_canonical = select(0.0, canonical.confidence_weight / m_total, m_total > 0.0);
    let m_other = select(0.0, other.confidence_weight / m_total, m_total > 0.0);
    let canonical_weight = m_canonical * canonical_tf * canonical.unbiased_contribution_weight;
    let other_weight = m_other * (other_tf * jacobian) * other.unbiased_contribution_weight;
    combined.weight_sum = canonical_weight + other_weight;

    if rand_f(rng) < other_weight / combined.weight_sum {
        combined.sample_point_world_position = other.sample_point_world_position;
        combined.sample_point_world_normal = other.sample_point_world_normal;
        combined.radiance = other.radiance;
        combined.unbiased_contribution_weight = select(0.0, combined.weight_sum / other_tf, other_tf > 0.0);
        return GiMergeResult(combined, other.radiance, other_wi);
    }
    combined.sample_point_world_position = canonical.sample_point_world_position;
    combined.sample_point_world_normal = canonical.sample_point_world_normal;
    combined.radiance = canonical.radiance;
    combined.unbiased_contribution_weight = select(0.0, combined.weight_sum / canonical_tf, canonical_tf > 0.0);
    return GiMergeResult(combined, canonical.radiance, canonical_wi);
}

fn gi_jacobian(new_world_position: vec3<f32>, original_world_position: vec3<f32>, sample_point_world_position: vec3<f32>, sample_point_world_normal: vec3<f32>) -> f32 {
    let r = new_world_position - sample_point_world_position;
    let q = original_world_position - sample_point_world_position;
    let rl = length(r);
    let ql = length(q);
    let phi_r = saturate(dot(r / rl, sample_point_world_normal));
    let phi_q = saturate(dot(q / ql, sample_point_world_normal));
    let jacobian = (phi_r * ql * ql) / (phi_q * rl * rl);
    return select(jacobian, 0.0, isinf(jacobian) || isnan(jacobian));
}

// --------------------------------------------------------------------- helpers

/// Virtual reconnection-vertex distance for skylight GI samples — far enough
/// that the reuse jacobian is ~1, well inside `RAY_T_MAX` so the shade pass's
/// visibility ray still traces.
const SKY_VERTEX_DISTANCE = 10000.0;

/// The sky's raw radiance in `direction` (the baked atmosphere / skybox cube ×
/// its brightness). Zero when the view has neither.
fn sky_radiance(direction: vec3<f32>) -> vec3<f32> {
    if solari_view.environment_brightness == 0.0 {
        return vec3(0.0);
    }
    let sky = textureSampleLevel(environment_map, environment_map_sampler, direction, 0.0).rgb;
    return sky * solari_view.environment_brightness;
}

/// The camera ray through `pixel` (jittered, like the visibility pass's).
fn primary_ray_direction(pixel: vec2<u32>) -> vec3<f32> {
    let pixel_center = vec2<f32>(pixel) + 0.5;
    let pixel_uv = pixel_center / view.main_pass_viewport.zw;
    let pixel_ndc = pixel_uv * 2.0 - 1.0;
    let ray_target = view.world_from_clip * vec4(pixel_ndc.x, -pixel_ndc.y, 1.0, 1.0);
    return normalize((ray_target.xyz / ray_target.w) - view.world_position);
}

fn load_surface(pixel: vec2<u32>) -> Surface {
    var surface: Surface;
    let gpos = textureLoad(gbuffer_position, pixel);
    surface.valid = gpos.w >= 0.0;
    if !surface.valid {
        return surface;
    }
    surface.world_position = gpos.xyz;
    surface.world_normal = textureLoad(gbuffer_normal, pixel).xyz;
    let uv = textureLoad(gbuffer_uv, pixel).xy;
    surface.material = resolve_material(materials[u32(gpos.w)], uv);
    return surface;
}

fn surface_similar(p0: vec3<f32>, n0: vec3<f32>, p1: vec3<f32>, n1: vec3<f32>) -> bool {
    let camera_distance = length(view.world_position - p0);
    // Plane distance, not point distance: a translating surface (a moving
    // car, the road sliding under it) stays within its own tangent plane, so
    // its history survives — while a depth disocclusion still rejects. A
    // point-distance test invalidates movers every frame, which strobes their
    // lighting and leaves denoiser afterimages where they were.
    if abs(dot(p0 - p1, n1)) > 0.01 * camera_distance {
        return false;
    }
    return dot(n0, n1) > 0.9;
}

fn isinf(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u;
}

fn reservoir_index(pixel: vec2<u32>) -> u32 {
    return pixel.x + pixel.y * u32(view.main_pass_viewport.z);
}

fn empty_reservoir() -> Reservoir {
    return Reservoir(NULL_LIGHT_ID, 0u, 0.0, 0.0);
}

fn reservoir_valid(reservoir: Reservoir) -> bool {
    return reservoir.light_id != NULL_LIGHT_ID;
}

fn gi_empty() -> GiReservoir {
    return GiReservoir(vec3(0.0), 0.0, vec3(0.0), 0.0, vec3(0.0), 0.0);
}

fn pack_reservoir(reservoir: Reservoir) -> vec4<u32> {
    return vec4(
        reservoir.light_id,
        reservoir.seed,
        bitcast<u32>(reservoir.confidence_weight),
        bitcast<u32>(reservoir.unbiased_contribution_weight),
    );
}

fn unpack_reservoir(packed: vec4<u32>) -> Reservoir {
    return Reservoir(packed.x, packed.y, bitcast<f32>(packed.z), bitcast<f32>(packed.w));
}

fn load_reservoir_a(pixel: vec2<u32>) -> Reservoir {
    return unpack_reservoir(reservoir_a[reservoir_index(pixel)]);
}

fn load_reservoir_b(pixel: vec2<u32>) -> Reservoir {
    return unpack_reservoir(reservoir_b[reservoir_index(pixel)]);
}

fn store_reservoir_a(pixel: vec2<u32>, reservoir: Reservoir) {
    reservoir_a[reservoir_index(pixel)] = pack_reservoir(reservoir);
}

fn store_reservoir_b(pixel: vec2<u32>, reservoir: Reservoir) {
    reservoir_b[reservoir_index(pixel)] = pack_reservoir(reservoir);
}

// ------------------------------------------------------------------ debug view
//
// Reservoir / ReGIR visualizations (`SolariDebugView::{DiWeight, DiConfidence,
// DiLight, RegirCells}`). Dispatched by the restir node after compose when
// `solari_view.debug_mode != 0` — these read buffers only this bind group
// sees, so they can't live in the generic overlay pass.

fn debug_hash_color(id: u32) -> vec3<f32> {
    let h = id * 747796405u + 2891336453u;
    return vec3(
        f32((h >> 0u) & 1023u),
        f32((h >> 10u) & 1023u),
        f32((h >> 20u) & 1023u),
    ) / 1023.0;
}

/// Blue → green → red heat ramp over `x` in [0, 1].
fn debug_heat(x: f32) -> vec3<f32> {
    let t = clamp(x, 0.0, 1.0);
    return vec3(smoothstep(0.5, 1.0, t), 1.0 - abs(t - 0.5) * 2.0, 1.0 - smoothstep(0.0, 0.5, t));
}

@compute @workgroup_size(8, 8, 1)
fn restir_debug(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }
    let pixel = global_id.xy;

    let surface = load_surface(pixel);
    if !surface.valid {
        textureStore(view_output, pixel, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    var color = vec3(0.0);
    let reservoir = load_reservoir_a(pixel);
    switch solari_view.debug_mode {
        case 1u: { // DI contribution weight: log heat; NaN = magenta, invalid = black.
            let w = reservoir.unbiased_contribution_weight;
            if isnan(w) {
                color = vec3(1.0, 0.0, 1.0);
            } else if reservoir_valid(reservoir) {
                color = debug_heat(log2(1.0 + w) / 16.0);
            }
        }
        case 2u: { // DI confidence (M) over its cap.
            color = vec3(reservoir.confidence_weight / DI_CONFIDENCE_WEIGHT_CAP);
        }
        case 3u: { // DI light identity (stable slot), hashed.
            if reservoir_valid(reservoir) {
                color = debug_hash_color(reservoir.light_id >> 16u);
            }
        }
        case 4u: { // ReGIR cell, hashed; red = cold/none.
            var rng = reservoir_index(pixel) + view.frame_count * 5782582u;
            let cell = regir_find(surface.world_position, surface.world_normal, view.world_position, &rng);
            if cell == REGIR_CELL_NONE {
                color = vec3(1.0, 0.0, 0.0);
            } else {
                color = debug_hash_color(cell);
            }
        }
        default: {}
    }
    textureStore(view_output, pixel, vec4(color, 1.0));
}
