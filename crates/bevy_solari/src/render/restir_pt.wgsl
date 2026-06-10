enable wgpu_ray_query;

// Passes 3 & 4 — ReSTIR DI (direct) + ReSTIR GI (one-bounce indirect).
//
// `initial_and_temporal`: per primary-hit pixel, build + temporally reuse both
//   a DI reservoir (a light sample) and a GI reservoir (a hemisphere
//   reconnection vertex with its incoming radiance). Reprojection is shared:
//   the current world position is projected with the previous-frame clip matrix
//   and validated against the previous-frame G-buffer. Results → the `_b`
//   (intermediate) reservoir buffers.
// `spatial_and_shade`: shade direct (DI reservoir, one final visibility ray) +
//   indirect (GI reservoir, one visibility ray to the reconnection vertex),
//   sum → view_output, and persist `_b` → `_a` as next frame's history.
//   Spatial reuse is still a no-op (3c).
//
// DI: light samples are world-space → no shift/Jacobian. GI: the reconnection
// vertex needs a geometry Jacobian on reuse. Both use confidence-weighted
// (M-based) MIS. No world cache → GI is a single indirect bounce (direct
// lighting at the reconnection vertex); multi-bounce returns with the unified
// PT reservoir later.

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_solari::pbr::{rand_f, sample_uniform_hemisphere, uniform_hemisphere_inverse_pdf, sample_disk}
#import bevy_render::maths::{PI, orthonormalize}
#import bevy_solari::brdf::{evaluate_brdf, evaluate_diffuse_brdf, evaluate_specular_brdf, F_AB}
#import bevy_solari::sampling::{LightSample, generate_random_light_sample, resolve_light_sample, calculate_resolved_light_contribution, trace_light_visibility, trace_point_visibility, sample_random_light, sample_ggx_vndf, ggx_vndf_pdf, ggx_vndf_sample_invalid, random_emissive_light_pdf, power_heuristic, isnan, NULL_LIGHT_ID}
#import bevy_solari::scene_bindings::{trace_ray, set_view_cull_mask, resolve_ray_hit_full, resolve_material, materials, light_sources, ResolvedMaterial, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD}
#import bevy_solari::restir_bindings::{view, view_output, gbuffer_position, gbuffer_normal, previous_gbuffer_position, previous_gbuffer_normal, gbuffer_uv, motion_vectors, reservoir_a, reservoir_b, gi_reservoir_a, gi_reservoir_b, GiReservoir, solari_view}

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

@compute @workgroup_size(8, 8, 1)
fn initial_and_temporal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    set_view_cull_mask(solari_view.cull_mask.x);
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }
    let pixel = global_id.xy;
    let index = reservoir_index(pixel);

    let surface = load_surface(pixel);
    if !surface.valid {
        store_reservoir_b(pixel, empty_reservoir());
        gi_reservoir_b[index] = gi_empty();
        return;
    }

    var rng = index + view.frame_count * 5782582u;
    let diffuse_brdf = surface.material.base_color / PI;

    // Direct (DI).
    let di_initial = generate_initial_reservoir(surface.world_position, surface.world_normal, diffuse_brdf, &rng);
    let di_temporal = load_temporal_reservoir(pixel, surface.world_position, surface.world_normal);
    let di_merged = merge_reservoirs(di_initial, di_temporal, surface.world_position, surface.world_normal, diffuse_brdf, &rng);
    store_reservoir_b(pixel, di_merged.reservoir);

    // Indirect (GI).
    let gi_initial = gi_generate_initial(surface.world_position, surface.world_normal, &rng);
    let gi_temporal = gi_load_temporal(pixel, surface.world_position, surface.world_normal);
    let gi_merged = gi_merge(gi_initial, gi_temporal.reservoir, gi_temporal.world_position, surface.world_position, surface.world_normal, diffuse_brdf, &rng);
    gi_reservoir_b[index] = gi_merged.reservoir;
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
        textureStore(view_output, pixel, vec4(0.0, 0.0, 0.0, 1.0));
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
    let NdotV = max(dot(surface.world_normal, wo), 0.0001);
    let F_ab = F_AB(surface.material.perceptual_roughness, NdotV);

    var radiance = surface.material.emissive;

    // Direct lighting from the DI reservoir (final visibility traced once).
    // Diffuse always; specular NEE too unless the surface is near-mirror, where
    // the specular GI pass takes over via BRDF sampling.
    if reservoir_valid(di_reservoir) {
        let contribution = reservoir_contribution(di_reservoir, surface.world_position, surface.world_normal, diffuse_brdf);
        let visibility = trace_light_visibility(surface.world_position + surface.world_normal * RAY_T_MIN, contribution.world_position);
        var brdf = evaluate_diffuse_brdf(wo, contribution.wi, surface.world_normal, surface.material, F_ab);
        if surface.material.roughness > SPECULAR_GI_FOR_DI_ROUGHNESS_THRESHOLD {
            brdf += evaluate_specular_brdf(wo, contribution.wi, surface.world_normal, surface.material, F_ab);
        }
        radiance += contribution.radiance * di_reservoir.unbiased_contribution_weight * visibility * brdf;
    }

    // Indirect lighting from the GI reservoir (visibility to the reconnection
    // vertex traced once).
    if gi_reservoir.confidence_weight > 0.0 {
        let wi = normalize(gi_reservoir.sample_point_world_position - surface.world_position);
        let visibility = trace_point_visibility(surface.world_position + surface.world_normal * RAY_T_MIN, gi_reservoir.sample_point_world_position);
        let brdf = evaluate_diffuse_brdf(wo, wi, surface.world_normal, surface.material, F_ab);
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
        return; // diffuse passes already wrote view_output (black on miss).
    }

    var rng = reservoir_index(pixel) + view.frame_count * 5782582u + 0x68bc21ebu;

    let wo = normalize(view.world_position - surface.world_position);

    // Sample the GGX specular lobe in tangent space.
    let TBN = orthonormalize(surface.world_normal);
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

    var radiance = trace_specular_path(surface, wi, pdf, &rng);
    if surface.material.roughness > MIRROR_ROUGHNESS_THRESHOLD {
        radiance /= pdf;
    }

    let NdotV = max(dot(surface.world_normal, wo), 0.0001);
    let F_ab = F_AB(surface.material.perceptual_roughness, NdotV);
    radiance *= evaluate_specular_brdf(wo, wi, surface.world_normal, surface.material, F_ab);

    let existing = textureLoad(view_output, pixel).rgb;
    textureStore(view_output, pixel, vec4(existing + radiance, 1.0));
}

// Up to 3 GGX-sampled specular bounces with NEE. No world cache, so a glossy
// chain just terminates (no diffuse-cache fallback at the end).
fn trace_specular_path(primary_surface: Surface, initial_wi: vec3<f32>, initial_pdf: f32, rng: ptr<function, u32>) -> vec3<f32> {
    var radiance = vec3(0.0);
    var throughput = vec3(1.0);

    var ray_origin = primary_surface.world_position + primary_surface.world_normal * RAY_T_MIN;
    var wi = initial_wi;
    var p_bounce = initial_pdf;

    for (var i = 0u; i < 3u; i++) {
        let ray = trace_ray(ray_origin, wi, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
        if ray.kind == RAY_QUERY_INTERSECTION_NONE {
            break;
        }
        let hit = resolve_ray_hit_full(ray);

        let TBN = orthonormalize(hit.world_normal);
        let T = TBN[0];
        let B = TBN[1];
        let N = TBN[2];
        let wo = -wi;
        let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));
        let NdotV = max(dot(hit.world_normal, wo), 0.0001);
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
            let light = sample_random_light(hit.world_position, hit.world_normal, rng);
            let mis = specular_nee_mis_weight(light.inverse_pdf, light.brdf_rays_can_hit, light.wi, wo_tangent, hit.material.roughness, TBN);
            radiance += throughput * mis * light.radiance * light.inverse_pdf * evaluate_brdf(wo, light.wi, hit.world_normal, hit.material, F_ab);
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

fn generate_initial_reservoir(world_position: vec3<f32>, world_normal: vec3<f32>, diffuse_brdf: vec3<f32>, rng: ptr<function, u32>) -> Reservoir {
    var reservoir = empty_reservoir();
    var weight_sum = 0.0;
    var reservoir_target_function = 0.0;
    let mis_weight = 1.0 / f32(INITIAL_SAMPLES);

    for (var i = 0u; i < INITIAL_SAMPLES; i++) {
        let sample = generate_random_light_sample(rng);
        let contribution = calculate_resolved_light_contribution(sample.resolved_light_sample, world_position, world_normal);
        let target_function = luminance(contribution.radiance * diffuse_brdf * saturate(dot(contribution.wi, world_normal)));
        let resampling_weight = mis_weight * (target_function * contribution.inverse_pdf);

        weight_sum += resampling_weight;
        if rand_f(rng) < resampling_weight / weight_sum {
            reservoir.light_id = sample.light_sample.light_id;
            reservoir.seed = sample.light_sample.seed;
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
    // NOTE: the cross-frame light-id remap (old scene binding 11) was removed — it
    // was a CPU relic, broken for emissive scenes, and only this placeholder restir
    // path used it. `reservoir.light_id` is reused as-is; when restir is built for
    // real, do temporal light identity GPU-native via a stable light slot.
    reservoir.confidence_weight = min(reservoir.confidence_weight, DI_CONFIDENCE_WEIGHT_CAP);
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
    let sample = LightSample(reservoir.light_id, reservoir.seed);
    let resolved = resolve_light_sample(sample, light_sources[reservoir.light_id >> 16u]);
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
    if length(p0 - p1) > 0.01 * camera_distance {
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
