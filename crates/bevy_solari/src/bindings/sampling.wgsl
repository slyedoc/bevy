enable wgpu_ray_query;

#define_import_path bevy_solari::sampling

#import bevy_solari::pbr::D_GGX
#import bevy_solari::pbr::{rand_f, rand_vec2f, rand_u, rand_range_u}
#import bevy_render::maths::{PI_2, orthonormalize}
#import bevy_solari::scene_bindings::{trace_ray, RAY_T_MIN, RAY_T_MAX, light_sources, active_light_list, directional_lights, LightSource, LIGHT_SOURCE_KIND_DIRECTIONAL, resolve_triangle_data_full, ResolvedRayHitFull, MIRROR_ROUGHNESS_THRESHOLD, clusters, instance_cluster_ranges}

fn power_heuristic(f: f32, g: f32) -> f32 {
    return balance_heuristic(f * f, g * g);
}

fn balance_heuristic(f: f32, g: f32) -> f32 {
    // Need to guard against NaNs since ReSTIR reservoirs can have UCW=0
    if f == 0.0 {
        return 0.0;
    }
    return max(0.0, 1.0 / (1.0 + (g / f)));
}

// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 1)
// Result is invalid when output.z <= 0.0, and must be discarded
fn sample_ggx_vndf(wi_tangent: vec3<f32>, roughness: f32, rng: ptr<function, u32>) -> vec3<f32> {
    // Mirror BRDF case
    if roughness <= MIRROR_ROUGHNESS_THRESHOLD {
        return vec3(-wi_tangent.xy, wi_tangent.z);
    }

    let i = wi_tangent;
    let rand = rand_vec2f(rng);
    let i_std = normalize(vec3(i.xy * roughness, i.z));
    let phi = PI_2 * rand.x;
    let a = roughness;
    let s = 1.0 + length(vec2(i.xy));
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
    let b = select(i_std.z, k * i_std.z, i.z > 0.0);
    let z = fma(1.0 - rand.y, 1.0 + b, -b);
    let sin_theta = sqrt(saturate(1.0 - z * z));
    let o_std = vec3(sin_theta * cos(phi), sin_theta * sin(phi), z);
    let m_std = i_std + o_std;
    let m = normalize(vec3(m_std.xy * roughness, m_std.z));
    return 2.0 * dot(i, m) * m - i;
}

fn ggx_vndf_sample_invalid(ray_tangent: vec3<f32>) -> bool {
    return !(ray_tangent.z > 0.0);
}

// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 2)
fn ggx_vndf_pdf(wi_tangent: vec3<f32>, wo_tangent: vec3<f32>, roughness: f32) -> f32 {
    // Mirror BRDF case
    if roughness <= MIRROR_ROUGHNESS_THRESHOLD {
        let mirror_wo = vec3(-wi_tangent.xy, wi_tangent.z);
        if all(abs(mirror_wo - wo_tangent) < vec3(0.0001)) {
            return bitcast<f32>(0x7F800000u); // INF
        } else {
            return 0.0;
        }
    }

    let i = wi_tangent;
    let o = wo_tangent;
    let m = normalize(i + o);
    let ndf = D_GGX(roughness, saturate(m.z));
    let ai = roughness * i.xy;
    let len2 = dot(ai, ai);
    let t = sqrt(len2 + i.z * i.z);
    var pdf: f32;
    if i.z >= 0.0 {
        let a = roughness;
        let s = 1.0 + length(i.xy);
        let a2 = a * a;
        let s2 = s * s;
        let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
        pdf = ndf / (2.0 * (k * i.z + t));
    } else {
        pdf = ndf * (t - i.z) / (2.0 * len2);
    }

    return select(pdf, 0.0, isnan(pdf));
}

fn isnan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

const NULL_LIGHT_ID = 0xFFFFFFFFu;

struct LightSample {
    light_id: u32,
    seed: u32,
}

struct ResolvedLightSample {
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    radiance: vec3<f32>,
    inverse_pdf: f32,
}

struct LightContribution {
    radiance: vec3<f32>,
    inverse_pdf: f32,
    wi: vec3<f32>,
    brdf_rays_can_hit: bool,
}

struct LightContributionNoPdf {
    radiance: vec3<f32>,
    wi: vec3<f32>,
}

struct GenerateRandomLightSampleResult {
    light_sample: LightSample,
    resolved_light_sample: ResolvedLightSample,
}

/// Number of active emissive-mesh lights (the `active_light_list` header).
fn emissive_light_count() -> u32 {
    return active_light_list[0];
}

/// Number of active directional lights (the `active_light_list` header).
fn directional_light_count() -> u32 {
    return active_light_list[1];
}

fn sample_random_light(ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, rng: ptr<function, u32>) -> LightContribution {
    let sample = generate_random_light_sample(rng);
    if sample.light_sample.light_id == NULL_LIGHT_ID {
        return LightContribution(vec3(0.0), 0.0, vec3(0.0, 1.0, 0.0), false);
    }
    var light_contribution = calculate_resolved_light_contribution(sample.resolved_light_sample, ray_origin, origin_world_normal);
    light_contribution.radiance *= trace_light_visibility(ray_origin, sample.resolved_light_sample.world_position);
    return light_contribution;
}

/// The pdf with which [`sample_random_light`] would have generated a sample on
/// this emissive hit — the BSDF-vs-NEE MIS counterpart. Must mirror the
/// stratified pick above exactly.
fn random_emissive_light_pdf(hit: ResolvedRayHitFull) -> f32 {
    let emissive_count = emissive_light_count();
    let stratum_probability = select(1.0, 0.5, directional_light_count() > 0u);
    return stratum_probability / (f32(emissive_count) * f32(hit.triangle_count) * hit.triangle_area);
}

/// One uniformly random EMISSIVE light sample — the light-tile pool's source.
/// The restir DI reservoirs manage emissive lights only: directional lights
/// are a deterministic handful, shaded every frame outside ReSTIR (a one-slot
/// reservoir forced to choose between the sun and a nearby emissive
/// patchworks the screen into per-light winners that reuse then correlates
/// into visible blobs).
fn generate_random_emissive_light_sample(rng: ptr<function, u32>) -> GenerateRandomLightSampleResult {
    let emissive_count = emissive_light_count();
    if emissive_count == 0u {
        let null_resolved = ResolvedLightSample(vec4(0.0, 1.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0), 0.0);
        return GenerateRandomLightSampleResult(LightSample(NULL_LIGHT_ID, 0u), null_resolved);
    }

    let pick = rand_range_u(emissive_count, rng);
    let light_id = active_light_list[2u + pick];
    let light_source = light_sources[light_id];

    let triangle_count = light_source.kind >> 1u;
    let triangle_id = rand_range_u(triangle_count, rng);

    let seed = rand_u(rng);
    let light_sample = LightSample((light_id << 16u) | triangle_id, seed);

    var resolved_light_sample = resolve_light_sample(light_sample, light_source);
    resolved_light_sample.inverse_pdf *= f32(emissive_count);

    return GenerateRandomLightSampleResult(light_sample, resolved_light_sample);
}

/// One stratified random light sample: pick the directional stratum (the sun)
/// or the emissive stratum with probability ½ each (when both exist), then
/// uniformly within the stratum; `inverse_pdf` carries the full pick pdf.
///
/// A single uniform pick over ALL sources samples the sun only 1/total of the
/// time at total× weight — with thousands of emissives that's firefly variance
/// on every sunlit surface (and a progressive accumulator keeps each outlier
/// visible for thousands of frames). [`random_emissive_light_pdf`] is the MIS
/// counterpart and must mirror this pick exactly.
///
/// The pick indexes `active_light_list` (dense, this frame's live lights);
/// the resulting `light_id` packs the picked light's **stable slot** into
/// `light_sources` — the identity reservoirs and light tiles store, valid
/// across frames regardless of lights being added or removed.
///
/// Returns a `NULL_LIGHT_ID` sample (zero radiance) when the scene has no
/// lights at all.
fn generate_random_light_sample(rng: ptr<function, u32>) -> GenerateRandomLightSampleResult {
    let emissive_count = emissive_light_count();
    let directional_count = directional_light_count();

    var stratum_base = 0u;
    var stratum_count = emissive_count;
    var stratum_probability = 1.0;
    if directional_count > 0u && emissive_count > 0u {
        stratum_probability = 0.5;
        if rand_f(rng) < 0.5 {
            stratum_base = emissive_count;
            stratum_count = directional_count;
        }
    } else if directional_count > 0u {
        stratum_base = emissive_count;
        stratum_count = directional_count;
    } else if emissive_count == 0u {
        let null_resolved = ResolvedLightSample(vec4(0.0, 1.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0), 0.0);
        return GenerateRandomLightSampleResult(LightSample(NULL_LIGHT_ID, 0u), null_resolved);
    }

    let pick = stratum_base + rand_range_u(stratum_count, rng);
    let light_id = active_light_list[2u + pick];
    let light_source = light_sources[light_id];

    var triangle_id = 0u;
    if light_source.kind != LIGHT_SOURCE_KIND_DIRECTIONAL {
        let triangle_count = light_source.kind >> 1u;
        triangle_id = rand_range_u(triangle_count, rng);
    }

    let seed = rand_u(rng);
    let light_sample = LightSample((light_id << 16u) | triangle_id, seed);

    var resolved_light_sample = resolve_light_sample(light_sample, light_source);
    resolved_light_sample.inverse_pdf *= f32(stratum_count) / stratum_probability;

    return GenerateRandomLightSampleResult(light_sample, resolved_light_sample);
}

fn resolve_light_sample(light_sample: LightSample, light_source: LightSource) -> ResolvedLightSample {
    if light_source.kind == LIGHT_SOURCE_KIND_DIRECTIONAL {
        let directional_light = directional_lights[light_source.id];

#ifndef NO_DIRECTIONAL_LIGHT_SOFT_SHADOWS
        // Sample a random direction within a cone whose base is the sun approximated as a disk
        // https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec30%3A305
        var rng = light_sample.seed;
        let random = rand_vec2f(&rng);
        let cos_theta = (1.0 - random.x) + random.x * directional_light.cos_theta_max;
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        let phi = random.y * PI_2;
        let x = cos(phi) * sin_theta;
        let y = sin(phi) * sin_theta;
        var direction_to_light = vec3(x, y, cos_theta);

        // Rotate the ray so that the cone it was sampled from is aligned with the light direction
        direction_to_light = orthonormalize(directional_light.direction_to_light) * direction_to_light;
#else
        let direction_to_light = directional_light.direction_to_light;
#endif

        return ResolvedLightSample(
        vec4(direction_to_light, 0.0),
        -direction_to_light,
        directional_light.luminance,
        directional_light.inverse_pdf,
    );
} else {
        let triangle_count = light_source.kind >> 1u;
        let instance_id = light_source.id;
        let instance_triangle_id = light_sample.light_id & 0xFFFFu;
        let barycentrics = triangle_barycentrics(light_sample.seed);

        // Walk this instance's clusters in pool order, prefix-summing
        // `cluster.triangle_count` until we cover `instance_triangle_id`.
        // The cluster that crosses the threshold owns the chosen
        // triangle; `triangle_in_cluster` is its local index.
        let range = instance_cluster_ranges[instance_id];
        let cluster_base = range.cluster_base;
        let cluster_count = range.cluster_count;

        var picked_cluster: u32 = cluster_base;
        var triangle_in_cluster: u32 = 0u;
        var running: u32 = 0u;
        for (var i: u32 = 0u; i < cluster_count; i = i + 1u) {
            let cluster_global = cluster_base + i;
            let tris = clusters[cluster_global].triangle_count;
            let next = running + tris;
            if instance_triangle_id < next {
                picked_cluster = cluster_global;
                triangle_in_cluster = instance_triangle_id - running;
                break;
            }
            running = next;
        }

        let triangle_data = resolve_triangle_data_full(instance_id, picked_cluster, triangle_in_cluster, barycentrics);

        return ResolvedLightSample(
            vec4(triangle_data.world_position, 1.0),
            triangle_data.world_normal,
            triangle_data.material.emissive.rgb,
            f32(triangle_count) * triangle_data.triangle_area,
        );
    }
}

fn calculate_resolved_light_contribution(resolved_light_sample: ResolvedLightSample, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightContribution {
    let ray = resolved_light_sample.world_position.xyz - (resolved_light_sample.world_position.w * ray_origin);
    // Clamp the inverse-square at contact range: a sample point numerically on
    // the receiver explodes `1/d²` to inf, and `inf × 0` from the visibility
    // trace (which returns 0 inside `RAY_T_MIN`) is NaN — one NaN permanently
    // poisons a progressive accumulator's running average.
    let light_distance = max(length(ray), RAY_T_MIN);
    let wi = ray / light_distance;

    let cos_theta_light = saturate(dot(-wi, resolved_light_sample.world_normal));
    let light_distance_squared = light_distance * light_distance;

    let radiance = resolved_light_sample.radiance * (cos_theta_light / light_distance_squared);

    return LightContribution(radiance, resolved_light_sample.inverse_pdf, wi, resolved_light_sample.world_position.w == 1.0);
}

fn resolve_and_calculate_light_contribution(light_sample: LightSample, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightContributionNoPdf {
    let resolved_light_sample = resolve_light_sample(light_sample, light_sources[light_sample.light_id >> 16u]);
    let light_contribution = calculate_resolved_light_contribution(resolved_light_sample, ray_origin, origin_world_normal);
    return LightContributionNoPdf(light_contribution.radiance, light_contribution.wi);
}

fn trace_light_visibility(ray_origin: vec3<f32>, light_sample_world_position: vec4<f32>) -> f32 {
    var ray_direction = light_sample_world_position.xyz;
    var ray_t_max = RAY_T_MAX;

    if light_sample_world_position.w == 1.0 {
        let ray = ray_direction - ray_origin;
        let dist = length(ray);
        ray_direction = ray / dist;
        ray_t_max = dist - RAY_T_MIN;
    }

    if ray_t_max < RAY_T_MIN { return 0.0; }

    let ray_hit = trace_ray(ray_origin, ray_direction, RAY_T_MIN, ray_t_max, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    return f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

fn trace_point_visibility(ray_origin: vec3<f32>, point: vec3<f32>) -> f32 {
    let ray = point - ray_origin;
    let dist = length(ray);
    let ray_direction = ray / dist;

    let ray_t_max = dist - RAY_T_MIN;
    if ray_t_max < RAY_T_MIN { return 0.0; }

    let ray_hit = trace_ray(ray_origin, ray_direction, RAY_T_MIN, ray_t_max, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    return f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec22%3A297
fn triangle_barycentrics(seed: u32) -> vec3<f32> {
    var rng = seed;
    var barycentrics = rand_vec2f(&rng);
    if barycentrics.x + barycentrics.y > 1.0 { barycentrics = 1.0 - barycentrics; }
    return vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics);
}
