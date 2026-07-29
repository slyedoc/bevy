// Light sampling that reaches the BINDLESS geometry/material path.
//
// Split out of `bevy_solari::sampling` because every function here transitively
// does a `physical_load` buffer-device-address read (`load_material_bindless`,
// `resolve_triangle_data_full`). That needs the `PhysicalStorageBufferAddresses`
// capability, which only the rt_pipeline's raw WGSL->SPIR-V path enables — a plain
// wgpu compute pipeline has no way to get it, and naga_oil pulls every function of
// an imported module into the consumer whether it is called or not. So `sampling`
// must stay capability-free for the wgpu spatial pass to build, and everything
// bindless lives here, imported only by RT-pipeline shaders.

#define_import_path bevy_solari::light_sampling

#import bevy_render::maths::PI_2
#import bevy_render::maths::orthonormalize
#import bevy_solari::pbr::{rand_f, rand_u, rand_vec2f, rand_range_u}
#import bevy_solari::sampling::{LightSample, ResolvedLightSample, GenerateRandomLightSampleResult, NULL_LIGHT_ID, emissive_light_count, directional_light_count, pick_emissive_weighted, pick_luminance, triangle_barycentrics}
#import bevy_solari::scene_bindings::{light_sources, active_light_list, directional_lights, LightSource, LIGHT_SOURCE_KIND_DIRECTIONAL, resolve_triangle_data_full, load_material_bindless, material_ids, ResolvedRayHitFull, MIRROR_ROUGHNESS_THRESHOLD, clusters, instance_cluster_ranges}

/// The pdf with which the NEE picker would have generated a sample on this
/// emissive hit — the BSDF-vs-NEE MIS counterpart. Must mirror the stratified
/// pick in `generate_random_emissive_light_sample` exactly.
fn random_emissive_light_pdf(hit: ResolvedRayHitFull) -> f32 {
    let stratum_probability = select(1.0, 0.5, directional_light_count() > 0u);
    return stratum_probability * random_emissive_light_pdf_flux(hit);
}
/// The same mirror WITHOUT the directional stratum factor — the restir estimator
/// samples emissives and directionals as separate techniques (no 50/50 pick).
fn random_emissive_light_pdf_flux(hit: ResolvedRayHitFull) -> f32 {
    let emissive_count = emissive_light_count();
    let cdf_base = 2u + emissive_count + directional_light_count();
    let total_flux = bitcast<f32>(active_light_list[cdf_base + emissive_count]);
    var pick_prob = 1.0 / f32(emissive_count);
    if total_flux > 0.0 {
        // Flux from the BASE (untextured) material emissive — the CPU CDF has no
        // access to textures, so the mirror must ignore them too.
        let base = load_material_bindless(hit.material_id);
        pick_prob = pick_luminance(base.emissive) * f32(hit.triangle_count) / total_flux;
    }
    return pick_prob / (f32(hit.triangle_count) * hit.triangle_area);
}
/// Re-resolve a stored reservoir sample at shading time with the FULL
/// flux-weighted technique pdf folded into `inverse_pdf` — must produce the
/// same pdf `generate_random_emissive_light_sample` gave the sample when it was
/// a candidate, so a re-evaluated target p̂ is the same function of
/// (surface, sample) on every pixel/frame that touches it.
fn resolve_emissive_for_restir(ls: LightSample) -> ResolvedLightSample {
    let light_source = light_sources[ls.light_id >> 16u];
    var resolved = resolve_light_sample(ls, light_source);
    let emissive_count = emissive_light_count();
    let cdf_base = 2u + emissive_count + directional_light_count();
    let total_flux = bitcast<f32>(active_light_list[cdf_base + emissive_count]);
    var pick_prob = 1.0 / f32(emissive_count);
    if total_flux > 0.0 {
        let triangle_count = light_source.kind >> 1u;
        let base = load_material_bindless(material_ids[light_source.id]);
        pick_prob = pick_luminance(base.emissive) * f32(triangle_count) / total_flux;
    }
    resolved.inverse_pdf *= 1.0 / max(pick_prob, 1e-12);
    return resolved;
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

    let picked = pick_emissive_weighted(emissive_count, directional_light_count(), rng);
    let light_id = active_light_list[2u + picked.index];
    let light_source = light_sources[light_id];

    let triangle_count = light_source.kind >> 1u;
    let triangle_id = rand_range_u(triangle_count, rng);

    let seed = rand_u(rng);
    let light_sample = LightSample((light_id << 16u) | triangle_id, seed);

    var resolved_light_sample = resolve_light_sample(light_sample, light_source);
    resolved_light_sample.inverse_pdf *= 1.0 / picked.prob;

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

    var emissive_stratum = emissive_count > 0u;
    var stratum_probability = 1.0;
    if directional_count > 0u && emissive_count > 0u {
        stratum_probability = 0.5;
        emissive_stratum = rand_f(rng) >= 0.5;
    } else if emissive_count == 0u && directional_count == 0u {
        let null_resolved = ResolvedLightSample(vec4(0.0, 1.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0), 0.0);
        return GenerateRandomLightSampleResult(LightSample(NULL_LIGHT_ID, 0u), null_resolved);
    }

    var pick: u32;
    var pick_prob: f32;
    if emissive_stratum {
        let picked = pick_emissive_weighted(emissive_count, directional_count, rng);
        pick = picked.index;
        pick_prob = picked.prob;
    } else {
        pick = emissive_count + rand_range_u(directional_count, rng);
        pick_prob = 1.0 / f32(directional_count);
    }
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
    resolved_light_sample.inverse_pdf *= 1.0 / (pick_prob * stratum_probability);

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
