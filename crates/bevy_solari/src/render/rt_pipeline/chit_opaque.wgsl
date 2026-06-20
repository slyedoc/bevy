// Opaque-surface closest-hit: emissive (MIS-weighted), next-event estimation,
// and a BRDF-sampled continuation ray. The register-isolated counterpart of the
// megakernel's opaque branch (pathtracer.wgsl) — the raygen driver owns the
// bounce loop + throughput; this shader fills the payload with the radiance
// contributed at this vertex and the next ray to trace.
//
// Shadow / light-visibility rays use inline `rayQuery` *inside* this closest-hit
// (sample_random_light → trace_light_visibility) — valid in any stage, so NEE
// needs no separate shadow miss/SBT program.
enable wgpu_ray_tracing_pipeline;
enable wgpu_ray_query;
enable primitive_index;

#import bevy_solari::rt_payload::RtPayload
#import bevy_solari::pbr::rand_f
#import bevy_solari::brdf::{evaluate_brdf, evaluate_and_sample_brdf, brdf_pdf, F_AB, bend_shading_normal}
#import bevy_solari::sampling::{sample_random_light, random_emissive_light_pdf, power_heuristic}
#import bevy_solari::scene_bindings::{resolve_triangle_data_full, offset_ray_origin, MIRROR_ROUGHNESS_THRESHOLD}

var<incoming_ray_payload> payload: RtPayload;
// Driver-provided triangle barycentrics (GLSL `hitAttributeEXT vec2`) — the
// fixed-function triangle intersection writes (u, v); w = 1 - u - v.
var<hit_attribute> bary: vec2<f32>;

@closest_hit
@incoming_payload(payload)
fn chit_opaque(
    @builtin(instance_id) instance_id: u32,
    // NV cluster-AS hit cluster (the global id baked as `cluster_id` into the CLAS
    // in both the static and animated builds) — indexes `clusters[]` directly. The
    // pipeline's `geometry_index` carries the baked value too, but `cluster_id`
    // (ClusterIDNV) is the canonical source and pairs with cluster-local
    // `primitive_index` for this cluster-referencing BLAS.
    @builtin(cluster_id) cluster_id: u32,
    // Triangle within the cluster (cluster-local, since the cluster comes from
    // ClusterIDNV).
    @builtin(primitive_index) primitive_index: u32,
    @builtin(world_ray_direction) ray_direction: vec3<f32>,
) {
    var rng = payload.rng;
    let barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    let ray_hit = resolve_triangle_data_full(instance_id, cluster_id, primitive_index, barycentrics);

    let wo = -ray_direction;
    // Bend the smooth shading normal into the view hemisphere so silhouette
    // edges (interpolated normal past 90°) don't black out.
    let world_normal = bend_shading_normal(ray_hit.world_normal, wo);
    let NdotV = max(dot(world_normal, wo), 0.0001);
    let F_ab = F_AB(ray_hit.material.perceptual_roughness, NdotV);

    // Emissive contribution, MIS-weighted against NEE on all but the primary ray.
    var mis_weight = 1.0;
    if payload.p_bounce != 0.0 {
        let p_light = random_emissive_light_pdf(ray_hit);
        mis_weight = power_heuristic(payload.p_bounce, p_light);
    }
    var emitted = mis_weight * ray_hit.material.emissive;

    // Next-event estimation (skip on mirror-like surfaces — a delta lobe can't be
    // importance-sampled by area light NEE).
    let is_perfectly_specular =
        ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && ray_hit.material.metallic > 0.9999;
    if !is_perfectly_specular {
        let direct_lighting = sample_random_light(ray_hit.world_position, world_normal, &rng);
        var nee_mis = 1.0;
        if direct_lighting.brdf_rays_can_hit {
            let pdf_of_bounce = brdf_pdf(wo, direct_lighting.wi, world_normal, ray_hit.material, F_ab);
            nee_mis = power_heuristic(1.0 / direct_lighting.inverse_pdf, pdf_of_bounce);
        }
        let direct_brdf = evaluate_brdf(wo, direct_lighting.wi, world_normal, ray_hit.material, F_ab);
        emitted += nee_mis * direct_lighting.radiance * direct_lighting.inverse_pdf * direct_brdf;
    }

    payload.emitted = emitted;

    // BRDF-sampled continuation ray for the next bounce.
    let next_bounce = evaluate_and_sample_brdf(wo, world_normal, ray_hit.material, F_ab, &rng);
    if next_bounce.pdf == 0.0 {
        payload.bounce = 0u; // dead path — terminate
        payload.rng = rng;
        return;
    }
    payload.attenuation = next_bounce.throughput;
    payload.next_origin = offset_ray_origin(ray_hit.world_position, ray_hit.geometric_world_normal);
    payload.next_direction = next_bounce.wi;
    payload.p_bounce = next_bounce.pdf;
    payload.bounce = 1u;
    payload.rng = rng;
}
