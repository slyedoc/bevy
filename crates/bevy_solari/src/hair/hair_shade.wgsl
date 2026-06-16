enable wgpu_ray_query;

// Scene-integrated hair shading, shared by the pathtracer and the realtime
// (ReSTIR) visibility pass so the fiber lighting lives in one place. The pure
// BSDF math is `bevy_solari::hair`; this layer adds light sampling + traversal.
#define_import_path bevy_solari::hair_shade

#import bevy_solari::scene_bindings::{trace_ray, resolve_hair_hit, is_hair_instance, ResolvedHairHit, RAY_T_MIN, RAY_T_MAX}
#import bevy_solari::hair::{eval_hair_bsdf, sample_hair_bsdf, pdf_hair_bsdf}
#import bevy_solari::sampling::{sample_random_light, power_heuristic}

/// One next-event-estimation sample on a hair hit (fiber BSDF + MIS). Returns
/// radiance per unit path throughput; the caller multiplies by its running
/// throughput. Used by both the pathtracer's hair branch and `shade_hair_path`.
fn hair_direct_lighting(
    world_position: vec3<f32>,
    wo: vec3<f32>,
    hair: ResolvedHairHit,
    rng: ptr<function, u32>,
) -> vec3<f32> {
    let direct = sample_random_light(world_position, hair.tangent, rng);
    var mis_weight = 1.0;
    if direct.brdf_rays_can_hit {
        let pdf_b = pdf_hair_bsdf(wo, direct.wi, hair.tangent, hair.beta_m, hair.beta_n, hair.alpha, hair.ior);
        mis_weight = power_heuristic(1.0 / direct.inverse_pdf, pdf_b);
    }
    let f = eval_hair_bsdf(wo, direct.wi, hair.tangent, hair.sigma_a, hair.beta_m, hair.beta_n, hair.alpha, hair.ior);
    return mis_weight * direct.radiance * direct.inverse_pdf * f;
}

/// Self-contained hair path trace for the realtime path (where opaque geometry
/// isn't path-traced): NEE at each vertex plus a few strand-strand bounces.
/// Terminates on a miss or on an opaque hit (whose lighting the deferred path
/// owns). Returns raw radiance — the caller applies exposure.
// The first hit is passed as scalars (`first_t` / `first_instance` /
// `first_primitive`), NOT a `RayIntersection`: naga_oil doesn't unify the builtin
// `RayIntersection` type across module boundaries, so it can't be a cross-module
// argument. (Each `trace_ray` inside this module returns its own local one.)
fn shade_hair_path(
    ray_origin_in: vec3<f32>,
    ray_direction_in: vec3<f32>,
    first_t: f32,
    first_instance: u32,
    first_primitive: u32,
    max_bounces: u32,
    rng: ptr<function, u32>,
) -> vec3<f32> {
    var radiance = vec3(0.0);
    var throughput = vec3(1.0);
    var ray_direction = ray_direction_in;
    // Carry the hit across bounces as scalars (each `trace_ray` stays a fresh `let`).
    var world_position = ray_origin_in + ray_direction_in * first_t;
    var instance_index = first_instance;
    var primitive_index = first_primitive;

    for (var b = 0u; b < max_bounces; b += 1u) {
        let hair = resolve_hair_hit(instance_index, primitive_index, world_position);
        let wo = -ray_direction;
        radiance += throughput * hair_direct_lighting(world_position, wo, hair, rng);

        let next = sample_hair_bsdf(wo, hair.tangent, hair.sigma_a, hair.beta_m, hair.beta_n, hair.alpha, hair.ior, rng);
        if next.pdf == 0.0 { break; }
        throughput *= next.throughput;
        let ray_origin = world_position + next.wi * 1e-3;
        ray_direction = next.wi;
        let hit = trace_ray(ray_origin, ray_direction, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
        if hit.kind == RAY_QUERY_INTERSECTION_NONE { break; }
        if !is_hair_instance(hit.instance_index) { break; }
        world_position = ray_origin + ray_direction * hit.t;
        instance_index = hit.instance_index;
        primitive_index = hit.primitive_index;
    }
    return radiance;
}
