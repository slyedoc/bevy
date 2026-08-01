// Hair closest-hit: the fiber path on the RT pipeline. Structured like
// `chit_opaque`: next-event estimation with a `TraceRay` shadow ray to the
// dedicated `miss_shadow` program (fixed-function traversal, off the register
// file), and a BSDF-sampled continuation ray the RAYGEN loop drives. Distinct
// from the inline integrator in `hair_shade.wgsl` (which runs its own
// `rayQuery` bounce loop); only the Chiang fiber BSDF and `resolve_hair_hit`
// are shared.
//
// The DLSS ray-reconstruction guide writes are unconditional: the set-1
// layout always carries the G-buffer bindings and the WGSL twin compiled with
// `SOLARI_DLSS` always set.
//
// Regenerate SPIR-V (from crates/bevy_solari/src/render/rt_pipeline/):
//   slangc chit_hair.slang -target spirv -o chit_hair.spv

import rt_payload;
import scene_resolve;
import brdf;
import sampling;
import hair;

// Camera UBO (set 1 binding 1) + the DLSS RR guide G-buffer — same
// bindings/packing as chit_opaque. Written for the primary hair hit so RR has
// a real surface (depth/motion/normal + a NON-ZERO albedo) to demodulate
// against; without it, the sky default's zero albedo renders hair black.
[[vk::binding(1, 1)]] ConstantBuffer<RtCamera> camera;
[[vk::binding(5, 1)]] RWStructuredBuffer<float4> gbuffer_normal_roughness;
[[vk::binding(6, 1)]] RWStructuredBuffer<float4> gbuffer_diffuse;
[[vk::binding(7, 1)]] RWStructuredBuffer<float4> gbuffer_specular;
[[vk::binding(8, 1)]] RWStructuredBuffer<float4> gbuffer_motion;
static const uint NO_GBUFFER = 0xffffffff;

// SBT miss index of `miss_shadow` (miss 0 = primary, 1 = shadow). Same as chit_opaque.
static const uint SHADOW_MISS_INDEX = 1;
static const uint SHADOW_RAY_FLAGS =
    RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

// NEE visibility test: assume occluded; `miss_shadow` clears the flag iff the
// ray reaches the light unobstructed.
bool light_visible(float3 origin, float3 direction, float tmax)
{
    ShadowPayload shadow_payload;
    shadow_payload.anyhit_count = 0;
    shadow_payload.occluded = 1;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = RAY_T_MIN;
    ray.TMax = tmax;
    TraceRay(tlas, SHADOW_RAY_FLAGS, 0xff, 0, 0, SHADOW_MISS_INDEX, ray, shadow_payload);
    return shadow_payload.occluded == 0;
}

[shader("closesthit")]
void chit_hair(inout RtPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    var rng = payload.rng;
    // PTLAS instance id (== the `instance_id` baked in ptlas_hair_write); the
    // key `resolve_hair_hit` maps to a hair index via `- hair_params.base`.
    let instance_id = InstanceIndex();
    // LSS segment within the hair instance.
    let primitive_index = PrimitiveIndex();
    let ray_origin = WorldRayOrigin();
    let ray_direction = WorldRayDirection();
    let ray_t = RayTCurrent();

    // ── Opaque branch path (`SolariBranches`: bark/wood on the same LSS geometry) ──
    // When the instance carries a real material slot it's not fiber hair —
    // shade it as an opaque surface (round-cone normal + BRDF), structured
    // exactly like `chit_opaque`, then return before the fiber path below.
    let lss_material = lss_material_id(instance_id);
    if (lss_material != HAIR_MATERIAL_NONE)
    {
        let hit_position = ray_origin + ray_direction * ray_t;
        let surf = resolve_lss_surface(instance_id, primitive_index, hit_position);
        let material = resolve_material_lod(load_material_bindless(lss_material), float2(0.0), 0.0);
        let wo = -ray_direction;
        let world_normal = bend_shading_normal(surf.world_normal, wo);
        let NdotV = max(dot(world_normal, wo), 0.0001);
        let F_ab = F_AB(material.perceptual_roughness, NdotV);

        // Branches aren't registered emissive lights, so no NEE-MIS on emissive.
        var emitted = material.emissive;
        let is_specular =
            material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && material.metallic > 0.9999;
        if (!is_specular)
        {
            let sample = generate_random_light_sample(rng);
            if (sample.light_sample.light_id != NULL_LIGHT_ID)
            {
                let lc = calculate_resolved_light_contribution(
                    sample.resolved_light_sample, hit_position, world_normal);
                if (lc.inverse_pdf > 0.0)
                {
                    let shadow_origin = offset_ray_origin(hit_position, surf.world_normal);
                    let light_pos = sample.resolved_light_sample.world_position;
                    var shadow_dir = light_pos.xyz;
                    var shadow_tmax = RAY_T_MAX;
                    if (light_pos.w == 1.0)
                    {
                        let to_light = shadow_dir - shadow_origin;
                        let dist = length(to_light);
                        shadow_dir = to_light / dist;
                        shadow_tmax = dist - RAY_T_MIN;
                    }
                    var visible = false;
                    if (shadow_tmax >= RAY_T_MIN)
                    {
                        visible = light_visible(shadow_origin, shadow_dir, shadow_tmax);
                    }
                    if (visible)
                    {
                        var nee_mis = 1.0;
                        if (lc.brdf_rays_can_hit)
                        {
                            let pdf_b = brdf_pdf(wo, lc.wi, world_normal, material, F_ab);
                            nee_mis = power_heuristic(1.0 / lc.inverse_pdf, pdf_b);
                        }
                        let direct = evaluate_brdf(wo, lc.wi, world_normal, material, F_ab);
                        emitted += nee_mis * lc.radiance * lc.inverse_pdf * direct
                            * saturate(dot(world_normal, lc.wi));
                    }
                }
            }
        }
        payload.emitted = emitted;

        if (payload.gbuffer_pixel != NO_GBUFFER)
        {
            let px = payload.gbuffer_pixel;
            let cur_clip = mat4_mul(camera.clip_from_world, float4(hit_position, 1.0));
            if (cur_clip.w > 1.0e-4)
            {
                gbuffer_normal_roughness[px] = float4(world_normal, material.roughness);
                let view_pos = mat4_mul(camera.view_from_world, float4(hit_position, 1.0));
                let diffuse = material.base_color * (1.0 - material.metallic);
                gbuffer_diffuse[px] = float4(diffuse, max(-view_pos.z, 1.0e-4));
                let specular = lerp(float3(0.04), material.base_color, material.metallic);
                gbuffer_specular[px] = float4(specular, 0.0);
                let prev_clip = mat4_mul(camera.prev_clip_from_world, float4(hit_position, 1.0));
                var motion = float2(0.0);
                if (prev_clip.w > 1.0e-4)
                {
                    let cur_uv = (cur_clip.xy / cur_clip.w) * float2(0.5, -0.5);
                    let prev_uv = (prev_clip.xy / prev_clip.w) * float2(0.5, -0.5);
                    motion = cur_uv - prev_uv;
                }
                gbuffer_motion[px] = float4(motion, 0.0, 0.0);
            }
        }

        let next = evaluate_and_sample_brdf(wo, world_normal, material, F_ab, rng);
        if (next.pdf == 0.0)
        {
            payload.bounce = 0; // dead path — terminate
            payload.rng = rng;
            return;
        }
        payload.attenuation = next.throughput;
        payload.next_origin = offset_ray_origin(hit_position, surf.world_normal);
        payload.next_direction = next.wi;
        payload.p_bounce = next.pdf;
        payload.bounce = 1;
        payload.rng = rng;
        return;
    }

    let world_position = ray_origin + ray_direction * ray_t;
    let hair_hit = resolve_hair_hit(instance_id, primitive_index, world_position);
    let wo = -ray_direction;

    // Hair has no surface normal, only a fiber tangent. Derive a view-facing
    // fiber normal (wo with the along-fiber component removed) for the
    // self-intersection offset and the DLSS guide; fall back to up when
    // looking straight down a strand.
    var normal = wo - hair_hit.tangent * dot(wo, hair_hit.tangent);
    let normal_len = length(normal);
    normal = normal_len > 1.0e-4 ? normal / normal_len : float3(0.0, 1.0, 0.0);
    let surface_origin = offset_ray_origin(world_position, normal);

    // Next-event estimation: sample a light, test visibility with a shadow
    // ray, and weight with the fiber BSDF. (The light-contribution normal arg
    // is unused.)
    var emitted = float3(0.0);
    let sample = generate_random_light_sample(rng);
    if (sample.light_sample.light_id != NULL_LIGHT_ID)
    {
        let lc = calculate_resolved_light_contribution(
            sample.resolved_light_sample, world_position, normal);
        if (lc.inverse_pdf > 0.0)
        {
            let light_pos = sample.resolved_light_sample.world_position;
            var shadow_dir = light_pos.xyz;
            var shadow_tmax = RAY_T_MAX;
            if (light_pos.w == 1.0)
            {
                let to_light = shadow_dir - surface_origin;
                let dist = length(to_light);
                shadow_dir = to_light / dist;
                shadow_tmax = dist - RAY_T_MIN;
            }
            var visible = false;
            if (shadow_tmax >= RAY_T_MIN)
            {
                visible = light_visible(surface_origin, shadow_dir, shadow_tmax);
            }
            if (visible)
            {
                var nee_mis = 1.0;
                if (lc.brdf_rays_can_hit)
                {
                    let pdf_b = pdf_hair_bsdf(
                        wo, lc.wi, hair_hit.tangent,
                        hair_hit.beta_m, hair_hit.beta_n, hair_hit.alpha, hair_hit.ior);
                    nee_mis = power_heuristic(1.0 / lc.inverse_pdf, pdf_b);
                }
                let f = eval_hair_bsdf(
                    wo, lc.wi, hair_hit.tangent,
                    hair_hit.sigma_a, hair_hit.beta_m, hair_hit.beta_n,
                    hair_hit.alpha, hair_hit.ior);
                emitted += nee_mis * lc.radiance * lc.inverse_pdf * f;
            }
        }
    }

    payload.emitted = emitted;

    if (payload.gbuffer_pixel != NO_GBUFFER)
    {
        let px = payload.gbuffer_pixel;
        let cur_clip = mat4_mul(camera.clip_from_world, float4(world_position, 1.0));
        // Skip the guide if the fiber is at/behind the eye (clip.w → 0 would
        // emit a NaN motion vector and hang DLSS); leave the finite raygen sky
        // default.
        if (cur_clip.w > 1.0e-4)
        {
            // Hair is rough-looking → roughness 1 (denoise aggressively).
            gbuffer_normal_roughness[px] = float4(normal, 1.0);
            let view_pos = mat4_mul(camera.view_from_world, float4(world_position, 1.0));
            // Non-zero diffuse albedo so RR doesn't demodulate hair to black —
            // the fiber's residual transmission `exp(-sigma_a)` is a cheap proxy.
            let albedo = exp(-hair_hit.sigma_a);
            gbuffer_diffuse[px] = float4(albedo, max(-view_pos.z, 1.0e-4));
            gbuffer_specular[px] = float4(0.04, 0.04, 0.04, 0.0);
            // Camera-motion reprojection (hair's own per-frame motion isn't
            // tracked; the same world position through both projections
            // captures the camera).
            let prev_clip = mat4_mul(camera.prev_clip_from_world, float4(world_position, 1.0));
            var motion = float2(0.0);
            if (prev_clip.w > 1.0e-4)
            {
                let cur_uv = (cur_clip.xy / cur_clip.w) * float2(0.5, -0.5);
                let prev_uv = (prev_clip.xy / prev_clip.w) * float2(0.5, -0.5);
                motion = cur_uv - prev_uv;
            }
            gbuffer_motion[px] = float4(motion, 0.0, 0.0);
        }
    }

    // BSDF-sampled continuation ray for the next bounce (raygen drives the
    // loop + Russian roulette). The hair throughput already folds in the cosine.
    let next = sample_hair_bsdf(
        wo, hair_hit.tangent, hair_hit.sigma_a, hair_hit.beta_m, hair_hit.beta_n,
        hair_hit.alpha, hair_hit.ior, rng);
    if (next.pdf == 0.0)
    {
        payload.bounce = 0; // dead path — terminate
        payload.rng = rng;
        return;
    }
    payload.attenuation = next.throughput;
    payload.next_origin = surface_origin;
    payload.next_direction = next.wi;
    payload.p_bounce = next.pdf;
    payload.bounce = 1;
    payload.rng = rng;
}
