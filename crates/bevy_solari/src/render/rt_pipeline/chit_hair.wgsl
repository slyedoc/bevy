// Hair closest-hit — the rt_pipeline-NATIVE fiber path. Structured exactly like
// `chit_opaque`: next-event estimation with a `traceRay` shadow ray to the
// dedicated `miss_shadow` program (fixed-function traversal, off the register
// file), and a BSDF-sampled continuation ray the RAYGEN loop drives. It is NOT
// the inline integrator in `hair_shade.wgsl` (which runs its own `rayQuery` bounce
// loop) — keeping hair on the pipeline path is the whole point, and lets the
// inline `wgpu_ray_query` machinery be retired. Only the pure Chiang fiber BSDF
// (`bevy_solari::hair`) and `resolve_hair_hit` are reused (both rayQuery-free).
enable wgpu_ray_tracing_pipeline;
enable primitive_index;

#import bevy_solari::rt_payload::{RtPayload, ShadowPayload, RtCamera}
#import bevy_solari::hair::{eval_hair_bsdf, sample_hair_bsdf, pdf_hair_bsdf}
#import bevy_solari::brdf::{evaluate_brdf, evaluate_and_sample_brdf, brdf_pdf, F_AB, bend_shading_normal}
#import bevy_solari::sampling::{generate_random_light_sample, calculate_resolved_light_contribution, power_heuristic, NULL_LIGHT_ID}
#import bevy_solari::scene_bindings::{resolve_hair_hit, lss_material_id, resolve_lss_surface, load_material_bindless, resolve_material_lod, HAIR_MATERIAL_NONE, MIRROR_ROUGHNESS_THRESHOLD, offset_ray_origin, tlas, RAY_T_MIN, RAY_T_MAX}

var<incoming_ray_payload> payload: RtPayload;
// Outgoing payload for the NEE shadow ray (see `miss_shadow`).
var<ray_payload> shadow_payload: ShadowPayload;

#ifdef SOLARI_DLSS
// DLSS Ray Reconstruction guide G-buffer (set 1) — same bindings/packing as
// chit_opaque. Written for the primary hair hit so RR has a real surface
// (depth/motion/normal + a NON-ZERO albedo) to demodulate against; without it,
// the sky default's zero albedo makes RR render hair black.
@group(1) @binding(1) var<uniform> camera: RtCamera;
@group(1) @binding(5) var<storage, read_write> gbuffer_normal_roughness: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> gbuffer_diffuse: array<vec4<f32>>;
@group(1) @binding(7) var<storage, read_write> gbuffer_specular: array<vec4<f32>>;
@group(1) @binding(8) var<storage, read_write> gbuffer_motion: array<vec4<f32>>;
const NO_GBUFFER: u32 = 0xffffffffu;
#endif

// SBT miss index of `miss_shadow` (miss 0 = primary, 1 = shadow). Same as chit_opaque.
const SHADOW_MISS_INDEX: u32 = 1u;
const SHADOW_RAY_FLAGS: u32 =
    RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

@closest_hit
@incoming_payload(payload)
fn chit_hair(
    // PTLAS instance id (== the `instance_id` baked in ptlas_hair_write); the key
    // `resolve_hair_hit` maps to a hair index via `- hair_params.base`.
    @builtin(instance_id) instance_id: u32,
    // LSS segment within the hair instance.
    @builtin(primitive_index) primitive_index: u32,
    @builtin(world_ray_origin) ray_origin: vec3<f32>,
    @builtin(world_ray_direction) ray_direction: vec3<f32>,
    @builtin(ray_t_current_max) ray_t: f32,
) {
    var rng = payload.rng;

    // ── Opaque branch path (`SolariBranches`: bark/wood on the same LSS geometry) ──
    // When the instance carries a real material slot it's not fiber hair — shade it
    // as an opaque surface (round-cone normal + BRDF), structured exactly like
    // `chit_opaque`, then return before the fiber path below.
    let lss_material = lss_material_id(instance_id);
    if lss_material != HAIR_MATERIAL_NONE {
        let hit_position = ray_origin + ray_direction * ray_t;
        let surf = resolve_lss_surface(instance_id, primitive_index, hit_position);
        let material = resolve_material_lod(load_material_bindless(lss_material), vec2<f32>(0.0), 0.0);
        let wo = -ray_direction;
        let world_normal = bend_shading_normal(surf.world_normal, wo);
        let NdotV = max(dot(world_normal, wo), 0.0001);
        let F_ab = F_AB(material.perceptual_roughness, NdotV);

        // Branches aren't registered emissive lights, so no NEE-MIS on emissive.
        var emitted = material.emissive;
        let is_specular =
            material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && material.metallic > 0.9999;
        if !is_specular {
            let sample = generate_random_light_sample(&rng);
            if sample.light_sample.light_id != NULL_LIGHT_ID {
                let lc = calculate_resolved_light_contribution(
                    sample.resolved_light_sample, hit_position, world_normal);
                if lc.inverse_pdf > 0.0 {
                    let shadow_origin = offset_ray_origin(hit_position, surf.world_normal);
                    let light_pos = sample.resolved_light_sample.world_position;
                    var shadow_dir = light_pos.xyz;
                    var shadow_tmax = RAY_T_MAX;
                    if light_pos.w == 1.0 {
                        let to_light = shadow_dir - shadow_origin;
                        let dist = length(to_light);
                        shadow_dir = to_light / dist;
                        shadow_tmax = dist - RAY_T_MIN;
                    }
                    var visible = false;
                    if shadow_tmax >= RAY_T_MIN {
                        shadow_payload.occluded = 1u;
                        traceRay(
                            tlas,
                            RayDesc(SHADOW_RAY_FLAGS, 0xffu, RAY_T_MIN, shadow_tmax, shadow_origin, shadow_dir),
                            0u, 0u, SHADOW_MISS_INDEX, &shadow_payload);
                        visible = shadow_payload.occluded == 0u;
                    }
                    if visible {
                        var nee_mis = 1.0;
                        if lc.brdf_rays_can_hit {
                            let pdf_b = brdf_pdf(wo, lc.wi, world_normal, material, F_ab);
                            nee_mis = power_heuristic(1.0 / lc.inverse_pdf, pdf_b);
                        }
                        let direct = evaluate_brdf(wo, lc.wi, world_normal, material, F_ab);
                        emitted += nee_mis * lc.radiance * lc.inverse_pdf * direct * saturate(dot(world_normal, lc.wi));
                    }
                }
            }
        }
        payload.emitted = emitted;

#ifdef SOLARI_DLSS
        if payload.gbuffer_pixel != NO_GBUFFER {
            let px = payload.gbuffer_pixel;
            let cur_clip = camera.clip_from_world * vec4<f32>(hit_position, 1.0);
            if cur_clip.w > 1.0e-4 {
                gbuffer_normal_roughness[px] = vec4<f32>(world_normal, material.roughness);
                let view_pos = camera.view_from_world * vec4<f32>(hit_position, 1.0);
                let diffuse = material.base_color * (1.0 - material.metallic);
                gbuffer_diffuse[px] = vec4<f32>(diffuse, max(-view_pos.z, 1.0e-4));
                let specular = mix(vec3<f32>(0.04), material.base_color, material.metallic);
                gbuffer_specular[px] = vec4<f32>(specular, 0.0);
                let prev_clip = camera.prev_clip_from_world * vec4<f32>(hit_position, 1.0);
                var motion = vec2<f32>(0.0);
                if prev_clip.w > 1.0e-4 {
                    let cur_uv = (cur_clip.xy / cur_clip.w) * vec2<f32>(0.5, -0.5);
                    let prev_uv = (prev_clip.xy / prev_clip.w) * vec2<f32>(0.5, -0.5);
                    motion = cur_uv - prev_uv;
                }
                gbuffer_motion[px] = vec4<f32>(motion, 0.0, 0.0);
            }
        }
#endif

        let next = evaluate_and_sample_brdf(wo, world_normal, material, F_ab, &rng);
        if next.pdf == 0.0 {
            payload.bounce = 0u; // dead path — terminate
            payload.rng = rng;
            return;
        }
        payload.attenuation = next.throughput;
        payload.next_origin = offset_ray_origin(hit_position, surf.world_normal);
        payload.next_direction = next.wi;
        payload.p_bounce = next.pdf;
        payload.bounce = 1u;
        payload.rng = rng;
        return;
    }

    let world_position = ray_origin + ray_direction * ray_t;
    let hair = resolve_hair_hit(instance_id, primitive_index, world_position);
    let wo = -ray_direction;

    // Hair has no surface normal, only a fiber tangent. Derive a view-facing fiber
    // normal (wo with the along-fiber component removed) for the self-intersection
    // offset and the DLSS guide; fall back to up when looking straight down a strand.
    var normal = wo - hair.tangent * dot(wo, hair.tangent);
    let normal_len = length(normal);
    normal = select(vec3<f32>(0.0, 1.0, 0.0), normal / normal_len, normal_len > 1.0e-4);
    let surface_origin = offset_ray_origin(world_position, normal);

    // Next-event estimation: sample a light, test visibility with a shadow ray, and
    // weight with the fiber BSDF. (The light-contribution normal arg is unused.)
    var emitted = vec3<f32>(0.0);
    let sample = generate_random_light_sample(&rng);
    if sample.light_sample.light_id != NULL_LIGHT_ID {
        let lc = calculate_resolved_light_contribution(
            sample.resolved_light_sample,
            world_position,
            normal,
        );
        if lc.inverse_pdf > 0.0 {
            let light_pos = sample.resolved_light_sample.world_position;
            var shadow_dir = light_pos.xyz;
            var shadow_tmax = RAY_T_MAX;
            if light_pos.w == 1.0 {
                let to_light = shadow_dir - surface_origin;
                let dist = length(to_light);
                shadow_dir = to_light / dist;
                shadow_tmax = dist - RAY_T_MIN;
            }
            var visible = false;
            if shadow_tmax >= RAY_T_MIN {
                shadow_payload.occluded = 1u;
                traceRay(
                    tlas,
                    RayDesc(SHADOW_RAY_FLAGS, 0xffu, RAY_T_MIN, shadow_tmax, surface_origin, shadow_dir),
                    0u,
                    0u,
                    SHADOW_MISS_INDEX,
                    &shadow_payload,
                );
                visible = shadow_payload.occluded == 0u;
            }
            if visible {
                var nee_mis = 1.0;
                if lc.brdf_rays_can_hit {
                    let pdf_b = pdf_hair_bsdf(
                        wo, lc.wi, hair.tangent, hair.beta_m, hair.beta_n, hair.alpha, hair.ior);
                    nee_mis = power_heuristic(1.0 / lc.inverse_pdf, pdf_b);
                }
                let f = eval_hair_bsdf(
                    wo, lc.wi, hair.tangent,
                    hair.sigma_a, hair.beta_m, hair.beta_n, hair.alpha, hair.ior);
                emitted += nee_mis * lc.radiance * lc.inverse_pdf * f;
            }
        }
    }

    payload.emitted = emitted;

#ifdef SOLARI_DLSS
    if payload.gbuffer_pixel != NO_GBUFFER {
        let px = payload.gbuffer_pixel;
        let cur_clip = camera.clip_from_world * vec4<f32>(world_position, 1.0);
        // Skip the guide if the fiber is at/behind the eye (clip.w -> 0 would emit a
        // NaN motion vector and hang DLSS); leave the finite raygen sky default.
        if cur_clip.w > 1.0e-4 {
            // Hair is rough-looking → roughness 1 (denoise aggressively).
            gbuffer_normal_roughness[px] = vec4<f32>(normal, 1.0);
            let view_pos = camera.view_from_world * vec4<f32>(world_position, 1.0);
            // Non-zero diffuse albedo so RR doesn't demodulate hair to black —
            // the fiber's residual transmission `exp(-sigma_a)` is a cheap proxy.
            let albedo = exp(-hair.sigma_a);
            gbuffer_diffuse[px] = vec4<f32>(albedo, max(-view_pos.z, 1.0e-4));
            gbuffer_specular[px] = vec4<f32>(0.04, 0.04, 0.04, 0.0);
            // Camera-motion reprojection (hair's own per-frame motion isn't tracked;
            // the same world position through both projections captures the camera).
            let prev_clip = camera.prev_clip_from_world * vec4<f32>(world_position, 1.0);
            var motion = vec2<f32>(0.0);
            if prev_clip.w > 1.0e-4 {
                let cur_uv = (cur_clip.xy / cur_clip.w) * vec2<f32>(0.5, -0.5);
                let prev_uv = (prev_clip.xy / prev_clip.w) * vec2<f32>(0.5, -0.5);
                motion = cur_uv - prev_uv;
            }
            gbuffer_motion[px] = vec4<f32>(motion, 0.0, 0.0);
        }
    }
#endif

    // BSDF-sampled continuation ray for the next bounce (raygen drives the loop +
    // Russian roulette). The hair throughput already folds in the cosine.
    let next = sample_hair_bsdf(
        wo, hair.tangent, hair.sigma_a, hair.beta_m, hair.beta_n, hair.alpha, hair.ior, &rng);
    if next.pdf == 0.0 {
        payload.bounce = 0u; // dead path — terminate
        payload.rng = rng;
        return;
    }
    payload.attenuation = next.throughput;
    payload.next_origin = surface_origin;
    payload.next_direction = next.wi;
    payload.p_bounce = next.pdf;
    payload.bounce = 1u;
    payload.rng = rng;
}
