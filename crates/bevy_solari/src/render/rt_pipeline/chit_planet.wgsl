// Planet-surface closest-hit: identical shading to `chit_opaque` (emissive + NEE +
// BRDF-sampled continuation), except the albedo comes from the mesh's per-vertex
// `vertex_custom` (a baked biome color), not the material's flat base_color. This is
// the first consumer of the generic custom-vertex-attribute + custom-chit extension
// seam; everything past the base_color override matches the opaque program.
enable wgpu_ray_tracing_pipeline;
enable primitive_index;

#import bevy_solari::rt_payload::{RtPayload, ShadowPayload, RtCamera}
#import bevy_solari::brdf::{evaluate_brdf, evaluate_and_sample_brdf, brdf_pdf, F_AB, bend_shading_normal}
#import bevy_solari::sampling::{generate_random_light_sample, calculate_resolved_light_contribution, random_emissive_light_pdf, power_heuristic, NULL_LIGHT_ID}
#import bevy_solari::scene_bindings::{resolve_triangle_data_full_mat_fetch, load_triangle_custom, offset_ray_origin, tlas, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD, load_material_bindless, sample_texture_lod, TEXTURE_MAP_NONE}

var<incoming_ray_payload> payload: RtPayload;
var<ray_payload> shadow_payload: ShadowPayload;
var<hit_attribute> bary: vec2<f32>;

#ifdef SOLARI_DLSS
@group(1) @binding(1) var<uniform> camera: RtCamera;
@group(1) @binding(5) var<storage, read_write> gbuffer_normal_roughness: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> gbuffer_diffuse: array<vec4<f32>>;
@group(1) @binding(7) var<storage, read_write> gbuffer_specular: array<vec4<f32>>;
@group(1) @binding(8) var<storage, read_write> gbuffer_motion: array<vec4<f32>>;
const NO_GBUFFER: u32 = 0xffffffffu;
#endif

const SHADOW_MISS_INDEX: u32 = 1u;
const SHADOW_RAY_FLAGS: u32 =
    RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

struct SbtRecord {
    material_id: u32,
}
var<shader_record> sbt: SbtRecord;

// Unpack the 8-bit rgb from a packed `vertex_custom` u32 (`id<<24 | r<<16 | g<<8 | b`).
fn unpack_custom_rgb(packed: u32) -> vec3<f32> {
    return vec3<f32>(
        f32((packed >> 16u) & 0xffu),
        f32((packed >> 8u) & 0xffu),
        f32(packed & 0xffu),
    ) * (1.0 / 255.0);
}

@closest_hit
@incoming_payload(payload)
fn chit_planet(
    @builtin(instance_id) instance_id: u32,
    @builtin(cluster_id) cluster_id: u32,
    @builtin(primitive_index) primitive_index: u32,
    @builtin(world_ray_direction) ray_direction: vec3<f32>,
    @builtin(object_to_world) object_to_world: mat4x3<f32>,
    @builtin(hit_triangle_vertex_positions) hit_positions: array<vec3<f32>, 3>,
) {
    var rng = payload.rng;
    payload.hit_cluster = cluster_id;
    payload.hit_primitive = primitive_index;
    let barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    let transform = mat3x4<f32>(
        vec4<f32>(object_to_world[0].x, object_to_world[1].x, object_to_world[2].x, object_to_world[3].x),
        vec4<f32>(object_to_world[0].y, object_to_world[1].y, object_to_world[2].y, object_to_world[3].y),
        vec4<f32>(object_to_world[0].z, object_to_world[1].z, object_to_world[2].z, object_to_world[3].z),
    );
    var ray_hit = resolve_triangle_data_full_mat_fetch(instance_id, sbt.material_id, transform, cluster_id, primitive_index, barycentrics, hit_positions);

    // Biome albedo: barycentric-blend the three vertices' baked colors and override
    // the flat material base_color. (This is the only departure from chit_opaque.)
    let custom = load_triangle_custom(cluster_id, primitive_index);
    let biome_albedo = unpack_custom_rgb(custom.x) * barycentrics.x
        + unpack_custom_rgb(custom.y) * barycentrics.y
        + unpack_custom_rgb(custom.z) * barycentrics.z;
    ray_hit.material.base_color = biome_albedo;

    let wo = -ray_direction;
    let world_normal = bend_shading_normal(ray_hit.world_normal, wo);
    let NdotV = max(dot(world_normal, wo), 0.0001);
    let F_ab = F_AB(ray_hit.material.perceptual_roughness, NdotV);

    var mis_weight = 1.0;
    if payload.p_bounce != 0.0 {
        let p_light = random_emissive_light_pdf(ray_hit);
        mis_weight = power_heuristic(payload.p_bounce, p_light);
    }
    var emitted = mis_weight * ray_hit.material.emissive;

    let is_perfectly_specular =
        ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && ray_hit.material.metallic > 0.9999;
    if !is_perfectly_specular {
        let sample = generate_random_light_sample(&rng);
        if sample.light_sample.light_id != NULL_LIGHT_ID {
            let lc = calculate_resolved_light_contribution(
                sample.resolved_light_sample,
                ray_hit.world_position,
                world_normal,
            );
            if lc.inverse_pdf > 0.0 {
                let shadow_origin =
                    offset_ray_origin(ray_hit.world_position, ray_hit.geometric_world_normal);
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
                        let pdf_of_bounce = brdf_pdf(wo, lc.wi, world_normal, ray_hit.material, F_ab);
                        nee_mis = power_heuristic(1.0 / lc.inverse_pdf, pdf_of_bounce);
                    }
                    let direct_brdf = evaluate_brdf(wo, lc.wi, world_normal, ray_hit.material, F_ab);
                    emitted += nee_mis * lc.radiance * lc.inverse_pdf * direct_brdf;
                }
            }
        }
    }

    payload.emitted = emitted;

#ifdef SOLARI_DLSS
    if payload.gbuffer_pixel != NO_GBUFFER {
        let px = payload.gbuffer_pixel;
        let cur_clip = camera.clip_from_world * vec4<f32>(ray_hit.world_position, 1.0);
        if cur_clip.w > 1.0e-4 {
            gbuffer_normal_roughness[px] = vec4<f32>(world_normal, ray_hit.material.roughness);
            let view_pos = camera.view_from_world * vec4<f32>(ray_hit.world_position, 1.0);
            let linear_depth = max(-view_pos.z, 1.0e-4);
            let diffuse = ray_hit.material.base_color * (1.0 - ray_hit.material.metallic);
            gbuffer_diffuse[px] = vec4<f32>(diffuse, linear_depth);
            let specular = mix(vec3<f32>(0.04), ray_hit.material.base_color, ray_hit.material.metallic);
            gbuffer_specular[px] = vec4<f32>(specular, 0.0);
            let prev_clip =
                camera.prev_clip_from_world * vec4<f32>(ray_hit.previous_frame_world_position, 1.0);
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

    let next_bounce = evaluate_and_sample_brdf(wo, world_normal, ray_hit.material, F_ab, &rng);
    if next_bounce.pdf == 0.0 {
        payload.bounce = 0u;
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
