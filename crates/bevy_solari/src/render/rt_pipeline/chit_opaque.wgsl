// Opaque-surface closest-hit: emissive (MIS-weighted) + next-event estimation +
// a BRDF-sampled continuation ray. Direct illumination follows the NVIDIA-canonical
// structure: NEE samples a light and tests visibility with a `traceRay` shadow ray
// (SKIP_CLOSEST_HIT + TERMINATE_ON_FIRST_HIT, routed to the dedicated `miss_shadow`
// program via an explicit miss index). The fixed-function traversal keeps the
// shadow query OFF the chit's register file — unlike an inline `rayQuery`, which is
// what previously blew this shader's occupancy. The raygen driver owns the bounce
// loop + throughput; this shader fills the payload with the radiance at this vertex
// and the next ray to trace.
enable wgpu_ray_tracing_pipeline;
enable primitive_index;

#import bevy_solari::rt_payload::{RtPayload, ShadowPayload, RtCamera}
#import bevy_solari::brdf::{evaluate_brdf, evaluate_and_sample_brdf, brdf_pdf, F_AB, bend_shading_normal}
#import bevy_solari::sampling::{generate_random_light_sample, calculate_resolved_light_contribution, random_emissive_light_pdf, power_heuristic, NULL_LIGHT_ID}
#import bevy_solari::scene_bindings::{resolve_triangle_data_full_mat_fetch, offset_ray_origin, tlas, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD, load_material_bindless, sample_texture_lod, TEXTURE_MAP_NONE}

var<incoming_ray_payload> payload: RtPayload;
// Outgoing payload for the NEE shadow ray (see `miss_shadow`).
var<ray_payload> shadow_payload: ShadowPayload;
// Driver-provided triangle barycentrics (GLSL `hitAttributeEXT vec2`) — the
// fixed-function triangle intersection writes (u, v); w = 1 - u - v.
var<hit_attribute> bary: vec2<f32>;

#ifdef SOLARI_DLSS
// DLSS Ray Reconstruction guide G-buffer (set 1) — written for the primary hit only
// (chit-direct), indexed by the launch pixel the raygen threads through the payload.
// Same bindings/packing as raygen's declarations; the camera supplies the view +
// motion matrices for the depth and motion-vector guides.
@group(1) @binding(1) var<uniform> camera: RtCamera;
@group(1) @binding(5) var<storage, read_write> gbuffer_normal_roughness: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> gbuffer_diffuse: array<vec4<f32>>;
@group(1) @binding(7) var<storage, read_write> gbuffer_specular: array<vec4<f32>>;
@group(1) @binding(8) var<storage, read_write> gbuffer_motion: array<vec4<f32>>;
const NO_GBUFFER: u32 = 0xffffffffu;
#endif

// SBT miss index of `miss_shadow` (miss 0 = miss_primary, miss 1 = miss_shadow).
const SHADOW_MISS_INDEX: u32 = 1u;
// Shadow rays skip the closest-hit (we only need occlusion) and stop at the first
// hit. NOT force-opaque: alpha-masked geometry (foliage) runs the alpha-cutout
// any-hit on the opaque hit group, so light passes through the holes — cutout
// shadows. Opaque geometry still commits in hardware via its OPAQUE flag, so only
// foliage shadow-ray hits pay the alpha test (parity with the inline rayQuery path).
const SHADOW_RAY_FLAGS: u32 =
    RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

// Per-material SBT shader record: each hit record bakes its material id (the
// record index = the material slot, set as `instance_contribution_to_hit_group_index`
// in ptlas_fill). Reading material identity from here — rather than the
// `material_ids[instance_id]` indirection — is the canonical NVIDIA path: the
// value is uniform per shader-record, so after SER reorders the warp by hit
// (the hit object carries this same SBT record index), `materials[material_id]`
// and the texture-array fetches become uniform, sidestepping non-uniform
// descriptor divergence.
struct SbtRecord {
    material_id: u32,
}
var<shader_record> sbt: SbtRecord;

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
    // The instance's object→world transform straight from the TLAS hit — no need
    // to read `transforms[instance_id]` (the acceleration structure already holds
    // it). `ObjectToWorldKHR` is mat4x3 (4 columns × 3 rows); convert to the
    // resolve's row-form mat3x4 below.
    @builtin(object_to_world) object_to_world: mat4x3<f32>,
    // The hit triangle's three object-space vertex positions, read straight from the
    // CLAS (VK_KHR_ray_tracing_position_fetch) — the resolve uses these instead of
    // re-fetching positions from the vertex pool. Requires the CLAS built with
    // `ALLOW_DATA_ACCESS` (see `clas_arena`).
    @builtin(hit_triangle_vertex_positions) hit_positions: array<vec3<f32>, 3>,
) {
    var rng = payload.rng;
    // Geometry-debug views (cluster / triangle color): record this hit's global
    // cluster id + cluster-local triangle so raygen can hash them to a flat color on
    // the primary hit. Cheap; ignored unless `camera.frame.z` selects those views.
    payload.hit_cluster = cluster_id;
    payload.hit_primitive = primitive_index;
    let barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    // Row-form affine (m[r] = (basis_row_r, translation_r)) the resolve expects.
    let transform = mat3x4<f32>(
        vec4<f32>(object_to_world[0].x, object_to_world[1].x, object_to_world[2].x, object_to_world[3].x),
        vec4<f32>(object_to_world[0].y, object_to_world[1].y, object_to_world[2].y, object_to_world[3].y),
        vec4<f32>(object_to_world[0].z, object_to_world[1].z, object_to_world[2].z, object_to_world[3].z),
    );
    let ray_hit = resolve_triangle_data_full_mat_fetch(instance_id, sbt.material_id, transform, cluster_id, primitive_index, barycentrics, hit_positions);

    // Displacement debug view (`frame.w == 1`): replace shading with the surface's height map in
    // grayscale, validating the displacement wiring (which map → which surface, the UVs, the sign)
    // BEFORE tessellation actually moves geometry. Surfaces with no displacement map read dim grey
    // for context. Terminate the path and divide by exposure so the raygen's `radiance *= exposure`
    // cancels — the pixel shows the raw `[0,1]` height.
    if camera.frame.w == 1u {
        let dmat = load_material_bindless(ray_hit.material_id);
        var height = 0.02;
        if dmat.displacement_texture_id != TEXTURE_MAP_NONE {
            height = sample_texture_lod(dmat.displacement_texture_id, ray_hit.uv, 0.0).r;
        }
        payload.emitted = vec3<f32>(height) / max(camera.camera_position.w, 1e-6);
        payload.bounce = 0u;
        payload.rng = rng;
        return;
    }

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
    // importance-sampled by area-light NEE).
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
                // Build the shadow ray toward the sampled light (positional w==1 →
                // finite range to the light; directional w==0 → far miss).
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
                    // Assume occluded; `miss_shadow` clears this iff the ray reaches
                    // the light. Fixed-function traversal → no register cost here.
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
    // Primary-hit ray-reconstruction guide (chit-direct): only the primary bounce
    // carries a real pixel index; secondary bounces pass NO_GBUFFER and skip this.
    // F0 = 0.04 for dielectrics, base_color for metals; diffuse is the
    // energy-conserving complement of the metallic split. The `.w` slots carry
    // linear depth (diffuse) and specular hit distance (specular — defaulted 0
    // here, filled by raygen after the first continuation ray).
    if payload.gbuffer_pixel != NO_GBUFFER {
        let px = payload.gbuffer_pixel;
        let cur_clip = camera.clip_from_world * vec4<f32>(ray_hit.world_position, 1.0);
        // A hit at/behind the eye (camera clipping into a surface) drives clip.w → 0;
        // the perspective divides below would emit NaN/Inf into the guide and hang
        // DLSS. Skip the guide for such a pixel, leaving the raygen sky default.
        if cur_clip.w > 1.0e-4 {
            gbuffer_normal_roughness[px] = vec4<f32>(world_normal, ray_hit.material.roughness);
            // View-space linear depth, positive into the scene (RR `DepthMode::Linear`).
            let view_pos = camera.view_from_world * vec4<f32>(ray_hit.world_position, 1.0);
            let linear_depth = max(-view_pos.z, 1.0e-4);
            let diffuse = ray_hit.material.base_color * (1.0 - ray_hit.material.metallic);
            gbuffer_diffuse[px] = vec4<f32>(diffuse, linear_depth);
            let specular = mix(vec3<f32>(0.04), ray_hit.material.base_color, ray_hit.material.metallic);
            // `.w` = specular hit distance; raygen overwrites it after bounce 1, so 0
            // here just defaults surfaces whose continuation ray hits nothing.
            gbuffer_specular[px] = vec4<f32>(specular, 0.0);
            // Screen-space motion vector: current vs previous UNJITTERED clip position,
            // UV space with y flipped. previous_frame_world_position handles moving
            // instances (parent/skin), the matrices handle the camera. Guard last
            // frame's divide too; zero motion if the surface was at the eye then.
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
