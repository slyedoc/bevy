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
#import bevy_solari::pbr::{rand_f, rand_u}
#import bevy_solari::sampling::{generate_random_light_sample, generate_random_emissive_light_sample, calculate_resolved_light_contribution, random_emissive_light_pdf, random_emissive_light_pdf_flux, resolve_emissive_for_restir, resolve_light_sample, emissive_light_count, directional_light_count, power_heuristic, pick_luminance, LightSample, Reservoir, SurfaceGbuf, StoredLight, pack_stored_light, ResolvedLightSample, NULL_LIGHT_ID}
#import bevy_solari::scene_bindings::{resolve_triangle_data_full_mat_fetch, offset_ray_origin, tlas, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD, load_material_bindless, sample_texture_lod, TEXTURE_MAP_NONE, light_sources, active_light_list}
#import bevy_render::utils::{octahedral_encode, octahedral_decode_signed}

var<incoming_ray_payload> payload: RtPayload;
// Outgoing payload for the NEE shadow ray (see `miss_shadow`).
var<ray_payload> shadow_payload: ShadowPayload;
// Driver-provided triangle barycentrics (GLSL `hitAttributeEXT vec2`) — the
// fixed-function triangle intersection writes (u, v); w = 1 - u - v.
var<hit_attribute> bary: vec2<f32>;

// Camera UBO (set 1, binding 1) — always present in the layout. Used by the
// displacement-debug view (`camera.frame.w`) regardless of DLSS, and by the DLSS
// guide writes below; declared unconditionally so a non-DLSS build still compiles.
@group(1) @binding(1) var<uniform> camera: RtCamera;
#ifdef SOLARI_DLSS
// DLSS Ray Reconstruction guide G-buffer (set 1) — written for the primary hit only
// (chit-direct), indexed by the launch pixel the raygen threads through the payload.
// Same bindings/packing as raygen's declarations; the camera supplies the view +
// motion matrices for the depth and motion-vector guides.
@group(1) @binding(5) var<storage, read_write> gbuffer_normal_roughness: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> gbuffer_diffuse: array<vec4<f32>>;
@group(1) @binding(7) var<storage, read_write> gbuffer_specular: array<vec4<f32>>;
@group(1) @binding(8) var<storage, read_write> gbuffer_motion: array<vec4<f32>>;
#endif
// ReSTIR DI reservoirs (rung 3): 2 interleaved slots per pixel (see `Reservoir`).
// The primary hit merges last frame's slot temporally and writes this frame's.
@group(1) @binding(9) var<storage, read_write> reservoirs: array<Reservoir>;
// Primary-hit surface attributes for the spatial merge+shade pass (flag bit 3):
// with spatial on, the chit stores the post-temporal reservoir + this surface
// and SKIPS the emissive winner's shadow ray — the pass owns merge+shade.
@group(1) @binding(10) var<storage, read_write> surfaces: array<SurfaceGbuf>;
// Winner's resolved emissive sample per reservoir slot — the spatial pass reshades
// neighbors from this (it can't `physical_load`). Slot-indexed like `reservoirs`.
@group(1) @binding(11) var<storage, read_write> light_samples: array<StoredLight>;
const NO_GBUFFER: u32 = 0xffffffffu;


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
    // Hit distance (world units) — converts the light's area pdf to solid angle
    // for the emissive-vs-NEE MIS weight.
    @builtin(ray_t_current_max) ray_t: f32,
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

    // Normal-facing debug view: stash the pre-bend shading normal + the RAW winding
    // normal (cross of the fetched world positions, NOT ray_hit.geometric_world_normal
    // which is sign-matched to the vertex normal). Same edge convention as the resolve.
    payload.hit_normal_oct = pack2x16snorm(octahedral_encode(ray_hit.world_normal) * 2.0 - 1.0);
    let wp0 = vec3<f32>(dot(transform[0].xyz, hit_positions[0]) + transform[0].w, dot(transform[1].xyz, hit_positions[0]) + transform[1].w, dot(transform[2].xyz, hit_positions[0]) + transform[2].w);
    let wp1 = vec3<f32>(dot(transform[0].xyz, hit_positions[1]) + transform[0].w, dot(transform[1].xyz, hit_positions[1]) + transform[1].w, dot(transform[2].xyz, hit_positions[1]) + transform[2].w);
    let wp2 = vec3<f32>(dot(transform[0].xyz, hit_positions[2]) + transform[0].w, dot(transform[1].xyz, hit_positions[2]) + transform[1].w, dot(transform[2].xyz, hit_positions[2]) + transform[2].w);
    let geo_raw = normalize(cross(wp0 - wp1, wp0 - wp2));
    payload.hit_geo_normal_oct = pack2x16snorm(octahedral_encode(geo_raw) * 2.0 - 1.0);

    // Displacement debug view (`frame.w == 1`): replace shading with the surface's height map in
    // grayscale, validating the displacement wiring (which map → which surface, the UVs, the sign)
    // BEFORE tessellation actually moves geometry. Surfaces with no displacement map read dim grey
    // for context. The blit passes exposure 1.0 for debug views, so the raw `[0,1]` height displays.
    if camera.frame.w == 1u {
        let dmat = load_material_bindless(ray_hit.material_id);
        var height = 0.02;
        if dmat.displacement_texture_id != TEXTURE_MAP_NONE {
            height = sample_texture_lod(dmat.displacement_texture_id, ray_hit.uv, 0.0).r;
        }
        payload.emitted = vec3<f32>(height);
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

    // Estimator flags (RtCamera.atmo.w): bit 0 = NEE off (SolariReference validation
    // lever — BSDF-only). With NEE off, emissive MIS weights MUST stay 1 or the
    // technique's share of the energy is simply dropped (biased dark).
    // Bit 1 = ReSTIR DI (rung 3): reservoir + temporal reuse at the primary vertex,
    // emissive-only single-sample NEE at bounce vertices, directionals sampled
    // deterministically per light everywhere (the sun never enters a reservoir).
    let flags = bitcast<u32>(camera.atmo.w);
    let nee_off = (flags & 1u) != 0u;
    let restir_mode = (flags & 2u) != 0u;
    let spatial_on = restir_mode && (flags & 8u) != 0u;
    // ReSTIR GI (flag bit 5): raygen reshades GI from the stored sample and
    // needs this surface's shading inputs in the SurfaceGbuf.
    let gi_mode = (flags & 32u) != 0u;
    let is_primary = payload.gbuffer_pixel != NO_GBUFFER;

    // Emissive contribution, MIS-weighted against NEE on all but the primary ray.
    // Both pdfs in SOLID-ANGLE measure: NEE's pdf is per-light-area, so convert by
    // the d²/cosθ Jacobian of the same direction the BRDF pdf is expressed in —
    // mismatched measures still partition unity (unbiased) but skew the balance
    // with distance, paying variance near large emitters.
    var mis_weight = 1.0;
    if payload.p_bounce != 0.0 && !nee_off {
        let cos_l = abs(dot(ray_direction, ray_hit.geometric_world_normal));
        // The mirror must match the light technique that COULD have produced this
        // direction: stratified pick normally, flux-only in restir mode (no stratum).
        var p_area = random_emissive_light_pdf(ray_hit);
        if restir_mode {
            p_area = random_emissive_light_pdf_flux(ray_hit);
        }
        let p_light = p_area * (ray_t * ray_t) / max(cos_l, 1e-4);
        mis_weight = power_heuristic(payload.p_bounce, p_light);
    }
    var emitted = mis_weight * ray_hit.material.emissive;
    payload.emissive_mis = emitted;

    // Direct lighting via RIS (rung 2): stream M light candidates through a
    // one-slot weighted reservoir — each weighted w = p̂/p, where the target
    // p̂ = luminance(w_mis · BRDF · L · G) folds the NEE-vs-BSDF MIS weight into
    // the technique's integrand (RIS is unbiased for ANY integrand, and the
    // emissive-hit MIS side keeps using the SOURCE pdf, so the pair stays exact).
    // One shadow ray for the winner only; shade by f(y) · W with the unbiased
    // contribution weight W = Σw / (M·p̂(y)). M = 1 reduces algebraically to
    // plain NEE (W = 1/p). M rides estimator-flag bits 8..15 (SolariReference).
    let is_perfectly_specular =
        ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && ray_hit.material.metallic > 0.9999;
    if !is_perfectly_specular && !nee_off {
        // Emissive DI: all three paths below fill the same reservoir state and
        // share the winner's shadow ray + shade (estimator = sel_f · Σw/(M·p̂)).
        let ris_m = max((flags >> 8u) & 0xffu, 1u);
        var w_sum = 0.0;
        var sel = LightSample(NULL_LIGHT_ID, 0u);
        var sel_f = vec3<f32>(0.0);
        var sel_pos = vec4<f32>(0.0);
        var sel_phat = 0.0;
        var res_m = 0.0;
        if restir_mode {
            if emissive_light_count() > 0u {
                // Candidates: M at the primary vertex (the reservoir), single-sample
                // NEE at bounce vertices — rung 2 proved per-bounce M is pure cost.
                let cand_m = select(1u, ris_m, is_primary);
                for (var c = 0u; c < cand_m; c += 1u) {
                    let cand = generate_random_emissive_light_sample(&rng);
                    if cand.light_sample.light_id == NULL_LIGHT_ID {
                        break;
                    }
                    let lc = calculate_resolved_light_contribution(
                        cand.resolved_light_sample, ray_hit.world_position, world_normal);
                    if lc.inverse_pdf <= 0.0 {
                        continue;
                    }
                    var w_mis = 1.0;
                    if lc.brdf_rays_can_hit {
                        w_mis = power_heuristic(lc.pdf_solid, brdf_pdf(wo, lc.wi, world_normal, ray_hit.material, F_ab));
                    }
                    let f = w_mis * lc.radiance * saturate(dot(world_normal, lc.wi)) * evaluate_brdf(wo, lc.wi, world_normal, ray_hit.material, F_ab);
                    let phat = pick_luminance(f);
                    let w = phat * lc.inverse_pdf;
                    if w <= 0.0 {
                        continue;
                    }
                    w_sum += w;
                    if w == w_sum || rand_f(&rng) * w_sum < w {
                        sel = cand.light_sample;
                        sel_f = f;
                        sel_pos = cand.resolved_light_sample.world_position;
                        sel_phat = phat;
                    }
                }
                res_m = f32(cand_m);

                // Temporal reuse (primary vertex only): reproject through the DLSS
                // motion matrices, validate the history surface (depth + normal),
                // re-evaluate its sample's p̂ HERE, and merge — the history counts
                // as prev.m candidates at the cost of one resolve, no rays.
                if is_primary {
                    let prev_clip = camera.prev_clip_from_world
                        * vec4<f32>(ray_hit.previous_frame_world_position, 1.0);
                    if prev_clip.w > 1.0e-4 {
                        let prev_uv = (prev_clip.xy / prev_clip.w) * vec2<f32>(0.5, -0.5) + 0.5;
                        if all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv < vec2<f32>(1.0)) {
                            let pp = vec2<u32>(prev_uv * camera.dims.xy);
                            let prev_slot = (pp.y * u32(camera.dims.x) + pp.x) * 2u
                                + (1u - (camera.frame.x & 1u));
                            let prev = reservoirs[prev_slot];
                            let prev_n = octahedral_decode_signed(unpack2x16snorm(prev.normal_oct));
                            // prev.depth was that pixel's view depth when written;
                            // prev_clip.w is THIS surface's view depth in that camera.
                            let depth_ok = abs(prev.depth - prev_clip.w) <= 0.1 * max(prev.depth, prev_clip.w);
                            if prev.m > 0.0 && prev.light_id != NULL_LIGHT_ID
                                && depth_ok && dot(prev_n, world_normal) > 0.9 {
                                let prev_m = min(prev.m, max(camera.dims.z, 1.0) * res_m);
                                let pls = LightSample(prev.light_id, prev.seed);
                                let presolved = resolve_emissive_for_restir(pls);
                                let plc = calculate_resolved_light_contribution(
                                    presolved, ray_hit.world_position, world_normal);
                                if plc.inverse_pdf > 0.0 {
                                    var w_mis = 1.0;
                                    if plc.brdf_rays_can_hit {
                                        w_mis = power_heuristic(plc.pdf_solid, brdf_pdf(wo, plc.wi, world_normal, ray_hit.material, F_ab));
                                    }
                                    let f = w_mis * plc.radiance * saturate(dot(world_normal, plc.wi)) * evaluate_brdf(wo, plc.wi, world_normal, ray_hit.material, F_ab);
                                    let phat = pick_luminance(f);
                                    let w = phat * prev.w * prev_m;
                                    w_sum += w;
                                    if w > 0.0 && rand_f(&rng) * w_sum < w {
                                        sel = pls;
                                        sel_f = f;
                                        sel_pos = presolved.world_position;
                                        sel_phat = phat;
                                    }
                                }
                                // M merges whenever the neighbor is accepted, even at
                                // zero re-target weight (Algorithm-4 semantics).
                                res_m += prev_m;
                            }
                        }
                    }
                }
            }
        } else {
            // Rung-2 path (restir off), byte-identical estimator: stratified
            // candidates (directional + emissive) through the one-slot reservoir.
            for (var c = 0u; c < ris_m; c += 1u) {
                let cand = generate_random_light_sample(&rng);
                if cand.light_sample.light_id == NULL_LIGHT_ID {
                    break; // no lights in the scene
                }
                let lc = calculate_resolved_light_contribution(
                    cand.resolved_light_sample,
                    ray_hit.world_position,
                    world_normal,
                );
                if lc.inverse_pdf <= 0.0 {
                    continue;
                }
                var w_mis = 1.0;
                if lc.brdf_rays_can_hit {
                    let pdf_of_bounce = brdf_pdf(wo, lc.wi, world_normal, ray_hit.material, F_ab);
                    w_mis = power_heuristic(lc.pdf_solid, pdf_of_bounce);
                }
                let f = w_mis * lc.radiance * saturate(dot(world_normal, lc.wi)) * evaluate_brdf(wo, lc.wi, world_normal, ray_hit.material, F_ab);
                let phat = pick_luminance(f);
                let w = phat * lc.inverse_pdf;
                if w <= 0.0 {
                    continue;
                }
                w_sum += w;
                // Streaming keep: first survivor unconditionally (no rand — keeps M=1
                // stream-identical to plain NEE), then probability w/w_sum.
                if w == w_sum || rand_f(&rng) * w_sum < w {
                    sel_f = f;
                    sel_pos = cand.resolved_light_sample.world_position;
                    sel_phat = phat;
                }
            }
            res_m = f32(ris_m);
        }
        // Visibility rays, one shared loop: iterations 0..dir_rays are the
        // deterministic directional lights (the sun never enters a reservoir — a
        // one-slot reservoir arbitrating sun-vs-lamp patchworks the screen;
        // brdf_rays_can_hit is false for directionals so no MIS weight applies);
        // the final iteration is the emissive reservoir winner.
        var dir_rays = 0u;
        if restir_mode {
            dir_rays = directional_light_count();
        }
        for (var v = 0u; v <= dir_rays; v += 1u) {
            var f_vis = vec3<f32>(0.0);
            var vis_target = vec4<f32>(0.0);
            if v < dir_rays {
                let slot = active_light_list[2u + emissive_light_count() + v];
                let dls = LightSample(slot << 16u, rand_u(&rng));
                let dresolved = resolve_light_sample(dls, light_sources[slot]);
                let dlc = calculate_resolved_light_contribution(
                    dresolved, ray_hit.world_position, world_normal);
                if dlc.inverse_pdf <= 0.0 {
                    continue;
                }
                f_vis = dlc.radiance * saturate(dot(world_normal, dlc.wi))
                    * evaluate_brdf(wo, dlc.wi, world_normal, ray_hit.material, F_ab)
                    * dlc.inverse_pdf;
                // Directional sample: world_position = (unit direction, w=0).
                vis_target = dresolved.world_position;
            } else {
                // Spatial pass owns the emissive winner's visibility + shade.
                if spatial_on || sel_phat <= 0.0 {
                    continue;
                }
                f_vis = sel_f * (w_sum / (res_m * sel_phat));
                vis_target = sel_pos;
            }
            if pick_luminance(f_vis) <= 0.0 {
                continue;
            }
            // Positional target (w==1) → finite range to the light; directional
            // (w==0) → far miss.
            let shadow_origin =
                offset_ray_origin(ray_hit.world_position, ray_hit.geometric_world_normal);
            var shadow_dir = vis_target.xyz;
            var shadow_tmax = RAY_T_MAX;
            if vis_target.w == 1.0 {
                let to_light = shadow_dir - shadow_origin;
                let dist = length(to_light);
                shadow_dir = to_light / dist;
                shadow_tmax = dist - RAY_T_MIN;
            }
            if shadow_tmax < RAY_T_MIN {
                continue;
            }
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
            if shadow_payload.occluded == 0u {
                emitted += f_vis;
            }
        }
        // Persist the merged reservoir for next frame's temporal pass. Stored
        // regardless of the winner's visibility (no visibility reuse yet — zeroing
        // W on occlusion is the session-2 bias study, it darkens under merge).
        if (restir_mode || gi_mode) && is_primary {
            let view_z = -(camera.view_from_world * vec4<f32>(ray_hit.world_position, 1.0)).z;
            if restir_mode {
                var store_w = 0.0;
                if sel_phat > 0.0 && res_m > 0.0 {
                    store_w = w_sum / (res_m * sel_phat);
                }
                reservoirs[payload.gbuffer_pixel * 2u + (camera.frame.x & 1u)] = Reservoir(
                    sel.light_id,
                    sel.seed,
                    res_m,
                    store_w,
                    pack2x16snorm(octahedral_encode(world_normal) * 2.0 - 1.0),
                    max(view_z, 1.0e-4),
                    0u,
                    0u,
                );
            }
            // Surface attrs: the spatial pass's p̂ re-target + shade, and (gi_mode)
            // raygen's GI reshade at path end.
            if spatial_on || gi_mode {
                surfaces[payload.gbuffer_pixel] = SurfaceGbuf(
                    ray_hit.world_position.x,
                    ray_hit.world_position.y,
                    ray_hit.world_position.z,
                    max(view_z, 1.0e-4),
                    pack2x16snorm(octahedral_encode(world_normal) * 2.0 - 1.0),
                    pack2x16snorm(octahedral_encode(ray_hit.geometric_world_normal) * 2.0 - 1.0),
                    pack2x16float(ray_hit.material.base_color.rg),
                    pack2x16float(vec2<f32>(ray_hit.material.base_color.b, ray_hit.material.metallic)),
                    pack2x16float(vec2<f32>(ray_hit.material.roughness, ray_hit.material.perceptual_roughness)),
                    pack2x16float(vec2<f32>(ray_hit.material.reflectance, 0.0)),
                    pack2x16snorm(octahedral_encode(wo) * 2.0 - 1.0),
                    0u,
                );
            }
            if spatial_on {
                // Resolve the winner once here (chit has `physical_load`) and store it
                // so the wgpu spatial pass can reshade neighbors without bindless loads.
                let slot = payload.gbuffer_pixel * 2u + (camera.frame.x & 1u);
                if sel.light_id != NULL_LIGHT_ID {
                    light_samples[slot] = pack_stored_light(resolve_emissive_for_restir(sel), sel_f, sel_phat);
                } else {
                    light_samples[slot] = pack_stored_light(
                        ResolvedLightSample(vec4<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0), vec3<f32>(0.0), 0.0);
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

    // DI-only estimator (flag bit 2): terminate at this vertex — the standard
    // ReSTIR evaluation image, where the DI variance win isn't buried in GI noise.
    if (flags & 4u) != 0u {
        payload.bounce = 0u;
        payload.rng = rng;
        return;
    }

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
