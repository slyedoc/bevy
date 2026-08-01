// Opaque-surface closest-hit: emissive (MIS-weighted) + next-event estimation
// + a BRDF-sampled continuation ray. Direct illumination follows the
// NVIDIA-canonical structure: NEE samples a light and tests visibility with a
// `TraceRay` shadow ray (SKIP_CLOSEST_HIT + TERMINATE_ON_FIRST_HIT, routed to
// the dedicated `miss_shadow` program via an explicit miss index). The
// fixed-function traversal keeps the shadow query OFF the chit's register
// file. The raygen driver owns the bounce loop + throughput; this shader fills
// the payload with the radiance at this vertex and the next ray to trace.
//
// The DLSS ray-reconstruction guide writes are unconditional: the set-1
// layout always carries the G-buffer bindings and the WGSL twin compiled with
// `SOLARI_DLSS` always set.
//
// Regenerate SPIR-V (from crates/bevy_solari/src/render/rt_pipeline/):
//   slangc chit_opaque.slang -target spirv -o chit_opaque.spv

import rt_payload;
import scene_resolve;
import brdf;
import sampling;

// Vulkan `gl_HitKindEXT` for a back-facing triangle (front-facing is 0xFE).
static const uint HIT_KIND_BACK_FACING = 0xFF;

// Camera UBO (set 1, binding 1) — always present in the layout. Used by the
// displacement-debug view (`camera.frame.w`) and the DLSS guide writes.
[[vk::binding(1, 1)]] ConstantBuffer<RtCamera> camera;
// DLSS Ray Reconstruction guide G-buffer (set 1) — written for the primary hit
// only (chit-direct), indexed by the launch pixel the raygen threads through
// the payload. Same bindings/packing as raygen's declarations.
[[vk::binding(5, 1)]] RWStructuredBuffer<float4> gbuffer_normal_roughness;
[[vk::binding(6, 1)]] RWStructuredBuffer<float4> gbuffer_diffuse;
[[vk::binding(7, 1)]] RWStructuredBuffer<float4> gbuffer_specular;
[[vk::binding(8, 1)]] RWStructuredBuffer<float4> gbuffer_motion;
// ReSTIR DI reservoirs: 2 interleaved slots per pixel (see `Reservoir`).
// The primary hit merges last frame's slot temporally and writes this frame's.
[[vk::binding(9, 1)]] RWStructuredBuffer<Reservoir> reservoirs;
// Primary-hit surface attributes for the spatial merge+shade pass (flag bit 3):
// with spatial on, the chit stores the post-temporal reservoir + this surface
// and SKIPS the emissive winner's shadow ray — the pass owns merge+shade.
[[vk::binding(10, 1)]] RWStructuredBuffer<SurfaceGbuf> surfaces;
// Winner's resolved emissive sample per reservoir slot — the spatial pass
// reshades neighbors from this (it can't do bindless loads). Slot-indexed like
// `reservoirs`.
[[vk::binding(11, 1)]] RWStructuredBuffer<StoredLight> light_samples;
static const uint NO_GBUFFER = 0xffffffff;

// SBT miss index of `miss_shadow` (miss 0 = miss_primary, miss 1 = miss_shadow).
static const uint SHADOW_MISS_INDEX = 1;
// Shadow rays skip the closest-hit (we only need occlusion) and stop at the
// first hit. NOT force-opaque: alpha-masked geometry (foliage) runs the
// alpha-cutout any-hit on the opaque hit group, so light passes through the
// holes — cutout shadows.
static const uint SHADOW_RAY_FLAGS =
    RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

// Per-material SBT shader record: each hit record bakes its material id (the
// record index = the material slot, set as
// `instance_contribution_to_hit_group_index` in ptlas_fill). Uniform per
// shader-record, so after SER reorders the warp by hit, material + texture
// fetches stay uniform — sidestepping non-uniform descriptor divergence.
struct SbtRecord
{
    uint material_id;
}
[[vk::shader_record]] ConstantBuffer<SbtRecord> sbt;

// NEE visibility test: assume occluded; `miss_shadow` clears the flag iff the
// ray reaches the light. Fixed-function traversal → no register cost here.
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
void chit_opaque(inout RtPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    var rng = payload.rng;
    // Tess hits synthesize their previous position from the current one; seed
    // the delta so it lands in the previous frame's origin like the mesh path.
    set_reproj_origin_delta(camera.origin_delta.xyz);
    let instance_id = InstanceIndex();
    // NV cluster-AS hit cluster (ClusterIDNV) — indexes `clusters[]` directly
    // and pairs with cluster-local `PrimitiveIndex()`.
    let hit_cluster_id = cluster_id();
    let primitive_index = PrimitiveIndex();
    let ray_direction = WorldRayDirection();
    // Hit distance (world units) — converts the light's area pdf to solid
    // angle for the emissive-vs-NEE MIS weight.
    let ray_t = RayTCurrent();
    // Geometry-debug views (cluster / triangle color): record this hit so
    // raygen can hash it to a flat color on the primary hit.
    payload.hit_cluster = hit_cluster_id;
    payload.hit_primitive = primitive_index;
    let bary = attribs.barycentrics;
    let barycentrics = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    // `ObjectToWorld3x4()` rows ARE the row-form affine the resolve expects —
    // straight from the TLAS hit, no buffer read needed.
    let o2w = ObjectToWorld3x4();
    Affine transform;
    transform.r0 = float4(o2w[0][0], o2w[0][1], o2w[0][2], o2w[0][3]);
    transform.r1 = float4(o2w[1][0], o2w[1][1], o2w[1][2], o2w[1][3]);
    transform.r2 = float4(o2w[2][0], o2w[2][1], o2w[2][2], o2w[2][3]);
    // The hit triangle's three object-space vertex positions, read straight
    // from the CLAS (VK_KHR_ray_tracing_position_fetch). Requires the CLAS
    // built with `ALLOW_DATA_ACCESS` (see `clas_arena`).
    float3 hit_positions[3] = {
        HitTriangleVertexPosition(0),
        HitTriangleVertexPosition(1),
        HitTriangleVertexPosition(2),
    };
    var ray_hit = resolve_triangle_data_full_mat_fetch(
        instance_id, sbt.material_id, transform, hit_cluster_id, primitive_index,
        barycentrics, hit_positions);
    // Lighting-only debug view (estimator bit 28): shade every surface with a
    // WHITE base color so the image is pure light transport. Downstream
    // consumers (NRC records, surface G-buffer, DLSS guides, reshades) all
    // read the overridden material, so the whole pipeline sees the white world
    // consistently.
    if ((asuint(camera.atmo.w) & (1u << 28)) != 0)
    {
        ray_hit.material.base_color = float3(1.0);
    }

    // Normal-facing debug view: stash the pre-bend shading normal + the RAW
    // winding normal (cross of the fetched world positions, NOT
    // ray_hit.geometric_world_normal which is sign-matched to the vertex
    // normal).
    payload.hit_normal_oct = pack_snorm2x16(octahedral_encode(ray_hit.world_normal) * 2.0 - 1.0);
    // NRC record material: r5g6b5 base color + metallic8 + roughness8.
    {
        let bc = clamp(ray_hit.material.base_color, float3(0.0), float3(1.0));
        payload.hit_material = (uint(bc.r * 31.0 + 0.5) << 27)
            | (uint(bc.g * 63.0 + 0.5) << 21)
            | (uint(bc.b * 31.0 + 0.5) << 16)
            | (uint(clamp(ray_hit.material.metallic, 0.0, 1.0) * 255.0 + 0.5) << 8)
            | uint(clamp(ray_hit.material.roughness, 0.0, 1.0) * 255.0 + 0.5);
    }
    let wp0 = affine_transform_point(transform, hit_positions[0]);
    let wp1 = affine_transform_point(transform, hit_positions[1]);
    let wp2 = affine_transform_point(transform, hit_positions[2]);
    let geo_raw = normalize(cross(wp0 - wp1, wp0 - wp2));
    payload.hit_geo_normal_oct = pack_snorm2x16(octahedral_encode(geo_raw) * 2.0 - 1.0);

    // Displacement debug view (`frame.w == 1`): replace shading with the
    // surface's height map in grayscale, validating the displacement wiring
    // BEFORE tessellation actually moves geometry.
    if (camera.frame.w == 1)
    {
        let dmat = load_material_bindless(ray_hit.material_id);
        var height = 0.02;
        if (dmat.displacement_texture_id != TEXTURE_MAP_NONE)
        {
            height = sample_texture_lod(dmat.displacement_texture_id, ray_hit.uv, 0.0).r;
        }
        payload.emitted = float3(height);
        payload.bounce = 0;
        payload.rng = rng;
        return;
    }

    let wo = -ray_direction;
    // Two-sided opaque shading: flip both normals to the hit side so the
    // surface shades as a solid from either side, keyed on the hardware
    // winding (`HitKind`), not the interpolated normal.
    if (HitKind() == HIT_KIND_BACK_FACING)
    {
        ray_hit.geometric_world_normal = -ray_hit.geometric_world_normal;
        ray_hit.world_normal = -ray_hit.world_normal;
    }
    // Bend the smooth shading normal into the view hemisphere so silhouette
    // edges (interpolated normal past 90°) don't black out.
    let world_normal = bend_shading_normal(ray_hit.world_normal, wo);
    let NdotV = max(dot(world_normal, wo), 0.0001);
    let F_ab = F_AB(ray_hit.material.perceptual_roughness, NdotV);

    // Estimator flags (RtCamera.atmo.w): bit 0 = NEE off (BSDF-only
    // validation lever). Bit 1 = ReSTIR DI. Bit 3 = spatial pass. Bit 5 =
    // ReSTIR GI (raygen reshades GI from the stored sample and needs this
    // surface's shading inputs in the SurfaceGbuf).
    let flags = asuint(camera.atmo.w);
    let nee_off = (flags & 1) != 0;
    let restir_mode = (flags & 2) != 0;
    let spatial_on = restir_mode && (flags & 8) != 0;
    let gi_mode = (flags & 32) != 0;
    let is_primary = payload.gbuffer_pixel != NO_GBUFFER;

    // Emissive contribution, MIS-weighted against NEE on all but the primary
    // ray. Both pdfs in SOLID-ANGLE measure: NEE's pdf is per-light-area, so
    // convert by the d²/cosθ Jacobian.
    var mis_weight = 1.0;
    if (payload.p_bounce != 0.0 && !nee_off)
    {
        let cos_l = abs(dot(ray_direction, ray_hit.geometric_world_normal));
        // The mirror must match the light technique that COULD have produced
        // this direction: stratified pick normally, flux-only in restir mode.
        var p_area = random_emissive_light_pdf(ray_hit);
        if (restir_mode)
        {
            p_area = random_emissive_light_pdf_flux(ray_hit);
        }
        let p_light = p_area * (ray_t * ray_t) / max(cos_l, 1e-4);
        mis_weight = power_heuristic(payload.p_bounce, p_light);
    }
    var emitted = mis_weight * ray_hit.material.emissive;
    payload.emissive_mis = emitted;

    // Direct lighting via RIS: stream M light candidates through a one-slot
    // weighted reservoir — each weighted w = p̂/p, where the target
    // p̂ = luminance(w_mis · BRDF · L · G) folds the NEE-vs-BSDF MIS weight
    // into the technique's integrand. One shadow ray for the winner only;
    // shade by f(y) · W with W = Σw / (M·p̂(y)). M = 1 reduces algebraically
    // to plain NEE. M rides estimator-flag bits 8..15.
    let is_perfectly_specular = ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD
        && ray_hit.material.metallic > 0.9999;
    if (!is_perfectly_specular && !nee_off)
    {
        // Emissive DI: all three paths below fill the same reservoir state and
        // share the winner's shadow ray + shade (estimator = sel_f · Σw/(M·p̂)).
        let ris_m = max((flags >> 8) & 0xff, 1);
        var w_sum = 0.0;
        LightSample sel = { NULL_LIGHT_ID, 0 };
        var sel_f = float3(0.0);
        var sel_pos = float4(0.0);
        var sel_phat = 0.0;
        var res_m = 0.0;
        if (restir_mode)
        {
            if (emissive_light_count() > 0)
            {
                // Candidates: M at the primary vertex (the reservoir),
                // single-sample NEE at bounce vertices — per-bounce M is pure cost.
                let cand_m = is_primary ? ris_m : 1;
                for (uint c = 0; c < cand_m; c++)
                {
                    let cand = generate_random_emissive_light_sample(rng);
                    if (cand.light_sample.light_id == NULL_LIGHT_ID)
                    {
                        break;
                    }
                    let lc = calculate_resolved_light_contribution(
                        cand.resolved_light_sample, ray_hit.world_position, world_normal);
                    if (lc.inverse_pdf <= 0.0)
                    {
                        continue;
                    }
                    var w_mis = 1.0;
                    if (lc.brdf_rays_can_hit)
                    {
                        w_mis = power_heuristic(lc.pdf_solid,
                            brdf_pdf(wo, lc.wi, world_normal, ray_hit.material, F_ab));
                    }
                    let f = w_mis * lc.radiance * saturate(dot(world_normal, lc.wi))
                        * evaluate_brdf(wo, lc.wi, world_normal, ray_hit.material, F_ab);
                    let phat = pick_luminance(f);
                    let w = phat * lc.inverse_pdf;
                    if (w <= 0.0)
                    {
                        continue;
                    }
                    w_sum += w;
                    if (w == w_sum || rand_f(rng) * w_sum < w)
                    {
                        sel = cand.light_sample;
                        sel_f = f;
                        sel_pos = cand.resolved_light_sample.world_position;
                        sel_phat = phat;
                    }
                }
                res_m = float(cand_m);

                // Temporal reuse (primary vertex only): reproject through the
                // DLSS motion matrices, validate the history surface (depth +
                // normal), re-evaluate its sample's p̂ HERE, and merge — the
                // history counts as prev.m candidates at the cost of one
                // resolve, no rays.
                if (is_primary)
                {
                    // previous_frame_world_position comes from last frame's
                    // camera-relative transform (PREV origin); the rebased prev
                    // matrix expects CURRENT-origin input — convert, or statics
                    // reproject off by the camera delta (history slides).
                    let prev_pos =
                        ray_hit.previous_frame_world_position - camera.origin_delta.xyz;
                    let prev_clip = mat4_mul(camera.prev_clip_from_world, float4(prev_pos, 1.0));
                    if (prev_clip.w > 1.0e-4)
                    {
                        let prev_uv = (prev_clip.xy / prev_clip.w) * float2(0.5, -0.5) + 0.5;
                        if (all(prev_uv >= float2(0.0)) && all(prev_uv < float2(1.0)))
                        {
                            // DI history uses the nearest fetch; GI's
                            // stochastic-bilinear decorrelation is not applied
                            // to DI.
                            let pp = uint2(prev_uv * camera.dims.xy);
                            let prev_slot = (pp.y * uint(camera.dims.x) + pp.x) * 2
                                + (1 - (camera.frame.x & 1));
                            let prev = reservoirs[prev_slot];
                            let prev_n = octahedral_decode_signed(unpack_snorm2x16(prev.normal_oct));
                            // prev.depth was that pixel's view depth when
                            // written; prev_clip.w is THIS surface's view depth
                            // in that camera.
                            let depth_ok =
                                abs(prev.depth - prev_clip.w) <= 0.1 * max(prev.depth, prev_clip.w);
                            if (prev.m > 0.0 && prev.light_id != NULL_LIGHT_ID
                                && depth_ok && dot(prev_n, world_normal) > 0.9)
                            {
                                let prev_m = min(prev.m, max(camera.dims.z, 1.0) * res_m);
                                LightSample pls = { prev.light_id, prev.seed };
                                let presolved = resolve_emissive_for_restir(pls);
                                let plc = calculate_resolved_light_contribution(
                                    presolved, ray_hit.world_position, world_normal);
                                if (plc.inverse_pdf > 0.0)
                                {
                                    var w_mis = 1.0;
                                    if (plc.brdf_rays_can_hit)
                                    {
                                        w_mis = power_heuristic(plc.pdf_solid,
                                            brdf_pdf(wo, plc.wi, world_normal, ray_hit.material, F_ab));
                                    }
                                    let f = w_mis * plc.radiance
                                        * saturate(dot(world_normal, plc.wi))
                                        * evaluate_brdf(wo, plc.wi, world_normal, ray_hit.material, F_ab);
                                    let phat = pick_luminance(f);
                                    let w = phat * prev.w * prev_m;
                                    w_sum += w;
                                    if (w > 0.0 && rand_f(rng) * w_sum < w)
                                    {
                                        sel = pls;
                                        sel_f = f;
                                        sel_pos = presolved.world_position;
                                        sel_phat = phat;
                                    }
                                }
                                // M merges whenever the neighbor is accepted,
                                // even at zero re-target weight (Algorithm-4
                                // semantics).
                                res_m += prev_m;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            // ReSTIR off: stratified candidates (directional + emissive)
            // through the one-slot reservoir.
            for (uint c = 0; c < ris_m; c++)
            {
                let cand = generate_random_light_sample(rng);
                if (cand.light_sample.light_id == NULL_LIGHT_ID)
                {
                    break; // no lights in the scene
                }
                let lc = calculate_resolved_light_contribution(
                    cand.resolved_light_sample, ray_hit.world_position, world_normal);
                if (lc.inverse_pdf <= 0.0)
                {
                    continue;
                }
                var w_mis = 1.0;
                if (lc.brdf_rays_can_hit)
                {
                    let pdf_of_bounce = brdf_pdf(wo, lc.wi, world_normal, ray_hit.material, F_ab);
                    w_mis = power_heuristic(lc.pdf_solid, pdf_of_bounce);
                }
                let f = w_mis * lc.radiance * saturate(dot(world_normal, lc.wi))
                    * evaluate_brdf(wo, lc.wi, world_normal, ray_hit.material, F_ab);
                let phat = pick_luminance(f);
                let w = phat * lc.inverse_pdf;
                if (w <= 0.0)
                {
                    continue;
                }
                w_sum += w;
                // Streaming keep: first survivor unconditionally (no rand —
                // keeps M=1 stream-identical to plain NEE), then probability
                // w/w_sum.
                if (w == w_sum || rand_f(rng) * w_sum < w)
                {
                    sel_f = f;
                    sel_pos = cand.resolved_light_sample.world_position;
                    sel_phat = phat;
                }
            }
            res_m = float(ris_m);
        }
        // Visibility rays, one shared loop: iterations 0..dir_rays are the
        // deterministic directional lights (the sun never enters a reservoir —
        // a one-slot reservoir arbitrating sun-vs-lamp patchworks the screen;
        // brdf_rays_can_hit is false for directionals so no MIS weight
        // applies); the final iteration is the emissive reservoir winner.
        var dir_rays = 0u;
        if (restir_mode)
        {
            dir_rays = directional_light_count();
        }
        for (uint v = 0; v <= dir_rays; v++)
        {
            var f_vis = float3(0.0);
            var vis_target = float4(0.0);
            if (v < dir_rays)
            {
                let slot = active_light_list[2 + emissive_light_count() + v];
                LightSample dls = { slot << 16, rand_u(rng) };
                let dresolved = resolve_light_sample(dls, light_sources[slot]);
                let dlc = calculate_resolved_light_contribution(
                    dresolved, ray_hit.world_position, world_normal);
                if (dlc.inverse_pdf <= 0.0)
                {
                    continue;
                }
                f_vis = dlc.radiance * saturate(dot(world_normal, dlc.wi))
                    * evaluate_brdf(wo, dlc.wi, world_normal, ray_hit.material, F_ab)
                    * dlc.inverse_pdf;
                // Directional sample: world_position = (unit direction, w=0).
                vis_target = dresolved.world_position;
            }
            else
            {
                // Spatial pass owns the PRIMARY vertex's emissive winner (it
                // reshades from the per-pixel reservoir); bounce vertices have
                // no pass-side stand-in — their NEE winner must shade here or
                // the GI arm loses NEE's whole share of the MIS partition.
                if ((spatial_on && is_primary) || sel_phat <= 0.0)
                {
                    continue;
                }
                f_vis = sel_f * (w_sum / (res_m * sel_phat));
                vis_target = sel_pos;
            }
            if (pick_luminance(f_vis) <= 0.0)
            {
                continue;
            }
            // Positional target (w==1) → finite range to the light;
            // directional (w==0) → far miss.
            let shadow_origin =
                offset_ray_origin(ray_hit.world_position, ray_hit.geometric_world_normal);
            var shadow_dir = vis_target.xyz;
            var shadow_tmax = RAY_T_MAX;
            if (vis_target.w == 1.0)
            {
                let to_light = shadow_dir - shadow_origin;
                let dist = length(to_light);
                shadow_dir = to_light / dist;
                shadow_tmax = dist - RAY_T_MIN;
            }
            if (shadow_tmax < RAY_T_MIN)
            {
                continue;
            }
            if (light_visible(shadow_origin, shadow_dir, shadow_tmax))
            {
                emitted += f_vis;
            }
        }
        // Persist the merged reservoir for next frame's temporal pass. Stored
        // regardless of the winner's visibility: killing chains conditioned on
        // their own winner's shadow result biases the merge — the unbiased
        // route is visibility in the re-target p̂, one extra ray per merge.
        if ((restir_mode || gi_mode) && is_primary)
        {
            let view_z = -mat4_mul(camera.view_from_world, float4(ray_hit.world_position, 1.0)).z;
            if (restir_mode)
            {
                var store_w = 0.0;
                if (sel_phat > 0.0 && res_m > 0.0)
                {
                    store_w = w_sum / (res_m * sel_phat);
                }
                Reservoir r;
                r.light_id = sel.light_id;
                r.seed = sel.seed;
                r.m = res_m;
                r.w = store_w;
                r.normal_oct = pack_snorm2x16(octahedral_encode(world_normal) * 2.0 - 1.0);
                r.depth = max(view_z, 1.0e-4);
                r.pad_a = 0;
                r.pad_b = 0;
                reservoirs[payload.gbuffer_pixel * 2 + (camera.frame.x & 1)] = r;
            }
            // Surface attrs: the spatial pass's p̂ re-target + shade, and
            // (gi_mode) raygen's GI reshade at path end.
            if (spatial_on || gi_mode)
            {
                SurfaceGbuf s;
                s.pos_x = ray_hit.world_position.x;
                s.pos_y = ray_hit.world_position.y;
                s.pos_z = ray_hit.world_position.z;
                s.view_z = max(view_z, 1.0e-4);
                s.normal_oct = pack_snorm2x16(octahedral_encode(world_normal) * 2.0 - 1.0);
                s.geo_normal_oct =
                    pack_snorm2x16(octahedral_encode(ray_hit.geometric_world_normal) * 2.0 - 1.0);
                s.color_rg = pack_half2x16(ray_hit.material.base_color.rg);
                s.color_b_metallic =
                    pack_half2x16(float2(ray_hit.material.base_color.b, ray_hit.material.metallic));
                s.rough_prough = pack_half2x16(
                    float2(ray_hit.material.roughness, ray_hit.material.perceptual_roughness));
                s.reflectance = pack_half2x16(float2(ray_hit.material.reflectance, 0.0));
                s.wo_oct = pack_snorm2x16(octahedral_encode(wo) * 2.0 - 1.0);
                // f_ab rides along: raygen/spatial must never re-sample the DFG
                // LUT (see SurfaceGbuf.f_ab_packed).
                s.f_ab_packed = pack_half2x16(F_ab);
                surfaces[payload.gbuffer_pixel] = s;
            }
            if (spatial_on)
            {
                // Resolve the winner once here (the chit has the bindless
                // resolve) and store it so the wgpu spatial pass can reshade
                // neighbors without bindless loads.
                let slot = payload.gbuffer_pixel * 2 + (camera.frame.x & 1);
                if (sel.light_id != NULL_LIGHT_ID)
                {
                    light_samples[slot] =
                        pack_stored_light(resolve_emissive_for_restir(sel), sel_f, sel_phat);
                }
                else
                {
                    ResolvedLightSample none =
                        { float4(0.0), float3(0.0), float3(0.0), 0.0 };
                    light_samples[slot] = pack_stored_light(none, float3(0.0), 0.0);
                }
            }
        }
    }

    payload.emitted = emitted;

    // Primary-hit ray-reconstruction guide (chit-direct): only the primary
    // bounce carries a real pixel index; secondary bounces pass NO_GBUFFER and
    // skip this. F0 = 0.04 for dielectrics, base_color for metals; diffuse is
    // the energy-conserving complement of the metallic split.
    if (payload.gbuffer_pixel != NO_GBUFFER)
    {
        let px = payload.gbuffer_pixel;
        let cur_clip = mat4_mul(camera.clip_from_world, float4(ray_hit.world_position, 1.0));
        // A hit at/behind the eye (camera clipping into a surface) drives
        // clip.w → 0; the perspective divides below would emit NaN/Inf into
        // the guide and hang DLSS. Skip the guide for such a pixel, leaving
        // the raygen sky default.
        if (cur_clip.w > 1.0e-4)
        {
            gbuffer_normal_roughness[px] = float4(world_normal, ray_hit.material.roughness);
            // View-space linear depth, positive into the scene (RR
            // `DepthMode::Linear`).
            let view_pos = mat4_mul(camera.view_from_world, float4(ray_hit.world_position, 1.0));
            let linear_depth = max(-view_pos.z, 1.0e-4);
            let diffuse = ray_hit.material.base_color * (1.0 - ray_hit.material.metallic);
            gbuffer_diffuse[px] = float4(diffuse, linear_depth);
            let specular = lerp(float3(0.04), ray_hit.material.base_color, ray_hit.material.metallic);
            // `.w` = specular hit distance; raygen overwrites it after bounce
            // 1, so 0 here just defaults surfaces whose continuation ray hits
            // nothing.
            gbuffer_specular[px] = float4(specular, 0.0);
            // Screen-space motion vector: current vs previous UNJITTERED clip
            // position, UV space with y flipped. previous_frame_world_position
            // handles moving instances, the matrices handle the camera. It's
            // in the PREVIOUS frame's origin; the rebased prev matrix expects
            // CURRENT-origin input — convert, or static MVs are off by the
            // camera delta and RR's history smears.
            let prev_clip = mat4_mul(camera.prev_clip_from_world,
                float4(ray_hit.previous_frame_world_position - camera.origin_delta.xyz, 1.0));
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

    // DI-only estimator (flag bit 2): terminate at this vertex — the standard
    // ReSTIR evaluation image, where the DI variance win isn't buried in GI
    // noise.
    if ((flags & 4) != 0)
    {
        payload.bounce = 0;
        payload.rng = rng;
        return;
    }

    // BRDF-sampled continuation ray for the next bounce.
    let next_bounce = evaluate_and_sample_brdf(wo, world_normal, ray_hit.material, F_ab, rng);
    if (next_bounce.pdf == 0.0)
    {
        payload.bounce = 0; // dead path — terminate
        payload.rng = rng;
        return;
    }
    payload.attenuation = next_bounce.throughput;
    payload.next_origin = offset_ray_origin(ray_hit.world_position, ray_hit.geometric_world_normal);
    payload.next_direction = next_bounce.wi;
    payload.p_bounce = next_bounce.pdf;
    payload.bounce = 1;
    payload.rng = rng;
}
