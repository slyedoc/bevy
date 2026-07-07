enable wgpu_ray_query;

#define_import_path bevy_solari::sampling

#import bevy_solari::pbr::D_GGX
#import bevy_solari::pbr::{rand_f, rand_vec2f, rand_u, rand_range_u}
#import bevy_render::maths::{PI_2, orthonormalize}
#import bevy_render::utils::octahedral_decode_signed
#import bevy_solari::scene_bindings::{trace_ray, RAY_T_MIN, RAY_T_MAX, light_sources, active_light_list, directional_lights, LightSource, LIGHT_SOURCE_KIND_DIRECTIONAL, resolve_triangle_data_full, resolve_ray_hit_full, offset_ray_origin, load_material_bindless, material_ids, ResolvedRayHitFull, ResolvedMaterial, MIRROR_ROUGHNESS_THRESHOLD, clusters, instance_cluster_ranges}

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

// One ReSTIR DI reservoir (32 B). Two slots per pixel, INTERLEAVED by frame
// parity (`pixel*2 + (frame&1)` = current, `pixel*2 + (1-(frame&1))` = previous)
// so no stage needs the pixel total to find its halves. `light_id`/`seed`
// reconstruct the light sample via `resolve_light_sample` (stable light slots);
// `w` is the unbiased contribution weight W; `normal_oct`/`depth` validate
// temporal reprojection (both from the frame the reservoir was written).
// Defined here (not rt_payload) so the wgpu spatial pass and the raw-VK RT
// shaders share ONE definition — both shader registries load this module.
struct Reservoir {
    light_id: u32,
    seed: u32,
    m: f32,
    w: f32,
    normal_oct: u32,
    depth: f32,
    pad_a: u32,
    pad_b: u32,
}

// Primary-hit surface attributes for the ReSTIR spatial pass (48 B/pixel): the
// exact shading inputs `evaluate_brdf`/`brdf_pdf`/`F_AB` need, f16-packed
// (≤~0.1% shade error vs the chit's textured resolve — far under the 1% gate).
// Positions are camera-relative (trace space); `wo = -normalize(pos)`.
struct SurfaceGbuf {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    view_z: f32,
    normal_oct: u32,     // SHADING normal (bent), snorm-oct
    geo_normal_oct: u32, // geometric normal (shadow-ray origin offset)
    color_rg: u32,       // pack2x16float(base_color.rg)
    color_b_metallic: u32,
    rough_prough: u32,   // pack2x16float(roughness, perceptual_roughness)
    reflectance: u32,    // pack2x16float(reflectance, 0)
    wo_oct: u32,         // view dir (-ray_direction) — world_position is ABSOLUTE, so
                         // the pass can't reconstruct wo from position alone
    pad_b: u32,
}

// Canonical ReSTIR GI sample: the first-bounce reconnection vertex + the suffix
// radiance through it. Size must stay a 16-byte multiple (raw-VK vs wgpu layout).
struct GiSample {
    pos_x: f32,      // x_s (bounce-1 offset ray origin)
    pos_y: f32,
    pos_z: f32,
    normal_oct: u32, // n_s, snorm-oct
    l_r: f32,        // reusable suffix radiance (bounce-1 emission excluded)
    l_g: f32,
    l_b: f32,
    w: f32,          // unbiased contribution weight (1/pdf when canonical)
    a0_r: f32,       // exact primary BSDF weight f·cos/pdf (incl. its RR share)
    a0_g: f32,
    a0_b: f32,
    m: f32,          // sample count; 0 = dead sample
    surf_normal_oct: u32, // the GENERATING surface, for temporal validation
    surf_view_z: f32,
    pad_a: u32,
    pad_b: u32,
}

// SurfaceGbuf decode shared by the spatial pass and raygen. `f_ab` stays zero —
// F_AB lives in `brdf`; callers fill it.
struct Surf {
    pos: vec3<f32>,
    view_z: f32,
    ns: vec3<f32>,
    ng: vec3<f32>,
    mat: ResolvedMaterial,
    wo: vec3<f32>,
    f_ab: vec2<f32>,
}

fn unpack_surface(s: SurfaceGbuf) -> Surf {
    var out: Surf;
    out.pos = vec3<f32>(s.pos_x, s.pos_y, s.pos_z);
    out.view_z = s.view_z;
    out.ns = octahedral_decode_signed(unpack2x16snorm(s.normal_oct));
    out.ng = octahedral_decode_signed(unpack2x16snorm(s.geo_normal_oct));
    let c_rg = unpack2x16float(s.color_rg);
    let c_bm = unpack2x16float(s.color_b_metallic);
    let r_pr = unpack2x16float(s.rough_prough);
    let refl = unpack2x16float(s.reflectance);
    var m: ResolvedMaterial;
    m.base_color = vec3<f32>(c_rg, c_bm.x);
    m.emissive = vec3<f32>(0.0);
    m.reflectance = refl.x;
    m.roughness = r_pr.x;
    m.perceptual_roughness = r_pr.y;
    m.metallic = c_bm.y;
    m.specular_transmission = 0.0;
    m.ior = 1.5;
    m.dispersion = 0.0;
    m.extinction = vec3<f32>(0.0);
    m.nested_priority = 0u;
    out.mat = m;
    // wo is stored by the chit: world_position is absolute, so the consumer
    // can't reconstruct the view dir as normalize(-pos).
    out.wo = octahedral_decode_signed(unpack2x16snorm(s.wo_oct));
    out.f_ab = vec2<f32>(0.0);
    return out;
}

struct ResolvedLightSample {
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    radiance: vec3<f32>,
    inverse_pdf: f32,
}

// Per-reservoir-slot winner sample (64 B), written by the chit (which has the
// bindless `physical_load` resolve) so the wgpu spatial pass — which CAN'T
// `physical_load` — reshades from stored data alone. Two halves: the RESOLVED
// LIGHT (for neighbors re-targeting this sample at THEIR surface) and the chit's
// EXACT shaded f + p̂ (for the OWN pixel, so its shade can't drift from the chit's
// live value — the G-buffer-reconstructed recompute darkens dim pixels ~6%).
// Plain f32 so it matches the chit under the 1% gate.
struct StoredLight {
    px: f32, py: f32, pz: f32, pw: f32, // world_position (w = 1 area, 0 directional)
    nx: f32, ny: f32, nz: f32,          // world_normal
    rr: f32, rg: f32, rb: f32,          // radiance
    inv_pdf: f32,
    fr: f32, fg: f32, fb: f32,          // chit's exact winner f (w_mis · L · BRDF)
    phat: f32,                          // chit's exact p̂ = luminance(f)
    pad: f32,                           // pad to 64 B — a struct SHARED between the
                                        // raw-VK RT shaders and the wgpu compute pass
                                        // MUST be 16-byte-aligned in size (else the two
                                        // naga paths lay out the tail fields differently)
}

fn pack_stored_light(r: ResolvedLightSample, f: vec3<f32>, phat: f32) -> StoredLight {
    return StoredLight(
        r.world_position.x, r.world_position.y, r.world_position.z, r.world_position.w,
        r.world_normal.x, r.world_normal.y, r.world_normal.z,
        r.radiance.x, r.radiance.y, r.radiance.z,
        r.inverse_pdf,
        f.x, f.y, f.z, phat, 0.0,
    );
}

fn unpack_stored_light(s: StoredLight) -> ResolvedLightSample {
    return ResolvedLightSample(
        vec4<f32>(s.px, s.py, s.pz, s.pw),
        vec3<f32>(s.nx, s.ny, s.nz),
        vec3<f32>(s.rr, s.rg, s.rb),
        s.inv_pdf,
    );
}

struct LightContribution {
    radiance: vec3<f32>,
    inverse_pdf: f32,
    wi: vec3<f32>,
    brdf_rays_can_hit: bool,
    /// The sample's pdf in SOLID-ANGLE measure (area pdf × d²/cosθ_light) — the
    /// value MIS weights compare against a BRDF pdf. `inverse_pdf` stays area-measure
    /// (the estimator's `cosθ/d²` is folded into `radiance`).
    pdf_solid: f32,
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

// Rec. 709 luminance — the CPU flux basis (`prepare_light_sources`) uses the
// same coefficients; the two MUST stay identical or the pick pdf de-mirrors.
fn pick_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

struct WeightedPick {
    index: u32,
    prob: f32,
}

/// Power-weighted emissive pick: binary-search the normalized flux CDF appended
/// (as f32 bits) after the slot lists in `active_light_list`, followed by the
/// total flux (0 = uniform-pick sentinel, see `SolariUniformLights`).
fn pick_emissive_weighted(
    emissive_count: u32,
    directional_count: u32,
    rng: ptr<function, u32>,
) -> WeightedPick {
    let cdf_base = 2u + emissive_count + directional_count;
    let total_flux = bitcast<f32>(active_light_list[cdf_base + emissive_count]);
    if total_flux <= 0.0 {
        return WeightedPick(rand_range_u(emissive_count, rng), 1.0 / f32(emissive_count));
    }
    let u = rand_f(rng);
    var lo = 0u;
    var hi = emissive_count - 1u;
    while lo < hi {
        let mid = (lo + hi) >> 1u;
        if u < bitcast<f32>(active_light_list[cdf_base + mid]) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }
    let c1 = bitcast<f32>(active_light_list[cdf_base + lo]);
    var c0 = 0.0;
    if lo > 0u {
        c0 = bitcast<f32>(active_light_list[cdf_base + lo - 1u]);
    }
    return WeightedPick(lo, max(c1 - c0, 1e-9));
}

/// Number of active directional lights (the `active_light_list` header).
fn directional_light_count() -> u32 {
    return active_light_list[1];
}

fn sample_random_light(ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, rng: ptr<function, u32>) -> LightContribution {
    let sample = generate_random_light_sample(rng);
    if sample.light_sample.light_id == NULL_LIGHT_ID {
        return LightContribution(vec3(0.0), 0.0, vec3(0.0, 1.0, 0.0), false, 0.0);
    }
    var light_contribution = calculate_resolved_light_contribution(sample.resolved_light_sample, ray_origin, origin_world_normal);
    // Tinted visibility: stained glass between the surface and the light colors
    // the light instead of blocking it, so sunlit floors pool with window color.
    light_contribution.radiance *= trace_light_transmittance(ray_origin, sample.resolved_light_sample.world_position);
    return light_contribution;
}

/// The pdf with which [`sample_random_light`] would have generated a sample on
/// this emissive hit — the BSDF-vs-NEE MIS counterpart. Must mirror the
/// stratified pick above exactly.
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

    // Solid-angle pdf for MIS: area pdf × d²/cosθ. For a directional light (w == 0)
    // d = 1 and cosθ = 1, so this is already its per-solid-angle cone pdf.
    let pdf_area = select(0.0, 1.0 / resolved_light_sample.inverse_pdf, resolved_light_sample.inverse_pdf > 0.0);
    let pdf_solid = pdf_area * light_distance_squared / max(cos_theta_light, 1e-4);

    return LightContribution(radiance, resolved_light_sample.inverse_pdf, wi, resolved_light_sample.world_position.w == 1.0, pdf_solid);
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

// Number of transmissive boundary crossings a visibility ray tracks before
// giving up. Each colored-glass pane is two crossings (front + back face), so
// this caps the march at four stacked panes between a point and the light.
const MAX_TRANSMISSION_HITS = 8u;

// Like `trace_light_visibility`, but transmissive surfaces (stained glass) tint
// the ray instead of hard-blocking it. Returns a per-channel transmittance:
// `1` fully lit, `0` shadowed by an opaque caster, or a color for light that
// reached the point through colored glass.
//
// The tint is Beer-Lambert (`exp(-extinction * thickness)`) accumulated over
// each pane's TRUE thickness — the same absorption the path tracer applies to a
// camera ray crossing that glass (pathtracer.wgsl, `medium_extinction`) — so
// tinted floor pools and colored sun shafts match the glass they pass through.
// The march goes straight (no refraction bend): correct for the soft tint, but
// it does NOT focus caustics — that needs the photon-grid path.
fn trace_light_transmittance(ray_origin: vec3<f32>, light_sample_world_position: vec4<f32>) -> vec3<f32> {
    var ray_direction = light_sample_world_position.xyz;
    var ray_t_max = RAY_T_MAX;

    if light_sample_world_position.w == 1.0 {
        let ray = ray_direction - ray_origin;
        let dist = length(ray);
        ray_direction = ray / dist;
        ray_t_max = dist - RAY_T_MIN;
    }

    if ray_t_max < RAY_T_MIN { return vec3(0.0); }

    var transmittance = vec3(1.0);
    var origin = ray_origin;
    // Absorption of the volume the ray is currently inside (air = 0).
    var medium_extinction = vec3(0.0);
    for (var i = 0u; i < MAX_TRANSMISSION_HITS; i += 1u) {
        let hit = trace_ray(origin, ray_direction, RAY_T_MIN, ray_t_max, RAY_FLAG_NONE);

        // Attenuate over the segment just travelled through the active medium
        // (air contributes nothing; the interior of a glass pane tints).
        let segment = select(ray_t_max, hit.t, hit.kind != RAY_QUERY_INTERSECTION_NONE);
        transmittance *= exp(-medium_extinction * segment);
        if hit.kind == RAY_QUERY_INTERSECTION_NONE { break; } // reached the light

        // Cheap opaque test before resolving: a non-transmissive surface is a
        // hard shadow caster, so the point is fully occluded.
        let material = load_material_bindless(material_ids[hit.instance_index]);
        if material.specular_transmission < 0.5 { return vec3(0.0); }

        // Transmissive boundary: flip the active medium (entering glass → its
        // extinction, exiting → air), keyed on the geometric normal's sign to
        // match the path tracer's `entering` test.
        let resolved = resolve_ray_hit_full(hit);
        let facing = dot(ray_direction, resolved.geometric_world_normal);
        let entering = facing < 0.0;
        medium_extinction = select(vec3(0.0), resolved.material.extinction, entering);

        // Continue from just past the surface, on the side the ray is heading.
        let go_normal = select(-resolved.geometric_world_normal, resolved.geometric_world_normal, facing > 0.0);
        origin = offset_ray_origin(resolved.world_position, go_normal);
        ray_t_max -= hit.t;
        if ray_t_max < RAY_T_MIN { break; }
        if all(transmittance < vec3(0.003)) { return vec3(0.0); } // fully absorbed
    }

    return transmittance;
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
