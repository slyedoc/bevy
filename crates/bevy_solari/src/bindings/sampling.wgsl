enable wgpu_ray_query;

#define_import_path bevy_solari::sampling

#import bevy_solari::pbr::D_GGX
#import bevy_solari::pbr::{rand_f, rand_vec2f, rand_u, rand_range_u}
#import bevy_render::maths::{PI_2, orthonormalize}
#import bevy_render::utils::octahedral_decode_signed
#import bevy_solari::scene_bindings::{RAY_T_MIN, RAY_T_MAX, light_sources, active_light_list, directional_lights, LightSource, LIGHT_SOURCE_KIND_DIRECTIONAL, resolve_triangle_data_full, resolve_ray_hit_full, offset_ray_origin, load_material_bindless, material_ids, ResolvedRayHitFull, ResolvedMaterial, MIRROR_ROUGHNESS_THRESHOLD, clusters, instance_cluster_ranges}

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
// (≤~0.1% shade error vs the chit's textured resolve).
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
    f_ab_packed: u32,    // pack2x16float(F_AB) — the chit's split-sum LUT read,
                         // carried so consumers never re-sample the DFG LUT (a
                         // textureSampleLevel in raygen's sample loop measured
                         // ~5 ms/spp — the ReSTIR-GI store-path regression)
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

// SurfaceGbuf decode shared by the spatial pass and raygen. `f_ab` is the
// chit's packed LUT read — decoded here, never re-sampled.
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
    out.f_ab = unpack2x16float(s.f_ab_packed);
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
// live value). Plain f32 so the reshade matches the chit.
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

// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec22%3A297
fn triangle_barycentrics(seed: u32) -> vec3<f32> {
    var rng = seed;
    var barycentrics = rand_vec2f(&rng);
    if barycentrics.x + barycentrics.y > 1.0 { barycentrics = 1.0 - barycentrics; }
    return vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics);
}
