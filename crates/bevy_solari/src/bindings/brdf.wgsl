enable wgpu_ray_query;

#define_import_path bevy_solari::brdf

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_solari::pbr::{D_GGX, V_SmithGGXCorrelated, specular_multiscatter}
#import bevy_solari::pbr::calculate_F0_dielectric
#import bevy_solari::pbr::{rand_f, sample_cosine_hemisphere}
#import bevy_render::maths::{PI, orthonormalize}
#import bevy_solari::sampling::{sample_ggx_vndf, ggx_vndf_pdf, ggx_vndf_sample_invalid}
#import bevy_solari::scene_bindings::{ResolvedMaterial, MIRROR_ROUGHNESS_THRESHOLD, brdf_dfg_lut, brdf_dfg_lut_sampler}

struct EvaluateAndSampleBrdfResult {
    wi: vec3<f32>,
    throughput: vec3<f32>,
    pdf: f32,
    diffuse_selected: bool,
}

struct LobeReflectances {
    specular: vec3<f32>,
    diffuse: vec3<f32>,
}

// Hemispherical reflectance of each lobe
fn lobe_reflectances(F0_metal: vec3<f32>, F0_dielectric: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>) -> LobeReflectances {
    let multiscattering_factor = 1.0 / (F_ab.x + F_ab.y) - 1.0;
    let rho_specular_metallic = (F0_metal * F_ab.x + F_ab.y) * (1.0 + F0_metal * multiscattering_factor);
    let rho_specular_dielectric = (F0_dielectric * F_ab.x + F_ab.y) * (1.0 + F0_dielectric * multiscattering_factor);
    return LobeReflectances(
        mix(rho_specular_dielectric, rho_specular_metallic, material.metallic),
        (1.0 - material.metallic) * (1.0 - rho_specular_dielectric) * material.base_color,
    );
}

fn evaluate_and_sample_brdf(
    wo: vec3<f32>,
    world_normal: vec3<f32>,
    material: ResolvedMaterial,
    F_ab: vec2<f32>,
    rng: ptr<function, u32>,
) -> EvaluateAndSampleBrdfResult {
    let NdotV = dot(world_normal, wo);
    if NdotV < 0.0001 { return EvaluateAndSampleBrdfResult(vec3(0.0), vec3(0.0), 0.0, false); }
    let F0_metal = material.base_color;
    let F0_dielectric = calculate_F0_dielectric(vec3(material.reflectance));
    let rho = lobe_reflectances(F0_metal, F0_dielectric, material, F_ab);
    let specular_weight = luminance(rho.specular) / luminance(rho.specular + rho.diffuse);
    let diffuse_weight = 1.0 - specular_weight;

    let TBN = orthonormalize(world_normal);
    let T = TBN[0];
    let B = TBN[1];
    let N = TBN[2];

    let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));

    var wi: vec3<f32>;
    var wi_tangent: vec3<f32>;
    let diffuse_selected = rand_f(rng) < diffuse_weight;
    if diffuse_selected {
        wi = sample_cosine_hemisphere(world_normal, rng);
        wi_tangent = vec3(dot(wi, T), dot(wi, B), dot(wi, N));
    } else {
        wi_tangent = sample_ggx_vndf(wo_tangent, material.roughness, rng);
        if ggx_vndf_sample_invalid(wi_tangent) {
            return EvaluateAndSampleBrdfResult(vec3(0.0), vec3(0.0), 0.0, false);
        }
        wi = wi_tangent.x * T + wi_tangent.y * B + wi_tangent.z * N;

        // Mirror specular is a delta function
        if material.roughness <= MIRROR_ROUGHNESS_THRESHOLD {
            return EvaluateAndSampleBrdfResult(
                wi,
                evaluate_specular_brdf(wo, wi, world_normal, material, F_ab) / specular_weight,
                bitcast<f32>(0x7F800000u), // INF
                false,
            );
        }
    }

    let diffuse_pdf = wi_tangent.z / PI;
    let specular_pdf = ggx_vndf_pdf(wo_tangent, wi_tangent, material.roughness);
    let pdf = (diffuse_weight * diffuse_pdf) + (specular_weight * specular_pdf);
    let throughput = evaluate_brdf(wo, wi, world_normal, material, F_ab) / pdf;
    return EvaluateAndSampleBrdfResult(wi, throughput, pdf, diffuse_selected);
}

fn evaluate_brdf(
    wo: vec3<f32>,
    wi: vec3<f32>,
    world_normal: vec3<f32>,
    material: ResolvedMaterial,
    F_ab: vec2<f32>,
) -> vec3<f32> {
    return max(evaluate_diffuse_brdf(wo, wi, world_normal, material, F_ab) + evaluate_specular_brdf(wo, wi, world_normal, material, F_ab), vec3(0.0));
}

fn evaluate_diffuse_brdf(wo: vec3<f32>, wi: vec3<f32>, world_normal: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>) -> vec3<f32> {
    let NdotL = dot(world_normal, wi);
    let NdotV = dot(world_normal, wo);
    if NdotL < 0.0001 || NdotV < 0.0001 { return vec3(0.0); }
    let F0_metal = material.base_color;
    let F0_dielectric = calculate_F0_dielectric(vec3(material.reflectance));
    let rho = lobe_reflectances(F0_metal, F0_dielectric, material, F_ab);
    return rho.diffuse / PI * NdotL;
}

fn evaluate_specular_brdf(wo: vec3<f32>, wi: vec3<f32>, world_normal: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>) -> vec3<f32> {
    let H = normalize(wi + wo);
    let NdotL = dot(world_normal, wi);
    let NdotH = dot(world_normal, H);
    let LdotH = dot(wi, H);
    let NdotV = dot(world_normal, wo);
    if NdotL < 0.0001 || NdotH < 0.0001 || LdotH < 0.0001 || NdotV < 0.0001 { return vec3(0.0); }

    let F0_metal = material.base_color;
    let F0_dielectric = calculate_F0_dielectric(vec3(material.reflectance));

    if material.roughness <= MIRROR_ROUGHNESS_THRESHOLD {
        if abs(NdotH - 1.0) < 0.0001 {
            let F_metal = fresnel(F0_metal, LdotH);
            let F_dielectric = fresnel(F0_dielectric, LdotH);
            return mix(F_dielectric, F_metal, material.metallic);
        } else {
            return vec3(0.0);
        }
    }

    let D = D_GGX(material.roughness, NdotH);
    let Vs = V_SmithGGXCorrelated(material.roughness, NdotV, NdotL);
    let F_metal = fresnel(F0_metal, LdotH);
    let F_dielectric = fresnel(F0_dielectric, LdotH);
    return mix(specular_multiscatter(D, Vs, F_dielectric, F0_dielectric, F_ab, 1.0),
               specular_multiscatter(D, Vs, F_metal, F0_metal, F_ab, 1.0),
               material.metallic) * NdotL;
}

fn brdf_pdf(wo: vec3<f32>, wi: vec3<f32>, world_normal: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>) -> f32 {
    let NdotV = max(dot(world_normal, wo), 0.0001);
    let F0_metal = material.base_color;
    let F0_dielectric = calculate_F0_dielectric(vec3(material.reflectance));
    let rho = lobe_reflectances(F0_metal, F0_dielectric, material, F_ab);
    let specular_weight = luminance(rho.specular) / luminance(rho.specular + rho.diffuse);
    let diffuse_weight = 1.0 - specular_weight;

    let TBN = orthonormalize(world_normal);
    let T = TBN[0];
    let B = TBN[1];
    let N = TBN[2];

    let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));
    let wi_tangent = vec3(dot(wi, T), dot(wi, B), dot(wi, N));

    let diffuse_pdf = wi_tangent.z / PI;
    let specular_pdf = ggx_vndf_pdf(wo_tangent, wi_tangent, material.roughness);
    return (diffuse_weight * diffuse_pdf) + (specular_weight * specular_pdf);
}

fn fresnel(f0: vec3<f32>, LdotH: f32) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - LdotH, 5.0);
}

// Exact unpolarized dielectric Fresnel. `eta` is n_incident / n_transmitted
// along the ray; returns 1.0 past the critical angle (total internal
// reflection).
fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t >= 1.0 {
        return 1.0;
    }
    let cos_t = sqrt(1.0 - sin2_t);
    let r_parallel = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    let r_perpendicular = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    return 0.5 * (r_parallel * r_parallel + r_perpendicular * r_perpendicular);
}

// ── Spectral dispersion ──────────────────────────────────────────────────────

// Sampled visible-spectrum range for hero-wavelength paths (nm).
const SPECTRUM_MIN_NM = 380.0;
const SPECTRUM_MAX_NM = 730.0;

// Wavelength-dependent IOR via a two-term Cauchy fit. `dispersion` follows
// `KHR_materials_dispersion` (`20 / Abbe number`, 0 = none) with `base_ior`
// the value at the Fraunhofer d line (587.6 nm): the Cauchy B coefficient is
// `(n_d − 1) / (V_d (λ_F⁻² − λ_C⁻²))`, folded into a constant with the F/C
// lines at 486.1/656.3 nm. Returns `base_ior` exactly when `dispersion` is 0.
fn dispersive_ior(base_ior: f32, dispersion: f32, lambda_nm: f32) -> f32 {
    let lambda_um = lambda_nm * 1e-3;
    let b = (base_ior - 1.0) * dispersion * 0.0261829;
    return base_ior + b * (1.0 / (lambda_um * lambda_um) - 2.89663);
}

// Hero-wavelength → RGB conversion weight for a λ sampled UNIFORMLY over
// [SPECTRUM_MIN_NM, SPECTRUM_MAX_NM]: Gaussian channel responses, each
// normalized by its integral so a spectrally flat path averages back to
// white (range / (σ√2π) at the peak). The path's RGB throughput collapses
// to this on its first dispersive interface.
fn spectral_rgb_weight(lambda_nm: f32) -> vec3<f32> {
    let d = vec3(lambda_nm) - vec3(612.0, 549.0, 465.0);
    let sigma = vec3(45.0, 42.0, 38.0);
    let g = exp(-d * d / (2.0 * sigma * sigma));
    return g * ((SPECTRUM_MAX_NM - SPECTRUM_MIN_NM) / 2.5066283) / sigma;
}

// One uniformly sampled hero wavelength over the visible range — what a path
// collapses to at its first dispersive interface (naga_oil can't export
// consts across modules, so the range is behind this function).
fn sample_hero_wavelength(rng: ptr<function, u32>) -> f32 {
    return mix(SPECTRUM_MIN_NM, SPECTRUM_MAX_NM, rand_f(rng));
}

// The fixed wavelengths (nm) of the realtime path's deterministic 3-channel
// split — the Gaussian response peaks of `spectral_rgb_weight`, so each
// channel's chain carries exactly that channel's energy.
fn spectral_lambda_rgb() -> vec3<f32> {
    return vec3(612.0, 549.0, 465.0);
}

struct GlassBsdfSample {
    wi: vec3<f32>,
    throughput: vec3<f32>,
    refracted: bool,
}

// Smooth-dielectric sample for transmissive materials: one delta lobe chosen
// by exact Fresnel (the selection probability cancels the lobe weight, so the
// interface itself is lossless — color comes from volume absorption inside
// the medium, not from filtering at the surface). `normal` must lie in `wo`'s
// hemisphere; `eta` is n_incident / n_transmitted across the interface (the
// caller knows both media — nested dielectrics make neither side "air").
fn sample_glass_bsdf(
    wo: vec3<f32>,
    normal: vec3<f32>,
    eta: f32,
    rng: ptr<function, u32>,
) -> GlassBsdfSample {
    let cos_i = min(dot(wo, normal), 1.0);
    let reflectance = fresnel_dielectric(cos_i, eta);
    if rand_f(rng) < reflectance {
        return GlassBsdfSample(reflect(-wo, normal), vec3(1.0), false);
    }
    return GlassBsdfSample(refract(-wo, normal, eta), vec3(1.0), true);
}

// Shading-normal adaptation (cf. Schüssler 2017). Smooth (interpolated)
// shading normals tilt past the view horizon at silhouette edges, making
// `NdotV < 0` so every BRDF term zeroes out and the edge renders black (only
// emissive survives). Bend the shading normal the minimum amount needed to
// bring it just into the view hemisphere (`NdotV ≈ eps`); interior shading,
// where `NdotV` is already positive, is left untouched. `wo` is the
// (normalized) direction toward the viewer.
fn bend_shading_normal(shading_normal: vec3<f32>, wo: vec3<f32>) -> vec3<f32> {
    let eps = 1e-3;
    let NdotV = dot(shading_normal, wo);
    if NdotV >= eps {
        return shading_normal;
    }
    return normalize(shading_normal + (eps - NdotV) * wo);
}

// Scale/bias approximation
fn F_AB(perceptual_roughness: f32, NdotV: f32) -> vec2<f32> {
    return textureSampleLevel(brdf_dfg_lut, brdf_dfg_lut_sampler, vec2<f32>(NdotV, perceptual_roughness), 0.0).rg;
}
