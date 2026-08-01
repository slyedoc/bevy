// BRDF + sampling module for the RT-pipeline Slang stages — Slang mirror of
// the pieces of `bindings/brdf.wgsl` (+ the PCG RNG from `bindings/pbr.wgsl`)
// the ported closest-hits use. Pure functions, no bindings. Grows with each
// chit port; numerics are kept bit-faithful to the WGSL (same constants, same
// operation order) so a ported stage can't drift from its naga twin.
//
// Not compiled standalone: `import brdf;` (slangc resolves same-dir imports).
module brdf;

import scene_resolve;

public static const float PI = 3.141592653589793;
public static const float PI_2 = 6.283185307179586;

// ── RNG ─────────────────────────────────────────────────────────────────────

// PCG hash RNG (bevy_solari::pbr parity — bit-exact sequences).
// https://www.pcg-random.org
public uint rand_u(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}

public float rand_f(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737u;
    return float((word >> 22) ^ word) * asfloat(0x2f800004);
}

public float2 rand_vec2f(inout uint state)
{
    let x = rand_f(state);
    let y = rand_f(state);
    return float2(x, y);
}

public uint rand_range_u(uint n, inout uint state)
{
    return rand_u(state) % n;
}

// ── Basis helpers ───────────────────────────────────────────────────────────

// Copy the sign bit from B onto A (bevy_render::maths parity — negative-zero
// handling must match the Rust orthonormalize).
float copysign_f(float a, float b)
{
    return asfloat((asuint(a) & 0x7FFFFFFF) | (asuint(b) & 0x80000000));
}

// Right-handed orthonormal basis from a unit Z vector
// (https://jcgt.org/published/0006/01/01/paper.pdf). Mirrors the WGSL
// `orthonormalize` mat3x3: `t`/`b`/`n` are its columns 0/1/2.
public struct Basis
{
    public float3 t;
    public float3 b;
    public float3 n;
}

public Basis orthonormalize(float3 z_basis)
{
    let sign = copysign_f(1.0, z_basis.z);
    let a = -1.0 / (sign + z_basis.z);
    let bv = z_basis.x * z_basis.y * a;
    Basis basis;
    basis.t = float3(1.0 + sign * z_basis.x * z_basis.x * a, sign * bv, -sign * z_basis.x);
    basis.b = float3(bv, sign + z_basis.y * z_basis.y * a, -z_basis.y);
    basis.n = z_basis;
    return basis;
}

// WGSL `orthonormalize(z) * v` (column combination).
public float3 basis_mul(Basis m, float3 v)
{
    return m.t * v.x + m.b * v.y + m.n * v.z;
}

// Rec. 709 luma (inlined from bevy_core_pipeline::tonemapping).
public float luminance(float3 v)
{
    return dot(v, float3(0.2126, 0.7152, 0.0722));
}

// ── PBR primitives (bevy_solari::pbr parity) ────────────────────────────────

public float3 calculate_F0_dielectric(float3 reflectance)
{
    return 0.16 * reflectance * reflectance;
}

public float D_GGX(float roughness, float NdotH)
{
    let oneMinusNdotHSquared = 1.0 - NdotH * NdotH;
    let a = NdotH * roughness;
    let k = roughness / (oneMinusNdotHSquared + a * a);
    return k * k * (1.0 / PI);
}

public float V_SmithGGXCorrelated(float roughness, float NdotV, float NdotL)
{
    let a2 = roughness * roughness;
    let lambdaV = NdotL * sqrt((NdotV - a2 * NdotV) * NdotV + a2);
    let lambdaL = NdotV * sqrt((NdotL - a2 * NdotL) * NdotL + a2);
    return 0.5 / (lambdaV + lambdaL);
}

// Filament multiscattering specular approximation.
public float3 specular_multiscatter(
    float D, float V, float3 F, float3 F0, float2 F_ab, float specular_intensity)
{
    var Fr = (specular_intensity * D * V) * F;
    Fr *= 1.0 + F0 * (1.0 / (F_ab.x + F_ab.y) - 1.0);
    return Fr;
}

// https://www.realtimerendering.com/raytracinggems/
public float3 sample_cosine_hemisphere(float3 normal, inout uint rng)
{
    let cos_theta = 1.0 - 2.0 * rand_f(rng);
    let phi = PI_2 * rand_f(rng);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let direction = normal + float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let len_sq = dot(direction, direction);
    if (len_sq < 1e-8)
    {
        return normal;
    }
    return direction * rsqrt(len_sq);
}

// Split-sum scale/bias from the self-baked 64×64 INCLUSIVE-grid LUT (texel j =
// the value at j/63, endpoints included — see tests/bake_dfg.rs). The remap
// puts uv 0/1 exactly on the first/last texel centers.
public float2 F_AB(float perceptual_roughness, float NdotV)
{
    let uv = float2(NdotV, perceptual_roughness) * (63.0 / 64.0) + (0.5 / 64.0);
    return brdf_dfg_lut.SampleLevel(brdf_dfg_lut_sampler, uv, 0.0).rg;
}

// ── GGX VNDF sampling (bevy_solari::sampling parity) ────────────────────────

// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 1)
// Result is invalid when output.z <= 0.0, and must be discarded.
public float3 sample_ggx_vndf(float3 wi_tangent, float roughness, inout uint rng)
{
    // Mirror BRDF case
    if (roughness <= MIRROR_ROUGHNESS_THRESHOLD)
    {
        return float3(-wi_tangent.xy, wi_tangent.z);
    }

    let i = wi_tangent;
    let rand = rand_vec2f(rng);
    let i_std = normalize(float3(i.xy * roughness, i.z));
    let phi = PI_2 * rand.x;
    let a = roughness;
    let s = 1.0 + length(i.xy);
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
    let b = i.z > 0.0 ? k * i_std.z : i_std.z;
    let z = fma(1.0 - rand.y, 1.0 + b, -b);
    let sin_theta = sqrt(saturate(1.0 - z * z));
    let o_std = float3(sin_theta * cos(phi), sin_theta * sin(phi), z);
    let m_std = i_std + o_std;
    let m = normalize(float3(m_std.xy * roughness, m_std.z));
    return 2.0 * dot(i, m) * m - i;
}

public bool ggx_vndf_sample_invalid(float3 ray_tangent)
{
    return !(ray_tangent.z > 0.0);
}

// naga-parity NaN test (bit pattern, not the isnan intrinsic — fast-math-safe).
bool isnan_f(float x)
{
    return (asuint(x) & 0x7fffffff) > 0x7f800000;
}

// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 2)
public float ggx_vndf_pdf(float3 wi_tangent, float3 wo_tangent, float roughness)
{
    // Mirror BRDF case
    if (roughness <= MIRROR_ROUGHNESS_THRESHOLD)
    {
        let mirror_wo = float3(-wi_tangent.xy, wi_tangent.z);
        if (all(abs(mirror_wo - wo_tangent) < float3(0.0001)))
        {
            return asfloat(0x7F800000); // INF
        }
        return 0.0;
    }

    let i = wi_tangent;
    let o = wo_tangent;
    let m = normalize(i + o);
    let ndf = D_GGX(roughness, saturate(m.z));
    let ai = roughness * i.xy;
    let len2 = dot(ai, ai);
    let t = sqrt(len2 + i.z * i.z);
    var pdf : float;
    if (i.z >= 0.0)
    {
        let a = roughness;
        let s = 1.0 + length(i.xy);
        let a2 = a * a;
        let s2 = s * s;
        let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
        pdf = ndf / (2.0 * (k * i.z + t));
    }
    else
    {
        pdf = ndf * (t - i.z) / (2.0 * len2);
    }

    return isnan_f(pdf) ? 0.0 : pdf;
}

// ── The layered opaque BRDF (bevy_solari::brdf parity) ──────────────────────

public struct EvaluateAndSampleBrdfResult
{
    public float3 wi;
    public float3 throughput;
    public float pdf;
    public bool diffuse_selected;
}

struct LobeReflectances
{
    float3 specular;
    float3 diffuse;
}

// Hemispherical reflectance of each lobe.
LobeReflectances lobe_reflectances(
    float3 F0_metal, float3 F0_dielectric, ResolvedMaterial material, float2 F_ab)
{
    let multiscattering_factor = 1.0 / (F_ab.x + F_ab.y) - 1.0;
    let rho_specular_metallic =
        (F0_metal * F_ab.x + F_ab.y) * (1.0 + F0_metal * multiscattering_factor);
    let rho_specular_dielectric =
        (F0_dielectric * F_ab.x + F_ab.y) * (1.0 + F0_dielectric * multiscattering_factor);
    LobeReflectances rho;
    rho.specular = lerp(rho_specular_dielectric, rho_specular_metallic, material.metallic);
    rho.diffuse = (1.0 - material.metallic) * (1.0 - rho_specular_dielectric) * material.base_color;
    return rho;
}

public float3 fresnel(float3 f0, float LdotH)
{
    return f0 + (1.0 - f0) * pow(1.0 - LdotH, 5.0);
}

public float3 evaluate_diffuse_brdf(
    float3 wo, float3 wi, float3 world_normal, ResolvedMaterial material, float2 F_ab)
{
    let NdotL = dot(world_normal, wi);
    let NdotV = dot(world_normal, wo);
    if (NdotL < 0.0001 || NdotV < 0.0001)
    {
        return float3(0.0);
    }
    let F0_metal = material.base_color;
    let F0_dielectric = calculate_F0_dielectric(float3(material.reflectance));
    let rho = lobe_reflectances(F0_metal, F0_dielectric, material, F_ab);
    return rho.diffuse / PI;
}

public float3 evaluate_specular_brdf(
    float3 wo, float3 wi, float3 world_normal, ResolvedMaterial material, float2 F_ab)
{
    let H = normalize(wi + wo);
    let NdotL = dot(world_normal, wi);
    let NdotH = dot(world_normal, H);
    let LdotH = dot(wi, H);
    let NdotV = dot(world_normal, wo);
    if (NdotL < 0.0001 || NdotH < 0.0001 || LdotH < 0.0001 || NdotV < 0.0001)
    {
        return float3(0.0);
    }

    let F0_metal = material.base_color;
    let F0_dielectric = calculate_F0_dielectric(float3(material.reflectance));

    if (material.roughness <= MIRROR_ROUGHNESS_THRESHOLD)
    {
        if (abs(NdotH - 1.0) < 0.0001)
        {
            let F_metal = fresnel(F0_metal, LdotH);
            let F_dielectric = fresnel(F0_dielectric, LdotH);
            // Delta lobe: f = F/cos, so f·cos collapses back to F at the caller.
            return lerp(F_dielectric, F_metal, material.metallic) / max(NdotL, 0.0001);
        }
        return float3(0.0);
    }

    let D = D_GGX(material.roughness, NdotH);
    let Vs = V_SmithGGXCorrelated(material.roughness, NdotV, NdotL);
    let F_metal = fresnel(F0_metal, LdotH);
    let F_dielectric = fresnel(F0_dielectric, LdotH);
    return lerp(
        specular_multiscatter(D, Vs, F_dielectric, F0_dielectric, F_ab, 1.0),
        specular_multiscatter(D, Vs, F_metal, F0_metal, F_ab, 1.0),
        material.metallic);
}

public float3 evaluate_brdf(
    float3 wo, float3 wi, float3 world_normal, ResolvedMaterial material, float2 F_ab)
{
    return max(
        evaluate_diffuse_brdf(wo, wi, world_normal, material, F_ab)
            + evaluate_specular_brdf(wo, wi, world_normal, material, F_ab),
        float3(0.0));
}

public EvaluateAndSampleBrdfResult evaluate_and_sample_brdf(
    float3 wo, float3 world_normal, ResolvedMaterial material, float2 F_ab, inout uint rng)
{
    EvaluateAndSampleBrdfResult result;
    result.wi = float3(0.0);
    result.throughput = float3(0.0);
    result.pdf = 0.0;
    result.diffuse_selected = false;

    let NdotV = dot(world_normal, wo);
    if (NdotV < 0.0001)
    {
        return result;
    }
    let F0_metal = material.base_color;
    let F0_dielectric = calculate_F0_dielectric(float3(material.reflectance));
    let rho = lobe_reflectances(F0_metal, F0_dielectric, material, F_ab);
    let specular_weight = luminance(rho.specular) / luminance(rho.specular + rho.diffuse);
    let diffuse_weight = 1.0 - specular_weight;

    let TBN = orthonormalize(world_normal);
    let T = TBN.t;
    let B = TBN.b;
    let N = TBN.n;

    let wo_tangent = float3(dot(wo, T), dot(wo, B), dot(wo, N));

    float3 wi;
    float3 wi_tangent;
    let diffuse_selected = rand_f(rng) < diffuse_weight;
    if (diffuse_selected)
    {
        wi = sample_cosine_hemisphere(world_normal, rng);
        wi_tangent = float3(dot(wi, T), dot(wi, B), dot(wi, N));
    }
    else
    {
        wi_tangent = sample_ggx_vndf(wo_tangent, material.roughness, rng);
        if (ggx_vndf_sample_invalid(wi_tangent))
        {
            return result;
        }
        wi = wi_tangent.x * T + wi_tangent.y * B + wi_tangent.z * N;

        // Mirror specular is a delta function
        if (material.roughness <= MIRROR_ROUGHNESS_THRESHOLD)
        {
            result.wi = wi;
            result.throughput = evaluate_specular_brdf(wo, wi, world_normal, material, F_ab)
                * max(dot(world_normal, wi), 0.0) / specular_weight;
            result.pdf = asfloat(0x7F800000); // INF
            return result;
        }
    }

    let diffuse_pdf = wi_tangent.z / PI;
    let specular_pdf = ggx_vndf_pdf(wo_tangent, wi_tangent, material.roughness);
    let pdf = (diffuse_weight * diffuse_pdf) + (specular_weight * specular_pdf);
    let throughput = evaluate_brdf(wo, wi, world_normal, material, F_ab)
        * max(dot(world_normal, wi), 0.0) / pdf;
    result.wi = wi;
    result.throughput = throughput;
    result.pdf = pdf;
    result.diffuse_selected = diffuse_selected;
    return result;
}

public float brdf_pdf(
    float3 wo, float3 wi, float3 world_normal, ResolvedMaterial material, float2 F_ab)
{
    let NdotV = max(dot(world_normal, wo), 0.0001);
    let F0_metal = material.base_color;
    let F0_dielectric = calculate_F0_dielectric(float3(material.reflectance));
    let rho = lobe_reflectances(F0_metal, F0_dielectric, material, F_ab);
    let specular_weight = luminance(rho.specular) / luminance(rho.specular + rho.diffuse);
    let diffuse_weight = 1.0 - specular_weight;

    let TBN = orthonormalize(world_normal);
    let T = TBN.t;
    let B = TBN.b;
    let N = TBN.n;

    let wo_tangent = float3(dot(wo, T), dot(wo, B), dot(wo, N));
    let wi_tangent = float3(dot(wi, T), dot(wi, B), dot(wi, N));

    let diffuse_pdf = wi_tangent.z / PI;
    let specular_pdf = ggx_vndf_pdf(wo_tangent, wi_tangent, material.roughness);
    return (diffuse_weight * diffuse_pdf) + (specular_weight * specular_pdf);
}

// ── Fresnel ─────────────────────────────────────────────────────────────────

// Exact unpolarized dielectric Fresnel. `eta` is n_incident / n_transmitted
// along the ray; returns 1.0 past the critical angle (total internal
// reflection).
public float fresnel_dielectric(float cos_i, float eta)
{
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if (sin2_t >= 1.0)
    {
        return 1.0;
    }
    let cos_t = sqrt(1.0 - sin2_t);
    let r_parallel = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    let r_perpendicular = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    return 0.5 * (r_parallel * r_parallel + r_perpendicular * r_perpendicular);
}

// ── Glass ───────────────────────────────────────────────────────────────────

public struct GlassBsdfSample
{
    public float3 wi;
    public float3 throughput;
    public bool refracted;
}

// Smooth-dielectric sample for transmissive materials: one delta lobe chosen
// by exact Fresnel (the selection probability cancels the lobe weight, so the
// interface itself is lossless — color comes from volume absorption inside
// the medium). `normal` must lie in `wo`'s hemisphere; `eta` is
// n_incident / n_transmitted across the interface.
public GlassBsdfSample sample_glass_bsdf(float3 wo, float3 normal, float eta, inout uint rng)
{
    let cos_i = min(dot(wo, normal), 1.0);
    let reflectance = fresnel_dielectric(cos_i, eta);
    GlassBsdfSample s;
    s.throughput = float3(1.0);
    if (rand_f(rng) < reflectance)
    {
        s.wi = reflect(-wo, normal);
        s.refracted = false;
    }
    else
    {
        s.wi = refract(-wo, normal, eta);
        s.refracted = true;
    }
    return s;
}

// Shading-normal adaptation (cf. Schüssler 2017). Smooth shading normals tilt
// past the view horizon at silhouette edges, making `NdotV < 0` so every BRDF
// term zeroes out and the edge renders black. Bend the shading normal the
// minimum amount needed to bring it just into the view hemisphere.
public float3 bend_shading_normal(float3 shading_normal, float3 wo)
{
    let eps = 1e-3;
    let NdotV = dot(shading_normal, wo);
    if (NdotV >= eps)
    {
        return shading_normal;
    }
    return normalize(shading_normal + (eps - NdotV) * wo);
}
