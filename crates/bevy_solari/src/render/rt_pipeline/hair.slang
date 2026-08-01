// Physically-based hair fiber BSDF following Chiang et al. 2016, "A Practical
// and Controllable Hair and Fur Model for Production Path Tracing" (as
// implemented by pbrt-v3/v4 `HairBSDF`) — the Slang mirror of
// `hair/hair.wgsl`, numerics kept bit-faithful.
//
// The fiber is modeled as a dielectric cylinder. A direction is parameterized
// by its longitudinal angle theta (measured from the plane perpendicular to
// the fiber axis) and its azimuthal angle phi (around the axis). Light is
// scattered through a small set of paths, each a "lobe":
//   p = 0 : R   -- reflection off the cuticle.
//   p = 1 : TT  -- transmit in, transmit out (the dominant transmissive lobe).
//   p = 2 : TRT -- transmit in, internal reflect, transmit out.
//   p = 3 : residual -- one lumped lobe approximating all higher-order paths.
// Each lobe is a product of a longitudinal term Mp(theta) and an azimuthal
// term Np(phi); attenuation Ap carries the Fresnel and per-path absorption.
//
// Not compiled standalone: `import hair;` (slangc resolves same-dir imports).
module hair;

import brdf;

static const float SQRT_PI_OVER_8 = 0.626657069; // sqrt(PI / 8)
static const int P_MAX = 3; // R, TT, TRT (residual is index P_MAX)

public struct HairSample
{
    public float3 wi;         // sampled incoming direction, world space, normalized
    public float3 throughput; // bsdf * |cos| / pdf  (already divided by pdf)
    public float pdf;         // pdf of the sampled direction (0.0 => discard)
}

// ----- small math helpers ----------------------------------------------------

float Sqr(float x)
{
    return x * x;
}

float SafeASin(float x)
{
    return asin(clamp(x, -1.0, 1.0));
}

float SafeSqrt(float x)
{
    return sqrt(max(0.0, x));
}

float Sinh(float x)
{
    return (exp(x) - exp(-x)) * 0.5;
}

// ----- Bessel I0 and its log, used by the longitudinal Mp term ---------------

float I0(float x)
{
    var val = 0.0;
    var x2i = 1.0;   // x^(2i)
    var ifact = 1.0; // i!
    var i4 = 1.0;    // 4^i
    // 10 series terms match pbrt's truncation.
    for (int i = 0; i < 10; i++)
    {
        if (i > 1)
        {
            ifact = ifact * float(i);
        }
        val = val + x2i / (i4 * Sqr(ifact));
        x2i = x2i * x * x;
        i4 = i4 * 4.0;
    }
    return val;
}

float LogI0(float x)
{
    if (x > 12.0)
    {
        // Asymptotic expansion for large argument (avoids I0 overflow).
        return x + 0.5 * (-log(2.0 * PI) + log(1.0 / x) + 1.0 / (8.0 * x));
    }
    return log(I0(x));
}

// ----- Longitudinal scattering Mp --------------------------------------------
// Gaussian detector response of the cone of scattered directions.
float Mp(float cosThetaI, float cosThetaO, float sinThetaI, float sinThetaO, float v)
{
    let a = cosThetaI * cosThetaO / v;
    let b = sinThetaI * sinThetaO / v;
    if (v <= 0.1)
    {
        return exp(LogI0(a) - b - 1.0 / v + 0.6931 + log(1.0 / (2.0 * v)));
    }
    return (exp(-b) * I0(a)) / (Sinh(1.0 / v) * 2.0 * v);
}

// ----- Logistic distribution helpers (for azimuthal Np) ----------------------

float Logistic(float x, float s)
{
    let ax = abs(x);
    return exp(-ax / s) / (s * Sqr(1.0 + exp(-ax / s)));
}

float LogisticCDF(float x, float s)
{
    return 1.0 / (1.0 + exp(-x / s));
}

// Logistic truncated to [-PI, PI], normalized over that interval.
float TrimmedLogistic(float x, float s)
{
    return Logistic(x, s) / (LogisticCDF(PI, s) - LogisticCDF(-PI, s));
}

float SampleTrimmedLogistic(float u, float s)
{
    let lo = LogisticCDF(-PI, s);
    let hi = LogisticCDF(PI, s);
    let k = u * (hi - lo) + lo;
    let x = -s * log(1.0 / k - 1.0);
    return clamp(x, -PI, PI);
}

// ----- Azimuthal scattering Np -----------------------------------------------
// Difference between the ideal specular azimuth Phi(p) and the actual phi,
// wrapped into [-PI, PI], shaped by a trimmed logistic of width s.
float Phi(int p, float gammaO, float gammaT)
{
    return 2.0 * float(p) * gammaT - 2.0 * gammaO + float(p) * PI;
}

float Np(float phi, int p, float s, float gammaO, float gammaT)
{
    var dphi = phi - Phi(p, gammaO, gammaT);
    // Wrap dphi to [-PI, PI].
    while (dphi > PI)
    {
        dphi = dphi - 2.0 * PI;
    }
    while (dphi < -PI)
    {
        dphi = dphi + 2.0 * PI;
    }
    return TrimmedLogistic(dphi, s);
}

// ----- Fresnel ---------------------------------------------------------------

// Fresnel reflectance for a dielectric interface (unpolarized).
float FrDielectric(float cosThetaI_in, float eta)
{
    var cosThetaI = clamp(cosThetaI_in, -1.0, 1.0);
    var etaLocal = eta;
    if (cosThetaI < 0.0)
    {
        etaLocal = 1.0 / eta;
        cosThetaI = -cosThetaI;
    }
    let sin2ThetaI = 1.0 - Sqr(cosThetaI);
    let sin2ThetaT = sin2ThetaI / Sqr(etaLocal);
    if (sin2ThetaT >= 1.0)
    {
        return 1.0; // total internal reflection
    }
    let cosThetaT = SafeSqrt(1.0 - sin2ThetaT);
    let rParl = (etaLocal * cosThetaI - cosThetaT) / (etaLocal * cosThetaI + cosThetaT);
    let rPerp = (cosThetaI - etaLocal * cosThetaT) / (cosThetaI + etaLocal * cosThetaT);
    return (Sqr(rParl) + Sqr(rPerp)) * 0.5;
}

// ----- Attenuation Ap --------------------------------------------------------
// Per-lobe color/Fresnel attenuation. Ap[0] is pure surface reflection; the
// transmissive lobes accumulate absorption T = exp(-sigma_a * path-length) and
// the remaining (1-f) Fresnel transmittance factors. Index P_MAX is the
// residual energy lumped into one trailing lobe.
void Ap(float cosThetaO, float eta, float h, float3 T, out float3 ap[4])
{
    let cosGammaO = SafeSqrt(1.0 - h * h);
    let cosTheta = cosThetaO * cosGammaO;
    let f = FrDielectric(cosTheta, eta);

    ap[0] = float3(f);
    ap[1] = Sqr(1.0 - f) * T;
    ap[2] = ap[1] * T * f;
    // Residual: geometric series tail (1-f)^2 * T^2 * f^2 / (1 - T*f).
    let denom = float3(1.0) - T * f;
    ap[3] = ap[2] * f * T / max(denom, float3(1e-5));
}

// ----- longitudinal variances v[] and azimuthal width s from roughness -------

void ComputeV(float beta_m, out float v[4])
{
    // pbrt's exact polynomial: 0.726*b + 0.812*b^2 + 3.7*b^20, squared -> variance.
    let v0 = Sqr(0.726 * beta_m + 0.812 * Sqr(beta_m) + 3.7 * pow(beta_m, 20.0));
    v[0] = v0;
    v[1] = 0.25 * v0;
    v[2] = 4.0 * v0;
    v[3] = v[2];
}

float ComputeS(float beta_n)
{
    // s = SqrtPiOver8 * (0.265*b + 1.194*b^2 + 5.372*b^22)
    return SQRT_PI_OVER_8 * (0.265 * beta_n + 1.194 * Sqr(beta_n) + 5.372 * pow(beta_n, 22.0));
}

// ----- cuticle-tilt sin/cos per lobe (alpha shifts each lobe's cone) ---------

void ComputeSin2kAlpha(float alpha, out float s[3])
{
    s[0] = sin(alpha);
    s[1] = 2.0 * s[0] * cos(alpha);              // sin(2a)
    s[2] = 2.0 * s[1] * (1.0 - 2.0 * Sqr(s[0])); // sin(4a) via double-angle
}

void ComputeCos2kAlpha(float sin2k[3], out float c[3])
{
    c[0] = SafeSqrt(1.0 - Sqr(sin2k[0]));
    c[1] = SafeSqrt(1.0 - Sqr(sin2k[1]));
    c[2] = SafeSqrt(1.0 - Sqr(sin2k[2]));
}

// ----- fiber frame -----------------------------------------------------------
// pbrt's hair frame uses the fiber tangent as the "u" axis. A direction d maps
// to (sinTheta = dot(d,u), phi = atan2(dot(d,w), dot(d,v))) where (v,w) span
// the cross-section plane.
struct FiberFrame
{
    float3 u; // fiber axis (tangent)
    float3 v;
    float3 w;
}

FiberFrame build_frame(float3 tangent)
{
    let u = normalize(tangent);
    var up = float3(0.0, 1.0, 0.0);
    if (abs(u.y) > 0.99)
    {
        up = float3(1.0, 0.0, 0.0);
    }
    let v = normalize(cross(up, u));
    let w = cross(u, v);
    FiberFrame f;
    f.u = u;
    f.v = v;
    f.w = w;
    return f;
}

// Returns (sinTheta, cosTheta, phi) for a world-space direction in the frame.
float3 to_fiber(FiberFrame frame, float3 d)
{
    let sinTheta = dot(d, frame.u);
    let cosTheta = SafeSqrt(1.0 - Sqr(sinTheta));
    let phi = atan2(dot(d, frame.w), dot(d, frame.v));
    return float3(sinTheta, cosTheta, phi);
}

// Rebuild a world direction from (sinTheta, phi) in the frame.
float3 from_fiber(FiberFrame frame, float sinTheta, float phi)
{
    let cosTheta = SafeSqrt(1.0 - Sqr(sinTheta));
    return frame.u * sinTheta
        + frame.v * (cosTheta * cos(phi))
        + frame.w * (cosTheta * sin(phi));
}

// ----- shared evaluation core ------------------------------------------------
// Computes the full BSDF (sum over lobes) and the combined pdf. `h` is the
// offset across the fiber width; with no geometric normal available it is
// derived from the azimuth of wo (a common real-time simplification).
struct HairEval
{
    float3 f;
    float pdf;
}

HairEval eval_core(
    float3 wo, float3 wi, FiberFrame frame,
    float3 sigma_a, float beta_m, float beta_n, float alpha, float eta)
{
    let fo = to_fiber(frame, wo);
    let fi = to_fiber(frame, wi);
    let sinThetaO = fo.x;
    let cosThetaO = fo.y;
    let phiO = fo.z;
    let sinThetaI = fi.x;
    let cosThetaI = fi.y;
    let phiI = fi.z;

    // Offset h across the fiber: derive from the azimuth of wo so the geometry
    // is self-consistent without an explicit surface normal. h in [-1, 1].
    let h = clamp(sin(phiO) * cosThetaO, -0.9999, 0.9999);

    let gammaO = SafeASin(h);

    // Refracted ray geometry (Bravais / modified IOR for the tilted fiber).
    let sinThetaT = sinThetaO / eta;
    let cosThetaT = SafeSqrt(1.0 - Sqr(sinThetaT));
    let etap = SafeSqrt(Sqr(eta) - Sqr(sinThetaO)) / max(cosThetaO, 1e-4);
    let sinGammaT = h / etap;
    let cosGammaT = SafeSqrt(1.0 - Sqr(sinGammaT));
    let gammaT = SafeASin(sinGammaT);

    // Absorption along the transmission path through the cross-section.
    // `sigma_a` is supplied directly (from melanin concentrations).
    let T = float3(
        exp(-sigma_a.x * 2.0 * cosGammaT / max(cosThetaT, 1e-4)),
        exp(-sigma_a.y * 2.0 * cosGammaT / max(cosThetaT, 1e-4)),
        exp(-sigma_a.z * 2.0 * cosGammaT / max(cosThetaT, 1e-4)));

    float3 ap[4];
    Ap(cosThetaO, eta, h, T, ap);
    float v[4];
    ComputeV(beta_m, v);
    let s = ComputeS(beta_n);
    float sin2k[3];
    ComputeSin2kAlpha(alpha, sin2k);
    float cos2k[3];
    ComputeCos2kAlpha(sin2k, cos2k);

    let phi = phiI - phiO;

    // ap pdf weights (luminance of each lobe's attenuation), for sampling/MIS.
    float apPdf[4];
    var sumY = 0.0;
    for (int p = 0; p < 4; p++)
    {
        let y = ap[p].x * 0.299 + ap[p].y * 0.587 + ap[p].z * 0.114;
        apPdf[p] = y;
        sumY = sumY + y;
    }
    let invSum = sumY > 0.0 ? 1.0 / sumY : 0.0;
    for (int p = 0; p < 4; p++)
    {
        apPdf[p] = apPdf[p] * invSum;
    }

    var fsum = float3(0.0);
    var pdf = 0.0;

    for (int p = 0; p < P_MAX; p++)
    {
        // Apply cuticle tilt: rotate the longitudinal cone for this lobe.
        float sinThetaOp;
        float cosThetaOp;
        if (p == 0)
        {
            sinThetaOp = sinThetaO * cos2k[1] - cosThetaO * sin2k[1];
            cosThetaOp = cosThetaO * cos2k[1] + sinThetaO * sin2k[1];
        }
        else if (p == 1)
        {
            sinThetaOp = sinThetaO * cos2k[0] + cosThetaO * sin2k[0];
            cosThetaOp = cosThetaO * cos2k[0] - sinThetaO * sin2k[0];
        }
        else
        {
            sinThetaOp = sinThetaO * cos2k[2] + cosThetaO * sin2k[2];
            cosThetaOp = cosThetaO * cos2k[2] - sinThetaO * sin2k[2];
        }
        cosThetaOp = abs(cosThetaOp);

        let mp = Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[p]);
        let np = Np(phi, p, s, gammaO, gammaT);
        fsum = fsum + mp * np * ap[p];
        pdf = pdf + mp * apPdf[p] * np;
    }

    // Residual lobe (no tilt applied, isotropic azimuth).
    let mpR = Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v[3]);
    fsum = fsum + mpR * ap[3] * (1.0 / (2.0 * PI));
    pdf = pdf + mpR * apPdf[3] * (1.0 / (2.0 * PI));

    HairEval result;
    result.f = fsum;
    result.pdf = pdf;
    return result;
}

// ----- public evaluation -----------------------------------------------------

public float3 eval_hair_bsdf(
    float3 wo, float3 wi, float3 tangent,
    float3 sigma_a, float beta_m, float beta_n, float alpha, float eta)
{
    let frame = build_frame(tangent);
    let e = eval_core(wo, wi, frame, sigma_a, beta_m, beta_n, alpha, eta);
    return e.f;
}

public float pdf_hair_bsdf(
    float3 wo, float3 wi, float3 tangent,
    float beta_m, float beta_n, float alpha, float eta)
{
    let frame = build_frame(tangent);
    // Absorption barely shifts the pdf shape (only ap luminance ratios); use
    // zero (T = 1) so the lobe-selection weights track the sampler's choice.
    let e = eval_core(wo, wi, frame, float3(0.0), beta_m, beta_n, alpha, eta);
    return e.pdf;
}

// ----- importance sampling ---------------------------------------------------

public HairSample sample_hair_bsdf(
    float3 wo,
    float3 tangent,
    float3 sigma_a,
    float beta_m,
    float beta_n,
    float alpha,
    float eta,
    inout uint rng)
{
    HairSample result;
    result.wi = float3(0.0, 0.0, 1.0);
    result.throughput = float3(0.0);
    result.pdf = 0.0;

    let frame = build_frame(tangent);
    let fo = to_fiber(frame, wo);
    let sinThetaO = fo.x;
    let cosThetaO = fo.y;
    let phiO = fo.z;

    let h = clamp(sin(phiO) * cosThetaO, -0.9999, 0.9999);
    let gammaO = SafeASin(h);

    let etap = SafeSqrt(Sqr(eta) - Sqr(sinThetaO)) / max(cosThetaO, 1e-4);
    let cosThetaT = SafeSqrt(1.0 - Sqr(sinThetaO / eta));
    let sinGammaT = h / etap;
    let cosGammaT = SafeSqrt(1.0 - Sqr(sinGammaT));
    let gammaT = SafeASin(sinGammaT);

    let T = float3(
        exp(-sigma_a.x * 2.0 * cosGammaT / max(cosThetaT, 1e-4)),
        exp(-sigma_a.y * 2.0 * cosGammaT / max(cosThetaT, 1e-4)),
        exp(-sigma_a.z * 2.0 * cosGammaT / max(cosThetaT, 1e-4)));
    float3 ap[4];
    Ap(cosThetaO, eta, h, T, ap);

    // Build the per-lobe selection cdf from ap luminance.
    float apPdf[4];
    var sumY = 0.0;
    for (int p = 0; p < 4; p++)
    {
        let y = ap[p].x * 0.299 + ap[p].y * 0.587 + ap[p].z * 0.114;
        apPdf[p] = y;
        sumY = sumY + y;
    }
    if (sumY <= 0.0)
    {
        return result;
    }
    let invSum = 1.0 / sumY;
    for (int p = 0; p < 4; p++)
    {
        apPdf[p] = apPdf[p] * invSum;
    }

    // Choose a lobe.
    let uLobe = rand_f(rng);
    var p = 0;
    var cdf = apPdf[0];
    // Linear scan over the 4 lobes.
    for (int k = 0; k < P_MAX; k++)
    {
        if (uLobe < cdf)
        {
            p = k;
            break;
        }
        p = k + 1;
        cdf = cdf + apPdf[k + 1];
    }

    float sin2k[3];
    ComputeSin2kAlpha(alpha, sin2k);
    float cos2k[3];
    ComputeCos2kAlpha(sin2k, cos2k);

    // Tilt-adjusted outgoing longitudinal angle for the chosen lobe.
    var sinThetaOp = sinThetaO;
    var cosThetaOp = cosThetaO;
    if (p == 0)
    {
        sinThetaOp = sinThetaO * cos2k[1] - cosThetaO * sin2k[1];
        cosThetaOp = cosThetaO * cos2k[1] + sinThetaO * sin2k[1];
    }
    else if (p == 1)
    {
        sinThetaOp = sinThetaO * cos2k[0] + cosThetaO * sin2k[0];
        cosThetaOp = cosThetaO * cos2k[0] - sinThetaO * sin2k[0];
    }
    else if (p == 2)
    {
        sinThetaOp = sinThetaO * cos2k[2] + cosThetaO * sin2k[2];
        cosThetaOp = cosThetaO * cos2k[2] - sinThetaO * sin2k[2];
    }

    float v[4];
    ComputeV(beta_m, v);
    let s = ComputeS(beta_n);

    // Sample the longitudinal angle theta_i from Mp of the chosen lobe.
    let u0 = rand_vec2f(rng);
    let u1 = rand_vec2f(rng);
    let vp = v[p];
    let cosTheta = 1.0 + vp * log(max(u0.x, 1e-5) + (1.0 - u0.x) * exp(-2.0 / vp));
    let sinTheta = SafeSqrt(1.0 - Sqr(cosTheta));
    let cosPhi = cos(2.0 * PI * u0.y);
    let sinThetaI = -cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp;
    let cosThetaI = SafeSqrt(1.0 - Sqr(sinThetaI));

    // Sample the azimuthal offset dphi.
    float dphi;
    if (p < P_MAX)
    {
        dphi = Phi(p, gammaO, gammaT) + SampleTrimmedLogistic(u1.x, s);
    }
    else
    {
        dphi = 2.0 * PI * u1.x;
    }
    let phiI = phiO + dphi;

    let wi = from_fiber(frame, sinThetaI, phiI);

    // Evaluate full BSDF + pdf for the sampled direction (consistent weighting).
    let e = eval_core(wo, wi, frame, sigma_a, beta_m, beta_n, alpha, eta);
    if (e.pdf <= 0.0)
    {
        return result;
    }

    result.wi = wi;
    result.pdf = e.pdf;
    // For hair the cosine factor is folded into Mp (the response is per solid
    // angle about the fiber), so throughput is simply f / pdf.
    result.throughput = e.f / e.pdf;
    return result;
}
