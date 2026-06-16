#define_import_path bevy_solari::hair

// Physically-based hair fiber BSDF following Chiang et al. 2016,
// "A Practical and Controllable Hair and Fur Model for Production Path Tracing"
// (as implemented by pbrt-v3/v4 `HairBSDF`).
//
// The fiber is modeled as a dielectric cylinder. A direction is parameterized
// by its longitudinal angle theta (measured from the plane perpendicular to the
// fiber axis) and its azimuthal angle phi (around the axis). Light is scattered
// through a small set of paths, each a "lobe":
//   p = 0 : R   -- reflection off the cuticle.
//   p = 1 : TT  -- transmit in, transmit out (the dominant transmissive lobe).
//   p = 2 : TRT -- transmit in, internal reflect, transmit out.
//   p = 3 : residual -- a single lumped lobe approximating all higher-order paths.
// Each lobe is a product of a longitudinal term Mp(theta) and an azimuthal term
// Np(phi). Mp is a Gaussian-on-a-cylinder (von-Mises-Fisher style) controlled by
// the longitudinal variance v; Np is a logistic distribution of width s centered
// on the perfect-specular azimuth Phi(p). Attenuation Ap(p) carries the Fresnel
// and per-path absorption (Beer-Lambert through the fiber via sigma_a).

#import bevy_solari::pbr::{rand_f, rand_vec2f}

const PI: f32 = 3.14159265358979;
const SQRT_PI_OVER_8: f32 = 0.626657069; // sqrt(PI / 8)
const P_MAX: i32 = 3;                     // R, TT, TRT (residual is index P_MAX)

struct HairSample {
    wi: vec3<f32>,          // sampled incoming direction (next bounce), world space, normalized
    throughput: vec3<f32>,  // bsdf * |cos| / pdf  (already divided by pdf)
    pdf: f32,               // pdf of the sampled direction (0.0 => discard)
}

// ----- small math helpers ----------------------------------------------------

fn Sqr(x: f32) -> f32 {
    return x * x;
}

fn SafeASin(x: f32) -> f32 {
    return asin(clamp(x, -1.0, 1.0));
}

fn SafeSqrt(x: f32) -> f32 {
    return sqrt(max(0.0, x));
}

// pow(base, 20) without a loop-with-no-return; small fixed unrolled chain.
fn Pow3(x: f32) -> f32 {
    return x * x * x;
}

// ----- Bessel I0 and its log, used by the longitudinal Mp term ---------------

fn I0(x: f32) -> f32 {
    var val: f32 = 0.0;
    var x2i: f32 = 1.0;            // x^(2i)
    var ifact: f32 = 1.0;          // i!
    var i4: f32 = 1.0;             // 4^i
    // 10 series terms match pbrt's truncation.
    for (var i: i32 = 0; i < 10; i = i + 1) {
        if (i > 1) {
            ifact = ifact * f32(i);
        }
        val = val + x2i / (i4 * Sqr(ifact));
        x2i = x2i * x * x;
        i4 = i4 * 4.0;
    }
    return val;
}

fn LogI0(x: f32) -> f32 {
    if (x > 12.0) {
        // Asymptotic expansion for large argument (avoids I0 overflow).
        return x + 0.5 * (-log(2.0 * PI) + log(1.0 / x) + 1.0 / (8.0 * x));
    } else {
        return log(I0(x));
    }
}

// ----- Longitudinal scattering Mp --------------------------------------------
// Gaussian detector response of the cone of scattered directions.
fn Mp(cosThetaI: f32, cosThetaO: f32, sinThetaI: f32, sinThetaO: f32, v: f32) -> f32 {
    let a = cosThetaI * cosThetaO / v;
    let b = sinThetaI * sinThetaO / v;
    var mp: f32;
    if (v <= 0.1) {
        mp = exp(LogI0(a) - b - 1.0 / v + 0.6931 + log(1.0 / (2.0 * v)));
    } else {
        mp = (exp(-b) * I0(a)) / (Sinh(1.0 / v) * 2.0 * v);
    }
    return mp;
}

fn Sinh(x: f32) -> f32 {
    return (exp(x) - exp(-x)) * 0.5;
}

// ----- Logistic distribution helpers (for azimuthal Np) ----------------------

fn Logistic(x: f32, s: f32) -> f32 {
    let ax = abs(x);
    return exp(-ax / s) / (s * Sqr(1.0 + exp(-ax / s)));
}

fn LogisticCDF(x: f32, s: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x / s));
}

// Logistic truncated to [-PI, PI], normalized over that interval.
fn TrimmedLogistic(x: f32, s: f32) -> f32 {
    return Logistic(x, s) / (LogisticCDF(PI, s) - LogisticCDF(-PI, s));
}

fn SampleTrimmedLogistic(u: f32, s: f32) -> f32 {
    let lo = LogisticCDF(-PI, s);
    let hi = LogisticCDF(PI, s);
    let k = u * (hi - lo) + lo;
    var x = -s * log(1.0 / k - 1.0);
    return clamp(x, -PI, PI);
}

// ----- Azimuthal scattering Np -----------------------------------------------
// Difference between the ideal specular azimuth Phi(p) and the actual phi,
// wrapped into [-PI, PI], shaped by a trimmed logistic of width s.
fn Phi(p: i32, gammaO: f32, gammaT: f32) -> f32 {
    return 2.0 * f32(p) * gammaT - 2.0 * gammaO + f32(p) * PI;
}

fn Np(phi: f32, p: i32, s: f32, gammaO: f32, gammaT: f32) -> f32 {
    var dphi = phi - Phi(p, gammaO, gammaT);
    // Wrap dphi to [-PI, PI].
    while (dphi > PI) {
        dphi = dphi - 2.0 * PI;
    }
    while (dphi < -PI) {
        dphi = dphi + 2.0 * PI;
    }
    return TrimmedLogistic(dphi, s);
}

// ----- Attenuation Ap --------------------------------------------------------
// Per-lobe color/Fresnel attenuation. Ap[0] is pure surface reflection; the
// transmissive lobes accumulate absorption T = exp(-sigma_a * path-length) and
// the remaining (1-f) Fresnel transmittance factors. Index P_MAX is the residual
// energy lumped into one trailing lobe.
fn Ap(cosThetaO: f32, eta: f32, h: f32, T: vec3<f32>) -> array<vec3<f32>, 4> {
    var ap: array<vec3<f32>, 4>;
    let cosGammaO = SafeSqrt(1.0 - h * h);
    let cosTheta = cosThetaO * cosGammaO;
    let f = FrDielectric(cosTheta, eta);

    ap[0] = vec3<f32>(f);
    ap[1] = Sqr(1.0 - f) * T;
    ap[2] = ap[1] * T * f;
    // Residual: geometric series tail (1-f)^2 * T^2 * f^2 / (1 - T*f).
    let denom = vec3<f32>(1.0) - T * f;
    ap[3] = ap[2] * f * T / max(denom, vec3<f32>(1e-5));
    return ap;
}

// Fresnel reflectance for a dielectric interface (unpolarized).
fn FrDielectric(cosThetaI_in: f32, eta: f32) -> f32 {
    var cosThetaI = clamp(cosThetaI_in, -1.0, 1.0);
    var etaLocal = eta;
    if (cosThetaI < 0.0) {
        etaLocal = 1.0 / eta;
        cosThetaI = -cosThetaI;
    }
    let sin2ThetaI = 1.0 - Sqr(cosThetaI);
    let sin2ThetaT = sin2ThetaI / Sqr(etaLocal);
    if (sin2ThetaT >= 1.0) {
        return 1.0; // total internal reflection
    }
    let cosThetaT = SafeSqrt(1.0 - sin2ThetaT);
    let rParl = (etaLocal * cosThetaI - cosThetaT) / (etaLocal * cosThetaI + cosThetaT);
    let rPerp = (cosThetaI - etaLocal * cosThetaT) / (cosThetaI + etaLocal * cosThetaT);
    return (Sqr(rParl) + Sqr(rPerp)) * 0.5;
}

// ----- sigma_a from a target reflectance (pbrt SigmaAFromReflectance) --------
// Inverts the model so that the supplied `color` is the resulting diffuse albedo
// for the given azimuthal roughness beta_n.
fn SigmaAFromReflectance(c: vec3<f32>, beta_n: f32) -> vec3<f32> {
    let denom = 5.969
        - 0.215 * beta_n
        + 2.532 * Sqr(beta_n)
        - 10.73 * Pow3(beta_n)
        + 5.574 * Sqr(Sqr(beta_n))
        + 0.245 * beta_n * Sqr(Sqr(beta_n));
    let lc = vec3<f32>(log(max(c.x, 1e-4)), log(max(c.y, 1e-4)), log(max(c.z, 1e-4)));
    let s = lc / denom;
    return s * s;
}

// ----- longitudinal variances v[] and azimuthal width s from roughness -------
fn ComputeV(beta_m: f32) -> array<f32, 4> {
    var v: array<f32, 4>;
    // pbrt's exact polynomial: 0.726*b + 0.812*b^2 + 3.7*b^20, squared -> variance.
    let v0 = Sqr(0.726 * beta_m + 0.812 * Sqr(beta_m) + 3.7 * pow(beta_m, 20.0));
    v[0] = v0;
    v[1] = 0.25 * v0;
    v[2] = 4.0 * v0;
    v[3] = v[2];
    return v;
}

fn ComputeS(beta_n: f32) -> f32 {
    // s = SqrtPiOver8 * (0.265*b + 1.194*b^2 + 5.372*b^22)
    return SQRT_PI_OVER_8 * (0.265 * beta_n + 1.194 * Sqr(beta_n) + 5.372 * pow(beta_n, 22.0));
}

// ----- cuticle-tilt sin/cos per lobe (alpha shifts each lobe's cone) ---------
fn ComputeSin2kAlpha(alpha: f32) -> array<f32, 3> {
    var s: array<f32, 3>;
    s[0] = sin(alpha);
    s[1] = 2.0 * s[0] * cos(alpha);                  // sin(2a)
    s[2] = 2.0 * s[1] * (1.0 - 2.0 * Sqr(s[0]));     // sin(4a) via double-angle
    return s;
}
fn ComputeCos2kAlpha(sin2k: array<f32, 3>) -> array<f32, 3> {
    var c: array<f32, 3>;
    c[0] = SafeSqrt(1.0 - Sqr(sin2k[0]));
    c[1] = SafeSqrt(1.0 - Sqr(sin2k[1]));
    c[2] = SafeSqrt(1.0 - Sqr(sin2k[2]));
    return c;
}

// ----- fiber frame -----------------------------------------------------------
// pbrt's hair frame uses the fiber tangent as the "u" axis. A direction d maps
// to (sinTheta = dot(d,u), and phi = atan2(dot(d,w), dot(d,v))) where (v,w) span
// the cross-section plane.
struct FiberFrame {
    u: vec3<f32>, // fiber axis (tangent)
    v: vec3<f32>,
    w: vec3<f32>,
}

fn build_frame(tangent: vec3<f32>) -> FiberFrame {
    let u = normalize(tangent);
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(u.y) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let v = normalize(cross(up, u));
    let w = cross(u, v);
    var f: FiberFrame;
    f.u = u;
    f.v = v;
    f.w = w;
    return f;
}

// Returns (sinTheta, cosTheta, phi) for a world-space direction in the frame.
fn to_fiber(frame: FiberFrame, d: vec3<f32>) -> vec3<f32> {
    let sinTheta = dot(d, frame.u);
    let cosTheta = SafeSqrt(1.0 - Sqr(sinTheta));
    let phi = atan2(dot(d, frame.w), dot(d, frame.v));
    return vec3<f32>(sinTheta, cosTheta, phi);
}

// Rebuild a world direction from (sinTheta, phi) in the frame.
fn from_fiber(frame: FiberFrame, sinTheta: f32, phi: f32) -> vec3<f32> {
    let cosTheta = SafeSqrt(1.0 - Sqr(sinTheta));
    return frame.u * sinTheta
        + frame.v * (cosTheta * cos(phi))
        + frame.w * (cosTheta * sin(phi));
}

// ----- shared evaluation core ------------------------------------------------
// Computes the full BSDF (sum over lobes) and, optionally, the per-lobe ap pdfs.
// `h` is the offset across the fiber width; with no geometric normal available
// we use the standard h = -1 + 2*gammaO mapping is not possible, so we fix h via
// the azimuth of wo (a common real-time simplification: h derived from phi_o).
struct HairEval {
    f: vec3<f32>,
    pdf: f32,
}

fn eval_core(
    wo: vec3<f32>, wi: vec3<f32>, frame: FiberFrame,
    sigma_a: vec3<f32>, beta_m: f32, beta_n: f32, alpha: f32, eta: f32,
) -> HairEval {
    let fo = to_fiber(frame, wo);
    let fi = to_fiber(frame, wi);
    let sinThetaO = fo.x;
    let cosThetaO = fo.y;
    let phiO = fo.z;
    let sinThetaI = fi.x;
    let cosThetaI = fi.y;
    let phiI = fi.z;

    // Offset h across the fiber: derive from the azimuth of wo so the geometry is
    // self-consistent without an explicit surface normal. h in [-1, 1].
    let h = clamp(sin(phiO) * cosThetaO, -0.9999, 0.9999);

    let gammaO = SafeASin(h);

    // Refracted ray geometry (Bravais / modified IOR for the tilted fiber).
    let sinThetaT = sinThetaO / eta;
    let cosThetaT = SafeSqrt(1.0 - Sqr(sinThetaT));
    let etap = SafeSqrt(Sqr(eta) - Sqr(sinThetaO)) / max(cosThetaO, 1e-4);
    let sinGammaT = h / etap;
    let cosGammaT = SafeSqrt(1.0 - Sqr(sinGammaT));
    let gammaT = SafeASin(sinGammaT);

    // Absorption along the transmission path through the cross-section. `sigma_a`
    // is supplied directly (from melanin concentrations), not derived from a color.
    let T = vec3<f32>(
        exp(-sigma_a.x * 2.0 * cosGammaT / max(cosThetaT, 1e-4)),
        exp(-sigma_a.y * 2.0 * cosGammaT / max(cosThetaT, 1e-4)),
        exp(-sigma_a.z * 2.0 * cosGammaT / max(cosThetaT, 1e-4)),
    );

    var ap = Ap(cosThetaO, eta, h, T);    // var: dynamically indexed by lobe p
    var v = ComputeV(beta_m);             // var: dynamically indexed by lobe p
    let s = ComputeS(beta_n);
    let sin2k = ComputeSin2kAlpha(alpha);
    let cos2k = ComputeCos2kAlpha(sin2k);

    let phi = phiI - phiO;

    // ap pdf weights (luminance of each lobe's attenuation), for sampling/MIS.
    var apPdf: array<f32, 4>;
    var sumY: f32 = 0.0;
    for (var p: i32 = 0; p < 4; p = p + 1) {
        let y = ap[p].x * 0.299 + ap[p].y * 0.587 + ap[p].z * 0.114;
        apPdf[p] = y;
        sumY = sumY + y;
    }
    let invSum = select(0.0, 1.0 / sumY, sumY > 0.0);
    for (var p: i32 = 0; p < 4; p = p + 1) {
        apPdf[p] = apPdf[p] * invSum;
    }

    var fsum = vec3<f32>(0.0);
    var pdf: f32 = 0.0;

    for (var p: i32 = 0; p < P_MAX; p = p + 1) {
        // Apply cuticle tilt: rotate the longitudinal cone for this lobe.
        var sinThetaOp: f32;
        var cosThetaOp: f32;
        if (p == 0) {
            sinThetaOp = sinThetaO * cos2k[1] - cosThetaO * sin2k[1];
            cosThetaOp = cosThetaO * cos2k[1] + sinThetaO * sin2k[1];
        } else if (p == 1) {
            sinThetaOp = sinThetaO * cos2k[0] + cosThetaO * sin2k[0];
            cosThetaOp = cosThetaO * cos2k[0] - sinThetaO * sin2k[0];
        } else {
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

    var result: HairEval;
    result.f = fsum;
    result.pdf = pdf;
    return result;
}

// ----- public evaluation -----------------------------------------------------
fn eval_hair_bsdf(
    wo: vec3<f32>, wi: vec3<f32>, tangent: vec3<f32>,
    sigma_a: vec3<f32>, beta_m: f32, beta_n: f32, alpha: f32, eta: f32,
) -> vec3<f32> {
    let frame = build_frame(tangent);
    let e = eval_core(wo, wi, frame, sigma_a, beta_m, beta_n, alpha, eta);
    return e.f;
}

fn pdf_hair_bsdf(
    wo: vec3<f32>, wi: vec3<f32>, tangent: vec3<f32>,
    beta_m: f32, beta_n: f32, alpha: f32, eta: f32,
) -> f32 {
    let frame = build_frame(tangent);
    // Absorption barely shifts the pdf shape (only ap luminance ratios); use zero
    // (T = 1) so the lobe-selection weights track the sampler's choice.
    let e = eval_core(wo, wi, frame, vec3<f32>(0.0), beta_m, beta_n, alpha, eta);
    return e.pdf;
}

// ----- importance sampling ---------------------------------------------------
fn sample_hair_bsdf(
    wo: vec3<f32>,
    tangent: vec3<f32>,
    sigma_a: vec3<f32>,
    beta_m: f32,
    beta_n: f32,
    alpha: f32,
    eta: f32,
    rng: ptr<function, u32>,
) -> HairSample {
    var out: HairSample;
    out.wi = vec3<f32>(0.0, 0.0, 1.0);
    out.throughput = vec3<f32>(0.0);
    out.pdf = 0.0;

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

    let T = vec3<f32>(
        exp(-sigma_a.x * 2.0 * cosGammaT / max(cosThetaT, 1e-4)),
        exp(-sigma_a.y * 2.0 * cosGammaT / max(cosThetaT, 1e-4)),
        exp(-sigma_a.z * 2.0 * cosGammaT / max(cosThetaT, 1e-4)),
    );
    var ap = Ap(cosThetaO, eta, h, T);    // var: dynamically indexed by lobe p

    // Build the per-lobe selection cdf from ap luminance.
    var apPdf: array<f32, 4>;
    var sumY: f32 = 0.0;
    for (var p: i32 = 0; p < 4; p = p + 1) {
        let y = ap[p].x * 0.299 + ap[p].y * 0.587 + ap[p].z * 0.114;
        apPdf[p] = y;
        sumY = sumY + y;
    }
    if (sumY <= 0.0) {
        return out;
    }
    let invSum = 1.0 / sumY;
    for (var p: i32 = 0; p < 4; p = p + 1) {
        apPdf[p] = apPdf[p] * invSum;
    }

    // Choose a lobe.
    let uLobe = rand_f(rng);
    var p: i32 = 0;
    var cdf: f32 = apPdf[0];
    // Linear scan over the 4 lobes.
    for (var k: i32 = 0; k < P_MAX; k = k + 1) {
        if (uLobe < cdf) {
            p = k;
            break;
        }
        p = k + 1;
        cdf = cdf + apPdf[k + 1];
    }

    let sin2k = ComputeSin2kAlpha(alpha);
    let cos2k = ComputeCos2kAlpha(sin2k);

    // Tilt-adjusted outgoing longitudinal angle for the chosen lobe.
    var sinThetaOp: f32 = sinThetaO;
    var cosThetaOp: f32 = cosThetaO;
    if (p == 0) {
        sinThetaOp = sinThetaO * cos2k[1] - cosThetaO * sin2k[1];
        cosThetaOp = cosThetaO * cos2k[1] + sinThetaO * sin2k[1];
    } else if (p == 1) {
        sinThetaOp = sinThetaO * cos2k[0] + cosThetaO * sin2k[0];
        cosThetaOp = cosThetaO * cos2k[0] - sinThetaO * sin2k[0];
    } else if (p == 2) {
        sinThetaOp = sinThetaO * cos2k[2] + cosThetaO * sin2k[2];
        cosThetaOp = cosThetaO * cos2k[2] - sinThetaO * sin2k[2];
    }

    var v = ComputeV(beta_m);             // var: dynamically indexed by lobe p
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
    var dphi: f32;
    if (p < P_MAX) {
        dphi = Phi(p, gammaO, gammaT) + SampleTrimmedLogistic(u1.x, s);
    } else {
        dphi = 2.0 * PI * u1.x;
    }
    let phiI = phiO + dphi;

    let wi = from_fiber(frame, sinThetaI, phiI);

    // Evaluate full BSDF + pdf for the sampled direction (consistent weighting).
    let e = eval_core(wo, wi, frame, sigma_a, beta_m, beta_n, alpha, eta);
    if (e.pdf <= 0.0) {
        return out;
    }

    out.wi = wi;
    out.pdf = e.pdf;
    // For hair the cosine factor is folded into Mp (the response is per solid
    // angle about the fiber), so throughput is simply f / pdf.
    out.throughput = e.f / e.pdf;
    return out;
}
