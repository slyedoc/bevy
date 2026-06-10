#define_import_path bevy_solari::pbr

// Pure helper functions vendored from bevy_pbr's shader libraries so the
// ray-tracing shaders compose with `PbrPlugin` disabled.
//
// bevy_solari traces everything and runs without PbrPlugin's raster stack —
// which means `bevy_pbr::{utils, lighting, pbr_functions}` are never registered
// as shader libraries (they're loaded in `PbrPlugin::build`). The RT shaders only
// use a handful of *pure* math functions from them, copied verbatim here. Keep in
// sync with the originals:
//   - render/utils.wgsl         (rand_*, sample_cosine/uniform_hemisphere, sample_disk)
//   - render/pbr_lighting.wgsl  (D_GGX, V_SmithGGXCorrelated, specular_multiscatter, perceptualRoughnessToRoughness)
//   - render/pbr_functions.wgsl (calculate_tbn_mikktspace, calculate_F0*, calculate_diffuse_color)
//
// `orthonormalize` stays imported from `bevy_render::maths` (registered by
// `RenderPlugin`, always present — not a PbrPlugin dependency).

#import bevy_render::maths::orthonormalize

const PI: f32 = 3.14159265358979;
const PI_2: f32 = 6.28318530717959;

// ---- bevy_pbr::utils ----

// PCG hash RNG. https://www.pcg-random.org
fn rand_u(state: ptr<function, u32>) -> u32 {
    *state = *state * 747796405u + 2891336453u;
    let word = ((*state >> ((*state >> 28u) + 4u)) ^ *state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f(state: ptr<function, u32>) -> f32 {
    *state = *state * 747796405u + 2891336453u;
    let word = ((*state >> ((*state >> 28u) + 4u)) ^ *state) * 277803737u;
    return f32((word >> 22u) ^ word) * bitcast<f32>(0x2f800004u);
}

fn rand_vec2f(state: ptr<function, u32>) -> vec2<f32> {
    return vec2(rand_f(state), rand_f(state));
}

fn rand_range_u(n: u32, state: ptr<function, u32>) -> u32 {
    return rand_u(state) % n;
}

// https://www.realtimerendering.com/raytracinggems/
fn sample_cosine_hemisphere(normal: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let cos_theta = 1.0 - 2.0 * rand_f(rng);
    let phi = PI_2 * rand_f(rng);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let direction = normal + vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let len_sq = dot(direction, direction);
    if len_sq < 1e-8 { return normal; }
    return direction * inverseSqrt(len_sq);
}

fn sample_uniform_hemisphere(normal: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let cos_theta = rand_f(rng);
    let phi = PI_2 * rand_f(rng);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let x = sin_theta * cos(phi);
    let y = sin_theta * sin(phi);
    let z = cos_theta;
    return orthonormalize(normal) * vec3(x, y, z);
}

fn uniform_hemisphere_inverse_pdf() -> f32 {
    return PI_2;
}

fn sample_disk(disk_radius: f32, rng: ptr<function, u32>) -> vec2<f32> {
    let ab = 2.0 * rand_vec2f(rng) - 1.0;
    let a = ab.x;
    var b = ab.y;
    if (b == 0.0) { b = 1.0; }

    var phi: f32;
    var r: f32;
    if (a * a > b * b) {
        r = disk_radius * a;
        phi = (PI / 4.0) * (b / a);
    } else {
        r = disk_radius * b;
        phi = (PI / 2.0) - (PI / 4.0) * (a / b);
    }

    let x = r * cos(phi);
    let y = r * sin(phi);
    return vec2(x, y);
}

// ---- bevy_pbr::lighting ----

fn D_GGX(roughness: f32, NdotH: f32) -> f32 {
    let oneMinusNdotHSquared = 1.0 - NdotH * NdotH;
    let a = NdotH * roughness;
    let k = roughness / (oneMinusNdotHSquared + a * a);
    let d = k * k * (1.0 / PI);
    return d;
}

fn V_SmithGGXCorrelated(roughness: f32, NdotV: f32, NdotL: f32) -> f32 {
    let a2 = roughness * roughness;
    let lambdaV = NdotL * sqrt((NdotV - a2 * NdotV) * NdotV + a2);
    let lambdaL = NdotV * sqrt((NdotL - a2 * NdotL) * NdotL + a2);
    let v = 0.5 / (lambdaV + lambdaL);
    return v;
}

// Filament multiscattering specular approximation.
fn specular_multiscatter(
    D: f32,
    V: f32,
    F: vec3<f32>,
    F0: vec3<f32>,
    F_ab: vec2<f32>,
    specular_intensity: f32,
) -> vec3<f32> {
    var Fr = (specular_intensity * D * V) * F;
    Fr *= 1.0 + F0 * (1.0 / (F_ab.x + F_ab.y) - 1.0);
    return Fr;
}

fn perceptualRoughnessToRoughness(perceptualRoughness: f32) -> f32 {
    let clampedPerceptualRoughness = clamp(perceptualRoughness, 0.089, 1.0);
    return clampedPerceptualRoughness * clampedPerceptualRoughness;
}

// ---- bevy_pbr::pbr_functions ----

// TBN per mikktspace (http://www.mikktspace.com/). Columns T, B, N.
fn calculate_tbn_mikktspace(world_normal: vec3<f32>, world_tangent: vec4<f32>) -> mat3x3<f32> {
    let N: vec3<f32> = world_normal;
    let T: vec3<f32> = world_tangent.xyz;
    let B: vec3<f32> = world_tangent.w * cross(N, T);
    return mat3x3(T, B, N);
}

// Remap [0,1] reflectance to F0 for dielectrics.
fn calculate_F0_dielectric(reflectance: vec3<f32>) -> vec3<f32> {
    return 0.16 * reflectance * reflectance;
}

fn calculate_F0(base_color: vec3<f32>, metallic: f32, reflectance: vec3<f32>) -> vec3<f32> {
    return mix(calculate_F0_dielectric(reflectance), base_color, metallic);
}

fn calculate_diffuse_color(
    base_color: vec3<f32>,
    metallic: f32,
    specular_transmission: f32,
    diffuse_transmission: f32,
) -> vec3<f32> {
    return base_color * (1.0 - metallic) * (1.0 - specular_transmission) *
        (1.0 - diffuse_transmission);
}
