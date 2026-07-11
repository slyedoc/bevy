// Gym-only synthetic batch generator, concatenated after nrc_mlp.wgsl.
// Emits encoded inputs into act_out (the layer-0 activation buffer) and an
// analytic radiance target: gaussian emitters × directional lobes, modulated
// by albedo/roughness/normal so every encoded feature carries gradient.

struct GenParams {
    seed: u32,
    batch: u32,
    pad_a: u32,
    pad_b: u32,
}

@group(0) @binding(23) var<uniform> gen: GenParams;

fn pcg4d(p: vec4<u32>) -> vec4<u32> {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v = v ^ (v >> vec4(16u));
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}

fn to_unit(u: u32) -> f32 {
    return f32(u >> 8u) / 16777216.0;
}

fn sphere_dir(a: f32, b: f32) -> vec3<f32> {
    let z = a * 2.0 - 1.0;
    let phi = b * 2.0 * PI;
    let r = sqrt(max(1.0 - z * z, 0.0));
    return vec3(r * cos(phi), r * sin(phi), z);
}

fn gym_target(
    pos: vec3<f32>,
    dir: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    diff_albedo: vec3<f32>,
    spec_albedo: vec3<f32>,
) -> vec3<f32> {
    var centers = array<vec3<f32>, 3>(
        vec3(0.25, 0.5, 0.3), vec3(0.7, 0.2, 0.8), vec3(0.5, 0.85, 0.5));
    var colors = array<vec3<f32>, 3>(
        vec3(4.0, 1.0, 0.2), vec3(0.3, 2.5, 3.0), vec3(1.5, 1.5, 0.4));
    var lobes = array<vec3<f32>, 3>(
        vec3(0.894, 0.447, 0.0), vec3(0.0, 0.707, -0.707), vec3(-0.577, 0.577, 0.577));
    var field = vec3(0.0);
    for (var g = 0; g < 3; g += 1) {
        let d = pos - centers[g];
        let lobe = 0.5 + 0.5 * dot(dir, lobes[g]);
        field += colors[g] * exp(-dot(d, d) * 8.0) * lobe * lobe;
    }
    let spec = pow(max(dot(dir, normal), 0.0), 4.0) * (1.0 - roughness);
    return diff_albedo * (field + vec3(0.05)) + spec_albedo * spec;
}

@compute @workgroup_size(64, 1, 1)
fn gym_gen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sample = gid.x;
    if sample >= gen.batch { return; }
    let ra = pcg4d(vec4(sample, gen.seed, 0x9e3779b9u, 0x85ebca6bu));
    let rb = pcg4d(vec4(sample, gen.seed, 0xc2b2ae35u, 0x27d4eb2fu));
    let rc = pcg4d(vec4(sample, gen.seed, 0x165667b1u, 0xd3a2646cu));

    let pos = vec3(to_unit(ra.x), to_unit(ra.y), to_unit(ra.z));
    let dir = sphere_dir(to_unit(ra.w), to_unit(rb.x));
    let normal = sphere_dir(to_unit(rb.y), to_unit(rb.z));
    let roughness = to_unit(rb.w);
    let diff_albedo = vec3(to_unit(rc.x), to_unit(rc.y), to_unit(rc.z));
    let spec_albedo = vec3(to_unit(rc.w), to_unit(ra.x ^ rb.y), to_unit(ra.y ^ rc.z));

    nrc_encode(pos, dir, normal, roughness, diff_albedo, spec_albedo, sample * WIDTH);

    let radiance = gym_target(pos, dir, normal, roughness, diff_albedo, spec_albedo);
    targets_out[sample * 4u] = radiance.x;
    targets_out[sample * 4u + 1u] = radiance.y;
    targets_out[sample * 4u + 2u] = radiance.z;
    targets_out[sample * 4u + 3u] = 1.0;
}
