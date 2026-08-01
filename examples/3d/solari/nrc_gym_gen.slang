// Gym-only synthetic batch generator. Emits encoded inputs into acts (the
// layer-0 activation buffer) and an analytic radiance target: gaussian
// emitters × directional lobes, modulated by albedo/roughness/normal so every
// encoded feature carries gradient. The encode comes from the crate's
// nrc_mlp module — the exact functions production uses. Bindings are
// contiguous from 0 (passthrough SPIR-V requirement — see nrc_mlp.slang).
//
// Regenerate SPIR-V (from examples/3d/solari/):
//   slangc nrc_gym_gen.slang -I ../../../crates/bevy_solari/src/nrc -target spirv -entry gym_gen -o nrc_gym_gen.spv
import nrc_mlp;

struct GenParams
{
    uint seed;
    uint batch;
    uint pad_a;
    uint pad_b;
}

[[vk::binding(0, 0)]] RWStructuredBuffer<half> act_out;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> targets_out;
[[vk::binding(2, 0)]] ConstantBuffer<GenParams> gen;

uint4 pcg4d(uint4 p)
{
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v = v ^ (v >> 16);
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}

float to_unit(uint u)
{
    return float(u >> 8) / 16777216.0;
}

float3 sphere_dir(float a, float b)
{
    let z = a * 2.0 - 1.0;
    let phi = b * 2.0 * PI;
    let r = sqrt(max(1.0 - z * z, 0.0));
    return float3(r * cos(phi), r * sin(phi), z);
}

float3 gym_target(
    float3 pos,
    float3 dir,
    float3 normal,
    float roughness,
    float3 diff_albedo,
    float3 spec_albedo)
{
    float3 centers[3] = {
        float3(0.25, 0.5, 0.3), float3(0.7, 0.2, 0.8), float3(0.5, 0.85, 0.5)
    };
    float3 colors[3] = {
        float3(4.0, 1.0, 0.2), float3(0.3, 2.5, 3.0), float3(1.5, 1.5, 0.4)
    };
    float3 lobes[3] = {
        float3(0.894, 0.447, 0.0), float3(0.0, 0.707, -0.707), float3(-0.577, 0.577, 0.577)
    };
    var field = float3(0.0);
    for (int g = 0; g < 3; g++)
    {
        let d = pos - centers[g];
        let lobe = 0.5 + 0.5 * dot(dir, lobes[g]);
        field += colors[g] * exp(-dot(d, d) * 8.0) * lobe * lobe;
    }
    let spec = pow(max(dot(dir, normal), 0.0), 4.0) * (1.0 - roughness);
    return diff_albedo * (field + float3(0.05)) + spec_albedo * spec;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gym_gen(uint3 gid: SV_DispatchThreadID)
{
    let sample = gid.x;
    if (sample >= gen.batch)
    {
        return;
    }
    let ra = pcg4d(uint4(sample, gen.seed, 0x9e3779b9, 0x85ebca6b));
    let rb = pcg4d(uint4(sample, gen.seed, 0xc2b2ae35, 0x27d4eb2f));
    let rc = pcg4d(uint4(sample, gen.seed, 0x165667b1, 0xd3a2646c));

    let pos = float3(to_unit(ra.x), to_unit(ra.y), to_unit(ra.z));
    let dir = sphere_dir(to_unit(ra.w), to_unit(rb.x));
    let normal = sphere_dir(to_unit(rb.y), to_unit(rb.z));
    let roughness = to_unit(rb.w);
    let diff_albedo = float3(to_unit(rc.x), to_unit(rc.y), to_unit(rc.z));
    let spec_albedo = float3(to_unit(rc.w), to_unit(ra.x ^ rb.y), to_unit(ra.y ^ rc.z));

    nrc_encode(act_out, pos, dir, normal, roughness, diff_albedo, spec_albedo, sample * WIDTH);

    let radiance = gym_target(pos, dir, normal, roughness, diff_albedo, spec_albedo);
    targets_out[sample * 4] = radiance.x;
    targets_out[sample * 4 + 1] = radiance.y;
    targets_out[sample * 4 + 2] = radiance.z;
    targets_out[sample * 4 + 3] = 1.0;
}
