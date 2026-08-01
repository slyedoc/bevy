// Primary/bounce miss: the environment (skybox / baked-atmosphere cube) becomes
// the path's terminal radiance — sampled, not marched. Env cube × brightness,
// or the clear color when there's no skybox.
//
// UNLIKE the other RT stages this is NOT precompiled: it composes the
// runtime-swappable `custom_sky` module (`SolariSky::Shader`), so the pipeline
// build compiles it through `gpu/slang.rs` (libslang) with the live module
// source. `rt_shaders_compile` covers the default composition headlessly.
import rt_payload;
import custom_sky;

// RT-pipeline-private set (set 1): camera carries sky brightness + clear color;
// the environment cube + sampler are bound here too (miss-only).
[[vk::binding(1, 1)]] ConstantBuffer<RtCamera> camera;
[[vk::binding(2, 1)]] TextureCube<float4> environment_map;
[[vk::binding(3, 1)]] SamplerState environment_sampler;

// Rotate v by quaternion q (xyzw). Identity (0,0,0,1) is an exact no-op.
float3 quat_rotate(float4 q, float3 v)
{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

[shader("miss")]
void miss_primary(inout RtPayload payload)
{
    let dir = WorldRayDirection();
    let brightness = camera.sky.x;
    // With atmosphere VOLUMES active, a PRIMARY miss is space — raygen's
    // volume march paints the actual sky/limb over it, and sampling the baked
    // cube here would double-count (and smear an in-atmosphere dome across
    // space). Bounce rays (p_bounce ≠ 0) keep the cube as cheap sky ambience.
    let volumes_active = asuint(camera.atmo.z) != 0;
    if (volumes_active && payload.p_bounce == 0.0)
    {
        payload.emitted = camera.sky.yzw;
    }
    else if (brightness < 0.0)
    {
        // `SolariSky::Procedural` / `SolariSky::Shader`: the composed
        // `custom_sky` module evaluates the sky in physical radiance.
        payload.emitted = sample_custom_sky(dir);
    }
    else if (brightness > 0.0)
    {
        // World→bake frame: a spherical-planet atmosphere bakes its cube in a
        // canonical up-=-+Y frame; `sky_frame` re-aims it at the camera's
        // radial up every frame (identity for flat scenes / plain skyboxes).
        let sdir = quat_rotate(camera.sky_frame, dir);
        let sky = environment_map.SampleLevel(environment_sampler, sdir, 0.0).rgb;
        payload.emitted = brightness * sky;
    }
    else
    {
        payload.emitted = camera.sky.yzw; // flat clear color (no skybox)
    }
    payload.bounce = 0; // sky terminates the path
}
