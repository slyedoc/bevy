// Primary/bounce miss: the environment (skybox / baked-atmosphere cube) becomes
// the path's terminal radiance — sampled, not marched. Matches the megakernel's
// miss: env cube × brightness, or the clear color when there's no skybox.
enable wgpu_ray_tracing_pipeline;

#import bevy_solari::rt_payload::{RtPayload, RtCamera}

// RT-pipeline-private set (set 1): camera carries sky brightness + clear color;
// the environment cube + sampler are bound here too (miss-only).
@group(1) @binding(1) var<uniform> camera: RtCamera;
@group(1) @binding(2) var environment_map: texture_cube<f32>;
@group(1) @binding(3) var environment_sampler: sampler;

var<incoming_ray_payload> payload: RtPayload;

// Rotate v by quaternion q (xyzw). Identity (0,0,0,1) is an exact no-op.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

@miss
@incoming_payload(payload)
fn miss_primary(@builtin(world_ray_direction) dir: vec3<f32>) {
    let brightness = camera.sky.x;
    // With atmosphere VOLUMES active, a PRIMARY miss is space — raygen's
    // volume march paints the actual sky/limb over it, and sampling the baked
    // cube here would double-count (and smear an in-atmosphere dome across
    // space). Bounce rays (p_bounce ≠ 0) keep the cube as cheap sky ambience.
    let volumes_active = bitcast<u32>(camera.atmo.z) != 0u;
    if volumes_active && payload.p_bounce == 0.0 {
        payload.emitted = camera.sky.yzw;
    } else if brightness > 0.0 {
        // World→bake frame: a spherical-planet atmosphere bakes its cube in a
        // canonical up-=-+Y frame; `sky_frame` re-aims it at the camera's
        // radial up every frame (identity for flat scenes / plain skyboxes).
        let sdir = quat_rotate(camera.sky_frame, dir);
        let sky = textureSampleLevel(environment_map, environment_sampler, sdir, 0.0).rgb;
        payload.emitted = brightness * sky;
    } else {
        payload.emitted = camera.sky.yzw; // flat clear color (no skybox)
    }
    payload.bounce = 0u; // sky terminates the path
}
