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

@miss
@incoming_payload(payload)
fn miss_primary(@builtin(world_ray_direction) dir: vec3<f32>) {
    let brightness = camera.sky.x;
    if brightness > 0.0 {
        let sky = textureSampleLevel(environment_map, environment_sampler, dir, 0.0).rgb;
        payload.emitted = brightness * sky;
    } else {
        payload.emitted = camera.sky.yzw; // flat clear color (no skybox)
    }
    payload.bounce = 0u; // sky terminates the path
}
