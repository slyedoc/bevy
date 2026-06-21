// Shadow-ray miss program. A visibility ray traced from a closest-hit toward a
// light (with RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_TERMINATE_ON_FIRST_HIT)
// invokes NO closest-hit; it either hits an occluder (terminates, no shader) or
// reaches the light unobstructed and lands here. So "this program ran" == "the
// light is visible": clear the occlusion flag. This is the NVIDIA-canonical DI
// visibility test — the traversal lives in the fixed-function unit, not in the
// closest-hit's registers.
enable wgpu_ray_tracing_pipeline;

#import bevy_solari::rt_payload::ShadowPayload

var<incoming_ray_payload> payload: ShadowPayload;

@miss
@incoming_payload(payload)
fn miss_shadow() {
    payload.occluded = 0u;
}
