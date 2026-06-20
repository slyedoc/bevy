// Hair closest-hit — a SEPARATE SBT program (HIT_GROUP_HAIR), routed for hair
// instances in `ptlas_hair_write.wgsl`. v1 stand-in: a warm absorbing tone that
// terminates the path (hair is dense). The real Chiang fiber BSDF (reusing
// `hair`/`hair_shade`) lands with the shading port.
enable wgpu_ray_tracing_pipeline;

#import bevy_solari::rt_payload::RtPayload

var<incoming_ray_payload> payload: RtPayload;

@closest_hit
@incoming_payload(payload)
fn chit_hair(@builtin(instance_custom_data) instance: u32) {
    payload.emitted = vec3<f32>(0.45, 0.30, 0.20);
    payload.bounce = 0u;
}
