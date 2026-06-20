// Glass/transmissive closest-hit — a SEPARATE SBT program from `chit_opaque`
// (HIT_GROUP_GLASS), the multi-material win. v1 stand-in: a fresnel-tinted ray
// that mostly continues forward (refraction-ish), so glass reads as see-through.
// The real nested-dielectric BSDF (reusing `brdf::sample_glass_bsdf`) lands with
// the shading port.
enable wgpu_ray_tracing_pipeline;

#import bevy_solari::rt_payload::RtPayload

var<incoming_ray_payload> payload: RtPayload;

@closest_hit
@incoming_payload(payload)
fn chit_glass(
    @builtin(world_ray_origin) ro: vec3<f32>,
    @builtin(world_ray_direction) rd: vec3<f32>,
    @builtin(ray_t_current_max) t: f32,
) {
    let hit = ro + rd * t;
    let fresnel = pow(1.0 - abs(rd.y), 5.0);

    payload.emitted = vec3<f32>(0.0);
    payload.attenuation = mix(vec3<f32>(0.7, 0.85, 0.95), vec3<f32>(1.0), fresnel);
    // Continue mostly forward through the surface (stand-in refraction).
    payload.next_origin = hit + rd * 0.001;
    payload.next_direction = rd;
    payload.bounce = 1u;
}
