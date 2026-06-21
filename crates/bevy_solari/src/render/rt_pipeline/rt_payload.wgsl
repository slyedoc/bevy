// Shared ray-payload + camera types for the RT-pipeline shaders. The payload is
// passed between raygen and the SBT-selected hit/miss shaders, so its layout
// MUST be identical in every stage — defining it once here (imported by all)
// prevents drift (a mismatch silently corrupts SER hit objects).
#define_import_path bevy_solari::rt_payload

// Per-bounce path state carried across the trace. raygen owns the bounce loop;
// a hit shader fills `emitted` + (if continuing) `attenuation` and the next ray,
// and threads the RNG back out.
struct RtPayload {
    emitted: vec3<f32>,        // radiance contributed at this vertex (sky at miss)
    attenuation: vec3<f32>,    // throughput multiplier for the next segment
    next_origin: vec3<f32>,
    next_direction: vec3<f32>,
    bounce: u32,               // 1 = continue, 0 = terminate (miss / absorb)
    rng: u32,                  // PCG state, advanced by the hit shader's sampling
    // pdf of the BRDF sample that generated the ray INTO this vertex (0 on the
    // primary ray). Threaded out by a hit shader so the NEXT vertex can MIS-weight
    // its emissive against next-event estimation (the BSDF-vs-NEE pair).
    p_bounce: f32,
}

// Shadow / visibility-ray payload — just an occlusion flag. The closest-hit sets
// `occluded = 1` before tracing a `traceRay` toward the light (with
// SKIP_CLOSEST_HIT + TERMINATE_ON_FIRST_HIT, miss index = the shadow miss); the
// dedicated `miss_shadow` program clears it to 0 when the ray reaches the light
// unobstructed. Tiny on purpose — fixed-function traversal keeps the chit lean.
struct ShadowPayload {
    occluded: u32,
}

struct RtCamera {
    inverse_view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    frame: vec4<u32>,          // .x = frame index (RNG seed)
    sky: vec4<f32>,            // .x = environment brightness (cd/m²); .yzw = clear color
}
