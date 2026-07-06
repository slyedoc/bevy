// Shared ray-payload + camera types for the RT-pipeline shaders. The payload is
// passed between raygen and the SBT-selected hit/miss shaders, so its layout
// MUST be identical in every stage — defining it once here (imported by all)
// prevents drift (a mismatch silently corrupts SER hit objects).
#define_import_path bevy_solari::rt_payload

// Per-bounce path state carried across the trace. raygen owns the bounce loop;
// a hit shader fills `emitted` + (if continuing) `attenuation` and the next ray,
// and threads the RNG back out.
struct RtPayload {
    // Any-hit invocation counter for the OMM-effectiveness debug heatmap. MUST be
    // the first field: `ahit_alpha` aliases the payload's first word (it runs for
    // both `RtPayload` and `ShadowPayload` rays), so keeping the counter first in
    // BOTH lets the any-hit increment it safely whichever ray invoked it. raygen
    // resets it before each trace and sums across bounces.
    anyhit_count: u32,
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
    // Geometry-debug views (cluster / triangle color). The closest-hit writes the
    // global cluster id + cluster-local primitive index of the hit; raygen captures
    // them on the PRIMARY hit and hashes them to a flat color. `hit_cluster` carries
    // a sentinel (set by raygen before the trace) when the primary ray missed.
    hit_cluster: u32,
    hit_primitive: u32,
    // Normal-facing debug view (frame.z == 5): primary hit's SHADING normal
    // (pre-bend, world space), oct-packed to one word. raygen colors front-vs-back
    // facing to expose inverted / back-wound normals.
    hit_normal_oct: u32,
    // Same view: the GEOMETRIC (winding/position) normal, so raygen can tell a
    // back-wound triangle (red) from a merely bad vertex normal (yellow).
    hit_geo_normal_oct: u32,
#ifdef SOLARI_DLSS
    // Pixel index (row-major `y*width + x`) the primary-hit closest-hit writes its
    // ray-reconstruction G-buffer to, or `NO_GBUFFER` on secondary bounces (no
    // guide written there). The single word the DLSS guide costs the payload — the
    // surface attributes themselves go straight to the G-buffer storage buffers
    // from the closest-hit (chit-direct), never riding the payload across bounces.
    gbuffer_pixel: u32,
#endif
}

// Shadow / visibility-ray payload — just an occlusion flag. The closest-hit sets
// `occluded = 1` before tracing a `traceRay` toward the light (with
// SKIP_CLOSEST_HIT + TERMINATE_ON_FIRST_HIT, miss index = the shadow miss); the
// dedicated `miss_shadow` program clears it to 0 when the ray reaches the light
// unobstructed. Tiny on purpose — fixed-function traversal keeps the chit lean.
struct ShadowPayload {
    // First word mirrors `RtPayload.anyhit_count` so `ahit_alpha`'s single-word
    // alias targets the counter whichever ray type invoked it (shadow-ray any-hits
    // land here and are discarded — raygen only reads the primary/bounce count).
    anyhit_count: u32,
    occluded: u32,
}

struct RtCamera {
    inverse_view_proj: mat4x4<f32>,
    view_from_world: mat4x4<f32>,        // DLSS guide: view-space linear depth
    clip_from_world: mat4x4<f32>,        // DLSS motion vectors: current (unjittered)
    prev_clip_from_world: mat4x4<f32>,   // DLSS motion vectors: previous (unjittered)
    camera_position: vec4<f32>,
    frame: vec4<u32>,          // .x = frame index (RNG seed); .y = SER material-hint bits; .z = debug view
    sky: vec4<f32>,            // .x = environment brightness (cd/m²); .yzw = clear color
    jitter: vec4<f32>,         // .xy = sub-pixel camera jitter (pixels); .zw = debug-heatmap colormap params
    misc: vec4<f32>,           // .x = time (s, wrapped); .y = pixel ray-cone tan (footprint LOD); .zw reserved
    sky_frame: vec4<f32>,      // world→bake sky quaternion (xyzw); identity for flat scenes/skyboxes
    atmo: vec4<f32>,           // .xy = volume-buffer device address (lo/hi bits); .z = volume count
}
