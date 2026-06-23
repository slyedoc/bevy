// Ray-generation: the path-tracer driver. Owns the bounce loop and the
// raygen-resident volume/traversal effects (fog extinction + black-hole ray
// bending) — these operate *between* traceRay calls, so they're here, not in any
// hit shader (Phase 4). Each bounce's surface response comes from the
// SBT-selected closest-hit (opaque/glass/hair), which fills the payload with the
// emitted radiance + attenuation + the next ray. v1 fidelity is stand-in (the
// hit shaders don't yet resolve real geometry/materials); the structure is real.
enable wgpu_ray_tracing_pipeline;

// The scene TLAS lives in the shared scene bind group (set 0) — imported so the
// pipeline layout matches the wgpu-built scene bind group bound at trace time.
#import bevy_solari::scene_bindings::tlas
#import bevy_solari::rt_payload::{RtPayload, RtCamera}
#import bevy_solari::pbr::rand_f

// Rec. 709 luminance, for Russian-roulette survival probability.
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Hard path-length cap; Russian roulette terminates almost every path far sooner.
const MAX_BOUNCES: u32 = 32u;
// Stand-in fog: per-unit-length extinction applied over each ray segment.
const FOG_DENSITY: f32 = 0.0;
// Stand-in black hole: a point mass that bends rays each step; capture radius
// terminates the path black. Disabled by default (strength 0).
const BH_STRENGTH: f32 = 0.0;
const BH_CENTER: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const BH_CAPTURE_RADIUS: f32 = 0.5;

// RT-pipeline-private set (set 1): per-pixel output buffer + camera. The scene
// columns bind group occupies set 2 (unused by raygen, present for the chits).
@group(1) @binding(0) var<storage, read_write> output: array<vec4<f32>>;
@group(1) @binding(1) var<uniform> camera: RtCamera;
#ifdef SOLARI_DLSS
// DLSS Ray Reconstruction guide G-buffer (chit-direct): the closest-hit writes the
// primary hit's packed surface attrs here; raygen clears each pixel to the sky/miss
// default first. Packed normal.xyz+roughness, diffuse.xyz+depth,
// specular.xyz+hit-distance (the `.w` depth/hit-distance slots land in a later phase).
@group(1) @binding(5) var<storage, read_write> gbuffer_normal_roughness: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> gbuffer_diffuse: array<vec4<f32>>;
@group(1) @binding(7) var<storage, read_write> gbuffer_specular: array<vec4<f32>>;
@group(1) @binding(8) var<storage, read_write> gbuffer_motion: array<vec4<f32>>;
// Sentinel pixel index: this bounce writes no guide (set on every non-primary bounce).
const NO_GBUFFER: u32 = 0xffffffffu;
#endif

var<ray_payload> payload: RtPayload;

@ray_generation
fn raygen(
    @builtin(ray_invocation_id) id: vec3<u32>,
    @builtin(num_ray_invocations) dims: vec3<u32>,
) {
    // Sub-pixel camera jitter for DLSS temporal accumulation (zero on the non-DLSS
    // path, so this is a no-op there).
    let pixel = vec2<f32>(id.xy) + 0.5 + camera.jitter.xy;
    let ndc = (pixel / vec2<f32>(dims.xy)) * 2.0 - 1.0;
    let far = camera.inverse_view_proj * vec4<f32>(ndc.x, -ndc.y, 1.0, 1.0);
    var origin = camera.camera_position.xyz;
    var direction = normalize(far.xyz / far.w - origin);

    var radiance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var captured = false;
    // pdf of the BRDF sample that produced this segment (0 on the primary ray),
    // threaded into the hit shader so it can MIS-weight its emissive vs NEE.
    var p_bounce = 0.0;
    // Per-pixel RNG, seeded by (pixel, frame) and threaded through the payload so
    // each hit shader's sampling advances the same stream.
    let pixel_index = id.x + id.y * dims.x;
    var rng = pixel_index + camera.frame.x * 5782582u;

#ifdef SOLARI_DLSS
    // Default this pixel's guide to "no surface" (sky/miss); a primary hit overwrites
    // it in the closest-hit. Zero normal + roughness 1 + zero albedo is the RR sky
    // convention; the `.w` depth/hit-distance slots are filled in a later phase.
    gbuffer_normal_roughness[pixel_index] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    gbuffer_diffuse[pixel_index] = vec4<f32>(0.0);
    gbuffer_specular[pixel_index] = vec4<f32>(0.0);
    gbuffer_motion[pixel_index] = vec4<f32>(0.0);
#endif

    for (var bounce = 0u; bounce < MAX_BOUNCES; bounce += 1u) {
        // Black-hole geodesic (stand-in): bend the ray toward the mass and
        // terminate if it crosses the capture radius.
        if BH_STRENGTH > 0.0 {
            let to_center = BH_CENTER - origin;
            let dist = length(to_center);
            if dist < BH_CAPTURE_RADIUS {
                captured = true;
                break;
            }
            direction = normalize(direction + (BH_STRENGTH / (dist * dist)) * normalize(to_center));
        }

        payload.emitted = vec3<f32>(0.0);
        payload.attenuation = vec3<f32>(0.0);
        payload.next_origin = origin;
        payload.next_direction = direction;
        payload.bounce = 0u;
        payload.rng = rng;
        payload.p_bounce = p_bounce;
#ifdef SOLARI_DLSS
        // Only the primary hit produces the visible guide; later bounces pass the
        // sentinel so their closest-hit leaves the G-buffer untouched.
        payload.gbuffer_pixel = select(NO_GBUFFER, pixel_index, bounce == 0u);
#endif
        // Shader Execution Reordering: trace into a hit object, regroup the warp
        // by MATERIAL, then run the selected closest-hit. With per-material SBT
        // records (instance_contribution_to_hit_group_index = material slot), the
        // hit object's SBT record index IS the material id — read it back and feed
        // it as an explicit coherence hint. Every record runs the same opaque
        // closest-hit, so the default `reorderThread(hit)` (which keys on the
        // shader to run) would NOT separate materials; the explicit hint does, and
        // it scales — huge scenes reuse a few materials across millions of
        // instances, so this packs each warp with one material → uniform
        // `materials[id]` + texture-array fetches, no descriptor divergence.
        // `camera.frame.y` carries ceil(log2(material_count)) hint bits.
        var hit: hit_object;
        hitObjectTraceRay(
            &hit,
            tlas,
            RayDesc(RAY_FLAG_NONE, 0xffu, 0.001, 1.0e9, origin, direction),
            &payload,
        );
        let material_hint = hitObjectGetSbtRecordIndex(&hit);
        reorderThread(&hit, material_hint, camera.frame.y);
        hitObjectExecuteShader(&hit, &payload);
        rng = payload.rng;

        // Fog extinction (stand-in): attenuate over the traversed segment.
        if FOG_DENSITY > 0.0 {
            let seg = length(payload.next_origin - origin);
            throughput *= exp(-FOG_DENSITY * seg);
        }

        radiance += throughput * payload.emitted;
        if payload.bounce == 0u {
            break;
        }
        throughput *= payload.attenuation;
        origin = payload.next_origin;
        direction = payload.next_direction;
        p_bounce = payload.p_bounce;

        // Russian roulette: survival capped below 1 (unbiased — the ÷p
        // compensates) so even lossless paths terminate.
        let p = min(luminance(throughput), 0.95);
        if rand_f(&rng) > p {
            break;
        }
        throughput /= p;
    }

    // Camera exposure (matches the megakernel's `radiance *= view.exposure`); the
    // physical sky/light radiance is otherwise far too bright. `.w` of the camera
    // position carries the exposure.
    var final_color = select(radiance, vec3<f32>(0.0), captured);
    final_color *= camera.camera_position.w;
    let index = id.y * dims.x + id.x;
    output[index] = vec4<f32>(final_color, 1.0);
}
