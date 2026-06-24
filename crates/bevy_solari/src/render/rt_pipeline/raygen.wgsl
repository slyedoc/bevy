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
// default first. Packed normal.xyz+roughness, diffuse.xyz+depth (chit-direct), and
// specular.xyz+hit-distance — the specular `.w` is filled by raygen after bounce 1.
@group(1) @binding(5) var<storage, read_write> gbuffer_normal_roughness: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> gbuffer_diffuse: array<vec4<f32>>;
@group(1) @binding(7) var<storage, read_write> gbuffer_specular: array<vec4<f32>>;
@group(1) @binding(8) var<storage, read_write> gbuffer_motion: array<vec4<f32>>;
// Sentinel pixel index: this bounce writes no guide (set on every non-primary bounce).
const NO_GBUFFER: u32 = 0xffffffffu;
#endif

var<ray_payload> payload: RtPayload;

#ifdef SOLARI_SHADER_CLOCK
// 32-bit (LO) clock delta, correcting a single wrap of the counter — the NVIDIA
// timer-instrumentation approach (the low 32 bits are plenty for a heatmap and keep
// register pressure down).
fn clock_delta(start: u32, end: u32) -> u32 {
    return select(~0u - (start - end), end - start, end >= start);
}

// 10-stop "temperature" colormap (NVIDIA timer-instrumentation): deep blue (cheap) →
// cyan → green → yellow → orange → red → magenta (expensive). Input clamped to [0, 1].
fn cost_heatmap(t: f32) -> vec3<f32> {
    var c = array<vec3<f32>, 10>(
        vec3<f32>(0.0, 2.0, 91.0),
        vec3<f32>(0.0, 108.0, 251.0),
        vec3<f32>(0.0, 221.0, 221.0),
        vec3<f32>(51.0, 221.0, 0.0),
        vec3<f32>(255.0, 252.0, 0.0),
        vec3<f32>(255.0, 180.0, 0.0),
        vec3<f32>(255.0, 104.0, 0.0),
        vec3<f32>(226.0, 22.0, 0.0),
        vec3<f32>(191.0, 0.0, 83.0),
        vec3<f32>(145.0, 0.0, 65.0),
    );
    let s = clamp(t, 0.0, 1.0) * 10.0;
    let cur = min(i32(s), 9);
    let prv = max(cur - 1, 0);
    let nxt = min(cur + 1, 9);
    let fc = f32(cur);
    let blur = 0.8;
    let wc = smoothstep(fc - blur, fc + blur, s) * (1.0 - smoothstep(fc + 1.0 - blur, fc + 1.0 + blur, s));
    let wp = 1.0 - smoothstep(fc - blur, fc + blur, s);
    let wn = smoothstep(fc + 1.0 - blur, fc + 1.0 + blur, s);
    return clamp((wc * c[cur] + wp * c[prv] + wn * c[nxt]) / 255.0, vec3(0.0), vec3(1.0));
}
#endif

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
    // convention; the specular `.w` hit-distance stays 0 unless bounce 1 fills it.
    gbuffer_normal_roughness[pixel_index] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    gbuffer_diffuse[pixel_index] = vec4<f32>(0.0);
    gbuffer_specular[pixel_index] = vec4<f32>(0.0);
    gbuffer_motion[pixel_index] = vec4<f32>(0.0);
#endif

#ifdef SOLARI_SHADER_CLOCK
    // Per-pixel cost measurement (heatmap). Read the start clock (32-bit LO) and fold
    // it into the RNG so the compiler can't sink the read past the bounce loop: the
    // loop consumes `rng`, which now depends on `clk0`, pinning the read before it.
    // Clock reads are otherwise freely reorderable (no NVAPI fake-UAV in Vulkan), and
    // the compiler was sinking the start read down next to the end read → ~0 delta.
    // Heatmap-mode only (frame.z == 1), where the shaded color is discarded, so it
    // never perturbs a normal render.
    var clk0 = 0u;
    if camera.frame.z == 1u {
        clk0 = u32(shader_clock());
        rng = rng ^ clk0;
    }
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

#ifdef SOLARI_DLSS
        // DLSS specular hit-distance guide. The primary surface's first
        // continuation ray (≈ the specular reflection in this 1-spp path) only
        // reveals where it lands HERE, after bounce 1 — the primary closest-hit
        // emitted the ray but not its hit point. Record that world-space distance
        // (primary hit → bounce-1 hit; `origin` is still the primary hit point at
        // this point in the loop) into the primary pixel's specular `.w`. A missed
        // or absorbed reflection leaves the cleared 0 (RR reads that as no
        // reflection lag); RR weights this by the specular albedo already in
        // `.xyz`, so writing it for every surface, not only mirrors, is correct.
        if bounce == 1u && payload.bounce == 1u {
            gbuffer_specular[pixel_index].w = length(payload.next_origin - origin);
        }
#endif

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

        // Never feed a degenerate (zero-length or non-finite) direction to the next
        // traceRay — the RT core hangs the GPU on a zero-length ray. A hit shader
        // can produce one from a float edge (e.g. glass critical-angle refraction)
        // or a bad normal; terminate the path instead of hanging. `dot > eps` is
        // false for both zero and NaN.
        if !(dot(direction, direction) > 1.0e-8) {
            break;
        }

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

#ifdef SOLARI_SHADER_CLOCK
    // Cost heatmap debug view (frame.z == 1): replace the shaded color with a
    // colormap of the clock delta across the whole trace. jitter.z scales clocks →
    // [0, 1] (tunable per scene). Bypasses DLSS naturally since the blit reads this
    // buffer; turn DLSS off for a clean read.
    if camera.frame.z == 1u {
        let cycles = f32(clock_delta(clk0, u32(shader_clock())));
        // Per-pixel cost spans orders of magnitude → map log2(cycles) → color, pivoting
        // at the data center so the knob is a CONTRAST control, not a shift. jitter.z =
        // center (the green midpoint; `-` / `=` slide it to the scene's midrange);
        // jitter.w = contrast (color change per log2 stop; `[` / `]` crank it — higher
        // pushes the slowest toward red and the fastest toward blue).
        let t = 0.5 + (log2(max(cycles, 1.0)) - camera.jitter.z) * camera.jitter.w;
        final_color = cost_heatmap(t);
    }
#endif

    let index = id.y * dims.x + id.x;
    output[index] = vec4<f32>(final_color, 1.0);
}
