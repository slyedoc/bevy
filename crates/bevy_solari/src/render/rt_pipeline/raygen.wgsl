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
#import bevy_solari::scene_bindings::{tlas, RAY_T_MIN, RAY_T_MAX}
#import bevy_solari::rt_payload::{RtPayload, RtCamera}
#import bevy_solari::sampling::{Reservoir, SurfaceGbuf, GiSample, Surf, unpack_surface, pick_luminance}
#import bevy_solari::brdf::{F_AB, gi_shade, gi_phat}
#import bevy_solari::pbr::rand_f
#import bevy_solari::atmosphere::{atmosphere_ray_sphere_near, atmosphere_ray_sphere_far, atmosphere_rayleigh_phase, atmosphere_mie_phase}
#import bevy_render::utils::octahedral_decode_signed

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
#endif
// ReSTIR DI reservoirs (rung 3): 2 slots per pixel, interleaved by frame parity
// (see `Reservoir`). raygen clears this pixel's CURRENT slot; the opaque chit
// fills it on a primary hit — sky/glass pixels then carry dead (M=0) history.
@group(1) @binding(9) var<storage, read_write> reservoirs: array<Reservoir>;
// Primary-surface shading inputs (chit-written in spatial/GI modes) — the GI
// reshade reads them at path end.
@group(1) @binding(10) var<storage, read_write> surfaces: array<SurfaceGbuf>;
// ReSTIR GI canonical samples: 2 slots/pixel, interleaved by frame parity.
@group(1) @binding(12) var<storage, read_write> gi_samples: array<GiSample>;
// Sentinel pixel index: this bounce writes no guide/reservoir (every non-primary bounce).
const NO_GBUFFER: u32 = 0xffffffffu;

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

// ── World-space atmosphere volumes (spherical planets) ─────────────────────
// Loaded by device address from `RtCamera.atmo` (mirrors
// `atmosphere.rs::GpuAtmosphereVolume(s)` — std430). Marched over the PRIMARY
// segment only: aerial perspective on surfaces, the limb from orbit, the sky
// through the shell on a miss.

struct AtmoVolume {
    center: vec3<f32>,        // shell center relative to the primary camera (world METERS)
    bottom_radius: f32,       // km
    rayleigh_scattering: vec3<f32>,
    rayleigh_scale_height: f32,
    top_radius: f32,          // km
    mie_scattering: f32,
    mie_extinction: f32,
    mie_scale_height: f32,
    mie_phase_g: f32,
    pad_a: f32,
    pad_b: f32,
    pad_c: f32,
}

struct AtmoHeader {
    sun_direction: vec3<f32>,
    sun_illuminance: f32,
    count: u32,
    pad_a: u32,
    pad_b: u32,
    pad_c: u32,
}

const ATMO_VOLUME_STEPS: u32 = 16u;
// Transmittance LUT (mirrors `atmosphere.rs` / `atmosphere_lut_bake.wgsl`):
// per volume, T(radius, sun-zenith cos) baked to a 256×64 vec4 grid living in
// the same buffer at byte 512. One bilinear lookup replaces the 8-step sun
// integral per march step (the 550→120 fps cost).
const ATMO_LUT_W: u32 = 256u;
const ATMO_LUT_H: u32 = 64u;
const ATMO_LUT_OFFSET: u32 = 512u;
const ATMO_LUT_LAYER_BYTES: u32 = ATMO_LUT_W * ATMO_LUT_H * 16u;

// Bilinear sun-transmittance lookup for volume `layer` at radius r (km) and
// sun-zenith cosine mu.
fn atmo_lut_sun_t(base: u64, layer: u32, r: f32, mu: f32, bottom: f32, top: f32) -> vec3<f32> {
    let fx = clamp(mu * 0.5 + 0.5, 0.0, 1.0) * f32(ATMO_LUT_W - 1u);
    let fy = clamp((r - bottom) / max(top - bottom, 1e-4), 0.0, 1.0) * f32(ATMO_LUT_H - 1u);
    let x0 = u32(fx);
    let y0 = u32(fy);
    let x1 = min(x0 + 1u, ATMO_LUT_W - 1u);
    let y1 = min(y0 + 1u, ATMO_LUT_H - 1u);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let row0 = base + u64(ATMO_LUT_OFFSET + layer * ATMO_LUT_LAYER_BYTES + y0 * ATMO_LUT_W * 16u);
    let row1 = base + u64(ATMO_LUT_OFFSET + layer * ATMO_LUT_LAYER_BYTES + y1 * ATMO_LUT_W * 16u);
    let t00 = physical_load<vec4<f32>>(row0 + u64(x0 * 16u)).xyz;
    let t10 = physical_load<vec4<f32>>(row0 + u64(x1 * 16u)).xyz;
    let t01 = physical_load<vec4<f32>>(row1 + u64(x0 * 16u)).xyz;
    let t11 = physical_load<vec4<f32>>(row1 + u64(x1 * 16u)).xyz;
    return mix(mix(t00, t10, tx), mix(t01, t11, tx), ty);
}

// March every volume the primary ray crosses; composite over `radiance`.
// `t_hit_m` = primary hit distance in world meters (huge sentinel on a miss).
fn atmosphere_volumes_apply(
    radiance: vec3<f32>, cam_origin: vec3<f32>, dir: vec3<f32>, t_hit_m: f32,
) -> vec3<f32> {
    let count = bitcast<u32>(camera.atmo.z);
    if count == 0u {
        return radiance;
    }
    let base = (u64(bitcast<u32>(camera.atmo.y)) << 32u) | u64(bitcast<u32>(camera.atmo.x));
    let header = physical_load<AtmoHeader>(base);
    var out = radiance;
    for (var v = 0u; v < min(count, 4u); v += 1u) {
        let vol = physical_load<AtmoVolume>(base + u64(32u) + u64(v) * u64(64u));
        // Shell-centered km space (the atmosphere functions' native frame).
        // The volume's center is camera-relative; re-anchor to this ray.
        let o_km = (cam_origin - (camera.camera_position.xyz + vol.center)) * 1e-3;
        let t_far = atmosphere_ray_sphere_far(o_km, dir, vol.top_radius);
        if t_far <= 0.0 {
            continue;
        }
        let t_enter = max(atmosphere_ray_sphere_near(o_km, dir, vol.top_radius), 0.0);
        let t_exit = min(t_far, t_hit_m * 1e-3);
        if t_exit <= t_enter {
            continue;
        }

        // Single-scatter march, sun transmittance from the baked LUT.
        let cos_theta = dot(dir, header.sun_direction);
        let phase_r = atmosphere_rayleigh_phase(cos_theta);
        let phase_m = atmosphere_mie_phase(vol.mie_phase_g, cos_theta);
        let march_origin = o_km + dir * t_enter;
        let ds = (t_exit - t_enter) / f32(ATMO_VOLUME_STEPS);
        var od_r = 0.0;
        var od_m = 0.0;
        var inscatter = vec3<f32>(0.0);
        for (var i = 0u; i < ATMO_VOLUME_STEPS; i += 1u) {
            let p = march_origin + dir * ((f32(i) + 0.5) * ds);
            let r = length(p);
            let h = r - vol.bottom_radius;
            let d = vec2<f32>(
                exp(-h / vol.rayleigh_scale_height),
                exp(-h / vol.mie_scale_height),
            );
            od_r += d.x * ds;
            od_m += d.y * ds;
            let t_view = exp(-(vol.rayleigh_scattering * od_r
                + vec3<f32>(vol.mie_extinction) * od_m));
            let mu_sun = dot(p / r, header.sun_direction);
            let t_sun = atmo_lut_sun_t(base, v, r, mu_sun, vol.bottom_radius, vol.top_radius);
            let scatter = vol.rayleigh_scattering * (d.x * phase_r)
                + vec3<f32>(vol.mie_scattering) * (d.y * phase_m);
            inscatter += t_view * t_sun * scatter * ds;
        }
        let transmittance = exp(-(vol.rayleigh_scattering * od_r
            + vec3<f32>(vol.mie_extinction) * od_m));

        // The whole gathered path enters the camera through this segment:
        // attenuate it, add the segment's (pre-illuminance) in-scatter.
        out = out * transmittance + inscatter * header.sun_illuminance;
    }
    return out;
}

// Hash an id (cluster or cluster⊕triangle) to a distinct, well-spread flat color for
// the geometry-debug views. PCG-style integer hash → hue via three decorrelated bytes,
// lifted off black so adjacent ids stay visually distinct.
fn id_hash_color(id: u32) -> vec3<f32> {
    var h = id * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    let rgb = vec3<f32>(
        f32(h & 0xffu),
        f32((h >> 8u) & 0xffu),
        f32((h >> 16u) & 0xffu),
    ) / 255.0;
    return 0.15 + 0.85 * rgb;
}

@ray_generation
fn raygen(
    @builtin(ray_invocation_id) id: vec3<u32>,
    @builtin(num_ray_invocations) dims: vec3<u32>,
) {
    let pixel_index = id.x + id.y * dims.x;
    // Reference accumulation (SolariReference): misc.z = samples already in the
    // mean, misc.w = paths this frame (0 = off -> one fresh sample, DLSS jitter).
    let accum_n = camera.misc.z;
    let ref_spf = u32(camera.misc.w);
    let reference = ref_spf > 0u;
    let rounds = max(ref_spf, 1u);

    // This frame's sample average (numerator; /rounds after the loop).
    var frame_sum = vec3<f32>(0.0);
    // Instrument: per-frame SUM of the would-be GI shades across samples (pad_b),
    // vs the last sample's (pad_a) — the pass's debug ratio paint reads both.
    var dbg_gi_lum_sum = 0.0;
    // Total alpha any-hit invocations across all primary/bounce traces this pixel
    // (the OMM-effectiveness heatmap). Shadow-ray any-hits use the shadow payload's
    // own counter and aren't summed here.
    var total_anyhit = 0u;
    // Primary hit's cluster + triangle (geometry-debug views). Sentinel = primary miss.
    var primary_cluster = 0xffffffffu;
    var primary_primitive = 0u;
    var primary_normal_oct = 0u;
    var primary_geo_normal_oct = 0u;
    // Primary-hit depth (reverse-Z NDC) for the gizmo-depth bridge, written into
    // the always-present output buffer's alpha so rasterized overlays (gizmos)
    // occlude against the ray-traced scene — with or without DLSS. -1 = miss/sky.
    var primary_depth = -1.0;
    // Last sample's primary direction (normal-facing debug view reads it post-loop).
    var cam_direction = vec3<f32>(0.0, 0.0, 1.0);
    var rng = 0u;

#ifdef SOLARI_DLSS
    // Default this pixel's guide to "no surface" (sky/miss); a primary hit overwrites
    // it in the closest-hit. Zero normal + roughness 1 + zero albedo is the RR sky
    // convention; the specular `.w` hit-distance stays 0 unless bounce 1 fills it.
    gbuffer_normal_roughness[pixel_index] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    gbuffer_diffuse[pixel_index] = vec4<f32>(0.0);
    gbuffer_specular[pixel_index] = vec4<f32>(0.0);
    gbuffer_motion[pixel_index] = vec4<f32>(0.0);
#endif

    // ReSTIR (estimator flag bit 1): clear this pixel's current-parity reservoir so
    // a primary miss / non-opaque hit can't leave 2-frame-old history in the slot.
    if (bitcast<u32>(camera.atmo.w) & 2u) != 0u {
        let parity = camera.frame.x & 1u;
        reservoirs[pixel_index * 2u + parity] = Reservoir(0u, 0u, 0.0, 0.0, 0u, 0.0, 0u, 0u);
    }

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
    }
#endif

    for (var s = 0u; s < rounds; s += 1u) {
        // Per-sample RNG stream: (pixel, frame, sample) so every path is decorrelated.
        rng = pixel_index + (camera.frame.x * rounds + s) * 5782582u;
#ifdef SOLARI_SHADER_CLOCK
        rng = rng ^ clk0; // pins the start clock read before the loop (see above)
#endif
        // Sub-pixel jitter: DLSS's when realtime; uniform pixel-area AA when accumulating.
        var jitter = camera.jitter.xy;
        if reference {
            jitter = vec2<f32>(rand_f(&rng), rand_f(&rng)) - 0.5;
        }
        let pixel = vec2<f32>(id.xy) + 0.5 + jitter;
        var origin = camera.camera_position.xyz;
        var direction: vec3<f32>;
        if camera.window_arc.y > 0.0 {
            // Cylindrical window (head-coupled curved screen): no matrix maps to a
            // curved surface, so skip the projection entirely — the ray goes from the
            // tracked eye through this pixel's physical point on the screen cylinder.
            // Screen space: center origin, +X right, +Y up, +Z toward viewer; the
            // camera sits at the eye with axes aligned to the screen, so a screen-
            // space direction IS a view-space direction.
            let uv = pixel / vec2<f32>(dims.xy);
            // .w slots offset the view rect (OS window) from the monitor center
            let theta = camera.window_arc.w + (uv.x - 0.5) * camera.window_arc.x;
            let r = camera.window_arc.y;
            let p_screen = vec3<f32>(
                r * sin(theta),
                camera.window_eye.w + (0.5 - uv.y) * camera.window_arc.z,
                r * (1.0 - cos(theta)), // concave: edges bow toward the viewer
            );
            let d_view = p_screen - camera.window_eye.xyz;
            direction = normalize((camera.world_from_view * vec4<f32>(d_view, 0.0)).xyz);
        } else {
            let ndc = (pixel / vec2<f32>(dims.xy)) * 2.0 - 1.0;
            let far = camera.inverse_view_proj * vec4<f32>(ndc.x, -ndc.y, 1.0, 1.0);
            direction = normalize(far.xyz / far.w - origin);
        }
        // Primary ray (pre-bounce) for the atmosphere-volume march.
        let cam_origin = origin;
        cam_direction = direction;
        var primary_t = 1e30;

        var radiance = vec3<f32>(0.0);
        var throughput = vec3<f32>(1.0);
        var captured = false;
        // pdf of the BRDF sample that produced this segment (0 on the primary ray),
        // threaded into the hit shader so it can MIS-weight its emissive vs NEE.
        var p_bounce = 0.0;
        // GI split: radiance ≡ di0 + a0·gi_L (a0 = primary BSDF weight,
        // gi_L = suffix radiance with a0 divided out).
        var di0 = vec3<f32>(0.0);
        var a0 = vec3<f32>(0.0);
        var gi_L = vec3<f32>(0.0);
        var gi_throughput = vec3<f32>(0.0);
        var gi_xs = vec3<f32>(0.0);
        var gi_ns_oct = 0u;
        var gi_pdf1 = 0.0;
        var gi_hit = false;
        var gi_e1 = vec3<f32>(0.0);

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
            payload.emissive_mis = vec3<f32>(0.0);
            payload.attenuation = vec3<f32>(0.0);
            payload.next_origin = origin;
            payload.next_direction = direction;
            payload.bounce = 0u;
            payload.rng = rng;
            payload.p_bounce = p_bounce;
            payload.anyhit_count = 0u;
            // Sentinel so a primary miss (sky) reads as "no cluster" (the miss shader
            // doesn't write these); the closest-hit overwrites on a hit.
            payload.hit_cluster = 0xffffffffu;
            // Only the primary hit produces the visible guide / reservoir write;
            // later bounces pass the sentinel so their closest-hit skips both.
            payload.gbuffer_pixel = select(NO_GBUFFER, pixel_index, bounce == 0u);
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
                RayDesc(RAY_FLAG_NONE, 0xffu, RAY_T_MIN, RAY_T_MAX, origin, direction),
                &payload,
            );
            let material_hint = hitObjectGetSbtRecordIndex(&hit);
            reorderThread(&hit, material_hint, camera.frame.y);
            hitObjectExecuteShader(&hit, &payload);
            rng = payload.rng;
            // Count any-hits on the PRIMARY ray only — its pass-through of unknown
            // cutout micro-regions before it commits, i.e. the any-hit cost of the
            // directly-visible pixel. Bounce (GI) rays would otherwise paint nearby
            // foliage onto the surfaces they illuminate ("plants through walls").
            if bounce == 0u {
                total_anyhit = payload.anyhit_count;
                // Capture the primary hit's cluster + triangle for the geometry-debug views.
                primary_cluster = payload.hit_cluster;
                primary_primitive = payload.hit_primitive;
                primary_normal_oct = payload.hit_normal_oct;
                primary_geo_normal_oct = payload.hit_geo_normal_oct;
            }

            // Capture the PRIMARY hit's depth on bounce 0. A miss leaves
            // `payload.next_origin` at the camera ray origin (the miss shader doesn't
            // touch it; raygen seeded it to `origin`), so a moved origin marks a hit.
            // `origin` is still the primary ray origin here — it's advanced below.
            if bounce == 0u {
                let hit_pos = payload.next_origin;
                if dot(hit_pos - origin, hit_pos - origin) > 1e-10 {
                    let clip = camera.clip_from_world * vec4<f32>(hit_pos, 1.0);
                    primary_depth = clip.z / clip.w;
                    primary_t = length(hit_pos - origin);
                }
            }

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
            if bounce == 0u {
                di0 = payload.emitted;
            } else {
                gi_L += gi_throughput * payload.emitted;
            }
            // Reconnection vertex: the bounce-1 hit (offset origin + its normal).
            if bounce == 1u {
                gi_hit = payload.hit_cluster != 0xffffffffu;
                gi_xs = payload.next_origin;
                gi_ns_oct = payload.hit_normal_oct;
                gi_e1 = payload.emissive_mis;
            }
            if payload.bounce == 0u {
                break;
            }
            throughput *= payload.attenuation;
            if bounce == 0u {
                a0 = payload.attenuation;
                gi_throughput = vec3<f32>(1.0);
                gi_pdf1 = payload.p_bounce;
            } else {
                gi_throughput *= payload.attenuation;
            }
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

            // Russian roulette (unbiased — the ÷p compensates; capped below 1 so
            // lossless paths still terminate). Reference mode defers it to bounce 3
            // (early bounces carry most of the energy; rouletting them buys noise);
            // realtime keeps the aggressive kill — low-albedo scenes would otherwise
            // trace 2×+ the bounce rays per frame.
            if !reference || bounce >= 3u {
                let p = min(luminance(throughput), 0.95);
                if rand_f(&rng) > p {
                    break;
                }
                throughput /= p;
                // Mirror the RR compensation into the split (a0 owns bounce 0's).
                if bounce == 0u {
                    a0 /= p;
                } else {
                    gi_throughput /= p;
                }
            }
        }

        // Estimator overrides, all built on the di0/GI split.
        let eflags = bitcast<u32>(camera.atmo.w);
        var gi_out = a0 * gi_L;
        var dead_draw = 0.0;
        // ReSTIR GI (flag bit 5): store this pixel's canonical sample, then shade
        // GI from the store — the exact stored a0, or (bit 6) f·cos·L/pdf
        // re-evaluated from the surface G-buffer. Dead samples (sky/delta, pdf=0)
        // keep the live a0·gi_L.
        if (eflags & 32u) != 0u {
            let surf_raw = surfaces[pixel_index];
            var surf = unpack_surface(surf_raw);
            surf.f_ab = F_AB(surf.mat.perceptual_roughness, max(dot(surf.ns, surf.wo), 1.0e-4));
            // Bounce-1 emission is this pixel's DI-by-MIS partner — its weight is
            // tied to this sampling event, so it shades per-frame, never reused.
            let l_reuse = gi_L - gi_e1;
            let canon_ok = gi_hit && gi_pdf1 > 0.0 && gi_pdf1 < 1.0e30;
            var sel = GiSample(
                gi_xs.x, gi_xs.y, gi_xs.z, gi_ns_oct,
                l_reuse.x, l_reuse.y, l_reuse.z,
                select(0.0, 1.0 / max(gi_pdf1, 1.0e-9), canon_ok),
                a0.x, a0.y, a0.z,
                select(0.0, 1.0, canon_ok),
                surf_raw.normal_oct, surf_raw.view_z, 0u, 0u,
            );
            // Domain split: a bounce-1 MISS (sky/env) or delta pdf has no
            // reconnection vertex — that suffix shades live per-frame, and the
            // draw still counts toward M (W averages over ALL draws).
            var gi_env = vec3<f32>(0.0);
            if !canon_ok {
                gi_env = a0 * (gi_L - gi_e1);
            }
            sel.m = 1.0;
            let cur_slot = pixel_index * 2u + (camera.frame.x & 1u);
            let has_surface = primary_cluster != 0xffffffffu;
            if !has_surface {
                sel.surf_view_z = 0.0;
            }
            // Spatial pass owns the reservoir shade (flag bit 16); raygen keeps
            // the per-pixel emission term and the dead-sample live fallback.
            let gi_pass_owns = (eflags & 65536u) != 0u;
            gi_out = a0 * gi_e1 + gi_env;
            if (eflags & 128u) != 0u && has_surface {
                // Temporal merge with last frame's slot (prev parity), validated by
                // the generating surface's depth/normal; p̂ re-evaluated here.
                var sel_phat = 0.0;
                var w_sum = 0.0;
                if canon_ok {
                    sel_phat = gi_phat(surf, sel);
                    w_sum = sel_phat * sel.w * sel.m;
                }
                var m_total = sel.m;
                let clip = camera.prev_clip_from_world * vec4<f32>(surf.pos, 1.0);
                if clip.w > 1.0e-4 {
                    let uv = (clip.xy / clip.w) * vec2<f32>(0.5, -0.5) + 0.5;
                    if all(uv >= vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0)) {
                        let pp = vec2<u32>(uv * camera.dims.xy);
                        var hist =
                            gi_samples[(pp.y * u32(camera.dims.x) + pp.x) * 2u + ((camera.frame.x + 1u) & 1u)];
                        // World_rel is camera-origin: last frame's stored x_s is
                        // in last frame's origin. Rebase or the reconnection
                        // point drifts every frame the camera moves (glitter).
                        hist.pos_x -= camera.origin_delta.x;
                        hist.pos_y -= camera.origin_delta.y;
                        hist.pos_z -= camera.origin_delta.z;
                        let hn = octahedral_decode_signed(unpack2x16snorm(hist.surf_normal_oct));
                        let depth_ok = abs(hist.surf_view_z - surf_raw.view_z) < 0.1 * surf_raw.view_z;
                        if hist.m > 0.0 && depth_ok && dot(hn, surf.ns) > 0.9 {
                            let m_h = min(hist.m, camera.dims.z);
                            let ph = gi_phat(surf, hist);
                            let wh = ph * hist.w * m_h;
                            w_sum += wh;
                            m_total += m_h;
                            if wh > 0.0 && rand_f(&rng) * w_sum < wh {
                                sel = hist;
                                sel_phat = ph;
                            }
                        }
                    }
                }
                sel.m = m_total;
                sel.w = 0.0;
                if sel_phat > 0.0 && m_total > 0.0 {
                    sel.w = w_sum / (m_total * sel_phat);
                }
                // Spatial-debug instrument (flag bit 18): stash the shade raygen
                // WOULD apply — the pass's ratio paint reads it (taps 0 ⇒ must be 1).
                if gi_pass_owns && (eflags & 262144u) != 0u {
                    var ref_lum = 0.0;
                    if sel.w > 0.0 {
                        ref_lum = pick_luminance(gi_shade(surf, sel) * sel.w);
                    }
                    dbg_gi_lum_sum += ref_lum;
                    sel.pad_a = bitcast<u32>(ref_lum);
                    sel.pad_b = bitcast<u32>(dbg_gi_lum_sum);
                }
                gi_samples[cur_slot] = sel;
                if !gi_pass_owns && sel.w > 0.0 {
                    gi_out += gi_shade(surf, sel) * sel.w;
                }
            } else {
                if !canon_ok {
                    sel.w = 0.0;
                }
                gi_samples[cur_slot] = sel;
                let stored = gi_samples[cur_slot];
                if stored.w > 0.0 {
                    if gi_pass_owns {
                    } else if (eflags & 64u) != 0u {
                        gi_out += gi_shade(surf, stored) * stored.w;
                    } else {
                        gi_out += vec3<f32>(stored.a0_r, stored.a0_g, stored.a0_b)
                            * vec3<f32>(stored.l_r, stored.l_g, stored.l_b);
                    }
                }
            }
            // Realtime firefly filter (dims.w; the reference passes 0 = off):
            // scale GI spikes down luminance-preserving. The pass-owned spatial
            // shade applies the same threshold on its own add.
            if camera.dims.w > 0.0 {
                let gi_lum = luminance(gi_out);
                if gi_lum > camera.dims.w {
                    gi_out *= camera.dims.w / gi_lum;
                }
            }
            radiance = di0 + gi_out;
            dead_draw = select(1.0, 0.0, canon_ok);
        }
        // GI-only estimator (flag bit 4): the complement of di_only —
        // di_only + gi_only must sum to the full image.
        if (eflags & 16u) != 0u {
            radiance = gi_out;
        }
        // Dead-rate instrument (flag bit 17): paint the dead-canonical
        // indicator; the accumulated mean IS the per-pixel dead-draw rate.
        if (eflags & 131072u) != 0u {
            radiance = vec3<f32>(dead_draw);
        }

        // Atmosphere volumes: attenuate + in-scatter over the primary segment
        // (aerial perspective / limb / sky-through-shell). Per sample.
        radiance = atmosphere_volumes_apply(radiance, cam_origin, cam_direction, primary_t);
        frame_sum += select(radiance, vec3<f32>(0.0), captured);
    } // sample loop

    // The output buffer holds PHYSICAL radiance — the display blit applies the
    // camera exposure at read, so accumulation/dumps/probes/debug paints stay
    // in physical units and an exposure change never resets the mean.
    var final_color = frame_sum / f32(rounds);

    // Reference accumulation: fold this frame's average into the running mean held
    // in the output buffer.
    if reference && accum_n > 0.0 {
        let w = f32(ref_spf) / (accum_n + f32(ref_spf));
        final_color = mix(output[pixel_index].rgb, final_color, w);
    }

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

    // Any-hit invocation heatmap (frame.z == 2): overlay ONLY where the primary
    // ray fired the alpha any-hit (unknown cutout micro-regions, or cutouts with no
    // OMM). Zero-any-hit pixels — resolved in hardware (OMM opaque/transparent, or
    // opaque geometry) — keep their real shaded color, so the scene reads naturally
    // and the warm leaf-contour tracery stands out against it instead of a flat
    // blue field. Warm ramp yellow → red by count; jitter.z scales count → [0, 1].
    if camera.frame.z == 2u && total_anyhit > 0u {
        let t = clamp(f32(total_anyhit) * camera.jitter.z, 0.0, 1.0);
        final_color = mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), t);
    }

    // Geometry-debug views: flat per-cluster (z==3) / per-triangle (z==4) color, a
    // hash of the primary hit's ids → distinct hue. Tessellation density (clusters)
    // and view-dependent LEVEL (triangles) read directly off the colors. A primary
    // miss (sentinel cluster) keeps its real (sky) color.
    if (camera.frame.z == 3u || camera.frame.z == 4u) && primary_cluster != 0xffffffffu {
        var key = primary_cluster;
        if camera.frame.z == 4u {
            key = primary_cluster * 0x9e3779b1u + primary_primitive;
        }
        final_color = id_hash_color(key);
    }

    // Normal-facing debug view (frame.z == 5): classify the primary hit.
    //   RED    = GEOMETRIC (winding) normal faces away → nearest hit is a back-wound
    //            triangle (inverted winding / inside-out / frame bug).
    //   ORANGE = shading normal is >90° from the true surface (dot(Ns,Ng) < 0) →
    //            genuinely INVERTED vertex normal (bad data — fix at the source).
    //   YELLOW = shading normal is valid (within 90° of the surface) but tilted past
    //            the VIEW horizon → benign low-poly grazing; two-sided shading handles it.
    //   BLUE   = shading normal faces you (correct); brightness = facing.
    // On watertight, correctly-wound, correctly-normaled geometry: all blue.
    if camera.frame.z == 5u && primary_cluster != 0xffffffffu {
        let ns = octahedral_decode_signed(unpack2x16snorm(primary_normal_oct));
        let ng = octahedral_decode_signed(unpack2x16snorm(primary_geo_normal_oct));
        let wo = -cam_direction;
        let sfacing = dot(ns, wo);
        let gfacing = dot(ng, wo);
        let sg = dot(ns, ng);          // shading vs true surface: < 0 means inverted
        if gfacing < 0.0 {
            final_color = vec3<f32>(1.0, 0.1, 0.1) * (0.15 + 0.85 * abs(gfacing));
        } else if sg < 0.0 {
            final_color = vec3<f32>(1.0, 0.45, 0.05) * (0.15 + 0.85 * abs(sg));
        } else if sfacing < 0.0 {
            final_color = vec3<f32>(1.0, 0.9, 0.15) * (0.15 + 0.85 * abs(sfacing));
        } else {
            final_color = vec3<f32>(0.1, 0.4, 1.0) * (0.15 + 0.85 * sfacing);
        }
    }

    let index = id.y * dims.x + id.x;
    // Alpha carries the primary-hit depth for the gizmo-depth bridge. The blit
    // forces the displayed alpha back to 1.0, and DLSS resolve reads only the
    // G-buffers, so this never affects the displayed image either way.
    output[index] = vec4<f32>(final_color, primary_depth);
}
