// Ray-generation: the path-tracer driver. Owns the bounce loop and the
// raygen-resident volume/traversal effects (fog extinction + black-hole ray
// bending) — these operate *between* traceRay calls, so they live here, not in
// any hit shader. Each bounce's surface response comes from the SBT-selected
// closest-hit (opaque/glass/hair), which fills the payload with the emitted
// radiance + attenuation + the next ray.
enable wgpu_ray_tracing_pipeline;
enable f16;
enable wgpu_cooperative_vector;

// The scene TLAS lives in the shared scene bind group (set 0) — imported so the
// pipeline layout matches the wgpu-built scene bind group bound at trace time.
#import bevy_solari::scene_bindings::{tlas, RAY_T_MIN, RAY_T_MAX}
#import bevy_solari::rt_payload::{RtPayload, RtCamera}
#import bevy_solari::sampling::{Reservoir, SurfaceGbuf, GiSample, pick_luminance}
#import bevy_solari::pbr::rand_f
#import bevy_solari::atmosphere::{atmosphere_ray_sphere_near, atmosphere_ray_sphere_far, atmosphere_rayleigh_phase, atmosphere_mie_phase}
#import bevy_render::utils::octahedral_decode_signed

// Rec. 709 luminance, for Russian-roulette survival probability.
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}


// Hard path-length backstop; the configured cap (estimator bits 22..27)
// arrives ≤ 32 and Russian roulette terminates almost every path far sooner.
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
// ReSTIR DI reservoirs: 2 slots per pixel, interleaved by frame parity
// (see `Reservoir`). raygen clears this pixel's CURRENT slot; the opaque chit
// fills it on a primary hit — sky/glass pixels then carry dead (M=0) history.
@group(1) @binding(9) var<storage, read_write> reservoirs: array<Reservoir>;
// Primary-surface shading inputs (chit-written in spatial/GI modes) — the GI
// reshade reads them at path end.
@group(1) @binding(10) var<storage, read_write> surfaces: array<SurfaceGbuf>;
// ReSTIR GI canonical samples: 2 slots/pixel, interleaved by frame parity.
@group(1) @binding(12) var<storage, read_write> gi_samples: array<GiSample>;

// NRC inference mirrors + training records.
// Struct layout MUST MATCH nrc/nrc_mlp.wgsl and nrc/mod.rs.
struct NrcRecord {
    pos_rough: vec4<f32>,
    dir_normal_cs: vec4<f32>,
    diff_target_r: vec4<f32>,
    spec_target_g: vec4<f32>,
    target_b_valid: vec4<f32>,
}
@group(1) @binding(13) var<storage, read> nrc_weights: array<f16>;
@group(1) @binding(14) var<storage, read> nrc_bias: array<f16>;
@group(1) @binding(15) var<storage, read_write> nrc_records: array<NrcRecord>;
// Termination-query ring, sized to the viewport (one query per pixel per
// frame — sample 0 only, so pixels are unique); the nrc_query_infer compute
// pass batch-evaluates the MLP and composites into the output buffer after the
// trace. Layout MUST MATCH nrc_mlp.wgsl and NRC_QUERY_SIZE in nrc/mod.rs.
struct NrcQueryGpu {
    // [pos_unit.xyz (f32 bits), packed material r5g6b5+m8+r8]
    v0: vec4<u32>,
    // [normal cyl (unorm2x16), -wo cyl (unorm2x16), throughput.rg (f16x2),
    //  throughput.b (f16x2, y unused)]
    v1: vec4<u32>,
    // [pixel index, unused ×3]
    v2: vec4<u32>,
}
struct NrcQueryBuf {
    count: atomic<u32>,
    // Training-slot allocator (scattered pixel selection claims record-ring
    // slots through it); cleared with the count each frame.
    train_count: atomic<u32>,
    pad_b: u32,
    pad_c: u32,
    q: array<NrcQueryGpu>,
}
@group(1) @binding(16) var<storage, read_write> nrc_queries: NrcQueryBuf;

const NRC_RECORD_CAP: u32 = 16384u;
const NRC_WIDTH: u32 = 64u;

fn nrc_pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn nrc_one_blob(x: f32, bin: u32) -> f32 {
    let center = (f32(bin) + 0.5) / 4.0;
    let d = x - center;
    return exp(-d * d * 32.0);
}

// Wrap-aware one-blob for the cylindrical phi coordinate; MUST MATCH
// nrc_mlp.wgsl::one_blob_wrap.
fn nrc_one_blob_wrap(x: f32, bin: u32) -> f32 {
    let center = (f32(bin) + 0.5) / 4.0;
    var d = x - center;
    d -= round(d);
    return exp(-d * d * 32.0);
}

// Cylindrical equal-area mapping (phi wrapped, z); MUST MATCH
// nrc_mlp.wgsl::cyl_encode.
fn nrc_cyl(v: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(
        atan2(v.x, v.z) / (2.0 * 3.14159265) + 0.5,
        v.y * 0.5 + 0.5,
    );
}

// Cache-query surface inputs unpacked from the payload's r5g6b5+m8+r8 material.
struct NrcSurf {
    diff_alb: vec3<f32>,
    spec_alb: vec3<f32>,
    rough: f32,
}

fn nrc_unpack_material(m: u32) -> NrcSurf {
    let base_color = vec3<f32>(
        f32((m >> 27u) & 0x1fu) / 31.0,
        f32((m >> 21u) & 0x3fu) / 63.0,
        f32((m >> 16u) & 0x1fu) / 31.0);
    let metallic = f32((m >> 8u) & 0xffu) / 255.0;
    var s: NrcSurf;
    s.rough = f32(m & 0xffu) / 255.0;
    s.diff_alb = base_color * (1.0 - metallic);
    s.spec_alb = mix(vec3(0.04), base_color, metallic);
    return s;
}

// Inline cache query (camera.nrc.z): the full MLP evaluated in-register via
// coopvec against the transposed weight mirror adam maintains. The encode
// layout MUST MATCH nrc_mlp.wgsl::nrc_encode_oct (62 features: 36 freq
// position, 4×4 one-blob dir/normal octs, 4 roughness blob, 3+3 albedos).
fn nrc_query(
    pos_unit: vec3<f32>,
    dir_cs: vec2<f32>,
    nrm_cs: vec2<f32>,
    roughness: f32,
    diff_albedo: vec3<f32>,
    spec_albedo: vec3<f32>,
) -> vec3<f32> {
    var v = coopVecSplat<coop_vec64<f16>>(0.0h);
    var pos_v = pos_unit;
    for (var d = 0u; d < 3u; d += 1u) {
        for (var oct = 0u; oct < 6u; oct += 1u) {
            let phase = pos_v[d] * 3.14159265 * f32(1u << oct);
            v = coopVecInsert(v, d * 12u + oct * 2u, f16(sin(phase)));
            v = coopVecInsert(v, d * 12u + oct * 2u + 1u, f16(cos(phase)));
        }
    }
    let rough_in = 1.0 - exp(-roughness);
    for (var b = 0u; b < 4u; b += 1u) {
        v = coopVecInsert(v, 36u + b, f16(nrc_one_blob_wrap(dir_cs.x, b)));
        v = coopVecInsert(v, 40u + b, f16(nrc_one_blob(dir_cs.y, b)));
        v = coopVecInsert(v, 44u + b, f16(nrc_one_blob_wrap(nrm_cs.x, b)));
        v = coopVecInsert(v, 48u + b, f16(nrc_one_blob(nrm_cs.y, b)));
        v = coopVecInsert(v, 52u + b, f16(nrc_one_blob(rough_in, b)));
    }
    var diff_v = diff_albedo;
    var spec_v = spec_albedo;
    for (var c = 0u; c < 3u; c += 1u) {
        v = coopVecInsert(v, 56u + c, f16(diff_v[c]));
        v = coopVecInsert(v, 59u + c, f16(spec_v[c]));
    }
    let zero_vec = coopVecSplat<coop_vec64<f16>>(0.0h);
    for (var l = 0u; l < 6u; l += 1u) {
        v = coopVecMatMulAdd<coop_vec64<f16>>(
            v, &nrc_weights, l * NRC_WIDTH * NRC_WIDTH, &nrc_bias, l * NRC_WIDTH);
        if l < 5u {
            v = coopVecMax(v, zero_vec);
        }
    }
    // Clamp the prediction to the target range [0, 256]: a cold or bad
    // cache must never inject a huge radiance into shading/bootstrap (it
    // poisons the accumulated mean and the TD target). f16 can also emit
    // inf — this catches it (clamp propagates NaN, so guard that too).
    var out = vec3<f32>(
        f32(coopVecExtract(v, 0u)),
        f32(coopVecExtract(v, 1u)),
        f32(coopVecExtract(v, 2u)),
    );
    out = select(out, vec3(0.0), out != out);
    return clamp(out, vec3(0.0), vec3(256.0));
}

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
// Branch-chain lookup, NOT an indexed array: a function-local array is
// dynamically indexed → local memory, and its 120 B would sit in EVERY
// thread's raygen frame in every config (the heatmap only paints when
// frame.z == 1, but stack allocation is the shader's static maximum). At
// raygen's register cliff, frame bytes are the scarcest resource on the
// card — see the ReSTIR-GI bit-5 comment. Divergence is irrelevant here:
// debug-view-only code.
fn heat_stop(i: i32) -> vec3<f32> {
    if i <= 0 { return vec3<f32>(0.0, 2.0, 91.0); }
    if i == 1 { return vec3<f32>(0.0, 108.0, 251.0); }
    if i == 2 { return vec3<f32>(0.0, 221.0, 221.0); }
    if i == 3 { return vec3<f32>(51.0, 221.0, 0.0); }
    if i == 4 { return vec3<f32>(255.0, 252.0, 0.0); }
    if i == 5 { return vec3<f32>(255.0, 180.0, 0.0); }
    if i == 6 { return vec3<f32>(255.0, 104.0, 0.0); }
    if i == 7 { return vec3<f32>(226.0, 22.0, 0.0); }
    if i == 8 { return vec3<f32>(191.0, 0.0, 83.0); }
    return vec3<f32>(145.0, 0.0, 65.0);
}

fn cost_heatmap(t: f32) -> vec3<f32> {
    let s = clamp(t, 0.0, 1.0) * 10.0;
    let cur = min(i32(s), 9);
    let prv = max(cur - 1, 0);
    let nxt = min(cur + 1, 9);
    let fc = f32(cur);
    let blur = 0.8;
    let wc = smoothstep(fc - blur, fc + blur, s) * (1.0 - smoothstep(fc + 1.0 - blur, fc + 1.0 + blur, s));
    let wp = 1.0 - smoothstep(fc - blur, fc + blur, s);
    let wn = smoothstep(fc + 1.0 - blur, fc + 1.0 + blur, s);
    return clamp((wc * heat_stop(cur) + wp * heat_stop(prv) + wn * heat_stop(nxt)) / 255.0, vec3(0.0), vec3(1.0));
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
// the same buffer at byte 512. One bilinear lookup replaces an 8-step sun
// integral per march step.
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

    // NRC training paths: hash-scattered pixels record up
    // to 4 path vertices each into the 16k record ring; targets are the
    // path's own suffix radiance, propagated after the walk. Scattering beats
    // a contiguous band: every training step sees the whole frame's light
    // distribution instead of one redundant stripe. ~2× over-selection
    // competes for the 4096 ring slots (first come) so every slot is claimed
    // every frame — an unclaimed slot would train on a stale record.
    let nrc_on = camera.nrc.x > 0.0;
    // Estimator lever (flag bit 20): terminate GI at bounce 2 into the cache.
    let nrc_gi_on =
        nrc_on && (bitcast<u32>(camera.atmo.w) & (1u << 20u)) != 0u;
    // Inline inference (camera.nrc.z): GI termination + debug view call
    // nrc_query directly in raygen instead of the batched query path.
    let nrc_inline_on = nrc_on && camera.nrc.z > 0.0;
    // ReSTIR GI (flag bit 5) also routes cache termination through the INLINE
    // query: the GI reservoir stores its sample (gi_L) at the end of the
    // bounce loop, but the batched composite lands post-trace — AFTER the
    // reservoir shade is stored — so a deferred cache tail would be missing
    // from every stored (and temporally/spatially reused) sample. Inline, the
    // tail lands in radiance/gi_L before the store.
    let nrc_restir_gi = (bitcast<u32>(camera.atmo.w) & (1u << 5u)) != 0u;
    let nrc_sel = max((dims.x * dims.y) / (NRC_RECORD_CAP / 2u), 1u);
    var nrc_training = nrc_on
        && nrc_pcg(pixel_index ^ (camera.frame.x * 0x9e3779b9u)) % nrc_sel == 0u;
    var nrc_base = 0u;
    if nrc_training {
        let slot = atomicAdd(&nrc_queries.train_count, 1u);
        if slot < NRC_RECORD_CAP / 4u {
            nrc_base = slot * 4u;
        } else {
            nrc_training = false;
        }
    }
    // 1-in-16 training paths trace their FULL suffix — no cache termination.
    // Pure-bootstrap TD has no absolute anchor at depth: a self-consistent
    // radiance field can inflate coherently (slow color drift to blowout);
    // the unbiased fraction pins it to measurement (paper §5.4).
    let nrc_unbiased = nrc_training
        && (nrc_pcg(pixel_index ^ (camera.frame.x * 2891336453u)) & 15u) == 0u;
    var nrc_mask = 0u;
    // Packed f16 (2×u32 per vec3): these arrays are dynamically indexed, so
    // they live in local memory in EVERY config — at raygen's register cliff
    // the footprint is what matters, and the NRC MLP consuming the values is
    // fp16 anyway. 96 B/thread -> 64 B/thread.
    var nrc_prefix_rad = array<vec2<u32>, 4>();
    var nrc_atten = array<vec2<u32>, 4>();
    // Total alpha any-hit invocations across all primary/bounce traces this pixel
    // (the OMM-effectiveness heatmap). Shadow-ray any-hits use the shadow payload's
    // own counter and aren't summed here.
    var total_anyhit = 0u;
    // Primary hit's cluster + triangle (geometry-debug views). Sentinel = primary miss.
    var primary_cluster = 0xffffffffu;
    var primary_primitive = 0u;
    var primary_normal_oct = 0u;
    var primary_geo_normal_oct = 0u;
    // NRC debug view (frame.z == 6): the primary hit's packed material +
    // position for the cache-prediction paint.
    var nrc_dbg_material = 0xffffffffu;
    var nrc_dbg_pos = vec3<f32>(0.0);
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
        // Sub-pixel jitter: DLSS's when realtime; uniform pixel-area AA when
        // accumulating. Estimator flag bit 21 disables the AA jitter (pixel
        // CENTER every sample) so reservoir merges see identical surface
        // points frame to frame — the target-mismatch isolation lever. It is
        // also set when DLSS drives: RR must see the jitter it suggested.
        var jitter = camera.jitter.xy;
        if reference && (bitcast<u32>(camera.atmo.w) & (1u << 21u)) == 0u {
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
        // NRC spread-based termination (paper §5.1): accumulate the ray's
        // footprint spread; terminate into the cache once it exceeds a
        // fraction of the primary vertex's — diffuse bounces spread fast
        // (terminate early, cache error hidden by the blur), sharp specular
        // slowly (run deeper, where cache error would show). Fewer, better-
        // placed queries than a hard bounce cutoff.
        var nrc_spread = 0.0;
        var nrc_prev_pdf = 1.0;
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

        // Path-length cap: estimator bits 22..27 carry the max INDIRECT
        // bounce count (bounce 0 is the primary trace), so `bounces: 1`
        // is classic one-bounce RTGI. With NRC armed the capped vertex
        // terminates INTO the cache (see nrc_term), and training paths are
        // EXEMPT from the cap — render short, train long: the cache learns
        // full transport while production pixels trace one bounce and
        // composite the cache tail.
        let bounce_cap = min((bitcast<u32>(camera.atmo.w) >> 22u) & 0x3fu, MAX_BOUNCES);
        var sample_cap = bounce_cap;
        if nrc_training && s == 0u {
            sample_cap = MAX_BOUNCES;
        } else if nrc_gi_on {
            // Sharp-footprint chains (mirror/glossy prefixes: delta pdf ⇒
            // the spread accumulator adds ~0) DEFER their cache query past
            // the path cap (see nrc_term's spread gate) — give them rope to
            // the bounce-5 hard stop. Diffuse chains still query at the cap.
            sample_cap = max(bounce_cap, 5u);
        }
        for (var bounce = 0u; bounce < sample_cap + 1u; bounce += 1u) {
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
            payload.hit_material = 0xffffffffu;
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
                // NRC debug view: the paint queries the cache with the
                // primary hit's material/position (same decode the training
                // records use).
                if camera.frame.z == 6u {
                    nrc_dbg_material = payload.hit_material;
                    nrc_dbg_pos = payload.next_origin;
                }
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

            // NRC vertex capture (sample 0 only — one path per pixel): inputs
            // straight into the ring, prefix radiance + incoming throughput in
            // registers for the backward target pass after the walk.
            if nrc_training && s == 0u && bounce < 4u
                && payload.hit_cluster != 0xffffffffu
                && payload.hit_material != 0xffffffffu {
                let ms = nrc_unpack_material(payload.hit_material);
                let nrm_cs = nrc_cyl(octahedral_decode_signed(
                    unpack2x16snorm(payload.hit_normal_oct)));
                let dir_cs = nrc_cyl(-direction);
                var rec: NrcRecord;
                // Anchor-relative position — the encode pass only scales + biases.
                rec.pos_rough = vec4<f32>(
                    payload.next_origin + camera.nrc_anchor.xyz, ms.rough);
                rec.dir_normal_cs = vec4<f32>(dir_cs, nrm_cs);
                rec.diff_target_r = vec4<f32>(ms.diff_alb, 0.0);
                rec.spec_target_g = vec4<f32>(ms.spec_alb, 0.0);
                // .y = valid (for the scatter, which runs before the
                // backward target pass rewrites this), .z = pixel, .w = bounce.
                rec.target_b_valid = vec4<f32>(0.0, 1.0, f32(pixel_index), f32(bounce));
                nrc_records[nrc_base + bounce] = rec;
                nrc_prefix_rad[bounce] = vec2<u32>(
                    pack2x16float(radiance.xy), pack2x16float(vec2<f32>(radiance.z, 0.0)));
                nrc_atten[bounce] = vec2<u32>(
                    pack2x16float(throughput.xy), pack2x16float(vec2<f32>(throughput.z, 0.0)));
                nrc_mask |= 1u << bounce;
            }

            // Grow the footprint spread by this segment (length / sqrt(pdf
            // that sampled the ray into this vertex)). Squared → an area.
            if bounce > 0u {
                let seg = length(payload.next_origin - origin);
                nrc_spread += seg / sqrt(max(nrc_prev_pdf, 1.0e-3));
            }
            let nrc_spread_area = nrc_spread * nrc_spread;
            let nrc_prim_area = primary_t * primary_t;
            // Terminate when the footprint has spread past c·(primary area),
            // never before bounce 2, always by bounce 5 (bounds query cost →
            // caps the ray-gen dispatch length, i.e. the CTX-switch watchdog).
            let nrc_spread_hit = nrc_spread_area
                > camera.nrc.y * max(nrc_prim_area, 1.0e-6);

            // Cache GI termination: spread-gated, at SECONDARY vertices —
            // this is the actual cache speedup. Default: append a query to
            // the ring and end the path; the post-trace nrc_query_infer pass
            // batch-evaluates the MLP and composites throughput × cache into
            // this pixel. Training paths (and camera.nrc.z, and ReSTIR GI —
            // see nrc_restir_gi) evaluate the MLP inline instead: the
            // backward target pass reads the loop-final radiance, so the
            // cache tail must land IN-LOOP for the targets to bootstrap
            // through the termination (TD, bounded contraction — full
            // measured suffixes diverge on the small-denominator
            // relative-L2). Without ReSTIR GI that's ≤4096 inline evals/frame
            // — the coherent batch carries the bulk.
            // `bounce >= bounce_cap` folds the path-length cap in: the last
            // allowed vertex queries the cache instead of dropping the tail —
            // but ONLY once the footprint has spread (nrc_spread_hit). A
            // mirror/glossy prefix has ~zero spread at the cap (delta pdf),
            // and querying there paints the cache's blurry, under-trained,
            // chromatic answer into a pinhole-sharp reflection (the colored
            // metal-sphere artifact in every grayscale rt render). Sharp
            // chains keep tracing REAL transport (sample_cap extends to 5
            // when NRC is armed) and query where the footprint widens.
            let nrc_term = bounce >= 5u || (bounce >= 2u && nrc_spread_hit)
                || (bounce >= bounce_cap && nrc_spread_hit);
            // Training paths terminate deeper (16× the spread threshold,
            // bounces 3-7): their TD targets then carry more measured
            // bounces before the cache bootstrap — the grounding that damps
            // the cache-trains-on-itself oscillation.
            let nrc_term_train = bounce >= 7u
                || (bounce >= 3u && nrc_spread_area
                    > 16.0 * camera.nrc.y * max(nrc_prim_area, 1.0e-6));
            let nrc_train_path = nrc_training && s == 0u;
            if nrc_gi_on
                && !(nrc_unbiased && s == 0u)
                && payload.hit_cluster != 0xffffffffu
                && payload.hit_material != 0xffffffffu
                && select(nrc_term, nrc_term_train, nrc_train_path) {
                let pos_unit = clamp(
                    (payload.next_origin + camera.nrc_anchor.xyz) / camera.nrc.x + 0.5,
                    vec3(0.0), vec3(1.0));
                if nrc_inline_on || nrc_train_path || nrc_restir_gi {
                    let ms = nrc_unpack_material(payload.hit_material);
                    let nrm_cs = nrc_cyl(octahedral_decode_signed(
                        unpack2x16snorm(payload.hit_normal_oct)));
                    let dir_cs = nrc_cyl(-direction);
                    let cache = nrc_query(
                        pos_unit, dir_cs, nrm_cs, ms.rough, ms.diff_alb, ms.spec_alb);
                    let tail = cache
                        * (ms.diff_alb + ms.spec_alb + vec3(1.0e-2))
                        / max(camera.camera_position.w, 1.0e-9);
                    radiance += throughput * tail;
                    // Keep the radiance ≡ di0 + a0·gi_L split exact: cache
                    // termination happens past bounce 1, where throughput =
                    // a0·gi_throughput — so the GI reservoir's stored suffix
                    // carries the tail too.
                    gi_L += gi_throughput * tail;
                    break;
                } else if s == 0u {
                    // One query per pixel per frame keeps the composite
                    // race-free; a full ring keeps tracing (unbiased fallback).
                    let slot = atomicAdd(&nrc_queries.count, 1u);
                    if slot < dims.x * dims.y {
                        var q: NrcQueryGpu;
                        q.v0 = vec4<u32>(
                            bitcast<u32>(pos_unit.x),
                            bitcast<u32>(pos_unit.y),
                            bitcast<u32>(pos_unit.z),
                            payload.hit_material);
                        q.v1 = vec4<u32>(
                            pack2x16unorm(nrc_cyl(octahedral_decode_signed(
                                unpack2x16snorm(payload.hit_normal_oct)))),
                            pack2x16unorm(nrc_cyl(-direction)),
                            pack2x16float(throughput.rg),
                            pack2x16float(vec2<f32>(throughput.b, 0.0)));
                        q.v2 = vec4<u32>(pixel_index, 0u, 0u, 0u);
                        nrc_queries.q[slot] = q;
                        break;
                    }
                }
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
            nrc_prev_pdf = payload.p_bounce;

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
            // ReSTIR GI, raygen side: EXPORT ONLY. This block once held the
            // temporal merge + reservoir shade and cost ~90 ms/frame at
            // 16 spp on room (138 ms vs 46 ms equal-time vs path-traced GI):
            // raygen's live-across-trace state (accumulators, dynamically
            // indexed NRC arrays, payload) leaves no register headroom, so a
            // composite local (Surf/ResolvedMaterial/GiSample — even inside
            // an inlined leaf helper) spills to local memory, and holding the
            // merge state in scalars blows register allocation instead,
            // slowing code that never touches it. Bisections: empty block
            // 39 ms, `unpack_surface` alone ~110 ms, scalarized full merge
            // ~120 ms. The merge + shade now run in `restir_spatial.wgsl`'s
            // `gi_finalize` dispatch (fresh register file, chit-cheap);
            // raygen writes the canonical sample from values ALREADY LIVE in
            // registers and touches no other memory.
            let l_reuse = gi_L - gi_e1;
            let canon_ok = gi_hit && gi_pdf1 > 0.0 && gi_pdf1 < 1.0e30;
            let cur_slot = pixel_index * 2u + (camera.frame.x & 1u);
            let has_surface = primary_cluster != 0xffffffffu;
            // Only the LAST sample's canonical survives to `gi_finalize`
            // (each store overwrites the slot and nothing reads it mid-loop),
            // so skip the export for samples 0..rounds-1 — at 16 spp that is
            // 15/16 of this block's memory traffic.
            if s + 1u == rounds {
            gi_samples[cur_slot].pos_x = gi_xs.x;
            gi_samples[cur_slot].pos_y = gi_xs.y;
            gi_samples[cur_slot].pos_z = gi_xs.z;
            gi_samples[cur_slot].normal_oct = gi_ns_oct;
            gi_samples[cur_slot].l_r = l_reuse.x;
            gi_samples[cur_slot].l_g = l_reuse.y;
            gi_samples[cur_slot].l_b = l_reuse.z;
            gi_samples[cur_slot].w = select(0.0, 1.0 / max(gi_pdf1, 1.0e-9), canon_ok);
            gi_samples[cur_slot].a0_r = a0.x;
            gi_samples[cur_slot].a0_g = a0.y;
            gi_samples[cur_slot].a0_b = a0.z;
            // The draw counts toward M even when dead (W averages ALL draws).
            gi_samples[cur_slot].m = 1.0;
            // Generating-surface fields: `gi_finalize` fills them from the
            // chit's SurfaceGbuf; view_z carries only the has-surface
            // sentinel here (0 = sky, next frame's validation rejects).
            gi_samples[cur_slot].surf_normal_oct = 0u;
            gi_samples[cur_slot].surf_view_z = select(0.0, 1.0, has_surface);
            gi_samples[cur_slot].pad_a = 0u;
            gi_samples[cur_slot].pad_b = 0u;
            }
            // Raygen keeps the per-pixel terms reuse can't carry: bounce-1
            // emission (this pixel's DI-by-MIS partner) and the dead-sample
            // live fallback (bounce-1 MISS / delta pdf has no reconnection
            // vertex — that suffix shades per-frame). The reservoir shade
            // lands in the output buffer from `gi_finalize`.
            var gi_env = vec3<f32>(0.0);
            if !canon_ok {
                gi_env = a0 * l_reuse;
            }
            gi_out = a0 * gi_e1 + gi_env;
            // Realtime firefly filter (dims.w; the reference passes 0 = off)
            // on the raygen-owned terms; `gi_finalize` clamps its own add.
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
        // NRC backward target pass: suffix radiance leaving vertex k toward
        // the camera path = (final − prefix_k) / throughput_into_k. RR keeps
        // throughput O(1) so the division is tame; guards catch the tail.
        if nrc_training && s == 0u {
            // TD self-training (paper §5): target_k = emission_k +
            // atten_k · cache(vertex_{k+1}). One real BSDF step, the tail is
            // the cache bootstrapping itself (stop-gradient — `targets` is a
            // constant to the backward kernels). No throughput DIVISION, so
            // no small-denominator blowup. Everything in exposure-scaled
            // physical radiance; the cache is factorized so multiply the next
            // vertex's albedo back before adding, divide THIS vertex's out.
            let E = camera.camera_position.w;
            for (var k = 0u; k < 4u; k += 1u) {
                let slot = nrc_base + k;
                if (nrc_mask & (1u << k)) == 0u {
                    nrc_records[slot].target_b_valid = vec4<f32>(0.0);
                    continue;
                }
                let alb = nrc_records[slot].diff_target_r.xyz
                    + nrc_records[slot].spec_target_g.xyz + vec3(1.0e-2);
                // Measured path suffix leaving vertex k (radiance from k
                // onward ÷ throughput into k). Clamp tames the
                // small-denominator tail.
                let a_p = nrc_atten[k];
                let atten_k = vec3<f32>(unpack2x16float(a_p.x), unpack2x16float(a_p.y).x);
                let r_p = nrc_prefix_rad[k];
                let prefix_k = vec3<f32>(unpack2x16float(r_p.x), unpack2x16float(r_p.y).x);
                let tin = max(atten_k, vec3(3.0e-2));
                var tgt = (radiance - prefix_k) / tin * E;
                if any(tgt != tgt) {
                    tgt = vec3(0.0);
                }
                tgt = clamp(tgt / alb, vec3(0.0), vec3(256.0));
                nrc_records[slot].diff_target_r.w = tgt.r;
                nrc_records[slot].spec_target_g.w = tgt.g;
                // .z/.w keep the capture's pixel/bounce (record debugging).
                nrc_records[slot].target_b_valid = vec4<f32>(
                    tgt.b, 1.0, f32(pixel_index), f32(k));
            }
        }

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
        // center (the green midpoint — slide it to the scene's midrange); jitter.w =
        // contrast (color change per log2 stop — higher pushes the slowest toward red
        // and the fastest toward blue).
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

    // NRC debug view (frame.z == 6): paint the cache prediction at the
    // primary hit (exposure-scaled radiance — directly displayable since the
    // blit uses exposure 1.0 for debug views). Inputs come from the payload's
    // primary-hit capture — the same decode the training records use.
    if camera.nrc.x > 0.0 {
        if camera.frame.z == 6u && nrc_dbg_material != 0xffffffffu {
            let ms = nrc_unpack_material(nrc_dbg_material);
            let nrm_cs = nrc_cyl(octahedral_decode_signed(
                unpack2x16snorm(primary_normal_oct)));
            let pos_unit = clamp(
                (nrc_dbg_pos + camera.nrc_anchor.xyz) / camera.nrc.x + 0.5,
                vec3(0.0), vec3(1.0));
            final_color = max(nrc_query(
                pos_unit,
                nrc_cyl(-cam_direction),
                nrm_cs,
                ms.rough,
                ms.diff_alb,
                ms.spec_alb,
            ), vec3(0.0)) * (ms.diff_alb + ms.spec_alb + vec3(1.0e-2));
        }
    }

    let index = id.y * dims.x + id.x;
    // Alpha carries the primary-hit depth for the gizmo-depth bridge. The blit
    // forces the displayed alpha back to 1.0, and DLSS resolve reads only the
    // G-buffers, so this never affects the displayed image either way.
    output[index] = vec4<f32>(final_color, primary_depth);
}
