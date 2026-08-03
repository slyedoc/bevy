// Ray-generation: the path-tracer driver. Owns the bounce loop and the
// raygen-resident volume/traversal effects (fog extinction + black-hole ray
// bending) — these operate *between* trace calls, so they live here, not in
// any hit shader. Each bounce's surface response comes from the SBT-selected
// closest-hit (opaque/glass/hair), which fills the payload with the emitted
// radiance + attenuation + the next ray.
//
// The `SOLARI_SHADER_CLOCK` define compiles in the per-pixel cost heatmap
// (`VK_KHR_shader_clock`); the pipeline build passes it iff the device
// enabled the extension.

import rt_payload;
import scene_resolve;
import brdf;
import sampling;

// Hard path-length backstop; the configured cap (estimator bits 22..27)
// arrives ≤ 32 and Russian roulette terminates almost every path far sooner.
static const uint MAX_BOUNCES = 32;
// Stand-in fog: per-unit-length extinction applied over each ray segment.
static const float FOG_DENSITY = 0.0;
// Stand-in black hole: a point mass that bends rays each step; capture radius
// terminates the path black. Disabled by default (strength 0).
static const float BH_STRENGTH = 0.0;
static const float3 BH_CENTER = float3(0.0);
static const float BH_CAPTURE_RADIUS = 0.5;

// RT-pipeline-private set (set 1): per-pixel output buffer + camera. The scene
// bindings arrive via `scene_resolve` (set 0); the columns set (2) is unused
// by raygen, present for the chits.
[[vk::binding(0, 1)]] RWStructuredBuffer<float4> output;
[[vk::binding(1, 1)]] ConstantBuffer<RtCamera> camera;
// DLSS Ray Reconstruction guide G-buffer (chit-direct): the closest-hit writes
// the primary hit's packed surface attrs; raygen clears each pixel to the
// sky/miss default first. The specular `.w` is filled by raygen after bounce 1.
[[vk::binding(5, 1)]] RWStructuredBuffer<float4> gbuffer_normal_roughness;
[[vk::binding(6, 1)]] RWStructuredBuffer<float4> gbuffer_diffuse;
[[vk::binding(7, 1)]] RWStructuredBuffer<float4> gbuffer_specular;
[[vk::binding(8, 1)]] RWStructuredBuffer<float4> gbuffer_motion;
// ReSTIR DI reservoirs: 2 slots per pixel, interleaved by frame parity.
// raygen clears this pixel's CURRENT slot; the opaque chit fills it on a
// primary hit — sky/glass pixels then carry dead (M=0) history.
[[vk::binding(9, 1)]] RWStructuredBuffer<Reservoir> reservoirs;
// ReSTIR GI canonical samples: 2 slots/pixel, interleaved by frame parity.
[[vk::binding(12, 1)]] RWStructuredBuffer<GiSample> gi_samples;

// NRC inference mirrors + training records.
// Struct layout MUST MATCH nrc/nrc_mlp.slang and nrc/mod.rs.
struct NrcRecord
{
    float4 pos_rough;
    float4 dir_normal_cs;
    float4 diff_target_r;
    float4 spec_target_g;
    float4 target_b_valid;
}
[[vk::binding(13, 1)]] ByteAddressBuffer nrc_weights;
[[vk::binding(14, 1)]] ByteAddressBuffer nrc_bias;
[[vk::binding(15, 1)]] RWStructuredBuffer<NrcRecord> nrc_records;
// Termination-query ring, sized to the viewport; the nrc_query_infer compute
// pass batch-evaluates the MLP and composites into the output buffer after
// the trace. Byte layout MUST MATCH nrc_mlp.slang / NRC_QUERY_SIZE in
// nrc/mod.rs: count@0, train_count@4 (both atomics), queries at 16, 48 B each
// ([pos.xyz bits, material], [normal cyl, -wo cyl, throughput f16x2 ×2],
// [pixel, unused ×3]).
[[vk::binding(16, 1)]] RWByteAddressBuffer nrc_queries;
static const uint NRC_QUERY_BASE = 16;
static const uint NRC_QUERY_STRIDE = 48;

static const uint NRC_RECORD_CAP = 16384;
static const uint NRC_WIDTH = 64;

typealias HVec = CoopVec<half, NRC_WIDTH>;

uint nrc_pcg(uint v)
{
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28) + 4)) ^ s) * 277803737u;
    return (w >> 22) ^ w;
}

float nrc_one_blob(float x, uint bin)
{
    let center = (float(bin) + 0.5) / 4.0;
    let d = x - center;
    return exp(-d * d * 32.0);
}

// Wrap-aware one-blob for the cylindrical phi coordinate; MUST MATCH
// nrc_mlp.slang::one_blob_wrap.
float nrc_one_blob_wrap(float x, uint bin)
{
    let center = (float(bin) + 0.5) / 4.0;
    var d = x - center;
    d -= round(d);
    return exp(-d * d * 32.0);
}

// Cylindrical equal-area mapping (phi wrapped, z); MUST MATCH
// nrc_mlp.slang::cyl_encode.
float2 nrc_cyl(float3 v)
{
    return float2(
        atan2(v.x, v.z) / (2.0 * 3.14159265) + 0.5,
        v.y * 0.5 + 0.5);
}

// Cache-query surface inputs unpacked from the payload's r5g6b5+m8+r8 material.
struct NrcSurf
{
    float3 diff_alb;
    float3 spec_alb;
    float rough;
}

NrcSurf nrc_unpack_material(uint m)
{
    let base_color = float3(
        float((m >> 27) & 0x1f) / 31.0,
        float((m >> 21) & 0x3f) / 63.0,
        float((m >> 16) & 0x1f) / 31.0);
    let metallic = float((m >> 8) & 0xff) / 255.0;
    NrcSurf s;
    s.rough = float(m & 0xff) / 255.0;
    s.diff_alb = base_color * (1.0 - metallic);
    s.spec_alb = lerp(float3(0.04), base_color, metallic);
    return s;
}

// WGSL `pack2x16unorm`.
uint pack_unorm2x16(float2 v)
{
    let c = round(clamp(v, 0.0, 1.0) * 65535.0);
    return uint(c.x) | (uint(c.y) << 16);
}

// Inline cache query (camera.nrc.z): the full MLP evaluated in-register via
// coopvec against the transposed weight mirror adam maintains. The encode
// layout MUST MATCH nrc_mlp.slang::nrc_encode_oct (62 features: 36 freq
// position, 4×4 one-blob dir/normal octs, 4 roughness blob, 3+3 albedos).
float3 nrc_query(
    float3 pos_unit,
    float2 dir_cs,
    float2 nrm_cs,
    float roughness,
    float3 diff_albedo,
    float3 spec_albedo)
{
    // Seed with arbitrary buffer bytes, then write EVERY element (62 features
    // + explicit zero tail) — a zero splat would lower to
    // OpCompositeConstructReplicateEXT, which the device doesn't enable.
    var v = HVec.load(nrc_weights, 0);
    for (uint d = 0; d < 3; d++)
    {
        for (uint oct = 0; oct < 6; oct++)
        {
            let phase = pos_unit[d] * 3.14159265 * float(1u << oct);
            v[d * 12 + oct * 2] = half(sin(phase));
            v[d * 12 + oct * 2 + 1] = half(cos(phase));
        }
    }
    let rough_in = 1.0 - exp(-roughness);
    for (uint b = 0; b < 4; b++)
    {
        v[36 + b] = half(nrc_one_blob_wrap(dir_cs.x, b));
        v[40 + b] = half(nrc_one_blob(dir_cs.y, b));
        v[44 + b] = half(nrc_one_blob_wrap(nrm_cs.x, b));
        v[48 + b] = half(nrc_one_blob(nrm_cs.y, b));
        v[52 + b] = half(nrc_one_blob(rough_in, b));
    }
    for (uint c = 0; c < 3; c++)
    {
        v[56 + c] = half(diff_albedo[c]);
        v[59 + c] = half(spec_albedo[c]);
    }
    v[62] = 0.0h;
    v[63] = 0.0h;
    for (uint l = 0; l < 6; l++)
    {
        v = coopVecMatMulAdd<half, NRC_WIDTH>(
            v, CoopVecComponentType.Float16,
            nrc_weights, l * NRC_WIDTH * NRC_WIDTH * 2, CoopVecComponentType.Float16,
            nrc_bias, l * NRC_WIDTH * 2, CoopVecComponentType.Float16,
            CoopVecMatrixLayout.RowMajor, false, NRC_WIDTH * 2);
        if (l < 5)
        {
            [ForceUnroll]
            for (int i = 0; i < NRC_WIDTH; i++)
                v[i] = max(v[i], 0.0h);
        }
    }
    // Clamp the prediction to the target range [0, 256]: a cold or bad cache
    // must never inject a huge radiance into shading/bootstrap. f16 can also
    // emit inf — this catches it (clamp propagates NaN, so guard that too).
    var out = float3(float(v[0]), float(v[1]), float(v[2]));
    out.x = out.x != out.x ? 0.0 : out.x;
    out.y = out.y != out.y ? 0.0 : out.y;
    out.z = out.z != out.z ? 0.0 : out.z;
    return clamp(out, float3(0.0), float3(256.0));
}

// Sentinel pixel index: this bounce writes no guide/reservoir (every
// non-primary bounce).
static const uint NO_GBUFFER = 0xffffffff;

#ifdef SOLARI_SHADER_CLOCK
// 32-bit (LO) clock delta, correcting a single wrap of the counter — the
// NVIDIA timer-instrumentation approach.
uint clock_delta(uint start, uint end)
{
    return end >= start ? end - start : ~0u - (start - end);
}

// 10-stop "temperature" colormap (NVIDIA timer-instrumentation): deep blue
// (cheap) → cyan → green → yellow → orange → red → magenta (expensive).
// Branch-chain lookup, NOT an indexed array: a function-local array is
// dynamically indexed → local memory in EVERY thread's raygen frame in every
// config. At raygen's register cliff, frame bytes are the scarcest resource.
float3 heat_stop(int i)
{
    if (i <= 0) { return float3(0.0, 2.0, 91.0); }
    if (i == 1) { return float3(0.0, 108.0, 251.0); }
    if (i == 2) { return float3(0.0, 221.0, 221.0); }
    if (i == 3) { return float3(51.0, 221.0, 0.0); }
    if (i == 4) { return float3(255.0, 252.0, 0.0); }
    if (i == 5) { return float3(255.0, 180.0, 0.0); }
    if (i == 6) { return float3(255.0, 104.0, 0.0); }
    if (i == 7) { return float3(226.0, 22.0, 0.0); }
    if (i == 8) { return float3(191.0, 0.0, 83.0); }
    return float3(145.0, 0.0, 65.0);
}

float3 cost_heatmap(float t)
{
    let s = clamp(t, 0.0, 1.0) * 10.0;
    let cur = min(int(s), 9);
    let prv = max(cur - 1, 0);
    let nxt = min(cur + 1, 9);
    let fc = float(cur);
    let blur = 0.8;
    let wc = smoothstep(fc - blur, fc + blur, s)
        * (1.0 - smoothstep(fc + 1.0 - blur, fc + 1.0 + blur, s));
    let wp = 1.0 - smoothstep(fc - blur, fc + blur, s);
    let wn = smoothstep(fc + 1.0 - blur, fc + 1.0 + blur, s);
    return clamp((wc * heat_stop(cur) + wp * heat_stop(prv) + wn * heat_stop(nxt)) / 255.0,
        float3(0.0), float3(1.0));
}
#endif

// ── World-space atmosphere volumes (spherical planets) ─────────────────────
// Loaded by device address from `RtCamera.atmo` (mirrors
// `atmosphere.rs::GpuAtmosphereVolume(s)` — std430). Marched over the PRIMARY
// segment only: aerial perspective on surfaces, the limb from orbit, the sky
// through the shell on a miss.

struct AtmoVolume
{
    float3 center;      // shell center relative to the primary camera (world METERS)
    float bottom_radius; // km
    float3 rayleigh_scattering;
    float rayleigh_scale_height;
    float top_radius;   // km
    float mie_scattering;
    float mie_extinction;
    float mie_scale_height;
    float mie_phase_g;
    float pad_a;
    float pad_b;
    float pad_c;
}

struct AtmoHeader
{
    float3 sun_direction;
    float sun_illuminance;
    uint count;
    uint pad_a;
    uint pad_b;
    uint pad_c;
}

// atmosphere.slang parity.
float atmosphere_ray_sphere_far(float3 o, float3 d, float r)
{
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if (disc < 0.0)
    {
        return -1.0;
    }
    return -b + sqrt(disc);
}

float atmosphere_ray_sphere_near(float3 o, float3 d, float r)
{
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if (disc < 0.0)
    {
        return -1.0;
    }
    return -b - sqrt(disc);
}

float atmosphere_rayleigh_phase(float cos_theta)
{
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

float atmosphere_mie_phase(float g, float cos_theta)
{
    return (1.0 / (4.0 * PI)) * ((1.0 - g * g)
        / pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1e-4), 1.5));
}

static const uint ATMO_VOLUME_STEPS = 16;
// Transmittance LUT (mirrors `atmosphere.rs` / `atmosphere_lut_bake.wgsl`):
// per volume, T(radius, sun-zenith cos) baked to a 256×64 vec4 grid living in
// the same buffer at byte 512.
static const uint ATMO_LUT_W = 256;
static const uint ATMO_LUT_H = 64;
static const uint ATMO_LUT_OFFSET = 512;
static const uint ATMO_LUT_LAYER_BYTES = ATMO_LUT_W * ATMO_LUT_H * 16;

// Bilinear sun-transmittance lookup for volume `layer` at radius r (km) and
// sun-zenith cosine mu.
float3 atmo_lut_sun_t(uint64_t base, uint layer, float r, float mu, float bottom, float top)
{
    let fx = clamp(mu * 0.5 + 0.5, 0.0, 1.0) * float(ATMO_LUT_W - 1);
    let fy = clamp((r - bottom) / max(top - bottom, 1e-4), 0.0, 1.0) * float(ATMO_LUT_H - 1);
    let x0 = uint(fx);
    let y0 = uint(fy);
    let x1 = min(x0 + 1, ATMO_LUT_W - 1);
    let y1 = min(y0 + 1, ATMO_LUT_H - 1);
    let tx = fx - float(x0);
    let ty = fy - float(y0);
    let row0 = base + uint64_t(ATMO_LUT_OFFSET + layer * ATMO_LUT_LAYER_BYTES + y0 * ATMO_LUT_W * 16);
    let row1 = base + uint64_t(ATMO_LUT_OFFSET + layer * ATMO_LUT_LAYER_BYTES + y1 * ATMO_LUT_W * 16);
    let t00 = (*(float4*)(row0 + uint64_t(x0 * 16))).xyz;
    let t10 = (*(float4*)(row0 + uint64_t(x1 * 16))).xyz;
    let t01 = (*(float4*)(row1 + uint64_t(x0 * 16))).xyz;
    let t11 = (*(float4*)(row1 + uint64_t(x1 * 16))).xyz;
    return lerp(lerp(t00, t10, tx), lerp(t01, t11, tx), ty);
}

// March every volume the primary ray crosses; composite over `radiance`.
// `t_hit_m` = primary hit distance in world meters (huge sentinel on a miss).
float3 atmosphere_volumes_apply(
    float3 radiance, float3 cam_origin, float3 dir, float t_hit_m)
{
    let count = asuint(camera.atmo.z);
    if (count == 0)
    {
        return radiance;
    }
    let base = (uint64_t(asuint(camera.atmo.y)) << 32) | uint64_t(asuint(camera.atmo.x));
    let header = *(AtmoHeader*)(base);
    var out = radiance;
    for (uint v = 0; v < min(count, 4u); v++)
    {
        let vol = *(AtmoVolume*)(base + 32 + uint64_t(v) * 64);
        // Shell-centered km space (the atmosphere functions' native frame).
        // The volume's center is camera-relative; re-anchor to this ray.
        let o_km = (cam_origin - (camera.camera_position.xyz + vol.center)) * 1e-3;
        let t_far = atmosphere_ray_sphere_far(o_km, dir, vol.top_radius);
        if (t_far <= 0.0)
        {
            continue;
        }
        let t_enter = max(atmosphere_ray_sphere_near(o_km, dir, vol.top_radius), 0.0);
        let t_exit = min(t_far, t_hit_m * 1e-3);
        if (t_exit <= t_enter)
        {
            continue;
        }

        // Single-scatter march, sun transmittance from the baked LUT.
        let cos_theta = dot(dir, header.sun_direction);
        let phase_r = atmosphere_rayleigh_phase(cos_theta);
        let phase_m = atmosphere_mie_phase(vol.mie_phase_g, cos_theta);
        let march_origin = o_km + dir * t_enter;
        let ds = (t_exit - t_enter) / float(ATMO_VOLUME_STEPS);
        var od_r = 0.0;
        var od_m = 0.0;
        var inscatter = float3(0.0);
        for (uint i = 0; i < ATMO_VOLUME_STEPS; i++)
        {
            let p = march_origin + dir * ((float(i) + 0.5) * ds);
            let r = length(p);
            let h = r - vol.bottom_radius;
            let d = float2(
                exp(-h / vol.rayleigh_scale_height),
                exp(-h / vol.mie_scale_height));
            od_r += d.x * ds;
            od_m += d.y * ds;
            let t_view = exp(-(vol.rayleigh_scattering * od_r
                + float3(vol.mie_extinction) * od_m));
            let mu_sun = dot(p / r, header.sun_direction);
            let t_sun = atmo_lut_sun_t(base, v, r, mu_sun, vol.bottom_radius, vol.top_radius);
            let scatter = vol.rayleigh_scattering * (d.x * phase_r)
                + float3(vol.mie_scattering) * (d.y * phase_m);
            inscatter += t_view * t_sun * scatter * ds;
        }
        let transmittance = exp(-(vol.rayleigh_scattering * od_r
            + float3(vol.mie_extinction) * od_m));

        // The whole gathered path enters the camera through this segment:
        // attenuate it, add the segment's (pre-illuminance) in-scatter.
        out = out * transmittance + inscatter * header.sun_illuminance;
    }
    return out;
}

// Hash an id (cluster or cluster⊕triangle) to a distinct, well-spread flat
// color for the geometry-debug views.
float3 id_hash_color(uint id)
{
    var h = id * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28) + 4)) ^ h) * 277803737u;
    h = (h >> 22) ^ h;
    let rgb = float3(
        float(h & 0xff),
        float((h >> 8) & 0xff),
        float((h >> 16) & 0xff)) / 255.0;
    return 0.15 + 0.85 * rgb;
}

[shader("raygeneration")]
void raygen()
{
    let id = DispatchRaysIndex().xy;
    let dims = DispatchRaysDimensions().xy;
    let pixel_index = id.x + id.y * dims.x;
    // Reference accumulation (SolariReference): misc.z = samples already in
    // the mean, misc.w = paths this frame (0 = off → one fresh sample).
    let accum_n = camera.misc.z;
    let ref_spf = uint(camera.misc.w);
    let reference = ref_spf > 0;
    let rounds = max(ref_spf, 1u);

    // This frame's sample average (numerator; /rounds after the loop).
    var frame_sum = float3(0.0);

    // NRC training paths: hash-scattered pixels record up to 4 path vertices
    // each into the 16k record ring; targets are the path's own suffix
    // radiance, propagated after the walk. ~2× over-selection competes for
    // the ring slots (first come) so every slot is claimed every frame.
    let nrc_on = camera.nrc.x > 0.0;
    // Estimator lever (flag bit 20): terminate GI at bounce 2 into the cache.
    let nrc_gi_on = nrc_on && (asuint(camera.atmo.w) & (1u << 20)) != 0;
    // Inline inference (camera.nrc.z): GI termination + debug view call
    // nrc_query directly in raygen instead of the batched query path.
    let nrc_inline_on = nrc_on && camera.nrc.z > 0.0;
    // ReSTIR GI (flag bit 5) also routes cache termination through the INLINE
    // query: the batched composite lands post-trace — AFTER the reservoir
    // shade is stored — so a deferred cache tail would be missing from every
    // stored (and reused) sample. Inline, the tail lands in radiance/gi_L
    // before the store.
    let nrc_restir_gi = (asuint(camera.atmo.w) & (1u << 5)) != 0;
    let nrc_sel = max((dims.x * dims.y) / (NRC_RECORD_CAP / 2), 1u);
    var nrc_training = nrc_on
        && nrc_pcg(pixel_index ^ (camera.frame.x * 0x9e3779b9)) % nrc_sel == 0;
    var nrc_base = 0u;
    if (nrc_training)
    {
        uint slot;
        nrc_queries.InterlockedAdd(4, 1, slot); // train_count
        if (slot < NRC_RECORD_CAP / 4)
        {
            nrc_base = slot * 4;
        }
        else
        {
            nrc_training = false;
        }
    }
    // 1-in-16 training paths trace their FULL suffix — no cache termination.
    // Pure-bootstrap TD has no absolute anchor at depth; the unbiased
    // fraction pins it to measurement (paper §5.4).
    let nrc_unbiased = nrc_training
        && (nrc_pcg(pixel_index ^ (camera.frame.x * 2891336453u)) & 15) == 0;
    var nrc_mask = 0u;
    // Packed f16 (2×u32 per vec3): these arrays are dynamically indexed, so
    // they live in local memory in EVERY config — at raygen's register cliff
    // the footprint is what matters. 96 B/thread → 64 B/thread.
    uint2 nrc_prefix_rad[4];
    uint2 nrc_atten[4];
    // Total alpha any-hit invocations across all primary/bounce traces this
    // pixel (the OMM-effectiveness heatmap).
    var total_anyhit = 0u;
    // Primary hit's cluster + triangle (geometry-debug views). Sentinel = miss.
    var primary_cluster = 0xffffffffu;
    var primary_primitive = 0u;
    var primary_normal_oct = 0u;
    var primary_geo_normal_oct = 0u;
    // NRC debug view (frame.z == 6): the primary hit's packed material +
    // position for the cache-prediction paint.
    var nrc_dbg_material = 0xffffffffu;
    var nrc_dbg_pos = float3(0.0);
    // Primary-hit depth (reverse-Z NDC) for the gizmo-depth bridge, written
    // into the output buffer's alpha so rasterized overlays occlude against
    // the ray-traced scene. -1 = miss/sky.
    var primary_depth = -1.0;
    // Last sample's primary direction (normal-facing debug view reads it
    // post-loop).
    var cam_direction = float3(0.0, 0.0, 1.0);
    var rng = 0u;

    // Default this pixel's guide to "no surface" (sky/miss); a primary hit
    // overwrites it in the closest-hit. Zero normal + roughness 1 + zero
    // albedo is the RR sky convention.
    gbuffer_normal_roughness[pixel_index] = float4(0.0, 0.0, 0.0, 1.0);
    gbuffer_diffuse[pixel_index] = float4(0.0);
    gbuffer_specular[pixel_index] = float4(0.0);
    gbuffer_motion[pixel_index] = float4(0.0);

    // ReSTIR (estimator flag bit 1): clear this pixel's current-parity
    // reservoir so a primary miss / non-opaque hit can't leave 2-frame-old
    // history in the slot.
    if ((asuint(camera.atmo.w) & 2) != 0)
    {
        let parity = camera.frame.x & 1;
        Reservoir zero_r;
        zero_r.light_id = 0;
        zero_r.seed = 0;
        zero_r.m = 0.0;
        zero_r.w = 0.0;
        zero_r.normal_oct = 0;
        zero_r.depth = 0.0;
        zero_r.pad_a = 0;
        zero_r.pad_b = 0;
        reservoirs[pixel_index * 2 + parity] = zero_r;
    }

#ifdef SOLARI_SHADER_CLOCK
    // Per-pixel cost measurement (heatmap). Read the start clock (32-bit LO)
    // and fold it into the RNG so the compiler can't sink the read past the
    // bounce loop: the loop consumes `rng`, which now depends on `clk0`,
    // pinning the read before it. Heatmap-mode only (frame.z == 1).
    var clk0 = 0u;
    if (camera.frame.z == 1)
    {
        clk0 = getRealtimeClock().x;
    }
#endif

    for (uint s = 0; s < rounds; s++)
    {
        // Per-sample RNG stream: (pixel, frame, sample) so every path is
        // decorrelated.
        rng = pixel_index + (camera.frame.x * rounds + s) * 5782582u;
#ifdef SOLARI_SHADER_CLOCK
        rng = rng ^ clk0; // pins the start clock read before the loop
#endif
        // Sub-pixel jitter: DLSS's when realtime; uniform pixel-area AA when
        // accumulating. Estimator flag bit 21 disables the AA jitter (pixel
        // CENTER every sample) — the target-mismatch isolation lever.
        var jitter = camera.jitter.xy;
        if (reference && (asuint(camera.atmo.w) & (1u << 21)) == 0)
        {
            jitter = float2(rand_f(rng), rand_f(rng)) - 0.5;
        }
        let pixel = float2(id) + 0.5 + jitter;
        var origin = camera.camera_position.xyz;
        float3 direction;
        if (camera.window_arc.y > 0.0)
        {
            // Cylindrical window (head-coupled curved screen): no matrix maps
            // to a curved surface — the ray goes from the tracked eye through
            // this pixel's physical point on the screen cylinder.
            let uv = pixel / float2(dims);
            // .w slots offset the view rect (OS window) from the monitor center
            let theta = camera.window_arc.w + (uv.x - 0.5) * camera.window_arc.x;
            let r = camera.window_arc.y;
            let p_screen = float3(
                r * sin(theta),
                camera.window_eye.w + (0.5 - uv.y) * camera.window_arc.z,
                r * (1.0 - cos(theta))); // concave: edges bow toward the viewer
            let d_view = p_screen - camera.window_eye.xyz;
            direction = normalize(mat4_mul(camera.world_from_view, float4(d_view, 0.0)).xyz);
        }
        else
        {
            let ndc = (pixel / float2(dims)) * 2.0 - 1.0;
            let far = mat4_mul(camera.inverse_view_proj, float4(ndc.x, -ndc.y, 1.0, 1.0));
            direction = normalize(far.xyz / far.w - origin);
        }
        // Primary ray (pre-bounce) for the atmosphere-volume march.
        let cam_origin = origin;
        cam_direction = direction;
        var primary_t = 1e30;

        var radiance = float3(0.0);
        var throughput = float3(1.0);
        var captured = false;
        // pdf of the BRDF sample that produced this segment (0 on the primary
        // ray), threaded into the hit shader so it can MIS-weight its
        // emissive vs NEE.
        var p_bounce = 0.0;
        // NRC spread-based termination (paper §5.1): accumulate the ray's
        // footprint spread; terminate into the cache once it exceeds a
        // fraction of the primary vertex's.
        var nrc_spread = 0.0;
        var nrc_prev_pdf = 1.0;
        // GI split: radiance ≡ di0 + a0·gi_L (a0 = primary BSDF weight,
        // gi_L = suffix radiance with a0 divided out).
        var di0 = float3(0.0);
        var a0 = float3(0.0);
        var gi_L = float3(0.0);
        var gi_throughput = float3(0.0);
        var gi_xs = float3(0.0);
        var gi_ns_oct = 0u;
        var gi_pdf1 = 0.0;
        var gi_hit = false;
        var gi_e1 = float3(0.0);

        // Path-length cap: estimator bits 22..27 carry the max INDIRECT
        // bounce count. With NRC armed the capped vertex terminates INTO the
        // cache, and training paths are EXEMPT from the cap — render short,
        // train long.
        let bounce_cap = min((asuint(camera.atmo.w) >> 22) & 0x3f, MAX_BOUNCES);
        var sample_cap = bounce_cap;
        if (nrc_training && s == 0)
        {
            sample_cap = MAX_BOUNCES;
        }
        else if (nrc_gi_on)
        {
            // Sharp-footprint chains (mirror/glossy prefixes: delta pdf ⇒ the
            // spread accumulator adds ~0) DEFER their cache query past the
            // path cap — give them rope to the bounce-5 hard stop.
            sample_cap = max(bounce_cap, 5u);
        }
        for (uint bounce = 0; bounce < sample_cap + 1; bounce++)
        {
            // Black-hole geodesic (stand-in): bend the ray toward the mass
            // and terminate if it crosses the capture radius.
            if (BH_STRENGTH > 0.0)
            {
                let to_center = BH_CENTER - origin;
                let dist = length(to_center);
                if (dist < BH_CAPTURE_RADIUS)
                {
                    captured = true;
                    break;
                }
                direction = normalize(
                    direction + (BH_STRENGTH / (dist * dist)) * normalize(to_center));
            }

            RtPayload payload;
            payload.emitted = float3(0.0);
            payload.emissive_mis = float3(0.0);
            payload.attenuation = float3(0.0);
            payload.next_origin = origin;
            payload.next_direction = direction;
            payload.bounce = 0;
            payload.rng = rng;
            payload.p_bounce = p_bounce;
            payload.anyhit_count = 0;
            // Sentinel so a primary miss (sky) reads as "no cluster" (the miss
            // shader doesn't write these); the closest-hit overwrites on a hit.
            payload.hit_cluster = 0xffffffff;
            payload.hit_material = 0xffffffff;
            payload.hit_primitive = 0;
            payload.hit_normal_oct = 0;
            payload.hit_geo_normal_oct = 0;
            // Only the primary hit produces the visible guide / reservoir
            // write; later bounces pass the sentinel so their closest-hit
            // skips both.
            payload.gbuffer_pixel = bounce == 0 ? pixel_index : NO_GBUFFER;
            // Shader Execution Reordering: trace into a hit object, regroup
            // the warp by MATERIAL, then run the selected closest-hit. With
            // per-material SBT records, the hit object's SBT record index IS
            // the material id — read it back and feed it as an explicit
            // coherence hint. Every record runs the same opaque closest-hit,
            // so the default reorder (which keys on the shader to run) would
            // NOT separate materials; the explicit hint does. `camera.frame.y`
            // carries ceil(log2(material_count)) hint bits.
            RayDesc ray;
            ray.Origin = origin;
            ray.Direction = direction;
            ray.TMin = RAY_T_MIN;
            ray.TMax = RAY_T_MAX;
            HitObject hit = HitObject::TraceRay(
                tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);
            let material_hint = hit.GetShaderTableIndex();
            ReorderThread(hit, material_hint, camera.frame.y);
            HitObject::Invoke(tlas, hit, payload);
            rng = payload.rng;
            // Count any-hits on the PRIMARY ray only — the any-hit cost of
            // the directly-visible pixel. Bounce rays would otherwise paint
            // nearby foliage onto the surfaces they illuminate.
            if (bounce == 0)
            {
                total_anyhit = payload.anyhit_count;
                // Capture the primary hit for the geometry-debug views.
                primary_cluster = payload.hit_cluster;
                primary_primitive = payload.hit_primitive;
                primary_normal_oct = payload.hit_normal_oct;
                primary_geo_normal_oct = payload.hit_geo_normal_oct;
                // NRC debug view: the paint queries the cache with the
                // primary hit's material/position.
                if (camera.frame.z == 6)
                {
                    nrc_dbg_material = payload.hit_material;
                    nrc_dbg_pos = payload.next_origin;
                }
            }

            // Capture the PRIMARY hit's depth on bounce 0. A miss leaves
            // `payload.next_origin` at the camera ray origin, so a moved
            // origin marks a hit. `origin` is still the primary ray origin
            // here — it's advanced below.
            if (bounce == 0)
            {
                let hit_pos = payload.next_origin;
                if (dot(hit_pos - origin, hit_pos - origin) > 1e-10)
                {
                    let clip = mat4_mul(camera.clip_from_world, float4(hit_pos, 1.0));
                    primary_depth = clip.z / clip.w;
                    primary_t = length(hit_pos - origin);
                }
            }

            // DLSS specular hit-distance guide. The primary surface's first
            // continuation ray only reveals where it lands HERE, after bounce
            // 1. Record that world-space distance (primary hit → bounce-1
            // hit) into the primary pixel's specular `.w`.
            if (bounce == 1 && payload.bounce == 1)
            {
                gbuffer_specular[pixel_index].w = length(payload.next_origin - origin);
            }

            // Fog extinction (stand-in): attenuate over the traversed segment.
            if (FOG_DENSITY > 0.0)
            {
                let seg = length(payload.next_origin - origin);
                throughput *= exp(-FOG_DENSITY * seg);
            }

            // NRC vertex capture (sample 0 only — one path per pixel): inputs
            // straight into the ring, prefix radiance + incoming throughput
            // in registers for the backward target pass after the walk.
            if (nrc_training && s == 0 && bounce < 4
                && payload.hit_cluster != 0xffffffff
                && payload.hit_material != 0xffffffff)
            {
                let ms = nrc_unpack_material(payload.hit_material);
                let nrm_cs = nrc_cyl(octahedral_decode_signed(
                    unpack_snorm2x16(payload.hit_normal_oct)));
                let dir_cs = nrc_cyl(-direction);
                NrcRecord rec;
                // Anchor-relative position — the encode pass only scales + biases.
                rec.pos_rough = float4(
                    payload.next_origin + camera.nrc_anchor.xyz, ms.rough);
                rec.dir_normal_cs = float4(dir_cs, nrm_cs);
                rec.diff_target_r = float4(ms.diff_alb, 0.0);
                rec.spec_target_g = float4(ms.spec_alb, 0.0);
                // .y = valid (for the scatter, which runs before the backward
                // target pass rewrites this), .z = pixel, .w = bounce.
                rec.target_b_valid = float4(0.0, 1.0, float(pixel_index), float(bounce));
                nrc_records[nrc_base + bounce] = rec;
                nrc_prefix_rad[bounce] = uint2(
                    pack_half2x16(radiance.xy), pack_half2x16(float2(radiance.z, 0.0)));
                nrc_atten[bounce] = uint2(
                    pack_half2x16(throughput.xy), pack_half2x16(float2(throughput.z, 0.0)));
                nrc_mask |= 1u << bounce;
            }

            // Grow the footprint spread by this segment (length / sqrt(pdf
            // that sampled the ray into this vertex)). Squared → an area.
            if (bounce > 0)
            {
                let seg = length(payload.next_origin - origin);
                nrc_spread += seg / sqrt(max(nrc_prev_pdf, 1.0e-3));
            }
            let nrc_spread_area = nrc_spread * nrc_spread;
            let nrc_prim_area = primary_t * primary_t;
            // Terminate when the footprint has spread past c·(primary area),
            // never before bounce 2, always by bounce 5.
            let nrc_spread_hit = nrc_spread_area
                > camera.nrc.y * max(nrc_prim_area, 1.0e-6);

            // Cache GI termination: spread-gated, at SECONDARY vertices —
            // this is the actual cache speedup. Default: append a query to
            // the ring and end the path; the post-trace nrc_query_infer pass
            // batch-evaluates the MLP and composites into this pixel.
            // Training paths (and camera.nrc.z, and ReSTIR GI) evaluate the
            // MLP inline instead: the cache tail must land IN-LOOP for the
            // targets to bootstrap through the termination.
            // `bounce >= bounce_cap` folds the path-length cap in: the last
            // allowed vertex queries the cache instead of dropping the tail —
            // but ONLY once the footprint has spread (nrc_spread_hit): a
            // mirror/glossy prefix has ~zero spread at the cap, and querying
            // there paints the cache's blurry answer into a pinhole-sharp
            // reflection. Sharp chains keep tracing REAL transport and query
            // where the footprint widens.
            let nrc_term = bounce >= 5 || (bounce >= 2 && nrc_spread_hit)
                || (bounce >= bounce_cap && nrc_spread_hit);
            // Training paths terminate deeper (16× the spread threshold,
            // bounces 3-7): their TD targets then carry more measured bounces
            // before the cache bootstrap.
            let nrc_term_train = bounce >= 7
                || (bounce >= 3 && nrc_spread_area
                    > 16.0 * camera.nrc.y * max(nrc_prim_area, 1.0e-6));
            let nrc_train_path = nrc_training && s == 0;
            if (nrc_gi_on
                && !(nrc_unbiased && s == 0)
                && payload.hit_cluster != 0xffffffff
                && payload.hit_material != 0xffffffff
                && (nrc_train_path ? nrc_term_train : nrc_term))
            {
                let pos_unit = clamp(
                    (payload.next_origin + camera.nrc_anchor.xyz) / camera.nrc.x + 0.5,
                    float3(0.0), float3(1.0));
                if (nrc_inline_on || nrc_train_path || nrc_restir_gi)
                {
                    let ms = nrc_unpack_material(payload.hit_material);
                    let nrm_cs = nrc_cyl(octahedral_decode_signed(
                        unpack_snorm2x16(payload.hit_normal_oct)));
                    let dir_cs = nrc_cyl(-direction);
                    let cache = nrc_query(
                        pos_unit, dir_cs, nrm_cs, ms.rough, ms.diff_alb, ms.spec_alb);
                    let tail = cache
                        * (ms.diff_alb + ms.spec_alb + float3(1.0e-2))
                        / max(camera.camera_position.w, 1.0e-9);
                    radiance += throughput * tail;
                    // Keep the radiance ≡ di0 + a0·gi_L split exact: cache
                    // termination happens past bounce 1, where throughput =
                    // a0·gi_throughput — so the GI reservoir's stored suffix
                    // carries the tail too.
                    gi_L += gi_throughput * tail;
                    break;
                }
                else if (s == 0)
                {
                    // One query per pixel per frame keeps the composite
                    // race-free; a full ring keeps tracing (unbiased fallback).
                    uint slot;
                    nrc_queries.InterlockedAdd(0, 1, slot);
                    if (slot < dims.x * dims.y)
                    {
                        let q_off = NRC_QUERY_BASE + slot * NRC_QUERY_STRIDE;
                        nrc_queries.Store(q_off, uint4(
                            asuint(pos_unit.x),
                            asuint(pos_unit.y),
                            asuint(pos_unit.z),
                            payload.hit_material));
                        nrc_queries.Store(q_off + 16, uint4(
                            pack_unorm2x16(nrc_cyl(octahedral_decode_signed(
                                unpack_snorm2x16(payload.hit_normal_oct)))),
                            pack_unorm2x16(nrc_cyl(-direction)),
                            pack_half2x16(throughput.rg),
                            pack_half2x16(float2(throughput.b, 0.0))));
                        nrc_queries.Store(q_off + 32, uint4(pixel_index, 0, 0, 0));
                        break;
                    }
                }
            }

            radiance += throughput * payload.emitted;
            if (bounce == 0)
            {
                di0 = payload.emitted;
            }
            else
            {
                gi_L += gi_throughput * payload.emitted;
            }
            // Reconnection vertex: the bounce-1 hit (offset origin + its normal).
            if (bounce == 1)
            {
                gi_hit = payload.hit_cluster != 0xffffffff;
                gi_xs = payload.next_origin;
                gi_ns_oct = payload.hit_normal_oct;
                gi_e1 = payload.emissive_mis;
            }
            if (payload.bounce == 0)
            {
                break;
            }
            throughput *= payload.attenuation;
            if (bounce == 0)
            {
                a0 = payload.attenuation;
                gi_throughput = float3(1.0);
                gi_pdf1 = payload.p_bounce;
            }
            else
            {
                gi_throughput *= payload.attenuation;
            }
            origin = payload.next_origin;
            direction = payload.next_direction;
            p_bounce = payload.p_bounce;
            nrc_prev_pdf = payload.p_bounce;

            // Never feed a degenerate (zero-length or non-finite) direction to
            // the next trace — the RT core hangs the GPU on a zero-length ray.
            // `dot > eps` is false for both zero and NaN.
            if (!(dot(direction, direction) > 1.0e-8))
            {
                break;
            }

            // Russian roulette (unbiased — the ÷p compensates; capped below 1
            // so lossless paths still terminate). Reference mode defers it to
            // bounce 3; realtime keeps the aggressive kill.
            if (!reference || bounce >= 3)
            {
                let p = min(luminance(throughput), 0.95);
                if (rand_f(rng) > p)
                {
                    break;
                }
                throughput /= p;
                // Mirror the RR compensation into the split (a0 owns bounce 0's).
                if (bounce == 0)
                {
                    a0 /= p;
                }
                else
                {
                    gi_throughput /= p;
                }
            }
        }

        // Estimator overrides, all built on the di0/GI split.
        let eflags = asuint(camera.atmo.w);
        var gi_out = a0 * gi_L;
        var dead_draw = 0.0;
        // ReSTIR GI (flag bit 5): store this pixel's canonical sample, then
        // shade GI from the store. Dead samples (sky/delta, pdf=0) keep the
        // live a0·gi_L.
        if ((eflags & 32) != 0)
        {
            // ReSTIR GI, raygen side: EXPORT ONLY. This block once held the
            // temporal merge + reservoir shade and cost ~90 ms/frame at 16 spp
            // on room: raygen's live-across-trace state leaves no register
            // headroom, so a composite local spills to local memory, and
            // holding the merge state in scalars blows register allocation
            // instead. The merge + shade run in `restir_spatial.slang`'s
            // `gi_finalize` dispatch (fresh register file); raygen writes the
            // canonical sample from values ALREADY LIVE in registers and
            // touches no other memory.
            let l_reuse = gi_L - gi_e1;
            let canon_ok = gi_hit && gi_pdf1 > 0.0 && gi_pdf1 < 1.0e30;
            let cur_slot = pixel_index * 2 + (camera.frame.x & 1);
            let has_surface = primary_cluster != 0xffffffff;
            // Only the LAST sample's canonical survives to `gi_finalize`
            // (each store overwrites the slot and nothing reads it mid-loop),
            // so skip the export for samples 0..rounds-1.
            if (s + 1 == rounds)
            {
                gi_samples[cur_slot].pos_x = gi_xs.x;
                gi_samples[cur_slot].pos_y = gi_xs.y;
                gi_samples[cur_slot].pos_z = gi_xs.z;
                gi_samples[cur_slot].normal_oct = gi_ns_oct;
                gi_samples[cur_slot].l_r = l_reuse.x;
                gi_samples[cur_slot].l_g = l_reuse.y;
                gi_samples[cur_slot].l_b = l_reuse.z;
                gi_samples[cur_slot].w = canon_ok ? 1.0 / max(gi_pdf1, 1.0e-9) : 0.0;
                gi_samples[cur_slot].a0_r = a0.x;
                gi_samples[cur_slot].a0_g = a0.y;
                gi_samples[cur_slot].a0_b = a0.z;
                // The draw counts toward M even when dead (W averages ALL draws).
                gi_samples[cur_slot].m = 1.0;
                // Generating-surface fields: `gi_finalize` fills them from the
                // chit's SurfaceGbuf; view_z carries only the has-surface
                // sentinel here (0 = sky, next frame's validation rejects).
                gi_samples[cur_slot].surf_normal_oct = 0;
                gi_samples[cur_slot].surf_view_z = has_surface ? 1.0 : 0.0;
                gi_samples[cur_slot].pad_a = 0;
                gi_samples[cur_slot].pad_b = 0;
            }
            // Raygen keeps the per-pixel terms reuse can't carry: bounce-1
            // emission (this pixel's DI-by-MIS partner) and the dead-sample
            // live fallback. The reservoir shade lands in the output buffer
            // from `gi_finalize`.
            var gi_env = float3(0.0);
            if (!canon_ok)
            {
                gi_env = a0 * l_reuse;
            }
            gi_out = a0 * gi_e1 + gi_env;
            // Realtime firefly filter (dims.w; the reference passes 0 = off)
            // on the raygen-owned terms; `gi_finalize` clamps its own add.
            if (camera.dims.w > 0.0)
            {
                let gi_lum = luminance(gi_out);
                if (gi_lum > camera.dims.w)
                {
                    gi_out *= camera.dims.w / gi_lum;
                }
            }
            radiance = di0 + gi_out;
            dead_draw = canon_ok ? 0.0 : 1.0;
        }
        // GI-only estimator (flag bit 4): the complement of di_only —
        // di_only + gi_only must sum to the full image.
        if ((eflags & 16) != 0)
        {
            radiance = gi_out;
        }
        // Dead-rate instrument (flag bit 17): paint the dead-canonical
        // indicator; the accumulated mean IS the per-pixel dead-draw rate.
        if ((eflags & 131072) != 0)
        {
            radiance = float3(dead_draw);
        }

        // Atmosphere volumes: attenuate + in-scatter over the primary segment
        // (aerial perspective / limb / sky-through-shell). Per sample.
        radiance = atmosphere_volumes_apply(radiance, cam_origin, cam_direction, primary_t);
        // NRC backward target pass: suffix radiance leaving vertex k toward
        // the camera path = (final − prefix_k) / throughput_into_k. RR keeps
        // throughput O(1) so the division is tame; guards catch the tail.
        if (nrc_training && s == 0)
        {
            // TD self-training (paper §5): target_k = emission_k +
            // atten_k · cache(vertex_{k+1}). One real BSDF step, the tail is
            // the cache bootstrapping itself (stop-gradient). Everything in
            // exposure-scaled physical radiance; the cache is factorized so
            // multiply the next vertex's albedo back before adding, divide
            // THIS vertex's out.
            let E = camera.camera_position.w;
            for (uint k = 0; k < 4; k++)
            {
                let slot = nrc_base + k;
                if ((nrc_mask & (1u << k)) == 0)
                {
                    nrc_records[slot].target_b_valid = float4(0.0);
                    continue;
                }
                let alb = nrc_records[slot].diff_target_r.xyz
                    + nrc_records[slot].spec_target_g.xyz + float3(1.0e-2);
                // Measured path suffix leaving vertex k (radiance from k
                // onward ÷ throughput into k). Clamp tames the
                // small-denominator tail.
                let a_p = nrc_atten[k];
                let atten_k = float3(unpack_half2x16(a_p.x), unpack_half2x16(a_p.y).x);
                let r_p = nrc_prefix_rad[k];
                let prefix_k = float3(unpack_half2x16(r_p.x), unpack_half2x16(r_p.y).x);
                let tin = max(atten_k, float3(3.0e-2));
                var tgt = (radiance - prefix_k) / tin * E;
                if (any(tgt != tgt))
                {
                    tgt = float3(0.0);
                }
                tgt = clamp(tgt / alb, float3(0.0), float3(256.0));
                nrc_records[slot].diff_target_r.w = tgt.r;
                nrc_records[slot].spec_target_g.w = tgt.g;
                // .z/.w keep the capture's pixel/bounce (record debugging).
                nrc_records[slot].target_b_valid = float4(
                    tgt.b, 1.0, float(pixel_index), float(k));
            }
        }

        frame_sum += captured ? float3(0.0) : radiance;
    } // sample loop

    // The output buffer holds PHYSICAL radiance — the display blit applies
    // the camera exposure at read, so accumulation/dumps/probes/debug paints
    // stay in physical units and an exposure change never resets the mean.
    var final_color = frame_sum / float(rounds);

    // Reference accumulation: fold this frame's average into the running mean
    // held in the output buffer.
    if (reference && accum_n > 0.0)
    {
        let w = float(ref_spf) / (accum_n + float(ref_spf));
        final_color = lerp(output[pixel_index].rgb, final_color, w);
    }

#ifdef SOLARI_SHADER_CLOCK
    // Cost heatmap debug view (frame.z == 1): replace the shaded color with a
    // colormap of the clock delta across the whole trace.
    if (camera.frame.z == 1)
    {
        let cycles = float(clock_delta(clk0, getRealtimeClock().x));
        // Per-pixel cost spans orders of magnitude → map log2(cycles) →
        // color, pivoting at the data center so the knob is a CONTRAST
        // control, not a shift. jitter.z = center; jitter.w = contrast.
        let t = 0.5 + (log2(max(cycles, 1.0)) - camera.jitter.z) * camera.jitter.w;
        final_color = cost_heatmap(t);
    }
#endif

    // Any-hit invocation heatmap (frame.z == 2): overlay ONLY where the
    // primary ray fired the alpha any-hit. Zero-any-hit pixels keep their
    // real shaded color. Warm ramp yellow → red by count; jitter.z scales
    // count → [0, 1].
    if (camera.frame.z == 2 && total_anyhit > 0)
    {
        let t = clamp(float(total_anyhit) * camera.jitter.z, 0.0, 1.0);
        final_color = lerp(float3(1.0, 1.0, 0.0), float3(1.0, 0.0, 0.0), t);
    }

    // Geometry-debug views: flat per-cluster (z==3) / per-triangle (z==4)
    // color, a hash of the primary hit's ids → distinct hue. A primary miss
    // (sentinel cluster) keeps its real (sky) color.
    if ((camera.frame.z == 3 || camera.frame.z == 4) && primary_cluster != 0xffffffff)
    {
        var key = primary_cluster;
        if (camera.frame.z == 4)
        {
            key = primary_cluster * 0x9e3779b1 + primary_primitive;
        }
        final_color = id_hash_color(key);
    }

    // Normal-facing debug view (frame.z == 5): classify the primary hit.
    //   RED    = GEOMETRIC (winding) normal faces away → back-wound triangle.
    //   ORANGE = shading normal >90° from the true surface → inverted vertex
    //            normal (bad data).
    //   YELLOW = shading normal valid but tilted past the VIEW horizon →
    //            benign low-poly grazing.
    //   BLUE   = shading normal faces you (correct); brightness = facing.
    if (camera.frame.z == 5 && primary_cluster != 0xffffffff)
    {
        let ns = octahedral_decode_signed(unpack_snorm2x16(primary_normal_oct));
        let ng = octahedral_decode_signed(unpack_snorm2x16(primary_geo_normal_oct));
        let wo = -cam_direction;
        let sfacing = dot(ns, wo);
        let gfacing = dot(ng, wo);
        let sg = dot(ns, ng); // shading vs true surface: < 0 means inverted
        if (gfacing < 0.0)
        {
            final_color = float3(1.0, 0.1, 0.1) * (0.15 + 0.85 * abs(gfacing));
        }
        else if (sg < 0.0)
        {
            final_color = float3(1.0, 0.45, 0.05) * (0.15 + 0.85 * abs(sg));
        }
        else if (sfacing < 0.0)
        {
            final_color = float3(1.0, 0.9, 0.15) * (0.15 + 0.85 * abs(sfacing));
        }
        else
        {
            final_color = float3(0.1, 0.4, 1.0) * (0.15 + 0.85 * sfacing);
        }
    }

    // NRC debug view (frame.z == 6): paint the cache prediction at the
    // primary hit (exposure-scaled radiance — directly displayable since the
    // blit uses exposure 1.0 for debug views).
    if (camera.nrc.x > 0.0)
    {
        if (camera.frame.z == 6 && nrc_dbg_material != 0xffffffff)
        {
            let ms = nrc_unpack_material(nrc_dbg_material);
            let nrm_cs = nrc_cyl(octahedral_decode_signed(
                unpack_snorm2x16(primary_normal_oct)));
            let pos_unit = clamp(
                (nrc_dbg_pos + camera.nrc_anchor.xyz) / camera.nrc.x + 0.5,
                float3(0.0), float3(1.0));
            final_color = max(nrc_query(
                pos_unit,
                nrc_cyl(-cam_direction),
                nrm_cs,
                ms.rough,
                ms.diff_alb,
                ms.spec_alb), float3(0.0)) * (ms.diff_alb + ms.spec_alb + float3(1.0e-2));
        }
    }

    let index = id.y * dims.x + id.x;
    // Alpha carries the primary-hit depth for the gizmo-depth bridge. The
    // blit forces the displayed alpha back to 1.0, and DLSS resolve reads
    // only the G-buffers, so this never affects the displayed image.
    output[index] = float4(final_color, primary_depth);
}
