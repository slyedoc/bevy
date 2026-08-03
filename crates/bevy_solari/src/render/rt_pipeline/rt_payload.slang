// Shared ray-payload + camera types for the RT-pipeline shaders. Every stage
// that carries a payload is Slang and imports these definitions, so payload
// layout agreement is by construction. (Historical Law 1 — the driver maps
// payloads MEMBER-WISE between stages, so a reshaped mirror silently drops
// writes on SER-executed hits — still applies if a non-Slang stage ever
// carries the payload again.)
module rt_payload;

// Per-bounce path state carried across the trace. raygen owns the bounce loop;
// a hit shader fills `emitted` + (if continuing) `attenuation` and the next
// ray, and threads the RNG back out. `anyhit_count` MUST stay the first field:
// `ahit_alpha` aliases the payload's first word (it runs for both `RtPayload`
// and `ShadowPayload` rays).
public struct RtPayload
{
    // Any-hit invocation counter for the OMM-effectiveness debug heatmap.
    // raygen resets it before each trace and sums across bounces.
    public uint anyhit_count;
    public float3 emitted;      // radiance contributed at this vertex (sky at miss)
    // The mis-weighted emission part of `emitted` (zero from miss/glass; raygen
    // clears it per bounce). At bounce 1 this is the pixel's DI-by-MIS term.
    public float3 emissive_mis;
    public float3 attenuation;  // throughput multiplier for the next segment
    public float3 next_origin;
    public float3 next_direction;
    public uint bounce;         // 1 = continue, 0 = terminate (miss / absorb)
    public uint rng;            // PCG state, advanced by the hit shader's sampling
    // pdf of the BRDF sample that generated the ray INTO this vertex (0 on the
    // primary ray). Threaded out by a hit shader so the NEXT vertex can
    // MIS-weight its emissive against next-event estimation.
    public float p_bounce;
    // Geometry-debug views (cluster / triangle color): global cluster id +
    // cluster-local primitive index of the hit; raygen captures them on the
    // PRIMARY hit. `hit_cluster` carries a sentinel when the primary ray missed.
    public uint hit_cluster;
    public uint hit_primitive;
    // Normal-facing debug view: primary hit's SHADING normal (pre-bend, world
    // space) and GEOMETRIC normal, oct-packed — raygen tells a back-wound
    // triangle (red) from a merely bad vertex normal (yellow).
    public uint hit_normal_oct;
    public uint hit_geo_normal_oct;
    // NRC: packed surface material at the hit (r5g6b5 base color, metallic8,
    // roughness8); 0xffffffff = no material (miss / non-opaque chit).
    public uint hit_material;
    // Pixel index (row-major `y*width + x`) on the PRIMARY bounce, sentinel
    // after. The DLSS guides + ReSTIR reservoir writes both key off it.
    public uint gbuffer_pixel;
}

// Shadow-ray payload: two u32 words. The first word mirrors
// `RtPayload.anyhit_count` so `ahit_alpha`'s single-word alias targets the
// counter whichever ray type invoked it.
public struct ShadowPayload
{
    public uint anyhit_count;
    public uint occluded;
}

// ── Camera UBO (set 1 binding 1) ────────────────────────────────────────────
// This is a LAYOUT mirror (not a payload): naga stores each mat4x4 as four
// column vec4s (ColMajor, MatrixStride 16). `Mat4` spells the columns out and
// `mat4_mul` reproduces WGSL's `m * v` — sidestepping the Slang/HLSL
// row/column-major mapping entirely.

public struct Mat4
{
    public float4 c0;
    public float4 c1;
    public float4 c2;
    public float4 c3;
}

// WGSL `m * v` (column combination).
public float4 mat4_mul(Mat4 m, float4 v)
{
    return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z + m.c3 * v.w;
}

// The camera UBO layout the CPU writes (std140: matrices @0/64/128/192, vec4s
// @256..368, world_from_view @384, vec4s @448..512). The Rust `RtCamera`
// mirrors this — keep them field-for-field.
public struct RtCamera
{
    public Mat4 inverse_view_proj;
    public Mat4 view_from_world;      // DLSS guide: view-space linear depth
    public Mat4 clip_from_world;      // DLSS motion vectors: current (unjittered)
    public Mat4 prev_clip_from_world; // DLSS motion vectors: previous (unjittered)
    public float4 camera_position;
    public uint4 frame;        // .x = frame index (RNG seed); .y = SER hint bits; .z = debug view
    public float4 sky;         // .x = environment brightness (cd/m²); .yzw = clear color
    public float4 jitter;      // .xy = sub-pixel camera jitter (pixels); .zw = heatmap params
    public float4 misc;        // .x = time (s); .y = pixel ray-cone tan; .zw reserved
    public float4 sky_frame;   // world→bake sky quaternion (xyzw)
    public float4 atmo;        // .xy = volume-buffer address (lo/hi); .z = volume count; .w = estimator flags
    public float4 dims;        // .xy = viewport pixels; .z = history M-cap ×M
    public Mat4 world_from_view; // camera basis: view→world ray dirs
    public float4 window_arc;  // .x = arc angle (rad); .y = radius m; .z = height m
    public float4 window_eye;  // .xyz = eye in screen space
    public float4 origin_delta; // .xyz = origin_now − origin_prev: cross-frame rebase
    public float4 nrc;         // .x = NRC scene scale m; .y = spread c; .z = inline coopvec
    public float4 nrc_anchor;  // .xyz = camera-relative → anchor-relative offset
}
