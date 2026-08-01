// Slang mirror of the ray payload in rt_payload.wgsl. The payload crosses the
// naga/Slang stage boundary, and the driver maps it MEMBER-WISE between each
// module's own view of the struct — so the member SHAPE must match the WGSL
// declaration exactly (same count, same types, same order). A byte-compatible
// but reshaped struct (e.g. scalarized vec3s) silently drops every write on
// SER-executed hits. Do not add, split, or reorder members here without the
// same change in rt_payload.wgsl.
module rt_payload;

// MUST MATCH RtPayload in rt_payload.wgsl member-for-member.
public struct RtPayload
{
    public uint anyhit_count;
    public float3 emitted;
    public float3 emissive_mis;
    public float3 attenuation;
    public float3 next_origin;
    public float3 next_direction;
    public uint bounce;
    public uint rng;
    public float p_bounce;
    public uint hit_cluster;
    public uint hit_primitive;
    public uint hit_normal_oct;
    public uint hit_geo_normal_oct;
    public uint hit_material;
    public uint gbuffer_pixel;
}

// MUST MATCH ShadowPayload in rt_payload.wgsl: two u32 words. The first word
// mirrors `RtPayload.anyhit_count` so `ahit_alpha`'s single-word alias targets
// the counter whichever ray type invoked it.
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

// Mirrors `RtCamera` in rt_payload.wgsl (std140: matrices @0/64/128/192,
// vec4s @256..368, world_from_view @384, vec4s @448..512).
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
