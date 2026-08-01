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
