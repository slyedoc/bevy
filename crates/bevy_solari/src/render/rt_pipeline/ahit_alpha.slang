// Any-hit alpha-cutout test — the RT-pipeline analog of the inline rayQuery
// candidate test (`scene_bindings::alpha_test`), for foliage / fences / cutouts.
//
// Attached to the OPAQUE triangle hit group. The hardware runs an any-hit ONLY for
// non-opaque geometry, and PTLAS marks exactly the alpha-masked instances
// `FORCE_NO_OPAQUE` (`material.traversal_alpha_cutoff() >= 0`), so opaque geometry
// commits in hardware and pays nothing here. When the base-color alpha at the hit
// UV is below the material's cutoff the texel is a hole, so `IgnoreHit()`
// rejects the candidate and the ray continues through it — for primary, bounce, AND
// shadow rays (shadow rays are not force-opaque, so foliage casts cutout
// shadows; see chit_opaque `SHADOW_RAY_FLAGS`).
//
// Regenerate SPIR-V (from crates/bevy_solari/src/render/rt_pipeline/):
//   slangc ahit_alpha.slang -target spirv -o ahit_alpha.spv
//
// Binding/layout contract: every declaration here mirrors
// bindings/raytracing_scene_bindings.wgsl and MUST keep byte-identical
// layouts (the descriptor sets are created against the WGSL stages;
// rt_shaders_compile cross-checks the Cluster/Material offsets).

// Mirrors `Cluster` (std430 stride 48: bounds_sphere vec4 @ 16).
struct Cluster
{
    uint vertex_offset;
    uint vertex_count;
    uint index_offset;
    uint triangle_count;
    float4 bounds_sphere;
    uint local_material_id;
    uint lod_level;
    uint pad_a;
    uint pad_b;
}

// Mirrors `SolariGeometryAddresses` (set 1 binding 4): the bindless BDA table.
struct SolariGeometryAddresses
{
    uint64_t vertex_packed;
    uint64_t vertex_positions;
    uint64_t materials;
    uint material_stride;
    uint pad;
    uint64_t tess_clusters;
    uint64_t vertex_custom;
    uint64_t deform_normals;
    uint64_t deform_tangents;
    uint64_t animated_table;
}

static const uint TEXTURE_MAP_NONE = 0xFFFFFFFF;
// Mirrors `bindings::MAX_TEXTURE_COUNT` (the descriptor-set array size).
static const uint MAX_TEXTURE_COUNT = 5000;

[[vk::binding(0, 0)]] StructuredBuffer<uint> cluster_indices;
[[vk::binding(1, 0)]] StructuredBuffer<Cluster> clusters;
[[vk::binding(2, 0)]] Texture2D textures[MAX_TEXTURE_COUNT];
[[vk::binding(3, 0)]] SamplerState samplers[MAX_TEXTURE_COUNT];
[[vk::binding(4, 1)]] ConstantBuffer<SolariGeometryAddresses> geometry_addresses;
// GPU scene column (set 2 = SOLARI_SCENE_COLUMNS_GROUP).
[[vk::binding(2, 2)]] StructuredBuffer<uint> material_ids;

// Byte offsets into the bindless `Material` record (validated against the
// naga-emitted struct by rt_shaders_compile): only the two traversal fields
// are read here; the full mirror arrives with the closest-hit ports.
static const uint MATERIAL_BASE_COLOR_TEXTURE_ID_OFFSET = 4;
static const uint MATERIAL_ALPHA_MASK_OFFSET = 76;

// ClusterIDNV — the canonical cluster index into `clusters[]`, exactly as the
// closest-hit uses it. NOT GeometryIndex(): in the pipeline that's the raw
// per-CLAS geometry index (≈0), which would read the wrong cluster's UVs and
// fail the alpha test everywhere.
uint cluster_id()
{
    return spirv_asm
    {
        OpCapability RayTracingClusterAccelerationStructureNV;
        OpExtension "SPV_NV_cluster_acceleration_structure";
        result: $$uint = OpLoad builtin(ClusterIDNV: uint)
    };
}

// `vertex_packed` is a 16-byte record per vertex (normal@0, tangent@4, uv@8).
float2 load_packed_uv(uint vertex_index)
{
    return *(float2*)(geometry_addresses.vertex_packed
        + uint64_t(vertex_index) * 16 + 8);
}

bool alpha_passes(uint material_id, uint cluster_index, uint primitive_index, float2 bary)
{
    let material_addr = geometry_addresses.materials
        + uint64_t(material_id) * uint64_t(geometry_addresses.material_stride);
    let alpha_mask = *(float*)(material_addr + MATERIAL_ALPHA_MASK_OFFSET);
    if (alpha_mask < 0.0)
    {
        return true; // opaque material on a non-opaque instance (stale flag)
    }
    let texture_id = *(uint*)(material_addr + MATERIAL_BASE_COLOR_TEXTURE_ID_OFFSET);
    if (texture_id == TEXTURE_MAP_NONE)
    {
        return alpha_mask <= 1.0;
    }
    let cluster = clusters[cluster_index];
    let idx_base = cluster.index_offset + primitive_index * 3;
    let uv0 = load_packed_uv(cluster.vertex_offset + cluster_indices[idx_base + 0]);
    let uv1 = load_packed_uv(cluster.vertex_offset + cluster_indices[idx_base + 1]);
    let uv2 = load_packed_uv(cluster.vertex_offset + cluster_indices[idx_base + 2]);
    let uv = uv0 * (1.0 - bary.x - bary.y) + uv1 * bary.x + uv2 * bary.y;
    let alpha = textures[NonUniformResourceIndex(texture_id)].SampleLevel(
        samplers[NonUniformResourceIndex(texture_id)], uv, 0.0).a;
    return alpha >= alpha_mask;
}

// Single-word payload alias: `RtPayload` and `ShadowPayload` both keep
// `anyhit_count` as their first word, so this safely targets the counter whichever
// ray invoked the any-hit. Incrementing it per invocation drives the
// OMM-effectiveness heatmap (raygen colormaps the per-pixel total) — with OMM, the
// RT cores resolve opaque/transparent micro-regions in hardware and the any-hit
// never runs there, so OMM-covered foliage reads "cold".
struct AlphaHitPayload
{
    uint anyhit_count;
}

[shader("anyhit")]
void ahit_alpha(inout AlphaHitPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    // Count every any-hit invocation (the cost OMM removes), pass or fail.
    payload.anyhit_count += 1;
    if (!alpha_passes(
            material_ids[InstanceIndex()], cluster_id(), PrimitiveIndex(),
            attribs.barycentrics))
    {
        IgnoreHit();
    }
}
