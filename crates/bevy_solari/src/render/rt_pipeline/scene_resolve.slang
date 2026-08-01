// Shared scene-resolve module for the RT-pipeline Slang stages — the Slang
// mirror of `bindings/raytracing_scene_bindings.wgsl`'s hit-resolve surface:
// the bindless geometry/material fetch (`resolve_triangle_data_full_mat_fetch`
// + callees), the alpha-cutout test, and the shared ray/affine helpers.
//
// Not compiled standalone: `import scene_resolve;` from a stage in this
// directory (slangc resolves same-dir imports). Each stage's regen command is
// in its own header.
//
// Layout contract: while any naga-compiled stage remains in the pipeline, the
// buffer-facing structs here (Cluster, Material, SolariGeometryAddresses,
// Affine) mirror the CPU-baked layout byte-for-byte — offset-diff against a
// naga stage dump when touching them (slang_plan.md, Law 2). The vertex pools
// are PACKED (12/36-byte strides): BDA float2/float3 loads on them must not
// assume vector alignment — use the element-wise load helpers.
module scene_resolve;

public static const uint TEXTURE_MAP_NONE = 0xFFFFFFFF;
// Mirrors `bindings::MAX_TEXTURE_COUNT` (the descriptor-set array size).
public static const uint MAX_TEXTURE_COUNT = 5000;
public static const float MIRROR_ROUGHNESS_THRESHOLD = 0.001;
public static const float RAY_T_MIN = 0.001;
// Effectively infinite ray length. Finite, not `inf`: an `inf` tmax produces
// NaN in the slab test (`inf * 0`) and silently drops hits.
public static const float RAY_T_MAX = 1.0e30;
public static const uint RAY_NO_CULL = 0xFF;

/// Tess CLAS `cluster_id` base — every tessellated CLAS bakes
/// `TESS_CLUSTER_ID_BASE + global_tess_clas_index` as its ClusterIDNV,
/// comfortably above any real cluster pool so the closest-hit detects the tess
/// hit and recovers the tess index. Must match `tess_template::TESS_CLUSTER_ID_BASE`.
static const uint TESS_CLUSTER_ID_BASE = 0xF0000000;

/// Minimum perceptual roughness for tessellated displacement hits. Sub-pixel
/// displacement detail acts as extra microfacet roughness; clamping the lobe
/// wider keeps direct-light specular from aliasing into DLSS flicker.
static const float TESS_ROUGHNESS_FLOOR = 0.5;

// Cluster pool record — mirrors `bevy_solari::cluster::asset::Cluster`
// (std430 stride 48: offsets 0,4,8,12,16,32,36 + explicit tail pads).
public struct Cluster
{
    public uint vertex_offset;
    public uint vertex_count;
    public uint index_offset;
    public uint triangle_count;
    public float4 bounds_sphere;
    public uint local_material_id;
    public uint lod_level;
    public uint pad_a;
    public uint pad_b;
}

// Full bindless `Material` record — naga offsets 0,4,8,12,16,28,32,44,48,60,
// 64,68,72,76,80,84,88,92,96,100,104,108,112 (the WGSL struct packs each vec3
// with a trailing scalar, so natural and std430 layout agree on every offset).
public struct Material
{
    public uint normal_map_texture_id;         // @0
    public uint base_color_texture_id;         // @4
    public uint emissive_texture_id;           // @8
    public uint metallic_roughness_texture_id; // @12
    public float3 base_color;                  // @16
    public float perceptual_roughness;         // @28
    public float3 emissive;                    // @32
    public float metallic;                     // @44
    // Beer-Lambert extinction per channel (1/world-unit) inside the volume.
    public float3 extinction;                  // @48
    public float reflectance;                  // @60
    public float ior;                          // @64
    public float specular_transmission;        // @68
    public uint nested_priority;               // @72
    // Alpha-mask cutoff; negative = opaque (no alpha test during traversal).
    public float alpha_mask;                   // @76
    // Chromatic dispersion (20/Abbe, `KHR_materials_dispersion`); 0 = none.
    public float dispersion;                   // @80
    // `0.5·log2(w·h)` of the base-color texture — the texture-size term of the
    // ray-cone LOD, baked CPU-side.
    public float texel_lod_bias;               // @84
    public uint displacement_texture_id;       // @88
    public float displacement_scale;           // @92
    public float displacement_bias;            // @96
    public uint texture_array_a_id;            // @100
    public uint texture_array_b_id;            // @104
    public uint texture_array_c_id;            // @108
    // Opaque per-material data for a custom closest-hit.
    public uint4 chit_data;                    // @112
}

public struct Vertex
{
    public float3 position;
    public float3 normal;
    public float2 uv;
    public float4 tangent;
}

public struct ResolvedMaterial
{
    public float3 base_color;
    public float3 emissive;
    public float reflectance;
    public float perceptual_roughness;
    public float roughness;
    public float metallic;
    public float specular_transmission;
    public float ior;
    public float dispersion;
    public float3 extinction;
    public uint nested_priority;
}

public struct ResolvedRayHitFull
{
    public float3 world_position;
    public float3 previous_frame_world_position;
    public float3 world_normal;
    public float3 geometric_world_normal;
    public float4 world_tangent;
    public float2 uv;
    public float triangle_area;
    public uint triangle_count;
    public ResolvedMaterial material;
    // Stable material slot — the identity nested-dielectric stacks pop on.
    public uint material_id;
}

// Row-packed affine (mat3x4: r_k = the 4x4's row k), std430 stride 48 —
// mirrors `transforms` / `previous_frame_transforms`.
public struct Affine
{
    public float4 r0;
    public float4 r1;
    public float4 r2;
}

// Mirrors `SolariGeometryAddresses` (set 1 binding 4, std140: u64s @0,8,16 +
// u32 @24, then u64s @32,40,48,56,64): the bindless BDA table.
public struct SolariGeometryAddresses
{
    public uint64_t vertex_packed;
    public uint64_t vertex_positions;
    public uint64_t materials;
    public uint material_stride;
    public uint pad;
    public uint64_t tess_clusters;
    public uint64_t vertex_custom;
    public uint64_t deform_normals;
    public uint64_t deform_tangents;
    public uint64_t animated_table;
}

public struct LightSource
{
    public uint kind; // 1 bit for kind, 31 bits for extra data
    public uint id;
}

public static const uint LIGHT_SOURCE_KIND_EMISSIVE_MESH = 0;
public static const uint LIGHT_SOURCE_KIND_DIRECTIONAL = 1;

// Mirrors `DirectionalLight` (std430 stride 32: 0,12,16,28).
public struct DirectionalLight
{
    public float3 direction_to_light;
    public float cos_theta_max;
    public float3 luminance;
    public float inverse_pdf;
}

// Per-instance LOD inputs, slot-indexed (16 B stride) — the RT path only needs
// `(cluster_base, cluster_count)` but MUST match the full stride.
public struct InstanceClusterRange
{
    public uint cluster_base;
    public uint cluster_count;
    public uint group_base;
    public uint root_group;
}

// ── Hair (linear swept spheres) ──
// Per-instance record (mirrors `GpuHairInstance`, Rust; std430 stride 64 —
// size 52 rounded up by the vec3's 16-alignment, pads made explicit).
public struct GpuHairInstance
{
    public float3 sigma_a;      // absorption from melanin (+ dye), CPU-derived
    public uint transform_slot; // index into `hair_world` (×3 rows)
    public uint segment_base;   // base into `hair_segments`
    public float beta_m;
    public float beta_n;
    public float alpha;
    public float ior;
    public uint blas_address_lo;
    public uint blas_address_hi;
    public uint mask;
    // Opaque-surface material slot for `SolariBranches` (bark/wood/etc), or
    // `HAIR_MATERIAL_NONE` for fiber hair shaded by the Chiang BSDF.
    public uint material_id;
    public uint pad_a;
    public uint pad_b;
    public uint pad_c;
}

// `GpuHairInstance.material_id` sentinel: shade as fiber hair, not an opaque surface.
public static const uint HAIR_MATERIAL_NONE = 0xFFFFFFFF;

// Per swept segment: the two capped-cylinder endpoints + radii, local space.
public struct HairSegment
{
    public float4 p0; // xyz position, w radius
    public float4 p1;
}

public struct HairSceneParams
{
    public uint base; // PTLAS instance_index of hair instance 0
    public uint count;
    public uint pad0;
    public uint pad1;
}

[[vk::binding(0, 0)]] public StructuredBuffer<uint> cluster_indices;
[[vk::binding(1, 0)]] public StructuredBuffer<Cluster> clusters;
[[vk::binding(2, 0)]] public Texture2D textures[MAX_TEXTURE_COUNT];
[[vk::binding(3, 0)]] public SamplerState samplers[MAX_TEXTURE_COUNT];
[[vk::binding(4, 0)]] public RaytracingAccelerationStructure tlas;
// STABLE-SLOT indexed: a light keeps its index for its lifetime.
[[vk::binding(5, 0)]] public StructuredBuffer<LightSource> light_sources;
// BRDF DFG LUT.
[[vk::binding(6, 0)]] public Texture2D brdf_dfg_lut;
[[vk::binding(7, 0)]] public SamplerState brdf_dfg_lut_sampler;
// The uniform-pick list over ACTIVE lights: `[emissive_count,
// directional_count]` header, then the active emissive slots, then the active
// directional slots, then the flux CDF (f32 bits) + total flux.
[[vk::binding(8, 0)]] public StructuredBuffer<uint> active_light_list;
[[vk::binding(9, 0)]] public StructuredBuffer<HairSegment> hair_segments;
[[vk::binding(10, 0)]] public StructuredBuffer<GpuHairInstance> hair_instances;
// A single record (WGSL binds the bare struct; element 0 here).
[[vk::binding(11, 0)]] public StructuredBuffer<HairSceneParams> hair_params;
// GPU transform table: 3 `float4` rows (mat3x4) per node, indexed `slot*3 + k`.
[[vk::binding(12, 0)]] public StructuredBuffer<float4> hair_world;
[[vk::binding(4, 1)]] public ConstantBuffer<SolariGeometryAddresses> geometry_addresses;
// GPU scene columns (set 2 = SOLARI_SCENE_COLUMNS_GROUP).
[[vk::binding(0, 2)]] public StructuredBuffer<Affine> transforms;
[[vk::binding(1, 2)]] public StructuredBuffer<Affine> previous_frame_transforms;
[[vk::binding(2, 2)]] public StructuredBuffer<uint> material_ids;
[[vk::binding(3, 2)]] public StructuredBuffer<DirectionalLight> directional_lights;
[[vk::binding(4, 2)]] public StructuredBuffer<InstanceClusterRange> instance_cluster_ranges;

// ClusterIDNV — the canonical cluster index into `clusters[]`, exactly as the
// closest-hit uses it. NOT GeometryIndex(): in the pipeline that's the raw
// per-CLAS geometry index (≈0), which would read the wrong cluster's data.
public uint cluster_id()
{
    return spirv_asm
    {
        OpCapability RayTracingClusterAccelerationStructureNV;
        OpExtension "SPV_NV_cluster_acceleration_structure";
        result: $$uint = OpLoad builtin(ClusterIDNV: uint)
    };
}

// Per-view floating-origin delta (`origin_now − origin_prev`), seeded by
// entries that consume `previous_frame_world_position`. The tess branches
// synthesize a STATIC previous position from the current one, which must land
// in the PREVIOUS frame's origin — current + this delta gets it there.
static float3 reproj_origin_delta = float3(0.0);

public void set_reproj_origin_delta(float3 delta)
{
    reproj_origin_delta = delta;
}

// ── Decode helpers ──────────────────────────────────────────────────────────

// WGSL `unpack2x16snorm`: two s16s, each clamp(v/32767, -1, 1).
public float2 unpack_snorm2x16(uint v)
{
    let iv = int2(int(v << 16) >> 16, int(v) >> 16);
    return clamp(float2(iv) / 32767.0, -1.0, 1.0);
}

// Octahedral decode, [-1,1] input (bevy_render::utils parity).
public float3 octahedral_decode_signed(float2 v)
{
    var n = float3(v.x, v.y, 1.0 - abs(v.x) - abs(v.y));
    let t = saturate(-n.z);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

// WGSL `pack2x16snorm`: round(clamp(e, -1, 1) * 32767) per component.
public uint pack_snorm2x16(float2 v)
{
    let c = clamp(v, -1.0, 1.0) * 32767.0;
    let x = uint(int(round(c.x))) & 0xffff;
    let y = uint(int(round(c.y))) & 0xffff;
    return x | (y << 16);
}

// WGSL `pack2x16float`: two f16s.
public uint pack_half2x16(float2 v)
{
    return f32tof16(v.x) | (f32tof16(v.y) << 16);
}

public float2 unpack_half2x16(uint v)
{
    return float2(f16tof32(v & 0xffff), f16tof32(v >> 16));
}

// Octahedral encode → [0,1] (bevy_render::utils parity; callers remap to
// [-1,1] before snorm-packing).
public float2 octahedral_encode(float3 v)
{
    let n = v / (abs(v.x) + abs(v.y) + abs(v.z));
    if (n.z >= 0.0)
    {
        return n.xy * 0.5 + 0.5;
    }
    let w = float2(
        (1.0 - abs(n.y)) * (n.x > 0.0 ? 1.0 : -1.0),
        (1.0 - abs(n.x)) * (n.y > 0.0 ? 1.0 : -1.0));
    return w * 0.5 + 0.5;
}

// Decode a `PackedVertex::tangent` u32 → (xyz dir, bitangent sign in w): 10
// bits per direction component (unorm of [-1,1]) + sign in bit 30. Renormalize
// since 10-bit quantization perturbs the length slightly.
float4 unpack_tangent(uint p)
{
    let x = float(p & 0x3ff) / 1023.0 * 2.0 - 1.0;
    let y = float((p >> 10) & 0x3ff) / 1023.0 * 2.0 - 1.0;
    let z = float((p >> 20) & 0x3ff) / 1023.0 * 2.0 - 1.0;
    let w = ((p >> 30) & 1) == 1 ? -1.0 : 1.0;
    return float4(normalize(float3(x, y, z)), w);
}

// ── BDA loads ───────────────────────────────────────────────────────────────

// Element-wise float2 load — the packed pools guarantee only 4-byte alignment.
float2 load_float2_bda(uint64_t addr)
{
    return float2(*(float*)(addr), *(float*)(addr + 4));
}

// Bindless material fetch: load the whole record by buffer-device-address
// (uniform across the warp after SER).
public Material load_material_bindless(uint material_id)
{
    let addr = geometry_addresses.materials
        + uint64_t(material_id) * uint64_t(geometry_addresses.material_stride);
    return *(Material*)(addr);
}

// `vertex_packed` is a 16-byte record per vertex (normal@0, tangent@4, uv@8).
public float2 load_packed_uv(uint vertex_index)
{
    return load_float2_bda(geometry_addresses.vertex_packed
        + uint64_t(vertex_index) * 16 + 8);
}

/// Load (decode) one fully-populated vertex at a global vertex index: position
/// from the SoA pool (stride 12 — the same buffer the CLAS build reads),
/// shading attrs from the interleaved 16-byte packed pool.
Vertex load_cluster_vertex(uint vertex_index)
{
    var v = load_cluster_vertex_attrs(vertex_index);
    v.position = *(float3*)(geometry_addresses.vertex_positions
        + uint64_t(vertex_index) * 12);
    return v;
}

/// Shading attrs only (normal / tangent / uv) from the packed pool: the
/// position-fetch path reads the three hit-triangle positions from the CLAS
/// builtin instead. `position` is left zero; the caller fills it.
Vertex load_cluster_vertex_attrs(uint vertex_index)
{
    let base = geometry_addresses.vertex_packed + uint64_t(vertex_index) * 16;
    Vertex v;
    v.position = float3(0.0);
    v.normal = octahedral_decode_signed(unpack_snorm2x16(*(uint*)(base)));
    v.tangent = unpack_tangent(*(uint*)(base + 4));
    v.uv = load_float2_bda(base + 8);
    return v;
}

/// Override a vertex's normal/tangent with the deformed values at deform-pool
/// slot `di` (positions come from the hit; uv is deform-invariant).
Vertex apply_deform_attrs(Vertex v, uint di)
{
    var out = v;
    out.normal = octahedral_decode_signed(unpack_snorm2x16(
        *(uint*)(geometry_addresses.deform_normals + uint64_t(di) * 4)));
    out.tangent = *(float4*)(geometry_addresses.deform_tangents + uint64_t(di) * 16);
    return out;
}

// ── Textures ────────────────────────────────────────────────────────────────

// A `partial_lod` at or below this means "sample mip 0" — used by callers that
// don't track a ray cone.
public static const float TEXTURE_LOD_MIP0 = -1000.0;

public float3 sample_texture(uint id, float2 uv)
{
    return textures[NonUniformResourceIndex(id)]
        .SampleLevel(samplers[NonUniformResourceIndex(id)], uv, 0.0).rgb;
}

// As `sample_texture`, but at an explicit ray-cone LOD (the texture's
// `0.5·log2(w·h)` is already folded in by the caller via `Material.texel_lod_bias`).
// `lod <= TEXTURE_LOD_MIP0` ⇒ mip 0.
public float3 sample_texture_lod(uint id, float2 uv, float lod)
{
    let level = max(lod, 0.0); // sentinel & cone-magnification both clamp to mip 0
    return textures[NonUniformResourceIndex(id)]
        .SampleLevel(samplers[NonUniformResourceIndex(id)], uv, level).rgb;
}

public ResolvedMaterial resolve_material_lod(Material material, float2 uv, float partial_lod)
{
    ResolvedMaterial m;
    // Fold in this material's texture-size term (mip-0 sentinel stays < 0).
    let lod = partial_lod + material.texel_lod_bias;

    m.base_color = material.base_color;
    if (material.base_color_texture_id != TEXTURE_MAP_NONE)
    {
        m.base_color *= sample_texture_lod(material.base_color_texture_id, uv, lod);
    }

    m.emissive = material.emissive;
    if (material.emissive_texture_id != TEXTURE_MAP_NONE)
    {
        m.emissive *= sample_texture_lod(material.emissive_texture_id, uv, lod);
    }

    m.reflectance = material.reflectance;

    m.perceptual_roughness = material.perceptual_roughness;
    m.metallic = material.metallic;
    if (material.metallic_roughness_texture_id != TEXTURE_MAP_NONE)
    {
        let metallic_roughness = sample_texture_lod(material.metallic_roughness_texture_id, uv, lod);
        m.perceptual_roughness *= metallic_roughness.g;
        m.metallic *= metallic_roughness.b;
    }

    m.roughness = m.perceptual_roughness * m.perceptual_roughness;

    m.specular_transmission = material.specular_transmission;
    m.ior = material.ior;
    m.dispersion = material.dispersion;
    m.extinction = material.extinction;
    m.nested_priority = material.nested_priority;

    return m;
}

// ── Affine helpers ──────────────────────────────────────────────────────────

// k = dot(linear row k, p) + translation k.
public float3 affine_transform_point(Affine m, float3 p)
{
    return float3(
        dot(m.r0.xyz, p) + m.r0.w,
        dot(m.r1.xyz, p) + m.r1.w,
        dot(m.r2.xyz, p) + m.r2.w);
}

// Apply the linear part only (for directions).
public float3 affine_transform_direction(Affine m, float3 v)
{
    return float3(dot(m.r0.xyz, v), dot(m.r1.xyz, v), dot(m.r2.xyz, v));
}

// Transform a NORMAL by `m`'s linear part — the inverse-transpose (cofactor
// matrix), NOT `affine_transform_direction`. Non-uniform / sheared transforms
// skew normals unless the cofactor is used. Caller normalizes.
public float3 affine_transform_normal(Affine m, float3 n)
{
    let c0 = float3(m.r0.x, m.r1.x, m.r2.x);
    let c1 = float3(m.r0.y, m.r1.y, m.r2.y);
    let c2 = float3(m.r0.z, m.r1.z, m.r2.z);
    return cross(c1, c2) * n.x + cross(c2, c0) * n.y + cross(c0, c1) * n.z;
}

// ── Ray helpers ─────────────────────────────────────────────────────────────

// Self-intersection-free ray origin for continuation rays (Wächter & Binder,
// Ray Tracing Gems ch. 6): offset the hit point along the geometric normal by
// a few ULPs of its own float representation. Trace from the result with
// `t_min = 0`.
public float3 offset_ray_origin(float3 p, float3 geometric_normal)
{
    let int_offset = int3(geometric_normal * 256.0);
    let p_int = float3(
        asfloat(asint(p.x) + (p.x < 0.0 ? -int_offset.x : int_offset.x)),
        asfloat(asint(p.y) + (p.y < 0.0 ? -int_offset.y : int_offset.y)),
        asfloat(asint(p.z) + (p.z < 0.0 ? -int_offset.z : int_offset.z)));
    // Near zero a fixed float offset replaces the integer bump (the ULP size
    // collapses as the exponent does).
    return float3(
        abs(p.x) < (1.0 / 32.0) ? p.x + geometric_normal.x * (1.0 / 65536.0) : p_int.x,
        abs(p.y) < (1.0 / 32.0) ? p.y + geometric_normal.y * (1.0 / 65536.0) : p_int.y,
        abs(p.z) < (1.0 / 32.0) ? p.z + geometric_normal.z * (1.0 / 65536.0) : p_int.z);
}

// ── Hair / LSS hit resolve ──────────────────────────────────────────────────

/// Transform a local point by a transform-table node's world matrix (3 rows).
float3 hair_world_point(uint transform_slot, float3 p)
{
    let s = transform_slot * 3;
    let r0 = hair_world[s];
    let r1 = hair_world[s + 1];
    let r2 = hair_world[s + 2];
    return float3(
        dot(r0.xyz, p) + r0.w,
        dot(r1.xyz, p) + r1.w,
        dot(r2.xyz, p) + r2.w);
}

/// Resolved hair hit: world position, fiber tangent, and the instance record
/// (color + fiber params). The fiber axis comes from the hit segment's two
/// endpoints, the surface point from the ray distance.
public struct ResolvedHairHit
{
    public float3 world_position;
    public float3 tangent;
    public float3 sigma_a;
    public float beta_m;
    public float beta_n;
    public float alpha;
    public float ior;
}

public ResolvedHairHit resolve_hair_hit(
    uint instance_index,
    uint primitive_index,
    float3 world_position)
{
    let h = hair_instances[instance_index - hair_params[0].base];
    let seg = hair_segments[h.segment_base + primitive_index];
    let wp0 = hair_world_point(h.transform_slot, seg.p0.xyz);
    let wp1 = hair_world_point(h.transform_slot, seg.p1.xyz);
    ResolvedHairHit hit;
    hit.world_position = world_position;
    hit.tangent = normalize(wp1 - wp0);
    hit.sigma_a = h.sigma_a;
    hit.beta_m = h.beta_m;
    hit.beta_n = h.beta_n;
    hit.alpha = h.alpha;
    hit.ior = h.ior;
    return hit;
}

/// The opaque-surface material slot of an LSS instance (`HAIR_MATERIAL_NONE`
/// for fiber hair).
public uint lss_material_id(uint instance_index)
{
    return hair_instances[instance_index - hair_params[0].base].material_id;
}

/// An opaque hit on a linear-swept-sphere (round-cone) primitive.
public struct LssSurface
{
    public float3 world_position;
    public float3 world_normal;
    public uint material_id;
}

/// Resolve an opaque LSS (branch) hit: reconstruct the hit segment's world
/// endpoints + radii, then build the round-cone surface normal — the outward
/// radial direction from the swept-sphere axis, tilted along the axis by the
/// taper slope `(r0 - r1)/L` so the cone's slant is shaded.
public LssSurface resolve_lss_surface(
    uint instance_index,
    uint primitive_index,
    float3 world_position)
{
    let h = hair_instances[instance_index - hair_params[0].base];
    let seg = hair_segments[h.segment_base + primitive_index];
    let wp0 = hair_world_point(h.transform_slot, seg.p0.xyz);
    let wp1 = hair_world_point(h.transform_slot, seg.p1.xyz);
    let axis = wp1 - wp0;
    let len = max(length(axis), 1.0e-6);
    let dir = axis / len;
    // Closest axis point (clamped to the segment), then the outward radial.
    let h_along = clamp(dot(world_position - wp0, dir), 0.0, len);
    let radial = world_position - (wp0 + dir * h_along);
    let rl = length(radial);
    let n_radial = rl > 1.0e-6 ? radial / rl : float3(0.0, 1.0, 0.0);
    let slope = (seg.p0.w - seg.p1.w) / len;
    LssSurface s;
    s.world_position = world_position;
    s.world_normal = normalize(n_radial + dir * slope);
    s.material_id = h.material_id;
    return s;
}

// ── Alpha cutout (any-hit) ──────────────────────────────────────────────────

// Mask-test core for `ahit_alpha`: base-color texture alpha at the hit UV
// against the material's cutoff. `true` = keep the hit (solid texel), `false`
// = cut it. No texture = solid. Field-wise material loads: the any-hit only
// needs 8 bytes of the record.
public bool alpha_passes(uint material_id, uint cluster_index, uint primitive_index, float2 bary)
{
    let material_addr = geometry_addresses.materials
        + uint64_t(material_id) * uint64_t(geometry_addresses.material_stride);
    let alpha_mask = *(float*)(material_addr + 76); // Material.alpha_mask
    if (alpha_mask < 0.0)
    {
        return true; // opaque material on a non-opaque instance (stale flag)
    }
    let texture_id = *(uint*)(material_addr + 4); // Material.base_color_texture_id
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

// ── The resolve ─────────────────────────────────────────────────────────────

/// Shared resolve body: given the hit triangle's three fully-populated
/// vertices, the instance transform, and the cluster's triangle count, produce
/// the full shading hit.
ResolvedRayHitFull resolve_triangle_data_core(
    uint instance_id,
    uint material_id,
    Affine transform,
    uint triangle_count,
    float3 barycentrics,
    float cone_width,
    Vertex vertices[3])
{
    let material = load_material_bindless(material_id);
    let previous_frame_transform = previous_frame_transforms[instance_id];

    let wv0 = affine_transform_point(transform, vertices[0].position);
    let wv1 = affine_transform_point(transform, vertices[1].position);
    let wv2 = affine_transform_point(transform, vertices[2].position);
    let world_position = wv0 * barycentrics.x + wv1 * barycentrics.y + wv2 * barycentrics.z;

    let pv0 = affine_transform_point(previous_frame_transform, vertices[0].position);
    let pv1 = affine_transform_point(previous_frame_transform, vertices[1].position);
    let pv2 = affine_transform_point(previous_frame_transform, vertices[2].position);
    let previous_frame_world_position =
        pv0 * barycentrics.x + pv1 * barycentrics.y + pv2 * barycentrics.z;

    let uv = vertices[0].uv * barycentrics.x + vertices[1].uv * barycentrics.y
        + vertices[2].uv * barycentrics.z;

    let local_tangent = vertices[0].tangent.xyz * barycentrics.x
        + vertices[1].tangent.xyz * barycentrics.y
        + vertices[2].tangent.xyz * barycentrics.z;
    let world_tangent = float4(
        normalize(affine_transform_direction(transform, local_tangent)),
        vertices[0].tangent.w);

    let local_normal = vertices[0].normal * barycentrics.x
        + vertices[1].normal * barycentrics.y
        + vertices[2].normal * barycentrics.z;
    var world_normal = normalize(affine_transform_normal(transform, local_normal));

    let triangle_edge0 = wv0 - wv1;
    let triangle_edge1 = wv0 - wv2;
    let triangle_cross = cross(triangle_edge0, triangle_edge1);
    let triangle_area = length(triangle_cross) / 2.0;

    // True (planar) triangle normal, outward by the CCW winding; closest-hit
    // shaders reorient it to the hit side via the hit kind.
    let geometric_world_normal = normalize(triangle_cross);

    // Ray-cone texture LOD (Akenine-Möller et al., Ray Tracing Gems ch.20) —
    // the texture-size-INDEPENDENT term. A negative `cone_width` keeps the
    // mip-0 sentinel.
    var partial_lod = TEXTURE_LOD_MIP0;
    if (cone_width >= 0.0)
    {
        let uv_edge0 = vertices[1].uv - vertices[0].uv;
        let uv_edge1 = vertices[2].uv - vertices[0].uv;
        let uv_area = 0.5 * abs(uv_edge0.x * uv_edge1.y - uv_edge0.y * uv_edge1.x);
        partial_lod = 0.5 * log2(max(uv_area, 1e-12) / max(triangle_area, 1e-8))
            + log2(max(cone_width, 1e-6));
    }

    if (material.normal_map_texture_id != TEXTURE_MAP_NONE)
    {
        // TBN per mikktspace (bevy_solari::pbr parity).
        let N = world_normal;
        let T = world_tangent.xyz;
        let B = world_tangent.w * cross(N, T);
        // Texels store the tangent-space normal remapped to [0,1] — decode
        // before the TBN transform.
        let Nt = sample_texture_lod(material.normal_map_texture_id, uv,
                     partial_lod + material.texel_lod_bias) * 2.0 - 1.0;
        world_normal = normalize(Nt.x * T + Nt.y * B + Nt.z * N);
    }

    let resolved_material = resolve_material_lod(material, uv, partial_lod);

    ResolvedRayHitFull hit;
    hit.world_position = world_position;
    hit.previous_frame_world_position = previous_frame_world_position;
    hit.world_normal = world_normal;
    hit.geometric_world_normal = geometric_world_normal;
    hit.world_tangent = world_tangent;
    hit.uv = uv;
    hit.triangle_area = triangle_area;
    hit.triangle_count = triangle_count;
    hit.material = resolved_material;
    hit.material_id = material_id;
    return hit;
}

/// Position-fetch resolve for the RT closest-hit: the hit triangle's three
/// object-space vertex positions come from the CLAS
/// (`VK_KHR_ray_tracing_position_fetch`), so only normal / tangent / uv are
/// read from the packed pool. Mip-0 (no ray cone).
public ResolvedRayHitFull resolve_triangle_data_full_mat_fetch(
    uint instance_id,
    uint material_id,
    Affine transform,
    uint cluster_global_id,
    uint triangle_id,
    float3 barycentrics,
    float3 object_positions[3])
{
    // Tessellation: its CLAS is instantiated with a ClusterIDNV above the real
    // cluster pool, so it has no `clusters[]` entry and its displaced
    // micro-vertices have no vertex-pool attributes. Two shading paths:
    //   - SMOOTH (`tess_clusters != 0`): recover the three micro-vertex
    //     smooth, displacement-aware normals from the per-CLAS metadata table
    //     + per-instance normal buffer.
    //   - FACET (table unwired): geometric normal from the fetched positions,
    //     fixed UV.
    uint cluster_count;
    uint cluster_stride;
    clusters.GetDimensions(cluster_count, cluster_stride);
    if (cluster_global_id >= cluster_count)
    {
        Vertex sv0;
        Vertex sv1;
        Vertex sv2;
        if (geometry_addresses.tess_clusters != 0)
        {
            // Per-CLAS metadata (16 B): normal-buffer address (lo/hi u32) +
            // primitive base. Loaded field-wise.
            let tc_addr = geometry_addresses.tess_clusters
                + uint64_t(cluster_global_id - TESS_CLUSTER_ID_BASE) * 16;
            let normals_lo = *(uint*)(tc_addr + 0);
            let normals_hi = *(uint*)(tc_addr + 4);
            let primitive_base = *(uint*)(tc_addr + 8);
            let attrs = uint64_t(normals_lo) | (uint64_t(normals_hi) << 32);
            // Denormalized per-vertex attrs: 3 × (packed normal @0 + UV @4) =
            // 36 B / tri — 4-byte alignment only.
            let base = attrs + uint64_t(primitive_base + triangle_id) * 36;
            let a0 = base + 0;
            let a1 = base + 12;
            let a2 = base + 24;
            let n0 = octahedral_decode_signed(unpack_snorm2x16(*(uint*)(a0)));
            let n1 = octahedral_decode_signed(unpack_snorm2x16(*(uint*)(a1)));
            let n2 = octahedral_decode_signed(unpack_snorm2x16(*(uint*)(a2)));
            let uv0 = load_float2_bda(a0 + 4);
            let uv1 = load_float2_bda(a1 + 4);
            let uv2 = load_float2_bda(a2 + 4);
            // Real UV-gradient tangent (needed for normal maps) derived from
            // the micro-triangle's OBJECT-space positions + UVs — the tess
            // attrs carry no tangent. Gram-Schmidt against n0 + handedness
            // sign; arbitrary tangent on degenerate UV.
            let e1 = object_positions[1] - object_positions[0];
            let e2 = object_positions[2] - object_positions[0];
            let duv1 = uv1 - uv0;
            let duv2 = uv2 - uv0;
            let det = duv1.x * duv2.y - duv2.x * duv1.y;
            float4 tangent;
            if (abs(det) > 1e-12)
            {
                let r = 1.0 / det;
                let t = (e1 * duv2.y - e2 * duv1.y) * r;
                let bt = (e2 * duv1.x - e1 * duv2.x) * r;
                let tan = normalize(t - n0 * dot(n0, t));
                let w = dot(cross(n0, tan), bt) > 0.0 ? 1.0 : -1.0;
                tangent = float4(tan, w);
            }
            else
            {
                let up = abs(n0.y) > 0.99 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
                tangent = float4(normalize(cross(up, n0)), 1.0);
            }
            sv0.position = object_positions[0]; sv0.normal = n0; sv0.uv = uv0; sv0.tangent = tangent;
            sv1.position = object_positions[1]; sv1.normal = n1; sv1.uv = uv1; sv1.tangent = tangent;
            sv2.position = object_positions[2]; sv2.normal = n2; sv2.uv = uv2; sv2.tangent = tangent;
        }
        else
        {
            let cr = cross(
                object_positions[1] - object_positions[0],
                object_positions[2] - object_positions[0]);
            let cl = length(cr);
            let flat_normal = cl > 1e-12 ? cr / cl : float3(0.0, 0.0, 1.0);
            // Arbitrary tangent perpendicular to the normal (only consulted if
            // the material carries a normal map).
            let up = abs(flat_normal.y) > 0.99 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
            let tangent = float4(normalize(cross(up, flat_normal)), 1.0);
            sv0.position = object_positions[0]; sv0.normal = flat_normal; sv0.uv = float2(0.5); sv0.tangent = tangent;
            sv1.position = object_positions[1]; sv1.normal = flat_normal; sv1.uv = float2(0.5); sv1.tangent = tangent;
            sv2.position = object_positions[2]; sv2.normal = flat_normal; sv2.uv = float2(0.5); sv2.tangent = tangent;
        }
        Vertex tess_vertices[3] = { sv0, sv1, sv2 };
        var hit = resolve_triangle_data_core(
            instance_id, material_id, transform, 1, barycentrics, -1.0, tess_vertices);
        // Tess motion vectors: the per-instance previous-frame transform is
        // unreliable for tessellated surfaces (the base mesh is hidden, its
        // slot stale or unresolved). Treat the surface as STATIC: previous
        // world position == current (motion = camera-only), exact for static
        // displaced geometry. MOVING tessellated surfaces would ghost and need
        // the real per-instance previous transform — TODO.
        // + delta: previous positions are in the PREVIOUS frame's origin.
        hit.previous_frame_world_position = hit.world_position + reproj_origin_delta;
        hit.material.perceptual_roughness = max(hit.material.perceptual_roughness, TESS_ROUGHNESS_FLOOR);
        hit.material.roughness = hit.material.perceptual_roughness * hit.material.perceptual_roughness;
        return hit;
    }

    let cluster = clusters[cluster_global_id];
    // `cluster_indices` stores per-cluster LOCAL vertex indices; the cluster's
    // (globally-rebased) `vertex_offset` reconstructs the global pool slot.
    let idx_base = cluster.index_offset + triangle_id * 3;
    let gvi0 = cluster.vertex_offset + cluster_indices[idx_base + 0];
    let gvi1 = cluster.vertex_offset + cluster_indices[idx_base + 1];
    let gvi2 = cluster.vertex_offset + cluster_indices[idx_base + 2];
    // Attrs-only fetch (skips the position head); the builtin supplies position.
    var v0 = load_cluster_vertex_attrs(gvi0);
    var v1 = load_cluster_vertex_attrs(gvi1);
    var v2 = load_cluster_vertex_attrs(gvi2);
    v0.position = object_positions[0];
    v1.position = object_positions[1];
    v2.position = object_positions[2];
    // Animated instance: positions already come from the deformed hit
    // triangle, but the pooled attrs are rest-pose — swap normal/tangent for
    // the deformed values.
    if (geometry_addresses.animated_table != 0)
    {
        let a = geometry_addresses.animated_table + uint64_t(instance_id) * 16;
        if (*(uint*)(a) == 1)
        {
            let pool_base = *(uint*)(a + 4);
            let mvb = *(uint*)(a + 8);
            v0 = apply_deform_attrs(v0, pool_base + (gvi0 - mvb));
            v1 = apply_deform_attrs(v1, pool_base + (gvi1 - mvb));
            v2 = apply_deform_attrs(v2, pool_base + (gvi2 - mvb));
        }
    }
    Vertex vertices[3] = { v0, v1, v2 };
    return resolve_triangle_data_core(
        instance_id, material_id, transform, cluster.triangle_count, barycentrics, -1.0, vertices);
}

/// Full vertex-pool resolve (positions from the SoA pool, material via the
/// `material_ids` / `transforms` columns, mip-0) — light sampling resolves the
/// picked emissive triangle through this, matching the WGSL
/// `resolve_triangle_data_full`.
public ResolvedRayHitFull resolve_triangle_data_full(
    uint instance_id,
    uint cluster_global_id,
    uint triangle_id,
    float3 barycentrics)
{
    let cluster = clusters[cluster_global_id];
    let idx_base = cluster.index_offset + triangle_id * 3;
    Vertex vertices[3] = {
        load_cluster_vertex(cluster.vertex_offset + cluster_indices[idx_base + 0]),
        load_cluster_vertex(cluster.vertex_offset + cluster_indices[idx_base + 1]),
        load_cluster_vertex(cluster.vertex_offset + cluster_indices[idx_base + 2]),
    };
    return resolve_triangle_data_core(
        instance_id,
        material_ids[instance_id],
        transforms[instance_id],
        cluster.triangle_count,
        barycentrics,
        -1.0,
        vertices);
}
