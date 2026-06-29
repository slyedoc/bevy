enable wgpu_ray_query;

#define_import_path bevy_solari::scene_bindings

#import bevy_solari::pbr::perceptualRoughnessToRoughness
#import bevy_solari::pbr::calculate_tbn_mikktspace
#import bevy_solari::atmosphere::atmosphere_mie_phase
#import bevy_render::utils::octahedral_decode_signed

// Cluster pool layout — mirrors `bevy_solari::cluster::asset::Cluster`.
// `vertex_offset` / `index_offset` are GLOBAL slots (rebased to the
// shared cluster pool by `ClusterMeshManager` at upload time).
struct Cluster {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    triangle_count: u32,
    bounds_sphere: vec4<f32>,
    local_material_id: u32,
    lod_level: u32,
}

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    tangent: vec4<f32>,
}

struct Material {
    normal_map_texture_id: u32,
    base_color_texture_id: u32,
    emissive_texture_id: u32,
    metallic_roughness_texture_id: u32,

    base_color: vec3<f32>,
    perceptual_roughness: f32,
    emissive: vec3<f32>,
    metallic: f32,
    // Beer-Lambert extinction per channel (1/world-unit) inside the volume.
    extinction: vec3<f32>,
    reflectance: f32,
    ior: f32,
    specular_transmission: f32,
    nested_priority: u32,
    // Alpha-mask cutoff; negative = opaque (no alpha test during traversal).
    alpha_mask: f32,
    // Chromatic dispersion (20/Abbe, `KHR_materials_dispersion`); 0 = none.
    dispersion: f32,
    // `0.5·log2(w·h)` of the base-color texture — the texture-size term of the
    // ray-cone LOD, baked CPU-side so no per-hit `textureDimensions` query.
    texel_lod_bias: f32,
    // Displacement (height) map; `TEXTURE_MAP_NONE` if absent. Sampled per
    // generated micro-vertex during tessellation: offset along the normal by
    // `height * displacement_scale + displacement_bias` (world units).
    displacement_texture_id: u32,
    displacement_scale: f32,
    displacement_bias: f32,
    // Opaque per-material data for a custom closest-hit (StandardSolariMaterial::chit_data).
    chit_data: vec4<u32>,
}

const TEXTURE_MAP_NONE = 0xFFFFFFFFu;

const MIRROR_ROUGHNESS_THRESHOLD = 0.001f;

struct LightSource {
    kind: u32, // 1 bit for kind, 31 bits for extra data
    id: u32,
}

const LIGHT_SOURCE_KIND_EMISSIVE_MESH = 0u;
const LIGHT_SOURCE_KIND_DIRECTIONAL = 1u;
// A freed slot in the slot-indexed `light_sources` table: an emissive light
// with zero triangles (`kind == 0`), which nothing can sample. A reservoir
// whose stored slot resolves to this holds a dead light — reject it.
const LIGHT_SOURCE_KIND_NONE = 0u;

struct DirectionalLight {
    direction_to_light: vec3<f32>,
    cos_theta_max: f32,
    luminance: vec3<f32>,
    inverse_pdf: f32,
}

// Cluster mesh pool — shared across every `RaytracingMesh3d` instance in the
// scene. Indices stored here are GLOBAL pool slots. Vertex attributes + materials
// are reached bindlessly by buffer-device-address (`geometry_addresses`); only the
// cluster table + index pool are bound here.
@group(0) @binding(0)  var<storage> cluster_indices: array<u32>;
@group(0) @binding(1)  var<storage> clusters: array<Cluster>;

// Per-instance ray-shading state — written by
// `prepare_raytracing_scene_bindings` from `InstanceManager`.
// Slot-indexed world transforms as `mat3x4` affines (column k = 4x4
// row k), GPU-scattered by `cluster::gpu_instances`. Apply via the
// `affine_*` helpers below.
// `transforms` / `previous_frame_transforms` / `material_ids` are GPU columns,
// bound from the shared scene-columns group (`ecs_gpu::SceneColumns`) — the group
// index is supplied per-pipeline via the `SOLARI_SCENE_COLUMNS_GROUP` shader-def.
@group(#{SOLARI_SCENE_COLUMNS_GROUP}) @binding(0) var<storage> transforms: array<mat3x4<f32>>;
@group(#{SOLARI_SCENE_COLUMNS_GROUP}) @binding(1) var<storage> previous_frame_transforms: array<mat3x4<f32>>;
@group(#{SOLARI_SCENE_COLUMNS_GROUP}) @binding(2) var<storage> material_ids: array<u32>;

// Textures (materials are loaded bindlessly via `geometry_addresses`).
@group(0) @binding(2) var textures: binding_array<texture_2d<f32>>;
@group(0) @binding(3) var samplers: binding_array<sampler>;

// Ray-tracing acceleration structure + lighting.
@group(0) @binding(4) var tlas: acceleration_structure;
// STABLE-SLOT indexed: a light keeps its index for its lifetime (freed slots
// hole out as `LIGHT_SOURCE_KIND_NONE` and are reused), so a slot stored in a
// reservoir / light tile stays a valid identity across light add/remove.
@group(0) @binding(5) var<storage> light_sources: array<LightSource>;
// The uniform-pick list over ACTIVE lights: `[emissive_count,
// directional_count]` header, then the active emissive slots, then the active
// directional slots (strata contiguous for the stratified pick).
@group(0) @binding(8) var<storage> active_light_list: array<u32>;
// `directional_lights` is also a scene column (the lights table owns it).
@group(#{SOLARI_SCENE_COLUMNS_GROUP}) @binding(3) var<storage> directional_lights: array<DirectionalLight>;

// BRDF DFG LUT.
@group(0) @binding(6) var brdf_dfg_lut: texture_2d<f32>;
@group(0) @binding(7) var brdf_dfg_lut_sampler: sampler;


// ── Hair (linear swept spheres) ──
// Per-instance record (mirrors `GpuHairInstance`, Rust). The path tracer reads
// it when a ray hits a hair instance (PTLAS index in `[base, base+count)`). The
// world transform isn't stored — it lives in the GPU transform table (`hair_world`)
// at `transform_slot`, like directional lights.
struct GpuHairInstance {
    sigma_a: vec3<f32>,             // absorption from melanin (+ dye), CPU-derived
    transform_slot: u32,            // index into `hair_world` (×3 rows)
    segment_base: u32,              // base into `hair_segments`
    beta_m: f32,
    beta_n: f32,
    alpha: f32,
    ior: f32,
    blas_address_lo: u32,
    blas_address_hi: u32,
    mask: u32,
    // Opaque-surface material slot for `SolariBranches` (bark/wood/etc), or
    // `HAIR_MATERIAL_NONE` for fiber hair shaded by the Chiang BSDF.
    material_id: u32,
}

// `GpuHairInstance.material_id` sentinel: shade as fiber hair, not an opaque surface.
const HAIR_MATERIAL_NONE = 0xFFFFFFFFu;
// Per swept segment: the two capped-cylinder endpoints + radii, local space.
struct HairSegment {
    p0: vec4<f32>, // xyz position, w radius
    p1: vec4<f32>,
}
struct HairSceneParams {
    base: u32,  // PTLAS instance_index of hair instance 0
    count: u32,
    pad0: u32,
    pad1: u32,
}
@group(0) @binding(9) var<storage> hair_segments: array<HairSegment>;
@group(0) @binding(10) var<storage> hair_instances: array<GpuHairInstance>;
@group(0) @binding(11) var<storage> hair_params: HairSceneParams;
// GPU transform table: 3 `vec4` rows (mat3x4) per node, indexed `slot*3 + k`.
@group(0) @binding(12) var<storage> hair_world: array<vec4<f32>>;

/// A hit's PTLAS `instance_index` lands in the hair range.
fn is_hair_instance(instance_index: u32) -> bool {
    return hair_params.count > 0u
        && instance_index >= hair_params.base
        && instance_index < hair_params.base + hair_params.count;
}

/// Transform a local point by a transform-table node's world matrix (3 rows).
fn hair_world_point(transform_slot: u32, p: vec3<f32>) -> vec3<f32> {
    let s = transform_slot * 3u;
    let r0 = hair_world[s];
    let r1 = hair_world[s + 1u];
    let r2 = hair_world[s + 2u];
    return vec3<f32>(
        dot(r0.xyz, p) + r0.w,
        dot(r1.xyz, p) + r1.w,
        dot(r2.xyz, p) + r2.w,
    );
}

/// Resolved hair hit: world position, fiber tangent, and the instance record
/// (color + fiber params). Reconstructed from the hit segment's two endpoints —
/// the path tracer's ray query doesn't expose the NV LSS curve getters, so the
/// fiber axis comes from the segment, the surface point from the ray distance.
struct ResolvedHairHit {
    world_position: vec3<f32>,
    tangent: vec3<f32>,
    sigma_a: vec3<f32>,
    beta_m: f32,
    beta_n: f32,
    alpha: f32,
    ior: f32,
}

fn resolve_hair_hit(
    instance_index: u32,
    primitive_index: u32,
    world_position: vec3<f32>,
) -> ResolvedHairHit {
    let h = hair_instances[instance_index - hair_params.base];
    let seg = hair_segments[h.segment_base + primitive_index];
    let wp0 = hair_world_point(h.transform_slot, seg.p0.xyz);
    let wp1 = hair_world_point(h.transform_slot, seg.p1.xyz);
    var hit: ResolvedHairHit;
    hit.world_position = world_position;
    hit.tangent = normalize(wp1 - wp0);
    hit.sigma_a = h.sigma_a;
    hit.beta_m = h.beta_m;
    hit.beta_n = h.beta_n;
    hit.alpha = h.alpha;
    hit.ior = h.ior;
    return hit;
}

/// The opaque-surface material slot of an LSS instance (`HAIR_MATERIAL_NONE` for
/// fiber hair). Lets the hair closest-hit route opaque `SolariBranches` to BRDF
/// shading without the per-segment geometry work of `resolve_lss_surface`.
fn lss_material_id(instance_index: u32) -> u32 {
    return hair_instances[instance_index - hair_params.base].material_id;
}

/// An opaque hit on a linear-swept-sphere (round-cone) primitive: the surface
/// point, the round-cone surface normal (taper-tilted radial), and the material.
struct LssSurface {
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    material_id: u32,
}

/// Resolve an opaque LSS (branch) hit: reconstruct the hit segment's world
/// endpoints + radii, then build the round-cone surface normal. The normal is the
/// outward radial direction from the swept-sphere axis, tilted along the axis by
/// the taper slope `(r0 - r1)/L` so the cone's slant is shaded (not a pure
/// cylinder). Radii are object-space; branch transforms are ~rigid, so the slope
/// ratio is taken directly.
fn resolve_lss_surface(
    instance_index: u32,
    primitive_index: u32,
    world_position: vec3<f32>,
) -> LssSurface {
    let h = hair_instances[instance_index - hair_params.base];
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
    let n_radial = select(vec3<f32>(0.0, 1.0, 0.0), radial / rl, rl > 1.0e-6);
    let slope = (seg.p0.w - seg.p1.w) / len;
    var s: LssSurface;
    s.world_position = world_position;
    s.world_normal = normalize(n_radial + dir * slope);
    s.material_id = h.material_id;
    return s;
}

// Per-instance LOD inputs, slot-indexed. This is the SAME buffer the
// cluster path binds as `cluster_instance_lod_inputs` (16 B stride:
// cluster_base, cluster_count, group_base, root_group) — the RT path
// only needs `(cluster_base, cluster_count)` but MUST match the full
// stride or every slot past 0 reads misaligned. Used by emissive-mesh
// light sampling to walk this instance's clusters, and by the
// geometry-check debug view.
struct InstanceClusterRange {
    cluster_base: u32,
    cluster_count: u32,
    group_base: u32,
    root_group: u32,
}
@group(#{SOLARI_SCENE_COLUMNS_GROUP}) @binding(4) var<storage> instance_cluster_ranges: array<InstanceClusterRange>;


// Fog volumes (`bindings::fog_volume`, the `SolariFogVolumes` gpu_table):
// bounded participating media the aerial-perspective marches sample on top of
// the global height fog. The unit shape (box half-extents 1, or sphere radius
// 1) lives in entity-local space — the transform places/scales/rotates it.
// All-zero entries (tombstones, grown-buffer tail) are inert: zero extinction
// is skipped before the transform is read.
struct FogVolume {
    // World → entity-local affine (row-packed, column k = the 4x4's row k).
    local_from_world: mat3x4<f32>,
    // xyz = scattering σ_s (1/world unit), w = extinction σ_t.
    scattering: vec4<f32>,
    // x = HG phase g, y = edge softness (0..1), z = 1 for sphere / 0 for box.
    params: vec4<f32>,
}
@group(#{SOLARI_SCENE_COLUMNS_GROUP}) @binding(7) var<storage> fog_volumes: array<FogVolume>;

// Ray portals (`bindings::portal`, the `SolariPortals` gpu_table, indexed by
// portal slot): the instance-slot pairing `portal_redirect` matches a hit
// against. The teleport map derives at hit time from the live instance
// transform column (`transforms`), so portals on GPU-propagated / moving
// parents stay exact. `valid = 0` is inert (tombstone / grown-buffer tail).
struct Portal {
    instance_slot: u32,
    target_slot: u32,
    valid: u32,
    _pad: u32,
}
@group(#{SOLARI_SCENE_COLUMNS_GROUP}) @binding(5) var<storage> portals: array<Portal>;

const RAY_T_MIN = 0.001f;
// Effectively infinite ray length (primary, shadow, GI). Finite, not `inf`: an `inf`
// tmax produces NaN in the slab test (`inf * 0`) and silently drops hits.
const RAY_T_MAX = 1.0e30f;

// Self-intersection-free ray origin for continuation rays (Wächter & Binder,
// "A Fast and Robust Method for Avoiding Self-Intersection", Ray Tracing
// Gems ch. 6). Offsets the hit point along the geometric normal by a few ULPs
// of its own float representation — exactly as much as precision requires, so
// it can neither re-hit the surface it left nor skip real geometry (no fixed
// world-space epsilon to outgrow a millimeter-scale wine glass or underflow a
// kilometer-scale city). Trace from the result with `t_min = 0`.
fn offset_ray_origin(p: vec3<f32>, geometric_normal: vec3<f32>) -> vec3<f32> {
    let int_offset = vec3<i32>(geometric_normal * 256.0);
    let p_int = vec3<f32>(
        bitcast<f32>(bitcast<i32>(p.x) + select(int_offset.x, -int_offset.x, p.x < 0.0)),
        bitcast<f32>(bitcast<i32>(p.y) + select(int_offset.y, -int_offset.y, p.y < 0.0)),
        bitcast<f32>(bitcast<i32>(p.z) + select(int_offset.z, -int_offset.z, p.z < 0.0)),
    );
    // Near zero a fixed float offset replaces the integer bump (the ULP size
    // collapses as the exponent does).
    let near_origin = abs(p) < vec3(1.0 / 32.0);
    return select(p_int, p + geometric_normal * (1.0 / 65536.0), near_origin);
}

const RAY_NO_CULL = 0xFFu;

// Per-view RT cull mask (the camera's `RenderLayers` → low 8 bits). `trace_ray`
// passes it as the ray `cullMask`, so the hardware skips any instance whose
// `mask` shares no bit — the RT analog of render layers, free during traversal.
// Per-view, so each compute entry seeds it via `set_view_cull_mask` from its
// `solari_view` uniform. Defaults to no-cull so a pass that forgets to seed it
// still renders everything (never a black screen) rather than nothing.
var<private> view_cull_mask: u32 = RAY_NO_CULL;

fn set_view_cull_mask(mask: u32) {
    view_cull_mask = mask;
}

fn trace_ray(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32) -> RayIntersection {
    let ray = RayDesc(ray_flag, view_cull_mask, ray_t_min, ray_t_max, ray_origin, ray_direction);
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, ray);
    // Opaque instances commit in hardware and never enter this loop; only
    // alpha-masked instances (PTLAS `FORCE_NO_OPAQUE`) surface candidates.
    // Confirming only mask-passing candidates makes cutouts (foliage,
    // fences) hold for primary, bounce, AND shadow rays alike.
    while rayQueryProceed(&rq) {
        let candidate = rayQueryGetCandidateIntersection(&rq);
        if candidate.kind == RAY_QUERY_INTERSECTION_TRIANGLE && alpha_test(candidate) {
            rayQueryConfirmIntersection(&rq);
        }
    }
    return rayQueryGetCommittedIntersection(&rq);
}

// Mask-test core, addressable from both the inline rayQuery candidate
// (`alpha_test`) and the RT-pipeline any-hit shader (`ahit_alpha`): base-color
// texture alpha at the hit UV against the material's cutoff. `true` = keep the hit
// (solid texel), `false` = cut it (a hole the ray passes through). No texture =
// solid (alpha 1). The base-color FACTOR's alpha is not applied (the binder stores
// rgb only) — glTF cutouts author the mask in the texture. `cluster_index` and
// `bary` (the triangle's u, v) come from the hit; w is reconstructed here.
fn alpha_passes(material_id: u32, cluster_index: u32, primitive_index: u32, bary: vec2<f32>) -> bool {
    let material = load_material_bindless(material_id);
    if material.alpha_mask < 0.0 {
        return true; // opaque material on a non-opaque instance (stale flag)
    }
    let texture_id = material.base_color_texture_id;
    if texture_id == TEXTURE_MAP_NONE {
        return material.alpha_mask <= 1.0;
    }
    let cluster = clusters[cluster_index];
    let idx_base = cluster.index_offset + primitive_index * 3u;
    let uv0 = load_packed_uv(cluster.vertex_offset + cluster_indices[idx_base + 0u]);
    let uv1 = load_packed_uv(cluster.vertex_offset + cluster_indices[idx_base + 1u]);
    let uv2 = load_packed_uv(cluster.vertex_offset + cluster_indices[idx_base + 2u]);
    let barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    let uv = mat3x2(uv0, uv1, uv2) * barycentrics;
    let alpha = textureSampleLevel(textures[texture_id], samplers[texture_id], uv, 0.0).a;
    return alpha >= material.alpha_mask;
}

// Mask test for an inline rayQuery candidate triangle.
fn alpha_test(hit: RayIntersection) -> bool {
    return alpha_passes(material_ids[hit.instance_index], hit.geometry_index, hit.primitive_index, hit.barycentrics);
}

// A `partial_lod` at or below this means "sample mip 0" — used by callers that
// don't track a ray cone (NEE visibility, ReStIR, DLSS resolve).
const TEXTURE_LOD_MIP0: f32 = -1000.0;

fn sample_texture(id: u32, uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(textures[id], samplers[id], uv, 0.0).rgb;
}

// As `sample_texture`, but at an explicit ray-cone LOD (the texture's
// `0.5·log2(w·h)` is already folded in by the caller via `Material.texel_lod_bias`,
// so this never queries `textureDimensions`). `lod <= TEXTURE_LOD_MIP0` ⇒ mip 0.
fn sample_texture_lod(id: u32, uv: vec2<f32>, lod: f32) -> vec3<f32> {
    let level = max(lod, 0.0); // sentinel & cone-magnification both clamp to mip 0
    return textureSampleLevel(textures[id], samplers[id], uv, level).rgb;
}

struct ResolvedMaterial {
    base_color: vec3<f32>,
    emissive: vec3<f32>,
    reflectance: f32,
    perceptual_roughness: f32,
    roughness: f32,
    metallic: f32,
    specular_transmission: f32,
    ior: f32,
    dispersion: f32,
    extinction: vec3<f32>,
    nested_priority: u32,
}

struct ResolvedRayHitFull {
    world_position: vec3<f32>,
    previous_frame_world_position: vec3<f32>,
    world_normal: vec3<f32>,
    geometric_world_normal: vec3<f32>,
    world_tangent: vec4<f32>,
    uv: vec2<f32>,
    triangle_area: f32,
    triangle_count: u32,
    material: ResolvedMaterial,
    // Stable material slot — the identity nested-dielectric stacks pop on.
    material_id: u32,
}

fn resolve_material_lod(material: Material, uv: vec2<f32>, partial_lod: f32) -> ResolvedMaterial {
    var m: ResolvedMaterial;
    // Fold in this material's texture-size term (mip-0 sentinel stays < 0).
    let lod = partial_lod + material.texel_lod_bias;

    m.base_color = material.base_color.rgb;
    if material.base_color_texture_id != TEXTURE_MAP_NONE {
        m.base_color *= sample_texture_lod(material.base_color_texture_id, uv, lod);
    }

    m.emissive = material.emissive.rgb;
    if material.emissive_texture_id != TEXTURE_MAP_NONE {
        m.emissive *= sample_texture_lod(material.emissive_texture_id, uv, lod);
    }

    m.reflectance = material.reflectance;

    m.perceptual_roughness = material.perceptual_roughness;
    m.metallic = material.metallic;
    if material.metallic_roughness_texture_id != TEXTURE_MAP_NONE {
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

// Bindless geometry addresses (set 1, binding 4). The interleaved `PackedVertex`
// pool, the SoA position pool, and the materials buffer are reached by
// buffer-device-address via `physical_load` — `vertex_packed` is a 16-byte record
// per vertex (normal@0, tangent@4, uv@8); position lives in `vertex_positions`
// (stride 12) and is read by the closest-hit from the AS via position fetch.
struct SolariGeometryAddresses {
    vertex_packed: u64,
    vertex_positions: u64,
    materials: u64,
    material_stride: u32,
    _pad: u32,
    // Per-CLAS tessellation metadata table (`TessCluster` records). 0 = smooth-tess
    // path off → tess hits shade with the facet normal.
    tess_clusters: u64,
    // Optional per-vertex custom-data pool (stride 4). A custom closest-hit reads
    // it via `load_vertex_custom` / `load_triangle_custom`; built-in chits ignore it.
    vertex_custom: u64,
    // Deformed normals (u32 octa) + tangents (vec4) pools, and the slot-indexed
    // animated table (16 B/instance: flag, deform_pool_base, mesh_vertex_base). 0 =
    // no animation; the resolve then shades animated hits with rest-pose attrs.
    deform_normals: u64,
    deform_tangents: u64,
    animated_table: u64,
}
@group(1) @binding(4) var<uniform> geometry_addresses: SolariGeometryAddresses;

// Load one vertex's custom `u32` (generic user data baked into the mesh) by global
// vertex index. Zero for meshes that didn't author the attribute.
fn load_vertex_custom(vertex_index: u32) -> u32 {
    return physical_load<u32>(geometry_addresses.vertex_custom + u64(vertex_index) * u64(4u));
}

// The custom `u32` of a triangle's three vertices (real-cluster path only — same
// index math as the resolve). A custom closest-hit unpacks + barycentric-blends.
fn load_triangle_custom(cluster_global_id: u32, triangle_id: u32) -> vec3<u32> {
    let cluster = clusters[cluster_global_id];
    let idx_base = cluster.index_offset + triangle_id * 3u;
    return vec3<u32>(
        load_vertex_custom(cluster.vertex_offset + cluster_indices[idx_base + 0u]),
        load_vertex_custom(cluster.vertex_offset + cluster_indices[idx_base + 1u]),
        load_vertex_custom(cluster.vertex_offset + cluster_indices[idx_base + 2u]),
    );
}

// Per-CLAS tessellation metadata (16 B, `geometry_addresses.tess_clusters` +
// `(cluster_id - TESS_CLUSTER_ID_BASE) * 16`) is loaded field-wise in the closest-hit:
// normal-buffer device address (two u32) + the CLAS's first micro-triangle.

/// Tess CLAS `cluster_id` base — every tessellated CLAS bakes
/// `TESS_CLUSTER_ID_BASE + global_tess_clas_index` as its ClusterIDNV, comfortably
/// above any real cluster pool so the closest-hit detects the tess hit and recovers
/// the tess index. Must match `tess_template::TESS_CLUSTER_ID_BASE`.
const TESS_CLUSTER_ID_BASE: u32 = 0xF0000000u;

/// Minimum perceptual roughness for tessellated displacement hits. Sub-pixel
/// displacement detail acts as extra microfacet roughness; clamping the lobe wider
/// keeps the direct-light specular from aliasing into DLSS flicker on the bumpy
/// surface (physically: the geometry you can't resolve becomes roughness).
const TESS_ROUGHNESS_FLOOR: f32 = 0.5;

// Bindless material fetch: load the whole Material struct by buffer-device-address
// (uniform across the warp after SER).
fn load_material_bindless(material_id: u32) -> Material {
    let addr = geometry_addresses.materials
        + u64(material_id) * u64(geometry_addresses.material_stride);
    return physical_load<Material>(addr);
}

/// Load (decode) one interleaved `PackedVertex` at a global vertex index from the
/// bindless pool — one contiguous record instead of four parallel SoA fetches.
// Decode a [`PackedVertex::tangent`] u32 → (xyz dir, bitangent sign in w): 10 bits
// per direction component (unorm of [-1,1]) + sign in bit 30. Renormalize since
// 10-bit quantization perturbs the length slightly.
fn unpack_tangent(p: u32) -> vec4<f32> {
    let x = f32(p & 0x3ffu) / 1023.0 * 2.0 - 1.0;
    let y = f32((p >> 10u) & 0x3ffu) / 1023.0 * 2.0 - 1.0;
    let z = f32((p >> 20u) & 0x3ffu) / 1023.0 * 2.0 - 1.0;
    let w = select(1.0, -1.0, ((p >> 30u) & 1u) == 1u);
    return vec4<f32>(normalize(vec3<f32>(x, y, z)), w);
}

fn load_cluster_vertex(vertex_index: u32) -> Vertex {
    var v: Vertex;
    // Position lives in the separate SoA pool (stride 12) — the same buffer the CLAS
    // build reads. The interleaved packed pool holds only the 16-byte shading attrs.
    v.position = physical_load<vec3<f32>>(
        geometry_addresses.vertex_positions + u64(vertex_index) * u64(12u),
    );
    let base = geometry_addresses.vertex_packed + u64(vertex_index) * u64(16u);
    v.normal = octahedral_decode_signed(unpack2x16snorm(physical_load<u32>(base)));
    v.tangent = unpack_tangent(physical_load<u32>(base + u64(4u)));
    v.uv = physical_load<vec2<f32>>(base + u64(8u));
    return v;
}

/// Like [`load_cluster_vertex`] but skips position entirely: the position-fetch path
/// reads the three hit-triangle positions from `@builtin(hit_triangle_vertex_positions)`
/// instead, so only normal / tangent / uv are fetched from the 16-byte packed pool.
/// `position` is left at the `Vertex` default; the caller fills it from the builtin.
fn load_cluster_vertex_attrs(vertex_index: u32) -> Vertex {
    var v: Vertex;
    let base = geometry_addresses.vertex_packed + u64(vertex_index) * u64(16u);
    v.normal = octahedral_decode_signed(unpack2x16snorm(physical_load<u32>(base)));
    v.tangent = unpack_tangent(physical_load<u32>(base + u64(4u)));
    v.uv = physical_load<vec2<f32>>(base + u64(8u));
    return v;
}

/// Override a vertex's normal/tangent with the deformed values at deform-pool slot
/// `di` (positions come from the hit; uv is deform-invariant).
fn apply_deform_attrs(v: Vertex, di: u32) -> Vertex {
    var out = v;
    out.normal = octahedral_decode_signed(unpack2x16snorm(
        physical_load<u32>(geometry_addresses.deform_normals + u64(di) * u64(4u)),
    ));
    out.tangent = physical_load<vec4<f32>>(geometry_addresses.deform_tangents + u64(di) * u64(16u));
    return out;
}

/// Just the UV of a packed vertex (the alpha-test fast path needs no other field).
fn load_packed_uv(vertex_index: u32) -> vec2<f32> {
    return physical_load<vec2<f32>>(
        geometry_addresses.vertex_packed + u64(vertex_index) * u64(16u) + u64(8u),
    );
}

fn resolve_ray_hit_full(ray_hit: RayIntersection) -> ResolvedRayHitFull {
    let barycentrics = vec3(1.0 - ray_hit.barycentrics.x - ray_hit.barycentrics.y, ray_hit.barycentrics);
    return resolve_triangle_data_full(
        ray_hit.instance_index,
        ray_hit.geometry_index,
        ray_hit.primitive_index,
        barycentrics,
    );
}

// Apply a `mat3x4` affine (column k = 4x4 row k) to a point: component
// k = dot(linear row k, p) + translation k. Mirrors the cluster path's
// `apply_affine`.
fn affine_transform_point(m: mat3x4<f32>, p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(m[0].xyz, p) + m[0].w,
        dot(m[1].xyz, p) + m[1].w,
        dot(m[2].xyz, p) + m[2].w,
    );
}

// Apply the linear part only (for directions / normals).
fn affine_transform_direction(m: mat3x4<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(m[0].xyz, v), dot(m[1].xyz, v), dot(m[2].xyz, v));
}

// Nudge past the exit portal's surface so the continued ray doesn't immediately
// re-hit it.
const PORTAL_SURFACE_OFFSET: f32 = 1e-4;

// Inverse of a row-packed affine (mat3x4, column k = the 4x4's row k):
// 3x3 adjugate for the linear part, `-L⁻¹·t` for the translation.
fn affine_inverse(m: mat3x4<f32>) -> mat3x4<f32> {
    // The linear part's COLUMNS (standard math convention).
    let c0 = vec3(m[0].x, m[1].x, m[2].x);
    let c1 = vec3(m[0].y, m[1].y, m[2].y);
    let c2 = vec3(m[0].z, m[1].z, m[2].z);
    let t = vec3(m[0].w, m[1].w, m[2].w);
    let inv_det = 1.0 / dot(c0, cross(c1, c2));
    let r0 = cross(c1, c2) * inv_det;
    let r1 = cross(c2, c0) * inv_det;
    let r2 = cross(c0, c1) * inv_det;
    return mat3x4<f32>(
        vec4(r0, -dot(r0, t)),
        vec4(r1, -dot(r1, t)),
        vec4(r2, -dot(r2, t)),
    );
}

// If `hit_instance` is a portal surface, rewrite the ray to continue from the
// paired portal and return true; else leave the ray and return false. The map is
// `W_target · R_y(π) · W_portal⁻¹` — into portal-local space, a half-turn about
// local Y (so the BACK of the target shows looking into the front of this one),
// out through the target's frame. Reads the live instance transform column, so
// moving / GPU-propagated portals stay exact. Light is NOT transported: portals
// carry the view, not next-event estimation. Called only from `chit_portal`
// (SBT-routed), so the scan + this math never touch the opaque chit's registers.
fn portal_redirect(
    hit_instance: u32,
    hit_position: vec3<f32>,
    ray_origin: ptr<function, vec3<f32>>,
    ray_direction: ptr<function, vec3<f32>>,
) -> bool {
    for (var i = 0u; i < arrayLength(&portals); i += 1u) {
        if portals[i].valid == 0u || portals[i].instance_slot != hit_instance {
            continue;
        }
        let into_portal = affine_inverse(transforms[portals[i].instance_slot]);
        let out_of_target = transforms[portals[i].target_slot];
        var p = affine_transform_point(into_portal, hit_position);
        var d = affine_transform_direction(into_portal, *ray_direction);
        p = vec3(-p.x, p.y, -p.z);
        d = vec3(-d.x, d.y, -d.z);
        let direction = normalize(affine_transform_direction(out_of_target, d));
        *ray_origin = affine_transform_point(out_of_target, p) + direction * PORTAL_SURFACE_OFFSET;
        *ray_direction = direction;
        return true;
    }
    return false;
}


// One march step's combined fog-volume medium at `p`: extinction, plain
// scattering (the isotropic sky-ambient term), and sun-phase-weighted
// scattering — each volume applies its own HG lobe at the caller's sun angle.
// The global height fog is NOT included; the marches add it (its phase uses
// the per-view atmosphere `g`). `phase_g_sum / phase_weight` is the
// scattering-weighted HG asymmetry for evaluating the phase toward an
// ARBITRARY direction (a local-light NEE sample) when overlapping media
// carry different `g` — the sun direction is fixed per ray, light samples
// aren't.
struct FogSample {
    sigma_t: f32,
    sigma_s: vec3<f32>,
    sun_scatter: vec3<f32>,
    phase_g_sum: f32,
    phase_weight: f32,
}

fn fog_volumes_sample(p: vec3<f32>, cos_theta: f32) -> FogSample {
    var s = FogSample(0.0, vec3(0.0), vec3(0.0), 0.0, 0.0);
    for (var i = 0u; i < arrayLength(&fog_volumes); i += 1u) {
        let vol = fog_volumes[i];
        if vol.scattering.w < 1e-7 {
            continue;
        }
        let lp = abs(affine_transform_point(vol.local_from_world, p));
        // Distance metric to the unit shape — 1 at the boundary: Chebyshev
        // for the box, Euclidean for the sphere.
        var m = max(lp.x, max(lp.y, lp.z));
        if vol.params.z > 0.5 {
            m = length(lp);
        }
        if m >= 1.0 {
            continue;
        }
        // Edge softness: full density in the core, fading to zero over the
        // outer `softness` fraction of the shape.
        let density = saturate((1.0 - m) / max(vol.params.y, 1e-4));
        let sigma_s = vol.scattering.xyz * density;
        s.sigma_t += vol.scattering.w * density;
        s.sigma_s += sigma_s;
        s.sun_scatter += sigma_s * atmosphere_mie_phase(vol.params.x, cos_theta);
        let w = sigma_s.x + sigma_s.y + sigma_s.z;
        s.phase_g_sum += vol.params.x * w;
        s.phase_weight += w;
    }
    return s;
}

// The `[t_entry, t_exit]` span of `[0, t_cap]` along a ray that can contain
// any fog volume (union over volumes; `y <= x` = none). Lets the aerial
// marches concentrate their steps on the occupied segment — and skip the
// march entirely — when the global height fog is off.
fn fog_volumes_range(origin: vec3<f32>, direction: vec3<f32>, t_cap: f32) -> vec2<f32> {
    var range = vec2(t_cap, 0.0);
    for (var i = 0u; i < arrayLength(&fog_volumes); i += 1u) {
        let vol = fog_volumes[i];
        if vol.scattering.w < 1e-7 {
            continue;
        }
        let lo = affine_transform_point(vol.local_from_world, origin);
        let ld = affine_transform_direction(vol.local_from_world, direction);
        var t0: f32;
        var t1: f32;
        if vol.params.z > 0.5 {
            // Unit sphere.
            let a = dot(ld, ld);
            let b = dot(lo, ld);
            let disc = b * b - a * (dot(lo, lo) - 1.0);
            if disc < 0.0 {
                continue;
            }
            let sq = sqrt(disc);
            t0 = (-b - sq) / a;
            t1 = (-b + sq) / a;
        } else {
            // Unit-box slabs (axis-parallel rays resolve through ±inf).
            let inv_d = 1.0 / ld;
            let ta = (vec3(-1.0) - lo) * inv_d;
            let tb = (vec3(1.0) - lo) * inv_d;
            let tmin = min(ta, tb);
            let tmax = max(ta, tb);
            t0 = max(tmin.x, max(tmin.y, tmin.z));
            t1 = min(tmax.x, min(tmax.y, tmax.z));
            if t0 > t1 {
                continue;
            }
        }
        if t1 < 0.0 || t0 > t_cap {
            continue;
        }
        range.x = min(range.x, max(t0, 0.0));
        range.y = max(range.y, min(t1, t_cap));
    }
    return range;
}


fn transform_positions(transform: mat3x4<f32>, vertices: array<Vertex, 3>) -> array<vec3<f32>, 3> {
    return array<vec3<f32>, 3>(
        affine_transform_point(transform, vertices[0].position),
        affine_transform_point(transform, vertices[1].position),
        affine_transform_point(transform, vertices[2].position),
    );
}

/// Resolve full shading data for a ray hit. `cluster_global_id`
/// comes straight from `RayIntersection.geometry_index` — baked
/// into each CLAS's `base_geometry_index_and_geometry_flags` at
/// upload time as the global cluster pool slot.
fn resolve_triangle_data_full(
    instance_id: u32,
    cluster_global_id: u32,
    triangle_id: u32,
    barycentrics: vec3<f32>,
) -> ResolvedRayHitFull {
    // No ray cone (cone_width < 0) → every texture samples mip 0.
    return resolve_triangle_data_full_cone(instance_id, cluster_global_id, triangle_id, barycentrics, -1.0, vec3(0.0));
}

/// Mip-0, material-explicit resolve (see `resolve_triangle_data_full_cone_mat`):
/// the closest-hit passes the material id from its SBT shader record.
fn resolve_triangle_data_full_mat(
    instance_id: u32,
    material_id: u32,
    transform: mat3x4<f32>,
    cluster_global_id: u32,
    triangle_id: u32,
    barycentrics: vec3<f32>,
) -> ResolvedRayHitFull {
    return resolve_triangle_data_full_cone_mat(instance_id, material_id, transform, cluster_global_id, triangle_id, barycentrics, -1.0, vec3(0.0));
}

fn resolve_triangle_data_full_cone(
    instance_id: u32,
    cluster_global_id: u32,
    triangle_id: u32,
    barycentrics: vec3<f32>,
    cone_width: f32,
    ray_direction: vec3<f32>,
) -> ResolvedRayHitFull {
    // Default material binding: the `material_ids[instance_id]` indirection. The
    // inline-rayQuery (megakernel) path resolves material this way.
    return resolve_triangle_data_full_cone_mat(
        instance_id,
        material_ids[instance_id],
        transforms[instance_id],
        cluster_global_id,
        triangle_id,
        barycentrics,
        cone_width,
        ray_direction,
    );
}

/// Material-explicit variant: the caller supplies `material_id` directly instead
/// of the `material_ids[instance_id]` indirection — e.g. a closest-hit reading it
/// from its per-material SBT shader record (uniform per warp after SER, the
/// canonical RT material binding). Geometry still resolves from `instance_id` /
/// `cluster_global_id` / `triangle_id`.
fn resolve_triangle_data_full_cone_mat(
    instance_id: u32,
    material_id: u32,
    // Object→world affine (row-form). The megakernel passes `transforms[instance_id]`;
    // the RT closest-hit passes the TLAS hit's `ObjectToWorld` builtin directly (the
    // acceleration structure already holds it — no buffer read needed).
    transform: mat3x4<f32>,
    cluster_global_id: u32,
    triangle_id: u32,
    barycentrics: vec3<f32>,
    cone_width: f32,
    ray_direction: vec3<f32>,
) -> ResolvedRayHitFull {
    let cluster = clusters[cluster_global_id];
    // `cluster_indices` stores per-cluster LOCAL vertex indices
    // (0..vertex_count) — the bake widens meshopt's u8 locals to
    // u32 but does NOT rebase them. The matching cluster CLAS was
    // built with `vertex_buffer = vertex_buffer_addr + vertex_offset
    // * 12`, so the driver applies the same offset internally.
    // Reconstruct the global vertex-pool slot here: add the
    // cluster's (already globally-rebased) `vertex_offset`.
    let idx_base = cluster.index_offset + triangle_id * 3u;
    let vertices = array<Vertex, 3>(
        load_cluster_vertex(cluster.vertex_offset + cluster_indices[idx_base + 0u]),
        load_cluster_vertex(cluster.vertex_offset + cluster_indices[idx_base + 1u]),
        load_cluster_vertex(cluster.vertex_offset + cluster_indices[idx_base + 2u]),
    );
    return resolve_triangle_data_core(
        instance_id,
        material_id,
        transform,
        cluster.triangle_count,
        barycentrics,
        cone_width,
        vertices,
    );
}

/// Position-fetch variant for the RT closest-hit: the hit triangle's three
/// object-space vertex positions come from `@builtin(hit_triangle_vertex_positions)`
/// (`VK_KHR_ray_tracing_position_fetch`) rather than the vertex pool, so the three
/// `physical_load<vec3<f32>>` position fetches are skipped — only normal / tangent /
/// uv are read. These positions are exactly the geometry the ray traversed (the CLAS
/// triangle), so `world_position` + the geometric normal come from the true hit
/// surface. Mip-0 (no ray cone), matching `resolve_triangle_data_full_mat`.
fn resolve_triangle_data_full_mat_fetch(
    instance_id: u32,
    material_id: u32,
    transform: mat3x4<f32>,
    cluster_global_id: u32,
    triangle_id: u32,
    barycentrics: vec3<f32>,
    object_positions: array<vec3<f32>, 3>,
) -> ResolvedRayHitFull {
    // Tessellation showcase: its CLAS is instantiated with a ClusterIDNV above the
    // real cluster pool, so it has no `clusters[]` entry and its displaced
    // micro-vertices have no vertex-pool attributes. Two shading paths:
    //   - SMOOTH (`geometry_addresses.tess_clusters != 0`): recover the three
    //     micro-vertex smooth, displacement-aware normals from the per-CLAS metadata
    //     table + per-instance normal buffer and let the core interpolate them, so the
    //     displaced surface shades without micro-triangle faceting.
    //   - FACET (table unwired): geometric normal from the position-fetch positions.
    // UV is fixed (no per-micro-vertex UVs yet) in both; textured tess shading is a
    // later brick.
    if cluster_global_id >= arrayLength(&clusters) {
        if geometry_addresses.tess_clusters != 0 {
            // Per-CLAS metadata (16 B): normal-buffer address (lo/hi u32) + primitive
            // base. Loaded field-wise (struct `physical_load` of a u64 member is risky).
            let tc_addr = geometry_addresses.tess_clusters
                + u64(cluster_global_id - TESS_CLUSTER_ID_BASE) * u64(16u);
            let normals_lo = physical_load<u32>(tc_addr + u64(0u));
            let normals_hi = physical_load<u32>(tc_addr + u64(4u));
            let primitive_base = physical_load<u32>(tc_addr + u64(8u));
            let attrs = u64(normals_lo) | (u64(normals_hi) << 32u);
            // Denormalized per-vertex attrs: 3 × (packed normal @0 + UV @4) = 36 B / tri.
            let base = attrs + u64(primitive_base + triangle_id) * u64(36u);
            let a0 = base + u64(0u);
            let a1 = base + u64(12u);
            let a2 = base + u64(24u);
            let n0 = octahedral_decode_signed(unpack2x16snorm(physical_load<u32>(a0)));
            let n1 = octahedral_decode_signed(unpack2x16snorm(physical_load<u32>(a1)));
            let n2 = octahedral_decode_signed(unpack2x16snorm(physical_load<u32>(a2)));
            let uv0 = physical_load<vec2<f32>>(a0 + u64(4u));
            let uv1 = physical_load<vec2<f32>>(a1 + u64(4u));
            let uv2 = physical_load<vec2<f32>>(a2 + u64(4u));
            // Real UV-gradient tangent (needed for normal maps) derived from the
            // micro-triangle's positions + UVs — the tess attrs carry no tangent, and a
            // fixed `cross(up,n)` tangent makes normal maps light wrong. The positions
            // are world-space (gen bakes world; inject is identity) and so is the result;
            // the identity `transform` leaves it world below. Gram-Schmidt against n0 +
            // handedness sign; fall back to an arbitrary tangent on degenerate UVs.
            let e1 = object_positions[1] - object_positions[0];
            let e2 = object_positions[2] - object_positions[0];
            let duv1 = uv1 - uv0;
            let duv2 = uv2 - uv0;
            let det = duv1.x * duv2.y - duv2.x * duv1.y;
            var tangent: vec4<f32>;
            if abs(det) > 1e-12 {
                let r = 1.0 / det;
                let t = (e1 * duv2.y - e2 * duv1.y) * r;
                let bt = (e2 * duv1.x - e1 * duv2.x) * r;
                let tan = normalize(t - n0 * dot(n0, t));
                let w = select(-1.0, 1.0, dot(cross(n0, tan), bt) > 0.0);
                tangent = vec4<f32>(tan, w);
            } else {
                let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n0.y) > 0.99);
                tangent = vec4<f32>(normalize(cross(up, n0)), 1.0);
            }
            let sv0 = Vertex(object_positions[0], n0, uv0, tangent);
            let sv1 = Vertex(object_positions[1], n1, uv1, tangent);
            let sv2 = Vertex(object_positions[2], n2, uv2, tangent);
            var hit = resolve_triangle_data_core(
                instance_id, material_id, transform, 1u, barycentrics, -1.0,
                array<Vertex, 3>(sv0, sv1, sv2),
            );
            // Tess motion vectors: the per-instance previous-frame transform is
            // unreliable for tessellated surfaces — the base mesh is hidden
            // (`RenderLayers::none`), so its `previous_frame_transforms` slot is stale or
            // unresolved (`instance_id` falls back to 0). The resulting previous position
            // is garbage → wrong motion → DLSS can't accumulate the illumination and any
            // shadows on the surface strobe. Treat the surface as STATIC: previous world
            // position == current (motion = camera-only), which is exact for static
            // displaced geometry (floors, planet terrain). MOVING tessellated surfaces
            // would ghost and need the real per-instance previous transform — TODO.
            hit.previous_frame_world_position = hit.world_position;
            hit.material.perceptual_roughness = max(hit.material.perceptual_roughness, TESS_ROUGHNESS_FLOOR);
            hit.material.roughness = hit.material.perceptual_roughness * hit.material.perceptual_roughness;
            return hit;
        }
        let cr = cross(
            object_positions[1] - object_positions[0],
            object_positions[2] - object_positions[0],
        );
        let cl = length(cr);
        let flat_normal = select(vec3<f32>(0.0, 0.0, 1.0), cr / cl, cl > 1e-12);
        // Arbitrary tangent perpendicular to the normal (only consulted if the
        // material carries a normal map).
        let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(flat_normal.y) > 0.99);
        let tangent = vec4<f32>(normalize(cross(up, flat_normal)), 1.0);
        let tv0 = Vertex(object_positions[0], flat_normal, vec2<f32>(0.5), tangent);
        let tv1 = Vertex(object_positions[1], flat_normal, vec2<f32>(0.5), tangent);
        let tv2 = Vertex(object_positions[2], flat_normal, vec2<f32>(0.5), tangent);
        var hit = resolve_triangle_data_core(
            instance_id, material_id, transform, 1u, barycentrics, -1.0,
            array<Vertex, 3>(tv0, tv1, tv2),
        );
        // Same tess motion fix as the smooth branch: static previous position
        // (motion = camera-only) instead of the unreliable per-instance lookup.
        hit.previous_frame_world_position = hit.world_position;
        hit.material.perceptual_roughness = max(hit.material.perceptual_roughness, TESS_ROUGHNESS_FLOOR);
        hit.material.roughness = hit.material.perceptual_roughness * hit.material.perceptual_roughness;
        return hit;
    }

    let cluster = clusters[cluster_global_id];
    let idx_base = cluster.index_offset + triangle_id * 3u;
    let gvi0 = cluster.vertex_offset + cluster_indices[idx_base + 0u];
    let gvi1 = cluster.vertex_offset + cluster_indices[idx_base + 1u];
    let gvi2 = cluster.vertex_offset + cluster_indices[idx_base + 2u];
    // Attrs-only fetch (skips the position head); the builtin supplies position.
    var v0 = load_cluster_vertex_attrs(gvi0);
    var v1 = load_cluster_vertex_attrs(gvi1);
    var v2 = load_cluster_vertex_attrs(gvi2);
    v0.position = object_positions[0];
    v1.position = object_positions[1];
    v2.position = object_positions[2];
    // Animated instance: positions already come from the deformed hit triangle, but
    // the pooled attrs are rest-pose — swap normal/tangent for the deformed values.
    if geometry_addresses.animated_table != 0 {
        let a = geometry_addresses.animated_table + u64(instance_id) * u64(16u);
        if physical_load<u32>(a) == 1u {
            let pool_base = physical_load<u32>(a + u64(4u));
            let mvb = physical_load<u32>(a + u64(8u));
            v0 = apply_deform_attrs(v0, pool_base + (gvi0 - mvb));
            v1 = apply_deform_attrs(v1, pool_base + (gvi1 - mvb));
            v2 = apply_deform_attrs(v2, pool_base + (gvi2 - mvb));
        }
    }
    return resolve_triangle_data_core(
        instance_id,
        material_id,
        transform,
        cluster.triangle_count,
        barycentrics,
        -1.0,
        array<Vertex, 3>(v0, v1, v2),
    );
}

/// Shared resolve body: given the hit triangle's three fully-populated vertices
/// (position + normal + tangent + uv), the instance transform, and the cluster's
/// triangle count, produce the full shading hit. Fed by both the vertex-pool path
/// (`resolve_triangle_data_full_cone_mat`) and the position-fetch path
/// (`resolve_triangle_data_full_mat_fetch`).
fn resolve_triangle_data_core(
    instance_id: u32,
    material_id: u32,
    transform: mat3x4<f32>,
    triangle_count: u32,
    barycentrics: vec3<f32>,
    cone_width: f32,
    vertices: array<Vertex, 3>,
) -> ResolvedRayHitFull {
    let material = load_material_bindless(material_id);
    let previous_frame_transform = previous_frame_transforms[instance_id];

    let world_vertices = transform_positions(transform, vertices);
    let world_position = mat3x3(world_vertices[0], world_vertices[1], world_vertices[2]) * barycentrics;

    let previous_frame_world_vertices = transform_positions(previous_frame_transform, vertices);
    let previous_frame_world_position = mat3x3(previous_frame_world_vertices[0], previous_frame_world_vertices[1], previous_frame_world_vertices[2]) * barycentrics;

    let uv = mat3x2(vertices[0].uv, vertices[1].uv, vertices[2].uv) * barycentrics;

    let local_tangent = mat3x3(vertices[0].tangent.xyz, vertices[1].tangent.xyz, vertices[2].tangent.xyz) * barycentrics;
    let world_tangent = vec4(
        normalize(affine_transform_direction(transform, local_tangent)),
        vertices[0].tangent.w,
    );

    let local_normal = mat3x3(vertices[0].normal, vertices[1].normal, vertices[2].normal) * barycentrics;
    var world_normal = normalize(affine_transform_direction(transform, local_normal));

    let triangle_edge0 = world_vertices[0] - world_vertices[1];
    let triangle_edge1 = world_vertices[0] - world_vertices[2];
    let triangle_cross = cross(triangle_edge0, triangle_edge1);
    let triangle_area = length(triangle_cross) / 2.0;

    // True (planar) triangle normal. The cross product's sign follows the
    // index winding, which these assets don't keep consistent (authored for
    // double-sided raster) — sign-match it to the interpolated vertex normal,
    // which IS consistently outward, so inside/outside tests get an exact
    // plane with a stable orientation.
    var geometric_world_normal = normalize(triangle_cross);
    if dot(geometric_world_normal, world_normal) < 0.0 {
        geometric_world_normal = -geometric_world_normal;
    }

    // Ray-cone texture LOD (Akenine-Möller et al., Ray Tracing Gems ch.20). The
    // texture-size-INDEPENDENT term: the triangle's texel density (UV area vs
    // world area) plus the cone footprint at this hit; `Material.texel_lod_bias`
    // adds each texture's own `0.5·log2(w·h)`. The grazing-angle (1/cosθ) term is
    // omitted to keep this off the megakernel's register-pressure path. A
    // negative `cone_width` (non-path-tracer callers) keeps the mip-0 sentinel.
    var partial_lod = TEXTURE_LOD_MIP0;
    if cone_width >= 0.0 {
        let uv_edge0 = vertices[1].uv - vertices[0].uv;
        let uv_edge1 = vertices[2].uv - vertices[0].uv;
        let uv_area = 0.5 * abs(uv_edge0.x * uv_edge1.y - uv_edge0.y * uv_edge1.x);
        partial_lod = 0.5 * log2(max(uv_area, 1e-12) / max(triangle_area, 1e-8))
            + log2(max(cone_width, 1e-6));
    }

    if material.normal_map_texture_id != TEXTURE_MAP_NONE {
        let TBN = calculate_tbn_mikktspace(world_normal, world_tangent);
        let T = TBN[0];
        let B = TBN[1];
        let N = TBN[2];
        let Nt = sample_texture_lod(material.normal_map_texture_id, uv, partial_lod + material.texel_lod_bias);
        world_normal = normalize(Nt.x * T + Nt.y * B + Nt.z * N);
    }

    let resolved_material = resolve_material_lod(material, uv, partial_lod);

    return ResolvedRayHitFull(
        world_position,
        previous_frame_world_position,
        world_normal,
        geometric_world_normal,
        world_tangent,
        uv,
        triangle_area,
        triangle_count,
        resolved_material,
        material_id,
    );
}
