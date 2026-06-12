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

// Cluster mesh pool — shared across every `RaytracingMesh3d` instance
// in the scene. Indices stored here are GLOBAL pool slots.
@group(0) @binding(0)  var<storage> vertex_positions: array<f32>;        // stride 3 floats / vertex
@group(0) @binding(1)  var<storage> vertex_normals: array<u32>;          // octahedral 2x16snorm
@group(0) @binding(2)  var<storage> vertex_tangents: array<vec4<f32>>;
@group(0) @binding(3)  var<storage> vertex_uvs: array<vec2<f32>>;
@group(0) @binding(4)  var<storage> cluster_indices: array<u32>;
@group(0) @binding(5)  var<storage> clusters: array<Cluster>;

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

// Materials + textures.
@group(0) @binding(6)  var<storage> materials: array<Material>;
@group(0) @binding(7) var textures: binding_array<texture_2d<f32>>;
@group(0) @binding(8) var samplers: binding_array<sampler>;

// Ray-tracing acceleration structure + lighting.
@group(0) @binding(9) var tlas: acceleration_structure;
// STABLE-SLOT indexed: a light keeps its index for its lifetime (freed slots
// hole out as `LIGHT_SOURCE_KIND_NONE` and are reused), so a slot stored in a
// reservoir / light tile stays a valid identity across light add/remove.
@group(0) @binding(10) var<storage> light_sources: array<LightSource>;
// The uniform-pick list over ACTIVE lights: `[emissive_count,
// directional_count]` header, then the active emissive slots, then the active
// directional slots (strata contiguous for the stratified pick).
@group(0) @binding(17) var<storage> active_light_list: array<u32>;
// `directional_lights` is also a scene column (the lights table owns it).
@group(#{SOLARI_SCENE_COLUMNS_GROUP}) @binding(3) var<storage> directional_lights: array<DirectionalLight>;

// BRDF DFG LUT.
@group(0) @binding(11) var brdf_dfg_lut: texture_2d<f32>;
@group(0) @binding(12) var brdf_dfg_lut_sampler: sampler;

// Skeletal cluster animation: per-instance deformed-vertex pool + the
// slot-indexed table the hit shader branches on. `deform_positions` /
// `deform_normals` hold MESH-LOCAL skinned verts (the deform pass premultiplies
// the inverse instance world), so the same `transforms[instance_id]` is applied
// downstream exactly as for static instances. Filled by `bindings::binder`.
struct AnimatedInstance {
    flag: u32,             // 1 = animated (read the deform pool), 0 = static
    deform_pool_base: u32, // base (in vertices) into the deform pool for this instance
    mesh_vertex_base: u32, // global vertex slot of this mesh's vertex 0
    _pad: u32,
}
@group(0) @binding(13) var<storage> deform_positions: array<f32>; // stride 3 f32 / vertex
@group(0) @binding(14) var<storage> deform_normals: array<u32>;   // octahedral 2x16snorm
@group(0) @binding(15) var<storage> instance_animated: array<AnimatedInstance>;
@group(0) @binding(16) var<storage> deform_tangents: array<vec4<f32>>; // xyz + w bitangent sign

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

// Ray portals (`bindings::portal`, the `SolariPortals` gpu_table, indexed by
// portal slot): the instance-slot pairing the traversal scan matches hits
// against. The ray map derives on hit from the LIVE GPU transform table
// (`transforms`), so portals on GPU-propagated / moving parents stay exact.
// `valid == 0` covers tombstones AND the zero-initialized tail of a grown
// column buffer (instance slot 0 is real — zeros must read as inert).
struct Portal {
    instance_slot: u32,
    target_slot: u32,
    valid: u32,
    _pad: u32,
}
@group(#{SOLARI_SCENE_COLUMNS_GROUP}) @binding(5) var<storage> portals: array<Portal>;

// Black holes (`bindings::black_hole`, the `SolariBlackHoles` gpu_table):
// regions that gravitationally bend rays. All-zero entries (tombstones,
// grown-buffer tail) are inert: a zero influence radius never matches a ray.
struct BlackHole {
    // xyz = world center, w = influence radius.
    center_influence: vec4<f32>,
    // xyz = accretion-disk plane normal, w = Schwarzschild radius.
    normal_rs: vec4<f32>,
    // x = disk inner radius, y = outer radius, z = emission scale.
    disk: vec4<f32>,
}
@group(#{SOLARI_SCENE_COLUMNS_GROUP}) @binding(6) var<storage> black_holes: array<BlackHole>;

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

const RAY_T_MIN = 0.001f;
const RAY_T_MAX = 100000.0f;

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

// Mask test for a candidate triangle: base-color texture alpha at the hit UV
// against the material's cutoff. No texture = solid (alpha 1). The base-color
// FACTOR's alpha is not applied (the binder stores rgb only) — glTF cutouts
// author the mask in the texture.
fn alpha_test(hit: RayIntersection) -> bool {
    let material = materials[material_ids[hit.instance_index]];
    if material.alpha_mask < 0.0 {
        return true; // opaque material on a non-opaque instance (stale flag)
    }
    let texture_id = material.base_color_texture_id;
    if texture_id == TEXTURE_MAP_NONE {
        return material.alpha_mask <= 1.0;
    }
    let cluster = clusters[hit.geometry_index];
    let idx_base = cluster.index_offset + hit.primitive_index * 3u;
    let uv0 = vertex_uvs[cluster.vertex_offset + cluster_indices[idx_base + 0u]];
    let uv1 = vertex_uvs[cluster.vertex_offset + cluster_indices[idx_base + 1u]];
    let uv2 = vertex_uvs[cluster.vertex_offset + cluster_indices[idx_base + 2u]];
    let barycentrics = vec3(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics);
    let uv = mat3x2(uv0, uv1, uv2) * barycentrics;
    let alpha = textureSampleLevel(textures[texture_id], samplers[texture_id], uv, 0.0).a;
    return alpha >= material.alpha_mask;
}

fn sample_texture(id: u32, uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(textures[id], samplers[id], uv, 0.0).rgb; // TODO: Mipmap
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

fn resolve_material(material: Material, uv: vec2<f32>) -> ResolvedMaterial {
    var m: ResolvedMaterial;

    m.base_color = material.base_color.rgb;
    if material.base_color_texture_id != TEXTURE_MAP_NONE {
        m.base_color *= sample_texture(material.base_color_texture_id, uv);
    }

    m.emissive = material.emissive.rgb;
    if material.emissive_texture_id != TEXTURE_MAP_NONE {
        m.emissive *= sample_texture(material.emissive_texture_id, uv);
    }

    m.reflectance = material.reflectance;

    m.perceptual_roughness = material.perceptual_roughness;
    m.metallic = material.metallic;
    if material.metallic_roughness_texture_id != TEXTURE_MAP_NONE {
        let metallic_roughness = sample_texture(material.metallic_roughness_texture_id, uv);
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

/// Load one vertex from the cluster pool. `vertex_index` is a global
/// pool slot (post-rebase by `ClusterMeshManager`). Vertex positions
/// are stored as a packed `array<f32>` with stride 3 (12 B per
/// vertex); other streams are parallel arrays indexed by the same
/// `vertex_index`.
fn load_cluster_vertex(vertex_index: u32) -> Vertex {
    var v: Vertex;
    let base = vertex_index * 3u;
    v.position = vec3<f32>(
        vertex_positions[base],
        vertex_positions[base + 1u],
        vertex_positions[base + 2u],
    );
    v.normal = octahedral_decode_signed(unpack2x16snorm(vertex_normals[vertex_index]));
    v.uv = vertex_uvs[vertex_index];
    v.tangent = vertex_tangents[vertex_index];
    return v;
}

/// As [`load_cluster_vertex`], but reads position + normal from the per-instance
/// deform pool when the instance is animated. UV + tangent are deform-invariant
/// (UV) / approximate (tangent), so they stay from the static cluster pool.
fn load_cluster_vertex_animated(vertex_index: u32, anim: AnimatedInstance) -> Vertex {
    var v: Vertex;
    if anim.flag == 1u {
        // Deform pool is per-instance, indexed mesh-local (vertex_index is global).
        let di = anim.deform_pool_base + (vertex_index - anim.mesh_vertex_base);
        let base = di * 3u;
        v.position = vec3<f32>(
            deform_positions[base],
            deform_positions[base + 1u],
            deform_positions[base + 2u],
        );
        v.normal = octahedral_decode_signed(unpack2x16snorm(deform_normals[di]));
        v.tangent = deform_tangents[di];
    } else {
        let base = vertex_index * 3u;
        v.position = vec3<f32>(
            vertex_positions[base],
            vertex_positions[base + 1u],
            vertex_positions[base + 2u],
        );
        v.normal = octahedral_decode_signed(unpack2x16snorm(vertex_normals[vertex_index]));
        v.tangent = vertex_tangents[vertex_index];
    }
    v.uv = vertex_uvs[vertex_index]; // UV is deform-invariant
    return v;
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

// Nudge past the exit portal's surface so the continued ray doesn't
// immediately re-hit it.
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

// If `hit_instance` is a portal surface, teleport the ray and return true
// (the caller re-traces instead of shading). The map is
// `W_target · R_y(π) · W_portal⁻¹` — into portal-local space, a half-turn
// about local Y (negate x and z: walk in face-first, exit face-first), out
// through the target's frame — derived from the live `transforms` column.
// Light is NOT transported: portals carry the view, not next-event
// estimation, so each side is lit by its own surroundings.
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
        *ray_origin = affine_transform_point(out_of_target, p)
            + direction * PORTAL_SURFACE_OFFSET;
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

const GRAVITY_MARCH_STEPS = 256u;

// Procedural accretion-disk radiance at disk radius `r`: white-hot inner
// edge cooling to deep orange-red at the rim.
fn black_hole_disk_emission(r: f32, disk: vec4<f32>) -> vec3<f32> {
    let t = saturate((r - disk.x) / max(disk.y - disk.x, 1e-4));
    let brightness = disk.z * (0.05 + pow(1.0 - t, 2.0));
    let color = mix(vec3(1.0, 0.96, 0.9), vec3(1.0, 0.3, 0.05), saturate(t * 1.6));
    return color * brightness;
}

// March the ray through black hole `i` from `*ray_origin` (already on or
// inside the influence sphere) until it exits or is captured. Schwarzschild
// photon bend: d(dir)/ds ∝ −1.5·r_s·h²·r⃗/r⁵ (h² = |r⃗×d̂|² is conserved) —
// the 1.5 is the GR factor that produces the photon ring. Thin-disk
// crossings accumulate emission into `*emitted`. Returns true if captured.
fn black_hole_march(
    i: u32,
    ray_origin: ptr<function, vec3<f32>>,
    ray_direction: ptr<function, vec3<f32>>,
    emitted: ptr<function, vec3<f32>>,
) -> bool {
    let hole = black_holes[i];
    let center = hole.center_influence.xyz;
    let influence = hole.center_influence.w;
    let disk_normal = hole.normal_rs.xyz;
    let rs = hole.normal_rs.w;

    var p = *ray_origin - center;
    var d = *ray_direction;
    let h = cross(p, d);
    let h2 = dot(h, h);
    var side = dot(p, disk_normal);

    for (var step = 0u; step < GRAVITY_MARCH_STEPS; step += 1u) {
        let r = length(p);
        if r < rs {
            return true; // inside the horizon — captured (black, not sky)
        }
        if r > influence {
            *ray_origin = p + center;
            *ray_direction = d;
            return false;
        }
        // Adaptive step: fine near the horizon (where the bend rate explodes),
        // coarse near the influence boundary.
        let ds = clamp(r * 0.1, rs * 0.05, influence * 0.04);
        d = normalize(d - (1.5 * rs * h2 / pow(r, 5.0)) * p * ds);
        p += d * ds;
        // Thin accretion disk: a plane crossing inside the annulus emits.
        // Wrapped rays cross repeatedly — the photon ring brightens itself.
        let new_side = dot(p, disk_normal);
        if hole.disk.z > 0.0 && side * new_side < 0.0 {
            let disk_r = length(p - disk_normal * new_side);
            if disk_r > hole.disk.x && disk_r < hole.disk.y {
                *emitted += black_hole_disk_emission(disk_r, hole.disk);
            }
        }
        side = new_side;
    }
    return true; // step budget spent spiraling — treat as captured
}

// Nearest black-hole influence-sphere entry along the segment [0, max_t),
// or `0xFFFFFFFF` if the segment never enters one. `entry_t` may be 0 (the
// origin is already inside a region).
fn nearest_black_hole_entry(origin: vec3<f32>, direction: vec3<f32>, max_t: f32, entry_t: ptr<function, f32>) -> u32 {
    var best = 0xFFFFFFFFu;
    for (var i = 0u; i < arrayLength(&black_holes); i += 1u) {
        let influence = black_holes[i].center_influence.w;
        if influence <= 0.0 {
            continue;
        }
        let oc = origin - black_holes[i].center_influence.xyz;
        let b = dot(oc, direction);
        let c = dot(oc, oc) - influence * influence;
        let disc = b * b - c;
        if disc < 0.0 {
            continue;
        }
        let t0 = max(-b - sqrt(disc), 0.0);
        let t1 = -b + sqrt(disc);
        if t1 <= 0.0 || t0 >= max_t || t0 >= *entry_t {
            continue;
        }
        *entry_t = t0;
        best = i;
    }
    return best;
}

// `trace_ray`, following portal teleports and black-hole bends (≤ 6 hops):
// origin/direction are updated in place so the caller's ray state matches
// the returned intersection. Accretion-disk radiance accumulates into
// `*emitted` (scale by throughput and add); `*captured` is set when the ray
// fell into a horizon — the caller must show BLACK, not the sky, and ignore
// the returned (stale) intersection. Structured with `let`s only — naga
// rejects re-assigning a `var` of the special RayIntersection type.
fn trace_ray_traversal(
    ray_origin: ptr<function, vec3<f32>>,
    ray_direction: ptr<function, vec3<f32>>,
    ray_t_min: f32,
    ray_flag: u32,
    emitted: ptr<function, vec3<f32>>,
    captured: ptr<function, u32>,
) -> RayIntersection {
    *captured = 0u;
    for (var hop = 0u; hop < 6u; hop += 1u) {
        let ray = trace_ray(
            *ray_origin,
            *ray_direction,
            select(0.0, ray_t_min, hop == 0u),
            RAY_T_MAX,
            ray_flag,
        );
        let scene_t = select(1e30, ray.t, ray.kind != RAY_QUERY_INTERSECTION_NONE);

        // A black-hole region before the scene hit bends the ray first.
        var entry_t = scene_t;
        let hole = nearest_black_hole_entry(*ray_origin, *ray_direction, scene_t, &entry_t);
        if hole != 0xFFFFFFFFu {
            *ray_origin += *ray_direction * entry_t;
            if black_hole_march(hole, ray_origin, ray_direction, emitted) {
                *captured = 1u;
                return ray;
            }
            continue; // re-trace from the region exit
        }

        if ray.kind == RAY_QUERY_INTERSECTION_NONE
            || !portal_redirect(ray.instance_index, *ray_origin + *ray_direction * ray.t, ray_origin, ray_direction) {
            return ray;
        }
    }
    return trace_ray(*ray_origin, *ray_direction, 0.0, RAY_T_MAX, ray_flag);
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
    let material_id = material_ids[instance_id];
    let material = materials[material_id];

    let transform = transforms[instance_id];
    let previous_frame_transform = previous_frame_transforms[instance_id];

    let cluster = clusters[cluster_global_id];
    // `cluster_indices` stores per-cluster LOCAL vertex indices
    // (0..vertex_count) — the bake widens meshopt's u8 locals to
    // u32 but does NOT rebase them. The matching cluster CLAS was
    // built with `vertex_buffer = vertex_buffer_addr + vertex_offset
    // * 12`, so the driver applies the same offset internally.
    // Reconstruct the global vertex-pool slot here: add the
    // cluster's (already globally-rebased) `vertex_offset`.
    let idx_base = cluster.index_offset + triangle_id * 3u;
    let anim = instance_animated[instance_id];
    let vertices = array<Vertex, 3>(
        load_cluster_vertex_animated(cluster.vertex_offset + cluster_indices[idx_base + 0u], anim),
        load_cluster_vertex_animated(cluster.vertex_offset + cluster_indices[idx_base + 1u], anim),
        load_cluster_vertex_animated(cluster.vertex_offset + cluster_indices[idx_base + 2u], anim),
    );

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

    if material.normal_map_texture_id != TEXTURE_MAP_NONE {
        let TBN = calculate_tbn_mikktspace(world_normal, world_tangent);
        let T = TBN[0];
        let B = TBN[1];
        let N = TBN[2];
        let Nt = sample_texture(material.normal_map_texture_id, uv);
        world_normal = normalize(Nt.x * T + Nt.y * B + Nt.z * N);
    }

    let resolved_material = resolve_material(material, uv);

    return ResolvedRayHitFull(
        world_position,
        previous_frame_world_position,
        world_normal,
        geometric_world_normal,
        world_tangent,
        uv,
        triangle_area,
        cluster.triangle_count,
        resolved_material,
        material_id,
    );
}
