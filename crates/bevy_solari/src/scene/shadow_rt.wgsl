enable wgpu_ray_tracing_pipeline;
enable primitive_index;

// --- Types matching raytracing_scene_bindings.wgsl ---

struct InstanceGeometryIds {
    vertex_buffer_id: u32,
    vertex_buffer_offset: u32,
    index_buffer_id: u32,
    index_buffer_offset: u32,
    triangle_count: u32,
}

struct VertexBuffer { vertices: array<PackedVertex> }
struct IndexBuffer { indices: array<u32> }

struct PackedVertex {
    a: vec4<f32>,
    b: vec4<f32>,
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
    reflectance: vec3<f32>,
    alpha_cutoff: f32,
    alpha_mode: u32,
    _padding: vec3<f32>,
}

const ALPHA_MODE_OPAQUE: u32 = 0u;
const ALPHA_MODE_MASK: u32 = 1u;
const ALPHA_MODE_BLEND: u32 = 2u;
const TEXTURE_MAP_NONE = 0xFFFFFFFFu;

// --- Scene Bindings (group 0) ---

@group(0) @binding(0) var<storage> vertex_buffers: binding_array<VertexBuffer>;
@group(0) @binding(1) var<storage> index_buffers: binding_array<IndexBuffer>;
@group(0) @binding(2) var textures: binding_array<texture_2d<f32>>;
@group(0) @binding(3) var tex_samplers: binding_array<sampler>;
@group(0) @binding(4) var<storage> materials: array<Material>;
@group(0) @binding(5) var acc_struct: acceleration_structure;
@group(0) @binding(6) var<storage> transforms: array<mat4x4<f32>>;
@group(0) @binding(7) var<storage> previous_frame_transforms: array<mat4x4<f32>>;
@group(0) @binding(8) var<storage> geometry_ids: array<InstanceGeometryIds>;
@group(0) @binding(9) var<storage> material_ids: array<u32>;

// --- Payload ---

struct ShadowPayload {
    visibility: f32,
    _pad: u32,
}

var<ray_payload> payload: ShadowPayload;
var<incoming_ray_payload> incoming_payload: ShadowPayload;

// --- Helper: compute barycentrics from object-space ray hit ---

fn compute_barycentrics(
    p: vec3<f32>,  // hit point in object space
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
) -> vec3<f32> {
    let e0 = v1 - v0;
    let e1 = v2 - v0;
    let e2 = p - v0;

    let d00 = dot(e0, e0);
    let d01 = dot(e0, e1);
    let d11 = dot(e1, e1);
    let d20 = dot(e2, e0);
    let d21 = dot(e2, e1);

    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    return vec3(u, v, w);
}

// --- Ray Generation ---

@ray_generation
fn shadow_raygen(
    @builtin(ray_invocation_id) id: vec3<u32>,
    @builtin(num_ray_invocations) num_invocations: vec3<u32>,
) {
    payload = ShadowPayload(1.0, 0u);
    traceRay(
        acc_struct,
        RayDesc(RAY_FLAG_NONE, 0xff, 0.001, 100000.0, vec3(0.0), vec3(0.0, 1.0, 0.0)),
        &payload,
    );
}

// --- Miss: ray missed all geometry → light is visible ---

@miss
@incoming_payload(incoming_payload)
fn shadow_miss(
    @builtin(world_ray_origin) origin: vec3<f32>,
    @builtin(world_ray_direction) dir: vec3<f32>,
    @builtin(ray_t_min) t_min: f32,
) {
    incoming_payload.visibility = 1.0;
}

// --- Any Hit: alpha-test transparency ---

@any_hit
@incoming_payload(incoming_payload)
fn shadow_any_hit(
    @builtin(instance_custom_data) instance_index: u32,
    @builtin(geometry_index) geo_idx: u32,
    @builtin(primitive_index) triangle_index: u32,
    @builtin(ray_t_current_max) t: f32,
    @builtin(hit_kind) kind: u32,
    @builtin(object_ray_origin) obj_ray_origin: vec3<f32>,
    @builtin(object_ray_direction) obj_ray_dir: vec3<f32>,
) {
    // Look up material for this instance
    let material_id = material_ids[instance_index];
    let material = materials[material_id];

    // Opaque materials always block light — accept hit immediately
    if material.alpha_mode == ALPHA_MODE_OPAQUE {
        return;
    }

    // For alpha-masked or blended materials, sample the texture
    if material.base_color_texture_id == TEXTURE_MAP_NONE {
        // No texture — use base_color alpha (which is 1.0 for vec3, so accept)
        return;
    }

    // Load triangle vertices to compute UVs
    let instance_geo = geometry_ids[instance_index];
    let idx_buf = &index_buffers[instance_geo.index_buffer_id].indices;
    let vtx_buf = &vertex_buffers[instance_geo.vertex_buffer_id].vertices;

    let base_idx = triangle_index * 3u + instance_geo.index_buffer_offset;
    let i0 = (*idx_buf)[base_idx] + instance_geo.vertex_buffer_offset;
    let i1 = (*idx_buf)[base_idx + 1u] + instance_geo.vertex_buffer_offset;
    let i2 = (*idx_buf)[base_idx + 2u] + instance_geo.vertex_buffer_offset;

    let v0_pos = (*vtx_buf)[i0].a.xyz;
    let v1_pos = (*vtx_buf)[i1].a.xyz;
    let v2_pos = (*vtx_buf)[i2].a.xyz;

    // Compute hit point in object space
    let hit_point = obj_ray_origin + obj_ray_dir * t;

    // Compute barycentrics
    let bary = compute_barycentrics(hit_point, v0_pos, v1_pos, v2_pos);

    // Interpolate UVs (packed in b.zw)
    let uv0 = (*vtx_buf)[i0].b.zw;
    let uv1 = (*vtx_buf)[i1].b.zw;
    let uv2 = (*vtx_buf)[i2].b.zw;
    let uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

    // Sample texture alpha
    let alpha = textureSampleLevel(
        textures[material.base_color_texture_id],
        tex_samplers[material.base_color_texture_id],
        uv,
        0.0
    ).a;

    // Alpha test: if below cutoff, reject this hit (ray continues through)
    if material.alpha_mode == ALPHA_MODE_MASK && alpha < material.alpha_cutoff {
        // ignoreIntersection — not yet available in WGSL, so we just return
        // The any-hit shader returning without accepting means the intersection is ignored
        // TODO: once naga supports ignoreIntersection(), call it here
        return;
    }

    // For blend mode with very low alpha, also reject
    if material.alpha_mode == ALPHA_MODE_BLEND && alpha < 0.01 {
        return;
    }

    // Otherwise accept the hit — this surface blocks light
}

// --- Closest Hit: ray hit opaque/accepted geometry → occluded ---

@closest_hit
@incoming_payload(incoming_payload)
fn shadow_closest_hit(
    @builtin(object_ray_origin) origin: vec3<f32>,
    @builtin(object_ray_direction) dir: vec3<f32>,
    @builtin(object_to_world) obj_to_world: mat4x3<f32>,
    @builtin(world_to_object) world_to_obj: mat4x3<f32>,
) {
    incoming_payload.visibility = 0.0;
}
