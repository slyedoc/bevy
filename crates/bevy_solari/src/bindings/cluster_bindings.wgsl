#define_import_path bevy_solari::cluster_bindings

#import bevy_render::utils::octahedral_decode_signed

// On-disk cluster — mirrors `bevy_solari::cluster::asset::Cluster`.
// `vertex_offset` / `index_offset` are pool slots **after** the
// CPU-side rebase done by `ClusterMeshManager::queue_upload_if_needed`.
struct Cluster {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    triangle_count: u32,
    bounds_sphere: vec4<f32>,
    local_material_id: u32,
    lod_level: u32,
}

// DAG-cut unit — mirrors `bevy_solari::cluster::asset::ClusterLodGroup`.
struct ClusterLodGroup {
    cluster_start: u32,
    cluster_count: u32,
    children_offset: u32,
    children_count: u32,
    traversal_sphere: vec4<f32>,
    max_quadric_error: f32,
    parent_group: u32,
    lod_level: u32,
    _pad: u32,
}

// Interior DAG node — mirrors `bevy_solari::cluster::asset::ClusterBvhNode`.
struct ClusterBvhNode {
    traversal_sphere: vec4<f32>,
    max_quadric_error: f32,
    children_offset: u32,
    children_packed: u32,
    _pad: u32,
}

fn cluster_bvh_node_children_count(n: ClusterBvhNode) -> u32 {
    return n.children_packed & 0xFFFFu;
}

fn cluster_bvh_node_is_group_leaf(n: ClusterBvhNode) -> bool {
    return ((n.children_packed >> 16u) & 1u) == 1u;
}

// ----------------------------------------------------------------------
// Bind group layout: one shared group for cluster scene state.
// Consumer shaders should bind a layout that matches this ordering.
// ----------------------------------------------------------------------

// Mesh-pool buffers (persistent across frames).
@group(0) @binding(0) var<storage, read> vertex_positions: array<f32>; // stride 3 floats per vertex
@group(0) @binding(1) var<storage, read> vertex_normals: array<u32>;   // octahedral 2x16snorm
@group(0) @binding(2) var<storage, read> vertex_tangents: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> vertex_uvs: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> cluster_indices: array<u32>;
@group(0) @binding(5) var<storage, read> child_table: array<u32>;
@group(0) @binding(6) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(7) var<storage, read> cluster_groups: array<ClusterLodGroup>;
@group(0) @binding(8) var<storage, read> cluster_nodes: array<ClusterBvhNode>;
@group(0) @binding(9) var<storage, read> cluster_to_group: array<u32>;

// Per-instance world transforms — slot-indexed `mat3x4` affine
// (column k = 4x4 row k), GPU-scattered from a CPU delta by
// `cluster::gpu_instances`. Replaces the old full `array<Mesh>`
// re-upload.
@group(0) @binding(10) var<storage, read> cluster_instance_transforms: array<mat3x4<f32>>;
@group(0) @binding(11) var<storage, read> cluster_instance_group_bases: array<u32>;

// Packed per-instance LOD inputs (slot-indexed). Mirrors
// `cluster::instance_manager::InstanceLodInputGpu`. `cluster_base` /
// `cluster_count` drive the selector; `group_base` (geometry key) +
// `root_group` (global root-group index) drive the BLAS-sharing
// classify pass.
struct InstanceLodInput {
    cluster_base: u32,
    cluster_count: u32,
    group_base: u32,
    root_group: u32,
}

@group(0) @binding(12) var<storage, read> cluster_instance_lod_inputs: array<InstanceLodInput>;

// ----------------------------------------------------------------------
// Accessor helpers.
// ----------------------------------------------------------------------

// Vertex positions are stored as a packed `array<f32>` with stride 3
// (12 B per vertex). Reading as `array<vec3<f32>>` would imply 16 B
// stride per WGSL's storage rules, which mismatches the on-disk layout.
fn get_vertex_position(vertex_index: u32) -> vec3<f32> {
    let base = vertex_index * 3u;
    return vec3<f32>(
        vertex_positions[base],
        vertex_positions[base + 1u],
        vertex_positions[base + 2u],
    );
}

// Decode an octahedral-packed normal back to a unit `vec3<f32>`.
fn get_vertex_normal(vertex_index: u32) -> vec3<f32> {
    return octahedral_decode_signed(unpack2x16snorm(vertex_normals[vertex_index]));
}
