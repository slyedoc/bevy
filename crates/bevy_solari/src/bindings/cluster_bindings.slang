// The shared cluster-scene set-0 surface — the Slang mirror of the
// `ClusterSceneBindGroup` heap mirror (`bindings/bind_groups.rs`): the
// mesh-pool buffers plus the per-instance columns, mapped at constant heap
// offsets via `gpu::rt_pipeline::cluster_heap_mappings`. Imported by the
// cluster AS kernels (selector / BLAS sharing / PTLAS fill).
//
// Not compiled standalone: `import cluster_bindings;` from an entry file.
module cluster_bindings;

// On-disk cluster — mirrors `bevy_solari::cluster::asset::Cluster`
// (std430 stride 48: the vec4 bounds forces 16-alignment, pads explicit).
// `vertex_offset` / `index_offset` are pool slots **after** the
// CPU-side rebase done by `ClusterMeshManager::queue_upload_if_needed`.
public struct Cluster {
    public uint vertex_offset;
    public uint vertex_count;
    public uint index_offset;
    public uint triangle_count;
    public float4 bounds_sphere;
    public uint local_material_id;
    public uint lod_level;
    uint _pad0;
    uint _pad1;
}

// DAG-cut unit — mirrors `bevy_solari::cluster::asset::ClusterLodGroup`
// (std430 stride 48).
public struct ClusterLodGroup {
    public uint cluster_start;
    public uint cluster_count;
    public uint children_offset;
    public uint children_count;
    public float4 traversal_sphere;
    public float max_quadric_error;
    public uint parent_group;
    public uint lod_level;
    uint _pad;
}

// Interior DAG node — mirrors `bevy_solari::cluster::asset::ClusterBvhNode`
// (std430 stride 32).
public struct ClusterBvhNode {
    public float4 traversal_sphere;
    public float max_quadric_error;
    public uint children_offset;
    public uint children_packed;
    uint _pad;
}

public uint cluster_bvh_node_children_count(ClusterBvhNode n) {
    return n.children_packed & 0xFFFF;
}

public bool cluster_bvh_node_is_group_leaf(ClusterBvhNode n) {
    return ((n.children_packed >> 16) & 1) == 1;
}

// ----------------------------------------------------------------------
// The cluster-scene set 0 — one shared surface for cluster scene state,
// binding numbers matching `ClusterSceneBindGroup`'s layout.
// ----------------------------------------------------------------------

// Mesh-pool buffers (persistent across frames).
[[vk::binding(0, 0)]] public StructuredBuffer<float> vertex_positions; // stride 3 floats per vertex
[[vk::binding(1, 0)]] public StructuredBuffer<uint> vertex_normals;    // octahedral 2x16snorm
[[vk::binding(2, 0)]] public StructuredBuffer<float4> vertex_tangents;
[[vk::binding(3, 0)]] public StructuredBuffer<float2> vertex_uvs;
[[vk::binding(4, 0)]] public StructuredBuffer<uint> cluster_indices;
[[vk::binding(5, 0)]] public StructuredBuffer<uint> child_table;
[[vk::binding(6, 0)]] public StructuredBuffer<Cluster> clusters;
[[vk::binding(7, 0)]] public StructuredBuffer<ClusterLodGroup> cluster_groups;
[[vk::binding(8, 0)]] public StructuredBuffer<ClusterBvhNode> cluster_nodes;
[[vk::binding(9, 0)]] public StructuredBuffer<uint> cluster_to_group;

// Per-instance world transforms — slot-indexed `mat3x4` affine stored as
// 3 `float4` rows per slot (row k = the standard 4x4's row k),
// GPU-scattered from a CPU delta by `cluster::gpu_instances`. Read through
// [`cluster_instance_transform`].
[[vk::binding(10, 0)]] public StructuredBuffer<float4> cluster_instance_transforms;
[[vk::binding(11, 0)]] public StructuredBuffer<uint> cluster_instance_group_bases;

// Packed per-instance LOD inputs (slot-indexed). Mirrors
// `cluster::instance_manager::InstanceLodInputGpu`. `cluster_base` /
// `cluster_count` drive the selector; `group_base` (geometry key) +
// `root_group` (global root-group index) drive the BLAS-sharing
// classify pass.
public struct InstanceLodInput {
    public uint cluster_base;
    public uint cluster_count;
    public uint group_base;
    public uint root_group;
}

[[vk::binding(12, 0)]] public StructuredBuffer<InstanceLodInput> cluster_instance_lod_inputs;

// ----------------------------------------------------------------------
// Accessor helpers.
// ----------------------------------------------------------------------

/// The slot's affine world transform as a `float3x4`: row `k` is the
/// standard 4x4's row `k` (`.xyz` = linear row, `.w` = translation).
public float3x4 cluster_instance_transform(uint slot) {
    let base = slot * 3;
    return float3x4(
        cluster_instance_transforms[base],
        cluster_instance_transforms[base + 1],
        cluster_instance_transforms[base + 2],
    );
}

// Vertex positions are stored as a packed `array<f32>` with stride 3
// (12 B per vertex) — a `StructuredBuffer<float3>` would imply 16 B
// stride on some layouts, which mismatches the on-disk layout.
public float3 get_vertex_position(uint vertex_index) {
    let base = vertex_index * 3;
    return float3(
        vertex_positions[base],
        vertex_positions[base + 1],
        vertex_positions[base + 2],
    );
}

// WGSL `unpack2x16snorm`: two 16-bit signed normalized values from one u32.
float2 unpack2x16snorm(uint v) {
    let sx = int(v << 16) >> 16;
    let sy = int(v) >> 16;
    return clamp(float2(sx, sy) / 32767.0, -1.0, 1.0);
}

// `bevy_render::utils::octahedral_decode_signed` — decode a signed
// octahedral-packed direction back to a unit vector.
float3 octahedral_decode_signed(float2 v) {
    float3 n = float3(v, 1.0 - abs(v.x) - abs(v.y));
    let t = saturate(-n.z);
    let w = float2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    n = float3(n.xy + w, n.z);
    return normalize(n);
}

// Decode an octahedral-packed normal back to a unit `float3`.
public float3 get_vertex_normal(uint vertex_index) {
    return octahedral_decode_signed(unpack2x16snorm(vertex_normals[vertex_index]));
}
