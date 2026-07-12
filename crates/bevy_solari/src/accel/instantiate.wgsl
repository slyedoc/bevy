// Instantiate animated CLAS templates with this frame's deformed vertices.
//
// One thread per active animated slot. For each of the slot's finest-LOD
// clusters (`lod_level == 0`, a complete surface cover) it emits a
// `VkClusterAccelerationStructureInstantiateClusterInfoNV` record pointing the
// cluster's topology-only template at its deformed-position slice in the deform
// pool. The raw-VK INSTANTIATE build (op `INSTANTIATE_TRIANGLE_CLUSTER`) then
// produces a fresh per-instance CLAS per record. Each slot also emits one
// `BuildClustersBottomLevelInfoNV` arg whose cluster-reference list is the
// contiguous block of CLAS addresses the INSTANTIATE build writes — consumed by
// the per-instance BLAS build. See `accel/animated_blas.rs`.

// Mirror of `deform.rs::AnimatedSlotGpu`.
struct AnimatedSlot {
    instance_slot: u32,
    mesh_vertex_base: u32,
    mesh_vertex_count: u32,
    deform_pool_base: u32,
    joint_count: u32,
    palette_base: u32,
    inverse_bind_base: u32,
    joint_base: u32,
    node_slot: u32,
}

// Mirror of the shared cluster pool record (48 B) — see raytracing_scene_bindings.
struct Cluster {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    triangle_count: u32,
    bounds_sphere: vec4<f32>,
    local_material_id: u32,
    lod_level: u32,
    _pad0: u32,
    _pad1: u32,
}

// Mirror of `InstanceLodInputGpu` (16 B).
struct InstanceLodInput {
    cluster_base: u32,
    cluster_count: u32,
    group_base: u32,
    root_group: u32,
}

// Mirror of `cluster_bindings.wgsl::ClusterLodGroup` (48 B).
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

const PARENT_GROUP_NONE: u32 = 0xFFFFFFFFu;

struct InstantiateParams {
    num_slots: u32,
    blas_stride: u32,
    // Device address of the deform position pool (vec2<u32> little-endian u64).
    deform_positions_addr: vec2<u32>,
    // Device address of the instantiated-CLAS address table.
    instantiated_clas_addrs_addr: vec2<u32>,
    // Base device address of the per-instance animated BLAS pool. This instance's
    // BLAS lives at `blas_pool_base + slot_idx * blas_stride`.
    blas_pool_base: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> active_slots: array<AnimatedSlot>;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
// Global per-cluster template device address (vec2<u32>), indexed by global cluster id.
@group(0) @binding(2) var<storage, read> cluster_template_addresses: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> instance_lod_inputs: array<InstanceLodInput>;
@group(0) @binding(4) var<uniform> params: InstantiateParams;
// Output: InstantiateClusterInfoNV records (8 u32 = 32 B each).
@group(0) @binding(5) var<storage, read_write> instantiate_args: array<u32>;
// GPU record counter (INSTANTIATE build `srcInfosCount`).
@group(0) @binding(6) var<storage, read_write> count: array<atomic<u32>>;
// Output: per-slot BuildClustersBottomLevelInfoNV args (4 u32 = 16 B each).
@group(0) @binding(7) var<storage, read_write> blas_args: array<u32>;
// slot-indexed instance → BLAS device address (PTLAS reads this). Owned by
// `blas_sharing`; `assign_address` (Classify) wrote the static shared address —
// we run AFTER it (BuildAnimatedBlas stage) and overwrite for animated instances.
@group(0) @binding(8) var<storage, read_write> instance_blas_address: array<vec2<u32>>;
// LOD selection (reused from the static path — the cut is pose-independent):
// per-instance object-space error budget (written by `blas_sharing::classify`) +
// the cluster DAG (groups + cluster→group). The cut picks the same clusters a
// static instance of this geometry+band would get; we instantiate THEIR templates.
@group(0) @binding(9) var<storage, read> instance_e_build: array<f32>;
@group(0) @binding(10) var<storage, read> cluster_groups: array<ClusterLodGroup>;
@group(0) @binding(11) var<storage, read> cluster_to_group: array<u32>;

// base (u64 as vec2<u32>) + offset (u32) with carry.
fn u64_add(base: vec2<u32>, off: u32) -> vec2<u32> {
    let lo = base.x + off;
    let carry = select(0u, 1u, lo < base.x);
    return vec2<u32>(lo, base.y + carry);
}

// Object-space DAG cut — identical accept rule to `selector.wgsl::select_main`:
// emit the coarsest group whose own error fits and whose parent's does not;
// finest level is the fallback. Single-coverage (each surface region once).
fn cluster_accepted(global_cluster: u32, group_base: u32, e_build: f32) -> bool {
    let group = cluster_groups[cluster_to_group[global_cluster]];
    let own_fits = group.max_quadric_error <= e_build;
    var parent_fits = false;
    if group.parent_group != PARENT_GROUP_NONE {
        let parent = cluster_groups[group_base + group.parent_group];
        // A parent only represents a valid coarser cut if it's STRICTLY coarser.
        // The bake's back-fill (`from_mesh.rs`) gives single-LOD meshes a same-LOD
        // "parent" (no coarser level exists); treating that as a real parent would
        // cull every group but the root. Requiring a coarser level is a no-op for
        // real multi-LOD DAGs (their parents are always coarser).
        if parent.lod_level > group.lod_level {
            parent_fits = parent.max_quadric_error <= e_build;
        }
    }
    let is_finest = clusters[global_cluster].lod_level == 0u;
    return (own_fits && !parent_fits) || (is_finest && !own_fits);
}

@compute @workgroup_size(1)
fn instantiate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot_idx = gid.x;
    if slot_idx >= params.num_slots {
        return;
    }
    let s = active_slots[slot_idx];
    let lod = instance_lod_inputs[s.instance_slot];
    let e_build = instance_e_build[s.instance_slot];

    // Count the DAG-cut clusters for this instance's LOD, then reserve a block.
    var n = 0u;
    for (var c = 0u; c < lod.cluster_count; c++) {
        if cluster_accepted(lod.cluster_base + c, lod.group_base, e_build) {
            n += 1u;
        }
    }
    let block_base = atomicAdd(&count[0], n);

    // Emit one InstantiateClusterInfoNV per selected cluster.
    var w = 0u;
    for (var c = 0u; c < lod.cluster_count; c++) {
        let global_cluster = lod.cluster_base + c;
        let cl = clusters[global_cluster];
        if !cluster_accepted(global_cluster, lod.group_base, e_build) {
            continue;
        }
        let template_addr = cluster_template_addresses[global_cluster];
        // Deformed position of this cluster's first vertex: the deform pool is
        // per-slot, indexed mesh-local (cluster.vertex_offset is global).
        let local_vertex = cl.vertex_offset - s.mesh_vertex_base;
        let deform_vertex = s.deform_pool_base + local_vertex;
        let vaddr = u64_add(params.deform_positions_addr, deform_vertex * 12u);

        let rec = block_base + w;
        let b = rec * 8u;
        instantiate_args[b + 0u] = 0u; // cluster_id_offset
        instantiate_args[b + 1u] = 0u; // geometry_index_offset_and_reserved (keep template's baked id)
        instantiate_args[b + 2u] = template_addr.x; // cluster_template_address lo
        instantiate_args[b + 3u] = template_addr.y; // hi
        instantiate_args[b + 4u] = vaddr.x; // vertex_buffer.address lo
        instantiate_args[b + 5u] = vaddr.y; // hi
        instantiate_args[b + 6u] = 12u; // vertex_buffer.stride lo
        instantiate_args[b + 7u] = 0u; // hi
        w += 1u;
    }

    // Per-slot BLAS arg: cluster refs = the contiguous CLAS-address block the
    // INSTANTIATE build writes at instantiated_clas_addrs[block_base..].
    let refs_addr = u64_add(params.instantiated_clas_addrs_addr, block_base * 8u);
    let ba = slot_idx * 4u;
    blas_args[ba + 0u] = n; // cluster_references_count
    blas_args[ba + 1u] = 8u; // cluster_references_stride
    blas_args[ba + 2u] = refs_addr.x;
    blas_args[ba + 3u] = refs_addr.y;

    // Repoint this instance at its per-instance animated BLAS (stable address
    // `blas_pool_base + slot_idx * blas_stride`, where the EXPLICIT BLAS build
    // writes it). Overwrites the static shared-BLAS address from `assign_address`.
    let blas_addr = u64_add(params.blas_pool_base, slot_idx * params.blas_stride);
    instance_blas_address[s.instance_slot] = blas_addr;
}
