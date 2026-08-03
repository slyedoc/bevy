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
    uint instance_slot;
    uint mesh_vertex_base;
    uint mesh_vertex_count;
    uint deform_pool_base;
    uint joint_count;
    uint palette_base;
    uint inverse_bind_base;
    uint joint_base;
    uint node_slot;
}

// Mirror of the shared cluster pool record (48 B) — see raytracing_scene_bindings.
struct Cluster {
    uint vertex_offset;
    uint vertex_count;
    uint index_offset;
    uint triangle_count;
    float4 bounds_sphere;
    uint local_material_id;
    uint lod_level;
    uint _pad0;
    uint _pad1;
}

// Mirror of `InstanceLodInputGpu` (16 B).
struct InstanceLodInput {
    uint cluster_base;
    uint cluster_count;
    uint group_base;
    uint root_group;
}

// Mirror of `cluster_bindings.slang::ClusterLodGroup` (48 B).
struct ClusterLodGroup {
    uint cluster_start;
    uint cluster_count;
    uint children_offset;
    uint children_count;
    float4 traversal_sphere;
    float max_quadric_error;
    uint parent_group;
    uint lod_level;
    uint _pad;
}

static const uint PARENT_GROUP_NONE = 0xFFFFFFFF;

struct InstantiateParams {
    uint num_slots;
    uint blas_stride;
    // Device address of the deform position pool (uint2 little-endian u64).
    uint2 deform_positions_addr;
    // Device address of the instantiated-CLAS address table.
    uint2 instantiated_clas_addrs_addr;
    // Base device address of the per-instance animated BLAS pool. This instance's
    // BLAS lives at `blas_pool_base + slot_idx * blas_stride`.
    uint2 blas_pool_base;
}

[[vk::push_constant]] InstantiateParams params;

[[vk::binding(0, 0)]] StructuredBuffer<AnimatedSlot> active_slots;
[[vk::binding(1, 0)]] StructuredBuffer<Cluster> clusters;
// Global per-cluster template device address (uint2), indexed by global cluster id.
[[vk::binding(2, 0)]] StructuredBuffer<uint2> cluster_template_addresses;
[[vk::binding(3, 0)]] StructuredBuffer<InstanceLodInput> instance_lod_inputs;
// Output: InstantiateClusterInfoNV records (8 u32 = 32 B each).
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> instantiate_args;
// GPU record counter (INSTANTIATE build `srcInfosCount`).
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> count;
// Output: per-slot BuildClustersBottomLevelInfoNV args (4 u32 = 16 B each).
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> blas_args;
// slot-indexed instance → BLAS device address (PTLAS reads this). Owned by
// `blas_sharing`; `assign_address` (Classify) wrote the static shared address —
// we run AFTER it (BuildAnimatedBlas stage) and overwrite for animated instances.
[[vk::binding(7, 0)]] RWStructuredBuffer<uint2> instance_blas_address;
// LOD selection (reused from the static path — the cut is pose-independent):
// per-instance object-space error budget (written by `blas_sharing::classify`) +
// the cluster DAG (groups + cluster→group). The cut picks the same clusters a
// static instance of this geometry+band would get; we instantiate THEIR templates.
[[vk::binding(8, 0)]] StructuredBuffer<float> instance_e_build;
[[vk::binding(9, 0)]] StructuredBuffer<ClusterLodGroup> cluster_groups;
[[vk::binding(10, 0)]] StructuredBuffer<uint> cluster_to_group;

// base (u64 as uint2) + offset (u32) with carry.
uint2 u64_add(uint2 base, uint off) {
    let lo = base.x + off;
    let carry = lo < base.x ? 1u : 0u;
    return uint2(lo, base.y + carry);
}

// Object-space DAG cut — identical accept rule to `selector.slang::select_main`:
// emit the coarsest group whose own error fits and whose parent's does not;
// finest level is the fallback. Single-coverage (each surface region once).
bool cluster_accepted(uint global_cluster, uint group_base, float e_build) {
    let group = cluster_groups[cluster_to_group[global_cluster]];
    let own_fits = group.max_quadric_error <= e_build;
    var parent_fits = false;
    if (group.parent_group != PARENT_GROUP_NONE) {
        let parent = cluster_groups[group_base + group.parent_group];
        // A parent only represents a valid coarser cut if it's STRICTLY coarser.
        // The bake's back-fill (`from_mesh.rs`) gives single-LOD meshes a same-LOD
        // "parent" (no coarser level exists); treating that as a real parent would
        // cull every group but the root. Requiring a coarser level is a no-op for
        // real multi-LOD DAGs (their parents are always coarser).
        if (parent.lod_level > group.lod_level) {
            parent_fits = parent.max_quadric_error <= e_build;
        }
    }
    let is_finest = clusters[global_cluster].lod_level == 0;
    return (own_fits && !parent_fits) || (is_finest && !own_fits);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void instantiate(uint3 gid: SV_DispatchThreadID) {
    let slot_idx = gid.x;
    if (slot_idx >= params.num_slots) {
        return;
    }
    let s = active_slots[slot_idx];
    let lod = instance_lod_inputs[s.instance_slot];
    let e_build = instance_e_build[s.instance_slot];

    // Count the DAG-cut clusters for this instance's LOD, then reserve a block.
    var n = 0u;
    for (var c = 0u; c < lod.cluster_count; c++) {
        if (cluster_accepted(lod.cluster_base + c, lod.group_base, e_build)) {
            n += 1;
        }
    }
    uint block_base;
    InterlockedAdd(count[0], n, block_base);

    // Emit one InstantiateClusterInfoNV per selected cluster.
    var w = 0u;
    for (var c = 0u; c < lod.cluster_count; c++) {
        let global_cluster = lod.cluster_base + c;
        let cl = clusters[global_cluster];
        if (!cluster_accepted(global_cluster, lod.group_base, e_build)) {
            continue;
        }
        let template_addr = cluster_template_addresses[global_cluster];
        // Deformed position of this cluster's first vertex: the deform pool is
        // per-slot, indexed mesh-local (cluster.vertex_offset is global).
        let local_vertex = cl.vertex_offset - s.mesh_vertex_base;
        let deform_vertex = s.deform_pool_base + local_vertex;
        let vaddr = u64_add(params.deform_positions_addr, deform_vertex * 12);

        let rec = block_base + w;
        let b = rec * 8;
        instantiate_args[b + 0] = 0; // cluster_id_offset
        // geometry_index_offset (24 bits) + reserved(8)=0: templates bake a
        // MESH-LOCAL geometry index (a global one blows up their encoding — see
        // `clas_template`), so re-add the mesh's cluster base here to land the
        // global id the inline ray-query path reads back.
        instantiate_args[b + 1] = lod.cluster_base & 0x00FFFFFF;
        instantiate_args[b + 2] = template_addr.x; // cluster_template_address lo
        instantiate_args[b + 3] = template_addr.y; // hi
        instantiate_args[b + 4] = vaddr.x; // vertex_buffer.address lo
        instantiate_args[b + 5] = vaddr.y; // hi
        instantiate_args[b + 6] = 12; // vertex_buffer.stride lo
        instantiate_args[b + 7] = 0; // hi
        w += 1;
    }

    // Per-slot BLAS arg: cluster refs = the contiguous CLAS-address block the
    // INSTANTIATE build writes at instantiated_clas_addrs[block_base..].
    let refs_addr = u64_add(params.instantiated_clas_addrs_addr, block_base * 8);
    let ba = slot_idx * 4;
    blas_args[ba + 0] = n; // cluster_references_count
    blas_args[ba + 1] = 8; // cluster_references_stride
    blas_args[ba + 2] = refs_addr.x;
    blas_args[ba + 3] = refs_addr.y;

    // Repoint this instance at its per-instance animated BLAS (stable address
    // `blas_pool_base + slot_idx * blas_stride`, where the EXPLICIT BLAS build
    // writes it). Overwrites the static shared-BLAS address from `assign_address`.
    let blas_addr = u64_add(params.blas_pool_base, slot_idx * params.blas_stride);
    instance_blas_address[s.instance_slot] = blas_addr;
}
