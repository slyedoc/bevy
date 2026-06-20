// Fills the partitioned-AS WRITE record buffer consumed by NV's
// `vkCmdBuildPartitionedAccelerationStructuresNV`, then GPU-writes the
// build's instance count. Three passes:
//
//   fill_seed        — one thread per CPU-seeded delta record (added ∪
//                      moved ∪ disabled). Writes `write_data[i]`.
//   fill_incremental — one thread per ACTIVE instance. Appends a WRITE
//                      record for any instance that (a) MOVED this frame —
//                      its gathered world transform differs from last
//                      frame's (the gather shifts current→previous before
//                      this pass) — or (b) whose GEOMETRY was rebuilt this
//                      frame (`geometry_dirty`), its shared BLAS content
//                      changed in place. (a) is how the PTLAS learns which
//                      instances moved now that the CPU no longer tracks
//                      moves; comparing actual world transforms also catches
//                      parent-driven moves the CPU `Changed<Transform>` path
//                      missed. `force_all` writes every active instance (full
//                      rebuild). Static frame → nothing moved/dirty → pure CPU
//                      delta (added ∪ disabled).
//   finalize         — one thread. Writes the live record count into the
//                      build op's `arg_count` field (GPU-driven count).
//
// Each instance carries a STABLE PTLAS `instance_index == GpuEntity`,
// so the driver carries untouched instances across from the previous
// PTLAS (`src`). A BLAS address of 0 removes the instance (disabled).
//
// CPU-seeded records [0, cpu_count) and GPU-appended records
// [cpu_count, count) occupy disjoint regions of `write_data`; the GPU
// atomic counter is seeded to `cpu_count` CPU-side. Duplicate writes for
// the same instance (e.g. a moved instance that also band-crossed) are
// idempotent — same `instance_index`, same record.

#import bevy_solari::cluster_bindings::cluster_instance_transforms
#import bevy_solari::instance_mask::{instance_hardware_mask, material_vk_flags, material_hit_group}

/// Mirror of `VkPartitionedAccelerationStructureWriteInstanceDataNV`
/// (104 B). Field order + size must match the Rust side byte-for-byte.
struct WriteInstanceData {
    transform: array<f32, 12>,
    explicit_aabb: array<f32, 6>,
    instance_id: u32,
    instance_mask: u32,
    instance_contribution_to_hit_group_index: u32,
    instance_flags: u32,
    instance_index: u32,
    partition_index: u32,
    /// BLAS device address as `vec2<u32>` (little-endian VkDeviceAddress).
    acceleration_structure: vec2<u32>,
}

struct PtlasFillParams {
    /// Active instances this frame (fill_incremental dispatch bound).
    active_count: u32,
    /// CPU-seeded delta records — the GPU atomic counter starts here.
    cpu_count: u32,
    /// 1 → write every active instance (full rebuild); 0 → only
    /// instances of geometries rebuilt this frame.
    force_all: u32,
}

/// The NV global-partition sentinel
/// (`VK_PARTITIONED_ACCELERATION_STRUCTURE_PARTITION_INDEX_GLOBAL_NV`).
const PTLAS_GLOBAL_PARTITION: u32 = 0xffffffffu;

// slot-indexed: instance → its current BLAS device address (from
// `blas_sharing::assign_address`).
@group(1) @binding(0) var<storage, read> instance_blas_address: array<vec2<u32>>;
// geometry → 1 if its shared BLAS was rebuilt this frame
// (`blas_sharing::elect_dirty`).
@group(1) @binding(1) var<storage, read> geometry_dirty: array<u32>;
// Single GPU record counter. Seeded to `cpu_count` CPU-side.
@group(1) @binding(2) var<storage, read_write> write_count: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> write_data: array<WriteInstanceData>;
// CPU-seeded delta: `.x` = slot, `.y` = null flag (1 → disabled → null AS).
@group(1) @binding(4) var<storage, read> write_slots_cpu: array<vec2<u32>>;
// active (dense) index → real GpuEntity.
@group(1) @binding(5) var<storage, read> active_to_slot: array<u32>;
// `VkBuildPartitionedAccelerationStructureIndirectCommandNV` as u32s;
// `[1]` is `arg_count`, written by `finalize`.
@group(1) @binding(6) var<storage, read_write> src_infos: array<u32>;
@group(1) @binding(7) var<uniform> params: PtlasFillParams;
// slot-indexed: instance → dense geometry id.
@group(1) @binding(8) var<storage, read> instance_geometry_ids: array<u32>;
// slot-indexed: instance → 8-bit RT cull mask (from `RenderLayers`). A
// camera's `cullMask` hides non-matching instances during BVH traversal.
@group(1) @binding(9) var<storage, read> instance_masks: array<u32>;
// slot-indexed: instance → last frame's world transform (the gather's
// previous-frame buffer). Compared against `cluster_instance_transforms`
// (this frame's) to detect moved instances GPU-side.
@group(1) @binding(10) var<storage, read> instance_previous_transforms: array<mat3x4<f32>>;
// slot-indexed: instance → stable material slot (`MaterialColumn`).
@group(1) @binding(11) var<storage, read> instance_material_ids: array<u32>;
// material-slot-indexed traversal flags (`MaterialTraversalFlags`).
@group(1) @binding(12) var<storage, read> material_traversal_flags: array<u32>;
// slot-indexed: the `instance_flags` value last WRITTEN into this slot's
// PTLAS record (persistent). `fill_incremental` re-specifies an instance
// when its derived flags drift from this — the flags twin of the move
// compare — which is what lets the opacity flag be derived purely from the
// MATERIAL: late loads, swaps, and live asset edits all self-heal.
@group(1) @binding(13) var<storage, read_write> instance_written_flags: array<u32>;
// slot-indexed: instance → its transform-table node slot (`NodeSlotColumn`).
@group(1) @binding(14) var<storage, read> node_slots: array<u32>;
// node-indexed: 1 if the node's entity is `TransformStatic`
// (`Presence<StaticColumn>`). Read via the instance's node slot.
@group(1) @binding(15) var<storage, read> static_flags: array<u32>;
// slot-indexed: the partition last WRITTEN into this slot's PTLAS record
// (persistent). `fill_incremental` re-specifies an instance when its resolved
// partition drifts from this — the partition twin of the `instance_written_flags`
// self-heal. Closes the window where an instance is placed (e.g. in global)
// before its `TransformStatic` flag has scattered: once the flag lands, the
// drift migrates it into the static partition.
@group(1) @binding(16) var<storage, read_write> instance_written_partition: array<u32>;

/// The PTLAS partition this instance belongs to. A static instance (its node
/// tagged `TransformStatic`) goes to the single regular partition, so it's
/// written once and then carried from `src` untouched. A mover (or any untagged
/// / node-less instance) goes to the global partition, which NV builds
/// per-instance — so a moved mover never dirties the static partition.
fn resolve_partition(slot: u32) -> u32 {
    let node = node_slots[slot];
    // No transform node (defensive) or not static → global.
    if node == PTLAS_GLOBAL_PARTITION || static_flags[node] == 0u {
        return PTLAS_GLOBAL_PARTITION;
    }
    return 0u;
}

/// The `instance_flags` an instance's record should carry, derived from its
/// material — not stored per instance anywhere on the CPU.
fn derived_vk_flags(slot: u32) -> u32 {
    return material_vk_flags(material_traversal_flags[instance_material_ids[slot]]);
}

/// The RT-pipeline SBT hit record this cluster instance selects: one record PER
/// MATERIAL (index = material slot), not just per class. This gives Shader
/// Execution Reordering a per-material key so warps cohere by material → material
/// + texture data is uniform per dispatch (no descriptor-array divergence). The
/// record's shader handle still encodes the class. Ignored by the inline-rayQuery
/// path.
fn derived_hit_group(slot: u32) -> u32 {
    return instance_material_ids[slot];
}

fn make_record(slot: u32, addr: vec2<u32>) -> WriteInstanceData {
    // `cluster_instance_transforms` is bevy_pbr's affine `mat3x4`
    // (column k = the standard 4x4's row k). NV `TransformMatrixKHR`
    // wants row-major 3x4, so each column maps directly to a row.
    let m = cluster_instance_transforms[slot];
    var transform: array<f32, 12>;
    transform[0]  = m[0].x; transform[1]  = m[0].y; transform[2]  = m[0].z; transform[3]  = m[0].w;
    transform[4]  = m[1].x; transform[5]  = m[1].y; transform[6]  = m[1].z; transform[7]  = m[1].w;
    transform[8]  = m[2].x; transform[9]  = m[2].y; transform[10] = m[2].z; transform[11] = m[2].w;

    // Zero AABB + zero flags — the driver derives bounds from the BLAS.
    var explicit_aabb: array<f32, 6>;
    explicit_aabb[0] = 0.0; explicit_aabb[1] = 0.0; explicit_aabb[2] = 0.0;
    explicit_aabb[3] = 0.0; explicit_aabb[4] = 0.0; explicit_aabb[5] = 0.0;

    // Stamp the flags + partition this record carries (side effect — every record
    // write goes through here, so the mirrors track exactly what the PTLAS holds).
    let vk_flags = derived_vk_flags(slot);
    instance_written_flags[slot] = vk_flags;
    let part = resolve_partition(slot);
    instance_written_partition[slot] = part;

    return WriteInstanceData(
        transform,
        explicit_aabb,
        slot,                                            // instance_id (presented to hit shaders)
        instance_hardware_mask(instance_masks[slot]),    // 8-bit RenderLayers cull mask
        derived_hit_group(slot),                         // SBT hit-group offset (opaque/glass)
        vk_flags,                                        // alpha-tested material → FORCE_NO_OPAQUE
        slot,                                            // instance_index — STABLE PTLAS slot
        part,
        addr,
    );
}

/// CPU-seeded delta records: added ∪ moved ∪ disabled.
@compute @workgroup_size(64)
fn fill_seed(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    // Flat index across a 2D-split dispatch (X capped at 65535, rest in Y).
    let i = gid.x + gid.y * num_workgroups.x * 64u;
    if i >= params.cpu_count {
        return;
    }
    let pair = write_slots_cpu[i];
    let slot = pair.x;
    let is_null = pair.y;
    var addr = vec2<u32>(0u, 0u);
    if is_null == 0u {
        addr = instance_blas_address[slot];
    }
    write_data[i] = make_record(slot, addr);
}

/// Move + band-cross detection over active instances. Appends a WRITE record
/// for any instance that moved this frame or whose BLAS address changed (or
/// all, on full rebuild).
@compute @workgroup_size(64)
fn fill_incremental(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    // Flat index across a 2D-split dispatch (X capped at 65535, rest in Y).
    let d = gid.x + gid.y * num_workgroups.x * 64u;
    if d >= params.active_count {
        return;
    }
    let slot = active_to_slot[d];
    let geom = instance_geometry_ids[slot];
    // Not a full rebuild and the shared BLAS wasn't rebuilt → only re-specify
    // this instance if it MOVED: its gathered world transform differs from last
    // frame's (`previous`, shifted by the gather before this pass). A static
    // instance's world is recomputed bit-identically, so it compares equal and
    // is carried unchanged from `src`.
    if params.force_all == 0u && geometry_dirty[geom] == 0u {
        let cur = cluster_instance_transforms[slot];
        let prev = instance_previous_transforms[slot];
        let moved = any(cur[0] != prev[0]) || any(cur[1] != prev[1]) || any(cur[2] != prev[2]);
        // Material-derived flags drifted from the record (material loaded /
        // swapped / edited since it was written) → re-specify.
        let flags_changed = derived_vk_flags(slot) != instance_written_flags[slot];
        // Partition drifted: the instance was placed before its `TransformStatic`
        // flag scattered (so it sits in global), and the flag has since landed →
        // migrate it into the static partition. Settled instances compute the same
        // partition they hold, so this is false and they're carried from `src`.
        let partition_changed = resolve_partition(slot) != instance_written_partition[slot];
        if !moved && !flags_changed && !partition_changed {
            return;
        }
    }
    let addr = instance_blas_address[slot];
    let idx = atomicAdd(&write_count[0], 1u);
    write_data[idx] = make_record(slot, addr);
}

/// Publish the GPU-driven record count into the build op's arg_count.
@compute @workgroup_size(1)
fn finalize() {
    src_infos[1] = atomicLoad(&write_count[0]);
}
