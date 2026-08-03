// Fills the partitioned-AS WRITE record buffer consumed by NV's
// `vkCmdBuildPartitionedAccelerationStructuresNV`, then GPU-writes the
// build's instance count. Four passes:
//
//   fill_seed        — one thread per CPU-seeded delta record (added ∪
//                      moved ∪ disabled). Writes `write_data[i]`.
//   fill_incremental — one thread per ACTIVE instance. Appends a WRITE
//                      record for any instance that (a) MOVED this frame —
//                      its gathered world transform differs from last
//                      frame's (the gather shifts current→previous before
//                      this pass) — or (b) whose GEOMETRY was rebuilt this
//                      frame (`geometry_dirty`), its shared BLAS content
//                      changed in place. Move detection is GPU-side only;
//                      comparing actual world transforms also catches
//                      parent-driven moves. `force_all` writes every active
//                      instance (full rebuild). Static frame → nothing
//                      moved/dirty → pure CPU delta (added ∪ disabled).
//   finalize         — one thread. Writes the live record count into the
//                      build op's `arg_count` field (GPU-driven count).
//   validate         — debug (SolariSettings::ptlas_validate): flag +
//                      defuse records that would MMU-fault the build.
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
//
// A multi-set heap kernel (`HeapKernel::new_with_mappings`): the cluster
// scene set 0 comes from `cluster_bindings` mapped at constant heap
// offsets (`cluster_heap_mappings`); this pass's own set-1 bindings come
// from the push blob's slot array behind the params block.

import cluster_bindings;
import instance_mask;

/// Mirror of `VkPartitionedAccelerationStructureWriteInstanceDataNV`
/// (104 B). Field order + size must match the Rust side byte-for-byte.
struct WriteInstanceData {
    float transform[12];
    float explicit_aabb[6];
    uint instance_id;
    uint instance_mask;
    uint instance_contribution_to_hit_group_index;
    uint instance_flags;
    uint instance_index;
    uint partition_index;
    /// BLAS device address as `uint2` (little-endian VkDeviceAddress).
    uint2 acceleration_structure;
}

struct PtlasFillParams {
    /// Active instances this frame (fill_incremental dispatch bound).
    uint active_count;
    /// CPU-seeded delta records — the GPU atomic counter starts here.
    uint cpu_count;
    /// 1 → write every active instance (full rebuild); 0 → only
    /// instances of geometries rebuilt this frame.
    uint force_all;
    /// This build's seed-epoch stamp (≥1, bumped per build) — see `seed_epoch`.
    uint epoch;
    /// X workgroup count of THIS entry's 2D-split dispatch (X capped at
    /// 65535, rest in Y) — pushed per dispatch, so each entry gets its own.
    uint groups_x;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}

[[vk::push_constant]] PtlasFillParams params;

/// The NV global-partition sentinel
/// (`VK_PARTITIONED_ACCELERATION_STRUCTURE_PARTITION_INDEX_GLOBAL_NV`).
static const uint PTLAS_GLOBAL_PARTITION = 0xffffffff;
/// The geometry's CLASes were built with a baked opacity micromap attached.
static const uint GEOMETRY_FLAG_HAS_OMM = 1;
static const uint BUILT_NO_LEVEL = 0xFFFFFFFF;

// slot-indexed: instance → its current BLAS device address (from
// `blas_sharing::assign_address`).
[[vk::binding(0, 1)]] StructuredBuffer<uint2> instance_blas_address;
// geometry → 1 if its shared BLAS was rebuilt this frame
// (`blas_sharing::elect_dirty`).
[[vk::binding(1, 1)]] StructuredBuffer<uint> geometry_dirty;
// Single GPU record counter. Seeded to `cpu_count` CPU-side.
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> write_count;
[[vk::binding(3, 1)]] RWStructuredBuffer<WriteInstanceData> write_data;
// CPU-seeded delta: `.x` = slot, `.y` = null flag (1 → disabled → null AS).
[[vk::binding(4, 1)]] StructuredBuffer<uint2> write_slots_cpu;
// active (dense) index → real GpuEntity.
[[vk::binding(5, 1)]] StructuredBuffer<uint> active_to_slot;
// `VkBuildPartitionedAccelerationStructureIndirectCommandNV` as u32s;
// `[1]` is `arg_count`, written by `finalize`.
[[vk::binding(6, 1)]] RWStructuredBuffer<uint> src_infos;
// slot-indexed: instance → dense geometry id.
[[vk::binding(7, 1)]] StructuredBuffer<uint> instance_geometry_ids;
// slot-indexed: instance → 8-bit RT cull mask (from `RenderLayers`). A
// camera's `cullMask` hides non-matching instances during BVH traversal.
[[vk::binding(8, 1)]] StructuredBuffer<uint> instance_masks;
// slot-indexed: instance → last frame's world transform (the gather's
// previous-frame buffer, 3 `float4` rows per slot). Compared against
// `cluster_instance_transforms` (this frame's) to detect moved instances
// GPU-side.
[[vk::binding(9, 1)]] StructuredBuffer<float4> instance_previous_transforms;
// slot-indexed: instance → stable material slot (`MaterialColumn`).
[[vk::binding(10, 1)]] StructuredBuffer<uint> instance_material_ids;
// material-slot-indexed traversal flags (`MaterialTraversalFlags`).
[[vk::binding(11, 1)]] StructuredBuffer<uint> material_traversal_flags;
// slot-indexed: the `instance_flags` value last WRITTEN into this slot's
// PTLAS record (persistent). `fill_incremental` re-specifies an instance
// when its derived flags drift from this — the flags twin of the move
// compare — which is what lets the opacity flag be derived purely from the
// MATERIAL: late loads, swaps, and live asset edits all self-heal.
[[vk::binding(12, 1)]] RWStructuredBuffer<uint> instance_written_flags;
// slot-indexed: instance → its transform-table node slot (`NodeSlotColumn`).
[[vk::binding(13, 1)]] StructuredBuffer<uint> node_slots;
// node-indexed: 1 if the node's entity is `TransformStatic`
// (`Presence<StaticColumn>`). Read via the instance's node slot.
[[vk::binding(14, 1)]] StructuredBuffer<uint> static_flags;
// slot-indexed: the partition last WRITTEN into this slot's PTLAS record
// (persistent). `fill_incremental` re-specifies an instance when its resolved
// partition drifts from this — the partition twin of the `instance_written_flags`
// self-heal. Closes the window where an instance is placed (e.g. in global)
// before its `TransformStatic` flag has scattered: once the flag lands, the
// drift migrates it into the static partition.
[[vk::binding(15, 1)]] RWStructuredBuffer<uint> instance_written_partition;
// Debug lane (SolariSettings::ptlas_validate): [0..4)=valid VA span lo/hi
// (CPU-written u64 halves), [4]=instance capacity, [5]=partition count,
// [6]=bad count, [7..)=up to 15 × (kind, record, slot, value.x, value.y).
// kind: 1=address outside the span, 2=instance_index >= capacity, 3=bad
// partition_index. [82]/[83] are the always-on heal counters.
[[vk::binding(16, 1)]] RWStructuredBuffer<uint> validate_report;
// slot-indexed: the epoch `fill_seed` last wrote this slot's record in.
// `fill_incremental` skips epoch-stamped slots — an instance must be WRITTEN
// at most once per partitioned-AS build (a CPU-seeded add would otherwise
// also be appended as a "mover": its fresh slot's previous transform never
// matches). The CPU seed is authoritative.
[[vk::binding(17, 1)]] RWStructuredBuffer<uint> seed_epoch;
// slot-indexed: PTLAS regular-partition hint (0xffffffff = derive from the
// static flag). CPU-assigned per streamed cell — tight per-cell partitions.
[[vk::binding(18, 1)]] StructuredBuffer<uint> partition_hints;
// geometry → LOD level its shared BLAS was ACTUALLY built at (committed only
// after the build chain records — `blas_sharing::commit_built`). `NO_LEVEL`
// means the pool bytes at this geometry's address don't exist yet: a record
// written now would bake a hollow zero-extent leaf into the TLAS. Count it into
// the heal word so rebuild-until-clean restamps once the content lands.
[[vk::binding(19, 1)]] StructuredBuffer<uint> geometry_built_level;
/// geometry → flag bits; see `BlasSharing::geometry_flags`.
[[vk::binding(20, 1)]] StructuredBuffer<uint> geometry_flags;

/// The PTLAS partition this instance belongs to. A static instance (its node
/// tagged `TransformStatic`) goes to the single regular partition, so it's
/// written once and then carried from `src` untouched. A mover (or any untagged
/// / node-less instance) goes to the global partition, which NV builds
/// per-instance — so a moved mover never dirties the static partition.
uint resolve_partition(uint slot) {
    // Explicit spatial hint wins: one tight partition per streamed cell gives
    // the driver a per-cell BVH + per-cell incremental rebuilds.
    let hint = partition_hints[slot];
    if (hint != PTLAS_GLOBAL_PARTITION) {
        return hint;
    }
    let node = node_slots[slot];
    // No transform node (defensive) or not static → global.
    if (node == PTLAS_GLOBAL_PARTITION || static_flags[node] == 0) {
        return PTLAS_GLOBAL_PARTITION;
    }
    return 0;
}

/// The `instance_flags` an instance's record should carry, derived from its
/// geometry + material — not stored per instance anywhere on the CPU.
uint derived_vk_flags(uint slot) {
    // A geometry whose CLASes carry a baked opacity micromap: the micromap
    // drives traversal per micro-triangle (opaque/transparent commit/skip in
    // hardware, any-hit only on "unknown" regions; a 2-state bake has none).
    // Forcing `FORCE_NO_OPAQUE` here would invoke the any-hit on every hit and
    // waste the micromap.
    if ((geometry_flags[instance_geometry_ids[slot]] & GEOMETRY_FLAG_HAS_OMM) != 0) {
        return 0;
    }
    // No micromap: an alpha-tested material needs the any-hit on every hit
    // (`FORCE_NO_OPAQUE`), else its cutouts render solid.
    return material_vk_flags(material_traversal_flags[instance_material_ids[slot]]);
}

/// The RT-pipeline SBT hit record this cluster instance selects: one record PER
/// MATERIAL (index = material slot), not just per class. This gives Shader
/// Execution Reordering a per-material key so warps cohere by material → material
/// + texture data is uniform per dispatch (no descriptor-array divergence). The
/// record's shader handle still encodes the class. Ignored by the inline-rayQuery
/// path.
uint derived_hit_group(uint slot) {
    return instance_material_ids[slot];
}

WriteInstanceData make_record(uint slot, uint2 addr) {
    // `cluster_instance_transform` is bevy_pbr's affine `mat3x4` (row k =
    // the standard 4x4's row k). NV `TransformMatrixKHR` wants row-major
    // 3x4, so each row maps directly.
    let m = cluster_instance_transform(slot);
    WriteInstanceData rec;
    rec.transform[0]  = m[0].x; rec.transform[1]  = m[0].y; rec.transform[2]  = m[0].z; rec.transform[3]  = m[0].w;
    rec.transform[4]  = m[1].x; rec.transform[5]  = m[1].y; rec.transform[6]  = m[1].z; rec.transform[7]  = m[1].w;
    rec.transform[8]  = m[2].x; rec.transform[9]  = m[2].y; rec.transform[10] = m[2].z; rec.transform[11] = m[2].w;

    // Zero AABB + zero flags — the driver derives bounds from the BLAS.
    rec.explicit_aabb[0] = 0.0; rec.explicit_aabb[1] = 0.0; rec.explicit_aabb[2] = 0.0;
    rec.explicit_aabb[3] = 0.0; rec.explicit_aabb[4] = 0.0; rec.explicit_aabb[5] = 0.0;

    // Stamp the flags + partition this record carries (side effect — every record
    // write goes through here, so the mirrors track exactly what the PTLAS holds).
    let vk_flags = derived_vk_flags(slot);
    instance_written_flags[slot] = vk_flags;
    let part = resolve_partition(slot);
    instance_written_partition[slot] = part;

    rec.instance_id = slot;                                        // presented to hit shaders
    rec.instance_mask = instance_hardware_mask(instance_masks[slot]); // 8-bit RenderLayers cull mask
    rec.instance_contribution_to_hit_group_index = derived_hit_group(slot); // SBT hit-group offset
    rec.instance_flags = vk_flags;             // alpha-tested material → FORCE_NO_OPAQUE
    rec.instance_index = slot;                 // instance_index — STABLE PTLAS slot
    rec.partition_index = part;
    rec.acceleration_structure = addr;
    return rec;
}

/// Whether `slot`'s world transform is ready to feed the AS build. A slot that
/// was just (re)bound but whose transform-table node hasn't been propagated yet
/// reads the zero-initialized `world` buffer, so its gathered `object_to_world`
/// is the all-zero (degenerate) placeholder — handing that to the partitioned-AS
/// builder produces a zero-volume / NaN AABB that can hang the build. Such an
/// instance is written as inactive (null AS) until its transform lands; it pops
/// in the next frame. A real transform always has a non-zero linear (3x3) part,
/// even a pure translation (identity 3x3), so an all-zero linear part is the
/// not-ready sentinel.
bool transform_ready(uint slot) {
    let m = cluster_instance_transform(slot);
    let linear_zero =
        all(m[0].xyz == float3(0.0))
        && all(m[1].xyz == float3(0.0))
        && all(m[2].xyz == float3(0.0));
    if (linear_zero) {
        return false;
    }
    // Reject non-finite (NaN: v != v; inf: |v| past the f32 max).
    let s = m[0] + m[1] + m[2];
    let finite =
        (s.x == s.x) && (s.y == s.y) && (s.z == s.z) && (s.w == s.w)
        && abs(s.x) < 3.0e38 && abs(s.y) < 3.0e38 && abs(s.z) < 3.0e38 && abs(s.w) < 3.0e38;
    return finite;
}

/// CPU-seeded delta records: added ∪ moved ∪ disabled.
[shader("compute")]
[numthreads(64, 1, 1)]
void fill_seed(uint3 gid: SV_DispatchThreadID) {
    // Flat index across the 2D-split dispatch (X capped at 65535, rest in Y).
    let i = gid.x + gid.y * params.groups_x * 64;
    if (i >= params.cpu_count) {
        return;
    }
    let pair = write_slots_cpu[i];
    let slot = pair.x;
    let is_null = pair.y;
    var addr = uint2(0, 0);
    // Active only if not a removal AND its transform is ready — a not-ready slot
    // is written inactive (null AS) so a degenerate transform never reaches the build.
    if (is_null == 0 && transform_ready(slot)) {
        addr = instance_blas_address[slot];
    }
    // Count unintended nulls (word 82; deliberate removals excluded) — feeds the
    // CPU's rebuild-until-clean warmup heal, same as fill_incremental's counter.
    if (is_null == 0 && addr.x == 0 && addr.y == 0) {
        InterlockedAdd(validate_report[82], 1);
    }
    write_data[i] = make_record(slot, addr);
    seed_epoch[slot] = params.epoch;
}

/// Move + band-cross detection over active instances. Appends a WRITE record
/// for any instance that moved this frame or whose BLAS address changed (or
/// all, on full rebuild).
[shader("compute")]
[numthreads(64, 1, 1)]
void fill_incremental(uint3 gid: SV_DispatchThreadID) {
    // Flat index across the 2D-split dispatch (X capped at 65535, rest in Y).
    let d = gid.x + gid.y * params.groups_x * 64;
    if (d >= params.active_count) {
        return;
    }
    let slot = active_to_slot[d];
    // Already CPU-seeded this build — a second WRITE for the same instance in
    // one op is spec-UB (and corrupts the partitioned build under churn).
    if (seed_epoch[slot] == params.epoch) {
        return;
    }
    let geom = instance_geometry_ids[slot];
    // Not a full rebuild and the shared BLAS wasn't rebuilt → only re-specify
    // this instance if it MOVED: its gathered world transform differs from last
    // frame's (`previous`, shifted by the gather before this pass). A static
    // instance's world is recomputed bit-identically, so it compares equal and
    // is carried unchanged from `src`.
    if (params.force_all == 0 && geometry_dirty[geom] == 0) {
        let cur = cluster_instance_transform(slot);
        let pb = slot * 3;
        let moved = any(cur[0] != instance_previous_transforms[pb])
            || any(cur[1] != instance_previous_transforms[pb + 1])
            || any(cur[2] != instance_previous_transforms[pb + 2]);
        // Material-derived flags drifted from the record (material loaded /
        // swapped / edited since it was written) → re-specify.
        let flags_changed = derived_vk_flags(slot) != instance_written_flags[slot];
        // Partition drifted: the instance was placed before its `TransformStatic`
        // flag scattered (so it sits in global), and the flag has since landed →
        // migrate it into the static partition. Settled instances compute the same
        // partition they hold, so this is false and they're carried from `src`.
        let partition_changed = resolve_partition(slot) != instance_written_partition[slot];
        if (!moved && !flags_changed && !partition_changed) {
            return;
        }
    }
    // Not-ready slots (transform not yet propagated) are written inactive so a
    // degenerate object-to-world never reaches the partitioned-AS build.
    var addr = uint2(0, 0);
    if (transform_ready(slot)) {
        addr = instance_blas_address[slot];
    }
    // Heal counters (word 82): (a) null-AS records — inactive with no future
    // re-spec trigger; (b) records whose geometry's shared BLAS content hasn't
    // actually been BUILT yet (address assigned ≠ bytes built — the multi-thread
    // warmup window that bakes hollow zero-extent TLAS leaves). Either kind makes
    // the CPU force full restamps until a build lands fully resolved.
    let blas_unbuilt = geometry_built_level[geom] == BUILT_NO_LEVEL;
    if ((addr.x == 0 && addr.y == 0) || blas_unbuilt) {
        InterlockedAdd(validate_report[82], 1);
    }
    // Count REGULAR-partition writes in INCREMENTAL builds (word 83): the NV
    // driver (580.159) silently mishandles incremental partitioned writes into
    // regular partitions (the same constraint that forces full rebuilds on CPU
    // churn) — so a GPU-detected re-spec of a static instance lands here and the
    // CPU answers with a driver-safe full rebuild next frame.
    if (params.force_all == 0 && resolve_partition(slot) != 0xffffffff) {
        InterlockedAdd(validate_report[83], 1);
    }
    uint idx;
    InterlockedAdd(write_count[0], 1, idx);
    write_data[idx] = make_record(slot, addr);
}

/// Publish the GPU-driven record count into the build op's arg_count.
[shader("compute")]
[numthreads(1, 1, 1)]
void finalize() {
    src_infos[1] = write_count[0];
}

bool addr_ge(uint2 a, uint2 b) {
    return a.y > b.y || (a.y == b.y && a.x >= b.x);
}

void validate_flag(uint kind, uint record, uint slot, uint2 value) {
    uint n;
    InterlockedAdd(validate_report[6], 1, n);
    if (n < 15) {
        let e = 7 + n * 5;
        validate_report[e] = kind;
        validate_report[e + 1] = record;
        validate_report[e + 2] = slot;
        validate_report[e + 3] = value.x;
        validate_report[e + 4] = value.y;
    }
}

/// SolariSettings::ptlas_validate: after finalize, flag + defuse any record that would
/// MMU-fault the partitioned-AS build: a BLAS address outside the allocator's
/// sparse VA span, an instance_index past the build's instance_count, or a
/// partition_index that is neither GLOBAL nor a real partition.
[shader("compute")]
[numthreads(64, 1, 1)]
void validate(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x + gid.y * params.groups_x * 64;
    if (i >= write_count[0]) {
        return;
    }
    let slot = write_data[i].instance_index;
    let cap = validate_report[4];
    if (slot >= cap) {
        validate_flag(2, i, slot, uint2(slot, 0));
        // Defuse: retarget the record at instance 0 as a null write.
        write_data[i].instance_index = 0;
        write_data[i].acceleration_structure = uint2(0, 0);
        return;
    }
    let part = write_data[i].partition_index;
    if (part != PTLAS_GLOBAL_PARTITION && part >= validate_report[5]) {
        validate_flag(3, i, slot, uint2(part, 0));
        write_data[i].partition_index = PTLAS_GLOBAL_PARTITION;
    }
    let addr = write_data[i].acceleration_structure;
    if ((addr.x | addr.y) == 0) {
        return;
    }
    let lo = uint2(validate_report[0], validate_report[1]);
    let hi = uint2(validate_report[2], validate_report[3]);
    if (addr_ge(addr, lo) && !addr_ge(addr, hi)) {
        return;
    }
    validate_flag(1, i, slot, addr);
    write_data[i].acceleration_structure = uint2(0, 0);
}
