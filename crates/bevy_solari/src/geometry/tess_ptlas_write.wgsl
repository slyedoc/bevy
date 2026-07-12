// Append every tessellated displacement instance to the partitioned-AS WRITE
// record stream, mirroring `hair/ptlas_hair_write.wgsl`. Recorded inside
// `accel::ptlas::dispatch_ptlas` after the hair write and before `finalize`, so the
// single GPU record count covers it.
//
// One thread per tessellated instance; each atomic-appends a `WriteInstanceData`.
// Per-instance transform / world AABB / BLAS address ride in the `instances`
// storage buffer; the shared cull mask / SBT record / partition come from the
// params UBO. Instances occupy PTLAS indices `[tess_base, tess_base + tess_count)`,
// above the cluster slots + hair.

/// Must match `VkPartitionedAccelerationStructureWriteInstanceDataNV` /
/// `ptlas_fill.wgsl::WriteInstanceData` (104 B).
struct WriteInstanceData {
    transform: array<f32, 12>,
    explicit_aabb: array<f32, 6>,
    instance_id: u32,
    instance_mask: u32,
    instance_contribution_to_hit_group_index: u32,
    instance_flags: u32,
    instance_index: u32,
    partition_index: u32,
    acceleration_structure: vec2<u32>,
}

/// Mirrors `TessWriteParams` (Rust, `tess_displace.rs`) — shared across instances.
struct TessWriteParams {
    tess_count: u32,
    tess_base: u32,
    sbt_record: u32,
    mask: u32,
    partition_index: u32,
}

/// Mirrors `TessInstanceGpu` (Rust). Transform is 3 `vec4` rows (a `mat3x4`);
/// explicit world AABB so the build never derives bounds from the BLAS. The BLAS
/// device address is NOT carried here — it's read GPU-side from `blas_addresses`
/// at `blas_slot` (written by the per-instance BLAS build, so the trace never
/// stalls on a CPU readback).
struct TessInstance {
    transform_r0: vec4<f32>,
    transform_r1: vec4<f32>,
    transform_r2: vec4<f32>,
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    blas_slot: u32,
    sbt_record: u32,
    instance_id: u32,
}

@group(0) @binding(0) var<storage, read_write> write_count: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> write_data: array<WriteInstanceData>;
@group(0) @binding(2) var<uniform> params: TessWriteParams;
@group(0) @binding(3) var<storage, read> instances: array<TessInstance>;
// Per-instance BLAS device addresses (u64 as two u32 each), written GPU-side by
// the per-instance BLAS builds. `0` = not (yet) built → inactive PTLAS instance.
@group(0) @binding(4) var<storage, read> blas_addresses: array<u32>;

@compute @workgroup_size(64)
fn tess_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.tess_count {
        return;
    }
    let inst = instances[i];
    // BLAS address read GPU-side (no CPU readback): two u32 at `blas_slot * 2`.
    let addr_base = inst.blas_slot * 2u;
    let blas_address = vec2<u32>(blas_addresses[addr_base], blas_addresses[addr_base + 1u]);

    var transform: array<f32, 12>;
    transform[0]  = inst.transform_r0.x; transform[1]  = inst.transform_r0.y;
    transform[2]  = inst.transform_r0.z; transform[3]  = inst.transform_r0.w;
    transform[4]  = inst.transform_r1.x; transform[5]  = inst.transform_r1.y;
    transform[6]  = inst.transform_r1.z; transform[7]  = inst.transform_r1.w;
    transform[8]  = inst.transform_r2.x; transform[9]  = inst.transform_r2.y;
    transform[10] = inst.transform_r2.z; transform[11] = inst.transform_r2.w;

    // Explicit world AABB so the partitioned build never derives bounds from the
    // BLAS (a zero/NaN derived AABB hangs the build).
    var explicit_aabb: array<f32, 6>;
    explicit_aabb[0] = inst.aabb_min.x; explicit_aabb[1] = inst.aabb_min.y; explicit_aabb[2] = inst.aabb_min.z;
    explicit_aabb[3] = inst.aabb_max.x; explicit_aabb[4] = inst.aabb_max.y; explicit_aabb[5] = inst.aabb_max.z;

    let slot = params.tess_base + i;
    let idx = atomicAdd(&write_count[0], 1u);
    write_data[idx] = WriteInstanceData(
        transform,
        explicit_aabb,
        // instance_id: the source entity's CLUSTER slot, so the closest-hit reads this
        // surface's real `previous_frame_transforms[]` entry (correct motion vectors /
        // no DLSS flicker) — not a guessed 0. Always a valid cluster slot (< high_water).
        inst.instance_id,
        params.mask,         // 8-bit cull mask
        inst.sbt_record,     // per-instance hit-group record (this surface's material)
        0u,                  // instance_flags — opaque, no force-no-opaque
        slot,                // instance_index — stable PTLAS slot (unique)
        params.partition_index,
        blas_address,
    );
}
