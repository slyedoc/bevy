// Appends hair instances to the partitioned-AS WRITE record stream, after the
// cluster fill passes (`ptlas_fill.wgsl`) and before its `finalize`. One thread
// per hair instance; each atomic-appends a `WriteInstanceData` record to the
// shared `write_data` buffer (the same `write_count` the cluster fill uses), so
// `finalize` publishes the combined count.
//
// Hair occupies PTLAS instance indices `[hair_base, hair_base + hair_count)`,
// above the cluster slot high-water. Hair records are re-specified every frame
// (small N), always in the global partition, referencing the asset's LSS BLAS.

/// Must match `GpuHairInstance` (Rust, `hair/mod.rs`) byte-for-byte. 48 B.
struct GpuHairInstance {
    sigma_a: vec3<f32>,  // absorption from melanin (+ dye), CPU-derived
    transform_slot: u32, // index into the GPU `world` buffer (×3 rows)
    segment_base: u32,
    beta_m: f32,
    beta_n: f32,
    alpha: f32,
    ior: f32,
    blas_address_lo: u32,
    blas_address_hi: u32,
    mask: u32,
}

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

struct HairWriteParams {
    hair_count: u32,
    hair_base: u32,
    // SBT hit-record index hair routes to (RtPipeline::hair_sbt_record) — the
    // record baked with the hair closest-hit handle.
    hair_sbt_record: u32,
    pad1: u32,
}

const PTLAS_GLOBAL_PARTITION: u32 = 0xffffffffu;

@group(0) @binding(0) var<storage, read> hair_instances: array<GpuHairInstance>;
@group(0) @binding(1) var<storage, read_write> write_count: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> write_data: array<WriteInstanceData>;
@group(0) @binding(3) var<uniform> params: HairWriteParams;
// GPU transform table: 3 `vec4` rows (mat3x4) per node, indexed `slot*3 + k`.
@group(0) @binding(4) var<storage, read> world: array<vec4<f32>>;

@compute @workgroup_size(64)
fn hair_write(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let i = gid.x + gid.y * num_workgroups.x * 64u;
    if i >= params.hair_count {
        return;
    }
    let h = hair_instances[i];

    // World transform from the GPU transform table (row-major 3x4) — copy
    // straight into the VK `TransformMatrixKHR` record.
    let s = h.transform_slot * 3u;
    let r0 = world[s];
    let r1 = world[s + 1u];
    let r2 = world[s + 2u];
    var transform: array<f32, 12>;
    transform[0]  = r0.x; transform[1]  = r0.y; transform[2]  = r0.z; transform[3]  = r0.w;
    transform[4]  = r1.x; transform[5]  = r1.y; transform[6]  = r1.z; transform[7]  = r1.w;
    transform[8]  = r2.x; transform[9]  = r2.y; transform[10] = r2.z; transform[11] = r2.w;

    var explicit_aabb: array<f32, 6>;
    explicit_aabb[0] = 0.0; explicit_aabb[1] = 0.0; explicit_aabb[2] = 0.0;
    explicit_aabb[3] = 0.0; explicit_aabb[4] = 0.0; explicit_aabb[5] = 0.0;

    let inst = params.hair_base + i;
    let idx = atomicAdd(&write_count[0], 1u);
    write_data[idx] = WriteInstanceData(
        transform,
        explicit_aabb,
        inst,           // instance_id (presented to the hit shader as instance_index)
        h.mask,         // 8-bit cull mask (hair visible to all view masks)
        params.hair_sbt_record, // hit-group contribution — the reserved hair SBT record
        0u,             // instance_flags — opaque LSS geometry, no force-no-opaque
        inst,           // instance_index — PTLAS slot
        PTLAS_GLOBAL_PARTITION,
        vec2<u32>(h.blas_address_lo, h.blas_address_hi),
    );
}
