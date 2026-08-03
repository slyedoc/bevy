// Appends hair instances to the partitioned-AS WRITE record stream, after the
// cluster fill passes (`ptlas_fill.slang`) and before its `finalize`. One thread
// per hair instance; each atomic-appends a `WriteInstanceData` record to the
// shared `write_data` buffer (the same `write_count` the cluster fill uses), so
// `finalize` publishes the combined count.
//
// Hair occupies PTLAS instance indices `[hair_base, hair_base + hair_count)`,
// above the cluster slot high-water. Hair records are re-specified every frame
// (small N), always in the global partition, referencing the asset's LSS BLAS.

// Must match `GpuHairInstance` (Rust, `hair/mod.rs`) byte-for-byte. 64 B
// (size 52 rounded up by the vec3's 16-alignment, pads made explicit).
struct GpuHairInstance {
    float3 sigma_a;      // absorption from melanin (+ dye), CPU-derived
    uint transform_slot; // index into the GPU `world` buffer (×3 rows)
    uint segment_base;
    float beta_m;
    float beta_n;
    float alpha;
    float ior;
    uint blas_address_lo;
    uint blas_address_hi;
    uint mask;
    // Opaque-surface material slot (`SolariBranches`) or 0xffffffff for fiber hair.
    // Unused by the PTLAS write, but present so the stride matches the path tracer.
    uint material_id;
    uint pad_a;
    uint pad_b;
    uint pad_c;
}

// Must match `VkPartitionedAccelerationStructureWriteInstanceDataNV` /
// `ptlas_fill.slang::WriteInstanceData` (104 B).
struct WriteInstanceData {
    float transform[12];
    float explicit_aabb[6];
    uint instance_id;
    uint instance_mask;
    uint instance_contribution_to_hit_group_index;
    uint instance_flags;
    uint instance_index;
    uint partition_index;
    uint2 acceleration_structure;
}

struct HairWriteParams {
    uint hair_count;
    uint hair_base;
    // SBT hit-record index hair routes to (RtPipeline::hair_sbt_record) — the
    // record baked with the hair closest-hit handle.
    uint hair_sbt_record;
    // X workgroup count of the 2D-split dispatch (X capped at 65535, rest in
    // Y) — flat-index reconstruction below.
    uint groups_x;
}

static const uint PTLAS_GLOBAL_PARTITION = 0xffffffff;

[[vk::push_constant]] HairWriteParams params;

[[vk::binding(0, 0)]] StructuredBuffer<GpuHairInstance> hair_instances;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> write_count;
[[vk::binding(2, 0)]] RWStructuredBuffer<WriteInstanceData> write_data;
// GPU transform table: 3 `float4` rows (mat3x4) per node, indexed `slot*3 + k`.
[[vk::binding(3, 0)]] StructuredBuffer<float4> world;

[shader("compute")]
[numthreads(64, 1, 1)]
void hair_write(uint3 gid: SV_DispatchThreadID) {
    // Flat index across the 2D-split dispatch.
    let i = gid.x + gid.y * params.groups_x * 64;
    if (i >= params.hair_count) {
        return;
    }
    let h = hair_instances[i];

    // World transform from the GPU transform table (row-major 3x4) — copy
    // straight into the VK `TransformMatrixKHR` record.
    let s = h.transform_slot * 3;
    let r0 = world[s];
    let r1 = world[s + 1];
    let r2 = world[s + 2];

    let inst = params.hair_base + i;
    uint idx;
    InterlockedAdd(write_count[0], 1, idx);

    WriteInstanceData rec;
    rec.transform[0]  = r0.x; rec.transform[1]  = r0.y; rec.transform[2]  = r0.z; rec.transform[3]  = r0.w;
    rec.transform[4]  = r1.x; rec.transform[5]  = r1.y; rec.transform[6]  = r1.z; rec.transform[7]  = r1.w;
    rec.transform[8]  = r2.x; rec.transform[9]  = r2.y; rec.transform[10] = r2.z; rec.transform[11] = r2.w;
    rec.explicit_aabb[0] = 0.0; rec.explicit_aabb[1] = 0.0; rec.explicit_aabb[2] = 0.0;
    rec.explicit_aabb[3] = 0.0; rec.explicit_aabb[4] = 0.0; rec.explicit_aabb[5] = 0.0;
    rec.instance_id = inst; // presented to the hit shader as instance_index
    rec.instance_mask = h.mask; // 8-bit cull mask (hair visible to all view masks)
    // Hit-group contribution — the reserved hair SBT record.
    rec.instance_contribution_to_hit_group_index = params.hair_sbt_record;
    rec.instance_flags = 0; // opaque LSS geometry, no force-no-opaque
    rec.instance_index = inst; // PTLAS slot
    rec.partition_index = PTLAS_GLOBAL_PARTITION;
    rec.acceleration_structure = uint2(h.blas_address_lo, h.blas_address_hi);
    write_data[idx] = rec;
}
