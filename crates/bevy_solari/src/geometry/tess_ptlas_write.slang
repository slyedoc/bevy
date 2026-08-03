// Append every tessellated displacement instance to the partitioned-AS WRITE
// record stream, mirroring `hair/ptlas_hair_write.slang`. Recorded inside
// `accel::ptlas::dispatch_ptlas` after the hair write and before `finalize`, so the
// single GPU record count covers it.
//
// One thread per tessellated instance; each atomic-appends a `WriteInstanceData`.
// Per-instance transform / world AABB / BLAS address ride in the `instances`
// storage buffer; the shared cull mask / SBT record / partition come from the
// push params. Instances occupy PTLAS indices `[tess_base, tess_base + tess_count)`,
// above the cluster slots + hair.

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

// Mirrors `TessWriteParams` (Rust, `tess_displace.rs`) — shared across instances.
struct TessWriteParams {
    uint tess_count;
    uint tess_base;
    uint sbt_record;
    uint mask;
    uint partition_index;
}

// Mirrors `TessInstanceGpu` (Rust; 48 B, pad made explicit). The object-space
// BLAS is placed by the PTLAS instance transform = `transforms[instance_id]`
// (the origin-relative `world_rel` column), and the explicit world AABB is
// derived from that same transform (so the build never derives bounds from the
// BLAS). The BLAS device address is NOT carried here — it's read GPU-side from
// `blas_addresses` at `blas_slot` (written by the per-instance BLAS build, so
// the trace never stalls on a CPU readback).
struct TessInstance {
    // Object-space AABB center / half (half pre-inflated by the displacement margin).
    // The instance transform + explicit WORLD AABB are derived here from
    // `transforms[instance_id]` (the origin-relative `world_rel` column).
    float4 aabb_center;
    float4 aabb_half;
    uint blas_slot;
    uint sbt_record;
    uint instance_id;
    uint _pad;
}

[[vk::push_constant]] TessWriteParams params;

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> write_count;
[[vk::binding(1, 0)]] RWStructuredBuffer<WriteInstanceData> write_data;
[[vk::binding(2, 0)]] StructuredBuffer<TessInstance> instances;
// Per-instance BLAS device addresses (u64 as two u32 each), written GPU-side by
// the per-instance BLAS builds. `0` = not (yet) built → inactive PTLAS instance.
[[vk::binding(3, 0)]] StructuredBuffer<uint> blas_addresses;
// Gathered cluster-indexed origin-relative world column (mat3x4 per instance slot
// as 3 `float4` rows, row k = (basis_row_k, t_k)) — the same `transforms` the
// closest-hit reads. Indexed by `instance_id` (the cluster slot), which is why it
// must be the gathered column and not the NODE-indexed `world_rel`.
[[vk::binding(4, 0)]] StructuredBuffer<float4> transforms;

[shader("compute")]
[numthreads(64, 1, 1)]
void tess_write(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (i >= params.tess_count) {
        return;
    }
    let inst = instances[i];
    // BLAS address read GPU-side (no CPU readback): two u32 at `blas_slot * 2`.
    let addr_base = inst.blas_slot * 2;
    let blas_address = uint2(blas_addresses[addr_base], blas_addresses[addr_base + 1]);

    // PTLAS instance transform = the origin-relative world column (`world_rel[slot]`),
    // read this frame. The BLAS is baked in OBJECT space, so this transform places it
    // into the floating-origin world — like an ordinary cluster instance — and the
    // closest-hit re-applies it (ObjectToWorld) to the position-fetched vertices.
    // Read here (at TLAS build, after the subtract pass) so it tracks the moving origin.
    let m0 = transforms[inst.instance_id * 3];
    let m1 = transforms[inst.instance_id * 3 + 1];
    let m2 = transforms[inst.instance_id * 3 + 2];

    // Explicit world AABB so the partitioned build never derives bounds from the
    // BLAS (a zero/NaN derived AABB hangs the build). Derived from the 8 object-space
    // corners under the same world transform as the instance.
    let c = inst.aabb_center.xyz;
    let h = inst.aabb_half.xyz;
    var wmin = float3(1e30);
    var wmax = float3(-1e30);
    for (var corner = 0u; corner < 8; corner++) {
        let s = float3(
            (corner & 1) != 0 ? 1.0 : -1.0,
            (corner & 2) != 0 ? 1.0 : -1.0,
            (corner & 4) != 0 ? 1.0 : -1.0,
        );
        let p = float4(c + s * h, 1.0);
        let w = float3(dot(m0, p), dot(m1, p), dot(m2, p));
        wmin = min(wmin, w);
        wmax = max(wmax, w);
    }

    let slot = params.tess_base + i;
    uint idx;
    InterlockedAdd(write_count[0], 1, idx);

    WriteInstanceData rec;
    rec.transform[0]  = m0.x; rec.transform[1]  = m0.y; rec.transform[2]  = m0.z; rec.transform[3]  = m0.w;
    rec.transform[4]  = m1.x; rec.transform[5]  = m1.y; rec.transform[6]  = m1.z; rec.transform[7]  = m1.w;
    rec.transform[8]  = m2.x; rec.transform[9]  = m2.y; rec.transform[10] = m2.z; rec.transform[11] = m2.w;
    rec.explicit_aabb[0] = wmin.x; rec.explicit_aabb[1] = wmin.y; rec.explicit_aabb[2] = wmin.z;
    rec.explicit_aabb[3] = wmax.x; rec.explicit_aabb[4] = wmax.y; rec.explicit_aabb[5] = wmax.z;
    // instance_id: the source entity's CLUSTER slot, so the closest-hit reads this
    // surface's real `previous_frame_transforms[]` entry (correct motion vectors /
    // no DLSS flicker) — not a guessed 0. Always a valid cluster slot (< high_water).
    rec.instance_id = inst.instance_id;
    rec.instance_mask = params.mask; // 8-bit cull mask
    // Per-instance hit-group record (this surface's material).
    rec.instance_contribution_to_hit_group_index = inst.sbt_record;
    rec.instance_flags = 0; // opaque, no force-no-opaque
    rec.instance_index = slot; // stable PTLAS slot (unique)
    rec.partition_index = params.partition_index;
    rec.acceleration_structure = blas_address;
    write_data[idx] = rec;
}
