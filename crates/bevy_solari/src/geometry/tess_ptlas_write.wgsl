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

/// Mirrors `TessInstanceGpu` (Rust). The object-space BLAS is placed by the PTLAS
/// instance transform = `transforms[instance_id]` (the origin-relative `world_rel`
/// column), and the explicit world AABB is derived from that same transform (so the
/// build never derives bounds from the BLAS). The BLAS device address is NOT carried
/// here — it's read GPU-side from `blas_addresses` at `blas_slot` (written by the
/// per-instance BLAS build, so the trace never stalls on a CPU readback).
struct TessInstance {
    // Object-space AABB center / half (half pre-inflated by the displacement margin).
    // The instance transform + explicit WORLD AABB are derived here from
    // `transforms[instance_id]` (the origin-relative `world_rel` column).
    aabb_center: vec4<f32>,
    aabb_half: vec4<f32>,
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
// Gathered cluster-indexed origin-relative world column (mat3x4 per instance slot,
// row k = (basis_row_k, t_k)) — the same `transforms` the closest-hit reads. Indexed
// by `instance_id` (the cluster slot), which is why it must be the gathered column
// and not the NODE-indexed `world_rel`.
@group(0) @binding(5) var<storage, read> transforms: array<mat3x4<f32>>;

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

    // PTLAS instance transform = the origin-relative world column (`world_rel[slot]`),
    // read this frame. The BLAS is baked in OBJECT space, so this transform places it
    // into the floating-origin world — like an ordinary cluster instance — and the
    // closest-hit re-applies it (ObjectToWorld) to the position-fetched vertices.
    // Read here (at TLAS build, after the subtract pass) so it tracks the moving origin.
    let m = transforms[inst.instance_id];
    var transform: array<f32, 12>;
    transform[0]  = m[0].x; transform[1]  = m[0].y; transform[2]  = m[0].z; transform[3]  = m[0].w;
    transform[4]  = m[1].x; transform[5]  = m[1].y; transform[6]  = m[1].z; transform[7]  = m[1].w;
    transform[8]  = m[2].x; transform[9]  = m[2].y; transform[10] = m[2].z; transform[11] = m[2].w;

    // Explicit world AABB so the partitioned build never derives bounds from the
    // BLAS (a zero/NaN derived AABB hangs the build). Derived from the 8 object-space
    // corners under the same world transform as the instance.
    let c = inst.aabb_center.xyz;
    let h = inst.aabb_half.xyz;
    var wmin = vec3<f32>(1e30);
    var wmax = vec3<f32>(-1e30);
    for (var corner = 0u; corner < 8u; corner = corner + 1u) {
        let s = vec3<f32>(
            select(-1.0, 1.0, (corner & 1u) != 0u),
            select(-1.0, 1.0, (corner & 2u) != 0u),
            select(-1.0, 1.0, (corner & 4u) != 0u),
        );
        let p = vec4<f32>(c + s * h, 1.0);
        let w = vec3<f32>(dot(m[0], p), dot(m[1], p), dot(m[2], p));
        wmin = min(wmin, w);
        wmax = max(wmax, w);
    }
    var explicit_aabb: array<f32, 6>;
    explicit_aabb[0] = wmin.x; explicit_aabb[1] = wmin.y; explicit_aabb[2] = wmin.z;
    explicit_aabb[3] = wmax.x; explicit_aabb[4] = wmax.y; explicit_aabb[5] = wmax.z;

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
