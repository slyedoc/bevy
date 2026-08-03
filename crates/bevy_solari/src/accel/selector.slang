// Per-bucket cluster LOD selection — object-space DAG cut.
//
// BLAS sharing buckets instances by (geometry, LOD-band) upstream
// (`blas_sharing.slang`). This pass runs ONE workgroup per bucket and
// emits the cluster cut for that bucket's geometry at the band's fixed
// object-space error budget `E_build`. Every instance in the bucket
// shares the resulting BLAS, so the cut must NOT depend on any single
// instance — it is a pure function of (geometry, band):
//
//   emit group G iff G.error <= E_build ∧ (G is root ∨ parent(G).error > E_build)
//        ∨ (G is finest ∧ G.error > E_build)        // leaf fallback
//
// The screen-space projection collapses to a constant (the band's
// budget), so there is no camera, transform, or sphere here. Monotone
// `max_quadric_error` across the DAG (guaranteed by the bake) makes the
// cut single-coverage — every surface region appears exactly once.
//
// A multi-set heap kernel (`HeapKernel::new_with_mappings`): the cluster
// scene set 0 comes from `cluster_bindings`'s declarations mapped at
// constant heap offsets (`cluster_heap_mappings`); this pass's own set-1
// bindings come from the push blob's slot array behind the params block.

import cluster_bindings;

static const uint PARENT_GROUP_NONE = 0xFFFFFFFF;

struct SelectorParams {
    // Per-bucket slot stride into `selected_clas_refs` (in cluster
    // references = 8 bytes each).
    uint max_clusters_per_bucket;
    // Workgroups dispatched = bucket capacity; threads past the live
    // bucket count early-out.
    uint bucket_capacity;
    // `selected_clas_refs` base device address split as low/high u32.
    // The per-bucket refs address written to `args_buf` is
    // `base + bucket_id * max_clusters_per_bucket * 8`.
    uint selected_clas_refs_addr_lo;
    uint selected_clas_refs_addr_hi;
}

[[vk::push_constant]] SelectorParams params;

// `cluster_clas_addresses[global_cluster_id]` → `uint2` of the
// per-cluster CLAS device address (low/high). Built once at asset
// upload; shared across every instance/bucket.
[[vk::binding(0, 1)]] StructuredBuffer<uint2> cluster_clas_addresses;

// Per-bucket ref lists — bucket owns
// `[bucket*max_per .. (bucket+1)*max_per]`. `select_main` atomic-appends
// accepted clusters; the BLAS build reads the prefix via the count in
// `args_buf`.
[[vk::binding(1, 1)]] RWStructuredBuffer<uint2> selected_clas_refs;

// Per-bucket BLAS-build input records. Layout matches
// `VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV`:
//   x = cluster_references_count
//   y = cluster_references_stride (always 8 — u64 device address per slot)
//   z = cluster_references device address, low 32
//   w = cluster_references device address, high 32
[[vk::binding(2, 1)]] RWStructuredBuffer<uint4> args_buf;

// Per-bucket emit counter. Reset to 0 by `select_reset` each frame,
// atomic-incremented in `select_main`.
[[vk::binding(3, 1)]] RWStructuredBuffer<uint> per_bucket_counts;

// Bucket descriptor, 2× uint4 per bucket (mirrors `blas_sharing.slang`):
//   [0] = (group_base, cluster_base, cluster_count, root_group)
//   [1] = (e_build_bits, band, _, _)
[[vk::binding(4, 1)]] StructuredBuffer<uint4> bucket_desc;

// Live (monotonic) dense bucket count — the guard bound.
[[vk::binding(5, 1)]] StructuredBuffer<uint> live_bucket_count;

/// Zero out per-bucket counters at the start of each frame. One thread
/// per bucket; the overhang past the live count early-outs.
[shader("compute")]
[numthreads(64, 1, 1)]
void select_reset(uint3 gid: SV_DispatchThreadID) {
    let b = gid.x;
    if (b >= live_bucket_count[0]) {
        return;
    }
    per_bucket_counts[b] = 0;
}

/// One workgroup per bucket; the 64 threads stride through this
/// bucket's geometry clusters. Object-space DAG cut at the bucket's
/// `E_build`; atomic-append accepted CLAS addresses; thread 0 writes the
/// bucket's `args_buf` record.
[shader("compute")]
[numthreads(64, 1, 1)]
void select_main(uint3 wid: SV_GroupID, uint lid: SV_GroupIndex) {
    let bucket_id = wid.x;
    if (bucket_id >= live_bucket_count[0]) {
        return;
    }
    let d0 = bucket_desc[bucket_id * 2];
    let d1 = bucket_desc[bucket_id * 2 + 1];
    let group_base = d0.x;
    let cluster_base = d0.y;
    let cluster_count = d0.z;
    let e_build = asfloat(d1.x);

    let max_per = params.max_clusters_per_bucket;
    let refs_base = bucket_id * max_per;

    for (uint cid = lid; cid < cluster_count; cid += 64) {
        let cluster_global = cluster_base + cid;
        let cluster = clusters[cluster_global];
        // `cluster_to_group` is global (rebased at upload).
        // `group.parent_group` is mesh-local — rebase via `group_base`.
        let group_global = cluster_to_group[cluster_global];
        let group = cluster_groups[group_global];

        let own_fits = group.max_quadric_error <= e_build;

        let parent_local = group.parent_group;
        var parent_fits = false; // root: parent "error" is +inf → never fits
        if (parent_local != PARENT_GROUP_NONE) {
            let parent = cluster_groups[group_base + parent_local];
            parent_fits = parent.max_quadric_error <= e_build;
        }

        let is_finest = cluster.lod_level == 0;
        // Emit the coarsest level whose own error fits and whose parent's
        // does not; finest level is the fallback when nothing fits.
        let normal_emit = own_fits && !parent_fits;
        let leaf_fallback = is_finest && !own_fits;
        if (!(normal_emit || leaf_fallback)) {
            continue;
        }

        uint slot;
        InterlockedAdd(per_bucket_counts[bucket_id], 1, slot);
        if (slot >= max_per) {
            continue;
        }
        selected_clas_refs[refs_base + slot] = cluster_clas_addresses[cluster_global];
    }

    GroupMemoryBarrierWithGroupSync();
    if (lid == 0) {
        let count = min(per_bucket_counts[bucket_id], max_per);
        let byte_offset = bucket_id * max_per * 8;
        let new_lo = params.selected_clas_refs_addr_lo + byte_offset;
        // 32-bit carry — NV device addresses can straddle 4 GB.
        let carry = new_lo < params.selected_clas_refs_addr_lo ? 1u : 0u;
        let new_hi = params.selected_clas_refs_addr_hi + carry;
        args_buf[bucket_id] = uint4(count, 8, new_lo, new_hi);
    }
}
