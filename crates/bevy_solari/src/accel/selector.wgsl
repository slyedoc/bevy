// Per-bucket cluster LOD selection — object-space DAG cut.
//
// BLAS sharing buckets instances by (geometry, LOD-band) upstream
// (`blas_sharing.wgsl`). This pass runs ONE workgroup per bucket and
// emits the cluster cut for that bucket's geometry at the band's fixed
// object-space error budget `E_build`. Every instance in the bucket
// shares the resulting BLAS, so the cut must NOT depend on any single
// instance — it is a pure function of (geometry, band):
//
//   emit group G iff G.error <= E_build ∧ (G is root ∨ parent(G).error > E_build)
//        ∨ (G is finest ∧ G.error > E_build)        // leaf fallback
//
// This is the object-space analogue of the old per-instance
// screen-space cut: the projection collapses to a constant (the band's
// budget), so there is no camera, transform, or sphere here. Monotone
// `max_quadric_error` across the DAG (guaranteed by the bake) makes the
// cut single-coverage — every surface region appears exactly once.

#import bevy_solari::cluster_bindings::{
    clusters,
    cluster_groups,
    cluster_to_group,
}

const PARENT_GROUP_NONE: u32 = 0xFFFFFFFFu;

struct SelectorParams {
    // Per-bucket slot stride into `selected_clas_refs` (in cluster
    // references = 8 bytes each).
    max_clusters_per_bucket: u32,
    // Workgroups dispatched = bucket capacity; threads past the live
    // bucket count early-out.
    bucket_capacity: u32,
    // `selected_clas_refs` base device address split as low/high u32
    // (WGSL has no native u64). The per-bucket refs address written to
    // `args_buf` is `base + bucket_id * max_clusters_per_bucket * 8`.
    selected_clas_refs_addr_lo: u32,
    selected_clas_refs_addr_hi: u32,
}

// `cluster_clas_addresses[global_cluster_id]` → `vec2<u32>` of the
// per-cluster CLAS device address (low/high). Built once at asset
// upload; shared across every instance/bucket.
@group(1) @binding(0) var<storage, read> cluster_clas_addresses: array<vec2<u32>>;

// Per-bucket ref lists — bucket owns
// `[bucket*max_per .. (bucket+1)*max_per]`. `select_main` atomic-appends
// accepted clusters; the BLAS build reads the prefix via the count in
// `args_buf`.
@group(1) @binding(1) var<storage, read_write> selected_clas_refs: array<vec2<u32>>;

// Per-bucket BLAS-build input records. Layout matches
// `VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV`:
//   x = cluster_references_count
//   y = cluster_references_stride (always 8 — u64 device address per slot)
//   z = cluster_references device address, low 32
//   w = cluster_references device address, high 32
@group(1) @binding(2) var<storage, read_write> args_buf: array<vec4<u32>>;

// Per-bucket emit counter. Reset to 0 by `select_reset` each frame,
// atomic-incremented in `select_main`.
@group(1) @binding(3) var<storage, read_write> per_bucket_counts: array<atomic<u32>>;

@group(1) @binding(4) var<uniform> params: SelectorParams;

// Bucket descriptor, 2× vec4<u32> per bucket (mirrors
// `blas_sharing.wgsl`):
//   [0] = (group_base, cluster_base, cluster_count, root_group)
//   [1] = (e_build_bits, band, _, _)
@group(1) @binding(5) var<storage, read> bucket_desc: array<vec4<u32>>;

// Live (monotonic) dense bucket count — the guard bound.
@group(1) @binding(6) var<storage, read> live_bucket_count: array<u32>;

/// Zero out per-bucket counters at the start of each frame. One thread
/// per bucket; the overhang past the live count early-outs.
@compute @workgroup_size(64)
fn select_reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if b >= live_bucket_count[0] {
        return;
    }
    atomicStore(&per_bucket_counts[b], 0u);
}

/// One workgroup per bucket; the 64 threads stride through this
/// bucket's geometry clusters. Object-space DAG cut at the bucket's
/// `E_build`; atomic-append accepted CLAS addresses; thread 0 writes the
/// bucket's `args_buf` record.
@compute @workgroup_size(64)
fn select_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let bucket_id = wid.x;
    if bucket_id >= live_bucket_count[0] {
        return;
    }
    let d0 = bucket_desc[bucket_id * 2u];
    let d1 = bucket_desc[bucket_id * 2u + 1u];
    let group_base = d0.x;
    let cluster_base = d0.y;
    let cluster_count = d0.z;
    let e_build = bitcast<f32>(d1.x);

    let max_per = params.max_clusters_per_bucket;
    let refs_base = bucket_id * max_per;

    for (var cid: u32 = lid; cid < cluster_count; cid += 64u) {
        let cluster_global = cluster_base + cid;
        let cluster = clusters[cluster_global];
        // `cluster_to_group` is global (rebased at upload).
        // `group.parent_group` is mesh-local — rebase via `group_base`.
        let group_global = cluster_to_group[cluster_global];
        let group = cluster_groups[group_global];

        let own_fits = group.max_quadric_error <= e_build;

        let parent_local = group.parent_group;
        var parent_fits = false; // root: parent "error" is +inf → never fits
        if parent_local != PARENT_GROUP_NONE {
            let parent = cluster_groups[group_base + parent_local];
            parent_fits = parent.max_quadric_error <= e_build;
        }

        let is_finest = cluster.lod_level == 0u;
        // Emit the coarsest level whose own error fits and whose parent's
        // does not; finest level is the fallback when nothing fits.
        let normal_emit = own_fits && !parent_fits;
        let leaf_fallback = is_finest && !own_fits;
        if !(normal_emit || leaf_fallback) {
            continue;
        }

        let slot = atomicAdd(&per_bucket_counts[bucket_id], 1u);
        if slot >= max_per {
            continue;
        }
        selected_clas_refs[refs_base + slot] = cluster_clas_addresses[cluster_global];
    }

    workgroupBarrier();
    if lid == 0u {
        let count = min(atomicLoad(&per_bucket_counts[bucket_id]), max_per);
        let byte_offset: u32 = bucket_id * max_per * 8u;
        let new_lo: u32 = params.selected_clas_refs_addr_lo + byte_offset;
        // 32-bit carry — NV device addresses can straddle 4 GB.
        let carry: u32 = select(0u, 1u, new_lo < params.selected_clas_refs_addr_lo);
        let new_hi: u32 = params.selected_clas_refs_addr_hi + carry;
        args_buf[bucket_id] = vec4<u32>(count, 8u, new_lo, new_hi);
    }
}
