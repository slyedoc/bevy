// BLAS sharing — ONE shared BLAS per geometry (not per instance, not
// per camera-distance band). All instances of a `ClusterMesh` reference
// the same BLAS, built at a single per-geometry discrete LOD level
// chosen from the closest (most-demanding) visible instance.
//
// This is NVIDIA "BLAS merging" (vk_lod_clusters) applied to a fully
// resident engine: the BLAS count is bounded by the resident geometry
// universe — scene-stable and view-independent, unlike a
// (geometry, distance-band) bucketing whose count grows with the view.
// Geometry BLAS addresses are `pool_base + geometry_id * stride`, stable
// for the geometry's life, so a static instance is an incremental-PTLAS
// no-op. A geometry only rebuilds when its chosen LOD level changes
// (camera crossing a band for the *closest* instance), and only the
// instances of rebuilt geometries get re-written to the PTLAS.
//
// Per-frame passes (one thread per active instance / per geometry):
//   0. geom_reset   — desired_level[gid]=NO_LEVEL, dirty[gid]=0.
//   1. classify     — instance screen-error → discrete level; atomicMin
//                     into geometry_desired_level[gid]; stash the
//                     geometry's static descriptor.
//   2. elect_dirty  — per geometry: if desired != built (and an instance
//                     wants it), append to the dirty build list, set
//                     built=desired, mark dirty, write the build
//                     descriptor + stable dst address.
//   3. finalize_count — build_count = min(dirty_count, capacity).
//   4. assign       — instance_blas_address[slot] = pool_base + gid*stride.
//
// A multi-set heap kernel (`HeapKernel::new_with_mappings`): the cluster
// scene set 0 comes from `cluster_bindings` mapped at constant heap
// offsets (`cluster_heap_mappings`); this pass's own set-1 bindings come
// from the push blob's slot array behind the params block. The classify
// projection reads `params.focal_px` — the one View-derived input
// (precomputed CPU-side from the camera's viewport + projection).

import cluster_bindings;

// Sentinel: no instance requested this geometry this frame (atomicMin
// floor). Also the initial built_level so first sighting is dirty.
static const uint NO_LEVEL = 0xFFFFFFFF;
// bucket_desc stride: 2× uint4 per dirty entry (selector format).
static const uint DESC_VEC4 = 2;

struct SharingParams {
    // Active instances this frame (classify/assign dispatch bound).
    uint active_count;
    // Resident geometry high-water (reset/elect dispatch bound).
    uint geometry_count;
    // Dirty-build capacity (== geometry pool capacity); elect clamps here.
    uint geometry_capacity;
    // Max LOD band index (classify clamps to [0, max_band]).
    uint max_band;
    // Band math: level = floor((log2(E_ideal) - e_min_log2) * inv_log2_ratio).
    float e_min_log2;
    float inv_log2_ratio;
    float pixel_error_threshold;
    float near_distance;
    // geometry_blas_pool base device address (lo/hi) + per-geometry stride.
    uint pool_base_lo;
    uint pool_base_hi;
    uint geometry_stride;
    // The camera's focal length in pixels — `viewport_height/2 · P[1][1]`,
    // the only View input the classify projection needs (fed CPU-side).
    float focal_px;
    // X workgroup count of the per-instance 2D-split dispatches (X capped
    // at 65535, rest in Y) — flat-index reconstruction in classify/assign.
    uint groups_x;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}

[[vk::push_constant]] SharingParams params;

// dense active index → real GpuEntity.
[[vk::binding(0, 1)]] StructuredBuffer<uint> active_to_slot;
// slot → dense geometry id.
[[vk::binding(1, 1)]] StructuredBuffer<uint> instance_geometry_ids;
// geometry → finest LOD level any visible instance wants (atomicMin).
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> geometry_desired_level;
// geometry → LOD level its resident BLAS was built at (PERSISTENT).
[[vk::binding(3, 1)]] RWStructuredBuffer<uint> geometry_built_level;
// geometry → 1 if rebuilt this frame (PTLAS re-writes its instances).
[[vk::binding(4, 1)]] RWStructuredBuffer<uint> geometry_dirty;
// dirty build count (= number of BLAS built this frame); atomic alloc.
[[vk::binding(5, 1)]] RWStructuredBuffer<uint> dirty_count;
// dirty entry i → geometry id.
[[vk::binding(6, 1)]] RWStructuredBuffer<uint> dirty_gid;
// dirty entry i → build descriptor (selector "bucket_desc" format):
//   [0] = (group_base, cluster_base, cluster_count, root_group)
//   [1] = (e_build_bits, level, _, _)
[[vk::binding(7, 1)]] RWStructuredBuffer<uint4> bucket_desc;
// dirty entry i → geometry's STABLE BLAS address (the build dst array).
[[vk::binding(8, 1)]] RWStructuredBuffer<uint2> bucket_dst_addresses;
// slot → instance BLAS device address (PTLAS fill reads this).
[[vk::binding(9, 1)]] RWStructuredBuffer<uint2> instance_blas_address;
// clamped build count = min(dirty_count, capacity); build srcInfosCount.
[[vk::binding(10, 1)]] RWStructuredBuffer<uint> build_count;
// geometry → static descriptor (group_base, cluster_base, cluster_count,
// root_group). Written by classify (redundant, identical per geometry),
// read by elect_dirty (which is per-geometry and has no instance handle).
[[vk::binding(11, 1)]] RWStructuredBuffer<uint4> geometry_desc;
// slot → per-instance object-space error budget (`e_ideal`, unbanded). Written by
// `classify` and consumed by the shared-BLAS per-instance DAG cut.
[[vk::binding(12, 1)]] RWStructuredBuffer<float> instance_e_build;
// dirty entry i → the selector's per-bucket build args; .x = emitted cluster
// count (0 = empty/incomplete build → commit_built must not commit).
[[vk::binding(13, 1)]] StructuredBuffer<uint4> build_args;
// geometry → 1 once its CLAS bytes exist (CPU-written at upload/instantiate).
[[vk::binding(14, 1)]] StructuredBuffer<uint> clas_ready;

// ---------------------------------------------------------------------
// 64-bit helpers (32-bit carry math, mirrors selector.slang).
// ---------------------------------------------------------------------

uint2 umul_u32(uint a, uint b) {
    let a_lo = a & 0xFFFF;
    let a_hi = a >> 16;
    let b_lo = b & 0xFFFF;
    let b_hi = b >> 16;
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    let mid = (ll >> 16) + (lh & 0xFFFF) + (hl & 0xFFFF);
    let lo = (ll & 0xFFFF) | (mid << 16);
    let hi = hh + (lh >> 16) + (hl >> 16) + (mid >> 16);
    return uint2(lo, hi);
}

uint2 uadd_u64(uint2 a, uint2 b) {
    let lo = a.x + b.x;
    let carry = lo < a.x ? 1u : 0u;
    return uint2(lo, a.y + b.y + carry);
}

uint2 geometry_address(uint gid) {
    let offset = umul_u32(gid, params.geometry_stride);
    return uadd_u64(uint2(params.pool_base_lo, params.pool_base_hi), offset);
}

// ---------------------------------------------------------------------
// Discrete LOD level from one representative root-sphere projection.
// (No DAG walk — same band math as the selector's object-space cut.)
// ---------------------------------------------------------------------

float instance_max_linear_scale(uint slot) {
    let m = cluster_instance_transform(slot);
    let lx = length(float3(m[0].x, m[1].x, m[2].x));
    let ly = length(float3(m[0].y, m[1].y, m[2].y));
    let lz = length(float3(m[0].z, m[1].z, m[2].z));
    return max(max(lx, ly), lz);
}

float3 apply_affine(uint slot, float3 p) {
    let m = cluster_instance_transform(slot);
    return float3(
        dot(m[0].xyz, p) + m[0].w,
        dot(m[1].xyz, p) + m[1].w,
        dot(m[2].xyz, p) + m[2].w,
    );
}

// Object-space error budget reproducing the screen-space cut at this
// instance's distance, quantized to a discrete band. Smaller level =
// finer. Per-geometry clamp to the root's coarsest error so far
// instances of a geometry all collapse to its coarsest level.
uint classify_level(uint slot, uint root_group) {
    let root = cluster_groups[root_group];
    let scale = instance_max_linear_scale(slot);
    let focal_px = params.focal_px;
    let world_center = apply_affine(slot, root.traversal_sphere.xyz);
    let world_radius = root.traversal_sphere.w * scale;
    // `cluster_instance_transforms` is the ORIGIN-RELATIVE world (origin = the
    // camera itself), so the camera sits at (0,0,0) in this space and distance
    // is just the center's length. Subtracting the camera's ABSOLUTE world
    // position would mix spaces: far from the world origin every instance
    // classifies as origin-distance away (coarsest band), and camera motion
    // spuriously crosses bands (shared-BLAS rebuild storms).
    let center_dist = length(world_center);
    let surface_dist = max(params.near_distance, center_dist - world_radius);
    let denom = max(scale * focal_px, 1e-12);
    let e_ideal = params.pixel_error_threshold * surface_dist / denom;
    // Unbanded budget for the shared-BLAS per-instance DAG cut.
    instance_e_build[slot] = e_ideal;
    let band_f = floor((log2(max(e_ideal, 1e-12)) - params.e_min_log2) * params.inv_log2_ratio);
    let band_max_geom = floor(
        (log2(max(root.max_quadric_error, 1e-12)) - params.e_min_log2) * params.inv_log2_ratio,
    );
    let hi = clamp(band_max_geom, 0.0, float(params.max_band));
    let band = clamp(band_f, 0.0, hi);
    return uint(band);
}

float band_error_budget(uint level) {
    return exp2(params.e_min_log2 + float(level) / params.inv_log2_ratio);
}

// =====================================================================
// Pass 0: per-frame reset of the per-geometry scratch.
// =====================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void geom_reset(uint3 gid: SV_DispatchThreadID) {
    let g = gid.x;
    if (g >= params.geometry_count) {
        return;
    }
    geometry_desired_level[g] = NO_LEVEL;
    geometry_dirty[g] = 0;
}

// =====================================================================
// Pass 1: classify each instance → discrete level; reduce per geometry;
// stash the geometry's static descriptor.
// =====================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void classify(uint3 gid: SV_DispatchThreadID) {
    // Flat index across the 2D-split dispatch (X capped at 65535, rest in Y).
    let d = gid.x + gid.y * params.groups_x * 64;
    if (d >= params.active_count) {
        return;
    }
    let slot = active_to_slot[d];
    let geom = instance_geometry_ids[slot];
    let lod = cluster_instance_lod_inputs[slot];
    let level = classify_level(slot, lod.root_group);
    InterlockedMin(geometry_desired_level[geom], level);
    // Static per-geometry descriptor (identical for every instance of
    // this geometry — last writer wins, all writes equal).
    geometry_desc[geom] = uint4(lod.group_base, lod.cluster_base, lod.cluster_count, lod.root_group);
}

// =====================================================================
// Pass 2: per geometry — decide rebuild, append to the dirty build list.
// =====================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void elect_dirty(uint3 gid: SV_DispatchThreadID) {
    let g = gid.x;
    if (g >= params.geometry_count) {
        return;
    }
    let desired = geometry_desired_level[g];
    if (desired == NO_LEVEL) {
        return; // no visible instance this frame
    }
    // CLAS bytes not resident yet → electing now would build a hollow BLAS and
    // bake a zero-extent leaf into the TLAS. Wait; re-checked every frame.
    if (clas_ready[g] == 0) {
        return;
    }
    if (desired == geometry_built_level[g]) {
        return; // resident BLAS already at the wanted level
    }
    // Rebuild needed. Claim a dirty-build slot. `built_level` is NOT committed
    // here: the selector/blas_rebuild consumers have their own cold-start bails,
    // and an optimistic commit over a bailed build leaves a permanent lie
    // ("built") over unbuilt pool bytes — the missing-static-scene startup race.
    // `commit_built` below stamps it only after the build chain actually records.
    uint i;
    InterlockedAdd(dirty_count[0], 1, i);
    if (i >= params.geometry_capacity) {
        return; // overflow (shouldn't happen — capacity == geometry_count)
    }
    geometry_dirty[g] = 1;
    dirty_gid[i] = g;

    let desc = geometry_desc[g]; // (group_base, cluster_base, cluster_count, root_group)
    let e_build = band_error_budget(desired);
    bucket_desc[i * DESC_VEC4] = desc;
    bucket_desc[i * DESC_VEC4 + 1] = uint4(asuint(e_build), desired, 0, 0);
    bucket_dst_addresses[i] = geometry_address(g);
}

// =====================================================================
// Pass 3: clamp the dirty count to capacity for the build srcInfosCount.
// =====================================================================
[shader("compute")]
[numthreads(1, 1, 1)]
void finalize_count() {
    build_count[0] = min(dirty_count[0], params.geometry_capacity);
}

// =====================================================================
// Pass 4: per instance — assign its geometry's stable BLAS address.
// Pure function of geometry_id, independent of the build, so stable
// across frames unless the instance's geometry changes.
// =====================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void assign_address(uint3 gid: SV_DispatchThreadID) {
    // Flat index across the 2D-split dispatch (X capped at 65535, rest in Y).
    let d = gid.x + gid.y * params.groups_x * 64;
    if (d >= params.active_count) {
        return;
    }
    let slot = active_to_slot[d];
    instance_blas_address[slot] = geometry_address(instance_geometry_ids[slot]);
}

// =====================================================================
// Pass 5 (dispatched from `dispatch_blas_rebuild`, AFTER the raw cluster-BLAS
// build is recorded): commit each dirty geometry's built level. Gated on the
// whole selector→build chain actually recording this frame — a cold-start bail
// anywhere leaves `built_level` untouched, so `elect_dirty` re-fires next frame
// and the build retries until it truly lands (rebuild-until-built).
// =====================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void commit_built(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (i >= min(dirty_count[0], params.geometry_capacity)) {
        return;
    }
    // An empty cut (selector emitted no clusters — CLAS data not resident yet)
    // must NOT commit: the pool holds a hollow BLAS. Leave built_level untouched
    // so the bucket re-elects until a real cut lands.
    if (build_args[i].x == 0) {
        return;
    }
    geometry_built_level[dirty_gid[i]] = bucket_desc[i * DESC_VEC4 + 1].y;
}
