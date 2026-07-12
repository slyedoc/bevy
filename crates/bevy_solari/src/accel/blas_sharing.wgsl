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

#import bevy_solari::cluster_bindings::{
    cluster_groups,
    cluster_instance_transforms,
    cluster_instance_lod_inputs,
}
#import bevy_render::view::View

// Sentinel: no instance requested this geometry this frame (atomicMin
// floor). Also the initial built_level so first sighting is dirty.
const NO_LEVEL: u32 = 0xFFFFFFFFu;
// bucket_desc stride: 2× vec4<u32> per dirty entry (selector format).
const DESC_VEC4: u32 = 2u;

struct SharingParams {
    // Active instances this frame (classify/assign dispatch bound).
    active_count: u32,
    // Resident geometry high-water (reset/elect dispatch bound).
    geometry_count: u32,
    // Dirty-build capacity (== geometry pool capacity); elect clamps here.
    geometry_capacity: u32,
    // Max LOD band index (classify clamps to [0, max_band]).
    max_band: u32,
    // Band math: level = floor((log2(E_ideal) - e_min_log2) * inv_log2_ratio).
    e_min_log2: f32,
    inv_log2_ratio: f32,
    pixel_error_threshold: f32,
    near_distance: f32,
    // geometry_blas_pool base device address (lo/hi) + per-geometry stride.
    pool_base_lo: u32,
    pool_base_hi: u32,
    geometry_stride: u32,
    _pad0: u32,
}

@group(1) @binding(0) var<uniform> view: View;
@group(1) @binding(1) var<uniform> params: SharingParams;
// dense active index → real GpuEntity.
@group(1) @binding(2) var<storage, read> active_to_slot: array<u32>;
// slot → dense geometry id.
@group(1) @binding(3) var<storage, read> instance_geometry_ids: array<u32>;
// geometry → finest LOD level any visible instance wants (atomicMin).
@group(1) @binding(4) var<storage, read_write> geometry_desired_level: array<atomic<u32>>;
// geometry → LOD level its resident BLAS was built at (PERSISTENT).
@group(1) @binding(5) var<storage, read_write> geometry_built_level: array<u32>;
// geometry → 1 if rebuilt this frame (PTLAS re-writes its instances).
@group(1) @binding(6) var<storage, read_write> geometry_dirty: array<u32>;
// dirty build count (= number of BLAS built this frame); atomic alloc.
@group(1) @binding(7) var<storage, read_write> dirty_count: array<atomic<u32>>;
// dirty entry i → geometry id.
@group(1) @binding(8) var<storage, read_write> dirty_gid: array<u32>;
// dirty entry i → build descriptor (selector "bucket_desc" format):
//   [0] = (group_base, cluster_base, cluster_count, root_group)
//   [1] = (e_build_bits, level, _, _)
@group(1) @binding(9) var<storage, read_write> bucket_desc: array<vec4<u32>>;
// dirty entry i → geometry's STABLE BLAS address (the build dst array).
@group(1) @binding(10) var<storage, read_write> bucket_dst_addresses: array<vec2<u32>>;
// slot → instance BLAS device address (PTLAS fill reads this).
@group(1) @binding(11) var<storage, read_write> instance_blas_address: array<vec2<u32>>;
// clamped build count = min(dirty_count, capacity); build srcInfosCount.
@group(1) @binding(12) var<storage, read_write> build_count: array<u32>;
// geometry → static descriptor (group_base, cluster_base, cluster_count,
// root_group). Written by classify (redundant, identical per geometry),
// read by elect_dirty (which is per-geometry and has no instance handle).
@group(1) @binding(13) var<storage, read_write> geometry_desc: array<vec4<u32>>;
// slot → per-instance object-space error budget (`e_ideal`, unbanded). Written by
// `classify` and consumed by the shared-BLAS per-instance DAG cut.
@group(1) @binding(14) var<storage, read_write> instance_e_build: array<f32>;
// dirty entry i → the selector's per-bucket build args; .x = emitted cluster
// count (0 = empty/incomplete build → commit_built must not commit).
@group(1) @binding(15) var<storage, read> build_args: array<vec4<u32>>;
// geometry → 1 once its CLAS bytes exist (CPU-written at upload/instantiate).
@group(1) @binding(16) var<storage, read> clas_ready: array<u32>;

// ---------------------------------------------------------------------
// 64-bit helpers (WGSL has no u64). Mirror selector.wgsl carry math.
// ---------------------------------------------------------------------

fn umul_u32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    let mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    let lo = (ll & 0xFFFFu) | (mid << 16u);
    let hi = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

fn uadd_u64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    return vec2<u32>(lo, a.y + b.y + carry);
}

fn geometry_address(gid: u32) -> vec2<u32> {
    let offset = umul_u32(gid, params.geometry_stride);
    return uadd_u64(vec2<u32>(params.pool_base_lo, params.pool_base_hi), offset);
}

// ---------------------------------------------------------------------
// Discrete LOD level from one representative root-sphere projection.
// (No DAG walk — same band math as the selector's object-space cut.)
// ---------------------------------------------------------------------

fn instance_max_linear_scale(slot: u32) -> f32 {
    let m = cluster_instance_transforms[slot];
    let lx = length(vec3<f32>(m[0].x, m[1].x, m[2].x));
    let ly = length(vec3<f32>(m[0].y, m[1].y, m[2].y));
    let lz = length(vec3<f32>(m[0].z, m[1].z, m[2].z));
    return max(max(lx, ly), lz);
}

fn apply_affine(slot: u32, p: vec3<f32>) -> vec3<f32> {
    let m = cluster_instance_transforms[slot];
    return vec3<f32>(
        dot(m[0].xyz, p) + m[0].w,
        dot(m[1].xyz, p) + m[1].w,
        dot(m[2].xyz, p) + m[2].w,
    );
}

fn view_focal_px() -> f32 {
    return view.viewport.w * 0.5 * view.clip_from_view[1][1];
}

// Object-space error budget reproducing the screen-space cut at this
// instance's distance, quantized to a discrete band. Smaller level =
// finer. Per-geometry clamp to the root's coarsest error so far
// instances of a geometry all collapse to its coarsest level.
fn classify_level(slot: u32, root_group: u32) -> u32 {
    let root = cluster_groups[root_group];
    let scale = instance_max_linear_scale(slot);
    let focal_px = view_focal_px();
    let world_center = apply_affine(slot, root.traversal_sphere.xyz);
    let world_radius = root.traversal_sphere.w * scale;
    // `cluster_instance_transforms` is the ORIGIN-RELATIVE world (origin = the
    // camera itself), so the camera sits at (0,0,0) in this space and distance
    // is just the center's length. Subtracting `view.world_position` (ABSOLUTE)
    // would mix spaces: far from the world origin every instance classifies as
    // origin-distance away (coarsest band), and camera motion spuriously
    // crosses bands (shared-BLAS rebuild storms).
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
    let hi = clamp(band_max_geom, 0.0, f32(params.max_band));
    let band = clamp(band_f, 0.0, hi);
    return u32(band);
}

fn band_error_budget(level: u32) -> f32 {
    return exp2(params.e_min_log2 + f32(level) / params.inv_log2_ratio);
}

// =====================================================================
// Pass 0: per-frame reset of the per-geometry scratch.
// =====================================================================
@compute @workgroup_size(64)
fn geom_reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if g >= params.geometry_count {
        return;
    }
    atomicStore(&geometry_desired_level[g], NO_LEVEL);
    geometry_dirty[g] = 0u;
}

// =====================================================================
// Pass 1: classify each instance → discrete level; reduce per geometry;
// stash the geometry's static descriptor.
// =====================================================================
@compute @workgroup_size(64)
fn classify(
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
    let lod = cluster_instance_lod_inputs[slot];
    let level = classify_level(slot, lod.root_group);
    atomicMin(&geometry_desired_level[geom], level);
    // Static per-geometry descriptor (identical for every instance of
    // this geometry — last writer wins, all writes equal).
    geometry_desc[geom] = vec4<u32>(lod.group_base, lod.cluster_base, lod.cluster_count, lod.root_group);
}

// =====================================================================
// Pass 2: per geometry — decide rebuild, append to the dirty build list.
// =====================================================================
@compute @workgroup_size(64)
fn elect_dirty(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if g >= params.geometry_count {
        return;
    }
    let desired = atomicLoad(&geometry_desired_level[g]);
    if desired == NO_LEVEL {
        return; // no visible instance this frame
    }
    // CLAS bytes not resident yet → electing now would build a hollow BLAS and
    // bake a zero-extent leaf into the TLAS. Wait; re-checked every frame.
    if clas_ready[g] == 0u {
        return;
    }
    if desired == geometry_built_level[g] {
        return; // resident BLAS already at the wanted level
    }
    // Rebuild needed. Claim a dirty-build slot. `built_level` is NOT committed
    // here: the selector/blas_rebuild consumers have their own cold-start bails,
    // and an optimistic commit over a bailed build leaves a permanent lie
    // ("built") over unbuilt pool bytes — the missing-static-scene startup race.
    // `commit_built` below stamps it only after the build chain actually records.
    let i = atomicAdd(&dirty_count[0], 1u);
    if i >= params.geometry_capacity {
        return; // overflow (shouldn't happen — capacity == geometry_count)
    }
    geometry_dirty[g] = 1u;
    dirty_gid[i] = g;

    let desc = geometry_desc[g]; // (group_base, cluster_base, cluster_count, root_group)
    let e_build = band_error_budget(desired);
    bucket_desc[i * DESC_VEC4] = desc;
    bucket_desc[i * DESC_VEC4 + 1u] = vec4<u32>(bitcast<u32>(e_build), desired, 0u, 0u);
    bucket_dst_addresses[i] = geometry_address(g);
}

// =====================================================================
// Pass 3: clamp the dirty count to capacity for the build srcInfosCount.
// =====================================================================
@compute @workgroup_size(1)
fn finalize_count() {
    build_count[0] = min(atomicLoad(&dirty_count[0]), params.geometry_capacity);
}

// =====================================================================
// Pass 4: per instance — assign its geometry's stable BLAS address.
// Pure function of geometry_id, independent of the build, so stable
// across frames unless the instance's geometry changes.
// =====================================================================
@compute @workgroup_size(64)
fn assign_address(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    // Flat index across a 2D-split dispatch (X capped at 65535, rest in Y).
    let d = gid.x + gid.y * num_workgroups.x * 64u;
    if d >= params.active_count {
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
@compute @workgroup_size(64)
fn commit_built(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= min(atomicLoad(&dirty_count[0]), params.geometry_capacity) {
        return;
    }
    // An empty cut (selector emitted no clusters — CLAS data not resident yet)
    // must NOT commit: the pool holds a hollow BLAS. Leave built_level untouched
    // so the bucket re-elects until a real cut lands.
    if build_args[i].x == 0u {
        return;
    }
    geometry_built_level[dirty_gid[i]] = bucket_desc[i * DESC_VEC4 + 1u].y;
}
