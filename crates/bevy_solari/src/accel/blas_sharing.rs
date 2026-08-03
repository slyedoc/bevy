// BLAS sharing — ONE shared BLAS per geometry, built at a single
// per-geometry discrete LOD level (NVIDIA "BLAS merging" applied to a
// resident engine). All instances of a `ClusterMesh` reference the same
// BLAS at `geometry_blas_pool.base + geometry_id * stride`, stable for
// the geometry's life. The BLAS count is bounded by the resident
// geometry universe — scene-stable, view-independent — and a geometry
// only rebuilds when its chosen LOD level changes.
//
// This module owns the classify / dirty-election / address-assignment
// GPU passes and the per-geometry BLAS storage pool. The selector and
// `blas_rebuild` consume the dirty-build list here (they treat a "bucket"
// as a dirty-geometry build entry); `ptlas` reads the slot-indexed
// `instance_blas_address`.
//
// See `blas_sharing.slang` for the pass algorithms.
#![allow(unsafe_code, reason = "device-address plumbing + raw heap-kernel dispatch")]

use ash::vk;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{Buffer, RawBufferVec},
    renderer::{RenderContext, RenderDevice, RenderQueue},
    view::ExtractedView,
};
use bytemuck::{Pod, Zeroable};
use wgpu::hal::api::Vulkan as VkApi;

use crate::bindings::ClusterSceneBindGroup;
use crate::ecs_gpu::GpuColumn;
use crate::instance::{GeometryIdColumn, InstanceManager};
use crate::geometry::ClusterMeshManager;

use crate::gpu::allocator::{Allocator, SparseBuffer};
use super::blas_rebuild::{query_blas_size, BLAS_REGION_ALIGN};
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::extension::ClusterExtensionFns;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
use super::selector::ClusterSelectorSettings;

/// Max distinct resident geometries. The geometry-indexed scratch
/// buffers are fixed at this size (cheap: a few MB total), and the
/// selector's 1-workgroup-per-dirty-entry dispatch stays within Vulkan's
/// 65535 per-dim cap. (The selector / blas_rebuild treat each dirty-build
/// entry as a "bucket" — a build entry never exceeds the geometry count.)
pub const MAX_GEOMETRIES: u32 = 65535;
/// Sentinel: geometry never built / not wanted (mirrors `NO_LEVEL`).
const NO_LEVEL: u32 = 0xFFFFFFFF;
/// Global LOD-band ceiling (classify also clamps per-geometry).
const GLOBAL_MAX_BAND: u32 = 127;
/// LOD-band geometric ratio. Finer = less popping, more rebuilds on
/// camera motion. One level per geometry, so this is cheap to keep fine.
const BAND_RATIO: f32 = 1.25;
/// `log2(E_MIN)` — object-space error budget at band 0's lower edge.
const E_MIN_LOG2: f32 = -10.0;
/// Cluster-count cap the per-geometry BLAS region stride is sized for
/// when no mesh is resident yet (grown to the real max as meshes load).
const MIN_STRIDE_CLUSTERS: u32 = 1;

/// Virtual span of the per-geometry BLAS pool — 16 GB, sparse-backed.
const GEOMETRY_BLAS_POOL_VIRTUAL_BYTES: u64 = 16 * 1024 * 1024 * 1024;
/// Virtual span for the slot-indexed `instance_blas_address` (1 GB).
const SLOT_BUFFER_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;
/// Virtual span for the geometry-indexed dst-address table.
const GEOMETRY_DST_VIRTUAL_BYTES: u64 = 64 * 1024 * 1024;

/// Push mirror of `blas_sharing.slang::SharingParams` (64 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct SharingParamsGpu {
    pub active_count: u32,
    pub geometry_count: u32,
    pub geometry_capacity: u32,
    pub max_band: u32,
    pub e_min_log2: f32,
    pub inv_log2_ratio: f32,
    pub pixel_error_threshold: f32,
    pub near_distance: f32,
    pub pool_base_lo: u32,
    pub pool_base_hi: u32,
    pub geometry_stride: u32,
    /// The camera's focal length in pixels — `viewport_height/2 · P[1][1]`,
    /// the only View input the classify projection reads. Filled per
    /// dispatch from [`ExtractedView`] (the same camera the old
    /// dynamic-offset `View` uniform selected).
    pub focal_px: f32,
    /// X workgroup count of the per-instance 2D-split dispatches (filled
    /// per dispatch).
    pub groups_x: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(size_of::<SharingParamsGpu>() == 64);

/// The Slang modules `blas_sharing.slang` imports (referenced by the
/// compile test too).
pub(crate) const BLAS_SHARING_MODULES: &[(&str, &str)] = &[(
    "cluster_bindings",
    include_str!("../bindings/cluster_bindings.slang"),
)];

/// Render-world resource for the per-geometry BLAS-sharing pass.
///
/// Bundles everything the pass owns — the persistent per-geometry state,
/// the per-frame I/O buffers, the five compute pipelines + bind-group
/// layout, and the per-frame bind group.
#[derive(Resource)]
pub struct BlasSharing {
    /// Per-geometry cluster-BLAS storage (EXPLICIT_DESTINATIONS target).
    /// Geometry `g`'s BLAS lives at `address + g * worst_case_stride`.
    pub geometry_blas_pool: SparseBuffer,
    /// Frozen worst-case single-BLAS byte size (region stride). Grows as
    /// bigger meshes stream in; per-geometry rebuild absorbs the move.
    pub worst_case_stride: u64,

    /// geometry → finest LOD level any visible instance wants (atomicMin,
    /// reset each frame by `geom_reset`).
    pub geometry_desired_level: Buffer,
    /// geometry → LOD level its resident BLAS was built at. PERSISTENT;
    /// init to NO_LEVEL so first sighting rebuilds.
    pub geometry_built_level: Buffer,
    /// geometry → 1 if rebuilt this frame (PTLAS re-writes its instances).
    pub geometry_dirty: Buffer,
    /// geometry → static descriptor (group_base, cluster_base,
    /// cluster_count, root_group). Written by classify.
    pub geometry_desc: Buffer,
    /// geometry → 1 once its CLAS bytes actually exist (CPU-written by the
    /// static CLAS upload / procedural instantiate). `elect_dirty` refuses to
    /// elect before this: electing over absent CLAS builds a hollow BLAS that
    /// gets committed and baked into the TLAS as a zero-extent leaf — the
    /// intermittent missing-static-scene startup race.
    pub clas_ready: Buffer,
    /// geometry → flag bits (bit 0 = has baked OMM): set once the geometry's
    /// CLASes were built with an opacity micromap attached (CPU-written at
    /// CLAS upload, next to `clas_ready`). The PTLAS fill reads it per instance:
    /// an OMM'd geometry lets the micromap drive traversal, while an alpha-tested
    /// material without one keeps `FORCE_NO_OPAQUE` (shader alpha test).
    pub geometry_flags: Buffer,
    /// dirty entry i → geometry id.
    pub dirty_gid: Buffer,
    /// dirty build count (atomic alloc). The selector guards its
    /// per-entry dispatch on this.
    pub dirty_build_count: Buffer,
    /// Clamped build count = min(dirty_count, capacity). The BLAS build
    /// reads this as `srcInfosCount`.
    pub build_count: Buffer,
    /// dirty entry i → build descriptor (selector "build_desc" format).
    pub build_desc: Buffer,
    /// dirty entry i → geometry's stable BLAS address (build dst array).
    pub geometry_dst_addresses: SparseBuffer,
    /// slot-indexed: instance → BLAS device address (PTLAS reads this).
    pub instance_blas_address: SparseBuffer,
    /// slot-indexed: instance → per-instance object-space LOD error budget,
    /// written by `classify` and consumed by the shared-BLAS DAG cut.
    pub instance_e_build: SparseBuffer,
    /// dense active index → real `GpuEntity`. Re-uploaded only when
    /// the active set changes.
    pub active_to_slot: RawBufferVec<u32>,
    /// Per-frame push params, filled in `Render::Prepare`; `focal_px` /
    /// `groups_x` are stamped per dispatch.
    pub params: SharingParamsGpu,

    /// CPU-tracked geometry capacity (== resident geometry high-water,
    /// clamped to `MAX_GEOMETRIES`). Private — read it through
    /// [`BlasSharing::build_entry_capacity`] so the clamp has one owner.
    geometry_count: u32,
    /// One-time init of the persistent `geometry_built_level` to NO_LEVEL.
    needs_init: bool,
    /// Slots / geometries whose freshly committed table pages were zeroed —
    /// sparse pages are UNDEFINED on commit, and these tables hold device
    /// addresses AS builds dereference (garbage → MMU fault → device loss).
    cleared_slots: u64,
    cleared_geoms: u64,

    /// The heap kernels, built lazily on the first ready dispatch (see
    /// [`SharingKernels`]).
    pub kernels: Option<SharingKernels>,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for BlasSharing {
    fn drop(&mut self) {
        if let Some(kernels) = self.kernels.take() {
            self._device_keepalive.quiesce_before_raw_destroy();
            // SAFETY: quiesced; handles exclusively owned here.
            unsafe {
                for kernel in kernels.all() {
                    kernel.destroy(&self.raw_device);
                }
            }
        }
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for BlasSharing {}
unsafe impl Sync for BlasSharing {}

/// The six BLAS-sharing heap kernels + the persistent slot table their set-1
/// parameters are written through. `commit_built` is dispatched from
/// [`super::blas_rebuild::dispatch_blas_rebuild`], after the raw BLAS build
/// records; the other five run as one chain in [`dispatch_blas_sharing`].
pub struct SharingKernels {
    pub geom_reset: HeapKernel,
    pub classify: HeapKernel,
    pub elect_dirty: HeapKernel,
    pub finalize_count: HeapKernel,
    pub assign_address: HeapKernel,
    pub commit_built: HeapKernel,
    /// One slot per set-1 buffer, indexed by [`sharing_slot_table`]'s order.
    pub slots: KernelSlots,
}

impl SharingKernels {
    fn all(&self) -> [&HeapKernel; 6] {
        [
            &self.geom_reset,
            &self.classify,
            &self.elect_dirty,
            &self.finalize_count,
            &self.assign_address,
            &self.commit_built,
        ]
    }
}

impl BlasSharing {
    /// The number of live BLAS-build entries this frame — i.e. the
    /// resident geometry count, already clamped to `[1, MAX_GEOMETRIES]` when
    /// assigned in `prepare_blas_sharing`.
    ///
    /// **Single source of truth** for the dispatch dimension / sparse
    /// commit sizing the selector, `blas_rebuild`, and PTLAS read. Call
    /// this instead of re-applying `.min(MAX_GEOMETRIES)` at each site.
    #[inline]
    pub fn build_entry_capacity(&self) -> u32 {
        debug_assert!(
            self.geometry_count <= MAX_GEOMETRIES,
            "geometry_count must be clamped at its assignment in prepare_blas_sharing"
        );
        self.geometry_count
    }
}

fn storage_buf(render_device: &RenderDevice, label: &'static str, size: u64) -> Buffer {
    render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// `RenderStartup`: allocate the BLAS-sharing buffers + insert the
/// [`BlasSharing`] resource. No-op when the raw-VK [`Allocator`] is
/// absent — downstream sharing systems guard on the resource's presence.
pub fn init_blas_sharing(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
) {
    let Some(allocator) = allocator else {
        return;
    };

    let geometry_blas_pool = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        wgpu::BufferUsages::COPY_DST,
        GEOMETRY_BLAS_POOL_VIRTUAL_BYTES,
        "blas_sharing.geometry_blas_pool",
    );
    let geometry_dst_addresses = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::TRANSFER_DST,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        GEOMETRY_DST_VIRTUAL_BYTES,
        "blas_sharing.geometry_dst_addresses",
    );
    let instance_blas_address = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        SLOT_BUFFER_VIRTUAL_BYTES,
        "blas_sharing.instance_blas_address",
    );
    let instance_e_build = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        SLOT_BUFFER_VIRTUAL_BYTES,
        "blas_sharing.instance_e_build",
    );

    let cap = MAX_GEOMETRIES as u64;
    let geometry_desired_level = storage_buf(&render_device, "blas_sharing.desired_level", cap * 4);
    let geometry_built_level = storage_buf(&render_device, "blas_sharing.built_level", cap * 4);
    let geometry_dirty = storage_buf(&render_device, "blas_sharing.geometry_dirty", cap * 4);
    let geometry_desc = storage_buf(&render_device, "blas_sharing.geometry_desc", cap * 16);
    let clas_ready = storage_buf(&render_device, "blas_sharing.clas_ready", cap * 4);
    let geometry_flags = storage_buf(&render_device, "blas_sharing.geometry_flags", cap * 4);
    let dirty_gid = storage_buf(&render_device, "blas_sharing.dirty_gid", cap * 4);
    let dirty_build_count = storage_buf(&render_device, "blas_sharing.dirty_count", 4);
    let build_desc = storage_buf(&render_device, "blas_sharing.build_desc", cap * 2 * 16);
    let build_count = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("blas_sharing.build_count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut active_to_slot = RawBufferVec::<u32>::new(wgpu::BufferUsages::STORAGE);
    active_to_slot.set_label(Some("blas_sharing.active_to_slot"));

    commands.insert_resource(BlasSharing {
        geometry_blas_pool,
        worst_case_stride: 0,
        geometry_desired_level,
        geometry_built_level,
        geometry_dirty,
        geometry_desc,
        clas_ready,
        geometry_flags,
        dirty_gid,
        dirty_build_count,
        build_count,
        build_desc,
        geometry_dst_addresses,
        instance_blas_address,
        instance_e_build,
        active_to_slot,
        params: SharingParamsGpu::default(),
        geometry_count: 1,
        needs_init: true,
        cleared_slots: 0,
        cleared_geoms: 0,
        kernels: None,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

/// `Render::Prepare`: size the region stride (grow-safe), grow geometry
/// capacity, fill params, refresh `active_to_slot`, zero per-frame
/// counters, init persistent built-level once, commit sparse pages.
pub fn prepare_blas_sharing(
    mut resources: Option<ResMut<BlasSharing>>,
    instances: Option<Res<InstanceManager>>,
    cluster_meshes: Option<Res<ClusterMeshManager>>,
    fns: Option<Res<ClusterExtensionFns>>,
    settings: Res<ClusterSelectorSettings>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(resources), Some(instances), Some(cluster_meshes), Some(fns)) =
        (resources.as_deref_mut(), instances, cluster_meshes, fns)
    else {
        return;
    };
    let Some(cluster_fns) = fns.cluster.as_ref() else {
        return;
    };
    let active_count = instances.active_count() as u32;
    if active_count == 0 {
        return;
    }

    // Per-geometry BLAS region stride = worst-case single-mesh BLAS,
    // sized from the largest resident mesh (monotonic, grow-only). A
    // stride change moves every geometry's region — fine, the affected
    // geometries simply rebuild (built_level mismatch after a reset).
    // Same per-bucket cluster cap the selector / rebuild use (single
    // source), floored to MIN_STRIDE_CLUSTERS so the region is never
    // degenerately small.
    let max_per = instances
        .max_clusters_per_bucket()
        .max(MIN_STRIDE_CLUSTERS);
    let want_stride = query_blas_size(cluster_fns, max_per)
        .max(BLAS_REGION_ALIGN)
        .next_multiple_of(BLAS_REGION_ALIGN);
    let stride_changed = want_stride > resources.worst_case_stride;
    resources.worst_case_stride = resources.worst_case_stride.max(want_stride);
    if stride_changed {
        tracing::debug!(
            "blas_sharing: worst_case_stride -> {} bytes (max_per={max_per} clusters, omm-aware)",
            resources.worst_case_stride,
        );
    }

    // Geometry capacity == resident geometry high-water (bounded set).
    let geometry_count = cluster_meshes.geometry_count().min(MAX_GEOMETRIES);
    resources.geometry_count = geometry_count.max(1);
    let geometry_capacity = MAX_GEOMETRIES; // fixed-size scratch buffers

    // Reset built levels to NO_LEVEL on the one-time init of the persistent
    // table (so every geometry's first sighting triggers a build) and whenever
    // the stride grows (every geometry's region address moved → full rebuild).
    if resources.needs_init || stride_changed {
        let init = vec![NO_LEVEL; MAX_GEOMETRIES as usize];
        render_queue.write_buffer(
            &resources.geometry_built_level,
            0,
            bytemuck::cast_slice(&init),
        );
        resources.needs_init = false;
    }

    let pool_base = resources.geometry_blas_pool.address;
    let stride = resources.worst_case_stride;
    // `focal_px` / `groups_x` are per-dispatch; the dispatches stamp them.
    resources.params = SharingParamsGpu {
        active_count,
        geometry_count,
        geometry_capacity,
        max_band: GLOBAL_MAX_BAND,
        e_min_log2: E_MIN_LOG2,
        inv_log2_ratio: 1.0 / BAND_RATIO.log2(),
        pixel_error_threshold: settings.pixel_error_threshold,
        near_distance: settings.near_distance,
        pool_base_lo: (pool_base & 0xFFFF_FFFF) as u32,
        pool_base_hi: (pool_base >> 32) as u32,
        geometry_stride: stride as u32,
        focal_px: 0.0,
        groups_x: 0,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    // Zero the per-frame dirty counters (desired_level + dirty flags are
    // reset GPU-side by `geom_reset`).
    render_queue.write_buffer(&resources.dirty_build_count, 0, &0u32.to_le_bytes());
    render_queue.write_buffer(&resources.build_count, 0, &0u32.to_le_bytes());

    // `active_to_slot` (dense → slot) only changes when the active set
    // changes — skip the rebuild + upload otherwise.
    let active_set_changed =
        !instances.added_slots().is_empty() || !instances.released_slots().is_empty();
    if active_set_changed || resources.active_to_slot.len() == 0 {
        resources.active_to_slot.clear();
        for slot in instances.active_slots() {
            resources.active_to_slot.push(slot.0);
        }
        if resources.active_to_slot.is_empty() {
            resources.active_to_slot.push(0);
        }
        resources
            .active_to_slot
            .write_buffer(&render_device, &render_queue);
    }

    // Commit sparse pages for the resident geometry set + active slots. Each BLAS
    // is <= stride, so geometry_count * stride tiles the regions without overlap.
    let high_water = instances.slot_high_water() as u64;
    resources
        .geometry_blas_pool
        .commit(0..(geometry_count as u64 * stride).max(1));
    resources
        .geometry_dst_addresses
        .commit(0..(geometry_count as u64 * 8).max(1));
    resources
        .instance_blas_address
        .commit(0..(high_water * 8).max(1));
    resources
        .instance_e_build
        .commit(0..(high_water * 4).max(1));

    // Zero newly committed regions: unwritten slots must read address 0 (the
    // PTLAS fill's null/inactive encoding), never undefined page contents —
    // an active instance whose geometry hasn't built yet otherwise hands the
    // partitioned-AS build a garbage BLAS address.
    let geoms = geometry_count as u64;
    if high_water > resources.cleared_slots || geoms > resources.cleared_geoms {
        let mut encoder = render_device.create_command_encoder(&Default::default());
        if high_water > resources.cleared_slots {
            let lo = resources.cleared_slots;
            encoder.clear_buffer(
                &resources.instance_blas_address.wgpu_buffer,
                lo * 8,
                Some((high_water - lo) * 8),
            );
            encoder.clear_buffer(
                &resources.instance_e_build.wgpu_buffer,
                lo * 4,
                Some((high_water - lo) * 4),
            );
            resources.cleared_slots = high_water;
        }
        if geoms > resources.cleared_geoms {
            let lo = resources.cleared_geoms;
            encoder.clear_buffer(
                &resources.geometry_dst_addresses.wgpu_buffer,
                lo * 8,
                Some((geoms - lo) * 8),
            );
            resources.cleared_geoms = geoms;
        }
        render_queue.submit([encoder.finish()]);
    }
}

const WORKGROUP_SIZE: u32 = 64;

/// Write the sharing set-1 descriptors into `slots` and return the
/// `(parameter name, heap slot)` pairs every sharing entry's push blob is
/// assembled from (each entry filters to the bindings surviving in its own
/// SPIR-V). Shared with `dispatch_blas_rebuild`'s `commit_built` dispatch,
/// so the name → slot-index assignment has one owner.
pub(crate) fn sharing_slot_table<'a>(
    seam: &BindingSeam,
    kernels: &SharingKernels,
    sharing: &BlasSharing,
    geometry_ids: &Buffer,
    active_to_slot: &Buffer,
    selector: &super::selector::Selector,
) -> Vec<(&'a str, u32)> {
    let slots = &kernels.slots;
    vec![
        ("active_to_slot", slots.buffer(seam, 0, active_to_slot)),
        ("instance_geometry_ids", slots.buffer(seam, 1, geometry_ids)),
        (
            "geometry_desired_level",
            slots.buffer(seam, 2, &sharing.geometry_desired_level),
        ),
        (
            "geometry_built_level",
            slots.buffer(seam, 3, &sharing.geometry_built_level),
        ),
        ("geometry_dirty", slots.buffer(seam, 4, &sharing.geometry_dirty)),
        ("dirty_count", slots.buffer(seam, 5, &sharing.dirty_build_count)),
        ("dirty_gid", slots.buffer(seam, 6, &sharing.dirty_gid)),
        ("bucket_desc", slots.buffer(seam, 7, &sharing.build_desc)),
        (
            "bucket_dst_addresses",
            slots.buffer(seam, 8, &sharing.geometry_dst_addresses.wgpu_buffer),
        ),
        (
            "instance_blas_address",
            slots.buffer(seam, 9, &sharing.instance_blas_address.wgpu_buffer),
        ),
        ("build_count", slots.buffer(seam, 10, &sharing.build_count)),
        ("geometry_desc", slots.buffer(seam, 11, &sharing.geometry_desc)),
        (
            "instance_e_build",
            slots.buffer(seam, 12, &sharing.instance_e_build.wgpu_buffer),
        ),
        ("build_args", slots.buffer(seam, 13, &selector.args_buf.wgpu_buffer)),
        ("clas_ready", slots.buffer(seam, 14, &sharing.clas_ready)),
    ]
}

/// A sharing entry's push blob: the shared params + the slot-table subset
/// surviving in that entry's SPIR-V.
pub(crate) fn sharing_push_blob(
    kernel: &HeapKernel,
    label: &str,
    params: &SharingParamsGpu,
    table: &[(&str, u32)],
) -> Vec<u8> {
    let named: Vec<(&str, u32)> = table
        .iter()
        .filter(|(name, _)| kernel.bindings.iter().any(|(n, _)| n == name))
        .copied()
        .collect();
    kernel.push_blob(label, bytemuck::bytes_of(params), &named)
}

/// `Render::Render`: geom_reset → classify → elect_dirty →
/// finalize_count → assign. Must run after the instance-column scatter
/// (reads `transforms`) and before the selector / `blas_rebuild`. A raw
/// heap-kernel chain: buffer slots rewritten per dispatch, params + slot
/// array in push data, explicit barriers between the dependent steps.
pub fn dispatch_blas_sharing(
    sharing: Option<ResMut<BlasSharing>>,
    seam: Option<Res<BindingSeam>>,
    scene_bind_group: Res<ClusterSceneBindGroup>,
    instances: Option<Res<InstanceManager>>,
    geometry_ids: Option<Res<GpuColumn<GeometryIdColumn>>>,
    selector: Option<Res<super::selector::Selector>>,
    view_query: bevy_ecs::system::Query<
        &ExtractedView,
        bevy_ecs::query::With<ExtractedCamera>,
    >,
    mut ctx: RenderContext,
) {
    let (Some(sharing), Some(seam), Some(instances), Some(geometry_ids), Some(selector)) =
        (sharing, seam, instances, geometry_ids, selector)
    else {
        return;
    };
    let sharing = sharing.into_inner();
    let active_count = instances.active_count() as u32;
    if active_count == 0 {
        return;
    }
    let geometry_count = sharing.build_entry_capacity();
    // The cluster-scene heap mirror is the set-0 surface (and doubles as the
    // scene-ready signal).
    let (Some(_), Some(heap_slots)) = (
        scene_bind_group.bind_group.as_ref(),
        scene_bind_group.heap_slots.as_ref(),
    ) else {
        return;
    };
    let Some(active_to_slot) = sharing.active_to_slot.buffer() else {
        return;
    };
    let Some(view) = view_query.iter().next() else {
        return;
    };
    // The one View input the classify projection reads (what the old
    // dynamic-offset `View` uniform supplied): focal length in pixels =
    // `viewport_height/2 · clip_from_view[1][1]`.
    let focal_px = view.viewport.w as f32 * 0.5 * view.clip_from_view.y_axis.y;

    // Built lazily on the first ready frame (see `SelectorKernels`' rationale).
    if sharing.kernels.is_none() {
        let base = crate::gpu::rt_pipeline::cluster_heap_mappings(&seam, heap_slots);
        let params_size = size_of::<SharingParamsGpu>() as u32;
        let make = |entry: &str| {
            HeapKernel::new_with_mappings(
                &seam,
                "blas_sharing.slang",
                include_str!("blas_sharing.slang"),
                entry,
                BLAS_SHARING_MODULES,
                &[],
                &[],
                &format!("blas_sharing_{entry}"),
                params_size,
                &base,
            )
        };
        let (
            Some(geom_reset),
            Some(classify),
            Some(elect_dirty),
            Some(finalize_count),
            Some(assign_address),
            Some(commit_built),
        ) = (
            make("geom_reset"),
            make("classify"),
            make("elect_dirty"),
            make("finalize_count"),
            make("assign_address"),
            make("commit_built"),
        ) else {
            return;
        };
        sharing.kernels = Some(SharingKernels {
            geom_reset,
            classify,
            elect_dirty,
            finalize_count,
            assign_address,
            commit_built,
            slots: KernelSlots::new(&seam, 15),
        });
    }
    let kernels = sharing.kernels.as_ref().unwrap();

    let table = sharing_slot_table(
        &seam,
        kernels,
        sharing,
        geometry_ids.buffer(),
        active_to_slot,
        &selector,
    );
    let active_groups = active_count.div_ceil(WORKGROUP_SIZE);
    let geom_groups = geometry_count.div_ceil(WORKGROUP_SIZE);
    // Per-instance passes (classify / assign_address) 2D-split past 65535
    // workgroups; the geometry passes stay 1D (no-op). The shaders that can
    // exceed the limit reconstruct the flat index from `params.groups_x`.
    let (agx, agy, agz) = crate::ecs_gpu::linear_dispatch(active_groups);
    let mut params = sharing.params;
    params.focal_px = focal_px;
    params.groups_x = agx;

    // The dependent chain: each step reads the previous step's writes.
    let steps: [(&HeapKernel, &str, (u32, u32, u32)); 5] = [
        (&kernels.geom_reset, "blas_sharing_geom_reset", (geom_groups, 1, 1)),
        (&kernels.classify, "blas_sharing_classify", (agx, agy, agz)),
        (&kernels.elect_dirty, "blas_sharing_elect_dirty", (geom_groups, 1, 1)),
        (&kernels.finalize_count, "blas_sharing_finalize_count", (1, 1, 1)),
        (&kernels.assign_address, "blas_sharing_assign_address", (agx, agy, agz)),
    ];
    let blobs: Vec<Vec<u8>> = steps
        .iter()
        .map(|(kernel, label, _)| sharing_push_blob(kernel, label, &params, &table))
        .collect();

    let encoder = ctx.command_encoder();
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // barriers bracket each step against its producers (raw dispatches are
    // invisible to wgpu's tracking).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &sharing.raw_device;
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER,
                )
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)];
            let dep = vk::DependencyInfo::default().memory_barriers(&barrier);
            seam.bind_heaps(cb);
            // Column scatter / gather writes + the counter-zeroing transfers
            // -> our reads; then one barrier per producer→consumer step.
            for ((kernel, _, (gx, gy, gz)), blob) in steps.iter().zip(&blobs) {
                dev.cmd_pipeline_barrier2(cb, &dep);
                seam.push_data(cb, blob);
                dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
                dev.cmd_dispatch(cb, *gx, *gy, *gz);
            }
            // Our writes (dirty list, addresses, counts) -> the selector /
            // PTLAS fill compute readers.
            dev.cmd_pipeline_barrier2(cb, &dep);
        });
    }
}
