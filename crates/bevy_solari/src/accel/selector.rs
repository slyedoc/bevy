#![allow(unsafe_code, reason = "raw heap-kernel dispatch via as_hal_mut")]

//! Per-frame cluster LOD-selection compute pass — **per bucket**. All I/O
//! buffers are sparse-allocated and grow by page commit, not reallocation.
//!
//! BLAS sharing (`super::blas_sharing`) groups instances into
//! `(geometry, LOD-band)` buckets upstream. This pass runs one workgroup
//! per bucket and emits an **object-space** DAG cut for that bucket's
//! geometry at the band's fixed error budget — so every instance in the
//! bucket shares one BLAS. The cut depends only on `(geometry, band)`,
//! never on a single instance (see `selector.slang`).
//!
//! Inputs (set 0 — the cluster-scene heap surface, constant-offset mapped
//! from the [`ClusterSceneBindGroup`] mirror):
//! - Mesh pool (clusters, groups, child table).
//!
//! Inputs (set 1 — selector I/O, push-index mapped):
//! - `cluster_clas_addresses` — global per-cluster CLAS device-address
//!   table (filled by [`crate::scene::clas_arena`]).
//! - `bucket_desc` / `live_bucket_count` — from
//!   [`crate::scene::blas_sharing`] (which bucket has which geometry +
//!   error budget, and how many buckets are live).
//! - `params` push block — strides + ref-buffer base address.
//!
//! Outputs (set 1), all **bucket-indexed**:
//! - `selected_clas_refs` — per-bucket contiguous ref lists.
//! - `args_buf` — per-bucket
//!   `VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV`
//!   records ready for the BLAS rebuild pass.
//! - `per_bucket_counts` — atomic counters (reset each frame by
//!   `select_reset` before the main pass).
//!
//! See `selector.slang` for the full algorithm.

use ash::vk;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::renderer::{RenderContext, RenderDevice};
use bytemuck::{Pod, Zeroable};
use wgpu::hal::api::Vulkan as VkApi;

use crate::bindings::ClusterSceneBindGroup;
use crate::instance::InstanceManager;

use super::blas_sharing::BlasSharing;
use crate::geometry::ClasArena;
use crate::gpu::allocator::{Allocator, SparseBuffer};
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};

/// User-facing selector knobs. Consumed by the BLAS-sharing classify pass
/// (which owns the screen-space projection); they live here because the
/// selector is the user-visible LOD resource.
#[derive(Resource, Copy, Clone, Debug)]
pub struct ClusterSelectorSettings {
    /// Coarsest LOD whose projected world-space error is ≤ this many
    /// screen pixels gets picked. 1.0 = pixel-perfect; higher values
    /// cull more aggressively.
    pub pixel_error_threshold: f32,
    /// Floor on `dist(camera, group_sphere_surface)` in the
    /// projection formula — prevents division blow-up when the
    /// camera is inside a group's traversal sphere.
    pub near_distance: f32,
}

impl Default for ClusterSelectorSettings {
    fn default() -> Self {
        Self {
            pixel_error_threshold: 1.0,
            near_distance: 0.1,
        }
    }
}

/// Push params shared with `selector.slang::SelectorParams` (16 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct SelectorParamsGpu {
    pub max_clusters_per_bucket: u32,
    pub bucket_capacity: u32,
    pub selected_clas_refs_addr_lo: u32,
    pub selected_clas_refs_addr_hi: u32,
}

const _: () = assert!(size_of::<SelectorParamsGpu>() == 16);

/// The Slang modules `selector.slang` imports (referenced by the compile
/// test too).
pub(crate) const SELECTOR_MODULES: &[(&str, &str)] = &[(
    "cluster_bindings",
    include_str!("../bindings/cluster_bindings.slang"),
)];

/// The selector's two heap kernels + the persistent slot table its set-1
/// parameters are written through. Built lazily on the first dispatch with
/// the cluster-scene heap mirror present (the mapping table bakes those
/// constant slot offsets).
pub struct SelectorKernels {
    pub reset: HeapKernel,
    pub main: HeapKernel,
    /// One slot per set-1 buffer, indexed by [`selector_slot_table`]'s order.
    pub slots: KernelSlots,
}

/// Render-world resource for the cluster LOD-selection compute pass —
/// per-frame frame-state only (sparse I/O buffers + the `SelectorParams`
/// push mirror + the lazily-built heap kernels). All buffers are
/// bucket-indexed.
#[derive(Resource)]
pub struct Selector {
    /// Sparse `array<vec2<u32>>` of per-bucket ref lists.
    pub selected_clas_refs: SparseBuffer,
    /// Sparse `array<vec4<u32>>` of per-bucket BLAS-build inputs.
    pub args_buf: SparseBuffer,
    /// Sparse `array<atomic<u32>>` of per-bucket emit counters.
    pub per_bucket_counts: SparseBuffer,
    /// Per-frame push params, filled in `Render::Prepare`.
    pub params: SelectorParamsGpu,
    /// The heap kernels, built lazily (see [`SelectorKernels`]).
    pub kernels: Option<SelectorKernels>,
    /// True iff [`dispatch_selector`] actually recorded this frame — the
    /// downstream BLAS build (and its `commit_built`) gate on it, so a
    /// cold-start bail here can't be papered over (rebuild-until-built).
    pub recorded: bool,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for Selector {
    fn drop(&mut self) {
        if let Some(kernels) = self.kernels.take() {
            self._device_keepalive.quiesce_before_raw_destroy();
            // SAFETY: quiesced; handles exclusively owned here.
            unsafe {
                kernels.reset.destroy(&self.raw_device);
                kernels.main.destroy(&self.raw_device);
            }
        }
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for Selector {}
unsafe impl Sync for Selector {}

/// `RenderStartup`: allocate the selector's sparse I/O buffers + insert
/// the [`Selector`] resource. No-op when the raw-VK [`Allocator`] is
/// absent — downstream selector systems guard on the resource's presence.
pub fn init_selector(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
) {
    let Some(allocator) = allocator else {
        return;
    };
    // Virtual address space for the selector's per-frame I/O buffers.
    // Capped at 1 GB each — wgpu enforces `max_storage_buffer_binding_size`
    // (~2 GB on desktop adapters) regardless of sparse commit footprint.
    // 1 GB / 8 B = 128 M slots for refs (≈ 64 K buckets × 2 K clusters),
    // 1 GB / 16 B = 64 M bucket args, 1 GB / 4 B = 256 M counts. All
    // sparse-backed; physical RAM scales with the live bucket count.
    const SELECTED_REFS_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;
    const ARGS_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;
    const COUNTS_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;

    let selected_clas_refs = allocator.create_sparse_buffer(
        &render_device,
        // BLAS_INPUT in wgpu maps to AS_BUILD_INPUT_READ_ONLY +
        // SHADER_DEVICE_ADDRESS — the BLAS rebuild reads this buffer
        // by device address via the `cluster_references` field of
        // each per-bucket args record.
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        SELECTED_REFS_VIRTUAL_BYTES,
        "selector.selected_clas_refs",
    );
    let args_buf = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        ARGS_VIRTUAL_BYTES,
        "selector.args_buf",
    );
    let per_bucket_counts = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        wgpu::BufferUsages::STORAGE,
        COUNTS_VIRTUAL_BYTES,
        "selector.per_bucket_counts",
    );

    commands.insert_resource(Selector {
        selected_clas_refs,
        args_buf,
        per_bucket_counts,
        params: SelectorParamsGpu::default(),
        kernels: None,
        recorded: false,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

/// `Render::Prepare`: fill the per-frame `SelectorParams` push mirror +
/// commit sparse pages on the I/O buffers (sized on the bucket
/// capacity, not instance count).
pub fn prepare_selector_params(
    mut selector: Option<ResMut<Selector>>,
    instances: Option<Res<InstanceManager>>,
    sharing: Option<Res<BlasSharing>>,
) {
    let (Some(resources), Some(instances), Some(sharing)) =
        (selector.as_deref_mut(), instances, sharing)
    else {
        return;
    };

    if instances.active_count() == 0 {
        return;
    }
    let bucket_capacity = sharing.build_entry_capacity();
    let selected_refs_addr = resources.selected_clas_refs.address;

    // Worst-case clusters in any one bucket's BLAS. Shared with
    // `blas_rebuild`'s build sizing + `blas_sharing`'s region stride via
    // the single source `InstanceManager::max_clusters_per_bucket`.
    let max_per = instances.max_clusters_per_bucket();

    resources.params = SelectorParamsGpu {
        max_clusters_per_bucket: max_per,
        bucket_capacity,
        selected_clas_refs_addr_lo: (selected_refs_addr & 0xFFFF_FFFF) as u32,
        selected_clas_refs_addr_hi: (selected_refs_addr >> 32) as u32,
    };

    let cap = bucket_capacity as u64;
    let max_per = max_per as u64;
    resources.selected_clas_refs.commit(0..cap * max_per * 8);
    resources.args_buf.commit(0..cap * 16);
    resources.per_bucket_counts.commit(0..cap * 4);
}

/// Write the selector's set-1 descriptors into `slots` and return the
/// `(parameter name, heap slot)` pairs both entries' push blobs are
/// assembled from (each entry filters to the bindings surviving in its own
/// SPIR-V).
fn selector_slot_table<'a>(
    seam: &BindingSeam,
    slots: &KernelSlots,
    selector: &Selector,
    clas_arena: &ClasArena,
    sharing: &BlasSharing,
) -> Vec<(&'a str, u32)> {
    vec![
        (
            "cluster_clas_addresses",
            slots.buffer(seam, 0, &clas_arena.cluster_clas_addresses.wgpu_buffer),
        ),
        (
            "selected_clas_refs",
            slots.buffer(seam, 1, &selector.selected_clas_refs.wgpu_buffer),
        ),
        ("args_buf", slots.buffer(seam, 2, &selector.args_buf.wgpu_buffer)),
        (
            "per_bucket_counts",
            slots.buffer(seam, 3, &selector.per_bucket_counts.wgpu_buffer),
        ),
        ("bucket_desc", slots.buffer(seam, 4, &sharing.build_desc)),
        (
            "live_bucket_count",
            slots.buffer(seam, 5, &sharing.dirty_build_count),
        ),
    ]
}

/// `Render::Render`: dispatch the selector — one workgroup per bucket.
/// The live bucket count is GPU-side, so we dispatch the full bucket
/// capacity and let `select_main` early-out past `live_bucket_count`.
/// A raw heap dispatch: buffer slots rewritten per dispatch, params +
/// slot array in push data.
pub fn dispatch_selector(
    selector: Option<ResMut<Selector>>,
    seam: Option<Res<BindingSeam>>,
    scene_bind_group: Res<ClusterSceneBindGroup>,
    instances: Option<Res<InstanceManager>>,
    clas_arena: Option<Res<ClasArena>>,
    sharing: Option<Res<BlasSharing>>,
    mut ctx: RenderContext,
) {
    let (Some(selector), Some(seam), Some(instances), Some(clas_arena), Some(sharing)) =
        (selector, seam, instances, clas_arena, sharing)
    else {
        return;
    };
    let selector = selector.into_inner();
    // Cleared every frame; only an actually-recorded dispatch below sets it.
    selector.recorded = false;
    if instances.active_count() == 0 {
        return;
    }
    let bucket_capacity = sharing.build_entry_capacity();
    if bucket_capacity == 0 {
        return;
    }
    // The cluster-scene heap mirror is the set-0 surface (and doubles as the
    // scene-ready signal — absent until meshes + instance columns exist).
    let (Some(_), Some(heap_slots)) = (
        scene_bind_group.bind_group.as_ref(),
        scene_bind_group.heap_slots.as_ref(),
    ) else {
        return;
    };
    // Built lazily on the first ready frame: the mapping table bakes the
    // cluster-scene heap slots, which exist only once the mirror has run
    // (the slots are allocated once and rewritten in place, so the table
    // never goes stale).
    if selector.kernels.is_none() {
        let base = crate::gpu::rt_pipeline::cluster_heap_mappings(&seam, heap_slots);
        let params_size = size_of::<SelectorParamsGpu>() as u32;
        let make = |entry: &str, label: &str| {
            HeapKernel::new_with_mappings(
                &seam,
                "selector.slang",
                include_str!("selector.slang"),
                entry,
                SELECTOR_MODULES,
                &[],
                &[],
                label,
                params_size,
                &base,
            )
        };
        let (Some(reset), Some(main)) = (
            make("select_reset", "selector_reset"),
            make("select_main", "selector_main"),
        ) else {
            return;
        };
        selector.kernels = Some(SelectorKernels {
            reset,
            main,
            slots: KernelSlots::new(&seam, 6),
        });
    }
    let kernels = selector.kernels.as_ref().unwrap();

    let named = selector_slot_table(&seam, &kernels.slots, selector, &clas_arena, &sharing);
    let filtered = |kernel: &HeapKernel| -> Vec<(&str, u32)> {
        named
            .iter()
            .filter(|(name, _)| kernel.bindings.iter().any(|(n, _)| n == name))
            .copied()
            .collect()
    };
    let params = bytemuck::bytes_of(&selector.params);
    let reset_blob = kernels
        .reset
        .push_blob("selector_reset", params, &filtered(&kernels.reset));
    let main_blob = kernels
        .main
        .push_blob("selector_main", params, &filtered(&kernels.main));

    let encoder = ctx.command_encoder();
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // barriers bracket the dispatches against the surrounding compute (raw
    // dispatches are invisible to wgpu's tracking).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &selector.raw_device;
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER,
                )
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)];
            let dep = vk::DependencyInfo::default().memory_barriers(&barrier);
            seam.bind_heaps(cb);
            // BLAS sharing's bucket writes (+ any transfer) -> our reads.
            dev.cmd_pipeline_barrier2(cb, &dep);
            seam.push_data(cb, &reset_blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernels.reset.pipeline);
            dev.cmd_dispatch(cb, bucket_capacity.div_ceil(64), 1, 1);
            // Counter reset -> main's atomic appends.
            dev.cmd_pipeline_barrier2(cb, &dep);
            seam.push_data(cb, &main_blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernels.main.pipeline);
            // One workgroup per bucket. `bucket_capacity ≤ MAX_GEOMETRIES ≤
            // 65535`, so a single dispatch dimension suffices; the
            // `select_main` guard early-outs buckets past `live_bucket_count`.
            dev.cmd_dispatch(cb, bucket_capacity, 1, 1);
            // Our ref/args/count writes -> downstream compute readers
            // (`commit_built`); the BLAS build's own seam covers build input.
            dev.cmd_pipeline_barrier2(cb, &dep);
        });
    }
    selector.recorded = true;
}
