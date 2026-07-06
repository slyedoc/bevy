// Selector dispatches a wgpu compute pipeline against the cluster
// scene bind group (group 0) and a selector-specific I/O bind group
// (group 1). All buffer-allocation knobs use the sparse-buffer
// allocator — no hardcoded byte caps; growth on demand.
#![allow(unsafe_code, reason = "compute dispatch reads cluster scene buffers via raw VK")]

//! Per-frame cluster LOD-selection compute pass — **per bucket**.
//!
//! BLAS sharing (`super::blas_sharing`) groups instances into
//! `(geometry, LOD-band)` buckets upstream. This pass runs one workgroup
//! per bucket and emits an **object-space** DAG cut for that bucket's
//! geometry at the band's fixed error budget — so every instance in the
//! bucket shares one BLAS. The cut depends only on `(geometry, band)`,
//! never on a single instance (see `selector.wgsl`).
//!
//! Inputs (`@group(0)` = `cluster_scene_bind_group`):
//! - Mesh pool (clusters, groups, child table).
//!
//! Inputs (`@group(1)` = selector I/O):
//! - `cluster_clas_addresses` — global per-cluster CLAS device-address
//!   table (filled by [`crate::scene::clas_arena`]).
//! - `build_desc` / `dirty_build_count` — from
//!   [`crate::scene::blas_sharing`] (which bucket has which geometry +
//!   error budget, and how many buckets are live).
//! - `params` uniform — strides + ref-buffer base address.
//!
//! Outputs (`@group(1)`), all **bucket-indexed**:
//! - `selected_clas_refs` — per-bucket contiguous ref lists.
//! - `args_buf` — per-bucket
//!   `VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV`
//!   records ready for the BLAS rebuild pass.
//! - `per_bucket_counts` — atomic counters (reset each frame by
//!   `select_reset` before the main pass).
//!
//! See `selector.wgsl` for the full algorithm.

use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        BindGroup, BindGroupEntries, ComputePassDescriptor, PipelineCache, ShaderType,
        UniformBuffer,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
};
use bytemuck::{Pod, Zeroable};

use crate::bindings::ClusterSceneBindGroup;
use crate::instance::InstanceManager;

use crate::gpu::allocator::{Allocator, SparseBuffer};
use super::blas_sharing::BlasSharing;
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;
use crate::geometry::ClasArena;

/// User-facing selector knobs. Defaults match aurora's published
/// "research" config. Consumed by the BLAS-sharing classify pass (which
/// owns the screen-space projection now); kept here as the selector is
/// the user-visible LOD resource.
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

/// Uniform layout shared with `selector.wgsl::SelectorParams` (16 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
pub struct SelectorParamsGpu {
    pub max_clusters_per_bucket: u32,
    pub bucket_capacity: u32,
    pub selected_clas_refs_addr_lo: u32,
    pub selected_clas_refs_addr_hi: u32,
}

const _: () = assert!(size_of::<SelectorParamsGpu>() == 16);

/// Virtual address space for the selector's per-frame I/O buffers.
/// Capped at 1 GB each — wgpu enforces `max_storage_buffer_binding_size`
/// (~2 GB on desktop adapters) at bind-group creation regardless of
/// sparse commit footprint. 1 GB / 8 B = 128 M slots for refs
/// (≈ 64 K buckets × 2 K clusters), 1 GB / 16 B = 64 M bucket args,
/// 1 GB / 4 B = 256 M counts. All sparse-backed; physical RAM scales
/// with the live bucket count.
const SELECTED_REFS_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;
const ARGS_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;
const COUNTS_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;

/// Render-world resource for the cluster LOD-selection compute pass —
/// per-frame frame-state only (sparse I/O buffers + the `SelectorParams`
/// uniform + the rebuilt bind group). The compute pipeline ids live on
/// [`SolariPipelines`] and the bind-group layout on [`SolariResourceManager`].
/// All buffers are bucket-indexed.
#[derive(Resource)]
pub struct Selector {
    /// Sparse `array<vec2<u32>>` of per-bucket ref lists.
    pub selected_clas_refs: SparseBuffer,
    /// Sparse `array<vec4<u32>>` of per-bucket BLAS-build inputs.
    pub args_buf: SparseBuffer,
    /// Sparse `array<atomic<u32>>` of per-bucket emit counters.
    pub per_bucket_counts: SparseBuffer,
    pub params: UniformBuffer<SelectorParamsGpu>,
    /// Per-frame bind group, rebuilt in `Render::PrepareBindGroups`.
    pub bind_group: Option<BindGroup>,
    /// True iff [`dispatch_selector`] actually recorded this frame — the
    /// downstream BLAS build (and its `commit_built`) gate on it, so a
    /// cold-start bail here can't be papered over (rebuild-until-built).
    pub recorded: bool,
}

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
    let selected_clas_refs = allocator.create_sparse_buffer(
        &render_device,
        // BLAS_INPUT in wgpu maps to AS_BUILD_INPUT_READ_ONLY +
        // SHADER_DEVICE_ADDRESS — the BLAS rebuild reads this buffer
        // by device address via the `cluster_references` field of
        // each per-bucket args record.
        ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        SELECTED_REFS_VIRTUAL_BYTES,
        "selector.selected_clas_refs",
    );
    let args_buf = allocator.create_sparse_buffer(
        &render_device,
        ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        ARGS_VIRTUAL_BYTES,
        "selector.args_buf",
    );
    let per_bucket_counts = allocator.create_sparse_buffer(
        &render_device,
        ash::vk::BufferUsageFlags::STORAGE_BUFFER,
        wgpu::BufferUsages::STORAGE,
        COUNTS_VIRTUAL_BYTES,
        "selector.per_bucket_counts",
    );

    let mut params: UniformBuffer<SelectorParamsGpu> = UniformBuffer::default();
    params.set_label(Some("selector.params"));

    commands.insert_resource(Selector {
        selected_clas_refs,
        args_buf,
        per_bucket_counts,
        params,
        bind_group: None,
        recorded: false,
    });
}

/// `Render::Prepare`: fill the per-frame `SelectorParams` uniform +
/// commit sparse pages on the I/O buffers (sized on the bucket
/// capacity, not instance count).
pub fn prepare_selector_params(
    mut selector: Option<ResMut<Selector>>,
    instances: Option<Res<InstanceManager>>,
    sharing: Option<Res<BlasSharing>>,
    allocator: Option<Res<Allocator>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(resources), Some(instances), Some(sharing), Some(_allocator)) =
        (selector.as_deref_mut(), instances, sharing, allocator)
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

    let params = SelectorParamsGpu {
        max_clusters_per_bucket: max_per,
        bucket_capacity,
        selected_clas_refs_addr_lo: (selected_refs_addr & 0xFFFF_FFFF) as u32,
        selected_clas_refs_addr_hi: (selected_refs_addr >> 32) as u32,
    };
    *resources.params.get_mut() = params;
    resources.params.write_buffer(&render_device, &render_queue);

    let cap = bucket_capacity as u64;
    let max_per = max_per as u64;
    resources.selected_clas_refs.commit(0..cap * max_per * 8);
    resources.args_buf.commit(0..cap * 16);
    resources.per_bucket_counts.commit(0..cap * 4);
}

/// `Render::PrepareBindGroups`: rebuild the selector bind group from the
/// per-frame resources + CLAS arena's address table + the BLAS-sharing
/// bucket descriptors.
pub fn prepare_selector_bind_group(
    mut selector: Option<ResMut<Selector>>,
    resource_manager: Option<Res<SolariResourceManager>>,
    pipeline_cache: Res<PipelineCache>,
    clas_arena: Option<Res<ClasArena>>,
    sharing: Option<Res<BlasSharing>>,
    render_device: Res<RenderDevice>,
) {
    let Some(selector) = selector.as_deref_mut() else {
        return;
    };
    let (Some(resource_manager), Some(clas_arena), Some(sharing)) =
        (resource_manager, clas_arena, sharing)
    else {
        selector.bind_group = None;
        return;
    };
    let Some(params_binding) = selector.params.binding() else {
        selector.bind_group = None;
        return;
    };

    let group = render_device.create_bind_group(
        "cluster_selector_bind_group",
        &pipeline_cache.get_bind_group_layout(&resource_manager.selector),
        &BindGroupEntries::sequential((
            clas_arena.cluster_clas_addresses.wgpu_buffer.as_entire_binding(),
            selector.selected_clas_refs.wgpu_buffer.as_entire_binding(),
            selector.args_buf.wgpu_buffer.as_entire_binding(),
            selector.per_bucket_counts.wgpu_buffer.as_entire_binding(),
            params_binding,
            sharing.build_desc.as_entire_binding(),
            sharing.dirty_build_count.as_entire_binding(),
        )),
    );
    selector.bind_group = Some(group);
}

/// `Render::Render`: dispatch the selector — one workgroup per bucket.
/// The live bucket count is GPU-side, so we dispatch the full bucket
/// capacity and let `select_main` early-out past `dirty_build_count`.
pub fn dispatch_selector(
    pipeline_cache: Res<PipelineCache>,
    selector: Option<ResMut<Selector>>,
    pipelines: Res<SolariPipelines>,
    scene_bind_group: Res<ClusterSceneBindGroup>,
    instances: Option<Res<InstanceManager>>,
    sharing: Option<Res<BlasSharing>>,
    mut ctx: RenderContext,
) {
    let (Some(selector), Some(instances), Some(sharing)) = (selector, instances, sharing) else {
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
    let (Some(scene_bg), Some(selector_bg)) =
        (scene_bind_group.bind_group.as_ref(), selector.bind_group.as_ref())
    else {
        return;
    };
    let (Some(reset_pipeline), Some(main_pipeline)) = (
        pipeline_cache.get_compute_pipeline(pipelines.selector_reset),
        pipeline_cache.get_compute_pipeline(pipelines.selector_main),
    ) else {
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("cluster_selector"),
        timestamp_writes: None,
    });
    pass.set_bind_group(0, scene_bg, &[]);
    pass.set_bind_group(1, selector_bg, &[]);

    let d = diagnostics.time_span(&mut pass, "cluster_selector");
    pass.set_pipeline(reset_pipeline);
    pass.dispatch_workgroups(bucket_capacity.div_ceil(64), 1, 1);

    pass.set_pipeline(main_pipeline);
    // One workgroup per bucket. `bucket_capacity ≤ MAX_GEOMETRIES ≤
    // 65535`, so a single dispatch dimension suffices; the `select_main`
    // guard early-outs buckets past `dirty_build_count`.
    pass.dispatch_workgroups(bucket_capacity, 1, 1);
    d.end(&mut pass);
    drop(pass);
    selector.recorded = true;
}
