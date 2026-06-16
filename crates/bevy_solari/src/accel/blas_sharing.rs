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
// as a dirty-geometry build entry — same buffer layout as before);
// `ptlas` reads the slot-indexed `instance_blas_address`.
//
// See `blas_sharing.wgsl` for the pass algorithms.
#![allow(unsafe_code, reason = "device-address plumbing for the geometry BLAS pool")]

use ash::vk;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        BindGroup, BindGroupEntries, Buffer, ComputePassDescriptor, PipelineCache, RawBufferVec,
        ShaderType, UniformBuffer,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    view::{ViewUniformOffset, ViewUniforms},
};
use bytemuck::{Pod, Zeroable};

use crate::bindings::ClusterSceneBindGroup;
use crate::ecs_gpu::GpuColumn;
use crate::instance::{GeometryIdColumn, InstanceManager};
use crate::geometry::ClusterMeshManager;

use crate::gpu::allocator::{Allocator, SparseBuffer};
use super::blas_rebuild::{query_blas_size, BLAS_REGION_ALIGN};
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;
use crate::gpu::extension::ClusterExtensionFns;
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

/// Uniform mirror of `blas_sharing.wgsl::SharingParams` (48 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
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
    pub _pad0: u32,
}

const _: () = assert!(size_of::<SharingParamsGpu>() == 48);

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
    /// written by `classify`. The animated instantiate reads it to run the same
    /// DAG cut as the static path (the cut is pose-independent).
    pub instance_e_build: SparseBuffer,
    /// dense active index → real `GpuEntity`. Re-uploaded only when
    /// the active set changes.
    pub active_to_slot: RawBufferVec<u32>,
    pub params: UniformBuffer<SharingParamsGpu>,

    /// CPU-tracked geometry capacity (== resident geometry high-water,
    /// clamped to `MAX_GEOMETRIES`). Private — read it through
    /// [`BlasSharing::build_entry_capacity`] so the clamp has one owner.
    geometry_count: u32,
    /// One-time init of the persistent `geometry_built_level` to NO_LEVEL.
    needs_init: bool,

    /// Per-frame bind group, rebuilt in `Render::PrepareBindGroups`. The compute
    /// pipeline ids live on [`SolariPipelines`], the layout on [`SolariResourceManager`].
    pub bind_group: Option<BindGroup>,
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
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        wgpu::BufferUsages::STORAGE,
        GEOMETRY_DST_VIRTUAL_BYTES,
        "blas_sharing.geometry_dst_addresses",
    );
    let instance_blas_address = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        wgpu::BufferUsages::STORAGE,
        SLOT_BUFFER_VIRTUAL_BYTES,
        "blas_sharing.instance_blas_address",
    );
    let instance_e_build = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        wgpu::BufferUsages::STORAGE,
        SLOT_BUFFER_VIRTUAL_BYTES,
        "blas_sharing.instance_e_build",
    );

    let cap = MAX_GEOMETRIES as u64;
    let geometry_desired_level = storage_buf(&render_device, "blas_sharing.desired_level", cap * 4);
    let geometry_built_level = storage_buf(&render_device, "blas_sharing.built_level", cap * 4);
    let geometry_dirty = storage_buf(&render_device, "blas_sharing.geometry_dirty", cap * 4);
    let geometry_desc = storage_buf(&render_device, "blas_sharing.geometry_desc", cap * 16);
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
    let mut params = UniformBuffer::<SharingParamsGpu>::default();
    params.set_label(Some("blas_sharing.params"));

    commands.insert_resource(BlasSharing {
        geometry_blas_pool,
        worst_case_stride: 0,
        geometry_desired_level,
        geometry_built_level,
        geometry_dirty,
        geometry_desc,
        dirty_gid,
        dirty_build_count,
        build_count,
        build_desc,
        geometry_dst_addresses,
        instance_blas_address,
        instance_e_build,
        active_to_slot,
        params,
        geometry_count: 1,
        needs_init: true,
        bind_group: None,
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

    // One-time: init persistent built-level to NO_LEVEL so every
    // geometry's first sighting triggers a build.
    if resources.needs_init {
        let init = vec![NO_LEVEL; MAX_GEOMETRIES as usize];
        render_queue.write_buffer(
            &resources.geometry_built_level,
            0,
            bytemuck::cast_slice(&init),
        );
        resources.needs_init = false;
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

    // Geometry capacity == resident geometry high-water (bounded set).
    let geometry_count = cluster_meshes.geometry_count().min(MAX_GEOMETRIES);
    resources.geometry_count = geometry_count.max(1);
    let geometry_capacity = MAX_GEOMETRIES; // fixed-size scratch buffers

    // If the stride grew, every geometry's region address moved → force
    // a full rebuild by resetting built levels to NO_LEVEL.
    if stride_changed && !resources.needs_init {
        let init = vec![NO_LEVEL; MAX_GEOMETRIES as usize];
        render_queue.write_buffer(
            &resources.geometry_built_level,
            0,
            bytemuck::cast_slice(&init),
        );
    }

    let pool_base = resources.geometry_blas_pool.address;
    let stride = resources.worst_case_stride;
    *resources.params.get_mut() = SharingParamsGpu {
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
        _pad0: 0,
    };
    resources.params.write_buffer(&render_device, &render_queue);

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

    // Commit sparse pages for the resident geometry set + active slots.
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
}

/// `Render::PrepareBindGroups`: rebuild the sharing bind group.
pub fn prepare_blas_sharing_bind_group(
    mut sharing: Option<ResMut<BlasSharing>>,
    resource_manager: Option<Res<SolariResourceManager>>,
    pipeline_cache: Res<PipelineCache>,
    geometry_ids: Option<Res<GpuColumn<GeometryIdColumn>>>,
    view_uniforms: Res<ViewUniforms>,
    render_device: Res<RenderDevice>,
) {
    let Some(sharing) = sharing.as_deref_mut() else {
        return;
    };
    let (Some(resource_manager), Some(geometry_ids)) = (resource_manager, geometry_ids) else {
        sharing.bind_group = None;
        return;
    };
    let (Some(params_binding), Some(view_binding), Some(active_to_slot)) = (
        sharing.params.binding(),
        view_uniforms.uniforms.binding(),
        sharing.active_to_slot.buffer(),
    ) else {
        sharing.bind_group = None;
        return;
    };
    let geometry_ids = geometry_ids.buffer().as_entire_binding();

    let group = render_device.create_bind_group(
        "blas_sharing_bind_group",
        &pipeline_cache.get_bind_group_layout(&resource_manager.blas_sharing),
        &BindGroupEntries::sequential((
            view_binding,
            params_binding,
            active_to_slot.as_entire_binding(),
            geometry_ids,
            sharing.geometry_desired_level.as_entire_binding(),
            sharing.geometry_built_level.as_entire_binding(),
            sharing.geometry_dirty.as_entire_binding(),
            sharing.dirty_build_count.as_entire_binding(),
            sharing.dirty_gid.as_entire_binding(),
            sharing.build_desc.as_entire_binding(),
            sharing.geometry_dst_addresses.wgpu_buffer.as_entire_binding(),
            sharing.instance_blas_address.wgpu_buffer.as_entire_binding(),
            sharing.build_count.as_entire_binding(),
            sharing.geometry_desc.as_entire_binding(),
            sharing.instance_e_build.wgpu_buffer.as_entire_binding(),
        )),
    );
    sharing.bind_group = Some(group);
}

const WORKGROUP_SIZE: u32 = 64;

/// `Render::Render`: geom_reset → classify → elect_dirty →
/// finalize_count → assign. Must run after the instance-column scatter
/// (reads `transforms`) and before the selector / `blas_rebuild`.
pub fn dispatch_blas_sharing(
    pipeline_cache: Res<PipelineCache>,
    sharing: Option<Res<BlasSharing>>,
    pipelines: Res<SolariPipelines>,
    scene_bind_group: Res<ClusterSceneBindGroup>,
    instances: Option<Res<InstanceManager>>,
    view_query: bevy_ecs::system::Query<&ViewUniformOffset, bevy_ecs::query::With<ExtractedCamera>>,
    mut ctx: RenderContext,
) {
    let (Some(sharing), Some(instances)) = (sharing, instances) else {
        return;
    };
    let active_count = instances.active_count() as u32;
    if active_count == 0 {
        return;
    }
    let geometry_count = sharing.build_entry_capacity();
    let (Some(scene_bg), Some(sharing_bg)) =
        (scene_bind_group.bind_group.as_ref(), sharing.bind_group.as_ref())
    else {
        return;
    };
    let Some(view_offset) = view_query.iter().next() else {
        return;
    };
    let view_offset = view_offset.offset;
    let (
        Some(geom_reset),
        Some(classify),
        Some(elect_dirty),
        Some(finalize_count),
        Some(assign_address),
    ) = (
        pipeline_cache.get_compute_pipeline(pipelines.blas_sharing_geom_reset),
        pipeline_cache.get_compute_pipeline(pipelines.blas_sharing_classify),
        pipeline_cache.get_compute_pipeline(pipelines.blas_sharing_elect_dirty),
        pipeline_cache.get_compute_pipeline(pipelines.blas_sharing_finalize_count),
        pipeline_cache.get_compute_pipeline(pipelines.blas_sharing_assign_address),
    ) else {
        return;
    };

    let active_groups = active_count.div_ceil(WORKGROUP_SIZE);
    let geom_groups = geometry_count.div_ceil(WORKGROUP_SIZE);

    // Separate passes so wgpu inserts the producer→consumer barriers.
    let steps: [(&wgpu::ComputePipeline, u32); 5] = [
        (geom_reset, geom_groups),
        (classify, active_groups),
        (elect_dirty, geom_groups),
        (finalize_count, 1),
        (assign_address, active_groups),
    ];

    // Separate passes (one encoder) so wgpu inserts the producer→consumer
    // barriers between them. Records into the shared `RenderContext`
    // encoder — the graph submits once for the frame.
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    // One span over the whole producer→consumer chain (separate passes, so the
    // span is taken on the encoder rather than a single pass).
    let d = diagnostics.time_span(encoder, "blas_sharing");
    for (pipeline, groups) in steps {
        let mut p = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("blas_sharing"),
            timestamp_writes: None,
        });
        p.set_bind_group(0, scene_bg, &[]);
        p.set_bind_group(1, sharing_bg, &[view_offset]);
        p.set_pipeline(pipeline);
        // Per-instance passes (classify / assign_address) 2D-split past 65535
        // workgroups; the geometry passes stay 1D (no-op). The shaders that can
        // exceed the limit reconstruct the flat index from `num_workgroups`.
        let (gx, gy, gz) = crate::ecs_gpu::linear_dispatch(groups);
        p.dispatch_workgroups(gx, gy, gz);
    }
    d.end(encoder);
}
