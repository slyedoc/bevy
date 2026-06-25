//! Reusable inline-`rayQuery` batch trace — a wgpu compute pass that traces a
//! buffer of [`Ray`]s against the scene TLAS and writes a buffer of [`Hit`]s.
//!
//! The shader is **self-contained**: it imports only `tlas` from the scene
//! bindings and traces inline, so it pulls in none of the bindless `physical_load`
//! resolve path (`resolve_ray_hit_full` / `load_material_bindless` / the shared
//! `trace_ray`). `physical_load` needs the `PhysicalStorageBufferAddresses` SPIR-V
//! capability, which only the rt_pipeline's hand-rolled WGSL→SPIR-V enables — a
//! plain wgpu compute pipeline can't, so it stays out. Two bind groups:
//!
//! - `@group(0)` — the scene group ([`RaytracingSceneBindings`]), for `tlas`.
//! - `@group(1)` — this pass's own I/O: `rays` (in), `hits` (out), `params`.
//!
//! Each hit carries `t` + `world_position` + the `instance` / `primitive` /
//! `geometry` indices; the `world_normal` is left zero (resolving it would need
//! vertex data via `physical_load` or a vertex-pool storage binding — deferred).
//!
//! There is no built-in producer: `ray_count` defaults to 0, so the pass is a
//! wired no-op until something fills [`SolariRayQuery::rays`] and sets
//! [`SolariRayQuery::ray_count`]. The buffers + count are `pub` for that producer;
//! it reads its results back from [`SolariRayQuery::hits`].
//!
//! Runs in [`SolariClusterSystems::RayQueries`], after `BuildTlas` (it needs the
//! built TLAS) and before `Cleanup`.

pub mod picking;

use ash::vk;
use bevy_app::{App, Plugin};
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_core_pipeline::schedule::camera_driver;
use bevy_ecs::{
    resource::Resource,
    schedule::{common_conditions::resource_exists, IntoScheduleConfigs},
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
        ShaderStages,
    },
    renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
    Render, RenderApp, RenderStartup, RenderSystems,
};
use bytemuck::{Pod, Zeroable};
use core::num::NonZeroU64;

use crate::bindings::RaytracingSceneBindings;
use crate::ecs_gpu::GpuColumn;
use crate::gpu::allocator::{Allocator, MemoryLocation, SparseBuffer};
use crate::instance::NodeSlotColumn;
use crate::transform::NodeEntityColumn;
use crate::{SolariClusterSystems, SolariSetup};

/// Virtual address space reserved for the ray / hit buffers. Sparse-backed, so
/// physical RAM scales with the committed (`ray_count`) prefix, not this reserve.
/// 256 MB / 32 B = 8 M rays; 256 MB / 48 B ≈ 5.5 M hits.
const RAYS_VIRTUAL_BYTES: u64 = 256 * 1024 * 1024;
const HITS_VIRTUAL_BYTES: u64 = 256 * 1024 * 1024;

/// Default ray `t_min` / `t_max` for a producer with no closer bound — mirrors
/// `scene_bindings::RAY_T_MIN` / `RAY_T_MAX`.
pub const RAY_T_MIN_DEFAULT: f32 = 0.001;
pub const RAY_T_MAX_DEFAULT: f32 = 100000.0;

/// One ray to trace — mirrors `ray_query.wgsl::Ray` (std430, 32 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct Ray {
    pub origin: [f32; 3],
    pub t_min: f32,
    pub direction: [f32; 3],
    pub t_max: f32,
}

const _: () = assert!(size_of::<Ray>() == 32);

/// One resolved hit — mirrors `ray_query.wgsl::Hit` (std430, 48 B). A miss has
/// `t < 0.0`; the indices + entity bits are only meaningful on a hit. `entity_lo` /
/// `entity_hi` are the picked entity's `Entity::to_bits` as `[lo, hi]` (resolved
/// GPU-side via the instance-slot → node-slot → entity indirection); reconstruct
/// with `Entity::from_bits(entity_lo as u64 | (entity_hi as u64) << 32)`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct Hit {
    pub world_position: [f32; 3],
    pub t: f32,
    pub world_normal: [f32; 3],
    pub instance_index: u32,
    pub primitive_index: u32,
    pub geometry_index: u32,
    pub entity_lo: u32,
    pub entity_hi: u32,
}

const _: () = assert!(size_of::<Hit>() == 48);

/// Dispatch parameters — mirrors `ray_query.wgsl::Params` (uniform, 16 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct Params {
    pub ray_count: u32,
    pub ray_flags: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(size_of::<Params>() == 16);

/// Render-world resource for the batch ray-query pass — its sparse I/O buffers,
/// the `params` uniform, and the per-frame I/O bind group. The compute pipeline id
/// lives on [`SolariPipelines`](crate::pipelines::SolariPipelines) and the layout on
/// [`SolariResourceManager`](crate::resource_manager).
///
/// A producer writes `rays[0..ray_count]`, sets `ray_count`, and reads `hits`.
#[derive(Resource)]
pub struct SolariRayQuery {
    /// Input rays — sparse `array<Ray>`. A producer fills `[0..ray_count]`.
    pub rays: SparseBuffer,
    /// Output hits — sparse `array<Hit>`, one per input ray.
    pub hits: SparseBuffer,
    /// `@group(1) @binding(2)` dispatch params (ray count + flags) — a raw uniform.
    params: Buffer,
    /// How many leading rays to trace this frame. Defaults to 0 (no producer yet),
    /// making the pass a no-op. A producer sets it after filling `rays`.
    pub ray_count: u32,
    /// `RayDesc` flag word for every traced ray (e.g. `RAY_FLAG_NONE`). Set by the
    /// producer alongside `ray_count`.
    pub ray_flags: u32,
    /// Per-frame I/O bind group (`@group(1)`), rebuilt in `Render::PrepareBindGroups`.
    pub bind_group: Option<BindGroup>,
}

/// The batch ray-query `@group(1)` I/O layout: rays (in), hits (out), params, plus
/// the two entity-indirection columns (`node_slots`, `node_entity`).
pub fn ray_query_io_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "ray_query_io_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 rays
                storage_buffer_sized(false, None),           // 1 hits
                uniform_buffer_sized(false, NonZeroU64::new(size_of::<Params>() as u64)), // 2 params
                storage_buffer_read_only_sized(false, None), // 3 node_slots (NodeSlotColumn)
                storage_buffer_read_only_sized(false, None), // 4 node_entity (NodeEntityColumn)
            ),
        ),
    )
}

/// `RenderStartup`: allocate the batch ray-query I/O buffers + the params uniform
/// and insert [`SolariRayQuery`]. No-op without the raw-VK [`Allocator`]; downstream
/// systems guard on the resource's presence.
pub fn init_ray_query(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
) {
    let Some(allocator) = allocator else {
        return;
    };
    let rays = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        RAYS_VIRTUAL_BYTES,
        "ray_query.rays",
    );
    let hits = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        HITS_VIRTUAL_BYTES,
        "ray_query.hits",
    );
    let params = allocator
        .create_buffer(
            &render_device,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size_of::<Params>() as u64,
            MemoryLocation::GpuOnly,
            "ray_query.params",
        )
        .into();

    commands.insert_resource(SolariRayQuery {
        rays,
        hits,
        params,
        ray_count: 0,
        ray_flags: 0,
        bind_group: None,
    });
}

/// `Render::Prepare`: commit sparse pages for `ray_count` rays / hits and write the
/// `params` uniform.
pub fn prepare_ray_query(
    ray_query: Option<ResMut<SolariRayQuery>>,
    render_queue: Res<RenderQueue>,
) {
    let Some(resources) = ray_query else {
        return;
    };
    let resources = resources.into_inner();

    let params = Params {
        ray_count: resources.ray_count,
        ray_flags: resources.ray_flags,
        _pad0: 0,
        _pad1: 0,
    };
    render_queue.write_buffer(&resources.params, 0, bytemuck::bytes_of(&params));

    if resources.ray_count == 0 {
        return;
    }
    resources
        .rays
        .commit(0..resources.ray_count as u64 * size_of::<Ray>() as u64);
    resources
        .hits
        .commit(0..resources.ray_count as u64 * size_of::<Hit>() as u64);
}

/// `Render::PrepareBindGroups`: rebuild this pass's `@group(1)` I/O bind group from
/// the rays / hits / params + the two entity-indirection columns.
///
/// The columns are bound at their **committed** sizes (via [`GpuColumn::binding`]),
/// not the full sparse reservation, so a shader `arrayLength()` / out-of-range index
/// is bounds-checked rather than a page fault — mirroring the scene-columns builder's
/// committed-bytes guard. If either column hasn't committed any pages yet (`binding`
/// returns `None`), the bind group is left `None` and the dispatch skips this frame.
pub fn prepare_ray_query_bind_group(
    ray_query: Option<ResMut<SolariRayQuery>>,
    resource_manager: Option<Res<crate::resource_manager::SolariResourceManager>>,
    node_slots: Option<Res<GpuColumn<NodeSlotColumn>>>,
    node_entity: Option<Res<GpuColumn<NodeEntityColumn>>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let Some(ray_query) = ray_query else {
        return;
    };
    let ray_query = ray_query.into_inner();
    let (Some(resource_manager), Some(node_slots), Some(node_entity)) =
        (resource_manager, node_slots, node_entity)
    else {
        ray_query.bind_group = None;
        return;
    };
    // Bind each column to exactly its committed range; `None` until the first page
    // is committed → skip this frame (no stale-buffer read, no page fault).
    let (Some(node_slots_binding), Some(node_entity_binding)) =
        (node_slots.binding(), node_entity.binding())
    else {
        ray_query.bind_group = None;
        return;
    };

    let group = render_device.create_bind_group(
        "ray_query_io_bind_group",
        &pipeline_cache.get_bind_group_layout(&resource_manager.ray_query_io),
        &BindGroupEntries::sequential((
            ray_query.rays.wgpu_buffer.as_entire_binding(),
            ray_query.hits.wgpu_buffer.as_entire_binding(),
            ray_query.params.as_entire_binding(),
            node_slots_binding,
            node_entity_binding,
        )),
    );
    ray_query.bind_group = Some(group);
}

/// `RenderGraph`: dispatch the batch ray-query — one thread per ray. Early-returns
/// when there's nothing to trace or the scene group / pipeline isn't ready.
pub fn dispatch_ray_query(
    pipeline_cache: Res<PipelineCache>,
    ray_query: Option<Res<SolariRayQuery>>,
    pipelines: Res<crate::pipelines::SolariPipelines>,
    scene_bindings: Res<RaytracingSceneBindings>,
    mut ctx: RenderContext,
) {
    let Some(ray_query) = ray_query else {
        return;
    };
    if ray_query.ray_count == 0 {
        return;
    }
    let (Some(io_bg), Some(scene_bg)) = (
        ray_query.bind_group.as_ref(),
        scene_bindings.bind_group.as_ref(),
    ) else {
        return;
    };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.ray_query) else {
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("ray_query"),
        timestamp_writes: None,
    });
    // Scene group (0) for `tlas`; this pass's I/O (1).
    pass.set_bind_group(0, scene_bg, &[]);
    pass.set_bind_group(1, io_bg, &[]);

    let d = diagnostics.time_span(&mut pass, "ray_query");
    pass.set_pipeline(pipeline);
    pass.dispatch_workgroups(ray_query.ray_count.div_ceil(64), 1, 1);
    d.end(&mut pass);
}

/// Queue the batch ray-query compute pipeline. Called from
/// [`init_solari_pipelines`](crate::pipelines::init_solari_pipelines) with the
/// composited layout `[scene (group 0), I/O (group 1)]` so the two bound groups
/// match the WGSL's `@group` numbers.
pub fn queue_ray_query_pipeline(
    pipeline_cache: &PipelineCache,
    asset_server: &AssetServer,
    layout: Vec<BindGroupLayoutDescriptor>,
) -> CachedComputePipelineId {
    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("ray_query".into()),
        layout,
        shader: load_embedded_asset!(asset_server, "ray_query.wgsl"),
        shader_defs: vec![],
        entry_point: Some("query_rays".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    })
}

/// Batch ray-query plugin: embeds the shader, registers the resource + bind-group
/// resource, and schedules the pass in [`SolariClusterSystems::RayQueries`].
pub struct RayQueryPlugin;

impl Plugin for RayQueryPlugin {
    fn build(&self, app: &mut App) {
        // The shader is embedded centrally in `crate::pipelines::embed_solari_shaders`,
        // co-located with the rest of the solari compute shaders. This plugin wires only
        // the trace SERVICE; a consumer (e.g. `picking::SolariPickingPlugin`) drives it.

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .configure_sets(
                RenderGraph,
                SolariClusterSystems::RayQueries
                    .after(SolariClusterSystems::BuildTlas)
                    .before(SolariClusterSystems::Cleanup)
                    .in_set(RenderGraphSystems::Render)
                    .before(camera_driver),
            )
            .add_systems(RenderStartup, init_ray_query.after(SolariSetup))
            .add_systems(
                Render,
                (
                    prepare_ray_query.in_set(RenderSystems::Prepare),
                    prepare_ray_query_bind_group.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                RenderGraph,
                dispatch_ray_query
                    .run_if(resource_exists::<crate::pipelines::SolariPipelines>)
                    .in_set(SolariClusterSystems::RayQueries),
            );
    }
}
