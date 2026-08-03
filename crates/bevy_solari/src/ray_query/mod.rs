//! Reusable inline-`RayQuery` batch trace — a heap-kernel compute pass that
//! traces a buffer of [`Ray`]s against the scene TLAS and writes a buffer of
//! [`Hit`]s.
//!
//! The shader (`ray_query.slang`) is **self-contained**: it declares the TLAS
//! locally (at the scene surface's (0,4) binding number, push-address-mapped)
//! and traces inline, importing nothing from `scene_resolve` — so the kernel
//! builds at startup with no scene heap slots, via
//! [`HeapKernel::new_with_mappings`]. Set 1 is this pass's own I/O: `rays`
//! (in), `hits` (out), `params`, plus the two entity-indirection columns
//! (`node_slots`, `node_entity`), all pushed as heap slots per dispatch.
//!
//! Each hit carries `t` + `world_position` + the `instance` / `primitive` /
//! `geometry` indices; the `world_normal` is left zero (resolving it would need
//! the scene's vertex pools — deferred).
//!
//! There is no built-in producer: `ray_count` defaults to 0, so the pass is a
//! wired no-op until something fills [`SolariRayQuery::rays`] and sets
//! [`SolariRayQuery::ray_count`]. The buffers + count are `pub` for that producer;
//! it reads its results back from [`SolariRayQuery::hits`].
//!
//! Runs in [`SolariClusterSystems::RayQueries`], after `BuildTlas` (it needs the
//! built TLAS) and before `Cleanup`.
#![allow(unsafe_code)]

pub mod picking;

use ash::vk;
use bevy_app::{App, Plugin};
use bevy_core_pipeline::schedule::camera_driver;
use bevy_ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    render_resource::{Buffer, CommandEncoderDescriptor},
    renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
    Render, RenderApp, RenderStartup, RenderSystems,
};
use bytemuck::{Pod, Zeroable};
use wgpu::hal::api::Vulkan as VkApi;

use crate::ecs_gpu::GpuColumn;
use crate::gpu::allocator::{Allocator, MemoryLocation, SparseBuffer};
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
use crate::instance::NodeSlotColumn;
use crate::transform::NodeEntityColumn;
use crate::{SolariClusterSystems, SolariSetup};

/// Virtual address space reserved for the ray / hit buffers. Sparse-backed, so
/// physical RAM scales with the committed (`ray_count`) prefix, not this reserve.
/// 256 MB / 32 B = 8 M rays; 256 MB / 48 B ≈ 5.5 M hits.
const RAYS_VIRTUAL_BYTES: u64 = 256 * 1024 * 1024;
const HITS_VIRTUAL_BYTES: u64 = 256 * 1024 * 1024;

/// Default ray `t_min` / `t_max` for a producer with no closer bound — mirrors
/// `scene_resolve::RAY_T_MIN` / `RAY_T_MAX`.
pub const RAY_T_MIN_DEFAULT: f32 = 0.001;
pub const RAY_T_MAX_DEFAULT: f32 = 1.0e30;

/// One ray to trace — mirrors `ray_query.slang::Ray` (std430, 32 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct Ray {
    pub origin: [f32; 3],
    pub t_min: f32,
    pub direction: [f32; 3],
    pub t_max: f32,
}

const _: () = assert!(size_of::<Ray>() == 32);

/// One resolved hit — mirrors `ray_query.slang::Hit` (std430, 48 B). A miss has
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

/// Dispatch parameters — mirrors `ray_query.slang::Params` (uniform, 16 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct Params {
    pub ray_count: u32,
    pub ray_flags: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(size_of::<Params>() == 16);

/// Render-world resource for the batch ray-query pass — the heap kernel, its
/// slots, the sparse I/O buffers, and the `params` uniform.
///
/// A producer writes `rays[0..ray_count]`, sets `ray_count`, and reads `hits`.
#[derive(Resource)]
pub struct SolariRayQuery {
    /// Input rays — sparse `array<Ray>`. A producer fills `[0..ray_count]`.
    pub rays: SparseBuffer,
    /// Output hits — sparse `array<Hit>`, one per input ray.
    pub hits: SparseBuffer,
    /// Dispatch params (ray count + flags) — a uniform-buffer heap slot (a
    /// `[[vk::push_constant]]` block would collide with the push blob's TLAS
    /// address at offset 0).
    params: Buffer,
    /// How many leading rays to trace this frame. Defaults to 0 (no producer yet),
    /// making the pass a no-op. A producer sets it after filling `rays`.
    pub ray_count: u32,
    /// Per-ray flag word (e.g. `RAY_FLAG_NONE`). Set by the producer alongside
    /// `ray_count`.
    pub ray_flags: u32,
    kernel: HeapKernel,
    /// rays, hits, params, node_slots, node_entity — rewritten per dispatch.
    slots: KernelSlots,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for SolariRayQuery {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for SolariRayQuery {}
unsafe impl Sync for SolariRayQuery {}

/// `RenderStartup` (after `SolariSetup`): compile the kernel — a layout-free
/// heap pipeline whose only non-set-1 binding is the push-address TLAS —
/// allocate the I/O buffers + the params uniform, and insert [`SolariRayQuery`].
/// No-op without the raw-VK [`Allocator`]; downstream systems guard on the
/// resource's presence.
pub fn init_ray_query(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
    seam: Option<Res<BindingSeam>>,
) {
    let (Some(allocator), Some(seam)) = (allocator, seam) else {
        return;
    };
    let Some(kernel) = HeapKernel::new_with_mappings(
        &seam,
        "ray_query.slang",
        include_str!("ray_query.slang"),
        "query_rays",
        &[],
        &[],
        crate::gpu::slang::RAY_QUERY_CAPABILITIES,
        "ray_query",
        // The 8-byte TLAS device address leads the push blob.
        8,
        &[seam.map_binding_push_address(0, 4, 0)],
    ) else {
        return;
    };
    let slots = KernelSlots::new(&seam, 5);
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
            // TRANSFER_DST must be on the VK usage to match the wgpu `COPY_DST`:
            // `write_buffer` stages this UBO via `vkCmdCopyBuffer`, which wgpu allows
            // (COPY_DST on the wrapper) but VK rejects unless the VkBuffer itself
            // carries TRANSFER_DST (VUID-vkCmdCopyBuffer-dstBuffer-00120).
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
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
        kernel,
        slots,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
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

/// `RenderGraph`: dispatch the batch ray-query — one thread per ray, a raw heap
/// dispatch on its own command buffer. Early-returns when there's nothing to
/// trace or the TLAS/columns aren't built yet.
///
/// The entity-indirection columns are bound at their **committed** sizes, not
/// the full sparse reservation, so a shader out-of-range index is
/// bounds-checked rather than a page fault — mirroring the scene-columns
/// builder's committed-bytes guard. If either column hasn't committed any
/// pages yet, the dispatch skips this frame.
pub fn dispatch_ray_query(
    ray_query: Option<Res<SolariRayQuery>>,
    seam: Option<Res<BindingSeam>>,
    node_slots: Option<Res<GpuColumn<NodeSlotColumn>>>,
    node_entity: Option<Res<GpuColumn<NodeEntityColumn>>>,
    ptlas: Option<Res<crate::accel::ptlas::Ptlas>>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (Some(ray_query), Some(seam), Some(node_slots), Some(node_entity)) =
        (ray_query, seam, node_slots, node_entity)
    else {
        return;
    };
    if ray_query.ray_count == 0 {
        return;
    }
    // The current PTLAS's device address, pushed for the TLAS's `PUSH_ADDRESS`
    // mapping.
    let tlas_address = match ptlas.as_deref() {
        Some(p) if p.has_built && p.as_handle_device_address[p.current] != 0 => {
            p.as_handle_device_address[p.current]
        }
        _ => return,
    };
    let (node_slots_bytes, node_entity_bytes) =
        (node_slots.committed_bytes(), node_entity.committed_bytes());
    if node_slots_bytes == 0 || node_entity_bytes == 0 {
        return;
    }

    let blob = ray_query.kernel.push_blob(
        "ray_query",
        &tlas_address.to_le_bytes(),
        &[
            ("rays", ray_query.slots.buffer(&seam, 0, &ray_query.rays.wgpu_buffer)),
            ("hits", ray_query.slots.buffer(&seam, 1, &ray_query.hits.wgpu_buffer)),
            ("params", ray_query.slots.uniform(&seam, 2, &ray_query.params)),
            (
                "node_slots",
                ray_query
                    .slots
                    .buffer_sized(&seam, 3, node_slots.buffer(), node_slots_bytes),
            ),
            (
                "node_entity",
                ray_query
                    .slots
                    .buffer_sized(&seam, 4, node_entity.buffer(), node_entity_bytes),
            ),
        ],
    );

    // Own command buffer — the fork panics if one encoder mixes wgpu passes
    // with raw as_hal_mut; `add_command_buffer` flushes pending ctx work first.
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("ray_query"),
    });
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // barriers bracket this dispatch against the AS build, the producer's ray
    // upload, and the consumer's hit readback (raw dispatches are invisible to
    // wgpu's tracking).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &ray_query.raw_device;
            // AS build + column scatters + the ray/params transfers → our
            // inline-query traversal and buffer reads.
            let pre = [vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
                        | vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::ALL_TRANSFER,
                )
                .src_access_mask(
                    vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR
                        | vk::AccessFlags2::SHADER_WRITE
                        | vk::AccessFlags2::TRANSFER_WRITE,
                )
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(
                    vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
                        | vk::AccessFlags2::SHADER_READ
                        | vk::AccessFlags2::SHADER_WRITE,
                )];
            dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&pre));
            seam.bind_heaps(cb);
            seam.push_data(cb, &blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, ray_query.kernel.pipeline);
            dev.cmd_dispatch(cb, ray_query.ray_count.div_ceil(64), 1, 1);
            // Hit writes → the consumer's readback copy / compute reads.
            let post = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::ALL_TRANSFER,
                )
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::TRANSFER_READ,
                )];
            dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&post));
        });
    }
    ctx.add_command_buffer(encoder.finish());
}

/// Batch ray-query plugin: registers the resource and schedules the pass in
/// [`SolariClusterSystems::RayQueries`].
pub struct RayQueryPlugin;

impl Plugin for RayQueryPlugin {
    fn build(&self, app: &mut App) {
        // This plugin wires only the trace SERVICE; a consumer (e.g.
        // `picking::SolariPickingPlugin`) drives it.

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
                prepare_ray_query.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                RenderGraph,
                dispatch_ray_query.in_set(SolariClusterSystems::RayQueries),
            );
    }
}
