//! GPU frontier expansion — turn this frame's *changed* nodes into changed nodes
//! **plus all their descendants**, entirely on the GPU.
//!
//! The propagate walk recomposes only the nodes in its dispatch list, and a
//! moving parent's descendants didn't change their own `local` — so the changed
//! `local` delta (plus [`GpuFrameSeeds`] — GPU-moved nodes with no CPU change edge)
//! seeds a worklist that expands level-by-level through the `first_child` /
//! `next_sibling` columns, deduped by a per-node frame-epoch stamp. The result —
//! `frontier[HEADER..HEADER+total]` — is the propagate/readback dispatch list,
//! consumed via indirect dispatch with args this pass writes.
//!
//! Kernel sequence (one raw command stream, barriers between each step):
//! `seed` → (`finalize` → `expand`)×MAX_LEVELS → `finalize`. `finalize`
//! (1 thread) closes the current level's bounds and writes the next expand's
//! indirect args; empty levels dispatch zero workgroups.

#![allow(unsafe_code)]

use ash::vk;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    render_resource::{Buffer, BufferDescriptor, BufferUsages, StorageBuffer},
    renderer::{RenderContext, RenderDevice, RenderQueue},
};
use bytemuck::{Pod, Zeroable};
use wgpu::hal::api::Vulkan as VkApi;

use crate::ecs_gpu::{GpuColumn, GpuTable};
use crate::gpu::allocator::Allocator;
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};

use super::graph::{
    FirstChildColumn, GpuFrameSeeds, LocalTranslationColumn, NextSiblingColumn, ParentColumn,
    TransformGraph,
};

const WORKGROUP_SIZE: u32 = 64;

static XFORM_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Latch [`SolariSettings::xform_debug`](crate::SolariSettings) at plugin `finish`.
pub(crate) fn latch_xform_debug(on: bool) {
    let _ = XFORM_DEBUG.set(on);
}

/// [`SolariSettings::xform_debug`](crate::SolariSettings): trace the changed-path
/// seed/walk decisions — the silent-bail points where cold-start seed loss hides.
pub(crate) fn xform_debug() -> bool {
    XFORM_DEBUG.get().copied().unwrap_or(false)
}
/// Max expansion depth (hierarchy levels below a moved node). Must match
/// `transform_frontier.slang`; deeper descendants go stale (mirror of the walk's
/// `MAX_DEPTH = 64` guard — realistic scenes are ≤ 8 deep).
pub const MAX_LEVELS: u32 = 16;
/// Header u32s at the front of the frontier buffer (must match the Slang):
/// `[total(atomic), total_plain, current_level, pad, level_begin[MAX_LEVELS+2], pad…]`.
pub const FRONTIER_HEADER_WORDS: u32 = 4 + MAX_LEVELS + 2 + 2; // = 24, 16B-aligned
/// Indirect-args u32s: `(x,y,z)` per expand level + one shared entry for the
/// propagate/readback consumers at `[MAX_LEVELS * 3..]`.
const INDIRECT_WORDS: u64 = ((MAX_LEVELS + 1) * 3) as u64;
/// Byte offset of the propagate/readback consumers' indirect args.
pub const CONSUMER_ARGS_OFFSET: u64 = (MAX_LEVELS * 3) as u64 * 4;

/// Push params shared with `transform_frontier.slang::FrontierParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct FrontierParams {
    /// Changed-`local` records this frame (seed source 0).
    changed_count: u32,
    /// Words per changed record; slot at `k * record_stride`.
    record_stride: u32,
    /// [`GpuFrameSeeds`] entries (seed source 1, appended after the delta's).
    extra_count: u32,
    /// This frame's epoch stamp (monotonic from 1 — a fresh zeroed epoch buffer
    /// can never collide).
    frame_id: u32,
    /// Node coverage (out-of-range guard).
    node_count: u32,
    /// X workgroup count of the seed's 2D-split dispatch.
    groups_x: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Render-world resource: the frontier worklist + epoch + indirect buffers and
/// the seed/finalize/expand heap kernels.
#[derive(Resource)]
pub struct TransformFrontier {
    /// `[header, nodes…]` worklist (see [`FRONTIER_HEADER_WORDS`]). Plain buffer,
    /// pow2-regrown; the heap descriptor is rewritten every dispatch.
    frontier: Buffer,
    /// Per-node frame-epoch stamp (dedupe). Zero-init on (re)creation; `frame_id`
    /// only grows, so recreation can't alias a live stamp.
    epoch: Buffer,
    /// Dispatch args: per-level expand + the shared consumer entry.
    indirect: Buffer,
    /// [`GpuFrameSeeds`] uploaded for the seed kernel.
    extra_seeds: StorageBuffer<Vec<u32>>,
    params: FrontierParams,
    /// 2D-split groups of the seed dispatch.
    seed_groups: (u32, u32, u32),
    capacity_slots: u32,
    frame_id: u32,
    seed_count: u32,
    /// Whether the seed/expand/finalize chain actually recorded this frame. The
    /// walk's changed path must NOT dispatch from the indirect args otherwise —
    /// they're stale (or zero), and the seeds' delta records are consumed by the
    /// column scatter this frame, so a silent skip loses those nodes for good.
    ran: bool,
    seed: HeapKernel,
    finalize: HeapKernel,
    expand: HeapKernel,
    slots: KernelSlots,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for TransformFrontier {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        for kernel in [&self.seed, &self.finalize, &self.expand] {
            // SAFETY: quiesced; handles exclusively owned here.
            unsafe { kernel.destroy(&self.raw_device) };
        }
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for TransformFrontier {}
unsafe impl Sync for TransformFrontier {}

impl TransformFrontier {
    /// The worklist buffer consumers (propagate/readback) bind: count at word 1,
    /// slots from word [`FRONTIER_HEADER_WORDS`].
    #[inline]
    pub fn frontier_buffer(&self) -> &Buffer {
        &self.frontier
    }

    /// The indirect-args buffer; consumers dispatch at [`CONSUMER_ARGS_OFFSET`].
    #[inline]
    pub fn indirect_buffer(&self) -> &Buffer {
        &self.indirect
    }

    /// CPU-known seed count (changed records + gpu-frame seeds). Zero → the whole
    /// frontier is empty this frame and every consumer can skip.
    #[inline]
    pub fn seed_count(&self) -> u32 {
        self.seed_count
    }

    /// Whether the frontier chain recorded this frame — the indirect args are
    /// fresh iff true. See the field docs.
    #[inline]
    pub fn ran(&self) -> bool {
        self.ran
    }
}

/// `RenderStartup` (after `SolariSetup`): compile the seed/finalize/expand
/// kernels — layout-free heap pipelines ([`HeapKernel`]), Slang from source —
/// plus the fixed-size indirect buffer; the worklist/epoch buffers are created
/// on first growth in prepare.
pub fn init_transform_frontier(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let make = |entry: &str, label: &str| {
        HeapKernel::new(
            &seam,
            "transform_frontier.slang",
            include_str!("transform_frontier.slang"),
            entry,
            &[],
            &[],
            label,
            size_of::<FrontierParams>() as u32,
        )
    };
    let (Some(seed), Some(finalize), Some(expand)) = (
        make("seed", "transform_frontier_seed"),
        make("finalize", "transform_frontier_finalize"),
        make("expand", "transform_frontier_expand"),
    ) else {
        return;
    };
    let indirect = render_device.create_buffer(&BufferDescriptor {
        label: Some("transform.frontier_indirect"),
        size: INDIRECT_WORDS * 4,
        usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut extra_seeds = StorageBuffer::<Vec<u32>>::default();
    extra_seeds.set_label(Some("transform.frontier_extra_seeds"));
    commands.insert_resource(TransformFrontier {
        frontier: make_frontier_buffer(&render_device, 1),
        epoch: make_epoch_buffer(&render_device, 1),
        indirect,
        extra_seeds,
        params: FrontierParams::default(),
        seed_groups: (0, 0, 0),
        capacity_slots: 1,
        frame_id: 0,
        seed_count: 0,
        ran: false,
        seed,
        finalize,
        expand,
        slots: KernelSlots::new(&seam, 7),
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

fn make_frontier_buffer(render_device: &RenderDevice, slots: u32) -> Buffer {
    render_device.create_buffer(&BufferDescriptor {
        label: Some("transform.frontier"),
        size: (FRONTIER_HEADER_WORDS + slots) as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_epoch_buffer(render_device: &RenderDevice, slots: u32) -> Buffer {
    render_device.create_buffer(&BufferDescriptor {
        label: Some("transform.frontier_epoch"),
        size: slots as u64 * 4,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

/// `Render::Prepare` (before the propagate's prepare): grow the worklist/epoch to
/// the node high-water, upload the extra seeds, zero the header, set params.
pub fn prepare_transform_frontier(
    mut frontier: Option<ResMut<TransformFrontier>>,
    graph: Option<Res<TransformGraph>>,
    local_t: Option<Res<GpuColumn<LocalTranslationColumn>>>,
    seeds: Option<Res<GpuFrameSeeds>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(frontier), Some(graph), Some(local_t)) = (frontier.as_deref_mut(), graph, local_t)
    else {
        return;
    };
    let node_count = graph.high_water();
    if node_count > frontier.capacity_slots {
        frontier.capacity_slots = node_count.next_power_of_two();
        frontier.frontier = make_frontier_buffer(&render_device, frontier.capacity_slots);
        // Fresh epoch buffer is zeroed; `frame_id` keeps rising so no stamp aliases.
        frontier.epoch = make_epoch_buffer(&render_device, frontier.capacity_slots);
    }

    let extra: &[u32] = seeds.as_ref().map(|s| s.0.as_slice()).unwrap_or(&[]);
    let changed_count = local_t.pending();
    frontier.seed_count = changed_count + extra.len() as u32;
    frontier.ran = false;
    frontier.frame_id = frontier.frame_id.wrapping_add(1).max(1);
    if frontier.seed_count == 0 {
        return; // consumers key off seed_count — nothing to reset or upload.
    }

    // Zero the header (total/current_level/level_begin) before the seed kernel.
    render_queue.write_buffer(
        &frontier.frontier,
        0,
        bytemuck::cast_slice(&[0u32; FRONTIER_HEADER_WORDS as usize]),
    );
    frontier.extra_seeds.set(extra.to_vec());
    frontier.extra_seeds.write_buffer(&render_device, &render_queue);

    frontier.seed_groups =
        crate::ecs_gpu::linear_dispatch(frontier.seed_count.div_ceil(WORKGROUP_SIZE));
    frontier.params = FrontierParams {
        changed_count,
        record_stride: local_t.record_stride(),
        extra_count: extra.len() as u32,
        frame_id: frontier.frame_id,
        node_count,
        groups_x: frontier.seed_groups.0,
        _pad0: 0,
        _pad1: 0,
    };
}

/// `RenderGraph` (`Propagate`, before the walk): seed, then alternate finalize /
/// indirect expand per level, then a final finalize that publishes the consumer
/// count + indirect args. One raw command stream — each step reads what the
/// previous wrote (frontier/epoch/args), so a barrier separates every dispatch,
/// and the expands consume the args `finalize` just wrote as indirect commands.
pub fn dispatch_transform_frontier(
    frontier: Option<ResMut<TransformFrontier>>,
    seam: Option<Res<BindingSeam>>,
    local_t: Option<Res<GpuColumn<LocalTranslationColumn>>>,
    first_child: Option<Res<GpuColumn<FirstChildColumn>>>,
    next_sibling: Option<Res<GpuColumn<NextSiblingColumn>>>,
    parent: Option<Res<GpuColumn<ParentColumn>>>,
    mut ctx: RenderContext,
) {
    let (
        Some(mut frontier),
        Some(seam),
        Some(local_t),
        Some(first_child),
        Some(next_sibling),
        Some(parent),
    ) = (frontier, seam, local_t, first_child, next_sibling, parent)
    else {
        return;
    };
    let frontier = &mut *frontier;
    if frontier.seed_count == 0 {
        return;
    }
    let Some(extra) = frontier.extra_seeds.buffer() else {
        if xform_debug() {
            bevy_log::info!(
                "frontier: bail, extra-seeds buffer missing (seed_count {})",
                frontier.seed_count
            );
        }
        return;
    };
    // Empty-delta fallback: the shader never reads past `changed_count == 0`.
    let changed = local_t.delta_buffer().unwrap_or_else(|| parent.buffer());
    let s_changed = frontier.slots.buffer(&seam, 0, changed);
    let s_extra = frontier.slots.buffer(&seam, 1, extra);
    let s_first_child = frontier.slots.buffer(&seam, 2, first_child.buffer());
    let s_next_sibling = frontier.slots.buffer(&seam, 3, next_sibling.buffer());
    let s_frontier = frontier.slots.buffer(&seam, 4, &frontier.frontier);
    let s_epoch = frontier.slots.buffer(&seam, 5, &frontier.epoch);
    let s_indirect = frontier.slots.buffer(&seam, 6, &frontier.indirect);
    let params = bytemuck::bytes_of(&frontier.params);
    let seed_blob = frontier.seed.push_blob(
        "transform_frontier_seed",
        params,
        &[
            ("changed", s_changed),
            ("extra_seeds", s_extra),
            ("frontier", s_frontier),
            ("epoch", s_epoch),
        ],
    );
    let finalize_blob = frontier.finalize.push_blob(
        "transform_frontier_finalize",
        params,
        &[("frontier", s_frontier), ("indirect", s_indirect)],
    );
    let expand_blob = frontier.expand.push_blob(
        "transform_frontier_expand",
        params,
        &[
            ("first_child", s_first_child),
            ("next_sibling", s_next_sibling),
            ("frontier", s_frontier),
            ("epoch", s_epoch),
            ("indirect", s_indirect),
        ],
    );
    let (gx, gy, gz) = frontier.seed_groups;
    let encoder = ctx.command_encoder();
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // barriers bracket every dispatch (each step reads the previous step's
    // frontier/epoch/args writes, invisible to wgpu's tracking).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &frontier.raw_device;
            let raw_indirect = frontier
                .indirect
                .as_hal::<VkApi>()
                .map(|b| b.raw_handle())
                .expect("bevy_solari requires the Vulkan backend");
            // One barrier serves every edge in the chain: compute writes → the
            // next step's storage reads/writes AND its indirect-args read.
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::DRAW_INDIRECT,
                )
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_READ
                        | vk::AccessFlags2::SHADER_WRITE
                        | vk::AccessFlags2::INDIRECT_COMMAND_READ,
                )];
            let dep = vk::DependencyInfo::default().memory_barriers(&barrier);
            // Column-scatter writes -> the seed's delta/epoch access.
            dev.cmd_pipeline_barrier2(cb, &dep);
            seam.bind_heaps(cb);
            seam.push_data(cb, &seed_blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, frontier.seed.pipeline);
            dev.cmd_dispatch(cb, gx, gy, gz);
            for level in 0..MAX_LEVELS {
                dev.cmd_pipeline_barrier2(cb, &dep);
                seam.push_data(cb, &finalize_blob);
                dev.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    frontier.finalize.pipeline,
                );
                dev.cmd_dispatch(cb, 1, 1, 1);
                dev.cmd_pipeline_barrier2(cb, &dep);
                seam.push_data(cb, &expand_blob);
                dev.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    frontier.expand.pipeline,
                );
                dev.cmd_dispatch_indirect(cb, raw_indirect, u64::from(level) * 12);
            }
            // Close the last level + publish `total_plain` and the consumer indirect args.
            dev.cmd_pipeline_barrier2(cb, &dep);
            seam.push_data(cb, &finalize_blob);
            dev.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                frontier.finalize.pipeline,
            );
            dev.cmd_dispatch(cb, 1, 1, 1);
            // The worklist + consumer args -> the propagate/readback dispatches.
            dev.cmd_pipeline_barrier2(cb, &dep);
        });
    }
    frontier.ran = true;
}
