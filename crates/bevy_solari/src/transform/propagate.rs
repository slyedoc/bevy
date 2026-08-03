//! GPU transform propagation — single-pass ancestor-walk over the changed set,
//! producing each node's ABSOLUTE world (native-`f64` translation via `SHADER_F64`).
//!
//! `world[node]` is a pure function of the node's own `local`/`parent` ancestor
//! chain (`world = local[root] ∘ … ∘ local[parent] ∘ local[node]`), independent
//! of every other node's world. So one thread walks a node's chain and writes
//! its world in a SINGLE pass — no iteration, no ping-pong, no hazard.
//!
//! The buffers are **persistent**: each frame only the nodes whose `local`
//! changed are recomputed (the column's delta — `changed[k*stride]` is the
//! slot), and static nodes keep last frame's value. The cold-start
//! `full_rebuild` walks every node once (one thread per node).
//!
//! The translation is accumulated in `f64`, so a node's absolute position survives
//! at AU/interstellar magnitude — the huge magnitude is NOT subtracted here. The
//! separate subtract pass ([`super::subtract`]) subtracts the camera's own absolute
//! world (the origin) to emit the small origin-relative f32 `world_rel` every RT
//! consumer reads. De-fusing the subtract from the walk keeps this pass
//! changed-only: the origin moving every frame re-runs only the cheap flat
//! subtract, not this walk.
//!
//! Descendants of a moved parent are re-walked automatically: the frontier pass
//! ([`super::frontier`]) expands the changed set through the child columns
//! GPU-side, and this walk consumes the expanded worklist via indirect dispatch.

#![allow(unsafe_code)]

use ash::vk;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    render_resource::{Buffer, BufferUsages},
    renderer::{RenderContext, RenderDevice},
};
use bytemuck::{Pod, Zeroable};
use core::ops::Range;
use wgpu::hal::api::Vulkan as VkApi;

use crate::ecs_gpu::{GpuColumn, GpuTable};
use crate::gpu::allocator::{Allocator, SparseBuffer};
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};

use super::frontier::{TransformFrontier, CONSUMER_ARGS_OFFSET};
use super::graph::{LocalRSColumn, LocalTranslationColumn, ParentColumn, TransformGraph};

const WORKGROUP_SIZE: u32 = 64;
/// Bytes per node in the absolute world's LINEAR buffer: 3 `vec4<f32>` = 48 B —
/// row k is (linear row k .xyz, unused .w), same packing as the relative world.
const WORLD_ABS_LINEAR_STRIDE: u64 = 48;
/// Bytes per node in the absolute world's TRANSLATION buffer: 3×`f64` = 24 B,
/// bound as a flat f64 array (a `vec3<f64>` would force 32-byte alignment).
const WORLD_ABS_T_STRIDE: u64 = 24;
/// Bytes per node in the RELATIVE world buffer: `mat3x4<f32>` = 48 B, same packing as
/// `Affine3x4` — what every RT consumer reads (`current_world()`).
const WORLD_REL_STRIDE: u64 = 48;
/// Virtual address space reserved for each sparse world buffer (pages committed on
/// growth). Fixed handle/address across growth → consumer bind groups stay valid;
/// 1 GiB covers ~22M nodes at the 48-byte strides. Cf. the AS-side
/// `*_VIRTUAL_BYTES` and `ecs_gpu::column`'s `COLUMN_VIRTUAL_BYTES`.
const WORLD_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;

/// Push params shared with `transform_propagate.slang::PropagateParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct PropagateParams {
    /// Threads to dispatch: `full_rebuild` → node_count; else changed-record count.
    count: u32,
    /// 1 → walk every node (id = slot); 0 → walk only the frontier worklist.
    full_rebuild: u32,
    /// X workgroup count of the full-rebuild 2D-split dispatch (the changed
    /// path is indirect; the shader recomputes its split from the worklist count).
    groups_x: u32,
    _pad: u32,
}

/// Render-world resource: the persistent world buffers + the ancestor-walk kernel.
#[derive(Resource)]
pub struct TransformPropagate {
    /// Persistent **sparse** absolute-world LINEAR buffer (3 `vec4<f32>` per node): the
    /// walk's f32 rotation·scale output. Only changed nodes are rewritten each frame;
    /// static nodes retain their value across growth (pages persist — no realloc).
    world_abs_linear: SparseBuffer,
    /// Persistent **sparse** absolute-world TRANSLATION buffer (flat f64 array, 3 per
    /// node): the walk's f64 translation output. The subtract pass reads this — and the
    /// origin (`world_abs_t[camera_slot]`) from it.
    world_abs_t: SparseBuffer,
    /// Persistent **sparse** RELATIVE world buffer (`mat3x4` per node): the subtract pass's
    /// output — the camera-origin-relative f32 world every RT consumer reads (`current_world`).
    world_rel: SparseBuffer,
    /// Slots whose pages are committed (and zeroed). Grows by `next_power_of_two`.
    capacity_slots: u32,
    node_count: u32,
    /// Threads to dispatch this frame (`full_rebuild` ? node_count : changed count).
    dispatch_count: u32,
    /// Whether the walk wrote any world this frame (or a growth needs the subtract's
    /// first fill) — gates the subtract pass, which re-runs iff a node (or the origin) moved.
    world_dirty: bool,
    /// The one-time cold-start guard: set at init, cleared only after
    /// [`dispatch_transform_propagate`] *actually* runs the full repopulate. While
    /// the frontier chain hasn't recorded yet the dispatch bails and the latch
    /// stays set, so the first successful frame walks every node created during
    /// warmup. After that it stays clear: new nodes are caught by the per-frame
    /// changed-path, and the worlds persist across growth (sparse — no realloc),
    /// so growth needs no re-rebuild. Same retain-until-consumed rule
    /// `dispatch_column` uses for its `pending`.
    needs_full_rebuild: bool,
    /// Range of newly-committed sparse pages (in *slots*) a growth needs zeroed (pages
    /// are UNDEFINED on first residency); [`dispatch_transform_propagate`] records the clear
    /// (scaled per-buffer by stride) before the walk, then takes it.
    pending_clear: Option<Range<u32>>,
    params: PropagateParams,
    /// 2D-split groups of the full-rebuild dispatch.
    groups: (u32, u32, u32),
    kernel: HeapKernel,
    slots: KernelSlots,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for TransformPropagate {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for TransformPropagate {}
unsafe impl Sync for TransformPropagate {}

impl TransformPropagate {
    /// The RELATIVE world buffer holding the origin-relative transforms. Every RT consumer
    /// (gather, rt_camera, readback) reads `world_rel[node_slot[i]]` from this persistent buffer.
    #[inline]
    pub fn current_world(&self) -> &Buffer {
        self.world_rel.buffer()
    }

    /// The absolute world's f32 LINEAR buffer (3 `vec4` per node) — the subtract pass
    /// copies rows from this.
    #[inline]
    pub fn world_abs_linear(&self) -> &Buffer {
        self.world_abs_linear.buffer()
    }

    /// The absolute world's f64 TRANSLATION buffer (flat f64 array, 3 per node) — the
    /// subtract pass reads this and the origin (`world_abs_t[camera_slot]`) from it.
    #[inline]
    pub fn world_abs_t(&self) -> &Buffer {
        self.world_abs_t.buffer()
    }

    /// Node count the world buffers cover this frame (for out-of-range guards).
    #[inline]
    pub fn node_count(&self) -> u32 {
        self.node_count
    }

    /// Whether the walk touched the absolute world this frame → the subtract pass must re-run.
    #[inline]
    pub fn world_dirty(&self) -> bool {
        self.world_dirty
    }
}

/// `RenderStartup` (after `SolariSetup`): compile the ancestor-walk kernel — a
/// layout-free heap pipeline ([`HeapKernel`]), Slang from source — and build the
/// (initially empty) sparse world buffers.
pub fn init_transform_propagate(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    // The sparse world buffers need the cluster allocator (created in
    // `SolariSetup`, which this runs after). Absent → device lacks the support
    // solari needs; skip (every consumer reads `Option<Res<TransformPropagate>>`).
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let Some(kernel) = HeapKernel::new(
        &seam,
        "transform_propagate.slang",
        include_str!("transform_propagate.slang"),
        "propagate",
        &[],
        &[],
        "transform_propagate",
        size_of::<PropagateParams>() as u32,
    ) else {
        return;
    };

    let world_abs_linear = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
        WORLD_VIRTUAL_BYTES,
        "transform.world_abs_linear",
    );
    let world_abs_t = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
        WORLD_VIRTUAL_BYTES,
        "transform.world_abs_t",
    );
    let world_rel = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
        WORLD_VIRTUAL_BYTES,
        "transform.world_rel",
    );
    commands.insert_resource(TransformPropagate {
        world_abs_linear,
        world_abs_t,
        world_rel,
        capacity_slots: 0,
        node_count: 0,
        dispatch_count: 0,
        world_dirty: false,
        // Armed once; the first frame the frontier chain records walks every node
        // that appeared during warmup, then it stays clear (see the field docs).
        needs_full_rebuild: true,
        pending_clear: None,
        params: PropagateParams::default(),
        groups: (0, 0, 0),
        kernel,
        slots: KernelSlots::new(&seam, 6),
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

/// `Render::Prepare` (after the column prepares): grow the persistent world
/// buffers (commit sparse pages) to the table's node high-water and pick this
/// frame's work. A growth latches `full_rebuild` (cold-start guard); otherwise we
/// walk only the local columns' changed records this frame.
pub fn prepare_transform_propagate(
    mut propagate: Option<ResMut<TransformPropagate>>,
    graph: Option<Res<TransformGraph>>,
    frontier: Option<Res<TransformFrontier>>,
) {
    let (Some(propagate), Some(graph), Some(frontier)) =
        (propagate.as_deref_mut(), graph, frontier)
    else {
        return;
    };

    propagate.node_count = graph.high_water();
    let high_water = propagate.node_count.max(1);

    // Growth commits more sparse pages to the same buffers — existing nodes' worlds
    // persist (no realloc), so growth needs NO full rebuild: new nodes are walked
    // by the per-frame changed-path on the frame their `local` is set, and static
    // nodes keep their value. (`needs_full_rebuild` is the one-time cold-start
    // guard, armed at init — deliberately not re-armed here; re-arming would
    // re-walk every node on each streaming growth, an O(n) hitch at 1M nodes.) The
    // newly-committed region is undefined, so it's queued for a zero-clear before
    // any read (see the dispatch). The clear range is in *slots*; the dispatch
    // scales it per-buffer by each stride.
    let grew = high_water > propagate.capacity_slots;
    if grew {
        let old_slots = propagate.capacity_slots;
        propagate.capacity_slots = high_water.next_power_of_two();
        let cap = propagate.capacity_slots;
        propagate
            .world_abs_linear
            .commit(0..cap as u64 * WORLD_ABS_LINEAR_STRIDE);
        propagate
            .world_abs_t
            .commit(0..cap as u64 * WORLD_ABS_T_STRIDE);
        propagate.world_rel.commit(0..cap as u64 * WORLD_REL_STRIDE);
        propagate.pending_clear = Some(old_slots..cap);
    }

    let full_rebuild = propagate.needs_full_rebuild;
    // The frontier's seed count (changed records + gpu-frame seeds) gates the walk;
    // the true dispatch count — seeds + GPU-expanded descendants — lives GPU-side
    // and is consumed via indirect dispatch.
    propagate.dispatch_count = if full_rebuild {
        propagate.node_count
    } else {
        frontier.seed_count()
    };
    // The subtract pass re-runs iff the walk wrote anything (a node — possibly the camera
    // origin — moved) or a growth needs its first fill; an idle frame leaves `world_rel` as is.
    propagate.world_dirty = propagate.dispatch_count > 0 || grew;
    propagate.groups =
        crate::ecs_gpu::linear_dispatch(propagate.dispatch_count.div_ceil(WORKGROUP_SIZE));
    propagate.params = PropagateParams {
        count: propagate.dispatch_count,
        full_rebuild: full_rebuild as u32,
        groups_x: propagate.groups.0,
        _pad: 0,
    };
}

/// `RenderGraph` (`Propagate`): one ancestor-walk dispatch. Each thread walks a
/// node's parent chain and writes its world — `full_rebuild` covers every node
/// (id = slot), otherwise just this frame's changed nodes. No ping-pong: a
/// thread reads only the read-only local/`parent` columns and writes its own
/// world slots, so a single pass is exact. A raw heap dispatch: buffer slots
/// rewritten per dispatch, params + slot array in push data.
pub fn dispatch_transform_propagate(
    propagate: Option<ResMut<TransformPropagate>>,
    frontier: Option<Res<TransformFrontier>>,
    seam: Option<Res<BindingSeam>>,
    local_t: Option<Res<GpuColumn<LocalTranslationColumn>>>,
    local_rs: Option<Res<GpuColumn<LocalRSColumn>>>,
    parent: Option<Res<GpuColumn<ParentColumn>>>,
    mut ctx: RenderContext,
) {
    let (Some(mut propagate), Some(seam)) = (propagate, seam) else {
        return;
    };
    let propagate = &mut *propagate;
    let clear = propagate.pending_clear.take();
    // Nothing to zero and nothing to walk → world persists from last frame.
    if clear.is_none() && propagate.dispatch_count == 0 {
        return;
    }
    // How to size the walk: full rebuild is a CPU-sized 2D-split; the changed
    // path consumes the frontier's consumer indirect args — gated on the
    // frontier having actually recorded this frame (its args are stale/zero
    // otherwise, and dispatching from them would count as "walked" while
    // touching none of the changed nodes).
    let indirect = (!propagate.needs_full_rebuild)
        .then(|| frontier.as_ref().filter(|f| f.ran()).map(|f| f.indirect_buffer()))
        .flatten();
    // The walk needs its input columns + the frontier worklist binding; the
    // zero-clear below runs regardless (a consumer must never read undefined
    // world data, even on a frame the walk skips).
    let blob = match (propagate.dispatch_count > 0, &frontier, &local_t, &local_rs, &parent) {
        (true, Some(frontier), Some(local_t), Some(local_rs), Some(parent))
            if propagate.needs_full_rebuild || indirect.is_some() =>
        {
            Some(propagate.kernel.push_blob(
                "transform_propagate",
                bytemuck::bytes_of(&propagate.params),
                &[
                    ("local_t", propagate.slots.buffer(&seam, 0, local_t.buffer())),
                    ("local_rs", propagate.slots.buffer(&seam, 1, local_rs.buffer())),
                    ("parent", propagate.slots.buffer(&seam, 2, parent.buffer())),
                    (
                        "world_abs_linear",
                        propagate
                            .slots
                            .buffer(&seam, 3, propagate.world_abs_linear.buffer()),
                    ),
                    (
                        "world_abs_t",
                        propagate.slots.buffer(&seam, 4, propagate.world_abs_t.buffer()),
                    ),
                    (
                        "frontier",
                        propagate.slots.buffer(&seam, 5, frontier.frontier_buffer()),
                    ),
                ],
            ))
        }
        _ => None,
    };
    let (gx, gy, gz) = propagate.groups;
    let mut walked = false;
    let encoder = ctx.command_encoder();
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // barriers bracket the fill + walk against the surrounding passes (raw
    // commands are invisible to wgpu's tracking).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &propagate.raw_device;
            // Zero the freshly-committed sparse region of ALL world buffers
            // (undefined on first residency) before the walk — runs even on a
            // frame the walk skips, so a consumer reading `world_rel` sees 0
            // (origin), never garbage/NaN.
            if let Some(slots) = &clear {
                let len = (slots.end - slots.start) as u64;
                if len > 0 {
                    let pre = [vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)];
                    dev.cmd_pipeline_barrier2(
                        cb,
                        &vk::DependencyInfo::default().memory_barriers(&pre),
                    );
                    for (buffer, stride) in [
                        (&propagate.world_abs_linear, WORLD_ABS_LINEAR_STRIDE),
                        (&propagate.world_abs_t, WORLD_ABS_T_STRIDE),
                        (&propagate.world_rel, WORLD_REL_STRIDE),
                    ] {
                        dev.cmd_fill_buffer(
                            cb,
                            buffer.raw(),
                            slots.start as u64 * stride,
                            len * stride,
                            0,
                        );
                    }
                    let post = [vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(
                            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                        )];
                    dev.cmd_pipeline_barrier2(
                        cb,
                        &vk::DependencyInfo::default().memory_barriers(&post),
                    );
                }
            }
            if let Some(blob) = blob.as_ref() {
                // The frontier's worklist/args writes -> our reads (and its
                // args as indirect commands on the changed path).
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
                dev.cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().memory_barriers(&barrier),
                );
                seam.bind_heaps(cb);
                seam.push_data(cb, blob);
                dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, propagate.kernel.pipeline);
                if let Some(indirect) = indirect {
                    let raw_indirect = indirect
                        .as_hal::<VkApi>()
                        .map(|b| b.raw_handle())
                        .expect("bevy_solari requires the Vulkan backend");
                    dev.cmd_dispatch_indirect(cb, raw_indirect, CONSUMER_ARGS_OFFSET);
                } else {
                    dev.cmd_dispatch(cb, gx, gy, gz);
                }
                walked = true;
                // Our world_abs writes -> the subtract/readback reads.
                let post = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(
                        vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                    )];
                dev.cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().memory_barriers(&post),
                );
            }
        });
    }
    // Clear the latch only once the repopulate actually ran — a growth frame that
    // bailed (columns not yet resident) keeps it set, re-firing the walk next
    // frame instead of losing it. (The clear above is one-shot: it was already
    // taken, and is re-recorded on the next growth.)
    if walked {
        propagate.needs_full_rebuild = false;
    } else if propagate.dispatch_count > 0 && !propagate.needs_full_rebuild {
        // Changed-path seeds existed but nothing walked them (the frontier chain
        // couldn't record this frame). The seeds' delta records are consumed by
        // the column scatter this frame — retrying the changed path next frame
        // would walk nothing, silently freezing those nodes at a zero world
        // (invisible instances, identity camera) for the whole session. Re-arm
        // the full-rebuild latch instead: one O(n) walk recovers every node once
        // everything lands — cold-start-priced, and only fires cold-start.
        propagate.needs_full_rebuild = true;
        bevy_log::info!(
            "transform: {} changed-path seeds unwalked (frontier ran: {}) — full walk re-armed",
            propagate.dispatch_count,
            frontier.as_ref().is_some_and(|f| f.ran()),
        );
    }
}
