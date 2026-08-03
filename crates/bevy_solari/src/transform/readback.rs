//! GPU → CPU `GlobalTransform` readback.
//!
//! With `TransformPlugin` disabled and the GPU owning propagation, a CPU system
//! that reads an entity's `GlobalTransform` would see a stale value. By default
//! this module writes it back: every entity whose world transform **changed this
//! frame** (the same `local`-column delta propagation processes) gets its
//! GPU-computed world copied back into `GlobalTransform`, 1–3 frames late (async,
//! no stall). Static entities are written once (when they first appear / move)
//! and then left — they don't change, so their `GlobalTransform` stays correct.
//!
//! **Opt out** with [`NoGpuGlobalTransformReadback`]: that entity's transform lives only on
//! the GPU (render-only), and its `GlobalTransform` is never written back. Tag
//! the render-only bulk (e.g. a city's static meshes, or animated-but-purely-
//! visual movers) to skip the writeback cost.
//!
//! Cost is change-driven (movers, not the whole scene), but it is **not free**:
//! reading back N movers/frame is ~`N×52 B` of transfer plus N scattered
//! `GlobalTransform` writes. Keep the *non*-opted-out moving set in mind.
//!
//! Flow: a GPU gather walks the `local` delta and atomic-appends `(slot, world)`
//! records — skipping nodes whose `no_readback` flag is set (the
//! [`NoGpuGlobalTransformReadback`] opt-out, scattered as a transform-table column) so
//! they never cost transfer — into a `ShaderBuffer` → bevy's [`Readback`] streams
//! it to the main world → an observer resolves the owning entity (carried in each
//! record as `Entity::to_bits`) and writes its `GlobalTransform`.

#![allow(unsafe_code)]

use ash::vk;
use bevy_app::{App, Startup};
use bevy_asset::{Assets, Handle, RenderAssetUsages};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    observer::On,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::{DAffine3, DMat3, DVec3};
use bevy_render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    gpu_readback::{Readback, ReadbackComplete},
    render_asset::RenderAssets,
    render_resource::BufferUsages,
    renderer::RenderContext,
    storage::{GpuShaderBuffer, ShaderBuffer},
};
use bevy_transform::components::GlobalTransform;
use bytemuck::{Pod, Zeroable};
use wgpu::hal::api::Vulkan as VkApi;

use crate::ecs_gpu::GpuColumn;
use crate::gpu::allocator::Allocator;
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};

use super::frontier::{TransformFrontier, CONSUMER_ARGS_OFFSET};
use super::graph::{NoReadbackColumn, NodeEntityColumn, ParentColumn};
use super::propagate::TransformPropagate;

/// Master switch for the whole readback (gather dispatch + the `GlobalTransform`
/// writeback). `false` leaves the buffer/`Readback` allocated but inert — use it
/// for a pure-render benchmark where no system reads GPU-authored globals on the
/// CPU. Per-entity opt-out is [`NoGpuGlobalTransformReadback`]; this is the global off.
const READBACK_ENABLED: bool = true;

/// `u32`s of header at the front of the readback buffer: `[count, _, _, _]`.
const HEADER_WORDS: u32 = 4;
/// `u32`s per record: `slot` + the absolute world's 3×3 linear (9 floats) + its
/// f64 translation (3×2 words) + the owning entity's bits (`[lo, hi]` — the
/// identity the writeback resolves by).
const RECORD_WORDS: u32 = 18;
/// Max records read back per frame; the output buffer is sized to this. The
/// transfer window is count-scoped from the previous delivery (see
/// [`write_readback_global_transforms`]), so the full `capacity × 72 B` is not
/// streamed every frame. A frame whose changed set exceeds it drops the
/// overflow (warning). Tune to the scene's moving set.
const READBACK_CAPACITY: u32 = 131072;
/// Idle frames to retain a gather's records (keep the count header) after it ran. The transfer
/// window is sized reactively from the previous delivery's count (see
/// [`write_readback_global_transforms`]), so a burst landing after a low-count idle — a `.bsn`
/// spawning all its static parts at once — is delivered through a too-small window and the overflow
/// dropped. Statics upload `local` once, so a dropped first-sight record would never retry and the
/// node's `GlobalTransform` stays default forever. Retaining lets the now-grown window re-read the
/// same records until all are delivered; covers the 1–3 frame readback latency twice over.
const HEADER_RETAIN_FRAMES: u32 = 8;

/// Opt **out** of `GlobalTransform` readback: this entity's transform is
/// render-only (lives only on the GPU), so the CPU `GlobalTransform` is never
/// written back. Two reasons to tag it:
/// - **Perf:** the render-only bulk never needs a CPU `GlobalTransform`, so skip
///   the per-mover writeback + transfer.
/// - **Correctness:** an entity whose `GlobalTransform` is CPU-authored (e.g. a
///   camera/light/UI node kept alive by a CPU carve-out while `TransformPlugin`
///   is disabled) must NOT be readback — the lagged GPU value would clobber the
///   current CPU one. Derives `Default` so it works as a required component
///   (`register_required_components::<Node, NoGpuGlobalTransformReadback>()`).
#[derive(Component, Default, Clone)]
pub struct NoGpuGlobalTransformReadback;

/// Main-world handle to the readback output `ShaderBuffer` (the gather writes it;
/// bevy's [`Readback`] streams it back). Extracted to the render world so the
/// gather dispatch can resolve the prepared GPU buffer.
#[derive(Resource, Clone, ExtractResource)]
pub struct TransformReadbackTarget {
    buffer: Handle<ShaderBuffer>,
}

/// Push params shared with `transform_readback.slang::ReadbackParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct ReadbackParams {
    /// World-buffer node coverage (out-of-range guard).
    node_count: u32,
    /// Max output records (overflow drops past this).
    capacity: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Render-world resource: the gather heap kernel + its slots.
#[derive(Resource)]
pub struct TransformReadback {
    params: ReadbackParams,
    changed_count: u32,
    /// Idle frames left to retain the last gather's records (see [`HEADER_RETAIN_FRAMES`]).
    header_retain: u32,
    kernel: HeapKernel,
    slots: KernelSlots,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for TransformReadback {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for TransformReadback {}
unsafe impl Sync for TransformReadback {}

/// Main world: create the readback output buffer and spawn the [`Readback`] that
/// streams it back, with the decode-and-write observer attached. Runs once.
pub fn setup_transform_readback(mut commands: Commands, mut buffers: ResMut<Assets<ShaderBuffer>>) {
    let words = (HEADER_WORDS + READBACK_CAPACITY * RECORD_WORDS) as usize;
    let mut buffer = ShaderBuffer::with_size(words * 4, RenderAssetUsages::RENDER_WORLD);
    // STORAGE: the gather writes it. COPY_SRC: `Readback` streams it.
    // COPY_DST: `prepare` resets the count header each frame via `write_buffer`.
    buffer.buffer_description.usage =
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let handle = buffers.add(buffer);

    commands.insert_resource(TransformReadbackTarget {
        buffer: handle.clone(),
    });
    commands
        .spawn(Readback::buffer(handle))
        .observe(write_readback_global_transforms);
}

/// `RenderStartup` (after `SolariSetup`): compile the gather kernel — a
/// layout-free heap pipeline ([`HeapKernel`]), Slang from source.
pub fn init_transform_readback(
    mut commands: Commands,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let Some(kernel) = HeapKernel::new(
        &seam,
        "transform_readback.slang",
        include_str!("transform_readback.slang"),
        "readback",
        &[],
        &[],
        "transform_readback",
        size_of::<ReadbackParams>() as u32,
    ) else {
        return;
    };
    commands.insert_resource(TransformReadback {
        params: ReadbackParams::default(),
        changed_count: 0,
        header_retain: 0,
        kernel,
        slots: KernelSlots::new(&seam, 7),
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

/// `Render::Prepare` (after the column prepares): set params from the `local`
/// delta + reset the output count header to 0 (the gather atomic-appends from
/// there). The readback set is the changed nodes — same delta propagation walks.
pub fn prepare_transform_readback(
    mut readback: Option<ResMut<TransformReadback>>,
    frontier: Option<Res<TransformFrontier>>,
    propagate: Option<Res<TransformPropagate>>,
    target: Option<Res<TransformReadbackTarget>>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    render_queue: Res<bevy_render::renderer::RenderQueue>,
) {
    let (Some(readback), Some(frontier), Some(propagate), Some(target)) =
        (readback.as_deref_mut(), frontier, propagate, target)
    else {
        return;
    };
    // Seed count is the CPU-visible "anything moved?" gate; the true gather count
    // (seeds + GPU-expanded descendants) is in the frontier header, dispatched indirect.
    readback.changed_count = frontier.seed_count();
    readback.params = ReadbackParams {
        node_count: propagate.node_count(),
        capacity: READBACK_CAPACITY,
        _pad0: 0,
        _pad1: 0,
    };

    // Reset the atomic count header before a gather that repopulates it; otherwise
    // RETAIN the last gather's records for a few idle frames so a transfer window that
    // throttled the burst can re-deliver the dropped first-sight records (see
    // `HEADER_RETAIN_FRAMES` — statics upload `local` once, so a drop is never retried).
    let reset_header = if readback.changed_count > 0 {
        readback.header_retain = HEADER_RETAIN_FRAMES;
        true
    } else if readback.header_retain > 0 {
        readback.header_retain -= 1;
        false
    } else {
        true
    };
    if reset_header {
        if let Some(out) = gpu_buffers.get(&target.buffer) {
            render_queue.write_buffer(&out.buffer, 0, bytemuck::bytes_of(&0u32));
        }
    }
}

/// `RenderGraph` (`Propagate`, after the gather): walk the frontier worklist and
/// atomic-append `(slot, world[slot])` for each changed node. bevy's `Readback`
/// (`RenderSystems::Cleanup`, after this) streams the buffer to the main world.
/// A raw heap dispatch: buffer slots rewritten per dispatch, params + slot array
/// in push data, consuming the frontier's consumer indirect args.
pub fn dispatch_transform_readback(
    readback: Option<Res<TransformReadback>>,
    frontier: Option<Res<TransformFrontier>>,
    seam: Option<Res<BindingSeam>>,
    propagate: Option<Res<TransformPropagate>>,
    no_readback: Option<Res<GpuColumn<NoReadbackColumn>>>,
    parent: Option<Res<GpuColumn<ParentColumn>>>,
    entity: Option<Res<GpuColumn<NodeEntityColumn>>>,
    target: Option<Res<TransformReadbackTarget>>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    mut ctx: RenderContext,
) {
    if !READBACK_ENABLED {
        return;
    }
    let (
        Some(readback),
        Some(frontier),
        Some(seam),
        Some(propagate),
        Some(no_readback),
        Some(parent),
        Some(entity),
        Some(target),
    ) = (
        readback, frontier, seam, propagate, no_readback, parent, entity, target,
    )
    else {
        return;
    };
    if readback.changed_count == 0 {
        return; // nothing moved → the reset-to-0 header already says count = 0.
    }
    if !frontier.ran() {
        return; // the frontier chain skipped this frame — its indirect args are stale.
    }
    let Some(out) = gpu_buffers.get(&target.buffer) else {
        return;
    };
    let blob = readback.kernel.push_blob(
        "transform_readback",
        bytemuck::bytes_of(&readback.params),
        &[
            (
                "frontier",
                readback.slots.buffer(&seam, 0, frontier.frontier_buffer()),
            ),
            (
                "world_abs_linear",
                readback.slots.buffer(&seam, 1, propagate.world_abs_linear()),
            ),
            (
                "world_abs_t",
                readback.slots.buffer(&seam, 2, propagate.world_abs_t()),
            ),
            ("out", readback.slots.buffer(&seam, 3, &out.buffer)),
            (
                "no_readback",
                readback.slots.buffer(&seam, 4, no_readback.buffer()),
            ),
            ("parent", readback.slots.buffer(&seam, 5, parent.buffer())),
            ("node_entity", readback.slots.buffer(&seam, 6, entity.buffer())),
        ],
    );
    let encoder = ctx.command_encoder();
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // barriers bracket this dispatch against the surrounding passes (raw
    // dispatches are invisible to wgpu's tracking).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &readback.raw_device;
            let raw_indirect = frontier
                .indirect_buffer()
                .as_hal::<VkApi>()
                .map(|b| b.raw_handle())
                .expect("bevy_solari requires the Vulkan backend");
            // The walk's world writes + the frontier's worklist/args writes ->
            // our reads (and the args as indirect commands).
            let pre = [vk::MemoryBarrier2::default()
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
            dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&pre));
            seam.bind_heaps(cb);
            seam.push_data(cb, &blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, readback.kernel.pipeline);
            dev.cmd_dispatch_indirect(cb, raw_indirect, CONSUMER_ARGS_OFFSET);
            // Our record writes -> the `Readback` transfer that streams the
            // buffer to the CPU (and any downstream compute).
            let post = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::COPY,
                )
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_READ
                        | vk::AccessFlags2::SHADER_WRITE
                        | vk::AccessFlags2::TRANSFER_READ,
                )];
            dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&post));
        });
    }
}

/// Main world: decode a delivered readback buffer and write `GlobalTransform` for
/// each record. Opt-out ([`NoGpuGlobalTransformReadback`]) is enforced GPU-side (those
/// nodes never make it into the buffer), so this resolves the owning entity straight
/// from the record (it carries `Entity::to_bits`) and writes — then **count-scopes**
/// the next readback's transfer to this frame's record count (it can't be sized
/// GPU-side per frame, so we drive bevy's `Readback` range from the lagged count).
fn write_readback_global_transforms(
    event: On<ReadbackComplete>,
    mut transforms: Query<&mut GlobalTransform>,
    mut readbacks: Query<&mut Readback>,
) {
    let words: &[u32] = bytemuck::cast_slice(&event.data);
    let header = HEADER_WORDS as usize;
    let rec = RECORD_WORDS as usize;
    if words.len() <= header {
        return;
    }
    // Total records the GPU wrote this frame (its atomic counter). The range may
    // have lagged it (records past the transferred window land next cycle once
    // the range grows below).
    let true_count = words[0];
    if true_count > READBACK_CAPACITY {
        bevy_log::warn_once!(
            "More than {READBACK_CAPACITY} entities need `GlobalTransform` readback this frame; \
             the overflow won't be written back. Raise `READBACK_CAPACITY` or mark render-only \
             entities `NoGpuGlobalTransformReadback`."
        );
    }
    let transferred = (words.len() - header) / rec;
    let count = (true_count as usize)
        .min(transferred)
        .min(READBACK_CAPACITY as usize);

    // Size the NEXT transfer to this frame's count + headroom for growth, so a
    // small / mostly-opted-out changed set doesn't stream the whole buffer.
    if let Ok(mut readback) = readbacks.get_mut(event.entity) {
        let next_records = (true_count + true_count / 4 + 64).min(READBACK_CAPACITY);
        let size_bytes = u64::from((HEADER_WORDS + next_records * RECORD_WORDS) * 4);
        if let Readback::Buffer {
            start_offset_and_size,
            ..
        } = &mut *readback
        {
            *start_offset_and_size = Some((0, size_bytes));
        }
    }

    for k in 0..count {
        let base = header + k * rec;
        // Resolve the owning entity straight from the record (entity-keyed identity,
        // lag-independent): a recycled slot's new occupant carries a different entity,
        // and a despawned occupant's `get_mut` fails — either way no stale splat. The
        // record's slot field (`words[base]`) is unused here; the gather only needs it
        // to index `world`.
        let bits = (words[base + 16] as u64) | ((words[base + 17] as u64) << 32);
        let Some(entity) = Entity::try_from_bits(bits) else {
            continue; // never-written / torn slot id — skip.
        };
        let Ok(mut global) = transforms.get_mut(entity) else {
            continue; // despawned / recycled since dispatch — skip.
        };
        // The ABSOLUTE world: 9-float linear rows + f64 translation word pairs.
        // Rebuild the column-major `DAffine3`: column j = (row0[j], row1[j], row2[j]).
        // Absolute — the readback GlobalTransform holds true f64 world positions and
        // never changes when only the camera (origin) moves.
        let f = |w: usize| f32::from_bits(words[base + 1 + w]);
        let (r0x, r0y, r0z) = (f(0), f(1), f(2));
        let (r1x, r1y, r1z) = (f(3), f(4), f(5));
        let (r2x, r2y, r2z) = (f(6), f(7), f(8));
        let d = |w: usize| {
            f64::from_bits(
                u64::from(words[base + 10 + w * 2]) | (u64::from(words[base + 11 + w * 2]) << 32),
            )
        };
        let (tx, ty, tz) = (d(0), d(1), d(2));
        *global = GlobalTransform::from(DAffine3 {
            matrix3: DMat3::from_cols(
                DVec3::new(f64::from(r0x), f64::from(r1x), f64::from(r2x)),
                DVec3::new(f64::from(r0y), f64::from(r1y), f64::from(r2y)),
                DVec3::new(f64::from(r0z), f64::from(r1z), f64::from(r2z)),
            ),
            translation: DVec3::new(tx, ty, tz),
        });
    }
}

/// Main-app wiring: the readback output buffer + the `Readback`/observer.
pub(super) fn build_readback_main(app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<TransformReadbackTarget>::default())
        .add_systems(Startup, setup_transform_readback);
}
