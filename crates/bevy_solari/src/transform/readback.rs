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
//! it to the main world → an observer resolves slot→entity (the allocator's
//! reverse map) and writes `GlobalTransform`.

use bevy_app::{App, Startup};
use bevy_asset::{Assets, Handle, RenderAssetUsages};
use bevy_ecs::{
    component::Component,
    observer::On,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::{Affine3A, Mat3A, Vec3A};
use bevy_render::{
    diagnostic::RecordDiagnostics as _,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    gpu_readback::{Readback, ReadbackComplete},
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        BufferUsages, ComputePassDescriptor, PipelineCache, ShaderStages, ShaderType,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    storage::{GpuShaderBuffer, ShaderBuffer},
};
use bevy_transform::components::GlobalTransform;
use bytemuck::{Pod, Zeroable};

use crate::ecs_gpu::{GpuColumn, GpuSlotAllocator};
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;

use super::graph::{LocalColumn, NoReadbackColumn, ParentColumn, TransformGraph};
use super::propagate::TransformPropagate;

/// Master switch for the whole readback (gather dispatch + the `GlobalTransform`
/// writeback). `false` leaves the buffer/`Readback` allocated but inert — use it
/// for a pure-render benchmark where no system reads GPU-authored globals on the
/// CPU. Per-entity opt-out is [`NoGpuGlobalTransformReadback`]; this is the global off.
const READBACK_ENABLED: bool = true;

const WORKGROUP_SIZE: u32 = 64;
/// `u32`s of header at the front of the readback buffer: `[count, _, _, _]`.
const HEADER_WORDS: u32 = 4;
/// `u32`s per record: `slot` + a `mat3x4` world transform (12 floats).
const RECORD_WORDS: u32 = 13;
/// Max records read back per frame. The output buffer is sized to this and bevy's
/// `Readback` streams the **whole** buffer each frame (it can't size to the live
/// count), so this is also the per-frame transfer (`capacity × 52 B` ≈ 6.8 MB at
/// 131072). A frame whose changed set exceeds it drops the overflow (warning).
/// Tune to the scene's moving set; count-scoping the transfer is a follow-up.
const READBACK_CAPACITY: u32 = 131072;

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
#[derive(Component, Default)]
pub struct NoGpuGlobalTransformReadback;

/// Main-world handle to the readback output `ShaderBuffer` (the gather writes it;
/// bevy's [`Readback`] streams it back). Extracted to the render world so the
/// gather bind group can resolve the prepared GPU buffer.
#[derive(Resource, Clone, ExtractResource)]
pub struct TransformReadbackTarget {
    buffer: Handle<ShaderBuffer>,
}

/// Uniform shared with `transform_readback.wgsl::ReadbackParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
struct ReadbackParams {
    /// Number of changed records in the `local` delta this frame (dispatch bound).
    changed_count: u32,
    /// Words per `local` delta record (`WORDS + 1`); slot at `k * record_stride`.
    record_stride: u32,
    /// World-buffer node coverage (out-of-range guard).
    node_count: u32,
    /// Max output records (overflow drops past this).
    capacity: u32,
}

/// Render-world resource: the gather pipeline + its params/bind group.
#[derive(Resource)]
pub struct TransformReadback {
    params: bevy_render::render_resource::UniformBuffer<ReadbackParams>,
    bind_group: Option<BindGroup>,
    changed_count: u32,
}

/// The readback bind-group layout. Owned by
/// [`SolariResourceManager`](crate::resource_manager::SolariResourceManager).
pub fn transform_readback_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "transform_readback",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 local delta (changed records)
                storage_buffer_read_only_sized(false, None), // 1 world
                storage_buffer_sized(false, None),           // 2 out (rw, atomic count + records)
                uniform_buffer::<ReadbackParams>(false),     // 3 params
                storage_buffer_read_only_sized(false, None), // 4 no_readback (per-node opt-out flag)
                storage_buffer_read_only_sized(false, None), // 5 parent (ancestor walk for cascade)
            ),
        ),
    )
}

/// Main world: create the readback output buffer and spawn the [`Readback`] that
/// streams it back, with the decode-and-write observer attached. Runs once.
pub fn setup_transform_readback(
    mut commands: Commands,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
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

/// `RenderStartup`: the readback pass owns only its params buffer + bind group;
/// the layout lives in `SolariResourceManager`, the pipeline id in `SolariPipelines`.
pub fn init_transform_readback(mut commands: Commands) {
    let mut params = bevy_render::render_resource::UniformBuffer::<ReadbackParams>::default();
    params.set_label(Some("transform_readback"));

    commands.insert_resource(TransformReadback {
        params,
        bind_group: None,
        changed_count: 0,
    });
}

/// `Render::Prepare` (after the column prepares): set params from the `local`
/// delta + reset the output count header to 0 (the gather atomic-appends from
/// there). The readback set is the changed nodes — same delta propagation walks.
pub fn prepare_transform_readback(
    mut readback: ResMut<TransformReadback>,
    local: Option<Res<GpuColumn<LocalColumn>>>,
    propagate: Option<Res<TransformPropagate>>,
    target: Option<Res<TransformReadbackTarget>>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(local), Some(propagate), Some(target)) = (local, propagate, target) else {
        return;
    };
    readback.changed_count = local.pending();
    *readback.params.get_mut() = ReadbackParams {
        changed_count: readback.changed_count,
        record_stride: local.record_stride(),
        node_count: propagate.node_count(),
        capacity: READBACK_CAPACITY,
    };
    readback.params.write_buffer(&render_device, &render_queue);

    // Reset the atomic count header for this frame's atomic-append.
    if let Some(out) = gpu_buffers.get(&target.buffer) {
        render_queue.write_buffer(&out.buffer, 0, bytemuck::bytes_of(&0u32));
    }
}

/// `Render::PrepareBindGroups`: (re)build the gather bind group. Needs the
/// prepared output `ShaderBuffer`; if it isn't ready, leaves `None` (skip).
pub fn prepare_transform_readback_bind_group(
    mut readback: ResMut<TransformReadback>,
    resource_manager: Option<Res<SolariResourceManager>>,
    local: Option<Res<GpuColumn<LocalColumn>>>,
    no_readback: Option<Res<GpuColumn<NoReadbackColumn>>>,
    parent: Option<Res<GpuColumn<ParentColumn>>>,
    propagate: Option<Res<TransformPropagate>>,
    target: Option<Res<TransformReadbackTarget>>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let (Some(resource_manager), Some(local), Some(no_readback), Some(parent), Some(propagate), Some(target)) =
        (resource_manager, local, no_readback, parent, propagate, target)
    else {
        readback.bind_group = None;
        return;
    };
    let (Some(out), Some(params)) = (gpu_buffers.get(&target.buffer), readback.params.binding())
    else {
        readback.bind_group = None;
        return;
    };
    // When the delta is empty, fall back to the column buffer for the binding
    // (the shader returns before reading it — `changed_count` is 0).
    let changed = local.delta_buffer().unwrap_or_else(|| local.buffer());
    let layout = pipeline_cache.get_bind_group_layout(&resource_manager.transform_readback);
    readback.bind_group = Some(render_device.create_bind_group(
        "transform_readback",
        &layout,
        &BindGroupEntries::sequential((
            changed.as_entire_binding(),
            propagate.current_world().as_entire_binding(),
            out.buffer.as_entire_binding(),
            params,
            no_readback.buffer().as_entire_binding(),
            parent.buffer().as_entire_binding(),
        )),
    ));
}

/// `RenderGraph` (`Propagate`, after the gather): walk the `local` delta and
/// atomic-append `(slot, world[slot])` for each changed node. bevy's `Readback`
/// (`RenderSystems::Cleanup`, after this) streams the buffer to the main world.
pub fn dispatch_transform_readback(
    readback: Option<Res<TransformReadback>>,
    pipelines: Res<SolariPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !READBACK_ENABLED {
        return;
    }
    let Some(readback) = readback else {
        return;
    };
    if readback.changed_count == 0 {
        return; // nothing moved → the reset-to-0 header already says count = 0.
    }
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.transform_readback) else {
        return;
    };
    let Some(bind_group) = readback.bind_group.as_ref() else {
        return;
    };
    let groups = readback.changed_count.div_ceil(WORKGROUP_SIZE);
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("transform_readback"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    // Times the gather + ancestor-walk (the per-node `parent`-chain opt-out scan).
    let d = diagnostics.time_span(&mut pass, "transform_readback");
    pass.dispatch_workgroups(groups, 1, 1);
    d.end(&mut pass);
}

/// Main world: decode a delivered readback buffer and write `GlobalTransform` for
/// each record. Opt-out ([`NoGpuGlobalTransformReadback`]) is enforced GPU-side (those
/// nodes never make it into the buffer), so this just resolves slot→entity via
/// the allocator's reverse map and writes — then **count-scopes** the next
/// readback's transfer to this frame's record count (it can't be sized GPU-side
/// per frame, so we drive bevy's `Readback` range from the lagged count).
fn write_readback_global_transforms(
    event: On<ReadbackComplete>,
    allocator: Res<GpuSlotAllocator<TransformGraph>>,
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
        let slot = words[base];
        let Some(entity) = allocator.entity(slot) else {
            continue;
        };
        let Ok(mut global) = transforms.get_mut(entity) else {
            continue; // despawned / reused slot since dispatch — skip.
        };
        // 3 rows of a `mat3x4` (row k = linear row k .xyz, translation k .w).
        // Rebuild the column-major `Affine3A`: column j = (row0[j], row1[j], row2[j]).
        let f = |w: usize| f32::from_bits(words[base + 1 + w]);
        let (r0x, r0y, r0z, tx) = (f(0), f(1), f(2), f(3));
        let (r1x, r1y, r1z, ty) = (f(4), f(5), f(6), f(7));
        let (r2x, r2y, r2z, tz) = (f(8), f(9), f(10), f(11));
        *global = GlobalTransform::from(Affine3A {
            matrix3: Mat3A::from_cols(
                Vec3A::new(r0x, r1x, r2x),
                Vec3A::new(r0y, r1y, r2y),
                Vec3A::new(r0z, r1z, r2z),
            ),
            translation: Vec3A::new(tx, ty, tz),
        });
    }
}

/// Main-app wiring: the readback output buffer + the `Readback`/observer.
pub(super) fn build_readback_main(app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<TransformReadbackTarget>::default())
        .add_systems(Startup, setup_transform_readback);
}
