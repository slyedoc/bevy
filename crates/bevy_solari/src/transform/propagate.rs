//! GPU transform propagation — single-pass ancestor-walk over the changed set.
//!
//! `world[node]` is a pure function of the node's own `local`/`parent` ancestor
//! chain (`world = local[root] ∘ … ∘ local[parent] ∘ local[node]`), independent
//! of every other node's world. So one thread walks a node's chain and writes
//! its world in a SINGLE pass — no Jacobi iteration, no ping-pong, no hazard.
//!
//! The buffer is **persistent**: each frame we recompute only the nodes whose
//! `local` changed (the column's delta — `changed[k*stride]` is the slot), and
//! static nodes keep last frame's value. This is the symmetric twin of the PTLAS
//! move detection — use the change delta we already build instead of brute-forcing
//! all nodes every frame. A capacity growth repopulates the whole buffer once
//! (`full_rebuild`: one thread per node).
//!
//! Boundary (see the shader): updates a *changed* node's own world. Animating a
//! parent (descendants move without their own `local` changing) or a bare
//! re-parent won't re-walk the descendants. bevy_city animates only leaves, so
//! it's exact there; a general hierarchy needs descendant dirtying too — later.

use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
        BufferUsages, ComputePassDescriptor, PipelineCache, ShaderStages, ShaderType, UniformBuffer,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use core::ops::Range;

use crate::ecs_gpu::{GpuColumn, GpuTable};
use crate::gpu::allocator::{Allocator, SparseBuffer};
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;

use super::graph::{LocalColumn, ParentColumn, TransformGraph};

const WORKGROUP_SIZE: u32 = 64;
/// Bytes per node world entry: `mat3x4<f32>` = 48 B, same packing as `Affine3x4`.
const WORLD_STRIDE: u64 = 48;
/// Virtual address space reserved for the sparse `world` buffer (pages committed
/// on growth). Fixed handle/address across growth → consumer bind groups stay
/// valid; 1 GiB covers ~22M nodes at the 48-byte stride. Cf. the AS-side
/// `*_VIRTUAL_BYTES` and `ecs_gpu::column`'s `COLUMN_VIRTUAL_BYTES`.
const WORLD_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;

/// Uniform shared with `transform_propagate.wgsl::PropagateParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
struct PropagateParams {
    /// Threads to dispatch: `full_rebuild` → node_count; else changed-record count.
    count: u32,
    /// Words per changed record (`WORDS + 1`); slot at `k * record_stride`.
    record_stride: u32,
    /// 1 → walk every node (id = slot); 0 → walk only `changed[k*stride]`.
    full_rebuild: u32,
    _pad: u32,
}

/// Render-world resource: the persistent world buffer + the ancestor-walk pipeline.
#[derive(Resource)]
pub struct TransformPropagate {
    /// Persistent **sparse** world buffer (`mat3x4` per node): a fixed virtual
    /// range reserved once, pages committed on growth. Only changed nodes are
    /// rewritten each frame; static nodes retain their value across growth (the
    /// pages persist — no realloc). The gather pass reads this.
    world: SparseBuffer,
    /// Slots whose pages are committed (and zeroed). Grows by `next_power_of_two`.
    capacity_slots: u32,
    node_count: u32,
    /// Threads to dispatch this frame (`full_rebuild` ? node_count : changed count).
    dispatch_count: u32,
    /// The one-time cold-start guard: set at init, cleared only after
    /// [`dispatch_transform_propagate`] *actually* runs the full repopulate. While
    /// the propagate pipeline is still compiling the dispatch bails and the latch
    /// stays set, so the first successful frame walks every node created during
    /// warmup (the old all-black-on-load race). After that it stays clear: new
    /// nodes are caught by the per-frame changed-path, and `world` persists across
    /// growth (sparse — no realloc), so growth needs no re-rebuild. Same retain-
    /// until-consumed rule `dispatch_column` uses for its `pending`.
    needs_full_rebuild: bool,
    /// Byte range of newly-committed sparse pages a growth needs zeroed (pages are
    /// UNDEFINED on first residency); [`dispatch_transform_propagate`] records the
    /// clear before the walk, then takes it.
    pending_clear: Option<Range<u64>>,
    params: UniformBuffer<PropagateParams>,
    bind_group: Option<BindGroup>,
}

/// The propagate bind-group layout. Owned by
/// [`SolariResourceManager`](crate::resource_manager::SolariResourceManager); kept
/// here next to [`PropagateParams`] and the bind-group code that must match it.
pub fn transform_propagate_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "transform_propagate",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 local
                storage_buffer_read_only_sized(false, None), // 1 parent
                storage_buffer_sized(false, None),           // 2 world (rw, persistent)
                storage_buffer_read_only_sized(false, None), // 3 changed (delta records)
                uniform_buffer::<PropagateParams>(false),    // 4 params
            ),
        ),
    )
}

impl TransformPropagate {
    /// The world buffer holding the propagated transforms. The gather pass reads
    /// `world[node_slot[i]]` from this persistent buffer.
    #[inline]
    pub fn current_world(&self) -> &Buffer {
        self.world.buffer()
    }

    /// Node count the world buffer covers this frame (for out-of-range guards).
    #[inline]
    pub fn node_count(&self) -> u32 {
        self.node_count
    }
}

/// `RenderStartup`: build the ancestor-walk bind-group layout + compute pipeline
/// and the (initially empty) sparse world buffer.
pub fn init_transform_propagate(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
) {
    // The sparse `world` buffer needs the cluster allocator (created in
    // `SolariSetup`, which this runs after). Absent → device lacks the support
    // solari needs; skip (every consumer reads `Option<Res<TransformPropagate>>`).
    let Some(allocator) = allocator else {
        return;
    };
    // The bind-group layout lives in `SolariResourceManager`; the compute pipeline
    // id lives in `SolariPipelines`. This init owns only the pass's buffers.
    let mut params = UniformBuffer::<PropagateParams>::default();
    params.set_label(Some("transform_propagate"));

    let world = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
        WORLD_VIRTUAL_BYTES,
        "transform.world",
    );
    commands.insert_resource(TransformPropagate {
        world,
        capacity_slots: 0,
        node_count: 0,
        dispatch_count: 0,
        // Armed once; the first frame the pipeline is ready walks every node that
        // appeared during warmup, then it stays clear (see the field docs).
        needs_full_rebuild: true,
        pending_clear: None,
        params,
        bind_group: None,
    });
}

/// `Render::Prepare` (after the column prepares): grow the persistent world
/// buffer (commit sparse pages) to the table's node high-water and pick this
/// frame's work. A growth latches `full_rebuild` (cold-start guard); otherwise we
/// walk only the `local` column's changed records this frame.
pub fn prepare_transform_propagate(
    mut propagate: Option<ResMut<TransformPropagate>>,
    graph: Option<Res<TransformGraph>>,
    local: Option<Res<GpuColumn<LocalColumn>>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(propagate), Some(graph), Some(local)) = (propagate.as_deref_mut(), graph, local)
    else {
        return;
    };
    propagate.node_count = graph.high_water();
    let high_water = propagate.node_count.max(1);

    // Growth commits more sparse pages to the same buffer — existing nodes' worlds
    // persist (no realloc), so growth needs NO full rebuild: new nodes are walked
    // by the per-frame changed-path on the frame their `local` is set, and static
    // nodes keep their value. (`needs_full_rebuild` is the one-time cold-start
    // guard, armed at init — deliberately not re-armed here; re-arming would
    // re-walk every node on each streaming growth, an O(n) hitch at 1M nodes.) The
    // newly-committed region is undefined, so it's queued for a zero-clear before
    // any read (see the dispatch).
    let grew = high_water > propagate.capacity_slots;
    if grew {
        let old_slots = propagate.capacity_slots;
        propagate.capacity_slots = high_water.next_power_of_two();
        let committed = propagate.capacity_slots as u64 * WORLD_STRIDE;
        propagate.world.commit(0..committed);
        propagate.pending_clear = Some(old_slots as u64 * WORLD_STRIDE..committed);
    }

    let full_rebuild = propagate.needs_full_rebuild;
    let changed_count = local.pending();
    propagate.dispatch_count = if full_rebuild {
        propagate.node_count
    } else {
        changed_count
    };
    *propagate.params.get_mut() = PropagateParams {
        count: propagate.dispatch_count,
        record_stride: local.record_stride(),
        full_rebuild: full_rebuild as u32,
        _pad: 0,
    };
    propagate.params.write_buffer(&render_device, &render_queue);
}

/// `Render::PrepareBindGroups`: (re)build the bind group. Rebuilt every frame —
/// cheap, and transparently picks up a `local`/`parent`/delta buffer that
/// reallocated in its own prepare.
pub fn prepare_transform_propagate_bind_groups(
    mut propagate: Option<ResMut<TransformPropagate>>,
    resource_manager: Option<Res<SolariResourceManager>>,
    local: Option<Res<GpuColumn<LocalColumn>>>,
    parent: Option<Res<GpuColumn<ParentColumn>>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let (Some(propagate), Some(resource_manager), Some(local), Some(parent)) =
        (propagate.as_deref_mut(), resource_manager, local, parent)
    else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&resource_manager.transform_propagate);
    let Some(params) = propagate.params.binding() else {
        return;
    };
    // The changed-slot list. When empty (`pending == 0`) the dispatch is skipped
    // or the shader returns before reading it, so a fallback buffer is harmless —
    // bind the always-present `local` buffer so the bind group is still valid.
    let changed = local.delta_buffer().unwrap_or_else(|| local.buffer());

    let bind_group = render_device.create_bind_group(
        "transform_propagate",
        &layout,
        &BindGroupEntries::sequential((
            local.buffer().as_entire_binding(),
            parent.buffer().as_entire_binding(),
            propagate.world.buffer().as_entire_binding(),
            changed.as_entire_binding(),
            params,
        )),
    );
    propagate.bind_group = Some(bind_group);
}

/// `RenderGraph` (`Propagate`): one ancestor-walk pass. Each thread walks a
/// node's parent chain and writes its world — `full_rebuild` covers every node
/// (id = slot), otherwise just this frame's changed nodes. No ping-pong: a
/// thread reads only the read-only `local`/`parent` columns and writes its own
/// `world` slot, so a single pass is exact.
pub fn dispatch_transform_propagate(
    propagate: Option<ResMut<TransformPropagate>>,
    pipelines: Res<SolariPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let Some(mut propagate) = propagate else {
        return;
    };
    let clear = propagate.pending_clear.take();
    // Nothing to zero and nothing to walk → world persists from last frame.
    if clear.is_none() && propagate.dispatch_count == 0 {
        return;
    }
    // `None` until the pipeline compiles → the walk is skipped this frame (but the
    // clear above still runs, so a consumer never reads undefined `world`).
    let pipeline = pipeline_cache.get_compute_pipeline(pipelines.transform_propagate);
    let groups = propagate.dispatch_count.div_ceil(WORKGROUP_SIZE);
    let mut walked = false;
    {
        let diagnostics = ctx.diagnostic_recorder();
        let diagnostics = diagnostics.as_deref();
        let encoder = ctx.command_encoder();
        // Zero the freshly-committed sparse region (undefined on first residency)
        // before the walk — no pipeline needed, so even a cold-pipeline frame
        // leaves a consumer reading `world` at 0 (origin), never garbage/NaN.
        if let Some(range) = &clear {
            let len = range.end - range.start;
            if len > 0 {
                encoder.clear_buffer(&propagate.world.wgpu_buffer, range.start, Some(len));
            }
        }
        if propagate.dispatch_count > 0 {
            if let (Some(pipeline), Some(bind_group)) = (pipeline, propagate.bind_group.as_ref()) {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("transform_propagate"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                let d = diagnostics.time_span(&mut pass, "transform_propagate");
                pass.dispatch_workgroups(groups, 1, 1);
                d.end(&mut pass);
                walked = true;
            }
        }
    }
    // Clear the latch only once the repopulate actually ran — a growth frame that
    // bailed (pipeline/bind group not yet compiled) keeps it set, re-firing the
    // walk next frame instead of losing it. (The clear above is one-shot: it was
    // already taken, and is re-recorded on the next growth.)
    if walked {
        propagate.needs_full_rebuild = false;
    }
}
