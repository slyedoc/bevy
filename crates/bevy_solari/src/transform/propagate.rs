//! GPU transform propagation — single-pass ancestor-walk over the changed set,
//! producing each node's ABSOLUTE world (native-`f64` translation via `SHADER_F64`).
//!
//! `world[node]` is a pure function of the node's own `local`/`parent` ancestor
//! chain (`world = local[root] ∘ … ∘ local[parent] ∘ local[node]`), independent
//! of every other node's world. So one thread walks a node's chain and writes
//! its world in a SINGLE pass — no Jacobi iteration, no ping-pong, no hazard.
//!
//! The buffers are **persistent**: each frame we recompute only the nodes whose
//! `local` changed (the column's delta — `changed[k*stride]` is the slot), and
//! static nodes keep last frame's value. This is the symmetric twin of the PTLAS
//! move detection — use the change delta we already build instead of brute-forcing
//! all nodes every frame. A capacity growth repopulates the whole buffer once
//! (`full_rebuild`: one thread per node).
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

use ash::vk;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
        BufferUsages, ComputePassDescriptor, PipelineCache, ShaderStages, ShaderType,
        UniformBuffer,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
};
use bytemuck::{Pod, Zeroable};
use core::ops::Range;

use crate::ecs_gpu::{GpuColumn, GpuTable};
use crate::gpu::allocator::{Allocator, SparseBuffer};
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;

use super::frontier::{TransformFrontier, CONSUMER_ARGS_OFFSET};
use super::graph::{LocalRSColumn, LocalTranslationColumn, ParentColumn, TransformGraph};

const WORKGROUP_SIZE: u32 = 64;
/// Bytes per node in the absolute world's LINEAR buffer: 3 `vec4<f32>` = 48 B —
/// row k is (linear row k .xyz, unused .w), same packing as the relative world.
const WORLD_ABS_LINEAR_STRIDE: u64 = 48;
/// Bytes per node in the absolute world's TRANSLATION buffer: 3×`f64` = 24 B,
/// bound as a flat `array<f64>` (a `vec3<f64>` would force 32-byte alignment).
const WORLD_ABS_T_STRIDE: u64 = 24;
/// Bytes per node in the RELATIVE world buffer: `mat3x4<f32>` = 48 B, same packing as
/// `Affine3x4` — what every RT consumer reads (`current_world()`).
const WORLD_REL_STRIDE: u64 = 48;
/// Virtual address space reserved for each sparse world buffer (pages committed on
/// growth). Fixed handle/address across growth → consumer bind groups stay valid;
/// 1 GiB covers ~22M nodes at the 48-byte strides. Cf. the AS-side
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

/// Render-world resource: the persistent world buffers + the ancestor-walk pipeline.
#[derive(Resource)]
pub struct TransformPropagate {
    /// Persistent **sparse** absolute-world LINEAR buffer (3 `vec4<f32>` per node): the
    /// walk's f32 rotation·scale output. Only changed nodes are rewritten each frame;
    /// static nodes retain their value across growth (pages persist — no realloc).
    world_abs_linear: SparseBuffer,
    /// Persistent **sparse** absolute-world TRANSLATION buffer (flat `array<f64>`, 3 per
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
    /// the propagate pipeline is still compiling the dispatch bails and the latch
    /// stays set, so the first successful frame walks every node created during
    /// warmup (the old all-black-on-load race). After that it stays clear: new
    /// nodes are caught by the per-frame changed-path, and the worlds persist across
    /// growth (sparse — no realloc), so growth needs no re-rebuild. Same retain-
    /// until-consumed rule `dispatch_column` uses for its `pending`.
    needs_full_rebuild: bool,
    /// Range of newly-committed sparse pages (in *slots*) a growth needs zeroed (pages
    /// are UNDEFINED on first residency); [`dispatch_transform_propagate`] records the clear
    /// (scaled per-buffer by stride) before the walk, then takes it.
    pending_clear: Option<Range<u32>>,
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
                storage_buffer_read_only_sized(false, None), // 0 local_t (array<f64>)
                storage_buffer_read_only_sized(false, None), // 1 local_rs (array<f32>)
                storage_buffer_read_only_sized(false, None), // 2 parent
                storage_buffer_sized(false, None),           // 3 world_abs_linear (rw, persistent)
                storage_buffer_sized(false, None),           // 4 world_abs_t (rw, persistent)
                storage_buffer_read_only_sized(false, None), // 5 frontier worklist (count + slots)
                uniform_buffer::<PropagateParams>(false),    // 6 params
            ),
        ),
    )
}

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

    /// The absolute world's f64 TRANSLATION buffer (flat `array<f64>`, 3 per node) — the
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

/// `RenderStartup`: build the ancestor-walk bind-group layout + compute pipeline
/// and the (initially empty) sparse world buffers.
pub fn init_transform_propagate(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
) {
    // The sparse world buffers need the cluster allocator (created in
    // `SolariSetup`, which this runs after). Absent → device lacks the support
    // solari needs; skip (every consumer reads `Option<Res<TransformPropagate>>`).
    let Some(allocator) = allocator else {
        return;
    };
    // The bind-group layout lives in `SolariResourceManager`; the compute pipeline
    // id lives in `SolariPipelines`. This init owns only the pass's buffers.
    let mut params = UniformBuffer::<PropagateParams>::default();
    params.set_label(Some("transform_propagate"));

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
        // Armed once; the first frame the pipeline is ready walks every node that
        // appeared during warmup, then it stays clear (see the field docs).
        needs_full_rebuild: true,
        pending_clear: None,
        params,
        bind_group: None,
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
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
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
    *propagate.params.get_mut() = PropagateParams {
        count: propagate.dispatch_count,
        record_stride: 1,
        full_rebuild: full_rebuild as u32,
        _pad: 0,
    };
    propagate.params.write_buffer(&render_device, &render_queue);
}

/// `Render::PrepareBindGroups`: (re)build the bind group. Rebuilt every frame —
/// cheap, and transparently picks up a local/`parent`/delta buffer that
/// reallocated in its own prepare.
pub fn prepare_transform_propagate_bind_groups(
    mut propagate: Option<ResMut<TransformPropagate>>,
    resource_manager: Option<Res<SolariResourceManager>>,
    local_t: Option<Res<GpuColumn<LocalTranslationColumn>>>,
    local_rs: Option<Res<GpuColumn<LocalRSColumn>>>,
    parent: Option<Res<GpuColumn<ParentColumn>>>,
    frontier: Option<Res<TransformFrontier>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let (
        Some(propagate),
        Some(resource_manager),
        Some(local_t),
        Some(local_rs),
        Some(parent),
        Some(frontier),
    ) = (
        propagate.as_deref_mut(),
        resource_manager,
        local_t,
        local_rs,
        parent,
        frontier,
    )
    else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&resource_manager.transform_propagate);
    let Some(params) = propagate.params.binding() else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        "transform_propagate",
        &layout,
        &BindGroupEntries::sequential((
            local_t.buffer().as_entire_binding(),
            local_rs.buffer().as_entire_binding(),
            parent.buffer().as_entire_binding(),
            propagate.world_abs_linear.buffer().as_entire_binding(),
            propagate.world_abs_t.buffer().as_entire_binding(),
            frontier.frontier_buffer().as_entire_binding(),
            params,
        )),
    );
    propagate.bind_group = Some(bind_group);
}

/// `RenderGraph` (`Propagate`): one ancestor-walk pass. Each thread walks a
/// node's parent chain and writes its world — `full_rebuild` covers every node
/// (id = slot), otherwise just this frame's changed nodes. No ping-pong: a
/// thread reads only the read-only local/`parent` columns and writes its own
/// world slots, so a single pass is exact.
pub fn dispatch_transform_propagate(
    propagate: Option<ResMut<TransformPropagate>>,
    frontier: Option<Res<TransformFrontier>>,
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
    // clear above still runs, so a consumer never reads undefined world data).
    let pipeline = pipeline_cache.get_compute_pipeline(pipelines.transform_propagate);
    let groups = propagate.dispatch_count.div_ceil(WORKGROUP_SIZE);
    let mut walked = false;
    {
        let diagnostics = ctx.diagnostic_recorder();
        let diagnostics = diagnostics.as_deref();
        let encoder = ctx.command_encoder();
        // Zero the freshly-committed sparse region of ALL world buffers (undefined on
        // first residency) before the walk — no pipeline needed, so even a cold-pipeline
        // frame leaves a consumer reading `world_rel` at 0 (origin), never garbage/NaN.
        if let Some(slots) = &clear {
            let len = (slots.end - slots.start) as u64;
            if len > 0 {
                for (buffer, stride) in [
                    (&propagate.world_abs_linear, WORLD_ABS_LINEAR_STRIDE),
                    (&propagate.world_abs_t, WORLD_ABS_T_STRIDE),
                    (&propagate.world_rel, WORLD_REL_STRIDE),
                ] {
                    encoder.clear_buffer(
                        &buffer.wgpu_buffer,
                        slots.start as u64 * stride,
                        Some(len * stride),
                    );
                }
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
                if propagate.needs_full_rebuild {
                    // 2D-split to stay under the 65535 per-dimension dispatch limit
                    // (node_count / 64 exceeds it past ~4.2M nodes); the shader
                    // reconstructs the flat index from `gid` + `num_workgroups`.
                    let (gx, gy, gz) = crate::ecs_gpu::linear_dispatch(groups);
                    pass.dispatch_workgroups(gx, gy, gz);
                    walked = true;
                } else if let Some(frontier) = frontier.as_ref().filter(|f| f.ran()) {
                    // Changed path: the dispatch size is GPU-side (seeds + expanded
                    // descendants) — consume the frontier's indirect args. Gated on
                    // the frontier having actually recorded this frame: its args are
                    // stale/zero otherwise, and dispatching from them would count as
                    // "walked" while touching none of the changed nodes.
                    pass.dispatch_workgroups_indirect(
                        frontier.indirect_buffer(),
                        CONSUMER_ARGS_OFFSET,
                    );
                    walked = true;
                }
                d.end(&mut pass);
            }
        }
    }
    // Clear the latch only once the repopulate actually ran — a growth frame that
    // bailed (pipeline/bind group not yet compiled) keeps it set, re-firing the
    // walk next frame instead of losing it. (The clear above is one-shot: it was
    // already taken, and is re-recorded on the next growth.)
    if walked {
        propagate.needs_full_rebuild = false;
    } else if propagate.dispatch_count > 0 && !propagate.needs_full_rebuild {
        // Changed-path seeds existed but nothing walked them (the frontier chain or
        // the walk itself couldn't record this frame). The seeds' delta records are
        // consumed by the column scatter this frame — retrying the changed path
        // next frame would walk nothing, silently freezing those nodes at a zero
        // world (invisible instances, identity camera) for the whole session.
        // Re-arm the full-rebuild latch instead: one O(n) walk recovers every node
        // once everything lands — cold-start-priced, and only fires cold-start.
        propagate.needs_full_rebuild = true;
        bevy_log::info!(
            "transform: {} changed-path seeds unwalked (frontier ran: {}) — full walk re-armed",
            propagate.dispatch_count,
            frontier.as_ref().is_some_and(|f| f.ran()),
        );
    }
}
