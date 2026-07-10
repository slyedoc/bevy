//! GPU frontier expansion — turn this frame's *changed* nodes into changed nodes
//! **plus all their descendants**, entirely on the GPU.
//!
//! The propagate walk recomposes only the nodes in its dispatch list; a moving
//! parent's descendants didn't change their own `local`, so they used to need a
//! CPU-side marker + subtree re-push (the late `SolariFrame`). This pass replaces that: the changed
//! `local` delta (plus [`GpuFrameSeeds`] — GPU-moved nodes with no CPU change edge)
//! seeds a worklist that expands level-by-level through the `first_child` /
//! `next_sibling` columns, deduped by a per-node frame-epoch stamp. The result —
//! `frontier[HEADER..HEADER+total]` — is the propagate/readback dispatch list,
//! consumed via `dispatch_workgroups_indirect` with args this pass writes.
//!
//! Kernel sequence (one compute pass; WebGPU orders storage/indirect access
//! between dispatches): `seed` → (`finalize` → `expand`)×MAX_LEVELS → `finalize`.
//! `finalize` (1 thread) closes the current level's bounds and writes the next
//! expand's indirect args; empty levels dispatch zero workgroups.

use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
        BufferDescriptor, BufferUsages, ComputePassDescriptor, PipelineCache, ShaderStages,
        ShaderType, StorageBuffer, UniformBuffer,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
};
use bytemuck::{Pod, Zeroable};

use crate::ecs_gpu::{GpuColumn, GpuTable};
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;

use super::graph::{
    FirstChildColumn, GpuFrameSeeds, LocalTranslationColumn, NextSiblingColumn, ParentColumn,
    TransformGraph,
};

const WORKGROUP_SIZE: u32 = 64;

/// `SOLARI_XFORM_DEBUG=1`: trace the changed-path seed/walk decisions — the
/// silent-bail points where cold-start seed loss hides.
pub(crate) fn xform_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SOLARI_XFORM_DEBUG").as_deref() == Ok("1"))
}
/// Max expansion depth (hierarchy levels below a moved node). Must match
/// `transform_frontier.wgsl`; deeper descendants go stale (mirror of the walk's
/// `MAX_DEPTH = 64` guard — realistic scenes are ≤ 8 deep).
pub const MAX_LEVELS: u32 = 16;
/// Header u32s at the front of the frontier buffer (must match the WGSL):
/// `[total(atomic), total_plain, current_level, pad, level_begin[MAX_LEVELS+2], pad…]`.
pub const FRONTIER_HEADER_WORDS: u32 = 4 + MAX_LEVELS + 2 + 2; // = 24, 16B-aligned
/// Indirect-args u32s: `(x,y,z)` per expand level + one shared entry for the
/// propagate/readback consumers at `[MAX_LEVELS * 3..]`.
const INDIRECT_WORDS: u64 = ((MAX_LEVELS + 1) * 3) as u64;
/// Byte offset of the propagate/readback consumers' indirect args.
pub const CONSUMER_ARGS_OFFSET: u64 = (MAX_LEVELS * 3) as u64 * 4;

/// Uniform shared with `transform_frontier.wgsl::FrontierParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Render-world resource: the frontier worklist + epoch + indirect buffers and
/// the seed/expand/finalize pipelines' shared bind group.
#[derive(Resource)]
pub struct TransformFrontier {
    /// `[header, nodes…]` worklist (see [`FRONTIER_HEADER_WORDS`]). Plain buffer,
    /// pow2-regrown; consumers rebind every frame.
    frontier: Buffer,
    /// Per-node frame-epoch stamp (dedupe). Zero-init on (re)creation; `frame_id`
    /// only grows, so recreation can't alias a live stamp.
    epoch: Buffer,
    /// Dispatch args: per-level expand + the shared consumer entry.
    indirect: Buffer,
    /// Stand-in for the indirect binding in the seed/expand bind group: those
    /// kernels never touch it, and binding the real one as `read_write` storage
    /// would conflict with the same dispatch's INDIRECT usage (exclusive).
    indirect_dummy: Buffer,
    /// [`GpuFrameSeeds`] uploaded for the seed kernel.
    extra_seeds: StorageBuffer<Vec<u32>>,
    params: UniformBuffer<FrontierParams>,
    capacity_slots: u32,
    frame_id: u32,
    seed_count: u32,
    /// Whether the seed/expand/finalize chain actually recorded this frame. The
    /// walk's changed path must NOT dispatch from the indirect args otherwise —
    /// they're stale (or zero), and the seeds' delta records are consumed by the
    /// column scatter this frame, so a silent skip loses those nodes for good.
    ran: bool,
    /// seed/expand (dummy in the indirect slot).
    bind_group_walk: Option<BindGroup>,
    /// finalize (the real indirect-args buffer).
    bind_group_finalize: Option<BindGroup>,
}

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

/// The frontier bind-group layout (shared by seed/expand/finalize). Owned by
/// [`SolariResourceManager`].
pub fn transform_frontier_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "transform_frontier",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 changed (local delta records)
                storage_buffer_read_only_sized(false, None), // 1 extra seeds (gpu-frame slots)
                storage_buffer_read_only_sized(false, None), // 2 first_child
                storage_buffer_read_only_sized(false, None), // 3 next_sibling
                storage_buffer_sized(false, None),           // 4 frontier (rw, atomic header)
                storage_buffer_sized(false, None),           // 5 epoch (rw, atomic)
                storage_buffer_sized(false, None),           // 6 indirect args (rw)
                uniform_buffer::<FrontierParams>(false),     // 7 params
            ),
        ),
    )
}

/// `RenderStartup`: params + the fixed-size indirect buffer; the worklist/epoch
/// buffers are created on first growth in prepare.
pub fn init_transform_frontier(mut commands: Commands, render_device: Res<RenderDevice>) {
    let mut params = UniformBuffer::<FrontierParams>::default();
    params.set_label(Some("transform_frontier"));
    let indirect = render_device.create_buffer(&BufferDescriptor {
        label: Some("transform.frontier_indirect"),
        size: INDIRECT_WORDS * 4,
        usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let indirect_dummy = render_device.create_buffer(&BufferDescriptor {
        label: Some("transform.frontier_indirect_dummy"),
        size: INDIRECT_WORDS * 4,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut extra_seeds = StorageBuffer::<Vec<u32>>::default();
    extra_seeds.set_label(Some("transform.frontier_extra_seeds"));
    commands.insert_resource(TransformFrontier {
        frontier: make_frontier_buffer(&render_device, 1),
        epoch: make_epoch_buffer(&render_device, 1),
        indirect,
        indirect_dummy,
        extra_seeds,
        params,
        capacity_slots: 1,
        frame_id: 0,
        seed_count: 0,
        ran: false,
        bind_group_walk: None,
        bind_group_finalize: None,
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

    *frontier.params.get_mut() = FrontierParams {
        changed_count,
        record_stride: local_t.record_stride(),
        extra_count: extra.len() as u32,
        frame_id: frontier.frame_id,
        node_count,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    frontier.params.write_buffer(&render_device, &render_queue);
}

/// `Render::PrepareBindGroups`: rebuild every frame (worklist/epoch/delta buffers
/// can all reallocate).
pub fn prepare_transform_frontier_bind_group(
    mut frontier: Option<ResMut<TransformFrontier>>,
    resource_manager: Option<Res<SolariResourceManager>>,
    local_t: Option<Res<GpuColumn<LocalTranslationColumn>>>,
    first_child: Option<Res<GpuColumn<FirstChildColumn>>>,
    next_sibling: Option<Res<GpuColumn<NextSiblingColumn>>>,
    parent: Option<Res<GpuColumn<ParentColumn>>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let (
        Some(frontier),
        Some(resource_manager),
        Some(local_t),
        Some(first_child),
        Some(next_sibling),
        Some(parent),
    ) = (
        frontier.as_deref_mut(),
        resource_manager,
        local_t,
        first_child,
        next_sibling,
        parent,
    )
    else {
        return;
    };
    let (Some(params), Some(extra)) = (frontier.params.binding(), frontier.extra_seeds.binding())
    else {
        frontier.bind_group_walk = None;
        frontier.bind_group_finalize = None;
        return;
    };
    // Empty-delta fallback: the shader never reads past `changed_count == 0`.
    let changed = local_t.delta_buffer().unwrap_or_else(|| parent.buffer());
    let layout = pipeline_cache.get_bind_group_layout(&resource_manager.transform_frontier);
    let make = |indirect: &Buffer, label| {
        render_device.create_bind_group(
            label,
            &layout,
            &BindGroupEntries::sequential((
                changed.as_entire_binding(),
                extra.clone(),
                first_child.buffer().as_entire_binding(),
                next_sibling.buffer().as_entire_binding(),
                frontier.frontier.as_entire_binding(),
                frontier.epoch.as_entire_binding(),
                indirect.as_entire_binding(),
                params.clone(),
            )),
        )
    };
    let walk = make(&frontier.indirect_dummy, "transform_frontier_walk");
    let finalize = make(&frontier.indirect, "transform_frontier_finalize");
    frontier.bind_group_walk = Some(walk);
    frontier.bind_group_finalize = Some(finalize);
}

/// `RenderGraph` (`Propagate`, before the walk): seed, then alternate finalize /
/// indirect expand per level, then a final finalize that publishes the consumer
/// count + indirect args.
pub fn dispatch_transform_frontier(
    frontier: Option<ResMut<TransformFrontier>>,
    pipelines: Res<SolariPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let Some(mut frontier) = frontier else {
        return;
    };
    let frontier = &mut *frontier;
    if frontier.seed_count == 0 {
        return;
    }
    let (Some(seed), Some(expand), Some(finalize)) = (
        pipeline_cache.get_compute_pipeline(pipelines.transform_frontier_seed),
        pipeline_cache.get_compute_pipeline(pipelines.transform_frontier_expand),
        pipeline_cache.get_compute_pipeline(pipelines.transform_frontier_finalize),
    ) else {
        if xform_debug() {
            bevy_log::info!(
                "frontier: bail, pipelines cold (seed_count {})",
                frontier.seed_count
            );
        }
        return;
    };
    if xform_debug() && (frontier.bind_group_walk.is_none() || frontier.bind_group_finalize.is_none()) {
        bevy_log::info!(
            "frontier: bail, bind groups missing (seed_count {}, params {:?}, extra_seeds buffer {:?})",
            frontier.seed_count,
            frontier.params.binding().is_some(),
            frontier.extra_seeds.buffer().map(|b| b.size()),
        );
    }
    let (Some(walk_group), Some(finalize_group)) = (
        frontier.bind_group_walk.as_ref(),
        frontier.bind_group_finalize.as_ref(),
    ) else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("transform_frontier"),
        timestamp_writes: None,
    });
    let d = diagnostics.time_span(&mut pass, "transform_frontier");

    pass.set_pipeline(seed);
    pass.set_bind_group(0, walk_group, &[]);
    let (gx, gy, gz) =
        crate::ecs_gpu::linear_dispatch(frontier.seed_count.div_ceil(WORKGROUP_SIZE));
    pass.dispatch_workgroups(gx, gy, gz);

    for level in 0..MAX_LEVELS {
        pass.set_pipeline(finalize);
        pass.set_bind_group(0, finalize_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(expand);
        pass.set_bind_group(0, walk_group, &[]);
        pass.dispatch_workgroups_indirect(&frontier.indirect, u64::from(level) * 12);
    }
    // Close the last level + publish `total_plain` and the consumer indirect args.
    pass.set_pipeline(finalize);
    pass.set_bind_group(0, finalize_group, &[]);
    pass.dispatch_workgroups(1, 1, 1);

    d.end(&mut pass);
    drop(pass);
    frontier.ran = true;
}
