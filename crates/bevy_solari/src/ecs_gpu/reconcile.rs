//! GPU instance reconcile pass — folds the CPU's absolute-state change journal
//! ([`crate::instance::RtJournal`]) into the per-instance GPU columns. One thread
//! per journal record writes **all** of a slot's columns from the single record,
//! so a reused slot is re-initialized atomically (no partial/stale column — the
//! aliasing fix). See `reconcile.wgsl`.
//!
//! The pass is self-contained: it owns its bind-group layout, pipeline id, and
//! params buffer, and dispatches in [`SolariClusterSystems::Scatter`] alongside the
//! per-column scatter. The journal carries the same per-slot values the scatter
//! derives from its deltas, so the two writes agree; making the reconcile the sole
//! writer (and deleting the redundant scatter) is the authority-flip step, gated on
//! GPU validation.

use bevy_app::{App, Plugin};
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
        ShaderStages, ShaderType, UniformBuffer,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    Render, RenderApp, RenderStartup, RenderSystems,
};
use bytemuck::{Pod, Zeroable};

use crate::instance::{PartitionColumn, 
    GeometryIdColumn, GroupBaseColumn, LodInputColumn, NodeSlotColumn, RtJournal,
};
use crate::ecs_gpu::GpuColumn;

const WORKGROUP_SIZE: u32 = 64;

/// Uniform shared with `reconcile.wgsl::ReconcileParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
struct ReconcileParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Render-world resource: the reconcile pipeline + its bind group.
#[derive(Resource)]
pub struct RtReconcile {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
    params: UniformBuffer<ReconcileParams>,
    bind_group: Option<BindGroup>,
    /// Journal record count this frame (dispatch bound).
    count: u32,
}

fn reconcile_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "rt_reconcile",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 journal
                uniform_buffer::<ReconcileParams>(false),    // 1 params
                storage_buffer_sized(false, None),           // 2 node_slots (rw)
                storage_buffer_sized(false, None),           // 3 geometry_ids (rw)
                storage_buffer_sized(false, None),           // 4 group_bases (rw)
                storage_buffer_sized(false, None),           // 5 lod_inputs (rw)
                storage_buffer_sized(false, None),           // 6 partition_hints (rw)
            ),
        ),
    )
}

/// `RenderStartup`: create the layout + queue the pipeline + insert [`RtReconcile`].
/// No-op when the journal is absent (non-solari device → reconcile guards on it).
pub fn init_rt_reconcile(
    mut commands: Commands,
    journal: Option<Res<RtJournal>>,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
    mut registry: ResMut<super::SolariPipelineRegistry>,
) {
    if journal.is_none() {
        return;
    }
    let layout = reconcile_bind_group_layout();
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("rt_reconcile".into()),
        layout: vec![layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "reconcile.wgsl"),
        entry_point: Some("reconcile_apply".into()),
        ..Default::default()
    });
    registry.register("rt_reconcile", pipeline);
    let mut params = UniformBuffer::<ReconcileParams>::default();
    params.set_label(Some("rt_reconcile"));
    commands.insert_resource(RtReconcile {
        layout,
        pipeline,
        params,
        bind_group: None,
        count: 0,
    });
}

/// `Render::Prepare`: publish this frame's journal record count.
pub fn prepare_rt_reconcile(
    reconcile: Option<ResMut<RtReconcile>>,
    journal: Option<Res<RtJournal>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(mut reconcile), Some(journal)) = (reconcile, journal) else {
        return;
    };
    reconcile.count = journal.count;
    *reconcile.params.get_mut() = ReconcileParams {
        count: journal.count,
        ..Default::default()
    };
    reconcile.params.write_buffer(&render_device, &render_queue);
}

/// `Render::PrepareBindGroups`: build the reconcile bind group. The journal ring and
/// every column buffer are stable-address (sparse), so once built it stays valid.
#[allow(clippy::too_many_arguments)]
pub fn prepare_rt_reconcile_bind_group(
    reconcile: Option<ResMut<RtReconcile>>,
    journal: Option<Res<RtJournal>>,
    node_slots: Option<Res<GpuColumn<NodeSlotColumn>>>,
    geometry_ids: Option<Res<GpuColumn<GeometryIdColumn>>>,
    group_bases: Option<Res<GpuColumn<GroupBaseColumn>>>,
    lod_inputs: Option<Res<GpuColumn<LodInputColumn>>>,
    partition_hints: Option<Res<GpuColumn<PartitionColumn>>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let (
        Some(mut reconcile),
        Some(journal),
        Some(node_slots),
        Some(geometry_ids),
        Some(group_bases),
        Some(lod_inputs),
        Some(partition_hints),
    ) = (reconcile, journal, node_slots, geometry_ids, group_bases, lod_inputs, partition_hints)
    else {
        return;
    };
    let Some(params) = reconcile.params.binding() else {
        return;
    };
    // Bind the WHOLE sparse buffer (stable handle) per column, not the committed
    // range: the columns grow (regen), so a committed-sized binding would freeze
    // at the first size and the reconcile couldn't write slots past it. The
    // reconcile is slot-indexed and never calls `arrayLength`, so the whole-range
    // bind is safe (no `arrayLength` hang) and growth-proof — every slot
    // `< high_water` is always in range.
    let layout = pipeline_cache.get_bind_group_layout(&reconcile.layout);
    reconcile.bind_group = Some(render_device.create_bind_group(
        "rt_reconcile",
        &layout,
        &BindGroupEntries::sequential((
            journal.buffer.buffer().as_entire_binding(),
            params,
            node_slots.buffer().as_entire_binding(),
            geometry_ids.buffer().as_entire_binding(),
            group_bases.buffer().as_entire_binding(),
            lod_inputs.buffer().as_entire_binding(),
            partition_hints.buffer().as_entire_binding(),
        )),
    ));
}

/// `RenderGraph` (`Scatter`): apply this frame's journal to the columns. Clears the
/// journal's pending records (`mark_folded`) **only** after a real dispatch, so a
/// cold-pipeline frame retains them to retry next frame (the retain-until-folded latch).
pub fn dispatch_rt_reconcile(
    reconcile: Option<Res<RtReconcile>>,
    journal: Option<ResMut<RtJournal>>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let Some(reconcile) = reconcile else {
        return;
    };
    if reconcile.count == 0 {
        return;
    }
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(reconcile.pipeline) else {
        return;
    };
    let Some(bind_group) = reconcile.bind_group.as_ref() else {
        return;
    };
    let groups = crate::ecs_gpu::linear_dispatch(reconcile.count.div_ceil(WORKGROUP_SIZE));
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("rt_reconcile"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    drop(pass);
    // Folded: the GPU consumed the journal from its buffer this frame, so the CPU
    // staging can be dropped. Until this runs (cold pipeline / no bind group), the
    // records stay live and are re-uploaded next frame.
    if let Some(mut journal) = journal {
        journal.mark_folded();
    }
}

/// Wires the reconcile pass (embedded shader + the prepare/dispatch systems).
pub struct ReconcilePlugin;

impl Plugin for ReconcilePlugin {
    fn build(&self, app: &mut App) {
        bevy_asset::embedded_asset!(app, "reconcile.wgsl");
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            // After `init_rt_journal` (which is itself after `SolariSetup`/the
            // allocator): the reconcile only initializes when the journal exists, so
            // it must observe the journal already inserted this startup.
            .add_systems(
                RenderStartup,
                init_rt_reconcile.after(crate::instance::init_rt_journal),
            )
            .add_systems(
                Render,
                (
                    // MUST run after the journal upload — it sets `journal.count`, which
                    // this reads for the dispatch bound + params. Without the order, a
                    // stale (0) count means the reconcile never dispatches.
                    prepare_rt_reconcile
                        .in_set(RenderSystems::PrepareResources)
                        .after(crate::instance::upload_rt_journal),
                    prepare_rt_reconcile_bind_group.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            // Dispatch in `Propagate` (after the whole `Scatter` set): `GpuColumn`'s
            // zero-clear of newly-committed pages runs in `Scatter`'s `dispatch_column`,
            // so writing the columns there would race the clear on a growth/regen frame.
            // Ordered before the gather, which reads `node_slot`.
            .add_systems(
                bevy_render::renderer::RenderGraph,
                dispatch_rt_reconcile
                    .in_set(crate::SolariClusterSystems::Propagate)
                    .before(crate::transform::dispatch_transform_gather),
            );
    }
}
