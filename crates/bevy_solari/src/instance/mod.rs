//! Instance domain — per-`RaytracingMesh3d` render-world state.
//!
//! - [`instance_manager`] — change-driven extract of mesh entities into
//!   stable per-entity slots + the per-slot metadata GPU buffers
//!   (group bases, LOD inputs, geometry ids) and per-frame deltas.
//! - [`gpu_instances`] — the diff-driven GPU-scatter columns
//!   (transforms / previous transforms / material ids), keyed by slot.
//! - [`material_slots`] — stable `SolariMaterial` → slot allocation the
//!   column scatter resolves `material_id` against.

use bevy_app::{App, Plugin};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    renderer::RenderGraph, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};

use crate::bindings::RaytracingMesh3d;
use crate::ecs_gpu::GpuColumnPrepareSet;
// Material-slot allocation lives in `crate::material` now (an asset-keyed
// `ecs_gpu::SlotPool`); the instance plugin schedules its prepare since the
// per-instance `MaterialColumn` resolves against it.
use crate::material::{
    init_material_slots, prepare_material_slots, prepare_material_traversal_flags,
    MaterialTraversalFlags,
};
use crate::SolariClusterSystems;

pub mod gpu_instances;
pub mod instance_manager;
pub mod journal;

pub use gpu_instances::{
    cluster_columns_ready, Affine3x4, GeometryIdColumn, GroupBaseColumn, GpuInstancesPlugin,
    InstanceColumns, InstanceMaskColumn, LodInputColumn, MaterialColumn, NodeSlotColumn,
    TransformColumn,
};
pub use instance_manager::{
    clear_instance_deltas, flush_cluster_instances, free_cluster_slot, init_instance_manager,
    log_slot_ratchet, mark_instance_added, mark_instance_layers_changed,
    mark_instance_material_changed, resolve_instance_material_ids, InstanceManager,
    RaytracingGpuEntity, RtInstanceChanges, RtSlotMap,
};
pub use journal::{
    init_rt_journal, upload_rt_journal, InstanceJournalRecord, RtJournal, JOURNAL_OP_UPSERT,
};

/// Instance domain plugin: per-`RaytracingMesh3d` slot tracking, the
/// diff-driven transform/material column scatter, material-slot
/// resolution, and the extract + despawn-observer that drive them.
pub struct InstancePlugin;

impl Plugin for InstancePlugin {
    fn build(&self, app: &mut App) {
        bevy_shader::load_shader_library!(app, "instance_mask.wgsl");
        app.register_type::<RaytracingMesh3d>();
        // Each per-instance GPU column is its own `GpuColumnPlugin` (parallel
        // prepare + scatter). They schedule into `GpuColumnPrepareSet`, ordered
        // after material resolution below.
        app.add_plugins(GpuInstancesPlugin);

        // Main world: observer-driven instance-change set, replacing the per-frame
        // `Or<(Added, Changed, Changed)>` query the extract used to scan over every
        // RT-mesh entity. Observers fire only on real bind / material / cull-layer
        // events, so steady state (movement only) flags nothing; the render-world
        // flush drains the set.
        app.init_resource::<RtInstanceChanges>()
            .add_observer(mark_instance_added)
            .add_observer(mark_instance_material_changed)
            .add_observer(mark_instance_layers_changed);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<RtSlotMap>()
            .init_resource::<MaterialTraversalFlags>()
            .add_systems(
                RenderStartup,
                (init_instance_manager, init_material_slots, init_rt_journal),
            )
            // Serial flush: drain the change set, resolve slot, bind / update the
            // `InstanceManager`. Gated until the column scatter pipelines have
            // compiled — a bind whose delta can't be scattered yet would leave the
            // column zero on a static scene (see `cluster_columns_ready`). The
            // change set persists across the cold-pipeline frames (drained only
            // when the flush runs), so no bind is missed.
            .add_systems(
                ExtractSchedule,
                (
                    flush_cluster_instances.run_if(cluster_columns_ready),
                    // TEMP slot-ratchet diagnostic — runs after the flush so it
                    // logs post-allocate state. Remove once the decay is fixed.
                    log_slot_ratchet.after(flush_cluster_instances),
                ),
            )
            .add_observer(free_cluster_slot)
            .add_systems(
                Render,
                (
                    prepare_material_slots.in_set(RenderSystems::Prepare),
                    // Slot-aligned alpha-test flags the PTLAS fill derives
                    // instance opacity from (GPU-side).
                    prepare_material_traversal_flags
                        .in_set(RenderSystems::Prepare)
                        .after(prepare_material_slots),
                    // Resolves `material_id` the `MaterialColumn` reads, so it
                    // must precede every column prepare.
                    resolve_instance_material_ids
                        .in_set(RenderSystems::Prepare)
                        .after(prepare_material_slots)
                        .before(GpuColumnPrepareSet),
                    // Upload this frame's instance change journal for the GPU
                    // reconcile. Runs after the flush appended its records (flush is
                    // in ExtractSchedule, before Prepare).
                    upload_rt_journal.in_set(RenderSystems::Prepare),
                ),
            )
            // The delta-clear records in the render graph after the scatters
            // (its ordering vs the AS passes is the `SolariClusterSystems`
            // chain configured by `AccelPlugin`).
            .add_systems(
                RenderGraph,
                clear_instance_deltas.in_set(SolariClusterSystems::Cleanup),
            );
    }
}
