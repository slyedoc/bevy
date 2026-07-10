//! The transform table — Bevy's transform hierarchy mirrored to the GPU and
//! propagated there. See `crates/bevy_solari/gpu_transform_propagation.md`.
//!
//! [`graph`] declares the table with [`gpu_table!`](crate::gpu_table) (resource,
//! columns, component-indexed slots, and `TransformTablePlugin` — all generated)
//! plus the hand-written extract. [`propagate`] is the GPU ancestor-walk that
//! reads the `local`/`parent` columns and writes a world transform per node
//! (changed nodes only). [`gather`] then copies each instance's world transform
//! into the instance `TransformColumn` the RT path reads — so GPU propagation
//! drives the rendered scene (movement is fully GPU-side; the CPU only mirrors
//! the per-frame local/parent deltas).

use bevy_app::{App, Plugin, PostUpdate};
use bevy_ecs::name::{HashedStr, Name};
use bevy_ecs::prelude::With;
use bevy_ecs::schedule::{common_conditions::resource_exists, IntoScheduleConfigs};
use bevy_math::{Quat, Vec3};
use bevy_render::{
    renderer::RenderGraph, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_transform::components::{GlobalTransform, Transform};
use bevy_transform::systems::{propagate_transforms_for, sync_simple_transforms};
use bevy_ui::Node;

use crate::ecs_gpu::{GpuColumnPrepareSet, GpuPresenceColumnPlugin};
use crate::pipelines::SolariPipelines;
use crate::{SolariClusterSystems, SolariSetup};

mod frontier;
mod gather;
mod graph;
mod propagate;
mod readback;
mod subtract;

pub use frontier::{
    dispatch_transform_frontier, init_transform_frontier, prepare_transform_frontier,
    prepare_transform_frontier_bind_group, transform_frontier_bind_group_layout,
    TransformFrontier,
};
pub use gather::{
    dispatch_transform_gather, init_transform_gather, prepare_transform_gather,
    prepare_transform_gather_bind_group, transform_gather_bind_group_layout, TransformGather,
};
pub use graph::{
    clear_static_first_sight, enqueue_node_first_sight, enqueue_static_first_sight,
    extract_transform_graph, FirstChildColumn, GpuFrameSeeds, LocalRSColumn,
    LocalTranslationColumn, NextSiblingColumn, NodeEntityColumn, ParentColumn, SolariGpuFrame,
    StaticColumn, StaticFirstSightQueue, TransformGraph, TransformStatic, TransformTablePlugin,
    NO_NODE, ROOT_PARENT,
};
pub use propagate::{
    dispatch_transform_propagate, init_transform_propagate, prepare_transform_propagate,
    prepare_transform_propagate_bind_groups, transform_propagate_bind_group_layout,
    TransformPropagate,
};
use readback::{
    build_readback_main, dispatch_transform_readback, prepare_transform_readback,
    prepare_transform_readback_bind_group,
};
pub use readback::{
    init_transform_readback, transform_readback_bind_group_layout, NoGpuGlobalTransformReadback,
    TransformReadback,
};
pub use subtract::{
    dispatch_transform_subtract, extract_origin_slot, init_transform_subtract,
    prepare_transform_subtract, prepare_transform_subtract_bind_group,
    transform_subtract_bind_group_layout, SolariOriginSlot, TransformSubtract,
};

/// The transform-table plugin: the macro-generated table (`local`/`parent`
/// columns + component-indexed slots + extract + clear) plus the GPU
/// ancestor-walk world-propagation pass and the gather that drives the RT
/// scene's instance transforms. See the module docs.
pub struct SolariTransformPlugin;

impl Plugin for SolariTransformPlugin {
    fn build(&self, app: &mut App) {
        // Solari owns the transform system on the full-RT path (`TransformPlugin` is disabled), so
        // it registers the transform components + the glam types it's responsible for. This also
        // lets dynamic `.bsn` scenes name `Transform { translation: Vec3, rotation: Quat, scale:
        // Vec3 }` without relying on the `reflect_auto_register` feature being enabled.
        app.register_type::<Transform>()
            .register_type::<GlobalTransform>()
            .register_type::<Vec3>()
            .register_type::<Quat>()
            // Let dynamic `.bsn` scenes carry a `Name("...")`. `Name(HashedStr)` — register both,
            // plus a `String -> HashedStr` conversion so the loader builds the field from a string
            // literal (the same `ReflectConvert` path `Handle<T>` uses for asset-path strings).
            .register_type::<Name>()
            .register_type::<HashedStr>()
            .register_type_conversion::<String, HashedStr, _>(|s| Ok(s.into()));

        // The transform pass shaders are embedded centrally in `crate::pipelines`,
        // co-located with their `SolariPipelines` builds.
        // Columns, slot index, extract, and Cleanup clear — all generated.
        app.add_plugins(TransformTablePlugin)
            // The TransformStatic presence flag (node-slot indexed) the PTLAS fill
            // reads to choose an instance's partition. Observer-fed, zero per-frame cost.
            .add_plugins(GpuPresenceColumnPlugin::<StaticColumn>::default())
            // Born-static first-sight queue: makes `TransformStatic` safe to add at spawn
            // (the observer queues it; the extract does the one-time upload, see `graph`).
            .init_resource::<StaticFirstSightQueue>()
            .add_observer(enqueue_static_first_sight)
            // Reliable first-sight for every node: queue it when its slot is assigned, so the `local`
            // upload never depends on the extract catching a cross-world change edge.
            .add_observer(enqueue_node_first_sight)
            // TODO: handle few few transforms locally, not sending to gpu, might not need anymore
            .add_systems(
                PostUpdate,
                (
                    sync_simple_transforms,
                    propagate_transforms_for::<With<Node>>,
                ),
            );
        // Main-app side of the GlobalTransform readback (the output buffer +
        // the `Readback` entity / decode observer).
        build_readback_main(app);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            // The floating origin is the primary camera's own transform-table node. The subtract
            // pass reads `world_abs_t[camera_slot]` as the f64 origin on the GPU — the CPU only
            // carries the slot index, so a camera childed to a player/ship/patch renders at 0.
            .init_resource::<SolariOriginSlot>()
            // The gpu-frame frontier seeds, refreshed by the table extract.
            .init_resource::<GpuFrameSeeds>()
            .add_systems(ExtractSchedule, extract_origin_slot)
            // Drain the born-static first-sight queue the extract just consumed. Same cold-start
            // gate as the extract (so events accumulate until pipelines compile), ordered after it.
            .add_systems(
                ExtractSchedule,
                clear_static_first_sight
                    .after(extract_transform_graph)
                    .run_if(crate::ecs_gpu::solari_pipelines_ready),
            )
            .add_systems(
                RenderStartup,
                (
                    init_transform_frontier,
                    init_transform_propagate,
                    init_transform_subtract,
                    init_transform_gather,
                    init_transform_readback,
                )
                    .after(SolariSetup),
            )
            .add_systems(
                Render,
                (
                    (
                        prepare_transform_frontier,
                        prepare_transform_propagate,
                        prepare_transform_subtract,
                        prepare_transform_gather,
                        prepare_transform_readback,
                    )
                        .chain()
                        .in_set(RenderSystems::PrepareResources)
                        .after(GpuColumnPrepareSet),
                    (
                        prepare_transform_frontier_bind_group,
                        prepare_transform_propagate_bind_groups,
                        prepare_transform_subtract_bind_group,
                        prepare_transform_gather_bind_group,
                        prepare_transform_readback_bind_group,
                    )
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                RenderGraph,
                // Frontier (changed set → +descendants, GPU-expanded) → propagate
                // (ancestor-walk → absolute f64 world) → subtract the camera origin
                // (→ relative f32 world) → gather into the instance TransformColumn
                // → readback gather for marked entities.
                (
                    dispatch_transform_frontier,
                    dispatch_transform_propagate,
                    dispatch_transform_subtract,
                    dispatch_transform_gather,
                    dispatch_transform_readback,
                )
                    .chain()
                    .run_if(resource_exists::<SolariPipelines>)
                    .in_set(SolariClusterSystems::Propagate),
            );
    }
}
