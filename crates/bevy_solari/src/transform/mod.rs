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
use bevy_ecs::{
    prelude::With,
    schedule::{common_conditions::resource_exists, IntoScheduleConfigs},
};
use bevy_render::{renderer::RenderGraph, Render, RenderApp, RenderStartup, RenderSystems};
use bevy_transform::systems::{propagate_transforms_for, sync_simple_transforms};
use bevy_ui::Node;

use crate::ecs_gpu::GpuColumnPrepareSet;
use crate::pipelines::SolariPipelines;
use crate::{SolariClusterSystems, SolariSetup};

mod gather;
mod graph;
mod propagate;
mod readback;

pub use gather::{
    dispatch_transform_gather, init_transform_gather, prepare_transform_gather,
    prepare_transform_gather_bind_group, transform_gather_bind_group_layout, TransformGather,
};
pub use graph::{
    extract_transform_graph, LocalColumn, ParentColumn, TransformGraph, TransformStatic,
    TransformTablePlugin, ROOT_PARENT,
};
pub use propagate::{
    dispatch_transform_propagate, init_transform_propagate, prepare_transform_propagate,
    prepare_transform_propagate_bind_groups, transform_propagate_bind_group_layout,
    TransformPropagate,
};
pub use readback::{
    init_transform_readback, transform_readback_bind_group_layout, NoGpuGlobalTransformReadback,
    TransformReadback,
};
use readback::{
    build_readback_main, dispatch_transform_readback, prepare_transform_readback,
    prepare_transform_readback_bind_group,
};

/// The transform-table plugin: the macro-generated table (`local`/`parent`
/// columns + component-indexed slots + extract + clear) plus the GPU
/// ancestor-walk world-propagation pass and the gather that drives the RT
/// scene's instance transforms. See the module docs.
pub struct SolariTransformPlugin;

impl Plugin for SolariTransformPlugin {
    fn build(&self, app: &mut App) {

        // The transform pass shaders are embedded centrally in `crate::pipelines`,
        // co-located with their `SolariPipelines` builds.
        // Columns, slot index, extract, and Cleanup clear — all generated.
        app.add_plugins(TransformTablePlugin)
        // TODO: handle few few transforms locally, not sending to gpu, might not need anymore
        .add_systems(
            PostUpdate,
            (sync_simple_transforms, propagate_transforms_for::<With<Node>>),
        );
        // Main-app side of the GlobalTransform readback (the output buffer +
        // the `Readback` entity / decode observer).
        build_readback_main(app);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(
                RenderStartup,
                (
                    init_transform_propagate,
                    init_transform_gather,
                    init_transform_readback,
                )
                    .after(SolariSetup),
            )
            .add_systems(
                Render,
                (
                    (
                        prepare_transform_propagate,
                        prepare_transform_gather,
                        prepare_transform_readback,
                    )
                        .chain()
                        .in_set(RenderSystems::Prepare)
                        .after(GpuColumnPrepareSet),
                    (
                        prepare_transform_propagate_bind_groups,
                        prepare_transform_gather_bind_group,
                        prepare_transform_readback_bind_group,
                    )
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                RenderGraph,
                // Propagate (ancestor-walk) → gather into the instance
                // TransformColumn → readback gather for marked entities.
                (
                    dispatch_transform_propagate,
                    dispatch_transform_gather,
                    dispatch_transform_readback,
                )
                    .chain()
                    .run_if(resource_exists::<SolariPipelines>)
                    .in_set(SolariClusterSystems::Propagate),
            );
    }
}
