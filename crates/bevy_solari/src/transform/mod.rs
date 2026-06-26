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

use bevy_app::{App, Plugin, PostUpdate, Update};
use bevy_ecs::{
    prelude::{Query, ResMut, With},
    schedule::{common_conditions::resource_exists, IntoScheduleConfigs},
};
use bevy_math::Vec3;
use bevy_render::{
    extract_resource::ExtractResourcePlugin, renderer::RenderGraph, Render, RenderApp,
    RenderStartup, RenderSystems,
};
use bevy_transform::components::Transform;
use bevy_transform::systems::{propagate_transforms_for, sync_simple_transforms};
use bevy_ui::Node;

use crate::ecs_gpu::{GpuColumnPrepareSet, GpuPresenceColumnPlugin};
use crate::pipelines::SolariPipelines;
use crate::render::{CameraReset, SolariCamera};
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
    extract_transform_graph, CellColumn, LocalColumn, NodeEntityColumn, ParentColumn,
    SolariFloatingOrigin, SolariGridCell, StaticColumn, TransformGraph, TransformStatic,
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

/// Keep the floating origin glued to the [`SolariCamera`]. When the camera's local
/// `Transform` drifts past half a cell, shift [`SolariFloatingOrigin::origin_cell`] by
/// the whole cells crossed and wrap the transform back toward the cell centre — the
/// camera's world position is unchanged, just re-expressed relative to the new origin
/// cell, keeping its rendered coordinates small (large coords are what make the 1-spp
/// path tracer jitter and pixelate).
///
/// The origin jump re-worlds every instance for one frame (the propagate's
/// `needs_full_rebuild` latch fires on the origin change), so we pulse [`CameraReset`]
/// — a one-frame DLSS + ReSTIR history reset so reprojection doesn't smear across the
/// discontinuity. No-op when no floating origin is configured (`cell_edge == 0`), so
/// scenes without a floating origin pay only an early-returning query.
pub fn recenter_floating_origin(
    mut origin: ResMut<SolariFloatingOrigin>,
    mut camera: Query<(&mut Transform, &mut CameraReset), With<SolariCamera>>,
) {
    let edge = origin.cell_edge;
    if edge <= 0.0 {
        return;
    }
    let Ok((mut transform, mut reset)) = camera.single_mut() else {
        return;
    };
    // Whole cells the camera has drifted from its cell centre (round → nearest cell, so
    // the local stays within ±½ cell; handles multi-cell jumps from a fast camera too).
    let drift = (transform.translation / edge).round();
    if drift == Vec3::ZERO {
        return;
    }
    origin.origin_cell[0] += drift.x as i32;
    origin.origin_cell[1] += drift.y as i32;
    origin.origin_cell[2] += drift.z as i32;
    transform.translation -= drift * edge;
    reset.0 = true;
}

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
        // The TransformStatic presence flag (node-slot indexed) the PTLAS fill
        // reads to choose an instance's partition. Observer-fed, zero per-frame cost.
        .add_plugins(GpuPresenceColumnPlugin::<StaticColumn>::default())
        // Solari's native floating origin (the GPU-table reimplementation of big_space's
        // grid; credited). Default (origin 0, edge 0) = no offset, so non-floating-origin
        // scenes are unaffected. Mirrored to the render world for the propagate pass.
        .init_resource::<SolariFloatingOrigin>()
        .add_plugins(ExtractResourcePlugin::<SolariFloatingOrigin>::default())
        // Camera-follow recenter: keeps the origin cell glued to the camera so its
        // rendered coords stay small. Runs in Update (after camera controllers move it),
        // before the PostUpdate transform sync + the render extract pick up the change.
        .add_systems(Update, recenter_floating_origin)
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


