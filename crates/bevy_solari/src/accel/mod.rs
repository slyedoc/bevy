//! Acceleration-structure domain — the per-frame GPU pipeline that turns
//! the cluster scene into a ray-traceable TLAS.
//!
//! Four staged passes, run in order each frame:
//! 1. [`blas_sharing`] — classify instances to LOD bands, elect one
//!    provider BLAS per `(geometry, band)` bucket, assign per-instance
//!    BLAS addresses.
//! 2. [`selector`] — per-bucket object-space DAG cut → CLAS ref lists +
//!    per-bucket BLAS-build args.
//! 3. [`blas_rebuild`] — raw-VK indirect cluster-BLAS build (GPU-driven
//!    count) into the shared per-geometry BLAS pool.
//! 4. [`ptlas`] — incremental partitioned-TLAS fill + build.
//!
//! Each pass owns one resource (buffers + pipelines + bind group); the
//! `RaytracingScenePlugin` in [`crate::scene`] schedules them.

use bevy_app::{App, Plugin};
use bevy_core_pipeline::schedule::camera_driver;
use bevy_ecs::schedule::{common_conditions::resource_exists, IntoScheduleConfigs};
use bevy_render::{
    renderer::{RenderGraph, RenderGraphSystems},
    Render, RenderApp, RenderStartup, RenderSystems,
};

use crate::pipelines::SolariPipelines;
use crate::{SolariClusterSystems, SolariSetup};

pub mod blas_rebuild;
pub mod blas_sharing;
pub mod partition_alloc;
pub mod pipelines;
pub mod ptlas;
pub mod selector;

pub use blas_rebuild::{dispatch_blas_rebuild, init_blas_rebuild};
pub use pipelines::{
    blas_sharing_bind_group_layout, ptlas_bind_group_layout, selector_bind_group_layout,
};
pub use blas_sharing::{
    dispatch_blas_sharing, init_blas_sharing, prepare_blas_sharing, prepare_blas_sharing_bind_group,
};
pub use partition_alloc::{block_of, PartitionAllocator, BLOCK_SHIFT};
pub use ptlas::{dispatch_ptlas, init_ptlas, prepare_ptlas_fill_bind_group, prepare_ptlas_params};
pub use selector::{
    dispatch_selector, init_selector, prepare_selector_bind_group, prepare_selector_params,
    ClusterSelectorSettings,
};



/// Acceleration-structure domain plugin. Owns the AS-build pipeline
/// (classify → select → build BLAS → build TLAS) + its compute pipelines,
/// and configures the [`SolariClusterSystems`] ordering the whole cluster
pub struct AccelPlugin;

impl Plugin for AccelPlugin {
    fn build(&self, app: &mut App) {
        // All AS-pass shaders (selector/blas_sharing/ptlas_fill) are embedded
        // centrally in `crate::pipelines`, co-located with their queue.

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ClusterSelectorSettings>()
            // The cluster AS pipeline is scene-global (one shared TLAS),
            // so it records in the top-level `RenderGraph` schedule before
            // `camera_driver` runs any camera's pathtracer — built once,
            // submitted with the rest of the frame's graph work.
            .configure_sets(
                RenderGraph,
                (
                    SolariClusterSystems::Scatter,
                    SolariClusterSystems::Propagate,
                    SolariClusterSystems::Classify,
                    SolariClusterSystems::Select,
                    SolariClusterSystems::BuildBlas,
                    SolariClusterSystems::BuildTlas,
                    SolariClusterSystems::Cleanup,
                )
                    .chain()
                    .in_set(RenderGraphSystems::Render)
                    .before(camera_driver),
            )
            .add_systems(
                RenderStartup,
                (
                    init_selector.after(SolariSetup),
                    init_blas_sharing.after(SolariSetup),
                    init_blas_rebuild.after(SolariSetup),
                    init_ptlas.after(SolariSetup),
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_blas_sharing.in_set(RenderSystems::Prepare),
                    prepare_selector_params
                        .in_set(RenderSystems::Prepare)
                        .after(prepare_blas_sharing),
                    prepare_ptlas_params
                        .in_set(RenderSystems::Prepare)
                        .after(prepare_selector_params),
                    prepare_selector_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    prepare_blas_sharing_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    prepare_ptlas_fill_bind_group.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                RenderGraph,
                (
                    dispatch_blas_sharing
                        .run_if(resource_exists::<SolariPipelines>)
                        .in_set(SolariClusterSystems::Classify),
                    dispatch_selector
                        .run_if(resource_exists::<SolariPipelines>)
                        .in_set(SolariClusterSystems::Select),
                    dispatch_blas_rebuild.in_set(SolariClusterSystems::BuildBlas),
                    dispatch_ptlas
                        .run_if(resource_exists::<SolariPipelines>)
                        .in_set(SolariClusterSystems::BuildTlas),
                ),
            );
    }
}
