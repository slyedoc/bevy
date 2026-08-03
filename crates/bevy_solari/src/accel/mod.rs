//! Acceleration-structure domain — the per-frame GPU pipeline that turns
//! the cluster scene into a ray-traceable TLAS.
//!
//! Four staged passes, run in order each frame:
//! 1. [`blas_sharing`] — classify instances to LOD bands, elect dirty
//!    geometries for rebuild, assign per-instance BLAS addresses.
//! 2. [`selector`] — per-bucket object-space DAG cut → CLAS ref lists +
//!    per-bucket BLAS-build args.
//! 3. [`blas_rebuild`] — raw-VK indirect cluster-BLAS build (GPU-driven
//!    count) into the shared per-geometry BLAS pool.
//! 4. [`ptlas`] — incremental partitioned-TLAS fill + build.
//!
//! Each pass owns one resource (buffers + lazily-built heap kernels); the
//! `RaytracingScenePlugin` in [`crate::scene`] schedules them.

use bevy_app::{App, Plugin};
use bevy_core_pipeline::schedule::camera_driver;
use bevy_ecs::schedule::{common_conditions::resource_exists, IntoScheduleConfigs};
use bevy_render::{
    renderer::{RenderGraph, RenderGraphSystems},
    ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};

use crate::pipelines::SolariPipelines;
use crate::{SolariClusterSystems, SolariSetup};

pub mod animated_blas;
pub mod blas_rebuild;
pub mod blas_sharing;
pub mod deform;
pub mod partition_alloc;
pub mod ptlas;
pub mod selector;

pub use blas_rebuild::{dispatch_blas_rebuild, init_blas_rebuild};
pub use blas_sharing::{dispatch_blas_sharing, init_blas_sharing, prepare_blas_sharing};
pub use partition_alloc::{block_of, PartitionAllocator, BLOCK_SHIFT};
pub use ptlas::{dispatch_ptlas, init_ptlas, prepare_ptlas_params};
pub use selector::{
    dispatch_selector, init_selector, prepare_selector_params, ClusterSelectorSettings,
};



/// Acceleration-structure domain plugin. Owns the AS-build pipeline
/// (classify → select → build BLAS → build TLAS) + its compute pipelines,
/// and configures the [`SolariClusterSystems`] ordering the whole cluster
/// pipeline runs under.
pub struct AccelPlugin;

impl Plugin for AccelPlugin {
    fn build(&self, app: &mut App) {
        // The AS-pass shaders are embedded in `crate::pipelines`, co-located
        // with the code that queues them.

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
                    SolariClusterSystems::Deform,
                    SolariClusterSystems::Classify,
                    SolariClusterSystems::Select,
                    SolariClusterSystems::BuildBlas,
                    SolariClusterSystems::BuildAnimatedBlas,
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
                    deform::init_deform.after(SolariSetup),
                    animated_blas::init_animated_blas.after(SolariSetup),
                ),
            )
            .add_systems(ExtractSchedule, deform::extract_animated_skins)
            .add_systems(
                Render,
                (
                    prepare_blas_sharing.in_set(RenderSystems::PrepareResources),
                    prepare_selector_params
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_blas_sharing),
                    prepare_ptlas_params
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_selector_params)
                        // The tess slot reservation reads `TessClassify::blas_ready`, which
                        // `run_tess_classify` latches — order after it so the reservation
                        // and `prepare_tess_ptlas_write`'s record count never disagree on
                        // the transition frame (mismatch → out-of-bounds PTLAS write).
                        .after(crate::geometry::tess_classify::run_tess_classify),
                    deform::prepare_deform.in_set(RenderSystems::PrepareResources),
                    animated_blas::prepare_animated_blas.in_set(RenderSystems::PrepareResources),
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
                    deform::dispatch_deform
                        .run_if(resource_exists::<SolariPipelines>)
                        .in_set(SolariClusterSystems::Deform),
                    dispatch_blas_rebuild
                        .run_if(resource_exists::<SolariPipelines>)
                        .in_set(SolariClusterSystems::BuildBlas),
                    animated_blas::dispatch_animated_blas
                        .run_if(resource_exists::<SolariPipelines>)
                        .in_set(SolariClusterSystems::BuildAnimatedBlas),
                    dispatch_ptlas
                        .run_if(resource_exists::<SolariPipelines>)
                        .in_set(SolariClusterSystems::BuildTlas),
                ),
            );
    }
}
