//! Centralized pipelines for `bevy_solari` — one [`SolariPipelines`] resource
//! holding every pass's compiled-pipeline id, built once at `RenderStartup`, so a
//! dispatch reads its id from **one** place instead of N scattered per-resource
//! accessors. The companion to [`SolariResourceManager`](crate::resource_manager)
//! (the layouts): together they are `bevy_pbr`'s meshlet split — `MeshletPipelines`
//! (ids) + `ResourceManager` (layouts).
//!
//! Field names mirror [`SolariResourceManager`] 1:1, so a pass's id
//! (`pipelines.gizmo_depth`) and its layout (`resource_manager.gizmo_depth`) share
//! a key: [`init_solari_pipelines`] reads each layout from the manager to queue the
//! matching pipeline.
//!
//! The generic per-column **scatter** pipeline is the one exception: it's created
//! by the generic [`GpuColumnPlugin`](crate::ecs_gpu) per column type, so it stays
//! with that infrastructure rather than being enumerated here. The shading
//! integrator is the raw-VK ray-tracing pipeline (`gpu::rt_pipeline`), built on
//! its own — not a compute pipeline, so not enumerated here either. The
//! acceleration-structure passes (selector / BLAS sharing / PTLAS fill) are
//! layout-free Slang heap kernels owned by their pass resources (`accel/*`).

use bevy_app::App;
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer};
use bevy_core_pipeline::FullscreenShader;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::render_resource::CachedRenderPipelineId;

use crate::resource_manager::SolariResourceManager;

/// Every solari wgpu-pipeline id, queued once at `RenderStartup`. Fields
/// are `pub`: a dispatch reads its id directly and resolves it through
/// `PipelineCache`. Absent only on devices lacking the cluster/sparse support
/// solari needs (the manager gates on the allocator).
#[derive(Resource)]
pub struct SolariPipelines {
    /// Fullscreen depth-write bridging RT primary-hit depth into the hardware depth
    /// buffer so gizmos occlude against the ray-traced scene (`render/gizmo_depth.wgsl`).
    pub gizmo_depth: CachedRenderPipelineId,
}

/// Register every solari wgpu shader as an embedded asset. Called from
/// [`crate::SolariPlugin::build`] so the `embedded_asset!` here and the
/// `load_embedded_asset!` in [`init_solari_pipelines`] are co-located — the
/// embedded path matches by construction, no cross-module path drift.
pub fn embed_solari_shaders(app: &mut App) {
    embedded_asset!(app, "render/gizmo_depth.wgsl");
}

/// `RenderStartup`, after [`SolariResourceManager`] is built: queue every pass's
/// pipeline against the layout the manager owns, and store the ids. Gated on the
/// manager (absent on unsupported devices), so [`SolariPipelines`] — and every
/// `run_if`-gated dispatch — never exists there.
pub fn init_solari_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<bevy_render::render_resource::PipelineCache>,
    asset_server: Res<AssetServer>,
    // The single source of every pass layout. `Option` — absent on devices without
    // the cluster/sparse support solari needs (the manager gates on the allocator).
    resource_manager: Option<Res<SolariResourceManager>>,
    // Fullscreen vertex for the gizmo-depth bridge raster pass.
    fullscreen_shader: Res<FullscreenShader>,
    mut registry: ResMut<crate::ecs_gpu::SolariPipelineRegistry>,
) {
    let Some(resource_manager) = resource_manager else {
        return;
    };

    let gizmo_depth = crate::render::gizmo_depth::gizmo_depth_pipeline(
        &pipeline_cache,
        &fullscreen_shader,
        load_embedded_asset!(asset_server.as_ref(), "render/gizmo_depth.wgsl"),
        resource_manager.gizmo_depth.clone(),
    );
    registry.register_render("gizmo_depth", gizmo_depth);

    commands.insert_resource(SolariPipelines { gizmo_depth });
}
