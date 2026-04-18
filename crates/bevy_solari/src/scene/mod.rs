mod binder;
pub mod blas;
mod extract;
mod rt_pipeline;
mod types;

use bevy_shader::load_shader_library;
pub use binder::RaytracingSceneBindings;
pub use rt_pipeline::ShadowRtPipeline;
pub use types::RaytracingMesh3d;

use crate::SolariPlugins;
use bevy_app::{App, Plugin};
use bevy_ecs::{schedule::IntoScheduleConfigs, system::Commands};
use bevy_render::{
    extract_resource::ExtractResourcePlugin,
    mesh::{
        allocator::{allocate_and_free_meshes, MeshAllocatorSettings},
        RenderMesh,
    },
    render_asset::prepare_assets,
    render_resource::{BufferUsages, PipelineCache},
    renderer::RenderDevice,
    ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_ecs::system::Res;
use binder::prepare_raytracing_scene_bindings;
use blas::{compact_raytracing_blas, prepare_raytracing_blas, BlasManager};
use extract::{extract_raytracing_scene, StandardMaterialAssets};
use tracing::{info, warn};

/// Creates acceleration structures and binding arrays of resources for raytracing.
pub struct RaytracingScenePlugin;

impl Plugin for RaytracingScenePlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "brdf.wgsl");
        load_shader_library!(app, "raytracing_scene_bindings.wgsl");
        load_shader_library!(app, "sampling.wgsl");
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        let render_device = render_app.world().resource::<RenderDevice>();
        let features = render_device.features();
        if !features.contains(SolariPlugins::required_wgpu_features()) {
            warn!(
                "RaytracingScenePlugin not loaded. GPU lacks support for required features: {:?}.",
                SolariPlugins::required_wgpu_features().difference(features)
            );
            return;
        }

        app.add_plugins(ExtractResourcePlugin::<StandardMaterialAssets>::default());

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .world_mut()
            .resource_mut::<MeshAllocatorSettings>()
            .extra_buffer_usages |= BufferUsages::BLAS_INPUT | BufferUsages::STORAGE;

        render_app
            .init_gpu_resource::<BlasManager>()
            .init_gpu_resource::<StandardMaterialAssets>()
            .insert_resource(RaytracingSceneBindings::new())
            .add_systems(RenderStartup, init_shadow_rt_pipeline)
            .add_systems(ExtractSchedule, extract_raytracing_scene)
            .add_systems(
                Render,
                (
                    prepare_raytracing_blas
                        .in_set(RenderSystems::PrepareAssets)
                        .before(prepare_assets::<RenderMesh>)
                        .after(allocate_and_free_meshes),
                    compact_raytracing_blas
                        .in_set(RenderSystems::PrepareAssets)
                        .after(prepare_raytracing_blas),
                    prepare_raytracing_scene_bindings.in_set(RenderSystems::PrepareBindGroups),
                ),
            );
    }
}

fn init_shadow_rt_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    scene_bindings: Res<RaytracingSceneBindings>,
) {
    let shader_source = include_str!("shadow_rt.wgsl");
    let pipeline = ShadowRtPipeline::new(
        &render_device,
        &pipeline_cache,
        &scene_bindings,
        shader_source,
    );
    info!("Shadow RT pipeline created successfully");
    commands.insert_resource(pipeline);
}
