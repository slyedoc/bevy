//! The realtime full-RT render path: a ray-tracing-pipeline integrator
//! (raygen + per-material SBT closest-hit shaders) over the shared
//! [`RaytracingSceneBindings`](crate::bindings::RaytracingSceneBindings) scene,
//! plus the atmosphere sky bake. Everything is raytraced — no world cache, no
//! deferred prepass. Selected via [`SolariViewState`](view::SolariViewState).

pub mod atmosphere;
pub mod rt_pipeline;
mod reset;
pub mod view;
pub mod view_cull;

use bevy_app::{App, Plugin};
use bevy_camera::Hdr;
use bevy_core_pipeline::{
    schedule::{Core3d, Core3dSystems},
    tonemapping::tonemapping,
};
use bevy_ecs::schedule::{common_conditions::resource_exists, IntoScheduleConfigs, SystemCondition};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    extract_resource::ExtractResourcePlugin,
    ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_ecs::component::Component;
use bevy_ecs::reflect::ReflectComponent;
use bevy_shader::load_shader_library;

use crate::bindings::RaytracingSceneBindings;
use crate::ecs_gpu::SceneColumns;
use crate::pipelines::SolariPipelines;
use crate::render::view::{rt_pipeline_enabled, SolariViewState};
pub use reset::CameraReset;

pub struct SolarRenderPlugin;

impl Plugin for SolarRenderPlugin {
    fn build(&self, app: &mut App) {
        // Shared single-scattering atmosphere physics (sky bake + aerial perspective).
        load_shader_library!(app, "atmosphere.wgsl");

        app.init_resource::<SolariViewState>()
            .add_plugins(ExtractResourcePlugin::<SolariViewState>::default())
            // SolariCamera is the render-world filter (`With<SolariCamera>`) every
            // Solari prepare/render system keys off; it must be extracted or those
            // systems match nothing and nothing renders.
            .add_plugins(ExtractComponentPlugin::<SolariCamera>::default())
            .register_type::<SolariViewState>()
            .register_type::<atmosphere::SolariAtmosphere>()
            .register_type::<atmosphere::SolariGlobalFog>();

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<view_cull::SolariViewUniforms>()
            .init_resource::<atmosphere::SolariAtmosphereGpu>()
            .add_systems(RenderStartup, atmosphere::init_atmosphere_pipeline)
            .add_systems(RenderStartup, rt_pipeline::init_rt_blit)
            .add_systems(
                ExtractSchedule,
                (
                    (
                        reset::clear_camera_reset,
                        reset::reset_render_on_view_state_change,
                        reset::reset_render_on_request,
                    )
                        .chain(),
                    view_cull::extract_solari_view_cull_masks,
                    view_cull::extract_solari_skybox,
                    atmosphere::extract_solari_atmosphere,
                ),
            )
            .add_systems(
                Render,
                (
                    rt_pipeline::prepare_rt_output,
                    view_cull::prepare_solari_view_uniforms,
                    atmosphere::prepare_atmosphere_sky,
                )
                    .in_set(RenderSystems::PrepareResources),
            )
            // .add_systems(
            //     Render,
            //     // Zero the camera jitter — the RT path writes a fresh frame and
            //     // has no temporal accumulator, so a jittered projection would just
            //     // shimmer.
            //     jitter::zero_solari_jitter.in_set(RenderSystems::PrepareViews),
            // )
            .add_systems(
                Render,
                atmosphere::prepare_atmosphere_bind_group.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Core3d,
                atmosphere::dispatch_atmosphere_bake
                    .after(Core3dSystems::MainPass)
                    .before(rt_pipeline::rt_pipeline)
                    .run_if(resource_exists::<SolariPipelines>),
            )
            .add_systems(
                Core3d,
                rt_pipeline::rt_pipeline
                    .after(Core3dSystems::MainPass)
                    .before(tonemapping)
                    // No `resource_exists::<RtPipeline>` gate — the system
                    // lazily builds it on the first ready frame.
                    .run_if(
                        rt_pipeline_enabled
                            .and_then(resource_exists::<rt_pipeline::RtBlit>)
                            .and_then(resource_exists::<RaytracingSceneBindings>)
                            .and_then(resource_exists::<SceneColumns>),
                    ),
            );
    }
}

#[derive(Component, Default, Reflect, Clone, ExtractComponent)]
#[reflect(Component, Default, Clone)]
#[require(Hdr, CameraReset)]
pub struct SolariCamera;
