//! The realtime full-RT render path: a ray-tracing-pipeline integrator
//! (raygen + per-material SBT closest-hit shaders) over the shared
//! [`RaytracingSceneBindings`](crate::bindings::RaytracingSceneBindings) scene,
//! plus the atmosphere sky bake. Everything is raytraced — no world cache, no
//! deferred prepass. Selected via [`SolariViewState`](view::SolariViewState).

pub mod atmosphere;
#[cfg(feature = "dlss")]
pub mod dlss;
pub mod gizmo_depth;
pub mod rt_pipeline;
mod reset;
pub mod view;
pub mod view_cull;

use bevy_app::{App, Plugin, Update};
use bevy_camera::Hdr;
use bevy_core_pipeline::{
    core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
    schedule::{Core3d, Core3dSystems},
};
// Only the DLSS resolve/render systems order against tonemapping now that the RT
// compose runs inside `MainPass`; gate the import so the non-DLSS build is clean.
#[cfg(feature = "dlss")]
use bevy_core_pipeline::tonemapping::tonemapping;
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
pub use reset::{CameraReframe, CameraReset};

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

        // DLSS Ray Reconstruction quality mode (global, extracted). The SDK is
        // created in `SolariPlugin::finish` (`dlss::init_dlss`); the per-view context,
        // guide buffers, and RR dispatch land in later phases.
        #[cfg(feature = "dlss")]
        app.init_resource::<dlss::SolariDlssMode>()
            .add_plugins(ExtractResourcePlugin::<dlss::SolariDlssMode>::default());
        app.init_resource::<rt_pipeline::SolariCostHeatmap>()
            .add_plugins(ExtractResourcePlugin::<rt_pipeline::SolariCostHeatmap>::default());
        app.init_resource::<rt_pipeline::SolariAnyHitHeatmap>()
            .add_plugins(ExtractResourcePlugin::<rt_pipeline::SolariAnyHitHeatmap>::default());
        app.init_resource::<rt_pipeline::SolariClusterView>()
            .add_plugins(ExtractResourcePlugin::<rt_pipeline::SolariClusterView>::default());
        app.init_resource::<rt_pipeline::SolariTriangleView>()
            .add_plugins(ExtractResourcePlugin::<rt_pipeline::SolariTriangleView>::default());
        app.init_resource::<rt_pipeline::SolariShowDisplacement>()
            .add_plugins(ExtractResourcePlugin::<rt_pipeline::SolariShowDisplacement>::default());
        // In-situ tessellation: collect every displacement-mapped instance (main
        // world) + extract its mesh/transform/texture for the render-world
        // GPU tessellation path to tessellate in place.
        app.init_resource::<crate::geometry::tess_displace::TessShowcaseInstances>()
            .add_plugins(
                ExtractResourcePlugin::<crate::geometry::tess_displace::TessShowcaseInstances>::default(),
            )
            .add_systems(
                Update,
                (
                    crate::geometry::tess_displace::find_tess_showcase_instances,
                    crate::geometry::tess_displace::hide_tessellated_base_instances,
                ),
            );

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            // Empty; surfaces register into it (built-ins via `SolariPlugin`).
            .init_resource::<crate::gpu::rt_pipeline::SolariHitGroupRegistry>()
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
                    (reset::clear_camera_reframe, reset::extract_camera_reframe).chain(),
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
            // Compose-FIRST: bake the sky, then trace + blit the RT image into the
            // view target BEFORE the raster main pass, so the rasterized opaque +
            // transparent phases (gizmos, debug overlays) draw ON TOP of the
            // ray-traced scene instead of being clobbered by a blit that runs after
            // them. (The camera must not clear — `ClearColorConfig::None` — or the
            // opaque pass would wipe the composed image; the RT pass already covers
            // every pixel via the sky/miss shader.) Mirrors solari-pt's `compose`
            // ordering; `gizmo_depth` (Stage 2) then bridges RT depth between the
            // opaque and transparent phases so overlays occlude correctly.
            .add_systems(
                Core3d,
                (
                    atmosphere::dispatch_atmosphere_bake
                        .run_if(resource_exists::<SolariPipelines>),
                    rt_pipeline::rt_pipeline
                        // No `resource_exists::<RtPipeline>` gate — the system
                        // lazily builds it on the first ready frame.
                        .run_if(
                            rt_pipeline_enabled
                                .and_then(resource_exists::<rt_pipeline::RtBlit>)
                                .and_then(resource_exists::<RaytracingSceneBindings>)
                                .and_then(resource_exists::<SceneColumns>),
                        ),
                )
                    .chain()
                    .before(main_opaque_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            )
            // Bridge the RT primary-hit depth (packed in the output buffer's `.w`)
            // into the hardware depth buffer AFTER the opaque pass clears it and
            // BEFORE the transparent pass draws gizmos, so overlays occlude against
            // the ray-traced scene. Runs per `SolariCamera` view (its `ViewQuery`
            // only matches once the view's `RtOutputBuffer` exists).
            .add_systems(
                Core3d,
                gizmo_depth::solari_gizmo_depth
                    .after(main_opaque_pass_3d)
                    .before(main_transparent_pass_3d)
                    .run_if(resource_exists::<SolariPipelines>),
            );

        // DLSS Ray Reconstruction: resolve the trace's guide buffers into textures,
        // then denoise/upscale the view target. Gated on the SDK existing (RR
        // supported); per-view component presence (set by `prepare_solari_dlss`) gates
        // the actual mode. Both run after the trace + blit, before tonemapping.
        #[cfg(feature = "dlss")]
        render_app
            .add_systems(RenderStartup, dlss::init_solari_dlss)
            .add_systems(
                Render,
                dlss::prepare_solari_dlss.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                (
                    dlss::solari_dlss_resolve
                        .after(rt_pipeline::rt_pipeline)
                        .before(tonemapping)
                        .run_if(resource_exists::<dlss::SolariDlssSdk>.and_then(dlss::dlss_enabled)),
                    dlss::solari_dlss_render
                        .after(dlss::solari_dlss_resolve)
                        .before(tonemapping)
                        .run_if(resource_exists::<dlss::SolariDlssSdk>.and_then(dlss::dlss_enabled)),
                )
                    .chain(),
            );
    }
}

#[derive(Component, Default, Reflect, Clone, ExtractComponent)]
#[reflect(Component, Default, Clone)]
#[require(Hdr, CameraReset, CameraReframe)]
pub struct SolariCamera;
