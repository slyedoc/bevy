//! Experimental full-RT ReSTIR path tracer.
//!
//! A self-contained realtime path: its own primary-visibility pass writes a
//! lean PT G-buffer, light-tile presampling feeds a unified-path-reservoir
//! ReSTIR loop (initial+temporal → spatial+shade) over the shared
//! [`RaytracingSceneBindings`](crate::bindings::RaytracingSceneBindings)
//! scene, and a compose pass writes the final color. No world cache, no
//! deferred prepass — everything is raytraced.
//!
//! Modeled on [`crate::realtime`]'s multi-pass single-compute-pass dispatch
//! and [`crate::pathtracer`]'s extract/prepare cleanliness. Intended to
//! eventually replace [`crate::realtime`].
//!
//! Not added by [`SolariPlugins`](crate::SolariPlugins) — add [`RestirPlugin`]
//! manually.


pub mod pathtracer;
pub mod atmosphere;
mod node;
mod gizmo_depth;
mod overlay;
mod prepare;
mod reset;
pub mod view;
pub mod view_cull;
#[cfg(feature = "dlss")]
mod dlss;

use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_camera::Hdr;
use bevy_core_pipeline::{
    core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
    schedule::{Core3d, Core3dSystems}, tonemapping::tonemapping,
};
use bevy_ecs::{
    component::Component,
    reflect::ReflectComponent,
    schedule::{IntoScheduleConfigs, SystemCondition, common_conditions::{not, resource_exists}},
};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems, camera::TemporalJitter, extract_component::{ExtractComponent, ExtractComponentPlugin}
};
use bevy_shader::load_shader_library;
use crate::{pipelines::SolariPipelines, resource_manager::SolariResourceManager, render::view::{SolariOverlay, overlay_is}, render::reset::CameraReset};
use overlay::register_overlays;
use node::{prepare_restir_jitter, restir};
pub use gizmo_depth::{gizmo_depth_bind_group_layout, gizmo_depth_pipeline};
pub use node::restir_bind_group_layout;
#[cfg(feature = "dlss")]
pub use dlss::{restir_dlss_resolve_bind_group_layout, restir_dlss_resolve_pipeline};
use prepare::prepare_restir_resources;

pub struct SolarRenderPlugin;

impl Plugin for SolarRenderPlugin {
    fn build(&self, app: &mut App) {
        // Shared group(1) bindings + reservoir struct, imported by every pass.
        load_shader_library!(app, "restir_bindings.wgsl");
        // Shared single-scattering atmosphere physics (sky bake + aerial perspective).
        load_shader_library!(app, "atmosphere.wgsl");

        embedded_asset!(app, "debug_overlay.wgsl");
        embedded_asset!(app, "gizmo_depth.wgsl");
        // The restir passes (visibility/presample/restir_pt/compose), the pathtracer,
        // and the atmosphere bake are embedded centrally in `crate::pipelines`,
        // co-located with their pipeline queue.

        #[cfg(feature = "dlss")]
        embedded_asset!(app, "dlss_resolve.wgsl");

        app
        .add_plugins(ExtractComponentPlugin::<SolariOverlay>::default())
        // SolariCamera is the render-world filter (`With<SolariCamera>`) every
        // Solari prepare/render system keys off; it must be extracted or those
        // systems match nothing and nothing renders.
        .add_plugins(ExtractComponentPlugin::<SolariCamera>::default())
        .register_type::<SolariOverlay>()
        .register_type::<atmosphere::SolariAtmosphere>();
           

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<view_cull::SolariViewUniforms>()
            .init_resource::<atmosphere::SolariAtmosphereGpu>()
            .add_systems(RenderStartup, atmosphere::init_atmosphere_pipeline)
            .add_systems(
                ExtractSchedule,
                (
                    (reset::clear_camera_reset, reset::reset_render_on_camera_move).chain(),
                    view_cull::extract_solari_view_cull_masks,
                    view_cull::extract_solari_skybox,
                    atmosphere::extract_solari_atmosphere,
                ),
            )
            .add_systems(
                Render,
                (
                    pathtracer::prepare_pathtracer_accumulation_texture,
                    view_cull::prepare_solari_view_uniforms,
                    atmosphere::prepare_atmosphere_sky,
                )
                    .in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Render,
                atmosphere::prepare_atmosphere_bind_group
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Core3d,
                atmosphere::dispatch_atmosphere_bake
                    .after(Core3dSystems::MainPass)
                    .before(pathtracer::pathtracer)
                    .run_if(resource_exists::<SolariPipelines>),
            )
            .add_systems(
                Core3d,
                pathtracer::pathtracer
                    .after(Core3dSystems::MainPass)
                    .before(tonemapping)
                    .run_if(
                        overlay_is(SolariOverlay::Pathtrace)
                            .and_then(resource_exists::<SolariResourceManager>)
                            .and_then(resource_exists::<SolariPipelines>),
                    ),
            );

    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .add_systems(
                Render,
                (
                    prepare_restir_jitter.in_set(RenderSystems::PrepareViews),
                    prepare_restir_resources.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_systems(
                Core3d,
                restir
                    .before(main_opaque_pass_3d)
                    .in_set(Core3dSystems::MainPass)
                    // Pathtrace views are rendered by the reference path tracer.
                    .run_if(
                        not(overlay_is(SolariOverlay::Pathtrace))
                            .and_then(resource_exists::<SolariResourceManager>)
                            .and_then(resource_exists::<SolariPipelines>),
                    ),
            )
            .add_systems(
                Core3d,
                // Write RT primary-hit depth into the hardware depth buffer
                // between the opaque clear and the gizmo (Transparent3d) pass, so
                // rasterized overlays occlude against the ray-traced scene.
                gizmo_depth::solari_gizmo_depth
                    .after(main_opaque_pass_3d)
                    .before(main_transparent_pass_3d)
                    .run_if(
                        not(overlay_is(SolariOverlay::Pathtrace))
                            .and_then(resource_exists::<SolariResourceManager>)
                            .and_then(resource_exists::<SolariPipelines>),
                    ),
            );

        // One run-condition-gated overlay system per debug view, in
        // `EarlyPostProcess` (after compose, before DLSS, so the chosen
        // visualization is upscaled along with the rest).
        register_overlays(render_app);

        // DLSS Ray Reconstruction (direct dlss_wgpu, prepass-free). Creates the
        // SDK if supported, then prepares the per-view context + guide buffers +
        // resolution override each frame. The guide resolve pass + render node
        // are added in later increments.
        #[cfg(feature = "dlss")]
        if dlss::init_dlss(app) {
            let render_app = app.sub_app_mut(RenderApp);
            // Guide-buffer overlays only make sense (and only have textures)
            // when DLSS is active.
            overlay::register_dlss_overlays(render_app);
            render_app
                .add_systems(
                    Render,
                    dlss::prepare_restir_dlss
                        .in_set(RenderSystems::PrepareViews)
                        .after(prepare_restir_jitter),
                )
                .add_systems(
                    Core3d,
                    (dlss::restir_dlss_resolve, dlss::restir_dlss)
                        .chain()
                        // After every overlay (so the viz is upscaled too), and
                        // skipped entirely for the full-res pathtrace view.
                        .after(overlay::SolariDebugOverlay)
                        .in_set(Core3dSystems::EarlyPostProcess)
                        // `restir_dlss_resolve` reads the central pipeline + layout.
                        .run_if(
                            not(overlay_is(SolariOverlay::Pathtrace))
                                .and_then(resource_exists::<SolariResourceManager>)
                                .and_then(resource_exists::<SolariPipelines>),
                        ),
                );
        }
    }
}

#[derive(Component, Default, Reflect, Clone, ExtractComponent)]
#[reflect(Component, Default, Clone)]
#[require(Hdr, TemporalJitter, SolariOverlay, CameraReset)]
pub struct SolariCamera;

