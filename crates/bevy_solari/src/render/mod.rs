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
use bevy_camera::{CameraMainTextureUsages, Hdr};
use bevy_light::cluster::ClusterConfig;
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
    render_resource::TextureUsages,
    ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_ecs::component::Component;
use bevy_ecs::lifecycle::Add;
use bevy_ecs::observer::On;
use bevy_ecs::reflect::ReflectComponent;
use bevy_ecs::system::{Commands, Query};
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
            .add_plugins(ExtractComponentPlugin::<SolariReference>::default())
            .register_type::<SolariReference>()
            .init_resource::<rt_pipeline::SolariFreezeDiff>()
            .add_plugins(ExtractResourcePlugin::<rt_pipeline::SolariFreezeDiff>::default())
            // Solari does its own light sampling and never reads the clustered-forward
            // light clusters, so opt every `SolariCamera` out of the per-view cluster
            // assignment (bevy_light's `assign_objects_to_clusters`) — a free CPU win.
            // `Camera3d` still requires `Clusters`; `ClusterConfig::None` just makes the
            // assign pass clear-and-skip it. Overridable: a view that explicitly sets a
            // `ClusterConfig` (e.g. a hybrid raster view) wins over this default.
            .register_required_components_with::<SolariCamera, ClusterConfig>(
                || ClusterConfig::None,
            )
            .register_type::<SolariViewState>()
            .register_type::<atmosphere::SolariAtmosphere>()
            .register_type::<atmosphere::SolariGlobalFog>();

        // Solari's RT compute pass writes the view's main texture directly, which needs
        // `STORAGE_BINDING`. `Camera` already requires `CameraMainTextureUsages` (without
        // it), so a required-component default on `SolariCamera` would be *ignored* (a
        // required component only fills a slot that isn't already there). Instead, OR the
        // flag in when a camera becomes a `SolariCamera` — order-independent, and it adds
        // to (rather than replaces) the camera's other usages.
        app.add_observer(
            |add: On<Add, SolariCamera>,
             mut usages: Query<&mut CameraMainTextureUsages>,
             mut commands: Commands| {
                if let Ok(mut u) = usages.get_mut(add.entity) {
                    u.0 |= TextureUsages::STORAGE_BINDING;
                } else {
                    commands.entity(add.entity).insert(
                        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
                    );
                }
            },
        );

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
        app.init_resource::<rt_pipeline::SolariNormalFacing>()
            .add_plugins(ExtractResourcePlugin::<rt_pipeline::SolariNormalFacing>::default());
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
            .init_resource::<atmosphere::SolariAtmosphereVolumesGpu>()
            .add_systems(RenderStartup, atmosphere::init_atmosphere_pipeline)
            .add_systems(RenderStartup, rt_pipeline::init_rt_blit)
            .add_systems(RenderStartup, rt_pipeline::init_restir_spatial)
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
                    atmosphere::extract_atmosphere_volumes,
                    rt_pipeline::extract_rt_camera_slot,
                    rt_pipeline::extract_cylindrical_window,
                ),
            )
            .add_systems(
                Render,
                (
                    rt_pipeline::prepare_rt_output,
                    view_cull::prepare_solari_view_uniforms,
                    atmosphere::prepare_atmosphere_sky,
                    atmosphere::prepare_atmosphere_volumes,
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
            // Rung-0 harness ops (freeze snapshot / PFM dump) after the frame's trace.
            .add_systems(
                Render,
                rt_pipeline::rt_freeze_ops.in_set(RenderSystems::Cleanup),
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
                    atmosphere::dispatch_atmosphere_lut_bake
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

/// Reference path-tracer mode: while the camera holds still, every frame's samples
/// are averaged into the output buffer (progressive accumulation) — the ground-truth
/// image every realtime technique is validated against. Any camera move, projection,
/// or viewport change resets the accumulator (exposure doesn't — the buffer holds
/// physical radiance and the blit exposes at read). Assumes a static scene
/// (movers keep re-rendering into the average as ghosting). Debug views and DLSS
/// bypass accumulation — don't combine.
#[derive(Component, Reflect, Clone, ExtractComponent)]
#[reflect(Component, Default, Clone)]
pub struct SolariReference {
    /// Paths traced per pixel per frame (inner raygen loop). Raise to converge faster.
    pub samples_per_frame: u32,
    /// Disable next-event estimation (BSDF-only brute force). Validation lever: NEE
    /// on and off MUST converge to the same image — any difference is a pdf/MIS bug.
    pub nee_off: bool,
    /// RIS candidates per NEE sample (rung 2): M light candidates stream through a
    /// one-slot reservoir, one shadow ray for the winner. 1 = plain NEE (identical
    /// estimator); raise for receiver-aware light selection (clamped to 255).
    pub ris_candidates: u32,
    /// ReSTIR DI (rung 3): persist the primary vertex's reservoir per pixel and
    /// temporally merge last frame's (reprojected + geometry-validated). Emissive
    /// candidates then run at the primary vertex only; bounce vertices fall back
    /// to single-sample emissive NEE and directionals are shaded per light.
    pub restir: bool,
    /// Temporal history cap, ×`ris_candidates` — history counts for at most this
    /// many frames' worth of candidates (uncapped M = frozen shadows/stale lights).
    pub restir_m_cap: f32,
    /// Direct illumination only: terminate every path at the primary vertex
    /// (emissive + one NEE/reservoir estimate, no bounces). The standard ReSTIR
    /// evaluation image — indirect noise otherwise buries the DI variance win.
    pub di_only: bool,
    /// Indirect only (rung 4a): the complement of `di_only` — output `A₀·L_gi`,
    /// the suffix energy past the primary vertex. di_only + gi_only = full image.
    pub gi_only: bool,
    /// ReSTIR GI (rung 4a.1): raygen stores the canonical GI sample
    /// `{x_s, n_s, L_gi, pdf, a0}` per pixel and shades GI from the STORED sample
    /// (buffer round-trip gate — must equal plain PT per-path).
    pub restir_gi: bool,
    /// With `restir_gi`: reshade `f(x_v,ω)·cos·L/pdf` from the surface G-buffer
    /// instead of the stored exact `a0` — the reconnection-shift shading path
    /// temporal/spatial reuse will rely on (gate: accumulated unbiasedness).
    pub gi_recon: bool,
    /// With `restir_gi`: temporally merge last frame's reprojected GI reservoir
    /// (surface depth/normal validated, capped by [`Self::restir_m_cap`]).
    pub gi_temporal: bool,
    /// With `restir_gi`: the spatial pass merges neighbors' GI reservoirs
    /// (reconnection-Jacobian weighted, winner visibility) and owns the GI shade.
    pub gi_spatial: bool,
    /// When false, render fresh frames instead of averaging (estimator levers stay
    /// active). With the rung-0 dump this captures a SINGLE warmed restir frame —
    /// the per-frame variance metric temporal reuse actually improves.
    pub accumulate: bool,
    /// Spatial reuse (rung 3 session 2): a post-trace compute pass merges each
    /// pixel's reservoir with `spatial_taps` disk neighbors and owns the winner's
    /// visibility + shade. Requires `restir`.
    pub spatial: bool,
    /// Neighbor taps per pixel (≤8).
    pub spatial_taps: u32,
    /// Neighbor disk radius, pixels.
    pub spatial_radius: f32,
    /// false = naive M-sum combiner (BIASED — the visible-darkening study);
    /// true = Z-count (only M whose surface could produce the winner).
    pub spatial_unbiased: bool,
    /// Debug: the spatial pass paints which stage killed each pixel
    /// (red = dead reservoir, yellow = zero re-target, blue = occluded, green = lit).
    pub spatial_debug: bool,
}

impl Default for SolariReference {
    fn default() -> Self {
        Self {
            samples_per_frame: 4,
            nee_off: false,
            ris_candidates: 1,
            restir: false,
            restir_m_cap: 20.0,
            di_only: false,
            gi_only: false,
            restir_gi: false,
            gi_recon: false,
            gi_temporal: false,
            gi_spatial: false,
            accumulate: true,
            spatial: false,
            spatial_taps: 5,
            spatial_radius: 20.0,
            spatial_unbiased: false,
            spatial_debug: false,
        }
    }
}
