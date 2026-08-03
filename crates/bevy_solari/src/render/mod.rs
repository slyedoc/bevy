//! The realtime full-RT render path: a ray-tracing-pipeline integrator
//! (raygen + per-material SBT closest-hit shaders) over the shared
//! [`RaytracingSceneBindings`](crate::bindings::RaytracingSceneBindings) scene,
//! plus the atmosphere sky bake. Everything is raytraced — no world cache, no
//! deferred prepass. Cameras opt in (and pick their integrator) via [`SolariCamera`].

pub mod atmosphere;
pub mod dlss;
pub mod gizmo_depth;
pub mod rt_pipeline;
mod reset;
pub mod sky;
pub mod view_cull;

use bevy_app::{App, Plugin, Update};
use bevy_camera::{CameraMainTextureUsages, Exposure, Hdr};
use bevy_light::cluster::ClusterConfig;
use bevy_core_pipeline::{
    core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
    schedule::{Core3d, Core3dSystems},
};
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

use crate::bindings::RaytracingSceneBindings;
use crate::ecs_gpu::SceneColumns;
use crate::pipelines::SolariPipelines;
pub use reset::{CameraReframe, CameraReset};

pub struct SolarRenderPlugin;

impl Plugin for SolarRenderPlugin {
    fn build(&self, app: &mut App) {
        // SolariCamera is the render-world filter (`With<SolariCamera>`) every
        // Solari prepare/render system keys off; it must be extracted or those
        // systems match nothing and nothing renders. Its value carries the
        // integrator mode + settings the rt_pipeline dispatch reads per view.
        app.add_plugins(ExtractComponentPlugin::<SolariCamera>::default())
            .register_type::<SolariCamera>()
            .register_type::<SolariLighting>()
            .register_type::<SolariReference>()
            .register_type::<SolariRestir>()
            .register_type::<GiArm>()
            .register_type::<DiEstimator>()
            .register_type::<DiNee>()
            .register_type::<DiRestir>()
            .register_type::<GiEstimator>()
            .register_type::<GiRestir>()
            .register_type::<SpatialReuse>()
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
            .register_type::<sky::SolariSky>();

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
        // created in `SolariPlugin::finish` (`dlss::init_dlss`).
        app.init_resource::<dlss::SolariDlssMode>()
            .add_plugins(ExtractResourcePlugin::<dlss::SolariDlssMode>::default());
        app.init_resource::<crate::nrc::SolariNrc>()
            .add_plugins(ExtractResourcePlugin::<crate::nrc::SolariNrc>::default());
        // The debug view rides inside `SolariCamera` (its `debug` field).
        app.register_type::<rt_pipeline::SolariDebugView>();
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

        // The `.slang` sources as embedded assets: with bevy_asset's
        // `embedded_watcher` feature (the app's opt-in, as for bevy's own
        // shaders), a saved edit fires an asset event and the RT stages hot
        // reload; without it they're static embedded bytes.
        {
            use crate::gpu::slang_sources::{SlangSource, SlangSourceHandles, SlangSourceLoader};
            use bevy_asset::{embedded_asset, load_embedded_asset, AssetApp};
            app.init_asset::<SlangSource>()
                .register_asset_loader(SlangSourceLoader);
            // Both bevy macros take plain path literals, so the crate-relative
            // path is spelled alongside its file-name key.
            macro_rules! watched {
                ($file:literal, $path:literal) => {{
                    embedded_asset!(app, $path);
                    ($file, load_embedded_asset!(app.world(), $path))
                }};
            }
            let handles = SlangSourceHandles(vec![
                watched!("raygen.slang", "rt_pipeline/raygen.slang"),
                watched!("miss.slang", "rt_pipeline/miss.slang"),
                watched!("miss_shadow.slang", "rt_pipeline/miss_shadow.slang"),
                watched!("ahit_alpha.slang", "rt_pipeline/ahit_alpha.slang"),
                watched!("chit_opaque.slang", "rt_pipeline/chit_opaque.slang"),
                watched!("chit_glass.slang", "rt_pipeline/chit_glass.slang"),
                watched!("chit_hair.slang", "rt_pipeline/chit_hair.slang"),
                watched!("chit_portal.slang", "rt_pipeline/chit_portal.slang"),
                watched!("rt_payload.slang", "rt_pipeline/rt_payload.slang"),
                watched!("scene_resolve.slang", "rt_pipeline/scene_resolve.slang"),
                watched!("brdf.slang", "rt_pipeline/brdf.slang"),
                watched!("sampling.slang", "rt_pipeline/sampling.slang"),
                watched!("hair.slang", "rt_pipeline/hair.slang"),
            ]);
            app.insert_resource(handles);
        }

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            // Empty; surfaces register into it (built-ins via `SolariPlugin`).
            .init_resource::<crate::gpu::rt_pipeline::SolariHitGroupRegistry>()
            .init_resource::<sky::SolariCustomSky>()
            // The live shader-source registry: RT stages hot reload on edit.
            .init_resource::<crate::gpu::slang_sources::SlangSources>()
            .add_systems(
                ExtractSchedule,
                crate::gpu::slang_sources::extract_slang_sources,
            )
            // After `SolariSetup`: the blit + rt_camera are heap-flagged raw
            // compute, built through the seam the allocator init inserts.
            .add_systems(
                RenderStartup,
                (rt_pipeline::init_rt_blit, rt_pipeline::init_rt_camera)
                    .after(crate::SolariSetup),
            )
            // After `SolariSetup`: the spatial kernel is heap-flagged raw
            // compute, built through the seam the allocator init inserts.
            .add_systems(
                RenderStartup,
                rt_pipeline::init_restir_spatial.after(crate::SolariSetup),
            )
            // After `SolariSetup`: the pipelines are heap-flagged raw compute,
            // built through the seam the allocator init inserts.
            .add_systems(
                RenderStartup,
                crate::nrc::init_nrc_pipelines.after(crate::SolariSetup),
            )
            .add_systems(
                Render,
                crate::nrc::init_nrc_buffers.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Render,
                crate::nrc::log_nrc_loss.in_set(RenderSystems::Cleanup),
            )
            .add_systems(
                ExtractSchedule,
                (
                    (
                        reset::clear_camera_reset,
                        reset::reset_render_on_mode_change,
                        reset::reset_render_on_request,
                    )
                        .chain(),
                    (reset::clear_camera_reframe, reset::extract_camera_reframe).chain(),
                    view_cull::extract_solari_sky,
                    sky::extract_solari_custom_sky,
                    rt_pipeline::extract_rt_camera_slot,
                    rt_pipeline::extract_cylindrical_window,
                ),
            )
            .add_systems(
                Render,
                rt_pipeline::prepare_rt_output.in_set(RenderSystems::PrepareResources),
            )
            // Freeze-diff harness ops (freeze snapshot / EXR dump) after the frame's trace.
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
            // every pixel via the sky/miss shader.) `gizmo_depth` then bridges RT
            // depth between the opaque and transparent phases so overlays occlude
            // correctly.
            .add_systems(
                Core3d,
                rt_pipeline::rt_pipeline
                    // No `resource_exists::<RtPipeline>` gate — the system
                    // lazily builds it on the first ready frame.
                    .run_if(
                        resource_exists::<rt_pipeline::RtBlit>
                            .and_then(resource_exists::<RaytracingSceneBindings>)
                            .and_then(resource_exists::<SceneColumns>),
                    )
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
        //
        // Also BEFORE the raster main pass, for the same compose-first reason the blit
        // is: `solari_dlss_render` does a `post_process_write` over the whole view
        // target, so it must land before the opaque/transparent phases draw overlays.
        // `.before(tonemapping)` alone leaves it unordered against those phases — they
        // conflict on `ViewTarget`, so the executor serializes them in an arbitrary
        // order that can flip between frames, and a frame where DLSS ran last lost its
        // gizmos (the navmesh overlay flickered).
        render_app
            // After `SolariSetup`: the resolve is a heap-flagged raw compute
            // kernel, built through the seam the allocator init inserts.
            .add_systems(
                RenderStartup,
                dlss::init_solari_dlss.after(crate::SolariSetup),
            )
            .add_systems(
                Render,
                dlss::prepare_solari_dlss.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                (
                    dlss::solari_dlss_resolve
                        .after(rt_pipeline::rt_pipeline)
                        .before(main_opaque_pass_3d)
                        .before(tonemapping)
                        .run_if(resource_exists::<dlss::SolariDlssSdk>.and_then(dlss::dlss_enabled)),
                    dlss::solari_dlss_render
                        .after(dlss::solari_dlss_resolve)
                        .before(main_opaque_pass_3d)
                        .before(tonemapping)
                        .run_if(resource_exists::<dlss::SolariDlssSdk>.and_then(dlss::dlss_enabled)),
                )
                    .chain(),
            );
    }
}

/// Opts a camera into solari's ray-traced rendering: the integrator [`mode`]
/// plus the active debug-view paint. The default mode is the production
/// realtime stack; the reference integrator is the ground-truth harness
/// every realtime technique is validated against.
///
/// Changing [`mode`] (the variant, or the active variant's levers) drops
/// temporal history via [`CameraReset`]; changing [`debug`] deliberately does
/// NOT — a debug paint pauses reference accumulation and it resumes untouched.
///
/// [`Self::name`] derives a compact truthful slug from the configuration —
/// tools identify runs by it.
///
/// [`mode`]: Self::mode
/// [`debug`]: Self::debug
#[cfg_attr(feature = "bevy_solari_debug", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bevy_solari_debug", serde(default))]
#[derive(Component, Reflect, Clone, ExtractComponent)]
#[reflect(Component, Default, Clone)]
#[require(Hdr, Exposure, CameraReset, CameraReframe)]
pub struct SolariCamera {
    /// How this camera integrates light.
    pub mode: SolariLighting,
    /// Replace the shaded image with an instrument paint
    /// ([`None`](rt_pipeline::SolariDebugView::None) = normal rendering).
    /// Composes with any mode — the cost/any-hit/NRC paints measure the
    /// active integrator's trace.
    pub debug: rt_pipeline::SolariDebugView,
}

impl Default for SolariCamera {
    fn default() -> Self {
        Self {
            mode: SolariLighting::default(),
            debug: rt_pipeline::SolariDebugView::None,
        }
    }
}

impl SolariCamera {
    /// The derived name: a compact slug computed FROM the configuration, so
    /// it always tells the truth about what renders — grader rows, window
    /// titles, and report labels all use it. Session knobs (samples per
    /// frame, accumulation) and debug paints are not identity and don't
    /// appear.
    ///
    /// `rt` = the shipped realtime default (deviations append: `rt-nogi`,
    /// `rt-m5`); `ref` = the plain reference accumulator; estimator arms
    /// append with their levers: `ref-di4m20s3-gis3` is ReSTIR DI (RIS 4,
    /// m-cap 20, spatial 3 taps) + ReSTIR GI with a 3-tap spatial pass.
    pub fn name(&self) -> String {
        fn num(v: f32) -> String {
            if v.fract() == 0.0 {
                format!("{}", v as i64)
            } else {
                format!("{v}")
            }
        }
        fn sp(spatial: &Option<SpatialReuse>) -> String {
            spatial.as_ref().map_or(String::new(), |sp| format!("s{}", sp.taps))
        }
        match &self.mode {
            SolariLighting::Realtime(rt) => {
                // Deviations from the shipped default name themselves, so
                // bare `rt` always means exactly `SolariRestir::default()`.
                let d = SolariRestir::default();
                let mut n = String::from("rt");
                if rt.ris_candidates != d.ris_candidates {
                    n.push_str(&format!("-ris{}", rt.ris_candidates));
                }
                if rt.m_cap != d.m_cap {
                    n.push_str(&format!("-m{}", num(rt.m_cap)));
                }
                if let Some(spatial) = &rt.spatial {
                    n.push_str(&format!("-s{}", spatial.taps));
                }
                if rt.gi != d.gi {
                    n.push_str("-nogi");
                }
                if rt.nrc_gi != d.nrc_gi {
                    n.push_str("-nonrc");
                }
                if rt.bounces != d.bounces {
                    n.push_str(&format!("-b{}", rt.bounces));
                }
                if rt.firefly_clamp != d.firefly_clamp {
                    n.push_str(&format!("-clamp{}", num(rt.firefly_clamp)));
                }
                n
            }
            SolariLighting::Reference(r) => {
                let mut n = String::from("ref");
                match &r.di {
                    DiEstimator::BsdfOnly => n.push_str("-bsdf"),
                    // The default NEE arm is what bare "ref" means; deviations
                    // (including plain NEE-1, the identity A/B) name themselves.
                    DiEstimator::Nee(nee) if *nee == DiNee::default() => {}
                    DiEstimator::Nee(nee) => {
                        n.push_str(&format!("-nee{}", nee.ris_candidates));
                    }
                    DiEstimator::Restir(di) => {
                        n.push_str(&format!(
                            "-di{}m{}{}",
                            di.ris_candidates,
                            num(di.m_cap),
                            sp(&di.spatial)
                        ));
                    }
                }
                match &r.gi {
                    None => n.push_str("-nogi"),
                    Some(arm) => {
                        if arm.only {
                            n.push_str("-gionly");
                        }
                        if let GiEstimator::Restir(gi) = &arm.estimator {
                            n.push_str("-gi");
                            if !gi.recon {
                                n.push_str("norecon");
                            }
                            if !gi.temporal {
                                n.push_str("notemporal");
                            }
                            n.push_str(&sp(&gi.spatial));
                        }
                        if arm.nrc {
                            n.push_str("-nrc");
                        }
                        if arm.bounces != GiArm::default().bounces {
                            n.push_str(&format!("-b{}", arm.bounces));
                        }
                    }
                }
                if !r.jitter {
                    n.push_str("-nojit");
                }
                n
            }
        }
    }

    /// The reference settings, if this camera runs the reference integrator.
    pub fn reference(&self) -> Option<&SolariReference> {
        match &self.mode {
            SolariLighting::Reference(reference) => Some(reference),
            SolariLighting::Realtime(_) => None,
        }
    }

    /// The realtime ReSTIR settings, if this camera runs the production integrator.
    pub fn restir(&self) -> Option<&SolariRestir> {
        match &self.mode {
            SolariLighting::Realtime(restir) => Some(restir),
            SolariLighting::Reference(_) => None,
        }
    }
}

/// How a [`SolariCamera`] integrates light.
#[cfg_attr(feature = "bevy_solari_debug", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub enum SolariLighting {
    /// Production per-frame stack: ReSTIR DI + GI reconnection reservoirs with
    /// NRC termination, feeding DLSS Ray Reconstruction as the denoiser.
    Realtime(SolariRestir),
    /// Ground-truth progressive accumulation with per-estimator levers.
    Reference(SolariReference),
}

impl Default for SolariLighting {
    fn default() -> Self {
        Self::Realtime(SolariRestir::default())
    }
}

/// Reference path-tracer mode: while the camera holds still, every frame's samples
/// are averaged into the output buffer (progressive accumulation) — the ground-truth
/// image every realtime technique is validated against. Any camera move, projection,
/// or viewport change resets the accumulator (exposure doesn't — the buffer holds
/// physical radiance and the blit exposes at read). Assumes a static scene
/// (movers keep re-rendering into the average as ghosting). Debug views and DLSS
/// bypass accumulation — don't combine.
///
/// The estimator is a tree, so what composes is visible in the types: spatial
/// reuse only exists inside a ReSTIR arm, reconnection shading only inside the
/// GI ReSTIR arm. Selected via [`SolariLighting::Reference`].
#[cfg_attr(feature = "bevy_solari_debug", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bevy_solari_debug", serde(default))]
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub struct SolariReference {
    /// Paths traced per pixel per frame (inner raygen loop). Raise to converge faster.
    #[reflect(@1.0..=16.0f32)]
    pub samples_per_frame: u32,
    /// When false, render fresh frames instead of averaging (estimator levers stay
    /// active). With the freeze/EXR dump this captures a SINGLE warmed restir frame —
    /// the per-frame variance temporal reuse actually improves.
    pub accumulate: bool,
    /// The direct-illumination estimator at each path vertex.
    pub di: DiEstimator,
    /// The indirect (suffix) transport past the primary vertex. `None`
    /// terminates every path at the primary vertex — the standard ReSTIR
    /// DI evaluation image (direct variance not buried in GI noise). The
    /// GI-only display and NRC termination are properties OF the arm, so
    /// they can't be configured without transport to act on.
    pub gi: Option<GiArm>,
    /// Sub-pixel AA jitter (uniform pixel-area). Off = every sample through
    /// the pixel CENTER: the converged image is aliased, but temporal
    /// reprojection lands on the exact same surface points every frame —
    /// the isolation lever for jitter-induced target-function mismatch in
    /// reservoir merges.
    pub jitter: bool,
}

impl Default for SolariReference {
    fn default() -> Self {
        Self {
            samples_per_frame: 1,
            accumulate: false,
            di: DiEstimator::Nee(DiNee::default()),
            gi: Some(GiArm::default()),
            jitter: true,
        }
    }
}

impl SolariReference {
    /// ReSTIR DI temporal reuse is on.
    pub fn di_restir(&self) -> bool {
        matches!(self.di, DiEstimator::Restir(_))
    }

    /// ReSTIR GI reconnection reservoirs are on.
    pub fn gi_restir(&self) -> bool {
        matches!(&self.gi, Some(arm) if matches!(arm.estimator, GiEstimator::Restir(_)))
    }

    /// NRC GI termination is on (an arm property — no arm, no termination).
    pub fn nrc_gi(&self) -> bool {
        self.gi.as_ref().is_some_and(|arm| arm.nrc)
    }

    /// Maximum indirect bounces (the GI arm's; without an arm paths already
    /// terminate at the primary vertex, so the cap is moot).
    pub fn bounces(&self) -> u32 {
        self.gi.as_ref().map_or_else(|| GiArm::default().bounces, |arm| arm.bounces)
    }

    /// The DI spatial-reuse settings, when the ReSTIR DI spatial pass is on.
    pub fn di_spatial(&self) -> Option<&SpatialReuse> {
        match &self.di {
            DiEstimator::Restir(di) => di.spatial.as_ref(),
            _ => None,
        }
    }

    /// The GI spatial-reuse settings, when the ReSTIR GI spatial pass is on.
    pub fn gi_spatial(&self) -> Option<&SpatialReuse> {
        match self.gi.as_ref().map(|arm| &arm.estimator) {
            Some(GiEstimator::Restir(gi)) => gi.spatial.as_ref(),
            _ => None,
        }
    }

    /// RIS candidates per NEE sample (1 for the BSDF-only estimator).
    pub fn ris_candidates(&self) -> u32 {
        match &self.di {
            DiEstimator::BsdfOnly => 1,
            DiEstimator::Nee(DiNee { ris_candidates })
            | DiEstimator::Restir(DiRestir { ris_candidates, .. }) => (*ris_candidates).max(1),
        }
    }

    /// The temporal history cap: ReSTIR DI's, or [`DiRestir`]'s default when
    /// the DI arm has no reservoir (the GI reservoirs still cap by it).
    pub fn m_cap(&self) -> f32 {
        match &self.di {
            DiEstimator::Restir(DiRestir { m_cap, .. }) => *m_cap,
            _ => DiRestir::default().m_cap,
        }
    }

    /// The GI dead-canonical-draw instrument paint is on.
    pub fn gi_dead_view(&self) -> bool {
        matches!(
            self.gi.as_ref().map(|arm| &arm.estimator),
            Some(GiEstimator::Restir(gi)) if gi.dead_view
        )
    }

    /// Any spatial pass's kill-stage debug paint is on.
    pub fn spatial_debug_paint(&self) -> bool {
        self.di_spatial().is_some_and(|sp| sp.debug_paint)
            || self.gi_spatial().is_some_and(|sp| sp.debug_paint)
    }
}

/// The indirect (suffix) transport arm of a [`SolariReference`].
#[cfg_attr(feature = "bevy_solari_debug", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bevy_solari_debug", serde(default))]
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub struct GiArm {
    /// How the suffix is estimated.
    pub estimator: GiEstimator,
    /// Display only this arm's bucket (`A₀·L_gi`, the suffix energy past the
    /// primary vertex). The DI estimator still runs — it sets the MIS weights
    /// this bucket is measured under. di-terminated + gi-only images sum to
    /// the full image.
    pub only: bool,
    /// Terminate this arm's paths at bounce 2 into the neural radiance cache
    /// (estimator flag bit 20). Biased by cache error.
    pub nrc: bool,
    /// Maximum indirect bounces (path segments past the primary hit).
    /// Default 1, matching [`SolariRestir::bounces`] — each deeper traced
    /// bounce adds little energy at spike variance. Truncation is biased by
    /// the missing tail: converged ground-truth renders that want full
    /// transport spell `bounces: 32` (Russian roulette then does the terminating).
    #[reflect(@1.0..=32.0f32)]
    pub bounces: u32,
}

impl Default for GiArm {
    fn default() -> Self {
        Self {
            estimator: GiEstimator::default(),
            only: false,
            nrc: false,
            bounces: 1,
        }
    }
}

/// The direct-illumination estimator at each path vertex.
#[cfg_attr(feature = "bevy_solari_debug", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Clone, PartialEq)]
pub enum DiEstimator {
    /// BSDF-only brute force (no next-event estimation). Validation lever: NEE
    /// on and off MUST converge to the same image — any difference is a pdf/MIS bug.
    BsdfOnly,
    /// Next-event estimation with RIS: M light candidates stream through a
    /// one-slot reservoir, one shadow ray for the winner.
    Nee(DiNee),
    /// ReSTIR DI: persist the primary vertex's reservoir per pixel and
    /// temporally merge last frame's (reprojected + geometry-validated). Emissive
    /// candidates then run at the primary vertex only; bounce vertices fall back
    /// to single-sample emissive NEE and directionals are shaded per light.
    Restir(DiRestir),
}

/// [`DiEstimator::Nee`] levers. The payload is a struct so switching to the
/// variant (UI, RON) constructs the canonical defaults, not zeroes.
#[cfg_attr(feature = "bevy_solari_debug", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bevy_solari_debug", serde(default))]
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub struct DiNee {
    /// RIS candidates per NEE sample. 1 = plain NEE (identical estimator);
    /// raise for receiver-aware light selection (clamped to 255).
    #[reflect(@1.0..=255.0f32)]
    pub ris_candidates: u32,
}

impl Default for DiNee {
    fn default() -> Self {
        Self { ris_candidates: 4 }
    }
}

/// [`DiEstimator::Restir`] levers.
#[cfg_attr(feature = "bevy_solari_debug", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bevy_solari_debug", serde(default))]
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub struct DiRestir {
    /// Initial RIS candidates per pixel streamed through the reservoir.
    #[reflect(@1.0..=255.0f32)]
    pub ris_candidates: u32,
    /// Temporal history cap, ×`ris_candidates` — history counts for at most
    /// this many frames' worth of candidates (uncapped M = frozen shadows).
    /// Default 1 — see [`SolariRestir::m_cap`]. History never aids accumulated
    /// convergence either: correlation slows it, and under AA jitter the
    /// merge's target-function mismatch biases it.
    #[reflect(@1.0..=64.0f32)]
    pub m_cap: f32,
    /// Spatial reuse: a post-trace compute pass merges each pixel's
    /// reservoir with disk neighbors and owns the winner's visibility + shade.
    pub spatial: Option<SpatialReuse>,
}

impl Default for DiRestir {
    fn default() -> Self {
        Self {
            ris_candidates: 4,
            m_cap: 1.0,
            spatial: None,
        }
    }
}

/// The indirect (suffix) estimator past the primary vertex.
#[cfg_attr(feature = "bevy_solari_debug", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Clone, PartialEq, Debug, Default)]
#[reflect(Default, Clone, PartialEq)]
pub enum GiEstimator {
    /// Plain path tracing of the suffix.
    #[default]
    PathTraced,
    /// ReSTIR GI: raygen stores the canonical GI sample
    /// `{x_s, n_s, L_gi, pdf, a0}` per pixel and shades GI from the STORED
    /// sample.
    Restir(GiRestir),
}

/// [`GiEstimator::Restir`] levers.
#[cfg_attr(feature = "bevy_solari_debug", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bevy_solari_debug", serde(default))]
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub struct GiRestir {
    /// Reshade `f(x_v,ω)·cos·L/pdf` from the surface G-buffer instead of
    /// the stored exact `a0` — the reconnection-shift shading path
    /// temporal/spatial reuse relies on (gate: accumulated unbiasedness).
    pub recon: bool,
    /// Temporally merge last frame's reprojected GI reservoir (surface
    /// depth/normal validated, capped by the DI arm's m_cap).
    pub temporal: bool,
    /// The spatial pass merges neighbors' GI reservoirs
    /// (reconnection-Jacobian weighted, winner visibility) and owns the
    /// GI shade.
    pub spatial: Option<SpatialReuse>,
    /// Instrument: paint 1 where the canonical GI draw is dead (bounce-1
    /// miss or delta pdf) — the accumulated mean IS the dead-draw rate.
    pub dead_view: bool,
}

impl Default for GiRestir {
    fn default() -> Self {
        Self {
            recon: true,
            temporal: true,
            spatial: None,
            dead_view: false,
        }
    }
}

/// Spatial reservoir reuse settings, shared by the DI and GI ReSTIR arms.
#[cfg_attr(feature = "bevy_solari_debug", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bevy_solari_debug", serde(default))]
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub struct SpatialReuse {
    /// Neighbor taps per pixel (≤8).
    #[reflect(@1.0..=8.0f32)]
    pub taps: u32,
    /// Neighbor disk radius, pixels.
    #[reflect(@1.0..=128.0f32)]
    pub radius: f32,
    /// false = naive M-sum combiner (BIASED — visibly darkens);
    /// true = Z-count (only M whose surface could produce the winner). Costs
    /// one shadow ray per contributor.
    pub unbiased_zcount: bool,
    /// Debug: the spatial pass paints which stage killed each pixel
    /// (red = dead reservoir, yellow = zero re-target, blue = occluded, green = lit).
    pub debug_paint: bool,
}

impl Default for SpatialReuse {
    fn default() -> Self {
        Self {
            taps: 3,
            radius: 20.0,
            unbiased_zcount: false,
            debug_paint: false,
        }
    }
}

/// Production ReSTIR on a [`SolariCamera`]: per-frame (no accumulation) DI
/// reservoirs with temporal + optional spatial reuse, and GI reconnection
/// reservoirs, feeding DLSS Ray Reconstruction as the denoiser.
///
/// Selected via [`SolariLighting::Realtime`] (the default).
#[cfg_attr(feature = "bevy_solari_debug", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bevy_solari_debug", serde(default))]
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub struct SolariRestir {
    /// Initial light candidates per pixel streamed through the DI reservoir.
    #[reflect(@1.0..=255.0f32)]
    pub ris_candidates: u32,
    /// Temporal history cap, ×`ris_candidates` — history counts for at most
    /// this many frames' worth of candidates. Default 1: DLSS-RR does the
    /// temporal accumulation downstream, and reservoir history it can't see
    /// is temporally-sticky error it preserves as detail — one frame's worth
    /// keeps the merge while handing RR temporally-white input.
    #[reflect(@1.0..=64.0f32)]
    pub m_cap: f32,
    /// Spatial reuse. DEFAULT None — under DLSS RR, spatial reuse's disk-sized
    /// winner patches read as swimming pool-caustic light (correlated noise
    /// poses as illumination structure); RR does the variance reduction
    /// instead. The spatial path stays available for non-RR consumers.
    pub spatial: Option<SpatialReuse>,
    /// GI reconnection reservoirs (temporal + spatial). Off = ReSTIR DI only.
    pub gi: bool,
    /// Max GI luminance per frame, in DISPLAY-referred units (post-exposure,
    /// where ~1.0 is a well-exposed white) — reservoir spikes above it are
    /// scaled down luminance-preserving. 0 = off. The reference path never
    /// clamps (policy); this is the realtime firefly filter.
    #[reflect(@0.0..=100.0f32)]
    pub firefly_clamp: f32,
    /// Terminate GI paths into the neural radiance cache (estimator flag
    /// bit 20) once the cache is mature (`NrcBuffers::step > 300`) — until
    /// then paths trace their full suffix. Biased by cache error. With
    /// [`gi`](Self::gi), cache-terminated paths query the cache inline in
    /// raygen so the GI reservoirs store the full suffix energy.
    pub nrc_gi: bool,
    /// Maximum indirect bounces. Default 1: each traced bounce past the first
    /// adds little energy at enormous variance — blowout noise the denoiser
    /// smears — while with [`nrc_gi`](Self::nrc_gi) the capped vertex
    /// terminates into the cache, which carries the deep tail (training paths
    /// are exempt from the cap, so the cache learns full transport).
    #[reflect(@1.0..=32.0f32)]
    pub bounces: u32,
}

impl Default for SolariRestir {
    fn default() -> Self {
        Self {
            ris_candidates: 4,
            m_cap: 1.0,
            spatial: None,
            gi: true,
            firefly_clamp: 10.0,
            nrc_gi: true,
            bounces: 1,
        }
    }
}
