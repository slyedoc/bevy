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
pub use reset::{CameraReframe, CameraReset};

pub struct SolarRenderPlugin;

impl Plugin for SolarRenderPlugin {
    fn build(&self, app: &mut App) {
        // Shared single-scattering atmosphere physics (sky bake + aerial perspective).
        load_shader_library!(app, "atmosphere.wgsl");

        // SolariCamera is the render-world filter (`With<SolariCamera>`) every
        // Solari prepare/render system keys off; it must be extracted or those
        // systems match nothing and nothing renders. Its value carries the
        // integrator mode + settings the rt_pipeline dispatch reads per view.
        app.add_plugins(ExtractComponentPlugin::<SolariCamera>::default())
            .register_type::<SolariCamera>()
            .register_type::<SolariLighting>()
            .register_type::<SolariReference>()
            .register_type::<SolariRestir>()
            .register_type::<ReferenceOutput>()
            .register_type::<DiEstimator>()
            .register_type::<GiEstimator>()
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
            .add_systems(RenderStartup, crate::nrc::init_nrc_pipelines)
            .add_systems(
                Render,
                rt_pipeline::queue_restir_spatial_pipeline.in_set(RenderSystems::PrepareResources),
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
            // Rung-0 harness ops (freeze snapshot / EXR dump) after the frame's trace.
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
                            resource_exists::<rt_pipeline::RtBlit>
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

/// Opts a camera into solari's ray-traced rendering: the integrator [`mode`]
/// plus the active debug-view paint. The default mode is the production
/// realtime stack; the reference integrator is the exam/ground-truth harness
/// every realtime technique is validated against.
///
/// Changing [`mode`] (the variant, or the active variant's levers) drops
/// temporal history via [`CameraReset`]; changing [`debug`] deliberately does
/// NOT — a debug paint pauses reference accumulation and it resumes untouched.
///
/// Common setups are stamped out by [`SolariRecipe`] (see
/// [`SolariCamera::from_recipe`]); tweak the resulting levers freely after.
///
/// [`mode`]: Self::mode
/// [`debug`]: Self::debug
#[derive(Component, Reflect, Clone, ExtractComponent)]
#[reflect(Component, Default, Clone)]
#[require(Hdr, CameraReset, CameraReframe)]
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
    /// A camera preconfigured by `recipe` (normal rendering, no debug paint).
    pub fn from_recipe(recipe: SolariRecipe) -> Self {
        Self {
            mode: recipe.mode(),
            debug: rt_pipeline::SolariDebugView::None,
        }
    }

    /// Stamp `recipe`'s mode onto the camera, preserving the session knobs
    /// (samples per frame, accumulation) when both sides are reference modes —
    /// a recipe names an ESTIMATOR; the work budget and accumulation policy
    /// belong to the session (see [`SolariRecipe::matches`]).
    pub fn apply_recipe(&mut self, recipe: SolariRecipe) {
        let mut mode = recipe.mode();
        if let (SolariLighting::Reference(new), SolariLighting::Reference(old)) =
            (&mut mode, &self.mode)
        {
            new.samples_per_frame = old.samples_per_frame;
            new.accumulate = old.accumulate;
        }
        self.mode = mode;
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
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub enum SolariLighting {
    /// Production per-frame stack: ReSTIR DI + GI reconnection reservoirs with
    /// NRC termination, feeding DLSS Ray Reconstruction as the denoiser.
    Realtime(SolariRestir),
    /// Ground-truth progressive accumulation with the exam levers.
    Reference(SolariReference),
}

impl Default for SolariLighting {
    fn default() -> Self {
        Self::Realtime(SolariRestir::default())
    }
}

/// Named estimator setups — one call stamps out a fully-configured
/// [`SolariLighting`], so A/B sweeps and UIs can iterate [`Self::ALL`] instead
/// of hand-assembling lever combinations. A recipe is a constructor, not
/// stored state: tweak the resulting levers freely after.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "bevy_solari_debug", derive(clap::ValueEnum))]
pub enum SolariRecipe {
    /// Reference NEE accumulator (plain next-event estimation, 1 candidate).
    Reference,
    /// Brute-force BSDF-only accumulator — the unbiasedness A/B against
    /// [`Reference`](Self::Reference) (both MUST converge to the same image).
    Bsdf,
    /// ReSTIR DI: temporal reservoir reuse at the primary vertex, on the
    /// reference accumulator.
    RestirDi,
    /// ReSTIR DI with the spatial merge+shade pass on top.
    RestirDiSpatial,
    /// ReSTIR DI + GI reconnection reservoirs (recon shading + temporal reuse).
    RestirGi,
    /// ReSTIR DI + GI with spatial reuse on both arms.
    RestirGiSpatial,
    /// Reference accumulator with NRC GI termination (biased by cache error —
    /// grade by freeze-diff/RMSE against [`Reference`](Self::Reference)).
    Nrc,
    /// The stack exactly as shipped: [`SolariLighting::default()`] (per-frame
    /// ReSTIR DI + GI with NRC termination, DLSS RR downstream). Deliberately
    /// NOT a separate configuration — the default IS production, NRC and all.
    Default,
}

impl SolariRecipe {
    /// Every recipe, for UI dropdowns and benchmark sweeps.
    pub const ALL: [SolariRecipe; 8] = [
        SolariRecipe::Reference,
        SolariRecipe::Bsdf,
        SolariRecipe::RestirDi,
        SolariRecipe::RestirDiSpatial,
        SolariRecipe::RestirGi,
        SolariRecipe::RestirGiSpatial,
        SolariRecipe::Nrc,
        SolariRecipe::Default,
    ];

    /// Short kebab-case name (matches the CLI `--recipe` values).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Bsdf => "bsdf",
            Self::RestirDi => "restir-di",
            Self::RestirDiSpatial => "restir-di-spatial",
            Self::RestirGi => "restir-gi",
            Self::RestirGiSpatial => "restir-gi-spatial",
            Self::Nrc => "nrc",
            Self::Default => "default",
        }
    }

    /// Whether `mode` runs this recipe's estimator. The session knobs
    /// (samples per frame, accumulation) are ignored — they tune how fast the
    /// estimator converges / whether frames average, not what it computes.
    pub fn matches(&self, mode: &SolariLighting) -> bool {
        let mut canonical = mode.clone();
        if let SolariLighting::Reference(reference) = &mut canonical {
            let defaults = SolariReference::default();
            reference.samples_per_frame = defaults.samples_per_frame;
            reference.accumulate = defaults.accumulate;
        }
        canonical == self.mode()
    }

    /// The fully-configured lighting mode this recipe names.
    pub fn mode(&self) -> SolariLighting {
        let di_restir = DiEstimator::Restir {
            ris_candidates: 4,
            m_cap: 20.0,
            spatial: None,
        };
        match self {
            Self::Reference => SolariLighting::Reference(SolariReference::default()),
            Self::Bsdf => SolariLighting::Reference(SolariReference {
                di: DiEstimator::BsdfOnly,
                ..Default::default()
            }),
            Self::RestirDi => SolariLighting::Reference(SolariReference {
                di: di_restir,
                ..Default::default()
            }),
            Self::RestirDiSpatial => SolariLighting::Reference(SolariReference {
                di: DiEstimator::Restir {
                    ris_candidates: 4,
                    m_cap: 20.0,
                    spatial: Some(SpatialReuse::default()),
                },
                ..Default::default()
            }),
            Self::RestirGi => SolariLighting::Reference(SolariReference {
                di: di_restir,
                gi: GiEstimator::Restir {
                    recon: true,
                    temporal: true,
                    spatial: None,
                    dead_view: false,
                },
                ..Default::default()
            }),
            Self::RestirGiSpatial => SolariLighting::Reference(SolariReference {
                di: DiEstimator::Restir {
                    ris_candidates: 4,
                    m_cap: 20.0,
                    spatial: Some(SpatialReuse::default()),
                },
                gi: GiEstimator::Restir {
                    recon: true,
                    temporal: true,
                    spatial: Some(SpatialReuse::default()),
                    dead_view: false,
                },
                ..Default::default()
            }),
            Self::Nrc => SolariLighting::Reference(SolariReference {
                nrc_gi: true,
                ..Default::default()
            }),
            Self::Default => SolariLighting::default(),
        }
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
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub struct SolariReference {
    /// Paths traced per pixel per frame (inner raygen loop). Raise to converge faster.
    pub samples_per_frame: u32,
    /// When false, render fresh frames instead of averaging (estimator levers stay
    /// active). With the rung-0 dump this captures a SINGLE warmed restir frame —
    /// the per-frame variance metric temporal reuse actually improves.
    pub accumulate: bool,
    /// Which part of the transport to output (full, DI-only, GI-only).
    pub output: ReferenceOutput,
    /// The direct-illumination estimator at each path vertex.
    pub di: DiEstimator,
    /// The indirect (suffix) estimator past the primary vertex.
    pub gi: GiEstimator,
    /// NRC (zero/docs/nrc.md rung 2): terminate GI paths at bounce 2 into the
    /// neural radiance cache (estimator flag bit 20). Composes with either
    /// [`GiEstimator`]. Biased by cache error — graded by freeze-diff/RMSE
    /// against the untouched reference.
    pub nrc_gi: bool,
}

impl Default for SolariReference {
    fn default() -> Self {
        Self {
            samples_per_frame: 4,
            accumulate: true,
            output: ReferenceOutput::Full,
            di: DiEstimator::Nee { ris_candidates: 1 },
            gi: GiEstimator::PathTraced,
            nrc_gi: false,
        }
    }
}

impl SolariReference {
    /// ReSTIR DI temporal reuse is on.
    pub fn di_restir(&self) -> bool {
        matches!(self.di, DiEstimator::Restir { .. })
    }

    /// ReSTIR GI reconnection reservoirs are on.
    pub fn gi_restir(&self) -> bool {
        matches!(self.gi, GiEstimator::Restir { .. })
    }

    /// The DI spatial-reuse settings, when the ReSTIR DI spatial pass is on.
    pub fn di_spatial(&self) -> Option<&SpatialReuse> {
        match &self.di {
            DiEstimator::Restir { spatial, .. } => spatial.as_ref(),
            _ => None,
        }
    }

    /// The GI spatial-reuse settings, when the ReSTIR GI spatial pass is on.
    pub fn gi_spatial(&self) -> Option<&SpatialReuse> {
        match &self.gi {
            GiEstimator::Restir { spatial, .. } => spatial.as_ref(),
            GiEstimator::PathTraced => None,
        }
    }

    /// RIS candidates per NEE sample (1 for the BSDF-only estimator).
    pub fn ris_candidates(&self) -> u32 {
        match self.di {
            DiEstimator::BsdfOnly => 1,
            DiEstimator::Nee { ris_candidates }
            | DiEstimator::Restir { ris_candidates, .. } => ris_candidates.max(1),
        }
    }

    /// The temporal history cap (ReSTIR DI's, or the 20-frame default).
    pub fn m_cap(&self) -> f32 {
        match self.di {
            DiEstimator::Restir { m_cap, .. } => m_cap,
            _ => 20.0,
        }
    }

    /// The GI dead-canonical-draw instrument paint is on.
    pub fn gi_dead_view(&self) -> bool {
        matches!(self.gi, GiEstimator::Restir { dead_view: true, .. })
    }

    /// Any spatial pass's kill-stage debug paint is on.
    pub fn spatial_debug_paint(&self) -> bool {
        self.di_spatial().is_some_and(|sp| sp.debug_paint)
            || self.gi_spatial().is_some_and(|sp| sp.debug_paint)
    }
}

/// Which part of the transport a [`SolariReference`] camera outputs.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Default, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub enum ReferenceOutput {
    /// The full image.
    #[default]
    Full,
    /// Direct illumination only: terminate every path at the primary vertex
    /// (emissive + one NEE/reservoir estimate, no bounces). The standard ReSTIR
    /// evaluation image — indirect noise otherwise buries the DI variance win.
    DiOnly,
    /// Indirect only (rung 4a): the complement — output `A₀·L_gi`, the suffix
    /// energy past the primary vertex. DiOnly + GiOnly = the full image.
    GiOnly,
}

/// The direct-illumination estimator at each path vertex.
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Clone, PartialEq)]
pub enum DiEstimator {
    /// BSDF-only brute force (no next-event estimation). Validation lever: NEE
    /// on and off MUST converge to the same image — any difference is a pdf/MIS bug.
    BsdfOnly,
    /// Next-event estimation with RIS (rung 2): M light candidates stream
    /// through a one-slot reservoir, one shadow ray for the winner.
    Nee {
        /// RIS candidates per NEE sample. 1 = plain NEE (identical estimator);
        /// raise for receiver-aware light selection (clamped to 255).
        ris_candidates: u32,
    },
    /// ReSTIR DI (rung 3): persist the primary vertex's reservoir per pixel and
    /// temporally merge last frame's (reprojected + geometry-validated). Emissive
    /// candidates then run at the primary vertex only; bounce vertices fall back
    /// to single-sample emissive NEE and directionals are shaded per light.
    Restir {
        /// Initial RIS candidates per pixel streamed through the reservoir.
        ris_candidates: u32,
        /// Temporal history cap, ×`ris_candidates` — history counts for at most
        /// this many frames' worth of candidates (uncapped M = frozen shadows).
        m_cap: f32,
        /// Spatial reuse (rung 3 session 2): a post-trace compute pass merges
        /// each pixel's reservoir with disk neighbors and owns the winner's
        /// visibility + shade.
        spatial: Option<SpatialReuse>,
    },
}

/// The indirect (suffix) estimator past the primary vertex.
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Clone, PartialEq)]
pub enum GiEstimator {
    /// Plain path tracing of the suffix.
    PathTraced,
    /// ReSTIR GI (rung 4a.1): raygen stores the canonical GI sample
    /// `{x_s, n_s, L_gi, pdf, a0}` per pixel and shades GI from the STORED
    /// sample.
    Restir {
        /// Reshade `f(x_v,ω)·cos·L/pdf` from the surface G-buffer instead of
        /// the stored exact `a0` — the reconnection-shift shading path
        /// temporal/spatial reuse relies on (gate: accumulated unbiasedness).
        recon: bool,
        /// Temporally merge last frame's reprojected GI reservoir (surface
        /// depth/normal validated, capped by the DI arm's m_cap).
        temporal: bool,
        /// The spatial pass merges neighbors' GI reservoirs
        /// (reconnection-Jacobian weighted, winner visibility) and owns the
        /// GI shade.
        spatial: Option<SpatialReuse>,
        /// Instrument: paint 1 where the canonical GI draw is dead (bounce-1
        /// miss or delta pdf) — the accumulated mean IS the dead-draw rate.
        dead_view: bool,
    },
}

/// Spatial reservoir reuse settings, shared by the DI and GI ReSTIR arms.
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub struct SpatialReuse {
    /// Neighbor taps per pixel (≤8).
    pub taps: u32,
    /// Neighbor disk radius, pixels.
    pub radius: f32,
    /// false = naive M-sum combiner (BIASED — the visible-darkening study);
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
/// reservoirs — the certified full stack (restir_roadmap rungs 3–4a), feeding
/// DLSS Ray Reconstruction as the denoiser.
///
/// Selected via [`SolariLighting::Realtime`] (the default).
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub struct SolariRestir {
    /// Initial light candidates per pixel streamed through the DI reservoir.
    pub ris_candidates: u32,
    /// Temporal history cap, ×`ris_candidates` — history counts for at most
    /// this many frames' worth of candidates.
    pub m_cap: f32,
    /// Spatial reuse. DEFAULT None — under DLSS RR, spatial reuse's disk-sized
    /// winner patches read as swimming pool-caustic light (correlated noise
    /// poses as illumination structure); RR does the variance reduction
    /// instead. The exam-certified spatial path stays available for non-RR
    /// consumers (equal-time tables in restir_roadmap).
    pub spatial: Option<SpatialReuse>,
    /// GI reconnection reservoirs (temporal + spatial). Off = ReSTIR DI only.
    pub gi: bool,
    /// Max GI luminance per frame, in DISPLAY-referred units (post-exposure,
    /// where ~1.0 is a well-exposed white) — reservoir spikes above it are
    /// scaled down luminance-preserving. 0 = off. The reference path never
    /// clamps (policy); this is the realtime firefly filter.
    pub firefly_clamp: f32,
    /// Terminate GI paths into the neural radiance cache (estimator flag
    /// bit 20) once the cache is mature (`NrcBuffers::step > 300`) — until
    /// then paths trace their full suffix. Biased by cache error. With
    /// [`gi`](Self::gi), cache-terminated paths query the cache inline in
    /// raygen so the GI reservoirs store the full suffix energy.
    pub nrc_gi: bool,
}

impl Default for SolariRestir {
    fn default() -> Self {
        Self {
            ris_candidates: 4,
            m_cap: 20.0,
            spatial: None,
            gi: true,
            firefly_clamp: 10.0,
            nrc_gi: true,
        }
    }
}
