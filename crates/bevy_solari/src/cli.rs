//! Reusable camera CLI for solari examples and tools (part of the
//! `bevy_solari_debug` feature).
//!
//! [`SolariCameraArgs`] is a [`clap::Args`] block: `--recipe` picks a named
//! [`SolariRecipe`] setup, and the remaining levers modify it. Flatten it into
//! any binary's arg struct and spawn what [`SolariCameraArgs::camera`] returns
//! on the camera, and that binary speaks the same estimator dialect as
//! `solari_furnace`:
//!
//! ```ignore
//! #[derive(clap::Parser)]
//! struct Args {
//!     #[command(flatten)]
//!     camera: SolariCameraArgs,
//! }
//!
//! commands.spawn((Camera3d::default(), args.camera.camera(), /* ... */));
//! ```

use crate::render::rt_pipeline::SolariDebugView;
use crate::render::{
    DiEstimator, GiEstimator, ReferenceOutput, SolariCamera, SolariLighting, SolariRecipe,
    SpatialReuse,
};

/// The shared camera levers: a [`SolariRecipe`] plus modifiers. The default is
/// the reference estimator rendering fresh frames (live feedback for
/// interactive lever testing); `--accum` opts into progressive accumulation
/// (the exam/convergence mode), and `--recipe default` selects the shipped
/// realtime stack (`SolariLighting::default()`).
#[derive(clap::Args, Clone, Debug)]
pub struct SolariCameraArgs {
    /// estimator recipe: reference | bsdf | restir-di | restir-di-spatial |
    /// restir-gi | restir-gi-spatial | nrc | default (the shipped realtime
    /// stack, `SolariLighting::default()`). The levers below modify it
    #[arg(long, value_enum, default_value_t = SolariRecipe::Reference)]
    pub recipe: SolariRecipe,

    /// reference: paths per pixel per frame (default is interactive-friendly;
    /// exams typically pass 16+)
    #[arg(long, default_value_t = 4)]
    pub spp: u32,

    /// reference: progressively accumulate frames into a converging mean (the
    /// exam mode). Default is fresh frames — live feedback while testing
    /// levers, restir history stays warm
    #[arg(long)]
    pub accum: bool,

    /// direct illumination only (terminate at the primary vertex)
    #[arg(long)]
    pub di_only: bool,

    /// indirect only (suffix energy past the primary vertex);
    /// di_only + gi_only must sum to the full image
    #[arg(long)]
    pub gi_only: bool,

    /// override RIS/initial candidates per pixel
    #[arg(long)]
    pub ris: Option<u32>,

    /// override spatial neighbor taps (>0 enables spatial reuse on the
    /// recipe's ReSTIR arms, 0 disables it)
    #[arg(long)]
    pub taps: Option<u32>,

    /// override the spatial neighbor disk radius in pixels
    #[arg(long)]
    pub radius: Option<f32>,

    /// visibility-aware 1/Z spatial combiner (research lever; costs one
    /// shadow ray per contributor)
    #[arg(long)]
    pub zcount: bool,

    /// override the temporal history cap (reservoir persistence in frames)
    #[arg(long)]
    pub mcap: Option<f32>,

    /// default recipe: override the GI firefly clamp, display-referred luminance
    /// (0 = off)
    #[arg(long)]
    pub clamp: Option<f32>,

    /// reference recipes: force NRC GI termination ON
    #[arg(long)]
    pub nrc_gi: bool,

    /// default recipe: disable GI reconnection reservoirs (ReSTIR DI only)
    #[arg(long)]
    pub no_gi: bool,

    /// debug view: `heatmap` (per-pixel cost), `any-hit`, `displacement`,
    /// `cluster`, `triangles`, `normal-facing`, `nrc` (cache paint), `spatial`
    /// (kill-stage paint on the recipe's spatial arms), `gi-dead`
    /// (dead-canonical rate; needs a restir-gi recipe)
    #[arg(long, default_value = "")]
    pub debug_view: String,
}

/// Apply the spatial-reuse overrides to one ReSTIR arm's settings.
fn override_spatial(
    spatial: &mut Option<SpatialReuse>,
    taps: Option<u32>,
    radius: Option<f32>,
    zcount: bool,
    debug_paint: bool,
) {
    if let Some(taps) = taps {
        if taps == 0 {
            *spatial = None;
            return;
        }
        spatial.get_or_insert_default().taps = taps;
    }
    if let Some(sp) = spatial {
        if let Some(radius) = radius {
            sp.radius = radius;
        }
        if zcount {
            sp.unbiased_zcount = true;
        }
        if debug_paint {
            sp.debug_paint = true;
        }
    }
}

impl SolariCameraArgs {
    /// The [`SolariCamera`] these levers select — spawn it on the camera
    /// entity. The recipe is stamped out first, then the modifiers apply.
    pub fn camera(&self) -> SolariCamera {
        let mut mode = self.recipe.mode();
        let spatial_paint = self.debug_view == "spatial";
        match &mut mode {
            SolariLighting::Realtime(rt) => {
                if let Some(ris) = self.ris {
                    rt.ris_candidates = ris.max(1);
                }
                if let Some(m_cap) = self.mcap {
                    rt.m_cap = m_cap;
                }
                if let Some(clamp) = self.clamp {
                    rt.firefly_clamp = clamp;
                }
                if self.no_gi {
                    rt.gi = false;
                }
                if self.nrc_gi {
                    rt.nrc_gi = true;
                }
                override_spatial(&mut rt.spatial, self.taps, self.radius, self.zcount, spatial_paint);
            }
            SolariLighting::Reference(r) => {
                r.samples_per_frame = self.spp;
                r.accumulate = self.accum;
                if self.di_only {
                    r.output = ReferenceOutput::DiOnly;
                }
                if self.gi_only {
                    r.output = ReferenceOutput::GiOnly;
                }
                if self.nrc_gi {
                    r.nrc_gi = true;
                }
                match &mut r.di {
                    DiEstimator::BsdfOnly => {}
                    DiEstimator::Nee { ris_candidates } => {
                        if let Some(ris) = self.ris {
                            *ris_candidates = ris.max(1);
                        }
                    }
                    DiEstimator::Restir { ris_candidates, m_cap, spatial } => {
                        if let Some(ris) = self.ris {
                            *ris_candidates = ris.max(1);
                        }
                        if let Some(cap) = self.mcap {
                            *m_cap = cap;
                        }
                        override_spatial(spatial, self.taps, self.radius, self.zcount, spatial_paint);
                    }
                }
                if let GiEstimator::Restir { spatial, dead_view, .. } = &mut r.gi {
                    override_spatial(spatial, self.taps, self.radius, self.zcount, spatial_paint);
                    if self.debug_view == "gi-dead" {
                        *dead_view = true;
                    }
                }
            }
        }
        SolariCamera {
            mode,
            debug: self.debug_view(),
        }
    }

    /// The [`SolariDebugView`] the `--debug-view` string selects. `spatial`
    /// and `gi-dead` are estimator paints (flags inside the mode tree, set by
    /// [`Self::camera`]), not views — they map to [`SolariDebugView::None`].
    pub fn debug_view(&self) -> SolariDebugView {
        match self.debug_view.as_str() {
            "heatmap" | "cost" => SolariDebugView::cost_heatmap(),
            "any-hit" => SolariDebugView::any_hit_count(),
            "displacement" => SolariDebugView::Displacement,
            "cluster" | "clusters" => SolariDebugView::Clusters,
            "triangles" => SolariDebugView::Triangles,
            "normal-facing" => SolariDebugView::NormalFacing,
            "nrc" => SolariDebugView::NrcCache,
            _ => SolariDebugView::None,
        }
    }
}
