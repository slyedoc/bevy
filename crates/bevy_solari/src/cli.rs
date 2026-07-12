//! Reusable camera CLI for solari examples and tools (part of the
//! `bevy_solari_debug` feature).
//!
//! [`SolariCameraArgs`] is a [`clap::Args`] block with ONE identity lane:
//! `--camera <RON>`, a complete [`SolariCamera`]. Every run prints its
//! effective config as `solari_camera: <RON>` — copy it, tweak a lever, and
//! feed it back. The session knobs (`--spp`, `--accum`) and `--debug-view`
//! ride alongside because they are protocol/instrumentation, not estimator
//! identity (see [`SolariCamera::name`]).
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
use crate::render::{SolariCamera, SolariLighting};

/// The shared camera levers. Default is the reference estimator rendering
/// fresh frames (live feedback for interactive testing); `--accum` opts into
/// progressive accumulation (the convergence mode); `--camera` selects
/// any estimator configuration verbatim.
#[derive(clap::Args, Clone, Debug)]
pub struct SolariCameraArgs {
    /// a complete SolariCamera as RON, e.g. '(mode: Realtime(()))' for the
    /// shipped realtime stack or '(mode: Reference((di: Restir(()))))' for
    /// ReSTIR DI on the reference frame. Struct fields may be omitted — their
    /// canonical defaults apply. Every run prints its effective config as
    /// `solari_camera:` — replay or tweak from that line. Default: the plain
    /// reference accumulator
    #[arg(long)]
    pub camera: Option<String>,

    /// reference: paths per pixel per frame (convergence runs typically use 16+)
    #[arg(long, default_value_t = 1)]
    pub spp: u32,

    /// reference: progressively accumulate frames into a converging mean.
    /// Default is fresh frames — live feedback while testing levers, restir
    /// history stays warm
    #[arg(long)]
    pub accum: bool,

    /// debug view: `heatmap` (per-pixel cost), `any-hit`, `displacement`,
    /// `cluster`, `triangles`, `normal-facing`, `nrc` (cache paint), `white`
    /// (lighting only — white base color, pure light transport)
    #[arg(long, default_value = "")]
    pub debug_view: String,
}

/// One-line RON for a [`SolariCamera`] — the `solari_camera:` wire format
/// every tool prints at startup and `--camera` accepts back.
pub fn camera_ron(camera: &SolariCamera) -> String {
    ron::to_string(camera).unwrap_or_default()
}

impl SolariCameraArgs {
    /// The [`SolariCamera`] these levers select — spawn it on the camera
    /// entity. `--camera <RON>` picks the estimator; the session knobs
    /// (--spp/--accum) apply on top of reference modes, since the work budget
    /// and accumulation policy belong to the session, not the identity.
    pub fn camera(&self) -> SolariCamera {
        let mut camera: SolariCamera = match &self.camera {
            Some(ron) => ron::from_str(ron)
                .unwrap_or_else(|e| panic!("--camera: invalid SolariCamera RON: {e}")),
            None => SolariCamera {
                mode: SolariLighting::Reference(Default::default()),
                debug: SolariDebugView::None,
            },
        };
        if let SolariLighting::Reference(r) = &mut camera.mode {
            r.samples_per_frame = self.spp;
            r.accumulate = self.accum;
        }
        if camera.debug == SolariDebugView::None {
            camera.debug = self.debug_view();
        }
        camera
    }

    /// The [`SolariDebugView`] the `--debug-view` string selects.
    pub fn debug_view(&self) -> SolariDebugView {
        match self.debug_view.as_str() {
            "heatmap" | "cost" => SolariDebugView::cost_heatmap(),
            "any-hit" => SolariDebugView::any_hit_count(),
            "displacement" => SolariDebugView::Displacement,
            "cluster" | "clusters" => SolariDebugView::Clusters,
            "triangles" => SolariDebugView::Triangles,
            "normal-facing" => SolariDebugView::NormalFacing,
            "nrc" => SolariDebugView::NrcCache,
            "white" | "lighting" => SolariDebugView::WhiteWorld,
            "" | "none" => SolariDebugView::None,
            other => {
                bevy_log::warn!(
                    "--debug-view: unknown value `{other}` (expected one of: heatmap/cost, \
                    any-hit, displacement, cluster/clusters, triangles, normal-facing, nrc, \
                    white/lighting, none); using none"
                );
                SolariDebugView::None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{DiEstimator, DiRestir, GiArm, GiEstimator, GiRestir, SolariReference, SpatialReuse};

    /// The identity lane must round-trip: serialize any camera, read it back,
    /// same estimator, same derived name.
    #[test]
    fn camera_ron_round_trips() {
        let cameras = [
            SolariCamera::default(),
            SolariCamera {
                mode: SolariLighting::Reference(SolariReference {
                    di: DiEstimator::Restir(DiRestir {
                        spatial: Some(SpatialReuse::default()),
                        ..Default::default()
                    }),
                    gi: Some(GiArm {
                        estimator: GiEstimator::Restir(GiRestir {
                            spatial: Some(SpatialReuse::default()),
                            ..Default::default()
                        }),
                        nrc: true,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                debug: SolariDebugView::None,
            },
        ];
        for camera in cameras {
            let ron = camera_ron(&camera);
            let back: SolariCamera =
                ron::from_str(&ron).unwrap_or_else(|e| panic!("{}: {e}\n{ron}", camera.name()));
            assert_eq!(back.mode, camera.mode, "{ron}");
            assert_eq!(back.name(), camera.name());
        }
    }

    /// Terse RON: struct fields may be omitted (serde defaults), so the
    /// realtime stack is just `(mode: Realtime(()))`.
    #[test]
    fn terse_ron_parses() {
        let camera: SolariCamera = ron::from_str("(mode: Realtime(()))").unwrap();
        assert_eq!(camera.mode, SolariLighting::default());
        assert_eq!(camera.name(), "rt");
        let camera: SolariCamera = ron::from_str("(mode: Reference(()))").unwrap();
        assert_eq!(camera.name(), "ref");
        let camera: SolariCamera =
            ron::from_str("(mode: Reference((di: Restir(()), gi: Some((estimator: Restir(()))))))")
                .unwrap();
        assert_eq!(camera.name(), "ref-di4m1-gi");
        let camera: SolariCamera = ron::from_str("(mode: Reference((gi: None)))").unwrap();
        assert_eq!(camera.name(), "ref-nogi");
        let camera: SolariCamera =
            ron::from_str("(mode: Reference((jitter: false, gi: None)))").unwrap();
        assert_eq!(camera.name(), "ref-nogi-nojit");
        let camera: SolariCamera =
            ron::from_str("(mode: Reference((gi: Some((bounces: 1)))))").unwrap();
        assert_eq!(camera.name(), "ref");
        let camera: SolariCamera =
            ron::from_str("(mode: Reference((gi: Some((bounces: 32)))))").unwrap();
        assert_eq!(camera.name(), "ref-b32");
        let camera: SolariCamera = ron::from_str("(mode: Realtime((bounces: 1)))").unwrap();
        assert_eq!(camera.name(), "rt");
        let camera: SolariCamera = ron::from_str("(mode: Realtime((bounces: 32)))").unwrap();
        assert_eq!(camera.name(), "rt-b32");
    }
}
