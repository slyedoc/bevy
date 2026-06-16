//! The [`HairAsset`] — a set of hair/fur strands as poly-lines of control
//! points with per-point radii. Uploaded to the GPU and built into a
//! linear-swept-sphere BLAS by [`super::manager`].

use bevy_asset::Asset;
use bevy_math::Vec3;
use bevy_reflect::TypePath;

/// A groom: one or more strands. Each strand is swept into a chain of capped
/// cylinders (linear swept spheres) at BLAS-build time.
#[derive(Asset, TypePath, Clone, Debug, Default)]
pub struct HairAsset {
    pub strands: Vec<HairStrand>,
}

/// A single strand: a poly-line of control points with a radius at each point
/// (radius is linearly swept between consecutive points). `points` and `radii`
/// must be the same length and have at least two entries.
#[derive(Clone, Debug, Default)]
pub struct HairStrand {
    pub points: Vec<Vec3>,
    pub radii: Vec<f32>,
}

impl HairStrand {
    /// A strand from control points, tapering linearly from `root_radius` at
    /// the first point to `tip_radius` at the last — the usual hair profile.
    pub fn tapered(points: Vec<Vec3>, root_radius: f32, tip_radius: f32) -> Self {
        let n = points.len();
        let radii = (0..n)
            .map(|i| {
                let t = if n <= 1 { 0.0 } else { i as f32 / (n - 1) as f32 };
                root_radius * (1.0 - t) + tip_radius * t
            })
            .collect();
        Self { points, radii }
    }
}

impl HairAsset {
    /// Total control points across all strands (debugging / sizing).
    pub fn vertex_count(&self) -> usize {
        self.strands.iter().map(|s| s.points.len()).sum()
    }

    /// Total swept segments (`Σ (points - 1)`).
    pub fn segment_count(&self) -> usize {
        self.strands
            .iter()
            .map(|s| s.points.len().saturating_sub(1))
            .sum()
    }
}
