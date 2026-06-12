//! Caustics (phase B) — photon-mapped light through specular chains.
//!
//! The reservoir passes structurally cannot find light → glass → diffuse
//! transport: NEE can't thread a delta chain, and BSDF rays from a floor
//! pixel almost never blunder through a prism into the sun. The caustic
//! passes invert the search: photons are fired FROM the sun, traced through
//! the existing glass code (exact Fresnel, nested media, dispersion —
//! rainbows come along free), and whatever lands on a diffuse surface after
//! a specular chain is deposited into a world-space hash grid the realtime
//! shade pass gathers as an irradiance estimate.
//!
//! Everything lives on the GPU, in scene-column terms (`caustics.wgsl`):
//! a reduce pass scans the live instance columns — material ids for
//! "transmissive?", the mesh-local AABB column × the LIVE transform column
//! for world bounds — and derives the photon-emission rect (the union of
//! the glass, projected perpendicular to the first directional light); a
//! finalize pass turns it into per-photon power. No CPU mirror of the scene
//! is consulted: material swaps, GPU-propagated transforms, and sun motion
//! are picked up the frame they happen, and the cluster asset (consumed at
//! upload) is never needed again.
//!
//! Because the grid lives in world space and keeps a ~16-frame exponential
//! running average, the caustic is stable under camera motion — walk around
//! the spectrum and it just sits there — and re-converges after a light or
//! glass change with no explicit invalidation. The pathtracer deliberately
//! does NOT gather (it keeps its path-regularized caustic NEE), so it
//! remains the independent reference to validate the grid against.

/// Photons emitted per frame. Keep in sync with `CAUSTIC_PHOTONS` in
/// `caustics.wgsl`.
pub const CAUSTIC_PHOTONS: u32 = 262_144;

/// Words in the caustic emitter scratch/parameter buffer (see the layout in
/// `restir_bindings.wgsl`).
pub const CAUSTIC_EMITTER_WORDS: u64 = 32;
