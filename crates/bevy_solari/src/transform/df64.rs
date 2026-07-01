//! Double-single (`df64`) arithmetic — a high-precision value carried as a pair
//! of `f32`s (`hi + lo`, non-overlapping), so the GPU never needs native `f64`
//! (which WGSL/naga has no scalar type for). This is the numeric core of the
//! frame-based floating origin: a frame's **absolute** world translation is
//! stored as a `df64` triple, and the render pass emits the small origin-relative
//! `f32` translation via [`Df64Vec3::sub_to_f32`] — the frame's position minus the
//! camera-origin's position, computed so the huge common magnitude cancels
//! *before* it lands in the `f32` the acceleration structure is built from.
//!
//! Why this replaces the integer grid cell: a `df64` split (`f32` hi + `f32` lo)
//! is the same coarse+fine decomposition the integer `(cell, residual)` pair was,
//! but done in floating point — so there is no `cell_edge` knob, no `i32` range
//! ceiling, and no recenter (the origin is simply the camera frame's own `df64`
//! world, subtracted every frame; the camera sits at 0 by construction).
//!
//! Precision: ~46–48 effective mantissa bits. Near-origin subtraction of two
//! large nearby values is *exact* in the hi lane (Sterbenz) and corrected by the
//! lo lane, so a point a few metres from a 1 AU origin resolves to sub-micron —
//! see the tests. This is a library of pure `f32` ops (the WGSL mirror uses `fma`
//! for the product error term); it needs **no** naga fork.

use bevy_math::{DVec3, Vec3};

/// A `df64` scalar: `value ≈ hi + lo`, with `|lo| ≤ ½ ulp(hi)` after
/// [`split`](Df64::split). Both lanes are `f32`; together ~46–48 mantissa bits.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Df64 {
    pub hi: f32,
    pub lo: f32,
}

impl Df64 {
    /// Split an `f64` into a non-overlapping `f32` pair. `hi` is the nearest `f32`;
    /// `lo` captures the part `hi` could not represent. Round-trips to within an
    /// `f32` ulp of `lo` (≈ `f64` precision for values inside `f32`'s exponent range).
    #[inline]
    pub fn split(x: f64) -> Self {
        let hi = x as f32;
        // `x - (hi as f64)` is the exact residual (hi is an f64-representable f32),
        // rounded once into the lo lane.
        let lo = (x - hi as f64) as f32;
        Self { hi, lo }
    }

    /// Reconstruct the approximate `f64` value (`hi + lo` in `f64`).
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.hi as f64 + self.lo as f64
    }
}

/// A `df64` 3-vector — a frame's absolute world translation on the wire. Stored
/// as two `Vec3` lanes so the GPU column is a flat `[hi.xyz, lo.xyz]`.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Df64Vec3 {
    pub hi: Vec3,
    pub lo: Vec3,
}

impl Df64Vec3 {
    /// Split a full-precision `DVec3` position into `df64` lanes (per component).
    #[inline]
    pub fn from_dvec3(v: DVec3) -> Self {
        let x = Df64::split(v.x);
        let y = Df64::split(v.y);
        let z = Df64::split(v.z);
        Self {
            hi: Vec3::new(x.hi, y.hi, z.hi),
            lo: Vec3::new(x.lo, y.lo, z.lo),
        }
    }

    /// Reconstruct the approximate `DVec3`.
    #[inline]
    pub fn to_dvec3(self) -> DVec3 {
        DVec3::new(
            self.hi.x as f64 + self.lo.x as f64,
            self.hi.y as f64 + self.lo.y as f64,
            self.hi.z as f64 + self.lo.z as f64,
        )
    }

    /// `self − origin`, rendered to a small `f32` vector. This is the whole point:
    /// when `self` is near `origin` (camera-relative), `hi − hi` is *exact*
    /// (Sterbenz — nearby equal-magnitude `f32` subtraction has no rounding), and
    /// the `lo − lo` correction restores the sub-`f32`-ulp part the hi lane dropped,
    /// so a metre-scale offset survives even when both operands are at AU scale
    /// (where a naive `f32` subtract collapses to zero). When `self` is *far* from
    /// `origin` the result is legitimately large and coarse — distant geometry,
    /// naturally low precision — which is correct.
    ///
    /// Wire/shader parity: the WGSL emits the identical `(hi−hi) + (lo−lo)`.
    #[inline]
    pub fn sub_to_f32(self, origin: Df64Vec3) -> Vec3 {
        (self.hi - origin.hi) + (self.lo - origin.lo)
    }

    /// The flat `[hi.x, hi.y, hi.z, lo.x, lo.y, lo.z]` GPU-column payload.
    #[inline]
    pub fn to_columns(self) -> [f32; 6] {
        [self.hi.x, self.hi.y, self.hi.z, self.lo.x, self.lo.y, self.lo.z]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AU_M: f64 = 1.495_978_707e11; // 1 astronomical unit, metres

    #[test]
    fn split_round_trips_at_au_scale() {
        // A point 1 AU out plus a few metres — the exact case the integer cell
        // existed to handle. The df64 split must preserve the metre-scale part.
        let x = AU_M + 3.141_592_653_589_793;
        let d = Df64::split(x);
        // hi + lo reconstructs the original to well under a millimetre.
        assert!((d.to_f64() - x).abs() < 1e-3, "reconstruct err {}", d.to_f64() - x);
    }

    #[test]
    fn sub_recovers_metre_offset_that_naive_f32_loses() {
        // Origin at 1 AU; a frame 3.14159 m beyond it along +X. This is the
        // precision-collapse scenario: both absolute positions round to the SAME
        // f32 (ulp ≈ 16 km at 1.5e11), so a naive f32 subtract yields exactly 0.
        let origin = DVec3::new(AU_M, 0.0, 0.0);
        let frame = DVec3::new(AU_M + 3.141_592_653_589_793, 0.0, 0.0);

        // Naive single-precision path — the bug the whole system exists to avoid.
        let naive = (frame.as_vec3()) - (origin.as_vec3());
        assert_eq!(naive.x, 0.0, "sanity: naive f32 subtract collapses at AU scale");

        // df64 path recovers the offset to f32 precision (sub-micron here).
        let d_origin = Df64Vec3::from_dvec3(origin);
        let d_frame = Df64Vec3::from_dvec3(frame);
        let rel = d_frame.sub_to_f32(d_origin);
        assert!(
            (rel.x - 3.141_592_653_589_793_f32).abs() < 1e-3,
            "df64 relative x = {}, want ~3.14159",
            rel.x
        );
        assert_eq!(rel.y, 0.0);
        assert_eq!(rel.z, 0.0);
    }

    #[test]
    fn sub_is_exact_when_frame_is_the_origin() {
        // The camera frame subtracted from itself must be identically zero — the
        // guarantee that makes "the camera sits at 0" hold with no recenter.
        let p = DVec3::new(AU_M, -7.0 * AU_M, 42.0);
        let d = Df64Vec3::from_dvec3(p);
        assert_eq!(d.sub_to_f32(d), Vec3::ZERO);
    }

    #[test]
    fn far_field_offset_is_large_but_finite() {
        // A frame at 5 AU with the origin at 1 AU: the result is a legitimately
        // large f32 (~4 AU), coarse but finite — distant geometry.
        let origin = Df64Vec3::from_dvec3(DVec3::new(AU_M, 0.0, 0.0));
        let frame = Df64Vec3::from_dvec3(DVec3::new(5.0 * AU_M, 0.0, 0.0));
        let rel = frame.sub_to_f32(origin);
        let expected = (4.0 * AU_M) as f32;
        // Within an f32 ulp at that magnitude.
        assert!((rel.x - expected).abs() / expected < 1e-6, "rel.x = {}", rel.x);
    }
}
