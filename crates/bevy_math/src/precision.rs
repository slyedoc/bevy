//! Precision-switched spatial type aliases.
//!
//! By default the aliases resolve to the ordinary `f32` glam types. With the
//! `transform_f64` cargo feature they resolve to the double-precision `D*`
//! types, widening `Transform`-space math engine-wide (Godot's
//! `precision=double` / Unreal's Large World Coordinates model). Exactly one
//! mode exists per build: the feature is a single additive opt-in, and cargo
//! feature unification means any crate in the graph enabling it (e.g.
//! `bevy_solari`) switches the whole application.
//!
//! Only *transform-space* types are aliased. Render-boundary types stay `f32`
//! in both modes — [`Dir3`](crate::Dir3), [`Ray3d`](crate::Ray3d),
//! [`Isometry3d`](crate::Isometry3d), bounding volumes, frusta — because
//! directions and camera-relative quantities don't accumulate magnitude; the
//! precision the feature buys is in *positions*.
//!
//! The [`ToPrecision`] / [`ToRender`] traits bridge the two worlds without
//! `cfg` at call sites: `.to_precision()` widens (or passes through) into the
//! active transform precision and is always exact when the source is `f32`;
//! `.to_render()` truncates (or passes through) to `f32` and is lossy by
//! contract in `transform_f64` mode — call it only where an `f32` result is
//! semantically fine (GPU uploads, audio, gizmos, picking).

use crate::{
    Affine3A, DAffine3, DMat3, DMat4, DQuat, DVec2, DVec3, Mat3, Mat4, Quat, Vec2, Vec3, Vec3A,
};

#[cfg(not(feature = "transform_f64"))]
mod aliases {
    /// Scalar type of `Transform`-space math (`f32`, or `f64` with `transform_f64`).
    pub type TReal = f32;
    /// 2D vector in transform precision.
    pub type TVec2 = crate::Vec2;
    /// 3D vector in transform precision.
    pub type TVec3 = crate::Vec3;
    /// Quaternion in transform precision.
    pub type TQuat = crate::Quat;
    /// 3×3 matrix in transform precision.
    pub type TMat3 = crate::Mat3;
    /// 4×4 matrix in transform precision.
    pub type TMat4 = crate::Mat4;
    /// Affine transform in transform precision. SIMD [`Affine3A`](crate::Affine3A)
    /// in `f32` mode, scalar [`DAffine3`](crate::DAffine3) with `transform_f64`.
    pub type TAffine3 = crate::Affine3A;
}

#[cfg(feature = "transform_f64")]
mod aliases {
    /// Scalar type of `Transform`-space math (`f32`, or `f64` with `transform_f64`).
    pub type TReal = f64;
    /// 2D vector in transform precision.
    pub type TVec2 = crate::DVec2;
    /// 3D vector in transform precision.
    pub type TVec3 = crate::DVec3;
    /// Quaternion in transform precision.
    pub type TQuat = crate::DQuat;
    /// 3×3 matrix in transform precision.
    pub type TMat3 = crate::DMat3;
    /// 4×4 matrix in transform precision.
    pub type TMat4 = crate::DMat4;
    /// Affine transform in transform precision. SIMD [`Affine3A`](crate::Affine3A)
    /// in `f32` mode, scalar [`DAffine3`](crate::DAffine3) with `transform_f64`.
    pub type TAffine3 = crate::DAffine3;
}

pub use aliases::*;

/// Convert into the active transform precision — widening (exact) in
/// `transform_f64` mode, identity otherwise. Never loses information when the
/// source is `f32`; converting an `f64` source in `f32` mode truncates (you
/// asked for the active precision, which is `f32`).
pub trait ToPrecision: Sized {
    /// The transform-precision counterpart of this type.
    type Precise;
    /// Convert into the active transform precision.
    fn to_precision(self) -> Self::Precise;
}

/// Convert to the `f32` render types — truncating in `transform_f64` mode,
/// identity otherwise. Lossy by contract: call only at boundaries where `f32`
/// is semantically fine (GPU uploads, audio, gizmos, picking, directions).
pub trait ToRender: Sized {
    /// The `f32` counterpart of this type.
    type Render;
    /// Convert to the `f32` render representation.
    fn to_render(self) -> Self::Render;
}

macro_rules! impl_precision_pair {
    ($f32_ty:ty, $f64_ty:ty, $alias:ty, $widen:ident, $narrow:ident) => {
        impl ToPrecision for $f32_ty {
            type Precise = $alias;
            #[inline]
            fn to_precision(self) -> $alias {
                #[cfg(not(feature = "transform_f64"))]
                {
                    self
                }
                #[cfg(feature = "transform_f64")]
                {
                    self.$widen()
                }
            }
        }
        impl ToPrecision for $f64_ty {
            type Precise = $alias;
            #[inline]
            fn to_precision(self) -> $alias {
                #[cfg(not(feature = "transform_f64"))]
                {
                    self.$narrow()
                }
                #[cfg(feature = "transform_f64")]
                {
                    self
                }
            }
        }
        impl ToRender for $f32_ty {
            type Render = $f32_ty;
            #[inline]
            fn to_render(self) -> $f32_ty {
                self
            }
        }
        impl ToRender for $f64_ty {
            type Render = $f32_ty;
            #[inline]
            fn to_render(self) -> $f32_ty {
                self.$narrow()
            }
        }
    };
}

impl_precision_pair!(Vec2, DVec2, TVec2, as_dvec2, as_vec2);
impl_precision_pair!(Vec3, DVec3, TVec3, as_dvec3, as_vec3);
impl_precision_pair!(Quat, DQuat, TQuat, as_dquat, as_quat);
impl_precision_pair!(Mat3, DMat3, TMat3, as_dmat3, as_mat3);
impl_precision_pair!(Mat4, DMat4, TMat4, as_dmat4, as_mat4);
impl_precision_pair!(Affine3A, DAffine3, TAffine3, as_daffine3, as_affine3a);

impl ToPrecision for f32 {
    type Precise = TReal;
    #[inline]
    fn to_precision(self) -> TReal {
        TReal::from(self)
    }
}
impl ToPrecision for f64 {
    type Precise = TReal;
    #[inline]
    fn to_precision(self) -> TReal {
        #[cfg(not(feature = "transform_f64"))]
        {
            self as f32
        }
        #[cfg(feature = "transform_f64")]
        {
            self
        }
    }
}
impl ToRender for f32 {
    type Render = f32;
    #[inline]
    fn to_render(self) -> f32 {
        self
    }
}
impl ToRender for f64 {
    type Render = f32;
    #[inline]
    fn to_render(self) -> f32 {
        self as f32
    }
}

// `Vec3A` is a render-side SIMD type with no f64 analog: widening is
// one-directional, narrowing goes to plain `Vec3`.
impl ToPrecision for Vec3A {
    type Precise = TVec3;
    #[inline]
    fn to_precision(self) -> TVec3 {
        #[cfg(not(feature = "transform_f64"))]
        {
            Vec3::from(self)
        }
        #[cfg(feature = "transform_f64")]
        {
            self.as_dvec3()
        }
    }
}
impl ToRender for Vec3A {
    type Render = Vec3;
    #[inline]
    fn to_render(self) -> Vec3 {
        Vec3::from(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_are_exact_from_f32() {
        let v = Vec3::new(1.5, -2.25, 3.125);
        assert_eq!(v.to_precision().to_render(), v);
        let q = Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        assert_eq!(q.to_precision().to_render(), q);
        let a = Affine3A::from_translation(v);
        assert_eq!(a.to_precision().to_render(), a);
    }

    #[test]
    fn alias_matches_mode() {
        #[cfg(feature = "transform_f64")]
        assert_eq!(core::mem::size_of::<TVec3>(), 24);
        #[cfg(not(feature = "transform_f64"))]
        assert_eq!(core::mem::size_of::<TVec3>(), 12);
    }
}
