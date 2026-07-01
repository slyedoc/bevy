use bevy_math::{Affine3A, Isometry3d, Mat4, ToPrecision, ToRender, Vec3};

use crate::prelude::{GlobalTransform, Transform};

/// A trait for point transformation methods.
///
/// Always `f32` (lossy under `transform_f64`); use the inherent
/// `transform_point` methods for full transform precision.
pub trait TransformPoint {
    /// Transform a point.
    fn transform_point(&self, point: impl Into<Vec3>) -> Vec3;
}

impl TransformPoint for Transform {
    #[inline]
    fn transform_point(&self, point: impl Into<Vec3>) -> Vec3 {
        self.transform_point(point.into().to_precision())
            .to_render()
    }
}

impl TransformPoint for GlobalTransform {
    #[inline]
    fn transform_point(&self, point: impl Into<Vec3>) -> Vec3 {
        self.transform_point(point.into().to_precision())
            .to_render()
    }
}

impl TransformPoint for Mat4 {
    #[inline]
    fn transform_point(&self, point: impl Into<Vec3>) -> Vec3 {
        self.transform_point3(point.into())
    }
}

impl TransformPoint for Affine3A {
    #[inline]
    fn transform_point(&self, point: impl Into<Vec3>) -> Vec3 {
        self.transform_point3(point.into())
    }
}

impl TransformPoint for Isometry3d {
    #[inline]
    fn transform_point(&self, point: impl Into<Vec3>) -> Vec3 {
        self.transform_point(point.into()).into()
    }
}
