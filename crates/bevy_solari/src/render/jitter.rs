//! Camera sub-pixel jitter handling for the RT path.

use crate::render::SolariCamera;
use bevy_ecs::prelude::*;
use bevy_math::Vec2;
use bevy_render::camera::TemporalJitter;

/// Zero the camera's sub-pixel jitter every frame. Jitter exists for a temporal
/// resolver; the RT path writes a fresh frame with no temporal accumulator, so a
/// camera offset is pure image wobble — sub-pixel on directly-seen surfaces, but
/// lens-magnified to several pixels through refractive glass. Keep it at zero.
pub fn zero_solari_jitter(mut views: Query<&mut TemporalJitter, With<SolariCamera>>) {
    for mut jitter in &mut views {
        jitter.offset = Vec2::ZERO;
    }
}
