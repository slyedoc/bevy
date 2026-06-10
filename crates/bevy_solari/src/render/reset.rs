use bevy_camera::Camera;
use bevy_ecs::{change_detection::DetectChanges, component::Component, query::With, system::{Commands, Query, Res}, world::Ref};
use bevy_reflect::Reflect;
use bevy_render::{Extract, sync_world::RenderEntity};
use bevy_transform::components::GlobalTransform;

use crate::render::view::SolariViewState;
use crate::render::SolariCamera;

#[derive(Component, Default, Reflect, Clone)]
pub struct CameraReset(pub bool);

/// Clears the per-frame reset flag at the start of extract, before any
/// reset-reason system runs. The render entity is retained, so without this a
/// reason that fired once (e.g. a camera move) would stay latched on forever.
/// Every reset-reason system runs `.after` this and only ever sets it `true`.
pub fn clear_camera_reset(mut resets: Query<&mut CameraReset>) {
    for mut reset in &mut resets {
        reset.0 = false;
    }
}

/// Reset reason: the camera moved this frame — temporal history is invalid, so
/// force a reset (pathtracer re-clears accumulation, DLSS drops its history).
pub fn reset_render_on_camera_move(
    cameras_3d: Extract<
        Query<(
            RenderEntity,
            &Camera,
            Ref<GlobalTransform>,
        ), With<SolariCamera>>,
    >,
    mut commands: Commands,
) {
    for (e, camera, global_transform) in &cameras_3d {
        if camera.is_active && global_transform.is_changed() {
            commands.entity(e).insert(CameraReset(true));
        }
    }
}

/// Reset reason: [`SolariViewState`] changed (integrator or debug view
/// switched) — accumulated history belongs to the previous mode, and the scene
/// kept moving while the other mode rendered.
pub fn reset_render_on_view_state_change(
    state: Extract<Res<SolariViewState>>,
    cameras: Extract<Query<RenderEntity, With<SolariCamera>>>,
    mut commands: Commands,
) {
    if !state.is_changed() {
        return;
    }
    for e in &cameras {
        commands.entity(e).insert(CameraReset(true));
    }
}