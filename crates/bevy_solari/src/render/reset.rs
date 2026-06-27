use bevy_ecs::{change_detection::DetectChanges, component::Component, query::With, system::{Commands, Query, Res}, world::Ref};
use bevy_reflect::Reflect;
use bevy_render::{Extract, sync_world::RenderEntity};

use crate::render::view::SolariViewState;
use crate::render::SolariCamera;

/// Per-frame "drop temporal history" flags on a solari camera's render entity.
/// Reset-reason systems in extract set the bits; the main-world copy (required by
/// [`SolariCamera`]) doubles as an app-side request channel — write
/// [`CameraReset::request`] to it (e.g. after changing lighting under a
/// progressive integrator) and [`reset_render_on_request`] forwards it for that
/// frame.
///
/// Two reasons, because a floating-origin **reframe** (a recenter, or a frame
/// handoff when the camera crosses from one [`SolariFrame`](crate::transform::SolariFrame)
/// into another) is stronger than a plain history drop: the previous frame's
/// instance transforms *and* view-projection were expressed in the OLD origin
/// frame, so motion-vector reprojection across the discontinuity is invalid —
/// not just stale. Denoisers must drop history (`history`) AND consumers that
/// reproject via motion vectors must treat this frame as having no valid history
/// (`reframed`), since the rebase rotates/translates the whole basis.
#[derive(Component, Default, Reflect, Clone)]
pub struct CameraReset {
    /// Drop temporal denoiser / accumulation history this frame (DLSS reset,
    /// ReSTIR temporal reservoirs). Set by any reset reason.
    pub history: bool,
    /// The origin reference frame rotated/translated this frame (recenter or
    /// frame handoff): previous-frame transforms / view-proj are in the old
    /// frame, so motion-vector reprojection is invalid. Implies `history`.
    pub reframed: bool,
}

impl CameraReset {
    /// App-side request for a plain history drop (the old `CameraReset(true)`).
    pub const fn request() -> Self {
        Self { history: true, reframed: false }
    }

    /// A floating-origin reframe: drop history *and* invalidate motion-vector
    /// reprojection (the basis moved). Pulsed by the recenter / frame handoff.
    pub const fn reframe() -> Self {
        Self { history: true, reframed: true }
    }

    /// Any reset is active this frame (a reframe always drops history too).
    #[inline]
    pub const fn active(&self) -> bool {
        self.history || self.reframed
    }
}

/// Clears the per-frame reset flags at the start of extract, before any
/// reset-reason system runs. The render entity is retained, so without this a
/// reason that fired once (e.g. a camera move) would stay latched on forever.
/// Every reset-reason system runs `.after` this and only ever sets bits.
pub fn clear_camera_reset(mut resets: Query<&mut CameraReset>) {
    for mut reset in &mut resets {
        *reset = CameraReset::default();
    }
}

/// Reset reason: the app (or the recenter) set a reset on the main-world
/// [`CameraReset`] (edge-triggered by change detection, so a flag left set
/// doesn't keep resetting). Forwards both reasons so a recenter's `reframed`
/// bit reaches the render world.
pub fn reset_render_on_request(
    cameras: Extract<Query<(RenderEntity, Ref<CameraReset>), With<SolariCamera>>>,
    mut commands: Commands,
) {
    for (e, reset) in &cameras {
        if reset.is_changed() && reset.active() {
            commands.entity(e).insert((*reset).clone());
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
        commands.entity(e).insert(CameraReset::request());
    }
}