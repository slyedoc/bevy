use bevy_ecs::{change_detection::DetectChanges, component::Component, query::With, system::{Commands, Query, Res}, world::Ref};
use bevy_math::Mat4;
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
/// handoff when the camera crosses from one reference frame's basis
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

/// A one-frame, **motion-vector-continuous** basis correction for a floating-origin
/// frame handoff (the camera crossed from one reference frame's
/// basis into another). Where [`CameraReset::reframe`] *drops* temporal history at the
/// discontinuity, this **re-expresses** last frame's view-projection in the new origin
/// basis, so motion vectors stay continuous through the handoff — prefer it when the
/// source/destination frame poses are known, and fall back to the reset drop otherwise.
///
/// **Neutral by construction:** solari never names the physics layer. A bridge in the app
/// (which depends on both crates) observes the handoff — e.g. avian's `WorldTransferred`
/// fact event — reads the two frame anchors' poses (and `AngularVelocity` if it wants
/// continuity *during* a spinning-frame handoff, not just at the jump), and writes
/// `prev_from_current`: the rigid map taking a **current**-frame world point into the
/// **previous** frame's basis. solari post-multiplies the cached `prev_clip_from_world`
/// by it for exactly one frame (after which the prev cache is re-stored in the new basis).
/// Identity (the default) = no correction.
///
/// Rotational part follows the handoff: `R = R_to⁻¹ · R_from` (the source basis seen from
/// the destination); the translational part carries the anchors' relative offset at the
/// handoff instant. Both come off the frame entities — the single shared seam.
#[derive(Component, Clone, Copy, Debug, Reflect)]
pub struct CameraReframe {
    /// `prev_world_from_current_world`: cached `prev_clip_from_world` is post-multiplied
    /// by this so a current-basis world point reprojects to its previous-frame screen pos.
    pub prev_from_current: Mat4,
}

impl Default for CameraReframe {
    fn default() -> Self {
        Self { prev_from_current: Mat4::IDENTITY }
    }
}

impl CameraReframe {
    /// A correction from a `prev_world_from_current_world` rigid transform.
    pub const fn from_prev_from_current(prev_from_current: Mat4) -> Self {
        Self { prev_from_current }
    }

    /// No correction this frame (the default) — solari leaves `prev_clip_from_world` as-is.
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.prev_from_current == Mat4::IDENTITY
    }
}

/// Clears any forwarded render-world [`CameraReframe`] at extract start (mirrors
/// [`clear_camera_reset`]): the correction is one-frame, so a handoff that forwarded
/// one last frame must not keep applying it. [`extract_camera_reframe`] re-forwards it
/// only on the frame the main-world value actually changes.
pub fn clear_camera_reframe(mut reframes: Query<&mut CameraReframe>) {
    for mut reframe in &mut reframes {
        *reframe = CameraReframe::default();
    }
}

/// Forwards a main-world [`CameraReframe`] the app's handoff bridge set this frame to the
/// camera's render entity (edge-triggered + non-identity, so a steady identity never
/// touches the render world). The render-world copy is consumed by the rt_pipeline and
/// cleared next frame by [`clear_camera_reframe`].
pub fn extract_camera_reframe(
    cameras: Extract<Query<(RenderEntity, Ref<CameraReframe>), With<SolariCamera>>>,
    mut commands: Commands,
) {
    for (e, reframe) in &cameras {
        if reframe.is_changed() && !reframe.is_identity() {
            commands.entity(e).insert(*reframe);
        }
    }
}