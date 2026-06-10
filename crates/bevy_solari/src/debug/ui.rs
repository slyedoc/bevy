//! In-engine debug-view selector for [`SolariOverlay`].
//!
//! One Feathers **dropdown menu per Solari camera**, each targeted at its own
//! camera with [`UiTargetCamera`] so it sits in that camera's viewport
//! (split-screen / multi-view friendly), and wrapped in a [`TabGroup`] so it
//! participates in Feathers' tab navigation (no more "no tab groups" warning,
//! and no raw `Tab` hotkey fighting the focus system). The button caption shows
//! the camera's current view; picking an item sets that camera's
//! [`SolariOverlay`].

use crate::render::{view::SolariOverlay, SolariCamera};
use bevy_dev_tools::render_debug::{RenderDebugMode, RenderDebugOverlay};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    hierarchy::Children,
    observer::On,
    query::{With, Without},
    system::{Commands, Query},
};
use bevy_feathers::{
    controls::{FeathersMenu, FeathersMenuButton, FeathersMenuItem, FeathersMenuPopup},
    theme::ThemedText,
};
use bevy_input_focus::tab_navigation::TabGroup;
use bevy_scene::prelude::*;
use bevy_ui::{px, widget::Text, Node, PositionType, UiTargetCamera};
use bevy_ui_widgets::Activate;

/// Marker on a Solari camera that already has a debug dropdown, so
/// [`spawn_debug_panels`] never spawns a second one for it.
#[derive(Component)]
pub struct DebugPanelSpawned;

/// On the menu button's caption text, bound to the `camera` whose
/// [`SolariOverlay`] it shows. [`update_view_label`] keeps it in sync.
/// `Default` (placeholder) is required so it can be a `bsn!` patch; the real
/// camera is supplied via `ViewLabel({camera})`.
#[derive(Component, Clone, Copy)]
pub struct ViewLabel(pub Entity);

impl Default for ViewLabel {
    fn default() -> Self {
        ViewLabel(Entity::PLACEHOLDER)
    }
}

/// One dropdown menu item: its caption is the view's name, and activating it
/// sets `camera`'s [`SolariOverlay`] to `view`.
fn view_item(camera: Entity, view: SolariOverlay) -> impl Scene {
    bsn! {
        @FeathersMenuItem {
            @caption: bsn! { Text({view.to_string()}) ThemedText }
        }
        on(move |_: On<Activate>, mut overlays: Query<&mut SolariOverlay>| {
            if let Ok(mut overlay) = overlays.get_mut(camera) {
                *overlay = view;
            }
        })
    }
}

/// Spawn one bottom-left dropdown per [`SolariCamera`], targeted at that camera.
/// Runs every frame; the [`DebugPanelSpawned`] marker keeps it idempotent and
/// lets it pick up cameras spawned after startup.
pub fn spawn_debug_panels(
    cameras: Query<Entity, (With<SolariCamera>, Without<DebugPanelSpawned>)>,
    mut commands: Commands,
) {
    for camera in &cameras {
        // The root carries `UiTargetCamera` in its spawn bundle (so it propagates
        // to the whole dropdown and confines it to this camera's viewport); the
        // menu is attached as a child scene.
        commands
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    bottom: px(8),
                    left: px(8),
                    ..Default::default()
                },
                TabGroup::default(),
                UiTargetCamera(camera),
            ))
            .queue_spawn_related_scenes::<Children>(bsn_list! {
                (
                    @FeathersMenu
                    Children [
                        (
                            @FeathersMenuButton {
                                @caption: bsn! { Text("pathtrace") ThemedText ViewLabel({camera}) }
                            }
                        ),
                        (
                            @FeathersMenuPopup
                            Children [
                                view_item(camera, SolariOverlay::None),
                                view_item(camera, SolariOverlay::Pathtrace),
                                view_item(camera, SolariOverlay::Lod),
                                view_item(camera, SolariOverlay::Cluster),
                                view_item(camera, SolariOverlay::Triangle),
                                view_item(camera, SolariOverlay::GeometryCheck),
                                view_item(camera, SolariOverlay::WorldPosition),
                                view_item(camera, SolariOverlay::MaterialId),
                                view_item(camera, SolariOverlay::WorldNormal),
                                view_item(camera, SolariOverlay::Uv),
                                view_item(camera, SolariOverlay::MotionVectors),
                            ]
                        )
                    ]
                )
            });
        commands.entity(camera).insert(DebugPanelSpawned);
    }
}

/// Keep each dropdown button's caption in sync with its camera's selected view.
pub fn update_view_label(views: Query<&SolariOverlay>, mut labels: Query<(&mut Text, &ViewLabel)>) {
    for (mut text, label) in &mut labels {
        let Ok(view) = views.get(label.0) else {
            continue;
        };
        let want = view.to_string();
        if text.0 != want {
            text.0 = want;
        }
    }
}

// ── Bevy render debug overlay (raster cameras) ──────────────────────────────
//
// The same dropdown shape, but for bevy's `RenderDebugOverlay` (depth, normals,
// deferred channels, depth pyramid). Spawned for any camera that opts in by
// carrying a `RenderDebugOverlay` component — typically the non-Solari raster
// camera (which also needs `DepthPrepass` / `OcclusionCulling` for those modes).

/// Marker on a camera that already has a render-debug dropdown.
#[derive(Component)]
pub struct RenderDebugPanelSpawned;

/// On the render-debug button caption; bound to the camera whose
/// [`RenderDebugOverlay`] it shows.
#[derive(Component, Clone, Copy)]
pub struct RenderDebugLabel(pub Entity);

impl Default for RenderDebugLabel {
    fn default() -> Self {
        RenderDebugLabel(Entity::PLACEHOLDER)
    }
}

/// Short label for a render-debug mode.
fn render_debug_mode_name(mode: RenderDebugMode) -> &'static str {
    match mode {
        RenderDebugMode::Depth => "depth",
        RenderDebugMode::Normal => "normal",
        RenderDebugMode::MotionVectors => "motion vectors",
        RenderDebugMode::Deferred => "deferred",
        RenderDebugMode::DeferredBaseColor => "deferred base color",
        RenderDebugMode::DeferredEmissive => "deferred emissive",
        RenderDebugMode::DeferredMetallicRoughness => "deferred metallic+roughness",
        RenderDebugMode::DepthPyramid { .. } => "depth pyramid",
    }
}

/// One render-debug menu item: activating it sets `camera`'s
/// [`RenderDebugOverlay`] (`enabled` + `mode`).
fn render_debug_item(
    camera: Entity,
    label: &'static str,
    enabled: bool,
    mode: RenderDebugMode,
) -> impl Scene {
    bsn! {
        @FeathersMenuItem {
            @caption: bsn! { Text({label.to_string()}) ThemedText }
        }
        on(move |_: On<Activate>, mut overlays: Query<&mut RenderDebugOverlay>| {
            if let Ok(mut overlay) = overlays.get_mut(camera) {
                overlay.enabled = enabled;
                overlay.mode = mode;
            }
        })
    }
}

/// Spawn one bottom-left render-debug dropdown per camera carrying a
/// [`RenderDebugOverlay`], targeted at that camera.
pub fn spawn_render_debug_panels(
    cameras: Query<Entity, (With<RenderDebugOverlay>, Without<RenderDebugPanelSpawned>)>,
    mut commands: Commands,
) {
    for camera in &cameras {
        commands
            .spawn((
                // Bottom-RIGHT (the Solari dropdown is bottom-left) so the two
                // never overlap — the Feathers menu currently isn't honoring
                // `UiTargetCamera` per-viewport, so both can land on the same
                // camera; opposite corners keep them readable regardless.
                Node {
                    position_type: PositionType::Absolute,
                    bottom: px(8),
                    right: px(8),
                    ..Default::default()
                },
                TabGroup::default(),
                UiTargetCamera(camera),
            ))
            .queue_spawn_related_scenes::<Children>(bsn_list! {
                (
                    @FeathersMenu
                    Children [
                        (
                            @FeathersMenuButton {
                                @caption: bsn! { Text("render: off") ThemedText RenderDebugLabel({camera}) }
                            }
                        ),
                        (
                            @FeathersMenuPopup
                            Children [
                                render_debug_item(camera, "off", false, RenderDebugMode::Depth),
                                render_debug_item(camera, "depth", true, RenderDebugMode::Depth),
                                render_debug_item(camera, "normal", true, RenderDebugMode::Normal),
                                render_debug_item(camera, "motion vectors", true, RenderDebugMode::MotionVectors),
                                render_debug_item(camera, "deferred", true, RenderDebugMode::Deferred),
                                render_debug_item(camera, "deferred base color", true, RenderDebugMode::DeferredBaseColor),
                                render_debug_item(camera, "deferred emissive", true, RenderDebugMode::DeferredEmissive),
                                render_debug_item(camera, "deferred metallic+roughness", true, RenderDebugMode::DeferredMetallicRoughness),
                                render_debug_item(camera, "depth pyramid", true, RenderDebugMode::DepthPyramid { mip_level: 0 }),
                            ]
                        )
                    ]
                )
            });
        commands.entity(camera).insert(RenderDebugPanelSpawned);
    }
}

/// Keep each render-debug button caption in sync with its camera's overlay.
pub fn update_render_debug_label(
    overlays: Query<&RenderDebugOverlay>,
    mut labels: Query<(&mut Text, &RenderDebugLabel)>,
) {
    for (mut text, label) in &mut labels {
        let Ok(overlay) = overlays.get(label.0) else {
            continue;
        };
        let want = if overlay.enabled {
            format!("render: {}", render_debug_mode_name(overlay.mode))
        } else {
            "render: off".to_string()
        };
        if text.0 != want {
            text.0 = want;
        }
    }
}
