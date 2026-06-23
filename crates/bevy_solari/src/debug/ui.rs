//! In-engine render-debug dropdown for raster cameras (bevy's
//! [`RenderDebugOverlay`]: depth, normals, deferred channels, depth pyramid).
//!
//! One global Feathers dropdown per camera that opts in by carrying a
//! [`RenderDebugOverlay`] component — typically the non-Solari raster camera.

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

/// Spawn one bottom-right render-debug dropdown per camera carrying a
/// [`RenderDebugOverlay`], targeted at that camera.
pub fn spawn_render_debug_panels(
    cameras: Query<Entity, (With<RenderDebugOverlay>, Without<RenderDebugPanelSpawned>)>,
    mut commands: Commands,
) {
    for camera in &cameras {
        commands
            .spawn((
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

// --- DLSS Ray Reconstruction mode dropdown (per `SolariCamera`) ----------------
// Unlike the render-debug overlay (a per-camera component), the DLSS mode is a single
// global resource, so every menu item just sets `SolariDlssMode`.

#[cfg(feature = "dlss")]
pub use dlss_dropdown::{spawn_dlss_panels, update_dlss_label};

#[cfg(feature = "dlss")]
mod dlss_dropdown {
    use super::*;
    use crate::render::dlss::SolariDlssMode;
    use crate::render::SolariCamera;
    use bevy_ecs::system::{Res, ResMut};

    /// Marker on a camera that already has a DLSS dropdown.
    #[derive(Component)]
    pub struct DlssPanelSpawned;

    /// Marker on the DLSS button caption.
    #[derive(Component, Default, Clone)]
    pub struct DlssLabel;

    /// One DLSS menu item: activating it sets the global [`SolariDlssMode`].
    fn dlss_item(mode: SolariDlssMode) -> impl Scene {
        bsn! {
            @FeathersMenuItem {
                @caption: bsn! { Text({mode.label().to_string()}) ThemedText }
            }
            on(move |_: On<Activate>, mut dlss_mode: ResMut<SolariDlssMode>| {
                *dlss_mode = mode;
            })
        }
    }

    /// Spawn one bottom-left DLSS dropdown per [`SolariCamera`] (the render-debug
    /// dropdown sits bottom-right and targets raster cameras, so they don't collide).
    pub fn spawn_dlss_panels(
        cameras: Query<Entity, (With<SolariCamera>, Without<DlssPanelSpawned>)>,
        mut commands: Commands,
    ) {
        for camera in &cameras {
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
                                    @caption: bsn! { Text("dlss: off") ThemedText DlssLabel }
                                }
                            ),
                            (
                                @FeathersMenuPopup
                                Children [
                                    dlss_item(SolariDlssMode::Off),
                                    dlss_item(SolariDlssMode::Auto),
                                    dlss_item(SolariDlssMode::Dlaa),
                                    dlss_item(SolariDlssMode::Quality),
                                    dlss_item(SolariDlssMode::Balanced),
                                    dlss_item(SolariDlssMode::Performance),
                                    dlss_item(SolariDlssMode::UltraPerformance),
                                ]
                            )
                        ]
                    )
                });
            commands.entity(camera).insert(DlssPanelSpawned);
        }
    }

    /// Keep the DLSS button caption in sync with the global mode resource.
    pub fn update_dlss_label(mode: Res<SolariDlssMode>, mut labels: Query<&mut Text, With<DlssLabel>>) {
        for mut text in &mut labels {
            let want = format!("dlss: {}", *mode);
            if text.0 != want {
                text.0 = want;
            }
        }
    }
}
