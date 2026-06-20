//! In-engine selector for [`SolariViewState`]: lighting (pathtrace / restir)
//! and the debug views.
//!
//! One global Feathers dropdown, anchored to the first [`SolariCamera`]'s
//! viewport via [`UiTargetCamera`] and wrapped in a [`TabGroup`] so it
//! participates in Feathers' tab navigation (no more "no tab groups" warning,
//! and no raw `Tab` hotkey fighting the focus system). The button caption shows
//! the current state; picking an item writes [`SolariViewState`].

#[cfg(feature = "dlss")]
use crate::render::SolariDlssMode;
use crate::render::{
    view::{SolariDebugView, SolariLighting, SolariViewState},
    SolariCamera,
};
use bevy_dev_tools::render_debug::{RenderDebugMode, RenderDebugOverlay};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    hierarchy::Children,
    observer::On,
    query::{With, Without},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_feathers::{
    controls::{FeathersMenu, FeathersMenuButton, FeathersMenuItem, FeathersMenuPopup},
    theme::ThemedText,
};
use bevy_input_focus::tab_navigation::TabGroup;
use bevy_scene::prelude::*;
use bevy_ui::{px, widget::Text, Node, PositionType, UiTargetCamera};
use bevy_ui_widgets::Activate;

/// Marker resource: the (single, global) debug dropdown exists, so
/// [`spawn_debug_panels`] never spawns a second one.
#[derive(Resource)]
pub struct DebugPanelSpawned;

/// Marker on the integrator menu button's caption text; [`update_view_label`]
/// keeps it in sync with [`SolariViewState::lighting`].
#[derive(Component, Default, Clone, Copy)]
pub struct ViewLabel;

/// Marker on the debug-view menu button's caption text; [`update_view_label`]
/// keeps it in sync with [`SolariViewState::debug`].
#[derive(Component, Default, Clone, Copy)]
pub struct DebugViewLabel;

/// A lighting menu item: activating it selects `lighting` and clears any debug
/// view.
fn lighting_item(lighting: SolariLighting) -> impl Scene {
    bsn! {
        @FeathersMenuItem {
            @caption: bsn! { Text({lighting.to_string()}) ThemedText }
        }
        on(move |_: On<Activate>, mut state: ResMut<SolariViewState>| {
            state.lighting = lighting;
            state.debug = None;
        })
    }
}

/// A debug-view menu item: activating it overlays `view` (lighting selection is
/// kept for when the view is cleared).
fn debug_item(view: SolariDebugView) -> impl Scene {
    bsn! {
        @FeathersMenuItem {
            @caption: bsn! { Text({view.to_string()}) ThemedText }
        }
        on(move |_: On<Activate>, mut state: ResMut<SolariViewState>| {
            state.debug = Some(view);
        })
    }
}

/// Spawn the global bottom-left dropdown once the first [`SolariCamera`]
/// exists, anchored to that camera's viewport. Runs every frame; the
/// [`DebugPanelSpawned`] resource keeps it idempotent.
pub fn spawn_debug_panels(
    spawned: Option<Res<DebugPanelSpawned>>,
    cameras: Query<Entity, With<SolariCamera>>,
    mut commands: Commands,
) {
    if spawned.is_some() {
        return;
    }
    let Some(camera) = cameras.iter().next() else {
        return;
    };
    commands.insert_resource(DebugPanelSpawned);
    // The root carries `UiTargetCamera` in its spawn bundle (so it propagates
    // to the whole dropdown and confines it to this camera's viewport); the
    // menu is attached as a child scene.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: px(8),
                left: px(8),
                column_gap: px(8),
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
                            @caption: bsn! { Text("pathtrace") ThemedText ViewLabel }
                        }
                    ),
                    (
                        @FeathersMenuPopup
                        Children [
                            lighting_item(SolariLighting::Pathtracer),
                            lighting_item(SolariLighting::Restir),
                            lighting_item(SolariLighting::RtPipeline),
                        ]
                    )
                ]
            ),
            (
                @FeathersMenu
                Children [
                    (
                        @FeathersMenuButton {
                            @caption: bsn! { Text("debug: none") ThemedText DebugViewLabel }
                        }
                    ),
                    (
                        @FeathersMenuPopup
                        DebugMenuPopup
                        Children [
                            (
                                @FeathersMenuItem {
                                    @caption: bsn! { Text("none") ThemedText }
                                }
                                on(|_: On<Activate>, mut state: ResMut<SolariViewState>| {
                                    state.debug = None;
                                })
                            ),
                            debug_item(SolariDebugView::Lod),
                            debug_item(SolariDebugView::Cluster),
                            debug_item(SolariDebugView::Triangle),
                            debug_item(SolariDebugView::GeometryCheck),
                            debug_item(SolariDebugView::MaterialId),
                            debug_item(SolariDebugView::Uv),
                            debug_item(SolariDebugView::MotionVectors),
                            debug_item(SolariDebugView::DiWeight),
                            debug_item(SolariDebugView::DiConfidence),
                            debug_item(SolariDebugView::DiLight),
                            debug_item(SolariDebugView::RegirCells),
                        ]
                    )
                ]
            )
        });
}

/// Marks the debug-view menu's popup, so the DLSS guide views can be appended
/// as separate children (a `bsn_list!` item can't carry a `#[cfg]`, and the
/// guide views only exist under the `dlss` feature).
#[derive(Component, Default, Clone)]
pub struct DebugMenuPopup;

/// Append the DLSS guide-buffer views to the debug-view dropdown once its
/// popup exists.
#[cfg(feature = "dlss")]
pub fn append_dlss_guide_items(
    popup: Query<Entity, bevy_ecs::prelude::Added<DebugMenuPopup>>,
    mut commands: Commands,
) {
    for entity in &popup {
        commands
            .entity(entity)
            .queue_spawn_related_scenes::<Children>(bsn_list! {
                debug_item(SolariDebugView::DlssDepth),
                debug_item(SolariDebugView::DlssNormalRoughness),
                debug_item(SolariDebugView::DlssDiffuseAlbedo),
                debug_item(SolariDebugView::DlssSpecularAlbedo),
                debug_item(SolariDebugView::DlssSpecularMotion),
            });
    }
}

/// Keep both menu captions in sync with [`SolariViewState`]: the integrator
/// button shows the lighting selection, the debug button the active view.
pub fn update_view_label(
    state: Res<SolariViewState>,
    mut labels: Query<&mut Text, With<ViewLabel>>,
    mut debug_labels: Query<&mut Text, (With<DebugViewLabel>, Without<ViewLabel>)>,
) {
    let want = state.lighting.to_string();
    for mut text in &mut labels {
        if text.0 != want {
            text.0 = want.clone();
        }
    }
    let want = match state.debug {
        Some(view) => format!("debug: {view}"),
        None => "debug: none".to_string(),
    };
    for mut text in &mut debug_labels {
        if text.0 != want {
            text.0 = want.clone();
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

// ── DLSS quality dropdown ────────────────────────────────────────────────────
//
// Present only when DLSS Ray Reconstruction is active (the `SolariDlssMode`
// resource exists). Same shape as the view dropdown, stacked above it.

/// Marker resource: the DLSS dropdown exists.
#[cfg(feature = "dlss")]
#[derive(Resource)]
pub struct DlssPanelSpawned;

/// On the DLSS menu button's caption text; [`update_dlss_label`] keeps it in
/// sync with [`SolariDlssMode`].
#[cfg(feature = "dlss")]
#[derive(Component, Default, Clone, Copy)]
pub struct DlssLabel;

/// One DLSS menu item: activating it selects `mode` (the per-view RR context
/// is recreated at the mode's render resolution).
#[cfg(feature = "dlss")]
fn dlss_item(mode: SolariDlssMode) -> impl Scene {
    bsn! {
        @FeathersMenuItem {
            @caption: bsn! { Text({mode.to_string()}) ThemedText }
        }
        on(move |_: On<Activate>, mut current: ResMut<SolariDlssMode>| {
            *current = mode;
        })
    }
}

/// Spawn the DLSS quality dropdown once DLSS is active and a [`SolariCamera`]
/// exists, just above the view dropdown.
#[cfg(feature = "dlss")]
pub fn spawn_dlss_panel(
    mode: Option<Res<SolariDlssMode>>,
    spawned: Option<Res<DlssPanelSpawned>>,
    cameras: Query<Entity, With<SolariCamera>>,
    mut commands: Commands,
) {
    if mode.is_none() || spawned.is_some() {
        return;
    }
    let Some(camera) = cameras.iter().next() else {
        return;
    };
    commands.insert_resource(DlssPanelSpawned);
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: px(48),
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
                            @caption: bsn! { Text("dlss: dlaa") ThemedText DlssLabel }
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
}

/// Keep the DLSS button's caption in sync with the selected mode.
#[cfg(feature = "dlss")]
pub fn update_dlss_label(
    mode: Option<Res<SolariDlssMode>>,
    mut labels: Query<&mut Text, With<DlssLabel>>,
) {
    let Some(mode) = mode else {
        return;
    };
    let want = format!("dlss: {}", *mode);
    for mut text in &mut labels {
        if text.0 != want {
            text.0 = want.clone();
        }
    }
}
