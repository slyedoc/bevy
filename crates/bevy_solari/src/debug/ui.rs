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

// --- "view" dropdown (normal / time heatmap / displacement) + heatmap sliders --
// A bottom-left "view" dropdown selects normal rendering, the per-pixel cost
// heatmap (`SolariCostHeatmap.enabled`), or the displacement-map debug view
// (`SolariShowDisplacement.enabled`). In heatmap view the two sliders (center,
// contrast) appear and the DLSS dropdown hides; otherwise it's the reverse.

pub use view_panel::{spawn_view_panels, toggle_heatmap_controls, update_view_label};

mod view_panel {
    use super::*;
    use crate::render::rt_pipeline::{SolariCostHeatmap, SolariShowDisplacement};
    use crate::render::SolariCamera;
    use bevy_ecs::system::{Res, ResMut};
    use bevy_feathers::controls::FeathersSlider;
    use bevy_feathers::theme::ThemeBackgroundColor;
    use bevy_feathers::tokens::WINDOW_BG;
    use bevy_ui::{Display, FlexDirection, UiRect};
    use bevy_ui_widgets::{slider_self_update, SliderPrecision, ValueChange};

    /// Marker on a camera that already has a view panel.
    #[derive(Component)]
    pub struct ViewPanelSpawned;

    /// Marker on the view button caption.
    #[derive(Component, Default, Clone)]
    pub struct ViewLabel;

    /// Marker on the container holding the heatmap sliders (shown only in heatmap view).
    #[derive(Component, Default, Clone)]
    pub struct HeatmapControls;

    /// One view menu item: selects the active debug view by setting the two view toggles (they're
    /// mutually exclusive — `normal` clears both, `heatmap`/`displacement` set exactly one).
    fn view_item(heatmap_on: bool, displacement_on: bool, label: &'static str) -> impl Scene {
        bsn! {
            @FeathersMenuItem {
                @caption: bsn! { Text({label.to_string()}) ThemedText }
            }
            on(move |_: On<Activate>,
                     mut heatmap: ResMut<SolariCostHeatmap>,
                     mut displacement: ResMut<SolariShowDisplacement>| {
                heatmap.enabled = heatmap_on;
                displacement.enabled = displacement_on;
            })
        }
    }

    /// Spawn one bottom-left view dropdown + heatmap sliders per [`SolariCamera`]
    /// (sits above the DLSS dropdown at `bottom: 8`).
    pub fn spawn_view_panels(
        cameras: Query<Entity, (With<SolariCamera>, Without<ViewPanelSpawned>)>,
        mut commands: Commands,
    ) {
        for camera in &cameras {
            commands
                .spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        bottom: px(48),
                        left: px(8),
                        flex_direction: FlexDirection::Column,
                        row_gap: px(4),
                        padding: UiRect::all(px(8)),
                        ..Default::default()
                    },
                    // Card background (the Feathers "window" surface token), so the
                    // dropdown + sliders read as one panel instead of floating bare.
                    ThemeBackgroundColor(WINDOW_BG),
                    TabGroup::default(),
                    UiTargetCamera(camera),
                ))
                .queue_spawn_related_scenes::<Children>(bsn_list! {
                    (
                        @FeathersMenu
                        Children [
                            (
                                @FeathersMenuButton {
                                    @caption: bsn! { Text("view: normal") ThemedText ViewLabel }
                                }
                            ),
                            (
                                @FeathersMenuPopup
                                Children [
                                    view_item(false, false, "normal"),
                                    view_item(true, false, "time heatmap"),
                                    view_item(false, true, "displacement"),
                                ]
                            )
                        ]
                    ),
                    (
                        Node {
                            display: Display::None,
                            flex_direction: FlexDirection::Column,
                            row_gap: px(2),
                        }
                        HeatmapControls
                        Children [
                            (Text("center") ThemedText),
                            (
                                @FeathersSlider { @min: 10.0, @max: 24.0, @value: 16.0 }
                                SliderPrecision(1)
                                on(slider_self_update)
                                on(|c: On<ValueChange<f32>>, mut heatmap: ResMut<SolariCostHeatmap>| {
                                    heatmap.center = c.value;
                                })
                            ),
                            (Text("contrast") ThemedText),
                            (
                                @FeathersSlider { @min: 0.0, @max: 1.0, @value: 0.15 }
                                SliderPrecision(2)
                                on(slider_self_update)
                                on(|c: On<ValueChange<f32>>, mut heatmap: ResMut<SolariCostHeatmap>| {
                                    heatmap.contrast = c.value;
                                })
                            ),
                        ]
                    )
                });
            commands.entity(camera).insert(ViewPanelSpawned);
        }
    }

    /// Keep the view button caption in sync with the active view toggle.
    pub fn update_view_label(
        heatmap: Res<SolariCostHeatmap>,
        displacement: Res<SolariShowDisplacement>,
        mut labels: Query<&mut Text, With<ViewLabel>>,
    ) {
        let want = if heatmap.enabled {
            "view: time heatmap"
        } else if displacement.enabled {
            "view: displacement"
        } else {
            "view: normal"
        };
        for mut text in &mut labels {
            if text.0 != want {
                text.0 = want.to_string();
            }
        }
    }

    /// Show the heatmap sliders only in heatmap view.
    pub fn toggle_heatmap_controls(
        heatmap: Res<SolariCostHeatmap>,
        mut controls: Query<&mut Node, With<HeatmapControls>>,
    ) {
        let want = if heatmap.enabled {
            Display::Flex
        } else {
            Display::None
        };
        for mut node in &mut controls {
            if node.display != want {
                node.display = want;
            }
        }
    }
}

// --- DLSS Ray Reconstruction mode dropdown (per `SolariCamera`) ----------------
// Unlike the render-debug overlay (a per-camera component), the DLSS mode is a single
// global resource, so every menu item just sets `SolariDlssMode`.

#[cfg(feature = "dlss")]
pub use dlss_dropdown::{spawn_dlss_panels, toggle_dlss_visibility, update_dlss_label};

#[cfg(feature = "dlss")]
mod dlss_dropdown {
    use super::*;
    use crate::render::dlss::SolariDlssMode;
    use crate::render::rt_pipeline::SolariCostHeatmap;
    use crate::render::SolariCamera;
    use bevy_ecs::system::{Res, ResMut};
    use bevy_ui::Display;

    /// Marker on a camera that already has a DLSS dropdown.
    #[derive(Component)]
    pub struct DlssPanelSpawned;

    /// Marker on the DLSS dropdown root node, so it can be hidden in heatmap view.
    #[derive(Component)]
    pub struct DlssPanelRoot;

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
                    DlssPanelRoot,
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

    /// Hide the DLSS dropdown in heatmap view — DLSS would denoise the heatmap, and the
    /// view panel's sliders take its place.
    pub fn toggle_dlss_visibility(
        heatmap: Res<SolariCostHeatmap>,
        mut roots: Query<&mut Node, With<DlssPanelRoot>>,
    ) {
        let want = if heatmap.enabled {
            Display::None
        } else {
            Display::Flex
        };
        for mut node in &mut roots {
            if node.display != want {
                node.display = want;
            }
        }
    }
}
