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

pub use view_panel::{
    spawn_view_panels, toggle_heatmap_controls, update_stats_label, update_view_label,
    ViewPanelRoot,
};

mod view_panel {
    use super::*;
    use crate::render::rt_pipeline::{
        SolariAnyHitHeatmap, SolariClusterView, SolariCostHeatmap, SolariShowDisplacement,
        SolariTriangleView,
    };
    use crate::render::SolariCamera;
    use bevy_ecs::system::{Res, ResMut};
    use bevy_feathers::controls::FeathersSlider;
    use bevy_feathers::theme::ThemeBackgroundColor;
    use bevy_feathers::tokens::WINDOW_BG;
    use bevy_feathers::constants::{fonts, size};
    use bevy_feathers::theme::ThemeTextColor;
    use bevy_feathers::tokens;
    use bevy_text::{FontSourceTemplate, FontWeight, LineBreak, TextFont, TextLayout};
    use bevy_ui::{Display, FlexDirection, UiRect};
    use bevy_ui_widgets::{slider_self_update, SliderPrecision, ValueChange};

    /// Marker on a camera that already has a view panel.
    #[derive(Component)]
    pub struct ViewPanelSpawned;

    /// Marker on the view button caption.
    #[derive(Component, Default, Clone)]
    pub struct ViewLabel;

    /// Marker on the view panel card root (the shared debug card — stats,
    /// view dropdown, heatmap sliders, and the DLSS dropdown all live in it).
    #[derive(Component, Default, Clone)]
    pub struct ViewPanelRoot;

    /// Marker on the live transform-count line.
    #[derive(Component, Default, Clone)]
    pub struct StatsTransformsLabel;

    /// Marker on the live RT-instance-count line.
    #[derive(Component, Default, Clone)]
    pub struct StatsRtLabel;

    /// Marker on the container holding the heatmap sliders (shown only in heatmap view).
    #[derive(Component, Default, Clone)]
    pub struct HeatmapControls;

    /// One view menu item: selects the active debug view by setting the view toggles (mutually
    /// exclusive — `normal` clears all, every other item sets exactly one).
    fn view_item(
        heatmap_on: bool,
        anyhit_on: bool,
        displacement_on: bool,
        cluster_on: bool,
        triangle_on: bool,
        label: &'static str,
    ) -> impl Scene {
        bsn! {
            @FeathersMenuItem {
                @caption: bsn! { Text({label.to_string()}) ThemedText }
            }
            on(move |_: On<Activate>,
                     mut heatmap: ResMut<SolariCostHeatmap>,
                     mut anyhit: ResMut<SolariAnyHitHeatmap>,
                     mut displacement: ResMut<SolariShowDisplacement>,
                     mut cluster: ResMut<SolariClusterView>,
                     mut triangle: ResMut<SolariTriangleView>| {
                heatmap.enabled = heatmap_on;
                anyhit.enabled = anyhit_on;
                displacement.enabled = displacement_on;
                cluster.enabled = cluster_on;
                triangle.enabled = triangle_on;
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
                    ViewPanelRoot,
                ))
                .queue_spawn_related_scenes::<Children>(bsn_list! {
                    (
                        Text("")
                        TextFont {
                            font: FontSourceTemplate::Handle(fonts::REGULAR),
                            font_size: size::EXTRA_SMALL_FONT,
                            weight: FontWeight::NORMAL,
                        }
                        TextLayout { linebreak: LineBreak::NoWrap }
                        ThemeTextColor(tokens::TEXT_DIM)
                        StatsTransformsLabel
                    ),
                    (
                        Text("")
                        TextFont {
                            font: FontSourceTemplate::Handle(fonts::REGULAR),
                            font_size: size::EXTRA_SMALL_FONT,
                            weight: FontWeight::NORMAL,
                        }
                        TextLayout { linebreak: LineBreak::NoWrap }
                        ThemeTextColor(tokens::TEXT_DIM)
                        StatsRtLabel
                    ),
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
                                    view_item(false, false, false, false, false, "normal"),
                                    view_item(true, false, false, false, false, "time heatmap"),
                                    view_item(false, true, false, false, false, "any-hit count"),
                                    view_item(false, false, true, false, false, "displacement"),
                                    view_item(false, false, false, true, false, "clusters"),
                                    view_item(false, false, false, false, true, "triangles"),
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
        anyhit: Res<SolariAnyHitHeatmap>,
        displacement: Res<SolariShowDisplacement>,
        cluster: Res<SolariClusterView>,
        triangle: Res<SolariTriangleView>,
        mut labels: Query<&mut Text, With<ViewLabel>>,
    ) {
        let want = if heatmap.enabled {
            "view: time heatmap"
        } else if anyhit.enabled {
            "view: any-hit count"
        } else if displacement.enabled {
            "view: displacement"
        } else if cluster.enabled {
            "view: clusters"
        } else if triangle.enabled {
            "view: triangles"
        } else {
            "view: normal"
        };
        for mut text in &mut labels {
            if text.0 != want {
                text.0 = want.to_string();
            }
        }
    }

    /// Live scene stats above the view dropdown: transform-table nodes and
    /// main-world RT mesh entities (thousands-grouped), one line each.
    pub fn update_stats_label(
        transforms: Res<crate::ecs_gpu::GpuSlotAllocator<crate::transform::TransformGraph>>,
        rt: Query<(), With<crate::bindings::RaytracingMesh3d>>,
        mut tf_labels: Query<&mut Text, (With<StatsTransformsLabel>, Without<StatsRtLabel>)>,
        mut rt_labels: Query<&mut Text, With<StatsRtLabel>>,
    ) {
        fn k(n: u32) -> String {
            match n {
                0..=9_999 => n.to_string(),
                10_000..=999_999 => format!("{:.1}k", n as f64 / 1_000.0),
                _ => format!("{:.2}M", n as f64 / 1_000_000.0),
            }
        }
        let tf = format!("transforms {}", k(transforms.live()));
        let rt_count = format!("rt {}", k(rt.iter().len() as u32));
        for mut text in &mut tf_labels {
            if text.0 != tf {
                text.0 = tf.clone();
            }
        }
        for mut text in &mut rt_labels {
            if text.0 != rt_count {
                text.0 = rt_count.clone();
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

    /// Marker on the DLSS dropdown row (inside the view card), hidden in heatmap view.
    #[derive(Component, Default, Clone)]
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

    /// Add the DLSS dropdown INTO each camera's view-panel card (below the view
    /// dropdown / sliders), so the debug widgets read as one panel.
    pub fn spawn_dlss_panels(
        cameras: Query<(), (With<SolariCamera>, Without<DlssPanelSpawned>)>,
        panels: Query<(Entity, &UiTargetCamera), With<ViewPanelRoot>>,
        mut commands: Commands,
    ) {
        for (panel, target) in &panels {
            if cameras.get(target.0).is_err() {
                continue; // camera already has its dropdown (or isn't solari)
            }
            commands
                .entity(panel)
                .queue_spawn_related_scenes::<Children>(bsn_list! {
                    (
                        Node {}
                        DlssPanelRoot
                        Children [
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
                        ]
                    )
                });
            commands.entity(target.0).insert(DlssPanelSpawned);
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
