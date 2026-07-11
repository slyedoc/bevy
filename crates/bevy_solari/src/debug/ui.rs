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

// --- SolariCamera debug card: stats + derived name + component inspector ------
// One bottom-left card per camera. The estimator tree, debug-view enum, and
// session knobs are all edited through the reflection-driven inspector — the
// UI never enumerates levers by hand, so new fields appear automatically.

pub use view_panel::{spawn_view_panels, update_mode_label, update_stats_label, ViewPanelRoot};

mod view_panel {
    use super::*;
    use crate::render::SolariCamera;
    use bevy_ecs::system::Res;
    use bevy_feathers::constants::{fonts, size};
    use bevy_feathers::theme::{ThemeBackgroundColor, ThemeTextColor};
    use bevy_feathers::tokens;
    use bevy_feathers::tokens::WINDOW_BG;
    use bevy_feathers_inspector::BuildComponentInspector;
    use bevy_ecs::template::EntityTemplate;
    use bevy_feathers::controls::FeathersScrollbar;
    use bevy_text::{FontSourceTemplate, FontWeight, LineBreak, TextFont, TextLayout};
    use bevy_ui::{percent, FlexDirection, Overflow, UiRect};
    use bevy_ui_widgets::{ControlOrientation, ScrollArea};
    use core::any::TypeId;

    /// Marker on a camera that already has a view panel.
    #[derive(Component)]
    pub struct ViewPanelSpawned;

    /// Marker on the view panel card root (the shared debug card — stats, the
    /// derived mode name, the [`SolariCamera`] inspector, and the DLSS
    /// dropdown all live in it). External crates append their own rows.
    #[derive(Component, Default, Clone)]
    pub struct ViewPanelRoot;

    /// Marker on the live transform-count line.
    #[derive(Component, Default, Clone)]
    pub struct StatsTransformsLabel;

    /// Marker on the live RT-instance-count line.
    #[derive(Component, Default, Clone)]
    pub struct StatsRtLabel;

    /// On the derived-name line; bound to the camera whose
    /// [`SolariCamera::name`] it shows.
    #[derive(Component, Clone, Copy)]
    pub struct ModeNameLabel(pub Entity);

    impl Default for ModeNameLabel {
        fn default() -> Self {
            ModeNameLabel(Entity::PLACEHOLDER)
        }
    }

    /// Spawn one bottom-left debug card per [`SolariCamera`]: live stats, the
    /// derived mode name, and a reflection-driven inspector for the whole
    /// camera component — every estimator lever, the debug-view enum, and the
    /// session knobs are live-editable. Edits write back through reflection,
    /// so `Changed<SolariCamera>` fires and mode edits reset temporal history
    /// exactly like code/CLI changes.
    pub fn spawn_view_panels(
        cameras: Query<Entity, (With<SolariCamera>, Without<ViewPanelSpawned>)>,
        mut commands: Commands,
    ) {
        for camera in &cameras {
            // Listview shape: an outer frame holds the scrolling content node
            // and an absolute scrollbar (a bar INSIDE the scroller would
            // scroll away with the content). The inspector sections outgrow
            // the window, so the frame caps at 85% and the content wheels.
            let frame = commands
                .spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        bottom: px(48),
                        left: px(8),
                        width: px(340),
                        max_height: percent(85),
                        flex_direction: FlexDirection::Column,
                        padding: UiRect::all(px(8)),
                        ..Default::default()
                    },
                    // Card background (the Feathers "window" surface token), so
                    // the rows read as one panel instead of floating bare.
                    ThemeBackgroundColor(WINDOW_BG),
                    UiTargetCamera(camera),
                ))
                .id();
            let card = commands
                .spawn((
                    Node {
                        overflow: Overflow::scroll_y(),
                        flex_direction: FlexDirection::Column,
                        row_gap: px(4),
                        ..Default::default()
                    },
                    ScrollArea,
                    TabGroup::default(),
                    UiTargetCamera(camera),
                    ViewPanelRoot,
                ))
                .id();
            commands.entity(frame).add_child(card);
            commands.entity(frame).queue_spawn_related_scenes::<Children>(bsn_list! {
                (
                    @FeathersScrollbar {
                        @target: {EntityTemplate::from(card)},
                        @orientation: {ControlOrientation::Vertical}
                    }
                    Node {
                        position_type: PositionType::Absolute,
                        right: px(0),
                        top: px(0),
                        bottom: px(0),
                        width: px(6),
                    }
                )
            });
            commands.entity(card).queue_spawn_related_scenes::<Children>(bsn_list! {
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
                    Text("")
                    TextLayout { linebreak: LineBreak::NoWrap }
                    ThemedText
                    ModeNameLabel({camera})
                ),
            });
            // One inspector section per camera-owned component: the estimator
            // tree, then exposure (EV100 — the blit applies it at read, so it
            // never resets accumulation).
            for type_id in [
                TypeId::of::<SolariCamera>(),
                TypeId::of::<bevy_camera::Exposure>(),
            ] {
                let panel = commands
                    .spawn(Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: px(2),
                        ..Default::default()
                    })
                    .id();
                commands.entity(card).add_child(panel);
                commands.queue(BuildComponentInspector { target: camera, type_id, panel });
            }
            commands.entity(camera).insert(ViewPanelSpawned);
        }
    }

    /// Keep the derived-name line in sync with its camera — the same slug the
    /// grader and window titles use, computed from the live configuration.
    pub fn update_mode_label(
        cameras: Query<&SolariCamera>,
        mut labels: Query<(&mut Text, &ModeNameLabel)>,
    ) {
        for (mut text, label) in &mut labels {
            let Ok(camera) = cameras.get(label.0) else {
                continue;
            };
            let want = camera.name();
            if text.0 != want {
                text.0 = want;
            }
        }
    }

    /// Live scene stats above the inspector: transform-table nodes and
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
}

// --- DLSS Ray Reconstruction mode dropdown (per `SolariCamera`) ----------------
// Unlike the render-debug overlay (a per-camera component), the DLSS mode is a single
// global resource, so every menu item just sets `SolariDlssMode`.

pub use dlss_dropdown::{spawn_dlss_panels, toggle_dlss_visibility, update_dlss_label};

mod dlss_dropdown {
    use super::*;
    use crate::render::dlss::SolariDlssMode;
    use crate::render::rt_pipeline::SolariDebugView;
    use crate::render::SolariCamera;
    use bevy_ecs::system::{Res, ResMut};
    use bevy_ui::Display;

    /// Marker on a camera that already has a DLSS dropdown.
    #[derive(Component)]
    pub struct DlssPanelSpawned;

    /// On the DLSS dropdown row (inside the view card), hidden in heatmap view;
    /// bound to the camera whose [`SolariDebugView`] gates it.
    #[derive(Component, Clone, Copy)]
    pub struct DlssPanelRoot(pub Entity);

    impl Default for DlssPanelRoot {
        fn default() -> Self {
            DlssPanelRoot(Entity::PLACEHOLDER)
        }
    }

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
            let camera = target.0;
            commands
                .entity(panel)
                .queue_spawn_related_scenes::<Children>(bsn_list! {
                    (
                        Node {}
                        DlssPanelRoot({camera})
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

    /// Hide the DLSS dropdown while its camera is in heatmap view — DLSS would
    /// denoise the heatmap, and the view panel's sliders take its place.
    pub fn toggle_dlss_visibility(
        cameras: Query<&SolariCamera>,
        mut roots: Query<(&mut Node, &DlssPanelRoot)>,
    ) {
        for (mut node, bound) in &mut roots {
            let Ok(camera) = cameras.get(bound.0) else {
                continue;
            };
            let want = if matches!(camera.debug, SolariDebugView::CostHeatmap { .. }) {
                Display::None
            } else {
                Display::Flex
            };
            if node.display != want {
                node.display = want;
            }
        }
    }
}
