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

// --- "view" dropdown (SolariDebugView) + heatmap sliders ----------------------
// A bottom-left "view" dropdown per camera sets that camera's `SolariDebugView`
// component — normal rendering or one of the debug paints. In cost-heatmap view
// the two sliders (center, contrast) appear and the DLSS dropdown hides;
// otherwise it's the reverse.

pub use view_panel::{
    spawn_view_panels, sync_accumulate_checkbox, toggle_heatmap_controls, update_recipe_label,
    update_stats_label, update_view_label, ViewPanelRoot,
};

mod view_panel {
    use super::*;
    use crate::render::rt_pipeline::SolariDebugView;
    use crate::render::{SolariCamera, SolariLighting, SolariRecipe};
    use bevy_ecs::query::Has;
    use bevy_ecs::system::Res;
    use bevy_feathers::controls::FeathersSlider;
    use bevy_feathers::theme::ThemeBackgroundColor;
    use bevy_feathers::tokens::WINDOW_BG;
    use bevy_feathers::constants::{fonts, size};
    use bevy_feathers::theme::ThemeTextColor;
    use bevy_feathers::tokens;
    use bevy_text::{FontSourceTemplate, FontWeight, LineBreak, TextFont, TextLayout};
    use bevy_ui::{Display, FlexDirection, UiRect};
    use bevy_feathers::controls::FeathersCheckbox;
    use bevy_ui::Checked;
    use bevy_ui_widgets::{slider_self_update, SliderPrecision, ValueChange};

    /// Marker on a camera that already has a view panel.
    #[derive(Component)]
    pub struct ViewPanelSpawned;

    /// On the view button caption; bound to the camera whose
    /// [`SolariDebugView`] it shows.
    #[derive(Component, Clone, Copy)]
    pub struct ViewLabel(pub Entity);

    impl Default for ViewLabel {
        fn default() -> Self {
            ViewLabel(Entity::PLACEHOLDER)
        }
    }

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

    /// On the recipe button caption; bound to the camera whose
    /// [`SolariCamera::mode`] it names.
    #[derive(Component, Clone, Copy)]
    pub struct RecipeLabel(pub Entity);

    impl Default for RecipeLabel {
        fn default() -> Self {
            RecipeLabel(Entity::PLACEHOLDER)
        }
    }

    /// On the accumulate checkbox; bound to the camera whose reference mode it
    /// toggles. [`sync_accumulate_checkbox`] owns the `Checked` marker and the
    /// row's visibility (reference modes only).
    #[derive(Component, Clone, Copy)]
    pub struct AccumulateCheckbox(pub Entity);

    impl Default for AccumulateCheckbox {
        fn default() -> Self {
            AccumulateCheckbox(Entity::PLACEHOLDER)
        }
    }

    /// On the container holding the heatmap sliders (shown only in heatmap
    /// view); bound to the camera whose [`SolariDebugView`] gates it.
    #[derive(Component, Clone, Copy)]
    pub struct HeatmapControls(pub Entity);

    impl Default for HeatmapControls {
        fn default() -> Self {
            HeatmapControls(Entity::PLACEHOLDER)
        }
    }

    /// One view menu item: activating it sets `camera`'s
    /// [`SolariCamera::debug`] (a single enum, so views are mutually
    /// exclusive by construction).
    fn view_item(camera: Entity, view: SolariDebugView, label: &'static str) -> impl Scene {
        bsn! {
            @FeathersMenuItem {
                @caption: bsn! { Text({label.to_string()}) ThemedText }
            }
            on(move |_: On<Activate>, mut cameras: Query<&mut SolariCamera>| {
                if let Ok(mut solari) = cameras.get_mut(camera) {
                    solari.debug = view;
                }
            })
        }
    }

    /// One recipe menu item: activating it stamps the recipe's mode onto
    /// `camera` (the debug view is untouched).
    fn recipe_item(camera: Entity, recipe: SolariRecipe) -> impl Scene {
        bsn! {
            @FeathersMenuItem {
                @caption: bsn! { Text({recipe.name().to_string()}) ThemedText }
            }
            on(move |_: On<Activate>, mut cameras: Query<&mut SolariCamera>| {
                if let Ok(mut solari) = cameras.get_mut(camera) {
                    solari.apply_recipe(recipe);
                }
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
                                    @caption: bsn! { Text("mode: custom") ThemedText RecipeLabel({camera}) }
                                }
                            ),
                            (
                                @FeathersMenuPopup
                                Children [
                                    recipe_item(camera, SolariRecipe::Reference),
                                    recipe_item(camera, SolariRecipe::Bsdf),
                                    recipe_item(camera, SolariRecipe::RestirDi),
                                    recipe_item(camera, SolariRecipe::RestirDiSpatial),
                                    recipe_item(camera, SolariRecipe::RestirGi),
                                    recipe_item(camera, SolariRecipe::RestirGiSpatial),
                                    recipe_item(camera, SolariRecipe::Nrc),
                                    recipe_item(camera, SolariRecipe::Default),
                                ]
                            )
                        ]
                    ),
                    (
                        @FeathersMenu
                        Children [
                            (
                                @FeathersMenuButton {
                                    @caption: bsn! { Text("view: normal") ThemedText ViewLabel({camera}) }
                                }
                            ),
                            (
                                @FeathersMenuPopup
                                Children [
                                    view_item(camera, SolariDebugView::None, "normal"),
                                    view_item(camera, SolariDebugView::cost_heatmap(), "time heatmap"),
                                    view_item(camera, SolariDebugView::any_hit_count(), "any-hit count"),
                                    view_item(camera, SolariDebugView::Displacement, "displacement"),
                                    view_item(camera, SolariDebugView::Clusters, "clusters"),
                                    view_item(camera, SolariDebugView::Triangles, "triangles"),
                                    view_item(camera, SolariDebugView::NormalFacing, "normal facing"),
                                    view_item(camera, SolariDebugView::NrcCache, "nrc cache"),
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
                        HeatmapControls({camera})
                        Children [
                            (Text("center") ThemedText),
                            (
                                @FeathersSlider { @min: 10.0, @max: 24.0, @value: 16.0 }
                                SliderPrecision(1)
                                on(slider_self_update)
                                on(move |c: On<ValueChange<f32>>, mut cameras: Query<&mut SolariCamera>| {
                                    if let Ok(mut solari) = cameras.get_mut(camera)
                                        && let SolariDebugView::CostHeatmap { center, .. } = &mut solari.debug
                                    {
                                        *center = c.value;
                                    }
                                })
                            ),
                            (Text("contrast") ThemedText),
                            (
                                @FeathersSlider { @min: 0.0, @max: 1.0, @value: 0.15 }
                                SliderPrecision(2)
                                on(slider_self_update)
                                on(move |c: On<ValueChange<f32>>, mut cameras: Query<&mut SolariCamera>| {
                                    if let Ok(mut solari) = cameras.get_mut(camera)
                                        && let SolariDebugView::CostHeatmap { contrast, .. } = &mut solari.debug
                                    {
                                        *contrast = c.value;
                                    }
                                })
                            ),
                        ]
                    ),
                    (
                        @FeathersCheckbox {
                            @caption: bsn! { Text("accumulate") ThemedText }
                        }
                        AccumulateCheckbox({camera})
                        on(move |change: On<ValueChange<bool>>, mut cameras: Query<&mut SolariCamera>| {
                            if let Ok(mut solari) = cameras.get_mut(camera)
                                && let SolariLighting::Reference(reference) = &mut solari.mode
                            {
                                reference.accumulate = change.value;
                            }
                        })
                    )
                });
            commands.entity(camera).insert(ViewPanelSpawned);
        }
    }

    /// Keep each view button caption in sync with its camera's debug view.
    pub fn update_view_label(
        cameras: Query<&SolariCamera>,
        mut labels: Query<(&mut Text, &ViewLabel)>,
    ) {
        for (mut text, label) in &mut labels {
            let Ok(camera) = cameras.get(label.0) else {
                continue;
            };
            let want = match camera.debug {
                SolariDebugView::None => "view: normal",
                SolariDebugView::CostHeatmap { .. } => "view: time heatmap",
                SolariDebugView::AnyHitCount { .. } => "view: any-hit count",
                SolariDebugView::Displacement => "view: displacement",
                SolariDebugView::Clusters => "view: clusters",
                SolariDebugView::Triangles => "view: triangles",
                SolariDebugView::NormalFacing => "view: normal facing",
                SolariDebugView::NrcCache => "view: nrc cache",
            };
            if text.0 != want {
                text.0 = want.to_string();
            }
        }
    }

    /// Keep each recipe button caption in sync with its camera's mode:
    /// the matching recipe's name, or "custom" once the levers are hand-tweaked.
    pub fn update_recipe_label(
        cameras: Query<&SolariCamera>,
        mut labels: Query<(&mut Text, &RecipeLabel)>,
    ) {
        for (mut text, label) in &mut labels {
            let Ok(camera) = cameras.get(label.0) else {
                continue;
            };
            let name = SolariRecipe::ALL
                .iter()
                .find(|recipe| recipe.matches(&camera.mode))
                .map_or("custom", SolariRecipe::name);
            let want = format!("mode: {name}");
            if text.0 != want {
                text.0 = want;
            }
        }
    }

    /// Mirror each camera's reference-accumulation state onto its checkbox
    /// (`Checked` marker + row visibility) — the camera is the single source
    /// of truth, so CLI/recipe/code changes all reflect in the UI. The
    /// checkbox's `ValueChange` observer writes the camera; this closes the loop.
    pub fn sync_accumulate_checkbox(
        cameras: Query<&SolariCamera>,
        mut boxes: Query<(Entity, &mut Node, &AccumulateCheckbox, Has<Checked>)>,
        mut commands: Commands,
    ) {
        for (entity, mut node, bound, checked) in &mut boxes {
            let Ok(camera) = cameras.get(bound.0) else {
                continue;
            };
            let reference = camera.reference();
            let want_display = if reference.is_some() { Display::Flex } else { Display::None };
            if node.display != want_display {
                node.display = want_display;
            }
            let want_checked = reference.is_some_and(|r| r.accumulate);
            if want_checked != checked {
                if want_checked {
                    commands.entity(entity).insert(Checked);
                } else {
                    commands.entity(entity).remove::<Checked>();
                }
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

    /// Show the heatmap sliders only while their camera is in cost-heatmap view.
    pub fn toggle_heatmap_controls(
        cameras: Query<&SolariCamera>,
        mut controls: Query<(&mut Node, &HeatmapControls)>,
    ) {
        for (mut node, bound) in &mut controls {
            let Ok(camera) = cameras.get(bound.0) else {
                continue;
            };
            let want = if matches!(camera.debug, SolariDebugView::CostHeatmap { .. }) {
                Display::Flex
            } else {
                Display::None
            };
            if node.display != want {
                node.display = want;
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
