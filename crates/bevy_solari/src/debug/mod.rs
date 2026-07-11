mod ui;

/// The shared per-camera debug card root — external crates (e.g. the planet
/// crate's terrain-view dropdown) append their own rows into it.
pub use ui::ViewPanelRoot;

use bevy_app::{App, Plugin, Update};
use bevy_ecs::{resource::Resource, schedule::IntoScheduleConfigs, system::Res};
use bevy_feathers::FeathersCorePlugin;

/// Master switch for the debug UI (default on). Harness runs that capture
/// screenshots insert `SolariDebugUi(false)` so the panels never spawn —
/// screenshots grab the full window surface, UI included.
#[derive(Resource)]
pub struct SolariDebugUi(pub bool);

fn debug_ui_enabled(enabled: Option<Res<SolariDebugUi>>) -> bool {
    enabled.is_none_or(|e| e.0)
}

/// Plugin that adds the in-engine debug UI: per-[`SolariCamera`](crate::render::SolariCamera)
/// Feathers dropdowns for the render-debug overlay and the "view" selector (normal vs
/// per-pixel cost heatmap), plus the heatmap's center/contrast sliders and the DLSS mode
/// dropdown. Each is a focusable widget in a `TabGroup`, so `Tab` navigates them.
pub struct SolariDebugPlugin;

impl Plugin for SolariDebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                ui::spawn_render_debug_panels,
                ui::update_render_debug_label,
                // "view" dropdown (normal / time heatmap) + the cost-heatmap sliders.
                ui::spawn_view_panels,
                ui::update_view_label,
                ui::update_recipe_label,
                ui::sync_accumulate_checkbox,
                ui::update_stats_label,
                ui::toggle_heatmap_controls,
            )
                .run_if(debug_ui_enabled),
        );

        // DLSS Ray Reconstruction mode dropdown (one per SolariCamera); hidden in
        // heatmap view.
        app.add_systems(
            Update,
            (
                ui::spawn_dlss_panels,
                ui::update_dlss_label,
                ui::toggle_dlss_visibility,
            )
                .run_if(debug_ui_enabled),
        );
    }

    fn finish(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<FeathersCorePlugin>(),
            "SolariDebugPlugin requires `FeathersCorePlugin` to be added to the app.",
        );
    }
}
