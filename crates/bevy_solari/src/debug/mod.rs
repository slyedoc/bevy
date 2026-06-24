mod ui;

use bevy_app::{App, Plugin, Update};
use bevy_feathers::FeathersCorePlugin;

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
                ui::toggle_heatmap_controls,
            ),
        );

        // DLSS Ray Reconstruction mode dropdown (one per SolariCamera); hidden in
        // heatmap view.
        #[cfg(feature = "dlss")]
        app.add_systems(
            Update,
            (
                ui::spawn_dlss_panels,
                ui::update_dlss_label,
                ui::toggle_dlss_visibility,
            ),
        );
    }

    fn finish(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<FeathersCorePlugin>(),
            "SolariDebugPlugin requires `FeathersCorePlugin` to be added to the app.",
        );
    }
}
