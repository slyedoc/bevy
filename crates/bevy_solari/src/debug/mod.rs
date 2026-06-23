mod ui;

use bevy_app::{App, Plugin, Update};
use bevy_feathers::FeathersCorePlugin;

/// Plugin that adds the in-engine debug-view selector: one Feathers dropdown
/// per [`SolariCamera`](crate::render::SolariCamera), each in its own viewport.
/// Picking a view sets that camera's
/// [`SolariViewState`](crate::render::view::SolariViewState).
///
/// Replaces the old `Tab`/`Shift+Tab` hotkey, which collided with Feathers' tab
/// navigation — the dropdown is a focusable widget in a `TabGroup`, so `Tab`
/// navigates it instead.
pub struct SolariDebugPlugin;

impl Plugin for SolariDebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                ui::spawn_render_debug_panels,
                ui::update_render_debug_label,
            ),
        );

        // DLSS Ray Reconstruction mode dropdown (one per SolariCamera).
        #[cfg(feature = "dlss")]
        app.add_systems(Update, (ui::spawn_dlss_panels, ui::update_dlss_label));
    }

    fn finish(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<FeathersCorePlugin>(),
            "SolariDebugPlugin requires `FeathersCorePlugin` to be added to the app.",
        );
    }
}
