//! `bevy_feathers` is a collection of styled and themed widgets for building editors and
//! inspectors.
//!
//! The aesthetic choices made here are designed with a future Bevy Editor in mind,
//! but this crate is deliberately exposed to the public to allow the broader ecosystem to easily create
//! tooling for themselves and others that fits cohesively together.
//!
//! While it may be tempting to use this crate for your game's UI, it's deliberately not intended for that.
//! We've opted for a clean, functional style, and prioritized consistency over customization.
//! That said, if you like what you see, it can be a helpful learning tool.
//! Consider copying this code into your own project,
//! and refining the styles and abstractions provided to meet your needs.
//!
//! ## Best practices for event propagation
//!
//! Generally, when a widget handles an event,
//! propagation of that event to parent entities should be stopped.
//! This is important when writing your custom widgets, and understanding the behavior of existing widgets.
//!
//! For more guidance on this, see the documentation for [`EntityEvent`](bevy_ecs::event::EntityEvent).
//!
//! ## Using feathers without `bevy_render`
//!
//! Almost all of feathers is renderer-agnostic: it is themes, layout, cursors and observers on top
//! of `bevy_ui` and `bevy_ui_widgets`. Only two pieces actually draw: the checkerboard alpha
//! pattern behind color swatches and color sliders, and the `color_plane` color-picker control.
//! Both are `UiMaterial`s, so both need `bevy_render`.
//!
//! Those two live behind the `render_materials` crate feature, which is on by default. Turn it off
//! (`bevy` feature `bevy_feathers_core` instead of `bevy_feathers`) and `bevy_render`,
//! `bevy_shader` and `bevy_ui_render` leave the dependency graph entirely: `color_plane` is not
//! compiled, and color swatches / sliders lose the checkerboard behind their translucent colors but
//! otherwise work as normal. This is meant for projects that drive `bevy_ui` with their own
//! rendering backend.
//!
//! ## Warning: Experimental!
//! All that said, this crate is still experimental and unfinished!
//! It will change in breaking ways, and there will be both bugs and limitations.
//!
//! Please report issues, submit fixes and propose changes.
//! Thanks for stress-testing; let's build something better together.

extern crate alloc;

use bevy_app::{
    HierarchyPropagatePlugin, Plugin, PluginGroup, PluginGroupBuilder, PostUpdate, PropagateSet,
};
use bevy_asset::embedded_asset;
use bevy_ecs::{query::With, schedule::IntoScheduleConfigs};
use bevy_input_focus::tab_navigation::TabNavigationPlugin;
use bevy_text::{TextColor, TextFont};
use bevy_ui::UiSystems;
#[cfg(feature = "render_materials")]
use bevy_ui_render::UiMaterialPlugin;

#[cfg(feature = "render_materials")]
use crate::alpha_pattern::{AlphaPatternMaterial, AlphaPatternResource};
use crate::{
    controls::ControlsPlugin,
    cursor::{CursorIconPlugin, DefaultCursor, EntityCursor},
    theme::{ThemedText, UiTheme},
};

mod alpha_pattern;
pub mod constants;
pub mod containers;
pub mod controls;
pub mod cursor;
pub mod dark_theme;
pub mod display;
pub mod focus;
pub mod font_styles;
pub mod palette;
pub mod rounded_corners;
pub mod theme;
pub mod tokens;

/// Plugin which installs observers and systems for feathers themes, cursors, and all controls.
pub struct FeathersCorePlugin;

impl Plugin for FeathersCorePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_resource::<UiTheme>();

        // Embedded font
        embedded_asset!(app, "assets/fonts/FiraSans-Bold.ttf");
        embedded_asset!(app, "assets/fonts/FiraSans-BoldItalic.ttf");
        embedded_asset!(app, "assets/fonts/FiraSans-Regular.ttf");
        embedded_asset!(app, "assets/fonts/FiraSans-Italic.ttf");
        embedded_asset!(app, "assets/fonts/FiraMono-Medium.ttf");

        // Embedded icons
        embedded_asset!(app, "assets/icons/chevron-down.png");
        embedded_asset!(app, "assets/icons/chevron-right.png");
        embedded_asset!(app, "assets/icons/x.png");

        // Embedded shader
        #[cfg(feature = "render_materials")]
        {
            embedded_asset!(app, "assets/shaders/alpha_pattern.wgsl");
            embedded_asset!(app, "assets/shaders/color_plane.wgsl");
        }

        app.add_plugins((
            ControlsPlugin,
            CursorIconPlugin,
            HierarchyPropagatePlugin::<TextColor, With<ThemedText>>::new(PostUpdate),
            HierarchyPropagatePlugin::<TextFont, With<ThemedText>>::new(PostUpdate),
            focus::FocusOutlinesPlugin,
        ));

        #[cfg(feature = "render_materials")]
        app.add_plugins(UiMaterialPlugin::<AlphaPatternMaterial>::default());

        // This needs to run in UiSystems::Propagate so the fonts are up-to-date for `measure_text_system`
        // and `detect_text_needs_rerender` in UiSystems::Content
        app.configure_sets(
            PostUpdate,
            PropagateSet::<TextFont>::default().in_set(UiSystems::Propagate),
        );

        app.insert_resource(DefaultCursor(EntityCursor::System(
            bevy_window::SystemCursorIcon::Default,
        )));

        app.add_systems(
            PostUpdate,
            (
                theme::update_theme,
                display::update_themed_icons.after(PropagateSet::<TextColor>::default()),
            ),
        )
        .add_observer(theme::on_changed_background)
        .add_observer(theme::on_changed_border)
        .add_observer(theme::on_changed_font_color)
        .add_observer(theme::on_changed_text_color)
        .add_observer(font_styles::on_changed_font);

        #[cfg(feature = "render_materials")]
        app.init_resource::<AlphaPatternResource>();
    }
}

/// A plugin group that adds all dependencies for Feathers
pub struct FeathersPlugins;

impl PluginGroup for FeathersPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(TabNavigationPlugin)
            .add(FeathersCorePlugin)
    }
}

// Feathers without its render half is a configuration nothing else in the workspace exercises,
// so guard it with a smoke test: the whole plugin group has to build and tick with no
// `bevy_render` in the process at all.
#[cfg(all(test, not(feature = "render_materials")))]
mod no_render_tests {
    use super::*;
    use bevy_asset::AssetApp;

    /// `FeathersPlugins` builds and runs a frame with no renderer behind it.
    #[test]
    fn feathers_plugins_run_without_bevy_render() {
        let mut app = bevy_app::App::new();
        app.add_plugins((
            bevy_app::TaskPoolPlugin::default(),
            bevy_time::TimePlugin,
            bevy_asset::AssetPlugin::default(),
            bevy_window::WindowPlugin::default(),
            bevy_input::InputPlugin,
            bevy_picking::PickingPlugin,
            bevy_picking::InteractionPlugin,
            bevy_text::TextPlugin,
            bevy_ui::UiPlugin,
            bevy_input_focus::InputFocusPlugin,
            bevy_input_focus::InputDispatchPlugin,
            FeathersPlugins,
        ));
        // Normally initialized by `RenderPlugin` / `ImagePlugin`, which are exactly what this
        // configuration does without.
        app.init_asset::<bevy_image::Image>();
        app.init_asset::<bevy_image::TextureAtlasLayout>();

        app.finish();
        app.cleanup();
        app.update();
    }
}
