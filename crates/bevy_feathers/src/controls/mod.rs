//! Meta-module containing all feathers controls (widgets that are interactive).

mod button;
mod checkbox;
mod color_input;
mod color_plane;
mod color_slider;
mod color_swatch;
mod color_swatch_grid;
mod color_wheel;
mod dialog;
mod disclosure_toggle;
mod listview;
mod menu;
mod number_input;
mod radio;
mod scrollbar;
mod select;
mod slider;
mod split_pane;
mod text_input;
mod toggle_switch;
mod tree_view;
mod virtual_keyboard;

pub use button::*;
pub use checkbox::*;
pub use color_input::*;
pub use color_plane::*;
pub use color_slider::*;
pub use color_swatch::*;
pub use color_swatch_grid::*;
pub use color_wheel::*;
pub use dialog::*;
pub use disclosure_toggle::*;
pub use listview::*;
pub use menu::*;
pub use number_input::*;
pub use radio::*;
pub use scrollbar::*;
pub use select::*;
pub use slider::*;
pub use split_pane::*;
pub use text_input::*;
pub use toggle_switch::*;
pub use tree_view::*;
pub use virtual_keyboard::*;

#[cfg(feature = "render_materials")]
use crate::alpha_pattern::AlphaPatternPlugin;
use bevy_app::{PluginGroup, PluginGroupBuilder};

/// Plugin group which registers all `bevy_feathers` controls.
pub struct ControlsPlugin;

impl PluginGroup for ControlsPlugin {
    fn build(self) -> PluginGroupBuilder {
        let group = PluginGroupBuilder::start::<Self>()
            .add(ButtonPlugin)
            .add(CheckboxPlugin)
            .add(ColorInputPlugin)
            .add(ColorPlanePlugin)
            .add(ColorSliderPlugin)
            .add(ColorSwatchPlugin)
            .add(ColorSwatchGridPlugin)
            .add(ColorWheelPlugin)
            .add(DisclosureTogglePlugin)
            .add(ListViewPlugin)
            .add(MenuPlugin)
            .add(NumberInputPlugin)
            .add(RadioPlugin)
            .add(ScrollbarPlugin)
            .add(SelectPlugin)
            .add(SliderPlugin)
            .add(SplitPanePlugin)
            .add(TextInputPlugin)
            .add(ToggleSwitchPlugin)
            .add(TreeViewPlugin);
        #[cfg(feature = "render_materials")]
        let group = group.add(AlphaPatternPlugin);
        group
    }
}
