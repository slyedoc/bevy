//! The per-type widget dispatch channel.
//!
//! [`ReflectInspectorWidget`] is a [`TypeData`](bevy_reflect::TypeData) registered in the
//! [`TypeRegistry`](bevy_reflect::TypeRegistry) for each leaf type the inspector can edit. It
//! mirrors `bevy-inspector-egui`'s `InspectorEguiImpl`, but instead of drawing egui it returns a
//! boxed feathers [`Scene`] bound to the value.

use bevy_app::{Plugin, PluginGroup, PluginGroupBuilder};
use bevy_feathers::controls::{
    FeathersCheckbox, FeathersCheckboxProps, FeathersSlider, FeathersSliderProps,
};
use bevy_reflect::{FromType, ParsedPath, PartialReflect};
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_ui::Checked;
use bevy_ui_widgets::{checkbox_self_update, slider_self_update, SliderPrecision};

use crate::binding::{inspector_writeback, InspectorBinding, InspectorRoot};

/// A builder that produces a feathers widget [`Scene`] for a specific reflected leaf type.
///
/// The `build` function is monomorphized per type at registration time (see [`FromType`]), so it
/// knows the concrete `T` and can attach the correctly-typed [`inspector_writeback`] observer.
///
/// `TypeData` is blanket-implemented for `Clone` types, so deriving [`Clone`] is all that is needed
/// to store this in the [`TypeRegistry`](bevy_reflect::TypeRegistry).
#[derive(Clone)]
pub struct ReflectInspectorWidget {
    /// Builds a bound widget scene for a value of the registered type.
    pub build: fn(root: InspectorRoot, path: ParsedPath, value: &dyn PartialReflect) -> Box<dyn Scene>,
}

impl FromType<f32> for ReflectInspectorWidget {
    fn from_type() -> Self {
        Self { build: build_f32 }
    }
}

impl FromType<bool> for ReflectInspectorWidget {
    fn from_type() -> Self {
        Self { build: build_bool }
    }
}

/// A feathers slider bound to an `f32` field.
///
/// The slider carries its value as a `SliderValue` component set directly in the scene, so the
/// value displays immediately with a visible track and thumb. Until per-field ranges arrive (via
/// `#[reflect(@range)]` in a later phase) we derive a simple range that keeps the current value
/// visible mid-track.
fn build_f32(root: InspectorRoot, path: ParsedPath, value: &dyn PartialReflect) -> Box<dyn Scene> {
    let current = value
        .try_as_reflect()
        .and_then(|r| r.downcast_ref::<f32>())
        .copied()
        .unwrap_or_default();
    let span = (current.abs() * 2.0).max(1.0);
    let (min, max) = if current < 0.0 {
        (-span, 0.0)
    } else {
        (0.0, span)
    };
    Box::new((
        <FeathersSlider as SceneComponent>::scene(FeathersSliderProps {
            value: current,
            min,
            max,
        }),
        // `update_slider_pos` requires `SliderPrecision`; without it the bar and value text
        // never update.
        template_value(SliderPrecision(3)),
        template_value(InspectorBinding {
            root: Some(root),
            path,
        }),
        // `slider_self_update` moves the thumb; our observer writes the data back.
        on(slider_self_update),
        on(inspector_writeback::<f32>),
    ))
}

/// A feathers checkbox bound to a `bool` field.
fn build_bool(root: InspectorRoot, path: ParsedPath, value: &dyn PartialReflect) -> Box<dyn Scene> {
    let current = value
        .try_as_reflect()
        .and_then(|r| r.downcast_ref::<bool>())
        .copied()
        .unwrap_or_default();
    // The `Checked` marker seeds the visual state; `checkbox_self_update` keeps it in sync while
    // our observer writes the data back. A checked/unchecked branch avoids an `Option<Scene>`,
    // which is not a `Scene` in this version.
    if current {
        Box::new((
            <FeathersCheckbox as SceneComponent>::scene(FeathersCheckboxProps::default()),
            template_value(Checked),
            template_value(InspectorBinding {
                root: Some(root),
                path,
            }),
            on(checkbox_self_update),
            on(inspector_writeback::<bool>),
        ))
    } else {
        Box::new((
            <FeathersCheckbox as SceneComponent>::scene(FeathersCheckboxProps::default()),
            template_value(InspectorBinding {
                root: Some(root),
                path,
            }),
            on(checkbox_self_update),
            on(inspector_writeback::<bool>),
        ))
    }
}

/// Registers the built-in leaf widgets ([`f32`], [`bool`]) with the [`TypeRegistry`].
pub struct DefaultInspectorWidgetsPlugin;

impl Plugin for DefaultInspectorWidgetsPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.register_type::<f32>();
        app.register_type::<bool>();
        app.register_type_data::<f32, ReflectInspectorWidget>();
        app.register_type_data::<bool, ReflectInspectorWidget>();
    }
}

/// Plugin group installing the inspector's default widget set.
pub struct FeathersInspectorPlugins;

impl PluginGroup for FeathersInspectorPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>().add(DefaultInspectorWidgetsPlugin)
    }
}
