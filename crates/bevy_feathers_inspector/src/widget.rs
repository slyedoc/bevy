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
use bevy_reflect::{FromType, PartialReflect};
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_ui::Checked;
use bevy_ui_widgets::{checkbox_self_update, slider_self_update, SliderPrecision};

use crate::attributes::FieldCtx;
use crate::binding::{
    inspector_writeback_bool, inspector_writeback_slider, InspectorBinding,
};
use crate::recurse::{parse_path, BuildCx};

/// Function that builds a bound widget [`Scene`] for a reflected value.
pub type WidgetBuildFn =
    fn(cx: &BuildCx, path: &str, value: &dyn PartialReflect, field: &FieldCtx) -> Box<dyn Scene>;

/// A builder that produces a feathers widget [`Scene`] for a specific reflected leaf type.
///
/// The `build` function is monomorphized per type at registration time (see [`FromType`]), so it
/// knows the concrete `T` and can attach the correctly-typed writeback observer.
#[derive(Clone)]
pub struct ReflectInspectorWidget {
    /// Builds a bound widget scene for a value of the registered type.
    pub build: WidgetBuildFn,
}

/// A numeric type editable with a slider.
///
/// The slider always works in `f32`; this trait converts to and from the field's real type so
/// integers round-trip correctly.
pub trait SliderScalar: PartialReflect + Send + Sync + Sized + 'static {
    /// Displayed decimal precision (0 for integers).
    const PRECISION: i32;
    /// Convert the slider's `f32` back to this type.
    fn from_slider_f32(v: f32) -> Self;
    /// Read the current value out of a reflected reference as `f32`.
    fn to_slider_f32(value: &dyn PartialReflect) -> Option<f32>;
}

macro_rules! impl_slider_scalar {
    ($t:ty, $precision:expr, $from:expr) => {
        impl SliderScalar for $t {
            const PRECISION: i32 = $precision;
            fn from_slider_f32(v: f32) -> Self {
                let f = $from;
                f(v)
            }
            fn to_slider_f32(value: &dyn PartialReflect) -> Option<f32> {
                value.try_as_reflect()?.downcast_ref::<$t>().map(|&x| x as f32)
            }
        }
    };
}

impl_slider_scalar!(f32, 3, |v: f32| v);
impl_slider_scalar!(f64, 3, |v: f32| v as f64);
impl_slider_scalar!(i32, 0, |v: f32| v.round() as i32);
impl_slider_scalar!(i64, 0, |v: f32| v.round() as i64);
impl_slider_scalar!(u32, 0, |v: f32| v.round().max(0.0) as u32);
impl_slider_scalar!(u64, 0, |v: f32| v.round().max(0.0) as u64);
impl_slider_scalar!(usize, 0, |v: f32| v.round().max(0.0) as usize);

/// A feathers slider bound to a numeric field.
fn build_numeric<T: SliderScalar>(
    cx: &BuildCx,
    path: &str,
    value: &dyn PartialReflect,
    field: &FieldCtx,
) -> Box<dyn Scene> {
    let current = T::to_slider_f32(value).unwrap_or_default();
    let (min, max) = field.range.unwrap_or_else(|| derived_range(current));
    Box::new((
        <FeathersSlider as SceneComponent>::scene(FeathersSliderProps {
            value: current,
            min,
            max,
        }),
        // `update_slider_pos` requires `SliderPrecision`; without it the bar/text never update.
        template_value(SliderPrecision(T::PRECISION)),
        template_value(InspectorBinding {
            root: Some(cx.root.clone()),
            path: parse_path(path),
        }),
        // `slider_self_update` moves the thumb; our observer writes the data back.
        on(slider_self_update),
        on(inspector_writeback_slider::<T>),
    ))
}

/// When a field has no explicit range, derive one that keeps the value visible mid-track.
fn derived_range(current: f32) -> (f32, f32) {
    let span = (current.abs() * 2.0).max(1.0);
    if current < 0.0 {
        (-span, 0.0)
    } else {
        (0.0, span)
    }
}

/// A feathers checkbox bound to a `bool` field.
fn build_bool(
    cx: &BuildCx,
    path: &str,
    value: &dyn PartialReflect,
    _field: &FieldCtx,
) -> Box<dyn Scene> {
    let current = value
        .try_as_reflect()
        .and_then(|r| r.downcast_ref::<bool>())
        .copied()
        .unwrap_or_default();
    let binding = InspectorBinding {
        root: Some(cx.root.clone()),
        path: parse_path(path),
    };
    // The `Checked` marker seeds the visual state; `checkbox_self_update` keeps it in sync while
    // our observer writes the data back. A checked/unchecked branch avoids an `Option<Scene>`.
    if current {
        Box::new((
            <FeathersCheckbox as SceneComponent>::scene(FeathersCheckboxProps::default()),
            template_value(Checked),
            template_value(binding),
            on(checkbox_self_update),
            on(inspector_writeback_bool),
        ))
    } else {
        Box::new((
            <FeathersCheckbox as SceneComponent>::scene(FeathersCheckboxProps::default()),
            template_value(binding),
            on(checkbox_self_update),
            on(inspector_writeback_bool),
        ))
    }
}

macro_rules! impl_numeric_widget {
    ($t:ty) => {
        impl FromType<$t> for ReflectInspectorWidget {
            fn from_type() -> Self {
                Self {
                    build: build_numeric::<$t>,
                }
            }
        }
    };
}

impl_numeric_widget!(f32);
impl_numeric_widget!(f64);
impl_numeric_widget!(i32);
impl_numeric_widget!(i64);
impl_numeric_widget!(u32);
impl_numeric_widget!(u64);
impl_numeric_widget!(usize);

impl FromType<bool> for ReflectInspectorWidget {
    fn from_type() -> Self {
        Self { build: build_bool }
    }
}

/// Registers the built-in leaf widgets with the [`TypeRegistry`].
pub struct DefaultInspectorWidgetsPlugin;

impl Plugin for DefaultInspectorWidgetsPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        macro_rules! register {
            ($t:ty) => {
                app.register_type::<$t>();
                app.register_type_data::<$t, ReflectInspectorWidget>();
            };
        }
        register!(f32);
        register!(f64);
        register!(i32);
        register!(i64);
        register!(u32);
        register!(u64);
        register!(usize);
        register!(bool);
    }
}

/// Plugin group installing the inspector's default widget set.
pub struct FeathersInspectorPlugins;

impl PluginGroup for FeathersInspectorPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>().add(DefaultInspectorWidgetsPlugin)
    }
}
