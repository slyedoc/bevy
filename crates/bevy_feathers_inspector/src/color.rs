//! Colour picker for [`bevy_color::Color`] fields: a swatch plus HSL hue / saturation /
//! lightness colour sliders, driven by one [`ColorPicker`] on the widget root.
//!
//! The 2D colour plane is left out on purpose: its gradient is a `UiMaterial`, which
//! renderers without `bevy_render` do not paint, while the sliders' fills are plain
//! `BackgroundGradient`s that every backend draws.
//!
//! Data flow: a slider's [`ValueChange<f32>`] updates the picker's HSL and writes the colour
//! back through reflection in the field's own variant ([`ColorPicker::color`]);
//! [`refresh_color_pickers`] then pushes the new colour into every slider's base colour /
//! value and the swatch, so the other sliders' gradients follow. External edits of the field
//! reach the picker through [`crate::sync`].

use bevy_color::{Color, Hsla, LinearRgba, Srgba};
use bevy_ecs::hierarchy::{ChildOf, Children};
use bevy_ecs::prelude::*;
use bevy_feathers::controls::{
    ColorChannel, ColorSlider, ColorSwatchValue, FeathersColorSlider, FeathersColorSliderProps,
    FeathersColorSwatch, FeathersColorSwatchProps, SliderBaseColor,
};
use bevy_reflect::{CreateTypeData, PartialReflect};
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_ui::{percent, px, AlignSelf, BorderRadius, Display, FlexDirection, Node};
use bevy_ui_widgets::{SliderValue, ValueChange};

use crate::attributes::FieldCtx;
use crate::binding::{with_field_reflect_mut, InspectorBinding};
use crate::recurse::{parse_path, BuildCx};
use crate::widget::ReflectInspectorWidget;

/// Which [`Color`] variant the edited field holds, so writes keep it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    Srgba,
    LinearRgba,
    Other,
}

/// State of one colour picker: the colour in HSL (what the sliders edit) and the variant
/// to write back.
#[derive(Component, Clone)]
pub struct ColorPicker {
    /// The edited colour.
    pub hsl: Hsla,
    variant: Variant,
}

impl ColorPicker {
    pub(crate) fn from_color(color: Color) -> Self {
        let variant = match color {
            Color::Srgba(_) => Variant::Srgba,
            Color::LinearRgba(_) => Variant::LinearRgba,
            _ => Variant::Other,
        };
        Self {
            hsl: Hsla::from(color),
            variant,
        }
    }

    /// The colour as the field's own variant.
    pub fn color(&self) -> Color {
        match self.variant {
            Variant::Srgba => Color::Srgba(Srgba::from(self.hsl)),
            Variant::LinearRgba => Color::LinearRgba(LinearRgba::from(self.hsl)),
            Variant::Other => Color::Hsla(self.hsl),
        }
    }

    /// Whether `color` differs from the picker's colour beyond slider precision.
    pub(crate) fn differs_from(&self, color: Color) -> bool {
        let other = Hsla::from(color);
        (self.hsl.hue - other.hue).abs() > 1e-3
            || (self.hsl.saturation - other.saturation).abs() > 1e-4
            || (self.hsl.lightness - other.lightness).abs() > 1e-4
            || (self.hsl.alpha - other.alpha).abs() > 1e-4
    }
}

/// The bound colour picker scene for a `Color` field.
fn build_color(
    cx: &BuildCx,
    path: &str,
    value: &dyn PartialReflect,
    _field: &FieldCtx,
) -> Box<dyn Scene> {
    let current = value
        .try_as_reflect()
        .and_then(|r| r.downcast_ref::<Color>())
        .copied()
        .unwrap_or(Color::WHITE);
    let picker = ColorPicker::from_color(current);
    let base = Color::from(picker.hsl);
    let binding = InspectorBinding {
        root: Some(cx.root.clone()),
        path: parse_path(path),
    };

    let swatch: Vec<Box<dyn Scene>> = vec![Box::new((
        <FeathersColorSwatch as SceneComponent>::scene(FeathersColorSwatchProps::default()),
        // Full-width strip instead of the default square.
        template_value(Node {
            height: px(18),
            width: percent(100),
            border_radius: BorderRadius::all(px(5)),
            ..Default::default()
        }),
        template_value(ColorSwatchValue(base)),
    ))];
    let sliders: Vec<Box<dyn Scene>> = [
        (ColorChannel::HslHue, picker.hsl.hue),
        (ColorChannel::HslSaturation, picker.hsl.saturation),
        (ColorChannel::HslLightness, picker.hsl.lightness),
    ]
    .into_iter()
    .map(|(channel, value)| {
        Box::new((
            <FeathersColorSlider as SceneComponent>::scene(FeathersColorSliderProps {
                value,
                channel,
            }),
            template_value(SliderBaseColor(base)),
            on(on_color_slider_change),
        )) as Box<dyn Scene>
    })
    .collect();

    Box::new((
        bsn! {
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                align_self: AlignSelf::Stretch,
                flex_grow: 1.0,
                row_gap: px(4),
            }
            Children [ {swatch}, {sliders} ]
        },
        template_value(picker),
        template_value(binding),
    ))
}

impl CreateTypeData<Color> for ReflectInspectorWidget {
    fn create_type_data(_: ()) -> Self {
        Self { build: build_color }
    }
}

/// A colour slider moved: update the picker's HSL and write the colour back to the field.
fn on_color_slider_change(
    event: On<ValueChange<f32>>,
    sliders: Query<&ColorSlider>,
    parents: Query<&ChildOf>,
    mut pickers: Query<(&mut ColorPicker, &InspectorBinding)>,
    mut commands: Commands,
) {
    let Ok(slider) = sliders.get(event.source) else {
        return;
    };
    // The slider sits somewhere under the picker root.
    let mut entity = event.source;
    let root_entity = loop {
        if pickers.contains(entity) {
            break entity;
        }
        match parents.get(entity) {
            Ok(child_of) => entity = child_of.parent(),
            Err(_) => return,
        }
    };
    let Ok((mut picker, binding)) = pickers.get_mut(root_entity) else {
        return;
    };
    match slider.channel {
        ColorChannel::HslHue => picker.hsl.hue = event.value,
        ColorChannel::HslSaturation => picker.hsl.saturation = event.value,
        ColorChannel::HslLightness => picker.hsl.lightness = event.value,
        _ => return,
    }
    let Some(root) = binding.root.clone() else {
        return;
    };
    let path = binding.path.clone();
    let color = picker.color();
    commands.queue(move |world: &mut World| {
        with_field_reflect_mut(world, &root, &path, |target| {
            let _ = target.try_apply(color.as_partial_reflect());
        });
    });
}

/// Push a changed picker colour into its sliders (value + gradient base) and swatch.
pub fn refresh_color_pickers(
    pickers: Query<(Entity, &ColorPicker), Changed<ColorPicker>>,
    children: Query<&Children>,
    mut sliders: Query<(&ColorSlider, &mut SliderBaseColor, &SliderValue)>,
    mut swatches: Query<&mut ColorSwatchValue>,
    mut commands: Commands,
) {
    for (root, picker) in &pickers {
        let base = Color::from(picker.hsl);
        for entity in children.iter_descendants(root) {
            if let Ok((slider, mut base_color, value)) = sliders.get_mut(entity) {
                base_color.0 = base;
                let channel_value = match slider.channel {
                    ColorChannel::HslHue => picker.hsl.hue,
                    ColorChannel::HslSaturation => picker.hsl.saturation,
                    ColorChannel::HslLightness => picker.hsl.lightness,
                    _ => continue,
                };
                // `SliderValue` is immutable: replaced by insert, like the core slider does.
                if (value.0 - channel_value).abs() > f32::EPSILON {
                    commands.entity(entity).insert(SliderValue(channel_value));
                }
            }
            if let Ok(mut swatch) = swatches.get_mut(entity) {
                swatch.0 = base;
            }
        }
    }
}
