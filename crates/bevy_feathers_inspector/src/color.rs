//! Colour picker for [`bevy_color::Color`] fields: a swatch and hex entry, a red/blue plane, and
//! red, green, blue and alpha sliders, all driven by one [`ColorPicker`] on the widget root.
//!
//! Data flow: a sub-widget's change updates the picker's sRGBA and the picker root emits a
//! [`ValueChange<Color>`] in the field's own variant ([`ColorPicker::color`]), which a bound
//! field writes back; [`refresh_color_pickers`] then pushes the new colour into the swatch,
//! plane, sliders and hex text. Set the colour from outside by inserting a [`ColorPicker`]; bound
//! fields get that from [`crate::sync`].
//!
//! [`color_picker`] is the same widget unbound, for any UI that edits a colour.

use bevy_color::{Color, LinearRgba, Srgba};
use bevy_ecs::hierarchy::{ChildOf, Children};
use bevy_ecs::prelude::*;
use bevy_feathers::controls::{
    ColorChannel, ColorPlaneValue, ColorSwatchValue, FeathersColorPlane, FeathersColorSlider,
    FeathersColorSwatch, FeathersTextInput, FeathersTextInputContainer, SliderBaseColor,
};
use bevy_input::keyboard::{KeyCode, KeyboardInput};
use bevy_input::ButtonState;
use bevy_input_focus::{FocusLost, FocusedInput, InputFocus};
use bevy_math::{Vec2, Vec3};
use bevy_reflect::{CreateTypeData, PartialReflect};
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_text::EditableText;
use bevy_ui::{px, AlignItems, AlignSelf, Display, FlexDirection, JustifyContent, Node};
use bevy_ui_widgets::{SliderValue, ValueChange};

use crate::attributes::FieldCtx;
use crate::binding::{queue_edit, InspectorBinding};
use crate::recurse::{parse_path, BuildCx};
use crate::text::{replace_text, ShownText};
use crate::widget::ReflectInspectorWidget;

/// Which [`Color`] variant the edited field holds, so writes keep it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Variant {
    Srgba,
    LinearRgba,
    Other,
}

/// State of one colour picker: the colour in sRGBA (what its widgets edit) and the variant to
/// write back.
#[derive(Component, Clone, Debug)]
pub struct ColorPicker {
    /// The edited colour.
    pub rgba: Srgba,
    variant: Variant,
}

impl ColorPicker {
    /// A picker showing `color`, which it emits edits in the variant of.
    pub fn from_color(color: Color) -> Self {
        let variant = match color {
            Color::Srgba(_) => Variant::Srgba,
            Color::LinearRgba(_) => Variant::LinearRgba,
            _ => Variant::Other,
        };
        Self {
            rgba: color.to_srgba(),
            variant,
        }
    }

    /// The colour as the field's own variant (sRGBA for the variants it does not keep).
    pub fn color(&self) -> Color {
        match self.variant {
            Variant::LinearRgba => Color::LinearRgba(LinearRgba::from(self.rgba)),
            Variant::Srgba | Variant::Other => Color::Srgba(self.rgba),
        }
    }

    /// Whether `color` differs from the picker's colour beyond slider precision.
    pub(crate) fn differs_from(&self, color: Color) -> bool {
        let other = color.to_srgba();
        (self.rgba.red - other.red).abs() > 1e-4
            || (self.rgba.green - other.green).abs() > 1e-4
            || (self.rgba.blue - other.blue).abs() > 1e-4
            || (self.rgba.alpha - other.alpha).abs() > 1e-4
    }
}

/// Which part of the picker a sub-widget edits.
#[derive(Component, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum ColorPart {
    #[default]
    Swatch,
    Plane,
    Channel(usize),
    Hex,
}

fn plane_value(rgba: Srgba) -> Vec3 {
    Vec3::new(rgba.red, rgba.blue, rgba.green)
}

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
    let binding = InspectorBinding {
        root: Some(cx.root),
        path: parse_path(path),
    };
    Box::new(bsn! {
        @color_picker(current)
        binding
        on(writeback_color)
    })
}

/// The colour picker, unbound: it emits a [`ValueChange<Color>`] from its root on every edit
/// (`is_final` when a drag ends or the hex text commits).
pub fn color_picker(current: Color) -> impl Scene {
    let picker = ColorPicker::from_color(current);
    let rgba = picker.rgba;
    let color = Color::Srgba(rgba);
    let hex = ShownText {
        text: rgba.to_hex(),
        seeded: false,
    };
    let plane = plane_value(rgba);

    let sliders: Vec<Box<dyn Scene>> = [
        (ColorChannel::Red, rgba.red),
        (ColorChannel::Green, rgba.green),
        (ColorChannel::Blue, rgba.blue),
        (ColorChannel::Alpha, rgba.alpha),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (channel, value))| {
        let part = ColorPart::Channel(i);
        Box::new(bsn! {
            @FeathersColorSlider { @value: {value}, @channel: {channel} }
            SliderBaseColor({color})
            part
            on(on_channel_change)
        }) as Box<dyn Scene>
    })
    .collect();

    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            align_self: AlignSelf::Stretch,
            flex_grow: 1.0,
            row_gap: px(4),
        }
        picker
        Children [
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::SpaceBetween,
                column_gap: px(6),
            }
            Children [
                @FeathersTextInputContainer
                Node { flex_grow: 0.0 }
                Children [
                    @FeathersTextInput { @visible_width: 10f32, @max_characters: 9usize }
                    hex
                    ColorPart::Hex
                    on(on_hex_key)
                    on(on_hex_focus_lost)
                ]
                --
                @FeathersColorSwatch
                ColorSwatchValue({color})
                ColorPart::Swatch
            ]
            --
            @FeathersColorPlane::RedBlue
            ColorPlaneValue({plane})
            ColorPart::Plane
            on(on_plane_change)
            --
            {sliders}
        ]
    }
}

impl CreateTypeData<Color> for ReflectInspectorWidget {
    fn create_type_data(_: ()) -> Self {
        Self { build: build_color }
    }
}

/// The picker root above a sub-widget.
fn picker_root(
    entity: Entity,
    parents: &Query<&ChildOf>,
    pickers: &Query<&mut ColorPicker>,
) -> Option<Entity> {
    core::iter::once(entity)
        .chain(parents.iter_ancestors(entity))
        .find(|e| pickers.contains(*e))
}

/// Set the picker's colour with `edit` and emit it from the picker root.
fn edit_picker(
    source: Entity,
    is_final: bool,
    parents: &Query<&ChildOf>,
    pickers: &mut Query<&mut ColorPicker>,
    commands: &mut Commands,
    edit: impl FnOnce(&mut Srgba),
) {
    let Some(root) = picker_root(source, parents, pickers) else {
        return;
    };
    let Ok(mut picker) = pickers.get_mut(root) else {
        return;
    };
    edit(&mut picker.rgba);
    commands.trigger(ValueChange {
        source: root,
        value: picker.color(),
        is_final,
    });
}

/// A bound picker's edit, written back to its field.
fn writeback_color(
    change: On<ValueChange<Color>>,
    bindings: Query<&InspectorBinding>,
    mut commands: Commands,
) {
    if let Ok(binding) = bindings.get(change.source) {
        queue_edit(&mut commands, binding, Box::new(change.value), change.is_final);
    }
}

fn on_plane_change(
    change: On<ValueChange<Vec2>>,
    parents: Query<&ChildOf>,
    mut pickers: Query<&mut ColorPicker>,
    mut commands: Commands,
) {
    let value = change.value;
    edit_picker(
        change.source,
        change.is_final,
        &parents,
        &mut pickers,
        &mut commands,
        |rgba| {
            rgba.red = value.x;
            rgba.blue = value.y;
        },
    );
}

fn on_channel_change(
    change: On<ValueChange<f32>>,
    parts: Query<&ColorPart>,
    parents: Query<&ChildOf>,
    mut pickers: Query<&mut ColorPicker>,
    mut commands: Commands,
) {
    let Ok(ColorPart::Channel(channel)) = parts.get(change.source).copied() else {
        return;
    };
    let value = change.value;
    edit_picker(
        change.source,
        change.is_final,
        &parents,
        &mut pickers,
        &mut commands,
        |rgba| match channel {
            0 => rgba.red = value,
            1 => rgba.green = value,
            2 => rgba.blue = value,
            _ => rgba.alpha = value,
        },
    );
}

/// Commit the hex text if it parses; otherwise the next refresh puts the colour's text back.
fn commit_hex(
    input: Entity,
    texts: &Query<&EditableText>,
    parents: &Query<&ChildOf>,
    pickers: &mut Query<&mut ColorPicker>,
    commands: &mut Commands,
) {
    let Ok(editable) = texts.get(input) else {
        return;
    };
    match Srgba::hex(editable.value().to_string()) {
        Ok(parsed) => edit_picker(input, true, parents, pickers, commands, |rgba| *rgba = parsed),
        Err(_) => {
            if let Some(root) = picker_root(input, parents, pickers)
                && let Ok(mut picker) = pickers.get_mut(root)
            {
                picker.set_changed();
            }
        }
    }
}

fn on_hex_key(
    key: On<FocusedInput<KeyboardInput>>,
    texts: Query<&EditableText>,
    parents: Query<&ChildOf>,
    mut pickers: Query<&mut ColorPicker>,
    mut focus: ResMut<InputFocus>,
    mut commands: Commands,
) {
    if key.input.state != ButtonState::Pressed || key.input.key_code != KeyCode::Enter {
        return;
    }
    let input = key.event_target();
    commit_hex(input, &texts, &parents, &mut pickers, &mut commands);
    if focus.get() == Some(input) {
        focus.clear();
    }
}

fn on_hex_focus_lost(
    lost: On<FocusLost>,
    texts: Query<&EditableText>,
    parents: Query<&ChildOf>,
    mut pickers: Query<&mut ColorPicker>,
    mut commands: Commands,
) {
    commit_hex(
        lost.event_target(),
        &texts,
        &parents,
        &mut pickers,
        &mut commands,
    );
}

/// Push a changed picker colour into its swatch, plane, sliders and (unless typed in) hex text.
pub(crate) fn refresh_color_pickers(
    pickers: Query<(Entity, &ColorPicker), Changed<ColorPicker>>,
    children: Query<&Children>,
    parts: Query<&ColorPart>,
    mut texts: Query<(&mut EditableText, &mut ShownText)>,
    focus: Res<InputFocus>,
    mut commands: Commands,
) {
    for (root, picker) in &pickers {
        let rgba = picker.rgba;
        let color = Color::Srgba(rgba);
        for entity in children.iter_descendants(root) {
            let Ok(part) = parts.get(entity) else {
                continue;
            };
            match *part {
                ColorPart::Swatch => {
                    commands.entity(entity).try_insert(ColorSwatchValue(color));
                }
                ColorPart::Plane => {
                    commands.entity(entity).try_insert(ColorPlaneValue(plane_value(rgba)));
                }
                ColorPart::Channel(channel) => {
                    let value = [rgba.red, rgba.green, rgba.blue, rgba.alpha][channel];
                    commands
                        .entity(entity)
                        .try_insert((SliderValue(value), SliderBaseColor(color)));
                }
                ColorPart::Hex => {
                    if focus.get() != Some(entity)
                        && let Ok((mut editable, mut shown)) = texts.get_mut(entity)
                    {
                        shown.text = rgba.to_hex();
                        replace_text(&mut editable, &shown.text);
                    }
                }
            }
        }
    }
}
