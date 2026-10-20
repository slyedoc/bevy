//! Numbers without a range: feathers' number input (drag to scrub, or type). Vectors are a row of
//! them, one per axis, each bound to its own component; a `Quat` is edited as XYZ Euler degrees.
//!
//! A field with a `#[reflect(@min..=max)]` range keeps the slider (see [`crate::widget`]).

use bevy_ecs::prelude::*;
use bevy_feathers::controls::{
    FeathersNumberInput, HardLimit, NumberInputPrecision, NumberInputStep,
};
use bevy_feathers::tokens;
use bevy_math::{EulerRot, Quat, Vec2, Vec3, Vec4};
use bevy_reflect::{CreateTypeData, ParsedPath, PartialReflect, ReflectRef};
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_ui::{px, Display, FlexDirection, Node};
use bevy_ui_widgets::{NumericValue, ValueChange};

use crate::attributes::FieldCtx;
use crate::binding::{apply_edit, queue_edit, read_field, InspectorBinding};
use crate::recurse::{parse_path, BuildCx};
use crate::widget::{build_slider, ReflectInspectorWidget, SliderScalar};

/// A numeric type the number input edits: the format it is shown in, and back.
pub trait NumberField: SliderScalar + Copy + Default {
    /// What the number input emits for this type's format.
    type Emit: Copy + Send + Sync + 'static;
    /// The value as the number input holds it.
    fn numeric(self) -> NumericValue;
    /// A value the number input emitted, as this type.
    fn from_emit(value: Self::Emit) -> Self;
    /// The range a typed or dragged value is clamped to.
    fn hard_limit() -> HardLimit;
    /// Decimal places shown.
    const DECIMALS: i32;
}

macro_rules! number_field {
    ($t:ty, $emit:ty, $variant:ident, $limit:expr, $decimals:expr, $from:expr) => {
        impl NumberField for $t {
            type Emit = $emit;
            fn numeric(self) -> NumericValue {
                NumericValue::$variant(self as $emit)
            }
            fn from_emit(value: $emit) -> Self {
                let from = $from;
                from(value)
            }
            fn hard_limit() -> HardLimit {
                $limit
            }
            const DECIMALS: i32 = $decimals;
        }
    };
}

number_field!(f32, f32, F32, HardLimit::f32(f32::MIN..=f32::MAX), 3, |v: f32| v);
number_field!(f64, f64, F64, HardLimit::f64(f64::MIN..=f64::MAX), 3, |v: f64| v);
number_field!(i32, i32, I32, HardLimit::i32(i32::MIN..=i32::MAX), 0, |v: i32| v);
number_field!(i64, i64, I64, HardLimit::i64(i64::MIN..=i64::MAX), 0, |v: i64| v);
number_field!(u32, i64, I64, HardLimit::i64(0..=u32::MAX as i64), 0, |v: i64| v.max(0) as u32);
number_field!(u64, i64, I64, HardLimit::i64(0..=i64::MAX), 0, |v: i64| v.max(0) as u64);
number_field!(usize, i64, I64, HardLimit::i64(0..=i64::MAX), 0, |v: i64| v.max(0) as usize);

/// The field's value in the number input's form, or `None` if it is not `T`.
fn numeric_of<T: NumberField>(value: &dyn PartialReflect) -> Option<NumericValue> {
    value
        .try_as_reflect()?
        .downcast_ref::<T>()
        .map(|v| v.numeric())
}

/// A number field: a slider when it declares a range, else a number input.
pub(crate) fn build_number<T: NumberField>(
    cx: &BuildCx,
    path: &str,
    value: &dyn PartialReflect,
    field: &FieldCtx,
) -> Box<dyn Scene> {
    if field.range.is_some() {
        return build_slider::<T>(cx, path, value, field);
    }
    let numeric = numeric_of::<T>(value).unwrap_or(T::default().numeric());
    let binding = InspectorBinding {
        root: Some(cx.root.clone()),
        path: parse_path(path),
    };
    let limit = T::hard_limit();
    let step = field.step.map(|s| bsn! { NumberInputStep({s as f64}) });
    Box::new((
        bsn! {
            @FeathersNumberInput
            NumberInputPrecision({T::DECIMALS})
            limit
            numeric
            binding
            Node { flex_grow: 1.0 }
            on(writeback_number::<T>)
        },
        step,
    ))
}

/// Writeback for a number input: show the new value (the input does not update itself), then
/// edit the field.
fn writeback_number<T: NumberField>(
    change: On<ValueChange<T::Emit>>,
    bindings: Query<&InspectorBinding>,
    mut commands: Commands,
) {
    let value = T::from_emit(change.value);
    commands.entity(change.source).insert(value.numeric());
    if let Ok(binding) = bindings.get(change.source) {
        queue_edit(&mut commands, binding, Box::new(value), change.is_final);
    }
}

macro_rules! number_widget {
    ($($t:ty),*) => {$(
        impl CreateTypeData<$t> for ReflectInspectorWidget {
            fn create_type_data(_: ()) -> Self {
                Self {
                    build: build_number::<$t>,
                }
            }
        }
    )*};
}

number_widget!(f32, f64, i32, i64, u32, u64, usize);

/// One axis input of a vector or rotation row.
fn axis_input(
    axis: usize,
    numeric: NumericValue,
    binding: InspectorBinding,
    step: Option<f32>,
) -> Box<dyn Scene> {
    let step = step.map(|s| bsn! { NumberInputStep({s as f64}) });
    macro_rules! input {
        ($token:expr, $label:expr) => {
            Box::new((
                bsn! {
                    @FeathersNumberInput { @sigil_color: $token, @label_text: $label }
                    NumberInputPrecision(3)
                    numeric
                    binding
                    Node { flex_grow: 1.0, min_width: px(0) }
                    on(writeback_number::<f32>)
                },
                step,
            ))
        };
    }
    match axis {
        0 => input!(tokens::TEXT_INPUT_X_AXIS, "X"),
        1 => input!(tokens::TEXT_INPUT_Y_AXIS, "Y"),
        2 => input!(tokens::TEXT_INPUT_Z_AXIS, "Z"),
        _ => input!(tokens::TEXT_INPUT_W_AXIS, "W"),
    }
}

fn axis_row(inputs: Vec<Box<dyn Scene>>) -> Box<dyn Scene> {
    Box::new(bsn! {
        Node {
            flex_grow: 1.0,
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            column_gap: px(4),
        }
        Children [ {inputs} ]
    })
}

/// A vector: one `f32` input per named component, each bound to `path.x`, `path.y`, ...
fn build_vector(
    cx: &BuildCx,
    path: &str,
    value: &dyn PartialReflect,
    step: Option<f32>,
) -> Box<dyn Scene> {
    let ReflectRef::Struct(components) = value.reflect_ref() else {
        return Box::new(bevy_feathers::display::label_dim(format!("{value:?}")));
    };
    let inputs = (0..components.field_len())
        .filter_map(|i| {
            let name = components.name_at(i)?;
            let current = components
                .field_at(i)?
                .try_as_reflect()?
                .downcast_ref::<f32>()
                .map(|v| *v)?;
            let binding = InspectorBinding {
                root: Some(cx.root.clone()),
                path: parse_path(&format!("{path}.{name}")),
            };
            Some(axis_input(i, NumericValue::F32(current), binding, step))
        })
        .collect();
    axis_row(inputs)
}

fn build_vec(
    cx: &BuildCx,
    path: &str,
    value: &dyn PartialReflect,
    field: &FieldCtx,
) -> Box<dyn Scene> {
    build_vector(cx, path, value, field.step)
}

macro_rules! vector_widget {
    ($($t:ty),*) => {$(
        impl CreateTypeData<$t> for ReflectInspectorWidget {
            fn create_type_data(_: ()) -> Self {
                Self { build: build_vec }
            }
        }
    )*};
}

vector_widget!(Vec2, Vec3, Vec4);

/// Which Euler angle (0 x, 1 y, 2 z) of a rotation an input edits.
#[derive(Component, Clone, Copy, Default)]
pub struct EulerAxis(pub usize);

pub(crate) const EULER: EulerRot = EulerRot::XYZ;

/// A rotation's Euler angles, in degrees to a thousandth (and never `-0`).
pub(crate) fn euler_degrees(rotation: Quat) -> [f32; 3] {
    let (x, y, z) = rotation.to_euler(EULER);
    [x, y, z].map(|a| (a.to_degrees() * 1000.0).round() / 1000.0 + 0.0)
}

fn build_quat(
    cx: &BuildCx,
    path: &str,
    value: &dyn PartialReflect,
    _field: &FieldCtx,
) -> Box<dyn Scene> {
    let rotation = value
        .try_as_reflect()
        .and_then(|r| r.downcast_ref::<Quat>())
        .map(|q| *q)
        .unwrap_or_default();
    let angles = euler_degrees(rotation);
    let inputs = (0..3)
        .map(|axis| {
            let binding = InspectorBinding {
                root: Some(cx.root.clone()),
                path: parse_path(path),
            };
            let numeric = NumericValue::F32(angles[axis]);
            let euler = EulerAxis(axis);
            macro_rules! input {
                ($token:expr, $label:expr) => {
                    Box::new(bsn! {
                        @FeathersNumberInput { @sigil_color: $token, @label_text: $label }
                        NumberInputPrecision(1)
                        numeric
                        binding
                        euler
                        Node { flex_grow: 1.0, min_width: px(0) }
                        on(writeback_euler)
                    }) as Box<dyn Scene>
                };
            }
            match axis {
                0 => input!(tokens::TEXT_INPUT_X_AXIS, "X"),
                1 => input!(tokens::TEXT_INPUT_Y_AXIS, "Y"),
                _ => input!(tokens::TEXT_INPUT_Z_AXIS, "Z"),
            }
        })
        .collect();
    axis_row(inputs)
}

impl CreateTypeData<Quat> for ReflectInspectorWidget {
    fn create_type_data(_: ()) -> Self {
        Self { build: build_quat }
    }
}

/// Writeback for one Euler angle: the rotation is rebuilt from the field's current angles with
/// this one replaced.
fn writeback_euler(
    change: On<ValueChange<f32>>,
    inputs: Query<(&InspectorBinding, &EulerAxis)>,
    mut commands: Commands,
) {
    commands
        .entity(change.source)
        .insert(NumericValue::F32(change.value));
    let Ok((binding, axis)) = inputs.get(change.source) else {
        return;
    };
    let Some(root) = binding.root.clone() else {
        return;
    };
    let path: ParsedPath = binding.path.clone();
    let (axis, degrees, is_final) = (axis.0, change.value, change.is_final);
    commands.queue(move |world: &mut World| {
        let Some(Some(rotation)) = read_field(world, &root, &path, |v| {
            v.try_as_reflect()
                .and_then(|r| r.downcast_ref::<Quat>())
                .map(|q| *q)
        }) else {
            return;
        };
        let mut angles = euler_degrees(rotation);
        angles[axis] = degrees;
        let rotation = Quat::from_euler(
            EULER,
            angles[0].to_radians(),
            angles[1].to_radians(),
            angles[2].to_radians(),
        );
        apply_edit(world, &root, &path, &rotation, is_final);
    });
}
