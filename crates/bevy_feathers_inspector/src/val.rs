//! UI [`Val`] fields: a number input and a unit select on one row. `UiRect` and the other
//! structs of `Val`s need nothing more; they recurse into these.

use bevy_ecs::prelude::*;
use bevy_feathers::controls::{
    list_rows_from_strings, FeathersNumberInput, FeathersSelect, NumberInputPrecision,
    OptionIndex,
};
use bevy_reflect::{CreateTypeData, PartialReflect};
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_ui::{px, Display, FlexDirection, InteractionDisabled, Node, Val};
use bevy_ui_widgets::{NumericValue, ValueChange};

use crate::attributes::FieldCtx;
use crate::binding::{apply_edit, read_field, InspectorBinding};
use crate::recurse::{parse_path, BuildCx};
use crate::widget::ReflectInspectorWidget;

const UNITS: [&str; 9] = ["auto", "px", "%", "vw", "vh", "vmin", "vmax", "em", "rem"];

fn unit_of(val: Val) -> usize {
    match val {
        Val::Auto => 0,
        Val::Px(_) => 1,
        Val::Percent(_) => 2,
        Val::Vw(_) => 3,
        Val::Vh(_) => 4,
        Val::VMin(_) => 5,
        Val::VMax(_) => 6,
        Val::Em(_) => 7,
        Val::Rem(_) => 8,
    }
}

fn number_of(val: Val) -> f32 {
    match val {
        Val::Auto => 0.0,
        Val::Px(v)
        | Val::Percent(v)
        | Val::Vw(v)
        | Val::Vh(v)
        | Val::VMin(v)
        | Val::VMax(v)
        | Val::Em(v)
        | Val::Rem(v) => v,
    }
}

fn val_of(unit: usize, number: f32) -> Val {
    match unit {
        1 => Val::Px(number),
        2 => Val::Percent(number),
        3 => Val::Vw(number),
        4 => Val::Vh(number),
        5 => Val::VMin(number),
        6 => Val::VMax(number),
        7 => Val::Em(number),
        8 => Val::Rem(number),
        _ => Val::Auto,
    }
}

/// Which half of a `Val` row a widget is.
#[derive(Component, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum ValPart {
    #[default]
    Number,
    Unit,
}

fn read_val(value: &dyn PartialReflect) -> Option<Val> {
    value.try_as_reflect()?.downcast_ref::<Val>().map(|v| *v)
}

fn build_val(
    cx: &BuildCx,
    path: &str,
    value: &dyn PartialReflect,
    _field: &FieldCtx,
) -> Box<dyn Scene> {
    let current = read_val(value).unwrap_or_default();
    let binding = InspectorBinding {
        root: Some(cx.root),
        path: parse_path(path),
    };
    let number = NumericValue::F32(number_of(current));
    let number_binding = binding.clone();
    let rows = list_rows_from_strings(UNITS, Some(unit_of(current)));
    let number_input: Box<dyn Scene> = if current == Val::Auto {
        Box::new(bsn! {
            @FeathersNumberInput
            NumberInputPrecision(2)
            number
            number_binding
            ValPart::Number
            InteractionDisabled
            Node { flex_grow: 1.0, min_width: px(0) }
            on(writeback_number)
        })
    } else {
        Box::new(bsn! {
            @FeathersNumberInput
            NumberInputPrecision(2)
            number
            number_binding
            ValPart::Number
            Node { flex_grow: 1.0, min_width: px(0) }
            on(writeback_number)
        })
    };
    let children: Vec<Box<dyn Scene>> = vec![
        number_input,
        Box::new(bsn! {
            @FeathersSelect { @options: {rows} }
            binding
            ValPart::Unit
            Node { width: px(64) }
            on(writeback_unit)
        }),
    ];
    Box::new(bsn! {
        Node {
            flex_grow: 1.0,
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            column_gap: px(4),
        }
        Children [ {children} ]
    })
}

impl CreateTypeData<Val> for ReflectInspectorWidget {
    fn create_type_data(_: ()) -> Self {
        Self { build: build_val }
    }
}

/// Edit the `Val` a widget is bound to with `edit`, which sees the current one.
fn edit_val(
    commands: &mut Commands,
    binding: &InspectorBinding,
    is_final: bool,
    edit: impl FnOnce(Val) -> Val + Send + 'static,
) {
    let Some(root) = binding.root else {
        return;
    };
    let path = binding.path.clone();
    commands.queue(move |world: &mut World| {
        let Some(Some(current)) = read_field(world, &root, &path, read_val) else {
            return;
        };
        apply_edit(world, &root, &path, &edit(current), is_final);
    });
}

fn writeback_number(
    change: On<ValueChange<f32>>,
    bindings: Query<&InspectorBinding>,
    mut commands: Commands,
) {
    commands
        .entity(change.source)
        .insert(NumericValue::F32(change.value));
    if let Ok(binding) = bindings.get(change.source) {
        let number = change.value;
        edit_val(&mut commands, binding, change.is_final, move |current| {
            val_of(unit_of(current), number)
        });
    }
}

fn writeback_unit(
    change: On<ValueChange<Entity>>,
    bindings: Query<&InspectorBinding>,
    options: Query<&OptionIndex>,
    mut commands: Commands,
) {
    let (Ok(binding), Ok(OptionIndex(unit))) =
        (bindings.get(change.source), options.get(change.value))
    else {
        return;
    };
    let unit = *unit;
    edit_val(&mut commands, binding, true, move |current| {
        val_of(unit, number_of(current))
    });
}

/// Keep the number half showing its `Val`, and disabled while the unit is `auto`.
pub(crate) fn sync_val_numbers(
    world: &mut World,
    roots: impl Fn(&World, &crate::binding::InspectorRoot) -> bool,
) {
    let numbers: Vec<_> = world
        .query::<(Entity, &InspectorBinding, &ValPart, &NumericValue)>()
        .iter(world)
        .filter(|(_, _, part, _)| **part == ValPart::Number)
        .filter_map(|(entity, binding, _, shown)| {
            Some((entity, binding.root?, binding.path.clone(), *shown))
        })
        .collect();
    for (entity, root, path, shown) in numbers {
        if !roots(world, &root) {
            continue;
        }
        let Some(Some(val)) = read_field(world, &root, &path, read_val) else {
            continue;
        };
        let want = NumericValue::F32(number_of(val));
        let mut entity = world.entity_mut(entity);
        if shown != want {
            entity.insert(want);
        }
        match (val == Val::Auto, entity.contains::<InteractionDisabled>()) {
            (true, false) => {
                entity.insert(InteractionDisabled);
            }
            (false, true) => {
                entity.remove::<InteractionDisabled>();
            }
            _ => {}
        }
    }
}
