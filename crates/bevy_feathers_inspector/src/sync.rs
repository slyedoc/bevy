//! External-change sync: keep widgets showing the source value when it changes elsewhere (an
//! undo, a gizmo drag, the game itself).
//!
//! Each frame, for the roots whose data changed since the last pass (a component's or resource's
//! change tick; assets and custom roots are always read), it reads every bound widget's source
//! value back through reflection and, if it differs from what the widget shows (and the widget
//! isn't focused), pushes the new value in. Focused widgets are skipped so live editing is never
//! overwritten.

use bevy_color::Color;
use bevy_ecs::hierarchy::ChildOf;
use bevy_ecs::change_detection::Tick;
use bevy_ecs::prelude::*;
use bevy_input_focus::InputFocus;
use bevy_math::Quat;
use bevy_reflect::{ParsedPath, PartialReflect};
use bevy_text::EditableText;
use bevy_ui::Checked;
use bevy_ui_widgets::{Checkbox, NumericValue, SliderValue};

use crate::binding::{
    read_field, reflect_to_bool, reflect_to_f32, InspectorBinding, InspectorRoot,
};
use crate::color::ColorPicker;
use crate::number::{euler_degrees, EulerAxis, NumberField};
use crate::text::{replace_text, ShownText};

/// Push source values into bound widgets when they drift from the data.
pub fn sync_inspector_widgets(world: &mut World, mut last_run: Local<Option<Tick>>) {
    let this_run = world.change_tick();
    let since = last_run.replace(this_run);
    let focused = world.get_resource::<InputFocus>().and_then(InputFocus::get);
    let pass = Pass {
        focused,
        since,
        this_run,
    };

    sync_sliders(world, &pass);
    sync_checkboxes(world, &pass);
    sync_numbers(world, &pass);
    sync_euler(world, &pass);
    sync_text(world, &pass);
    sync_color_pickers(world, &pass);
    crate::val::sync_val_numbers(world, |world, root| pass.changed(world, root));
}

/// One sync pass: what has focus, and the tick window a root must have changed in.
struct Pass {
    focused: Option<Entity>,
    /// `None` on the first pass, which reads everything.
    since: Option<Tick>,
    this_run: Tick,
}

impl Pass {
    /// Whether `root`'s data may have changed since the last pass.
    fn changed(&self, world: &World, root: &InspectorRoot) -> bool {
        let Some(since) = self.since else {
            return true;
        };
        let (entity, type_id) = match root {
            InspectorRoot::Component { entity, type_id } => (*entity, *type_id),
            InspectorRoot::Resource { type_id } => {
                let Some(entity) = world
                    .components()
                    .get_id(*type_id)
                    .and_then(|id| world.resource_entities().get(id))
                else {
                    return true;
                };
                (entity, *type_id)
            }
            InspectorRoot::Asset { .. } | InspectorRoot::Custom { .. } => return true,
        };
        let Some(id) = world.components().get_id(type_id) else {
            return true;
        };
        world
            .get_entity(entity)
            .ok()
            .and_then(|entity| entity.get_change_ticks_by_id(id))
            .is_none_or(|ticks| ticks.is_changed(since, self.this_run))
    }
}

/// Whether the focused entity is `entity` or inside it (a number input's text, a picker's hex).
fn holds_focus(world: &World, focused: Option<Entity>, entity: Entity) -> bool {
    let Some(mut current) = focused else {
        return false;
    };
    loop {
        if current == entity {
            return true;
        }
        match world.get::<ChildOf>(current) {
            Some(child_of) => current = child_of.parent(),
            None => return false,
        }
    }
}

/// Every widget with `C` bound to a root that changed this pass, with the component.
fn bound<C: Component + Clone, F: bevy_ecs::query::QueryFilter>(
    world: &mut World,
    pass: &Pass,
) -> Vec<(Entity, InspectorRoot, ParsedPath, C)> {
    let mut query = world.query_filtered::<(Entity, &InspectorBinding, &C), F>();
    let found: Vec<_> = query
        .iter(world)
        .filter_map(|(entity, binding, c)| {
            Some((entity, binding.root?, binding.path.clone(), c.clone()))
        })
        .collect();
    found
        .into_iter()
        .filter(|(_, root, ..)| pass.changed(world, root))
        .collect()
}

fn sync_color_pickers(world: &mut World, pass: &Pass) {
    let focused = pass.focused;
    for (entity, root, path, picker) in bound::<ColorPicker, ()>(world, pass) {
        if holds_focus(world, focused, entity) {
            continue;
        }
        let source = read_field(world, &root, &path, |v| {
            v.try_as_reflect()
                .and_then(|r| r.downcast_ref::<Color>())
                .map(|c| *c)
        });
        if let Some(Some(color)) = source
            && picker.differs_from(color)
        {
            world.entity_mut(entity).insert(ColorPicker::from_color(color));
        }
    }
}

fn sync_sliders(world: &mut World, pass: &Pass) {
    let focused = pass.focused;
    for (entity, root, path, SliderValue(current)) in bound::<SliderValue, ()>(world, pass) {
        if Some(entity) == focused {
            continue;
        }
        if let Some(Some(source)) = read_field(world, &root, &path, reflect_to_f32)
            && (source - current).abs() > f32::EPSILON
        {
            world.entity_mut(entity).insert(SliderValue(source));
        }
    }
}

fn sync_checkboxes(world: &mut World, pass: &Pass) {
    let focused = pass.focused;
    let checkboxes: Vec<(Entity, InspectorRoot, ParsedPath, bool)> = {
        let mut query =
            world.query_filtered::<(Entity, &InspectorBinding, Has<Checked>), With<Checkbox>>();
        query
            .iter(world)
            .filter_map(|(entity, binding, checked)| {
                Some((entity, binding.root?, binding.path.clone(), checked))
            })
            .collect()
    };
    let checkboxes: Vec<_> = checkboxes
        .into_iter()
        .filter(|(_, root, ..)| pass.changed(world, root))
        .collect();
    for (entity, root, path, is_checked) in checkboxes {
        if Some(entity) == focused {
            continue;
        }
        if let Some(Some(source)) = read_field(world, &root, &path, reflect_to_bool)
            && source != is_checked
        {
            let mut entity_mut = world.entity_mut(entity);
            if source {
                entity_mut.insert(Checked);
            } else {
                entity_mut.remove::<Checked>();
            }
        }
    }
}

/// A reflected number as the number input holds it.
fn numeric(value: &dyn PartialReflect) -> Option<NumericValue> {
    let value = value.try_as_reflect()?;
    macro_rules! try_types {
        ($($t:ty),*) => {$(
            if let Some(v) = value.downcast_ref::<$t>() {
                return Some(v.numeric());
            }
        )*};
    }
    try_types!(f32, f64, i32, i64, u32, u64, usize);
    None
}

fn sync_numbers(world: &mut World, pass: &Pass) {
    let focused = pass.focused;
    for (entity, root, path, current) in bound::<NumericValue, Without<EulerAxis>>(world, pass) {
        if holds_focus(world, focused, entity) {
            continue;
        }
        if let Some(Some(source)) = read_field(world, &root, &path, numeric)
            && source != current
        {
            world.entity_mut(entity).insert(source);
        }
    }
}

fn sync_euler(world: &mut World, pass: &Pass) {
    let focused = pass.focused;
    for (entity, root, path, EulerAxis(axis)) in bound::<EulerAxis, ()>(world, pass) {
        if holds_focus(world, focused, entity) {
            continue;
        }
        let Some(Some(rotation)) = read_field(world, &root, &path, |v| {
            v.try_as_reflect()
                .and_then(|r| r.downcast_ref::<Quat>())
                .map(|q| *q)
        }) else {
            continue;
        };
        let degrees = euler_degrees(rotation)[axis];
        let shown = world.get::<NumericValue>(entity).copied();
        if !matches!(shown, Some(NumericValue::F32(v)) if (v - degrees).abs() < 1e-3) {
            world.entity_mut(entity).insert(NumericValue::F32(degrees));
        }
    }
}

fn sync_text(world: &mut World, pass: &Pass) {
    let focused = pass.focused;
    for (entity, root, path, shown) in bound::<ShownText, ()>(world, pass) {
        if Some(entity) == focused || !shown.seeded {
            continue;
        }
        let Some(Some(source)) = read_field(world, &root, &path, |v| {
            v.try_as_reflect()
                .and_then(|r| r.downcast_ref::<String>())
                .cloned()
        }) else {
            continue;
        };
        if source == shown.text {
            continue;
        }
        let mut entity_mut = world.entity_mut(entity);
        if let Some(mut editable) = entity_mut.get_mut::<EditableText>() {
            replace_text(&mut editable, &source);
        }
        if let Some(mut shown) = entity_mut.get_mut::<ShownText>() {
            shown.text = source;
        }
    }
}
