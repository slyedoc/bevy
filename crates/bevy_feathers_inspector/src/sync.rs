//! External-change sync: keep slider/checkbox widgets showing the source value when it changes
//! elsewhere.
//!
//! This is best-effort. Each frame it reads every bound widget's source value back through
//! reflection and, if it differs from what the widget shows (and the widget isn't focused), pushes
//! the new value in. Focused widgets are skipped so live editing is never overwritten.

use bevy_ecs::prelude::*;
use bevy_input_focus::InputFocus;
use bevy_reflect::ParsedPath;
use bevy_ui::Checked;
use bevy_ui_widgets::{Checkbox, SliderValue};

use crate::binding::{
    read_field, reflect_to_bool, reflect_to_f32, InspectorBinding, InspectorRoot,
};

/// Push source values into slider and checkbox widgets when they drift from the data.
pub fn sync_inspector_widgets(world: &mut World) {
    let focused = world.get_resource::<InputFocus>().and_then(InputFocus::get);

    sync_sliders(world, focused);
    sync_checkboxes(world, focused);
}

fn sync_sliders(world: &mut World, focused: Option<Entity>) {
    let sliders: Vec<(Entity, InspectorRoot, ParsedPath, f32)> = {
        let mut query = world.query::<(Entity, &InspectorBinding, &SliderValue)>();
        query
            .iter(world)
            .filter_map(|(entity, binding, value)| {
                Some((entity, binding.root.clone()?, binding.path.clone(), value.0))
            })
            .collect()
    };

    for (entity, root, path, current) in sliders {
        if Some(entity) == focused {
            continue;
        }
        if let Some(Some(source)) = read_field(world, &root, &path, reflect_to_f32)
            && (source - current).abs() > f32::EPSILON
            && let Ok(mut entity_mut) = world.get_entity_mut(entity)
        {
            entity_mut.insert(SliderValue(source));
        }
    }
}

fn sync_checkboxes(world: &mut World, focused: Option<Entity>) {
    let checkboxes: Vec<(Entity, InspectorRoot, ParsedPath, bool)> = {
        let mut query =
            world.query_filtered::<(Entity, &InspectorBinding, Has<Checked>), With<Checkbox>>();
        query
            .iter(world)
            .filter_map(|(entity, binding, checked)| {
                Some((entity, binding.root.clone()?, binding.path.clone(), checked))
            })
            .collect()
    };

    for (entity, root, path, is_checked) in checkboxes {
        if Some(entity) == focused {
            continue;
        }
        if let Some(Some(source)) = read_field(world, &root, &path, reflect_to_bool)
            && source != is_checked
            && let Ok(mut entity_mut) = world.get_entity_mut(entity)
        {
            if source {
                entity_mut.insert(Checked);
            } else {
                entity_mut.remove::<Checked>();
            }
        }
    }
}
