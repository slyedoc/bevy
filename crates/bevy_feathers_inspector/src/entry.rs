//! Top-level entry points that bridge the ECS `World` to the recursion engine.
//!
//! [`build_entity_inspector`] enumerates an entity's reflectable components and spawns an editing
//! section per component under a panel entity. The panel is marked with [`InspectorPanel`] so
//! structural edits (enum variant switches, list add/remove) can [`rebuild_panel`] it.

use core::any::TypeId;

use bevy_ecs::component::ComponentId;
use bevy_ecs::hierarchy::{ChildOf, Children};
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_ecs::system::Command;
use bevy_reflect::Reflect;
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_ui::{px, Display, FlexDirection, Node};

use bevy_feathers::display::label;

use crate::attributes::FieldCtx;
use crate::binding::InspectorRoot;
use crate::recurse::{build_value, BuildCx};

/// Marks a panel entity built by the inspector, recording what it inspects so structural edits can
/// [`rebuild_panel`] it.
#[derive(Component, Clone, Copy)]
pub enum InspectorPanel {
    /// Shows all components of an entity.
    Entity(Entity),
    /// Shows a single resource.
    Resource(TypeId),
    /// The world inspector (entities + resources).
    World,
}

/// Enumerate `target`'s reflectable components and (re)build an editing section per component as
/// children of `panel`. Idempotent: existing panel children are cleared first.
pub fn build_entity_inspector(world: &mut World, target: Entity, panel: Entity) {
    clear_children(world, panel);
    if let Ok(mut panel_mut) = world.get_entity_mut(panel) {
        panel_mut.insert(InspectorPanel::Entity(target));
    }

    // Clone the `Arc` so the read guard does not borrow `world`.
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();

    let component_ids: Vec<ComponentId> = match world.get_entity(target) {
        Ok(entity_ref) => entity_ref.archetype().components().to_vec(),
        Err(_) => return,
    };

    let mut sections: Vec<Box<dyn Scene>> = Vec::new();
    for component_id in component_ids {
        let Some(type_id) = world
            .components()
            .get_info(component_id)
            .and_then(|info| info.type_id())
        else {
            continue;
        };
        let Some(registration) = registry.get(type_id) else {
            continue;
        };
        let Some(reflect_component) = registration.data::<ReflectComponent>() else {
            continue;
        };
        let Some(reflected) = reflect_component.reflect(world.entity(target)) else {
            continue;
        };
        let name = registration.type_info().ty().short_path();
        let root = InspectorRoot::Component {
            entity: target,
            type_id,
        };
        sections.push(section_for(&registry, root, name, reflected));
    }

    drop(registry);

    if let Ok(panel_mut) = world.get_entity_mut(panel) {
        panel_mut.queue_spawn_related_scenes::<Children>(sections);
    }
}

/// (Re)build a single-resource inspector as the sole child of `panel`.
pub fn build_resource_inspector(world: &mut World, type_id: TypeId, panel: Entity) {
    clear_children(world, panel);
    if let Ok(mut panel_mut) = world.get_entity_mut(panel) {
        panel_mut.insert(InspectorPanel::Resource(type_id));
    }

    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();

    let Some(section) = resource_section(world, &registry, type_id) else {
        return;
    };
    drop(registry);

    if let Ok(panel_mut) = world.get_entity_mut(panel) {
        panel_mut.queue_spawn_related_scenes::<Children>(vec![section]);
    }
}

/// Build a titled section for a resource, or `None` if it isn't reflectable / present.
pub(crate) fn resource_section(
    world: &World,
    registry: &bevy_reflect::TypeRegistry,
    type_id: TypeId,
) -> Option<Box<dyn Scene>> {
    let registration = registry.get(type_id)?;
    let reflect_component = registration.data::<ReflectComponent>()?;
    let resource_entity = world
        .components()
        .get_id(type_id)
        .and_then(|id| world.resource_entities().get(id))?;
    let reflected = reflect_component.reflect(world.entity(resource_entity))?;
    let name = registration.type_info().ty().short_path();
    Some(section_for(
        registry,
        InspectorRoot::Resource { type_id },
        name,
        reflected,
    ))
}

/// Build one titled section (a boxed scene) for a reflected value reachable from `root`.
pub(crate) fn section_for(
    registry: &bevy_reflect::TypeRegistry,
    root: InspectorRoot,
    name: &str,
    reflected: &dyn Reflect,
) -> Box<dyn Scene> {
    let cx = BuildCx { registry, root };
    let body = build_value(&cx, "", reflected.as_partial_reflect(), &FieldCtx::default());
    Box::new(section(name, body))
}

/// Rebuild a panel according to what its [`InspectorPanel`] records.
pub fn rebuild_panel(world: &mut World, panel: Entity) {
    let Some(kind) = world.get::<InspectorPanel>(panel).copied() else {
        return;
    };
    match kind {
        InspectorPanel::Entity(target) => build_entity_inspector(world, target, panel),
        InspectorPanel::Resource(type_id) => build_resource_inspector(world, type_id, panel),
        InspectorPanel::World => crate::world_panel::build_world_panel(world, panel),
    }
}

/// Walk up the hierarchy from `entity` to find the enclosing [`InspectorPanel`].
pub fn find_ancestor_panel(
    entity: Entity,
    parents: &Query<&ChildOf>,
    panels: &Query<&InspectorPanel>,
) -> Option<Entity> {
    let mut current = entity;
    loop {
        if panels.contains(current) {
            return Some(current);
        }
        match parents.get(current) {
            Ok(child_of) => current = child_of.parent(),
            Err(_) => return None,
        }
    }
}

/// Despawn all children of `panel` (recursively).
pub(crate) fn clear_children(world: &mut World, panel: Entity) {
    let children: Vec<Entity> = world
        .get::<Children>(panel)
        .map(|c| c.iter().collect())
        .unwrap_or_default();
    for child in children {
        if let Ok(entity_mut) = world.get_entity_mut(child) {
            entity_mut.despawn();
        }
    }
}

/// A titled section wrapping one component's fields.
fn section(title: &str, body: Box<dyn Scene>) -> impl Scene {
    let children: Vec<Box<dyn Scene>> = vec![Box::new(label(title.to_string())), body];
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(4),
            padding: px(6),
        }
        Children [ {children} ]
    }
}

/// Command form of [`build_entity_inspector`], for queuing from a non-exclusive system.
pub struct BuildEntityInspector {
    /// The entity whose components will be inspected.
    pub target: Entity,
    /// The panel entity the inspector sections are spawned under.
    pub panel: Entity,
}

impl Command for BuildEntityInspector {
    type Out = ();
    fn apply(self, world: &mut World) {
        build_entity_inspector(world, self.target, self.panel);
    }
}

/// Command form of [`build_resource_inspector`].
pub struct BuildResourceInspector {
    /// The resource type to inspect.
    pub type_id: TypeId,
    /// The panel entity the section is spawned under.
    pub panel: Entity,
}

impl Command for BuildResourceInspector {
    type Out = ();
    fn apply(self, world: &mut World) {
        build_resource_inspector(world, self.type_id, self.panel);
    }
}
