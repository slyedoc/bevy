//! Top-level entry points that bridge the ECS `World` to the recursion engine.
//!
//! [`build_entity_inspector`] enumerates an entity's reflectable components and spawns an editing
//! section per component under a panel entity. The panel is marked with [`InspectorPanel`] so
//! structural edits (enum variant switches, list add/remove) can [`rebuild_panel`] it.

use core::any::TypeId;

use bevy_asset::{ReflectAsset, UntypedAssetId};
use bevy_ecs::component::ComponentId;
use bevy_ecs::hierarchy::{ChildOf, Children};
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_ecs::system::Command;
use bevy_reflect::Reflect;
use bevy_scene::prelude::*;
use bevy_scene::Scene;

use crate::attributes::FieldCtx;
use crate::binding::{read_field, CustomRead, CustomWrite, InspectorRoot};
use crate::collapse::InspectorCollapsed;
use crate::recurse::{build_value, field_entry, group_card, parse_path, BuildCx};

/// Marks a panel entity built by the inspector, recording what it inspects so structural edits can
/// [`rebuild_panel`] it.
#[derive(Component, Clone)]
pub enum InspectorPanel {
    /// Shows all components of an entity.
    Entity(Entity),
    /// Shows one component of an entity.
    Component(Entity, TypeId),
    /// Shows the fields of the value at a path below a root, with no card around them, for a
    /// host that draws its own.
    Body(InspectorRoot, String),
    /// Shows one labeled field: a root, the path below it, and the label.
    Field(InspectorRoot, String, String),
    /// Shows a single resource.
    Resource(TypeId),
    /// Shows a single loaded asset.
    Asset(UntypedAssetId, TypeId),
    /// Shows whatever a pair of resolver functions reaches.
    Custom(CustomRead, CustomWrite),
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

    let sections = entity_component_sections(world, &registry, target);

    drop(registry);

    attach(world, panel, sections);
}

/// Build a titled section for each of an entity's reflectable components.
pub(crate) fn entity_component_sections(
    world: &World,
    registry: &bevy_reflect::TypeRegistry,
    entity: Entity,
) -> Vec<Box<dyn Scene>> {
    let component_ids: Vec<ComponentId> = match world.get_entity(entity) {
        Ok(entity_ref) => entity_ref.archetype().components().to_vec(),
        Err(_) => return Vec::new(),
    };
    let mut sections: Vec<Box<dyn Scene>> = Vec::new();
    for component_id in component_ids {
        let Some(type_id) = world
            .components()
            .get_info(component_id)
            .and_then(bevy_ecs::component::ComponentInfo::type_id)
        else {
            continue;
        };
        let Some(registration) = registry.get(type_id) else {
            continue;
        };
        let Some(reflect_component) = registration.data::<ReflectComponent>() else {
            continue;
        };
        let Some(reflected) = reflect_component.reflect(world.entity(entity)) else {
            continue;
        };
        let name = registration.type_info().ty().short_path();
        let root = InspectorRoot::Component { entity, type_id };
        sections.push(section_for(world, registry, root, name, reflected));
    }
    sections
}

/// (Re)build a single-component inspector as the sole child of `panel` —
/// embed one component's editor in your own UI instead of the whole entity.
pub fn build_component_inspector(
    world: &mut World,
    target: Entity,
    type_id: TypeId,
    panel: Entity,
) {
    clear_children(world, panel);
    if let Ok(mut panel_mut) = world.get_entity_mut(panel) {
        panel_mut.insert(InspectorPanel::Component(target, type_id));
    }

    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();

    let Some(section) = component_section(world, &registry, target, type_id) else {
        return;
    };
    drop(registry);

    attach(world, panel, vec![section]);
}

/// Build one component's titled section, or `None` if it isn't reflectable /
/// present on `target`.
pub(crate) fn component_section(
    world: &World,
    registry: &bevy_reflect::TypeRegistry,
    target: Entity,
    type_id: TypeId,
) -> Option<Box<dyn Scene>> {
    let registration = registry.get(type_id)?;
    let reflect_component = registration.data::<ReflectComponent>()?;
    let reflected = reflect_component.reflect(world.get_entity(target).ok()?)?;
    let name = registration.type_info().ty().short_path();
    let root = InspectorRoot::Component {
        entity: target,
        type_id,
    };
    Some(section_for(world, registry, root, name, reflected))
}

/// (Re)build the fields of the value at `root`, with no card, as the sole child of `panel`: for
/// an editor that frames each value itself.
pub fn build_body(world: &mut World, root: InspectorRoot, panel: Entity) {
    build_body_at(world, root, "", panel);
}

/// [`build_body`] for the value at `path` below `root` (one entry of a list, say).
pub fn build_body_at(world: &mut World, root: InspectorRoot, path: &str, panel: Entity) {
    clear_children(world, panel);
    if let Ok(mut panel_mut) = world.get_entity_mut(panel) {
        panel_mut.insert(InspectorPanel::Body(root, path.to_string()));
    }

    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let body = read_field(world, &root, &parse_path(path), |value| {
        let cx = BuildCx {
            registry: &registry,
            root,
            collapsed: world.resource::<InspectorCollapsed>(),
        };
        build_value(&cx, path, value, &FieldCtx::default())
    });
    drop(registry);

    if let Some(body) = body {
        attach(world, panel, vec![body]);
    }
}

/// Append one labeled field (a row, or a card for a compound value) for the value at `path` below
/// `root` under `parent`, beside whatever `parent` already holds: for an editor that lays a value
/// out by hand and wants the inspector's widget for one of its fields.
pub fn build_field(
    world: &mut World,
    root: InspectorRoot,
    path: &str,
    label: &str,
    parent: Entity,
) -> Entity {
    let panel = world
        .spawn((
            bevy_ui::Node {
                width: bevy_ui::percent(100),
                ..Default::default()
            },
            ChildOf(parent),
        ))
        .id();
    rebuild_field(world, root, path, label, panel);
    panel
}

fn rebuild_field(world: &mut World, root: InspectorRoot, path: &str, label: &str, panel: Entity) {
    clear_children(world, panel);
    if let Ok(mut panel_mut) = world.get_entity_mut(panel) {
        panel_mut.insert(InspectorPanel::Field(root, path.to_string(), label.to_string()));
    }
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let field = read_field(world, &root, &parse_path(path), |value| {
        let cx = BuildCx {
            registry: &registry,
            root,
            collapsed: world.resource::<InspectorCollapsed>(),
        };
        field_entry(&cx, label, path, value, &FieldCtx::default())
    });
    drop(registry);
    if let Some(field) = field {
        attach(world, panel, vec![field]);
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

    attach(world, panel, vec![section]);
}

/// Enumerate one loaded asset's fields and (re)build an editing section under `panel`.
///
/// The asset type must have been registered with `register_asset_reflect`; a plain `init_asset`
/// leaves no [`ReflectAsset`] to reach it through and this is a no-op.
pub fn build_asset_inspector(
    world: &mut World,
    asset_id: UntypedAssetId,
    type_id: TypeId,
    panel: Entity,
) {
    clear_children(world, panel);
    if let Ok(mut panel_mut) = world.get_entity_mut(panel) {
        panel_mut.insert(InspectorPanel::Asset(asset_id, type_id));
    }

    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();

    let Some(section) = asset_section(world, &registry, asset_id, type_id) else {
        return;
    };
    drop(registry);

    attach(world, panel, vec![section]);
}

/// Build a titled section for a loaded asset, or `None` if it isn't reflectable / loaded.
pub(crate) fn asset_section(
    world: &World,
    registry: &bevy_reflect::TypeRegistry,
    asset_id: UntypedAssetId,
    type_id: TypeId,
) -> Option<Box<dyn Scene>> {
    let registration = registry.get(type_id)?;
    let reflect_asset = registration.data::<ReflectAsset>()?;
    let reflected = reflect_asset.get(world, asset_id)?;
    let name = registration.type_info().ty().short_path();
    Some(section_for(
        world,
        registry,
        InspectorRoot::Asset { asset_id, type_id },
        name,
        reflected,
    ))
}

/// (Re)build an inspector over a value reached by resolver functions, as the sole child of
/// `panel`.
///
/// The escape hatch for data no [`InspectorRoot::Component`] / `Resource` / `Asset` can address:
/// a value behind a `#[reflect(ignore)]` field, an element keyed by something a reflection path
/// cannot spell, or a `dyn` trait object whose concrete type is only known at runtime. The
/// section is titled after the type the resolver actually yields, so it follows the selection
/// without the caller having to name it.
pub fn build_custom_inspector(
    world: &mut World,
    read: CustomRead,
    write: CustomWrite,
    panel: Entity,
) {
    clear_children(world, panel);
    if let Ok(mut panel_mut) = world.get_entity_mut(panel) {
        panel_mut.insert(InspectorPanel::Custom(read, write));
    }

    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();

    let mut section = None;
    read(world, &mut |reflected: &dyn Reflect| {
        section = Some(section_for(
            world,
            &registry,
            InspectorRoot::Custom { read, write },
            reflected.reflect_short_type_path(),
            reflected,
        ));
    });
    drop(registry);

    let Some(section) = section else {
        return;
    };
    attach(world, panel, vec![section]);
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
        world,
        registry,
        InspectorRoot::Resource { type_id },
        name,
        reflected,
    ))
}

/// Build one titled section (a boxed scene) for a reflected value reachable from `root`.
pub(crate) fn section_for(
    world: &World,
    registry: &bevy_reflect::TypeRegistry,
    root: InspectorRoot,
    name: &str,
    reflected: &dyn Reflect,
) -> Box<dyn Scene> {
    let cx = BuildCx {
        registry,
        root,
        collapsed: world.resource::<InspectorCollapsed>(),
    };
    let body = build_value(
        &cx,
        "",
        reflected.as_partial_reflect(),
        &FieldCtx::default(),
    );
    Box::new(group_card(&cx, "", name, body))
}

/// Rebuild a panel according to what its [`InspectorPanel`] records.
pub fn rebuild_panel(world: &mut World, panel: Entity) {
    let Some(kind) = world.get::<InspectorPanel>(panel).cloned() else {
        return;
    };
    match kind {
        InspectorPanel::Entity(target) => build_entity_inspector(world, target, panel),
        InspectorPanel::Component(target, type_id) => {
            build_component_inspector(world, target, type_id, panel);
        }
        InspectorPanel::Body(root, path) => build_body_at(world, root, &path, panel),
        InspectorPanel::Field(root, path, label) => {
            rebuild_field(world, root, &path, &label, panel);
        }
        InspectorPanel::Resource(type_id) => build_resource_inspector(world, type_id, panel),
        InspectorPanel::Asset(asset_id, type_id) => {
            build_asset_inspector(world, asset_id, type_id, panel);
        }
        InspectorPanel::Custom(read, write) => build_custom_inspector(world, read, write, panel),
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

/// Spawn `scenes` as children of `panel` now, in order. Not queued: a host that rebuilds (an
/// editor's inspector on a selection change) can despawn `panel` before a queued spawn lands,
/// and the orphaned rows would become UI roots drawn across the window.
fn attach(world: &mut World, panel: Entity, scenes: Vec<Box<dyn Scene>>) {
    if world.get_entity(panel).is_err() {
        return;
    }
    for scene in scenes {
        match world.spawn_scene(scene) {
            Ok(mut spawned) => {
                spawned.insert(ChildOf(panel));
            }
            Err(err) => bevy_log::warn!("inspector: a section did not spawn: {err}"),
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

/// Command form of [`build_component_inspector`].
pub struct BuildComponentInspector {
    /// The entity holding the component.
    pub target: Entity,
    /// The component type to inspect.
    pub type_id: TypeId,
    /// The panel entity the section is spawned under.
    pub panel: Entity,
}

impl Command for BuildComponentInspector {
    type Out = ();
    fn apply(self, world: &mut World) {
        build_component_inspector(world, self.target, self.type_id, self.panel);
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

/// Command form of [`build_custom_inspector`].
pub struct BuildCustomInspector {
    /// Resolves the value for reading.
    pub read: CustomRead,
    /// Resolves it for writing.
    pub write: CustomWrite,
    /// The panel entity the section is spawned under.
    pub panel: Entity,
}

impl Command for BuildCustomInspector {
    type Out = ();
    fn apply(self, world: &mut World) {
        build_custom_inspector(world, self.read, self.write, self.panel);
    }
}

/// Command form of [`build_asset_inspector`].
pub struct BuildAssetInspector {
    /// Which asset, within its collection.
    pub asset_id: UntypedAssetId,
    /// The asset type to inspect.
    pub type_id: TypeId,
    /// The panel entity the section is spawned under.
    pub panel: Entity,
}

impl Command for BuildAssetInspector {
    type Out = ();
    fn apply(self, world: &mut World) {
        build_asset_inspector(world, self.asset_id, self.type_id, self.panel);
    }
}
