//! A turnkey, egui-style world inspector: a single toggleable panel with a collapsible tree of
//! entities and resources.
//!
//! Add [`WorldInspectorPlugin`] and press the toggle key (backtick by default) to open an overlay
//! listing every entity (filtered by `F`) and every reflectable resource. Each row has a disclosure
//! chevron; expanding an entity shows its component cards, expanding a resource shows its fields —
//! all editable, reusing the same widgets as the rest of the crate.
//!
//! Filtering works like `bevy-inspector-egui`: `WorldInspectorPlugin::new()` lists all entities,
//! while `WorldInspectorPlugin::<With<Foo>>::default()` lists only entities matching the filter.

use core::any::TypeId;
use core::marker::PhantomData;

use bevy_app::{Plugin, Startup, Update};
use bevy_ecs::hierarchy::Children;
use bevy_ecs::name::Name;
use bevy_ecs::prelude::*;
use bevy_ecs::query::QueryFilter;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectResource};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_input::keyboard::KeyCode;
use bevy_input::ButtonInput;
use bevy_platform::collections::HashSet;
use bevy_reflect::TypeRegistry;
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_ui::{
    percent, px, AlignItems, Checked, Display, FlexDirection, Node, Overflow, PositionType, UiRect,
};
use bevy_ui_widgets::ValueChange;

use bevy_feathers::controls::FeathersDisclosureToggle;
use bevy_feathers::display::{label, label_dim};
use bevy_feathers::theme::ThemeBackgroundColor;
use bevy_feathers::tokens;

use crate::entry::{
    clear_children, entity_component_sections, resource_section,
};

/// Toggle + expansion state of the world inspector.
#[derive(Resource)]
pub struct WorldInspectorState {
    /// Whether the panel is currently shown.
    pub open: bool,
    /// Set when the tree needs to be rebuilt (toggled, expanded/collapsed).
    needs_rebuild: bool,
    expanded_entities: HashSet<Entity>,
    expanded_resources: HashSet<TypeId>,
}

impl Default for WorldInspectorState {
    fn default() -> Self {
        Self {
            open: false,
            needs_rebuild: true,
            expanded_entities: HashSet::default(),
            expanded_resources: HashSet::default(),
        }
    }
}

/// Cache of the last-built entity list, to avoid rebuilding every frame.
#[derive(Resource, Default)]
struct WorldInspectorCache {
    entities: Vec<Entity>,
}

/// Key that toggles the panel (if any).
#[derive(Resource)]
struct WorldInspectorConfig {
    toggle_key: Option<KeyCode>,
}

/// Marks the root overlay entity of the world inspector.
#[derive(Component)]
struct WorldInspectorRoot;

/// What a disclosure toggle expands/collapses.
#[derive(Component, Clone, Default)]
enum DisclosureTarget {
    #[default]
    None,
    Entity(Entity),
    Resource(TypeId),
}

/// A world inspector, optionally filtered by query `F` (default: all entities).
///
/// By default there is no toggle key — the panel is driven by [`WorldInspectorState::open`]. Call
/// [`with_toggle_key`](Self::with_toggle_key) to bind a key that flips it:
///
/// ```ignore
/// // All entities, toggle with F12:
/// WorldInspectorPlugin::new().with_toggle_key(KeyCode::F12)
/// // Only named entities, no toggle key (open it yourself via `WorldInspectorState::open`):
/// WorldInspectorPlugin::<With<Name>>::default()
/// ```
pub struct WorldInspectorPlugin<F: QueryFilter = ()> {
    toggle_key: Option<KeyCode>,
    _filter: PhantomData<fn() -> F>,
}

impl<F: QueryFilter> Default for WorldInspectorPlugin<F> {
    fn default() -> Self {
        Self {
            toggle_key: None,
            _filter: PhantomData,
        }
    }
}

impl WorldInspectorPlugin<()> {
    /// A world inspector listing all entities (no toggle key by default).
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: QueryFilter> WorldInspectorPlugin<F> {
    /// Bind a key that toggles the panel. Without this, drive [`WorldInspectorState::open`]
    /// yourself.
    pub fn with_toggle_key(mut self, key: KeyCode) -> Self {
        self.toggle_key = Some(key);
        self
    }
}

impl<F: QueryFilter + Send + Sync + 'static> Plugin for WorldInspectorPlugin<F> {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_resource::<WorldInspectorState>();
        app.init_resource::<WorldInspectorCache>();
        // The component cards this panel embeds are collapsible on their own.
        app.init_resource::<crate::collapse::InspectorCollapsed>();
        app.insert_resource(WorldInspectorConfig {
            toggle_key: self.toggle_key,
        });
        app.add_systems(Startup, spawn_root);
        app.add_systems(
            Update,
            (toggle_open, world_inspector_rebuild::<F>).chain(),
        );
    }
}

/// Spawn the (initially hidden) overlay panel.
fn spawn_root(mut commands: Commands) {
    commands.spawn((
        WorldInspectorRoot,
        bevy_ui_widgets::ScrollArea,
        Node {
            position_type: PositionType::Absolute,
            top: px(8),
            right: px(8),
            width: px(380),
            max_height: percent(92),
            display: Display::None,
            flex_direction: FlexDirection::Column,
            row_gap: px(4),
            padding: UiRect::all(px(8)),
            overflow: Overflow::scroll_y(),
            ..Default::default()
        },
        ThemeBackgroundColor(tokens::WINDOW_BG),
    ));
}

/// Flip `open` on the toggle key.
fn toggle_open(
    keys: Res<ButtonInput<KeyCode>>,
    config: Res<WorldInspectorConfig>,
    mut state: ResMut<WorldInspectorState>,
) {
    if let Some(key) = config.toggle_key
        && keys.just_pressed(key)
    {
        state.open = !state.open;
        state.needs_rebuild = true;
    }
}

/// Show/hide the panel and rebuild its tree when the entity set or expansion state changes.
fn world_inspector_rebuild<F: QueryFilter + 'static>(world: &mut World) {
    let Some(root) = world
        .query_filtered::<Entity, With<WorldInspectorRoot>>()
        .iter(world)
        .next()
    else {
        return;
    };

    let open = world.resource::<WorldInspectorState>().open;
    if let Some(mut node) = world.get_mut::<Node>(root) {
        node.display = if open {
            Display::Flex
        } else {
            Display::None
        };
    }
    if !open {
        return;
    }

    // Exclude the inspector's own UI entities so listing them doesn't churn the rebuild.
    let inspector = collect_descendants(world, root);
    let entities: Vec<Entity> = {
        let mut query = world.query_filtered::<Entity, F>();
        query
            .iter(world)
            .filter(|e| !inspector.contains(e))
            .collect()
    };

    let needs = world.resource::<WorldInspectorState>().needs_rebuild;
    let unchanged = world.resource::<WorldInspectorCache>().entities == entities;
    if !needs && unchanged {
        return;
    }

    build_tree(world, root, &entities);
    world.resource_mut::<WorldInspectorCache>().entities = entities;
    world.resource_mut::<WorldInspectorState>().needs_rebuild = false;
}

/// Rebuild the tree contents under `root`.
fn build_tree(world: &mut World, root: Entity, entities: &[Entity]) {
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();

    let (expanded_entities, expanded_resources) = {
        let state = world.resource::<WorldInspectorState>();
        (
            state.expanded_entities.clone(),
            state.expanded_resources.clone(),
        )
    };

    let mut children: Vec<Box<dyn Scene>> = Vec::new();
    children.push(Box::new(label("World Inspector".to_string())));

    // Entities.
    children.push(Box::new(label_dim("Entities".to_string())));
    for &entity in entities {
        let name = entity_label(world, entity);
        let expanded = expanded_entities.contains(&entity);
        children.push(disclosure_row(DisclosureTarget::Entity(entity), name, expanded));
        if expanded {
            for section in entity_component_sections(world, &registry, entity) {
                children.push(indent(section));
            }
        }
    }

    // Resources.
    children.push(Box::new(label_dim("Resources".to_string())));
    for type_id in reflectable_resources(world, &registry) {
        let name = registry
            .get(type_id)
            .map(|r| r.type_info().ty().short_path().to_string())
            .unwrap_or_default();
        let expanded = expanded_resources.contains(&type_id);
        children.push(disclosure_row(
            DisclosureTarget::Resource(type_id),
            name,
            expanded,
        ));
        if expanded
            && let Some(section) = resource_section(world, &registry, type_id)
        {
            children.push(indent(section));
        }
    }

    drop(registry);

    clear_children(world, root);
    if let Ok(root_mut) = world.get_entity_mut(root) {
        root_mut.queue_spawn_related_scenes::<Children>(children);
    }
}

/// A disclosure toggle + label row.
fn disclosure_row(target: DisclosureTarget, name: String, expanded: bool) -> Box<dyn Scene> {
    let toggle: Box<dyn Scene> = if expanded {
        Box::new((
            <FeathersDisclosureToggle as SceneComponent>::scene(()),
            template_value(Checked),
            template_value(target),
            on(on_disclosure),
        ))
    } else {
        Box::new((
            <FeathersDisclosureToggle as SceneComponent>::scene(()),
            template_value(target),
            on(on_disclosure),
        ))
    };
    let children: Vec<Box<dyn Scene>> = vec![toggle, Box::new(label(name))];
    Box::new(bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(4),
            min_height: px(24),
        }
        Children [ {children} ]
    })
}

/// Observer: flip the expansion state for the disclosure's target.
fn on_disclosure(
    change: On<ValueChange<bool>>,
    targets: Query<&DisclosureTarget>,
    mut state: ResMut<WorldInspectorState>,
) {
    let Ok(target) = targets.get(change.source) else {
        return;
    };
    match target {
        DisclosureTarget::Entity(entity) => {
            if change.value {
                state.expanded_entities.insert(*entity);
            } else {
                state.expanded_entities.remove(entity);
            }
        }
        DisclosureTarget::Resource(type_id) => {
            if change.value {
                state.expanded_resources.insert(*type_id);
            } else {
                state.expanded_resources.remove(type_id);
            }
        }
        DisclosureTarget::None => {}
    }
    state.needs_rebuild = true;
}

/// Indent a scene under a disclosure row.
fn indent(scene: Box<dyn Scene>) -> Box<dyn Scene> {
    let content: Vec<Box<dyn Scene>> = vec![scene];
    Box::new(bsn! {
        Node {
            width: percent(100),
            padding: {UiRect::left(px(18))},
        }
        Children [ {content} ]
    })
}

/// The display name for an entity (its `Name`, else `Entity <index>`).
fn entity_label(world: &World, entity: Entity) -> String {
    world
        .get::<Name>(entity)
        .map(|name| name.as_str().to_string())
        .unwrap_or_else(|| format!("Entity {}", entity.index()))
}

/// All registered resource types that are reflectable and currently present, sorted by name.
fn reflectable_resources(world: &World, registry: &TypeRegistry) -> Vec<TypeId> {
    let mut resources: Vec<(TypeId, &'static str)> = registry
        .iter()
        .filter(|reg| reg.data::<ReflectResource>().is_some())
        .map(|reg| (reg.type_id(), reg.type_info().ty().short_path()))
        .filter(|(type_id, _)| {
            world
                .components()
                .get_id(*type_id)
                .and_then(|id| world.resource_entities().get(id))
                .is_some()
        })
        .collect();
    resources.sort_by_key(|(_, name)| *name);
    resources.into_iter().map(|(type_id, _)| type_id).collect()
}

/// Collect `root` and all of its descendants.
fn collect_descendants(world: &World, root: Entity) -> HashSet<Entity> {
    let mut set = HashSet::default();
    let mut stack = vec![root];
    while let Some(entity) = stack.pop() {
        if !set.insert(entity) {
            continue;
        }
        if let Some(children) = world.get::<Children>(entity) {
            stack.extend(children.iter());
        }
    }
    set
}
