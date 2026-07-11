//! The world inspector: a browsable list of named entities.
//!
//! Building a full entity/resource tree in one panel is unwieldy, so the world panel lists named
//! entities as buttons; clicking one builds that entity's inspector into a separate detail panel
//! (pointed at by the [`InspectorDetailPanel`] resource). Resources are inspected directly with
//! [`build_resource_inspector`](crate::entry::build_resource_inspector).

use bevy_ecs::hierarchy::Children;
use bevy_ecs::name::Name;
use bevy_ecs::prelude::*;
use bevy_ecs::system::Command;
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_ui_widgets::Activate;

use bevy_feathers::controls::{FeathersButton, FeathersButtonProps};
use bevy_feathers::display::label;

use crate::entry::{build_entity_inspector, clear_children, InspectorPanel};

/// Points at the panel where a selected entity's inspector is built.
#[derive(Resource, Clone, Copy)]
pub struct InspectorDetailPanel(pub Entity);

/// Records which entity a world-panel button selects.
#[derive(Component, Clone, Copy, Default)]
pub(crate) struct EntitySelectButton {
    entity: Option<Entity>,
}

/// (Re)build the world panel: a button per named entity.
pub fn build_world_panel(world: &mut World, panel: Entity) {
    clear_children(world, panel);
    if let Ok(mut panel_mut) = world.get_entity_mut(panel) {
        panel_mut.insert(InspectorPanel::World);
    }

    let mut named: Vec<(Entity, String)> = {
        let mut query = world.query::<(Entity, &Name)>();
        query
            .iter(world)
            .map(|(entity, name)| (entity, name.as_str().to_string()))
            .collect()
    };
    named.sort_by(|a, b| a.1.cmp(&b.1));

    let mut children: Vec<Box<dyn Scene>> = vec![Box::new(label("Entities".to_string()))];
    for (entity, name) in named {
        let caption: Box<dyn SceneList> =
            Box::new(vec![Box::new(label(name)) as Box<dyn Scene>]);
        children.push(Box::new((
            <FeathersButton as SceneComponent>::scene(FeathersButtonProps {
                caption,
                ..Default::default()
            }),
            template_value(EntitySelectButton {
                entity: Some(entity),
            }),
            on(on_entity_select),
        )));
    }

    if let Ok(panel_mut) = world.get_entity_mut(panel) {
        panel_mut.queue_spawn_related_scenes::<Children>(children);
    }
}

/// Observer: build the clicked entity's inspector into the detail panel.
fn on_entity_select(
    activate: On<Activate>,
    buttons: Query<&EntitySelectButton>,
    detail: Option<Res<InspectorDetailPanel>>,
    mut commands: Commands,
) {
    let Ok(button) = buttons.get(activate.event_target()) else {
        return;
    };
    let Some(entity) = button.entity else {
        return;
    };
    let Some(detail) = detail.map(|d| d.0) else {
        return;
    };
    commands.queue(move |world: &mut World| {
        build_entity_inspector(world, entity, detail);
    });
}

/// Command form of [`build_world_panel`].
pub struct BuildWorldInspector {
    /// The panel entity to build the entity list under.
    pub panel: Entity,
}

impl Command for BuildWorldInspector {
    type Out = ();
    fn apply(self, world: &mut World) {
        build_world_panel(world, self.panel);
    }
}
