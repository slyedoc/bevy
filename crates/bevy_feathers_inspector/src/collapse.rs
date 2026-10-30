//! Collapsible group cards.
//!
//! Every card the inspector builds — a component section, or a nested struct/enum/list field — gets
//! a disclosure chevron in its header that hides the card's body. Clicking it, or anywhere on the
//! header (a far easier target for a laser pointer), flips the body's [`Display`] in place, so the
//! widgets inside keep their state and nothing is rebuilt.
//!
//! The collapsed set lives in [`InspectorCollapsed`], keyed by the card's [`InspectorRoot`] plus its
//! reflection path, so a card stays collapsed across the panel rebuilds that structural edits (enum
//! variant switches, list add/remove) trigger.

use bevy_ecs::hierarchy::{ChildOf, Children};
use bevy_ecs::prelude::*;
use bevy_picking::events::PointerClick;
use bevy_platform::collections::HashSet;
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_ui::{Checked, Display, Node};
use bevy_ui_widgets::{checkbox_self_update, ValueChange};

use bevy_feathers::controls::FeathersDisclosureToggle;

use crate::binding::InspectorRoot;

/// Which inspector cards are currently collapsed.
///
/// Insert or clear entries to drive collapse from code — cards read this when they are built, so
/// changes land on the next panel rebuild.
#[derive(Resource, Default)]
pub struct InspectorCollapsed {
    collapsed: HashSet<(InspectorRoot, String)>,
}

impl InspectorCollapsed {
    /// Whether the card at `root`/`path` is collapsed (`path` is empty for a component's own card).
    pub fn is_collapsed(&self, root: &InspectorRoot, path: &str) -> bool {
        self.collapsed.contains(&(root.clone(), path.to_string()))
    }

    /// Collapse or expand the card at `root`/`path`.
    pub fn set(&mut self, root: &InspectorRoot, path: &str, collapsed: bool) {
        let key = (root.clone(), path.to_string());
        if collapsed {
            self.collapsed.insert(key);
        } else {
            self.collapsed.remove(&key);
        }
    }
}

/// Marks a group card root, so its header's toggle can find the body to hide.
#[derive(Component, Default, Clone)]
pub(crate) struct CollapseGroup;

/// Marks the body of a group card — the part the disclosure toggle hides.
#[derive(Component, Default, Clone)]
pub(crate) struct CollapseBody;

/// On a card's disclosure toggle: which card it collapses.
#[derive(Component, Clone, Default)]
pub(crate) struct CollapseToggle {
    root: Option<InspectorRoot>,
    path: String,
}

/// The disclosure chevron for a card's header. `Checked` means expanded.
pub(crate) fn collapse_toggle(root: &InspectorRoot, path: &str, collapsed: bool) -> Box<dyn Scene> {
    let toggle = CollapseToggle {
        root: Some(root.clone()),
        path: path.to_string(),
    };
    // `checkbox_self_update` flips `Checked` (which rotates the chevron); our observer hides the
    // body and records the state. A checked/unchecked branch avoids an `Option<Scene>`.
    if collapsed {
        Box::new(bsn! {
            @FeathersDisclosureToggle
            toggle
            on(checkbox_self_update)
            on(on_collapse_toggle)
        })
    } else {
        Box::new(bsn! {
            @FeathersDisclosureToggle
            Checked
            toggle
            on(checkbox_self_update)
            on(on_collapse_toggle)
        })
    }
}

/// Observer on a card's header: a click anywhere on it flips the header's chevron, as a click on
/// the chevron does (that one stops at the chevron, so it never flips twice).
pub(crate) fn on_header_click(
    click: On<PointerClick>,
    children: Query<&Children>,
    toggles: Query<Has<Checked>, With<CollapseToggle>>,
    mut commands: Commands,
) {
    let Some((toggle, checked)) = children
        .get(click.entity)
        .into_iter()
        .flatten()
        .find_map(|&child| toggles.get(child).ok().map(|checked| (child, checked)))
    else {
        return;
    };
    commands.trigger(ValueChange {
        source: toggle,
        value: !checked,
        is_final: true,
    });
}

/// Observer: show/hide the toggled card's body and remember the new state.
fn on_collapse_toggle(
    change: On<ValueChange<bool>>,
    toggles: Query<&CollapseToggle>,
    parents: Query<&ChildOf>,
    children: Query<&Children>,
    groups: Query<(), With<CollapseGroup>>,
    mut bodies: Query<&mut Node, With<CollapseBody>>,
    mut collapsed: ResMut<InspectorCollapsed>,
) {
    let Ok(toggle) = toggles.get(change.source) else {
        return;
    };
    let expanded = change.value;
    if let Some(root) = &toggle.root {
        collapsed.set(root, &toggle.path, !expanded);
    }

    // Only the card's own body: nested cards are further down, not direct children of this group.
    let Some(group) = ancestor_group(change.source, &parents, &groups) else {
        return;
    };
    let Ok(group_children) = children.get(group) else {
        return;
    };
    for child in group_children.iter() {
        if let Ok(mut node) = bodies.get_mut(child) {
            let want = if expanded {
                Display::Flex
            } else {
                Display::None
            };
            if node.display != want {
                node.display = want;
            }
        }
    }
}

/// Walk up from a toggle to the [`CollapseGroup`] card that encloses it.
fn ancestor_group(
    entity: Entity,
    parents: &Query<&ChildOf>,
    groups: &Query<(), With<CollapseGroup>>,
) -> Option<Entity> {
    let mut current = entity;
    loop {
        if groups.contains(current) {
            return Some(current);
        }
        current = parents.get(current).ok()?.parent();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::App;
    use bevy_math::Vec2;
    use bevy_picking::{
        backend::HitData,
        events::Pointer,
        pointer::{Location, PointerButton, PointerId},
    };
    use bevy_ui_widgets::{Checkbox, CheckboxPlugin};
    use bevy_window::{PrimaryWindow, Window, WindowRef};
    use core::any::TypeId;

    fn click(app: &mut App, entity: Entity, window: Entity) {
        let location = Location {
            target: bevy_camera::NormalizedRenderTarget::Window(
                WindowRef::Entity(window).normalize(Some(window)).unwrap(),
            ),
            position: Vec2::ZERO,
        };
        app.world_mut().trigger(PointerClick {
            entity,
            pointer: Pointer::new(PointerId::Mouse, location),
            button: PointerButton::Primary,
            hit: HitData::new(window, 0.0, None, None),
            duration: core::time::Duration::from_millis(10),
            count: 1,
        });
        app.update();
    }

    /// A click anywhere on a card's header collapses or expands it, and a click on the chevron
    /// flips it once, not once for the chevron and again for the header.
    #[test]
    fn the_whole_header_toggles_its_card() {
        let mut app = App::new();
        app.add_plugins(CheckboxPlugin)
            .init_resource::<InspectorCollapsed>();
        let window = app
            .world_mut()
            .spawn((Window::default(), PrimaryWindow))
            .id();
        let root = InspectorRoot::Resource {
            type_id: TypeId::of::<u32>(),
        };
        let header = app.world_mut().spawn_empty().observe(on_header_click).id();
        let toggle = app
            .world_mut()
            .spawn((
                Checkbox,
                Checked,
                CollapseToggle {
                    root: Some(root),
                    path: String::new(),
                },
                ChildOf(header),
            ))
            .observe(checkbox_self_update)
            .observe(on_collapse_toggle)
            .id();
        app.update();
        let collapsed = |app: &App| {
            (
                !app.world().entity(toggle).contains::<Checked>(),
                app.world()
                    .resource::<InspectorCollapsed>()
                    .is_collapsed(&root, ""),
            )
        };

        click(&mut app, header, window);
        assert_eq!(
            collapsed(&app),
            (true, true),
            "a click on the header collapses"
        );
        click(&mut app, toggle, window);
        assert_eq!(
            collapsed(&app),
            (false, false),
            "a click on the chevron expands, once"
        );
        click(&mut app, header, window);
        assert_eq!(collapsed(&app), (true, true));
    }
}
