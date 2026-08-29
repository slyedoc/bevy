//! List/Array editing: a row per element plus add/remove buttons.
//!
//! Adding and removing elements changes structure, so (like enum variant switches) the edit is
//! applied through reflection and the panel is rebuilt.

use core::any::TypeId;

use bevy_ecs::hierarchy::ChildOf;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::AppTypeRegistry;
use bevy_reflect::list::List;
use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::{ParsedPath, ReflectMut};
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_ui::{px, AlignItems, Display, FlexDirection, Node};
use bevy_ui_widgets::Activate;

use bevy_feathers::controls::{FeathersButton, FeathersButtonProps};
use bevy_feathers::display::label;

use crate::attributes::FieldCtx;
use crate::binding::{with_field_reflect_mut, InspectorRoot};
use crate::entry::{find_ancestor_panel, rebuild_panel, InspectorPanel};
use crate::recurse::{build_value, column, parse_path, BuildCx};

/// A structural list edit.
#[derive(Clone)]
enum ListOp {
    /// Append a default-constructed element.
    Push,
    /// Remove the element at this index.
    Remove(usize),
}

/// Records a pending list edit for a button.
#[derive(Component, Clone)]
pub(crate) struct ListButton {
    root: Option<InspectorRoot>,
    path: ParsedPath,
    op: Option<ListOp>,
    item_type_id: Option<TypeId>,
}

impl Default for ListButton {
    fn default() -> Self {
        Self {
            root: None,
            path: ParsedPath(Vec::new()),
            op: None,
            item_type_id: None,
        }
    }
}

/// Build a row per element (each with a remove button) plus a trailing add button.
pub fn build_list(cx: &BuildCx, path: &str, list: &dyn List) -> Box<dyn Scene> {
    let item_type_id = list_item_type_id(list);
    let mut rows: Vec<Box<dyn Scene>> = Vec::new();

    for i in 0..list.len() {
        if let Some(child) = list.get(i) {
            let child_path = format!("{path}[{i}]");
            let widget = build_value(cx, &child_path, child, &FieldCtx::default());
            let remove = op_button(cx, path, ListOp::Remove(i), item_type_id, "x");
            let children: Vec<Box<dyn Scene>> =
                vec![Box::new(label(i.to_string())), widget, remove];
            rows.push(Box::new(bsn! {
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(6),
                    min_height: px(26),
                }
                Children [ {children} ]
            }));
        }
    }

    rows.push(op_button(cx, path, ListOp::Push, item_type_id, "+ add"));
    Box::new(column(rows))
}

/// A feathers button carrying a pending list edit.
fn op_button(
    cx: &BuildCx,
    path: &str,
    op: ListOp,
    item_type_id: Option<TypeId>,
    caption: &str,
) -> Box<dyn Scene> {
    let button = ListButton {
        root: Some(cx.root.clone()),
        path: parse_path(path),
        op: Some(op),
        item_type_id,
    };
    let caption_list: Box<dyn SceneList> =
        Box::new(vec![Box::new(label(caption.to_string())) as Box<dyn Scene>]);
    Box::new((
        <FeathersButton as SceneComponent>::scene(FeathersButtonProps {
            caption: caption_list,
            ..Default::default()
        }),
        template_value(button),
        on(on_list_button_click),
    ))
}

/// Observer: apply the pending list edit and rebuild the owning panel.
fn on_list_button_click(
    activate: On<Activate>,
    buttons: Query<&ListButton>,
    parents: Query<&ChildOf>,
    panels: Query<&InspectorPanel>,
    mut commands: Commands,
) {
    let Ok(button) = buttons.get(activate.event_target()) else {
        return;
    };
    let (Some(root), Some(op)) = (button.root.clone(), button.op.clone()) else {
        return;
    };
    let path = button.path.clone();
    let item_type_id = button.item_type_id;
    let panel = find_ancestor_panel(activate.event_target(), &parents, &panels);

    commands.queue(move |world: &mut World| {
        apply_list_op(world, &root, &path, &op, item_type_id);
        if let Some(panel) = panel {
            rebuild_panel(world, panel);
        }
    });
}

/// Apply a push/remove to the list at `root`/`path`.
fn apply_list_op(
    world: &mut World,
    root: &InspectorRoot,
    path: &ParsedPath,
    op: &ListOp,
    item_type_id: Option<TypeId>,
) {
    match op {
        ListOp::Remove(index) => {
            let index = *index;
            with_field_reflect_mut(world, root, path, move |field| {
                if let ReflectMut::List(list) = field.reflect_mut()
                    && index < list.len()
                {
                    list.remove(index);
                }
            });
        }
        ListOp::Push => {
            let registry = world.resource::<AppTypeRegistry>().clone();
            let default = {
                let registry = registry.read();
                item_type_id
                    .and_then(|id| registry.get(id))
                    .and_then(|reg| reg.data::<ReflectDefault>())
                    .map(bevy_reflect::std_traits::ReflectDefault::default)
            };
            let Some(default) = default else {
                return;
            };
            with_field_reflect_mut(world, root, path, move |field| {
                if let ReflectMut::List(list) = field.reflect_mut() {
                    list.push(default.into_partial_reflect());
                }
            });
        }
    }
}

/// Determine the element type of a list (from an existing element, else its list metadata).
fn list_item_type_id(list: &dyn List) -> Option<TypeId> {
    if let Some(first) = list.get(0) {
        return first
            .get_represented_type_info()
            .map(bevy_reflect::TypeInfo::type_id);
    }
    list.get_represented_type_info()
        .and_then(|i| i.as_list().ok())
        .map(|list_info| list_info.item_ty().id())
}
