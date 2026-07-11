//! The reflection recursion engine.
//!
//! [`build_value`] is the inspector's analog of `bevy-inspector-egui`'s `ui_for_reflect`: given a
//! reflected value it first checks for a registered per-type widget ([`ReflectInspectorWidget`]),
//! and otherwise recurses structurally, emitting a labeled row per field. It threads a reflection
//! path string down the recursion so each leaf widget knows how to address its field from the root.

use bevy_ecs::hierarchy::Children;
use bevy_reflect::{ParsedPath, PartialReflect, ReflectRef, TypeRegistry};
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_ui::{px, AlignItems, Display, FlexDirection, Node};

use bevy_feathers::display::{label, label_dim};

use crate::binding::InspectorRoot;
use crate::widget::ReflectInspectorWidget;

/// Build an editing scene for `value`, reachable from `root` via the reflection path `path`.
///
/// `path` is a reflection path string relative to the root (empty for the root itself), e.g.
/// `.translation.x`. It is parsed into a [`ParsedPath`] only at leaf widgets.
pub fn build_value(
    registry: &TypeRegistry,
    root: InspectorRoot,
    path: String,
    value: &dyn PartialReflect,
) -> Box<dyn Scene> {
    // 1. Per-type widget override (leaf).
    if let Some(type_id) = value.get_represented_type_info().map(|info| info.type_id())
        && let Some(widget) = registry.get_type_data::<ReflectInspectorWidget>(type_id)
    {
        let parsed = parse_path(&path);
        return (widget.build)(root, parsed, value);
    }

    // 2. Structural recursion.
    match value.reflect_ref() {
        ReflectRef::Struct(strukt) => {
            let rows: Vec<Box<dyn Scene>> = (0..strukt.field_len())
                .filter_map(|i| {
                    let name = strukt.name_at(i)?;
                    let child = strukt.field_at(i)?;
                    let child_path = format!("{path}.{name}");
                    let field = build_value(registry, root.clone(), child_path, child);
                    Some(Box::new(field_row(name, field)) as Box<dyn Scene>)
                })
                .collect();
            Box::new(column(rows))
        }
        // Other kinds (enums, collections, opaque leaves) become read-only labels in Phase 1.
        _ => Box::new(read_only_leaf(value)),
    }
}

/// Parse a reflection path string, treating the empty string as the identity (root) path.
fn parse_path(path: &str) -> ParsedPath {
    if path.is_empty() {
        ParsedPath(Vec::new())
    } else {
        ParsedPath::parse(path).unwrap_or_else(|_| ParsedPath(Vec::new()))
    }
}

/// A labeled row: field name on the left, editing widget on the right.
///
/// The children are collected into one `Vec` (a `SceneList`) because a bare `Scene` embedded via
/// `{}` in a `Children [ ... ]` slot must be a `SceneList`, not a `Scene`.
fn field_row(name: &str, field: Box<dyn Scene>) -> impl Scene {
    let children: Vec<Box<dyn Scene>> = vec![Box::new(label(name.to_string())), field];
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(8),
            min_height: px(26),
        }
        Children [ {children} ]
    }
}

/// A vertical stack of rows.
fn column(rows: Vec<Box<dyn Scene>>) -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(2),
        }
        Children [ {rows} ]
    }
}

/// A read-only fallback label for values without a registered widget.
fn read_only_leaf(value: &dyn PartialReflect) -> impl Scene {
    let text = value
        .get_represented_type_info()
        .map(|info| info.type_path().to_string())
        .unwrap_or_else(|| "<opaque>".to_string());
    label_dim(text)
}
