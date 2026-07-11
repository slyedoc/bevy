//! The reflection recursion engine.
//!
//! [`build_value`] is the inspector's analog of `bevy-inspector-egui`'s `ui_for_reflect`: given a
//! reflected value it first checks for a registered per-type widget ([`ReflectInspectorWidget`]),
//! and otherwise recurses structurally, emitting a labeled row per field. It threads a reflection
//! path string (so each leaf knows how to address its field from the root) and a [`FieldCtx`] (the
//! field's custom-attribute configuration).

use bevy_ecs::hierarchy::Children;
use bevy_reflect::structs::Struct;
use bevy_reflect::tuple_struct::TupleStruct;
use bevy_reflect::{ParsedPath, PartialReflect, ReflectRef, TypeRegistry};
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_ui::{px, AlignItems, Display, FlexDirection, Node};

use bevy_feathers::display::{label, label_dim};

use crate::attributes::FieldCtx;
use crate::binding::InspectorRoot;
use crate::enums::build_enum;
use crate::lists::build_list;
use crate::widget::ReflectInspectorWidget;

/// Invariant context shared across one component/resource's recursion.
pub struct BuildCx<'a> {
    /// The type registry (for widget lookup and recursion).
    pub registry: &'a TypeRegistry,
    /// The reflected value this subtree edits.
    pub root: InspectorRoot,
}

/// Build an editing scene for `value`, reachable from `cx.root` via the reflection path `path`.
pub fn build_value(
    cx: &BuildCx,
    path: &str,
    value: &dyn PartialReflect,
    field: &FieldCtx,
) -> Box<dyn Scene> {
    // Read-only fields display their value without an editing widget.
    if field.read_only {
        return Box::new(read_only_leaf(value));
    }

    // Per-type widget override (leaf).
    if let Some(type_id) = value.get_represented_type_info().map(|info| info.type_id())
        && let Some(widget) = cx.registry.get_type_data::<ReflectInspectorWidget>(type_id)
    {
        return (widget.build)(cx, path, value, field);
    }

    // Structural recursion.
    match value.reflect_ref() {
        ReflectRef::Struct(strukt) => build_struct(cx, path, value, strukt),
        ReflectRef::TupleStruct(tuple_struct) => build_tuple_struct(cx, path, tuple_struct),
        ReflectRef::Enum(enum_ref) => build_enum(cx, path, value, enum_ref),
        ReflectRef::List(list) => build_list(cx, path, list),
        ReflectRef::Array(array) => {
            // Fixed-size arrays: rows per element, no add/remove.
            let rows = (0..array.len())
                .filter_map(|i| {
                    let child = array.get(i)?;
                    let child_path = format!("{path}[{i}]");
                    let widget = build_value(cx, &child_path, child, &FieldCtx::default());
                    Some(Box::new(field_row(&i.to_string(), widget)) as Box<dyn Scene>)
                })
                .collect();
            Box::new(column(rows))
        }
        _ => Box::new(read_only_leaf(value)),
    }
}

/// Recurse into a named struct, one labeled row per field.
fn build_struct(
    cx: &BuildCx,
    path: &str,
    value: &dyn PartialReflect,
    strukt: &dyn Struct,
) -> Box<dyn Scene> {
    let info = value
        .get_represented_type_info()
        .and_then(|i| i.as_struct().ok());
    let rows: Vec<Box<dyn Scene>> = (0..strukt.field_len())
        .filter_map(|i| {
            let name = strukt.name_at(i)?;
            let child = strukt.field_at(i)?;
            let field_info = info.and_then(|si| si.field_at(i));
            if field_info.is_some_and(FieldCtx::is_hidden) {
                return None;
            }
            let child_field = field_info.map(FieldCtx::from_field).unwrap_or_default();
            let child_path = format!("{path}.{name}");
            let widget = build_value(cx, &child_path, child, &child_field);
            Some(Box::new(field_row(name, widget)) as Box<dyn Scene>)
        })
        .collect();
    Box::new(column(rows))
}

/// Recurse into a tuple struct, one row per positional field.
fn build_tuple_struct(cx: &BuildCx, path: &str, tuple_struct: &dyn TupleStruct) -> Box<dyn Scene> {
    let rows: Vec<Box<dyn Scene>> = (0..tuple_struct.field_len())
        .filter_map(|i| {
            let child = tuple_struct.field(i)?;
            let child_path = format!("{path}.{i}");
            let widget = build_value(cx, &child_path, child, &FieldCtx::default());
            Some(Box::new(field_row(&i.to_string(), widget)) as Box<dyn Scene>)
        })
        .collect();
    Box::new(column(rows))
}

/// Parse a reflection path string, treating the empty string as the identity (root) path.
pub fn parse_path(path: &str) -> ParsedPath {
    if path.is_empty() {
        ParsedPath(Vec::new())
    } else {
        ParsedPath::parse(path).unwrap_or_else(|_| ParsedPath(Vec::new()))
    }
}

/// A labeled row: field name on the left, editing widget on the right.
///
/// The children are collected into one `Vec` (a `SceneList`) because a bare `Scene` embedded via
/// `{}` in a `Children [ ... ]` slot must be a `SceneList`.
pub fn field_row(name: &str, widget: Box<dyn Scene>) -> impl Scene {
    let children: Vec<Box<dyn Scene>> = vec![Box::new(label(name.to_string())), widget];
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
pub fn column(rows: Vec<Box<dyn Scene>>) -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(2),
        }
        Children [ {rows} ]
    }
}

/// A read-only fallback label showing the value (or its type when it can't be stringified).
fn read_only_leaf(value: &dyn PartialReflect) -> impl Scene {
    label_dim(value_to_string(value))
}

/// Best-effort stringification of a reflected value for read-only display.
pub fn value_to_string(value: &dyn PartialReflect) -> String {
    if let Some(reflect) = value.try_as_reflect() {
        if let Some(v) = reflect.downcast_ref::<f32>() {
            return v.to_string();
        }
        if let Some(v) = reflect.downcast_ref::<f64>() {
            return v.to_string();
        }
        if let Some(v) = reflect.downcast_ref::<i32>() {
            return v.to_string();
        }
        if let Some(v) = reflect.downcast_ref::<i64>() {
            return v.to_string();
        }
        if let Some(v) = reflect.downcast_ref::<u32>() {
            return v.to_string();
        }
        if let Some(v) = reflect.downcast_ref::<usize>() {
            return v.to_string();
        }
        if let Some(v) = reflect.downcast_ref::<bool>() {
            return v.to_string();
        }
        if let Some(v) = reflect.downcast_ref::<String>() {
            return v.clone();
        }
    }
    value
        .get_represented_type_info()
        .map(|info| info.type_path().to_string())
        .unwrap_or_else(|| "<opaque>".to_string())
}
