//! Enum editing: a variant selector plus the active variant's fields.
//!
//! Switching variants is a *structural* change, so instead of patching in place we apply a
//! [`DynamicEnum`] for the chosen variant (its fields default-constructed) and rebuild the panel.

use core::any::TypeId;

use bevy_ecs::hierarchy::ChildOf;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::AppTypeRegistry;
use bevy_reflect::enums::{DynamicEnum, DynamicVariant, Enum, VariantInfo};
use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::structs::DynamicStruct;
use bevy_reflect::tuple::DynamicTuple;
use bevy_reflect::{ParsedPath, PartialReflect, TypeRegistry};
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_ui::{px, Display, FlexDirection, Node};
use bevy_ui_widgets::Activate;

use bevy_feathers::controls::{ButtonVariant, FeathersButton, FeathersButtonProps};
use bevy_feathers::display::label;

use crate::attributes::FieldCtx;
use crate::binding::{with_field_reflect_mut, InspectorRoot};
use crate::entry::{find_ancestor_panel, rebuild_panel, InspectorPanel};
use crate::recurse::{column, field_entry, parse_path, BuildCx};

/// Records how to switch an enum to a specific variant when its button is clicked.
#[derive(Component, Clone)]
pub(crate) struct EnumVariantButton {
    root: Option<InspectorRoot>,
    path: ParsedPath,
    variant: String,
    enum_type_id: Option<TypeId>,
}

impl Default for EnumVariantButton {
    fn default() -> Self {
        Self {
            root: None,
            path: ParsedPath(Vec::new()),
            variant: String::new(),
            enum_type_id: None,
        }
    }
}

/// Build a variant selector row followed by the active variant's fields.
pub fn build_enum(
    cx: &BuildCx,
    path: &str,
    value: &dyn PartialReflect,
    enum_ref: &dyn Enum,
) -> Box<dyn Scene> {
    let enum_type_id = value.get_represented_type_info().map(|i| i.type_id());
    let variant_names: Vec<String> = value
        .get_represented_type_info()
        .and_then(|i| i.as_enum().ok())
        .map(|ei| ei.variant_names().iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();
    let current = enum_ref.variant_name().to_string();

    let mut children: Vec<Box<dyn Scene>> = Vec::new();

    // Variant selector: one button per variant, current one highlighted.
    let buttons: Vec<Box<dyn Scene>> = variant_names
        .iter()
        .map(|name| {
            let is_current = *name == current;
            let button = EnumVariantButton {
                root: Some(cx.root.clone()),
                path: parse_path(path),
                variant: name.clone(),
                enum_type_id,
            };
            let caption: Box<dyn SceneList> =
                Box::new(vec![Box::new(label(name.clone())) as Box<dyn Scene>]);
            Box::new((
                <FeathersButton as SceneComponent>::scene(FeathersButtonProps {
                    caption,
                    variant: if is_current {
                        ButtonVariant::Primary
                    } else {
                        ButtonVariant::Normal
                    },
                    ..Default::default()
                }),
                template_value(button),
                on(on_enum_variant_click),
            )) as Box<dyn Scene>
        })
        .collect();
    children.push(Box::new(bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            column_gap: px(4),
        }
        Children [ {buttons} ]
    }));

    // Active variant fields.
    for i in 0..enum_ref.field_len() {
        if let Some(child) = enum_ref.field_at(i) {
            let field_name = enum_ref.name_at(i);
            let sub_path = match field_name {
                Some(name) => format!("{path}.{name}"),
                None => format!("{path}.{i}"),
            };
            let label_text = field_name
                .map(|s| s.to_string())
                .unwrap_or_else(|| i.to_string());
            children.push(field_entry(cx, &label_text, &sub_path, child, &FieldCtx::default()));
        }
    }

    Box::new(column(children))
}

/// Observer: apply the chosen variant and rebuild the owning panel.
fn on_enum_variant_click(
    activate: On<Activate>,
    buttons: Query<&EnumVariantButton>,
    parents: Query<&ChildOf>,
    panels: Query<&InspectorPanel>,
    mut commands: Commands,
) {
    let Ok(button) = buttons.get(activate.event_target()) else {
        return;
    };
    let Some(root) = button.root.clone() else {
        return;
    };
    let Some(enum_type_id) = button.enum_type_id else {
        return;
    };
    let path = button.path.clone();
    let variant = button.variant.clone();
    let panel = find_ancestor_panel(activate.event_target(), &parents, &panels);

    commands.queue(move |world: &mut World| {
        switch_enum_variant(world, &root, &path, enum_type_id, &variant);
        if let Some(panel) = panel {
            rebuild_panel(world, panel);
        }
    });
}

/// Apply a [`DynamicEnum`] for `variant_name` to the enum at `root`/`path`.
fn switch_enum_variant(
    world: &mut World,
    root: &InspectorRoot,
    path: &ParsedPath,
    enum_type_id: TypeId,
    variant_name: &str,
) {
    let registry = world.resource::<AppTypeRegistry>().clone();
    let dynamic = {
        let registry = registry.read();
        build_default_variant(&registry, enum_type_id, variant_name)
    };
    let Some(dynamic) = dynamic else {
        return;
    };
    with_field_reflect_mut(world, root, path, |enum_field| {
        let _ = enum_field.try_apply(dynamic.as_partial_reflect());
    });
}

/// Construct a [`DynamicEnum`] for `variant_name`, default-constructing any variant fields.
///
/// Returns `None` if a data variant has a field whose type has no `ReflectDefault`.
fn build_default_variant(
    registry: &TypeRegistry,
    enum_type_id: TypeId,
    variant_name: &str,
) -> Option<DynamicEnum> {
    let registration = registry.get(enum_type_id)?;
    let enum_info = registration.type_info().as_enum().ok()?;
    let variant = enum_info.variant(variant_name)?;

    let dynamic_variant = match variant {
        VariantInfo::Unit(_) => DynamicVariant::Unit,
        VariantInfo::Tuple(tuple_info) => {
            let mut tuple = DynamicTuple::default();
            for i in 0..tuple_info.field_len() {
                let field = tuple_info.field_at(i)?;
                tuple.insert_boxed(default_value(registry, field.type_id())?);
            }
            DynamicVariant::Tuple(tuple)
        }
        VariantInfo::Struct(struct_info) => {
            let mut structure = DynamicStruct::default();
            for i in 0..struct_info.field_len() {
                let field = struct_info.field_at(i)?;
                structure.insert_boxed(field.name(), default_value(registry, field.type_id())?);
            }
            DynamicVariant::Struct(structure)
        }
    };

    let mut dynamic = DynamicEnum::new(variant_name, dynamic_variant);
    dynamic.set_represented_type(Some(registration.type_info()));
    Some(dynamic)
}

/// Default-construct a value of `type_id` via its `ReflectDefault`, as a boxed partial reflect.
fn default_value(registry: &TypeRegistry, type_id: TypeId) -> Option<Box<dyn PartialReflect>> {
    let default = registry.get(type_id)?.data::<ReflectDefault>()?.default();
    Some(default.into_partial_reflect())
}
