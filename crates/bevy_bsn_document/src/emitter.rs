//! BSN text emitter: document AST to `.bsn` text.
//!
//! The document is lowered to a `bevy_bsn` [`BsnDocument`] and printed by its printer, one
//! field per line so an edit changes one line. Emission order is fully determined by the
//! document's `Vec<Entity>` fields (`SceneBsnAst::roots`, `BsnPatches`, `BsnStructFields`,
//! `Children` lists), so emitting the same document twice yields byte-identical text. A map is
//! written as a list of `(key, value)` pairs, sorted by key.

use bevy_bsn::{
    BsnDocument, BsnNodeId, BsnNodeKind, BsnPatchPrefix, BsnPath, BsnValueId, PatchBody,
    PrintOptions,
};
use bevy_ecs::entity::Entity;

use crate::{BsnField, BsnPatch, BsnValue, SceneBsnAst};

/// Emits a complete `.bsn` file from the document AST, one top-level entity per root.
pub fn emit_scene(ast: &SceneBsnAst) -> String {
    emit_entities(ast, &ast.roots)
}

/// Emits BSN text for a single entity (and its children) from the AST. Used for clipboard
/// copy: the output is valid `bsn!` macro input.
pub fn emit_entity(ast: &SceneBsnAst, patches_entity: Entity) -> String {
    emit_entities(ast, &[patches_entity])
}

/// Emits BSN text for several entities, separated by `--` like a multi-root scene.
pub fn emit_entities(ast: &SceneBsnAst, entities: &[Entity]) -> String {
    let mut doc = BsnDocument::new();
    for &entity in entities {
        if let Some(root) = lower_entity(ast, &mut doc, entity, 0) {
            doc.push_root(root);
        }
    }
    let mut out = String::new();
    bevy_bsn::write_document_with(
        &doc,
        &mut out,
        &PrintOptions {
            one_field_per_line: true,
            blank_line_between_roots: false,
            ..PrintOptions::default()
        },
    )
    .expect("writing to a String does not fail");
    out
}

fn lower_entity(
    ast: &SceneBsnAst,
    doc: &mut BsnDocument,
    patches_entity: Entity,
    depth: usize,
) -> Option<BsnNodeId> {
    // A `Children` cycle stops here rather than writing text until memory runs out.
    if depth >= crate::MAX_AST_DEPTH {
        log::warn!(
            "document node {patches_entity} is deeper than {}; it was not emitted",
            crate::MAX_AST_DEPTH
        );
        return None;
    }
    let patches = ast.get_patches(patches_entity)?;
    let (mut name, mut base) = (None, None);
    let (mut patch_nodes, mut relations) = (Vec::new(), Vec::new());
    for &patch_entity in &patches.0 {
        let Some(patch) = ast.get_patch(patch_entity) else {
            continue;
        };
        match patch {
            BsnPatch::Name(n) => name = Some(n.clone()),
            BsnPatch::Base(b) => base = Some(b.clone()),
            BsnPatch::Type(type_path) => patch_nodes.push(doc.push_patch(
                BsnPatchPrefix::FromTemplate,
                path(type_path),
                PatchBody::Unit,
            )),
            BsnPatch::Struct(data) => {
                let body = if data.fields.0.is_empty() {
                    PatchBody::Unit
                } else {
                    PatchBody::Struct(lower_fields(doc, &data.fields.0))
                };
                patch_nodes.push(doc.push_patch(
                    BsnPatchPrefix::FromTemplate,
                    path(&data.type_path),
                    body,
                ));
            }
            BsnPatch::TupleStruct(data) => {
                let items = data.values.iter().map(|v| lower_value(doc, v)).collect();
                patch_nodes.push(doc.push_patch(
                    BsnPatchPrefix::FromTemplate,
                    path(&data.type_path),
                    PatchBody::Tuple(items),
                ));
            }
            BsnPatch::Template(type_path, fields) => {
                let body = match fields {
                    Some(fields) => PatchBody::Struct(lower_fields(doc, &fields.0)),
                    None => PatchBody::Unit,
                };
                patch_nodes.push(doc.push_patch(BsnPatchPrefix::Template, path(type_path), body));
            }
            BsnPatch::Children(children) => {
                let entities = children
                    .iter()
                    .filter_map(|&child| lower_entity(ast, doc, child, depth + 1))
                    .collect();
                relations.push(doc.push_node(BsnNodeKind::Relation {
                    target_symbol: path("bevy_ecs::hierarchy::Children"),
                    entities,
                }));
            }
        }
    }
    Some(doc.push_node(BsnNodeKind::Entity {
        name,
        name_span: None,
        base,
        base_span: None,
        patches: patch_nodes,
        relations,
    }))
}

fn lower_fields(doc: &mut BsnDocument, fields: &[BsnField]) -> Vec<(String, BsnValueId)> {
    fields
        .iter()
        .map(|field| (field.name.clone(), lower_value(doc, &field.value)))
        .collect()
}

fn lower_value(doc: &mut BsnDocument, value: &BsnValue) -> BsnValueId {
    let value = match value {
        BsnValue::Float(f) => bevy_bsn::BsnValue::Float(*f),
        BsnValue::Int(i) => bevy_bsn::BsnValue::Int(*i),
        BsnValue::Bool(b) => bevy_bsn::BsnValue::Bool(*b),
        BsnValue::String(s) => bevy_bsn::BsnValue::String(s.clone()),
        BsnValue::Type(type_path) => bevy_bsn::BsnValue::Path(path(type_path)),
        BsnValue::Struct(data) => {
            bevy_bsn::BsnValue::Struct(path(&data.type_path), lower_fields(doc, &data.fields.0))
        }
        BsnValue::TupleStruct(data) => bevy_bsn::BsnValue::NamedTuple(
            path(&data.type_path),
            data.values.iter().map(|v| lower_value(doc, v)).collect(),
        ),
        BsnValue::List(items) => {
            bevy_bsn::BsnValue::List(items.iter().map(|v| lower_value(doc, v)).collect())
        }
        BsnValue::Map(entries) => {
            let mut sorted: Vec<_> = entries.iter().collect();
            sorted.sort_by_cached_key(|(key, _)| format!("{key:?}"));
            let pairs = sorted
                .into_iter()
                .map(|(key, value)| {
                    let pair = vec![lower_value(doc, key), lower_value(doc, value)];
                    doc.push_value(bevy_bsn::BsnValue::Tuple(pair))
                })
                .collect();
            bevy_bsn::BsnValue::List(pairs)
        }
    };
    doc.push_value(value)
}

/// A document type path as a `bevy_bsn` path. One that does not parse is written as it is (and
/// will not load back) rather than failing the whole save.
fn path(type_path: &str) -> BsnPath {
    BsnPath::from_type_path(type_path).unwrap_or_else(|| {
        log::warn!("`{type_path}` is not a valid type path; it is written as it is");
        BsnPath::from_segments([type_path])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BsnPatches, BsnStructData, BsnStructFields, BsnTupleStructData};

    #[test]
    fn emit_simple_entity() {
        let mut ast = SceneBsnAst::default();

        let name_patch = ast.world.spawn(BsnPatch::Name("Root".into())).id();
        let transform_patch = ast
            .world
            .spawn(BsnPatch::Type(
                "bevy_transform::components::transform::Transform".into(),
            ))
            .id();
        let vis_patch = ast
            .world
            .spawn(BsnPatch::Type(
                "bevy_camera::visibility::Visibility::Visible".into(),
            ))
            .id();

        let patches_entity = ast
            .world
            .spawn(BsnPatches(vec![name_patch, transform_patch, vis_patch]))
            .id();
        ast.roots.push(patches_entity);

        let text = emit_scene(&ast);
        assert!(text.contains("#Root"));
        assert!(text.contains("bevy_transform::components::transform::Transform"));
        assert!(text.contains("bevy_camera::visibility::Visibility::Visible"));
    }

    #[test]
    fn emit_struct_with_fields() {
        let mut ast = SceneBsnAst::default();

        let patch = ast
            .world
            .spawn(BsnPatch::Struct(BsnStructData {
                type_path: "bevy_light::directional_light::DirectionalLight".into(),
                fields: BsnStructFields(vec![BsnField {
                    name: "shadow_maps_enabled".into(),
                    value: BsnValue::Bool(true),
                }]),
            }))
            .id();

        let entity = ast.world.spawn(BsnPatches(vec![patch])).id();
        ast.roots.push(entity);

        let text = emit_scene(&ast);
        assert!(text.contains("DirectionalLight {"));
        assert!(text.contains("shadow_maps_enabled: true,"));
    }

    #[test]
    fn emit_children() {
        let mut ast = SceneBsnAst::default();

        let child_name = ast.world.spawn(BsnPatch::Name("Child".into())).id();
        let child = ast.world.spawn(BsnPatches(vec![child_name])).id();

        let root_name = ast.world.spawn(BsnPatch::Name("Root".into())).id();
        let children_patch = ast.world.spawn(BsnPatch::Children(vec![child])).id();
        let root = ast
            .world
            .spawn(BsnPatches(vec![root_name, children_patch]))
            .id();
        ast.roots.push(root);

        let text = emit_scene(&ast);
        assert!(text.contains("#Root"));
        assert!(text.contains("bevy_ecs::hierarchy::Children ["));
        assert!(text.contains("    #Child"));
        assert!(text.contains("]"));
    }

    #[test]
    fn emit_tuple_struct() {
        let mut ast = SceneBsnAst::default();

        let patch = ast
            .world
            .spawn(BsnPatch::TupleStruct(BsnTupleStructData {
                type_path: "bevy_scene::components::SceneRoot".into(),
                values: vec![BsnValue::String(
                    "models/FlightHelmet/FlightHelmet.gltf#Scene0".into(),
                )],
            }))
            .id();

        let entity = ast.world.spawn(BsnPatches(vec![patch])).id();
        ast.roots.push(entity);

        let text = emit_scene(&ast);
        assert!(text.contains("SceneRoot(\"models/FlightHelmet/FlightHelmet.gltf#Scene0\")"));
    }

    #[test]
    fn every_row_of_a_list_ends_in_a_comma_so_adding_one_touches_one_line() {
        let mut ast = SceneBsnAst::default();

        let row = |item: &str| {
            BsnValue::Struct(BsnStructData {
                type_path: "test::LootRoll".into(),
                fields: BsnStructFields(vec![BsnField {
                    name: "item".into(),
                    value: BsnValue::String(item.into()),
                }]),
            })
        };
        let patch = ast
            .world
            .spawn(BsnPatch::Struct(BsnStructData {
                type_path: "test::ItemDef".into(),
                fields: BsnStructFields(vec![BsnField {
                    name: "loot".into(),
                    value: BsnValue::List(vec![row("coin"), row("gem")]),
                }]),
            }))
            .id();
        let entity = ast.world.spawn(BsnPatches(vec![patch])).id();
        ast.roots.push(entity);

        let expected = "test::ItemDef {\n    loot: [\n        test::LootRoll {\n            item: \"coin\",\n        },\n        test::LootRoll {\n            item: \"gem\",\n        },\n    ],\n}\n";
        assert_eq!(emit_scene(&ast), expected);
    }

    #[test]
    fn emit_is_deterministic_for_multi_field_document() {
        let mut ast = SceneBsnAst::default();

        let patch = ast
            .world
            .spawn(BsnPatch::Struct(BsnStructData {
                type_path: "test::Widget".into(),
                fields: BsnStructFields(vec![
                    BsnField {
                        name: "third".into(),
                        value: BsnValue::Int(3),
                    },
                    BsnField {
                        name: "first".into(),
                        value: BsnValue::Int(1),
                    },
                    BsnField {
                        name: "second".into(),
                        value: BsnValue::Int(2),
                    },
                ]),
            }))
            .id();
        let entity = ast.world.spawn(BsnPatches(vec![patch])).id();
        ast.roots.push(entity);

        let expected = "test::Widget {\n    third: 3,\n    first: 1,\n    second: 2,\n}\n";
        let first = emit_scene(&ast);
        let second = emit_scene(&ast);
        assert_eq!(first, expected);
        assert_eq!(first, second);
    }
}
