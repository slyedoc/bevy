//! Reader, editor document, and writer for the `.bsn` scene format.
//!
//! The parser builds the editor document ([`SceneBsnAst`]) directly from
//! `.bsn` source text; there is no separate parse-time representation. The
//! apply path resolves the document to ECS components, and the emitter
//! writes the document back to `.bsn` text. The grammar rules track the
//! dynamic-BSN work in bevyengine/bevy#23576.

pub mod apply;
pub mod binary;
pub mod catalog;
pub mod delta;
pub mod document;
pub mod emitter;
pub mod file;
pub mod loader;
pub mod parse;
pub mod sync;
pub mod writer;

pub use catalog::{
    adopt_asset_roots, append_assets_to_ast, asset_roots, asset_value_from_root, entity_roots,
    is_asset_root, load_asset_root, load_bsn_assets, load_bsn_scene, serialize_assets_to_bsn,
    serialize_assets_to_bsn_reporting, CatalogAssetRef, CatalogEntry, LoadedBsnScene,
};

pub use binary::{is_binary, BinaryError, DecodedDocument};

pub use file::{
    binary_twin, convert_to_binary, convert_to_text, document_as_text, document_bytes,
    document_from_bytes, document_text_from_bytes, existing_form, export_binary, is_binary_path,
    is_document_extension, is_document_path, leading_comments, read_document, read_document_text,
    text_as_binary, text_twin, write_document_text, Document, DocumentError, DocumentForm,
    BINARY_EXTENSION, TEXT_EXTENSION,
};

pub use parse::{parse_bsn, ParseError};

pub use delta::{apply_deltas, bsn_value_eq, shallow_diff};

pub use document::{
    bsn_value_as_int, clone_node_into, clone_subtree_into, component_to_bsn_patch,
    component_to_bsn_patch_with_assets, is_enum_variant_of, patch_type_path, type_paths_include,
    AstNodeRef, BsnAssetContext, BsnField, BsnPatch, BsnPatches, BsnRelated, BsnStructData,
    BsnStructFields, BsnTupleStructData, BsnValue, SceneBsnAst, MAX_AST_DEPTH,
};
pub use emitter::{emit_entities, emit_entity, emit_scene};
pub use loader::{parse_bsn_text, BsnLoadError};

pub use apply::{
    apply_ast_to_ecs, apply_component_patch, apply_dirty_ast_patches, apply_reference_map,
    bsn_value_to_reflect, get_bsn_field, insert_relationship, remove_bsn_field, set_bsn_field,
    spawn_ast_node, spawn_from_ast, AstDirty, BsnApplyAssets, BsnAssetPaths, BsnProjectAssets,
    BsnSceneAssets, DocumentOnlyTypes, UnresolvedTypes,
};

pub use sync::{
    create_entity_in_ast, delete_entity_from_ast, sync_hierarchy_to_ast, sync_hierarchy_to_ast_at,
    sync_to_ast,
};

pub use writer::{
    append_world_to_ast, serialize_to_bsn, serialize_to_bsn_with_config, BsnWriterConfig,
};

use bevy_app::{App, Plugin};

/// Registers the BSN document resources an editor reads.
///
/// The apply path ([`apply_dirty_ast_patches`]) is called explicitly during
/// scene load, so it is deliberately not registered as a per-frame system.
pub struct BsnDocumentPlugin;

impl Plugin for BsnDocumentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SceneBsnAst>()
            .init_resource::<DocumentOnlyTypes>()
            .init_resource::<UnresolvedTypes>();
    }
}
