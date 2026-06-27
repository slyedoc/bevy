//! BSN assets loaded from `.bsn` files.
//!
//! This module contains the runtime parser/loader for the `.bsn` text format. Its public items are
//! AST node types and parser plumbing used by the generated grammar; they are implementation
//! details rather than a stable public API, so `missing_docs` is allowed here.
#![allow(missing_docs)]

use bevy_asset::{
    io::Reader, AssetLoader, AssetPath, AssetServer, LoadContext, ReflectAsset, ReflectHandle,
    UntypedHandle,
};
use bevy_ecs::{
    bundle::BundleWriter,
    entity::Entity,
    error::{BevyError, Result as EcsResult},
    hierarchy::ChildOf,
    name::Name,
    prelude::{Component, Resource},
    reflect::AppTypeRegistry,
    template::{SceneEntityReference, TemplateContext},
    world::{FromWorld, World},
};
use bevy_log::error;
use bevy_platform::collections::HashMap;
use bevy_reflect::{
    convert::ReflectConvert,
    enums::{DynamicEnum, DynamicVariant, StructVariantInfo, VariantInfoError},
    list::DynamicList,
    prelude::ReflectDefault,
    structs::{DynamicStruct, Struct, StructInfo},
    tuple::DynamicTuple,
    tuple_struct::{DynamicTupleStruct, TupleStruct},
    NamedField, PartialReflect, Reflect, ReflectMut, ReflectRef, TypePath, TypeRegistration,
    TypeRegistry,
};
use core::{
    any::{Any, TypeId},
    cell::RefCell,
    fmt::Write,
    mem,
    str::Utf8Error,
};
use std::io::Error as IoError;
use thiserror::Error;

use crate::{
    dynamic_bsn_grammar::TopLevelPatchesParser, dynamic_bsn_lexer::Lexer, CachedSceneAsset,
    ErasedComponentTemplate, NameEntityReference, ResolveContext, ResolveSceneError, ResolvedScene,
    Scene, SceneDependencies, ScenePatch, SceneScope,
};

#[derive(Default)]
pub struct BsnAst(pub World);

#[derive(Resource, Default)]
pub struct BsnNameStore {
    pub name_indices: HashMap<String, usize>,
    pub next_name_index: usize,
}

#[derive(Component)]
pub struct BsnPatches(pub Vec<Entity>);

#[derive(Component)]
pub enum BsnPatch {
    Name(String, usize),
    Base(String),
    Var(BsnVar),
    Struct(BsnStruct),
    NamedTuple(BsnNamedTuple),
    Relation(BsnRelation),
}

#[derive(Clone)]
pub struct BsnVar(pub BsnSymbol, pub bool);

#[derive(Clone)]
pub struct BsnSymbol(pub Vec<String>, pub String);

pub struct BsnStruct(pub BsnSymbol, pub Vec<BsnField>, pub bool);

pub struct BsnField(pub String, pub Entity);

pub struct BsnNamedTuple(pub BsnSymbol, pub Vec<Entity>, pub bool);

pub struct BsnRelation(pub BsnSymbol, pub Vec<Entity>);

#[derive(Component)]
pub enum BsnExpr {
    Var(BsnVar),
    Struct(BsnStruct),
    NamedTuple(BsnNamedTuple),
    StringLit(String),
    FloatLit(f64),
    BoolLit(bool),
    IntLit(i128),
    List(Vec<Entity>),
}

impl BsnSymbol {
    pub fn from_ident(ident: String) -> BsnSymbol {
        BsnSymbol(vec![], ident)
    }

    pub fn append(mut self, ident: String) -> BsnSymbol {
        self.0.push(mem::replace(&mut self.1, ident));
        self
    }
}

#[derive(TypePath)]
pub struct DynamicBsnLoader {
    type_registry: AppTypeRegistry,
}

impl FromWorld for DynamicBsnLoader {
    fn from_world(world: &mut World) -> Self {
        DynamicBsnLoader {
            type_registry: world.resource::<AppTypeRegistry>().clone(),
        }
    }
}

// TODO: Report multiple errors
#[derive(Error, Debug)]
pub enum DynamicBsnLoaderError {
    #[error("I/O error: {0}")]
    Io(#[from] IoError),
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] Utf8Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("no such AST node")]
    NoSuchAstNode,
    #[error("only `Children` relations supported")]
    OnlyChildRelationsSupported,
    #[error("type doesn't implement `Default`: {0}")]
    TypeDoesntImplementDefault(String),
    #[error("type isn't a tuple structure")]
    TypeNotNamedTuple,
    #[error("type isn't a structure")]
    TypeNotStruct,
    #[error("variant isn't a tuple variant: {0}")]
    VariantNotTuple(#[from] VariantInfoError),
    #[error("structure doesn't have a field named `{0}`")]
    StructDoesntHaveField(String),
    #[error("unknown type: `{0}`")]
    UnknownType(String),
    #[error("type mismatch")]
    TypeMismatch,
    #[error("type mismatch, expected `f32` or `f64`")]
    FloatLitTypeMismatch,
    #[error(
        "type mismatch, expected `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, \
        `isize`, or `usize`"
    )]
    IntLitTypeMismatch,
}

impl AssetLoader for DynamicBsnLoader {
    type Asset = ScenePatch;

    type Settings = ();

    type Error = DynamicBsnLoaderError;

    fn extensions(&self) -> &[&str] {
        &["bsn"]
    }

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut buffer = vec![];
        reader.read_to_end(&mut buffer).await?;
        let input = str::from_utf8(&buffer)?;

        let mut world = World::new();
        world.init_resource::<BsnNameStore>();
        let ast = RefCell::new(BsnAst(world));

        let lexer = Lexer::new(input);
        let patches_id = match TopLevelPatchesParser::new().parse(&ast, lexer) {
            Ok(patches_id) => patches_id,
            Err(err) => {
                return Err(DynamicBsnLoaderError::Parse(format!("{:?}", err)));
            }
        };

        let ast = ast.into_inner();
        let patch = ast.convert_bsn_patches_to_patch(patches_id, &self.type_registry)?;

        // FIXME: We throw the AST away here. Probably not what we want to do
        // for the editor!

        Ok(ScenePatch {
            scene: Some(Box::new(SceneScope(patch.scene))),
            dependencies: patch.dependencies,
            resolved: None,
        })
    }
}

impl BsnAst {
    fn convert_bsn_patches_to_patch(
        &self,
        patches_id: Entity,
        app_type_registry: &AppTypeRegistry,
    ) -> Result<ScenePatch, DynamicBsnLoaderError> {
        let Some(patches) = self.0.get::<BsnPatches>(patches_id) else {
            return Err(DynamicBsnLoaderError::NoSuchAstNode);
        };
        let mut scene_patches: Vec<_> = patches
            .0
            .iter()
            .map(|patch_id| self.convert_bsn_patch_to_patch(*patch_id, app_type_registry))
            .collect::<Result<Vec<_>, _>>()?;
        let dependencies: Vec<_> = scene_patches
            .iter_mut()
            .flat_map(|scene_patch| mem::take(&mut scene_patch.dependencies))
            .collect();
        Ok(ScenePatch {
            scene: Some(Box::new(MultiPatch(
                scene_patches
                    .into_iter()
                    .filter_map(|scene_patch| scene_patch.scene)
                    .collect(),
            ))),
            dependencies,
            resolved: None,
        })
    }

    fn convert_bsn_patch_to_patch(
        &self,
        patch_id: Entity,
        app_type_registry: &AppTypeRegistry,
    ) -> Result<ScenePatch, DynamicBsnLoaderError> {
        let Some(patch) = self.0.get::<BsnPatch>(patch_id) else {
            return Err(DynamicBsnLoaderError::NoSuchAstNode);
        };

        let patch = match *patch {
            BsnPatch::Name(ref name, index) => Box::new(NameEntityReference {
                name: Name(name.clone().into()),
                reference: SceneEntityReference::new(("bsn", index, 0), index, 0),
            }) as Box<dyn Scene>,

            BsnPatch::Base(ref base) => {
                Box::new(CachedSceneAsset::from(base.clone())) as Box<dyn Scene>
            }

            BsnPatch::Var(BsnVar(ref symbol, is_template)) => {
                let symbol = symbol.clone();

                let type_registry = app_type_registry.read();
                let resolved_symbol =
                    symbol.resolve_type_or_enum_variant_to_template(&type_registry, is_template)?;

                let app_type_registry = app_type_registry.clone();

                Box::new(ErasedTemplatePatch {
                    template_type_id: resolved_symbol.template_type_id,
                    app_type_registry: app_type_registry.clone(),
                    fun: move |reflect, _context| {
                        // This could be an enum variant
                        // (`some_crate::Enum::Variant`) or a unit struct.
                        if !resolved_symbol.template_is_enum {
                            // This is a unit struct. It should already be instantiated.
                            return;
                        }

                        let ReflectMut::Enum(enum_reflect) = reflect.reflect_mut() else {
                            error!("Expected an enum");
                            return;
                        };

                        let dynamic_enum = DynamicEnum::new(symbol.1.clone(), DynamicVariant::Unit);
                        enum_reflect.apply(&dynamic_enum);
                    },
                }) as Box<dyn Scene>
            }

            BsnPatch::Struct(BsnStruct(ref symbol, ref fields, is_template)) => {
                let symbol = symbol.clone();

                let type_registry = app_type_registry.read();
                let resolved_symbol =
                    symbol.resolve_type_or_enum_variant_to_template(&type_registry, is_template)?;

                let template_type_registration =
                    type_registry.get(resolved_symbol.template_type_id).unwrap();
                let field_infos = if let Ok(structure) =
                    template_type_registration.type_info().as_struct()
                {
                    StructOrStructVariant::Struct(structure)
                } else if let Ok(enumeration) = template_type_registration.type_info().as_enum() {
                    StructOrStructVariant::StructVariant(
                        enumeration
                            .variant(&symbol.1)
                            .unwrap()
                            .as_struct_variant()?,
                    )
                } else {
                    return Err(DynamicBsnLoaderError::TypeNotStruct);
                };

                let mut dynamic_struct = DynamicStruct::default();
                for field in fields.iter() {
                    let Some(field_info) = field_infos.get(&field.0) else {
                        return Err(DynamicBsnLoaderError::StructDoesntHaveField(
                            field.0.clone(),
                        ));
                    };
                    let reflect = self.convert_bsn_expr_to_reflect(
                        field.1,
                        app_type_registry,
                        field_info.ty().id(),
                    )?;
                    dynamic_struct.insert_boxed(field.0.clone(), reflect);
                }

                let app_type_registry = app_type_registry.clone();

                Box::new(ErasedTemplatePatch {
                    template_type_id: resolved_symbol.template_type_id,
                    app_type_registry: app_type_registry.clone(),
                    fun: move |reflect, _context| {
                        // This could be an enum variant
                        // (`some_crate::Enum::Variant`) or a unit struct.
                        // First, look for a struct.
                        let struct_type_path = symbol.as_path();
                        if !resolved_symbol.template_is_enum {
                            // This is a struct. Rebuild the target as a fresh `DynamicStruct` from
                            // its current (default) fields with the specified fields replaced
                            // wholesale. Replacing rather than `apply`-ing avoids the reflection
                            // kind/variant check that would reject e.g. a `HandleTemplate`/asset
                            // value destined for a `Handle` field.
                            let ReflectRef::Struct(current) = reflect.reflect_ref() else {
                                error!("Expected a struct: `{}`", struct_type_path);
                                return;
                            };
                            let mut rebuilt = DynamicStruct::default();
                            rebuilt.set_represented_type(current.get_represented_type_info());
                            for i in 0..current.field_len() {
                                if let (Some(name), Some(value)) =
                                    (current.name_at(i), current.field_at(i))
                                {
                                    rebuilt.insert_boxed(name.to_owned(), value.to_dynamic());
                                }
                            }
                            for i in 0..dynamic_struct.field_len() {
                                if let (Some(name), Some(value)) =
                                    (dynamic_struct.name_at(i), dynamic_struct.field_at(i))
                                {
                                    rebuilt.insert_boxed(name.to_owned(), value.to_dynamic());
                                }
                            }
                            *reflect = Box::new(rebuilt);
                            return;
                        }

                        // Enum struct variant: wrap DynamicStruct in DynamicEnum and apply onto the
                        // dynamic target (replaces the variant wholesale).
                        let dynamic_enum = DynamicEnum::new(
                            symbol.1.clone(),
                            DynamicVariant::Struct(dynamic_struct.to_dynamic_struct()),
                        );
                        let ReflectMut::Enum(reflect_enum) = reflect.reflect_mut() else {
                            error!("Expected an enum: `{}`", struct_type_path);
                            return;
                        };
                        reflect_enum.apply(&dynamic_enum);
                    },
                }) as Box<dyn Scene>
            }

            BsnPatch::NamedTuple(BsnNamedTuple(ref symbol, ref fields, is_template)) => {
                let symbol = symbol.clone();

                let type_registry = app_type_registry.read();
                let resolved_symbol =
                    symbol.resolve_type_or_enum_variant_to_template(&type_registry, is_template)?;

                let template_type_registration =
                    type_registry.get(resolved_symbol.template_type_id).unwrap();
                let field_infos = if let Ok(tuple_struct) =
                    template_type_registration.type_info().as_tuple_struct()
                {
                    tuple_struct.iter()
                } else if let Ok(enumeration) = template_type_registration.type_info().as_enum() {
                    enumeration
                        .variant(&symbol.1)
                        .unwrap()
                        .as_tuple_variant()?
                        .iter()
                } else {
                    return Err(DynamicBsnLoaderError::TypeNotNamedTuple);
                };

                let mut dynamic_tuple_struct = DynamicTupleStruct::default();
                for (field, field_info) in fields.iter().zip(field_infos) {
                    let reflect = self.convert_bsn_expr_to_reflect(
                        *field,
                        app_type_registry,
                        field_info.ty().id(),
                    )?;
                    dynamic_tuple_struct.insert_boxed(reflect);
                }

                let app_type_registry = app_type_registry.clone();

                Box::new(ErasedTemplatePatch {
                    template_type_id: resolved_symbol.template_type_id,
                    app_type_registry: app_type_registry.clone(),
                    fun: move |reflect, _context| {
                        // This could be an enum variant
                        // (`some_crate::Enum::Variant`) or a tuple struct.
                        // First, look for a struct.
                        let struct_type_path = symbol.as_path();
                        if !resolved_symbol.template_is_enum {
                            // This is a tuple struct. Rebuild the target as a fresh
                            // `DynamicTupleStruct` from its current (default) fields, replacing the
                            // leading fields with the specified values. Replacing rather than
                            // `apply`-ing avoids the reflection kind/variant check that would reject
                            // e.g. a `HandleTemplate`/asset value destined for a `Handle` field.
                            let ReflectRef::TupleStruct(current) = reflect.reflect_ref() else {
                                error!("Expected a tuple struct: `{}`", struct_type_path);
                                return;
                            };
                            let mut rebuilt = DynamicTupleStruct::default();
                            rebuilt.set_represented_type(current.get_represented_type_info());
                            for i in 0..current.field_len() {
                                let value = if i < dynamic_tuple_struct.field_len() {
                                    dynamic_tuple_struct.field(i).unwrap().to_dynamic()
                                } else {
                                    current.field(i).unwrap().to_dynamic()
                                };
                                rebuilt.insert_boxed(value);
                            }
                            *reflect = Box::new(rebuilt);
                            return;
                        }

                        // Enum tuple variant: wrap DynamicTupleStruct in DynamicEnum and apply
                        let dynamic_tuple = DynamicTuple::from_iter(
                            (0..dynamic_tuple_struct.field_len())
                                .map(|i| dynamic_tuple_struct.field(i).unwrap().to_dynamic()),
                        );
                        let dynamic_enum = DynamicEnum::new(
                            symbol.1.clone(),
                            DynamicVariant::Tuple(dynamic_tuple),
                        );
                        let ReflectMut::Enum(reflect_enum) = reflect.reflect_mut() else {
                            error!("Expected an enum: `{}`", struct_type_path);
                            return;
                        };
                        reflect_enum.apply(&dynamic_enum);
                    },
                }) as Box<dyn Scene>
            }

            BsnPatch::Relation(BsnRelation(ref relation_symbol, ref patches)) => {
                // FIXME: What a hack!
                if &*relation_symbol.as_path() != "bevy_ecs::hierarchy::Children" {
                    return Err(DynamicBsnLoaderError::OnlyChildRelationsSupported);
                }
                let related_template_list: Vec<_> = patches
                    .iter()
                    .map(|patches_id| {
                        // FIXME: seems fishy to throw away dependencies like this
                        Ok(self
                            .convert_bsn_patches_to_patch(*patches_id, app_type_registry)?
                            .scene)
                    })
                    .filter_map(Result::transpose)
                    .collect::<Result<Vec<_>, DynamicBsnLoaderError>>()?;
                Box::new(DynamicRelatedScenes {
                    relationship: TypeId::of::<ChildOf>(),
                    related_template_list,
                }) as Box<dyn Scene>
            }
        };

        Ok(ScenePatch {
            scene: Some(patch),
            dependencies: vec![],
            resolved: None,
        })
    }

    fn convert_bsn_expr_to_reflect(
        &self,
        expr_id: Entity,
        app_type_registry: &AppTypeRegistry,
        expected_template_type: TypeId,
    ) -> Result<Box<dyn PartialReflect>, DynamicBsnLoaderError> {
        let Some(expr) = self.0.get::<BsnExpr>(expr_id) else {
            return Err(DynamicBsnLoaderError::NoSuchAstNode);
        };

        let type_registry = app_type_registry.read();

        match *expr {
            BsnExpr::Var(BsnVar(ref symbol, is_template)) => {
                let resolved_symbol =
                    symbol.resolve_type_or_enum_variant_to_template(&type_registry, is_template)?;

                let template_type_registration =
                    type_registry.get(resolved_symbol.template_type_id).unwrap();

                let mut reflect =
                    create_reflect_default_from_type_registration(template_type_registration)?;

                // This could be an enum variant
                // (`some_crate::Enum::Variant`) or a unit struct.
                if !resolved_symbol.template_is_enum {
                    // This is a unit struct. Just instantiate it.
                    return Ok(reflect.into_partial_reflect());
                }

                // This is a unit enum variant.
                let ReflectMut::Enum(enum_reflect) = reflect.reflect_mut() else {
                    return Err(DynamicBsnLoaderError::UnknownType(
                        template_type_registration
                            .type_info()
                            .type_path()
                            .to_owned(),
                    ));
                };

                let dynamic_enum = DynamicEnum::new(symbol.1.clone(), DynamicVariant::Unit);
                enum_reflect.apply(&dynamic_enum);
                Ok(reflect.into_partial_reflect())
            }

            BsnExpr::Struct(ref bsn_struct) => {
                let resolved_symbol = bsn_struct
                    .0
                    .resolve_type_or_enum_variant_to_template(&type_registry, bsn_struct.2)?;

                let template_type_registration =
                    type_registry.get(resolved_symbol.template_type_id).unwrap();
                let mut reflect =
                    create_reflect_default_from_type_registration(template_type_registration)?;

                // This could be an enum variant (`some_crate::Enum::Variant`)
                // or a struct.
                if !resolved_symbol.template_is_enum {
                    // This is a struct. Build a *dynamic* representation seeded from the default
                    // (so unspecified fields keep their defaults), with the specified fields
                    // replaced wholesale. Keeping it dynamic lets fields hold `HandleTemplate` /
                    // `Some(HandleTemplate)` / nested asset values that the apply step resolves into
                    // concrete handles; applying those onto a concrete field would kind-mismatch.
                    let ReflectRef::Struct(default_struct) = reflect.reflect_ref() else {
                        return Err(DynamicBsnLoaderError::UnknownType(
                            template_type_registration
                                .type_info()
                                .type_path()
                                .to_owned(),
                        ));
                    };

                    let Ok(struct_info) = template_type_registration.type_info().as_struct() else {
                        return Err(DynamicBsnLoaderError::TypeNotStruct);
                    };

                    let mut dynamic_struct = DynamicStruct::default();
                    dynamic_struct.set_represented_type(reflect.get_represented_type_info());
                    for i in 0..default_struct.field_len() {
                        if let (Some(name), Some(value)) =
                            (default_struct.name_at(i), default_struct.field_at(i))
                        {
                            dynamic_struct.insert_boxed(name.to_owned(), value.to_dynamic());
                        }
                    }
                    for field in &bsn_struct.1 {
                        let Some(field_info) = struct_info.field(&field.0) else {
                            return Err(DynamicBsnLoaderError::StructDoesntHaveField(
                                field.0.clone(),
                            ));
                        };
                        let reflect = self.convert_bsn_expr_to_reflect(
                            field.1,
                            app_type_registry,
                            field_info.ty().id(),
                        )?;
                        dynamic_struct.insert_boxed(field.0.clone(), reflect);
                    }
                    return Ok(Box::new(dynamic_struct));
                }

                // Enum struct variant: build fields and wrap in DynamicEnum
                let enum_info = template_type_registration
                    .type_info()
                    .as_enum()
                    .map_err(|_| DynamicBsnLoaderError::TypeNotStruct)?;
                let variant_info = enum_info
                    .variant(&bsn_struct.0 .1)
                    .ok_or_else(|| {
                        DynamicBsnLoaderError::UnknownType(bsn_struct.0.as_path())
                    })?
                    .as_struct_variant()?;

                let mut dynamic_struct = DynamicStruct::default();
                for field in &bsn_struct.1 {
                    let Some(field_info) = variant_info.field(&field.0) else {
                        return Err(DynamicBsnLoaderError::StructDoesntHaveField(
                            field.0.clone(),
                        ));
                    };
                    let reflected = self.convert_bsn_expr_to_reflect(
                        field.1,
                        app_type_registry,
                        field_info.ty().id(),
                    )?;
                    dynamic_struct.insert_boxed(field.0.clone(), reflected);
                }

                let dynamic_enum = DynamicEnum::new(
                    bsn_struct.0 .1.clone(),
                    DynamicVariant::Struct(dynamic_struct),
                );
                let ReflectMut::Enum(reflect_enum) = reflect.reflect_mut() else {
                    return Err(DynamicBsnLoaderError::UnknownType(
                        template_type_registration
                            .type_info()
                            .type_path()
                            .to_owned(),
                    ));
                };
                reflect_enum.apply(&dynamic_enum);
                Ok(reflect.into_partial_reflect())
            }

            BsnExpr::NamedTuple(ref named_tuple) => {
                let resolved_symbol = named_tuple
                    .0
                    .resolve_type_or_enum_variant_to_template(&type_registry, named_tuple.2)?;

                let template_type_registration =
                    type_registry.get(resolved_symbol.template_type_id).unwrap();
                let mut reflect =
                    create_reflect_default_from_type_registration(template_type_registration)?;

                // This could be an enum tuple variant (`some_crate::Enum::Variant(..)`, e.g.
                // `AlphaMode::Mask(0.5)`) or a tuple struct.
                if !resolved_symbol.template_is_enum {
                    let Ok(tuple_info) = template_type_registration.type_info().as_tuple_struct()
                    else {
                        return Err(DynamicBsnLoaderError::TypeNotNamedTuple);
                    };

                    // Build a *dynamic* representation seeded from the default (so unspecified
                    // trailing fields keep their defaults), with the specified leading fields
                    // replaced wholesale. Keeping it dynamic lets fields hold `HandleTemplate` /
                    // `Some(HandleTemplate)` / nested asset values that the apply step resolves
                    // into concrete handles.
                    let mut parsed_fields = Vec::with_capacity(named_tuple.1.len());
                    for (field_id, field_info) in named_tuple.1.iter().zip(tuple_info.iter()) {
                        parsed_fields.push(self.convert_bsn_expr_to_reflect(
                            *field_id,
                            app_type_registry,
                            field_info.ty().id(),
                        )?);
                    }

                    let mut dynamic_tuple_struct = DynamicTupleStruct::default();
                    dynamic_tuple_struct.set_represented_type(reflect.get_represented_type_info());
                    let ReflectRef::TupleStruct(default_ts) = reflect.reflect_ref() else {
                        return Err(DynamicBsnLoaderError::TypeNotNamedTuple);
                    };
                    let mut parsed = parsed_fields.into_iter();
                    for i in 0..default_ts.field_len() {
                        match parsed.next() {
                            Some(value) => dynamic_tuple_struct.insert_boxed(value),
                            None => dynamic_tuple_struct
                                .insert_boxed(default_ts.field(i).unwrap().to_dynamic()),
                        }
                    }
                    return Ok(Box::new(dynamic_tuple_struct));
                }

                // Enum tuple variant: build the field values and wrap in a `DynamicEnum`, then
                // apply onto the default-instantiated enum (switching it to this variant). The
                // result is a concrete-typed enum value, so it applies cleanly onto the target
                // field (e.g. an `alpha_mode: AlphaMode`).
                let enum_info = template_type_registration
                    .type_info()
                    .as_enum()
                    .map_err(|_| DynamicBsnLoaderError::TypeNotNamedTuple)?;
                let variant_info = enum_info
                    .variant(&named_tuple.0 .1)
                    .ok_or_else(|| DynamicBsnLoaderError::UnknownType(named_tuple.0.as_path()))?
                    .as_tuple_variant()?;

                let mut dynamic_tuple = DynamicTuple::default();
                for (field_id, field_info) in named_tuple.1.iter().zip(variant_info.iter()) {
                    dynamic_tuple.insert_boxed(self.convert_bsn_expr_to_reflect(
                        *field_id,
                        app_type_registry,
                        field_info.ty().id(),
                    )?);
                }

                let dynamic_enum =
                    DynamicEnum::new(named_tuple.0 .1.clone(), DynamicVariant::Tuple(dynamic_tuple));
                let ReflectMut::Enum(reflect_enum) = reflect.reflect_mut() else {
                    return Err(DynamicBsnLoaderError::UnknownType(
                        template_type_registration.type_info().type_path().to_owned(),
                    ));
                };
                reflect_enum.apply(&dynamic_enum);
                Ok(reflect.into_partial_reflect())
            }

            BsnExpr::StringLit(ref string) => {
                let expected_type_registration = type_registry.get(expected_template_type).unwrap();

                // TODO: Support `&str`, `Cow<str>`, `Arc<str>`, etc. too?
                if expected_template_type == TypeId::of::<String>() {
                    let mut reflect =
                        create_reflect_default_from_type_registration(expected_type_registration)?;
                    reflect.apply(string);
                    return Ok(reflect.into_partial_reflect());
                }

                // An `Option<Handle<T>>` field: build the inner `HandleTemplate<T>::Path` exactly as
                // for a bare `Handle<T>`, then wrap it in `Option::Some` (a bare string is always
                // `Some`; we never author `None`). The apply step later resolves the inner
                // `HandleTemplate` to a concrete `Handle<T>`.
                let expected_type_path = expected_type_registration.type_info().type_path();
                if let Some(inner_handle_type_path) =
                    option_handle_inner_type_path(expected_type_path)
                {
                    let Some(handle_template) = handle_template_type_path(inner_handle_type_path)
                        .and_then(|path| type_registry.get_with_type_path(&path))
                    else {
                        return Err(DynamicBsnLoaderError::TypeMismatch);
                    };
                    let Some(reflect_convert) = handle_template.data::<ReflectConvert>() else {
                        return Err(DynamicBsnLoaderError::TypeMismatch);
                    };
                    let Ok(converted) = reflect_convert.try_convert_from(Box::new(string.clone()))
                    else {
                        return Err(DynamicBsnLoaderError::TypeMismatch);
                    };

                    let mut some_tuple = DynamicTuple::default();
                    some_tuple.insert_boxed(converted.into_partial_reflect());
                    let mut dynamic_option =
                        DynamicEnum::new("Some", DynamicVariant::Tuple(some_tuple));
                    dynamic_option.set_represented_type(Some(expected_type_registration.type_info()));
                    return Ok(Box::new(dynamic_option));
                }

                // Otherwise, look for a registered `String` -> `expected` conversion. The conversion
                // is registered as type data on the type to be converted *into* (see
                // `ReflectConvert` / `TypeRegistry::register_type_conversion`). Assigning a string
                // literal to a `Handle<T>` field works via this mechanism: `bevy_asset` registers a
                // `String` -> `HandleTemplate<T>` conversion, and `HandleTemplate<T>` is the
                // template for `Handle<T>`. We therefore convert into the field's *template* type
                // (`HandleTemplate<T>` for a `Handle<T>` field), which the apply step later builds
                // into the final `Handle<T>` using the `AssetServer`.
                let target_registration = handle_template_registration(
                    &type_registry,
                    expected_type_registration,
                )
                .unwrap_or(expected_type_registration);

                if let Some(reflect_convert) = target_registration.data::<ReflectConvert>() {
                    if let Ok(converted) =
                        reflect_convert.try_convert_from(Box::new(string.clone()))
                    {
                        return Ok(converted.into_partial_reflect());
                    }
                }

                Err(DynamicBsnLoaderError::TypeMismatch)
            }

            BsnExpr::FloatLit(float_lit) => {
                let mut reflect = create_reflect_default(&type_registry, expected_template_type)?;

                if expected_template_type == TypeId::of::<f32>() {
                    reflect.apply(&(float_lit as f32));
                    return Ok(reflect.into_partial_reflect());
                }
                if expected_template_type == TypeId::of::<f64>() {
                    reflect.apply(&float_lit);
                    return Ok(reflect.into_partial_reflect());
                }
                Err(DynamicBsnLoaderError::FloatLitTypeMismatch)
            }

            BsnExpr::BoolLit(bool_lit) => {
                let mut reflect = create_reflect_default(&type_registry, expected_template_type)?;

                if expected_template_type == TypeId::of::<bool>() {
                    reflect.apply(&bool_lit);
                    return Ok(reflect.into_partial_reflect());
                }
                Err(DynamicBsnLoaderError::TypeMismatch)
            }

            BsnExpr::List(ref items) => {
                let type_registration =
                    type_registry.get(expected_template_type).ok_or_else(|| {
                        DynamicBsnLoaderError::UnknownType(format!(
                            "TypeId {:?}",
                            expected_template_type
                        ))
                    })?;
                let list_info = type_registration
                    .type_info()
                    .as_list()
                    .map_err(|_| DynamicBsnLoaderError::TypeMismatch)?;
                let item_type_id = list_info.item_ty().id();

                let mut dynamic_list = DynamicList::default();
                for &item_id in items {
                    let reflect =
                        self.convert_bsn_expr_to_reflect(item_id, app_type_registry, item_type_id)?;
                    dynamic_list.push_box(reflect);
                }
                dynamic_list.set_represented_type(Some(type_registration.type_info()));
                Ok(Box::new(dynamic_list) as Box<dyn PartialReflect>)
            }

            BsnExpr::IntLit(int_lit) => {
                let mut reflect = create_reflect_default(&type_registry, expected_template_type)?;

                if expected_template_type == TypeId::of::<i8>() {
                    reflect.apply(&(int_lit as i8));
                    return Ok(reflect.into_partial_reflect());
                }
                if expected_template_type == TypeId::of::<u8>() {
                    reflect.apply(&(int_lit as u8));
                    return Ok(reflect.into_partial_reflect());
                }
                if expected_template_type == TypeId::of::<i16>() {
                    reflect.apply(&(int_lit as i16));
                    return Ok(reflect.into_partial_reflect());
                }
                if expected_template_type == TypeId::of::<u16>() {
                    reflect.apply(&(int_lit as u16));
                    return Ok(reflect.into_partial_reflect());
                }
                if expected_template_type == TypeId::of::<i32>() {
                    reflect.apply(&(int_lit as i32));
                    return Ok(reflect.into_partial_reflect());
                }
                if expected_template_type == TypeId::of::<u32>() {
                    reflect.apply(&(int_lit as u32));
                    return Ok(reflect.into_partial_reflect());
                }
                if expected_template_type == TypeId::of::<i64>() {
                    reflect.apply(&(int_lit as i64));
                    return Ok(reflect.into_partial_reflect());
                }
                if expected_template_type == TypeId::of::<u64>() {
                    reflect.apply(&(int_lit as u64));
                    return Ok(reflect.into_partial_reflect());
                }
                if expected_template_type == TypeId::of::<isize>() {
                    reflect.apply(&(int_lit as isize));
                    return Ok(reflect.into_partial_reflect());
                }
                if expected_template_type == TypeId::of::<usize>() {
                    reflect.apply(&(int_lit as usize));
                    return Ok(reflect.into_partial_reflect());
                }
                Err(DynamicBsnLoaderError::IntLitTypeMismatch)
            }
        }
    }

    pub fn create_patches(&mut self, patches: Vec<Entity>) -> Entity {
        self.0.spawn(BsnPatches(patches)).id()
    }

    pub fn create_patch(&mut self, patch: BsnPatch) -> Entity {
        self.0.spawn(patch).id()
    }

    pub fn create_expr(&mut self, expr: BsnExpr) -> Entity {
        self.0.spawn(expr).id()
    }

    pub fn create_name_patch(&mut self, name: String) -> Entity {
        let mut name_store = self.0.resource_mut::<BsnNameStore>();
        let index = match name_store.name_indices.get(&*name) {
            Some(index) => *index,
            None => {
                let index = name_store.next_name_index;
                name_store.next_name_index += 1;
                name_store.name_indices.insert(name.clone(), index);
                index
            }
        };
        self.create_patch(BsnPatch::Name(name, index))
    }
}

fn create_reflect_default(
    type_registry: &TypeRegistry,
    expected_template_type: TypeId,
) -> Result<Box<dyn Reflect>, DynamicBsnLoaderError> {
    let expected_type_registration = type_registry.get(expected_template_type).unwrap();
    create_reflect_default_from_type_registration(expected_type_registration)
}

fn create_reflect_default_from_type_registration(
    expected_type_registration: &TypeRegistration,
) -> Result<Box<dyn Reflect>, DynamicBsnLoaderError> {
    let Some(reflect_default) = expected_type_registration.data::<ReflectDefault>() else {
        return Err(DynamicBsnLoaderError::TypeDoesntImplementDefault(
            expected_type_registration
                .type_info()
                .type_path()
                .to_owned(),
        ));
    };
    Ok(reflect_default.default())
}

/// Given a `bevy_asset::handle::Handle<T>` type path, returns the type path of the corresponding
/// `bevy_asset::handle::HandleTemplate<T>` (the [`Template`](bevy_ecs::template::Template) that
/// produces the handle), or `None` if `handle_type_path` is not a `Handle<T>` type path.
fn handle_template_type_path(handle_type_path: &str) -> Option<String> {
    const HANDLE_PREFIX: &str = "bevy_asset::handle::Handle<";
    const HANDLE_TEMPLATE_PREFIX: &str = "bevy_asset::handle::HandleTemplate<";

    let inner = handle_type_path.strip_prefix(HANDLE_PREFIX)?;
    Some(format!("{HANDLE_TEMPLATE_PREFIX}{inner}"))
}

/// If `type_path` is `core::option::Option<bevy_asset::handle::Handle<T>>`, returns the inner
/// `bevy_asset::handle::Handle<T>` type path. Otherwise returns `None`.
fn option_handle_inner_type_path(type_path: &str) -> Option<&str> {
    const OPTION_PREFIX: &str = "core::option::Option<";
    const HANDLE_PREFIX: &str = "bevy_asset::handle::Handle<";

    let inner = type_path
        .strip_prefix(OPTION_PREFIX)?
        .strip_suffix('>')?;
    inner.starts_with(HANDLE_PREFIX).then_some(inner)
}

/// If `registration` is a `bevy_asset::handle::Handle<T>`, returns the registration for the
/// corresponding `bevy_asset::handle::HandleTemplate<T>` (the [`Template`](bevy_ecs::template::Template)
/// that produces the handle), if it is registered. Otherwise returns `None`.
///
/// This is how a string literal assigned to a `Handle<T>` field is resolved: the string is
/// converted into a `HandleTemplate<T>` (via [`ReflectConvert`]), which is later built into the
/// final `Handle<T>` using the `AssetServer`.
fn handle_template_registration<'a>(
    type_registry: &'a TypeRegistry,
    registration: &TypeRegistration,
) -> Option<&'a TypeRegistration> {
    let template_type_path = handle_template_type_path(registration.type_info().type_path())?;
    type_registry.get_with_type_path(&template_type_path)
}

pub struct MultiPatch(Vec<Box<dyn Scene>>);

impl Scene for MultiPatch {
    fn resolve(
        self,
        context: &mut ResolveContext,
        scene: &mut ResolvedScene,
    ) -> Result<(), ResolveSceneError> {
        for subscene in self.0 {
            subscene.resolve(context, scene)?;
        }

        Ok(())
    }

    fn register_dependencies(&self, dependencies: &mut SceneDependencies) {
        for subscene in self.0.iter() {
            subscene.register_dependencies(dependencies);
        }
    }
}

pub struct DynamicRelatedScenes {
    relationship: TypeId,
    related_template_list: Vec<Box<dyn Scene>>,
}

impl Scene for DynamicRelatedScenes {
    fn resolve(
        self,
        context: &mut ResolveContext,
        scene: &mut ResolvedScene,
    ) -> Result<(), ResolveSceneError> {
        if self.relationship != TypeId::of::<ChildOf>() {
            return Err(ResolveSceneError::UnsupportedRelationship);
        }

        let mut resolved_scenes = Vec::with_capacity(self.related_template_list.len());
        for child_scene in self.related_template_list {
            let mut resolved_scene = ResolvedScene::default();
            child_scene.resolve(context, &mut resolved_scene)?;
            resolved_scenes.push(resolved_scene);
        }

        let related = scene.get_or_insert_related_resolved_scenes::<ChildOf>();
        related.scenes.extend(resolved_scenes);

        Ok(())
    }

    fn register_dependencies(&self, dependencies: &mut SceneDependencies) {
        for scene in self.related_template_list.iter() {
            scene.register_dependencies(dependencies);
        }
    }
}

impl BsnSymbol {
    fn resolve_type_or_enum_variant_to_template(
        &self,
        type_registry: &TypeRegistry,
        is_template: bool,
    ) -> Result<ResolvedSymbol, DynamicBsnLoaderError> {
        // First, look for a unit struct.
        let unit_struct_type_path = self.as_path();
        if let Some(type_registration) = type_registry.get_with_type_path(&unit_struct_type_path) {
            return Ok(ResolvedSymbol::new(type_registration, false, is_template));
        }

        // Next, look for a unit enum variant.
        let Some(enum_type_path) = self.as_path_skip_last() else {
            return Err(DynamicBsnLoaderError::UnknownType(
                unit_struct_type_path.to_owned(),
            ));
        };
        let Some(type_registration) = type_registry.get_with_type_path(&enum_type_path) else {
            return Err(DynamicBsnLoaderError::UnknownType(
                enum_type_path.to_owned(),
            ));
        };
        Ok(ResolvedSymbol::new(type_registration, true, is_template))
    }

    fn as_path(&self) -> String {
        let mut path = String::new();
        for component in &self.0 {
            let _ = write!(&mut path, "{}::", &**component);
        }
        path.push_str(&self.1);
        path
    }

    fn as_path_skip_last(&self) -> Option<String> {
        if self.0.is_empty() {
            return None;
        }
        let mut enum_type_path = String::new();
        for component_index in 0..(self.0.len() - 1) {
            let _ = write!(&mut enum_type_path, "{}::", self.0[component_index]);
        }
        enum_type_path.push_str(self.0.last().unwrap());
        Some(enum_type_path)
    }
}

struct ErasedTemplatePatch<F>
where
    F: Fn(&mut Box<dyn PartialReflect>, &mut ResolveContext),
{
    pub fun: F,
    pub template_type_id: TypeId,
    // FIXME: Not a good place for this. Put it in the patch context instead?
    pub app_type_registry: AppTypeRegistry,
}

/// Stores the in-progress component as a *dynamic* representation (a `DynamicStruct` /
/// `DynamicTupleStruct` / `DynamicEnum` carrying the output component as its represented type).
///
/// Keeping it dynamic is what makes BSN field patching work for fields whose final type differs from
/// the value the loader parses: a `Handle<T>` field is given either a `HandleTemplate<T>` (string
/// path) or a raw asset value (inline). If we patched onto a *concrete* component, applying those
/// values onto the concrete `Handle<T>` field would panic with a reflection kind/variant mismatch.
/// Instead, dynamic fields are *replaced wholesale*; the handle fields are then resolved to concrete
/// `Handle<T>` values in [`build_handle_template_fields`] before `insert_reflect` materializes the
/// component via `FromReflect`.
struct DefaultDynamicErasedTemplate(Box<dyn PartialReflect>);

impl<F> Scene for ErasedTemplatePatch<F>
where
    F: Fn(&mut Box<dyn PartialReflect>, &mut ResolveContext) + Send + Sync + 'static,
{
    fn resolve(
        self,
        context: &mut ResolveContext,
        scene: &mut ResolvedScene,
    ) -> Result<(), ResolveSceneError> {
        let template_type_id = self.template_type_id;
        let app_type_registry = self.app_type_registry.clone();

        // Verify that everything is OK before we enter the closure below and
        // start unwrapping things.
        {
            let type_registry = app_type_registry.read();
            let Some(template_type_registration) = type_registry.get(template_type_id) else {
                return Err(ResolveSceneError::TypeNotReflectable);
            };
            if !template_type_registration.contains::<ReflectDefault>() {
                return Err(ResolveSceneError::TypeDoesntReflectDefault);
            };
        }

        let template =
            scene.get_or_insert_erased_template(context, self.template_type_id, move || {
                let reflect = {
                    let type_registry = app_type_registry.read();
                    let type_registration = type_registry.get(template_type_id).unwrap();
                    let reflect_default = type_registration.data::<ReflectDefault>().unwrap();
                    // Seed with a dynamic representation of the default value, preserving the
                    // represented type so it can be materialized via `FromReflect` later.
                    reflect_default.default().to_dynamic()
                };
                Box::new(DefaultDynamicErasedTemplate(reflect))
            });
        // The template was created (or cloned) by us as a `DefaultDynamicErasedTemplate`, so we can
        // recover mutable access to the underlying reflected value via `Any` downcasting.
        let Some(template) = (template as &mut dyn Any).downcast_mut::<DefaultDynamicErasedTemplate>()
        else {
            return Err(ResolveSceneError::TypeNotReflectable);
        };
        (self.fun)(&mut template.0, context);

        Ok(())
    }
}

impl ErasedComponentTemplate for DefaultDynamicErasedTemplate {
    unsafe fn apply(
        &self,
        context: &mut TemplateContext,
        _bundle_writer: &mut BundleWriter,
    ) -> EcsResult<(), BevyError> {
        // The stored value is a *dynamic* representation of the output component (built from
        // `ReflectDefault` and patched in `resolve`). Any field that was assigned a string-literal
        // asset path holds a `HandleTemplate<T>` value, and any inline asset value holds a raw
        // asset; resolve both to concrete `Handle<T>` values, then materialize the component via
        // `FromReflect` (`insert_reflect`).
        let mut output = self.0.to_dynamic();
        build_handle_template_fields(&mut output, context);

        context.entity.insert_reflect(output);
        Ok(())
    }

    fn clone_template(&self) -> Box<dyn ErasedComponentTemplate> {
        Box::new(DefaultDynamicErasedTemplate(self.0.to_dynamic()))
    }
}

/// Walks the fields of a reflected struct / tuple struct and resolves any field that names an asset
/// into a concrete `Handle<T>`, replacing the field value. The forms handled, all produced by
/// [`BsnAst::convert_bsn_expr_to_reflect`], are:
///
/// - A **string literal** assigned to a `Handle<T>` field becomes a `HandleTemplate<T>::Path`
///   (via [`ReflectConvert`]); here we read the path back out and load it through the
///   [`AssetServer`](bevy_asset::AssetServer).
/// - A **string literal** assigned to an `Option<Handle<T>>` field becomes a
///   `Some(HandleTemplate<T>::Path)`; here we resolve the inner handle and re-wrap in `Some`.
/// - An **inline asset value** (`SomeAsset { .. }`) is left in the field as a reflected asset value;
///   here we first recurse into the asset value to resolve *its* handle / option-handle fields, then
///   add it to `Assets<T>` via [`ReflectAsset`] and use the resulting handle.
///
/// In all cases [`ReflectHandle`] (registered on `Handle<T>`) types the resulting handle.
///
/// `reflect` is a *dynamic* representation (`DynamicStruct` / `DynamicTupleStruct`) of the output
/// component / inline asset; resolved fields are replaced wholesale (the container is rebuilt) so the
/// resolved value cleanly supersedes the `HandleTemplate`/asset value, avoiding a kind-mismatch
/// `.apply`.
fn build_handle_template_fields(
    reflect: &mut Box<dyn PartialReflect>,
    context: &mut TemplateContext,
) {
    let app_type_registry = context.resource::<AppTypeRegistry>().clone();

    let field_count = match reflect.reflect_ref() {
        ReflectRef::Struct(reflect_struct) => reflect_struct.field_len(),
        ReflectRef::TupleStruct(reflect_tuple_struct) => reflect_tuple_struct.field_len(),
        _ => return,
    };

    // Resolved replacement value per field index (`None` keeps the original).
    let mut replacements: Vec<Option<Box<dyn PartialReflect>>> = Vec::with_capacity(field_count);

    for i in 0..field_count {
        let replacement = {
            let field = match reflect.reflect_ref() {
                ReflectRef::Struct(reflect_struct) => reflect_struct.field_at(i),
                ReflectRef::TupleStruct(reflect_tuple_struct) => reflect_tuple_struct.field(i),
                _ => None,
            };
            field.and_then(|field| resolve_handle_field(field, &app_type_registry, context))
        };
        replacements.push(replacement);
    }

    // Rebuild the container with the resolved fields replaced. Build the new value fully
    // (borrowing the current value), then reassign.
    let represented_type = reflect.get_represented_type_info();
    let rebuilt: Option<Box<dyn PartialReflect>> = match reflect.reflect_ref() {
        ReflectRef::Struct(current) => {
            let mut rebuilt = DynamicStruct::default();
            rebuilt.set_represented_type(represented_type);
            for i in 0..field_count {
                let name = current.name_at(i).unwrap_or_default().to_owned();
                let value = match replacements[i].take() {
                    Some(value) => value,
                    None => current.field_at(i).unwrap().to_dynamic(),
                };
                rebuilt.insert_boxed(name, value);
            }
            Some(Box::new(rebuilt))
        }
        ReflectRef::TupleStruct(current) => {
            let mut rebuilt = DynamicTupleStruct::default();
            rebuilt.set_represented_type(represented_type);
            for i in 0..field_count {
                let value = match replacements[i].take() {
                    Some(value) => value,
                    None => current.field(i).unwrap().to_dynamic(),
                };
                rebuilt.insert_boxed(value);
            }
            Some(Box::new(rebuilt))
        }
        _ => None,
    };
    if let Some(rebuilt) = rebuilt {
        *reflect = rebuilt;
    }
}

/// Reads the `AssetPath` out of a reflected `HandleTemplate::Path(AssetPath)` value, along with the
/// `Handle<T>` type path it corresponds to. Returns `None` if `reflect` is not a `HandleTemplate`
/// in the `Path` variant (the only variant `convert_bsn_expr_to_reflect` produces from a string).
fn handle_template_path(reflect: &dyn PartialReflect) -> Option<(AssetPath<'static>, String)> {
    const HANDLE_TEMPLATE_PREFIX: &str = "bevy_asset::handle::HandleTemplate<";
    const HANDLE_PREFIX: &str = "bevy_asset::handle::Handle<";

    let type_path = reflect.get_represented_type_info()?.type_path();
    let inner = type_path.strip_prefix(HANDLE_TEMPLATE_PREFIX)?;
    let handle_type_path = format!("{HANDLE_PREFIX}{inner}");
    let ReflectRef::Enum(enum_value) = reflect.reflect_ref() else {
        return None;
    };
    if enum_value.variant_name() != "Path" {
        return None;
    }
    let asset_path = enum_value
        .field_at(0)?
        .try_downcast_ref::<AssetPath<'static>>()?
        .clone();
    Some((asset_path, handle_type_path))
}

/// Loads `asset_path` for the asset behind `handle_type_path` (a `Handle<T>` type path) and returns
/// the concrete typed `Handle<T>` reflected value, using [`ReflectHandle`].
fn load_typed_handle(
    handle_type_path: &str,
    asset_path: AssetPath<'static>,
    type_registry: &AppTypeRegistry,
    context: &mut TemplateContext,
) -> Option<Box<dyn Reflect>> {
    let reflect_handle = type_registry
        .read()
        .get_with_type_path(handle_type_path)
        .and_then(|registration| registration.data::<ReflectHandle>())
        .cloned()?;
    let untyped: UntypedHandle = context
        .resource::<AssetServer>()
        .load_builder()
        .load_erased(reflect_handle.asset_type_id(), asset_path);
    Some(reflect_handle.typed(untyped))
}

/// Resolves a single field value, returning its replacement (as a dynamic value) or `None` to keep
/// the original. Handles bare-`Handle` string paths, `Option<Handle>` (`Some(HandleTemplate::Path)`)
/// values, and inline asset values (recursing into the asset to resolve *its* handle fields before
/// adding it to `Assets<T>`).
fn resolve_handle_field(
    field: &dyn PartialReflect,
    type_registry: &AppTypeRegistry,
    context: &mut TemplateContext,
) -> Option<Box<dyn PartialReflect>> {
    const HANDLE_PREFIX: &str = "bevy_asset::handle::Handle<";

    // Bare `Handle<T>` field holding a `HandleTemplate::Path`.
    if let Some((asset_path, handle_type_path)) = handle_template_path(field) {
        return load_typed_handle(&handle_type_path, asset_path, type_registry, context)
            .map(|handle| handle.to_dynamic());
    }

    let field_type_path = field.get_represented_type_info()?.type_path();

    // `Option<Handle<T>>` field holding `Some(HandleTemplate::Path)`: resolve the inner handle and
    // re-wrap in `Some`.
    if let Some(inner_handle_type_path) = option_handle_inner_type_path(field_type_path) {
        let ReflectRef::Enum(enum_value) = field.reflect_ref() else {
            return None;
        };
        if enum_value.variant_name() != "Some" {
            return None;
        }
        let (asset_path, handle_type_path) = handle_template_path(enum_value.field_at(0)?)?;
        debug_assert_eq!(handle_type_path, inner_handle_type_path);
        let handle = load_typed_handle(&handle_type_path, asset_path, type_registry, context)?;
        let mut some_tuple = DynamicTuple::default();
        some_tuple.insert_boxed(handle.into_partial_reflect());
        let mut dynamic_option = DynamicEnum::new("Some", DynamicVariant::Tuple(some_tuple));
        dynamic_option.set_represented_type(field.get_represented_type_info());
        return Some(Box::new(dynamic_option));
    }

    // Inline asset value of a registered asset type: recurse to resolve its handle fields, then add
    // it to `Assets<T>` and replace the field with the resulting concrete `Handle<T>`.
    let is_asset = type_registry
        .read()
        .get_with_type_path(field_type_path)
        .is_some_and(|registration| registration.data::<ReflectAsset>().is_some());
    if !is_asset {
        return None;
    }

    let asset_type_path = field_type_path.to_owned();
    let handle_type_path = format!("{HANDLE_PREFIX}{asset_type_path}>");

    // Recurse into a dynamic copy of the asset value to resolve its nested handle / option-handle
    // fields before adding it, so the stored asset holds concrete `Handle`s.
    let mut asset_value = field.to_dynamic();
    build_handle_template_fields(&mut asset_value, context);

    let reflect_asset = type_registry
        .read()
        .get_with_type_path(&asset_type_path)
        .and_then(|registration| registration.data::<ReflectAsset>())
        .cloned()?;
    let reflect_handle = type_registry
        .read()
        .get_with_type_path(&handle_type_path)
        .and_then(|registration| registration.data::<ReflectHandle>())
        .cloned()?;
    let untyped = context
        .entity
        .world_scope(|world| reflect_asset.add(world, asset_value.as_partial_reflect()));
    Some(reflect_handle.typed(untyped).to_dynamic())
}

#[derive(Clone)]
struct ResolvedSymbol {
    template_type_id: TypeId,
    template_is_enum: bool,
}

impl ResolvedSymbol {
    fn new(
        type_registration: &TypeRegistration,
        template_is_enum: bool,
        _is_template: bool,
    ) -> ResolvedSymbol {
        // We build the output component type directly via reflection (using `ReflectDefault`). The
        // type registry does not expose an output -> template type mapping, so the "template type"
        // we operate on is the reflected type itself. Asset-handle fields are resolved separately
        // via `ReflectConvert` (see `convert_bsn_expr_to_reflect` / `build_handle_template_fields`).
        ResolvedSymbol {
            template_type_id: type_registration.type_id(),
            template_is_enum,
        }
    }
}

enum StructOrStructVariant<'a> {
    Struct(&'a StructInfo),
    StructVariant(&'a StructVariantInfo),
}

impl<'a> StructOrStructVariant<'a> {
    fn get(&self, field_name: &str) -> Option<&'a NamedField> {
        match *self {
            StructOrStructVariant::Struct(structure) => structure.field(field_name),
            StructOrStructVariant::StructVariant(struct_variant) => {
                struct_variant.field(field_name)
            }
        }
    }
}
