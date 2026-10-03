//! Lowering a parsed [`BsnDocument`] into a [`DynamicScene`]: symbol resolution, type-data
//! capture and value construction, all performed once at load time.

use alloc::sync::Arc;
use core::any::TypeId;

use bevy_asset::AssetPath;
use bevy_bsn::{BsnDocument, BsnNodeId, BsnNodeKind, BsnPatchPrefix, BsnPath, BsnValue, Span};
use bevy_ecs::{
    name::Name,
    reflect::{
        AppTypeRegistry, ReflectComponent, ReflectFromTemplate, ReflectRelationshipTarget,
        ReflectTemplate,
    },
    template::SceneEntityReference,
};
use bevy_platform::collections::HashMap;
use bevy_reflect::{
    std_traits::ReflectDefault, ReflectFromReflect, TypeInfo, TypeRegistration, TypeRegistry,
};
use thiserror::Error;

use crate::{
    scene::{
        DynamicName, DynamicPatch, DynamicPatchValue, DynamicRelation, DynamicScene,
        DynamicSceneEntity, DynamicSceneInner,
    },
    value::{self, EnumInput},
};
use bevy_scene::ScenePatch;

/// The deepest entity/value nesting the builder will walk.
///
/// The parser enforces its own nesting limit, but a [`BsnDocument`] can also be built by hand (and
/// can then be cyclic), so the builder guards against unbounded recursion as well.
const MAX_DEPTH: u32 = 128;

/// An error produced while lowering a [`BsnDocument`] into a [`DynamicScene`].
///
/// Every variant carries the [`Span`] of the offending piece of source text. The asset loader adds
/// the file name and turns the span into a line/column pair.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum DynamicSceneBuildError {
    /// A named type is not in the [`TypeRegistry`].
    #[error("`{type_path}` is not a registered type. Register it with `app.register_type::<{type_path}>()`.")]
    UnknownType {
        /// The type path that could not be resolved.
        type_path: String,
        /// Where the type was named.
        span: Span,
    },
    /// A type referenced indirectly — a template type, an output type or a field type — is not in
    /// the [`TypeRegistry`].
    #[error("`{type_path}` is not a registered type. Register it with `app.register_type::<{type_path}>()`; a component's generated template type also needs the `#[template(reflect)]` attribute before it can be registered.")]
    TypeNotRegistered {
        /// The type path that could not be resolved.
        type_path: String,
        /// Where the type was needed.
        span: Span,
    },
    /// A type that has to be default-constructed reflectively has no `ReflectDefault` type data.
    #[error("`{type_path}` has no `ReflectDefault` type data, so it cannot be created from a scene asset. Add `#[reflect(Default)]` to it.")]
    MissingReflectDefault {
        /// The type path that is missing `ReflectDefault`.
        type_path: String,
        /// Where the type was needed.
        span: Span,
    },
    /// A patched component type has no `ReflectComponent` type data.
    #[error("`{type_path}` has no `ReflectComponent` type data, so it cannot be inserted as a component. Add `#[reflect(Component)]` to it.")]
    MissingReflectComponent {
        /// The type path that is missing `ReflectComponent`.
        type_path: String,
        /// Where the component was named.
        span: Span,
    },
    /// An enum type does not have the named variant.
    #[error("`{type_path}` has no variant `{variant}`.")]
    UnknownVariant {
        /// The enum's type path.
        type_path: String,
        /// The variant that does not exist.
        variant: String,
        /// Where the variant was named.
        span: Span,
    },
    /// Named-field syntax was used on a type that is not a struct.
    #[error("`{type_path}` is not a struct, so it cannot be written with named fields.")]
    TypeNotStruct {
        /// The type path.
        type_path: String,
        /// Where the value was written.
        span: Span,
    },
    /// Positional-field syntax was used on a type that is not a tuple struct.
    #[error(
        "`{type_path}` is not a tuple struct, so it cannot be written with positional fields."
    )]
    TypeNotTupleStruct {
        /// The type path.
        type_path: String,
        /// Where the value was written.
        span: Span,
    },
    /// A named field does not exist on the type being patched.
    #[error("`{type_path}` has no field `{field}`.")]
    UnknownField {
        /// The type path.
        type_path: String,
        /// The field that does not exist.
        field: String,
        /// Where the field was written.
        span: Span,
    },
    /// The same field was given twice in one value.
    #[error("The field `{field}` of `{type_path}` is set more than once.")]
    DuplicateField {
        /// The type path.
        type_path: String,
        /// The duplicated field.
        field: String,
        /// Where the duplicate was written.
        span: Span,
    },
    /// More positional values were given than the tuple has fields.
    #[error("`{type_path}` has {expected} field(s), but {given} were given.")]
    TooManyTupleFields {
        /// The type path.
        type_path: String,
        /// How many values were written.
        given: usize,
        /// How many fields the type has.
        expected: usize,
        /// Where the value was written.
        span: Span,
    },
    /// An integer literal does not fit in the destination type.
    #[error("The literal `{value}` does not fit in `{type_path}`.")]
    IntegerOutOfRange {
        /// The literal.
        value: i128,
        /// The destination type path.
        type_path: String,
        /// Where the literal was written.
        span: Span,
    },
    /// An integer literal cannot be represented exactly by the destination floating-point type.
    #[error("The literal `{value}` cannot be represented exactly by `{type_path}`. Write it as a floating-point literal if the loss of precision is intended.")]
    LiteralNotRepresentable {
        /// The literal.
        value: i128,
        /// The destination type path.
        type_path: String,
        /// Where the literal was written.
        span: Span,
    },
    /// A value cannot be converted to the type the destination expects.
    #[error("Expected a value of type `{expected}`, but found `{found}`.")]
    ValueTypeMismatch {
        /// The type path of the value that was produced.
        found: String,
        /// The type path the destination expects.
        expected: String,
        /// Where the value was written.
        span: Span,
    },
    /// Reflectively applying a constructed value onto a default value failed.
    #[error("Failed to build a value of type `{type_path}`: {error}")]
    ValueApplyFailed {
        /// The type path being constructed.
        type_path: String,
        /// The underlying reflection error, rendered.
        error: String,
        /// Where the value was written.
        span: Span,
    },
    /// A relation block names a type that is not a usable relationship target.
    #[error("`{type_path}` cannot be used as a relationship in a scene. It must be a `RelationshipTarget` registered with `#[reflect(RelationshipTarget)]`.")]
    UnsupportedRelationship {
        /// The type path of the unsupported relationship target.
        type_path: String,
        /// Where the relation was written.
        span: Span,
    },
    /// A `@Type` scene-component entry was used, which scene assets do not support.
    #[error("`@{type_path}` (a scene component) cannot be used from a scene asset.")]
    SceneComponentUnsupported {
        /// The type path of the scene component.
        type_path: String,
        /// Where the entry was written.
        span: Span,
    },
    /// A `#Name` reference names an entity the document does not declare.
    #[error("No entity in this document is named `#{name}`.")]
    UnknownEntityName {
        /// The referenced name.
        name: String,
        /// Where the reference was written.
        span: Span,
    },
    /// A string that has to be an asset path is not one.
    #[error("`{path}` is not a valid asset path.")]
    InvalidAssetPath {
        /// The offending string.
        path: String,
        /// Where the string was written.
        span: Span,
    },
    /// The document describes more than one root entity.
    #[error(
        "a scene asset must contain exactly one root entity, found {count}. Wrap them in a \
         single root, or split them into separate files."
    )]
    MultipleRoots {
        /// How many roots the document has.
        count: usize,
        /// Where the second root starts.
        span: Span,
    },
    /// The document's arenas are inconsistent: a dangling id, a node of the wrong kind, or nesting
    /// deeper than the builder walks. A parsed document never produces this; a hand-built one can.
    #[error("The scene document is malformed: {message}")]
    MalformedDocument {
        /// What is wrong.
        message: String,
        /// Where the problem is, as far as it could be located.
        span: Span,
    },
}

impl DynamicSceneBuildError {
    /// The span of the source text this error refers to.
    pub fn span(&self) -> Span {
        match self {
            Self::UnknownType { span, .. }
            | Self::TypeNotRegistered { span, .. }
            | Self::MissingReflectDefault { span, .. }
            | Self::MissingReflectComponent { span, .. }
            | Self::UnknownVariant { span, .. }
            | Self::TypeNotStruct { span, .. }
            | Self::TypeNotTupleStruct { span, .. }
            | Self::UnknownField { span, .. }
            | Self::DuplicateField { span, .. }
            | Self::TooManyTupleFields { span, .. }
            | Self::IntegerOutOfRange { span, .. }
            | Self::LiteralNotRepresentable { span, .. }
            | Self::ValueTypeMismatch { span, .. }
            | Self::ValueApplyFailed { span, .. }
            | Self::UnsupportedRelationship { span, .. }
            | Self::SceneComponentUnsupported { span, .. }
            | Self::UnknownEntityName { span, .. }
            | Self::InvalidAssetPath { span, .. }
            | Self::MultipleRoots { span, .. }
            | Self::MalformedDocument { span, .. } => *span,
        }
    }

    /// Renders the error with the 1-based line and column of its span within `source`.
    pub fn render(&self, source: &str) -> String {
        let (line, column) = self.span().line_col(source);
        format!("{line}:{column}: {self}")
    }
}

/// Shared state for one document lowering.
pub(crate) struct BuildCx<'a> {
    /// The registry, read-locked for the whole build.
    pub(crate) registry: &'a TypeRegistry,
    /// The document being lowered.
    pub(crate) document: &'a BsnDocument,
    /// The digest of the document's asset path, used for `#Name` reference identity.
    pub(crate) source_path_hash: u64,
    /// Every `#Name` declared in the document, mapped to the node id of the entity that declares
    /// it. The first declaration of a name wins.
    pub(crate) names: HashMap<String, u32>,
    /// Asset dependencies discovered so far.
    pub(crate) dependencies: Vec<(TypeId, AssetPath<'static>)>,
    /// The current recursion depth.
    depth: u32,
}

impl<'a> BuildCx<'a> {
    /// Creates a build context for `document`, which was parsed from the asset path `source`.
    pub(crate) fn new(registry: &'a TypeRegistry, document: &'a BsnDocument, source: &str) -> Self {
        Self {
            registry,
            document,
            source_path_hash: SceneEntityReference::asset_path_hash(source),
            names: collect_names(document),
            dependencies: Vec::new(),
            depth: 0,
        }
    }

    /// Enters one level of recursion, failing if the document nests too deeply.
    ///
    /// Callers pair this with [`BuildCx::exit`] on the **success** path only: depth is only
    /// meaningful on the success path; any error aborts the whole build, so an unbalanced
    /// `enter` on an error path is never observed.
    pub(crate) fn enter(&mut self, span: Span) -> Result<(), DynamicSceneBuildError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(DynamicSceneBuildError::MalformedDocument {
                message: format!("nested more than {MAX_DEPTH} levels deep"),
                span,
            });
        }
        Ok(())
    }

    /// Leaves one level of recursion.
    pub(crate) fn exit(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Looks up a registration by [`TypeId`], or reports [`DynamicSceneBuildError::TypeNotRegistered`].
    pub(crate) fn registration(
        &self,
        type_id: TypeId,
        type_path: &str,
        span: Span,
    ) -> Result<&'a TypeRegistration, DynamicSceneBuildError> {
        self.registry
            .get(type_id)
            .ok_or_else(|| DynamicSceneBuildError::TypeNotRegistered {
                type_path: type_path.to_string(),
                span,
            })
    }
}

impl DynamicScene {
    /// Lowers a parsed `.bsn` document into a resolvable [`Scene`](bevy_scene::Scene).
    ///
    /// `source` is the asset path the document was parsed from; it is used for error messages and
    /// as the identity of the document's `#Name` entity references, so two spawns of the same
    /// asset resolve `#Name` consistently while two different assets never alias each other.
    ///
    /// `registry` is read once, up front: every symbol is resolved, every literal is converted and
    /// every piece of type data is cloned out of it here, so resolving and spawning the returned
    /// scene performs no further registry lookups.
    ///
    /// # Errors
    ///
    /// Returns a [`DynamicSceneBuildError`] carrying the [`Span`] of the offending source text if
    /// the document names an unregistered type, an unknown field or variant, a literal that does
    /// not fit its destination, or a value that cannot be converted.
    pub fn from_document(
        document: &BsnDocument,
        source: impl Into<Arc<str>>,
        registry: &AppTypeRegistry,
    ) -> Result<Self, DynamicSceneBuildError> {
        let source: Arc<str> = source.into();
        let (root, dependencies) = {
            let guard = registry.read();
            let mut cx = BuildCx::new(&guard, document, &source);

            let root = match document.roots.split_first() {
                None => DynamicSceneEntity {
                    base: None,
                    name: None,
                    patches: Vec::new(),
                    relations: Vec::new(),
                },
                Some((first, rest)) => {
                    if let Some(second) = rest.first() {
                        return Err(DynamicSceneBuildError::MultipleRoots {
                            count: document.roots.len(),
                            span: document.node(*second).map_or(Span::NONE, |node| node.span),
                        });
                    }
                    build_entity(&mut cx, *first, true)?
                }
            };

            (root, cx.dependencies)
        };

        Ok(DynamicScene(Arc::new(DynamicSceneInner {
            root,
            dependencies,
            source,
            type_registry: registry.clone(),
        })))
    }
}

/// Maps every `#Name` in the document to the node id of the entity that declares it.
///
/// Entities are visited in ascending node id, which is document order, so the *first* declaration
/// of a repeated name is the one references resolve to.
fn collect_names(document: &BsnDocument) -> HashMap<String, u32> {
    let mut names = HashMap::default();
    for node in document.entities() {
        if let BsnNodeKind::Entity {
            name: Some(name), ..
        } = &node.kind
        {
            names.entry(name.clone()).or_insert(node.id.0);
        }
    }
    names
}

/// Builds one entity, recursing into its relations.
fn build_entity(
    cx: &mut BuildCx,
    node_id: BsnNodeId,
    is_root: bool,
) -> Result<DynamicSceneEntity, DynamicSceneBuildError> {
    let node = cx
        .document
        .node(node_id)
        .ok_or_else(|| malformed(format!("no node with id {}", node_id.0), Span::NONE))?;
    let span = node.span;
    cx.enter(span)?;

    let BsnNodeKind::Entity {
        name,
        base,
        base_span,
        patches,
        relations,
        ..
    } = &node.kind
    else {
        return Err(malformed(
            format!("node {} is not an entity", node_id.0),
            span,
        ));
    };

    let base = match base {
        Some(base) => {
            let span = base_span.unwrap_or(span);
            let path = AssetPath::try_parse(base)
                .map_err(|_| DynamicSceneBuildError::InvalidAssetPath {
                    path: base.clone(),
                    span,
                })?
                .into_owned();
            // The root's base is registered directly by `Scene::register_dependencies`; nested
            // ones have to be flattened into the shared list.
            if !is_root {
                cx.dependencies
                    .push((TypeId::of::<ScenePatch>(), path.clone()));
            }
            Some(path)
        }
        None => None,
    };

    let name = name.as_ref().map(|name| DynamicName {
        name: Name::new(name.clone()),
        reference: SceneEntityReference::from_asset_hashed(cx.source_path_hash, node_id.0),
    });

    let mut built_patches = Vec::with_capacity(patches.len());
    for patch in patches {
        built_patches.push(build_patch(cx, *patch)?);
    }

    let mut built_relations = Vec::with_capacity(relations.len());
    for relation in relations {
        built_relations.push(build_relation(cx, *relation)?);
    }

    cx.exit();
    Ok(DynamicSceneEntity {
        base,
        name,
        patches: built_patches,
        relations: built_relations,
    })
}

/// Builds one `Children [ … ]`-style relation block.
fn build_relation(
    cx: &mut BuildCx,
    node_id: BsnNodeId,
) -> Result<DynamicRelation, DynamicSceneBuildError> {
    let node = cx
        .document
        .node(node_id)
        .ok_or_else(|| malformed(format!("no node with id {}", node_id.0), Span::NONE))?;
    let span = node.span;
    let BsnNodeKind::Relation {
        target_symbol,
        entities,
    } = &node.kind
    else {
        return Err(malformed(
            format!("node {} is not a relation", node_id.0),
            span,
        ));
    };

    let symbol = resolve_symbol(cx.registry, target_symbol, span)?;
    let data = symbol
        .registration
        .data::<ReflectRelationshipTarget>()
        .cloned()
        .ok_or_else(|| DynamicSceneBuildError::UnsupportedRelationship {
            type_path: symbol.registration.type_info().type_path().to_string(),
            span,
        })?;

    let mut children = Vec::with_capacity(entities.len());
    for entity in entities {
        children.push(build_entity(cx, *entity, false)?);
    }

    Ok(DynamicRelation { data, children })
}

/// Builds one `Type { … }` / `~Type(…)` entry.
fn build_patch(
    cx: &mut BuildCx,
    node_id: BsnNodeId,
) -> Result<DynamicPatch, DynamicSceneBuildError> {
    let node = cx
        .document
        .node(node_id)
        .ok_or_else(|| malformed(format!("no node with id {}", node_id.0), Span::NONE))?;
    let span = node.span;
    let BsnNodeKind::Patch {
        symbol,
        prefix,
        value,
    } = &node.kind
    else {
        return Err(malformed(
            format!("node {} is not a patch", node_id.0),
            span,
        ));
    };

    if *prefix == BsnPatchPrefix::SceneComponent {
        return Err(DynamicSceneBuildError::SceneComponentUnsupported {
            type_path: symbol.to_type_path(),
            span,
        });
    }

    let named = resolve_symbol(cx.registry, symbol, span)?;
    let (template, output) = template_registration(cx.registry, named.registration, *prefix, span)?;

    let template_type_path = template.type_info().type_path();
    let reflect_default = template.data::<ReflectDefault>().cloned().ok_or_else(|| {
        DynamicSceneBuildError::MissingReflectDefault {
            type_path: template_type_path.to_string(),
            span,
        }
    })?;
    let reflect_component = output.data::<ReflectComponent>().cloned().ok_or_else(|| {
        DynamicSceneBuildError::MissingReflectComponent {
            type_path: output.type_info().type_path().to_string(),
            span,
        }
    })?;
    let reflect_template = template.data::<ReflectTemplate>().cloned();
    let reflect_from_reflect = template.data::<ReflectFromReflect>().cloned();

    let value_node = cx
        .document
        .value(*value)
        .ok_or_else(|| malformed(format!("no value with id {}", value.0), span))?;
    let value_span = value_node.span;

    let value = match (&value_node.value, named.variant) {
        (BsnValue::Path(_), None) => DynamicPatchValue::Ensure,
        (BsnValue::Struct(_, fields), None) => DynamicPatchValue::Partial(
            value::build_partial_struct(cx, template, fields, value_span)?,
        ),
        (BsnValue::NamedTuple(_, items), None) => DynamicPatchValue::Partial(
            value::build_partial_tuple_struct(cx, template, items, value_span)?,
        ),
        (body, Some(variant)) => {
            let input = match body {
                BsnValue::Path(_) => EnumInput::Unit,
                BsnValue::Struct(_, fields) => EnumInput::Named(fields),
                BsnValue::NamedTuple(_, items) => EnumInput::Tuple(items),
                _ => {
                    return Err(malformed(
                        "a patch's value must be a path, struct or tuple".to_string(),
                        value_span,
                    ))
                }
            };
            let (full, partial) =
                value::build_enum_forms(cx, template, variant, input, value_span)?;
            DynamicPatchValue::EnumVariant {
                variant,
                full,
                partial,
            }
        }
        _ => {
            return Err(malformed(
                "a patch's value must be a path, struct or tuple".to_string(),
                value_span,
            ))
        }
    };

    Ok(DynamicPatch {
        template_type_id: template.type_id(),
        template_type_path,
        reflect_default,
        reflect_component,
        reflect_template,
        reflect_from_reflect,
        value,
    })
}

/// A type path resolved against the [`TypeRegistry`], plus the enum variant it named (if any).
pub(crate) struct ResolvedSymbol<'a> {
    /// The registration of the *named* type — the enum itself, when `variant` is `Some`.
    pub(crate) registration: &'a TypeRegistration,
    /// The name of the enum variant the last path segment named, if it named one.
    pub(crate) variant: Option<&'static str>,
}

/// Resolves a `.bsn` type path against the registry.
///
/// The path is tried as a full type path, then as a short type path (which is what makes both
/// `bevy_ecs::hierarchy::Children` and a bare `Children` work, and is also the only way a `use`-
/// style path such as `bevy_ecs::prelude::Children` can resolve, since the registry records the
/// type's *definition* path). Failing that, the parent path is tried, so that `a::b::MyEnum::Unit`
/// resolves to `a::b::MyEnum` plus the variant name.
pub(crate) fn resolve_symbol<'a>(
    registry: &'a TypeRegistry,
    path: &BsnPath,
    span: Span,
) -> Result<ResolvedSymbol<'a>, DynamicSceneBuildError> {
    let type_path = path.to_type_path();
    if let Some(registration) = lookup(registry, &type_path) {
        return Ok(ResolvedSymbol {
            registration,
            variant: None,
        });
    }

    if let Some(parent) = path.parent_type_path()
        && let Some(registration) = lookup(registry, &parent)
    {
        let last = path.last_ident();
        if let TypeInfo::Enum(info) = registration.type_info() {
            let variant =
                info.variant(last)
                    .ok_or_else(|| DynamicSceneBuildError::UnknownVariant {
                        type_path: parent.clone(),
                        variant: last.to_string(),
                        span,
                    })?;
            return Ok(ResolvedSymbol {
                registration,
                variant: Some(variant.name()),
            });
        }
    }

    Err(DynamicSceneBuildError::UnknownType { type_path, span })
}

/// Looks a type path up by full path, then by short path.
fn lookup<'a>(registry: &'a TypeRegistry, type_path: &str) -> Option<&'a TypeRegistration> {
    registry
        .get_with_type_path(type_path)
        .or_else(|| registry.get_with_short_type_path(type_path))
}

/// Maps the type a patch *names* to the template type its slot is keyed by and the component type
/// it produces.
///
/// * `Type { … }` names a component: its template comes from
///   [`ReflectFromTemplate`], defaulting to the component itself (the `Clone + Default` blanket).
/// * `~Type { … }` names a template directly: its output comes from [`ReflectTemplate`], defaulting
///   to the template itself.
pub(crate) fn template_registration<'a>(
    registry: &'a TypeRegistry,
    named: &'a TypeRegistration,
    prefix: BsnPatchPrefix,
    span: Span,
) -> Result<(&'a TypeRegistration, &'a TypeRegistration), DynamicSceneBuildError> {
    match prefix {
        BsnPatchPrefix::Template => {
            let Some(reflect_template) = named.data::<ReflectTemplate>() else {
                // No `ReflectTemplate` means the output type equals the template type.
                return Ok((named, named));
            };
            let output = registry
                .get(reflect_template.output_type_id)
                .ok_or_else(|| DynamicSceneBuildError::TypeNotRegistered {
                    type_path: format!("<{} as Template>::Output", named.type_info().type_path()),
                    span,
                })?;
            Ok((named, output))
        }
        BsnPatchPrefix::FromTemplate => {
            let Some(from_template) = named.data::<ReflectFromTemplate>() else {
                // No `ReflectFromTemplate` means the component *is* its own template.
                return Ok((named, named));
            };
            let template = registry
                .get(from_template.template_type_id)
                .ok_or_else(|| DynamicSceneBuildError::TypeNotRegistered {
                    type_path: from_template.template_type_path.to_string(),
                    span,
                })?;
            Ok((template, named))
        }
        BsnPatchPrefix::SceneComponent => Err(DynamicSceneBuildError::SceneComponentUnsupported {
            type_path: named.type_info().type_path().to_string(),
            span,
        }),
    }
}

/// Shorthand for [`DynamicSceneBuildError::MalformedDocument`].
pub(crate) fn malformed(message: String, span: Span) -> DynamicSceneBuildError {
    DynamicSceneBuildError::MalformedDocument { message, span }
}

#[cfg(test)]
mod tests {
    use bevy_bsn::{BsnNodeKind, BsnValue};

    use super::*;
    use crate::{
        scene::DynamicPatchValue,
        tests::{doc, test_app, Choice, Marker, Position, Sprite, SpriteTemplate},
    };

    /// Builds a scene from `source`, returning the error if there is one.
    fn try_build(source: &str) -> Result<DynamicScene, DynamicSceneBuildError> {
        let app = test_app();
        let registry = app.world().resource::<AppTypeRegistry>().clone();
        DynamicScene::from_document(&doc(source), "test.bsn", &registry)
    }

    /// Builds a scene from `source`, panicking with a rendered diagnostic on failure.
    fn build(source: &str) -> DynamicScene {
        match try_build(source) {
            Ok(scene) => scene,
            Err(error) => panic!("{}", error.render(source)),
        }
    }

    fn symbol_of(source: &str) -> Result<TypeId, DynamicSceneBuildError> {
        let app = test_app();
        let registry = app.world().resource::<AppTypeRegistry>().clone();
        let guard = registry.read();
        let path = BsnPath::from_type_path(source).expect("a valid path");
        resolve_symbol(&guard, &path, Span::NONE).map(|symbol| symbol.registration.type_id())
    }

    fn variant_of(source: &str) -> Result<Option<String>, DynamicSceneBuildError> {
        let app = test_app();
        let registry = app.world().resource::<AppTypeRegistry>().clone();
        let guard = registry.read();
        let path = BsnPath::from_type_path(source).expect("a valid path");
        resolve_symbol(&guard, &path, Span::NONE)
            .map(|symbol| symbol.variant.map(ToString::to_string))
    }

    #[test]
    fn resolve_symbol_unit_struct() {
        assert_eq!(symbol_of("Marker").unwrap(), TypeId::of::<Marker>());
        assert_eq!(variant_of("Marker").unwrap(), None);
    }

    #[test]
    fn resolve_symbol_enum_variant() {
        assert_eq!(symbol_of("Choice::Qux").unwrap(), TypeId::of::<Choice>());
        assert_eq!(variant_of("Choice::Qux").unwrap(), Some("Qux".to_string()));
    }

    #[test]
    fn resolve_symbol_fully_qualified() {
        assert_eq!(
            symbol_of("bevy_ecs::hierarchy::Children").unwrap(),
            TypeId::of::<bevy_ecs::hierarchy::Children>()
        );
    }

    #[test]
    fn resolve_symbol_unknown_type_errors() {
        let error = symbol_of("does::not::Exist").unwrap_err();
        assert!(
            matches!(&error, DynamicSceneBuildError::UnknownType { type_path, .. }
                if type_path == "does::not::Exist"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn resolve_symbol_unknown_variant_errors() {
        let error = symbol_of("Choice::Nope").unwrap_err();
        assert!(
            matches!(&error, DynamicSceneBuildError::UnknownVariant { variant, .. }
                if variant == "Nope"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn resolve_symbol_variant_of_a_non_enum_errors() {
        // `Marker` resolves, but it has no variants, so the whole path is simply unknown.
        let error = symbol_of("Marker::Nope").unwrap_err();
        assert!(
            matches!(&error, DynamicSceneBuildError::UnknownType { type_path, .. }
                if type_path == "Marker::Nope"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn every_error_variant_reports_its_span() {
        // The asset loader turns `span()` into the line/column of every diagnostic, so a variant
        // that forgets to report its span silently points at the top of the file.
        let span = Span::new(3, 7);
        let type_path = "some::Type".to_string();
        let errors = [
            DynamicSceneBuildError::UnknownType {
                type_path: type_path.clone(),
                span,
            },
            DynamicSceneBuildError::TypeNotRegistered {
                type_path: type_path.clone(),
                span,
            },
            DynamicSceneBuildError::MissingReflectDefault {
                type_path: type_path.clone(),
                span,
            },
            DynamicSceneBuildError::MissingReflectComponent {
                type_path: type_path.clone(),
                span,
            },
            DynamicSceneBuildError::UnknownVariant {
                type_path: type_path.clone(),
                variant: "V".to_string(),
                span,
            },
            DynamicSceneBuildError::TypeNotStruct {
                type_path: type_path.clone(),
                span,
            },
            DynamicSceneBuildError::TypeNotTupleStruct {
                type_path: type_path.clone(),
                span,
            },
            DynamicSceneBuildError::UnknownField {
                type_path: type_path.clone(),
                field: "f".to_string(),
                span,
            },
            DynamicSceneBuildError::DuplicateField {
                type_path: type_path.clone(),
                field: "f".to_string(),
                span,
            },
            DynamicSceneBuildError::TooManyTupleFields {
                type_path: type_path.clone(),
                given: 2,
                expected: 1,
                span,
            },
            DynamicSceneBuildError::IntegerOutOfRange {
                value: 300,
                type_path: type_path.clone(),
                span,
            },
            DynamicSceneBuildError::LiteralNotRepresentable {
                value: 300,
                type_path: type_path.clone(),
                span,
            },
            DynamicSceneBuildError::ValueTypeMismatch {
                found: "u32".to_string(),
                expected: type_path.clone(),
                span,
            },
            DynamicSceneBuildError::ValueApplyFailed {
                type_path: type_path.clone(),
                error: "why".to_string(),
                span,
            },
            DynamicSceneBuildError::UnsupportedRelationship {
                type_path: type_path.clone(),
                span,
            },
            DynamicSceneBuildError::SceneComponentUnsupported {
                type_path: type_path.clone(),
                span,
            },
            DynamicSceneBuildError::UnknownEntityName {
                name: "N".to_string(),
                span,
            },
            DynamicSceneBuildError::InvalidAssetPath {
                path: "p".to_string(),
                span,
            },
            DynamicSceneBuildError::MultipleRoots { count: 2, span },
            DynamicSceneBuildError::MalformedDocument {
                message: "why".to_string(),
                span,
            },
        ];

        let source = "0123456789";
        for error in errors {
            assert_eq!(error.span(), span, "{error:?}");
            // `render` prefixes the 1-based line and column of that span.
            assert!(
                error.render(source).starts_with("1:4: "),
                "unexpected rendering: {}",
                error.render(source)
            );
        }
    }

    #[test]
    fn unregistered_field_type_errors() {
        // A registry that knows a component but not one of its field types is a build error, not
        // a panic.
        let registry = TypeRegistry::empty();
        let document = BsnDocument::new();
        let cx = BuildCx::new(&registry, &document, "test.bsn");
        let error = cx
            .registration(TypeId::of::<u32>(), "u32", Span::NONE)
            .unwrap_err();
        assert!(
            matches!(&error, DynamicSceneBuildError::TypeNotRegistered { type_path, .. }
                if type_path == "u32"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn template_registration_needs_both_ends_registered() {
        // `~SpriteTemplate` needs the component the template builds…
        let mut registry = TypeRegistry::empty();
        registry.register::<SpriteTemplate>();
        let named = registry
            .get(TypeId::of::<SpriteTemplate>())
            .expect("just registered");
        let error = template_registration(&registry, named, BsnPatchPrefix::Template, Span::NONE)
            .unwrap_err();
        assert!(
            matches!(&error, DynamicSceneBuildError::TypeNotRegistered { type_path, .. }
                if type_path.contains("as Template>::Output")),
            "unexpected error: {error}"
        );

        // …and `Sprite` needs the template type it is built from.
        let mut registry = TypeRegistry::empty();
        registry.register::<Sprite>();
        let named = registry
            .get(TypeId::of::<Sprite>())
            .expect("just registered");
        let error =
            template_registration(&registry, named, BsnPatchPrefix::FromTemplate, Span::NONE)
                .unwrap_err();
        assert!(
            matches!(&error, DynamicSceneBuildError::TypeNotRegistered { type_path, .. }
                if type_path.ends_with("SpriteTemplate")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn template_without_reflect_default_errors() {
        // `Tricky` is registered with `#[reflect(Component)]` but no `Default`, so its template
        // value cannot be constructed.
        let error = try_build("Tricky::B").unwrap_err();
        assert!(
            matches!(&error, DynamicSceneBuildError::MissingReflectDefault { type_path, .. }
                if type_path.ends_with("Tricky")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn malformed_base_asset_path_errors() {
        let error = try_build(":\"bad#source://x.png\"\nMarker").unwrap_err();
        assert!(
            matches!(&error, DynamicSceneBuildError::InvalidAssetPath { path, .. }
                if path == "bad#source://x.png"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn template_type_id_uses_reflect_from_template() {
        let scene = build(r#"Sprite("a.png")"#);
        assert_eq!(
            scene.0.root.patches[0].template_type_id,
            TypeId::of::<SpriteTemplate>()
        );
        assert_ne!(
            scene.0.root.patches[0].template_type_id,
            TypeId::of::<Sprite>()
        );
    }

    #[test]
    fn template_type_id_defaults_to_self() {
        let scene = build("Position { x: 1.0 }");
        assert_eq!(
            scene.0.root.patches[0].template_type_id,
            TypeId::of::<Position>()
        );
    }

    #[test]
    fn tilde_template_uses_named_type_directly() {
        let scene = build(r#"~SpriteTemplate("a.png")"#);
        assert_eq!(
            scene.0.root.patches[0].template_type_id,
            TypeId::of::<SpriteTemplate>()
        );
        // Its output type differs from the template, so `ReflectTemplate` was captured.
        assert!(scene.0.root.patches[0].reflect_template.is_some());
    }

    #[test]
    fn unit_patch_is_ensure_only() {
        let scene = build("Marker");
        assert!(matches!(
            scene.0.root.patches[0].value,
            DynamicPatchValue::Ensure
        ));
    }

    #[test]
    fn enum_patch_stores_both_forms() {
        let scene = build("Choice::Bar { x: 1 }");
        let DynamicPatchValue::EnumVariant {
            variant,
            full,
            partial,
        } = &scene.0.root.patches[0].value
        else {
            panic!("expected an enum-variant patch");
        };
        assert_eq!(&**variant, "Bar");
        let bevy_reflect::ReflectRef::Enum(full) = full.reflect_ref() else {
            panic!("expected an enum");
        };
        let bevy_reflect::ReflectRef::Enum(partial) = partial.reflect_ref() else {
            panic!("expected an enum");
        };
        assert_eq!(full.field_len(), 3);
        assert_eq!(partial.field_len(), 1);
    }

    #[test]
    fn scene_component_entry_errors() {
        let error = try_build("@Marker").unwrap_err();
        assert!(
            matches!(
                &error,
                DynamicSceneBuildError::SceneComponentUnsupported { type_path, .. }
                    if type_path == "Marker"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn unknown_relationship_errors() {
        let error = try_build("Marker [ ]").unwrap_err();
        assert!(
            matches!(
                &error,
                DynamicSceneBuildError::UnsupportedRelationship { type_path, .. }
                    if type_path.ends_with("Marker")
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn unregistered_type_errors() {
        let error = try_build("NotAThing").unwrap_err();
        assert!(
            matches!(&error, DynamicSceneBuildError::UnknownType { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn build_errors_carry_spans() {
        let source = "Position { x: 1.0 }\nPosition { nope: 1.0 }";
        let error = try_build(source).unwrap_err();
        let span = error.span();
        assert!(
            span.text(source).contains("nope"),
            "span {span:?} covers {:?}",
            span.text(source)
        );
        assert_eq!(span.line_col(source).0, 2);
        assert!(error.render(source).starts_with("2:"));
    }

    #[test]
    fn nested_base_is_a_flat_dependency_and_the_root_base_is_not() {
        let scene = build(":\"root.bsn\"\nChildren [ (:\"child.bsn\") ]");
        let dependencies: Vec<_> = scene.dependencies().collect();
        assert_eq!(dependencies.len(), 1);
        assert_eq!(dependencies[0].1.path().to_string_lossy(), "child.bsn");

        let mut registered = bevy_scene::SceneDependencies::default();
        bevy_scene::Scene::register_dependencies(&scene, &mut registered);
        let paths: Vec<_> = registered
            .iter()
            .map(|dependency| dependency.path.path().to_string_lossy().to_string())
            .collect();
        assert_eq!(paths, ["root.bsn", "child.bsn"]);
    }

    #[test]
    fn empty_document_builds_an_empty_scene() {
        let app = test_app();
        let registry = app.world().resource::<AppTypeRegistry>().clone();
        let scene =
            DynamicScene::from_document(&BsnDocument::new(), "test.bsn", &registry).unwrap();
        assert!(scene.0.root.patches.is_empty());
        assert!(scene.0.root.relations.is_empty());
    }

    #[test]
    fn multiple_roots_error() {
        let app = test_app();
        let registry = app.world().resource::<AppTypeRegistry>().clone();
        let mut document = BsnDocument::new();
        for _ in 0..2 {
            let id = document.push_node(BsnNodeKind::Entity {
                name: None,
                name_span: None,
                base: None,
                base_span: None,
                patches: Vec::new(),
                relations: Vec::new(),
            });
            document.push_root(id);
        }
        assert!(matches!(
            DynamicScene::from_document(&document, "test.bsn", &registry),
            Err(DynamicSceneBuildError::MultipleRoots { count: 2, .. })
        ));
    }

    #[test]
    fn malformed_documents_error_rather_than_panic() {
        let app = test_app();
        let registry = app.world().resource::<AppTypeRegistry>().clone();

        // A root id that points at nothing.
        let mut document = BsnDocument::new();
        document.push_root(BsnNodeId(9));
        assert!(DynamicScene::from_document(&document, "test.bsn", &registry).is_err());

        // A root id that points at a patch rather than an entity.
        let mut document = BsnDocument::new();
        let patch = document.push_patch(
            BsnPatchPrefix::FromTemplate,
            BsnPath::from_segments(["Marker"]),
            bevy_bsn::PatchBody::Unit,
        );
        document.push_root(patch);
        assert!(DynamicScene::from_document(&document, "test.bsn", &registry).is_err());

        // An entity whose patch list points at an entity.
        let mut document = BsnDocument::new();
        let inner = document.push_node(BsnNodeKind::Entity {
            name: None,
            name_span: None,
            base: None,
            base_span: None,
            patches: Vec::new(),
            relations: Vec::new(),
        });
        let root = document.push_node(BsnNodeKind::Entity {
            name: None,
            name_span: None,
            base: None,
            base_span: None,
            patches: vec![inner],
            relations: Vec::new(),
        });
        document.push_root(root);
        assert!(DynamicScene::from_document(&document, "test.bsn", &registry).is_err());

        // A patch whose value id dangles.
        let mut document = BsnDocument::new();
        let patch = document.push_node(BsnNodeKind::Patch {
            symbol: BsnPath::from_segments(["Marker"]),
            prefix: BsnPatchPrefix::FromTemplate,
            value: bevy_bsn::BsnValueId(42),
        });
        let root = document.push_node(BsnNodeKind::Entity {
            name: None,
            name_span: None,
            base: None,
            base_span: None,
            patches: vec![patch],
            relations: Vec::new(),
        });
        document.push_root(root);
        assert!(DynamicScene::from_document(&document, "test.bsn", &registry).is_err());

        // A patch whose value is a bare literal rather than a path/struct/tuple.
        let mut document = BsnDocument::new();
        let value = document.push_value(BsnValue::Int(1));
        let patch = document.push_node(BsnNodeKind::Patch {
            symbol: BsnPath::from_segments(["Marker"]),
            prefix: BsnPatchPrefix::FromTemplate,
            value,
        });
        let root = document.push_node(BsnNodeKind::Entity {
            name: None,
            name_span: None,
            base: None,
            base_span: None,
            patches: vec![patch],
            relations: Vec::new(),
        });
        document.push_root(root);
        assert!(DynamicScene::from_document(&document, "test.bsn", &registry).is_err());

        // An entity whose relation list points at an entity rather than a relation.
        let mut document = BsnDocument::new();
        let inner = document.push_node(BsnNodeKind::Entity {
            name: None,
            name_span: None,
            base: None,
            base_span: None,
            patches: Vec::new(),
            relations: Vec::new(),
        });
        let root = document.push_node(BsnNodeKind::Entity {
            name: None,
            name_span: None,
            base: None,
            base_span: None,
            patches: Vec::new(),
            relations: vec![inner],
        });
        document.push_root(root);
        assert!(DynamicScene::from_document(&document, "test.bsn", &registry).is_err());

        // An enum-variant patch whose value is a bare literal.
        let mut document = BsnDocument::new();
        let value = document.push_value(BsnValue::Int(1));
        let patch = document.push_node(BsnNodeKind::Patch {
            symbol: BsnPath::from_segments(["Choice", "Qux"]),
            prefix: BsnPatchPrefix::FromTemplate,
            value,
        });
        let root = document.push_node(BsnNodeKind::Entity {
            name: None,
            name_span: None,
            base: None,
            base_span: None,
            patches: vec![patch],
            relations: Vec::new(),
        });
        document.push_root(root);
        assert!(matches!(
            DynamicScene::from_document(&document, "test.bsn", &registry),
            Err(DynamicSceneBuildError::MalformedDocument { .. })
        ));

        // An extreme integer literal.
        assert!(matches!(
            try_build("Position { x: 170141183460469231731687303715884105727 }"),
            Err(DynamicSceneBuildError::LiteralNotRepresentable { .. })
        ));
    }

    #[test]
    fn cyclic_entity_documents_error_rather_than_overflow() {
        let app = test_app();
        let registry = app.world().resource::<AppTypeRegistry>().clone();

        // An entity whose `Children` relation contains the entity itself.
        let mut document = BsnDocument::new();
        let relation = document.push_node(BsnNodeKind::Relation {
            target_symbol: BsnPath::from_segments(["Children"]),
            entities: Vec::new(),
        });
        let root = document.push_node(BsnNodeKind::Entity {
            name: None,
            name_span: None,
            base: None,
            base_span: None,
            patches: Vec::new(),
            relations: vec![relation],
        });
        document.nodes[relation.0 as usize].kind = BsnNodeKind::Relation {
            target_symbol: BsnPath::from_segments(["Children"]),
            entities: vec![root],
        };
        document.push_root(root);

        assert!(matches!(
            DynamicScene::from_document(&document, "test.bsn", &registry),
            Err(DynamicSceneBuildError::MalformedDocument { .. })
        ));
    }

    /// Builds a linear `Children` chain exactly `depth` entities deep (root = depth 1).
    fn chain_document(depth: u32) -> BsnDocument {
        let mut document = BsnDocument::new();
        let mut child: Option<BsnNodeId> = None;
        for _ in 0..depth {
            let relations = match child {
                Some(child) => vec![document.push_node(BsnNodeKind::Relation {
                    target_symbol: BsnPath::from_segments(["Children"]),
                    entities: vec![child],
                })],
                None => Vec::new(),
            };
            child = Some(document.push_node(BsnNodeKind::Entity {
                name: None,
                name_span: None,
                base: None,
                base_span: None,
                patches: Vec::new(),
                relations,
            }));
        }
        document.push_root(child.unwrap());
        document
    }

    /// The depth guard is exact: `MAX_DEPTH` builds, one more errors — and `exit` really
    /// unwinds, so unbounded *width* at shallow depth never trips the guard.
    #[test]
    fn depth_guard_boundary_is_exact() {
        let app = test_app();
        let registry = app.world().resource::<AppTypeRegistry>().clone();

        assert!(
            DynamicScene::from_document(&chain_document(MAX_DEPTH), "test.bsn", &registry).is_ok()
        );
        assert!(matches!(
            DynamicScene::from_document(&chain_document(MAX_DEPTH + 1), "test.bsn", &registry),
            Err(DynamicSceneBuildError::MalformedDocument { .. })
        ));

        // 200 siblings at depth 2: only `exit` decrementing per child keeps this legal.
        let mut document = BsnDocument::new();
        let children: Vec<BsnNodeId> = (0..200)
            .map(|_| {
                document.push_node(BsnNodeKind::Entity {
                    name: None,
                    name_span: None,
                    base: None,
                    base_span: None,
                    patches: Vec::new(),
                    relations: Vec::new(),
                })
            })
            .collect();
        let relation = document.push_node(BsnNodeKind::Relation {
            target_symbol: BsnPath::from_segments(["Children"]),
            entities: children,
        });
        let root = document.push_node(BsnNodeKind::Entity {
            name: None,
            name_span: None,
            base: None,
            base_span: None,
            patches: Vec::new(),
            relations: vec![relation],
        });
        document.push_root(root);
        assert!(DynamicScene::from_document(&document, "test.bsn", &registry).is_ok());
    }
}
