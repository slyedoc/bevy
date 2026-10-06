//! The [`DynamicScene`] data model and its [`Scene`] implementation.

use alloc::sync::Arc;
use core::any::TypeId;

use bevy_asset::AssetPath;
use bevy_ecs::{
    name::Name,
    reflect::{AppTypeRegistry, ReflectComponent, ReflectRelationshipTarget, ReflectTemplate},
    template::SceneEntityReference,
};
use bevy_reflect::{std_traits::ReflectDefault, PartialReflect, ReflectFromReflect, ReflectRef};

use crate::template::DynamicComponentTemplate;
use bevy_scene::{
    erased_template_as_partial_reflect_mut, CachedSceneAsset, NameEntityReference, ResolveContext,
    SceneBase,
    ResolveSceneError, ResolvedScene, Scene, SceneDependencies, ScenePatch,
};

/// A [`Scene`] built from a parsed `.bsn` document.
///
/// Create one with [`DynamicScene::from_document`]. Cloning is cheap (the whole document lowering
/// lives behind an [`Arc`]), which lets a [`ScenePatch`] retain a copy for hot reload.
///
/// A `DynamicScene` describes exactly one root entity, mirroring [`Scene`]'s "always a single root
/// entity" rule. Documents with more than one root are rejected at build time.
#[derive(Clone)]
pub struct DynamicScene(pub(crate) Arc<DynamicSceneInner>);

impl core::fmt::Debug for DynamicScene {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DynamicScene")
            .field("source", &self.0.source)
            .field("dependencies", &self.0.dependencies.len())
            .finish_non_exhaustive()
    }
}

impl DynamicScene {
    /// Every asset dependency discovered while building this scene, flattened over the whole
    /// document: `(asset type id, path)` pairs, plus one entry per nested `:"base.bsn"` include.
    ///
    /// The root entity's own base include is *not* in this list; it is registered separately by
    /// [`Scene::register_dependencies`].
    pub fn dependencies(&self) -> impl Iterator<Item = (TypeId, &AssetPath<'static>)> {
        self.0
            .dependencies
            .iter()
            .map(|(type_id, path)| (*type_id, path))
    }
}

/// The shared, immutable body of a [`DynamicScene`].
pub(crate) struct DynamicSceneInner {
    /// The document's single root entity.
    pub(crate) root: DynamicSceneEntity,
    /// Flattened asset dependencies (see [`DynamicScene::dependencies`]).
    pub(crate) dependencies: Vec<(TypeId, AssetPath<'static>)>,
    /// The asset path this document was parsed from.
    pub(crate) source: Arc<str>,
    /// A handle on the app's type registry, needed by
    /// [`DynamicComponentTemplate::apply`](crate::DynamicComponentTemplate) at spawn time (where
    /// no registry is otherwise reachable) and by the typed-slot recovery path in
    /// [`resolve_patch`].
    pub(crate) type_registry: AppTypeRegistry,
}

/// One entity's worth of pre-resolved instructions, in document order.
pub(crate) struct DynamicSceneEntity {
    /// `:"base.bsn"`. Resolved *first*, because [`ResolvedScene::include_cached`] rejects a late
    /// include and everything after it depends on the cached scene being in place.
    pub(crate) base: Option<AssetPath<'static>>,
    /// `#Name`.
    pub(crate) name: Option<DynamicName>,
    /// Component/template patches, in document order.
    pub(crate) patches: Vec<DynamicPatch>,
    /// Relations (`Children [ … ]`), in document order.
    pub(crate) relations: Vec<DynamicRelation>,
}

/// A `#Name` declaration and the reference identity it establishes.
pub(crate) struct DynamicName {
    pub(crate) name: Name,
    pub(crate) reference: SceneEntityReference,
}

/// A `Children [ … ]`-style relation block.
pub(crate) struct DynamicRelation {
    /// Cloned from the relationship-target type's registration at build time.
    pub(crate) data: ReflectRelationshipTarget,
    /// The related entity scenes, in document order.
    pub(crate) children: Vec<DynamicSceneEntity>,
}

/// A single `Type { … }` / `Type(…)` / `Type::Variant { … }` / `~Type { … }` entry.
pub(crate) struct DynamicPatch {
    /// The **real** template [`TypeId`] — the slot key in the [`ResolvedScene`]. This is what makes
    /// dynamic and `bsn!` patches of the same component merge.
    pub(crate) template_type_id: TypeId,
    /// Type path of the template type. Errors only.
    pub(crate) template_type_path: &'static str,

    /// Constructs a fresh template value. From the template type's registration.
    pub(crate) reflect_default: ReflectDefault,
    /// Inserts the built output into a `BundleWriter`. From the *output* type's registration.
    pub(crate) reflect_component: ReflectComponent,
    /// Present iff the template's output type differs from the template type.
    pub(crate) reflect_template: Option<ReflectTemplate>,
    /// Used by the fallback ladder in [`clone_template`], implemented for the dynamic path by
    /// [`DynamicComponentTemplate`].
    ///
    /// [`clone_template`]: bevy_scene::ErasedTemplate::clone_template
    pub(crate) reflect_from_reflect: Option<ReflectFromReflect>,

    /// What to do to the template value at resolve time.
    pub(crate) value: DynamicPatchValue,
}

/// What a [`DynamicPatch`] does to its template slot.
pub(crate) enum DynamicPatchValue {
    /// `Foo` with no fields: ensure the slot exists, change nothing.
    Ensure,
    /// `Foo { a: 1 }` / `Foo(1)`: apply this partial reflected value on top of the current one.
    Partial(Box<dyn PartialReflect>),
    /// `Foo::Bar { x: 1 }` / `Foo::Qux`: "match-or-reset", mirroring what the `bsn!` macro emits.
    EnumVariant {
        /// The target variant's name.
        variant: &'static str,
        /// Every non-ignored field of the variant defaulted, then the supplied fields overlaid.
        /// Applied when the current variant *differs*, because a reflect variant switch requires a
        /// complete field set.
        full: Box<dyn PartialReflect>,
        /// Only the supplied fields. Applied when the current variant already matches, so that
        /// untouched fields survive.
        partial: Box<dyn PartialReflect>,
    },
}

impl Scene for DynamicScene {
    fn resolve(
        self,
        context: &mut ResolveContext,
        scene: &mut ResolvedScene,
    ) -> Result<(), ResolveSceneError> {
        let inner = &*self.0;
        resolve_entity(&inner.root, &inner.type_registry, context, scene)
    }

    fn register_dependencies(&self, dependencies: &mut SceneDependencies) {
        if let Some(base) = &self.0.root.base {
            dependencies.register::<ScenePatch>(base.clone());
        }
        for (type_id, path) in &self.0.dependencies {
            dependencies.register_erased(*type_id, path.clone());
        }
    }
}

/// Resolves one entity of the document into `scene`, recursing into its relations.
fn resolve_entity(
    entity: &DynamicSceneEntity,
    registry: &AppTypeRegistry,
    context: &mut ResolveContext,
    scene: &mut ResolvedScene,
) -> Result<(), ResolveSceneError> {
    // (1) The base include has to come first: `include_cached` rejects a late include, and the
    //     copy-on-write behavior of every template access below depends on `context.cached`.
    if let Some(base) = &entity.base {
        CachedSceneAsset(base.clone()).resolve(context, scene)?;
        // Recorded so a writer can save only what differs from the base.
        *scene.get_or_insert_template::<SceneBase>(context) = SceneBase(base.to_string());
    }

    // (2) The `#Name`, through the same inline path the `bsn!` macro uses, so that the `Name`
    //     component participates in copy-on-write like any other template.
    if let Some(name) = &entity.name {
        NameEntityReference {
            name: name.name.clone(),
            reference: name.reference,
        }
        .resolve_inline(context, scene);
    }

    // (3) Patches, in document order.
    for patch in &entity.patches {
        resolve_patch(patch, registry, context, scene)?;
    }

    // (4) Relations, in document order.
    for relation in &entity.relations {
        // A related entity must not see the *parent's* cached scene: it would clone the base
        // root's templates into a child that has no cached scene of its own, which then panics
        // inside `get_or_insert_erased_template`. Children of a cached base are appended by that
        // cached scene at apply time instead.
        let saved = context.cached.take();
        let related = scene.get_or_insert_related_resolved_scenes_erased(&relation.data);
        let mut result = Ok(());
        for child in &relation.children {
            // Reset per child, not just per relation block: a child with its own base sets
            // `context.cached` (via `CachedSceneAsset::resolve`) and nothing restores it, so a
            // later sibling would otherwise resolve against a cached scene it does not own.
            context.cached = None;
            let mut child_scene = ResolvedScene::default();
            result = resolve_entity(child, registry, context, &mut child_scene);
            if result.is_err() {
                break;
            }
            related.scenes.push(child_scene);
        }
        context.cached = saved;
        result?;
    }

    Ok(())
}

/// Applies a single [`DynamicPatch`] to its template slot in `scene`.
fn resolve_patch(
    patch: &DynamicPatch,
    registry: &AppTypeRegistry,
    context: &mut ResolveContext,
    scene: &mut ResolvedScene,
) -> Result<(), ResolveSceneError> {
    // The type data is only cloned when the slot has to be created: `default` is an `FnOnce` that
    // borrows `patch` and `registry`, both disjoint from the `&mut` borrows of `scene`/`context`.
    let erased = scene.get_or_insert_erased_template(context, patch.template_type_id, || {
        Box::new(DynamicComponentTemplate::new(
            patch.template_type_id,
            patch.reflect_default.default(),
            patch.reflect_component.clone(),
            patch.reflect_template.clone(),
            patch.reflect_default.clone(),
            patch.reflect_from_reflect.clone(),
            registry.clone(),
        ))
    });

    // Prefer the registry the resolve entry point already read-locked: taking a second read lock
    // on the same registry from inside `Scene::resolve` risks deadlocking against a queued writer.
    let guard;
    let type_registry = match context.type_registry {
        Some(type_registry) => type_registry,
        None => {
            guard = registry.read();
            &guard
        }
    };
    let Some(value) = erased_template_as_partial_reflect_mut(erased, type_registry) else {
        return Err(ResolveSceneError::UnpatchableTemplate {
            type_path: patch.template_type_path.to_string(),
        });
    };

    match &patch.value {
        DynamicPatchValue::Ensure => {}
        DynamicPatchValue::Partial(partial) => {
            value
                .try_apply(&**partial)
                .map_err(|error| ResolveSceneError::ApplyFailed {
                    type_path: patch.template_type_path.to_string(),
                    error,
                })?;
        }
        DynamicPatchValue::EnumVariant {
            variant,
            full,
            partial,
        } => {
            // The reflection equivalent of the macro's `if !matches!(node, T::V { .. }) { *node =
            // T::default_v(); }` followed by the field assignments.
            let matches_variant = match value.reflect_ref() {
                ReflectRef::Enum(current) => current.variant_name() == *variant,
                _ => false,
            };
            let source = if matches_variant { partial } else { full };
            value
                .try_apply(&**source)
                .map_err(|error| ResolveSceneError::ApplyFailed {
                    type_path: patch.template_type_path.to_string(),
                    error,
                })?;
        }
    }

    Ok(())
}
