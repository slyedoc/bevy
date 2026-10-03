//! Functionality that relates to the [`Template`] trait.
pub use bevy_ecs_macros::FromTemplate;

use core::{hash::Hash, ops::Deref};

use crate::{
    component::Mutable,
    entity::Entity,
    error::{BevyError, Result},
    resource::Resource,
    world::{EntityWorldMut, Mut, World},
};
use alloc::vec::Vec;
use bevy_platform::{collections::hash_map::RawEntryMut, hash::Hashed};
#[cfg(feature = "bevy_reflect")]
use bevy_reflect::std_traits::ReflectDefault;
use bevy_utils::PreHashMap;
use indexmap::Equivalent;
use variadics_please::all_tuples;

/// A [`Template`] is something that, given a spawn context (target [`Entity`], [`World`], etc), can produce a [`Template::Output`].
///
/// [`Template`] is the cornerstone of scene systems. It enables define types (and hierarchies) that require no [`World`] or [`Entity`] context to define,
/// but can _use_ that context to produce the final runtime state. A [`Template`] is notably:
/// * **Repeatable**: Building a [`Template`] does not consume it. This enables reusing "baked" scenes / avoids rebuilding scenes each time we want to spawn one.
/// * **Clone-able**: Templates can be duplicated via [`Template::clone_template`], enabling scenes to be duplicated, supporting copy-on-write behaviors, etc.
/// * **(Often) Serializable**: Templates are intended to be easily serialized and deserialized, as they are typically composed of raw data.
///
/// Asset handles and [`Entity`] are two commonly [`Template`]-ed types. Asset handles are often "loaded" from an "asset path". The "asset path" would be the [`Template`].
/// Likewise [`Entity`] on its own has no reasonable default. A type with an [`Entity`] reference could use an "entity path" template to point to a specific entity, relative
/// to the current spawn context.
///
/// See [`FromTemplate`], which defines the canonical [`Template`] for a type. This can be derived, which will generate a [`Template`] for the deriving type.
pub trait Template {
    /// The type of value produced by this [`Template`].
    type Output;

    /// Uses this template and the given `entity` context to produce a [`Template::Output`].
    fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output>;

    /// Clones this template. See [`Clone`].
    fn clone_template(&self) -> Self;
}

/// The context used to apply the current [`Template`]. This contains a reference to the entity that the template is being
/// applied to (via an [`EntityWorldMut`]).
pub struct TemplateContext<'a, 'w> {
    /// The current entity the template is being applied to
    pub entity: &'a mut EntityWorldMut<'w>,
    /// A mapping of [`SceneEntityReference`] to [`Entity`] used for resolving `#Name` entity references
    pub entity_references: &'a mut SceneEntityReferences,
}

impl<'a, 'w> TemplateContext<'a, 'w> {
    /// Creates a new [`TemplateContext`].
    pub fn new(
        entity: &'a mut EntityWorldMut<'w>,
        entity_references: &'a mut SceneEntityReferences,
    ) -> Self {
        Self {
            entity,
            entity_references,
        }
    }
    /// Get the entity associated with the [`SceneEntityReference`], spawning a new one
    /// if this is the first call with this index.
    pub fn get_entity(&mut self, reference: SceneEntityReference) -> Entity {
        self.entity_references.get(
            reference,
            // Safety: only used to create a new Entity
            unsafe { self.entity.world_mut() },
        )
    }

    /// Retrieves a reference to the given resource `R`.
    #[inline]
    pub fn resource<R: Resource>(&self) -> &R {
        self.entity.resource()
    }

    /// Retrieves a mutable reference to the given resource `R`.
    #[inline]
    pub fn resource_mut<R: Resource<Mutability = Mutable>>(&mut self) -> Mut<'_, R> {
        self.entity.resource_mut()
    }

    /// Retrieves the entity associated with the given resource `R`, if it exists.
    #[inline]
    pub fn resource_entity<R: Resource>(&self) -> Option<Entity> {
        self.entity.resource_entity::<R>()
    }
}

/// Struct to store a mapping from [`SceneEntityReference`] to [`Entity`]
/// which are used for resolving `#Name` entity references in bsn! macros
#[derive(Default)]
pub struct SceneEntityReferences(PreHashMap<InnerSceneEntityReference, Entity>);

impl SceneEntityReferences {
    /// Get the [`Entity`] associated with this [`SceneEntityReference`]
    /// If the index is unknown, spawn a new empty [`Entity`] and store it
    pub fn get(&mut self, reference: SceneEntityReference, world: &mut World) -> Entity {
        let inner = reference.0;
        let entry = self
            .0
            .raw_entry_mut()
            .from_key_hashed_nocheck(inner.hash(), &inner);
        match entry {
            RawEntryMut::Occupied(entry) => *entry.get(),
            RawEntryMut::Vacant(view) => {
                let entity = world.spawn_empty().id();
                view.insert_hashed_nocheck(inner.hash(), inner, entity);
                entity
            }
        }
    }

    /// Iterates over every [`Entity`] this map resolved or materialized.
    ///
    /// This includes entities spawned by [`Self::get`] for forward references that no scene
    /// entity ever claimed — scene bookkeeping (e.g. hot reload) must account for those too, or
    /// they leak on every re-application.
    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.0.values().copied()
    }

    /// Set the [`Entity`] associated with a [`SceneEntityReference`]
    pub fn set(&mut self, reference: SceneEntityReference, entity: Entity) {
        let inner = reference.0;
        match self
            .0
            .raw_entry_mut()
            .from_key_hashed_nocheck(inner.hash(), &inner)
        {
            RawEntryMut::Occupied(_) => {}
            RawEntryMut::Vacant(view) => {
                view.insert_hashed_nocheck(inner.hash(), inner, entity);
            }
        };
    }
}

/// Identifies the "definition scope" that a [`SceneEntityReference`] belongs to. Two references
/// with different sources are never equal.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum SceneEntityReferenceSource {
    /// The reference was produced by a macro (such as `bsn!`) expanded at a source location.
    CallSite {
        /// The source file of the macro invocation.
        file: &'static str,
        /// The line of the macro invocation.
        line: usize,
        /// The column of the macro invocation.
        column: usize,
    },
    /// The reference was produced by a scene *asset* (such as a `.bsn` file).
    ///
    /// `path_hash` is a deterministic digest of the asset path
    /// ([`SceneEntityReference::asset_path_hash`]). The path string itself is intentionally not
    /// stored, so that [`SceneEntityReference`] stays [`Copy`], allocation-free, and small. Two
    /// distinct asset paths whose digests collide would alias their `#Name` references; at 64 bits
    /// this is ~1e-10 for six million distinct scene files, and the digest is never persisted.
    Asset {
        /// A deterministic digest of the asset path this reference came from.
        path_hash: u64,
    },
}

/// A unique reference for a named entity in a scene.
/// Usually used by `bevy_scene` in generated code
///
/// Hashed here should allow implementing compile-time hashing in the future
///
/// The uniqueness of this is ensured by the following factors:
/// - the [`SceneEntityReferenceSource`]: either a macro invocation location (filename, line and
///   column) or a digest of the asset path the scene was loaded from
/// - the `name_id` should uniquely identify a name in the individual macro's scope (for asset
///   sources it is the stable node id of the named entity inside the document)
/// - runtime, per-scope counter for each runtime call (usually from a static `AtomicU64`). Asset
///   sources always use `0`: an asset document is parsed and resolved once and its resolved scene
///   is shared, so a per-resolve counter would add no distinctness.
///
/// # Invariant
///
/// A [`SceneEntityReferences`] map must never be shared across two applications of a scene.
/// `SceneEntityReference`s produced from a scene *asset* (see
/// [`SceneEntityReference::from_asset`]) are identical for every spawn of that asset, so a shared
/// map would alias entities across spawns. `ResolvedSceneRoot::apply` and
/// `ResolvedSceneListRoot::spawn_with` in `bevy_scene` build a fresh map per call; keep it that
/// way.
///
/// # Limitations
///
/// If the *same* cached scene is included by two different entities inside one spawn tree, both
/// copies carry identical `SceneEntityReference` values, so [`SceneEntityReferences::set`]
/// (first-writer-wins) maps the second occurrence's `#Name` to the first occurrence's entity. This
/// is already true for macro-defined cached scenes (a scene patch asset is resolved once and its
/// resolved scene is shared); the asset variant behaves identically.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(
    feature = "bevy_reflect",
    reflect(opaque, Clone, PartialEq, Hash, Debug)
)]
pub struct SceneEntityReference(Hashed<InnerSceneEntityReference>);

/// The inner struct actually storing the unique index
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct InnerSceneEntityReference {
    source: SceneEntityReferenceSource,
    name_id: usize,
    runtime: u64,
}
impl SceneEntityReference {
    /// Create a new [`SceneEntityReference`] from the invocation location, runtime time, and a local (per-macro) counter for names
    pub fn new(
        (file, line, column): (&'static str, usize, usize),
        name_id: usize,
        runtime: u64,
    ) -> Self {
        Self::from_source(
            SceneEntityReferenceSource::CallSite { file, line, column },
            name_id,
            runtime,
        )
    }

    /// Create a [`SceneEntityReference`] for a named entity defined in a scene *asset*.
    ///
    /// * `asset_path` — the full path of the asset the scene was loaded from. Callers **must**
    ///   pass the fully-qualified asset path string (including asset source and label, e.g.
    ///   `"embedded://ui/menu.bsn#footer"`), so that identically-named files in different asset
    ///   sources stay distinct.
    /// * `node_id` — an identifier for the node inside that document that is **stable across
    ///   re-parses of an unchanged file** (e.g. a node index assigned in document order).
    ///
    /// The resulting reference is stable for a given `(asset_path, node_id)` pair, and distinct
    /// from every macro-produced reference.
    pub fn from_asset(asset_path: &str, node_id: u32) -> Self {
        Self::from_asset_hashed(Self::asset_path_hash(asset_path), node_id)
    }

    /// [`SceneEntityReference::from_asset`] for callers that already computed the digest once
    /// (e.g. once per loaded document rather than once per named entity).
    pub fn from_asset_hashed(path_hash: u64, node_id: u32) -> Self {
        Self::from_source(
            SceneEntityReferenceSource::Asset { path_hash },
            node_id as usize,
            0,
        )
    }

    /// The digest used by [`SceneEntityReference::from_asset`].
    ///
    /// Deterministic within and across processes for a given Bevy build, but an implementation
    /// detail: never persist it, never compare it across Bevy versions.
    pub fn asset_path_hash(asset_path: &str) -> u64 {
        bevy_platform::hash::fixed_hash_one(asset_path)
    }

    /// The definition scope this reference belongs to.
    pub fn source(&self) -> SceneEntityReferenceSource {
        self.0.source
    }

    fn from_source(source: SceneEntityReferenceSource, name_id: usize, runtime: u64) -> Self {
        Self(Hashed::new(InnerSceneEntityReference {
            source,
            name_id,
            runtime,
        }))
    }
}

impl core::fmt::Display for SceneEntityReference {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.source {
            SceneEntityReferenceSource::CallSite { file, line, column } => {
                f.write_fmt(format_args!(
                    "global={file}:{line}:{column} name_id={} runtime={:?}",
                    self.name_id, self.runtime
                ))
            }
            SceneEntityReferenceSource::Asset { path_hash } => f.write_fmt(format_args!(
                "asset=#{path_hash:016x} node_id={}",
                self.name_id
            )),
        }
    }
}

impl Deref for SceneEntityReference {
    type Target = Hashed<InnerSceneEntityReference>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Equivalent<Hashed<InnerSceneEntityReference>> for SceneEntityReference {
    #[inline]
    fn equivalent(&self, key: &Hashed<InnerSceneEntityReference>) -> bool {
        &self.0 == key
    }
}

/// [`FromTemplate`] is implemented for types that can be produced by a specific, canonical [`Template`]. This creates a way to correlate to the [`Template`] using the
/// desired template output type. This is used by Bevy's scene system.
///
/// Both [`FromTemplate`] and [`Template`] are blanket implemented for types that implement [`Default`] and [`Clone`], meaning most types you would want to use
/// _already have templates_.
///
/// It is best to think of [`FromTemplate`] as an alternative to [`Default`] for types that require world/spawn context to instantiate. Note that because of the blanket
/// impl, you cannot implement [`FromTemplate`], [`Default`], and [`Clone`] together on the same type, as it would result in two conflicting [`FromTemplate`] impls.
/// This is also why [`Template`] has its own [`Template::clone_template`] method (to avoid using the [`Clone`] impl, which would pull in the auto-impl).
///
/// You can _and should_ prefer deriving [`Default`] and [`Clone`] instead of an explicit [`FromTemplate`] impl, unless your type uses something that requires (or uses)
/// a [`Template`]. Handles in an asset system or [`Entity`] are examples of "templated" types. If you want your type to support templates of them, you probably want
/// to derive [`FromTemplate`].
///
/// [`FromTemplate`] can be derived for types whose fields _also_ implement [`FromTemplate`]:
/// ```
/// # use bevy_ecs::prelude::*;
/// # #[derive(Default, Clone)]
/// # struct Handle<T>(core::marker::PhantomData<T>);
/// # #[derive(Default, Clone)]
/// # struct Image;
/// #[derive(FromTemplate)]
/// struct Player {
///     image: Handle<Image>
/// }
/// ```
///
/// Deriving [`FromTemplate`] will generate a [`Template`] type for the deriving type. The example above would generate a `PlayerTemplate` like this:
/// ```
/// # use bevy_ecs::{prelude::*, template::TemplateContext};
/// # #[derive(FromTemplate)]
/// # struct Handle<T: core::marker::Unpin>(core::marker::PhantomData<T>);
/// # #[derive(Default, Clone)]
/// # struct Image;
/// struct Player {
///     image: Handle<Image>
/// }
///
/// impl FromTemplate for Player {
///     type Template = PlayerTemplate;
/// }
///
/// struct PlayerTemplate {
///     image: HandleTemplate<Image>,
/// }
///
/// impl Template for PlayerTemplate {
///     type Output = Player;
///     fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output> {
///         Ok(Player {
///             image: self.image.build_template(context)?,
///         })
///     }
///
///     fn clone_template(&self) -> Self {
///         PlayerTemplate {
///             image: self.image.clone_template(),
///         }
///     }
/// }
/// ```
///
/// [`FromTemplate`] derives can specify custom templates to use instead of a canonical [`FromTemplate`]:
/// ```
/// # use bevy_ecs::{prelude::*, template::TemplateContext};
/// # struct Image;
/// #[derive(FromTemplate)]
/// struct Counter {
///     #[template(Always10)]
///     count: usize
/// }
///
/// #[derive(Default)]
/// struct Always10;
///
/// impl Template for Always10 {
///     type Output = usize;
///
///     fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output> {
///         Ok(10)
///     }
///
///     fn clone_template(&self) -> Self {
///         Always10
///     }
/// }
/// ```
///
/// [`FromTemplate`] is automatically implemented for anything that is [`Default`] and [`Clone`]. "Built in" collection types like
/// [`Option`] and [`Vec`] pick up this "blanket" implementation, which is generally a good thing because it means these collection
/// types work with [`FromTemplate`] derives by default. However if the items in the collection have a custom [`FromTemplate`] impl
/// (ex: a manual implementation like `Handle<T>` for assets or an explicit [`FromTemplate`] derive), then relying on a [`Default`] /
/// [`Clone`] implementation doesn't work, as that won't run the template logic!
///
/// Therefore, cases like [`Option<Handle<T>>`] need something other than [`FromTemplate`] to determine the type. One option is to specify
/// the template manually:
///
/// ```
/// # use bevy_ecs::{prelude::*, template::{TemplateContext, OptionTemplate}};
/// # use core::marker::PhantomData;
/// # struct Handle<T>(PhantomData<T>);
/// # struct HandleTemplate<T>(PhantomData<T>);
/// # struct Image;
/// # impl<T> FromTemplate for Handle<T> {
/// #     type Template = HandleTemplate<T>;
/// # }
/// # impl<T> Template for HandleTemplate<T> {
/// #    type Output = Handle<T>;
/// #    fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output> {
/// #        unimplemented!()
/// #    }
/// #    fn clone_template(&self) -> Self {
/// #        unimplemented!()
/// #    }
/// # }
/// #[derive(FromTemplate)]
/// struct Widget {
///     #[template(OptionTemplate<HandleTemplate<Image>>)]
///     image: Option<Handle<Image>>
/// }
/// ```
///
/// However that is a bit of a mouthful! This is where [`BuiltInTemplate`] comes in. It fills the same role
/// as [`FromTemplate`], but has no blanket implementation for [`Default`] and [`Clone`], meaning we can have
/// custom implementations for types like [`Option`] and [`Vec`].
///
/// If you are deriving [`FromTemplate`] and you have a "built in" type like [`Option<Handle<T>>`] which has custom template logic,
/// annotate it with the `template(built_in)` attribute to use [`BuiltInTemplate`] instead of [`FromTemplate`]:
///
/// ```
/// # use bevy_ecs::{prelude::*, template::TemplateContext};
/// # use core::marker::PhantomData;
/// # struct Handle<T>(PhantomData<T>);
/// # struct HandleTemplate<T>(PhantomData<T>);
/// # struct Image;
/// # impl<T> FromTemplate for Handle<T> {
/// #     type Template = HandleTemplate<T>;
/// # }
/// # impl<T> Template for HandleTemplate<T> {
/// #    type Output = Handle<T>;
/// #    fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output> {
/// #        unimplemented!()
/// #    }
/// #    fn clone_template(&self) -> Self {
/// #        unimplemented!()
/// #    }
/// # }
/// #[derive(FromTemplate)]
/// struct Widget {
///     #[template(built_in)]
///     image: Option<Handle<Image>>
/// }
/// ```
/// ## Making the generated template reflectable
///
/// By default the generated template type derives nothing, so it is not
/// [`Reflect`](bevy_reflect::Reflect). Reflection-driven scene formats (e.g. dynamically loaded
/// `.bsn` assets) need to *construct and patch* the template from data alone, which requires it
/// to be reflectable. Opt in with the container-level `#[template(reflect)]` attribute:
///
/// ```
/// # use bevy_ecs::prelude::*;
/// # use bevy_reflect::prelude::*;
/// #[derive(Component, FromTemplate, Reflect)]
/// #[reflect(Component, FromTemplate)]
/// #[template(reflect)]
/// struct Score {
///     points: u32,
/// }
/// ```
///
/// This makes the generated `ScoreTemplate` derive
/// [`Reflect`](bevy_reflect::Reflect) and register
/// [`ReflectDefault`] and
/// [`ReflectTemplate`](crate::reflect::ReflectTemplate) type data (plus the
/// `ReflectFromReflect` the `Reflect` derive always adds).
///
/// Two requirements come with it:
///
/// * the deriving type itself must be [`Reflect`](bevy_reflect::Reflect), because
///   [`ReflectTemplate`](crate::reflect::ReflectTemplate) is only registerable for templates
///   whose [`Template::Output`] is reflectable;
/// * **every** field's template type must itself be [`Reflect`](bevy_reflect::Reflect) — for a
///   field whose template comes from another `#[derive(FromTemplate)]` type, that means the
///   other type needs `#[template(reflect)]` too.
///
/// Pair it with `#[reflect(FromTemplate)]` on the component so that consumers can find the
/// template type from the component type; see
/// [`ReflectFromTemplate`](crate::reflect::ReflectFromTemplate).
pub trait FromTemplate: Sized {
    /// The [`Template`] for this type.
    type Template: Template<Output = Self>;
}

macro_rules! template_impl {
    ($($template: ident),*) => {
        #[expect(
            clippy::allow_attributes,
            reason = "This is a tuple-related macro; as such, the lints below may not always apply."
        )]
        impl<$($template: Template),*> Template for TemplateTuple<($($template,)*)> {
            type Output = ($($template::Output,)*);
            fn build_template(&self, _context: &mut TemplateContext) -> Result<Self::Output> {
                #[allow(
                    non_snake_case,
                    reason = "The names of these variables are provided by the caller, not by us."
                )]
                let ($($template,)*) = &self.0;
                Ok(($($template.build_template(_context)?,)*))
            }

            fn clone_template(&self) -> Self {
                #[allow(
                    non_snake_case,
                    reason = "The names of these variables are provided by the caller, not by us."
                )]
                let ($($template,)*) = &self.0;
                TemplateTuple(($($template.clone_template(),)*))
            }
        }
    }
}

/// A wrapper over a tuple of [`Template`] implementations, which also implements [`Template`]. This exists because [`Template`] cannot
/// be directly implemented for tuples of [`Template`] implementations.
pub struct TemplateTuple<T>(pub T);

all_tuples!(template_impl, 0, 12, T);

// This includes `Unpin` to enable specialization for Templates that also implement Default, by using the
// ["auto trait specialization" trick](https://github.com/coolcatcoder/rust_techniques/issues/1)
impl<T: Clone + Unpin> Template for T {
    type Output = T;

    fn build_template(&self, _context: &mut TemplateContext) -> Result<Self::Output> {
        Ok(self.clone())
    }

    fn clone_template(&self) -> Self {
        self.clone()
    }
}

// This includes `Unpin` to enable specialization for Templates that also implement Default, by using the
// ["auto trait specialization" trick](https://github.com/coolcatcoder/rust_techniques/issues/1)
impl<T: Clone + Default + Unpin> FromTemplate for T {
    type Template = T;
}

/// This is used to help improve error messages related to [`FromTemplate`] specialization. Developers should generally just ignore
/// this trait and read the error message when they encounter it.
#[diagnostic::on_unimplemented(
    message = "This type does not manually implement FromTemplate, and it must. If you are deriving FromTemplate and you see this, it is likely because \
               a field does not have a FromTemplate impl. This can usually be fixed by using a custom template for that field. \
               Ex: for an Option<Handle<Image>> field, annotate the field with `#[template(OptionTemplate<HandleTemplate<Image>>)]`",
    note = "FromTemplate currently uses pseudo-specialization to enable FromTemplate to override Default. This error message is a consequence of t."
)]
pub trait SpecializeFromTemplate: Sized {}

/// A [`Template`] reference to an [`Entity`].
///
/// This is only valid during scene spawning and should **never** be used as a [`Component`](bevy_ecs::prelude::Component) field.
#[derive(Copy, Clone, Default, Debug)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[cfg_attr(
    feature = "bevy_reflect",
    reflect(Default, Clone, Debug, crate::reflect::Template)
)]
pub enum EntityTemplate {
    /// A reference to a specific [`Entity`]
    Entity(Entity),
    /// A reference to an entity via a unique reference
    SceneEntityReference(SceneEntityReference),
    /// An entity has not been specified. Building a template with this variant will result in an error.
    #[default]
    None,
}
impl Unpin for EntityTemplate where for<'a> [()]: SpecializeFromTemplate {}

impl EntityTemplate {
    /// Create a [`EntityTemplate::SceneEntityReference`] from the data needed for [`SceneEntityReference`]
    pub fn from_reference(
        invocation: (&'static str, usize, usize),
        name_id: usize,
        runtime: u64,
    ) -> Self {
        Self::SceneEntityReference(SceneEntityReference::new(invocation, name_id, runtime))
    }

    /// Create an [`EntityTemplate::SceneEntityReference`] pointing at a named entity defined in a
    /// scene asset. See [`SceneEntityReference::from_asset`].
    pub fn from_asset_reference(asset_path: &str, node_id: u32) -> Self {
        Self::SceneEntityReference(SceneEntityReference::from_asset(asset_path, node_id))
    }
}

impl From<Entity> for EntityTemplate {
    fn from(entity: Entity) -> Self {
        Self::Entity(entity)
    }
}

impl Template for EntityTemplate {
    type Output = Entity;

    fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output> {
        Ok(match self {
            Self::Entity(entity) => *entity,
            Self::SceneEntityReference(reference) => context.get_entity(*reference),
            Self::None => {
                return Err(BevyError::error(
                    "Failed to specify an entity for this EntityTemplate",
                ))
            }
        })
    }

    fn clone_template(&self) -> Self {
        match self {
            Self::Entity(entity) => Self::Entity(*entity),
            Self::SceneEntityReference(reference) => Self::SceneEntityReference(*reference),
            Self::None => Self::None,
        }
    }
}

impl FromTemplate for Entity {
    type Template = EntityTemplate;
}

/// A [`Template`] driven by a function that returns an output. This is used to create "free floating" templates without
/// defining a new type. See [`template`] for usage.
pub struct FnTemplate<F: Fn(&mut TemplateContext) -> Result<O>, O>(pub F);

impl<F: Fn(&mut TemplateContext) -> Result<O> + Clone, O> Template for FnTemplate<F, O> {
    type Output = O;

    fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output> {
        (self.0)(context)
    }

    fn clone_template(&self) -> Self {
        Self(self.0.clone())
    }
}

/// Returns a "free floating" template for a given `func`. This prevents the need to define a custom type for one-off templates.
pub fn template<F: Fn(&mut TemplateContext) -> Result<O>, O>(func: F) -> FnTemplate<F, O> {
    FnTemplate(func)
}

/// Roughly equivalent to [`FromTemplate`], but does not have a blanket implementation for [`Default`] + [`Clone`] types.
/// This is generally used for common generic collection types like [`Option`] and [`Vec`], which have [`Default`] + [`Clone`] impls and
/// therefore also pick up the [`FromTemplate`] behavior. This is fine when the `T` in [`Option<T>`] is not "templated"
/// (ex: does not have an explicit [`FromTemplate`] derive). But if `T` is "templated", such as [`Option<Handle<T>>`], then it would require
/// a manual `#[template(OptionTemplate<HandleTemplate<T>>)]` field annotation. This isn't fun to type out.
///
/// [`BuiltInTemplate`] enables equivalent "template type inference", by annotating a field with a type that implements [`BuiltInTemplate`] with
/// `#[template(built_in)]`.
pub trait BuiltInTemplate: Sized {
    /// The template to consider the "built in" template for this type.
    type Template: Template;
}

impl<T: FromTemplate> BuiltInTemplate for Option<T> {
    type Template = OptionTemplate<T::Template>;
}

impl<T: FromTemplate> BuiltInTemplate for Vec<T> {
    type Template = VecTemplate<T::Template>;
}

/// A [`Template`] for [`Option`].
#[derive(Default)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub enum OptionTemplate<T> {
    /// Template of [`Option::Some`].
    Some(T),
    /// Template of [`Option::None`].
    #[default]
    None,
}

impl<T> From<Option<T>> for OptionTemplate<T> {
    fn from(value: Option<T>) -> Self {
        match value {
            Some(value) => OptionTemplate::Some(value),
            None => OptionTemplate::None,
        }
    }
}

impl<T> From<T> for OptionTemplate<T> {
    fn from(value: T) -> Self {
        OptionTemplate::Some(value)
    }
}

impl<T: Template> Template for OptionTemplate<T> {
    type Output = Option<T::Output>;

    fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output> {
        Ok(match &self {
            OptionTemplate::Some(template) => Some(template.build_template(context)?),
            OptionTemplate::None => None,
        })
    }

    fn clone_template(&self) -> Self {
        match self {
            OptionTemplate::Some(value) => OptionTemplate::Some(value.clone_template()),
            OptionTemplate::None => OptionTemplate::None,
        }
    }
}

/// A [`Template`] for [`Vec`].
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub struct VecTemplate<T>(pub Vec<T>);

impl<T> Default for VecTemplate<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T: Template> Template for VecTemplate<T> {
    type Output = Vec<T::Output>;

    fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output> {
        let mut output = Vec::with_capacity(self.0.len());
        for value in &self.0 {
            output.push(value.build_template(context)?);
        }
        Ok(output)
    }

    fn clone_template(&self) -> Self {
        VecTemplate(self.0.iter().map(Template::clone_template).collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use alloc::string::{String, ToString};

    #[cfg(feature = "bevy_reflect")]
    mod reflect {
        use crate::{
            entity::Entity,
            prelude::*,
            reflect::{try_from_reflect_with_fallback, ReflectTemplate},
            template::{EntityTemplate, OptionTemplate, SceneEntityReference, VecTemplate},
        };
        use bevy_reflect::{
            std_traits::ReflectDefault, structs::DynamicStruct, PartialReflect, Reflect,
            ReflectFromReflect, ReflectRef, TypeRegistry,
        };
        use core::any::TypeId;

        #[derive(FromTemplate, Reflect, PartialEq, Debug)]
        #[template(reflect)]
        struct Foo {
            count: usize,
            other: usize,
        }

        /// A `#[derive(FromTemplate)]` without `#[template(reflect)]` still compiles, and its
        /// template simply is not `Reflect`.
        #[derive(FromTemplate)]
        struct NotReflected {
            #[expect(dead_code, reason = "only used to prove the derive still compiles")]
            count: usize,
        }

        #[test]
        fn template_reflect_attribute_generates_reflect_template() {
            let mut registry = TypeRegistry::empty();
            registry.register::<FooTemplate>();

            let data = registry
                .get_type_data::<ReflectTemplate>(TypeId::of::<FooTemplate>())
                .unwrap();
            assert_eq!(data.output_type_id, TypeId::of::<Foo>());
        }

        #[test]
        fn generated_template_has_from_reflect() {
            let mut registry = TypeRegistry::empty();
            registry.register::<FooTemplate>();

            assert!(registry
                .get_type_data::<ReflectFromReflect>(TypeId::of::<FooTemplate>())
                .is_some());
            assert!(registry
                .get_type_data::<ReflectDefault>(TypeId::of::<FooTemplate>())
                .is_some());
        }

        #[test]
        fn generated_template_round_trips_through_from_reflect() {
            let mut registry = TypeRegistry::empty();
            registry.register::<FooTemplate>();

            let mut patch = DynamicStruct::default();
            patch.insert("count", 3usize);

            let template =
                try_from_reflect_with_fallback::<FooTemplate>(&patch, &registry).unwrap();
            assert_eq!(template.count, 3);
            assert_eq!(template.other, 0);
        }

        #[test]
        fn template_reflect_attribute_is_opt_in() {
            // Compile-level assertion: `NotReflectedTemplate` exists and is usable without being
            // `Reflect`.
            let template = NotReflectedTemplate { count: 7 };
            assert_eq!(template.count, 7);
        }

        #[test]
        fn entity_template_is_reflect() {
            let mut registry = TypeRegistry::empty();
            registry.register::<EntityTemplate>();

            assert!(registry
                .get_type_data::<ReflectDefault>(TypeId::of::<EntityTemplate>())
                .is_some());
            let data = registry
                .get_type_data::<ReflectTemplate>(TypeId::of::<EntityTemplate>())
                .unwrap();
            assert_eq!(data.output_type_id, TypeId::of::<Entity>());
        }

        #[test]
        fn scene_entity_reference_is_opaque() {
            let reference = SceneEntityReference::new(("file.rs", 1, 2), 3, 4);
            assert!(matches!(reference.reflect_ref(), ReflectRef::Opaque(_)));
            let cloned = reference.reflect_clone().unwrap();
            assert_eq!(
                *cloned.downcast::<SceneEntityReference>().unwrap(),
                reference
            );
        }

        #[test]
        fn option_and_vec_templates_are_reflect() {
            let mut registry = TypeRegistry::empty();
            registry.register::<OptionTemplate<u32>>();
            registry.register::<VecTemplate<u32>>();

            assert!(registry.get(TypeId::of::<OptionTemplate<u32>>()).is_some());
            assert!(registry.get(TypeId::of::<VecTemplate<u32>>()).is_some());
            assert!(OptionTemplate::<u32>::None
                .get_represented_type_info()
                .is_some());
            assert!(VecTemplate::<u32>::default()
                .get_represented_type_info()
                .is_some());
        }
    }

    /// Tests for [`SceneEntityReference`]'s definition-scope identity.
    mod scene_entity_reference {
        use crate::{
            template::{
                EntityTemplate, SceneEntityReference, SceneEntityReferenceSource,
                SceneEntityReferences, Template,
            },
            world::World,
        };
        use alloc::format;

        // 15 — C4
        #[test]
        fn asset_scene_entity_reference_is_stable() {
            let a = SceneEntityReference::from_asset("a.bsn", 3);
            let b = SceneEntityReference::from_asset("a.bsn", 3);
            assert_eq!(a, b);
            assert_eq!(a.hash(), b.hash());
            assert_eq!(
                a.source(),
                SceneEntityReferenceSource::Asset {
                    path_hash: SceneEntityReference::asset_path_hash("a.bsn")
                }
            );
        }

        // 16 — C4
        #[test]
        fn asset_scene_entity_references_differ_by_path_and_node() {
            assert_ne!(
                SceneEntityReference::from_asset("a.bsn", 3),
                SceneEntityReference::from_asset("b.bsn", 3)
            );
            assert_ne!(
                SceneEntityReference::from_asset("a.bsn", 3),
                SceneEntityReference::from_asset("a.bsn", 4)
            );
        }

        // 17 — C4
        #[test]
        fn asset_and_call_site_references_never_collide() {
            assert_ne!(
                SceneEntityReference::from_asset("x", 0),
                SceneEntityReference::new(("x", 0, 0), 0, 0)
            );
        }

        // 18 — C4
        #[test]
        fn scene_entity_references_map_resolves_asset_references() {
            let mut world = World::new();
            let mut references = SceneEntityReferences::default();

            let first = references.get(SceneEntityReference::from_asset("a.bsn", 1), &mut world);
            let first_again =
                references.get(SceneEntityReference::from_asset("a.bsn", 1), &mut world);
            let second = references.get(SceneEntityReference::from_asset("a.bsn", 2), &mut world);

            assert_eq!(first, first_again);
            assert_ne!(first, second);
            assert!(world.get_entity(first).is_ok());
            assert!(world.get_entity(second).is_ok());
        }

        // 19 — C4 invariant: a reference map must never be shared across applies.
        #[test]
        fn fresh_reference_map_yields_fresh_entities() {
            let mut world = World::new();
            let reference = SceneEntityReference::from_asset("a.bsn", 1);

            let first = SceneEntityReferences::default().get(reference, &mut world);
            let second = SceneEntityReferences::default().get(reference, &mut world);

            assert_ne!(first, second);
        }

        // 20 — C4
        #[test]
        fn entity_template_from_asset_reference() {
            let template = EntityTemplate::from_asset_reference("a.bsn", 1);
            let expected = SceneEntityReference::from_asset("a.bsn", 1);

            let EntityTemplate::SceneEntityReference(reference) = template else {
                panic!("expected a SceneEntityReference variant, got {template:?}");
            };
            assert_eq!(reference, expected);

            let EntityTemplate::SceneEntityReference(cloned) =
                Template::clone_template(&EntityTemplate::from_asset_reference("a.bsn", 1))
            else {
                panic!("expected clone_template to preserve the variant");
            };
            assert_eq!(cloned, expected);
        }

        // 21 — C4 regression: call-site references are unchanged.
        #[test]
        fn call_site_scene_entity_reference_unchanged() {
            let reference = SceneEntityReference::new(("f.rs", 1, 2), 3, 4);
            assert_eq!(reference, SceneEntityReference::new(("f.rs", 1, 2), 3, 4));
            assert_ne!(reference, SceneEntityReference::new(("f.rs", 1, 2), 3, 5));
            assert_eq!(
                reference.source(),
                SceneEntityReferenceSource::CallSite {
                    file: "f.rs",
                    line: 1,
                    column: 2
                }
            );
            assert!(format!("{reference}").starts_with("global=f.rs:1:2"));
            assert!(
                format!("{}", SceneEntityReference::from_asset("a.bsn", 1)).starts_with("asset=#")
            );
        }
    }

    #[test]
    fn option_template() {
        #[derive(FromTemplate)]
        struct Handle(String);

        #[derive(FromTemplate)]
        struct Foo {
            #[template(built_in)]
            handle: Option<Handle>,
        }

        let mut world = World::new();
        let foo_template = FooTemplate {
            handle: Some(HandleTemplate("handle_path".to_string())).into(),
        };
        let foo = world.spawn_empty().build_template(&foo_template).unwrap();
        assert_eq!(foo.handle.unwrap().0, "handle_path".to_string());
    }
}
