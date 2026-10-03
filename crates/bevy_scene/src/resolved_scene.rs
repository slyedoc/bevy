use crate::{Ready, ResolveContext, ResolveSceneError, Scene, SceneList, ScenePatch};
use bevy_asset::{AssetId, AssetPath, AssetServer, Assets, Handle, UntypedAssetId};
use bevy_ecs::{
    bundle::{Bundle, BundleScratch, BundleWriter},
    component::{Component, ComponentsRegistrator},
    entity::Entity,
    error::{BevyError, Result},
    reflect::ReflectRelationshipTarget,
    relationship::{Relationship, RelationshipTarget},
    template::{
        FromTemplate, SceneEntityReference, SceneEntityReferences, Template, TemplateContext,
    },
    world::{EntityWorldMut, World},
};
use bevy_platform::collections::HashSet;
use bevy_reflect::{PartialReflect, ReflectFromPtr, ReflectFromReflect, TypeRegistry};
use bevy_utils::{TypeIdHashMap, TypeIdIndexMap};
use core::any::{Any, TypeId};
use thiserror::Error;
use tracing::error;

/// A final "spawnable" root [`ResolvedScene`].
pub struct ResolvedSceneRoot {
    /// The root [`ResolvedScene`].
    pub scene: ResolvedScene,
}

impl ResolvedSceneRoot {
    /// Resolves the current `scene` (using [`Scene::resolve`]). This should only be called after every dependency has loaded from the `scene`'s
    /// [`Scene::register_dependencies`].
    ///
    /// `type_registry` is required by reflection-driven [`Scene`] implementations and ignored by
    /// statically-typed ones. See [`ResolveContext::type_registry`].
    pub fn resolve(
        scene: Box<dyn Scene>,
        assets: &AssetServer,
        patches: &Assets<ScenePatch>,
        type_registry: Option<&TypeRegistry>,
    ) -> Result<Self, ResolveSceneError> {
        let mut resolved_scene = ResolvedScene::default();
        scene.resolve_box(
            &mut ResolveContext {
                assets,
                patches,
                cached: None,
                type_registry,
            },
            &mut resolved_scene,
        )?;
        Ok(ResolvedSceneRoot {
            scene: resolved_scene,
        })
    }

    /// This will spawn a new [`Entity`], then call [`ResolvedSceneRoot::apply`] on it.
    /// If this fails mid-spawn, the intermediate entity will be despawned.
    pub fn spawn<'w>(&self, world: &'w mut World) -> Result<EntityWorldMut<'w>, ApplySceneError> {
        let mut entity = world.spawn_empty();
        let result = self.apply(&mut entity, &mut BundleScratch::default());
        match result {
            Ok(_) => Ok(entity),
            Err(err) => {
                entity.despawn();
                Err(err)
            }
        }
    }

    /// Applies this scene to the given [`EntityWorldMut`].
    ///
    /// This will apply all of the [`Template`]s in this root [`ResolvedScene`] to the entity. It will also
    /// spawn all of this [`ResolvedScene`]'s related entities.
    ///
    /// If this root [`ResolvedScene`] includes a cached scene, that scene will be applied _first_.
    pub fn apply(
        &self,
        entity: &mut EntityWorldMut,
        bundle_scratch: &mut BundleScratch,
    ) -> Result<(), ApplySceneError> {
        self.apply_inner(entity, bundle_scratch, None)
    }

    /// Applies this scene to `entity` exactly like [`ResolvedSceneRoot::apply`], and additionally
    /// appends every [`Entity`] spawned during the application to `spawned`. `entity` itself is
    /// never appended, and `spawned` is sorted and deduplicated on return.
    ///
    /// This is what makes hot reload able to clean up after itself: the recorded entities are the
    /// scene's previous generation, and are despawned before the scene is re-applied. See
    /// [`SceneInstanceState`], which stores the result.
    ///
    /// Every entity the application creates is recorded: every related entity at any depth, and
    /// every entity materialized for a `#Name` reference — including a forward reference that is
    /// only ever used as a component value (`Reference(#Ghost)` with no `#Ghost` entity in the
    /// scene). The instance root itself is never recorded, even when a reference resolves to it.
    ///
    /// [`SceneInstanceState`]: crate::SceneInstanceState
    pub fn apply_recording(
        &self,
        entity: &mut EntityWorldMut,
        bundle_scratch: &mut BundleScratch,
        spawned: &mut Vec<Entity>,
    ) -> Result<(), ApplySceneError> {
        let root = entity.id();
        let result = self.apply_inner(entity, bundle_scratch, Some(&mut *spawned));
        // A related scene whose `#Name` resolves to the root would otherwise schedule the instance
        // entity itself for despawn on the next reload.
        spawned.retain(|spawned| *spawned != root);
        spawned.sort_unstable();
        spawned.dedup();
        result
    }

    fn apply_inner(
        &self,
        entity: &mut EntityWorldMut,
        bundle_scratch: &mut BundleScratch,
        mut recorder: Option<&mut Vec<Entity>>,
    ) -> Result<(), ApplySceneError> {
        // A *fresh* map per apply is load-bearing: see `SceneEntityReference`'s `# Invariant`
        // section. References produced from a scene asset are identical for every spawn of that
        // asset, so a shared map would alias entities across spawns.
        let mut entity_references = SceneEntityReferences::default();
        let mut context = TemplateContext::new(entity, &mut entity_references);

        let result = self
            .scene
            .apply(&mut context, bundle_scratch, recorder.as_deref_mut());

        // Union the reference map into the record: a forward `#Name` used only as a component
        // value materializes an entity through `SceneEntityReferences::get` that `apply_related`
        // never sees. Without this, every re-application leaks one ghost entity per such
        // reference. (Entities also recorded by `apply_related`, and a reference resolving to
        // the root, are handled by the caller's sort/dedup/retain.)
        if let Some(recorder) = recorder {
            recorder.extend(entity_references.iter());
        }
        if !bundle_scratch.is_empty() {
            // SAFETY: Components comes from the same world as the `context` passed in to self.scene.apply above
            unsafe {
                bundle_scratch.manual_drop(entity.world().components());
            }
        }
        result
    }
}

/// A final "spawnable" root list of [`ResolvedScene`]s.
pub struct ResolvedSceneListRoot {
    /// The root [`ResolvedScene`] list.
    pub scenes: Vec<ResolvedScene>,
}

impl ResolvedSceneListRoot {
    /// Resolves the current `scene_list` (using [`SceneList::resolve_list`]). This should only be
    /// called after every dependency has loaded from the `scene_list`'s [`SceneList::register_dependencies`].
    ///
    /// `type_registry` is required by reflection-driven [`Scene`] implementations and ignored by
    /// statically-typed ones. See [`ResolveContext::type_registry`].
    pub fn resolve(
        scene_list: Box<dyn SceneList>,
        assets: &AssetServer,
        patches: &Assets<ScenePatch>,
        type_registry: Option<&TypeRegistry>,
    ) -> Result<Self, ResolveSceneError> {
        let mut resolved_scenes = Vec::new();
        scene_list.resolve_list_box(
            &mut ResolveContext {
                assets,
                patches,
                cached: None,
                type_registry,
            },
            &mut resolved_scenes,
        )?;
        Ok(ResolvedSceneListRoot {
            scenes: resolved_scenes,
        })
    }
    /// Spawns a new [`Entity`] for each [`ResolvedScene`] in the list, and applies that [`ResolvedScene`] to them.
    pub fn spawn<'w>(&self, world: &'w mut World) -> Result<Vec<Entity>, ApplySceneError> {
        self.spawn_with(world, |_| {})
    }

    pub(crate) fn spawn_with(
        &self,
        world: &mut World,
        func: impl Fn(&mut EntityWorldMut),
    ) -> Result<Vec<Entity>, ApplySceneError> {
        let mut entities = Vec::new();
        // A *fresh* map per spawn is load-bearing: see `SceneEntityReference`'s `# Invariant`
        // section.
        let mut entity_references = SceneEntityReferences::default();
        let mut bundle_scratch = BundleScratch::default();
        for scene in self.scenes.iter() {
            let mut entity = if let Some(entity_index) = scene.entity_references.first().copied() {
                let entity = entity_references.get(entity_index, world);
                world.entity_mut(entity)
            } else {
                world.spawn_empty()
            };

            func(&mut entity);
            entities.push(entity.id());
            let result = scene.apply(
                &mut TemplateContext::new(&mut entity, &mut entity_references),
                &mut bundle_scratch,
                // Scene-list hot reload is out of scope: a scene list spawns N roots with no
                // owning instance entity to key re-application off.
                None,
            );
            if let Err(err) = result {
                // SAFETY: Components comes from the same world as the `context` passed in to self.scene.apply above
                unsafe {
                    bundle_scratch.manual_drop(entity.world().components());
                }
                return Err(err);
            }
        }

        Ok(entities)
    }
}

/// A final resolved scene (usually produced by calling [`Scene::resolve`]). This consists of:
/// 1. A collection of [`Template`]s to apply to a spawned [`Entity`], which are stored as [`ErasedTemplate`]s and [`ErasedBundleTemplate`]s.
/// 2. A collection of [`RelatedResolvedScenes`], which will be spawned as "related" entities (ex: [`Children`] entities).
/// 3. An optional cached [`ScenePatch`].
///
/// This uses "copy-on-write" behavior for cached scenes. If a [`Template`] is requested which the cached scene has as well,
/// it will be cloned (using [`Template::clone_template`]) and added to the current [`ResolvedScene`].
///
/// When applying this [`ResolvedScene`] to an [`Entity`], the cached scene (including its related scenes) is applied _first_. _Then_ this
/// [`ResolvedScene`] is applied.
///
/// [`Scene::resolve`]: crate::Scene::resolve
/// [`Children`]: bevy_ecs::hierarchy::Children
#[derive(Default)]
pub struct ResolvedScene {
    /// The collection of component [`Template`]s to apply to a spawned [`Entity`]. This can have multiple copies of the same [`Template`].
    component_templates: Vec<Box<dyn ErasedTemplate>>,
    /// The collection of Bundle templates to apply to a spawned [`Entity`].
    bundle_templates: Vec<Box<dyn ErasedBundleTemplate>>,
    /// The collection of [`RelatedResolvedScenes`], which will be spawned as "related" entities (ex: [`Children`] entities).
    ///
    /// [`Children`]: bevy_ecs::hierarchy::Children
    // PERF: special casing Children might make sense here to avoid hashing
    related: TypeIdIndexMap<RelatedResolvedScenes>,
    /// The cached [`ScenePatch`] to apply _first_ before applying this [`ResolvedScene`].
    cached: Option<CachedSceneInfo>,
    /// A [`TypeId`] to `templates` index mapping. If a [`Template`] is intended to be shared / patched across scenes, it should be registered
    /// here.
    template_indices: TypeIdHashMap<usize>,
    /// A list of all [`SceneEntityReference`] values associated with this entity. There can be more than one if this scene uses
    /// "flattened" caching.
    pub entity_references: Vec<SceneEntityReference>,
}

impl core::fmt::Debug for ResolvedScene {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ResolvedScene")
            .field("cached", &self.cached)
            .field("template_types", &self.template_indices.keys())
            .field("related", &self.related)
            .field("entity_references", &self.entity_references)
            .finish()
    }
}

impl ResolvedScene {
    /// Applies this scene to the given [`TemplateContext`] (which holds an already-spawned [`EntityWorldMut`]).
    ///
    /// This will apply all of the [`Template`]s in this [`ResolvedScene`] to the entity in the [`TemplateContext`]. It will also
    /// spawn all of this [`ResolvedScene`]'s related entities.
    ///
    /// If this [`ResolvedScene`] includes a cached scene, that scene will be applied _first_.
    fn apply(
        &self,
        context: &mut TemplateContext,
        bundle_scratch: &mut BundleScratch,
        recorder: Option<&mut Vec<Entity>>,
    ) -> Result<(), ApplySceneError> {
        self.apply_with(context, bundle_scratch, |_, _| {}, recorder)
    }

    /// Applies this scene to the given [`TemplateContext`] (which holds an already-spawned [`EntityWorldMut`]).
    ///
    /// This will apply all of the [`Template`]s in this [`ResolvedScene`] to the entity in the [`TemplateContext`]. It will also
    /// spawn all of this [`ResolvedScene`]'s related entities.
    ///
    /// If this [`ResolvedScene`] includes a cached scene, that scene will be applied _first_.
    ///
    /// This will call `writer_ops` right before calling [`BundleWriter::write`]. This will pass in the `context` value,
    /// which is the same context used to write all of the scene components to the [`BundleWriter`]. This ensures that
    /// writing to [`BundleWriter`] with the [`TemplateContext`] is safe (although those functions, if they are called, are still
    /// unsafe functions / the caller should verify they are using the passed in `context`).
    fn apply_with(
        &self,
        context: &mut TemplateContext,
        bundle_scratch: &mut BundleScratch,
        writer_ops: impl FnOnce(&mut TemplateContext, &mut BundleWriter),
        mut recorder: Option<&mut Vec<Entity>>,
    ) -> Result<(), ApplySceneError> {
        let mut bundle_writer = bundle_scratch.writer();
        for entity_reference in self.entity_references.iter().copied() {
            context
                .entity_references
                .set(entity_reference, context.entity.id());
        }
        if let Some(cached) = &self.cached {
            let scene_patches = context.resource::<Assets<ScenePatch>>();
            let Some(patch) = scene_patches.get(&cached.handle) else {
                return Err(ApplySceneError::MissingCachedScene {
                    path: cached.handle.path().cloned(),
                    id: cached.handle.id(),
                });
            };
            let Some(resolved_cached) = &patch.resolved else {
                return Err(ApplySceneError::UnresolvedCachedScene {
                    path: cached.handle.path().cloned(),
                    id: cached.handle.id(),
                });
            };
            let resolved_cached = resolved_cached.clone();
            // SAFETY: bundle_writer is used with the same World across all template.apply calls,
            // and the next bundle_writer.write call
            unsafe {
                resolved_cached
                    .scene
                    .apply_templates_without_bundle_write(
                        context,
                        &mut bundle_writer,
                        // Skip any template that has a local slot in the current scene:
                        // cached templates are copy-on-write, and locally-present types
                        // shadow the cached ones. Keying on the local slot map (rather than
                        // bookkeeping recorded at resolve time) also covers a dependent that
                        // resolved while its base was still unresolved — its local templates
                        // were created without base knowledge, and applying the base's copy
                        // too would push the same component twice into one bundle write.
                        &self.template_indices,
                    )
                    .map_err(|e| ApplySceneError::CachedSceneApplyError {
                        cached: cached.handle.path().cloned(),
                        error: Box::new(e),
                    })?;
                self.apply_templates_without_bundle_write(context, &mut bundle_writer, ())?;
                // SAFETY: World is only used for component registration, which does not affect
                // the entity location
                let components = &mut context.entity.world_mut().components_registrator();
                // This inserts empty RelationshipTarget collections to avoid archetype moves when then related entities are spawned
                // It pre-allocates space in the collection to avoid reallocs as related entities are added.
                for related in self.related.values() {
                    (related.insert_relationship_target)(
                        &mut bundle_writer,
                        components,
                        related.scenes.len(),
                    );
                }

                (writer_ops)(context, &mut bundle_writer);

                bundle_writer.write(context.entity);

                resolved_cached.scene.apply_related(
                    context,
                    bundle_scratch,
                    recorder.as_deref_mut(),
                )?;
                self.apply_related(context, bundle_scratch, recorder)?;
            }
        } else {
            // SAFETY: bundle_writer was used with the same World across all cases in this function,
            unsafe {
                self.apply_templates_without_bundle_write(context, &mut bundle_writer, ())?;
                // SAFETY: World is only used for component registration, which does not affect
                // the entity location
                let components = &mut context.entity.world_mut().components_registrator();
                // This inserts empty RelationshipTarget collections to avoid archetype moves when then related entities are spawned
                // It pre-allocates space in the collection to avoid reallocs as related entities are added.
                for related in self.related.values() {
                    (related.insert_relationship_target)(
                        &mut bundle_writer,
                        components,
                        related.scenes.len(),
                    );
                }
                (writer_ops)(context, &mut bundle_writer);
                bundle_writer.write(context.entity);
                self.apply_related(context, bundle_scratch, recorder)?;
            }
        };

        let entity = context.entity.id();
        context.entity.world_scope(|world| {
            world.trigger(Ready { entity });
        });

        Ok(())
    }

    /// # Safety
    ///
    /// `bundle_writer` must either be empty or only contain components registered with the given
    /// `context`'s World.
    unsafe fn apply_templates_without_bundle_write(
        &self,
        context: &mut TemplateContext,
        bundle_writer: &mut BundleWriter,
        skip_templates: impl SkipTemplate,
    ) -> Result<(), ApplySceneError> {
        for template in &self.component_templates {
            if skip_templates.should_skip(template.template_type_id()) {
                continue;
            }
            // SAFETY: bundle_writer is used with the same World across all template.apply calls,
            // and the next bundle_writer.write call
            unsafe {
                template
                    .apply(context, bundle_writer)
                    .map_err(ApplySceneError::TemplateBuildError)?;
            }
        }

        for template in &self.bundle_templates {
            template
                .apply(context)
                .map_err(ApplySceneError::TemplateBuildError)?;
        }
        Ok(())
    }

    fn apply_related(
        &self,
        context: &mut TemplateContext,
        bundle_scratch: &mut BundleScratch,
        mut recorder: Option<&mut Vec<Entity>>,
    ) -> Result<(), ApplySceneError> {
        for related_resolved_scenes in self.related.values() {
            let target = context.entity.id();
            let TemplateContext {
                entity,
                entity_references,
            } = context;
            let recorder = &mut recorder;
            entity.world_scope(|world| -> Result<(), ApplySceneError> {
                for (index, scene) in related_resolved_scenes.scenes.iter().enumerate() {
                    let mut entity =
                        if let Some(entity_reference) = scene.entity_references.first().copied() {
                            let entity = entity_references.get(entity_reference, world);
                            world.entity_mut(entity)
                        } else {
                            world.spawn_empty()
                        };

                    // A related entity is scene-owned however it was obtained: either it was
                    // spawned right here, or it was materialized earlier in this same apply by a
                    // forward `#Name` reference.
                    if let Some(spawned) = recorder.as_deref_mut() {
                        spawned.push(entity.id());
                    }

                    scene
                        .apply_with(
                            &mut TemplateContext::new(&mut entity, entity_references),
                            bundle_scratch,
                            |context, bundle_writer| {
                                // SAFETY: `context` is used to write all previous `bundle_writer` components
                                // and is also used to write this relationship component
                                unsafe {
                                    (related_resolved_scenes.insert_relationship)(
                                        bundle_writer,
                                        // SAFETY: World is only used for component registration, which does not affect
                                        // the entity location
                                        &mut context.entity.world_mut().components_registrator(),
                                        target,
                                    );
                                }
                            },
                            recorder.as_deref_mut(),
                        )
                        .map_err(|e| ApplySceneError::RelatedSceneError {
                            relationship_type_name: related_resolved_scenes.relationship_name,
                            index,
                            error: Box::new(e),
                        })?;
                }
                Ok(())
            })?;
        }

        Ok(())
    }

    /// This will get the [`Template`], if it already exists in this [`ResolvedScene`]. If it doesn't exist,
    /// it will use [`Default`] to create a new [`Template`].
    ///
    /// This uses "copy-on-write" behavior for cached scenes. If a [`Template`] is requested which the cached scene has as well,
    /// it will be cloned (using [`Template::clone_template`]), added to the current [`ResolvedScene`], and returned.
    ///
    /// This will ignore [`Template`]s added to this scene using [`ResolvedScene::push_template`], as these are not registered as the "canonical"
    /// [`Template`] for a given [`TypeId`].
    ///
    /// If the slot for `T` is occupied by a template of a _different_ concrete type — which happens
    /// when a cached scene resolved from a scene asset stored a reflection-driven template there —
    /// this converts that template back into a `T`, preserving its field values, using
    /// [`ReflectFromReflect`] from [`ResolveContext::type_registry`]. Fields the reflection system
    /// cannot see (`#[reflect(ignore)]`) come back as their [`Default`]. If no registry is
    /// available, or `T` cannot be produced from reflection, the slot is reset to `T::default()`
    /// and an error is logged: patched values are lost, but resolution never panics.
    pub fn get_or_insert_template<
        'a,
        T: Template<Output: SceneEffect> + Default + Send + Sync + 'static,
    >(
        &'a mut self,
        context: &mut ResolveContext,
    ) -> &'a mut T {
        let index = self.get_or_insert_erased_template_index(context, TypeId::of::<T>(), || {
            Box::new(T::default())
        });
        let slot = &mut self.component_templates[index];

        if !(&**slot as &dyn Any).is::<T>() {
            // Two statements: `*slot = Box::new(recover(&**slot, ..))` would hold a shared borrow
            // of `*slot` across the assignment and be rejected by the borrow checker.
            let recovered = recover_typed_template::<T>(&**slot, context.type_registry);
            *slot = Box::new(recovered);
        }

        // PERF: this could be unchecked, given that we control what is stored here
        // The method isn't stable yet, and it would require making get_or_insert_erased_template unsafe
        // Infallible: the branch above replaced any occupant that was not a `T`.
        (&mut **slot as &mut dyn Any).downcast_mut().unwrap()
    }

    /// Like [`Self::get_or_insert_template`], but it takes a [`FromTemplate`] type, which is used to look
    /// up the [`Template`] to add.
    pub fn get_or_insert_from_template<
        'a,
        T: FromTemplate<Template: Template<Output: SceneEffect> + Default + Send + Sync + 'static>,
    >(
        &'a mut self,
        context: &mut ResolveContext,
    ) -> &'a mut T::Template {
        self.get_or_insert_template::<T::Template>(context)
    }

    /// Inserts the given [`Template`]. This will overwrite the existing [`Template`] of that type if it already exists.
    pub fn insert_template<T: Template<Output: SceneEffect> + Send + Sync + 'static>(
        &mut self,
        template: T,
    ) {
        self.insert_erased_template(TypeId::of::<T>(), Box::new(template));
    }

    /// Inserts the given [`Template`] with the given `type_id`. This will overwrite the existing [`Template`] of that type if it already exists.
    ///
    /// For correctness, the stored template's [`ErasedTemplate::template_type_id`] must equal `type_id`.
    pub fn insert_erased_template(
        &mut self,
        type_id: TypeId,
        template: Box<dyn ErasedTemplate>,
    ) {
        match self.template_indices.entry(type_id) {
            bevy_utils::TypeIdHashMapEntry::Occupied(occupied_entry) => {
                let index = *occupied_entry.get();
                // SAFETY: just looked up a valid index
                let stored_template = unsafe { self.component_templates.get_unchecked_mut(index) };
                *stored_template = template;
            }
            bevy_utils::TypeIdHashMapEntry::Vacant(vacant_entry) => {
                vacant_entry.insert(self.component_templates.len());
                self.component_templates.push(template);
            }
        }
    }

    /// This will get the [`ErasedTemplate`] for the given [`TypeId`], if it already exists in this [`ResolvedScene`]. If it doesn't exist,
    /// it will use the `default` function to create a new [`ErasedTemplate`]. _For correctness, the [`TypeId`] of the [`Template`] returned
    /// by `default` should match the passed in `type_id`_. More precisely, the stored template's
    /// [`ErasedTemplate::template_type_id`] must equal `type_id`.
    ///
    /// `default` is only called when neither this [`ResolvedScene`] nor its cached scene already contains a template for `type_id`.
    ///
    /// This uses "copy-on-write" behavior for cached scenes. If a [`Template`] is requested which the cached scene has as well,
    /// it will be cloned (using [`Template::clone_template`]), added to the current [`ResolvedScene`], and returned.
    ///
    /// This will ignore [`Template`]s added to this scene using [`ResolvedScene::push_template`], as these are not registered as the "canonical"
    /// [`Template`] for a given [`TypeId`].
    pub fn get_or_insert_erased_template<'a>(
        &'a mut self,
        context: &mut ResolveContext,
        type_id: TypeId,
        default: impl FnOnce() -> Box<dyn ErasedTemplate>,
    ) -> &'a mut dyn ErasedTemplate {
        let index = self.get_or_insert_erased_template_index(context, type_id, default);
        // The index was just produced by the call above, so it is in bounds.
        &mut *self.component_templates[index]
    }

    /// The shared implementation of [`ResolvedScene::get_or_insert_erased_template`], returning the
    /// index into `component_templates` so that callers can re-borrow `self`.
    fn get_or_insert_erased_template_index(
        &mut self,
        context: &mut ResolveContext,
        type_id: TypeId,
        default: impl FnOnce() -> Box<dyn ErasedTemplate>,
    ) -> usize {
        *self.template_indices.entry(type_id).or_insert_with(|| {
            let index = self.component_templates.len();
            // Copy-on-write: seed the local slot from the cached scene's template when one
            // exists. Apply-time shadowing keys on `template_indices` itself, so no extra
            // bookkeeping is needed here.
            let value = if let Some(cached_patch) = &mut context.cached
                && let Some(resolved_cached) = &cached_patch.resolved
                && let Some(cached_template) =
                    resolved_cached.scene.get_direct_erased_template(type_id)
            {
                cached_template.clone_template()
            } else {
                default()
            };
            self.component_templates.push(value);
            index
        })
    }

    /// Returns the [`ErasedTemplate`] for the given `type_id`, if it exists in this [`ResolvedScene`]. This ignores cached scenes.
    pub fn get_direct_erased_template(&self, type_id: TypeId) -> Option<&dyn ErasedTemplate> {
        let index = self.template_indices.get(&type_id)?;
        Some(&*self.component_templates[*index])
    }

    /// Adds the `template` to the "back" of the [`ResolvedScene`] (it will applied later than earlier [`Template`]s).
    pub fn push_template<T: Template<Output: SceneEffect> + Send + Sync + 'static>(
        &mut self,
        template: T,
    ) {
        self.push_template_erased(Box::new(template));
    }

    /// Adds the `template` to the "back" of the [`ResolvedScene`] (it will applied later than earlier [`Template`]s).
    pub fn push_template_erased(&mut self, template: Box<dyn ErasedTemplate>) {
        self.component_templates.push(template);
    }

    /// Adds the `template` to the "back" of the [`ResolvedScene`] (it will applied later than earlier [`Template`]s).
    pub fn push_bundle_template<T: Template<Output: Bundle> + Send + Sync + 'static>(
        &mut self,
        template: T,
    ) {
        self.push_bundle_template_erased(Box::new(template));
    }

    /// Adds the `template` to the "back" of the [`ResolvedScene`] (it will applied later than earlier [`Template`]s).
    pub fn push_bundle_template_erased(&mut self, template: Box<dyn ErasedBundleTemplate>) {
        self.bundle_templates.push(template);
    }
    /// This will return the existing [`RelatedResolvedScenes`], if it exists. If not, a new empty [`RelatedResolvedScenes`] will be inserted and returned.
    ///
    /// This is used to add new related scenes and read existing related scenes.
    pub fn get_or_insert_related_resolved_scenes<R: Relationship>(
        &mut self,
    ) -> &mut RelatedResolvedScenes {
        self.related
            .entry(TypeId::of::<R>())
            .or_insert_with(RelatedResolvedScenes::new::<R>)
    }

    /// The type-erased counterpart of [`ResolvedScene::get_or_insert_related_resolved_scenes`],
    /// for callers that only have a [`ReflectRelationshipTarget`] (looked up from the
    /// [`TypeRegistry`] on a [`RelationshipTarget`] type such as [`Children`]).
    ///
    /// This uses the **same** keying as the generic method —
    /// [`ReflectRelationshipTarget::relationship_type_id`], i.e. `TypeId::of::<ChildOf>()` — so
    /// statically- and dynamically-defined children of the same relationship land in a single
    /// [`RelatedResolvedScenes`] and are spawned as one contiguous group, in the order the scenes
    /// were pushed.
    ///
    /// [`Children`]: bevy_ecs::hierarchy::Children
    pub fn get_or_insert_related_resolved_scenes_erased(
        &mut self,
        data: &ReflectRelationshipTarget,
    ) -> &mut RelatedResolvedScenes {
        self.related
            .entry(data.relationship_type_id)
            .or_insert_with(|| {
                RelatedResolvedScenes::new_erased(
                    data.insert_relationship,
                    data.insert_relationship_target,
                    data.relationship_name,
                )
            })
    }

    /// Configures this [`ResolvedScene`] to include the given [`ScenePatch`] cached.
    ///
    /// If this [`ResolvedScene`] already includes a cached scene, it will return [`CachedSceneError::MultipleCached`].
    /// If this [`ResolvedScene`] already has [`Template`]s or related scenes, it will return [`CachedSceneError::LateCached`].
    pub fn include_cached(&mut self, handle: Handle<ScenePatch>) -> Result<(), CachedSceneError> {
        if let Some(cached) = &self.cached {
            return Err(CachedSceneError::MultipleCached {
                id: cached.handle.id().untyped(),
                path: cached.handle.path().cloned(),
            });
        }
        if !(self.component_templates.is_empty() && self.related.is_empty()) {
            return Err(CachedSceneError::LateCached {
                id: handle.id().untyped(),
                path: handle.path().cloned(),
            });
        }
        self.cached = Some(CachedSceneInfo { handle });
        Ok(())
    }
}

/// Information about a [`ResolvedScene`]'s cached scene.
#[derive(Debug)]
pub(crate) struct CachedSceneInfo {
    /// The handle of the cached scene.
    pub(crate) handle: Handle<ScenePatch>,
}

/// The error returned by [`ResolvedScene::include_cached`].
#[derive(Error, Debug)]
pub enum CachedSceneError {
    /// Caused when attempting to include a second cached scene.
    #[error(
        "Attempted to include a second cached scene (id {id:?}, path: {path:?}), which is not allowed."
    )]
    MultipleCached {
        /// The asset id of the second cached scene.
        id: UntypedAssetId,
        /// The path of the second cached scene.
        path: Option<AssetPath<'static>>,
    },
    /// Caused when attempting to include a cached scene when a [`ResolvedScene`] already has [`Template`]s or related scenes.
    #[error("Attempted to include cached scene (id {id:?}, path: {path:?}), but the resolved scene already has templates. For correctness, the cached scene should always be included first.")]
    LateCached {
        /// The asset id of the cached scene that was included late.
        id: UntypedAssetId,
        /// The path of the cached scene that was included late.
        path: Option<AssetPath<'static>>,
    },
}

/// An error produced when applying a [`ResolvedScene`].
#[derive(Error, Debug)]
pub enum ApplySceneError {
    /// Caused when a [`Template`] fails to build
    #[error("Failed to build a Template in the current Scene: {0}")]
    TemplateBuildError(BevyError),
    /// Caused when the cached [`ResolvedScene`] fails to apply a [`ResolvedScene`].
    #[error("Failed to apply the cached Scene (asset path: \"{cached:?}\"): {error}")]
    CachedSceneApplyError {
        /// The asset path of the cached scene that failed to apply.
        cached: Option<AssetPath<'static>>,
        /// The error that occurred while applying the cached scene.
        error: Box<ApplySceneError>,
    },
    /// Caused when an cached scene is not present.
    #[error("The cached scene (id: {id:?}, path: \"{path:?}\") does not exist.")]
    MissingCachedScene {
        /// The path of the cached scene.
        path: Option<AssetPath<'static>>,
        /// The asset id of the cached scene.
        id: AssetId<ScenePatch>,
    },
    /// Caused when an cached scene has not been resolved yet.
    #[error("The cached scene (id: {id:?}, path: \"{path:?}\") has not been resolved yet.")]
    UnresolvedCachedScene {
        /// The path of the cached scene.
        path: Option<AssetPath<'static>>,
        /// The asset id of the cached scene.
        id: AssetId<ScenePatch>,
    },
    /// Caused when a related [`ResolvedScene`] fails to apply.
    #[error(
        "Failed to apply the related {relationship_type_name} Scene at index {index}: {error}"
    )]
    RelatedSceneError {
        /// The type name of the relationship.
        relationship_type_name: &'static str,
        /// The index of the related scene that failed to apply.
        index: usize,
        /// The error that occurred when applying the related scene.
        error: Box<ApplySceneError>,
    },
}

/// A collection of [`ResolvedScene`]s that are related to a given [`ResolvedScene`] by a [`Relationship`].
/// Each [`ResolvedScene`] added here will be spawned as a new [`Entity`] when the "parent" [`ResolvedScene`] is spawned.
pub struct RelatedResolvedScenes {
    /// The related resolved scenes. Each entry in the list corresponds to a new related entity that will be spawned with the given scene.
    pub scenes: Vec<ResolvedScene>,
    /// The function that will be called to add the relationship to the spawned related scene.
    pub insert_relationship:
        unsafe fn(&mut BundleWriter, &mut ComponentsRegistrator, target: Entity),
    /// The function that will be called to add the relationship target to the spawned scene with the given capacity.
    pub insert_relationship_target: unsafe fn(&mut BundleWriter, &mut ComponentsRegistrator, usize),
    /// The type name of the relationship. This is used for more helpful error message.
    pub relationship_name: &'static str,
}

impl core::fmt::Debug for RelatedResolvedScenes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ResolvedRelatedScenes")
            .field("scenes", &self.scenes)
            .finish()
    }
}

impl RelatedResolvedScenes {
    /// Creates a new empty [`RelatedResolvedScenes`] for the given relationship type.
    pub fn new<R: Relationship>() -> Self {
        Self::new_erased(
            |bundle_writer, components_registrator, target| {
                // SAFETY: caller ensures bundler_writer is always used with the same World
                unsafe { bundle_writer.push_component(components_registrator, R::from(target)) };
            },
            |bundle_writer, components_registrator, capacity| {
                let relationship_target =
                    <<R as Relationship>::RelationshipTarget as RelationshipTarget>::with_capacity(
                        capacity,
                    );
                // SAFETY: caller ensures bundler_writer is always used with the same World
                unsafe {
                    bundle_writer.push_component(components_registrator, relationship_target);
                };
            },
            core::any::type_name::<R>(),
        )
    }

    /// Creates a new empty [`RelatedResolvedScenes`] from already type-erased relationship
    /// functions, for callers that only have runtime type information.
    ///
    /// The three arguments have exactly the semantics of the same-named fields; the canonical
    /// source of a matching set is [`ReflectRelationshipTarget`], whose function pointers have the
    /// same bodies as [`RelatedResolvedScenes::new`]'s.
    ///
    /// The stored function pointers are `unsafe fn` whose contracts are discharged at *call* time
    /// by [`ResolvedScene`]'s apply path, so this constructor itself is safe. They must nonetheless
    /// be a matching pair for a single [`Relationship`] type, or spawning will produce entities
    /// with mismatched relationship / relationship-target components.
    pub fn new_erased(
        insert_relationship: unsafe fn(&mut BundleWriter, &mut ComponentsRegistrator, Entity),
        insert_relationship_target: unsafe fn(&mut BundleWriter, &mut ComponentsRegistrator, usize),
        relationship_name: &'static str,
    ) -> Self {
        Self {
            scenes: Vec::new(),
            insert_relationship,
            insert_relationship_target,
            relationship_name,
        }
    }
}

/// A type-erased, object-safe, downcastable version of [`Template`] that produces a [`Component`], which will be added to the
/// given [`BundleWriter`].
pub trait ErasedTemplate: Any + Send + Sync {
    /// Applies this template to the given `entity`.
    ///
    /// # Safety
    ///
    /// `bundle_writer` must always be used with the same World that is stored in `context`. This
    /// is intended to be used by a scene system in a scoped / controlled / easily verifiable context.
    /// If you are calling it outside of that context, you are almost certainly doing something wrong!
    unsafe fn apply(
        &self,
        context: &mut TemplateContext,
        bundle_writer: &mut BundleWriter,
    ) -> Result<(), BevyError>;

    /// Clones this template. See [`Clone`].
    fn clone_template(&self) -> Box<dyn ErasedTemplate>;

    /// Returns a [`PartialReflect`] view of the value this template will build from, if this
    /// template stores its data in a type-erased, reflected form.
    ///
    /// This returns `None` for ordinary statically-typed templates (everything covered by the
    /// blanket `impl<T: Template<Output: Component>> ErasedTemplate for T`), because the
    /// blanket impl cannot know whether `T` implements [`PartialReflect`]. Use
    /// [`erased_template_as_partial_reflect`] instead, which additionally recovers a reflected
    /// view of statically-typed templates through the [`TypeRegistry`].
    fn try_as_partial_reflect(&self) -> Option<&dyn PartialReflect> {
        None
    }

    /// The mutable counterpart of [`ErasedTemplate::try_as_partial_reflect`]. This is
    /// how a runtime-constructed patch (e.g. one produced by a scene asset loader) writes fields
    /// into a template it did not create.
    ///
    /// See [`erased_template_as_partial_reflect_mut`].
    fn try_as_partial_reflect_mut(&mut self) -> Option<&mut dyn PartialReflect> {
        None
    }

    /// The [`TypeId`] of the [`Template`] this erased template stands in for — the key under which
    /// it was filed by [`ResolvedScene::insert_erased_template`] /
    /// [`ResolvedScene::get_or_insert_erased_template`].
    ///
    /// This defaults to the implementor's own concrete Rust type, which is correct for every
    /// statically-typed template (the blanket impl's `Self` _is_ the template type). Templates
    /// that store their value type-erased — such as a template built by a scene asset loader —
    /// **must** override this to return the template type they represent, or the cached
    /// copy-on-write duplicate check will fail to skip them and the same component will be
    /// written twice.
    fn template_type_id(&self) -> TypeId {
        Any::type_id(self)
    }
}

/// Returns a [`PartialReflect`] view of an [`ErasedTemplate`], if one can be obtained.
///
/// This first asks the template itself ([`ErasedTemplate::try_as_partial_reflect`]),
/// which is how templates that store an erased `Box<dyn Reflect>` answer. If the template does not
/// answer, and the template's *concrete* Rust type is a registered [`Reflect`] type, the value is
/// viewed through [`ReflectFromPtr`]. This second path is what makes a statically-defined `bsn!`
/// template (for example the canonical template of a `#[derive(Component, Reflect, Clone, Default)]`
/// component, which is the component type itself) patchable by a runtime-constructed scene.
///
/// Returns `None` if neither path applies — typically a template type generated by
/// `#[derive(FromTemplate)]` without `#[template(reflect)]`, or a [`Reflect`] type that was never
/// registered.
///
/// [`Reflect`]: bevy_reflect::Reflect
pub fn erased_template_as_partial_reflect<'a>(
    template: &'a dyn ErasedTemplate,
    type_registry: &TypeRegistry,
) -> Option<&'a dyn PartialReflect> {
    // See the `_mut` version for why the call is duplicated.
    if template.try_as_partial_reflect().is_some() {
        return template.try_as_partial_reflect();
    }

    let any: &dyn Any = template;
    let from_ptr = type_registry.get_type_data::<ReflectFromPtr>(any.type_id())?;
    Some(from_ptr.as_reflect(any)?.as_partial_reflect())
}

/// The mutable counterpart of [`erased_template_as_partial_reflect`].
pub fn erased_template_as_partial_reflect_mut<'a>(
    template: &'a mut dyn ErasedTemplate,
    type_registry: &TypeRegistry,
) -> Option<&'a mut dyn PartialReflect> {
    // Templates that store their value type-erased know their own reflected view, which is not
    // their concrete Rust type. Ask them first.
    //
    // NOTE: the call is deliberately duplicated: NLL cannot see that the borrow taken by the `if`
    // condition ends before the `return`. Both trait impls are trivial accessors.
    if template.try_as_partial_reflect_mut().is_some() {
        return template.try_as_partial_reflect_mut();
    }

    let any: &mut dyn Any = template;
    let from_ptr = type_registry.get_type_data::<ReflectFromPtr>((*any).type_id())?;
    Some(from_ptr.as_reflect_mut(any)?.as_partial_reflect_mut())
}

/// Produces a `T` from a template that occupies `T`'s slot but is not a `T`.
///
/// This happens when a cached scene (typically resolved from a `.bsn` asset) stored a
/// reflection-driven template under `T`'s [`TypeId`], and a statically-typed patch then asks for
/// `&mut T`. Field values from `occupant` are preserved when both a reflected view of it and
/// [`ReflectFromReflect`] for `T` are available; otherwise `T::default()` is returned and an error
/// is logged. A `.bsn` file is user data, so this must never panic.
fn recover_typed_template<T: Default + Send + Sync + 'static>(
    occupant: &dyn ErasedTemplate,
    type_registry: Option<&TypeRegistry>,
) -> T {
    let recovered = type_registry.and_then(|type_registry| {
        let reflect = erased_template_as_partial_reflect(occupant, type_registry)?;
        let from_reflect = type_registry.get_type_data::<ReflectFromReflect>(TypeId::of::<T>())?;
        from_reflect.from_reflect(reflect)?.downcast::<T>().ok()
    });

    match recovered {
        Some(value) => *value,
        None => {
            let hint = if type_registry.is_some() {
                "It must derive `Reflect` (which registers `ReflectFromReflect`) and be registered \
                 in the TypeRegistry."
            } else {
                "No TypeRegistry was available to convert it: resolve this scene through \
                 `World::spawn_scene`, `EntityWorldMut::apply_scene` or the \
                 `resolve_scene_patches` system, or pass a `TypeRegistry` to \
                 `ResolvedSceneRoot::resolve`."
            };
            error!(
                "The template slot for `{0}` holds a template of a different type that could not \
                 be converted back to `{0}`, so its values were reset to `Default`. {hint}",
                core::any::type_name::<T>()
            );
            T::default()
        }
    }
}

impl<T: Template<Output: SceneEffect> + Send + Sync + 'static> ErasedTemplate for T {
    unsafe fn apply(
        &self,
        context: &mut TemplateContext,
        bundle_writer: &mut BundleWriter,
    ) -> Result<(), BevyError> {
        let output = self.build_template(context)?;
        output.apply(context, bundle_writer);
        Ok(())
    }

    fn clone_template(&self) -> Box<dyn ErasedTemplate> {
        Box::new(Template::clone_template(self))
    }
}

/// Something that has an effect on the final applied scene. This is usually a [`Component`].
pub trait SceneEffect {
    /// Applies the scene effect to the current context.
    fn apply(self, context: &mut TemplateContext, bundle_writer: &mut BundleWriter);
}

/// A scene effect that does nothing.
pub struct EmptySceneEffect;

impl SceneEffect for EmptySceneEffect {
    #[inline]
    fn apply(self, _context: &mut TemplateContext, _bundle_writer: &mut BundleWriter) {}
}

impl<C: Component> SceneEffect for C {
    fn apply(self, context: &mut TemplateContext, bundle_writer: &mut BundleWriter) {
        // SAFETY: world_mut is only used to register components, which does not affect entity location
        let mut components = unsafe { context.entity.world_mut().components_registrator() };
        // SAFETY: The caller verifies that `bundle_writer` is always used with the same World.
        unsafe { bundle_writer.push_component(&mut components, self) };
    }
}

/// A type-erased, object-safe, downcastable version of [`Template`] that produces a [`Bundle`], which will be added
/// immediately to a given `entity`.
pub trait ErasedBundleTemplate: Any + Send + Sync {
    /// Applies this template to the given `entity`.
    fn apply(&self, context: &mut TemplateContext) -> Result<(), BevyError>;

    /// Clones this template. See [`Clone`].
    fn clone_template(&self) -> Box<dyn ErasedBundleTemplate>;
}

impl<T: Template<Output: Bundle> + Send + Sync + 'static> ErasedBundleTemplate for T {
    fn apply(&self, context: &mut TemplateContext) -> Result<(), BevyError> {
        let bundle = self.build_template(context)?;
        context.entity.insert(bundle);
        Ok(())
    }

    fn clone_template(&self) -> Box<dyn ErasedBundleTemplate> {
        Box::new(Template::clone_template(self))
    }
}

/// A filter to skip the template for a given `TypeId`
trait SkipTemplate {
    /// Returns true if the template with `type_id` should be skipped.
    fn should_skip(&self, type_id: TypeId) -> bool;
}

impl SkipTemplate for &TypeIdHashMap<usize> {
    #[inline]
    fn should_skip(&self, type_id: TypeId) -> bool {
        self.contains_key(&type_id)
    }
}

impl SkipTemplate for &HashSet<TypeId> {
    #[inline]
    fn should_skip(&self, type_id: TypeId) -> bool {
        self.contains(&type_id)
    }
}

impl SkipTemplate for () {
    #[inline]
    fn should_skip(&self, _type_id: TypeId) -> bool {
        false
    }
}

/// Focused tests for this module's `unsafe` surface, designed to run under Miri.
///
/// They deliberately use only a bare [`World`] — no assets, no task pools, no app — so
/// that `cargo miri test -p bevy_scene --no-default-features unsafe_paths` stays fast.
/// CI runs exactly that filter (see the `miri` job); everything here also runs as a
/// normal test on every platform.
#[cfg(test)]
mod unsafe_paths {
    use super::*;
    use alloc::{boxed::Box, vec::Vec};
    use bevy_ecs::{
        component::Component,
        hierarchy::{ChildOf, Children},
        name::Name,
        prelude::ReflectComponent,
        reflect::ReflectRelationshipTarget,
    };
    use bevy_reflect::{prelude::ReflectDefault, Reflect};

    #[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
    #[reflect(Component, Default)]
    struct Health(u32);

    fn registry_with<T: bevy_reflect::GetTypeRegistration>() -> TypeRegistry {
        let mut registry = TypeRegistry::empty();
        registry.register::<T>();
        registry
    }

    /// Shared and mutable `ReflectFromPtr` views of a typed template: the fat→thin
    /// pointer casts in `erased_template_as_partial_reflect{,_mut}` are exactly what
    /// Miri's provenance tracking is for.
    #[test]
    fn reflect_views_of_typed_template_are_sound() {
        let registry = registry_with::<Health>();
        let mut erased: Box<dyn ErasedTemplate> = Box::new(Health(7));

        let shared = erased_template_as_partial_reflect(&*erased, &registry)
            .expect("registered template must be viewable");
        assert_eq!(shared.try_downcast_ref::<Health>(), Some(&Health(7)));

        let exclusive = erased_template_as_partial_reflect_mut(&mut *erased, &registry)
            .expect("registered template must be viewable mutably");
        *exclusive
            .try_downcast_mut::<Health>()
            .expect("concrete type") = Health(9);

        let reread = erased_template_as_partial_reflect(&*erased, &registry).unwrap();
        assert_eq!(reread.try_downcast_ref::<Health>(), Some(&Health(9)));
    }

    /// An unregistered template type must yield `None`, never a wild cast.
    #[test]
    fn reflect_view_of_unregistered_template_is_none() {
        let registry = TypeRegistry::empty();
        let erased: Box<dyn ErasedTemplate> = Box::new(Health(1));
        assert!(erased_template_as_partial_reflect(&*erased, &registry).is_none());
    }

    /// The erased relationship fn pointers write through a `BundleWriter` into a bare
    /// `World`: exercises `push_component` + the recursive write path under Miri.
    #[test]
    fn erased_relationship_insertion_is_sound() {
        let mut registry = TypeRegistry::empty();
        registry.register::<Children>();
        let data = registry
            .get_type_data::<ReflectRelationshipTarget>(TypeId::of::<Children>())
            .unwrap()
            .clone();

        let mut world = World::new();
        let parent = world.spawn_empty().id();
        let mut scratch = BundleScratch::default();
        let mut writer = scratch.writer();
        // SAFETY: `writer` and `components` come from the same `World`, and `writer`
        // is written to `child` below, an entity of that same `World`.
        let child = unsafe {
            let mut components = world.components_registrator();
            (data.insert_relationship)(&mut writer, &mut components, parent);
            let mut child = world.spawn_empty();
            writer.write(&mut child);
            child.id()
        };
        assert_eq!(world.get::<ChildOf>(child).unwrap().parent(), parent);
    }

    /// A full erased apply through `ResolvedScene`: template + related child, applied
    /// to a bare world, exercising `BundleScratch::manual_drop` bookkeeping too.
    #[test]
    fn resolved_scene_apply_is_sound_on_bare_world() {
        let mut registry = TypeRegistry::empty();
        registry.register::<Health>();
        registry.register::<Children>();
        registry.register::<Name>();

        let mut world = World::new();
        let mut scene = ResolvedScene::default();
        // `Health` is `Clone + Default`, so it is its own `Template` and picks up the
        // blanket `ErasedTemplate` impl directly.
        scene.push_template_erased(Box::new(Health(3)));

        let root = ResolvedSceneRoot { scene };
        let mut spawned: Vec<Entity> = Vec::new();
        let mut entity = world.spawn_empty();
        root.apply_recording(&mut entity, &mut BundleScratch::default(), &mut spawned)
            .unwrap();
        let id = entity.id();
        assert_eq!(world.get::<Health>(id), Some(&Health(3)));
        assert!(spawned.is_empty());
    }
}
