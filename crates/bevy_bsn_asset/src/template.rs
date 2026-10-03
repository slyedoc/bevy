//! [`DynamicComponentTemplate`]: the [`ErasedTemplate`] behind every `.bsn` component
//! patch.

use core::any::TypeId;

use bevy_ecs::{
    bundle::BundleWriter,
    error::BevyError,
    reflect::{AppTypeRegistry, ReflectComponent, ReflectTemplate},
    template::TemplateContext,
};
use bevy_reflect::{std_traits::ReflectDefault, PartialReflect, Reflect, ReflectFromReflect};

use bevy_scene::ErasedTemplate;

/// An [`ErasedTemplate`] whose template value is held reflectively.
///
/// This is what a `.bsn` document's `Type { … }` entry becomes in a
/// [`ResolvedScene`](bevy_scene::ResolvedScene). It stores a **concrete** value of the component's
/// template type (never a `Dynamic*` value), plus the type data needed to build and insert the
/// component without touching the [`TypeRegistry`](bevy_reflect::TypeRegistry) again.
///
/// # This type must not implement `Clone`
///
/// [`ErasedTemplate`] is blanket-implemented for every `T: Template<Output: Component>`,
/// and [`Template`](bevy_ecs::template::Template) is itself blanket-implemented for
/// `T: Clone + Unpin`. A `Clone` (and therefore `Unpin`) `DynamicComponentTemplate` would pick up
/// both blankets and collide with the manual implementation below — a coherence error. It cannot
/// derive `Clone` anyway, since `Box<dyn Reflect>` is not `Clone`. Duplication goes through
/// [`ErasedTemplate::clone_template`] and nothing else.
pub struct DynamicComponentTemplate {
    /// The slot key: the [`TypeId`] of the concrete value held in `value`.
    template_type_id: TypeId,
    /// The template value. Always concrete, constructed by [`ReflectDefault`] and mutated by
    /// `try_apply`.
    value: Box<dyn Reflect>,
    /// From the *output* type's registration; provides `push_to_bundle_writer`.
    reflect_component: ReflectComponent,
    /// `Some` iff the template's output type differs from the template type.
    reflect_template: Option<ReflectTemplate>,
    /// Fallbacks for [`ErasedTemplate::clone_template`], which has no error channel.
    reflect_default: ReflectDefault,
    reflect_from_reflect: Option<ReflectFromReflect>,
    /// A live registry handle.
    ///
    /// This is *not* made redundant by `ResolveContext::type_registry`: that only exists during
    /// [`Scene::resolve`](bevy_scene::Scene::resolve), whereas
    /// [`ErasedTemplate::apply`] runs later and receives a [`TemplateContext`], which has
    /// no registry — yet `push_to_bundle_writer` needs one, and reading it out of the context's
    /// world would hold a borrow of the entity across the `world_mut()` call in `apply`.
    type_registry: AppTypeRegistry,
}

impl DynamicComponentTemplate {
    /// Creates a template around an already-constructed, concrete template `value`.
    ///
    /// `template_type_id` must be the [`TypeId`] of `value`'s concrete type; `reflect_component`
    /// must come from the registration of the type the template *builds*, and the remaining type
    /// data from the registration of the template type itself.
    pub(crate) fn new(
        template_type_id: TypeId,
        value: Box<dyn Reflect>,
        reflect_component: ReflectComponent,
        reflect_template: Option<ReflectTemplate>,
        reflect_default: ReflectDefault,
        reflect_from_reflect: Option<ReflectFromReflect>,
        type_registry: AppTypeRegistry,
    ) -> Self {
        debug_assert_eq!(
            (*value).type_id(),
            template_type_id,
            "a DynamicComponentTemplate's value must be of its template type"
        );
        Self {
            template_type_id,
            value,
            reflect_component,
            reflect_template,
            reflect_default,
            reflect_from_reflect,
            type_registry,
        }
    }
}

impl ErasedTemplate for DynamicComponentTemplate {
    unsafe fn apply(
        &self,
        context: &mut TemplateContext,
        bundle_writer: &mut BundleWriter,
    ) -> Result<(), BevyError> {
        // 1. Build the output *before* acquiring the registry guard: a user `build_template` may
        //    lock the registry itself, and a re-entrant read can deadlock behind a queued writer.
        let output: Box<dyn Reflect> = match &self.reflect_template {
            Some(reflect_template) => (reflect_template.build_template)(&*self.value, context)?,
            // No `ReflectTemplate` means the output type *is* the template type. A bare
            // `reflect_clone` is not enough here: it refuses `#[reflect(ignore)]` fields and
            // opaque fields without a registered clone, all of which spawn fine through the
            // static `bsn!` path (which uses real `Clone`). Fall back to `ReflectFromReflect`,
            // which reads the reflected fields and defaults the ignored ones.
            None => match self.value.reflect_clone() {
                Ok(value) => value,
                Err(clone_error) => self
                    .reflect_from_reflect
                    .as_ref()
                    .and_then(|from_reflect| {
                        from_reflect.from_reflect(self.value.as_partial_reflect())
                    })
                    .ok_or(clone_error)?,
            },
        };

        // 2. The guard comes from our own handle, so it is independent of any borrow of `context`.
        let type_registry = self.type_registry.read();

        // SAFETY: `world_mut` is only used to register components, which does not move the entity.
        let mut components = unsafe { context.entity.world_mut().components_registrator() };

        // SAFETY: `bundle_writer` and `components` come from the same `World`, per this method's
        // own safety contract.
        unsafe {
            self.reflect_component.push_to_bundle_writer(
                output.into_partial_reflect(),
                &type_registry,
                &mut components,
                bundle_writer,
            )?;
        }

        Ok(())
    }

    fn clone_template(&self) -> Box<dyn ErasedTemplate> {
        // `clone_template` has no error channel, so this ladder must be total. Every rung yields a
        // value of the same concrete template type, which `apply` and the typed-recovery path in
        // `ResolvedScene::get_or_insert_template` both rely on.
        let value = self
            .value
            .reflect_clone()
            .ok()
            .or_else(|| {
                self.reflect_from_reflect.as_ref().and_then(|from_reflect| {
                    from_reflect.from_reflect(self.value.as_partial_reflect())
                })
            })
            .unwrap_or_else(|| {
                let mut value = self.reflect_default.default();
                if let Ok(dynamic) = self.value.to_dynamic() {
                    // Best effort: a field that cannot be applied keeps its default.
                    let _ = value.try_apply(&*dynamic);
                }
                value
            });

        Box::new(DynamicComponentTemplate {
            template_type_id: self.template_type_id,
            value,
            reflect_component: self.reflect_component.clone(),
            reflect_template: self.reflect_template.clone(),
            reflect_default: self.reflect_default.clone(),
            reflect_from_reflect: self.reflect_from_reflect.clone(),
            type_registry: self.type_registry.clone(),
        })
    }

    fn try_as_partial_reflect(&self) -> Option<&dyn PartialReflect> {
        Some(self.value.as_partial_reflect())
    }

    fn try_as_partial_reflect_mut(&mut self) -> Option<&mut dyn PartialReflect> {
        Some(self.value.as_partial_reflect_mut())
    }

    fn template_type_id(&self) -> TypeId {
        self.template_type_id
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;

    use bevy_asset::{AssetServer, Assets, Handle, HandleTemplate};
    use bevy_ecs::world::World;
    use bevy_reflect::{tuple_struct::DynamicTupleStruct, TypePath, TypeRegistration};

    use super::*;
    use crate::tests::{test_app, Image, Position, Skipped, Sprite, SpriteTemplate, Unreflected};
    use bevy_scene::{ResolveContext, ResolvedScene, ResolvedSceneRoot, ScenePatch};

    /// Builds a `DynamicComponentTemplate` for `T`, whose output type is `Output`.
    fn template_for<T: 'static, Output: 'static>(
        registry: &AppTypeRegistry,
    ) -> DynamicComponentTemplate {
        let guard = registry.read();
        let template: &TypeRegistration = guard
            .get(TypeId::of::<T>())
            .expect("the template type should be registered");
        let output: &TypeRegistration = guard
            .get(TypeId::of::<Output>())
            .expect("the output type should be registered");
        let reflect_default = template
            .data::<ReflectDefault>()
            .expect("the template should have ReflectDefault")
            .clone();
        let value = reflect_default.default();
        DynamicComponentTemplate::new(
            TypeId::of::<T>(),
            value,
            output
                .data::<ReflectComponent>()
                .expect("the output should have ReflectComponent")
                .clone(),
            template.data::<ReflectTemplate>().cloned(),
            reflect_default,
            template.data::<ReflectFromReflect>().cloned(),
            registry.clone(),
        )
    }

    /// Spawns a `ResolvedScene` holding the given templates and returns the entity.
    fn spawn_with(
        world: &mut World,
        templates: Vec<(TypeId, Box<dyn ErasedTemplate>)>,
    ) -> bevy_ecs::entity::Entity {
        let mut scene = ResolvedScene::default();
        for (type_id, template) in templates {
            scene.insert_erased_template(type_id, template);
        }
        let root = ResolvedSceneRoot { scene };
        root.spawn(world).expect("the scene should spawn").id()
    }

    #[test]
    fn dynamic_template_applies_component() {
        let mut app = test_app();
        let registry = app.world().resource::<AppTypeRegistry>().clone();
        let mut template = template_for::<Position, Position>(&registry);
        template
            .try_as_partial_reflect_mut()
            .unwrap()
            .try_apply(&Position {
                x: 1.,
                y: 2.,
                z: 3.,
            })
            .unwrap();

        let world = app.world_mut();
        let id = spawn_with(world, vec![(TypeId::of::<Position>(), Box::new(template))]);
        assert_eq!(
            *world.entity(id).get::<Position>().unwrap(),
            Position {
                x: 1.,
                y: 2.,
                z: 3.
            }
        );
    }

    #[test]
    fn dynamic_template_single_archetype_move() {
        let mut app = test_app();
        let registry = app.world().resource::<AppTypeRegistry>().clone();
        let position = template_for::<Position, Position>(&registry);
        let marker = template_for::<crate::tests::Marker, crate::tests::Marker>(&registry);
        let bar = template_for::<crate::tests::Bar, crate::tests::Bar>(&registry);

        let world = app.world_mut();
        // Warm up the empty archetype so that the delta only counts archetypes the spawn creates.
        world.spawn_empty();
        let before = world.archetypes().len();
        let id = spawn_with(
            world,
            vec![
                (TypeId::of::<Position>(), Box::new(position)),
                (TypeId::of::<crate::tests::Marker>(), Box::new(marker)),
                (TypeId::of::<crate::tests::Bar>(), Box::new(bar)),
            ],
        );
        let after = world.archetypes().len();

        // Three components, one archetype move: exactly one new archetype.
        assert_eq!(after - before, 1);
        assert_eq!(world.entity(id).archetype().component_count(), 3);
    }

    #[test]
    fn dynamic_template_uses_build_template() {
        let mut app = test_app();
        let expected: Handle<Image> = app.world().resource::<AssetServer>().load("a.png");
        let registry = app.world().resource::<AppTypeRegistry>().clone();
        let mut template = template_for::<SpriteTemplate, Sprite>(&registry);
        let mut patch = DynamicTupleStruct::default();
        patch.insert_boxed(Box::new(HandleTemplate::<Image>::Path("a.png".into())));
        template
            .try_as_partial_reflect_mut()
            .unwrap()
            .try_apply(&patch)
            .unwrap();

        let world = app.world_mut();
        let id = spawn_with(
            world,
            vec![(TypeId::of::<SpriteTemplate>(), Box::new(template))],
        );
        // The template built a `Handle`, not a `HandleTemplate`.
        assert_eq!(world.entity(id).get::<Sprite>().unwrap().0, expected);
    }

    #[test]
    fn clone_template_preserves_concrete_type() {
        let app = test_app();
        let registry = app.world().resource::<AppTypeRegistry>().clone();
        let template = template_for::<Position, Position>(&registry);

        let cloned = template.clone_template();
        let reflect = cloned
            .try_as_partial_reflect()
            .expect("a dynamic template always exposes its value");
        assert_eq!(
            reflect.reflect_type_path(),
            <Position as TypePath>::type_path()
        );
        assert_eq!(cloned.template_type_id(), TypeId::of::<Position>());
    }

    #[test]
    fn clone_template_falls_back_to_from_reflect() {
        let app = test_app();
        let registry = app.world().resource::<AppTypeRegistry>().clone();
        let mut template = template_for::<Skipped, Skipped>(&registry);
        template
            .try_as_partial_reflect_mut()
            .unwrap()
            .try_apply(&Skipped {
                value: 7,
                extra: Default::default(),
            })
            .unwrap();

        // `Skipped` has an ignored non-`Clone` field, so `reflect_clone` fails and the ladder has
        // to fall through to `ReflectFromReflect`.
        assert!(template.value.reflect_clone().is_err());

        let cloned = template.clone_template();
        let reflect = cloned.try_as_partial_reflect().unwrap();
        assert_eq!(
            reflect.reflect_type_path(),
            <Skipped as TypePath>::type_path()
        );
        let value = reflect
            .reflect_ref()
            .as_struct()
            .unwrap()
            .field("value")
            .unwrap()
            .try_downcast_ref::<u32>()
            .unwrap();
        assert_eq!(*value, 7);
    }

    #[test]
    fn template_type_id_is_the_template_not_the_wrapper() {
        let app = test_app();
        let registry = app.world().resource::<AppTypeRegistry>().clone();
        let template = template_for::<SpriteTemplate, Sprite>(&registry);

        assert_eq!(template.template_type_id(), TypeId::of::<SpriteTemplate>());
        assert_ne!(
            template.template_type_id(),
            core::any::Any::type_id(&template)
        );
    }

    #[test]
    fn unpatchable_slot_errors() {
        // A slot filed under `Position`'s id but holding a template of an unregistered,
        // non-reflectable type cannot be patched — and must produce an error, not a panic.
        let mut app = test_app();
        let scene = crate::tests::scene(&app, "a.bsn", "Position { x: 1.0 }");

        let world = app.world_mut();
        let assets = world.resource::<AssetServer>().clone();
        let patches = world.resource::<Assets<ScenePatch>>();
        let registry = world.resource::<AppTypeRegistry>().clone();
        let guard = registry.read();
        let mut context = ResolveContext {
            assets: &assets,
            patches,
            cached: None,
            type_registry: Some(&guard),
        };
        let mut resolved = ResolvedScene::default();
        resolved.insert_erased_template(TypeId::of::<Position>(), Box::new(Unreflected(1)));

        let error = bevy_scene::Scene::resolve(scene, &mut context, &mut resolved).unwrap_err();
        assert!(
            matches!(
                error,
                bevy_scene::ResolveSceneError::UnpatchableTemplate { .. }
            ),
            "unexpected error: {error}"
        );
    }

    /// A probe that answers `true` only when `T: Clone`.
    struct CloneProbe<T>(PhantomData<T>);

    trait NotClone {
        fn is_clone(&self) -> bool {
            false
        }
    }

    impl<T> NotClone for CloneProbe<T> {}

    impl<T: Clone> CloneProbe<T> {
        fn is_clone(&self) -> bool {
            true
        }
    }

    #[test]
    fn dynamic_component_template_is_not_clone() {
        // `DynamicComponentTemplate` must never gain a `Clone` impl: `Clone + Unpin` would pull in
        // the blanket `Template` and `ErasedTemplate` impls and collide with the manual
        // one in this module.
        assert!(!CloneProbe::<DynamicComponentTemplate>(PhantomData).is_clone());
        assert!(CloneProbe::<Position>(PhantomData).is_clone());
    }
}
