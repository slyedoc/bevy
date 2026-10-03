//! Definitions for [`Template`] and [`FromTemplate`] reflection.
//!
//! These two type data types let reflection-driven scene formats (e.g. dynamically loaded
//! `.bsn` assets) answer two questions that are otherwise only expressible in the type system:
//!
//! * Given a component type `C`, what is its [`FromTemplate::Template`] type?
//!   → [`ReflectFromTemplate`], registered with `#[reflect(FromTemplate)]` on `C`.
//! * Given a template value, how do I run [`Template::build_template`] on it?
//!   → [`ReflectTemplate`], registered with `#[reflect(Template)]` on the template type.
//!
//! Both are *opt-in*: their absence has a defined meaning (see each type's docs), so the
//! overwhelmingly common `Clone + Default` blanket case needs no annotation at all.

use alloc::boxed::Box;
use core::any::TypeId;

use bevy_reflect::{CreateTypeData, Reflect, TypePath};

use crate::{
    error::BevyError,
    template::{FromTemplate, Template, TemplateContext},
};

/// Type data registered on a **component** type `C` that has a custom [`FromTemplate`] impl,
/// recording which type is its [`FromTemplate::Template`].
///
/// Register it with `#[reflect(Component, FromTemplate)]` on the component (the
/// `ReflectFromTemplate` type must be in scope, e.g. via `bevy_ecs::prelude::*`):
///
/// ```
/// # use bevy_ecs::prelude::*;
/// # use bevy_ecs::template::{Template, TemplateContext};
/// # use bevy_reflect::prelude::*;
/// #[derive(Component, Reflect, PartialEq, Debug)]
/// #[reflect(Component, FromTemplate)]
/// struct Health(u32);
///
/// #[derive(Reflect, Default)]
/// #[reflect(Default, Template)]
/// struct HealthTemplate(u32);
///
/// impl FromTemplate for Health {
///     type Template = HealthTemplate;
/// }
///
/// impl Template for HealthTemplate {
///     type Output = Health;
///     fn build_template(&self, _context: &mut TemplateContext) -> Result<Self::Output> {
///         Ok(Health(self.0))
///     }
///     fn clone_template(&self) -> Self {
///         HealthTemplate(self.0)
///     }
/// }
/// ```
///
/// **The absence of this type data means `<C as FromTemplate>::Template == C`** — i.e. the
/// `Clone + Default + Unpin` blanket impl in [`crate::template`]. Consumers must default to
/// `TypeId::of::<C>()` when this data is missing, *not* error.
#[derive(Clone, Debug)]
pub struct ReflectFromTemplate {
    /// `TypeId::of::<<C as FromTemplate>::Template>()`.
    pub template_type_id: TypeId,
    /// `<<C as FromTemplate>::Template as TypePath>::type_path()`.
    ///
    /// Consumers need this for error messages in exactly the case where the [`TypeId`] is
    /// useless: when the template type is *not* present in the `TypeRegistry` and so cannot be
    /// named from the id alone.
    pub template_type_path: &'static str,
}

impl<T> CreateTypeData<T> for ReflectFromTemplate
where
    T: FromTemplate,
    T::Template: Reflect + TypePath,
    <T::Template as Template>::Output: Reflect,
{
    fn create_type_data(_input: ()) -> Self {
        ReflectFromTemplate {
            template_type_id: TypeId::of::<T::Template>(),
            template_type_path: <T::Template as TypePath>::type_path(),
        }
    }
}

/// Type data registered on a **template** type `T` whose [`Template::Output`] differs from `T`,
/// exposing an erased [`Template::build_template`].
///
/// Register it with `#[reflect(Template)]` on the template type. See [`ReflectFromTemplate`] for
/// a complete example.
///
/// **The absence of this type data means the output equals the template**, so a consumer should
/// fall back to cloning the template value
/// (`PartialReflect::reflect_clone`).
#[derive(Clone)]
pub struct ReflectTemplate {
    /// `TypeId::of::<<T as Template>::Output>()`.
    ///
    /// Lets a consumer answer "what does this template produce?" without building it — for
    /// example to locate [`ReflectComponent`](crate::reflect::ReflectComponent) on the output
    /// type up front, or to discover `Handle`-typed outputs for asset-dependency registration.
    pub output_type_id: TypeId,
    /// Erased [`Template::build_template`].
    ///
    /// The first argument must be a value of the template type this data was registered for;
    /// passing anything else returns an error rather than panicking. The returned box always
    /// holds a concrete `<T as Template>::Output`, i.e. its `Any::type_id()` always equals
    /// [`Self::output_type_id`].
    pub build_template:
        fn(&dyn Reflect, &mut TemplateContext) -> Result<Box<dyn Reflect>, BevyError>,
}

impl core::fmt::Debug for ReflectTemplate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ReflectTemplate")
            .field("output_type_id", &self.output_type_id)
            .finish_non_exhaustive()
    }
}

impl<T> CreateTypeData<T> for ReflectTemplate
where
    T: Template + Reflect,
    <T as Template>::Output: Reflect,
{
    fn create_type_data(_input: ()) -> Self {
        ReflectTemplate {
            output_type_id: TypeId::of::<<T as Template>::Output>(),
            build_template: |this, context| {
                let Some(this) = this.downcast_ref::<T>() else {
                    return Err(BevyError::error(
                        "`ReflectTemplate::build_template` was called with a value that is not \
                         of the template type it was registered for",
                    ));
                };
                Ok(Box::new(<T as Template>::build_template(this, context)?))
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        component::Component, entity::Entity, error::Result, reflect::ReflectComponent,
        template::SceneEntityReferences, world::World,
    };
    use bevy_reflect::{std_traits::ReflectDefault, TypeRegistry};

    #[derive(Component, Reflect, PartialEq, Debug)]
    #[reflect(Component, FromTemplate)]
    struct CustomComponent(u32);

    #[derive(Reflect, Default)]
    #[reflect(Default, Template)]
    struct CustomComponentTemplate(u32);

    impl FromTemplate for CustomComponent {
        type Template = CustomComponentTemplate;
    }

    impl Template for CustomComponentTemplate {
        type Output = CustomComponent;

        fn build_template(&self, _context: &mut TemplateContext) -> Result<Self::Output> {
            Ok(CustomComponent(self.0))
        }

        fn clone_template(&self) -> Self {
            CustomComponentTemplate(self.0)
        }
    }

    /// A template whose `build_template` always fails.
    #[derive(Reflect, Default)]
    #[reflect(Default, Template)]
    struct FailingTemplate;

    impl Template for FailingTemplate {
        type Output = CustomComponent;

        fn build_template(&self, _context: &mut TemplateContext) -> Result<Self::Output> {
            Err(BevyError::error("boom"))
        }

        fn clone_template(&self) -> Self {
            FailingTemplate
        }
    }

    /// A template that uses the [`TemplateContext`], proving the higher-ranked function pointer
    /// signature is usable.
    #[derive(Reflect, Default)]
    #[reflect(Default, Template)]
    struct ContextTemplate;

    impl Template for ContextTemplate {
        type Output = ContextOutput;

        fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output> {
            Ok(ContextOutput(context.entity.id()))
        }

        fn clone_template(&self) -> Self {
            ContextTemplate
        }
    }

    #[derive(Reflect, Debug, PartialEq)]
    struct ContextOutput(Entity);

    #[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
    #[reflect(Component, Default, FromTemplate)]
    struct Plain(u32);

    #[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
    #[reflect(Component, Default)]
    struct Unannotated(u32);

    fn registry() -> TypeRegistry {
        let mut registry = TypeRegistry::empty();
        registry.register::<CustomComponent>();
        registry.register::<CustomComponentTemplate>();
        registry.register::<FailingTemplate>();
        registry.register::<ContextTemplate>();
        registry.register::<Plain>();
        registry.register::<Unannotated>();
        registry
    }

    #[test]
    fn from_template_reports_custom_template() {
        let registry = registry();
        let data = registry
            .get_type_data::<ReflectFromTemplate>(TypeId::of::<CustomComponent>())
            .unwrap();
        assert_eq!(
            data.template_type_id,
            TypeId::of::<CustomComponentTemplate>()
        );
    }

    #[test]
    fn from_template_reports_template_type_path() {
        // Deliberately does *not* register `CustomComponentTemplate`: the whole point of the
        // `template_type_path` field is to name a template that is missing from the registry.
        let mut registry = TypeRegistry::empty();
        registry.register::<CustomComponent>();
        let data = registry
            .get_type_data::<ReflectFromTemplate>(TypeId::of::<CustomComponent>())
            .unwrap();
        assert!(registry.get(data.template_type_id).is_none());
        assert_eq!(
            data.template_type_path,
            <CustomComponentTemplate as TypePath>::type_path()
        );
    }

    #[test]
    fn template_reports_output_type_id() {
        let registry = registry();
        let data = registry
            .get_type_data::<ReflectTemplate>(TypeId::of::<CustomComponentTemplate>())
            .unwrap();
        assert_eq!(data.output_type_id, TypeId::of::<CustomComponent>());
    }

    #[test]
    fn template_output_type_id_matches_built_value() {
        let registry = registry();
        let data = registry
            .get_type_data::<ReflectTemplate>(TypeId::of::<CustomComponentTemplate>())
            .unwrap();
        let template = CustomComponentTemplate(7);

        let mut world = World::new();
        let mut references = SceneEntityReferences::default();
        let mut entity = world.spawn_empty();
        let mut context = TemplateContext::new(&mut entity, &mut references);

        let built = (data.build_template)(&template, &mut context).unwrap();
        assert_eq!(built.as_any().type_id(), data.output_type_id);
    }

    #[test]
    fn from_template_absent_means_self() {
        let registry = registry();
        assert!(registry
            .get_type_data::<ReflectFromTemplate>(TypeId::of::<Unannotated>())
            .is_none());
    }

    #[test]
    fn from_template_registered_on_blanket_type_is_self() {
        let registry = registry();
        let data = registry
            .get_type_data::<ReflectFromTemplate>(TypeId::of::<Plain>())
            .unwrap();
        assert_eq!(data.template_type_id, TypeId::of::<Plain>());
    }

    #[test]
    fn template_build_produces_output() {
        let registry = registry();
        let data = registry
            .get_type_data::<ReflectTemplate>(TypeId::of::<CustomComponentTemplate>())
            .unwrap();
        let template = CustomComponentTemplate(7);

        let mut world = World::new();
        let mut references = SceneEntityReferences::default();
        let mut entity = world.spawn_empty();
        let mut context = TemplateContext::new(&mut entity, &mut references);

        let built = (data.build_template)(&template, &mut context).unwrap();
        assert_eq!(
            *built.downcast::<CustomComponent>().unwrap(),
            CustomComponent(7)
        );
    }

    #[test]
    fn template_build_rejects_wrong_receiver() {
        let registry = registry();
        let data = registry
            .get_type_data::<ReflectTemplate>(TypeId::of::<CustomComponentTemplate>())
            .unwrap();

        let mut world = World::new();
        let mut references = SceneEntityReferences::default();
        let mut entity = world.spawn_empty();
        let mut context = TemplateContext::new(&mut entity, &mut references);

        let wrong = Plain(1);
        assert!((data.build_template)(&wrong, &mut context).is_err());
    }

    #[test]
    fn template_build_propagates_error() {
        let registry = registry();
        let data = registry
            .get_type_data::<ReflectTemplate>(TypeId::of::<FailingTemplate>())
            .unwrap();

        let mut world = World::new();
        let mut references = SceneEntityReferences::default();
        let mut entity = world.spawn_empty();
        let mut context = TemplateContext::new(&mut entity, &mut references);

        let error = (data.build_template)(&FailingTemplate, &mut context).unwrap_err();
        assert!(alloc::format!("{error}").contains("boom"));
    }

    #[test]
    fn template_build_can_use_context() {
        let registry = registry();
        let data = registry
            .get_type_data::<ReflectTemplate>(TypeId::of::<ContextTemplate>())
            .unwrap();

        let mut world = World::new();
        let mut references = SceneEntityReferences::default();
        let mut entity = world.spawn_empty();
        let expected = entity.id();
        let mut context = TemplateContext::new(&mut entity, &mut references);

        let built = (data.build_template)(&ContextTemplate, &mut context).unwrap();
        assert_eq!(
            *built.downcast::<ContextOutput>().unwrap(),
            ContextOutput(expected)
        );
    }
}
