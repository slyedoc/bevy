//! Types that enable reflection support.
//!
//! Most of the type data here mirrors an ECS trait so that it can be used for types only known at
//! runtime: [`ReflectComponent`], [`ReflectResource`], [`ReflectBundle`], [`ReflectEvent`],
//! [`ReflectMessage`] and [`ReflectFromWorld`].
//!
//! Three additional type data types exist for reflection-driven *scene* formats (e.g. dynamically
//! loaded `.bsn` assets), which must build [`Template`](crate::template::Template)s and
//! relationships from data alone:
//!
//! * [`ReflectFromTemplate`] — registered on a component with `#[reflect(FromTemplate)]`, it says
//!   which type is that component's [`FromTemplate::Template`](crate::template::FromTemplate::Template).
//!   Its *absence* means the template is the component type itself.
//! * [`ReflectTemplate`] — registered on a template type with `#[reflect(Template)]`, it exposes an
//!   erased [`Template::build_template`](crate::template::Template::build_template). Its *absence*
//!   means the template's output is the template itself.
//! * [`ReflectRelationshipTarget`] — registered on a
//!   [`RelationshipTarget`](crate::relationship::RelationshipTarget) with
//!   `#[reflect(RelationshipTarget)]`, it exposes the pair of erased pushes needed to build a
//!   relationship between two entities.
//!
//! Building values from reflected data is done with [`from_reflect_with_fallback`] (panicking) or
//! [`try_from_reflect_with_fallback`] (returning [`FromReflectError`]). Scene formats operate on
//! user data and should always prefer the latter.

use core::{
    any::TypeId,
    ops::{Deref, DerefMut},
};

use crate::{resource::Resource, world::World};
use alloc::boxed::Box;
use bevy_reflect::{
    std_traits::ReflectDefault, ApplyError, PartialReflect, Reflect, ReflectFromReflect, TypePath,
    TypeRegistry, TypeRegistryArc,
};
use thiserror::Error;

mod bundle;
mod component;
mod entity_commands;
mod event;
mod from_world;
mod map_entities;
mod message;
mod relationship;
mod resource;
mod template;

use bevy_utils::prelude::DebugName;
pub use bundle::{ReflectBundle, ReflectBundleFns};
pub use component::{ReflectComponent, ReflectComponentFns};
pub use entity_commands::ReflectCommandExt;
pub use event::{ReflectEvent, ReflectEventFns};
pub use from_world::{ReflectFromWorld, ReflectFromWorldFns};
pub use map_entities::ReflectMapEntities;
pub use message::{ReflectMessage, ReflectMessageFns};
pub use relationship::ReflectRelationshipTarget;
pub use resource::ReflectResource;
pub use template::{ReflectFromTemplate, ReflectTemplate};

/// A [`Resource`] storing [`TypeRegistry`] for
/// type registrations relevant to a whole app.
#[derive(Resource, Clone, Default)]
pub struct AppTypeRegistry(pub TypeRegistryArc);

impl Deref for AppTypeRegistry {
    type Target = TypeRegistryArc;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for AppTypeRegistry {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AppTypeRegistry {
    /// Creates [`AppTypeRegistry`] and automatically registers all types deriving [`Reflect`].
    ///
    /// See [`TypeRegistry::register_derived_types`] for more details.
    #[cfg(feature = "reflect_auto_register")]
    pub fn new_with_derived_types() -> Self {
        let app_registry = AppTypeRegistry::default();
        app_registry.write().register_derived_types();
        app_registry
    }
}

/// A [`Resource`] storing [`FunctionRegistry`] for
/// function registrations relevant to a whole app.
///
/// [`FunctionRegistry`]: bevy_reflect::func::FunctionRegistry
#[cfg(feature = "reflect_functions")]
#[derive(Resource, Clone, Default)]
pub struct AppFunctionRegistry(pub bevy_reflect::func::FunctionRegistryArc);

#[cfg(feature = "reflect_functions")]
impl Deref for AppFunctionRegistry {
    type Target = bevy_reflect::func::FunctionRegistryArc;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(feature = "reflect_functions")]
impl DerefMut for AppFunctionRegistry {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Errors returned by [`try_from_reflect_with_fallback`].
#[derive(Error, Debug)]
pub enum FromReflectError {
    /// The target type is not present in the [`TypeRegistry`] at all.
    #[error("The type `{type_name}` is not registered in the `TypeRegistry`")]
    NotRegistered {
        /// The name of the type that could not be constructed.
        type_name: DebugName,
    },
    /// The target type is registered, but carries neither the type data needed to construct a
    /// value nor a way to fall back to a default.
    #[error(
        "Couldn't create an instance of `{type_name}` using the reflected {traits}. \
         Are you perhaps missing a `#[reflect(Default)]` attribute?"
    )]
    MissingConstructor {
        /// The name of the type that could not be constructed.
        type_name: DebugName,
        /// A human-readable list of the traits that were tried.
        traits: &'static str,
    },
    /// A default value was produced, but applying the reflected value on top of it failed.
    #[error("Failed to apply a reflected value onto a default instance of `{type_name}`: {error}")]
    ApplyFailed {
        /// The name of the type being constructed.
        type_name: DebugName,
        /// The underlying apply error.
        #[source]
        error: ApplyError,
    },
    /// The registered type data produced a value of the wrong concrete type.
    #[error(
        "The registration for the reflected `{source_trait}` trait for the type `{type_name}` \
         produced a value of a different type"
    )]
    MismatchedType {
        /// The name of the expected type.
        type_name: DebugName,
        /// Which reflected trait produced the wrong value.
        source_trait: &'static str,
    },
}

/// The shared, type-erased implementation behind [`from_reflect_with_fallback`] and
/// [`try_from_reflect_with_fallback`].
///
/// Strategies, in order: reflected `FromReflect`; reflected `Default` + `try_apply`; reflected
/// `FromWorld` + `try_apply` (only when `world` is `Some`).
#[inline(never)]
fn from_reflect_erased(
    reflected: &dyn PartialReflect,
    world: Option<&mut World>,
    registry: &TypeRegistry,
    type_id: TypeId,
    type_name: DebugName,
) -> Result<Box<dyn Reflect>, FromReflectError> {
    // `world` is consumed by the `.zip(world)` below, so remember whether it was available for
    // the error message.
    let world_was_available = world.is_some();

    if registry.get(type_id).is_none() {
        return Err(FromReflectError::NotRegistered { type_name });
    }

    // First, try `FromReflect`. This is handled differently from the others because
    // it doesn't need a subsequent `apply` and may fail.
    // If it fails it's ok, we can continue checking `Default` and `FromWorld`.
    let (mut value, source_trait) = if let Some(value) = registry
        .get_type_data::<ReflectFromReflect>(type_id)
        .and_then(|reflect_from_reflect| reflect_from_reflect.from_reflect(reflected))
    {
        (value, "FromReflect")
    }
    // Create an instance using either the reflected `Default` or `FromWorld`.
    else if let Some(reflect_default) = registry.get_type_data::<ReflectDefault>(type_id) {
        (reflect_default.default(), "Default")
    } else if let Some((reflect_from_world, world)) = registry
        .get_type_data::<ReflectFromWorld>(type_id)
        .zip(world)
    {
        (reflect_from_world.from_world(world), "FromWorld")
    } else {
        return Err(FromReflectError::MissingConstructor {
            type_name,
            traits: if world_was_available {
                "`FromReflect`, `Default` or `FromWorld` traits"
            } else {
                "`FromReflect` or `Default` traits"
            },
        });
    };

    if source_trait != "FromReflect" {
        value
            .try_apply(reflected)
            .map_err(|error| FromReflectError::ApplyFailed {
                type_name: type_name.clone(),
                error,
            })?;
    }

    if value.as_any().type_id() != type_id {
        return Err(FromReflectError::MismatchedType {
            type_name,
            source_trait,
        });
    }
    Ok(value)
}

/// Creates a `T` from a `&dyn PartialReflect`, returning an error instead of panicking.
///
/// This will try the following strategies, in this order:
///
/// - use the reflected `FromReflect`, if it's present and doesn't fail;
/// - use the reflected `Default`, if it's present, and then call `try_apply` on the result.
///
/// Unlike [`from_reflect_with_fallback`] this has no access to a [`World`], so the reflected
/// `FromWorld` strategy is not available: only reflected `FromReflect` and `Default` are tried.
/// This is the variant used by the reflection-driven scene path, where the input is user data
/// and failures must be reported rather than aborting the process.
pub fn try_from_reflect_with_fallback<T: Reflect>(
    reflected: &dyn PartialReflect,
    registry: &TypeRegistry,
) -> Result<T, FromReflectError> {
    // FIXME: once we have unique reflect, use `TypePath`.
    let type_name = DebugName::type_name::<T>();
    let value = from_reflect_erased(reflected, None, registry, TypeId::of::<T>(), type_name)?;
    // `from_reflect_erased` already verified the concrete `TypeId` matches `T`.
    match value.downcast::<T>() {
        Ok(value) => Ok(*value),
        Err(_) => Err(FromReflectError::MismatchedType {
            type_name: DebugName::type_name::<T>(),
            source_trait: "unknown",
        }),
    }
}

/// Creates a `T` from a `&dyn PartialReflect`.
///
/// This will try the following strategies, in this order:
///
/// - use the reflected `FromReflect`, if it's present and doesn't fail;
/// - use the reflected `Default`, if it's present, and then call `apply` on the result;
/// - use the reflected `FromWorld`, just like the `Default`.
///
/// The first one that is present and doesn't fail will be used.
///
/// See [`try_from_reflect_with_fallback`] for a non-panicking variant.
///
/// # Panics
///
/// If any strategy produces a `Box<dyn Reflect>` that doesn't store a value of type `T`
/// this method will panic.
///
/// If none of the strategies succeed, this method will panic.
pub fn from_reflect_with_fallback<T: Reflect + TypePath>(
    reflected: &dyn PartialReflect,
    world: &mut World,
    registry: &TypeRegistry,
) -> T {
    // FIXME: once we have unique reflect, use `TypePath`.
    let type_name = DebugName::type_name::<T>();
    match from_reflect_erased(
        reflected,
        Some(world),
        registry,
        TypeId::of::<T>(),
        type_name,
    ) {
        Ok(value) => *value
            .downcast::<T>()
            .unwrap_or_else(|_| panic!("Reflected value was not of the expected type")),
        Err(error) => panic!("{error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::FromWorld;
    use bevy_reflect::{structs::DynamicStruct, TypeRegistry};

    #[derive(Reflect, PartialEq, Debug)]
    struct Unregistered {
        value: u32,
    }

    #[derive(Reflect, PartialEq, Debug)]
    #[reflect(from_reflect = false)]
    struct NoConstructor {
        value: u32,
    }

    #[derive(Reflect, PartialEq, Debug)]
    struct FromReflectable {
        value: u32,
        other: u32,
    }

    #[derive(Reflect, Default, PartialEq, Debug)]
    #[reflect(Default, from_reflect = false)]
    struct DefaultOnly {
        value: u32,
        other: u32,
    }

    #[derive(Reflect, PartialEq, Debug)]
    #[reflect(FromWorld, from_reflect = false)]
    struct FromWorldOnly {
        value: u32,
        other: u32,
    }

    impl FromWorld for FromWorldOnly {
        fn from_world(_world: &mut World) -> Self {
            FromWorldOnly {
                value: 0,
                other: 42,
            }
        }
    }

    fn registry() -> TypeRegistry {
        let mut registry = TypeRegistry::empty();
        registry.register::<NoConstructor>();
        registry.register::<FromReflectable>();
        registry.register::<DefaultOnly>();
        registry.register::<FromWorldOnly>();
        registry
    }

    fn dynamic(fields: &[(&str, u32)]) -> DynamicStruct {
        let mut value = DynamicStruct::default();
        for (name, field) in fields {
            value.insert(*name, *field);
        }
        value
    }

    #[test]
    fn try_from_reflect_unregistered_type_errors() {
        let registry = registry();
        let value = dynamic(&[("value", 3)]);
        let error = try_from_reflect_with_fallback::<Unregistered>(&value, &registry).unwrap_err();
        assert!(matches!(error, FromReflectError::NotRegistered { .. }));
    }

    #[test]
    fn try_from_reflect_missing_constructor_errors() {
        let registry = registry();
        let value = dynamic(&[("value", 3)]);
        let error = try_from_reflect_with_fallback::<NoConstructor>(&value, &registry).unwrap_err();
        assert!(matches!(error, FromReflectError::MissingConstructor { .. }));
        let message = alloc::format!("{error}");
        assert!(
            message.contains("`FromReflect` or `Default` traits"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn try_from_reflect_uses_from_reflect() {
        let registry = registry();
        let value = dynamic(&[("value", 3), ("other", 5)]);
        let built = try_from_reflect_with_fallback::<FromReflectable>(&value, &registry).unwrap();
        assert_eq!(built, FromReflectable { value: 3, other: 5 });
    }

    #[test]
    fn try_from_reflect_uses_default_and_applies() {
        let registry = registry();
        let value = dynamic(&[("value", 3)]);
        let built = try_from_reflect_with_fallback::<DefaultOnly>(&value, &registry).unwrap();
        assert_eq!(built, DefaultOnly { value: 3, other: 0 });
    }

    #[test]
    fn from_reflect_with_fallback_still_uses_from_world() {
        let registry = registry();
        let mut world = World::new();
        let value = dynamic(&[("value", 3)]);
        let built = from_reflect_with_fallback::<FromWorldOnly>(&value, &mut world, &registry);
        assert_eq!(
            built,
            FromWorldOnly {
                value: 3,
                other: 42
            }
        );
    }
}
