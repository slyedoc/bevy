//! Definitions for [`Relationship`] reflection.

use core::any::TypeId;

use bevy_reflect::{CreateTypeData, Reflect, TypePath, TypeRegistration};

use crate::{
    bundle::BundleWriter,
    component::ComponentsRegistrator,
    entity::Entity,
    reflect::ReflectComponent,
    relationship::{Relationship, RelationshipTarget},
};

/// Type data registered on a [`RelationshipTarget`] type (e.g. [`Children`]), exposing enough
/// of the [`Relationship`] pair to build relationships from types only known at runtime.
///
/// It is registered on the **target** rather than on the [`Relationship`] itself because scene
/// formats name the target: `Children [ ... ]`.
///
/// Register it with `#[reflect(RelationshipTarget)]` on the relationship target — note the
/// attribute ident is the *target* trait's name, since the `Reflect` derive prefixes it to reach
/// `ReflectRelationshipTarget`. This also registers [`ReflectComponent`] for the target (via
/// [`CreateTypeData::insert_dependencies`]), so `#[reflect(RelationshipTarget)]` implies
/// `#[reflect(Component)]`.
///
/// [`Children`]: crate::hierarchy::Children
#[derive(Clone, Debug)]
pub struct ReflectRelationshipTarget {
    /// `TypeId::of::<<T as RelationshipTarget>::Relationship>()` — e.g. `ChildOf`.
    ///
    /// The *target*'s own [`TypeId`] is deliberately not stored: it is the id this type data is
    /// registered under, so every consumer already has it.
    pub relationship_type_id: TypeId,
    /// `core::any::type_name::<<T as RelationshipTarget>::Relationship>()`, for error messages.
    pub relationship_name: &'static str,
    /// Pushes `Relationship::from(target)` into a [`BundleWriter`].
    ///
    /// # Safety
    ///
    /// `components` must come from the same [`World`](crate::world::World) as every other push
    /// on `bundle_writer` and as the following [`BundleWriter::write`].
    pub insert_relationship:
        unsafe fn(&mut BundleWriter, &mut ComponentsRegistrator, target: Entity),
    /// Pushes `RelationshipTarget::with_capacity(capacity)` into a [`BundleWriter`].
    ///
    /// # Safety
    ///
    /// Same as [`Self::insert_relationship`].
    pub insert_relationship_target: unsafe fn(&mut BundleWriter, &mut ComponentsRegistrator, usize),
}

impl<T: RelationshipTarget + Reflect + TypePath> CreateTypeData<T> for ReflectRelationshipTarget {
    fn create_type_data(_input: ()) -> Self {
        ReflectRelationshipTarget {
            relationship_type_id: TypeId::of::<T::Relationship>(),
            relationship_name: core::any::type_name::<T::Relationship>(),
            insert_relationship: |bundle_writer, components_registrator, target| {
                let relationship = <T::Relationship as Relationship>::from(target);
                // SAFETY: the caller guarantees `bundle_writer` is only ever used with the
                // `World` that `components_registrator` came from.
                unsafe { bundle_writer.push_component(components_registrator, relationship) };
            },
            insert_relationship_target: |bundle_writer, components_registrator, capacity| {
                let relationship_target = <T as RelationshipTarget>::with_capacity(capacity);
                // SAFETY: the caller guarantees `bundle_writer` is only ever used with the
                // `World` that `components_registrator` came from.
                unsafe {
                    bundle_writer.push_component(components_registrator, relationship_target);
                };
            },
        }
    }

    fn insert_dependencies(type_registration: &mut TypeRegistration) {
        type_registration.register_type_data::<ReflectComponent, T>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bundle::BundleScratch,
        component::Component,
        hierarchy::{ChildOf, Children},
        world::World,
    };
    use alloc::vec::Vec;
    use bevy_reflect::TypeRegistry;

    #[test]
    fn children_registers_reflect_relationship() {
        let mut registry = TypeRegistry::empty();
        registry.register::<Children>();

        let data = registry
            .get_type_data::<ReflectRelationshipTarget>(TypeId::of::<Children>())
            .unwrap();
        assert_eq!(data.relationship_type_id, TypeId::of::<ChildOf>());
        assert_eq!(data.relationship_name, core::any::type_name::<ChildOf>());
    }

    #[test]
    fn reflect_relationship_target_registers_reflect_component() {
        let mut registry = TypeRegistry::empty();
        registry.register::<Children>();

        assert!(registry
            .get_type_data::<ReflectComponent>(TypeId::of::<Children>())
            .is_some());
    }

    #[test]
    fn reflect_relationship_target_inserts_relationship() {
        let mut registry = TypeRegistry::empty();
        registry.register::<Children>();
        let data = registry
            .get_type_data::<ReflectRelationshipTarget>(TypeId::of::<Children>())
            .unwrap()
            .clone();

        let mut world = World::new();
        let parent = world.spawn_empty().id();
        let mut bundle_scratch = BundleScratch::default();
        let mut bundle_writer = bundle_scratch.writer();
        // SAFETY: the same world is used for every bundle_writer operation
        let child = unsafe {
            let mut components = world.components_registrator();
            (data.insert_relationship)(&mut bundle_writer, &mut components, parent);
            let mut child = world.spawn_empty();
            bundle_writer.write(&mut child);
            child.id()
        };

        assert_eq!(
            world.entity(child).get::<ChildOf>().unwrap().parent(),
            parent
        );
        assert_eq!(
            world
                .entity(parent)
                .get::<Children>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [child]
        );
    }

    #[test]
    fn reflect_relationship_target_inserts_relationship_target_with_capacity() {
        let mut registry = TypeRegistry::empty();
        registry.register::<Children>();
        let data = registry
            .get_type_data::<ReflectRelationshipTarget>(TypeId::of::<Children>())
            .unwrap()
            .clone();

        let mut world = World::new();
        let mut bundle_scratch = BundleScratch::default();
        let mut bundle_writer = bundle_scratch.writer();
        // SAFETY: the same world is used for every bundle_writer operation
        let entity = unsafe {
            let mut components = world.components_registrator();
            (data.insert_relationship_target)(&mut bundle_writer, &mut components, 4);
            let mut entity = world.spawn_empty();
            bundle_writer.write(&mut entity);
            entity.id()
        };

        let children = world.entity(entity).get::<Children>().unwrap();
        assert!(children.is_empty());
    }

    #[test]
    fn reflect_relationship_target_custom() {
        #[derive(Component, Reflect)]
        #[relationship(relationship_target = LikedBy)]
        #[reflect(Component)]
        struct Likes(Entity);

        #[derive(Component, Reflect)]
        #[relationship_target(relationship = Likes)]
        #[reflect(Component, RelationshipTarget)]
        struct LikedBy(Vec<Entity>);

        let mut registry = TypeRegistry::empty();
        registry.register::<LikedBy>();

        let data = registry
            .get_type_data::<ReflectRelationshipTarget>(TypeId::of::<LikedBy>())
            .unwrap();
        assert_eq!(data.relationship_type_id, TypeId::of::<Likes>());
        assert_eq!(data.relationship_name, core::any::type_name::<Likes>());
    }
}
