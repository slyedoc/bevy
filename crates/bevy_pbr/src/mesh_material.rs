use crate::Material;
use bevy_asset::{AsAssetId, AssetId, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    reflect::{ReflectComponent, ReflectFromTemplate},
    template::FromTemplate,
};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use derive_more::derive::From;

/// A [material](Material) used for rendering a [`Mesh3d`].
///
/// See [`Material`] for general information about 3D materials and how to implement your own materials.
///
/// [`Mesh3d`]: bevy_mesh::Mesh3d
///
/// # Example
///
/// ```
/// # use bevy_pbr::{Material, MeshMaterial3d, StandardMaterial};
/// # use bevy_ecs::prelude::*;
/// # use bevy_mesh::{Mesh, Mesh3d};
/// # use bevy_color::palettes::basic::RED;
/// # use bevy_asset::Assets;
/// # use bevy_shape::Capsule3d;
/// #
/// // Spawn an entity with a mesh using `StandardMaterial`.
/// fn setup(
///     mut commands: Commands,
///     mut meshes: ResMut<Assets<Mesh>>,
///     mut materials: ResMut<Assets<StandardMaterial>>,
/// ) {
///     commands.spawn((
///         Mesh3d(meshes.add(Capsule3d::default())),
///         MeshMaterial3d(materials.add(StandardMaterial {
///             base_color: RED.into(),
///             ..Default::default()
///         })),
///     ));
/// }
/// ```
#[derive(Component, FromTemplate, Clone, Debug, Deref, DerefMut, Reflect, From)]
#[reflect(Component, Default, Clone, PartialEq, FromTemplate)]
#[template(reflect)]
pub struct MeshMaterial3d<M: Material>(pub Handle<M>);

impl<M: Material> Default for MeshMaterial3d<M> {
    fn default() -> Self {
        Self(Handle::default())
    }
}

impl<M: Material> PartialEq for MeshMaterial3d<M> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<M: Material> Eq for MeshMaterial3d<M> {}

impl<M: Material> From<MeshMaterial3d<M>> for AssetId<M> {
    fn from(material: MeshMaterial3d<M>) -> Self {
        material.id()
    }
}

impl<M: Material> From<&MeshMaterial3d<M>> for AssetId<M> {
    fn from(material: &MeshMaterial3d<M>) -> Self {
        material.id()
    }
}

impl<M: Material> AsAssetId for MeshMaterial3d<M> {
    type Asset = M;

    fn as_asset_id(&self) -> AssetId<Self::Asset> {
        self.id()
    }
}

#[cfg(test)]
mod template_reflect_tests {
    use super::*;
    use bevy_ecs::reflect::{ReflectFromTemplate, ReflectTemplate};
    use bevy_reflect::{std_traits::ReflectDefault, TypeRegistry};
    use core::any::TypeId;

    /// Walks the exact lookup chain a reflection-driven scene format performs: component type →
    /// [`ReflectFromTemplate`] → template registration → [`ReflectTemplate`] + `ReflectDefault`.
    fn assert_template_chain<C: bevy_reflect::GetTypeRegistration + 'static>(
        registry: &TypeRegistry,
    ) {
        let from_template = registry
            .get_type_data::<ReflectFromTemplate>(TypeId::of::<C>())
            .expect("component should carry `ReflectFromTemplate`");
        let template = registry
            .get(from_template.template_type_id)
            .unwrap_or_else(|| panic!("`{}` is not registered", from_template.template_type_path));
        assert!(template.data::<ReflectTemplate>().is_some());
        assert!(template.data::<ReflectDefault>().is_some());
    }

    #[test]
    fn template_type_registered_for_seed_set() {
        let mut registry = TypeRegistry::empty();
        registry.register::<MeshMaterial3d<crate::StandardMaterial>>();
        registry.register::<MeshMaterial3dTemplate<crate::StandardMaterial>>();
        assert_template_chain::<MeshMaterial3d<crate::StandardMaterial>>(&registry);
    }
}
