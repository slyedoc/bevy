use bevy_asset::{AssetId, Handle};
use bevy_camera::visibility::NoCpuCulling;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{component::Component, prelude::ReflectComponent, template::FromTemplate};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::sync_world::SyncToRenderWorld;
use bevy_transform::components::{GlobalTransform, Transform};
use derive_more::derive::From;

use crate::geometry::ClusterMesh;
use crate::material::SolariMaterial3d;

/// Component that marks an entity as a ray-tracing instance.
///
/// Holds a [`Handle`] to a baked [`ClusterMesh`] asset (clustered-LOD
/// geometry). At runtime the entity's clusters are tracked by
/// [`crate::instance::InstanceManager`], fed through the DAG-cut
/// selector, and referenced by a per-instance BLAS that the
/// partitioned TLAS traverses for ray hits.
///
/// Pairs with [`SolariMaterial3d`] — `bevy_solari` owns its material type
/// (`StandardSolariMaterial`) rather than depending on `bevy_pbr`'s raster material
/// infrastructure, so the RT path can run with `PbrPlugin` disabled.
#[derive(
    Component, FromTemplate, Clone, Debug, Default, Deref, DerefMut, Reflect, PartialEq, Eq, From,
)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(
    SolariMaterial3d,
    Transform,
    GlobalTransform,
    NoCpuCulling,
    SyncToRenderWorld,
)]
pub struct RaytracingMesh3d(pub Handle<ClusterMesh>);

impl From<RaytracingMesh3d> for AssetId<ClusterMesh> {
    fn from(mesh: RaytracingMesh3d) -> Self {
        mesh.id()
    }
}

impl From<&RaytracingMesh3d> for AssetId<ClusterMesh> {
    fn from(mesh: &RaytracingMesh3d) -> Self {
        mesh.id()
    }
}
