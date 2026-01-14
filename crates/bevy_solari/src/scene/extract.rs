use super::RaytracingMesh3d;
use bevy_asset::{Assets, UntypedAssetId};
use bevy_derive::Deref;
use bevy_ecs::{
    component::Component,
    resource::Resource,
    schedule::SystemSet,
    system::{Commands, Query, Res, ResMut},
};
use bevy_pbr::{ExtendedMaterial, MaterialExtension, MeshMaterial3d, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_render::{sync_world::RenderEntity, Extract};
use bevy_transform::components::GlobalTransform;

/// System set for raytracing material extraction.
/// All material extraction systems should run after [`RaytracingMaterialExtractionSystems::Clear`].
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum RaytracingMaterialExtractionSystems {
    /// Clears the material storage before extraction.
    Clear,
    /// Extracts materials into the storage.
    Extract,
}

/// Extracted raytracing material data, keyed by UntypedAssetId.
/// This allows both StandardMaterial and ExtendedMaterial<StandardMaterial, E> to be stored.
#[derive(Resource, Deref, Default)]
pub struct RaytracingMaterialAssets(pub HashMap<UntypedAssetId, StandardMaterial>);

/// Component storing the material reference for raytracing instances.
#[derive(Component, Clone)]
pub struct ExtractedRaytracingMaterial(pub UntypedAssetId);

/// Clears raytracing material storage before extraction.
/// This must run before all material extraction systems.
pub fn clear_raytracing_materials(mut raytracing_materials: ResMut<RaytracingMaterialAssets>) {
    raytracing_materials.0.clear();
}

/// Extracts StandardMaterial assets for raytracing.
pub fn extract_standard_materials(
    materials: Extract<Res<Assets<StandardMaterial>>>,
    mut raytracing_materials: ResMut<RaytracingMaterialAssets>,
) {
    for (id, material) in materials.iter() {
        raytracing_materials.0.insert(id.untyped(), material.clone());
    }
}

/// Extracts ExtendedMaterial<StandardMaterial, E> assets for raytracing.
/// Only the base StandardMaterial is extracted since raytracing doesn't evaluate custom shaders.
pub fn extract_extended_materials<E: MaterialExtension>(
    materials: Extract<Res<Assets<ExtendedMaterial<StandardMaterial, E>>>>,
    mut raytracing_materials: ResMut<RaytracingMaterialAssets>,
) {
    for (id, material) in materials.iter() {
        raytracing_materials.0.insert(id.untyped(), material.base.clone());
    }
}

/// Extracts raytracing instances with StandardMaterial.
pub fn extract_raytracing_instances_standard(
    instances: Extract<
        Query<(
            RenderEntity,
            &RaytracingMesh3d,
            &MeshMaterial3d<StandardMaterial>,
            &GlobalTransform,
        )>,
    >,
    mut commands: Commands,
) {
    for (render_entity, mesh, material, transform) in &instances {
        commands.entity(render_entity).insert((
            mesh.clone(),
            ExtractedRaytracingMaterial(material.id().untyped()),
            *transform,
        ));
    }
}

/// Extracts raytracing instances with ExtendedMaterial<StandardMaterial, E>.
pub fn extract_raytracing_instances_extended<E: MaterialExtension>(
    instances: Extract<
        Query<(
            RenderEntity,
            &RaytracingMesh3d,
            &MeshMaterial3d<ExtendedMaterial<StandardMaterial, E>>,
            &GlobalTransform,
        )>,
    >,
    mut commands: Commands,
) {
    for (render_entity, mesh, material, transform) in &instances {
        commands.entity(render_entity).insert((
            mesh.clone(),
            ExtractedRaytracingMaterial(material.id().untyped()),
            *transform,
        ));
    }
}
