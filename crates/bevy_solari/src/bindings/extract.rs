use bevy_asset::{AssetId, Assets};
use bevy_derive::Deref;
use bevy_ecs::resource::Resource;
use bevy_platform::collections::HashMap;
use bevy_render::extract_resource::ExtractResource;

use crate::material::StandardSolariMaterial;

/// Snapshot of `Assets<StandardSolariMaterial>` from the main world, extracted into the
/// render world each frame so the cluster-path scene binder can produce
/// per-material `GpuMaterial` entries without holding the main-world asset store
/// across schedules.
///
/// Per-entity extraction (transforms / `RaytracingMesh3d` handles) is
/// change-driven through [`crate::instance::InstanceManager`].
#[derive(Resource, Deref, Default)]
pub struct SolariMaterialAssets(HashMap<AssetId<StandardSolariMaterial>, StandardSolariMaterial>);

impl ExtractResource for SolariMaterialAssets {
    type Source = Assets<StandardSolariMaterial>;

    fn extract_resource(source: &Self::Source) -> Self {
        Self(
            source
                .iter()
                .map(|(asset_id, material)| (asset_id, material.clone()))
                .collect(),
        )
    }
}
