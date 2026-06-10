use bevy_asset::{AssetId, Assets};
use bevy_derive::Deref;
use bevy_ecs::resource::Resource;
use bevy_platform::collections::HashMap;
use bevy_render::extract_resource::ExtractResource;

use crate::material::SolariMaterial;

/// Snapshot of `Assets<SolariMaterial>` from the main world, extracted into the
/// render world each frame so the cluster-path scene binder can produce
/// per-material `GpuMaterial` entries without holding the main-world asset store
/// across schedules.
///
/// Per-entity extraction (transforms / `RaytracingMesh3d` handles) happens via
/// [`crate::instance::extract_cluster_instances`], which drives `InstanceManager`
/// directly — no need for a separate `extract_raytracing_scene` system.
#[derive(Resource, Deref, Default)]
pub struct SolariMaterialAssets(HashMap<AssetId<SolariMaterial>, SolariMaterial>);

impl ExtractResource for SolariMaterialAssets {
    type Source = Assets<SolariMaterial>;

    fn extract_resource(source: &Self::Source) -> Self {
        Self(
            source
                .iter()
                .map(|(asset_id, material)| (asset_id, material.clone()))
                .collect(),
        )
    }
}
