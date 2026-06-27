//! Geometry domain — the cluster-mesh asset and its GPU residency.
//!
//! Owns the on-disk [`ClusterMesh`] asset + loader/saver, the offline
//! `Mesh → ClusterMesh` bake (`from_mesh`, gated on `cluster_processor`),
//! the growable [`PersistentGpuBuffer`] backing store, the render-world
//! [`ClusterMeshManager`] that uploads asset data into the shared GPU
//! pools, and the per-cluster CLAS arena ([`ClasArena`]) built once at
//! upload.
//!
//! Everything here is keyed by `AssetId<ClusterMesh>` and lives for the
//! asset's lifetime — distinct from the per-frame instance / acceleration
//! -structure work in the sibling domains.

pub mod asset;
pub mod clas_arena;
#[cfg(feature = "cluster_processor")]
pub mod from_mesh;
pub mod indices;
pub mod mesh_manager;
pub mod tess_displace;
pub mod tess_template;
pub mod tessellation;


pub use self::asset::{
    write_cluster_mesh_sync, Cluster, ClusterBvhNode, ClusterLodGroup, ClusterMesh,
    ClusterMeshAabb, ClusterMeshLoader, ClusterMeshSaveOrLoadError, ClusterMeshSaver,
    CLUSTER_MESH_ASSET_VERSION,
};
pub use self::clas_arena::{init_clas_arena, upload_pending_clas, ClasArena};
pub use self::indices::{ClusterIndex, GroupIndex, GpuEntity, NodeIndex};
pub use self::mesh_manager::{
    init_cluster_mesh_manager, perform_pending_cluster_mesh_writes, ClusterMeshManager,
    ClusterMeshUpload, PendingClasUpload,
};
pub use crate::gpu::persistent_buffer::{PersistentGpuBuffer, PersistentGpuBufferable};

#[cfg(feature = "cluster_processor")]
pub use self::from_mesh::{
    MeshToClusterMeshConversionError, MAX_CLUSTER_TRIANGLES, MAX_CLUSTER_VERTICES, MAX_LOD_LEVELS,
    MERGE_ADDITIVE_FACTOR, MERGE_PREV_FACTOR, SIMPLIFY_TARGET_FRACTION, TARGET_GROUP_SIZE,
};

use bevy_app::{App, Plugin};
use bevy_asset::AssetApp;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{Render, RenderApp, RenderStartup, RenderSystems};

use crate::SolariSetup;

/// Geometry domain plugin: registers the [`ClusterMesh`] asset + loader
/// and the render-world systems that upload asset data into the shared
/// GPU pools and build the per-cluster CLAS.
pub struct GeometryPlugin;

impl Plugin for GeometryPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<ClusterMesh>()
            .init_asset_loader::<ClusterMeshLoader>()
            // `RaytracingMesh3d("path")` in `.bsn` resolves a `Handle<ClusterMesh>` from a string;
            // that needs the `ReflectHandle` + `String -> HandleTemplate<ClusterMesh>` machinery.
            .register_asset_reflect::<ClusterMesh>();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(
                RenderStartup,
                (
                    init_cluster_mesh_manager.after(SolariSetup),
                    init_clas_arena.after(SolariSetup),
                    tess_template::init_tessellation_templates.after(SolariSetup),
                    tess_displace::init_tess_displace.after(SolariSetup),
                    tess_displace::init_tess_normals.after(SolariSetup),
                    tess_displace::init_tess_ptlas_write.after(SolariSetup),
                ),
            )
            .add_systems(
                Render,
                (
                    perform_pending_cluster_mesh_writes.in_set(RenderSystems::PrepareAssets),
                    upload_pending_clas
                        .in_set(RenderSystems::PrepareAssets)
                        .after(perform_pending_cluster_mesh_writes),
                    tess_displace::tess_displace_selftest.in_set(RenderSystems::PrepareAssets),
                    tess_displace::prepare_tess_ptlas_write.in_set(RenderSystems::Prepare),
                    tess_displace::prepare_tess_ptlas_write_bind_group
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            );
    }
}
