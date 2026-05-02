//! Aurora's geometry pipeline: `MeshletMesh` → CLAS → BLAS → TLAS.
//!
//! All three layers are built through the cluster-AS path on the bottom two
//! (`VK_NV_cluster_acceleration_structure`) and the standard KHR path for the
//! TLAS (`VK_KHR_acceleration_structure`). Aurora bypasses wgpu's `Blas` /
//! `Tlas` abstractions because:
//!
//! - Cluster ASes don't fit the wgpu types — they're produced by indirect
//!   builds that write opaque payloads into a single big GPU buffer, with
//!   per-CLAS identity captured only by `VkDeviceAddress`.
//! - wgpu's `Blas` carries instance-tracking metadata Aurora doesn't need.
//!
//! The actual `VkBuffer` allocations and `VkAccelerationStructureKHR` handles
//! are owned by the [`ClusterAsManager`] and [`TlasManager`] resources and
//! freed on render-app teardown.

pub mod blas;
pub mod cluster_as;
pub mod meshlet_loader;
pub mod raw_vk;
pub mod test_triangle_blas;
pub mod tlas;

use core::ops::Deref;

use bevy_app::{App, Plugin};
use bevy_asset::{Assets, Handle};
use bevy_ecs::{
    component::Component,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res, ResMut},
};
use bevy_pbr::experimental::meshlet::MeshletMesh;
use bevy_reflect::Reflect;
use bevy_render::{
    render_resource::WgpuFeatures,
    renderer::{RenderDevice, RenderQueue},
    Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_transform::components::GlobalTransform;

pub use cluster_as::{ClusterAsManager, MeshClusterHandle, MeshClusters};
pub use meshlet_loader::{
    dequantize_meshlet_into, dequantize_meshlet_mesh, DequantizedMeshlet, DequantizedMeshletMesh,
};
pub use tlas::{TlasInstance, TlasManager};

/// Component to attach to an entity that should be rendered through Aurora's
/// pure-RT pipeline. Replaces `MeshletMesh3d` for Aurora-rendered entities;
/// no `Mesh3d` is involved.
#[derive(Component, Reflect, Clone, Debug, Default)]
pub struct AuroraMeshlet3d(pub Handle<MeshletMesh>);

/// Render-world mirror of a main-world `(AuroraMeshlet3d, GlobalTransform)`
/// pair, populated by the extract schedule.
#[derive(Component, Debug, Clone)]
pub struct ExtractedAuroraInstance {
    pub mesh: Handle<MeshletMesh>,
    pub world_from_local: bevy_math::Mat4,
}

/// Map from `Handle<MeshletMesh>` to the [`MeshClusterHandle`] returned by the
/// last successful `ClusterAsManager::upload_mesh`. Sits in the render world.
#[derive(bevy_ecs::resource::Resource, Default, Debug)]
pub struct UploadedMeshes(pub bevy_platform::collections::HashMap<Handle<MeshletMesh>, MeshClusterHandle>);

/// Dequantized meshes captured during the Extract schedule (which has main-
/// world access to `Assets<MeshletMesh>`) but not yet built into CLAS+BLAS.
/// `process_uploads` drains this resource each render-world frame.
#[derive(bevy_ecs::resource::Resource, Default)]
pub struct PendingUploads(pub bevy_platform::collections::HashMap<Handle<MeshletMesh>, DequantizedMeshletMesh>);

/// Builds + maintains the `cluster_AS` / BLAS / TLAS pipeline backing Aurora's
/// ray-traced rendering.
pub struct AuroraScenePlugin;

impl Plugin for AuroraScenePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<AuroraMeshlet3d>();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<UploadedMeshes>()
            .init_resource::<PendingUploads>()
            .add_systems(RenderStartup, init_managers)
            .add_systems(ExtractSchedule, extract_aurora_instances)
            .add_systems(
                Render,
                (
                    process_uploads.in_set(RenderSystems::PrepareResources),
                    rebuild_tlas
                        .in_set(RenderSystems::PrepareResources)
                        .after(process_uploads),
                ),
            );
    }
}

/// Allocate the `ClusterAsManager` + `TlasManager` resources. Runs once at
/// render-startup; uses `as_hal` to reach the underlying ash device.
fn init_managers(
    mut commands: Commands,
    device: Res<RenderDevice>,
) {
    if !device
        .features()
        .contains(WgpuFeatures::EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE)
    {
        tracing::warn!(
            target: "bevy_aurora",
            "EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE not enabled -- AuroraScenePlugin inactive"
        );
        return;
    }

    // SAFETY: `as_hal::<Vulkan>()` returns a `vulkan::Device` only when the
    // wgpu device was created on the Vulkan backend. We hold the borrow only
    // for the synchronous allocation call.
    unsafe {
        let wgpu_device = device.wgpu_device();
        let Some(hal_device) = wgpu_device.as_hal::<wgpu::hal::api::Vulkan>() else {
            tracing::warn!(
                target: "bevy_aurora",
                "device is not Vulkan -- AuroraScenePlugin inactive",
            );
            return;
        };
        let hal_device: &wgpu::hal::vulkan::Device = hal_device.deref();
        let raw_device = hal_device.raw_device();
        let raw_instance = hal_device.shared_instance().raw_instance();
        let raw_phys = hal_device.raw_physical_device();
        let mem_props = raw_instance.get_physical_device_memory_properties(raw_phys);

        let manager = ClusterAsManager::new(
            raw_device,
            &mem_props,
            cluster_as::DEFAULT_CLAS_STORAGE_BYTES,
            cluster_as::DEFAULT_BLAS_STORAGE_BYTES,
        );
        commands.insert_resource(manager);
        commands.insert_resource(TlasManager::default());
    }
}

/// Extract `(AuroraMeshlet3d, GlobalTransform)` from main world to render world
/// as `ExtractedAuroraInstance`s, and (one-shot per handle) dequantize the
/// underlying `MeshletMesh` asset into [`PendingUploads`] so the render-world
/// upload system can build CLAS+BLAS for it.
///
/// `Assets<MeshletMesh>` is main-world-only -- this is the bridge.
fn extract_aurora_instances(
    mut commands: Commands,
    main_world_q: Extract<Query<(&AuroraMeshlet3d, &GlobalTransform)>>,
    main_meshes: Extract<Res<Assets<MeshletMesh>>>,
    uploaded: Res<UploadedMeshes>,
    mut pending: ResMut<PendingUploads>,
) {
    for (am, xform) in &main_world_q {
        commands.spawn(ExtractedAuroraInstance {
            mesh: am.0.clone(),
            world_from_local: xform.to_matrix(),
        });

        // Queue a one-shot dequantize for this mesh if it's loaded and we
        // haven't already uploaded or queued it.
        if uploaded.0.contains_key(&am.0) || pending.0.contains_key(&am.0) {
            continue;
        }
        let Some(asset) = main_meshes.get(&am.0) else {
            continue;
        };
        let dequantized = dequantize_meshlet_mesh(asset);
        tracing::info!(
            target: "bevy_aurora",
            meshlets = dequantized.meshlets.len(),
            verts = dequantized.positions.len(),
            indices = dequantized.indices.len(),
            "dequantized meshlet mesh; queued for upload",
        );
        pending.0.insert(am.0.clone(), dequantized);
    }
}

/// Drain pending uploads through `ClusterAsManager`. For every extracted
/// instance whose mesh hasn't been uploaded yet, dequantize + build CLAS+BLAS
/// and record the mapping in [`UploadedMeshes`].
/// Drain [`PendingUploads`] (filled in Extract by [`extract_aurora_instances`])
/// through `ClusterAsManager` and record the resulting handles in
/// [`UploadedMeshes`].
fn process_uploads(
    manager: Option<ResMut<ClusterAsManager>>,
    mut uploaded: ResMut<UploadedMeshes>,
    mut pending: ResMut<PendingUploads>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let Some(mut manager) = manager else {
        return;
    };
    if pending.0.is_empty() {
        return;
    }

    // Drain into a local Vec so we don't hold the ResMut borrow across the
    // upload calls.
    let mut drained: Vec<(Handle<MeshletMesh>, DequantizedMeshletMesh)> =
        pending.0.drain().collect();
    drained.sort_by_key(|(h, _)| h.id());

    // SAFETY: same pattern as `init_managers` -- as_hal escape held for the
    // synchronous upload calls only.
    unsafe {
        let wgpu_device = device.wgpu_device();
        let Some(hal_device) = wgpu_device.as_hal::<wgpu::hal::api::Vulkan>() else {
            return;
        };
        let hal_device: &wgpu::hal::vulkan::Device = hal_device.deref();
        let raw_device = hal_device.raw_device();
        let raw_instance = hal_device.shared_instance().raw_instance();
        let raw_phys = hal_device.raw_physical_device();
        let mem_props = raw_instance.get_physical_device_memory_properties(raw_phys);

        for (handle, dequantized) in drained {
            tracing::info!(
                target: "bevy_aurora",
                meshlets = dequantized.meshlets.len(),
                "uploading meshlet mesh",
            );
            let cluster_handle = manager.upload_mesh(
                raw_device,
                &mem_props,
                hal_device,
                wgpu_device,
                queue.0.as_ref(),
                &dequantized,
            );
            uploaded.0.insert(handle, cluster_handle);
        }
    }
}

/// Build the TLAS once the extracted instance set + uploaded BLASes are
/// available. Idempotent: [`TlasManager::build`] returns immediately on
/// subsequent calls (M-B sub-2 scope is single-build; M-D adds rebuilds).
///
/// Diagnostic: if the env var `AURORA_TEST_TRIANGLE_BLAS=1` is set, the TLAS
/// is built with a single regular KHR-AS triangle BLAS instead of any
/// Aurora cluster-built BLAS. This is the M-B sub-2c control test --
/// confirms whether the wrap+bind+ray-query path works at all (rays hit
/// the triangle), or whether something is broken regardless of BLAS source.
fn rebuild_tlas(
    instances: Query<&ExtractedAuroraInstance>,
    manager: Option<Res<ClusterAsManager>>,
    uploaded: Res<UploadedMeshes>,
    tlas: Option<ResMut<TlasManager>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut test_blas: bevy_ecs::system::Local<Option<test_triangle_blas::TestTriangleBlas>>,
) {
    let Some(mut tlas) = tlas else {
        return;
    };
    if tlas.is_built() {
        return;
    }

    let triangle_diag = std::env::var("AURORA_TEST_TRIANGLE_BLAS")
        .ok()
        .is_some_and(|v| v != "0" && !v.is_empty());

    if triangle_diag {
        // Build the triangle BLAS once and cache it in this system's Local.
        if test_blas.is_none() {
            unsafe {
                let wgpu_device = device.wgpu_device();
                let Some(hal_device) = wgpu_device.as_hal::<wgpu::hal::api::Vulkan>() else {
                    return;
                };
                let hal_device: &wgpu::hal::vulkan::Device = hal_device.deref();
                let raw_device = hal_device.raw_device();
                let raw_instance = hal_device.shared_instance().raw_instance();
                let raw_phys = hal_device.raw_physical_device();
                let mem_props = raw_instance.get_physical_device_memory_properties(raw_phys);
                *test_blas = Some(test_triangle_blas::build(
                    raw_device,
                    &mem_props,
                    raw_instance,
                    wgpu_device,
                    queue.0.as_ref(),
                ));
            }
        }
        let triangle = test_blas.as_ref().unwrap();
        let inst = TlasInstance::opaque(bevy_math::Mat4::IDENTITY, triangle.address);
        unsafe {
            tlas.build(device.wgpu_device(), queue.0.as_ref(), &[inst]);
        }
        return;
    }

    let Some(manager) = manager else {
        return;
    };
    let mut tlas_instances = Vec::new();
    for inst in &instances {
        let Some(handle) = uploaded.0.get(&inst.mesh) else {
            continue;
        };
        let mc = manager.mesh(*handle);
        if mc.blas_address == 0 {
            continue;
        }
        tlas_instances.push(TlasInstance::opaque(inst.world_from_local, mc.blas_address));
    }
    if tlas_instances.is_empty() {
        return;
    }

    // SAFETY: blas_addresses come from ClusterAsManager which retains the
    // backing storage; init_managers + this system run on the same render
    // thread, so no concurrent GPU work races our `vkDeviceWaitIdle`.
    unsafe {
        tlas.build(device.wgpu_device(), queue.0.as_ref(), &tlas_instances);
    }
}

