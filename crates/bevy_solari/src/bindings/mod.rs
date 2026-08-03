//! Bindings domain — the GPU bind groups that expose the cluster scene
//! to shaders, plus the shared WGSL libraries.
//!
//! - [`bind_groups`] — `ClusterSceneBindGroup`: the cluster scene set 0
//!   (cluster mesh pools + per-instance columns) the selector / sharing /
//!   PTLAS-fill heap kernels read via its descriptor-heap mirror
//!   (`cluster_bindings.slang`).
//! - [`binder`] — [`RaytracingSceneBindings`]: the `@group(0)` the
//!   path-tracer / realtime ray-trace shaders read (pools, columns,
//!   materials, textures, lights, the PTLAS).
//! - [`types`] — the public [`RaytracingMesh3d`] component.
//! - `extract` — `SolariMaterialAssets`, the extracted material set the
//!   binder builds its material array from.
//!
//! The WGSL shader libraries (`scene_bindings`, `sampling`, `brdf`) live
//! here too;
//! `cluster_bindings.slang` is a Slang module compiled from source by the
//! AS heap kernels.

use bevy_app::{App, Plugin};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    extract_resource::ExtractResourcePlugin, Render, RenderApp, RenderStartup, RenderSystems,
};

use crate::SolariSetup;

mod bind_groups;
mod binder;
mod extract;
mod fog_volume;
pub mod portal;
mod types;

pub use bind_groups::{
    init_cluster_scene_bind_group_layout, prepare_cluster_scene_bind_group, ClusterSceneBindGroup,
    ClusterSceneBindGroupLayout,
};
pub use binder::{prepare_raytracing_scene_bindings, RaytracingSceneBindings, SceneHeapSlots};
pub(crate) use binder::{write_image_descriptor, write_sampler_descriptor, GPU_MATERIAL_SIZE};
pub use extract::SolariMaterialAssets;
pub use fog_volume::{SolariFogVolume, SolariFogVolumes, SolariFogVolumesTablePlugin};
pub use portal::{SolariPortal, SolariPortals, SolariPortalsTablePlugin};
pub use types::RaytracingMesh3d;

/// The shared `octahedral` Slang module (normal codec + snorm16 packing) as a
/// compile-call `modules` list — imported by the deform + tessellation
/// kernels (and referenced by the compile tests).
pub(crate) const OCTAHEDRAL_MODULES: &[(&str, &str)] =
    &[("octahedral", include_str!("octahedral.slang"))];

/// Bindings domain plugin: the WGSL libraries, the cluster scene bind
/// group (`@group(0)` for the compute passes), and the raytracing scene
/// bind group (`@group(0)` for the ray-trace consumers).
pub struct BindingsPlugin;

impl Plugin for BindingsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<SolariMaterialAssets>::default());
        app.register_type::<SolariFogVolume>();
        app.register_type::<SolariPortal>();
        // Fog-volume + portal gpu_table!s (slot allocator, column scatter,
        // change-driven extract; their columns join the scene-columns group).
        app.add_plugins((SolariFogVolumesTablePlugin, SolariPortalsTablePlugin));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ClusterSceneBindGroup>()
            .insert_resource(RaytracingSceneBindings::new())
            .add_systems(
                RenderStartup,
                (
                    init_cluster_scene_bind_group_layout.in_set(SolariSetup),
                    binder::init_solari_scene_buffers.after(SolariSetup),
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_cluster_scene_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    prepare_raytracing_scene_bindings
                        .in_set(RenderSystems::PrepareBindGroups)
                        .after(prepare_cluster_scene_bind_group),
                ),
            );
    }
}
