//! Bindings domain — the GPU bind groups that expose the cluster scene
//! to shaders, plus the shared WGSL libraries.
//!
//! - [`bind_groups`] — `ClusterSceneBindGroup`: the `@group(0)` the
//!   selector / sharing / PTLAS-fill compute passes read (cluster mesh
//!   pools + per-instance columns).
//! - [`binder`] — [`RaytracingSceneBindings`]: the `@group(0)` the
//!   path-tracer / realtime ray-trace shaders read (pools, columns,
//!   materials, textures, lights, the PTLAS).
//! - [`types`] — the public [`RaytracingMesh3d`] component.
//! - `extract` — `SolariMaterialAssets`, the extracted material set the
//!   binder builds its material array from.
//!
//! The WGSL shader libraries (`cluster_bindings`, `scene_bindings`,
//! `sampling`, `brdf`) live here too and are registered via
//! [`register_cluster_shaders`] / [`register_scene_shaders`].

use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    extract_resource::ExtractResourcePlugin, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::load_shader_library;

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
pub use binder::{prepare_raytracing_scene_bindings, RaytracingSceneBindings};
pub(crate) use binder::GPU_MATERIAL_SIZE;
#[cfg(test)]
pub(crate) use binder::MAX_TEXTURE_COUNT;
pub use extract::SolariMaterialAssets;
pub use fog_volume::{SolariFogVolume, SolariFogVolumes, SolariFogVolumesTablePlugin};
pub use portal::{SolariPortal, SolariPortals, SolariPortalsTablePlugin};
pub use types::RaytracingMesh3d;

/// Register the cluster scene-bind-group shader library (the `@group(0)`
/// accessors the compute passes import). Called by `ClusterPlugin`.
pub fn register_cluster_shaders(app: &mut App) {
    load_shader_library!(app, "cluster_bindings.wgsl");
    embedded_asset!(app, "cluster_bindings.wgsl");
}

/// Register the ray-trace scene-binding shader libraries (the consumer
/// `@group(0)` + shading helpers). Called by `RaytracingScenePlugin`.
pub fn register_scene_shaders(app: &mut App) {
    // Vendored pure pbr/utils helpers (`bevy_solari::pbr`) — loaded first since the
    // others import it. Lets the RT shaders run with `PbrPlugin` disabled (its
    // `bevy_pbr::{utils,lighting,pbr_functions}` libs would otherwise be missing).
    load_shader_library!(app, "pbr.wgsl");
    load_shader_library!(app, "brdf.wgsl");
    load_shader_library!(app, "raytracing_scene_bindings.wgsl");
    load_shader_library!(app, "sampling.wgsl");
}

/// Bindings domain plugin: the WGSL libraries, the cluster scene bind
/// group (`@group(0)` for the compute passes), and the raytracing scene
/// bind group (`@group(0)` for the ray-trace consumers).
pub struct BindingsPlugin;

impl Plugin for BindingsPlugin {
    fn build(&self, app: &mut App) {
        register_cluster_shaders(app);
        register_scene_shaders(app);
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
