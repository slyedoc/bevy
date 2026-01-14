mod binder;
mod blas;
mod extract;
mod types;

use bevy_shader::load_shader_library;
pub use binder::RaytracingSceneBindings;
pub use types::RaytracingMesh3d;
// SolariMaterialApp is defined at the bottom of this file

use crate::SolariPlugins;
use bevy_app::{App, Plugin};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_pbr::MaterialExtension;
use bevy_render::{
    mesh::{
        allocator::{allocate_and_free_meshes, MeshAllocator},
        RenderMesh,
    },
    render_asset::prepare_assets,
    render_resource::BufferUsages,
    renderer::RenderDevice,
    ExtractSchedule, Render, RenderApp, RenderSystems,
};
use binder::prepare_raytracing_scene_bindings;
use blas::{compact_raytracing_blas, prepare_raytracing_blas, BlasManager};
use extract::{
    clear_raytracing_materials, extract_extended_materials, extract_raytracing_instances_extended,
    extract_raytracing_instances_standard, extract_standard_materials,
    RaytracingMaterialAssets, RaytracingMaterialExtractionSystems,
};
use tracing::warn;

/// Creates acceleration structures and binding arrays of resources for raytracing.
pub struct RaytracingScenePlugin;

impl Plugin for RaytracingScenePlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "brdf.wgsl");
        load_shader_library!(app, "raytracing_scene_bindings.wgsl");
        load_shader_library!(app, "sampling.wgsl");
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        let render_device = render_app.world().resource::<RenderDevice>();
        let features = render_device.features();
        if !features.contains(SolariPlugins::required_wgpu_features()) {
            warn!(
                "RaytracingScenePlugin not loaded. GPU lacks support for required features: {:?}.",
                SolariPlugins::required_wgpu_features().difference(features)
            );
            return;
        }

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .world_mut()
            .resource_mut::<MeshAllocator>()
            .extra_buffer_usages |= BufferUsages::BLAS_INPUT | BufferUsages::STORAGE;

        render_app
            .init_resource::<BlasManager>()
            .init_resource::<RaytracingMaterialAssets>()
            .insert_resource(RaytracingSceneBindings::new())
            // Configure system set ordering: Clear must run before Extract
            .configure_sets(
                ExtractSchedule,
                RaytracingMaterialExtractionSystems::Clear
                    .before(RaytracingMaterialExtractionSystems::Extract),
            )
            .add_systems(
                ExtractSchedule,
                clear_raytracing_materials.in_set(RaytracingMaterialExtractionSystems::Clear),
            )
            .add_systems(
                ExtractSchedule,
                (
                    extract_standard_materials,
                    extract_raytracing_instances_standard,
                )
                    .in_set(RaytracingMaterialExtractionSystems::Extract),
            )
            .add_systems(
                Render,
                (
                    prepare_raytracing_blas
                        .in_set(RenderSystems::PrepareAssets)
                        .before(prepare_assets::<RenderMesh>)
                        .after(allocate_and_free_meshes),
                    compact_raytracing_blas
                        .in_set(RenderSystems::PrepareAssets)
                        .after(prepare_raytracing_blas),
                    prepare_raytracing_scene_bindings.in_set(RenderSystems::PrepareBindGroups),
                ),
            );
    }
}

/// Extension trait for registering ExtendedMaterial types for Solari raytracing.
///
/// # Example
///
/// ```ignore
/// use bevy_solari::scene::SolariMaterialApp;
///
/// app.add_solari_material::<MyMaterialExtension>();
/// ```
pub trait SolariMaterialApp {
    /// Register an ExtendedMaterial<StandardMaterial, E> for Solari raytracing.
    ///
    /// This extracts the base StandardMaterial properties from the ExtendedMaterial
    /// for use in raytracing calculations (global illumination, reflections, etc.).
    /// The custom shader logic in the extension is not evaluated during raytracing.
    fn add_solari_material<E: MaterialExtension>(&mut self) -> &mut Self;
}

impl SolariMaterialApp for App {
    fn add_solari_material<E: MaterialExtension>(&mut self) -> &mut Self {
        let render_app = self.sub_app_mut(RenderApp);
        render_app.add_systems(
            ExtractSchedule,
            (
                extract_extended_materials::<E>,
                extract_raytracing_instances_extended::<E>,
            )
                .in_set(RaytracingMaterialExtractionSystems::Extract),
        );
        self
    }
}

/// A convenience plugin that registers an ExtendedMaterial for both rasterization and Solari raytracing.
///
/// This combines [`MaterialPlugin`] and [`SolariMaterialApp::add_solari_material`] into a single plugin,
/// so you don't need to register the material twice.
///
/// # Example
///
/// ```ignore
/// use bevy_solari::prelude::*;
///
/// // Instead of:
/// //   app.add_plugins(MaterialPlugin::<MyExtendedMaterial>::default());
/// //   app.add_solari_material::<MyExtension>();
///
/// // Just use:
/// app.add_plugins(SolariMaterialPlugin::<MyExtension>::default());
/// ```
pub struct SolariMaterialPlugin<E: MaterialExtension> {
    _marker: core::marker::PhantomData<E>,
}

impl<E: MaterialExtension> Default for SolariMaterialPlugin<E> {
    fn default() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<E: MaterialExtension> Plugin for SolariMaterialPlugin<E>
where
    bevy_pbr::ExtendedMaterial<bevy_pbr::StandardMaterial, E>: bevy_pbr::Material,
    <bevy_pbr::ExtendedMaterial<bevy_pbr::StandardMaterial, E> as bevy_render::render_resource::AsBindGroup>::Data:
        PartialEq + Eq + core::hash::Hash + Clone,
{
    fn build(&self, app: &mut App) {
        // Register for rasterization
        app.add_plugins(bevy_pbr::MaterialPlugin::<
            bevy_pbr::ExtendedMaterial<bevy_pbr::StandardMaterial, E>,
        >::default());
    }

    fn finish(&self, app: &mut App) {
        // Register for Solari raytracing (must be done in finish after RenderApp exists)
        app.add_solari_material::<E>();
    }
}
