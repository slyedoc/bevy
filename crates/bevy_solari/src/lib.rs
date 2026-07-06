#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]

//! Provides raytraced rendering.
//!
//! See [`SolariPlugins`] for more info.
//!
//! ![`bevy_solari` logo](https://raw.githubusercontent.com/bevyengine/bevy/refs/heads/main/assets/branding/bevy_solari.svg)
//!
//! ## Pipeline readiness
//!
//! Modeled on `bevy_pbr`'s meshlet split. Two crate-global resources, built once
//! at `RenderStartup`, are the single home for the two static halves of every pass:
//!
//! - [`SolariResourceManager`](resource_manager::SolariResourceManager) owns every
//!   pass's bind-group **layout** (mirrors meshlet's `ResourceManager`).
//! - [`SolariPipelines`](pipelines::SolariPipelines) owns every pass's compiled
//!   **pipeline id** (mirrors `MeshletPipelines`). It reads layouts from the manager
//!   to queue each pipeline; field names match the manager 1:1.
//!
//! Both gate on the cluster [`Allocator`](gpu::allocator::Allocator), so on an
//! unsupported device neither exists and every dispatch's
//! `run_if(resource_exists::<…>)` skips. A dispatch fetches its pipeline with a bare
//! `pipeline_cache.get_compute_pipeline(pipelines.<field>)` (bail-on-`None` for
//! compile readiness); a `prepare_*_bind_group` reads its layout from the manager.
//!
//! Each pass resource keeps only its **mutable** per-frame state (buffers, staging,
//! cold-start latches) — deliberately *not* centralized, so the prepare systems stay
//! parallelizable (meshlet likewise keeps mutable state in `InstanceManager` /
//! `MeshletViewResources`, not `ResourceManager`).
//!
//! The generic per-column **scatter** pipeline/layout is the one exception: created
//! by the generic [`GpuColumnPlugin`](ecs_gpu) per column type, it stays with that
//! infrastructure rather than being enumerated centrally.

extern crate alloc;

pub mod accel;
pub mod bindings;

pub mod ecs_gpu;
pub mod geometry;
pub mod gpu;
pub mod pipelines;
pub mod resource_manager;

pub mod hair;
pub mod instance;
pub mod lights;
pub mod material;
pub mod ray_query;
pub mod render;
pub mod transform;

#[cfg(feature = "bevy_solari_debug")]
pub mod debug;
#[cfg(feature = "cluster_processor")]
pub mod helper;

use bevy_app::{App, Plugin};
use bevy_ecs::schedule::{IntoScheduleConfigs, SystemSet};
use bevy_log::warn;
use bevy_render::settings::WgpuFeatures;
use bevy_render::{renderer::RenderDevice, Render, RenderApp, RenderStartup, RenderSystems};

use crate::accel::AccelPlugin;
use crate::bindings::BindingsPlugin;
use crate::ecs_gpu::{ReconcilePlugin, SceneColumnsPlugin};
use crate::geometry::GeometryPlugin;
use crate::hair::HairPlugin;
use crate::instance::InstancePlugin;
use crate::lights::SolariLightsPlugin;
use crate::material::StandardSolariMaterialPlugin;
use crate::ray_query::RayQueryPlugin;
use crate::render::SolarRenderPlugin;
use crate::transform::SolariTransformPlugin;

#[cfg(feature = "bevy_solari_debug")]
use crate::debug::SolariDebugPlugin;

pub use crate::gpu::rt_pipeline::{SolariAnyHitDef, SolariHitGroupDef, SolariHitGroupRegistry};
pub use crate::render::rt_pipeline::{GlassSurface, HairSurface, OpaqueSurface, PortalSurface};

/// Register an RT closest-hit; returns its SBT class for a material's `chit_class`.
/// Call during plugin `build`, after [`SolariPlugin`].
pub trait SolariChitRegistryAppExt {
    fn register_solari_chit(&mut self, def: SolariHitGroupDef) -> u32;
}

impl SolariChitRegistryAppExt for App {
    fn register_solari_chit(&mut self, def: SolariHitGroupDef) -> u32 {
        self.sub_app_mut(RenderApp)
            .world_mut()
            .resource_mut::<SolariHitGroupRegistry>()
            .register(def)
    }
}

/// A registrable ray-traced surface: add [`SolariMaterialPlugin<S>`], then set a
/// material's `chit_class` from [`SolariMaterialClass<S>`].
pub trait SolariMaterial: Send + Sync + 'static {
    fn hit_group() -> SolariHitGroupDef;
}

/// SBT class assigned to surface `S`; read to set a material's `chit_class`.
#[derive(bevy_ecs::resource::Resource)]
pub struct SolariMaterialClass<S: SolariMaterial> {
    class: u32,
    _marker: core::marker::PhantomData<fn() -> S>,
}

impl<S: SolariMaterial> SolariMaterialClass<S> {
    pub fn get(&self) -> u32 {
        self.class
    }
}

/// Registers surface `S` and exposes [`SolariMaterialClass<S>`]. Add after [`SolariPlugin`].
pub struct SolariMaterialPlugin<S: SolariMaterial>(core::marker::PhantomData<fn() -> S>);

impl<S: SolariMaterial> Default for SolariMaterialPlugin<S> {
    fn default() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<S: SolariMaterial> Plugin for SolariMaterialPlugin<S> {
    fn build(&self, app: &mut App) {
        let class = app.register_solari_chit(S::hit_group());
        app.insert_resource(SolariMaterialClass::<S> {
            class,
            _marker: core::marker::PhantomData,
        });
    }
}

/// `RenderStartup` ordering anchor: the foundational resources (raw-VK
/// allocator + extension fn tables, the cluster scene bind-group layout)
/// every other init reads. Resource inits that need them join
/// `.after(SolariSetup)`.
#[derive(SystemSet, Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct SolariSetup;

/// `Render`-schedule ordering for the cluster acceleration-structure
/// pipeline. The stages run in sequence each frame; systems join their
/// stage's set instead of hard-coding `.after`/`.before` on each other,
/// so the domain plugins order against named anchors without referencing
/// one another's systems.
#[derive(SystemSet, Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub enum SolariClusterSystems {
    /// GPU-scatter the per-instance + transform-table column deltas.
    Scatter,
    /// GPU transform propagation (Jacobi) — reads the scattered local/parent
    /// columns, writes the per-node world buffer.
    Propagate,
    /// Skin animated instances into the deform pool (LBS from the transform table).
    Deform,
    /// Classify instances → LOD bands; elect per-geometry dirty BLAS builds.
    Classify,
    /// Per-bucket object-space DAG cut → CLAS ref lists.
    Select,
    /// Build the per-geometry shared BLASes.
    BuildBlas,
    /// Instantiate per-instance animated CLAS → per-instance BLAS; repoint addresses.
    BuildAnimatedBlas,
    /// Incremental partitioned-TLAS fill + build.
    BuildTlas,
    /// Reusable batch ray-query trace against the built TLAS (no-op until a
    /// producer fills its ray buffer).
    RayQueries,
    /// Clear the per-frame instance deltas.
    Cleanup,
}

/// The solari prelude.
///
/// This includes the most common types in this crate, re-exported for your convenience.
pub mod prelude {
    pub use crate::{
        bindings::RaytracingMesh3d,
        bindings::SolariFogVolume,
        bindings::SolariPortal,
        geometry::ClusterMesh,
        hair::{Hair, HairAsset, HairMaterial, HairStrand, SolariBranches},
        lights::SolariDirectionLight,
        material::{SolariMaterial3d, StandardSolariMaterial},
        ray_query::picking::SolariPickingPlugin,
        accel::ClusterSelectorSettings,
        instance::SolariPartition,
        render::atmosphere::{SolariAtmosphere, SolariAtmosphereVolume, SolariGlobalFog},
        render::rt_pipeline::SolariAnyHitHeatmap,
        render::rt_pipeline::SolariClusterView,
        render::rt_pipeline::SolariCostHeatmap,
        render::rt_pipeline::SolariNormalFacing,
        render::rt_pipeline::SolariShowDisplacement,
        render::rt_pipeline::SolariTriangleView,
        render::CameraReframe,
        render::CameraReset,
        render::SolariCamera,
        transform::{NoGpuGlobalTransformReadback, SolariGpuFrame, TransformStatic},
        SolariInitPlugin, SolariPlugin,
    };

    #[cfg(feature = "dlss")]
    pub use crate::render::dlss::SolariDlssMode;

    #[cfg(feature = "cluster_processor")]
    pub use crate::geometry::from_mesh::*;

    #[cfg(feature = "cluster_processor")]
    pub use crate::helper::{
        convert_marked_meshes_to_raytracing, convert_meshes_to_raytracing,
        convert_standard_materials_to_solari, ConvertToRaytracing,
    };
}

/// Solari-side bootstrap plugin that must be added **before**
/// [`bevy_render::RenderPlugin`]. Registers Vulkan device-creation
/// callbacks for any solari feature that needs to enable extensions
/// or feature structs at adapter init — currently only the
/// `cluster_runtime` feature (NV cluster-AS / partitioned-AS).
///
/// Mirrors [`bevy_anti_alias::dlss::DlssInitPlugin`]'s role.
/// `DefaultPlugins` adds this plugin (gated on `bevy_solari`) right
/// before `RenderPlugin`, so apps using `DefaultPlugins` get it
/// automatically.
#[derive(Default)]
pub struct SolariInitPlugin;

impl Plugin for SolariInitPlugin {
    #[allow(unsafe_code)]
    fn build(&self, app: &mut App) {
        let mut settings = app
            .world_mut()
            .get_resource_or_init::<bevy_render::renderer::raw_vulkan_init::RawVulkanInitSettings>(
        );
        // SAFETY: callback only adds extensions + feature structs;
        // never removes anything.
        unsafe {
            gpu::extension::register_cluster_extension_callback(&mut settings);
        }
    }
}
pub struct SolariPlugin;

impl Plugin for SolariPlugin {
    fn build(&self, app: &mut App) {
        // Embed every solari shader (must run during plugin build) — co-located
        // with their pipeline builds in `crate::pipelines`.
        pipelines::embed_solari_shaders(app);
        app.add_plugins((
            SceneColumnsPlugin,
            ReconcilePlugin,
            BindingsPlugin,
            GeometryPlugin,
            InstancePlugin,
            AccelPlugin,
            RayQueryPlugin,
            SolarRenderPlugin,
            #[cfg(feature = "bevy_solari_debug")]
            SolariDebugPlugin,
            // ecs gpu tables
            SolariTransformPlugin,
            StandardSolariMaterialPlugin,
            SolariLightsPlugin,
            HairPlugin,
        ));

        // Built-in surfaces; order sets SBT class: opaque=0 (default), glass=1, hair=2, portal=3.
        app.add_plugins((
            SolariMaterialPlugin::<OpaqueSurface>::default(),
            SolariMaterialPlugin::<GlassSurface>::default(),
            SolariMaterialPlugin::<HairSurface>::default(),
            SolariMaterialPlugin::<PortalSurface>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<gpu::retire::GpuRetire>()
            .add_systems(Render, gpu::retire::reap_retired.in_set(RenderSystems::Cleanup))
            .add_systems(
                RenderStartup,
                (
                    gpu::allocator::init_allocator,
                    gpu::extension::init_cluster_extension_fns,
                )
                    .in_set(SolariSetup),
            )
            .add_systems(
                RenderStartup,
                // Build every pass layout into `SolariResourceManager` (gated on the
                // allocator, like the pass resources) before the pipelines read them.
                resource_manager::init_solari_resource_manager.after(SolariSetup),
            )
            .add_systems(
                RenderStartup,
                // After the layouts exist in `SolariResourceManager`.
                pipelines::init_solari_pipelines
                    .after(resource_manager::init_solari_resource_manager),
            );
    }

    fn finish(&self, app: &mut App) {
        let render_device = app.world().resource::<RenderDevice>();
        let features = render_device.features();
        if !features.contains(SolariPlugin::required_wgpu_features()) {
            warn!(
                "SolariPlugin: GPU lacks support for required features: {:?}.",
                SolariPlugin::required_wgpu_features().difference(features)
            );
            return;
        }

        // Bring up the DLSS Ray Reconstruction SDK (no-op / graceful when the feature
        // is off or RR is unsupported). The per-view context + RR dispatch are wired
        // by `SolarRenderPlugin`; this just creates the shared `SolariDlssSdk`.
        #[cfg(feature = "dlss")]
        render::dlss::init_dlss(app);
    }
}

impl SolariPlugin {
    /// [`WgpuFeatures`] required for these plugins to function.
    pub fn required_wgpu_features() -> WgpuFeatures {
        WgpuFeatures::EXPERIMENTAL_RAY_QUERY
            | WgpuFeatures::BUFFER_BINDING_ARRAY
            | WgpuFeatures::TEXTURE_BINDING_ARRAY
            | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | WgpuFeatures::PARTIALLY_BOUND_BINDING_ARRAY
            // 64-bit ints for buffer-device-address arithmetic in the RT-pipeline
            // bindless geometry path (`physical_load<T>(addr: u64)`).
            | WgpuFeatures::SHADER_INT64
            // Native f64 for the transform table: the propagate walk accumulates the
            // absolute world translation in f64 and the subtract pass relativizes it
            // against the camera origin (see `transform/`).
            | WgpuFeatures::SHADER_F64
    }
}
