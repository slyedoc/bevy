mod extract;
mod node;
mod prepare;

use crate::SolariPlugins;
use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_camera::Hdr;
use bevy_core_pipeline::{
    core_3d::main_opaque_pass_3d,
    prepass::{
        DeferredPrepass, DeferredPrepassDoubleBuffer, DepthPrepass, DepthPrepassDoubleBuffer,
        MotionVectorPrepass,
    },
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::{component::Component, reflect::ReflectComponent, schedule::IntoScheduleConfigs};
use bevy_pbr::DefaultOpaqueRendererMethod;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    renderer::RenderDevice, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::load_shader_library;
use extract::extract_solari_lighting;
use node::{init_solari_lighting_pipelines, solari_lighting};
use prepare::prepare_solari_lighting_resources;
use tracing::warn;

/// Raytraced direct and indirect lighting.
///
/// When using this plugin, it's highly recommended to set `shadow_maps_enabled: false` on all lights, as Solari replaces
/// traditional shadow mapping.
pub struct SolariLightingPlugin;

impl Plugin for SolariLightingPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "gbuffer_utils.wgsl");
        load_shader_library!(app, "realtime_bindings.wgsl");
        load_shader_library!(app, "presample_light_tiles.wgsl");
        embedded_asset!(app, "restir_di.wgsl");
        embedded_asset!(app, "restir_gi.wgsl");
        load_shader_library!(app, "specular_gi.wgsl");
        load_shader_library!(app, "world_cache_query.wgsl");
        embedded_asset!(app, "world_cache_compact.wgsl");
        embedded_asset!(app, "world_cache_update.wgsl");
        embedded_asset!(app, "debug_view.wgsl");

        load_shader_library!(app, "resolve_dlss_rr_textures.wgsl");

        app.register_type::<SolariDebugView>();

        app.insert_resource(DefaultOpaqueRendererMethod::deferred());
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        let render_device = render_app.world().resource::<RenderDevice>();
        let features = render_device.features();
        if !features.contains(SolariPlugins::required_wgpu_features()) {
            warn!(
                "SolariLightingPlugin not loaded. GPU lacks support for required features: {:?}.",
                SolariPlugins::required_wgpu_features().difference(features)
            );
            return;
        }

        render_app
            .add_systems(RenderStartup, init_solari_lighting_pipelines)
            .add_systems(ExtractSchedule, extract_solari_lighting)
            .add_systems(
                Render,
                prepare_solari_lighting_resources.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                solari_lighting
                    .before(main_opaque_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            );
    }
}

/// A component for a 3d camera entity to enable the Solari raytraced lighting system.
///
/// Must be used with `CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING)`, and
/// `Msaa::Off`.
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Default, Clone)]
#[require(
    Hdr,
    DeferredPrepass,
    DepthPrepass,
    MotionVectorPrepass,
    DeferredPrepassDoubleBuffer,
    DepthPrepassDoubleBuffer
)]
pub struct SolariLighting {
    /// Set to true to delete the saved temporal history (past frames).
    ///
    /// Useful for preventing ghosting when the history is no longer
    /// representative of the current frame, such as in sudden camera cuts.
    ///
    /// After setting this to true, it will automatically be toggled
    /// back to false at the end of the frame.
    pub reset: bool,

    /// When `Some`, an extra compute pass runs after the lighting pipeline
    /// and overwrites the camera output with a debug visualisation of the
    /// chosen channel (read from the deferred gbuffer + depth/motion
    /// prepass). Useful for inspecting the inputs that feed the lighting
    /// pass and for cross-renderer comparisons.
    ///
    /// `None` (the default) skips the pass and leaves the lit composite
    /// untouched.
    ///
    /// DLSS-RR will denoise whatever lands in the camera output, so debug
    /// channels look smeared with `Dlss<DlssRayReconstructionFeature>`
    /// attached. Detach DLSS while inspecting debug views.
    pub debug_view: Option<SolariDebugView>,
}

impl Default for SolariLighting {
    fn default() -> Self {
        Self {
            reset: true, // No temporal history on the first frame
            debug_view: None,
        }
    }
}

/// Selects which channel `SolariLighting::debug_view` overwrites the
/// camera output with. Discriminants must stay in sync with `debug_view.wgsl`.
#[repr(u32)]
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[reflect(Default, Clone)]
pub enum SolariDebugView {
    /// World-space shading normal, remapped from `[-1, 1]` to `[0, 1]`.
    WorldNormal = 1,
    /// `fract(world_position * 0.5)` -- 2 m bands along each axis.
    WorldPosition = 2,
    /// PBR base colour (linear).
    BaseColor = 3,
    /// PCG-hashed `gpixel.r` (packed base+roughness) -- approximates a
    /// per-material hue. The deferred gbuffer carries no explicit material
    /// slot, so identical materials hash to the same colour.
    Material = 4,
    /// Linear view-space depth, ramped from near (white) to 32 m (black).
    Depth = 5,
    /// `perceptual_roughness` as greyscale.
    Roughness = 6,
    /// `metallic` as greyscale.
    Metallic = 7,
    /// Raw emissive radiance (linear, can clip if intense -- pre-tonemap).
    Emissive = 8,
    /// Per-pixel screen-space motion (`prev_uv - curr_uv`), biased around
    /// mid-grey, scaled 50x so per-frame motion is legible.
    MotionVector = 9,
}

impl Default for SolariDebugView {
    fn default() -> Self {
        Self::WorldNormal
    }
}
