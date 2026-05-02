mod node;
mod phase;
mod texture;

use bevy_app::{App, Plugin};
use bevy_camera::Camera3d;
use bevy_core_pipeline::{
    core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::{prelude::*, schedule::IntoScheduleConfigs};
use bevy_reflect::prelude::*;
use bevy_render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_phase::{sort_phase_system, AddRenderCommand, DrawFunctions, ViewSortedRenderPhases},
    ExtractSchedule, Render, RenderApp, RenderSystems,
};
use bevy_shader::load_shader_library;
pub use node::main_transmissive_pass_3d;
pub use phase::Transmissive3d;
pub use texture::ViewTransmissionTexture;

use texture::prepare_core_3d_transmission_textures;

use crate::{DrawMaterial, MeshPipelineKey};

/// Enables screen-space transmission for cameras.
pub struct ScreenSpaceTransmissionPlugin;

impl Plugin for ScreenSpaceTransmissionPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "transmission.wgsl");

        // NOTE: do not auto-require ScreenSpaceTransmission on every Camera3d.
        // The mesh-view bind group includes the SST texture/sampler bindings
        // whenever the component is *present* (`Has<ScreenSpaceTransmission>`),
        // but the opaque pipeline layout only includes them when the
        // `MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_RESERVED_BITS`
        // are set (i.e. when an actual transmissive material is being drawn).
        // Auto-requiring the component caused the bind-group / pipeline layouts
        // to disagree on bindings 25/26 the moment any opaque mesh was drawn
        // without an env map (see e.g. running `examples/3d/meshlet.rs` after
        // PR #22763). Treat SST as opt-in until upstream realigns the two
        // layout paths.
        app.add_plugins(ExtractComponentPlugin::<ScreenSpaceTransmission>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<DrawFunctions<Transmissive3d>>()
            .init_resource::<ViewSortedRenderPhases<Transmissive3d>>()
            .add_render_command::<Transmissive3d, DrawMaterial>()
            .add_systems(
                Render,
                sort_phase_system::<Transmissive3d>.in_set(RenderSystems::PhaseSort),
            )
            .add_systems(ExtractSchedule, phase::extract_transmissive_camera_phases)
            .add_systems(
                Render,
                prepare_core_3d_transmission_textures.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                main_transmissive_pass_3d
                    .after(main_opaque_pass_3d)
                    .before(main_transparent_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            );
    }
}

/// Configures transmission behavior, offering a trade-off between performance and visual fidelity.
#[derive(Component, Reflect, Clone, ExtractComponent)]
#[reflect(Component, Default, Clone)]
pub struct ScreenSpaceTransmission {
    /// How many individual steps should be performed in the `Transmissive3d` pass.
    ///
    /// Roughly corresponds to how many layers of transparency are rendered for screen space
    /// specular transmissive objects. Each step requires making one additional
    /// texture copy, so it's recommended to keep this number to a reasonably low value. Defaults to `1`.
    ///
    /// ### Notes
    ///
    /// - No copies will be performed if there are no transmissive materials currently being rendered,
    ///   regardless of this setting.
    /// - Setting this to `0` disables the screen-space refraction effect entirely, and falls
    ///   back to refracting only the environment map light's texture.
    /// - If set to more than `0`, any opaque [`clear_color`](bevy_camera::Camera::clear_color) will obscure the environment
    ///   map light's texture, preventing it from being visible through transmissive materials. If you'd like
    ///   to still have the environment map show up in your refractions, you can set the clear color's alpha to `0.0`.
    ///   Keep in mind that depending on the platform and your window settings, this may cause the window to become
    ///   transparent.
    pub steps: usize,
    /// The quality of the screen space specular transmission blur effect, applied to whatever's behind transmissive
    /// objects when their `roughness` is greater than `0.0`.
    ///
    /// Higher qualities are more GPU-intensive.
    ///
    /// **Note:** You can get better-looking results at any quality level by enabling TAA. See: `TemporalAntiAliasPlugin`
    pub quality: ScreenSpaceTransmissionQuality,
}

impl Default for ScreenSpaceTransmission {
    fn default() -> Self {
        Self {
            steps: 1,
            quality: Default::default(),
        }
    }
}

/// The quality of the screen space transmission blur effect, applied to whatever's behind transmissive
/// objects when their `roughness` is greater than `0.0`.
///
/// Higher qualities are more GPU-intensive.
///
/// **Note:** You can get better-looking results at any quality level by enabling TAA. See: `TemporalAntiAliasPlugin`
#[derive(Default, Clone, Copy, Reflect, PartialEq, PartialOrd, Debug)]
#[reflect(Default, Clone, Debug, PartialEq)]
pub enum ScreenSpaceTransmissionQuality {
    /// Best performance at the cost of quality. Suitable for lower end GPUs. (e.g. Mobile)
    ///
    /// `num_taps` = 4
    Low,

    /// A balanced option between quality and performance.
    ///
    /// `num_taps` = 8
    #[default]
    Medium,

    /// Better quality. Suitable for high end GPUs. (e.g. Desktop)
    ///
    /// `num_taps` = 16
    High,

    /// The highest quality, suitable for non-realtime rendering. (e.g. Pre-rendered cinematics and photo mode)
    ///
    /// `num_taps` = 32
    Ultra,
}

impl ScreenSpaceTransmissionQuality {
    pub const fn pipeline_key(self) -> MeshPipelineKey {
        match self {
            ScreenSpaceTransmissionQuality::Low => {
                MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_LOW
            }
            ScreenSpaceTransmissionQuality::Medium => {
                MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_MEDIUM
            }
            ScreenSpaceTransmissionQuality::High => {
                MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_HIGH
            }
            ScreenSpaceTransmissionQuality::Ultra => {
                MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_ULTRA
            }
        }
    }
}
