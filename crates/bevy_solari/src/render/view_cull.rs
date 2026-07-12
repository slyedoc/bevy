//! Per-view ray-tracing cull mask — the RT analog of `RenderLayers`.
//!
//! Every PTLAS instance is written with an 8-bit `mask` derived from its
//! entity's [`RenderLayers`] (see [`render_layers_to_mask`], used by the
//! instance extract). Every ray is cast with a per-camera `cullMask`, also
//! derived from the camera's [`RenderLayers`]. The GPU tests `mask & cullMask`
//! during BVH traversal, so a camera only sees instances on a shared layer —
//! exactly [`RenderLayers::intersects`], restricted to the low 8 layers.

use crate::render::sky::{SolariSky, SolariViewClearColor, SolariViewSkyShader};
use crate::render::SolariCamera;
use bevy_asset::Handle;
use bevy_camera::visibility::{RenderLayers, DEFAULT_LAYERS};
use bevy_camera::{Camera, ClearColor, ClearColorConfig};
use bevy_color::Color;
use bevy_core_pipeline::Skybox;
use bevy_ecs::{
    component::Component,
    query::With,
    system::{Commands, ResMut},
};
use bevy_image::Image;
use bevy_math::Vec3;
use bevy_render::{sync_world::RenderEntity, MainWorld};

/// Map an entity/camera's [`RenderLayers`] to the 8-bit RT mask: the low 64
/// layers' first byte (layers 0–7). Absent component → `DEFAULT_LAYERS`
/// (layer 0 → `0x01`), matching bevy's "no component ⇒ layer 0" rule, so a
/// default instance and a default camera intersect. An empty
/// [`RenderLayers::none`] maps to `0` → invisible to every camera, which is the
/// correct render-layers meaning.
///
/// **Hard cap of 8 layers** (the hardware `mask`/`cullMask` width). Layers ≥ 8
/// are truncated; scenes needing more would require multiple TLASes.
#[inline]
pub fn render_layers_to_mask(layers: Option<&RenderLayers>) -> u32 {
    let layers = layers.unwrap_or(DEFAULT_LAYERS);
    (layers.bits().first().copied().unwrap_or(0) & 0xFF) as u32
}

/// Render-world stash of the camera's `DepthOfField` lens parameters, for
/// the pathtracer's true thin-lens ray generation.
#[derive(Component, Clone, Copy)]
pub struct SolariViewLens {
    pub focal_distance: f32,
    pub aperture_f_stops: f32,
    pub sensor_height: f32,
}

/// Render-world stash of the camera's environment cube ([`SolariSky::Image`]:
/// image + raw cd/m² brightness). The miss shader samples it as the environment.
/// Absent ⇒ brightness 0 ⇒ misses show the clear color.
#[derive(Component, Clone)]
pub struct SolariEnvironmentMap {
    pub image: Handle<Image>,
    pub brightness: f32,
}

/// `ExtractSchedule`: resolve each solari camera's [`SolariSky`] into
/// [`SolariEnvironmentMap`] (image + brightness), the [`SolariViewSkyShader`]
/// marker, and the resolved [`SolariViewClearColor`]. Also **removes** any
/// render-world `Skybox` — solari ignores the raster component ([`SolariSky`] is
/// the sky API), and stripping it keeps bevy's `SkyboxNode` from compositing over
/// the trace. Queries the main world directly (an insert-only extract wouldn't
/// see removals).
pub fn extract_solari_sky(mut main_world: ResMut<MainWorld>, mut commands: Commands) {
    let default_clear = main_world
        .get_resource::<ClearColor>()
        .map_or(Color::BLACK, |c| c.0);
    let mut cameras = main_world
        .query_filtered::<(RenderEntity, Option<&SolariSky>, Option<&Camera>), With<SolariCamera>>();
    for (render_entity, sky, camera) in cameras.iter(&main_world) {
        let Ok(mut entity) = commands.get_entity(render_entity) else {
            continue;
        };
        let environment = match sky {
            Some(SolariSky::Image { image, brightness }) => Some((image.clone(), *brightness)),
            _ => None,
        };
        match environment {
            Some((image, brightness)) => {
                entity.insert(SolariEnvironmentMap { image, brightness });
            }
            None => {
                entity.remove::<SolariEnvironmentMap>();
            }
        }
        if matches!(sky, Some(SolariSky::Procedural | SolariSky::Shader(_))) {
            entity.insert(SolariViewSkyShader);
        } else {
            entity.remove::<SolariViewSkyShader>();
        }
        // The camera's resolved clear color, linear RGB — the miss shader's flat
        // background when no environment/sky module is active.
        let clear = match camera.map(|c| &c.clear_color) {
            Some(ClearColorConfig::Custom(color)) => *color,
            Some(ClearColorConfig::None) => Color::BLACK,
            _ => default_clear,
        }
        .to_linear();
        entity.insert(SolariViewClearColor(Vec3::new(
            clear.red,
            clear.green,
            clear.blue,
        )));
        entity.remove::<Skybox>();
    }
}
