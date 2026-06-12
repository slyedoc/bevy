//! Per-view ray-tracing cull mask — the RT analog of `RenderLayers`.
//!
//! Every PTLAS instance is written with an 8-bit `mask` derived from its
//! entity's [`RenderLayers`] (see [`render_layers_to_mask`], used by the
//! instance extract). Every ray is cast with a per-camera `cullMask`, also
//! derived from the camera's [`RenderLayers`]. The GPU tests `mask & cullMask`
//! during BVH traversal, so a camera only sees instances on a shared layer —
//! exactly [`RenderLayers::intersects`], restricted to the low 8 layers.
//!
//! The cull mask is per-view (different solari cameras may use different masks
//! in the same frame), so it can't live in the scene-global group(0) bindings.
//! It rides a small dynamic uniform bound in each pass's group(1); the shared
//! `trace_ray` reads it out of a `var<private>` each entry seeds.

use crate::render::atmosphere::SolariAtmosphereView;
use crate::render::SolariCamera;
use bevy_asset::Handle;
use bevy_camera::visibility::{RenderLayers, DEFAULT_LAYERS};
use bevy_camera::{Camera, ClearColor, ClearColorConfig};
use bevy_color::Color;
use bevy_core_pipeline::Skybox;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_image::Image;
use bevy_math::{UVec4, Vec3};
use bevy_render::{
    render_resource::{DynamicUniformBuffer, ShaderType},
    renderer::{RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    Extract, MainWorld,
};

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


/// Per-view uniform carrying the camera's RT cull mask + sky (skybox) brightness.
/// `cull_mask` is `vec4<u32>` (`.x` = mask) for an unambiguous 16-byte alignment;
/// `environment_brightness` follows. Bound in each pass's group(1); each tracing
/// entry seeds `view_cull_mask` from it, and the pathtracer scales the sky by the
/// brightness (`0.0` ⇒ no skybox ⇒ a ray miss stays black).
///
/// `environment_brightness` is the **raw cd/m²** (`Skybox::brightness`, not
/// pre-exposed): the pathtracer accumulates raw radiance and applies the camera
/// exposure once at the end, so the sky is exposed exactly like the rest of the
/// scene.
#[derive(Clone, Copy, ShaderType)]
pub struct SolariViewUniform {
    pub cull_mask: UVec4,
    /// Linear-RGB background for a **primary**-ray miss when there's no skybox
    /// (the camera's resolved `ClearColor`). Displayed as-is (the pathtracer
    /// divides by exposure so the end-of-loop `× exposure` cancels), then
    /// tonemapped — matching the raster framebuffer clear.
    pub clear_color: Vec3,
    pub environment_brightness: f32,
    /// `restir_debug` visualization mode (0 = none; see
    /// [`SolariDebugView::restir_debug_mode`](crate::render::view::SolariDebugView)).
    pub debug_mode: u32,
}

/// Render-world component: the solari view's resolved linear-RGB clear color (the
/// primary-miss background when no skybox). Inserted by
/// [`extract_solari_view_cull_masks`].
#[derive(Component, Clone, Copy)]
pub struct SolariViewClearColor(pub Vec3);

/// Render-world stash of the camera's skybox, from its [`Skybox`] component (image
/// + raw cd/m² brightness). The pathtracer samples it as the environment on a ray
/// miss. Inserted by [`extract_solari_skybox`], which also **removes** the
/// render-world `Skybox` so bevy's raster `SkyboxNode` doesn't also draw it (the
/// solari camera runs the Core3d graph). Absent ⇒ brightness 0 ⇒ misses stay black.
#[derive(Component, Clone)]
pub struct SolariEnvironmentMap {
    pub image: Handle<Image>,
    pub brightness: f32,
}

/// Render-world resource: the per-view [`SolariViewUniform`] buffer. Mirrors
/// bevy's `ViewUniforms`; written each frame by [`prepare_solari_view_uniforms`].
#[derive(Resource, Default)]
pub struct SolariViewUniforms {
    pub uniforms: DynamicUniformBuffer<SolariViewUniform>,
}

/// Render-world component: the extracted 8-bit cull mask for a solari view,
/// derived from its camera's `RenderLayers`. Inserted by
/// [`extract_solari_view_cull_masks`].
#[derive(Component, Clone, Copy)]
pub struct SolariViewCullMask(pub u32);

/// Render-world component: the dynamic-offset into [`SolariViewUniforms`] for a
/// view. Mirrors bevy's `ViewUniformOffset`; the render nodes bind group(1)
/// with it. Inserted by [`prepare_solari_view_uniforms`].
#[derive(Component, Clone, Copy)]
pub struct SolariViewOffset(pub u32);

/// `ExtractSchedule`: copy each solari camera's `RenderLayers` (→ 8-bit mask) and
/// its resolved [`ClearColor`] (→ [`SolariViewClearColor`], the no-skybox primary
/// miss background) onto its render-world view entity.
pub fn extract_solari_view_cull_masks(
    cameras: Extract<Query<(RenderEntity, Option<&RenderLayers>, &Camera), With<SolariCamera>>>,
    clear_color: Extract<Res<ClearColor>>,
    mut commands: Commands,
) {
    for (render_entity, layers, camera) in &cameras {
        let clear = match camera.clear_color {
            ClearColorConfig::Default => clear_color.0,
            ClearColorConfig::Custom(color) => color,
            // No framebuffer clear → treat a primary miss as black.
            ClearColorConfig::None => Color::BLACK,
        };
        let linear = clear.to_linear();
        commands.entity(render_entity).insert((
            SolariViewCullMask(render_layers_to_mask(layers)),
            SolariViewClearColor(Vec3::new(linear.red, linear.green, linear.blue)),
        ));
    }
}

/// `ExtractSchedule`: stash each solari camera's [`Skybox`] (image + brightness)
/// as [`SolariEnvironmentMap`] on its render entity, and **remove** the
/// render-world `Skybox` so bevy's raster `SkyboxNode` skips it — the pathtracer
/// draws the sky itself, at ray misses. Queries the main world directly (like
/// bevy's own skybox extract is insert-only and wouldn't see removals); a camera
/// with no skybox image gets `SolariEnvironmentMap` removed (miss stays black).
pub fn extract_solari_skybox(mut main_world: ResMut<MainWorld>, mut commands: Commands) {
    let mut cameras =
        main_world.query_filtered::<(RenderEntity, Option<&Skybox>), With<SolariCamera>>();
    for (render_entity, skybox) in cameras.iter(&main_world) {
        let Ok(mut entity) = commands.get_entity(render_entity) else {
            continue;
        };
        match skybox.and_then(|s| s.image.as_ref().map(|image| (image.clone(), s.brightness))) {
            Some((image, brightness)) => {
                entity.insert(SolariEnvironmentMap { image, brightness });
            }
            None => {
                entity.remove::<SolariEnvironmentMap>();
            }
        }
        entity.remove::<Skybox>();
    }
}

/// `Render` (`PrepareResources`): pack one [`SolariViewUniform`] per solari view
/// into the dynamic uniform buffer and record each view's offset.
pub fn prepare_solari_view_uniforms(
    mut uniforms: ResMut<SolariViewUniforms>,
    state: Res<crate::render::view::SolariViewState>,
    views: Query<
        (
            Entity,
            &SolariViewCullMask,
            &SolariViewClearColor,
            Option<&SolariEnvironmentMap>,
            Option<&SolariAtmosphereView>,
        ),
        With<SolariCamera>,
    >,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut commands: Commands,
) {
    let debug_mode = state
        .debug
        .and_then(|view| view.restir_debug_mode())
        .unwrap_or(0);
    uniforms.uniforms.clear();
    for (entity, mask, clear_color, environment_map, atmosphere_view) in &views {
        // The baked atmosphere cube already holds physical radiance (scaled by sun
        // illuminance), so it's used as-is (brightness 1.0). Otherwise the skybox's
        // raw cd/m² brightness, or 0.0 (no sky ⇒ miss stays black).
        let environment_brightness = if atmosphere_view.is_some() {
            1.0
        } else {
            environment_map.map_or(0.0, |env| env.brightness)
        };
        let offset = uniforms.uniforms.push(&SolariViewUniform {
            cull_mask: UVec4::new(mask.0, 0, 0, 0),
            clear_color: clear_color.0,
            environment_brightness,
            debug_mode,
        });
        commands.entity(entity).insert(SolariViewOffset(offset));
    }
    uniforms
        .uniforms
        .write_buffer(&render_device, &render_queue);
}
