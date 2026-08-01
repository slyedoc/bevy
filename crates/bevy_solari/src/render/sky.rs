//! Unified per-camera sky.
//!
//! [`SolariSky`] is one component covering every way a ray miss can be shaded:
//! the camera's clear color, an environment cubemap, the built-in procedural
//! gradient, or a user Slang module. Solari ignores the raster `Skybox`
//! component entirely.
//!
//! The `Procedural`/`Shader` modes compose a `custom_sky` Slang module into the
//! primary-miss stage (compiled at pipeline build via `gpu/slang.rs`). The
//! module source lives in [`SolariCustomSky`]; swapping it (mutating a
//! [`SolariSky::Shader`] component) bumps the generation, and the RT dispatch
//! rebuilds the pipeline — the compiled SPIR-V is baked into the `VkPipeline`.

use alloc::borrow::Cow;
use bevy_asset::Handle;
use bevy_ecs::{
    component::Component,
    reflect::ReflectComponent,
    resource::Resource,
    system::{Local, Query, ResMut},
};
use bevy_image::Image;
use bevy_log::warn;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::Extract;

/// The default `custom_sky` module: the built-in procedural gradient.
pub const DEFAULT_CUSTOM_SKY: &str = include_str!("rt_pipeline/custom_sky.slang");

/// How rays that miss all geometry are shaded (the background and environment
/// lighting) for a [`SolariCamera`](super::SolariCamera).
///
/// Optional: a camera without it shows the clear color on miss. The raster
/// `Skybox` component is ignored (and stripped) on solari cameras.
///
/// An active [`SolariAtmosphere`](super::atmosphere::SolariAtmosphere) takes
/// precedence over all of these — it owns the sky it bakes.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Default, Clone)]
pub enum SolariSky {
    /// The camera's clear color, in physical light units like all traced radiance.
    #[default]
    ClearColor,
    /// An environment cubemap, sampled on miss (`brightness` in cd/m²).
    Image {
        image: Handle<Image>,
        brightness: f32,
    },
    /// The built-in procedural gradient sky.
    Procedural,
    /// A custom sky shader, replacing the built-in procedural module.
    ///
    /// The source is a self-contained Slang module defining
    /// `public float3 sample_custom_sky(float3 ray_direction)`, returning
    /// radiance in physical light units (cd/m²). It composes into the
    /// primary-miss stage only, so keep it free of scene-binding imports.
    /// The module is app-global: one custom sky shader at a time. Mutating
    /// the source hot-swaps the sky (the RT pipeline rebuilds).
    Shader(Cow<'static, str>),
}

/// Render-world view marker: this camera's misses evaluate the composed
/// `custom_sky` module ([`SolariSky::Procedural`] / [`SolariSky::Shader`]).
/// The dispatch encodes it as a negative sky brightness in `RtCamera.sky.x`.
#[derive(Component, Clone, Copy)]
pub struct SolariViewSkyShader;

/// Render-world view component: the camera's resolved clear color (linear RGB) —
/// the miss shader's flat background when no environment/sky module is active.
/// Carried in `RtCamera.sky.yzw`.
#[derive(Component, Clone, Copy)]
pub struct SolariViewClearColor(pub bevy_math::Vec3);

/// Render-world resource: the live `custom_sky` Slang module source the RT
/// pipeline compiles into the primary miss shader. `generation` changes with
/// the source; [`RtPipeline`](crate::gpu::rt_pipeline::RtPipeline) bakes the
/// generation it was built with, and the dispatch rebuilds on mismatch.
#[derive(Resource)]
pub struct SolariCustomSky {
    pub source: Cow<'static, str>,
    pub generation: u64,
}

impl Default for SolariCustomSky {
    fn default() -> Self {
        Self {
            source: Cow::Borrowed(DEFAULT_CUSTOM_SKY),
            generation: 0,
        }
    }
}

/// `ExtractSchedule`: mirror the first [`SolariSky::Shader`] camera's module
/// source into [`SolariCustomSky`], restoring the built-in gradient when no
/// camera uses one. Compares sources, so mutating the component re-installs it.
pub fn extract_solari_custom_sky(
    skies: Extract<Query<&SolariSky>>,
    mut custom_sky: ResMut<SolariCustomSky>,
    mut warned: Local<bool>,
) {
    let user_shader = skies.iter().find_map(|sky| match sky {
        SolariSky::Shader(source) => Some(source),
        _ => None,
    });

    match user_shader {
        Some(source) => {
            if !source.contains("sample_custom_sky") {
                if !*warned {
                    warn!(
                        "Custom sky module must define \
                         `public float3 sample_custom_sky(float3 ray_direction)`."
                    );
                    *warned = true;
                }
                return;
            }
            if custom_sky.source != *source {
                custom_sky.source = Cow::Owned(source.clone().into_owned());
                custom_sky.generation += 1;
                *warned = false;
            }
        }
        None => {
            if custom_sky.source != DEFAULT_CUSTOM_SKY {
                custom_sky.source = Cow::Borrowed(DEFAULT_CUSTOM_SKY);
                custom_sky.generation += 1;
                *warned = false;
            }
        }
    }
}
