use bevy_render::{
    render_resource::{
        binding_types::{sampler, texture_cube, texture_storage_2d, uniform_buffer},
        BindGroupLayoutDescriptor, BindGroupLayoutEntries, SamplerBindingType, ShaderStages,
        StorageTextureAccess, TextureFormat, TextureSampleType,
    },
    view::ViewUniform,
};
use crate::render::atmosphere::GpuSolariAtmosphere;
use crate::render::view_cull::SolariViewUniform;

/// Group index the pathtracer binds the shared scene-columns group at
/// (`@group(0)` scene, `@group(1)` pass, `@group(2)` columns).
pub const SCENE_COLUMNS_GROUP: u32 = 2;

/// The pathtracer's `@group(1)` bind-group layout. Owned by
/// [`SolariResourceManager`](crate::resource_manager::SolariResourceManager); the
/// full pipeline layout is the composite of the scene-bindings group + this + the
/// scene-columns group, assembled in [`crate::pipelines::init_solari_pipelines`].
pub fn pathtracer_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "pathtracer_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<ViewUniform>(true),
                // 3: per-view RT cull mask + env-map intensity
                uniform_buffer::<SolariViewUniform>(true),
                // 4/5: environment map (sky) cube + sampler, sampled on a ray miss.
                // Bound to the skybox / baked-atmosphere cube, or the fallback cube
                // when the view has no env map (brightness 0 → unused).
                texture_cube(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                // 6: atmosphere params + sun, for primary-ray aerial perspective
                // (haze). Disabled (`aerial_enabled = 0`) when the view has no
                // atmosphere; always bound so the slot is valid.
                uniform_buffer::<GpuSolariAtmosphere>(false),
            ),
        ),
    )
}
