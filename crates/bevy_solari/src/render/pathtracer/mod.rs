pub mod pipelines;

use crate::{bindings::RaytracingSceneBindings, pipelines::SolariPipelines, resource_manager::SolariResourceManager, render::{CameraReset, SolariCamera, atmosphere::{AtmosphereSky, SolariAtmosphereGpu, SolariAtmosphereView}, view_cull::{SolariEnvironmentMap, SolariViewOffset, SolariViewUniforms}}};
use bevy_ecs::{prelude::*, system::Commands};
use bevy_image::ToExtents;
use bevy_render::{
    camera::ExtractedCamera, diagnostic::RecordDiagnostics as _, render_asset::RenderAssets,
    render_resource::{BindGroupEntries, TextureUsages, TextureDimension, ComputePassDescriptor, ImageSubresourceRange, PipelineCache, TextureDescriptor, TextureFormat}, renderer::{RenderContext, RenderDevice, ViewQuery}, texture::{CachedTexture, FallbackImage, GpuImage, TextureCache}, view::{ViewTarget, ViewUniformOffset, ViewUniforms}
};

pub fn pathtracer(
    view: ViewQuery<(
        &CameraReset,
        &PathtracerAccumulationTexture,
        &ExtractedCamera,
        &ViewTarget,
        &ViewUniformOffset,
        &SolariViewOffset,
        Option<&SolariEnvironmentMap>,
        Option<&SolariAtmosphereView>,
    )>,
    resource_manager: Res<SolariResourceManager>,
    solari_pipelines: Res<SolariPipelines>,
    pipeline_cache: Res<PipelineCache>,
    scene_bindings: Res<RaytracingSceneBindings>,
    scene_columns: Res<crate::ecs_gpu::SceneColumns>,
    view_uniforms: Res<ViewUniforms>,
    solari_view_uniforms: Res<SolariViewUniforms>,
    solari_atmosphere: Res<SolariAtmosphereGpu>,
    texture_assets: Res<RenderAssets<GpuImage>>,
    atmosphere_sky: Option<Res<AtmosphereSky>>,
    fallback_image: Res<FallbackImage>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (
        pathtracer_reset,
        accumulation_texture,
        camera,
        view_target,
        view_uniform_offset,
        solari_view_offset,
        environment_map,
        atmosphere_view,
    ) = view.into_inner();

    // Environment map (sky), sampled on a ray miss. Priority: the baked atmosphere
    // cube if this view has one (its sampler is the fallback's filtering sampler),
    // else the view's skybox cube if present + uploaded, else the fallback cube.
    // When neither, `environment_brightness` is 0 so the bound texture is unused.
    let (environment_map_view, environment_map_sampler) = atmosphere_view
        .and(atmosphere_sky.as_ref())
        .map(|sky| (&sky.cube_view, &fallback_image.cube.sampler))
        .or_else(|| {
            environment_map
                .and_then(|env| texture_assets.get(&env.image))
                .map(|image| (&image.texture_view, &image.sampler))
        })
        .unwrap_or((
            &fallback_image.cube.texture_view,
            &fallback_image.cube.sampler,
        ));

    let (
        Some(pipeline),
        Some(scene_bind_group),
        Some(scene_columns_bind_group),
        Some(viewport),
        Some(view_uniforms_binding),
        Some(solari_view_binding),
        Some(atmosphere_binding),
    ) = (
        pipeline_cache.get_compute_pipeline(solari_pipelines.pathtracer),
        &scene_bindings.bind_group,
        &scene_columns.bind_group,
        camera.physical_viewport_size,
        view_uniforms.uniforms.binding(),
        solari_view_uniforms.uniforms.binding(),
        solari_atmosphere.binding(),
    ) else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        "pathtracer_bind_group",
        &pipeline_cache.get_bind_group_layout(&resource_manager.pathtracer),
        &BindGroupEntries::sequential((
            &accumulation_texture.0.default_view,
            view_target.get_unsampled_color_attachment().view,
            view_uniforms_binding,
            solari_view_binding,
            environment_map_view,
            environment_map_sampler,
            atmosphere_binding,
        )),
    );

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let command_encoder = ctx.command_encoder();

    if pathtracer_reset.0 {
        command_encoder.clear_texture(
            &accumulation_texture.0.texture,
            &ImageSubresourceRange::default(),
        );
    }

    let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("pathtracer"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, scene_bind_group, &[]);
    pass.set_bind_group(
        1,
        &bind_group,
        &[view_uniform_offset.offset, solari_view_offset.0],
    );
    pass.set_bind_group(pipelines::SCENE_COLUMNS_GROUP, scene_columns_bind_group, &[]);

    let d = diagnostics.time_span(&mut pass, "pathtracer");
    pass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);
    d.end(&mut pass);
}

#[derive(Component)]
pub struct PathtracerAccumulationTexture(pub CachedTexture);

pub fn prepare_pathtracer_accumulation_texture(
    query: Query<(Entity, &ExtractedCamera), With<SolariCamera>>,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for (entity, camera) in &query {
        let Some(viewport) = camera.physical_viewport_size else {
            continue;
        };

        let descriptor = TextureDescriptor {
            label: Some("pathtracer_accumulation_texture"),
            size: viewport.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        commands
            .entity(entity)
            .insert(PathtracerAccumulationTexture(
                texture_cache.get(&render_device, descriptor),
            ));
    }
}


