use super::{prepare::RestirResources};
use crate::bindings::RaytracingSceneBindings;
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;
use crate::render::SolariCamera;
use crate::render::view_cull::{SolariViewOffset, SolariViewUniform, SolariViewUniforms};
use bevy_diagnostic::FrameCount;
use bevy_ecs::prelude::*;
use bevy_math::Vec2;
use bevy_render::camera::TemporalJitter;
use bevy_render::{
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        binding_types::{storage_buffer_sized, texture_storage_2d, uniform_buffer},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, ComputePassDescriptor,
        LoadOp, PipelineCache, RenderPassDescriptor, ShaderStages, StorageTextureAccess,
        TextureFormat,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
};

use super::prepare::LIGHT_TILE_BLOCKS;

/// The ReSTIR `@group(1)` bind-group layout (shared by all six passes). Owned by
/// [`SolariResourceManager`](crate::resource_manager::SolariResourceManager); the
/// six pipeline ids live in [`SolariPipelines`](crate::pipelines::SolariPipelines),
/// queued against the composite (scene-bindings group + this + scene-columns group).
pub fn restir_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "restir_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // 0: view output (final color target)
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite),
                // 1: world position (current)
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                // 2: world position (previous)
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                // 3: world normal (current)
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite),
                // 4: world normal (previous)
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite),
                // 5: motion vectors
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite),
                // 6: reservoirs (current)
                storage_buffer_sized(false, None),
                // 7: reservoirs (previous)
                storage_buffer_sized(false, None),
                // 8: light tiles
                storage_buffer_sized(false, None),
                // 9: view uniform
                uniform_buffer::<ViewUniform>(true),
                // 10: primary-hit uv (for shade-time material re-resolve)
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                // 11: previous-frame clip_from_world ping-pong (2 mat4x4)
                storage_buffer_sized(false, None),
                // 12-13: GI reservoirs (history, intermediate)
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                // 14: per-view RT cull mask
                uniform_buffer::<SolariViewUniform>(true),
            ),
        ),
    )
}

pub fn restir(
    view: ViewQuery<(
        &RestirResources,
        &ViewTarget,
        &ViewUniformOffset,
        &SolariViewOffset,
    ), With<SolariCamera>>,
    pipelines: Res<SolariPipelines>,
    resource_manager: Res<SolariResourceManager>,
    pipeline_cache: Res<PipelineCache>,
    scene_bindings: Res<RaytracingSceneBindings>,
    scene_columns: Res<crate::ecs_gpu::SceneColumns>,
    view_uniforms: Res<ViewUniforms>,
    solari_view_uniforms: Res<SolariViewUniforms>,
    frame_count: Res<FrameCount>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    // Gated by `run_if(not(view_is(Pathtrace)) + the two resources exist)` at
    // registration — the reference path tracer renders pathtrace views instead.
    let (resources, view_target, view_uniform_offset, solari_view_offset) = view.into_inner();

    let (
        Some(visibility_pipeline),
        Some(presample_pipeline),
        Some(initial_and_temporal_pipeline),
        Some(spatial_and_shade_pipeline),
        Some(specular_gi_pipeline),
        Some(compose_pipeline),
    ) = (
        pipeline_cache.get_compute_pipeline(pipelines.restir_visibility),
        pipeline_cache.get_compute_pipeline(pipelines.restir_presample),
        pipeline_cache.get_compute_pipeline(pipelines.restir_initial_and_temporal),
        pipeline_cache.get_compute_pipeline(pipelines.restir_spatial_and_shade),
        pipeline_cache.get_compute_pipeline(pipelines.restir_specular_gi),
        pipeline_cache.get_compute_pipeline(pipelines.restir_compose),
    ) else {
        return;
    };
    let (
        Some(scene_bind_group),
        Some(scene_columns_bind_group),
        Some(view_uniforms_binding),
        Some(solari_view_binding),
    ) = (
        &scene_bindings.bind_group,
        &scene_columns.bind_group,
        view_uniforms.uniforms.binding(),
        solari_view_uniforms.uniforms.binding(),
    )
    else {
        return;
    };

    // Ping-pong: current = frame parity, previous = the other slot.
    let curr = (frame_count.0 & 1) as usize;
    let prev = curr ^ 1;

    let view_target_attachment = view_target.get_unsampled_color_attachment();

    let bind_group = render_device.create_bind_group(
        "restir_bind_group",
        &pipeline_cache.get_bind_group_layout(&resource_manager.restir),
        &BindGroupEntries::sequential((
            view_target_attachment.view,
            &resources.world_position[curr],
            &resources.world_position[prev],
            &resources.world_normal[curr],
            &resources.world_normal[prev],
            &resources.motion_vectors,
            // Fixed roles: [0] = history, [1] = intermediate (not parity-swapped).
            resources.reservoirs[0].as_entire_binding(),
            resources.reservoirs[1].as_entire_binding(),
            resources.light_tiles.as_entire_binding(),
            view_uniforms_binding,
            &resources.uv,
            resources.view_clip_from_world.as_entire_binding(),
            resources.gi_reservoirs[0].as_entire_binding(),
            resources.gi_reservoirs[1].as_entire_binding(),
            solari_view_binding,
        )),
    );

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let command_encoder = ctx.command_encoder();

    // Clear the view target if we're the first node to write to it.
    if matches!(view_target_attachment.ops.load, LoadOp::Clear(_)) {
        command_encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("restir_clear"),
            color_attachments: &[Some(view_target_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("restir"),
        timestamp_writes: None,
    });

    let dx = resources.view_size.x.div_ceil(8);
    let dy = resources.view_size.y.div_ceil(8);

    pass.set_bind_group(0, scene_bind_group, &[]);
    pass.set_bind_group(
        1,
        &bind_group,
        &[view_uniform_offset.offset, solari_view_offset.0],
    );
    pass.set_bind_group(2, scene_columns_bind_group, &[]);

    // 1. Primary visibility → lean PT G-buffer.
    let d = diagnostics.time_span(&mut pass, "restir/visibility");
    pass.set_pipeline(visibility_pipeline);    pass.dispatch_workgroups(dx, dy, 1);
    d.end(&mut pass);

    // 2. Presample analytic + emissive lights into the tile pool.
    let d = diagnostics.time_span(&mut pass, "restir/presample");
    pass.set_pipeline(presample_pipeline);    pass.dispatch_workgroups(LIGHT_TILE_BLOCKS as u32, 1, 1);
    d.end(&mut pass);

    // 3. Initial candidate generation + temporal reuse.
    let d = diagnostics.time_span(&mut pass, "restir/initial_and_temporal");
    pass.set_pipeline(initial_and_temporal_pipeline);    pass.dispatch_workgroups(dx, dy, 1);
    d.end(&mut pass);

    // 4. Spatial reuse + shade (diffuse direct + diffuse indirect).
    let d = diagnostics.time_span(&mut pass, "restir/spatial_and_shade");
    pass.set_pipeline(spatial_and_shade_pipeline);    pass.dispatch_workgroups(dx, dy, 1);
    d.end(&mut pass);

    // 4b. Specular GI (adds specular indirect on top).
    let d = diagnostics.time_span(&mut pass, "restir/specular_gi");
    pass.set_pipeline(specular_gi_pipeline);    pass.dispatch_workgroups(dx, dy, 1);
    d.end(&mut pass);

    // 5. Compose final color → view target.
    let d = diagnostics.time_span(&mut pass, "restir/compose");
    pass.set_pipeline(compose_pipeline);    pass.dispatch_workgroups(dx, dy, 1);
    d.end(&mut pass);
}

/// Drives the camera's sub-pixel jitter (Halton (2,3) − 0.5) for ReSTIR views.
///
/// bevy's `prepare_view_uniforms` applies this offset to `clip_from_world`, so
/// the visibility ray (which uses `view.world_from_clip`) samples a different
/// sub-pixel each frame — anti-aliasing / temporal-upscaler input. Motion
/// vectors are computed from `unjittered_clip_from_world`, so they stay
/// jitter-free. Pre-DLSS only: once DLSS is attached it drives the jitter
/// itself (via `suggested_jitter`).
pub fn prepare_restir_jitter(
    frame_count: Res<FrameCount>,
    mut views: Query<&mut TemporalJitter, With<SolariCamera>>,
) {
    // Halton (2, 3) − 0.5, matching bevy's TAA.
    const HALTON: [Vec2; 8] = [
        Vec2::new(0.0, 0.0),
        Vec2::new(0.0, -0.16666666),
        Vec2::new(-0.25, 0.16666669),
        Vec2::new(0.25, -0.3888889),
        Vec2::new(-0.375, -0.055555552),
        Vec2::new(0.125, 0.2777778),
        Vec2::new(-0.125, -0.2777778),
        Vec2::new(0.375, 0.055555582),
    ];
    let offset = HALTON[frame_count.0 as usize % HALTON.len()];
    for mut jitter in &mut views {
        jitter.offset = offset;
    }
}

