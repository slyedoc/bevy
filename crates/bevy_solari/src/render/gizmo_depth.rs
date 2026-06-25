//! Bridges the RT pipeline's primary-hit depth into the hardware depth buffer so
//! rasterized overlays (gizmos drawn in the `Transparent3d` phase) depth-test
//! against the ray-traced scene.
//!
//! Solari's trace is a raw compute pass writing a per-pixel storage buffer; a
//! depth-format texture can only be written by the fixed-function depth unit in a
//! raster pass. [`solari_gizmo_depth`] is that raster pass — a fullscreen triangle
//! whose fragment shader reads the reverse-Z NDC depth the raygen shader packed
//! into [`RtOutputBuffer`]'s `.w` and emits it via `@builtin(frag_depth)`. Ordered
//! after `main_opaque_pass_3d` (depth cleared to far) and before
//! `main_transparent_pass_3d` (gizmos). Works with or without DLSS — the depth
//! rides in the always-present color output buffer, not the DLSS-only G-buffer.

use crate::pipelines::SolariPipelines;
use crate::render::rt_pipeline::RtOutputBuffer;
use crate::render::SolariCamera;
use crate::resource_manager::SolariResourceManager;
use bevy_asset::Handle;
use bevy_core_pipeline::{core_3d::CORE_3D_DEPTH_FORMAT, FullscreenShader};
use bevy_shader::Shader;
use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, uniform_buffer},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedRenderPipelineId, CompareFunction, DepthBiasState, DepthStencilState, FragmentState,
        PipelineCache, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages, StencilState,
        StoreOp,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::{ViewDepthTexture, ViewUniform, ViewUniformOffset, ViewUniforms},
};
use bevy_utils::default;

/// The fullscreen depth-write `@group(0)` layout (rt output buffer + view uniform).
/// Owned by [`SolariResourceManager`](crate::resource_manager::SolariResourceManager).
pub fn gizmo_depth_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "solari_gizmo_depth_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                // 0: the RT per-pixel output buffer; `.w` is the primary-hit depth.
                storage_buffer_read_only_sized(false, None),
                // 1: view uniform (viewport dims → row-major buffer index).
                uniform_buffer::<ViewUniform>(true),
            ),
        ),
    )
}

/// Queue the fullscreen depth-write render pipeline. Called from
/// [`init_solari_pipelines`](crate::pipelines::init_solari_pipelines) — its bulky
/// descriptor (fullscreen vertex + depth state) stays here with its imports; only
/// the id lands in [`SolariPipelines`]. The view depth texture is single-sampled —
/// Solari is a per-pixel compute path, so its camera is `Msaa::Off`.
pub fn gizmo_depth_pipeline(
    pipeline_cache: &PipelineCache,
    fullscreen_shader: &FullscreenShader,
    shader: Handle<Shader>,
    layout: BindGroupLayoutDescriptor,
) -> CachedRenderPipelineId {
    pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("solari_gizmo_depth_pipeline".into()),
        layout: vec![layout],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader,
            entry_point: Some("fragment".into()),
            // Depth-only pass: no color targets.
            targets: vec![],
            ..default()
        }),
        depth_stencil: Some(DepthStencilState {
            format: CORE_3D_DEPTH_FORMAT,
            // Overwrite the opaque pass's cleared far-plane depth with RT depth.
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Always),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        ..default()
    })
}

/// Fullscreen pass: reconstruct the hardware depth buffer from the RT primary-hit
/// depth (packed in [`RtOutputBuffer`]'s `.w`), so the `Transparent3d` phase
/// (gizmos) occludes against the ray-traced scene.
pub fn solari_gizmo_depth(
    view: ViewQuery<
        (
            &RtOutputBuffer,
            &ViewDepthTexture,
            &ExtractedCamera,
            &ViewUniformOffset,
        ),
        With<SolariCamera>,
    >,
    pipelines: Res<SolariPipelines>,
    resource_manager: Res<SolariResourceManager>,
    pipeline_cache: Res<PipelineCache>,
    view_uniforms: Res<ViewUniforms>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (output, depth, camera, view_uniform_offset) = view.into_inner();

    let (Some(render_pipeline), Some(view_uniforms_binding)) = (
        pipeline_cache.get_render_pipeline(pipelines.gizmo_depth),
        view_uniforms.uniforms.binding(),
    ) else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        "solari_gizmo_depth_bind_group",
        &pipeline_cache.get_bind_group_layout(&resource_manager.gizmo_depth),
        &BindGroupEntries::sequential((output.buffer.as_entire_binding(), view_uniforms_binding)),
    );

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("solari_gizmo_depth"),
        color_attachments: &[],
        // Load (the opaque pass already cleared); we overwrite via Always-compare.
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }

    pass.set_render_pipeline(render_pipeline);
    pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
    pass.draw(0..3, 0..1);
}
