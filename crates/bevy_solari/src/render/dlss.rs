//! DLSS Ray Reconstruction for the restir path — driven directly via
//! `dlss_wgpu`, NOT bevy_anti_alias's `Dlss` component.
//!
//! The `Dlss` component `#[require]`s `DepthPrepass` + `MotionVectorPrepass`,
//! which would pull the raster prepass back into our self-contained RT path.
//! Instead we supply our own depth / motion / albedo / normal-roughness guide
//! buffers and call ray reconstruction ourselves. Vulkan extension registration
//! and the `DlssProjectId` still come from bevy_anti_alias's `DlssInitPlugin`
//! (added by `DefaultPlugins` when the `dlss` feature is on).

use std::{
    ops::Deref,
    sync::{Arc, Mutex},
};

use bevy_anti_alias::dlss::{DlssProjectId, DlssRayReconstructionSupported};
use bevy_app::App;
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_camera::MainPassResolutionOverride;
use bevy_diagnostic::FrameCount;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    resource::Resource,
    system::{Commands, Query, Res},
};
use bevy_image::ToExtents;
use bevy_math::{UVec2, Vec4Swizzles};
use bevy_render::{
    camera::TemporalJitter,
    extract_resource::ExtractResource,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, texture_storage_2d, uniform_buffer},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
        ShaderStages, StorageTextureAccess, TextureDescriptor, TextureDimension, TextureFormat,
        TextureUsages, TextureView, TextureViewDescriptor,
    },
    renderer::{
        raw_vulkan_init::AdditionalVulkanFeatures, RenderAdapter, RenderContext, RenderDevice,
        RenderQueue, ViewQuery,
    },
    view::{ExtractedView, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
    RenderApp,
};
use bevy_utils::default;
use derive_more::Display;
use dlss_wgpu::{
    ray_reconstruction::{
        DlssRayReconstruction, DlssRayReconstructionDepthMode,
        DlssRayReconstructionRenderParameters, DlssRayReconstructionRoughnessMode,
        DlssRayReconstructionSpecularGuide,
    },
    DlssFeatureFlags, DlssPerfQualityMode, DlssSdk,
};
use tracing::info;

use crate::bindings::RaytracingSceneBindings;
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;
use super::{prepare::RestirResources, SolariCamera, reset::CameraReset};

/// The DLSS SDK handle, shared by every per-view ray-reconstruction context.
/// Present in the render world only when DLSS Ray Reconstruction is supported.
#[derive(Resource, Clone)]
pub struct RestirDlssSdk(pub Arc<Mutex<DlssSdk>>);

/// DLSS quality mode for every solari view (a main-world resource, extracted —
/// global like [`SolariViewState`](super::view::SolariViewState)). Switching it
/// recreates the per-view RR context at the mode's render resolution. Present
/// only when DLSS Ray Reconstruction is active, so the debug UI keys its
/// dropdown on it.
#[derive(Resource, ExtractResource, Display, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SolariDlssMode {
    /// Native-resolution denoise + anti-aliasing, no upscaling.
    #[default]
    #[display("dlaa")]
    Dlaa,
    #[display("quality")]
    Quality,
    #[display("balanced")]
    Balanced,
    #[display("performance")]
    Performance,
    #[display("ultra performance")]
    UltraPerformance,
}

impl SolariDlssMode {
    /// Every mode, in dropdown order.
    pub const ALL: &'static [SolariDlssMode] = &[
        Self::Dlaa,
        Self::Quality,
        Self::Balanced,
        Self::Performance,
        Self::UltraPerformance,
    ];

    fn perf_quality_mode(self) -> DlssPerfQualityMode {
        match self {
            Self::Dlaa => DlssPerfQualityMode::Dlaa,
            Self::Quality => DlssPerfQualityMode::Quality,
            Self::Balanced => DlssPerfQualityMode::Balanced,
            Self::Performance => DlssPerfQualityMode::Performance,
            Self::UltraPerformance => DlssPerfQualityMode::UltraPerformance,
        }
    }
}

/// Per-view DLSS Ray Reconstruction context (resolution- and mode-specific;
/// recreated on resize or [`SolariDlssMode`] change).
#[derive(Component)]
pub struct RestirDlssContext {
    pub context: Mutex<DlssRayReconstruction>,
    feature_flags: DlssFeatureFlags,
    mode: SolariDlssMode,
}

/// Per-view DLSS guide buffers (render resolution), filled by the resolve pass
/// and consumed by the ray-reconstruction render node.
#[derive(Component)]
pub struct ViewRestirDlssTextures {
    /// Linear camera-space depth (R32Float).
    pub depth: TextureView,
    /// World-space normal (xyz) + linear roughness (w) — `Packed` roughness.
    pub normal_roughness: TextureView,
    pub diffuse_albedo: TextureView,
    pub specular_albedo: TextureView,
    pub specular_motion_vectors: TextureView,
}

/// Creates the DLSS SDK if Ray Reconstruction is supported on this machine and
/// inserts [`RestirDlssSdk`] into the render world. Returns whether it's active.
///
/// Mirrors `bevy_anti_alias::dlss::DlssPlugin::finish`, but only for the RR
/// feature and keyed to the restir path. Degrades gracefully (no DLSS, raw
/// noisy output) when unsupported or SDK init fails.
pub fn init_dlss(app: &mut App) -> bool {
    let ray_reconstruction_supported = app
        .sub_app(RenderApp)
        .world()
        .resource::<AdditionalVulkanFeatures>()
        .has::<DlssRayReconstructionSupported>();
    if !ray_reconstruction_supported {
        info!("RestirPlugin: DLSS Ray Reconstruction unsupported; running without it.");
        return false;
    }

    let project_id = app
        .world()
        .get_resource::<DlssProjectId>()
        .expect("`dlss` feature enabled but `DlssProjectId` was not inserted before plugins.")
        .0;
    let wgpu_device = app
        .sub_app(RenderApp)
        .world()
        .resource::<RenderDevice>()
        .wgpu_device()
        .clone();

    match DlssSdk::new(project_id, wgpu_device) {
        Ok(sdk) => {
            app.sub_app_mut(RenderApp).insert_resource(RestirDlssSdk(sdk));
            info!("RestirPlugin: DLSS Ray Reconstruction enabled.");
            true
        }
        Err(error) => {
            info!("RestirPlugin: DLSS SDK init failed ({error:?}); running without DLSS.");
            false
        }
    }
}

/// Creates / refreshes the per-view RR context + guide textures and sets the
/// render-resolution override. Runs only when the SDK exists (DLSS supported);
/// otherwise a no-op and the restir output is presented raw.
///
/// Must run after `prepare_restir_jitter` so DLSS's `suggested_jitter` wins.
pub fn prepare_restir_dlss(
    sdk: Option<Res<RestirDlssSdk>>,
    mode: Res<SolariDlssMode>,
    mut query: Query<
        (
            Entity,
            &ExtractedView,
            &mut TemporalJitter,
            Option<&mut RestirDlssContext>,
        ),
        With<SolariCamera>,
    >,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    frame_count: Res<FrameCount>,
    mut commands: Commands,
) {
    let Some(sdk) = sdk else {
        return;
    };

    // Linear depth (we have no hardware depth), so no `InvertedDepth`. HDR
    // because the color is linear Rgba16Float; auto-exposure since we don't feed
    // an exposure texture. Low-resolution MVs (ours are at render resolution).
    let feature_flags = DlssFeatureFlags::LowResolutionMotionVectors
        | DlssFeatureFlags::HighDynamicRange
        | DlssFeatureFlags::AutoExposure;
    let mode = *mode;

    for (entity, view, mut temporal_jitter, dlss_context) in &mut query {
        let upscaled_resolution = view.viewport.zw();

        let reuse = match dlss_context.as_deref() {
            Some(context) => {
                UVec2::from(context.context.lock().unwrap().upscaled_resolution())
                    == upscaled_resolution
                    && context.feature_flags == feature_flags
                    && context.mode == mode
            }
            None => false,
        };

        if reuse {
            let context = dlss_context.unwrap();
            let context = context.context.lock().unwrap();
            let render_resolution = UVec2::from(context.render_resolution());
            temporal_jitter.offset = context
                .suggested_jitter(frame_count.0, render_resolution.to_array())
                .into();
            continue;
        }

        let context = DlssRayReconstruction::new(
            upscaled_resolution.to_array(),
            mode.perf_quality_mode(),
            feature_flags,
            DlssRayReconstructionRoughnessMode::Packed,
            DlssRayReconstructionDepthMode::Linear,
            Arc::clone(&sdk.0),
            render_device.wgpu_device(),
            render_queue.deref(),
        )
        .expect("Failed to create restir DLSS Ray Reconstruction context");

        let render_resolution = UVec2::from(context.render_resolution());
        temporal_jitter.offset = context
            .suggested_jitter(frame_count.0, render_resolution.to_array())
            .into();

        let guide = |name: &str, format: TextureFormat| {
            render_device
                .create_texture(&TextureDescriptor {
                    label: Some(name),
                    size: render_resolution.to_extents(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        };

        commands.entity(entity).insert((
            RestirDlssContext {
                context: Mutex::new(context),
                feature_flags,
                mode,
            },
            ViewRestirDlssTextures {
                depth: guide("restir_dlss_depth", TextureFormat::R32Float),
                normal_roughness: guide("restir_dlss_normal_roughness", TextureFormat::Rgba16Float),
                diffuse_albedo: guide("restir_dlss_diffuse_albedo", TextureFormat::Rgba8Unorm),
                specular_albedo: guide("restir_dlss_specular_albedo", TextureFormat::Rgba8Unorm),
                specular_motion_vectors: guide(
                    "restir_dlss_specular_motion",
                    TextureFormat::Rg16Float,
                ),
            },
            MainPassResolutionOverride(render_resolution),
        ));
    }
}

/// The DLSS-guide-resolve `@group(1)` layout. Owned by
/// [`SolariResourceManager`](crate::resource_manager::SolariResourceManager).
pub fn restir_dlss_resolve_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "restir_dlss_resolve_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadOnly),
                uniform_buffer::<ViewUniform>(true),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rg16Float, StorageTextureAccess::WriteOnly),
                // 10: specular reflection first-hit distance
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                // 11: current/previous unjittered clip_from_world ping-pong
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    )
}

/// Queue the DLSS-guide resolve compute pipeline. Called by
/// [`init_solari_pipelines`](crate::pipelines::init_solari_pipelines); only the id
/// lands in `SolariPipelines`. `group(2)` is the shared scene-columns group (the
/// shader imports `scene_bindings`, so the `SOLARI_SCENE_COLUMNS_GROUP` token must
/// be supplied even if unused).
pub fn restir_dlss_resolve_pipeline(
    pipeline_cache: &PipelineCache,
    asset_server: &AssetServer,
    scene_bindings_layout: BindGroupLayoutDescriptor,
    dlss_layout: BindGroupLayoutDescriptor,
    scene_columns_layout: BindGroupLayoutDescriptor,
) -> CachedComputePipelineId {
    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("restir_dlss_resolve_pipeline".into()),
        layout: vec![scene_bindings_layout, dlss_layout, scene_columns_layout],
        shader: load_embedded_asset!(asset_server, "dlss_resolve.wgsl"),
        entry_point: Some("resolve".into()),
        shader_defs: vec![bevy_shader::ShaderDefVal::UInt(
            crate::ecs_gpu::SCENE_COLUMNS_GROUP_DEF.into(),
            2,
        )],
        ..default()
    })
}

/// Fills the DLSS guide buffers from the restir G-buffer (runs after the main
/// restir node, before the DLSS render node).
pub fn restir_dlss_resolve(
    view: ViewQuery<(
        &RestirResources,
        &ViewRestirDlssTextures,
        &ViewUniformOffset,
    )>,
    pipelines: Res<SolariPipelines>,
    resource_manager: Res<SolariResourceManager>,
    pipeline_cache: Res<PipelineCache>,
    scene_bindings: Res<RaytracingSceneBindings>,
    scene_columns: Res<crate::ecs_gpu::SceneColumns>,
    view_uniforms: Res<ViewUniforms>,
    frame_count: Res<FrameCount>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    // Gated by `run_if(not(view_is(Pathtrace)) + the two resources exist)`.
    let (resources, dlss_textures, view_uniform_offset) = view.into_inner();
    let (
        Some(compute_pipeline),
        Some(scene_bind_group),
        Some(scene_columns_bind_group),
        Some(view_uniforms_binding),
    ) = (
        pipeline_cache.get_compute_pipeline(pipelines.dlss_resolve),
        &scene_bindings.bind_group,
        &scene_columns.bind_group,
        view_uniforms.uniforms.binding(),
    ) else {
        return;
    };

    let curr = (frame_count.0 & 1) as usize;
    let bind_group = render_device.create_bind_group(
        "restir_dlss_resolve_bind_group",
        &pipeline_cache.get_bind_group_layout(&resource_manager.dlss_resolve),
        &BindGroupEntries::sequential((
            &resources.world_position[curr],
            &resources.world_normal[curr],
            &resources.uv,
            &resources.motion_vectors,
            view_uniforms_binding,
            &dlss_textures.depth,
            &dlss_textures.normal_roughness,
            &dlss_textures.diffuse_albedo,
            &dlss_textures.specular_albedo,
            &dlss_textures.specular_motion_vectors,
            &resources.specular_hit_distance,
            resources.view_clip_from_world.as_entire_binding(),
        )),
    );

    let dx = resources.view_size.x.div_ceil(8);
    let dy = resources.view_size.y.div_ceil(8);
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("restir_dlss_resolve"),
            timestamp_writes: None,
        });
    pass.set_pipeline(compute_pipeline);
    pass.set_bind_group(0, scene_bind_group, &[]);
    pass.set_bind_group(1, &bind_group, &[view_uniform_offset.offset]);
    pass.set_bind_group(2, scene_columns_bind_group, &[]);
    pass.dispatch_workgroups(dx, dy, 1);
}

/// Runs DLSS Ray Reconstruction: denoises + (optionally) upscales the restir
/// color using the guide buffers, writing the result to the view target.
pub fn restir_dlss(
    view: ViewQuery<(
        &RestirDlssContext,
        &ViewRestirDlssTextures,
        &RestirResources,
        &TemporalJitter,
        &CameraReset,
        &ViewTarget,        
    ), With<SolariCamera>>,
    adapter: Res<RenderAdapter>,
    mut ctx: RenderContext,
) {
    // Gated by `run_if(not(view_is(Pathtrace)))` at registration.
    let (dlss_context, dlss_textures, resources, temporal_jitter, reset, view_target) =
        view.into_inner();

    let view_target = view_target.post_process_write();

    let mut context = dlss_context.context.lock().unwrap();
    let render_resolution = UVec2::from(context.render_resolution());

    let render_parameters = DlssRayReconstructionRenderParameters {
        diffuse_albedo: &dlss_textures.diffuse_albedo,
        specular_albedo: &dlss_textures.specular_albedo,
        normals: &dlss_textures.normal_roughness,
        roughness: None, // packed into normals.w
        color: &view_target.source,
        depth: &dlss_textures.depth,
        motion_vectors: &resources.motion_vectors,
        specular_guide: DlssRayReconstructionSpecularGuide::SpecularMotionVectors(
            &dlss_textures.specular_motion_vectors,
        ),
        screen_space_subsurface_scattering_guide: None,
        bias: None,
        dlss_output: &view_target.destination,
        reset: reset.0,
        jitter_offset: (-temporal_jitter.offset).to_array(),
        partial_texture_size: Some(render_resolution.to_array()),
        motion_vector_scale: Some((-render_resolution.as_vec2()).to_array()),
    };

    let command_buffer = context
        .render(render_parameters, ctx.command_encoder(), &adapter)
        .expect("Failed to render restir DLSS Ray Reconstruction");
    ctx.add_command_buffer(command_buffer);
}
