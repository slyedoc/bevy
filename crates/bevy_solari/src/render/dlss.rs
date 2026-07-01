//! DLSS Ray Reconstruction for the rt_pipeline path — driven **directly** via
//! `dlss_wgpu`, not bevy_anti_alias's `Dlss` component.
//!
//! The `Dlss` component `#[require]`s `DepthPrepass` + `MotionVectorPrepass`, which
//! would pull a raster prepass back into our self-contained RT path. Instead we
//! produce our own depth / motion / albedo / normal-roughness / specular-hit-distance
//! guide buffers from the trace itself and call ray reconstruction ourselves. Vulkan
//! extension registration + the `DlssProjectId` still come from bevy_anti_alias's
//! `DlssInitPlugin` (added by `DefaultPlugins` when the `dlss` feature is on).
//!
//! Flow each frame (when a view has an active context):
//! 1. `rt_pipeline` traces at render resolution, writing the noisy color into the
//!    view target (the blit) and the packed guide STORAGE BUFFERS (chit-direct).
//! 2. [`solari_dlss_resolve`] unpacks those buffers into the guide TEXTURES RR wants.
//! 3. [`solari_dlss_render`] runs ray reconstruction: view-target color + guides →
//!    the denoised (and, later, upscaled) view-target output.
//!
//! Phase 3a (this milestone): DLAA only — render resolution == display resolution
//! (no upscaling). Every non-`Off` mode is treated as DLAA here; the true per-mode
//! render-resolution + `MainPassResolutionOverride` upscaling lands next.

use std::{
    ops::Deref,
    sync::{Arc, Mutex},
};

use bevy_anti_alias::dlss::DlssRayReconstructionSupported;
use bevy_app::App;
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_diagnostic::FrameCount;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    resource::Resource,
    system::{Commands, Query, Res},
};
use bevy_math::{Mat4, ToRender, UVec2, Vec2};
use bevy_render::{
    camera::ExtractedCamera,
    extract_resource::ExtractResource,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, texture_storage_2d},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
        PipelineCache, ShaderStages, StorageTextureAccess, Texture, TextureDescriptor,
        TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    },
    renderer::{
        raw_vulkan_init::AdditionalVulkanFeatures, RenderAdapter, RenderContext, RenderDevice,
        RenderQueue, ViewQuery,
    },
    view::{ExtractedView, ViewTarget},
    RenderApp,
};
use dlss_wgpu::{
    ray_reconstruction::{
        DlssRayReconstruction, DlssRayReconstructionDepthMode,
        DlssRayReconstructionRenderParameters, DlssRayReconstructionRoughnessMode,
        DlssRayReconstructionSpecularGuide,
    },
    DlssFeatureFlags, DlssPerfQualityMode, DlssSdk,
};
use tracing::info;

use crate::render::rt_pipeline::{RtOutputBuffer, SolariDlssJitter};
use crate::render::{CameraReset, SolariCamera};

/// The DLSS SDK handle, shared by every per-view ray-reconstruction context.
/// Present in the render world only when DLSS Ray Reconstruction is supported and
/// the SDK initialized — [`init_dlss`].
#[derive(Resource, Clone)]
pub struct SolariDlssSdk(pub Arc<Mutex<DlssSdk>>);

/// DLSS quality mode for the solari view — a main-world resource, extracted to the
/// render world. Switching it recreates the per-view RR context at the mode's render
/// resolution. `Off` bypasses DLSS entirely (raw noisy trace output), the baseline
/// for "is the denoiser causing this".
#[derive(Resource, ExtractResource, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SolariDlssMode {
    #[default]
    Off,
    /// DLSS picks the upscale factor for the output resolution.
    Auto,
    /// Native-resolution denoise + anti-aliasing, no upscaling (max ray cost).
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl SolariDlssMode {
    /// Every mode, in dropdown order.
    pub const ALL: &'static [SolariDlssMode] = &[
        Self::Off,
        Self::Auto,
        Self::Dlaa,
        Self::Quality,
        Self::Balanced,
        Self::Performance,
        Self::UltraPerformance,
    ];

    /// Lowercase label for UI / logs.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::Dlaa => "dlaa",
            Self::Quality => "quality",
            Self::Balanced => "balanced",
            Self::Performance => "performance",
            Self::UltraPerformance => "ultra performance",
        }
    }

    /// The `dlss_wgpu` quality mode this maps to (the render→display ratio).
    /// Used by the upscaling milestone; Phase 3a forces DLAA.
    #[allow(dead_code)]
    pub(crate) fn perf_quality_mode(self) -> DlssPerfQualityMode {
        match self {
            Self::Auto => DlssPerfQualityMode::Auto,
            Self::Off | Self::Dlaa => DlssPerfQualityMode::Dlaa,
            Self::Quality => DlssPerfQualityMode::Quality,
            Self::Balanced => DlssPerfQualityMode::Balanced,
            Self::Performance => DlssPerfQualityMode::Performance,
            Self::UltraPerformance => DlssPerfQualityMode::UltraPerformance,
        }
    }
}

impl core::fmt::Display for SolariDlssMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.label())
    }
}

/// Per-view DLSS Ray Reconstruction context (resolution- and mode-specific;
/// recreated on resize or [`SolariDlssMode`] change). `Mutex` because the context is
/// driven only from the single render dispatch (and to satisfy `Component: Sync`).
#[derive(Component)]
pub struct SolariDlssContext {
    pub context: Mutex<DlssRayReconstruction>,
    feature_flags: DlssFeatureFlags,
    // The `dlss_wgpu` quality the context was built at — NOT `SolariDlssMode`. In
    // Phase 3a every active mode maps to DLAA, so cycling modes keeps this constant
    // and the context is reused (recreating it mid-flight hangs the GPU).
    perf_quality: DlssPerfQualityMode,
}

/// Per-view DLSS guide textures (render resolution), filled by [`solari_dlss_resolve`]
/// from the trace's packed guide buffers and read by [`solari_dlss_render`].
/// Recreated when the render resolution changes.
#[derive(Component)]
pub struct SolariDlssTextures {
    size: UVec2,
    depth: TextureView,
    normal_roughness: TextureView,
    diffuse_albedo: TextureView,
    specular_albedo: TextureView,
    motion: TextureView,
    specular_hit_distance: TextureView,
    // Keep the textures alive for the views' lifetime (views alone don't own them).
    _textures: [Texture; 6],
}

/// The DLSS guide-resolve compute pipeline + its bind-group layout.
#[derive(Resource)]
pub struct SolariDlssResolve {
    pub layout: BindGroupLayoutDescriptor,
    pub pipeline: CachedComputePipelineId,
}

/// Create the DLSS SDK if Ray Reconstruction is supported on this machine and insert
/// [`SolariDlssSdk`] into the render world. Returns whether DLSS RR is active.
///
/// Mirrors `bevy_anti_alias::dlss::DlssPlugin::finish`, but only for the RR feature
/// and keyed to the rt_pipeline path. Degrades gracefully (no DLSS, raw noisy output)
/// when unsupported or SDK init fails. Call from [`SolariPlugin::finish`].
pub fn init_dlss(app: &mut App) -> bool {
    let ray_reconstruction_supported = app
        .sub_app(RenderApp)
        .world()
        .resource::<AdditionalVulkanFeatures>()
        .has::<DlssRayReconstructionSupported>();
    if !ray_reconstruction_supported {
        info!("SolariPlugin: DLSS Ray Reconstruction unsupported; running without it.");
        return false;
    }

    let project_id = app
        .world()
        .get_resource::<bevy_anti_alias::dlss::DlssProjectId>()
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
            app.sub_app_mut(RenderApp)
                .insert_resource(SolariDlssSdk(sdk));
            true
        }
        Err(error) => {
            info!("SolariPlugin: DLSS SDK init failed ({error:?}); running without DLSS.");
            false
        }
    }
}

/// `RenderStartup`: build the guide-resolve compute pipeline (independent of the
/// per-view DLSS state; harmless when DLSS ends up unsupported).
pub fn init_solari_dlss(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "solari_dlss_resolve_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // 0..4: packed guide buffers from the trace.
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                // 4..10: unpacked guide textures RR consumes.
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rg16Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("solari_dlss_resolve_pipeline".into()),
        layout: vec![layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "dlss_resolve.wgsl"),
        shader_defs: vec![],
        entry_point: Some("resolve".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    commands.insert_resource(SolariDlssResolve { layout, pipeline });
}

fn create_guide_texture(
    render_device: &RenderDevice,
    size: UVec2,
    format: TextureFormat,
    label: &'static str,
) -> (Texture, TextureView) {
    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        // Resolve writes (storage), DLSS reads.
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

fn create_dlss_textures(render_device: &RenderDevice, size: UVec2) -> SolariDlssTextures {
    let (t_depth, depth) = create_guide_texture(
        render_device,
        size,
        TextureFormat::R32Float,
        "solari_dlss_depth",
    );
    let (t_nr, normal_roughness) = create_guide_texture(
        render_device,
        size,
        TextureFormat::Rgba16Float,
        "solari_dlss_normal_roughness",
    );
    let (t_diff, diffuse_albedo) = create_guide_texture(
        render_device,
        size,
        TextureFormat::Rgba8Unorm,
        "solari_dlss_diffuse_albedo",
    );
    let (t_spec, specular_albedo) = create_guide_texture(
        render_device,
        size,
        TextureFormat::Rgba8Unorm,
        "solari_dlss_specular_albedo",
    );
    let (t_motion, motion) = create_guide_texture(
        render_device,
        size,
        TextureFormat::Rg16Float,
        "solari_dlss_motion",
    );
    let (t_shd, specular_hit_distance) = create_guide_texture(
        render_device,
        size,
        TextureFormat::R32Float,
        "solari_dlss_specular_hit_distance",
    );
    SolariDlssTextures {
        size,
        depth,
        normal_roughness,
        diffuse_albedo,
        specular_albedo,
        motion,
        specular_hit_distance,
        _textures: [t_depth, t_nr, t_diff, t_spec, t_motion, t_shd],
    }
}

/// `Render`/`PrepareResources`: create the per-view RR context + guide textures ONCE
/// (kept for the view's lifetime — the NGX lifecycle must not churn while the raw
/// trace is live) and set the sub-pixel jitter when DLSS is on. No-op when the SDK is
/// absent (DLSS unsupported). Runs before `rt_pipeline` (which reads the jitter later,
/// in `Core3d`).
///
/// Phase 3a: always DLAA (render == display); the true per-mode render resolution +
/// `MainPassResolutionOverride` upscaling lands in the next milestone.
pub fn prepare_solari_dlss(
    sdk: Option<Res<SolariDlssSdk>>,
    mode: Res<SolariDlssMode>,
    mut views: Query<
        (
            Entity,
            &ExtractedCamera,
            Option<&mut SolariDlssContext>,
            Option<&SolariDlssTextures>,
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

    // Linear depth (no hardware depth) ⇒ no `InvertedDepth`. HDR (linear Rgba16Float
    // color). Low-resolution MVs (ours are at render resolution). NO auto-exposure:
    // raygen already applies the camera exposure (`final_color *= exposure`), so the
    // color is pre-exposed and DLSS must stay exposure-neutral (assumes exposure 1.0).
    // With auto-exposure on top, a dark scene (e.g. a black/unbaked sky) makes DLSS
    // crank the gain and the lit meshes blow out.
    let feature_flags =
        DlssFeatureFlags::LowResolutionMotionVectors | DlssFeatureFlags::HighDynamicRange;
    // Phase 3a: every active mode runs as DLAA (render == display). The per-mode
    // `mode.perf_quality_mode()` render resolution is applied in the upscaling milestone.
    let perf_quality = DlssPerfQualityMode::Dlaa;

    for (entity, camera, context, textures) in &mut views {
        let Some(upscaled) = camera.physical_viewport_size else {
            continue;
        };

        // CRITICAL: the NGX feature lifecycle (create/destroy) must NEVER run while the
        // raw trace is live — it hard-hangs the GPU (the restir path didn't hit this
        // because it had no raw trace; draining alone didn't cover it). So the per-view
        // context is created ONCE, as early as possible — the first frame the viewport
        // is known, before the rt_pipeline has even built or traced — and kept for the
        // view's lifetime regardless of mode. `SolariDlssMode` gates ONLY whether the
        // resolve/render run (see the run conditions) and whether the trace is jittered,
        // never the context. Recreated only on a resolution/quality change (rare).
        let reuse = match context.as_deref() {
            Some(c) => {
                UVec2::from(c.context.lock().unwrap().upscaled_resolution()) == upscaled
                    && c.feature_flags == feature_flags
                    && c.perf_quality == perf_quality
            }
            None => false,
        };

        let render_resolution;
        let jitter: Vec2;
        if reuse {
            let context = context.unwrap();
            let locked = context.context.lock().unwrap();
            render_resolution = UVec2::from(locked.render_resolution());
            jitter = locked
                .suggested_jitter(frame_count.0, render_resolution.to_array())
                .into();
        } else {
            // Drain before the rare resolution/quality recreate (which drops the old
            // context at command flush). The very first create is pre-trace, so the GPU
            // is already idle.
            let _ = render_device
                .wgpu_device()
                .poll(wgpu::PollType::wait_indefinitely());
            let context = DlssRayReconstruction::new(
                upscaled.to_array(),
                perf_quality,
                feature_flags,
                DlssRayReconstructionRoughnessMode::Packed,
                DlssRayReconstructionDepthMode::Linear,
                Arc::clone(&sdk.0),
                render_device.wgpu_device(),
                render_queue.deref(),
            )
            .expect("Failed to create solari DLSS Ray Reconstruction context");

            render_resolution = UVec2::from(context.render_resolution());
            jitter = context
                .suggested_jitter(frame_count.0, render_resolution.to_array())
                .into();
            commands.entity(entity).insert(SolariDlssContext {
                context: Mutex::new(context),
                feature_flags,
                perf_quality,
            });
        }

        // (Re)allocate the guide textures when the render resolution changes.
        if textures.map_or(true, |t| t.size != render_resolution) {
            commands
                .entity(entity)
                .insert(create_dlss_textures(&render_device, render_resolution));
        }

        // Jitter the trace only when DLSS is actually running; Off renders a fresh,
        // unjittered frame (an unaccumulated jitter would just shimmer). The context
        // itself stays alive either way — the mode never touches its lifecycle.
        if *mode == SolariDlssMode::Off {
            commands.entity(entity).remove::<SolariDlssJitter>();
        } else {
            commands
                .entity(entity)
                .insert(SolariDlssJitter { offset: jitter });
        }
    }
}

/// Run condition: DLSS is enabled (mode != `Off`). The per-view context + guide
/// textures exist regardless (created eagerly and kept alive — see
/// [`prepare_solari_dlss`]), so this is the sole gate on the resolve + render
/// dispatch; toggling the mode never touches the NGX lifecycle.
pub fn dlss_enabled(mode: Res<SolariDlssMode>) -> bool {
    *mode != SolariDlssMode::Off
}

/// `Core3d` (after `rt_pipeline`, before `solari_dlss_render`): unpack the trace's
/// packed guide buffers into the guide textures RR consumes. Gated on
/// [`dlss_enabled`]; the context + textures are always present.
pub fn solari_dlss_resolve(
    view: ViewQuery<(&RtOutputBuffer, &SolariDlssTextures)>,
    resolve: Res<SolariDlssResolve>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (output, textures) = view.into_inner();
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(resolve.pipeline) else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        "solari_dlss_resolve_bind_group",
        &pipeline_cache.get_bind_group_layout(&resolve.layout),
        &BindGroupEntries::sequential((
            output.gbuffer[0].buffer.as_entire_binding(), // normal + roughness
            output.gbuffer[1].buffer.as_entire_binding(), // diffuse + depth
            output.gbuffer[2].buffer.as_entire_binding(), // specular + hit distance
            output.gbuffer[3].buffer.as_entire_binding(), // motion
            &textures.depth,
            &textures.normal_roughness,
            &textures.diffuse_albedo,
            &textures.specular_albedo,
            &textures.motion,
            &textures.specular_hit_distance,
        )),
    );

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("solari_dlss_resolve"),
            timestamp_writes: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(textures.size.x.div_ceil(8), textures.size.y.div_ceil(8), 1);
}

/// `Core3d` (after `solari_dlss_resolve`, before tonemapping): run DLSS Ray
/// Reconstruction — denoise (+ later upscale) the noisy view-target color using the
/// guide textures, writing the result back into the view target.
pub fn solari_dlss_render(
    view: ViewQuery<(
        &ExtractedView,
        &SolariDlssContext,
        &SolariDlssTextures,
        &SolariDlssJitter,
        &CameraReset,
        &ViewTarget,
    )>,
    adapter: Res<RenderAdapter>,
    mut ctx: RenderContext,
) {
    let (view, dlss_context, textures, jitter, reset, view_target) = view.into_inner();
    let post = view_target.post_process_write();

    let mut context = dlss_context.context.lock().unwrap();
    let render_resolution = UVec2::from(context.render_resolution());

    // Row-major camera matrices for the specular hit-distance guide. The guide is
    // consumed in origin-relative space where the primary camera sits at 0, so the
    // basis is rotation-only (`world_from_view` is the camera's ABSOLUTE world).
    let view_from_world = Mat4::from_quat(view.world_from_view.rotation().to_render()).inverse();
    let world_to_view_rows_array = view_from_world.transpose().to_cols_array();
    let view_to_clip_rows_array = view.clip_from_view.transpose().to_cols_array();

    let render_parameters = DlssRayReconstructionRenderParameters {
        diffuse_albedo: &textures.diffuse_albedo,
        specular_albedo: &textures.specular_albedo,
        normals: &textures.normal_roughness,
        roughness: None, // packed into normals.w
        color: post.source,
        depth: &textures.depth,
        motion_vectors: &textures.motion,
        specular_guide: DlssRayReconstructionSpecularGuide::SpecularHitDistance {
            texture_view: &textures.specular_hit_distance,
            world_to_view_rows_array,
            view_to_clip_rows_array,
        },
        screen_space_subsurface_scattering_guide: None,
        bias: None,
        dlss_output: post.destination,
        reset: reset.active(),
        jitter_offset: (-jitter.offset).to_array(),
        partial_texture_size: Some(render_resolution.to_array()),
        // MV are UV-space (NDC*0.5, y-flipped); RR wants pixel-space toward history.
        motion_vector_scale: Some((-render_resolution.as_vec2()).to_array()),
    };

    let command_buffer = context
        .render(render_parameters, ctx.command_encoder(), &adapter)
        .expect("Failed to render solari DLSS Ray Reconstruction");
    ctx.add_command_buffer(command_buffer);
}
