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
//!    the denoised view-target output.
//!
//! Upscaling is not yet implemented: every non-`Off` mode runs as DLAA (render
//! resolution == display resolution).

#![allow(unsafe_code)]

use std::{
    ops::Deref,
    sync::{Arc, Mutex},
};

use ash::vk;
use bevy_anti_alias::dlss::DlssRayReconstructionSupported;
use bevy_app::App;
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
        CommandEncoderDescriptor, Extent3d, Texture, TextureDescriptor, TextureDimension,
        TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    },
    renderer::{
        raw_vulkan_init::AdditionalVulkanFeatures, RenderAdapter, RenderContext, RenderDevice,
        RenderQueue, ViewQuery,
    },
    view::{ExtractedView, ViewTarget},
    RenderApp,
};
use wgpu::hal::api::Vulkan as VkApi;

use crate::gpu::allocator::Allocator;
use crate::gpu::binding_seam::{BindingSeam, HeapKind};
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
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
    // The `dlss_wgpu` quality the context was built at — NOT `SolariDlssMode`.
    // Every active mode maps to DLAA, so cycling modes keeps this constant and
    // the context is reused (recreating it mid-flight hangs the GPU).
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

/// The DLSS guide-resolve heap kernel (`dlss_resolve.slang`). The per-view
/// heap slots live on [`SolariDlssResolveSlots`].
#[derive(Resource)]
pub struct SolariDlssResolve {
    kernel: HeapKernel,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for SolariDlssResolve {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for SolariDlssResolve {}
unsafe impl Sync for SolariDlssResolve {}

/// Per-view heap slots for the guide resolve (4 packed-buffer + 6 storage-image
/// slots). Per view because slot descriptors resolve at execution time — one
/// shared set rewritten per view would leave every recorded dispatch reading
/// the last view's resources. Created once per view and kept (slots are
/// app-lifetime); a resolution change only changes what the per-dispatch
/// rewrites point them at.
#[derive(Component)]
pub struct SolariDlssResolveSlots(KernelSlots);

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

/// `RenderStartup` (after `SolariSetup`): compile the guide-resolve kernel —
/// a layout-free heap pipeline ([`HeapKernel`]), Slang from source (harmless
/// when DLSS ends up unsupported). No push params: the dispatch sizes itself
/// from the depth texture.
pub fn init_solari_dlss(
    mut commands: Commands,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let Some(kernel) = HeapKernel::new(
        &seam,
        "dlss_resolve.slang",
        include_str!("dlss_resolve.slang"),
        "resolve",
        &[],
        &[],
        "solari_dlss_resolve",
        0,
    ) else {
        return;
    };
    commands.insert_resource(SolariDlssResolve {
        kernel,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

fn create_guide_texture(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    size: UVec2,
    format: TextureFormat,
    label: &'static str,
) -> (Texture, TextureView) {
    let extent = Extent3d {
        width: size.x,
        height: size.y,
        depth_or_array_layers: 1,
    };
    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        // Resolve writes (raw storage), DLSS reads; COPY_DST for the
        // initializing write below.
        usage: TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Mark the texture initialized with a TRACKED write: wgpu lazily
    // zero-initializes a texture at its first tracked use, and these are
    // written only by the untracked raw resolve — so RR's first tracked read
    // would otherwise inject a zero-clear that wipes the resolved guides.
    let bytes_per_pixel = format
        .block_copy_size(None)
        .expect("guide formats are uncompressed color");
    render_queue.write_texture(
        texture.as_image_copy(),
        &vec![0u8; (size.x * size.y * bytes_per_pixel) as usize],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.x * bytes_per_pixel),
            rows_per_image: None,
        },
        extent,
    );
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

fn create_dlss_textures(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    size: UVec2,
) -> SolariDlssTextures {
    let (t_depth, depth) = create_guide_texture(
        render_device,
        render_queue,
        size,
        TextureFormat::R32Float,
        "solari_dlss_depth",
    );
    let (t_nr, normal_roughness) = create_guide_texture(
        render_device,
        render_queue,
        size,
        TextureFormat::Rgba16Float,
        "solari_dlss_normal_roughness",
    );
    let (t_diff, diffuse_albedo) = create_guide_texture(
        render_device,
        render_queue,
        size,
        TextureFormat::Rgba8Unorm,
        "solari_dlss_diffuse_albedo",
    );
    let (t_spec, specular_albedo) = create_guide_texture(
        render_device,
        render_queue,
        size,
        TextureFormat::Rgba8Unorm,
        "solari_dlss_specular_albedo",
    );
    let (t_motion, motion) = create_guide_texture(
        render_device,
        render_queue,
        size,
        TextureFormat::Rg16Float,
        "solari_dlss_motion",
    );
    let (t_shd, specular_hit_distance) = create_guide_texture(
        render_device,
        render_queue,
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
pub fn prepare_solari_dlss(
    sdk: Option<Res<SolariDlssSdk>>,
    mode: Res<SolariDlssMode>,
    mut views: Query<
        (
            Entity,
            &ExtractedCamera,
            Option<&mut SolariDlssContext>,
            Option<&SolariDlssTextures>,
            Option<&SolariDlssResolveSlots>,
        ),
        With<SolariCamera>,
    >,
    seam: Option<Res<BindingSeam>>,
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
    // the blit applies the camera exposure before RR reads the view target, so the
    // color is pre-exposed and DLSS must stay exposure-neutral (assumes exposure 1.0).
    // With auto-exposure on top, a dark scene (e.g. a black/unbaked sky) makes DLSS
    // crank the gain and the lit meshes blow out.
    let feature_flags =
        DlssFeatureFlags::LowResolutionMotionVectors | DlssFeatureFlags::HighDynamicRange;
    // Every active mode runs as DLAA (render == display) until upscaling is
    // implemented.
    let perf_quality = DlssPerfQualityMode::Dlaa;

    for (entity, camera, context, textures, resolve_slots) in &mut views {
        let Some(upscaled) = camera.physical_viewport_size else {
            continue;
        };

        // The resolve's per-view heap slots, once (4 buffers + 6 storage images).
        if resolve_slots.is_none() {
            if let Some(seam) = seam.as_deref() {
                commands
                    .entity(entity)
                    .insert(SolariDlssResolveSlots(KernelSlots::new_mixed(
                        seam,
                        &[
                            HeapKind::Buffer,
                            HeapKind::Buffer,
                            HeapKind::Buffer,
                            HeapKind::Buffer,
                            HeapKind::Image,
                            HeapKind::Image,
                            HeapKind::Image,
                            HeapKind::Image,
                            HeapKind::Image,
                            HeapKind::Image,
                        ],
                    )));
            }
        }

        // CRITICAL: the NGX feature lifecycle (create/destroy) must NEVER run while
        // the raw trace is live — it hard-hangs the GPU. So the per-view context is
        // created ONCE, as early as possible — the first frame the viewport is known,
        // before the rt_pipeline has even built or traced — and kept for the view's
        // lifetime regardless of mode. `SolariDlssMode` gates ONLY whether the
        // resolve/render run (see the run conditions) and whether the trace is
        // jittered, never the context. Recreated only on a resolution/quality change
        // (rare).
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
                .insert(create_dlss_textures(
                    &render_device,
                    &render_queue,
                    render_resolution,
                ));
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
/// [`dlss_enabled`]; the context + textures are always present. A raw heap
/// dispatch: the guide textures are first moved to their storage state
/// (GENERAL) through wgpu's tracker, so RR's tracked reads transition out of
/// it with the storage-write dependency that covers the raw writes.
pub fn solari_dlss_resolve(
    view: ViewQuery<(
        &RtOutputBuffer,
        &SolariDlssTextures,
        Option<&SolariDlssResolveSlots>,
    )>,
    resolve: Option<Res<SolariDlssResolve>>,
    seam: Option<Res<BindingSeam>>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (output, textures, slots) = view.into_inner();
    let (Some(resolve), Some(seam), Some(slots)) = (resolve, seam.as_deref(), slots) else {
        return;
    };

    // Move the guide textures to their storage state (GENERAL) through wgpu's
    // tracker BEFORE the raw storage-image writes — the raw dispatch is
    // invisible to wgpu, so the tracked state must match what the writes
    // require, and RR's later tracked reads transition FROM it.
    ctx.command_encoder().transition_resources(
        core::iter::empty(),
        textures._textures.iter().map(|texture| wgpu::TextureTransition {
            // bevy `Texture` → the wrapped `wgpu::Texture`.
            texture: &**texture,
            selector: None,
            state: wgpu::TextureUses::STORAGE_WRITE_ONLY,
        }),
    );

    let blob = resolve.kernel.push_blob(
        "solari_dlss_resolve",
        &[],
        &[
            ("nr_buf", slots.0.buffer(seam, 0, &output.gbuffer[0].buffer)), // normal + roughness
            ("diffuse_buf", slots.0.buffer(seam, 1, &output.gbuffer[1].buffer)), // diffuse + depth
            ("specular_buf", slots.0.buffer(seam, 2, &output.gbuffer[2].buffer)), // specular + hit distance
            ("motion_buf", slots.0.buffer(seam, 3, &output.gbuffer[3].buffer)), // motion
            ("out_depth", slots.0.storage_image(seam, 4, &textures.depth)),
            ("out_normal_roughness", slots.0.storage_image(seam, 5, &textures.normal_roughness)),
            ("out_diffuse", slots.0.storage_image(seam, 6, &textures.diffuse_albedo)),
            ("out_specular", slots.0.storage_image(seam, 7, &textures.specular_albedo)),
            ("out_motion", slots.0.storage_image(seam, 8, &textures.motion)),
            (
                "out_spec_hit_distance",
                slots.0.storage_image(seam, 9, &textures.specular_hit_distance),
            ),
        ],
    );
    // Own command buffer (the ctx encoder just recorded the wgpu transitions —
    // the fork panics if one encoder mixes wgpu work with raw `as_hal_mut`);
    // `add_command_buffer` flushes the transitions ahead of the dispatch.
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("solari_dlss_resolve"),
    });
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // pre-barrier orders prior compute writes before our buffer reads (the
    // trace's own trailing barrier covers the RT-stage gbuffer writes).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &resolve.raw_device;
            let pre = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)];
            dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&pre));
            seam.bind_heaps(cb);
            seam.push_data(cb, &blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, resolve.kernel.pipeline);
            dev.cmd_dispatch(cb, textures.size.x.div_ceil(8), textures.size.y.div_ceil(8), 1);
            // No trailing raw barrier: the resolve writes only the guide
            // images, and RR's tracked reads transition out of the storage
            // state with the storage-write dependency.
        });
    }
    ctx.add_command_buffer(encoder.finish());
}

/// `Core3d` (after `solari_dlss_resolve`, before tonemapping): run DLSS Ray
/// Reconstruction — denoise the noisy view-target color using the guide textures,
/// writing the result back into the view target.
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
