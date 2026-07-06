//! Ray-tracing-pipeline shading path (milestone: primary visibility only).
//!
//! The raw-VK [`RtPipeline`](crate::gpu::rt_pipeline::RtPipeline) records a
//! `cmd_trace_rays` that writes a per-pixel output **storage buffer** (no image
//! layout to fight wgpu over); a small wgpu compute pass ([`blit.wgsl`]) then
//! copies that buffer into the view's HDR storage texture — the same target the
//! reference path tracer writes — so the rest of the frame is unchanged.
//!
//! Selected via [`SolariLighting::RtPipeline`](crate::render::view::SolariLighting).
#![allow(unsafe_code)]

mod rt_camera;
pub use rt_camera::{rt_camera_bind_group_layout, RtCameraPassParams};

use ash::vk;
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_ecs::prelude::*;
use bevy_math::{Mat4, ToRender, UVec4, Vec2, Vec3, Vec4};
use bevy_render::{
    camera::ExtractedCamera,
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, texture_storage_2d},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BufferUsages,
        CachedComputePipelineId, CommandEncoderDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineCache, ShaderStages, StorageTextureAccess,
        TextureFormat, UniformBuffer,
    },
    renderer::{
        raw_vulkan_init::AdditionalVulkanFeatures, RenderContext, RenderDevice, RenderQueue,
        ViewQuery,
    },
    sync_world::RenderEntity,
    texture::{FallbackImage, GpuImage},
    view::{ExtractedView, ViewTarget},
    Extract,
};
use wgpu::hal::api::Vulkan as VkApi;

use crate::bindings::RaytracingSceneBindings;
use crate::ecs_gpu::{GpuSlot, SceneColumns};
use crate::geometry::ClusterMeshManager;
use crate::gpu::allocator::{Allocator, MemoryLocation};
use crate::gpu::extension::RayTracingPipelineFeature;
use crate::gpu::rt_pipeline::{
    RtCamera, RtGeometryAddresses, RtPipeline, RtViewBindings, SolariAnyHitDef, SolariHitGroupDef,
    SolariHitGroupRegistry,
};
use crate::gpu::RawTraceBindable;
use crate::material::{material_sbt_class, MaterialSlots, MaterialTraversalFlags};
use crate::pipelines::SolariPipelines;
use crate::render::atmosphere::{
    AtmosphereSky, SolariAtmosphereGpu, SolariAtmosphereView, SolariAtmosphereVolumesGpu,
};
use crate::render::view_cull::SolariEnvironmentMap;
use crate::render::{CameraReframe, SolariCamera};
use crate::resource_manager::SolariResourceManager;
use crate::transform::{TransformGraph, TransformPropagate};
use bevy_render::extract_resource::ExtractResource;

/// `RenderStartup`: build the RT pipeline (raygen/miss/chit + SBT) if the
/// `VK_KHR_ray_tracing_pipeline` feature is present and the raw-VK allocator
/// exists. Absent otherwise — the run condition then skips the dispatch.
/// Raw `VkDescriptorSetLayout` of the wgpu bind group built from `descriptor`,
/// or `None` if not Vulkan-backed. The pipeline-cache dedups layouts by
/// descriptor, so the handle is stable across frames — safe to bake into the RT
/// pipeline layout once.
fn raw_bgl(
    pipeline_cache: &PipelineCache,
    descriptor: &BindGroupLayoutDescriptor,
) -> Option<vk::DescriptorSetLayout> {
    let layout = pipeline_cache.get_bind_group_layout(descriptor);
    // SAFETY: Vulkan-backed; we read the layout handle, never destroy it.
    unsafe { layout.as_hal::<VkApi>() }.map(|l| l.raw_handle())
}

/// Raw `VkDescriptorSet` of a wgpu bind group, or `None` if not Vulkan-backed.
fn raw_set(bind_group: &bevy_render::render_resource::BindGroup) -> Option<vk::DescriptorSet> {
    // SAFETY: Vulkan-backed; we read the set handle for binding, never destroy it.
    unsafe { bind_group.as_hal::<VkApi>() }.map(|bg| bg.raw_descriptor_set())
}

/// Raw `VkImageView` of a wgpu texture view, or `None` if not Vulkan-backed. Used
/// to bake the environment-cube view into the RT pipeline's set 1 (the wgpu
/// texture owns it; we only read the handle).
fn raw_image_view(view: &bevy_render::render_resource::TextureView) -> Option<vk::ImageView> {
    // SAFETY: Vulkan-backed; we read the image-view handle, never destroy it.
    unsafe { view.as_hal::<VkApi>() }.map(|v| unsafe { v.raw_handle() })
}

/// Raw `VkImage` of a wgpu texture, or `None` if not Vulkan-backed. Needed to
/// transition the storage-written atmosphere cube into a sampleable layout for
/// the raw trace (the RT path never samples it via wgpu, so wgpu doesn't).
fn raw_image(texture: &bevy_render::render_resource::Texture) -> Option<vk::Image> {
    // SAFETY: Vulkan-backed; we read the image handle, never destroy it.
    unsafe { texture.as_hal::<VkApi>() }.map(|t| unsafe { t.raw_handle() })
}

/// Bundled env-image resources, grouped into one [`SystemParam`] to keep the
/// dispatch under bevy's 16-system-param limit.
#[derive(bevy_ecs::system::SystemParam)]
pub(crate) struct RtEnvImages<'w> {
    texture_assets: Res<'w, RenderAssets<GpuImage>>,
    fallback_image: Res<'w, FallbackImage>,
}

/// Per-pixel cost-heatmap debug view (requires `VK_KHR_shader_clock`). When
/// `enabled`, the raygen reads the shader clock around the trace and replaces the
/// shaded color with a colormap of the per-pixel cost; `scale` maps clocks → `[0, 1]`
/// for the colormap (tune per scene). Set it from the main world (e.g. on a
/// keypress) — `bevy_solari::prelude::SolariCostHeatmap`. A no-op when the device
/// lacks shader_clock (the raygen's clock reads compile out).
#[derive(Resource, Clone, Copy, ExtractResource)]
pub struct SolariCostHeatmap {
    pub enabled: bool,
    /// `log2(cycles)` that maps to the colormap midpoint (green) — slide it to the
    /// scene's midrange cost. `-` / `=` shift it. Default 16 (≈ 65k cycles).
    pub center: f32,
    /// Contrast: colormap change per `log2(cycles)` stop around [`center`](Self::center).
    /// Crank it up to push the slowest toward red and the fastest toward blue; `[` / `]`
    /// halve / 1.5× it.
    pub contrast: f32,
}

impl Default for SolariCostHeatmap {
    fn default() -> Self {
        Self {
            enabled: false,
            center: 16.0,
            // ~0.15/stop spreads a ±3-stop range across the colormap.
            contrast: 0.15,
        }
    }
}

/// Debug view that shows each surface's displacement (height) map as grayscale — a validation
/// for the displacement-map wiring BEFORE tessellation actually displaces geometry: confirms
/// which map lands on which surface, the UV mapping, and the height sign. When `enabled`, the
/// opaque closest-hit replaces shading with the sampled height (exposure-compensated) on surfaces
/// that have a `displacement_texture`, and a dim grey elsewhere. Set it from the main world (e.g.
/// on a keypress) — `bevy_solari::prelude::SolariShowDisplacement`.
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
pub struct SolariShowDisplacement {
    pub enabled: bool,
}

/// Debug view that colormaps the per-pixel count of alpha any-hit shader
/// invocations — the OMM-effectiveness view. Cold (blue) = the hit resolved in
/// hardware with no any-hit (opacity-micromap opaque/transparent micro-regions,
/// or plain opaque geometry); hot (red) = many any-hit invocations (unknown
/// micro-regions, or alpha cutouts with no baked OMM). With OMM working, foliage
/// interiors go cold and only the silhouette edges stay warm. Unlike the cost
/// heatmap this needs no `shader_clock`. Mutually exclusive with the other views.
#[derive(Resource, Clone, Copy, ExtractResource)]
pub struct SolariAnyHitHeatmap {
    pub enabled: bool,
    /// Count → colormap scale: `color = cost_heatmap(count * scale)`. Default 0.1
    /// (≈10 any-hits saturates to red). Tune per scene.
    pub scale: f32,
}

impl Default for SolariAnyHitHeatmap {
    fn default() -> Self {
        Self {
            enabled: false,
            scale: 0.1,
        }
    }
}

/// Debug view: flat per-CLUSTER color (a hash of the global cluster id). Shows the
/// cluster decomposition directly — each cluster a distinct hue — so tessellation
/// density (CLAS count) is visible at a glance. Mutually exclusive with the other views.
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
pub struct SolariClusterView {
    pub enabled: bool,
}

/// Debug view: flat per-TRIANGLE color (a hash of cluster id + primitive index). Each
/// (micro-)triangle gets a distinct hue, so view-dependent tessellation LEVEL reads
/// directly as triangle density. Mutually exclusive with the other views.
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
pub struct SolariTriangleView {
    pub enabled: bool,
}

/// Debug view: color each primary hit by whether its SHADING normal faces the
/// camera — blue toward the viewer, red away (inverted / back-wound), brightness
/// = facing magnitude so grazing reads dark. On watertight, correctly-wound
/// geometry everything is blue; red patches are the bug. Mutually exclusive with
/// the other views.
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
pub struct SolariNormalFacing {
    pub enabled: bool,
}

/// Debug/feature inputs bundled into one [`SystemParam`] to keep the dispatch under
/// bevy's 16-system-param limit: the device feature set + the debug-view toggles.
#[derive(bevy_ecs::system::SystemParam)]
pub(crate) struct RtDebug<'w> {
    additional: Res<'w, AdditionalVulkanFeatures>,
    cost_heatmap: Option<Res<'w, SolariCostHeatmap>>,
    anyhit_heatmap: Option<Res<'w, SolariAnyHitHeatmap>>,
    show_displacement: Option<Res<'w, SolariShowDisplacement>>,
    cluster_view: Option<Res<'w, SolariClusterView>>,
    triangle_view: Option<Res<'w, SolariTriangleView>>,
    normal_facing: Option<Res<'w, SolariNormalFacing>>,
}

/// Material routing inputs for the SBT, bundled into one [`SystemParam`] to keep
/// the dispatch under bevy's 16-system-param limit. `slots` sizes the per-material
/// hit records; `traversal_flags` carries the glass bit that selects each record's
/// hit-group class.
#[derive(bevy_ecs::system::SystemParam)]
pub(crate) struct RtMaterials<'w> {
    slots: Res<'w, MaterialSlots>,
    traversal_flags: Res<'w, MaterialTraversalFlags>,
}

impl RtMaterials<'_> {
    /// Number of material slots — the count of per-material SBT hit records.
    fn len(&self) -> u32 {
        self.slots.len()
    }

    /// Per-material-slot SBT hit-group class (opaque/glass), slot-aligned with the
    /// `materials[]` array — the routing key the SBT bakes into each material's hit
    /// record so glass instances reach `chit_glass` (see `material_sbt_class`).
    fn sbt_classes(&self) -> Vec<u32> {
        let flags = self.traversal_flags.buffer.get();
        (0..self.slots.len() as usize)
            .map(|slot| material_sbt_class(flags.get(slot).copied().unwrap_or(0)))
            .collect()
    }
}

/// The wgpu compute pipeline + layout copying the RT output buffer to the view.
#[derive(Resource)]
pub struct RtBlit {
    pub layout: BindGroupLayoutDescriptor,
    pub pipeline: CachedComputePipelineId,
}

/// `RenderStartup`: build the blit pipeline (independent of `SolariPipelines`).
pub fn init_rt_blit(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "rt_blit_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0: rt_output (array<vec4<f32>>)
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly), // 1: view_output
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("rt_blit_pipeline".into()),
        layout: vec![layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "blit.wgsl"),
        shader_defs: vec![],
        entry_point: Some("blit".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    commands.insert_resource(RtBlit { layout, pipeline });
}

/// Per-view output: a `width*height` `vec4<f32>` storage buffer the raygen shader
/// writes (raw) and the blit reads (wgpu). A `wgpu::Buffer` from [`Allocator`] so
/// we hold both its raw `VkBuffer` (for the RT descriptor) and the wgpu handle.
#[derive(Component)]
pub struct RtOutputBuffer {
    pub buffer: bevy_render::render_resource::Buffer,
    pub raw: vk::Buffer,
    pub size: u64,
    pub pixels: u32,
    /// This view's `RtCamera` GPU buffer — filled by the `rt_camera` compute pass
    /// (or a CPU `write_buffer`) each frame and bound as set-1 binding 1. Held with
    /// its raw `VkBuffer` (bridged into the RT descriptor) + the wgpu handle (the
    /// compute-pass bind group / `write_buffer` target). Reallocated with the output
    /// so its raw handle is refreshed whenever `create_view_bindings` rebuilds.
    pub camera_buffer: bevy_render::render_resource::Buffer,
    pub camera_raw: vk::Buffer,
    /// Persistent previous camera basis for the `rt_camera` pass's GPU-maintained DLSS
    /// motion vectors (read-then-write each frame). Zero-cleared on creation so its
    /// `valid` flag starts 0. Only the compute touches it — never bridged into the RT
    /// descriptor.
    pub camera_prev_buffer: bevy_render::render_resource::Buffer,
    /// DLSS ray-reconstruction guide G-buffers — normal+roughness, diffuse+depth,
    /// specular+hit-distance, and motion vectors — each `pixels` × `vec4<f32>`,
    /// allocated and reallocated alongside the color output. Written by the
    /// closest-hit (chit-direct), read by the resolve.
    #[cfg(feature = "dlss")]
    pub gbuffer: [RtGbuffer; 4],
}

/// Per-view cache of last frame's **unjittered** clip-from-world, for the DLSS
/// motion-vector guide. Written at the end of each `rt_pipeline` dispatch, read as
/// "previous" the next frame (absent on the first frame ⇒ zero motion). Defined
/// unconditionally so the `ViewQuery` tuple needn't cfg-gate a single element; only
/// the `dlss` path ever inserts or reads it.
#[derive(Component)]
pub struct RtPrevViewProj {
    pub clip_from_world: Mat4,
}

/// Per-view sub-pixel camera jitter (pixels) for DLSS temporal accumulation. Set by
/// `prepare_solari_dlss` from the RR context's `suggested_jitter`; `rt_pipeline`
/// folds it into the primary ray, and `solari_dlss_render` passes its negation as
/// the RR `jitter_offset`. Defined unconditionally so the `ViewQuery` tuple needn't
/// cfg-gate an element; absent (⇒ zero jitter) unless DLSS is active.
#[derive(Component, Default)]
pub struct SolariDlssJitter {
    pub offset: Vec2,
}

/// One DLSS ray-reconstruction guide buffer: a `pixels` × `vec4<f32>` GPU storage
/// buffer the closest-hit writes. Held with its raw `VkBuffer` (for the RT set-1
/// descriptor) and the wgpu handle (for the resolve pass).
#[cfg(feature = "dlss")]
pub struct RtGbuffer {
    pub buffer: bevy_render::render_resource::Buffer,
    pub raw: vk::Buffer,
}

/// Byte size of `rt_camera.wgsl`'s `PrevCamera` (`mat4x4` + 3×`f64` previous origin +
/// `u32`, std430-padded).
const RT_PREV_CAMERA_SIZE: u64 = 96;

/// `Prepare`: (re)allocate the per-view RT output buffer to fit the viewport.
pub fn prepare_rt_output(
    views: Query<(Entity, &ExtractedCamera, Option<&RtOutputBuffer>), With<SolariCamera>>,
    allocator: Option<Res<Allocator>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut commands: Commands,
) {
    let Some(allocator) = allocator else {
        return;
    };
    for (entity, camera, existing) in &views {
        let Some(viewport) = camera.physical_viewport_size else {
            continue;
        };
        let pixels = viewport.x * viewport.y;
        if existing.is_some_and(|b| b.pixels == pixels) {
            continue;
        }
        let size = pixels as u64 * 16;
        let buffer = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferUsages::STORAGE,
            size,
            MemoryLocation::GpuOnly,
            "rt_output_buffer",
        );
        // SAFETY: buffer is Vulkan-backed (Allocator only builds VkBuffers).
        let raw = unsafe { buffer.as_hal::<VkApi>() }
            .map(|b| b.raw_handle())
            .expect("rt_output buffer must be Vulkan-backed");
        // DLSS guide G-buffers: same size + lifetime as the color output, reallocated
        // with it (the `pixels` early-out above covers a viewport resize).
        #[cfg(feature = "dlss")]
        let gbuffer: [RtGbuffer; 4] = core::array::from_fn(|_| {
            let buffer = allocator.create_buffer(
                &render_device,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                BufferUsages::STORAGE,
                size,
                MemoryLocation::GpuOnly,
                "rt_gbuffer",
            );
            // SAFETY: Vulkan-backed (Allocator only builds VkBuffers).
            let raw = unsafe { buffer.as_hal::<VkApi>() }
                .map(|b| b.raw_handle())
                .expect("rt_gbuffer must be Vulkan-backed");
            RtGbuffer {
                buffer: buffer.into(),
                raw,
            }
        });
        // The GPU camera buffer (constant size — one `RtCamera`). STORAGE (the
        // `rt_camera` compute writes it) | UNIFORM (the RT trace reads it) | COPY_DST
        // (the CPU-fill fallback `write_buffer`). Recreated alongside the output so its
        // raw handle is fresh whenever the view bindings rebuild.
        let camera_buffer = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            size_of::<RtCamera>() as u64,
            MemoryLocation::GpuOnly,
            "rt_camera_buffer",
        );
        // SAFETY: Vulkan-backed (Allocator only builds VkBuffers).
        let camera_raw = unsafe { camera_buffer.as_hal::<VkApi>() }
            .map(|b| b.raw_handle())
            .expect("rt_camera buffer must be Vulkan-backed");
        // Persistent previous-basis buffer for GPU motion vectors — `rt_camera.wgsl`'s
        // `PrevCamera` (mat4 + `valid`, padded to 80 B). Zero-cleared so `valid` starts 0
        // (frame 1 ⇒ zero motion, not a read of uninitialized memory).
        let camera_prev_buffer = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            RT_PREV_CAMERA_SIZE,
            MemoryLocation::GpuOnly,
            "rt_camera_prev_buffer",
        );
        let camera_prev_buffer: bevy_render::render_resource::Buffer = camera_prev_buffer.into();
        render_queue.write_buffer(&camera_prev_buffer, 0, &[0u8; RT_PREV_CAMERA_SIZE as usize]);
        commands.entity(entity).insert(RtOutputBuffer {
            buffer: buffer.into(),
            raw,
            size,
            pixels,
            camera_buffer: camera_buffer.into(),
            camera_raw,
            camera_prev_buffer,
            #[cfg(feature = "dlss")]
            gbuffer,
        });
    }
}

/// The `SolariCamera`'s transform-table slot (`GpuSlot<TransformGraph>`), extracted
/// onto the render-world view so the `rt_camera` compute pass can read
/// `world[camera_slot]`. Absent until the camera's slot is allocated (the first frame
/// or two), where the pass falls back to the CPU-derived camera.
#[derive(Component, Clone, Copy)]
pub struct RtCameraSlot(pub u32);

/// `ExtractSchedule`: copy each `SolariCamera`'s transform-table slot index onto its
/// render-world view entity. The slot lives on the main-world camera (assigned in
/// `PostUpdate`); the render-world `rt_camera` pass needs it to index `world[…]`.
pub fn extract_rt_camera_slot(
    mut commands: Commands,
    cameras: Extract<Query<(&RenderEntity, &GpuSlot<TransformGraph>), With<SolariCamera>>>,
) {
    for (render_entity, slot) in &cameras {
        commands
            .entity(render_entity.id())
            .insert(RtCameraSlot(slot.index()));
    }
}

/// CPU-authored inputs to the `rt_camera` compute pass (projection + per-frame
/// scalars). The transform-derived matrices come from `world[camera_slot]` on the GPU.
struct RtCameraGpuInputs {
    clip_from_view: Mat4,
    /// Floating-origin recenter rebase for the previous basis (identity + `reframe_active`
    /// false on ordinary frames). The GPU keeps the previous `clip_from_world` itself.
    reframe_prev_from_current: Mat4,
    reframe_active: bool,
    frame: UVec4,
    sky: Vec4,
    jitter: Vec4,
    /// `.x` = time (s, wrapped), `.y` = pixel ray-cone tan; see `RtCamera::misc`.
    misc: Vec4,
    /// World→bake sky quaternion (xyzw); see `RtCamera::sky_frame`.
    sky_frame: Vec4,
    /// Atmosphere-volume buffer address bits + count; see `RtCamera::atmo`.
    atmo: Vec4,
    exposure: f32,
}

/// Dispatch the `rt_camera` compute pass to fill `output.camera_buffer` from the
/// camera's transform-table slot. Returns `false` (⇒ caller does the CPU fallback)
/// when the pass can't run this frame: unsupported device, pipeline/layout not ready,
/// or the camera's slot isn't allocated / in range yet.
#[allow(clippy::too_many_arguments)]
fn try_dispatch_rt_camera(
    ctx: &mut RenderContext,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    pipeline_cache: &PipelineCache,
    pipelines: Option<&SolariPipelines>,
    resources: Option<&SolariResourceManager>,
    propagate: Option<&TransformPropagate>,
    camera_slot: Option<&RtCameraSlot>,
    output: &RtOutputBuffer,
    inputs: RtCameraGpuInputs,
) -> bool {
    let (Some(pipelines), Some(resources), Some(propagate), Some(slot)) =
        (pipelines, resources, propagate, camera_slot)
    else {
        return false; // cold start (slot extracted a frame after the camera spawns).
    };
    let node_count = propagate.node_count();
    // Slot not yet propagated → CPU fallback (correct for a root camera; a transient
    // first-frame-or-two for a childed one, before its slot lands).
    if slot.0 >= node_count {
        return false;
    }
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.rt_camera) else {
        return false; // still compiling (or failed — the cache logs a compile error).
    };

    let mut params = UniformBuffer::from(RtCameraPassParams {
        clip_from_view: inputs.clip_from_view,
        view_from_clip: inputs.clip_from_view.inverse(),
        reframe_prev_from_current: inputs.reframe_prev_from_current,
        frame: inputs.frame,
        sky: inputs.sky,
        jitter: inputs.jitter,
        misc: inputs.misc,
        sky_frame: inputs.sky_frame,
        atmo: inputs.atmo,
        camera_slot: slot.0,
        node_count,
        exposure: inputs.exposure,
        valid: 1,
        reframe_active: inputs.reframe_active as u32,
    });
    params.write_buffer(render_device, render_queue);
    let Some(params_binding) = params.binding() else {
        return false;
    };

    let layout = pipeline_cache.get_bind_group_layout(&resources.rt_camera);
    let bind_group = render_device.create_bind_group(
        "rt_camera",
        &layout,
        &BindGroupEntries::sequential((
            propagate.current_world().as_entire_binding(),
            params_binding,
            output.camera_buffer.as_entire_binding(),
            output.camera_prev_buffer.as_entire_binding(),
            propagate.world_abs_t().as_entire_binding(),
        )),
    );

    // Recorded on the ctx encoder → flushed (and its write made visible by the trace's
    // pre-barrier) before the trace's own command buffer runs.
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("rt_camera"),
            timestamp_writes: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(1, 1, 1);
    true
}

// Built-in surfaces, registered (in order, by `SolariPlugin`) like any downstream one.

/// Default opaque surface (class 0) — BRDF + NEE + alpha-cutout any-hit.
pub struct OpaqueSurface;

impl crate::SolariMaterial for OpaqueSurface {
    fn hit_group() -> SolariHitGroupDef {
        SolariHitGroupDef {
            label: "opaque",
            closest_hit_wgsl: include_str!("chit_opaque.wgsl"),
            closest_hit_file: "chit_opaque.wgsl",
            closest_hit_entry: "chit_opaque",
            any_hit: Some(SolariAnyHitDef {
                wgsl: include_str!("ahit_alpha.wgsl"),
                file: "ahit_alpha.wgsl",
                entry: "ahit_alpha",
            }),
            composable_modules: &[],
        }
    }
}

/// Glass surface (class 1) — Fresnel reflect/refract; routed by `specular_transmission > 0`.
pub struct GlassSurface;

impl crate::SolariMaterial for GlassSurface {
    fn hit_group() -> SolariHitGroupDef {
        SolariHitGroupDef {
            label: "glass",
            closest_hit_wgsl: include_str!("chit_glass.wgsl"),
            closest_hit_file: "chit_glass.wgsl",
            closest_hit_entry: "chit_glass",
            any_hit: None,
            composable_modules: &[],
        }
    }
}

/// Hair surface (class 2) — Chiang fiber BSDF / LSS bark; reached via the reserved hair record.
pub struct HairSurface;

impl crate::SolariMaterial for HairSurface {
    fn hit_group() -> SolariHitGroupDef {
        SolariHitGroupDef {
            label: "hair",
            closest_hit_wgsl: include_str!("chit_hair.wgsl"),
            closest_hit_file: "chit_hair.wgsl",
            closest_hit_entry: "chit_hair",
            any_hit: None,
            composable_modules: &[],
        }
    }
}

/// Portal surface (class 3) — teleports rays; pair the instance with a [`SolariPortal`](crate::bindings::SolariPortal).
pub struct PortalSurface;

impl crate::SolariMaterial for PortalSurface {
    fn hit_group() -> SolariHitGroupDef {
        SolariHitGroupDef {
            label: "portal",
            closest_hit_wgsl: include_str!("chit_portal.wgsl"),
            closest_hit_file: "chit_portal.wgsl",
            closest_hit_entry: "chit_portal",
            any_hit: None,
            composable_modules: &[],
        }
    }
}

/// `RenderGraph`: cast primary rays through the RT pipeline and blit the result into
/// the view. Lazily builds the RT pipeline on the first frame the scene + columns bind
/// groups (and their layouts) exist — its layout must match wgpu's exact descriptor
/// set layouts, which only exist once those are built.
pub(crate) fn rt_pipeline(
    view: ViewQuery<(
        &ExtractedView,
        &ExtractedCamera,
        &ViewTarget,
        &RtOutputBuffer,
        Option<&RtViewBindings>,
        Option<&SolariAtmosphereView>,
        Option<&SolariEnvironmentMap>,
        Option<&RtPrevViewProj>,
        Option<&SolariDlssJitter>,
        Option<&CameraReframe>,
        Option<&RtCameraSlot>,
    )>,
    rt: Option<Res<RtPipeline>>,
    rt_blit: Res<RtBlit>,
    allocator: Option<Res<Allocator>>,
    debug: RtDebug,
    scene_bindings: Res<RaytracingSceneBindings>,
    scene_columns: Res<SceneColumns>,
    // Tupled into one system param (the system is at bevy's 16-param ceiling).
    geometry_res: (
        Option<Res<ClusterMeshManager>>,
        Option<Res<crate::geometry::tess_classify::TessClassify>>,
        Option<Res<SolariHitGroupRegistry>>,
        Option<Res<crate::accel::deform::Deform>>,
    ),
    materials: RtMaterials,
    // Tupled: baked sky cube + atmosphere GPU state (sky_frame quat) +
    // world-space atmosphere volumes (address for raygen's march).
    atmosphere_res: (
        Option<Res<AtmosphereSky>>,
        Option<Res<SolariAtmosphereGpu>>,
        Option<Res<SolariAtmosphereVolumesGpu>>,
    ),
    env_images: RtEnvImages,
    pipeline_cache: Res<PipelineCache>,
    // Tupled to stay under bevy's 16-param system ceiling. `SolariPipelines`/
    // `SolariResourceManager`/`TransformPropagate` drive the `rt_camera` compute pass
    // (`Option` — absent on unsupported devices ⇒ CPU-derived camera fallback).
    render_res: (
        Res<RenderDevice>,
        Res<RenderQueue>,
        Option<Res<SolariPipelines>>,
        Option<Res<SolariResourceManager>>,
        Option<Res<TransformPropagate>>,
        Res<bevy_time::Time>,
    ),
    mut frame_counter: Local<u32>,
    mut commands: Commands,
    mut ctx: RenderContext,
) {
    let (render_device, render_queue, solari_pipelines, solari_resources, transform_propagate, time) =
        render_res;
    let (cluster_mesh_manager, tess_classify, hit_group_registry, deform) = geometry_res;
    let (atmosphere_sky, atmosphere_gpu, atmosphere_volumes) = atmosphere_res;
    let view_entity = view.entity();
    let (
        view,
        camera,
        view_target,
        output,
        view_bindings,
        atmosphere_view,
        environment_map,
        prev_view_proj,
        dlss_jitter,
        reframe,
        camera_slot,
    ) = view.into_inner();

    // Environment cube the miss shader samples (same priority as the megakernel):
    // the baked atmosphere cube if this view has one, else the view's skybox image,
    // else the fallback cube. The atmosphere cube is a STORAGE image (GENERAL) the
    // RT path never samples via wgpu, so we transition it ourselves; the skybox /
    // fallback are wgpu-sampled (read-optimal already), so no transition.
    let (environment_map_view, environment_map_image) =
        match atmosphere_view.and(atmosphere_sky.as_ref()) {
            Some(sky) => (&sky.cube_view, raw_image(&sky.texture)),
            None => {
                let view = environment_map
                    .and_then(|env| env_images.texture_assets.get(&env.image))
                    .map(|image| &image.texture_view)
                    .unwrap_or(&env_images.fallback_image.cube.texture_view);
                (view, None)
            }
        };
    // Match the megakernel (view_cull.rs): the baked atmosphere cube is already
    // physical radiance (brightness 1.0); otherwise the skybox's raw cd/m²; else 0
    // (no sky ⇒ miss stays at the clear color). Brightness stays 0 while the skybox
    // image is still loading — the stand-in is the WHITE fallback cube, and lighting
    // it up would flash the whole sky white until the real cubemap lands.
    let environment_brightness = if atmosphere_view.is_some() {
        1.0
    } else {
        environment_map
            .filter(|env| env_images.texture_assets.get(&env.image).is_some())
            .map_or(0.0, |env| env.brightness)
    };

    // Scene (set 0) + columns (set 2) bind groups — built each frame by the
    // binder / scene-columns prepare. Both required: they carry the TLAS,
    // geometry, materials, lights, textures, and transforms the chits resolve.
    let (Some(scene_bg), Some(columns_bg), Some(columns_layout_desc)) = (
        scene_bindings.bind_group.as_ref(),
        scene_columns.bind_group.as_ref(),
        scene_columns.layout(),
    ) else {
        return;
    };

    // Per-material-slot SBT hit-group class the pipeline bakes into each material's
    // hit record — the routing key that sends glass instances to `chit_glass`
    // instead of every hit landing on `chit_opaque`. Derived from the already
    // CPU-computed `MaterialTraversalFlags` (glass bit) and slot-aligned with
    // `materials[]`, so `classes[slot]` is the class of the material in that record.
    // Recomputed each frame; the pipeline rebuilds below when it changes.
    let material_classes = materials.sbt_classes();

    // Lazily build the view-independent RT pipeline once the scene + columns
    // layouts exist (its pipeline layout bakes in their raw VkDescriptorSetLayouts)
    // and materials are present (the SBT sizes one hit record per material slot).
    // The per-view set 1 (output/camera/env) is built separately below. Inserted
    // via commands → live next frame.
    let Some(rt) = rt else {
        if debug.additional.has::<RayTracingPipelineFeature>() && materials.len() > 0 {
            if let (Some(allocator), Some(scene_layout), Some(columns_layout), Some(registry)) = (
                allocator.as_deref(),
                raw_bgl(&pipeline_cache, &scene_bindings.bind_group_layout),
                raw_bgl(&pipeline_cache, columns_layout_desc),
                hit_group_registry.as_deref(),
            ) {
                if let Some(built) = RtPipeline::new(
                    allocator,
                    scene_layout,
                    columns_layout,
                    &material_classes,
                    &registry.groups,
                ) {
                    commands.insert_resource(built);
                }
            }
        }
        return;
    };

    // Materials can stream in after the pipeline was first built (the SBT bakes
    // one hit record per material slot at build time + headroom). Rebuild when
    // either the live count has outgrown those records (an instance routing to a
    // slot past the hit region would read out of bounds) OR a material's CLASS
    // changed (glass loading/unloading, a live edit) — the record's hit-group
    // handle is baked, so the new class only takes effect after a rebuild. Drain
    // the GPU first so dropping the old pipeline (when this frame's `remove`
    // command applies) can't free a `VkPipeline` a still-executing trace uses.
    if materials.len() > rt.capacity() || rt.classes_changed(&material_classes) {
        let _ = render_device
            .wgpu_device()
            .poll(wgpu::PollType::wait_indefinitely());
        commands.remove_resource::<RtPipeline>();
        return;
    }

    // Build/rebuild THIS view's set-1 bindings (output buffer @0, camera UBO @1,
    // env cube @2). Per-view so split-screen views each trace into their own
    // output with their own camera/env. Built lazily once the env cube is ready
    // (baked into the set once — building before it exists would freeze the
    // fallback cube in as the sky) and rebuilt if the view's output buffer was
    // reallocated (viewport resize) OR the env view changed (a skybox that
    // finished loading / got swapped), since the set is written once, never updated.
    let current_env_view = raw_image_view(environment_map_view);
    let view_bindings = match view_bindings {
        Some(vb)
            if vb.output_buffer() == output.raw
                && Some(vb.env_map_view()) == current_env_view =>
        {
            vb
        }
        existing => {
            // Ready = the declared sky actually exists: a pending atmosphere bake or a
            // still-loading skybox image would bake the (white) fallback cube in forever.
            let env_ready = if atmosphere_view.is_some() {
                atmosphere_sky.is_some()
            } else {
                environment_map
                    .is_none_or(|env| env_images.texture_assets.get(&env.image).is_some())
            };
            if env_ready {
                if let (Some(allocator), Some(env_view)) = (allocator.as_deref(), current_env_view)
                {
                    // Resize rebuild: the old bindings point at a freed output
                    // buffer (RtOutputBuffer realloc'd). Drain the GPU so dropping
                    // the stale component can't free an in-flight set/camera UBO.
                    if existing.is_some() {
                        let _ = render_device
                            .wgpu_device()
                            .poll(wgpu::PollType::wait_indefinitely());
                    }
                    // DLSS guide G-buffers (empty without the feature) — bound into
                    // set 1 alongside the color output, same size, same lifetime.
                    #[cfg(feature = "dlss")]
                    let gbuffers = [
                        (output.gbuffer[0].raw, output.size),
                        (output.gbuffer[1].raw, output.size),
                        (output.gbuffer[2].raw, output.size),
                        (output.gbuffer[3].raw, output.size),
                    ];
                    #[cfg(not(feature = "dlss"))]
                    let gbuffers: [(vk::Buffer, u64); 0] = [];
                    if let Some(built) = rt.create_view_bindings(
                        allocator,
                        output.raw,
                        output.size,
                        output.camera_raw,
                        &gbuffers,
                        env_view,
                        environment_map_image,
                    ) {
                        commands.entity(view_entity).insert(built);
                    }
                }
            }
            return;
        }
    };

    let (Some(viewport), Some(blit_pipeline)) = (
        camera.physical_viewport_size,
        pipeline_cache.get_compute_pipeline(rt_blit.pipeline),
    ) else {
        return;
    };

    // Raw scene + columns descriptor sets (sets 0 and 2).
    let (Some(scene_set), Some(columns_set)) = (raw_set(scene_bg), raw_set(columns_bg)) else {
        return;
    };

    // Camera inputs for the raygen unprojection + RNG frame seed.
    // `view.clip_from_world` is usually None; derive world_from_clip from the
    // always-present view transform + projection:
    // world_from_clip = world_from_view * inverse(clip_from_view).
    // The trace consumes ORIGIN-RELATIVE space and the primary camera IS the
    // origin, so this CPU fallback basis is rotation-only (translation 0) —
    // `view.world_from_view` is the camera's ABSOLUTE world.
    let world_from_view = Mat4::from_quat(view.world_from_view.rotation().to_render());
    let view_from_world = world_from_view.inverse();
    let world_from_clip = world_from_view * view.clip_from_view.inverse();
    // Unjittered clip-from-world for the DLSS motion-vector guide; "previous" comes
    // from the per-view cache (absent on the first frame ⇒ current ⇒ zero motion).
    // Computed unconditionally (a few matrix ops) so the dlss/non-dlss camera path
    // stays identical; only the chit (under `#ifdef SOLARI_DLSS`) reads these.
    let clip_from_world = view.clip_from_view * view_from_world;
    // Cached previous clip-from-world (current ⇒ zero motion on the first frame). On a
    // floating-origin frame handoff the cache is still in the OLD origin basis, so the
    // bridge-supplied reframe re-expresses it in the new basis (post-multiply by
    // prev_world_from_current_world) for this one frame — keeping motion vectors
    // continuous across the discontinuity instead of dropping history.
    let mut prev_clip_from_world = prev_view_proj.map_or(clip_from_world, |p| p.clip_from_world);
    if let Some(reframe) = reframe.filter(|r| !r.is_identity()) {
        prev_clip_from_world *= reframe.prev_from_current;
    }
    let camera_inputs = RtCamera {
        inverse_view_proj: world_from_clip.to_cols_array(),
        view_from_world: view_from_world.to_cols_array(),
        clip_from_world: clip_from_world.to_cols_array(),
        prev_clip_from_world: prev_clip_from_world.to_cols_array(),
        // .xyz = ray origin; .w = camera exposure (raygen scales final radiance by
        // it, like the megakernel's `radiance *= view.exposure`).
        camera_position: Vec3::ZERO.extend(camera.exposure).to_array(),
        // .x = frame index (RNG seed); .y = SER material-hint bits =
        // ceil(log2(material_count)), the number of low bits of the SBT-record-index
        // hint reorderThread should sort by (driver clamps to its own max).
        frame: [
            *frame_counter,
            (u32::BITS - materials.len().max(1).leading_zeros()),
            // .z = debug view selector: 0 = normal, 1 = cost (clock) heatmap (needs
            // SOLARI_SHADER_CLOCK), 2 = any-hit-count heatmap (OMM effectiveness),
            // 3 = per-cluster color, 4 = per-triangle color, 5 = normal-facing
            // (blue toward camera / red away). The view dropdown keeps these
            // mutually exclusive.
            if debug.cost_heatmap.as_deref().is_some_and(|h| h.enabled) {
                1u32
            } else if debug.anyhit_heatmap.as_deref().is_some_and(|h| h.enabled) {
                2u32
            } else if debug.cluster_view.as_deref().is_some_and(|v| v.enabled) {
                3u32
            } else if debug.triangle_view.as_deref().is_some_and(|v| v.enabled) {
                4u32
            } else if debug.normal_facing.as_deref().is_some_and(|v| v.enabled) {
                5u32
            } else {
                0u32
            },
            // .w = displacement debug view (1 = on); the opaque chit shows each
            // surface's height map (grayscale) to validate the displacement wiring.
            debug
                .show_displacement
                .as_deref()
                .is_some_and(|d| d.enabled) as u32,
        ],
        // .x = sky brightness; .yzw = clear color (black for bevy_city).
        sky: [environment_brightness, 0.0, 0.0, 0.0],
        // Sub-pixel jitter (pixels) for DLSS temporal accumulation; zero without an
        // active DLSS context (the trace renders a fresh frame with no accumulator,
        // so an unaccumulated jitter would only shimmer).
        jitter: {
            let j = dlss_jitter.map_or(Vec2::ZERO, |j| j.offset);
            // .zw = cost-heatmap log2 center + contrast (read by the raygen colormap).
            // The any-hit heatmap reuses .z as its count→colormap scale (only one
            // debug view is active at a time, so the slot is unambiguous).
            let hm = debug.cost_heatmap.as_deref();
            let anyhit = debug.anyhit_heatmap.as_deref();
            let (center, contrast) = if anyhit.is_some_and(|h| h.enabled) {
                (anyhit.map_or(0.1, |h| h.scale), 0.0)
            } else {
                (hm.map_or(0.0, |h| h.center), hm.map_or(0.0, |h| h.contrast))
            };
            [j.x, j.y, center, contrast]
        },
        // .x = time for animated surfaces; .y = per-pixel ray-cone tangent for
        // footprint-based shading LOD: full vertical FOV spans `viewport.y` pixels,
        // so one pixel subtends 2·tan(fov/2)/height. `clip_from_view[1][1]` is
        // 1/tan(fov_y/2) for a perspective projection (garbage-but-harmless for ortho).
        misc: [
            time.elapsed_secs_wrapped(),
            2.0 / (view.clip_from_view.y_axis.y * viewport.y as f32),
            0.0,
            0.0,
        ],
        // World→bake sky rotation (identity unless a spherical-planet
        // atmosphere set one) — the miss shader rotates cube sample dirs.
        sky_frame: atmosphere_gpu
            .as_deref()
            .map_or([0.0, 0.0, 0.0, 1.0], |a| a.sky_frame.to_array()),
        // Atmosphere volumes: device address (bit-preserved through f32) +
        // live count. Zero count ⇒ raygen skips the march entirely.
        atmo: atmosphere_volumes.as_deref().map_or([0.0; 4], |v| {
            [
                f32::from_bits(v.address as u32),
                f32::from_bits((v.address >> 32) as u32),
                f32::from_bits(v.count),
                0.0,
            ]
        }),
    };
    // Fill this view's GPU camera buffer (bound at a constant dynamic offset 0). The
    // GPU-authoritative path derives the basis from `world[camera_slot]` in the
    // `rt_camera` compute pass — so the camera is composed on the GPU through any
    // hierarchy + floating-origin offset, same-frame, with no dependence on the CPU
    // `GlobalTransform`. Falls back to the CPU-derived `camera_inputs` (a `write_buffer`
    // TRANSFER write) when the pass isn't ready (device unsupported, pipeline still
    // compiling, or the camera's slot not yet allocated) — both fills are covered by
    // the trace's existing pre-barrier (`{TRANSFER,SHADER}_WRITE → SHADER_READ`).
    let gpu_camera = try_dispatch_rt_camera(
        &mut ctx,
        &render_device,
        &render_queue,
        &pipeline_cache,
        solari_pipelines.as_deref(),
        solari_resources.as_deref(),
        transform_propagate.as_deref(),
        camera_slot,
        output,
        {
            // The recenter rebase the GPU applies to its stored previous basis (mirrors
            // the CPU path's `prev *= reframe.prev_from_current`), or identity/off.
            let active = reframe.filter(|r| !r.is_identity());
            RtCameraGpuInputs {
                clip_from_view: view.clip_from_view,
                reframe_prev_from_current: active.map_or(Mat4::IDENTITY, |r| r.prev_from_current),
                reframe_active: active.is_some(),
                frame: UVec4::from_array(camera_inputs.frame),
                sky: Vec4::from_array(camera_inputs.sky),
                jitter: Vec4::from_array(camera_inputs.jitter),
                misc: Vec4::from_array(camera_inputs.misc),
                sky_frame: Vec4::from_array(camera_inputs.sky_frame),
                atmo: Vec4::from_array(camera_inputs.atmo),
                exposure: camera.exposure,
            }
        },
    );
    if !gpu_camera {
        render_queue.write_buffer(&output.camera_buffer, 0, bytemuck::bytes_of(&camera_inputs));
    }
    let camera_dynamic_offset = 0u32;
    *frame_counter = frame_counter.wrapping_add(1);
    // Cache this frame's unjittered clip-from-world as next frame's "previous".
    commands
        .entity(view_entity)
        .insert(RtPrevViewProj { clip_from_world });

    // Bindless geometry addresses for the chit's `physical_load` resolve. Both come
    // from stable-address (`RawTraceBindable`) buffers, so the captured addresses
    // stay valid for any in-flight trace — `trace_device_address` won't compile on a
    // reallocating buffer. The materials address is captured at bind time (binder.rs).
    if let Some(cluster_mesh_manager) = cluster_mesh_manager.as_deref() {
        // Smooth-tess metadata table address (0 when the smooth path is off → the
        // closest-hit falls back to the facet normal). The GPU-classify path's per-part
        // metadata (real UVs + smooth normals) reached via `geometry_addresses.tess_clusters`.
        let tess_clusters = tess_classify.as_ref().map_or(0, |c| c.gen_attrs_meta_addr);
        view_bindings.set_geometry_addresses(&RtGeometryAddresses {
            vertex_packed: cluster_mesh_manager.vertex_packed.trace_device_address().get(),
            vertex_positions: cluster_mesh_manager.vertex_positions.trace_device_address().get(),
            materials: scene_bindings.materials_device_address.get(),
            material_stride: crate::bindings::GPU_MATERIAL_SIZE,
            _pad: 0,
            tess_clusters,
            vertex_custom: cluster_mesh_manager.vertex_custom.trace_device_address().get(),
            deform_normals: deform.as_ref().map_or(0, |d| d.normals_addr.get()),
            deform_tangents: deform.as_ref().map_or(0, |d| d.tangents_addr.get()),
            animated_table: deform.as_ref().map_or(0, |d| d.animated_table_addr.get()),
        });
    }

    // The raw cmd_trace_rays must go in its OWN command buffer — wgpu-core
    // panics if a single encoder mixes wgpu passes (the blit) with raw
    // `as_hal_mut`. `add_command_buffer` flushes any pending ctx work then
    // appends this, so the trace is submitted before the blit (same queue); the
    // trace's trailing SHADER_WRITE→READ barrier covers the blit's read.
    let mut trace_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("rt_pipeline_trace"),
    });
    // SAFETY: Vulkan backend; descriptors were just updated for this frame; the
    // scene/columns sets match the layouts the pipeline was built with.
    unsafe {
        trace_encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("rt_pipeline requires the Vulkan backend");
            let command_buffer = hal_encoder.raw_handle();
            rt.trace(
                command_buffer,
                scene_set,
                view_bindings,
                columns_set,
                camera_dynamic_offset,
                viewport.x,
                viewport.y,
            );
        });
    }
    // `rt.trace` bound the scene + columns descriptor sets via raw
    // `cmd_bind_descriptor_sets`, bypassing wgpu's tracker — so wgpu would free
    // those descriptor sets as soon as the bind groups are dropped (the binder
    // rebuilds the scene group every frame), even while this trace is still
    // in-flight (the regenerate device-lost). Registering them here ties their
    // lifetime to this submission's completion via wgpu's normal deferred-free.
    trace_encoder.keep_bind_group_alive(scene_bg);
    trace_encoder.keep_bind_group_alive(columns_bg);
    ctx.add_command_buffer(trace_encoder.finish());

    // Blit the per-pixel output buffer into the view's HDR storage texture (a
    // normal wgpu compute pass on the shared ctx encoder → runs after the trace
    // buffer, so the view target stays wgpu-layout-tracked).
    let bind_group = render_device.create_bind_group(
        "rt_blit_bind_group",
        &pipeline_cache.get_bind_group_layout(&rt_blit.layout),
        &BindGroupEntries::sequential((
            output.buffer.as_entire_binding(),
            view_target.get_unsampled_color_attachment().view,
        )),
    );
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("rt_blit"),
        timestamp_writes: None,
    });
    pass.set_pipeline(blit_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);
}
