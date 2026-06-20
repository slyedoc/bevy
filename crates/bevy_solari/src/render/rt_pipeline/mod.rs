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

use ash::vk;
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, texture_storage_2d},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BufferUsages,
        CachedComputePipelineId, CommandEncoderDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineCache,
        ShaderStages, StorageTextureAccess, TextureFormat,
    },
    renderer::{raw_vulkan_init::AdditionalVulkanFeatures, RenderContext, RenderDevice, ViewQuery},
    texture::{FallbackImage, GpuImage},
    view::{ExtractedView, ViewTarget},
};
use wgpu::hal::api::Vulkan as VkApi;

use crate::bindings::RaytracingSceneBindings;
use crate::ecs_gpu::SceneColumns;
use crate::material::MaterialSlots;
use crate::gpu::allocator::{Allocator, MemoryLocation};
use crate::gpu::extension::RayTracingPipelineFeature;
use crate::gpu::rt_pipeline::{RtCamera, RtPipeline};
use crate::render::atmosphere::{AtmosphereSky, SolariAtmosphereView};
use crate::render::view_cull::SolariEnvironmentMap;
use crate::render::SolariCamera;

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
fn raw_image_view(
    view: &bevy_render::render_resource::TextureView,
) -> Option<vk::ImageView> {
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
}

/// `Prepare`: (re)allocate the per-view RT output buffer to fit the viewport.
pub fn prepare_rt_output(
    views: Query<(Entity, &ExtractedCamera, Option<&RtOutputBuffer>), With<SolariCamera>>,
    allocator: Option<Res<Allocator>>,
    render_device: Res<RenderDevice>,
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
        commands.entity(entity).insert(RtOutputBuffer {
            buffer: buffer.into(),
            raw,
            size,
            pixels,
        });
    }
}

/// `RenderGraph`: cast primary rays through the RT pipeline and blit the result
/// into the view. Lazily builds the RT pipeline on the first frame the scene +
/// columns bind groups (and their layouts) are ready — its layout must match
/// wgpu's exact descriptor set layouts, which only exist once those are built.
pub fn rt_pipeline(
    view: ViewQuery<(
        &ExtractedView,
        &ExtractedCamera,
        &ViewTarget,
        &RtOutputBuffer,
        Option<&SolariAtmosphereView>,
        Option<&SolariEnvironmentMap>,
    )>,
    rt: Option<Res<RtPipeline>>,
    rt_blit: Res<RtBlit>,
    allocator: Option<Res<Allocator>>,
    additional: Res<AdditionalVulkanFeatures>,
    scene_bindings: Res<RaytracingSceneBindings>,
    scene_columns: Res<SceneColumns>,
    material_slots: Res<MaterialSlots>,
    atmosphere_sky: Option<Res<AtmosphereSky>>,
    texture_assets: Res<RenderAssets<GpuImage>>,
    fallback_image: Res<FallbackImage>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut frame_counter: Local<u32>,
    mut commands: Commands,
    mut ctx: RenderContext,
) {
    let (view, camera, view_target, output, atmosphere_view, environment_map) =
        view.into_inner();

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
                    .and_then(|env| texture_assets.get(&env.image))
                    .map(|image| &image.texture_view)
                    .unwrap_or(&fallback_image.cube.texture_view);
                (view, None)
            }
        };
    // Match the megakernel (view_cull.rs): the baked atmosphere cube is already
    // physical radiance (brightness 1.0); otherwise the skybox's raw cd/m²; else 0
    // (no sky ⇒ miss stays at the clear color). NB: `Skybox` is stripped from the
    // render world, so brightness comes from `SolariEnvironmentMap`.
    let environment_brightness = if atmosphere_view.is_some() {
        1.0
    } else {
        environment_map.map_or(0.0, |env| env.brightness)
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

    // Lazily build the RT pipeline once the scene + columns layouts exist (its
    // pipeline layout bakes in their raw VkDescriptorSetLayouts). Inserted via
    // commands → live next frame.
    let Some(rt) = rt else {
        // Wait for materials (the SBT sizes one hit record per material slot) AND,
        // if this view uses the atmosphere, its baked cube — the env view is baked
        // into set 1 once at build, so building before the cube exists would freeze
        // the fallback cube in as the sky.
        let env_ready = atmosphere_view.is_none() || atmosphere_sky.is_some();
        if additional.has::<RayTracingPipelineFeature>() && material_slots.len() > 0 && env_ready {
            if let (Some(allocator), Some(scene_layout), Some(columns_layout), Some(env_view)) = (
                allocator.as_deref(),
                raw_bgl(&pipeline_cache, &scene_bindings.bind_group_layout),
                raw_bgl(&pipeline_cache, columns_layout_desc),
                raw_image_view(environment_map_view),
            ) {
                if let Some(built) = RtPipeline::new(
                    allocator,
                    scene_layout,
                    columns_layout,
                    output.raw,
                    output.size,
                    material_slots.len(),
                    env_view,
                    environment_map_image,
                ) {
                    commands.insert_resource(built);
                }
            }
        }
        return;
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
    let world_from_view = view.world_from_view.to_matrix();
    let world_from_clip = world_from_view * view.clip_from_view.inverse();
    let camera_inputs = RtCamera {
        inverse_view_proj: world_from_clip.to_cols_array(),
        // .xyz = ray origin; .w = camera exposure (raygen scales final radiance by
        // it, like the megakernel's `radiance *= view.exposure`).
        camera_position: view.world_from_view.translation().extend(camera.exposure).to_array(),
        frame: [*frame_counter, 0, 0, 0],
        // .x = sky brightness; .yzw = clear color (black for bevy_city).
        sky: [environment_brightness, 0.0, 0.0, 0.0],
    };
    *frame_counter = frame_counter.wrapping_add(1);
    rt.set_camera(&camera_inputs);

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
            rt.trace(command_buffer, scene_set, columns_set, viewport.x, viewport.y);
        });
    }
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
