//! Ray-tracing-pipeline shading path.
//!
//! The raw-VK [`RtPipeline`](crate::gpu::rt_pipeline::RtPipeline) records a
//! `cmd_trace_rays` that writes a per-pixel output **storage buffer** (no image
//! layout to fight wgpu over); a small wgpu compute pass ([`blit.wgsl`]) then
//! copies that buffer into the view's HDR storage texture, so the rest of the
//! frame is unchanged.
//!
//! Runs for every [`SolariCamera`] view; the component's variant selects the
//! integrator (realtime ReSTIR vs reference accumulation).
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
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
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
use crate::gpu::rt_pipeline::{
    RtCamera, RtGeometryAddresses, RtLibraryCache, RtPipeline, RtViewBindings, SolariAnyHitDef,
    SolariChitSource, SolariHitGroupDef, SolariHitGroupRegistry,
};
use crate::gpu::RawTraceBindable;
use crate::material::{material_sbt_class, MaterialSlots, MaterialTraversalFlags};
use crate::pipelines::SolariPipelines;
use crate::render::atmosphere::{
    AtmosphereSky, SolariAtmosphereGpu, SolariAtmosphereView, SolariAtmosphereVolumesGpu,
};
use crate::render::sky::{SolariCustomSky, SolariViewClearColor, SolariViewSkyShader};
use crate::render::view_cull::SolariEnvironmentMap;
use crate::render::{
    CameraReframe, DiEstimator, GiEstimator, SolariCamera, SolariReference,
};
use crate::resource_manager::SolariResourceManager;
use crate::transform::{TransformGraph, TransformPropagate};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};

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

/// The camera's active debug view (the [`SolariCamera::debug`] field, default
/// [`None`](Self::None) = normal rendering), at most one view at a time. Each
/// variant replaces the shaded image with an instrument paint; set it from the
/// main world (e.g. on a keypress, or via the debug UI's view dropdown).
/// Switching variants resets the new variant's knobs to its defaults — the
/// knobs live in the variant payload.
#[cfg_attr(feature = "bevy_solari_debug", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Clone, Copy, PartialEq, Default, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub enum SolariDebugView {
    /// Normal rendering.
    #[default]
    None,
    /// Per-pixel cost heatmap (requires `VK_KHR_shader_clock`): the raygen reads
    /// the shader clock around the trace and replaces the shaded color with a
    /// colormap of the per-pixel cost. A no-op when the device lacks
    /// shader_clock (the raygen's clock reads compile out).
    CostHeatmap {
        /// `log2(cycles)` that maps to the colormap midpoint (green) — slide it
        /// to the scene's midrange cost.
        center: f32,
        /// Contrast: colormap change per `log2(cycles)` stop around `center`.
        /// Crank it up to push the slowest toward red and the fastest toward blue.
        contrast: f32,
    },
    /// Colormap of the per-pixel count of alpha any-hit shader invocations —
    /// the OMM-effectiveness view. Cold (blue) = the hit resolved in hardware
    /// with no any-hit (opacity-micromap opaque/transparent micro-regions, or
    /// plain opaque geometry); hot (red) = many any-hit invocations (unknown
    /// micro-regions, or alpha cutouts with no baked OMM). With OMM working,
    /// foliage interiors go cold and only the silhouette edges stay warm.
    /// Unlike the cost heatmap this needs no `shader_clock`.
    AnyHitCount {
        /// Count → colormap scale: `color = cost_heatmap(count * scale)`.
        scale: f32,
    },
    /// Show each surface's displacement (height) map as grayscale — a validation
    /// for the displacement-map wiring BEFORE tessellation actually displaces
    /// geometry: confirms which map lands on which surface, the UV mapping, and
    /// the height sign. The opaque closest-hit replaces shading with the sampled
    /// height on surfaces that have a `displacement_texture`, and a dim grey
    /// elsewhere.
    Displacement,
    /// Flat per-CLUSTER color (a hash of the global cluster id). Shows the
    /// cluster decomposition directly — each cluster a distinct hue — so
    /// tessellation density (CLAS count) is visible at a glance.
    Clusters,
    /// Flat per-TRIANGLE color (a hash of cluster id + primitive index). Each
    /// (micro-)triangle gets a distinct hue, so view-dependent tessellation
    /// LEVEL reads directly as triangle density.
    Triangles,
    /// Color each primary hit by whether its SHADING normal faces the camera —
    /// blue toward the viewer, red away (inverted / back-wound), brightness =
    /// facing magnitude so grazing reads dark. On watertight, correctly-wound
    /// geometry everything is blue; red patches are the bug.
    NormalFacing,
    /// Paint the NRC cache prediction at the primary hit (inline coopvec MLP
    /// inference per pixel). Shows only while [`SolariNrc::enabled`]
    /// (`crate::nrc::SolariNrc`) — the paint queries the live cache weights.
    NrcCache,
    /// Lighting only: every opaque surface shades with a WHITE base color, so
    /// the image is pure light transport — the albedo-demodulated presentation
    /// GI papers use for estimator comparison. Not a paint: it renders and
    /// accumulates like a normal image (estimator flag bit 28, which also
    /// restarts any running mean when toggled).
    WhiteWorld,
}

impl SolariDebugView {
    /// [`CostHeatmap`](Self::CostHeatmap) with the default colormap mapping:
    /// center 16 (≈ 65k cycles), contrast 0.15/stop (spreads a ±3-stop range
    /// across the colormap).
    pub fn cost_heatmap() -> Self {
        Self::CostHeatmap { center: 16.0, contrast: 0.15 }
    }

    /// [`AnyHitCount`](Self::AnyHitCount) at the default scale 0.1 (≈10
    /// any-hits saturates to red). Tune per scene.
    pub fn any_hit_count() -> Self {
        Self::AnyHitCount { scale: 0.1 }
    }
}

/// Debug inputs bundled into one [`SystemParam`] to keep the dispatch under
/// bevy's 16-system-param limit.
#[derive(bevy_ecs::system::SystemParam)]
pub(crate) struct RtDebug<'w> {
    freeze_diff: Option<Res<'w, SolariFreezeDiff>>,
    nrc: Option<Res<'w, crate::nrc::SolariNrc>>,
    nrc_buffers: Option<ResMut<'w, crate::nrc::NrcBuffers>>,
    nrc_pipelines: Option<Res<'w, crate::nrc::NrcPipelines>>,
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
    /// 16-byte uniform for the diff view: `[mode, scale, 0, 0]` (see `blit.wgsl`).
    pub params: bevy_render::render_resource::Buffer,
}

/// Freeze/diff harness controls (main-world, extracted). Bump `freeze_epoch` to
/// snapshot the current accumulated image; `diff` displays `|current − frozen|`
/// as a heatmap through the blit; bump `dump_epoch` to write the accumulation
/// buffer as an EXR into `target/tmp/` for offline RMSE/FLIP.
#[derive(Resource, Clone, bevy_render::extract_resource::ExtractResource)]
pub struct SolariFreezeDiff {
    pub freeze_epoch: u32,
    pub dump_epoch: u32,
    pub diff: bool,
    pub diff_scale: f32,
    /// Auto-dump once when the accumulation crosses this spp (0 = off) — the
    /// fps-independent way to capture EQUAL-SAMPLE images for RMSE comparisons.
    pub dump_at_spp: u32,
}

impl Default for SolariFreezeDiff {
    fn default() -> Self {
        Self { freeze_epoch: 0, dump_epoch: 0, diff: false, diff_scale: 4.0, dump_at_spp: 0 }
    }
}

/// Per-view frozen snapshot of the accumulated output (the diff reference).
#[derive(Component)]
pub struct RtFrozen {
    pub buffer: bevy_render::render_resource::Buffer,
    pub pixels: u32,
    pub spp: u32,
}

/// `Render` (`Cleanup`): execute [`SolariFreezeDiff`] epoch bumps — snapshot the
/// accumulated output into [`RtFrozen`] (freeze) and/or write it as an EXR into
/// `target/tmp/` (dump; blocking readback — a manual harness op, hitch accepted).
pub fn rt_freeze_ops(
    views: Query<(
        Entity,
        &RtOutputBuffer,
        Option<&RtAccumulation>,
        Option<&RtFrozen>,
        &ExtractedCamera,
    )>,
    freeze_diff: Option<Res<SolariFreezeDiff>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut seen: Local<(u32, u32, u32)>,
    mut commands: Commands,
) {
    let Some(fd) = freeze_diff else { return };
    let do_freeze = fd.freeze_epoch != seen.0;
    let mut do_dump = fd.dump_epoch != seen.1;
    seen.0 = fd.freeze_epoch;
    seen.1 = fd.dump_epoch;
    // Re-arm the spp latch when the configured threshold changes (0 = none fired).
    if seen.2 != 0 && seen.2 != fd.dump_at_spp {
        seen.2 = 0;
    }
    for (entity, output, accumulation, frozen, camera) in &views {
        let spp = accumulation.map_or(0, |a| a.n);
        // Equal-sample capture: fire once when crossing the spp threshold.
        if fd.dump_at_spp > 0 && spp >= fd.dump_at_spp && seen.2 != fd.dump_at_spp {
            do_dump = true;
            seen.2 = fd.dump_at_spp;
        }
        if !do_freeze && !do_dump {
            continue;
        }
        if do_freeze {
            let buffer = match frozen {
                Some(f) if f.buffer.size() == output.size => f.buffer.clone(),
                _ => render_device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rt_frozen"),
                    size: output.size,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            };
            let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("rt_freeze_copy"),
            });
            encoder.copy_buffer_to_buffer(&output.buffer, 0, &buffer, 0, output.size);
            render_queue.submit([encoder.finish()]);
            commands.entity(entity).insert(RtFrozen { buffer, pixels: output.pixels, spp });
            bevy_log::info!("solari freeze: reference snapshot at {spp} spp");
        }
        if do_dump {
            let Some(viewport) = camera.physical_viewport_size else { continue };
            let staging = render_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rt_dump_staging"),
                size: output.size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("rt_dump_copy"),
            });
            encoder.copy_buffer_to_buffer(&output.buffer, 0, &staging, 0, output.size);
            render_queue.submit([encoder.finish()]);
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = render_device.poll(wgpu::PollType::wait_indefinitely());
            if rx.recv().map(|r| r.is_err()).unwrap_or(true) {
                bevy_log::warn!("solari dump: readback map failed");
                continue;
            }
            let (w, h) = (viewport.x as usize, viewport.y as usize);
            let _ = std::fs::create_dir_all("target/tmp");
            let path = format!("target/tmp/solari-{w}x{h}-{spp}spp-{}.exr", fd.dump_epoch);
            let result = {
                let data = slice.get_mapped_range();
                write_dump_exr(&path, w, h, &data)
            };
            staging.unmap();
            match result {
                Ok(()) => bevy_log::info!("solari dump: wrote {path} ({spp} spp)"),
                Err(e) => bevy_log::warn!("solari dump: {e}"),
            }
        }
    }
}

/// Write the mapped RGBA32F output rows (top-down, 16 B/pixel, `.w` = packed
/// depth-alpha) as an RGB OpenEXR. Uncompressed scanlines: downstream tools
/// (FLIP's tinyexr reader) crash on compressed encodings.
fn write_dump_exr(path: &str, w: usize, h: usize, data: &[u8]) -> Result<(), exr::error::Error> {
    use exr::prelude::*;
    let texel = |x: usize, y: usize, c: usize| -> f32 {
        let i = (y * w + x) * 16 + c * 4;
        f32::from_le_bytes(data[i..i + 4].try_into().unwrap())
    };
    let mut image = Image::from_channels(
        (w, h),
        SpecificChannels::rgb(|pos: Vec2<usize>| {
            (
                texel(pos.x(), pos.y(), 0),
                texel(pos.x(), pos.y(), 1),
                texel(pos.x(), pos.y(), 2),
            )
        }),
    );
    image.layer_data.encoding = Encoding::UNCOMPRESSED;
    image.write().to_file(path)
}

/// `RenderStartup`: build the blit pipeline (independent of `SolariPipelines`).
pub fn init_rt_blit(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
    render_device: Res<RenderDevice>,
    mut registry: ResMut<crate::ecs_gpu::SolariPipelineRegistry>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "rt_blit_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0: rt_output (array<vec4<f32>>)
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly), // 1: view_output
                storage_buffer_read_only_sized(false, None), // 2: frozen snapshot (diff view)
                bevy_render::render_resource::binding_types::uniform_buffer_sized(false, None), // 3: diff params
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
    registry.register("rt_blit", pipeline);
    let params = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rt_blit_params"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    commands.insert_resource(RtBlit { layout, pipeline, params });
}

/// The ReSTIR spatial merge+shade pass: pipeline + its group-1 layout
/// (group 0 is the shared scene bind group — TLAS/lights/materials/DFG LUT).
#[derive(Resource)]
pub struct RestirSpatial {
    pub layout: BindGroupLayoutDescriptor,
    /// Queued lazily on the first dispatch frame: the pipeline layout needs the
    /// scene-columns bind-group layout (the resolve reads `transforms` from it),
    /// which doesn't exist yet at `RenderStartup` — same reason the RT pipeline
    /// itself builds lazily.
    pub pipeline: Option<CachedComputePipelineId>,
    pub shader: bevy_asset::Handle<bevy_shader::Shader>,
    /// `SpatialParams` uniform (see `restir_spatial.wgsl`).
    pub params: bevy_render::render_resource::Buffer,
    /// The GI-finalize dispatch's own `SpatialParams` (phase = 1).
    pub params_finalize: bevy_render::render_resource::Buffer,
    /// group(1) binding(4): `scene_bindings` hard-codes `geometry_addresses` here and
    /// it rides in transitively via brdf. The spatial pass never dereferences it (no
    /// `physical_load`), but the binding must exist — a zeroed uniform satisfies it.
    pub geo_addr: bevy_render::render_resource::Buffer,
}

/// CPU mirror of `restir_spatial.wgsl::SpatialParams`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RestirSpatialParams {
    pub width: u32,
    pub height: u32,
    pub parity: u32,
    pub frame: u32,
    pub taps: u32,
    pub radius: f32,
    pub blend_w: f32,
    pub firefly_clamp: f32,
    pub unbiased: u32,
    pub pad_a: u32,
    pub di_on: u32,
    pub gi_on: u32,
    /// 1 = the GI-finalize dispatch (temporal merge + reservoir shade — the
    /// work moved out of raygen's register-cliffed sample loop), 0 = spatial.
    pub phase: u32,
    pub pad_b: u32,
    pub pad_c: u32,
    pub pad_d: u32,
}

/// `RenderStartup`: build the spatial pass pipeline (scene group 0 + own group 1).
pub fn init_restir_spatial(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    render_device: Res<RenderDevice>,
) {
    use bevy_render::render_resource::binding_types::{storage_buffer_sized, uniform_buffer_sized};
    let layout = BindGroupLayoutDescriptor::new(
        "restir_spatial_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_sized(false, None),           // 0: reservoirs (rw)
                storage_buffer_read_only_sized(false, None), // 1: surfaces
                storage_buffer_sized(false, None),           // 2: rt_output (rw, += DI)
                uniform_buffer_sized(false, None),           // 3: params
                uniform_buffer_sized(false, None),           // 4: geometry_addresses (unused, transitive)
                storage_buffer_read_only_sized(false, None), // 5: light_samples (chit-written)
                storage_buffer_sized(false, None),           // 6: gi_samples (rw: finalize merges + writes back)
                uniform_buffer_sized(false, None),           // 7: RtCamera (reprojection + estimator flags)
            ),
        ),
    );
    let shader = load_embedded_asset!(asset_server.as_ref(), "restir_spatial.wgsl");
    let params = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("restir_spatial_params"),
        size: size_of::<RestirSpatialParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // The finalize dispatch runs the same pipeline in the same frame with its
    // own parameter values — it needs its own uniform buffer.
    let params_finalize = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("restir_gi_finalize_params"),
        size: size_of::<RestirSpatialParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let geo_addr = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("restir_spatial_geo_addr"),
        size: size_of::<RtGeometryAddresses>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    commands.insert_resource(RestirSpatial {
        layout,
        pipeline: None,
        shader,
        params,
        params_finalize,
        geo_addr,
    });
}

/// `Render::Prepare`: queue the spatial pipeline the moment the scene-columns
/// layout exists (frame ~2 — it can't be built at `RenderStartup`), so it
/// compiles alongside the cold-start batch instead of lazily on first use.
/// Registered into the one readiness gate like every other pipeline.
pub fn queue_restir_spatial_pipeline(
    restir_spatial: Option<ResMut<RestirSpatial>>,
    scene_bindings: Res<RaytracingSceneBindings>,
    scene_columns: Res<SceneColumns>,
    pipeline_cache: Res<PipelineCache>,
    mut registry: ResMut<crate::ecs_gpu::SolariPipelineRegistry>,
) {
    let Some(mut rs) = restir_spatial else { return };
    if rs.pipeline.is_some() {
        return;
    }
    let Some(columns_layout) = scene_columns.layout() else {
        return;
    };
    let id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("restir_spatial".into()),
        layout: vec![
            scene_bindings.bind_group_layout.clone(),
            rs.layout.clone(),
            columns_layout.clone(),
        ],
        shader: rs.shader.clone(),
        shader_defs: vec![bevy_shader::ShaderDefVal::UInt(
            "SOLARI_SCENE_COLUMNS_GROUP".into(),
            2,
        )],
        entry_point: Some("spatial".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    rs.pipeline = Some(id);
    registry.register("restir_spatial", id);
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
    /// ReSTIR DI reservoirs: 2 interleaved 32-B slots per pixel (current/previous by
    /// frame parity). Zero-cleared on creation (M=0 = dead history); persists across
    /// frames — the trace both reads last frame's half and writes this frame's.
    pub reservoirs: bevy_render::render_resource::Buffer,
    pub reservoirs_raw: vk::Buffer,
    pub reservoirs_size: u64,
    /// NRC termination-query ring (16-byte count header + one 48-byte slot
    /// per pixel), appended by raygen, consumed by nrc_query_infer.
    pub nrc_queries: bevy_render::render_resource::Buffer,
    pub nrc_queries_raw: vk::Buffer,
    pub nrc_queries_size: u64,
    /// ReSTIR primary-hit surface G-buffer (48 B/pixel) — chit-written when the
    /// spatial pass is on; read by `restir_spatial.wgsl` for p̂ re-target + shade.
    pub surface: bevy_render::render_resource::Buffer,
    pub surface_raw: vk::Buffer,
    pub surface_size: u64,
    /// ReSTIR winner resolved-light samples: 2 slots/pixel × 48 B, slot-indexed like
    /// `reservoirs`. Chit-written (it has `physical_load`); the wgpu spatial pass
    /// reads it to reshade neighbors without bindless loads.
    pub light_samples: bevy_render::render_resource::Buffer,
    pub light_samples_raw: vk::Buffer,
    pub light_samples_size: u64,
    /// ReSTIR GI canonical samples: 2 slots/pixel × 48 B, slot-indexed
    /// like `reservoirs`. Raygen-written at path end (the suffix radiance is only
    /// known there), raygen-read for the store/recon shade gates.
    pub gi_samples: bevy_render::render_resource::Buffer,
    pub gi_samples_raw: vk::Buffer,
    pub gi_samples_size: u64,
    /// DLSS ray-reconstruction guide G-buffers — normal+roughness, diffuse+depth,
    /// specular+hit-distance, and motion vectors — each `pixels` × `vec4<f32>`,
    /// allocated and reallocated alongside the color output. Written by the
    /// closest-hit (chit-direct), read by the resolve.
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

/// Per-view reference-accumulation progress ([`SolariReference`]): samples averaged
/// so far and the camera/viewport state they were taken under — any change resets
/// `n` to 0 (the mean restarts). Exposure is NOT part of the identity: the buffer
/// holds physical radiance, so exposure changes recompose at the blit for free.
#[derive(Component, Clone)]
pub struct RtAccumulation {
    pub n: u32,
    camera: bevy_transform::components::GlobalTransform,
    clip_from_view: Mat4,
    pixels: u32,
    /// Estimator flag bits (nee_off/restir/RIS-M) — an estimator switch restarts
    /// the mean, otherwise a live A/B toggle averages two different estimators.
    flags: u32,
}

/// One DLSS ray-reconstruction guide buffer: a `pixels` × `vec4<f32>` GPU storage
/// buffer the closest-hit writes. Held with its raw `VkBuffer` (for the RT set-1
/// descriptor) and the wgpu handle (for the resolve pass).
pub struct RtGbuffer {
    pub buffer: bevy_render::render_resource::Buffer,
    pub raw: vk::Buffer,
}

/// Byte size of `rt_camera.wgsl`'s `PrevCamera` (`mat4x4` + 3×`f64` previous
/// origin + 2×`u32` + 3×`f64` held NRC anchor, std430-padded).
const RT_PREV_CAMERA_SIZE: u64 = 128;

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
        // COPY_SRC: validation harnesses (furnace tests) read pixels back.
        let buffer = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
        // `PrevCamera` ([`RT_PREV_CAMERA_SIZE`]). Zero-cleared so `valid` starts 0
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
        // ReSTIR reservoirs: 2 slots × 32 B per pixel, zero-cleared (M=0 = no history).
        let reservoirs_size = pixels as u64 * 64;
        let reservoirs = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            reservoirs_size,
            MemoryLocation::GpuOnly,
            "rt_reservoirs",
        );
        // SAFETY: Vulkan-backed (Allocator only builds VkBuffers).
        let reservoirs_raw = unsafe { reservoirs.as_hal::<VkApi>() }
            .map(|b| b.raw_handle())
            .expect("rt_reservoirs buffer must be Vulkan-backed");
        let reservoirs: bevy_render::render_resource::Buffer = reservoirs.into();
        // Spatial-pass surface G-buffer (48 B/pixel); stale data is gated by the
        // reservoir's M anyway, zero-cleared once for hygiene.
        let surface_size = pixels as u64 * 48;
        let surface = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            surface_size,
            MemoryLocation::GpuOnly,
            "rt_restir_surface",
        );
        // SAFETY: Vulkan-backed (Allocator only builds VkBuffers).
        let surface_raw = unsafe { surface.as_hal::<VkApi>() }
            .map(|b| b.raw_handle())
            .expect("rt_restir_surface buffer must be Vulkan-backed");
        let surface: bevy_render::render_resource::Buffer = surface.into();
        // Winner samples: 2 slots × 64 B per pixel (resolved light + chit's exact
        // f/p̂), slot-indexed like the reservoirs; zero-cleared (gated by M anyway).
        let light_samples_size = pixels as u64 * 128;
        let light_samples = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            light_samples_size,
            MemoryLocation::GpuOnly,
            "rt_restir_light_samples",
        );
        // SAFETY: Vulkan-backed (Allocator only builds VkBuffers).
        let light_samples_raw = unsafe { light_samples.as_hal::<VkApi>() }
            .map(|b| b.raw_handle())
            .expect("rt_restir_light_samples buffer must be Vulkan-backed");
        let light_samples: bevy_render::render_resource::Buffer = light_samples.into();
        // GI reservoirs: 2 slots × 64 B per pixel, zero-cleared (m=0 = dead).
        let gi_samples_size = pixels as u64 * 128;
        let gi_samples = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            gi_samples_size,
            MemoryLocation::GpuOnly,
            "rt_restir_gi_samples",
        );
        // SAFETY: Vulkan-backed (Allocator only builds VkBuffers).
        let gi_samples_raw = unsafe { gi_samples.as_hal::<VkApi>() }
            .map(|b| b.raw_handle())
            .expect("rt_restir_gi_samples buffer must be Vulkan-backed");
        let gi_samples: bevy_render::render_resource::Buffer = gi_samples.into();
        // NRC termination-query ring: count header + one slot per pixel.
        let nrc_queries_size = 16 + pixels as u64 * crate::nrc::NRC_QUERY_SIZE as u64;
        let nrc_queries = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            nrc_queries_size,
            MemoryLocation::GpuOnly,
            "rt_nrc_queries",
        );
        // SAFETY: Vulkan-backed (Allocator only builds VkBuffers).
        let nrc_queries_raw = unsafe { nrc_queries.as_hal::<VkApi>() }
            .map(|b| b.raw_handle())
            .expect("rt_nrc_queries buffer must be Vulkan-backed");
        let nrc_queries: bevy_render::render_resource::Buffer = nrc_queries.into();
        let mut clear_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("rt_reservoirs_clear"),
        });
        clear_encoder.clear_buffer(&reservoirs, 0, None);
        clear_encoder.clear_buffer(&surface, 0, None);
        clear_encoder.clear_buffer(&light_samples, 0, None);
        clear_encoder.clear_buffer(&gi_samples, 0, None);
        clear_encoder.clear_buffer(&nrc_queries, 0, None);
        render_queue.submit([clear_encoder.finish()]);
        commands.entity(entity).insert(RtOutputBuffer {
            buffer: buffer.into(),
            raw,
            size,
            pixels,
            camera_buffer: camera_buffer.into(),
            camera_raw,
            camera_prev_buffer,
            reservoirs,
            reservoirs_raw,
            reservoirs_size,
            nrc_queries,
            nrc_queries_raw,
            nrc_queries_size,
            surface,
            surface_raw,
            surface_size,
            light_samples,
            light_samples_raw,
            light_samples_size,
            gi_samples,
            gi_samples_raw,
            gi_samples_size,
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

/// Cylindrical-window raygen (head-coupled perspective on a curved monitor): primary
/// rays go from the tracked eye through each pixel's physical point on the screen
/// cylinder instead of a planar unproject — no 4×4 can map to a curved screen. The
/// camera entity must sit at `eye` relative to the screen-center anchor with axes
/// aligned to it (the same rig `OffAxisProjection` uses for flat screens); the
/// CPU-authored projection is still used for culling/LOD, so supply a conservative
/// (slightly inflated) off-axis frustum.
#[derive(Component, Clone, Copy)]
pub struct SolariCylindricalWindow {
    /// Horizontal arc angle subtended by the view rect, radians (arc_length / radius)
    pub arc_angle: f32,
    /// Curvature radius, meters (1000R = 1.0)
    pub radius: f32,
    /// View-rect height, meters
    pub height: f32,
    /// View-rect center offset from the monitor center (an OS window's offset):
    /// `.x` = arc-length meters along the cylinder, `.y` = vertical meters
    pub center: Vec2,
    /// Viewer eye in screen space (monitor-center origin, +X right, +Y up, +Z toward viewer)
    pub eye: Vec3,
}

/// `ExtractSchedule`: mirror each camera's [`SolariCylindricalWindow`] (or its absence)
/// onto the render-world view entity.
pub fn extract_cylindrical_window(
    mut commands: Commands,
    cameras: Extract<Query<(&RenderEntity, Option<&SolariCylindricalWindow>), With<SolariCamera>>>,
) {
    for (render_entity, window) in &cameras {
        match window {
            Some(window) => {
                commands.entity(render_entity.id()).insert(*window);
            }
            None => {
                commands
                    .entity(render_entity.id())
                    .remove::<SolariCylindricalWindow>();
            }
        }
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
    /// Viewport pixels + restir M-cap; see `RtCamera::dims`.
    dims: Vec4,
    /// Cylindrical window params; see `RtCamera::window_arc`/`window_eye`.
    window_arc: Vec4,
    window_eye: Vec4,
    /// NRC scene scale (`.x`, meters; 0 = off); see `RtCamera::nrc`.
    nrc: Vec4,
    exposure: f32,
}

static CAMERA_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Latch [`SolariSettings::camera_debug`](crate::SolariSettings) at plugin `finish`.
pub(crate) fn latch_camera_debug(on: bool) {
    let _ = CAMERA_DEBUG.set(on);
}

/// [`SolariSettings::camera_debug`](crate::SolariSettings): log which camera path
/// fills the buffer (GPU pass vs CPU fallback) and why — the fallback is silent
/// by design, which makes a wrong-basis frame indistinguishable from a right one
/// in logs.
fn camera_debug() -> bool {
    CAMERA_DEBUG.get().copied().unwrap_or(false)
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
        if camera_debug() {
            bevy_log::info!(
                "rt_camera: CPU fallback (pipelines={} resources={} propagate={} slot={:?})",
                pipelines.is_some(),
                resources.is_some(),
                propagate.is_some(),
                camera_slot.map(|s| s.0),
            );
        }
        return false; // cold start (slot extracted a frame after the camera spawns).
    };
    let node_count = propagate.node_count();
    // Slot not yet propagated → CPU fallback (correct for a root camera; a transient
    // first-frame-or-two for a childed one, before its slot lands).
    if slot.0 >= node_count {
        if camera_debug() {
            bevy_log::info!("rt_camera: CPU fallback (slot {} >= node_count {node_count})", slot.0);
        }
        return false;
    }
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.rt_camera) else {
        if camera_debug() {
            bevy_log::info!("rt_camera: CPU fallback (pipeline compiling)");
        }
        return false; // still compiling (or failed — the cache logs a compile error).
    };
    if camera_debug() {
        static COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 5 || n % 300 == 0 {
            bevy_log::info!("rt_camera: GPU pass (slot {} node_count {node_count})", slot.0);
        }
    }

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
        dims: inputs.dims,
        window_arc: inputs.window_arc,
        window_eye: inputs.window_eye,
        nrc: inputs.nrc,
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

impl crate::SolariHitGroup for OpaqueSurface {
    fn hit_group() -> SolariHitGroupDef {
        SolariHitGroupDef {
            label: "opaque",
            closest_hit: SolariChitSource::SpirV(include_bytes!("chit_opaque.spv")),
            any_hit: Some(SolariAnyHitDef {
                spirv: include_bytes!("ahit_alpha.spv"),
                entry: "main",
            }),
            composable_modules: &[],
        }
    }
}

/// Glass surface (class 1) — Fresnel reflect/refract; routed by `specular_transmission > 0`.
pub struct GlassSurface;

impl crate::SolariHitGroup for GlassSurface {
    fn hit_group() -> SolariHitGroupDef {
        SolariHitGroupDef {
            label: "glass",
            closest_hit: SolariChitSource::SpirV(include_bytes!("chit_glass.spv")),
            any_hit: None,
            composable_modules: &[],
        }
    }
}

/// Hair surface (class 2) — Chiang fiber BSDF / LSS bark; reached via the reserved hair record.
pub struct HairSurface;

impl crate::SolariHitGroup for HairSurface {
    fn hit_group() -> SolariHitGroupDef {
        SolariHitGroupDef {
            label: "hair",
            closest_hit: SolariChitSource::SpirV(include_bytes!("chit_hair.spv")),
            any_hit: None,
            composable_modules: &[],
        }
    }
}

/// Portal surface (class 3) — teleports rays; pair the instance with a [`SolariPortal`](crate::bindings::SolariPortal).
pub struct PortalSurface;

impl crate::SolariHitGroup for PortalSurface {
    fn hit_group() -> SolariHitGroupDef {
        SolariHitGroupDef {
            label: "portal",
            closest_hit: SolariChitSource::SpirV(include_bytes!("chit_portal.spv")),
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
        (
            Option<&SolariEnvironmentMap>,
            Option<&SolariViewClearColor>,
            Has<SolariViewSkyShader>,
        ),
        Option<&RtPrevViewProj>,
        Option<&SolariDlssJitter>,
        Option<&CameraReframe>,
        Option<&RtCameraSlot>,
        &SolariCamera,
        Option<&RtAccumulation>,
        Option<&RtFrozen>,
        Option<&SolariCylindricalWindow>,
    )>,
    rt: Option<Res<RtPipeline>>,
    rt_blit: Res<RtBlit>,
    allocator: Option<Res<Allocator>>,
    mut debug: RtDebug,
    scene_bindings: Res<RaytracingSceneBindings>,
    scene_columns: Res<SceneColumns>,
    // Tupled into one system param (the system is at bevy's 16-param ceiling).
    geometry_res: (
        Option<Res<ClusterMeshManager>>,
        Option<Res<crate::geometry::tess_classify::TessClassify>>,
        Option<Res<SolariHitGroupRegistry>>,
        Option<Res<crate::accel::deform::Deform>>,
        Option<ResMut<RtLibraryCache>>,
        Option<Res<crate::gpu::binding_seam::BindingSeam>>,
    ),
    materials: RtMaterials,
    // Tupled: baked sky cube + atmosphere GPU state (sky_frame quat) +
    // world-space atmosphere volumes (address for raygen's march) + the live
    // custom-sky module the miss shader composes.
    atmosphere_res: (
        Option<Res<AtmosphereSky>>,
        Option<Res<SolariAtmosphereGpu>>,
        Option<Res<SolariAtmosphereVolumesGpu>>,
        Res<SolariCustomSky>,
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
        Option<ResMut<RestirSpatial>>,
        Option<Res<crate::ecs_gpu::SolariPipelineRegistry>>,
        Option<Res<crate::instance::RtJournal>>,
        Option<Res<crate::accel::ptlas::Ptlas>>,
    ),
    counters: (Local<u32>, Local<u32>),
    mut commands: Commands,
    mut ctx: RenderContext,
) {
    let (
        render_device,
        render_queue,
        solari_pipelines,
        solari_resources,
        transform_propagate,
        time,
        mut restir_spatial,
        pipeline_registry,
        journal,
        ptlas,
    ) = render_res;
    let (mut frame_counter, mut settle_frames) = counters;
    let (cluster_mesh_manager, tess_classify, hit_group_registry, deform, library_cache, seam) =
        geometry_res;
    let (atmosphere_sky, atmosphere_gpu, atmosphere_volumes, custom_sky) = atmosphere_res;
    let view_entity = view.entity();
    let (
        view,
        camera,
        view_target,
        output,
        view_bindings,
        atmosphere_view,
        (environment_map, view_clear_color, sky_shader),
        prev_view_proj,
        dlss_jitter,
        reframe,
        camera_slot,
        solari_camera,
        accumulation,
        frozen,
        cyl_window,
    ) = view.into_inner();
    // The camera's [`SolariCamera`] variant selects the integrator: production
    // ReSTIR (realtime) or reference accumulation.
    let (reference, restir_rt) = (solari_camera.reference(), solari_camera.restir());

    // Environment cube the miss shader samples:
    // the baked atmosphere cube if this view has one, else the view's skybox image,
    // else the fallback cube. The atmosphere cube is a STORAGE image (GENERAL) the
    // RT path never samples via wgpu, so we transition it ourselves. The skybox /
    // fallback are wgpu-tracked — but the trace's sampled read is invisible to the
    // tracker, so declare it below via `transition_resources`.
    let (environment_map_view, environment_map_image, environment_map_texture) =
        match atmosphere_view.and(atmosphere_sky.as_ref()) {
            Some(sky) => (&sky.cube_view, raw_image(&sky.texture), None),
            None => {
                let image = environment_map
                    .and_then(|env| env_images.texture_assets.get(&env.image))
                    .unwrap_or(&env_images.fallback_image.cube);
                (&image.texture_view, None, Some(&image.texture))
            }
        };
    // Make the untracked sampled read visible to wgpu: on the frames the skybox
    // asset uploads, its tracked layout is TRANSFER_DST and no pass would
    // otherwise transition it to read-only before the trace samples it
    // (VUID-vkCmdDraw-None-09600). A no-op once the state already matches.
    if let Some(texture) = environment_map_texture {
        ctx.command_encoder().transition_resources(
            core::iter::empty(),
            core::iter::once(wgpu::TextureTransition {
                // bevy `Texture` → the wrapped `wgpu::Texture`.
                texture: &**texture,
                selector: None,
                state: wgpu::TextureUses::RESOURCE,
            }),
        );
    }
    // Brightness convention (matches view_cull.rs): the baked atmosphere cube is
    // already physical radiance (brightness 1.0); otherwise the skybox's raw cd/m²; else 0
    // (no sky ⇒ miss stays at the clear color). Brightness stays 0 while the skybox
    // image is still loading — the stand-in is the WHITE fallback cube, and lighting
    // it up would flash the whole sky white until the real cubemap lands.
    // Negative brightness = `SolariSky::Procedural`/`Shader`: the miss evaluates
    // the composed `custom_sky` module instead (atmosphere still wins).
    let environment_brightness = if atmosphere_view.is_some() {
        1.0
    } else if sky_shader {
        -1.0
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
        if materials.len() > 0 {
            if let (Some(allocator), Some(scene_layout), Some(columns_layout), Some(registry)) = (
                allocator.as_deref(),
                raw_bgl(&pipeline_cache, &scene_bindings.bind_group_layout),
                raw_bgl(&pipeline_cache, columns_layout_desc),
                hit_group_registry.as_deref(),
            ) {
                // Get-or-recreate the stage-library cache: it lives across
                // pipeline rebuilds (a sky/material change relinks cached
                // libraries instead of recompiling every stage), but its
                // shared pipeline layout bakes in the wgpu raw layouts, so a
                // layout handle change starts a fresh cache (the stale one is
                // dropped when `insert_resource` overwrites it).
                let mut fresh_cache = None;
                let cache: &mut RtLibraryCache = match library_cache {
                    Some(cache) if cache.layout_key() == (scene_layout, columns_layout) => {
                        cache.into_inner()
                    }
                    _ => {
                        fresh_cache =
                            RtLibraryCache::new(allocator, scene_layout, columns_layout);
                        match fresh_cache.as_mut() {
                            Some(cache) => cache,
                            None => return,
                        }
                    }
                };
                if let Some(built) = RtPipeline::new(
                    allocator,
                    cache,
                    &material_classes,
                    &registry.groups,
                    (&custom_sky.source, custom_sky.generation),
                ) {
                    commands.insert_resource(built);
                }
                if let Some(fresh) = fresh_cache {
                    commands.insert_resource(fresh);
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
    // handle is baked, so the new class only takes effect after a rebuild — or the
    // custom-sky module changed (its composed SPIR-V is baked into the miss). Drain
    // the GPU first so dropping the old pipeline (when this frame's `remove`
    // command applies) can't free a `VkPipeline` a still-executing trace uses.
    if materials.len() > rt.capacity()
        || rt.classes_changed(&material_classes)
        || rt.custom_sky_generation() != custom_sky.generation
    {
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
                    // DLSS guide G-buffers — bound into set 1 alongside the
                    // color output, same size, same lifetime.
                    let gbuffers = [
                        (output.gbuffer[0].raw, output.size),
                        (output.gbuffer[1].raw, output.size),
                        (output.gbuffer[2].raw, output.size),
                        (output.gbuffer[3].raw, output.size),
                    ];
                    // NRC buffers are created in Prepare (needs the Allocator);
                    // the static set-1 bindings bake their raw handles, so wait.
                    let Some(nrc_bufs) = debug.nrc_buffers.as_deref() else {
                        return;
                    };
                    if let Some(built) = rt.create_view_bindings(
                        allocator,
                        output.raw,
                        output.size,
                        output.camera_raw,
                        &gbuffers,
                        (output.reservoirs_raw, output.reservoirs_size),
                        (output.surface_raw, output.surface_size),
                        (output.light_samples_raw, output.light_samples_size),
                        (output.gi_samples_raw, output.gi_samples_size),
                        (nrc_bufs.weights_t_raw, nrc_bufs.weights_t_size),
                        (nrc_bufs.bias16_raw, nrc_bufs.bias16_size),
                        (
                            nrc_bufs.records_raw,
                            (crate::nrc::NRC_RECORD_CAP * crate::nrc::NRC_RECORD_SIZE) as u64,
                        ),
                        (output.nrc_queries_raw, output.nrc_queries_size),
                        env_view,
                        environment_map_image,
                        seam.as_deref(),
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
    // Debug-view selector ([`SolariDebugView`] → the shader ABI: `frame.z` view
    // id + `frame.w` displacement flag), hoisted so the reference accumulator
    // can bypass itself while a debug view owns the pixel.
    let active_view = solari_camera.debug;
    let debug_view = match active_view {
        SolariDebugView::CostHeatmap { .. } => 1u32,
        SolariDebugView::AnyHitCount { .. } => 2u32,
        SolariDebugView::Clusters => 3u32,
        SolariDebugView::Triangles => 4u32,
        SolariDebugView::NormalFacing => 5u32,
        // The cache paint queries the live NRC weights — nothing to show
        // unless the cache is running.
        SolariDebugView::NrcCache if debug.nrc.as_deref().is_some_and(|n| n.enabled) => 6u32,
        _ => 0u32,
    };
    let show_displacement = active_view == SolariDebugView::Displacement;

    // Reference accumulation ([`SolariReference`]): misc.z = samples already in the
    // mean, misc.w = samples this frame (0 ⇒ off). Any camera/exposure/viewport/debug
    // change restarts the mean. `n` advances CPU-side; the raygen blends by
    // spf/(n_prev+spf) in place in the output buffer.
    let mut accum_n = 0u32;
    let mut accum_spf = 0u32;
    // Estimator flags (also packed into `atmo.w` below): bit 0 = NEE off,
    // bit 1 = ReSTIR DI, bit 2 = DI only, bit 3 = spatial pass, bit 4 = GI only,
    // bit 5 = ReSTIR GI, bit 6 = GI shade reconstructed from the surface
    // G-buffer (vs the stored exact a0), bit 7 = GI temporal reuse,
    // bits 8..15 = RIS M, bit 16 = GI spatial (the pass owns the GI shade),
    // bit 17 = dead-canonical rate paint, bit 18 = spatial debug (raygen
    // stashes its would-be GI shade for the pass's ratio paint), bit 20 =
    // NRC GI termination (armed only past cache maturity), bit 21 = AA
    // jitter off (pixel-center sampling — the reservoir-merge
    // target-mismatch isolation lever; also set when DLSS drives, so RR
    // always sees the jitter it suggested), bits 22..27 = max indirect
    // bounces (raygen's path-length cap; roulette terminates sooner at
    // the default 32).
    let estimator_flags = if let Some(rt) = restir_rt {
        // Production per-frame stack: ReSTIR DI + GI reconnection reservoirs,
        // temporal always, spatial only when taps > 0 (at 0 the pass doesn't
        // run and the chit/raygen shade their own reservoirs directly). DLSS
        // RR is the denoiser downstream.
        // Bit 19 = history decorrelation (stochastic-bilinear reprojection
        // fetch + temporal reconnection Jacobian): whitens the moiré grid the
        // quantized fetch shows under motion — for the RR path only, so the
        // certified reference estimator stays untouched.
        let spatial = rt.spatial.is_some();
        // Bit 20 = NRC GI termination, armed only once the cache is mature
        // (a cold cache's extrapolations would composite garbage into every
        // frame; until then paths trace their full suffix).
        let nrc_ready = debug.nrc_buffers.as_deref().is_some_and(|b| b.step > 300);
        1 << 1
            | (spatial as u32) << 3
            | ((rt.gi as u32) * (1 << 5 | 1 << 7))
            | ((rt.gi && spatial) as u32) << 16
            | (rt.ris_candidates.min(255) << 8)
            | 1 << 19
            | ((rt.nrc_gi && nrc_ready) as u32) << 20
            | rt.bounces.clamp(1, 32) << 22
    } else {
        // The reference estimator tree → the flag bits (the tree makes legal
        // combinations structural: spatial only inside a ReSTIR arm, recon
        // only inside the GI ReSTIR arm).
        let r = reference.expect("SolariLighting is Realtime or Reference");
        let (gi_recon, gi_temporal) = match r.gi.as_ref().map(|arm| &arm.estimator) {
            Some(GiEstimator::Restir(gi)) => (gi.recon, gi.temporal),
            _ => (false, false),
        };
        matches!(r.di, DiEstimator::BsdfOnly) as u32
            | (r.di_restir() as u32) << 1
            // No GI arm = terminate at the primary vertex (bit 2).
            | (r.gi.is_none() as u32) << 2
            | (r.di_spatial().is_some() as u32) << 3
            | (r.gi.as_ref().is_some_and(|arm| arm.only) as u32) << 4
            | (r.gi_restir() as u32) << 5
            | (gi_recon as u32) << 6
            | (gi_temporal as u32) << 7
            | (r.ris_candidates().min(255) << 8)
            | (r.gi_spatial().is_some() as u32) << 16
            | (r.gi_dead_view() as u32) << 17
            | (r.spatial_debug_paint() as u32) << 18
            | ((r.nrc_gi()
                && debug.nrc_buffers.as_deref().is_some_and(|b| b.step > 300))
                as u32)
                << 20
            // DLSS-RR must see the jitter it suggested: when RR drives, raygen
            // keeps camera.jitter for every sample (accumulation still gets AA
            // from the Halton sweep) instead of rolling its own.
            | ((!r.jitter || dlss_jitter.is_some()) as u32) << 21
            | r.bounces().clamp(1, 32) << 22
    };
    // The lighting-only view rides the estimator flags (bit 28), not the
    // frame.z paint id: it is a converging image — accumulation stays live,
    // and the flags-keyed restart resets the mean when it toggles.
    let estimator_flags =
        estimator_flags | ((active_view == SolariDebugView::WhiteWorld) as u32) << 28;
    if let Some(reference) = reference {
        // Hold accumulation until EVERY solari pipeline is compiled (the one
        // readiness gate — a warmup frame with any column/pass missing bakes
        // zero/garbage samples into the running mean permanently), plus the
        // lazily-queued spatial pass when its levers are on. Pipelines are not
        // enough: the cluster→BLAS→PTLAS stream lands the scene several frames
        // AFTER the last pipeline compiles, and accumulating those black frames
        // is a permanent ~K/N energy deficit. So also require the scene quiet — no pending journal
        // records or mesh uploads, PTLAS built — for a few consecutive frames
        // (build latency the CPU can't observe directly).
        let spatial_pending = (reference.di_spatial().is_some()
            || reference.gi_spatial().is_some())
            && !restir_spatial
                .as_deref()
                .and_then(|rs| rs.pipeline)
                .is_some_and(|id| pipeline_cache.get_compute_pipeline(id).is_some());
        let pipelines_ready = pipeline_registry.is_some_and(|r| r.ready(&pipeline_cache));
        let scene_quiet = journal.is_none_or(|j| j.count == 0)
            && cluster_mesh_manager.as_ref().is_none_or(|m| {
                m.pending_clas_uploads.is_empty() && m.pending_procedural.is_empty()
            })
            && ptlas.is_some_and(|p| p.has_built);
        *settle_frames = if pipelines_ready && scene_quiet && !spatial_pending {
            settle_frames.saturating_add(1)
        } else {
            0
        };
        // Reservoir history must also match the ESTIMATOR the mean will run:
        // chains only start warming once the scene is resident, and their
        // stationary W depends on the RR mode (realtime RR ⇒ ~half the canonical
        // draws are dead ⇒ legitimately diluted W). So warmup frames run the
        // SAME reference estimator (spf on the GPU ⇒ reference RR), and the
        // history-maturity window lets the chain reach ITS stationary state
        // before n starts advancing — else the first m-cap frames shade ~0.4×
        // and bake a permanent deficit.
        let history_frames = if reference.di_restir() || reference.gi_restir() {
            reference.m_cap().ceil() as u32 + 4
        } else {
            0
        };
        let warmup = *settle_frames < 8 + history_frames;
        // `accumulate: false` = fresh frames (estimator levers stay live) — the
        // per-frame variance instrument; accum_spf 0 disables the raygen blend.
        // Fresh frames still honor samples_per_frame (N samples averaged per
        // frame, accum_n stays 0 so the shader full-replaces) — UNLESS DLSS
        // is driving: spf > 0 switches raygen to its own AA jitter, and RR
        // must see the jitter it was told about.
        if debug_view == 0 && !show_displacement && !reference.accumulate && dlss_jitter.is_none()
        {
            accum_spf = reference.samples_per_frame.max(1);
        }
        if debug_view == 0 && !show_displacement && reference.accumulate {
            accum_spf = reference.samples_per_frame.max(1);
            if warmup {
                // Reference estimator runs, mean doesn't: accum_n stays 0 (no
                // raygen blend) and any stale accumulation state is dropped so
                // the mean starts fresh at gate-open.
                commands.entity(view_entity).remove::<RtAccumulation>();
            } else {
                let same = accumulation.is_some_and(|a| {
                    a.camera == view.world_from_view
                        && a.clip_from_view == view.clip_from_view
                        && a.pixels == output.pixels
                        && a.flags == estimator_flags
                });
                accum_n = if same { accumulation.unwrap().n } else { 0 };
                let n_new = accum_n + accum_spf;
                if accum_n == 0 || n_new.leading_zeros() != accum_n.leading_zeros() {
                    bevy_log::info!("solari reference: {n_new} spp");
                }
                if dlss_jitter.is_some() {
                    bevy_log::warn_once!(
                        "SolariReference with DLSS active: the resolve overwrites the accumulated image — disable DLSS on this camera"
                    );
                }
                commands.entity(view_entity).insert(RtAccumulation {
                    n: n_new,
                    camera: view.world_from_view,
                    clip_from_view: view.clip_from_view,
                    pixels: output.pixels,
                    flags: estimator_flags,
                });
            }
        }
    }

    // Cylindrical-window raygen params (zeros = mode off, planar unproject).
    // `.w` slots carry the view-rect center: arc angle offset + vertical meters.
    let window_arc = cyl_window.map_or(Vec4::ZERO, |w| {
        Vec4::new(w.arc_angle, w.radius, w.height, w.center.x / w.radius.max(1e-3))
    });
    let window_eye = cyl_window.map_or(Vec4::ZERO, |w| w.eye.extend(w.center.y));

    // The camera decides whether the cache runs at all: without an NRC
    // consumer (rt `nrc_gi`, a reference nrc arm, or the cache debug paint),
    // record writes, inference, and training all idle — an rt-nonrc camera
    // pays zero NRC cost.
    let wants_nrc = restir_rt.is_some_and(|rt| rt.nrc_gi)
        || reference.is_some_and(|r| r.nrc_gi())
        || active_view == SolariDebugView::NrcCache;
    // NRC master gate → RtCamera.nrc.x (0 = record writes + debug view off);
    // .z selects inline coopvec inference over the batched query path.
    let nrc_vec = debug
        .nrc
        .as_deref()
        .filter(|n| n.enabled && wants_nrc)
        .map_or(Vec4::ZERO, |n| {
            Vec4::new(
                n.scene_scale,
                n.spread_c,
                if n.inline_coopvec { 1.0 } else { 0.0 },
                0.0,
            )
        });

    let camera_inputs = RtCamera {
        inverse_view_proj: world_from_clip.to_cols_array(),
        view_from_world: view_from_world.to_cols_array(),
        clip_from_world: clip_from_world.to_cols_array(),
        prev_clip_from_world: prev_clip_from_world.to_cols_array(),
        // .xyz = ray origin; .w = camera exposure (carried for shader-side tooling
        // only — radiance stays physical, the blit applies exposure).
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
            // (blue toward camera / red away), 6 = NRC cache paint. One view at
            // a time — `SolariDebugView` is a single enum.
            debug_view,
            // .w = displacement debug view (1 = on); the opaque chit shows each
            // surface's height map (grayscale) to validate the displacement wiring.
            show_displacement as u32,
        ],
        // .x = sky brightness (< 0 ⇒ custom sky shader); .yzw = the camera's clear
        // color — physical radiance like everything traced (the blit exposes it).
        sky: {
            let clear = view_clear_color.map_or(Vec3::ZERO, |c| c.0);
            [environment_brightness, clear.x, clear.y, clear.z]
        },
        // Sub-pixel jitter (pixels) for DLSS temporal accumulation; zero without an
        // active DLSS context (the trace renders a fresh frame with no accumulator,
        // so an unaccumulated jitter would only shimmer).
        jitter: {
            let j = dlss_jitter.map_or(Vec2::ZERO, |j| j.offset);
            // .zw = cost-heatmap log2 center + contrast (read by the raygen colormap).
            // The any-hit heatmap reuses .z as its count→colormap scale (only one
            // debug view is active at a time, so the slot is unambiguous).
            let (center, contrast) = match active_view {
                SolariDebugView::CostHeatmap { center, contrast } => (center, contrast),
                SolariDebugView::AnyHitCount { scale } => (scale, 0.0),
                _ => (0.0, 0.0),
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
            // .z = reference samples already accumulated; .w = samples this frame
            // (0 ⇒ reference mode off — the raygen renders one fresh sample).
            accum_n as f32,
            accum_spf as f32,
        ],
        // World→bake sky rotation (identity unless a spherical-planet
        // atmosphere set one) — the miss shader rotates cube sample dirs.
        sky_frame: atmosphere_gpu
            .as_deref()
            .map_or([0.0, 0.0, 0.0, 1.0], |a| a.sky_frame.to_array()),
        // Atmosphere volumes: device address (bit-preserved through f32) +
        // live count. Zero count ⇒ raygen skips the march entirely.
        // .w = estimator flags (see `estimator_flags` above). SolariReference levers.
        atmo: {
            let flags = f32::from_bits(estimator_flags);
            atmosphere_volumes.as_deref().map_or([0.0, 0.0, 0.0, flags], |v| {
                [
                    f32::from_bits(v.address as u32),
                    f32::from_bits((v.address >> 32) as u32),
                    f32::from_bits(v.count),
                    flags,
                ]
            })
        },
        // Viewport pixels (restir temporal reprojection) + history M-cap +
        // GI firefly clamp (PHYSICAL radiance; 0 = off — the reference never
        // clamps, so it always passes 0).
        dims: [
            viewport.x as f32,
            viewport.y as f32,
            restir_rt.map_or_else(
                || reference.expect("SolariLighting is Realtime or Reference").m_cap(),
                |rt| rt.m_cap,
            ),
            restir_rt.map_or(0.0, |rt| rt.firefly_clamp / camera.exposure.max(1.0e-9)),
        ],
        world_from_view: world_from_view.to_cols_array(),
        window_arc: window_arc.to_array(),
        nrc: nrc_vec.to_array(),
        // GPU camera pass computes the world-snapped anchor from the absolute
        // f64 origin; the CPU fallback stays camera-anchored.
        nrc_anchor: [0.0; 4],
        window_eye: window_eye.to_array(),
        // CPU fallback path: no per-frame origin delta (the GPU camera pass
        // computes it in f64) — cross-frame reservoir reuse is stale-by-one
        // under motion there, matching the fallback's best-effort contract.
        origin_delta: [0.0; 4],
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
                dims: Vec4::from_array(camera_inputs.dims),
                window_arc,
                window_eye,
                nrc: nrc_vec,
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
    let geo_addrs = cluster_mesh_manager.as_deref().map(|cluster_mesh_manager| {
        // Smooth-tess metadata table address (0 when the smooth path is off → the
        // closest-hit falls back to the facet normal). The GPU-classify path's per-part
        // metadata (real UVs + smooth normals) reached via `geometry_addresses.tess_clusters`.
        let tess_clusters = tess_classify.as_ref().map_or(0, |c| c.gen_attrs_meta_addr);
        RtGeometryAddresses {
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
        }
    });
    if let Some(ga) = geo_addrs.as_ref() {
        view_bindings.set_geometry_addresses(ga);
    }

    // Reset the termination-query ring's counter on the ctx encoder BEFORE the
    // trace (the pre-trace flush orders this write ahead of raygen's appends).
    if debug.nrc.as_deref().is_some_and(|n| n.enabled) {
        ctx.command_encoder()
            .clear_buffer(&output.nrc_queries, 0, Some(16));
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

    // ReSTIR spatial merge+shade: after the trace (all reservoirs +
    // surfaces exist), before the blit. Adds `blend_w · DI` (physical radiance)
    // into the accumulated output — the same blend weight the raygen used this
    // frame, so accumulation composes without a history buffer.
    if let Some(rs) = restir_spatial.as_deref_mut() {
        let di_spatial = restir_rt.is_some_and(|rt| rt.spatial.is_some())
            || reference.is_some_and(|r| r.di_spatial().is_some());
        let gi_spatial = restir_rt.is_some_and(|rt| rt.gi && rt.spatial.is_some())
            || reference.is_some_and(|r| r.gi_spatial().is_some());
        // GI reservoir finalize (phase 1): whenever the GI ReSTIR arm is on
        // (estimator bit 5), the temporal merge + reservoir shade run HERE —
        // raygen only exports the canonical sample (its sample loop is past a
        // register cliff; see raygen.slang's bit-5 comment). Must run before
        // the spatial dispatch so neighbors read merged chains.
        let gi_finalize = restir_rt.is_some_and(|rt| rt.gi)
            || reference.is_some_and(SolariReference::gi_restir);
        let spatial_on = (di_spatial || gi_spatial) && debug_view == 0 && !show_displacement;
        let finalize_on = gi_finalize && debug_view == 0 && !show_displacement;
        if spatial_on || finalize_on {
            // Queued by `queue_restir_spatial_pipeline` (Prepare) as soon as the
            // scene-columns layout exists — compiled with the cold-start batch.
            if let Some(spatial_pipeline) = rs
                .pipeline
                .and_then(|id| pipeline_cache.get_compute_pipeline(id))
            {
                let blend_w = if accum_spf > 0 && accum_n > 0 {
                    accum_spf as f32 / (accum_n + accum_spf) as f32
                } else {
                    1.0
                };
                let base = RestirSpatialParams {
                    width: viewport.x,
                    height: viewport.y,
                    parity: camera_inputs.frame[0] & 1,
                    frame: camera_inputs.frame[0],
                    taps: 0,
                    radius: 0.0,
                    blend_w,
                    firefly_clamp: camera_inputs.dims[3],
                    unbiased: 0,
                    pad_a: 0,
                    di_on: 0,
                    gi_on: 0,
                    phase: 0,
                    pad_b: 0,
                    pad_c: 0,
                    pad_d: 0,
                };
                render_queue.write_buffer(
                    &rs.geo_addr,
                    0,
                    bytemuck::bytes_of(&geo_addrs.unwrap_or(bytemuck::Zeroable::zeroed())),
                );
                let layout = pipeline_cache.get_bind_group_layout(&rs.layout);
                let mut dispatch = |params_buf: &bevy_render::render_resource::Buffer,
                                    params: RestirSpatialParams,
                                    label: &'static str| {
                    render_queue.write_buffer(params_buf, 0, bytemuck::bytes_of(&params));
                    let bind_group = render_device.create_bind_group(
                        label,
                        &layout,
                        &BindGroupEntries::sequential((
                            output.reservoirs.as_entire_binding(),
                            output.surface.as_entire_binding(),
                            output.buffer.as_entire_binding(),
                            params_buf.as_entire_binding(),
                            rs.geo_addr.as_entire_binding(),
                            output.light_samples.as_entire_binding(),
                            output.gi_samples.as_entire_binding(),
                            output.camera_buffer.as_entire_binding(),
                        )),
                    );
                    let encoder = ctx.command_encoder();
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some(label),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(spatial_pipeline);
                    pass.set_bind_group(0, scene_bg, &[]);
                    pass.set_bind_group(1, &bind_group, &[]);
                    pass.set_bind_group(2, columns_bg, &[]);
                    pass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);
                };
                if finalize_on {
                    dispatch(
                        &rs.params_finalize,
                        RestirSpatialParams { phase: 1, ..base },
                        "restir_gi_finalize",
                    );
                }
                if spatial_on {
                    // One parameter set feeds the pass (DI and GI arms share it);
                    // prefer the realtime settings, then the DI arm's, then the GI arm's.
                    let sp = restir_rt
                        .and_then(|rt| rt.spatial.as_ref())
                        .or_else(|| reference.and_then(SolariReference::di_spatial))
                        .or_else(|| reference.and_then(SolariReference::gi_spatial))
                        .expect("spatial_on requires a configured spatial arm");
                    dispatch(
                        &rs.params,
                        RestirSpatialParams {
                            taps: sp.taps.min(8),
                            radius: sp.radius,
                            unbiased: sp.unbiased_zcount as u32,
                            pad_a: sp.debug_paint as u32,
                            di_on: di_spatial as u32,
                            gi_on: gi_spatial as u32,
                            ..base
                        },
                        "restir_spatial",
                    );
                }
            } else {
                bevy_log::warn_once!("restir_spatial: pipeline not ready — DI missing this frame");
            }
        }
    }

    // NRC termination-query inference + composite: batch-evaluate the MLP for
    // every query raygen appended and add throughput × cache into the output
    // buffer — before training so this frame's queries see this frame's
    // weights. `scale` folds in the raygen accumulation blend (w / rounds), so
    // the composite lands exactly as if the radiance had been added in-loop.
    // Skipped in debug views (they paint over the output). Runs only when
    // bit 20 is armed with ReSTIR GI (bit 5) OFF — with GI reservoirs the
    // raygen terminates via the inline query instead (this deferred composite
    // would land after the reservoir already stored its shade), so the ring
    // stays empty. DLSS RR: the composite adds the cache tail to the output
    // buffer only, not the RR diffuse/specular guides — acceptable, the
    // guides are primary-surface attributes, not radiance.
    if let (Some(nrc_bufs), Some(nrc_pipelines)) =
        (debug.nrc_buffers.as_deref(), debug.nrc_pipelines.as_deref())
    {
        let nrc_batched =
            estimator_flags & (1 << 20) != 0 && estimator_flags & (1 << 5) == 0;
        if debug.nrc.as_deref().is_some_and(|n| n.enabled && !n.inline_coopvec)
            && nrc_batched
            && debug_view == 0
            && !show_displacement
        {
            let rounds = accum_spf.max(1) as f32;
            let blend = if accum_spf > 0 && accum_n > 0 {
                accum_spf as f32 / (accum_n as f32 + accum_spf as f32)
            } else {
                1.0
            };
            let _ = crate::nrc::dispatch_nrc_query_infer(
                ctx.command_encoder(),
                nrc_bufs,
                nrc_pipelines,
                &render_device,
                &render_queue,
                &output.nrc_queries,
                &output.buffer,
                output.pixels as u32,
                blend / rounds,
                camera.exposure,
            );
        }
    }

    // NRC online training: this frame's raygen-written
    // records → encode → fwd → loss → bwd → adam, all on the shared ctx encoder.
    // The inference mirrors the NEXT frame's raygen reads update at the end —
    // one frame of cache latency, invisible for a cache converging over dozens.
    if let (Some(nrc_bufs), Some(nrc_pipelines)) =
        (debug.nrc_buffers.as_deref_mut(), debug.nrc_pipelines.as_deref())
    {
        if let Some(nrc_cfg) = debug.nrc.as_deref() {
            // `wants_nrc`: no consumer ⇒ no training — raygen wrote no fresh
            // records this frame (nrc.x is zeroed), so a step would train on
            // a stale ring while burning the full training cost.
            if nrc_cfg.enabled
                && nrc_cfg.training
                && wants_nrc
                && (*frame_counter).is_multiple_of(nrc_cfg.train_interval.max(1))
            {
                let _ = crate::nrc::dispatch_training(
                    &mut ctx,
                    nrc_bufs,
                    nrc_pipelines,
                    &render_device,
                    &render_queue,
                    nrc_cfg,
                );
            }
        }
    }

    // Blit the per-pixel output buffer into the view's HDR storage texture (a
    // normal wgpu compute pass on the shared ctx encoder → runs after the trace
    // buffer, so the view target stays wgpu-layout-tracked). The diff view
    // rides here: |current − frozen| heatmap when enabled.
    // Exposure applies HERE: the buffer holds physical radiance. Debug views
    // paint raw non-radiance values → exposure 1.0 so they display verbatim.
    let frozen_valid = frozen.filter(|f| f.pixels == output.pixels);
    let diff_on = debug
        .freeze_diff
        .as_deref()
        .is_some_and(|fd| fd.diff && frozen_valid.is_some());
    let diff_scale = debug.freeze_diff.as_deref().map_or(4.0, |fd| fd.diff_scale);
    let debug_paint = debug_view != 0
        || show_displacement
        || reference.is_some_and(|r| r.spatial_debug_paint() || r.gi_dead_view());
    let blit_exposure = if debug_paint { 1.0 } else { camera.exposure };
    render_queue.write_buffer(
        &rt_blit.params,
        0,
        bytemuck::bytes_of(&[diff_on as u32 as f32, diff_scale, blit_exposure, 0.0]),
    );
    let frozen_binding = frozen_valid.map_or(&output.buffer, |f| &f.buffer);
    let bind_group = render_device.create_bind_group(
        "rt_blit_bind_group",
        &pipeline_cache.get_bind_group_layout(&rt_blit.layout),
        &BindGroupEntries::sequential((
            output.buffer.as_entire_binding(),
            view_target.get_unsampled_color_attachment().view,
            frozen_binding.as_entire_binding(),
            rt_blit.params.as_entire_binding(),
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
