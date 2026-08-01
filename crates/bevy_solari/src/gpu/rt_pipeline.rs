// Raw Vulkan ray-tracing PIPELINE (raygen / closest-hit / miss / any-hit +
// shader binding table + cmd_trace_rays), the multi-material SBT shading path.
// Self-contained: builds its own descriptor set layout, pool, sets, pipeline
// layout, pipeline, and SBT in raw `ash` — no wgpu-hal accessor additions. The
// RT-stage shaders are all Slang: precompiled SPIR-V blobs, except the primary
// miss (and any downstream `SolariChitSource::Slang` closest-hit), compiled at
// build via `gpu/slang.rs`, then handed to `vkCreateRayTracingPipelinesKHR`.
//
// Mirrors `gpu/allocator.rs`'s raw-VK style; gated on the
// `RayTracingPipelineFeature` device feature.
#![allow(unsafe_code)]

use ash::khr;
use ash::vk::{self, TaggedStructure};
use bevy_ecs::component::Component;
use bevy_ecs::resource::Resource;
use core::ffi::CStr;
#[cfg(test)]
use wgpu::naga;

use super::allocator::Allocator;

// RT-pipeline-private descriptor set (set 1). Set 0 is the shared scene bind
// group (raytracing_scene_bindings, incl. the TLAS at its binding 9) and set 2
// is the scene-columns bind group — both built by wgpu and bound raw via
// `BindGroup::as_hal`, so the chits' `#import bevy_solari::scene_bindings` etc.
// resolve real geometry/materials/lights/textures.
const BINDING_OUTPUT: u32 = 0; // storage buffer, vec4<f32> per pixel (raygen writes)
const BINDING_CAMERA: u32 = 1; // uniform buffer (ray gen inputs)
const BINDING_ENV_MAP: u32 = 2; // environment/skybox cube (miss samples)
const BINDING_ENV_SAMPLER: u32 = 3; // sampler for the environment cube
const BINDING_GEOMETRY: u32 = 4; // uniform: bindless geometry buffer-device-addresses (chit reads)
// DLSS Ray Reconstruction guide G-buffer (set 1, written chit-direct). Present only
// with the `dlss` feature; the WGSL declares the matching `@group(1)` bindings under
// `#ifdef SOLARI_DLSS`, so the layout slots and the SPIR-V binding numbers stay in
// lockstep (set 1 is hand-built, so naga emits these numbers verbatim — no wgpu
// descriptor compaction). Contiguous after BINDING_GEOMETRY (no gaps).
const BINDING_GBUFFER_NORMAL: u32 = 5; // storage: normal.xyz + linear roughness (.w)
const BINDING_GBUFFER_DIFFUSE: u32 = 6; // storage: diffuse albedo.xyz + linear depth (.w)
const BINDING_GBUFFER_SPECULAR: u32 = 7; // storage: specular albedo.xyz + hit distance (.w)
const BINDING_GBUFFER_MOTION: u32 = 8; // storage: screen-space motion vector.xy (.zw unused)
// ReSTIR DI reservoirs: 2 interleaved 32-B slots per pixel, written by
// raygen (current-slot clear) + the opaque closest-hit (merge/store). Always
// present — a fixed binding number past the DLSS range; Vulkan set layouts
// tolerate the 5–8 gap when the `dlss` feature is off.
const BINDING_RESERVOIRS: u32 = 9;
// ReSTIR primary-hit surface G-buffer (48 B/pixel): the chit writes it when the
// spatial pass is on; the wgpu spatial pass reads it. Always bound.
const BINDING_SURFACE: u32 = 10;
// ReSTIR winner resolved-light samples (2 slots × 48 B/pixel): chit-written, read
// by the wgpu spatial pass so it can reshade without `physical_load`. Always bound.
const BINDING_LIGHT_SAMPLES: u32 = 11;
/// ReSTIR GI canonical samples: 2 interleaved 48-B slots per pixel.
const BINDING_GI_SAMPLES: u32 = 12;
// NRC: transposed f16 weight mirror + f16 biases (raygen
// inline coopvec inference), the training-record ring (raygen writes one
// record per rotating pixel subset each frame), and the per-view
// termination-query ring (raygen appends, the infer pass consumes).
const BINDING_NRC_WEIGHTS: u32 = 13;
const BINDING_NRC_BIAS: u32 = 14;
const BINDING_NRC_RECORDS: u32 = 15;
const BINDING_NRC_QUERIES: u32 = 16;

/// Per-frame camera inputs the raygen shader reads — std140-compatible
/// (mat4 + vec4). `inverse_view_proj` reconstructs a world-space ray per pixel;
/// `camera_position` is the ray origin.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RtCamera {
    pub inverse_view_proj: [f32; 16],
    /// View-from-world — the DLSS guide's view-space linear depth (`-view_z`).
    pub view_from_world: [f32; 16],
    /// Unjittered clip-from-world this frame — DLSS motion vectors (current).
    pub clip_from_world: [f32; 16],
    /// Unjittered clip-from-world last frame — DLSS motion vectors (previous).
    pub prev_clip_from_world: [f32; 16],
    pub camera_position: [f32; 4],
    /// `frame_index` in `.x` (RNG seed for temporal variation); `.yzw` pad to a
    /// 16-byte std140 slot.
    pub frame: [u32; 4],
    /// `.x` = sky/environment brightness (raw cd/m²; 0 ⇒ no skybox, miss stays
    /// at the clear color in `.yzw`; < 0 ⇒ the composed `custom_sky` module
    /// evaluates the sky instead).
    pub sky: [f32; 4],
    /// `.xy` = sub-pixel camera jitter in pixels (added to the primary ray);
    /// `.zw` = debug-heatmap colormap params (center/contrast — one debug view
    /// active at a time). Jitter zero until DLSS drives it from `suggested_jitter`.
    pub jitter: [f32; 4],
    /// `.x` = wall-clock time in seconds (wrapped) for animated surfaces (waves,
    /// gas-giant bands); `.y` = per-pixel ray-cone tangent (`2·tan(fov/2)/height`)
    /// for footprint-based shading LOD; `.zw` reserved.
    pub misc: [f32; 4],
    /// World→bake quaternion (xyzw) for the environment/atmosphere cube: the
    /// miss shader rotates sample directions by it so the baked sky (up = +Y
    /// convention) follows a spherical planet's local frame. Identity
    /// `(0,0,0,1)` for flat scenes / plain skyboxes.
    pub sky_frame: [f32; 4],
    /// Atmosphere volumes: `.xy` = device address (lo/hi bits) of the
    /// `GpuAtmosphereVolumes` buffer, `.z` = live volume count (0 ⇒ raygen
    /// skips the march and the miss shader keeps the cube on primary rays).
    pub atmo: [f32; 4],
    /// `.xy` = viewport pixels (restir temporal reprojection); `.z` = history
    /// M-cap as a multiple of the candidate count.
    pub dims: [f32; 4],
    /// World-from-view camera basis — cylindrical-window raygen rotates view-space
    /// ray directions into world space (w=0, so only the linear part is read).
    pub world_from_view: [f32; 16],
    /// Cylindrical window (head-coupled curved screen): `.x` = total horizontal arc
    /// angle (rad), `.y` = curvature radius m (0 ⇒ mode off, planar unproject),
    /// `.z` = screen height m, `.w` reserved.
    pub window_arc: [f32; 4],
    /// `.xyz` = viewer eye in screen space (screen-center origin, +X right, +Y up,
    /// +Z toward viewer, meters) — matches the camera's local translation.
    pub window_eye: [f32; 4],
    /// `.xyz` = this frame's floating-origin translation minus last frame's
    /// (`origin_now − origin_prev`, f64-computed GPU-side). World_rel is
    /// camera-origin, so cross-frame reservoir positions must be rebased by
    /// this or they go stale every frame the camera moves (motion glitter).
    /// Zero on frame 1 and on the CPU fallback path.
    pub origin_delta: [f32; 4],
    /// NRC: `.x` = position-encoding scene scale in meters
    /// (0 ⇒ NRC off: no record writes, debug view black); `.y` = spread-
    /// termination threshold c; `.z` = inline coopvec inference (0 = deferred).
    pub nrc: [f32; 4],
    /// NRC world-snapped anchor offset (`.xyz`, GPU-computed from the camera's
    /// absolute f64 position): camera-relative positions plus this are
    /// anchor-relative, so the cache encoding survives camera translation.
    /// Zero on the CPU fallback path (camera-anchored there).
    pub nrc_anchor: [f32; 4],
}

/// Bindless geometry buffer-device-addresses the closest-hit reads via
/// `physical_load` (set 1, binding 4). `_pad` rounds to a 16-byte std140 slot.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RtGeometryAddresses {
    /// Base device address of the interleaved [`PackedVertex`] pool
    /// (`ClusterMeshManager::vertex_packed`) — 16-byte shading attrs (normal,
    /// tangent, uv); position is NOT here.
    pub vertex_packed: u64,
    /// Base device address of the SoA position pool
    /// (`ClusterMeshManager::vertex_positions`, stride 12). Position source for the
    /// sampling/non-hit-triangle resolve path (the closest-hit gets positions from
    /// the AS via position fetch instead).
    pub vertex_positions: u64,
    /// Base device address of the materials storage buffer.
    pub materials: u64,
    /// Byte stride of one material record (`GPU_MATERIAL_SIZE`).
    pub material_stride: u32,
    pub _pad: u32,
    /// Base device address of the tessellation per-CLAS metadata table (one
    /// `TessCluster` per sentinel tess CLAS: its per-instance smooth-normal buffer
    /// address + primitive base). `0` when the smooth-tess path is off — the
    /// closest-hit then shades tess hits with the facet normal. See
    /// [`crate::geometry::tess_classify::TessClassify`].
    pub tess_clusters: u64,
    /// Base device address of the optional per-vertex custom-data pool
    /// (`ClusterMeshManager::vertex_custom`, stride 4). A custom closest-hit reads
    /// `vertex_custom[global_vertex_index]`; the built-in chits never touch it.
    pub vertex_custom: u64,
    /// Deformed octahedral normals + tangents pools (`Deform`), and the slot-indexed
    /// animated table (flag/deform_pool_base/mesh_vertex_base per instance). `0` when
    /// no animation — the resolve then shades animated hits with rest-pose attrs.
    pub deform_normals: u64,
    pub deform_tangents: u64,
    pub animated_table: u64,
}

/// Closest-hit stage source. The built-in stages carry Slang-precompiled
/// SPIR-V (the regen command lives in each `.slang` header); downstream chits
/// may instead ship Slang source, compiled at pipeline build via `gpu/slang.rs`
/// with the built-in module set (`scene_resolve`/`brdf`/`sampling`/…) importable.
#[derive(Clone)]
pub enum SolariChitSource {
    /// Slang source compiled at pipeline build. `entry` is the entry function's
    /// name in `source` (the compiled `OpEntryPoint` is `"main"`, like every
    /// slangc-built stage). May `import` the built-in modules and the group's
    /// [`composable_modules`](SolariHitGroupDef::composable_modules).
    Slang {
        source: &'static str,
        file: &'static str,
        entry: &'static str,
    },
    /// Slang-precompiled SPIR-V words (entry point "main").
    SpirV(&'static [u8]),
}

/// One RT closest-hit program; its registry index is its SBT class. Sources are
/// `&'static` (usually `include_str!`/`include_bytes!`) so downstream crates
/// register without forking.
#[derive(Clone)]
pub struct SolariHitGroupDef {
    pub label: &'static str,
    pub closest_hit: SolariChitSource,
    pub any_hit: Option<SolariAnyHitDef>,
    /// Extra `(module_name, source)` Slang modules importable by this group's
    /// [`SolariChitSource::Slang`] closest-hit alongside the built-in set — lets
    /// a downstream crate `import` its own shared Slang (e.g. a terrain function
    /// used by both a compute pass and a closest-hit) without forking. They may
    /// import the built-ins and each other. Scoped to this group: other groups
    /// never see them. Unused by a `SpirV` closest-hit (precompiled stages
    /// resolve imports at regen time).
    pub composable_modules: &'static [(&'static str, &'static str)],
}

/// An any-hit program attached to a [`SolariHitGroupDef`] (alpha cutout, etc.).
/// Any-hits are Slang-precompiled SPIR-V (see `ahit_alpha.slang` — the regen
/// command is in its header): they never compose runtime modules, so nothing
/// needs the WGSL composer here.
#[derive(Clone)]
pub struct SolariAnyHitDef {
    pub spirv: &'static [u8],
    pub entry: &'static str,
}

/// Ordered RT hit groups consumed by [`RtPipeline::new`]; index = SBT class.
#[derive(bevy_ecs::resource::Resource, Default, Clone)]
pub struct SolariHitGroupRegistry {
    pub groups: Vec<SolariHitGroupDef>,
}

impl SolariHitGroupRegistry {
    /// Append a hit group; returns its SBT class (its index).
    ///
    /// A `Slang` closest-hit is compiled eagerly (with its `composable_modules`)
    /// so a broken user shader is reported at registration — at pipeline-build
    /// time a compile failure in ANY group aborts the whole RT pipeline, which
    /// is far harder to attribute.
    pub fn register(&mut self, group: SolariHitGroupDef) -> u32 {
        if matches!(group.closest_hit, SolariChitSource::Slang { .. }) {
            if let Err(e) = compile_slang_chit(&group) {
                bevy_log::error!(
                    "rt_pipeline: hit group '{}': closest-hit failed to compile: {e}. \
                     The RT pipeline will fail to build until this is fixed.",
                    group.label
                );
            }
        }
        let class = self.groups.len() as u32;
        self.groups.push(group);
        class
    }
}

/// Ray payload / hit-attribute sizes shared by every pipeline library and the
/// linked pipeline (`VkRayTracingPipelineInterfaceCreateInfoKHR`, mandatory
/// once stages live in libraries, and required to agree across the link).
/// Payload: `RtPayload` (rt_payload.slang) is 128 B under std430 (vec3 slots
/// pad to 16 B), plus headroom for a couple of future fields. Attributes:
/// triangle/LSS barycentrics, two floats.
const MAX_RAY_PAYLOAD_SIZE: u32 = 160;
const MAX_HIT_ATTRIBUTE_SIZE: u32 = 8;

/// One cached `VK_KHR_pipeline_library` compile (a stage subset of the RT
/// pipeline) plus the shader modules it references.
struct RtLibrary {
    pipeline: vk::Pipeline,
    modules: Vec<vk::ShaderModule>,
}

/// Render-world cache of RT pipeline LIBRARIES, living across [`RtPipeline`]
/// rebuilds so each rebuild recompiles only what changed and RELINKS the
/// rest: a custom-sky swap recompiles the primary-miss library alone; a new
/// registry hit group compiles just its own library; SBT growth or material
/// class churn relinks with zero shader compiles. Also owns the shared
/// set-1 descriptor layout + pipeline layout every library and every linked
/// pipeline is built against — the cache is recreated wholesale when either
/// wgpu-owned raw set layout changes (the dispatch compares
/// [`Self::layout_key`]).
#[derive(Resource)]
pub struct RtLibraryCache {
    device: ash::Device,
    rt: khr::ray_tracing_pipeline::Device,
    /// The wgpu-owned scene (set 0) + columns (set 2) raw layouts the shared
    /// pipeline layout bakes in; a handle change invalidates the whole cache.
    layout_key: (vk::DescriptorSetLayout, vk::DescriptorSetLayout),
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    raygen: Option<RtLibrary>,
    /// Composed with the `custom_sky` module; the key is the module source's
    /// generation, so a sky swap rebuilds exactly this library.
    miss: Option<(u64, RtLibrary)>,
    miss_shadow: Option<RtLibrary>,
    /// Index-aligned with [`SolariHitGroupRegistry::groups`], which is
    /// append-only — existing entries never change identity, so cached
    /// libraries stay valid and only NEW registry entries compile.
    hit_groups: Vec<RtLibrary>,
    /// See the twin field on [`RtPipeline`].
    _device_keepalive: Allocator,
}

// SAFETY: plain Vulkan handles; used solely from the single render-schedule
// dispatch system.
unsafe impl Send for RtLibraryCache {}
unsafe impl Sync for RtLibraryCache {}

impl RtLibraryCache {
    /// Build the shared set-1 descriptor layout + pipeline layout. Libraries
    /// compile lazily via the `ensure_*` methods on first pipeline build.
    pub fn new(
        allocator: &Allocator,
        scene_layout: vk::DescriptorSetLayout,
        columns_layout: vk::DescriptorSetLayout,
    ) -> Option<Self> {
        let device = allocator.device().clone();
        // SAFETY: instance + device are live; loading the RT-pipeline function
        // table is valid because the extension was enabled at device creation.
        let rt = khr::ray_tracing_pipeline::Device::load(allocator.instance(), &device);

        // --- Descriptor set layout (set 1: output + camera) --------------------
        // TLAS is NOT here — it comes from the scene bind group (set 0). raygen
        // writes the output buffer; raygen reads the camera.
        let mut bindings = vec![
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_OUTPUT)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_CAMERA)
                // The camera is a GPU buffer filled by the `rt_camera` compute pass (or a
                // CPU fallback `write_buffer`) each frame; `*_DYNAMIC` bound at a constant
                // offset of 0 (see `trace`).
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(1)
                // raygen unprojects; miss reads sky brightness/clear color; the
                // closest-hit reads the view matrices for the DLSS depth/motion guide.
                .stage_flags(
                    vk::ShaderStageFlags::RAYGEN_KHR
                        | vk::ShaderStageFlags::MISS_KHR
                        | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                ),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_ENV_MAP)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::MISS_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_ENV_SAMPLER)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::MISS_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_GEOMETRY)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                // The closest-hit's geometry resolve AND the alpha-cutout any-hit both
                // read the bindless geometry addresses (packed-vertex UV + materials).
                .stage_flags(
                    vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::ANY_HIT_KHR,
                ),
        ];
        // DLSS guide G-buffers: raygen clears the primary pixel (sky/miss default);
        // the closest-hit overwrites it on a primary hit. Layout slots must match the
        // WGSL `#ifdef SOLARI_DLSS` bindings 5/6/7 exactly.
        for binding in [
            BINDING_GBUFFER_NORMAL,
            BINDING_GBUFFER_DIFFUSE,
            BINDING_GBUFFER_SPECULAR,
            BINDING_GBUFFER_MOTION,
        ] {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(
                        vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                    ),
            );
        }
        // ReSTIR reservoirs: raygen clears the current slot, the chit merges + stores.
        // Surface G-buffer: the chit writes it for the spatial merge+shade pass.
        for binding in [BINDING_RESERVOIRS, BINDING_SURFACE, BINDING_LIGHT_SAMPLES, BINDING_GI_SAMPLES, BINDING_NRC_WEIGHTS, BINDING_NRC_BIAS, BINDING_NRC_RECORDS, BINDING_NRC_QUERIES] {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(
                        vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                    ),
            );
        }
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: well-formed create info; device live.
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&dsl_info, None) }.ok()?;

        // Pipeline layout: [scene (set 0), rt-private (set 1), columns (set 2)].
        let set_layouts = [scene_layout, descriptor_set_layout, columns_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: well-formed; device live; the scene/columns layouts outlive this
        // cache (owned by wgpu's bind-group-layout cache; the dispatch recreates
        // the cache when they change).
        let pipeline_layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
            Ok(l) => l,
            Err(e) => {
                bevy_log::error!("rt_pipeline: create_pipeline_layout failed: {e:?}");
                // SAFETY: just created above; nothing references it yet.
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return None;
            }
        };

        Some(Self {
            device,
            rt,
            layout_key: (scene_layout, columns_layout),
            pipeline_layout,
            descriptor_set_layout,
            raygen: None,
            miss: None,
            miss_shadow: None,
            hit_groups: Vec::new(),
            _device_keepalive: allocator.clone(),
        })
    }

    /// The wgpu raw layouts this cache's pipeline layout was built against;
    /// the dispatch recreates the cache when they no longer match.
    pub fn layout_key(&self) -> (vk::DescriptorSetLayout, vk::DescriptorSetLayout) {
        self.layout_key
    }

    /// Compile one library: `stages` + `groups` against the shared layout,
    /// with the shared ray interface. Every library (and the link) opts into
    /// cluster acceleration structures and opacity micromaps — these must
    /// agree across the whole linked pipeline.
    fn create_library(
        &self,
        stages: &[vk::PipelineShaderStageCreateInfo],
        groups: &[vk::RayTracingShaderGroupCreateInfoKHR],
        modules: Vec<vk::ShaderModule>,
    ) -> Option<RtLibrary> {
        let cluster_info =
            vk::RayTracingPipelineClusterAccelerationStructureCreateInfoNV::default()
                .allow_cluster_acceleration_structure(true);
        let interface = vk::RayTracingPipelineInterfaceCreateInfoKHR::default()
            .max_pipeline_ray_payload_size(MAX_RAY_PAYLOAD_SIZE)
            .max_pipeline_ray_hit_attribute_size(MAX_HIT_ATTRIBUTE_SIZE);
        let mut info = vk::RayTracingPipelineCreateInfoKHR::default()
            .flags(
                vk::PipelineCreateFlags::LIBRARY_KHR
                    | vk::PipelineCreateFlags::RAY_TRACING_OPACITY_MICROMAP_EXT,
            )
            .stages(stages)
            .groups(groups)
            // Depth 2: raygen's hit object executes the closest-hit (1), which
            // traces a NEE shadow ray (2). Must agree with the link.
            .max_pipeline_ray_recursion_depth(2)
            .library_interface(&interface)
            .layout(self.pipeline_layout);
        // ash doesn't register the cluster struct as an extender (no typed
        // `push_next`); chain it via raw `p_next`. It outlives the call.
        info.p_next =
            (&cluster_info as *const vk::RayTracingPipelineClusterAccelerationStructureCreateInfoNV)
                .cast();
        // SAFETY: stages/groups reference live modules; layout live.
        match unsafe {
            self.rt.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[info],
                None,
            )
        } {
            Ok(p) => Some(RtLibrary {
                pipeline: p.into_iter().next()?,
                modules,
            }),
            Err(e) => {
                bevy_log::error!("rt_pipeline: library compile failed: {:?}", e.1);
                for m in modules {
                    // SAFETY: modules were created for this library; nothing
                    // else references them.
                    unsafe { self.device.destroy_shader_module(m, None) };
                }
                None
            }
        }
    }

    fn destroy_library(&self, lib: RtLibrary) {
        // The linked pipeline that referenced this library is already gone (the
        // dispatch removes RtPipeline before rebuilding); drain in-flight work
        // then destroy.
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe {
            self.device.destroy_pipeline(lib.pipeline, None);
            for m in lib.modules {
                self.device.destroy_shader_module(m, None);
            }
        }
    }

    /// Raygen library (static per app run — the shader-clock variant choice
    /// is fixed at device creation).
    fn ensure_raygen(&mut self) -> Option<()> {
        if self.raygen.is_some() {
            return Some(());
        }
        // Slang-precompiled raygen (regen commands in raygen.slang): the
        // `SOLARI_SHADER_CLOCK` variant carries the cost-heatmap clock reads,
        // legal only when the device enabled `VK_KHR_shader_clock`.
        let raygen_spv: &'static [u8] = if crate::gpu::extension::shader_clock_available() {
            include_bytes!("../render/rt_pipeline/raygen_clock.spv")
        } else {
            include_bytes!("../render/rt_pipeline/raygen.spv")
        };
        let module = create_shader_module(&self.device, &spirv_words(raygen_spv))?;
        let lib = self.create_library(
            &[shader_stage(vk::ShaderStageFlags::RAYGEN_KHR, module, c"main")],
            &[general_group(0)],
            vec![module],
        )?;
        self.raygen = Some(lib);
        Some(())
    }

    /// Primary-miss library — the one stage compiled at BUILD time: it
    /// composes the swappable `custom_sky` module (`SolariSky::Shader`;
    /// defaults to the built-in procedural gradient), so its SPIR-V can't be
    /// checked in. A generation change rebuilds exactly this library.
    fn ensure_miss(&mut self, custom_sky: (&str, u64)) -> Option<()> {
        let (custom_sky_source, generation) = custom_sky;
        if matches!(&self.miss, Some((cached, _)) if *cached == generation) {
            return Some(());
        }
        let miss_spv = crate::gpu::slang::compile_rt_slang(
            "miss.slang",
            include_str!("../render/rt_pipeline/miss.slang"),
            "miss_primary",
            crate::gpu::slang::SlangRtStage::Miss,
            &[
                (
                    "rt_payload",
                    include_str!("../render/rt_pipeline/rt_payload.slang"),
                ),
                ("custom_sky", custom_sky_source),
            ],
        )
        .map_err(|e| bevy_log::error!("rt_pipeline: {e}"))
        .ok()?;
        let module = create_shader_module(&self.device, &miss_spv)?;
        let lib = self.create_library(
            &[shader_stage(vk::ShaderStageFlags::MISS_KHR, module, c"main")],
            &[general_group(0)],
            vec![module],
        )?;
        if let Some((_, old)) = self.miss.take() {
            self.destroy_library(old);
        }
        self.miss = Some((generation, lib));
        Some(())
    }

    /// Shadow-miss library (Slang-precompiled, static).
    fn ensure_shadow(&mut self) -> Option<()> {
        if self.miss_shadow.is_some() {
            return Some(());
        }
        // Slang-precompiled (see miss_shadow.slang for the regen command).
        let module = create_shader_module(
            &self.device,
            &spirv_words(include_bytes!("../render/rt_pipeline/miss_shadow.spv")),
        )?;
        let lib = self.create_library(
            &[shader_stage(vk::ShaderStageFlags::MISS_KHR, module, c"main")],
            &[general_group(0)],
            vec![module],
        )?;
        self.miss_shadow = Some(lib);
        Some(())
    }

    /// One library per registry hit group (chit + optional any-hit). The
    /// registry is append-only, so only entries past the cached count compile.
    fn ensure_hit_groups(&mut self, hit_groups: &[SolariHitGroupDef]) -> Option<()> {
        for hg in &hit_groups[self.hit_groups.len()..] {
            let chit_mod = match &hg.closest_hit {
                SolariChitSource::Slang { .. } => {
                    let spv = compile_slang_chit(hg)
                        .map_err(|e| bevy_log::error!("rt_pipeline: {e}"))
                        .ok()?;
                    create_shader_module(&self.device, &spv)?
                }
                SolariChitSource::SpirV(bytes) => {
                    create_shader_module(&self.device, &spirv_words(bytes))?
                }
            };
            let lib = if let Some(ah) = &hg.any_hit {
                let ah_mod = create_shader_module(&self.device, &spirv_words(ah.spirv))?;
                let entry = std::ffi::CString::new(ah.entry)
                    .expect("shader entry name has interior NUL");
                self.create_library(
                    &[
                        shader_stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, chit_mod, c"main"),
                        shader_stage(vk::ShaderStageFlags::ANY_HIT_KHR, ah_mod, entry.as_c_str()),
                    ],
                    &[hit_group_with_any_hit(0, 1)],
                    vec![chit_mod, ah_mod],
                )?
            } else {
                self.create_library(
                    &[shader_stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, chit_mod, c"main")],
                    &[hit_group(0)],
                    vec![chit_mod],
                )?
            };
            self.hit_groups.push(lib);
        }
        Some(())
    }

    /// Link the cached libraries into an executable pipeline. Group numbering
    /// is the concatenation of the libraries' groups in list order — raygen
    /// (0), primary miss (1), hit groups (2..), shadow miss last — matching
    /// the SBT layout [`RtPipeline::new`] bakes.
    fn link(&self) -> Option<vk::Pipeline> {
        let mut libs: Vec<vk::Pipeline> =
            Vec::with_capacity(3 + self.hit_groups.len());
        libs.push(self.raygen.as_ref()?.pipeline);
        libs.push(self.miss.as_ref()?.1.pipeline);
        libs.extend(self.hit_groups.iter().map(|l| l.pipeline));
        libs.push(self.miss_shadow.as_ref()?.pipeline);

        let cluster_info =
            vk::RayTracingPipelineClusterAccelerationStructureCreateInfoNV::default()
                .allow_cluster_acceleration_structure(true);
        let library_info = vk::PipelineLibraryCreateInfoKHR::default().libraries(&libs);
        let interface = vk::RayTracingPipelineInterfaceCreateInfoKHR::default()
            .max_pipeline_ray_payload_size(MAX_RAY_PAYLOAD_SIZE)
            .max_pipeline_ray_hit_attribute_size(MAX_HIT_ATTRIBUTE_SIZE);
        // Opt into opacity micromaps. Unlike ray queries (which honor OMM straight
        // from the AS), a ray-tracing *pipeline* ignores opacity micromaps entirely
        // unless created with this flag — the driver invokes the any-hit shader on
        // every micro-triangle as if no OMM were present. OMM is a required
        // extension (solari disables without it), so the flag is unconditional.
        //
        // Dynamic stack size: the driver's DEFAULT stack for a pipeline linked
        // from libraries is unreliable (computed per library, not across the
        // link), and an undersized stack corrupts payload/local spills with no
        // validation error. The dispatch sets an explicit size (queried per
        // group) via `vkCmdSetRayTracingPipelineStackSizeKHR` — which requires
        // opting into the dynamic state here.
        let dynamic_states = [vk::DynamicState::RAY_TRACING_PIPELINE_STACK_SIZE_KHR];
        let dynamic_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut info = vk::RayTracingPipelineCreateInfoKHR::default()
            .flags(vk::PipelineCreateFlags::RAY_TRACING_OPACITY_MICROMAP_EXT)
            .max_pipeline_ray_recursion_depth(2)
            .library_info(&library_info)
            .library_interface(&interface)
            .dynamic_state(&dynamic_info)
            .layout(self.pipeline_layout);
        info.p_next =
            (&cluster_info as *const vk::RayTracingPipelineClusterAccelerationStructureCreateInfoNV)
                .cast();
        // SAFETY: libraries + layout live (owned by this cache).
        match unsafe {
            self.rt.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[info],
                None,
            )
        } {
            Ok(p) => p.into_iter().next(),
            Err(e) => {
                bevy_log::error!("rt_pipeline: pipeline link failed: {:?}", e.1);
                None
            }
        }
    }
}

impl Drop for RtLibraryCache {
    fn drop(&mut self) {
        // In-flight traces may still reference the layouts; drain first.
        self._device_keepalive.quiesce_before_raw_destroy();
        let libs = self
            .raygen
            .take()
            .into_iter()
            .chain(self.miss.take().map(|(_, l)| l))
            .chain(self.miss_shadow.take())
            .chain(std::mem::take(&mut self.hit_groups));
        for lib in libs {
            // SAFETY: quiesced above; handles exclusively owned here.
            unsafe {
                self.device.destroy_pipeline(lib.pipeline, None);
                for m in lib.modules {
                    self.device.destroy_shader_module(m, None);
                }
            }
        }
        // SAFETY: the set-1 layout outlives the per-view pools/sets allocated
        // from it (destroying a layout with live sets is legal), and those
        // sets drop with their RtViewBindings.
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// A raw host-visible buffer kept with its memory + mapping, for the SBT and the
/// camera UBO (which `Allocator::create_buffer` can't expose — it hides the
/// `VkDeviceMemory`).
struct MappedBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    size: u64,
    device_address: vk::DeviceAddress,
}

/// Render-world resource owning the **view-independent** ray-tracing pipeline +
/// SBT + the set-1 descriptor *layout* and the shared env sampler. The per-view
/// resources (set-1 descriptor set, camera UBO, output-buffer binding, env cube)
/// live in [`RtViewBindings`], a component, so multiple views (split-screen) each
/// trace into their own output with their own camera/env.
///
/// Built lazily once the scene layouts and materials exist; `RayTracingPipelineFeature`
/// is a hard requirement, so absence here only ever means "not built yet".
#[derive(Resource)]
pub struct RtPipeline {
    device: ash::Device,
    rt: khr::ray_tracing_pipeline::Device,

    pipeline: vk::Pipeline,
    /// Handle COPIES of the shared layouts [`RtLibraryCache`] owns (the cache
    /// outlives every linked pipeline; a wgpu layout change recreates both).
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,

    sbt: MappedBuffer,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,
    callable_region: vk::StridedDeviceAddressRegionKHR,
    /// Number of per-material hit records the SBT holds. An instance routes to
    /// record = its material slot, so once the live material count exceeds this
    /// the pipeline must be rebuilt (the dispatch checks [`Self::capacity`]).
    record_capacity: u32,

    /// Per-material-slot SBT hit-group class the records were baked with (0 =
    /// opaque, 1 = glass, …; see `material::material_sbt_class`). Each record's
    /// shader handle is `handle(2 + class)`, so a material changing class (glass
    /// loading/unloading, a live edit) needs the records rebuilt — the handle is
    /// baked, unlike the per-frame GPU `FORCE_NO_OPAQUE` flag. The dispatch
    /// compares this via [`Self::classes_changed`].
    material_classes: Vec<u32>,

    /// Generation of the `bevy_solari::custom_sky` module source the primary miss
    /// shader was composed with. `SolariSky::Shader` swaps the module at runtime;
    /// the composed SPIR-V is baked, so a source change needs a pipeline rebuild —
    /// the dispatch compares this against [`SolariCustomSky`](crate::render::sky::SolariCustomSky).
    custom_sky_generation: u64,

    /// Explicit pipeline stack size (bytes), queried per group from the linked
    /// pipeline and set dynamically at trace time — the driver default is
    /// unreliable for library-linked pipelines.
    stack_size: u32,

    /// Keeps the `VkDevice` alive until this drops. The cloned `ash::Device`
    /// above is a bare handle + fn table with NO ownership: at app teardown the
    /// render world drops resources in arbitrary order, and if wgpu destroys the
    /// device first, [`Drop`]'s raw destroys segfault against a dead device. The
    /// allocator transitively holds wgpu's queue → device, so holding it pins
    /// the device across our Drop.
    _device_keepalive: Allocator,
}

// SAFETY: all fields are plain Vulkan handles owned by this resource; the only
// interior raw pointers (host-visible mappings) moved to RtViewBindings. Used
// solely from the single render-schedule dispatch system.
unsafe impl Send for RtPipeline {}
unsafe impl Sync for RtPipeline {}

/// Per-view ray-tracing resources: the set-1 descriptor set (output buffer @0,
/// camera UBO @1, env cube @2, env sampler @3), the camera UBO it points at, and
/// the env image to transition around the trace. One per [`SolariCamera`] view,
/// so split-screen views don't share an output buffer or camera. Built from
/// [`RtPipeline::create_view_bindings`]; rebuilt when the view's output buffer is
/// reallocated (viewport resize).
#[derive(Component)]
pub struct RtViewBindings {
    device: ash::Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    /// Bindless geometry addresses (set 1, binding 4); refreshed per frame via the
    /// mapping (`set_geometry_addresses`). Not ringed: the addresses are stable
    /// (stable-address `RawTraceBindable` buffers), so an in-flight overwrite writes
    /// identical bytes — benign, unlike the per-frame-varying camera.
    geometry: MappedBuffer,
    /// This view's own linear env-cube sampler (destroyed on drop). Owned here, not
    /// on `RtPipeline`, so the per-view set survives a pipeline rebuild — the set
    /// is compatible-by-content with the rebuilt set-1 layout and references
    /// nothing the rebuilt pipeline owns.
    env_map_sampler: vk::Sampler,
    /// `Some` ⇒ the env cube is the storage atmosphere cube (GENERAL); transition
    /// it around each trace. `None` ⇒ already a read-optimal wgpu-sampled texture.
    env_map_image: Option<vk::Image>,
    /// The output `VkBuffer` baked into binding 0. The dispatch rebuilds this
    /// component if the view's output buffer changes (resize) — the descriptor is
    /// written once and never updated (updating an in-flight set device-losts).
    output_buffer: vk::Buffer,
    /// The env cube view baked into binding 2, for the same rebuild check: a
    /// skybox that finishes loading (or is swapped) changes the view, and the
    /// baked-once set would otherwise sample the stale cube forever.
    env_map_view: vk::ImageView,
    /// Parallel descriptor-heap slots for this view's set-1 resources —
    /// M2 staging: allocated alongside the classic descriptor set when the
    /// [`BindingSeam`](crate::gpu::binding_seam::BindingSeam) exists, unread
    /// until the heap-flagged pipeline flip consumes them as mapping targets.
    #[expect(dead_code, reason = "M2 staging: read at the heap-pipeline flip")]
    heap: Option<RtViewHeapSlots>,
    /// Keeps the `VkDevice` alive until this drops — see the twin field on
    /// [`RtPipeline`]; without it, teardown drop order decides whether [`Drop`]'s
    /// raw destroys run against a dead device.
    _device_keepalive: Allocator,
}

/// The set-1 resources as descriptor-heap slots (buffer-region indices, plus
/// the env sampler in the sampler region). Slot order mirrors the set-1
/// binding list: output, camera (uniform), geometry (uniform), the DLSS
/// G-buffers, then reservoirs/surface/light_samples/gi_samples/nrc×4.
/// The env-cube IMAGE slot arrives with the scene-set conversion (heap image
/// descriptors are written from `ImageViewCreateInfo`, which needs the raw
/// image + params plumbed from the caller).
pub struct RtViewHeapSlots {
    seam: crate::gpu::binding_seam::BindingSeam,
    buffers: Vec<u32>,
    env_sampler: u32,
}

impl Drop for RtViewHeapSlots {
    fn drop(&mut self) {
        use crate::gpu::binding_seam::HeapKind;
        // The view's trace is drained by RtViewBindings' Drop (quiesce) before
        // these slots recycle.
        for &slot in &self.buffers {
            self.seam.free_heap_index(HeapKind::Buffer, slot);
        }
        self.seam.free_heap_index(HeapKind::Sampler, self.env_sampler);
    }
}

// SAFETY: the host-visible geometry-address mapping is written only from the single
// render-schedule dispatch (via `&self` + coherent memory), never shared across
// threads. All other fields are plain Vulkan handles.
unsafe impl Send for RtViewBindings {}
unsafe impl Sync for RtViewBindings {}

impl RtPipeline {
    /// Build the RT pipeline (raygen + miss + opaque/glass/hair closest-hit)
    /// by LINKING the per-stage libraries cached in `libraries` — only stages
    /// the cache hasn't seen (a new sky generation, a newly registered hit
    /// group) compile; everything else relinks. The cache also owns the
    /// shared pipeline layout (compatible with the wgpu scene/columns bind
    /// groups bound at trace time). Built lazily (see the dispatch) once
    /// those bind groups exist. Returns `None` if SPIR-V compilation or any
    /// Vulkan step fails (logged).
    pub fn new(
        allocator: &Allocator,
        libraries: &mut RtLibraryCache,
        material_classes: &[u32],
        hit_groups: &[SolariHitGroupDef],
        custom_sky: (&str, u64),
    ) -> Option<Self> {
        let custom_sky_generation = custom_sky.1;
        // One hit record per material slot; `material_classes[slot]` selects the
        // record's hit-group handle (opaque/glass/hair).
        let material_count = material_classes.len() as u32;
        let device = allocator.device().clone();
        let instance = allocator.instance();
        let physical_device = allocator.physical_device();

        // SAFETY: instance + device are live; loading the RT-pipeline function
        // table is valid because the extension was enabled at device creation.
        let rt = khr::ray_tracing_pipeline::Device::load(instance, &device);

        // Pipeline properties (SBT alignment), queried via the properties2 chain.
        let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut rt_props);
        // SAFETY: physical_device valid; props2 chain well-formed.
        unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
        let handle_size = rt_props.shader_group_handle_size as u64;
        let handle_align = rt_props.shader_group_handle_alignment as u64;
        let base_align = rt_props.shader_group_base_alignment as u64;

        // --- Stage libraries (cached) + link -----------------------------------
        // Fixed general programs (raygen + the two miss shaders); every closest-hit
        // ("hit group", + optional any-hit) comes from `hit_groups` (the registry), so
        // adding a surface shader needs no edit here — Solari's own opaque/glass/hair/
        // portal register the same way as any downstream material (see SolariPlugin).
        libraries.ensure_raygen()?;
        libraries.ensure_miss(custom_sky)?;
        libraries.ensure_shadow()?;
        libraries.ensure_hit_groups(hit_groups)?;
        let pipeline = libraries.link()?;

        // Explicit pipeline stack size (spec formula for recursion depth 2, no
        // intersection/callable stages). The driver's DEFAULT stack for a
        // pipeline linked from LIBRARIES is unreliable — an undersized stack
        // corrupts payload/local spills with no validation error — so query
        // the per-group stack sizes from the linked pipeline and set the size
        // dynamically at trace time.
        let group_stack = |group: u32, ty: vk::ShaderGroupShaderKHR| -> u64 {
            // SAFETY: pipeline live; `group` is within the linked group range.
            unsafe { rt.get_ray_tracing_shader_group_stack_size(pipeline, group, ty) }
        };
        let shadow_group = 2 + hit_groups.len() as u32;
        let raygen_stack = group_stack(0, vk::ShaderGroupShaderKHR::GENERAL);
        let miss_stack = group_stack(1, vk::ShaderGroupShaderKHR::GENERAL)
            .max(group_stack(shadow_group, vk::ShaderGroupShaderKHR::GENERAL));
        let mut chit_stack = 0u64;
        let mut any_hit_stack = 0u64;
        for (i, hg) in hit_groups.iter().enumerate() {
            let g = 2 + i as u32;
            chit_stack = chit_stack.max(group_stack(g, vk::ShaderGroupShaderKHR::CLOSEST_HIT));
            if hg.any_hit.is_some() {
                any_hit_stack =
                    any_hit_stack.max(group_stack(g, vk::ShaderGroupShaderKHR::ANY_HIT));
            }
        }
        // rayGen + depth×max(chit + anyHit, miss): slightly conservative vs the
        // spec's exact expression (folds the any-hit into every level).
        let stack_size = (raygen_stack + 2 * (chit_stack + any_hit_stack).max(miss_stack)) as u32;

        // Linked group numbering (see `RtLibraryCache::link`): raygen (0), primary
        // miss (1), one hit group per registry entry (its index = its SBT class;
        // class c -> group 2+c -> handle(2+c)), shadow miss LAST.
        let shadow_miss_group = 2 + hit_groups.len() as u32;
        let group_count = shadow_miss_group + 1;
        // The hair hit group's SBT handle index, for the reserved hair record below.
        let hair_group = hit_groups
            .iter()
            .position(|g| g.label == "hair")
            .map_or(2u32, |i| 2 + i as u32);
        // Max valid SBT class (registry index); out-of-range material classes fall back.
        let max_class = (hit_groups.len() as u32).saturating_sub(1);

        // The set-1 descriptor set + camera UBO + env binding are per-view, built
        // lazily in `create_view_bindings` (one per `SolariCamera`).

        // --- SBT: raygen + 2 miss + ONE HIT RECORD PER MATERIAL (+ hair) -------
        // Three regions, each base-aligned. The hit region holds one record per
        // material slot; an instance's `instance_contribution_to_hit_group_index`
        // = its material slot selects its record. Each HIT record is
        // [shader group handle | shader-record data]; the data slot holds the
        // material id (= record index), which `chit_opaque`'s `var<shader_record>`
        // reads as the canonical material binding (uniform per record → uniform
        // per warp after SER). Distinct per-material records also give SER a
        // per-material reorder key.
        const MISS_COUNT: u64 = 2; // miss index 0 = primary, 1 = shadow
        const HIT_RECORD_DATA: u64 = 4; // bytes of shader-record data (u32 material id)
        const RECORD_HEADROOM: u32 = 1024; // absorb streaming material growth post-build
        // One extra hit record, appended AFTER the per-material records, baked with
        // the hair hit-group handle (group 4). Hair instances route to it
        // (`hair_sbt_record`) via `ptlas_hair_write`; it's a single shared record
        // (chit_hair keys off the instance, not a per-record material id). Kept off
        // the material region so material routing/SER is untouched.
        const HAIR_RECORDS: u64 = 1;
        let record_capacity = material_count + RECORD_HEADROOM;
        let total_records = record_capacity as u64 + HAIR_RECORDS;
        let handle_stride = align_up(handle_size, handle_align);
        // Hit records carry the material-id data slot, so they're wider than a
        // bare handle.
        let hit_record_stride = align_up(handle_size + HIT_RECORD_DATA, handle_align);
        let raygen_offset = 0u64;
        let miss_offset = align_up(handle_stride, base_align);
        // The miss region holds MISS_COUNT contiguous handle-stride records.
        let hit_offset = align_up(miss_offset + MISS_COUNT * handle_stride, base_align);
        let sbt_size = hit_offset + total_records * hit_record_stride;
        let sbt = alloc_mapped_buffer(
            allocator,
            sbt_size,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
        )?;

        // SAFETY: pipeline live; handle data sized to group_count * handle_size.
        let handles = unsafe {
            rt.get_ray_tracing_shader_group_handles(
                pipeline,
                0,
                group_count,
                (group_count as u64 * handle_size) as usize,
            )
        }
        .ok()?;
        let handle = |g: usize| &handles[g * handle_size as usize..(g + 1) * handle_size as usize];
        // raygen (group 0), primary miss (group 1 → miss index 0), shadow miss
        // (last group → miss index 1).
        for &(g, off) in [
            (0usize, raygen_offset),
            (1usize, miss_offset),
            (shadow_miss_group as usize, miss_offset + handle_stride),
        ]
        .iter()
        {
            // SAFETY: mapped covers sbt_size; off + handle_size within bounds.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    handle(g).as_ptr(),
                    sbt.mapped.add(off as usize),
                    handle_size as usize,
                );
            }
        }
        // One record per material slot: [class hit-group handle | material id].
        // The record's handle is `handle(2 + class)` — opaque (group 2), glass
        // (group 3), or portal (group 5, class 3) — so an instance routing to its
        // material's record lands on the closest-hit its CLASS selects, not always
        // the opaque one. (Hair is class 2 → group 4, reached via the reserved
        // record below, not a material.) Headroom records past the live material
        // count, and any out-of-range class, fall back to opaque. The data slot =
        // the record index = the material id, read back via `var<shader_record>`
        // (uniform per record → uniform per warp after SER).
        for record in 0..total_records {
            let rec_off = hit_offset + record * hit_record_stride;
            // The appended hair record (index == record_capacity) always routes to
            // the hair closest-hit (group 4); every other record picks its material
            // class handle (headroom / out-of-range → opaque).
            let group_handle = if record == record_capacity as u64 {
                handle(hair_group as usize)
            } else {
                let class = material_classes
                    .get(record as usize)
                    .copied()
                    .unwrap_or(0)
                    .min(max_class);
                handle(2 + class as usize)
            };
            // SAFETY: rec_off + handle_size + 4 within the record.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    group_handle.as_ptr(),
                    sbt.mapped.add(rec_off as usize),
                    handle_size as usize,
                );
                core::ptr::copy_nonoverlapping(
                    (record as u32).to_le_bytes().as_ptr(),
                    sbt.mapped.add((rec_off + handle_size) as usize),
                    4,
                );
            }
        }

        let raygen_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt.device_address + raygen_offset)
            .stride(handle_stride)
            .size(handle_stride);
        let miss_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt.device_address + miss_offset)
            .stride(handle_stride)
            .size(MISS_COUNT * handle_stride);
        // Per-material records + the appended hair record; stride steps one record.
        // An instance's material slot indexes the material records; hair indexes the
        // last one (`hair_sbt_record`).
        let hit_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt.device_address + hit_offset)
            .stride(hit_record_stride)
            .size(total_records * hit_record_stride);
        let callable_region = vk::StridedDeviceAddressRegionKHR::default();

        let out = Self {
            device,
            rt,
            pipeline,
            pipeline_layout: libraries.pipeline_layout,
            descriptor_set_layout: libraries.descriptor_set_layout,
            sbt,
            raygen_region,
            miss_region,
            hit_region,
            callable_region,
            record_capacity,
            material_classes: material_classes.to_vec(),
            custom_sky_generation,
            stack_size,
            _device_keepalive: allocator.clone(),
        };
        Some(out)
    }

    /// Generation of the `bevy_solari::custom_sky` module the primary miss shader
    /// was composed with; the dispatch rebuilds the pipeline when the live
    /// [`SolariCustomSky`](crate::render::sky::SolariCustomSky) generation differs.
    pub fn custom_sky_generation(&self) -> u64 {
        self.custom_sky_generation
    }

    /// Per-material hit-record count the SBT was built for. The dispatch rebuilds
    /// the pipeline once the live material count exceeds this (an instance routes
    /// to `record = material slot`, so a slot past the end would read OOB).
    pub fn capacity(&self) -> u32 {
        self.record_capacity
    }

    /// SBT hit-record index reserved for hair (the single record appended after the
    /// per-material records, baked with the hair closest-hit handle). Hair PTLAS
    /// instances set this as their `instance_contribution_to_hit_group_index` so
    /// they reach `chit_hair` instead of mis-indexing a material record. Stable for
    /// the pipeline's lifetime; it only changes on a capacity rebuild (which also
    /// skips that frame's trace, so a one-frame-stale value is never consumed).
    pub fn hair_sbt_record(&self) -> u32 {
        self.record_capacity
    }

    /// Whether this frame's per-material SBT classes differ from those the hit
    /// records were baked with — a glass material loading/unloading or a live
    /// class edit. The class selects a record's shader handle, which is baked at
    /// build time (unlike the per-frame GPU `FORCE_NO_OPAQUE` flag), so a change
    /// needs a rebuild. The dispatch checks this alongside [`Self::capacity`] and
    /// drains + recreates the pipeline when it returns `true`.
    pub fn classes_changed(&self, current: &[u32]) -> bool {
        let n = self.material_classes.len();
        if current.len() < n {
            return self.material_classes.as_slice() != current;
        }
        // Baked prefix must match exactly; records past it were baked with the
        // opaque (class 0) handle + their record index, so a NEW class-0
        // material inside the headroom is already routed correctly — only a
        // non-opaque class arriving there forces a rebuild.
        self.material_classes[..] != current[..n] || current[n..].iter().any(|&c| c != 0)
    }

    /// Build the per-view set-1 resources (descriptor set) for one view: a fresh pool,
    /// a set allocated from the shared set-1 layout, and the descriptor written ONCE
    /// (output buffer @0, camera @1, env cube @2, shared sampler @3). The set is never
    /// updated again — updating one while a prior frame's command buffer still binds it
    /// is illegal and device-losts; the camera *contents* change per frame in the bound
    /// GPU buffer (the `rt_camera` compute pass writes it), and both the output and
    /// camera buffers are stable (the dispatch rebuilds this whole component if the
    /// view's output buffer is reallocated). `env_map_image` is `Some` when the env cube
    /// is the storage atmosphere cube (transitioned around the trace).
    pub fn create_view_bindings(
        &self,
        allocator: &Allocator,
        output_buffer: vk::Buffer,
        output_size: u64,
        // The per-view `RtCamera` `VkBuffer` (wgpu-owned) baked into binding 1.
        camera_buffer: vk::Buffer,
        // DLSS guide G-buffers `(VkBuffer, size)` bound at BINDING_GBUFFER_* (set 1).
        // Empty unless the `dlss` feature is on; its length sizes the storage-buffer
        // pool slot, so it must agree with the layout the pipeline was built with.
        gbuffers: &[(vk::Buffer, u64)],
        // ReSTIR reservoir buffer `(VkBuffer, size)` bound at BINDING_RESERVOIRS.
        reservoirs: (vk::Buffer, u64),
        // ReSTIR surface G-buffer `(VkBuffer, size)` bound at BINDING_SURFACE.
        surface: (vk::Buffer, u64),
        // ReSTIR winner light samples `(VkBuffer, size)` bound at BINDING_LIGHT_SAMPLES.
        light_samples: (vk::Buffer, u64),
        // ReSTIR GI canonical samples `(VkBuffer, size)` bound at BINDING_GI_SAMPLES.
        gi_samples: (vk::Buffer, u64),
        // NRC inference weights / biases / training records / screen (BINDING_NRC_*).
        nrc_weights: (vk::Buffer, u64),
        nrc_bias: (vk::Buffer, u64),
        nrc_records: (vk::Buffer, u64),
        nrc_queries: (vk::Buffer, u64),
        env_map_view: vk::ImageView,
        env_map_image: Option<vk::Image>,
        // When the binding seam exists, every set-1 resource also gets a
        // descriptor-heap slot (M2 staging — see [`RtViewHeapSlots`]).
        seam: Option<&crate::gpu::binding_seam::BindingSeam>,
    ) -> Option<RtViewBindings> {
        // One pool per view, sized for exactly this view's single set-1 set.
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(9 + gbuffers.len() as u32), // + 4×NRC (weights/bias/records/queries)
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(1), // camera (ringed)
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1), // geometry addresses
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(1),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        // SAFETY: well-formed; device live.
        let descriptor_pool =
            unsafe { self.device.create_descriptor_pool(&pool_info, None) }.ok()?;
        let own_set_layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&own_set_layouts);
        // SAFETY: pool + layout live.
        let descriptor_set = match unsafe { self.device.allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => sets.into_iter().next()?,
            Err(e) => {
                bevy_log::error!("rt_pipeline: allocate_descriptor_sets failed: {e:?}");
                // SAFETY: pool just created, no sets in use.
                unsafe { self.device.destroy_descriptor_pool(descriptor_pool, None) };
                return None;
            }
        };

        let geometry = alloc_mapped_buffer(
            allocator,
            size_of::<RtGeometryAddresses>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        // This view's own linear env-cube sampler (we own it rather than reaching
        // into wgpu's Sampler, which has no raw accessor). Clamp-to-edge is fine
        // for a cube.
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(vk::LOD_CLAMP_NONE);
        // SAFETY: well-formed; device live.
        let env_map_sampler = unsafe { self.device.create_sampler(&sampler_info, None) }.ok()?;

        let output_info = [vk::DescriptorBufferInfo::default()
            .buffer(output_buffer)
            .offset(0)
            .range(output_size)];
        // Dynamic uniform bound at a constant offset 0 (`trace` passes 0) — the
        // GPU-written `RtCamera` buffer, filled by the `rt_camera` compute pass (or
        // a CPU `write_buffer`) before the trace reads it.
        let camera_info = [vk::DescriptorBufferInfo::default()
            .buffer(camera_buffer)
            .offset(0)
            .range(size_of::<RtCamera>() as u64)];
        // `trace()` transitions the atmosphere cube to SHADER_READ_ONLY_OPTIMAL
        // around the dispatch (skybox/fallback are already in this layout), so the
        // descriptor always sees read-optimal.
        let env_image_info = [vk::DescriptorImageInfo::default()
            .image_view(env_map_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let env_sampler_info =
            [vk::DescriptorImageInfo::default().sampler(env_map_sampler)];
        let geometry_info = [vk::DescriptorBufferInfo::default()
            .buffer(geometry.buffer)
            .offset(0)
            .range(geometry.size)];
        let reservoirs_info = [vk::DescriptorBufferInfo::default()
            .buffer(reservoirs.0)
            .offset(0)
            .range(reservoirs.1)];
        let surface_info = [vk::DescriptorBufferInfo::default()
            .buffer(surface.0)
            .offset(0)
            .range(surface.1)];
        let light_samples_info = [vk::DescriptorBufferInfo::default()
            .buffer(light_samples.0)
            .offset(0)
            .range(light_samples.1)];
        let gi_samples_info = [vk::DescriptorBufferInfo::default()
            .buffer(gi_samples.0)
            .offset(0)
            .range(gi_samples.1)];
        let nrc_weights_info = [vk::DescriptorBufferInfo::default()
            .buffer(nrc_weights.0)
            .offset(0)
            .range(nrc_weights.1)];
        let nrc_bias_info = [vk::DescriptorBufferInfo::default()
            .buffer(nrc_bias.0)
            .offset(0)
            .range(nrc_bias.1)];
        let nrc_records_info = [vk::DescriptorBufferInfo::default()
            .buffer(nrc_records.0)
            .offset(0)
            .range(nrc_records.1)];
        let nrc_queries_info = [vk::DescriptorBufferInfo::default()
            .buffer(nrc_queries.0)
            .offset(0)
            .range(nrc_queries.1)];
        // DLSS guide descriptors built outside `writes` so the per-binding infos
        // outlive `update_descriptor_sets` (empty when the feature is off).
        let gbuffer_infos: Vec<[vk::DescriptorBufferInfo; 1]> = gbuffers
            .iter()
            .map(|&(buf, size)| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(buf)
                    .offset(0)
                    .range(size)]
            })
            .collect();
        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_OUTPUT)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&output_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_CAMERA)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&camera_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_ENV_MAP)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&env_image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_ENV_SAMPLER)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&env_sampler_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_GEOMETRY)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&geometry_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_RESERVOIRS)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&reservoirs_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_SURFACE)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&surface_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_LIGHT_SAMPLES)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&light_samples_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_GI_SAMPLES)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&gi_samples_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_NRC_WEIGHTS)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&nrc_weights_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_NRC_BIAS)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&nrc_bias_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_NRC_RECORDS)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&nrc_records_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_NRC_QUERIES)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&nrc_queries_info),
        ];
        {
            let gbuffer_bindings = [
                BINDING_GBUFFER_NORMAL,
                BINDING_GBUFFER_DIFFUSE,
                BINDING_GBUFFER_SPECULAR,
                BINDING_GBUFFER_MOTION,
            ];
            for (i, info) in gbuffer_infos.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(gbuffer_bindings[i])
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info),
                );
            }
        }
        // SAFETY: targets the freshly-allocated set; buffers + image/sampler live.
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        // M2 staging: mirror every set-1 resource into the descriptor heap.
        // Uniform vs storage matches the set-1 layout (camera + geometry are
        // uniforms); slot order mirrors the binding list so the mapping table
        // at the heap-pipeline flip reads straight off this Vec.
        let heap = seam.map(|seam| {
            use crate::gpu::binding_seam::HeapResource;
            let storage = |buffer: vk::Buffer, size: u64| HeapResource::Buffer {
                address: seam.raw_buffer_address(buffer),
                size,
            };
            let mut resources = vec![
                storage(output_buffer, output_size),
                HeapResource::UniformBuffer {
                    address: seam.raw_buffer_address(camera_buffer),
                    size: size_of::<RtCamera>() as u64,
                },
                HeapResource::UniformBuffer {
                    address: geometry.device_address,
                    size: geometry.size,
                },
            ];
            resources.extend(gbuffers.iter().map(|&(buf, size)| storage(buf, size)));
            resources.extend(
                [
                    reservoirs,
                    surface,
                    light_samples,
                    gi_samples,
                    nrc_weights,
                    nrc_bias,
                    nrc_records,
                    nrc_queries,
                ]
                .into_iter()
                .map(|(buf, size)| storage(buf, size)),
            );
            let buffers = resources
                .into_iter()
                .map(|r| seam.alloc_heap_index(r))
                .collect();
            let env_sampler_slot =
                seam.alloc_heap_index(HeapResource::Sampler(&sampler_info));
            RtViewHeapSlots {
                seam: seam.clone(),
                buffers,
                env_sampler: env_sampler_slot,
            }
        });

        Some(RtViewBindings {
            device: self.device.clone(),
            descriptor_pool,
            descriptor_set,
            geometry,
            env_map_sampler,
            env_map_image,
            output_buffer,
            env_map_view,
            heap,
            _device_keepalive: allocator.clone(),
        })
    }

    /// Record bind + `cmd_trace_rays` into `command_buffer` for a `width`×`height`
    /// dispatch, writing this view's per-pixel output storage buffer. `scene_set` /
    /// `columns_set` are the raw `VkDescriptorSet`s of wgpu's scene + columns bind
    /// groups (sets 0 and 2), from `BindGroup::raw_descriptor_set`; `view` carries
    /// the per-view set 1 + env image to transition.
    ///
    /// # Safety
    /// `command_buffer` must be recording; the scene/columns sets must be valid
    /// and match the layouts the pipeline was built with; `view` must have been
    /// built by `self.create_view_bindings`.
    pub unsafe fn trace(
        &self,
        command_buffer: vk::CommandBuffer,
        scene_set: vk::DescriptorSet,
        view: &RtViewBindings,
        columns_set: vk::DescriptorSet,
        // Dynamic offset for the camera buffer (binding 1). Always 0 — the buffer holds
        // exactly this frame's `RtCamera`; the binding stays dynamic only to reuse the
        // set-1 layout unchanged.
        camera_dynamic_offset: u32,
        width: u32,
        height: u32,
    ) {
        unsafe {
            // Pre-trace hazard barrier. wgpu records the PTLAS/BLAS build and the
            // scene-column scatter passes into earlier command buffers but has NO
            // idea this raw trace reads the TLAS + those storage buffers, so it
            // inserts none of the dependencies it would for its own megakernel
            // compute pass. Without this, traversal can read the acceleration
            // structure while it's still building (or read stale columns) → GPU
            // page fault / device loss that CPU-side validation never catches.
            //   AS build write  -> RT-shader AS read
            //   compute storage write -> RT-shader storage read
            //   transfer write -> RT-shader read (the tess smooth-normal metadata
            //   table is uploaded via a staging copy before the trace reads it)
            let pre = vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::TRANSFER_WRITE,
                )
                .dst_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                        | vk::AccessFlags::SHADER_READ,
                );
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                    | vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[pre],
                &[],
                &[],
            );

            // The atmosphere env cube is a STORAGE image wgpu leaves in GENERAL;
            // transition it to SHADER_READ_ONLY_OPTIMAL for the miss's sampled read
            // (restored to GENERAL after the trace so wgpu's layout tracking stays
            // consistent). Cube = 6 array layers.
            let env_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            };
            if let Some(env_image) = view.env_map_image {
                let to_read = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(env_image)
                    .subresource_range(env_range);
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_read],
                );
            }

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline,
            );
            // The linked pipeline opts into the dynamic stack-size state (see
            // `RtLibraryCache::link`); the queried-per-group size replaces the
            // driver default, which is unreliable across library boundaries.
            self.rt
                .cmd_set_ray_tracing_pipeline_stack_size(command_buffer, self.stack_size);
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout,
                0,
                &[scene_set, view.descriptor_set, columns_set],
                // One dynamic offset, for set 1's camera (binding 1) — the only
                // dynamic descriptor across the three bound sets (scene/columns are
                // wgpu-built with none, hence the prior empty slice).
                &[camera_dynamic_offset],
            );
            self.rt.cmd_trace_rays(
                command_buffer,
                &self.raygen_region,
                &self.miss_region,
                &self.hit_region,
                &self.callable_region,
                width,
                height,
                1,
            );
            // Make the per-pixel output-buffer writes available to the wgpu
            // compute blit that reads them next. wgpu doesn't see this raw write,
            // so the dependency is ours to insert (a global memory barrier — the
            // buffer has no layout to transition).
            let mem = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[mem],
                &[],
                &[],
            );

            // Restore the env cube to GENERAL so wgpu's tracked layout stays valid.
            if let Some(env_image) = view.env_map_image {
                let to_general = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(env_image)
                    .subresource_range(env_range);
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
            }
        }
    }

}

impl RtViewBindings {
    /// The output `VkBuffer` baked into binding 0. The dispatch compares this to
    /// the view's current output buffer to detect a resize-driven reallocation.
    pub fn output_buffer(&self) -> vk::Buffer {
        self.output_buffer
    }

    /// The env cube view baked into binding 2. The dispatch compares this to the
    /// view's current environment view to detect a loaded/swapped skybox.
    pub fn env_map_view(&self) -> vk::ImageView {
        self.env_map_view
    }

    /// Upload this frame's bindless geometry addresses (host-visible, coherent).
    pub fn set_geometry_addresses(&self, addresses: &RtGeometryAddresses) {
        // SAFETY: `geometry.mapped` is a valid HOST_VISIBLE|COHERENT mapping of at
        // least size_of::<RtGeometryAddresses>() bytes; the struct is Pod.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytemuck::bytes_of(addresses).as_ptr(),
                self.geometry.mapped,
                size_of::<RtGeometryAddresses>(),
            );
        }
    }
}

impl Drop for RtViewBindings {
    fn drop(&mut self) {
        // In-flight traces may still reference the set/pool; drain first (near-
        // free when the rebuild path already drained).
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: the pool + camera buffer were created for this view; the queue
        // is drained and the device alive (keepalive). Destroying the pool frees
        // its descriptor set.
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_sampler(self.env_map_sampler, None);
            // `camera_buffer` is wgpu-owned (the per-view `RtOutputBuffer`), freed there.
            self.device.destroy_buffer(self.geometry.buffer, None);
            self.device.free_memory(self.geometry.memory, None);
        }
    }
}

impl Drop for RtPipeline {
    fn drop(&mut self) {
        // In-flight traces may still reference the pipeline/SBT; drain first.
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: the linked pipeline + SBT were created by this resource; the
        // queue is drained and the device alive (keepalive). The layouts and
        // stage libraries belong to `RtLibraryCache` and survive the rebuild.
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_buffer(self.sbt.buffer, None);
            self.device.free_memory(self.sbt.memory, None);
        }
    }
}

/// The built-in Slang modules importable by runtime-compiled RT stages
/// (`import scene_resolve;` etc.) — the same set the precompiled stages resolve
/// as files beside them at regen time.
const RT_SLANG_MODULES: &[(&str, &str)] = &[
    (
        "rt_payload",
        include_str!("../render/rt_pipeline/rt_payload.slang"),
    ),
    (
        "scene_resolve",
        include_str!("../render/rt_pipeline/scene_resolve.slang"),
    ),
    ("brdf", include_str!("../render/rt_pipeline/brdf.slang")),
    (
        "sampling",
        include_str!("../render/rt_pipeline/sampling.slang"),
    ),
    ("hair", include_str!("../render/rt_pipeline/hair.slang")),
];

/// Compile a [`SolariChitSource::Slang`] closest-hit with the built-in module
/// set plus the group's own `composable_modules` importable. Shared by eager
/// registration validation and the pipeline build.
fn compile_slang_chit(group: &SolariHitGroupDef) -> Result<Vec<u32>, String> {
    let SolariChitSource::Slang { source, file, entry } = &group.closest_hit else {
        unreachable!("compile_slang_chit called on a precompiled closest-hit");
    };
    let mut modules = RT_SLANG_MODULES.to_vec();
    modules.extend_from_slice(group.composable_modules);
    crate::gpu::slang::compile_rt_slang(
        file,
        source,
        entry,
        crate::gpu::slang::SlangRtStage::ClosestHit,
        &modules,
    )
}

/// naga capabilities the WGSL RT-adjacent shaders need — used only by the
/// headless `rt_shaders_compile` validation of the remaining stock-naga WGSL
/// (`restir_spatial`, which traces inline ray queries); every runtime RT stage
/// is Slang.
#[cfg(test)]
fn rt_capabilities() -> naga::valid::Capabilities {
    // Ray queries for the inline visibility traces; f16-in-f32 for the
    // `pack2x16float`/`unpack2x16float` G-buffer codecs. The runtime compile
    // gets both from the device features via the wgpu pipeline cache.
    naga::valid::Capabilities::RAY_QUERY
        | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
}

/// Build a naga_oil composer pre-loaded with the built-in importable modules the
/// remaining WGSL passes may `#import` — test-only: it backs the headless
/// `rt_shaders_compile` check of `restir_spatial` (whose runtime compile goes
/// through the wgpu `PipelineCache`).
#[cfg(test)]
fn rt_composer() -> Option<naga_oil::compose::Composer> {
    use naga_oil::compose::{ComposableModuleDescriptor, Composer};

    // Compose via naga_oil so the WGSL passes can `#import` solari's
    // scene-binding / BRDF / sampling modules (raw `naga::parse_str` can't
    // resolve `#import`). Composer validates the composed module, so it needs
    // the same capabilities as the spv backend (ray query, binding arrays).
    // `with_capabilities` purges modules, so set it first.
    let mut composer = Composer::default().with_capabilities(rt_capabilities());
    macro_rules! register {
        ($path:expr, $source:expr) => {
            if let Err(e) = composer.add_composable_module(ComposableModuleDescriptor {
                source: $source,
                file_path: $path,
                ..Default::default()
            }) {
                bevy_log::error!("rt_pipeline: compose register {}: {e:?}", $path);
                return None;
            }
        };
        ($path:literal) => {
            register!($path, include_str!($path))
        };
    }
    // Registered leaf-first: naga_oil resolves each module's `#import`s at
    // add-time, so every dependency must already be registered. This is
    // `restir_spatial`'s import closure.
    register!("../../../bevy_render/src/maths.wgsl"); // bevy_render::maths (leaf)
    register!("../../../bevy_render/src/utils.wgsl"); // bevy_render::utils (leaf)
    register!("../render/rt_pipeline/rt_payload.wgsl"); // bevy_solari::rt_payload (leaf)
    register!("../bindings/pbr.wgsl"); // -> maths
    register!("../bindings/raytracing_scene_bindings.wgsl"); // (leaf)
    register!("../bindings/sampling.wgsl"); // -> pbr, scene_bindings, maths
    register!("../bindings/brdf.wgsl"); // -> pbr, sampling, scene_bindings, maths
    Some(composer)
}

/// Test-only compose→validate→SPIR-V of a WGSL shader (see [`rt_composer`]) —
/// the headless shader test asserts on the failure value, so shader edits fail
/// at `cargo test` with the real error.
#[cfg(test)]
fn try_compile_rt_wgsl(source: &str, file_path: &str) -> Result<Vec<u32>, String> {
    use naga_oil::compose::{NagaModuleDescriptor, ShaderDefValue};

    let mut composer =
        rt_composer().ok_or_else(|| "composable module registration failed".to_string())?;

    // Shader-def axes for the RT shaders. This is the "pipeline key": each def is a
    // compile-out feature axis the raygen/chits can `#ifdef` on. Keep the axes few
    // and orthogonal (debug views ride a runtime uniform, not a def, to avoid a
    // variant explosion). `SOLARI_DLSS` is compile-time (tied to the cargo feature):
    // when set, the trace emits the ray-reconstruction guide G-buffer.
    #[allow(unused_mut)]
    let mut shader_defs: std::collections::HashMap<String, ShaderDefValue> = [(
        // The scene-columns bind-group index the scene bindings are written with.
        "SOLARI_SCENE_COLUMNS_GROUP".to_string(),
        ShaderDefValue::UInt(2),
    )]
    .into_iter()
    .collect();
    shader_defs.insert("SOLARI_DLSS".to_string(), ShaderDefValue::Bool(true));
    // Compile in the `shader_clock()` reads only when the device enabled
    // `VK_KHR_shader_clock`; otherwise the cost-heatmap path compiles out.
    if crate::gpu::extension::shader_clock_available() {
        shader_defs.insert("SOLARI_SHADER_CLOCK".to_string(), ShaderDefValue::Bool(true));
    }

    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source,
            file_path,
            shader_defs,
            ..Default::default()
        })
        .map_err(|e| format!("compose: {e:?}"))?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), rt_capabilities())
        .validate(&module)
        .map_err(|e| format!("WGSL validation failed: {e:?}"))?;
    let mut options = naga::back::spv::Options::default();
    options.lang_version = (1, 4);
    // The scene `textures`/`samplers` are unsized `binding_array`s in WGSL. wgpu's
    // own pipeline compile bakes a FIXED descriptor count into the SPIR-V (the
    // device doesn't enable `runtimeDescriptorArray`, so an `OpTypeRuntimeArray`
    // descriptor variable is invalid). Mirror that for every unsized binding
    // array the composed module actually contains — derived from the module, not
    // hardcoded group/binding numbers, so a scene-binding renumber can't silently
    // reintroduce the illegal runtime-array variable. The substituted count must
    // match the descriptor set layout, which sizes all of them `MAX_TEXTURE_COUNT`.
    options.fake_missing_bindings = true;
    for (_, var) in module.global_variables.iter() {
        let Some(ref binding) = var.binding else {
            continue;
        };
        if let naga::TypeInner::BindingArray {
            size: naga::ArraySize::Dynamic,
            ..
        } = module.types[var.ty].inner
        {
            options.binding_map.insert(
                binding.clone(),
                naga::back::spv::BindingInfo {
                    descriptor_set: binding.group,
                    binding: binding.binding,
                    binding_array_size: Some(crate::bindings::MAX_TEXTURE_COUNT.get()),
                },
            );
        }
    }
    naga::back::spv::write_vec(&module, &info, &options, None)
        .map_err(|e| format!("SPIR-V emit failed: {e:?}"))
}

/// Byte-align an embedded Slang-precompiled SPIR-V blob into words
/// (`include_bytes!` carries no u32 alignment guarantee).
fn spirv_words(bytes: &'static [u8]) -> Vec<u32> {
    debug_assert_eq!(bytes.len() % 4, 0, "SPIR-V blob length not word-aligned");
    bytemuck::pod_collect_to_vec(bytes)
}

fn create_shader_module(device: &ash::Device, spv: &[u32]) -> Option<vk::ShaderModule> {
    let info = vk::ShaderModuleCreateInfo::default().code(spv);
    // SAFETY: spv is valid SPIR-V words from naga; device live.
    match unsafe { device.create_shader_module(&info, None) } {
        Ok(m) => Some(m),
        Err(e) => {
            bevy_log::error!("rt_pipeline: create_shader_module failed: {e:?}");
            None
        }
    }
}

fn shader_stage(
    stage: vk::ShaderStageFlags,
    module: vk::ShaderModule,
    name: &CStr,
) -> vk::PipelineShaderStageCreateInfo<'_> {
    vk::PipelineShaderStageCreateInfo::default()
        .stage(stage)
        .module(module)
        .name(name)
}

fn general_group(shader: u32) -> vk::RayTracingShaderGroupCreateInfoKHR<'static> {
    vk::RayTracingShaderGroupCreateInfoKHR::default()
        .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
        .general_shader(shader)
        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
        .any_hit_shader(vk::SHADER_UNUSED_KHR)
        .intersection_shader(vk::SHADER_UNUSED_KHR)
}

/// A triangle hit group whose closest-hit is stage index `shader`.
fn hit_group(shader: u32) -> vk::RayTracingShaderGroupCreateInfoKHR<'static> {
    vk::RayTracingShaderGroupCreateInfoKHR::default()
        .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
        .general_shader(vk::SHADER_UNUSED_KHR)
        .closest_hit_shader(shader)
        .any_hit_shader(vk::SHADER_UNUSED_KHR)
        .intersection_shader(vk::SHADER_UNUSED_KHR)
}

/// A triangle hit group with closest-hit stage `closest` + any-hit stage `any_hit`.
/// The any-hit (alpha cutout) fires only for non-opaque geometry — i.e. instances
/// PTLAS marked `FORCE_NO_OPAQUE` (alpha-masked materials). Opaque geometry commits
/// in hardware and never invokes it, so opaque materials pay no alpha-test cost.
fn hit_group_with_any_hit(
    closest: u32,
    any_hit: u32,
) -> vk::RayTracingShaderGroupCreateInfoKHR<'static> {
    vk::RayTracingShaderGroupCreateInfoKHR::default()
        .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
        .general_shader(vk::SHADER_UNUSED_KHR)
        .closest_hit_shader(closest)
        .any_hit_shader(any_hit)
        .intersection_shader(vk::SHADER_UNUSED_KHR)
}

#[inline]
fn align_up(value: u64, align: u64) -> u64 {
    (value + align - 1) & !(align - 1)
}

/// Allocate a HOST_VISIBLE|COHERENT, SHADER_DEVICE_ADDRESS raw buffer, map it
/// persistently, and return its handles + device address. Used for the SBT and
/// camera UBO, which need a kept mapping.
fn alloc_mapped_buffer(
    allocator: &Allocator,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Option<MappedBuffer> {
    let device = allocator.device();
    let instance = allocator.instance();
    let physical_device = allocator.physical_device();
    let size = size.max(4);

    let create_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: well-formed; device live.
    let buffer = unsafe { device.create_buffer(&create_info, None) }.ok()?;
    // SAFETY: buffer valid.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    // SAFETY: physical_device valid.
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let memory_type_index = (0..mem_props.memory_type_count).find(|&i| {
        let supported = requirements.memory_type_bits & (1 << i) != 0;
        let props = mem_props.memory_types[i as usize].property_flags;
        supported && props.contains(wanted)
    })?;

    let mut flags_info =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index)
        .push(&mut flags_info);
    // SAFETY: well-formed; flags_info valid extension struct.
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }.ok()?;
    // SAFETY: buffer + memory valid; offset 0 for dedicated allocation.
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.ok()?;
    // SAFETY: memory is HOST_VISIBLE; mapping the whole range.
    let mapped = unsafe {
        device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
    }
    .ok()? as *mut u8;
    // SAFETY: buffer created with SHADER_DEVICE_ADDRESS.
    let device_address = unsafe {
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
    };

    Some(MappedBuffer {
        buffer,
        memory,
        mapped,
        size,
        device_address,
    })
}

#[cfg(test)]
mod tests {
    use super::try_compile_rt_wgsl;

    /// Rejects what VUID-StandaloneSpirv-OpTypeRuntimeArray-04680 rejects: a
    /// descriptor variable instantiating `OpTypeRuntimeArray` (a `UniformConstant`
    /// pointer to a runtime array). Happens when an unsized `binding_array` misses
    /// the fixed-size substitution in the SPIR-V binding map.
    /// `restir_spatial` is the one shader here that is NOT built through the raw
    /// RT path — wgpu's `PipelineCache` compiles it as an ordinary compute
    /// pipeline, without `rt_capabilities`. Reaching any bindless helper that does
    /// a `physical_load` therefore builds fine in this test but fails at runtime
    /// with "using physical_load requires ... PhysicalStorageBufferAddresses".
    fn assert_no_physical_storage_buffer(file: &str, spv: &[u32]) {
        const OP_CAPABILITY: u32 = 17;
        const CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES: u32 = 5347;
        let mut i = 5; // skip the SPIR-V header
        while i < spv.len() {
            let (opcode, word_count) = (spv[i] & 0xffff, (spv[i] >> 16) as usize);
            assert!(word_count > 0, "{file}: malformed SPIR-V");
            // Capabilities are all declared up front; stop at the first non-capability.
            if opcode != OP_CAPABILITY {
                break;
            }
            assert!(
                spv[i + 1] != CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES,
                "{file}: reaches a `physical_load` buffer-device-address helper \
                 (e.g. `load_material_bindless` / `alpha_test` / `resolve_ray_hit_full`), \
                 but it is built as a plain wgpu compute pipeline which has no \
                 PhysicalStorageBufferAddresses capability"
            );
            i += word_count;
        }
    }

    fn assert_no_runtime_descriptor_array(file: &str, spv: &[u32]) {
        const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
        const OP_TYPE_POINTER: u32 = 32;
        const STORAGE_UNIFORM_CONSTANT: u32 = 0;
        let mut runtime_arrays = std::collections::HashSet::new();
        let mut i = 5; // skip the SPIR-V header
        while i < spv.len() {
            let (opcode, word_count) = (spv[i] & 0xffff, (spv[i] >> 16) as usize);
            assert!(word_count > 0, "{file}: malformed SPIR-V");
            match opcode {
                OP_TYPE_RUNTIME_ARRAY => {
                    runtime_arrays.insert(spv[i + 1]);
                }
                OP_TYPE_POINTER => {
                    assert!(
                        !(spv[i + 2] == STORAGE_UNIFORM_CONSTANT
                            && runtime_arrays.contains(&spv[i + 3])),
                        "{file}: UniformConstant pointer to OpTypeRuntimeArray — an \
                         unsized binding_array escaped the fixed-size binding_map \
                         substitution (VUID-StandaloneSpirv-OpTypeRuntimeArray-04680)"
                    );
                }
                _ => {}
            }
            i += word_count;
        }
    }

    // Headless compose→validate→SPIR-V of every built-in RT shader — shader edits
    // fail here at `cargo test` time instead of as a runtime pipeline-build black
    // screen.
    #[test]
    fn rt_shaders_compile() {
        for (file, source) in [
            // The wgpu spatial pass — composed via PipelineCache at runtime, but
            // its imports are all registered here too, so validate it headlessly.
            (
                "restir_spatial.wgsl",
                include_str!("../render/rt_pipeline/restir_spatial.wgsl"),
            ),
        ] {
            match try_compile_rt_wgsl(source, file) {
                Ok(spv) => {
                    assert_no_runtime_descriptor_array(file, &spv);
                    if file == "restir_spatial.wgsl" {
                        assert_no_physical_storage_buffer(file, &spv);
                    }
                }
                Err(e) => panic!("{file}: {e}"),
            }
        }

        // The primary miss composes the swappable `custom_sky` module through
        // libslang at pipeline build (the one runtime-compiled stage). Compile
        // it here with the default procedural sky AND a user-style replacement
        // — the `SolariSky::Shader` path — so both stay proven headlessly.
        // (Needs the pinned Slang toolchain — see `gpu/slang.rs`.)
        for custom_sky in [
            crate::render::sky::DEFAULT_CUSTOM_SKY,
            "module custom_sky;\n\
             public float3 sample_custom_sky(float3 ray_direction)\n\
             { return float3(0.5, 0.6, 1.0) * 4000.0; }\n",
        ] {
            let spv = crate::gpu::slang::compile_rt_slang(
                "miss.slang",
                include_str!("../render/rt_pipeline/miss.slang"),
                "miss_primary",
                crate::gpu::slang::SlangRtStage::Miss,
                &[
                    (
                        "rt_payload",
                        include_str!("../render/rt_pipeline/rt_payload.slang"),
                    ),
                    ("custom_sky", custom_sky),
                ],
            )
            .unwrap_or_else(|e| panic!("miss.slang: {e}"));
            assert_no_runtime_descriptor_array("miss.slang", &spv);
        }

        // Slang-precompiled stages: sanity-check the embedded blobs (magic +
        // word alignment). The binding/layout cross-checks against the WGSL
        // stages (Cluster stride 48, Material field offsets, payload words)
        // happen at regen time — see the .slang headers.
        for (file, blob) in [
            (
                "miss_shadow.spv",
                include_bytes!("../render/rt_pipeline/miss_shadow.spv").as_slice(),
            ),
            (
                "ahit_alpha.spv",
                include_bytes!("../render/rt_pipeline/ahit_alpha.spv").as_slice(),
            ),
            (
                "chit_portal.spv",
                include_bytes!("../render/rt_pipeline/chit_portal.spv").as_slice(),
            ),
            (
                "chit_glass.spv",
                include_bytes!("../render/rt_pipeline/chit_glass.spv").as_slice(),
            ),
            (
                "chit_hair.spv",
                include_bytes!("../render/rt_pipeline/chit_hair.spv").as_slice(),
            ),
            (
                "chit_opaque.spv",
                include_bytes!("../render/rt_pipeline/chit_opaque.spv").as_slice(),
            ),
            (
                "raygen.spv",
                include_bytes!("../render/rt_pipeline/raygen.spv").as_slice(),
            ),
            (
                "raygen_clock.spv",
                include_bytes!("../render/rt_pipeline/raygen_clock.spv").as_slice(),
            ),
        ] {
            assert!(blob.len() % 4 == 0 && blob.len() > 20, "{file}: truncated");
            let magic = u32::from_le_bytes(blob[0..4].try_into().unwrap());
            assert_eq!(magic, 0x0723_0203, "{file}: not SPIR-V");
        }
    }
}
