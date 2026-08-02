// Raw Vulkan ray-tracing PIPELINE (raygen / closest-hit / miss / any-hit +
// shader binding table + cmd_trace_rays), the multi-material SBT shading path.
//
// LAYOUT-FREE: the pipeline is created with
// `PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT` — no descriptor set layouts, no
// pipeline layout, no pools/sets. Shaders keep their classic `[[vk::binding]]`
// declarations; every (set, binding) is sourced host-side from the
// [`BindingSeam`](crate::gpu::binding_seam::BindingSeam)'s descriptor heap via
// per-stage mapping tables ([`build_heap_mappings`]): scene set 0 and columns
// set 2 at constant heap offsets (their slots are app-lifetime), the per-view
// set 1 through push-data slot indices (one linked pipeline serves every
// view), and the TLAS from a device address in push data (shader-side heap AS
// access device-losts on current NVIDIA drivers). The trace binds the heaps +
// pushes 76 bytes; `VK_EXT_descriptor_heap` (NVIDIA R610+) is required.
//
// The RT-stage shaders are all Slang, compiled from source at library build
// via `gpu/slang.rs` (variant axes are preprocessor defines and swappable
// modules), then handed to `vkCreateRayTracingPipelinesKHR`. Mirrors
// `gpu/allocator.rs`'s raw-VK style; gated on the `RayTracingPipelineFeature`
// device feature.
#![allow(unsafe_code)]

use ash::khr;
use ash::vk::{self, TaggedStructure};
use bevy_ecs::component::Component;
use bevy_ecs::resource::Resource;
use core::ffi::CStr;
#[cfg(test)]
use wgpu::naga;

use super::allocator::Allocator;
use super::slang_sources::SlangSources;

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

// Push-data layout (`vkCmdPushDataEXT`, one blob per trace). The shaders have
// no `[[vk::push_constant]]` block — push data exists purely as a mapping
// source: the TLAS device address, then one u32 heap-slot index per set-1
// binding (the per-view resources; see `build_heap_mappings` and the
// `HEAP_WITH_PUSH_INDEX` entries it emits, whose `push_offset`s index here).
const PUSH_TLAS_ADDRESS_OFFSET: usize = 0;
const PUSH_VIEW_SLOTS_OFFSET: usize = 8;
const PUSH_DATA_SIZE: usize = PUSH_VIEW_SLOTS_OFFSET + (BINDING_NRC_QUERIES as usize + 1) * 4;

/// Build the mapping table chained onto every RT stage: how each classic
/// `[[vk::binding(b, set)]]` the shaders declare is sourced under the
/// layout-free heap pipeline.
///
/// - Scene set 0 + columns set 2: constant heap offsets from the staging
///   mirrors' slots ([`SceneHeapSlots`], [`SceneColumns::heap_slots`]) — both
///   are allocated once and rewritten in place, so the baked offsets never go
///   stale. The scene's non-buffer binding numbers (2/3/4/6/7/13/14) mirror
///   `raytracing_scene_bindings.wgsl` / the binder's layout.
/// - The TLAS (0,4): a device address in push data (`PUSH_ADDRESS`) — the
///   PTLAS double-buffers, and shader-side heap AS access device-losts.
/// - Set 1: heap indices read from push data (`HEAP_WITH_PUSH_INDEX`), so one
///   linked pipeline serves every view and survives view rebuilds
///   (resize/skybox swap) without relinking.
///
/// [`SceneHeapSlots`]: crate::bindings::SceneHeapSlots
/// [`SceneColumns::heap_slots`]: crate::ecs_gpu::SceneColumns
pub fn build_heap_mappings(
    seam: &crate::gpu::binding_seam::BindingSeam,
    scene: &crate::bindings::SceneHeapSlots,
    columns: &[(u32, u32)],
) -> Vec<vk::DescriptorSetAndBindingMappingEXT<'static>> {
    use crate::gpu::binding_seam::HeapKind;
    let mut mappings =
        Vec::with_capacity(scene.buffers.len() + 7 + (BINDING_NRC_QUERIES as usize + 1) + columns.len());
    for &(binding, slot) in &scene.buffers {
        mappings.push(seam.map_binding(0, binding, HeapKind::Buffer, slot));
    }
    mappings.push(seam.map_binding(0, 2, HeapKind::Image, scene.texture_block));
    mappings.push(seam.map_binding(0, 3, HeapKind::Sampler, scene.sampler_block));
    mappings.push(seam.map_binding_push_address(0, 4, PUSH_TLAS_ADDRESS_OFFSET as u32));
    mappings.push(seam.map_binding(0, 6, HeapKind::Image, scene.dfg_lut));
    mappings.push(seam.map_binding(0, 7, HeapKind::Sampler, scene.dfg_sampler));
    mappings.push(seam.map_binding(0, 13, HeapKind::Image, scene.texture_array_block));
    mappings.push(seam.map_binding(0, 14, HeapKind::Sampler, scene.array_sampler));
    for binding in 0..=BINDING_NRC_QUERIES {
        let kind = match binding {
            BINDING_ENV_MAP => HeapKind::Image,
            BINDING_ENV_SAMPLER => HeapKind::Sampler,
            _ => HeapKind::Buffer,
        };
        mappings.push(seam.map_binding_push_index(
            1,
            binding,
            kind,
            (PUSH_VIEW_SLOTS_OFFSET + binding as usize * 4) as u32,
        ));
    }
    for &(binding, slot) in columns {
        mappings.push(seam.map_binding(2, binding, HeapKind::Buffer, slot));
    }
    // Set 3: record-sourced bindings. (3,0) is the chits' per-material record
    // block (`SbtRecord` in chit_opaque/chit_glass), read inline from the hit
    // record's data bytes — the fields `write_record` bakes.
    mappings.push(seam.map_binding_shader_record_data(3, 0, 0));
    mappings
}

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

/// One Slang shader stage of a hit group, compiled at pipeline build via
/// `gpu/slang.rs`. `entry` is the entry function's name in `source` (the
/// compiled `OpEntryPoint` is renamed `"main"`). The source may `import` the
/// built-in module set (`scene_resolve`/`brdf`/`sampling`/…) and its group's
/// [`composable_modules`](SolariHitGroupDef::composable_modules). `&'static`
/// (usually `include_str!`) so downstream crates register without forking.
#[derive(Clone)]
pub struct SolariRtShader {
    pub source: &'static str,
    pub file: &'static str,
    pub entry: &'static str,
}

/// One RT hit group (closest-hit + optional any-hit, e.g. alpha cutout); its
/// registry index is its SBT class.
#[derive(Clone)]
pub struct SolariHitGroupDef {
    pub label: &'static str,
    pub closest_hit: SolariRtShader,
    pub any_hit: Option<SolariRtShader>,
    /// Extra `(module_name, source)` Slang modules importable by this group's
    /// stages alongside the built-in set — lets a downstream crate `import` its
    /// own shared Slang (e.g. a terrain function used by both a compute pass
    /// and a closest-hit) without forking. They may import the built-ins and
    /// each other. Scoped to this group: other groups never see them.
    pub composable_modules: &'static [(&'static str, &'static str)],
}

/// Ordered RT hit groups consumed by [`RtPipeline::new`]; index = SBT class.
#[derive(bevy_ecs::resource::Resource, Default, Clone)]
pub struct SolariHitGroupRegistry {
    pub groups: Vec<SolariHitGroupDef>,
}

impl SolariHitGroupRegistry {
    /// Append a hit group; returns its SBT class (its index).
    ///
    /// The group's stages are compiled eagerly (with its `composable_modules`)
    /// so a broken user shader is reported at registration — at pipeline-build
    /// time a compile failure in ANY group aborts the whole RT pipeline, which
    /// is far harder to attribute.
    pub fn register(&mut self, group: SolariHitGroupDef) -> u32 {
        for (kind, shader) in [
            ("closest-hit", Some(&group.closest_hit)),
            ("any-hit", group.any_hit.as_ref()),
        ] {
            let Some(shader) = shader else { continue };
            if let Err(e) = compile_group_shader(&group, shader, None) {
                bevy_log::error!(
                    "rt_pipeline: hit group '{}': {kind} failed to compile: {e}. \
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
/// class churn relinks with zero shader compiles. Also owns the descriptor
/// mapping table chained onto every stage — its constant-offset entries bake
/// the scene/columns heap slots, which are allocated once and rewritten in
/// place, so the table never goes stale and the cache is never invalidated.
#[derive(Resource)]
pub struct RtLibraryCache {
    device: ash::Device,
    rt: khr::ray_tracing_pipeline::Device,
    /// The per-stage descriptor mapping table (see [`build_heap_mappings`]),
    /// chained onto every library stage's create info — kept because library
    /// compiles are lazy (a sky swap or new hit group compiles long after the
    /// cache was created). The driver copies the mappings at create time.
    heap_mappings: Vec<vk::DescriptorSetAndBindingMappingEXT<'static>>,
    raygen: Option<RtLibrary>,
    /// Composed with the `custom_sky` module; the key is the module source's
    /// generation, so a sky swap rebuilds exactly this library.
    miss: Option<(u64, RtLibrary)>,
    miss_shadow: Option<RtLibrary>,
    /// Index-aligned with [`SolariHitGroupRegistry::groups`], which is
    /// append-only — existing entries never change identity, so cached
    /// libraries stay valid and only NEW registry entries compile.
    hit_groups: Vec<RtLibrary>,
    /// The [`SlangSources`] generation the cached libraries were compiled
    /// from; a mismatch means a shader file was edited — the dispatch calls
    /// [`invalidate_sources`](Self::invalidate_sources).
    sources_generation: u64,
    /// See the twin field on [`RtPipeline`].
    _device_keepalive: Allocator,
}

// SAFETY: plain Vulkan handles; used solely from the single render-schedule
// dispatch system.
unsafe impl Send for RtLibraryCache {}
unsafe impl Sync for RtLibraryCache {}

impl RtLibraryCache {
    /// Store the mapping table; libraries compile lazily via the `ensure_*`
    /// methods on first pipeline build, each stage chained with the table.
    pub fn new(
        allocator: &Allocator,
        heap_mappings: Vec<vk::DescriptorSetAndBindingMappingEXT<'static>>,
        sources_generation: u64,
    ) -> Self {
        let device = allocator.device().clone();
        // SAFETY: instance + device are live; loading the RT-pipeline function
        // table is valid because the extension was enabled at device creation.
        let rt = khr::ray_tracing_pipeline::Device::load(allocator.instance(), &device);
        Self {
            device,
            rt,
            heap_mappings,
            raygen: None,
            miss: None,
            miss_shadow: None,
            hit_groups: Vec::new(),
            sources_generation,
            _device_keepalive: allocator.clone(),
        }
    }

    /// The [`SlangSources`] generation the cached libraries came from.
    pub fn sources_generation(&self) -> u64 {
        self.sources_generation
    }

    /// Destroy every cached library: a source edit invalidates all compiled
    /// SPIR-V (the shared modules cross every stage). The next pipeline
    /// build recompiles from the live sources.
    pub fn invalidate_sources(&mut self, generation: u64) {
        self.sources_generation = generation;
        let libraries: Vec<RtLibrary> = self
            .raygen
            .take()
            .into_iter()
            .chain(self.miss.take().map(|(_, lib)| lib))
            .chain(self.miss_shadow.take())
            .chain(std::mem::take(&mut self.hit_groups))
            .collect();
        for lib in libraries {
            self.destroy_library(lib);
        }
    }

    /// Compile one library: `stages` + `groups`, LAYOUT-FREE
    /// (`DESCRIPTOR_HEAP_EXT`), each stage chained with the cache's mapping
    /// table, with the shared ray interface. Every library (and the link)
    /// opts into cluster acceleration structures and opacity micromaps —
    /// these must agree across the whole linked pipeline.
    fn create_library(
        &self,
        stages: &[vk::PipelineShaderStageCreateInfo],
        groups: &[vk::RayTracingShaderGroupCreateInfoKHR],
        modules: Vec<vk::ShaderModule>,
    ) -> Option<RtLibrary> {
        // One mapping-info struct shared read-only by every stage's pNext.
        let mut mapping_info = vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default();
        mapping_info.mapping_count = self.heap_mappings.len() as u32;
        mapping_info.p_mappings = self.heap_mappings.as_ptr();
        let stages: Vec<vk::PipelineShaderStageCreateInfo> = stages
            .iter()
            .map(|s| {
                let mut s = *s;
                debug_assert!(s.p_next.is_null());
                s.p_next = (&mapping_info
                    as *const vk::ShaderDescriptorSetAndBindingMappingInfoEXT)
                    .cast();
                s
            })
            .collect();
        let cluster_info =
            vk::RayTracingPipelineClusterAccelerationStructureCreateInfoNV::default()
                .allow_cluster_acceleration_structure(true);
        // With a `PipelineCreateFlags2CreateInfo` chained, the legacy `flags`
        // field is ignored — every flag (incl. the heap opt-in, which has no
        // legacy bit) lives here.
        let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default().flags(
            vk::PipelineCreateFlags2::LIBRARY_KHR
                | vk::PipelineCreateFlags2::RAY_TRACING_OPACITY_MICROMAP_EXT
                | vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT,
        );
        let interface = vk::RayTracingPipelineInterfaceCreateInfoKHR::default()
            .max_pipeline_ray_payload_size(MAX_RAY_PAYLOAD_SIZE)
            .max_pipeline_ray_hit_attribute_size(MAX_HIT_ATTRIBUTE_SIZE);
        let mut info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(groups)
            // Depth 2: raygen's hit object executes the closest-hit (1), which
            // traces a NEE shadow ray (2). Must agree with the link.
            .max_pipeline_ray_recursion_depth(2)
            .library_interface(&interface);
        // ash doesn't register the cluster struct as an extender (no typed
        // `push_next`); chain flags2 -> cluster via raw `p_next`. Both outlive
        // the call.
        flags2.p_next =
            (&cluster_info as *const vk::RayTracingPipelineClusterAccelerationStructureCreateInfoNV)
                .cast();
        info.p_next = (&flags2 as *const vk::PipelineCreateFlags2CreateInfo).cast();
        // SAFETY: stages/groups reference live modules; the mapping table
        // outlives the call (owned by self).
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
    ///
    /// [`RAYGEN_CAPABILITIES`] pins Shader Execution Reordering to the NV
    /// SPIR-V flavor — the `VK_NV_ray_tracing_invocation_reorder` extension is
    /// what the device enables; left unconstrained, slang emits the EXT flavor
    /// and the module is invalid on this device (VUID 08740/08742).
    fn ensure_raygen(&mut self, sources: &SlangSources) -> Option<()> {
        if self.raygen.is_some() {
            return Some(());
        }
        // The `SOLARI_SHADER_CLOCK` define compiles in the cost-heatmap clock
        // reads, legal only when the device enabled `VK_KHR_shader_clock`.
        let defines: &[(&str, &str)] = if crate::gpu::extension::shader_clock_available() {
            &[("SOLARI_SHADER_CLOCK", "1")]
        } else {
            &[]
        };
        let raygen_spv = crate::gpu::slang::compile_rt_slang(
            "raygen.slang",
            sources.source("raygen.slang"),
            "raygen",
            &rt_slang_modules(Some(sources)),
            defines,
            RAYGEN_CAPABILITIES,
        )
        .map_err(|e| bevy_log::error!("rt_pipeline: {e}"))
        .ok()?;
        let module = create_shader_module(&self.device, &raygen_spv.spirv)?;
        let lib = self.create_library(
            &[shader_stage(vk::ShaderStageFlags::RAYGEN_KHR, module, c"main")],
            &[general_group(0)],
            vec![module],
        )?;
        self.raygen = Some(lib);
        Some(())
    }

    /// Primary-miss library — composes the swappable `custom_sky` module
    /// (`SolariSky::Shader`; defaults to the built-in procedural gradient).
    /// A generation change rebuilds exactly this library.
    fn ensure_miss(&mut self, sources: &SlangSources, custom_sky: (&str, u64)) -> Option<()> {
        let (custom_sky_source, generation) = custom_sky;
        if matches!(&self.miss, Some((cached, _)) if *cached == generation) {
            return Some(());
        }
        let miss_spv = crate::gpu::slang::compile_rt_slang(
            "miss.slang",
            sources.source("miss.slang"),
            "miss_primary",
            &[
                ("rt_payload", sources.source("rt_payload.slang")),
                ("custom_sky", custom_sky_source),
            ],
            &[],
            &[],
        )
        .map_err(|e| bevy_log::error!("rt_pipeline: {e}"))
        .ok()?;
        let module = create_shader_module(&self.device, &miss_spv.spirv)?;
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

    /// Shadow-miss library (static — no modules, no variant axes).
    fn ensure_shadow(&mut self, sources: &SlangSources) -> Option<()> {
        if self.miss_shadow.is_some() {
            return Some(());
        }
        let shadow_spv = crate::gpu::slang::compile_rt_slang(
            "miss_shadow.slang",
            sources.source("miss_shadow.slang"),
            "miss_shadow",
            &[],
            &[],
            &[],
        )
        .map_err(|e| bevy_log::error!("rt_pipeline: {e}"))
        .ok()?;
        let module = create_shader_module(&self.device, &shadow_spv.spirv)?;
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
    fn ensure_hit_groups(
        &mut self,
        sources: &SlangSources,
        hit_groups: &[SolariHitGroupDef],
    ) -> Option<()> {
        for hg in &hit_groups[self.hit_groups.len()..] {
            let compile = |shader| {
                let shader = compile_group_shader(hg, shader, Some(sources))
                    .map_err(|e| bevy_log::error!("rt_pipeline: {e}"))
                    .ok()?;
                create_shader_module(&self.device, &shader.spirv)
            };
            let chit_mod = compile(&hg.closest_hit)?;
            let lib = if let Some(ah) = &hg.any_hit {
                let ah_mod = compile(ah)?;
                self.create_library(
                    &[
                        shader_stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, chit_mod, c"main"),
                        shader_stage(vk::ShaderStageFlags::ANY_HIT_KHR, ah_mod, c"main"),
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
    ///
    /// Profiler note: Nsight's SASS↔source correlation is currently lost
    /// across the library link — the RT stages show as Unattributed, while
    /// the same modules attribute fine in a monolithic create (and in the
    /// heap compute pipelines). Findings + leads in `docs/slang_lib.md`.
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
        let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default().flags(
            vk::PipelineCreateFlags2::RAY_TRACING_OPACITY_MICROMAP_EXT
                | vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT,
        );
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
            .max_pipeline_ray_recursion_depth(2)
            .library_info(&library_info)
            .library_interface(&interface)
            .dynamic_state(&dynamic_info);
        flags2.p_next =
            (&cluster_info as *const vk::RayTracingPipelineClusterAccelerationStructureCreateInfoNV)
                .cast();
        info.p_next = (&flags2 as *const vk::PipelineCreateFlags2CreateInfo).cast();
        // SAFETY: libraries live (owned by this cache).
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
/// SBT. The per-view resources (heap slots, camera UBO, output-buffer slot, env
/// cube) live in [`RtViewBindings`], a component, so multiple views
/// (split-screen) each trace into their own output with their own camera/env.
///
/// Built lazily once the scene heap slots and materials exist;
/// `RayTracingPipelineFeature` + the descriptor heap are hard requirements, so
/// absence here only ever means "not built yet".
#[derive(Resource)]
pub struct RtPipeline {
    device: ash::Device,
    rt: khr::ray_tracing_pipeline::Device,

    pipeline: vk::Pipeline,

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

/// Per-view ray-tracing resources: the descriptor-heap slots for the set-1
/// bindings (pushed as indices at trace time), the geometry-address UBO, and
/// the env image to transition around the trace. One per [`SolariCamera`]
/// view, so split-screen views don't share an output buffer or camera. Built
/// from [`RtPipeline::create_view_bindings`]; rebuilt when the view's output
/// buffer is reallocated (viewport resize) or the env view changes.
#[derive(Component)]
pub struct RtViewBindings {
    device: ash::Device,
    /// Bindless geometry addresses (set 1, binding 4); refreshed per frame via the
    /// mapping (`set_geometry_addresses`). Not ringed: the addresses are stable
    /// (stable-address `RawTraceBindable` buffers), so an in-flight overwrite writes
    /// identical bytes — benign, unlike the per-frame-varying camera.
    geometry: MappedBuffer,
    /// `Some` ⇒ the env cube is the storage atmosphere cube (GENERAL); transition
    /// it around each trace. `None` ⇒ already a read-optimal wgpu-sampled texture.
    env_map_image: Option<vk::Image>,
    /// The output `VkBuffer` behind the output heap slot. The dispatch rebuilds
    /// this component if the view's output buffer changes (resize).
    output_buffer: vk::Buffer,
    /// The env cube view behind the env heap slot, for the same rebuild check:
    /// a skybox that finishes loading (or is swapped) changes the view, and the
    /// slots would otherwise sample the stale cube forever.
    env_map_view: vk::ImageView,
    /// This view's set-1 resources as heap slots; the trace pushes the indices
    /// (see `PUSH_VIEW_SLOTS_OFFSET`) for the pipeline's `HEAP_WITH_PUSH_INDEX`
    /// mappings.
    heap: RtViewHeapSlots,
    /// Keeps the `VkDevice` alive until this drops — see the twin field on
    /// [`RtPipeline`]; without it, teardown drop order decides whether [`Drop`]'s
    /// raw destroys run against a dead device.
    _device_keepalive: Allocator,
}

/// The set-1 resources as descriptor-heap slots (buffer-region indices, plus
/// the env cube in the image region and its sampler in the sampler region).
/// Slot order mirrors the set-1 binding list: output, camera (uniform),
/// geometry (uniform), the DLSS G-buffers, then
/// reservoirs/surface/light_samples/gi_samples/nrc×4.
pub struct RtViewHeapSlots {
    seam: crate::gpu::binding_seam::BindingSeam,
    buffers: Vec<u32>,
    env_map: u32,
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
        self.seam.free_heap_index(HeapKind::Image, self.env_map);
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
        seam: &crate::gpu::binding_seam::BindingSeam,
        libraries: &mut RtLibraryCache,
        material_classes: &[u32],
        hit_groups: &[SolariHitGroupDef],
        custom_sky: (&str, u64),
        sources: &SlangSources,
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
        libraries.ensure_raygen(sources)?;
        libraries.ensure_miss(sources, custom_sky)?;
        libraries.ensure_shadow(sources)?;
        libraries.ensure_hit_groups(sources, hit_groups)?;
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

        // --- SBT: raygen + 2 miss local; ONE HIT RECORD PER MATERIAL (+ hair)
        // in the seam's record table -------------------------------------------
        // The hit region holds one record per material slot; an instance's
        // `instance_contribution_to_hit_group_index` = its material slot selects
        // its record. Each HIT record is [shader group handle | fields]; the
        // fields hold the material id (= record index), which the chits read
        // through the record-sourced set-3 mapping (uniform per record →
        // uniform per warp after SER). Distinct per-material records also give
        // SER a per-material reorder key. Records live in the seam's table
        // (`write_record`), rewritten wholesale here — safe because every
        // rebuild path drains the GPU before dropping the old pipeline, and the
        // first build precedes any trace.
        const MISS_COUNT: u64 = 2; // miss index 0 = primary, 1 = shadow
        const RECORD_HEADROOM: u32 = 1024; // absorb streaming material growth post-build
        // One extra hit record, appended AFTER the per-material records, baked with
        // the hair hit-group handle (group 4). Hair instances route to it
        // (`hair_sbt_record`) via `ptlas_hair_write`; it's a single shared record
        // (chit_hair keys off the instance, not a per-record material id). Kept off
        // the material region so material routing/SER is untouched.
        const HAIR_RECORDS: u64 = 1;
        // Headroom clamps to the seam's record table; the live material count
        // itself must fit outright.
        let max_capacity = (crate::gpu::binding_seam::MAX_RECORDS - HAIR_RECORDS) as u32;
        assert!(
            material_count <= max_capacity,
            "rt_pipeline: {material_count} material slots exceed the \
             {}-record SBT table",
            crate::gpu::binding_seam::MAX_RECORDS,
        );
        let record_capacity = (material_count + RECORD_HEADROOM).min(max_capacity);
        let total_records = record_capacity as u64 + HAIR_RECORDS;
        let handle_stride = align_up(handle_size, handle_align);
        let raygen_offset = 0u64;
        let miss_offset = align_up(handle_stride, base_align);
        // The local SBT holds only raygen + the MISS_COUNT miss records.
        let sbt_size = miss_offset + MISS_COUNT * handle_stride;
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
        // count, and any out-of-range class, fall back to opaque. The fields =
        // the record index = the material id, read back through the set-3
        // record mapping (uniform per record → uniform per warp after SER).
        for record in 0..total_records {
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
            seam.write_record(
                record as u32,
                group_handle,
                &(record as u32).to_le_bytes(),
            );
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
        let hit_region = seam.record_region(0, total_records as u32);
        let callable_region = vk::StridedDeviceAddressRegionKHR::default();

        let out = Self {
            device,
            rt,
            pipeline,
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

    /// Build the per-view resources for one view: the geometry-address UBO and
    /// the heap slots for every set-1 binding ([`RtViewHeapSlots`] — the trace
    /// pushes the slot indices). Descriptors are written once; the camera
    /// *contents* change per frame in the GPU buffer behind its slot (the
    /// `rt_camera` compute pass writes it), and both the output and camera
    /// buffers are stable (the dispatch rebuilds this whole component if the
    /// view's output buffer is reallocated). `env_map_image` is `Some` when the
    /// env cube is the storage atmosphere cube (transitioned around the trace).
    pub fn create_view_bindings(
        &self,
        allocator: &Allocator,
        seam: &crate::gpu::binding_seam::BindingSeam,
        output_buffer: vk::Buffer,
        output_size: u64,
        // The per-view `RtCamera` `VkBuffer` (wgpu-owned) behind the camera slot.
        camera_buffer: vk::Buffer,
        // DLSS guide G-buffers `(VkBuffer, size)` at BINDING_GBUFFER_* (set 1).
        // Empty unless the `dlss` feature is on (the WGSL then has no gbuffer
        // bindings either).
        gbuffers: &[(vk::Buffer, u64)],
        // ReSTIR reservoir buffer `(VkBuffer, size)` at BINDING_RESERVOIRS.
        reservoirs: (vk::Buffer, u64),
        // ReSTIR surface G-buffer `(VkBuffer, size)` at BINDING_SURFACE.
        surface: (vk::Buffer, u64),
        // ReSTIR winner light samples `(VkBuffer, size)` at BINDING_LIGHT_SAMPLES.
        light_samples: (vk::Buffer, u64),
        // ReSTIR GI canonical samples `(VkBuffer, size)` at BINDING_GI_SAMPLES.
        gi_samples: (vk::Buffer, u64),
        // NRC inference weights / biases / training records / screen (BINDING_NRC_*).
        nrc_weights: (vk::Buffer, u64),
        nrc_bias: (vk::Buffer, u64),
        nrc_records: (vk::Buffer, u64),
        nrc_queries: (vk::Buffer, u64),
        env_map_view: vk::ImageView,
        env_map_image: Option<vk::Image>,
        // The env view's create info (the fork's hal `TextureView` records it) —
        // heap image descriptors are written from create info, not the live
        // view above.
        env_view_info: &vk::ImageViewCreateInfo<'static>,
    ) -> Option<RtViewBindings> {
        use crate::gpu::binding_seam::HeapResource;

        let geometry = alloc_mapped_buffer(
            allocator,
            size_of::<RtGeometryAddresses>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        // This view's env-cube sampler CONFIG — the heap sampler descriptor is
        // written from create info; no `VkSampler` object exists. Clamp-to-edge
        // linear is fine for a cube.
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(vk::LOD_CLAMP_NONE);

        // Heap slots for every set-1 resource. Uniform vs storage matches the
        // shader declarations (camera + geometry are uniforms); the buffer-slot
        // order mirrors the binding list (see `trace`'s push-blob assembly).
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
        // The trace transitions the env cube to SHADER_READ_ONLY_OPTIMAL around
        // the dispatch, so the descriptor always sees read-optimal.
        let env_map = seam.alloc_heap_index(HeapResource::SampledImage {
            view: env_view_info,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        });
        let env_sampler = seam.alloc_heap_index(HeapResource::Sampler(&sampler_info));
        let heap = RtViewHeapSlots {
            seam: seam.clone(),
            buffers,
            env_map,
            env_sampler,
        };

        Some(RtViewBindings {
            device: self.device.clone(),
            geometry,
            env_map_image,
            output_buffer,
            env_map_view,
            heap,
            _device_keepalive: allocator.clone(),
        })
    }

    /// Record bind + `cmd_trace_rays` into `command_buffer` for a `width`×`height`
    /// dispatch, writing this view's per-pixel output storage buffer. There are
    /// no descriptor sets: the heaps are bound, then one push blob supplies the
    /// TLAS device address plus this view's heap-slot indices for the
    /// pipeline's push-sourced mappings.
    ///
    /// # Safety
    /// `command_buffer` must be recording; `tlas_address` must be the current
    /// PTLAS's device address; `view` must have been built by
    /// `self.create_view_bindings` against the same seam the pipeline's
    /// mappings were.
    pub unsafe fn trace(
        &self,
        command_buffer: vk::CommandBuffer,
        seam: &crate::gpu::binding_seam::BindingSeam,
        view: &RtViewBindings,
        tlas_address: u64,
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

            // Heaps + push data replace descriptor sets entirely. The push blob
            // feeds the pipeline's mapping sources: TLAS device address at
            // offset 0 (`PUSH_ADDRESS`), then one u32 heap-slot index per set-1
            // binding (`HEAP_WITH_PUSH_INDEX`) — indexed by binding number, so
            // the assembly below must mirror `build_heap_mappings`' set-1 loop.
            seam.bind_heaps(command_buffer);
            let mut push = [0u8; PUSH_DATA_SIZE];
            push[PUSH_TLAS_ADDRESS_OFFSET..PUSH_TLAS_ADDRESS_OFFSET + 8]
                .copy_from_slice(&tlas_address.to_le_bytes());
            {
                let h = &view.heap;
                let mut by_binding = [0u32; BINDING_NRC_QUERIES as usize + 1];
                by_binding[BINDING_OUTPUT as usize] = h.buffers[0];
                by_binding[BINDING_CAMERA as usize] = h.buffers[1];
                by_binding[BINDING_ENV_MAP as usize] = h.env_map;
                by_binding[BINDING_ENV_SAMPLER as usize] = h.env_sampler;
                by_binding[BINDING_GEOMETRY as usize] = h.buffers[2];
                // The buffer-slot list continues [gbuffers×0/4, reservoirs,
                // surface, light_samples, gi_samples, nrc×4] (see
                // `create_view_bindings`); without DLSS the gbuffer bindings
                // stay 0 — the shaders don't declare them.
                let gbuffer_count = h.buffers.len() - 11;
                let gbuffer_bindings = [
                    BINDING_GBUFFER_NORMAL,
                    BINDING_GBUFFER_DIFFUSE,
                    BINDING_GBUFFER_SPECULAR,
                    BINDING_GBUFFER_MOTION,
                ];
                let mut next = 3;
                for &binding in &gbuffer_bindings[..gbuffer_count] {
                    by_binding[binding as usize] = h.buffers[next];
                    next += 1;
                }
                for binding in [
                    BINDING_RESERVOIRS,
                    BINDING_SURFACE,
                    BINDING_LIGHT_SAMPLES,
                    BINDING_GI_SAMPLES,
                    BINDING_NRC_WEIGHTS,
                    BINDING_NRC_BIAS,
                    BINDING_NRC_RECORDS,
                    BINDING_NRC_QUERIES,
                ] {
                    by_binding[binding as usize] = h.buffers[next];
                    next += 1;
                }
                for (i, slot) in by_binding.into_iter().enumerate() {
                    push[PUSH_VIEW_SLOTS_OFFSET + i * 4..PUSH_VIEW_SLOTS_OFFSET + i * 4 + 4]
                        .copy_from_slice(&slot.to_le_bytes());
                }
            }
            seam.push_data(command_buffer, &push);

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

    /// Heap slot of the per-pixel output buffer (set-1 binding 0) — the NRC
    /// query composite writes through it.
    pub fn output_slot(&self) -> u32 {
        self.heap.buffers[0]
    }

    /// Heap slot of the NRC termination-query ring (set-1 binding 16, the last
    /// buffer slot) — the NRC query-infer kernel consumes it.
    pub fn nrc_queries_slot(&self) -> u32 {
        *self.heap.buffers.last().unwrap()
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
        // In-flight traces may still read the geometry UBO and the heap slots
        // (freed by `RtViewHeapSlots`' own Drop right after this body); drain
        // first (near-free when the rebuild path already drained).
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: the geometry buffer was created for this view; the queue is
        // drained and the device alive (keepalive). `camera_buffer` is
        // wgpu-owned (the per-view `RtOutputBuffer`), freed there.
        unsafe {
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

/// Raygen's target capability atoms (`slangc -capability` equivalents) —
/// see [`RtLibraryCache::ensure_raygen`]. Declaring any atom makes the set
/// raygen's whole target profile, so it must cover everything the entry
/// uses (clock + coopvec too) or slang warns "profile implicitly upgraded".
/// Declaring clock support is fine for the non-clock variant — the set says
/// what the target supports, not what the shader must use.
const RAYGEN_CAPABILITIES: &[&str] = &[
    "spvShaderInvocationReorderNV",
    "spvCooperativeVectorNV",
    "spvShaderClockKHR",
];

/// The built-in Slang modules importable by every RT stage
/// (`import scene_resolve;` etc.).
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

/// The built-in module set, resolved through the live source registry when
/// one is at hand (a pipeline build — module edits hot reload) and from the
/// embedded copies otherwise (registration-time validation, tests).
fn rt_slang_modules(sources: Option<&SlangSources>) -> Vec<(&'static str, &'static str)> {
    RT_SLANG_MODULES
        .iter()
        .map(|&(name, embedded)| {
            let live = sources.and_then(|s| s.get(&format!("{name}.slang")));
            (name, live.unwrap_or(embedded))
        })
        .collect()
}

/// Compile one of a hit group's stages with the built-in module set plus the
/// group's own `composable_modules` importable. Shared by eager registration
/// validation (`sources` = `None`: the registered snapshot is what's being
/// validated) and the pipeline build (the live registry wins, so the
/// built-in hit shaders hot reload; a downstream group's own files aren't
/// watched and always use their registered source).
fn compile_group_shader(
    group: &SolariHitGroupDef,
    shader: &SolariRtShader,
    sources: Option<&SlangSources>,
) -> Result<crate::gpu::slang::CompiledShader, String> {
    let mut modules = rt_slang_modules(sources);
    modules.extend_from_slice(group.composable_modules);
    let live = sources.and_then(|s| s.get(shader.file));
    crate::gpu::slang::compile_rt_slang(
        shader.file,
        live.unwrap_or(shader.source),
        shader.entry,
        &modules,
        &[],
        &[],
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

    /// Every `(set, binding)` a stage declares must be covered by the heap
    /// mapping surface (`build_heap_mappings` + the push blob): scene set 0
    /// bindings 0..=14, view set 1 bindings 0..=16, columns set 2 (mapped as
    /// a runtime-sized block), record set 3 binding 0. A shader gaining a
    /// binding without the table/push-blob growing would otherwise surface as
    /// a misrouted descriptor at runtime.
    fn assert_bindings_mapped(file: &str, spv: &[u32]) {
        for (set, binding) in crate::gpu::binding_seam::spirv_descriptor_bindings(spv) {
            let mapped = match set {
                0 => binding <= 14,
                1 => binding <= 16,
                2 => true,
                3 => binding == 0,
                _ => false,
            };
            assert!(
                mapped,
                "{file}: (set {set}, binding {binding}) has no heap mapping"
            );
        }
    }

    /// Every pipeline stage is created with entry name `"main"` — assert the
    /// compiled module's `OpEntryPoint` actually carries that name (the
    /// compile path renames the entry function).
    fn assert_entry_is_main(file: &str, spv: &[u32]) {
        let mut i = 5;
        while i < spv.len() {
            let word_count = (spv[i] >> 16) as usize;
            if word_count == 0 || i + word_count > spv.len() {
                break;
            }
            if spv[i] & 0xFFFF == 15 {
                // OpEntryPoint: word 1 = execution model, 2 = entry id,
                // 3.. = the literal name string.
                let bytes: Vec<u8> = spv[i + 3..i + word_count]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect();
                let name_end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                let name = String::from_utf8_lossy(&bytes[..name_end]).into_owned();
                assert_eq!(name, "main", "{file}: OpEntryPoint is not named main");
                return;
            }
            i += word_count;
        }
        panic!("{file}: no OpEntryPoint found");
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

        // The primary miss composes the swappable `custom_sky` module. Compile
        // it with the default procedural sky AND a user-style replacement —
        // the `SolariSky::Shader` path — so both stay proven headlessly.
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
                &[
                    (
                        "rt_payload",
                        include_str!("../render/rt_pipeline/rt_payload.slang"),
                    ),
                    ("custom_sky", custom_sky),
                ],
                &[],
                &[],
            )
            .unwrap_or_else(|e| panic!("miss.slang: {e}"));
            assert_no_runtime_descriptor_array("miss.slang", &spv.spirv);
            assert_bindings_mapped("miss.slang", &spv.spirv);
        }

        // Every other stage compiles from source at pipeline build; compile
        // them all here — including both raygen define variants — so a shader
        // or module edit fails at `cargo test` instead of as a runtime
        // pipeline-build black screen. The binding/layout agreement across
        // modules (Cluster stride, Material offsets, payload words) holds by
        // construction: one compiler compiles every module in a link.
        use super::{RAYGEN_CAPABILITIES, RT_SLANG_MODULES};
        let clock: &[(&str, &str)] = &[("SOLARI_SHADER_CLOCK", "1")];
        let no_caps: &[&str] = &[];
        for (file, source, entry, defines, caps) in [
            (
                "raygen.slang",
                include_str!("../render/rt_pipeline/raygen.slang"),
                "raygen",
                &[][..],
                RAYGEN_CAPABILITIES,
            ),
            (
                "raygen.slang",
                include_str!("../render/rt_pipeline/raygen.slang"),
                "raygen",
                clock,
                RAYGEN_CAPABILITIES,
            ),
            (
                "miss_shadow.slang",
                include_str!("../render/rt_pipeline/miss_shadow.slang"),
                "miss_shadow",
                &[],
                no_caps,
            ),
            (
                "ahit_alpha.slang",
                include_str!("../render/rt_pipeline/ahit_alpha.slang"),
                "ahit_alpha",
                &[],
                no_caps,
            ),
            (
                "chit_opaque.slang",
                include_str!("../render/rt_pipeline/chit_opaque.slang"),
                "chit_opaque",
                &[],
                no_caps,
            ),
            (
                "chit_glass.slang",
                include_str!("../render/rt_pipeline/chit_glass.slang"),
                "chit_glass",
                &[],
                no_caps,
            ),
            (
                "chit_hair.slang",
                include_str!("../render/rt_pipeline/chit_hair.slang"),
                "chit_hair",
                &[],
                no_caps,
            ),
            (
                "chit_portal.slang",
                include_str!("../render/rt_pipeline/chit_portal.slang"),
                "chit_portal",
                &[],
                no_caps,
            ),
        ] {
            let spv =
                crate::gpu::slang::compile_rt_slang(file, source, entry, RT_SLANG_MODULES, defines, caps)
                    .unwrap_or_else(|e| panic!("{file}: {e}"));
            assert_no_runtime_descriptor_array(file, &spv.spirv);
            assert_bindings_mapped(file, &spv.spirv);
            assert_entry_is_main(file, &spv.spirv);
            // Profiler attribution (Nsight): every stage must carry source-
            // level debug info with the source text embedded.
            let bytes: Vec<u8> = spv.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
            let contains = |needle: &[u8]| bytes.windows(needle.len()).any(|w| w == needle);
            assert!(
                contains(b"NonSemantic.Shader.DebugInfo.100"),
                "{file}: no shader debug info emitted"
            );
            assert!(
                contains(b"[shader("),
                "{file}: source text not embedded in the debug info"
            );
            // Reflection is what dispatch tables are assembled from; every
            // binding surviving in the SPIR-V must appear there (the reverse
            // need not hold — reflection also lists DCE'd parameters).
            for pair in crate::gpu::binding_seam::spirv_descriptor_bindings(&spv.spirv) {
                assert!(
                    spv.bindings.iter().any(|&(_, s, b)| (s, b) == pair),
                    "{file}: SPIR-V binding {pair:?} missing from slang reflection"
                );
            }
            // The target capability must pin SER to the NV flavor — the EXT
            // capability/extension is invalid on the device (VUID 08740).
            let bytes: Vec<u8> = spv.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
            assert!(
                !bytes
                    .windows(b"SPV_EXT_shader_invocation_reorder".len())
                    .any(|w| w == b"SPV_EXT_shader_invocation_reorder"),
                "{file}: emitted the EXT shader-invocation-reorder flavor"
            );
        }
    }
}
