// Raw Vulkan ray-tracing PIPELINE (raygen / closest-hit / miss / any-hit +
// shader binding table + cmd_trace_rays), the multi-material SBT shading path.
// Self-contained: builds its own descriptor set layout, pool, sets, pipeline
// layout, pipeline, and SBT in raw `ash` — no wgpu-hal accessor additions. The
// WGSL ray-tracing-stage shaders are compiled to SPIR-V through the re-exported
// `wgpu::naga` (whose WGSL frontend + SPIR-V backend support RayGeneration /
// ClosestHit / AnyHit / Miss stages), then handed to
// `vkCreateRayTracingPipelinesKHR`.
//
// Mirrors `gpu/allocator.rs`'s raw-VK style; gated on the
// `RayTracingPipelineFeature` device feature.
#![allow(unsafe_code)]

use ash::khr;
use ash::vk::{self, TaggedStructure};
use bevy_ecs::component::Component;
use bevy_ecs::resource::Resource;
use core::ffi::CStr;
use wgpu::naga;

use super::allocator::Allocator;

/// Descriptor set 0 bindings the milestone RT shaders use. Kept tiny on purpose
/// — real material/scene resources arrive when the shading branches are ported.
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
#[cfg(feature = "dlss")]
const BINDING_GBUFFER_NORMAL: u32 = 5; // storage: normal.xyz + linear roughness (.w)
#[cfg(feature = "dlss")]
const BINDING_GBUFFER_DIFFUSE: u32 = 6; // storage: diffuse albedo.xyz + linear depth (.w)
#[cfg(feature = "dlss")]
const BINDING_GBUFFER_SPECULAR: u32 = 7; // storage: specular albedo.xyz + hit distance (.w)
#[cfg(feature = "dlss")]
const BINDING_GBUFFER_MOTION: u32 = 8; // storage: screen-space motion vector.xy (.zw unused)

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
    /// at the clear color in `.yzw`).
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

/// One RT closest-hit program; its registry index is its SBT class. WGSL is `&'static`
/// (usually `include_str!`) so downstream crates register without forking.
#[derive(Clone)]
pub struct SolariHitGroupDef {
    pub label: &'static str,
    pub closest_hit_wgsl: &'static str,
    pub closest_hit_file: &'static str,
    pub closest_hit_entry: &'static str,
    pub any_hit: Option<SolariAnyHitDef>,
    /// Extra `(file_path, source)` naga_oil modules composed into this group's
    /// chit/any-hit alongside the built-in `bevy_solari::*` set — lets a downstream
    /// crate `#import` its own shared WGSL (e.g. a terrain function used by both a
    /// compute pass and a closest-hit) without forking. Registered in slice order
    /// AFTER the built-ins, so they may import `bevy_solari::*` and each other
    /// (dependencies first). Scoped to this group: other groups never see them.
    pub composable_modules: &'static [(&'static str, &'static str)],
}

/// An any-hit program attached to a [`SolariHitGroupDef`] (alpha cutout, etc.).
#[derive(Clone)]
pub struct SolariAnyHitDef {
    pub wgsl: &'static str,
    pub file: &'static str,
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
    /// The group's `composable_modules` are validated eagerly (composed against the
    /// built-in module set) so a broken user module is reported at registration —
    /// at pipeline-build time a compose failure in ANY group aborts the whole RT
    /// pipeline, which is far harder to attribute.
    pub fn register(&mut self, group: SolariHitGroupDef) -> u32 {
        if !group.composable_modules.is_empty() {
            if let Err(e) = validate_composable_modules(group.composable_modules) {
                bevy_log::error!(
                    "rt_pipeline: hit group '{}': composable module failed to compose: {e}. \
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
/// Present only when `RayTracingPipelineFeature` is enabled; absence means the
/// inline-`rayQuery` compute path is the only shading path.
#[derive(Resource)]
pub struct RtPipeline {
    device: ash::Device,
    rt: khr::ray_tracing_pipeline::Device,

    pipeline: vk::Pipeline,
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

    /// Shader modules retained for the pipeline's lifetime (destroyed on drop).
    modules: Vec<vk::ShaderModule>,
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
}

// SAFETY: the host-visible geometry-address mapping is written only from the single
// render-schedule dispatch (via `&self` + coherent memory), never shared across
// threads. All other fields are plain Vulkan handles.
unsafe impl Send for RtViewBindings {}
unsafe impl Sync for RtViewBindings {}

impl RtPipeline {
    /// Build the RT pipeline (raygen + miss + opaque/glass/hair closest-hit).
    /// `scene_layout` / `columns_layout` are the raw `VkDescriptorSetLayout`s of
    /// wgpu's raytracing scene bind group (set 0) and scene-columns bind group
    /// (set 2) — obtained via `BindGroupLayout::as_hal().raw_handle()` — so the
    /// pipeline layout is compatible with the wgpu bind groups bound at trace
    /// time. Built lazily (see the dispatch) once those bind groups exist.
    /// Returns `None` if SPIR-V compilation or any Vulkan step fails (logged).
    pub fn new(
        allocator: &Allocator,
        scene_layout: vk::DescriptorSetLayout,
        columns_layout: vk::DescriptorSetLayout,
        material_classes: &[u32],
        hit_groups: &[SolariHitGroupDef],
    ) -> Option<Self> {
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

        // --- Shaders: WGSL -> SPIR-V -> VkShaderModule -------------------------
        // Fixed general programs (raygen + the two miss shaders); every closest-hit
        // ("hit group", + optional any-hit) comes from `hit_groups` (the registry), so
        // adding a surface shader needs no edit here — Solari's own opaque/glass/hair/
        // portal register the same way as any downstream material (see SolariPlugin).
        let raygen_mod = create_shader_module(
            &device,
            &compile_rt_wgsl(include_str!("../render/rt_pipeline/raygen.wgsl"), "raygen.wgsl", &[])?,
        )?;
        let miss_mod = create_shader_module(
            &device,
            &compile_rt_wgsl(include_str!("../render/rt_pipeline/miss.wgsl"), "miss.wgsl", &[])?,
        )?;
        let miss_shadow_mod = create_shader_module(
            &device,
            &compile_rt_wgsl(include_str!("../render/rt_pipeline/miss_shadow.wgsl"), "miss_shadow.wgsl", &[])?,
        )?;

        // Stage table: (flags, module, entry). Fixed stages first (raygen 0, primary
        // miss 1, shadow miss 2), then each hit group's chit (+ any-hit). Entry names
        // are `&'static`; CString'd just before pipeline create (kept alive there).
        let mut modules = vec![raygen_mod, miss_mod, miss_shadow_mod];
        let mut stage_specs: Vec<(vk::ShaderStageFlags, vk::ShaderModule, &'static str)> = vec![
            (vk::ShaderStageFlags::RAYGEN_KHR, raygen_mod, "raygen"),
            (vk::ShaderStageFlags::MISS_KHR, miss_mod, "miss_primary"),
            (vk::ShaderStageFlags::MISS_KHR, miss_shadow_mod, "miss_shadow"),
        ];
        // Per hit group: compile chit (+ any-hit), recording their stage indices.
        let mut hit_group_stages: Vec<(u32, Option<u32>)> = Vec::with_capacity(hit_groups.len());
        for hg in hit_groups {
            let chit_mod = create_shader_module(
                &device,
                &compile_rt_wgsl(hg.closest_hit_wgsl, hg.closest_hit_file, hg.composable_modules)?,
            )?;
            let chit_stage = stage_specs.len() as u32;
            modules.push(chit_mod);
            stage_specs.push((vk::ShaderStageFlags::CLOSEST_HIT_KHR, chit_mod, hg.closest_hit_entry));
            let any_hit_stage = if let Some(ah) = &hg.any_hit {
                let ah_mod = create_shader_module(
                    &device,
                    &compile_rt_wgsl(ah.wgsl, ah.file, hg.composable_modules)?,
                )?;
                let s = stage_specs.len() as u32;
                modules.push(ah_mod);
                stage_specs.push((vk::ShaderStageFlags::ANY_HIT_KHR, ah_mod, ah.entry));
                Some(s)
            } else {
                None
            };
            hit_group_stages.push((chit_stage, any_hit_stage));
        }

        // CStrings outlive the stage create-infos (which hold raw ptrs) until create.
        let entry_cstrings: Vec<std::ffi::CString> = stage_specs
            .iter()
            .map(|(_, _, e)| std::ffi::CString::new(*e).expect("shader entry name has interior NUL"))
            .collect();
        let stages: Vec<vk::PipelineShaderStageCreateInfo> = stage_specs
            .iter()
            .zip(entry_cstrings.iter())
            .map(|((flags, module, _), name)| shader_stage(*flags, *module, name.as_c_str()))
            .collect();

        // Groups: raygen (0), primary miss (1), one hit group per registry entry (its
        // index = its SBT class; class c -> group 2+c -> handle(2+c)), shadow miss LAST.
        let mut groups = vec![general_group(0), general_group(1)];
        for (chit, any_hit) in &hit_group_stages {
            groups.push(match any_hit {
                Some(a) => hit_group_with_any_hit(*chit, *a),
                None => hit_group(*chit),
            });
        }
        let shadow_miss_group = groups.len() as u32; // = 2 + hit_groups.len()
        groups.push(general_group(2)); // shadow miss (miss index 1)
        let group_count = groups.len() as u32;
        // The hair hit group's SBT handle index, for the reserved hair record below.
        let hair_group = hit_groups
            .iter()
            .position(|g| g.label == "hair")
            .map_or(2u32, |i| 2 + i as u32);
        // Max valid SBT class (registry index); out-of-range material classes fall back.
        let max_class = (hit_groups.len() as u32).saturating_sub(1);

        // --- Descriptor set layout (set 1: output + camera) --------------------
        // TLAS is NOT here — it comes from the scene bind group (set 0). raygen
        // writes the output buffer; raygen reads the camera.
        #[cfg_attr(not(feature = "dlss"), allow(unused_mut))]
        let mut bindings = vec![
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_OUTPUT)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_CAMERA)
                // The camera is a GPU buffer filled by the `rt_camera` compute pass (or a
                // CPU fallback `write_buffer`) each frame. Kept `*_DYNAMIC` — bound at a
                // constant offset 0 — so replacing the old ring needed no layout change.
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
        #[cfg(feature = "dlss")]
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
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: well-formed create info; device live.
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&dsl_info, None) }.ok()?;

        // Pipeline layout: [scene (set 0), rt-private (set 1), columns (set 2)].
        let set_layouts = [scene_layout, descriptor_set_layout, columns_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: well-formed; device live; the scene/columns layouts outlive this
        // pipeline (owned by wgpu's bind-group-layout cache).
        let pipeline_layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
            Ok(l) => l,
            Err(e) => {
                bevy_log::error!("rt_pipeline: create_pipeline_layout failed: {e:?}");
                return None;
            }
        };

        // --- Ray-tracing pipeline ---------------------------------------------
        // Opt into NV cluster acceleration structures so `@builtin(cluster_id)`
        // (ClusterIDNV) is valid in the hit shaders. ash doesn't register this
        // struct as a RayTracingPipelineCreateInfoKHR extender (no typed
        // `push_next`), so chain it via raw `p_next`. `cluster_info` must outlive
        // the create call below.
        let cluster_info =
            vk::RayTracingPipelineClusterAccelerationStructureCreateInfoNV::default()
                .allow_cluster_acceleration_structure(true);
        // Opt into opacity micromaps. Unlike ray queries (which honor OMM straight
        // from the AS), a ray-tracing *pipeline* ignores opacity micromaps entirely
        // unless created with this flag — the driver invokes the any-hit shader on
        // every micro-triangle as if no OMM were present. Gate on the extension so
        // pipeline creation stays valid where OMM is unsupported.
        let pipeline_flags = if crate::gpu::extension::opacity_micromap_available() {
            vk::PipelineCreateFlags::RAY_TRACING_OPACITY_MICROMAP_EXT
        } else {
            vk::PipelineCreateFlags::empty()
        };
        let mut pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .flags(pipeline_flags)
            .stages(&stages)
            .groups(&groups)
            // Depth 2: raygen's hit object executes the closest-hit (1), which
            // traces a NEE shadow ray (2). Shadow rays skip the closest-hit, so the
            // chain bottoms out there.
            .max_pipeline_ray_recursion_depth(2)
            .layout(pipeline_layout);
        pipeline_info.p_next =
            (&cluster_info as *const vk::RayTracingPipelineClusterAccelerationStructureCreateInfoNV)
                .cast();
        // SAFETY: stages/groups reference live modules; layout live.
        let pipeline = match unsafe {
            rt.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        } {
            Ok(p) => p.into_iter().next()?,
            Err(e) => {
                bevy_log::error!("rt_pipeline: vkCreateRayTracingPipelinesKHR failed: {:?}", e.1);
                return None;
            }
        };

        // The set-1 descriptor set + camera UBO + env binding are per-view, built
        // lazily in `create_view_bindings` (one per `SolariCamera`).

        // --- SBT: raygen(1) + miss(1) + hit(ONE RECORD PER MATERIAL) ----------
        // Three regions, each base-aligned. The hit region holds one record per
        // material slot; an instance's `instance_contribution_to_hit_group_index`
        // = its material slot selects its record. Each HIT record is
        // [shader group handle | shader-record data]; the data slot holds the
        // material id (= record index), which `chit_opaque`'s `var<shader_record>`
        // reads as the canonical material binding (uniform per record → uniform
        // per warp after SER). Distinct per-material records also give SER a
        // per-material reorder key and a slot for future per-class handles.
        // raygen, primary miss, N hit groups (registry), shadow miss — computed above.
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
            pipeline_layout,
            descriptor_set_layout,
            sbt,
            raygen_region,
            miss_region,
            hit_region,
            callable_region,
            record_capacity,
            material_classes: material_classes.to_vec(),
            modules,
        };
        Some(out)
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
        // non-opaque class arriving there forces a rebuild. (The old exact
        // Vec compare rebuilt the whole RT pipeline — a ~2 s driver compile —
        // every time ANY material streamed in, making the headroom dead code.)
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
        env_map_view: vk::ImageView,
        env_map_image: Option<vk::Image>,
    ) -> Option<RtViewBindings> {
        // One pool per view, sized for exactly this view's single set-1 set.
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1 + gbuffers.len() as u32), // output + DLSS guides
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
        // DLSS guide descriptors built outside `writes` so the per-binding infos
        // outlive `update_descriptor_sets` (empty when the feature is off).
        #[cfg(feature = "dlss")]
        let gbuffer_infos: Vec<[vk::DescriptorBufferInfo; 1]> = gbuffers
            .iter()
            .map(|&(buf, size)| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(buf)
                    .offset(0)
                    .range(size)]
            })
            .collect();
        #[cfg_attr(not(feature = "dlss"), allow(unused_mut))]
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
        ];
        #[cfg(feature = "dlss")]
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

        Some(RtViewBindings {
            device: self.device.clone(),
            descriptor_pool,
            descriptor_set,
            geometry,
            env_map_sampler,
            env_map_image,
            output_buffer,
            env_map_view,
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
        // SAFETY: the pool + camera buffer were created for this view and are
        // unused at teardown (the dispatch drains the GPU before rebuilding, and
        // the render world is otherwise idle at shutdown). Destroying the pool
        // frees its descriptor set.
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
        // SAFETY: all handles were created by this resource and are unused at
        // teardown (render world shutting down). The set-1 layout outlives the
        // per-view pools/sets allocated from it (destroying a layout with live
        // sets is legal), and those sets are dropped with their RtViewBindings.
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            for m in &self.modules {
                self.device.destroy_shader_module(*m, None);
            }
            self.device.destroy_buffer(self.sbt.buffer, None);
            self.device.free_memory(self.sbt.memory, None);
        }
    }
}

/// naga capabilities the RT shaders need — RT (query/pipeline/vertex-return) plus
/// the material texture/sampler binding arrays (indexed non-uniformly by material
/// id) the imported scene bindings carry. Used by BOTH the composer and the
/// post-compose validator, so they can't drift.
fn rt_capabilities() -> naga::valid::Capabilities {
    naga::valid::Capabilities::RAY_QUERY
        | naga::valid::Capabilities::RAY_HIT_VERTEX_POSITION
        | naga::valid::Capabilities::RAY_TRACING_PIPELINE
        | naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY
        | naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING
        // 64-bit ints for buffer-device-address arithmetic in the bindless
        // geometry path (`physical_load<T>(addr: u64)`).
        | naga::valid::Capabilities::SHADER_INT64
        // f64 for planet-scale domain math in user composable modules
        // (SHADER_F64 is a required solari device feature post-transform_f64).
        | naga::valid::Capabilities::FLOAT64
        // f16 pack/unpack builtins (planet erosion maps decode 4×f16 texels).
        | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
}

/// Build a naga_oil composer pre-loaded with the built-in importable modules the RT
/// shaders may `#import`, then any `extra_modules` (`(file_path, source)`, registered
/// in slice order so later entries may import earlier ones and the built-ins).
/// `None` when a module fails to compose (logged).
fn rt_composer(
    extra_modules: &[(&'static str, &'static str)],
) -> Option<naga_oil::compose::Composer> {
    use naga_oil::compose::{ComposableModuleDescriptor, Composer};

    // Compose via naga_oil so the RT shaders can `#import` solari's scene-binding
    // / BRDF / sampling modules (raw `naga::parse_str` can't resolve `#import`).
    // Register the importable modules (solari + the self-contained bevy_render
    // helpers); naga_oil pulls in only what each shader actually imports, roughly
    // leaf-first so each module's deps are present when it's added.
    // Composer validates the composed module, so it needs the same RT
    // capabilities as the spv backend (the modules use `acceleration_structure`
    // / ray-query / SER). `with_capabilities` purges modules, so set it first.
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
    // add-time, so every dependency must already be registered.
    register!("../../../bevy_render/src/maths.wgsl"); // bevy_render::maths (leaf)
    register!("../../../bevy_render/src/utils.wgsl"); // bevy_render::utils (leaf)
    register!("../render/rt_pipeline/rt_payload.wgsl"); // bevy_solari::rt_payload (leaf)
    register!("../render/atmosphere.wgsl"); // bevy_solari::atmosphere (leaf)
    register!("../instance/instance_mask.wgsl"); // bevy_solari::instance_mask (leaf)
    register!("../bindings/pbr.wgsl"); // -> maths
    register!("../bindings/cluster_bindings.wgsl"); // -> utils
    register!("../bindings/raytracing_scene_bindings.wgsl"); // -> pbr, atmosphere, utils
    register!("../bindings/sampling.wgsl"); // -> pbr, scene_bindings, maths
    register!("../bindings/brdf.wgsl"); // -> pbr, sampling, scene_bindings, maths
    register!("../hair/hair.wgsl"); // bevy_solari::hair (Chiang fiber BSDF) -> pbr
    // Downstream modules (a hit group's `composable_modules`), after the built-ins
    // so they can import them.
    for (path, source) in extra_modules {
        register!(*path, *source);
    }
    Some(composer)
}

/// Registration-time check that a hit group's extra modules compose against the
/// built-in set — surfaces "module X won't parse / imports something unregistered"
/// at `register_solari_chit` time instead of as an opaque whole-pipeline build
/// failure. Composition errors inside are logged by `rt_composer` itself.
fn validate_composable_modules(
    modules: &[(&'static str, &'static str)],
) -> Result<(), &'static str> {
    match rt_composer(modules) {
        Some(_) => Ok(()),
        None => Err("see preceding rt_pipeline compose error"),
    }
}

/// Compile a WGSL ray-tracing-stage shader to SPIR-V (1.4, RT capabilities) via
/// the re-exported naga, with the group's extra composable modules (if any)
/// available for `#import`. Returns `None` on parse/validate/emit failure (logged).
fn compile_rt_wgsl(
    source: &str,
    file_path: &str,
    extra_modules: &[(&'static str, &'static str)],
) -> Option<Vec<u32>> {
    use naga_oil::compose::{NagaModuleDescriptor, ShaderDefValue};

    let mut composer = rt_composer(extra_modules)?;

    // Shader-def axes for the RT shaders. This is the "pipeline key": each def is a
    // compile-out feature axis the raygen/chits can `#ifdef` on. Keep the axes few
    // and orthogonal (debug views ride a runtime uniform, not a def, to avoid a
    // variant explosion). `SOLARI_DLSS` is compile-time (tied to the cargo feature):
    // when set, the trace emits the ray-reconstruction guide G-buffer. A future
    // `SOLARI_RESTIR` axis would slot in here the same way.
    #[allow(unused_mut)]
    let mut shader_defs: std::collections::HashMap<String, ShaderDefValue> = [(
        // The scene-columns bind-group index the scene bindings are written with.
        "SOLARI_SCENE_COLUMNS_GROUP".to_string(),
        ShaderDefValue::UInt(2),
    )]
    .into_iter()
    .collect();
    #[cfg(feature = "dlss")]
    shader_defs.insert("SOLARI_DLSS".to_string(), ShaderDefValue::Bool(true));
    // Compile in the `shader_clock()` reads only when the device enabled
    // `VK_KHR_shader_clock`; otherwise the cost-heatmap path compiles out.
    if crate::gpu::extension::shader_clock_available() {
        shader_defs.insert("SOLARI_SHADER_CLOCK".to_string(), ShaderDefValue::Bool(true));
    }

    let module = match composer.make_naga_module(NagaModuleDescriptor {
        source,
        file_path,
        shader_defs,
        ..Default::default()
    }) {
        Ok(m) => m,
        Err(e) => {
            bevy_log::error!("rt_pipeline: compose {file_path}: {e:?}");
            return None;
        }
    };
    let info = match naga::valid::Validator::new(naga::valid::ValidationFlags::all(), rt_capabilities())
        .validate(&module)
    {
        Ok(i) => i,
        Err(e) => {
            bevy_log::error!("rt_pipeline: WGSL validation failed: {e:?}");
            return None;
        }
    };
    let mut options = naga::back::spv::Options::default();
    options.lang_version = (1, 4);
    // The scene `textures`/`samplers` are unsized `binding_array`s in WGSL. wgpu's
    // own pipeline compile bakes a FIXED descriptor count into the SPIR-V (the
    // device doesn't enable `runtimeDescriptorArray`, so an `OpTypeRuntimeArray`
    // descriptor variable is invalid). Mirror that: override @group(0) bindings
    // 7 (textures) and 8 (samplers) to the layout's `MAX_TEXTURE_COUNT`. All other
    // resources fall back to their own group/binding (`fake_missing_bindings`).
    options.fake_missing_bindings = true;
    for binding in [7u32, 8u32] {
        options.binding_map.insert(
            naga::ResourceBinding { group: 0, binding },
            naga::back::spv::BindingInfo {
                descriptor_set: 0,
                binding,
                binding_array_size: Some(crate::bindings::MAX_TEXTURE_COUNT.get()),
            },
        );
    }
    match naga::back::spv::write_vec(&module, &info, &options, None) {
        Ok(spv) => Some(spv),
        Err(e) => {
            bevy_log::error!("rt_pipeline: SPIR-V emit failed: {e:?}");
            None
        }
    }
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
