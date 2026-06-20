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

/// Per-frame camera inputs the raygen shader reads — std140-compatible
/// (mat4 + vec4). `inverse_view_proj` reconstructs a world-space ray per pixel;
/// `camera_position` is the ray origin.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RtCamera {
    pub inverse_view_proj: [f32; 16],
    pub camera_position: [f32; 4],
    /// `frame_index` in `.x` (RNG seed for temporal variation); `.yzw` pad to a
    /// 16-byte std140 slot.
    pub frame: [u32; 4],
    /// `.x` = sky/environment brightness (raw cd/m²; 0 ⇒ no skybox, miss stays
    /// at the clear color in `.yzw`).
    pub sky: [f32; 4],
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
    camera: MappedBuffer,
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
}

// SAFETY: the host-visible camera mapping is written only from the single
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
        material_count: u32,
    ) -> Option<Self> {
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
        // raygen, miss, and THREE closest-hit programs (opaque / glass / hair) —
        // one per material class. Separate SBT programs are the multi-material
        // win: a hit pays only its own shader's register footprint.
        let raygen_mod = create_shader_module(
            &device,
            &compile_rt_wgsl(include_str!("../render/rt_pipeline/raygen.wgsl"), "raygen.wgsl")?,
        )?;
        let miss_mod = create_shader_module(
            &device,
            &compile_rt_wgsl(include_str!("../render/rt_pipeline/miss.wgsl"), "miss.wgsl")?,
        )?;
        let chit_opaque_mod = create_shader_module(
            &device,
            &compile_rt_wgsl(include_str!("../render/rt_pipeline/chit_opaque.wgsl"), "chit_opaque.wgsl")?,
        )?;
        let chit_glass_mod = create_shader_module(
            &device,
            &compile_rt_wgsl(include_str!("../render/rt_pipeline/chit_glass.wgsl"), "chit_glass.wgsl")?,
        )?;
        let chit_hair_mod = create_shader_module(
            &device,
            &compile_rt_wgsl(include_str!("../render/rt_pipeline/chit_hair.wgsl"), "chit_hair.wgsl")?,
        )?;
        let modules = vec![
            raygen_mod,
            miss_mod,
            chit_opaque_mod,
            chit_glass_mod,
            chit_hair_mod,
        ];

        // naga emits each entry point under its WGSL function name.
        let stages = [
            shader_stage(vk::ShaderStageFlags::RAYGEN_KHR, raygen_mod, c"raygen"),
            shader_stage(vk::ShaderStageFlags::MISS_KHR, miss_mod, c"miss_primary"),
            shader_stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, chit_opaque_mod, c"chit_opaque"),
            shader_stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, chit_glass_mod, c"chit_glass"),
            shader_stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, chit_hair_mod, c"chit_hair"),
        ];

        // Group 0 = raygen, 1 = miss (both general), 2/3/4 = the opaque/glass/hair
        // triangle hit groups — their order in the hit region IS the SBT offset
        // (HIT_GROUP_OPAQUE=0, _GLASS=1, _HAIR=2) routed in ptlas_fill /
        // ptlas_hair_write.
        let groups = [
            general_group(0),
            general_group(1),
            hit_group(2),
            hit_group(3),
            hit_group(4),
        ];

        // --- Descriptor set layout (set 1: output + camera) --------------------
        // TLAS is NOT here — it comes from the scene bind group (set 0). raygen
        // writes the output buffer; raygen reads the camera.
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_OUTPUT)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_CAMERA)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                // raygen unprojects; miss reads sky brightness/clear color.
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::MISS_KHR),
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
        ];
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
        let mut pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(1)
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
        const GROUP_COUNT: u32 = 5; // raygen, miss, opaque, glass, hair
        const HIT_RECORD_DATA: u64 = 4; // bytes of shader-record data (u32 material id)
        const RECORD_HEADROOM: u32 = 64; // absorb a little material growth post-build
        let record_capacity = material_count + RECORD_HEADROOM;
        let handle_stride = align_up(handle_size, handle_align);
        // Hit records carry the material-id data slot, so they're wider than a
        // bare handle.
        let hit_record_stride = align_up(handle_size + HIT_RECORD_DATA, handle_align);
        let raygen_offset = 0u64;
        let miss_offset = align_up(handle_stride, base_align);
        let hit_offset = align_up(miss_offset + handle_stride, base_align);
        let sbt_size = hit_offset + record_capacity as u64 * hit_record_stride;
        let sbt = alloc_mapped_buffer(
            allocator,
            sbt_size,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
        )?;

        // SAFETY: pipeline live; handle data sized to GROUP_COUNT * handle_size.
        let handles = unsafe {
            rt.get_ray_tracing_shader_group_handles(
                pipeline,
                0,
                GROUP_COUNT,
                (GROUP_COUNT as u64 * handle_size) as usize,
            )
        }
        .ok()?;
        let handle = |g: usize| &handles[g * handle_size as usize..(g + 1) * handle_size as usize];
        // raygen (group 0) + miss (group 1) handles.
        for &(g, off) in [(0usize, raygen_offset), (1usize, miss_offset)].iter() {
            // SAFETY: mapped covers sbt_size; off + handle_size within bounds.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    handle(g).as_ptr(),
                    sbt.mapped.add(off as usize),
                    handle_size as usize,
                );
            }
        }
        // One record per material slot: [opaque handle | material id]. Every
        // record uses the OPAQUE closest-hit (group 2) for now; glass/hair
        // per-material handles are a follow-up. The data slot = the record index =
        // the material id the instance routes to, read back via `var<shader_record>`.
        let opaque_handle = handle(2);
        for record in 0..record_capacity as u64 {
            let rec_off = hit_offset + record * hit_record_stride;
            // SAFETY: rec_off + handle_size + 4 within the record.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    opaque_handle.as_ptr(),
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
            .size(handle_stride);
        // Per-material records: stride steps one record; the instance's material
        // slot indexes them.
        let hit_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt.device_address + hit_offset)
            .stride(hit_record_stride)
            .size(record_capacity as u64 * hit_record_stride);
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

    /// Build the per-view set-1 resources (descriptor set + camera UBO) for one
    /// view: a fresh pool, a set allocated from the shared set-1 layout, a camera
    /// UBO, and the descriptor written ONCE (output buffer @0, camera @1, env cube
    /// @2, shared sampler @3). The set is never updated again — updating one while
    /// a prior frame's command buffer still binds it is illegal and device-losts;
    /// the camera *contents* change per frame via the mapping (`set_camera`), and
    /// the output buffer is stable (the dispatch rebuilds this whole component if
    /// the view's output buffer is reallocated). `env_map_image` is `Some` when the
    /// env cube is the storage atmosphere cube (transitioned around the trace).
    pub fn create_view_bindings(
        &self,
        allocator: &Allocator,
        output_buffer: vk::Buffer,
        output_size: u64,
        env_map_view: vk::ImageView,
        env_map_image: Option<vk::Image>,
    ) -> Option<RtViewBindings> {
        // One pool per view, sized for exactly this view's single set-1 set.
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
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

        let camera = alloc_mapped_buffer(
            allocator,
            size_of::<RtCamera>() as u64,
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
        let camera_info = [vk::DescriptorBufferInfo::default()
            .buffer(camera.buffer)
            .offset(0)
            .range(camera.size)];
        // `trace()` transitions the atmosphere cube to SHADER_READ_ONLY_OPTIMAL
        // around the dispatch (skybox/fallback are already in this layout), so the
        // descriptor always sees read-optimal.
        let env_image_info = [vk::DescriptorImageInfo::default()
            .image_view(env_map_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let env_sampler_info =
            [vk::DescriptorImageInfo::default().sampler(env_map_sampler)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_OUTPUT)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&output_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(BINDING_CAMERA)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
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
        ];
        // SAFETY: targets the freshly-allocated set; buffers + image/sampler live.
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        Some(RtViewBindings {
            device: self.device.clone(),
            descriptor_pool,
            descriptor_set,
            camera,
            env_map_sampler,
            env_map_image,
            output_buffer,
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
            let pre = vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
                        | vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                        | vk::AccessFlags::SHADER_READ,
                );
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
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
                &[],
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
    /// Upload this frame's camera inputs for this view (host-visible, coherent).
    pub fn set_camera(&self, camera: &RtCamera) {
        // SAFETY: `camera.mapped` is a valid HOST_VISIBLE|COHERENT mapping of at
        // least size_of::<RtCamera>() bytes; RtCamera is Pod.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytemuck::bytes_of(camera).as_ptr(),
                self.camera.mapped,
                size_of::<RtCamera>(),
            );
        }
    }

    /// The output `VkBuffer` baked into binding 0. The dispatch compares this to
    /// the view's current output buffer to detect a resize-driven reallocation.
    pub fn output_buffer(&self) -> vk::Buffer {
        self.output_buffer
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
            self.device.destroy_buffer(self.camera.buffer, None);
            self.device.free_memory(self.camera.memory, None);
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
}

/// Compile a WGSL ray-tracing-stage shader to SPIR-V (1.4, RT capabilities) via
/// the re-exported naga. Returns `None` on parse/validate/emit failure (logged).
fn compile_rt_wgsl(source: &str, file_path: &str) -> Option<Vec<u32>> {
    use naga_oil::compose::{
        ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue,
    };

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
        ($path:literal) => {
            if let Err(e) = composer.add_composable_module(ComposableModuleDescriptor {
                source: include_str!($path),
                file_path: $path,
                ..Default::default()
            }) {
                bevy_log::error!("rt_pipeline: compose register {}: {e:?}", $path);
                return None;
            }
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

    let shader_defs = [(
        // The scene-columns bind-group index the scene bindings are written with.
        "SOLARI_SCENE_COLUMNS_GROUP".to_string(),
        ShaderDefValue::UInt(2),
    )]
    .into_iter()
    .collect();

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
