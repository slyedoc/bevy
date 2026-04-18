use bevy_ecs::resource::Resource;
use bevy_render::{
    render_resource::{
        BufferInitDescriptor, BufferUsages, PipelineLayoutDescriptor, RayTracingPipeline,
        RayTracingPipelineDescriptor, RayTracingPipelineStageDescriptor,
        RayTracingShaderGroupDescriptor, RayTracingShaderGroupType, ShaderBindingTableRegion,
        ShaderModuleDescriptor, ShaderSource,
    },
    renderer::RenderDevice,
};
use wgpu::util::DeviceExt as _;

use super::binder::RaytracingSceneBindings;
use bevy_render::render_resource::PipelineCache;

/// SBT handle size — 32 bytes on NVIDIA (Vulkan minimum).
const SHADER_GROUP_HANDLE_SIZE: u32 = 32;
/// SBT base alignment — 64 bytes on NVIDIA.
const SHADER_GROUP_BASE_ALIGNMENT: u64 = 64;

fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

/// Manages the ray tracing pipeline and shader binding table for shadow rays.
#[derive(Resource)]
pub struct ShadowRtPipeline {
    pub pipeline: RayTracingPipeline,
    /// Must stay alive as long as the pipeline is used — device addresses reference this buffer.
    pub _sbt_buffer: wgpu::Buffer,
    pub raygen_region: ShaderBindingTableRegion,
    pub miss_region: ShaderBindingTableRegion,
    pub hit_region: ShaderBindingTableRegion,
    pub callable_region: ShaderBindingTableRegion,
}

impl ShadowRtPipeline {
    pub fn new(
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        scene_bindings: &RaytracingSceneBindings,
        shader_source: &str,
    ) -> Self {
        let device = render_device.wgpu_device();

        let raw_scene_bgl =
            pipeline_cache.get_bind_group_layout(&scene_bindings.bind_group_layout);

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("shadow_rt_pipeline_layout"),
            bind_group_layouts: &[Some(&raw_scene_bgl)],
            immediate_size: 0,
        });

        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("shadow_rt_shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = device.create_ray_tracing_pipeline(&RayTracingPipelineDescriptor {
            label: Some("shadow_rt_pipeline"),
            layout: Some(&pipeline_layout),
            stages: &[
                RayTracingPipelineStageDescriptor {
                    module: &shader_module,
                    entry_point: Some("shadow_raygen"),
                    compilation_options: Default::default(),
                },
                RayTracingPipelineStageDescriptor {
                    module: &shader_module,
                    entry_point: Some("shadow_miss"),
                    compilation_options: Default::default(),
                },
                RayTracingPipelineStageDescriptor {
                    module: &shader_module,
                    entry_point: Some("shadow_any_hit"),
                    compilation_options: Default::default(),
                },
                RayTracingPipelineStageDescriptor {
                    module: &shader_module,
                    entry_point: Some("shadow_closest_hit"),
                    compilation_options: Default::default(),
                },
            ],
            groups: &[
                // Group 0: Raygen
                RayTracingShaderGroupDescriptor {
                    group_type: RayTracingShaderGroupType::General,
                    general_stage_index: Some(0),
                    closest_hit_stage_index: None,
                    any_hit_stage_index: None,
                    intersection_stage_index: None,
                },
                // Group 1: Miss
                RayTracingShaderGroupDescriptor {
                    group_type: RayTracingShaderGroupType::General,
                    general_stage_index: Some(1),
                    closest_hit_stage_index: None,
                    any_hit_stage_index: None,
                    intersection_stage_index: None,
                },
                // Group 2: Hit (any-hit + closest-hit)
                RayTracingShaderGroupDescriptor {
                    group_type: RayTracingShaderGroupType::TrianglesHitGroup,
                    general_stage_index: None,
                    closest_hit_stage_index: Some(3),
                    any_hit_stage_index: Some(2),
                    intersection_stage_index: None,
                },
            ],
            max_pipeline_ray_recursion_depth: 1,
            cache: None,
        });

        // Build the Shader Binding Table
        let handle_size = SHADER_GROUP_HANDLE_SIZE as u64;
        let group_count = 3u32; // raygen, miss, hit

        // Get raw shader group handles from the driver
        let handles = device.get_ray_tracing_shader_group_handles(&pipeline, 0, group_count);

        // Layout: [raygen | padding | miss | padding | hit | padding]
        // Each region is aligned to SHADER_GROUP_BASE_ALIGNMENT
        let raygen_size = align_up(handle_size, SHADER_GROUP_BASE_ALIGNMENT);
        let miss_size = align_up(handle_size, SHADER_GROUP_BASE_ALIGNMENT);
        let hit_size = align_up(handle_size, SHADER_GROUP_BASE_ALIGNMENT);
        let sbt_total_size = raygen_size + miss_size + hit_size;

        // Build SBT data in CPU memory
        let hs = handle_size as usize;
        let mut sbt_data = vec![0u8; sbt_total_size as usize];

        // Raygen handle at offset 0
        if handles.len() >= hs {
            sbt_data[..hs].copy_from_slice(&handles[..hs]);
        }
        // Miss handle at offset raygen_size
        let miss_offset = raygen_size as usize;
        if handles.len() >= hs * 2 {
            sbt_data[miss_offset..miss_offset + hs].copy_from_slice(&handles[hs..hs * 2]);
        }
        // Hit handle at offset raygen_size + miss_size
        let hit_offset = (raygen_size + miss_size) as usize;
        if handles.len() >= hs * 3 {
            sbt_data[hit_offset..hit_offset + hs].copy_from_slice(&handles[hs * 2..hs * 3]);
        }

        // Create SBT buffer with initial data
        let sbt_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("shadow_rt_sbt"),
            contents: &sbt_data,
            usage: BufferUsages::SHADER_BINDING_TABLE,
        });

        // Get the buffer device address
        let sbt_address = device.get_buffer_device_address(&sbt_buffer);

        let raygen_region = ShaderBindingTableRegion {
            device_address: sbt_address,
            stride: raygen_size,
            size: raygen_size,
        };
        let miss_region = ShaderBindingTableRegion {
            device_address: sbt_address + raygen_size,
            stride: miss_size,
            size: miss_size,
        };
        let hit_region = ShaderBindingTableRegion {
            device_address: sbt_address + raygen_size + miss_size,
            stride: hit_size,
            size: hit_size,
        };
        let callable_region = ShaderBindingTableRegion::default();

        Self {
            pipeline,
            _sbt_buffer: sbt_buffer,
            raygen_region,
            miss_region,
            hit_region,
            callable_region,
        }
    }
}
