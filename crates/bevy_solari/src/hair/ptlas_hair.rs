//! PTLAS injection for hair: a compute pass that appends hair instances to the
//! partitioned-AS WRITE record stream the cluster fill produces. Recorded from
//! inside [`crate::accel::ptlas::dispatch_ptlas`] (between the cluster
//! `fill_incremental` and `finalize` passes) so the single GPU record count
//! covers both.

use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        PipelineCache, ShaderStages, ShaderType, UniformBuffer,
    },
    renderer::{RenderDevice, RenderQueue},
};

use crate::accel::ptlas::Ptlas;
use crate::gpu::rt_pipeline::RtPipeline;
use crate::resource_manager::SolariResourceManager;
use crate::transform::TransformPropagate;

use super::HairInstances;

/// Uniform shared with `ptlas_hair_write.wgsl::HairWriteParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, ShaderType)]
pub struct HairWriteParams {
    pub hair_count: u32,
    pub hair_base: u32,
    /// SBT hit-record index hair instances route to ([`RtPipeline::hair_sbt_record`]),
    /// baked into each hair record's `instance_contribution_to_hit_group_index` so
    /// the trace reaches `chit_hair`. 0 until the RT pipeline exists (no trace yet).
    pub hair_sbt_record: u32,
    pub pad1: u32,
}

/// Render-world resource: the hair PTLAS-write params + bind group.
#[derive(Resource)]
pub struct HairPtlasWrite {
    pub params: UniformBuffer<HairWriteParams>,
    pub hair_count: u32,
    pub bind_group: Option<BindGroup>,
}

/// The hair PTLAS-write `@group(0)` layout. Owned by [`SolariResourceManager`].
pub fn ptlas_hair_write_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "ptlas_hair_write",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 hair_instances
                storage_buffer_sized(false, None),           // 1 write_count (rw atomic)
                storage_buffer_sized(false, None),           // 2 write_data (rw)
                uniform_buffer::<HairWriteParams>(false),    // 3 params
                storage_buffer_read_only_sized(false, None), // 4 world (transform table)
            ),
        ),
    )
}

/// `RenderStartup`: the hair-write params buffer + resource.
pub fn init_hair_ptlas_write(mut commands: Commands) {
    let mut params = UniformBuffer::<HairWriteParams>::default();
    params.set_label(Some("ptlas_hair_write"));
    commands.insert_resource(HairPtlasWrite {
        params,
        hair_count: 0,
        bind_group: None,
    });
}

/// `Render::Prepare`: set the hair-write params (count + PTLAS base). The base
/// is assigned by [`crate::accel::ptlas::prepare_ptlas_params`] onto
/// [`HairInstances`] before this runs.
pub fn prepare_hair_ptlas_write(
    mut write: ResMut<HairPtlasWrite>,
    instances: Option<Res<HairInstances>>,
    rt_pipeline: Option<Res<RtPipeline>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(instances) = instances else {
        write.hair_count = 0;
        return;
    };
    // The hair SBT record lives in the RT pipeline's SBT; 0 until it's built (hair
    // can't be traced before then anyway). Reads last frame's pipeline, which is
    // this frame's except on a capacity rebuild — and a rebuild frame skips the
    // trace, so a stale index is never consumed.
    let hair_sbt_record = rt_pipeline.map_or(0, |rt| rt.hair_sbt_record());
    write.hair_count = instances.count;
    *write.params.get_mut() = HairWriteParams {
        hair_count: instances.count,
        hair_base: instances.base,
        hair_sbt_record,
        pad1: 0,
    };
    write.params.write_buffer(&render_device, &render_queue);
}

/// `Render::PrepareBindGroups`: build the hair-write bind group from the PTLAS
/// record buffers + the hair instance buffer. Rebuilt each frame (the hair
/// instance buffer can be reallocated by `RawBufferVec` growth).
pub fn prepare_hair_ptlas_write_bind_group(
    mut write: ResMut<HairPtlasWrite>,
    resource_manager: Option<Res<SolariResourceManager>>,
    ptlas: Option<Res<Ptlas>>,
    instances: Option<Res<HairInstances>>,
    propagate: Option<Res<TransformPropagate>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let (Some(resource_manager), Some(ptlas), Some(instances), Some(propagate)) =
        (resource_manager, ptlas, instances, propagate)
    else {
        write.bind_group = None;
        return;
    };
    let (Some(params), Some(hair_buffer)) = (write.params.binding(), instances.buffer.buffer())
    else {
        write.bind_group = None;
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&resource_manager.ptlas_hair_write);
    write.bind_group = Some(render_device.create_bind_group(
        "ptlas_hair_write",
        &layout,
        &BindGroupEntries::sequential((
            hair_buffer.as_entire_binding(),
            ptlas.write_count.as_entire_binding(),
            ptlas.write_data.wgpu_buffer.as_entire_binding(),
            params,
            propagate.current_world().as_entire_binding(),
        )),
    ));
}
