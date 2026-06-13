//! Transform gather — the bridge that makes GPU propagation drive the render.
//!
//! Copies each RT instance's GPU-propagated world transform into the instance
//! `TransformColumn` buffer that every RT consumer already reads (PTLAS fill,
//! blas sharing, the raytracing scene), indexed by `node_slot[i]`:
//! `transforms[i] = world[node_slot[i]]`. No consumer changes — they keep
//! indexing transforms by instance slot. Runs after the propagation pass and
//! also shifts current → previous (motion vectors / ReSTIR temporal, and the
//! PTLAS-fill move detection). The instance world transform is produced entirely
//! GPU-side now — there is no CPU instance-transform path.

use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        ComputePassDescriptor, PipelineCache, ShaderStages, ShaderType, UniformBuffer,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
};
use bytemuck::{Pod, Zeroable};

use crate::ecs_gpu::GpuColumn;
use crate::instance::{InstanceManager, NodeSlotColumn, TransformColumn};
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;

use super::propagate::TransformPropagate;

const WORKGROUP_SIZE: u32 = 64;

/// Uniform shared with `transform_gather.wgsl::GatherParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
struct GatherParams {
    instance_count: u32,
    node_count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Render-world resource: the gather pipeline + its bind group.
#[derive(Resource)]
pub struct TransformGather {
    instance_count: u32,
    params: UniformBuffer<GatherParams>,
    bind_group: Option<BindGroup>,
}

/// The gather bind-group layout. Owned by
/// [`SolariResourceManager`](crate::resource_manager::SolariResourceManager).
pub fn transform_gather_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "transform_gather",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 node_slot
                storage_buffer_read_only_sized(false, None), // 1 world
                storage_buffer_sized(false, None),           // 2 transforms current (rw)
                storage_buffer_sized(false, None),           // 3 transforms previous (rw)
                uniform_buffer::<GatherParams>(false),       // 4 params
            ),
        ),
    )
}

/// `RenderStartup`: the gather pass owns only its params buffer + bind group; the
/// layout lives in `SolariResourceManager`, the pipeline id in `SolariPipelines`.
pub fn init_transform_gather(mut commands: Commands) {
    let mut params = UniformBuffer::<GatherParams>::default();
    params.set_label(Some("transform_gather"));

    commands.insert_resource(TransformGather {
        instance_count: 0,
        params,
        bind_group: None,
    });
}

/// `Render::Prepare`: set `instance_count` (instance high-water) + `node_count`
/// (world coverage, for the out-of-range guard).
pub fn prepare_transform_gather(
    mut gather: Option<ResMut<TransformGather>>,
    instances: Option<Res<InstanceManager>>,
    propagate: Option<Res<TransformPropagate>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(gather), Some(instances), Some(propagate)) =
        (gather.as_deref_mut(), instances, propagate)
    else {
        return;
    };
    gather.instance_count = instances.slot_high_water();
    *gather.params.get_mut() = GatherParams {
        instance_count: gather.instance_count,
        node_count: propagate.node_count(),
        ..Default::default()
    };
    gather.params.write_buffer(&render_device, &render_queue);
}

/// `Render::PrepareBindGroups`: build the gather bind group **once**. Every buffer
/// it binds — `world`, the `TransformColumn` (current + previous) and
/// `NodeSlotColumn` sparse stores — has a stable handle across growth, so once
/// built it stays valid; no per-frame rebuild needed.
pub fn prepare_transform_gather_bind_group(
    mut gather: Option<ResMut<TransformGather>>,
    resource_manager: Option<Res<SolariResourceManager>>,
    node_slots: Option<Res<GpuColumn<NodeSlotColumn>>>,
    transforms: Option<Res<GpuColumn<TransformColumn>>>,
    propagate: Option<Res<TransformPropagate>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let (Some(gather), Some(resource_manager), Some(node_slots), Some(transforms), Some(propagate)) =
        (gather.as_deref_mut(), resource_manager, node_slots, transforms, propagate)
    else {
        return;
    };
    if gather.bind_group.is_some() {
        return;
    }
    let Some(params) = gather.params.binding() else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&resource_manager.transform_gather);
    let previous = transforms
        .previous_buffer()
        .expect("TransformColumn keeps a previous-frame buffer (KEEP_PREVIOUS)");
    gather.bind_group = Some(render_device.create_bind_group(
        "transform_gather",
        &layout,
        &BindGroupEntries::sequential((
            node_slots.buffer().as_entire_binding(),
            propagate.current_world().as_entire_binding(),
            transforms.buffer().as_entire_binding(),
            previous.as_entire_binding(),
            params,
        )),
    ));
}

/// `RenderGraph` (`Propagate`, after the propagation pass): gather GPU-propagated
/// world transforms into the instance `TransformColumn`.
pub fn dispatch_transform_gather(
    gather: Option<Res<TransformGather>>,
    pipelines: Res<SolariPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let Some(gather) = gather else {
        return;
    };
    if gather.instance_count == 0 {
        return;
    }
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.transform_gather) else {
        return;
    };
    let Some(bind_group) = gather.bind_group.as_ref() else {
        return;
    };
    let groups = crate::ecs_gpu::linear_dispatch(gather.instance_count.div_ceil(WORKGROUP_SIZE));
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("transform_gather"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    let d = diagnostics.time_span(&mut pass, "transform_gather");
    pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    d.end(&mut pass);
}
