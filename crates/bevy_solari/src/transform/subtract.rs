//! Transform subtract — the floating-origin pass, fully GPU-driven.
//!
//! The ancestor walk ([`super::propagate`]) produces each node's **absolute** world
//! (f32 linear + native-f64 translation). This pass subtracts the **origin** — the
//! primary [`SolariCamera`]'s own absolute world, read straight off the GPU as
//! `world_abs_t[camera_slot]` — and writes the small **origin-relative** f32 world
//! (`world_rel`, `mat3x4`) every RT consumer reads. No CPU computes the origin: the
//! only CPU input is the camera's slot index ([`SolariOriginSlot`]).
//!
//! De-fusing this from the walk is the whole point: the origin moves every frame the camera
//! moves, but re-relativizing is a flat one-op-per-node kernel (no chain walk), so the camera
//! can be the live origin without forcing the expensive walk to re-run over static nodes.

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
    Extract,
};
use bytemuck::{Pod, Zeroable};

use crate::ecs_gpu::GpuSlot;
use crate::pipelines::SolariPipelines;
use crate::render::SolariCamera;
use crate::resource_manager::SolariResourceManager;

use super::graph::TransformGraph;
use super::propagate::TransformPropagate;

const WORKGROUP_SIZE: u32 = 64;

/// The primary [`SolariCamera`]'s transform-table slot — the floating origin. The subtract
/// shader reads `world_abs_t[slot]` as the f64 origin; `valid == false` (no camera slotted yet)
/// passes the absolute world through unchanged. This is the *only* CPU input to the origin;
/// the value itself is computed on the GPU.
#[derive(Resource, Default)]
pub struct SolariOriginSlot {
    pub slot: u32,
    pub valid: bool,
}

/// `ExtractSchedule`: point [`SolariOriginSlot`] at the primary `SolariCamera`'s transform-table
/// slot. A camera childed to a player/ship/patch resolves through the walk, so the origin follows
/// its true composed position with no CPU math — just its slot.
pub fn extract_origin_slot(
    mut origin: ResMut<SolariOriginSlot>,
    cameras: Extract<
        bevy_ecs::prelude::Query<&GpuSlot<TransformGraph>, bevy_ecs::prelude::With<SolariCamera>>,
    >,
) {
    match cameras.iter().next() {
        Some(slot) => {
            origin.slot = slot.index();
            origin.valid = true;
        }
        None => origin.valid = false,
    }
}

/// Uniform shared with `transform_subtract.wgsl::SubtractParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
struct SubtractParams {
    count: u32,
    camera_slot: u32,
    node_count: u32,
    origin_valid: u32,
}

/// Render-world resource: the subtract pipeline + its params/bind group.
#[derive(Resource)]
pub struct TransformSubtract {
    count: u32,
    /// The walk wrote `world_abs` and the subtract hasn't consumed it yet. Retained
    /// until the dispatch *actually* runs (a cold pipeline or missing bind group must
    /// not drop it — the scene could go idle and leave `world_rel` stale/zero forever).
    /// Same retain-until-consumed rule as the propagate's `needs_full_rebuild`.
    dirty: bool,
    params: UniformBuffer<SubtractParams>,
    bind_group: Option<BindGroup>,
}

/// The subtract bind-group layout. Owned by [`SolariResourceManager`].
pub fn transform_subtract_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "transform_subtract",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 world_abs_linear
                storage_buffer_read_only_sized(false, None), // 1 world_abs_t (array<f64>)
                storage_buffer_sized(false, None),           // 2 world_rel (rw)
                uniform_buffer::<SubtractParams>(false),     // 3 params
            ),
        ),
    )
}

/// `RenderStartup`: the subtract pass owns only its params buffer + bind group.
pub fn init_transform_subtract(mut commands: Commands) {
    let mut params = UniformBuffer::<SubtractParams>::default();
    params.set_label(Some("transform_subtract"));
    commands.insert_resource(TransformSubtract {
        count: 0,
        dirty: false,
        params,
        bind_group: None,
    });
}

/// `Render::Prepare`: set the node count + origin, and whether the walk dirtied `world_abs`.
pub fn prepare_transform_subtract(
    mut subtract: Option<ResMut<TransformSubtract>>,
    propagate: Option<Res<TransformPropagate>>,
    origin: Option<Res<SolariOriginSlot>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(subtract), Some(propagate)) = (subtract.as_deref_mut(), propagate) else {
        return;
    };
    let (slot, valid) = origin.map(|o| (o.slot, o.valid)).unwrap_or((0, false));
    subtract.count = propagate.node_count();
    subtract.dirty |= propagate.world_dirty();
    *subtract.params.get_mut() = SubtractParams {
        count: subtract.count,
        camera_slot: slot,
        node_count: propagate.node_count(),
        origin_valid: valid as u32,
    };
    subtract.params.write_buffer(&render_device, &render_queue);
}

/// `Render::PrepareBindGroups`: build the subtract bind group **once** — both world buffers
/// have stable sparse handles across growth, so it stays valid.
pub fn prepare_transform_subtract_bind_group(
    mut subtract: Option<ResMut<TransformSubtract>>,
    resource_manager: Option<Res<SolariResourceManager>>,
    propagate: Option<Res<TransformPropagate>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let (Some(subtract), Some(resource_manager), Some(propagate)) =
        (subtract.as_deref_mut(), resource_manager, propagate)
    else {
        return;
    };
    if subtract.bind_group.is_some() {
        return;
    }
    let Some(params) = subtract.params.binding() else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&resource_manager.transform_subtract);
    subtract.bind_group = Some(render_device.create_bind_group(
        "transform_subtract",
        &layout,
        &BindGroupEntries::sequential((
            propagate.world_abs_linear().as_entire_binding(),
            propagate.world_abs_t().as_entire_binding(),
            propagate.current_world().as_entire_binding(),
            params,
        )),
    ));
}

/// `RenderGraph` (`Propagate`, between the walk and the gather): subtract the origin,
/// producing the origin-relative `world_rel`. Skipped on an idle frame (nothing walked).
pub fn dispatch_transform_subtract(
    subtract: Option<ResMut<TransformSubtract>>,
    pipelines: Res<SolariPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let Some(mut subtract) = subtract else {
        return;
    };
    if !subtract.dirty || subtract.count == 0 {
        return;
    }
    // A cold pipeline / missing bind group bails WITHOUT clearing `dirty` — the
    // pending subtract is retained until it actually runs.
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.transform_subtract) else {
        return;
    };
    if subtract.bind_group.is_none() {
        return;
    }
    subtract.dirty = false;
    let subtract = subtract.into_inner();
    let bind_group = subtract.bind_group.as_ref().unwrap();
    let groups = crate::ecs_gpu::linear_dispatch(subtract.count.div_ceil(WORKGROUP_SIZE));
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("transform_subtract"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    let d = diagnostics.time_span(&mut pass, "transform_subtract");
    pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    d.end(&mut pass);
}
