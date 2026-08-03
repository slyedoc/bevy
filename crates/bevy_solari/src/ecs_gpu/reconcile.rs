//! GPU instance reconcile pass — folds the CPU's absolute-state change journal
//! ([`crate::instance::RtJournal`]) into the per-instance GPU columns. One thread
//! per journal record writes **all** of a slot's columns from the single record,
//! so a reused slot is re-initialized atomically (no partial/stale column).
//! See `reconcile.slang`.
//!
//! The pass is self-contained: it owns its heap kernel and slots. The reconcile
//! is the sole writer of the bind-only instance columns (journal
//! `UPSERT`/`REMOVE` is GPU-reconcile-authoritative).

#![allow(unsafe_code)]

use ash::vk;
use bevy_app::{App, Plugin};
use bevy_ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Res, ResMut},
};
use bevy_render::{renderer::RenderContext, Render, RenderApp, RenderStartup, RenderSystems};
use bytemuck::{Pod, Zeroable};
use wgpu::hal::api::Vulkan as VkApi;

use crate::ecs_gpu::GpuColumn;
use crate::gpu::allocator::Allocator;
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
use crate::instance::{
    GeometryIdColumn, GroupBaseColumn, LodInputColumn, NodeSlotColumn, PartitionColumn, RtJournal,
};

const WORKGROUP_SIZE: u32 = 64;

/// Push params shared with `reconcile.slang::ReconcileParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct ReconcileParams {
    count: u32,
    groups_x: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Render-world resource: the reconcile heap kernel + its slots.
#[derive(Resource)]
pub struct RtReconcile {
    /// Journal record count this frame (dispatch bound).
    count: u32,
    groups: (u32, u32, u32),
    params: ReconcileParams,
    kernel: HeapKernel,
    slots: KernelSlots,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for RtReconcile {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for RtReconcile {}
unsafe impl Sync for RtReconcile {}

/// `RenderStartup`: compile the reconcile kernel + insert [`RtReconcile`].
/// No-op when the journal is absent (non-solari device → reconcile guards on it).
pub fn init_rt_reconcile(
    mut commands: Commands,
    journal: Option<Res<RtJournal>>,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(_journal), Some(seam), Some(allocator)) = (journal, seam, allocator) else {
        return;
    };
    let Some(kernel) = HeapKernel::new(
        &seam,
        "reconcile.slang",
        include_str!("reconcile.slang"),
        "reconcile_apply",
        &[],
        &[],
        "rt_reconcile",
        size_of::<ReconcileParams>() as u32,
    ) else {
        return;
    };
    let slots = KernelSlots::new(&seam, 6);
    commands.insert_resource(RtReconcile {
        count: 0,
        groups: (0, 0, 0),
        params: ReconcileParams::default(),
        kernel,
        slots,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

/// `Render::Prepare`: publish this frame's journal record count.
pub fn prepare_rt_reconcile(
    reconcile: Option<ResMut<RtReconcile>>,
    journal: Option<Res<RtJournal>>,
) {
    let (Some(mut reconcile), Some(journal)) = (reconcile, journal) else {
        return;
    };
    reconcile.count = journal.count;
    reconcile.groups = crate::ecs_gpu::linear_dispatch(journal.count.div_ceil(WORKGROUP_SIZE));
    reconcile.params = ReconcileParams {
        count: journal.count,
        groups_x: reconcile.groups.0,
        _pad0: 0,
        _pad1: 0,
    };
}

/// `RenderGraph` (`Propagate`, before the gather): apply this frame's journal to
/// the columns. A raw heap dispatch — the journal ring and every column buffer
/// are slot-indexed, so descriptors cover the whole (stable-address sparse)
/// buffers. Clears the journal's pending records (`mark_folded`) after the
/// dispatch records; the records stay live on frames where the kernel is absent.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rt_reconcile(
    reconcile: Option<Res<RtReconcile>>,
    journal: Option<ResMut<RtJournal>>,
    seam: Option<Res<BindingSeam>>,
    node_slots: Option<Res<GpuColumn<NodeSlotColumn>>>,
    geometry_ids: Option<Res<GpuColumn<GeometryIdColumn>>>,
    group_bases: Option<Res<GpuColumn<GroupBaseColumn>>>,
    lod_inputs: Option<Res<GpuColumn<LodInputColumn>>>,
    partition_hints: Option<Res<GpuColumn<PartitionColumn>>>,
    mut ctx: RenderContext,
) {
    let (
        Some(reconcile),
        Some(mut journal),
        Some(seam),
        Some(node_slots),
        Some(geometry_ids),
        Some(group_bases),
        Some(lod_inputs),
        Some(partition_hints),
    ) = (
        reconcile,
        journal,
        seam,
        node_slots,
        geometry_ids,
        group_bases,
        lod_inputs,
        partition_hints,
    )
    else {
        return;
    };
    if reconcile.count == 0 {
        return;
    }
    let blob = reconcile.kernel.push_blob(
        "rt_reconcile",
        bytemuck::bytes_of(&reconcile.params),
        &[
            ("journal", reconcile.slots.buffer(&seam, 0, journal.buffer.buffer())),
            ("node_slots", reconcile.slots.buffer(&seam, 1, node_slots.buffer())),
            ("geometry_ids", reconcile.slots.buffer(&seam, 2, geometry_ids.buffer())),
            ("group_bases", reconcile.slots.buffer(&seam, 3, group_bases.buffer())),
            ("lod_inputs", reconcile.slots.buffer(&seam, 4, lod_inputs.buffer())),
            ("partition_hints", reconcile.slots.buffer(&seam, 5, partition_hints.buffer())),
        ],
    );
    let (gx, gy, gz) = reconcile.groups;
    let encoder = ctx.command_encoder();
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // barriers bracket this dispatch against the surrounding wgpu compute
    // passes (raw dispatches are invisible to wgpu's tracking).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &reconcile.raw_device;
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)];
            let dep = vk::DependencyInfo::default().memory_barriers(&barrier);
            // The Scatter set's column zero-clears/writes -> our column writes.
            dev.cmd_pipeline_barrier2(cb, &dep);
            seam.bind_heaps(cb);
            seam.push_data(cb, &blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, reconcile.kernel.pipeline);
            dev.cmd_dispatch(cb, gx, gy, gz);
            // Our column writes -> the gather's node_slot read + downstream.
            dev.cmd_pipeline_barrier2(cb, &dep);
        });
    }
    // Folded: the GPU consumed the journal from its buffer this frame, so the CPU
    // staging can be dropped. Until this runs, the records stay live and are
    // re-uploaded next frame.
    journal.mark_folded();
}

/// Wires the reconcile pass (kernel + the prepare/dispatch systems).
pub struct ReconcilePlugin;

impl Plugin for ReconcilePlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            // After `init_rt_journal` (which is itself after `SolariSetup`/the
            // allocator): the reconcile only initializes when the journal exists, so
            // it must observe the journal already inserted this startup.
            .add_systems(
                RenderStartup,
                init_rt_reconcile.after(crate::instance::init_rt_journal),
            )
            .add_systems(
                Render,
                // MUST run after the journal upload — it sets `journal.count`, which
                // this reads for the dispatch bound + params. Without the order, a
                // stale (0) count means the reconcile never dispatches.
                prepare_rt_reconcile
                    .in_set(RenderSystems::PrepareResources)
                    .after(crate::instance::upload_rt_journal),
            )
            // Dispatch in `Propagate` (after the whole `Scatter` set): `GpuColumn`'s
            // zero-clear of newly-committed pages runs in `Scatter`'s `dispatch_column`,
            // so writing the columns there would race the clear on a growth/regen frame.
            // Ordered before the gather, which reads `node_slot`.
            .add_systems(
                bevy_render::renderer::RenderGraph,
                dispatch_rt_reconcile
                    .in_set(crate::SolariClusterSystems::Propagate)
                    .before(crate::transform::dispatch_transform_gather),
            );
    }
}
