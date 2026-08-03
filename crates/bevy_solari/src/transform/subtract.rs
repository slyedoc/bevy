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

#![allow(unsafe_code)]

use ash::vk;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{renderer::RenderContext, Extract};
use bytemuck::{Pod, Zeroable};
use wgpu::hal::api::Vulkan as VkApi;

use crate::ecs_gpu::GpuSlot;
use crate::gpu::allocator::Allocator;
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
use crate::render::SolariCamera;

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

/// Push params shared with `transform_subtract.slang::SubtractParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct SubtractParams {
    count: u32,
    camera_slot: u32,
    node_count: u32,
    origin_valid: u32,
    groups_x: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Render-world resource: the subtract heap kernel + its slots.
#[derive(Resource)]
pub struct TransformSubtract {
    count: u32,
    /// The walk wrote `world_abs` and the subtract hasn't consumed it yet. Retained
    /// until the dispatch *actually* runs (an idle scene must not drop it — that
    /// would leave `world_rel` stale/zero forever). Same retain-until-consumed rule
    /// as the propagate's `needs_full_rebuild`.
    dirty: bool,
    groups: (u32, u32, u32),
    params: SubtractParams,
    kernel: HeapKernel,
    slots: KernelSlots,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for TransformSubtract {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for TransformSubtract {}
unsafe impl Sync for TransformSubtract {}

/// `RenderStartup` (after `SolariSetup`): compile the subtract kernel — a
/// layout-free heap pipeline ([`HeapKernel`]), Slang from source.
pub fn init_transform_subtract(
    mut commands: Commands,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let Some(kernel) = HeapKernel::new(
        &seam,
        "transform_subtract.slang",
        include_str!("transform_subtract.slang"),
        "subtract",
        &[],
        &[],
        "transform_subtract",
        size_of::<SubtractParams>() as u32,
    ) else {
        return;
    };
    let slots = KernelSlots::new(&seam, 3);
    commands.insert_resource(TransformSubtract {
        count: 0,
        dirty: false,
        groups: (0, 0, 0),
        params: SubtractParams::default(),
        kernel,
        slots,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

/// `Render::Prepare`: set the node count + origin, and whether the walk dirtied `world_abs`.
pub fn prepare_transform_subtract(
    mut subtract: Option<ResMut<TransformSubtract>>,
    propagate: Option<Res<TransformPropagate>>,
    origin: Option<Res<SolariOriginSlot>>,
) {
    let (Some(subtract), Some(propagate)) = (subtract.as_deref_mut(), propagate) else {
        return;
    };
    let (slot, valid) = origin.map(|o| (o.slot, o.valid)).unwrap_or((0, false));
    subtract.count = propagate.node_count();
    subtract.dirty |= propagate.world_dirty();
    subtract.groups = crate::ecs_gpu::linear_dispatch(subtract.count.div_ceil(WORKGROUP_SIZE));
    subtract.params = SubtractParams {
        count: subtract.count,
        camera_slot: slot,
        node_count: propagate.node_count(),
        origin_valid: valid as u32,
        groups_x: subtract.groups.0,
        ..Default::default()
    };
}

/// `RenderGraph` (`Propagate`, between the walk and the gather): subtract the origin,
/// producing the origin-relative `world_rel`. Skipped on an idle frame (nothing walked).
/// A raw heap dispatch: buffer slots rewritten per dispatch, params + slot array in
/// push data.
pub fn dispatch_transform_subtract(
    subtract: Option<ResMut<TransformSubtract>>,
    seam: Option<Res<BindingSeam>>,
    propagate: Option<Res<TransformPropagate>>,
    mut ctx: RenderContext,
) {
    let (Some(mut subtract), Some(seam), Some(propagate)) = (subtract, seam, propagate) else {
        return;
    };
    if !subtract.dirty || subtract.count == 0 {
        return;
    }
    subtract.dirty = false;
    let subtract = subtract.into_inner();
    let blob = subtract.kernel.push_blob(
        "transform_subtract",
        bytemuck::bytes_of(&subtract.params),
        &[
            (
                "world_abs_linear",
                subtract.slots.buffer(&seam, 0, propagate.world_abs_linear()),
            ),
            ("world_abs_t", subtract.slots.buffer(&seam, 1, propagate.world_abs_t())),
            ("world_rel", subtract.slots.buffer(&seam, 2, propagate.current_world())),
        ],
    );
    let (gx, gy, gz) = subtract.groups;
    let encoder = ctx.command_encoder();
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // barriers bracket this dispatch against the surrounding wgpu compute
    // passes (raw dispatches are invisible to wgpu's tracking).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &subtract.raw_device;
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)];
            let dep = vk::DependencyInfo::default().memory_barriers(&barrier);
            // The walk's world_abs writes -> our reads.
            dev.cmd_pipeline_barrier2(cb, &dep);
            seam.bind_heaps(cb);
            seam.push_data(cb, &blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, subtract.kernel.pipeline);
            dev.cmd_dispatch(cb, gx, gy, gz);
            // Our world_rel writes -> the gather / light-resolve reads.
            dev.cmd_pipeline_barrier2(cb, &dep);
        });
    }
}
