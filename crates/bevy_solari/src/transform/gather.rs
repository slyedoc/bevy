//! Transform gather — the bridge that makes GPU propagation drive the render.
//!
//! Copies each RT instance's GPU-propagated world transform into the instance
//! `TransformColumn` buffer that every RT consumer already reads (PTLAS fill,
//! blas sharing, the raytracing scene), indexed by `node_slot[i]`:
//! `transforms[i] = world[node_slot[i]]`. Consumers keep indexing transforms by
//! instance slot. Runs after the propagation pass and also shifts
//! current → previous (motion vectors / ReSTIR temporal, and the PTLAS-fill
//! move detection). The instance world transform is produced entirely GPU-side;
//! there is no CPU instance-transform path.

#![allow(unsafe_code)]

use ash::vk;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::renderer::RenderContext;
use bytemuck::{Pod, Zeroable};
use wgpu::hal::api::Vulkan as VkApi;

use crate::ecs_gpu::GpuColumn;
use crate::gpu::allocator::Allocator;
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
use crate::instance::{InstanceManager, NodeSlotColumn, TransformColumn};

use super::propagate::TransformPropagate;

const WORKGROUP_SIZE: u32 = 64;

/// Push params shared with `transform_gather.slang::GatherParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct GatherParams {
    instance_count: u32,
    node_count: u32,
    groups_x: u32,
    _pad: u32,
}

/// Render-world resource: the gather heap kernel + its slots.
#[derive(Resource)]
pub struct TransformGather {
    instance_count: u32,
    groups: (u32, u32, u32),
    params: GatherParams,
    kernel: HeapKernel,
    slots: KernelSlots,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for TransformGather {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for TransformGather {}
unsafe impl Sync for TransformGather {}

/// `RenderStartup` (after `SolariSetup`): compile the gather kernel — a
/// layout-free heap pipeline ([`HeapKernel`]), Slang from source.
pub fn init_transform_gather(
    mut commands: Commands,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let Some(kernel) = HeapKernel::new(
        &seam,
        "transform_gather.slang",
        include_str!("transform_gather.slang"),
        "gather",
        &[],
        &[],
        "transform_gather",
        size_of::<GatherParams>() as u32,
    ) else {
        return;
    };
    let slots = KernelSlots::new(&seam, 4);
    commands.insert_resource(TransformGather {
        instance_count: 0,
        groups: (0, 0, 0),
        params: GatherParams::default(),
        kernel,
        slots,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

/// `Render::Prepare`: set `instance_count` (instance high-water) + `node_count`
/// (world coverage, for the out-of-range guard).
pub fn prepare_transform_gather(
    mut gather: Option<ResMut<TransformGather>>,
    instances: Option<Res<InstanceManager>>,
    propagate: Option<Res<TransformPropagate>>,
) {
    let (Some(gather), Some(instances), Some(propagate)) =
        (gather.as_deref_mut(), instances, propagate)
    else {
        return;
    };
    gather.instance_count = instances.slot_high_water();
    gather.groups =
        crate::ecs_gpu::linear_dispatch(gather.instance_count.div_ceil(WORKGROUP_SIZE));
    gather.params = GatherParams {
        instance_count: gather.instance_count,
        node_count: propagate.node_count(),
        groups_x: gather.groups.0,
        _pad: 0,
    };
}

/// `RenderGraph` (`Propagate`, after the propagation pass): gather GPU-propagated
/// world transforms into the instance `TransformColumn`. A raw heap dispatch:
/// buffer slots rewritten per dispatch, params + slot array in push data.
pub fn dispatch_transform_gather(
    gather: Option<Res<TransformGather>>,
    seam: Option<Res<BindingSeam>>,
    node_slots: Option<Res<GpuColumn<NodeSlotColumn>>>,
    transforms: Option<Res<GpuColumn<TransformColumn>>>,
    propagate: Option<Res<TransformPropagate>>,
    mut ctx: RenderContext,
) {
    let (Some(gather), Some(seam), Some(node_slots), Some(transforms), Some(propagate)) =
        (gather, seam, node_slots, transforms, propagate)
    else {
        return;
    };
    if gather.instance_count == 0 {
        return;
    }
    let previous = transforms
        .previous_buffer()
        .expect("TransformColumn keeps a previous-frame buffer (KEEP_PREVIOUS)");
    let blob = gather.kernel.push_blob(
        "transform_gather",
        bytemuck::bytes_of(&gather.params),
        &[
            ("node_slot", gather.slots.buffer(&seam, 0, node_slots.buffer())),
            ("world", gather.slots.buffer(&seam, 1, propagate.current_world())),
            ("transforms", gather.slots.buffer(&seam, 2, transforms.buffer())),
            ("previous", gather.slots.buffer(&seam, 3, previous)),
        ],
    );
    let (gx, gy, gz) = gather.groups;
    let encoder = ctx.command_encoder();
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // barriers bracket this dispatch against the surrounding wgpu compute
    // passes (raw dispatches are invisible to wgpu's tracking).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &gather.raw_device;
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)];
            let dep = vk::DependencyInfo::default().memory_barriers(&barrier);
            // Propagate/subtract writes -> our reads (and our transform writes).
            dev.cmd_pipeline_barrier2(cb, &dep);
            seam.bind_heaps(cb);
            seam.push_data(cb, &blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, gather.kernel.pipeline);
            dev.cmd_dispatch(cb, gx, gy, gz);
            // Our transform writes -> downstream compute reads (PTLAS fill,
            // blas sharing); the trace's own pre-barrier covers RT visibility.
            dev.cmd_pipeline_barrier2(cb, &dep);
        });
    }
}
