//! PTLAS injection for hair: a compute pass that appends hair instances to the
//! partitioned-AS WRITE record stream the cluster fill produces. Recorded from
//! inside [`crate::accel::ptlas::dispatch_ptlas`] (between the cluster
//! `fill_incremental` and `finalize` passes) so the single GPU record count
//! covers both.

#![allow(unsafe_code)]

use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bytemuck::{Pod, Zeroable};

use crate::gpu::allocator::Allocator;
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
use crate::gpu::rt_pipeline::RtPipeline;

use super::HairInstances;

const WORKGROUP_SIZE: u32 = 64;

/// Push params shared with `ptlas_hair_write.slang::HairWriteParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct HairWriteParams {
    pub hair_count: u32,
    pub hair_base: u32,
    /// SBT hit-record index hair instances route to ([`RtPipeline::hair_sbt_record`]),
    /// baked into each hair record's `instance_contribution_to_hit_group_index` so
    /// the trace reaches `chit_hair`. 0 until the RT pipeline exists (no trace yet).
    pub hair_sbt_record: u32,
    /// X workgroup count of the 2D-split dispatch (flat-index reconstruction).
    pub groups_x: u32,
}

/// Render-world resource: the hair PTLAS-write heap kernel + its slots.
#[derive(Resource)]
pub struct HairPtlasWrite {
    pub hair_count: u32,
    pub groups: (u32, u32, u32),
    pub params: HairWriteParams,
    pub kernel: HeapKernel,
    pub slots: KernelSlots,
    pub raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for HairPtlasWrite {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for HairPtlasWrite {}
unsafe impl Sync for HairPtlasWrite {}

/// `RenderStartup` (after `SolariSetup`): compile the hair-write kernel — a
/// layout-free heap pipeline ([`HeapKernel`]), Slang from source.
pub fn init_hair_ptlas_write(
    mut commands: Commands,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let Some(kernel) = HeapKernel::new(
        &seam,
        "ptlas_hair_write.slang",
        include_str!("ptlas_hair_write.slang"),
        "hair_write",
        &[],
        &[],
        "ptlas_hair_write",
        size_of::<HairWriteParams>() as u32,
    ) else {
        return;
    };
    let slots = KernelSlots::new(&seam, 4);
    commands.insert_resource(HairPtlasWrite {
        hair_count: 0,
        groups: (0, 0, 0),
        params: HairWriteParams::default(),
        kernel,
        slots,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

/// `Render::Prepare`: set the hair-write params (count + PTLAS base). The base
/// is assigned by [`crate::accel::ptlas::prepare_ptlas_params`] onto
/// [`HairInstances`] before this runs.
pub fn prepare_hair_ptlas_write(
    write: Option<ResMut<HairPtlasWrite>>,
    instances: Option<Res<HairInstances>>,
    rt_pipeline: Option<Res<RtPipeline>>,
) {
    let Some(mut write) = write else {
        return;
    };
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
    write.groups = crate::ecs_gpu::linear_dispatch(instances.count.div_ceil(WORKGROUP_SIZE));
    write.params = HairWriteParams {
        hair_count: instances.count,
        hair_base: instances.base,
        hair_sbt_record,
        groups_x: write.groups.0,
    };
}
