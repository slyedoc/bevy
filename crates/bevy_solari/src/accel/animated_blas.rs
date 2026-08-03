// Animated per-instance CLAS instantiation + BLAS build.
//
// Three GPU steps per frame (only when animated instances are active):
//   1. instantiate compute (`instantiate.slang`) — per active slot, emit
//      `InstantiateClusterInfoNV` records pointing each finest-LOD cluster's
//      template at its deformed-position slice, plus a per-slot
//      `BuildClustersBottomLevelInfoNV` arg.
//   2. raw-VK INSTANTIATE build (`INSTANTIATE_TRIANGLE_CLUSTER`,
//      IMPLICIT_DESTINATIONS) — fresh per-cluster CLAS into a per-frame arena;
//      addresses written to `instantiated_clas_addrs`.
//   3. raw-VK per-instance BLAS build (`BUILD_CLUSTERS_BOTTOM_LEVEL`,
//      EXPLICIT_DESTINATIONS) — one BLAS per active slot into a stable pool.
//
// The per-instance BLAS lives at `blas_pool.address + slot_idx * stride`, and the
// instantiate compute repoints `instance_blas_address[slot]` at it (overwriting
// the static shared-BLAS address from `blas_sharing::assign_address`, since this
// stage runs after it). `prepare_ptlas_params` re-specifies animated slots every
// frame (their BLAS content changes in place), and the resolve shader reads the
// deform pool for animated hits. Runs in the `BuildAnimatedBlas` stage.
#![allow(unsafe_code, reason = "raw-VK cluster instantiate + BLAS builds")]

use ash::vk::{self, TaggedStructure};
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    render_resource::{Buffer, BufferDescriptor, BufferUsages},
    renderer::{RenderContext, RenderDevice, RenderQueue},
};
use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};
use wgpu::hal::api::Vulkan as VkApi;

use super::blas_rebuild::query_blas_size;
use super::deform::{Deform, MAX_ANIMATED_INSTANCES};
use crate::ecs_gpu::GpuColumn;
use crate::geometry::{ClusterMeshManager, ClusterTemplateArena};
use crate::gpu::allocator::{Allocator, MemoryLocation, SparseBuffer};
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::extension::{AsSeams, ClusterExtensionFns};
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
use crate::instance::{InstanceManager, LodInputColumn};

/// Worst-case finest-LOD clusters per animated mesh (sizes the per-frame arenas).
pub const MAX_CLUSTERS_PER_ANIMATED_MESH: u32 = 2048;
/// Worst-case total animated clusters across all active slots this frame.
pub const MAX_TOTAL_ANIMATED_CLUSTERS: u32 = MAX_ANIMATED_INSTANCES * MAX_CLUSTERS_PER_ANIMATED_MESH;
/// `VkClusterAccelerationStructureInstantiateClusterInfoNV` size.
const INSTANTIATE_INFO_STRIDE: u64 = 32;
/// `VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV` size.
const BLAS_ARG_STRIDE: u64 = 16;
const STORAGE_ALIGN: u64 = 256;
const SCRATCH_ALIGN: u64 = 256;
/// Worst-case instantiated-CLAS bytes per cluster (sparse arena sizing).
const WORST_INSTANTIATE_BYTES_PER_CLUSTER: u64 = 8192;
const INSTANTIATE_STORAGE_VIRTUAL_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const ANIMATED_BLAS_POOL_VIRTUAL_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const SCRATCH_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;

/// Push params shared with `instantiate.slang::InstantiateParams` (32 B). The
/// address fields are `UVec2` (→ `uint2`), u64 device addresses split lo/hi.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct InstantiateParams {
    num_slots: u32,
    blas_stride: u32,
    deform_positions_addr: UVec2,
    instantiated_clas_addrs_addr: UVec2,
    blas_pool_base: UVec2,
}

/// Render-world resource owning the animated instantiate + BLAS pipeline.
#[derive(Resource)]
pub struct AnimatedBlas {
    // Instantiate compute (a layout-free heap kernel).
    params: InstantiateParams,
    kernel: HeapKernel,
    slots: KernelSlots,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,

    /// InstantiateClusterInfoNV records (GPU-written).
    instantiate_args: Buffer,
    /// GPU record counter (INSTANTIATE `srcInfosCount`).
    count: Buffer,
    /// Per-slot BuildClustersBottomLevelInfoNV args (GPU-written).
    blas_args: Buffer,

    /// Per-frame instantiated-CLAS storage arena.
    instantiated_clas_storage: SparseBuffer,
    /// INSTANTIATE dst addresses (per cluster) — BLAS cluster refs point here.
    instantiated_clas_addrs: Buffer,
    /// INSTANTIATE dst sizes (per cluster) — required or CLAS is silently invalid.
    instantiated_clas_sizes: Buffer,
    instantiate_scratch: SparseBuffer,

    /// Per-instance BLAS storage pool (stable: address = base + slot * stride).
    pub blas_pool: SparseBuffer,
    /// EXPLICIT BLAS dst addresses (per slot) — CPU-written each frame.
    blas_dst_addresses: Buffer,
    /// BLAS `srcInfosCount` (= active slot count), CPU-written each frame.
    blas_count: Buffer,
    blas_scratch: SparseBuffer,
    /// Worst-case single-instance BLAS byte size (pool region stride).
    pub blas_stride: u64,
}

impl Drop for AnimatedBlas {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for AnimatedBlas {}
unsafe impl Sync for AnimatedBlas {}

fn raw_storage(device: &RenderDevice, label: &'static str, size: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::BLAS_INPUT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// `RenderStartup` (after `SolariSetup`): compile the instantiate kernel — a
/// layout-free heap pipeline ([`HeapKernel`]) — and allocate the
/// instantiate/BLAS buffers. No-op when the raw-VK [`Allocator`] is absent.
pub fn init_animated_blas(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let Some(kernel) = HeapKernel::new(
        &seam,
        "instantiate.slang",
        include_str!("instantiate.slang"),
        "instantiate",
        &[],
        &[],
        "animated_instantiate",
        size_of::<InstantiateParams>() as u32,
    ) else {
        return;
    };
    let slots = KernelSlots::new(&seam, 11);
    let instantiate_args = raw_storage(
        &render_device,
        "animated.instantiate_args",
        MAX_TOTAL_ANIMATED_CLUSTERS as u64 * INSTANTIATE_INFO_STRIDE,
    );
    let count = render_device.create_buffer(&BufferDescriptor {
        label: Some("animated.count"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::BLAS_INPUT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let blas_args = raw_storage(
        &render_device,
        "animated.blas_args",
        MAX_ANIMATED_INSTANCES as u64 * BLAS_ARG_STRIDE,
    );
    let instantiated_clas_storage = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        BufferUsages::COPY_DST,
        INSTANTIATE_STORAGE_VIRTUAL_BYTES,
        "animated.instantiated_clas_storage",
    );
    let instantiated_clas_addrs = allocator.create_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        BufferUsages::STORAGE,
        MAX_TOTAL_ANIMATED_CLUSTERS as u64 * 8,
        MemoryLocation::GpuOnly,
        "animated.instantiated_clas_addrs",
    ).into();
    let instantiated_clas_sizes = allocator.create_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        BufferUsages::STORAGE,
        MAX_TOTAL_ANIMATED_CLUSTERS as u64 * 4,
        MemoryLocation::GpuOnly,
        "animated.instantiated_clas_sizes",
    ).into();
    let instantiate_scratch = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        BufferUsages::STORAGE,
        SCRATCH_VIRTUAL_BYTES,
        "animated.instantiate_scratch",
    );

    let blas_pool = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        BufferUsages::COPY_DST,
        ANIMATED_BLAS_POOL_VIRTUAL_BYTES,
        "animated.blas_pool",
    );
    let blas_dst_addresses = allocator.create_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::TRANSFER_DST,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
        MAX_ANIMATED_INSTANCES as u64 * 8,
        MemoryLocation::GpuOnly,
        "animated.blas_dst_addresses",
    ).into();
    let blas_count = render_device.create_buffer(&BufferDescriptor {
        label: Some("animated.blas_count"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::BLAS_INPUT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let blas_scratch = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        BufferUsages::STORAGE,
        SCRATCH_VIRTUAL_BYTES,
        "animated.blas_scratch",
    );

    commands.insert_resource(AnimatedBlas {
        params: InstantiateParams::default(),
        kernel,
        slots,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
        instantiate_args,
        count,
        blas_args,
        instantiated_clas_storage,
        instantiated_clas_addrs,
        instantiated_clas_sizes,
        instantiate_scratch,
        blas_pool,
        blas_dst_addresses,
        blas_count,
        blas_scratch,
        blas_stride: 0,
    });
}

/// `Render::Prepare`: size the BLAS pool stride, write per-frame CPU inputs
/// (count zero, blas_count, blas dst addresses, the push params), commit
/// sparse pages.
pub fn prepare_animated_blas(
    resources: Option<ResMut<AnimatedBlas>>,
    deform: Option<Res<Deform>>,
    instances: Option<Res<InstanceManager>>,
    fns: Option<Res<ClusterExtensionFns>>,
    allocator: Option<Res<Allocator>>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(mut resources), Some(deform), Some(instances), Some(fns), Some(allocator)) =
        (resources, deform, instances, fns, allocator)
    else {
        return;
    };
    let Some(cluster_fns) = fns.cluster.as_ref() else {
        return;
    };
    let active = deform.active_count();
    if active == 0 {
        return;
    }

    // Per-instance BLAS region stride: worst-case single-mesh BLAS.
    let max_per = instances
        .max_clusters_per_bucket()
        .min(MAX_CLUSTERS_PER_ANIMATED_MESH)
        .max(1);
    let want = query_blas_size(cluster_fns, max_per)
        .max(STORAGE_ALIGN)
        .next_multiple_of(STORAGE_ALIGN);
    resources.blas_stride = resources.blas_stride.max(want);
    let stride = resources.blas_stride;

    // Zero the instantiate counter; set BLAS op count = active slots.
    render_queue.write_buffer(&resources.count, 0, &0u32.to_le_bytes());
    render_queue.write_buffer(&resources.blas_count, 0, &active.to_le_bytes());

    // EXPLICIT BLAS dst addresses: pool_base + slot_idx * stride.
    let pool_base = resources.blas_pool.address;
    let mut dst = Vec::with_capacity(active as usize);
    for i in 0..active as u64 {
        dst.push(pool_base + i * stride);
    }
    render_queue.write_buffer(
        &resources.blas_dst_addresses,
        0,
        bytemuck::cast_slice(&dst),
    );

    // Params: deform pool + instantiated-CLAS-addr device addresses.
    let deform_addr = deform.positions.stable_addr().get();
    let clas_addrs = allocator.wgpu_buffer_device_address(&resources.instantiated_clas_addrs).get();
    resources.params = InstantiateParams {
        num_slots: active,
        blas_stride: stride as u32,
        deform_positions_addr: UVec2::new(
            (deform_addr & 0xFFFF_FFFF) as u32,
            (deform_addr >> 32) as u32,
        ),
        instantiated_clas_addrs_addr: UVec2::new(
            (clas_addrs & 0xFFFF_FFFF) as u32,
            (clas_addrs >> 32) as u32,
        ),
        blas_pool_base: UVec2::new(
            (pool_base & 0xFFFF_FFFF) as u32,
            (pool_base >> 32) as u32,
        ),
    };

    // Commit sparse pages for this frame's worst case.
    let total_clusters = MAX_TOTAL_ANIMATED_CLUSTERS as u64;
    resources
        .instantiated_clas_storage
        .commit(0..(total_clusters * WORST_INSTANTIATE_BYTES_PER_CLUSTER));
    resources
        .instantiate_scratch
        .commit(0..(64 * 1024 * 1024));
    resources
        .blas_pool
        .commit(0..((active as u64) * stride).max(1));
    resources.blas_scratch.commit(0..(64 * 1024 * 1024));
}

/// `RenderGraph` (`BuildAnimatedBlas`): instantiate compute → raw-VK INSTANTIATE
/// build → raw-VK per-instance BLAS build.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_animated_blas(
    resources: Option<Res<AnimatedBlas>>,
    deform: Option<Res<Deform>>,
    instances: Option<Res<InstanceManager>>,
    fns: Option<Res<ClusterExtensionFns>>,
    allocator: Option<Res<Allocator>>,
    seam: Option<Res<BindingSeam>>,
    cluster_meshes: Option<Res<ClusterMeshManager>>,
    templates: Option<Res<ClusterTemplateArena>>,
    lod_inputs: Option<Res<GpuColumn<LodInputColumn>>>,
    sharing: Option<Res<super::blas_sharing::BlasSharing>>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (
        Some(resources),
        Some(deform),
        Some(instances),
        Some(fns),
        Some(allocator),
        Some(seam),
        Some(cluster_meshes),
        Some(templates),
        Some(lod_inputs),
        Some(sharing),
    ) = (
        resources,
        deform,
        instances,
        fns,
        allocator,
        seam,
        cluster_meshes,
        templates,
        lod_inputs,
        sharing,
    )
    else {
        return;
    };
    let Some(cluster_fns) = fns.cluster.as_ref() else {
        return;
    };
    let active = deform.active_count();
    if active == 0 {
        return;
    }
    let Some(slots_buffer) = deform.slots_buffer() else {
        return;
    };

    // 1. Instantiate compute — writes instantiate_args + count + blas_args. A raw
    // heap dispatch: buffer slots rewritten per dispatch, params + slot array in
    // push data.
    let blob = resources.kernel.push_blob(
        "animated_instantiate",
        bytemuck::bytes_of(&resources.params),
        &[
            ("active_slots", resources.slots.buffer(&seam, 0, slots_buffer)),
            ("clusters", resources.slots.buffer(&seam, 1, cluster_meshes.clusters.buffer())),
            (
                "cluster_template_addresses",
                resources.slots.buffer(&seam, 2, &templates.cluster_template_addresses.wgpu_buffer),
            ),
            ("instance_lod_inputs", resources.slots.buffer(&seam, 3, lod_inputs.buffer())),
            ("instantiate_args", resources.slots.buffer(&seam, 4, &resources.instantiate_args)),
            ("count", resources.slots.buffer(&seam, 5, &resources.count)),
            ("blas_args", resources.slots.buffer(&seam, 6, &resources.blas_args)),
            (
                "instance_blas_address",
                resources.slots.buffer(&seam, 7, &sharing.instance_blas_address.wgpu_buffer),
            ),
            (
                "instance_e_build",
                resources.slots.buffer(&seam, 8, &sharing.instance_e_build.wgpu_buffer),
            ),
            ("cluster_groups", resources.slots.buffer(&seam, 9, cluster_meshes.groups.buffer())),
            (
                "cluster_to_group",
                resources.slots.buffer(&seam, 10, cluster_meshes.cluster_to_group.buffer()),
            ),
        ],
    );
    {
        let encoder = ctx.command_encoder();
        // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
        // barriers bracket this dispatch against the surrounding wgpu compute
        // passes (raw dispatches are invisible to wgpu's tracking).
        unsafe {
            encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
                let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
                let cb = hal_encoder.raw_handle();
                let dev = &resources.raw_device;
                let barrier = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(
                        vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                    )];
                let dep = vk::DependencyInfo::default().memory_barriers(&barrier);
                // The classify/assign_address column writes -> our reads (and our
                // overwrite of `instance_blas_address`).
                dev.cmd_pipeline_barrier2(cb, &dep);
                seam.bind_heaps(cb);
                seam.push_data(cb, &blob);
                dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, resources.kernel.pipeline);
                dev.cmd_dispatch(cb, active, 1, 1);
                // Our arg/count writes -> downstream compute reads; the builds'
                // own seam covers build-input visibility.
                dev.cmd_pipeline_barrier2(cb, &dep);
            });
        }
    }

    // 2 + 3. Raw-VK INSTANTIATE + BLAS builds in their own encoder (the fork
    // panics if one encoder mixes wgpu passes with raw `as_hal_mut`).
    let max_total = MAX_TOTAL_ANIMATED_CLUSTERS;
    let max_per = instances
        .max_clusters_per_bucket()
        .min(MAX_CLUSTERS_PER_ANIMATED_MESH)
        .max(1);

    // INSTANTIATE size query.
    let mut tri_input = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .max_geometry_index_value(u32::MAX >> 8)
        .max_cluster_unique_geometry_count(1)
        .max_cluster_triangle_count(256)
        .max_cluster_vertex_count(256)
        .max_total_triangle_count(max_total * 256)
        .max_total_vertex_count(max_total * 256)
        .min_position_truncate_bit_count(0);
    let inst_op_input = vk::ClusterAccelerationStructureOpInputNV {
        p_triangle_clusters: &mut tri_input as *mut _,
    };
    let inst_size_input = vk::ClusterAccelerationStructureInputInfoNV::default()
        .max_acceleration_structure_count(max_total)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .op_type(vk::ClusterAccelerationStructureOpTypeNV::INSTANTIATE_TRIANGLE_CLUSTER)
        .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
        .op_input(inst_op_input);
    let mut inst_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: inputs populated; cluster-AS fn table loaded.
    unsafe {
        cluster_fns
            .get_cluster_acceleration_structure_build_sizes(&inst_size_input, &mut inst_sizes);
    }

    // BLAS size query.
    let mut bl_input = vk::ClusterAccelerationStructureClustersBottomLevelInputNV::default()
        .max_total_cluster_count(max_total)
        .max_cluster_count_per_acceleration_structure(max_per);
    let bl_op_input = vk::ClusterAccelerationStructureOpInputNV {
        p_clusters_bottom_level: &mut bl_input as *mut _,
    };
    let bl_size_input = vk::ClusterAccelerationStructureInputInfoNV::default()
        .max_acceleration_structure_count(MAX_ANIMATED_INSTANCES)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_CLUSTERS_BOTTOM_LEVEL)
        .op_mode(vk::ClusterAccelerationStructureOpModeNV::EXPLICIT_DESTINATIONS)
        .op_input(bl_op_input);
    let mut bl_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: as above.
    unsafe {
        cluster_fns
            .get_cluster_acceleration_structure_build_sizes(&bl_size_input, &mut bl_sizes);
    }

    let inst_scratch = align_addr(resources.instantiate_scratch.address, SCRATCH_ALIGN);
    let blas_scratch = align_addr(resources.blas_scratch.address, SCRATCH_ALIGN);

    // The scratch commits are fixed-size, so validate the driver's reported
    // scratch requirements against the committed pages BEFORE recording: a
    // scratch overflow is a GPU write past committed sparse memory — a
    // device-lost, not a validation error. Skipping the builds keeps last
    // frame's animated BLASes (poses freeze) instead. The destination pools
    // are not checked here: their sizing is workload-derived (per-cluster
    // instantiate bound; per-BLAS stride from the single-AS size query), while
    // these batch-input queries report a uniform worst case over the input
    // maxima — far above any real workload.
    let inst_scratch_needed =
        (inst_scratch - resources.instantiate_scratch.address) + inst_sizes.build_scratch_size;
    let blas_scratch_needed =
        (blas_scratch - resources.blas_scratch.address) + bl_sizes.build_scratch_size;
    if !resources.instantiate_scratch.is_committed(0..inst_scratch_needed)
        || !resources.blas_scratch.is_committed(0..blas_scratch_needed)
    {
        bevy_log::error_once!(
            "animated_blas: driver-required scratch exceeds committed capacity \
             (instantiate {} B, blas {} B) — skipping animated builds",
            inst_sizes.build_scratch_size,
            bl_sizes.build_scratch_size,
        );
        return;
    }

    let inst_args_addr = allocator.wgpu_buffer_device_address(&resources.instantiate_args).get();
    let inst_count_addr = allocator.wgpu_buffer_device_address(&resources.count).get();
    let inst_addrs_addr = allocator.wgpu_buffer_device_address(&resources.instantiated_clas_addrs).get();
    let inst_sizes_addr = allocator.wgpu_buffer_device_address(&resources.instantiated_clas_sizes).get();
    let blas_args_addr = allocator.wgpu_buffer_device_address(&resources.blas_args).get();
    let blas_dst_addr = allocator.wgpu_buffer_device_address(&resources.blas_dst_addresses).get();
    let blas_count_addr = allocator.wgpu_buffer_device_address(&resources.blas_count).get();

    let inst_cmd = vk::ClusterAccelerationStructureCommandsInfoNV {
        s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
        p_next: core::ptr::null_mut(),
        input: inst_size_input,
        dst_implicit_data: resources.instantiated_clas_storage.address,
        scratch_data: inst_scratch,
        dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
            device_address: inst_addrs_addr,
            stride: 8,
            size: max_total as u64 * 8,
        },
        dst_sizes_array: vk::StridedDeviceAddressRegionKHR {
            device_address: inst_sizes_addr,
            stride: 4,
            size: max_total as u64 * 4,
        },
        src_infos_array: vk::StridedDeviceAddressRegionKHR {
            device_address: inst_args_addr,
            stride: INSTANTIATE_INFO_STRIDE,
            size: max_total as u64 * INSTANTIATE_INFO_STRIDE,
        },
        src_infos_count: inst_count_addr,
        address_resolution_flags:
            vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
        _marker: core::marker::PhantomData,
    };

    let blas_cmd = vk::ClusterAccelerationStructureCommandsInfoNV {
        s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
        p_next: core::ptr::null_mut(),
        input: bl_size_input,
        dst_implicit_data: resources.blas_pool.address,
        scratch_data: blas_scratch,
        dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
            device_address: blas_dst_addr,
            stride: 8,
            size: MAX_ANIMATED_INSTANCES as u64 * 8,
        },
        dst_sizes_array: vk::StridedDeviceAddressRegionKHR::default(),
        src_infos_array: vk::StridedDeviceAddressRegionKHR {
            device_address: blas_args_addr,
            stride: BLAS_ARG_STRIDE,
            size: MAX_ANIMATED_INSTANCES as u64 * BLAS_ARG_STRIDE,
        },
        src_infos_count: blas_count_addr,
        address_resolution_flags:
            vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
        _marker: core::marker::PhantomData,
    };

    // Declared access (SolariSettings::validate): the CPU-knowable ranges of the two
    // raw builds — instantiate writes the CLAS storage + scratch, the BLAS
    // build writes the pool + scratch (per-cluster dst addrs are GPU-computed).
    let total_clusters = MAX_TOTAL_ANIMATED_CLUSTERS as u64;
    crate::gpu::extension::validate_raw_access(&crate::gpu::extension::RawAccess {
        op: "animated_blas.builds",
        reads: &[],
        writes: &[
            (
                &resources.instantiated_clas_storage,
                0..total_clusters * WORST_INSTANTIATE_BYTES_PER_CLUSTER,
            ),
            (&resources.instantiate_scratch, 0..64 * 1024 * 1024),
            (&resources.blas_scratch, 0..64 * 1024 * 1024),
        ],
    });

    let mut encoder = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("animated_blas.builds"),
    });
    // SAFETY: cluster-AS fn table loaded; encoder open + Vulkan-backed; the
    // command infos reference live buffers committed this frame. The seams make
    // the instantiate-compute writes + INSTANTIATE output visible as build input.
    // `blas_pool` is rebuilt in place at stable addresses every frame, so the
    // leading seam also orders the previous frame's trace ahead of the rewrite.
    unsafe {
        crate::gpu::extension::cmd_as_seam(
            &mut encoder,
            &render_device,
            AsSeams::COMPUTE_TO_BUILD_INPUT | AsSeams::TRACE_TO_BUILD_WAR,
        );
        crate::gpu::extension::cmd_build_cluster_acceleration_structures_indirect(
            &mut encoder,
            &fns,
            &inst_cmd,
        );
        crate::gpu::extension::cmd_as_seam(&mut encoder, &render_device, AsSeams::BUILD_TO_BUILD_INPUT);
        crate::gpu::extension::cmd_build_cluster_acceleration_structures_indirect(
            &mut encoder,
            &fns,
            &blas_cmd,
        );
        crate::gpu::extension::cmd_as_seam(
            &mut encoder,
            &render_device,
            AsSeams::BUILD_TO_BUILD_INPUT | AsSeams::BUILD_TO_TRACE,
        );
    }
    ctx.add_command_buffer(encoder.finish());
}

fn align_addr(addr: u64, align: u64) -> u64 {
    let m = addr & (align - 1);
    if m == 0 {
        addr
    } else {
        addr + (align - m)
    }
}
