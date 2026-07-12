#![allow(unsafe_code)]

//! Per-instance cluster BLAS build for the GPU tessellation path (raw Vulkan for
//! the cluster-AS build, like `clas_arena`).
//!
//! `tess_classify` generates one CLAS per tessellated part on the GPU;
//! [`record_build_per_instance_blas`] records (no submit) a bottom-level
//! acceleration structure over those CLASes — a tessellated instance gets its
//! *own* BLAS over its generated CLASes, unlike solari's shared
//! per-`(geometry, LOD)` BLAS. The per-CLAS `cluster_id`s are baked from
//! [`TESS_CLUSTER_ID_BASE`] so the closest-hit detects the tess hit and recovers
//! the per-CLAS metadata index.

use ash::vk::{self, TaggedStructure};
use bevy_render::renderer::{RenderDevice, RenderQueue};

use crate::gpu::allocator::{Allocator, MemoryLocation};
use crate::gpu::extension::ClusterExtensionFns;

/// Base `cluster_id` (ClusterIDNV) for tessellated CLASes — each bakes
/// `TESS_CLUSTER_ID_BASE + global_tess_clas_index`, comfortably above any real
/// cluster pool so the closest-hit detects the tess hit (`cluster_id >=
/// arrayLength(&clusters)`) and recovers the tess index for the smooth-normal
/// metadata lookup. Must match `scene_bindings::TESS_CLUSTER_ID_BASE`.
pub const TESS_CLUSTER_ID_BASE: u32 = 0xF000_0000;

/// Output of [`record_build_per_instance_blas`]: the BLAS AS storage (kept alive
/// for the trace) and the build-only transients. The BLAS device address is
/// written GPU-side to the caller's `blas_dst_addr` (no readback).
pub struct InstanceBlas {
    pub storage: wgpu::Buffer,
    /// Mixed buffer types, boxed — held by the caller until the submit completes.
    pub transient: Vec<Box<dyn core::any::Any + Send + Sync>>,
}

/// Record (no submit) a **per-instance** bottom-level acceleration structure over
/// `cluster_count` CLASes whose device addresses live GPU-side at
/// `cluster_references` — the addresses are read straight from that buffer, never
/// uploaded from the CPU. The driver writes the chosen BLAS device address
/// (IMPLICIT) to `blas_dst_addr` — a device address into the caller's persistent
/// per-instance address buffer (e.g. `+ instance_index * 8`) — consumed GPU-side,
/// never read back. Mirrors the `BUILD_CLUSTERS_BOTTOM_LEVEL` op in
/// [`crate::accel::blas_rebuild`]. The build is bracketed with AS barriers; the
/// caller submits the encoder once.
///
/// # Panics
///
/// Panics if the cluster-AS function table is missing or the size query fails.
pub fn record_build_per_instance_blas(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    allocator: &Allocator,
    fns: &ClusterExtensionFns,
    encoder: &mut wgpu::CommandEncoder,
    cluster_references: vk::DeviceAddress,
    cluster_count: u32,
    blas_dst_addr: vk::DeviceAddress,
) -> InstanceBlas {
    let cluster_fns = fns
        .cluster
        .as_ref()
        .expect("tess_template.record_build_per_instance_blas: cluster-AS extension missing");

    let info = vk::ClusterAccelerationStructureBuildClustersBottomLevelInfoNV {
        cluster_references_count: cluster_count,
        cluster_references_stride: 8,
        cluster_references,
    };
    let stride =
        size_of::<vk::ClusterAccelerationStructureBuildClustersBottomLevelInfoNV>() as u64;
    // SAFETY: BuildClustersBottomLevelInfoNV is repr(C) POD; flat byte view ok.
    let info_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(core::ptr::from_ref(&info).cast::<u8>(), stride as usize)
    };
    let src_infos = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tess_template.blas.src_infos"),
        size: stride,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    render_queue.write_buffer(&src_infos, 0, info_bytes);
    let src_infos_addr = allocator.wgpu_buffer_device_address(&src_infos).get();

    let count_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tess_template.blas.count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    render_queue.write_buffer(&count_buf, 0, &1u32.to_le_bytes());
    let count_addr = allocator.wgpu_buffer_device_address(&count_buf).get();

    let mut clusters_input =
        vk::ClusterAccelerationStructureClustersBottomLevelInputNV::default()
            .max_total_cluster_count(cluster_count)
            .max_cluster_count_per_acceleration_structure(cluster_count);
    let op_input = vk::ClusterAccelerationStructureOpInputNV {
        p_clusters_bottom_level: &mut clusters_input as *mut _,
    };
    let size_input = vk::ClusterAccelerationStructureInputInfoNV::default()
        .max_acceleration_structure_count(1)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_CLUSTERS_BOTTOM_LEVEL)
        .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
        .op_input(op_input);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: input fully populated; function table loaded above.
    unsafe {
        cluster_fns.get_cluster_acceleration_structure_build_sizes(&size_input, &mut sizes);
    }

    let storage = allocator.create_buffer(
        render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        wgpu::BufferUsages::STORAGE,
        sizes.acceleration_structure_size.max(1),
        MemoryLocation::GpuOnly,
        "tess_template.blas.storage",
    );
    let storage_addr = allocator.wgpu_buffer_device_address(&storage).get();

    let scratch = allocator.create_buffer(
        render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        wgpu::BufferUsages::STORAGE,
        sizes.build_scratch_size.max(1) + 255,
        MemoryLocation::GpuOnly,
        "tess_template.blas.scratch",
    );
    let scratch_addr = {
        let base = allocator.wgpu_buffer_device_address(&scratch).get();
        let m = base & 255;
        if m == 0 { base } else { base + (256 - m) }
    };

    // The driver writes the chosen BLAS address (IMPLICIT) to `blas_dst_addr` — a
    // slot in the caller's persistent per-instance address buffer (which carries the
    // ACCELERATION_STRUCTURE_STORAGE usage VUID-...-pCommandInfos-10459 requires).
    let cmd = vk::ClusterAccelerationStructureCommandsInfoNV {
        s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
        p_next: core::ptr::null_mut(),
        input: size_input,
        dst_implicit_data: storage_addr,
        scratch_data: scratch_addr,
        dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
            device_address: blas_dst_addr,
            stride: 8,
            size: 8,
        },
        dst_sizes_array: vk::StridedDeviceAddressRegionKHR::default(),
        src_infos_array: vk::StridedDeviceAddressRegionKHR {
            device_address: src_infos_addr,
            stride,
            size: stride,
        },
        src_infos_count: count_addr,
        address_resolution_flags:
            vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
        _marker: core::marker::PhantomData,
    };
    // SAFETY: function table loaded; encoder open + Vulkan-backed; the descriptor
    // strides/addresses reference the buffers above. Leading barrier: the preceding
    // instantiate's CLAS addresses are visible as `cluster_references`. Trailing
    // barrier: the BLAS address + payload are visible to the later trace.
    unsafe {
        crate::gpu::extension::cmd_global_as_barrier(encoder, render_device, false);
        crate::gpu::extension::cmd_build_cluster_acceleration_structures_indirect(
            encoder,
            fns,
            &cmd,
        );
        crate::gpu::extension::cmd_global_as_barrier(encoder, render_device, false);
    }

    let transient: Vec<Box<dyn core::any::Any + Send + Sync>> =
        vec![Box::new(src_infos), Box::new(count_buf), Box::new(scratch)];
    InstanceBlas { storage, transient }
}
