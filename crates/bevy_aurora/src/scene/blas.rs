#![allow(
    clippy::undocumented_unsafe_blocks,
    reason = "Each unsafe fn documents its own contract; per-block SAFETY comments would just restate it."
)]
#![allow(
    clippy::doc_markdown,
    reason = "Vulkan / Cargo identifiers (cluster_AS, vk::*, etc.) need not be backticked."
)]
//! Cluster bottom-level acceleration structure build.
//!
//! Given the per-meshlet CLAS device addresses produced by
//! [`super::cluster_as::ClusterAsManager::upload_mesh`], this module builds a
//! single BLAS that references all of them via the cluster_AS extension's
//! `BUILD_CLUSTERS_BOTTOM_LEVEL` op_type. The resulting BLAS device address
//! is what a TLAS instance's `accelerationStructureReference` points at.
//!
//! Like the CLAS build, this lives in a persistent storage buffer
//! (`blas_storage` on the manager) and the address survives across frames.

use core::ops::Range;

use ash::vk::{self, TaggedStructure as _};

use super::cluster_as::CLAS_STORAGE_ALIGNMENT;
use super::raw_vk::{PersistentBuffer, RangeAllocator, RawBuffer};

/// Result of [`build_blas_for_clusters`].
pub(super) struct BuiltBlas {
    /// Byte range within the manager's blas_storage that holds this BLAS.
    pub storage_range: Range<u64>,
    /// Device address of the built BLAS, suitable to use as
    /// `accelerationStructureReference` in a TLAS instance.
    pub address: u64,
    /// `VkAccelerationStructureKHR` handle aliasing [`Self::storage_range`].
    /// Created via `vkCreateAccelerationStructureKHR` (not via the cluster_AS
    /// extension); the cluster_AS build writes the AS payload into the
    /// storage range, and this handle gives that region a "real" KHR-AS
    /// identity so the TLAS path recognises it.
    ///
    /// Diagnostic toggle: if Aurora was reading the cluster_AS
    /// `dst_addresses[0]` directly (the "load-bearing claim" path) and that
    /// wasn't producing a valid TLAS reference, calling
    /// `vkGetAccelerationStructureDeviceAddressKHR(handle)` instead returns
    /// the address every KHR-AS-driven path expects.
    pub handle: ash::vk::AccelerationStructureKHR,
}

/// Build a single cluster-bottom-level BLAS that references `clas_addresses`.
///
/// Invariant on the M-A scope: one BLAS per uploaded mesh, all clusters are
/// resident from creation. Streaming variants will replace this with a
/// rebuild path that re-references a subset of CLAS addresses.
///
/// # Safety
///
/// - `device`/`mem_props` must come from the same physical device that
///   produced `wgpu_device`.
/// - The CLAS addresses in `clas_addresses` must point at valid built CLAS
///   payloads in storage that outlives the resulting BLAS.
/// - No in-flight GPU work may reference the scratch / dst_addresses
///   transient buffers; we `vkDeviceWaitIdle` before reading them back.
pub(super) unsafe fn build_blas_for_clusters(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    hal_device: &wgpu::hal::vulkan::Device,
    wgpu_device: &wgpu::Device,
    queue: &wgpu::Queue,
    blas_storage: &mut PersistentBuffer,
    blas_free_ranges: &mut RangeAllocator,
    clas_addresses: &[u64],
) -> BuiltBlas {
    let cluster_count = clas_addresses.len();
    assert!(cluster_count > 0, "BLAS build requires at least one CLAS");

    // ---- Stage cluster_references array (u64 device addrs) -------------------
    let cluster_refs_bytes = bytemuck::cast_slice::<u64, u8>(clas_addresses);
    let cluster_refs = unsafe {
        RawBuffer::alloc(
            device,
            mem_props,
            cluster_refs_bytes.len() as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            true,
        )
    };
    unsafe { cluster_refs.upload(device, cluster_refs_bytes) };

    // ---- Build per-BLAS descriptor -------------------------------------------
    // Plain `#[repr(C)]` struct; ash exposes the inner fields differently
    // across versions, so we declare it locally to be robust.
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct BuildClustersBottomLevelInfo {
        cluster_references_count: u32,
        cluster_references_stride: u32,
        cluster_references: u64,
    }
    let blas_desc = BuildClustersBottomLevelInfo {
        cluster_references_count: cluster_count as u32,
        cluster_references_stride: 8, // sizeof VkDeviceAddress
        cluster_references: cluster_refs.addr,
    };
    let desc_bytes = unsafe {
        core::slice::from_raw_parts(
            core::ptr::from_ref(&blas_desc).cast::<u8>(),
            size_of_val(&blas_desc),
        )
    };
    let src_infos = unsafe {
        RawBuffer::alloc(
            device,
            mem_props,
            desc_bytes.len() as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            true,
        )
    };
    unsafe { src_infos.upload(device, desc_bytes) };

    // ---- Size query ----------------------------------------------------------
    let bottom_level_input = vk::ClusterAccelerationStructureClustersBottomLevelInputNV::default()
        .max_total_cluster_count(cluster_count as u32)
        .max_cluster_count_per_acceleration_structure(cluster_count as u32);
    let op_input = vk::ClusterAccelerationStructureOpInputNV {
        p_clusters_bottom_level: core::ptr::from_ref(&bottom_level_input).cast_mut(),
    };
    let input_info = vk::ClusterAccelerationStructureInputInfoNV::default()
        .max_acceleration_structure_count(1)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_CLUSTERS_BOTTOM_LEVEL)
        .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
        .op_input(op_input);

    let sizes = unsafe { hal_device.get_cluster_build_sizes(&input_info) };

    // ---- Allocate range in blas_storage + transient outputs ------------------
    let storage_range = blas_free_ranges
        .alloc(sizes.acceleration_structure_size, CLAS_STORAGE_ALIGNMENT)
        .unwrap_or_else(|| {
            panic!(
                "BLAS storage exhausted: requested {} bytes, used {} / {}",
                sizes.acceleration_structure_size,
                blas_free_ranges.used(),
                blas_free_ranges.capacity(),
            )
        });
    unsafe { blas_storage.ensure_resident(storage_range.clone()) };

    let scratch = unsafe {
        RawBuffer::alloc(
            device,
            mem_props,
            sizes.build_scratch_size.max(1),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            false,
        )
    };
    let dst_addresses = unsafe {
        RawBuffer::alloc(
            device,
            mem_props,
            8,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            true,
        )
    };
    let count_buf = unsafe {
        RawBuffer::alloc(
            device,
            mem_props,
            4,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            true,
        )
    };
    unsafe { count_buf.upload(device, &1u32.to_le_bytes()) };

    // ---- Record + submit ----------------------------------------------------
    let stride = size_of_val(&blas_desc) as u64;
    let commands_info = vk::ClusterAccelerationStructureCommandsInfoNV {
        s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
        p_next: core::ptr::null_mut(),
        input: input_info,
        dst_implicit_data: blas_storage.device_address(storage_range.start),
        scratch_data: scratch.addr,
        dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
            device_address: dst_addresses.addr,
            stride: 8,
            size: 8,
        },
        dst_sizes_array: vk::StridedDeviceAddressRegionKHR::default(),
        src_infos_array: vk::StridedDeviceAddressRegionKHR {
            device_address: src_infos.addr,
            stride,
            size: desc_bytes.len() as u64,
        },
        src_infos_count: count_buf.addr,
        address_resolution_flags:
            vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
        _marker: core::marker::PhantomData,
    };

    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("aurora.cluster_as.build_blas"),
    });
    unsafe {
        encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|maybe_cmd| {
            if let Some(cmd) = maybe_cmd {
                cmd.cmd_build_cluster_acceleration_structures_indirect(&commands_info);
            }
        });
    }
    let cb = encoder.finish();
    queue.submit([cb]);
    unsafe { device.device_wait_idle().expect("BLAS device_wait_idle") };

    // ---- Read BLAS device address (cluster_AS-reported) ---------------------
    let cluster_reported_addr = unsafe {
        let ptr = device
            .map_memory(dst_addresses.mem, 0, 8, vk::MemoryMapFlags::empty())
            .expect("map BLAS dst_addresses")
            .cast::<u64>();
        let addr = core::ptr::read_unaligned(ptr);
        device.unmap_memory(dst_addresses.mem);
        addr
    };
    assert!(
        cluster_reported_addr != 0,
        "BLAS build returned a zero device address -- driver did not write the output"
    );

    // KHR-AS wrap is performed by the caller (cluster_as.rs::upload_mesh)
    // because that path has the `ash::Instance` reference handy.
    unsafe {
        cluster_refs.destroy(device);
        src_infos.destroy(device);
        scratch.destroy(device);
        dst_addresses.destroy(device);
        count_buf.destroy(device);
    }

    tracing::debug!(
        target: "bevy_aurora",
        clusters = cluster_count,
        blas_storage_bytes = sizes.acceleration_structure_size,
        scratch_bytes = sizes.build_scratch_size,
        addr_cluster_reported = format!("{cluster_reported_addr:#018x}"),
        "BLAS built (cluster_AS); KHR-AS wrap deferred to caller",
    );

    BuiltBlas {
        storage_range,
        address: cluster_reported_addr,
        // Set to NULL by build; the caller wraps via `wrap_built_blas` once
        // it has an `ash::Instance` reference handy.
        handle: vk::AccelerationStructureKHR::null(),
    }
}

/// Wrap the BLAS storage range in a `VkAccelerationStructureKHR` handle and
/// rewrite [`BuiltBlas::address`] / [`BuiltBlas::handle`] to use that
/// KHR-AS path instead of the cluster_AS-reported address.
///
/// # Safety
///
/// - `device`/`raw_instance` must come from the same physical device that
///   built the BLAS.
/// - `blas_storage` must outlive the returned handle.
/// - The build phase must have completed (i.e., `vkDeviceWaitIdle` has run).
pub(super) unsafe fn wrap_built_blas(
    device: &ash::Device,
    raw_instance: &ash::Instance,
    blas_storage: &PersistentBuffer,
    built: &mut BuiltBlas,
    blas_size: u64,
) {
    let khr_as = ash::khr::acceleration_structure::Device::load(raw_instance, device);
    let create = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(blas_storage.raw())
        .offset(built.storage_range.start)
        .size(blas_size)
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
    let handle = unsafe {
        khr_as
            .create_acceleration_structure(&create, None)
            .expect("create_acceleration_structure (cluster_AS BLAS wrap)")
    };
    let khr_addr = unsafe {
        khr_as.get_acceleration_structure_device_address(
            &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle),
        )
    };
    tracing::info!(
        target: "bevy_aurora",
        cluster_reported = format!("0x{:016x}", built.address),
        khr_wrapped     = format!("0x{khr_addr:016x}"),
        equal = (built.address == khr_addr),
        "BLAS device address comparison (cluster_AS vs KHR-AS wrap)"
    );
    built.handle = handle;
    built.address = khr_addr;
}
