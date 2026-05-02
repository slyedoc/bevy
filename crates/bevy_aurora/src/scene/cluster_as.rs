#![allow(
    clippy::undocumented_unsafe_blocks,
    reason = "Each unsafe fn documents its own contract; per-block SAFETY comments would just restate it."
)]
#![allow(
    clippy::doc_markdown,
    reason = "Vulkan / Cargo identifiers (cluster_AS, vk::*, M-A.streaming) need not be backticked."
)]
//! Cluster acceleration structure manager.
//!
//! [`ClusterAsManager`] owns the persistent CLAS storage buffer and the
//! per-mesh metadata for everything Aurora has built. New meshes are uploaded
//! through [`ClusterAsManager::upload_mesh`], which:
//!
//! 1. Stages each meshlet's dequantized positions + indices into transient
//!    host-visible Vulkan buffers.
//! 2. Builds a per-cluster
//!    `VkClusterAccelerationStructureBuildTriangleClusterInfoNV` descriptor
//!    array referencing those buffers by device address.
//! 3. Allocates a range out of [`Self::clas_storage`] sized via
//!    `vkGetClusterAccelerationStructureBuildSizesNV`.
//! 4. Dispatches `vkCmdBuildClusterAccelerationStructureIndirectNV` (one call
//!    builds every meshlet's CLAS in parallel on the GPU).
//! 5. Reads back per-CLAS device addresses and stashes them in a
//!    [`MeshClusters`] entry.
//!
//! All transient buffers are freed before [`Self::upload_mesh`] returns; only
//! the slice of `clas_storage` holding the CLAS payloads survives. The
//! committed-vs-sparse decision (M-A vs M-A.streaming) is hidden inside
//! [`crate::scene::raw_vk::PersistentBuffer`].

use core::ops::Range;

use ash::vk::{self, TaggedStructure as _};
use bevy_ecs::resource::Resource;

use super::blas::build_blas_for_clusters;
use super::meshlet_loader::{DequantizedMeshlet, DequantizedMeshletMesh};
use super::raw_vk::{PersistentBuffer, RangeAllocator, RawBuffer};

/// Default size of the persistent CLAS storage buffer in M-A: 256 MiB,
/// enough for ~660k 384-byte CLASes (a 1-triangle CLAS occupies 384 bytes
/// on Blackwell). M-A.streaming will lift this via sparse residency.
pub const DEFAULT_CLAS_STORAGE_BYTES: u64 = 256 * 1024 * 1024;

/// Default size of the persistent BLAS storage buffer in M-A: 32 MiB.
/// Cluster BLASes are small (a 1-CLAS BLAS is ~1.5 KiB on Blackwell)
/// because they only carry CLAS-reference indices, not triangles.
pub const DEFAULT_BLAS_STORAGE_BYTES: u64 = 32 * 1024 * 1024;

/// CLAS storage requires this alignment on NVIDIA hardware (matches the
/// `accelerationStructure*` minimum in `VkPhysicalDeviceAccelerationStructurePropertiesKHR`).
/// 256 is the conservative figure; the driver may report less but never more.
pub const CLAS_STORAGE_ALIGNMENT: u64 = 256;

/// Opaque handle for a mesh that's been uploaded through [`ClusterAsManager`].
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct MeshClusterHandle(u32);

/// Per-mesh metadata after a successful CLAS + BLAS upload.
#[derive(Debug)]
pub struct MeshClusters {
    /// Byte range within [`ClusterAsManager::clas_storage`] that holds this
    /// mesh's CLAS payloads.
    pub clas_storage_range: Range<u64>,
    /// One entry per meshlet (in source order). Each is the
    /// `VkDeviceAddress` of the corresponding CLAS, suitable as a
    /// `clusterReferences` entry in a `BUILD_CLUSTERS_BOTTOM_LEVEL` build.
    pub clas_addresses: Vec<u64>,
    /// Byte range within [`ClusterAsManager::blas_storage`] that holds this
    /// mesh's BLAS payload.
    pub blas_storage_range: Range<u64>,
    /// Device address of the cluster-bottom-level BLAS that references all
    /// of [`Self::clas_addresses`]. This is what a TLAS instance's
    /// `accelerationStructureReference` points at.
    pub blas_address: u64,
}

impl MeshClusters {
    pub fn meshlet_count(&self) -> usize {
        self.clas_addresses.len()
    }
}

/// Persistent owner of CLAS storage + per-mesh metadata.
///
/// Inserted into the render world as a [`Resource`] by
/// [`crate::scene::AuroraScenePlugin`]. Methods that need GPU access take an
/// [`ash::Device`] + memory properties + a `wgpu::hal::vulkan::Device` (for
/// the cluster_AS function dispatch path) as parameters; the manager itself
/// stores no Vulkan device handle.
#[derive(Resource)]
pub struct ClusterAsManager {
    clas_storage: PersistentBuffer,
    clas_free_ranges: RangeAllocator,
    blas_storage: PersistentBuffer,
    blas_free_ranges: RangeAllocator,
    meshes: Vec<MeshClusters>,
}

impl ClusterAsManager {
    /// Allocate the persistent CLAS storage buffer (committed memory in M-A).
    ///
    /// # Safety
    ///
    /// `device` and `mem_props` must come from the same physical device, and
    /// the device must remain valid until [`Self::destroy`] is called.
    pub unsafe fn new(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        clas_storage_size: u64,
        blas_storage_size: u64,
    ) -> Self {
        let clas_storage = unsafe {
            PersistentBuffer::alloc(
                "aurora.clas_storage",
                device,
                mem_props,
                clas_storage_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
                false,
            )
        };
        let blas_storage = unsafe {
            PersistentBuffer::alloc(
                "aurora.blas_storage",
                device,
                mem_props,
                blas_storage_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
                false,
            )
        };
        Self {
            clas_storage,
            clas_free_ranges: RangeAllocator::new(clas_storage_size),
            blas_storage,
            blas_free_ranges: RangeAllocator::new(blas_storage_size),
            meshes: Vec::new(),
        }
    }

    /// Returns the previously uploaded mesh entry. Panics on stale handle.
    pub fn mesh(&self, handle: MeshClusterHandle) -> &MeshClusters {
        &self.meshes[handle.0 as usize]
    }

    /// Build CLASes for every meshlet in `mesh` and append the per-meshlet
    /// device addresses to the manager.
    ///
    /// All transient input/output buffers (positions, indices, descriptors,
    /// scratch, dst_addresses, count_buf) are torn down before returning;
    /// only the slice of [`Self::clas_storage`] holding the CLAS payloads
    /// survives.
    ///
    /// # Safety
    ///
    /// - `device`/`mem_props` must come from the same physical device that
    ///   created `wgpu_device`. - `wgpu_device` must have been created with
    ///   `Features::EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE` enabled.
    /// - The caller must not be running other GPU work that races with this
    ///   upload's `vkDeviceWaitIdle`. M-A scope: this is only called at
    ///   asset-load time before any rendering.
    pub unsafe fn upload_mesh(
        &mut self,
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        hal_device: &wgpu::hal::vulkan::Device,
        wgpu_device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &DequantizedMeshletMesh,
    ) -> MeshClusterHandle {
        let meshlet_count = mesh.meshlets.len();
        if meshlet_count == 0 {
            // Empty mesh: still register a handle so callers can index uniformly.
            self.meshes.push(MeshClusters {
                clas_storage_range: 0..0,
                clas_addresses: Vec::new(),
                blas_storage_range: 0..0,
                blas_address: 0,
            });
            return MeshClusterHandle((self.meshes.len() - 1) as u32);
        }

        // ---- Stage positions + indices ---------------------------------------
        let pos_bytes = bytemuck::cast_slice::<[f32; 3], u8>(&mesh.positions);
        let pos_buf = unsafe {
            RawBuffer::alloc(
                device,
                mem_props,
                pos_bytes.len() as u64,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                /* host_visible = */ true,
            )
        };
        unsafe { pos_buf.upload(device, pos_bytes) };

        // Indices are now u32 (widened from the meshlet asset's u8 storage
        // because the cluster_AS NV path requires 32-bit indices).
        let idx_bytes: &[u8] = bytemuck::cast_slice(&mesh.indices);
        let idx_buf = unsafe {
            RawBuffer::alloc(
                device,
                mem_props,
                (idx_bytes.len() as u64).max(1),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                true,
            )
        };
        unsafe { idx_buf.upload(device, idx_bytes) };

        // ---- Build per-cluster descriptors -----------------------------------
        // VkClusterAccelerationStructureBuildTriangleClusterInfoNV layout:
        //   bits  0..9  triangleCount
        //   bits  9..18 vertexCount
        //   bits 18..24 positionTruncateBitCount (0 = no truncation)
        //   bits 24..28 indexType  (1 = 8-bit, 2 = 16-bit, 4 = 32-bit)
        //   bits 28..32 opacityMicromapIndexType (0 = none)
        //
        // NV driver path appears to require 32-bit indices despite the
        // spec listing 8-bit as a supported format -- Unreal's
        // NaniteRayTracingDecodePageClusters.usf also uses IndexFormat=4.
        const INDEX_TYPE_32BIT: u32 = 4;
        // Geometry-flags layout: bits 0..24 geometryIndex, 24..29 reserved,
        // 29..32 geometryFlags. Bit 29 = OPAQUE.
        const OPAQUE_GEOMETRY_FLAGS: u32 = 1 << 29;

        let mut max_tris = 0u32;
        let mut max_verts = 0u32;
        let mut total_tris = 0u32;
        let mut total_verts = 0u32;
        let mut descriptors =
            Vec::with_capacity(meshlet_count);
        for (cluster_id, m) in mesh.meshlets.iter().enumerate() {
            let DequantizedMeshlet {
                vertex_offset,
                vertex_count,
                index_offset,
                triangle_count,
            } = *m;

            // bevy meshlets cap triangle_count + vertex_count at u8::MAX < 0x1FF
            // and we hardcode INDEX_TYPE_8BIT = 1 < 0xF, so the masks are
            // statically no-ops, but we keep them to match the spec field
            // widths (triangleCount:9, vertexCount:9, indexType:4).
            #[expect(
                clippy::identity_op,
                reason = "explicit shifts document the bitfield layout in the spec"
            )]
            let packed_bitfield: u32 = (triangle_count & 0x1FF)
                | ((vertex_count & 0x1FF) << 9)
                | (0u32 << 18)
                | ((INDEX_TYPE_32BIT & 0xF) << 24)
                | (0u32 << 28);

            // Strides are *per-index* (not per-triangle). For 32-bit
            // indices, consecutive indices are 4 bytes apart. Unreal's
            // NaniteRayTracingDecodePageClusters.usf is explicit about
            // this: `IndexBuffer = base + IndexBufferOffset *
            // IndexStrideInBytes`.
            descriptors.push(vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV {
                cluster_id: cluster_id as u32,
                cluster_flags: vk::ClusterAccelerationStructureClusterFlagsNV::default(),
                packed_bitfield,
                base_geometry_index_and_geometry_flags:
                    vk::ClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV(
                        OPAQUE_GEOMETRY_FLAGS,
                    ),
                index_buffer_stride: 4, // 4 bytes per 32-bit index
                vertex_buffer_stride: 12, // f32x3
                geometry_index_and_flags_buffer_stride: 0,
                opacity_micromap_index_buffer_stride: 0,
                // index_offset is now a u32-count; multiply by 4 for the
                // byte-address into the index buffer.
                index_buffer: idx_buf.addr + u64::from(index_offset) * 4,
                vertex_buffer: pos_buf.addr + u64::from(vertex_offset) * 12,
                geometry_index_and_flags_buffer: 0,
                opacity_micromap_array: 0,
                opacity_micromap_index_buffer: 0,
            });

            max_tris = max_tris.max(triangle_count);
            max_verts = max_verts.max(vertex_count);
            total_tris += triangle_count;
            total_verts += vertex_count;
        }

        let desc_bytes_len =
            descriptors.len() * size_of::<vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV>();
        let desc_bytes = unsafe {
            core::slice::from_raw_parts(descriptors.as_ptr().cast::<u8>(), desc_bytes_len)
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

        // ---- Size query ------------------------------------------------------
        // Match Unreal's NaniteRayTracing constants (NaniteRayTracingDefinitions.h):
        //   MaxGeometryIndex     = 15
        //   MaxUniqueGeometries  = 10
        // Aurora's data is single-geometry (geometryIndex == 0), so tighter
        // caps would be valid per spec, but matching the known-working
        // production config eliminates "driver implicitly requires
        // headroom" as a suspect.
        let triangle_input = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .max_geometry_index_value(15)
            .max_cluster_unique_geometry_count(10)
            .max_cluster_triangle_count(max_tris)
            .max_cluster_vertex_count(max_verts)
            .max_total_triangle_count(total_tris)
            .max_total_vertex_count(total_verts);
        // The OpInputNV union takes a `*mut` even though the input is read-only;
        // the driver doesn't write through this pointer.
        let op_input = vk::ClusterAccelerationStructureOpInputNV {
            p_triangle_clusters: core::ptr::from_ref(&triangle_input).cast_mut(),
        };
        let input_info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(meshlet_count as u32)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(op_input);

        let sizes = unsafe { hal_device.get_cluster_build_sizes(&input_info) };

        // ---- Allocate range in clas_storage + transient outputs --------------
        let clas_storage_range = self
            .clas_free_ranges
            .alloc(sizes.acceleration_structure_size, CLAS_STORAGE_ALIGNMENT)
            .unwrap_or_else(|| {
                panic!(
                    "ClusterAsManager out of CLAS storage: requested {} bytes, used {} / {}",
                    sizes.acceleration_structure_size,
                    self.clas_free_ranges.used(),
                    self.clas_free_ranges.capacity(),
                )
            });
        unsafe {
            self.clas_storage
                .ensure_resident(clas_storage_range.clone());
        }

        let scratch = unsafe {
            RawBuffer::alloc(
                device,
                mem_props,
                sizes.build_scratch_size.max(1),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                false,
            )
        };

        let dst_addresses_size = (meshlet_count as u64) * 8;
        let dst_addresses = unsafe {
            RawBuffer::alloc(
                device,
                mem_props,
                dst_addresses_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                true,
            )
        };

        // For IMPLICIT/EXPLICIT_DESTINATIONS modes srcInfosCount is a device
        // address pointing at a 4-byte-aligned u32 holding the actual count
        // (see VUID-VkClusterAccelerationStructureCommandsInfoNV-srcInfosCount-*).
        let count_buf = unsafe {
            RawBuffer::alloc(
                device,
                mem_props,
                4,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                true,
            )
        };
        unsafe { count_buf.upload(device, &(meshlet_count as u32).to_le_bytes()) };

        // ---- Record + submit -------------------------------------------------
        let stride = size_of::<vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV>() as u64;
        let commands_info = vk::ClusterAccelerationStructureCommandsInfoNV {
            s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
            p_next: core::ptr::null_mut(),
            input: input_info,
            dst_implicit_data: self.clas_storage.device_address(clas_storage_range.start),
            scratch_data: scratch.addr,
            dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
                device_address: dst_addresses.addr,
                stride: 8,
                size: dst_addresses_size,
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
            label: Some("aurora.cluster_as.upload_mesh"),
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

        // M-A scope: synchronous upload via vkDeviceWaitIdle. Acceptable at
        // asset-load time. The eventual streaming path will need a fence per
        // upload + a "ready when fence signaled" handshake instead.
        unsafe {
            device.device_wait_idle().expect("device_wait_idle");
        }

        // ---- Read back per-CLAS device addresses -----------------------------
        let mut clas_addresses = vec![0u64; meshlet_count];
        unsafe {
            let ptr = device
                .map_memory(
                    dst_addresses.mem,
                    0,
                    dst_addresses_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("map dst_addresses")
                .cast::<u64>();
            core::ptr::copy_nonoverlapping(ptr, clas_addresses.as_mut_ptr(), meshlet_count);
            device.unmap_memory(dst_addresses.mem);
        }

        // ---- Cleanup transient buffers ---------------------------------------
        unsafe {
            pos_buf.destroy(device);
            idx_buf.destroy(device);
            src_infos.destroy(device);
            scratch.destroy(device);
            dst_addresses.destroy(device);
            count_buf.destroy(device);
        }

        tracing::debug!(
            target: "bevy_aurora",
            meshlets = meshlet_count,
            clas_storage_bytes = sizes.acceleration_structure_size,
            scratch_bytes = sizes.build_scratch_size,
            "uploaded mesh: {} CLASes built into [{:#x}, {:#x})",
            meshlet_count,
            self.clas_storage.device_address(clas_storage_range.start),
            self.clas_storage.device_address(clas_storage_range.end),
        );

        // ---- Build the BLAS that references those CLASes -------------------
        let mut blas = unsafe {
            build_blas_for_clusters(
                device,
                mem_props,
                hal_device,
                wgpu_device,
                queue,
                &mut self.blas_storage,
                &mut self.blas_free_ranges,
                &clas_addresses,
            )
        };

        // Diagnostic / fix for M-B sub-2c: wrap the cluster-built BLAS in a
        // regular KHR-AS handle and use its device address in TLAS instances
        // instead of the cluster_AS-reported `dst_addresses[0]`. The triangle
        // BLAS control test confirmed the KHR-AS-wrapped path produces ray
        // hits; if `dst_addresses[0]` returns a different address (or a
        // non-`accelerationStructureReference`-compatible value), this wrap
        // is the load-bearing fix. Logs both addresses so we can see whether
        // they match.
        let blas_size = blas.storage_range.end - blas.storage_range.start;
        let raw_instance =
            unsafe { hal_device.shared_instance() }.raw_instance();
        unsafe {
            super::blas::wrap_built_blas(
                device,
                raw_instance,
                &self.blas_storage,
                &mut blas,
                blas_size,
            );
        }

        let entry = MeshClusters {
            clas_storage_range,
            clas_addresses,
            blas_storage_range: blas.storage_range,
            blas_address: blas.address,
        };
        let handle = MeshClusterHandle(self.meshes.len() as u32);
        self.meshes.push(entry);
        handle
    }

    /// Free all GPU memory owned by the manager.
    ///
    /// # Safety
    ///
    /// No in-flight GPU work may reference any CLAS / BLAS / TLAS produced
    /// through this manager.
    pub unsafe fn destroy(self, device: &ash::Device) {
        unsafe {
            self.clas_storage.destroy(device);
            self.blas_storage.destroy(device);
        }
    }
}
