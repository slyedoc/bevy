// CLAS arena reaches into raw Vulkan for the cluster-AS build itself
// (`vkCmdBuildClusterAccelerationStructureIndirectNV`) and for the
// per-cluster device-address readback. Every `unsafe` block notes
// the Vulkan rule being honored.
#![allow(unsafe_code)]

//! Static CLAS arena: builds a [`vk::ClusterAccelerationStructure`]
//! per cluster in each [`ClusterMesh`] at upload time and records
//! the resulting per-cluster `VkDeviceAddress`es for later BLAS
//! builds.
//!
//! Design
//! ------
//!
//! - **Storage pool**: one big `wgpu::Buffer` created with raw
//!   Vulkan flags via [`Allocator::create_buffer`] (`AS_STORAGE_KHR`).
//!   Sub-allocated per mesh via [`RangeAllocator`].
//! - **Per-cluster CLAS-build inputs** (positions, indices, build
//!   descriptors): vertex / index data is shared with the
//!   [`ClusterMeshManager`]'s existing `PersistentGpuBuffer` pools —
//!   the AS build references them by device address with per-cluster
//!   offsets, no duplicate upload.
//! - **Build mode**: `ImplicitDestinations` — the driver picks each
//!   CLAS's address inside the per-mesh range we provide; we read
//!   the addresses back via a host-visible staging buffer.
//!   First-version debug shape; will migrate to `ExplicitDestinations`
//!   (CPU-assigned addresses, no readback) once the upload path is
//!   trusted.
//! - **Trigger**: a render-world system in [`RenderSystems::PrepareAssets`]
//!   walks [`ClusterMeshManager::cluster_mesh_slices`] for newly-uploaded
//!   meshes and runs [`ClasArena::upload_mesh`] for each.
//! - **Removal**: TBD — meshes leak in the arena for V1. Add eviction
//!   when [`AssetEvent::Unused`] arrives, matching the manager.

use ash::vk::{self, TaggedStructure};
use bevy_asset::AssetId;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_platform::collections::HashMap;
use bevy_render::renderer::{RenderDevice, RenderQueue};
use core::ops::Range;
use range_alloc::RangeAllocator;
use wgpu::CommandEncoderDescriptor;

use crate::gpu::allocator::{Allocator, SparseBuffer};
use crate::gpu::extension::ClusterExtensionFns;
use super::{Cluster, ClusterIndex, ClusterMesh, ClusterMeshManager};

/// Virtual address space reserved for the CLAS storage arena —
/// 16 GB. Backing memory is committed lazily by sparse binding (see
/// [`SparseBuffer`]); we pay nothing for the unused tail. Picked to
/// be unreachable by any plausible single-scene cluster_AS workload;
/// can grow further once we hit the limit (just bump this constant).
pub const CLAS_POOL_VIRTUAL_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// Virtual address space reserved for the global per-cluster CLAS
/// device-address table — 1 GB → 128 M cluster slots (8 B each).
/// Same sparse-backed pattern as the storage arena.
pub const CLUSTER_CLAS_ADDRESSES_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;

/// Conservative NV CLAS storage alignment
/// (`clusterByteAlignment` reported in
/// `VkPhysicalDeviceClusterAccelerationStructurePropertiesNV`).
/// 256 is the worst-case across all NV CLAS path requirements.
pub const CLAS_STORAGE_ALIGN: u64 = 256;

/// Conservative scratch alignment (`clusterScratchByteAlignment`).
pub const CLAS_SCRATCH_ALIGN: u64 = 256;

/// Per-mesh CLAS-build record kept after upload. Used downstream by
/// the selector → BLAS pipeline (cluster device addresses live in
/// [`ClasArena::cluster_clas_addresses`] indexed by
/// `cluster_base + local_cluster_id`).
#[derive(Debug)]
pub struct ClasMeshEntry {
    /// Byte range inside [`ClasArena::storage`] holding this mesh's
    /// CLAS payloads. Driver-chosen by `ImplicitDestinations`.
    pub storage_range: Range<u64>,
    /// Cluster count — sufficient to free this mesh's slot in the
    /// global address table on eviction. Per-cluster addresses
    /// themselves live only on GPU.
    pub cluster_count: u32,
}

/// Render-world resource owning the CLAS storage pool, per-mesh
/// metadata, and the global per-cluster device-address table.
///
/// Both buffers are [`SparseBuffer`]s — they reserve a large virtual
/// address range but pay only for committed pages. Buffer handles +
/// device addresses stay stable across growth: bind groups don't
/// rebuild, recorded CLAS device addresses don't go stale, no
/// resize-copy.
#[derive(Resource)]
pub struct ClasArena {
    /// Sparse-bound AS-storage buffer holding every cluster's CLAS
    /// payload. Created with `ACCELERATION_STRUCTURE_STORAGE_KHR`
    /// since wgpu doesn't expose that flag. Per-mesh sub-ranges are
    /// tracked CPU-side by [`Self::allocator`]; sparse pages backing
    /// them are committed at upload time via [`SparseBuffer::commit`].
    pub storage: SparseBuffer,
    /// Sub-allocator over [`Self::storage`]'s virtual range. Returns
    /// byte ranges for per-mesh CLAS payloads.
    allocator: RangeAllocator<u64>,
    /// Per-mesh entries: storage range + cluster count.
    meshes: HashMap<AssetId<ClusterMesh>, ClasMeshEntry>,
    /// Sparse-bound global per-cluster CLAS device-address table —
    /// indexed by global cluster id (`cluster_base + local_id`).
    /// The selector reads `cluster_clas_addresses[cluster_base +
    /// local_id]` to resolve the cluster's CLAS device address for
    /// the BLAS-build input list. WGSL view is `array<vec2<u32>>`
    /// (no native u64). Filled GPU-side via `copy_buffer_to_buffer`
    /// from each CLAS build's `dst_addresses_buf`.
    pub cluster_clas_addresses: SparseBuffer,
}

impl ClasArena {
    pub fn new(render_device: &RenderDevice, allocator: &Allocator) -> Self {
        let storage = allocator.create_sparse_buffer(
            render_device,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            // wgpu's tracker only sees this as opaque storage — we
            // never bind it to a shader. COPY_DST keeps wgpu's barrier
            // tracking happy if anything ever copies into it.
            wgpu::BufferUsages::COPY_DST,
            CLAS_POOL_VIRTUAL_BYTES,
            "clas_arena.storage",
        );

        let cluster_clas_addresses = allocator.create_sparse_buffer(
            render_device,
            // Buffer is read by the selector compute (storage) and
            // written by CLAS-build's `copy_buffer_to_buffer` (COPY_DST).
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            CLUSTER_CLAS_ADDRESSES_VIRTUAL_BYTES,
            "clas_arena.cluster_clas_addresses",
        );

        Self {
            storage,
            allocator: RangeAllocator::new(0..CLAS_POOL_VIRTUAL_BYTES),
            meshes: HashMap::default(),
            cluster_clas_addresses,
        }
    }

    /// Lookup the per-mesh CLAS record for `asset_id`. `None` if the
    /// mesh hasn't been CLAS-uploaded yet (either still pending or
    /// the asset doesn't exist).
    pub fn mesh(&self, asset_id: AssetId<ClusterMesh>) -> Option<&ClasMeshEntry> {
        self.meshes.get(&asset_id)
    }

    /// Build CLAS for every cluster in `mesh` and record the
    /// per-cluster device addresses. The 24-bit `geometry_index`
    /// slot of each CLAS carries the **global** cluster id
    /// (`cluster_base + local_id`) so ray hits report
    /// `RayIntersection.geometry_index = global_cluster_id`,
    /// matching the aurora convention.
    ///
    /// `vertex_buffer_addr` / `index_buffer_addr` are the
    /// `VkDeviceAddress`es of [`ClusterMeshManager`]'s shared
    /// `vertex_positions` / `indices` `PersistentGpuBuffer`s; each
    /// cluster contributes its own per-cluster `vertex_offset` /
    /// `index_offset` (mesh-local pool slots — already rebased
    /// to global pool slots by the manager).
    ///
    /// # Panics
    ///
    /// Panics if cluster_AS extension function table is missing, the
    /// arena is exhausted, the build submission fails, or readback
    /// mapping fails. All setup-time failure modes.
    pub fn upload_mesh(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        allocator: &Allocator,
        fns: &ClusterExtensionFns,
        asset_id: AssetId<ClusterMesh>,
        clusters: &[Cluster],
        cluster_base: ClusterIndex,
        vertex_buffer_addr: vk::DeviceAddress,
        index_buffer_addr: vk::DeviceAddress,
        vertex_base: u32,
        index_base: u32,
    ) {
        if self.meshes.contains_key(&asset_id) {
            return;
        }
        let cluster_count = clusters.len();
        if cluster_count == 0 {
            self.meshes.insert(
                asset_id,
                ClasMeshEntry {
                    storage_range: 0..0,
                    cluster_count: 0,
                },
            );
            return;
        }
        let cluster_fns = fns
            .cluster
            .as_ref()
            .expect("clas_arena.upload_mesh: cluster-AS extension function table missing");

        // 1. Per-cluster build descriptors. Cluster ids: cluster_id
        //    is mesh-local; geometry_index is global (matches aurora).
        //    NV index type 4 = 32-bit indices (the only index type the
        //    cluster path supports in our pipeline). OPAQUE = 0b100 in
        //    the 3-bit geometry-flags subfield — see memory note
        //    `aurora_cluster_as_opaque.md`.
        const INDEX_TYPE_32BIT: u32 = 4;
        const OPAQUE_GEOMETRY_FLAG: u8 = 0b100;

        let (mut max_tris, mut max_verts) = (0u32, 0u32);
        let (mut total_tris, mut total_verts) = (0u32, 0u32);
        let mut max_global_cluster_id = 0u32;
        let mut descriptors: Vec<vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV> =
            Vec::with_capacity(cluster_count);
        for (local_id, cluster) in clusters.iter().enumerate() {
            let Cluster {
                vertex_offset,
                vertex_count,
                index_offset,
                triangle_count,
                ..
            } = *cluster;
            let global_id = cluster_base.0.wrapping_add(local_id as u32);

            descriptors.push(vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV {
                // Global cluster id so the RT-pipeline hit shader's
                // `@builtin(cluster_id)` (ClusterIDNV) indexes `clusters[]`
                // directly. The ray-query path uses the baked `base_geometry_index`
                // instead, so this is free for it.
                cluster_id: global_id,
                cluster_flags: vk::ClusterAccelerationStructureClusterFlagsNV::default(),
                triangle_cluster_info_packed: vk::Packed9_9_6_4_4::new(
                    triangle_count,
                    vertex_count,
                    0, // position_truncate_bit_count
                    INDEX_TYPE_32BIT,
                    0, // opacity_micromap_index_type
                ),
                base_geometry_index_and_geometry_flags:
                    vk::ClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV {
                        geometry_index_and_geometry_flags: vk::Packed24_5_3::new(
                            global_id,
                            OPAQUE_GEOMETRY_FLAG,
                        ),
                    },
                index_buffer_stride: 4,
                vertex_buffer_stride: 12,
                geometry_index_and_flags_buffer_stride: 0,
                opacity_micromap_index_buffer_stride: 0,
                // `cluster.vertex_offset` / `index_offset` are
                // MESH-LOCAL — the global-slot rebase only happens
                // during the GPU upload of `clusters` (see
                // `PersistentGpuBufferable` for `Arc<[Cluster]>`).
                // The CPU-side `clusters` slice we iterate here still
                // holds the local values, so we must add the pool
                // bases ourselves before turning them into absolute
                // device addresses for the CLAS build. Without this,
                // every mesh past the first reads vertex / index
                // data from mesh 0's region and the resulting CLAS
                // contains the wrong triangles.
                index_buffer: index_buffer_addr
                    + u64::from(index_base + index_offset) * 4,
                vertex_buffer: vertex_buffer_addr
                    + u64::from(vertex_base + vertex_offset) * 12,
                geometry_index_and_flags_buffer: 0,
                opacity_micromap_array: 0,
                opacity_micromap_index_buffer: 0,
            });

            max_tris = max_tris.max(triangle_count);
            max_verts = max_verts.max(vertex_count);
            total_tris += triangle_count;
            total_verts += vertex_count;
            max_global_cluster_id = max_global_cluster_id.max(global_id);
        }

        // 2. Upload src_infos (cluster build descriptors).
        let desc_stride =
            size_of::<vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV>() as u64;
        let desc_bytes_len = (descriptors.len() as u64) * desc_stride;
        // SAFETY: ClusterAccelerationStructureBuildTriangleClusterInfoNV
        // is repr(C) POD; a flat byte view is well-defined for upload.
        let desc_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                descriptors.as_ptr().cast::<u8>(),
                desc_bytes_len as usize,
            )
        };
        let src_infos_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clas_arena.src_infos"),
            size: desc_bytes_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&src_infos_buf, 0, desc_bytes);
        let src_infos_addr = allocator.wgpu_buffer_device_address(&src_infos_buf);

        let count_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clas_arena.src_infos_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&count_buf, 0, &(cluster_count as u32).to_le_bytes());

        // 3. Build-shape size query.
        let mut triangle_input =
            vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .max_geometry_index_value(max_global_cluster_id)
                .max_cluster_unique_geometry_count(1)
                .max_cluster_triangle_count(max_tris)
                .max_cluster_vertex_count(max_verts)
                .max_total_triangle_count(total_tris)
                .max_total_vertex_count(total_verts)
                .min_position_truncate_bit_count(0);
        let op_input = vk::ClusterAccelerationStructureOpInputNV {
            p_triangle_clusters: &mut triangle_input as *mut _,
        };
        // ALLOW_DATA_ACCESS so the closest-hit can read the hit triangle's three
        // object-space vertex positions back from the CLAS via
        // `@builtin(hit_triangle_vertex_positions)` (VK_KHR_ray_tracing_position_fetch,
        // part of the required RTX feature set). The vertices live in the CLAS — not
        // the cluster-referencing BLAS — so the flag belongs on this build. The same
        // flags drive the size query and the build (the size depends on them):
        // `size_input` is reused as `cmd_info.input` below.
        let size_input = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(cluster_count as u32)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
            )
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(op_input);
        let mut sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: size_input is fully populated; the cluster_AS function
        // table is loaded (checked above). The query only reads inputs
        // and writes back to sizes_info.
        unsafe {
            cluster_fns.get_cluster_acceleration_structure_build_sizes(
                &size_input,
                &mut sizes_info,
            );
        }

        // 4. Allocate CLAS storage range + scratch + dst_addresses.
        let storage_range = self
            .allocator
            .allocate_range_aligned(
                sizes_info.acceleration_structure_size,
                CLAS_STORAGE_ALIGN,
            )
            .expect("clas_arena: storage virtual-address space exhausted");

        // Commit the sparse pages covering this mesh's range BEFORE
        // we record the build. sparse-bind ops aren't queue-ordered
        // against subsequent submits without explicit sync, so
        // `commit` blocks on a fence until the binding lands.
        self.storage.commit(storage_range.clone());

        // Same for the global per-cluster device-address table — its
        // slice [cluster_base*8 .. (cluster_base+cluster_count)*8]
        // gets copy_buffer_to_buffer-d into below.
        let addr_byte_start = (cluster_base.0 as u64) * 8;
        let addr_byte_end = addr_byte_start + (cluster_count as u64) * 8;
        self.cluster_clas_addresses
            .commit(addr_byte_start..addr_byte_end);

        let scratch_size = sizes_info.build_scratch_size.max(1);
        // Allocator routes the buffer through `vkCreateBuffer` with
        // `SHADER_DEVICE_ADDRESS_BIT` so `vkGetBufferDeviceAddress`
        // succeeds (wgpu's standard buffer flags don't include SDA
        // unless BLAS_INPUT is set, which scratch can't be).
        let scratch_buf = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            wgpu::BufferUsages::STORAGE,
            scratch_size + CLAS_SCRATCH_ALIGN - 1,
            crate::gpu::allocator::MemoryLocation::GpuOnly,
            "clas_arena.scratch",
        );
        let scratch_addr_base = allocator.wgpu_buffer_device_address(&scratch_buf);
        let scratch_misalign = scratch_addr_base & (CLAS_SCRATCH_ALIGN - 1);
        let scratch_offset = if scratch_misalign == 0 {
            0
        } else {
            CLAS_SCRATCH_ALIGN - scratch_misalign
        };
        let scratch_addr = scratch_addr_base + scratch_offset;

        let dst_addresses_size = (cluster_count as u64) * 8;
        // VUID-vkCmdBuildClusterAccelerationStructureIndirectNV-pCommandInfos-10459
        // requires the buffer behind `dstAddressesArray` to carry
        // `ACCELERATION_STRUCTURE_STORAGE_KHR`. SDA from Allocator
        // satisfies the device-address query.
        let dst_addresses_buf = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::TRANSFER_SRC,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            dst_addresses_size,
            crate::gpu::allocator::MemoryLocation::GpuOnly,
            "clas_arena.dst_addresses",
        );
        let dst_addresses_addr = allocator.wgpu_buffer_device_address(&dst_addresses_buf);

        // 5. Encode + submit the cluster_AS build + address-table copy.
        let mut encoder =
            render_device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("clas_arena.upload_mesh"),
            });

        let dst_implicit_addr = self.storage.address + storage_range.start;
        let cmd_info = vk::ClusterAccelerationStructureCommandsInfoNV {
            s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
            p_next: core::ptr::null_mut(),
            input: size_input,
            dst_implicit_data: dst_implicit_addr,
            scratch_data: scratch_addr,
            dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
                device_address: dst_addresses_addr,
                stride: 8,
                size: dst_addresses_size,
            },
            dst_sizes_array: vk::StridedDeviceAddressRegionKHR::default(),
            src_infos_array: vk::StridedDeviceAddressRegionKHR {
                device_address: src_infos_addr,
                stride: desc_stride,
                size: desc_bytes_len,
            },
            src_infos_count: allocator.wgpu_buffer_device_address(&count_buf),
            address_resolution_flags:
                vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
            _marker: core::marker::PhantomData,
        };

        // SAFETY: the cluster-AS function table is loaded (checked
        // above); the wgpu encoder is open and Vulkan-backed; the
        // descriptor strides + addresses point at the buffers we just
        // created. We do NOT touch this encoder via wgpu commands
        // afterward — wgpu refuses to mix raw + high-level encoding.
        unsafe {
            crate::gpu::extension::cmd_build_cluster_acceleration_structures_indirect(
                &mut encoder,
                fns,
                &cmd_info,
            );
            // AS_WRITE → TRANSFER_READ + SHADER_READ barrier so the
            // follow-up `copy_buffer_to_buffer` (and later selector
            // reads of `cluster_clas_addresses`) sees the freshly
            // written per-cluster CLAS device addresses.
            // wgpu's tracker can't insert this because the build
            // happened via raw VK on a buffer wgpu only sees as
            // STORAGE.
            crate::gpu::extension::cmd_global_as_barrier(&mut encoder, &render_device, false);
        }
        let build_idx = render_queue.submit([encoder.finish()]);
        // Block the CPU until the raw-VK cluster build finishes so the
        // wgpu copy below sees the freshly written dst_addresses. The
        // in-encoder AS-barrier we already emit covers same-encoder
        // visibility but doesn't reach across the submit boundary into
        // wgpu's separately tracked copy encoder; without this wait,
        // the copy reads stale zeros, the global address table stays
        // empty, BLAS rebuild dereferences NULL CLAS pointers, and the
        // GPU faults (ERROR_DEVICE_LOST).
        //
        // TODO(cluster-as): replace with an inter-encoder pipeline
        // barrier on the same submit (e.g. by routing the copy through
        // raw VK as well) so per-mesh uploads don't block the render
        // thread.
        {
            let _span = tracing::info_span!("clas.build_poll_wait").entered();
            let _ = render_device.wgpu_device().poll(wgpu::PollType::Wait {
                submission_index: Some(build_idx),
                timeout: None,
            });
        }

        // Second encoder for the dst_addresses → global table copy.
        // Separate-encoder pattern keeps the raw build encoder pure
        // (no wgpu commands) and the copy encoder pure (no raw VK);
        // queue submission order serializes execution on GPU.
        let mut copy_encoder =
            render_device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("clas_arena.copy_addresses"),
            });
        copy_encoder.copy_buffer_to_buffer(
            &dst_addresses_buf,
            0,
            &self.cluster_clas_addresses.wgpu_buffer,
            addr_byte_start,
            dst_addresses_size,
        );
        render_queue.submit([copy_encoder.finish()]);

        self.meshes.insert(
            asset_id,
            ClasMeshEntry {
                storage_range,
                cluster_count: cluster_count as u32,
            },
        );
    }
}

/// `RenderStartup` system: insert [`ClasArena`] iff [`Allocator`] is
/// present (i.e. cluster-AS extension was enabled on the device).
pub fn init_clas_arena(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
) {
    let Some(allocator) = allocator else {
        return;
    };
    commands.insert_resource(ClasArena::new(&render_device, &allocator));
}

/// `Render::PrepareAssets` system: drain
/// [`ClusterMeshManager::pending_clas_uploads`] and build CLAS for
/// each freshly-uploaded mesh. Runs after
/// [`crate::geometry::perform_pending_cluster_mesh_writes`] —
/// the vertex / index buffers land in submit order, so the CLAS
/// build's encoder (which gets submitted later in the frame) sees
/// the uploaded data on GPU.
pub fn upload_pending_clas(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    allocator: Option<Res<Allocator>>,
    fns: Option<Res<ClusterExtensionFns>>,
    mut cluster_meshes: ResMut<ClusterMeshManager>,
    clas_arena: Option<ResMut<ClasArena>>,
) {
    let (Some(allocator), Some(fns), Some(mut clas_arena)) = (allocator, fns, clas_arena)
    else {
        // No cluster-AS support (or arena not initialized). Drop any
        // queued uploads on the floor — they'd never get serviced.
        cluster_meshes.pending_clas_uploads.clear();
        return;
    };
    if fns.cluster.is_none() || cluster_meshes.pending_clas_uploads.is_empty() {
        cluster_meshes.pending_clas_uploads.clear();
        return;
    }

    let vertex_addr =
        allocator.wgpu_buffer_device_address(cluster_meshes.vertex_positions.buffer());
    let index_addr =
        allocator.wgpu_buffer_device_address(cluster_meshes.indices.buffer());

    let pending = std::mem::take(&mut cluster_meshes.pending_clas_uploads);
    for entry in pending {
        clas_arena.upload_mesh(
            &render_device,
            &render_queue,
            &allocator,
            &fns,
            entry.asset_id,
            &entry.clusters,
            entry.cluster_base,
            vertex_addr,
            index_addr,
            entry.vertex_base,
            entry.index_base,
        );
    }
}
