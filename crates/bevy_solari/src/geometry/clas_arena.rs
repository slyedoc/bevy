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

use crate::gpu::allocator::{Allocator, MemoryLocation, SparseBuffer};
use crate::gpu::extension::ClusterExtensionFns;
use super::mesh_manager::OmmUploadData;
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
    /// Built opacity micro-map for this mesh (alpha-cutout meshes only). Keeps
    /// the `VkMicromapEXT` + its backing/index buffers alive for as long as the
    /// CLAS references them; `None` for opaque meshes.
    pub omm: Option<MeshOmm>,
}

/// A built opacity micro-map kept alive for a mesh's lifetime. The CLAS
/// references `backing` (via `opacity_micromap_array`) at traversal time and the
/// per-triangle `index` at build time, so both buffers must outlive the CLAS.
//
// TODO(omm): `micromap` (a `VkMicromapEXT` handle) is never destroyed — it leaks
// on mesh eviction. Acceptable while meshes live for the app lifetime (the arena
// has no eviction yet, see module docs); add `vkDestroyMicromapEXT` alongside
// CLAS eviction.
#[derive(Debug)]
pub struct MeshOmm {
    pub micromap: vk::MicromapEXT,
    /// Micro-map array storage (referenced by `opacity_micromap_array`).
    pub backing: wgpu::Buffer,
    /// Per-triangle OMM index buffer (referenced by `opacity_micromap_index_buffer`).
    pub index: wgpu::Buffer,
}

/// All buffers + handle for one mesh's opacity micro-map, produced by
/// [`ClasArena::create_micromap`] before the CLAS build records its build. The
/// `*_addr` fields feed `VkMicromapBuildInfoEXT` / the per-cluster descriptors.
/// The build inputs + scratch are transients (kept alive only until the build
/// submit drains); [`Self::into_mesh_omm`] discards them and keeps the parts the
/// CLAS references for its lifetime.
struct MicromapBuild {
    micromap: vk::MicromapEXT,
    backing: wgpu::Buffer,
    backing_addr: vk::DeviceAddress,
    index: wgpu::Buffer,
    index_addr: vk::DeviceAddress,
    _array_input: wgpu::Buffer,
    array_input_addr: vk::DeviceAddress,
    _descs_input: wgpu::Buffer,
    descs_addr: vk::DeviceAddress,
    _scratch: wgpu::Buffer,
    scratch_addr: vk::DeviceAddress,
    usage_counts: Vec<vk::MicromapUsageEXT>,
}

impl MicromapBuild {
    fn into_mesh_omm(self) -> MeshOmm {
        MeshOmm {
            micromap: self.micromap,
            backing: self.backing,
            index: self.index,
        }
    }
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

    /// Evict a removed/modified mesh's CLAS so a later upload (incl. a recycled
    /// `AssetId`) rebuilds instead of hitting the `contains_key` skip and serving
    /// a stale CLAS over reused pool memory. Frees its storage-pool range.
    pub fn remove(&mut self, asset_id: &AssetId<ClusterMesh>) {
        if let Some(entry) = self.meshes.remove(asset_id) {
            if entry.storage_range.end > entry.storage_range.start {
                self.allocator.free_range(entry.storage_range);
            }
        }
    }

    /// Create + size (not yet record the build of) one mesh's opacity micro-map:
    /// uploads the baked array / descriptor / per-triangle-index inputs, queries
    /// the build sizes, allocates the backing + scratch buffers, and creates the
    /// `VkMicromapEXT` handle. The caller records `vkCmdBuildMicromapsEXT` into
    /// the CLAS encoder (so it lands in the same submit, before the CLAS build).
    fn create_micromap(
        &self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        allocator: &Allocator,
        fns: &ClusterExtensionFns,
        omm: &OmmUploadData,
    ) -> MicromapBuild {
        let omm_fns = fns
            .opacity_micromap
            .as_ref()
            .expect("clas_arena.create_micromap: opacity-micromap extension not enabled");

        // Build-input buffers need `MICROMAP_BUILD_INPUT_READ_ONLY_EXT` (wgpu
        // can't express it → allocator path); `write_buffer` stages the upload
        // into the device-local buffer, ordered before the raw build (same as the
        // existing `src_infos` upload).
        let make_input = |bytes: &[u8], label: &'static str| {
            // `write_buffer` requires the copy size to respect `COPY_BUFFER_ALIGNMENT`
            // (4). `array_data` is a raw byte blob of arbitrary length; pad up. The
            // micromap build only reads bytes its `descArray` references, so trailing
            // zero padding is inert.
            let padded_len = bytes.len().next_multiple_of(4).max(4);
            // TRANSFER_DST must be on the VK usage (the `wgpu_usage` arg only sets the
            // wgpu wrapper's view) so `write_buffer`'s staged copy is valid.
            let buf = allocator.create_buffer(
                render_device,
                vk::BufferUsageFlags::MICROMAP_BUILD_INPUT_READ_ONLY_EXT
                    | vk::BufferUsageFlags::TRANSFER_DST,
                wgpu::BufferUsages::COPY_DST,
                padded_len as u64,
                MemoryLocation::GpuOnly,
                label,
            );
            if bytes.len() == padded_len {
                render_queue.write_buffer(&buf, 0, bytes);
            } else {
                let mut padded = bytes.to_vec();
                padded.resize(padded_len, 0);
                render_queue.write_buffer(&buf, 0, &padded);
            }
            let addr = allocator.wgpu_buffer_device_address(&buf);
            (buf, addr)
        };
        let (array_input, array_input_addr) = make_input(&omm.array_data[..], "omm.array_input");
        let (descs_input, descs_addr) =
            make_input(bytemuck::cast_slice(&omm.descs[..]), "omm.descs");
        let (index, index_addr) = make_input(bytemuck::cast_slice(&omm.index[..]), "omm.index");

        let usage_counts: Vec<vk::MicromapUsageEXT> = omm
            .usage
            .iter()
            .map(|u| {
                vk::MicromapUsageEXT::default()
                    .count(u.count)
                    .subdivision_level(u.subdivision_level as u32)
                    .format(u.format as u32)
            })
            .collect();

        // Size query: only the usage histogram determines the sizes.
        let size_info_in = vk::MicromapBuildInfoEXT::default()
            .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP)
            .flags(vk::BuildMicromapFlagsEXT::PREFER_FAST_TRACE)
            .mode(vk::BuildMicromapModeEXT::BUILD)
            .usage_counts(&usage_counts);
        let mut sizes = vk::MicromapBuildSizesInfoEXT::default();
        // SAFETY: omm_fns loaded (checked); size_info_in fully populated; the
        // query only reads inputs and writes `sizes`. ash exposes only the raw
        // fp for this extension, so we call it directly (device handle + ptrs).
        unsafe {
            (omm_fns.fp().get_micromap_build_sizes_ext)(
                omm_fns.device(),
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &size_info_in,
                &mut sizes,
            );
        }

        // Backing storage (referenced at trace via `opacity_micromap_array`) +
        // the micromap handle on it.
        let (backing, backing_raw) = allocator.create_buffer_raw(
            render_device,
            vk::BufferUsageFlags::MICROMAP_STORAGE_EXT,
            wgpu::BufferUsages::STORAGE,
            sizes.micromap_size,
            MemoryLocation::GpuOnly,
            "omm.backing",
        );
        let backing_addr = allocator.wgpu_buffer_device_address(&backing);
        let create_info = vk::MicromapCreateInfoEXT::default()
            .buffer(backing_raw)
            .offset(0)
            .size(sizes.micromap_size)
            .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP);
        // SAFETY: create_info references the backing buffer just created at a
        // valid offset/size; omm_fns is loaded. Raw fp (no safe wrapper).
        let mut micromap = vk::MicromapEXT::null();
        let res = unsafe {
            (omm_fns.fp().create_micromap_ext)(
                omm_fns.device(),
                &create_info,
                core::ptr::null(),
                &mut micromap,
            )
        };
        assert_eq!(
            res,
            vk::Result::SUCCESS,
            "clas_arena.create_micromap: vkCreateMicromapEXT failed: {res:?}"
        );

        // Micromap-build scratch must be aligned (minAccelerationStructureScratchOffsetAlignment,
        // ≤256 on NV). Over-allocate by the alignment and round the device address up —
        // an unaligned scratch makes vkCmdBuildMicromapsEXT emit garbage (all-unknown).
        const OMM_SCRATCH_ALIGN: u64 = 256;
        let scratch = allocator.create_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            wgpu::BufferUsages::STORAGE,
            sizes.build_scratch_size.max(1) + OMM_SCRATCH_ALIGN - 1,
            MemoryLocation::GpuOnly,
            "omm.scratch",
        );
        let scratch_base = allocator.wgpu_buffer_device_address(&scratch);
        let scratch_addr = scratch_base.next_multiple_of(OMM_SCRATCH_ALIGN);

        // tracing::info!(
        //     "clas: micromap built — array_in={}B descs={} per_tri_idx={} micromap_size={} backing_addr={:#x}",
        //     omm.array_data.len(),
        //     omm.descs.len(),
        //     omm.index.len(),
        //     sizes.micromap_size,
        //     backing_addr,
        // );

        MicromapBuild {
            micromap,
            backing,
            backing_addr,
            index,
            index_addr,
            _array_input: array_input,
            array_input_addr,
            _descs_input: descs_input,
            descs_addr,
            _scratch: scratch,
            scratch_addr,
            usage_counts,
        }
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
    ///
    /// `omm` (when `Some`) attaches a per-mesh opacity micro-map: a single
    /// `VkMicromapEXT` is built from the baked array + descriptors and referenced
    /// by every cluster's build descriptor, so the RT cores skip `ahit_alpha` on
    /// resolved opaque/transparent micro-regions.
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
        omm: Option<&OmmUploadData>,
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
                    omm: None,
                },
            );
            return;
        }
        let cluster_fns = fns
            .cluster
            .as_ref()
            .expect("clas_arena.upload_mesh: cluster-AS extension function table missing");

        // 0. Opacity micro-map: build the per-mesh `VkMicromapEXT` + its input /
        //    backing / index buffers up front so the per-cluster descriptors can
        //    reference its device addresses. The build itself is recorded into the
        //    same encoder as the CLAS build below (with a barrier between).
        let omm_build = omm.map(|o| self.create_micromap(render_device, render_queue, allocator, fns, o));
        if omm_build.is_some() {
            // tracing::info!(
            //     "clas: OMM attached to {asset_id:?} ({cluster_count} clusters)"
            // );
        }

        // 1. Per-cluster build descriptors. Cluster ids: cluster_id
        //    is mesh-local; geometry_index is global (matches aurora).
        //    NV index type 4 = 32-bit indices (the only index type the
        //    cluster path supports in our pipeline). OPAQUE = 0b100 in
        //    the 3-bit geometry-flags subfield — see memory note
        //    `aurora_cluster_as_opaque.md`.
        const INDEX_TYPE_32BIT: u32 = 4;
        const OPAQUE_GEOMETRY_FLAG: u8 = 0b100;
        // OMM per-triangle index is 32-bit signed (i32, negative = special index).
        // The packed `opacity_micromap_index_type` subfield uses the same NV index
        // encoding as the triangle index type (4 = 32-bit). 4-byte stride.
        const OMM_INDEX_TYPE_32BIT: u32 = 4;
        const OMM_INDEX_STRIDE: u16 = 4;

        // OMM clusters must NOT be flagged OPAQUE: an opaque geometry commits in
        // hardware WITHOUT consulting the micromap, so its transparent/unknown
        // micro-regions would be ignored (card renders solid). The OMM provides the
        // per-micro-triangle opacity instead. Non-OMM clusters keep OPAQUE for the
        // RTCore fast path (any-hit is then driven by the instance FORCE_NO_OPAQUE).
        let geometry_flags: u8 = if omm_build.is_some() {
            0
        } else {
            OPAQUE_GEOMETRY_FLAG
        };
        // Per-cluster OMM opt-in. The NV cluster build REJECTS per-cluster
        // opacity_micromap_array unless the cluster declares OMM participation
        // (validation: CLUSTER_OP_OMM_NOT_ALLOWED). The only OMM cluster flag is
        // ALLOW_DISABLE_OPACITY_MICROMAPS — setting it marks the cluster as
        // OMM-bearing (and lets an instance disable it via the disable flag).
        let cluster_flags = if omm_build.is_some() {
            vk::ClusterAccelerationStructureClusterFlagsNV::ALLOW_DISABLE_OPACITY_MICROMAPS
        } else {
            vk::ClusterAccelerationStructureClusterFlagsNV::default()
        };

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

            // OMM attach (this cluster's slice of the per-mesh micro-map). The
            // OMM index buffer is mesh-local (one i32 per triangle, parallel to
            // the mesh's index buffer / 3), so the cluster's triangle base is its
            // mesh-local `index_offset / 3`.
            let (omm_array_addr, omm_index_addr, omm_index_type, omm_index_stride) =
                match &omm_build {
                    Some(b) => (
                        b.backing_addr,
                        b.index_addr + u64::from(index_offset / 3) * u64::from(OMM_INDEX_STRIDE),
                        OMM_INDEX_TYPE_32BIT,
                        OMM_INDEX_STRIDE,
                    ),
                    None => (0, 0, 0, 0),
                };

            descriptors.push(vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV {
                // Global cluster id so the RT-pipeline hit shader's
                // `@builtin(cluster_id)` (ClusterIDNV) indexes `clusters[]`
                // directly. The ray-query path uses the baked `base_geometry_index`
                // instead, so this is free for it.
                cluster_id: global_id,
                cluster_flags,
                triangle_cluster_info_packed: vk::Packed9_9_6_4_4::new(
                    triangle_count,
                    vertex_count,
                    0, // position_truncate_bit_count
                    INDEX_TYPE_32BIT,
                    omm_index_type,
                ),
                base_geometry_index_and_geometry_flags:
                    vk::ClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV {
                        geometry_index_and_geometry_flags: vk::Packed24_5_3::new(
                            global_id,
                            geometry_flags,
                        ),
                    },
                index_buffer_stride: 4,
                vertex_buffer_stride: 12,
                geometry_index_and_flags_buffer_stride: 0,
                opacity_micromap_index_buffer_stride: omm_index_stride,
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
                opacity_micromap_array: omm_array_addr,
                opacity_micromap_index_buffer: omm_index_addr,
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
        // The cluster build operation must DECLARE OMM participation (validation:
        // CLUSTER_OP_OMM_NOT_ALLOWED) or it rejects the per-cluster
        // opacity_micromap_array. The build-op opt-in is on the InputInfo flags
        // (shared by the size query + build); ALLOW_DISABLE_OPACITY_MICROMAPS_EXT
        // marks the build as OMM-bearing. Only set it for OMM meshes so non-OMM
        // builds keep the lean flag set.
        let mut build_flags = vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
            | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS;
        if omm_build.is_some() {
            // OMM opt-in for the cluster build (else CLUSTER_OP_OMM_NOT_ALLOWED).
            // Use ONLY the lightest "update" flag — DATA_UPDATE (0x100) reserves
            // space to rewrite the whole OMM and explodes the CLAS/BLAS size
            // (saw 3.4 GB); our OMM is static (baked offline), no update needed.
            build_flags |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_OPACITY_MICROMAP_UPDATE_EXT;
            static LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::debug!("clas: OMM build_flags = {:#x}", build_flags.as_raw());
            }
        }
        let size_input = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(cluster_count as u32)
            .flags(build_flags)
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
            MemoryLocation::GpuOnly,
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
            MemoryLocation::GpuOnly,
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
            // Build the opacity micro-map FIRST, then barrier so the cluster
            // build (which references it via `opacity_micromap_array`) sees the
            // finished data. Same encoder/submit → ordered before the CLAS build.
            if let Some(b) = &omm_build {
                let build_info = vk::MicromapBuildInfoEXT::default()
                    .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP)
                    .flags(vk::BuildMicromapFlagsEXT::PREFER_FAST_TRACE)
                    .mode(vk::BuildMicromapModeEXT::BUILD)
                    .usage_counts(&b.usage_counts)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: b.array_input_addr,
                    })
                    .triangle_array(vk::DeviceOrHostAddressConstKHR {
                        device_address: b.descs_addr,
                    })
                    .triangle_array_stride(size_of::<super::asset::OmmDesc>() as u64)
                    .dst_micromap(b.micromap)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: b.scratch_addr,
                    });
                crate::gpu::extension::cmd_build_micromaps(&mut encoder, fns, &build_info);
                crate::gpu::extension::cmd_micromap_barrier(&mut encoder, &render_device);
            }
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

        // Build transients (input/descs/scratch) are now safe to drop after the
        // poll-wait above; keep only the micromap + its backing/index buffers.
        self.meshes.insert(
            asset_id,
            ClasMeshEntry {
                storage_range,
                cluster_count: cluster_count as u32,
                omm: omm_build.map(MicromapBuild::into_mesh_omm),
            },
        );
    }

    /// Phase 2a: build each cluster's CLAS through the NV
    /// **template → instantiate** path instead of the direct
    /// `BUILD_TRIANGLE_CLUSTER` of [`Self::upload_mesh`]. The arguments
    /// and the produced [`Self::cluster_clas_addresses`] entries are
    /// identical, so this is a drop-in A/B alternative (the caller
    /// chooses via the `SOLARI_CLAS_TEMPLATE` env var) used to prove the
    /// driver's template + instantiate ops before they carry subdivided
    /// (tessellated) topology in Phase 2b/2c.
    ///
    /// Two GPU operations with a CPU readback between them:
    ///
    /// 1. `BUILD_TRIANGLE_CLUSTER_TEMPLATE` — the cluster's index
    ///    topology + reference vertices compile into a per-cluster
    ///    *template* (cluster id / geometry index baked to 0). The
    ///    driver-chosen template addresses are read back to the CPU.
    /// 2. `INSTANTIATE_TRIANGLE_CLUSTER` — each template plus the
    ///    cluster's own vertices produce a CLAS, with the global cluster
    ///    id / geometry index restored via the instantiate *offsets*
    ///    (`cluster_id_offset` + `geometry_index_offset`). The resulting
    ///    CLAS is byte-for-byte equivalent to the direct build.
    ///
    /// For the 1:1 case here the template's reference vertices and the
    /// instantiate's vertices are the same pool slots; in Phase 2b/2c the
    /// template will hold a *subdivided* topology and the instantiate
    /// vertices will be displaced micro-vertices.
    ///
    /// # Panics
    ///
    /// Same setup-time failure modes as [`Self::upload_mesh`], plus
    /// template-address readback mapping failure.
    pub fn upload_mesh_via_template(
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
                    omm: None,
                },
            );
            return;
        }
        let cluster_fns = fns.cluster.as_ref().expect(
            "clas_arena.upload_mesh_via_template: cluster-AS extension function table missing",
        );

        const INDEX_TYPE_32BIT: u32 = 4;
        const OPAQUE_GEOMETRY_FLAG: u8 = 0b100;

        // Shared geometry-shape maxima for the `TriangleClusterInputNV`
        // (same for the template build, the instantiate, and both size
        // queries — see `upload_mesh`).
        let (mut max_tris, mut max_verts) = (0u32, 0u32);
        let (mut total_tris, mut total_verts) = (0u32, 0u32);
        let mut max_global_cluster_id = 0u32;
        for (local_id, cluster) in clusters.iter().enumerate() {
            let global_id = cluster_base.0.wrapping_add(local_id as u32);
            max_tris = max_tris.max(cluster.triangle_count);
            max_verts = max_verts.max(cluster.vertex_count);
            total_tris += cluster.triangle_count;
            total_verts += cluster.vertex_count;
            max_global_cluster_id = max_global_cluster_id.max(global_id);
        }

        // The op input (`pTriangleClusters`) is identical for the
        // template build and the instantiate; the only thing that varies
        // is the op type. Built once and pointed at by both size queries
        // and both command infos.
        let mut triangle_input = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
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

        // The shared per-cluster count buffer (same value for both ops).
        let count_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clas_arena.template.count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&count_buf, 0, &(cluster_count as u32).to_le_bytes());
        let count_addr = allocator.wgpu_buffer_device_address(&count_buf);

        // ----------------------------------------------------------------
        // Step 1 — build the per-cluster templates.
        // ----------------------------------------------------------------
        let mut tmpl_descriptors: Vec<
            vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV,
        > = Vec::with_capacity(cluster_count);
        for cluster in clusters.iter() {
            let Cluster {
                vertex_offset,
                vertex_count,
                index_offset,
                triangle_count,
                ..
            } = *cluster;
            tmpl_descriptors.push(
                vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV {
                    // Template ids are 0; the instantiate restores the
                    // global cluster id / geometry index via its offsets.
                    cluster_id: 0,
                    cluster_flags: vk::ClusterAccelerationStructureClusterFlagsNV::default(),
                    triangle_cluster_info_packed: vk::Packed9_9_6_4_4::new(
                        triangle_count,
                        vertex_count,
                        0,
                        INDEX_TYPE_32BIT,
                        0,
                    ),
                    base_geometry_index_and_geometry_flags:
                        vk::ClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV {
                            geometry_index_and_geometry_flags: vk::Packed24_5_3::new(
                                0,
                                OPAQUE_GEOMETRY_FLAG,
                            ),
                        },
                    index_buffer_stride: 4,
                    vertex_buffer_stride: 12,
                    geometry_index_and_flags_buffer_stride: 0,
                    opacity_micromap_index_buffer_stride: 0,
                    index_buffer: index_buffer_addr + u64::from(index_base + index_offset) * 4,
                    vertex_buffer: vertex_buffer_addr + u64::from(vertex_base + vertex_offset) * 12,
                    geometry_index_and_flags_buffer: 0,
                    opacity_micromap_array: 0,
                    opacity_micromap_index_buffer: 0,
                    // 0 = no instantiation bounding-box clamp.
                    instantiation_bounding_box_limit: 0,
                },
            );
        }

        let tmpl_stride =
            size_of::<vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV>() as u64;
        let tmpl_bytes_len = (tmpl_descriptors.len() as u64) * tmpl_stride;
        // SAFETY: the template info is repr(C) POD; a flat byte view is
        // well-defined for upload.
        let tmpl_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                tmpl_descriptors.as_ptr().cast::<u8>(),
                tmpl_bytes_len as usize,
            )
        };
        let tmpl_src_infos = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clas_arena.template.src_infos"),
            size: tmpl_bytes_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&tmpl_src_infos, 0, tmpl_bytes);
        let tmpl_src_infos_addr = allocator.wgpu_buffer_device_address(&tmpl_src_infos);

        // Template size query.
        let tmpl_size_input = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(cluster_count as u32)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
            )
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER_TEMPLATE)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(op_input);
        let mut tmpl_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: input fully populated; function table loaded above.
        unsafe {
            cluster_fns
                .get_cluster_acceleration_structure_build_sizes(&tmpl_size_input, &mut tmpl_sizes);
        }

        // Template storage is transient (consumed by the instantiate this
        // call records); a plain GpuOnly buffer, not the sparse arena.
        let tmpl_storage = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            wgpu::BufferUsages::STORAGE,
            tmpl_sizes.acceleration_structure_size.max(1),
            MemoryLocation::GpuOnly,
            "clas_arena.template.storage",
        );
        let tmpl_storage_addr = allocator.wgpu_buffer_device_address(&tmpl_storage);

        let tmpl_scratch = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            wgpu::BufferUsages::STORAGE,
            tmpl_sizes.build_scratch_size.max(1) + CLAS_SCRATCH_ALIGN - 1,
            MemoryLocation::GpuOnly,
            "clas_arena.template.scratch",
        );
        let tmpl_scratch_addr = align_up(
            allocator.wgpu_buffer_device_address(&tmpl_scratch),
            CLAS_SCRATCH_ALIGN,
        );

        let addr_array_size = (cluster_count as u64) * 8;
        let tmpl_addresses = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::TRANSFER_SRC,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            addr_array_size,
            MemoryLocation::GpuOnly,
            "clas_arena.template.addresses",
        );
        let tmpl_addresses_addr = allocator.wgpu_buffer_device_address(&tmpl_addresses);

        let mut tmpl_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("clas_arena.template.build"),
        });
        let tmpl_cmd = vk::ClusterAccelerationStructureCommandsInfoNV {
            s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
            p_next: core::ptr::null_mut(),
            input: tmpl_size_input,
            dst_implicit_data: tmpl_storage_addr,
            scratch_data: tmpl_scratch_addr,
            dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
                device_address: tmpl_addresses_addr,
                stride: 8,
                size: addr_array_size,
            },
            dst_sizes_array: vk::StridedDeviceAddressRegionKHR::default(),
            src_infos_array: vk::StridedDeviceAddressRegionKHR {
                device_address: tmpl_src_infos_addr,
                stride: tmpl_stride,
                size: tmpl_bytes_len,
            },
            src_infos_count: count_addr,
            address_resolution_flags:
                vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
            _marker: core::marker::PhantomData,
        };
        // SAFETY: function table loaded; encoder open + Vulkan-backed;
        // descriptor strides/addresses reference the buffers above. No
        // wgpu commands touch this encoder afterward.
        unsafe {
            crate::gpu::extension::cmd_build_cluster_acceleration_structures_indirect(
                &mut tmpl_encoder,
                fns,
                &tmpl_cmd,
            );
            crate::gpu::extension::cmd_global_as_barrier(&mut tmpl_encoder, &render_device, false);
        }
        let tmpl_idx = render_queue.submit([tmpl_encoder.finish()]);
        {
            let _span = tracing::info_span!("clas.template_build_poll_wait").entered();
            let _ = render_device.wgpu_device().poll(wgpu::PollType::Wait {
                submission_index: Some(tmpl_idx),
                timeout: None,
            });
        }

        // Read the driver-chosen template addresses back to the CPU so we
        // can fill the per-cluster instantiate descriptors below.
        let readback = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clas_arena.template.addr_readback"),
            size: addr_array_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut rb_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("clas_arena.template.addr_copy"),
        });
        rb_encoder.copy_buffer_to_buffer(&tmpl_addresses, 0, &readback, 0, addr_array_size);
        let rb_idx = render_queue.submit([rb_encoder.finish()]);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| {
                result.expect("clas_arena.upload_mesh_via_template: template address readback map failed");
            });
        {
            let _span = tracing::info_span!("clas.template_addr_readback_wait").entered();
            let _ = render_device.wgpu_device().poll(wgpu::PollType::Wait {
                submission_index: Some(rb_idx),
                timeout: None,
            });
        }
        let template_addrs: Vec<u64> = {
            let mapped = readback.slice(..).get_mapped_range();
            mapped
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        readback.unmap();

        // ----------------------------------------------------------------
        // Step 2 — instantiate each template into a CLAS.
        // ----------------------------------------------------------------
        let mut inst_descriptors: Vec<
            vk::ClusterAccelerationStructureInstantiateClusterInfoNV,
        > = Vec::with_capacity(cluster_count);
        for (local_id, cluster) in clusters.iter().enumerate() {
            let global_id = cluster_base.0.wrapping_add(local_id as u32);
            inst_descriptors.push(vk::ClusterAccelerationStructureInstantiateClusterInfoNV {
                // Restore the global cluster id / geometry index that the
                // direct build bakes directly (template ids were 0).
                cluster_id_offset: global_id,
                geometry_index_offset_and_reserved: vk::Packed24_8::new(global_id, 0),
                cluster_template_address: template_addrs[local_id],
                vertex_buffer: vk::StridedDeviceAddressNV {
                    start_address: vertex_buffer_addr
                        + u64::from(vertex_base + cluster.vertex_offset) * 12,
                    stride_in_bytes: 12,
                },
            });
        }

        let inst_stride =
            size_of::<vk::ClusterAccelerationStructureInstantiateClusterInfoNV>() as u64;
        let inst_bytes_len = (inst_descriptors.len() as u64) * inst_stride;
        // SAFETY: instantiate info is repr(C) POD; flat byte view is valid.
        let inst_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                inst_descriptors.as_ptr().cast::<u8>(),
                inst_bytes_len as usize,
            )
        };
        let inst_src_infos = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clas_arena.instantiate.src_infos"),
            size: inst_bytes_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&inst_src_infos, 0, inst_bytes);
        let inst_src_infos_addr = allocator.wgpu_buffer_device_address(&inst_src_infos);

        // Instantiate size query (same `pTriangleClusters` input, new op).
        let inst_size_input = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(cluster_count as u32)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
            )
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::INSTANTIATE_TRIANGLE_CLUSTER)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(op_input);
        let mut inst_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: input fully populated; function table loaded above.
        unsafe {
            cluster_fns
                .get_cluster_acceleration_structure_build_sizes(&inst_size_input, &mut inst_sizes);
        }

        // Final CLAS lands in the persistent sparse arena (like the direct
        // build) so the downstream selector → BLAS path is unchanged.
        let storage_range = self
            .allocator
            .allocate_range_aligned(inst_sizes.acceleration_structure_size, CLAS_STORAGE_ALIGN)
            .expect("clas_arena: storage virtual-address space exhausted");
        self.storage.commit(storage_range.clone());

        let addr_byte_start = (cluster_base.0 as u64) * 8;
        let addr_byte_end = addr_byte_start + (cluster_count as u64) * 8;
        self.cluster_clas_addresses
            .commit(addr_byte_start..addr_byte_end);

        let inst_scratch = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            wgpu::BufferUsages::STORAGE,
            inst_sizes.build_scratch_size.max(1) + CLAS_SCRATCH_ALIGN - 1,
            MemoryLocation::GpuOnly,
            "clas_arena.instantiate.scratch",
        );
        let inst_scratch_addr = align_up(
            allocator.wgpu_buffer_device_address(&inst_scratch),
            CLAS_SCRATCH_ALIGN,
        );

        let inst_dst_addresses = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::TRANSFER_SRC,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            addr_array_size,
            MemoryLocation::GpuOnly,
            "clas_arena.instantiate.dst_addresses",
        );
        let inst_dst_addresses_addr = allocator.wgpu_buffer_device_address(&inst_dst_addresses);

        let mut inst_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("clas_arena.instantiate.build"),
        });
        let inst_dst_implicit = self.storage.address + storage_range.start;
        let inst_cmd = vk::ClusterAccelerationStructureCommandsInfoNV {
            s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
            p_next: core::ptr::null_mut(),
            input: inst_size_input,
            dst_implicit_data: inst_dst_implicit,
            scratch_data: inst_scratch_addr,
            dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
                device_address: inst_dst_addresses_addr,
                stride: 8,
                size: addr_array_size,
            },
            dst_sizes_array: vk::StridedDeviceAddressRegionKHR::default(),
            src_infos_array: vk::StridedDeviceAddressRegionKHR {
                device_address: inst_src_infos_addr,
                stride: inst_stride,
                size: inst_bytes_len,
            },
            src_infos_count: count_addr,
            address_resolution_flags:
                vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
            _marker: core::marker::PhantomData,
        };
        // SAFETY: same contract as the template build above.
        unsafe {
            crate::gpu::extension::cmd_build_cluster_acceleration_structures_indirect(
                &mut inst_encoder,
                fns,
                &inst_cmd,
            );
            crate::gpu::extension::cmd_global_as_barrier(&mut inst_encoder, &render_device, false);
        }
        let inst_idx = render_queue.submit([inst_encoder.finish()]);
        {
            let _span = tracing::info_span!("clas.instantiate_build_poll_wait").entered();
            let _ = render_device.wgpu_device().poll(wgpu::PollType::Wait {
                submission_index: Some(inst_idx),
                timeout: None,
            });
        }

        let mut copy_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("clas_arena.instantiate.copy_addresses"),
        });
        copy_encoder.copy_buffer_to_buffer(
            &inst_dst_addresses,
            0,
            &self.cluster_clas_addresses.wgpu_buffer,
            addr_byte_start,
            addr_array_size,
        );
        render_queue.submit([copy_encoder.finish()]);

        self.meshes.insert(
            asset_id,
            ClasMeshEntry {
                storage_range,
                cluster_count: cluster_count as u32,
                // OMM not wired on the template A/B path (MVP); the default
                // BUILD_TRIANGLE_CLUSTER path carries opacity micro-maps.
                omm: None,
            },
        );
    }
}

/// Round `addr` up to the next multiple of `align` (a power of two).
fn align_up(addr: u64, align: u64) -> u64 {
    let misalign = addr & (align - 1);
    if misalign == 0 {
        addr
    } else {
        addr + (align - misalign)
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

    // A/B switch for Phase 2a: `SOLARI_CLAS_TEMPLATE` routes every CLAS
    // build through the NV template→instantiate path instead of the
    // direct `BUILD_TRIANGLE_CLUSTER`. Both produce identical
    // `cluster_clas_addresses`, so the render output should be
    // pixel-identical — the toggle exists to prove the template ops on
    // the driver before Phase 2b/2c put subdivided topology in them.
    let via_template = std::env::var_os("SOLARI_CLAS_TEMPLATE").is_some();

    let pending = std::mem::take(&mut cluster_meshes.pending_clas_uploads);
    for entry in pending {
        if via_template {
            clas_arena.upload_mesh_via_template(
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
        } else {
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
                entry.omm.as_ref(),
            );
        }
    }
}
