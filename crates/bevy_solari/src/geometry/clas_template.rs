// CLAS template arena — topology-only per-cluster acceleration-structure
// templates for ANIMATED meshes. A template encodes a cluster's connectivity
// (triangle/vertex counts, index buffer, geometry id) plus an
// `instantiationBoundingBoxLimit` deform envelope, but NOT final vertex
// positions. Each frame, `geometry::…instantiate` feeds the just-deformed
// vertex positions into these templates (op `INSTANTIATE_TRIANGLE_CLUSTER`) to
// produce a fresh per-instance CLAS, which a per-instance BLAS then references.
//
// Built once at upload, mirroring `clas_arena::upload_mesh` almost exactly: the
// only differences are the op type (`BUILD_TRIANGLE_CLUSTER_TEMPLATE`), the
// per-cluster `instantiation_bounding_box_limit`, and the destination table
// (`cluster_template_addresses` instead of `cluster_clas_addresses`).
#![allow(unsafe_code, reason = "raw-VK cluster-AS template build")]

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

use super::clas_arena::{CLAS_SCRATCH_ALIGN, CLAS_STORAGE_ALIGN, CLUSTER_CLAS_ADDRESSES_VIRTUAL_BYTES};
use super::{Cluster, ClusterBloatAabb, ClusterIndex, ClusterMesh, ClusterMeshManager};
use crate::gpu::allocator::{Allocator, MemoryLocation, SparseBuffer};
use crate::gpu::extension::ClusterExtensionFns;
use crate::gpu::retire::GpuRetire;

/// Virtual address space reserved for the template-CLAS storage arena —
/// 16 GB, sparse-backed (we pay only for committed pages). Templates are
/// resident for a mesh's lifetime, like the static CLAS arena.
pub const TEMPLATE_POOL_VIRTUAL_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// Per-cluster stride of the tight instantiation-bbox buffer NV reads at
/// `instantiationBoundingBoxLimit`: a contiguous `VkAabbPositionsKHR`
/// (minXYZ, maxXYZ = 6 × f32 = 24 B) padded to 32 B. The crate's
/// [`ClusterBloatAabb`] is `{min:[f32;4], max:[f32;4]}` (padded, non-contiguous),
/// so the template build repacks it into this tight layout.
const BBOX_STRIDE: u64 = 32;

/// Per-mesh template record.
#[derive(Debug)]
pub struct TemplateMeshEntry {
    /// Byte range in [`ClusterTemplateArena::storage`] holding this mesh's
    /// template payloads.
    pub storage_range: Range<u64>,
    pub cluster_count: u32,
}

/// Render-world resource owning the template-CLAS storage pool and the
/// global per-cluster template device-address table. Both are sparse-backed
/// (stable handle + address across growth).
#[derive(Resource)]
pub struct ClusterTemplateArena {
    /// Sparse AS-storage buffer holding every animated cluster's template.
    pub storage: SparseBuffer,
    allocator: RangeAllocator<u64>,
    meshes: HashMap<AssetId<ClusterMesh>, TemplateMeshEntry>,
    /// Global per-cluster template device-address table, indexed by global
    /// cluster id (`cluster_base + local_id`) — the SAME index space as
    /// `ClasArena::cluster_clas_addresses`, so the instantiate pass can read
    /// `cluster_template_addresses[global_cluster_id]`. WGSL view is
    /// `array<vec2<u32>>`.
    pub cluster_template_addresses: SparseBuffer,
}

impl ClusterTemplateArena {
    pub fn new(render_device: &RenderDevice, allocator: &Allocator) -> Self {
        let storage = allocator.create_sparse_buffer(
            render_device,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            wgpu::BufferUsages::COPY_DST,
            TEMPLATE_POOL_VIRTUAL_BYTES,
            "clas_template.storage",
        );
        let cluster_template_addresses = allocator.create_sparse_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            CLUSTER_CLAS_ADDRESSES_VIRTUAL_BYTES,
            "clas_template.cluster_template_addresses",
        );
        Self {
            storage,
            allocator: RangeAllocator::new(0..TEMPLATE_POOL_VIRTUAL_BYTES),
            meshes: HashMap::default(),
            cluster_template_addresses,
        }
    }

    pub fn mesh(&self, asset_id: AssetId<ClusterMesh>) -> Option<&TemplateMeshEntry> {
        self.meshes.get(&asset_id)
    }

    /// Build a topology-only CLAS template for every cluster in `mesh` and
    /// record the per-cluster template device addresses. `clusters` are the
    /// CPU-side (mesh-local) cluster records; `bloat_aabbs` is parallel to
    /// them. Mirrors [`super::clas_arena::ClasArena::upload_mesh`].
    ///
    /// # Panics
    /// Panics on the same setup-time failure modes as the static arena.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_mesh_templates(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        allocator: &Allocator,
        fns: &ClusterExtensionFns,
        retire: &mut GpuRetire,
        asset_id: AssetId<ClusterMesh>,
        clusters: &[Cluster],
        bloat_aabbs: &[ClusterBloatAabb],
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
                TemplateMeshEntry { storage_range: 0..0, cluster_count: 0 },
            );
            return;
        }
        debug_assert_eq!(
            bloat_aabbs.len(),
            cluster_count,
            "clas_template: bloat AABB count must match cluster count"
        );
        let cluster_fns = fns
            .cluster
            .as_ref()
            .expect("clas_template.upload_mesh_templates: cluster-AS extension fn table missing");

        // NV index type 4 = 32-bit indices. OPAQUE = 0b100 in the 3-bit
        // geometry-flags subfield (see memory `aurora_cluster_as_opaque.md`).
        const INDEX_TYPE_32BIT: u32 = 4;
        const OPAQUE_GEOMETRY_FLAG: u8 = 0b100;

        // 0. Tight instantiation-bbox buffer: repack the padded
        //    `ClusterBloatAabb` into contiguous minXYZ,maxXYZ (24 B) at a 32 B
        //    stride. One entry per cluster, addressed per-cluster below.
        let bbox_floats_per = (BBOX_STRIDE / 4) as usize; // 8 (6 used + 2 pad)
        let mut bbox_data = vec![0f32; cluster_count * bbox_floats_per];
        for (i, aabb) in bloat_aabbs.iter().enumerate() {
            let o = i * bbox_floats_per;
            bbox_data[o] = aabb.min[0];
            bbox_data[o + 1] = aabb.min[1];
            bbox_data[o + 2] = aabb.min[2];
            bbox_data[o + 3] = aabb.max[0];
            bbox_data[o + 4] = aabb.max[1];
            bbox_data[o + 5] = aabb.max[2];
        }
        let bbox_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clas_template.bbox"),
            size: (cluster_count as u64) * BBOX_STRIDE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&bbox_buf, 0, bytemuck::cast_slice(&bbox_data));
        let bbox_addr = allocator.wgpu_buffer_device_address(&bbox_buf).get();

        // 1. Per-cluster template build descriptors.
        let (mut max_tris, mut max_verts) = (0u32, 0u32);
        let (mut total_tris, mut total_verts) = (0u32, 0u32);
        let mut max_global_cluster_id = 0u32;
        let mut descriptors: Vec<
            vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV,
        > = Vec::with_capacity(cluster_count);
        for (local_id, cluster) in clusters.iter().enumerate() {
            let Cluster {
                vertex_offset,
                vertex_count,
                index_offset,
                triangle_count,
                ..
            } = *cluster;
            let global_id = cluster_base.0.wrapping_add(local_id as u32);

            descriptors.push(
                vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV {
                    // Global cluster id so the RT-pipeline hit shader's
                    // `@builtin(cluster_id)` (ClusterIDNV) indexes `clusters[]`
                    // directly. The ray-query path uses the baked
                    // `base_geometry_index` instead, so this is free for it.
                    cluster_id: global_id,
                    cluster_flags: vk::ClusterAccelerationStructureClusterFlagsNV::default(),
                    // 9_9_6_4_4: tris(9), verts(9), truncate(6), index_type(4), omm(4).
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
                                global_id,
                                OPAQUE_GEOMETRY_FLAG,
                            ),
                        },
                    index_buffer_stride: 4,
                    vertex_buffer_stride: 12,
                    geometry_index_and_flags_buffer_stride: 0,
                    opacity_micromap_index_buffer_stride: 0,
                    // Mesh-local offsets → absolute device addresses (same
                    // rebase the static build does — see clas_arena.rs).
                    index_buffer: index_buffer_addr
                        + u64::from(index_base + index_offset) * 4,
                    vertex_buffer: vertex_buffer_addr
                        + u64::from(vertex_base + vertex_offset) * 12,
                    geometry_index_and_flags_buffer: 0,
                    opacity_micromap_array: 0,
                    opacity_micromap_index_buffer: 0,
                    instantiation_bounding_box_limit: bbox_addr
                        + (local_id as u64) * BBOX_STRIDE,
                },
            );

            max_tris = max_tris.max(triangle_count);
            max_verts = max_verts.max(vertex_count);
            total_tris += triangle_count;
            total_verts += vertex_count;
            max_global_cluster_id = max_global_cluster_id.max(global_id);
        }

        // 2. Upload src_infos.
        let desc_stride = size_of::<
            vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV,
        >() as u64;
        let desc_bytes_len = (descriptors.len() as u64) * desc_stride;
        // SAFETY: the template info struct is repr(C) POD; a flat byte view is
        // well-defined for upload.
        let desc_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                descriptors.as_ptr().cast::<u8>(),
                desc_bytes_len as usize,
            )
        };
        let src_infos_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clas_template.src_infos"),
            size: desc_bytes_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&src_infos_buf, 0, desc_bytes);
        let src_infos_addr = allocator.wgpu_buffer_device_address(&src_infos_buf).get();

        let count_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clas_template.src_infos_count"),
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
        let size_input = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(cluster_count as u32)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER_TEMPLATE)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(op_input);
        let mut sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: size_input fully populated; cluster-AS fn table loaded.
        unsafe {
            cluster_fns
                .get_cluster_acceleration_structure_build_sizes(&size_input, &mut sizes_info);
        }

        // 4. Allocate template storage + scratch + dst_addresses.
        let storage_range = self
            .allocator
            .allocate_range_aligned(sizes_info.acceleration_structure_size, CLAS_STORAGE_ALIGN)
            .expect("clas_template: storage virtual-address space exhausted");
        self.storage.commit(storage_range.clone());

        let addr_byte_start = (cluster_base.0 as u64) * 8;
        let addr_byte_end = addr_byte_start + (cluster_count as u64) * 8;
        self.cluster_template_addresses
            .commit(addr_byte_start..addr_byte_end);

        let scratch_size = sizes_info.build_scratch_size.max(1);
        let scratch_buf = allocator.create_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            wgpu::BufferUsages::STORAGE,
            scratch_size + CLAS_SCRATCH_ALIGN - 1,
            MemoryLocation::GpuOnly,
            "clas_template.scratch",
        );
        let scratch_addr_base = allocator.wgpu_buffer_device_address(&scratch_buf).get();
        let scratch_misalign = scratch_addr_base & (CLAS_SCRATCH_ALIGN - 1);
        let scratch_offset = if scratch_misalign == 0 {
            0
        } else {
            CLAS_SCRATCH_ALIGN - scratch_misalign
        };
        let scratch_addr = scratch_addr_base + scratch_offset;

        let dst_addresses_size = (cluster_count as u64) * 8;
        let dst_addresses_buf = allocator.create_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::TRANSFER_SRC,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            dst_addresses_size,
            MemoryLocation::GpuOnly,
            "clas_template.dst_addresses",
        );
        let dst_addresses_addr = allocator.wgpu_buffer_device_address(&dst_addresses_buf).get();

        // Declared access (SOLARI_VALIDATE): the implicit-dst window of the
        // template pool this mesh's templates land in.
        crate::gpu::extension::validate_raw_access(&crate::gpu::extension::RawAccess {
            op: "clas_template.upload",
            reads: &[],
            writes: &[(&self.storage, storage_range.clone())],
        });

        // 5. Encode + submit the template build.
        let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("clas_template.upload"),
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
            src_infos_count: allocator.wgpu_buffer_device_address(&count_buf).get(),
            address_resolution_flags:
                vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
            _marker: core::marker::PhantomData,
        };

        // SAFETY: cluster-AS fn table loaded; encoder open + Vulkan-backed;
        // descriptor strides + addresses point at the buffers just created. No
        // wgpu commands touch this encoder afterward.
        unsafe {
            crate::gpu::extension::cmd_build_cluster_acceleration_structures_indirect(
                &mut encoder,
                fns,
                &cmd_info,
            );
            crate::gpu::extension::cmd_global_as_barrier(&mut encoder, render_device, false);
        }
        // No CPU wait: trailing global AS barrier + submission order make the
        // template addresses visible to the copy.
        render_queue.submit([encoder.finish()]);

        // 6. Copy dst_addresses → global template-address table.
        let mut copy_encoder =
            render_device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("clas_template.copy_addresses"),
            });
        copy_encoder.copy_buffer_to_buffer(
            &dst_addresses_buf,
            0,
            &self.cluster_template_addresses.wgpu_buffer,
            addr_byte_start,
            dst_addresses_size,
        );
        render_queue.submit([copy_encoder.finish()]);

        self.meshes.insert(
            asset_id,
            TemplateMeshEntry {
                storage_range,
                cluster_count: cluster_count as u32,
            },
        );
        // The raw build references these by device address — reaper-owned.
        retire.retire(
            render_queue,
            "clas_template.upload_transients",
            (bbox_buf, src_infos_buf, count_buf, scratch_buf, dst_addresses_buf),
        );
    }
}

/// `RenderStartup`: insert [`ClusterTemplateArena`] iff the raw-VK
/// [`Allocator`] is present (cluster-AS extension enabled).
pub fn init_clas_template_arena(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
) {
    let Some(allocator) = allocator else {
        return;
    };
    commands.insert_resource(ClusterTemplateArena::new(&render_device, &allocator));
}

/// `Render::PrepareAssets` (after `perform_pending_cluster_mesh_writes`, before
/// `upload_pending_clas` consumes the queue): build a topology template for each
/// pending **animated** mesh (non-empty bloat AABBs). `upload_mesh_templates`
/// dedupes by asset id, so re-reading the queue each frame is harmless.
pub fn upload_pending_templates(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    allocator: Option<Res<Allocator>>,
    fns: Option<Res<ClusterExtensionFns>>,
    cluster_meshes: Res<ClusterMeshManager>,
    arena: Option<ResMut<ClusterTemplateArena>>,
    mut retire: ResMut<GpuRetire>,
) {
    let (Some(allocator), Some(fns), Some(mut arena)) = (allocator, fns, arena) else {
        return;
    };
    if fns.cluster.is_none() || cluster_meshes.pending_clas_uploads.is_empty() {
        return;
    }

    let vertex_addr =
        allocator.wgpu_buffer_device_address(cluster_meshes.vertex_positions.buffer()).get();
    let index_addr = allocator.wgpu_buffer_device_address(cluster_meshes.indices.buffer()).get();

    for entry in &cluster_meshes.pending_clas_uploads {
        if entry.bloat_aabbs.is_empty() {
            continue;
        }
        arena.upload_mesh_templates(
            &render_device,
            &render_queue,
            &allocator,
            &fns,
            &mut retire,
            entry.asset_id,
            &entry.clusters,
            &entry.bloat_aabbs,
            entry.cluster_base,
            vertex_addr,
            index_addr,
            entry.vertex_base,
            entry.index_base,
        );
    }
}
