// Builds the per-level CLAS *templates* the tessellation path instantiates.
// Raw Vulkan for the cluster-AS template build, like `clas_arena`.
#![allow(unsafe_code)]

//! Per-subdivision-level CLAS templates (Phase 2b, GPU half).
//!
//! A *template* is a CLAS topology compiled once and re-instantiated each
//! frame with fresh vertex positions. Here we build one template per uniform
//! subdivision level (1..=[`MAX_TESS_LEVEL`]) from the clean-room
//! [`SubdividedTriangle`] topology: the template's reference vertices are the
//! micro-vertices' barycentric coordinates, its indices the micro-triangle
//! connectivity. Instantiating level `L`'s template with `L`'s
//! `(L+1)(L+2)/2` displaced micro-vertex positions yields a CLAS holding `L²`
//! displaced micro-triangles (Phase 2c).
//!
//! This is the clean-room counterpart of the reference's
//! `TessellationTable::initTemplates`, restricted to uniform per-level
//! subdivision (the reference keys templates by three per-edge factors for
//! crack-free adaptive seams — a later extension over the same lattice).
//!
//! The build reuses the exact `BUILD_TRIANGLE_CLUSTER_TEMPLATE` op proven by
//! [`ClasArena::upload_mesh_via_template`](super::clas_arena): all levels in
//! one indirect build, `IMPLICIT_DESTINATIONS` + a CPU readback of the
//! driver-chosen template addresses.

use ash::vk::{self, TaggedStructure};
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res},
};
use bevy_render::renderer::{RenderDevice, RenderQueue};
use wgpu::CommandEncoderDescriptor;

use crate::gpu::allocator::{Allocator, MemoryLocation};
use crate::gpu::extension::ClusterExtensionFns;
use super::tessellation::SubdividedTriangle;

/// Highest uniform subdivision level a template is built for. Level `L`
/// produces `L²` micro-triangles and `(L+1)(L+2)/2` micro-vertices per base
/// triangle. The showcase instantiates ONE base triangle per CLAS, so the bound
/// is the NV per-cluster cap (256 triangles): `L = 16` → 256 micro-triangles / 153
/// micro-vertices, right at the cap. Finer than this per base triangle needs
/// RECURSIVE SPLIT (split the base triangle into a grid of smaller triangles, each
/// its own CLAS) — the "split" half of adaptive classify/split. (A future packing
/// path that puts many base triangles in one CLAS would lower this bound.)
pub const MAX_TESS_LEVEL: u32 = 16;

/// Base `cluster_id` (ClusterIDNV) for tessellated CLASes — each bakes
/// `TESS_CLUSTER_ID_BASE + global_tess_clas_index`, comfortably above any real
/// cluster pool so the closest-hit detects the tess hit (`cluster_id >=
/// arrayLength(&clusters)`) and recovers the tess index for the smooth-normal
/// metadata lookup. Must match `scene_bindings::TESS_CLUSTER_ID_BASE`.
pub const TESS_CLUSTER_ID_BASE: u32 = 0xF000_0000;

/// Render-world resource holding the built per-level templates.
///
/// Indexed by `level - 1` (so `templates[0]` is level 1, the trivial
/// pass-through subdivision). Present only when the cluster-AS extension is
/// available (gated on [`Allocator`], like [`ClasArena`](super::clas_arena)).
#[derive(Resource)]
pub struct TessellationTemplates {
    /// Per-level template device address. Fed to
    /// `ClusterAccelerationStructureInstantiateClusterInfoNV::cluster_template_address`.
    pub template_addresses: Vec<vk::DeviceAddress>,
    /// Per-level micro-triangle count (`level²`).
    pub triangle_counts: Vec<u32>,
    /// Per-level micro-vertex count (`(level+1)(level+2)/2`).
    pub vertex_counts: Vec<u32>,
    /// Backing AS storage for every template. Kept alive for the resource's
    /// lifetime: the template device addresses point into it.
    _template_storage: wgpu::Buffer,
}

/// Output of [`TessellationTemplates::record_instantiate_displaced_batch`]: the
/// GPU buffer the build wrote the per-CLAS device addresses into (its device
/// address is the BLAS `cluster_references` — consumed GPU-side, never read back),
/// the CLAS AS storage (kept alive for the trace), and the build-only transients
/// the caller holds until the submit completes.
pub struct InstantiatedClasBatch {
    pub clas_addresses: wgpu::Buffer,
    pub storage: wgpu::Buffer,
    /// Mixed allocator `wgpu::Buffer` (scratch) + bevy `Buffer` (src-infos / count),
    /// so boxed — held by the caller until the submit completes.
    pub transient: Vec<Box<dyn core::any::Any + Send + Sync>>,
}

/// Output of [`record_build_per_instance_blas`]: the BLAS AS storage (kept alive
/// for the trace) and the build-only transients. The BLAS device address is
/// written GPU-side to the caller's `blas_dst_addr` (no readback).
pub struct InstanceBlas {
    pub storage: wgpu::Buffer,
    /// Mixed buffer types, boxed — held by the caller until the submit completes.
    pub transient: Vec<Box<dyn core::any::Any + Send + Sync>>,
}

impl TessellationTemplates {
    /// Template device address for `level` (1..=[`MAX_TESS_LEVEL`]).
    #[inline]
    pub fn address(&self, level: u32) -> vk::DeviceAddress {
        self.template_addresses[(level - 1) as usize]
    }

    /// Phase 2c brick B2 — instantiate `level`'s template with
    /// `displaced_positions_addr` (device address of `vertex_counts[level-1]`
    /// tightly-packed `vec3` positions, e.g. the displacement compute's
    /// `out_positions`) into a single tessellated CLAS. Returns its device
    /// address plus the backing storage buffer (kept alive by the caller — the
    /// CLAS address points into it).
    ///
    /// `cluster_id` is restored into the CLAS via the instantiate offsets
    /// (`cluster_id_offset` + `geometry_index_offset`), exactly like the
    /// RTX-validated 1:1 instantiate in
    /// [`ClasArena::upload_mesh_via_template`](super::clas_arena). Self-contained
    /// (own scratch + storage + a CPU readback of the CLAS address); production
    /// will batch many per per-instance BLAS.
    ///
    /// # Panics
    ///
    /// Panics if the cluster-AS function table is missing or the build /
    /// readback fails.
    pub fn instantiate_displaced(
        &self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        allocator: &Allocator,
        fns: &ClusterExtensionFns,
        level: u32,
        displaced_positions_addr: vk::DeviceAddress,
        cluster_id: u32,
    ) -> (vk::DeviceAddress, wgpu::Buffer) {
        let cluster_fns = fns
            .cluster
            .as_ref()
            .expect("tess_template.instantiate_displaced: cluster-AS extension missing");
        let lvl0 = (level - 1) as usize;
        let tri_count = self.triangle_counts[lvl0];
        let vert_count = self.vertex_counts[lvl0];

        let desc = vk::ClusterAccelerationStructureInstantiateClusterInfoNV {
            // `cluster_id` becomes the CLAS's ClusterIDNV — what the closest-hit
            // reads via `@builtin(cluster_id)`. The tess showcase passes a SENTINEL
            // above the real cluster pool so the chit detects the hit (no
            // `clusters[]` entry) and shades via the facet normal. The geometry
            // index stays 0 (single geometry) so the build doesn't size itself for
            // a sentinel-sized geometry-index space.
            cluster_id_offset: cluster_id,
            geometry_index_offset_and_reserved: vk::Packed24_8::new(0, 0),
            cluster_template_address: self.template_addresses[lvl0],
            vertex_buffer: vk::StridedDeviceAddressNV {
                start_address: displaced_positions_addr,
                stride_in_bytes: 12,
            },
        };
        let stride =
            size_of::<vk::ClusterAccelerationStructureInstantiateClusterInfoNV>() as u64;
        // SAFETY: instantiate info is repr(C) POD; flat byte view is valid.
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(core::ptr::from_ref(&desc).cast::<u8>(), stride as usize)
        };
        let src_infos = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tess_template.instantiate.src_infos"),
            size: stride,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&src_infos, 0, bytes);
        let src_infos_addr = allocator.wgpu_buffer_device_address(&src_infos);

        let count_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tess_template.instantiate.count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&count_buf, 0, &1u32.to_le_bytes());
        let count_addr = allocator.wgpu_buffer_device_address(&count_buf);

        let mut triangle_input = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            // Single geometry, index 0 (the ClusterIDNV sentinel rides in
            // `cluster_id_offset`, not the geometry index).
            .max_geometry_index_value(0)
            .max_cluster_unique_geometry_count(1)
            .max_cluster_triangle_count(tri_count)
            .max_cluster_vertex_count(vert_count)
            .max_total_triangle_count(tri_count)
            .max_total_vertex_count(vert_count)
            .min_position_truncate_bit_count(0);
        let op_input = vk::ClusterAccelerationStructureOpInputNV {
            p_triangle_clusters: &mut triangle_input as *mut _,
        };
        let size_input = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(1)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
            )
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::INSTANTIATE_TRIANGLE_CLUSTER)
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
            "tess_template.instantiate.storage",
        );
        let storage_addr = allocator.wgpu_buffer_device_address(&storage);

        let scratch = allocator.create_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            wgpu::BufferUsages::STORAGE,
            sizes.build_scratch_size.max(1) + 255,
            MemoryLocation::GpuOnly,
            "tess_template.instantiate.scratch",
        );
        let scratch_addr = {
            let base = allocator.wgpu_buffer_device_address(&scratch);
            let m = base & 255;
            if m == 0 { base } else { base + (256 - m) }
        };

        let dst_addresses = allocator.create_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::TRANSFER_SRC,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            8,
            MemoryLocation::GpuOnly,
            "tess_template.instantiate.dst_addresses",
        );
        let dst_addresses_addr = allocator.wgpu_buffer_device_address(&dst_addresses);

        let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("tess_template.instantiate"),
        });
        let cmd = vk::ClusterAccelerationStructureCommandsInfoNV {
            s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
            p_next: core::ptr::null_mut(),
            input: size_input,
            dst_implicit_data: storage_addr,
            scratch_data: scratch_addr,
            dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
                device_address: dst_addresses_addr,
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
        // SAFETY: function table loaded; encoder open + Vulkan-backed; the
        // descriptor strides/addresses reference the buffers above.
        unsafe {
            crate::gpu::extension::cmd_build_cluster_acceleration_structures_indirect(
                &mut encoder,
                fns,
                &cmd,
            );
            crate::gpu::extension::cmd_global_as_barrier(&mut encoder, render_device, false);
        }
        let idx = render_queue.submit([encoder.finish()]);
        {
            let _span = tracing::info_span!("tess_template.instantiate_poll_wait").entered();
            let _ = render_device.wgpu_device().poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            });
        }

        let readback = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tess_template.instantiate.addr_readback"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut rb = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("tess_template.instantiate.addr_copy"),
        });
        rb.copy_buffer_to_buffer(&dst_addresses, 0, &readback, 0, 8);
        let rb_idx = render_queue.submit([rb.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |r| {
            r.expect("tess_template.instantiate_displaced: CLAS address readback map failed");
        });
        let _ = render_device.wgpu_device().poll(wgpu::PollType::Wait {
            submission_index: Some(rb_idx),
            timeout: None,
        });
        let clas_addr = {
            let mapped = readback.slice(..).get_mapped_range();
            u64::from_le_bytes(mapped[..8].try_into().unwrap())
        };
        readback.unmap();

        (clas_addr, storage)
    }

    /// Batched instantiate of `base_triangle_count` CLASes — one per contiguous
    /// displaced micro-vertex region in `displaced_positions_addr` (region `i` at
    /// `+ i * vertex_count * 12`), all at `level`, in ONE indirect build RECORDED
    /// into `encoder` (no submit). The per-CLAS device addresses are written
    /// GPU-side into the returned [`InstantiatedClasBatch::clas_addresses`] buffer
    /// (stride 8) — feed its device address straight into
    /// [`record_build_per_instance_blas`] as `cluster_references`; they're consumed
    /// GPU-side, never read back. The caller submits the encoder (after inserting an
    /// AS barrier before the BLAS build) and keeps the returned buffers alive until
    /// it completes.
    ///
    /// # Panics
    ///
    /// Panics if the cluster-AS function table is missing or the size query fails.
    pub fn record_instantiate_displaced_batch(
        &self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        allocator: &Allocator,
        fns: &ClusterExtensionFns,
        encoder: &mut wgpu::CommandEncoder,
        level: u32,
        displaced_positions_addr: vk::DeviceAddress,
        base_triangle_count: u32,
        cluster_id_base: u32,
    ) -> InstantiatedClasBatch {
        let cluster_fns = fns
            .cluster
            .as_ref()
            .expect("tess_template.instantiate_displaced_batch: cluster-AS extension missing");
        let lvl0 = (level - 1) as usize;
        let tri_count = self.triangle_counts[lvl0];
        let vert_count = self.vertex_counts[lvl0];
        let n = base_triangle_count.max(1);

        // One INSTANTIATE descriptor per base triangle, each pointing at its own
        // contiguous micro-vertex region in the displaced-positions buffer. CLAS `i`
        // bakes `cluster_id_base + i` as its ClusterIDNV so the closest-hit can recover
        // a unique per-CLAS tess index for the smooth-normal metadata lookup.
        let descs: Vec<vk::ClusterAccelerationStructureInstantiateClusterInfoNV> = (0..n)
            .map(|i| vk::ClusterAccelerationStructureInstantiateClusterInfoNV {
                cluster_id_offset: cluster_id_base + i,
                geometry_index_offset_and_reserved: vk::Packed24_8::new(0, 0),
                cluster_template_address: self.template_addresses[lvl0],
                vertex_buffer: vk::StridedDeviceAddressNV {
                    start_address: displaced_positions_addr
                        + (i as u64) * (vert_count as u64) * 12,
                    stride_in_bytes: 12,
                },
            })
            .collect();
        let stride =
            size_of::<vk::ClusterAccelerationStructureInstantiateClusterInfoNV>() as u64;
        let descs_len = (n as u64) * stride;
        // SAFETY: InstantiateClusterInfoNV is repr(C) POD; flat byte view valid.
        let descs_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(descs.as_ptr().cast::<u8>(), descs_len as usize)
        };
        let src_infos = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tess_template.instantiate_batch.src_infos"),
            size: descs_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&src_infos, 0, descs_bytes);
        let src_infos_addr = allocator.wgpu_buffer_device_address(&src_infos);

        let count_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tess_template.instantiate_batch.count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&count_buf, 0, &n.to_le_bytes());
        let count_addr = allocator.wgpu_buffer_device_address(&count_buf);

        let mut triangle_input = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .max_geometry_index_value(0)
            .max_cluster_unique_geometry_count(1)
            .max_cluster_triangle_count(tri_count)
            .max_cluster_vertex_count(vert_count)
            .max_total_triangle_count(tri_count * n)
            .max_total_vertex_count(vert_count * n)
            .min_position_truncate_bit_count(0);
        let op_input = vk::ClusterAccelerationStructureOpInputNV {
            p_triangle_clusters: &mut triangle_input as *mut _,
        };
        let size_input = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(n)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
            )
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::INSTANTIATE_TRIANGLE_CLUSTER)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(op_input);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: input fully populated; function table loaded.
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
            "tess_template.instantiate_batch.storage",
        );
        let storage_addr = allocator.wgpu_buffer_device_address(&storage);

        let scratch = allocator.create_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            wgpu::BufferUsages::STORAGE,
            sizes.build_scratch_size.max(1) + 255,
            MemoryLocation::GpuOnly,
            "tess_template.instantiate_batch.scratch",
        );
        let scratch_addr = {
            let base = allocator.wgpu_buffer_device_address(&scratch);
            let m = base & 255;
            if m == 0 { base } else { base + (256 - m) }
        };

        let addr_array_size = (n as u64) * 8;
        let dst_addresses = allocator.create_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::TRANSFER_SRC,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            addr_array_size,
            MemoryLocation::GpuOnly,
            "tess_template.instantiate_batch.dst_addresses",
        );
        let dst_addresses_addr = allocator.wgpu_buffer_device_address(&dst_addresses);

        let cmd = vk::ClusterAccelerationStructureCommandsInfoNV {
            s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
            p_next: core::ptr::null_mut(),
            input: size_input,
            dst_implicit_data: storage_addr,
            scratch_data: scratch_addr,
            dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
                device_address: dst_addresses_addr,
                stride: 8,
                size: addr_array_size,
            },
            dst_sizes_array: vk::StridedDeviceAddressRegionKHR::default(),
            src_infos_array: vk::StridedDeviceAddressRegionKHR {
                device_address: src_infos_addr,
                stride,
                size: descs_len,
            },
            src_infos_count: count_addr,
            address_resolution_flags:
                vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
            _marker: core::marker::PhantomData,
        };
        // SAFETY: function table loaded; encoder open + Vulkan-backed; the
        // descriptor strides/addresses reference the buffers above. Leading
        // barrier: the displace compute's positions (or a prior instance's BLAS)
        // are visible as build input. Trailing barrier: the written CLAS addresses
        // are visible to the BLAS `cluster_references` read.
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
        InstantiatedClasBatch {
            clas_addresses: dst_addresses,
            storage,
            transient,
        }
    }

    /// Build a template for every level `1..=MAX_TESS_LEVEL` in one indirect
    /// cluster-AS build.
    ///
    /// # Panics
    ///
    /// Panics if the cluster-AS function table is missing or the build /
    /// readback fails — all setup-time failure modes, mirroring
    /// [`ClasArena::upload_mesh`](super::clas_arena).
    pub fn build(
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        allocator: &Allocator,
        fns: &ClusterExtensionFns,
    ) -> Self {
        const INDEX_TYPE_32BIT: u32 = 4;
        const OPAQUE_GEOMETRY_FLAG: u8 = 0b100;

        let cluster_fns = fns
            .cluster
            .as_ref()
            .expect("tess_template.build: cluster-AS extension function table missing");

        // Concatenate every level's reference topology into one vertex pool
        // (barycentric coords as positions) and one index pool, recording
        // each level's sub-range.
        let mut ref_verts: Vec<[f32; 3]> = Vec::new();
        let mut ref_indices: Vec<u32> = Vec::new();
        let mut triangle_counts: Vec<u32> = Vec::with_capacity(MAX_TESS_LEVEL as usize);
        let mut vertex_counts: Vec<u32> = Vec::with_capacity(MAX_TESS_LEVEL as usize);
        // Per-level (first_vertex, first_index) into the concatenated pools.
        let mut level_offsets: Vec<(u32, u32)> = Vec::with_capacity(MAX_TESS_LEVEL as usize);

        let (mut max_tris, mut max_verts) = (0u32, 0u32);
        for level in 1..=MAX_TESS_LEVEL {
            let subdiv = SubdividedTriangle::new(level);
            let first_vertex = ref_verts.len() as u32;
            let first_index = ref_indices.len() as u32;
            level_offsets.push((first_vertex, first_index));

            let v = subdiv.vertex_count() as u32;
            let t = subdiv.triangle_count() as u32;
            vertex_counts.push(v);
            triangle_counts.push(t);
            max_verts = max_verts.max(v);
            max_tris = max_tris.max(t);

            ref_verts.extend_from_slice(&subdiv.barycentrics);
            ref_indices.extend(subdiv.indices.iter().copied());
        }
        let total_verts = ref_verts.len() as u32;
        let total_tris = (ref_indices.len() / 3) as u32;
        let level_count = MAX_TESS_LEVEL as usize;

        // Upload the reference topology pools. BLAS_INPUT routes them through
        // the device-address path so the build can reference them.
        let verts_bytes: &[u8] = bytemuck_cast_slice(&ref_verts);
        let ref_verts_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tess_template.ref_verts"),
            size: verts_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&ref_verts_buf, 0, verts_bytes);
        let ref_verts_addr = allocator.wgpu_buffer_device_address(&ref_verts_buf);

        let indices_bytes: &[u8] = bytemuck_cast_slice(&ref_indices);
        let ref_indices_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tess_template.ref_indices"),
            size: indices_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&ref_indices_buf, 0, indices_bytes);
        let ref_indices_addr = allocator.wgpu_buffer_device_address(&ref_indices_buf);

        // One template descriptor per level.
        let mut descriptors: Vec<
            vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV,
        > = Vec::with_capacity(level_count);
        for lvl0 in 0..level_count {
            let (first_vertex, first_index) = level_offsets[lvl0];
            descriptors.push(
                vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV {
                    cluster_id: 0,
                    cluster_flags: vk::ClusterAccelerationStructureClusterFlagsNV::default(),
                    triangle_cluster_info_packed: vk::Packed9_9_6_4_4::new(
                        triangle_counts[lvl0],
                        vertex_counts[lvl0],
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
                    index_buffer: ref_indices_addr + u64::from(first_index) * 4,
                    vertex_buffer: ref_verts_addr + u64::from(first_vertex) * 12,
                    geometry_index_and_flags_buffer: 0,
                    opacity_micromap_array: 0,
                    opacity_micromap_index_buffer: 0,
                    instantiation_bounding_box_limit: 0,
                },
            );
        }

        let desc_stride =
            size_of::<vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV>() as u64;
        let desc_bytes_len = (descriptors.len() as u64) * desc_stride;
        // SAFETY: template info is repr(C) POD; flat byte view is valid.
        let desc_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(descriptors.as_ptr().cast::<u8>(), desc_bytes_len as usize)
        };
        let src_infos_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tess_template.src_infos"),
            size: desc_bytes_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&src_infos_buf, 0, desc_bytes);
        let src_infos_addr = allocator.wgpu_buffer_device_address(&src_infos_buf);

        let count_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tess_template.count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&count_buf, 0, &(level_count as u32).to_le_bytes());
        let count_addr = allocator.wgpu_buffer_device_address(&count_buf);

        // Size query (templates have geometry index 0, so
        // `max_geometry_index_value` is 0).
        let mut triangle_input = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .max_geometry_index_value(0)
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
            .max_acceleration_structure_count(level_count as u32)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
            )
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER_TEMPLATE)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(op_input);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: input fully populated; function table loaded above.
        unsafe {
            cluster_fns.get_cluster_acceleration_structure_build_sizes(&size_input, &mut sizes);
        }

        // Persistent template storage (kept in the resource), plus transient
        // scratch + a readback of the driver-chosen template addresses.
        let template_storage = allocator.create_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            wgpu::BufferUsages::STORAGE,
            sizes.acceleration_structure_size.max(1),
            MemoryLocation::GpuOnly,
            "tess_template.storage",
        );
        let template_storage_addr = allocator.wgpu_buffer_device_address(&template_storage);

        let scratch = allocator.create_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            wgpu::BufferUsages::STORAGE,
            sizes.build_scratch_size.max(1) + 255,
            MemoryLocation::GpuOnly,
            "tess_template.scratch",
        );
        let scratch_addr = {
            let base = allocator.wgpu_buffer_device_address(&scratch);
            let m = base & 255;
            if m == 0 { base } else { base + (256 - m) }
        };

        let addr_array_size = (level_count as u64) * 8;
        let dst_addresses = allocator.create_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::TRANSFER_SRC,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            addr_array_size,
            MemoryLocation::GpuOnly,
            "tess_template.dst_addresses",
        );
        let dst_addresses_addr = allocator.wgpu_buffer_device_address(&dst_addresses);

        let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("tess_template.build"),
        });
        let cmd = vk::ClusterAccelerationStructureCommandsInfoNV {
            s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
            p_next: core::ptr::null_mut(),
            input: size_input,
            dst_implicit_data: template_storage_addr,
            scratch_data: scratch_addr,
            dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
                device_address: dst_addresses_addr,
                stride: 8,
                size: addr_array_size,
            },
            dst_sizes_array: vk::StridedDeviceAddressRegionKHR::default(),
            src_infos_array: vk::StridedDeviceAddressRegionKHR {
                device_address: src_infos_addr,
                stride: desc_stride,
                size: desc_bytes_len,
            },
            src_infos_count: count_addr,
            address_resolution_flags:
                vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
            _marker: core::marker::PhantomData,
        };
        // SAFETY: function table loaded; encoder open + Vulkan-backed; the
        // descriptor strides/addresses reference the buffers above. No wgpu
        // commands touch this encoder afterward.
        unsafe {
            crate::gpu::extension::cmd_build_cluster_acceleration_structures_indirect(
                &mut encoder,
                fns,
                &cmd,
            );
            crate::gpu::extension::cmd_global_as_barrier(&mut encoder, render_device, false);
        }
        let build_idx = render_queue.submit([encoder.finish()]);
        {
            let _span = tracing::info_span!("tess_template.build_poll_wait").entered();
            let _ = render_device.wgpu_device().poll(wgpu::PollType::Wait {
                submission_index: Some(build_idx),
                timeout: None,
            });
        }

        // Read the per-level template addresses back to the CPU.
        let readback = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tess_template.addr_readback"),
            size: addr_array_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut rb_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("tess_template.addr_copy"),
        });
        rb_encoder.copy_buffer_to_buffer(&dst_addresses, 0, &readback, 0, addr_array_size);
        let rb_idx = render_queue.submit([rb_encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |result| {
            result.expect("tess_template.build: template address readback map failed");
        });
        {
            let _span = tracing::info_span!("tess_template.addr_readback_wait").entered();
            let _ = render_device.wgpu_device().poll(wgpu::PollType::Wait {
                submission_index: Some(rb_idx),
                timeout: None,
            });
        }
        let template_addresses: Vec<u64> = {
            let mapped = readback.slice(..).get_mapped_range();
            mapped
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        readback.unmap();

        Self {
            template_addresses,
            triangle_counts,
            vertex_counts,
            _template_storage: template_storage,
        }
    }
}

/// Phase 2c brick B3 — record (no submit) a **per-instance** bottom-level
/// acceleration structure over `cluster_count` CLASes whose device addresses live
/// GPU-side at `cluster_references` (e.g. the
/// [`InstantiatedClasBatch::clas_addresses`] buffer's device address) — the
/// addresses are read straight from that buffer, never uploaded from the CPU. The
/// driver writes the chosen BLAS address (IMPLICIT) into the returned
/// [`InstanceBlas::blas_address_buf`]; the caller reads it back after the submit.
///
/// This is the architectural fork from solari's shared per-`(geometry, LOD)`
/// BLAS: a tessellated instance gets its *own* BLAS over its generated CLAS.
/// Mirrors the `BUILD_CLUSTERS_BOTTOM_LEVEL` op in [`crate::accel::blas_rebuild`].
/// The driver writes the chosen BLAS device address (IMPLICIT) to `blas_dst_addr`
/// — a device address into the caller's persistent per-instance address buffer
/// (e.g. `+ instance_index * 8`), consumed GPU-side, never read back. The caller
/// brackets this build with AS barriers and submits once.
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
    let src_infos_addr = allocator.wgpu_buffer_device_address(&src_infos);

    let count_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tess_template.blas.count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    render_queue.write_buffer(&count_buf, 0, &1u32.to_le_bytes());
    let count_addr = allocator.wgpu_buffer_device_address(&count_buf);

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
    let storage_addr = allocator.wgpu_buffer_device_address(&storage);

    let scratch = allocator.create_buffer(
        render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        wgpu::BufferUsages::STORAGE,
        sizes.build_scratch_size.max(1) + 255,
        MemoryLocation::GpuOnly,
        "tess_template.blas.scratch",
    );
    let scratch_addr = {
        let base = allocator.wgpu_buffer_device_address(&scratch);
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
    // barrier: the BLAS address + payload are visible to the readback copy and the
    // later trace.
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

/// `RenderStartup` system: build the per-level templates once, iff the
/// cluster-AS extension is available (i.e. [`Allocator`] +
/// [`ClusterExtensionFns`] are present).
pub fn init_tessellation_templates(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    allocator: Option<Res<Allocator>>,
    fns: Option<Res<ClusterExtensionFns>>,
) {
    // Built once at startup on any cluster-AS-capable device (a small raw-VK build +
    // readback of the per-level templates); the displacement showcase instantiates
    // them whenever a scene has displacement-mapped instances.
    let (Some(allocator), Some(fns)) = (allocator, fns) else {
        return;
    };
    if fns.cluster.is_none() {
        return;
    }
    let templates =
        TessellationTemplates::build(&render_device, &render_queue, &allocator, &fns);
    commands.insert_resource(templates);
}

/// Flat byte view of a `[f32; 3]` / `u32` slice for upload. Avoids a
/// `bytemuck` dependency on `[f32; 3]` (which isn't `Pod` without the
/// feature); both element types are plain repr-transparent PODs.
fn bytemuck_cast_slice<T: Copy>(slice: &[T]) -> &[u8] {
    // SAFETY: `T` here is only ever `[f32; 3]` or `u32` (both POD with no
    // padding or invalid bit patterns); a read-only byte view over the
    // slice's contiguous storage is well-defined and outlives the borrow.
    unsafe {
        core::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), size_of_val(slice))
    }
}
