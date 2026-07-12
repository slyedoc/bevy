// The per-pattern tessellation table + its cluster templates, ported from
// `vk_tessellated_clusters/src/tessellation_table.cpp`.
//
// Where that sample builds templates with 8-bit cluster indices, the table's
// packed 8-bit indices are expanded to 32-bit and the UV-packed barycentrics to
// `vec3` for the CLAS-template build. The output is a per-edge-segment lookup
// table of cluster-template addresses that the GPU classify/tessellate passes
// index by a triangle's three edge factors — giving crack-free adaptive
// tessellation.
#![allow(unsafe_code, reason = "raw VK cluster-AS template build via as_hal_mut")]

use ash::vk::{self, TaggedStructure};
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Res};
use bevy_render::render_resource::Buffer;
use bevy_render::renderer::{RenderDevice, RenderQueue};
use bytemuck::{Pod, Zeroable};
use wgpu::CommandEncoderDescriptor;

use crate::gpu::allocator::{Allocator, MemoryLocation};
use crate::gpu::extension::ClusterExtensionFns;
use super::tess_table_data as table;

/// One subdivision pattern's slice into the shared vertex / triangle pools.
/// Matches `vk_tessellated_clusters`' `ConfigEntry` and the raw `CONFIGS` layout
/// (4 × `u16`). The GPU tessellate passes read this (keyed by
/// [`TessellationTable::lookup_index`]) to decode a triangle's microtopology.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ConfigEntry {
    pub first_triangle: u16,
    pub first_vertex: u16,
    pub num_triangles: u16,
    pub num_vertices: u16,
}

/// Render-world resource: the adaptive-tessellation pattern table and its
/// per-pattern cluster templates. Built once on a cluster-AS-capable device.
///
/// `template_addresses` is indexed by [`Self::lookup_index`] (the three per-edge
/// segment counts), with `0` in unused slots. The flipped-winding template is
/// stored for the mirrored permutation `(x, z, y)`, so any edge ordering resolves
/// to a watertight pattern.
#[derive(Resource)]
pub struct TessellationTable {
    /// Max segments per edge the table covers (11).
    pub max_size: u32,
    /// `max_size` rounded up to a power of two (16) — the lookup stride per axis.
    pub max_size_configs: u32,
    /// `max_size_configs³` — the lookup table length (4096).
    pub num_configs: usize,
    /// Largest per-pattern triangle / vertex counts (cluster-AS sizing).
    pub max_triangles: u32,
    pub max_vertices: u32,

    /// Shader-visible table data (consumed by the GPU tessellate passes):
    /// UV-packed barycentrics, 8-bit-packed triangle indices, and the
    /// lookup-indexed [`ConfigEntry`] array.
    pub vertices: Buffer,
    pub indices: Buffer,
    pub configs: Buffer,
    /// `u64` cluster-template address per lookup slot (`0` = unused).
    pub template_addresses: Buffer,

    /// Backing AS storage for every template — kept alive for the resource's
    /// lifetime since `template_addresses` point into it.
    _template_storage: wgpu::Buffer,
}

impl TessellationTable {
    /// Lookup index for a triangle whose three edges carry `(x, y, z)` segments
    /// (each in `1..=max_size`). Mirrors `vk_tessellated_clusters`' power-of-two
    /// addressing, biased so `(1,1,1)` maps to 0.
    #[inline]
    pub fn lookup_index(&self, x: u32, y: u32, z: u32) -> usize {
        let s = self.max_size_configs;
        (x + y * s + z * s * s - (1 + s + s * s)) as usize
    }

    /// Build the table + cluster templates. Panics on cluster-AS setup failure
    /// (all are device-creation-time errors, like the rest of the cluster path).
    ///
    /// `position_truncate_bit_count` matches the value the per-cluster
    /// instantiation will use (0 for full precision).
    pub fn build(
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        allocator: &Allocator,
        fns: &ClusterExtensionFns,
        position_truncate_bit_count: u32,
    ) -> Self {
        // NV cluster index type for 32-bit indices (the only one this path uses).
        const INDEX_TYPE_32BIT: u32 = 4;
        const OPAQUE_GEOMETRY_FLAG: u8 = 0b100;

        let cluster_fns = fns
            .cluster
            .as_ref()
            .expect("tess_table.build: cluster-AS extension function table missing");

        let max_size = table::MAX_EDGE_SEGMENTS;
        let max_size_configs = max_size.next_power_of_two();
        let num_configs = (max_size_configs * max_size_configs * max_size_configs) as usize;
        let max_configs = table::MAX_CONFIGS;
        // Two template sets: [0, max_configs) normal winding, [max_configs, 2*max_configs)
        // flipped — used for the mirrored (x, z, y) permutation.
        let total_templates = max_configs * 2;

        let entries = |i: usize| ConfigEntry {
            first_triangle: table::CONFIGS[i * 4],
            first_vertex: table::CONFIGS[i * 4 + 1],
            num_triangles: table::CONFIGS[i * 4 + 2],
            num_vertices: table::CONFIGS[i * 4 + 3],
        };

        // ── Expand the packed table into cluster-AS template inputs ──────────
        // Vertices: UV-packed barycentrics → vec3 (z = 0). Flipped set negates the
        // first barycentric (x' = 1 - x - y) so the mirrored pattern stays valid.
        let mut verts: Vec<[f32; 3]> = Vec::with_capacity(table::VERTICES.len() * 2);
        for &v in table::VERTICES.iter() {
            let u = (v & 0xFFFF) as f32 / 32768.0;
            let w = (v >> 16) as f32 / 32768.0;
            verts.push([u, w, 0.0]);
        }
        for &v in table::VERTICES.iter() {
            let u = (v & 0xFFFF) as f32 / 32768.0;
            let w = (v >> 16) as f32 / 32768.0;
            verts.push([1.0 - u - w, w, 0.0]);
        }
        // Triangle indices: packed 3 × 8-bit → 32-bit. Flipped set swaps the last
        // two indices (winding flip).
        let mut indices32: Vec<u32> = Vec::with_capacity(table::TRIANGLES.len() * 3 * 2);
        for &t in table::TRIANGLES.iter() {
            indices32.push(t & 0xFF);
            indices32.push((t >> 8) & 0xFF);
            indices32.push((t >> 16) & 0xFF);
        }
        for &t in table::TRIANGLES.iter() {
            indices32.push(t & 0xFF);
            indices32.push((t >> 16) & 0xFF);
            indices32.push((t >> 8) & 0xFF);
        }
        let verts_per_set = table::VERTICES.len() as u64; // vec3 stride 12
        let tris_per_set = table::TRIANGLES.len() as u64; // 3 indices each, stride 4

        let verts_bytes: &[u8] = bytemuck::cast_slice(&verts);
        let verts_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tess_table.template_verts"),
            size: verts_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&verts_buf, 0, verts_bytes);
        let verts_addr = allocator.wgpu_buffer_device_address(&verts_buf).get();

        let idx_bytes: &[u8] = bytemuck::cast_slice(&indices32);
        let idx_buf = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tess_table.template_indices"),
            size: idx_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&idx_buf, 0, idx_bytes);
        let idx_addr = allocator.wgpu_buffer_device_address(&idx_buf).get();

        // ── Template descriptors (normal then flipped) ──────────────────────
        let mut max_tris = 0u32;
        let mut max_verts = 0u32;
        let mut total_tris = 0u64;
        let mut total_verts = 0u64;
        let mut descriptors: Vec<
            vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV,
        > = Vec::with_capacity(total_templates);
        for c in 0..total_templates {
            let flipped = c >= max_configs;
            let e = entries(c % max_configs);
            max_tris = max_tris.max(e.num_triangles as u32);
            max_verts = max_verts.max(e.num_vertices as u32);
            total_tris += e.num_triangles as u64;
            total_verts += e.num_vertices as u64;

            let vset = if flipped { verts_per_set } else { 0 };
            let tset = if flipped { tris_per_set } else { 0 };
            let vertex_buffer = verts_addr + (vset + e.first_vertex as u64) * 12;
            let index_buffer = idx_addr + (tset + e.first_triangle as u64) * 3 * 4;

            descriptors.push(
                vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV {
                    cluster_id: 0,
                    cluster_flags: vk::ClusterAccelerationStructureClusterFlagsNV::default(),
                    triangle_cluster_info_packed: vk::Packed9_9_6_4_4::new(
                        e.num_triangles as u32,
                        e.num_vertices as u32,
                        position_truncate_bit_count,
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
                    index_buffer,
                    vertex_buffer,
                    geometry_index_and_flags_buffer: 0,
                    opacity_micromap_array: 0,
                    opacity_micromap_index_buffer: 0,
                    instantiation_bounding_box_limit: 0,
                },
            );
        }

        // ── Build all templates (IMPLICIT destinations, read addresses back) ──
        let template_addresses_raw = Self::build_templates(
            render_device,
            render_queue,
            allocator,
            fns,
            cluster_fns,
            &descriptors,
            max_tris,
            max_verts,
            total_tris as u32,
            total_verts as u32,
        );

        // ── Remap (raw config order) → (lookup-indexed) tables, with symmetry ─
        let mut configs_lut = vec![ConfigEntry::default(); num_configs];
        let mut addresses_lut = vec![0u64; num_configs];

        let s = max_size_configs;
        let lookup = |x: u32, y: u32, z: u32| -> usize {
            (x + y * s + z * s * s - (1 + s + s * s)) as usize
        };
        let mut config_idx = 0usize;
        for x in 1..=max_size {
            for y in 1..=x {
                for z in 1..=y {
                    let e = entries(config_idx);
                    // Normal winding at (x, y, z).
                    let li = lookup(x, y, z);
                    configs_lut[li] = e;
                    addresses_lut[li] = template_addresses_raw.addresses[config_idx];
                    // Mirrored permutation (x, z, y) uses the flipped template.
                    if z != y && x > 1 {
                        let lf = lookup(x, z, y);
                        let fc = config_idx + max_configs;
                        configs_lut[lf] = e;
                        addresses_lut[lf] = template_addresses_raw.addresses[fc];
                    }
                    config_idx += 1;
                }
            }
        }
        debug_assert_eq!(config_idx, max_configs);

        // ── Upload shader-visible buffers ───────────────────────────────────
        let make_storage = |label: &'static str, bytes: &[u8]| -> Buffer {
            let buf = render_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.len().max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            render_queue.write_buffer(&buf, 0, bytes);
            buf
        };
        let vertices = make_storage("tess_table.vertices", bytemuck::cast_slice(&table::VERTICES));
        let indices = make_storage("tess_table.indices", bytemuck::cast_slice(&table::TRIANGLES));
        let configs = make_storage("tess_table.configs", bytemuck::cast_slice(&configs_lut));
        let template_addresses =
            make_storage("tess_table.template_addresses", bytemuck::cast_slice(&addresses_lut));

        tracing::debug!(
            "tess_table: built {} patterns ({} templates incl. flips), max_tris={} max_verts={} \
             num_configs={}",
            max_configs,
            total_templates,
            max_tris,
            max_verts,
            num_configs,
        );

        Self {
            max_size,
            max_size_configs,
            num_configs,
            max_triangles: max_tris,
            max_vertices: max_verts,
            vertices,
            indices,
            configs,
            template_addresses,
            _template_storage: template_addresses_raw.storage,
        }
    }

    /// `BUILD_TRIANGLE_CLUSTER_TEMPLATE` over every descriptor (IMPLICIT
    /// destinations), reading the driver-chosen template addresses back to the
    /// CPU. Returns the addresses (raw config order, incl. flips) + the backing
    /// storage to keep alive.
    fn build_templates(
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        allocator: &Allocator,
        fns: &ClusterExtensionFns,
        cluster_fns: &ash::nv::cluster_acceleration_structure::Device,
        descriptors: &[vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV],
        max_tris: u32,
        max_verts: u32,
        total_tris: u32,
        total_verts: u32,
    ) -> TemplateBuildResult {
        let count = descriptors.len() as u32;
        let desc_stride =
            size_of::<vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV>() as u64;
        let desc_bytes_len = (descriptors.len() as u64) * desc_stride;
        // SAFETY: template info is repr(C) POD; flat byte view valid.
        let desc_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(descriptors.as_ptr().cast::<u8>(), desc_bytes_len as usize)
        };
        let src_infos = blas_input_buffer(render_device, render_queue, "tess_table.tpl.src", desc_bytes);
        let src_infos_addr = allocator.wgpu_buffer_device_address(&src_infos).get();
        let count_buf =
            blas_input_buffer(render_device, render_queue, "tess_table.tpl.count", &count.to_le_bytes());
        let count_addr = allocator.wgpu_buffer_device_address(&count_buf).get();

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
            .max_acceleration_structure_count(count)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
            )
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER_TEMPLATE)
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
            "tess_table.template_storage",
        );
        let storage_addr = allocator.wgpu_buffer_device_address(&storage).get();
        let scratch = allocator.create_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            wgpu::BufferUsages::STORAGE,
            sizes.build_scratch_size.max(1) + 255,
            MemoryLocation::GpuOnly,
            "tess_table.template_scratch",
        );
        let scratch_addr = align256(allocator.wgpu_buffer_device_address(&scratch).get());

        let addr_array_size = (count as u64) * 8;
        let dst_addresses = allocator.create_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::TRANSFER_SRC,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            addr_array_size,
            MemoryLocation::GpuOnly,
            "tess_table.template_dst_addresses",
        );
        let dst_addresses_addr = allocator.wgpu_buffer_device_address(&dst_addresses).get();

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
                stride: desc_stride,
                size: desc_bytes_len,
            },
            src_infos_count: count_addr,
            address_resolution_flags:
                vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
            _marker: core::marker::PhantomData,
        };
        let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("tess_table.template_build"),
        });
        // SAFETY: fns loaded; encoder Vulkan-backed; descriptors reference live buffers.
        unsafe {
            crate::gpu::extension::cmd_build_cluster_acceleration_structures_indirect(
                &mut encoder, fns, &cmd,
            );
            crate::gpu::extension::cmd_global_as_barrier(&mut encoder, render_device, false);
        }
        let addresses = submit_and_read_u64(render_device, render_queue, encoder, &dst_addresses, count);

        TemplateBuildResult { addresses, storage }
    }
}

/// `RenderStartup`: build the per-pattern tessellation table once on a
/// cluster-AS-capable device and insert it as a resource. No-op otherwise (the
/// GPU tessellate passes guard on the resource's presence).
pub fn init_tessellation_table(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    allocator: Option<Res<Allocator>>,
    fns: Option<Res<ClusterExtensionFns>>,
) {
    let (Some(allocator), Some(fns)) = (allocator, fns) else {
        return;
    };
    if fns.cluster.is_none() {
        return;
    }
    let table = TessellationTable::build(&render_device, &render_queue, &allocator, &fns, 0);
    commands.insert_resource(table);
}

struct TemplateBuildResult {
    addresses: Vec<u64>,
    storage: wgpu::Buffer,
}

/// Round a device address up to 256 (NV cluster-AS scratch alignment).
fn align256(addr: u64) -> u64 {
    let m = addr & 255;
    if m == 0 { addr } else { addr + (256 - m) }
}

/// Small device-address-readable input buffer (src-infos / count for the builds).
fn blas_input_buffer(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    label: &'static str,
    bytes: &[u8],
) -> Buffer {
    let buf = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    render_queue.write_buffer(&buf, 0, bytes);
    buf
}

/// Submit `encoder`, wait, then copy `count` × `u64` out of `src` to the CPU.
fn submit_and_read_u64(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    encoder: wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    count: u32,
) -> Vec<u64> {
    let bytes = (count as u64) * 8;
    let build_idx = render_queue.submit([encoder.finish()]);
    let _ = render_device.wgpu_device().poll(wgpu::PollType::Wait {
        submission_index: Some(build_idx),
        timeout: None,
    });
    let readback = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tess_table.readback_u64"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut rb = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("tess_table.readback_copy_u64"),
    });
    rb.copy_buffer_to_buffer(src, 0, &readback, 0, bytes);
    let rb_idx = render_queue.submit([rb.finish()]);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.expect("tess_table: u64 readback map failed"));
    let _ = render_device.wgpu_device().poll(wgpu::PollType::Wait {
        submission_index: Some(rb_idx),
        timeout: None,
    });
    let out: Vec<u64> = {
        let mapped = readback.slice(..).get_mapped_range();
        mapped
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    readback.unmap();
    out
}
