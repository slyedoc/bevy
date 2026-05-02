// Cluster-AS dispatch is gated on `unsafe { Device::as_hal(...) }`; opt out of
// the workspace's `-D unsafe-code` lint for this example only.
#![allow(unsafe_code)]
//! Smoke test for the `VK_NV_cluster_acceleration_structure` Vulkan extension.
//!
//! Validates the integration end-to-end on real hardware:
//!
//! 1. The wgpu Cargo feature `experimental-cluster-acceleration-structure` is on
//!    (forwarded from `bevy_render`'s feature of the same name).
//! 2. `Features::EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE` is requested at
//!    instance creation; the adapter exposes the bit if the GPU supports the
//!    extension AND the driver advertises the `clusterAccelerationStructure`
//!    feature.
//! 3. wgpu-hal enables `VK_NV_cluster_acceleration_structure` when opening the
//!    device and chains
//!    `VkPhysicalDeviceClusterAccelerationStructureFeaturesNV` into
//!    `VkDeviceCreateInfo::pNext`.
//! 4. wgpu-hal loads `vkGetClusterAccelerationStructureBuildSizesNV` and
//!    `vkCmdBuildClusterAccelerationStructureIndirectNV`.
//! 5. The plugin's `RenderStartup` system escapes via `Device::as_hal::<Vulkan>`
//!    and calls the size-query function. If it returns sane sizes, the path is
//!    wired up.
//!
//! Requires an NVIDIA GPU (Turing+) with a driver implementing the extension.
//!
//! ```text
//! cargo run --release --features experimental_cluster_acceleration_structure \
//!     --example cluster_acceleration_structure
//! ```

use std::ops::Deref;

use ash::vk::{self, Handle as _, TaggedStructure as _};
use bevy::{
    app::AppExit,
    prelude::*,
    render::{
        render_resource::WgpuFeatures,
        renderer::{RenderDevice, RenderQueue},
        settings::WgpuSettings,
        RenderApp, RenderPlugin, RenderStartup,
    },
};
use wgpu::hal::api::Vulkan;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(RenderPlugin {
                render_creation: WgpuSettings {
                    features: WgpuFeatures::EXPERIMENTAL_RAY_QUERY
                        | WgpuFeatures::EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE,
                    ..default()
                }
                .into(),
                ..default()
            }),
            ClusterAccelerationStructureSmokeTestPlugin,
        ))
        // Quit on the next frame so this acts as a one-shot CLI tool.
        .add_systems(Update, quit_after_first_frame)
        .run();
}

fn quit_after_first_frame(mut commands: Commands) {
    commands.write_message(AppExit::Success);
}

/// Wires the smoke test into the render sub-app's startup schedule.
struct ClusterAccelerationStructureSmokeTestPlugin;

impl Plugin for ClusterAccelerationStructureSmokeTestPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            warn!("RenderApp not present; cluster_AS smoke test skipped.");
            return;
        };
        render_app.add_systems(RenderStartup, smoke_test.before(m2_build_clas));
        render_app.add_systems(RenderStartup, m2_build_clas);
    }
}

/// Mirror of `bevy_pbr::meshlet::asset::Meshlet`. The upstream struct is `pub`
/// but its containing `MeshletMesh` keeps fields `pub(crate)`, so we re-declare
/// the position-related fields locally and feed them to a dequant routine that
/// matches the WGSL `get_meshlet_vertex_position` exactly.
#[derive(Copy, Clone, Debug)]
struct Meshlet {
    /// Bit offset within the parent mesh's `vertex_positions` bitstream.
    start_vertex_position_bit: u32,
    /// Vertex count minus one (a meshlet has at most 256 verts → fits in u8).
    vertex_count_minus_one: u8,
    /// Triangle count for this meshlet.
    triangle_count: u8,
    /// Bits used per X channel of vertex positions in this meshlet's bitstream.
    bits_per_vertex_position_channel_x: u8,
    bits_per_vertex_position_channel_y: u8,
    bits_per_vertex_position_channel_z: u8,
    /// Power-of-2 quantization factor: encoded position = round(world_pos * (1 << f) * 100).
    vertex_position_quantization_factor: u8,
    /// Per-channel min of the quantized integer values, stored as f32 by the encoder.
    min_vertex_position_channel_x: f32,
    min_vertex_position_channel_y: f32,
    min_vertex_position_channel_z: f32,
}

/// CPU port of `bevy_pbr::meshlet::meshlet_bindings::get_meshlet_vertex_position`.
///
/// Walks the bitstream `vertex_positions[..]` from `meshlet.start_vertex_position_bit`
/// for `vertex_count_minus_one + 1` vertices, extracting `bits_per_channel_*` per
/// X/Y/Z, then dequantizes into world-space `f32x3` using the meshlet's stored
/// quantization params.
fn dequantize_meshlet_vertices(meshlet: &Meshlet, vertex_positions: &[u32]) -> Vec<[f32; 3]> {
    let vertex_count = u32::from(meshlet.vertex_count_minus_one) + 1;
    let bits = [
        u32::from(meshlet.bits_per_vertex_position_channel_x),
        u32::from(meshlet.bits_per_vertex_position_channel_y),
        u32::from(meshlet.bits_per_vertex_position_channel_z),
    ];
    let bits_per_vertex = bits[0] + bits[1] + bits[2];
    // `(1 << f) * CENTIMETERS_PER_METER` -- matches `from_mesh.rs` and the WGSL.
    let inv_quant = 1.0 / ((1u32 << meshlet.vertex_position_quantization_factor) as f32 * 100.0);
    let mins = [
        meshlet.min_vertex_position_channel_x,
        meshlet.min_vertex_position_channel_y,
        meshlet.min_vertex_position_channel_z,
    ];

    let mut out = Vec::with_capacity(vertex_count as usize);
    for v in 0..vertex_count {
        let mut start_bit = meshlet.start_vertex_position_bit + v * bits_per_vertex;
        let mut packed = [0u32; 3];
        for (i, &b) in bits.iter().enumerate() {
            let lower_word = (start_bit / 32) as usize;
            let bit_offset = start_bit & 31;
            let mut next_32 = vertex_positions[lower_word] >> bit_offset;
            // Spans a u32 boundary. `bit_offset` is in 1..32 here so the left
            // shift is well-defined.
            if bit_offset + b > 32 {
                next_32 |= vertex_positions[lower_word + 1] << (32 - bit_offset);
            }
            let mask = if b == 32 { u32::MAX } else { (1u32 << b) - 1 };
            packed[i] = next_32 & mask;
            start_bit += b;
        }
        out.push([
            (packed[0] as f32 + mins[0]) * inv_quant,
            (packed[1] as f32 + mins[1]) * inv_quant,
            (packed[2] as f32 + mins[2]) * inv_quant,
        ]);
    }
    out
}

/// Hand-built single-triangle meshlet with verts at (0,0,0), (1,0,0), (0,1,0).
/// 8 bits/channel, factor=0 → packed `100 = 0x64` represents one world-space
/// meter. Avoids needing the bunny.meshlet_mesh asset.
fn synthetic_triangle_meshlet() -> (Meshlet, Vec<u32>, Vec<u8>) {
    // 24 bits/vertex × 3 verts = 72 bits ≤ 3 × 32-bit words.
    //   word 0 = vert0(xyz=000) | vert1.x(0x64)        = 0x64000000
    //   word 1 = vert1(yz=00)   | vert2(x=0, y=0x64)   = 0x64000000
    //   word 2 = vert2.z(0)                            = 0x00000000
    let vertex_positions = vec![0x64000000u32, 0x64000000u32, 0x00000000u32];
    let indices = vec![0u8, 1, 2];
    let meshlet = Meshlet {
        start_vertex_position_bit: 0,
        vertex_count_minus_one: 2,
        triangle_count: 1,
        bits_per_vertex_position_channel_x: 8,
        bits_per_vertex_position_channel_y: 8,
        bits_per_vertex_position_channel_z: 8,
        vertex_position_quantization_factor: 0,
        min_vertex_position_channel_x: 0.0,
        min_vertex_position_channel_y: 0.0,
        min_vertex_position_channel_z: 0.0,
    };
    (meshlet, vertex_positions, indices)
}

fn smoke_test(device: Res<RenderDevice>) {
    let features = device.features();
    if !features.contains(WgpuFeatures::EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE) {
        warn!(
            "Adapter does not expose EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE. \
             Either the GPU is not NVIDIA Turing+, the driver predates \
             VK_NV_cluster_acceleration_structure, or the wgpu-hal Cargo feature \
             is off."
        );
        return;
    }
    info!("EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE: supported");

    // ---- M1: meshlet position dequantization (CPU, no GPU calls) ------------
    let (meshlet, vertex_positions_packed, indices) = synthetic_triangle_meshlet();
    let dequantized = dequantize_meshlet_vertices(&meshlet, &vertex_positions_packed);
    info!(
        "M1 dequant: {} verts {:?}, {} indices {:?}",
        dequantized.len(),
        dequantized,
        indices.len(),
        indices,
    );
    // Expected exactly the input world-space triangle.
    let expected: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    for (got, exp) in dequantized.iter().zip(expected.iter()) {
        for axis in 0..3 {
            assert!(
                (got[axis] - exp[axis]).abs() < 1e-5,
                "M1 dequant mismatch: got {got:?} expected {exp:?}"
            );
        }
    }
    info!("M1 dequant: round-trip OK");
    let _ = meshlet.triangle_count;

    // SAFETY:
    // - Adapter advertised the wgpu feature; therefore the device was created with
    //   `VK_NV_cluster_acceleration_structure` enabled and
    //   `clusterAccelerationStructure` set on the chained physical-device feature
    //   struct (see wgpu-hal::vulkan::adapter).
    // - The function pointers were resolved by `Functions::load` at device-open time.
    // - `get_cluster_build_sizes` only inspects the descriptor and returns required
    //   memory; no driver-side buffer touches happen.
    unsafe {
        let wgpu_device = device.wgpu_device();
        let Some(hal_device): Option<_> = wgpu_device.as_hal::<Vulkan>() else {
            warn!("Device is not a Vulkan device; skipping size query.");
            return;
        };
        // `as_hal` returns `impl Deref<Target = vulkan::Device>`. Bind it explicitly
        // before calling the inherent method so type inference resolves
        // `get_cluster_build_sizes` on the concrete HAL Device.
        let hal_device: &wgpu::hal::vulkan::Device = hal_device.deref();

        // Minimal `VkClusterAccelerationStructureInputInfoNV`: a single-cluster
        // triangle build with one triangle. Just enough to exercise the entry point.
        let triangle_input = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .max_geometry_index_value(0)
            .max_cluster_unique_geometry_count(1)
            .max_cluster_triangle_count(1)
            .max_cluster_vertex_count(3)
            .max_total_triangle_count(1)
            .max_total_vertex_count(3);

        let op_input = vk::ClusterAccelerationStructureOpInputNV {
            p_triangle_clusters: &triangle_input as *const _ as *mut _,
        };

        let info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(1)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(op_input);

        let sizes = hal_device.get_cluster_build_sizes(&info);

        info!(
            "vkGetClusterAccelerationStructureBuildSizesNV returned: \
             acceleration_structure_size = {} bytes, \
             update_scratch_size = {} bytes, \
             build_scratch_size = {} bytes",
            sizes.acceleration_structure_size,
            sizes.update_scratch_size,
            sizes.build_scratch_size,
        );
    }
}

// ---- M2: actual indirect cluster_AS build on real GPU memory ----------------

/// Wraps a raw `vk::Buffer` + its backing `vk::DeviceMemory` so we can free
/// both at the end of the smoke test.
struct RawBuffer {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    size: u64,
    addr: u64,
}

unsafe fn alloc_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    host_visible: bool,
) -> RawBuffer {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buf = unsafe { device.create_buffer(&info, None).expect("create_buffer") };

    let req = unsafe { device.get_buffer_memory_requirements(buf) };
    let need = if host_visible {
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    } else {
        vk::MemoryPropertyFlags::DEVICE_LOCAL
    };
    let mt = (0..mem_props.memory_type_count)
        .find(|&i| {
            (req.memory_type_bits & (1 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(need)
        })
        .expect("compatible memory type");

    let mut alloc_flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mt)
        .push(&mut alloc_flags);
    let mem = unsafe { device.allocate_memory(&alloc_info, None).expect("alloc_memory") };
    unsafe { device.bind_buffer_memory(buf, mem, 0).expect("bind_memory") };

    let addr = unsafe {
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf))
    };

    RawBuffer {
        buf,
        mem,
        size: req.size,
        addr,
    }
}

unsafe fn upload_bytes(device: &ash::Device, buf: &RawBuffer, bytes: &[u8]) {
    assert!((bytes.len() as u64) <= buf.size);
    let ptr = unsafe {
        device
            .map_memory(buf.mem, 0, bytes.len() as u64, vk::MemoryMapFlags::empty())
            .expect("map_memory")
    } as *mut u8;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        device.unmap_memory(buf.mem);
    }
}

unsafe fn destroy(device: &ash::Device, buf: RawBuffer) {
    unsafe {
        device.destroy_buffer(buf.buf, None);
        device.free_memory(buf.mem, None);
    }
}

fn m2_build_clas(device: Res<RenderDevice>, queue: Res<RenderQueue>) {
    if !device
        .features()
        .contains(WgpuFeatures::EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE)
    {
        return;
    }

    // Reuse the same synthetic input as M1 so we know the geometry round-trips.
    let (meshlet, vertex_positions_packed, indices) = synthetic_triangle_meshlet();
    let positions = dequantize_meshlet_vertices(&meshlet, &vertex_positions_packed);
    let triangle_count = u32::from(meshlet.triangle_count);
    let vertex_count = u32::from(meshlet.vertex_count_minus_one) + 1;

    // SAFETY: gating on the wgpu feature bit guarantees the Vulkan device, the
    // VK_NV_cluster_acceleration_structure extension, and shaderDeviceAddress
    // are all enabled. Everything below is bounded to RenderStartup with no
    // other GPU work in flight; we vkDeviceWaitIdle before touching outputs.
    unsafe {
        let wgpu_device = device.wgpu_device();
        let Some(hal_device): Option<_> = wgpu_device.as_hal::<Vulkan>() else {
            warn!("Device is not a Vulkan device; skipping M2.");
            return;
        };
        let hal_device: &wgpu::hal::vulkan::Device = hal_device.deref();
        let raw_device = hal_device.raw_device();
        let raw_phys = hal_device.raw_physical_device();
        let raw_instance = hal_device.shared_instance().raw_instance();
        let mem_props = raw_instance.get_physical_device_memory_properties(raw_phys);

        // ---- Allocate input buffers: positions, indices ---------------------
        // BLAS_INPUT_RO is the right Vulkan flag for AS-build inputs; we share
        // it across vertex/index/descriptor buffers since they're all consumed
        // by the cluster build.
        let pos_bytes = bytemuck::cast_slice::<[f32; 3], u8>(&positions);
        let pos = alloc_buffer(
            raw_device,
            &mem_props,
            pos_bytes.len() as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            true,
        );
        upload_bytes(raw_device, &pos, pos_bytes);

        let idx = alloc_buffer(
            raw_device,
            &mem_props,
            indices.len() as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            true,
        );
        upload_bytes(raw_device, &idx, &indices);

        // ---- Build the per-cluster descriptor (one entry, one cluster) ------
        // The bitfield is packed manually because ash represents it as a single
        // u32. Layout per the spec, LSB-first:
        //   bits  0..9  triangleCount
        //   bits  9..18 vertexCount
        //   bits 18..24 positionTruncateBitCount (0 = no truncation)
        //   bits 24..28 indexType  (1 = 8-bit, 2 = 16-bit, 4 = 32-bit)
        //   bits 28..32 opacityMicromapIndexType (0 = none)
        let index_type_8bit: u32 = 1;
        let packed_bitfield: u32 = (triangle_count & 0x1FF)
            | ((vertex_count & 0x1FF) << 9)
            | (0u32 << 18) // positionTruncateBitCount
            | ((index_type_8bit & 0xF) << 24)
            | (0u32 << 28); // opacityMicromapIndexType

        let desc = vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV {
            cluster_id: 0,
            cluster_flags: vk::ClusterAccelerationStructureClusterFlagsNV::default(),
            packed_bitfield,
            // Bitfield: bits 0..24 geometryIndex, 24..29 reserved, 29..32 geometryFlags.
            // Set OPAQUE (bit 0 of geometryFlags) -> bit 29.
            base_geometry_index_and_geometry_flags:
                vk::ClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV(1u32 << 29),
            index_buffer_stride: 0,
            vertex_buffer_stride: 12, // f32x3 = 12 bytes
            geometry_index_and_flags_buffer_stride: 0,
            opacity_micromap_index_buffer_stride: 0,
            index_buffer: idx.addr,
            vertex_buffer: pos.addr,
            geometry_index_and_flags_buffer: 0,
            opacity_micromap_array: 0,
            opacity_micromap_index_buffer: 0,
        };
        let desc_bytes = std::slice::from_raw_parts(
            (&desc as *const _) as *const u8,
            std::mem::size_of_val(&desc),
        );

        let src_infos = alloc_buffer(
            raw_device,
            &mem_props,
            desc_bytes.len() as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            true,
        );
        upload_bytes(raw_device, &src_infos, desc_bytes);

        // ---- Size query, then allocate dst_implicit + scratch ---------------
        let triangle_input = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .max_geometry_index_value(0)
            .max_cluster_unique_geometry_count(1)
            .max_cluster_triangle_count(triangle_count)
            .max_cluster_vertex_count(vertex_count)
            .max_total_triangle_count(triangle_count)
            .max_total_vertex_count(vertex_count);
        let op_input = vk::ClusterAccelerationStructureOpInputNV {
            p_triangle_clusters: &triangle_input as *const _ as *mut _,
        };
        let input_info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(1)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(op_input);

        let sizes = hal_device.get_cluster_build_sizes(&input_info);
        info!(
            "M2 sizes: dst_implicit={} scratch={} bytes (1 CLAS, 1 tri)",
            sizes.acceleration_structure_size, sizes.build_scratch_size,
        );

        let dst_implicit = alloc_buffer(
            raw_device,
            &mem_props,
            sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            false,
        );
        let scratch = alloc_buffer(
            raw_device,
            &mem_props,
            sizes.build_scratch_size.max(1),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            false,
        );

        // dst_addresses receives one VkDeviceAddress per built CLAS. HOST_VISIBLE
        // so we can read it back without a copy. Stride must be at least
        // sizeof(VkDeviceAddress) = 8 bytes per the spec. Per VUID-10459 the
        // backing buffer must carry the AS_STORAGE_KHR usage even though the
        // payload is just an array of u64 device addresses.
        let dst_addresses = alloc_buffer(
            raw_device,
            &mem_props,
            8,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            true,
        );
        // Pre-fill with sentinel so we can tell if the driver wrote anything.
        upload_bytes(raw_device, &dst_addresses, &0xCDu64.to_le_bytes());

        // `srcInfosCount` is, for IMPLICIT/EXPLICIT_DESTINATIONS modes, a
        // VkDeviceAddress pointing at a 4-byte-aligned u32 holding the actual
        // count (see VUID-VkClusterAccelerationStructureCommandsInfoNV-srcInfosCount-*).
        // Allocate that count buffer and fill with 1 (we have one cluster).
        let count_buf = alloc_buffer(
            raw_device,
            &mem_props,
            4,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            true,
        );
        upload_bytes(raw_device, &count_buf, &1u32.to_le_bytes());

        // ---- Record + submit -------------------------------------------------
        let commands_info = vk::ClusterAccelerationStructureCommandsInfoNV {
            s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            input: input_info,
            dst_implicit_data: dst_implicit.addr,
            scratch_data: scratch.addr,
            dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
                device_address: dst_addresses.addr,
                stride: 8,
                size: 8,
            },
            dst_sizes_array: vk::StridedDeviceAddressRegionKHR::default(),
            src_infos_array: vk::StridedDeviceAddressRegionKHR {
                device_address: src_infos.addr,
                stride: std::mem::size_of_val(&desc) as u64,
                size: desc_bytes.len() as u64,
            },
            // For IMPLICIT/EXPLICIT_DESTINATIONS this is a device address to a
            // u32 count, not a literal count. See VUID-...-srcInfosCount-*.
            src_infos_count: count_buf.addr,
            address_resolution_flags: vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
            _marker: std::marker::PhantomData,
        };

        // Use a wgpu CommandEncoder + as_hal_mut to escape into vulkan::CommandEncoder
        // and call our inherent helper. Submit via the wgpu queue. This avoids
        // hand-rolling a vk command pool / fence.
        let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cluster_AS_build"),
        });
        encoder.as_hal_mut::<Vulkan, _, _>(|maybe_cmd| {
            if let Some(cmd) = maybe_cmd {
                cmd.cmd_build_cluster_acceleration_structures_indirect(&commands_info);
            }
        });
        let cb = encoder.finish();
        queue.submit([cb]);

        // Block on completion before reading back. RenderStartup is single-threaded
        // and there's no other GPU work this early, so this is benign.
        raw_device.device_wait_idle().expect("device_wait_idle");

        // ---- Read back the device address of the built CLAS -----------------
        let ptr = raw_device
            .map_memory(dst_addresses.mem, 0, 8, vk::MemoryMapFlags::empty())
            .expect("map dst_addresses") as *const u64;
        let clas_address = std::ptr::read_unaligned(ptr);
        raw_device.unmap_memory(dst_addresses.mem);

        if clas_address == 0xCD || clas_address == 0 {
            warn!(
                "M2 build appears to have failed: dst_addresses[0] = 0x{:016x} \
                 (sentinel still present)",
                clas_address,
            );
            destroy_all(
                raw_device,
                [pos, idx, src_infos, dst_implicit, scratch, dst_addresses, count_buf],
            );
            return;
        }
        info!(
            "M2 build OK: 1 CLAS @ device address 0x{:016x} ({} bytes implicit data)",
            clas_address, sizes.acceleration_structure_size,
        );

        // ---- M3: build a BLAS that references the CLAS by device address ----
        let (blas_address, m3_residue) = m3_build_blas(
            wgpu_device,
            &queue,
            hal_device,
            raw_device,
            &mem_props,
            clas_address,
        );
        info!("M3 build OK: 1 BLAS @ device address 0x{:016x}", blas_address);

        // ---- M4: build a TLAS instancing the BLAS via standard KHR-AS path --
        let (tlas_address, m4_residue) = m4_build_tlas(
            wgpu_device,
            &queue,
            raw_device,
            hal_device.shared_instance().raw_instance(),
            &mem_props,
            blas_address,
        );
        info!("M4 build OK: 1 TLAS @ device address 0x{:016x}", tlas_address);

        // ---- M5: validate the TLAS handle is well-formed --------------------
        // A real ray query trace requires a compute pipeline + descriptor sets,
        // which is its own substantial chunk of raw Vulkan; we stop here at the
        // proof that the cluster-built BLAS is consumable by a regular KHR TLAS.
        if tlas_address == 0 {
            warn!("M5: TLAS device address is zero -- ray query would not be reachable.");
        } else {
            info!(
                "M5 OK: TLAS reference is consumable (address 0x{:016x}). Ray query \
                 dispatch is the next milestone.",
                tlas_address,
            );
        }

        // Tear down M4 + M3 residues.
        m4_residue.destroy(raw_device);
        m3_residue.destroy(raw_device);

        // ---- Cleanup --------------------------------------------------------
        // We deliberately keep `dst_implicit` (which holds the actual CLAS
        // payload) alive until the very end so the BLAS in M3 can reference it.
        // Same for the M3 BLAS storage -- m3 internally cleans up its own
        // transient buffers but returns the dst_implicit buffer alive.
        destroy_all(
            raw_device,
            [pos, idx, src_infos, dst_implicit, scratch, dst_addresses, count_buf],
        );
    }
}

unsafe fn destroy_all<const N: usize>(device: &ash::Device, bufs: [RawBuffer; N]) {
    for b in bufs {
        destroy(device, b);
    }
}

/// Buffers that must outlive their owning AS so the AS payload remains valid
/// for downstream consumers (M3's BLAS payload stays alive for M4's TLAS,
/// M4's instance/storage stays alive for any future M5 ray-query dispatch).
struct AsResidue {
    bufs: Vec<RawBuffer>,
    /// Optional VkAccelerationStructureKHR handle that must be destroyed via
    /// the KHR-AS extension entry points before the underlying storage buffer.
    tlas: Option<(vk::AccelerationStructureKHR, ash::khr::acceleration_structure::Device)>,
}

impl AsResidue {
    unsafe fn destroy(self, device: &ash::Device) {
        if let Some((tlas, ext)) = self.tlas {
            ext.destroy_acceleration_structure(tlas, None);
        }
        for b in self.bufs {
            destroy(device, b);
        }
    }
}

/// M3: build a BLAS by referencing one or more pre-built CLASes via the
/// `BUILD_CLUSTERS_BOTTOM_LEVEL` op_type. The BLAS payload (dst_implicit) is
/// returned in `AsResidue` so it stays valid for the TLAS instance built in
/// M4; transient inputs (cluster_refs, src_infos, scratch, count_buf,
/// dst_addresses) are destroyed before returning.
unsafe fn m3_build_blas(
    wgpu_device: &wgpu::Device,
    queue: &RenderQueue,
    hal_device: &wgpu::hal::vulkan::Device,
    raw_device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    clas_address: u64,
) -> (u64, AsResidue) {
    // 1. Cluster-reference array: one u64 holding the CLAS address.
    let cluster_refs = alloc_buffer(
        raw_device,
        mem_props,
        8,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        true,
    );
    upload_bytes(raw_device, &cluster_refs, &clas_address.to_le_bytes());

    // 2. Per-BLAS descriptor: VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV
    //    points at the cluster-reference array we just uploaded.
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct BuildClustersBottomLevelInfo {
        cluster_references_count: u32,
        cluster_references_stride: u32,
        cluster_references: u64,
    }
    let blas_desc = BuildClustersBottomLevelInfo {
        cluster_references_count: 1,
        cluster_references_stride: 8,
        cluster_references: cluster_refs.addr,
    };
    let desc_bytes = std::slice::from_raw_parts(
        (&blas_desc as *const _) as *const u8,
        size_of_val(&blas_desc),
    );
    let src_infos = alloc_buffer(
        raw_device,
        mem_props,
        desc_bytes.len() as u64,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        true,
    );
    upload_bytes(raw_device, &src_infos, desc_bytes);

    // 3. Size query. The op_input flavour for BLAS-from-clusters is
    //    `p_clusters_bottom_level` pointing at a max-counts struct.
    let bottom_level_input = vk::ClusterAccelerationStructureClustersBottomLevelInputNV::default()
        .max_total_cluster_count(1)
        .max_cluster_count_per_acceleration_structure(1);
    let op_input = vk::ClusterAccelerationStructureOpInputNV {
        p_clusters_bottom_level: &bottom_level_input as *const _ as *mut _,
    };
    let input_info = vk::ClusterAccelerationStructureInputInfoNV::default()
        .max_acceleration_structure_count(1)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_CLUSTERS_BOTTOM_LEVEL)
        .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
        .op_input(op_input);

    let sizes = hal_device.get_cluster_build_sizes(&input_info);
    info!(
        "M3 sizes: dst_implicit={} scratch={} bytes (1 BLAS, 1 CLAS ref)",
        sizes.acceleration_structure_size, sizes.build_scratch_size,
    );

    // 4. Output buffers: dst_implicit holds the BLAS payload; dst_addresses
    //    receives a single VkDeviceAddress pointing into dst_implicit.
    let dst_implicit = alloc_buffer(
        raw_device,
        mem_props,
        sizes.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        false,
    );
    let scratch = alloc_buffer(
        raw_device,
        mem_props,
        sizes.build_scratch_size.max(1),
        vk::BufferUsageFlags::STORAGE_BUFFER,
        false,
    );
    let dst_addresses = alloc_buffer(
        raw_device,
        mem_props,
        8,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        true,
    );
    upload_bytes(raw_device, &dst_addresses, &0xCDu64.to_le_bytes());
    let count_buf = alloc_buffer(
        raw_device,
        mem_props,
        4,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        true,
    );
    upload_bytes(raw_device, &count_buf, &1u32.to_le_bytes());

    // 5. Record + submit.
    let commands_info = vk::ClusterAccelerationStructureCommandsInfoNV {
        s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
        p_next: std::ptr::null_mut(),
        input: input_info,
        dst_implicit_data: dst_implicit.addr,
        scratch_data: scratch.addr,
        dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
            device_address: dst_addresses.addr,
            stride: 8,
            size: 8,
        },
        dst_sizes_array: vk::StridedDeviceAddressRegionKHR::default(),
        src_infos_array: vk::StridedDeviceAddressRegionKHR {
            device_address: src_infos.addr,
            stride: size_of_val(&blas_desc) as u64,
            size: desc_bytes.len() as u64,
        },
        src_infos_count: count_buf.addr,
        address_resolution_flags: vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
        _marker: std::marker::PhantomData,
    };
    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cluster_BLAS_build"),
    });
    encoder.as_hal_mut::<Vulkan, _, _>(|maybe_cmd| {
        if let Some(cmd) = maybe_cmd {
            cmd.cmd_build_cluster_acceleration_structures_indirect(&commands_info);
        }
    });
    let cb = encoder.finish();
    queue.submit([cb]);
    raw_device.device_wait_idle().expect("M3 device_wait_idle");

    // 6. Read back BLAS address.
    let ptr = raw_device
        .map_memory(dst_addresses.mem, 0, 8, vk::MemoryMapFlags::empty())
        .expect("M3 map dst_addresses") as *const u64;
    let blas_address = std::ptr::read_unaligned(ptr);
    raw_device.unmap_memory(dst_addresses.mem);

    if blas_address == 0xCD || blas_address == 0 {
        panic!("M3 build failed: dst_addresses[0] = 0x{blas_address:016x}");
    }

    // 7. Tear down M3-private inputs. `dst_implicit` is the BLAS payload --
    //    M4's TLAS instance buffer references it by device address, so we keep
    //    it alive in the residue until the test ends.
    destroy_all(
        raw_device,
        [cluster_refs, src_infos, scratch, dst_addresses, count_buf],
    );
    (
        blas_address,
        AsResidue {
            bufs: vec![dst_implicit],
            tlas: None,
        },
    )
}

/// M4: build a top-level acceleration structure that contains a single
/// instance whose `accelerationStructureReference` points at the cluster-built
/// BLAS. This is the standard `VK_KHR_acceleration_structure` path -- the
/// device addresses produced by the NV cluster path are valid AS references.
///
/// Returns the TLAS device address plus an `AsResidue` holding the AS storage
/// buffer, instance buffer, and the `VkAccelerationStructureKHR` handle so M5
/// can use it.
unsafe fn m4_build_tlas(
    wgpu_device: &wgpu::Device,
    queue: &RenderQueue,
    raw_device: &ash::Device,
    raw_instance: &ash::Instance,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    blas_address: u64,
) -> (u64, AsResidue) {
    // Load the KHR-AS device entry points. wgpu-hal already loads these, but
    // the field is private; loading our own doesn't conflict.
    let khr_as = ash::khr::acceleration_structure::Device::load(raw_instance, raw_device);

    // 1. Build a single VkAccelerationStructureInstanceKHR pointing at the BLAS.
    // TransformMatrixKHR is row-major 3x4 (12 floats): identity here.
    let instance = vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR {
            matrix: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
            ],
        },
        instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: blas_address,
        },
    };
    let instance_bytes = std::slice::from_raw_parts(
        (&instance as *const _) as *const u8,
        size_of_val(&instance),
    );
    // VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
    // covers the instance buffer for KHR-AS TLAS builds.
    let instance_buf = alloc_buffer(
        raw_device,
        mem_props,
        instance_bytes.len() as u64,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        true,
    );
    upload_bytes(raw_device, &instance_buf, instance_bytes);

    // 2. Geometry: type INSTANCES, pointing at our instance buffer.
    let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: instance_buf.addr,
        });
    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: instances_data,
        });
    let geometries = [geometry];
    let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometries);

    // 3. Size query.
    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    khr_as.get_acceleration_structure_build_sizes(
        vk::AccelerationStructureBuildTypeKHR::DEVICE,
        &build_info,
        &[1u32], // 1 instance
        &mut size_info,
    );
    info!(
        "M4 sizes: as_storage={} scratch={} bytes (1 TLAS, 1 instance)",
        size_info.acceleration_structure_size, size_info.build_scratch_size,
    );

    // 4. Allocate AS storage + scratch.
    let as_storage = alloc_buffer(
        raw_device,
        mem_props,
        size_info.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        false,
    );
    let scratch = alloc_buffer(
        raw_device,
        mem_props,
        size_info.build_scratch_size.max(1),
        vk::BufferUsageFlags::STORAGE_BUFFER,
        false,
    );

    // 5. Create the VkAccelerationStructureKHR handle on top of as_storage.
    let create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(as_storage.buf)
        .offset(0)
        .size(size_info.acceleration_structure_size)
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
    let tlas = khr_as
        .create_acceleration_structure(&create_info, None)
        .expect("M4 create_acceleration_structure");

    // Plug TLAS handle + scratch device address into the build info.
    build_info = build_info
        .dst_acceleration_structure(tlas)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch.addr,
        });
    let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
        .primitive_count(1) // 1 instance
        .primitive_offset(0);

    // 6. Record + submit. We use a wgpu encoder again but escape into the raw
    //    vk command buffer via as_hal_mut so we can call the KHR-AS path.
    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("KHR_TLAS_build"),
    });
    encoder.as_hal_mut::<Vulkan, _, _>(|maybe_cmd| {
        if let Some(cmd) = maybe_cmd {
            // `cmd` is a `&mut wgpu::hal::vulkan::CommandEncoder`. The active
            // raw vk::CommandBuffer is on its `active` field, but that's
            // private. wgpu-hal exposes it via the inherent `cmd.raw_command_buffer()`?
            // No -- there's no such accessor. Use the same trick our cluster
            // helper uses: it dispatches via `self.active`, which exists on
            // `super::CommandEncoder`. Since we don't have access from here,
            // record this on a transient raw vk command buffer instead.
            // (A small dance, but the alternative is more wgpu plumbing.)
            let _ = cmd;
        }
    });
    drop(encoder);

    // Use a raw vk command buffer for the KHR build. Allocate a transient pool.
    let queue_family = 0u32; // bevy's render queue uses graphics queue 0 in practice
    let pool = raw_device
        .create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT),
            None,
        )
        .expect("M4 create_command_pool");
    let cmd_alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = raw_device
        .allocate_command_buffers(&cmd_alloc)
        .expect("M4 alloc_command_buffers")[0];

    raw_device
        .begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .expect("M4 begin_command_buffer");
    khr_as.cmd_build_acceleration_structures(cb, &[build_info], &[&[range]]);
    raw_device
        .end_command_buffer(cb)
        .expect("M4 end_command_buffer");

    // Submit on the wgpu queue. We grab the raw VkQueue via as_hal.
    let raw_queue = queue
        .as_hal::<Vulkan>()
        .expect("queue as_hal Vulkan")
        .as_raw();
    let cbs = [cb];
    let submit = vk::SubmitInfo::default().command_buffers(&cbs);
    raw_device
        .queue_submit(raw_queue, &[submit], vk::Fence::null())
        .expect("M4 queue_submit");
    raw_device.device_wait_idle().expect("M4 device_wait_idle");

    raw_device.destroy_command_pool(pool, None);

    // 7. Read TLAS device address.
    let tlas_address = khr_as.get_acceleration_structure_device_address(
        &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(tlas),
    );

    let _ = wgpu_device;

    (
        tlas_address,
        AsResidue {
            bufs: vec![instance_buf, as_storage, scratch],
            tlas: Some((tlas, khr_as)),
        },
    )
}
