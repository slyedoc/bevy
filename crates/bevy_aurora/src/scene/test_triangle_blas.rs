#![allow(
    clippy::undocumented_unsafe_blocks,
    reason = "Each unsafe fn documents its own contract; per-block SAFETY comments would just restate it."
)]
#![allow(
    clippy::doc_markdown,
    reason = "Vulkan / Cargo identifiers (vk::*, BLAS, etc.) need not be backticked."
)]
//! Diagnostic: build a regular single-triangle BLAS via the standard
//! `VK_KHR_acceleration_structure` path (no cluster_AS) and return its device
//! address.
//!
//! This is a control test for M-B sub-2c. The cluster-built BLAS path
//! produces a device address that goes straight into TLAS instances, but no
//! rays hit anything when we trace against the wrapped TLAS. By plugging
//! *this* triangle's address into the same TLAS instance instead, we isolate
//! whether the failure is in the cluster_AS interop (rays hit the triangle
//! → cluster path is broken) or our TLAS build / wrap path (rays still miss
//! → our TLAS code is broken regardless of the BLAS source).
//!
//! The triangle spans the bunny's bounding box at scale 0.2 (roughly
//! `[-1, 1] × [0, 2]`) so any reasonable camera framing the bunny will hit
//! it.

use ash::vk::{self, TaggedStructure as _};

use super::raw_vk::RawBuffer;

/// Built triangle BLAS that survives for the lifetime of the caller.
pub struct TestTriangleBlas {
    /// Storage buffer holding the AS payload.
    storage: RawBuffer,
    /// Vertex buffer (kept resident for the lifetime of the BLAS).
    #[allow(dead_code, reason = "kept resident; not read after build")]
    vertices: RawBuffer,
    /// `VkAccelerationStructureKHR` handle. Aurora's TLAS only reads its
    /// device address (next field), but the handle has to outlive the TLAS
    /// because the TLAS holds a reference to the AS storage by address.
    handle: vk::AccelerationStructureKHR,
    /// Device address obtained via `vkGetAccelerationStructureDeviceAddressKHR`.
    /// This is what goes into a TLAS instance's
    /// `accelerationStructureReference` for a regular KHR-built BLAS.
    pub address: u64,
}

impl TestTriangleBlas {
    /// Free the AS handle + backing buffers.
    ///
    /// # Safety
    ///
    /// No in-flight GPU work may reference this BLAS.
    pub unsafe fn destroy(
        self,
        device: &ash::Device,
        khr_as: &ash::khr::acceleration_structure::Device,
    ) {
        unsafe {
            khr_as.destroy_acceleration_structure(self.handle, None);
            self.storage.destroy(device);
            self.vertices.destroy(device);
        }
    }
}

/// Build a single-triangle BLAS via standard `vkCmdBuildAccelerationStructuresKHR`.
///
/// The triangle has corners at `(-1, 0, 0)`, `(1, 0, 0)`, `(0, 2, 0)` -- a
/// large vertical-ish triangle straddling the bunny's framing.
///
/// # Safety
///
/// - `device`/`mem_props` must come from the same physical device.
/// - The returned BLAS must be destroyed via [`TestTriangleBlas::destroy`]
///   before the device is destroyed.
/// - No in-flight GPU work may race the build's `vkDeviceWaitIdle`.
pub unsafe fn build(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    raw_instance: &ash::Instance,
    wgpu_device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> TestTriangleBlas {
    let khr_as = ash::khr::acceleration_structure::Device::load(raw_instance, device);

    // ---- Vertex buffer (3 × vec3) ------------------------------------------
    let verts: [f32; 9] = [
        -1.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, //
        0.0, 2.0, 0.0, //
    ];
    let vertex_bytes: &[u8] = bytemuck::cast_slice(&verts);
    let vertices = unsafe {
        RawBuffer::alloc(
            device,
            mem_props,
            vertex_bytes.len() as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            true,
        )
    };
    unsafe { vertices.upload(device, vertex_bytes) };

    // ---- Geometry descriptor ----------------------------------------------
    let triangles_data = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: vertices.addr,
        })
        .vertex_stride(12)
        .max_vertex(2)
        .index_type(vk::IndexType::NONE_KHR);
    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: triangles_data,
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);
    let geometries = [geometry];

    let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometries);

    // ---- Size query -------------------------------------------------------
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        khr_as.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[1],
            &mut sizes,
        );
    }

    // ---- Storage + scratch ------------------------------------------------
    let storage = unsafe {
        RawBuffer::alloc(
            device,
            mem_props,
            sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            false,
        )
    };
    let scratch = unsafe {
        RawBuffer::alloc(
            device,
            mem_props,
            sizes.build_scratch_size.max(1),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            false,
        )
    };

    // ---- Create AS handle -------------------------------------------------
    let create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(storage.buf)
        .offset(0)
        .size(sizes.acceleration_structure_size)
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
    let handle = unsafe {
        khr_as
            .create_acceleration_structure(&create_info, None)
            .expect("create_acceleration_structure (test BLAS)")
    };
    build_info.dst_acceleration_structure = handle;
    build_info.scratch_data = vk::DeviceOrHostAddressKHR {
        device_address: scratch.addr,
    };

    // ---- Record build through wgpu encoder + as_hal_mut -------------------
    let range = vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1);
    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("aurora.test_triangle_blas.build"),
    });
    unsafe {
        encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|maybe_cmd| {
            if let Some(cmd) = maybe_cmd {
                let cb = cmd.raw_handle();
                khr_as.cmd_build_acceleration_structures(cb, &[build_info], &[&[range]]);
            }
        });
    }
    let cb = encoder.finish();
    queue.submit([cb]);
    unsafe { device.device_wait_idle().expect("test BLAS device_wait_idle") };

    // ---- Free scratch -----------------------------------------------------
    unsafe { scratch.destroy(device) };

    // ---- Read AS device address ------------------------------------------
    let address = unsafe {
        khr_as.get_acceleration_structure_device_address(
            &vk::AccelerationStructureDeviceAddressInfoKHR {
                s_type: vk::AccelerationStructureDeviceAddressInfoKHR::STRUCTURE_TYPE,
                p_next: core::ptr::null(),
                acceleration_structure: handle,
                _marker: core::marker::PhantomData,
            },
        )
    };
    assert_ne!(address, 0, "test BLAS device address came back zero");

    tracing::info!(
        target: "bevy_aurora",
        addr = format!("0x{address:016x}"),
        storage_bytes = sizes.acceleration_structure_size,
        scratch_bytes = sizes.build_scratch_size,
        "test triangle BLAS built (KHR-AS, no cluster_AS)"
    );

    TestTriangleBlas {
        storage,
        vertices,
        handle,
        address,
    }
}
