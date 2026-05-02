#![allow(
    clippy::undocumented_unsafe_blocks,
    reason = "Each unsafe fn documents its own contract; per-block SAFETY comments would just restate it."
)]
#![allow(
    clippy::doc_markdown,
    reason = "Vulkan / Cargo identifiers (vk::*, cluster_AS, etc.) need not be backticked."
)]
//! Top-level acceleration structure manager.
//!
//! [`TlasManager`] owns one shared [`vk::AccelerationStructureKHR`] containing
//! one instance per uploaded `(transform, BLAS device address)` pair. Aurora
//! rebuilds the TLAS in-place each frame the instance set changes — the
//! underlying AS handle survives so any descriptor-set reference to it stays
//! valid; only the contents update.
//!
//! Uses the standard `VK_KHR_acceleration_structure` build path
//! (`vkCmdBuildAccelerationStructuresKHR`), not the cluster_AS path. The KHR
//! ecosystem accepts a cluster-built BLAS device address as a valid
//! `accelerationStructureReference` — that's the load-bearing claim of
//! NVIDIA's MegaGeometry design.

use ash::vk;
use bevy_ecs::resource::Resource;
use bevy_math::Mat4;

use super::raw_vk::{PersistentBuffer, RangeAllocator, RawBuffer};

/// One instance Aurora wants in the TLAS this frame.
#[derive(Debug, Clone, Copy)]
pub struct TlasInstance {
    /// World-from-local transform. Vulkan stores 3×4 row-major; we accept a
    /// `bevy_math::Mat4` and drop the bottom row at upload time.
    pub world_from_local: Mat4,
    /// Cluster-built BLAS device address (from
    /// [`crate::scene::cluster_as::MeshClusters::blas_address`]).
    pub blas_address: u64,
    /// Hit-shader instance index (24 bits). Aurora doesn't use shader
    /// binding tables yet; pass 0.
    pub instance_custom_index: u32,
    /// Visibility mask (8 bits). 0xFF = visible to all rays.
    pub mask: u8,
}

impl TlasInstance {
    pub fn opaque(world_from_local: Mat4, blas_address: u64) -> Self {
        Self {
            world_from_local,
            blas_address,
            instance_custom_index: 0,
            mask: 0xFF,
        }
    }
}

/// Default starting capacity for the TLAS storage / scratch / instance
/// buffers. Aurora reallocates only when the actual instance count exceeds
/// these caps; for prototype scenes the defaults fit comfortably.
pub const DEFAULT_MAX_INSTANCES: u32 = 4096;

/// Owns the persistent TLAS handle + backing buffer. Inserted into the render
/// world as a [`Resource`] alongside [`super::cluster_as::ClusterAsManager`].
#[derive(Resource)]
pub struct TlasManager {
    /// Storage backing the [`Self::handle`] AS. Sized for `max_instances`.
    storage: Option<PersistentBuffer>,
    /// Scratch buffer reused across frame builds (sized for max_instances).
    scratch: Option<RawBuffer>,
    /// Persistent instance buffer (host-visible, AS_BUILD_INPUT_RO).
    /// Aurora rewrites its contents each frame the instance list changes.
    instance_buf: Option<RawBuffer>,
    /// The actual TLAS handle. Stable across rebuilds — only the payload
    /// changes — so descriptor sets bound to it stay valid.
    handle: vk::AccelerationStructureKHR,
    /// Cached KHR-AS device-level fn pointer table.
    khr_as: Option<ash::khr::acceleration_structure::Device>,
    /// Total instance capacity. Realloc trigger.
    max_instances: u32,
    /// Number of valid instances in the most recent build (for telemetry).
    last_built: u32,
    /// Free-list over the storage buffer. Single-allocation in M-A; the
    /// allocator is here for symmetry with the CLAS / BLAS managers and so
    /// reallocation has a place to grow into.
    storage_free_ranges: RangeAllocator,
}

impl Default for TlasManager {
    fn default() -> Self {
        Self {
            storage: None,
            scratch: None,
            instance_buf: None,
            handle: vk::AccelerationStructureKHR::null(),
            khr_as: None,
            max_instances: 0,
            last_built: 0,
            storage_free_ranges: RangeAllocator::new(0),
        }
    }
}

impl TlasManager {
    /// The TLAS device address Aurora's lighting / primary-visibility shaders
    /// trace against. Returns 0 if no build has happened yet.
    pub fn device_address(&self) -> u64 {
        if self.handle == vk::AccelerationStructureKHR::null() {
            return 0;
        }
        let khr = self
            .khr_as
            .as_ref()
            .expect("TlasManager::device_address called before build");
        unsafe {
            khr.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(self.handle),
            )
        }
    }

    /// Raw VK handle for descriptor-set binding.
    pub fn handle(&self) -> vk::AccelerationStructureKHR {
        self.handle
    }

    pub fn last_built_instance_count(&self) -> u32 {
        self.last_built
    }

    /// Rebuild the TLAS for `instances`. The first call lazily allocates the
    /// storage / scratch / instance buffers; subsequent calls reuse them.
    /// If `instances.len()` exceeds the current capacity we reallocate and
    /// double the cap. The TLAS handle survives -- callers' descriptor sets
    /// stay valid.
    ///
    /// # Safety
    ///
    /// - All `blas_address` values must point at AS payloads in still-resident
    ///   storage.
    /// - No in-flight GPU work may reference the TLAS handle when this is
    ///   called -- we synchronously `vkDeviceWaitIdle` before mutating it.
    pub unsafe fn rebuild(
        &mut self,
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        raw_instance: &ash::Instance,
        wgpu_device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[TlasInstance],
    ) {
        // Lazy KHR-AS load.
        if self.khr_as.is_none() {
            self.khr_as = Some(ash::khr::acceleration_structure::Device::load(
                raw_instance,
                device,
            ));
        }
        // Grow / first-time alloc. Realloc may take `&mut self`, so clone the
        // KHR-AS fns first (cheap -- ash extension Device structs are just
        // function pointer tables wrapping a u64 device handle).
        let khr = self.khr_as.clone().expect("just loaded");
        let needed = instances.len().max(1) as u32;
        if needed > self.max_instances {
            unsafe { self.realloc_for(device, mem_props, &khr, needed.max(DEFAULT_MAX_INSTANCES)) };
        }

        // ---- Upload instance payload ----------------------------------------
        let instance_buf = self
            .instance_buf
            .as_ref()
            .expect("instance_buf allocated by realloc_for");
        if !instances.is_empty() {
            let mut instance_bytes =
                Vec::with_capacity(instances.len() * size_of::<vk::AccelerationStructureInstanceKHR>());
            for inst in instances {
                let m = inst.world_from_local.to_cols_array();
                // Vulkan TransformMatrixKHR is row-major 3×4 (12 floats).
                // bevy Mat4::to_cols_array is column-major 4×4 (16). Convert.
                let transform = vk::TransformMatrixKHR {
                    matrix: [
                        m[0], m[4], m[8], m[12], // row 0
                        m[1], m[5], m[9], m[13], // row 1
                        m[2], m[6], m[10], m[14], // row 2
                    ],
                };
                let raw = vk::AccelerationStructureInstanceKHR {
                    transform,
                    instance_custom_index_and_mask: vk::Packed24_8::new(
                        inst.instance_custom_index & 0x00FF_FFFF,
                        inst.mask,
                    ),
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                        0, 0,
                    ),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: inst.blas_address,
                    },
                };
                instance_bytes.extend_from_slice(unsafe {
                    core::slice::from_raw_parts(
                        core::ptr::from_ref(&raw).cast::<u8>(),
                        size_of::<vk::AccelerationStructureInstanceKHR>(),
                    )
                });
            }
            unsafe { instance_buf.upload(device, &instance_bytes) };
        }

        // ---- Build geometry ---------------------------------------------------
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

        let scratch = self.scratch.as_ref().expect("scratch allocated by realloc_for");
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries)
            .dst_acceleration_structure(self.handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch.addr,
            });
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instances.len() as u32);

        // ---- Submit via raw vk command pool (wgpu's encoder doesn't expose
        // the active VkCommandBuffer for raw KHR-AS calls) ---------------------
        let pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(0)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                )
                .expect("TLAS create_command_pool")
        };
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cb = unsafe {
            device
                .allocate_command_buffers(&cmd_alloc)
                .expect("TLAS alloc_command_buffers")[0]
        };
        unsafe {
            device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("TLAS begin_command_buffer");
            khr.cmd_build_acceleration_structures(cb, &[build_info], &[&[range]]);
            device.end_command_buffer(cb).expect("TLAS end_command_buffer");
        }

        let raw_queue = unsafe {
            queue
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("TLAS queue as_hal")
                .as_raw()
        };
        let cbs = [cb];
        let submit = vk::SubmitInfo::default().command_buffers(&cbs);
        unsafe {
            device
                .queue_submit(raw_queue, &[submit], vk::Fence::null())
                .expect("TLAS queue_submit");
            device.device_wait_idle().expect("TLAS device_wait_idle");
            device.destroy_command_pool(pool, None);
        }

        let _ = wgpu_device; // currently unused; reserved if we move to wgpu encoder later
        self.last_built = instances.len() as u32;
        tracing::debug!(
            target: "bevy_aurora",
            instances = instances.len(),
            cap = self.max_instances,
            "TLAS built",
        );
    }

    /// (Re)allocate storage / scratch / instance buffers + recreate the TLAS
    /// handle to fit `new_max_instances`. Destroys the previous handle.
    ///
    /// # Safety
    ///
    /// No in-flight GPU work may reference the existing TLAS handle or its
    /// backing buffers. Inherits all of [`Self::rebuild`]'s safety contract.
    unsafe fn realloc_for(
        &mut self,
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        khr: &ash::khr::acceleration_structure::Device,
        new_max_instances: u32,
    ) {
        // Tear down previous (no-op on first call).
        unsafe {
            if self.handle != vk::AccelerationStructureKHR::null() {
                khr.destroy_acceleration_structure(self.handle, None);
                self.handle = vk::AccelerationStructureKHR::null();
            }
            if let Some(s) = self.storage.take() {
                s.destroy(device);
            }
            if let Some(s) = self.scratch.take() {
                s.destroy(device);
            }
            if let Some(s) = self.instance_buf.take() {
                s.destroy(device);
            }
        }

        // Size query.
        let dummy_data = vk::AccelerationStructureGeometryInstancesDataKHR::default();
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: dummy_data,
            });
        let geometries = [geometry];
        let probe = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);
        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            khr.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &probe,
                &[new_max_instances],
                &mut size_info,
            );
        }

        // Storage buffer + AS handle.
        let storage = unsafe {
            PersistentBuffer::alloc(
                "aurora.tlas_storage",
                device,
                mem_props,
                size_info.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
                false,
            )
        };
        let create = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(storage.raw())
            .offset(0)
            .size(size_info.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
        let handle = unsafe {
            khr.create_acceleration_structure(&create, None)
                .expect("create_acceleration_structure (TLAS)")
        };

        let scratch = unsafe {
            RawBuffer::alloc(
                device,
                mem_props,
                size_info.build_scratch_size.max(1),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                false,
            )
        };

        let instance_buf = unsafe {
            RawBuffer::alloc(
                device,
                mem_props,
                u64::from(new_max_instances)
                    * size_of::<vk::AccelerationStructureInstanceKHR>() as u64,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                true,
            )
        };

        self.storage_free_ranges = RangeAllocator::new(size_info.acceleration_structure_size);
        self.storage = Some(storage);
        self.scratch = Some(scratch);
        self.instance_buf = Some(instance_buf);
        self.handle = handle;
        self.max_instances = new_max_instances;

        tracing::debug!(
            target: "bevy_aurora",
            cap = new_max_instances,
            storage_bytes = size_info.acceleration_structure_size,
            scratch_bytes = size_info.build_scratch_size,
            "TLAS reallocated",
        );
    }

    /// Free all GPU memory + destroy the TLAS handle. Must be called before
    /// `device` is destroyed.
    ///
    /// # Safety
    ///
    /// No in-flight GPU work may reference the TLAS handle.
    pub unsafe fn destroy(self, device: &ash::Device) {
        unsafe {
            if let Some(khr) = &self.khr_as
                && self.handle != vk::AccelerationStructureKHR::null() {
                    khr.destroy_acceleration_structure(self.handle, None);
                }
            if let Some(s) = self.storage {
                s.destroy(device);
            }
            if let Some(s) = self.scratch {
                s.destroy(device);
            }
            if let Some(s) = self.instance_buf {
                s.destroy(device);
            }
        }
    }
}
