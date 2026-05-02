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
//! [`TlasManager`] builds a partitioned TLAS through
//! `VK_NV_partitioned_acceleration_structure` (paired with the cluster-built
//! BLASes Aurora produces in [`super::cluster_as`]). The standard KHR TLAS
//! path (`vkCmdBuildAccelerationStructuresKHR` with
//! `AccelerationStructureEntries::Instances`) cannot traverse cluster-built
//! BLAS internal references -- ray queries silently miss. The partitioned
//! extension produces a TLAS whose traversal *does* follow the cluster-AS
//! references; this is the load-bearing companion behind NVIDIA's RTX
//! MegaGeometry pipeline (validated decisively by the M-B sub-2c
//! triangle-BLAS control test, which hit perfectly through the same wrap +
//! bind + ray-query path Aurora uses for the bunny).
//!
//! After the build, the resulting hal acceleration structure is wrapped as
//! a [`wgpu::Tlas`] via `Device::create_tlas_from_hal` so it is bindable as
//! an `acceleration_structure` resource in compute pipelines without leaving
//! the safe wgpu surface.
//!
//! ## Scope (M-B sub-2)
//!
//! Single-build, single-partition (one global partition spanning all
//! instances). Rebuild + multi-partition support lands later.

use core::ops::Deref;

use ash::vk;
use bevy_ecs::resource::Resource;
use bevy_math::Mat4;
use wgpu::hal as wgh;
// Bring the hal trait methods (`tlas_instance_to_bytes`,
// `create_acceleration_structure`, `build_acceleration_structures`, ...) into
// scope. They are trait items, not inherent methods on the concrete vulkan
// types.
use wgpu::hal::CommandEncoder as _;
use wgpu::hal::Device as _;

use super::raw_vk::RawBuffer;

/// One instance Aurora wants in the TLAS this frame.
#[derive(Debug, Clone, Copy)]
pub struct TlasInstance {
    /// World-from-local transform. Vulkan's `TransformMatrixKHR` is row-major
    /// 3×4 (12 floats); we accept a `bevy_math::Mat4` and convert at upload.
    pub world_from_local: Mat4,
    /// Cluster-built BLAS device address (from
    /// [`crate::scene::cluster_as::MeshClusters::blas_address`]).
    pub blas_address: u64,
    /// Hit-shader instance index (24 bits).
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

/// Soft cap on the number of instances per build. Just a sanity guard for the
/// prototype — Aurora bails out before allocating an instance buffer larger
/// than this.
pub const DEFAULT_MAX_INSTANCES: u32 = 4096;

/// Builds + holds the wrapped [`wgpu::Tlas`].
///
/// Inserted into the render world as a [`Resource`] alongside
/// [`super::cluster_as::ClusterAsManager`]. Once [`Self::build`] succeeds,
/// [`Self::tlas`] returns a stable `wgpu::Tlas` reference suitable for
/// descriptor binding.
#[derive(Resource, Default)]
pub struct TlasManager {
    /// Wrapped wgpu TLAS. `None` until the first successful build.
    tlas: Option<wgpu::Tlas>,
    /// Reserved for incremental rebuild support (M-D): the partitioned-AS
    /// extension reads instance / indirect / count buffers at build time
    /// only, so M-B sub-2's single build can free them immediately. Once
    /// rebuilds land, this slot will hold the persistent
    /// `WriteInstanceDataNV[]` buffer that subsequent
    /// `WRITE_INSTANCE` / `UPDATE_INSTANCE` ops mutate.
    #[allow(dead_code, reason = "reserved for M-D incremental rebuilds")]
    instance_buf: Option<RawBuffer>,
    /// Number of instances in the most recent build (telemetry).
    last_built: u32,
}

impl TlasManager {
    /// Returns the wrapped TLAS, or `None` if no build has succeeded yet.
    pub fn tlas(&self) -> Option<&wgpu::Tlas> {
        self.tlas.as_ref()
    }

    /// `true` once a TLAS has been built.
    pub fn is_built(&self) -> bool {
        self.tlas.is_some()
    }

    /// Number of instances in the most recent build.
    pub fn last_built_instance_count(&self) -> u32 {
        self.last_built
    }

    /// Build the TLAS once for `instances`. Idempotent: subsequent calls with
    /// any instance set return immediately.
    ///
    /// # Safety
    ///
    /// - All `blas_address` values must point at AS payloads in still-resident
    ///   storage.
    /// - `device` must have been created with the Vulkan backend; this method
    ///   panics on any other backend.
    /// - No in-flight GPU work may race with the build's `vkDeviceWaitIdle`.
    pub unsafe fn build(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[TlasInstance],
    ) {
        if self.tlas.is_some() || instances.is_empty() {
            return;
        }
        assert!(
            instances.len() as u32 <= DEFAULT_MAX_INSTANCES,
            "aurora TLAS instance count {} exceeds DEFAULT_MAX_INSTANCES {}",
            instances.len(),
            DEFAULT_MAX_INSTANCES,
        );
        unsafe { self.build_inner(device, queue, instances) };
    }

    /// Free Aurora-owned resources backing this TLAS.
    ///
    /// The wrapped `wgpu::Tlas` cleans up its own AS handle + storage on
    /// drop; only the persistent instance buffer is on us.
    ///
    /// # Safety
    ///
    /// No in-flight GPU work may reference the TLAS or its instance buffer.
    pub unsafe fn destroy(self, device: &ash::Device) {
        if let Some(buf) = self.instance_buf {
            unsafe { buf.destroy(device) };
        }
        // `self.tlas` drops here; wgpu's destroy_acceleration_structure runs
        // through wgpu-hal cleanup paths.
    }

    unsafe fn build_inner(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[TlasInstance],
    ) {
        let max_instances = instances.len() as u32;
        let flags = vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE;

        // The hal AS outlives the guard scope; ownership moves into
        // `create_tlas_from_hal` after the guard drops.
        let hal_as = unsafe {
            let hal_device_guard = device
                .as_hal::<wgh::api::Vulkan>()
                .expect("aurora requires the Vulkan backend");
            let hal_device: &wgh::vulkan::Device = hal_device_guard.deref();
            let raw_device = hal_device.raw_device();
            let raw_instance = hal_device.shared_instance().raw_instance();
            let raw_phys = hal_device.raw_physical_device();
            let mem_props = raw_instance.get_physical_device_memory_properties(raw_phys);
            let khr_as = ash::khr::acceleration_structure::Device::load(raw_instance, raw_device);

            // ---- Build the WriteInstanceData[] payload --------------------
            // Each instance is described by a partitioned-AS-specific struct,
            // *not* the standard `vk::AccelerationStructureInstanceKHR`. The
            // BLAS reference is still a raw `VkDeviceAddress`, so cluster-built
            // BLAS addresses slot in directly.
            let mut writes = Vec::with_capacity(instances.len());
            for (idx, inst) in instances.iter().enumerate() {
                let m = inst.world_from_local.to_cols_array();
                writes.push(vk::PartitionedAccelerationStructureWriteInstanceDataNV {
                    transform: vk::TransformMatrixKHR {
                        matrix: [
                            m[0], m[4], m[8], m[12],
                            m[1], m[5], m[9], m[13],
                            m[2], m[6], m[10], m[14],
                        ],
                    },
                    // `[0; 6]` lets the build derive each instance's AABB
                    // from its referenced BLAS. Aurora doesn't have explicit
                    // bounds today.
                    explicit_aabb: [0.0; 6],
                    instance_id: inst.instance_custom_index & 0x00FF_FFFF,
                    instance_mask: u32::from(inst.mask),
                    instance_contribution_to_hit_group_index: 0,
                    instance_flags:
                        vk::PartitionedAccelerationStructureInstanceFlagsNV::default(),
                    instance_index: idx as u32,
                    // Single global partition for now -- multi-partition
                    // build-time support comes once Aurora has a meaningful
                    // partitioning policy (tile / cluster / lod).
                    partition_index: 0,
                    acceleration_structure: inst.blas_address,
                });
            }
            let writes_bytes = core::slice::from_raw_parts(
                writes.as_ptr().cast::<u8>(),
                core::mem::size_of_val(writes.as_slice()),
            );
            let instance_buf = RawBuffer::alloc(
                raw_device,
                &mem_props,
                writes_bytes.len() as u64,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                true,
            );
            instance_buf.upload(raw_device, writes_bytes);

            // ---- Indirect command -----------------------------------------
            // One operation: WRITE_INSTANCE covering all instances. arg_data
            // points at the WriteInstanceDataNV[] array we just uploaded.
            let stride =
                core::mem::size_of::<vk::PartitionedAccelerationStructureWriteInstanceDataNV>()
                    as u64;
            let indirect_cmd = vk::BuildPartitionedAccelerationStructureIndirectCommandNV {
                op_type: vk::PartitionedAccelerationStructureOpTypeNV::WRITE_INSTANCE,
                arg_count: max_instances,
                arg_data: vk::StridedDeviceAddressNV {
                    start_address: instance_buf.addr,
                    stride_in_bytes: stride,
                },
            };
            let indirect_buf = RawBuffer::alloc(
                raw_device,
                &mem_props,
                core::mem::size_of_val(&indirect_cmd) as u64,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                true,
            );
            indirect_buf.upload(
                raw_device,
                core::slice::from_raw_parts(
                    core::ptr::from_ref(&indirect_cmd).cast::<u8>(),
                    core::mem::size_of_val(&indirect_cmd),
                ),
            );

            // `srcInfosCount` is a device address pointing at a u32 holding
            // the actual indirect-command count. We have one command.
            let count_buf = RawBuffer::alloc(
                raw_device,
                &mem_props,
                4,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                true,
            );
            count_buf.upload(raw_device, &1u32.to_le_bytes());

            // ---- Input descriptor -----------------------------------------
            let input_info =
                vk::PartitionedAccelerationStructureInstancesInputNV::default()
                    .flags(flags)
                    .instance_count(max_instances)
                    .max_instance_per_partition_count(max_instances)
                    .partition_count(1)
                    .max_instance_in_global_partition_count(max_instances);

            // ---- Size query + AS storage allocation -----------------------
            let size_info = hal_device.get_partitioned_build_sizes(&input_info);
            let scratch = RawBuffer::alloc(
                raw_device,
                &mem_props,
                size_info.build_scratch_size.max(1),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                false,
            );
            // Allocate the partitioned TLAS storage via wgpu-hal so wgpu
            // owns the lifecycle. The handle's device address is what the
            // build writes into.
            let hal_as = hal_device
                .create_acceleration_structure(&wgh::AccelerationStructureDescriptor {
                    label: Some("aurora.partitioned_tlas"),
                    size: size_info.acceleration_structure_size,
                    format: wgh::AccelerationStructureFormat::TopLevel,
                    allow_compaction: false,
                })
                .expect("create_acceleration_structure (partitioned TLAS storage)");
            // The partitioned build's `dstAccelerationStructureData` is a
            // *buffer* device address, not an AS handle's device address.
            // (Vulkan validation layer is explicit about this: the dst
            // address must come from a `vkGetBufferDeviceAddress` call.)
            // Aurora-allocated KHR ASes always live at offset 0 in their
            // backing buffer (see `wgpu-hal/src/vulkan/device.rs`), so the
            // buffer's device address points at the AS storage start.
            let buffer_address_fns =
                ash::khr::buffer_device_address::Device::load(raw_instance, raw_device);
            let dst_addr = buffer_address_fns.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(hal_as.raw_buffer()),
            );
            let _ = khr_as;

            // ---- Encode + submit ------------------------------------------
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora.partitioned_tlas.build"),
            });
            encoder.as_hal_mut::<wgh::api::Vulkan, _, _>(|maybe_cmd| {
                if let Some(cmd) = maybe_cmd {
                    // Fresh build (no prior partitioned AS to update). The
                    // spec says srcAccelerationStructureData=0 is valid for
                    // this case; the validation layer flags zero as a
                    // VUID-parameter error but the build still proceeds and
                    // the contents of `srcAccelerationStructureData` are
                    // unread for `WRITE_INSTANCE` ops. Passing a non-zero
                    // overlapping address triggers a different VUID
                    // (src/dst must not overlap), so 0 is the right call
                    // here -- validation noise is preferable to a wrong fix.
                    let build_info = vk::BuildPartitionedAccelerationStructureInfoNV::default()
                        .input(input_info)
                        .src_acceleration_structure_data(0)
                        .dst_acceleration_structure_data(dst_addr)
                        .scratch_data(scratch.addr)
                        .src_infos(indirect_buf.addr)
                        .src_infos_count(count_buf.addr);
                    cmd.cmd_build_partitioned_acceleration_structures(&build_info);
                }
            });
            let cb = encoder.finish();
            queue.submit([cb]);
            raw_device
                .device_wait_idle()
                .expect("partitioned TLAS device_wait_idle");

            // ---- Free transient + persistent buffers ----------------------
            // The partitioned-AS spec, like the standard KHR TLAS path, reads
            // instance + indirect + count buffers at build time only -- after
            // the AS is built they are no longer referenced. M-B sub-2 builds
            // once and never updates, so all three transient buffers can drop
            // here. Persistent rebuild support (M-D) will keep them around.
            scratch.destroy(raw_device);
            instance_buf.destroy(raw_device);
            indirect_buf.destroy(raw_device);
            count_buf.destroy(raw_device);

            hal_as
        };

        // Hand the hal AS to wgpu-core. From this point on, lifecycle of the
        // underlying VkAccelerationStructureKHR is wgpu-managed.
        let tlas = unsafe {
            device.create_tlas_from_hal::<wgh::api::Vulkan>(
                hal_as,
                &wgpu::CreateTlasDescriptor {
                    label: Some("aurora.partitioned_tlas"),
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    max_instances,
                },
            )
        };

        self.tlas = Some(tlas);
        self.last_built = max_instances;
        tracing::info!(
            target: "bevy_aurora",
            instances = max_instances,
            "Partitioned TLAS built (cluster_AS BLAS instancing via VK_NV_partitioned_acceleration_structure)",
        );
    }
}
