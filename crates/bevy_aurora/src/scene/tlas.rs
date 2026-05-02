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
//! [`TlasManager`] builds a TLAS through `wgpu-hal` directly (because Aurora's
//! BLASes are cluster-built and addressed by raw `VkDeviceAddress`, which sits
//! below wgpu-core's `wgpu::Blas` type) and then wraps the resulting hal
//! acceleration structure as a [`wgpu::Tlas`] via the
//! `Device::create_tlas_from_hal` API. The wrapped `Tlas` is bindable as an
//! `acceleration_structure` resource in compute pipelines without leaving the
//! safe wgpu surface.
//!
//! ## Scope (M-B sub-2)
//!
//! Single-build only — the bunny scene is static. Rebuild support lands when
//! M-D introduces dynamic / hybrid scenes.

use core::iter::once;
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
    /// Persistent instance buffer the TLAS references via device address.
    /// Must outlive the TLAS — the AS keeps a pointer to its contents.
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
        let flags = wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE;

        // The hal acceleration structure outlives the guard scope; ownership
        // is moved into `create_tlas_from_hal` after the guard drops.
        let hal_as = unsafe {
            let hal_device_guard = device
                .as_hal::<wgh::api::Vulkan>()
                .expect("aurora requires the Vulkan backend");
            let hal_device: &wgh::vulkan::Device = hal_device_guard.deref();
            let raw_device = hal_device.raw_device();
            let raw_instance = hal_device.shared_instance().raw_instance();
            let raw_phys = hal_device.raw_physical_device();
            let mem_props = raw_instance.get_physical_device_memory_properties(raw_phys);

            // Pre-encode the instance bytes via the hal helper.
            let mut bytes = Vec::with_capacity(instances.len() * 64);
            for inst in instances {
                let m = inst.world_from_local.to_cols_array();
                bytes.extend_from_slice(&hal_device.tlas_instance_to_bytes(wgh::TlasInstance {
                    transform: [
                        m[0], m[4], m[8], m[12],
                        m[1], m[5], m[9], m[13],
                        m[2], m[6], m[10], m[14],
                    ],
                    custom_data: inst.instance_custom_index & 0x00FF_FFFF,
                    mask: inst.mask,
                    blas_address: inst.blas_address,
                }));
            }

            // Persistent instance buffer (AS reads it via device address).
            let instance_buf = RawBuffer::alloc(
                raw_device,
                &mem_props,
                bytes.len() as u64,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                true,
            );
            instance_buf.upload(raw_device, &bytes);
            let instance_buf_hal = wgh::vulkan::Buffer::from_raw(instance_buf.buf);

            let entries = wgh::AccelerationStructureEntries::<wgh::vulkan::Buffer>::Instances(
                wgh::AccelerationStructureInstances {
                    buffer: Some(&instance_buf_hal),
                    offset: 0,
                    count: max_instances,
                },
            );

            // Size query + scratch + AS allocation.
            let size_info = hal_device.get_acceleration_structure_build_sizes(
                &wgh::GetAccelerationStructureBuildSizesDescriptor {
                    entries: &entries,
                    flags,
                },
            );
            let scratch = RawBuffer::alloc(
                raw_device,
                &mem_props,
                size_info.build_scratch_size.max(1),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                false,
            );
            let scratch_hal = wgh::vulkan::Buffer::from_raw(scratch.buf);

            let hal_as = hal_device
                .create_acceleration_structure(&wgh::AccelerationStructureDescriptor {
                    label: Some("aurora.tlas"),
                    size: size_info.acceleration_structure_size,
                    format: wgh::AccelerationStructureFormat::TopLevel,
                    allow_compaction: false,
                })
                .expect("create_acceleration_structure (TLAS)");

            // Encode the build through wgpu's encoder so wgpu-core sees the
            // command stream; the cluster_AS hal extension also rides this
            // pattern (`encoder.as_hal_mut`).
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora.tlas.build"),
            });
            encoder.as_hal_mut::<wgh::api::Vulkan, _, _>(|maybe_cmd| {
                if let Some(cmd) = maybe_cmd {
                    let desc = wgh::BuildAccelerationStructureDescriptor::<
                        wgh::vulkan::Buffer,
                        wgh::vulkan::AccelerationStructure,
                    > {
                        entries: &entries,
                        mode: wgh::AccelerationStructureBuildMode::Build,
                        flags,
                        source_acceleration_structure: None,
                        destination_acceleration_structure: &hal_as,
                        scratch_buffer: &scratch_hal,
                        scratch_buffer_offset: 0,
                    };
                    cmd.build_acceleration_structures(1, once(desc));
                }
            });
            let cb = encoder.finish();
            queue.submit([cb]);
            raw_device
                .device_wait_idle()
                .expect("TLAS device_wait_idle");

            // Scratch is single-use; instance buffer must outlive the wrap.
            scratch.destroy(raw_device);
            self.instance_buf = Some(instance_buf);

            // The hal Buffer wrappers (`from_raw`) are pure handles with no
            // owned memory; dropping them at end-of-scope is a no-op.
            let _ = (instance_buf_hal, scratch_hal);
            hal_as
        };

        // Hand the hal AS to wgpu-core. From this point on, lifecycle of the
        // underlying VkAccelerationStructureKHR is wgpu-managed.
        let tlas = unsafe {
            device.create_tlas_from_hal::<wgh::api::Vulkan>(
                hal_as,
                &wgpu::CreateTlasDescriptor {
                    label: Some("aurora.tlas"),
                    flags,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    max_instances,
                },
            )
        };

        self.tlas = Some(tlas);
        self.last_built = max_instances;
        tracing::debug!(
            target: "bevy_aurora",
            instances = max_instances,
            "TLAS built (wgpu-hal -> create_tlas_from_hal)",
        );
    }
}
