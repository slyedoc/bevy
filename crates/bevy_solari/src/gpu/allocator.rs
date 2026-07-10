// This module reaches into raw Vulkan via `as_hal` because the
// cluster-AS pipeline needs buffers with usage flags wgpu does not
// expose publicly (`ACCELERATION_STRUCTURE_STORAGE_BIT_KHR`,
// `ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR`). The
// resulting buffers are still owned by wgpu — `create_buffer_from_hal`
// wraps the raw `VkBuffer` + `VkDeviceMemory` as a `wgpu::Buffer` so
// downstream code uses wgpu's normal bind-group / lifetime
// machinery, mirroring how `dlss_wgpu` interops with wgpu.
#![allow(unsafe_code)]

//! Raw-Vulkan buffer creation for the cluster acceleration-structure
//! pipeline, wrapped as `wgpu::Buffer`.
//!
//! Why
//! ---
//!
//! NV cluster_AS / partitioned_AS storage and AS-build-input buffers
//! need Vulkan usage flags that `wgpu::BufferUsages` does not expose
//! (the maintainers are explicitly moving away from Vulkan-only flag
//! additions — wgpu issue [#7872]). The blessed interop pattern, used
//! by wgpu's own KHR BLAS implementation
//! (`wgpu-hal/src/vulkan/device.rs`) and by `dlss_wgpu` (wgpu issue
//! [#4067]), is:
//!
//! 1. Caller creates the raw `VkBuffer` with whatever Vulkan flags it
//!    needs, allocates and binds `VkDeviceMemory`.
//! 2. Caller wraps the pair via [`wgpu::hal::vulkan::Buffer::from_raw_managed`].
//! 3. Caller hands the hal buffer to [`wgpu::Device::create_buffer_from_hal`].
//!
//! From step 3 on the buffer is a normal `wgpu::Buffer` — bind groups,
//! tracker, `Drop` (which frees both buffer + memory) all work as
//! usual. The cluster-AS build paths reach the raw `VkBuffer` via
//! `buffer.as_hal::<Vulkan>()` when needed.
//!
//! Memory
//! ------
//!
//! No gpu-allocator. The AS pipeline allocates a small number of
//! monolithic buffers (one CLAS arena, one BLAS pool, one PTLAS
//! storage, a few scratch + per-frame buffers). Sub-ranges within
//! the arenas are tracked CPU-side as byte offsets — `range-alloc`
//! does the bookkeeping. Direct `vkAllocateMemory` per buffer avoids
//! the gpu-allocator/wgpu double-ownership of memory the
//! [`Buffer::from_raw_managed`] contract creates.
//!
//! [#7872]: https://github.com/gfx-rs/wgpu/issues/7872
//! [#4067]: https://github.com/gfx-rs/wgpu/issues/4067

use ash::vk::{self, TaggedStructure};
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res},
};
use bevy_render::render_resource::Buffer;
use bevy_render::renderer::{
    raw_vulkan_init::AdditionalVulkanFeatures, RenderDevice, RenderQueue,
};
use core::ops::{Deref, Range};
use std::sync::{Arc, Mutex};
use wgpu::hal::api::Vulkan as VkApi;

use super::extension::ClusterAccelerationStructureFeature;

/// Where a buffer's backing memory lives. Selects the
/// `VkMemoryPropertyFlags` mask passed to memory-type selection.
#[derive(Copy, Clone, Debug)]
pub enum MemoryLocation {
    /// GPU-side only — `DEVICE_LOCAL`. AS storage, BLAS pool, scratch,
    /// most per-frame compute outputs.
    GpuOnly,
    /// CPU-writable, GPU-readable — `HOST_VISIBLE | HOST_COHERENT`.
    /// Args buffers staged from CPU, vertex/index uploads, etc.
    CpuToGpu,
    /// GPU-writable, CPU-readable — `HOST_VISIBLE | HOST_COHERENT`.
    /// Readback for one-shot debugging (e.g. mapping per-cluster CLAS
    /// addresses back during CLAS-arena bring-up).
    GpuToCpu,
}

/// Device address stable for its buffer's WHOLE lifetime — only the
/// never-moving wrappers ([`SparseBuffer`], `StableStorageBuffer`,
/// `PersistentGpuBuffer`) mint one. Safe to store across frames (CPU structs,
/// GPU tables, cached params).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct StableAddr(u64);

impl StableAddr {
    #[inline]
    pub(crate) fn new(addr: vk::DeviceAddress) -> Self {
        Self(addr)
    }

    #[inline]
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Device address valid only for work recorded before this frame's submit —
/// consume it inline; do NOT store it. An address that must outlive the frame
/// belongs to a stable wrapper ([`StableAddr`]) or behind a
/// [`GpuRetire`](super::retire::GpuRetire) guard.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct SubmitAddr(u64);

impl SubmitAddr {
    #[inline]
    pub fn get(self) -> u64 {
        self.0
    }
}

/// A buffer whose device address escaped to GPU consumers and which is
/// therefore PINNED: created once, never replaced. Owns the only handle —
/// swapping the buffer means dropping the wrapper, a deliberate, greppable
/// act (and a bug unless every capturing consumer is provably done).
pub struct PinnedBuffer {
    buffer: Buffer,
    addr: StableAddr,
}

impl PinnedBuffer {
    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    #[inline]
    pub fn stable_addr(&self) -> StableAddr {
        self.addr
    }
}

/// Render-world resource holding raw Vulkan handles for the cluster-AS
/// pipeline's buffer allocation path. Cloning is cheap (internal Arc)
/// so downstream sub-managers can each hold their own handle.
#[derive(Resource, Clone)]
pub struct Allocator {
    inner: Arc<AllocatorInner>,
}

struct AllocatorInner {
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// wgpu's queue. Sparse binds go through `Queue::as_hal_locked`,
    /// which holds wgpu's submission-serializing lock around the raw
    /// `vkQueueBindSparse` — the ONLY way to satisfy Vulkan's external-
    /// synchronization requirement on `VkQueue` against wgpu's own
    /// `vkQueueSubmit`/`vkQueuePresentKHR` on other threads. Commits run
    /// on Compute Task Pool threads while the render thread submits;
    /// an unshared lock here is an intermittent Xid 32 (corrupted
    /// pushbuffer) device-loss, worst during startup's commit storm.
    /// All desktop GPUs expose `SPARSE_BINDING_BIT` on their graphics
    /// queue, so binding on wgpu's queue is always legal.
    queue: RenderQueue,
    /// (min, max) VA over every sparse reservation — the "plausible device
    /// address" span for debug validation (`SOLARI_PTLAS_VALIDATE`).
    sparse_va_span: Mutex<(u64, u64)>,
}

impl Allocator {
    /// Pull raw Vulkan handles from `render_device` and cache physical-
    /// device memory properties for type-index lookup. Returns `None`
    /// if the device is not Vulkan-backed.
    pub fn try_new(render_device: &RenderDevice, render_queue: &RenderQueue) -> Option<Self> {
        // SAFETY: as_hal yields the raw Vulkan device while the wgpu
        // Device is alive; we clone the ash handles (Arc-internally)
        // so the cloned values outlive the guard.
        let (instance, device, physical_device) = unsafe {
            let guard = render_device.wgpu_device().as_hal::<VkApi>()?;
            let hal_device: &wgpu::hal::vulkan::Device = guard.deref();
            (
                hal_device.shared_instance().raw_instance().clone(),
                hal_device.raw_device().clone(),
                hal_device.raw_physical_device(),
            )
        };

        // SAFETY: physical_device is valid; this Vulkan call has no
        // chained extension structs.
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Some(Self {
            inner: Arc::new(AllocatorInner {
                instance,
                device,
                physical_device,
                memory_properties,
                queue: render_queue.clone(),
                sparse_va_span: Mutex::new((u64::MAX, 0)),
            }),
        })
    }

    /// (min, max) VA across all sparse reservations so far — every pool-held
    /// BLAS/CLAS device address lives inside this span.
    pub fn sparse_va_span(&self) -> (u64, u64) {
        *self.inner.sparse_va_span.lock().unwrap()
    }

    /// Drain the queue before a raw `vkFree*`/`vkDestroy*`. wgpu defers its own
    /// destruction behind fence waits, but raw destroys in `Drop` run immediately
    /// — at teardown the last frame's submissions can still reference the object
    /// (VUID-vkFreeMemory-memory-00677 et al.). `vkQueueWaitIdle` under wgpu's
    /// submission lock is externally-synced and near-free when already idle.
    /// Call from every raw-destroying `Drop` (see `SparseBuffer`, `MeshOmm`,
    /// `RtViewBindings`, `RtPipeline`).
    pub fn quiesce_before_raw_destroy(&self) {
        // SAFETY: the callback holds the submission lock (external sync on the
        // queue); the device handle is alive (self keeps it so).
        unsafe {
            self.inner.queue.as_hal_locked::<VkApi, _>(|queue| {
                if let Some(queue) = queue {
                    let _ = self.inner.device.queue_wait_idle(queue.as_raw());
                }
            });
        }
    }

    /// Pin `buffer`: take ownership and capture its address as [`StableAddr`].
    /// The wrapper must outlive every GPU consumer of the address — in
    /// practice, live in an init-created resource for the app's lifetime.
    pub fn pin_buffer(&self, buffer: Buffer) -> PinnedBuffer {
        let addr = StableAddr(self.wgpu_buffer_device_address(&buffer).get());
        PinnedBuffer { buffer, addr }
    }

    /// Raw `ash::Device` for callers issuing raw Vulkan commands
    /// (`vkCmdPipelineBarrier`, descriptor-set updates, etc.).
    #[inline]
    pub fn device(&self) -> &ash::Device {
        &self.inner.device
    }

    /// Raw `ash::Instance` — exposed for physical-device property
    /// queries (e.g. cluster_AS limits).
    #[inline]
    pub fn instance(&self) -> &ash::Instance {
        &self.inner.instance
    }

    #[inline]
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.inner.physical_device
    }

    /// Allocate a `wgpu::Buffer` whose underlying `VkBuffer` has the
    /// supplied raw Vulkan usage flags. wgpu manages lifetime via
    /// [`wgpu::hal::vulkan::Buffer::from_raw_managed`] — dropping the
    /// returned buffer destroys the `VkBuffer` and frees the
    /// `VkDeviceMemory`.
    ///
    /// `vk_flags` are the raw Vulkan flags (e.g.
    /// `ACCELERATION_STRUCTURE_STORAGE_KHR`,
    /// `ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR`).
    /// `SHADER_DEVICE_ADDRESS_BIT` is added automatically since every
    /// AS-pipeline buffer is reachable via device address.
    ///
    /// `wgpu_usage` is what wgpu's tracker thinks the buffer can do —
    /// pass `BufferUsages::STORAGE | COPY_DST | COPY_SRC` for buffers
    /// bound to compute shaders, or just `COPY_DST | COPY_SRC` for
    /// buffers used only via raw VK (e.g. AS-storage payload pools).
    ///
    /// # Panics
    ///
    /// Panics if `vkCreateBuffer`, `vkAllocateMemory`, or
    /// `vkBindBufferMemory` fail. These are setup-time errors that
    /// indicate driver / memory-budget misconfiguration; the caller
    /// can't usefully recover.
    pub fn create_buffer(
        &self,
        render_device: &RenderDevice,
        vk_flags: vk::BufferUsageFlags,
        wgpu_usage: wgpu::BufferUsages,
        size: u64,
        location: MemoryLocation,
        label: &'static str,
    ) -> wgpu::Buffer {
        self.create_buffer_raw(render_device, vk_flags, wgpu_usage, size, location, label)
            .0
    }

    /// Like [`create_buffer`](Self::create_buffer) but also returns the raw
    /// `vk::Buffer` handle backing the wgpu buffer — needed by raw-VK APIs that
    /// take a `VkBuffer` directly (e.g. `vkCreateMicromapEXT`'s `buffer` field),
    /// which can't be recovered from the wgpu wrapper. The handle's lifetime is
    /// tied to the returned `wgpu::Buffer` (which owns + drops it).
    pub fn create_buffer_raw(
        &self,
        render_device: &RenderDevice,
        vk_flags: vk::BufferUsageFlags,
        wgpu_usage: wgpu::BufferUsages,
        size: u64,
        location: MemoryLocation,
        label: &'static str,
    ) -> (wgpu::Buffer, vk::Buffer) {
        let size = size.max(4);
        let device = &self.inner.device;

        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk_flags | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: create_info is fully populated; no extension chain.
        let raw_buffer = unsafe { device.create_buffer(&create_info, None) }
            .expect("allocator.create_buffer: vkCreateBuffer failed");

        // SAFETY: raw_buffer is a valid handle from create_buffer above.
        let requirements = unsafe { device.get_buffer_memory_requirements(raw_buffer) };

        let property_flags = match location {
            MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryLocation::CpuToGpu | MemoryLocation::GpuToCpu => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
        };
        let memory_type_index = self
            .find_memory_type(requirements.memory_type_bits, property_flags)
            .unwrap_or_else(|| {
                panic!(
                    "allocator.create_buffer: no memory type matches \
                     requirements.bits=0x{:x} props={:?}",
                    requirements.memory_type_bits, property_flags,
                )
            });

        // The buffer was created with SHADER_DEVICE_ADDRESS, so its
        // memory MUST be allocated with the DEVICE_ADDRESS allocate
        // flag (Vulkan spec VUID-vkAllocateMemory-pNext-09127).
        let mut flags_info = vk::MemoryAllocateFlagsInfo::default()
            .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push(&mut flags_info);

        // SAFETY: alloc_info is fully populated; flags_info is
        // a valid extension struct.
        let memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .expect("allocator.create_buffer: vkAllocateMemory failed");

        // SAFETY: buffer + memory are both valid; offset 0 is the
        // canonical bind point for a dedicated-per-buffer allocation.
        unsafe {
            device
                .bind_buffer_memory(raw_buffer, memory, 0)
                .expect("allocator.create_buffer: vkBindBufferMemory failed");
        }

        // SAFETY: from_raw_managed contract:
        // - We will not touch raw_buffer or memory after this call
        //   except through the returned wgpu Buffer (which drops both
        //   when its tracker decides).
        // - offset = 0 and size = requirements.size are valid against
        //   the allocation we just created.
        // - Externally imported buffers can't be wgpu-mapped — we
        //   never call `wgpu::Buffer::slice(..).map_async` on these.
        let hal_buffer = unsafe {
            wgpu::hal::vulkan::Buffer::from_raw_managed(
                raw_buffer,
                memory,
                0,
                requirements.size,
            )
        };

        // SAFETY: create_buffer_from_hal contract:
        // - hal_buffer was created from this device's raw handle.
        // - hal_buffer respects desc: size matches, the requested
        //   wgpu_usage flags are a subset of what the underlying
        //   VkBuffer supports (we added SHADER_DEVICE_ADDRESS but
        //   that's a strict superset of wgpu's view of the buffer).
        // - hal_buffer is initialized (vkCreateBuffer + bind).
        // - size > 0.
        let wgpu_buffer = unsafe {
            render_device.wgpu_device().create_buffer_from_hal::<VkApi>(
                hal_buffer,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu_usage,
                    mapped_at_creation: false,
                },
            )
        };
        // Crash forensics: dedicated allocation ⇒ this VA range is exclusively
        // this buffer's, so an Aftermath fault VA inside it names it directly.
        let addr = unsafe {
            device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(raw_buffer),
            )
        };
        tracing::debug!(
            "buffer {label}: va 0x{addr:x}..0x{:x} ({} KiB)",
            addr + requirements.size,
            requirements.size >> 10,
        );

        (wgpu_buffer, raw_buffer)
    }

    /// Allocate a sparse `wgpu::Buffer`. Reserves a virtual address
    /// range of `virtual_size` bytes but binds no memory upfront —
    /// pages must be committed via [`SparseBuffer::commit`] before
    /// access. The `VkBuffer` handle + `VkDeviceAddress` stay stable
    /// for the buffer's lifetime; growing the committed region
    /// requires no copy and no buffer recreation.
    ///
    /// Backed by `VK_BUFFER_CREATE_SPARSE_BINDING_BIT` (core Vulkan
    /// 1.0). The `sparse_binding` device feature must have been
    /// enabled at device creation — `bevy_solari` does this in
    /// [`super::extension::register_cluster_extension_callback`].
    ///
    /// `vk_flags` / `wgpu_usage` mirror [`create_buffer`](Self::create_buffer).
    /// `SHADER_DEVICE_ADDRESS` is added automatically.
    ///
    /// # Panics
    ///
    /// Panics on `vkCreateBuffer` failure (e.g. device doesn't
    /// support `sparseBinding`).
    pub fn create_sparse_buffer(
        &self,
        render_device: &RenderDevice,
        vk_flags: vk::BufferUsageFlags,
        wgpu_usage: wgpu::BufferUsages,
        virtual_size: u64,
        label: &'static str,
    ) -> SparseBuffer {
        let virtual_size = virtual_size.max(4);
        let device = &self.inner.device;

        let create_info = vk::BufferCreateInfo::default()
            .size(virtual_size)
            .usage(vk_flags | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .flags(vk::BufferCreateFlags::SPARSE_BINDING);

        // SAFETY: create_info is fully populated; no extension chain.
        let raw_buffer = unsafe { device.create_buffer(&create_info, None) }
            .expect("allocator.create_sparse_buffer: vkCreateBuffer failed");

        // SAFETY: raw_buffer is a valid handle from create_buffer.
        let requirements = unsafe { device.get_buffer_memory_requirements(raw_buffer) };
        let page_size = requirements.alignment;
        debug_assert!(page_size.is_power_of_two());

        // SAFETY: from_raw contract: caller manages memory. Our
        // [`SparseBuffer`] drop unbinds all pages + destroys the
        // VkBuffer + frees memory chunks.
        let hal_buffer = unsafe { wgpu::hal::vulkan::Buffer::from_raw(raw_buffer) };

        // SAFETY: create_buffer_from_hal contract:
        // - hal_buffer was created from this device.
        // - desc matches the underlying VkBuffer's size / usage.
        // - hal_buffer is "initialized" (vkCreateBuffer succeeded).
        //   wgpu only reads the handle for command recording; it
        //   never dereferences memory. Sparse-uncommitted pages would
        //   fault on shader access, but barrier insertion / handle
        //   tracking is fine.
        // - virtual_size > 0.
        let wgpu_buffer = unsafe {
            render_device.wgpu_device().create_buffer_from_hal::<VkApi>(
                hal_buffer,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: virtual_size,
                    usage: wgpu_usage,
                    mapped_at_creation: false,
                },
            )
        };

        // SAFETY: buffer created with SHADER_DEVICE_ADDRESS.
        let address = unsafe {
            self.inner.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(raw_buffer),
            )
        };

        // Crash forensics: an Aftermath dump names only a faulting GPU VA —
        // this line is the map from that VA back to a buffer. Grep the run
        // log for the range containing the fault address.
        tracing::info!(
            "sparse buffer {label}: va 0x{address:x}..0x{:x} ({} MiB virtual)",
            address + virtual_size,
            virtual_size >> 20,
        );
        {
            let mut span = self.inner.sparse_va_span.lock().unwrap();
            span.0 = span.0.min(address);
            span.1 = span.1.max(address + virtual_size);
        }

        let bevy_buffer = Buffer::from(wgpu_buffer.clone());
        SparseBuffer {
            raw: raw_buffer,
            wgpu_buffer,
            bevy_buffer,
            address,
            virtual_size,
            page_size,
            memory_type_bits: requirements.memory_type_bits,
            committed_pages: Mutex::new(Vec::new()),
            memory_chunks: Mutex::new(Vec::new()),
            allocator: self.clone(),
            label,
        }
    }

    /// Resolve a `wgpu::Buffer`'s `VkDeviceAddress` — as a [`SubmitAddr`]:
    /// valid for THIS frame's recording only, never stored. Requires
    /// `SHADER_DEVICE_ADDRESS_BIT` on the underlying `VkBuffer` — set
    /// automatically for buffers from [`create_buffer`](Self::create_buffer),
    /// and set by wgpu itself when `bufferDeviceAddress` is enabled.
    pub fn wgpu_buffer_device_address(&self, buf: &wgpu::Buffer) -> SubmitAddr {
        // SAFETY: as_hal yields the raw VkBuffer while `buf` is alive;
        // we only read the handle.
        let raw = unsafe {
            buf.as_hal::<VkApi>()
                .expect("allocator: wgpu buffer is not Vulkan-backed")
                .raw_handle()
        };
        // SAFETY: raw is a live VkBuffer; SHADER_DEVICE_ADDRESS
        // guaranteed by the contract above.
        let addr = unsafe {
            self.inner.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(raw),
            )
        };
        // Crash forensics: every buffer a raw-VK op references by address
        // passes through here — an Aftermath fault VA inside [addr, addr+size)
        // names it. Debug-gated (persistent consumers re-query per frame).
        tracing::debug!("addr taken: 0x{addr:x}+{} ({:?})", buf.size(), buf);
        SubmitAddr(addr)
    }

    fn find_memory_type(
        &self,
        type_bits: u32,
        required_props: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let props = &self.inner.memory_properties;
        (0..props.memory_type_count).find(|&i| {
            let has_bit = type_bits & (1 << i) != 0;
            let has_props = props.memory_types[i as usize]
                .property_flags
                .contains(required_props);
            has_bit && has_props
        })
    }
}

/// Sparse-bound `wgpu::Buffer` with a fixed virtual address range and
/// lazily-committed memory pages. Backing memory is bound in
/// page-sized chunks via [`vkQueueBindSparse`](https://docs.vulkan.org/refpages/latest/refpages/source/vkQueueBindSparse.html)
/// the first time [`Self::commit`] is called for a given range.
///
/// **Invariants**:
/// - [`Self::raw`] handle + [`Self::address`] stay stable for the
///   buffer's lifetime — bind groups + recorded device addresses
///   keep pointing at the same buffer across growth.
/// - Reading / writing a page that hasn't been committed faults on
///   the GPU. Callers MUST `commit` before submitting work that
///   touches a page.
///
/// **Lifecycle**: `Drop` unbinds every committed page, destroys the
/// `VkBuffer` (after the wgpu Buffer handle is also dropped — wgpu
/// tracks lifetime through `from_raw`), then frees every memory
/// chunk we ever allocated. Memory chunks are *never* freed
/// individually mid-lifetime — `commit` only grows the committed set.
/// Page eviction is a future milestone.
pub struct SparseBuffer {
    /// Raw `VkBuffer` — sparse-bindable. Stable for the lifetime.
    raw: vk::Buffer,
    /// `wgpu::Buffer` wrapper around [`Self::raw`] via `Buffer::from_raw`.
    /// wgpu manages bind-group / barrier tracking; we own memory.
    pub wgpu_buffer: wgpu::Buffer,
    /// Stable bevy `Buffer` view of [`Self::wgpu_buffer`] (a clone of the same
    /// handle), created once so its `Buffer::id()` is fixed for the lifetime —
    /// for consumers that key bind-group caches on the id or store a `Buffer`
    /// (e.g. `GpuColumn`, the transform `world`). Held here so both handles drop
    /// together inside this struct, after [`Drop`] frees the memory chunks.
    bevy_buffer: Buffer,
    /// `vkGetBufferDeviceAddress` of [`Self::raw`]. Stable for the
    /// lifetime.
    pub address: vk::DeviceAddress,
    /// Total virtual size in bytes.
    pub virtual_size: u64,
    /// Page granularity reported by `vkGetBufferMemoryRequirements.alignment`.
    /// Typically 64 KB on NV.
    pub page_size: u64,
    /// Memory type bits compatible with this sparse buffer (also from
    /// `vkGetBufferMemoryRequirements`). Each backing chunk must come
    /// from a memory type covered by this mask.
    memory_type_bits: u32,
    /// Bitset of committed pages — index = page_index, true = bound.
    /// `Mutex` so commits can be issued from systems that hold
    /// `Res<Allocator>` rather than `ResMut`.
    committed_pages: Mutex<Vec<u8>>,
    /// All `VkDeviceMemory` chunks we've ever allocated for this
    /// buffer. Kept so `Drop` can free them; not used for re-binding.
    memory_chunks: Mutex<Vec<vk::DeviceMemory>>,
    /// Cloned [`Allocator`] handle — keeps the device + queue alive
    /// for the buffer's lifetime so `Drop` can free safely.
    allocator: Allocator,
    /// Debug label propagated to memory allocations.
    label: &'static str,
}

impl SparseBuffer {
    /// Stable bevy `Buffer` view of [`Self::wgpu_buffer`] — created once, so its
    /// `Buffer::id()` is fixed for the lifetime. For consumers that key bind-group
    /// caches on the id or store the `Buffer` (e.g. `GpuColumn`, the transform
    /// `world`); binds the same underlying buffer as [`Self::wgpu_buffer`].
    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.bevy_buffer
    }

    /// The buffer's [`StableAddr`] — sparse reservations never move or free
    /// while the wrapper lives, so this is safe to store across frames.
    #[inline]
    pub fn stable_addr(&self) -> StableAddr {
        StableAddr(self.address)
    }

    #[inline]
    pub fn label(&self) -> &'static str {
        self.label
    }

    /// Whether every page covering `byte_range` is currently committed —
    /// a raw op consuming an uncommitted range is a future device-lost.
    pub fn is_committed(&self, byte_range: Range<u64>) -> bool {
        if byte_range.is_empty() {
            return true;
        }
        let committed = self.committed_pages.lock().unwrap();
        let start_page = byte_range.start / self.page_size;
        let end_page = byte_range.end.div_ceil(self.page_size);
        (start_page..end_page).all(|p| bit_get(&committed, p as usize))
    }

    /// Ensure every page covering `byte_range` is backed by memory.
    /// See [`Self::commit_many`] — one bind + fence wait per call, so
    /// batch scattered ranges through that instead of looping this.
    pub fn commit(&self, byte_range: Range<u64>) {
        self.commit_many(core::iter::once(byte_range));
    }

    /// Ensure every page covering each range is backed by memory.
    /// Already-committed pages are skipped. ALL ranges fold into one
    /// `vkQueueBindSparse` + one CPU fence wait (sparse binds aren't
    /// ordered against subsequent queue submits without explicit sync).
    ///
    /// # Panics
    ///
    /// Panics on `vkAllocateMemory` / `vkQueueBindSparse` /
    /// `vkWaitForFences` failure.
    pub fn commit_many(&self, byte_ranges: impl IntoIterator<Item = Range<u64>>) {
        let _span = tracing::info_span!("SparseBuffer::commit", label = self.label).entered();
        let page_size = self.page_size;

        let mut committed = self.committed_pages.lock().unwrap();
        let mut binds: Vec<vk::SparseMemoryBind> = Vec::new();
        let mut new_memories: Vec<vk::DeviceMemory> = Vec::new();
        for byte_range in byte_ranges {
            let start_page = byte_range.start / page_size;
            let end_page = byte_range.end.div_ceil(page_size);
            debug_assert!((end_page * page_size) <= self.virtual_size);

            // Grow bitset to cover up to end_page.
            let needed_bytes = (end_page as usize).div_ceil(8);
            if committed.len() < needed_bytes {
                committed.resize(needed_bytes, 0u8);
            }

            // Find contiguous runs of uncommitted pages and bind them in
            // one VkSparseMemoryBind per run.
            let mut p = start_page;
            while p < end_page {
                // Skip already-committed pages.
                while p < end_page && bit_get(&committed, p as usize) {
                    p += 1;
                }
                if p >= end_page {
                    break;
                }
                let run_start = p;
                while p < end_page && !bit_get(&committed, p as usize) {
                    bit_set(&mut committed, p as usize);
                    p += 1;
                }
                let run_pages = p - run_start;
                let run_bytes = run_pages * page_size;

                // Allocate fresh VkDeviceMemory for this run, split into chunks no
                // larger than `MAX_CHUNK_BYTES`: a single `vkAllocateMemory` must stay
                // under `maxMemoryAllocationSize` (~4 GB on NV), and large OMM-bearing
                // BLAS pools blow past that in one run. Page-aligned so each bind lands
                // on a page boundary. Sub-allocating from a shared pool would reduce the
                // VkDeviceMemory count; first-version keeps it dumb.
                const MAX_CHUNK_BYTES: u64 = 2 * 1024 * 1024 * 1024;
                let max_chunk = (MAX_CHUNK_BYTES / page_size) * page_size;
                let mut chunk_offset = run_start * page_size;
                let mut remaining = run_bytes;
                while remaining > 0 {
                    let chunk = remaining.min(max_chunk);
                    let memory = self.allocate_chunk(chunk);
                    new_memories.push(memory);
                    binds.push(vk::SparseMemoryBind {
                        resource_offset: chunk_offset,
                        size: chunk,
                        memory,
                        memory_offset: 0,
                        flags: vk::SparseMemoryBindFlags::empty(),
                    });
                    chunk_offset += chunk;
                    remaining -= chunk;
                }
            }
        }

        if binds.is_empty() {
            return;
        }

        // Stash the new chunks before issuing the bind so Drop can
        // free them even if vkQueueBindSparse panics below.
        self.memory_chunks.lock().unwrap().extend(&new_memories);

        let buffer_bind = vk::SparseBufferMemoryBindInfo::default()
            .buffer(self.raw)
            .binds(&binds);
        let buffer_binds = [buffer_bind];
        let bind_info = vk::BindSparseInfo::default().buffer_binds(&buffer_binds);
        let bind_infos = [bind_info];

        let device = &self.allocator.inner.device;
        // Fence to wait on — sparse-bind ops aren't queue-ordered
        // against subsequent submits without explicit sync.
        // SAFETY: default fence create-info, no chained structs.
        let fence = unsafe {
            device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("SparseBuffer.commit: vkCreateFence failed")
        };

        // SAFETY: queue + fence + bind_infos all valid; binds[] points
        // at memory we just allocated; resource ranges fit inside the
        // sparse buffer's virtual size. `as_hal_locked` holds wgpu's
        // submission lock across the bind + wait — Vulkan requires
        // `VkQueue` to be externally synchronized, INCLUDING against
        // wgpu's own submits/presents on other threads.
        // The actual page bind + blocking fence wait — the part that
        // only runs when new pages are committed (growth / streaming).
        // Steady state skips this entirely (binds.is_empty() above).
        let _bind_span = tracing::info_span!(
            "queue_bind_sparse+wait",
            label = self.label,
            pages = binds.iter().map(|b| b.size / page_size).sum::<u64>(),
        )
        .entered();
        unsafe {
            self.allocator.inner.queue.as_hal_locked::<VkApi, _>(|queue| {
                let queue = queue
                    .expect("SparseBuffer.commit: wgpu queue is not Vulkan-backed")
                    .as_raw();
                device
                    .queue_bind_sparse(queue, &bind_infos, fence)
                    .expect("SparseBuffer.commit: vkQueueBindSparse failed");
                if let Err(err) = device.wait_for_fences(&[fence], true, u64::MAX) {
                    // Device lost: name the last AS pass the GPU reached before
                    // dying (per-pass checkpoints), then die as before.
                    crate::gpu::extension::report_queue_checkpoints(queue);
                    panic!("SparseBuffer.commit: vkWaitForFences failed: {err:?}");
                }
                device.destroy_fence(fence, None);
            });
        }
    }

    fn allocate_chunk(&self, size: u64) -> vk::DeviceMemory {
        let device = &self.allocator.inner.device;
        let memory_type_index = self
            .allocator
            .find_memory_type(self.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .unwrap_or_else(|| {
                panic!(
                    "SparseBuffer({}): no DEVICE_LOCAL memory type covers \
                     requirements.bits=0x{:x}",
                    self.label, self.memory_type_bits,
                )
            });
        let mut flags_info = vk::MemoryAllocateFlagsInfo::default()
            .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index)
            .push(&mut flags_info);
        // SAFETY: alloc_info populated; DEVICE_ADDRESS flag matches
        // the buffer's SHADER_DEVICE_ADDRESS usage.
        unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .expect("SparseBuffer.allocate_chunk: vkAllocateMemory failed")
        }
    }
}

impl Drop for SparseBuffer {
    fn drop(&mut self) {
        // In-flight submissions may still reference this buffer's memory
        // (teardown drops mid-last-frame); drain before freeing.
        self.allocator.quiesce_before_raw_destroy();
        let device = &self.allocator.inner.device;
        // `wgpu_buffer` (and the `bevy_buffer` clone of the same handle)
        // drop via their own Drop after this method returns — the last
        // of the two calls vkDestroyBuffer on `raw`. We do NOT destroy
        // `raw` ourselves to avoid double-free; wgpu's from_raw wrapper
        // owns the destroy. Both are fields here, so neither outlives the
        // struct (the destroy fires once, after the chunks are freed below).
        //
        // But: sparse bindings hold the memory chunks live as far
        // as the driver is concerned. After wgpu destroys the
        // buffer, all bindings are implicitly released, and the
        // memory can be freed. So we just free the chunks here
        // and let wgpu destroy the buffer.
        let chunks = std::mem::take(&mut *self.memory_chunks.lock().unwrap());
        for memory in chunks {
            // SAFETY: chunks were allocated via vkAllocateMemory on
            // the same device; no aliasing references remain.
            unsafe {
                device.free_memory(memory, None);
            }
        }
    }
}

#[inline]
fn bit_get(bits: &[u8], i: usize) -> bool {
    bits.get(i >> 3).is_some_and(|b| (b >> (i & 7)) & 1 == 1)
}

#[inline]
fn bit_set(bits: &mut [u8], i: usize) {
    bits[i >> 3] |= 1u8 << (i & 7);
}

/// `RenderStartup` system: insert [`Allocator`] iff the cluster-AS
/// extension was enabled on the device by [`crate::SolariInitPlugin`].
/// Downstream cluster-AS systems should treat it as
/// `Option<Res<Allocator>>` and skip on unsupported hardware.
pub fn init_allocator(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    additional: Res<AdditionalVulkanFeatures>,
) {
    if !additional.has::<ClusterAccelerationStructureFeature>()
        || !additional.has::<crate::gpu::extension::OpacityMicromapFeature>()
    {
        // Solari is silently disabled if we just return with no allocator — every
        // downstream `Option<Res<Allocator>>` init then no-ops, with no signal as to
        // why. Say it once so a "nothing renders" report has an obvious first answer.
        // Opacity micromaps are required alongside cluster AS: every driver with
        // the NV cluster extensions has VK_EXT_opacity_micromap, and requiring it
        // keeps a no-OMM fallback out of every build/attach path.
        bevy_log::warn_once!(
            "bevy_solari disabled: the GPU/driver lacks the cluster acceleration-structure \
             feature (VK_NV_cluster_acceleration_structure et al.) or VK_EXT_opacity_micromap. \
             All solari passes no-op."
        );
        return;
    }
    let Some(memory) = Allocator::try_new(&render_device, &render_queue) else {
        bevy_log::warn_once!(
            "bevy_solari disabled: raw-VK allocator init failed despite the cluster-AS feature \
             being present. All solari passes no-op."
        );
        return;
    };
    commands.insert_resource(memory);
}
