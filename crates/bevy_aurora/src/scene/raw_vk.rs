#![allow(
    clippy::undocumented_unsafe_blocks,
    reason = "Each unsafe fn in this module documents its own safety contract; per-block \
              SAFETY comments would just restate the fn-level contract verbatim."
)]
#![allow(
    clippy::doc_markdown,
    reason = "Vulkan / Cargo identifiers (`vk::*`, `M-A.streaming`, `cluster_AS`, etc.) \
              would each need backticks otherwise."
)]
//! Raw Vulkan primitives owned by Aurora's scene management.
//!
//! Aurora bypasses wgpu's `Buffer` abstraction for the cluster_AS pipeline
//! because the relevant Vulkan buffer usages
//! (`ACCELERATION_STRUCTURE_STORAGE_KHR`,
//! `ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR`) aren't exposed by
//! [`wgpu::BufferUsages`], and because cluster ASes don't fit wgpu's `Blas` /
//! `Tlas` types. Everything in this module is `unsafe` and assumes the caller
//! has acquired a working [`ash::Device`] via Aurora's plugin setup.
//!
//! Three primitives:
//!
//! - [`RawBuffer`] — single committed VkBuffer + memory + device address.
//!   Used for transient one-shot allocations (descriptor staging, count
//!   buffers, scratch).
//! - [`PersistentBuffer`] — long-lived backing for the CLAS array, AS storage,
//!   etc. M-A is committed-memory; `M-A.streaming` will swap the backend to
//!   sparse-residency (`VK_BUFFER_CREATE_SPARSE_BINDING_BIT` +
//!   `vkQueueBindSparse`) without changing the public API.
//! - [`RangeAllocator`] — bump allocator over a virtual byte range. M-A only
//!   needs allocate-forward; `M-A.streaming` will replace the implementation
//!   with a real free-list (probably an interval tree or buddy allocator).
//!   `free()` is a no-op stub today so call sites compile against the future API.

use ash::vk::{self, TaggedStructure as _};
use core::ops::Range;

// ---- RawBuffer --------------------------------------------------------------

/// A single VkBuffer + its backing VkDeviceMemory + cached device address.
///
/// `RawBuffer` is unaware of bevy's resource lifecycle — the caller is
/// responsible for calling [`Self::destroy`] before the underlying
/// [`ash::Device`] is destroyed, otherwise we leak the GPU memory.
pub struct RawBuffer {
    pub buf: vk::Buffer,
    pub mem: vk::DeviceMemory,
    pub size: u64,
    pub addr: u64,
}

impl RawBuffer {
    /// Allocate a committed buffer.
    ///
    /// `usage` is OR-ed with `SHADER_DEVICE_ADDRESS` so the cached `.addr`
    /// field is queryable. Memory is allocated with the
    /// `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` flag.
    ///
    /// `host_visible = true` selects host-coherent memory (uploads); `false`
    /// selects `DEVICE_LOCAL` (most AS storage / scratch).
    ///
    /// # Safety
    ///
    /// `device` and `mem_props` must come from the same physical device and
    /// remain valid until [`Self::destroy`] is called.
    pub unsafe fn alloc(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        size: u64,
        usage: vk::BufferUsageFlags,
        host_visible: bool,
    ) -> Self {
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
        let mem = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .expect("allocate_memory")
        };
        unsafe {
            device
                .bind_buffer_memory(buf, mem, 0)
                .expect("bind_buffer_memory");
        }

        let addr = unsafe {
            device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(buf),
            )
        };

        Self {
            buf,
            mem,
            size: req.size,
            addr,
        }
    }

    /// Map memory, copy `bytes`, unmap. Requires `host_visible = true` at
    /// allocation time. For non-coherent memory (which we don't allocate
    /// today), the caller would need `vkFlushMappedMemoryRanges`.
    ///
    /// # Safety
    ///
    /// Caller must guarantee no concurrent map of this buffer's memory and that
    /// `bytes.len() <= self.size`.
    pub unsafe fn upload(&self, device: &ash::Device, bytes: &[u8]) {
        debug_assert!((bytes.len() as u64) <= self.size);
        let ptr = unsafe {
            device
                .map_memory(self.mem, 0, bytes.len() as u64, vk::MemoryMapFlags::empty())
                .expect("RawBuffer::upload map_memory")
        }.cast::<u8>();
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            device.unmap_memory(self.mem);
        }
    }

    /// Free the underlying GPU memory. Must be called before `device` is
    /// destroyed.
    ///
    /// # Safety
    ///
    /// Caller must ensure no in-flight GPU work references this buffer.
    pub unsafe fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buf, None);
            device.free_memory(self.mem, None);
        }
    }
}

// ---- PersistentBuffer -------------------------------------------------------

/// A long-lived backing buffer for Aurora's persistent GPU state (CLAS array,
/// AS storage, etc.).
///
/// In M-A this is a thin wrapper over a committed [`RawBuffer`] — the entire
/// `size` range is physically resident from creation, and [`Self::ensure_resident`]
/// is a no-op. In `M-A.streaming` the inner allocation will switch to a
/// sparse-residency `VkBuffer` and `ensure_resident` will gain real semantics
/// (issuing `vkQueueBindSparse` for unbound page ranges). Callers see the same
/// `device_address(offset)` interface either way.
pub struct PersistentBuffer {
    pub label: &'static str,
    inner: RawBuffer,
}

impl PersistentBuffer {
    /// Allocate a persistent committed buffer of `size` bytes.
    ///
    /// # Safety
    ///
    /// Same as [`RawBuffer::alloc`].
    pub unsafe fn alloc(
        label: &'static str,
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        size: u64,
        usage: vk::BufferUsageFlags,
        host_visible: bool,
    ) -> Self {
        let inner = unsafe { RawBuffer::alloc(device, mem_props, size, usage, host_visible) };
        tracing::debug!(
            target: "bevy_aurora",
            label,
            size,
            addr = format!("0x{:016x}", inner.addr),
            "PersistentBuffer allocated (committed)",
        );
        Self { label, inner }
    }

    /// Device address of `offset` bytes into this buffer.
    #[inline]
    pub fn device_address(&self, offset: u64) -> u64 {
        debug_assert!(offset <= self.inner.size);
        self.inner.addr + offset
    }

    /// Total virtual size in bytes.
    #[inline]
    pub fn size(&self) -> u64 {
        self.inner.size
    }

    /// Raw VkBuffer handle. Required to pass this buffer to
    /// `vkCreateAccelerationStructureKHR`.
    #[inline]
    pub fn raw(&self) -> vk::Buffer {
        self.inner.buf
    }

    /// Ensure `range` is backed by physical memory.
    ///
    /// M-A is committed-memory only so this is a no-op. `M-A.streaming` will
    /// issue `vkQueueBindSparse` for any unbound page ranges that intersect
    /// `range`. Callers should call this before writing into a previously
    /// uninitialized range.
    ///
    /// # Safety
    ///
    /// The eventual sparse-binding implementation requires that no in-flight
    /// command buffer reads from the soon-to-be-bound range. M-A's no-op is
    /// trivially safe.
    #[inline]
    pub unsafe fn ensure_resident(&mut self, range: Range<u64>) {
        debug_assert!(range.end <= self.inner.size);
        // committed-only in M-A; placeholder for M-A.streaming.
        let _ = range;
    }

    /// Release `range` from physical residency. M-A no-op.
    ///
    /// # Safety
    ///
    /// Same constraints as [`Self::ensure_resident`].
    #[inline]
    pub unsafe fn release(&mut self, range: Range<u64>) {
        debug_assert!(range.end <= self.inner.size);
        let _ = range;
    }

    /// Map a sub-range, copy `bytes` at `offset`, unmap.
    ///
    /// # Safety
    ///
    /// Same constraints as [`RawBuffer::upload`], and
    /// `offset + bytes.len() <= self.size()`.
    pub unsafe fn upload(&self, device: &ash::Device, offset: u64, bytes: &[u8]) {
        debug_assert!(offset + bytes.len() as u64 <= self.inner.size);
        let ptr = unsafe {
            device
                .map_memory(
                    self.inner.mem,
                    offset,
                    bytes.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("PersistentBuffer::upload map_memory")
        }.cast::<u8>();
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            device.unmap_memory(self.inner.mem);
        }
    }

    /// Free GPU memory. Must be called before `device` is destroyed.
    ///
    /// # Safety
    ///
    /// No in-flight GPU work may reference this buffer.
    pub unsafe fn destroy(self, device: &ash::Device) {
        unsafe { self.inner.destroy(device) }
    }
}

// ---- RangeAllocator ---------------------------------------------------------

/// Bump allocator over a virtual byte range.
///
/// M-A only allocates — never frees. M-A.streaming will swap the implementation
/// for a real free-list. The current `free()` is a no-op stub so call sites
/// compile against the future API.
#[derive(Debug, Clone)]
pub struct RangeAllocator {
    capacity: u64,
    head: u64,
}

impl RangeAllocator {
    pub fn new(capacity: u64) -> Self {
        Self { capacity, head: 0 }
    }

    /// Allocate `size` bytes aligned to `alignment` (must be a power of two).
    /// Returns `None` if the allocator can't satisfy the request without
    /// exceeding `capacity` (or arithmetic would overflow).
    pub fn alloc(&mut self, size: u64, alignment: u64) -> Option<Range<u64>> {
        debug_assert!(alignment.is_power_of_two());
        let aligned_start = self.head.checked_add(alignment - 1)? & !(alignment - 1);
        let aligned_end = aligned_start.checked_add(size)?;
        if aligned_end > self.capacity {
            return None;
        }
        self.head = aligned_end;
        Some(aligned_start..aligned_end)
    }

    /// Free a previously allocated range. M-A bump allocator: no-op (memory is
    /// reclaimed only via [`Self::reset`]).
    #[inline]
    pub fn free(&mut self, _range: Range<u64>) {}

    /// Reset the allocator to empty. Useful for per-frame transient pools.
    #[inline]
    pub fn reset(&mut self) {
        self.head = 0;
    }

    #[inline]
    pub fn used(&self) -> u64 {
        self.head
    }

    #[inline]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    #[inline]
    pub fn remaining(&self) -> u64 {
        self.capacity.saturating_sub(self.head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bump_alloc_aligns_and_advances() {
        let mut a = RangeAllocator::new(1024);
        assert_eq!(a.alloc(100, 1), Some(0..100));
        assert_eq!(a.used(), 100);

        let r = a.alloc(64, 64).unwrap();
        assert_eq!(r.start, 128, "next 64-byte alignment after 100");
        assert_eq!(r.end, 192);

        assert!(a.alloc(2000, 1).is_none(), "exceeds capacity");
        assert_eq!(a.used(), 192, "failed alloc must not advance head");
    }

    #[test]
    fn bump_alloc_zero_size_works() {
        let mut a = RangeAllocator::new(1024);
        let r = a.alloc(0, 1).unwrap();
        assert_eq!(r.start, 0);
        assert_eq!(r.end, 0);
        assert_eq!(a.used(), 0);
    }

    #[test]
    fn bump_alloc_at_full_capacity() {
        let mut a = RangeAllocator::new(64);
        assert_eq!(a.alloc(64, 1), Some(0..64));
        assert!(a.alloc(1, 1).is_none());
        assert_eq!(a.remaining(), 0);
    }

    #[test]
    fn bump_alloc_overflow_returns_none() {
        let mut a = RangeAllocator::new(u64::MAX);
        a.head = u64::MAX - 10;
        assert!(
            a.alloc(100, 1).is_none(),
            "overflow on aligned_end must yield None, not panic"
        );
    }

    #[test]
    fn bump_alloc_alignment_overflow_returns_none() {
        let mut a = RangeAllocator::new(u64::MAX);
        a.head = u64::MAX;
        // Even alignment of 4 should fail without panicking via checked_add.
        assert!(a.alloc(0, 4).is_none());
    }

    #[test]
    fn reset_returns_to_empty() {
        let mut a = RangeAllocator::new(1024);
        a.alloc(500, 1).unwrap();
        a.reset();
        assert_eq!(a.used(), 0);
        assert!(a.alloc(1024, 1).is_some());
    }
}
