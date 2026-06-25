use bevy_render::{
    render_resource::{BindingResource, Buffer, BufferAddress, BufferUsages, COPY_BUFFER_ALIGNMENT},
    renderer::{RenderDevice, RenderQueue},
};
use ash::vk;
use core::{num::NonZero, ops::Range};
use range_alloc::RangeAllocator;
use wgpu_types::WriteOnly;

use super::allocator::{Allocator, SparseBuffer};

/// Wrapper for a GPU buffer holding a large amount of data that persists across
/// frames. Backed by a [`SparseBuffer`]: growth commits more physical pages to the
/// same (stable-handle) buffer rather than reallocating + copying, so consumer bind
/// groups never invalidate and recorded device addresses stay valid across growth.
pub struct PersistentGpuBuffer<T: PersistentGpuBufferable> {
    /// Sparse backing buffer — fixed virtual range, pages committed on growth.
    buffer: SparseBuffer,
    /// High-water (bytes) the planner has grown to and we've committed pages for.
    /// `0` until the first upload (see [`Self::is_empty`]).
    committed_bytes: u64,
    /// Tracks free slices of the buffer.
    allocation_planner: RangeAllocator<BufferAddress>,
    /// Queue of pending writes, and associated metadata.
    write_queue: Vec<(T, T::Metadata, Range<BufferAddress>)>,
}

impl<T: PersistentGpuBufferable> PersistentGpuBuffer<T> {
    /// Create a new persistent buffer backed by a sparse buffer.
    pub fn new(label: &'static str, render_device: &RenderDevice, allocator: &Allocator) -> Self {
        // Reserve the sparse buffer's fixed virtual size at the largest range a
        // single storage binding can cover (`max_storage_buffer_binding_size`,
        // ~2 GiB). Pages commit lazily, but the whole range is bound `as_entire`
        // (mesh pools index by bounded per-cluster offsets and never call
        // `arrayLength`), so the reservation must not exceed the binding limit —
        // and a pool can't hold more than this anyway (the hardware ceiling for one
        // storage binding). The stable handle/address survives growth.
        let virtual_bytes = render_device.limits().max_storage_buffer_binding_size as u64;
        Self {
            // STORAGE: shaders read it. TRANSFER_DST/COPY_DST: `perform_writes`
            // uploads via the queue. AS-build-input + SHADER_DEVICE_ADDRESS (the
            // latter added by `create_sparse_buffer`): the cluster_AS / blas_rebuild
            // paths resolve per-cluster vertex/index byte offsets by
            // `vkGetBufferDeviceAddress`. Without SDA, the cluster build silently
            // consumes garbage (VUID-VkBufferDeviceAddressInfo-buffer-02601).
            buffer: allocator.create_sparse_buffer(
                render_device,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::BLAS_INPUT,
                virtual_bytes,
                label,
            ),
            committed_bytes: 0,
            allocation_planner: RangeAllocator::new(0..0),
            write_queue: Vec::new(),
        }
    }

    /// Queue an item of type T to be added to the buffer, returning the byte range within the buffer that it will be located at.
    pub fn queue_write(&mut self, data: T, metadata: T::Metadata) -> Range<BufferAddress> {
        let data_size = data.size_in_bytes() as u64;
        debug_assert!(data_size.is_multiple_of(COPY_BUFFER_ALIGNMENT));
        if let Ok(buffer_slice) = self.allocation_planner.allocate_range(data_size) {
            self.write_queue
                .push((data, metadata, buffer_slice.clone()));
            return buffer_slice;
        }

        let buffer_size = self.allocation_planner.initial_range();
        let double_buffer_size = (buffer_size.end - buffer_size.start) * 2;
        let new_size = double_buffer_size.max(data_size);
        self.allocation_planner.grow_to(buffer_size.end + new_size);

        let buffer_slice = self.allocation_planner.allocate_range(data_size).unwrap();
        self.write_queue
            .push((data, metadata, buffer_slice.clone()));
        buffer_slice
    }

    /// Upload all pending data to the GPU buffer, committing more sparse pages
    /// first if the planner outgrew what's resident.
    pub fn perform_writes(&mut self, render_queue: &RenderQueue) {
        // Commit pages up to the planner high-water. Unlike a realloc this keeps the
        // same buffer handle/address — existing pages keep their data, so there's no
        // old→new copy and no bind-group invalidation. Newly-committed pages are
        // undefined, but every byte a consumer reads is in a written allocation
        // (inter-allocation gaps are never indexed).
        let needed = self.allocation_planner.initial_range().end;
        if needed > self.committed_bytes {
            self.buffer.commit(0..needed);
            self.committed_bytes = needed;
        }

        let queue_count = self.write_queue.len();

        for (data, metadata, buffer_slice) in self.write_queue.drain(..) {
            let buffer_slice_size =
                NonZero::<u64>::new(buffer_slice.end - buffer_slice.start).unwrap();
            let mut buffer_view = render_queue
                .write_buffer_with(&self.buffer.wgpu_buffer, buffer_slice.start, buffer_slice_size)
                .unwrap();
            data.write_bytes_le(metadata, buffer_view.slice(..), buffer_slice.start);
        }

        let queue_saturation = queue_count as f32 / self.write_queue.capacity() as f32;
        if queue_saturation < 0.3 {
            self.write_queue = Vec::new();
        }
    }

    /// Mark a section of the GPU buffer as no longer needed.
    pub fn mark_slice_unused(&mut self, buffer_slice: Range<BufferAddress>) {
        // Empty slices were never actually allocated (e.g. a small mesh with no
        // interior `nodes`, or the deliberately-`0..0` `child_table`), and
        // `free_range` panics on an empty range — so freeing one is a no-op.
        // This only matters when cluster meshes are *removed* (live re-bake in
        // tools like sly_tree, geometry churn); static scenes never hit it.
        if buffer_slice.start >= buffer_slice.end {
            return;
        }
        self.allocation_planner.free_range(buffer_slice);
    }

    /// Whether nothing has been uploaded yet (no pages committed). Consumers skip
    /// building bind groups until a pool has data — the sparse buffer is always
    /// virtual-sized, so `buffer().size()` can no longer signal emptiness with `0`.
    pub fn is_empty(&self) -> bool {
        self.committed_bytes == 0
    }

    pub fn binding(&self) -> BindingResource<'_> {
        // The whole (virtual) buffer: mesh-pool shaders index by bounded per-cluster
        // offsets and never call `arrayLength`, so binding the full range is safe and
        // lets the bind-group cache (keyed on the stable `Buffer::id`) skip rebuilds
        // on growth.
        self.buffer.buffer().as_entire_binding()
    }

    /// The bevy `Buffer` view — its `Buffer::id()` is stable across growth, so
    /// bind-group caches keyed on it never invalidate. Also used by the cluster-AS
    /// pipeline to query the buffer's (now stable) `VkDeviceAddress`.
    pub fn buffer(&self) -> &Buffer {
        self.buffer.buffer()
    }

    /// Base device address — stable for the buffer's lifetime (the sparse backing
    /// is never freed). Safe to capture for the raw RT trace; see
    /// [`RawTraceBindable`](super::raw_trace::RawTraceBindable).
    #[inline]
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.buffer.address
    }
}

/// A trait representing data that can be written to a [`PersistentGpuBuffer`].
pub trait PersistentGpuBufferable {
    /// Additional metadata associated with each item, made available during `write_bytes_le`.
    type Metadata;

    /// The size in bytes of `self`. This will determine the size of the buffer passed into
    /// `write_bytes_le`.
    ///
    /// All data written must be in a multiple of `wgpu::COPY_BUFFER_ALIGNMENT` bytes. Failure to do so will
    /// result in a panic when using [`PersistentGpuBuffer`].
    fn size_in_bytes(&self) -> usize;

    /// Convert `self` + `metadata` into bytes (little-endian), and write to the provided buffer slice.
    /// Any bytes not written to in the slice will be zeroed out when uploaded to the GPU.
    fn write_bytes_le(
        &self,
        metadata: Self::Metadata,
        buffer_slice: WriteOnly<[u8]>,
        buffer_offset: BufferAddress,
    );
}
