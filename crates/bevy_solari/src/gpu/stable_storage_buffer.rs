//! [`StableStorageBuffer`] — a persistent, **stable-address** storage buffer; a
//! drop-in for the transient `render_resource::StorageBuffer<V>` whose backing GPU
//! allocation is recreated whenever its data grows.
//!
//! Backed by a [`SparseBuffer`]: a fixed virtual range is reserved once and
//! physical pages are committed grow-only, so the buffer handle **and its device
//! address never change**, and the allocation is **never freed**.
//!
//! Why this exists: the raw RT-pipeline trace reads scene data by raw device
//! address (`physical_load`) and raw descriptor set, both of which bypass wgpu's
//! resource tracking. A transient `StorageBuffer` that reallocates when its data
//! changes (materials/lights on a regenerate) leaves an in-flight trace reading a
//! freed buffer → device-lost. A stable address that is never freed removes the
//! hazard at the root — and skips the per-frame reallocation cost. It's the AoS
//! sibling of the SoA [`GpuColumn`](crate::ecs_gpu::GpuColumn): use a column for
//! large, per-element, individually-read fields; use this for small tables read as
//! whole structs (materials, light sources).

use ash::vk;
use bevy_render::{
    render_resource::{
        encase::{internal::WriteInto, ShaderType, StorageBuffer as EncaseStorageBuffer},
        BindingResource, Buffer, BufferBinding, BufferUsages,
    },
    renderer::{RenderDevice, RenderQueue},
};
use core::num::NonZeroU64;

use super::allocator::{Allocator, SparseBuffer};

/// A persistent storage buffer at a stable device address. `V` is the whole stored
/// value (e.g. `Vec<GpuMaterial>`), matching `render_resource::StorageBuffer<V>`;
/// it's serialized std430 via encase on [`write_buffer`](Self::write_buffer).
pub struct StableStorageBuffer<V: ShaderType + WriteInto> {
    value: V,
    /// Stable-handle, stable-address sparse backing — pages committed grow-only.
    sparse: SparseBuffer,
    /// Pages committed so far, in bytes (grow-only — never shrinks/frees).
    committed_bytes: u64,
    /// This frame's serialized byte length — the size consumers must bind (a range
    /// of exactly this, NOT the virtual reservation: binding the whole sparse range
    /// makes shader `arrayLength()` enormous and hangs the GPU, see
    /// [`GpuColumn::committed_bytes`](crate::ecs_gpu::GpuColumn::committed_bytes)).
    current_bytes: u64,
    /// Reused std430 serialization scratch — avoids a per-frame allocation.
    scratch: Vec<u8>,
}

impl<V: ShaderType + WriteInto> StableStorageBuffer<V> {
    /// Reserve the buffer (largest range one storage binding can cover; pages
    /// commit lazily on first write). `SHADER_DEVICE_ADDRESS` is always added by
    /// [`Allocator::create_sparse_buffer`], so [`device_address`](Self::device_address)
    /// is valid immediately — the bindless `physical_load` read path needs it.
    pub fn new(
        value: V,
        allocator: &Allocator,
        render_device: &RenderDevice,
        label: &'static str,
    ) -> Self {
        let virtual_bytes = render_device.limits().max_storage_buffer_binding_size as u64;
        let sparse = allocator.create_sparse_buffer(
            render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            virtual_bytes,
            label,
        );
        Self {
            value,
            sparse,
            committed_bytes: 0,
            current_bytes: 0,
            scratch: Vec::new(),
        }
    }

    /// The stored value (read).
    #[inline]
    pub fn get(&self) -> &V {
        &self.value
    }

    /// The stored value (mutate before [`write_buffer`](Self::write_buffer)).
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.value
    }

    /// Replace the stored value.
    #[inline]
    pub fn set(&mut self, value: V) {
        self.value = value;
    }

    /// Serialize the value (std430) and upload it. Commits more pages first if the
    /// data grew — the address stays fixed and existing pages keep their data, so
    /// there is no reallocation and no in-flight trace ever reads a freed buffer.
    pub fn write_buffer(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        let _ = render_device; // kept for API symmetry with `StorageBuffer::write_buffer`
        self.scratch.clear();
        let mut wrapper = EncaseStorageBuffer::new(&mut self.scratch);
        wrapper
            .write(&self.value)
            .expect("StableStorageBuffer: encase serialization failed");
        self.current_bytes = self.scratch.len() as u64;
        if self.current_bytes == 0 {
            return;
        }
        if self.current_bytes > self.committed_bytes {
            self.sparse.commit(0..self.current_bytes);
            self.committed_bytes = self.current_bytes;
        }
        render_queue.write_buffer(self.sparse.buffer(), 0, &self.scratch);
    }

    /// The bevy `Buffer` view — `Buffer::id()` is stable across growth, so
    /// bind-group caches keyed on it never invalidate.
    #[inline]
    pub fn buffer(&self) -> &Buffer {
        self.sparse.buffer()
    }

    /// Base device address — stable for the buffer's lifetime. For the bindless
    /// read path (`physical_load<T>(addr + i * stride)`).
    #[inline]
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.sparse.address
    }

    /// A descriptor binding sized to **exactly this frame's data**, so shader
    /// `arrayLength()` is the real element count. `None` until the first non-empty
    /// [`write_buffer`](Self::write_buffer) (a consumer skips its bind group then,
    /// like the other RT bindings).
    #[inline]
    pub fn binding(&self) -> Option<BindingResource<'_>> {
        Some(BindingResource::Buffer(BufferBinding {
            buffer: self.sparse.buffer(),
            offset: 0,
            size: Some(NonZeroU64::new(self.current_bytes)?),
        }))
    }
}
