//! [`RawTraceBindable`] — the type-level contract for buffers whose device address
//! may be captured and handed to the raw RT-pipeline trace.
//!
//! The trace reads scene data by raw `VkDeviceAddress` (`physical_load`) and raw
//! descriptor set, both of which bypass wgpu's resource tracking and outlive the
//! frame that recorded them (a submitted trace completes several frames late). A
//! captured address is only safe if its buffer's address is **stable for the
//! buffer's lifetime and never freed** — otherwise an in-flight trace reads a
//! reallocated/freed buffer → device lost (the regenerate crash class).
//!
//! This sealed trait is implemented *only* for the sparse / stable-address wrappers
//! ([`SparseBuffer`], [`StableStorageBuffer`], [`PersistentGpuBuffer`]). The trace
//! capture path takes [`trace_device_address`](RawTraceBindable::trace_device_address),
//! so passing a transient `RawBufferVec` / `StorageBuffer` (which reallocate on
//! growth) is a **compile error**, not a runtime device-lost. Same-submission build
//! scratch that never outlives its command buffer keeps using
//! [`Allocator::wgpu_buffer_device_address`](super::allocator::Allocator::wgpu_buffer_device_address)
//! directly — it is not a cross-frame trace read and is not constrained here.

use bevy_render::render_resource::encase::{internal::WriteInto, ShaderType};

use super::allocator::{SparseBuffer, StableAddr};
use super::persistent_buffer::{PersistentGpuBuffer, PersistentGpuBufferable};
use super::stable_storage_buffer::StableStorageBuffer;

mod sealed {
    pub trait Sealed {}
}

/// A buffer with a **stable, never-freed** device address — safe to capture for the
/// raw RT trace. See the module docs.
pub trait RawTraceBindable: sealed::Sealed {
    /// The buffer's base device address. Stable for the buffer's lifetime, so a
    /// captured copy stays valid for any in-flight trace that recorded it.
    fn trace_device_address(&self) -> StableAddr;
}

impl sealed::Sealed for SparseBuffer {}
impl RawTraceBindable for SparseBuffer {
    #[inline]
    fn trace_device_address(&self) -> StableAddr {
        self.stable_addr()
    }
}

impl<V: ShaderType + WriteInto> sealed::Sealed for StableStorageBuffer<V> {}
impl<V: ShaderType + WriteInto> RawTraceBindable for StableStorageBuffer<V> {
    #[inline]
    fn trace_device_address(&self) -> StableAddr {
        self.device_address()
    }
}

impl<T: PersistentGpuBufferable> sealed::Sealed for PersistentGpuBuffer<T> {}
impl<T: PersistentGpuBufferable> RawTraceBindable for PersistentGpuBuffer<T> {
    #[inline]
    fn trace_device_address(&self) -> StableAddr {
        self.device_address()
    }
}
