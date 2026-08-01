//! The binding seam (`moonshot_plan.md`): every descriptor-heap, device-address,
//! and SBT-record interaction in solari goes through THREE entry points —
//! [`BindingSeam::alloc_heap_index`], [`BindingSeam::device_address`], and
//! [`BindingSeam::write_record`]. Shaders keep plain `[[vk::binding]]`
//! declarations; descriptors are sourced from the heap host-side via
//! `VkShaderDescriptorSetAndBindingMappingInfoEXT` (see
//! [`BindingSeam::map_binding`]), so shader source never re-targets when the
//! binding model changes. Everything volatile — the mapping-source union arms,
//! per-type descriptor strides, reserved ranges, heap bind commands — lives in
//! this one file; when the KHR descriptor heap lands, this file changes and
//! nothing else does.
//!
//! Proven by `examples/3d/solari/heap_probe.rs --mode mapped` (M0):
//! - mapped bindings compile to CLASSIC descriptor SPIR-V, so coopvec ops work
//!   (the `.Handle()` untyped-pointer path hangs the driver — never use it);
//! - mapping offsets are BYTES into the heap (per-type descriptor sizes are a
//!   host-side concern — this module's concern);
//! - several bindings may alias one heap descriptor (e.g. float/half/BAB views
//!   of one buffer);
//! - a classic `[[vk::push_constant]]` block is fed by `vkCmdPushDataEXT`.

#![allow(unsafe_code)]

use super::allocator::{Allocator, SubmitAddr};
use ash::vk::{self, TaggedStructure};
use bevy_ecs::resource::Resource;
use std::sync::{Arc, Mutex};

/// Descriptor capacity of each per-type heap region. Fixed for now; growing a
/// region means a new heap + rewriting live descriptors, which the diff-driven
/// column pipeline can do once a consumer actually overflows these.
const BUFFER_SLOTS: u64 = 4096;
const IMAGE_SLOTS: u64 = 2048;
const SAMPLER_SLOTS: u64 = 256;

/// SBT records: one per material slot, hit-group handle followed by the
/// material's pointers / heap indices / constants.
const MAX_RECORDS: u64 = 4096;
/// Bytes of per-record data after the hit-group handle.
const RECORD_DATA_SIZE: u64 = 96;

/// Which per-type heap region a slot lives in. Descriptor sizes differ per
/// type (16/32/32 B on the 5090), so indices are only meaningful within their
/// region — the seam converts region-local indices to byte offsets when it
/// builds mappings.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeapKind {
    Buffer,
    Image,
    Sampler,
}

/// A resource to allocate a heap descriptor for. Image and sampler
/// descriptors are written from CREATE-INFO structs — the heap extension
/// never references live `VkImageView`/`VkSampler` objects.
pub enum HeapResource<'a> {
    /// `STORAGE_BUFFER` descriptor over `[address, address + size)`.
    Buffer { address: u64, size: u64 },
    /// `UNIFORM_BUFFER` descriptor over `[address, address + size)`.
    UniformBuffer { address: u64, size: u64 },
    /// `SAMPLED_IMAGE` descriptor.
    SampledImage {
        view: &'a vk::ImageViewCreateInfo<'a>,
        layout: vk::ImageLayout,
    },
    /// `STORAGE_IMAGE` descriptor.
    StorageImage {
        view: &'a vk::ImageViewCreateInfo<'a>,
        layout: vk::ImageLayout,
    },
    /// Sampler descriptor (lives in the SAMPLER heap, bound separately).
    Sampler(&'a vk::SamplerCreateInfo<'a>),
}

impl HeapResource<'_> {
    fn kind(&self) -> HeapKind {
        match self {
            HeapResource::Buffer { .. } | HeapResource::UniformBuffer { .. } => HeapKind::Buffer,
            HeapResource::SampledImage { .. } | HeapResource::StorageImage { .. } => {
                HeapKind::Image
            }
            HeapResource::Sampler(_) => HeapKind::Sampler,
        }
    }
}

/// One raw host-visible heap buffer with a persistent map.
struct RawHeap {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    host_ptr: *mut u8,
    address: u64,
    size: u64,
    reserved_offset: u64,
    reserved_size: u64,
}

// SAFETY: host_ptr is a persistently-mapped HOST_COHERENT allocation owned by
// the seam; all writes go through &self methods serialized by the free-list
// Mutex where they must be.
unsafe impl Send for RawHeap {}
// SAFETY: see the Send impl above.
unsafe impl Sync for RawHeap {}

/// Region-local free lists, one per [`HeapKind`], plus the record table's.
#[derive(Default)]
struct FreeLists {
    buffer: FreeList,
    image: FreeList,
    sampler: FreeList,
}

#[derive(Default)]
struct FreeList {
    next: u32,
    free: Vec<u32>,
    capacity: u32,
}

impl FreeList {
    fn alloc(&mut self, kind: HeapKind) -> u32 {
        if let Some(idx) = self.free.pop() {
            return idx;
        }
        assert!(
            self.next < self.capacity,
            "binding_seam: {kind:?} heap region full ({} slots)",
            self.capacity
        );
        let idx = self.next;
        self.next += 1;
        idx
    }
}

#[derive(Resource, Clone)]
pub struct BindingSeam {
    inner: Arc<SeamInner>,
}

struct SeamInner {
    allocator: Allocator,
    heap_fns: ash::ext::descriptor_heap::Device,
    /// Per-type descriptor sizes/alignments from
    /// `VkPhysicalDeviceDescriptorHeapPropertiesEXT`.
    buffer_desc_size: u64,
    image_desc_size: u64,
    sampler_desc_size: u64,
    /// Byte offset of the image region inside the resource heap (the buffer
    /// region starts at 0).
    image_region_offset: u64,
    resource_heap: RawHeap,
    sampler_heap: RawHeap,
    /// SBT record table: `MAX_RECORDS` records of `record_stride` bytes each,
    /// hit-group handle first. Host-visible for now; the diff-driven column
    /// pipeline takes over writes when materials go GPU-authored.
    records: RawHeap,
    record_stride: u64,
    handle_size: u64,
    free: Mutex<FreeLists>,
}

impl BindingSeam {
    /// Build the heaps and the record table. Returns `None` when the device
    /// doesn't expose `VK_EXT_descriptor_heap` (the fork's hal enables it
    /// automatically when supported).
    pub fn try_new(allocator: &Allocator) -> Option<Self> {
        let instance = allocator.instance();
        let device = allocator.device();
        let phys = allocator.physical_device();

        let mut heap_props = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
        let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push(&mut heap_props)
            .push(&mut rt_props);
        // SAFETY: phys is valid; the chain structs are defaulted + stack-owned.
        unsafe { instance.get_physical_device_properties2(phys, &mut props2) };
        // An unsupported chained struct is left untouched by the driver, so a
        // zero descriptor size is the "extension absent" signal.
        if heap_props.buffer_descriptor_size == 0 {
            return None;
        }

        let heap_fns = ash::ext::descriptor_heap::Device::load(instance, device);

        // Resource heap: [buffer region][image region][reserved range].
        let image_region_offset = align_up(
            BUFFER_SLOTS * heap_props.buffer_descriptor_size,
            heap_props.image_descriptor_alignment.max(1),
        );
        let resource_descriptors_end =
            image_region_offset + IMAGE_SLOTS * heap_props.image_descriptor_size;
        let resource_heap = RawHeap::new(
            allocator,
            vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT,
            resource_descriptors_end,
            heap_props.resource_heap_alignment.max(1),
            heap_props.min_resource_heap_reserved_range,
        );
        // Sampler heap: [sampler region][reserved range].
        let sampler_heap = RawHeap::new(
            allocator,
            vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT,
            SAMPLER_SLOTS * heap_props.sampler_descriptor_size,
            heap_props.sampler_heap_alignment.max(1),
            heap_props.min_sampler_heap_reserved_range,
        );

        // Record table. The stride must be a multiple of the hit-group handle
        // alignment; the table base (a dedicated allocation) satisfies the
        // 64 B base alignment on its own.
        let handle_size = rt_props.shader_group_handle_size as u64;
        let record_stride = align_up(
            handle_size + RECORD_DATA_SIZE,
            (rt_props.shader_group_handle_alignment as u64).max(1),
        );
        let records = RawHeap::new(
            allocator,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR | vk::BufferUsageFlags::TRANSFER_DST,
            MAX_RECORDS * record_stride,
            (rt_props.shader_group_base_alignment as u64).max(1),
            0,
        );

        Some(Self {
            inner: Arc::new(SeamInner {
                allocator: allocator.clone(),
                heap_fns,
                buffer_desc_size: heap_props.buffer_descriptor_size,
                image_desc_size: heap_props.image_descriptor_size,
                sampler_desc_size: heap_props.sampler_descriptor_size,
                image_region_offset,
                resource_heap,
                sampler_heap,
                records,
                record_stride,
                handle_size,
                free: Mutex::new(FreeLists {
                    buffer: FreeList {
                        capacity: BUFFER_SLOTS as u32,
                        ..Default::default()
                    },
                    image: FreeList {
                        capacity: IMAGE_SLOTS as u32,
                        ..Default::default()
                    },
                    sampler: FreeList {
                        capacity: SAMPLER_SLOTS as u32,
                        ..Default::default()
                    },
                }),
            }),
        })
    }

    /// Write a descriptor for `resource` into its per-type heap region and
    /// return the region-local index (the value shaders receive in push data /
    /// records, and [`map_binding`](Self::map_binding) turns into a byte
    /// offset).
    pub fn alloc_heap_index(&self, resource: HeapResource) -> u32 {
        let kind = resource.kind();
        let index = {
            let mut free = self.inner.free.lock().unwrap();
            match kind {
                HeapKind::Buffer => free.buffer.alloc(kind),
                HeapKind::Image => free.image.alloc(kind),
                HeapKind::Sampler => free.sampler.alloc(kind),
            }
        };
        self.write_descriptor(&resource, index);
        index
    }

    /// Rewrite the descriptor at an existing slot (resource replaced in
    /// place — the index, and everything referencing it, stays valid).
    /// `kind` must match the original allocation's.
    pub fn rewrite_heap_index(&self, kind: HeapKind, index: u32, resource: HeapResource) {
        assert_eq!(resource.kind(), kind, "binding_seam: heap-kind mismatch");
        self.write_descriptor(&resource, index);
    }

    /// Return a slot to its region's free list. The caller owns the hazard:
    /// no in-flight trace/dispatch may still read the descriptor.
    pub fn free_heap_index(&self, kind: HeapKind, index: u32) {
        let mut free = self.inner.free.lock().unwrap();
        match kind {
            HeapKind::Buffer => free.buffer.free.push(index),
            HeapKind::Image => free.image.free.push(index),
            HeapKind::Sampler => free.sampler.free.push(index),
        }
    }

    /// Device address of a wgpu buffer (the fork gives every storage buffer
    /// `SHADER_DEVICE_ADDRESS`). Same submit-scoped semantics as
    /// [`Allocator::wgpu_buffer_device_address`].
    pub fn device_address(&self, buffer: &wgpu::Buffer) -> SubmitAddr {
        self.inner.allocator.wgpu_buffer_device_address(buffer)
    }

    /// Device address of a raw `VkBuffer` (created with
    /// `SHADER_DEVICE_ADDRESS`, as every solari buffer is), for
    /// [`HeapResource::Buffer`]/[`HeapResource::UniformBuffer`] descriptors.
    pub fn raw_buffer_address(&self, buffer: vk::Buffer) -> u64 {
        // SAFETY: the handle comes from a live solari-owned buffer created
        // with SHADER_DEVICE_ADDRESS usage.
        unsafe {
            self.inner
                .allocator
                .device()
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        }
    }

    /// Write one SBT record: the hit-group `handle`, then `fields` (the
    /// material's pointers / heap indices / constants).
    pub fn write_record(&self, slot: u32, handle: &[u8], fields: &[u8]) {
        let inner = &self.inner;
        assert!((slot as u64) < MAX_RECORDS, "binding_seam: record slot {slot} out of range");
        assert_eq!(handle.len() as u64, inner.handle_size);
        assert!(
            fields.len() as u64 <= inner.record_stride - inner.handle_size,
            "binding_seam: record fields ({} B) exceed stride budget ({} B)",
            fields.len(),
            inner.record_stride - inner.handle_size,
        );
        let base = slot as u64 * inner.record_stride;
        // SAFETY: base + handle + fields stays inside the records allocation
        // (asserted above); the map is persistent and HOST_COHERENT.
        unsafe {
            let dst = inner.records.host_ptr.add(base as usize);
            std::ptr::copy_nonoverlapping(handle.as_ptr(), dst, handle.len());
            std::ptr::copy_nonoverlapping(
                fields.as_ptr(),
                dst.add(handle.len()),
                fields.len(),
            );
        }
    }

    /// The record table as an SBT region for `vkCmdTraceRaysKHR`'s hit-group
    /// table, covering `count` records from `first_slot`.
    pub fn record_region(&self, first_slot: u32, count: u32) -> vk::StridedDeviceAddressRegionKHR {
        let inner = &self.inner;
        vk::StridedDeviceAddressRegionKHR {
            device_address: inner.records.address + first_slot as u64 * inner.record_stride,
            stride: inner.record_stride,
            size: count as u64 * inner.record_stride,
        }
    }

    /// Bytes available for `fields` in each record.
    pub fn record_data_capacity(&self) -> u64 {
        self.inner.record_stride - self.inner.handle_size
    }

    /// Mapping for one classic `[[vk::binding(binding, set)]]` declaration:
    /// descriptor sourced from the heap at the given region-local index
    /// (`HEAP_WITH_CONSTANT_OFFSET` — the offset is BYTES, computed here).
    /// Chain the returned mappings onto the stage create info via
    /// `ShaderDescriptorSetAndBindingMappingInfoEXT`; the pipeline itself
    /// needs `PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT` and no layout.
    pub fn map_binding(
        &self,
        set: u32,
        binding: u32,
        kind: HeapKind,
        index: u32,
    ) -> vk::DescriptorSetAndBindingMappingEXT<'static> {
        let (byte_offset, mask) = match kind {
            HeapKind::Buffer => (
                index as u64 * self.inner.buffer_desc_size,
                vk::SpirvResourceTypeFlagsEXT::READ_ONLY_STORAGE_BUFFER
                    | vk::SpirvResourceTypeFlagsEXT::READ_WRITE_STORAGE_BUFFER
                    | vk::SpirvResourceTypeFlagsEXT::UNIFORM_BUFFER,
            ),
            HeapKind::Image => (
                self.inner.image_region_offset + index as u64 * self.inner.image_desc_size,
                vk::SpirvResourceTypeFlagsEXT::SAMPLED_IMAGE
                    | vk::SpirvResourceTypeFlagsEXT::READ_ONLY_IMAGE
                    | vk::SpirvResourceTypeFlagsEXT::READ_WRITE_IMAGE,
            ),
            HeapKind::Sampler => (
                index as u64 * self.inner.sampler_desc_size,
                vk::SpirvResourceTypeFlagsEXT::SAMPLER,
            ),
        };
        vk::DescriptorSetAndBindingMappingEXT::default()
            .descriptor_set(set)
            .first_binding(binding)
            .binding_count(1)
            .resource_mask(mask)
            .source(vk::DescriptorMappingSourceEXT::HEAP_WITH_CONSTANT_OFFSET)
            .source_data(vk::DescriptorMappingSourceDataEXT {
                constant_offset: vk::DescriptorMappingSourceConstantOffsetEXT::default()
                    .heap_offset(byte_offset as u32),
            })
    }

    /// Bind both heaps on a raw command buffer. Once per command buffer,
    /// before any heap-flagged pipeline dispatch/trace.
    ///
    /// # Safety
    /// `cb` must be in the recording state on a queue owned by this device.
    pub unsafe fn bind_heaps(&self, cb: vk::CommandBuffer) {
        let inner = &self.inner;
        let bind_info = |heap: &RawHeap| {
            let mut bind = vk::BindHeapInfoEXT::default();
            bind.heap_range = vk::DeviceAddressRangeEXT {
                address: heap.address,
                size: heap.size,
            };
            bind.reserved_range_offset = heap.reserved_offset;
            bind.reserved_range_size = heap.reserved_size;
            bind
        };
        // SAFETY: per the function contract; the heap buffers outlive the
        // command buffer (the seam is app-lifetime).
        unsafe {
            inner
                .heap_fns
                .cmd_bind_resource_heap(cb, &bind_info(&inner.resource_heap));
            inner
                .heap_fns
                .cmd_bind_sampler_heap(cb, &bind_info(&inner.sampler_heap));
        }
    }

    /// Push `data` for heap pipelines (replaces push constants; classic
    /// `[[vk::push_constant]]` blocks read from this range).
    ///
    /// # Safety
    /// `cb` must be in the recording state; `data` must match the largest
    /// push block of every pipeline dispatched under it.
    pub unsafe fn push_data(&self, cb: vk::CommandBuffer, data: &[u8]) {
        let mut range = vk::HostAddressRangeConstEXT::default();
        range.address = data.as_ptr().cast();
        range.size = data.len();
        let mut info = vk::PushDataInfoEXT::default();
        info.offset = 0;
        info.data = range;
        // SAFETY: per the function contract; the data is copied by the call.
        unsafe { self.inner.heap_fns.cmd_push_data(cb, &info) };
    }

    fn write_descriptor(&self, resource: &HeapResource, index: u32) {
        let inner = &self.inner;
        let (heap, byte_offset, desc_size) = match resource.kind() {
            HeapKind::Buffer => (
                &inner.resource_heap,
                index as u64 * inner.buffer_desc_size,
                inner.buffer_desc_size,
            ),
            HeapKind::Image => (
                &inner.resource_heap,
                inner.image_region_offset + index as u64 * inner.image_desc_size,
                inner.image_desc_size,
            ),
            HeapKind::Sampler => (
                &inner.sampler_heap,
                index as u64 * inner.sampler_desc_size,
                inner.sampler_desc_size,
            ),
        };
        let mut dst = vk::HostAddressRangeEXT::default();
        // SAFETY: index was allocated from (or asserted against) the region's
        // capacity, so the write stays inside the mapped heap.
        dst.address = unsafe { heap.host_ptr.add(byte_offset as usize).cast() };
        dst.size = desc_size as usize;

        match resource {
            HeapResource::Buffer { address, size }
            | HeapResource::UniformBuffer { address, size } => {
                let range = vk::DeviceAddressRangeEXT {
                    address: *address,
                    size: *size,
                };
                let ty = if matches!(resource, HeapResource::UniformBuffer { .. }) {
                    vk::DescriptorType::UNIFORM_BUFFER
                } else {
                    vk::DescriptorType::STORAGE_BUFFER
                };
                let info = vk::ResourceDescriptorInfoEXT {
                    ty,
                    data: vk::ResourceDescriptorDataEXT {
                        p_address_range: &range,
                    },
                    ..Default::default()
                };
                // SAFETY: info + dst are fully populated and in scope.
                unsafe { inner.heap_fns.write_resource_descriptors(&[info], &[dst]) }
                    .expect("binding_seam: write buffer descriptor");
            }
            HeapResource::SampledImage { view, layout }
            | HeapResource::StorageImage { view, layout } => {
                let ty = if matches!(resource, HeapResource::StorageImage { .. }) {
                    vk::DescriptorType::STORAGE_IMAGE
                } else {
                    vk::DescriptorType::SAMPLED_IMAGE
                };
                let image_info = vk::ImageDescriptorInfoEXT {
                    p_view: *view,
                    layout: *layout,
                    ..Default::default()
                };
                let info = vk::ResourceDescriptorInfoEXT {
                    ty,
                    data: vk::ResourceDescriptorDataEXT {
                        p_image: &image_info,
                    },
                    ..Default::default()
                };
                // SAFETY: info + dst are fully populated and in scope.
                unsafe { inner.heap_fns.write_resource_descriptors(&[info], &[dst]) }
                    .expect("binding_seam: write image descriptor");
            }
            HeapResource::Sampler(create_info) => {
                // SAFETY: create_info + dst are fully populated and in scope.
                unsafe {
                    inner
                        .heap_fns
                        .write_sampler_descriptors(&[**create_info], &[dst])
                }
                .expect("binding_seam: write sampler descriptor");
            }
        }
    }
}

impl RawHeap {
    /// Dedicated host-visible allocation: `descriptor_bytes` of descriptor
    /// space, then the reserved range (aligned to `heap_alignment`),
    /// persistently mapped.
    fn new(
        allocator: &Allocator,
        usage: vk::BufferUsageFlags,
        descriptor_bytes: u64,
        heap_alignment: u64,
        reserved_size: u64,
    ) -> RawHeap {
        let device = allocator.device();
        let reserved_offset = align_up(descriptor_bytes.max(4), heap_alignment);
        let size = align_up(reserved_offset + reserved_size, heap_alignment);

        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: create_info is fully populated; no extension chain.
        let buffer = unsafe { device.create_buffer(&create_info, None) }
            .expect("binding_seam: vkCreateBuffer");
        // SAFETY: buffer is a valid handle from create_buffer above.
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let type_index = allocator
            .find_memory_type(
                reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .expect("binding_seam: no host-visible memory type");
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        // SAFETY: allocation info is fully populated; DEVICE_ADDRESS matches
        // the buffer's SHADER_DEVICE_ADDRESS usage (VUID 09127).
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(type_index)
                    .push(&mut flags),
                None,
            )
        }
        .expect("binding_seam: vkAllocateMemory");
        // SAFETY: fresh buffer + memory from the same device; offset 0.
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .expect("binding_seam: vkBindBufferMemory");
        // SAFETY: memory is host-visible and unmapped.
        let host_ptr = unsafe {
            device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .expect("binding_seam: vkMapMemory")
        .cast::<u8>();
        // SAFETY: buffer was created with SHADER_DEVICE_ADDRESS.
        let address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };

        RawHeap {
            buffer,
            memory,
            host_ptr,
            address,
            size,
            reserved_offset,
            reserved_size,
        }
    }
}

impl Drop for SeamInner {
    fn drop(&mut self) {
        // Raw-Drop discipline: drain the queue before raw destroys — the last
        // frame's submissions may still reference the heaps.
        self.allocator.quiesce_before_raw_destroy();
        let device = self.allocator.device();
        for heap in [&self.resource_heap, &self.sampler_heap, &self.records] {
            // SAFETY: quiesced above; handles are exclusively owned here.
            unsafe {
                device.destroy_buffer(heap.buffer, None);
                device.free_memory(heap.memory, None);
            }
        }
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}
