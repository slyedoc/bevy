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
/// column pipeline can do once a consumer actually overflows these. The image
/// and sampler regions are sized for the scene's bindless arrays (a contiguous
/// [`BindingSeam::alloc_heap_block`] each, `MAX_TEXTURE_COUNT` entries) plus
/// singles and slack; each is clamped against the device's max heap size.
const BUFFER_SLOTS: u64 = 4096;
const IMAGE_SLOTS: u64 = 8192;
const SAMPLER_SLOTS: u64 = 5120;

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
        self.alloc_block(kind, 1)
    }

    /// Bump-allocate `count` CONTIGUOUS slots. A multi-slot block backs a
    /// bindless array and is app-lifetime (there is no block free); a
    /// single-slot block is an ordinary slot and may go back through
    /// [`free`](FreeList::free).
    fn alloc_block(&mut self, kind: HeapKind, count: u32) -> u32 {
        assert!(
            self.next + count <= self.capacity,
            "binding_seam: {kind:?} heap region full ({} of {} slots used, {count} requested)",
            self.next,
            self.capacity
        );
        let base = self.next;
        self.next += count;
        base
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

        // Region capacities, clamped so each heap (descriptors + reserved
        // range) fits the device's max heap size. A clamp below what the
        // scene's bindless arrays need surfaces as a region-full panic with
        // the numbers logged here.
        let image_slots = IMAGE_SLOTS.min(
            heap_props
                .max_resource_heap_size
                .saturating_sub(
                    heap_props.min_resource_heap_reserved_range
                        + BUFFER_SLOTS * heap_props.buffer_descriptor_size,
                )
                / heap_props.image_descriptor_size,
        );
        let sampler_slots = SAMPLER_SLOTS.min(
            heap_props
                .max_sampler_heap_size
                .saturating_sub(heap_props.min_sampler_heap_reserved_range)
                / heap_props.sampler_descriptor_size,
        );
        bevy_log::info!(
            "binding_seam: descriptor sizes buffer/image/sampler = {}/{}/{} B, \
             max heap sizes resource/sampler = {}/{} B, regions = {BUFFER_SLOTS} buffers \
             + {image_slots} images + {sampler_slots} samplers",
            heap_props.buffer_descriptor_size,
            heap_props.image_descriptor_size,
            heap_props.sampler_descriptor_size,
            heap_props.max_resource_heap_size,
            heap_props.max_sampler_heap_size,
        );

        // Resource heap: [buffer region][image region][reserved range].
        let image_region_offset = align_up(
            BUFFER_SLOTS * heap_props.buffer_descriptor_size,
            heap_props.image_descriptor_alignment.max(1),
        );
        let resource_descriptors_end =
            image_region_offset + image_slots * heap_props.image_descriptor_size;
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
            sampler_slots * heap_props.sampler_descriptor_size,
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
                        capacity: image_slots as u32,
                        ..Default::default()
                    },
                    sampler: FreeList {
                        capacity: sampler_slots as u32,
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

    /// Total slot capacity of `kind`'s region — the [`SAMPLER_SLOTS`]-style
    /// constants clamped to the device's max heap size (NVIDIA caps the
    /// sampler heap at 128 KB ⇒ 4096 slots, well under the buffer/image
    /// regions). Block sizing for bindless arrays consults this.
    pub fn region_capacity(&self, kind: HeapKind) -> u32 {
        let free = self.inner.free.lock().unwrap();
        match kind {
            HeapKind::Buffer => free.buffer.capacity,
            HeapKind::Image => free.image.capacity,
            HeapKind::Sampler => free.sampler.capacity,
        }
    }

    /// Reserve `count` CONTIGUOUS slots in `kind`'s region and return the base
    /// index, without writing any descriptors — a bindless array's backing
    /// (`binding_array<..., N>` maps to `N` consecutive heap descriptors).
    /// Elements are written via [`rewrite_heap_index`](Self::rewrite_heap_index)
    /// at `base + i`. A multi-slot block is app-lifetime (no block free); a
    /// single-slot block is an ordinary slot —
    /// [`free_heap_index`](Self::free_heap_index) works on it.
    pub fn alloc_heap_block(&self, kind: HeapKind, count: u32) -> u32 {
        let mut free = self.inner.free.lock().unwrap();
        match kind {
            HeapKind::Buffer => free.buffer.alloc_block(kind, count),
            HeapKind::Image => free.image.alloc_block(kind, count),
            HeapKind::Sampler => free.sampler.alloc_block(kind, count),
        }
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

    /// The SPIR-V resource types a mapping for `kind` covers (broad on
    /// purpose: one builder serves storage + uniform buffers, sampled +
    /// storage images).
    fn resource_mask(kind: HeapKind) -> vk::SpirvResourceTypeFlagsEXT {
        match kind {
            HeapKind::Buffer => {
                vk::SpirvResourceTypeFlagsEXT::READ_ONLY_STORAGE_BUFFER
                    | vk::SpirvResourceTypeFlagsEXT::READ_WRITE_STORAGE_BUFFER
                    | vk::SpirvResourceTypeFlagsEXT::UNIFORM_BUFFER
            }
            HeapKind::Image => {
                vk::SpirvResourceTypeFlagsEXT::SAMPLED_IMAGE
                    | vk::SpirvResourceTypeFlagsEXT::READ_ONLY_IMAGE
                    | vk::SpirvResourceTypeFlagsEXT::READ_WRITE_IMAGE
            }
            HeapKind::Sampler => vk::SpirvResourceTypeFlagsEXT::SAMPLER,
        }
    }

    fn map_entry(
        set: u32,
        binding: u32,
        mask: vk::SpirvResourceTypeFlagsEXT,
        source: vk::DescriptorMappingSourceEXT,
        source_data: vk::DescriptorMappingSourceDataEXT<'static>,
    ) -> vk::DescriptorSetAndBindingMappingEXT<'static> {
        vk::DescriptorSetAndBindingMappingEXT::default()
            .descriptor_set(set)
            .first_binding(binding)
            .binding_count(1)
            .resource_mask(mask)
            .source(source)
            .source_data(source_data)
    }

    /// Mapping for one classic `[[vk::binding(binding, set)]]` declaration:
    /// descriptor sourced from the heap at the given region-local index
    /// (`HEAP_WITH_CONSTANT_OFFSET` — the offset is BYTES, computed here).
    /// The array stride is always set, so an arrayed binding (`binding_array`)
    /// reads consecutive descriptors from `index` — pass a
    /// [`alloc_heap_block`](Self::alloc_heap_block) base for those.
    /// Chain the returned mappings onto each stage create info via
    /// `ShaderDescriptorSetAndBindingMappingInfoEXT`; the pipeline itself
    /// needs `PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT` and no layout.
    pub fn map_binding(
        &self,
        set: u32,
        binding: u32,
        kind: HeapKind,
        index: u32,
    ) -> vk::DescriptorSetAndBindingMappingEXT<'static> {
        let inner = &self.inner;
        // Samplers live in the separately-bound sampler heap; their mapping
        // fields are the `sampler_*` half of the source union.
        let source_data = match kind {
            HeapKind::Buffer => vk::DescriptorMappingSourceDataEXT {
                constant_offset: vk::DescriptorMappingSourceConstantOffsetEXT {
                    heap_offset: (index as u64 * inner.buffer_desc_size) as u32,
                    heap_array_stride: inner.buffer_desc_size as u32,
                    ..Default::default()
                },
            },
            HeapKind::Image => vk::DescriptorMappingSourceDataEXT {
                constant_offset: vk::DescriptorMappingSourceConstantOffsetEXT {
                    heap_offset: (inner.image_region_offset
                        + index as u64 * inner.image_desc_size)
                        as u32,
                    heap_array_stride: inner.image_desc_size as u32,
                    ..Default::default()
                },
            },
            HeapKind::Sampler => vk::DescriptorMappingSourceDataEXT {
                constant_offset: vk::DescriptorMappingSourceConstantOffsetEXT {
                    sampler_heap_offset: (index as u64 * inner.sampler_desc_size) as u32,
                    sampler_heap_array_stride: inner.sampler_desc_size as u32,
                    ..Default::default()
                },
            },
        };
        Self::map_entry(
            set,
            binding,
            Self::resource_mask(kind),
            vk::DescriptorMappingSourceEXT::HEAP_WITH_CONSTANT_OFFSET,
            source_data,
        )
    }

    /// Mapping whose heap INDEX is read from push data at `push_offset`
    /// (`HEAP_WITH_PUSH_INDEX`): the descriptor lives at
    /// `region base + pushed_index × descriptor size`. Per-view resources use
    /// this so one linked pipeline serves every view — the trace pushes the
    /// view's region-local slot indices.
    pub fn map_binding_push_index(
        &self,
        set: u32,
        binding: u32,
        kind: HeapKind,
        push_offset: u32,
    ) -> vk::DescriptorSetAndBindingMappingEXT<'static> {
        let inner = &self.inner;
        let source_data = match kind {
            HeapKind::Buffer => vk::DescriptorMappingSourceDataEXT {
                push_index: vk::DescriptorMappingSourcePushIndexEXT {
                    heap_offset: 0,
                    push_offset,
                    heap_index_stride: inner.buffer_desc_size as u32,
                    heap_array_stride: inner.buffer_desc_size as u32,
                    ..Default::default()
                },
            },
            HeapKind::Image => vk::DescriptorMappingSourceDataEXT {
                push_index: vk::DescriptorMappingSourcePushIndexEXT {
                    heap_offset: inner.image_region_offset as u32,
                    push_offset,
                    heap_index_stride: inner.image_desc_size as u32,
                    heap_array_stride: inner.image_desc_size as u32,
                    ..Default::default()
                },
            },
            HeapKind::Sampler => vk::DescriptorMappingSourceDataEXT {
                push_index: vk::DescriptorMappingSourcePushIndexEXT {
                    sampler_heap_offset: 0,
                    sampler_push_offset: push_offset,
                    sampler_heap_index_stride: inner.sampler_desc_size as u32,
                    sampler_heap_array_stride: inner.sampler_desc_size as u32,
                    ..Default::default()
                },
            },
        };
        Self::map_entry(
            set,
            binding,
            Self::resource_mask(kind),
            vk::DescriptorMappingSourceEXT::HEAP_WITH_PUSH_INDEX,
            source_data,
        )
    }

    /// Acceleration-structure mapping sourced from a DEVICE ADDRESS in push
    /// data at `push_offset` (`PUSH_ADDRESS`). The TLAS must come this way:
    /// shader-side heap AS access device-losts on current NVIDIA drivers,
    /// while address sourcing is solid.
    pub fn map_binding_push_address(
        &self,
        set: u32,
        binding: u32,
        push_offset: u32,
    ) -> vk::DescriptorSetAndBindingMappingEXT<'static> {
        Self::map_entry(
            set,
            binding,
            vk::SpirvResourceTypeFlagsEXT::ACCELERATION_STRUCTURE,
            vk::DescriptorMappingSourceEXT::PUSH_ADDRESS,
            vk::DescriptorMappingSourceDataEXT {
                push_address_offset: push_offset,
            },
        )
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
