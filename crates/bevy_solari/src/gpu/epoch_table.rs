//! Slot-indexed epoch stamps — a GPU-visible "already touched this
//! build/frame" set, the shared shape behind seed-dedup, delta-clears, and
//! reserve-vs-fill gating.
//!
//! Protocol: the CPU calls [`EpochTable::begin`] once per cycle; a producer
//! pass stamps `table[slot] = epoch`; any later pass tests membership with
//! `table[slot] == epoch`. Zero is never a valid epoch, so a fresh (zeroed)
//! buffer is "empty" — growth recreates fresh, and `begin` reports it so the
//! caller can pair growth with whatever full refresh its protocol requires.

use bevy_render::render_resource::Buffer;
use bevy_render::renderer::RenderDevice;

pub struct EpochTable {
    buffer: Buffer,
    capacity: u32,
    epoch: u32,
    label: &'static str,
}

impl EpochTable {
    pub fn new(render_device: &RenderDevice, label: &'static str) -> Self {
        Self {
            buffer: Self::alloc(render_device, label, 1),
            capacity: 1,
            epoch: 0,
            label,
        }
    }

    fn alloc(render_device: &RenderDevice, label: &'static str, slots: u32) -> Buffer {
        render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: slots.max(1) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    /// Start a new epoch over `capacity` slots; returns `(epoch, grew)`.
    /// Epochs are never 0 (wrapping skips it), so stale/fresh entries never
    /// match. A grown table is fresh — all prior stamps are forgotten.
    pub fn begin(&mut self, render_device: &RenderDevice, capacity: u32) -> (u32, bool) {
        let mut grew = false;
        if capacity > self.capacity {
            self.buffer = Self::alloc(render_device, self.label, capacity);
            self.capacity = capacity;
            grew = true;
        }
        self.epoch = self.epoch.wrapping_add(1).max(1);
        (self.epoch, grew)
    }

    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// The current epoch — what the stamping pass writes and testers compare.
    #[inline]
    pub fn epoch(&self) -> u32 {
        self.epoch
    }
}
