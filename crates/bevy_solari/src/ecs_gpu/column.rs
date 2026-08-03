//! `GpuColumn<C>` — one persistent, slot-indexed GPU buffer per
//! [`GpuColumnDesc`], the GPU-side "component" storage of a [`GpuEntity`].
//!
//! A column is described once (its value type, where the value comes from,
//! which entities changed this frame, and whether it keeps a previous-frame
//! copy); [`GpuColumnPlugin<C>`] turns that into buffers + init / prepare /
//! dispatch systems, all generic over `C`. Each column is its own resource,
//! so Bevy runs the per-column prepares in parallel.
//!
//! Updates are scattered: only the entities that changed cross the bus as a
//! compact `(slot, value)` delta, and the byte-wise scatter kernel
//! (`gpu_instances_scatter.slang`) places them. Every column shares the same
//! two heap pipelines ([`ColumnScatterKernels`]) — the buffers are
//! per-dispatch heap slots, so nothing per-column is compiled. A
//! `KEEP_PREVIOUS` column also shifts the current value into a previous-frame
//! buffer on the GPU (the GPU already holds last frame's value — no second
//! upload).

#![allow(unsafe_code)]

use core::marker::PhantomData;
use core::num::NonZero;

use bevy_app::{App, Plugin};
use bevy_ecs::{
    resource::Resource,
    schedule::{IntoScheduleConfigs, SystemSet},
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    render_resource::{Buffer, BufferUsages, RawBufferVec},
    renderer::{RenderContext, RenderDevice, RenderQueue},
    Render, RenderApp, RenderStartup, RenderSystems,
};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use core::ops::Range;
use wgpu::hal::api::Vulkan as VkApi;

use crate::gpu::allocator::{Allocator, SparseBuffer};
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
use crate::{SolariClusterSystems, SolariSetup};

/// Workgroup size of the scatter shader (`@workgroup_size(64)`).
const SCATTER_WORKGROUP_SIZE: u32 = 64;

/// Virtual address space reserved per column's sparse store (and its previous-
/// frame buffer). Physical pages are committed lazily on growth; only this much
/// *virtual* range is reserved up front, so the buffer handle/address — and thus
/// every consumer bind group and recorded device address — stays stable across
/// growth. 1 GiB covers ~22M slots even at the 48-byte `Affine3x4` stride.
/// Matches the AS-side `*_VIRTUAL_BYTES` convention (`accel`/`geometry`).
const COLUMN_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;

/// Push params shared with `gpu_instances_scatter.slang::ScatterParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct ScatterParams {
    count: u32,
    words_per_value: u32,
    /// X workgroup count of the 2D-split dispatch (flat-index reconstruction).
    groups_x: u32,
    _pad: u32,
}

/// All `GpuColumn` prepare systems run in this set, so callers can order
/// producers (e.g. `resolve_instance_material_ids`, which writes the values a
/// column reads) before every column with one edge.
#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct GpuColumnPrepareSet;

/// A GPU-mirrored table: a slot space whose columns scatter per-frame deltas.
/// `InstanceManager` (per-instance columns) and `TransformGraph` (transform-table
/// columns) are tables. `high_water` is a property of the *table* — shared by all
/// its columns — so it lives here, not duplicated on every [`GpuColumnDesc`].
pub trait GpuTable: Resource {
    /// Highest slot index + 1 every column of this table must cover this frame.
    fn high_water(&self) -> u32;
}

/// Describes a per-[`GpuEntity`] GPU column: its value type, where the value
/// is read from, which entities changed each frame, and whether it keeps a
/// previous-frame copy. [`GpuColumnPlugin<C>`] turns this into a
/// self-contained scattered column.
pub trait GpuColumnDesc: Send + Sync + 'static {
    /// The per-slot value (`Pod`, size a multiple of 4 bytes).
    type Value: Pod;
    /// The table this column belongs to — its slot space and delta source.
    /// One scatter mechanism serves every table through this associated type.
    type Table: GpuTable;
    /// Debug label for the GPU buffers.
    const LABEL: &'static str;
    /// If set, the column keeps a previous-frame buffer; the scatter shifts
    /// `current → previous` on the GPU before writing (no `previous` upload).
    const KEEP_PREVIOUS: bool = false;
    /// Binding index in the shared scene-columns bind group ([`SceneColumns`]) if
    /// the RT scene shaders read this column; `None` = compute-only column (not in
    /// the scene group). It is the single source of truth shared by the layout, the
    /// bind-group entry, and the `@binding` in `gpu_columns.wgsl`.
    const SCENE_BINDING: Option<u32> = None;
    /// Binding index for the **previous-frame** buffer in the scene group, for a
    /// `KEEP_PREVIOUS` column the scene reads (e.g. `previous_frame_transforms`).
    const SCENE_BINDING_PREVIOUS: Option<u32> = None;
    /// This frame's delta as the raw `[slot, value-words…]` records the scatter
    /// consumes, pre-built by the producer (extract / material resolve). The
    /// column uploads the slice **straight to the GPU buffer** — no per-slot CPU
    /// mirror to gather, no copy. The GPU buffer is the column's only home; a
    /// capacity growth preserves it by GPU buffer copy, not a CPU re-scatter.
    fn delta_records(table: &Self::Table) -> &[u32];
}

/// The GPU buffers + heap slots for column `C`. One resource per column type;
/// the scatter pipelines are shared ([`ColumnScatterKernels`]).
#[derive(Resource)]
pub struct GpuColumn<C: GpuColumnDesc> {
    /// Slot-indexed `array<C::Value>`, a **sparse** buffer: a fixed virtual range
    /// ([`COLUMN_VIRTUAL_BYTES`]) reserved once, physical pages committed on
    /// growth. The handle/address never changes, so consumer bind groups stay
    /// valid across growth. Bound read-only by consumers, read-write by scatter.
    buffer: SparseBuffer,
    /// Previous-frame buffer — `Some` iff `C::KEEP_PREVIOUS`. Filled GPU-side by
    /// the history scatter; consumers bind it for temporal reuse. Also sparse.
    previous: Option<SparseBuffer>,
    /// Slots whose pages are committed (and zeroed). Grows by `next_power_of_two`.
    capacity_slots: u32,
    /// Packed delta: `pending` records of `[slot, value-words…]`.
    delta: RawBufferVec<u32>,
    /// This column's heap slots (delta / column / previous), rewritten per
    /// dispatch — per-column so parallel column dispatches never share a slot.
    slots: KernelSlots,
    /// Records to scatter this frame (the dispatch gates on this).
    pending: u32,
    /// Byte range of newly-committed sparse pages a growth needs zeroed (pages
    /// are UNDEFINED on first residency). [`dispatch_column`] records the clear
    /// before the scatter, then takes it.
    pending_clear: Option<Range<u64>>,
    _marker: PhantomData<fn() -> C>,
}

impl<C: GpuColumnDesc> GpuColumn<C> {
    /// `u32` words per value (`size_of::<C::Value>() / 4`).
    const WORDS: u32 = (size_of::<C::Value>() / 4) as u32;
    const STRIDE: u64 = size_of::<C::Value>() as u64;

    /// The raw column buffer handle (the inner `wgpu::Buffer` of the sparse store;
    /// stable across growth). Use this for bind-group cache keys (`Buffer::id()`) or
    /// when the binding size is set explicitly. **Do not** `as_entire_binding()` this
    /// into a shader that calls `arrayLength()` — that binds the whole sparse virtual
    /// reservation and `arrayLength()` becomes enormous (the GPU-hang class); use
    /// [`binding`](Self::binding), which sizes to the committed range, instead.
    #[inline]
    pub fn buffer(&self) -> &Buffer {
        self.buffer.buffer()
    }

    /// A descriptor binding sized to **exactly the committed range**
    /// ([`committed_bytes`](Self::committed_bytes)), so a shader's `arrayLength()` is
    /// the real slot count and an out-of-range index is bounds-checked rather than a
    /// page fault. `None` until the first page is committed (a consumer skips its
    /// bind group that frame). The committed-sized sibling of
    /// [`buffer`](Self::buffer); prefer this for any consumer that doesn't set the
    /// binding size itself.
    #[inline]
    pub fn binding(&self) -> Option<bevy_render::render_resource::BindingResource<'_>> {
        use bevy_render::render_resource::{BindingResource, BufferBinding};
        Some(BindingResource::Buffer(BufferBinding {
            buffer: self.buffer.buffer(),
            offset: 0,
            size: Some(NonZero::<u64>::new(self.committed_bytes())?),
        }))
    }

    /// The previous-frame buffer (`KEEP_PREVIOUS` columns only).
    #[inline]
    pub fn previous_buffer(&self) -> Option<&Buffer> {
        self.previous.as_ref().map(|previous| previous.buffer())
    }

    /// Bytes the committed pages cover this frame (`capacity_slots * STRIDE`).
    /// Consumers MUST bind a range of exactly this size — NOT the whole sparse
    /// buffer — so `arrayLength()` in shaders is the real slot count, not the
    /// `COLUMN_VIRTUAL_BYTES` reservation (a shader loop bounded by it hangs the
    /// GPU), and so an out-of-range index is bounds-checked, not a page fault.
    #[inline]
    pub fn committed_bytes(&self) -> u64 {
        self.capacity_slots as u64 * Self::STRIDE
    }

    /// This frame's delta record buffer: `pending` records of
    /// `[slot, value-words…]` (`record_stride` words each). `None` until the
    /// first upload. A consumer that wants the *changed-slot list* (e.g.
    /// transform propagation dispatching only over moved nodes) reads the `slot`
    /// field — `delta[k * record_stride]` for record `k`. Re-fetched each frame
    /// as the delta may reallocate.
    #[inline]
    pub fn delta_buffer(&self) -> Option<&Buffer> {
        self.delta.buffer()
    }

    /// Number of `[slot, …]` records in this frame's delta — the count of slots
    /// that changed this frame, i.e. the dispatch bound for a delta-driven pass.
    #[inline]
    pub fn pending(&self) -> u32 {
        self.pending
    }

    /// Words per delta record (`WORDS + 1`, the leading `slot` index plus the
    /// value words). Stride for indexing the [`delta_buffer`](Self::delta_buffer).
    #[inline]
    pub fn record_stride(&self) -> u32 {
        Self::WORDS + 1
    }

    /// Commit more sparse pages if `high_water` outgrew capacity. The virtual
    /// address is fixed, so growth binds physical pages to the *same* handle — no
    /// realloc, no old→new copy, and consumer bind groups stay valid (existing
    /// pages keep their data). The newly-committed region is undefined on first
    /// residency, so it's queued for a zero-clear in [`dispatch_column`] before
    /// the scatter.
    fn ensure_capacity(&mut self, high_water: u32) {
        if high_water <= self.capacity_slots {
            return;
        }
        let old_slots = self.capacity_slots;
        self.capacity_slots = high_water.next_power_of_two();
        let committed = self.capacity_slots as u64 * Self::STRIDE;
        self.buffer.commit(0..committed);
        if let Some(previous) = self.previous.as_ref() {
            previous.commit(0..committed);
        }
        self.pending_clear = Some(old_slots as u64 * Self::STRIDE..committed);
    }

    /// Upload a pre-built `[slot, words…]` delta straight to the GPU delta
    /// buffer. Sparse pages persist across growth, so the history scatter
    /// always shifts `current → previous`. The delta's heap descriptor is
    /// rewritten each dispatch, picking up any buffer the `reserve`
    /// reallocated.
    fn upload_prebuilt(&mut self, records: &[u32], device: &RenderDevice, queue: &RenderQueue) {
        self.pending = records.len() as u32 / (Self::WORDS + 1);
        self.delta.reserve(records.len(), device);
        // Upload via the staging-view path (`write_buffer_with`), NOT `write_buffer`:
        // the plain `write_buffer` copy path corrupts multi-MB deltas (a mass
        // regenerate makes every column's delta large), yielding garbage
        // `parent[]`/`material_id[]`.
        if let (Some(buffer), Some(bytes)) = (
            self.delta.buffer(),
            NonZero::<u64>::new(records.len() as u64 * 4),
        ) {
            let mut view = queue
                .write_buffer_with(buffer, 0, bytes)
                .expect("column delta staging allocation failed");
            view.copy_from_slice(bytemuck::cast_slice(records));
        }
    }

    /// Like [`upload_prebuilt`](Self::upload_prebuilt), but the producer writes
    /// the records straight into the queue's staging memory instead of handing
    /// over a finished CPU slice — one copy (producer → staging) instead of two
    /// (producer → merge `Vec` → staging). Worth it only for the big per-frame
    /// deltas (the transform `local` column: one record per mover); small
    /// columns should keep the plain [`upload_prebuilt`] path.
    ///
    /// `fill` receives exactly `total_words * 4` bytes of staging memory and
    /// must write **all** of them (staging is uninitialized — unwritten bytes
    /// would upload garbage records).
    pub fn write_delta_direct(
        &mut self,
        total_words: usize,
        device: &RenderDevice,
        queue: &RenderQueue,
        fill: impl FnOnce(wgpu::WriteOnly<'_, [u8]>),
    ) {
        debug_assert_eq!(total_words as u32 % (Self::WORDS + 1), 0);
        self.pending = total_words as u32 / (Self::WORDS + 1);
        self.delta.reserve(total_words, device);
        if let (Some(buffer), Some(bytes)) = (
            self.delta.buffer(),
            NonZero::<u64>::new(total_words as u64 * 4),
        ) {
            let mut view = queue
                .write_buffer_with(buffer, 0, bytes)
                .expect("delta staging allocation failed");
            fill(view.slice(..));
        }
    }

}

/// Registers column `C`: a `GpuColumn<C>` resource (with its own pipeline) and
/// its init / prepare / bind-group / scatter systems (all generic over `C`).
/// Adding a column to the cluster pipeline is one `GpuColumnPlugin::<C>::default()`.
pub struct GpuColumnPlugin<C: GpuColumnDesc>(PhantomData<fn() -> C>);

impl<C: GpuColumnDesc> Default for GpuColumnPlugin<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C: GpuColumnDesc> Plugin for GpuColumnPlugin<C> {
    fn build(&self, app: &mut App) {
        // If this column is read by the RT scene shaders, register it into the
        // shared scene-columns bind group (a no-op unless `C::SCENE_BINDING` is set).
        super::scene_columns::register_scene_column::<C>(app);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        // The two scatter kernels every column shares — registered by the first
        // column plugin, used by all.
        if render_app
            .world()
            .get_resource::<ScatterKernelsRegistered>()
            .is_none()
        {
            render_app.insert_resource(ScatterKernelsRegistered);
            render_app.add_systems(
                RenderStartup,
                init_column_scatter_kernels.after(SolariSetup),
            );
        }
        render_app
            .init_resource::<super::SolariPipelineRegistry>()
            .add_systems(RenderStartup, init_column::<C>.after(SolariSetup))
            .add_systems(
                Render,
                prepare_column::<C>
                    .in_set(RenderSystems::PrepareResources)
                    .in_set(GpuColumnPrepareSet),
            )
            .add_systems(
                bevy_render::renderer::RenderGraph,
                dispatch_column::<C>.in_set(SolariClusterSystems::Scatter),
            );
    }
}

/// Marks the shared-kernel init as registered (the column plugins race to be
/// first; exactly one registers it).
#[derive(Resource)]
struct ScatterKernelsRegistered;

/// The two scatter kernels every [`GpuColumn`] shares: the buffers are
/// per-dispatch heap slots, so one pipeline per entry point serves every
/// column type.
#[derive(Resource)]
pub struct ColumnScatterKernels {
    scatter: HeapKernel,
    with_history: HeapKernel,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for ColumnScatterKernels {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        for kernel in [&self.scatter, &self.with_history] {
            // SAFETY: quiesced; handles exclusively owned here.
            unsafe { kernel.destroy(&self.raw_device) };
        }
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for ColumnScatterKernels {}
unsafe impl Sync for ColumnScatterKernels {}

/// `RenderStartup` (after `SolariSetup`): compile the shared scatter kernels.
fn init_column_scatter_kernels(
    mut commands: Commands,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let make = |entry: &str, label: &str| {
        HeapKernel::new(
            &seam,
            "gpu_instances_scatter.slang",
            include_str!("gpu_instances_scatter.slang"),
            entry,
            &[],
            &[],
            label,
            size_of::<ScatterParams>() as u32,
        )
    };
    let (Some(scatter), Some(with_history)) = (
        make("scatter", "column_scatter"),
        make("scatter_with_history", "column_scatter_with_history"),
    ) else {
        return;
    };
    commands.insert_resource(ColumnScatterKernels {
        scatter,
        with_history,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

/// `RenderStartup`: build column `C`'s buffers + heap slots.
fn init_column<C: GpuColumnDesc>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
    seam: Option<Res<BindingSeam>>,
) {
    debug_assert_eq!(
        size_of::<C::Value>() % 4,
        0,
        "GpuColumn value must be a 4-byte multiple"
    );
    // Columns live in sparse buffers (stable handle across growth). The cluster
    // allocator is created in `SolariSetup`, which this runs after; if it's
    // absent the device lacks the cluster/sparse support solari needs, so skip —
    // every consumer reads the column via `Option<Res<GpuColumn<C>>>`.
    let (Some(allocator), Some(seam)) = (allocator, seam) else {
        return;
    };
    let make_sparse = |label: &'static str| {
        allocator.create_sparse_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            COLUMN_VIRTUAL_BYTES,
            label,
        )
    };
    let previous = C::KEEP_PREVIOUS.then(|| {
        // Distinct label: the Aftermath VA-map triage resolves faulting
        // addresses by buffer label, and two buffers both named `C::LABEL`
        // make the current/previous pair ambiguous. One leak per history
        // column at startup, bounded by the column count.
        make_sparse(Box::leak(format!("{}.previous", C::LABEL).into_boxed_str()))
    });
    let mut delta = RawBufferVec::<u32>::new(BufferUsages::STORAGE);
    delta.set_label(Some(C::LABEL));

    commands.insert_resource(GpuColumn::<C> {
        buffer: make_sparse(C::LABEL),
        previous,
        capacity_slots: 0,
        delta,
        slots: KernelSlots::new(&seam, 3),
        pending: 0,
        pending_clear: None,
        _marker: PhantomData,
    });
}

/// `Render::Prepare`: grow the column and upload this frame's delta. Runs after
/// `extract` populates the table. Parallel with the other columns.
fn prepare_column<C: GpuColumnDesc>(
    mut column: Option<ResMut<GpuColumn<C>>>,
    table: Option<Res<C::Table>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(table) = table else {
        return;
    };
    let Some(column) = column.as_deref_mut() else {
        // Tripwire: the table's delta is cleared unconditionally at Cleanup on
        // the assumption this system moved it into the column's retained
        // `pending`. With no column resource to receive it, the records are
        // about to be dropped on the floor — a silent one-shot loss (the
        // frontier-seed-loss class). Say so instead of rendering wrong forever.
        let dropped = C::delta_records(&table).len();
        if dropped > 0 {
            bevy_log::warn_once!(
                "GpuColumn<{}>: {dropped} delta word(s) with no column resource to \
                 receive them — cleared unconsumed at Cleanup (column init skipped?)",
                C::LABEL,
            );
        }
        return;
    };
    let high_water = table.high_water().max(1);
    column.ensure_capacity(high_water);

    // The producer already built this frame's `[slot, words…]` records; upload
    // the slice straight to the GPU buffer.
    let records = C::delta_records(&table);
    if records.is_empty() {
        // Nothing new this frame. DON'T clear `pending` here: if a previous
        // delta is still un-scattered (`dispatch_column` skipped it because the
        // kernels weren't ready), zeroing `pending` would abandon it forever
        // — fatal for a fully-static scene whose instances all bound before
        // warmup. `dispatch_column` clears `pending` once it actually
        // scatters (retain-until-consumed).
        return;
    }
    column.upload_prebuilt(records, &render_device, &render_queue);
}

/// `RenderGraph` (`Scatter`): on a growth, zero the newly-committed pages, then
/// scatter column `C`'s delta with the shared kernel. One raw encoder segment:
/// the fill (transfer) and the scatter (compute) with explicit barriers.
fn dispatch_column<C: GpuColumnDesc>(
    column: Option<ResMut<GpuColumn<C>>>,
    kernels: Option<Res<ColumnScatterKernels>>,
    seam: Option<Res<BindingSeam>>,
    mut ctx: RenderContext,
) {
    let (Some(mut column), Some(kernels), Some(seam)) = (column, kernels, seam) else {
        // `pending`/`pending_clear` stay live and retry next frame.
        return;
    };
    let column = &mut *column;
    let clear = column.pending_clear.take();
    let scatter = column.pending > 0 && column.delta.buffer().is_some();
    if clear.is_none() && !scatter {
        return;
    }

    let kernel = if C::KEEP_PREVIOUS {
        &kernels.with_history
    } else {
        &kernels.scatter
    };
    let groups = super::linear_dispatch(column.pending.div_ceil(SCATTER_WORKGROUP_SIZE));
    let blob = scatter.then(|| {
        let params = ScatterParams {
            count: column.pending,
            words_per_value: GpuColumn::<C>::WORDS,
            groups_x: groups.0,
            _pad: 0,
        };
        let delta = column.delta.buffer().unwrap();
        let mut named = vec![
            ("delta", column.slots.buffer(&seam, 0, delta)),
            ("column", column.slots.buffer(&seam, 1, column.buffer.buffer())),
        ];
        if let Some(previous) = column.previous.as_ref() {
            named.push(("previous", column.slots.buffer(&seam, 2, previous.buffer())));
        }
        kernel.push_blob(C::LABEL, bytemuck::bytes_of(&params), &named)
    });

    let encoder = ctx.command_encoder();
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // barriers bracket the fill + scatter against the surrounding passes (raw
    // commands are invisible to wgpu's tracking).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &kernels.raw_device;
            // Newly-committed sparse pages are UNDEFINED on first residency.
            // Zero the grown `[old..new)` region before scattering so
            // unscattered-but-active slots read 0 (and the history scatter's
            // `previous = old current` reads a defined 0 for a brand-new
            // slot). Existing pages persist with their data, so only the new
            // region is cleared. Runs even when the scatter skips, so
            // `pending` stays live to scatter once ready.
            if let Some(range) = clear {
                let len = range.end - range.start;
                if len > 0 {
                    let pre = [vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)];
                    dev.cmd_pipeline_barrier2(
                        cb,
                        &vk::DependencyInfo::default().memory_barriers(&pre),
                    );
                    dev.cmd_fill_buffer(cb, column.buffer.raw(), range.start, len, 0);
                    if let Some(previous) = column.previous.as_ref() {
                        dev.cmd_fill_buffer(cb, previous.raw(), range.start, len, 0);
                    }
                    let post = [vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(
                            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                        )];
                    dev.cmd_pipeline_barrier2(
                        cb,
                        &vk::DependencyInfo::default().memory_barriers(&post),
                    );
                }
            }
            if let Some(blob) = blob.as_ref() {
                let barrier = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(
                        vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                    )];
                let dep = vk::DependencyInfo::default().memory_barriers(&barrier);
                // Prior compute writes -> the scatter's reads/writes.
                dev.cmd_pipeline_barrier2(cb, &dep);
                seam.bind_heaps(cb);
                seam.push_data(cb, blob);
                dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
                dev.cmd_dispatch(cb, groups.0, groups.1, groups.2);
                // The scatter's column writes -> downstream compute reads.
                dev.cmd_pipeline_barrier2(cb, &dep);
            }
        });
    }
    if scatter {
        // Delta is now consumed — clear so a later empty frame doesn't
        // re-scatter it (and so `prepare_column`'s "retain un-scattered delta"
        // guard releases).
        column.pending = 0;
    }
}
