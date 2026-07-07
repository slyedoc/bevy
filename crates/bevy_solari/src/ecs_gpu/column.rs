//! `GpuColumn<C>` — one persistent, slot-indexed GPU buffer per
//! [`GpuColumnDesc`], the GPU-side "component" storage of a [`GpuEntity`].
//!
//! A column is described once (its value type, where the value comes from,
//! which entities changed this frame, and whether it keeps a previous-frame
//! copy); [`GpuColumnPlugin<C>`] turns that into a **fully self-contained**
//! scatter pipeline — its own bind-group layout, compute pipeline, buffers,
//! and init/prepare/bind-group/dispatch systems, all generic over `C`. Each
//! column is its own resource, so Bevy runs the per-column prepares in
//! parallel.
//!
//! Updates are scattered: only the entities that changed cross the bus as a
//! compact `(slot, value)` delta, and the byte-wise scatter shader
//! (`gpu_instances_scatter.wgsl`) places them. A `KEEP_PREVIOUS` column also
//! shifts the current value into a previous-frame buffer on the GPU (the GPU
//! already holds last frame's value — no second upload). This is the same
//! family as Bevy's raster `UniformComponentPlugin<C>` / `GpuArrayBufferPlugin<T>`.

use core::marker::PhantomData;
use core::num::NonZero;

use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer};
use bevy_ecs::{
    resource::Resource,
    schedule::{IntoScheduleConfigs, SystemSet},
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
        BufferUsages, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
        PipelineCache, RawBufferVec, ShaderStages, ShaderType, UniformBuffer,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    Render, RenderApp, RenderStartup, RenderSystems,
};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use core::ops::Range;

use crate::gpu::allocator::{Allocator, SparseBuffer};
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

/// Uniform shared with `gpu_instances_scatter.wgsl::ScatterParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
struct ScatterParams {
    count: u32,
    words_per_value: u32,
    /// 1 on a grown buffer (history columns): `previous = new` instead of
    /// shifting the fresh buffer's garbage.
    force_init: u32,
    _pad1: u32,
}

/// All `GpuColumn` prepare systems run in this set, so callers can order
/// producers (e.g. `resolve_instance_material_ids`, which writes the values a
/// column reads) before every column with one edge.
#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct GpuColumnPrepareSet;

/// Describes a per-[`GpuEntity`] GPU column: its value type, where the value
/// is read from, which entities changed each frame, and whether it keeps a
/// previous-frame copy. [`GpuColumnPlugin<C>`] turns this into a self-contained
/// scatter pipeline.
/// A GPU-mirrored table: a slot space whose columns scatter per-frame deltas.
/// `InstanceManager` (per-instance columns) and `TransformGraph` (transform-table
/// columns) are tables. `high_water` is a property of the *table* — shared by all
/// its columns — so it lives here, not duplicated on every [`GpuColumnDesc`].
pub trait GpuTable: Resource {
    /// Highest slot index + 1 every column of this table must cover this frame.
    fn high_water(&self) -> u32;
}

pub trait GpuColumnDesc: Send + Sync + 'static {
    /// The per-slot value (`Pod`, size a multiple of 4 bytes).
    type Value: Pod;
    /// The table this column belongs to — its slot space and delta source.
    /// One scatter mechanism serves every table through this associated type.
    type Table: GpuTable;
    /// Debug label for the GPU buffers + pipeline.
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

/// The GPU buffers + its own scatter pipeline for column `C`. One resource per
/// column type.
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
    params: UniformBuffer<ScatterParams>,
    /// This column's own bind-group layout (3 bindings, or 4 with `previous`).
    layout: BindGroupLayoutDescriptor,
    /// This column's own scatter pipeline (plain or `scatter_with_history`).
    pipeline: CachedComputePipelineId,
    bind_group: Option<BindGroup>,
    /// Records to scatter this frame (the dispatch gates on this).
    pending: u32,
    /// Byte range of newly-committed sparse pages a growth needs zeroed (pages
    /// are UNDEFINED on first residency). [`dispatch_column`] records the clear
    /// before the scatter — matching the old wgpu zero-init — then takes it.
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
    /// `COLUMN_VIRTUAL_BYTES` reservation (binding the full 1 GiB made
    /// `arrayLength(&directional_lights)` ~33M → the path tracer's light loop hung
    /// the GPU), and so an out-of-range index is bounds-checked, not a page fault.
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

    /// Whether this column's scatter compute pipeline has finished compiling.
    /// Until it has, [`dispatch_column`] can't scatter a delta, so binding new
    /// instances must wait (a bind whose scatter is skipped would leave the
    /// column zero — see [`prepare_column`]).
    #[inline]
    pub fn scatter_pipeline_ready(&self, cache: &PipelineCache) -> bool {
        cache.get_compute_pipeline(self.pipeline).is_some()
    }

    /// Commit more sparse pages if `high_water` outgrew capacity. The virtual
    /// address is fixed, so growth binds physical pages to the *same* handle — no
    /// realloc, no old→new copy, and consumer bind groups stay valid (existing
    /// pages keep their data). The newly-committed region is undefined on first
    /// residency, so it's queued for a zero-clear in [`dispatch_column`] before
    /// the scatter (matching the old wgpu zero-init).
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

    /// Upload a pre-built `[slot, words…]` delta straight to the GPU buffer
    /// (the producer already built the records) — `reserve` the delta buffer,
    /// then `write_buffer` the slice into it directly. No CPU staging Vec is
    /// touched: one upload, no gather/copy/swap. History columns never
    /// `force_init` — a growth preserves the previous buffer by GPU copy, so the
    /// scatter always shifts `current → previous`. The bind group rebuilds each
    /// scattering frame, picking up any buffer the `reserve` reallocated.
    fn upload_prebuilt(&mut self, records: &[u32], device: &RenderDevice, queue: &RenderQueue) {
        self.pending = records.len() as u32 / (Self::WORDS + 1);
        self.delta.reserve(records.len(), device);
        // Upload via the staging-view path (`write_buffer_with`), NOT `write_buffer`.
        // On a mass regenerate a "tiny" column (parent / material_id / …) becomes a
        // multi-MB delta (every entity is first-sight); the plain `write_buffer` copy
        // path corrupts at that scale (garbage `parent[]`/`material_id[]` → wrong
        // transforms + materials), while the staging view the big `local` delta uses
        // is fine. Route everything through it.
        if let (Some(buffer), Some(bytes)) = (
            self.delta.buffer(),
            NonZero::<u64>::new(records.len() as u64 * 4),
        ) {
            let mut view = queue
                .write_buffer_with(buffer, 0, bytes)
                .expect("column delta staging allocation failed");
            view.copy_from_slice(bytemuck::cast_slice(records));
        }
        *self.params.get_mut() = ScatterParams {
            count: self.pending,
            words_per_value: Self::WORDS,
            force_init: 0,
            _pad1: 0,
        };
        self.params.write_buffer(device, queue);
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
        *self.params.get_mut() = ScatterParams {
            count: self.pending,
            words_per_value: Self::WORDS,
            force_init: 0,
            _pad1: 0,
        };
        self.params.write_buffer(device, queue);
    }

    fn prepare_bind_group(&mut self, device: &RenderDevice, cache: &PipelineCache) {
        let (Some(delta), Some(params)) = (self.delta.buffer(), self.params.binding()) else {
            self.bind_group = None;
            return;
        };
        let layout = cache.get_bind_group_layout(&self.layout);
        self.bind_group = Some(match &self.previous {
            Some(previous) => device.create_bind_group(
                C::LABEL,
                &layout,
                &BindGroupEntries::sequential((
                    delta.as_entire_binding(),
                    self.buffer.buffer().as_entire_binding(),
                    params,
                    previous.buffer().as_entire_binding(),
                )),
            ),
            None => device.create_bind_group(
                C::LABEL,
                &layout,
                &BindGroupEntries::sequential((
                    delta.as_entire_binding(),
                    self.buffer.buffer().as_entire_binding(),
                    params,
                )),
            ),
        });
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
        // The byte-wise scatter shader every column shares. Registered here (next
        // to the `load_embedded_asset!` site in `init_column`) so the embedded path
        // resolves to `ecs_gpu/`. Idempotent across the many column plugins.
        embedded_asset!(app, "gpu_instances_scatter.wgsl");

        // If this column is read by the RT scene shaders, register it into the
        // shared scene-columns bind group (a no-op unless `C::SCENE_BINDING` is set).
        super::scene_columns::register_scene_column::<C>(app);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<super::SolariPipelineRegistry>()
            .add_systems(RenderStartup, init_column::<C>.after(SolariSetup))
            .add_systems(
                Render,
                (
                    prepare_column::<C>
                        .in_set(RenderSystems::Prepare)
                        .in_set(GpuColumnPrepareSet),
                    prepare_column_bind_group::<C>.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                bevy_render::renderer::RenderGraph,
                dispatch_column::<C>.in_set(SolariClusterSystems::Scatter),
            );
    }
}

/// `RenderStartup`: build column `C`'s buffers + its own bind-group layout and
/// scatter pipeline (the 4-binding `scatter_with_history` variant when it keeps
/// a previous-frame copy).
fn init_column<C: GpuColumnDesc>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    allocator: Option<Res<Allocator>>,
    mut registry: ResMut<super::SolariPipelineRegistry>,
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
    let Some(allocator) = allocator else {
        return;
    };
    let make_sparse = || {
        allocator.create_sparse_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            COLUMN_VIRTUAL_BYTES,
            C::LABEL,
        )
    };
    let (layout, entry_point, previous) = if C::KEEP_PREVIOUS {
        (
            BindGroupLayoutDescriptor::new(
                C::LABEL,
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        storage_buffer_read_only_sized(false, None), // 0 delta
                        storage_buffer_sized(false, None),           // 1 column (rw)
                        uniform_buffer::<ScatterParams>(false),      // 2 params
                        storage_buffer_sized(false, None),           // 3 previous (rw)
                    ),
                ),
            ),
            "scatter_with_history",
            Some(make_sparse()),
        )
    } else {
        (
            BindGroupLayoutDescriptor::new(
                C::LABEL,
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        storage_buffer_read_only_sized(false, None), // 0 delta
                        storage_buffer_sized(false, None),           // 1 column (rw)
                        uniform_buffer::<ScatterParams>(false),      // 2 params
                    ),
                ),
            ),
            "scatter",
            None,
        )
    };
    let shader = load_embedded_asset!(asset_server.as_ref(), "gpu_instances_scatter.wgsl");
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(C::LABEL.into()),
        layout: vec![layout.clone()],
        shader,
        shader_defs: vec![],
        entry_point: Some(entry_point.into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    registry.register(C::LABEL, pipeline);

    let mut delta = RawBufferVec::<u32>::new(BufferUsages::STORAGE);
    delta.set_label(Some(C::LABEL));
    let mut params = UniformBuffer::<ScatterParams>::default();
    params.set_label(Some(C::LABEL));

    commands.insert_resource(GpuColumn::<C> {
        buffer: make_sparse(),
        previous,
        capacity_slots: 0,
        delta,
        params,
        layout,
        pipeline,
        bind_group: None,
        pending: 0,
        pending_clear: None,
        _marker: PhantomData,
    });
}

/// `Render::Prepare`: grow + build + upload column `C`'s delta. Runs after
/// `extract` populates the `InstanceManager`. Parallel with the other columns.
///
/// Growth no longer re-scatters every active slot from a CPU mirror — the
/// realloc stashes the old buffers and [`dispatch_column`] copies them old→new
/// on the GPU, so prepare only ever uploads this frame's change delta (whether
/// a column supplies it pre-built or gathers it via `dirty` + `value`).
fn prepare_column<C: GpuColumnDesc>(
    mut column: Option<ResMut<GpuColumn<C>>>,
    table: Option<Res<C::Table>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(column), Some(table)) = (column.as_deref_mut(), table) else {
        return;
    };
    let high_water = table.high_water().max(1);
    column.ensure_capacity(high_water);

    // The producer already built this frame's `[slot, words…]` records; upload
    // the slice straight to the GPU buffer (no gather, no copy, no staging Vec).
    // Nothing changed → skip the upload; `pending = 0` gates the dispatch and
    // the buffer keeps its data. Bind-only columns hit this every steady frame.
    let records = C::delta_records(&table);
    if records.is_empty() {
        // Nothing new this frame. DON'T clear `pending` here: if a previous
        // delta is still un-scattered (its compute pipeline hadn't compiled yet,
        // so `dispatch_column` skipped it), zeroing `pending` would abandon it
        // forever — fatal for a fully-static scene whose instances were all
        // bound on frame 1, before pipeline warmup (every column value stays
        // zero → transforms collapse every instance → all rays miss).
        // `dispatch_column` clears `pending` once it actually scatters, so a
        // steady-state static frame already has `pending == 0` here.
        return;
    }
    column.upload_prebuilt(records, &render_device, &render_queue);
}

/// `Render::PrepareBindGroups`: (re)build column `C`'s scatter bind group.
fn prepare_column_bind_group<C: GpuColumnDesc>(
    mut column: Option<ResMut<GpuColumn<C>>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let Some(column) = column.as_deref_mut() else {
        return;
    };
    // Nothing to scatter → the dispatch skips, so the bind group isn't used;
    // don't rebuild it. (A `pending > 0` frame always rebuilds, which also
    // covers the delta buffer reallocating.)
    if column.pending == 0 {
        return;
    }
    column.prepare_bind_group(&render_device, &pipeline_cache);
}

/// `RenderGraph` (`Scatter`): on a growth, copy the old buffers old→new on the
/// GPU (preserving every slot's value), then scatter column `C`'s delta with
/// its own pipeline. Both record into the shared `RenderContext` encoder.
fn dispatch_column<C: GpuColumnDesc>(
    column: Option<ResMut<GpuColumn<C>>>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let Some(mut column) = column else {
        return;
    };
    let column = &mut *column;
    let clear = column.pending_clear.take();
    let scatter = if column.pending > 0 {
        match (
            pipeline_cache.get_compute_pipeline(column.pipeline),
            column.bind_group.as_ref(),
        ) {
            (Some(pipeline), Some(bind_group)) => Some((pipeline, bind_group)),
            _ => None,
        }
    } else {
        None
    };
    if clear.is_none() && scatter.is_none() {
        return;
    }

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    // Newly-committed sparse pages are UNDEFINED on first residency. Zero the
    // grown `[old..new)` region before scattering so unscattered-but-active slots
    // read 0 (matching the old wgpu zero-init — and giving `scatter_with_history`'s
    // `previous = old current` a defined 0 for a brand-new slot). Existing pages
    // persist with their data, so only the new region is cleared. No pipeline
    // needed, so a cold-pipeline frame still zeroes; `pending` stays live to
    // scatter once the pipeline compiles.
    if let Some(range) = clear {
        let len = range.end - range.start;
        if len > 0 {
            encoder.clear_buffer(&column.buffer.wgpu_buffer, range.start, Some(len));
            if let Some(previous) = column.previous.as_ref() {
                encoder.clear_buffer(&previous.wgpu_buffer, range.start, Some(len));
            }
        }
    }
    if let Some((pipeline, bind_group)) = scatter {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(C::LABEL),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let d = diagnostics.time_span(&mut pass, C::LABEL);
        let (gx, gy, gz) =
            super::linear_dispatch(column.pending.div_ceil(SCATTER_WORKGROUP_SIZE));
        pass.dispatch_workgroups(gx, gy, gz);
        d.end(&mut pass);
        // Delta is now consumed — clear so a later empty frame doesn't re-scatter
        // it (and so `prepare_column`'s "retain un-scattered delta" guard
        // releases). Until this runs, a cold-pipeline skip keeps `pending` live.
        column.pending = 0;
    }
}
