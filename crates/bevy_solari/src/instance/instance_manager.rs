//! Per-entity instance tracking for [`RaytracingMesh3d`] entities.
//!
//! Event-driven, O(changes): lifecycle **observers** flag only the entities that
//! actually changed into [`RtInstanceChanges`] — `Add<RaytracingMesh3d>` (bind),
//! `Insert<SolariMaterial3d>` / `Insert<RenderLayers>` (update) — and the despawn
//! observer frees slots. The flush drains that set. No per-frame query scan: the
//! full RT-mesh set is *never* iterated (the old `Or<(Added, Changed, …)>` extract
//! had to tick-scan all 2M instances each frame just to find the few that
//! changed). Transform moves are deliberately NOT tracked — movement is fully
//! GPU-driven (transform table + PTLAS-fill move detection), so a moving scene
//! flags nothing here. `active_slots` is **persistent** (push on bind, O(1)
//! swap-remove on despawn via `slot_active_pos`).
//!
//! Per-slot data is written only when its source changes; static
//! instances are never touched. The deltas
//! (`added`/`rewrite`/`disabled`/`released`/column scatters) feed the
//! downstream GPU consumers (column scatter, BLAS regions, PTLAS op
//! stream). The ECS is the change oracle — there is no shadow cache,
//! refcount, or mark-and-sweep here.
//!
//! Slot lifecycle:
//! - Allocated on first `Added` (or first frame an entity's streamed
//!   `ClusterMesh` asset becomes ready — see the `pending` retry set).
//! - Freed on `RemovedComponents<RaytracingMesh3d>` / despawn.

use crate::geometry::{ClasArena, ClusterMeshManager, ClusterMeshUpload};
use crate::geometry::{ClusterIndex, GroupIndex, GpuEntity};
use crate::bindings::RaytracingMesh3d;
use crate::geometry::ClusterMesh;
use crate::material::MaterialSlots;
use super::journal::{InstanceJournalRecord, RtJournal};
use bevy_asset::{AssetEvent, AssetId, AssetServer, Assets};
use bytemuck::{Pod, Zeroable};
use bevy_ecs::{
    component::Component,
    entity::{Entity, EntityHashMap, EntityHashSet},
    lifecycle::{Add, Insert, Remove},
    message::MessageReader,
    observer::On,
    query::With,
    resource::Resource,
    system::{Commands, Local, Query, Res, ResMut, SystemState},
};
use crate::ecs_gpu::GpuSlot;
use crate::transform::TransformGraph;
use crate::render::view_cull::render_layers_to_mask;
use bevy_camera::visibility::RenderLayers;
use bevy_platform::collections::HashSet;

use crate::material::{SolariMaterial, SolariMaterial3d};
use bevy_render::{
    render_resource::ShaderType, renderer::RenderDevice, sync_world::RenderEntity, MainWorld,
};


/// Render-world component carrying this instance's GPU [`GpuEntity`].
///
/// Lives on the **synced render entity** (`RaytracingMesh3d` requires
/// `SyncToRenderWorld`, and `entity_sync_system` runs before
/// `ExtractSchedule`, so the render entity always exists by extract time).
/// The per-frame extract reads the slot back via `Query::get(render_entity)`
/// — the ECS's native entity index, replacing the old `EntityHashMap`.
/// Allocated on first sight of a resident mesh; freed by the
/// [`free_cluster_slot`] observer when the render entity despawns.
#[derive(Component, Clone, Copy)]
pub struct RaytracingGpuEntity(pub GpuEntity);

/// Per-slot global pointers into the [`ClusterMeshManager`] pools.
/// Cached at slot bind; cheap to keep since the asset upload is itself
/// persistent (slot reuse implies the asset upload was reused). Only
/// the fields actually consumed downstream are kept — `cluster_base` /
/// `cluster_count` (LOD inputs + per-slot BLAS region sizing),
/// `group_base` (selector), `root_group` (BLAS-sharing classify reads
/// the geometry's root traversal sphere), `total_triangle_count`
/// (emissive sampling).
#[derive(Clone, Copy, Debug)]
struct SlotMeshPointers {
    cluster_base: ClusterIndex,
    group_base: GroupIndex,
    root_group: GroupIndex,
    cluster_count: u32,
    total_triangle_count: u32,
    /// Dense geometry id (which unique `ClusterMesh`) — BLAS sharing key.
    geometry_id: u32,
}

/// Packed per-instance LOD inputs, slot-indexed. Consumed by the
/// selector compute pass (`cluster_base`/`cluster_count`) and the
/// BLAS-sharing classify pass (`group_base` = geometry key,
/// `root_group` = global index of the geometry's root group for the
/// representative-sphere projection). WGSL-aligned (16 B); all fields
/// are `u32` so the natural 4-byte alignment is preserved.
///
/// Layout mirror in WGSL (`cluster_bindings.wgsl`):
/// ```wgsl
/// struct InstanceLodInput {
///     cluster_base: u32,
///     cluster_count: u32,
///     group_base: u32,
///     root_group: u32,
/// }
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
pub struct InstanceLodInputGpu {
    pub cluster_base: u32,
    pub cluster_count: u32,
    pub group_base: u32,
    pub root_group: u32,
}

const _: () = assert!(size_of::<InstanceLodInputGpu>() == 16);

/// Render-world resource: per-entity instance tracking for
/// [`RaytracingMesh3d`] entities.
#[derive(Resource)]
pub struct InstanceManager {
    /// Free-list of slot indices for reuse after release.
    free_slots: Vec<u32>,
    /// Next slot index when the free-list is empty.
    next_slot: u32,
    /// Per-slot mesh-pool pointers cached at slot bind
    /// (cluster_base / group_base / cluster_count / triangle count).
    slot_mesh_pointers: Vec<Option<SlotMeshPointers>>,

    /// Per-slot material `AssetId`. Kept for the binder's emissive-light
    /// walk + to detect material changes; resolved to the GPU material index
    /// (a delta record pushed to [`Self::material_delta`]) by
    /// [`resolve_instance_material_ids`]. The per-instance `group_base` /
    /// `lod_input` / `geometry_id` / resolved `material_id` are no longer
    /// mirrored CPU-side at all — they are scattered delta-direct (see the
    /// per-column deltas below); the GPU buffer is their only home and is
    /// preserved across a growth by GPU buffer copy.
    instance_material_asset_ids: Vec<AssetId<SolariMaterial>>,

    /// Per-slot 8-bit RT cull mask (from `RenderLayers`). CPU mirror kept
    /// only to diff against on an update — a transform-only move must not
    /// re-push the mask. The GPU `InstanceMaskColumn` is the real home; this
    /// mirror is never read by a consumer. Mirrors the `material` change-guard.
    instance_masks: Vec<u32>,

    /// Currently-bound slots, **persistent** across frames — mutated
    /// only on bind (push) / despawn (swap-remove), never rebuilt by a
    /// per-frame iteration. The selector / `blas_rebuild` dispatch one
    /// item per entry; an entry's position is its *dense index* into
    /// their address tables.
    active_slots: Vec<GpuEntity>,
    /// slot → its index in [`Self::active_slots`] (`u32::MAX` = not
    /// active). Enables O(1) swap-remove on despawn.
    slot_active_pos: Vec<u32>,

    // ── Per-frame deltas (cleared in `begin_frame`) ───────────────
    // Driven by ECS change detection in the cluster extract:
    // `Added` → added, `Changed<RenderLayers>` → rewrite (mask),
    // the despawn observer → disabled + released. Transform moves are NOT
    // here — they're detected GPU-side by the PTLAS fill. Static instances
    // are never touched — that's what makes extract O(changes).
    /// Bound this frame → full `WRITE_INSTANCE` + all columns.
    added_slots: Vec<GpuEntity>,
    /// CPU identity changes that need an explicit PTLAS `WRITE_INSTANCE`
    /// because the driver can't carry them via `src` and `fill_incremental`
    /// can't detect them: a cull-mask (`RenderLayers`) change — the mask is
    /// baked into the instance's TLAS record. NOT transform moves: those are
    /// detected GPU-side by `fill_incremental` comparing current vs previous
    /// world transforms, so the CPU never tracks them.
    rewrite_slots: Vec<GpuEntity>,
    /// Despawned this frame → `WRITE_INSTANCE` with a null AS address
    /// to remove it from the PTLAS.
    disabled_slots: Vec<GpuEntity>,
    /// Despawned this frame → owners free per-slot resources (the BLAS
    /// region allocator in `blas_rebuild`).
    released_slots: Vec<GpuEntity>,
    /// Pre-built per-column scatter deltas — the raw `[slot, value-words…]`
    /// records each `GpuColumn` uploads verbatim. The value the column needs is
    /// appended here at its point of change (no per-slot CPU mirror to gather
    /// CPU column scatter deltas. Only the columns the CPU still owns are here:
    /// `material` (resolved when [`resolve_instance_material_ids`] resolves a changed
    /// slot) and the cull `mask` (bind + `RenderLayers` change). The bind-only columns
    /// (`group_base` / `lod_input` / `geometry_id` / `node_slot`) are written GPU-side
    /// by the reconcile pass from the journal, so they have no CPU delta.
    material_delta: Vec<u32>,
    /// Cull mask scatter delta — pushed at bind and on a `RenderLayers` change.
    instance_mask_delta: Vec<u32>,
    /// Slots whose material `AssetId` changed this frame (added ∪
    /// material-swapped). [`resolve_instance_material_ids`] re-resolves
    /// only these to a GPU material index; the binder also reads it to decide
    /// whether the emissive-light set might have changed.
    material_dirty: Vec<GpuEntity>,

    /// Running max of any bound instance's `cluster_count` — the
    /// worst-case per-instance cluster emit, used by `blas_rebuild` to
    /// size scratch / the cluster-AS build. Updated at `bind` (only
    /// grows). NOT recomputed on despawn: a stale-high value only
    /// over-sizes scratch (safe), never under-builds, and avoids an
    /// O(active) rescan every frame. Reset to 0 only when the scene
    /// fully empties.
    max_active_cluster_count: u32,
}

/// Sentinel in [`InstanceManager::slot_active_pos`]: slot not active.
const NOT_ACTIVE: u32 = u32::MAX;

impl InstanceManager {
    pub fn new() -> Self {
        Self {
            free_slots: Vec::new(),
            next_slot: 0,
            slot_mesh_pointers: Vec::new(),
            instance_material_asset_ids: Vec::new(),
            instance_masks: Vec::new(),
            active_slots: Vec::new(),
            slot_active_pos: Vec::new(),
            added_slots: Vec::new(),
            rewrite_slots: Vec::new(),
            disabled_slots: Vec::new(),
            released_slots: Vec::new(),
            material_delta: Vec::new(),
            instance_mask_delta: Vec::new(),
            material_dirty: Vec::new(),
            max_active_cluster_count: 0,
        }
    }

    /// Slots whose material `AssetId` changed this frame. The binder uses
    /// this (with add/remove) to decide whether the emissive-light set
    /// might have changed.
    #[inline]
    pub fn material_dirty(&self) -> &[GpuEntity] {
        &self.material_dirty
    }

    /// Resolve the GPU material index for every slot whose material changed
    /// this frame (`material_dirty`) and push each `[slot, material_id]` into
    /// the `material` scatter delta. Runs after material-slot assignment,
    /// before the column scatter.
    pub fn resolve_material_ids(&mut self, material_slots: &MaterialSlots) {
        for i in 0..self.material_dirty.len() {
            let slot = self.material_dirty[i];
            let asset_id = self.instance_material_asset_ids[slot.0 as usize];
            let id = material_slots.slot_of(asset_id).unwrap_or(0);
            push_delta_record(&mut self.material_delta, slot, id);
        }
    }

    /// Re-resolve **every** active instance's GPU material index — used when
    /// the material-slot map changed (e.g. a material streamed in after its
    /// instances bound). Each `[slot, material_id]` is pushed to the `material`
    /// scatter delta (re-scatter all; the GPU buffer has no CPU mirror to diff
    /// against). O(active), only on material-set changes (load time).
    pub fn resolve_all_material_ids(&mut self, material_slots: &MaterialSlots) {
        for i in 0..self.active_slots.len() {
            let slot = self.active_slots[i];
            let asset_id = self.instance_material_asset_ids[slot.0 as usize];
            let id = material_slots.slot_of(asset_id).unwrap_or(0);
            push_delta_record(&mut self.material_delta, slot, id);
        }
    }

    /// Slots whose entity was released (despawned) this frame — their
    /// persistent per-slot resources (e.g. the BLAS region keyed on
    /// the slot in `blas_rebuild`) must be freed by their owners.
    #[inline]
    pub fn released_slots(&self) -> &[GpuEntity] {
        &self.released_slots
    }

    /// Number of instances visible this frame.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_slots.len()
    }

    /// Iterator over this frame's active slots, in extract order.
    #[inline]
    pub fn active_slots(&self) -> &[GpuEntity] {
        &self.active_slots
    }

    /// Slots that need a full `WRITE_INSTANCE` this frame: newly added
    /// (or reused) slots and slots whose transform changed.
    #[inline]
    pub fn added_slots(&self) -> &[GpuEntity] {
        &self.added_slots
    }
    /// Active slots needing an explicit PTLAS re-write for a CPU identity
    /// change `fill_incremental` can't detect (a cull-mask change). Transform
    /// moves are detected GPU-side and are NOT here.
    #[inline]
    pub fn rewrite_slots(&self) -> &[GpuEntity] {
        &self.rewrite_slots
    }
    /// Slots despawned this frame — must be removed from the PTLAS
    /// (`WRITE_INSTANCE` with a null AS address).
    #[inline]
    pub fn disabled_slots(&self) -> &[GpuEntity] {
        &self.disabled_slots
    }

    /// High-water mark of allocated slot indices. The partitioned TLAS
    /// must be sized for this many instances, since `instance_index`
    /// (== slot id) can be any value in `0..slot_high_water`.
    #[inline]
    pub fn slot_high_water(&self) -> u32 {
        self.next_slot
    }

    /// Worst-case per-instance cluster count across all bound instances
    /// — for `blas_rebuild` scratch / cluster-AS build sizing. O(1):
    /// a running max maintained at bind (see `max_active_cluster_count`).
    /// May over-estimate after despawn of the largest mesh (safe — only
    /// over-sizes scratch), avoiding a per-frame O(active) rescan.
    #[inline]
    pub fn max_cluster_count(&self) -> u32 {
        self.max_active_cluster_count
    }

    /// The worst-case clusters a single bucket's BLAS can contain (an
    /// object-space cut covers at most the whole mesh), floored to 1.
    ///
    /// **Single source of truth.** The selector's per-bucket ref-list cap,
    /// `blas_rebuild`'s build sizing, and `blas_sharing`'s per-geometry
    /// BLAS region stride MUST all derive from this same value or the
    /// build over-runs the allocated refs / region (a GPU OOB write, not a
    /// panic). Call this everywhere instead of re-deriving
    /// `max_cluster_count().max(1)`.
    #[inline]
    pub fn max_clusters_per_bucket(&self) -> u32 {
        self.max_active_cluster_count.max(1)
    }

    /// This frame's `group_base` / `lod_input` / `geometry_id` / `material`
    /// This frame's material-id scatter delta — `[slot, material_id]` records
    /// pushed when [`resolve_instance_material_ids`] resolves a changed slot.
    /// Uploaded verbatim by [`super::gpu_instances::MaterialColumn`].
    #[inline]
    pub fn material_delta(&self) -> &[u32] {
        &self.material_delta
    }
    /// This frame's cull-mask scatter delta — `[slot, mask]` records pushed at
    /// bind and on a `RenderLayers` change. Uploaded verbatim by
    /// [`super::gpu_instances::InstanceMaskColumn`].
    #[inline]
    pub fn instance_mask_delta(&self) -> &[u32] {
        &self.instance_mask_delta
    }

    /// Allocate a fresh slot. Grows the persistent per-slot vectors to cover
    /// the new index here, once — so the change-driven path can read/write any
    /// allocated slot directly without per-call bounds growth.
    fn allocate_slot(&mut self, mesh_pointers: SlotMeshPointers) -> GpuEntity {
        let slot_index = self.free_slots.pop().unwrap_or_else(|| {
            let s = self.next_slot;
            self.next_slot += 1;
            s
        });
        let slot = GpuEntity(slot_index);
        let idx = slot_index as usize;

        // Grow the persistent per-slot storage to cover `idx`. Reused slots
        // (popped from the free-list) are already in range. The per-column
        // values are no longer mirrored CPU-side — they live only on the GPU.
        if idx >= self.slot_mesh_pointers.len() {
            self.slot_mesh_pointers.resize(idx + 1, None);
            ensure_indexed(&mut self.instance_material_asset_ids, idx, AssetId::default());
            ensure_indexed(&mut self.instance_masks, idx, 0);
            ensure_indexed(&mut self.slot_active_pos, idx, NOT_ACTIVE);
        }
        self.slot_mesh_pointers[idx] = Some(mesh_pointers);
        slot
    }

    /// Clear this frame's deltas. Runs after the deltas' consumers (the
    /// column scatter / BLAS / PTLAS passes), so the next frame's extract
    /// + despawn observer accumulate into a clean slate. `active_slots`
    /// and the per-slot data persist (event-driven model).
    fn clear_deltas(&mut self) {
        self.added_slots.clear();
        self.rewrite_slots.clear();
        self.disabled_slots.clear();
        self.released_slots.clear();
        self.material_delta.clear();
        self.instance_mask_delta.clear();
        self.material_dirty.clear();
    }

    /// Bind a newly-seen instance: allocate a slot, add it to the persistent active
    /// set, scatter its cull mask, and flag its material for resolution. The bind-only
    /// per-column values (`group_base` / `lod_input` / `geometry_id` / `node_slot`) are
    /// written GPU-side by the reconcile pass from the journal `UPSERT` emitted
    /// alongside this bind (the caller pushes it). The world transform is produced
    /// GPU-side too (transform gather → `TransformColumn`).
    fn bind(
        &mut self,
        ptrs: SlotMeshPointers,
        material_asset_id: AssetId<SolariMaterial>,
        cull_mask: u32,
    ) -> GpuEntity {
        let slot = self.allocate_slot(ptrs);
        // Running worst-case cluster count (monotonic; see field doc).
        self.max_active_cluster_count = self.max_active_cluster_count.max(ptrs.cluster_count);
        self.instance_material_asset_ids[slot.0 as usize] = material_asset_id;
        self.instance_masks[slot.0 as usize] = cull_mask;
        // Append to the persistent active set.
        self.slot_active_pos[slot.0 as usize] = self.active_slots.len() as u32;
        self.active_slots.push(slot);
        self.added_slots.push(slot);
        // The cull mask is CPU-owned (it changes post-bind via `update`).
        push_delta_record(&mut self.instance_mask_delta, slot, cull_mask);
        // New material → resolve its GPU index this frame.
        self.material_dirty.push(slot);
        slot
    }

    /// Update an already-bound slot whose transform / material changed. The world
    /// transform itself is produced GPU-side (the gather writes `TransformColumn`);
    /// here we only record the move (for the PTLAS) + any material/mask change.
    fn update(
        &mut self,
        slot: GpuEntity,
        material_asset_id: AssetId<SolariMaterial>,
        cull_mask: u32,
    ) {
        let idx = slot.0 as usize;
        // Re-resolve the GPU material index only on an actual material swap.
        // A material change needs no PTLAS re-write: the TLAS record carries
        // `instance_id = slot` (stable) and the material is resolved at trace
        // time, not baked in.
        if self.instance_material_asset_ids[idx] != material_asset_id {
            self.instance_material_asset_ids[idx] = material_asset_id;
            self.material_dirty.push(slot);
        }
        // Re-scatter the cull mask only when `RenderLayers` actually changed —
        // and re-specify the instance in the PTLAS, since the mask IS baked
        // into its TLAS record (`fill_incremental`'s transform compare can't
        // see a mask-only change).
        if self.instance_masks[idx] != cull_mask {
            self.instance_masks[idx] = cull_mask;
            push_delta_record(&mut self.instance_mask_delta, slot, cull_mask);
            self.rewrite_slots.push(slot);
        }
    }

    /// Free a `slot` whose render entity despawned: O(1) swap-remove from
    /// the active set + record disabled (PTLAS removal) / released
    /// (per-slot resource free) deltas. Called by the [`free_cluster_slot`]
    /// observer with the slot read off the despawning component.
    fn despawn_slot(&mut self, slot: GpuEntity) {
        let idx = slot.0 as usize;
        if let Some(ptr) = self.slot_mesh_pointers.get_mut(idx) {
            *ptr = None;
        }
        let pos = self.slot_active_pos[idx];
        if pos != NOT_ACTIVE {
            let pos = pos as usize;
            let last = self.active_slots.len() - 1;
            self.active_slots.swap_remove(pos);
            if pos != last {
                let moved = self.active_slots[pos];
                self.slot_active_pos[moved.0 as usize] = pos as u32;
            }
            self.slot_active_pos[idx] = NOT_ACTIVE;
        }
        self.free_slots.push(slot.0);
        self.disabled_slots.push(slot);
        self.released_slots.push(slot);
    }

    /// Per-slot material `AssetId<SolariMaterial>`. The raytracing
    /// scene binder resolves this to a local material-array index
    /// each frame.
    #[inline]
    pub fn instance_material_asset_id(&self, slot: GpuEntity) -> AssetId<SolariMaterial> {
        self.instance_material_asset_ids[slot.0 as usize]
    }

    /// Total triangle count across every cluster in this instance's
    /// mesh — sampled uniformly when this instance is an emissive
    /// light source.
    #[inline]
    pub fn instance_total_triangle_count(&self, slot: GpuEntity) -> u32 {
        self.slot_mesh_pointers[slot.0 as usize]
            .map(|p| p.total_triangle_count)
            .unwrap_or(0)
    }

    /// Per-instance `(cluster_base, cluster_count)` covering this
    /// instance's slice of the global cluster pool. Exposed for the
    /// raytracing scene binder.
    #[inline]
    pub fn instance_cluster_range(&self, slot: GpuEntity) -> (u32, u32) {
        self.slot_mesh_pointers[slot.0 as usize]
            .map(|p| (p.cluster_base.0, p.cluster_count))
            .unwrap_or((0, 0))
    }
}

fn ensure_indexed<T: Clone>(v: &mut Vec<T>, idx: usize, fill: T) {
    if idx >= v.len() {
        v.resize(idx + 1, fill);
    }
}

/// Append one `[slot, value-words…]` scatter record to a per-column delta —
/// `slot` then the `Pod` value's `u32` words, the exact layout the
/// [`crate::ecs_gpu::column::GpuColumn`] uploads and the scatter shader reads.
#[inline]
fn push_delta_record<T: Pod>(delta: &mut Vec<u32>, slot: GpuEntity, value: T) {
    delta.push(slot.0);
    delta.extend_from_slice(bytemuck::cast_slice(core::slice::from_ref(&value)));
}


pub fn init_instance_manager(mut commands: Commands, _render_device: Res<RenderDevice>) {
    commands.insert_resource(InstanceManager::new());
}

/// Main-world set of RT-mesh entities whose *instance identity* changed and need
/// (re)processing by the extract — populated by lifecycle observers
/// ([`mark_instance_added`] / [`mark_instance_material_changed`] /
/// [`mark_instance_layers_changed`]), drained by [`flush_cluster_instances`].
///
/// Replaces the old per-frame `Or<(Added, Changed, Changed)>` query, which had to
/// scan every RT-mesh entity's change-ticks each frame just to find the few that
/// changed (~0.76ms over 2M instances, even when nothing changed). Observers fire
/// only on real events, so in steady state — things only *moving* — this set
/// stays empty and the extract does zero work over zero entities.
///
/// Movement is deliberately absent: it's handled entirely on the GPU (transform
/// table → gather; the PTLAS fill detects moves by world-transform compare), so a
/// moving entity's slot / material / mask / mesh / node_slot are unchanged and
/// need no CPU re-pack.
#[derive(Resource, Default)]
pub struct RtInstanceChanges(EntityHashSet);

/// Observer: a newly-added [`RaytracingMesh3d`] needs binding.
pub fn mark_instance_added(add: On<Add, RaytracingMesh3d>, mut changes: ResMut<RtInstanceChanges>) {
    changes.0.insert(add.entity);
}

/// Observer: a material (re)assignment on an RT mesh needs a material re-resolve.
/// Filtered to RT-mesh entities (a non-RT `SolariMaterial3d` insert is harmless —
/// the flush's fetch skips it — but this keeps the set tight). A material inserted
/// *before* its `RaytracingMesh3d` is still caught: [`mark_instance_added`] queues
/// the entity on the mesh add, and the flush reads the by-then-present material.
pub fn mark_instance_material_changed(
    insert: On<Insert, SolariMaterial3d>,
    is_rt: Query<(), With<RaytracingMesh3d>>,
    mut changes: ResMut<RtInstanceChanges>,
) {
    if is_rt.get(insert.entity).is_ok() {
        changes.0.insert(insert.entity);
    }
}

/// Observer: a [`RenderLayers`] change on an RT mesh updates its cull mask.
/// Filtered to RT meshes (cameras / lights also carry `RenderLayers`).
pub fn mark_instance_layers_changed(
    insert: On<Insert, RenderLayers>,
    is_rt: Query<(), With<RaytracingMesh3d>>,
    mut changes: ResMut<RtInstanceChanges>,
) {
    if is_rt.get(insert.entity).is_ok() {
        changes.0.insert(insert.entity);
    }
}

/// Unfiltered fetch of one entity's data, keyed by main entity. `RenderEntity`
/// yields the synced render entity (where the slot/`RaytracingGpuEntity` lives).
type ExtractGetData = (
    RenderEntity,
    &'static RaytracingMesh3d,
    Option<&'static SolariMaterial3d>,
    Option<&'static RenderLayers>,
    Option<&'static GpuSlot<TransformGraph>>,
);

/// Render entity → its GPU [`GpuEntity`] slot. The persistent slot index,
/// keyed for O(1) lookup the way bevy's `RenderMeshInstances` keys
/// `MainEntity → input_index`. Inserted at bind, removed by the
/// [`free_cluster_slot`] despawn observer; read by [`flush_cluster_instances`]
/// to tell a bind (`None`) from an update (`Some`). The per-entity
/// `RaytracingGpuEntity` component drives that despawn observer.
///
/// **Deliberately not on `ecs_gpu`'s [`GpuSlotAllocator`]/[`GpuSlot<T>`].** Those
/// are a *main-world* pattern (the transform/lights tables assign slots in
/// `PostUpdate` and key the despawn off a main-entity component). The instance
/// table is *render-world*: a slot is a GPU index tied to `ClusterMesh` residency,
/// its lifecycle hangs off the **render** entity (`RaytracingGpuEntity`), and the
/// `InstanceManager` has render-side bookkeeping (`active_slots`, PTLAS
/// disabled/released deltas) that a freed slot must update — none of which the
/// main-world allocator models. Forcing the fit would need cross-world freed-slot
/// extraction for zero runtime gain. Since Stage 1 made this map off-hot-path
/// (touched only on bind/material/layer/despawn, never per-frame), there's no
/// perf reason to convert it either. Keep render-world.
///
/// [`GpuSlotAllocator`]: crate::ecs_gpu::GpuSlotAllocator
/// [`GpuSlot<T>`]: crate::ecs_gpu::GpuSlot
#[derive(Resource, Default)]
pub struct RtSlotMap(EntityHashMap<GpuEntity>);

/// `ExtractSchedule`: drain the observer-populated [`RtInstanceChanges`] set and
/// apply each entity to the render-world [`InstanceManager`]. For each changed
/// entity: resolve its slot ([`RtSlotMap`]) — `Some` → `update` (material / mask),
/// `None` → `bind` (or park in `pending` until its `ClusterMesh` streams in),
/// recording the new slot in [`RtSlotMap`] + tagging the render entity with
/// [`RaytracingGpuEntity`] for the [`free_cluster_slot`] despawn observer.
///
/// No per-frame scan: the change set is empty in steady state (movement is
/// GPU-driven), so this touches only genuinely changed entities — usually none.
pub fn flush_cluster_instances(
    mut commands: Commands,
    mut manager: ResMut<InstanceManager>,
    mut cluster_meshes: ResMut<ClusterMeshManager>,
    mut clas_arena: ResMut<ClasArena>,
    mut slot_map: ResMut<RtSlotMap>,
    mut main_world: ResMut<MainWorld>,
    // GPU instance-change journal — an absolute-state UPSERT is appended per bind
    // (the REMOVE counterpart is emitted by `free_cluster_slot`). Feeds the GPU
    // reconcile. `Option` so non-solari devices (no journal) are a no-op.
    mut journal: Option<ResMut<RtJournal>>,
    // Resolves the material `AssetId` to its stable GPU slot for the journal
    // record. Lags a frame for a brand-new material (slot allocated in Prepare);
    // the reconcile tolerates a 0 until the next bind/upsert refreshes it.
    material_slots: Option<Res<MaterialSlots>>,
    // Main entities whose `ClusterMesh` asset wasn't resident when added —
    // retried until ready. Touched only during load, never on a move.
    mut pending: Local<HashSet<Entity>>,
    #[allow(clippy::type_complexity)] mut system_state: Local<
        Option<
            SystemState<(
                Query<'static, 'static, ExtractGetData>,
                Res<'static, AssetServer>,
                ResMut<'static, Assets<ClusterMesh>>,
                MessageReader<'static, 'static, AssetEvent<ClusterMesh>>,
                ResMut<'static, RtInstanceChanges>,
            )>,
        >,
    >,
) {
    if system_state.is_none() {
        *system_state = Some(SystemState::new(&mut main_world));
    }
    let state = system_state.as_mut().unwrap();
    let Ok((get_one, asset_server, mut assets, mut asset_events, mut changes)) =
        state.get_mut(&mut main_world)
    else {
        return;
    };

    // Asset-level evictions before any per-frame work.
    for ev in asset_events.read() {
        if let AssetEvent::Unused { id } | AssetEvent::Modified { id } = ev {
            cluster_meshes.remove(&id);
            clas_arena.remove(&id);
        }
    }

    // Drain this frame's observer-flagged changes (binds + material / cull-layer
    // swaps). Empty in steady state, since movement never enters here.
    for main_entity in changes.0.drain() {
        let Ok((render_entity, mesh, material, render_layers, node_slot)) = get_one.get(main_entity)
        else {
            continue; // despawned, or not (yet) an RT mesh — skip.
        };
        let material_id = material.map(|m| m.0.id()).unwrap_or_default();
        let cull_mask = render_layers_to_mask(render_layers);
        let node_slot = node_slot.map(GpuSlot::index).unwrap_or(u32::MAX);
        if let Some(slot) = slot_map.0.get(&render_entity).copied() {
            // Already bound → material / cull-mask update (no-op if unchanged).
            manager.update(slot, material_id, cull_mask);
        } else if let Some((slot, pointers)) = try_bind_instance(
            &mut manager,
            &mut cluster_meshes,
            &mut assets,
            &asset_server,
            mesh.id(),
            material_id,
            cull_mask,
        ) {
            slot_map.0.insert(render_entity, slot);
            commands
                .entity(render_entity)
                .insert(RaytracingGpuEntity(slot));
            push_instance_upsert(
                journal.as_deref_mut(),
                material_slots.as_deref(),
                slot,
                &pointers,
                material_id,
                node_slot,
                cull_mask,
            );
            // Bound here this frame — drop any stale `pending` entry so the
            // retry below doesn't bind it a second time.
            pending.remove(&main_entity);
        } else {
            pending.insert(main_entity);
        }
    }

    // Retry entities still waiting on their asset. Despawned-while-pending
    // entities drop out (the `get_one` fetch fails).
    if !pending.is_empty() {
        for main_entity in core::mem::take(&mut *pending) {
            let Ok((render_entity, mesh, material, render_layers, node_slot)) =
                get_one.get(main_entity)
            else {
                continue;
            };
            // Already bound (e.g. it moved + bound before the asset
            // resolved)? Nothing to do.
            if slot_map.0.contains_key(&render_entity) {
                continue;
            }
            let material_id = material.map(|m| m.0.id()).unwrap_or_default();
            let cull_mask = render_layers_to_mask(render_layers);
            let node_slot = node_slot.map(GpuSlot::index).unwrap_or(u32::MAX);
            if let Some((slot, pointers)) = try_bind_instance(
                &mut manager,
                &mut cluster_meshes,
                &mut assets,
                &asset_server,
                mesh.id(),
                material_id,
                cull_mask,
            ) {
                slot_map.0.insert(render_entity, slot);
                commands
                    .entity(render_entity)
                    .insert(RaytracingGpuEntity(slot));
                push_instance_upsert(
                    journal.as_deref_mut(),
                    material_slots.as_deref(),
                    slot,
                    &pointers,
                    material_id,
                    node_slot,
                    cull_mask,
                );
            } else {
                pending.insert(main_entity);
            }
        }
    }
}

/// Append an absolute `UPSERT` for a freshly-bound instance into the GPU instance
/// journal, keyed by the stable render entity (the same key `free_cluster_slot`'s
/// REMOVE uses). No-op when the journal is absent (non-solari device).
#[allow(clippy::too_many_arguments)]
fn push_instance_upsert(
    journal: Option<&mut RtJournal>,
    material_slots: Option<&MaterialSlots>,
    slot: GpuEntity,
    pointers: &SlotMeshPointers,
    material: AssetId<SolariMaterial>,
    node_slot: u32,
    cull_mask: u32,
) {
    let Some(journal) = journal else {
        return;
    };
    let material_id = material_slots.and_then(|ms| ms.slot_of(material)).unwrap_or(0);
    journal.push(InstanceJournalRecord::upsert(
        slot.0,
        pointers.geometry_id,
        material_id,
        node_slot,
        cull_mask,
        0,
        pointers.group_base.0,
        pointers.cluster_base.0,
        pointers.cluster_count,
        pointers.root_group.0,
    ));
}

/// Upload the `ClusterMesh` (if needed) and bind a fresh slot, returning
/// it. `None` when the asset hasn't streamed in yet (caller parks the
/// entity in `pending`).
fn try_bind_instance(
    manager: &mut InstanceManager,
    cluster_meshes: &mut ClusterMeshManager,
    assets: &mut Assets<ClusterMesh>,
    asset_server: &AssetServer,
    mesh_id: AssetId<ClusterMesh>,
    material: AssetId<SolariMaterial>,
    cull_mask: u32,
) -> Option<(GpuEntity, SlotMeshPointers)> {
    if asset_server.is_managed(mesh_id) && !asset_server.is_loaded_with_dependencies(mesh_id) {
        return None;
    }
    let upload: ClusterMeshUpload = cluster_meshes.queue_upload_if_needed(mesh_id, assets);
    let pointers = SlotMeshPointers {
        cluster_base: upload.cluster_base,
        group_base: upload.group_base,
        root_group: upload.root_group,
        cluster_count: upload.cluster_count,
        total_triangle_count: upload.total_triangle_count,
        geometry_id: upload.geometry_id,
    };
    let slot = manager.bind(pointers, material, cull_mask);
    Some((slot, pointers))
}

/// Observer: free a slot when its render entity despawns (the sync world
/// despawns the render entity when its main entity goes away). Reads the
/// slot off the still-present component value, frees it in the
/// [`InstanceManager`], and drops the render entity's [`RtSlotMap`] entry so
/// the flush stops resolving a stale slot.
pub fn free_cluster_slot(
    remove: On<Remove, RaytracingGpuEntity>,
    slots: Query<&RaytracingGpuEntity>,
    mut manager: ResMut<InstanceManager>,
    mut slot_map: ResMut<RtSlotMap>,
    journal: Option<ResMut<RtJournal>>,
) {
    if let Ok(slot) = slots.get(remove.entity) {
        manager.despawn_slot(slot.0);
        // Emit an absolute REMOVE into the GPU instance journal for this slot — the
        // reconcile clears its PTLAS presence.
        if let Some(mut journal) = journal {
            journal.push(InstanceJournalRecord::remove(slot.0 .0));
        }
    }
    slot_map.0.remove(&remove.entity);
}

/// `Render::Prepare` (after `prepare_material_slots`, before
/// `prepare_gpu_instances`): cache the GPU material index for every
/// slot whose material changed this frame. The column scatter then reads
/// the cached `u32` instead of re-hashing the material asset id per slot.
pub fn resolve_instance_material_ids(
    mut instances: Option<ResMut<InstanceManager>>,
    material_slots: Option<Res<MaterialSlots>>,
    // Last material-slot generation we resolved against.
    mut last_generation: Local<u64>,
) {
    let (Some(instances), Some(material_slots)) = (instances.as_deref_mut(), material_slots) else {
        return;
    };
    let generation = material_slots.generation();
    if generation != *last_generation {
        // Material set changed (a material loaded/unloaded) — re-resolve
        // every active instance so any that bound before their material
        // was resident pick up the right index.
        *last_generation = generation;
        instances.resolve_all_material_ids(&material_slots);
    } else {
        instances.resolve_material_ids(&material_slots);
    }
}

/// `Render` (after the delta consumers): clear this frame's deltas so the
/// next frame's extract + despawn observer accumulate cleanly.
pub fn clear_instance_deltas(mut manager: ResMut<InstanceManager>) {
    manager.clear_deltas();
}

