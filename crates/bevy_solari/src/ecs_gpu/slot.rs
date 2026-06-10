//! Component-indexed GPU-table slots — entity→slot as a free, array-indexed ECS
//! component read instead of a per-frame `EntityHashMap` lookup (which is *not*
//! free at 1M+ and costs on every changed entity, i.e. every mover/joint).
//!
//! A member entity carries [`GpuSlot<T>`] holding its dense slot. Allocation is
//! an `Added`-only main-world system ([`assign_gpu_slots`]) — members already
//! slotted are skipped, so the steady-state per-frame cost is **zero**. Freeing
//! reads the slot off the still-present component in an [`On<Remove>`] observer
//! ([`free_gpu_slot`]) — the component *is* the entity→slot store, so no reverse
//! map is needed. This is the same pattern as the instance path's
//! `RaytracingGpuEntity` + `free_cluster_slot`, generalized over the table.
//!
//! The generic machinery here is written once; [`crate::gpu_table`] stamps out
//! only the per-table types and wiring.

use core::hash::Hash;
use core::marker::PhantomData;

use bevy_ecs::{
    component::Component,
    entity::Entity,
    lifecycle::Remove,
    observer::On,
    query::{QueryFilter, Without},
    resource::Resource,
    system::{Commands, Query, ResMut},
};
use bevy_platform::{collections::HashMap, hash::FixedHasher};
use bytemuck::Pod;

use super::column::GpuTable;

/// A GPU table whose slots are indexed by the [`GpuSlot<Self>`] component. The
/// table type is the render-world delta-storage resource (it is also its own
/// [`GpuTable`]); `Members` selects which entities get a slot.
pub trait GpuSlotTable: GpuTable {
    /// Entities that are members of this table (each is assigned a slot).
    type Members: QueryFilter + Send + Sync + 'static;
    /// Store the slot high-water extracted from the main-world allocator, so the
    /// table's columns size their buffers to cover every handed-out slot.
    fn set_high_water(&mut self, high_water: u32);
}

/// Per-entity slot index into table `T` — the component-as-index. Read for free
/// off the entity (array-indexed), never hashed.
#[derive(Component)]
pub struct GpuSlot<T: GpuSlotTable>(u32, PhantomData<fn() -> T>);

impl<T: GpuSlotTable> GpuSlot<T> {
    /// This entity's dense slot in table `T`.
    #[inline]
    pub fn index(&self) -> u32 {
        self.0
    }
}

/// Main-world slot allocator for table `T`: a monotonic counter + free-list. The
/// counter is the table high-water (every slot `< next` has been handed out).
///
/// Also keeps a `slot → Entity` reverse map. The forward [`GpuSlot<T>`] component
/// covers entity→slot, but a GPU→CPU readback (the transform table's
/// `GlobalTransform` writeback) arrives keyed by slot and must resolve the owning
/// entity — that's the reverse direction. The map is sized by slot; a freed
/// slot's entry is left stale (overwritten when the slot is reused), so a reader
/// must only trust it for currently-live slots.
#[derive(Resource)]
pub struct GpuSlotAllocator<T: GpuSlotTable> {
    next: u32,
    free: Vec<u32>,
    reverse: Vec<Entity>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: GpuSlotTable> Default for GpuSlotAllocator<T> {
    fn default() -> Self {
        Self {
            next: 0,
            free: Vec::new(),
            reverse: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<T: GpuSlotTable> GpuSlotAllocator<T> {
    /// Allocate a slot for `entity` (reusing a freed slot if any) and record the
    /// `slot → entity` reverse mapping.
    fn allocate(&mut self, entity: Entity) -> u32 {
        let slot = self.free.pop().unwrap_or_else(|| {
            let slot = self.next;
            self.next += 1;
            slot
        });
        let idx = slot as usize;
        if idx >= self.reverse.len() {
            self.reverse.resize(idx + 1, Entity::PLACEHOLDER);
        }
        self.reverse[idx] = entity;
        slot
    }

    /// Slot high-water — every slot `< high_water` has been handed out, so the
    /// table's columns must size their buffers to cover this many.
    #[inline]
    pub fn high_water(&self) -> u32 {
        self.next
    }

    /// The entity that currently owns `slot`, or `None` if out of range. Stale
    /// for freed-and-not-yet-reused slots — only trustworthy for live slots.
    #[inline]
    pub fn entity(&self, slot: u32) -> Option<Entity> {
        self.reverse.get(slot as usize).copied()
    }
}

/// Main world: assign a slot to every new member of table `T` (matches
/// `T::Members`, no `GpuSlot<T>` yet). O(new) — already-slotted members are
/// skipped, so a frame in which nothing is spawned does no work.
pub fn assign_gpu_slots<T: GpuSlotTable>(
    mut commands: Commands,
    new_members: Query<Entity, (T::Members, Without<GpuSlot<T>>)>,
    mut allocator: ResMut<GpuSlotAllocator<T>>,
) {
    for entity in &new_members {
        let slot = allocator.allocate(entity);
        commands.entity(entity).insert(GpuSlot::<T>(slot, PhantomData));
    }
}

/// Main world: reclaim a slot when its entity loses [`GpuSlot<T>`] / despawns.
/// Reads the slot off the still-present component, so no reverse map is needed.
pub fn free_gpu_slot<T: GpuSlotTable>(
    removed: On<Remove, GpuSlot<T>>,
    slots: Query<&GpuSlot<T>>,
    mut allocator: ResMut<GpuSlotAllocator<T>>,
) {
    if let Ok(slot) = slots.get(removed.entity) {
        allocator.free.push(slot.0);
    }
}

/// Append one `[slot, value-words…]` scatter record — the exact layout a
/// [`GpuColumn`](super::GpuColumn) uploads and the scatter shader reads. Used by
/// hand-written table extracts to fill the macro-generated delta buffers.
#[inline]
pub fn push_record<V: Pod>(delta: &mut Vec<u32>, slot: u32, value: V) {
    delta.push(slot);
    delta.extend_from_slice(bytemuck::cast_slice(core::slice::from_ref(&value)));
}

/// A stable `key → dense slot` allocator: a monotonic counter + free-list, with a
/// `generation` bumped on every allocate/free so consumers that cache resolved
/// slots can tell when to re-resolve.
///
/// The **key-indexed** sibling of [`GpuSlotAllocator`] (which is entity-component
/// indexed): use it where the GPU array is indexed by a non-entity key — e.g. a
/// material `AssetId` ([`MaterialSlots`](crate::material::MaterialSlots)). Slots
/// are reused before `len` grows; a freed slot's old contents are overwritten on
/// reuse, so only trust a slot for a currently-allocated key.
pub struct SlotPool<K> {
    map: HashMap<K, u32, FixedHasher>,
    free: Vec<u32>,
    next: u32,
    generation: u64,
}

impl<K> Default for SlotPool<K> {
    fn default() -> Self {
        Self {
            map: HashMap::default(),
            free: Vec::new(),
            next: 0,
            generation: 0,
        }
    }
}

impl<K: Copy + Eq + Hash> SlotPool<K> {
    /// Stable slot for `key`, if one is allocated.
    #[inline]
    pub fn slot_of(&self, key: K) -> Option<u32> {
        self.map.get(&key).copied()
    }

    /// Monotonic version, bumped on every allocate/free.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Number of slots handed out (== required GPU array length; freed slots leave
    /// holes until reused).
    #[inline]
    pub fn len(&self) -> u32 {
        self.next
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.next == 0
    }

    /// Iterate `(key, slot)` for every currently-allocated key.
    pub fn iter(&self) -> impl Iterator<Item = (K, u32)> + '_ {
        self.map.iter().map(|(k, s)| (*k, *s))
    }

    /// Allocate (or return the existing) stable slot for `key`.
    pub fn allocate(&mut self, key: K) -> u32 {
        if let Some(slot) = self.map.get(&key) {
            return *slot;
        }
        let slot = self.free.pop().unwrap_or_else(|| {
            let s = self.next;
            self.next += 1;
            s
        });
        self.map.insert(key, slot);
        self.generation += 1;
        slot
    }

    /// Free `key`'s slot (reused before `len` grows).
    pub fn free(&mut self, key: K) {
        if let Some(slot) = self.map.remove(&key) {
            self.free.push(slot);
            self.generation += 1;
        }
    }

    /// Allocate a slot for every key in `present` and free any tracked key not in
    /// it (`is_present` answers the mark-and-sweep membership test).
    pub fn reconcile<I, F>(&mut self, present: I, is_present: F)
    where
        I: IntoIterator<Item = K>,
        F: Fn(K) -> bool,
    {
        for key in present {
            self.allocate(key);
        }
        let dropped: Vec<K> = self.map.keys().copied().filter(|k| !is_present(*k)).collect();
        for key in dropped {
            self.free(key);
        }
    }
}
