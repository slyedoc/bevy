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

/// Per-entity slot handle into table `T` — the component-as-index. Read for free
/// off the entity (array-indexed), never hashed.
///
/// Slot identity carries no generation: the one consumer that needs ABA-safe
/// identity, the `GlobalTransform` readback, stamps the owning `Entity` into each
/// record and resolves by it (see `transform::readback`), so a recycled slot is
/// distinguished by a *different entity*, not a counter.
#[derive(Component)]
pub struct GpuSlot<T: GpuSlotTable> {
    index: u32,
    _marker: PhantomData<fn() -> T>,
}

impl<T: GpuSlotTable> GpuSlot<T> {
    /// This entity's dense slot in table `T`.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
}

/// Slot allocator core: a monotonic counter + free-list, split out from
/// [`GpuSlotAllocator`] so the recycle logic is unit-testable without the
/// `GpuSlotTable` machinery. A freed slot is returned to the pool immediately —
/// the slot-recycle ABA on the `GlobalTransform` readback is closed by stamping the
/// owning `Entity` into each readback record (entity-keyed, lag-independent), so no
/// reuse-deferral is needed here.
#[derive(Default)]
struct SlotFreeList {
    next: u32,
    /// Freed slots, reusable immediately (LIFO).
    free: Vec<u32>,
}

impl SlotFreeList {
    /// Reuse a freed slot if one is available, else bump the high-water.
    fn allocate(&mut self) -> u32 {
        self.free.pop().unwrap_or_else(|| {
            let slot = self.next;
            self.next += 1;
            slot
        })
    }

    /// Return `slot` to the reusable pool.
    fn free(&mut self, slot: u32) {
        self.free.push(slot);
    }
}

/// Main-world slot allocator for table `T`: a monotonic counter + free-list
/// ([`SlotFreeList`]). The forward [`GpuSlot<T>`] component covers entity→slot; the
/// reverse direction is no longer tracked here — the `GlobalTransform` readback
/// carries the owning `Entity` and resolves it directly (see `transform::readback`).
#[derive(Resource)]
pub struct GpuSlotAllocator<T: GpuSlotTable> {
    slots: SlotFreeList,
    _marker: PhantomData<fn() -> T>,
}

impl<T: GpuSlotTable> Default for GpuSlotAllocator<T> {
    fn default() -> Self {
        Self {
            slots: SlotFreeList::default(),
            _marker: PhantomData,
        }
    }
}

impl<T: GpuSlotTable> GpuSlotAllocator<T> {
    /// Allocate a slot (reusing a freed one if any).
    fn allocate(&mut self) -> u32 {
        self.slots.allocate()
    }

    /// Return `slot` to the reusable pool.
    fn free(&mut self, slot: u32) {
        self.slots.free(slot);
    }

    /// Slot high-water — every slot `< high_water` has been handed out, so the
    /// table's columns must size their buffers to cover this many.
    #[inline]
    pub fn high_water(&self) -> u32 {
        self.slots.next
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
        let index = allocator.allocate();
        commands.entity(entity).insert(GpuSlot::<T> {
            index,
            _marker: PhantomData,
        });
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
        allocator.free(slot.index());
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

    /// Reclaim trailing free slots: peel every freed index that sits at the top of
    /// the `[0, len)` range so [`len`](Self::len) tracks the live high-water rather
    /// than a peak that only ever ratchets up. Holes below the live range are kept
    /// (reused by the next [`allocate`](Self::allocate)). Returns the new `len`.
    ///
    /// `len` otherwise never shrinks — a transient spike in simultaneously-live keys
    /// raises it permanently. A consumer that caps on `len` (e.g. a PTLAS
    /// `partition_count <= max_partition_count` build guard) must `compact` first, or
    /// a one-frame spike past the cap bricks the build long after the live set
    /// shrank back. Does **not** bump `generation`: it only removes never-resolved
    /// slots, so no cached resolution is invalidated. `O(free·log free)`.
    pub fn compact(&mut self) -> u32 {
        if self.free.is_empty() {
            return self.next;
        }
        // Holes ascending; peel any that abut the top of the range.
        self.free.sort_unstable();
        while let Some(&top) = self.free.last() {
            if top + 1 == self.next {
                self.free.pop();
                self.next -= 1;
            } else {
                break;
            }
        }
        self.next
    }
}

#[cfg(test)]
mod tests {
    use super::{SlotFreeList, SlotPool};

    #[test]
    fn compact_reclaims_only_the_trailing_free_run() {
        let mut p: SlotPool<u32> = SlotPool::default();
        for k in 0..5 {
            p.allocate(k); // slots 0..5, len == 5
        }
        assert_eq!(p.len(), 5);
        // Free a tail run (3,4) and an interior hole (1). 0 and 2 stay resident.
        p.free(4);
        p.free(3);
        p.free(1);
        assert_eq!(p.len(), 5, "len does not shrink on its own");

        // Compaction peels the contiguous top run (4 then 3) but keeps the interior
        // hole at 1 — resident slots 0 and 2 are untouched.
        assert_eq!(p.compact(), 3);
        assert_eq!(p.len(), 3);
        assert_eq!(p.slot_of(0), Some(0));
        assert_eq!(p.slot_of(2), Some(2));
        assert_eq!(p.slot_of(1), None);
        // The interior hole is still reused before the high-water grows.
        let gen_before = p.generation();
        assert_eq!(p.allocate(99), 1, "interior hole reused");
        assert_eq!(p.allocate(100), 3, "then bump the (compacted) high-water");
        assert_eq!(p.len(), 4);
        assert!(
            p.generation() > gen_before,
            "allocate bumps generation; compact did not"
        );
    }

    #[test]
    fn compact_all_freed_drops_to_zero() {
        let mut p: SlotPool<u32> = SlotPool::default();
        for k in 0..1000 {
            p.allocate(k);
        }
        for k in 0..1000 {
            p.free(k);
        }
        assert_eq!(p.len(), 1000, "high-water ratchets until compaction");
        assert_eq!(p.compact(), 0, "everything freed → len collapses to 0");
        assert!(p.is_empty());
        assert_eq!(p.allocate(7), 0, "fresh allocation starts from 0 again");
    }


    #[test]
    fn allocate_is_monotonic_until_reuse() {
        let mut f = SlotFreeList::default();
        assert_eq!(f.allocate(), 0);
        assert_eq!(f.allocate(), 1);
        assert_eq!(f.allocate(), 2);
        assert_eq!(f.next, 3);
        assert_eq!(f.free.len(), 0);
    }

    #[test]
    fn freed_slot_is_reused_immediately() {
        let mut f = SlotFreeList::default();
        let a = f.allocate(); // 0
        let b = f.allocate(); // 1
        f.free(a);
        f.free(b);
        // Reuse is immediate (the readback's ABA is entity-keyed, not reuse-deferred):
        // the next allocations drain the free list (LIFO) before bumping the high-water.
        assert_eq!(f.allocate(), 1);
        assert_eq!(f.allocate(), 0);
        assert_eq!(f.allocate(), 2, "free list exhausted → bump high-water");
        assert_eq!(f.next, 3);
    }

    #[test]
    fn mass_free_then_reuse() {
        // Mirrors a regenerate: free a large batch, confirm it's all reusable at once
        // (no quarantine delay) before the high-water grows again.
        let mut f = SlotFreeList::default();
        let slots: Vec<u32> = (0..1000).map(|_| f.allocate()).collect();
        for &s in &slots {
            f.free(s);
        }
        assert_eq!(f.free.len(), 1000);
        for _ in 0..1000 {
            assert!(f.allocate() < 1000, "should reuse the freed batch, not grow");
        }
        assert_eq!(f.allocate(), 1000, "batch exhausted → bump high-water");
    }
}
