//! Block → PTLAS-partition-index allocator for the floating-origin acceleration
//! structure (design: `docs/big_space_native.md` §12.1). Each populated cell
//! **block** (`cell >> BLOCK_SHIFT`) maps to a dense PTLAS partition index whose
//! `WRITE_PARTITION_TRANSLATION` carries `(block_origin − origin) × cell_edge`, so
//! instances store only their block-local transform and a floating-origin recenter
//! rewrites `O(partitions)` translations instead of `O(instances)`.
//!
//! The "stable **and** dense" index map is solari's existing [`SlotPool`]: first-touch
//! allocation keeps a resident block's index fixed (so its static instances are never
//! re-written), and a freed index is reused only by a block that streams in *later*
//! (whose instances are written regardless). That is the key distinction from
//! big_space's `PartitionId`, which re-keys *resident* entities on merge/split.
//!
//! INVARIANTS (design §13 — enforce, don't assume):
//! 1. Block presence MUST be derived from "≥ 1 resident instance", never a separate
//!    streaming signal that can flicker — else [`reconcile`](PartitionAllocator::reconcile)
//!    could free then re-key a resident block to a different index.
//! 2. [`partition_count`](PartitionAllocator::partition_count) compacts the high-water
//!    before reporting, so a transient block-count spike cannot permanently inflate the
//!    count past the device `max_partition_count`.
//! 3. Eviction ([`free`](PartitionAllocator::free)) and the translation-op rebuild must
//!    be sequenced in one extract, so the GPU never reads a recycled index against a
//!    stale partition translation.

use bevy_ecs::resource::Resource;
use bevy_math::IVec3;

use crate::ecs_gpu::SlotPool;

/// Bits a cell coordinate is right-shifted by to get its block coordinate — a block
/// is a `2^BLOCK_SHIFT` cube of cells sharing one PTLAS partition (one translation).
/// Coarsening (raising this) trades partition count for a larger block-local transform
/// range; tune against the device `max_partition_count`. v1 = `0` (one cell per block).
pub const BLOCK_SHIFT: u32 = 0;

/// The block coordinate containing `cell` (`cell >> BLOCK_SHIFT`, component-wise).
/// Arithmetic right-shift floors toward −∞, so negative cells block correctly.
#[inline]
pub fn block_of(cell: IVec3) -> IVec3 {
    IVec3::new(
        cell.x >> BLOCK_SHIFT,
        cell.y >> BLOCK_SHIFT,
        cell.z >> BLOCK_SHIFT,
    )
}

/// Render-world resource: stable, dense block → PTLAS-partition-index map, keyed by
/// block coordinate (see [`block_of`]); the value indexes `0..partition_count()`.
/// See the module docs for the invariants this must be driven under.
#[derive(Resource, Default)]
pub struct PartitionAllocator {
    pool: SlotPool<IVec3>,
}

impl PartitionAllocator {
    /// Stable partition index for `block`, allocating one on first touch. Re-calling
    /// for a resident block is an idempotent no-op returning the same index, so the
    /// block's static instances never change partition (no spurious re-WRITE).
    #[inline]
    pub fn allocate(&mut self, block: IVec3) -> u32 {
        self.pool.allocate(block)
    }

    /// The partition index for `block`, if one is allocated.
    #[inline]
    pub fn index_of(&self, block: IVec3) -> Option<u32> {
        self.pool.slot_of(block)
    }

    /// Release `block`'s index to the free-list (reused by a later block). Invariant 3:
    /// sequence with the translation-op rebuild in one extract.
    #[inline]
    pub fn free(&mut self, block: IVec3) {
        self.pool.free(block);
    }

    /// Mark-and-sweep the live block set in one shot: allocate every present block,
    /// free any tracked block no longer present. `present` / `is_present` MUST come
    /// from resident-instance presence (invariant 1).
    pub fn reconcile<I, F>(&mut self, present: I, is_present: F)
    where
        I: IntoIterator<Item = IVec3>,
        F: Fn(IVec3) -> bool,
    {
        self.pool.reconcile(present, is_present);
    }

    /// `(block, partition_index)` for every live block — the source rows for the
    /// per-partition `WRITE_PARTITION_TRANSLATION` payload.
    pub fn iter(&self) -> impl Iterator<Item = (IVec3, u32)> + '_ {
        self.pool.iter()
    }

    /// The PTLAS `partition_count` to size/build with. Compacts the high-water first
    /// (invariant 2) so it tracks the live peak, not an old spike, then returns `None`
    /// if it would exceed `max_partition_count` — the caller coarsens [`BLOCK_SHIFT`]
    /// or falls back to the shader-add path rather than bricking the build.
    pub fn partition_count(&mut self, max_partition_count: u32) -> Option<u32> {
        let count = self.pool.compact();
        (count <= max_partition_count).then_some(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_of_floors_negative_cells() {
        // BLOCK_SHIFT = 0 → identity; assert the helper at least round-trips and that
        // arithmetic shift would floor (documented contract for BLOCK_SHIFT > 0).
        assert_eq!(block_of(IVec3::new(-3, 0, 7)), IVec3::new(-3, 0, 7));
        assert_eq!(-3i32 >> 1, -2, "arithmetic shift floors toward -inf");
    }

    #[test]
    fn resident_block_is_stable_count_compacts_and_caps() {
        let mut a = PartitionAllocator::default();
        let b0 = a.allocate(IVec3::new(10, 0, 0));
        assert_eq!(b0, 0);
        a.allocate(IVec3::new(11, 0, 0)); // index 1
        // Resident block keeps its index across frames → no instance re-WRITE.
        assert_eq!(a.allocate(IVec3::new(10, 0, 0)), b0);
        assert_eq!(a.partition_count(64), Some(2));

        // Evict the top block; partition_count compacts the freed high-water back.
        a.free(IVec3::new(11, 0, 0));
        assert_eq!(a.partition_count(64), Some(1));
        assert_eq!(a.index_of(IVec3::new(10, 0, 0)), Some(0), "survivor untouched");

        // Over the device cap → None (caller coarsens/falls back), never a panic.
        assert_eq!(a.partition_count(0), None);
    }
}
