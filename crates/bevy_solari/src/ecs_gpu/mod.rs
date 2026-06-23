//! GPU-resident ECS primitives — the generic "GPU table" machinery `bevy_solari`
//! uses to mirror ECS data to the GPU and keep it in sync with change-driven
//! deltas, independent of any particular domain (transforms, instances, lights).
//!
//! - [`column`] — [`GpuColumn`]: a slot-indexed GPU buffer scattered from a
//!   per-frame `[slot, value-words…]` delta (plus its plugin, prepare set, and
//!   the [`GpuColumnDesc`] / [`GpuTable`] traits).
//! - [`slot`] — [`GpuSlot<T>`]: an entity's dense slot as a component-as-index
//!   (no per-frame hashmap), with the main-world [`GpuSlotAllocator`].
//! - [`table`] — the [`gpu_table!`](crate::gpu_table) macro: declare a
//!   component-indexed table (resource + columns + slot index + plugin) in one
//!   place; you hand-write only the extract.
//!
//! Domain tables built on this: [`crate::transform`] (the transform table) and
//! [`crate::instance`] (the per-`RaytracingMesh3d` columns).

/// Split a 1D workgroup count into a `(x, y, z)` dispatch that respects the
/// `maxComputeWorkGroupCount` per-dimension limit (65535). A shader dispatched
/// this way must reconstruct its flat thread index as
/// `gid.x + gid.y * num_workgroups.x * WG_SIZE_X` (with `@builtin(num_workgroups)`),
/// and guard on its element count since the rounded-up Y adds spare threads.
///
/// Per-instance / per-node passes overflow a 1D dispatch once the element count
/// exceeds `65535 * workgroup_size` (≈ 4.2M at size 64); this lifts that to
/// `65535² * workgroup_size`.
pub fn linear_dispatch(workgroups: u32) -> (u32, u32, u32) {
    const MAX_DIM: u32 = 65535;
    if workgroups <= MAX_DIM {
        (workgroups, 1, 1)
    } else {
        (MAX_DIM, workgroups.div_ceil(MAX_DIM), 1)
    }
}

pub mod column;
pub mod presence;
pub mod reconcile;
pub mod scene_columns;
pub mod slot;
pub mod table;

pub use column::{GpuColumn, GpuColumnDesc, GpuColumnPlugin, GpuColumnPrepareSet, GpuTable};
pub use reconcile::ReconcilePlugin;
pub use presence::{
    GpuPresenceColumn, GpuPresenceColumnPlugin, Presence, PresenceEvents, PresenceTable,
};
pub use scene_columns::{SceneColumns, SceneColumnsPlugin, SCENE_COLUMNS_GROUP_DEF};
pub use slot::{
    assign_gpu_slots, free_gpu_slot, push_record, GpuSlot, GpuSlotAllocator, GpuSlotTable, SlotPool,
};
