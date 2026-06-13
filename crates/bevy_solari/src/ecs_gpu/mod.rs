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

pub mod column;
pub mod presence;
pub mod scene_columns;
pub mod slot;
pub mod table;

pub use column::{GpuColumn, GpuColumnDesc, GpuColumnPlugin, GpuColumnPrepareSet, GpuTable};
pub use presence::{
    GpuPresenceColumn, GpuPresenceColumnPlugin, Presence, PresenceEvents, PresenceTable,
};
pub use scene_columns::{SceneColumns, SceneColumnsPlugin, SCENE_COLUMNS_GROUP_DEF};
pub use slot::{
    assign_gpu_slots, free_gpu_slot, push_record, GpuSlot, GpuSlotAllocator, GpuSlotTable, SlotPool,
};
