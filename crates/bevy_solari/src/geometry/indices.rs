//! Typed slot identities for the cluster pipeline.
//!
//! Each newtype is `#[repr(transparent)] u32` so wire format,
//! bytemuck casts, and shader-side `u32` views are unchanged.
//! The compile-time benefit: a `GroupIndex` can't be passed where
//! a `ClusterIndex` is expected, and an [`GpuEntity`] can't be
//! confused with a pool item index.
//!
//! Asset-level fields (e.g. `ClusterLodGroup::cluster_start`) stay
//! as bare `u32` — they're mesh-local indices the
//! [`super::ClusterMeshManager`] rebases at upload time. The
//! manager's outputs are typed; that's the boundary where
//! type-correctness kicks in.

use bytemuck::{Pod, Zeroable};

/// Slot in the global cluster pool (`ClusterMeshManager::clusters`).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct ClusterIndex(pub u32);

impl ClusterIndex {
    /// Sentinel value indicating "no cluster".
    pub const NONE: Self = Self(u32::MAX);
}

/// Slot in the global group pool (`ClusterMeshManager::groups`).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct GroupIndex(pub u32);

impl GroupIndex {
    /// Sentinel value used by [`super::ClusterLodGroup::parent_group`]
    /// to mark a root group.
    pub const NONE: Self = Self(u32::MAX);
}

/// Slot in the global node pool (`ClusterMeshManager::nodes`).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct NodeIndex(pub u32);

impl NodeIndex {
    /// Sentinel used by [`super::ClusterMesh::root_node_id`] when
    /// the mesh has no interior node tree (selector enters at
    /// `root_group_id` directly).
    pub const NONE: Self = Self(u32::MAX);
}

/// Persistent per-entity slot allocated by `InstanceManager`. Stable
/// for the render-entity's lifetime in the render world.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct GpuEntity(pub u32);

impl GpuEntity {
    /// Sentinel for "no slot allocated".
    pub const NONE: Self = Self(u32::MAX);
}
