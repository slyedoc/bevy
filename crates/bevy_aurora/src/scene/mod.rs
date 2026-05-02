//! Aurora's geometry pipeline: `MeshletMesh` → CLAS → BLAS → TLAS.
//!
//! All three layers are built through the cluster-AS path on the bottom two
//! (`VK_NV_cluster_acceleration_structure`) and the standard KHR path for the
//! TLAS (`VK_KHR_acceleration_structure`). Aurora bypasses wgpu's `Blas`/`Tlas`
//! abstraction because:
//!
//! - Cluster ASes don't fit the wgpu types — they're produced by indirect builds
//!   that write opaque payloads into a single big GPU buffer, with per-CLAS
//!   identity captured only by `VkDeviceAddress`.
//! - wgpu's `Blas` carries instance-tracking metadata Aurora doesn't need.
//!
//! The actual `VkBuffer` allocations and `VkAccelerationStructureKHR` handles
//! are owned by Aurora and freed during render-app teardown.
//!
//! Milestone state: M-A scaffolding only. The cluster_AS smoke test that
//! validated the underlying pipeline lives in
//! `examples/3d/cluster_acceleration_structure.rs` and will be ported into this
//! module proper as `cluster_as.rs`, `blas.rs`, `tlas.rs`, and `meshlet_loader.rs`
//! during M-A implementation.

use bevy_app::{App, Plugin};

/// Builds + maintains the cluster_AS / BLAS / TLAS pipeline backing Aurora's
/// ray-traced rendering.
///
/// In its current scaffolding form this plugin is a no-op; M-A implementation
/// (loading `MeshletMesh` assets, dequantizing positions, and dispatching the
/// indirect cluster build) will fill it in.
pub struct AuroraScenePlugin;

impl Plugin for AuroraScenePlugin {
    fn build(&self, _app: &mut App) {
        // M-A in progress; no systems registered yet.
    }
}
