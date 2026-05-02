//! `bevy_aurora` — pure ray-traced rendering for clustered (meshlet) geometry.
//!
//! Aurora renders meshlet geometry through a fully ray-traced pipeline:
//!
//! 1. [`scene`] builds Cluster Acceleration Structures from `MeshletMesh` assets,
//!    composes them into BLASes, and instances those into a TLAS via the standard
//!    `VK_KHR_acceleration_structure` path.
//! 2. [`primary`] traces camera rays against that TLAS and writes a "primary
//!    G-buffer" (`PGBuffer`) of world position / normal / motion / material id —
//!    replacing the raster G-buffer that [`bevy_solari`] reads from.
//! 3. [`lighting`] runs `ReSTIR` DI / GI (cribbed from `bevy_solari` and rebound to
//!    the `PGBuffer`) to compute direct + indirect lighting.
//! 4. [`compose`] combines DI + GI, optionally upsamples a quarter-res GI buffer,
//!    and writes the camera's `view_target` so the rest of bevy's render graph
//!    (transparent, sky, post-process, UI) composites on top.
//! 5. [`dlss_rr`] (feature `dlss`) drives DLSS Ray Reconstruction for the final
//!    upscale.
//!
//! See `crates/bevy_aurora/plan.md` for the full milestone breakdown and
//! architectural rationale.
//!
//! [`bevy_solari`]: https://docs.rs/bevy_solari/latest/bevy_solari/

#![allow(
    missing_docs,
    reason = "Aurora is a research crate; doc completeness comes after API settles."
)]
#![allow(
    unsafe_code,
    reason = "Aurora drives Vulkan directly via as_hal escapes for the cluster_AS \
              pipeline (see crates/bevy_aurora/plan.md §2). The whole crate is \
              intrinsically unsafe; the workspace's -D unsafe-code lint doesn't apply."
)]

extern crate alloc;

use bevy_app::{PluginGroup, PluginGroupBuilder};
use bevy_render::settings::WgpuFeatures;

pub mod primary;
pub mod scene;

/// Re-exports of the most common Aurora types.
pub mod prelude {
    pub use super::AuroraPlugins;
}

/// All Aurora plugins, in the order they should be added.
///
/// Bevy users opt into Aurora by adding this plugin group to their `App`.
/// Future milestones will populate this with the full pipeline; currently it
/// installs only the cluster-AS scene plugin.
pub struct AuroraPlugins;

impl PluginGroup for AuroraPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(scene::AuroraScenePlugin)
            .add(primary::AuroraPgbufferPlugin)
            .add(primary::AuroraPrimaryVisibilityPlugin)
    }
}

impl AuroraPlugins {
    /// `WgpuFeatures` Aurora requires from the adapter.
    ///
    /// `EXPERIMENTAL_RAY_QUERY` is needed for the standard KHR-AS dependencies
    /// (`VK_KHR_acceleration_structure` + `VK_KHR_buffer_device_address`); the
    /// `cluster_AS` extension itself sits on top of those.
    pub fn required_wgpu_features() -> WgpuFeatures {
        WgpuFeatures::EXPERIMENTAL_RAY_QUERY
            | WgpuFeatures::EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE
            | WgpuFeatures::BUFFER_BINDING_ARRAY
            | WgpuFeatures::TEXTURE_BINDING_ARRAY
            | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | WgpuFeatures::PARTIALLY_BOUND_BINDING_ARRAY
    }
}
