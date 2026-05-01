// Cluster-AS dispatch is gated on `unsafe { Device::as_hal(...) }`; opt out of
// the workspace's `-D unsafe-code` lint for this example only.
#![allow(unsafe_code)]
//! Smoke test for the `VK_NV_cluster_acceleration_structure` Vulkan extension.
//!
//! Validates the integration end-to-end on real hardware:
//!
//! 1. The wgpu Cargo feature `experimental-cluster-acceleration-structure` is on
//!    (forwarded from `bevy_render`'s feature of the same name).
//! 2. `Features::EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE` is requested at
//!    instance creation; the adapter exposes the bit if the GPU supports the
//!    extension AND the driver advertises the `clusterAccelerationStructure`
//!    feature.
//! 3. wgpu-hal enables `VK_NV_cluster_acceleration_structure` when opening the
//!    device and chains
//!    `VkPhysicalDeviceClusterAccelerationStructureFeaturesNV` into
//!    `VkDeviceCreateInfo::pNext`.
//! 4. wgpu-hal loads `vkGetClusterAccelerationStructureBuildSizesNV` and
//!    `vkCmdBuildClusterAccelerationStructureIndirectNV`.
//! 5. The plugin's `RenderStartup` system escapes via `Device::as_hal::<Vulkan>`
//!    and calls the size-query function. If it returns sane sizes, the path is
//!    wired up.
//!
//! Requires an NVIDIA GPU (Turing+) with a driver implementing the extension.
//!
//! ```text
//! cargo run --release --features experimental_cluster_acceleration_structure \
//!     --example cluster_acceleration_structure
//! ```

use std::ops::Deref;

use ash::vk;
use bevy::{
    app::AppExit,
    prelude::*,
    render::{
        render_resource::WgpuFeatures, renderer::RenderDevice, settings::WgpuSettings, RenderApp,
        RenderPlugin, RenderStartup,
    },
};
use wgpu::hal::api::Vulkan;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(RenderPlugin {
                render_creation: WgpuSettings {
                    features: WgpuFeatures::EXPERIMENTAL_RAY_QUERY
                        | WgpuFeatures::EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE,
                    ..default()
                }
                .into(),
                ..default()
            }),
            ClusterAccelerationStructureSmokeTestPlugin,
        ))
        // Quit on the next frame so this acts as a one-shot CLI tool.
        .add_systems(Update, quit_after_first_frame)
        .run();
}

fn quit_after_first_frame(mut commands: Commands) {
    commands.write_message(AppExit::Success);
}

/// Wires the smoke test into the render sub-app's startup schedule.
struct ClusterAccelerationStructureSmokeTestPlugin;

impl Plugin for ClusterAccelerationStructureSmokeTestPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            warn!("RenderApp not present; cluster_AS smoke test skipped.");
            return;
        };
        render_app.add_systems(RenderStartup, smoke_test);
    }
}

fn smoke_test(device: Res<RenderDevice>) {
    let features = device.features();
    if !features.contains(WgpuFeatures::EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE) {
        warn!(
            "Adapter does not expose EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE. \
             Either the GPU is not NVIDIA Turing+, the driver predates \
             VK_NV_cluster_acceleration_structure, or the wgpu-hal Cargo feature \
             is off."
        );
        return;
    }
    info!("EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE: supported");

    // SAFETY:
    // - Adapter advertised the wgpu feature; therefore the device was created with
    //   `VK_NV_cluster_acceleration_structure` enabled and
    //   `clusterAccelerationStructure` set on the chained physical-device feature
    //   struct (see wgpu-hal::vulkan::adapter).
    // - The function pointers were resolved by `Functions::load` at device-open time.
    // - `get_cluster_build_sizes` only inspects the descriptor and returns required
    //   memory; no driver-side buffer touches happen.
    unsafe {
        let wgpu_device = device.wgpu_device();
        let Some(hal_device): Option<_> = wgpu_device.as_hal::<Vulkan>() else {
            warn!("Device is not a Vulkan device; skipping size query.");
            return;
        };
        // `as_hal` returns `impl Deref<Target = vulkan::Device>`. Bind it explicitly
        // before calling the inherent method so type inference resolves
        // `get_cluster_build_sizes` on the concrete HAL Device.
        let hal_device: &wgpu::hal::vulkan::Device = hal_device.deref();

        // Minimal `VkClusterAccelerationStructureInputInfoNV`: a single-cluster
        // triangle build with one triangle. Just enough to exercise the entry point.
        let triangle_input = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .max_geometry_index_value(0)
            .max_cluster_unique_geometry_count(1)
            .max_cluster_triangle_count(1)
            .max_cluster_vertex_count(3)
            .max_total_triangle_count(1)
            .max_total_vertex_count(3);

        let op_input = vk::ClusterAccelerationStructureOpInputNV {
            p_triangle_clusters: &triangle_input as *const _ as *mut _,
        };

        let info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(1)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(op_input);

        let sizes = hal_device.get_cluster_build_sizes(&info);

        info!(
            "vkGetClusterAccelerationStructureBuildSizesNV returned: \
             acceleration_structure_size = {} bytes, \
             update_scratch_size = {} bytes, \
             build_scratch_size = {} bytes",
            sizes.acceleration_structure_size,
            sizes.update_scratch_size,
            sizes.build_scratch_size,
        );
    }
}
