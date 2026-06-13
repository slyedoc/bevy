// Bevy's workspace lints deny `unsafe-code`. The whole point of
// this module is to call NV cluster-AS Vulkan extensions via raw
// `ash` bindings + `as_hal_mut` escape hatches — fundamentally
// unsafe by design. Every public unsafe function documents its
// invariants in a `# Safety` section.
#![allow(unsafe_code)]

//! NV cluster-AS / partitioned-AS extension wrappers — Vulkan +
//! NVIDIA (Turing+) only. bevy_solari assumes Vulkan; see
//! crate-level docs.
//!
//! Strategy
//! --------
//!
//! The cluster pipeline reaches into `wgpu::hal::vulkan` via
//! [`wgpu::CommandEncoder::as_hal_mut`] and calls
//! `VK_NV_cluster_acceleration_structure` /
//! `VK_NV_partitioned_acceleration_structure` directly through
//! [`ash`]. This is the maintainer-blessed pattern for
//! vendor-specific extensions (per wgpu issue [#4067] "Underlying
//! API Interoperability" and RT tracking [#6762]) — wgpu does not
//! expose these extensions and won't, because they're NV-only and
//! built around raw `VkDeviceAddress` references that don't fit
//! wgpu's safe abstraction.
//!
//! Device init
//! -----------
//!
//! Extension enable runs through bevy_render's `raw_vulkan_init`
//! infrastructure (see [`RawVulkanInitSettings`]).
//! [`crate::SolariInitPlugin`] registers a Vulkan device-creation
//! callback that probes adapter support and chains the cluster-AS
//! + partitioned-AS feature structs into the `VkDeviceCreateInfo`.
//!
//! [`SolariInitPlugin`] is added to `DefaultPlugins` before
//! `RenderPlugin` (same slot as `DlssInitPlugin`), so apps using
//! `DefaultPlugins` get the callbacks wired up automatically.
//!
//! [#4067]: https://github.com/gfx-rs/wgpu/issues/4067
//! [#6762]: https://github.com/gfx-rs/wgpu/issues/6762
//! [`SolariInitPlugin`]: crate::SolariInitPlugin

use ash::vk::TaggedStructure;
use ash::{khr, nv, vk};
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res},
};
use bevy_render::renderer::raw_vulkan_init::{AdditionalVulkanFeatures, RawVulkanInitSettings};
use bevy_render::renderer::RenderDevice;
use wgpu::hal::api::Vulkan as VkApi;

/// Marker type registered in [`AdditionalVulkanFeatures`] when the
/// `VK_NV_cluster_acceleration_structure` extension is enabled on
/// the device. Consumers check
/// `additional_features.has::<ClusterAccelerationStructureFeature>()`
/// to know whether the cluster runtime is usable on the current
/// adapter.
pub struct ClusterAccelerationStructureFeature;

/// Marker type for `VK_NV_partitioned_acceleration_structure`.
pub struct PartitionedAccelerationStructureFeature;

/// Register the cluster-AS + partitioned-AS Vulkan device-creation
/// callback. Called by `SolariInitPlugin::build` — apps using
/// `DefaultPlugins` get this wiring automatically.
///
/// # Safety
///
/// The callback only adds extensions + feature structs; it never
/// removes anything. This satisfies the safety contract of
/// [`RawVulkanInitSettings::add_create_device_callback`].
pub(crate) unsafe fn register_cluster_extension_callback(settings: &mut RawVulkanInitSettings) {
    // SAFETY: callback only pushes extensions + feature structs onto
    // the chain. It does not remove features or replace existing
    // pointers. The feature structs are heap-allocated via Box and
    // leaked into the callback closure for the lifetime of the
    // device-creation call.
    unsafe {
        settings.add_create_device_callback(|args, adapter, additional| {
            let physical_device = adapter.raw_physical_device();
            let instance = adapter.shared_instance().raw_instance();

            // Probe extension support on the physical device. The
            // callback is invoked once per matching adapter; non-NV
            // adapters or drivers without the extensions simply skip.
            let supported_extensions =
                match instance.enumerate_device_extension_properties(physical_device) {
                    Ok(extensions) => extensions,
                    Err(_) => return,
                };

            let supports = |name: &core::ffi::CStr| {
                supported_extensions.iter().any(|ext| {
                    ext.extension_name_as_c_str()
                        .map(|c| c == name)
                        .unwrap_or(false)
                })
            };

            // Enable core 1.0 `sparseBinding` so bevy_solari can back
            // its AS-pipeline arenas (CLAS storage, BLAS pool, PTLAS
            // storage, global cluster-address table) with sparse
            // buffers — stable `VkBuffer` handle + `VkDeviceAddress`
            // across growth, no copy on resize. Skips if the physical
            // device doesn't support it (all desktop NV/AMD/Intel do
            // since the 2010s; mobile may not).
            let phd_features = instance.get_physical_device_features(physical_device);
            if phd_features.sparse_binding != 0 {
                args.device_features.core_mut().sparse_binding = vk::TRUE;
            }

            // NOTE: do NOT enable `vulkanMemoryModelDeviceScope`. Naga's
            // SPIR-V emits Vulkan memory model + Device-scope atomics
            // for atomic-using compute shaders (the
            // `VUID-RuntimeSpirv-vulkanMemoryModel-06265` is what the
            // validation layer fires about). But enabling
            // device_scope makes the driver enforce the model
            // strictly — and naga doesn't emit
            // WorkgroupMemoryExplicitLayoutKHR / Offset decorations
            // on workgroup arrays, so unrelated bevy compute shaders
            // (e.g. `downsample.wgsl`'s
            // `array<array<f32, 16>, 16>` SPD intermediate) get
            // miscompiled → ERROR_DEVICE_LOST mid-frame.
            //
            // NV's driver tolerates the relaxed-model path, and our
            // atomic shaders (selector counters, BLAS args) have run
            // correctly against it (verified via readback). The VUID
            // is informational on this hardware; flip device_scope
            // ON once naga emits explicit workgroup layouts.

            if supports(nv::cluster_acceleration_structure::NAME) {
                args.extensions
                    .push(nv::cluster_acceleration_structure::NAME);
                additional.insert::<ClusterAccelerationStructureFeature>();
                // Chain the feature struct so the driver enables
                // `clusterAccelerationStructure = VK_TRUE` for this
                // device. The Box is leaked into args for the
                // device-creation lifetime.
                let features = Box::leak(Box::new(
                    vk::PhysicalDeviceClusterAccelerationStructureFeaturesNV::default()
                        .cluster_acceleration_structure(true),
                ));
                *args.create_info = core::mem::take(args.create_info).push(features);
            }

            if supports(nv::partitioned_acceleration_structure::NAME) {
                args.extensions
                    .push(nv::partitioned_acceleration_structure::NAME);
                additional.insert::<PartitionedAccelerationStructureFeature>();
                let features = Box::leak(Box::new(
                    vk::PhysicalDevicePartitionedAccelerationStructureFeaturesNV::default()
                        .partitioned_acceleration_structure(true),
                ));
                *args.create_info = core::mem::take(args.create_info).push(features);
            }
        });
    }
}

/// Loaded function pointers for the two NV cluster acceleration
/// extensions plus the standard KHR acceleration-structure handle
/// ops. Construct once per `RenderDevice`; `as_hal_mut` per call is
/// cheap but `vkGetDeviceProcAddr` on every call is wasteful.
#[derive(Resource)]
pub struct ClusterExtensionFns {
    /// Per-device function table for
    /// `VK_NV_cluster_acceleration_structure`. `None` if the
    /// extension wasn't enabled at device creation.
    pub cluster: Option<nv::cluster_acceleration_structure::Device>,
    /// Per-device function table for
    /// `VK_NV_partitioned_acceleration_structure`. `None` if the
    /// extension wasn't enabled at device creation.
    pub partitioned: Option<nv::partitioned_acceleration_structure::Device>,
    /// Per-device function table for
    /// `VK_KHR_acceleration_structure` — needed to create the
    /// `vk::AccelerationStructureKHR` handle that ray-trace shaders
    /// bind. The KHR AS extension is enabled by wgpu when
    /// ray-tracing features are requested; loading the function
    /// table here matches the pattern used for the NV extensions.
    pub acceleration_structure: khr::acceleration_structure::Device,
}

impl ClusterExtensionFns {
    /// Load the cluster + partitioned NV function pointers from a
    /// `RenderDevice`, gated on which extensions were registered in
    /// [`AdditionalVulkanFeatures`] at adapter init.
    ///
    /// Returns a `ClusterExtensionFns` where each field is `Some`
    /// iff the corresponding marker is present in `additional` —
    /// callers can check `.cluster.is_some()` to gate cluster-AS
    /// code paths on the device's actual capabilities.
    ///
    /// # Panics
    ///
    /// Panics if the wgpu device is not running on the Vulkan
    /// backend.
    pub fn load(render_device: &RenderDevice, additional: &AdditionalVulkanFeatures) -> Self {
        let has_cluster = additional.has::<ClusterAccelerationStructureFeature>();
        let has_partitioned = additional.has::<PartitionedAccelerationStructureFeature>();

        // SAFETY: as_hal yields the raw Vulkan device only while the
        // wgpu Device is alive; we only read function pointers and
        // do not mutate state.
        let hal_device = unsafe {
            render_device
                .wgpu_device()
                .as_hal::<VkApi>()
                .expect("ClusterExtensionFns::load requires Vulkan backend")
        };
        let raw_instance = hal_device.shared_instance().raw_instance();
        let raw_device = hal_device.raw_device();

        let acceleration_structure =
            khr::acceleration_structure::Device::load(raw_instance, raw_device);

        Self {
            cluster: has_cluster.then(|| {
                nv::cluster_acceleration_structure::Device::load(raw_instance, raw_device)
            }),
            partitioned: has_partitioned.then(|| {
                nv::partitioned_acceleration_structure::Device::load(raw_instance, raw_device)
            }),
            acceleration_structure,
        }
    }
}

/// `RenderStartup` system: load [`ClusterExtensionFns`] and insert
/// as a resource. Skips insertion (rather than inserting an all-`None`
/// table) when neither cluster nor partitioned-AS extension is
/// enabled, so downstream systems can use `Option<Res<ClusterExtensionFns>>`
/// to gate their work on hardware support.
pub fn init_cluster_extension_fns(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    additional: Res<AdditionalVulkanFeatures>,
) {
    // Skip on adapters that don't have the cluster-AS extension —
    // that's the gating signal for the whole cluster pipeline. The
    // standard KHR `acceleration_structure` table is loaded
    // unconditionally because it's always present when ray-tracing
    // features are enabled, but the consumers also check
    // `cluster.is_some()` before using the NV-specific paths.
    let has_cluster = additional.has::<ClusterAccelerationStructureFeature>();
    if !has_cluster {
        tracing::warn!(
            target: "bevy_solari.init",
            "VK_NV_cluster_acceleration_structure not available — cluster pipeline disabled"
        );
        return;
    }
    let fns = ClusterExtensionFns::load(&render_device, &additional);
    commands.insert_resource(fns);
}

/// Issue `vkCmdBuildClusterAccelerationStructureIndirectNV` against
/// the active Vulkan command buffer underlying `encoder`.
///
/// All inputs (op-input args buffer, scratch, dst arrays) are
/// addressed via `VkDeviceAddress` inside `commands_info` — the
/// caller resolves wgpu buffers to their device addresses (via
/// `vkGetBufferDeviceAddressKHR`, reachable through the standard
/// KHR acceleration-structure function table) and emits any
/// pre/post barriers around this call (cluster_AS does not
/// participate in wgpu-core's automatic barrier insertion).
///
/// # Safety
///
/// Caller must uphold every Vulkan rule of
/// `vkCmdBuildClusterAccelerationStructureIndirectNV`. The
/// destination buffers must be properly sized + bound; the source
/// info array's device addresses must point at valid per-op input
/// structs; `fns.cluster` must be `Some` (the extension was
/// enabled at device creation).
pub unsafe fn cmd_build_cluster_acceleration_structures_indirect(
    encoder: &mut wgpu::CommandEncoder,
    fns: &ClusterExtensionFns,
    commands_info: &vk::ClusterAccelerationStructureCommandsInfoNV<'_>,
) {
    // Raw-VK command recording bypasses wgpu's command encoder, so
    // wgpu's GPU diagnostics never see it and it's invisible on Tracy.
    // Span the encode here in the shared wrapper so every call site
    // (clas_arena / blas_rebuild) shows the
    // `vkCmdBuildClusterAccelerationStructureIndirectNV` CPU record
    // cost without each site repeating the span. (GPU execution time is
    // captured separately by the per-pass `*.gpu_wait` poll spans.)
    let _span = tracing::info_span!("vk.build_cluster_as_indirect").entered();
    let cluster = fns
        .cluster
        .as_ref()
        .expect("cmd_build_cluster_acceleration_structures_indirect: cluster-AS extension not enabled");
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect(
                "cmd_build_cluster_acceleration_structures_indirect requires Vulkan backend",
            );
            let command_buffer = hal_encoder.raw_handle();
            (cluster.fp().cmd_build_cluster_acceleration_structure_indirect_nv)(
                command_buffer,
                commands_info,
            );
        });
    }
}

/// Issue `vkCmdBuildPartitionedAccelerationStructuresNV` against the
/// active Vulkan command buffer underlying `encoder`.
///
/// `build_info`'s `src_acceleration_structure_data` /
/// `dst_acceleration_structure_data` are **storage buffer device
/// addresses**, not AS-handle addresses — NV's spec asks for the
/// underlying buffer's `vkGetBufferDeviceAddressKHR` result, and the
/// validation layer flags `vkGetAccelerationStructureDeviceAddressKHR`
/// results as `VUID-VkDeviceAddress-size-11364`. Reach the storage
/// buffer via
/// [`wgpu::hal::vulkan::AccelerationStructure::raw_buffer`] (added
/// in the solari-pt wgpu patches).
///
/// # Safety
///
/// Caller must uphold every Vulkan rule of
/// `vkCmdBuildPartitionedAccelerationStructuresNV` and ensure
/// `fns.partitioned` is `Some`.
/// Insert a kitchen-sink memory barrier covering raw-VK AS / scratch /
/// compute writes that wgpu's tracker doesn't see. Use after raw-VK
/// AS builds + compute writes to make the produced data visible to
/// subsequent traversals + AS-build inputs.
///
/// # Safety
///
/// Caller must hold an open Vulkan-backed encoder and pass the
/// matching Vulkan-backed `RenderDevice`.
pub unsafe fn cmd_global_as_barrier(
    encoder: &mut wgpu::CommandEncoder,
    render_device: &RenderDevice,
) {
    // Raw `vkCmdPipelineBarrier` — invisible to wgpu's profiler. Span
    // it so the AS-build barriers show on the Tracy CPU timeline.
    let _span = tracing::info_span!("vk.as_barrier").entered();
    let src_access = vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
        | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
        | vk::AccessFlags::SHADER_WRITE
        | vk::AccessFlags::SHADER_READ
        | vk::AccessFlags::TRANSFER_WRITE;
    let dst_access = src_access | vk::AccessFlags::MEMORY_READ;
    // No RAY_TRACING_SHADER_KHR: that stage requires the
    // rayTracingPipeline feature (we only enable rayQuery). Ray
    // queries are evaluated inside the COMPUTE_SHADER stage that
    // already appears here. Including it without the feature trips
    // VUID-vkCmdPipelineBarrier-dstStageMask-07949 and on NV
    // results in ERROR_DEVICE_LOST.
    let src_stage = vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
        | vk::PipelineStageFlags::COMPUTE_SHADER
        | vk::PipelineStageFlags::TRANSFER;
    let dst_stage = src_stage;
    unsafe {
        let hal_device = render_device
            .wgpu_device()
            .as_hal::<VkApi>()
            .expect("cmd_global_as_barrier requires Vulkan backend");
        let raw_device = hal_device.raw_device();
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder =
                hal_encoder.expect("cmd_global_as_barrier requires Vulkan backend");
            let command_buffer = hal_encoder.raw_handle();
            raw_device.cmd_pipeline_barrier(
                command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(src_access)
                    .dst_access_mask(dst_access)],
                &[],
                &[],
            );
        });
    }
}

pub unsafe fn cmd_build_partitioned_acceleration_structures(
    encoder: &mut wgpu::CommandEncoder,
    fns: &ClusterExtensionFns,
    build_info: &vk::BuildPartitionedAccelerationStructureInfoNV<'_>,
) {
    // Raw-VK PTLAS build record — invisible to wgpu's profiler. Span
    // the encode (GPU time is captured by `ptlas.*_gpu_wait`).
    let _span = tracing::info_span!("vk.build_partitioned_as").entered();
    let partitioned = fns.partitioned.as_ref().expect(
        "cmd_build_partitioned_acceleration_structures: partitioned-AS extension not enabled",
    );
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect(
                "cmd_build_partitioned_acceleration_structures requires Vulkan backend",
            );
            let command_buffer = hal_encoder.raw_handle();
            (partitioned
                .fp()
                .cmd_build_partitioned_acceleration_structures_nv)(
                command_buffer, build_info,
            );
        });
    }
}
