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
use ash::{ext, khr, nv, vk};
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


/// Marker type registered in [`AdditionalVulkanFeatures`] when
pub struct LinearSweptSpheresFeature;

/// Marker registered in [`AdditionalVulkanFeatures`] when
/// `VK_KHR_ray_tracing_pipeline` is enabled on the device. The
/// RT-pipeline shading path (raygen / closest-hit / miss / any-hit +
/// SBT + `cmd_trace_rays`) gates on
/// `additional_features.has::<RayTracingPipelineFeature>()`; on an
/// adapter without it, only the inline-`rayQuery` compute path runs.
pub struct RayTracingPipelineFeature;

/// Marker registered when `VK_NV_ray_tracing_invocation_reorder` is enabled —
/// Shader Execution Reordering (SER). The RT-pipeline raygen reorders by
/// hit/material (`hitObjectTraceRay` → `reorderThread` → `hitObjectExecuteShader`)
/// for shading coherence; without it the raygen falls back to plain `traceRay`.
pub struct RayTracingInvocationReorderFeature;

/// Marker registered when `VK_KHR_ray_tracing_position_fetch` is enabled. The
/// closest-hit shaders read the hit triangle's three object-space vertex
/// positions straight from the acceleration structure via
/// `@builtin(hit_triangle_vertex_positions)` instead of re-fetching them from the
/// vertex pool — but only when the AS was built with `ALLOW_DATA_ACCESS` (see
/// `clas_arena` / `blas_rebuild`). Both the build flag and the `SOLARI_POSITION_FETCH`
/// shader def gate on this marker; absent → the chit falls back to the pool load.
pub struct RayTracingPositionFetchFeature;

/// Marker registered when `VK_KHR_shader_clock` is enabled (`shaderSubgroupClock`
/// + `shaderDeviceClock`). The raygen reads the shader clock (`shader_clock()` →
/// Device-scope `OpReadClockKHR`) around the trace to write a per-pixel cost value
/// for the debug heatmap. Absent → the heatmap pass is skipped and the raygen's
/// clock reads compile out (`SOLARI_SHADER_CLOCK` shader def gates them).
pub struct ShaderClockFeature;

/// `true` once `VK_KHR_shader_clock` has been enabled on the device. The free
/// `compile_rt_wgsl` (no ECS access) reads this to decide whether to define the
/// `SOLARI_SHADER_CLOCK` shader def — emitting `OpReadClockKHR` without the
/// extension enabled is a device error, so the clock reads compile out when absent.
static SHADER_CLOCK_AVAILABLE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Whether `VK_KHR_shader_clock` was enabled at device creation.
pub fn shader_clock_available() -> bool {
    SHADER_CLOCK_AVAILABLE.load(core::sync::atomic::Ordering::Relaxed)
}

/// Marker registered when `VK_EXT_opacity_micromap` is enabled. Alpha-cutout
/// meshes carry a baked opacity micro-map (see [`ClusterMesh`]); with this
/// extension the RT cores resolve known opaque/transparent micro-regions in
/// hardware, skipping the `ahit_alpha` any-hit invocation. Absent → the OMM
/// build/attach is skipped and alpha cutouts fall back to pure any-hit.
///
/// [`ClusterMesh`]: crate::geometry::ClusterMesh
pub struct OpacityMicromapFeature;

/// `true` once `VK_EXT_opacity_micromap` has been enabled on the device. Read by
/// the micromap build / CLAS-attach paths to decide whether to wire OMM at all.
static OPACITY_MICROMAP_AVAILABLE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Whether `VK_EXT_opacity_micromap` was enabled at device creation.
pub fn opacity_micromap_available() -> bool {
    OPACITY_MICROMAP_AVAILABLE.load(core::sync::atomic::Ordering::Relaxed)
}

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

                // Read the device's maxPartitionCount once (Stage-0 residual; the
                // floating-origin PTLAS build clamps partition_count to it). Chained
                // into a properties2 query on the physical device we have in hand.
                let mut pas_props =
                    vk::PhysicalDevicePartitionedAccelerationStructurePropertiesNV::default();
                // The NV props struct isn't marked `ExtendsPhysicalDeviceProperties2` in
                // this ash fork (no `push_next`), so chain `p_next` manually; `pas_props`
                // outlives the query call below.
                let mut props2 = vk::PhysicalDeviceProperties2::default();
                props2.p_next = &mut pas_props as *mut _ as *mut core::ffi::c_void;
                instance.get_physical_device_properties2(physical_device, &mut props2);

            }

            // Ray-traced hair via linear swept spheres. Blackwell-only; on
            // older adapters the extension simply isn't advertised and the
            // hair pipeline stays disabled. Enables `linearSweptSpheres` (the
            // capped-cylinder hair primitive) and `spheres` (the point
            // primitive) so both geometry types can be built. Intersection
            // works from the inline `rayQuery` the path tracer already uses —
            // the extension shares the `rayTracing`/`rayQuery` features chained
            // by wgpu; no ray-tracing pipeline is required.
            if supports(nv::ray_tracing_linear_swept_spheres::NAME) {
                args.extensions
                    .push(nv::ray_tracing_linear_swept_spheres::NAME);
                additional.insert::<LinearSweptSpheresFeature>();
                let features = Box::leak(Box::new(
                    vk::PhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV::default()
                        .linear_swept_spheres(true)
                        .spheres(true),
                ));
                *args.create_info = core::mem::take(args.create_info).push(features);
            }

            // Ray-tracing PIPELINE (raygen / closest-hit / miss / any-hit + SBT
            // + cmd_trace_rays), for the multi-material SBT shading path. Distinct
            // from `rayQuery` (inline, which wgpu already enables): this is the
            // `VK_KHR_ray_tracing_pipeline` extension + `rayTracingPipeline`
            // feature. `VK_KHR_acceleration_structure` is already enabled by wgpu
            // for the ray-query feature, and SPIR-V 1.4 is core in Vulkan 1.2+.
            if supports(khr::ray_tracing_pipeline::NAME) {
                args.extensions.push(khr::ray_tracing_pipeline::NAME);
                additional.insert::<RayTracingPipelineFeature>();
                let features = Box::leak(Box::new(
                    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
                        .ray_tracing_pipeline(true),
                ));
                *args.create_info = core::mem::take(args.create_info).push(features);
            }

            // Shader Execution Reordering (SER) — reorders RT-pipeline invocations
            // by hit/material for shading coherence (the big win for a divergent
            // multi-material path tracer). NV-only; the raygen's
            // hitObjectTraceRay/reorderThread/hitObjectExecuteShader path needs it.
            if supports(nv::ray_tracing_invocation_reorder::NAME) {
                args.extensions
                    .push(nv::ray_tracing_invocation_reorder::NAME);
                additional.insert::<RayTracingInvocationReorderFeature>();
                let features = Box::leak(Box::new(
                    vk::PhysicalDeviceRayTracingInvocationReorderFeaturesNV::default()
                        .ray_tracing_invocation_reorder(true),
                ));
                *args.create_info = core::mem::take(args.create_info).push(features);
            }

            // Ray-tracing position fetch — the closest-hit reads the hit triangle's
            // three object-space vertex positions from the AS directly, instead of a
            // vertex-pool fetch (one less indirection per hit; geometric normal +
            // area come free). KHR core extension; pairs with the `ALLOW_DATA_ACCESS`
            // build flag the CLAS / cluster-BLAS builds set when this is present.
            if supports(khr::ray_tracing_position_fetch::NAME) {
                args.extensions
                    .push(khr::ray_tracing_position_fetch::NAME);
                additional.insert::<RayTracingPositionFetchFeature>();
                // Do NOT push `PhysicalDeviceRayTracingPositionFetchFeaturesKHR` here:
                // wgpu's adapter adds + enables it itself for any enabled extension in
                // the set (wgpu-hal `vulkan/adapter.rs` — `position_fetch` is built
                // `if enabled_extensions.contains(ray_tracing_position_fetch::NAME)`,
                // which is now true from the push above). Pushing our own copy too put
                // the struct in the device `pNext` chain twice
                // (VUID-VkDeviceCreateInfo-sType-unique). The extension name + marker
                // are enough; wgpu chains the (enabled) feature struct.
            }

            // Shader clock — `shader_clock()` (OpReadClockKHR) for the per-pixel cost
            // heatmap. The naga fork emits **Device** scope (globally-monotonic
            // counter), which the heatmap needs because it reads the clock across a
            // Shader-Execution-Reordering boundary that migrates the invocation
            // between SMs — a per-SM subgroup clock would give garbage there. So
            // enable `shaderDeviceClock` (not just `shaderSubgroupClock`); emitting
            // Device-scope `OpReadClockKHR` without it is a device error.
            if supports(khr::shader_clock::NAME) {
                args.extensions.push(khr::shader_clock::NAME);
                additional.insert::<ShaderClockFeature>();
                SHADER_CLOCK_AVAILABLE.store(true, core::sync::atomic::Ordering::Relaxed);
                let features = Box::leak(Box::new(
                    vk::PhysicalDeviceShaderClockFeaturesKHR::default()
                        .shader_subgroup_clock(true)
                        .shader_device_clock(true),
                ));
                *args.create_info = core::mem::take(args.create_info).push(features);
            }

            // NV ray-tracing validation — the driver's own RT-specific checks (AS
            // build/traversal sanity, SBT, invalid addresses during a trace) that
            // the standard validation layers can't see. Reported through the
            // VK_EXT_debug_utils messenger wgpu already registers, so the messages
            // surface alongside the other `wgpu_hal::vulkan::instance` lines; the
            // driver auto-flushes them at device idle / device lost. The driver
            // only EXPOSES the extension when the developer sets
            // `NV_ALLOW_RAYTRACING_VALIDATION=1`, so `supports()` gates it for free
            // (no cost in normal runs).
            // Opacity micro-maps — alpha-cutout meshes carry a baked OMM so the RT
            // cores skip the `ahit_alpha` any-hit on resolved opaque/transparent
            // micro-regions. The NV cluster CLAS build references the OMM array +
            // per-triangle index buffer (see `clas_arena`). Needs the extension here;
            // `VK_KHR_acceleration_structure` (enabled by wgpu) is the other half.
            // if supports(ext::opacity_micromap::NAME) {
            //     args.extensions.push(ext::opacity_micromap::NAME);
            //     additional.insert::<OpacityMicromapFeature>();
            //     OPACITY_MICROMAP_AVAILABLE.store(true, core::sync::atomic::Ordering::Relaxed);
            //     let features = Box::leak(Box::new(
            //         vk::PhysicalDeviceOpacityMicromapFeaturesEXT::default().micromap(true),
            //     ));
            //     *args.create_info = core::mem::take(args.create_info).push(features);
            // } else {
            //     tracing::warn!(
            //         "VK_EXT_opacity_micromap NOT exposed by this device — alpha cutouts fall back to \
            //          pure any-hit (no OMM acceleration)."
            //     );
            // }

            // if supports(nv::ray_tracing_validation::NAME) {
            //     args.extensions.push(nv::ray_tracing_validation::NAME);
            //     let features = Box::leak(Box::new(
            //         vk::PhysicalDeviceRayTracingValidationFeaturesNV::default()
            //             .ray_tracing_validation(true),
            //     ));
            //     *args.create_info = core::mem::take(args.create_info).push(features);
            // } else {
            //     tracing::warn!(
            //         "VK_NV_ray_tracing_validation NOT exposed by the driver — set \
            //          NV_ALLOW_RAYTRACING_VALIDATION=1 (and the driver must support it)."
            //     );
            // }
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
    /// Per-device function table for `VK_EXT_opacity_micromap`
    /// (`vkGetMicromapBuildSizesEXT` / `vkCreateMicromapEXT` /
    /// `vkCmdBuildMicromapsEXT`). `None` if the extension wasn't enabled
    /// at device creation. Used to build the per-mesh opacity micro-map
    /// the NV cluster CLAS references (see `geometry::clas_arena`).
    pub opacity_micromap: Option<ext::opacity_micromap::Device>,
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
        let has_opacity_micromap = additional.has::<OpacityMicromapFeature>();

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
            opacity_micromap: has_opacity_micromap
                .then(|| ext::opacity_micromap::Device::load(raw_instance, raw_device)),
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
    rt_pipeline: bool,
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
    // The shading path traverses the freshly-built AS in the ray-tracing-pipeline
    // stage (`vkCmdTraceRays`), so the post-build barrier MUST make AS writes
    // visible to `RAY_TRACING_SHADER_KHR` — otherwise the trace races the build
    // and reads an empty AS (every ray misses). That stage is only legal when the
    // `rayTracingPipeline` feature is enabled; including it without the feature
    // trips VUID-vkCmdPipelineBarrier-dstStageMask-07949 (ERROR_DEVICE_LOST on
    // NV), hence the gate. `COMPUTE_SHADER` stays for the AS-pass compute that
    // also reads/writes these buffers (selector, fill, etc.).
    let rt_stage = if rt_pipeline {
        vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
    } else {
        vk::PipelineStageFlags::empty()
    };
    let src_stage = vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
        | vk::PipelineStageFlags::COMPUTE_SHADER
        | vk::PipelineStageFlags::TRANSFER
        | rt_stage;
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

/// Issue `vkCmdBuildAccelerationStructuresKHR` against the active Vulkan
/// command buffer underlying `encoder` — the standard KHR build path,
/// used by the hair pipeline to build linear-swept-sphere BLASes (the NV
/// LSS extension adds no commands of its own; LSS geometry is fed through
/// the normal `VkAccelerationStructureGeometryKHR` with a chained
/// `VkAccelerationStructureGeometryLinearSweptSpheresDataNV`).
///
/// `build_info`'s `dst_acceleration_structure` must be a created handle,
/// its `scratch_data` a committed device address, and its geometry's
/// vertex/radius/index device addresses must point at committed buffers.
/// `range_infos[i]` gives the primitive count for `build_info`'s geometry.
///
/// # Safety
///
/// Caller must uphold every Vulkan rule of
/// `vkCmdBuildAccelerationStructuresKHR`: the dst handle, scratch, and all
/// geometry input addresses valid and sized, and surrounding barriers
/// emitted (this path is invisible to wgpu's tracker).
pub unsafe fn cmd_build_acceleration_structures(
    encoder: &mut wgpu::CommandEncoder,
    fns: &ClusterExtensionFns,
    build_info: &vk::AccelerationStructureBuildGeometryInfoKHR<'_>,
    range_infos: &[vk::AccelerationStructureBuildRangeInfoKHR],
) {
    let _span = tracing::info_span!("vk.build_acceleration_structures").entered();
    let as_fns = &fns.acceleration_structure;
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder
                .expect("cmd_build_acceleration_structures requires Vulkan backend");
            let command_buffer = hal_encoder.raw_handle();
            // One build-geometry-info, one slice of range infos for it.
            as_fns.cmd_build_acceleration_structures(
                command_buffer,
                core::slice::from_ref(build_info),
                &[Some(range_infos)],
            );
        });
    }
}

/// Barrier making a freshly-built opacity micro-map (`MICROMAP_BUILD_EXT` /
/// `MICROMAP_WRITE_EXT`) visible to the cluster/AS build that references it
/// (`ACCELERATION_STRUCTURE_BUILD_KHR` / `ACCELERATION_STRUCTURE_READ_KHR`) and
/// to later traversals (`MEMORY_READ`). Emit between `cmd_build_micromaps` and
/// the CLAS build in the same encoder.
///
/// # Safety
///
/// Caller must hold an open Vulkan-backed encoder and pass the matching device.
pub unsafe fn cmd_micromap_barrier(
    encoder: &mut wgpu::CommandEncoder,
    render_device: &RenderDevice,
) {
    let _span = tracing::info_span!("vk.micromap_barrier").entered();
    unsafe {
        let hal_device = render_device
            .wgpu_device()
            .as_hal::<VkApi>()
            .expect("cmd_micromap_barrier requires Vulkan backend");
        let raw_device = hal_device.raw_device();
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder =
                hal_encoder.expect("cmd_micromap_barrier requires Vulkan backend");
            let command_buffer = hal_encoder.raw_handle();
            // Legacy sync (sync2 isn't enabled): a micro-map build executes in the
            // ACCELERATION_STRUCTURE_BUILD stage with AS-write access in the
            // legacy fallback mapping (there is no legacy MICROMAP stage/access).
            raw_device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                    .dst_access_mask(
                        vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                            | vk::AccessFlags::MEMORY_READ,
                    )],
                &[],
                &[],
            );
        });
    }
}

/// Issue `vkCmdBuildMicromapsEXT` against the active Vulkan command buffer
/// underlying `encoder` — builds the opacity micro-maps the NV cluster CLAS
/// references. `build_info`'s `dst_micromap` must be a created handle, its
/// `data` / `triangle_array` device addresses point at committed
/// `MICROMAP_BUILD_INPUT_READ_ONLY_EXT` buffers, and `scratch_data` at a
/// committed scratch buffer. Pair with a barrier before the CLAS build / trace
/// that reads the result.
///
/// # Safety
///
/// Caller must uphold every Vulkan rule of `vkCmdBuildMicromapsEXT` and ensure
/// `fns.opacity_micromap` is `Some`.
pub unsafe fn cmd_build_micromaps(
    encoder: &mut wgpu::CommandEncoder,
    fns: &ClusterExtensionFns,
    build_info: &vk::MicromapBuildInfoEXT<'_>,
) {
    let _span = tracing::info_span!("vk.build_micromaps").entered();
    let omm = fns
        .opacity_micromap
        .as_ref()
        .expect("cmd_build_micromaps: opacity-micromap extension not enabled");
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder =
                hal_encoder.expect("cmd_build_micromaps requires Vulkan backend");
            let command_buffer = hal_encoder.raw_handle();
            // ash exposes only the raw fp for this extension (no safe wrapper).
            (omm.fp().cmd_build_micromaps_ext)(command_buffer, 1, build_info);
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
