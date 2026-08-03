// Raw `ash` + `as_hal_mut` extension calls are unavoidably unsafe; every
// public unsafe function documents its invariants in a `# Safety` section.
#![allow(unsafe_code)]

//! NV cluster-AS / partitioned-AS extension wrappers — Vulkan +
//! NVIDIA (Turing+) only. bevy_solari assumes Vulkan; see
//! crate-level docs.
//!
//! The cluster pipeline reaches into `wgpu::hal::vulkan` via
//! [`wgpu::CommandEncoder::as_hal_mut`] and calls
//! `VK_NV_cluster_acceleration_structure` /
//! `VK_NV_partitioned_acceleration_structure` directly through [`ash`] —
//! the maintainer-blessed pattern for vendor-specific extensions (wgpu
//! issues [#4067], [#6762]); wgpu does not expose them because they are
//! NV-only and built around raw `VkDeviceAddress` references that don't
//! fit wgpu's safe abstraction.
//!
//! Extension enable runs through bevy_render's `raw_vulkan_init`
//! infrastructure (see [`RawVulkanInitSettings`]):
//! [`crate::SolariInitPlugin`] registers a Vulkan device-creation
//! callback that probes adapter support and chains the feature structs
//! into the `VkDeviceCreateInfo`. [`SolariInitPlugin`] is added to
//! `DefaultPlugins` before `RenderPlugin` (same slot as `DlssInitPlugin`),
//! so apps using `DefaultPlugins` get the callbacks automatically.
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

/// Marker type for `VK_NV_device_diagnostic_checkpoints` (crash forensics:
/// per-pass GPU checkpoints reported on device loss).
pub struct DiagnosticCheckpointsFeature;

/// Marker type for `VK_NV_partitioned_acceleration_structure`.
pub struct PartitionedAccelerationStructureFeature;


/// Marker type for `VK_NV_ray_tracing_linear_swept_spheres` (ray-traced hair).
pub struct LinearSweptSpheresFeature;

/// Marker registered in [`AdditionalVulkanFeatures`] when
/// `VK_KHR_ray_tracing_pipeline` is enabled on the device. REQUIRED: shading is
/// the RT-pipeline path (raygen / closest-hit / miss / any-hit + SBT +
/// `cmd_trace_rays`), so its absence disables solari entirely (see
/// `init_allocator`). That in turn makes `RAY_TRACING_SHADER_KHR` always a legal
/// barrier stage — see [`seam_masks`].
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
/// meshes carry a baked opacity micro-map (see [`ClusterMesh`]); the RT cores
/// resolve known opaque/transparent micro-regions in hardware, skipping the
/// `ahit_alpha` any-hit invocation. REQUIRED: every driver exposing the NV
/// cluster-AS extensions (solari's hard floor) also exposes this, so its
/// absence disables solari entirely (see `init_allocator`) instead of carrying
/// a permanent no-OMM fallback through every build/attach path.
///
/// [`ClusterMesh`]: crate::geometry::ClusterMesh
pub struct OpacityMicromapFeature;

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

            // `synchronization2` — every AS barrier in the crate is a
            // `VkMemoryBarrier2`. The wgpu-hal fork enables the feature (and
            // extension) whenever the device supports it; chaining it here too
            // would duplicate the struct in the pNext chain
            // (VUID-VkDeviceCreateInfo-sType-unique).

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
            // NV's driver tolerates the relaxed-model path and the atomic
            // shaders (selector counters, BLAS args) run correctly against
            // it; the VUID is informational on this hardware. TODO: flip
            // device_scope ON once naga emits explicit workgroup layouts.

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

            if supports(nv::device_diagnostic_checkpoints::NAME) {
                args.extensions.push(nv::device_diagnostic_checkpoints::NAME);
                additional.insert::<DiagnosticCheckpointsFeature>();
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

                // Query the device's partitioned-AS properties (maxPartitionCount
                // bounds the PTLAS build's partition_count).
                let mut pas_props =
                    vk::PhysicalDevicePartitionedAccelerationStructurePropertiesNV::default();
                // The NV props struct isn't marked `ExtendsPhysicalDeviceProperties2` in
                // this ash fork (no `push_next`), so chain `p_next` manually; `pas_props`
                // outlives the query call below.
                let mut props2 = vk::PhysicalDeviceProperties2::default();
                props2.p_next = &mut pas_props as *mut _ as *mut core::ffi::c_void;
                instance.get_physical_device_properties2(physical_device, &mut props2);

                // Fail fast rather than clamp: the partition count is compiled into
                // ptlas_fill.slang's spatial hash, so a quietly smaller PTLAS would
                // desync the fill pass.
                assert!(
                    pas_props.max_partition_count >= crate::accel::ptlas::PTLAS_PARTITION_COUNT,
                    "device maxPartitionCount ({}) < PTLAS_PARTITION_COUNT ({}) — \
                     this adapter cannot run bevy_solari's partitioned-AS path",
                    pas_props.max_partition_count,
                    crate::accel::ptlas::PTLAS_PARTITION_COUNT,
                );
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

            // Opacity micro-maps — alpha-cutout meshes carry a baked OMM so the RT
            // cores skip the `ahit_alpha` any-hit on resolved opaque/transparent
            // micro-regions. The NV cluster CLAS build references the OMM array +
            // per-triangle index buffer (see `clas_arena`). Needs the extension here;
            // `VK_KHR_acceleration_structure` (enabled by wgpu) is the other half.
            if supports(ext::opacity_micromap::NAME) {
                args.extensions.push(ext::opacity_micromap::NAME);
                additional.insert::<OpacityMicromapFeature>();
                let features = Box::leak(Box::new(
                    vk::PhysicalDeviceOpacityMicromapFeaturesEXT::default().micromap(true),
                ));
                *args.create_info = core::mem::take(args.create_info).push(features);
            } else {
                tracing::warn!(
                    "VK_EXT_opacity_micromap NOT exposed by this device — it is required \
                     (every cluster-AS-capable driver has it), so solari will be disabled."
                );
            }

            // VK_NV_ray_tracing_validation (disabled): driver-side RT checks (AS
            // build/traversal sanity, SBT, invalid addresses during a trace) the
            // standard validation layers can't see, reported through the
            // VK_EXT_debug_utils messenger wgpu registers. The driver only exposes
            // the extension when `NV_ALLOW_RAYTRACING_VALIDATION=1` is set.
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
    /// Per-device function table for `VK_NV_device_diagnostic_checkpoints`
    /// — per-pass GPU checkpoints, read back on device loss to name the
    /// last AS pass the GPU reached. `None` when unsupported.
    pub checkpoints: Option<nv::device_diagnostic_checkpoints::Device>,
    /// Per-device function table for
    /// `VK_KHR_acceleration_structure` — needed to create the
    /// `vk::AccelerationStructureKHR` handle that ray-trace shaders
    /// bind. The KHR AS extension is enabled by wgpu when
    /// ray-tracing features are requested; loading the function
    /// table here matches the pattern used for the NV extensions.
    pub acceleration_structure: khr::acceleration_structure::Device,
    /// Per-device function table for `VK_EXT_opacity_micromap`
    /// (`vkGetMicromapBuildSizesEXT` / `vkCreateMicromapEXT` /
    /// `vkCmdBuildMicromapsEXT`). The extension is REQUIRED (solari disables
    /// entirely without it — see [`OpacityMicromapFeature`]), so the table is
    /// always loaded. Used to build the per-mesh opacity micro-map the NV
    /// cluster CLAS references (see `geometry::clas_arena`).
    pub opacity_micromap: ext::opacity_micromap::Device,
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

        let checkpoints = additional.has::<DiagnosticCheckpointsFeature>().then(|| {
            let fns = nv::device_diagnostic_checkpoints::Device::load(raw_instance, raw_device);
            let _ = CHECKPOINT_FNS.set(fns.clone());
            fns
        });

        Self {
            checkpoints,
            cluster: has_cluster.then(|| {
                nv::cluster_acceleration_structure::Device::load(raw_instance, raw_device)
            }),
            partitioned: has_partitioned.then(|| {
                nv::partitioned_acceleration_structure::Device::load(raw_instance, raw_device)
            }),
            acceleration_structure,
            // Required extension (solari is disabled when it's absent, so this
            // table is never called on a device without it — loading fn pointers
            // is safe either way).
            opacity_micromap: ext::opacity_micromap::Device::load(raw_instance, raw_device),
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

/// Global checkpoint fn table for device-lost reporting from anywhere (the
/// fence-wait victim sites don't carry `ClusterExtensionFns`).
static CHECKPOINT_FNS: std::sync::OnceLock<nv::device_diagnostic_checkpoints::Device> =
    std::sync::OnceLock::new();

/// On device loss: print the checkpoints the GPU last reached on `queue` —
/// each raw AS pass stamps one (marker encodes the op), so this names the
/// faulting pass. Safe to call with the queue externally synchronized.
pub fn report_queue_checkpoints(queue: vk::Queue) {
    let Some(fns) = CHECKPOINT_FNS.get() else {
        return;
    };
    // SAFETY: queue is valid + externally synchronized by the caller.
    unsafe {
        let len = fns.get_queue_checkpoint_data_len(queue);
        let mut data = vec![vk::CheckpointDataNV::default(); len];
        fns.get_queue_checkpoint_data(queue, &mut data);
        for d in &data {
            tracing::error!(
                "GPU checkpoint reached: marker {:#x} at stage {:?}",
                d.p_checkpoint_marker as usize,
                d.stage,
            );
        }
        if data.is_empty() {
            tracing::error!("GPU checkpoints: none reported on this queue");
        }
    }
}

/// Checkpoint marker base for cluster-AS ops: `0x1000 + op_type` —
/// 0x1000=move, 0x1001=BLAS-from-CLAS, 0x1002=direct CLAS build,
/// 0x1003=template build, 0x1004=instantiate. PTLAS = 0x2000, OMM = 0x3000,
/// standard KHR BLAS = 0x4000.
pub const CKPT_CLUSTER_OP_BASE: usize = 0x1000;
pub const CKPT_PTLAS: usize = 0x2000;
pub const CKPT_MICROMAP: usize = 0x3000;
pub const CKPT_BLAS_KHR: usize = 0x4000;

/// Declared buffer access for a raw-VK op — invisible to wgpu's tracker, so
/// [`validate_raw_access`] checks it instead under `SolariSettings::validate`:
/// every declared range must be fully committed before the op records.
pub struct RawAccess<'a> {
    pub op: &'static str,
    pub reads: &'a [(&'a crate::gpu::allocator::SparseBuffer, core::ops::Range<u64>)],
    pub writes: &'a [(&'a crate::gpu::allocator::SparseBuffer, core::ops::Range<u64>)],
}

static VALIDATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Latch [`SolariSettings::validate`](crate::SolariSettings) at plugin `finish`.
pub(crate) fn latch_validate(on: bool) {
    let _ = VALIDATE.set(on);
}

/// Umbrella debug gate ([`SolariSettings::validate`](crate::SolariSettings)):
/// raw-op access checks here + the PTLAS record validation pass.
pub fn solari_validate_enabled() -> bool {
    VALIDATE.get().copied().unwrap_or(false)
}


/// An uncommitted range consumed by a raw op is a future device-lost — log it
/// with the op + buffer named, BEFORE the GPU faults on an anonymous VA.
pub fn validate_raw_access(a: &RawAccess) {
    if !solari_validate_enabled() {
        return;
    }
    for (kind, set) in [("READ", a.reads), ("WRITE", a.writes)] {
        for (buf, range) in set.iter() {
            if !buf.is_committed(range.clone()) {
                tracing::error!(
                    "raw op {}: {kind} {:?} of sparse {} not fully committed",
                    a.op,
                    range,
                    buf.label(),
                );
            }
        }
    }
}

/// Issue `vkCmdBuildClusterAccelerationStructureIndirectNV` against the active
/// Vulkan command buffer underlying `encoder`.
///
/// All inputs (op-input args buffer, scratch, dst arrays) are addressed via
/// `VkDeviceAddress` inside `commands_info` — the caller resolves wgpu buffers
/// to their device addresses and emits any pre/post barriers around this call
/// (cluster_AS does not participate in wgpu-core's automatic barrier insertion).
///
/// # Safety
///
/// Caller must uphold every Vulkan rule of
/// `vkCmdBuildClusterAccelerationStructureIndirectNV`. The destination buffers
/// must be properly sized + bound; the source info array's device addresses
/// must point at valid per-op input structs; `fns.cluster` must be `Some`.
pub unsafe fn cmd_build_cluster_acceleration_structures_indirect(
    encoder: &mut wgpu::CommandEncoder,
    fns: &ClusterExtensionFns,
    commands_info: &vk::ClusterAccelerationStructureCommandsInfoNV<'_>,
) {
    // Raw-VK recording is invisible to wgpu's diagnostics and Tracy; span the
    // CPU record cost here for every call site (GPU time is captured by the
    // per-pass `*.gpu_wait` poll spans).
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
            if let Some(ck) = &fns.checkpoints {
                let marker = CKPT_CLUSTER_OP_BASE + commands_info.input.op_type.as_raw() as usize;
                (ck.fp().cmd_set_checkpoint_nv)(command_buffer, marker as *const _);
            }
            (cluster.fp().cmd_build_cluster_acceleration_structure_indirect_nv)(
                command_buffer,
                commands_info,
            );
        });
    }
}

bitflags::bitflags! {
    /// The hazards a barrier breaks, at a wgpu↔raw-VK seam. Compose the ones a
    /// site actually has; [`seam_masks`] unions their stage/access contributions.
    ///
    /// Raw AS builds address memory by `VkDeviceAddress`, so wgpu's tracker sees
    /// none of these dependencies and every one of them is stated here by hand.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct AsSeams: u16 {
        /// `write_buffer` / copy staged the build's inputs.
        const UPLOAD_TO_BUILD_INPUT = 1 << 0;
        /// A compute pass wrote the build's descriptors / args / counts.
        const COMPUTE_TO_BUILD_INPUT = 1 << 1;
        /// An earlier AS build produced this build's input (CLAS → BLAS → TLAS).
        const BUILD_TO_BUILD_INPUT = 1 << 2;
        /// Driver-written addresses / sizes are about to be copied out.
        const BUILD_TO_TRANSFER = 1 << 3;
        /// Publish freshly built AS bytes to traversal.
        const BUILD_TO_TRACE = 1 << 4;
        /// An in-place rewrite at stable addresses vs. a still-in-flight trace.
        /// Write-after-read: an execution dependency, no source access.
        const TRACE_TO_BUILD_WAR = 1 << 5;
        /// An opacity micromap build feeding the CLAS build that references it.
        const MICROMAP_TO_BUILD_INPUT = 1 << 6;
        /// Trace output consumed by a later compute / blit pass.
        const TRACE_TO_COMPUTE = 1 << 7;
    }
}

/// The four masks a seam set resolves to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeamMasks {
    pub src_stage: vk::PipelineStageFlags2,
    pub src_access: vk::AccessFlags2,
    pub dst_stage: vk::PipelineStageFlags2,
    pub dst_access: vk::AccessFlags2,
}

/// Resolve `seams` to its stage/access masks. Pure — no device, no encoder — so
/// the mapping is unit-testable without an adapter.
///
/// Every stage named here is unconditionally legal: `init_allocator` refuses to
/// initialize solari unless the cluster-AS, opacity-micromap AND ray-tracing-pipeline
/// features are all present, and every seam site bails without the
/// [`Allocator`](crate::gpu::allocator::Allocator) that gate guards. So there is no
/// feature bool to thread — naming `RAY_TRACING_SHADER_KHR` without the feature
/// (`VUID-vkCmdPipelineBarrier-dstStageMask-07949`) is unreachable by construction.
pub fn seam_masks(seams: AsSeams) -> SeamMasks {
    use vk::AccessFlags2 as A;
    use vk::PipelineStageFlags2 as S;

    // Both traversal consumers: the RT pipeline (`vkCmdTraceRays`) for shading, and
    // inline `rayQuery` from compute (ReSTIR spatial reuse, the batch ray-query
    // service). They run against the same AS, so a publish must cover both.
    let trace_stage = S::RAY_TRACING_SHADER_KHR | S::COMPUTE_SHADER;

    let mut m = SeamMasks {
        src_stage: S::NONE,
        src_access: A::NONE,
        dst_stage: S::NONE,
        dst_access: A::NONE,
    };
    let mut add = |src_stage, src_access, dst_stage, dst_access| {
        m.src_stage |= src_stage;
        m.src_access |= src_access;
        m.dst_stage |= dst_stage;
        m.dst_access |= dst_access;
    };

    if seams.contains(AsSeams::UPLOAD_TO_BUILD_INPUT) {
        add(
            S::COPY,
            A::TRANSFER_WRITE,
            S::ACCELERATION_STRUCTURE_BUILD_KHR | S::MICROMAP_BUILD_EXT,
            A::ACCELERATION_STRUCTURE_READ_KHR | A::MICROMAP_READ_EXT,
        );
    }
    if seams.contains(AsSeams::COMPUTE_TO_BUILD_INPUT) {
        add(
            S::COMPUTE_SHADER,
            A::SHADER_WRITE,
            S::ACCELERATION_STRUCTURE_BUILD_KHR,
            A::ACCELERATION_STRUCTURE_READ_KHR,
        );
    }
    if seams.contains(AsSeams::BUILD_TO_BUILD_INPUT) {
        add(
            S::ACCELERATION_STRUCTURE_BUILD_KHR,
            A::ACCELERATION_STRUCTURE_WRITE_KHR,
            S::ACCELERATION_STRUCTURE_BUILD_KHR,
            A::ACCELERATION_STRUCTURE_READ_KHR,
        );
    }
    if seams.contains(AsSeams::BUILD_TO_TRANSFER) {
        add(
            S::ACCELERATION_STRUCTURE_BUILD_KHR,
            A::ACCELERATION_STRUCTURE_WRITE_KHR,
            S::COPY,
            A::TRANSFER_READ,
        );
    }
    if seams.contains(AsSeams::BUILD_TO_TRACE) {
        add(
            S::ACCELERATION_STRUCTURE_BUILD_KHR,
            A::ACCELERATION_STRUCTURE_WRITE_KHR,
            trace_stage,
            A::ACCELERATION_STRUCTURE_READ_KHR | A::SHADER_READ,
        );
    }
    if seams.contains(AsSeams::TRACE_TO_BUILD_WAR) {
        // Write-after-read needs only an execution dependency: the read must
        // finish before the write starts. No source access mask, no cache flush.
        add(
            trace_stage,
            A::NONE,
            S::ACCELERATION_STRUCTURE_BUILD_KHR,
            A::ACCELERATION_STRUCTURE_WRITE_KHR,
        );
    }
    if seams.contains(AsSeams::MICROMAP_TO_BUILD_INPUT) {
        add(
            S::MICROMAP_BUILD_EXT,
            A::MICROMAP_WRITE_EXT,
            S::ACCELERATION_STRUCTURE_BUILD_KHR,
            A::ACCELERATION_STRUCTURE_READ_KHR,
        );
    }
    if seams.contains(AsSeams::TRACE_TO_COMPUTE) {
        add(
            trace_stage,
            A::SHADER_WRITE,
            S::COMPUTE_SHADER,
            A::SHADER_READ,
        );
    }
    m
}

/// Record a seam barrier against a raw command buffer.
///
/// # Safety
///
/// `cb` must be an open command buffer allocated from `device`.
pub unsafe fn cmd_as_seam_raw(device: &ash::Device, cb: vk::CommandBuffer, seams: AsSeams) {
    // Raw `vkCmdPipelineBarrier2` — invisible to wgpu's profiler. Span it so the
    // AS-build barriers show on the Tracy CPU timeline.
    let _span = tracing::info_span!("vk.as_seam").entered();
    let m = seam_masks(seams);
    let barriers = [vk::MemoryBarrier2::default()
        .src_stage_mask(m.src_stage)
        .src_access_mask(m.src_access)
        .dst_stage_mask(m.dst_stage)
        .dst_access_mask(m.dst_access)];
    let info = vk::DependencyInfo::default().memory_barriers(&barriers);
    unsafe { device.cmd_pipeline_barrier2(cb, &info) };
}

/// Record a seam barrier against the Vulkan command buffer underlying `encoder`.
///
/// # Safety
///
/// Caller must hold an open Vulkan-backed encoder and pass the matching
/// Vulkan-backed `RenderDevice`.
pub unsafe fn cmd_as_seam(
    encoder: &mut wgpu::CommandEncoder,
    render_device: &RenderDevice,
    seams: AsSeams,
) {
    unsafe {
        let hal_device = render_device
            .wgpu_device()
            .as_hal::<VkApi>()
            .expect("cmd_as_seam requires Vulkan backend");
        let raw_device = hal_device.raw_device();
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("cmd_as_seam requires Vulkan backend");
            cmd_as_seam_raw(raw_device, hal_encoder.raw_handle(), seams);
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
            if let Some(ck) = &fns.checkpoints {
                (ck.fp().cmd_set_checkpoint_nv)(command_buffer, CKPT_BLAS_KHR as *const _);
            }
            // One build-geometry-info, one slice of range infos for it.
            as_fns.cmd_build_acceleration_structures(
                command_buffer,
                core::slice::from_ref(build_info),
                &[Some(range_infos)],
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
/// Caller must uphold every Vulkan rule of `vkCmdBuildMicromapsEXT`.
pub unsafe fn cmd_build_micromaps(
    encoder: &mut wgpu::CommandEncoder,
    fns: &ClusterExtensionFns,
    build_info: &vk::MicromapBuildInfoEXT<'_>,
) {
    let _span = tracing::info_span!("vk.build_micromaps").entered();
    let omm = &fns.opacity_micromap;
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder =
                hal_encoder.expect("cmd_build_micromaps requires Vulkan backend");
            let command_buffer = hal_encoder.raw_handle();
            if let Some(ck) = &fns.checkpoints {
                (ck.fp().cmd_set_checkpoint_nv)(command_buffer, CKPT_MICROMAP as *const _);
            }
            // ash exposes only the raw fp for this extension (no safe wrapper).
            (omm.fp().cmd_build_micromaps_ext)(command_buffer, 1, build_info);
        });
    }
}

/// Issue `vkCmdBuildPartitionedAccelerationStructuresNV` against the active
/// Vulkan command buffer underlying `encoder`.
///
/// `build_info`'s `src_acceleration_structure_data` /
/// `dst_acceleration_structure_data` are **storage buffer device addresses**,
/// not AS-handle addresses — NV's spec asks for the underlying buffer's
/// `vkGetBufferDeviceAddressKHR` result, and the validation layer flags
/// `vkGetAccelerationStructureDeviceAddressKHR` results as
/// `VUID-VkDeviceAddress-size-11364`. Reach the storage buffer via
/// [`wgpu::hal::vulkan::AccelerationStructure::raw_buffer`] (added in the
/// solari wgpu patches).
///
/// # Safety
///
/// Caller must uphold every Vulkan rule of
/// `vkCmdBuildPartitionedAccelerationStructuresNV` and ensure
/// `fns.partitioned` is `Some`.
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
            if let Some(ck) = &fns.checkpoints {
                (ck.fp().cmd_set_checkpoint_nv)(command_buffer, CKPT_PTLAS as *const _);
            }
            (partitioned
                .fp()
                .cmd_build_partitioned_acceleration_structures_nv)(
                command_buffer, build_info,
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every representable seam set, so the invariants below are exhaustive
    /// rather than sampled.
    fn all_sets() -> impl Iterator<Item = AsSeams> {
        (0..=u16::MAX).filter_map(AsSeams::from_bits)
    }

    #[test]
    fn war_seam_orders_trace_before_build() {
        let m = seam_masks(AsSeams::TRACE_TO_BUILD_WAR);
        assert!(m
            .src_stage
            .contains(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR));
        assert!(m
            .dst_access
            .contains(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR));
        // Write-after-read is an execution dependency; flushing a source cache
        // would be meaningless here and the reader wrote nothing.
        assert_eq!(m.src_access, vk::AccessFlags2::NONE);
    }

    #[test]
    fn publish_seam_puts_the_trace_on_the_consuming_side() {
        let m = seam_masks(AsSeams::BUILD_TO_TRACE);
        let rt = vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
        assert!(m.dst_stage.contains(rt));
        assert!(!m.src_stage.contains(rt));
        assert!(m
            .src_access
            .contains(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR));
    }

    #[test]
    fn composition_is_the_field_wise_union() {
        for a in all_sets() {
            for b in [
                AsSeams::UPLOAD_TO_BUILD_INPUT,
                AsSeams::BUILD_TO_TRACE,
                AsSeams::TRACE_TO_BUILD_WAR,
                AsSeams::MICROMAP_TO_BUILD_INPUT,
            ] {
                let (ma, mb, mu) = (seam_masks(a), seam_masks(b), seam_masks(a | b));
                assert_eq!(mu.src_stage, ma.src_stage | mb.src_stage);
                assert_eq!(mu.src_access, ma.src_access | mb.src_access);
                assert_eq!(mu.dst_stage, ma.dst_stage | mb.dst_stage);
                assert_eq!(mu.dst_access, ma.dst_access | mb.dst_access);
            }
        }
    }

    #[test]
    fn every_non_empty_seam_set_produces_a_dependency() {
        for seams in all_sets().filter(|s| !s.is_empty()) {
            let m = seam_masks(seams);
            assert_ne!(m.src_stage, vk::PipelineStageFlags2::NONE, "{seams:?}");
            assert_ne!(m.dst_stage, vk::PipelineStageFlags2::NONE, "{seams:?}");
        }
    }
}
