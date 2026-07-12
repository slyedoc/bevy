#![allow(unsafe_code, reason = "raw VK build command via as_hal_mut")]

//! Per-frame per-bucket BLAS rebuild for the cluster-AS pipeline.
//!
//! Issues `vkCmdBuildClusterAccelerationStructureIndirectNV` (op type
//! BuildClustersBottomLevel) with EXPLICIT_DESTINATIONS: each bucket's BLAS is
//! built into its stable region inside `blas_sharing`'s `geometry_blas_pool`
//! (address = `pool_base + bucket_slot * worst_case_stride`), so the address
//! never changes while the bucket is live and the PTLAS treats an instance
//! whose bucket is unchanged as a no-op.
//!
//! The build count is GPU-driven: `src_infos_count` is the device address of
//! `blas_sharing`'s `dirty_build_count`, so the driver builds exactly the
//! live buckets — bounded by `(unique geometries × LOD bands)`, not the
//! instance count.
//!
//! Inputs:
//! - `selector.args_buf` — per-bucket
//!   `VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV`
//!   records (count + ref-list device address).
//! - `selector.selected_clas_refs` — per-bucket CLAS device-address
//!   lists, referenced from `args_buf`.
//! - `blas_sharing.geometry_dst_addresses` — per-bucket EXPLICIT
//!   destination addresses (written GPU-side from the stable
//!   `pool_base + slot * stride` formula).
//! - `blas_sharing.dirty_build_count` — GPU build count (`srcInfosCount`).
//! - `blas_sharing.geometry_blas_pool` — the AS-storage target.
//!
//! This pass owns only the transient build scratch; the BLAS storage,
//! destination addresses, and count all live in `blas_sharing`.

use ash::nv;
use ash::vk::{self, TaggedStructure};
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::render_resource::{ComputePassDescriptor, PipelineCache};
use bevy_render::renderer::{RenderContext, RenderDevice};
use bevy_render::camera::ExtractedCamera;
use bevy_render::view::ViewUniformOffset;
use wgpu::CommandEncoderDescriptor;

use crate::instance::InstanceManager;

use crate::gpu::allocator::{Allocator, SparseBuffer};
use super::blas_sharing::BlasSharing;
use crate::gpu::extension::ClusterExtensionFns;
use super::selector::Selector;

/// Address alignment for a per-bucket BLAS region. Vulkan requires
/// acceleration-structure storage at 256 B; NV's cluster BLAS
/// destinations inherit that.
pub const BLAS_REGION_ALIGN: u64 = 256;

/// Virtual address space for the build scratch — 8 GB, sparse; only the
/// per-frame build's scratch commits, so scenes that start small commit
/// little. OMM-bearing cluster builds need this much headroom.
pub const BLAS_SCRATCH_VIRTUAL_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// NV cluster-AS scratch alignment (`clusterScratchByteAlignment`).
pub const BLAS_SCRATCH_ALIGN: u64 = 256;

/// Render-world resource: the per-frame BLAS-rebuild scratch. The BLAS
/// storage / destinations / count live in [`BlasSharing`].
#[derive(Resource)]
pub struct BlasRebuildResources {
    /// Sparse build scratch. Aliased every frame; no contents survive.
    pub scratch: SparseBuffer,
}

pub fn init_blas_rebuild(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
) {
    let Some(allocator) = allocator else {
        return;
    };
    let scratch = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        wgpu::BufferUsages::STORAGE,
        BLAS_SCRATCH_VIRTUAL_BYTES,
        "blas_rebuild.scratch",
    );
    commands.insert_resource(BlasRebuildResources { scratch });
}

/// Build flags for the cluster→BLAS (CLUSTERS_BOTTOM_LEVEL) builds. The
/// bottom-level build MUST declare OMM, or the driver under-sizes the
/// per-geometry BLAS region (its referenced CLASes carry OMM) and the build
/// overflows the committed pool region (VUID-...opMode-10471). Mirrors the CLAS
/// build's OMM opt-in in `clas_arena`. Use for BOTH the stride-sizing query and
/// the actual build so they agree.
pub(crate) fn blas_build_flags() -> vk::BuildAccelerationStructureFlagsKHR {
    // OMM is a required extension (solari disables without it), so the opt-in
    // is unconditional.
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_OPACITY_MICROMAP_UPDATE_EXT
}

/// Worst-case single-BLAS byte size for a mesh with `cluster_count` clusters
/// (every cluster selected). Sizes a bucket's pool region; results are cached
/// by the caller.
pub(crate) fn query_blas_size(
    cluster_fns: &nv::cluster_acceleration_structure::Device,
    cluster_count: u32,
) -> u64 {
    let mut clusters_input = vk::ClusterAccelerationStructureClustersBottomLevelInputNV::default()
        .max_total_cluster_count(cluster_count)
        .max_cluster_count_per_acceleration_structure(cluster_count);
    let op_input = vk::ClusterAccelerationStructureOpInputNV {
        p_clusters_bottom_level: &mut clusters_input as *mut _,
    };
    let size_input = vk::ClusterAccelerationStructureInputInfoNV::default()
        .max_acceleration_structure_count(1)
        .flags(blas_build_flags())
        .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_CLUSTERS_BOTTOM_LEVEL)
        .op_mode(vk::ClusterAccelerationStructureOpModeNV::EXPLICIT_DESTINATIONS)
        .op_input(op_input);
    let mut sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: size_input fully populated; cluster-AS fn table loaded.
    unsafe {
        cluster_fns.get_cluster_acceleration_structure_build_sizes(&size_input, &mut sizes_info);
    }
    sizes_info.acceleration_structure_size
}

/// `RenderGraph` system: issue the per-frame per-bucket BLAS rebuild
/// against this frame's selector output. Runs after
/// [`super::selector::dispatch_selector`] and
/// [`super::blas_sharing::dispatch_blas_sharing`]. Records its raw-VK build
/// into its own encoder and hands it to the shared `RenderContext`; the
/// graph does one submit for the frame.
pub fn dispatch_blas_rebuild(
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
    fns: Option<Res<ClusterExtensionFns>>,
    mut resources: Option<ResMut<BlasRebuildResources>>,
    selector: Option<Res<Selector>>,
    sharing: Option<Res<BlasSharing>>,
    instances: Option<Res<InstanceManager>>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Res<crate::pipelines::SolariPipelines>,
    scene_bind_group: Res<crate::bindings::ClusterSceneBindGroup>,
    view_query: bevy_ecs::system::Query<&ViewUniformOffset, bevy_ecs::query::With<ExtractedCamera>>,
    mut ctx: RenderContext,
) {
    let (
        Some(allocator),
        Some(fns),
        Some(resources),
        Some(selector),
        Some(sharing),
        Some(instances),
    ) = (
        allocator,
        fns,
        resources.as_deref_mut(),
        selector,
        sharing,
        instances,
    ) else {
        return;
    };
    let cluster_fns = match fns.cluster.as_ref() {
        Some(c) => c,
        None => return,
    };
    if instances.active_count() == 0 {
        return;
    }
    // The selector didn't record this frame (cold-start) → its ref lists are
    // stale/empty. Skip the build; `built_level` stays uncommitted, so elect
    // re-fires next frame and the build retries (rebuild-until-built).
    if !selector.recorded {
        tracing::debug!("blas_rebuild: skipped (selector did not record)");
        return;
    }
    // Build capacity (CPU upper bound; the GPU `src_infos_count` is the
    // real, smaller count). `max_per` is the shared worst-case-clusters-
    // per-bucket cap (same source as the selector's ref-list cap +
    // `blas_sharing`'s region stride).
    let bucket_capacity = sharing.build_entry_capacity();
    let max_per = instances.max_clusters_per_bucket();
    let max_total = (bucket_capacity as u64) * (max_per as u64);

    // Size query (scratch sizing) for this frame's worst case: building
    // `bucket_capacity` BLASes, each up to `max_per` clusters.
    let mut clusters_input = vk::ClusterAccelerationStructureClustersBottomLevelInputNV::default()
        .max_total_cluster_count(max_total as u32)
        .max_cluster_count_per_acceleration_structure(max_per);
    let op_input = vk::ClusterAccelerationStructureOpInputNV {
        p_clusters_bottom_level: &mut clusters_input as *mut _,
    };
    let size_input = vk::ClusterAccelerationStructureInputInfoNV::default()
        .max_acceleration_structure_count(bucket_capacity)
        .flags(blas_build_flags())
        .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_CLUSTERS_BOTTOM_LEVEL)
        .op_mode(vk::ClusterAccelerationStructureOpModeNV::EXPLICIT_DESTINATIONS)
        .op_input(op_input);
    let mut sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: size_input is fully populated; cluster_AS function table
    // is loaded (checked above).
    {
        let _span = tracing::info_span!("blas.get_build_sizes").entered();
        unsafe {
            cluster_fns
                .get_cluster_acceleration_structure_build_sizes(&size_input, &mut sizes_info);
        }
    }

    // Commit scratch sparse pages. The BLAS pool + dst-address pages are
    // committed in `prepare_blas_sharing`.
    {
        let _span = tracing::info_span!("blas.commit").entered();
        let scratch_pad = BLAS_SCRATCH_ALIGN - 1;
        resources
            .scratch
            .commit(0..(sizes_info.build_scratch_size.max(1) + scratch_pad));
    }

    // Align scratch — NV cluster_AS scratch requires 256 B alignment of
    // the device address passed in.
    let scratch_base = resources.scratch.address;
    let scratch_misalign = scratch_base & (BLAS_SCRATCH_ALIGN - 1);
    let scratch_offset = if scratch_misalign == 0 {
        0
    } else {
        BLAS_SCRATCH_ALIGN - scratch_misalign
    };
    let scratch_addr = scratch_base + scratch_offset;

    // Build command-info. EXPLICIT_DESTINATIONS reads
    // `dst_addresses_array` as INPUT (the GPU-written per-bucket
    // addresses); `dst_implicit_data` is unused in this mode but the
    // validation layer requires a valid address.
    let args_addr = allocator.wgpu_buffer_device_address(&selector.args_buf.wgpu_buffer).get();
    let cmd_info = vk::ClusterAccelerationStructureCommandsInfoNV {
        s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
        p_next: core::ptr::null_mut(),
        input: size_input,
        dst_implicit_data: sharing.geometry_blas_pool.address,
        scratch_data: scratch_addr,
        dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
            device_address: sharing.geometry_dst_addresses.address,
            stride: 8,
            size: (bucket_capacity as u64) * 8,
        },
        dst_sizes_array: vk::StridedDeviceAddressRegionKHR::default(),
        src_infos_array: vk::StridedDeviceAddressRegionKHR {
            device_address: args_addr,
            // BuildClustersBottomLevelInfoNV is 16 bytes (count,
            // stride, refs_lo, refs_hi).
            stride: 16,
            size: (bucket_capacity as u64) * 16,
        },
        // GPU-driven build count — the device address of the clamped
        // `build_count` (= min(dirty_build_count, capacity)). The driver
        // builds exactly the live buckets, not the instance count, and
        // never more than `max_acceleration_structure_count`.
        src_infos_count: allocator.wgpu_buffer_device_address(&sharing.build_count).get(),
        address_resolution_flags:
            vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
        _marker: core::marker::PhantomData,
    };

    // Declared access (SolariSettings::validate): CPU-knowable ranges only — the BLAS
    // pool dst addresses are GPU-computed and can't be declared here.
    crate::gpu::extension::validate_raw_access(&crate::gpu::extension::RawAccess {
        op: "blas_rebuild.build",
        reads: &[
            (&selector.args_buf, 0..(bucket_capacity as u64) * 16),
            (&sharing.geometry_dst_addresses, 0..(bucket_capacity as u64) * 8),
        ],
        writes: &[(
            &resources.scratch,
            scratch_offset..(sizes_info.build_scratch_size.max(1) + BLAS_SCRATCH_ALIGN - 1),
        )],
    });

    // Raw-VK cluster-BLAS build in its OWN encoder (the solari-pt wgpu fork
    // panics if one encoder mixes wgpu passes with raw `as_hal_mut`), handed
    // to the shared render context: `add_command_buffer` flushes this frame's
    // prior wgpu work first, so on the single queue the build runs after it.
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("blas_rebuild.dispatch"),
    });
    // SAFETY: cluster-AS function table is loaded (checked); the
    // wgpu encoder is open and Vulkan-backed; descriptor strides +
    // addresses point at live buffers committed this frame.
    unsafe {
        // PRE-build barrier: the sharing + selector compute wrote
        // `geometry_dst_addresses`, `dirty_build_count`, `args_buf`, and
        // `selected_clas_refs`. A global barrier (submission-order
        // dependency on the same queue) makes those visible as build input.
        crate::gpu::extension::cmd_global_as_barrier(&mut encoder, &render_device, false);
        crate::gpu::extension::cmd_build_cluster_acceleration_structures_indirect(
            &mut encoder,
            &fns,
            &cmd_info,
        );
        // POST-build barrier: BUILD → SHADER_READ + BUILD_INPUT_READ so
        // the PTLAS fill compute sees fresh BLAS addresses and the PTLAS
        // build sees fresh BLAS payload bytes.
        crate::gpu::extension::cmd_global_as_barrier(&mut encoder, &render_device, false);
    }
    ctx.add_command_buffer(encoder.finish());

    // The build is truly recorded — commit each dirty geometry's `built_level`
    // (elect no longer commits optimistically; a bailed chain must re-elect).
    // Recorded into the ctx encoder AFTER `add_command_buffer`, so it lands in
    // a fresh encoder submitted after the build on the single queue.
    let (Some(commit_pipe), Some(scene_bg), Some(sharing_bg), Some(view_offset)) = (
        pipeline_cache.get_compute_pipeline(pipelines.blas_sharing_commit_built),
        scene_bind_group.bind_group.as_ref(),
        sharing.bind_group.as_ref(),
        view_query.iter().next(),
    ) else {
        tracing::debug!("blas_rebuild: commit_built skipped (pipeline/bind groups cold)");
        return;
    };
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("blas_sharing.commit_built"),
        timestamp_writes: None,
    });
    pass.set_bind_group(0, scene_bg, &[]);
    pass.set_bind_group(1, sharing_bg, &[view_offset.offset]);
    pass.set_pipeline(commit_pipe);
    pass.dispatch_workgroups(bucket_capacity.div_ceil(64), 1, 1);
}
