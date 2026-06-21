// Hair geometry residency + linear-swept-sphere BLAS builds.
//
// A `HairAsset` is a set of strands (control points + per-point radii). Each
// resident asset gets:
//   - a contiguous run in the shared vertex / radius / index arenas (the
//     inputs an LSS acceleration-structure build reads), and
//   - a contiguous run in the shader-visible `segments` arena (one
//     `{p0,r0, p1,r1}` record per swept segment, read at hit time to
//     reconstruct the fiber tangent + surface point), and
//   - one bottom-level acceleration structure built once via the standard
//     `vkCmdBuildAccelerationStructuresKHR` path with a
//     `VK_GEOMETRY_TYPE_LINEAR_SWEPT_SPHERES_NV` geometry.
//
// The BLAS device address is stable for the asset's life; hair instances
// reference it when they write their PTLAS records (see `super::ptlas_hair`).
#![allow(unsafe_code, reason = "raw-VK linear-swept-sphere BLAS builds")]

use ash::vk;
use bevy_asset::{AssetEvent, AssetId, Assets};
use core::ffi::c_void;
use bevy_ecs::{
    message::MessageReader,
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_platform::collections::HashMap;
use bevy_render::{
    renderer::raw_vulkan_init::AdditionalVulkanFeatures,
    renderer::{RenderContext, RenderDevice, RenderQueue},
    Extract,
};
use core::ops::Range;

use crate::gpu::allocator::{Allocator, SparseBuffer};
use crate::gpu::extension::ClusterExtensionFns;

use super::asset::HairAsset;

/// Virtual span of each hair arena. Hair is tiny next to cluster geometry,
/// but sparse reservation is free until committed.
const VERTEX_ARENA_BYTES: u64 = 256 * 1024 * 1024;
const RADIUS_ARENA_BYTES: u64 = 128 * 1024 * 1024;
const INDEX_ARENA_BYTES: u64 = 128 * 1024 * 1024;
const SEGMENT_ARENA_BYTES: u64 = 256 * 1024 * 1024;
const BLAS_POOL_BYTES: u64 = 1024 * 1024 * 1024;
const SCRATCH_BYTES: u64 = 256 * 1024 * 1024;

/// BLAS region alignment (`VkAccelerationStructureKHR` storage offset must be
/// 256-aligned).
const BLAS_REGION_ALIGN: u64 = 256;
/// Scratch alignment — `minAccelerationStructureScratchOffsetAlignment` is at
/// most 256 on all desktop NV; conservative.
const SCRATCH_ALIGN: u64 = 256;

/// CPU-flattened upload for one hair asset, produced in `extract_hair_assets`
/// (which reads the main-world `Assets<HairAsset>`) and consumed in
/// `prepare_hair_geometry`.
struct PendingUpload {
    id: AssetId<HairAsset>,
    /// Tightly packed `[x,y,z]` control points, all strands concatenated.
    vertices: Vec<f32>,
    /// Per-control-point radii, parallel to `vertices`.
    radii: Vec<f32>,
    /// SUCCESSIVE indexing: one entry per segment = the global vertex index
    /// of the segment's first control point (segment spans `[k, k+1]`).
    indices: Vec<u32>,
    /// `segments[i] = (p0,r0, p1,r1)` for each segment, shader-visible.
    segments: Vec<f32>,
    segment_count: u32,
}

/// A resident hair asset's GPU placement + built BLAS.
pub struct HairAssetEntry {
    /// Global segment index of this asset's first segment (into `segments`).
    /// Added to a hit's `primitive_index` to find the hit segment.
    pub segment_base: u32,
    pub segment_count: u32,
    /// Built LSS BLAS device address — what a hair instance references in its
    /// PTLAS record. Stable for the asset's life.
    pub blas_address: vk::DeviceAddress,
    blas_handle: vk::AccelerationStructureKHR,
    /// Build inputs (resolved device addresses), kept for the deferred
    /// graph-recorded build.
    build: Option<PendingBuild>,
}

/// A BLAS build scheduled in `prepare_hair_geometry`, recorded into the frame
/// encoder by `dispatch_hair_blas`.
struct PendingBuild {
    vertex_address: vk::DeviceAddress,
    radius_address: vk::DeviceAddress,
    index_address: vk::DeviceAddress,
    scratch_address: vk::DeviceAddress,
    segment_count: u32,
}

/// Render-world resource: the hair geometry arenas, the per-asset residency
/// table, and the bump allocators that place new assets.
#[derive(Resource)]
pub struct HairManager {
    /// `R32G32B32_SFLOAT` control points (stride 12) — LSS build vertex input.
    vertices: SparseBuffer,
    /// `R32_SFLOAT` radii (stride 4) — LSS build radius input.
    radii: SparseBuffer,
    /// `u32` segment indices — LSS build index input (SUCCESSIVE mode).
    indices: SparseBuffer,
    /// Shader-visible per-segment `{p0,r0, p1,r1}` records (stride 32).
    pub segments: SparseBuffer,
    /// Per-asset BLAS storage pool (EXPLICIT handles created over sub-ranges).
    blas_pool: SparseBuffer,
    /// Transient build scratch.
    scratch: SparseBuffer,

    /// Resident assets by id.
    entries: HashMap<AssetId<HairAsset>, HairAssetEntry>,

    // Bump allocators (monotonic; eviction is a future milestone — hair sets
    // are small and rarely unload).
    vertex_words: u64,
    radius_words: u64,
    index_words: u64,
    segment_count: u64,
    blas_bytes: u64,

    /// Uploads waiting for `prepare_hair_geometry`.
    pending: Vec<PendingUpload>,

    /// Whether `VK_NV_ray_tracing_linear_swept_spheres` is enabled. Arenas exist
    /// either way (so the scene group can always bind `segments`), but LSS BLAS
    /// builds are skipped without it — hair simply doesn't appear.
    lss_supported: bool,
}

impl HairManager {
    /// The resident entry for an asset, if uploaded + built.
    #[inline]
    pub fn entry(&self, id: AssetId<HairAsset>) -> Option<&HairAssetEntry> {
        self.entries.get(&id)
    }

    /// Total resident segments — the shader's `segments` binding covers
    /// `[0, segment_high_water)`.
    #[inline]
    pub fn segment_high_water(&self) -> u32 {
        self.segment_count as u32
    }
}

/// `RenderStartup`: allocate the hair arenas. No-op without the raw-VK
/// [`Allocator`] (non-Vulkan / unsupported adapter).
pub fn init_hair_manager(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
    additional: Res<AdditionalVulkanFeatures>,
) {
    let Some(allocator) = allocator else {
        return;
    };
    let lss_supported =
        additional.has::<crate::gpu::extension::LinearSweptSpheresFeature>();
    if !lss_supported {
        bevy_log::warn!(
            target: "bevy_solari.hair",
            "VK_NV_ray_tracing_linear_swept_spheres not available — ray-traced hair disabled (Blackwell / RTX 50-series + driver >= 572.63 required)"
        );
    }
    let as_input = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
    let vertices = allocator.create_sparse_buffer(
        &render_device,
        as_input,
        wgpu::BufferUsages::COPY_DST,
        VERTEX_ARENA_BYTES,
        "hair.vertices",
    );
    let radii = allocator.create_sparse_buffer(
        &render_device,
        as_input,
        wgpu::BufferUsages::COPY_DST,
        RADIUS_ARENA_BYTES,
        "hair.radii",
    );
    let indices = allocator.create_sparse_buffer(
        &render_device,
        as_input,
        wgpu::BufferUsages::COPY_DST,
        INDEX_ARENA_BYTES,
        "hair.indices",
    );
    let segments = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        SEGMENT_ARENA_BYTES,
        "hair.segments",
    );
    let blas_pool = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        wgpu::BufferUsages::COPY_DST,
        BLAS_POOL_BYTES,
        "hair.blas_pool",
    );
    let scratch = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        wgpu::BufferUsages::STORAGE,
        SCRATCH_BYTES,
        "hair.scratch",
    );

    commands.insert_resource(HairManager {
        vertices,
        radii,
        indices,
        segments,
        blas_pool,
        scratch,
        entries: HashMap::default(),
        vertex_words: 0,
        radius_words: 0,
        index_words: 0,
        segment_count: 0,
        blas_bytes: 0,
        pending: Vec::new(),
        lss_supported,
    });
}

/// `ExtractSchedule`: flatten newly added / modified hair assets from the
/// main world into pending uploads. Reads `Assets<HairAsset>` directly via
/// [`Extract`] (main-world access).
pub fn extract_hair_assets(
    mut events: Extract<MessageReader<AssetEvent<HairAsset>>>,
    assets: Extract<Res<Assets<HairAsset>>>,
    manager: Option<ResMut<HairManager>>,
) {
    let Some(mut manager) = manager else {
        return;
    };
    if !manager.lss_supported {
        return;
    }
    for event in events.read() {
        let id = match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => *id,
            _ => continue,
        };
        // Re-upload-as-new on modify is a future milestone; for v1 skip if
        // already resident (avoids orphaning a built BLAS).
        if manager.entries.contains_key(&id) {
            continue;
        }
        let Some(asset) = assets.get(id) else {
            continue;
        };
        if let Some(upload) = flatten(id, asset) {
            manager.pending.push(upload);
        }
    }
}

/// Flatten an asset's strands into the GPU layout (SUCCESSIVE indexing,
/// per-strand-contiguous vertices so a chain never bridges two strands).
fn flatten(id: AssetId<HairAsset>, asset: &HairAsset) -> Option<PendingUpload> {
    let mut vertices = Vec::new();
    let mut radii = Vec::new();
    let mut indices = Vec::new();
    let mut segments = Vec::new();
    let mut vertex_base = 0u32;
    for strand in &asset.strands {
        let n = strand.points.len();
        if n < 2 || strand.radii.len() != n {
            continue;
        }
        for (p, r) in strand.points.iter().zip(&strand.radii) {
            vertices.extend_from_slice(&[p.x, p.y, p.z]);
            radii.push(*r);
        }
        // n-1 segments; SUCCESSIVE index = first vertex of each segment.
        for k in 0..n - 1 {
            indices.push(vertex_base + k as u32);
            let p0 = strand.points[k];
            let p1 = strand.points[k + 1];
            let r0 = strand.radii[k];
            let r1 = strand.radii[k + 1];
            segments.extend_from_slice(&[p0.x, p0.y, p0.z, r0, p1.x, p1.y, p1.z, r1]);
        }
        vertex_base += n as u32;
    }
    let segment_count = indices.len() as u32;
    if segment_count == 0 {
        return None;
    }
    Some(PendingUpload {
        id,
        vertices,
        radii,
        indices,
        segments,
        segment_count,
    })
}

/// `Render::Prepare`: place each pending upload in the arenas, upload its
/// data, query its BLAS size, allocate + create the BLAS handle, and record a
/// [`PendingBuild`] consumed by [`dispatch_hair_blas`].
pub fn prepare_hair_geometry(
    manager: Option<ResMut<HairManager>>,
    fns: Option<Res<ClusterExtensionFns>>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(mut manager), Some(fns)) = (manager, fns) else {
        return;
    };
    if manager.pending.is_empty() {
        return;
    }

    let mut scratch_cursor = 0u64;
    let pending = core::mem::take(&mut manager.pending);
    for upload in pending {
        // ── Arena placement (bump). ──────────────────────────────────────
        let vertex_word_base = manager.vertex_words;
        let radius_word_base = manager.radius_words;
        let index_word_base = manager.index_words;
        let segment_base = manager.segment_count;

        let vertex_words = upload.vertices.len() as u64;
        let radius_words = upload.radii.len() as u64;
        let index_words = upload.indices.len() as u64;
        let seg_count = upload.segment_count as u64;

        manager.vertex_words += vertex_words;
        manager.radius_words += radius_words;
        manager.index_words += index_words;
        manager.segment_count += seg_count;

        // ── Commit + upload. ─────────────────────────────────────────────
        commit_and_write_f32(&manager.vertices, &render_queue, vertex_word_base, &upload.vertices);
        commit_and_write_f32(&manager.radii, &render_queue, radius_word_base, &upload.radii);
        commit_and_write_u32(&manager.indices, &render_queue, index_word_base, &upload.indices);
        commit_and_write_f32(&manager.segments, &render_queue, segment_base * 8, &upload.segments);

        let vertex_address = manager.vertices.address + vertex_word_base * 4;
        let radius_address = manager.radii.address + radius_word_base * 4;
        let index_address = manager.indices.address + index_word_base * 4;

        // ── BLAS size query. ─────────────────────────────────────────────
        // ash doesn't generate `push_next` for the geometry struct, so chain
        // the LSS data into `p_next` by hand. `lss` outlives the size query.
        let lss = lss_geometry_data(vertex_address, radius_address, index_address);
        let mut geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::LINEAR_SWEPT_SPHERES_NV)
            .flags(vk::GeometryFlagsKHR::OPAQUE);
        geometry.p_next = core::ptr::addr_of!(lss) as *const c_void;
        let geometries = [geometry];
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);

        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: build_info populated; max-primitive-counts has one entry for
        // the single geometry. Device-build size query, no GPU work.
        unsafe {
            fns.acceleration_structure.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                Some(&[upload.segment_count]),
                &mut sizes,
            );
        }

        // ── BLAS storage placement + handle. ─────────────────────────────
        let region = manager.blas_bytes.next_multiple_of(BLAS_REGION_ALIGN);
        let region_size = sizes
            .acceleration_structure_size
            .next_multiple_of(BLAS_REGION_ALIGN);
        manager.blas_bytes = region + region_size;
        manager
            .blas_pool
            .commit(region..region + region_size);

        // SAFETY: blas_pool is Vulkan-backed + ACCELERATION_STRUCTURE_STORAGE.
        let pool_vk_buffer = unsafe {
            manager
                .blas_pool
                .wgpu_buffer
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("hair blas_pool must be Vulkan-backed")
                .raw_handle()
        };
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(pool_vk_buffer)
            .offset(region)
            .size(sizes.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
        // SAFETY: buffer live with AS-storage usage; offset 256-aligned;
        // size from the matching build-size query.
        let blas_handle = unsafe {
            fns.acceleration_structure
                .create_acceleration_structure(&create_info, None)
                .expect("vkCreateAccelerationStructureKHR for hair BLAS")
        };
        // SAFETY: handle just created on this device.
        let blas_address = unsafe {
            fns.acceleration_structure
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(blas_handle),
                )
        };

        // ── Scratch placement. ───────────────────────────────────────────
        let scratch_off = scratch_cursor.next_multiple_of(SCRATCH_ALIGN);
        let scratch_size = sizes.build_scratch_size.next_multiple_of(SCRATCH_ALIGN);
        scratch_cursor = scratch_off + scratch_size;
        manager
            .scratch
            .commit(scratch_off..scratch_off + scratch_size);
        let scratch_address = manager.scratch.address + scratch_off;

        manager.entries.insert(
            upload.id,
            HairAssetEntry {
                segment_base: segment_base as u32,
                segment_count: upload.segment_count,
                blas_address,
                blas_handle,
                build: Some(PendingBuild {
                    vertex_address,
                    radius_address,
                    index_address,
                    scratch_address,
                    segment_count: upload.segment_count,
                }),
            },
        );
    }
}

/// `RenderGraph` (BuildBlas): record any pending hair LSS BLAS builds into the
/// frame encoder, then barrier so the PTLAS build + traversal see them.
pub fn dispatch_hair_blas(
    manager: Option<ResMut<HairManager>>,
    fns: Option<Res<ClusterExtensionFns>>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (Some(mut manager), Some(fns)) = (manager, fns) else {
        return;
    };
    let pending: Vec<(vk::AccelerationStructureKHR, PendingBuild)> = manager
        .entries
        .values_mut()
        .filter_map(|e| e.build.take().map(|b| (e.blas_handle, b)))
        .collect();
    if pending.is_empty() {
        return;
    }

    let encoder = ctx.command_encoder();
    for (handle, build) in pending {
        let lss = lss_geometry_data(
            build.vertex_address,
            build.radius_address,
            build.index_address,
        );
        let mut geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::LINEAR_SWEPT_SPHERES_NV)
            .flags(vk::GeometryFlagsKHR::OPAQUE);
        geometry.p_next = core::ptr::addr_of!(lss) as *const c_void;
        let geometries = [geometry];
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(handle)
            .geometries(&geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: build.scratch_address,
            });
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(build.segment_count);
        // SAFETY: dst handle created over committed AS-storage; scratch +
        // geometry addresses committed; barrier emitted after.
        unsafe {
            crate::gpu::extension::cmd_build_acceleration_structures(
                encoder,
                &fns,
                &build_info,
                &[range],
            );
        }
    }
    // SAFETY: encoder open + Vulkan-backed. Make the built BLASes visible to
    // the PTLAS build's instance reads + the traversal.
    unsafe {
        crate::gpu::extension::cmd_global_as_barrier(encoder, &render_device, false);
    }
}

/// Build a `VkAccelerationStructureGeometryLinearSweptSpheresDataNV` for a
/// strand set: `R32G32B32_SFLOAT` positions (stride 12), `R32_SFLOAT` radii
/// (stride 4), `u32` indices (stride 4), SUCCESSIVE indexing, CHAINED end
/// caps (continuous strands, no double-capped interior joints).
fn lss_geometry_data<'a>(
    vertex_address: vk::DeviceAddress,
    radius_address: vk::DeviceAddress,
    index_address: vk::DeviceAddress,
) -> vk::AccelerationStructureGeometryLinearSweptSpheresDataNV<'a> {
    vk::AccelerationStructureGeometryLinearSweptSpheresDataNV::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: vertex_address,
        })
        .vertex_stride(12)
        .radius_format(vk::Format::R32_SFLOAT)
        .radius_data(vk::DeviceOrHostAddressConstKHR {
            device_address: radius_address,
        })
        .radius_stride(4)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: index_address,
        })
        .index_stride(4)
        .indexing_mode(vk::RayTracingLssIndexingModeNV::SUCCESSIVE)
        .end_caps_mode(vk::RayTracingLssPrimitiveEndCapsModeNV::CHAINED)
}

fn commit_and_write_f32(buf: &SparseBuffer, queue: &RenderQueue, word_offset: u64, data: &[f32]) {
    if data.is_empty() {
        return;
    }
    let byte_range: Range<u64> = (word_offset * 4)..((word_offset + data.len() as u64) * 4);
    buf.commit(byte_range.clone());
    queue.write_buffer(buf.buffer(), byte_range.start, bytemuck::cast_slice(data));
}

fn commit_and_write_u32(buf: &SparseBuffer, queue: &RenderQueue, word_offset: u64, data: &[u32]) {
    if data.is_empty() {
        return;
    }
    let byte_range: Range<u64> = (word_offset * 4)..((word_offset + data.len() as u64) * 4);
    buf.commit(byte_range.clone());
    queue.write_buffer(buf.buffer(), byte_range.start, bytemuck::cast_slice(data));
}
