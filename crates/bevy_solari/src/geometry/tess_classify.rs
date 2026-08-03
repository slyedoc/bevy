// GPU per-base-triangle classification for the adaptive tessellation path.
//
// A self-contained pass chain (raw heap dispatches + submit) that classifies
// every tessellated instance's base triangles into per-edge tessellation factors
// and emits a work list (`part_triangles`) keyed to the [`TessellationTable`]
// patterns, consumed by the displace + instantiate passes. See
// `tess_classify.slang` for the per-triangle math.
#![allow(clippy::type_complexity)]
// Some render-resource types are also glob-re-exported via `render_resource::*`;
// keep the explicit `wgpu::` prefix at call sites.
#![allow(unused_qualifications)]
#![allow(unsafe_code, reason = "raw VK heap dispatches + cluster-AS instantiate")]

use bevy_ecs::{
    change_detection::DetectChanges,
    query::With,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_render::{
    camera::ExtractedCamera,
    render_asset::RenderAssets,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
    sync_world::MainEntity,
    texture::{FallbackImage, GpuImage},
    view::ExtractedView,
};
use bytemuck::{Pod, Zeroable};

use ash::vk::{self, TaggedStructure};

use wgpu::hal::api::Vulkan as VkApi;

use crate::gpu::retire::GpuRetire;
use crate::gpu::allocator::{Allocator, MemoryLocation};
use crate::gpu::binding_seam::{BindingSeam, HeapKind};
use crate::gpu::extension::{AsSeams, ClusterExtensionFns};
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
use crate::ecs_gpu::GpuColumn;
use crate::instance::{RaytracingGpuEntity, TransformColumn};
use super::clas_arena::CLAS_SCRATCH_ALIGN;
use super::mesh_manager::ClusterMeshManager;
use super::tess_displace::TessShowcaseInstances;
use super::tess_table::TessellationTable;
use super::tess_template::{record_build_per_instance_blas, InstanceBlas, TESS_CLUSTER_ID_BASE};

/// Round `addr` up to `align` (power of two).
fn align_up(addr: u64, align: u64) -> u64 {
    (addr + (align - 1)) & !(align - 1)
}

/// Capacity (in parts) of the whole tessellation path: the `part_triangles` work
/// list, the gen-vertex / instantiate / CLAS pools (bounds `gen_vertices` at
/// `GEN_CAPACITY * MAX_VERTS * 12` bytes). The classify shader drops (and does
/// not count) parts past this, so `counts[0]` — the raw build's
/// `src_infos_count` — can never index past the pools.
const GEN_CAPACITY: u32 = 1 << 18; // 262144 parts
/// Fixed micro-vertices per part slot (the table's max).
const MAX_VERTS: u32 = 78;
/// Max distinct tessellated instances the per-instance displacement heap block
/// reserves. The gen pass indexes it by `instance_index`, so it must cover the
/// showcase instance count; unused slots hold fallback descriptors (a heap
/// array has no "partially bound" — an unwritten descriptor is garbage).
/// MUST equal the sized `Texture2D displacement_maps[N]` in `tess_gen_verts.slang`.
const MAX_TESS_DISPLACEMENT_MAPS: u32 = 256;

/// Push params shared with `tess_classify.slang::ClassifyParams` (96 B).
/// `clip_from_world` is `Mat4::to_cols_array` — the shader applies the four
/// columns explicitly, so no matrix-layout convention crosses the boundary.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct ClassifyParams {
    pub clip_from_world: [f32; 16],
    pub viewport: [f32; 2],
    pub px_per_segment: f32,
    pub work_cluster_count: u32,
    pub max_size: u32,
    pub max_size_configs: u32,
    pub part_capacity: u32,
    pub _pad: u32,
}

/// `tess_classify.slang::WorkCluster`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct WorkClusterGpu {
    instance_idx: u32,
    index_base: u32,
    triangle_count: u32,
    /// Global base of this cluster's vertices in the shared pool. The stored index
    /// values are SOURCE-CLUSTER-LOCAL (the mesh manager rebases `index_offset` /
    /// `vertex_offset` to the global pool but leaves the index *values* local, like
    /// the closest-hit), so `classify` must add this to every fetched index to land
    /// on the right global vertex.
    vertex_base: u32,
    /// Prefix-sum base part index for this cluster (Σ prior clusters' `triangle_count`).
    /// Part index = `part_base + triangle`, a DETERMINISTIC slot (every base triangle
    /// emits exactly one part) — so a part's `cluster_id` (`TESS_CLUSTER_ID_BASE + idx`)
    /// is stable across frames.
    part_base: u32,
}

/// Push params shared with `tess_gen_verts.slang::GenParams` (32 B).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GenParams {
    pub displacement_scale: f32,
    pub displacement_bias: f32,
    pub max_verts: u32,
    pub has_displacement: u32,
    /// Upper bound on the part slot a gen thread may touch (= `GEN_CAPACITY`). The
    /// 2D indirect dispatch over-covers (`ceil(n/65535) * 65535` workgroups), so
    /// threads past this must bail before writing out of `gen_vertices`.
    pub part_capacity: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Push params shared with `tess_gen_attrs.slang::AttrParams` (16 B).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct AttrParams {
    pub attr_addr_lo: u32,
    pub attr_addr_hi: u32,
    pub max_tris: u32,
    pub part_capacity: u32,
}

/// Push params shared with `tess_instantiate.slang::InstParams`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct InstParams {
    pub gen_base_lo: u32,
    pub gen_base_hi: u32,
    pub max_verts: u32,
    pub part_capacity: u32,
    pub cluster_id_base: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Render-world resource for the classify + downstream tessellation passes.
/// Every dispatch is a layout-free heap kernel ([`HeapKernel`]), Slang from
/// source, recorded raw into one encoder by [`run_tess_classify`].
#[derive(Resource)]
pub struct TessClassify {
    /// `classify` entry of `tess_classify.slang` + its buffer slots (shared
    /// with `finalize` — the two entries name overlapping parameters).
    classify_kernel: HeapKernel,
    /// `finalize` entry → writes `gen_dispatch` indirect args.
    finalize_kernel: HeapKernel,
    classify_slots: KernelSlots,
    /// Per-tess-instance cluster slot (tess index → RT instance slot), rebuilt each
    /// frame from the render-world slot map. Classify reads the origin-relative world
    /// as `transforms[slots[instance_idx]]` (the gathered, cluster-indexed column), so
    /// the screen-space density projects in the same frame as the camera.
    slots: RawBufferVec<u32>,
    work_clusters: RawBufferVec<WorkClusterGpu>,
    /// `counts[0]` = emitted part count (the indirect INSTANTIATE's `src_infos_count`).
    counts: Buffer,
    /// Emitted `tess_classify.slang::TessTriangleInfo` work list (gen-pass input).
    pub part_triangles: Buffer,
    /// `DispatchIndirectCommand` for the gen pass (one workgroup per part).
    gen_dispatch: Buffer,
    /// Work clusters this resource's CPU lists were last built for (rebuilt when
    /// the instance set changes — the showcase instance set is latch-stable).
    built_instances: usize,

    // ── Micro-vertex generation ──────────────────────────────────────────────
    /// `tess_gen_verts.slang` — set-1 buffers push-indexed, set-0 displacement
    /// array + sampler constant-offset over [`disp_block`](Self::disp_block).
    gen_kernel: HeapKernel,
    gen_slots: KernelSlots,
    /// Heap image block backing `displacement_maps[256]`: slot `base + i` is
    /// instance i's map, every unused slot a fallback descriptor. The block
    /// base is baked into the gen kernel's mapping table at init, so it never
    /// moves; content rewrites are change-driven ([`run_tess_classify`]).
    disp_block: u32,
    /// Sampler-heap slot for the shared filtering `displacement_sampler`.
    disp_sampler_slot: u32,
    /// Whether the displacement block currently mirrors the instance set's
    /// maps (false until first write; re-cleared when the set rebuilds).
    disp_written: bool,
    /// World-space micro-vertices, stride 3 f32, slot `part*MAX_VERTS + v`. The
    /// instantiate pass builds each part's CLAS against its slice of this.
    pub gen_vertices: Buffer,

    // ── Per-micro-triangle smooth normals + UVs ──────────────────────────────
    /// `tess_gen_attrs.slang` (entry `gen_attrs_main`).
    attr_kernel: HeapKernel,
    attr_slots: KernelSlots,
    /// Denormalized per-micro-triangle attrs (3 × {packed normal, uv} = 36 B), fixed
    /// `max_tris` stride per part, read by the closest-hit's smooth-tess branch.
    /// `None` until sized for the instance set (allocator-backed for a stable trace
    /// device address). Size = `total_base_tris * max_tris * 36`.
    gen_attrs: Option<Buffer>,
    /// Per-part metadata (`scene_bindings::TessCluster`, 16 B): attr address + `primitive_base`.
    /// `geometry_addresses.tess_clusters` points here so the chit shades smooth.
    pub gen_attrs_meta: Option<Buffer>,
    gen_attrs_addr: u64,
    pub gen_attrs_meta_addr: u64,

    // ── Per-part CLAS instantiate descriptors ────────────────────────────────
    /// Descriptor-builder heap kernel ([`HeapKernel`]) + its buffer slots.
    inst_kernel: HeapKernel,
    inst_slots: KernelSlots,
    /// One `VkClusterAccelerationStructureInstantiateClusterInfoNV` (8 u32 / 32 B)
    /// per emitted part — consumed by the raw-VK indirect INSTANTIATE.
    pub instantiate_infos: Buffer,
    /// Device address of `gen_vertices` (resolved once the allocator is present);
    /// baked into each descriptor's `vertex_buffer.start_address`.
    gen_vertices_addr: u64,
    /// Total base-triangle count of the latched work set (= the emitted part count,
    /// constant per instance set) — bounds the descriptor-build dispatch CPU-side.
    total_base_tris: u32,

    // ── Raw-VK indirect INSTANTIATE → tess CLAS pool ─────────────────────────
    /// Persistent CLAS storage (implicit-dst), sized once from the instantiate size
    /// query for `tess_clas_sized_for` parts and reused every frame.
    tess_clas_storage: Option<Buffer>,
    tess_clas_scratch: Option<Buffer>,
    /// GPU-written per-part CLAS device addresses (the per-instance BLAS input).
    pub tess_clas_addresses: Option<Buffer>,
    tess_clas_storage_addr: u64,
    tess_clas_scratch_addr: u64,
    tess_clas_addresses_addr: u64,
    /// Part count the CLAS pool was sized for (re-query + realloc if it changes).
    tess_clas_sized_for: u32,

    // ── Group CLAS by instance → per-instance BLAS ───────────────────────────
    /// Per-instance prefix-sum offsets (CPU), `[0, c0, c0+c1, …]`, len = num_instances.
    per_instance_offsets: Vec<u32>,
    /// Per-instance base-tri (= part) counts, len = num_instances.
    per_instance_counts: Vec<u32>,
    // ── Per-instance BLAS addresses ──────────────────────────────────────────
    /// GPU-written per-instance BLAS device addresses (the PTLAS-write input).
    pub blas_addresses: Option<Buffer>,
    blas_addresses_addr: u64,
    /// Keep this + last frame's `InstanceBlas` storage alive while their builds /
    /// any in-flight trace complete (2-deep ring).
    blas_keepalive: Vec<Vec<InstanceBlas>>,
    /// Instance count the scatter/BLAS buffers were sized for.
    blas_sized_for: u32,
    /// Latched true after the first per-instance BLAS build completes — the PTLAS inject
    /// gates on this so it never reserves slots / reads `blas_addresses` before the BLAS
    /// exist (allocator buffers aren't zero-initialized → garbage addr → device-loss).
    pub blas_ready: bool,

    /// Counts every entry into `run_tess_classify` (incl. early-bail frames) so the
    /// guard diagnostics can fire only on the first few frames instead of spamming.
    diag: u32,

    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for TessClassify {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe {
            self.classify_kernel.destroy(&self.raw_device);
            self.finalize_kernel.destroy(&self.raw_device);
            self.gen_kernel.destroy(&self.raw_device);
            self.attr_kernel.destroy(&self.raw_device);
            self.inst_kernel.destroy(&self.raw_device);
        }
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for TessClassify {}
unsafe impl Sync for TessClassify {}

/// `RenderStartup`: heap kernels + persistent output buffers.
pub fn init_tess_classify(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let Some(classify_kernel) = HeapKernel::new(
        &seam,
        "tess_classify.slang",
        include_str!("tess_classify.slang"),
        "classify",
        &[],
        &[],
        "tess_classify",
        size_of::<ClassifyParams>() as u32,
    ) else {
        return;
    };
    let Some(finalize_kernel) = HeapKernel::new(
        &seam,
        "tess_classify.slang",
        include_str!("tess_classify.slang"),
        "finalize",
        &[],
        &[],
        "tess_classify_finalize",
        size_of::<ClassifyParams>() as u32,
    ) else {
        return;
    };

    // Vertex-gen kernel. The set-1 buffers ride the push-indexed slot array;
    // the displacement-map array + sampler are constant-offset rows over a
    // heap block/slot allocated here, so their bases are baked into the
    // mapping table (blocks never move — content rewrites are change-driven).
    let disp_block = seam.alloc_heap_block(HeapKind::Image, MAX_TESS_DISPLACEMENT_MAPS);
    let disp_sampler_slot = seam.alloc_heap_block(HeapKind::Sampler, 1);
    let gen_base_mappings = [
        seam.map_binding(0, 6, HeapKind::Image, disp_block),
        seam.map_binding(0, 7, HeapKind::Sampler, disp_sampler_slot),
    ];
    let Some(gen_kernel) = HeapKernel::new_with_mappings(
        &seam,
        "tess_gen_verts.slang",
        include_str!("tess_gen_verts.slang"),
        "gen_verts",
        crate::bindings::OCTAHEDRAL_MODULES,
        &[],
        &[],
        "tess_gen_verts",
        size_of::<GenParams>() as u32,
        &gen_base_mappings,
    ) else {
        return;
    };

    // Per-micro-triangle smooth-normal/UV kernel.
    let Some(attr_kernel) = HeapKernel::new(
        &seam,
        "tess_gen_attrs.slang",
        include_str!("tess_gen_attrs.slang"),
        "gen_attrs_main",
        crate::bindings::OCTAHEDRAL_MODULES,
        &[],
        "tess_gen_attrs",
        size_of::<AttrParams>() as u32,
    ) else {
        return;
    };

    let counts = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tess_classify.counts"),
        size: 16, // u32 count + pad
        // BLAS_INPUT ⇒ wgpu adds SHADER_DEVICE_ADDRESS (counts[0] is the indirect
        // INSTANTIATE's `src_infos_count`, read by device address).
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let part_triangles = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tess_classify.part_triangles"),
        size: (GEN_CAPACITY as u64) * 32, // TessTriangleInfo = 32 B
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let gen_dispatch = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tess_classify.gen_dispatch"),
        size: 12, // DispatchIndirectCommand
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let gen_vertices = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tess_classify.gen_vertices"),
        size: (GEN_CAPACITY as u64) * (MAX_VERTS as u64) * 12,
        // BLAS_INPUT ⇒ SHADER_DEVICE_ADDRESS — the INSTANTIATE reads this as the
        // CLAS vertex source by device address.
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Per-part CLAS-instantiate descriptor builder — a layout-free heap
    // pipeline ([`HeapKernel`]), Slang from source.
    let Some(inst_kernel) = HeapKernel::new(
        &seam,
        "tess_instantiate.slang",
        include_str!("tess_instantiate.slang"),
        "build_infos",
        &[],
        &[],
        "tess_instantiate",
        size_of::<InstParams>() as u32,
    ) else {
        return;
    };
    let inst_slots = KernelSlots::new(&seam, 4);
    let instantiate_infos = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tess_classify.instantiate_infos"),
        // VkClusterAccelerationStructureInstantiateClusterInfoNV = 32 B / part.
        // BLAS_INPUT ⇒ SHADER_DEVICE_ADDRESS — this is the INSTANTIATE's `src_infos`.
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_SRC,
        size: (GEN_CAPACITY as u64) * 32,
        mapped_at_creation: false,
    });

    let mut slots = RawBufferVec::<u32>::new(wgpu::BufferUsages::STORAGE);
    slots.set_label(Some("tess_classify.slots"));
    let mut work_clusters = RawBufferVec::<WorkClusterGpu>::new(wgpu::BufferUsages::STORAGE);
    work_clusters.set_label(Some("tess_classify.work_clusters"));

    commands.insert_resource(TessClassify {
        classify_kernel,
        finalize_kernel,
        classify_slots: KernelSlots::new(&seam, 8),
        slots,
        work_clusters,
        counts,
        part_triangles,
        gen_dispatch,
        built_instances: usize::MAX,
        gen_kernel,
        gen_slots: KernelSlots::new(&seam, 6),
        disp_block,
        disp_sampler_slot,
        disp_written: false,
        gen_vertices,
        attr_kernel,
        attr_slots: KernelSlots::new(&seam, 7),
        gen_attrs: None,
        gen_attrs_meta: None,
        gen_attrs_addr: 0,
        gen_attrs_meta_addr: 0,
        inst_kernel,
        inst_slots,
        instantiate_infos,
        gen_vertices_addr: 0,
        total_base_tris: 0,
        tess_clas_storage: None,
        tess_clas_scratch: None,
        tess_clas_addresses: None,
        tess_clas_storage_addr: 0,
        tess_clas_scratch_addr: 0,
        tess_clas_addresses_addr: 0,
        tess_clas_sized_for: 0,
        per_instance_offsets: Vec::new(),
        per_instance_counts: Vec::new(),
        blas_addresses: None,
        blas_addresses_addr: 0,
        blas_keepalive: Vec::new(),
        blas_sized_for: 0,
        blas_ready: false,
        diag: 0,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

/// Build the shared `pTriangleClusters` op input for the tess-CLAS instantiate — the
/// per-part geometry-shape maxima MUST match what `tess_table` built the templates
/// with (mismatched maxima here device-loss the build). `count` is the max parts.
fn tess_instantiate_input<'a>(
    triangle_input: &mut vk::ClusterAccelerationStructureTriangleClusterInputNV<'a>,
    max_tris: u32,
    max_verts: u32,
    count: u32,
    op_mode: vk::ClusterAccelerationStructureOpModeNV,
) -> vk::ClusterAccelerationStructureInputInfoNV<'a> {
    *triangle_input = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .max_geometry_index_value(0)
        .max_cluster_unique_geometry_count(1)
        .max_cluster_triangle_count(max_tris)
        .max_cluster_vertex_count(max_verts)
        .max_total_triangle_count(max_tris * count)
        .max_total_vertex_count(max_verts * count)
        .min_position_truncate_bit_count(0);
    let op_input = vk::ClusterAccelerationStructureOpInputNV {
        p_triangle_clusters: triangle_input as *mut _,
    };
    vk::ClusterAccelerationStructureInputInfoNV::default()
        .max_acceleration_structure_count(count)
        .flags(
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
        )
        .op_type(vk::ClusterAccelerationStructureOpTypeNV::INSTANTIATE_TRIANGLE_CLUSTER)
        .op_mode(op_mode)
        .op_input(op_input)
}

/// `Render::Prepare`: build the CPU work lists (cached), derive params from the
/// camera, refresh the heap descriptors, then record + submit the whole raw
/// dispatch chain (classify → finalize → gen verts/attrs → instantiate → CLAS
/// build → per-instance BLAS). Self-submitting, so no render-graph wiring is
/// needed.
pub fn run_tess_classify(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut retire: ResMut<GpuRetire>,
    mut classify: Option<ResMut<TessClassify>>,
    showcase: Option<Res<TessShowcaseInstances>>,
    mesh_manager: Option<Res<ClusterMeshManager>>,
    table: Option<Res<TessellationTable>>,
    images: Res<RenderAssets<GpuImage>>,
    allocator: Option<Res<Allocator>>,
    fns: Option<Res<ClusterExtensionFns>>,
    views: Query<&ExtractedView, With<ExtractedCamera>>,
    settings: Res<crate::SolariSettings>,
    // The gathered, CLUSTER-slot-indexed origin-relative world column (the same buffer
    // the closest-hit's `transforms` reads) + the render-world slot map, so the tess
    // density projects origin-relative world in the same frame as the camera. NOTE: it
    // must be the gathered `TransformColumn`, NOT `current_world()` (world_rel is
    // NODE-indexed, so `transforms[cluster_slot]` there would read the wrong node).
    transforms_col: Option<Res<GpuColumn<TransformColumn>>>,
    gpu_entities: Query<(&MainEntity, &RaytracingGpuEntity)>,
    seam: Option<Res<BindingSeam>>,
    fallback_texture: Res<FallbackImage>,
) {
    let (present, sc, mm, tb) =
        (classify.is_some(), showcase.is_some(), mesh_manager.is_some(), table.is_some());
    let (Some(classify), Some(showcase), Some(mesh_manager), Some(table)) =
        (classify.as_deref_mut(), showcase, mesh_manager, table)
    else {
        // One-shot: a missing resource here means `init_tess_classify` never inserted it
        // (RenderStartup ran before the allocator existed and never retried). This is the
        // only guard whose state we can't see from a per-resource field.
        static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::debug!(
                "tess_classify bail: missing resource (classify={present} showcase={sc} mesh_manager={mm} table={tb})",
            );
        }
        return;
    };
    classify.diag = classify.diag.wrapping_add(1);
    let chatty = classify.diag <= 3;
    if !showcase.found || showcase.instances.is_empty() {
        if chatty {
            tracing::debug!(
                "tess_classify bail: showcase not ready (found={} instances={})",
                showcase.found,
                showcase.instances.len(),
            );
        }
        return;
    }
    // Every dispatch here is a raw heap kernel; without the seam (non-solari
    // device) `init_tess_classify` never inserted the resource, so this only
    // guards resource-order races.
    let Some(seam) = seam.as_deref() else {
        return;
    };
    // Geometry pools must be resident (binder bails the same way on cold start).
    if mesh_manager.vertex_positions.is_empty() {
        if chatty {
            tracing::debug!("tess_classify bail: mesh_manager.vertex_positions empty");
        }
        return;
    }
    let positions = mesh_manager.vertex_positions.buffer();
    let indices = mesh_manager.indices.buffer();
    let Some(view) = views.iter().next() else {
        if chatty {
            tracing::debug!("tess_classify bail: no extracted view");
        }
        return;
    };
    // Compose clip-from-world from the UNJITTERED projection (`clip_from_view`),
    // NOT `view.clip_from_world` (which can carry the DLSS sub-pixel jitter). Jitter
    // in the LOD metric makes a static camera re-derive different per-edge segment
    // counts at boundaries every frame → constant re-tessellation → the micro-triangle
    // numbering (and shading) strobes on dense surfaces. The unjittered metric keeps
    // tessellation stable when still and adapts as the camera actually moves.
    let clip_from_world = view.clip_from_view * view.world_from_view.to_matrix().inverse();

    // Build the per-instance + per-cluster work lists for this instance set. Rebuilt
    // when the set changes — but do NOT latch until the meshes are actually resident
    // (`tess_clusters` populated), else an early empty build would stick forever.
    if classify.built_instances != showcase.instances.len() {
        classify.work_clusters.clear();
        // Running prefix sum of base-triangle counts → each cluster's deterministic
        // part-index base (so `part = part_base + triangle` is stable across frames).
        let mut part_base = 0u32;
        for (i, inst) in showcase.instances.iter().enumerate() {
            if let Some(clusters) = mesh_manager.tess_clusters(inst.mesh) {
                for c in clusters {
                    classify.work_clusters.push(WorkClusterGpu {
                        instance_idx: i as u32,
                        index_base: c.index_offset,
                        triangle_count: c.triangle_count,
                        vertex_base: c.vertex_offset,
                        part_base,
                    });
                    part_base += c.triangle_count;
                }
            }
        }
        let real = classify
            .work_clusters
            .values()
            .iter()
            .filter(|c| c.triangle_count > 0)
            .count();
        if real == 0 {
            // Meshes not resident yet — retry next frame without latching.
            if chatty {
                tracing::debug!(
                    "tess_classify bail: 0 resident tess_clusters across {} instances (tess_clusters(mesh) all None / empty)",
                    showcase.instances.len(),
                );
            }
            return;
        }
        // Each base triangle emits exactly one part, so the total base-tri count IS the
        // emitted part count — used to bound the descriptor-build dispatch CPU-side
        // without an extra indirect pass. Clamped to `GEN_CAPACITY` to match the
        // classify shader's drop of parts past capacity.
        let total_base_tris: u32 = classify
            .work_clusters
            .values()
            .iter()
            .map(|c| c.triangle_count)
            .sum();
        if total_base_tris > GEN_CAPACITY {
            tracing::warn!(
                "tess_classify: {} base triangles exceed the part capacity ({}); \
                 the excess will not be tessellated",
                total_base_tris,
                GEN_CAPACITY,
            );
        }
        classify.total_base_tris = total_base_tris.min(GEN_CAPACITY);
        // Per-instance part counts (= base-tri counts) + prefix-sum base offsets into
        // `tess_clas_addresses` — drives the per-instance BLAS sublists. Clamped to
        // the same capacity so no BLAS references a part slot past the pools.
        let num_instances = showcase.instances.len();
        let mut counts = vec![0u32; num_instances];
        for c in classify.work_clusters.values() {
            if (c.instance_idx as usize) < num_instances {
                counts[c.instance_idx as usize] += c.triangle_count;
            }
        }
        let mut offsets = vec![0u32; num_instances];
        let mut acc = 0u32;
        for (o, c) in offsets.iter_mut().zip(counts.iter_mut()) {
            *o = acc.min(GEN_CAPACITY);
            acc += *c;
            *c = acc.min(GEN_CAPACITY) - *o;
        }
        classify.per_instance_counts = counts;
        classify.per_instance_offsets = offsets;
        classify.work_clusters.write_buffer(&render_device, &render_queue);
        classify.built_instances = showcase.instances.len();
        // The displacement heap block is indexed by `instance_idx` — remirror
        // it for the new instance order.
        classify.disp_written = false;
        tracing::debug!(
            "tess_classify: built work lists — {} instances, {} work clusters, {} base tris",
            showcase.instances.len(),
            classify.work_clusters.len(),
            classify.total_base_tris,
        );
    }
    // Resolve the gen_vertices device address once the allocator is present (baked into
    // each instantiate descriptor's vertex start address).
    if classify.gen_vertices_addr == 0 {
        if let Some(alloc) = allocator.as_ref() {
            classify.gen_vertices_addr = alloc.wgpu_buffer_device_address(&classify.gen_vertices).get();
        }
    }

    // Size + allocate the tess CLAS pool once per instance set (the part count is
    // fixed = total_base_tris). The instantiate size query bounds the implicit-dst
    // storage for the worst case (every part its max-size template); the actual
    // per-frame build uses the GPU `counts[0]` as `src_infos_count`.
    if let (Some(alloc), Some(fns_res)) = (allocator.as_ref(), fns.as_ref()) {
        if let Some(cluster_fns) = fns_res.cluster.as_ref() {
            if classify.total_base_tris > 0 && classify.tess_clas_sized_for != classify.total_base_tris
            {
                let count = classify.total_base_tris;
                let mut tri_input =
                    vk::ClusterAccelerationStructureTriangleClusterInputNV::default();
                // Size the implicit-dst storage with IMPLICIT_DESTINATIONS — COMPUTE_SIZES
                // returns acceleration_structure_size=0 (it computes per-cluster sizes into
                // a buffer, it doesn't build), which would 1-byte the pool and crash.
                let size_input = tess_instantiate_input(
                    &mut tri_input,
                    table.max_triangles,
                    table.max_vertices,
                    count,
                    vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS,
                );
                let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
                // SAFETY: input fully populated; function table loaded.
                unsafe {
                    cluster_fns
                        .get_cluster_acceleration_structure_build_sizes(&size_input, &mut sizes);
                }
                let storage = alloc.create_buffer(
                    &render_device,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
                    wgpu::BufferUsages::STORAGE,
                    sizes.acceleration_structure_size.max(1),
                    MemoryLocation::GpuOnly,
                    "tess_classify.clas_storage",
                );
                let scratch = alloc.create_buffer(
                    &render_device,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    wgpu::BufferUsages::STORAGE,
                    sizes.build_scratch_size.max(1) + CLAS_SCRATCH_ALIGN - 1,
                    MemoryLocation::GpuOnly,
                    "tess_classify.clas_scratch",
                );
                let addresses = alloc.create_buffer(
                    &render_device,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    (count as u64) * 8,
                    MemoryLocation::GpuOnly,
                    "tess_classify.clas_addresses",
                );
                classify.tess_clas_storage_addr = alloc.wgpu_buffer_device_address(&storage).get();
                classify.tess_clas_scratch_addr =
                    align_up(alloc.wgpu_buffer_device_address(&scratch).get(), CLAS_SCRATCH_ALIGN);
                classify.tess_clas_addresses_addr = alloc.wgpu_buffer_device_address(&addresses).get();
                if let Some(old) = classify.tess_clas_storage.replace(storage.into()) {
                    retire.retire(&render_queue, "tess_classify.tess_clas_storage", old);
                }
                if let Some(old) = classify.tess_clas_scratch.replace(scratch.into()) {
                    retire.retire(&render_queue, "tess_classify.tess_clas_scratch", old);
                }
                if let Some(old) = classify.tess_clas_addresses.replace(addresses.into()) {
                    retire.retire(&render_queue, "tess_classify.tess_clas_addresses", old);
                }
                classify.tess_clas_sized_for = count;
                tracing::debug!(
                    "tess_classify: sized CLAS pool for {} parts — storage {} MiB, scratch {} bytes",
                    count,
                    sizes.acceleration_structure_size / (1 << 20),
                    sizes.build_scratch_size,
                );

                // Per-instance BLAS addresses, same instance-set lifetime.
                let num_instances = classify.per_instance_counts.len().max(1) as u32;
                let blas_addresses = alloc.create_buffer(
                    &render_device,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    (num_instances as u64) * 8,
                    MemoryLocation::GpuOnly,
                    "tess_classify.blas_addresses",
                );
                classify.blas_addresses_addr = alloc.wgpu_buffer_device_address(&blas_addresses).get();
                if let Some(old) = classify.blas_addresses.replace(blas_addresses.into()) {
                    retire.retire(&render_queue, "tess_classify.blas_addresses", old);
                }
                classify.blas_sized_for = num_instances;

                // Shading attrs: denormalized per-micro-triangle {normal, uv} (36 B),
                // fixed `max_triangles` stride per part, + the per-part metadata table
                // the closest-hit reads via `geometry_addresses.tess_clusters`. Sized
                // for the exact part count (`count`) so the attr-pass `p < count` guard
                // keeps every write in bounds. Allocator-backed → stable trace addrs.
                let max_tris = table.max_triangles.max(1);
                let gen_attrs = alloc.create_buffer(
                    &render_device,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    (count as u64) * (max_tris as u64) * 36,
                    MemoryLocation::GpuOnly,
                    "tess_classify.gen_attrs",
                );
                classify.gen_attrs_addr = alloc.wgpu_buffer_device_address(&gen_attrs).get();
                if let Some(old) = classify.gen_attrs.replace(gen_attrs.into()) {
                    retire.retire(&render_queue, "tess_classify.gen_attrs", old);
                }
                let gen_attrs_meta = alloc.create_buffer(
                    &render_device,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    (count as u64) * 16,
                    MemoryLocation::GpuOnly,
                    "tess_classify.gen_attrs_meta",
                );
                classify.gen_attrs_meta_addr = alloc.wgpu_buffer_device_address(&gen_attrs_meta).get();
                if let Some(old) = classify.gen_attrs_meta.replace(gen_attrs_meta.into()) {
                    retire.retire(&render_queue, "tess_classify.gen_attrs_meta", old);
                }
                tracing::debug!(
                    "tess_classify: sized shading attrs for {} parts (stride {} tris) — {} MiB",
                    count,
                    max_tris,
                    (count as u64) * (max_tris as u64) * 36 / (1 << 20),
                );
            }
        }
    }
    let work_cluster_count = classify
        .work_clusters
        .values()
        .iter()
        .filter(|c| c.triangle_count > 0)
        .count() as u32;
    if work_cluster_count == 0 {
        if chatty {
            tracing::debug!("tess_classify bail: work_cluster_count == 0 (latched build was empty)");
        }
        return;
    }

    let classify_params = ClassifyParams {
        clip_from_world: clip_from_world.to_cols_array(),
        viewport: [view.viewport.z as f32, view.viewport.w as f32],
        // Live density dial (the shader clamps to `max(_, 1.0)` px/segment).
        px_per_segment: settings.tess_px_per_segment,
        work_cluster_count,
        max_size: table.max_size,
        max_size_configs: table.max_size_configs,
        part_capacity: GEN_CAPACITY,
        _pad: 0,
    };

    // Clear the counters each frame (CPU write — tiny).
    render_queue.write_buffer(&classify.counts, 0, &[0u8; 16]);

    // The gathered cluster-indexed origin-relative world column. Required — the
    // floating origin moves every frame, so the tess world transform must be read
    // from here per frame, exactly like the cluster BLAS path.
    let Some(transforms_col) = transforms_col.as_ref() else {
        if chatty {
            tracing::debug!("tess_classify bail: TransformColumn not present");
        }
        return;
    };
    let transforms = transforms_col.buffer();

    // Rebuild the per-tess-instance → cluster-slot map each frame (cheap; a handful
    // of instances). Cluster slots are stable once assigned, but a missing entry
    // during warmup falls back to slot 0 (origin) rather than a stale absolute pose.
    let slot_of: bevy_ecs::entity::EntityHashMap<u32> = gpu_entities
        .iter()
        .map(|(main, slot)| (main.id(), slot.0 .0))
        .collect();
    classify.slots.clear();
    for inst in &showcase.instances {
        classify
            .slots
            .push(slot_of.get(&inst.entity).copied().unwrap_or(0));
    }
    classify.slots.write_buffer(&render_device, &render_queue);

    // Per-instance displacement: every instance's own map, indexed by
    // `instance_index` in the gen shader. All must be resident — the showcase
    // latches only once materials load, so they are; otherwise skip the gen
    // this frame and retry. A shared filtering sampler serves all maps.
    let disp_views: Option<Vec<&wgpu::TextureView>> = showcase
        .instances
        .iter()
        .map(|inst| images.get(&inst.displacement).map(|img| &*img.texture_view))
        .collect();
    let disp_sampler = showcase
        .instances
        .first()
        .and_then(|inst| images.get(&inst.displacement))
        .map(|img| &img.sampler);
    let gen_ready = matches!(
        (&disp_views, disp_sampler),
        (Some(views), Some(_))
            if !views.is_empty() && views.len() <= MAX_TESS_DISPLACEMENT_MAPS as usize
    );

    // Mirror the maps into the displacement heap block: slot `base + i` is
    // instance i's view, EVERY remaining slot a fallback descriptor (a heap
    // array has no "partially bound" — an unwritten descriptor is garbage).
    // Rewritten only when the instance set rebuilt or `RenderAssets` changed
    // (a replaced asset swaps the underlying view); a steady-state frame
    // writes nothing.
    if gen_ready && (!classify.disp_written || images.is_changed()) {
        let views = disp_views.as_ref().unwrap();
        let fallback = &*fallback_texture.d2.texture_view;
        for i in 0..MAX_TESS_DISPLACEMENT_MAPS {
            let view = views.get(i as usize).copied().unwrap_or(fallback);
            crate::bindings::write_image_descriptor(seam, classify.disp_block + i, view);
        }
        crate::bindings::write_sampler_descriptor(
            seam,
            classify.disp_sampler_slot,
            disp_sampler.unwrap(),
        );
        classify.disp_written = true;
    }

    // Attr pass readiness: produces the smooth normals + UVs (denormalized per
    // micro-triangle) + the per-part metadata the closest-hit reads via
    // `geometry_addresses.tess_clusters`. Gated on the gen pass (shares its
    // inputs + indirect grid) + the attr buffers being sized. Clone the buffer
    // handles (Arc) so they don't borrow `classify` across the slot writes.
    let gen_attrs_buf = classify.gen_attrs.clone();
    let gen_attrs_meta_buf = classify.gen_attrs_meta.clone();
    let attr_ready = gen_ready && gen_attrs_buf.is_some() && gen_attrs_meta_buf.is_some();

    // Build one CLAS-instantiate descriptor per emitted part. Gated on the gen pass
    // running (it fills `gen_vertices`) + the allocator-resolved gen address.
    let inst_ready = gen_ready && classify.gen_vertices_addr != 0;

    // CLAS-build readiness: the descriptors built (inst_ready), the cluster-AS fns
    // loaded, and the CLAS pool sized for the current part count.
    let clas_build_ready = inst_ready
        && fns.as_ref().and_then(|f| f.cluster.as_ref()).is_some()
        && classify.tess_clas_storage.is_some()
        && classify.tess_clas_sized_for == classify.total_base_tris
        && allocator.is_some();

    let (Some(work_buf), Some(slots_buf)) =
        (classify.work_clusters.buffer(), classify.slots.buffer())
    else {
        if chatty {
            tracing::debug!("tess_classify bail: classify list buffers not allocated yet");
        }
        return;
    };

    // Assemble the push blobs (slot rewrites are host memcpys into the heap;
    // the pool buffers can grow/realloc, so rewrite each run). `counts` /
    // `gen_dispatch` slots are shared between the classify and finalize
    // entries — each entry's blob names exactly the bindings surviving in its
    // SPIR-V.
    let s_counts = classify.classify_slots.buffer(seam, 3, &classify.counts);
    let s_gen_dispatch = classify.classify_slots.buffer(seam, 5, &classify.gen_dispatch);
    let classify_blob = classify.classify_kernel.push_blob(
        "tess_classify",
        bytemuck::bytes_of(&classify_params),
        &[
            ("work_clusters", classify.classify_slots.buffer(seam, 0, work_buf)),
            ("vertex_positions", classify.classify_slots.buffer(seam, 1, positions)),
            ("indices", classify.classify_slots.buffer(seam, 2, indices)),
            ("counts", s_counts),
            (
                "part_triangles",
                classify.classify_slots.buffer(seam, 4, &classify.part_triangles),
            ),
            ("transforms", classify.classify_slots.buffer(seam, 6, transforms)),
            ("slots", classify.classify_slots.buffer(seam, 7, slots_buf)),
        ],
    );
    let finalize_blob = classify.finalize_kernel.push_blob(
        "tess_classify_finalize",
        bytemuck::bytes_of(&classify_params),
        &[("counts", s_counts), ("gen_dispatch", s_gen_dispatch)],
    );
    let gen_blob = gen_ready.then(|| {
        let gen_params = GenParams {
            displacement_scale: settings.tess_displacement_scale,
            displacement_bias: 0.0,
            max_verts: MAX_VERTS,
            has_displacement: 1,
            part_capacity: GEN_CAPACITY,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        classify.gen_kernel.push_blob(
            "tess_gen_verts",
            bytemuck::bytes_of(&gen_params),
            &[
                (
                    "part_triangles",
                    classify.gen_slots.buffer(seam, 0, &classify.part_triangles),
                ),
                ("configs", classify.gen_slots.buffer(seam, 1, &table.configs)),
                ("table_vertices", classify.gen_slots.buffer(seam, 2, &table.vertices)),
                ("base_positions", classify.gen_slots.buffer(seam, 3, positions)),
                (
                    "base_packed",
                    classify.gen_slots.buffer(seam, 4, mesh_manager.vertex_packed.buffer()),
                ),
                ("gen_vertices", classify.gen_slots.buffer(seam, 5, &classify.gen_vertices)),
            ],
        )
    });
    let attr_blob = attr_ready.then(|| {
        let attr_params = AttrParams {
            attr_addr_lo: classify.gen_attrs_addr as u32,
            attr_addr_hi: (classify.gen_attrs_addr >> 32) as u32,
            max_tris: table.max_triangles.max(1),
            part_capacity: classify.total_base_tris,
        };
        classify.attr_kernel.push_blob(
            "tess_gen_attrs",
            bytemuck::bytes_of(&attr_params),
            &[
                (
                    "part_triangles",
                    classify.attr_slots.buffer(seam, 0, &classify.part_triangles),
                ),
                ("configs", classify.attr_slots.buffer(seam, 1, &table.configs)),
                ("table_indices", classify.attr_slots.buffer(seam, 2, &table.indices)),
                ("table_vertices", classify.attr_slots.buffer(seam, 3, &table.vertices)),
                (
                    "base_packed",
                    classify.attr_slots.buffer(seam, 4, mesh_manager.vertex_packed.buffer()),
                ),
                ("gen_attrs", classify.attr_slots.buffer(seam, 5, gen_attrs_buf.as_ref().unwrap())),
                ("part_meta", classify.attr_slots.buffer(seam, 6, gen_attrs_meta_buf.as_ref().unwrap())),
            ],
        )
    });
    // Per-part CLAS-instantiate descriptors. Bounded by the CPU-known
    // base-tri count (the shader still guards `p >= counts[0]`); 64-wide.
    let inst_blob = inst_ready.then(|| {
        let addr = classify.gen_vertices_addr;
        let inst_params = InstParams {
            gen_base_lo: addr as u32,
            gen_base_hi: (addr >> 32) as u32,
            max_verts: MAX_VERTS,
            part_capacity: GEN_CAPACITY,
            // Sentinel ClusterIDNV base above the real cluster pool, so the
            // closest-hit detects tess hits.
            cluster_id_base: TESS_CLUSTER_ID_BASE,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        classify.inst_kernel.push_blob(
            "tess_instantiate",
            bytemuck::bytes_of(&inst_params),
            &[
                (
                    "part_triangles",
                    classify.inst_slots.buffer(seam, 0, &classify.part_triangles),
                ),
                ("counts", classify.inst_slots.buffer(seam, 1, &classify.counts)),
                (
                    "template_addresses",
                    classify.inst_slots.buffer(seam, 2, &table.template_addresses),
                ),
                (
                    "instantiate_infos",
                    classify.inst_slots.buffer(seam, 3, &classify.instantiate_infos),
                ),
            ],
        )
    });

    // The encoders below go out in ONE submit. Submission order within a
    // `vkQueueSubmit` is guaranteed, so each raw segment's leading barrier still
    // covers the segments recorded before it. The wgpu transition encoder stays
    // separate because the fork panics if one encoder mixes wgpu commands with
    // raw `as_hal_mut`.
    let mut submission = Vec::new();

    // Move the displacement textures to their sampled state (read-only optimal)
    // through wgpu's tracker BEFORE the raw gen dispatch samples them: their
    // last tracked use is the upload copy (COPY_DST), and with the gen pass now
    // raw no wgpu pass transitions them — the heap descriptors declare
    // SHADER_READ_ONLY_OPTIMAL. A no-op once the state already matches (same
    // idiom as the trace's skybox transition in `render/rt_pipeline/mod.rs`).
    if gen_ready {
        let mut transition_encoder =
            render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tess_classify.transitions"),
            });
        let mut seen = Vec::new();
        transition_encoder.transition_resources(
            core::iter::empty(),
            showcase
                .instances
                .iter()
                .filter(|inst| {
                    let id = inst.displacement.id();
                    !seen.contains(&id) && {
                        seen.push(id);
                        true
                    }
                })
                .filter_map(|inst| images.get(&inst.displacement))
                .map(|img| wgpu::TextureTransition {
                    // bevy `Texture` → the wrapped `wgpu::Texture`.
                    texture: &*img.texture,
                    selector: None,
                    state: wgpu::TextureUses::RESOURCE,
                }),
        );
        submission.push(transition_encoder.finish());
    }

    // Record the whole raw chain into ONE encoder: classify → finalize →
    // gen_verts (indirect) → gen_attrs (indirect) → instantiate.
    let groups = work_cluster_count.min(65535);
    let inst_groups = classify.total_base_tris.div_ceil(64).min(65535);
    let mut encoder = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tess_classify.dispatch"),
    });
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // barriers bracket every dispatch (each step reads the previous step's
    // writes — and the indirect args — invisibly to wgpu's tracking).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &classify.raw_device;
            let raw_gen_dispatch = classify
                .gen_dispatch
                .as_hal::<VkApi>()
                .map(|b| b.raw_handle())
                .expect("bevy_solari requires the Vulkan backend");
            // One barrier serves every edge in the chain: the queue's counts
            // clear + list uploads (transfer) and each step's compute writes →
            // the next step's storage access AND its indirect-args read.
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                )
                .src_access_mask(
                    vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_WRITE,
                )
                .dst_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::DRAW_INDIRECT,
                )
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_READ
                        | vk::AccessFlags2::SHADER_WRITE
                        | vk::AccessFlags2::INDIRECT_COMMAND_READ,
                )];
            let dep = vk::DependencyInfo::default().memory_barriers(&barrier);
            // The counts clear + work/slot-list uploads -> classify's access.
            dev.cmd_pipeline_barrier2(cb, &dep);
            seam.bind_heaps(cb);
            seam.push_data(cb, &classify_blob);
            dev.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                classify.classify_kernel.pipeline,
            );
            dev.cmd_dispatch(cb, groups, 1, 1);
            if let Some(gen_blob) = gen_blob.as_ref() {
                // classify's counts write -> finalize's read.
                dev.cmd_pipeline_barrier2(cb, &dep);
                seam.push_data(cb, &finalize_blob);
                dev.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    classify.finalize_kernel.pipeline,
                );
                dev.cmd_dispatch(cb, 1, 1, 1);
                // finalize's indirect args + classify's part list -> gen verts:
                // one workgroup per part (indirect).
                dev.cmd_pipeline_barrier2(cb, &dep);
                seam.push_data(cb, gen_blob);
                dev.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    classify.gen_kernel.pipeline,
                );
                dev.cmd_dispatch_indirect(cb, raw_gen_dispatch, 0);
                // shading: per-micro-triangle smooth normals + UVs + per-part
                // metadata. Same indirect grid as gen_verts (one workgroup per
                // part); reads part_triangles + the table topology, independent
                // of the gen_verts positions.
                if let Some(attr_blob) = attr_blob.as_ref() {
                    dev.cmd_pipeline_barrier2(cb, &dep);
                    seam.push_data(cb, attr_blob);
                    dev.cmd_bind_pipeline(
                        cb,
                        vk::PipelineBindPoint::COMPUTE,
                        classify.attr_kernel.pipeline,
                    );
                    dev.cmd_dispatch_indirect(cb, raw_gen_dispatch, 0);
                }
                if let Some(inst_blob) = inst_blob.as_ref() {
                    // The gen passes' part/vertex writes -> the descriptor build.
                    dev.cmd_pipeline_barrier2(cb, &dep);
                    seam.push_data(cb, inst_blob);
                    dev.cmd_bind_pipeline(
                        cb,
                        vk::PipelineBindPoint::COMPUTE,
                        classify.inst_kernel.pipeline,
                    );
                    dev.cmd_dispatch(cb, inst_groups, 1, 1);
                }
                // The chain's writes -> the raw INSTANTIATE build's input (its
                // own seam covers build-input visibility on top).
                dev.cmd_pipeline_barrier2(cb, &dep);
            }
        });
    }
    submission.push(encoder.finish());

    // Raw-VK indirect INSTANTIATE_TRIANGLE_CLUSTER — turn each part's GPU-built
    // descriptor + gen_vertices slice into a CLAS in the persistent pool. Must be its
    // OWN encoder (wgpu forbids mixing raw + wgpu-pass commands in one encoder); the
    // opening `as_barrier` synchronizes against the prior submit's compute writes
    // (same queue, submission order preserved). `src_infos_count = counts[0]` (the
    // GPU actual count); bounded by the size-query's `total_base_tris`.
    if clas_build_ready {
        let alloc = allocator.as_ref().unwrap();
        let fns_res = fns.as_ref().unwrap();
        let count = classify.total_base_tris;
        let counts_addr = alloc.wgpu_buffer_device_address(&classify.counts).get();
        let src_addr = alloc.wgpu_buffer_device_address(&classify.instantiate_infos).get();
        let mut tri_input = vk::ClusterAccelerationStructureTriangleClusterInputNV::default();
        let input = tess_instantiate_input(
            &mut tri_input,
            table.max_triangles,
            table.max_vertices,
            count,
            vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS,
        );
        let cmd = vk::ClusterAccelerationStructureCommandsInfoNV {
            s_type: vk::ClusterAccelerationStructureCommandsInfoNV::STRUCTURE_TYPE,
            p_next: core::ptr::null_mut(),
            input,
            dst_implicit_data: classify.tess_clas_storage_addr,
            scratch_data: classify.tess_clas_scratch_addr,
            dst_addresses_array: vk::StridedDeviceAddressRegionKHR {
                device_address: classify.tess_clas_addresses_addr,
                stride: 8,
                size: (count as u64) * 8,
            },
            dst_sizes_array: vk::StridedDeviceAddressRegionKHR::default(),
            src_infos_array: vk::StridedDeviceAddressRegionKHR {
                device_address: src_addr,
                stride: 32,
                size: (count as u64) * 32,
            },
            src_infos_count: counts_addr,
            address_resolution_flags:
                vk::ClusterAccelerationStructureAddressResolutionFlagsNV::default(),
            _marker: core::marker::PhantomData,
        };
        let mut build_encoder =
            render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tess_classify.clas_build"),
            });
        // SAFETY: raw-only encoder; the opening seam makes the prior submit's
        // descriptor/counts shader writes visible to the build; `tri_input` lives
        // through the record call (input maxima consumed at record time); all
        // addresses reference resident allocator buffers. `tess_clas_storage` is
        // rewritten in place at stable addresses every frame while the previous
        // frame's trace may still be walking BLASes that reference it.
        unsafe {
            crate::gpu::extension::cmd_as_seam(
                &mut build_encoder,
                &render_device,
                AsSeams::COMPUTE_TO_BUILD_INPUT | AsSeams::TRACE_TO_BUILD_WAR,
            );
            crate::gpu::extension::cmd_build_cluster_acceleration_structures_indirect(
                &mut build_encoder,
                fns_res,
                &cmd,
            );
            crate::gpu::extension::cmd_as_seam(&mut build_encoder, &render_device, AsSeams::BUILD_TO_BUILD_INPUT);
        }
        submission.push(build_encoder.finish());
    }

    // Build one BLAS per instance from its contiguous CLAS-address sublist.
    // Gated on the CLAS build + the BLAS-address buffers.
    let c3_ready = clas_build_ready
        && classify.blas_addresses.is_some()
        && classify.tess_clas_addresses.is_some();
    if c3_ready {
        let alloc = allocator.as_ref().unwrap();
        let fns_res = fns.as_ref().unwrap();

        // Per-instance BLAS: one BLAS from each instance's contiguous sublist of
        // `tess_clas_addresses` — the INSTANTIATE writes the per-part CLAS addresses
        // at DETERMINISTIC indices already grouped by instance, so the flat address
        // array is directly each BLAS's `cluster_references` (raw-only encoder;
        // `record_build_per_instance_blas` leads with an AS-barrier that syncs
        // against the CLAS-build submit). The driver writes each BLAS address into
        // `blas_addresses[i]` GPU-side.
        let mut blas_encoder =
            render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tess_classify.blas_build"),
            });
        let mut built: Vec<InstanceBlas> = Vec::new();
        for (i, &cnt) in classify.per_instance_counts.iter().enumerate() {
            if cnt == 0 {
                continue;
            }
            let refs_addr =
                classify.tess_clas_addresses_addr + (classify.per_instance_offsets[i] as u64) * 8;
            let blas = record_build_per_instance_blas(
                &render_device,
                &render_queue,
                alloc,
                fns_res,
                &mut blas_encoder,
                refs_addr,
                cnt,
                classify.blas_addresses_addr + (i as u64) * 8,
            );
            built.push(blas);
        }
        // One publish for the whole set: each build already carries a leading
        // seam, so consecutive builds are separated, and only the last one needs
        // its bytes made visible to traversal.
        // SAFETY: raw-only encoder, still open + Vulkan-backed.
        unsafe {
            crate::gpu::extension::cmd_as_seam(&mut blas_encoder, &render_device, AsSeams::BUILD_TO_TRACE);
        }
        submission.push(blas_encoder.finish());
        classify.blas_keepalive.push(built);
        // The BLAS addresses are now valid in `blas_addresses` — the PTLAS inject may run.
        classify.blas_ready = true;
    }

    render_queue.submit(submission);

    // Old BLAS sets outlive their last consuming submit via the reaper
    // (frame-count keepalives get outrun; completion flags don't). Armed after
    // the submit above, so `on_submitted_work_done` covers the builds that read
    // them.
    if classify.blas_keepalive.len() > 2 {
        let old = classify.blas_keepalive.remove(0);
        retire.retire(&render_queue, "tess_classify.blas_set", old);
    }

}
