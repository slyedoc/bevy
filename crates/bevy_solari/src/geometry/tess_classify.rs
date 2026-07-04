// Phase B of the vk_tessellated_clusters port: GPU per-base-triangle classify.
//
// A self-contained compute pass (own bind group + submit, like the tess showcase)
// that classifies every tessellated instance's base triangles into per-edge
// tessellation factors and emits a work list (`part_triangles`) keyed to the
// Phase-A [`TessellationTable`] patterns. Phase C consumes the list to displace +
// instantiate. See `tess_classify.wgsl` for the per-triangle math.
#![allow(clippy::type_complexity)]
// Explicit `wgpu::` qualification on render-resource types (some are also glob-
// re-exported via `render_resource::*`); keep the prefix for clarity at call sites.
#![allow(unused_qualifications)]
#![allow(unsafe_code, reason = "raw VK cluster-AS instantiate via the extension fns")]

use bevy_ecs::{
    query::With,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::{Mat4, Vec2};
use bevy_render::{
    camera::ExtractedCamera,
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::GpuImage,
    view::ExtractedView,
};
use bytemuck::{Pod, Zeroable};

use ash::vk::{self, TaggedStructure};

use crate::gpu::retire::GpuRetire;
use crate::gpu::allocator::{Allocator, MemoryLocation};
use crate::gpu::extension::ClusterExtensionFns;
use super::clas_arena::CLAS_SCRATCH_ALIGN;
use super::mesh_manager::ClusterMeshManager;
use super::tess_displace::TessShowcaseInstances;
use super::tess_table::TessellationTable;
use super::tess_template::{record_build_per_instance_blas, InstanceBlas, TESS_CLUSTER_ID_BASE};

/// Round `addr` up to `align` (power of two).
fn align_up(addr: u64, align: u64) -> u64 {
    (addr + (align - 1)) & !(align - 1)
}

/// Target screen pixels per edge segment (lower ⇒ denser tessellation, more height
/// detail). The view-dependent factor for an edge is `round(edge_pixels / this)`,
/// clamped to `1..=max_size`. Safe to lower freely: the CLAS pool + gen buffers are
/// pre-sized for every part at the table's MAX config, so denser tessellation only
/// costs GPU work, never memory.
const PX_PER_SEGMENT: f32 = 6.0;
/// Max `part_triangles` the work list holds (one per classified base triangle).
const PART_CAPACITY: u32 = 1 << 20;
/// Capacity (in parts) of the Phase-C gen-vertex / (later) CLAS pools. Bounded so
/// the GPU build can't explode like the CPU path — `gen_vertices` is
/// `GEN_CAPACITY * MAX_VERTS * 12` bytes. Sized for the displacement instances.
const GEN_CAPACITY: u32 = 1 << 18; // 262144 parts
/// Fixed micro-vertices per part slot (the table's max, validated by Phase A).
const MAX_VERTS: u32 = 78;
/// Max distinct tessellated instances the per-instance displacement binding array
/// reserves (partially bound — only the live instances are filled). The gen pass
/// indexes it by `instance_index`, so it must cover the showcase instance count.
/// MUST equal the sized `binding_array<texture_2d<f32>, N>` in `tess_gen_verts.wgsl`.
const MAX_TESS_DISPLACEMENT_MAPS: u32 = 256;
/// Displacement height (object units); `SOLARI_TESS_SCALE` overrides.
fn displacement_scale() -> f32 {
    std::env::var("SOLARI_TESS_SCALE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.05)
}

/// `tess_classify.wgsl::Params` mirror. `ShaderType` lays it out std140 to match
/// the uniform on the shader side.
#[derive(Clone, Copy, ShaderType, Default)]
pub struct ClassifyParams {
    pub clip_from_world: Mat4,
    pub viewport: Vec2,
    pub px_per_segment: f32,
    pub work_cluster_count: u32,
    pub max_size: u32,
    pub max_size_configs: u32,
    pub part_capacity: u32,
    pub _pad: u32,
}

/// `tess_classify.wgsl::Instance` — object→world affine, row-major mat3x4.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct InstanceGpu {
    rows: [[f32; 4]; 3],
}

/// `tess_classify.wgsl::WorkCluster`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct WorkClusterGpu {
    instance_idx: u32,
    index_base: u32,
    triangle_count: u32,
    /// Global base of this cluster's vertices in the shared pool. The stored index
    /// values are SOURCE-CLUSTER-LOCAL (the mesh manager rebases `index_offset` /
    /// `vertex_offset` to the global pool but leaves the index *values* local, like
    /// the chit + `tess_displace`), so `classify` must add this to every fetched
    /// index to land on the right global vertex.
    vertex_base: u32,
    /// Prefix-sum base part index for this cluster (Σ prior clusters' `triangle_count`).
    /// Part index = `part_base + triangle`, a DETERMINISTIC slot (every base triangle
    /// emits exactly one part) — so a part's `cluster_id` (`TESS_CLUSTER_ID_BASE + idx`)
    /// is stable across frames, unlike the old global-atomic append order (which
    /// strobed the cluster-debug colors and would desync per-part metadata).
    part_base: u32,
}

/// `tess_gen_verts.wgsl::GenParams` mirror.
#[derive(Clone, Copy, ShaderType, Default)]
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

/// `tess_gen_attrs.wgsl::AttrParams` mirror.
#[derive(Clone, Copy, ShaderType, Default)]
pub struct AttrParams {
    pub attr_addr_lo: u32,
    pub attr_addr_hi: u32,
    pub max_tris: u32,
    pub part_capacity: u32,
}

/// `tess_instantiate.wgsl::InstParams` mirror.
#[derive(Clone, Copy, ShaderType, Default)]
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

/// Render-world resource for the classify + (Phase C) vertex-gen passes.
#[derive(Resource)]
pub struct TessClassify {
    pub pipeline: CachedComputePipelineId,
    pub layout: BindGroupLayoutDescriptor,
    /// `finalize` entry of the classify shader → writes `gen_dispatch` indirect args.
    finalize_pipeline: CachedComputePipelineId,
    params: UniformBuffer<ClassifyParams>,
    instances: RawBufferVec<InstanceGpu>,
    work_clusters: RawBufferVec<WorkClusterGpu>,
    /// `[part, full, split]` atomic counters (part is also the append cursor).
    counts: Buffer,
    /// Emitted [`tess_classify.wgsl::TessTriangleInfo`] work list (Phase C input).
    pub part_triangles: Buffer,
    /// `DispatchIndirectCommand` for the gen pass (one workgroup per part).
    gen_dispatch: Buffer,
    /// Work clusters this resource's CPU lists were last built for (rebuilt when
    /// the instance set changes — the showcase instance set is latch-stable).
    built_instances: usize,
    bind_group: Option<BindGroup>,

    // ── Phase C step 1: micro-vertex generation ─────────────────────────────
    gen_pipeline: CachedComputePipelineId,
    gen_layout: BindGroupLayoutDescriptor,
    gen_params: StorageBuffer<GenParams>,
    /// World-space micro-vertices, stride 3 f32, slot `part*MAX_VERTS + v`. Phase C
    /// step 2 instantiates each part's template against its slice of this.
    pub gen_vertices: Buffer,
    gen_bind_group: Option<BindGroup>,

    // ── Phase C shading: per-micro-triangle smooth normals + UVs ─────────────
    attr_pipeline: CachedComputePipelineId,
    attr_layout: BindGroupLayoutDescriptor,
    attr_params: UniformBuffer<AttrParams>,
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
    attr_bind_group: Option<BindGroup>,

    // ── Phase C step 2a: per-part CLAS instantiate descriptors ───────────────
    instantiate_pipeline: CachedComputePipelineId,
    instantiate_layout: BindGroupLayoutDescriptor,
    inst_params: UniformBuffer<InstParams>,
    /// One `VkClusterAccelerationStructureInstantiateClusterInfoNV` (8 u32 / 32 B)
    /// per emitted part — consumed by the step-2b raw-VK indirect INSTANTIATE.
    pub instantiate_infos: Buffer,
    instantiate_bind_group: Option<BindGroup>,
    /// Device address of `gen_vertices` (resolved once the allocator is present);
    /// baked into each descriptor's `vertex_buffer.start_address`.
    gen_vertices_addr: u64,
    /// Total base-triangle count of the latched work set (= the emitted part count,
    /// constant per instance set) — bounds the descriptor-build dispatch CPU-side.
    total_base_tris: u32,

    // ── Phase C step 2b: raw-VK indirect INSTANTIATE → tess CLAS pool ─────────
    /// Persistent CLAS storage (implicit-dst), sized once from the instantiate size
    /// query for `tess_clas_sized_for` parts and reused every frame.
    tess_clas_storage: Option<Buffer>,
    tess_clas_scratch: Option<Buffer>,
    /// GPU-written per-part CLAS device addresses (the step-3 BLAS input).
    pub tess_clas_addresses: Option<Buffer>,
    tess_clas_storage_addr: u64,
    tess_clas_scratch_addr: u64,
    tess_clas_addresses_addr: u64,
    /// Part count the CLAS pool was sized for (re-query + realloc if it changes).
    tess_clas_sized_for: u32,

    // ── Phase C step 3a: group CLAS by instance → per-instance BLAS ──────────
    scatter_pipeline: CachedComputePipelineId,
    scatter_layout: BindGroupLayoutDescriptor,
    /// Per-instance contiguous CLAS-address array (the BLAS `cluster_references`).
    /// Identity copy of `clas_addresses` (parts are already grouped by instance).
    references: Option<Buffer>,
    references_addr: u64,
    scatter_bind_group: Option<BindGroup>,
    /// Per-instance prefix-sum offsets (CPU), `[0, c0, c0+c1, …]`, len = num_instances.
    per_instance_offsets: Vec<u32>,
    /// Per-instance base-tri (= part) counts, len = num_instances.
    per_instance_counts: Vec<u32>,
    // ── Phase C step 3b: per-instance BLAS addresses ─────────────────────────
    /// GPU-written per-instance BLAS device addresses (the PTLAS-write input).
    pub blas_addresses: Option<Buffer>,
    blas_addresses_addr: u64,
    /// Keep this + last frame's `InstanceBlas` storage alive while their builds /
    /// any in-flight trace complete (2-deep ring; nothing traces them yet).
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
}

/// `RenderStartup`: layout + pipeline + persistent output buffers.
pub fn init_tess_classify(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
) {
    if allocator.is_none() {
        return;
    }
    let layout = BindGroupLayoutDescriptor::new(
        "tess_classify_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, None),           // 0 params
                storage_buffer_read_only_sized(false, None), // 1 instances
                storage_buffer_read_only_sized(false, None), // 2 work_clusters
                storage_buffer_read_only_sized(false, None), // 3 vertex_positions
                storage_buffer_read_only_sized(false, None), // 4 indices
                storage_buffer_sized(false, None),           // 5 counts (rw)
                storage_buffer_sized(false, None),           // 6 part_triangles (rw)
                storage_buffer_sized(false, None),           // 7 gen_dispatch (rw)
            ),
        ),
    );
    let classify_shader =
        bevy_asset::load_embedded_asset!(asset_server.as_ref(), "tess_classify.wgsl");
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("tess_classify".into()),
        layout: vec![layout.clone()],
        shader: classify_shader.clone(),
        shader_defs: vec![],
        entry_point: Some("classify".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    let finalize_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("tess_classify_finalize".into()),
        layout: vec![layout.clone()],
        shader: classify_shader,
        shader_defs: vec![],
        entry_point: Some("finalize".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    // Phase C step 1: vertex-gen layout + pipeline.
    let gen_layout = BindGroupLayoutDescriptor::new(
        "tess_gen_verts_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // Storage, not uniform: a bind group can't mix a uniform buffer with
                // the binding array at slot 7.
                storage_buffer_read_only_sized(false, None), // 0 gen params
                storage_buffer_read_only_sized(false, None), // 1 part_triangles
                storage_buffer_read_only_sized(false, None), // 2 configs
                storage_buffer_read_only_sized(false, None), // 3 table_vertices
                storage_buffer_read_only_sized(false, None), // 4 base_positions
                storage_buffer_read_only_sized(false, None), // 5 base_packed
                storage_buffer_read_only_sized(false, None), // 6 instances
                // 7 per-instance displacement maps (binding array, partially bound).
                texture_2d(TextureSampleType::Float { filterable: true })
                    .count(core::num::NonZero::new(MAX_TESS_DISPLACEMENT_MAPS).unwrap()),
                sampler(SamplerBindingType::Filtering),      // 8 sampler
                storage_buffer_sized(false, None),           // 9 gen_vertices (rw)
            ),
        ),
    );
    let gen_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("tess_gen_verts".into()),
        layout: vec![gen_layout.clone()],
        shader: bevy_asset::load_embedded_asset!(asset_server.as_ref(), "tess_gen_verts.wgsl"),
        shader_defs: vec![],
        entry_point: Some("gen_verts".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    let counts = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tess_classify.counts"),
        size: 16, // 3 u32 + pad
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
        size: (PART_CAPACITY as u64) * 32, // TessTriangleInfo = 32 B
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
        // CLAS vertex source by device address. COPY_SRC for the gen readback validation.
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Phase C shading: per-micro-triangle smooth-normal/UV layout + pipeline.
    let attr_layout = BindGroupLayoutDescriptor::new(
        "tess_gen_attrs_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, None),           // 0 attr params
                storage_buffer_read_only_sized(false, None), // 1 part_triangles
                storage_buffer_read_only_sized(false, None), // 2 configs
                storage_buffer_read_only_sized(false, None), // 3 table_indices
                storage_buffer_read_only_sized(false, None), // 4 table_vertices
                storage_buffer_read_only_sized(false, None), // 5 base_packed
                storage_buffer_sized(false, None),           // 6 gen_attrs (rw)
                storage_buffer_sized(false, None),           // 7 meta (rw)
                storage_buffer_read_only_sized(false, None), // 8 instances
            ),
        ),
    );
    let attr_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("tess_gen_attrs".into()),
        layout: vec![attr_layout.clone()],
        shader: bevy_asset::load_embedded_asset!(asset_server.as_ref(), "tess_gen_attrs.wgsl"),
        shader_defs: vec![],
        entry_point: Some("gen_attrs_main".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    // Phase C step 2a: per-part CLAS-instantiate descriptor builder.
    let instantiate_layout = BindGroupLayoutDescriptor::new(
        "tess_instantiate_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, None),           // 0 inst params
                storage_buffer_read_only_sized(false, None), // 1 part_triangles
                storage_buffer_read_only_sized(false, None), // 2 counts
                storage_buffer_read_only_sized(false, None), // 3 template_addresses
                storage_buffer_sized(false, None),           // 4 instantiate_infos (rw)
            ),
        ),
    );
    let instantiate_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("tess_instantiate".into()),
        layout: vec![instantiate_layout.clone()],
        shader: bevy_asset::load_embedded_asset!(asset_server.as_ref(), "tess_instantiate.wgsl"),
        shader_defs: vec![],
        entry_point: Some("build_infos".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    // Phase C step 3a: CLAS-by-instance scatter.
    let scatter_layout = BindGroupLayoutDescriptor::new(
        "tess_scatter_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 clas_addresses
                storage_buffer_sized(false, None),           // 1 references (rw)
                storage_buffer_read_only_sized(false, None), // 2 counts
            ),
        ),
    );
    let scatter_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("tess_scatter".into()),
        layout: vec![scatter_layout.clone()],
        shader: bevy_asset::load_embedded_asset!(asset_server.as_ref(), "tess_scatter.wgsl"),
        shader_defs: vec![],
        entry_point: Some("scatter".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

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

    let mut params = UniformBuffer::<ClassifyParams>::default();
    params.set_label(Some("tess_classify.params"));
    let mut gen_params = StorageBuffer::<GenParams>::default();
    gen_params.set_label(Some("tess_classify.gen_params"));
    let mut attr_params = UniformBuffer::<AttrParams>::default();
    attr_params.set_label(Some("tess_classify.attr_params"));
    let mut inst_params = UniformBuffer::<InstParams>::default();
    inst_params.set_label(Some("tess_classify.inst_params"));
    let mut instances = RawBufferVec::<InstanceGpu>::new(wgpu::BufferUsages::STORAGE);
    instances.set_label(Some("tess_classify.instances"));
    let mut work_clusters = RawBufferVec::<WorkClusterGpu>::new(wgpu::BufferUsages::STORAGE);
    work_clusters.set_label(Some("tess_classify.work_clusters"));

    commands.insert_resource(TessClassify {
        pipeline,
        layout,
        finalize_pipeline,
        params,
        instances,
        work_clusters,
        counts,
        part_triangles,
        gen_dispatch,
        built_instances: usize::MAX,
        bind_group: None,
        gen_pipeline,
        gen_layout,
        gen_params,
        gen_vertices,
        gen_bind_group: None,
        attr_pipeline,
        attr_layout,
        attr_params,
        gen_attrs: None,
        gen_attrs_meta: None,
        gen_attrs_addr: 0,
        gen_attrs_meta_addr: 0,
        attr_bind_group: None,
        instantiate_pipeline,
        instantiate_layout,
        inst_params,
        instantiate_infos,
        instantiate_bind_group: None,
        gen_vertices_addr: 0,
        total_base_tris: 0,
        tess_clas_storage: None,
        tess_clas_scratch: None,
        tess_clas_addresses: None,
        tess_clas_storage_addr: 0,
        tess_clas_scratch_addr: 0,
        tess_clas_addresses_addr: 0,
        tess_clas_sized_for: 0,
        scatter_pipeline,
        scatter_layout,
        references: None,
        references_addr: 0,
        scatter_bind_group: None,
        per_instance_offsets: Vec::new(),
        per_instance_counts: Vec::new(),
        blas_addresses: None,
        blas_addresses_addr: 0,
        blas_keepalive: Vec::new(),
        blas_sized_for: 0,
        blas_ready: false,
        diag: 0,
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

/// `Render::Prepare`: build the CPU work lists (cached), write params from the
/// camera, (re)build the bind group, then record + submit the classify dispatch.
/// Self-submitting like the tess showcase, so no render-graph wiring is needed.
pub fn run_tess_classify(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut retire: ResMut<GpuRetire>,
    pipeline_cache: Res<PipelineCache>,
    mut classify: Option<ResMut<TessClassify>>,
    showcase: Option<Res<TessShowcaseInstances>>,
    mesh_manager: Option<Res<ClusterMeshManager>>,
    table: Option<Res<TessellationTable>>,
    images: Res<RenderAssets<GpuImage>>,
    allocator: Option<Res<Allocator>>,
    fns: Option<Res<ClusterExtensionFns>>,
    views: Query<&ExtractedView, With<ExtractedCamera>>,
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
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(classify.pipeline) else {
        if chatty {
            tracing::debug!("tess_classify bail: classify pipeline not ready (still compiling/failed)");
        }
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
    // tessellation stable when still; it still adapts as the camera actually moves.
    // Mirrors the rt_pipeline's unjittered `clip_from_world` (used for motion vectors).
    let clip_from_world = view.clip_from_view * view.world_from_view.to_matrix().inverse();

    // Build the per-instance + per-cluster work lists for this instance set. Rebuilt
    // when the set changes — but do NOT latch until the meshes are actually resident
    // (`tess_clusters` populated), else an early empty build would stick forever.
    if classify.built_instances != showcase.instances.len() {
        classify.instances.clear();
        classify.work_clusters.clear();
        // Running prefix sum of base-triangle counts → each cluster's deterministic
        // part-index base (so `part = part_base + triangle` is stable across frames).
        let mut part_base = 0u32;
        for (i, inst) in showcase.instances.iter().enumerate() {
            classify.instances.push(InstanceGpu {
                rows: inst.world_from_local,
            });
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
        // emitted part count (the constant `part=…` in the readback) — used to bound the
        // step-2a descriptor-build dispatch CPU-side without an extra indirect pass.
        classify.total_base_tris = classify
            .work_clusters
            .values()
            .iter()
            .map(|c| c.triangle_count)
            .sum();
        // Per-instance part counts (= base-tri counts) + prefix-sum base offsets into
        // `references` — drives the step-3a scatter + per-instance BLAS sublists.
        let num_instances = showcase.instances.len();
        let mut counts = vec![0u32; num_instances];
        for c in classify.work_clusters.values() {
            if (c.instance_idx as usize) < num_instances {
                counts[c.instance_idx as usize] += c.triangle_count;
            }
        }
        let mut offsets = vec![0u32; num_instances];
        let mut acc = 0u32;
        for (o, &c) in offsets.iter_mut().zip(counts.iter()) {
            *o = acc;
            acc += c;
        }
        classify.per_instance_counts = counts;
        classify.per_instance_offsets = offsets;
        classify.instances.write_buffer(&render_device, &render_queue);
        classify.work_clusters.write_buffer(&render_device, &render_queue);
        classify.built_instances = showcase.instances.len();
        tracing::debug!(
            "tess_classify: built work lists — {} instances, {} work clusters, {} base tris",
            classify.instances.len(),
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

    // Phase C step 2b: size + allocate the tess CLAS pool once per instance set (the part
    // count is fixed = total_base_tris). The instantiate size query bounds the implicit-dst
    // storage for the worst case (every part its max-size template); the actual per-frame
    // build uses the GPU `counts[0]` as `src_infos_count`.
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

                // Step 3a/3b buffers, same instance-set lifetime: the per-instance scatter
                // `references` (total_base_tris × u64 CLAS addr) + the per-instance BLAS
                // addresses. Parts are already grouped by instance (deterministic index),
                // so the scatter is an identity copy — no base-offset / cursor buffers.
                let num_instances = classify.per_instance_counts.len().max(1) as u32;
                let references = alloc.create_buffer(
                    &render_device,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    (count as u64) * 8,
                    MemoryLocation::GpuOnly,
                    "tess_classify.references",
                );
                classify.references_addr = alloc.wgpu_buffer_device_address(&references).get();
                if let Some(old) = classify.references.replace(references.into()) {
                    retire.retire(&render_queue, "tess_classify.references", old);
                }
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

    let viewport = Vec2::new(view.viewport.z as f32, view.viewport.w as f32);
    *classify.params.get_mut() = ClassifyParams {
        clip_from_world,
        viewport,
        px_per_segment: PX_PER_SEGMENT,
        work_cluster_count,
        max_size: table.max_size,
        max_size_configs: table.max_size_configs,
        part_capacity: PART_CAPACITY,
        _pad: 0,
    };
    classify.params.write_buffer(&render_device, &render_queue);

    // Clear the counters each frame (CPU write — tiny).
    render_queue.write_buffer(&classify.counts, 0, &[0u8; 16]);

    // (Re)build the bind group (cheap; the buffers are stable, but the geometry
    // pool buffers can grow/realloc, so rebuild each run).
    let (Some(params_binding), Some(instances_buf), Some(work_buf)) = (
        classify.params.binding(),
        classify.instances.buffer(),
        classify.work_clusters.buffer(),
    ) else {
        if chatty {
            tracing::debug!("tess_classify bail: classify bind-group buffers not allocated yet");
        }
        return;
    };
    let bind_group = render_device.create_bind_group(
        "tess_classify_bind_group",
        &pipeline_cache.get_bind_group_layout(&classify.layout),
        &BindGroupEntries::sequential((
            params_binding,
            instances_buf.as_entire_binding(),
            work_buf.as_entire_binding(),
            positions.as_entire_binding(),
            indices.as_entire_binding(),
            classify.counts.as_entire_binding(),
            classify.part_triangles.as_entire_binding(),
            classify.gen_dispatch.as_entire_binding(),
        )),
    );
    classify.bind_group = Some(bind_group);

    // Phase C step 1 wiring: resolve the (single) displacement map + build the
    // gen-verts bind group. Gated on the displacement being resident (the showcase
    // sets it); without it we still run classify but skip vertex gen.
    let gen_pipeline = pipeline_cache.get_compute_pipeline(classify.gen_pipeline);
    let finalize_pipeline = pipeline_cache.get_compute_pipeline(classify.finalize_pipeline);
    // Per-instance displacement: gather EVERY instance's own map into the binding
    // array (indexed by `instance_index` in the gen shader). All must be resident —
    // the showcase latches only once materials load, so they are; otherwise skip the
    // gen this frame and retry. A shared filtering sampler serves all maps.
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
    let gen_ready = match (gen_pipeline, finalize_pipeline, disp_views, disp_sampler) {
        (Some(_), Some(_), Some(disp_views), Some(disp_sampler))
            if !disp_views.is_empty() && disp_views.len() <= MAX_TESS_DISPLACEMENT_MAPS as usize =>
        {
            *classify.gen_params.get_mut() = GenParams {
                displacement_scale: displacement_scale(),
                displacement_bias: 0.0,
                max_verts: MAX_VERTS,
                has_displacement: 1,
                part_capacity: GEN_CAPACITY,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            classify.gen_params.write_buffer(&render_device, &render_queue);
            if let Some(gp) = classify.gen_params.binding() {
                let gbg = render_device.create_bind_group(
                    "tess_gen_verts_bind_group",
                    &pipeline_cache.get_bind_group_layout(&classify.gen_layout),
                    &BindGroupEntries::sequential((
                        gp,
                        classify.part_triangles.as_entire_binding(),
                        table.configs.as_entire_binding(),
                        table.vertices.as_entire_binding(),
                        positions.as_entire_binding(),
                        mesh_manager.vertex_packed.buffer().as_entire_binding(),
                        instances_buf.as_entire_binding(),
                        disp_views.as_slice(),
                        disp_sampler,
                        classify.gen_vertices.as_entire_binding(),
                    )),
                );
                classify.gen_bind_group = Some(gbg);
                true
            } else {
                false
            }
        }
        _ => false,
    };

    // Phase C shading: build the attr-pass bind group + params. Produces the smooth
    // normals + UVs (denormalized per micro-triangle) + the per-part metadata the
    // closest-hit reads via `geometry_addresses.tess_clusters`. Gated on the gen pass
    // (shares its inputs + indirect grid) + the attr buffers being sized. Clone the
    // buffer handles (Arc) so they don't borrow `classify` across the params write.
    let gen_attrs_buf = classify.gen_attrs.clone();
    let gen_attrs_meta_buf = classify.gen_attrs_meta.clone();
    let attr_pipeline_ready = pipeline_cache.get_compute_pipeline(classify.attr_pipeline).is_some();
    let attr_ready = if gen_ready && attr_pipeline_ready {
        if let (Some(gen_attrs), Some(meta)) = (gen_attrs_buf.as_ref(), gen_attrs_meta_buf.as_ref()) {
            *classify.attr_params.get_mut() = AttrParams {
                attr_addr_lo: classify.gen_attrs_addr as u32,
                attr_addr_hi: (classify.gen_attrs_addr >> 32) as u32,
                max_tris: table.max_triangles.max(1),
                part_capacity: classify.total_base_tris,
            };
            classify.attr_params.write_buffer(&render_device, &render_queue);
            if let Some(ap) = classify.attr_params.binding() {
                let abg = render_device.create_bind_group(
                    "tess_gen_attrs_bind_group",
                    &pipeline_cache.get_bind_group_layout(&classify.attr_layout),
                    &BindGroupEntries::sequential((
                        ap,
                        classify.part_triangles.as_entire_binding(),
                        table.configs.as_entire_binding(),
                        table.indices.as_entire_binding(),
                        table.vertices.as_entire_binding(),
                        mesh_manager.vertex_packed.buffer().as_entire_binding(),
                        gen_attrs.as_entire_binding(),
                        meta.as_entire_binding(),
                        instances_buf.as_entire_binding(),
                    )),
                );
                classify.attr_bind_group = Some(abg);
                true
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    };

    // Phase C step 2a: build one CLAS-instantiate descriptor per emitted part. Gated on
    // the gen pass running (it fills `gen_vertices`) + the allocator-resolved gen address.
    let instantiate_pipeline = pipeline_cache.get_compute_pipeline(classify.instantiate_pipeline);
    let inst_ready = match (gen_ready, instantiate_pipeline, classify.gen_vertices_addr) {
        (true, Some(_), addr) if addr != 0 => {
            *classify.inst_params.get_mut() = InstParams {
                gen_base_lo: addr as u32,
                gen_base_hi: (addr >> 32) as u32,
                max_verts: MAX_VERTS,
                part_capacity: GEN_CAPACITY,
                // Sentinel ClusterIDNV base above the real cluster pool (step 2b/3 shades
                // tess CLAS via the facet normal, like the CPU showcase).
                cluster_id_base: TESS_CLUSTER_ID_BASE,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            classify.inst_params.write_buffer(&render_device, &render_queue);
            if let Some(ip) = classify.inst_params.binding() {
                let ibg = render_device.create_bind_group(
                    "tess_instantiate_bind_group",
                    &pipeline_cache.get_bind_group_layout(&classify.instantiate_layout),
                    &BindGroupEntries::sequential((
                        ip,
                        classify.part_triangles.as_entire_binding(),
                        classify.counts.as_entire_binding(),
                        table.template_addresses.as_entire_binding(),
                        classify.instantiate_infos.as_entire_binding(),
                    )),
                );
                classify.instantiate_bind_group = Some(ibg);
                true
            } else {
                false
            }
        }
        _ => false,
    };

    // step 2b readiness: the descriptors built (inst_ready), the cluster-AS fns loaded,
    // and the CLAS pool sized for the current part count.
    let clas_build_ready = inst_ready
        && fns.as_ref().and_then(|f| f.cluster.as_ref()).is_some()
        && classify.tess_clas_storage.is_some()
        && classify.tess_clas_sized_for == classify.total_base_tris
        && allocator.is_some();

    // Record classify → finalize → gen (separate passes, so wgpu inserts the
    // storage→indirect barriers) and submit once.
    let groups = work_cluster_count.min(65535);
    let mut encoder = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tess_classify.dispatch"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tess_classify"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, classify.bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    if gen_ready {
        // finalize: part count → gen_dispatch indirect args.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tess_classify.finalize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(finalize_pipeline.unwrap());
            pass.set_bind_group(0, classify.bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // gen verts: one workgroup per part (indirect).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tess_gen_verts"),
                timestamp_writes: None,
            });
            pass.set_pipeline(gen_pipeline.unwrap());
            pass.set_bind_group(0, classify.gen_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups_indirect(&classify.gen_dispatch, 0);
        }
        // shading: per-micro-triangle smooth normals + UVs + per-part metadata. Same
        // indirect grid as gen_verts (one workgroup per part); reads part_triangles +
        // the table topology, independent of the gen_verts positions.
        if attr_ready {
            if let Some(attr_pipeline) = pipeline_cache.get_compute_pipeline(classify.attr_pipeline) {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("tess_gen_attrs"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(attr_pipeline);
                pass.set_bind_group(0, classify.attr_bind_group.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups_indirect(&classify.gen_dispatch, 0);
            }
        }
        // step 2a: build per-part CLAS-instantiate descriptors. Bounded by the CPU-known
        // base-tri count (the shader still guards `p >= counts[0]`); 64-wide.
        if inst_ready {
            let inst_groups = classify.total_base_tris.div_ceil(64).min(65535);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tess_instantiate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(instantiate_pipeline.unwrap());
            pass.set_bind_group(0, classify.instantiate_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(inst_groups, 1, 1);
        }
    }
    render_queue.submit([encoder.finish()]);

    // step 2b: raw-VK indirect INSTANTIATE_TRIANGLE_CLUSTER — turn each part's GPU-built
    // descriptor + gen_vertices slice into a CLAS in the persistent pool. Must be its OWN
    // encoder (wgpu forbids mixing raw + wgpu-pass commands in one encoder); the opening
    // `as_barrier` synchronizes against the prior submit's compute writes (same queue,
    // submission order preserved). `src_infos_count = counts[0]` (GPU actual count, so
    // Phase-D culling just works); bounded by the size-query's `total_base_tris`.
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
        // SAFETY: raw-only encoder; opening barrier makes the prior submit's descriptor/
        // counts shader writes visible to the build; `tri_input` lives through the record
        // call (input maxima consumed at record time); all addresses reference resident
        // allocator buffers.
        unsafe {
            crate::gpu::extension::cmd_global_as_barrier(&mut build_encoder, &render_device, false);
            crate::gpu::extension::cmd_build_cluster_acceleration_structures_indirect(
                &mut build_encoder,
                fns_res,
                &cmd,
            );
            crate::gpu::extension::cmd_global_as_barrier(&mut build_encoder, &render_device, false);
        }
        render_queue.submit([build_encoder.finish()]);
    }

    // step 3a/3b: group the per-part CLAS by instance (scatter) → build one BLAS per
    // instance. Gated on the 2b CLAS build + the scatter pipeline + the C3 buffers.
    let scatter_pipeline = pipeline_cache.get_compute_pipeline(classify.scatter_pipeline);
    let c3_ready = clas_build_ready
        && scatter_pipeline.is_some()
        && classify.references.is_some()
        && classify.blas_addresses.is_some()
        && classify.tess_clas_addresses.is_some();
    if c3_ready {
        let alloc = allocator.as_ref().unwrap();
        let fns_res = fns.as_ref().unwrap();

        // Scatter: own wgpu encoder (the prior 2b build submit's trailing AS-barrier makes
        // the CLAS addresses visible to this compute read; same queue, submission order).
        let sbg = render_device.create_bind_group(
            "tess_scatter_bind_group",
            &pipeline_cache.get_bind_group_layout(&classify.scatter_layout),
            &BindGroupEntries::sequential((
                classify.tess_clas_addresses.as_ref().unwrap().as_entire_binding(),
                classify.references.as_ref().unwrap().as_entire_binding(),
                classify.counts.as_entire_binding(),
            )),
        );
        classify.scatter_bind_group = Some(sbg);
        let mut scatter_encoder =
            render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tess_classify.scatter"),
            });
        {
            let mut pass = scatter_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tess_scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(scatter_pipeline.unwrap());
            pass.set_bind_group(0, classify.scatter_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(classify.total_base_tris.div_ceil(64).min(65535), 1, 1);
        }
        render_queue.submit([scatter_encoder.finish()]);

        // Per-instance BLAS: one BLAS from each instance's contiguous CLAS-reference
        // sublist (raw-only encoder; record_build_per_instance_blas leads with an
        // AS-barrier that syncs against the scatter submit). The driver writes each BLAS
        // address into `blas_addresses[i]` GPU-side.
        let mut blas_encoder =
            render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tess_classify.blas_build"),
            });
        let mut built: Vec<InstanceBlas> = Vec::new();
        for (i, &cnt) in classify.per_instance_counts.iter().enumerate() {
            if cnt == 0 {
                continue;
            }
            let refs_addr = classify.references_addr + (classify.per_instance_offsets[i] as u64) * 8;
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
        render_queue.submit([blas_encoder.finish()]);
        // Old BLAS sets outlive their last consuming submit via the reaper
        // (frame-count keepalives get outrun; completion flags don't).
        classify.blas_keepalive.push(built);
        if classify.blas_keepalive.len() > 2 {
            let old = classify.blas_keepalive.remove(0);
            retire.retire(&render_queue, "tess_classify.blas_set", old);
        }
        // The BLAS addresses are now valid in `blas_addresses` — the PTLAS inject may run.
        classify.blas_ready = true;
    }

}
