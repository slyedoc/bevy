//! Skeletal deform pass — linear-blend skinning of animated cluster meshes.
//!
//! Each frame, for every active animated instance (capped at
//! [`MAX_ANIMATED_INSTANCES`]), the [`extract_animated_skins`] system gathers the
//! skinning palette (joint → transform-table node slot) and the mesh's
//! inverse-bind poses, and [`dispatch_deform`] runs `deform.wgsl` to skin the
//! rest-pose vertices into a per-instance region of the deform pool. The pool is
//! consumed downstream by the instantiate pass (per-instance CLAS) and the
//! resolve shader (deformed-vertex fetch) — see
//! `crates/bevy_solari/cluster_animation_plan.md`.
//!
//! Skinning sources joint world transforms straight from the GPU transform
//! table (joints are nodes): the deform shader walks each joint's `local`/`parent`
//! ancestor chain to compute its current world. (It does NOT read the propagated
//! `world[]` buffer — that only re-walks nodes whose own local changed, so a joint
//! whose parent animates but whose own local is static, e.g. a toe bone, would be
//! stale.) No CPU skin-matrix upload — only the static inverse-bind poses are
//! mirrored to the GPU.

use ash::vk;
use bevy_asset::Assets;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Local, Query, Res, ResMut},
};
use bevy_math::{Mat4, Vec4};
use bevy_mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};
use bevy_render::{
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
        BufferUsages, ComputePassDescriptor, PipelineCache, RawBufferVec, ShaderStages, ShaderType,
        UniformBuffer,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    Extract,
};
use bytemuck::{Pod, Zeroable};

use crate::bindings::RaytracingMesh3d;
use crate::ecs_gpu::{GpuColumn, GpuSlot};
use crate::geometry::ClusterMeshManager;
use crate::gpu::allocator::{Allocator, SparseBuffer, StableAddr};
use crate::instance::{Affine3x4, InstanceManager, RaytracingGpuEntity};
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;
use crate::transform::{LocalRSColumn, LocalTranslationColumn, ParentColumn, TransformGraph};

/// Max concurrent animated instances per frame. Deform / instantiate / BLAS
/// pools are sized for this; overflow falls back to the static (rest-pose) path.
pub const MAX_ANIMATED_INSTANCES: u32 = 8;
/// Max vertices per animated mesh — the per-slot stride of the deform pool.
pub const MAX_VERTS_PER_ANIMATED_MESH: u32 = 65536;

const WORKGROUP_SIZE: u32 = 64;

/// Per active animated instance. Mirrors `deform.wgsl::AnimatedSlot` (36 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct AnimatedSlotGpu {
    pub instance_slot: u32,
    pub mesh_vertex_base: u32,
    pub mesh_vertex_count: u32,
    pub deform_pool_base: u32,
    pub joint_count: u32,
    pub palette_base: u32,
    pub inverse_bind_base: u32,
    pub joint_base: u32,
    /// The instance's transform-table node — the shader walks it for the instance's
    /// absolute world (same space as the joint walk; the gathered per-instance
    /// transforms are origin-relative and would displace the skin).
    pub node_slot: u32,
}

/// Uniform shared with `deform.wgsl::DeformParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
struct DeformParams {
    num_slots: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Slot-indexed entry the resolve shader branches on (mirror of
/// `raytracing_scene_bindings.wgsl::AnimatedInstance`, 16 B). `flag == 1` →
/// the instance's verts come from the deform pool.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuAnimatedInstance {
    pub flag: u32,
    pub deform_pool_base: u32,
    pub mesh_vertex_base: u32,
    pub _pad: u32,
}

/// Render-world resource: the deform pool + skinning inputs + pipeline.
#[derive(Resource)]
pub struct Deform {
    /// Per-slot deformed positions (stride 3 f32) — `MAX_ANIMATED_INSTANCES`
    /// regions of `MAX_VERTS_PER_ANIMATED_MESH` vertices. Sparse → stable address.
    pub positions: SparseBuffer,
    /// Per-slot deformed octahedral normals (u32).
    pub normals: SparseBuffer,
    /// Per-slot deformed tangents (vec4: xyz + w bitangent sign).
    pub tangents: SparseBuffer,

    /// Stable addresses for the bindless resolve (`RtGeometryAddresses`) and the
    /// instantiate pass — set once at init; sparse buffers never move.
    pub normals_addr: StableAddr,
    pub tangents_addr: StableAddr,
    pub animated_table_addr: StableAddr,

    // Per-frame skinning inputs (rebuilt by the extract, uploaded in prepare).
    slots_cpu: Vec<AnimatedSlotGpu>,
    palette_cpu: Vec<u32>,
    inverse_bind_cpu: Vec<Affine3x4>,
    slots: RawBufferVec<AnimatedSlotGpu>,
    palette: RawBufferVec<u32>,
    inverse_bind: RawBufferVec<Affine3x4>,
    params: UniformBuffer<DeformParams>,

    /// Active animated instances this frame.
    active_count: u32,
    /// Max vertex count across active slots (dispatch x-bound).
    max_vertex_count: u32,

    /// Slot-indexed [`GpuAnimatedInstance`] table the resolve shader reads.
    /// Sparse: grows by page COMMIT (stable address — the resolve reads it via
    /// `physical_load` across frames), maintained incrementally (this frame's
    /// animated entries written, last frame's cleared).
    animated_table: SparseBuffer,
    animated_table_capacity: u32,
    /// Slots written animated last frame — cleared this frame if no longer animated.
    prev_animated: Vec<u32>,

    bind_group: Option<BindGroup>,
}

/// The deform bind-group layout. Owned by
/// [`SolariResourceManager`](crate::resource_manager::SolariResourceManager).
pub fn deform_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "deform",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 active_slots
                storage_buffer_read_only_sized(false, None), // 1 transform-table local_t (array<f64>)
                storage_buffer_read_only_sized(false, None), // 2 inverse_bind
                storage_buffer_read_only_sized(false, None), // 3 palette
                storage_buffer_read_only_sized(false, None), // 4 rest_positions
                storage_buffer_read_only_sized(false, None), // 5 rest_normals
                storage_buffer_read_only_sized(false, None), // 6 joint_indices
                storage_buffer_read_only_sized(false, None), // 7 joint_weights
                uniform_buffer::<DeformParams>(false),       // 8 params
                storage_buffer_sized(false, None),           // 9 deform_positions (rw)
                storage_buffer_sized(false, None),           // 10 deform_normals (rw)
                storage_buffer_read_only_sized(false, None), // 11 transform-table parent
                storage_buffer_read_only_sized(false, None), // 12 rest_tangents
                storage_buffer_sized(false, None),           // 13 deform_tangents (rw)
                storage_buffer_read_only_sized(false, None), // 14 transform-table local_rs
            ),
        ),
    )
}

impl Deform {
    /// Active animated instances this frame.
    #[inline]
    pub fn active_count(&self) -> u32 {
        self.active_count
    }

    /// The per-active-slot [`AnimatedSlotGpu`] buffer (`None` until first upload).
    /// The instantiate pass reads it.
    #[inline]
    pub fn slots_buffer(&self) -> Option<&Buffer> {
        self.slots.buffer()
    }

    /// This frame's active animated slots (CPU side). The resolve-binding build
    /// reads `instance_slot` / `deform_pool_base` / `mesh_vertex_base` to fill the
    /// slot-indexed animated table the hit shader branches on.
    #[inline]
    pub fn active_slots(&self) -> &[AnimatedSlotGpu] {
        &self.slots_cpu
    }

    /// Slot-indexed [`GpuAnimatedInstance`] table the resolve shader reads.
    #[inline]
    pub fn animated_table(&self) -> &Buffer {
        self.animated_table.buffer()
    }

    fn begin_frame(&mut self) {
        self.slots_cpu.clear();
        self.palette_cpu.clear();
        self.inverse_bind_cpu.clear();
        self.active_count = 0;
        self.max_vertex_count = 0;
    }
}

/// Pack a glaM `Mat4` (column-major) into the `mat3x4` row layout the shader and
/// the transform table use: `rows[k]` = the 4×4's row `k`.
fn mat4_to_affine(m: Mat4) -> Affine3x4 {
    Affine3x4 {
        rows: [
            Vec4::new(m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x),
            Vec4::new(m.x_axis.y, m.y_axis.y, m.z_axis.y, m.w_axis.y),
            Vec4::new(m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z),
        ],
    }
}

// Sparse (stable-address) — the resolve reads these pools bindlessly via
// `physical_load` and the instantiate pass reads positions, both cross-frame.
fn pool_buffer(
    allocator: &Allocator,
    device: &RenderDevice,
    label: &'static str,
    bytes: u64,
) -> SparseBuffer {
    let buf = allocator.create_sparse_buffer(
        device,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        bytes,
        label,
    );
    buf.commit(0..bytes);
    buf
}

/// `RenderStartup`: build the deform pool buffers. No-op without the raw-VK
/// [`Allocator`]. The bind-group layout lives in `SolariResourceManager`.
pub fn init_deform(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
) {
    let Some(allocator) = allocator else {
        return;
    };
    let n = (MAX_ANIMATED_INSTANCES * MAX_VERTS_PER_ANIMATED_MESH) as u64;
    let positions = pool_buffer(&allocator, &render_device, "deform.positions", n * 12);
    let normals = pool_buffer(&allocator, &render_device, "deform.normals", n * 4);
    let tangents = pool_buffer(&allocator, &render_device, "deform.tangents", n * 16);
    let animated_table = make_animated_table(&allocator, &render_device, 1);

    let mut slots = RawBufferVec::<AnimatedSlotGpu>::new(BufferUsages::STORAGE);
    slots.set_label(Some("deform.slots"));
    let mut palette = RawBufferVec::<u32>::new(BufferUsages::STORAGE);
    palette.set_label(Some("deform.palette"));
    let mut inverse_bind = RawBufferVec::<Affine3x4>::new(BufferUsages::STORAGE);
    inverse_bind.set_label(Some("deform.inverse_bind"));
    let mut params = UniformBuffer::<DeformParams>::default();
    params.set_label(Some("deform.params"));

    let normals_addr = normals.stable_addr();
    let tangents_addr = tangents.stable_addr();
    let animated_table_addr = animated_table.stable_addr();
    commands.insert_resource(Deform {
        positions,
        normals,
        tangents,
        normals_addr,
        tangents_addr,
        animated_table_addr,
        slots_cpu: Vec::new(),
        palette_cpu: Vec::new(),
        inverse_bind_cpu: Vec::new(),
        slots,
        palette,
        inverse_bind,
        params,
        active_count: 0,
        max_vertex_count: 0,
        animated_table,
        animated_table_capacity: 1,
        prev_animated: Vec::new(),
        bind_group: None,
    });
}

/// 256 MB virtual ≈ 16M slots; pages committed as the slot space grows.
const ANIMATED_TABLE_VIRTUAL_BYTES: u64 = 256 * 1024 * 1024;

fn make_animated_table(allocator: &Allocator, device: &RenderDevice, slots: u32) -> SparseBuffer {
    let buf = allocator.create_sparse_buffer(
        device,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ANIMATED_TABLE_VIRTUAL_BYTES,
        "deform.animated_table",
    );
    buf.commit(0..slots.max(1) as u64 * size_of::<GpuAnimatedInstance>() as u64);
    buf
}

/// `ExtractSchedule`: gather this frame's active animated instances — their
/// skinning palette (joint → transform-table node slot) + inverse-bind poses,
/// mesh pool bases, and deform-pool slot assignment.
pub fn extract_animated_skins(
    rt_skins: Extract<Query<(bevy_ecs::entity::Entity, RenderEntity, &RaytracingMesh3d, &SkinnedMesh)>>,
    joint_slots: Extract<Query<&GpuSlot<TransformGraph>>>,
    bindposes: Extract<Res<Assets<SkinnedMeshInverseBindposes>>>,
    gpu_entities: Query<&RaytracingGpuEntity>,
    cluster_meshes: Option<Res<ClusterMeshManager>>,
    deform: Option<ResMut<Deform>>,
    // One-shot overflow warnings (fire once per process, not per frame).
    mut warned_cap: Local<bool>,
    mut warned_verts: Local<bool>,
) {
    let (Some(cluster_meshes), Some(mut deform)) = (cluster_meshes, deform) else {
        return;
    };
    deform.begin_frame();

    // Instances dropped because the per-frame cap is full — they fall back to the
    // static (rest-pose) path. Counted to warn once if the scene needs a bigger cap.
    let mut dropped_over_cap = 0u32;
    for (entity, render_entity, mesh3d, skin) in &rt_skins {
        // The instance's own transform-table node: the deform shader walks it for the
        // instance's absolute world. Not yet slotted (first frame) -> rest pose.
        let Ok(node_slot) = joint_slots.get(entity).map(GpuSlot::index) else {
            continue;
        };
        // Instance must be bound (has a GPU slot) and its mesh resident + animated.
        let Ok(gpu) = gpu_entities.get(render_entity) else {
            continue;
        };
        let Some(ptrs) = cluster_meshes.animated_mesh_pointers(mesh3d.0.id()) else {
            continue;
        };
        if deform.active_count >= MAX_ANIMATED_INSTANCES {
            dropped_over_cap += 1;
            continue;
        }
        // A mesh past the per-slot vertex cap would deform past its pool region
        // (into the next slot) — skip it (renders rest pose) rather than corrupt.
        if ptrs.vertex_count > MAX_VERTS_PER_ANIMATED_MESH {
            if !*warned_verts {
                *warned_verts = true;
                tracing::warn!(
                    target: "bevy_solari",
                    "animated mesh has {} vertices > MAX_VERTS_PER_ANIMATED_MESH ({}); \
                     rendering it in rest pose. Raise the cap or split the mesh.",
                    ptrs.vertex_count,
                    MAX_VERTS_PER_ANIMATED_MESH,
                );
            }
            continue;
        }
        let Some(ibp) = bindposes.get(&skin.inverse_bindposes) else {
            continue;
        };
        let joint_count = skin.joints.len().min(ibp.len()) as u32;
        if joint_count == 0 {
            continue;
        }

        let palette_base = deform.palette_cpu.len() as u32;
        let inverse_bind_base = deform.inverse_bind_cpu.len() as u32;
        for k in 0..joint_count as usize {
            let node = joint_slots
                .get(skin.joints[k])
                .map(GpuSlot::index)
                .unwrap_or(u32::MAX);
            deform.palette_cpu.push(node);
            let m = mat4_to_affine(ibp[k]);
            deform.inverse_bind_cpu.push(m);
        }

        let active = deform.active_count;
        deform.slots_cpu.push(AnimatedSlotGpu {
            instance_slot: gpu.0 .0,
            mesh_vertex_base: ptrs.vertex_base,
            mesh_vertex_count: ptrs.vertex_count,
            deform_pool_base: active * MAX_VERTS_PER_ANIMATED_MESH,
            joint_count,
            palette_base,
            inverse_bind_base,
            joint_base: ptrs.joint_base,
            node_slot,
        });
        deform.max_vertex_count = deform.max_vertex_count.max(ptrs.vertex_count);
        deform.active_count += 1;
    }

    if dropped_over_cap > 0 && !*warned_cap {
        *warned_cap = true;
        tracing::warn!(
            target: "bevy_solari",
            "{} animated instance(s) over MAX_ANIMATED_INSTANCES ({}) this frame — \
             the excess render in rest pose. Raise the cap to animate them all.",
            dropped_over_cap,
            MAX_ANIMATED_INSTANCES,
        );
    }
}

/// `Render::Prepare`: upload this frame's skinning inputs + params, and maintain
/// the slot-indexed animated table the resolve shader reads.
pub fn prepare_deform(
    deform: Option<ResMut<Deform>>,
    instances: Option<Res<InstanceManager>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(mut deform) = deform else {
        return;
    };
    let num = deform.active_count;
    *deform.params.get_mut() = DeformParams {
        num_slots: num,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    deform.params.write_buffer(&render_device, &render_queue);

    // Maintain the slot-indexed animated table (grow on high-water, then
    // diff-update: clear last frame's animated slots, write this frame's). Runs
    // even when `num == 0` so a stopped/despawned animated instance reverts to
    // the static path. A non-animated scene only ever touches it on growth.
    let stride = size_of::<GpuAnimatedInstance>() as u64;
    let high_water = instances.map(|i| i.slot_high_water()).unwrap_or(0).max(1);
    if high_water > deform.animated_table_capacity {
        let old = deform.animated_table_capacity as u64 * stride;
        deform.animated_table_capacity = high_water.next_power_of_two();
        let new = deform.animated_table_capacity as u64 * stride;
        // Grow by page COMMIT — the buffer (and its trace-captured address)
        // never moves. Fresh sparse pages are UNDEFINED: zero the new region.
        deform.animated_table.commit(0..new);
        render_queue.write_buffer(
            deform.animated_table.buffer(),
            old,
            &vec![0u8; (new - old) as usize],
        );
    }

    let prev = core::mem::take(&mut deform.prev_animated);
    let zero = GpuAnimatedInstance::default();
    for slot in &prev {
        render_queue.write_buffer(
            deform.animated_table.buffer(),
            *slot as u64 * stride,
            bytemuck::bytes_of(&zero),
        );
    }
    let entries: Vec<(u32, GpuAnimatedInstance)> = deform
        .slots_cpu
        .iter()
        .map(|s| {
            (
                s.instance_slot,
                GpuAnimatedInstance {
                    flag: 1,
                    deform_pool_base: s.deform_pool_base,
                    mesh_vertex_base: s.mesh_vertex_base,
                    _pad: 0,
                },
            )
        })
        .collect();
    for (slot, entry) in &entries {
        render_queue.write_buffer(
            deform.animated_table.buffer(),
            *slot as u64 * stride,
            bytemuck::bytes_of(entry),
        );
        deform.prev_animated.push(*slot);
    }

    if num == 0 {
        return;
    }

    // Re-pack the CPU vecs into the GPU RawBufferVecs (clear + push). Taken out
    // of `deform` first to avoid overlapping borrows.
    let Deform {
        slots,
        palette,
        inverse_bind,
        slots_cpu,
        palette_cpu,
        inverse_bind_cpu,
        ..
    } = &mut *deform;
    slots.clear();
    for v in slots_cpu.iter() {
        slots.push(*v);
    }
    palette.clear();
    for v in palette_cpu.iter() {
        palette.push(*v);
    }
    inverse_bind.clear();
    for v in inverse_bind_cpu.iter() {
        inverse_bind.push(*v);
    }
    slots.write_buffer(&render_device, &render_queue);
    palette.write_buffer(&render_device, &render_queue);
    inverse_bind.write_buffer(&render_device, &render_queue);
}

/// `Render::PrepareBindGroups`: (re)build the deform bind group. Skipped when no
/// animated instances are active (the joint pools may be empty / zero-sized).
pub fn prepare_deform_bind_group(
    deform: Option<ResMut<Deform>>,
    resource_manager: Option<Res<SolariResourceManager>>,
    cluster_meshes: Option<Res<ClusterMeshManager>>,
    local_t: Option<Res<GpuColumn<LocalTranslationColumn>>>,
    local_rs: Option<Res<GpuColumn<LocalRSColumn>>>,
    parent: Option<Res<GpuColumn<ParentColumn>>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let Some(mut deform) = deform else {
        return;
    };
    if deform.active_count == 0 {
        deform.bind_group = None;
        return;
    }
    let (
        Some(resource_manager),
        Some(cluster_meshes),
        Some(local_t),
        Some(local_rs),
        Some(parent),
    ) = (resource_manager, cluster_meshes, local_t, local_rs, parent)
    else {
        deform.bind_group = None;
        return;
    };
    let (Some(slots), Some(palette), Some(inverse_bind), Some(params)) = (
        deform.slots.buffer(),
        deform.palette.buffer(),
        deform.inverse_bind.buffer(),
        deform.params.binding(),
    ) else {
        deform.bind_group = None;
        return;
    };

    let layout = pipeline_cache.get_bind_group_layout(&resource_manager.deform);
    let bind_group = render_device.create_bind_group(
        "deform",
        &layout,
        &BindGroupEntries::sequential((
            slots.as_entire_binding(),
            local_t.buffer().as_entire_binding(),
            inverse_bind.as_entire_binding(),
            palette.as_entire_binding(),
            cluster_meshes.vertex_positions.buffer().as_entire_binding(),
            cluster_meshes.vertex_normals.buffer().as_entire_binding(),
            cluster_meshes
                .vertex_joint_indices
                .buffer()
                .as_entire_binding(),
            cluster_meshes
                .vertex_joint_weights
                .buffer()
                .as_entire_binding(),
            params,
            deform.positions.buffer().as_entire_binding(),
            deform.normals.buffer().as_entire_binding(),
            parent.buffer().as_entire_binding(),
            cluster_meshes.vertex_tangents.buffer().as_entire_binding(),
            deform.tangents.buffer().as_entire_binding(),
            local_rs.buffer().as_entire_binding(),
        )),
    );
    deform.bind_group = Some(bind_group);
}

/// `RenderGraph` (`Deform` stage): skin the active animated instances. Runs
/// after `Propagate` (world[] + instance transforms ready), before `Classify`.
pub fn dispatch_deform(
    deform: Option<Res<Deform>>,
    pipelines: Res<SolariPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let Some(deform) = deform else {
        return;
    };
    if deform.active_count == 0 || deform.max_vertex_count == 0 {
        return;
    }
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.deform) else {
        return;
    };
    let Some(bind_group) = deform.bind_group.as_ref() else {
        return;
    };

    let x = deform.max_vertex_count.div_ceil(WORKGROUP_SIZE);
    let y = deform.active_count;
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("deform"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    let d = diagnostics.time_span(&mut pass, "deform");
    pass.dispatch_workgroups(x, y, 1);
    d.end(&mut pass);
}
