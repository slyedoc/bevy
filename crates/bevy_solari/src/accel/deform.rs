//! Skeletal deform pass — linear-blend skinning of animated cluster meshes.
//!
//! Each frame, for every active animated instance (capped at
//! [`MAX_ANIMATED_INSTANCES`]), the [`extract_animated_skins`] system gathers the
//! skinning palette (joint → transform-table node slot) and the mesh's
//! inverse-bind poses, and [`dispatch_deform`] runs `deform.slang` to skin the
//! rest-pose vertices into a per-instance region of the deform pool. The pool is
//! consumed downstream by the instantiate pass (per-instance CLAS) and the
//! resolve shader (deformed-vertex fetch).
//!
//! Skinning sources joint world transforms straight from the GPU transform
//! table (joints are nodes): the deform shader walks each joint's `local`/`parent`
//! ancestor chain to compute its current world. (It does NOT read the propagated
//! `world[]` buffer — that only re-walks nodes whose own local changed, so a joint
//! whose parent animates but whose own local is static, e.g. a toe bone, would be
//! stale.) No CPU skin-matrix upload — only the static inverse-bind poses are
//! mirrored to the GPU.

#![allow(unsafe_code)]

use ash::vk;
use bevy_asset::Assets;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Local, Query, Res, ResMut},
};
use bevy_math::{Mat4, Vec4};
use bevy_mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};
use bevy_render::{
    render_resource::{Buffer, BufferUsages, RawBufferVec},
    renderer::{RenderContext, RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    Extract,
};
use bytemuck::{Pod, Zeroable};
use wgpu::hal::api::Vulkan as VkApi;

use crate::bindings::RaytracingMesh3d;
use crate::ecs_gpu::{GpuColumn, GpuSlot};
use crate::geometry::ClusterMeshManager;
use crate::gpu::allocator::{Allocator, SparseBuffer, StableAddr};
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
use crate::instance::{Affine3x4, InstanceManager, RaytracingGpuEntity};
use crate::transform::{LocalRSColumn, LocalTranslationColumn, ParentColumn, TransformGraph};

/// Max concurrent animated instances per frame. Deform / instantiate / BLAS
/// pools are sized for this; overflow falls back to the static (rest-pose) path.
/// A make_human character is 8 instances (one per part), so this is 4 characters.
pub const MAX_ANIMATED_INSTANCES: u32 = 32;
/// Max vertices per animated mesh — the per-slot stride of the deform pool.
/// Sized for a make_human skin (measured 114871 verts / 1296 clusters).
pub const MAX_VERTS_PER_ANIMATED_MESH: u32 = 131072;

const WORKGROUP_SIZE: u32 = 64;

/// Per active animated instance. Mirrors `deform.slang::AnimatedSlot` (36 B).
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

/// Push params shared with `deform.slang::DeformParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
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
    params: DeformParams,

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

    kernel: HeapKernel,
    kernel_slots: KernelSlots,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for Deform {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for Deform {}
unsafe impl Sync for Deform {}

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

/// `RenderStartup`: compile the deform kernel — a layout-free heap pipeline
/// ([`HeapKernel`]), Slang from source — and build the deform pool buffers.
/// No-op without the raw-VK [`Allocator`].
pub fn init_deform(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let Some(kernel) = HeapKernel::new(
        &seam,
        "deform.slang",
        include_str!("deform.slang"),
        "deform",
        crate::bindings::OCTAHEDRAL_MODULES,
        &[],
        "deform",
        size_of::<DeformParams>() as u32,
    ) else {
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
        params: DeformParams::default(),
        active_count: 0,
        max_vertex_count: 0,
        animated_table,
        animated_table_capacity: 1,
        prev_animated: Vec::new(),
        kernel,
        kernel_slots: KernelSlots::new(&seam, 14),
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
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
    deform.params = DeformParams {
        num_slots: num,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

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

/// `RenderGraph` (`Deform` stage): skin the active animated instances. Runs
/// after `Propagate` (world[] + instance transforms ready), before `Classify`.
/// A raw heap dispatch: buffer slots rewritten per dispatch, params + slot
/// array in push data.
pub fn dispatch_deform(
    deform: Option<Res<Deform>>,
    seam: Option<Res<BindingSeam>>,
    cluster_meshes: Option<Res<ClusterMeshManager>>,
    local_t: Option<Res<GpuColumn<LocalTranslationColumn>>>,
    local_rs: Option<Res<GpuColumn<LocalRSColumn>>>,
    parent: Option<Res<GpuColumn<ParentColumn>>>,
    mut ctx: RenderContext,
) {
    let (Some(deform), Some(seam), Some(cluster_meshes), Some(local_t), Some(local_rs), Some(parent)) =
        (deform, seam, cluster_meshes, local_t, local_rs, parent)
    else {
        return;
    };
    if deform.active_count == 0 || deform.max_vertex_count == 0 {
        return;
    }
    let (Some(slots), Some(palette), Some(inverse_bind)) = (
        deform.slots.buffer(),
        deform.palette.buffer(),
        deform.inverse_bind.buffer(),
    ) else {
        return;
    };

    let ks = &deform.kernel_slots;
    let blob = deform.kernel.push_blob(
        "deform",
        bytemuck::bytes_of(&deform.params),
        &[
            ("active_slots", ks.buffer(&seam, 0, slots)),
            ("local_t", ks.buffer(&seam, 1, local_t.buffer())),
            ("inverse_bind", ks.buffer(&seam, 2, inverse_bind)),
            ("palette", ks.buffer(&seam, 3, palette)),
            (
                "rest_positions",
                ks.buffer(&seam, 4, cluster_meshes.vertex_positions.buffer()),
            ),
            (
                "rest_normals",
                ks.buffer(&seam, 5, cluster_meshes.vertex_normals.buffer()),
            ),
            (
                "joint_indices",
                ks.buffer(&seam, 6, cluster_meshes.vertex_joint_indices.buffer()),
            ),
            (
                "joint_weights",
                ks.buffer(&seam, 7, cluster_meshes.vertex_joint_weights.buffer()),
            ),
            ("deform_positions", ks.buffer(&seam, 8, deform.positions.buffer())),
            ("deform_normals", ks.buffer(&seam, 9, deform.normals.buffer())),
            ("parent", ks.buffer(&seam, 10, parent.buffer())),
            (
                "rest_tangents",
                ks.buffer(&seam, 11, cluster_meshes.vertex_tangents.buffer()),
            ),
            ("deform_tangents", ks.buffer(&seam, 12, deform.tangents.buffer())),
            ("local_rs", ks.buffer(&seam, 13, local_rs.buffer())),
        ],
    );
    let x = deform.max_vertex_count.div_ceil(WORKGROUP_SIZE);
    let y = deform.active_count;
    let encoder = ctx.command_encoder();
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // barriers bracket this dispatch against the surrounding passes (raw
    // dispatches are invisible to wgpu's tracking).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &deform.raw_device;
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)];
            let dep = vk::DependencyInfo::default().memory_barriers(&barrier);
            // The column scatters' local/parent writes -> our reads.
            dev.cmd_pipeline_barrier2(cb, &dep);
            seam.bind_heaps(cb);
            seam.push_data(cb, &blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, deform.kernel.pipeline);
            dev.cmd_dispatch(cb, x, y, 1);
            // Our deform-pool writes -> the instantiate/BLAS-build reads.
            dev.cmd_pipeline_barrier2(cb, &dep);
        });
    }
}
