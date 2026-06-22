// Per-frame partitioned TLAS (NV partitioned_AS) build over the
// per-instance BLAS device addresses produced by
// `blas_rebuild::dispatch_blas_rebuild`. Fill compute writes
// `WriteInstanceData[]` records GPU-side; raw VK build dispatches
// `vkCmdBuildPartitionedAccelerationStructuresNV` in the same encoder.
#![allow(unsafe_code, reason = "raw VK build via as_hal_mut")]

//! Partitioned TLAS (PTLAS) build for the cluster-AS pipeline.
//!
//! Why PTLAS (and not standard KHR TLAS)
//! -------------------------------------
//!
//! A standard `vkAccelerationStructureKHR` TLAS *can* traverse into a
//! cluster-built BLAS (the BuildClustersBottomLevel result is a normal
//! BLAS as far as the TLAS is concerned — NVIDIA's own `vk_lod_clusters`
//! sample ray-traces cluster BLASes through a plain
//! `vkCmdBuildAccelerationStructuresKHR` TLAS). We use the NV
//! `partitioned_AS` instead for **incremental rebuilds at scale**:
//! `vkCmdBuildPartitionedAccelerationStructuresNV` lets us feed last
//! frame's PTLAS as the build source (`src` = last frame's buffer) and
//! apply only a *delta* of per-instance operations, rather than
//! re-specifying every instance every frame. That is the load-bearing
//! win for large streamed scenes (10⁵–10⁶ instances) where most
//! instances are static frame-to-frame.
//!
//! Op stream (per frame, see [`IndirectCommand`])
//! ----------------------------------------------
//! Each instance carries a **stable** PTLAS `instance_index == GpuEntity`,
//! so deltas target the same logical instance across frames. The only
//! op type emitted is:
//! - `WRITE_INSTANCE` (104 B full record) — newly added slots, slots
//!   whose transform changed, and disabled slots (written with a null
//!   AS address to remove them). Static instances emit **no op**:
//!   `blas_rebuild` builds in EXPLICIT_DESTINATIONS, so a static
//!   instance's BLAS device address is stable frame-to-frame and the
//!   driver carries the instance across from `src` untouched.
//!
//! There is no `UPDATE_INSTANCE` batch. `UPDATE_INSTANCE` only refreshes
//! a BLAS address (it cannot change a transform), and stable per-slot
//! addresses leave nothing to refresh — so it was removed.
//!
//! Build mode: incremental frames update the PTLAS **in place** —
//! `src == dst == storage` (a single buffer). The NV spec permits equal
//! `src`/`dst` ("if they are the same, the update happens in-place"), so the
//! driver applies only the op-delta to the existing structure, with no copy
//! into a second buffer. A full rebuild (`src = 0`, WRITE every active slot,
//! built from scratch into the same buffer) happens on the first build and
//! whenever capacity grows.
//!
//! Partitioning
//! ------------
//! Static instances (tagged `TransformStatic`, surfaced GPU-side by the
//! `Presence<StaticColumn>` flag) go to a single regular partition; movers
//! go to the **global** partition, which NV builds per-instance ("treated
//! as if in individual partitions") — so a moved instance rebuilds only its
//! own global entry, never the static partition. The static instances are
//! written once and then carried from `src` untouched, so they never
//! rebuild. This is the spec's recommended layout (frequent updates →
//! global; stable bulk → a regular partition). A spatial grid of regular
//! partitions was tried and reverted: hashing statics across many partitions
//! gave each partition a scene-spanning AABB, and the overlap inflated
//! ray-traversal cost far more than the cheaper per-cell rebuilds saved. One
//! regular partition keeps the driver's BVH coherent. The static/mover split
//! lives in `ptlas_fill.wgsl::resolve_partition`.

use ash::vk::{self, TaggedStructure};
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        AccelerationStructureFlags, AccelerationStructureUpdateMode, BindGroup, BindGroupEntries,
        Buffer, ComputePassDescriptor, CreateTlasDescriptor, PipelineCache, RawBufferVec,
        ShaderType, Tlas, UniformBuffer,
    },
    renderer::{raw_vulkan_init::AdditionalVulkanFeatures, RenderContext, RenderDevice, RenderQueue},
};
use wgpu::hal::api::Vulkan as VkApi;
use bytemuck::{Pod, Zeroable};
use wgpu::CommandEncoderDescriptor;

use crate::bindings::ClusterSceneBindGroup;
use crate::ecs_gpu::{GpuColumn, Presence};
use crate::instance::{
    GeometryIdColumn, InstanceManager, InstanceMaskColumn, MaterialColumn, NodeSlotColumn,
    TransformColumn,
};
use crate::material::MaterialTraversalFlags;
use crate::transform::StaticColumn;

use crate::gpu::allocator::{Allocator, SparseBuffer};
use super::blas_sharing::BlasSharing;
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;
use crate::gpu::extension::{ClusterExtensionFns, RayTracingPipelineFeature};

/// Virtual address space for the PTLAS storage buffer — 4 GB.
/// Sparse-backed.
pub const PTLAS_STORAGE_VIRTUAL_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Virtual scratch — 1 GB. Sparse-backed.
pub const PTLAS_SCRATCH_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;

/// Virtual `WriteInstanceData[]` — 1 GB ≈ 10 M instances × 104 B.
/// Sparse-backed.
pub const PTLAS_WRITE_DATA_VIRTUAL_BYTES: u64 = 1024 * 1024 * 1024;

/// NV partitioned-AS scratch alignment. NV's doc says cluster-AS
/// scratch alignment (`clusterScratchByteAlignment`) applies here too;
/// 256 is the conservative ceiling.
pub const PTLAS_SCRATCH_ALIGN: u64 = 256;

/// `VkPartitionedAccelerationStructureWriteInstanceDataNV` byte size —
/// must match `ptlas_fill.wgsl`'s `WriteInstanceData` struct (the full
/// 104 B record fed to `WRITE_INSTANCE`).
const WRITE_INSTANCE_DATA_SIZE: u64 = 104;

/// Maximum partitioned-AS operations the `src_infos` buffer is sized
/// for. Currently a single `WRITE_INSTANCE` op (the delta); the second
/// slot is reserved headroom for a future op.
const MAX_OPS: u64 = 2;

/// `write_slots_cpu` null flag: this slot is disabled this frame → the
/// fill writes a null AS address, removing it from the PTLAS.
const PAIR_NULL: u32 = 1;
const PAIR_NORMAL: u32 = 0;

/// Mirror of `VkBuildPartitionedAccelerationStructureIndirectCommandNV`.
/// One per partitioned-AS op; up to [`MAX_OPS`] live in `src_infos`.
#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
struct IndirectCommand {
    /// `vk::PartitionedAccelerationStructureOpTypeNV` value
    /// (`WRITE_INSTANCE = 0`).
    op_type: u32,
    arg_count: u32,
    /// `VkStridedDeviceAddressNV` { start_address: u64, stride: u64 }.
    arg_data_start_address: u64,
    arg_data_stride: u64,
}

const _: () = assert!(size_of::<IndirectCommand>() == 24);

/// CPU-seeded delta record: `slot` is the stable PTLAS `instance_index`;
/// `null_flag` is [`PAIR_NULL`] for a disabled slot (write a null AS) or
/// [`PAIR_NORMAL`] to read the slot's current BLAS address.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct PtlasWritePair {
    slot: u32,
    null_flag: u32,
}

/// Regular (static) partition count. Statics share one regular partition;
/// movers live in the global partition (built per-instance), so a moved mover
/// never dirties the static partition. A spatial grid of many regular
/// partitions was tried and reverted — see the `Partitioning` module note.
pub const PTLAS_PARTITION_COUNT: u32 = 1;

/// Uniform layout shared with `ptlas_fill.wgsl::PtlasFillParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
pub struct PtlasFillParamsGpu {
    pub active_count: u32,
    pub cpu_count: u32,
    pub force_all: u32,
}

/// Render-world resource for the incremental partitioned-TLAS pass.
///
/// Bundles everything the pass owns — the single AS storage buffer,
/// the per-frame fill I/O buffers, the three fill compute pipelines +
/// bind-group layout, and the per-frame fill bind group.
#[derive(Resource)]
pub struct Ptlas {
    /// PTLAS storage — a single sparse buffer. NV's partitioned build permits
    /// `src == dst` (in-place update: "if they are the same, the update happens
    /// in-place"), so an incremental update reads *and* writes this one buffer
    /// (`src = dst = storage`), with no copy into a second buffer. A full rebuild
    /// (`src = 0`) writes it from scratch in place.
    pub storage: SparseBuffer,
    /// Sparse build scratch.
    pub scratch: SparseBuffer,
    /// Sparse `WriteInstanceData[]` (104 B records) filled by the fill
    /// passes. CPU-seeded delta in `[0, cpu_count)`, GPU band-crossers
    /// in `[cpu_count, count)`. Each record's `instance_index` carries
    /// the stable slot.
    pub write_data: SparseBuffer,
    /// Single GPU record counter, seeded to `cpu_count` each frame and
    /// atomic-incremented by `fill_incremental`. `finalize` copies it
    /// into the build op's `arg_count`.
    pub write_count: Buffer,
    /// CPU-seeded `(slot, null_flag)` delta records (added ∪ moved ∪
    /// disabled). Always holds ≥1 element so the bind group has a valid
    /// buffer (`cpu_count` gates the seed dispatch).
    pub write_slots_cpu: RawBufferVec<PtlasWritePair>,
    /// Op buffer holding up to [`MAX_OPS`]
    /// `VkBuildPartitionedAccelerationStructureIndirectCommandNV`
    /// records. CPU writes everything but `arg_count`, which `finalize`
    /// writes GPU-side (the count is GPU-driven).
    pub src_infos: Buffer,
    /// 4-byte op count (1 = the single WRITE op). CPU-written.
    pub src_infos_count: Buffer,
    /// Per-frame fill-compute params uniform.
    pub fill_params: UniformBuffer<PtlasFillParamsGpu>,
    /// `wgpu::Tlas` wrapper over [`Self::storage`] — so ray-trace shaders bind the
    /// PTLAS through wgpu's standard `accelerationStructureEXT` slot.
    /// `current_tlas()` returns it. Wraps a `vkCreateAccelerationStructureKHR`
    /// handle over the storage buffer via [`wgpu::Device::create_tlas_from_hal`],
    /// recreated only when the build size changes (the old wrapper's Drop calls
    /// `vkDestroyAccelerationStructureKHR`).
    pub tlas: Option<Tlas>,
    /// Build-size of [`Self::tlas`]; mismatch with this frame's
    /// `sizes_info.acceleration_structure_size` recreates the handle.
    pub as_handle_size: u64,
    /// `vkGetAccelerationStructureDeviceAddressKHR` for the handle.
    pub as_handle_device_address: vk::DeviceAddress,
    /// Whether at least one PTLAS build has completed. Until then a
    /// full rebuild (`src = 0`) is forced.
    pub has_built: bool,
    /// Instance capacity the current PTLAS is sized for (== the
    /// `slot_high_water` at last (re)build). When this frame's
    /// high-water exceeds it, the AS must be resized → full rebuild.
    pub as_capacity: u32,

    // ── Per-frame op decisions: computed in `prepare_ptlas_params`,
    //    consumed in `dispatch_ptlas`. ─────────────────────────────
    /// CPU-seeded delta record count (= `fill_seed` thread count).
    pub cpu_count: u32,
    /// Live `src_infos` op count (0/1).
    pub op_count: u32,
    /// `true` → build from scratch (`src = 0`, WRITE every active slot
    /// via `force_all`); `false` → incremental in-place (`src = dst = storage`,
    /// WRITE only the CPU delta + GPU band-crossers, static instances
    /// reused in place).
    pub full_rebuild: bool,

    /// Per-frame fill bind group, rebuilt in `Render::PrepareBindGroups`. The fill
    /// compute pipeline ids live on [`SolariPipelines`], the layout on
    /// [`SolariResourceManager`].
    pub bind_group: Option<BindGroup>,

    /// Per-slot `instance_flags` value last WRITTEN into a PTLAS record
    /// (persistent). The fill derives each instance's flags from its
    /// material's traversal flags and re-specifies the instance when they
    /// differ — GPU change detection, the flags twin of the
    /// current-vs-previous transform compare. Recreated zeroed on slot
    /// growth (growth forces a full rebuild, which restamps every slot).
    pub instance_written_flags: Buffer,
    /// Slot capacity of [`Self::instance_written_flags`].
    pub written_flags_capacity: u32,

    /// Per-slot partition value last WRITTEN into a PTLAS record (persistent).
    /// The fill re-specifies an instance when its resolved partition drifts from
    /// this — the partition twin of [`Self::instance_written_flags`], closing the
    /// window where an instance is placed before its `TransformStatic` flag has
    /// scattered (it then migrates global → spatial cell once the flag lands).
    /// Recreated zeroed on slot growth (growth forces a full rebuild, restamping).
    pub instance_written_partition: Buffer,
    /// Slot capacity of [`Self::instance_written_partition`].
    pub written_partition_capacity: u32,
}

impl Ptlas {
    /// This frame's PTLAS — the build target the path-tracer binds.
    /// `None` until the first build has produced its handle.
    #[inline]
    pub fn current_tlas(&self) -> Option<&Tlas> {
        self.tlas.as_ref()
    }
}

/// `RenderStartup`: allocate the PTLAS storage + fill I/O buffers + insert
/// the [`Ptlas`] resource. No-op when the raw-VK [`Allocator`] is absent —
/// downstream PTLAS systems guard on the resource's presence.
pub fn init_ptlas(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
) {
    let Some(allocator) = allocator else {
        return;
    };

    // Single storage buffer — incremental builds update it in place (src == dst).
    let storage = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        wgpu::BufferUsages::COPY_DST,
        PTLAS_STORAGE_VIRTUAL_BYTES,
        "ptlas.storage",
    );
    let scratch = allocator.create_sparse_buffer(
        &render_device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        wgpu::BufferUsages::STORAGE,
        PTLAS_SCRATCH_VIRTUAL_BYTES,
        "ptlas.scratch",
    );
    let write_data = allocator.create_sparse_buffer(
        &render_device,
        // Build reads the records by device address with
        // AS_BUILD_INPUT semantics; STORAGE for the fill compute.
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        PTLAS_WRITE_DATA_VIRTUAL_BYTES,
        "ptlas.write_data",
    );
    let write_count = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ptlas.write_count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut write_slots_cpu = RawBufferVec::<PtlasWritePair>::new(wgpu::BufferUsages::STORAGE);
    write_slots_cpu.set_label(Some("ptlas.write_slots_cpu"));

    let src_infos = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ptlas.src_infos"),
        size: MAX_OPS * size_of::<IndirectCommand>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let src_infos_count = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ptlas.src_infos_count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut fill_params: UniformBuffer<PtlasFillParamsGpu> = UniformBuffer::default();
    fill_params.set_label(Some("ptlas.fill_params"));

    let instance_written_flags = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ptlas.instance_written_flags"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let instance_written_partition = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ptlas.instance_written_partition"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    commands.insert_resource(Ptlas {
        storage,
        scratch,
        write_data,
        write_count,
        write_slots_cpu,
        src_infos,
        src_infos_count,
        fill_params,
        tlas: None,
        as_handle_size: 0,
        as_handle_device_address: 0,
        has_built: false,
        as_capacity: 0,
        cpu_count: 0,
        op_count: 0,
        full_rebuild: false,
        bind_group: None,
        instance_written_flags,
        written_flags_capacity: 1,
        instance_written_partition,
        written_partition_capacity: 1,
    });
}

/// `Render::Prepare`: seed the partitioned-AS WRITE op — push the CPU
/// delta (added ∪ moved ∪ disabled) into `write_slots_cpu`, seed the GPU
/// record counter, write the `src_infos` op (with a GPU-filled
/// `arg_count`), set fill params, size the dst PTLAS + AS handle, and
/// commit sparse pages. The fill passes + VK build happen in
/// [`dispatch_ptlas`]; GPU band-cross detection appends the rest.
pub fn prepare_ptlas_params(
    mut resources: Option<ResMut<Ptlas>>,
    instances: Option<Res<InstanceManager>>,
    allocator: Option<Res<Allocator>>,
    fns: Option<Res<ClusterExtensionFns>>,
    hair_instances: Option<Res<crate::hair::HairInstances>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(resources), Some(instances), Some(_allocator), Some(fns)) =
        (resources.as_deref_mut(), instances, allocator, fns)
    else {
        return;
    };
    let Some(partitioned_fns) = fns.partitioned.as_ref() else {
        resources.op_count = 0;
        return;
    };

    // Hair instances occupy PTLAS indices `[cluster_high_water, +hair_count)`
    // above the cluster slots, written by `ptlas_hair_write` each frame. They
    // expand the PTLAS instance space but use none of the cluster fill state.
    let hair_count = hair_instances.as_ref().map(|h| h.count).unwrap_or(0);

    let cluster_high_water = instances.slot_high_water();
    // Total PTLAS instance space (cluster slots + hair).
    let high_water = cluster_high_water + hair_count;
    if high_water == 0 {
        resources.op_count = 0;
        return;
    }
    let active_count = instances.active_count() as u32;

    // Build kind. The PTLAS instance space only grows; growth needs a bigger
    // AS, which can't be an in-place update — so a full rebuild. Also full on
    // the very first build (no `src` to carry from). A change in hair count
    // also grows/shrinks the space and is folded into `high_water`.
    let grew = high_water > resources.as_capacity;
    // A mass despawn (regenerate) NULLs a huge number of slots in one incremental
    // op — many of them disabled AND re-added the same frame — which the
    // incremental PTLAS can't absorb (device lost). Any frame with a large despawn
    // takes the proven full-rebuild path instead; normal mover frames (a handful of
    // despawns, or none) stay incremental.
    const MASS_DESPAWN_FULL_REBUILD: usize = 4096;
    let mass_despawn = instances.disabled_slots().len() > MASS_DESPAWN_FULL_REBUILD;
    let full_rebuild = !resources.has_built || grew || mass_despawn;

    // Grow the per-slot written-flags mirror with the slot space. Fresh
    // buffer = all zeros, consistent because growth forces a full rebuild
    // (`force_all` restamps every active slot this frame).
    if high_water > resources.written_flags_capacity {
        resources.instance_written_flags =
            render_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ptlas.instance_written_flags"),
                size: high_water as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
        resources.written_flags_capacity = high_water;
        debug_assert!(full_rebuild);
    }
    if high_water > resources.written_partition_capacity {
        resources.instance_written_partition =
            render_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ptlas.instance_written_partition"),
                size: high_water as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
        resources.written_partition_capacity = high_water;
        debug_assert!(full_rebuild);
    }

    // We can't skip the build on a no-CPU-delta frame: a geometry's
    // shared BLAS can be rebuilt in place at a new LOD level (detected
    // GPU-side in `blas_sharing::elect_dirty`), and the instances
    // referencing it must be re-WRITTEN so the partition re-reads the
    // rebuilt BLAS bounds. `fill_incremental` writes only the CPU delta
    // plus instances of dirty geometries, so a truly static frame
    // produces an empty WRITE op (cheap). [A GPU "any-dirty" readback
    // could restore the full static-frame build skip — follow-up.]

    // CPU-seeded delta. Full rebuild seeds nothing — `force_all` writes
    // every active instance GPU-side. Incremental seeds added ∪ rewrite ∪
    // disabled — the CPU-only events GPU detection can't see: added's `last`
    // may alias a reused slot; rewrite is a cull-mask change (baked into the
    // TLAS record); disabled isn't active. MOVED instances are NOT seeded
    // here — `fill_incremental` detects those GPU-side by comparing each
    // active instance's world transform against last frame's. Static
    // instances are carried from `src`; movers and band-crossers are appended
    // by `fill_incremental`.
    resources.write_slots_cpu.clear();
    if !full_rebuild {
        for slot in instances
            .added_slots()
            .iter()
            .chain(instances.rewrite_slots())
        {
            resources.write_slots_cpu.push(PtlasWritePair {
                slot: slot.0,
                null_flag: PAIR_NORMAL,
            });
        }
        for slot in instances.disabled_slots() {
            resources.write_slots_cpu.push(PtlasWritePair {
                slot: slot.0,
                null_flag: PAIR_NULL,
            });
        }
    }
    let cpu_count = resources.write_slots_cpu.len() as u32;

    // RawBufferVec skips empty uploads, but the bind group needs a live
    // buffer. Keep a dummy element (`cpu_count` gates the seed dispatch).
    if resources.write_slots_cpu.is_empty() {
        resources.write_slots_cpu.push(PtlasWritePair::default());
    }
    resources
        .write_slots_cpu
        .write_buffer(&render_device, &render_queue);

    // Seed the GPU record counter to `cpu_count`; `fill_incremental`
    // atomic-appends band-crossers from there.
    render_queue.write_buffer(&resources.write_count, 0, &cpu_count.to_le_bytes());

    // Single WRITE op. CPU writes everything but `arg_count`, which the
    // `finalize` pass overwrites with the GPU-driven record count.
    let op = IndirectCommand {
        op_type: vk::PartitionedAccelerationStructureOpTypeNV::WRITE_INSTANCE.as_raw() as u32,
        arg_count: 0, // GPU-written by `finalize`
        arg_data_start_address: resources.write_data.address,
        arg_data_stride: WRITE_INSTANCE_DATA_SIZE,
    };
    render_queue.write_buffer(&resources.src_infos, 0, bytemuck::bytes_of(&op));
    render_queue.write_buffer(&resources.src_infos_count, 0, &1u32.to_le_bytes());
    let op_count = 1u32;

    // Worst-case record count = CPU delta + every active instance whose
    // geometry rebuilt this frame + every hair instance (re-specified each
    // frame). Commit `write_data` for it.
    let max_records = cpu_count as u64 + active_count as u64 + hair_count as u64;
    resources
        .write_data
        .commit(0..(max_records.max(1)) * WRITE_INSTANCE_DATA_SIZE);

    *resources.fill_params.get_mut() = PtlasFillParamsGpu {
        active_count,
        cpu_count,
        force_all: full_rebuild as u32,
    };
    resources
        .fill_params
        .write_buffer(&render_device, &render_queue);

    // ── Size + commit + AS handle for the storage buffer — must happen
    //    before the binder reads `current_tlas()` in PrepareBindGroups. ──
    let capacity = high_water;
    // One regular (static) partition plus the global (mover) partition. Maxima
    // stay at `capacity` (safe) until a GPU occupancy histogram lets us bound
    // them; a tighter cap faults the build if real occupancy exceeds it.
    let size_input = vk::PartitionedAccelerationStructureInstancesInputNV::default()
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .instance_count(capacity)
        .max_instance_per_partition_count(capacity)
        .partition_count(PTLAS_PARTITION_COUNT)
        .max_instance_in_global_partition_count(capacity);
    let mut sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: size_input populated; partitioned fn table loaded.
    {
        let _span = tracing::info_span!("ptlas.get_build_sizes").entered();
        unsafe {
            partitioned_fns
                .get_partitioned_acceleration_structures_build_sizes(&size_input, &mut sizes_info);
        }
    }
    {
        let _span = tracing::info_span!("ptlas.commit").entered();
        resources
            .storage
            .commit(0..sizes_info.acceleration_structure_size.max(1));
        let scratch_pad = PTLAS_SCRATCH_ALIGN - 1;
        resources
            .scratch
            .commit(0..(sizes_info.build_scratch_size.max(1) + scratch_pad));
    }

    // (Re)create the AS handle if its build size changed.
    if resources.tlas.is_none()
        || resources.as_handle_size != sizes_info.acceleration_structure_size
    {
        let _span = tracing::info_span!("ptlas.create_as_handle").entered();
        resources.tlas = None;
        // SAFETY: as_hal yields the raw VkBuffer while the SparseBuffer
        // is alive; created with ACCELERATION_STRUCTURE_STORAGE_KHR.
        let storage_vk_buffer = unsafe {
            resources
                .storage
                .wgpu_buffer
                .as_hal::<VkApi>()
                .expect("ptlas storage must be Vulkan-backed")
                .raw_handle()
        };
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(storage_vk_buffer)
            .offset(0)
            .size(sizes_info.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
        // SAFETY: buffer live + correct usage; offset 0 + matching size.
        let new_handle = unsafe {
            fns.acceleration_structure
                .create_acceleration_structure(&create_info, None)
                .expect("vkCreateAccelerationStructureKHR for PTLAS")
        };
        // SAFETY: handle is live.
        let new_addr = unsafe {
            fns.acceleration_structure
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(new_handle),
                )
        };
        // SAFETY: handle + buffer created on this device; both outlive
        // the wrapped Tlas (the storage SparseBuffer outlives this).
        let hal_as = unsafe {
            wgpu::hal::vulkan::AccelerationStructure::from_raw(new_handle, storage_vk_buffer)
        };
        let tlas_desc = CreateTlasDescriptor {
            label: Some("cluster_ptlas"),
            max_instances: capacity,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        };
        // SAFETY: hal_as from this device's hal; descriptor matches the
        // TOP_LEVEL AS type.
        let tlas = unsafe {
            render_device
                .wgpu_device()
                .create_tlas_from_hal::<VkApi>(hal_as, &tlas_desc)
        };
        resources.tlas = Some(tlas);
        resources.as_handle_size = sizes_info.acceleration_structure_size;
        resources.as_handle_device_address = new_addr;
    }
    resources.as_capacity = capacity;

    resources.cpu_count = cpu_count;
    resources.op_count = op_count;
    resources.full_rebuild = full_rebuild;
}

/// `Render::PrepareBindGroups`: rebuild the fill-compute bind group.
pub fn prepare_ptlas_fill_bind_group(
    mut ptlas: Option<ResMut<Ptlas>>,
    resource_manager: Option<Res<SolariResourceManager>>,
    pipeline_cache: Res<PipelineCache>,
    sharing: Option<Res<BlasSharing>>,
    geometry_ids: Option<Res<GpuColumn<GeometryIdColumn>>>,
    instance_masks: Option<Res<GpuColumn<InstanceMaskColumn>>>,
    material_ids: Option<Res<GpuColumn<MaterialColumn>>>,
    material_flags: Res<MaterialTraversalFlags>,
    transforms: Option<Res<GpuColumn<TransformColumn>>>,
    node_slots: Option<Res<GpuColumn<NodeSlotColumn>>>,
    static_flags: Option<Res<GpuColumn<Presence<StaticColumn>>>>,
    render_device: Res<RenderDevice>,
) {
    let Some(ptlas) = ptlas.as_deref_mut() else {
        return;
    };
    let (
        Some(resource_manager),
        Some(sharing),
        Some(geometry_ids),
        Some(instance_masks),
        Some(material_ids),
        Some(transforms),
        Some(node_slots),
        Some(static_flags),
    ) = (
        resource_manager,
        sharing,
        geometry_ids,
        instance_masks,
        material_ids,
        transforms,
        node_slots,
        static_flags,
    )
    else {
        ptlas.bind_group = None;
        return;
    };
    let (
        Some(params_binding),
        Some(write_slots),
        Some(active_to_slot),
        Some(previous_transforms),
        Some(material_flags),
    ) = (
        ptlas.fill_params.binding(),
        ptlas.write_slots_cpu.buffer(),
        sharing.active_to_slot.buffer(),
        // `fill_incremental` compares current vs previous to detect moves.
        // `TransformColumn` is `KEEP_PREVIOUS`, so this is always `Some`.
        transforms.previous_buffer(),
        material_flags.buffer.buffer(),
    ) else {
        // The buffers are populated earlier in Prepare; None here means
        // no instances yet.
        ptlas.bind_group = None;
        return;
    };
    let geometry_ids = geometry_ids.buffer().as_entire_binding();
    let instance_masks = instance_masks.buffer().as_entire_binding();
    let material_ids = material_ids.buffer().as_entire_binding();

    let group = render_device.create_bind_group(
        "ptlas_fill_bind_group",
        &pipeline_cache.get_bind_group_layout(&resource_manager.ptlas),
        &BindGroupEntries::sequential((
            sharing.instance_blas_address.wgpu_buffer.as_entire_binding(),
            sharing.geometry_dirty.as_entire_binding(),
            ptlas.write_count.as_entire_binding(),
            ptlas.write_data.wgpu_buffer.as_entire_binding(),
            write_slots.as_entire_binding(),
            active_to_slot.as_entire_binding(),
            ptlas.src_infos.as_entire_binding(),
            params_binding,
            geometry_ids,
            instance_masks,
            previous_transforms.as_entire_binding(),
            material_ids,
            material_flags.as_entire_binding(),
            ptlas.instance_written_flags.as_entire_binding(),
            // Instance → transform-table node slot, and the node-indexed
            // `TransformStatic` presence flag — `resolve_partition` reads
            // `static_flags[node_slots[slot]]` to pick the instance's partition.
            node_slots.buffer().as_entire_binding(),
            static_flags.buffer().as_entire_binding(),
            ptlas.instance_written_partition.as_entire_binding(),
        )),
    );
    ptlas.bind_group = Some(group);
}

/// `RenderGraph`: fill the WRITE/UPDATE record buffers then record the
/// partitioned-AS build. Runs after `dispatch_blas_rebuild` (whose fresh
/// BLAS addresses the fill samples). The op stream + batch counts were
/// prepared in [`prepare_ptlas_params`]. The wgpu fill records into the
/// shared `RenderContext` encoder; the raw-VK build records into its own
/// encoder, handed to the same context — the graph does one submit for the
/// frame.
pub fn dispatch_ptlas(
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
    fns: Option<Res<ClusterExtensionFns>>,
    mut resources: Option<ResMut<Ptlas>>,
    pipelines: Res<SolariPipelines>,
    scene_bind_group: Res<ClusterSceneBindGroup>,
    pipeline_cache: Res<PipelineCache>,
    instances: Option<Res<InstanceManager>>,
    hair_instances: Option<Res<crate::hair::HairInstances>>,
    hair_write: Option<Res<crate::hair::ptlas_hair::HairPtlasWrite>>,
    additional: Res<AdditionalVulkanFeatures>,
    mut ctx: RenderContext,
) {
    // The shading path is the RT pipeline (`vkCmdTraceRays`), so the post-build
    // barrier must publish AS writes to `RAY_TRACING_SHADER_KHR` — but only when
    // that feature is enabled (else the stage flag is illegal → device lost).
    let rt_pipeline = additional.has::<RayTracingPipelineFeature>();
    let (Some(allocator), Some(fns), Some(resources), Some(instances)) = (
        allocator,
        fns,
        resources.as_deref_mut(),
        instances,
    ) else {
        return;
    };
    if fns.partitioned.is_none() {
        return;
    }
    // `op_count == 0` means `prepare_ptlas_params` decided there was
    // nothing to build this frame (no delta) — `storage` already holds
    // the current PTLAS, so leave it alone.
    if resources.op_count == 0 {
        return;
    }
    let hair_count = hair_instances.as_ref().map(|h| h.count).unwrap_or(0);
    let capacity = instances.slot_high_water() + hair_count;
    if capacity == 0 {
        return;
    }
    let (Some(scene_bg), Some(fill_bg)) =
        (scene_bind_group.bind_group.as_ref(), resources.bind_group.as_ref())
    else {
        return;
    };
    let (Some(seed_pipe), Some(incremental_pipe), Some(finalize_pipe)) = (
        pipeline_cache.get_compute_pipeline(pipelines.ptlas_seed),
        pipeline_cache.get_compute_pipeline(pipelines.ptlas_incremental),
        pipeline_cache.get_compute_pipeline(pipelines.ptlas_finalize),
    ) else {
        return;
    };
    let active_count = instances.active_count() as u32;

    // The instances input for the build. Sizing, sparse commits, and
    // the AS handle were all done in `prepare_ptlas_params` (before the
    // binder); this is just the build's `input` descriptor. `partition_count`
    // must match the sizing query's.
    let size_input = vk::PartitionedAccelerationStructureInstancesInputNV::default()
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .instance_count(capacity)
        .max_instance_per_partition_count(capacity)
        .partition_count(PTLAS_PARTITION_COUNT)
        .max_instance_in_global_partition_count(capacity);

    let scratch_base = resources.scratch.address;
    let scratch_misalign = scratch_base & (PTLAS_SCRATCH_ALIGN - 1);
    let scratch_offset = if scratch_misalign == 0 {
        0
    } else {
        PTLAS_SCRATCH_ALIGN - scratch_misalign
    };
    let scratch_addr = scratch_base + scratch_offset;

    // Incremental: in-place update — `src == dst == storage` (the spec permits
    // equal src/dst, so the driver updates the structure in place instead of
    // copying the whole thing into a fresh buffer every frame). Full rebuild
    // keeps `src = 0` (built from scratch into the same buffer).
    //
    // EXPECTED VALIDATION NOISE: the in-place case trips
    // `VUID-vkCmdBuildPartitionedAccelerationStructuresNV-pBuildInfo-10549`
    // ("dst intersects src") every incremental frame. That VUID is the generic
    // KHR-AS no-overlap rule; the NV partitioned-AS extension explicitly allows
    // src == dst for in-place update, so it's a validation-layer false positive
    // here, not a bug. Do not "fix" it by ping-ponging buffers.
    let storage_addr = resources.storage.address;
    let src_acceleration_structure_data = if resources.full_rebuild {
        0
    } else {
        storage_addr
    };
    let build_info = vk::BuildPartitionedAccelerationStructureInfoNV {
        s_type: vk::BuildPartitionedAccelerationStructureInfoNV::STRUCTURE_TYPE,
        p_next: core::ptr::null_mut(),
        input: size_input,
        src_acceleration_structure_data,
        dst_acceleration_structure_data: storage_addr,
        scratch_data: scratch_addr,
        src_infos: allocator.wgpu_buffer_device_address(&resources.src_infos),
        src_infos_count: allocator.wgpu_buffer_device_address(&resources.src_infos_count),
        _marker: core::marker::PhantomData,
    };

    // wgpu fill records into the shared render-context encoder; the raw-VK
    // build gets its OWN encoder (the fork panics if one encoder mixes wgpu
    // passes with raw `as_hal_mut`). `add_command_buffer` flushes this fill
    // work first, so on the single queue the build still runs after it.
    {
        let diagnostics = ctx.diagnostic_recorder();
        let diagnostics = diagnostics.as_deref();
        let encoder = ctx.command_encoder();
        // One span over the fill passes (the raw-VK build below is invisible to
        // wgpu timestamp queries, so it can't be captured here). Taken on the
        // encoder since these are separate compute passes.
        let d = diagnostics.time_span(encoder, "ptlas_fill");
        // Separate passes so wgpu inserts the storage barriers each step's
        // producer→consumer chain needs (seed/incremental write `write_data`
        // + `write_count`; finalize reads `write_count` → `src_infos`).
        if resources.cpu_count > 0 {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("ptlas.fill_seed"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, scene_bg, &[]);
            pass.set_bind_group(1, fill_bg, &[]);
            pass.set_pipeline(seed_pipe);
            let (gx, gy, gz) = crate::ecs_gpu::linear_dispatch(resources.cpu_count.div_ceil(64));
            pass.dispatch_workgroups(gx, gy, gz);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("ptlas.fill_incremental"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, scene_bg, &[]);
            pass.set_bind_group(1, fill_bg, &[]);
            pass.set_pipeline(incremental_pipe);
            let (gx, gy, gz) = crate::ecs_gpu::linear_dispatch(active_count.div_ceil(64));
            pass.dispatch_workgroups(gx, gy, gz);
        }
        // Hair: append hair instances to the WRITE stream (same `write_count`),
        // after the cluster movers and before `finalize` publishes the count.
        if hair_count > 0 {
            if let (Some(hair_bg), Some(hair_pipe)) = (
                hair_write.as_ref().and_then(|w| w.bind_group.as_ref()),
                pipeline_cache.get_compute_pipeline(pipelines.ptlas_hair_write),
            ) {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("ptlas.hair_write"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(hair_pipe);
                pass.set_bind_group(0, hair_bg, &[]);
                let (gx, gy, gz) = crate::ecs_gpu::linear_dispatch(hair_count.div_ceil(64));
                pass.dispatch_workgroups(gx, gy, gz);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("ptlas.fill_finalize"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, scene_bg, &[]);
            pass.set_bind_group(1, fill_bg, &[]);
            pass.set_pipeline(finalize_pipe);
            pass.dispatch_workgroups(1, 1, 1);
        }
        d.end(encoder);
    }

    let mut build_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("ptlas.build"),
    });
    // SAFETY: encoder is open and Vulkan-backed; partitioned-AS
    // function table is loaded.
    unsafe {
        // PRE-build barrier: SHADER_WRITE → AS_BUILD_INPUT_READ so
        // the build sees the fill-compute's freshly-written
        // WriteInstanceData records. (Compute → build; no RT stage needed.)
        crate::gpu::extension::cmd_global_as_barrier(&mut build_encoder, &render_device, false);
        crate::gpu::extension::cmd_build_partitioned_acceleration_structures(
            &mut build_encoder,
            &fns,
            &build_info,
        );
        // POST-build barrier: AS_WRITE → RAY_TRACING_SHADER_READ so the RT-pipeline
        // trace sees fresh PTLAS contents. Without the RT stage here, the trace
        // races the build and reads an empty AS → every ray misses.
        crate::gpu::extension::cmd_global_as_barrier(&mut build_encoder, &render_device, rt_pipeline);
    }
    ctx.add_command_buffer(build_encoder.finish());

    // A PTLAS now exists in `storage`, so subsequent frames can build
    // incrementally in place (`src = dst = storage`) instead of from scratch.
    resources.has_built = true;
}
