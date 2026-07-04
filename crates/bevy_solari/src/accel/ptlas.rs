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
//! Build mode: **double-buffered** incremental update. The NV partitioned
//! build reads `src` (the previously-built PTLAS, "used as a basis") and
//! writes a fresh structure to `dst`; the spec REQUIRES the two not overlap
//! (VUID-...10549), so each build reads the last-built storage buffer and
//! writes the other, then flips `current`. The driver carries unchanged
//! partitions from `src` and applies only the op-delta. A full rebuild
//! (`src = 0`, WRITE every active slot) happens on the first build and
//! whenever capacity grows. Builds are skipped entirely on frames with no
//! op-delta (`op_count == 0`), so the per-build migration cost is paid only
//! when geometry actually changes.
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
use crate::gpu::epoch_table::EpochTable;
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

/// Regular-partition count. Partition 0 = the legacy shared static partition
/// (un-hinted statics); 1.. are CPU-assigned per streamed spatial cell via
/// `SolariPartition` — tight AABBs by construction (the earlier HASHED grid
/// gave scene-spanning partitions and was reverted; explicit cell ids don't).
/// Movers stay in the global partition.
pub const PTLAS_PARTITION_COUNT: u32 = 16384;

/// Uniform layout shared with `ptlas_fill.wgsl::PtlasFillParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
pub struct PtlasFillParamsGpu {
    pub active_count: u32,
    pub cpu_count: u32,
    pub force_all: u32,
    /// 1 → OMM-consult validation mode (drop FORCE_NO_OPAQUE so micromaps drive).
    /// Set from `SOLARI_OMM_CONSULT` when the extension is available.
    pub omm_consult: u32,
    /// 1 → force the OMM to 2-state at traversal (collapse "unknown" micro-tris to
    /// opaque/transparent via the bake's lean), so NO micro-triangle ever invokes
    /// the any-hit. Trades a sub-micro-triangle-precise cutout edge for zero any-hit.
    /// Set from `SOLARI_OMM_2STATE` when the extension is available.
    pub omm_force_2_state: u32,
    /// This build's seed-epoch stamp (≥1) — `fill_seed` marks its slots,
    /// `fill_incremental` skips them (one WRITE per instance per build).
    pub epoch: u32,
}

/// Render-world resource for the incremental partitioned-TLAS pass.
///
/// Bundles everything the pass owns — the single AS storage buffer,
/// the per-frame fill I/O buffers, the three fill compute pipelines +
/// bind-group layout, and the per-frame fill bind group.
#[derive(Resource)]
pub struct Ptlas {
    /// PTLAS storage — **double-buffered** (`[src, dst]` flip). NV's partitioned
    /// build reads `srcAccelerationStructureData` (the previous PTLAS, "used as a
    /// basis") and writes a fresh structure to `dstAccelerationStructureData`;
    /// the spec REQUIRES the two regions not overlap (VUID-...10549), so an
    /// incremental update can't alias one buffer. Each build reads the
    /// last-built buffer and writes the other, then flips [`Self::current`].
    /// A full rebuild (`src = 0`) writes the destination from scratch.
    pub storage: [SparseBuffer; 2],
    /// Index of the most-recently-built storage buffer — the one the trace binds
    /// (via [`Self::current_tlas`]). Flipped in `prepare_ptlas_params` on build
    /// frames (BEFORE the bind group is built, so the bound TLAS matches the
    /// buffer the build writes this frame). Static frames keep it (the prior
    /// PTLAS is still valid).
    pub current: usize,
    /// The source buffer index for THIS frame's build (the prior `current`),
    /// chosen in `prepare_ptlas_params` and consumed by `dispatch_ptlas`.
    pub build_src_idx: usize,
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
    /// `current_tlas()` returns the active one. Each wraps a
    /// `vkCreateAccelerationStructureKHR` handle over its storage buffer via
    /// [`wgpu::Device::create_tlas_from_hal`], recreated only when the build size
    /// changes (the old wrapper's Drop calls `vkDestroyAccelerationStructureKHR`).
    /// One per double-buffered storage buffer.
    pub tlas: [Option<Tlas>; 2],
    /// Build-size the [`Self::tlas`] handles were created for; mismatch with this
    /// frame's `sizes_info.acceleration_structure_size` recreates BOTH handles.
    pub as_handle_size: u64,
    /// `vkGetAccelerationStructureDeviceAddressKHR` per handle.
    pub as_handle_device_address: [vk::DeviceAddress; 2],
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
    /// via `force_all`); `false` → incremental (`src` = last-built buffer as
    /// basis, `dst` = the other; WRITE only the CPU delta + GPU band-crossers,
    /// static instances carried from `src`).
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

    /// Slot-indexed epoch stamps `fill_seed` writes / `fill_incremental` skips —
    /// dedupes CPU-seeded vs GPU-appended WRITEs within one build.
    pub seed_epoch: EpochTable,
    /// Bytes committed on [`Self::scratch`] this frame (sizing query + align pad).
    pub scratch_commit: u64,
    /// Slots freed AND re-bound in the same frame: their null lands this build,
    /// the re-write next build — an instance index must never move partitions
    /// (or fight a null) inside one incremental build.
    pub deferred_adds: Vec<u32>,

    /// Debug (SOLARI_PTLAS_VALIDATE): GPU report of corrupt BLAS addresses
    /// caught in the WRITE stream ([0..4)=valid span, [4]=count, [5..)=entries),
    /// its CPU-readback staging twin, and the in-flight flag.
    pub validate_report: Buffer,
    pub validate_staging: Buffer,
    pub validate_in_flight: bool,
}

/// u32 words in the validation report: 4 span + capacity + partition count +
/// bad count + 15 × 5-word entries.
const VALIDATE_REPORT_WORDS: u64 = 7 + 15 * 5;

/// SOLARI_PTLAS_VALIDATE=1 → scan + null corrupt BLAS addresses each build,
/// logging offenders (slot + address) instead of device-losting in the build.
fn ptlas_validate_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SOLARI_PTLAS_VALIDATE").as_deref() == Ok("1")
            || crate::gpu::extension::solari_validate_enabled()
    })
}

/// SOLARI_PTLAS_FULL_REBUILD=1 → build from scratch every frame (no `src`
/// carry). Bisect lever: if device-losts stop, the corruption lives in the
/// incremental/carry path.
fn ptlas_force_full_rebuild() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SOLARI_PTLAS_FULL_REBUILD").as_deref() == Ok("1"))
}

impl Ptlas {
    /// This frame's PTLAS — the build target the path-tracer binds.
    /// `None` until the first build has produced its handle.
    #[inline]
    pub fn current_tlas(&self) -> Option<&Tlas> {
        self.tlas[self.current].as_ref()
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

    // Double-buffered storage — the partitioned build reads one (basis) and
    // writes the other (spec forbids src/dst overlap, so no in-place aliasing).
    let storage = [
        allocator.create_sparse_buffer(
            &render_device,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            wgpu::BufferUsages::COPY_DST,
            PTLAS_STORAGE_VIRTUAL_BYTES,
            "ptlas.storage.a",
        ),
        allocator.create_sparse_buffer(
            &render_device,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            wgpu::BufferUsages::COPY_DST,
            PTLAS_STORAGE_VIRTUAL_BYTES,
            "ptlas.storage.b",
        ),
    ];
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

    let seed_epoch = EpochTable::new(&render_device, "ptlas.seed_epoch");

    let validate_report = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ptlas.validate_report"),
        size: VALIDATE_REPORT_WORDS * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let validate_staging = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ptlas.validate_staging"),
        size: VALIDATE_REPORT_WORDS * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
        tlas: [None, None],
        as_handle_size: 0,
        as_handle_device_address: [0, 0],
        current: 0,
        build_src_idx: 0,
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
        seed_epoch,
        scratch_commit: 0,
        deferred_adds: Vec::new(),
        validate_report,
        validate_staging,
        validate_in_flight: false,
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
    tess_found: Option<Res<crate::geometry::tess_displace::TessShowcaseInstances>>,
    tess_classify: Option<Res<crate::geometry::tess_classify::TessClassify>>,
    deform: Option<Res<super::deform::Deform>>,
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
    // The GPU-tessellated instances occupy PTLAS slots above the cluster + hair slots,
    // once `tess_classify` has built their per-instance BLAS. Gate + count identically to
    // `prepare_tess_ptlas_write` so the slot reservation matches its `tess_base`.
    let tess_blas_ready = tess_classify.as_ref().is_some_and(|c| c.blas_ready);
    let tess_count = match (&tess_found, tess_blas_ready) {
        (Some(f), true) if f.found => f.instances.len() as u32,
        _ => 0,
    };

    let cluster_high_water = instances.slot_high_water();
    // Total PTLAS instance space (cluster slots + hair + tessellation showcase).
    let high_water = cluster_high_water + hair_count + tess_count;
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
    // ANY CPU-known churn (add/remove/rewrite) takes the full-rebuild path:
    // incremental updates that WRITE instances across many regular partitions
    // fault the driver (580.159) — full rebuilds with 1024 partitions are clean
    // and measured fast. Pure mover / no-op frames stay incremental (or skip).
    let churn = !instances.added_slots().is_empty()
        || !instances.disabled_slots().is_empty()
        || !instances.rewrite_slots().is_empty()
        || !resources.deferred_adds.is_empty();
    let full_rebuild =
        !resources.has_built || grew || churn || ptlas_force_full_rebuild();
    if churn {
        tracing::debug!(
            "ptlas churn: +{} -{} ~{} deferred {}",
            instances.added_slots().len(),
            instances.disabled_slots().len(),
            instances.rewrite_slots().len(),
            resources.deferred_adds.len(),
        );
    }
    if full_rebuild {
        // force_all restamps every active slot — parked re-adds are covered.
        resources.deferred_adds.clear();
    }

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
        // ONE record per instance per build (duplicates are spec-UB and corrupt
        // the partitioned build), and an instance index freed + re-bound in one
        // frame is written over TWO builds: null now, re-write next build — a
        // same-build null/write pair (or a cross-partition move without an
        // intervening remove) corrupts the incremental partitioned build.
        let disabled: std::collections::HashSet<u32> =
            instances.disabled_slots().iter().map(|s| s.0).collect();
        let mut seeded: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for slot in core::mem::take(&mut resources.deferred_adds) {
            // Freed again while parked → the instance is gone; drop the re-add.
            if disabled.contains(&slot) {
                continue;
            }
            if seeded.insert(slot) {
                resources.write_slots_cpu.push(PtlasWritePair {
                    slot,
                    null_flag: PAIR_NORMAL,
                });
            }
        }
        for slot in instances
            .added_slots()
            .iter()
            .chain(instances.rewrite_slots())
        {
            if disabled.contains(&slot.0) {
                resources.deferred_adds.push(slot.0);
                continue;
            }
            if seeded.insert(slot.0) {
                resources.write_slots_cpu.push(PtlasWritePair {
                    slot: slot.0,
                    null_flag: PAIR_NORMAL,
                });
            }
        }
        // Animated instances rebuild their per-instance BLAS in place every frame
        // (stable address, new content), which `fill_incremental` can't detect — a
        // still fox wouldn't "move". Force-rewrite them so the partition re-reads.
        if let Some(deform) = deform.as_ref() {
            for s in deform.active_slots() {
                if disabled.contains(&s.instance_slot) {
                    continue;
                }
                if seeded.insert(s.instance_slot) {
                    resources.write_slots_cpu.push(PtlasWritePair {
                        slot: s.instance_slot,
                        null_flag: PAIR_NORMAL,
                    });
                }
            }
        }
        for slot in instances.disabled_slots() {
            if seeded.insert(slot.0) {
                resources.write_slots_cpu.push(PtlasWritePair {
                    slot: slot.0,
                    null_flag: PAIR_NULL,
                });
            }
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
    // frame) + the tessellation showcase. Commit `write_data` for it.
    let max_records =
        cpu_count as u64 + active_count as u64 + hair_count as u64 + tess_count as u64;
    resources
        .write_data
        .commit(0..(max_records.max(1)) * WRITE_INSTANCE_DATA_SIZE);

    let (epoch, seed_grew) = resources.seed_epoch.begin(&render_device, high_water);
    debug_assert!(!seed_grew || full_rebuild);
    *resources.fill_params.get_mut() = PtlasFillParamsGpu {
        active_count,
        cpu_count,
        force_all: full_rebuild as u32,
        epoch,
        // DEBUG: default ON whenever OMM is available so the micromap drives
        // traversal (drops FORCE_NO_OPAQUE). SOLARI_OMM_CONSULT=0 forces it off.
        // (Holes on non-OMM cutouts render solid under it — the per-geometry has_omm
        // column is the correct always-on fix.)
        omm_consult: {
            let avail = crate::gpu::extension::opacity_micromap_available();
            let off = std::env::var("SOLARI_OMM_CONSULT").as_deref() == Ok("0");
            let on = (avail && !off) as u32;
            static LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::debug!("ptlas: omm_consult={on} (omm_available={avail})");
            }
            on
        },
        // Force the micromap to 2-state at traversal (no any-hit on the unknown
        // edge band). OFF by default: forcing 2-state on a 4-state-FORMAT micromap
        // mis-resolves the cutout (the whole leaf collapses to transparent). The
        // correct zero-any-hit path is a NATIVE 2-state bake (OC1_2_State) in the
        // importer, not this runtime flag. Opt in with SOLARI_OMM_2STATE=1.
        omm_force_2_state: {
            let avail = crate::gpu::extension::opacity_micromap_available();
            let on = (std::env::var("SOLARI_OMM_2STATE").as_deref() == Ok("1")) as u32;
            static LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::debug!("ptlas: omm_force_2_state={on} (omm_available={avail})");
            }
            on
        },
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
    debug_assert!(
        sizes_info.build_scratch_size <= PTLAS_SCRATCH_VIRTUAL_BYTES,
        "PTLAS build scratch {} exceeds virtual reservation {}",
        sizes_info.build_scratch_size,
        PTLAS_SCRATCH_VIRTUAL_BYTES,
    );
    {
        let _span = tracing::info_span!("ptlas.commit").entered();
        // Commit BOTH storage buffers — the build reads one and writes the other,
        // and the bind group may reference either across frames.
        for s in &resources.storage {
            s.commit(0..sizes_info.acceleration_structure_size.max(1));
        }
        let scratch_pad = PTLAS_SCRATCH_ALIGN - 1;
        resources.scratch_commit = sizes_info.build_scratch_size.max(1) + scratch_pad;
        resources.scratch.commit(0..resources.scratch_commit);
    }

    // (Re)create BOTH AS handles if the build size changed (one per
    // double-buffered storage buffer). They share a single size.
    if resources.tlas[0].is_none()
        || resources.tlas[1].is_none()
        || resources.as_handle_size != sizes_info.acceleration_structure_size
    {
        let _span = tracing::info_span!("ptlas.create_as_handle").entered();
        tracing::info!(
            "ptlas: AS size {} MiB, scratch {} MiB (capacity {}, {} partitions)",
            sizes_info.acceleration_structure_size >> 20,
            sizes_info.build_scratch_size >> 20,
            capacity,
            PTLAS_PARTITION_COUNT,
        );
        tracing::info!(
            "ptlas: AS size {} MiB, scratch {} MiB (capacity {}, {} partitions)",
            sizes_info.acceleration_structure_size >> 20,
            sizes_info.build_scratch_size >> 20,
            capacity,
            PTLAS_PARTITION_COUNT,
        );
        for i in 0..2 {
            resources.tlas[i] = None; // Drop old handle first (frees the VkAS).
            // SAFETY: as_hal yields the raw VkBuffer while the SparseBuffer
            // is alive; created with ACCELERATION_STRUCTURE_STORAGE_KHR.
            let storage_vk_buffer = unsafe {
                resources.storage[i]
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
            resources.tlas[i] = Some(tlas);
            resources.as_handle_device_address[i] = new_addr;
        }
        resources.as_handle_size = sizes_info.acceleration_structure_size;
    }
    resources.as_capacity = capacity;

    resources.cpu_count = cpu_count;
    resources.op_count = op_count;
    resources.full_rebuild = full_rebuild;

    // Flip the double buffer for THIS frame's build, BEFORE the binder reads
    // `current_tlas()` in PrepareBindGroups — so the bound TLAS is the one the
    // build writes this frame. Only on build frames (`dispatch_ptlas` early-outs
    // when `op_count == 0`, leaving the prior buffer bound and valid). The build
    // reads `build_src_idx` (the prior `current`) as its basis and writes
    // `current`; a full rebuild ignores the basis (`src = 0`).
    if op_count > 0 {
        resources.build_src_idx = resources.current;
        resources.current = 1 - resources.current;
    }
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
    partition_hints: Option<Res<GpuColumn<crate::instance::PartitionColumn>>>,
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
        Some(partition_hints),
    ) = (
        resource_manager,
        sharing,
        geometry_ids,
        instance_masks,
        material_ids,
        transforms,
        node_slots,
        static_flags,
        partition_hints,
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
            ptlas.validate_report.as_entire_binding(),
            ptlas.seed_epoch.buffer().as_entire_binding(),
            partition_hints.buffer().as_entire_binding(),
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
    tess_write: Option<Res<crate::geometry::tess_displace::TessPtlasWrite>>,
    additional: Res<AdditionalVulkanFeatures>,
    render_queue: Res<RenderQueue>,
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
    // Log what LAST build's validation pass caught (its copy has long retired).
    drain_ptlas_validation(resources, &render_device);
    // `op_count == 0` means `prepare_ptlas_params` decided there was
    // nothing to build this frame (no delta) — `storage` already holds
    // the current PTLAS, so leave it alone.
    if resources.op_count == 0 {
        return;
    }
    let hair_count = hair_instances.as_ref().map(|h| h.count).unwrap_or(0);
    // Must match `prepare_ptlas_params`'s `high_water`: the showcase tess instance
    // occupies index `cluster_high_water + hair_count`, so the build's
    // `instance_count` must include it. Omitting it makes that record's
    // `instance_index` equal `instance_count` (one past the bound) → the build
    // faults → device lost.
    let tess_count = tess_write.as_ref().map_or(0, |t| t.tess_count);
    let capacity = instances.slot_high_water() + hair_count + tess_count;
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

    // Double-buffered build (spec requires src/dst not overlap, VUID-...10549):
    // read the previously-built buffer as the basis and write the other, which
    // `prepare_ptlas_params` already flipped `current` to (so the bound TLAS
    // matches). Full rebuild ignores the basis (`src = 0`, built from scratch).
    let storage_addr = resources.storage[resources.current].address;
    let src_acceleration_structure_data = if resources.full_rebuild {
        0
    } else {
        resources.storage[resources.build_src_idx].address
    };
    let build_info = vk::BuildPartitionedAccelerationStructureInfoNV {
        s_type: vk::BuildPartitionedAccelerationStructureInfoNV::STRUCTURE_TYPE,
        p_next: core::ptr::null_mut(),
        input: size_input,
        src_acceleration_structure_data,
        dst_acceleration_structure_data: storage_addr,
        scratch_data: scratch_addr,
        src_infos: allocator.wgpu_buffer_device_address(&resources.src_infos).get(),
        src_infos_count: allocator.wgpu_buffer_device_address(&resources.src_infos_count).get(),
        _marker: core::marker::PhantomData,
    };

    // Reset the validation report head: valid span, this build's instance
    // capacity + partition count, zeroed bad-record count.
    let validate = ptlas_validate_enabled();
    if validate {
        let (lo, hi) = allocator.sparse_va_span();
        let head = [
            lo as u32,
            (lo >> 32) as u32,
            hi as u32,
            (hi >> 32) as u32,
            capacity,
            PTLAS_PARTITION_COUNT,
            0u32,
        ];
        render_queue.write_buffer(&resources.validate_report, 0, bytemuck::cast_slice(&head));
    }
    let mut validate_recorded = false;

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
        // Tessellation showcase (brick B4): append its single instance to the
        // WRITE stream, after hair, before `finalize`.
        if let Some(tw) = tess_write.as_ref() {
            if tw.tess_count > 0 {
                if let (Some(tess_bg), Some(tess_pipe)) = (
                    tw.bind_group.as_ref(),
                    pipeline_cache.get_compute_pipeline(tw.pipeline),
                ) {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("ptlas.tess_write"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(tess_pipe);
                    pass.set_bind_group(0, tess_bg, &[]);
                    pass.dispatch_workgroups(tw.tess_count.div_ceil(64), 1, 1);
                }
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
        // Debug: scan the final WRITE stream, null + report corrupt BLAS
        // addresses before the raw build dereferences them.
        if validate {
            if let Some(validate_pipe) =
                pipeline_cache.get_compute_pipeline(pipelines.ptlas_validate)
            {
                let max_records = resources.cpu_count + active_count + hair_count + tess_count;
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("ptlas.validate"),
                        timestamp_writes: None,
                    });
                    pass.set_bind_group(0, scene_bg, &[]);
                    pass.set_bind_group(1, fill_bg, &[]);
                    pass.set_pipeline(validate_pipe);
                    let (gx, gy, gz) =
                        crate::ecs_gpu::linear_dispatch(max_records.max(1).div_ceil(64));
                    pass.dispatch_workgroups(gx, gy, gz);
                }
                encoder.copy_buffer_to_buffer(
                    &resources.validate_report,
                    0,
                    &resources.validate_staging,
                    0,
                    VALIDATE_REPORT_WORDS * 4,
                );
                validate_recorded = true;
            }
        }
        d.end(encoder);
    }
    resources.validate_in_flight = validate_recorded;

    // Declared access for the raw build — SOLARI_VALIDATE checks every range
    // is committed BEFORE the GPU faults on an anonymous VA.
    let max_record_bytes = (resources.cpu_count + active_count + hair_count + tess_count).max(1)
        as u64
        * WRITE_INSTANCE_DATA_SIZE;
    let src_read = if resources.full_rebuild {
        0..0
    } else {
        0..resources.as_handle_size
    };
    crate::gpu::extension::validate_raw_access(&crate::gpu::extension::RawAccess {
        op: "ptlas.build",
        reads: &[
            (&resources.write_data, 0..max_record_bytes),
            (&resources.storage[resources.build_src_idx], src_read),
        ],
        writes: &[
            (&resources.storage[resources.current], 0..resources.as_handle_size),
            (&resources.scratch, scratch_offset..resources.scratch_commit),
        ],
    });

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

    // A PTLAS now exists, so subsequent frames can build incrementally
    // (`src` = the buffer just built, `dst` = the other) instead of from scratch.
    resources.has_built = true;
}

/// Map last build's validation staging copy and log the corrupt records it
/// caught (the GPU already nulled them, so the build survived to report).
fn drain_ptlas_validation(resources: &mut Ptlas, render_device: &RenderDevice) {
    if !resources.validate_in_flight {
        return;
    }
    resources.validate_in_flight = false;
    let slice = resources.validate_staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = render_device.wgpu_device().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    if !matches!(rx.try_recv(), Ok(Ok(()))) {
        return;
    }
    {
        let data = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);
        static ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !ARMED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                "ptlas.validate armed: span 0x{:x}..0x{:x} capacity {}",
                words[0] as u64 | ((words[1] as u64) << 32),
                words[2] as u64 | ((words[3] as u64) << 32),
                words[4],
            );
        }
        let bad = words[6];
        if bad > 0 {
            tracing::error!(
                "ptlas.validate: {bad} corrupt WRITE record(s) defused, span 0x{:x}..0x{:x} capacity {}",
                words[0] as u64 | ((words[1] as u64) << 32),
                words[2] as u64 | ((words[3] as u64) << 32),
                words[4],
            );
            for e in 0..bad.min(15) as usize {
                let w = &words[7 + e * 5..7 + e * 5 + 5];
                let kind = match w[0] {
                    1 => "addr-out-of-span",
                    2 => "instance_index-oob",
                    3 => "partition-oob",
                    _ => "?",
                };
                tracing::error!(
                    "ptlas.validate: {kind} record {} slot {} value 0x{:x}",
                    w[1],
                    w[2],
                    w[3] as u64 | ((w[4] as u64) << 32),
                );
            }
        }
    }
    resources.validate_staging.unmap();
}
