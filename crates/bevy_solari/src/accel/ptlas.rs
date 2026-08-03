#![allow(unsafe_code, reason = "raw VK build via as_hal_mut")]

//! Partitioned TLAS (PTLAS) build for the cluster-AS pipeline: fill compute
//! writes `WriteInstanceData[]` records GPU-side over the per-instance BLAS
//! addresses from [`super::blas_rebuild`], then a raw-VK
//! `vkCmdBuildPartitionedAccelerationStructuresNV` consumes them.
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
//! `UPDATE_INSTANCE` is not used: it only refreshes a BLAS address (it
//! cannot change a transform), and stable per-slot addresses leave
//! nothing to refresh.
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
//! global; stable bulk → a regular partition). One regular partition keeps
//! the static BVH coherent; hashing statics across many partitions gives
//! each a scene-spanning AABB whose overlap inflates ray-traversal cost far
//! more than cheaper per-cell rebuilds save. The static/mover split lives in
//! `ptlas_fill.slang::resolve_partition`.

use ash::vk::{self, TaggedStructure};
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    render_resource::{
        AccelerationStructureFlags, AccelerationStructureUpdateMode, Buffer, CreateTlasDescriptor,
        RawBufferVec, Tlas,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
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
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::epoch_table::EpochTable;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
use super::blas_sharing::BlasSharing;
use crate::gpu::extension::{AsSeams, ClusterExtensionFns};

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
/// must match `ptlas_fill.slang`'s `WriteInstanceData` struct (the full
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

/// Regular-partition count. Partition 0 holds un-hinted statics; 1.. are
/// CPU-assigned per streamed spatial cell via `SolariPartition` — explicit
/// cell ids give tight AABBs by construction, where a hashed grid would give
/// scene-spanning partitions. Movers stay in the global partition.
pub const PTLAS_PARTITION_COUNT: u32 = 16384;

/// Push mirror of `ptlas_fill.slang::PtlasFillParams` (32 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct PtlasFillParamsGpu {
    pub active_count: u32,
    pub cpu_count: u32,
    pub force_all: u32,
    /// This build's seed-epoch stamp (≥1) — `fill_seed` marks its slots,
    /// `fill_incremental` skips them (one WRITE per instance per build).
    pub epoch: u32,
    /// X workgroup count of the entry's 2D-split dispatch — stamped per
    /// dispatch (each entry has its own thread bound).
    pub groups_x: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(size_of::<PtlasFillParamsGpu>() == 32);

/// The Slang modules `ptlas_fill.slang` imports (referenced by the compile
/// test too).
pub(crate) const PTLAS_FILL_MODULES: &[(&str, &str)] = &[
    (
        "cluster_bindings",
        include_str!("../bindings/cluster_bindings.slang"),
    ),
    (
        "instance_mask",
        include_str!("../instance/instance_mask.slang"),
    ),
];

/// The four PTLAS-fill heap kernels + the persistent slot table their set-1
/// parameters are written through. Built lazily on the first ready dispatch
/// (the mapping table bakes the cluster-scene heap slots).
pub struct PtlasKernels {
    pub seed: HeapKernel,
    pub incremental: HeapKernel,
    pub finalize: HeapKernel,
    pub validate: HeapKernel,
    /// One slot per set-1 buffer, indexed by the dispatch's slot-table order.
    pub slots: KernelSlots,
}

impl PtlasKernels {
    fn all(&self) -> [&HeapKernel; 4] {
        [&self.seed, &self.incremental, &self.finalize, &self.validate]
    }
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
    /// Per-frame fill-compute push params, filled in
    /// [`prepare_ptlas_params`]; `groups_x` is stamped per dispatch.
    pub fill_params: PtlasFillParamsGpu,
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

    /// The fill heap kernels, built lazily (see [`PtlasKernels`]).
    pub kernels: Option<PtlasKernels>,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,

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

    /// Debug (SolariSettings::ptlas_validate): GPU report of corrupt BLAS addresses
    /// caught in the WRITE stream ([0..4)=valid span, [4]=count, [5..)=entries),
    /// its CPU-readback staging twin, and the in-flight flag.
    pub validate_report: Buffer,
    pub validate_staging: Buffer,
    pub validate_in_flight: bool,

    /// Rebuild-until-clean: a build that stamps a record with a NULL BLAS address
    /// (its transform or BLAS address hadn't landed yet — the pipeline-warmup race)
    /// writes an instance that is INVISIBLE and, if it never moves, has no future
    /// re-spec trigger — the intermittent missing-static-scene-on-startup bug. The
    /// fill counts null records ([`NULL_COUNT_WORD`]); it's read back async and any
    /// non-zero count forces another `force_all` full rebuild, restamping everything
    /// until a build lands fully resolved. Readback latency (~2-4 frames) naturally
    /// paces the retries; steady state reads 0 and never re-arms.
    pub nulls_staging: Buffer,
    /// 0 = idle, 1 = copied (map next frame), 2 = map in flight.
    pub nulls_phase: u8,
    /// map_async result: 0 pending, 1 ok, 2 error.
    pub nulls_map_result: std::sync::Arc<std::sync::atomic::AtomicU8>,
    pub pending_null_rebuild: bool,
}

impl Drop for Ptlas {
    fn drop(&mut self) {
        if let Some(kernels) = self.kernels.take() {
            self._device_keepalive.quiesce_before_raw_destroy();
            // SAFETY: quiesced; handles exclusively owned here.
            unsafe {
                for kernel in kernels.all() {
                    kernel.destroy(&self.raw_device);
                }
            }
        }
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for Ptlas {}
unsafe impl Sync for Ptlas {}

/// u32 words in the validation report: 4 span + capacity + partition count +
/// bad count + 15 × 5-word entries.
// +2: trailing always-on heal counters (not just under SolariSettings::ptlas_validate):
// [82] null-AS records, [83] regular-partition writes in an incremental build.
// Both drive the rebuild-until-clean warmup heal below.
const VALIDATE_REPORT_WORDS: u64 = 7 + 15 * 5 + 2;
/// Report word holding the count of records the fill wrote with a null BLAS
/// address (transform not propagated / BLAS address not assigned yet).
const NULL_COUNT_WORD: u64 = 82;

static VALIDATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static FULL_REBUILD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Latch [`SolariSettings`](crate::SolariSettings)'s PTLAS debug levers at
/// plugin `finish`.
pub(crate) fn latch_debug_levers(validate: bool, full_rebuild: bool) {
    let _ = VALIDATE.set(validate);
    let _ = FULL_REBUILD.set(full_rebuild);
}

/// [`SolariSettings::ptlas_validate`](crate::SolariSettings) → scan + null
/// corrupt BLAS addresses each build, logging offenders (slot + address)
/// instead of device-losting in the build.
fn ptlas_validate_enabled() -> bool {
    VALIDATE.get().copied().unwrap_or(false)
}

/// [`SolariSettings::ptlas_full_rebuild`](crate::SolariSettings) → build from
/// scratch every frame (no `src` carry). Bisect lever: if device-losts stop,
/// the corruption lives in the incremental/carry path.
fn ptlas_force_full_rebuild() -> bool {
    FULL_REBUILD.get().copied().unwrap_or(false)
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
    let nulls_staging = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ptlas.nulls_staging"),
        size: 8,
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
        fill_params: PtlasFillParamsGpu::default(),
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
        kernels: None,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
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
        nulls_staging,
        nulls_phase: 0,
        nulls_map_result: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)),
        pending_null_rebuild: false,
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
    // fault the driver (580.159); full rebuilds are clean. Pure mover / no-op
    // frames stay incremental (or skip).
    // TODO: bounded-latency build batching — batch churn frames into a full
    // rebuild every N frames (adds/removes wait ≤N; geometry is pinned so a
    // lingering instance stays valid) to amortize the per-churn-frame rebuild.
    // Also re-test incremental multi-partition writes on each driver release —
    // if fixed, drop `|| churn` below.
    let churn = !instances.added_slots().is_empty()
        || !instances.disabled_slots().is_empty()
        || !instances.rewrite_slots().is_empty()
        || !resources.deferred_adds.is_empty();
    // `pending_null_rebuild`: last completed build stamped null-AS records
    // (warmup race) — restamp everything until a build lands fully resolved.
    let full_rebuild = !resources.has_built
        || grew
        || churn
        || ptlas_force_full_rebuild()
        || core::mem::take(&mut resources.pending_null_rebuild);
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

    // The build can't be skipped on a no-CPU-delta frame: a geometry's
    // shared BLAS can be rebuilt in place at a new LOD level (detected
    // GPU-side in `blas_sharing::elect_dirty`), and the instances
    // referencing it must be re-WRITTEN so the partition re-reads the
    // rebuilt BLAS bounds. `fill_incremental` writes only the CPU delta
    // plus instances of dirty geometries, so a truly static frame
    // produces an empty WRITE op (cheap). TODO: a GPU "any-dirty"
    // readback could skip the build entirely on static frames.

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
    // `groups_x` is per-entry; the dispatch stamps it into each push blob.
    resources.fill_params = PtlasFillParamsGpu {
        active_count,
        cpu_count,
        force_all: full_rebuild as u32,
        epoch,
        groups_x: 0,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

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

/// Non-blocking drain of the null-AS record counter copied after each build's
/// fill. Any non-zero count arms [`Ptlas::pending_null_rebuild`] — the
/// rebuild-until-clean warmup heal (see the field docs).
fn drain_null_count(resources: &mut Ptlas) {
    use std::sync::atomic::Ordering;
    match resources.nulls_phase {
        // Copied last build → the copy is submitted; start the map.
        1 => {
            resources.nulls_map_result.store(0, Ordering::Relaxed);
            let done = resources.nulls_map_result.clone();
            resources.nulls_staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                done.store(if r.is_ok() { 1 } else { 2 }, Ordering::Relaxed);
            });
            resources.nulls_phase = 2;
        }
        2 => match resources.nulls_map_result.load(Ordering::Relaxed) {
            1 => {
                let (nulls, regular_writes) = {
                    let data = resources.nulls_staging.slice(..).get_mapped_range();
                    (
                        u32::from_le_bytes([data[0], data[1], data[2], data[3]]),
                        u32::from_le_bytes([data[4], data[5], data[6], data[7]]),
                    )
                };
                resources.nulls_staging.unmap();
                resources.nulls_phase = 0;
                if nulls > 0 {
                    resources.pending_null_rebuild = true;
                    tracing::info!(
                        "ptlas: {nulls} null-AS record(s) in last build (warmup race) — forcing a full restamp"
                    );
                }
                if regular_writes > 0 {
                    // Incremental writes into regular partitions are silently
                    // driver-broken (the CPU-churn constraint, hit GPU-side) —
                    // redo them via the safe full-rebuild path. Fires EVERY
                    // frame the camera moves (camera-origin transforms churn
                    // all statics), so debug-level, not per-frame info spam.
                    resources.pending_null_rebuild = true;
                    tracing::debug!(
                        "ptlas: {regular_writes} regular-partition write(s) in an incremental build — forcing a full restamp (driver constraint)"
                    );
                }
            }
            2 => {
                resources.nulls_phase = 0; // map failed; retry after the next build
            }
            _ => {}
        },
        _ => {}
    }
}

/// Write the fill set-1 descriptors into the kernels' slot table and return
/// the `(parameter name, heap slot)` pairs each entry's push blob is
/// assembled from (each entry filters to the bindings surviving in its own
/// SPIR-V).
#[allow(clippy::too_many_arguments)]
fn ptlas_slot_table<'a>(
    seam: &BindingSeam,
    slots: &KernelSlots,
    ptlas: &Ptlas,
    sharing: &BlasSharing,
    write_slots: &Buffer,
    active_to_slot: &Buffer,
    previous_transforms: &Buffer,
    material_flags: &Buffer,
    columns: (&Buffer, &Buffer, &Buffer, &Buffer, &Buffer, &Buffer),
) -> Vec<(&'a str, u32)> {
    let (geometry_ids, instance_masks, material_ids, node_slots, static_flags, partition_hints) =
        columns;
    vec![
        (
            "instance_blas_address",
            slots.buffer(seam, 0, &sharing.instance_blas_address.wgpu_buffer),
        ),
        ("geometry_dirty", slots.buffer(seam, 1, &sharing.geometry_dirty)),
        ("write_count", slots.buffer(seam, 2, &ptlas.write_count)),
        ("write_data", slots.buffer(seam, 3, &ptlas.write_data.wgpu_buffer)),
        ("write_slots_cpu", slots.buffer(seam, 4, write_slots)),
        ("active_to_slot", slots.buffer(seam, 5, active_to_slot)),
        ("src_infos", slots.buffer(seam, 6, &ptlas.src_infos)),
        ("instance_geometry_ids", slots.buffer(seam, 7, geometry_ids)),
        ("instance_masks", slots.buffer(seam, 8, instance_masks)),
        (
            "instance_previous_transforms",
            slots.buffer(seam, 9, previous_transforms),
        ),
        ("instance_material_ids", slots.buffer(seam, 10, material_ids)),
        ("material_traversal_flags", slots.buffer(seam, 11, material_flags)),
        (
            "instance_written_flags",
            slots.buffer(seam, 12, &ptlas.instance_written_flags),
        ),
        // Instance → transform-table node slot, and the node-indexed
        // `TransformStatic` presence flag — `resolve_partition` reads
        // `static_flags[node_slots[slot]]` to pick the instance's partition.
        ("node_slots", slots.buffer(seam, 13, node_slots)),
        ("static_flags", slots.buffer(seam, 14, static_flags)),
        (
            "instance_written_partition",
            slots.buffer(seam, 15, &ptlas.instance_written_partition),
        ),
        ("validate_report", slots.buffer(seam, 16, &ptlas.validate_report)),
        ("seed_epoch", slots.buffer(seam, 17, ptlas.seed_epoch.buffer())),
        ("partition_hints", slots.buffer(seam, 18, partition_hints)),
        (
            "geometry_built_level",
            slots.buffer(seam, 19, &sharing.geometry_built_level),
        ),
        ("geometry_flags", slots.buffer(seam, 20, &sharing.geometry_flags)),
    ]
}

/// `RenderGraph`: fill the WRITE/UPDATE record buffers then record the
/// partitioned-AS build. Runs after `dispatch_blas_rebuild` (whose fresh
/// BLAS addresses the fill samples). The op stream + batch counts were
/// prepared in [`prepare_ptlas_params`]. Everything before the build is one
/// raw heap-kernel encoder (fills → hair/tess appends → finalize →
/// validate); the two staging copies stay wgpu ops on the shared
/// `RenderContext` encoder; the raw-VK build records into its own encoder —
/// the graph does one submit for the frame.
pub fn dispatch_ptlas(
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
    fns: Option<Res<ClusterExtensionFns>>,
    mut resources: Option<ResMut<Ptlas>>,
    scene_bind_group: Res<ClusterSceneBindGroup>,
    instances: Option<Res<InstanceManager>>,
    hair_instances: Option<Res<crate::hair::HairInstances>>,
    hair_write: Option<Res<crate::hair::ptlas_hair::HairPtlasWrite>>,
    tess_write: Option<Res<crate::geometry::tess_displace::TessPtlasWrite>>,
    (seam, propagate, tess_classify, transforms_col): (
        Option<Res<BindingSeam>>,
        Option<Res<crate::transform::TransformPropagate>>,
        Option<Res<crate::geometry::tess_classify::TessClassify>>,
        Option<Res<GpuColumn<TransformColumn>>>,
    ),
    (sharing, geometry_ids, instance_masks, material_ids, material_flags): (
        Option<Res<BlasSharing>>,
        Option<Res<GpuColumn<GeometryIdColumn>>>,
        Option<Res<GpuColumn<InstanceMaskColumn>>>,
        Option<Res<GpuColumn<MaterialColumn>>>,
        Res<MaterialTraversalFlags>,
    ),
    (node_slots, static_flags, partition_hints): (
        Option<Res<GpuColumn<NodeSlotColumn>>>,
        Option<Res<GpuColumn<Presence<StaticColumn>>>>,
        Option<Res<GpuColumn<crate::instance::PartitionColumn>>>,
    ),
    render_queue: Res<RenderQueue>,
    mut ctx: RenderContext,
) {
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
    drain_null_count(resources);
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
    // The cluster-scene heap mirror is the set-0 surface (and doubles as the
    // scene-ready signal).
    let (Some(_), Some(heap_slots), Some(seam)) = (
        scene_bind_group.bind_group.as_ref(),
        scene_bind_group.heap_slots.as_ref(),
        seam.as_deref(),
    ) else {
        tracing::debug!("ptlas.dispatch: build skipped (cluster-scene heap mirror cold)");
        return;
    };
    // The fill's set-1 inputs (the buffers the old fill bind group carried).
    let (
        Some(sharing),
        Some(geometry_ids),
        Some(instance_masks),
        Some(material_ids),
        Some(node_slots),
        Some(static_flags),
        Some(partition_hints),
        Some(transforms_col),
    ) = (
        sharing.as_deref(),
        geometry_ids.as_deref(),
        instance_masks.as_deref(),
        material_ids.as_deref(),
        node_slots.as_deref(),
        static_flags.as_deref(),
        partition_hints.as_deref(),
        transforms_col.as_deref(),
    )
    else {
        tracing::debug!("ptlas.dispatch: build skipped (instance columns cold)");
        return;
    };
    let (
        Some(write_slots),
        Some(active_to_slot),
        Some(previous_transforms),
        Some(material_flags_buf),
    ) = (
        resources.write_slots_cpu.buffer(),
        sharing.active_to_slot.buffer(),
        // `fill_incremental` compares current vs previous to detect moves.
        // `TransformColumn` is `KEEP_PREVIOUS`, so this is always `Some`.
        transforms_col.previous_buffer(),
        material_flags.buffer.buffer(),
    ) else {
        // The buffers are populated earlier in Prepare; None here means
        // no instances yet.
        tracing::debug!("ptlas.dispatch: build skipped (fill buffers cold)");
        return;
    };
    // Built lazily on the first ready frame: the mapping table bakes the
    // cluster-scene heap slots, which exist only once the mirror has run
    // (the slots are allocated once and rewritten in place, so the table
    // never goes stale).
    if resources.kernels.is_none() {
        let base = crate::gpu::rt_pipeline::cluster_heap_mappings(seam, heap_slots);
        let params_size = size_of::<PtlasFillParamsGpu>() as u32;
        let make = |entry: &str| {
            HeapKernel::new_with_mappings(
                seam,
                "ptlas_fill.slang",
                include_str!("ptlas_fill.slang"),
                entry,
                PTLAS_FILL_MODULES,
                &[],
                &[],
                &format!("ptlas_{entry}"),
                params_size,
                &base,
            )
        };
        let (Some(seed), Some(incremental), Some(finalize), Some(validate_kernel)) = (
            make("fill_seed"),
            make("fill_incremental"),
            make("finalize"),
            make("validate"),
        ) else {
            return;
        };
        resources.kernels = Some(PtlasKernels {
            seed,
            incremental,
            finalize,
            validate: validate_kernel,
            slots: KernelSlots::new(seam, 21),
        });
    }
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
    // Zero the heal counters every build (always on — they drive the
    // rebuild-until-clean warmup heal, not just SolariSettings::ptlas_validate).
    render_queue.write_buffer(
        &resources.validate_report,
        NULL_COUNT_WORD * 4,
        bytemuck::cast_slice(&[0u32, 0u32]),
    );
    let mut validate_recorded = false;

    // Everything before the AS build is ONE raw heap-kernel encoder — the
    // fills, the hair/tess appends, finalize, and validate, with a compute
    // w→r|w barrier at each producer→consumer step. Raw and wgpu work can't
    // share an encoder (the fork panics), so the two staging copies below
    // stay wgpu ops on the shared context encoder, ordered after this
    // encoder by the `add_command_buffer` splice.
    let kernels = resources.kernels.as_ref().unwrap();
    let table = ptlas_slot_table(
        seam,
        &kernels.slots,
        resources,
        sharing,
        write_slots,
        active_to_slot,
        previous_transforms,
        material_flags_buf,
        (
            geometry_ids.buffer(),
            instance_masks.buffer(),
            material_ids.buffer(),
            node_slots.buffer(),
            static_flags.buffer(),
            partition_hints.buffer(),
        ),
    );
    // Each entry's push blob: the shared fill params with the entry's own
    // dispatch split stamped in, plus the slot-table subset surviving in
    // that entry's SPIR-V.
    let blob = |kernel: &HeapKernel, label: &str, groups_x: u32| {
        let mut params = resources.fill_params;
        params.groups_x = groups_x;
        let named: Vec<(&str, u32)> = table
            .iter()
            .filter(|(name, _)| kernel.bindings.iter().any(|(n, _)| n == name))
            .copied()
            .collect();
        kernel.push_blob(label, bytemuck::bytes_of(&params), &named)
    };
    let seed_groups = crate::ecs_gpu::linear_dispatch(resources.cpu_count.div_ceil(64));
    let incremental_groups = crate::ecs_gpu::linear_dispatch(active_count.div_ceil(64));
    let max_records = resources.cpu_count + active_count + hair_count + tess_count;
    let validate_groups = crate::ecs_gpu::linear_dispatch(max_records.max(1).div_ceil(64));
    let seed_blob =
        (resources.cpu_count > 0).then(|| blob(&kernels.seed, "ptlas_fill_seed", seed_groups.0));
    let incremental_blob =
        blob(&kernels.incremental, "ptlas_fill_incremental", incremental_groups.0);
    let finalize_blob = blob(&kernels.finalize, "ptlas_finalize", 1);
    // Debug: scan the final WRITE stream, null + report corrupt BLAS
    // addresses before the raw build dereferences them.
    let validate_blob =
        validate.then(|| blob(&kernels.validate, "ptlas_validate", validate_groups.0));

    // Hair + tessellation: append their instances to the WRITE stream (same
    // `write_count`), after the cluster movers and before `finalize`
    // publishes the count.
    let hair_append = if hair_count > 0 {
        match (
            hair_write.as_ref(),
            propagate.as_ref(),
            hair_instances.as_ref().and_then(|h| h.buffer.buffer()),
        ) {
            (Some(hair_write), Some(propagate), Some(hair_buffer)) => Some((
                hair_write,
                hair_write.kernel.push_blob(
                    "ptlas_hair_write",
                    bytemuck::bytes_of(&hair_write.params),
                    &[
                        ("hair_instances", hair_write.slots.buffer(seam, 0, hair_buffer)),
                        ("write_count", hair_write.slots.buffer(seam, 1, &resources.write_count)),
                        (
                            "write_data",
                            hair_write.slots.buffer(seam, 2, &resources.write_data.wgpu_buffer),
                        ),
                        ("world", hair_write.slots.buffer(seam, 3, propagate.current_world())),
                    ],
                ),
            )),
            _ => None,
        }
    } else {
        None
    };
    let tess_append = match tess_write.as_ref() {
        Some(tw) if tw.tess_count > 0 => match (
            tw.instances.buffer(),
            tess_classify.as_ref().and_then(|c| c.blas_addresses.as_ref()),
        ) {
            (Some(instances_buf), Some(blas_addresses)) => Some((
                tw,
                tw.kernel.push_blob(
                    "tess_ptlas_write",
                    bytemuck::bytes_of(&tw.params),
                    &[
                        ("write_count", tw.slots.buffer(seam, 0, &resources.write_count)),
                        (
                            "write_data",
                            tw.slots.buffer(seam, 1, &resources.write_data.wgpu_buffer),
                        ),
                        ("instances", tw.slots.buffer(seam, 2, instances_buf)),
                        ("blas_addresses", tw.slots.buffer(seam, 3, blas_addresses)),
                        ("transforms", tw.slots.buffer(seam, 4, transforms_col.buffer())),
                    ],
                ),
            )),
            _ => None,
        },
        _ => None,
    };

    let mut fill_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("ptlas.fill"),
    });
    // SAFETY: Vulkan backend; the slots reference live heap descriptors; the
    // barriers bracket each step against its producers, and the post-barrier
    // additionally covers the staging copies' transfer reads (raw dispatches
    // are invisible to wgpu's tracking).
    unsafe {
        fill_encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &resources.raw_device;
            let step = [vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER,
                )
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)];
            let step_dep = vk::DependencyInfo::default().memory_barriers(&step);
            seam.bind_heaps(cb);
            // Upstream compute (blas sharing / commit_built) + the CPU seeds
            // (write_count / write_slots / src_infos transfers) -> our reads.
            dev.cmd_pipeline_barrier2(cb, &step_dep);
            if let Some(seed_blob) = seed_blob.as_ref() {
                seam.push_data(cb, seed_blob);
                dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernels.seed.pipeline);
                let (gx, gy, gz) = seed_groups;
                dev.cmd_dispatch(cb, gx, gy, gz);
                // Seed's epoch stamps + record writes -> incremental's reads.
                dev.cmd_pipeline_barrier2(cb, &step_dep);
            }
            seam.push_data(cb, &incremental_blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernels.incremental.pipeline);
            let (gx, gy, gz) = incremental_groups;
            dev.cmd_dispatch(cb, gx, gy, gz);
            // The cluster fills' record/count writes -> the appends / finalize.
            dev.cmd_pipeline_barrier2(cb, &step_dep);
            if let Some((hair_write, blob)) = hair_append.as_ref() {
                seam.push_data(cb, blob);
                dev.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    hair_write.kernel.pipeline,
                );
                let (gx, gy, gz) = hair_write.groups;
                dev.cmd_dispatch(cb, gx, gy, gz);
                // Hair's record/count writes -> the tess append + finalize.
                dev.cmd_pipeline_barrier2(cb, &step_dep);
            }
            if let Some((tw, blob)) = tess_append.as_ref() {
                seam.push_data(cb, blob);
                dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, tw.kernel.pipeline);
                dev.cmd_dispatch(cb, tw.tess_count.div_ceil(64), 1, 1);
                // Tess's record writes -> `finalize`'s count read.
                dev.cmd_pipeline_barrier2(cb, &step_dep);
            }
            seam.push_data(cb, &finalize_blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernels.finalize.pipeline);
            dev.cmd_dispatch(cb, 1, 1, 1);
            if let Some(validate_blob) = validate_blob.as_ref() {
                // Finalize's count publish -> validate's record scan.
                dev.cmd_pipeline_barrier2(cb, &step_dep);
                seam.push_data(cb, validate_blob);
                dev.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    kernels.validate.pipeline,
                );
                let (gx, gy, gz) = validate_groups;
                dev.cmd_dispatch(cb, gx, gy, gz);
            }
            // Our record/count/report writes -> the staging copies' transfer
            // reads and downstream compute; the AS build's own seam covers its
            // build-input reads.
            let post = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER,
                )
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_READ
                        | vk::AccessFlags2::SHADER_WRITE
                        | vk::AccessFlags2::TRANSFER_READ,
                )];
            dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&post));
        });
    }
    ctx.add_command_buffer(fill_encoder.finish());

    // The two staging copies stay wgpu ops (their dst buffers are wgpu
    // `map_async` targets — a raw copy would bypass wgpu's zero-init/usage
    // tracking), recorded on the context encoder AFTER the raw encoder was
    // spliced so they read the validate/heal words it wrote.
    {
        let encoder = ctx.command_encoder();
        if validate_blob.is_some() {
            encoder.copy_buffer_to_buffer(
                &resources.validate_report,
                0,
                &resources.validate_staging,
                0,
                VALIDATE_REPORT_WORDS * 4,
            );
            validate_recorded = true;
        }
        // Rebuild-until-clean: pull this build's null-AS record count (always on;
        // one 8-byte copy). Skipped while a previous readback is still in flight —
        // that latency is the retry pacing.
        if resources.nulls_phase == 0 {
            encoder.copy_buffer_to_buffer(
                &resources.validate_report,
                NULL_COUNT_WORD * 4,
                &resources.nulls_staging,
                0,
                8,
            );
            resources.nulls_phase = 1;
        }
    }
    resources.validate_in_flight = validate_recorded;

    // Declared access for the raw build — SolariSettings::validate checks every range
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
        // PRE-build seam: the build sees the fill-compute's freshly-written
        // WriteInstanceData records, and the previous build's `src` PTLAS bytes.
        crate::gpu::extension::cmd_as_seam(
            &mut build_encoder,
            &render_device,
            AsSeams::COMPUTE_TO_BUILD_INPUT | AsSeams::BUILD_TO_BUILD_INPUT,
        );
        crate::gpu::extension::cmd_build_partitioned_acceleration_structures(
            &mut build_encoder,
            &fns,
            &build_info,
        );
        // POST-build seam: the trace sees fresh PTLAS contents. Without it the
        // trace races the build and reads an empty AS → every ray misses.
        crate::gpu::extension::cmd_as_seam(&mut build_encoder, &render_device, AsSeams::BUILD_TO_TRACE);
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
