//! The **instance change journal** — the CPU→GPU channel that feeds the
//! GPU-owned instance table.
//!
//! Instead of the CPU mirroring instance state and computing per-column deltas
//! against an assumed prior GPU state (which drifts on a regenerate burst), the
//! CPU emits an append-only stream of **absolute-state** records keyed by a
//! stable entity id: an `UPSERT` carries an entity's complete desired state, a
//! `REMOVE` names the entity. A GPU reconcile pass (see `ecs_gpu::reconcile`)
//! folds the journal into the authoritative table idempotently — so a regen is
//! just "remove the old set, add the new set" and converges with no drift.
//!
//! This module owns the record format and the upload ring; it is produced here
//! (appended from the instance lifecycle observers) and consumed by the reconcile
//! pass. Modeled on the `vk_partitioned_tlas` host→device instance-write channel:
//! the host stages writes and never reads device state back.

use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res},
};
use bevy_render::renderer::{RenderDevice, RenderQueue};
use bytemuck::{Pod, Zeroable};

use crate::gpu::allocator::{Allocator, SparseBuffer};

/// `op`: write/refresh this entity's complete state into the table.
pub const JOURNAL_OP_UPSERT: u32 = 0;
/// `op`: remove this entity from the table (clear its PTLAS presence).
pub const JOURNAL_OP_REMOVE: u32 = 1;

/// No explicit partition: the fill derives static→regular-0 / mover→global.
pub const PARTITION_HINT_NONE: u32 = 0xffff_ffff;

/// One absolute-state instance change. 48 bytes (12 plain `u32`s) — must match
/// `reconcile.wgsl`'s `JournalRecord`.
///
/// `node_key` is the **stable** transform-table slot
/// (`GpuSlot<TransformGraph>`), resolved to a world matrix GPU-side — never a
/// frozen transform.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct InstanceJournalRecord {
    /// The instance slot this record targets (CPU-allocated by the
    /// `InstanceManager`). The reconcile writes all of this slot's columns
    /// atomically from one record, so a reused slot is fully re-initialized in a
    /// single pass — no partial/stale column survives.
    pub slot: u32,
    /// PTLAS regular-partition hint ([`PARTITION_HINT_NONE`] = derive from the
    /// static flag). Spatially-tight ids (e.g. one per vegetation cell) give
    /// the driver per-cell BVHs + per-cell incremental rebuilds.
    pub partition_hint: u32,
    /// [`JOURNAL_OP_UPSERT`] or [`JOURNAL_OP_REMOVE`].
    pub op: u32,
    /// Dense `ClusterMesh` geometry id (stable, asset-resident).
    pub geometry_id: u32,
    /// Resolved GPU material slot ([`crate::material::MaterialSlots`]).
    pub material_id: u32,
    /// Transform-table node slot (`GpuSlot<TransformGraph>` index).
    pub node_key: u32,
    /// 8-bit `RenderLayers` cull mask.
    pub cull_mask: u32,
    /// bit0 = `TransformStatic` present. Reserved otherwise.
    pub flags: u32,
    // ── Per-geometry column payload (folded in so the reconcile writes every
    //    column from the record — no separate `geometry_id`-indexed GPU table). ──
    /// `GroupBaseColumn` value (mesh-pool group base).
    pub group_base: u32,
    /// `LodInputColumn`: cluster base in the mesh pool.
    pub cluster_base: u32,
    /// `LodInputColumn`: cluster count.
    pub cluster_count: u32,
    /// `LodInputColumn`: root group index.
    pub root_group: u32,
}

// `reconcile.wgsl`'s `JournalRecord` reads these fields by their fixed order (12
// contiguous `u32`s). Pin the size AND every field offset so a reorder here can't
// silently desync the shader — the const-offset assertion that makes the CPU writer
// and the GPU reconcile provably agree on the record layout.
const _: () = {
    use core::mem::offset_of;
    assert!(size_of::<InstanceJournalRecord>() == 48);
    assert!(offset_of!(InstanceJournalRecord, slot) == 0);
    assert!(offset_of!(InstanceJournalRecord, partition_hint) == 4);
    assert!(offset_of!(InstanceJournalRecord, op) == 8);
    assert!(offset_of!(InstanceJournalRecord, geometry_id) == 12);
    assert!(offset_of!(InstanceJournalRecord, material_id) == 16);
    assert!(offset_of!(InstanceJournalRecord, node_key) == 20);
    assert!(offset_of!(InstanceJournalRecord, cull_mask) == 24);
    assert!(offset_of!(InstanceJournalRecord, flags) == 28);
    assert!(offset_of!(InstanceJournalRecord, group_base) == 32);
    assert!(offset_of!(InstanceJournalRecord, cluster_base) == 36);
    assert!(offset_of!(InstanceJournalRecord, cluster_count) == 40);
    assert!(offset_of!(InstanceJournalRecord, root_group) == 44);
};

impl InstanceJournalRecord {
    /// A `REMOVE` for `slot` — only the slot + op matter (the reconcile clears the
    /// slot's PTLAS presence; columns are left for the next UPSERT to overwrite).
    #[inline]
    pub fn remove(slot: u32) -> Self {
        Self {
            slot,
            op: JOURNAL_OP_REMOVE,
            partition_hint: PARTITION_HINT_NONE,
            ..Default::default()
        }
    }

    /// A full absolute `UPSERT` for `slot` carrying its complete desired state.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn upsert(
        slot: u32,
        geometry_id: u32,
        material_id: u32,
        node_key: u32,
        cull_mask: u32,
        flags: u32,
        group_base: u32,
        cluster_base: u32,
        cluster_count: u32,
        root_group: u32,
        partition_hint: u32,
    ) -> Self {
        Self {
            slot,
            partition_hint,
            op: JOURNAL_OP_UPSERT,
            geometry_id,
            material_id,
            node_key,
            cull_mask,
            flags,
            group_base,
            cluster_base,
            cluster_count,
            root_group,
        }
    }
}

/// Virtual reservation for the journal ring — sized well above the worst-case
/// burst (a regenerate's remove-all + add-all is `2 × instance_count` records ≈
/// 41 MB at 640 k). Sparse-backed: pages commit grow-only, the handle/address
/// never change.
pub const JOURNAL_VIRTUAL_BYTES: u64 = 256 * 1024 * 1024;

/// Render-world resource: the per-frame instance change journal. CPU appends
/// absolute-state records into [`Self::staging`]; [`upload_rt_journal`] copies
/// them into the stable-address [`Self::buffer`] for the GPU reconcile, then
/// clears the staging for the next frame.
#[derive(Resource)]
pub struct RtJournal {
    /// Pending records, appended by the instance lifecycle observers / extract.
    /// **Retained until folded:** cleared by [`mark_folded`](Self::mark_folded) only
    /// after the reconcile actually dispatches over them (the retain-until-folded
    /// latch — the twin of [`GpuColumn`](crate::ecs_gpu::GpuColumn)'s `pending`). On
    /// the cold-start frames before the reconcile pipeline compiles, the records stay
    /// here and are re-uploaded each frame, so the initial binds are never lost
    /// (the reconcile is the columns' sole writer).
    staging: Vec<InstanceJournalRecord>,
    /// Stable-address ring the reconcile reads. Grown by commit; never freed.
    pub buffer: SparseBuffer,
    /// Bytes committed so far (grow-only).
    committed_bytes: u64,
    /// Record count uploaded this frame — the reconcile's dispatch bound.
    pub count: u32,
}

impl RtJournal {
    /// Append one record (UPSERT or REMOVE) for this frame.
    #[inline]
    pub fn push(&mut self, record: InstanceJournalRecord) {
        self.staging.push(record);
    }

    /// Drop the pending records — called by the reconcile **only after it has
    /// actually dispatched** over them (with a live pipeline + bind group), so a
    /// cold-pipeline frame keeps them live to retry next frame. Capacity is kept.
    #[inline]
    pub fn mark_folded(&mut self) {
        self.staging.clear();
    }
}

/// `RenderStartup`: allocate the journal ring + insert [`RtJournal`]. No-op when
/// the raw-VK [`Allocator`] is absent (downstream reconcile guards on presence).
pub fn init_rt_journal(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
    additional: Res<bevy_render::renderer::raw_vulkan_init::AdditionalVulkanFeatures>,
) {
    let Some(allocator) = allocator else {
        // Ordering tripwire (release-safe): this system is `.after(SolariSetup)`, so
        // whenever solari is *supported* the allocator already exists. A missing
        // allocator while the feature IS present means an init-ordering regression
        // (someone dropped the `.after`) — which silently kills the journal AND the
        // whole reconcile, leaving instance columns unwritten. Fail loud, not dark.
        // (Feature absent → solari legitimately disabled; init_allocator already warned.)
        if additional.has::<crate::gpu::extension::ClusterAccelerationStructureFeature>() {
            bevy_log::error!(
                "init_rt_journal ran before the raw-VK allocator despite solari being supported \
                 — RenderStartup ordering regression; restore `.after(SolariSetup)`. The GPU \
                 instance reconcile will be absent and instance columns won't be written."
            );
        }
        return;
    };
    let buffer = allocator.create_sparse_buffer(
        &render_device,
        ash::vk::BufferUsageFlags::STORAGE_BUFFER | ash::vk::BufferUsageFlags::TRANSFER_DST,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        JOURNAL_VIRTUAL_BYTES,
        "rt.journal",
    );
    commands.insert_resource(RtJournal {
        staging: Vec::new(),
        buffer,
        committed_bytes: 0,
        count: 0,
    });
}

/// `Render::Prepare`: upload the pending records into the stable-address ring and
/// publish the count. Always starts at offset 0 — the reconcile reads `[0, count)`.
///
/// The staging is **not** cleared here: [`RtJournal::mark_folded`] clears it from the
/// reconcile dispatch, only once the records have actually been folded. So a frame
/// whose reconcile pipeline isn't ready yet re-uploads the same records next frame
/// rather than dropping them (the retain-until-folded latch).
pub fn upload_rt_journal(
    journal: Option<bevy_ecs::system::ResMut<RtJournal>>,
    render_queue: Res<RenderQueue>,
) {
    let Some(mut journal) = journal else {
        return;
    };
    let count = journal.staging.len() as u32;
    journal.count = count;
    if count > 0 {
        let bytes = journal.staging.len() as u64 * size_of::<InstanceJournalRecord>() as u64;
        if bytes > journal.committed_bytes {
            journal.buffer.commit(0..bytes);
            journal.committed_bytes = bytes;
        }
        let staging = std::mem::take(&mut journal.staging);
        render_queue.write_buffer(journal.buffer.buffer(), 0, bytemuck::cast_slice(&staging));
        journal.staging = staging;
    }
}
