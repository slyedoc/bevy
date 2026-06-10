//! The transform table — every entity with a `GlobalTransform` mirrored to the
//! GPU as a node carrying its local `Transform` and parent node-slot. Declared
//! with [`gpu_table!`](crate::gpu_table): the resource, the `LocalColumn` /
//! `ParentColumn`, the slot index ([`GpuSlot<TransformGraph>`] — a free
//! array-indexed component read, no `EntityHashMap`), and `TransformTablePlugin`
//! are all generated. Only the [`extract_transform_graph`] below is hand-written
//! (the domain logic: which component maps to which column).
//!
//! The propagation pass ([`super::propagate`]) reads the two columns and
//! computes a world transform per node (an ancestor-walk over changed nodes).

use bevy_ecs::hierarchy::ChildOf;
use bevy_ecs::prelude::*;
use bevy_ecs::query::Has;
use bevy_render::Extract;
use bevy_tasks::ComputeTaskPool;
use bevy_transform::components::{GlobalTransform, Transform};
use bevy_utils::Parallel;
use bytemuck::{Pod, Zeroable};

use crate::ecs_gpu::{push_record, GpuSlot};

use super::readback::NoGpuGlobalTransformReadback;

/// Root sentinel in the `parent` column: no parent, so propagation takes
/// `world = local`. Must match `ROOT_PARENT` in `transform_propagate.wgsl`.
pub const ROOT_PARENT: u32 = u32::MAX;

/// Per-node **local** transform on the wire: raw TRS — 10 floats (`translation`
/// .xyz, `rotation` xyzw quat, `scale` .xyz) = 40 B, vs a packed `mat3x4`'s 48 B.
/// Stored as `[f32; 10]` (align 4) so it stays tight (a `glam::Quat` field would
/// force 16-byte alignment → 48 B, defeating the point). The propagate shader
/// builds the matrix from this during the ancestor walk — fewer CPU bytes to build
/// + upload, and no CPU `compute_affine`. The *world* output stays `mat3x4`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct LocalTRS {
    data: [f32; 10],
}

impl LocalTRS {
    #[inline]
    fn from_transform(t: &Transform) -> Self {
        let (tr, r, s) = (t.translation, t.rotation, t.scale);
        Self {
            data: [tr.x, tr.y, tr.z, r.x, r.y, r.z, r.w, s.x, s.y, s.z],
        }
    }
}

crate::gpu_table! {
    /// Render-world transform table: per-node `local` transform + `parent` slot.
    /// Slots are indexed by the generated `GpuSlot<TransformGraph>` component on
    /// every `GlobalTransform` entity (allocated `Added`-only, freed by observer).
    pub table TransformGraph as TransformTablePlugin {
        members: With<GlobalTransform>,
        extract: extract_transform_graph,
        columns {
            LocalColumn  => local:  LocalTRS = "transform.local",
            ParentColumn => parent: u32        = "transform.parent",
            // 1 if the node opts out of CPU `GlobalTransform` readback
            // (`NoGpuGlobalTransformReadback`). Read by the readback gather to skip it.
            NoReadbackColumn => no_readback: u32 = "transform.no_readback",
        }
    }
}

/// Opt **out** of per-frame transform extraction: this entity's local `Transform`
/// is scattered to the GPU table **once** (first sight) and then never re-scanned
/// for movement. Tag the known-static bulk (buildings, terrain, roads, props) so
/// the per-frame `Changed<Transform>` scan visits only things that actually move.
///
/// Default (no marker) = scanned every frame, so untagged content always renders
/// correctly — this is purely an opt-out optimization. A tagged entity that *does*
/// move will **not** update on the GPU; only tag things you know are static.
/// Derives `Default` so it works as a required component.
#[derive(Component, Default)]
pub struct TransformStatic;

/// Spatial entities whose local transform or parentage changed (or just appeared).
/// `&GpuSlot<TransformGraph>` in the fetch means only already-slotted entities are
/// seen — true the same frame, since the assign system runs in `PostUpdate`.
///
/// `Without<TransformStatic>` is on the **whole** filter (not a sub-branch), so the
/// static bulk is excluded at *archetype* granularity — the query never visits
/// those entities. (Skipping only the tick *reads* didn't help: the cost is the
/// per-entity visit over 2M, not the comparison.) First sight is covered by
/// `Changed<Transform>` including `is_added`: an entity is extracted the frame its
/// `Transform` appears, before [`TransformStatic`] is tagged (tagging lags spawn —
/// it runs after the mesh converts). A `TransformStatic` added *before* an entity's
/// first extract would leave it at identity, so tag only post-spawn.
type TransformChangeFilter = (
    With<GlobalTransform>,
    Without<TransformStatic>,
    Or<(Changed<Transform>, Changed<ChildOf>)>,
);

/// Thread-local `(local, parent)` delta buffers for the parallel extract; merged
/// into the table columns after the `par_iter`. `Parallel` keeps each thread's
/// `Vec` capacity across frames, so steady state never reallocates.
#[derive(Default)]
pub struct TransformDeltaBuf {
    local: Vec<u32>,
    parent: Vec<u32>,
    no_readback: Vec<u32>,
}

impl TransformDeltaBuf {
    #[inline]
    fn push_local(&mut self, slot: u32, local: LocalTRS) {
        push_record(&mut self.local, slot, local);
    }
    #[inline]
    fn push_parent(&mut self, slot: u32, parent: u32) {
        push_record(&mut self.parent, slot, parent);
    }
    #[inline]
    fn push_no_readback(&mut self, slot: u32, flag: u32) {
        push_record(&mut self.no_readback, slot, flag);
    }
}

/// `ExtractSchedule`: mirror each changed entity's local `Transform` + parent
/// node-slot into the table's column deltas. Parallel over the changed set (the
/// moving cars are ~120k at size 100): `par_iter` fills thread-local buffers,
/// then a serial merge appends them to the columns.
///
/// Each column is (re)scattered **only when its own source changed**:
/// - `local` on `Changed<Transform>` (the mover hot path — just an affine pack).
/// - `parent` only on `Changed<ChildOf>` / first sight. A parent slot is stable,
///   so movers skip the random-access `nodes.get` parent lookup entirely — that
///   lookup (latency-bound) was the dominant residual cost when done per-frame.
///
/// The GPU column buffers persist, so a column not re-scattered this frame keeps
/// last value. Roots (no `ChildOf`) get `ROOT_PARENT`, scattered once at first sight.
#[expect(
    unsafe_code,
    clippy::uninit_vec,
    reason = "the parallel disjoint `copy_from_slice`s fill every word of `local` \
              before the GPU scatter reads it (the scope blocks to completion); \
              `set_len` after `reserve` skips a zero-init that would negate the merge"
)]
pub fn extract_transform_graph(
    members: Extract<
        Query<
            (
                Ref<Transform>,
                Option<Ref<ChildOf>>,
                &GpuSlot<TransformGraph>,
                Has<NoGpuGlobalTransformReadback>,
            ),
            TransformChangeFilter,
        >,
    >,
    nodes: Extract<Query<&GpuSlot<TransformGraph>>>,
    marker_added: Extract<Query<&GpuSlot<TransformGraph>, Added<NoGpuGlobalTransformReadback>>>,
    mut marker_removed: Extract<RemovedComponents<NoGpuGlobalTransformReadback>>,
    mut table: ResMut<TransformGraph>,
    mut queues: Local<Parallel<TransformDeltaBuf>>,
) {
    members.par_iter().for_each_init(
        || queues.borrow_local_mut(),
        |buf, (transform, child_of, slot, no_cpu_global)| {
            let slot = slot.index();
            // `is_changed()` includes the frame the component was added.
            if transform.is_changed() {
                // Raw TRS — the propagate shader builds the matrix (no CPU pack).
                buf.push_local(slot, LocalTRS::from_transform(&transform));
                // The readback opt-out flag scatters at first sight only; later
                // marker adds/removes are caught by the dedicated queries below
                // (re-scattering it per move paid ~0.25 ms/frame in bevy_city
                // for a flag that almost never changes).
                if transform.is_added() {
                    buf.push_no_readback(slot, no_cpu_global as u32);
                }
            }
            let parent_changed = match &child_of {
                Some(child_of) => child_of.is_changed(),
                // A root scatters its `ROOT_PARENT` once, the frame it appears.
                None => transform.is_added(),
            };
            if parent_changed {
                let parent = match &child_of {
                    Some(child_of) => nodes
                        .get(child_of.parent())
                        .map(GpuSlot::index)
                        .unwrap_or(ROOT_PARENT),
                    None => ROOT_PARENT,
                };
                buf.push_parent(slot, parent);
            }
        },
    );
    // `parent` / `no_readback` are tiny (reparent / first-sight only) — serial.
    for buf in queues.iter_mut() {
        table.parent.append(&mut buf.parent);
        table.no_readback.append(&mut buf.no_readback);
    }

    // Readback opt-out flag changes after first sight. Removals first: an entity
    // whose marker was removed *and* re-added this frame still matches the
    // `Added` query, so the later `1` record wins. A removed event for an entity
    // that despawned misses the `nodes` lookup and is skipped (slot was freed).
    // Catching these here (not in the mover scan) also covers `TransformStatic`
    // entities, which the change filter never visits after first sight.
    for entity in marker_removed.read() {
        if let Ok(slot) = nodes.get(entity) {
            push_record(&mut table.no_readback, slot.index(), 0u32);
        }
    }
    for slot in marker_added.iter() {
        push_record(&mut table.no_readback, slot.index(), 1u32);
    }

    // `local` is the big delta (one record per mover). Instead of a serial memcpy
    // appending each thread-local buffer, pre-size the column and copy the buffers
    // into disjoint slices in parallel across the compute pool.
    let parts: Vec<&[u32]> = queues.iter_mut().map(|b| b.local.as_slice()).collect();
    let total: usize = parts.iter().map(|s| s.len()).sum();
    table.local.clear();
    table.local.reserve(total);
    // SAFETY: the disjoint `copy_from_slice`s below write exactly `total` words —
    // the chunks partition `0..total` (their lengths sum to `total`) — so the whole
    // new length is initialised before any read. `u32` needs no drop.
    unsafe {
        table.local.set_len(total);
    }
    let mut dst: &mut [u32] = table.local.as_mut_slice();
    ComputeTaskPool::get().scope(|scope| {
        for &src in &parts {
            let (chunk, rest) = dst.split_at_mut(src.len());
            dst = rest;
            scope.spawn(async move {
                chunk.copy_from_slice(src);
            });
        }
    });
    drop(parts);
    // Keep each thread-local's capacity for next frame (we copied, not moved).
    for buf in queues.iter_mut() {
        buf.local.clear();
    }
}
