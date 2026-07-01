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

use bevy_ecs::hierarchy::{ChildOf, Children};
use bevy_ecs::prelude::*;
use bevy_ecs::query::Has;
use bevy_math::DVec3;
use bevy_render::render_resource::PipelineCache;
use bevy_render::{Extract, MainWorld};
use bevy_tasks::ComputeTaskPool;
use bevy_transform::components::{GlobalTransform, Transform};
use bevy_utils::Parallel;
use bytemuck::{Pod, Zeroable};

use bevy_render::renderer::{RenderDevice, RenderQueue};

use crate::ecs_gpu::{push_record, GpuColumn, GpuSlot};

use super::df64::Df64Vec3;
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

/// A node's **absolute world translation in double-single (`df64`) precision** — solari's
/// native floating-origin coordinate, the GPU-table successor to `big_space`'s integer
/// `CellCoord` (Aevyrie, MIT/Apache; used as the reference algorithm, credited). Where a
/// cell split a large position into `(integer cell, f32 residual)`, this carries the full
/// `f64` position and the GPU emits the small origin-relative `f32` by subtracting the
/// origin node's `df64` world — the primary camera's own `SolariFrameWorld`, read on the GPU
/// (see `transform_propagate.wgsl` + [`SolariOriginSlot`](super::SolariOriginSlot)).
/// No `cell_edge` knob, no `i32` range ceiling, no recenter — the camera sits at 0 by
/// construction because the origin *is* its own `df64` world.
///
/// The value is split into `hi + lo` `f32` lanes on extract ([`Df64Vec3`]) and stored in the
/// `frame_world` column. Author it on a frame body (a [`SolariFrame`], so its subtree
/// re-walks when the world moves); rotation/scale still come from the node's `Transform`
/// (only the translation needs `df64`). A GPU pass may also write this column directly —
/// e.g. an orbital solver computing a planet's world each frame — instead of this component.
/// Absent → the node adds no offset (world = local composition), exactly as with no origin.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct SolariFrameWorld {
    /// Absolute world-space translation of the frame (metres), full `f64` precision.
    pub world: DVec3,
}

impl SolariFrameWorld {
    /// A frame world from an absolute `f64` position.
    #[inline]
    pub fn new(world: DVec3) -> Self {
        Self { world }
    }

    /// The flat `[hi.xyz, lo.xyz, has_frame, pad]` `frame_world` column payload.
    #[inline]
    fn to_column(self) -> [f32; 8] {
        let c = Df64Vec3::from_dvec3(self.world).to_columns();
        [c[0], c[1], c[2], c[3], c[4], c[5], 1.0, 0.0]
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
            // The owning entity's bits (`Entity::to_bits` as `[lo, hi]`). The readback
            // gather stamps it into each record so the CPU writeback resolves the entity
            // directly and writes its `GlobalTransform`. Entity-keyed identity makes the
            // readback ABA-proof: a recycled slot's new occupant has a different entity,
            // and a despawned occupant's `get_mut` simply fails — no stale splat.
            NodeEntityColumn => entity: [u32; 2] = "transform.entity",
            // Absolute world translation in double-single precision + presence flag,
            // `[hi.x, hi.y, hi.z, lo.x, lo.y, lo.z, has_frame, pad]`, mirrored from
            // [`SolariFrameWorld`] (or written by a GPU pass). `has_frame == 0` (no component)
            // → the node adds no floating-origin offset. The propagate shader subtracts the
            // `df64` origin from this to emit the origin-relative `f32` translation.
            FrameWorldColumn => frame_world: [f32; 8] = "transform.frame_world",
        }
    }
}

/// A **reference frame**: a parent entity whose own `Transform` (rotation / scale /
/// translation) plus an optional [`SolariFrameWorld`] (its big `df64` offset) defines a
/// coordinate frame that its children are expressed **relative to**. Children carry small
/// frame-local `Transform`s and **no** frame world — the GPU ancestor-walk composes them through
/// the frame, so they inherit its orientation and offset for free. This is solari's analog
/// of `big_space`'s `Grid` (credited): a ship, a station, a planet, a surface tile, or — once
/// PTLAS partition routing lands — one co-resident *world* (each frame → one PTLAS partition,
/// the frame's big **translation** riding the per-partition translation while its rotation/scale
/// fold into the children's per-instance matrices on the free walk).
///
/// **Why it's a distinct marker (not just a celled parent):** the change-driven propagate
/// re-walks only nodes whose *own* `local` changed, so a frame that moves would leave its
/// descendants with a **stale world** (they didn't move relative to the frame). Tagging a
/// parent `SolariFrame` opts its subtree into a **descendant re-walk** when the frame moves —
/// including `TransformStatic` static-local children the change filter otherwise skips. Only
/// tag parents whose motion must drive their children on the RT path; a continuously rotating
/// frame re-walks its subtree every frame (intrinsic — its geometry *is* moving in origin space).
///
/// Authoring rule: put the big offset / orientation on the **frame**, keep children frame-local
/// with no [`SolariFrameWorld`] (one frame-world node per `ChildOf` chain — two would subtract
/// the origin twice). Nest frames (planet → tile) to keep each child's frame-local offset small
/// so the rotation fold stays f32-precise.
#[derive(Component, Default, Clone, Copy, Debug)]
pub struct SolariFrame;

/// A **GPU-driven reference frame**: like [`SolariFrame`], but its world is written on the
/// GPU each frame (into the [`FrameWorldColumn`] — e.g. by an orbital-mechanics compute pass
/// computing a planet's `df64` world from its elements + time), with **no CPU-side change**.
///
/// The change-driven propagate dispatches only nodes whose `local` changed on the CPU, so a
/// GPU-moved frame — and its whole subtree, which composes through it — would freeze. Tagging
/// it `SolariGpuFrame` opts the frame **and its descendants** into an **unconditional re-walk
/// every frame**: the extract seeds the frame itself (so its own `local` lands in the dispatch,
/// recomputing its origin-relative world from the freshly GPU-written `frame_world`) and walks
/// its current children (so streamed-in LOD tiles are picked up the frame after they spawn).
///
/// Use this **instead of** [`SolariFrame`] when the motion comes from the GPU rather than a CPU
/// `Transform`. It needs no [`SolariFrameWorld`] component — the GPU writes the column directly —
/// though authoring one seeds the frame's initial world. Cost is one subtree re-walk per frame,
/// the same a continuously-spinning [`SolariFrame`] pays; the GPU-native slot-list optimization
/// (re-dispatch without re-uploading unchanged locals) is a later step.
#[derive(Component, Default, Clone, Copy, Debug)]
pub struct SolariGpuFrame;

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

/// Presence column tracking [`TransformStatic`] over the transform table's nodes
/// — the GPU flag the PTLAS fill reads (via an instance's node slot) to route a
/// static instance to a spatial-grid partition and a mover to the global
/// partition. Observer-fed (zero per-frame cost); the flag buffer is node-slot
/// indexed. Read it via `Res<GpuColumn<Presence<StaticColumn>>>`.
pub struct StaticColumn;
impl crate::ecs_gpu::GpuPresenceColumn for StaticColumn {
    type SlotTable = TransformGraph;
    type Marker = TransformStatic;
    const LABEL: &'static str = "transform.static";
}

/// First-sight queue for entities **born** static. The change-driven extract excludes
/// `TransformStatic` at archetype granularity (the per-frame prune), so an entity tagged
/// *before* its first extract would never scatter its `local`/`parent`/`cell` and render
/// at the origin. The [`enqueue_static_first_sight`] observer pushes here on tag-add;
/// [`extract_transform_graph`] does the one-time upload (merged into its single local
/// write), then [`clear_static_first_sight`] drains the queue — both under the same
/// cold-start gate, so events accumulate (never lost) until the pipelines compile.
#[derive(Resource, Default)]
pub struct StaticFirstSightQueue {
    entities: Vec<Entity>,
}

/// Observer: a `TransformStatic` was added → queue the entity for its one-time first-sight
/// upload (idempotent — re-uploading an already-extracted static writes identical records).
pub fn enqueue_static_first_sight(
    add: On<Add, TransformStatic>,
    mut queue: ResMut<StaticFirstSightQueue>,
) {
    queue.entities.push(add.entity);
}

/// Observer: a node's slot was assigned (`GpuSlot` added) → queue its one-time first-sight upload.
/// Main-world observer-fed, so it fires deterministically when the slot lands — unlike the
/// change-driven extract, which can miss the cross-world `is_added` edge and never upload `local`.
/// Idempotent with [`enqueue_static_first_sight`] (a born-static node is queued by both).
pub fn enqueue_node_first_sight(
    add: On<Add, GpuSlot<TransformGraph>>,
    mut queue: ResMut<StaticFirstSightQueue>,
) {
    queue.entities.push(add.entity);
}

/// `ExtractSchedule` (gated like the extract, ordered after it): empty the queue the extract
/// just consumed. The main world is stalled during extract, so no observer can enqueue
/// between the read and this clear.
pub fn clear_static_first_sight(mut main_world: ResMut<MainWorld>) {
    if let Some(mut queue) = main_world.get_resource_mut::<StaticFirstSightQueue>() {
        queue.entities.clear();
    }
}

/// All transform columns' scatter pipelines compiled — the same cold-start gate the macro
/// puts on [`extract_transform_graph`], reused to hold the queue clear back in lockstep.
pub fn transform_columns_ready(
    local: Res<GpuColumn<LocalColumn>>,
    parent: Res<GpuColumn<ParentColumn>>,
    no_readback: Res<GpuColumn<NoReadbackColumn>>,
    entity: Res<GpuColumn<NodeEntityColumn>>,
    frame_world: Res<GpuColumn<FrameWorldColumn>>,
    cache: Res<PipelineCache>,
) -> bool {
    local.scatter_pipeline_ready(&cache)
        && parent.scatter_pipeline_ready(&cache)
        && no_readback.scatter_pipeline_ready(&cache)
        && entity.scatter_pipeline_ready(&cache)
        && frame_world.scatter_pipeline_ready(&cache)
}

/// Spatial entities whose local transform or parentage changed (or just appeared).
/// `&GpuSlot<TransformGraph>` in the fetch means only already-slotted entities are
/// seen — true the same frame, since the assign system runs in `PostUpdate`.
///
/// `Without<TransformStatic>` is on the **whole** filter (not a sub-branch), so the
/// static bulk is excluded at *archetype* granularity — the query never visits
/// those entities. (Skipping only the tick *reads* didn't help: the cost is the
/// per-entity visit over 2M, not the comparison.) An entity tagged *after* its first
/// extract was already scattered while still a mover; one tagged *before* (born static)
/// is caught instead by the [`StaticFirstSightQueue`] path below, so `TransformStatic`
/// is safe to add at spawn.
type TransformChangeFilter = (
    With<GlobalTransform>,
    Without<TransformStatic>,
    Or<(
        // First sight: the slot is inserted via deferred commands, so it can land a frame after
        // the `Changed<Transform>` edge went stale — key off the slot so the node still uploads.
        Added<GpuSlot<TransformGraph>>,
        Changed<Transform>,
        Changed<ChildOf>,
        Changed<SolariFrameWorld>,
    )>,
);

/// Thread-local `(local, parent)` delta buffers for the parallel extract; merged
/// into the table columns after the `par_iter`. `Parallel` keeps each thread's
/// `Vec` capacity across frames, so steady state never reallocates.
#[derive(Default)]
pub struct TransformDeltaBuf {
    local: Vec<u32>,
    parent: Vec<u32>,
    no_readback: Vec<u32>,
    entity: Vec<u32>,
    frame_world: Vec<u32>,
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
    #[inline]
    fn push_entity(&mut self, slot: u32, entity_bits: [u32; 2]) {
        push_record(&mut self.entity, slot, entity_bits);
    }
    #[inline]
    fn push_frame_world(&mut self, slot: u32, frame_world: [f32; 8]) {
        push_record(&mut self.frame_world, slot, frame_world);
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
pub fn extract_transform_graph(
    members: Extract<
        Query<
            (
                Entity,
                Ref<Transform>,
                Option<Ref<ChildOf>>,
                Option<Ref<SolariFrameWorld>>,
                Ref<GpuSlot<TransformGraph>>,
                Has<NoGpuGlobalTransformReadback>,
            ),
            TransformChangeFilter,
        >,
    >,
    nodes: Extract<Query<&GpuSlot<TransformGraph>>>,
    // Frames whose pose changed this frame — their subtree must re-walk (below).
    moved_frames: Extract<
        Query<
            Entity,
            (
                With<SolariFrame>,
                Or<(
                    Changed<Transform>,
                    Changed<SolariFrameWorld>,
                    Changed<ChildOf>,
                )>,
            ),
        >,
    >,
    // GPU-driven frames: re-walked unconditionally every frame (their world is written on
    // the GPU with no CPU change, so the change filter never dispatches them or their subtree).
    gpu_frames: Extract<Query<Entity, With<SolariGpuFrame>>>,
    // Hierarchy + per-descendant (local, slot) for the moved-frame subtree re-walk.
    children_q: Extract<Query<&Children>>,
    frame_descendants: Extract<Query<(&Transform, &GpuSlot<TransformGraph>)>>,
    marker_added: Extract<Query<&GpuSlot<TransformGraph>, Added<NoGpuGlobalTransformReadback>>>,
    mut marker_removed: Extract<RemovedComponents<NoGpuGlobalTransformReadback>>,
    // Born-static first sight: entities the change query skips (archetype prune) get their one
    // upload here. `static_data` is unfiltered so it can fetch any queued entity by id.
    static_queue: Extract<Res<StaticFirstSightQueue>>,
    static_data: Extract<
        Query<(
            &Transform,
            Option<&ChildOf>,
            Option<&SolariFrameWorld>,
            &GpuSlot<TransformGraph>,
            Has<NoGpuGlobalTransformReadback>,
        )>,
    >,
    mut table: ResMut<TransformGraph>,
    local_column: Option<ResMut<GpuColumn<LocalColumn>>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut queues: Local<Parallel<TransformDeltaBuf>>,
) {
    members.par_iter().for_each_init(
        || queues.borrow_local_mut(),
        |buf, (entity, transform, child_of, frame_world, slot, no_cpu_global)| {
            // First sight = the frame the slot landed; drives the one-time uploads below. Keyed on
            // the slot (not `transform.is_added()`) so a node whose slot arrives after its
            // Transform-change edge went stale still uploads exactly once.
            let first = slot.is_added();
            let slot = slot.index();
            // df64 world translation + presence flag. Pushed at first sight (every slot
            // initialized — absent → all-zero → no offset) and whenever `SolariFrameWorld`
            // changes. The propagate pass subtracts the df64 origin from this.
            let frame_world_changed = frame_world.as_ref().is_some_and(Ref::is_changed);
            if first || frame_world_changed {
                let fw = match &frame_world {
                    Some(f) => f.to_column(),
                    None => [0.0; 8],
                };
                buf.push_frame_world(slot, fw);
            }
            // Raw TRS — re-pushed whenever the node moves, and once at first sight (so a node
            // whose Transform-change edge was consumed before its slot existed still uploads).
            // Also re-pushed when only `frame_world` changed: `load_local` ignores a frame's
            // local translation (it uses the df64 world), but the node must still land in this
            // frame's dispatch to recompute its origin-relative world.
            if first || transform.is_changed() || frame_world_changed {
                buf.push_local(slot, LocalTRS::from_transform(&transform));
            }
            // Readback opt-out flag + owning-entity bits: first sight only — neither changes
            // over an occupant's lifetime, so movers never re-send them. A reused slot is
            // "first seen" by its new occupant (its `GpuSlot` is freshly `Added`), overwriting
            // the previous occupant's. Later marker adds/removes are caught by the passes below.
            if first {
                buf.push_no_readback(slot, no_cpu_global as u32);
                let bits = entity.to_bits();
                buf.push_entity(slot, [bits as u32, (bits >> 32) as u32]);
            }
            // Parent slot: at first sight and on reparent. Roots (no `ChildOf`) → `ROOT_PARENT`.
            let parent_changed = child_of.as_ref().is_some_and(Ref::is_changed);
            if first || parent_changed {
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
    // `parent` / `no_readback` / `entity` are tiny (reparent / first-sight only) —
    // serial.
    for buf in queues.iter_mut() {
        table.parent.append(&mut buf.parent);
        table.no_readback.append(&mut buf.no_readback);
        table.entity.append(&mut buf.entity);
        table.frame_world.append(&mut buf.frame_world);
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

    // `local` is the big delta (one record per mover): merging the thread-local
    // buffers through a CPU Vec would memcpy every record twice, so write them
    // straight into the queue's staging memory for the column's delta buffer —
    // disjoint chunks, copied in parallel across the compute pool. `table.local`
    // stays empty, so `prepare_column<LocalColumn>` sees no records and leaves
    // the pending count set here untouched. Extract is gated on scatter-pipeline
    // readiness (see `gpu_table!`), so the delta written here is always consumed.
    let Some(mut local_column) = local_column else {
        return;
    };

    // Moved-frame subtree re-walk. The change-driven propagate recomputes only
    // nodes whose *own* `local` changed; a frame's descendants didn't move
    // relative to the frame, so a rotating/translating frame would leave them
    // with a stale world (this includes `TransformStatic` static-local children
    // the change filter never visits). For each moved `SolariFrame`, re-push every
    // descendant's (unchanged) `local` so its slot lands in this frame's dispatch
    // and the ancestor walk recomposes it through the moved frame. Serial — frames
    // are few; a continuously spinning frame pays its subtree every frame (the
    // intrinsic cost of geometry that is genuinely moving in origin space).
    let mut frame_subtree: Vec<u32> = Vec::new();
    if !moved_frames.is_empty() || !gpu_frames.is_empty() {
        let mut stack: Vec<Entity> = Vec::new();
        // CPU-moved frames: the frame's own `local` was already pushed by the par_iter
        // (its `Transform` changed), so seed only its children — re-walk descendants.
        for frame in &moved_frames {
            if let Ok(children) = children_q.get(frame) {
                stack.extend(children.iter());
            }
        }
        // GPU-driven frames: the frame's world is written on the GPU (frame_world column) with
        // no CPU `Transform` change, so the par_iter never dispatched it. Seed the frame ITSELF
        // so the walk pushes its own `local` (dispatching it to recompute its origin-relative
        // world from the fresh GPU-written frame_world) AND every descendant, every frame.
        for frame in &gpu_frames {
            stack.push(frame);
        }
        while let Some(entity) = stack.pop() {
            if let Ok((transform, slot)) = frame_descendants.get(entity) {
                push_record(
                    &mut frame_subtree,
                    slot.index(),
                    LocalTRS::from_transform(transform),
                );
            }
            if let Ok(children) = children_q.get(entity) {
                stack.extend(children.iter());
            }
        }
    }

    // Born-static first sight: entities tagged `TransformStatic` before their first extract are
    // skipped by the change query (archetype prune), so do their one upload here. Idempotent for
    // statics tagged late (re-scatters identical records). Merged into the single `local` write.
    let mut static_local: Vec<u32> = Vec::new();
    for &entity in &static_queue.entities {
        let Ok((transform, child_of, frame_world, slot, no_cpu_global)) = static_data.get(entity)
        else {
            continue; // despawned / lost its slot before we ran
        };
        let slot = slot.index();
        push_record(&mut static_local, slot, LocalTRS::from_transform(transform));
        let fw = match frame_world {
            Some(f) => f.to_column(),
            None => [0.0; 8],
        };
        push_record(&mut table.frame_world, slot, fw);
        push_record(&mut table.no_readback, slot, no_cpu_global as u32);
        let bits = entity.to_bits();
        push_record(&mut table.entity, slot, [bits as u32, (bits >> 32) as u32]);
        let parent = match child_of {
            Some(child_of) => nodes
                .get(child_of.parent())
                .map(GpuSlot::index)
                .unwrap_or(ROOT_PARENT),
            None => ROOT_PARENT,
        };
        push_record(&mut table.parent, slot, parent);
    }

    let mut parts: Vec<&[u32]> = queues.iter_mut().map(|b| b.local.as_slice()).collect();
    if !frame_subtree.is_empty() {
        parts.push(frame_subtree.as_slice());
    }
    if !static_local.is_empty() {
        parts.push(static_local.as_slice());
    }
    let total: usize = parts.iter().map(|s| s.len()).sum();
    if total > 0 {
        local_column.write_delta_direct(total, &render_device, &render_queue, |dst| {
            // The chunks partition the staging view exactly: their lengths sum
            // to `total * 4`, so every byte is written before the upload.
            let mut dst = dst;
            ComputeTaskPool::get().scope(|scope| {
                for &src in &parts {
                    let (chunk, rest) = dst.split_at(size_of_val(src));
                    dst = rest;
                    scope.spawn(async move {
                        let mut chunk = chunk;
                        chunk.copy_from_slice(bytemuck::cast_slice(src));
                    });
                }
            });
        });
    }
    drop(parts);
    // Keep each thread-local's capacity for next frame (we copied, not moved).
    for buf in queues.iter_mut() {
        buf.local.clear();
    }
}
