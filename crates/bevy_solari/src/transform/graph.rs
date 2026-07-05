//! The transform table — every entity with a `GlobalTransform` mirrored to the
//! GPU as a node carrying its local `Transform` and parent node-slot. Declared
//! with [`gpu_table!`](crate::gpu_table): the resource, the local / `ParentColumn`
//! columns, the slot index ([`GpuSlot<TransformGraph>`] — a free array-indexed
//! component read, no `EntityHashMap`), and `TransformTablePlugin` are all
//! generated. Only the [`extract_transform_graph`] below is hand-written
//! (the domain logic: which component maps to which column).
//!
//! With the `transform_f64` cargo feature (which `bevy_solari` requires),
//! `Transform.translation` is a genuine `f64` vector, mirrored to the GPU as a
//! native-`f64` column ([`LocalTranslationColumn`]) — solari's floating-origin
//! coordinate, the successor to `big_space`'s integer `CellCoord` (Aevyrie,
//! MIT/Apache; used as the reference algorithm, credited). The propagation pass
//! ([`super::propagate`]) walks ancestors accumulating the translation in `f64`
//! (via `SHADER_F64`), producing each node's **absolute** world; the subtract
//! pass ([`super::subtract`]) then subtracts the camera's own absolute world so
//! the small origin-relative `f32` the acceleration structure is built from
//! survives at any magnitude. No `cell_edge` knob, no range ceiling, no
//! recenter, no per-entity opt-in — every `Transform` is precise.

use bevy_ecs::hierarchy::{ChildOf, Children};
use bevy_ecs::prelude::*;
use bevy_ecs::query::Has;
use bevy_render::render_resource::PipelineCache;
use bevy_render::{Extract, MainWorld};
use bevy_tasks::ComputeTaskPool;
use bevy_transform::components::{GlobalTransform, Transform};
use bevy_utils::Parallel;

use bevy_render::renderer::{RenderDevice, RenderQueue};

use crate::ecs_gpu::{push_record, GpuColumn, GpuSlot};

use super::readback::NoGpuGlobalTransformReadback;

/// Root sentinel in the `parent` column: no parent, so propagation takes
/// `world = local`. Must match `ROOT_PARENT` in `transform_propagate.wgsl`.
pub const ROOT_PARENT: u32 = u32::MAX;

/// Chain sentinel in the `first_child`/`next_sibling` columns: end of the sibling
/// list / childless node. Must match `NO_NODE` in `transform_frontier.wgsl`.
pub const NO_NODE: u32 = u32::MAX;

/// A node's local translation on the wire: 3×`f64` = 24 B, bound as `array<f64>`
/// by the propagate walk. Element `k` of node `n` sits at byte `24n + 8k` — always
/// 8-aligned, so the flat-`f64` view needs no padding.
type LocalTranslation = [f64; 3];

/// A node's local rotation + scale on the wire: quat `xyzw` then scale `xyz`,
/// 7×`f32` = 28 B. Rotation/scale never accumulate magnitude, so `f32` is exact
/// enough at any world position (`Transform`'s f64 rotation/scale narrow here).
type LocalRS = [f32; 7];

/// The two local records for a node, straight from its [`Transform`].
#[inline]
fn local_records(transform: &Transform) -> (LocalTranslation, LocalRS) {
    let t = transform.translation;
    let r = transform.rotation.as_quat();
    let s = transform.scale.as_vec3();
    ([t.x, t.y, t.z], [r.x, r.y, r.z, r.w, s.x, s.y, s.z])
}

crate::gpu_table! {
    /// Render-world transform table: per-node local transform + `parent` slot.
    /// Slots are indexed by the generated `GpuSlot<TransformGraph>` component on
    /// every `GlobalTransform` entity (allocated `Added`-only, freed by observer).
    pub table TransformGraph as TransformTablePlugin {
        members: With<GlobalTransform>,
        extract: extract_transform_graph,
        columns {
            // Native-f64 local translation. The two local columns are pushed in
            // lockstep (same slots, same order, same count), so this column's
            // delta doubles as the propagate/readback "changed nodes" list.
            LocalTranslationColumn => local_t: LocalTranslation = "transform.local_t",
            // f32 rotation quat + scale.
            LocalRSColumn => local_rs: LocalRS = "transform.local_rs",
            ParentColumn => parent: u32 = "transform.parent",
            // Downward topology for the GPU frontier expansion (`super::frontier`):
            // intrusive child list per node. Written at first sight and on
            // `Changed<Children>`; `NO_NODE` terminates. The frontier pass walks
            // these to re-walk a moved parent's descendants entirely GPU-side.
            FirstChildColumn  => first_child: u32 = "transform.first_child",
            NextSiblingColumn => next_sibling: u32 = "transform.next_sibling",
            // 1 if the node opts out of CPU `GlobalTransform` readback
            // (`NoGpuGlobalTransformReadback`). Read by the readback gather to skip it.
            NoReadbackColumn => no_readback: u32 = "transform.no_readback",
            // The owning entity's bits (`Entity::to_bits` as `[lo, hi]`). The readback
            // gather stamps it into each record so the CPU writeback resolves the entity
            // directly and writes its `GlobalTransform`. Entity-keyed identity makes the
            // readback ABA-proof: a recycled slot's new occupant has a different entity,
            // and a despawned occupant's `get_mut` simply fails — no stale splat.
            NodeEntityColumn => entity: [u32; 2] = "transform.entity",
        }
    }
}

/// A **GPU-moved node**: its motion comes from a GPU pass writing its `local`/world directly
/// (e.g. an orbital-mechanics compute pass), with **no CPU-side change**, so the change filter
/// never sees it. Tagged nodes are appended to the frontier **seed** every frame; the GPU
/// frontier then expands to their descendants. (The old per-frame CPU subtree re-push is gone.)
#[derive(Component, Default, Clone, Copy, Debug)]
pub struct SolariGpuFrame;

/// Render-world list of [`SolariGpuFrame`] node slots, refreshed each extract — the extra
/// frontier seeds beyond the changed-`local` delta (see `super::frontier`).
#[derive(Resource, Default)]
pub struct GpuFrameSeeds(pub Vec<u32>);

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
/// *before* its first extract would never scatter its `local`/`parent` and render
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
    local_t: Res<GpuColumn<LocalTranslationColumn>>,
    local_rs: Res<GpuColumn<LocalRSColumn>>,
    parent: Res<GpuColumn<ParentColumn>>,
    no_readback: Res<GpuColumn<NoReadbackColumn>>,
    entity: Res<GpuColumn<NodeEntityColumn>>,
    cache: Res<PipelineCache>,
) -> bool {
    local_t.scatter_pipeline_ready(&cache)
        && local_rs.scatter_pipeline_ready(&cache)
        && parent.scatter_pipeline_ready(&cache)
        && no_readback.scatter_pipeline_ready(&cache)
        && entity.scatter_pipeline_ready(&cache)
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
    )>,
);

/// Thread-local delta buffers for the parallel extract; merged into the table
/// columns after the `par_iter`. `Parallel` keeps each thread's `Vec` capacity
/// across frames, so steady state never reallocates.
#[derive(Default)]
pub struct TransformDeltaBuf {
    local_t: Vec<u32>,
    local_rs: Vec<u32>,
    parent: Vec<u32>,
    no_readback: Vec<u32>,
    entity: Vec<u32>,
    first_child: Vec<u32>,
    next_sibling: Vec<u32>,
}

impl TransformDeltaBuf {
    /// The two local columns are pushed in lockstep — same slot, both records —
    /// so their deltas stay congruent (the translation delta doubles as the
    /// changed-nodes list for the walk and the readback).
    #[inline]
    fn push_local(&mut self, slot: u32, transform: &Transform) {
        let (t, rs) = local_records(transform);
        push_record(&mut self.local_t, slot, t);
        push_record(&mut self.local_rs, slot, rs);
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
    fn push_chain(
        &mut self,
        parent_slot: u32,
        children: Option<&Children>,
        nodes: &Query<&GpuSlot<TransformGraph>>,
    ) {
        push_children_chain(&mut self.first_child, &mut self.next_sibling, parent_slot, children, nodes);
    }
}

/// Write a node's whole child chain: `first_child[parent]` plus each slotted child's
/// `next_sibling`, `NO_NODE`-terminated. Unslotted children (no `GlobalTransform`)
/// are skipped, keeping the chain closed over table nodes.
fn push_children_chain(
    first_child: &mut Vec<u32>,
    next_sibling: &mut Vec<u32>,
    parent_slot: u32,
    children: Option<&Children>,
    nodes: &Query<&GpuSlot<TransformGraph>>,
) {
    let mut head = NO_NODE;
    let mut prev: Option<u32> = None;
    for child in children.into_iter().flat_map(Children::iter) {
        let Ok(slot) = nodes.get(child) else { continue };
        let slot = slot.index();
        match prev {
            None => head = slot,
            Some(prev) => push_record(next_sibling, prev, slot),
        }
        prev = Some(slot);
    }
    if let Some(last) = prev {
        push_record(next_sibling, last, NO_NODE);
    }
    push_record(first_child, parent_slot, head);
}

/// `ExtractSchedule`: mirror each changed entity's local `Transform` + parent
/// node-slot into the table's column deltas. Parallel over the changed set (the
/// moving cars are ~120k at size 100): `par_iter` fills thread-local buffers,
/// then a serial merge appends them to the columns.
///
/// Each column is (re)scattered **only when its own source changed**:
/// - the local pair on `Changed<Transform>` (the mover hot path).
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
                Ref<GpuSlot<TransformGraph>>,
                Has<NoGpuGlobalTransformReadback>,
                Option<&Children>,
            ),
            TransformChangeFilter,
        >,
    >,
    nodes: Extract<Query<&GpuSlot<TransformGraph>>>,
    // Child-list maintenance for the GPU frontier's downward topology (chain rewrite
    // on `Changed<Children>`, chain clear on removal, gpu-frame seed refresh) —
    // tupled to stay under the system-param limit.
    (children_changed, mut children_removed, has_children, gpu_frames, mut gpu_seeds): (
        Extract<Query<(Ref<GpuSlot<TransformGraph>>, &Children), Changed<Children>>>,
        Extract<RemovedComponents<Children>>,
        Extract<Query<(), With<Children>>>,
        Extract<Query<&GpuSlot<TransformGraph>, With<SolariGpuFrame>>>,
        ResMut<GpuFrameSeeds>,
    ),
    marker_added: Extract<Query<&GpuSlot<TransformGraph>, Added<NoGpuGlobalTransformReadback>>>,
    mut marker_removed: Extract<RemovedComponents<NoGpuGlobalTransformReadback>>,
    // Born-static first sight: entities the change query skips (archetype prune) get their one
    // upload here. `static_data` is unfiltered so it can fetch any queued entity by id.
    static_queue: Extract<Res<StaticFirstSightQueue>>,
    static_data: Extract<
        Query<(
            &Transform,
            Option<&ChildOf>,
            &GpuSlot<TransformGraph>,
            Has<NoGpuGlobalTransformReadback>,
            Option<&Children>,
        )>,
    >,
    table: ResMut<TransformGraph>,
    local_t_column: Option<ResMut<GpuColumn<LocalTranslationColumn>>>,
    local_rs_column: Option<ResMut<GpuColumn<LocalRSColumn>>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut queues: Local<Parallel<TransformDeltaBuf>>,
) {
    // Plain `&mut` so disjoint column-`Vec` borrows work (`ResMut` derefs whole).
    let table = table.into_inner();
    members.par_iter().for_each_init(
        || queues.borrow_local_mut(),
        |buf, (entity, transform, child_of, slot, no_cpu_global, children)| {
            // First sight = the frame the slot landed; drives the one-time uploads below. Keyed on
            // the slot (not `transform.is_added()`) so a node whose slot arrives after its
            // Transform-change edge went stale still uploads exactly once.
            let first = slot.is_added();
            let slot = slot.index();
            // The local pair (f64 translation + f32 RS) — re-pushed whenever the node moves,
            // once at first sight, and on a bare reparent (the re-push seeds the GPU frontier,
            // which recomposes the node AND its descendants through the new ancestry).
            let parent_changed = child_of.as_ref().is_some_and(Ref::is_changed);
            if first || transform.is_changed() || parent_changed {
                buf.push_local(slot, &transform);
            }
            // Readback opt-out flag + owning-entity bits + the child chain: first sight only.
            // A reused slot is "first seen" by its new occupant (its `GpuSlot` is freshly
            // `Added`), overwriting the previous occupant's records. Later changes are caught
            // by the marker passes / `Changed<Children>` below.
            if first {
                buf.push_no_readback(slot, no_cpu_global as u32);
                let bits = entity.to_bits();
                buf.push_entity(slot, [bits as u32, (bits >> 32) as u32]);
                buf.push_chain(slot, children, &nodes);
            }
            // Parent slot: at first sight and on reparent. Roots (no `ChildOf`) → `ROOT_PARENT`.
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
    // `parent` / `no_readback` / `entity` / chain columns are tiny (reparent /
    // first-sight / child-list churn only) — serial.
    for buf in queues.iter_mut() {
        table.parent.append(&mut buf.parent);
        table.no_readback.append(&mut buf.no_readback);
        table.entity.append(&mut buf.entity);
        table.first_child.append(&mut buf.first_child);
        table.next_sibling.append(&mut buf.next_sibling);
    }

    // Child-list churn after first sight: rewrite the changed node's whole chain.
    // First-sight slots are skipped — the mover/static paths just wrote theirs, and
    // a duplicate record for one slot in a frame's delta is a scatter race.
    for (slot, children) in &children_changed {
        if slot.is_added() {
            continue;
        }
        push_children_chain(
            &mut table.first_child,
            &mut table.next_sibling,
            slot.index(),
            Some(children),
            &nodes,
        );
    }
    // Last child removed → `Children` is gone entirely. Skip if it was re-added the
    // same frame (the `Changed<Children>` pass above already wrote the new chain).
    for entity in children_removed.read() {
        if has_children.contains(entity) {
            continue;
        }
        if let Ok(slot) = nodes.get(entity) {
            push_record(&mut table.first_child, slot.index(), NO_NODE);
        }
    }
    // GPU-driven nodes seed the frontier every frame (their motion has no CPU edge).
    gpu_seeds.0.clear();
    gpu_seeds.0.extend(gpu_frames.iter().map(GpuSlot::index));

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

    // The local pair is the big delta (one record per mover per column): merging the
    // thread-local buffers through a CPU Vec would memcpy every record twice, so write
    // them straight into each column's staging memory for its delta buffer — disjoint
    // chunks, copied in parallel across the compute pool. `table.local_t`/`local_rs`
    // stay empty, so `prepare_column` sees no records and leaves the pending counts set
    // here untouched. Extract is gated on scatter-pipeline readiness (see `gpu_table!`),
    // so the delta written here is always consumed.
    let (Some(mut local_t_column), Some(mut local_rs_column)) = (local_t_column, local_rs_column)
    else {
        return;
    };

    // Born-static first sight: entities tagged `TransformStatic` before their first extract are
    // skipped by the change query (archetype prune), so do their one upload here. Idempotent for
    // statics tagged late (re-scatters identical records). Merged into the single local write.
    let mut static_t: Vec<u32> = Vec::new();
    let mut static_rs: Vec<u32> = Vec::new();
    for &entity in &static_queue.entities {
        let Ok((transform, child_of, slot, no_cpu_global, children)) = static_data.get(entity)
        else {
            continue; // despawned / lost its slot before we ran
        };
        let slot = slot.index();
        let (t, rs) = local_records(transform);
        push_record(&mut static_t, slot, t);
        push_record(&mut static_rs, slot, rs);
        push_record(&mut table.no_readback, slot, no_cpu_global as u32);
        let bits = entity.to_bits();
        push_record(&mut table.entity, slot, [bits as u32, (bits >> 32) as u32]);
        push_children_chain(&mut table.first_child, &mut table.next_sibling, slot, children, &nodes);
        let parent = match child_of {
            Some(child_of) => nodes
                .get(child_of.parent())
                .map(GpuSlot::index)
                .unwrap_or(ROOT_PARENT),
            None => ROOT_PARENT,
        };
        push_record(&mut table.parent, slot, parent);
    }

    // One parallel staging write per local column; identical structure, different strides.
    fn write_parts<'a, C: crate::ecs_gpu::GpuColumnDesc>(
        column: &mut GpuColumn<C>,
        mut parts: Vec<&'a [u32]>,
        extra: [&'a [u32]; 1],
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
    ) {
        for e in extra {
            if !e.is_empty() {
                parts.push(e);
            }
        }
        let total: usize = parts.iter().map(|s| s.len()).sum();
        if total > 0 {
            column.write_delta_direct(total, render_device, render_queue, |dst| {
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
    }
    let t_bufs: Vec<&[u32]> = queues.iter_mut().map(|b| b.local_t.as_slice()).collect();
    write_parts(
        &mut local_t_column,
        t_bufs,
        [static_t.as_slice()],
        &render_device,
        &render_queue,
    );
    let rs_bufs: Vec<&[u32]> = queues.iter_mut().map(|b| b.local_rs.as_slice()).collect();
    write_parts(
        &mut local_rs_column,
        rs_bufs,
        [static_rs.as_slice()],
        &render_device,
        &render_queue,
    );
    // Keep each thread-local's capacity for next frame (we copied, not moved).
    for buf in queues.iter_mut() {
        buf.local_t.clear();
        buf.local_rs.clear();
    }
}
