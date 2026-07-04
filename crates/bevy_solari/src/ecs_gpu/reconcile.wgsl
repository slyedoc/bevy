// GPU instance reconcile — folds the CPU's absolute-state change journal into the
// per-instance GPU columns. One thread per journal record; an UPSERT writes ALL of
// a slot's columns from the single record, so a reused slot is fully re-initialized
// in one pass (no partial/stale column survives — the aliasing fix). This replaces
// the per-column delta scatters: the journal carries every column's value, and the
// reconcile is the sole writer of these columns.

// Must match `instance::journal::InstanceJournalRecord` (48 B, 12 × u32).
struct JournalRecord {
    slot: u32,
    partition_hint: u32,
    op: u32,
    geometry_id: u32,
    material_id: u32,
    node_key: u32,
    cull_mask: u32,
    flags: u32,
    group_base: u32,
    cluster_base: u32,
    cluster_count: u32,
    root_group: u32,
}

const OP_UPSERT: u32 = 0u;
const OP_REMOVE: u32 = 1u;

struct ReconcileParams {
    // Journal record count this frame (dispatch bound).
    count: u32,
}

@group(0) @binding(0) var<storage, read> journal: array<JournalRecord>;
@group(0) @binding(1) var<uniform> params: ReconcileParams;

// The set-once-at-bind per-instance columns, slot-indexed (the same buffers the
// path tracer / PTLAS fill read). The reconcile is their sole writer — the CPU
// delta scatter for these is removed. `material`/`mask` are NOT here: they change
// after bind (material re-resolved in Prepare, mask on a RenderLayers change), so
// they stay on their existing change-driven scatter paths.
@group(0) @binding(2) var<storage, read_write> node_slots: array<u32>;
@group(0) @binding(3) var<storage, read_write> geometry_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> group_bases: array<u32>;
// `InstanceLodInputGpu` = (cluster_base, cluster_count, group_base, root_group).
@group(0) @binding(5) var<storage, read_write> lod_inputs: array<vec4<u32>>;
// PTLAS regular-partition hint (0xffffffff = derive from the static flag).
@group(0) @binding(6) var<storage, read_write> partition_hints: array<u32>;

@compute @workgroup_size(64)
fn reconcile_apply(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    // Flat index across a 2D-split dispatch (X capped at 65535, rest in Y).
    let i = gid.x + gid.y * num_workgroups.x * 64u;
    if i >= params.count {
        return;
    }
    let rec = journal[i];
    if rec.op != OP_UPSERT {
        // REMOVE: the slot's PTLAS presence is cleared elsewhere (the fill writes a
        // null AS for a disabled slot); its columns are left for the next UPSERT to
        // overwrite, so nothing to do here.
        return;
    }
    let slot = rec.slot;
    node_slots[slot] = rec.node_key;
    geometry_ids[slot] = rec.geometry_id;
    group_bases[slot] = rec.group_base;
    // This vec4 order MUST match `InstanceLodInputGpu` (cluster_base, cluster_count,
    // group_base, root_group) — the layout the CPU delta scatter writes — so the
    // reconcile and the CPU writer produce byte-identical columns (the authority flip
    // relies on this).
    lod_inputs[slot] = vec4<u32>(rec.cluster_base, rec.cluster_count, rec.group_base, rec.root_group);
    partition_hints[slot] = rec.partition_hint;
}
