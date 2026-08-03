// GPU instance reconcile — folds the CPU's absolute-state change journal into the
// per-instance GPU columns. One thread per journal record; an UPSERT writes ALL of
// a slot's columns from the single record, so a reused slot is fully re-initialized
// in one pass (no partial/stale column survives). The journal carries every
// column's value; the reconcile is the sole writer of these columns.

// Must match `instance::journal::InstanceJournalRecord` (48 B, 12 × u32).
struct JournalRecord {
    uint slot;
    uint partition_hint;
    uint op;
    uint geometry_id;
    uint material_id;
    uint node_key;
    uint cull_mask;
    uint flags;
    uint group_base;
    uint cluster_base;
    uint cluster_count;
    uint root_group;
}

static const uint OP_UPSERT = 0;
static const uint OP_REMOVE = 1;

struct ReconcileParams {
    // Journal record count this frame (dispatch bound).
    uint count;
    // X workgroup count of the 2D-split dispatch (X capped at 65535, rest in Y).
    uint groups_x;
    uint _pad0;
    uint _pad1;
}

[[vk::push_constant]] ReconcileParams params;

[[vk::binding(0, 0)]] StructuredBuffer<JournalRecord> journal;

// The set-once-at-bind per-instance columns, slot-indexed (the same buffers the
// path tracer / PTLAS fill read). The reconcile is their sole writer.
// `material`/`mask` are NOT here: they change after bind (material re-resolved
// in Prepare, mask on a RenderLayers change), so they use the change-driven
// scatter paths instead.
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> node_slots;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> geometry_ids;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> group_bases;
// `InstanceLodInputGpu` = (cluster_base, cluster_count, group_base, root_group).
[[vk::binding(4, 0)]] RWStructuredBuffer<uint4> lod_inputs;
// PTLAS regular-partition hint (0xffffffff = derive from the static flag).
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> partition_hints;

[shader("compute")]
[numthreads(64, 1, 1)]
void reconcile_apply(uint3 gid: SV_DispatchThreadID) {
    // Flat index across the 2D-split dispatch.
    let i = gid.x + gid.y * params.groups_x * 64;
    if (i >= params.count) {
        return;
    }
    let rec = journal[i];
    if (rec.op != OP_UPSERT) {
        // REMOVE: the slot's PTLAS presence is cleared elsewhere (the fill writes a
        // null AS for a disabled slot); its columns are left for the next UPSERT to
        // overwrite, so nothing to do here.
        return;
    }
    let slot = rec.slot;
    node_slots[slot] = rec.node_key;
    geometry_ids[slot] = rec.geometry_id;
    group_bases[slot] = rec.group_base;
    // This uint4 order MUST match `InstanceLodInputGpu` (cluster_base, cluster_count,
    // group_base, root_group).
    lod_inputs[slot] = uint4(rec.cluster_base, rec.cluster_count, rec.group_base, rec.root_group);
    partition_hints[slot] = rec.partition_hint;
}
