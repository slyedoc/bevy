// GPU → CPU GlobalTransform readback gather.
//
// One thread per CHANGED node (the local columns' delta this frame — the same
// set propagation walks). Each atomic-appends the node's ABSOLUTE world (f32
// linear + f64 translation, moved as raw word pairs) into the output buffer,
// which bevy's `Readback` streams to the main world. The CPU then resolves
// slot→entity and writes `GlobalTransform` (skipping opted-out ones). Absolute —
// not camera-relative — so a static entity's readback never changes when the
// camera moves, and gameplay/physics get true f64 world positions.
//
// `out[0]` is an atomic record counter, reset to 0 each frame by the host before
// this pass; it travels in the buffer so the CPU (reading 1-3 frames late) knows
// how many records are valid.
//
// Record k layout (u32 words, after the 4-word header):
//   [0]     = slot
//   [1..10] = absolute world linear, 3×3 rows (9 floats, bitcast)
//   [10..16] = absolute world translation, 3×f64 as raw word pairs [lo, hi] each
//   [16,17] = owning entity bits (Entity::to_bits as [lo, hi]); the CPU resolves the
//             entity from these and writes its GlobalTransform (entity-keyed identity)

struct ReadbackParams {
    uint node_count;
    uint capacity;  // max output records
    uint _pad0;
    uint _pad1;
}

[[vk::push_constant]] ReadbackParams params;

// The frontier worklist (changed nodes + GPU-expanded descendants): count at
// word 1, slots from `FRONTIER_HEADER`. Must match `transform_frontier.slang`.
[[vk::binding(0, 0)]] StructuredBuffer<uint> frontier;
[[vk::binding(1, 0)]] StructuredBuffer<float4> world_abs_linear; // 3 per node (.w unused)
// The f64 translation buffer viewed as raw word pairs — the gather just moves
// bits, so it needs no SHADER_F64 itself; the CPU reassembles f64 from the words.
[[vk::binding(2, 0)]] StructuredBuffer<uint2> world_abs_t;       // 3 per node (f64 bit pairs)
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> out;              // [0]=count (atomic), then records
[[vk::binding(4, 0)]] StructuredBuffer<uint> no_readback;        // 1 = opt out (NoGpuGlobalTransformReadback)
[[vk::binding(5, 0)]] StructuredBuffer<uint> parent;             // parent node-slot (ROOT_PARENT at roots)
[[vk::binding(6, 0)]] StructuredBuffer<uint2> node_entity;       // owning entity bits [lo, hi]

static const uint HEADER = 4;
static const uint RECORD = 18;
static const uint FRONTIER_HEADER = 24;
// Root sentinel — must match `ROOT_PARENT` in graph.rs / transform_propagate.slang.
static const uint ROOT_PARENT = 0xffffffff;
// Ancestor-walk depth guard (cycle / corrupt-parent backstop).
static const uint MAX_DEPTH = 64;

// The opt-out cascades down the hierarchy: a node is skipped if it OR any
// ancestor is marked `NoGpuGlobalTransformReadback`. Walks the parent chain (the
// columns persist, so ancestors' flags are present even when they didn't change
// this frame). Bounded by tree depth; `MAX_DEPTH`-guarded.
bool opted_out(uint start) {
    var p = start;
    for (uint i = 0; i < MAX_DEPTH; i++) {
        if (no_readback[p] != 0) {
            return true;
        }
        let next = parent[p];
        if (next == ROOT_PARENT || next >= params.node_count) {
            return false;
        }
        p = next;
    }
    return false;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void readback(uint3 gid: SV_DispatchThreadID) {
    // Flat index across the 2D-split indirect dispatch: the frontier's
    // `write_args` sized the split from the worklist count — recompute its X
    // group count here (mirror of that sizing).
    let total = frontier[1];
    let groups_x = min(max((total + 63) / 64, 1), 65535);
    let k = gid.x + gid.y * groups_x * 64;
    if (k >= total) {
        return;
    }
    let slot = frontier[FRONTIER_HEADER + k];
    if (slot >= params.node_count) {
        return; // unassigned / out-of-range node.
    }
    if (opted_out(slot)) {
        return; // opted out (self or an ancestor `NoGpuGlobalTransformReadback`).
    }

    // Allocate an output record slot (atomic counter at out[0]).
    uint idx;
    InterlockedAdd(out[0], 1, idx);
    if (idx >= params.capacity) {
        return; // overflow — dropped (host warns).
    }
    let base = HEADER + idx * RECORD;

    let s = slot * 3;
    let r0 = world_abs_linear[s];
    let r1 = world_abs_linear[s + 1];
    let r2 = world_abs_linear[s + 2];
    let tx = world_abs_t[s];
    let ty = world_abs_t[s + 1];
    let tz = world_abs_t[s + 2];
    // Disjoint `idx`, so these stores don't contend.
    out[base]      = slot;
    out[base + 1]  = asuint(r0.x);
    out[base + 2]  = asuint(r0.y);
    out[base + 3]  = asuint(r0.z);
    out[base + 4]  = asuint(r1.x);
    out[base + 5]  = asuint(r1.y);
    out[base + 6]  = asuint(r1.z);
    out[base + 7]  = asuint(r2.x);
    out[base + 8]  = asuint(r2.y);
    out[base + 9]  = asuint(r2.z);
    out[base + 10] = tx.x;
    out[base + 11] = tx.y;
    out[base + 12] = ty.x;
    out[base + 13] = ty.y;
    out[base + 14] = tz.x;
    out[base + 15] = tz.y;
    // Stamp the owning entity's bits; the CPU resolves the entity from these and writes
    // its GlobalTransform. A recycled slot carries a different entity, so a late
    // readback can never splat onto the slot's new occupant.
    let e = node_entity[slot];
    out[base + 16] = e.x;
    out[base + 17] = e.y;
}
