// GPU frontier expansion — changed nodes → changed nodes + all descendants.
//
// Seeds: this frame's changed-`local` delta records plus the gpu-frame extra
// list. Expansion walks the `first_child`/`next_sibling` columns level by level,
// deduping with a per-node frame-epoch stamp; `finalize` (1 thread, between
// levels) closes the level's bounds and writes the next expand's indirect args.
// The finished worklist (`frontier[HEADER..HEADER+total]`, count at word 1) is
// the propagate/readback dispatch list, dispatched indirect from the consumer
// args entry.
//
// Frontier layout (u32 words, must match frontier.rs):
//   [0] total (atomic append cursor)      [1] total_plain (published by finalize)
//   [2] current_level                     [3] pad
//   [4 .. 4+MAX_LEVELS+2] level_begin     … pad to HEADER = 24
//   [HEADER ..] node slots
// Indirect layout: (x,y,z) per expand level, consumer args at [MAX_LEVELS*3..].

struct FrontierParams {
    // Changed-`local` records this frame (seed source 0).
    uint changed_count;
    // Words per changed record; slot at `k * record_stride`.
    uint record_stride;
    // gpu-frame seeds (seed source 1, appended after the delta's).
    uint extra_count;
    // This frame's epoch stamp (monotonic from 1 — a fresh zeroed epoch buffer
    // can never collide).
    uint frame_id;
    // Node coverage (out-of-range guard).
    uint node_count;
    // X workgroup count of the seed's 2D-split dispatch (X capped at 65535,
    // rest in Y). The expands are dispatched indirect — they read their X
    // group count back out of their own level's args instead.
    uint groups_x;
    uint _pad0;
    uint _pad1;
}

[[vk::push_constant]] FrontierParams params;

static const uint NO_NODE = 0xffffffff;
static const uint MAX_LEVELS = 16;
static const uint HEADER = 24;
static const uint LEVEL_BEGIN = 4;
// Safety bound on one node's sibling-chain walk (malformed chain guard).
static const uint MAX_CHAIN = 4194304;

[[vk::binding(0, 0)]] StructuredBuffer<uint> changed;      // [slot, words…] records
[[vk::binding(1, 0)]] StructuredBuffer<uint> extra_seeds;  // gpu-frame slots
[[vk::binding(2, 0)]] StructuredBuffer<uint> first_child;
[[vk::binding(3, 0)]] StructuredBuffer<uint> next_sibling;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> frontier;   // atomic header + worklist
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> epoch;      // per-node frame stamp
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> indirect;   // (x,y,z) per level + consumer

// Stamp the node's epoch; if it wasn't already stamped this frame, append it.
void try_append(uint slot) {
    if (slot >= params.node_count) {
        return;
    }
    uint previous;
    InterlockedExchange(epoch[slot], params.frame_id, previous);
    if (previous == params.frame_id) {
        return; // already in the frontier this frame
    }
    uint pos;
    InterlockedAdd(frontier[0], 1, pos);
    frontier[HEADER + pos] = slot;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void seed(uint3 gid: SV_DispatchThreadID) {
    // Flat index across the 2D-split dispatch.
    let k = gid.x + gid.y * params.groups_x * 64;
    if (k >= params.changed_count + params.extra_count) {
        return;
    }
    uint slot;
    if (k < params.changed_count) {
        slot = changed[k * params.record_stride];
    } else {
        slot = extra_seeds[k - params.changed_count];
    }
    try_append(slot);
}

// 2D-split dispatch args (65535 per-dimension limit — mirror of
// `ecs_gpu::linear_dispatch`); `expand` reconstructs the flat id from `gid`
// plus the X group count it reads back out of its level's args.
void write_args(uint base, uint threads) {
    let groups = (threads + 63) / 64;
    indirect[base] = min(max(groups, 1), 65535);
    indirect[base + 1] = (groups + 65534) / 65535;
    indirect[base + 2] = 1;
}

// One thread: close the current level (its end = the append cursor), write the
// NEXT expand's indirect args from the level's size, publish the running total.
[shader("compute")]
[numthreads(1, 1, 1)]
void finalize() {
    let lvl = frontier[2];
    let total = frontier[0];
    frontier[LEVEL_BEGIN + lvl + 1] = total;
    frontier[2] = lvl + 1;
    if (lvl < MAX_LEVELS) {
        let begin = frontier[LEVEL_BEGIN + lvl];
        write_args(lvl * 3, total - begin);
    }
    // Consumer (propagate/readback) count + args — progressively correct, the
    // last finalize's write wins.
    frontier[1] = total;
    write_args(MAX_LEVELS * 3, total);
}

// Expand the just-closed level: each of its nodes appends its (unstamped) children.
[shader("compute")]
[numthreads(64, 1, 1)]
void expand(uint3 gid: SV_DispatchThreadID) {
    let lvl = frontier[2] - 1;
    let begin = frontier[LEVEL_BEGIN + lvl];
    let end = frontier[LEVEL_BEGIN + lvl + 1];
    // Flat index across the 2D-split indirect dispatch; the X group count is
    // this level's own args entry (written by `finalize`).
    let k = begin + gid.x + gid.y * indirect[lvl * 3] * 64;
    if (k >= end) {
        return;
    }
    let node = frontier[HEADER + k];
    var child = first_child[node];
    uint guard = 0;
    while (child != NO_NODE && guard < MAX_CHAIN) {
        try_append(child);
        child = next_sibling[child];
        guard = guard + 1;
    }
}
