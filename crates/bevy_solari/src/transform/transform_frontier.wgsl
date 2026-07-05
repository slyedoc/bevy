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
    changed_count: u32,
    record_stride: u32,
    extra_count: u32,
    frame_id: u32,
    node_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const NO_NODE: u32 = 0xffffffffu;
const MAX_LEVELS: u32 = 16u;
const HEADER: u32 = 24u;
const LEVEL_BEGIN: u32 = 4u;
// Safety bound on one node's sibling-chain walk (malformed chain guard).
const MAX_CHAIN: u32 = 4194304u;

@group(0) @binding(0) var<storage, read> changed: array<u32>;        // [slot, words…] records
@group(0) @binding(1) var<storage, read> extra_seeds: array<u32>;    // gpu-frame slots
@group(0) @binding(2) var<storage, read> first_child: array<u32>;
@group(0) @binding(3) var<storage, read> next_sibling: array<u32>;
@group(0) @binding(4) var<storage, read_write> frontier: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> epoch: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(7) var<uniform> params: FrontierParams;

// Stamp the node's epoch; if it wasn't already stamped this frame, append it.
fn try_append(slot: u32) {
    if slot >= params.node_count {
        return;
    }
    if atomicExchange(&epoch[slot], params.frame_id) == params.frame_id {
        return; // already in the frontier this frame
    }
    let pos = atomicAdd(&frontier[0], 1u);
    atomicStore(&frontier[HEADER + pos], slot);
}

@compute @workgroup_size(64)
fn seed(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let k = gid.x + gid.y * num_workgroups.x * 64u;
    if k >= params.changed_count + params.extra_count {
        return;
    }
    var slot: u32;
    if k < params.changed_count {
        slot = changed[k * params.record_stride];
    } else {
        slot = extra_seeds[k - params.changed_count];
    }
    try_append(slot);
}

// 2D-split dispatch args (65535 per-dimension limit — mirror of
// `ecs_gpu::linear_dispatch`); kernels reconstruct the flat id from
// `gid + num_workgroups`.
fn write_args(base: u32, threads: u32) {
    let groups = (threads + 63u) / 64u;
    indirect[base] = min(max(groups, 1u), 65535u);
    indirect[base + 1u] = (groups + 65534u) / 65535u;
    indirect[base + 2u] = 1u;
}

// One thread: close the current level (its end = the append cursor), write the
// NEXT expand's indirect args from the level's size, publish the running total.
@compute @workgroup_size(1)
fn finalize() {
    let lvl = atomicLoad(&frontier[2]);
    let total = atomicLoad(&frontier[0]);
    atomicStore(&frontier[LEVEL_BEGIN + lvl + 1u], total);
    atomicStore(&frontier[2], lvl + 1u);
    if lvl < MAX_LEVELS {
        let begin = atomicLoad(&frontier[LEVEL_BEGIN + lvl]);
        write_args(lvl * 3u, total - begin);
    }
    // Consumer (propagate/readback) count + args — progressively correct, the
    // last finalize's write wins.
    atomicStore(&frontier[1], total);
    write_args(MAX_LEVELS * 3u, total);
}

// Expand the just-closed level: each of its nodes appends its (unstamped) children.
@compute @workgroup_size(64)
fn expand(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let lvl = atomicLoad(&frontier[2]) - 1u;
    let begin = atomicLoad(&frontier[LEVEL_BEGIN + lvl]);
    let end = atomicLoad(&frontier[LEVEL_BEGIN + lvl + 1u]);
    let k = begin + gid.x + gid.y * num_workgroups.x * 64u;
    if k >= end {
        return;
    }
    let node = atomicLoad(&frontier[HEADER + k]);
    var child = first_child[node];
    var guard = 0u;
    while child != NO_NODE && guard < MAX_CHAIN {
        try_append(child);
        child = next_sibling[child];
        guard = guard + 1u;
    }
}
