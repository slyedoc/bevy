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
// how many records are valid. The whole buffer is typed `atomic<u32>` so the
// counter and the record words share one binding.
//
// Record k layout (u32 words, after the 4-word header):
//   [0]     = slot
//   [1..10] = absolute world linear, 3×3 rows (9 floats, bitcast)
//   [10..16] = absolute world translation, 3×f64 as raw word pairs [lo, hi] each
//   [16,17] = owning entity bits (Entity::to_bits as [lo, hi]); the CPU resolves the
//             entity from these and writes its GlobalTransform (entity-keyed identity)

struct ReadbackParams {
    changed_count: u32,
    record_stride: u32,  // words per `local` delta record (slot at k*stride)
    node_count: u32,
    capacity: u32,       // max output records
}

@group(0) @binding(0) var<storage, read> changed: array<u32>;        // local delta [slot, words…]
@group(0) @binding(1) var<storage, read> world_abs_linear: array<vec4<f32>>; // 3 per node (.w unused)
// The f64 translation buffer viewed as raw word pairs — the gather just moves
// bits, so it needs no SHADER_F64 itself; the CPU reassembles f64 from the words.
@group(0) @binding(2) var<storage, read> world_abs_t: array<vec2<u32>>; // 3 per node (f64 bit pairs)
@group(0) @binding(3) var<storage, read_write> out: array<atomic<u32>>; // [0]=count, then records
@group(0) @binding(4) var<uniform> params: ReadbackParams;
@group(0) @binding(5) var<storage, read> no_readback: array<u32>;    // 1 = opt out (NoGpuGlobalTransformReadback)
@group(0) @binding(6) var<storage, read> parent: array<u32>;         // parent node-slot (ROOT_PARENT at roots)
@group(0) @binding(7) var<storage, read> node_entity: array<vec2<u32>>; // owning entity bits [lo, hi]

const HEADER: u32 = 4u;
const RECORD: u32 = 18u;
// Root sentinel — must match `ROOT_PARENT` in graph.rs / transform_propagate.wgsl.
const ROOT_PARENT: u32 = 0xffffffffu;
// Ancestor-walk depth guard (cycle / corrupt-parent backstop).
const MAX_DEPTH: u32 = 64u;

// The opt-out cascades down the hierarchy: a node is skipped if it OR any
// ancestor is marked `NoGpuGlobalTransformReadback`. Walks the parent chain (the
// columns persist, so ancestors' flags are present even when they didn't change
// this frame). Bounded by tree depth; `MAX_DEPTH`-guarded.
fn opted_out(start: u32) -> bool {
    var p = start;
    for (var i = 0u; i < MAX_DEPTH; i = i + 1u) {
        if no_readback[p] != 0u {
            return true;
        }
        let next = parent[p];
        if next == ROOT_PARENT || next >= params.node_count {
            return false;
        }
        p = next;
    }
    return false;
}

@compute @workgroup_size(64)
fn readback(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    // Flat index across a 2D-split dispatch (X capped at 65535, rest in Y).
    let k = gid.x + gid.y * num_workgroups.x * 64u;
    if k >= params.changed_count {
        return;
    }
    let slot = changed[k * params.record_stride];
    if slot >= params.node_count {
        return; // unassigned / out-of-range node.
    }
    if opted_out(slot) {
        return; // opted out (self or an ancestor `NoGpuGlobalTransformReadback`).
    }

    // Allocate an output record slot (atomic counter at out[0]).
    let idx = atomicAdd(&out[0], 1u);
    if idx >= params.capacity {
        return; // overflow — dropped (host warns).
    }
    let base = HEADER + idx * RECORD;

    let s = slot * 3u;
    let r0 = world_abs_linear[s];
    let r1 = world_abs_linear[s + 1u];
    let r2 = world_abs_linear[s + 2u];
    let tx = world_abs_t[s];
    let ty = world_abs_t[s + 1u];
    let tz = world_abs_t[s + 2u];
    // Disjoint `idx`, so these stores don't contend; atomic store because the
    // binding is `atomic<u32>`.
    atomicStore(&out[base],        slot);
    atomicStore(&out[base + 1u],   bitcast<u32>(r0.x));
    atomicStore(&out[base + 2u],   bitcast<u32>(r0.y));
    atomicStore(&out[base + 3u],   bitcast<u32>(r0.z));
    atomicStore(&out[base + 4u],   bitcast<u32>(r1.x));
    atomicStore(&out[base + 5u],   bitcast<u32>(r1.y));
    atomicStore(&out[base + 6u],   bitcast<u32>(r1.z));
    atomicStore(&out[base + 7u],   bitcast<u32>(r2.x));
    atomicStore(&out[base + 8u],   bitcast<u32>(r2.y));
    atomicStore(&out[base + 9u],   bitcast<u32>(r2.z));
    atomicStore(&out[base + 10u],  tx.x);
    atomicStore(&out[base + 11u],  tx.y);
    atomicStore(&out[base + 12u],  ty.x);
    atomicStore(&out[base + 13u],  ty.y);
    atomicStore(&out[base + 14u],  tz.x);
    atomicStore(&out[base + 15u],  tz.y);
    // Stamp the owning entity's bits; the CPU resolves the entity from these and writes
    // its GlobalTransform. A recycled slot carries a different entity, so a late
    // readback can never splat onto the slot's new occupant.
    let e = node_entity[slot];
    atomicStore(&out[base + 16u],  e.x);
    atomicStore(&out[base + 17u],  e.y);
}
