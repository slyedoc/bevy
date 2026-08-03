// GPU transform propagation — ancestor-walk over the changed set, producing each
// node's ABSOLUTE world with a native-f64 translation (SHADER_F64).
//
// `world[node]` is a pure function of the node's own `local`/`parent` ancestor
// chain — it depends on no other node's world. So one thread per node can compute
// it independently by walking up the parent chain and composing locals
// (root∘…∘parent∘local), in a SINGLE pass: no iteration, no ping-pong, no
// read/write hazard (each thread writes only its own world slots and reads
// the read-only local/`parent` columns).
//
// We exploit that to do only the work that changed: the dispatch runs one thread
// per *changed* node (this frame's frontier worklist), recomputing just those
// worlds into the PERSISTENT world buffers; static nodes keep last frame's
// value. The cold-start `full_rebuild` instead runs one thread per node
// (id = slot).
//
// The translation is accumulated in f64 so a node's absolute position survives at
// AU/interstellar magnitude — the huge magnitude is NOT subtracted here. A separate
// pass (`transform_subtract.slang`) subtracts the camera's own f64 world (the origin)
// to emit the small origin-relative f32 the acceleration structure is built from.
// De-fusing the subtract from the walk is what keeps this pass changed-only: the
// origin moving every frame re-runs only the cheap flat subtract, not this walk.
//
// Descendants of a moved parent are included automatically: the frontier pass
// (`transform_frontier.slang`) expands the changed set through the child columns
// GPU-side, and this walk consumes the expanded worklist via indirect dispatch.
//
// Layout: `local_t` is a flat array of f64, 3 per node; `local_rs` is 7 f32 per
// node (rotation xyzw quat, scale .xyz), matching graph.rs's
// `LocalTranslation`/`LocalRS`. `world_abs_linear` is 3 float4 per node (linear
// row k .xyz, .w unused); `world_abs_t` is a flat array of f64, 3 per node.

struct PropagateParams {
    // Threads to dispatch on the `full_rebuild` path (node_count); the changed
    // path's true count lives GPU-side in the frontier header (word 1).
    uint count;
    // 1 → node = thread id (walk every node; cold-start rebuild); 0 → node =
    // `frontier[HEADER + k]` (this frame's changed nodes + their descendants,
    // expanded GPU-side by `transform_frontier.slang`).
    uint full_rebuild;
    // X workgroup count of the full-rebuild 2D-split dispatch (X capped at
    // 65535, rest in Y). The changed path is dispatched indirect from args the
    // frontier wrote — its X group count is recomputed from the worklist count
    // (mirror of the frontier's `write_args`).
    uint groups_x;
    uint _pad;
}

[[vk::push_constant]] PropagateParams params;

static const uint ROOT_PARENT = 0xffffffff;
// Safety bound on the ancestor walk (guards a malformed/cyclic parent chain).
static const uint MAX_DEPTH = 64;

[[vk::binding(0, 0)]] StructuredBuffer<double> local_t;             // 3 per node
[[vk::binding(1, 0)]] StructuredBuffer<float> local_rs;             // 7 per node (quat, scale)
[[vk::binding(2, 0)]] StructuredBuffer<uint> parent;                // 1 per node
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> world_abs_linear;  // 3 per node (persistent)
[[vk::binding(4, 0)]] RWStructuredBuffer<double> world_abs_t;       // 3 per node (persistent)
// The frontier worklist: `[total(unused here), count, level state…]` header then
// node slots from `FRONTIER_HEADER`. Must match `transform_frontier.slang`.
[[vk::binding(5, 0)]] StructuredBuffer<uint> frontier;

static const uint FRONTIER_HEADER = 24;

// A node's local transform: linear rows (rotation·scale, f32) + f64 translation.
struct Local {
    float3 r0;
    float3 r1;
    float3 r2;
    double tx;
    double ty;
    double tz;
}

// Build from a node's raw records: f64 translation, then rotation quat → 3×3 with
// columns scaled by `scale`.
Local load_local(uint node) {
    let tb = node * 3;
    let b = node * 7;
    let qx = local_rs[b]; let qy = local_rs[b + 1]; let qz = local_rs[b + 2]; let qw = local_rs[b + 3];
    let s = float3(local_rs[b + 4], local_rs[b + 5], local_rs[b + 6]);

    let xx = qx * qx; let yy = qy * qy; let zz = qz * qz;
    let xy = qx * qy; let xz = qx * qz; let yz = qy * qz;
    let wx = qw * qx; let wy = qw * qy; let wz = qw * qz;
    // Row-major rotation rows, columns scaled by `scale`.
    Local lo;
    lo.r0 = float3(1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)) * s;
    lo.r1 = float3(2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)) * s;
    lo.r2 = float3(2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)) * s;
    lo.tx = local_t[tb];
    lo.ty = local_t[tb + 1];
    lo.tz = local_t[tb + 2];
    return lo;
}

// f32 row · f64 translation, accumulated in f64.
double row_dot_t(float3 row, double tx, double ty, double tz) {
    return double(row.x) * tx + double(row.y) * ty + double(row.z) * tz;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void propagate(uint3 gid: SV_DispatchThreadID) {
    var node = 0u;
    if (params.full_rebuild == 0) {
        // Changed path (indirect dispatch): the frontier's `write_args` sized
        // the 2D-split from the worklist count — recompute its X group count
        // here to reconstruct the flat index.
        let total = frontier[1];
        let groups_x = min(max((total + 63) / 64, 1), 65535);
        let k = gid.x + gid.y * groups_x * 64;
        if (k >= total) {
            return;
        }
        node = frontier[FRONTIER_HEADER + k];
    } else {
        let k = gid.x + gid.y * params.groups_x * 64;
        if (k >= params.count) {
            return;
        }
        node = k;
    }

    // M starts as the node's own local; left-compose each ancestor walking up to the
    // root: M = local[root] ∘ … ∘ local[parent] ∘ local[node]. Linear part in f32,
    // translation accumulated in f64.
    let m = load_local(node);
    var r0 = m.r0;
    var r1 = m.r1;
    var r2 = m.r2;
    var tx = m.tx;
    var ty = m.ty;
    var tz = m.tz;
    var p = parent[node];
    for (uint step = 0; step < MAX_DEPTH; step++) {
        if (p == ROOT_PARENT) {
            break;
        }
        let a = load_local(p);
        // M = A ∘ M: linear = A.linear · M.linear; translation = A.linear · M.t + A.t.
        // Columns of M.linear (for the linear product).
        let c0 = float3(r0.x, r1.x, r2.x);
        let c1 = float3(r0.y, r1.y, r2.y);
        let c2 = float3(r0.z, r1.z, r2.z);
        r0 = float3(dot(a.r0, c0), dot(a.r0, c1), dot(a.r0, c2));
        r1 = float3(dot(a.r1, c0), dot(a.r1, c1), dot(a.r1, c2));
        r2 = float3(dot(a.r2, c0), dot(a.r2, c1), dot(a.r2, c2));
        // translation: A.linear · t + A.t, per component in f64.
        let ntx = row_dot_t(a.r0, tx, ty, tz) + a.tx;
        let nty = row_dot_t(a.r1, tx, ty, tz) + a.ty;
        let ntz = row_dot_t(a.r2, tx, ty, tz) + a.tz;
        tx = ntx;
        ty = nty;
        tz = ntz;
        p = parent[p];
    }

    let wb = node * 3;
    world_abs_linear[wb]     = float4(r0, 0.0);
    world_abs_linear[wb + 1] = float4(r1, 0.0);
    world_abs_linear[wb + 2] = float4(r2, 0.0);
    world_abs_t[wb]     = tx;
    world_abs_t[wb + 1] = ty;
    world_abs_t[wb + 2] = tz;
}
