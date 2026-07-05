// GPU transform propagation — ancestor-walk over the changed set, producing each
// node's ABSOLUTE world with a native-f64 translation (SHADER_F64).
//
// `world[node]` is a pure function of the node's own `local`/`parent` ancestor
// chain — it depends on no other node's world. So one thread per node can compute
// it independently by walking up the parent chain and composing locals
// (root∘…∘parent∘local), in a SINGLE pass: no Jacobi iteration, no ping-pong,
// no read/write hazard (each thread writes only its own world slots and reads
// the read-only local/`parent` columns).
//
// We exploit that to do only the work that changed: the dispatch runs one thread
// per *changed* node (the local columns' delta this frame — `changed[k*stride]`
// is the slot), recomputing just those worlds into the PERSISTENT world buffers;
// static nodes keep last frame's value. On a capacity growth (`full_rebuild`) we
// instead run one thread per node (id = slot).
//
// The translation is accumulated in f64 so a node's absolute position survives at
// AU/interstellar magnitude — the huge magnitude is NOT subtracted here. A separate
// pass (`transform_subtract.wgsl`) subtracts the camera's own f64 world (the origin)
// to emit the small origin-relative f32 the acceleration structure is built from.
// De-fusing the subtract from the walk is what keeps this pass changed-only: the
// origin moving every frame re-runs only the cheap flat subtract, not this walk.
//
// Descendants of a moved parent are included automatically: the frontier pass
// (`transform_frontier.wgsl`) expands the changed set through the child columns
// GPU-side, and this walk consumes the expanded worklist via indirect dispatch.
//
// Layout: `local_t` is a flat array<f64>, 3 per node; `local_rs` is 7 f32 per node
// (rotation xyzw quat, scale .xyz), matching graph.rs's `LocalTranslation`/`LocalRS`.
// `world_abs_linear` is 3 vec4 per node (linear row k .xyz, .w unused);
// `world_abs_t` is a flat array<f64>, 3 per node.
// NOTE: unsuffixed WGSL float literals are f32 — f64 constants need explicit typing.

struct PropagateParams {
    // Threads to dispatch on the `full_rebuild` path (node_count); the changed
    // path's true count lives GPU-side in the frontier header (word 1).
    count: u32,
    // Unused on the frontier path (slots are flat); kept for layout stability.
    record_stride: u32,
    // 1 → node = thread id (walk every node, e.g. after a growth); 0 → node =
    // `frontier[HEADER + k]` (this frame's changed nodes + their descendants,
    // expanded GPU-side by `transform_frontier.wgsl`).
    full_rebuild: u32,
    _pad: u32,
}

const ROOT_PARENT: u32 = 0xffffffffu;
// Safety bound on the ancestor walk (guards a malformed/cyclic parent chain).
const MAX_DEPTH: u32 = 64u;

@group(0) @binding(0) var<storage, read> local_t: array<f64>;                // 3 per node
@group(0) @binding(1) var<storage, read> local_rs: array<f32>;               // 7 per node (quat, scale)
@group(0) @binding(2) var<storage, read> parent: array<u32>;                 // 1 per node
@group(0) @binding(3) var<storage, read_write> world_abs_linear: array<vec4<f32>>; // 3 per node (persistent)
@group(0) @binding(4) var<storage, read_write> world_abs_t: array<f64>;      // 3 per node (persistent)
// The frontier worklist: `[total(unused here), count, level state…]` header then
// node slots from `FRONTIER_HEADER`. Must match `transform_frontier.wgsl`.
@group(0) @binding(5) var<storage, read> frontier: array<u32>;
@group(0) @binding(6) var<uniform> params: PropagateParams;

const FRONTIER_HEADER: u32 = 24u;

// A node's local transform: linear rows (rotation·scale, f32) + f64 translation.
struct Local {
    r0: vec3<f32>,
    r1: vec3<f32>,
    r2: vec3<f32>,
    tx: f64,
    ty: f64,
    tz: f64,
}

// Build from a node's raw records: f64 translation, then rotation quat → 3×3 with
// columns scaled by `scale`.
fn load_local(node: u32) -> Local {
    let tb = node * 3u;
    let b = node * 7u;
    let qx = local_rs[b]; let qy = local_rs[b + 1u]; let qz = local_rs[b + 2u]; let qw = local_rs[b + 3u];
    let s = vec3<f32>(local_rs[b + 4u], local_rs[b + 5u], local_rs[b + 6u]);

    let xx = qx * qx; let yy = qy * qy; let zz = qz * qz;
    let xy = qx * qy; let xz = qx * qz; let yz = qy * qz;
    let wx = qw * qx; let wy = qw * qy; let wz = qw * qz;
    // Row-major rotation rows, columns scaled by `scale`.
    var lo: Local;
    lo.r0 = vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)) * s;
    lo.r1 = vec3<f32>(2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)) * s;
    lo.r2 = vec3<f32>(2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)) * s;
    lo.tx = local_t[tb];
    lo.ty = local_t[tb + 1u];
    lo.tz = local_t[tb + 2u];
    return lo;
}

// f32 row · f64 translation, accumulated in f64.
fn row_dot_t(row: vec3<f32>, tx: f64, ty: f64, tz: f64) -> f64 {
    return f64(row.x) * tx + f64(row.y) * ty + f64(row.z) * tz;
}

@compute @workgroup_size(64)
fn propagate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let k = gid.x + gid.y * num_workgroups.x * 64u;
    var node = k;
    if params.full_rebuild == 0u {
        if k >= frontier[1u] {
            return;
        }
        node = frontier[FRONTIER_HEADER + k];
    } else if k >= params.count {
        return;
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
    for (var step = 0u; step < MAX_DEPTH; step = step + 1u) {
        if p == ROOT_PARENT {
            break;
        }
        let a = load_local(p);
        // M = A ∘ M: linear = A.linear · M.linear; translation = A.linear · M.t + A.t.
        // Columns of M.linear (for the linear product).
        let c0 = vec3<f32>(r0.x, r1.x, r2.x);
        let c1 = vec3<f32>(r0.y, r1.y, r2.y);
        let c2 = vec3<f32>(r0.z, r1.z, r2.z);
        r0 = vec3<f32>(dot(a.r0, c0), dot(a.r0, c1), dot(a.r0, c2));
        r1 = vec3<f32>(dot(a.r1, c0), dot(a.r1, c1), dot(a.r1, c2));
        r2 = vec3<f32>(dot(a.r2, c0), dot(a.r2, c1), dot(a.r2, c2));
        // translation: A.linear · t + A.t, per component in f64.
        let ntx = row_dot_t(a.r0, tx, ty, tz) + a.tx;
        let nty = row_dot_t(a.r1, tx, ty, tz) + a.ty;
        let ntz = row_dot_t(a.r2, tx, ty, tz) + a.tz;
        tx = ntx;
        ty = nty;
        tz = ntz;
        p = parent[p];
    }

    let wb = node * 3u;
    world_abs_linear[wb]      = vec4<f32>(r0, 0.0);
    world_abs_linear[wb + 1u] = vec4<f32>(r1, 0.0);
    world_abs_linear[wb + 2u] = vec4<f32>(r2, 0.0);
    world_abs_t[wb]      = tx;
    world_abs_t[wb + 1u] = ty;
    world_abs_t[wb + 2u] = tz;
}
