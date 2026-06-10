// GPU transform propagation — ancestor-walk over the changed set.
//
// `world[node]` is a pure function of the node's own `local`/`parent` ancestor
// chain — it depends on no other node's world. So one thread per node can
// compute it independently by walking up the parent chain and composing locals
// (root∘…∘parent∘local), in a SINGLE pass: no Jacobi iteration, no ping-pong,
// no read/write hazard (each thread writes only its own `world` slot and reads
// the read-only `local`/`parent` columns).
//
// We exploit that to do only the work that changed: the dispatch runs one thread
// per *changed* node (the `local` column's delta this frame — `changed[k*stride]`
// is the slot), recomputing just those worlds into the PERSISTENT `world` buffer;
// static nodes keep last frame's value. On a capacity growth (`full_rebuild`) we
// instead run one thread per node (id = slot) to repopulate the whole buffer.
//
// Boundary: this updates a changed node's own world. A change that moves a
// node's DESCENDANTS without changing their own `local` (animating a parent, or
// a bare re-parent) won't re-walk those descendants — they'd keep a stale world.
// bevy_city only animates leaves, so this is exact there; a general hierarchy
// would also dirty descendants (a downward 1-bit propagation) — a later step.
//
// Layout: `local` is raw TRS — 10 floats per node (translation.xyz, rotation
// xyzw quat, scale.xyz), matching `LocalTRS`; `load_local` builds the matrix from
// it. `world` is mat3x4<f32> as 3 vec4 rows per node — row k is (linear row k
// .xyz, translation component k .w), matching `Affine3x4`.

struct PropagateParams {
    // Threads to dispatch: `full_rebuild` → node_count; else changed-record count.
    count: u32,
    // Words per changed record (`WORDS + 1`); the slot is at `k * record_stride`.
    record_stride: u32,
    // 1 → node = thread id (walk every node, e.g. after a growth); 0 → node =
    // `changed[k * record_stride]` (walk only this frame's changed nodes).
    full_rebuild: u32,
    _pad: u32,
}

const ROOT_PARENT: u32 = 0xffffffffu;
// Safety bound on the ancestor walk (guards a malformed/cyclic parent chain).
const MAX_DEPTH: u32 = 64u;

@group(0) @binding(0) var<storage, read> local: array<f32>;              // 10 per node (TRS)
@group(0) @binding(1) var<storage, read> parent: array<u32>;             // 1 per node
@group(0) @binding(2) var<storage, read_write> world: array<vec4<f32>>;  // 3 per node (persistent)
@group(0) @binding(3) var<storage, read> changed: array<u32>;            // [slot, words…] per record
@group(0) @binding(4) var<uniform> params: PropagateParams;

// One row of `A ∘ B`: A's row `ar` (.xyz linear, .w translation) times B's linear
// columns `bc{0,1,2}` and translation `bt`.
fn mul_row(ar: vec4<f32>, bc0: vec3<f32>, bc1: vec3<f32>, bc2: vec3<f32>, bt: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(dot(ar.xyz, bc0), dot(ar.xyz, bc1), dot(ar.xyz, bc2), dot(ar.xyz, bt) + ar.w);
}

// A node's local transform as mat3x4 rows (linear row k .xyz, translation k .w).
struct Mat3x4 { r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32> }

// Build the mat3x4 from a node's raw TRS (`local[node*10 .. +10]`): rotation quat
// → 3×3, columns scaled by `scale`, translation in `.w`.
fn load_local(node: u32) -> Mat3x4 {
    let b = node * 10u;
    let t = vec3<f32>(local[b], local[b + 1u], local[b + 2u]);
    let qx = local[b + 3u]; let qy = local[b + 4u]; let qz = local[b + 5u]; let qw = local[b + 6u];
    let s = vec3<f32>(local[b + 7u], local[b + 8u], local[b + 9u]);

    let xx = qx * qx; let yy = qy * qy; let zz = qz * qz;
    let xy = qx * qy; let xz = qx * qz; let yz = qy * qz;
    let wx = qw * qx; let wy = qw * qy; let wz = qw * qz;
    // Row-major rotation rows.
    let rot0 = vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy));
    let rot1 = vec3<f32>(2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx));
    let rot2 = vec3<f32>(2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy));

    var m: Mat3x4;
    // Scale columns (component j by s[j]) and pack translation into .w.
    m.r0 = vec4<f32>(rot0 * s, t.x);
    m.r1 = vec4<f32>(rot1 * s, t.y);
    m.r2 = vec4<f32>(rot2 * s, t.z);
    return m;
}

@compute @workgroup_size(64)
fn propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.count {
        return;
    }
    var node = k;
    if params.full_rebuild == 0u {
        node = changed[k * params.record_stride];
    }

    // M starts as the node's own local, then left-compose each ancestor's local
    // walking up to the root: M = local[root] ∘ … ∘ local[parent] ∘ local[node].
    let m = load_local(node);
    var m0 = m.r0;
    var m1 = m.r1;
    var m2 = m.r2;
    var p = parent[node];
    for (var step = 0u; step < MAX_DEPTH; step = step + 1u) {
        if p == ROOT_PARENT {
            break;
        }
        let a = load_local(p);
        let a0 = a.r0;
        let a1 = a.r1;
        let a2 = a.r2;
        // M = local[p] ∘ M (parent ∘ child). Columns + translation of M:
        let bc0 = vec3<f32>(m0.x, m1.x, m2.x);
        let bc1 = vec3<f32>(m0.y, m1.y, m2.y);
        let bc2 = vec3<f32>(m0.z, m1.z, m2.z);
        let bt = vec3<f32>(m0.w, m1.w, m2.w);
        m0 = mul_row(a0, bc0, bc1, bc2, bt);
        m1 = mul_row(a1, bc0, bc1, bc2, bt);
        m2 = mul_row(a2, bc0, bc1, bc2, bt);
        p = parent[p];
    }

    let wb = node * 3u;
    world[wb] = m0;
    world[wb + 1u] = m1;
    world[wb + 2u] = m2;
}
