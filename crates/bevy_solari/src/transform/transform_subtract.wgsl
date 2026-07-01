// Transform subtract — the floating-origin pass. Reads each node's ABSOLUTE world
// (f32 linear rows + native-f64 translation, from transform_propagate) and writes
// its ORIGIN-RELATIVE f32 world (`world_rel`, a mat3x4 — what every RT consumer
// reads via `current_world()`).
//
// The origin is the primary camera's own absolute world, read straight off the GPU
// as `world_abs_t[camera_slot]` — no CPU-set origin. The subtraction happens in f64,
// so the huge shared magnitude cancels before the narrowing to f32 — a metre-scale
// offset survives even at AU scale where a naive f32 subtract rounds to zero. The
// camera node subtracts its own world → resolves to 0, so it renders at the origin
// by construction.
//
// This is a flat one-thread-per-node pass (no ancestor walk). It re-runs whenever the
// walk ran — i.e. when the camera (origin) moved or any node moved — which is cheap
// even over every node; the expensive chain walk stays changed-only in the other pass.

struct SubtractParams {
    // Nodes to process (world high-water).
    count: u32,
    // The primary camera's transform-table slot — `world_abs_t[camera_slot]` is the origin.
    camera_slot: u32,
    // World-buffer node high-water (bounds guard for `camera_slot`).
    node_count: u32,
    // 1 = a camera slot is live this frame → subtract the origin; 0 → pass the absolute
    // world through unchanged (no floating origin yet, e.g. cold start).
    origin_valid: u32,
}

@group(0) @binding(0) var<storage, read> world_abs_linear: array<vec4<f32>>; // 3 per node
@group(0) @binding(1) var<storage, read> world_abs_t: array<f64>;            // 3 per node
@group(0) @binding(2) var<storage, read_write> world_rel: array<vec4<f32>>;  // 3 per node (mat3x4)
@group(0) @binding(3) var<uniform> params: SubtractParams;

@compute @workgroup_size(64)
fn subtract(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let k = gid.x + gid.y * num_workgroups.x * 64u;
    if k >= params.count {
        return;
    }

    let b = k * 3u;
    var tx = world_abs_t[b];
    var ty = world_abs_t[b + 1u];
    var tz = world_abs_t[b + 2u];
    if params.origin_valid == 1u && params.camera_slot < params.node_count {
        let ob = params.camera_slot * 3u;
        tx = tx - world_abs_t[ob];
        ty = ty - world_abs_t[ob + 1u];
        tz = tz - world_abs_t[ob + 2u];
    }
    let rel = vec3<f32>(f32(tx), f32(ty), f32(tz));

    let r0 = world_abs_linear[b];
    let r1 = world_abs_linear[b + 1u];
    let r2 = world_abs_linear[b + 2u];
    world_rel[b]      = vec4<f32>(r0.xyz, rel.x);
    world_rel[b + 1u] = vec4<f32>(r1.xyz, rel.y);
    world_rel[b + 2u] = vec4<f32>(r2.xyz, rel.z);
}
