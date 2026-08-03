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
    uint count;
    // The primary camera's transform-table slot — `world_abs_t[camera_slot]` is the origin.
    uint camera_slot;
    // World-buffer node high-water (bounds guard for `camera_slot`).
    uint node_count;
    // 1 = a camera slot is live this frame → subtract the origin; 0 → pass the absolute
    // world through unchanged (no floating origin yet, e.g. cold start).
    uint origin_valid;
    // X workgroup count of the 2D-split dispatch (X capped at 65535, rest in Y).
    uint groups_x;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}

[[vk::push_constant]] SubtractParams params;

[[vk::binding(0, 0)]] StructuredBuffer<float4> world_abs_linear; // 3 per node
[[vk::binding(1, 0)]] StructuredBuffer<double> world_abs_t;      // 3 per node
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> world_rel;      // 3 per node (mat3x4)

[shader("compute")]
[numthreads(64, 1, 1)]
void subtract(uint3 gid: SV_DispatchThreadID) {
    let k = gid.x + gid.y * params.groups_x * 64;
    if (k >= params.count) {
        return;
    }

    let b = k * 3;
    var tx = world_abs_t[b];
    var ty = world_abs_t[b + 1];
    var tz = world_abs_t[b + 2];
    if (params.origin_valid == 1 && params.camera_slot < params.node_count) {
        let ob = params.camera_slot * 3;
        tx = tx - world_abs_t[ob];
        ty = ty - world_abs_t[ob + 1];
        tz = tz - world_abs_t[ob + 2];
    }
    let rel = float3(float(tx), float(ty), float(tz));

    let r0 = world_abs_linear[b];
    let r1 = world_abs_linear[b + 1];
    let r2 = world_abs_linear[b + 2];
    world_rel[b] = float4(r0.xyz, rel.x);
    world_rel[b + 1] = float4(r1.xyz, rel.y);
    world_rel[b + 2] = float4(r2.xyz, rel.z);
}
