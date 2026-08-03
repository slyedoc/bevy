// GPU-authoritative camera basis. One thread reads the camera's world transform
// straight from the transform table (`world[camera_slot]`, the same buffer the
// scene instances are built from) and derives the full `RtCamera` the raygen +
// DLSS passes read. This makes the camera "just another transform-table node":
// its view is composed on the GPU through any hierarchy + floating-origin cell
// offset, same-frame, with no dependence on a CPU `GlobalTransform` — so a camera
// childed to a moving parent (ship, station) renders correctly. Only the
// projection (CPU-authored) and the per-frame scalars (jitter/frame/sky) cross
// from the CPU, in `CameraPassParams`. Runs after `transform_propagate`, before
// the RT trace reads the produced buffer as set-1 `camera`.
//
// `RtCamera`/`Mat4` come from the shared rt_payload module the RT stages read,
// so the produced layout matches the trace's by construction.

import rt_payload;

// CPU-authored inputs (`RtCameraPassParams` in rt_camera.rs, std140 — matrices
// as four column float4s via `Mat4`). Too large for push data, so it rides a
// uniform-buffer heap slot instead of the push params block.
struct CameraPassParams {
    Mat4 clip_from_view;                     // projection (CPU-authored)
    Mat4 view_from_clip;                     // inverse projection (CPU-authored)
    // Floating-origin recenter rebase for DLSS motion vectors: on a jump frame the
    // stored previous `clip_from_world` is in the OLD origin basis, so it's post-
    // multiplied by this to re-express it in the new basis (identity + `reframe_active`
    // 0 on ordinary frames). Matches the CPU path's `prev *= reframe.prev_from_current`.
    Mat4 reframe_prev_from_current;
    uint4 frame;                             // passthrough → RtCamera.frame
    float4 sky;                              // passthrough → RtCamera.sky
    float4 jitter;                           // passthrough → RtCamera.jitter
    float4 misc;                             // passthrough → RtCamera.misc (time, cone tan)
    float4 sky_frame;                        // passthrough → RtCamera.sky_frame (world→bake quat)
    float4 atmo;                             // passthrough → RtCamera.atmo (volume addr bits + count)
    float4 dims;                             // passthrough → RtCamera.dims (viewport px + restir M-cap)
    float4 window_arc;                       // passthrough → RtCamera.window_arc (arc, radius, height)
    float4 window_eye;                       // passthrough → RtCamera.window_eye (eye in screen space)
    float4 nrc;                              // passthrough → RtCamera.nrc (scene scale)
    uint camera_slot;                        // the SolariCamera's transform-table slot
    uint node_count;                         // world-buffer node high-water (bounds guard)
    float exposure;                          // → camera_position.w (tooling only; the trace stays physical)
    // 1 = the camera slot is allocated + propagated this frame; 0 (cold start /
    // slot not yet live) → write an identity basis so the first frame is a stable
    // camera-at-origin instead of reading undefined `world`.
    uint valid;
    uint reframe_active;                     // 1 = apply `reframe_prev_from_current` this frame
}

// Persistent previous camera basis (one per view), maintained entirely on the GPU: the
// pass reads last frame's value, emits it as `RtCamera.prev_clip_from_world` (for DLSS
// motion vectors), then overwrites it with this frame's. `valid` is 0 until the first
// frame writes it (buffer zero-cleared on creation), so frame 1 reports zero motion
// instead of reading an uninitialized previous.
//
// `origin_*` is the ABSOLUTE floating origin (the camera's own f64 world translation)
// the stored basis was expressed against. The origin moves every frame the camera
// does, so the stored previous is in a *shifted* coordinate space: without a rebase,
// clip_from_world and prev_clip_from_world would differ only by rotation and static
// geometry would report ~zero motion under camera translation (DLSS smear). The next
// frame folds `origin_now − origin_prev` back in (below).
struct PrevCamera {
    Mat4 clip_from_world;
    double origin_x;
    double origin_y;
    double origin_z;
    uint valid;
    uint anchor_valid;
    // The held NRC anchor (absolute, f64): kept until the camera leaves its
    // cell by a margin — a bare round() flips at every boundary crossing.
    double anchor_x;
    double anchor_y;
    double anchor_z;
}

// The transform table's world buffer: `mat3x4<f32>` per node, stored as three
// rows (`row_k = (linear row k .xyz, translation k .w)`) — the same packing the
// readback gather decodes. 48 B/node ⇒ 3 `float4`s.
[[vk::binding(0, 0)]] StructuredBuffer<float4> world;
[[vk::binding(1, 0)]] ConstantBuffer<CameraPassParams> params;
[[vk::binding(2, 0)]] RWStructuredBuffer<RtCamera> out_camera;
[[vk::binding(3, 0)]] RWStructuredBuffer<PrevCamera> prev_cam;
// The absolute-world translation buffer (flat array of f64, 3 per node) — the camera's
// own entry IS the floating origin; its frame-to-frame delta drives the motion-vector
// rebase above.
[[vk::binding(4, 0)]] StructuredBuffer<double> world_abs_t;

// WGSL `a * b` on column-packed Mat4s (column j of the product = a · b.cj).
Mat4 mat4_mat4_mul(Mat4 a, Mat4 b) {
    Mat4 r;
    r.c0 = mat4_mul(a, b.c0);
    r.c1 = mat4_mul(a, b.c1);
    r.c2 = mat4_mul(a, b.c2);
    r.c3 = mat4_mul(a, b.c3);
    return r;
}

Mat4 mat4_identity() {
    Mat4 m;
    m.c0 = float4(1.0, 0.0, 0.0, 0.0);
    m.c1 = float4(0.0, 1.0, 0.0, 0.0);
    m.c2 = float4(0.0, 0.0, 1.0, 0.0);
    m.c3 = float4(0.0, 0.0, 0.0, 1.0);
    return m;
}

// Inverse of an affine transform (last row implicitly `(0,0,0,1)`). Handles a
// scaled camera basis (general 3×3 inverse via the column cross-products), not
// just rigid — a camera under a scaled parent frame still inverts correctly.
Mat4 inverse_affine(Mat4 m) {
    let a = m.c0.xyz; // column 0 of the upper 3×3
    let b = m.c1.xyz;
    let c = m.c2.xyz;
    let t = m.c3.xyz;
    // Rows of the 3×3 inverse: (b×c, c×a, a×b) / det, det = a·(b×c).
    let row0 = cross(b, c);
    let row1 = cross(c, a);
    let row2 = cross(a, b);
    let inv_det = 1.0 / dot(a, row0);
    let ri0 = row0 * inv_det;
    let ri1 = row1 * inv_det;
    let ri2 = row2 * inv_det;
    // Reassemble column-major (column j = (ri0[j], ri1[j], ri2[j])).
    let inv_t = -(ri0 * t.x + ri1 * t.y + ri2 * t.z);
    Mat4 r;
    r.c0 = float4(ri0.x, ri1.x, ri2.x, 0.0);
    r.c1 = float4(ri0.y, ri1.y, ri2.y, 0.0);
    r.c2 = float4(ri0.z, ri1.z, ri2.z, 0.0);
    r.c3 = float4(inv_t, 1.0);
    return r;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void rt_camera() {
    Mat4 world_from_view;
    float3 origin;
    if (params.valid == 1 && params.camera_slot < params.node_count) {
        // Reconstruct the column-major 4×4 from the three stored rows.
        let base = params.camera_slot * 3;
        let r0 = world[base + 0];
        let r1 = world[base + 1];
        let r2 = world[base + 2];
        world_from_view.c0 = float4(r0.x, r1.x, r2.x, 0.0);
        world_from_view.c1 = float4(r0.y, r1.y, r2.y, 0.0);
        world_from_view.c2 = float4(r0.z, r1.z, r2.z, 0.0);
        world_from_view.c3 = float4(r0.w, r1.w, r2.w, 1.0);
        origin = float3(r0.w, r1.w, r2.w);
    } else {
        world_from_view = mat4_identity();
        origin = float3(0.0);
    }

    // This frame's absolute origin (the camera's own f64 world translation).
    var origin_x = double(0.0);
    var origin_y = double(0.0);
    var origin_z = double(0.0);
    if (params.valid == 1 && params.camera_slot < params.node_count) {
        let ob = params.camera_slot * 3;
        origin_x = world_abs_t[ob];
        origin_y = world_abs_t[ob + 1];
        origin_z = world_abs_t[ob + 2];
    }

    let view_from_world = inverse_affine(world_from_view);
    // Same derivations the CPU path did from `view.world_from_view`, now from the
    // GPU-composed camera world: raygen's `inverse_view_proj` = world_from_clip.
    let clip_from_world = mat4_mat4_mul(params.clip_from_view, view_from_world);
    out_camera[0].inverse_view_proj = mat4_mat4_mul(world_from_view, params.view_from_clip);
    out_camera[0].view_from_world = view_from_world;
    out_camera[0].clip_from_world = clip_from_world;

    // Previous basis for DLSS motion vectors, maintained on the GPU. Frame 1 (no stored
    // previous) reports zero motion by using this frame's basis. The stored previous is
    // in LAST frame's origin basis: a point's previous origin-relative position is its
    // current one plus (origin_now − origin_prev), so fold that translation in — computed
    // in f64 (the absolute origins are huge; their difference is small) then narrowed.
    // Without this, camera translation produces ~zero motion vectors on static geometry.
    // The CPU-supplied reframe (teleports / explicit frame handoffs) composes after it.
    var prev = clip_from_world;
    var origin_delta = float3(0.0);
    if (prev_cam[0].valid == 1) {
        prev = prev_cam[0].clip_from_world;
        let dx = float(origin_x - prev_cam[0].origin_x);
        let dy = float(origin_y - prev_cam[0].origin_y);
        let dz = float(origin_z - prev_cam[0].origin_z);
        origin_delta = float3(dx, dy, dz);
        var origin_shift = mat4_identity();
        origin_shift.c3 = float4(dx, dy, dz, 1.0);
        prev = mat4_mat4_mul(prev, origin_shift);
        if (params.reframe_active == 1) {
            prev = mat4_mat4_mul(prev, params.reframe_prev_from_current);
        }
    }
    // Cross-frame reservoir positions (camera-origin space) rebase by this.
    out_camera[0].origin_delta = float4(origin_delta, 0.0);
    out_camera[0].prev_clip_from_world = prev;
    prev_cam[0].clip_from_world = clip_from_world;
    prev_cam[0].origin_x = origin_x;
    prev_cam[0].origin_y = origin_y;
    prev_cam[0].origin_z = origin_z;
    prev_cam[0].valid = 1;

    // NRC world-snapped anchor: the position encoding centers on the camera's
    // absolute position snapped to a scene_scale/2 grid, so translation
    // doesn't re-encode every surface (the cache relearns only on the rare
    // anchor hop). `.xyz` = f32(abs_origin − anchor) — bounded by the grid
    // cell, so exact in f32; raygen adds it to camera-relative positions to
    // make them anchor-relative. The cell index rounds in f32: beyond ~1e7
    // cells from the world origin the anchor coarsens gracefully.
    var nrc_anchor = float3(0.0);
    if (params.nrc.x > 0.0) {
        let grid = double(params.nrc.x * 0.5);
        var ax = double(round(float(origin_x / grid))) * grid;
        var ay = double(round(float(origin_y / grid))) * grid;
        var az = double(round(float(origin_z / grid))) * grid;
        // Hysteresis: hold the previous anchor until the camera has left its
        // cell by a quarter-grid margin. A bare round() re-anchors at every
        // boundary crossing — hovering near one re-encodes the entire world
        // each wiggle, and the cache whiplashes between disjoint domains
        // (random tint per hop; TD blowout if hops come fast).
        if (prev_cam[0].anchor_valid == 1) {
            let hx = abs(float(origin_x - prev_cam[0].anchor_x));
            let hy = abs(float(origin_y - prev_cam[0].anchor_y));
            let hz = abs(float(origin_z - prev_cam[0].anchor_z));
            let limit = 0.75 * float(grid);
            if (hx < limit && hy < limit && hz < limit) {
                ax = prev_cam[0].anchor_x;
                ay = prev_cam[0].anchor_y;
                az = prev_cam[0].anchor_z;
            }
        }
        prev_cam[0].anchor_x = ax;
        prev_cam[0].anchor_y = ay;
        prev_cam[0].anchor_z = az;
        prev_cam[0].anchor_valid = 1;
        nrc_anchor = float3(
            float(origin_x - ax),
            float(origin_y - ay),
            float(origin_z - az));
    }
    out_camera[0].nrc_anchor = float4(nrc_anchor, 0.0);

    out_camera[0].world_from_view = world_from_view;
    out_camera[0].window_arc = params.window_arc;
    out_camera[0].nrc = params.nrc;
    out_camera[0].window_eye = params.window_eye;
    out_camera[0].camera_position = float4(origin, params.exposure);
    out_camera[0].frame = params.frame;
    out_camera[0].sky = params.sky;
    out_camera[0].jitter = params.jitter;
    out_camera[0].misc = params.misc;
    out_camera[0].sky_frame = params.sky_frame;
    out_camera[0].atmo = params.atmo;
    out_camera[0].dims = params.dims;
}
