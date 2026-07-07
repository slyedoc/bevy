// GPU-authoritative camera basis. One thread reads the camera's world transform
// straight from the transform table (`world[camera_slot]`, the same buffer the
// scene instances are built from) and derives the full `RtCamera` the raygen +
// DLSS passes read. This makes the camera "just another transform-table node":
// its view is composed on the GPU through any hierarchy + floating-origin cell
// offset, same-frame, with no dependence on a CPU `GlobalTransform` — so a camera
// childed to a moving parent (ship, station) renders correctly with no
// leaf rule and no `NoGpuGlobalTransformReadback`. Only the projection (CPU-
// authored) and the per-frame scalars (jitter/frame/sky) cross from the CPU, in
// `CameraPassParams`. Runs after `transform_propagate`, before the RT trace reads
// the produced buffer as set-1 `camera`.

// Mirror of `rt_payload.wgsl::RtCamera` — defined inline (not `#import`ed) because that
// module is consumed by the raw-VK RT path, not registered as a naga_oil library for
// this `PipelineCache` compute pipeline. All members are 16-byte-aligned, so the std430
// layout written here matches the std140 layout the trace reads. Keep in lockstep.
struct RtCamera {
    inverse_view_proj: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
    prev_clip_from_world: mat4x4<f32>,
    camera_position: vec4<f32>,
    frame: vec4<u32>,
    sky: vec4<f32>,
    jitter: vec4<f32>,
    misc: vec4<f32>,
    sky_frame: vec4<f32>,
    atmo: vec4<f32>,
    dims: vec4<f32>,
    world_from_view: mat4x4<f32>,
    window_arc: vec4<f32>,
    window_eye: vec4<f32>,
}

struct CameraPassParams {
    clip_from_view: mat4x4<f32>,             // projection (CPU-authored)
    view_from_clip: mat4x4<f32>,             // inverse projection (CPU-authored)
    // Floating-origin recenter rebase for DLSS motion vectors: on a jump frame the
    // stored previous `clip_from_world` is in the OLD origin basis, so it's post-
    // multiplied by this to re-express it in the new basis (identity + `reframe_active`
    // 0 on ordinary frames). Matches the CPU path's `prev *= reframe.prev_from_current`.
    reframe_prev_from_current: mat4x4<f32>,
    frame: vec4<u32>,                        // passthrough → RtCamera.frame
    sky: vec4<f32>,                          // passthrough → RtCamera.sky
    jitter: vec4<f32>,                       // passthrough → RtCamera.jitter
    misc: vec4<f32>,                         // passthrough → RtCamera.misc (time, cone tan)
    sky_frame: vec4<f32>,                    // passthrough → RtCamera.sky_frame (world→bake quat)
    atmo: vec4<f32>,                         // passthrough → RtCamera.atmo (volume addr bits + count)
    dims: vec4<f32>,                         // passthrough → RtCamera.dims (viewport px + restir M-cap)
    window_arc: vec4<f32>,                      // passthrough → RtCamera.window_arc (arc, radius, height)
    window_eye: vec4<f32>,                      // passthrough → RtCamera.window_eye (eye in screen space)
    camera_slot: u32,                        // the SolariCamera's transform-table slot
    node_count: u32,                         // world-buffer node high-water (bounds guard)
    exposure: f32,                           // → camera_position.w
    // 1 = the camera slot is allocated + propagated this frame; 0 (cold start /
    // slot not yet live) → write an identity basis so the first frame is a stable
    // camera-at-origin instead of reading undefined `world`.
    valid: u32,
    reframe_active: u32,                     // 1 = apply `reframe_prev_from_current` this frame
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
    clip_from_world: mat4x4<f32>,
    origin_x: f64,
    origin_y: f64,
    origin_z: f64,
    valid: u32,
}

// The transform table's world buffer: `mat3x4<f32>` per node, stored as three
// rows (`row_k = (linear row k .xyz, translation k .w)`) — the same packing the
// readback gather decodes. 48 B/node ⇒ 3 `vec4`s.
@group(0) @binding(0) var<storage, read> world: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: CameraPassParams;
@group(0) @binding(2) var<storage, read_write> out_camera: RtCamera;
@group(0) @binding(3) var<storage, read_write> prev_cam: PrevCamera;
// The absolute-world translation buffer (flat array<f64>, 3 per node) — the camera's
// own entry IS the floating origin; its frame-to-frame delta drives the motion-vector
// rebase above.
@group(0) @binding(4) var<storage, read> world_abs_t: array<f64>;

// Inverse of an affine transform (last row implicitly `(0,0,0,1)`). Handles a
// scaled camera basis (general 3×3 inverse via the column cross-products), not
// just rigid — a camera under a scaled parent frame still inverts correctly.
fn inverse_affine(m: mat4x4<f32>) -> mat4x4<f32> {
    let a = m[0].xyz; // column 0 of the upper 3×3
    let b = m[1].xyz;
    let c = m[2].xyz;
    let t = m[3].xyz;
    // Rows of the 3×3 inverse: (b×c, c×a, a×b) / det, det = a·(b×c).
    let row0 = cross(b, c);
    let row1 = cross(c, a);
    let row2 = cross(a, b);
    let inv_det = 1.0 / dot(a, row0);
    let ri0 = row0 * inv_det;
    let ri1 = row1 * inv_det;
    let ri2 = row2 * inv_det;
    // Reassemble column-major (column j = (ri0[j], ri1[j], ri2[j])).
    let r_inv = mat3x3<f32>(
        vec3<f32>(ri0.x, ri1.x, ri2.x),
        vec3<f32>(ri0.y, ri1.y, ri2.y),
        vec3<f32>(ri0.z, ri1.z, ri2.z),
    );
    let inv_t = -(r_inv * t);
    return mat4x4<f32>(
        vec4<f32>(r_inv[0], 0.0),
        vec4<f32>(r_inv[1], 0.0),
        vec4<f32>(r_inv[2], 0.0),
        vec4<f32>(inv_t, 1.0),
    );
}

@compute @workgroup_size(1, 1, 1)
fn rt_camera() {
    var world_from_view: mat4x4<f32>;
    var origin: vec3<f32>;
    if params.valid == 1u && params.camera_slot < params.node_count {
        // Reconstruct the column-major 4×4 from the three stored rows.
        let base = params.camera_slot * 3u;
        let r0 = world[base + 0u];
        let r1 = world[base + 1u];
        let r2 = world[base + 2u];
        world_from_view = mat4x4<f32>(
            vec4<f32>(r0.x, r1.x, r2.x, 0.0),
            vec4<f32>(r0.y, r1.y, r2.y, 0.0),
            vec4<f32>(r0.z, r1.z, r2.z, 0.0),
            vec4<f32>(r0.w, r1.w, r2.w, 1.0),
        );
        origin = vec3<f32>(r0.w, r1.w, r2.w);
    } else {
        world_from_view = mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0, 0.0, 0.0, 1.0),
        );
        origin = vec3<f32>(0.0);
    }

    // This frame's absolute origin (the camera's own f64 world translation).
    var origin_x = f64(0.0);
    var origin_y = f64(0.0);
    var origin_z = f64(0.0);
    if params.valid == 1u && params.camera_slot < params.node_count {
        let ob = params.camera_slot * 3u;
        origin_x = world_abs_t[ob];
        origin_y = world_abs_t[ob + 1u];
        origin_z = world_abs_t[ob + 2u];
    }

    let view_from_world = inverse_affine(world_from_view);
    // Same derivations the CPU path did from `view.world_from_view`, now from the
    // GPU-composed camera world: raygen's `inverse_view_proj` = world_from_clip.
    let clip_from_world = params.clip_from_view * view_from_world;
    out_camera.inverse_view_proj = world_from_view * params.view_from_clip;
    out_camera.view_from_world = view_from_world;
    out_camera.clip_from_world = clip_from_world;

    // Previous basis for DLSS motion vectors, maintained on the GPU. Frame 1 (no stored
    // previous) reports zero motion by using this frame's basis. The stored previous is
    // in LAST frame's origin basis: a point's previous origin-relative position is its
    // current one plus (origin_now − origin_prev), so fold that translation in — computed
    // in f64 (the absolute origins are huge; their difference is small) then narrowed.
    // Without this, camera translation produces ~zero motion vectors on static geometry.
    // The CPU-supplied reframe (teleports / explicit frame handoffs) composes after it.
    var prev = clip_from_world;
    if prev_cam.valid == 1u {
        prev = prev_cam.clip_from_world;
        let dx = f32(origin_x - prev_cam.origin_x);
        let dy = f32(origin_y - prev_cam.origin_y);
        let dz = f32(origin_z - prev_cam.origin_z);
        let origin_shift = mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(dx, dy, dz, 1.0),
        );
        prev = prev * origin_shift;
        if params.reframe_active == 1u {
            prev = prev * params.reframe_prev_from_current;
        }
    }
    out_camera.prev_clip_from_world = prev;
    prev_cam.clip_from_world = clip_from_world;
    prev_cam.origin_x = origin_x;
    prev_cam.origin_y = origin_y;
    prev_cam.origin_z = origin_z;
    prev_cam.valid = 1u;

    out_camera.world_from_view = world_from_view;
    out_camera.window_arc = params.window_arc;
    out_camera.window_eye = params.window_eye;
    out_camera.camera_position = vec4<f32>(origin, params.exposure);
    out_camera.frame = params.frame;
    out_camera.sky = params.sky;
    out_camera.jitter = params.jitter;
    out_camera.misc = params.misc;
    out_camera.sky_frame = params.sky_frame;
    out_camera.atmo = params.atmo;
    out_camera.dims = params.dims;
}
