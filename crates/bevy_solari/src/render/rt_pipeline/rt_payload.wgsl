// WGSL mirror of `rt_payload.slang`'s `RtCamera` for the remaining wgpu compute
// passes (`restir_spatial` #imports it). The RT stages themselves are Slang and
// import rt_payload.slang; the CPU writes the camera UBO against this layout,
// so the two declarations MUST stay field-for-field identical (std140: matrices
// @0/64/128/192, vec4s @256..368, world_from_view @384, vec4s @448..512).
#define_import_path bevy_solari::rt_payload

struct RtCamera {
    inverse_view_proj: mat4x4<f32>,
    view_from_world: mat4x4<f32>,        // DLSS guide: view-space linear depth
    clip_from_world: mat4x4<f32>,        // DLSS motion vectors: current (unjittered)
    prev_clip_from_world: mat4x4<f32>,   // DLSS motion vectors: previous (unjittered)
    camera_position: vec4<f32>,
    frame: vec4<u32>,          // .x = frame index (RNG seed); .y = SER material-hint bits; .z = debug view
    sky: vec4<f32>,            // .x = environment brightness (cd/m²); .yzw = clear color
    jitter: vec4<f32>,         // .xy = sub-pixel camera jitter (pixels); .zw = debug-heatmap colormap params
    misc: vec4<f32>,           // .x = time (s, wrapped); .y = pixel ray-cone tan (footprint LOD); .zw reserved
    sky_frame: vec4<f32>,      // world→bake sky quaternion (xyzw); identity for flat scenes/skyboxes
    atmo: vec4<f32>,           // .xy = volume-buffer device address (lo/hi bits); .z = volume count
    dims: vec4<f32>,           // .xy = viewport pixels (ReSTIR reprojection); .z = history M-cap ×M
    world_from_view: mat4x4<f32>, // camera basis: view→world ray dirs (cylindrical window)
    window_arc: vec4<f32>,        // .x = arc angle (rad); .y = radius m (0 = flat); .z = height m
    window_eye: vec4<f32>,        // .xyz = eye in screen space (center origin, +Z toward viewer)
    origin_delta: vec4<f32>,      // .xyz = origin_now − origin_prev: cross-frame position rebase
    nrc: vec4<f32>,               // .x = NRC scene scale m (0 = off); .y = spread c; .z = inline coopvec
    nrc_anchor: vec4<f32>,        // .xyz = camera-relative → anchor-relative position offset
}
