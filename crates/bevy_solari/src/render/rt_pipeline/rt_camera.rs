//! GPU-authoritative camera basis pass (`rt_camera.wgsl`).
//!
//! A one-thread compute pass that derives the per-view [`RtCamera`](super::RtCamera)
//! from the camera's own transform-table slot (`world[camera_slot]`) instead of a CPU
//! `GlobalTransform`. Runs after `transform_propagate` and before the view's RT trace,
//! writing the same `RtCamera` byte layout the raygen + DLSS passes already read — so
//! only the *source* of the camera basis moves onto the GPU, not its consumers.
//!
//! Per-view: each [`RtViewBindings`](crate::gpu::rt_pipeline::RtViewBindings) owns its
//! output `RtCamera` buffer + a params uniform, so multi-camera / split-screen setups
//! each derive their own basis from their own slot. The projection (CPU-authored) and
//! the per-frame scalars (jitter/frame/sky) are the only inputs that still cross from
//! the CPU, carried in [`RtCameraPassParams`].

use bevy_math::{Mat4, UVec4, Vec4};
use bevy_render::render_resource::{
    binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
    BindGroupLayoutDescriptor, BindGroupLayoutEntries, ShaderStages, ShaderType,
};

/// Uniform shared with `rt_camera.wgsl::CameraPassParams`. Field order + types must
/// match; the std140 layout is: three `mat4x4` (projection, its inverse, previous
/// clip-from-world), then `frame`/`sky`/`jitter` (`vec4` each), then the four trailing
/// scalars packed into one 16-byte slot.
#[derive(Clone, Copy, ShaderType)]
pub struct RtCameraPassParams {
    /// Projection (CPU-authored). `clip_from_world = clip_from_view · view_from_world`.
    pub clip_from_view: Mat4,
    /// Inverse projection. `inverse_view_proj = world_from_view · view_from_clip`.
    pub view_from_clip: Mat4,
    /// Floating-origin recenter rebase for the previous basis (identity unless
    /// `reframe_active`); the GPU keeps the previous `clip_from_world` itself now.
    pub reframe_prev_from_current: Mat4,
    /// Passthrough → `RtCamera.frame` (`.x` RNG seed, `.y` SER hint bits, `.z` debug view).
    pub frame: UVec4,
    /// Passthrough → `RtCamera.sky` (`.x` env brightness, `.yzw` clear color).
    pub sky: Vec4,
    /// Passthrough → `RtCamera.jitter` (`.xy` sub-pixel jitter).
    pub jitter: Vec4,
    /// Passthrough → `RtCamera.misc` (`.x` time seconds, `.y` pixel ray-cone tan).
    pub misc: Vec4,
    /// Passthrough → `RtCamera.sky_frame` (world→bake sky quaternion, xyzw).
    pub sky_frame: Vec4,
    /// Passthrough → `RtCamera.atmo` (volume buffer address bits + count).
    pub atmo: Vec4,
    /// Passthrough → `RtCamera.dims` (`.xy` viewport pixels, `.z` restir M-cap).
    pub dims: Vec4,
    /// Passthrough → `RtCamera.window_arc` (cylindrical window: arc, radius, height).
    pub window_arc: Vec4,
    /// Passthrough → `RtCamera.window_eye` (`.xyz` eye in screen space).
    pub window_eye: Vec4,
    /// The `SolariCamera`'s transform-table slot (`GpuSlot<TransformGraph>`).
    pub camera_slot: u32,
    /// World-buffer node high-water — bounds guard for `camera_slot`.
    pub node_count: u32,
    /// `camera_position.w` — camera exposure (raygen scales final radiance by it).
    pub exposure: f32,
    /// 1 = the camera slot is live + propagated this frame; 0 (cold start / slot not
    /// yet allocated) → the shader writes an identity basis instead of reading
    /// undefined `world`.
    pub valid: u32,
    /// 1 = apply `reframe_prev_from_current` to the stored previous basis this frame
    /// (a floating-origin recenter happened).
    pub reframe_active: u32,
}

/// The `rt_camera` bind-group layout. Owned by
/// [`SolariResourceManager`](crate::resource_manager::SolariResourceManager); kept
/// here next to [`RtCameraPassParams`] and the shader it must match.
pub fn rt_camera_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "rt_camera",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 world (transform table)
                uniform_buffer::<RtCameraPassParams>(false), // 1 params
                storage_buffer_sized(false, None),           // 2 out_camera (rw, RtCamera)
                storage_buffer_sized(false, None),           // 3 prev_cam (rw, persistent)
                storage_buffer_read_only_sized(false, None), // 4 world_abs_t (f64 origin source)
            ),
        ),
    )
}
