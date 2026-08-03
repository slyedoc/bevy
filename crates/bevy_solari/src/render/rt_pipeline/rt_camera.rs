//! GPU-authoritative camera basis pass (`rt_camera.slang`).
//!
//! A one-thread compute pass that derives the per-view [`RtCamera`](super::RtCamera)
//! from the camera's own transform-table slot (`world[camera_slot]`) instead of a CPU
//! `GlobalTransform`. Runs after `transform_propagate` and before the view's RT trace,
//! writing the same `RtCamera` byte layout the raygen + DLSS passes already read — so
//! only the *source* of the camera basis moves onto the GPU, not its consumers.
//!
//! Per-view: each view's [`RtOutputBuffer`](super::RtOutputBuffer) owns its output
//! `RtCamera` buffer, and its [`RtViewKernelSlots`](super::RtViewKernelSlots) the
//! params uniform + heap slots, so multi-camera / split-screen setups each derive
//! their own basis from their own slot. The projection (CPU-authored) and the
//! per-frame scalars (jitter/frame/sky) are the only inputs that still cross from
//! the CPU, carried in [`RtCameraPassParams`].

#![allow(unsafe_code)]

use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res},
};
use bevy_math::{Mat4, UVec4, Vec4};
use bevy_render::render_resource::ShaderType;

use crate::gpu::allocator::Allocator;
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::heap_kernel::HeapKernel;

/// Uniform shared with `rt_camera.slang::CameraPassParams`. Field order + types must
/// match; the std140 layout is: three `mat4x4` (projection, its inverse, previous
/// clip-from-world), then `frame`/`sky`/`jitter` (`vec4` each), then the four trailing
/// scalars packed into one 16-byte slot. Encase-encoded into the per-view params
/// uniform buffer — at 384 B the block is too large for push data, so it rides a
/// uniform-buffer heap slot instead.
#[derive(Clone, Copy, ShaderType)]
pub struct RtCameraPassParams {
    /// Projection (CPU-authored). `clip_from_world = clip_from_view · view_from_world`.
    pub clip_from_view: Mat4,
    /// Inverse projection. `inverse_view_proj = world_from_view · view_from_clip`.
    pub view_from_clip: Mat4,
    /// Floating-origin recenter rebase for the previous basis (identity unless
    /// `reframe_active`); the GPU keeps the previous `clip_from_world` itself.
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
    /// Passthrough → `RtCamera.nrc` (`.x` scene scale m; 0 = off).
    pub nrc: Vec4,
    /// The `SolariCamera`'s transform-table slot (`GpuSlot<TransformGraph>`).
    pub camera_slot: u32,
    /// World-buffer node high-water — bounds guard for `camera_slot`.
    pub node_count: u32,
    /// `camera_position.w` — camera exposure (tooling only; the trace outputs
    /// physical radiance and the blit applies exposure at read).
    pub exposure: f32,
    /// 1 = the camera slot is live + propagated this frame; 0 (cold start / slot not
    /// yet allocated) → the shader writes an identity basis instead of reading
    /// undefined `world`.
    pub valid: u32,
    /// 1 = apply `reframe_prev_from_current` to the stored previous basis this frame
    /// (a floating-origin recenter happened).
    pub reframe_active: u32,
}

/// Render-world resource: the rt_camera heap kernel. The per-view heap slots
/// live on each view's [`RtViewKernelSlots`](super::RtViewKernelSlots) — slot
/// descriptors resolve at execution, so per-view dispatches can't share one
/// slot set.
#[derive(Resource)]
pub struct RtCameraKernel {
    pub(super) kernel: HeapKernel,
    pub(super) raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for RtCameraKernel {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for RtCameraKernel {}
unsafe impl Sync for RtCameraKernel {}

/// `RenderStartup` (after `SolariSetup`): compile the rt_camera kernel — a
/// layout-free heap pipeline ([`HeapKernel`]), Slang from source. No push
/// params: the [`RtCameraPassParams`] block rides a uniform-buffer heap slot.
pub fn init_rt_camera(
    mut commands: Commands,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let Some(kernel) = HeapKernel::new(
        &seam,
        "rt_camera.slang",
        include_str!("rt_camera.slang"),
        "rt_camera",
        &[("rt_payload", include_str!("rt_payload.slang"))],
        &[],
        "rt_camera",
        0,
    ) else {
        return;
    };
    commands.insert_resource(RtCameraKernel {
        kernel,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

/// The params uniform's byte size (std140), for the per-view buffer allocation.
pub(super) fn rt_camera_params_size() -> u64 {
    RtCameraPassParams::min_size().get()
}
