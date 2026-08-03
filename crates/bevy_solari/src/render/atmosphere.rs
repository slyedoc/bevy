//! Self-contained single-scattering atmosphere → sky cubemap for the pathtracer.
//!
//! A compute pre-pass (`atmosphere_bake.slang`) bakes the atmosphere
//! into a cube from [`SolariAtmosphere`] + the primary [`SolariDirectionLight`];
//! the pathtracer samples that cube on a ray miss (the skybox path).
//! The cube persists, so the bake re-runs only when its inputs change (a moving
//! sun re-bakes; a static sky is free). Fully solari-owned — no `bevy_pbr`
//! atmosphere / raster `GpuLights` coupling — so it works with `PbrPlugin`
//! disabled.

#![allow(unsafe_code)]

use ash::vk;
use bevy_app::{App, Plugin};
use bevy_core_pipeline::{
    core_3d::main_opaque_pass_3d,
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::{
    component::Component,
    query::With,
    resource::Resource,
    schedule::{common_conditions::resource_exists, IntoScheduleConfigs},
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::{Mat3, Quat, Vec3};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{
    render_resource::{
        CommandEncoderDescriptor, ShaderType, TextureDescriptor, TextureDimension, TextureFormat,
        TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_transform::components::GlobalTransform;
use wgpu::hal::api::Vulkan as VkApi;

use crate::gpu::allocator::Allocator;
use crate::gpu::binding_seam::{BindingSeam, HeapKind};
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
use crate::lights::SolariDirectionLight;
use crate::pipelines::SolariPipelines;
use crate::render::SolariCamera;
use crate::SolariSetup;

/// Opt-in atmosphere support: registers the [`SolariAtmosphere`] /
/// [`SolariGlobalFog`] / [`SolariAtmosphereVolume`] extract → bake → bind
/// systems. Add after [`SolariPlugin`](crate::SolariPlugin); without this
/// plugin those components are inert and solari registers none of the
/// atmosphere systems or GPU state.
pub struct SolariAtmospherePlugin;

impl Plugin for SolariAtmospherePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<SolariAtmosphere>()
            .register_type::<SolariGlobalFog>()
            .register_type::<SolariAtmosphereVolume>();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<SolariAtmosphereGpu>()
            .init_resource::<SolariAtmosphereVolumesGpu>()
            .add_systems(
                RenderStartup,
                (init_atmosphere_bake, init_atmosphere_lut_bake).after(SolariSetup),
            )
            .add_systems(
                ExtractSchedule,
                (extract_solari_atmosphere, extract_atmosphere_volumes),
            )
            .add_systems(
                Render,
                (prepare_atmosphere_sky, prepare_atmosphere_volumes)
                    .in_set(RenderSystems::PrepareResources),
            )
            // Bake before the trace consumes the cube (same MainPass slot as the
            // trace; see `SolarRenderPlugin`'s compose-first ordering).
            .add_systems(
                Core3d,
                (dispatch_atmosphere_bake, dispatch_atmosphere_lut_bake)
                    .chain()
                    .run_if(resource_exists::<SolariPipelines>)
                    .before(super::rt_pipeline::rt_pipeline)
                    .before(main_opaque_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            );
    }
}

/// Edge length of each face of the baked sky cube.
const SKY_SIZE: u32 = 256;

/// Self-contained atmosphere for the solari pathtracer. Add it to a
/// [`SolariCamera`] to bake a single-scattering sky (sampled on a ray miss
/// instead of a skybox), re-baked whenever these parameters or the sun change.
/// Defaults model Earth. Driving sun = the primary [`SolariDirectionLight`]'s
/// direction + illuminance.
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Default, Clone)]
pub struct SolariAtmosphere {
    /// Planet surface radius (km).
    pub bottom_radius: f32,
    /// Atmosphere-top radius (km).
    pub top_radius: f32,
    /// Rayleigh (air) scattering coefficient per channel (1/km) — the blue sky.
    pub rayleigh_scattering: Vec3,
    /// Rayleigh density scale height (km).
    pub rayleigh_scale_height: f32,
    /// Mie (aerosol) scattering coefficient (1/km) — haze + sun halo.
    pub mie_scattering: f32,
    /// Mie extinction coefficient (1/km); ≥ `mie_scattering` (the rest is absorption).
    pub mie_extinction: f32,
    /// Mie density scale height (km).
    pub mie_scale_height: f32,
    /// Mie phase asymmetry `g` in `[0, 1)` — forward-scatter sharpness of the halo.
    pub mie_phase_g: f32,
    /// Camera altitude above the surface (km).
    pub camera_altitude: f32,
    /// World-space "up" at the camera — the planet-radial direction for a
    /// spherical world. The bake keeps its canonical up-=-+Y frame; the miss
    /// shader rotates cube sample directions by the derived `sky_frame` quat,
    /// and the bake sun keeps only its zenith angle (quantized), so orbiting a
    /// planet re-bakes nothing until the sun angle actually changes. `Vec3::Y`
    /// (the default) preserves the flat-world behavior bit-exactly.
    pub up: Vec3,
}

/// Opt-in **global height fog** — the volumetric near-ground haze + sun shafts
/// (god rays) the path tracer marches along each primary ray. Add it to a
/// [`SolariCamera`](crate::SolariCamera) that also has a [`SolariAtmosphere`]
/// (the fog is lit by the atmosphere's sun + sky). **Absent ⇒ the aerial march
/// is skipped entirely**, so a scene pays nothing for it — the sky, sun, and
/// local [`SolariFogVolume`](crate::bindings::SolariFogVolume)s keep working.
///
/// Cost scales with how much of the view sits inside the fog layer: a camera
/// down in the fog traces a sun shadow ray per march step, so a ground-level
/// view is far more expensive than one shooting over the fog.
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Default, Clone)]
pub struct SolariGlobalFog {
    /// **Ground-level visibility** in WORLD units (Koschmieder meteorological
    /// range): the view distance at which a surface *in the densest fog* fades to
    /// ~2 % contrast. Lower ⇒ haze closer/thicker; higher ⇒ clearer. `0.0` (or
    /// non-finite) is treated as the fog being off.
    pub visibility: f32,
    /// Fog-layer **scale height** in WORLD units: density falls off as
    /// `exp(-(y - fog_base) / fog_height)`, so the haze is densest at the ground
    /// and thins with altitude (a low-lying fog the camera shoots over).
    pub fog_height: f32,
    /// World-space Y of the densest fog (ground level).
    pub fog_base: f32,
    /// Henyey-Greenstein asymmetry `g` in `(-1, 1)` for the volumetric sun shafts
    /// (god rays): higher ⇒ a tighter, brighter glow concentrated toward the sun.
    pub phase_g: f32,
}

impl Default for SolariGlobalFog {
    fn default() -> Self {
        // Generic defaults for a metres-scale world: ~12 km ground visibility, a
        // 100 m-tall fog layer at y = 0. Set to your scene's units.
        Self {
            visibility: 12000.0,
            fog_height: 100.0,
            fog_base: 0.0,
            phase_g: 0.4,
        }
    }
}

impl Default for SolariAtmosphere {
    fn default() -> Self {
        // Earth, the standard Bruneton/Hillaire values.
        Self {
            bottom_radius: 6360.0,
            top_radius: 6420.0,
            rayleigh_scattering: Vec3::new(5.802e-3, 13.558e-3, 33.1e-3),
            rayleigh_scale_height: 8.0,
            mie_scattering: 3.996e-3,
            mie_extinction: 4.44e-3,
            mie_scale_height: 1.2,
            mie_phase_g: 0.8,
            camera_altitude: 0.2,
            up: Vec3::Y,
        }
    }
}

/// GPU mirror of [`SolariAtmosphere`] + the sun — matches `Atmosphere` in
/// `atmosphere.slang`. `Default` is the "disabled" state (`aerial_enabled = 0`).
#[derive(Clone, Copy, Default, PartialEq, ShaderType)]
pub struct GpuSolariAtmosphere {
    bottom_radius: f32,
    top_radius: f32,
    rayleigh_scattering: Vec3,
    rayleigh_scale_height: f32,
    mie_scattering: f32,
    mie_extinction: f32,
    mie_scale_height: f32,
    mie_phase_g: f32,
    sun_direction: Vec3,
    sun_illuminance: f32,
    camera_altitude: f32,
    aerial_visibility: f32,
    aerial_fog_height: f32,
    aerial_fog_base: f32,
    aerial_phase_g: f32,
    /// `1.0` when this view has an atmosphere (the pathtracer applies aerial
    /// perspective); `0.0` disables it.
    aerial_enabled: f32,
}

/// Render-world resource: the extracted atmosphere params + whether any solari
/// camera enabled it this frame.
#[derive(Resource, Default)]
pub struct SolariAtmosphereGpu {
    pub enabled: bool,
    /// The last extracted params value, to detect changes; the bake pushes its
    /// encase-encoded bytes as the kernel's push params.
    current: GpuSolariAtmosphere,
    /// The sky cube is stale: the params changed while enabled. Cleared by
    /// [`dispatch_atmosphere_bake`] only once a bake is actually encoded, so a
    /// pending bake survives the kernel/cube coming up late.
    needs_bake: bool,
    /// World→bake rotation for cube sampling (see [`SolariAtmosphere::up`]).
    /// Changes freely per frame WITHOUT re-baking — it rides `RtCamera` to the
    /// miss shader, not the bake uniform. Identity in flat/up-=-+Y scenes.
    pub sky_frame: Quat,
}

// ── World-space atmosphere volumes (spherical planets, marched per pixel) ──

/// A planet's atmosphere as a WORLD-SPACE volume, marched by raygen along each
/// PRIMARY ray (entry→exit-or-hit through the shell): aerial perspective over
/// the terrain, the blue limb from orbit, the sky from inside — all from any
/// viewpoint, per planet. Attach to the PLANET entity (its `GlobalTransform`
/// is the shell center). Complements [`SolariAtmosphere`] (the baked cube),
/// which remains the cheap ambience bounce rays sample.
///
/// Units: radii/scale-heights in KM, scattering coefficients in 1/km — the
/// same convention as [`SolariAtmosphere`]; world units are meters.
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Clone)]
pub struct SolariAtmosphereVolume {
    /// Planet surface radius (km).
    pub bottom_radius: f32,
    /// Atmosphere-top radius (km).
    pub top_radius: f32,
    pub rayleigh_scattering: Vec3,
    pub rayleigh_scale_height: f32,
    pub mie_scattering: f32,
    pub mie_extinction: f32,
    pub mie_scale_height: f32,
    pub mie_phase_g: f32,
}

/// GPU record (mirrors `raygen.slang::AtmoVolume`; scalar layout after the
/// leading vec3s). `center` is the shell center relative to the PRIMARY
/// camera in world METERS — an f64 subtraction on the CPU, so the huge
/// absolute translations never reach f32.
#[derive(Clone, Copy, Default, PartialEq, ShaderType)]
struct GpuAtmosphereVolume {
    center: Vec3,
    bottom_radius: f32,
    rayleigh_scattering: Vec3,
    rayleigh_scale_height: f32,
    top_radius: f32,
    mie_scattering: f32,
    mie_extinction: f32,
    mie_scale_height: f32,
    mie_phase_g: f32,
    pad_a: f32,
    pad_b: f32,
    pad_c: f32,
}

/// Header + fixed-slot array (mirrors `raygen.slang::AtmoVolumes`).
#[derive(Clone, Copy, Default, PartialEq, ShaderType)]
struct GpuAtmosphereVolumes {
    sun_direction: Vec3,
    sun_illuminance: f32,
    count: u32,
    pad_a: u32,
    pad_b: u32,
    pad_c: u32,
    volumes: [GpuAtmosphereVolume; MAX_ATMOSPHERE_VOLUMES],
}

pub const MAX_ATMOSPHERE_VOLUMES: usize = 4;

// Transmittance LUT: T(radius, sun-zenith cosine) per volume, baked by
// `atmosphere_lut_bake.slang` into the SAME device-address buffer after the
// header+volumes block — the march does one bilinear buffer lookup per step
// instead of an inner sun integral. Layout constants mirror `raygen.slang` /
// the bake shader.
pub const ATMO_LUT_W: u32 = 256; // mu = cos(zenith) axis
pub const ATMO_LUT_H: u32 = 64; // radius axis (bottom→top)
pub const ATMO_LUT_OFFSET: u64 = 512; // header 32B + volumes 256B, padded
pub const ATMO_LUT_LAYER_BYTES: u64 = (ATMO_LUT_W * ATMO_LUT_H * 16) as u64;

/// Render-world: the volume buffer + its device address (rides
/// `RtCamera.atmo` so raygen reaches it via `physical_load` — no descriptor
/// set surgery).
#[derive(Resource, Default)]
pub struct SolariAtmosphereVolumesGpu {
    staged: GpuAtmosphereVolumes,
    buffer: Option<bevy_render::render_resource::Buffer>,
    /// Device address of `buffer` (0 until created).
    pub address: u64,
    /// Live volume count this frame (0 = feature idle, raygen skips).
    pub count: u32,
    /// The params (centers/sun zeroed) the LUT region was last baked for.
    baked_params: GpuAtmosphereVolumes,
    /// LUT region is stale (params changed) — cleared once a bake is encoded.
    needs_lut_bake: bool,
}

/// Render-world resource: the LUT-bake heap kernel + its slot.
#[derive(Resource)]
pub struct AtmosphereLutBake {
    kernel: HeapKernel,
    slots: KernelSlots,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for AtmosphereLutBake {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for AtmosphereLutBake {}
unsafe impl Sync for AtmosphereLutBake {}

/// `RenderStartup` (after `SolariSetup`): compile the LUT-bake kernel — a
/// layout-free heap pipeline ([`HeapKernel`]), Slang from source. No push
/// params: the layout constants are baked into the kernel.
pub fn init_atmosphere_lut_bake(
    mut commands: Commands,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let Some(kernel) = HeapKernel::new(
        &seam,
        "atmosphere_lut_bake.slang",
        include_str!("atmosphere_lut_bake.slang"),
        "bake",
        &[],
        &[],
        "solari_atmosphere_lut_bake",
        0,
    ) else {
        return;
    };
    let slots = KernelSlots::new(&seam, 1);
    commands.insert_resource(AtmosphereLutBake {
        kernel,
        slots,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

/// `ExtractSchedule`: gather atmosphere volumes, centers relative to the
/// primary camera (f64 CPU subtract — the render world is origin-relative to
/// that same camera, so raygen re-anchors with `camera_position`).
pub fn extract_atmosphere_volumes(
    volumes: Extract<Query<(&SolariAtmosphereVolume, &GlobalTransform)>>,
    cameras: Extract<Query<&GlobalTransform, With<SolariCamera>>>,
    suns: Extract<Query<(&GlobalTransform, &SolariDirectionLight)>>,
    mut gpu: ResMut<SolariAtmosphereVolumesGpu>,
) {
    let Some(camera_world) = cameras.iter().next().map(|t| t.translation()) else {
        gpu.count = 0;
        gpu.staged.count = 0;
        return;
    };
    let (sun_direction, sun_illuminance) = suns
        .iter()
        .next()
        .map(|(transform, light)| (transform.back().as_vec3(), light.illuminance))
        .unwrap_or((Vec3::Y, 0.0));

    let mut staged = GpuAtmosphereVolumes {
        sun_direction,
        sun_illuminance,
        ..Default::default()
    };
    for (volume, transform) in volumes.iter() {
        if staged.count as usize >= MAX_ATMOSPHERE_VOLUMES {
            break;
        }
        staged.volumes[staged.count as usize] = GpuAtmosphereVolume {
            center: (transform.translation() - camera_world).as_vec3(),
            bottom_radius: volume.bottom_radius,
            rayleigh_scattering: volume.rayleigh_scattering,
            rayleigh_scale_height: volume.rayleigh_scale_height,
            top_radius: volume.top_radius,
            mie_scattering: volume.mie_scattering,
            mie_extinction: volume.mie_extinction,
            mie_scale_height: volume.mie_scale_height,
            mie_phase_g: volume.mie_phase_g,
            pad_a: 0.0,
            pad_b: 0.0,
            pad_c: 0.0,
        };
        staged.count += 1;
    }
    gpu.count = staged.count;
    gpu.staged = staged;

    // LUT staleness: params only — centers move every frame and the sun
    // doesn't enter T(r, mu); neither may trigger a re-bake.
    let mut params = staged;
    params.sun_direction = Vec3::ZERO;
    params.sun_illuminance = 0.0;
    for v in params.volumes.iter_mut() {
        v.center = Vec3::ZERO;
    }
    let mut baked = gpu.baked_params;
    baked.sun_direction = Vec3::ZERO;
    baked.sun_illuminance = 0.0;
    if params != baked && params.count > 0 {
        gpu.baked_params = params;
        gpu.needs_lut_bake = true;
    }
}

/// `Render::PrepareResources`: (re)write the fixed-size volume buffer; resolve
/// its device address once (the buffer never reallocates).
pub fn prepare_atmosphere_volumes(
    mut gpu: ResMut<SolariAtmosphereVolumesGpu>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    allocator: Option<Res<Allocator>>,
) {
    if gpu.count == 0 && gpu.buffer.is_none() {
        return; // feature idle, never used
    }
    let mut bytes = bevy_render::render_resource::encase::StorageBuffer::new(Vec::<u8>::new());
    bytes.write(&gpu.staged).expect("atmosphere volumes encode");
    let bytes = bytes.into_inner();
    if gpu.buffer.is_none() {
        let Some(allocator) = allocator else { return };
        // Header + volumes at the front; the transmittance-LUT region (baked
        // by `atmosphere_lut_bake.slang`, read by raygen via device address)
        // follows at `ATMO_LUT_OFFSET`.
        let size = ATMO_LUT_OFFSET + MAX_ATMOSPHERE_VOLUMES as u64 * ATMO_LUT_LAYER_BYTES;
        let buffer =
            render_device.create_buffer(&bevy_render::render_resource::BufferDescriptor {
                label: Some("solari_atmosphere_volumes"),
                size,
                usage: bevy_render::render_resource::BufferUsages::STORAGE
                    | bevy_render::render_resource::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        gpu.address = allocator.wgpu_buffer_device_address(&buffer).get();
        gpu.buffer = Some(buffer);
    }
    if let Some(buffer) = &gpu.buffer {
        // Per-frame write covers ONLY the header+volumes block (288 B); the
        // LUT region belongs to the bake pass.
        render_queue.write_buffer(buffer, 0, &bytes);
    }
}

/// `Core3d` (before the trace, next to the sky-cube bake): re-bake the
/// transmittance LUT region when volume params changed. Reads params from and
/// writes texels into the same storage buffer. A raw heap dispatch: the buffer
/// slot rewritten per dispatch, the slot array in push data (no params).
pub fn dispatch_atmosphere_lut_bake(
    mut gpu: ResMut<SolariAtmosphereVolumesGpu>,
    bake: Option<Res<AtmosphereLutBake>>,
    seam: Option<Res<BindingSeam>>,
    mut ctx: RenderContext,
) {
    if !gpu.needs_lut_bake || gpu.count == 0 {
        return;
    }
    // Kernel/seam/buffer not up yet — flag stays set, retried next frame.
    let (Some(bake), Some(seam)) = (bake, seam) else {
        return;
    };
    let Some(buffer) = &gpu.buffer else { return };
    let blob = bake.kernel.push_blob(
        "solari_atmosphere_lut_bake",
        &[],
        &[("buf", bake.slots.buffer(&seam, 0, buffer))],
    );
    let count = gpu.count;
    gpu.needs_lut_bake = false;
    let encoder = ctx.command_encoder();
    // SAFETY: Vulkan backend; the slot references a live heap descriptor; the
    // barriers bracket this dispatch against the surrounding wgpu compute
    // passes (raw dispatches are invisible to wgpu's tracking).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &bake.raw_device;
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)];
            let dep = vk::DependencyInfo::default().memory_barriers(&barrier);
            // Prior compute writes -> our header reads + LUT writes.
            dev.cmd_pipeline_barrier2(cb, &dep);
            seam.bind_heaps(cb);
            seam.push_data(cb, &blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, bake.kernel.pipeline);
            dev.cmd_dispatch(cb, ATMO_LUT_W.div_ceil(8), ATMO_LUT_H.div_ceil(8), count);
            // Our LUT writes -> downstream compute reads; the trace's own
            // pre-barrier covers RT-stage visibility.
            dev.cmd_pipeline_barrier2(cb, &dep);
        });
    }
}

/// Render-world marker on a view whose camera has a [`SolariAtmosphere`] — the
/// pathtracer binds the baked sky cube for it.
#[derive(Component)]
pub struct SolariAtmosphereView;

/// Render-world resource: the baked sky cube + its views (a `2d-array` view to
/// write during the bake, a `cube` view for the pathtracer to sample).
#[derive(Resource)]
pub struct AtmosphereSky {
    pub cube_view: TextureView,
    array_view: TextureView,
    /// The backing storage texture (kept so consumers can transition its layout
    /// — the bake moves it to its storage state through wgpu's tracker, so
    /// wgpu leaves it in `GENERAL`).
    pub texture: bevy_render::render_resource::Texture,
}

/// Render-world resource: the sky-bake heap kernel + its storage-image slot.
#[derive(Resource)]
pub struct AtmosphereBake {
    kernel: HeapKernel,
    slots: KernelSlots,
    raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for AtmosphereBake {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for AtmosphereBake {}
unsafe impl Sync for AtmosphereBake {}

/// `RenderStartup` (after `SolariSetup`): compile the sky-bake kernel — a
/// layout-free heap pipeline ([`HeapKernel`]), Slang from source with the
/// shared `atmosphere` physics module. The `Atmosphere` params (96 B) ride
/// the push block.
pub fn init_atmosphere_bake(
    mut commands: Commands,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let Some(kernel) = HeapKernel::new(
        &seam,
        "atmosphere_bake.slang",
        include_str!("atmosphere_bake.slang"),
        "bake",
        &[("atmosphere", include_str!("atmosphere.slang"))],
        &[],
        "solari_atmosphere_bake",
        GpuSolariAtmosphere::min_size().get() as u32,
    ) else {
        return;
    };
    let slots = KernelSlots::new_mixed(&seam, &[HeapKind::Image]);
    commands.insert_resource(AtmosphereBake {
        kernel,
        slots,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
}

/// `ExtractSchedule`: gather the primary `SolariAtmosphere` + the primary sun into
/// the GPU uniform, mark atmosphere views, and clear the enable flag otherwise.
pub fn extract_solari_atmosphere(
    cameras: Extract<
        Query<(RenderEntity, &SolariAtmosphere, Option<&SolariGlobalFog>), With<SolariCamera>>,
    >,
    suns: Extract<Query<(&GlobalTransform, &SolariDirectionLight)>>,
    mut gpu: ResMut<SolariAtmosphereGpu>,
    mut commands: Commands,
) {
    // The primary sun: direction TO the sun is the light's local +Z (`back`),
    // matching `light_resolve`'s `world.back()` convention.
    let (sun_direction, sun_illuminance) = suns
        .iter()
        .next()
        .map(|(transform, light)| (transform.back().as_vec3(), light.illuminance))
        .unwrap_or((Vec3::Y, 0.0));

    let mut any = false;
    let mut next = GpuSolariAtmosphere::default();
    let mut sky_frame = Quat::IDENTITY;
    for (render_entity, atmosphere, global_fog) in &cameras {
        any = true;
        commands.entity(render_entity).insert(SolariAtmosphereView);
        // Spherical-planet frame: when `up` isn't world +Y, the bake keeps its
        // canonical up-=-+Y convention and we (a) hand it the sun at the same
        // ZENITH angle, azimuth 0 (quantized — the only thing that re-bakes),
        // and (b) derive the world→bake rotation the miss shader applies to
        // cube sample directions (rides `RtCamera.sky_frame`, never re-bakes).
        // Flat scenes (`up ≈ +Y`, the default) take the identity path.
        // NOTE: with a non-+Y frame the
        // world-space aerial fog march sees the frame-local sun — global fog
        // is a flat-world feature; leave `SolariGlobalFog` off on planets.
        let up = atmosphere.up.normalize_or(Vec3::Y);
        let bake_sun = if up.dot(Vec3::Y) > 0.999_999 {
            sun_direction
        } else {
            let tangent = sun_direction - up * sun_direction.dot(up);
            let x_axis = if tangent.length_squared() > 1e-10 {
                tangent.normalize()
            } else {
                up.any_orthonormal_vector()
            };
            let z_axis = x_axis.cross(up);
            sky_frame = Quat::from_mat3(&Mat3::from_cols(x_axis, up, z_axis)).inverse();
            // Quantize the zenith so orbital drift doesn't re-bake per frame.
            let cos_zenith = (sun_direction.dot(up)).clamp(-1.0, 1.0);
            let q = (cos_zenith * 512.0).round() / 512.0;
            Vec3::new((1.0 - q * q).max(0.0).sqrt(), q, 0.0)
        };
        // Global height fog is opt-in via `SolariGlobalFog`. Absent (or zero
        // visibility) = `aerial_enabled = 0` → the path tracer skips the aerial
        // march entirely; the sun + optical depths still light local fog volumes.
        let fog = global_fog.filter(|f| f.visibility > 0.0 && f.visibility.is_finite());
        next = GpuSolariAtmosphere {
            bottom_radius: atmosphere.bottom_radius,
            top_radius: atmosphere.top_radius,
            rayleigh_scattering: atmosphere.rayleigh_scattering,
            rayleigh_scale_height: atmosphere.rayleigh_scale_height,
            mie_scattering: atmosphere.mie_scattering,
            mie_extinction: atmosphere.mie_extinction,
            mie_scale_height: atmosphere.mie_scale_height,
            mie_phase_g: atmosphere.mie_phase_g,
            sun_direction: bake_sun,
            sun_illuminance,
            camera_altitude: atmosphere.camera_altitude,
            aerial_visibility: fog.map_or(0.0, |f| f.visibility),
            aerial_fog_height: fog.map_or(0.0, |f| f.fog_height),
            aerial_fog_base: fog.map_or(0.0, |f| f.fog_base),
            aerial_phase_g: fog.map_or(0.0, |f| f.phase_g),
            aerial_enabled: f32::from(fog.is_some()),
        };
    }
    // Re-bake only when the params actually changed — an unchanged sky needs
    // no re-bake. A typical frame (static sun + params) costs one compare here
    // and nothing on the GPU.
    if next != gpu.current {
        gpu.current = next;
        if any {
            gpu.needs_bake = true;
        }
    }
    // The rotation changes freely (every frame while orbiting) — it does NOT
    // touch the bake uniform, so no re-bake.
    gpu.sky_frame = sky_frame;
    gpu.enabled = any;
}

/// `Render::PrepareResources`: allocate the sky cube once.
pub fn prepare_atmosphere_sky(
    mut commands: Commands,
    sky: Option<Res<AtmosphereSky>>,
    render_device: Res<RenderDevice>,
    gpu: Res<SolariAtmosphereGpu>,
) {
    if !gpu.enabled || sky.is_some() {
        return;
    }
    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some("solari_atmosphere_sky"),
        size: bevy_render::render_resource::Extent3d {
            width: SKY_SIZE,
            height: SKY_SIZE,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let array_view = texture.create_view(&TextureViewDescriptor {
        label: Some("solari_atmosphere_sky_array"),
        dimension: Some(TextureViewDimension::D2Array),
        ..Default::default()
    });
    let cube_view = texture.create_view(&TextureViewDescriptor {
        label: Some("solari_atmosphere_sky_cube"),
        dimension: Some(TextureViewDimension::Cube),
        ..Default::default()
    });
    commands.insert_resource(AtmosphereSky {
        cube_view,
        array_view,
        texture,
    });
}

/// `RenderGraph` (before the pathtracer): bake the sky cube, but only when its
/// inputs changed — the cube persists, so a static sun + params re-bakes nothing.
/// A raw heap dispatch: the cube's storage-image slot rewritten per bake, the
/// `Atmosphere` params in push data.
pub fn dispatch_atmosphere_bake(
    bake: Option<Res<AtmosphereBake>>,
    seam: Option<Res<BindingSeam>>,
    sky: Option<Res<AtmosphereSky>>,
    mut gpu: ResMut<SolariAtmosphereGpu>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    if !gpu.enabled || !gpu.needs_bake {
        return;
    }
    // Kernel/seam/cube not up yet — flag stays set, retried next frame.
    let (Some(bake), Some(seam), Some(sky)) = (bake, seam, sky) else {
        return;
    };
    gpu.needs_bake = false;
    // Push params: the encase std140 bytes of `GpuSolariAtmosphere` — identical
    // to the shader's std430 push block for this flat f32/vec3 struct.
    let mut params = bevy_render::render_resource::encase::UniformBuffer::new(Vec::<u8>::new());
    params.write(&gpu.current).expect("atmosphere params encode");
    let blob = bake.kernel.push_blob(
        "solari_atmosphere_bake",
        &params.into_inner(),
        &[("sky", bake.slots.storage_image(&seam, 0, &sky.array_view))],
    );
    // Move the cube to its storage state (GENERAL) through wgpu's tracker
    // BEFORE the raw storage write: nothing wgpu-side ever touches this
    // texture otherwise (only the raw trace samples it, with its own
    // GENERAL↔READ_ONLY round trip), so this both performs the initial
    // UNDEFINED→GENERAL transition and keeps the tracked state matching the
    // "wgpu leaves it in GENERAL" contract the trace relies on.
    ctx.command_encoder().transition_resources(
        core::iter::empty(),
        core::iter::once(wgpu::TextureTransition {
            // bevy `Texture` → the wrapped `wgpu::Texture`.
            texture: &*sky.texture,
            selector: None,
            state: wgpu::TextureUses::STORAGE_WRITE_ONLY,
        }),
    );
    // Own command buffer (the ctx encoder just recorded the wgpu transition —
    // the fork panics if one encoder mixes wgpu work with raw `as_hal_mut`);
    // `add_command_buffer` flushes the transition ahead of the dispatch.
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("solari_atmosphere_bake"),
    });
    // SAFETY: Vulkan backend; the slot references a live heap descriptor. No
    // raw barriers: the bake reads only push data, ordering against the prior
    // frame's trace comes from the trace's own cube restore barrier
    // (RT→COMPUTE, dst SHADER_WRITE), and this frame's trace read is covered
    // by its GENERAL→READ_ONLY barrier (src COMPUTE SHADER_WRITE).
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &bake.raw_device;
            seam.bind_heaps(cb);
            seam.push_data(cb, &blob);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, bake.kernel.pipeline);
            dev.cmd_dispatch(cb, SKY_SIZE.div_ceil(8), SKY_SIZE.div_ceil(8), 6);
        });
    }
    ctx.add_command_buffer(encoder.finish());
}

