//! Self-contained single-scattering atmosphere → sky cubemap for the pathtracer.
//!
//! A compute pre-pass ([`atmosphere_bake.wgsl`](mod@self)) bakes the atmosphere
//! into a cube from [`SolariAtmosphere`] + the primary [`SolariDirectionLight`];
//! the pathtracer samples that cube on a ray miss (the existing skybox path).
//! The cube persists, so the bake re-runs only when its inputs change (a moving
//! sun re-bakes; a static sky is free). Fully solari-owned — no `bevy_pbr`
//! atmosphere / raster `GpuLights` coupling — so it works with `PbrPlugin`
//! disabled.

use bevy_ecs::{
    component::Component,
    query::With,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::Vec3;
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        binding_types::{texture_storage_2d_array, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        ComputePassDescriptor, PipelineCache,
        ShaderStages, ShaderType, StorageTextureAccess, TextureDescriptor, TextureDimension,
        TextureFormat, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
        UniformBuffer,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    Extract,
};
use bevy_transform::components::GlobalTransform;

use crate::lights::SolariDirectionLight;
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;
use crate::render::SolariCamera;

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
    /// Aerial-perspective **ground-level visibility** in WORLD units (Koschmieder
    /// meteorological range): the view distance at which a surface *in the densest
    /// fog* fades to ~2 % contrast. Lower ⇒ haze closer/thicker; higher ⇒ clearer.
    /// `0.0` (or non-finite) disables the global height fog entirely — the sky and
    /// sun keep working, and local
    /// [`SolariFogVolume`](crate::bindings::SolariFogVolume)s still march.
    pub aerial_visibility: f32,
    /// Fog-layer **scale height** in WORLD units: density falls off as
    /// `exp(-(y - fog_base) / fog_height)`, so the haze is densest at the ground and
    /// thins with altitude (a low-lying fog the camera shoots over). Larger ⇒ the
    /// fog reaches higher.
    pub aerial_fog_height: f32,
    /// World-space Y of the densest fog (ground level).
    pub aerial_fog_base: f32,
    /// Henyey-Greenstein asymmetry `g` in `(-1, 1)` for the volumetric sun shafts
    /// (god rays): higher ⇒ a tighter, brighter glow concentrated toward the sun.
    pub aerial_phase_g: f32,
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
            // Generic defaults for a metres-scale world: ~12 km ground visibility,
            // a 100 m-tall fog layer at y = 0. Set to your scene's units.
            aerial_visibility: 12000.0,
            aerial_fog_height: 100.0,
            aerial_fog_base: 0.0,
            aerial_phase_g: 0.4,
        }
    }
}

/// GPU mirror of [`SolariAtmosphere`] + the sun — matches `Atmosphere` in
/// `atmosphere.wgsl`. `Default` is the "disabled" state (`aerial_enabled = 0`),
/// bound by the pathtracer when no view has an atmosphere.
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

/// Render-world resource: the extracted atmosphere uniform + whether any solari
/// camera enabled it this frame.
#[derive(Resource, Default)]
pub struct SolariAtmosphereGpu {
    uniform: UniformBuffer<GpuSolariAtmosphere>,
    pub enabled: bool,
    /// The last extracted uniform value, to detect changes.
    current: GpuSolariAtmosphere,
    /// The sky cube is stale: the uniform changed while enabled. Cleared by
    /// [`dispatch_atmosphere_bake`] only once a bake is actually encoded, so a
    /// pending bake survives pipeline compilation.
    needs_bake: bool,
}

impl SolariAtmosphereGpu {
    /// The atmosphere uniform's binding, for the pathtracer's group(1) slot 6.
    /// `Some` once [`prepare_atmosphere_sky`] has created the buffer.
    pub fn binding(&self) -> Option<bevy_render::render_resource::BindingResource<'_>> {
        self.uniform.binding()
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
}

/// Render-world resource: the bake's per-frame bind group. The bind-group layout
/// lives in [`SolariResourceManager`](crate::resource_manager::SolariResourceManager)
/// and the compiled pipeline id in [`crate::pipelines::SolariPipelines`].
#[derive(Resource)]
pub struct AtmospherePipeline {
    bind_group: Option<BindGroup>,
}

/// The atmosphere-bake bind-group layout. Owned by
/// [`SolariResourceManager`](crate::resource_manager::SolariResourceManager).
pub(crate) fn atmosphere_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "solari_atmosphere",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<GpuSolariAtmosphere>(false),
                texture_storage_2d_array(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    )
}

/// `ExtractSchedule`: gather the primary `SolariAtmosphere` + the primary sun into
/// the GPU uniform, mark atmosphere views, and clear the enable flag otherwise.
pub fn extract_solari_atmosphere(
    cameras: Extract<Query<(RenderEntity, &SolariAtmosphere), With<SolariCamera>>>,
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
    for (render_entity, atmosphere) in &cameras {
        any = true;
        commands.entity(render_entity).insert(SolariAtmosphereView);
        next = GpuSolariAtmosphere {
            bottom_radius: atmosphere.bottom_radius,
            top_radius: atmosphere.top_radius,
            rayleigh_scattering: atmosphere.rayleigh_scattering,
            rayleigh_scale_height: atmosphere.rayleigh_scale_height,
            mie_scattering: atmosphere.mie_scattering,
            mie_extinction: atmosphere.mie_extinction,
            mie_scale_height: atmosphere.mie_scale_height,
            mie_phase_g: atmosphere.mie_phase_g,
            sun_direction,
            sun_illuminance,
            camera_altitude: atmosphere.camera_altitude,
            aerial_visibility: atmosphere.aerial_visibility,
            aerial_fog_height: atmosphere.aerial_fog_height,
            aerial_fog_base: atmosphere.aerial_fog_base,
            aerial_phase_g: atmosphere.aerial_phase_g,
            // Zero/non-finite visibility = no global height fog; the rest of
            // the uniform (sun, optical depths) still lights fog volumes.
            aerial_enabled: if atmosphere.aerial_visibility > 0.0
                && atmosphere.aerial_visibility.is_finite()
            {
                1.0
            } else {
                0.0
            },
        };
    }
    // No atmosphere view: a disabled (default) uniform keeps the pathtracer's
    // binding valid (it gates aerial perspective on `aerial_enabled`).
    //
    // Touch the uniform only when its contents actually changed — `get_mut`
    // marks it for re-upload, and an unchanged sky needs no re-bake. A typical
    // frame (static sun + params) costs one compare here and nothing on the GPU.
    if next != gpu.current {
        gpu.current = next;
        *gpu.uniform.get_mut() = next;
        if any {
            gpu.needs_bake = true;
        }
    }
    gpu.enabled = any;
}

/// `RenderStartup`: the bake owns only its per-frame bind group; the layout lives
/// in `SolariResourceManager`, the pipeline id in `SolariPipelines`.
pub fn init_atmosphere_pipeline(mut commands: Commands) {
    commands.insert_resource(AtmospherePipeline { bind_group: None });
}

/// `Render::PrepareResources`: allocate the sky cube once, upload the uniform.
pub fn prepare_atmosphere_sky(
    mut commands: Commands,
    sky: Option<Res<AtmosphereSky>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut gpu: ResMut<SolariAtmosphereGpu>,
) {
    // Keep the uniform binding valid every frame (the pathtracer binds it even
    // with no atmosphere view); `UniformBuffer` skips the upload when unchanged.
    gpu.uniform.write_buffer(&render_device, &render_queue);

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
    });
}

/// `Render::PrepareBindGroups`: build the bake bind group **once** — the uniform
/// buffer never reallocates after its first write and the sky cube is allocated
/// once, so the group stays valid for the resource lifetimes.
pub fn prepare_atmosphere_bind_group(
    mut pipeline: ResMut<AtmospherePipeline>,
    resource_manager: Option<Res<SolariResourceManager>>,
    sky: Option<Res<AtmosphereSky>>,
    gpu: Res<SolariAtmosphereGpu>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    if pipeline.bind_group.is_some() {
        return;
    }
    let (Some(resource_manager), Some(sky), Some(uniform)) =
        (resource_manager, sky, gpu.uniform.binding())
    else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&resource_manager.atmosphere);
    pipeline.bind_group = Some(render_device.create_bind_group(
        "solari_atmosphere",
        &layout,
        &BindGroupEntries::sequential((uniform, &sky.array_view)),
    ));
}

/// `RenderGraph` (before the pathtracer): bake the sky cube, but only when its
/// inputs changed — the cube persists, so a static sun + params re-bakes nothing.
pub fn dispatch_atmosphere_bake(
    pipeline: Res<AtmospherePipeline>,
    pipelines: Res<SolariPipelines>,
    mut gpu: ResMut<SolariAtmosphereGpu>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !gpu.enabled || !gpu.needs_bake {
        return;
    }
    let (Some(compute), Some(bind_group)) = (
        pipeline_cache.get_compute_pipeline(pipelines.atmosphere),
        pipeline.bind_group.as_ref(),
    ) else {
        return;
    };
    gpu.needs_bake = false;
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("solari_atmosphere_bake"),
        timestamp_writes: None,
    });
    pass.set_pipeline(compute);
    pass.set_bind_group(0, bind_group, &[]);
    let d = diagnostics.time_span(&mut pass, "solari_atmosphere_bake");
    pass.dispatch_workgroups(SKY_SIZE.div_ceil(8), SKY_SIZE.div_ceil(8), 6);
    d.end(&mut pass);
}

