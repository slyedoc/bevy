//! `bevy_solari`'s own directional-light table and the resolve pass that sources
//! the light **direction from the GPU transform table**.
//!
//! The ray tracer needs directional-light data in the render world but shouldn't
//! depend on `bevy_pbr` running (it owns its material type via [`crate::material`];
//! lights are the other `bevy_pbr` runtime dependency the scene binder had).
//!
//! [`SolariDirectionLight`] is a self-contained directional ("sun") light: a small
//! main-world component (color / illuminance / sun-disk angular size). It is a
//! [`gpu_table!`](crate::gpu_table) table like transforms/instances:
//! - [`DirectionalLightColumn`] holds the per-light [`GpuDirectionalLight`]
//!   settings, scattered **only on `Changed<SolariDirectionLight>`** (persistent
//!   buffer — no per-frame re-upload). It's the buffer the scene binder binds.
//! - [`LightTransformSlotColumn`] holds the light's transform-table slot.
//!
//! Because the light is also a transform-table node, its world transform lives on
//! the GPU, so its `direction_to_light` is resolved there ([`light_resolve.wgsl`])
//! from `world[transform_slot]` — the CPU never needs the light's `GlobalTransform`.
//! The binder only enumerates the (few) active lights for its light-source index
//! list ([`ActiveDirectionalLights`], rebuilt per frame — cheap).

use bevy_app::{App, Plugin};
use bevy_asset::AssetId;
use bevy_color::{Color, ColorToComponents, LinearRgba};
use bevy_ecs::{
    change_detection::DetectChanges,
    component::Component,
    entity::Entity,
    prelude::{ReflectComponent, Ref},
    resource::Resource,
    schedule::{common_conditions::resource_exists, IntoScheduleConfigs},
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::{ops::cos, Vec3};
use bevy_platform::{collections::HashSet, hash::FixedHasher};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        ComputePassDescriptor,
        PipelineCache,
        ShaderStages, ShaderType, UniformBuffer,
    },
    renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
    sync_world::{RenderEntity, SyncToRenderWorld},
    Extract, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_transform::components::Transform;
use bytemuck::{Pod, Zeroable};
use core::f32::consts::TAU;

use crate::bindings::SolariMaterialAssets;
use crate::ecs_gpu::{GpuColumn, GpuSlot, GpuTable};
use crate::instance::InstanceManager;
use crate::material::SolariMaterial;
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;
use crate::transform::{dispatch_transform_propagate, TransformGraph, TransformPropagate};
use crate::{SolariClusterSystems, SolariSetup};

const WORKGROUP_SIZE: u32 = 64;

/// Angular diameter of the sun as seen from Earth, in radians (~0.533°). Matches
/// the physical default `bevy_pbr` uses for its sun disk.
pub const DEFAULT_SUN_DISK_ANGULAR_SIZE: f32 = 0.009_308_42;

/// A directional ("sun") light for the `bevy_solari` ray tracer.
///
/// The light direction is the entity's transform `back()` (`+Z`), like
/// `bevy_pbr`'s `DirectionalLight` — but resolved from the GPU transform table,
/// so it works with `PbrPlugin`/`TransformPlugin` disabled. Requires `Transform`
/// (which pulls `GlobalTransform`, making the entity a transform-table node).
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component, Default, Clone)]
#[require(Transform, SyncToRenderWorld)]
pub struct SolariDirectionLight {
    /// Light color.
    pub color: Color,
    /// Illuminance in lux (lm/m²).
    pub illuminance: f32,
    /// Angular diameter of the sun disk, in radians — controls penumbra softness.
    pub sun_disk_angular_size: f32,
}

impl Default for SolariDirectionLight {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            // `bevy_light::light_consts::lux::AMBIENT_DAYLIGHT`, inlined to avoid a dep.
            illuminance: 10_000.0,
            sun_disk_angular_size: DEFAULT_SUN_DISK_ANGULAR_SIZE,
        }
    }
}

/// One directional light as the path tracer reads it (mirrors
/// `raytracing_scene_bindings.wgsl::GpuDirectionalLight` and `light_resolve.wgsl`).
/// `direction_to_light` is filled by the resolve pass; the other three are
/// settings. `repr(C)` Pod *and* WGSL-layout-compatible (the `f32` after the first
/// `vec3` fills its padding, so `luminance` lands at the 16-aligned offset).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuDirectionalLight {
    direction_to_light: Vec3,
    cos_theta_max: f32,
    luminance: Vec3,
    inverse_pdf: f32,
}

impl GpuDirectionalLight {
    /// Build the settings (everything but `direction_to_light`, which the resolve
    /// pass fills from the GPU transform).
    fn from_settings(light: &SolariDirectionLight) -> Self {
        let cos_theta_max = cos(light.sun_disk_angular_size / 2.0);
        let solid_angle = TAU * (1.0 - cos_theta_max);
        let luminance = (LinearRgba::from(light.color).to_vec3() * light.illuminance) / solid_angle;
        Self {
            direction_to_light: Vec3::ZERO, // resolved GPU-side
            cos_theta_max,
            luminance,
            inverse_pdf: solid_angle,
        }
    }
}

crate::gpu_table! {
    /// The directional-light table: per-light settings + its transform-table slot,
    /// both change-driven columns (see the module docs).
    pub table SolariLights as SolariLightsTablePlugin {
        members: bevy_ecs::query::With<SolariDirectionLight>,
        extract: extract_solari_lights,
        columns {
            DirectionalLightColumn   => directional_light: GpuDirectionalLight = "lights.directional" @ scene 3,
            LightTransformSlotColumn => transform_slot:     u32                = "lights.transform_slot",
        }
    }
}

/// Render-world list of this frame's active directional lights as
/// `(render entity, light slot)`, rebuilt by [`extract_solari_lights`]. The scene
/// binder reads it to append each to its light-source index list (id = slot) and
/// to track entities for previous-frame light reprojection. Cheap (few lights) —
/// the heavy settings go through the change-driven [`DirectionalLightColumn`].
#[derive(Resource, Default)]
pub struct ActiveDirectionalLights(pub Vec<(Entity, u32)>);

/// `ExtractSchedule` (the table's extract): rebuild the active list (all slotted
/// lights) and scatter the settings + transform-slot columns for any whose
/// `SolariDirectionLight` changed. Both `&GpuSlot` fetches mean only fully-slotted
/// lights are seen (true the same frame — slots are assigned in `PostUpdate`).
pub fn extract_solari_lights(
    lights: Extract<
        Query<(
            RenderEntity,
            Ref<SolariDirectionLight>,
            &GpuSlot<TransformGraph>,
            &GpuSlot<SolariLights>,
        )>,
    >,
    mut table: ResMut<SolariLights>,
    mut active: ResMut<ActiveDirectionalLights>,
) {
    active.0.clear();
    for (render_entity, light, transform_slot, light_slot) in &lights {
        let slot = light_slot.index();
        active.0.push((render_entity, slot));
        // `is_changed()` includes the frame the component was added.
        if light.is_changed() {
            crate::ecs_gpu::push_record(
                &mut table.directional_light,
                slot,
                GpuDirectionalLight::from_settings(&light),
            );
            crate::ecs_gpu::push_record(&mut table.transform_slot, slot, transform_slot.index());
        }
    }
}

/// One entry in the path tracer's light-source RIS list (`light_sources`), shared
/// by emissive-mesh lights and directional lights. Mirrors the WGSL `LightSource`.
#[derive(ShaderType, Clone)]
pub struct GpuLightSource {
    kind: u32,
    id: u32,
}

impl GpuLightSource {
    /// Emissive instance: `kind` packs the instance's total triangle count in bits
    /// 1..=31; bit 0 stays 0 so it doesn't collide with the directional kind (= 1).
    /// `id` is the PTLAS instance slot ray hits report as `instance_index`.
    pub fn new_emissive_mesh_light(instance_id: u32, triangle_count: u32) -> GpuLightSource {
        Self {
            kind: triangle_count << 1,
            id: instance_id,
        }
    }

    /// Directional light: `kind = 1`, `id` = the light's table slot (index into the
    /// `directional_lights` column buffer).
    pub fn new_directional_light(directional_light_id: u32) -> GpuLightSource {
        Self {
            kind: 1,
            id: directional_light_id,
        }
    }
}

/// Render-world cache of emissive-mesh `GpuLightSource`s, rebuilt by
/// [`prepare_emissive_lights`] only when the emissive set could have changed. The
/// scene binder seeds its `light_sources` list with this, then appends directional
/// lights (from [`ActiveDirectionalLights`]).
#[derive(Resource, Default)]
pub struct EmissiveLights {
    /// One [`GpuLightSource`] per active emissive-mesh instance.
    pub lights: Vec<GpuLightSource>,
    /// The emissive material set the cache was built against — a diff catches a
    /// material whose `emissive` was edited without any instance changing.
    cached_assets: HashSet<AssetId<SolariMaterial>, FixedHasher>,
}

/// `Render::Prepare`: rebuild the emissive-mesh light list — but only when the
/// emissive set could have changed (a material's emissiveness edited, or an
/// instance added / released / re-materialed). On a move-only frame the O(active)
/// walk is skipped and the cache reused — the headline saving. (Light *transforms*
/// are GPU-side via the PTLAS; this only tracks which instances emit.)
pub fn prepare_emissive_lights(
    mut emissive: ResMut<EmissiveLights>,
    materials: Res<SolariMaterialAssets>,
    instances: Res<InstanceManager>,
) {
    // Material assets whose `emissive` is non-black.
    let mut emissive_assets = HashSet::<AssetId<SolariMaterial>, FixedHasher>::default();
    for (asset_id, material) in materials.iter() {
        if material.emissive.to_vec3() != Vec3::ZERO {
            emissive_assets.insert(*asset_id);
        }
    }

    let changed = emissive_assets != emissive.cached_assets
        || !instances.added_slots().is_empty()
        || !instances.released_slots().is_empty()
        || !instances.material_dirty().is_empty();
    if !changed {
        return;
    }

    emissive.lights.clear();
    for &slot in instances.active_slots() {
        let asset_id = instances.instance_material_asset_id(slot);
        if emissive_assets.contains(&asset_id) {
            let triangle_count = instances.instance_total_triangle_count(slot);
            if triangle_count > 0 && triangle_count <= u16::MAX as u32 {
                emissive
                    .lights
                    .push(GpuLightSource::new_emissive_mesh_light(slot.0, triangle_count));
            }
        }
    }
    emissive.cached_assets = emissive_assets;
}

/// Uniform shared with `light_resolve.wgsl::ResolveParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, ShaderType)]
struct ResolveParams {
    light_count: u32,
    node_count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Render-world resource: the direction-resolve pipeline + bind group.
#[derive(Resource)]
pub struct LightResolve {
    light_count: u32,
    params: UniformBuffer<ResolveParams>,
    bind_group: Option<BindGroup>,
}

/// The light-resolve bind-group layout. Owned by
/// [`SolariResourceManager`](crate::resource_manager::SolariResourceManager).
pub(crate) fn light_resolve_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "light_resolve",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 world
                storage_buffer_read_only_sized(false, None), // 1 transform slots
                storage_buffer_sized(false, None),           // 2 directional_lights (rw)
                uniform_buffer::<ResolveParams>(false),      // 3 params
            ),
        ),
    )
}

/// `RenderStartup`: the resolve pass owns only its params buffer + bind group; the
/// layout lives in `SolariResourceManager`, the pipeline id in `SolariPipelines`.
pub fn init_light_resolve(mut commands: Commands) {
    let mut params = UniformBuffer::<ResolveParams>::default();
    params.set_label(Some("light_resolve"));

    commands.insert_resource(LightResolve {
        light_count: 0,
        params,
        bind_group: None,
    });
}

/// `Render::Prepare`: set the resolve params (slot coverage + world coverage).
pub fn prepare_light_resolve(
    mut resolve: ResMut<LightResolve>,
    table: Option<Res<SolariLights>>,
    propagate: Option<Res<TransformPropagate>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(table), Some(propagate)) = (table, propagate) else {
        return;
    };
    resolve.light_count = table.high_water();
    *resolve.params.get_mut() = ResolveParams {
        light_count: table.high_water(),
        node_count: propagate.node_count(),
        ..Default::default()
    };
    resolve.params.write_buffer(&render_device, &render_queue);
}

/// `Render::PrepareBindGroups`: build the resolve bind group **once**. Every buffer
/// it binds — `world`, the directional-light + transform-slot sparse columns — has
/// a stable handle across growth, so once built it stays valid; no per-frame rebuild.
pub fn prepare_light_resolve_bind_group(
    mut resolve: ResMut<LightResolve>,
    resource_manager: Option<Res<SolariResourceManager>>,
    directional: Option<Res<GpuColumn<DirectionalLightColumn>>>,
    transform_slot: Option<Res<GpuColumn<LightTransformSlotColumn>>>,
    propagate: Option<Res<TransformPropagate>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    if resolve.bind_group.is_some() {
        return;
    }
    let (Some(resource_manager), Some(directional), Some(transform_slot), Some(propagate), Some(params)) = (
        resource_manager,
        directional,
        transform_slot,
        propagate,
        resolve.params.binding(),
    ) else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&resource_manager.light_resolve);
    resolve.bind_group = Some(render_device.create_bind_group(
        "light_resolve",
        &layout,
        &BindGroupEntries::sequential((
            propagate.current_world().as_entire_binding(),
            transform_slot.buffer().as_entire_binding(),
            directional.buffer().as_entire_binding(),
            params,
        )),
    ));
}

/// `RenderGraph` (after transform propagation): fill each directional light's
/// `direction_to_light` from `world[transform_slot]`. Runs after the column
/// scatter (settings) and before the path tracer reads `directional_lights`.
pub fn dispatch_light_resolve(
    resolve: Option<Res<LightResolve>>,
    pipelines: Res<SolariPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let Some(resolve) = resolve else {
        return;
    };
    if resolve.light_count == 0 {
        return;
    }
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.light_resolve) else {
        return;
    };
    let Some(bind_group) = resolve.bind_group.as_ref() else {
        return;
    };
    let groups = resolve.light_count.div_ceil(WORKGROUP_SIZE);
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("light_resolve"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// Registers [`SolariDirectionLight`], its `gpu_table!` (columns + slot index +
/// change-driven extract), the active-light list, and the GPU direction-resolve
/// pass. Added by [`crate::SolariPlugin`].
pub struct SolariLightsPlugin;

impl Plugin for SolariLightsPlugin {
    fn build(&self, app: &mut App) {
        // `light_resolve.wgsl` is embedded centrally in `crate::pipelines`.
        app.register_type::<SolariDirectionLight>();
        // Columns, slot index, change-driven extract, Cleanup clear — all generated.
        app.add_plugins(SolariLightsTablePlugin);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ActiveDirectionalLights>()
            .init_resource::<EmissiveLights>()
            .add_systems(RenderStartup, init_light_resolve.after(SolariSetup))
            .add_systems(
                Render,
                (
                    prepare_emissive_lights.in_set(RenderSystems::Prepare),
                    prepare_light_resolve.in_set(RenderSystems::Prepare),
                    prepare_light_resolve_bind_group.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                RenderGraph,
                // World is written by `dispatch_transform_propagate` (Propagate);
                // the light columns scatter in `Scatter` (earlier set) — so by here
                // settings + slots are in place and the world is ready.
                dispatch_light_resolve
                    .run_if(resource_exists::<SolariPipelines>)
                    .in_set(SolariClusterSystems::Propagate)
                    .after(dispatch_transform_propagate),
            );
    }
}
