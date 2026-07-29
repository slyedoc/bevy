//! `bevy_solari`'s own directional-light table and the resolve pass that sources
//! the light **direction from the GPU transform table**.
//!
//! The ray tracer needs directional-light data in the render world but shouldn't
//! depend on `bevy_pbr` running — the crate likewise owns its material type via
//! [`crate::material`].
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
use bevy_platform::{collections::{HashMap, HashSet}, hash::FixedHasher};
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
use crate::ecs_gpu::{GpuColumn, GpuSlot, GpuTable, SlotPool};
use crate::instance::InstanceManager;
use crate::material::StandardSolariMaterial;
use crate::pipelines::SolariPipelines;
use crate::resource_manager::SolariResourceManager;
use crate::transform::{dispatch_transform_subtract, TransformGraph, TransformPropagate};
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

/// One entry in the path tracer's light-source table (`light_sources`), shared
/// by emissive-mesh lights and directional lights. Mirrors the WGSL `LightSource`.
#[derive(ShaderType, Clone)]
pub struct GpuLightSource {
    kind: u32,
    id: u32,
}

impl GpuLightSource {
    /// A freed slot in the slot-indexed table: `kind = 0` is an emissive light
    /// with zero triangles, which nothing can sample — consumers treat it as
    /// "no light here" (`LIGHT_SOURCE_KIND_NONE` in WGSL).
    pub const NONE: GpuLightSource = GpuLightSource { kind: 0, id: 0 };

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

/// Identity of one light source, keyed for [`LightSources`]' stable slots.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum LightKey {
    /// An emissive-mesh instance, by its (stable) instance slot.
    Emissive(u32),
    /// A directional light, by its (stable) lights-table slot.
    Directional(u32),
}

/// Render-world light-source table with **stable slots**: each light keeps its
/// `light_sources[]` index for its lifetime (a [`SlotPool`] keyed by instance /
/// lights-table slot; freed slots hole out as [`GpuLightSource::NONE`] and are
/// reused). ReSTIR reservoirs and light-tile samples store the slot as their
/// light identity, so in-flight history survives lights being added or removed
/// — a dense per-frame list would re-index every light on any set change,
/// silently repointing every reservoir at a different light. (A freed slot's
/// *reuse* still briefly misattributes history to the new occupant; bounded by
/// the confidence cap, and gone within a re-cap.)
///
/// Rebuilt by [`prepare_light_sources`] only when the light set could have
/// changed; the scene binder uploads `table` + `active` verbatim.
#[derive(Resource, Default)]
pub struct LightSources {
    /// Slot-indexed table (`pool.len()` entries, holes = `NONE`).
    pub table: Vec<GpuLightSource>,
    /// The uniform-pick list: `[emissive_count, directional_count]` header,
    /// then the active emissive slots, then the active directional slots
    /// (strata contiguous so the stratified pick indexes directly).
    pub active: Vec<u32>,
    pool: SlotPool<LightKey>,
    /// Per-asset emissive luminance the table was built against — a diff catches a
    /// material whose `emissive` was edited (including INTENSITY) without any
    /// instance changing; the values also feed the power-weighted pick CDF.
    cached_flux: HashMap<AssetId<StandardSolariMaterial>, f32, FixedHasher>,
    /// The directional lights-table slots the table was built against.
    cached_directional: Vec<u32>,
    /// Whether the CDF was built in forced-uniform mode ([`SolariUniformLights`]).
    cached_uniform: bool,
}

/// Debug lever: force UNIFORM emissive-light picking instead of the
/// power-weighted CDF. Converged images must match; only variance may differ.
#[derive(Resource, Clone, Default, bevy_render::extract_resource::ExtractResource)]
pub struct SolariUniformLights {
    pub enabled: bool,
}

/// `Render::Prepare`: rebuild the light-source table — but only when the light
/// set could have changed (a material's emissiveness edited, an instance added /
/// released / re-materialed, or a directional light added / removed). On a
/// move-only frame the O(active) walk is skipped and the table reused. (Light
/// *transforms* are GPU-side via the PTLAS; this only tracks which sources
/// exist.)
pub fn prepare_light_sources(
    mut lights: ResMut<LightSources>,
    materials: Res<SolariMaterialAssets>,
    instances: Res<InstanceManager>,
    active_directional: Res<ActiveDirectionalLights>,
    uniform_pick: Option<Res<SolariUniformLights>>,
) {
    // Emissive luminance per non-black material — the set membership AND the
    // per-light flux basis (× triangle count). MUST mirror the WGSL side:
    // `random_emissive_light_pdf` recomputes flux from the BASE (untextured)
    // material emissive with the same Rec.709 luminance.
    let mut emissive_assets =
        HashMap::<AssetId<StandardSolariMaterial>, f32, FixedHasher>::default();
    for (asset_id, material) in materials.iter() {
        let e = material.emissive;
        if e.to_vec3() != Vec3::ZERO {
            let lum = 0.2126 * e.red + 0.7152 * e.green + 0.0722 * e.blue;
            emissive_assets.insert(*asset_id, lum);
        }
    }
    let uniform = uniform_pick.is_some_and(|u| u.enabled);
    let directional: Vec<u32> = active_directional.0.iter().map(|&(_, slot)| slot).collect();

    let changed = emissive_assets != lights.cached_flux
        || uniform != lights.cached_uniform
        || directional != lights.cached_directional
        || !instances.added_slots().is_empty()
        || !instances.released_slots().is_empty()
        || !instances.material_dirty().is_empty();
    if !changed {
        return;
    }

    // The current light set, with each emissive instance's source entry and its
    // pick flux (luminance × triangle count; directional entries carry 0).
    let mut present: Vec<(LightKey, GpuLightSource)> = Vec::new();
    let mut present_flux: Vec<f32> = Vec::new();
    for &slot in instances.active_slots() {
        let asset_id = instances.instance_material_asset_id(slot);
        if let Some(&lum) = emissive_assets.get(&asset_id) {
            let triangle_count = instances.instance_total_triangle_count(slot);
            if triangle_count > 0 && triangle_count <= u16::MAX as u32 {
                present.push((
                    LightKey::Emissive(slot.0),
                    GpuLightSource::new_emissive_mesh_light(slot.0, triangle_count),
                ));
                present_flux.push(lum * triangle_count as f32);
            }
        }
    }
    for &slot in &directional {
        present.push((
            LightKey::Directional(slot),
            GpuLightSource::new_directional_light(slot),
        ));
    }

    let present_keys: HashSet<LightKey, FixedHasher> =
        present.iter().map(|&(key, _)| key).collect();
    let LightSources { pool, table, active, .. } = &mut *lights;
    pool.reconcile(present_keys.iter().copied(), |key| present_keys.contains(&key));
    // Reservoirs pack the slot into 16 bits (`light_id = slot << 16 | triangle`).
    assert!(
        pool.len() <= u16::MAX as u32,
        "too many light sources in the scene, maximum is 65535"
    );

    table.clear();
    table.resize(pool.len() as usize, GpuLightSource::NONE);
    active.clear();
    active.extend([0u32, 0u32]); // [emissive_count, directional_count]
    let mut emissive_flux: Vec<f32> = Vec::new();
    for (i, (key, source)) in present.iter().enumerate() {
        let slot = pool.slot_of(*key).unwrap();
        table[slot as usize] = source.clone();
        if matches!(key, LightKey::Emissive(_)) {
            active.push(slot);
            emissive_flux.push(present_flux[i]);
        }
    }
    active[0] = present.len() as u32 - directional.len() as u32;
    active[1] = directional.len() as u32;
    for (key, _) in &present {
        if let LightKey::Directional(_) = key {
            active.push(pool.slot_of(*key).unwrap());
        }
    }
    // Power-weighted pick CDF (normalized, last entry pinned to exactly 1.0) +
    // trailing total flux, appended as f32 bits — `sampling.wgsl` bitcasts them.
    // total = 0 is the uniform-pick sentinel (forced by [`SolariUniformLights`],
    // or a degenerate all-zero flux set).
    let total: f32 = emissive_flux.iter().sum();
    let n = emissive_flux.len();
    let mut acc = 0.0f32;
    for (i, f) in emissive_flux.iter().enumerate() {
        acc += f;
        let c = if i + 1 == n { 1.0 } else { acc / total.max(f32::MIN_POSITIVE) };
        active.push(c.to_bits());
    }
    let sentinel = if uniform || !(total > 0.0) { 0.0f32 } else { total };
    active.push(sentinel.to_bits());

    lights.cached_flux = emissive_assets;
    lights.cached_directional = directional;
    lights.cached_uniform = uniform;
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

/// `Render::PrepareBindGroups`: (re)build the resolve bind group. Rebuilt every
/// frame — cheap, and immune to a bound buffer changing allocation strategy later.
pub fn prepare_light_resolve_bind_group(
    mut resolve: ResMut<LightResolve>,
    resource_manager: Option<Res<SolariResourceManager>>,
    directional: Option<Res<GpuColumn<DirectionalLightColumn>>>,
    transform_slot: Option<Res<GpuColumn<LightTransformSlotColumn>>>,
    propagate: Option<Res<TransformPropagate>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
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
    let groups = crate::ecs_gpu::linear_dispatch(resolve.light_count.div_ceil(WORKGROUP_SIZE));
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("light_resolve"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups.0, groups.1, groups.2);
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
        app.init_resource::<SolariUniformLights>().add_plugins(
            bevy_render::extract_resource::ExtractResourcePlugin::<SolariUniformLights>::default(),
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ActiveDirectionalLights>()
            .init_resource::<LightSources>()
            .add_systems(RenderStartup, init_light_resolve.after(SolariSetup))
            .add_systems(
                Render,
                (
                    prepare_light_sources.in_set(RenderSystems::PrepareResources),
                    prepare_light_resolve.in_set(RenderSystems::PrepareResources),
                    prepare_light_resolve_bind_group.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                RenderGraph,
                // Resolve reads `current_world()` — the origin-relative `world_rel`
                // written by `dispatch_transform_subtract`, not by the propagate walk
                // that feeds it. The light columns scatter in `Scatter` (earlier set),
                // so by here settings + slots are in place too.
                dispatch_light_resolve
                    .run_if(resource_exists::<SolariPipelines>)
                    .in_set(SolariClusterSystems::Propagate)
                    .after(dispatch_transform_subtract),
            );
    }
}
