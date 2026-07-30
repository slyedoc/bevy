//! Ray-traced hair via NV linear swept spheres (`VK_NV_ray_tracing_linear_swept_spheres`).
//!
//! A [`HairMesh3d`] component references a [`HairAsset`] (strands of control
//! points + radii), paired with a [`HairMaterial3d`] fiber appearance. The asset
//! is uploaded and built into one linear-swept-sphere BLAS by
//! [`manager`]; each hair entity becomes a PTLAS instance referencing that BLAS
//! (see [`ptlas_hair`]); ray hits on hair shade with a Chiang fiber BSDF
//! (`hair.wgsl`), reconstructing the fiber tangent + surface point from the hit
//! segment's two endpoints (`segments` arena).
//!
//! Blackwell-only (4th-gen RT cores). On any other adapter the extension isn't
//! advertised, the [`HairManager`] is never inserted, and every hair system
//! early-outs — the rest of the path tracer is unaffected.

pub mod asset;
pub mod loader;
pub mod manager;
pub mod ptlas_hair;

pub use asset::{HairAsset, HairStrand};
pub use loader::HairLoader;
pub use manager::HairManager;

use bevy_app::{App, Plugin};
use bevy_asset::{Asset, AssetApp, AssetId, Assets, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    prelude::ReflectComponent,
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res, ResMut},
    template::FromTemplate,
};
use bevy_math::Vec3;
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{
    render_resource::{RawBufferVec, ShaderType, StorageBuffer},
    renderer::{RenderDevice, RenderGraph, RenderQueue},
    sync_world::SyncToRenderWorld,
    Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_transform::components::Transform;
use bytemuck::{Pod, Zeroable};
use derive_more::derive::From;

use crate::ecs_gpu::GpuSlot;
use crate::material::material_slots::MaterialSlots;
use crate::material::StandardSolariMaterial;
use crate::transform::TransformGraph;
use crate::{SolariClusterSystems, SolariSetup};

/// [`GpuHairInstance::material_id`] sentinel: shade as fiber hair (Chiang BSDF),
/// not as an opaque [`StandardSolariMaterial`] surface. Mirrors WGSL `HAIR_MATERIAL_NONE`.
pub const HAIR_MATERIAL_NONE: u32 = 0xFFFF_FFFF;

/// A hair/fur groom on an entity. The entity's `GlobalTransform` places the
/// groom in the world; the [`HairAsset`] supplies the strand geometry.
///
/// Pairs with [`HairMaterial3d`] (required, so a groom with no explicit material
/// resolves to the default [`HairMaterial`]) — the swept-sphere counterpart to
/// [`RaytracingMesh3d`](crate::bindings::RaytracingMesh3d) + `SolariMaterial3d`.
#[derive(
    Component, FromTemplate, Clone, Debug, Default, Deref, DerefMut, Reflect, PartialEq, Eq, From,
)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(HairMaterial3d, Transform, SyncToRenderWorld)]
pub struct HairMesh3d(pub Handle<HairAsset>);

/// Component holding the [`HairMaterial`] handle for a [`HairMesh3d`] groom — the
/// fiber appearance the Chiang BSDF shades its hits with.
#[derive(
    Component, FromTemplate, Clone, Debug, Default, Deref, DerefMut, Reflect, PartialEq, Eq, From,
)]
#[reflect(Component, Default, Clone, PartialEq)]
pub struct HairMaterial3d(pub Handle<HairMaterial>);

/// Branches/twigs (or any opaque swept-sphere geometry) rendered as ray-traced
/// linear swept spheres — the **same** LSS geometry as [`HairMesh3d`], but shaded as an
/// opaque BRDF surface with a normal [`StandardSolariMaterial`] (bark/wood) and the
/// round-cone surface normal, instead of the fiber BSDF. The entity's
/// `GlobalTransform` places it; the [`HairAsset`] supplies the strands (each a
/// node polyline with per-point radii). Blackwell-only, like [`HairMesh3d`].
#[derive(Component, Clone, Debug)]
#[require(Transform, SyncToRenderWorld)]
pub struct SolariBranches {
    /// Swept-sphere strand geometry (branch polylines + pipe radii).
    pub strands: Handle<HairAsset>,
    /// Opaque surface material (bark/wood/etc).
    pub material: Handle<StandardSolariMaterial>,
}

/// Physically-based hair appearance, parameterized by melanin like UE / NVIDIA
/// RTXCR rather than an arbitrary RGB color: the two natural pigments span every
/// human hair color, with an optional artistic dye on top. The BSDF absorption
/// `σa` is derived from these via [`melanin_absorption`].
///
/// Referenced by [`HairMaterial3d`].
#[derive(Asset, Clone, Copy, Debug, Reflect)]
#[reflect(Default, Clone)]
pub struct HairMaterial {
    /// Overall melanin amount (0 = platinum/white → 1 = black).
    pub melanin: f32,
    /// Eumelanin↔pheomelanin ratio (0 = ashy/brown → 1 = ginger/red).
    pub redness: f32,
    /// Optional artistic dye tint (added in absorption space). White = none.
    pub dye: Vec3,
    /// Longitudinal roughness βm (cuticle scatter along the fiber).
    pub longitudinal_roughness: f32,
    /// Azimuthal roughness βn (scatter around the fiber).
    pub azimuthal_roughness: f32,
    /// Cuticle scale tilt α, radians (~2° typical).
    pub cuticle_tilt: f32,
    /// Index of refraction (~1.55 for keratin).
    pub ior: f32,
}

impl Default for HairMaterial {
    fn default() -> Self {
        // A natural medium brown.
        Self {
            melanin: 0.55,
            redness: 0.1,
            dye: Vec3::ONE,
            longitudinal_roughness: 0.3,
            azimuthal_roughness: 0.3,
            cuticle_tilt: 0.035,
            ior: 1.55,
        }
    }
}

/// The `D` factor in pbrt's reflectance↔absorption mapping (a polynomial in the
/// azimuthal roughness `b`). Shared by [`hair_color_to_absorption`] /
/// [`hair_absorption_to_color`].
fn hair_d_factor(b: f32) -> f32 {
    let b2 = b * b;
    5.969 - 0.215 * b + 2.532 * b2 - 10.73 * b2 * b + 5.574 * b2 * b2 + 0.245 * b2 * b2 * b
}

/// Convert a target reflectance to absorption (UE `HairColorToAbsorption`), used
/// for the artistic dye tint.
fn hair_color_to_absorption(c: Vec3, b: f32) -> Vec3 {
    let d = hair_d_factor(b);
    let f = |x: f32| {
        let l = x.clamp(1e-4, 1.0).ln() / d;
        l * l
    };
    Vec3::new(f(c.x), f(c.y), f(c.z))
}

/// Convert absorption back to the resulting fiber color (UE `HairAbsorptionToColor`).
/// Used to preview the picked color in a swatch.
pub fn hair_absorption_to_color(a: Vec3, b: f32) -> Vec3 {
    let d = hair_d_factor(b);
    Vec3::new(
        (-a.x.max(0.0).sqrt() * d).exp(),
        (-a.y.max(0.0).sqrt() * d).exp(),
        (-a.z.max(0.0).sqrt() * d).exp(),
    )
}

/// Absorption `σa` from melanin (UE `GetHairColorFromMelanin`, pre-`ToColor`):
/// eumelanin + pheomelanin pigments plus the dye tint, in absorption space.
pub fn melanin_absorption(melanin: f32, redness: f32, dye: Vec3, beta_n: f32) -> Vec3 {
    let melanin = melanin.clamp(0.0, 1.0);
    let redness = redness.clamp(0.0, 1.0);
    let concentration = -(1.0 - melanin).max(1e-4).ln();
    let eumelanin = concentration * (1.0 - redness);
    let pheomelanin = concentration * redness;
    eumelanin * Vec3::new(0.506, 0.841, 1.653)
        + pheomelanin * Vec3::new(0.343, 0.733, 1.924)
        + hair_color_to_absorption(dye, beta_n)
}

/// One hair instance as the GPU reads it — bound both by the PTLAS hair-fill
/// pass (`transform_slot`, `blas_address`, `mask`) and by the path tracer at
/// shade time (`transform_slot`, `segment_base`, fiber params). The world
/// transform is NOT stored here: the hair entity is a transform-table node, so
/// its world matrix already lives in the GPU `world` buffer at `transform_slot`
/// (both passes read `world[transform_slot]`, like directional lights). Mirrors
/// `GpuHairInstance` in WGSL. `repr(C)`, 64 bytes, 16-aligned.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuHairInstance {
    /// BSDF absorption derived from the material's melanin (+ dye).
    pub sigma_a: Vec3,
    /// Transform-table node slot — index into the GPU `world` buffer (×3 rows).
    pub transform_slot: u32,
    /// Global base into the `segments` arena (added to a hit's primitive index).
    pub segment_base: u32,
    pub beta_m: f32,
    pub beta_n: f32,
    pub alpha: f32,
    pub ior: f32,
    pub blas_address_lo: u32,
    pub blas_address_hi: u32,
    /// 8-bit PTLAS instance cull mask (hair is visible to all view masks).
    pub mask: u32,
    /// Opaque-surface material slot ([`SolariBranches`]), or [`HAIR_MATERIAL_NONE`]
    /// for fiber hair — the hair closest-hit branches on this to BRDF shading.
    pub material_id: u32,
    pub _pad: [u32; 3],
}

/// Render-world copy of one extracted hair/branch entity, before its asset
/// residency is resolved.
pub struct ExtractedHairInstance {
    pub transform_slot: u32,
    pub asset: AssetId<HairAsset>,
    /// Fiber appearance (used only for [`HairMesh3d`]; ignored when `surface` is set).
    pub material: HairMaterial,
    /// Opaque surface material for [`SolariBranches`] (`None` = fiber hair).
    pub surface: Option<AssetId<StandardSolariMaterial>>,
}

/// Render-world list of this frame's hair entities (rebuilt each frame — hair
/// counts are small).
#[derive(Resource, Default)]
pub struct ExtractedHairInstances(pub Vec<ExtractedHairInstance>);

/// Scene-group buffer: the hair PTLAS index range, so the path tracer can map a
/// hit's `instance_index` to a hair index. A **storage** buffer (not uniform) —
/// the scene bind group has a texture binding array, and Vulkan forbids mixing a
/// binding array with a uniform buffer in one group. Mirrors WGSL `HairSceneParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, ShaderType)]
pub struct HairSceneParams {
    pub base: u32,
    pub count: u32,
    pub pad0: u32,
    pub pad1: u32,
}

/// Render-world GPU buffer of resolved [`GpuHairInstance`] records + the PTLAS
/// instance-index base assigned to hair (set by the PTLAS hair pass each frame).
#[derive(Resource)]
pub struct HairInstances {
    pub buffer: RawBufferVec<GpuHairInstance>,
    /// Number of resident hair instances written this frame.
    pub count: u32,
    /// PTLAS `instance_index` of hair instance 0 (= cluster slot high-water at
    /// build time). A hit's hair index is `instance_index - base`.
    pub base: u32,
    /// `(base, count)` storage buffer bound in the scene group for the path tracer.
    pub params: StorageBuffer<HairSceneParams>,
}

/// `RenderStartup`: the hair instance buffer.
pub fn init_hair_instances(mut commands: Commands) {
    let mut buffer = RawBufferVec::<GpuHairInstance>::new(wgpu::BufferUsages::STORAGE);
    buffer.set_label(Some("hair.instances"));
    let mut params = StorageBuffer::<HairSceneParams>::default();
    params.set_label(Some("hair.scene_params"));
    commands.insert_resource(HairInstances {
        buffer,
        count: 0,
        base: 0,
        params,
    });
}

/// `ExtractSchedule`: copy hair entities (transform-table slot + asset +
/// material) into the render world. The world transform stays on the GPU — the
/// hair entity is a transform-table node (it has `GlobalTransform`), so its
/// `GpuSlot<TransformGraph>` indexes the propagated `world` buffer that both the
/// PTLAS-write pass and shading read. Entities without a slot yet (first frame,
/// before `PostUpdate` assigns it) are skipped.
pub fn extract_hair_instances(
    hair: Extract<Query<(&GpuSlot<TransformGraph>, &HairMesh3d, &HairMaterial3d)>>,
    branches: Extract<Query<(&GpuSlot<TransformGraph>, &SolariBranches)>>,
    materials: Extract<Res<Assets<HairMaterial>>>,
    mut extracted: ResMut<ExtractedHairInstances>,
) {
    extracted.0.clear();
    for (slot, mesh, material) in &hair {
        // The fiber params are read out of the asset here (they end up inline in
        // the `GpuHairInstance`); a groom whose material hasn't loaded yet waits
        // for a later frame rather than flashing the default brown.
        let Some(material) = materials.get(&material.0) else {
            continue;
        };
        extracted.0.push(ExtractedHairInstance {
            transform_slot: slot.index(),
            asset: mesh.id(),
            material: *material,
            surface: None,
        });
    }
    // Opaque branch instances share the LSS geometry + instance buffer; the
    // surface material slot is resolved in `prepare_hair_instances`.
    for (slot, branches) in &branches {
        extracted.0.push(ExtractedHairInstance {
            transform_slot: slot.index(),
            asset: branches.strands.id(),
            material: HairMaterial::default(),
            surface: Some(branches.material.id()),
        });
    }
}

/// `Render::Prepare`: resolve each extracted hair entity's asset to its
/// resident BLAS + segment base, building the [`GpuHairInstance`] buffer.
/// Entities whose asset isn't resident yet are skipped (picked up next frame).
/// Also assigns the PTLAS instance base (= the cluster slot high-water), so
/// hair occupies indices above the cluster instances.
pub fn prepare_hair_instances(
    extracted: Res<ExtractedHairInstances>,
    manager: Option<Res<HairManager>>,
    instance_manager: Option<Res<crate::instance::InstanceManager>>,
    material_slots: Option<Res<MaterialSlots>>,
    mut instances: ResMut<HairInstances>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    instances.base = instance_manager
        .as_ref()
        .map(|im| im.slot_high_water())
        .unwrap_or(0);
    let Some(manager) = manager else {
        instances.count = 0;
        return;
    };
    instances.buffer.clear();
    for e in &extracted.0 {
        let Some(entry) = manager.entry(e.asset) else {
            continue;
        };
        let m = &e.material;
        let sigma_a = melanin_absorption(m.melanin, m.redness, m.dye, m.azimuthal_roughness);
        // Opaque branches resolve their surface material to a stable slot; fiber
        // hair (and a not-yet-slotted material) uses the sentinel → Chiang BSDF.
        let material_id = e
            .surface
            .and_then(|asset| material_slots.as_deref().and_then(|s| s.slot_of(asset)))
            .unwrap_or(HAIR_MATERIAL_NONE);
        instances.buffer.push(GpuHairInstance {
            sigma_a,
            transform_slot: e.transform_slot,
            segment_base: entry.segment_base,
            beta_m: m.longitudinal_roughness,
            beta_n: m.azimuthal_roughness,
            alpha: m.cuticle_tilt,
            ior: m.ior,
            blas_address_lo: (entry.blas_address & 0xFFFF_FFFF) as u32,
            blas_address_hi: (entry.blas_address >> 32) as u32,
            mask: 0xFF,
            material_id,
            _pad: [0; 3],
        });
    }
    instances.count = instances.buffer.len() as u32;
    // Keep a live buffer for the bind group even with zero hair.
    if instances.buffer.is_empty() {
        instances.buffer.push(GpuHairInstance {
            material_id: HAIR_MATERIAL_NONE,
            ..Default::default()
        });
    }
    instances
        .buffer
        .write_buffer(&render_device, &render_queue);

    let (base, count) = (instances.base, instances.count);
    *instances.params.get_mut() = HairSceneParams {
        base,
        count,
        pad0: 0,
        pad1: 0,
    };
    instances.params.write_buffer(&render_device, &render_queue);
}

/// Registers the [`HairAsset`] + [`HairMaterial`] assets, their components, the
/// geometry manager (arenas + LSS BLAS builds), instance extraction/prep, and the
/// PTLAS hair injection. Added by [`crate::SolariPlugin`].
pub struct HairPlugin;

impl Plugin for HairPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<HairAsset>()
            .init_asset_loader::<HairLoader>()
            .init_asset::<HairMaterial>()
            // `ReflectAsset` on `HairMaterial` + `ReflectHandle` on its handle, so
            // `.bsn` scenes can define hair materials inline (as
            // `StandardSolariMaterial` does).
            .register_asset_reflect::<HairMaterial>()
            .register_type::<HairMaterial>()
            .register_type::<HairMesh3d>()
            .register_type::<HairMaterial3d>();
        // A groom with no explicit `HairMaterial3d` gets the default handle
        // (`HairMesh3d` requires the component), so seed a material there —
        // mirroring how `StandardSolariMaterialPlugin` seeds its default.
        app.world_mut()
            .resource_mut::<Assets<HairMaterial>>()
            .insert(&Handle::<HairMaterial>::default(), HairMaterial::default())
            .unwrap();
        // The Chiang fiber BSDF (`bevy_solari::hair`), imported by `chit_hair`.
        bevy_shader::load_shader_library!(app, "hair.wgsl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ExtractedHairInstances>()
            .add_systems(
                RenderStartup,
                (
                    manager::init_hair_manager,
                    init_hair_instances,
                    ptlas_hair::init_hair_ptlas_write,
                )
                    .after(SolariSetup),
            )
            .add_systems(
                ExtractSchedule,
                (manager::extract_hair_assets, extract_hair_instances),
            )
            .add_systems(
                Render,
                (
                    manager::prepare_hair_geometry.in_set(RenderSystems::PrepareResources),
                    // Assigns the PTLAS hair base + count; must run before the
                    // PTLAS sizes its instance space.
                    prepare_hair_instances
                        .in_set(RenderSystems::PrepareResources)
                        .after(manager::prepare_hair_geometry)
                        // Needs each branch material's stable slot for `material_id`.
                        .after(crate::material::prepare_material_slots)
                        .before(crate::accel::ptlas::prepare_ptlas_params),
                    ptlas_hair::prepare_hair_ptlas_write
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_hair_instances),
                    ptlas_hair::prepare_hair_ptlas_write_bind_group
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                RenderGraph,
                manager::dispatch_hair_blas.in_set(SolariClusterSystems::BuildBlas),
            );
    }
}
