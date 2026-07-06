//! `bevy_solari`'s own material type and its plugin to avoid bevy_pbr systems and cpu time
//!
//! While `PbrPlugin` is still enabled, [`StandardSolariMaterial`] also provides a
//! `From<&StandardMaterial>` bridge so existing content (code-authored or
//! glTF-loaded — both arrive as `StandardMaterial`) can be converted to a
//! `StandardSolariMaterial` at ray-tracing conversion time without re-authoring assets.

use bevy_app::{App, Plugin};
use bevy_asset::{Asset, AssetApp, AssetId, Assets, Handle, RenderAssetUsages};
use bevy_color::{Color, LinearRgba};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{component::Component, prelude::ReflectComponent, template::FromTemplate};
use bevy_image::{Image, ImageSampler};
use bevy_material::AlphaMode;
use bevy_pbr::{DfgLut, StandardMaterial};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::RenderApp;
use derive_more::derive::From;

pub mod material_slots;
pub use material_slots::{
    init_material_slots, material_sbt_class, prepare_material_slots,
    prepare_material_traversal_flags, MaterialSlots, MaterialTraversalFlags,
};

#[cfg(feature = "gltf")]
mod gltf;
#[cfg(feature = "gltf")]
pub(crate) use gltf::nested_priority_from_extras;

/// A physically-based material consumed by the `bevy_solari` ray tracer. Mirrors
/// the `StandardMaterial` fields the scene binder reads; see the module docs for
/// why it is intentionally not a `Material`.
///
/// Pair with [`SolariMaterial3d`] and [`crate::bindings::RaytracingMesh3d`].
#[derive(Asset, Clone, Debug, Reflect)]
#[reflect(Default, Clone)]
pub struct StandardSolariMaterial {
    /// Base ("albedo") color. Linearized into the `GpuMaterial`.
    pub base_color: Color,
    /// Optional base-color texture (sampled at the ray hit's UV).
    pub base_color_texture: Option<Handle<Image>>,
    /// Emitted radiance. A non-black value turns the mesh into an emissive light
    /// source for next-event estimation.
    pub emissive: LinearRgba,
    /// Optional emissive texture.
    pub emissive_texture: Option<Handle<Image>>,
    /// Perceptual roughness in `[0, 1]` (remapped to `a = r*r` in the BRDF).
    /// `0.0` is a perfect mirror.
    pub perceptual_roughness: f32,
    /// Metalness in `[0, 1]`.
    pub metallic: f32,
    /// Optional metallic (B) + roughness (G) packed texture.
    pub metallic_roughness_texture: Option<Handle<Image>>,
    /// Dielectric specular reflectance at normal incidence (`F0`), scaled into
    /// `[0, 0.16]` like `StandardMaterial`.
    pub reflectance: f32,
    /// Fraction of light passing through the surface as specular transmission
    /// (glass/water). `> 0` makes the ray tracer refract through the mesh
    /// (treated as a closed volume) instead of shading it opaque.
    pub specular_transmission: f32,
    /// Index of refraction for transmissive surfaces (1.5 ≈ glass, 1.33 water).
    pub ior: f32,
    /// Strength of chromatic dispersion — how much the IOR varies across the
    /// visible spectrum (`KHR_materials_dispersion` convention: `20 / Abbe
    /// number`, with [`Self::ior`] as the spectrum-center value). `0.0` = no
    /// dispersion. Crown glass ≈ 0.34, dense flint ≈ 0.55, diamond ≈ 0.63.
    /// Refractions then split into rainbow fringes per wavelength.
    pub dispersion: f32,
    /// Distance (world units) light travels inside the volume before
    /// [`Self::attenuation_color`] remains (Beer–Lambert). `INFINITY` = clear.
    pub attenuation_distance: f32,
    /// The color remaining after white light travels [`Self::attenuation_distance`]
    /// through the volume.
    pub attenuation_color: Color,
    /// Nested-dielectric priority for transmissive volumes that overlap
    /// (wine modeled interpenetrating its glass): at a boundary inside a
    /// higher-priority volume the boundary is ignored, so the higher priority
    /// wins the overlap region. Read from the glTF material's
    /// `extras.nested_priority`. `0` = no nesting expected.
    pub nested_priority: u32,
    /// How the base-color texture's alpha channel is applied — bevy's
    /// [`AlphaMode`]. The ray tracer alpha-tests candidate hits during BVH
    /// traversal, so cutouts (foliage, fences, grates) hold for shading,
    /// reflections, refractions, and shadows alike. [`AlphaMode::Mask`] tests
    /// at its cutoff; every other non-opaque mode is approximated as a 0.5
    /// cutout (true stochastic alpha blending is not implemented).
    pub alpha_mode: AlphaMode,
    /// Optional tangent-space normal map.
    pub normal_map_texture: Option<Handle<Image>>,
    /// Optional depth map (bevy `StandardMaterial::depth_map` convention: a brighter
    /// texel is DEEPER, so the surface recedes there). Sampled per generated vertex
    /// during tessellation and used to offset the vertex along its interpolated normal
    /// by `depth_bias - depth * depth_scale` (world units).
    pub depth_map: Option<Handle<Image>>,
    /// Scales the `[0, 1]` depth-map value before it offsets the surface.
    pub depth_scale: f32,
    /// Added to the (negated) scaled depth (so a mid-grey-centered map can push the
    /// surface both inward and outward).
    pub depth_bias: f32,
    /// Optional layered (`texture_2d_array`) textures for a custom `chit_class`
    /// shader (terrain layer painting); sampled via `sample_texture_array`.
    pub texture_array_a: Option<Handle<Image>>,
    /// Second layered-texture slot (e.g. packed AO/roughness/metallic layers).
    pub texture_array_b: Option<Handle<Image>>,
    /// Third layered-texture slot (e.g. tangent-space normal layers).
    pub texture_array_c: Option<Handle<Image>>,
    /// SBT hit-group class (`0` = opaque/glass routing); set from a [`SolariMaterialClass<S>`](crate::SolariMaterialClass).
    pub chit_class: u32,
    /// Per-material data for the `chit_class` shader, read as `load_material_bindless(id).chit_data`.
    pub chit_data: [u32; 4],
}

impl StandardSolariMaterial {
    /// A material with the given base color and otherwise-default properties.
    /// Mirrors `StandardMaterial::from_color`.
    pub fn from_color(color: impl Into<Color>) -> Self {
        Self {
            base_color: color.into(),
            ..Default::default()
        }
    }

    /// The alpha cutoff `trace_ray`'s traversal alpha test uses, or a negative
    /// value for "opaque — commit hits in hardware, never invoke the test".
    /// Only [`AlphaMode::Mask`] (a genuine cutout) is alpha-tested, at its
    /// cutoff. Everything else — opaque, and alpha-`Blend` (a binary any-hit
    /// test can't express blending) — traverses as opaque; forcing non-opaque
    /// traversal on every `Blend` material is a large, needless any-hit cost.
    /// Transmissive materials are likewise opaque to traversal (refraction owns
    /// those surfaces).
    pub fn traversal_alpha_cutoff(&self) -> f32 {
        if self.specular_transmission > 0.0 {
            return -1.0;
        }
        match self.alpha_mode {
            AlphaMode::Mask(cutoff) => cutoff,
            _ => -1.0,
        }
    }
}

impl Default for StandardSolariMaterial {
    fn default() -> Self {
        // Match `StandardMaterial`'s defaults so ported content and struct-literal
        // authoring behave identically.
        Self {
            base_color: Color::WHITE,
            base_color_texture: None,
            emissive: LinearRgba::BLACK,
            emissive_texture: None,
            perceptual_roughness: 0.5,
            metallic: 0.0,
            metallic_roughness_texture: None,
            reflectance: 0.5,
            specular_transmission: 0.0,
            ior: 1.5,
            dispersion: 0.0,
            attenuation_distance: f32::INFINITY,
            attenuation_color: Color::WHITE,
            nested_priority: 0,
            alpha_mode: AlphaMode::Opaque,
            normal_map_texture: None,
            depth_map: None,
            depth_scale: 1.0,
            depth_bias: 0.0,
            texture_array_a: None,
            texture_array_b: None,
            texture_array_c: None,
            chit_class: 0,
            chit_data: [0; 4],
        }
    }
}

/// Bridge from `bevy_pbr`'s `StandardMaterial` (transitional — lets existing
/// `StandardMaterial`-authored or glTF-loaded content be converted to a
/// `StandardSolariMaterial` while `PbrPlugin` is still enabled). Copies exactly the
/// fields the binder reads. Remove once content authors `StandardSolariMaterial`
/// directly and `bevy_pbr` is dropped from the dependency graph.
impl From<&StandardMaterial> for StandardSolariMaterial {
    fn from(m: &StandardMaterial) -> Self {
        Self {
            base_color: m.base_color,
            base_color_texture: m.base_color_texture.clone(),
            emissive: m.emissive,
            emissive_texture: m.emissive_texture.clone(),
            perceptual_roughness: m.perceptual_roughness,
            metallic: m.metallic,
            metallic_roughness_texture: m.metallic_roughness_texture.clone(),
            reflectance: m.reflectance,
            specular_transmission: m.specular_transmission,
            ior: m.ior,
            // `StandardMaterial` has no dispersion field.
            dispersion: 0.0,
            attenuation_distance: m.attenuation_distance,
            attenuation_color: m.attenuation_color,
            nested_priority: 0,
            alpha_mode: m.alpha_mode,
            normal_map_texture: m.normal_map_texture.clone(),
            // `StandardMaterial` has no displacement concept — authored directly on
            // `StandardSolariMaterial` (or via the `.bsn` importer).
            depth_map: None,
            depth_scale: 1.0,
            depth_bias: 0.0,
            texture_array_a: None,
            texture_array_b: None,
            texture_array_c: None,
            // Custom hit-group routing (portal, planet, …) is opt-in via `chit_class`.
            chit_class: 0,
            chit_data: [0; 4],
        }
    }
}

/// Component holding a [`StandardSolariMaterial`] handle for a ray-tracing instance — the
/// `Material`-free counterpart to `MeshMaterial3d`.
///
/// Modeled on `bevy_pbr::MeshMaterial3d` (newtype over a `Handle`) but with no
/// `M: Material` bound, so it works with `PbrPlugin` disabled.
#[derive(
    Component, FromTemplate, Clone, Debug, Default, Deref, DerefMut, Reflect, PartialEq, Eq, From,
)]
#[reflect(Component, Default, Clone, PartialEq)]
pub struct SolariMaterial3d(pub Handle<StandardSolariMaterial>);

impl From<SolariMaterial3d> for AssetId<StandardSolariMaterial> {
    fn from(material: SolariMaterial3d) -> Self {
        material.id()
    }
}

impl From<&SolariMaterial3d> for AssetId<StandardSolariMaterial> {
    fn from(material: &SolariMaterial3d) -> Self {
        material.id()
    }
}

/// Registers [`StandardSolariMaterial`]. Added by [`crate::SolariPlugin`] ahead of the
/// binding/instance plugins so the asset exists before anything binds it.
///
/// Independent of `PbrPlugin`: a single `init_asset` plus a default material at
/// the default handle (mirroring how `PbrPlugin` seeds `StandardMaterial`) —
/// instances with no explicit material resolve here.
pub struct StandardSolariMaterialPlugin;

impl Plugin for StandardSolariMaterialPlugin {
    fn build(&self, app: &mut App) {
        app
        // add since we are disabling `PbrPlugin` under solari, so it doesn't register `Assets<StandardMaterial>`
        .init_asset::<StandardMaterial>()
        // our custom material
        .init_asset::<StandardSolariMaterial>()
            // `ReflectAsset` on `StandardSolariMaterial` + `ReflectHandle` on its handle, so
            // `.bsn` scenes can define materials inline (see `bevy_scene` dynamic BSN).
            .register_asset_reflect::<StandardSolariMaterial>()
            .register_type::<StandardSolariMaterial>()
            .register_type::<SolariMaterial3d>()
            // `StandardSolariMaterial.alpha_mode` is an `AlphaMode`; `.bsn` scenes name it as
            // `bevy_material::alpha::AlphaMode::Mask(..)`, so it must be in the registry. PbrPlugin
            // (which would otherwise register it) is disabled on the full-RT path.
            .register_type::<AlphaMode>();
        app.world_mut()
            .resource_mut::<Assets<StandardSolariMaterial>>()
            .insert(
                &Handle::<StandardSolariMaterial>::default(),
                StandardSolariMaterial::default(),
            )
            .unwrap();

        // Emit `StandardSolariMaterial` from glTF only when bevy_pbr's own glTF→`StandardMaterial`
        // handler is absent — i.e. `PbrPlugin` disabled (the full-RT path). With both
        // present they'd race to produce the same `{material}/std` label.
        #[cfg(feature = "gltf")]
        if !app.is_plugin_added::<bevy_pbr::PbrPlugin>() {
            gltf::register_gltf_material_handler(app);
        }

        insert_dfg_lut(app);
    }
}

/// Insert `bevy_pbr`'s [`DfgLut`] (the split-sum BRDF integration LUT) from a copy
/// of `dfg.ktx2` embedded in `bevy_solari`. The ray tracer's BRDF samples it, and
/// when `PbrPlugin` is disabled on the full-RT path nothing else provides it. No-op
/// if a `DfgLut` is already present (PbrPlugin enabled), so the two never conflict.
///
/// `Assets<Image>` comes from `bevy_image`'s `ImagePlugin` (always in
/// `DefaultPlugins`), independent of `PbrPlugin`.
fn insert_dfg_lut(app: &mut App) {
    let already_present = app
        .get_sub_app(RenderApp)
        .is_some_and(|render_app| render_app.world().contains_resource::<DfgLut>());
    if already_present {
        return;
    }

    // Self-baked split-sum table (64×64 RG f16) integrating THIS crate's exact
    // sampler/eval pair — tests/bake_dfg.rs regenerates it after any BRDF change.
    // bevy_pbr's dfg.ktx2 disagreed ~2.4% at the roughness-1 row (furnace leak).
    let mut lut = Image::new(
        bevy_render::render_resource::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        bevy_render::render_resource::TextureDimension::D2,
        include_bytes!("dfg_baked.bin").to_vec(),
        bevy_render::render_resource::TextureFormat::Rg16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    lut.sampler = ImageSampler::linear();
    let texture = app.world_mut().resource_mut::<Assets<Image>>().add(lut);

    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.world_mut().insert_resource(DfgLut { texture });
    }
}
