//! glTF → [`SolariMaterial`] loading (a `bevy_gltf` extension handler).
//!
//! With `PbrPlugin` disabled, bevy_pbr's glTF→`StandardMaterial` handler isn't
//! registered, so glTF scenes would get no materials. This handler emits a
//! [`SolariMaterial`] for each glTF material at load — at the same `{material}/std`
//! label bevy_pbr uses, since under the full-RT path only this handler runs — and
//! assigns [`SolariMaterial3d`] to the spawned mesh entities. Registered by
//! [`SolariMaterialPlugin`](super::SolariMaterialPlugin) under the `gltf` feature.

use bevy_app::App;
use bevy_asset::{Handle, LoadContext};
use bevy_ecs::world::EntityWorldMut;
use bevy_gltf::{
    extensions::{ErasedGltfExtensionHandler, GltfExtensionHandler, GltfExtensionHandlers},
    gltf, GltfAssetLabel, GltfLoaderSettings, GltfMaterial,
};

use crate::material::{SolariMaterial, SolariMaterial3d};

/// Convert a [`GltfMaterial`] to a [`SolariMaterial`] — the fields the RT path
/// reads. Mirrors `bevy_pbr::standard_material_from_gltf_material`, minus the
/// raster-only fields. `GltfMaterial`'s field types line up exactly.
///
/// Solari-only authoring with no standard glTF extension travels in the
/// material's `extras` (e.g. `"extras": {"nested_priority": 5}` for
/// nested-dielectric priority, as written by the bistro asset pipeline from
/// NVIDIA's pyscene data); `None` for the synthesized default material.
fn solari_material_from_gltf(
    material: &GltfMaterial,
    gltf_material: Option<&gltf::Material>,
) -> SolariMaterial {
    let extras = gltf_material
        .and_then(|m| m.extras().as_ref())
        .and_then(|extras| serde_json::from_str::<serde_json::Value>(extras.get()).ok());
    let nested_priority = extras
        .as_ref()
        .and_then(|value| value.get("nested_priority")?.as_u64())
        .map_or(0, |priority| priority as u32);

    SolariMaterial {
        base_color: material.base_color,
        base_color_texture: material.base_color_texture.clone(),
        emissive: material.emissive,
        emissive_texture: material.emissive_texture.clone(),
        perceptual_roughness: material.perceptual_roughness,
        metallic: material.metallic,
        metallic_roughness_texture: material.metallic_roughness_texture.clone(),
        reflectance: material.reflectance,
        specular_transmission: material.specular_transmission,
        ior: material.ior,
        attenuation_distance: material.attenuation_distance,
        attenuation_color: material.attenuation_color,
        nested_priority,
        normal_map_texture: material.normal_map_texture.clone(),
    }
}

#[derive(Default, Clone)]
struct SolariGltfMaterialHandler;

impl GltfExtensionHandler for SolariGltfMaterialHandler {
    fn dyn_clone(&self) -> Box<dyn ErasedGltfExtensionHandler> {
        Box::new(self.clone())
    }

    fn on_root(
        &mut self,
        load_context: &mut LoadContext<'_>,
        _gltf: &gltf::Gltf,
        _settings: &GltfLoaderSettings,
    ) {
        // The glTF default material, for meshes without one (`DefaultMaterial/std`).
        let label = format!("{}/std", GltfAssetLabel::DefaultMaterial);
        load_context
            .add_labeled_asset(label, solari_material_from_gltf(&GltfMaterial::default(), None));
    }

    fn on_material(
        &mut self,
        load_context: &mut LoadContext<'_>,
        gltf_material: &gltf::Material,
        _material: Handle<GltfMaterial>,
        material_asset: &GltfMaterial,
        material_label: &str,
    ) {
        let label = format!("{material_label}/std");
        let material = solari_material_from_gltf(material_asset, Some(gltf_material));
        load_context.add_labeled_asset(label, material);
    }

    fn on_spawn_mesh_and_material(
        &mut self,
        load_context: &mut LoadContext<'_>,
        _primitive: &gltf::Primitive,
        _mesh: &gltf::Mesh,
        _material: &gltf::Material,
        entity: &mut EntityWorldMut,
        material_label: &str,
    ) {
        let label = format!("{material_label}/std");
        let handle = load_context.get_label_handle::<SolariMaterial>(label);
        entity.insert(SolariMaterial3d(handle));
    }
}

/// Register the glTF → [`SolariMaterial`] handler into `bevy_gltf`'s handler list.
/// Called by [`SolariMaterialPlugin`](super::SolariMaterialPlugin) when PbrPlugin's
/// own glTF handler is absent (the full-RT path). No-op if `GltfPlugin` isn't
/// present (no `GltfExtensionHandlers` resource) — solari works without glTF.
///
/// Must run during plugin `build`: `GltfPlugin::finish` snapshots the handler list
/// into the loader, so a handler pushed after that point is never seen.
pub(super) fn register_gltf_material_handler(app: &mut App) {
    let Some(handlers) = app.world().get_resource::<GltfExtensionHandlers>() else {
        return;
    };
    let handlers = handlers.0.clone();

    #[cfg(not(target_family = "wasm"))]
    handlers
        .write_blocking()
        .push(Box::new(SolariGltfMaterialHandler));

    #[cfg(target_family = "wasm")]
    bevy_tasks::block_on(async {
        handlers.write().await.push(Box::new(SolariGltfMaterialHandler));
    });
}
