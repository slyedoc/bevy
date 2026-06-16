//! Optional convenience systems for getting existing scenes onto the
//! ray-tracing path.
//!
//! These are not added by any plugin — opt in by adding the system you want to
//! your own schedule. See [`convert_meshes_to_raytracing`].

use crate::geometry::ClusterMesh;
use crate::bindings::RaytracingMesh3d;
use crate::material::{SolariMaterial, SolariMaterial3d};
use bevy_asset::{AssetId, Assets, Handle};
use bevy_pbr::{MeshMaterial3d, StandardMaterial};
use bevy_camera::visibility::{InheritedVisibility, NoCpuCulling, Visibility};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    hierarchy::ChildOf,
    query::{With, Without},
    system::{Commands, Local, Query, Res, ResMut},
};
use bevy_mesh::{Mesh, Mesh3d};
use bevy_platform::collections::{HashMap, HashSet};
use tracing::warn;

/// Marks entities whose [`Mesh`] could not be baked into a [`ClusterMesh`]
/// (e.g. it is missing the `UV_0` attribute) so that
/// [`convert_meshes_to_raytracing`] stops retrying them every frame.
#[derive(Component)]
pub struct RaytracingBakeFailed;

/// Opt **this** entity in to ray-tracing conversion by
/// [`convert_marked_meshes_to_raytracing`].
///
/// Add it instead of running the blanket [`convert_meshes_to_raytracing`] when
/// only *some* entities should be ray traced and the rest must keep
/// rasterizing — e.g. one camera renders a ray-traced scene while another
/// renders a normal Bevy scene. Add it to a scene's mesh entities the same way
/// you'd add [`RenderLayers`](bevy_camera::visibility::RenderLayers) — e.g. in
/// the glTF/scene load observer. Conversion otherwise behaves identically
/// (polls until the mesh asset is ready, then swaps `Mesh3d` →
/// `RaytracingMesh3d` and removes `Mesh3d`).
#[derive(Component)]
pub struct ConvertToRaytracing;

/// Bakes each [`Mesh3d`]'s [`Mesh`] into a [`ClusterMesh`] and swaps [`Mesh3d`]
/// for [`RaytracingMesh3d`], removing [`Mesh3d`] so the entity is rendered
/// purely by the ray tracer (no rasterization).
///
/// This is intended for scenes whose meshes are spawned by something other than
/// the app itself — e.g. glTF or scene assets — where you can't add
/// [`RaytracingMesh3d`] at spawn time. Add it to [`Update`] to convert meshes as
/// they appear:
///
/// ```ignore
/// app.add_systems(Update, convert_meshes_to_raytracing);
/// ```
///
/// It polls rather than reacting to spawn events because a [`Mesh`] asset may
/// not be loaded the frame its entity appears; entities whose mesh isn't ready
/// keep their [`Mesh3d`] and are retried next frame. Converted entities gain
/// [`RaytracingMesh3d`] (and lose [`Mesh3d`]), so they no longer match the query.
///
/// **Removing [`Mesh3d`] only suits the pure path tracer.** The realtime
/// `SolariLighting` path relies on the rasterized G-buffer and needs [`Mesh3d`]
/// to remain; for that case write your own variant that keeps it.
///
/// The `cache` [`Local`] dedups `Mesh` → `ClusterMesh` bakes, so a mesh shared
/// across many entities is only baked once.

pub fn convert_meshes_to_raytracing(
    mut commands: Commands,
    query: Query<(Entity, &Mesh3d), (Without<RaytracingMesh3d>, Without<RaytracingBakeFailed>)>,
    parents: Query<&ChildOf>,
    meshes: Res<Assets<Mesh>>,
    mut cluster_meshes: ResMut<Assets<ClusterMesh>>,
    mut cache: Local<HashMap<AssetId<Mesh>, Handle<ClusterMesh>>>,
    mut stripped: Local<HashSet<Entity>>,
) {
    for (entity, mesh3d) in &query {
        bake_and_swap(
            &mut commands,
            entity,
            mesh3d,
            &parents,
            &meshes,
            &mut cluster_meshes,
            &mut cache,
            &mut stripped,
        );
    }
}

/// Bridge existing [`StandardMaterial`]-authored content onto `bevy_solari`'s
/// own [`SolariMaterial`]. [`RaytracingMesh3d`] requires [`SolariMaterial3d`]
/// (the path tracer reads `SolariMaterial`, not `bevy_pbr`'s raster material),
/// but most scenes — code-authored or glTF-loaded — arrive as `StandardMaterial`.
/// Add this to [`Update`] alongside [`convert_meshes_to_raytracing`] to convert
/// each ray-traced mesh's material:
///
/// ```ignore
/// app.add_systems(
///     Update,
///     (convert_meshes_to_raytracing, convert_standard_materials_to_solari),
/// );
/// ```
///
/// It polls (a material whose [`StandardMaterial`] asset isn't loaded yet is
/// retried next frame) and dedups by source asset via the `cache` [`Local`], so a
/// material shared across many meshes is converted once. Removing
/// [`MeshMaterial3d`] takes the entity out of `bevy_pbr`'s raster material path
/// and stops it re-matching.
///
/// Transitional: it exists to bridge `StandardMaterial` content while `bevy_pbr`
/// is still a dependency. Author [`SolariMaterial`] directly to skip it entirely.
pub fn convert_standard_materials_to_solari(
    mut commands: Commands,
    query: Query<(Entity, &MeshMaterial3d<StandardMaterial>), With<RaytracingMesh3d>>,
    std_materials: Res<Assets<StandardMaterial>>,
    mut solari_materials: ResMut<Assets<SolariMaterial>>,
    mut cache: Local<HashMap<AssetId<StandardMaterial>, Handle<SolariMaterial>>>,
    #[cfg(feature = "gltf")] extras_query: Query<&bevy_gltf::GltfMaterialExtras>,
) {
    for (entity, mesh_material) in &query {
        let std_id = mesh_material.0.id();
        let handle = if let Some(handle) = cache.get(&std_id) {
            handle.clone()
        } else {
            let Some(std_material) = std_materials.get(std_id) else {
                continue; // asset not loaded yet — retry next frame
            };
            #[allow(unused_mut)]
            let mut material = SolariMaterial::from(std_material);
            // Solari-only authoring with no glTF extension (nested-dielectric
            // priorities) rides the material's extras — `StandardMaterial`
            // drops them, but bevy_gltf leaves the raw JSON on the entity.
            #[cfg(feature = "gltf")]
            if let Ok(extras) = extras_query.get(entity) {
                material.nested_priority =
                    crate::material::nested_priority_from_extras(&extras.value);
            }
            let handle = solari_materials.add(material);
            cache.insert(std_id, handle.clone());
            handle
        };
        commands
            .entity(entity)
            .insert(SolariMaterial3d(handle))
            .remove::<MeshMaterial3d<StandardMaterial>>();
    }
}

/// Opt-in counterpart to [`convert_meshes_to_raytracing`]: converts only the
/// entities tagged [`ConvertToRaytracing`], leaving every other [`Mesh3d`] to
/// rasterize. Use it when a scene mixes ray-traced and rasterized meshes (e.g.
/// a split-screen with one Solari camera and one normal Bevy camera). Add it to
/// [`Update`] the same way:
///
/// ```ignore
/// app.add_systems(Update, convert_marked_meshes_to_raytracing);
/// ```
pub fn convert_marked_meshes_to_raytracing(
    mut commands: Commands,
    query: Query<
        (Entity, &Mesh3d),
        (
            With<ConvertToRaytracing>,
            Without<RaytracingMesh3d>,
            Without<RaytracingBakeFailed>,
        ),
    >,
    parents: Query<&ChildOf>,
    meshes: Res<Assets<Mesh>>,
    mut cluster_meshes: ResMut<Assets<ClusterMesh>>,
    mut cache: Local<HashMap<AssetId<Mesh>, Handle<ClusterMesh>>>,
    mut stripped: Local<HashSet<Entity>>,
) {
    for (entity, mesh3d) in &query {
        bake_and_swap(
            &mut commands,
            entity,
            mesh3d,
            &parents,
            &meshes,
            &mut cluster_meshes,
            &mut cache,
            &mut stripped,
        );
    }
}

/// Bake one entity's [`Mesh`] into a [`ClusterMesh`] (deduped via `cache`) and
/// swap [`Mesh3d`] → [`RaytracingMesh3d`]. Leaves [`Mesh3d`] in place if the
/// mesh asset isn't loaded yet (retried next frame); tags
/// [`RaytracingBakeFailed`] if the bake fails. Shared by the blanket and
/// marker-gated conversion systems.
fn bake_and_swap(
    commands: &mut Commands,
    entity: Entity,
    mesh3d: &Mesh3d,
    parents: &Query<&ChildOf>,
    meshes: &Assets<Mesh>,
    cluster_meshes: &mut Assets<ClusterMesh>,
    cache: &mut HashMap<AssetId<Mesh>, Handle<ClusterMesh>>,
    stripped: &mut HashSet<Entity>,
) {
    let id = mesh3d.0.id();
    let cluster = if let Some(handle) = cache.get(&id) {
        handle.clone()
    } else {
        // Asset not loaded yet — leave Mesh3d in place and retry next frame.
        let Some(mesh) = meshes.get(id) else {
            return;
        };
        match ClusterMesh::try_from(mesh) {
            Ok(cluster_mesh) => {
                let handle = cluster_meshes.add(cluster_mesh);
                cache.insert(id, handle.clone());
                handle
            }
            Err(err) => {
                warn!("cluster bake failed for {entity}: {err}");
                commands.entity(entity).insert(RaytracingBakeFailed);
                return;
            }
        }
    };
    commands
        .entity(entity)
        .insert((RaytracingMesh3d(cluster), NoCpuCulling))
        // Drop the raster mesh AND the visibility components: a pure-RT mesh is
        // never frustum-culled (BVH does it) and Solari extracts it regardless of
        // `ViewVisibility`, so its `Visibility`/`InheritedVisibility` are unused —
        // and keeping them puts the entity in `visibility_propagate` /
        // `check_visibility`'s per-frame `Changed<InheritedVisibility>` scans over
        // millions of entities. Removing them takes the RT meshes out of those
        // scans entirely; UI / camera / intermediate scene nodes keep theirs.
        // (User-set `Hidden` will be honored later via the instance cull mask.)
        .remove::<(Mesh3d, Visibility, InheritedVisibility)>();

    // Strip the mesh-less ancestor scene nodes (gltf transforms, WorldAssetRoots,
    // block parents…) too: once the RT leaf meshes drop their visibility
    // components, those ancestors' `InheritedVisibility` propagation is dead
    // work (nothing under the scene reads it), but it keeps them in the
    // per-frame `visibility_propagate` scan over millions of entities. Walk up
    // via `ChildOf`, stopping at the first already-stripped ancestor — its chain
    // is already done — so total work is O(unique ancestors), not O(meshes×depth).
    for ancestor in parents.iter_ancestors(entity) {
        if !stripped.insert(ancestor) {
            break;
        }
        commands
            .entity(ancestor)
            .remove::<(Visibility, InheritedVisibility)>();
    }
}
