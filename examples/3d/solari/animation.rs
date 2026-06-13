//! Bevy Solari demo — skeletal cluster animation.
//!
//! Loads the skinned glTF fox, plays its run animation, and renders it with the
//! full ray-traced Solari path. The skinned mesh is baked into a [`ClusterMesh`]
//! (carrying per-vertex joint streams + per-cluster bloat AABBs), then deformed
//! on the GPU each frame so the ray-traced geometry follows the skeleton — see
//! `crates/bevy_solari/cluster_animation_plan.md`.
//!
//! The skeleton is driven by the same GPU transform table the rest of the scene
//! uses (each joint is a node); the deform pass derives each joint's world from
//! the table's `local`/`parent` columns plus the mesh's inverse-bind poses, so no
//! CPU skin-matrix upload is needed.

use bevy::{
    camera::CameraMainTextureUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    input::common_conditions::input_just_pressed,
    prelude::*,
    render::render_resource::TextureUsages,
    solari::prelude::*,
    world_serialization::WorldInstanceReady,
    post_process::bloom::Bloom,
    camera::{Exposure, Hdr},
};

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

/// A skinned glTF model with a "run" animation (clip index 2).
const GLTF_PATH: &str = "models/animated/Fox.glb";

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "5417916c-0291-4e3f-8f65-326c1858ab96" // Don't copy paste this - generate your own UUID!
    )));

    app.add_plugins((
        DefaultPlugins
            .set(bevy::log::LogPlugin {
                filter: "wgpu=error,naga=warn,bevy_solari=info".into(),
                ..default()
            })   
            .disable::<TransformPlugin>()
            .disable::<bevy::pbr::PbrPlugin>()
            .disable::<bevy::dev_tools::render_debug::RenderDebugOverlayPlugin>(),
        SolariPlugin,
        FeathersPlugins,
        FreeCameraPlugin,
    ))
        .insert_resource(UiTheme(create_dark_theme()))
        // `PbrPlugin` is disabled, so re-register `Assets<StandardMaterial>` — the
        // glTF loader + `convert_standard_materials_to_solari` still need it.
        .init_asset::<StandardMaterial>()
        .add_systems(Startup, (setup_scene, setup_camera_and_light))
        // Space toggles the run animation (pause / resume).
        .add_systems(
            Update,
            toggle_animation.run_if(input_just_pressed(KeyCode::Space)),
        )
        // Bake glTF meshes into ClusterMeshes + swap to `RaytracingMesh3d`, and
        // convert their `StandardMaterial`s to `SolariMaterial`. Skinned meshes keep
        // their `SkinnedMesh` component through the swap, so the deform path picks up
        // the joint palette + inverse-bind poses.
        .add_systems(
            Update,
            (convert_meshes_to_raytracing, convert_standard_materials_to_solari).chain(),
        )
        .run();
}

/// Stores the animation we want to play, resolved once the scene spawns.
#[derive(Component)]
struct AnimationToPlay {
    graph_handle: Handle<AnimationGraph>,
    index: AnimationNodeIndex,
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground plane — spawned as a regular `Mesh3d` + `StandardMaterial`; the
    // `convert_*` systems bake it to a `ClusterMesh` / `SolariMaterial` like the
    // fox. Gives the ray tracer a surface for the fox's shadow + bounce light.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(400.0, 400.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.5, 0.3),
            ..default()
        })),
    ));
    // Clip index 2 is the fox "run" animation.
    let (graph, index) =
        AnimationGraph::from_clip(asset_server.load(GltfAssetLabel::Animation(2).from_asset(GLTF_PATH)));
    let graph_handle = graphs.add(graph);

    commands
        .spawn((
            AnimationToPlay {
                graph_handle,
                index,
            },
            WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset(GLTF_PATH))),
            Transform::from_scale(Vec3::splat(0.5)),
        ))
        .observe(play_animation_when_ready);
}

fn play_animation_when_ready(
    scene_ready: On<WorldInstanceReady>,
    mut commands: Commands,
    children: Query<&Children>,
    animations_to_play: Query<&AnimationToPlay>,
    mut players: Query<&mut AnimationPlayer>,
) {
    let Ok(animation_to_play) = animations_to_play.get(scene_ready.entity) else {
        return;
    };
    for child in children.iter_descendants(scene_ready.entity) {
        if let Ok(mut player) = players.get_mut(child) {
            player.play(animation_to_play.index).repeat();
            commands
                .entity(child)
                .insert(AnimationGraphHandle(animation_to_play.graph_handle.clone()));
        }
    }
}

/// Pause / resume every animation player on Space.
fn toggle_animation(mut players: Query<&mut AnimationPlayer>) {
    for mut player in &mut players {
        if player.all_paused() {
            player.resume_all();
        } else {
            player.pause_all();
        }
    }
}

fn setup_camera_and_light(mut commands: Commands) {
    // Under Solari (and with `PbrPlugin` disabled) the sun is a
    // `SolariDirectionLight`, not bevy_pbr's `DirectionalLight`. Its direction is
    // resolved from `GlobalTransform` (the GPU transform table), so we aim it with
    // `looking_at`.
    commands.spawn((
        SolariDirectionLight {
            illuminance: light_consts::lux::RAW_SUNLIGHT,
            ..default()
        },
        Transform::from_xyz(1.0, 1.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(0.02, 0.02, 0.05)),
            ..default()
        },
        FreeCamera {
            walk_speed: 80.0,
            run_speed: 200.0,
            ..default()
        },
        Transform::from_translation(Vec3::new(80.0, 70.0, 110.0))
            .looking_at(Vec3::new(0.0, 25.0, 0.0), Vec3::Y),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        Exposure::OVERCAST,
        Bloom::NATURAL,
        Hdr,
        SolariCamera,
        SolariAtmosphere::default(),
        SolariGlobalFog {
            visibility: 1200.0,
            fog_height: 5.5,
            fog_base: 0.0,
            ..default()
        },
    ));
}
