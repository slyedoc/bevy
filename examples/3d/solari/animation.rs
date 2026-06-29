//! Bevy Solari demo — skeletal cluster animation.
//!
//! Loads the skinned glTF fox and renders it with the full ray-traced Solari path.
//! The skinned mesh is baked into a [`ClusterMesh`] (per-vertex joint streams +
//! per-cluster bloat AABBs), then deformed on the GPU each frame so the ray-traced
//! geometry follows the skeleton — see `crates/bevy_solari/cluster_animation_plan.md`.
//!
//! A small feathers panel controls playback: play/pause, a speed slider, and a clip
//! selector (the fox's three glTF animations: survey / walk / run).

use bevy::{
    camera::{CameraMainTextureUsages, Exposure, Hdr},
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    feathers::{
        controls::*,
        dark_theme::create_dark_theme,
        theme::{ThemeBackgroundColor, ThemedText, UiTheme},
        tokens, FeathersPlugins,
    },
    input::common_conditions::input_just_pressed,
    post_process::bloom::Bloom,
    prelude::*,
    render::render_resource::TextureUsages,
    solari::prelude::*,
    ui::Checked,
    ui_widgets::{radio_self_update, slider_self_update, Activate, RadioGroup, SliderStep, SliderValue},
    world_serialization::WorldInstanceReady,
};

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

const GLTF_PATH: &str = "models/animated/Fox.glb";
/// Default clip on spawn (the "run" animation).
const DEFAULT_CLIP: usize = 2;

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
        .add_systems(Startup, (setup_scene, setup_camera_and_light, ui.spawn()))
        .add_systems(
            Update,
            (
                toggle_animation.run_if(input_just_pressed(KeyCode::Space)),
                // Drive the fox from the panel widgets' state.
                sync_speed,
                sync_clip,
                // Bake glTF meshes into ClusterMeshes + swap to `RaytracingMesh3d`,
                // and convert their `StandardMaterial`s — the skinned mesh keeps its
                // `SkinnedMesh` through the swap, so the deform path picks it up.
                (convert_meshes_to_raytracing, convert_standard_materials_to_solari).chain(),
            ),
        )
        .run();
}

/// Playback state shared between the panel widgets and the `AnimationPlayer`s.
#[derive(Resource)]
struct FoxControl {
    clips: Vec<AnimationNodeIndex>,
    current: usize,
    speed: f32,
    playing: bool,
}

/// Marks the speed slider so `sync_speed` can read its value.
#[derive(Component, Clone, Copy, Default)]
struct SpeedSlider;
/// The clip a radio selects.
#[derive(Component, Clone, Copy, Default)]
struct ClipChoice(usize);

/// Resolved once the glTF scene spawns: which graph + clip to start on.
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
    // Ground plane — baked to a `ClusterMesh` like the fox; gives the ray tracer a
    // surface for the fox's shadow + bounce light.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(400.0, 400.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.5, 0.3),
            ..default()
        })),
    ));

    // All three clips in one graph so the panel can switch between them.
    let (graph, clips) = AnimationGraph::from_clips([
        asset_server.load(GltfAssetLabel::Animation(0).from_asset(GLTF_PATH)),
        asset_server.load(GltfAssetLabel::Animation(1).from_asset(GLTF_PATH)),
        asset_server.load(GltfAssetLabel::Animation(2).from_asset(GLTF_PATH)),
    ]);
    let graph_handle = graphs.add(graph);

    commands.insert_resource(FoxControl {
        clips: clips.clone(),
        current: DEFAULT_CLIP,
        speed: 1.0,
        playing: true,
    });

    commands
        .spawn((
            AnimationToPlay {
                graph_handle,
                index: clips[DEFAULT_CLIP],
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

/// Space toggles play/pause (mirrors the panel button).
fn toggle_animation(mut players: Query<&mut AnimationPlayer>, ctl: Option<ResMut<FoxControl>>) {
    let Some(mut ctl) = ctl else { return };
    ctl.playing = !ctl.playing;
    for mut player in &mut players {
        if ctl.playing {
            player.resume_all();
        } else {
            player.pause_all();
        }
    }
}

/// Push the speed slider's value onto every playing animation.
fn sync_speed(
    slider: Query<&SliderValue, (With<SpeedSlider>, Changed<SliderValue>)>,
    mut players: Query<&mut AnimationPlayer>,
    ctl: Option<ResMut<FoxControl>>,
) {
    let (Some(value), Some(mut ctl)) = (slider.iter().next(), ctl) else {
        return;
    };
    ctl.speed = value.0;
    for mut player in &mut players {
        for (_, active) in player.playing_animations_mut() {
            active.set_speed(value.0);
        }
    }
}

/// When the selected clip radio changes, restart that clip on every player.
fn sync_clip(
    selected: Query<&ClipChoice, (With<Checked>, Changed<Checked>)>,
    mut players: Query<&mut AnimationPlayer>,
    ctl: Option<ResMut<FoxControl>>,
) {
    let (Some(choice), Some(mut ctl)) = (selected.iter().next(), ctl) else {
        return;
    };
    if choice.0 == ctl.current {
        return;
    }
    ctl.current = choice.0;
    let (index, speed, playing) = (ctl.clips[choice.0], ctl.speed, ctl.playing);
    for mut player in &mut players {
        player.stop_all();
        player.play(index).repeat().set_speed(speed);
        if !playing {
            player.pause_all();
        }
    }
}

/// The feathers control panel (top-left overlay).
fn ui() -> impl SceneList {
    bsn_list![(
        Node {
            position_type: PositionType::Absolute,
            left: px(12),
            top: px(12),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(8),
            padding: UiRect::all(px(12)),
            min_width: px(190),
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Children [
            (Text("Fox") ThemedText),
            (
                @FeathersButton { @caption: bsn! { Text("Play / Pause") ThemedText } }
                on(|_a: On<Activate>, mut players: Query<&mut AnimationPlayer>, ctl: Option<ResMut<FoxControl>>| {
                    let Some(mut ctl) = ctl else { return };
                    ctl.playing = !ctl.playing;
                    for mut player in &mut players {
                        if ctl.playing { player.resume_all(); } else { player.pause_all(); }
                    }
                })
            ),
            (Text("Speed") ThemedText),
            (
                @FeathersSlider { @max: 3.0, @value: 1.0 }
                SliderStep(0.1)
                SpeedSlider
                on(slider_self_update)
            ),
            (Text("Clip") ThemedText),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    row_gap: px(4),
                }
                RadioGroup
                on(radio_self_update)
                Children [
                    (@FeathersRadio { @caption: bsn! { Text("Survey") ThemedText } } ClipChoice(0)),
                    (@FeathersRadio { @caption: bsn! { Text("Walk") ThemedText } } ClipChoice(1)),
                    (@FeathersRadio { @caption: bsn! { Text("Run") ThemedText } } ClipChoice(2) Checked),
                ]
            ),
        ]
    )]
}

fn setup_camera_and_light(mut commands: Commands) {
    // Under Solari (PbrPlugin disabled) the sun is a `SolariDirectionLight`; its
    // direction comes from `GlobalTransform`, so aim it with `looking_at`.
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
