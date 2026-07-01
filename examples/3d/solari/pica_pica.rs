//! Bevy Solari demo — pica_pica diorama with a patrolling robot.
//!
//! Full-RT ReSTIR lighting on a small glTF scene. When built with the
//! `bevy_solari_debug` feature, [`SolariPlugin`] adds the debug overlay and a
//! per-pass GPU-timing panel ([`RenderDiagnosticsPlugin`]).
//!
//! Debug-view keys (also pickable from the bottom-left dropdown):
//! - `Tab` — next view (pathtrace reference, LOD, cluster, triangle, G-buffer).
//! - `Shift+Tab` — previous view.

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig},
    diagnostic::{Diagnostic, DiagnosticPath, DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    gltf::GltfMaterialName,
    math::DQuat,
    prelude::*,
    render::diagnostic::RenderDiagnosticsPlugin,
    solari::prelude::*,
    world_serialization::WorldInstanceReady,
};
use std::f32::consts::PI;

// The full-RT path supplies its own DLSS Ray Reconstruction internally (see
// `bevy_solari::render::dlss`); the SDK only needs the project id inserted
// before `RenderPlugin`. No `Dlss` component is added to the camera —
// `DlssRayReconstructionSupported` is read only to report status in the UI.
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::{DlssProjectId, DlssRayReconstructionSupported};

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "5417916c-0291-4e3f-8f65-326c1858ab96" // Don't copy paste this - generate your own UUID!
    )));

    app.insert_resource(UiTheme(create_dark_theme()));

    app.add_plugins((
        DefaultPlugins.set(bevy::log::LogPlugin {
            filter: "wgpu=error,naga=warn,bevy_solari=info".into(),
            ..default()
        }),
        SolariPlugin,
        FeathersPlugins,
        FreeCameraPlugin,
        RenderDiagnosticsPlugin,
        FrameTimeDiagnosticsPlugin::default(),
        FpsOverlayPlugin {
            config: FpsOverlayConfig {
                frame_time_graph_config: FrameTimeGraphConfig {
                    enabled: true,
                    target_fps: 240.0,
                    min_fps: 60.0,
                },
                ..default()
            },
        },
    ))
    .add_systems(Startup, setup_scene)
    .add_systems(PostUpdate, update_performance_text)
    .add_systems(Update, (pause_scene, toggle_lights, patrol_path))
    .add_systems(PostUpdate, update_control_text)
    // convert meshes to raytracing meshes after scene load, so the glTF meshes can be used as-is without extra processing for raytracing support (in particular, we generate missing UVs and tangents, which the pathtracer needs for its debug visualizations, but the forward renderer doesn't care about).
    // The glTF materials load as `StandardMaterial` (PbrPlugin is enabled), so also
    // convert them to `SolariMaterial` — otherwise every RT instance gets the
    // default material and the scene renders untextured / unlit.
    .add_systems(
        Update,
        (
            convert_meshes_to_raytracing,
            convert_standard_materials_to_solari,
        )
            .chain(),
    )
    .run();
}

fn setup_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands
        .spawn((
            WorldAssetRoot(
                asset_server.load(
                    GltfAssetLabel::Scene(0)
                        .from_asset("https://github.com/bevyengine/bevy_asset_files/raw/2a5950295a8b6d9d051d59c0df69e87abcda58c3/pica_pica/mini_diorama_01.glb")
                ),
            ),
            Transform::from_scale(Vec3::splat(10.0).to_precision()),
        ))
        .observe(fix_materials);

    commands
        .spawn((
            WorldAssetRoot(asset_server.load(
                GltfAssetLabel::Scene(0).from_asset("https://github.com/bevyengine/bevy_asset_files/raw/2a5950295a8b6d9d051d59c0df69e87abcda58c3/pica_pica/robot_01.glb")
            )),
            Transform::from_scale(Vec3::splat(2.0).to_precision())
                .with_translation(Vec3::new(-2.0, 0.05, -2.1).to_precision())
                .with_rotation(Quat::from_rotation_y(PI / 2.0).to_precision()),
            PatrolPath {
                path: vec![
                    (Vec3::new(-2.0, 0.05, -2.1), Quat::from_rotation_y(PI / 2.0)),
                    (Vec3::new(2.2, 0.05, -2.1), Quat::from_rotation_y(0.0)),
                    (
                        Vec3::new(2.2, 0.05, 2.1),
                        Quat::from_rotation_y(3.0 * PI / 2.0),
                    ),
                    (Vec3::new(-2.0, 0.05, 2.1), Quat::from_rotation_y(PI)),
                ],
                i: 0,
            },
        ))
        .observe(fix_materials);

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(DQuat::from_xyzw(
            -0.13334629,
            -0.86597735,
            -0.3586996,
            0.3219264,
        )),
    ));

    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        FreeCamera {
            walk_speed: 3.0,
            run_speed: 10.0,
            ..Default::default()
        },
        Transform::from_translation(Vec3::new(0.219417, 2.5764852, 6.9718704).to_precision())
            .with_rotation(
                Quat::from_xyzw(-0.1466768, 0.013738206, 0.002037309, 0.989087).to_precision(),
            ),
        Msaa::Off,
        SolariCamera,
    ));

    commands.spawn((
        ControlText,
        Text::default(),
        Node {
            position_type: PositionType::Absolute,
            bottom: px(12.0),
            left: px(12.0),
            ..default()
        },
    ));

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            right: px(0.0),
            padding: px(4.0).all(),
            border_radius: BorderRadius::bottom_left(px(4.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.10, 0.10, 0.10, 0.8)),
        children![(
            PerformanceText,
            Text::default(),
            TextFont {
                font_size: FontSize::Px(8.0),
                ..default()
            },
        )],
    ));
}

fn fix_materials(
    scene_ready: On<WorldInstanceReady>,
    children: Query<&Children>,
    mesh_query: Query<(&MeshMaterial3d<StandardMaterial>, Option<&GltfMaterialName>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    for descendant in children.iter_descendants(scene_ready.entity) {
        if let Ok((MeshMaterial3d(material_handle), material_name)) = mesh_query.get(descendant) {
            if material_name.map(|s| s.0.as_str()) == Some("material") {
                let mut material = materials.get_mut(material_handle).unwrap();
                material.emissive = LinearRgba::BLACK;
            }
            if material_name.map(|s| s.0.as_str()) == Some("Lights") {
                let mut material = materials.get_mut(material_handle).unwrap();
                material.emissive =
                    LinearRgba::from(Color::srgb(0.941, 0.714, 0.043)) * 1_000_000.0;
                material.alpha_mode = AlphaMode::Opaque;
                material.specular_transmission = 0.0;

                commands.insert_resource(RobotLightMaterial(material_handle.clone()));
            }
            if material_name.map(|s| s.0.as_str()) == Some("Glass_Dark_01") {
                let mut material = materials.get_mut(material_handle).unwrap();
                material.alpha_mode = AlphaMode::Opaque;
                material.specular_transmission = 0.0;
            }
        }
    }
}

fn pause_scene(mut time: ResMut<Time<Virtual>>, key_input: Res<ButtonInput<KeyCode>>) {
    if key_input.just_pressed(KeyCode::Space) {
        time.toggle();
    }
}

#[derive(Resource)]
struct RobotLightMaterial(Handle<StandardMaterial>);

fn toggle_lights(
    key_input: Res<ButtonInput<KeyCode>>,
    robot_light_material: Option<Res<RobotLightMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    directional_light: Query<Entity, With<DirectionalLight>>,
    mut commands: Commands,
) {
    if key_input.just_pressed(KeyCode::Digit1) {
        if let Ok(directional_light) = directional_light.single() {
            commands.entity(directional_light).despawn();
        } else {
            commands.spawn((
                DirectionalLight {
                    illuminance: light_consts::lux::FULL_DAYLIGHT,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::from_rotation(DQuat::from_xyzw(
                    -0.13334629,
                    -0.86597735,
                    -0.3586996,
                    0.3219264,
                )),
            ));
        }
    }

    if key_input.just_pressed(KeyCode::Digit2)
        && let Some(robot_light_material) = robot_light_material
    {
        let mut material = materials.get_mut(&robot_light_material.0).unwrap();
        if material.emissive == LinearRgba::BLACK {
            material.emissive = LinearRgba::from(Color::srgb(0.941, 0.714, 0.043)) * 1_000_000.0;
        } else {
            material.emissive = LinearRgba::BLACK;
        }
    }
}

#[derive(Component)]
struct PatrolPath {
    path: Vec<(Vec3, Quat)>,
    i: usize,
}

fn patrol_path(mut query: Query<(&mut PatrolPath, &mut Transform)>, time: Res<Time<Virtual>>) {
    for (mut path, mut transform) in query.iter_mut() {
        let (mut target_position, mut target_rotation) = path.path[path.i];
        let mut target_position = target_position.to_precision();
        let mut distance_to_target = transform.translation.distance(target_position);
        if distance_to_target < 0.01 {
            transform.translation = target_position;
            transform.rotation = target_rotation.to_precision();

            path.i = (path.i + 1) % path.path.len();
            (target_position, target_rotation) = {
                let (p, r) = path.path[path.i];
                (p.to_precision(), r)
            };
            distance_to_target = transform.translation.distance(target_position);
        }

        let direction = (target_position - transform.translation).normalize();
        let movement = direction * f64::from(time.delta_secs());

        if movement.length() > distance_to_target {
            transform.translation = target_position;
            transform.rotation = target_rotation.to_precision();
        } else {
            transform.translation += movement;
        }
    }
}

#[derive(Component)]
struct ControlText;

fn update_control_text(
    mut text: Single<&mut Text, With<ControlText>>,
    robot_light_material: Option<Res<RobotLightMaterial>>,
    materials: Res<Assets<StandardMaterial>>,
    directional_light: Query<Entity, With<DirectionalLight>>,
    time: Res<Time<Virtual>>,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] dlss_rr_supported: Option<
        Res<DlssRayReconstructionSupported>,
    >,
) {
    text.0.clear();

    if time.is_paused() {
        text.0.push_str("(Space): Resume");
    } else {
        text.0.push_str("(Space): Pause");
    }

    if directional_light.single().is_ok() {
        text.0.push_str("\n(1): Disable directional light");
    } else {
        text.0.push_str("\n(1): Enable directional light");
    }

    match robot_light_material.and_then(|m| materials.get(&m.0)) {
        Some(robot_light_material) if robot_light_material.emissive != LinearRgba::BLACK => {
            text.0.push_str("\n(2): Disable robot emissive light");
        }
        _ => {
            text.0.push_str("\n(2): Enable robot emissive light");
        }
    }

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    if dlss_rr_supported.is_some() {
        text.0
            .push_str("\nDenoising: DLSS Ray Reconstruction enabled");
    } else {
        text.0
            .push_str("\nDenoising: DLSS Ray Reconstruction not supported");
    }

    #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))]
    text.0
        .push_str("\nDenoising: App not compiled with DLSS support");
}

#[derive(Component)]
struct PerformanceText;

fn update_performance_text(
    mut text: Single<&mut Text, With<PerformanceText>>,
    diagnostics: Res<DiagnosticsStore>,
) {
    text.0.clear();

    let mut total = 0.0;
    let mut add_diagnostic = |name: &str, path: &'static str| {
        let path = DiagnosticPath::new(path);
        if let Some(value) = diagnostics.get(&path).and_then(Diagnostic::smoothed) {
            text.push_str(&format!("{name:17}  {value:.2} ms\n"));
            total += value;
        }
    };

    // Per-pass GPU timings from the full-RT ReSTIR path (see
    // `bevy_solari::render::node::restir`). Requires `RenderDiagnosticsPlugin`.
    (add_diagnostic)("Visibility", "render/restir/visibility/elapsed_gpu");
    (add_diagnostic)("Presample", "render/restir/presample/elapsed_gpu");
    (add_diagnostic)(
        "Initial+temporal",
        "render/restir/initial_and_temporal/elapsed_gpu",
    );
    (add_diagnostic)(
        "Spatial+shade",
        "render/restir/spatial_and_shade/elapsed_gpu",
    );
    (add_diagnostic)("Specular GI", "render/restir/specular_gi/elapsed_gpu");
    (add_diagnostic)("Compose", "render/restir/compose/elapsed_gpu");
    text.push_str(&format!("{:17}  {total:.2} ms\n", "Total"));
}
