//! Bevy Solari black hole — gravitationally lensed rays.
//!
//! A [`SolariBlackHole`] is a field, not geometry: rays crossing its
//! influence sphere march the Schwarzschild photon bend instead of flying
//! straight. The Einstein ring around the horizon, the doubled and smeared
//! images of the pillars behind it, the photon ring, and the accretion disk
//! wrapping over and under the hole (light from the disk's far side bent
//! into view) all EMERGE from the bend — nothing is painted on.
//!
//! Use the pathtrace view and hold still to converge (the disk's glow only
//! shows in primaries there; the realtime view gets the lensing and the
//! black horizon). Orbit the hole — the lensed background streams around
//! the horizon as you move.

use bevy::{
    camera::CameraMainTextureUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig},
    diagnostic::FrameTimeDiagnosticsPlugin,
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    prelude::*,
    render::render_resource::TextureUsages,
    solari::prelude::*,
};
use std::f32::consts::PI;

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "9b3f2c81-6a4e-4d2f-8e7b-1c5a9d0e4f62" // Don't copy paste this - generate your own UUID!
    )));

    app.insert_resource(UiTheme(create_dark_theme()))
        .add_plugins((
            DefaultPlugins,
            SolariPlugin,
            FeathersPlugins,
            FreeCameraPlugin,
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
        .add_systems(
            Update,
            (convert_meshes_to_raytracing, convert_standard_materials_to_solari).chain(),
        )
        .run();
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // The black hole, floating high enough that its influence sphere clears
    // the ground (geometry inside the region is not intersected during the
    // march). Tilted so the accretion disk reads edge-on-ish from the spawn
    // camera — the Interstellar framing, where the disk's far side bends
    // over and under the horizon.
    commands.spawn((
        SolariBlackHole {
            schwarzschild_radius: 0.25,
            influence_radius: 4.0,
            disk_inner_radius: 0.45,
            disk_outer_radius: 1.8,
            disk_emission: 60.0,
        },
        Transform::from_xyz(0.0, 7.0, -14.0)
            .with_rotation(Quat::from_euler(EulerRot::XYZ, 0.12, 0.0, 0.35)),
    ));

    // A backdrop to lens: a colonnade of colored pillars BEHIND the hole —
    // their doubled, smeared images wrap the horizon as you move.
    let pillar = meshes.add(Cuboid::new(0.8, 9.0, 0.8));
    for i in 0..9 {
        let x = (i as f32 - 4.0) * 3.0;
        let hue = i as f32 * 40.0;
        commands.spawn((
            Mesh3d(pillar.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::hsl(hue, 0.7, 0.55),
                perceptual_roughness: 0.8,
                ..default()
            })),
            Transform::from_xyz(x, 4.5, -26.0),
        ));
    }

    // The glass dragon below the hole — close enough to be lensed when the
    // camera lines it up behind the influence sphere.
    commands.spawn((
        WorldAssetRoot(
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/DragonAttenuation.glb")),
        ),
        Transform::from_xyz(3.0, 0.7306 * 0.8, -22.0)
            .with_scale(Vec3::splat(0.8))
            .with_rotation(Quat::from_rotation_y(PI / 3.0)),
    ));

    // Ground.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(80.0, 80.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.32),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -16.0),
    ));

    commands.spawn((
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, -0.7, -0.6, 0.0)),
        SolariDirectionLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            ..default()
        },
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
        Transform::from_xyz(0.0, 6.0, 0.0).looking_at(Vec3::new(0.0, 7.0, -14.0), Vec3::Y),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariCamera,
        SolariAtmosphere::default(),
    ));
}
