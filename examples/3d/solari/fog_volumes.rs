//! Bevy Solari fog volumes — local participating media.
//!
//! A [`SolariFogVolume`] is a medium, not geometry: a unit box or sphere in
//! entity-local space (sized by the `Transform`) that the aerial-perspective
//! march samples on top of the global height fog. Every step casts a real
//! shadow ray toward the sun, so the windowed wall slices the mist into
//! god-ray blades, the campfire's smoke puffs self-shadow against the sun,
//! and shaded fog stays dark instead of glowing.
//!
//! The staging is the classic one: a closed hall with a band of windows at
//! the far end and a low sun behind them — the only sunlight inside is what
//! comes through the openings, so each window projects a visible blade
//! through the mist.

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig},
    diagnostic::FrameTimeDiagnosticsPlugin,
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    math::DQuat,
    prelude::*,
    solari::prelude::*,
};

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
            (
                animate_smoke_puffs,
                (
                    convert_meshes_to_raytracing,
                    convert_standard_materials_to_solari,
                )
                    .chain(),
            ),
        )
        .run();
}

/// Where the campfire sits, outside the hall's left wall.
const FIRE_POS: Vec3 = Vec3::new(-26.0, 0.0, -20.0);

/// One looping smoke puff above the fire; `phase` staggers the puffs so the
/// column stays populated.
#[derive(Component)]
struct SmokePuff {
    phase: f32,
}

/// Rise, drift, swell, and thin each puff over a looping cycle — fog volumes
/// are change-driven, so moving the `Transform` and editing the component is
/// all it takes.
fn animate_smoke_puffs(
    time: Res<Time>,
    mut puffs: Query<(&SmokePuff, &mut Transform, &mut SolariFogVolume)>,
) {
    const CYCLE: f32 = 6.0;
    for (puff, mut transform, mut volume) in &mut puffs {
        let h = (time.elapsed_secs() / CYCLE + puff.phase).fract();
        transform.translation = (FIRE_POS
            + Vec3::new(
                h * 1.5 + (h * 9.0).sin() * 0.3, // wind drift + a little wobble
                1.5 + h * 8.0,
                (h * 7.0).cos() * 0.3,
            ))
        .to_precision();
        transform.scale = Vec3::splat(0.7 + h * 2.0).to_precision();
        // Fade in quickly at birth, thin out as the puff disperses.
        volume.density = 1.2 * (h * 6.0).min(1.0) * (1.0 - h);
    }
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Dusty air filling the whole hall, floor to roof: a thin box of white
    // medium — subtle enough to leave the air readable, dense enough to
    // catch the window blades. The hall shadows it from the low sun, so the
    // only light inside is what comes through the windows.
    commands.spawn((
        SolariFogVolume {
            density: 0.0018,
            albedo: Vec3::splat(0.9),
            phase_g: 0.45,
            softness: 0.2,
            spherical: false,
        },
        Transform::from_xyz(0.0, 6.0, -28.0).with_scale(Vec3::new(18.0, 6.0, 28.0).to_precision()),
    ));

    // A campfire outside the hall, off the left wall: a stone pit and an
    // emissive flame, with a column of smoke puffs rising out of it (spawned
    // below, animated by `animate_smoke_puffs`). The puffs stand in direct
    // sun, so they self-shadow and silhouette instead of reading as a blob.
    // Note the fire does NOT light the smoke — fog volumes are lit by the
    // sun + sky only.
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(1.3, 0.5))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.25, 0.23, 0.22),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::from_xyz(f64::from(FIRE_POS.x), 0.25, f64::from(FIRE_POS.z)),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cone {
            radius: 0.7,
            height: 1.8,
        })),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.45, 0.1),
            emissive: LinearRgba::rgb(60000.0, 18000.0, 3000.0),
            ..default()
        })),
        Transform::from_xyz(f64::from(FIRE_POS.x), 1.4, f64::from(FIRE_POS.z)),
    ));
    for i in 0..4 {
        commands.spawn((
            SmokePuff {
                phase: i as f32 * 0.25,
            },
            SolariFogVolume {
                density: 1.2,
                albedo: Vec3::splat(0.3),
                phase_g: 0.6,
                softness: 0.8,
                spherical: true,
            },
            Transform::from_xyz(f64::from(FIRE_POS.x), 2.0, f64::from(FIRE_POS.z))
                .with_scale(Vec3::splat(0.7).to_precision()),
        ));
    }

    // The hall: a 36 × 12 × 56 m box, closed except for a band of windows in
    // the far wall the sun comes through. Everything the camera sees is in
    // the hall's own shadow — the dark canvas the blades draw on.
    let wall_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.45, 0.42, 0.38),
        perceptual_roughness: 0.9,
        ..default()
    });

    // Far wall: a slab below the window band (y 2..8), a slab above it, and
    // piers splitting the band into window openings.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(36.0, 2.0, 0.6))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(0.0, 1.0, -56.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(36.0, 4.0, 0.6))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(0.0, 10.0, -56.0),
    ));
    // Piers: 1.6 m every 5.2 m leave 3.6 m window openings.
    let pier = meshes.add(Cuboid::new(1.6, 6.0, 0.6));
    for k in 0..7 {
        commands.spawn((
            Mesh3d(pier.clone()),
            MeshMaterial3d(wall_material.clone()),
            Transform::from_xyz(f64::from(k) * 5.2 - 15.6, 5.0, -56.0),
        ));
    }

    // Side walls, back wall (behind the camera), and roof close the box.
    // The left wall has a doorway (3 m wide, 4 m tall, at the fire's z) cut
    // into it: a section either side of the opening plus a lintel above.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.6, 12.0, 56.0))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(18.3, 6.0, -28.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.6, 12.0, 18.5))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(-18.3, 6.0, -9.25),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.6, 12.0, 34.5))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(-18.3, 6.0, -38.75),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.6, 8.0, 3.0))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(-18.3, 8.0, -20.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(37.8, 12.0, 0.6))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(0.0, 6.0, 0.3),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(37.8, 0.6, 57.6))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(0.0, 12.3, -28.2),
    ));

    // Floor.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(80.0, 80.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.32),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -28.0),
    ));

    // The sun, low BEHIND the far wall (to-sun is the light's +Z = `back`):
    // ~14° above the horizon, slightly off-axis so the blades rake across
    // the mist. Through a window at y ≈ 5 the light descends ~0.25 per metre
    // and lands on the floor ~20 m in — right inside the mist pool.
    commands.spawn((
        Transform::from_rotation(DQuat::from_euler(
            EulerRot::YXZ,
            std::f64::consts::PI - 0.15,
            -0.25,
            0.0,
        )),
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
            walk_speed: 5.0,
            run_speed: 15.0,
            ..Default::default()
        },
        Transform::from_xyz(0.0, 3.5, -6.0)
            .looking_at(Vec3::new(0.0, 4.0, -56.0).to_precision(), Vec3::Y),
        Msaa::Off,
        SolariCamera::default(),
        // The atmosphere drives the sun's color/attenuation and the sky; there's
        // no `SolariGlobalFog`, so the global height fog is off — all the fog in
        // this scene is the local volumes.
        SolariAtmosphere::default(),
    ));
}
