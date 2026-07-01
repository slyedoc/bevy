//! Bevy Solari **reference frames** — a moving [`SolariFrame`] whose children are
//! expressed relative to it and tracked on the GPU transform table as the frame moves.
//!
//! A slowly spinning "station" sits in front of the camera. Its hub and a ring of
//! emissive panels are **children of the frame** — they carry small frame-local
//! `Transform`s and no grid cell, and the GPU ancestor-walk composes them through the
//! frame, so they orbit with it. A row of static reference posts is **not** parented to
//! the frame: they stay rock-steady while the station turns, so you can see the frame's
//! subtree moving against a fixed world.
//!
//! The point this validates: solari's change-driven propagate only re-walks nodes whose
//! *own* local transform changed, so a moving parent would normally leave its children
//! with a stale world. Tagging the parent [`SolariFrame`] opts its whole subtree into a
//! re-walk each frame the frame moves — including static-local children. This is the
//! foundation for co-resident multi-world (each frame → one PTLAS partition): a ship, a
//! station, or a spinning planet whose surface tiles ride the frame.
//!
//! A floating origin composes for free (the station sits 1 AU out): `Transform` is
//! double-precision, the GPU walk carries the frame's big world in native f64, and the
//! subtract pass relativizes it against the camera origin, while its rotation folds into
//! the children's per-instance matrices on the free walk.

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    dev_tools::render_debug::RenderDebugOverlayPlugin,
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    math::{DQuat, DVec3},
    pbr::PbrPlugin,
    prelude::*,
    solari::prelude::*,
};

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

const AU_M: f64 = 1.496e11;

/// Marks the spinning station's frame entity so the `spin_station` system can find it.
#[derive(Component)]
struct Station;

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "f1c4a7e2-3d8b-4e6a-9c5f-0b2d7a1e9c84" // Don't copy paste this - generate your own UUID!
    )));

    app.insert_resource(UiTheme(create_dark_theme()))
        .add_plugins((
            DefaultPlugins
                .build()
                .disable::<PbrPlugin>()
                .disable::<TransformPlugin>()
                .disable::<RenderDebugOverlayPlugin>(),
            SolariPlugin,
            FeathersPlugins,
            FreeCameraPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                spin_station,
                convert_meshes_to_raytracing,
                convert_standard_materials_to_solari,
            )
                .chain(),
        )
        .run();
}

/// Rotate the station frame about its Y axis. Because the station is a [`SolariFrame`],
/// its children re-walk through the new orientation each frame — they orbit with it
/// without carrying any motion of their own.
fn spin_station(time: Res<Time>, mut frames: Query<&mut Transform, With<Station>>) {
    for mut transform in &mut frames {
        transform.rotate_y(0.3 * f64::from(time.delta_secs()));
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // The station sits 1 AU out along +X; the camera 25 m short of it, looking at it. The
    // camera *is* the floating origin — the subtract pass reads its own absolute world off
    // the GPU, so the station's ~1 AU offset stays crisp while its spin rides the
    // per-instance matrices. All of it is plain `Transform`s.
    let station_pos = DVec3::new(AU_M, 0.0, 0.0);
    let camera_pos = station_pos + DVec3::new(-25.0, 0.0, 0.0);

    // Key light.
    commands.spawn((
        Transform::from_xyz(0.0, 0.0, 0.0).looking_to(Vec3::new(-0.3, -1.0, -0.2), Vec3::Y),
        SolariDirectionLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            ..default()
        },
    ));

    // ── The station: a SolariFrame, one cell out along +X, slowly spinning. ──────────
    // Children below are parented to it and compose through its pose on the GPU walk.
    let hub_mesh = meshes.add(Cuboid::new(4.0, 4.0, 4.0));
    let panel_mesh = meshes.add(Cuboid::new(1.0, 6.0, 0.4));
    let hub_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.7, 0.75),
        perceptual_roughness: 0.4,
        ..default()
    });
    let panel_colors = [
        LinearRgba::rgb(12.0, 3.0, 3.0),
        LinearRgba::rgb(3.0, 12.0, 5.0),
        LinearRgba::rgb(3.0, 6.0, 12.0),
        LinearRgba::rgb(12.0, 10.0, 3.0),
        LinearRgba::rgb(10.0, 3.0, 12.0),
        LinearRgba::rgb(3.0, 12.0, 12.0),
    ];

    commands
        .spawn((
            Station,
            SolariFrame,
            // The frame's big world (1 AU out) — an ordinary f64 `Transform`.
            Transform::from_translation(station_pos),
        ))
        .with_children(|frame| {
            // The hub at the frame origin.
            frame.spawn((
                Mesh3d(hub_mesh.clone()),
                MeshMaterial3d(hub_mat.clone()),
                Transform::IDENTITY,
            ));
            // A ring of emissive panels — frame-local offsets, no cell. They orbit with
            // the frame purely through composition.
            let radius = 10.0;
            let count = panel_colors.len();
            for (i, color) in panel_colors.iter().enumerate() {
                let angle = i as f32 / count as f32 * std::f32::consts::TAU;
                let mat = materials.add(StandardMaterial {
                    base_color: Color::WHITE,
                    emissive: *color,
                    unlit: true,
                    ..default()
                });
                frame.spawn((
                    Mesh3d(panel_mesh.clone()),
                    MeshMaterial3d(mat),
                    Transform::from_xyz(
                        f64::from(angle.cos() * radius),
                        0.0,
                        f64::from(angle.sin() * radius),
                    )
                    .with_rotation(DQuat::from_rotation_y(-f64::from(angle))),
                ));
            }
        });

    // ── Static reference posts: NOT parented to the frame. ───────────────────────────
    // They sit in the same cell as the station so they share its neighbourhood, but stay
    // fixed while the station spins — the fixed-world backdrop the frame moves against.
    let post_mesh = meshes.add(Cuboid::new(0.6, 3.0, 0.6));
    let post_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.5, 0.55, 0.6),
        perceptual_roughness: 0.7,
        ..default()
    });
    for i in -2..=2 {
        commands.spawn((
            Mesh3d(post_mesh.clone()),
            MeshMaterial3d(post_mat.clone()),
            Transform::from_translation(station_pos + DVec3::new(0.0, -4.0, f64::from(i) * 6.0)),
        ));
    }

    // Camera: the floating origin itself — an ordinary `Transform` at 1 AU. It renders at
    // 0 by construction (the subtract pass reads its own absolute world as the origin), so
    // the station appears 25 m ahead (+X). The controller flies it directly.
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Msaa::Off,
        SolariCamera,
        FreeCamera {
            walk_speed: 100.0,
            run_speed: 2000.0,
            ..Default::default()
        },
        Transform::from_translation(camera_pos).looking_to(Vec3::X, Vec3::Y),
    ));
}
