//! Bevy Solari **native floating origin** — galactic-scale ray tracing without f32
//! collapse, done entirely in solari's GPU transform table via **double-single (`df64`)**
//! frame worlds.
//!
//! The camera sits **1 AU from the world origin** (the classic floating-origin stress
//! test). A cloud of glowing markers and a set of near reference cubes are placed at their
//! true `f64` world positions. With the floating origin working, the near cubes render
//! rock-steady and sub-meter crisp while the far markers streak past as you fly; drop the
//! [`SolariFrameWorld`] and the whole scene collapses toward the f32 origin and shimmers.
//! Fly toward the far markers (WASD) — the world stays precise the whole way, with no cells
//! and no quantized recenter.
//!
//! How it works: each body carries a [`SolariFrameWorld`] — its absolute `f64` position,
//! split into two `f32` lanes (`hi + lo`). Solari's GPU propagate subtracts the **`df64`
//! origin** (the camera's own world) from each body's `df64` world, so the huge shared
//! magnitude cancels *before* it reaches the `f32` the acceleration structure is built
//! from — a metre-scale offset survives even at 1 AU, where a naive `f32` subtract would
//! round to zero. The camera folds its per-frame motion into a `df64` accumulator and
//! renders at exactly 0 by construction, so there is no `cell_edge` knob, no `i32` range
//! ceiling, and no recenter. Successor to the integer-cell floating origin (which was
//! itself the GPU-table reimplementation of `big_space`, Aevyrie MIT/Apache — credited).

use bevy::{
    camera_controller::free_camera::{run_freecamera_controller, FreeCamera, FreeCameraPlugin},
    dev_tools::render_debug::RenderDebugOverlayPlugin,
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    math::DVec3,
    pbr::PbrPlugin,
    prelude::*,
    solari::prelude::*,
};

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

const AU_M: f64 = 1.496e11;

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "f1c4a7e2-3d8b-4e6a-9c5f-0b2d7a1e9c84" // Don't copy paste this - generate your own UUID!
    )));

    app.insert_resource(UiTheme(create_dark_theme()))
        // `FeathersPlugins` + the `bevy_solari_debug` feature give the in-engine
        // debug-view dropdown (bottom-left): pathtrace reference, G-buffer views, etc.
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
        // Fold the camera's per-frame motion into its df64 world and point the floating
        // origin at it (after the free-camera controller has moved the camera this frame).
        .add_systems(Update, fly_origin.after(run_freecamera_controller))
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

/// Fold the camera's per-frame motion into its own `SolariFrameWorld`. `FreeCamera` moved
/// the camera by `transform.translation` this frame (world space); add that to the camera's
/// df64 world and re-zero the local transform, so the camera stays at 0. The camera *is* the
/// origin (the propagate reads `frame_world[camera_slot]` as the df64 origin on the GPU), so
/// moving its frame world re-expresses every body against it — no cells, no recenter, no
/// separate origin resource. Only touch it when the camera actually moved, else we'd mark it
/// Changed every frame and force a full re-walk while stationary.
fn fly_origin(mut camera: Query<(&mut Transform, &mut SolariFrameWorld), With<SolariCamera>>) {
    let Ok((mut transform, mut world)) = camera.single_mut() else {
        return;
    };
    if transform.translation != Vec3::ZERO {
        world.world += transform.translation.as_dvec3();
        transform.translation = Vec3::ZERO;
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // The camera lives 1 AU out along +X — the stress test. Everything is expressed
    // relative to the camera's df64 world (its `SolariFrameWorld`), so this is the origin.
    let camera_pos = DVec3::new(AU_M, 0.0, 0.0);

    // Sunlight from the grid origin toward the camera, plus a faint fill so the cubes'
    // shaded sides aren't pure black.
    commands.spawn((
        Transform::from_xyz(0.0, 0.0, 0.0).looking_to(Vec3::NEG_X, Vec3::Y),
        SolariDirectionLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            ..default()
        },
    ));

    // A cloud of glowing markers spread across the grid cells AROUND the camera — the
    // populated world. Each sits in its own cell at km spacing (local Transform 0), so
    // with the floating origin working they form a ~12 km lattice you fly through,
    // recentering as you cross cells. **This is also the diagnostic:** if the GPU cell
    // offset were dead they'd all collapse onto the camera in one clump instead. Meshes
    // are normal-sized (50 m) so they build cleanly — a Sun-sized mesh 1 AU away both
    // breaks the cluster builder and is sub-pixel, which is why the earlier version
    // showed "nothing there".
    let marker = meshes.add(Sphere::new(50.0).mesh().ico(3).unwrap());
    let palette: Vec<Handle<StandardMaterial>> = [
        LinearRgba::rgb(18.0, 5.0, 5.0),
        LinearRgba::rgb(5.0, 18.0, 7.0),
        LinearRgba::rgb(5.0, 9.0, 18.0),
        LinearRgba::rgb(18.0, 15.0, 5.0),
        LinearRgba::rgb(15.0, 5.0, 18.0),
        LinearRgba::rgb(5.0, 18.0, 18.0),
    ]
    .into_iter()
    .map(|emissive| {
        materials.add(StandardMaterial {
            base_color: Color::WHITE,
            emissive,
            unlit: true,
            ..default()
        })
    })
    .collect();
    let spacing = 2000.0; // metres between markers → 2 km lattice
    for i in -3i32..=3 {
        for j in -1i32..=1 {
            for k in -3i32..=3 {
                if i == 0 && j == 0 && k == 0 {
                    continue; // leave the camera's own spot clear
                }
                let pos = camera_pos + DVec3::new(i as f64, j as f64, k as f64) * spacing;
                let mat = palette[(i + j + k).rem_euclid(palette.len() as i32) as usize].clone();
                commands.spawn((
                    SolariFrameWorld::new(pos),
                    Mesh3d(marker.clone()),
                    MeshMaterial3d(mat),
                    Transform::IDENTITY,
                ));
            }
        }
    }

    // Near-field reference cubes — right next to the camera. These are the precision
    // witnesses: 1 AU from the world origin, yet they must be crisp and stationary. A 3×3
    // grid of 2 m cubes a few metres in front of the camera, placed at their true world
    // position so the df64 subtract resolves them sub-mm.
    let cube = meshes.add(Cuboid::new(2.0, 2.0, 2.0));
    for iy in -1..=1 {
        for iz in -1..=1 {
            let mat = materials.add(StandardMaterial {
                base_color: Color::srgb(0.6, 0.7, 0.8),
                perceptual_roughness: 0.5,
                ..default()
            });
            commands.spawn((
                // 12 m in front of the camera (it looks −X), spread across Y/Z.
                SolariFrameWorld::new(
                    camera_pos + DVec3::new(-12.0, iy as f64 * 5.0, iz as f64 * 5.0),
                ),
                Mesh3d(cube.clone()),
                MeshMaterial3d(mat),
                Transform::IDENTITY,
            ));
        }
    }

    // The camera, rendering at the origin (translation 0), looking toward the world origin.
    // Its `SolariFrameWorld` holds its absolute df64 world and *is* the floating origin — the
    // propagate reads `frame_world[camera_slot]` as the origin, so the camera renders at 0 (it
    // subtracts its own world) and [`fly_origin`] folds its motion into that same frame world.
    // No separate origin resource. Solari derives the render view from the GPU transform table.
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Msaa::Off,
        SolariCamera,
        SolariFrameWorld::new(camera_pos),
        FreeCamera {
            walk_speed: 100.0,
            run_speed: 2000.0,
            ..Default::default()
        },
        Transform::default().looking_to(Vec3::NEG_X, Vec3::Y),
    ));
}
