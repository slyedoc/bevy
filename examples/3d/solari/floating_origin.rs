//! Bevy Solari **native floating origin** — galactic-scale ray tracing without f32
//! collapse, with **nothing to author**: `Transform` is double-precision (the
//! `transform_f64` feature, which `bevy_solari` enables) and the GPU transform table
//! does the rest.
//!
//! The camera sits **1 AU from the world origin** (the classic floating-origin stress
//! test). A cloud of glowing markers and a set of near reference cubes are placed at their
//! true `f64` world positions — as plain `Transform`s. With the floating origin working,
//! the near cubes render rock-steady and sub-meter crisp while the far markers streak past
//! as you fly. Fly toward the far markers (WASD) — the world stays precise the whole way,
//! with no cells, no quantized recenter, and no special components.
//!
//! How it works: the GPU propagate walk composes every node's **absolute** world with a
//! native-`f64` translation (`SHADER_F64`); a flat subtract pass then subtracts the
//! **camera's own absolute world** (the origin, read straight off the GPU as
//! `world_abs_t[camera_slot]`), so the huge shared magnitude cancels *before* it reaches
//! the `f32` the acceleration structure is built from — a metre-scale offset survives even
//! at 1 AU, where a naive `f32` subtract would round to zero. The camera renders at
//! exactly 0 by construction (it subtracts its own world), so there is no `cell_edge`
//! knob, no range ceiling, and no recenter. Successor to the integer-cell floating origin
//! (which was itself the GPU-table reimplementation of `big_space`, Aevyrie MIT/Apache —
//! credited).

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
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

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // The camera lives 1 AU out along +X — the stress test. Every body below is spawned
    // at its true absolute f64 world position, as an ordinary `Transform`.
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

    // A cloud of glowing markers spread AROUND the camera — the populated world, a ~12 km
    // lattice you fly through. **This is also the diagnostic:** if the f64 path were dead
    // they'd all collapse toward the camera in one clump instead. Meshes are normal-sized
    // (50 m) so they build cleanly — a Sun-sized mesh 1 AU away both breaks the cluster
    // builder and is sub-pixel, which is why the earlier version showed "nothing there".
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
                let pos =
                    camera_pos + DVec3::new(f64::from(i), f64::from(j), f64::from(k)) * spacing;
                let mat = palette[(i + j + k).rem_euclid(palette.len() as i32) as usize].clone();
                commands.spawn((
                    Mesh3d(marker.clone()),
                    MeshMaterial3d(mat),
                    Transform::from_translation(pos),
                ));
            }
        }
    }

    // Near-field reference cubes — right next to the camera. These are the precision
    // witnesses: 1 AU from the world origin, yet they must be crisp and stationary. A 3×3
    // grid of 2 m cubes a few metres in front of the camera, placed at their true world
    // position so the f64 subtract resolves them sub-mm.
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
                Mesh3d(cube.clone()),
                MeshMaterial3d(mat),
                Transform::from_translation(
                    camera_pos + DVec3::new(-12.0, f64::from(iy) * 5.0, f64::from(iz) * 5.0),
                ),
            ));
        }
    }

    // The camera — an ordinary `Transform` at 1 AU, moved directly by the free-camera
    // controller. The camera *is* the floating origin (the subtract pass reads its own
    // absolute world off the GPU as the origin), so it renders at exactly 0 no matter
    // where it flies — no fold-into-accumulator system, no origin resource, nothing.
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Msaa::Off,
        SolariCamera::default(),
        FreeCamera {
            walk_speed: 100.0,
            run_speed: 2000.0,
            ..Default::default()
        },
        Transform::from_translation(camera_pos).looking_to(Vec3::NEG_X, Vec3::Y),
    ));
}
