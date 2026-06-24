//! Bevy Solari **native floating origin** — galactic-scale ray tracing without f32
//! collapse, done entirely in solari's GPU transform table.
//!
//! The camera sits **1 AU from the grid origin** (the classic floating-origin stress
//! test). A handful of small reference cubes sit in the camera's own grid cell, and a
//! glowing "sun" marks the grid origin one AU away. With the floating origin ON, the
//! near cubes render rock-steady and sub-meter crisp while the sun is a distant glint;
//! turn it off (don't set [`SolariGridCell`]) and the whole scene collapses toward the
//! f32 origin and shimmers. Fly toward the sun (WASD) — the world recenters cell by cell
//! and stays precise the whole way.
//!
//! How it works: each body carries an integer [`SolariGridCell`] plus a small local
//! `Transform`; solari's GPU propagate pass adds `(cell − origin) × cell_edge` to the
//! node's world (computed with the integer subtraction FIRST, so a 1-AU cell index never
//! costs near-camera precision). The acceleration structure is therefore built in
//! camera-cell-relative space — no CPU per-object work, the change-only propagate keeps
//! static geometry free at steady state. This is the GPU-table reimplementation of
//! `big_space`'s floating origin (Aevyrie, MIT/Apache — used as the reference algorithm
//! and credited).

use bevy::{
    camera::CameraMainTextureUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    math::{DVec3, IVec3},
    prelude::*,
    render::render_resource::TextureUsages,
    solari::prelude::*,
    // The floating-origin types live in solari's transform module (not yet in the
    // prelude); import them explicitly.
    solari::transform::{SolariFloatingOrigin, SolariGridCell},
};

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

/// Metres per grid cell. 1 km cells put 1 AU at cell index ~1.5e8 — comfortably inside
/// `i32` (≈4 ly is the i32 ceiling at this edge; wider needs the i64 cell column).
const CELL_EDGE: f32 = 1000.0;
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
        .add_plugins((DefaultPlugins, SolariPlugin, FeathersPlugins, FreeCameraPlugin))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                (convert_meshes_to_raytracing, convert_standard_materials_to_solari).chain(),
                recenter_floating_origin,
            ),
        )
        .run();
}

/// Split an absolute metre position into `(cell, local)` so `cell × edge + local == pos`,
/// with `local` kept within half a cell (the floating-origin invariant). The same helper
/// `big_space::Grid::translation_to_grid` provides.
fn to_grid(pos: DVec3) -> (IVec3, Vec3) {
    let edge = CELL_EDGE as f64;
    let cell = (pos / edge).round();
    let local = pos - cell * edge;
    (
        IVec3::new(cell.x as i32, cell.y as i32, cell.z as i32),
        local.as_vec3(),
    )
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut origin: ResMut<SolariFloatingOrigin>,
) {
    // The camera lives 1 AU out along +X — the stress test. Everything is expressed
    // relative to its cell, so this is the floating origin.
    let camera_pos = DVec3::new(AU_M, 0.0, 0.0);
    let (camera_cell, camera_local) = to_grid(camera_pos);
    *origin = SolariFloatingOrigin {
        origin_cell: [camera_cell.x, camera_cell.y, camera_cell.z],
        cell_edge: CELL_EDGE,
    };

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
    let spacing = 2; // cells between markers → 2 km
    for i in -3..=3 {
        for j in -1..=1 {
            for k in -3..=3 {
                if i == 0 && j == 0 && k == 0 {
                    continue; // leave the camera's own cell clear
                }
                let cell = IVec3::new(
                    camera_cell.x + i * spacing,
                    camera_cell.y + j * spacing,
                    camera_cell.z + k * spacing,
                );
                let mat = palette[(i + j + k).rem_euclid(palette.len() as i32) as usize].clone();
                commands.spawn((
                    SolariGridCell::new(cell.x, cell.y, cell.z),
                    Mesh3d(marker.clone()),
                    MeshMaterial3d(mat),
                    Transform::IDENTITY,
                ));
            }
        }
    }

    // Near-field reference cubes — in the CAMERA's cell, small local offsets. These are
    // the precision witnesses: 1 AU from the grid origin, yet they must be crisp and
    // stationary. A 3×3 grid of 2 m cubes a few metres in front of the camera.
    let cube = meshes.add(Cuboid::new(2.0, 2.0, 2.0));
    for iy in -1..=1 {
        for iz in -1..=1 {
            let mat = materials.add(StandardMaterial {
                base_color: Color::srgb(0.6, 0.7, 0.8),
                perceptual_roughness: 0.5,
                ..default()
            });
            commands.spawn((
                SolariGridCell::new(camera_cell.x, camera_cell.y, camera_cell.z),
                Mesh3d(cube.clone()),
                MeshMaterial3d(mat),
                // 12 m in front of the camera (it looks −X), spread across Y/Z. Local
                // offsets are small (metres), well within a cell.
                Transform::from_translation(
                    camera_local + Vec3::new(-12.0, iy as f32 * 5.0, iz as f32 * 5.0),
                ),
            ));
        }
    }

    // The camera, parked at its cell's local position, looking toward the grid origin.
    // It carries NO `SolariGridCell`: the camera *is* the floating origin, and its cell
    // lives in `SolariFloatingOrigin`, not on the entity. A (static) cell here would put
    // the camera in solari's transform table with GPU→CPU readback, which after a recenter
    // overwrites its `GlobalTransform` with a stale-cell propagated value that fights the
    // wrapped `Transform` from bevy's `TransformPlugin` → the warp. `NoGpuGlobalTransformReadback`
    // keeps solari's hands off the camera's transform entirely (bevy + the recenter own it).
    commands.spawn((
        NoGpuGlobalTransformReadback,
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariCamera,
        // Fast enough to fly across cells (1 km each) and watch the world recenter.
        FreeCamera {
            walk_speed: 100.0,
            run_speed: 2000.0,
            ..Default::default()
        },
        Transform::from_translation(camera_local).looking_to(Vec3::NEG_X, Vec3::Y),
    ));
}

/// Keep the floating origin glued to the camera so its rendered coordinates stay small —
/// large camera coords are what make the ray tracer jitter and (because solari is a 1-spp
/// path tracer that denoises temporally) make the image pixelate as you fly off. When the
/// camera's local Transform drifts past half a cell, shift `origin_cell` by the whole
/// cells crossed and wrap the Transform back toward the cell centre: the world position is
/// unchanged, everything is just re-expressed relative to the new cell.
///
/// The origin jump moves every instance a full cell in one frame, so we pulse
/// [`CameraReset`] — a one-frame temporal-history reset (DLSS + ReSTIR) so the reprojection
/// doesn't smear/teleport across the discontinuity. (The O(partitions), zero-reset recenter
/// is the partition-translation layer; this is the foundation's O(cells)-re-propagate + a
/// 1-frame reset, which is imperceptible at a cell crossing.)
fn recenter_floating_origin(
    mut origin: ResMut<SolariFloatingOrigin>,
    mut camera: Query<(&mut Transform, &mut CameraReset), With<SolariCamera>>,
) {
    let Ok((mut transform, mut reset)) = camera.single_mut() else {
        return;
    };
    let edge = origin.cell_edge;
    if edge <= 0.0 {
        return;
    }
    // Whole cells the camera has drifted from its cell centre (round → nearest cell, so
    // the local stays within ±½ cell; handles multi-cell jumps from a fast camera too).
    let drift = (transform.translation / edge).round();
    if drift == Vec3::ZERO {
        return;
    }
    origin.origin_cell[0] += drift.x as i32;
    origin.origin_cell[1] += drift.y as i32;
    origin.origin_cell[2] += drift.z as i32;
    transform.translation -= drift * edge;
    reset.0 = true;
}
