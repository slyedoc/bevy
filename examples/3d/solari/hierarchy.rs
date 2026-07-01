//! Bevy Solari **transform hierarchy + floating origin** — the classic parent/child/
//! grandchild nesting test (`examples/ecs/hierarchy.rs`), in 3D, run entirely on solari's
//! **GPU transform table** with bevy's CPU `TransformPlugin` disabled *and* placed **1 AU
//! from the grid origin** so the nested walk composes with the floating-origin cell offset.
//!
//! A grey **root** cube spins in place. Parented to it are two emissive **arm** cubes that
//! orbit the root and spin about their own axis; parented to each arm is a smaller **hand**
//! cube that likewise spins; and parented to each hand is a tiny **finger** cube with a
//! purely static local `Transform` — it has no motion of its own, yet it traces a deeply
//! compound path because all three ancestors above it rotate. That static-local leaf is the
//! point of the test: its world is produced only by the GPU ancestor-walk composing
//! root → arm → hand → finger, so if it swings correctly the nested propagation is working.
//! A row of static posts + a floor (not parented) are the fixed-world backdrop.
//!
//! **The floating-origin twist:** the whole scene sits ~1 AU (1.5×10⁸ m) from the grid
//! origin — far past where f32 world coordinates hold sub-metre precision. The big offset
//! rides an integer [`SolariGridCell`] on the **root frame** (subtracted against the
//! camera's origin cell *before* the f32 fold), while the frame's spin and the children's
//! frame-local offsets stay small and precise. With it working, the cubes render rock-steady
//! and crisp at 1 AU; the nested motion is identical to the near-origin version. Turn the
//! cells off (drop `SolariGridCell` / the origin) and the whole scene collapses toward the
//! f32 origin and shimmers.
//!
//! How it works: `TransformPlugin` is disabled, so there is no CPU `GlobalTransform` — solari's
//! change-driven propagate composes worlds on the GPU. That pass re-walks only nodes whose own
//! local changed, so a moving parent would leave a static-local child stale; tagging the
//! spinning **root** [`SolariFrame`] opts its whole subtree (arms, hands, *and* the static
//! fingers) into a re-walk each frame it moves. The floating origin is `big_space`'s algorithm (Aevyrie,
//! MIT/Apache — credited) reimplemented in the GPU table. It just loops — nothing is despawned.

use bevy::{
    camera::CameraMainTextureUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    dev_tools::render_debug::RenderDebugOverlayPlugin,
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    math::{DVec3, IVec3},
    pbr::PbrPlugin,
    prelude::*,
    render::render_resource::TextureUsages,
    solari::prelude::*,
    solari::transform::{SolariFloatingOrigin, SolariGridCell},
};

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

/// Metres per grid cell — matches the floating-origin/frames examples (1 km cells put 1 AU
/// at cell index ~1.5e8, comfortably inside `i32`).
const CELL_EDGE: f32 = 1000.0;
const AU_M: f64 = 1.496e11;

/// The spinning parent at the top of the hierarchy.
#[derive(Component)]
struct Root;

/// An arm parented to the root — orbits with the root and spins on its own axis.
#[derive(Component)]
struct Arm;

/// A hand parented to an arm — orbits the arm and spins on its own axis, so its child
/// finger orbits it in turn.
#[derive(Component)]
struct Hand;

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
                // Solari propagates transforms on the GPU — disable bevy's CPU
                // `TransformPlugin` so this scene exercises that path exclusively.
                .disable::<PbrPlugin>()
                .disable::<TransformPlugin>()
                .disable::<RenderDebugOverlayPlugin>(),
            SolariPlugin,
            FeathersPlugins,
            FreeCameraPlugin,
        ))
        .add_systems(Startup, setup)
        // The camera-follow recenter is built into `SolariTransformPlugin`; `setup` only
        // seeds the initial origin cell + `cell_edge`.
        .add_systems(
            Update,
            (
                spin_root,
                spin_arms,
                spin_hands,
                convert_meshes_to_raytracing,
                convert_standard_materials_to_solari,
            )
                .chain(),
        )
        .run();
}

/// Split an absolute metre position into `(cell, local)` so `cell × edge + local == pos`,
/// with `local` kept within half a cell (the floating-origin invariant). Mirrors
/// `big_space::Grid::translation_to_grid`.
fn to_grid(pos: DVec3) -> (IVec3, Vec3) {
    let edge = CELL_EDGE as f64;
    let cell = (pos / edge).round();
    let local = pos - cell * edge;
    (
        IVec3::new(cell.x as i32, cell.y as i32, cell.z as i32),
        local.as_vec3(),
    )
}

/// Rotate the root. It's a [`SolariFrame`], so its whole subtree — the arms and their
/// static-local hands — re-walks through the new pose each frame.
fn spin_root(time: Res<Time>, mut roots: Query<&mut Transform, With<Root>>) {
    for mut transform in &mut roots {
        transform.rotate_y(0.5 * time.delta_secs());
    }
}

/// Rotate each arm about its own axis. Their locals change, so they re-walk normally.
fn spin_arms(time: Res<Time>, mut arms: Query<&mut Transform, With<Arm>>) {
    for mut transform in &mut arms {
        transform.rotate_y(1.5 * time.delta_secs());
    }
}

/// Rotate each hand about its own axis — this is what makes the static-local finger orbit
/// its parent. The finger itself never moves; it rides all three spins via the root frame's
/// subtree re-walk composing root → arm → hand → finger.
fn spin_hands(time: Res<Time>, mut hands: Query<&mut Transform, With<Hand>>) {
    for mut transform in &mut hands {
        transform.rotate_y(3.0 * time.delta_secs());
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut origin: ResMut<SolariFloatingOrigin>,
) {
    // Everything lives ~1 AU out along +X. The camera *is* the floating origin, so its cell
    // seeds `SolariFloatingOrigin`; the scene sits in that same cell neighbourhood and its
    // big offset is removed by the integer cell subtraction before the f32 fold.
    let scene_pos = DVec3::new(AU_M, 0.0, 0.0);
    let (cell, local) = to_grid(scene_pos);
    *origin = SolariFloatingOrigin {
        origin_cell: [cell.x, cell.y, cell.z],
        cell_edge: CELL_EDGE,
    };

    // Key light (directional — no position, so unaffected by the origin).
    commands.spawn((
        Transform::from_xyz(0.0, 0.0, 0.0).looking_to(Vec3::new(-0.3, -1.0, -0.2), Vec3::Y),
        SolariDirectionLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            ..default()
        },
    ));

    let root_mesh = meshes.add(Cuboid::new(2.0, 2.0, 2.0));
    let arm_mesh = meshes.add(Cuboid::new(1.2, 1.2, 1.2));
    let hand_mesh = meshes.add(Cuboid::new(0.6, 0.6, 0.6));
    let finger_mesh = meshes.add(Cuboid::new(0.3, 0.3, 0.3));

    let root_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.7, 0.75),
        perceptual_roughness: 0.4,
        ..default()
    });
    let arm_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        emissive: LinearRgba::rgb(3.0, 8.0, 12.0),
        unlit: true,
        ..default()
    });
    let hand_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        emissive: LinearRgba::rgb(12.0, 5.0, 3.0),
        unlit: true,
        ..default()
    });
    let finger_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        emissive: LinearRgba::rgb(4.0, 12.0, 4.0),
        unlit: true,
        ..default()
    });

    // ── The hierarchy: root → arm → hand. The ROOT carries the big offset — its integer
    // `SolariGridCell` (1 AU out) *and* the `SolariFrame` tag that re-walks the whole subtree
    // each spin. Children stay frame-local with NO cell (one celled node per ChildOf chain —
    // a second would subtract the origin twice). Local `Transform`s are cell-relative, so add
    // `local` (the cell-local scene position) to place them within the cell. ──
    commands
        .spawn((
            Root,
            SolariFrame,
            SolariGridCell::new(cell.x, cell.y, cell.z),
            Mesh3d(root_mesh),
            MeshMaterial3d(root_mat),
            Transform::from_translation(local + Vec3::new(0.0, 1.0, 0.0)),
        ))
        .with_children(|root| {
            // Two arms, on opposite sides of the root. Frame-local offsets, no cell.
            for side in [1.0_f32, -1.0] {
                root.spawn((
                    Arm,
                    Mesh3d(arm_mesh.clone()),
                    MeshMaterial3d(arm_mat.clone()),
                    Transform::from_xyz(4.0 * side, 0.0, 0.0),
                ))
                .with_children(|arm| {
                    // A hand that spins on its own axis, so its finger orbits it.
                    arm.spawn((
                        Hand,
                        Mesh3d(hand_mesh.clone()),
                        MeshMaterial3d(hand_mat.clone()),
                        Transform::from_xyz(2.0 * side, 0.0, 0.0),
                    ))
                    .with_children(|hand| {
                        // The deepest nesting witness: a finger with a purely static local
                        // Transform. It never moves relative to its hand, yet it orbits a
                        // compound path — its world is produced entirely by the GPU
                        // ancestor-walk composing root → arm → hand → finger.
                        hand.spawn((
                            Mesh3d(finger_mesh.clone()),
                            MeshMaterial3d(finger_mat.clone()),
                            Transform::from_xyz(1.0 * side, 0.0, 0.0),
                        ));
                    });
                });
            }
        });

    // ── Static reference posts: NOT parented to the root — the fixed-world backdrop the
    // hierarchy turns against. Each is its own ChildOf chain, so each carries the cell. ──
    let post_mesh = meshes.add(Cuboid::new(0.6, 3.0, 0.6));
    let post_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.5, 0.55, 0.6),
        perceptual_roughness: 0.7,
        ..default()
    });
    for i in -2..=2 {
        commands.spawn((
            SolariGridCell::new(cell.x, cell.y, cell.z),
            Mesh3d(post_mesh.clone()),
            MeshMaterial3d(post_mat.clone()),
            Transform::from_translation(local + Vec3::new(i as f32 * 3.0, -1.5, -8.0)),
        ));
    }

    // A large floor so the shadows and orbits read clearly.
    let floor_mesh = meshes.add(Cuboid::new(40.0, 0.2, 40.0));
    let floor_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 0.3, 0.32),
        perceptual_roughness: 0.9,
        ..default()
    });
    commands.spawn((
        SolariGridCell::new(cell.x, cell.y, cell.z),
        Mesh3d(floor_mesh),
        MeshMaterial3d(floor_mat),
        Transform::from_translation(local + Vec3::new(0.0, -3.0, 0.0)),
    ));

    // Camera: the floating origin itself, so it carries NO `SolariGridCell` (its cell lives
    // in `SolariFloatingOrigin`) and `NoGpuGlobalTransformReadback` keeps solari's hands off
    // its transform — bevy + the built-in recenter own it. It sits in the scene's cell,
    // looking at the root.
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
        FreeCamera {
            walk_speed: 20.0,
            run_speed: 2000.0,
            ..default()
        },
        Transform::from_translation(local + Vec3::new(0.0, 4.0, 16.0))
            .looking_at(local + Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));
}
