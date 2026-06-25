//! Bevy Solari stained-glass cathedral — colored sunlight through transmissive
//! glass.
//!
//! A low sun stands behind a row of jewel-toned lancet windows. The only light
//! inside the nave is what comes through the glass, and it arrives COLORED:
//!
//! * On the floor, each window paints a pool of its own color, because the
//!   surface's shadow ray toward the sun crosses the glass and picks up its
//!   Beer–Lambert tint instead of being hard-shadowed (`trace_light_transmittance`).
//! * In the air, the same tinted sun visibility colors the fog god-rays, so
//!   each window throws a shaft of its own hue through the mist.
//!
//! The tint is real volume absorption: each pane is `attenuation_distance`
//! thick and authored with that window's `attenuation_color`, so white sunlight
//! leaves the glass as exactly that color — the same absorption the path tracer
//! applies to a camera ray looking straight through the window.
//!
//! Switch to the pathtrace debug view (bottom-left dropdown) for the reference
//! image; the realtime ReSTIR path does not refract or tint shadows yet.

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

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "f1c4a7e2-3d8b-4e6a-9c5f-0b2d7a1e9c84" // Don't copy paste this - generate your own UUID!
    )));

    app.insert_resource(UiTheme(create_dark_theme()))
        .add_plugins((
            DefaultPlugins,
            SolariPlugin,
            // GPU picking backend: cursor rays traced against the PTLAS.
            SolariPickingPlugin,
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
        .add_observer(log_click)
        .run();
}

/// Picking test: log the entity (and world hit position) the cursor clicked,
/// resolved through the Solari ray-query picking backend. `Pointer<Click>`
/// bubbles up the hierarchy, so gate on the original target to log just the
/// directly-clicked entity once (not the parents it propagates to).
fn log_click(click: On<Pointer<Click>>) {
    if click.entity == click.original_event_target() {
        info!(
            "solari pick: clicked {} at {:?} (depth {:.2})",
            click.entity, click.hit.position, click.hit.depth,
        );
    }
}

/// One jewel tone per lancet, left to right. The color is the light that
/// survives the pane: white sun in, this color out.
const WINDOW_COLORS: [Srgba; 6] = [
    Srgba::new(0.85, 0.12, 0.18, 1.0), // deep rose
    Srgba::new(0.95, 0.55, 0.08, 1.0), // amber
    Srgba::new(0.10, 0.62, 0.30, 1.0), // emerald
    Srgba::new(0.10, 0.30, 0.85, 1.0), // sapphire
    Srgba::new(0.55, 0.15, 0.75, 1.0), // violet
    Srgba::new(0.10, 0.62, 0.66, 1.0), // teal
];

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut solari_materials: ResMut<Assets<SolariMaterial>>,
) {
    // Dusty air filling the nave, floor to vault: a thin box of white medium,
    // dense enough to catch the window shafts but light enough to read through.
    // The cathedral shadows it from the low sun, so the only light inside is
    // the colored sunlight coming through the glass.
    commands.spawn((
        SolariFogVolume {
            density: 0.0035,
            albedo: Vec3::splat(0.9),
            phase_g: 0.5,
            softness: 0.2,
            spherical: false,
        },
        Transform::from_xyz(0.0, 6.0, -28.0).with_scale(Vec3::new(18.0, 6.0, 28.0)),
    ));

    // Pale limestone walls and piers — light enough to show the colored pools.
    let stone = materials.add(StandardMaterial {
        base_color: Color::srgb(0.62, 0.58, 0.52),
        perceptual_roughness: 0.95,
        ..default()
    });

    // Far wall (z = -56): a slab below the window band (y 2..8), a slab above,
    // and piers splitting the band into six lancet openings.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(36.0, 2.0, 0.6))),
        MeshMaterial3d(stone.clone()),
        Transform::from_xyz(0.0, 1.0, -56.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(36.0, 4.0, 0.6))),
        MeshMaterial3d(stone.clone()),
        Transform::from_xyz(0.0, 10.0, -56.0),
    ));
    // Piers: 1.6 m wide every 5.2 m leave 3.6 m openings between them.
    let pier = meshes.add(Cuboid::new(1.6, 6.0, 0.6));
    for k in 0..7 {
        commands.spawn((
            Mesh3d(pier.clone()),
            MeshMaterial3d(stone.clone()),
            Transform::from_xyz(k as f32 * 5.2 - 15.6, 5.0, -56.0),
        ));
    }

    // The stained glass: one colored pane filling each opening, centered
    // between consecutive piers. Each is `THICKNESS` deep and given an
    // `attenuation_distance` equal to that thickness, so white sunlight crossing
    // it comes out as exactly `attenuation_color` — the pane's jewel tone.
    const THICKNESS: f32 = 0.3;
    let pane = meshes.add(Cuboid::new(3.4, 5.8, THICKNESS));
    for (i, color) in WINDOW_COLORS.iter().enumerate() {
        let center_x = (i as f32 - 2.5) * 5.2; // midpoints: -13, -7.8, ... 13
        commands.spawn((
            Mesh3d(pane.clone()),
            SolariMaterial3d(solari_materials.add(SolariMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.0,
                specular_transmission: 1.0,
                ior: 1.5,
                attenuation_color: (*color).into(),
                attenuation_distance: THICKNESS,
                ..default()
            })),
            Transform::from_xyz(center_x, 5.0, -56.0),
        ));
    }

    // Side walls, back wall (behind the camera), and vault close the box so the
    // interior sees only window light. Left wall has a doorway cut into it.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.6, 12.0, 56.0))),
        MeshMaterial3d(stone.clone()),
        Transform::from_xyz(18.3, 6.0, -28.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.6, 12.0, 18.5))),
        MeshMaterial3d(stone.clone()),
        Transform::from_xyz(-18.3, 6.0, -9.25),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.6, 12.0, 34.5))),
        MeshMaterial3d(stone.clone()),
        Transform::from_xyz(-18.3, 6.0, -38.75),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.6, 8.0, 3.0))),
        MeshMaterial3d(stone.clone()),
        Transform::from_xyz(-18.3, 8.0, -20.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(37.8, 12.0, 0.6))),
        MeshMaterial3d(stone.clone()),
        Transform::from_xyz(0.0, 6.0, 0.3),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(37.8, 0.6, 57.6))),
        MeshMaterial3d(stone.clone()),
        Transform::from_xyz(0.0, 12.3, -28.2),
    ));

    // Flagstone floor — pale, to catch the colored pools cleanly.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(80.0, 80.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.5, 0.48, 0.45),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -28.0),
    ));

    // The sun, low BEHIND the far wall (to-sun is the light's +Z = `back`):
    // ~14° above the horizon, slightly off-axis so the shafts rake across the
    // nave. Through a window at y ≈ 5 the light descends ~0.25 per metre and
    // lands on the floor ~20 m in — right inside the mist.
    commands.spawn((
        Transform::from_rotation(Quat::from_euler(
            EulerRot::YXZ,
            std::f32::consts::PI - 0.15,
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
        Transform::from_xyz(0.0, 3.5, -6.0).looking_at(Vec3::new(0.0, 4.0, -56.0), Vec3::Y),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariCamera,
        // The atmosphere drives the sun's color/attenuation and the sky; there's
        // no `SolariGlobalFog`, so all the fog in this scene is the local volume.
        SolariAtmosphere::default(),
    ));
}
