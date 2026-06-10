//! Bevy Solari refraction demo — a wine bottle and wine glass from NVIDIA's
//! Bistro scene on a small stage.
//!
//! Exercises transmissive materials end to end: exact-Fresnel reflect/refract,
//! total internal reflection, and Beer–Lambert volume absorption (the bottle is
//! dark green glass; the wine is red by absorption, not surface color).
//!
//! The asset is produced by
//! `examples/large_scenes/bistro/extract_refraction_assets.py`, which carves
//! the glassware out of the Bistro FBX and applies the glass-material spec
//! from NVIDIA's own `.pyscene` (IOR 1.55 glass / 1.33 wine, measured
//! absorption coefficients).
//!
//! Switch to the pathtrace debug view (bottom-left dropdown) for the reference
//! image; the realtime ReSTIR path does not refract yet.

use bevy::{
    camera::CameraMainTextureUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig},
    diagnostic::FrameTimeDiagnosticsPlugin,
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    prelude::*,
    render::render_resource::TextureUsages,
    solari::prelude::*,
    world_serialization::WorldInstanceReady,
};
use std::f32::consts::PI;

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "d4a9adeb-71d9-4d2b-8c1a-9f7e3c5b0a26" // Don't copy paste this - generate your own UUID!
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
        // The glTF materials load as `StandardMaterial` (PbrPlugin is enabled);
        // convert meshes + materials for the RT scene.
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
    // The glassware (bottles + glass + wine), floor at y = 0.
    commands.spawn(WorldAssetRoot(
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/refraction.glb")),
    ));

    // Khronos `DragonAttenuation` — the reference asset for
    // KHR_materials_transmission/volume (glass dragon with measured
    // attenuation). Scaled into the tabletop scene; the cloth backdrop the
    // asset ships with is stripped on spawn (we have our own stage).
    commands
        .spawn((
            WorldAssetRoot(
                asset_server
                    .load(GltfAssetLabel::Scene(0).from_asset("models/DragonAttenuation.glb")),
            ),
            // The asset's origin is not at the dragon's feet: the Dragon node
            // sits at y = -0.7306 (standing on the stripped backdrop cloth).
            Transform::from_xyz(-0.25, 0.7306 * 0.12, -0.05)
                .with_scale(Vec3::splat(0.12))
                .with_rotation(Quat::from_rotation_y(1.0)),
        ))
        .observe(
            |scene_ready: On<WorldInstanceReady>,
             children: Query<&Children>,
             names: Query<&Name>,
             mut commands: Commands| {
                for entity in children.iter_descendants(scene_ready.entity) {
                    if let Ok(name) = names.get(entity)
                        && name.contains("Backdrop")
                    {
                        commands.entity(entity).despawn();
                    }
                }
            },
        );

    // A small stage: checkered floor so refraction visibly bends straight
    // lines, and a back wall to catch caustic-ish light patterns.
    let dark = materials.add(StandardMaterial {
        base_color: Color::srgb(0.25, 0.25, 0.28),
        perceptual_roughness: 0.9,
        ..default()
    });
    let light = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.83, 0.78),
        perceptual_roughness: 0.9,
        ..default()
    });
    let tile = meshes.add(Plane3d::default().mesh().size(0.1, 0.1));
    for x in -8..8 {
        for z in -8..8 {
            let material = if (x + z) % 2 == 0 { &dark } else { &light };
            commands.spawn((
                Mesh3d(tile.clone()),
                MeshMaterial3d(material.clone()),
                Transform::from_xyz(x as f32 * 0.1 + 0.05, 0.0, z as f32 * 0.1 + 0.05),
            ));
        }
    }
    for x in -8..8 {
        for y in 0..8 {
            let material = if (x + y) % 2 == 0 { &dark } else { &light };
            commands.spawn((
                Mesh3d(tile.clone()),
                MeshMaterial3d(material.clone()),
                Transform::from_xyz(x as f32 * 0.1 + 0.05, y as f32 * 0.1 + 0.05, -0.8)
                    .with_rotation(Quat::from_rotation_x(PI / 2.0)),
            ));
        }
    }

    commands.spawn((
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, -0.8, -1.0, 0.0)),
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
            walk_speed: 0.5,
            run_speed: 2.0,
            ..Default::default()
        },
        Transform::from_xyz(0.0, 0.25, 0.55).looking_at(Vec3::new(0.0, 0.15, 0.0), Vec3::Y),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariCamera,
        SolariAtmosphere::default(),
    ));
}
