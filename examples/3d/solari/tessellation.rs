//! Bevy Solari GPU tessellation — real displaced geometry, not a shading trick.
//!
//! A row of cubes shares one displacement-mapped material (the parallax
//! example's height map — same input, very different mechanism). Where
//! `parallax_mapping` warps UVs to fake depth, solari subdivides each base
//! triangle on the GPU, displaces the micro-vertices along the surface normal,
//! and ray-traces the result: silhouettes bulge, displaced faces self-shadow,
//! and reflections/GI see the real shape.
//!
//! Tessellation is view-adaptive and crack-free: per-edge factors derive from
//! projected screen-space edge length, so the near cube gets dense micro-
//! triangles while the far cubes stay coarse — fly along the row (WASD) and
//! watch the detail follow you.
//!
//! Any `StandardSolariMaterial` with a `depth_map` opts in (brighter texel =
//! deeper, bevy's `StandardMaterial::depth_map` convention). Displacement
//! height comes from [`SolariSettings::tess_displacement_scale`].

use bevy::{
    camera::Hdr, camera_controller::free_camera::{FreeCamera, FreeCameraPlugin}, dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig}, diagnostic::FrameTimeDiagnosticsPlugin, feathers::{FeathersPlugins, dark_theme::create_dark_theme, theme::UiTheme}, math::DQuat, pbr::PbrPlugin, prelude::*, solari::prelude::*,
};

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "b6f3c9a1-7e2d-4b8f-a5c0-9d1e4f7a2b6c" // Don't copy paste this - generate your own UUID!
    )));

    app.insert_resource(SolariSettings {
        // Displacement height in object units (default 0.05). The cubes are 2 m,
        // so 0.08 carves visibly deep grooves.
        tess_displacement_scale: 0.08,
        ..default()
    })
    .add_plugins((
        DefaultPlugins
                .build()
                // Solari propagates transforms on the GPU — disable bevy's CPU
                // `TransformPlugin` so this scene exercises that path exclusively.
                .disable::<PbrPlugin>()
                .disable::<TransformPlugin>()
                .disable::<bevy::dev_tools::render_debug::RenderDebugOverlayPlugin>(),
        FeathersPlugins,
        SolariPlugin,
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
    .insert_resource(SolariSettings {
        tess_displacement_scale: 0.08,
        tess_px_per_segment: 1.0,   // 3× denser than default; 1.0 ≈ 6×
        ..default()
    })
    .insert_resource(UiTheme(create_dark_theme()))
    .add_systems(Startup, setup_scene)
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

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut solari_materials: ResMut<Assets<StandardSolariMaterial>>,
) {
    // The displacement-mapped material: `depth_map` is the tessellation opt-in.
    // Authored as `StandardSolariMaterial` directly — the `StandardMaterial`
    // converter deliberately drops `depth_map` (parallax content shouldn't
    // silently become tessellated geometry).
    let displaced = solari_materials.add(StandardSolariMaterial {
        base_color_texture: Some(asset_server.load("textures/parallax_example/cube_color.png")),
        depth_map: Some(asset_server.load("textures/parallax_example/cube_depth.png")),
        perceptual_roughness: 0.7,
        ..default()
    });

    // // A row of cubes receding from the camera — the adaptive-LOD demo. Near
    // // cubes tessellate finely, far ones coarsely; the split moves as you fly.
    let cube = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    for (i, z) in [1.0, 3.0, 5.0, 7.0, 9.0].into_iter().enumerate() {
        commands.spawn((
            Mesh3d(cube.clone()),
            SolariMaterial3d(displaced.clone()),
            Transform::from_xyz(0.0, 0.5, z).with_rotation(DQuat::from_rotation_y(0.35 * i as f64)),
        ));
    }

    // Ground plane: plain converted `StandardMaterial`, no depth map — it stays
    // a flat two-triangle raytraced quad and catches the cubes' shadows.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(120.0, 120.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.45, 0.44, 0.42),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -20.0),
    ));

    // Low raking sun: grazing light exaggerates the displaced relief.
    commands.spawn((
        Transform::from_rotation(DQuat::from_euler(EulerRot::YXZ, 2.3, -0.35, 0.0)),
        SolariDirectionLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            ..default()
        },
    ));

    commands.spawn((
        Camera3d::default(),
        //CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),   
        Transform::from_xyz(3.0, 2.5, 2.0)
            .looking_at(Vec3::new(0.0, 1.0, -12.0).to_precision(), Vec3::Y),
        Hdr,
        Msaa::Off,
        SolariCamera::default(),        
        SolariSky::Procedural,
        FreeCamera {
            walk_speed: 5.0,
            run_speed: 15.0,
            ..Default::default()
        },
    ));
}
