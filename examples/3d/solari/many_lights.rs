//! Bevy Solari stress test — 441 emissive sphere lights over a sea of
//! 8000 random cubes.
//!
//! Designed to surface scaling problems in the full-RT ReSTIR path tracer.
//! When built with the `bevy_solari_debug` feature, [`SolariPlugin`] adds the
//! debug overlay (press Tab / Shift+Tab to cycle views: pathtrace reference,
//! LOD, cluster, triangle, G-buffer). A per-pass GPU-timing panel is shown in
//! the corner.

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig},
    diagnostic::{Diagnostic, DiagnosticPath, DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    image::{ImageAddressMode, ImageLoaderSettings},
    mesh::VertexAttributeValues,
    pbr::PbrPlugin,
    post_process::bloom::Bloom,
    prelude::*,
    render::diagnostic::RenderDiagnosticsPlugin,
    solari::prelude::*,
};
use chacha20::ChaCha8Rng;
use rand::{RngExt, SeedableRng};

// The full-RT path supplies its own DLSS Ray Reconstruction internally (see
// `bevy_solari::render::dlss`); the SDK only needs the project id inserted
// before `RenderPlugin`. No `Dlss` component is added to the camera.
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "5417916c-0291-4e3f-8f65-326c1858ab96" // Don't copy paste this - generate your own UUID!
    )));

    app.add_plugins((
        DefaultPlugins
            .set(bevy::log::LogPlugin {
                filter: "wgpu=error,naga=warn,bevy_solari=info".into(),
                ..default()
            })
            .disable::<TransformPlugin>()
            .disable::<PbrPlugin>(),
        SolariPlugin,
        FeathersPlugins,
        FreeCameraPlugin,
        RenderDiagnosticsPlugin,
        FrameTimeDiagnosticsPlugin::default(),
        FpsOverlayPlugin {
            config: FpsOverlayConfig {
                frame_time_graph_config: FrameTimeGraphConfig {
                    enabled: true,
                    target_fps: 144.0,
                    min_fps: 30.0,
                },
                ..default()
            },
        },
    ))
    .insert_resource(UiTheme(create_dark_theme()))
    .add_systems(Startup, setup_scene)
    // The meshes are spawned as `RaytracingMesh3d` directly, but their materials
    // are authored as `StandardMaterial` — convert those to `SolariMaterial` so the
    // RT path has real materials. Without it every instance falls back to the
    // default material: the 441 emissive spheres aren't emissive and the scene is
    // unlit (geometry still shows in the debug overlays, which ignore materials).
    .add_systems(Update, convert_standard_materials_to_solari)
    .add_systems(PostUpdate, update_performance_text)
    .run();
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut cluster_meshes: ResMut<Assets<ClusterMesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        FreeCamera {
            walk_speed: 50.0,
            run_speed: 250.0,
            ..Default::default()
        },
        Transform::from_translation(Vec3::new(6.11329, 166.74896, 451.8226).to_precision())
            .with_rotation(
                Quat::from_xyzw(-0.183938, 0.009093744, 0.0017017953, 0.9828943).to_precision(),
            ),
        Msaa::Off,
        SolariCamera,
        Bloom {
            intensity: 0.1,
            ..Bloom::NATURAL
        },
    ));

    // commands.spawn((
    //     DirectionalLight {
    //         illuminance: light_consts::lux::FULL_DAYLIGHT,
    //         shadow_maps_enabled: false,
    //         ..default()
    //     },
    //     Transform::from_rotation(Quat::from_xyzw(
    //         -0.13334629,
    //         -0.86597735,
    //         -0.3586996,
    //         0.3219264,
    //     )),
    // ));

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut plane_mesh = Plane3d::default()
        .mesh()
        .size(400.0, 400.0)
        .build()
        .with_generated_tangents()
        .unwrap();
    match plane_mesh.attribute_mut(Mesh::ATTRIBUTE_UV_0).unwrap() {
        VertexAttributeValues::Float32x2(items) => {
            items.iter_mut().flatten().for_each(|x| *x *= 3.0);
        }
        _ => unreachable!(),
    }
    // Bake each primitive into a ClusterMesh once for the ray-tracing
    // path; rasterization keeps the standard `Mesh` handle.
    let plane_mesh =
        cluster_meshes.add(ClusterMesh::try_from(&plane_mesh).expect("plane cluster bake"));
    let cube_geo = Cuboid::default()
        .mesh()
        .build()
        .with_generated_tangents()
        .unwrap();
    let cube_mesh =
        cluster_meshes.add(ClusterMesh::try_from(&cube_geo).expect("cube cluster bake"));
    let sphere_geo = Sphere::new(1.0)
        .mesh()
        .build()
        .with_generated_tangents()
        .unwrap();
    let sphere_mesh =
        cluster_meshes.add(ClusterMesh::try_from(&sphere_geo).expect("sphere cluster bake"));

    commands.spawn((
        RaytracingMesh3d(plane_mesh.clone()),
        MeshMaterial3d(
            materials.add(StandardMaterial {
                base_color_texture: Some(
                    asset_server
                        .load_builder()
                        .with_settings::<ImageLoaderSettings>(|settings| {
                            settings
                                .sampler
                                .get_or_init_descriptor()
                                .set_address_mode(ImageAddressMode::Repeat);
                        })
                        .load("textures/uv_checker_bw.png"),
                ),
                perceptual_roughness: 0.0,
                ..default()
            }),
        ),
    ));

    for _ in 0..8000 {
        commands.spawn((
            RaytracingMesh3d(cube_mesh.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(rng.random(), rng.random(), rng.random()),
                perceptual_roughness: rng.random(),
                ..default()
            })),
            Transform::default()
                .with_scale(
                    Vec3 {
                        x: rng.random_range(0.2..=2.0),
                        y: rng.random_range(0.2..=2.0),
                        z: rng.random_range(0.2..=2.0),
                    }
                    .to_precision(),
                )
                .with_translation(
                    Vec3::new(
                        rng.random_range(-180.0..=180.0),
                        0.2,
                        rng.random_range(-180.0..=180.0),
                    )
                    .to_precision(),
                ),
        ));
    }

    for x in -10..=10 {
        for y in -10..=10 {
            commands.spawn((
                RaytracingMesh3d(sphere_mesh.clone()),
                MeshMaterial3d(
                    materials.add(StandardMaterial {
                        emissive: Color::linear_rgb(
                            rng.random::<f32>() * 60000.0,
                            rng.random::<f32>() * 60000.0,
                            rng.random::<f32>() * 60000.0,
                        )
                        .into(),
                        ..default()
                    }),
                ),
                Transform::default().with_translation(
                    Vec3::new((x * 20) as f32, 7.0, (y * 20) as f32).to_precision(),
                ),
            ));
        }
    }

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            right: px(0.0),
            padding: px(4.0).all(),
            border_radius: BorderRadius::bottom_left(px(4.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.10, 0.10, 0.10, 0.8)),
        children![(
            PerformanceText,
            Text::default(),
            TextFont {
                font_size: FontSize::Px(8.0),
                ..default()
            },
        )],
    ));
}

#[derive(Component)]
struct PerformanceText;

fn update_performance_text(
    mut text: Single<&mut Text, With<PerformanceText>>,
    diagnostics: Res<DiagnosticsStore>,
) {
    text.0.clear();

    let mut total = 0.0;
    let mut add_diagnostic = |name: &str, path: &'static str| {
        let path = DiagnosticPath::new(path);
        if let Some(value) = diagnostics.get(&path).and_then(Diagnostic::smoothed) {
            text.push_str(&format!("{name:17}  {value:.2} ms\n"));
            total += value;
        }
    };

    // Per-pass GPU timings from the full-RT ReSTIR path (see
    // `bevy_solari::render::node::restir`). Requires `RenderDiagnosticsPlugin`.
    (add_diagnostic)("Visibility", "render/restir/visibility/elapsed_gpu");
    (add_diagnostic)("Presample", "render/restir/presample/elapsed_gpu");
    (add_diagnostic)(
        "Initial+temporal",
        "render/restir/initial_and_temporal/elapsed_gpu",
    );
    (add_diagnostic)(
        "Spatial+shade",
        "render/restir/spatial_and_shade/elapsed_gpu",
    );
    (add_diagnostic)("Specular GI", "render/restir/specular_gi/elapsed_gpu");
    (add_diagnostic)("Compose", "render/restir/compose/elapsed_gpu");
    text.push_str(&format!("{:17}  {total:.2} ms\n", "Total"));
}
