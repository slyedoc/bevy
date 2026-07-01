//! Bevy Solari split-screen — the same scene, ray traced vs rasterized.
//!
//! The pica_pica **diorama** is loaded **twice** into one world, co-located:
//! - the ray-traced copy (layer `RT_LAYER`) is tagged [`ConvertToRaytracing`]
//!   and baked by [`convert_marked_meshes_to_raytracing`] into ray-tracing
//!   meshes, shown by a `SolariCamera` on the left,
//! - the raster copy (layer `RASTER_LAYER`) is left as regular `Mesh3d` and
//!   shown by a standard `Camera3d` on the right — the normal Bevy raster path.
//!
//! Because only the marked copy is converted, the other survives as `Mesh3d`.
//! `RenderLayers` keeps each copy to its own camera, and a shared light sits on
//! both layers so both halves are lit. The left camera is `FreeCamera`-driven
//! (WASD + mouse-look); the right camera mirrors its transform each frame, so
//! both halves show the identical scene from the same viewpoint — ray traced on
//! the left, rasterized on the right.
//!
//! ```text
//! cargo run --example solari_split_screen
//! ```

use bevy::{
    camera::{visibility::RenderLayers, Viewport},
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    core_pipeline::prepass::DepthPrepass,
    dev_tools::render_debug::RenderDebugOverlay,
    feathers::{
        dark_theme::create_dark_theme,
        theme::{ThemedText, UiTheme},
        FeathersPlugins,
    },
    gltf::GltfMaterialName,
    math::DQuat,
    prelude::*,
    render::occlusion_culling::OcclusionCulling,
    solari::prelude::*,
    text::{Justify, TextLayout},
    window::WindowResized,
    world_serialization::WorldInstanceReady,
};

/// Left half: the Solari ray-traced diorama.
const RT_LAYER: usize = 0;
/// Right half: the standard Bevy rasterized scene. Any non-RT layer works: the
/// directional light is placed on both layers, and its shadow view inherits the
/// light's `RenderLayers` (bevy#16658 fix in `bevy_pbr`), so the raster scene
/// casts shadows regardless of which layer it's on.
const RASTER_LAYER: usize = 1;

fn main() {
    let mut app = App::new();

    app.add_plugins((
        DefaultPlugins.set(bevy::log::LogPlugin {
            filter: "wgpu=error,naga=warn,bevy_solari=info".into(),
            ..default()
        }),
        SolariPlugin,
        FreeCameraPlugin,
    ));

    // The Solari debug overlay needs Feathers.
    app.add_plugins(FeathersPlugins)
        .insert_resource(UiTheme(create_dark_theme()))
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (set_camera_viewports, sync_mirror_camera, draw_demo_gizmos),
        )
        // Converts ONLY the entities tagged `ConvertToRaytracing` (the diorama),
        // leaving the raster scene's `Mesh3d` untouched.
        .add_systems(Update, convert_marked_meshes_to_raytracing)
        .run();
}

/// Tag for [`set_camera_viewports`]: `(0,0)` = left half, `(1,0)` = right half.
#[derive(Component)]
struct CameraSlot {
    pos: UVec2,
}

/// Draw a few gizmos to exercise the ray-traced overlay path. The default gizmo
/// config renders on `RenderLayers::layer(0)` == [`RT_LAYER`], so these land on
/// the Solari (left) half. Solari's `solari_gizmo_depth` pass writes ray-traced
/// primary-hit depth into the hardware depth buffer before the gizmo
/// (`Transparent3d`) pass, so the sphere's far hemisphere is occluded by the
/// diorama — confirming rasterized overlays depth-test against the RT scene.
fn draw_demo_gizmos(mut gizmos: Gizmos) {
    gizmos
        .sphere(
            Isometry3d::from_translation(Vec3::new(0.0, 1.0, 0.0)),
            2.0,
            Color::srgb(1.0, 0.25, 0.2),
        )
        .resolution(64);
    gizmos.axes(Transform::from_xyz(0.0, 0.0, 0.0), 2.0);
}

/// The right camera mirrors the left camera's transform each frame.
#[derive(Component)]
struct MirrorCamera;

fn setup_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    // The SAME diorama, loaded twice and co-located. Copy A (layer 0) is tagged
    // `ConvertToRaytracing`, baked to ray-tracing meshes, and shown by the Solari
    // camera on the left. Copy B (layer 1) is left as `Mesh3d` and rasterized by
    // a standard camera on the right — an apples-to-apples RT-vs-raster split of
    // the identical scene. `RenderLayers` keeps each copy to its own camera.
    let diorama = asset_server.load(
        GltfAssetLabel::Scene(0).from_asset("https://github.com/bevyengine/bevy_asset_files/raw/2a5950295a8b6d9d051d59c0df69e87abcda58c3/pica_pica/mini_diorama_01.glb"),
    );
    let diorama_transform = Transform::from_scale(Vec3::splat(10.0).to_precision());
    commands
        .spawn((WorldAssetRoot(diorama.clone()), diorama_transform))
        .observe(setup_diorama(RT_LAYER, true));
    commands
        .spawn((WorldAssetRoot(diorama), diorama_transform))
        .observe(setup_diorama(RASTER_LAYER, false));

    // One light on BOTH layers so the ray-traced (left) and rasterized (right)
    // views are both lit. (Lights respect `RenderLayers` in the raster path.)
    // Shadow maps ON so the *raster* half casts shadows — Solari does its own RT
    // shadows and ignores this map, so the left half is unaffected.
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(DQuat::from_xyzw(
            -0.13334629,
            -0.86597735,
            -0.3586996,
            0.3219264,
        )),
        RenderLayers::layer(RT_LAYER).with(RASTER_LAYER),
    ));

    let camera_transform =
        Transform::from_translation(Vec3::new(0.219417, 2.5764852, 6.9718704).to_precision())
            .with_rotation(
                Quat::from_xyzw(-0.1466768, 0.013738206, 0.002037309, 0.989087).to_precision(),
            );

    // Left camera — Solari ray tracing, free-fly controlled.
    let left_camera = commands
        .spawn((
            Camera3d::default(),
            Camera {
                order: 0,
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            Msaa::Off,
            SolariCamera,
            RenderLayers::layer(RT_LAYER),
            CameraSlot {
                pos: UVec2::new(0, 0),
            },
            FreeCamera {
                walk_speed: 3.0,
                run_speed: 10.0,
                ..default()
            },
            camera_transform,
        ))
        .id();

    // Right camera — a standard Bevy raster camera that mirrors the left one.
    // `RenderDebugOverlay` (disabled) opts it into the bevy render-debug dropdown
    // (SolariDebugPlugin spawns one for any camera that has the component);
    // `DepthPrepass` + `OcclusionCulling` give the depth / depth-pyramid modes
    // their buffers.
    let right_camera = commands
        .spawn((
            Camera3d::default(),
            Camera {
                order: 1,
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            RenderLayers::layer(RASTER_LAYER),
            CameraSlot {
                pos: UVec2::new(1, 0),
            },
            MirrorCamera,
            camera_transform,
            RenderDebugOverlay::default(),
            DepthPrepass,
            OcclusionCulling,
        ))
        .id();

    // One Feathers label per half, top-center, targeted at its own camera so the
    // text lands in that camera's viewport.
    commands.spawn((
        Text::new("Solari"),
        ThemedText,
        top_center_label_node(),
        TextLayout::justify(Justify::Center),
        UiTargetCamera(left_camera),
    ));
    commands.spawn((
        Text::new("Bevy"),
        ThemedText,
        top_center_label_node(),
        TextLayout::justify(Justify::Center),
        UiTargetCamera(right_camera),
    ));
}

/// Top-center placement shared by both per-screen labels: a full-width strip at
/// the top of the viewport (`left`/`right` = 0); `Justify::Center` centers the
/// text within it.
fn top_center_label_node() -> Node {
    Node {
        position_type: PositionType::Absolute,
        top: px(12.0),
        left: px(0.0),
        right: px(0.0),
        ..default()
    }
}

/// Scene-load observer factory for a diorama copy: fixes its emissive/glass
/// materials (as in the `pica_pica` example), tags every mesh entity with
/// `RenderLayers::layer(layer)` (so only the matching camera sees it), and — if
/// `convert` — adds [`ConvertToRaytracing`] so the copy is baked to ray-tracing
/// meshes. `convert == false` leaves the copy as `Mesh3d` for the raster camera.
fn setup_diorama(
    layer: usize,
    convert: bool,
) -> impl Fn(
    On<WorldInstanceReady>,
    Query<&Children>,
    Query<(&MeshMaterial3d<StandardMaterial>, Option<&GltfMaterialName>)>,
    ResMut<Assets<StandardMaterial>>,
    Commands,
) + Send
       + Sync
       + 'static {
    move |scene_ready, children, mesh_query, mut materials, mut commands| {
        for descendant in children.iter_descendants(scene_ready.entity) {
            let Ok((MeshMaterial3d(material_handle), material_name)) = mesh_query.get(descendant)
            else {
                continue;
            };
            match material_name.map(|s| s.0.as_str()) {
                Some("material") => {
                    if let Some(mut material) = materials.get_mut(material_handle) {
                        material.emissive = LinearRgba::BLACK;
                    }
                }
                Some("Lights") => {
                    if let Some(mut material) = materials.get_mut(material_handle) {
                        material.emissive =
                            LinearRgba::from(Color::srgb(0.941, 0.714, 0.043)) * 1_000_000.0;
                        material.alpha_mode = AlphaMode::Opaque;
                        material.specular_transmission = 0.0;
                    }
                }
                Some("Glass_Dark_01") => {
                    if let Some(mut material) = materials.get_mut(material_handle) {
                        material.alpha_mode = AlphaMode::Opaque;
                        material.specular_transmission = 0.0;
                    }
                }
                _ => {}
            }
            commands
                .entity(descendant)
                .insert(RenderLayers::layer(layer));
            if convert {
                commands.entity(descendant).insert(ConvertToRaytracing);
            }
        }
    }
}

/// Resize each camera's viewport on every `WindowResized` (also fired once at
/// startup). The two cameras share the window 50/50 horizontally.
fn set_camera_viewports(
    windows: Query<&Window>,
    mut resize_reader: MessageReader<WindowResized>,
    mut cameras: Query<(&CameraSlot, &mut Camera)>,
) {
    for resized in resize_reader.read() {
        let Ok(window) = windows.get(resized.window) else {
            continue;
        };
        let half = UVec2::new(window.physical_size().x / 2, window.physical_size().y);
        for (slot, mut camera) in &mut cameras {
            camera.viewport = Some(Viewport {
                physical_position: slot.pos * UVec2::new(half.x, 0),
                physical_size: half,
                ..default()
            });
        }
    }
}

/// Copy the free-fly (left) camera's transform onto the mirror (right) camera so
/// both halves render the same viewpoint — ray traced on the left, rasterized on
/// the right.
fn sync_mirror_camera(
    left: Query<&Transform, (With<FreeCamera>, Without<MirrorCamera>)>,
    mut right: Query<&mut Transform, With<MirrorCamera>>,
) {
    let Ok(left) = left.single() else {
        return;
    };
    for mut transform in &mut right {
        *transform = *left;
    }
}
