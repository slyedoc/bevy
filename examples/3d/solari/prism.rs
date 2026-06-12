//! Bevy Solari prism — the Newton experiment, staged to work.
//!
//! A slit of sunlight enters a flint-glass prism at minimum deviation and the
//! floor catches a real spectrum: the band order, width, and position all
//! EMERGE from refraction — per-wavelength IOR (Cauchy fit from the
//! material's `dispersion`), path-regularized caustic next-event estimation,
//! and hero-wavelength spectral paths. Nothing is painted.
//!
//! The geometry is solved, not eyeballed. For an equilateral prism at
//! n_d = 1.62, minimum deviation puts the incident beam 54.1° off the entry
//! face normal (the internal ray runs parallel to the base) and bends the
//! beam 48.2° in total. The prism is tilted 24.1° so that beam comes DOWN AT
//! 60° — afternoon sun through a window — which leaves the exit beam skimming
//! the floor at only 11.8°: the spectrum stretches from the prism's base
//! deep into the dark (violet lands at ~1.41 m, red at ~2.19 m; shallower
//! exit = longer throw, so red is the far end). The beam threads near the
//! prism's APEX — entry and exit close together keeps the effective source
//! compact, the classroom trick for clean bands. A huge
//! tilted shutter perpendicular to the beam carves the 24 mm slit and throws
//! everything else into shadow.
//!
//! Watch it in the PATHTRACE view (bottom-left dropdown) and hold still — the
//! spectrum is caustic transport, which the realtime path doesn't carry. The
//! slider raises the prism's dispersion: the fan stretches live.

use bevy::{
    camera::CameraMainTextureUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState},
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig},
    diagnostic::FrameTimeDiagnosticsPlugin,
    feathers::{
        self,
        controls::FeathersSlider,
        dark_theme::create_dark_theme,
        theme::{ThemeBackgroundColor, ThemeTextColor, UiTheme},
        FeathersPlugins,
    },
    prelude::*,
    render::render_resource::TextureUsages,
    solari::prelude::*,
    ui_widgets::{slider_self_update, SliderPrecision, SliderStep, ValueChange},
};

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "7c2e5b90-3f1a-4e8d-9b6c-2a8f0d4e7c13" // Don't copy paste this - generate your own UUID!
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
        .add_systems(Startup, (setup_scene, dispersion_ui.spawn()))
        .add_systems(
            Update,
            (convert_meshes_to_raytracing, convert_standard_materials_to_solari).chain(),
        )
        .run();
}

/// The window-light beam: descends at 60°, in the XY plane (see module docs).
const BEAM_DIRECTION: Vec3 = Vec3::new(0.5, -0.866, 0.0);
/// Prism tilt about its axis that puts that beam at minimum deviation.
const PRISM_TILT: f32 = 0.4206;

/// Marker for the prism entity, so the tilt slider can find it.
#[derive(Component)]
struct Prism;

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut solari_materials: ResMut<Assets<SolariMaterial>>,
) {
    // The prism: equilateral cross-section (side 0.24), axis along Z, tilted
    // to minimum deviation for the window beam and resting low on a small
    // stand — the spectrum starts at its base.
    commands.spawn((
        Mesh3d(meshes.add(Extrusion::new(
            Triangle2d::new(
                Vec2::new(0.0, 0.1386),
                Vec2::new(-0.12, -0.0693),
                Vec2::new(0.12, -0.0693),
            ),
            0.3,
        ))),
        SolariMaterial3d(solari_materials.add(SolariMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.0,
            specular_transmission: 1.0,
            ior: 1.62,
            dispersion: 0.63,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.28, 0.0).with_rotation(Quat::from_rotation_z(PRISM_TILT)),
        Prism,
    ));

    let dark = materials.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.08, 0.09),
        perceptual_roughness: 1.0,
        ..default()
    });

    // The stand under the prism's lowest edge.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.2, 0.165, 0.26))),
        MeshMaterial3d(dark.clone()),
        Transform::from_xyz(0.0, 0.0825, 0.0),
    ));

    // The window shutter: a huge plate pair PERPENDICULAR to the beam (a
    // tilted wall plane keeps the slit a clean 24 mm aperture — a slit
    // through a thick upright wall at 60° incidence would self-occlude).
    // The gap sits 0.6 m up-beam of a point 35% down the entry face from
    // the APEX (world (−0.065, 0.323)) — passing the light near the tip is
    // what keeps the bands clean: entry and exit sit close together, so the
    // effective source stays compact and the corner can't clip the beam.
    // The upper plate is long enough that its shadow covers the whole
    // spectrum landing zone; the lower one reaches just short of the floor.
    // A narrow beam is what makes the spectrum: full-face sunlight overlaps
    // a spectrum from every aperture point and washes back to white.
    let shutter_rotation = Quat::from_rotation_z(0.5236); // local +Y = beam normal
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.3, 0.02, 2.4))),
        MeshMaterial3d(dark.clone()),
        Transform::from_xyz(0.641, 1.424, 0.0).with_rotation(shutter_rotation),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.2, 0.02, 2.4))),
        MeshMaterial3d(dark.clone()),
        Transform::from_xyz(-0.895, 0.537, 0.0).with_rotation(shutter_rotation),
    ));
    // End caps: limit the slit's length to the prism's depth (z ±0.15) so
    // the beam is a short bar matched to the glass, not a 2.4 m line.
    for z in [-0.675, 0.675] {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(0.024, 0.02, 1.05))),
            MeshMaterial3d(dark.clone()),
            Transform::from_xyz(-0.365, 0.843, z).with_rotation(shutter_rotation),
        ));
    }

    // A matte near-white floor: the spectrum lands stretched from
    // x ≈ 1.41 (violet) to x ≈ 2.19 (red).
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(8.0, 8.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.82, 0.8, 0.76),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::IDENTITY,
    ));

    // The dark room. An open scene drowns the spectrum: the 250k-lux sun on
    // acres of white floor bounces more light into the shadow than the slit
    // delivers. Sealed walls + roof kill that; the only way in is a roof
    // opening at (−1.13, 0) — exactly where the beam crosses y ≈ 2 — sized
    // so everything it admits lands on the shutter plates below. Newton's
    // chamber: the room is black except the shaft, the prism, and the
    // spectrum.
    let wall = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 0.3, 0.32),
        perceptual_roughness: 0.95,
        ..default()
    });
    // Walls (interior x −1.6..2.2, z ±1.4, height 2).
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.1, 2.0, 2.8))),
        MeshMaterial3d(wall.clone()),
        Transform::from_xyz(-1.65, 1.0, 0.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.1, 2.0, 2.8))),
        MeshMaterial3d(wall.clone()),
        Transform::from_xyz(2.25, 1.0, 0.0),
    ));
    for z in [-1.45, 1.45] {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(3.9, 2.0, 0.1))),
            MeshMaterial3d(wall.clone()),
            Transform::from_xyz(0.3, 1.0, z),
        ));
    }
    // Roof, with the beam opening (x −1.33..−0.93, z ±0.3) framed out.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.37, 0.1, 2.9))),
        MeshMaterial3d(wall.clone()),
        Transform::from_xyz(-1.515, 2.05, 0.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(3.23, 0.1, 2.9))),
        MeshMaterial3d(wall.clone()),
        Transform::from_xyz(0.685, 2.05, 0.0),
    ));
    for z in [-0.875, 0.875] {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(0.4, 0.1, 1.15))),
            MeshMaterial3d(wall.clone()),
            Transform::from_xyz(-1.13, 2.05, z),
        ));
    }

    // A whisper of dust filling the room: the slit beam renders as a visible
    // shaft cutting down to the prism (the fog march's per-step sun shadow
    // rays thread the slit), at a density too thin to dim the spectrum.
    commands.spawn((
        SolariFogVolume {
            density: 0.008,
            albedo: Vec3::splat(0.85),
            phase_g: 0.5,
            softness: 0.15,
            spherical: false,
        },
        Transform::from_xyz(0.3, 1.0, 0.0).with_scale(Vec3::new(1.9, 1.0, 1.4)),
    ));

    // The sun, aimed exactly down the solved beam direction, with a tighter
    // disk than default — penumbra blurs the bands.
    commands.spawn((
        Transform::from_rotation(Quat::from_rotation_arc(Vec3::Z, -BEAM_DIRECTION)),
        SolariDirectionLight {
            // FAR hotter than real daylight: the slit throws away most of
            // the light and the grazing 11.8° exit spreads the rest thin —
            // the demo is the band, not exposure realism.
            illuminance: 800_000.0,
            sun_disk_angular_size: 0.003,
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
            walk_speed: 1.0,
            run_speed: 4.0,
            ..Default::default()
        },
        Transform::from_xyz(1.2, 0.6, 1.35).looking_at(Vec3::new(0.9, 0.1, 0.0), Vec3::Y),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariCamera,
        // Sky + sun color (global height fog defaults off).
        SolariAtmosphere::default(),
    ));
}

/// Dispersion control (top right): writes the prism's material and restarts
/// the pathtracer's accumulation.
fn dispersion_ui() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            top: px(10),
            right: px(10),
            padding: px(8),
        }
        ThemeBackgroundColor(feathers::tokens::WINDOW_BG)
        on(|_: On<Pointer<Over>>, mut free_camera_state: Single<&mut FreeCameraState>| {
            free_camera_state.enabled = false;
        })
        on(|_: On<Pointer<Out>>, mut free_camera_state: Single<&mut FreeCameraState>| {
            free_camera_state.enabled = true;
        })
        Children [(
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Stretch,
                justify_content: JustifyContent::Start,
                row_gap: px(8),
                min_width: px(180),
            }
            Children [
                Text("Prism"),
                (
                    Text("Tilt (deg)")
                    TextFont { font_size: FontSize::Px(14.0) }
                    ThemeTextColor(feathers::tokens::CHECKBOX_TEXT)
                ),
                (
                    // 24.1° is the solved minimum-deviation tilt for the 60°
                    // window beam — the spectrum is brightest and tightest
                    // there. Tilting away sweeps the fan across the floor,
                    // dims it (off-minimum reflectance losses), and far
                    // enough off the exit face goes total-internal-reflection
                    // and the spectrum dies.
                    @FeathersSlider {
                        @min: -10.0,
                        @max: 60.0,
                        @value: 24.1,
                    }
                    SliderStep(0.5)
                    SliderPrecision(1)
                    on(slider_self_update)
                    on(|change: On<ValueChange<f32>>,
                        mut prisms: Query<&mut Transform, With<Prism>>,
                        mut resets: Query<&mut CameraReset>| {
                        for mut transform in &mut prisms {
                            transform.rotation = Quat::from_rotation_z(change.value.to_radians());
                        }
                        for mut reset in &mut resets {
                            reset.0 = true;
                        }
                    })
                ),
                (
                    Text("Dispersion (20/Abbe)")
                    TextFont { font_size: FontSize::Px(14.0) }
                    ThemeTextColor(feathers::tokens::CHECKBOX_TEXT)
                ),
                (
                    // Flint glass ≈ 0.55, diamond ≈ 0.63; higher is
                    // exaggeration — the fan widens proportionally.
                    @FeathersSlider {
                        @min: 0.0,
                        @max: 1.5,
                        @value: 0.63,
                    }
                    SliderStep(0.01)
                    SliderPrecision(2)
                    on(slider_self_update)
                    on(|change: On<ValueChange<f32>>,
                        mut materials: ResMut<Assets<SolariMaterial>>,
                        mut resets: Query<&mut CameraReset>| {
                        let ids: Vec<_> = materials.ids().collect();
                        for id in ids {
                            if let Some(mut material) = materials.get_mut(id)
                                && material.specular_transmission > 0.0
                            {
                                material.dispersion = change.value;
                            }
                        }
                        for mut reset in &mut resets {
                            reset.0 = true;
                        }
                    })
                ),
            ]
        )]
    }
}
