//! Bevy Solari physical scope — a working 4× Galilean telescope built from
//! actual glass.
//!
//! Nothing here is a render-to-texture trick: the scope is two procedurally
//! lathed lens meshes (a biconvex objective, f ≈ +192mm, and a biconcave
//! eyepiece, f ≈ −48mm) with real spherical curvatures and IOR 1.52, mounted
//! in a tube. Magnification (f₁/|f₂| = 4×), the upright image, the eye-box,
//! and the edge distortion all EMERGE from rays refracting through the
//! elements — the same nested-dielectric tracing that renders the wine glass.
//!
//! Use the pathtrace view and hold still to converge. Walk (WASD) up to the
//! eyepiece and line up on the downrange target board; step off-axis to watch
//! the eye-box clip; back away to see the scope shrink to a bright exit
//! pupil.

use bevy::{
    asset::RenderAssetUsages,
    camera::CameraMainTextureUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig},
    diagnostic::FrameTimeDiagnosticsPlugin,
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::render_resource::TextureUsages,
    solari::prelude::*,
};
use std::f32::consts::PI;

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "31cb1b1e-79a8-49d3-a943-2c9f0c1e9a71" // Don't copy paste this - generate your own UUID!
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
        // The materials are authored as `StandardMaterial` (PbrPlugin is
        // enabled); convert meshes + materials for the RT scene.
        .add_systems(
            Update,
            (convert_meshes_to_raytracing, convert_standard_materials_to_solari).chain(),
        )
        .run();
}

/// Scope optical axis height above the floor.
const AXIS_Y: f32 = 0.30;
/// Glass IOR the focal lengths below are computed against.
const LENS_IOR: f32 = 1.52;

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut solari_materials: ResMut<Assets<SolariMaterial>>,
) {
    // BK7 crown glass, authored as `SolariMaterial` for its dispersion field
    // (Abbe 64.2 → 20/64.2): an uncorrected singlet telescope fringes color
    // at high-contrast edges — real chromatic aberration, not a post effect.
    let glass = solari_materials.add(SolariMaterial {
        base_color: Color::WHITE,
        specular_transmission: 1.0,
        ior: LENS_IOR,
        dispersion: 0.31,
        perceptual_roughness: 0.0,
        metallic: 0.0,
        ..default()
    });
    let tube_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.02, 0.02, 0.025),
        perceptual_roughness: 0.9,
        ..default()
    });

    // ── The telescope (Galilean: upright image, magnification f₁/|f₂| = 4×) ─
    //
    // Lensmaker (thin, symmetric): f = R / (2(n−1)).
    //   Objective: biconvex, R = ±0.20 m → f₁ ≈ +0.192 m.
    //   Eyepiece:  biconcave, R = ∓0.05 m → f₂ ≈ −0.048 m.
    // Element separation = f₁ + f₂ ≈ 0.144 m (the eyepiece intercepts the
    // converging cone before the objective's focus → parallel rays out).
    let objective_z = -0.072;
    let eyepiece_z = 0.072;

    let objective = meshes.add(lens_mesh(0.20, 0.20, 0.030, 0.012));
    commands.spawn((
        Mesh3d(objective),
        SolariMaterial3d(glass.clone()),
        Transform::from_xyz(0.0, AXIS_Y, objective_z),
    ));

    let eyepiece = meshes.add(lens_mesh(-0.05, -0.05, 0.012, 0.004));
    commands.spawn((
        Mesh3d(eyepiece),
        SolariMaterial3d(glass.clone()),
        Transform::from_xyz(0.0, AXIS_Y, eyepiece_z),
    ));

    // Barrel: a wide tube over the objective tapering to a narrow eyepiece
    // tube. Matte-dark inside, so off-axis rays die instead of ghosting.
    commands.spawn((
        Mesh3d(meshes.add(tube_mesh(0.031, 0.034, 0.17))),
        MeshMaterial3d(tube_material.clone()),
        Transform::from_xyz(0.0, AXIS_Y, -0.04),
    ));
    commands.spawn((
        Mesh3d(meshes.add(tube_mesh(0.013, 0.016, 0.06))),
        MeshMaterial3d(tube_material.clone()),
        Transform::from_xyz(0.0, AXIS_Y, 0.065),
    ));

    // Stand.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.02, AXIS_Y - 0.034, 0.02))),
        MeshMaterial3d(tube_material.clone()),
        Transform::from_xyz(0.0, (AXIS_Y - 0.034) / 2.0, -0.04),
    ));

    // ── Downrange ───────────────────────────────────────────────────────────

    // Target board on the optical axis at 10 m: concentric rings, sized so
    // the naked eye barely resolves them and the scope plainly does.
    let ring_colors = [
        Color::srgb(0.9, 0.9, 0.85),
        Color::srgb(0.8, 0.1, 0.1),
        Color::srgb(0.9, 0.9, 0.85),
        Color::srgb(0.8, 0.1, 0.1),
        Color::srgb(0.1, 0.1, 0.1),
    ];
    let radii = [0.50, 0.40, 0.30, 0.20, 0.10];
    for (i, (radius, color)) in radii.iter().zip(ring_colors).enumerate() {
        commands.spawn((
            Mesh3d(meshes.add(Cylinder::new(*radius, 0.01))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                perceptual_roughness: 0.9,
                ..default()
            })),
            Transform::from_xyz(0.0, AXIS_Y, -10.0 + i as f32 * 0.006)
                .with_rotation(Quat::from_rotation_x(PI / 2.0)),
        ));
    }

    // The glass dragon at 7 m, just off-axis — organic detail to magnify.
    commands.spawn((
        WorldAssetRoot(
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/DragonAttenuation.glb")),
        ),
        Transform::from_xyz(1.2, 0.7306 * 0.5, -7.0)
            .with_scale(Vec3::splat(0.5))
            .with_rotation(Quat::from_rotation_y(PI / 2.0)),
    ));
    // (Its backdrop cloth stays — a checkered surface downrange is useful.)

    // Ground.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(40.0, 40.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.36, 0.33),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -10.0),
    ));

    commands.spawn((
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, -0.6, -0.9, 0.0)),
        SolariDirectionLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            ..default()
        },
    ));

    // Eye at the eyepiece (Galilean eye relief is short). WASD to line up;
    // back away to watch the image collapse into the exit pupil.
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        FreeCamera {
            walk_speed: 0.25,
            run_speed: 1.5,
            ..Default::default()
        },
        Transform::from_xyz(0.0, AXIS_Y, eyepiece_z + 0.030)
            .looking_at(Vec3::new(0.0, AXIS_Y, -10.0), Vec3::Y),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariCamera,
        SolariAtmosphere::default(),
    ));
}

/// Spherical sag: depth of a sphere surface (radius `r`) at distance `x`
/// from the axis.
fn sag(x: f32, r: f32) -> f32 {
    r - (r * r - x * x).sqrt()
}

/// Lathe a spherical lens around the Z axis.
///
/// `r_front`/`r_back` are surface curvature radii: positive = convex
/// (bulging outward), negative = concave (cut inward). `aperture` is the
/// lens radius; `thickness` the center (axis) thickness. The front surface
/// faces +Z. Closed manifold: front cap, edge band, back cap.
fn lens_mesh(r_front: f32, r_back: f32, aperture: f32, thickness: f32) -> Mesh {
    // Surface height along Z at radial distance `x`.
    let z_front = |x: f32| -> f32 {
        if r_front > 0.0 {
            thickness / 2.0 - sag(x, r_front)
        } else {
            thickness / 2.0 + sag(x, -r_front)
        }
    };
    let z_back = |x: f32| -> f32 {
        if r_back > 0.0 {
            -thickness / 2.0 + sag(x, r_back)
        } else {
            -thickness / 2.0 - sag(x, -r_back)
        }
    };

    const SEGMENTS: usize = 96;
    const RINGS: usize = 24;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Numerical profile slope for the surface normal: the lathe of curve
    // (x, z(x)) has normal ∝ (-z'(x)·radial_dir, 1) for a +Z-facing surface.
    let slope = |z: &dyn Fn(f32) -> f32, x: f32| -> f32 {
        let eps = (aperture * 1e-3).max(1e-6);
        let lo = (x - eps).max(0.0);
        let hi = (x + eps).min(aperture);
        (z(hi) - z(lo)) / (hi - lo)
    };

    // One revolved surface: rings from the axis outward. `flip` = -1 for the
    // back surface (normal faces -Z, triangle winding reversed).
    let mut lathe_cap = |z: &dyn Fn(f32) -> f32, flip: f32| {
        let base = positions.len() as u32;
        // Apex vertex (a triangle fan center at the axis).
        positions.push([0.0, 0.0, z(0.0)]);
        normals.push([0.0, 0.0, flip]);
        uvs.push([0.5, 0.5]);
        for ring in 1..=RINGS {
            let x = aperture * ring as f32 / RINGS as f32;
            let dz = slope(z, x);
            for seg in 0..=SEGMENTS {
                let theta = 2.0 * PI * seg as f32 / SEGMENTS as f32;
                let (sin_t, cos_t) = theta.sin_cos();
                positions.push([x * cos_t, x * sin_t, z(x)]);
                let n = Vec3::new(-dz * cos_t, -dz * sin_t, 1.0).normalize() * flip;
                normals.push([n.x, n.y, n.z]);
                uvs.push([0.5 + 0.5 * (x / aperture) * cos_t, 0.5 + 0.5 * (x / aperture) * sin_t]);
            }
        }
        let ring_start = |ring: usize| base + 1 + ((ring - 1) * (SEGMENTS + 1)) as u32;
        // Fan: apex → first ring.
        for seg in 0..SEGMENTS as u32 {
            let (a, b) = (ring_start(1) + seg, ring_start(1) + seg + 1);
            if flip > 0.0 {
                indices.extend([base, a, b]);
            } else {
                indices.extend([base, b, a]);
            }
        }
        // Quads between rings.
        for ring in 1..RINGS {
            for seg in 0..SEGMENTS as u32 {
                let (a, b) = (ring_start(ring) + seg, ring_start(ring) + seg + 1);
                let (c, d) = (ring_start(ring + 1) + seg, ring_start(ring + 1) + seg + 1);
                if flip > 0.0 {
                    indices.extend([a, c, d, a, d, b]);
                } else {
                    indices.extend([a, d, c, a, b, d]);
                }
            }
        }
    };

    lathe_cap(&z_front, 1.0);
    lathe_cap(&z_back, -1.0);

    // Edge band: a cylinder at x = aperture joining the two surface rims.
    let band = positions.len() as u32;
    for (z, v) in [(z_front(aperture), 0.0f32), (z_back(aperture), 1.0)] {
        for seg in 0..=SEGMENTS {
            let theta = 2.0 * PI * seg as f32 / SEGMENTS as f32;
            let (sin_t, cos_t) = theta.sin_cos();
            positions.push([aperture * cos_t, aperture * sin_t, z]);
            normals.push([cos_t, sin_t, 0.0]);
            uvs.push([seg as f32 / SEGMENTS as f32, v]);
        }
    }
    for seg in 0..SEGMENTS as u32 {
        let (a, b) = (band + seg, band + seg + 1);
        let (c, d) = (band + (SEGMENTS as u32 + 1) + seg, band + (SEGMENTS as u32 + 1) + seg + 1);
        indices.extend([a, b, d, a, d, c]);
    }

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

/// A hollow open-ended tube (annular cross-section revolved around Z):
/// outer wall, inner wall, and the two end rings — a closed solid.
fn tube_mesh(inner: f32, outer: f32, length: f32) -> Mesh {
    const SEGMENTS: usize = 64;
    let h = length / 2.0;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Four bands with hard normals: (radius pair, z pair, normal).
    // Each band is a quad strip between two rings of vertices.
    let bands: [([f32; 2], [f32; 2], fn(f32, f32) -> [f32; 3]); 4] = [
        ([outer, outer], [-h, h], |c, s| [c, s, 0.0]),   // outer wall: +radial
        ([inner, inner], [h, -h], |c, s| [-c, -s, 0.0]), // inner wall: -radial
        ([inner, outer], [h, h], |_, _| [0.0, 0.0, 1.0]), // front ring: +Z
        ([outer, inner], [-h, -h], |_, _| [0.0, 0.0, -1.0]), // back ring: -Z
    ];

    for (radii, zs, normal) in bands {
        let base = positions.len() as u32;
        for k in 0..2 {
            for seg in 0..=SEGMENTS {
                let theta = 2.0 * PI * seg as f32 / SEGMENTS as f32;
                let (sin_t, cos_t) = theta.sin_cos();
                positions.push([radii[k] * cos_t, radii[k] * sin_t, zs[k]]);
                normals.push(normal(cos_t, sin_t));
                uvs.push([seg as f32 / SEGMENTS as f32, k as f32]);
            }
        }
        for seg in 0..SEGMENTS as u32 {
            let (a, b) = (base + seg, base + seg + 1);
            let (c, d) = (
                base + (SEGMENTS as u32 + 1) + seg,
                base + (SEGMENTS as u32 + 1) + seg + 1,
            );
            indices.extend([a, b, d, a, d, c]);
        }
    }

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}
