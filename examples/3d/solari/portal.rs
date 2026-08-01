//! Bevy Solari portals — surfaces that teleport rays.
//!
//! Two framed portals stand in opposite corners of a courtyard, each showing
//! the view out of the other. Nothing is rendered twice and there is no
//! render-to-texture: a ray that hits a portal surface is TELEPORTED and keeps
//! flying, so recursion is free — portals seen through portals, portals in
//! reflections. The portal surface's material is flagged `portal`, so the
//! rt_pipeline SBT routes its hits to the dedicated `chit_portal` program; the
//! opaque chit never pays for the redirect.
//!
//! Walk (WASD) through a portal and the camera teleports with the same
//! transform the rays use, so the view is seamless across the crossing.

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState},
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig},
    diagnostic::FrameTimeDiagnosticsPlugin,
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    math::Affine3A,
    prelude::*,    
};
use std::f32::consts::{FRAC_PI_2, PI};

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "7c2e8a40-5b1d-4f7e-9c3a-6e2d8b4f1a93" // Don't copy paste this - generate your own UUID!
    )));

    app.insert_resource(UiTheme(create_dark_theme()))
        .add_plugins((
            DefaultPlugins,
            SolariPlugin,
            SolariAtmospherePlugin,
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
            (
                (
                    convert_meshes_to_raytracing,
                    convert_standard_materials_to_solari,
                )
                    .chain(),
                walk_through_portals,
            ),
        )
        .run();
}

/// Portal surface half-extents (the quad is `2*HX × 2*HY`).
const PORTAL_HX: f32 = 0.6;
const PORTAL_HY: f32 = 1.1;

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut solari_materials: ResMut<Assets<StandardSolariMaterial>>,
    portal_class: Res<SolariHitGroupClass<PortalSurface>>,
) {
    // ── Two portals, FACING EACH OTHER ─────────────────────────────────────
    //
    // The portal surface is an ordinary ray-traced quad; its `portal` material
    // SBT-routes hits to `chit_portal`, and `SolariPortal` teleports rays to the
    // target's frame (looking in shows the view out of the target's front, +Z).
    // Each portal's front faces the other across the courtyard; behind each one
    // sits a big wall in the OTHER portal's colour, so looking into the orange
    // portal shows the blue wall (and vice-versa) — the teleport made obvious.
    let orange_spot = Transform::from_xyz(-4.0, f64::from(PORTAL_HY), -5.0)
        .with_rotation(Quat::from_rotation_y(FRAC_PI_2).to_precision()); // front (+Z) → +X, toward blue
    let blue_spot = Transform::from_xyz(4.0, f64::from(PORTAL_HY), -5.0)
        .with_rotation(Quat::from_rotation_y(-FRAC_PI_2).to_precision()); // front (+Z) → -X, toward orange

    let portal_surface = meshes.add(Plane3d::new(Vec3::Z, Vec2::new(PORTAL_HX, PORTAL_HY)));
    // Never shaded — rays redirect on hit. Route to the built-in `chit_portal` via the
    // registered `PortalSurface` class (the same mechanism any custom surface uses).
    let portal_material = solari_materials.add(StandardSolariMaterial {
        chit_class: portal_class.get(),
        ..default()
    });

    let orange_portal = commands
        .spawn((
            Mesh3d(portal_surface.clone()),
            SolariMaterial3d(portal_material.clone()),
            orange_spot,
        ))
        .id();
    let blue_portal = commands
        .spawn((
            Mesh3d(portal_surface.clone()),
            SolariMaterial3d(portal_material.clone()),
            blue_spot,
        ))
        .id();
    commands.entity(orange_portal).insert(SolariPortal {
        target: blue_portal,
    });
    commands.entity(blue_portal).insert(SolariPortal {
        target: orange_portal,
    });

    // Frames (so the portals read as objects, not holes in the world).
    for (spot, color) in [
        (orange_spot, Color::srgb(0.9, 0.45, 0.05)),
        (blue_spot, Color::srgb(0.1, 0.45, 0.95)),
    ] {
        let frame_material = materials.add(StandardMaterial {
            base_color: color,
            emissive: LinearRgba::from(color.to_linear()) * 20.0,
            perceptual_roughness: 0.6,
            ..default()
        });
        let bar_w = 0.08;
        for (offset, size) in [
            (
                Vec3::new(-(PORTAL_HX + bar_w / 2.0), 0.0, 0.0),
                Vec3::new(bar_w, 2.0 * PORTAL_HY + 2.0 * bar_w, bar_w),
            ),
            (
                Vec3::new(PORTAL_HX + bar_w / 2.0, 0.0, 0.0),
                Vec3::new(bar_w, 2.0 * PORTAL_HY + 2.0 * bar_w, bar_w),
            ),
            (
                Vec3::new(0.0, PORTAL_HY + bar_w / 2.0, 0.0),
                Vec3::new(2.0 * PORTAL_HX, bar_w, bar_w),
            ),
            (
                Vec3::new(0.0, -(PORTAL_HY + bar_w / 2.0), 0.0),
                Vec3::new(2.0 * PORTAL_HX, bar_w, bar_w),
            ),
        ] {
            commands.spawn((
                Mesh3d(meshes.add(Cuboid::from_size(size))),
                MeshMaterial3d(frame_material.clone()),
                Transform {
                    translation: spot.transform_point(offset.to_precision()),
                    rotation: spot.rotation,
                    ..default()
                },
            ));
        }
    }

    // ── A big wall behind each portal, in the OTHER's colour ───────────────
    // ~10 ft behind each portal (along its back). Looking into the orange portal
    // exits the blue portal's front and looks back at the blue wall behind
    // orange — so the orange portal reads BLUE, the blue portal reads ORANGE.
    // Bright (emissive) so the teleported view is unmistakable.
    let wall = meshes.add(Cuboid::new(0.3, 5.0, 9.0));
    for (x, color) in [
        (-7.5, Color::srgb(0.1, 0.45, 0.95)), // blue wall behind the ORANGE portal
        (7.5, Color::srgb(0.9, 0.45, 0.05)),  // orange wall behind the BLUE portal
    ] {
        commands.spawn((
            Mesh3d(wall.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                emissive: LinearRgba::from(color.to_linear()) * 3.0,
                perceptual_roughness: 0.9,
                ..default()
            })),
            Transform::from_xyz(x, 2.5, -5.0),
        ));
    }

    // Ground.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(40.0, 40.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.4, 0.4, 0.38),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -3.0),
    ));

    commands.spawn((
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, -0.7, -1.0, 0.0).to_precision()),
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
            walk_speed: 2.0,
            run_speed: 6.0,
            ..Default::default()
        },
        Transform::from_xyz(0.0, 2.0, 3.0)
            .looking_at(Vec3::new(0.0, 1.2, -5.0).to_precision(), Vec3::Y),
        Msaa::Off,
        SolariCamera::default(),
        SolariAtmosphere::default(),
    ));
}

/// Teleport the camera when it crosses a portal surface — the exact same
/// mapping the rays use (`target_world × R_y(π) × portal_world⁻¹`), so the view
/// is continuous across the crossing. The FreeCamera's yaw/pitch state is
/// re-derived from the teleported rotation so the controller doesn't snap it back.
fn walk_through_portals(
    portals: Query<(Entity, &SolariPortal, &GlobalTransform)>,
    transforms: Query<&GlobalTransform>,
    mut camera: Single<(&mut Transform, &mut FreeCameraState), With<SolariCamera>>,
    mut previous_position: Local<Option<Vec3>>,
    mut cameras_reset: Query<&mut CameraReset, With<SolariCamera>>,
) {
    let (camera_transform, free_camera_state) = &mut *camera;
    let current = camera_transform.translation.to_render();
    let Some(previous) = previous_position.replace(current) else {
        return;
    };
    if previous == current {
        return;
    }

    for (_, portal, portal_transform) in &portals {
        // Segment previous→current vs the portal plane, inside the quad.
        let local_prev = portal_transform
            .affine()
            .inverse()
            .transform_point3(previous);
        let local_cur = portal_transform
            .affine()
            .inverse()
            .transform_point3(current);
        if local_prev.z.signum() == local_cur.z.signum() {
            continue; // didn't cross the plane
        }
        let t = local_prev.z / (local_prev.z - local_cur.z);
        let hit = local_prev.lerp(local_cur, t);
        if hit.x.abs() > PORTAL_HX || hit.y.abs() > PORTAL_HY {
            continue; // crossed the plane outside the surface
        }
        let Ok(target) = transforms.get(portal.target) else {
            continue;
        };

        let map = Mat4::from(target.affine() * Affine3A::from_rotation_y(PI))
            * Mat4::from(portal_transform.affine()).inverse();
        let teleported = map
            * Mat4::from_scale_rotation_translation(
                Vec3::ONE,
                camera_transform.rotation.to_render(),
                camera_transform.translation.to_render(),
            );
        let (_, rotation, translation) = teleported.to_scale_rotation_translation();
        camera_transform.translation = translation.to_precision();
        camera_transform.rotation = rotation.to_precision();

        // Re-derive the controller's look state from the new rotation.
        let (yaw, pitch, _) = rotation.to_euler(EulerRot::YXZ);
        free_camera_state.yaw = yaw;
        free_camera_state.pitch = pitch;

        // The teleport is a camera jump: restart the pathtracer accumulation.
        for mut reset in &mut cameras_reset {
            reset.history = true;
        }

        *previous_position = Some(translation);
        break;
    }
}
