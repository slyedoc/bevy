//! Demonstrates how to load and spawn BSN assets at runtime.

use std::f32::consts::{FRAC_PI_4, PI};

use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, animate_light_direction)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // Load the camera + directional light hierarchy at runtime from BSN.
    let scene_patch = asset_server.load("serialized_worlds/example.bsn");
    commands.spawn(ScenePatchInstance(scene_patch));

    // Spawn a mesh so there's something visible for the BSN camera to look at.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::from_length(0.5))),
        MeshMaterial3d(materials.add(StandardMaterial::from_color(Color::srgb(
            0.4, 0.6, 0.9,
        )))),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}

fn animate_light_direction(
    time: Res<Time>,
    mut query: Query<&mut Transform, With<DirectionalLight>>,
) {
    for mut transform in &mut query {
        transform.rotation = Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            time.elapsed_secs() * PI / 5.0,
            -FRAC_PI_4,
        );
    }
}
