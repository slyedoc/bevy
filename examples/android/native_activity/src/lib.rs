use bevy::{
    log::*,
    prelude::*,
    render::settings::{WgpuSettings, WgpuSettingsPriority},
};

// Notes: I can't figure out anyway to use `bevy_main` to generate the android_main function
// without using global variable or using conditional parameters in main function which while it
// works it is not ideal and gives warnings

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(android_app: bevy::android::AndroidApp) {
    let mut app = App::new();
    #[cfg(target_os = "android")]
    app.insert_non_send_resource(bevy::android::AndroidResource { android_app });

    setup_app(&mut app);
}


pub fn setup_app( app: &mut App ) {

    // This configures the app to use the most compatible rendering settings.
    // They help with compatibility with as many devices as possible.
    app.insert_resource(WgpuSettings {
        priority: WgpuSettingsPriority::Compatibility,
        ..default()
    })
    .add_plugins(DefaultPlugins.set(LogPlugin {
        //filter: "android_activity=warn,wgpu=error".to_string(),
        filter: "android_activity=debug,wgpu=error".to_string(),
        level: Level::INFO,
    }))
    .add_startup_system(setup)
    .add_system(rotate_camera)
    .run();
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Plane { size: 5.0 })),
        material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
        ..default()
    });
    // cube
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
        transform: Transform::from_xyz(0.0, 0.5, 0.0),
        ..default()
    });
    // light
    commands.spawn(PointLightBundle {
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });
    // camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // asset test
    commands.spawn(ImageBundle {
        style: Style {
            size: Size::new(Val::Px(50.0), Val::Px(50.0)),
            position: UiRect {
                left: Val::Px(10.0),
                top: Val::Px(10.0),
                ..Default::default()
            },
            ..default()
        },
        image: UiImage::new(asset_server.load("icon.png")),
        ..default()
    });
}

/// Rotate the camera around the origin
fn rotate_camera(mut query: Query<&mut Transform, With<Camera>>, time: Res<Time>) {
    for mut transform in &mut query {
        transform.rotate_around(Vec3::ZERO, Quat::from_rotation_y(time.delta_seconds()));
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}
