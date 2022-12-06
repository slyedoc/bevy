use bevy::{
    app::AppExit, 
    log::{Level, LogPlugin},
    prelude::*
};

#[cfg(target_os = "android")]
use bevy::{
    render::settings::{WgpuLimits, WgpuSettings, WgpuSettingsPriority},
    winit::WinitSettings,
};

// Notes: I can't figure out anyway to use `bevy_main` to generate the android_main function
// without using global variable or using conditional parameters in main function which while it
// works it is not ideal and gives warnings, for now I will just use the android_main function here
// can see if i can move it too bevy_main later
#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(android_app: bevy::android::AndroidApp) {
    //android_logger::init_once(android_logger::Config::default().with_min_level(log::Level::Info));

    use bevy::android::AndroidResource;
    let mut app = App::new();
    app.insert_resource(AndroidResource {
        android_app: android_app.to_owned(),
    });
    build_app(&mut app);
    
}

pub fn build_app(
    #[cfg(target_os = "android")]
    app: &mut App
) {
    info!("Starting Build App");

    #[cfg(target_os = "android")]
    {
        // Android specific settings
        app
        .insert_resource(WgpuSettings {
            priority: WgpuSettingsPriority::Compatibility,
            limits: WgpuLimits {
                // Was required for my device and emulator
                max_storage_textures_per_shader_stage: 4,
                ..default()
            },
            ..default()
        });        
    }

    // Normal App Stuff
    app.add_plugins(DefaultPlugins.set(LogPlugin {
        filter: "android_activity=warn,wgpu=warn".to_string(),
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

    // Image, test asset server
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
        image: UiImage::new(asset_server.load("branding/icon.png")),
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

#[allow(dead_code)]
fn exit_soon(mut exit: EventWriter<AppExit>, mut count: Local<usize>) {
    *count += 1;
    if *count > 5 {
        exit.send(AppExit);
    }
}
