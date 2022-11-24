use bevy::{
    prelude::*,
    //render::settings::{WgpuSettings, WgpuSettingsPriority},
    winit::WinitAndroidApp,
};

// the `bevy_main` proc_macro generates the required android boilerplate

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(android_app: bevy::winit::android::activity::AndroidApp) {
    //android_logger::init_once(android_logger::Config::default().with_min_level(log::Level::Trace));

    // let event_loop = EventLoopBuilder::with_user_event()
    //     .with_android_app(app)
    //     .build();
    //main(event_loop);
    // TODO: figure out how to pass the event loop to main

    let mut app = App::new();
    app.insert_non_send_resource(WinitAndroidApp(Some(android_app)));

    app.add_plugins(
        DefaultPlugins, // See notes in WinitPlugin, this doesn't work
                        // .set(bevy::winit::WinitPlugin {
                        //     #[cfg(target_os = "android")]
                        //     android_app: Some(app),
                        // })
    )
    .add_startup_system(setup)
    .run();
}

// fn main(// #[cfg(target_os = "android")]
//     // android_app: bevy::winit::android::activity::AndroidApp
// ) {
//     let mut app = App::new();
//     // This configures the app to use the most compatible rendering settings.
//     // They help with compatibility with as many devices as possible.
//     // .insert_resource(WgpuSettings {
//     //     priority: WgpuSettingsPriority::Compatibility,
//     //     ..default()
//     // })
//     app.add_plugins(
//         DefaultPlugins, // See notes in WinitPlugin, this doesn't work
//                         // .set(bevy::winit::WinitPlugin {
//                         //     #[cfg(target_os = "android")]
//                         //     android_app: Some(app),
//                         // })
//     )
//     .add_startup_system(setup)
//     .run();
// }

// /// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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
}
