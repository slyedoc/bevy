use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_log::*;

//use winit::{event_loop::EventLoopBuilder, platform::android::EventLoopBuilderExtAndroid};
pub use winit::{
    // Reexport of android-activity
    platform::android::activity::*,
};
pub use ndk::asset::AssetManager;

/// A resource to store the Android App.
#[derive(Resource)]
pub struct AndroidResource {
    pub android_app: AndroidApp,    
}

pub fn hack_loop(android_app: AndroidApp, build_app: impl Fn(&mut App)) {

    info!("Starting Hack Loop");
    // build winit loop
    //let event_loop = event_loop::EventLoop::new();
    // let _event_loop = EventLoopBuilder::new()
    //     .with_android_app(android_app.to_owned())
    //     .build();

    let mut app = App::new();
    // Needed for Asset Server
    app.insert_resource(AndroidResource { 
        android_app
    });
    
    build_app(&mut app);
    app.run();
}