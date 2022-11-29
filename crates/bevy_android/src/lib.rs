use bevy_ecs::prelude::*;

pub use android_activity::*;
pub use ndk::asset::AssetManager;

#[derive(Resource)]
pub struct AndroidResource {
    pub android_app: AndroidApp,  
}
