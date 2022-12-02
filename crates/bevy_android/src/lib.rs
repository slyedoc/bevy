

use bevy_ecs::system::Resource;

pub use ndk::asset::AssetManager;
pub use winit::platform::android::activity::*;

/// A resource to store the Android App.

#[derive(Resource)]
pub struct AndroidResource {
    pub android_app: AndroidApp,
    //saver: RwLock<Vec<Arc<dyn AndroidSaver>>>,
}
pub const RESUME: i32 = 1;
pub const SUSPEND: i32 = 2;

pub struct HackLoop {
    pub status: BevyState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BevyState {
    Resume,
    Suspend,
}

//android_logger::init_once(android_logger::Config::default().with_min_level(log::Level::Info));
