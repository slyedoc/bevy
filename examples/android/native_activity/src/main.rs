use bevy::prelude::*;
fn main() {
    let mut app = App::new();
    android_native_activity::setup_app(&mut app);
}