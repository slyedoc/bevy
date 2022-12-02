use bevy::prelude::*;
fn main() {
    let mut app = App::new();
    android_native_activity::build_app(&mut app);
    app.run();
}