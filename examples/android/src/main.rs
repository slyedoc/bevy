use bevy::prelude::*;
fn main() {
    let mut app = App::new();
    bevy_android_example::build_app(&mut app);
    app.run();
}