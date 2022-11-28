//! Illustrates how to change window settings and shows how to affect
//! the mouse pointer in various ways.

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
    window::{CursorGrabMode, PresentMode, WindowLevel},
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            window: WindowDescriptor {
                title: "I am a window!".to_string(),
                width: 500.,
                height: 300.,
                present_mode: PresentMode::AutoVsync,
                level: WindowLevel::AlwaysOnTop,
                ..default()
            },
            ..default()
        }))
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin)
        .add_system(change_title)
        .add_system(toggle_cursor)
        .add_system(toggle_vsync)
        .add_system(cycle_cursor_icon)
        .add_system(cycle_level)
        .run();
}

/// This system toggles the vsync mode when pressing the button V.
/// You'll see fps increase displayed in the console.
fn toggle_vsync(input: Res<Input<KeyCode>>, mut windows: ResMut<Windows>) {
    if input.just_pressed(KeyCode::V) {
        let window = windows.primary_mut();

        window.set_present_mode(if matches!(window.present_mode(), PresentMode::AutoVsync) {
            PresentMode::AutoNoVsync
        } else {
            PresentMode::AutoVsync
        });
        info!("PRESENT_MODE: {:?}", window.present_mode());
    }
}

/// This system cycles whether the window is always on top, always on bottom, or normal when pressing the T button
/// You'll notice it won't be covered by other windows, all window, or normal.
///
/// This feature only works on some platforms. Please check the
/// [documentation](https://docs.rs/bevy/latest/bevy/prelude/struct.WindowDescriptor.html#structfield.level)
/// for more details.
fn cycle_level(input: Res<Input<KeyCode>>, mut windows: ResMut<Windows>) {
    if input.just_pressed(KeyCode::T) {
        let window = windows.primary_mut();

        let level: WindowLevel = window.level();

        match level {
            WindowLevel::AlwaysOnBottom => {
                info!("UNLOCKING WINDOW");
                window.set_level(WindowLevel::Normal);
            }
            WindowLevel::Normal => {
                info!("LOCKING WINDOW ON TOP");
                window.set_level(WindowLevel::AlwaysOnTop);
                
            },
            WindowLevel::AlwaysOnTop => {
                info!("LOCKING WINDOW ON BOTTOM");
                window.set_level(WindowLevel::AlwaysOnBottom);
            }
        };        
    }
}

/// This system will then change the title during execution
fn change_title(time: Res<Time>, mut windows: ResMut<Windows>) {
    let window = windows.primary_mut();
    window.set_title(format!(
        "Seconds since startup: {}",
        time.elapsed_seconds().round()
    ));
}

/// This system toggles the cursor's visibility when the space bar is pressed
fn toggle_cursor(input: Res<Input<KeyCode>>, mut windows: ResMut<Windows>) {
    let window = windows.primary_mut();
    if input.just_pressed(KeyCode::Space) {
        window.set_cursor_grab_mode(match window.cursor_grab_mode() {
            CursorGrabMode::None => CursorGrabMode::Locked,
            CursorGrabMode::Locked | CursorGrabMode::Confined => CursorGrabMode::None,
        });
        window.set_cursor_visibility(!window.cursor_visible());
    }
}

/// This system cycles the cursor's icon through a small set of icons when clicking
fn cycle_cursor_icon(
    input: Res<Input<MouseButton>>,
    mut windows: ResMut<Windows>,
    mut index: Local<usize>,
) {
    const ICONS: &[CursorIcon] = &[
        CursorIcon::Default,
        CursorIcon::Hand,
        CursorIcon::Wait,
        CursorIcon::Text,
        CursorIcon::Copy,
    ];
    let window = windows.primary_mut();
    if input.just_pressed(MouseButton::Left) {
        *index = (*index + 1) % ICONS.len();
        window.set_cursor_icon(ICONS[*index]);
    } else if input.just_pressed(MouseButton::Right) {
        *index = if *index == 0 {
            ICONS.len() - 1
        } else {
            *index - 1
        };
        window.set_cursor_icon(ICONS[*index]);
    }
}
