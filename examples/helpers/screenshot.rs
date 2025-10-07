use std::path::Path;

use bevy::{    
    input::common_conditions::input_just_pressed,
    prelude::*,
    render::view::screenshot::{save_to_disk, Capturing, Screenshot, ScreenshotCaptured},
    window::{CursorIcon, SystemCursorIcon, Window},
};

pub struct ScreenshotPlugin {
    /// The path where screenshots will be saved
    pub path: String,
    /// The key to trigger the screenshot
    pub key: KeyCode,
    /// If Some, the plugin will take a screenshot after this duration in seconds
    pub timer: Option<f32>,
    /// If true, the application will exit after taking the screenshot, only used if `timer` is set
    /// to Some.
    pub exit: bool,
}

impl Default for ScreenshotPlugin {
    fn default() -> Self {
        Self {
            path: ".".to_string(),
            key: KeyCode::F12,
            timer: None,
            exit: false,
        }
    }
}

impl ScreenshotPlugin {
    pub fn new(auto: bool) -> Self {    
        if auto {
            Self::automated()
        } else {
            Self::default()
        }
    }

    pub fn automated() -> Self{
        Self {
            path: ".".to_string(),
            key: KeyCode::F12,
            timer: Some(2.0),
            exit: true,
        }
    }
}

#[derive(Resource)]
pub struct ScreenshotPath(pub String);

#[derive(Resource)]
pub struct ScreenshotTimer {
    pub timer: Timer,
    pub exit: bool,
}   
    

impl Plugin for ScreenshotPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ScreenshotPath(self.path.clone()))
            .add_observer(take_screenshot)
            .add_systems(
                Update,
                (
                    (|mut commands: Commands| {
                        commands.trigger(TakeScreenshot(false));
                    })
                    .run_if(input_just_pressed(self.key)),
                    screenshot_saving,
                ),
            );

        if let Some(time) = self.timer {
            app.insert_resource(ScreenshotTimer {
                timer: Timer::from_seconds(time, TimerMode::Once),
                exit: self.exit,
            })
                .add_systems(Update, update_timer);

   
        }
        

    }
}

fn update_timer(
    mut commands: Commands, mut screen_shot_timer: ResMut<ScreenshotTimer>, time: Res<Time>
) {
    if screen_shot_timer.timer.tick(time.delta()).just_finished() {
        commands.trigger(TakeScreenshot(screen_shot_timer.exit));
    }
}

fn exit_on_capture(
    _: On<ScreenshotCaptured>,
    mut commands: Commands,     
) {
    commands.write_message(AppExit::Success);        
}

#[derive(Event)]
pub struct TakeScreenshot( pub bool); // exit after capture

fn take_screenshot(
    trigger: On<TakeScreenshot>,
    mut commands: Commands,
    screenshot_path: Res<ScreenshotPath>,
    mut counter: Local<u32>,
) {
    let file = format!("screenshot-{}.png", *counter);
    let path = Path::new(screenshot_path.0.as_str()).join(file);
    *counter += 1;
    let id = commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path))
        .id();
    if trigger.event().0 {
        commands.entity(id).observe(exit_on_capture);
    }
}

fn screenshot_saving(
    mut commands: Commands,
    screenshot_saving: Query<Entity, With<Capturing>>,
    window: Single<Entity, With<Window>>,
) {
    match screenshot_saving.iter().count() {
        0 => {
            commands.entity(*window).remove::<CursorIcon>();
        }
        x if x > 0 => {
            commands
                .entity(*window)
                .insert(CursorIcon::from(SystemCursorIcon::Progress));
        }
        _ => {}
    }
}
