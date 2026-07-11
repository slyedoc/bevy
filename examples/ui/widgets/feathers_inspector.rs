//! Reflection-driven property editing with Feathers widgets.
//!
//! Point the inspector at an entity and it builds a live editing UI for that entity's reflected
//! components out of `bevy_feathers` widgets. Editing a slider, toggling a checkbox, switching an
//! enum variant, or adding/removing list elements writes straight back into the component through
//! reflection, so `Changed<T>` fires — watch the console.

use bevy::{
    feathers::{
        dark_theme::create_dark_theme,
        theme::{ThemeBackgroundColor, UiTheme},
        tokens, FeathersPlugins,
    },
    feathers_inspector::{BuildEntityInspector, FeathersInspectorPlugins, Hidden, ReadOnly},
    prelude::*,
    ui::px,
};

/// A demo component exercising the inspector's widgets.
#[derive(Component, Reflect, Debug, Default)]
#[reflect(Component, Default)]
struct DemoSettings {
    /// Explicit slider range from a native reflect attribute.
    #[reflect(@0.0..=10.0_f32)]
    speed: f32,
    enabled: bool,
    gain: f32,
    count: i32,
    offset: Vec3,
    mode: DemoMode,
    tags: Vec<f32>,
    /// Shown, but not editable.
    #[reflect(@ReadOnly)]
    id: u32,
    /// Omitted from the inspector entirely.
    #[reflect(@Hidden)]
    internal: f32,
    nested: NestedSettings,
}

/// A nested struct, to show that the inspector recurses.
#[derive(Reflect, Debug, Default)]
#[reflect(Default)]
struct NestedSettings {
    scale: f32,
    active: bool,
}

/// An enum, to show the variant selector.
#[derive(Reflect, Debug, Default)]
#[reflect(Default)]
enum DemoMode {
    #[default]
    Off,
    Constant(f32),
    Ramp {
        from: f32,
        to: f32,
    },
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, FeathersPlugins, FeathersInspectorPlugins))
        .insert_resource(UiTheme(create_dark_theme()))
        .register_type::<DemoSettings>()
        .register_type::<NestedSettings>()
        .register_type::<DemoMode>()
        .add_systems(Startup, setup)
        .add_systems(Update, report_changes)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);

    // The panel that inspector sections are spawned under.
    let panel = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(16),
                top: px(16),
                width: px(360),
                flex_direction: FlexDirection::Column,
                row_gap: px(6),
                padding: UiRect::all(px(10)),
                ..default()
            },
            ThemeBackgroundColor(tokens::WINDOW_BG),
        ))
        .id();

    // The entity we want to edit.
    let target = commands
        .spawn(DemoSettings {
            speed: 5.0,
            enabled: true,
            gain: 0.75,
            count: 3,
            offset: Vec3::new(1.0, 2.0, 3.0),
            mode: DemoMode::Constant(4.0),
            tags: vec![1.0, 2.0],
            id: 42,
            internal: 99.0,
            nested: NestedSettings {
                scale: 2.0,
                active: false,
            },
        })
        .id();

    commands.queue(BuildEntityInspector { target, panel });
}

/// Prints the component whenever an inspector edit mutates it, proving writeback + change detection.
fn report_changes(changed: Query<&DemoSettings, Changed<DemoSettings>>) {
    for settings in &changed {
        info!("DemoSettings changed: {settings:?}");
    }
}
