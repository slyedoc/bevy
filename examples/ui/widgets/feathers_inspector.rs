//! Reflection-driven property editing with Feathers widgets.
//!
//! Point the inspector at an entity and it builds a live editing UI for that entity's reflected
//! components out of `bevy_feathers` widgets. Editing a number input or toggling a checkbox writes
//! straight back into the component through reflection, so `Changed<T>` fires — watch the console.

use bevy::{
    feathers::{
        dark_theme::create_dark_theme,
        theme::{ThemeBackgroundColor, UiTheme},
        tokens, FeathersPlugins,
    },
    feathers_inspector::{BuildEntityInspector, FeathersInspectorPlugins},
    prelude::*,
    ui::px,
};

/// A demo component with a mix of leaf types and a nested struct.
#[derive(Component, Reflect, Debug, Default)]
#[reflect(Component, Default)]
struct DemoSettings {
    speed: f32,
    enabled: bool,
    gain: f32,
    nested: NestedSettings,
}

/// A nested struct, to show that the inspector recurses.
#[derive(Reflect, Debug, Default)]
#[reflect(Default)]
struct NestedSettings {
    scale: f32,
    active: bool,
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, FeathersPlugins, FeathersInspectorPlugins))
        .insert_resource(UiTheme(create_dark_theme()))
        .register_type::<DemoSettings>()
        .register_type::<NestedSettings>()
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
                width: px(340),
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
            nested: NestedSettings {
                scale: 2.0,
                active: false,
            },
        })
        .id();

    // Build the inspector for `target` under `panel`.
    commands.queue(BuildEntityInspector { target, panel });
}

/// Prints the component whenever an inspector edit mutates it, proving writeback + change detection.
fn report_changes(changed: Query<&DemoSettings, Changed<DemoSettings>>) {
    for settings in &changed {
        info!("DemoSettings changed: {settings:?}");
    }
}
