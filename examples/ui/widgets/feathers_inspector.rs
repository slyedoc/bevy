//! Reflection-driven property editing with Feathers widgets.
//!
//! Adds a turnkey [`WorldInspectorPlugin`]: press `` ` `` (backtick) to toggle an overlay listing
//! entities and resources in a collapsible tree — expand a row to edit its components/fields. Edits
//! write straight back through reflection, so `Changed<T>` fires — watch the console.
//!
//! The plugin is filtered with `With<Name>` here (only named entities); use
//! `WorldInspectorPlugin::new()` for every entity, or any other query filter.

use bevy::{
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    feathers_inspector::{FeathersInspectorPlugins, Hidden, ReadOnly, WorldInspectorPlugin},
    prelude::*,
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

/// A resource, to show resource inspection.
#[derive(Resource, Reflect, Debug, Default)]
#[reflect(Resource, Default)]
struct DemoConfig {
    #[reflect(@0.0..=1.0_f32)]
    volume: f32,
    muted: bool,
    mode: DemoMode,
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            FeathersPlugins,
            FeathersInspectorPlugins,
            WorldInspectorPlugin::<With<Name>>::default().with_toggle_key(KeyCode::Backquote),
        ))
        .insert_resource(UiTheme(create_dark_theme()))
        .register_type::<DemoSettings>()
        .register_type::<NestedSettings>()
        .register_type::<DemoMode>()
        .register_type::<DemoConfig>()
        .insert_resource(DemoConfig {
            volume: 0.8,
            muted: false,
            mode: DemoMode::Ramp { from: 0.0, to: 1.0 },
        })
        .add_systems(Startup, setup)
        .add_systems(Update, report_changes)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);

    commands.spawn((
        Name::new("Player"),
        DemoSettings {
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
        },
    ));
    commands.spawn((
        Name::new("Enemy"),
        DemoSettings {
            speed: 8.0,
            enabled: false,
            count: 1,
            mode: DemoMode::Off,
            ..default()
        },
    ));

    info!("Press ` (backtick) to toggle the world inspector.");
}

/// Prints components/resources whenever an inspector edit mutates them.
fn report_changes(
    changed: Query<(&Name, &DemoSettings), Changed<DemoSettings>>,
    config: Option<Res<DemoConfig>>,
) {
    for (name, settings) in &changed {
        info!("{name} changed: {settings:?}");
    }
    if let Some(config) = config
        && config.is_changed()
        && !config.is_added()
    {
        info!("DemoConfig changed: {:?}", *config);
    }
}
