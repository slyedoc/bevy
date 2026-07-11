//! Reflection-driven property editing with Feathers widgets.
//!
//! Three panels: a **world inspector** listing named entities (click one to inspect it in the
//! **detail** panel), and a **resource** inspector. Editing a slider, toggling a checkbox, switching
//! an enum variant, or adding/removing list elements writes straight back through reflection, so
//! `Changed<T>` fires — watch the console.

use core::any::TypeId;

use bevy::{
    feathers::{
        dark_theme::create_dark_theme,
        theme::{ThemeBackgroundColor, UiTheme},
        tokens, FeathersPlugins,
    },
    feathers_inspector::{
        BuildResourceInspector, BuildWorldInspector, FeathersInspectorPlugins, Hidden,
        InspectorDetailPanel, ReadOnly,
    },
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
        .add_plugins((DefaultPlugins, FeathersPlugins, FeathersInspectorPlugins))
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

    // Named entities the world inspector will list.
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

    let world_panel = commands.spawn(panel(16.0, 200.0)).id();
    let detail_panel = commands.spawn(panel(228.0, 360.0)).id();
    let resource_panel = commands.spawn(panel(600.0, 360.0)).id();

    // Where a selected entity's inspector is rendered.
    commands.insert_resource(InspectorDetailPanel(detail_panel));

    commands.queue(BuildWorldInspector { panel: world_panel });
    commands.queue(BuildResourceInspector {
        type_id: TypeId::of::<DemoConfig>(),
        panel: resource_panel,
    });
}

/// A themed inspector panel positioned absolutely.
fn panel(left: f32, width: f32) -> impl Bundle {
    (
        Node {
            position_type: PositionType::Absolute,
            left: px(left),
            top: px(16),
            width: px(width),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
            padding: UiRect::all(px(10)),
            ..default()
        },
        ThemeBackgroundColor(tokens::WINDOW_BG),
    )
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
