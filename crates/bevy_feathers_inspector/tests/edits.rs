//! The inspector's widgets edit what they are bound to, and an app's [`InspectorEdits`] handler
//! sees every edit, previews and commits apart.

use core::any::TypeId;

use bevy_app::App;
use bevy_asset::AssetApp;
use bevy_ecs::prelude::*;
use bevy_feathers::FeathersPlugins;
use bevy_feathers_inspector::{
    apply_edit, build_body, number::EulerAxis, recurse::parse_path,
    FeathersInspectorPlugins, InspectorBinding, InspectorEdits, InspectorRoot,
};
use bevy_color::Color;
use bevy_feathers::controls::ColorPlaneValue;
use bevy_math::{EulerRot, Quat, Vec2, Vec3};
use bevy_reflect::{prelude::ReflectDefault, ParsedPath, PartialReflect, Reflect};
use bevy_ui_widgets::{NumericValue, ValueChange};

#[derive(Component, Reflect, Default, Debug)]
#[reflect(Component, Default)]
struct Probe {
    speed: f32,
    count: u32,
    offset: Vec3,
    turn: Quat,
    name: String,
    steps: Vec<f32>,
    tint: Color,
    width: bevy_ui::Val,
    margin: bevy_ui::UiRect,
}

fn app() -> App {
    let mut app = App::new();
    app.add_plugins((
        bevy_app::TaskPoolPlugin::default(),
        bevy_time::TimePlugin,
        bevy_asset::AssetPlugin::default(),
        bevy_window::WindowPlugin::default(),
        bevy_input::InputPlugin,
        bevy_picking::PickingPlugin,
        bevy_picking::InteractionPlugin,
        bevy_scene::ScenePlugin,
        bevy_text::TextPlugin,
        bevy_ui::UiPlugin,
        bevy_input_focus::InputFocusPlugin,
        bevy_input_focus::InputDispatchPlugin,
        FeathersPlugins,
        FeathersInspectorPlugins,
    ));
    app.init_asset::<bevy_image::Image>();
    app.init_asset::<bevy_image::TextureAtlasLayout>();
    app.register_type::<Probe>();
    app.finish();
    app.cleanup();
    app
}

/// A `Probe` with its fields built under a panel.
fn inspect(app: &mut App) -> Entity {
    let target = app
        .world_mut()
        .spawn(Probe {
            speed: 1.0,
            count: 2,
            turn: Quat::from_rotation_y(0.5),
            name: "a".into(),
            steps: vec![1.0, 2.0, 3.0],
            ..Default::default()
        })
        .id();
    let panel = app.world_mut().spawn(bevy_ui::Node::default()).id();
    let root = InspectorRoot::Component {
        entity: target,
        type_id: TypeId::of::<Probe>(),
    };
    build_body(app.world_mut(), root, panel);
    app.update();
    target
}

/// The widget bound to `path`: its number input when it has one (a `Val` row also binds its unit
/// select), else the first.
fn widget(app: &mut App, path: &str) -> Entity {
    let want = parse_path(path);
    let mut query = app
        .world_mut()
        .query::<(Entity, &InspectorBinding, Has<NumericValue>)>();
    let mut bound: Vec<(Entity, bool)> = query
        .iter(app.world())
        .filter(|(_, binding, _)| binding.path == want)
        .map(|(entity, _, number)| (entity, number))
        .collect();
    bound.sort_by_key(|(_, number)| !number);
    bound
        .first()
        .map(|(entity, _)| *entity)
        .unwrap_or_else(|| panic!("no widget bound to {path}"))
}

fn change<T: Send + Sync + 'static>(app: &mut App, source: Entity, value: T, is_final: bool) {
    app.world_mut().trigger(ValueChange {
        source,
        value,
        is_final,
    });
    app.update();
}

fn probe(app: &App, target: Entity) -> &Probe {
    app.world().get::<Probe>(target).unwrap()
}

#[test]
fn a_number_input_writes_its_field_and_shows_the_value() {
    let mut app = app();
    let target = inspect(&mut app);
    let speed = widget(&mut app, ".speed");
    change(&mut app, speed, 4.5f32, true);
    assert_eq!(probe(&app, target).speed, 4.5);
    assert_eq!(
        app.world().get::<NumericValue>(speed),
        Some(&NumericValue::F32(4.5))
    );
}

#[test]
fn an_unsigned_field_never_goes_below_zero() {
    let mut app = app();
    let target = inspect(&mut app);
    let count = widget(&mut app, ".count");
    change(&mut app, count, -3i64, true);
    assert_eq!(probe(&app, target).count, 0);
    change(&mut app, count, 7i64, true);
    assert_eq!(probe(&app, target).count, 7);
}

#[test]
fn a_vector_axis_edits_its_own_component() {
    let mut app = app();
    let target = inspect(&mut app);
    let y = widget(&mut app, ".offset.y");
    change(&mut app, y, 2.0f32, true);
    assert_eq!(probe(&app, target).offset, Vec3::new(0.0, 2.0, 0.0));
}

#[test]
fn an_euler_angle_rebuilds_the_rotation() {
    let mut app = app();
    let target = inspect(&mut app);
    let mut query = app
        .world_mut()
        .query::<(Entity, &InspectorBinding, &EulerAxis)>();
    let x = query
        .iter(app.world())
        .find(|(_, binding, axis)| binding.path == parse_path(".turn") && axis.0 == 0)
        .map(|(entity, ..)| entity)
        .expect("the rotation has an x input");
    change(&mut app, x, 90.0f32, true);
    let (ex, ey, _) = probe(&app, target).turn.to_euler(EulerRot::XYZ);
    assert!((ex.to_degrees() - 90.0).abs() < 1e-3, "x became 90: {ex}");
    assert!((ey - 0.5).abs() < 1e-3, "y kept: {ey}");
}

#[test]
fn a_string_commits_its_text() {
    let mut app = app();
    let target = inspect(&mut app);
    let name = widget(&mut app, ".name");
    change(&mut app, name, String::from("renamed"), true);
    assert_eq!(probe(&app, target).name, "renamed");
}

#[test]
fn a_shorter_list_replaces_the_longer_one() {
    let mut app = app();
    let target = inspect(&mut app);
    let root = InspectorRoot::Component {
        entity: target,
        type_id: TypeId::of::<Probe>(),
    };
    apply_edit(
        app.world_mut(),
        &root,
        &parse_path(".steps"),
        &vec![9.0f32],
        true,
    );
    assert_eq!(probe(&app, target).steps, vec![9.0]);
}

/// Every edit the handler saw: the path and whether it was final.
#[derive(Resource, Default)]
struct Seen(Vec<(String, bool)>);

fn record(
    world: &mut World,
    _root: &InspectorRoot,
    path: &ParsedPath,
    _value: &dyn PartialReflect,
    is_final: bool,
) -> bool {
    world
        .resource_mut::<Seen>()
        .0
        .push((path.to_string(), is_final));
    true
}

#[test]
fn an_edit_handler_takes_previews_and_commits() {
    let mut app = app();
    app.init_resource::<Seen>();
    app.insert_resource(InspectorEdits(record));
    let target = inspect(&mut app);
    let speed = widget(&mut app, ".speed");
    change(&mut app, speed, 2.0f32, false);
    change(&mut app, speed, 3.0f32, true);
    assert_eq!(
        app.world().resource::<Seen>().0,
        vec![(".speed".to_string(), false), (".speed".to_string(), true)]
    );
    assert_eq!(probe(&app, target).speed, 1.0, "the handler took the edit");
}

#[test]
fn the_color_plane_writes_red_and_blue() {
    let mut app = app();
    let target = inspect(&mut app);
    let mut planes = app
        .world_mut()
        .query_filtered::<Entity, With<ColorPlaneValue>>();
    let plane = planes.single(app.world()).expect("the tint has a plane");
    change(&mut app, plane, Vec2::new(0.25, 0.75), true);
    let tint = probe(&app, target).tint.to_srgba();
    assert!(
        (tint.red - 0.25).abs() < 1e-5 && (tint.blue - 0.75).abs() < 1e-5,
        "{tint:?}"
    );
    assert!(
        matches!(probe(&app, target).tint, Color::LinearRgba(_)),
        "the field keeps its variant"
    );
}

/// A host that rebuilds its panel right after asking for fields (an editor's inspector on a
/// selection change) must not leave the fields behind as UI roots drawn across the window.
#[test]
fn fields_go_with_a_panel_despawned_right_after_building() {
    let mut app = app();
    let target = app.world_mut().spawn(Probe::default()).id();
    let panel = app.world_mut().spawn(bevy_ui::Node::default()).id();
    let root = InspectorRoot::Component {
        entity: target,
        type_id: TypeId::of::<Probe>(),
    };
    build_body(app.world_mut(), root, panel);
    app.world_mut().entity_mut(panel).despawn();
    app.update();
    let mut bound = app.world_mut().query::<&InspectorBinding>();
    assert_eq!(bound.iter(app.world()).count(), 0, "no field outlived its panel");
}

/// A field changed by something other than the inspector (game code, an undo) shows up in its
/// widget once the component's change tick says so.
#[test]
fn a_widget_follows_its_field_changed_elsewhere() {
    let mut app = app();
    let target = inspect(&mut app);
    let speed = widget(&mut app, ".speed");
    app.world_mut().get_mut::<Probe>(target).unwrap().speed = 7.5;
    app.update();
    assert_eq!(
        app.world().get::<NumericValue>(speed),
        Some(&NumericValue::F32(7.5))
    );
}

#[test]
fn a_val_number_keeps_its_unit() {
    let mut app = app();
    let target = inspect(&mut app);
    app.world_mut().get_mut::<Probe>(target).unwrap().width = bevy_ui::Val::Percent(10.0);
    app.update();
    let width = widget(&mut app, ".width");
    change(&mut app, width, 25.0f32, true);
    assert_eq!(probe(&app, target).width, bevy_ui::Val::Percent(25.0));
}

#[test]
fn a_ui_rect_gets_a_val_row_per_side() {
    let mut app = app();
    let target = inspect(&mut app);
    let left = widget(&mut app, ".margin.left");
    change(&mut app, left, 4.0f32, true);
    assert_eq!(probe(&app, target).margin.left, bevy_ui::Val::Px(4.0));
}
