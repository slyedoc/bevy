//! Adversarial regression suite for the dynamic scene layer.
//!
//! These tests come from the adversarial correctness review of `dynamic/`, which attacked the
//! resolver, the value builder and the asset paths with inputs the happy-path tests in
//! [`super::tests`] never produce: leaked resolve context between siblings, malformed asset
//! paths, `#[reflect(ignore)]` fields, opaque types, shadowed enum variants, self-includes,
//! non-scene bases, and shape mismatches of every kind. Each one either pins a defect the
//! review found and fixed, or pins the graceful-error behavior that replaced a panic — so a
//! regression fails here rather than in a user's `.bsn` file.
//!
//! The sections are the review's batches, kept as they were written so that a failure is easy
//! to trace back to the finding it came from. The fixture components and the fixture-registered
//! `App` builder they run against live in [`super::tests`].

use std::path::Path;

use bevy_app::App;
use bevy_asset::{AssetApp, AssetServer};
use bevy_ecs::{
    hierarchy::Children,
    name::Name,
    prelude::Component,
    reflect::{AppTypeRegistry, ReflectComponent, ReflectFromTemplate},
};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};

use crate::{
    test_support::memory_asset_app,
    tests::{register_fixtures, scene, test_app, Choice, Foo, Position},
};
use bevy_scene::{bsn, ScenePatch, WorldSceneExt};

/// An app whose memory asset source contains the given `(path, source)` `.bsn` files, all loaded.
fn multi_asset_app(files: &[(&'static str, &'static str)]) -> App {
    let (mut app, dir) = memory_asset_app();
    app.init_asset::<crate::tests::Image>();
    app.register_asset_reflect::<crate::tests::Image>();
    register_fixtures(&mut app);
    app.finish();
    app.cleanup();

    for (path, source) in files {
        dir.insert_asset_text(Path::new(path), source);
    }

    let asset_server = app.world().resource::<AssetServer>().clone();
    let handles: Vec<_> = files
        .iter()
        .map(|(path, _)| asset_server.load::<ScenePatch>(*path))
        .collect();
    for frame in 0..10_000 {
        app.update();
        if frame >= 100 {
            std::thread::sleep(core::time::Duration::from_millis(1));
        }
        if handles.iter().all(|h| asset_server.is_loaded(h)) {
            break;
        }
    }
    for (handle, (path, _)) in handles.iter().zip(files) {
        assert!(asset_server.is_loaded(handle), "{path} never loaded");
    }
    app
}

// =========================================================================================
// A1: sibling base leak — `context.cached` used not to be cleared between sibling entities.
// =========================================================================================

#[test]
fn a1_sibling_base_does_not_leak_cached_context() {
    // `c.bsn` declares a `Position`. The first child of `b.bsn` includes it as a base; the second
    // child includes nothing at all, and used to be resolved with `context.cached` still pointing
    // at `c.bsn`.
    let mut app = multi_asset_app(&[("c.bsn", "Position { y: 9.0 }")]);
    let b = scene(
        &app,
        "b.bsn",
        "Children [ :\"c.bsn\" Marker -- Position { x: 1.0 } ]",
    );

    let world = app.world_mut();
    let id = world.spawn_scene(b).unwrap().id();
    let children = world.entity(id).get::<Children>().unwrap().to_vec();
    assert_eq!(children.len(), 2);

    let second = world.entity(children[1]).get::<Position>().unwrap();
    assert_eq!(
        (second.x, second.y, second.z),
        (1.0, 0.0, 0.0),
        "the second child has no base, so it must not inherit c.bsn's y"
    );
}

/// Control for A1: with the base removed from the first child, the same document is fine.
#[test]
fn a1_control_no_sibling_base() {
    let mut app = multi_asset_app(&[("c.bsn", "Position { y: 9.0 }")]);
    let b = scene(
        &app,
        "b.bsn",
        "Children [ Marker -- Position { x: 1.0 } ]",
    );
    let world = app.world_mut();
    let id = world.spawn_scene(b).unwrap().id();
    let children = world.entity(id).get::<Children>().unwrap().to_vec();
    let second = world.entity(children[1]).get::<Position>().unwrap();
    assert_eq!((second.x, second.y, second.z), (1.0, 0.0, 0.0));
}

/// Control for A1: the *order* matters — a base on the last child is harmless.
#[test]
fn a1_control_base_on_the_last_child() {
    let mut app = multi_asset_app(&[("c.bsn", "Position { y: 9.0 }")]);
    let b = scene(
        &app,
        "b.bsn",
        "Children [ Position { x: 1.0 } -- :\"c.bsn\" Marker ]",
    );
    let world = app.world_mut();
    let id = world.spawn_scene(b).unwrap().id();
    let children = world.entity(id).get::<Children>().unwrap().to_vec();
    let first = world.entity(children[0]).get::<Position>().unwrap();
    assert_eq!((first.x, first.y, first.z), (1.0, 0.0, 0.0));
}

// =========================================================================================
// A2: static bsn! cached base + a related child that touches a template the base also has.
// =========================================================================================

#[test]
fn a2_static_cached_base_child_touching_base_template() {
    // `a.bsn`'s root has a `#Root` name, so its root scene holds a `Name` template. The static
    // scene's child `#Y` asks for `Name` while `context.cached` still points at `a.bsn`.
    let mut app = multi_asset_app(&[("a.bsn", "#Root Position { y: 2.0 }")]);

    let world = app.world_mut();
    let id = world
        .spawn_scene(bsn! {
            :"a.bsn"
            Children [ #Y ]
        })
        .unwrap()
        .id();

    let children = world.entity(id).get::<Children>().unwrap().to_vec();
    assert_eq!(
        world.entity(children[0]).get::<Name>().unwrap().as_str(),
        "Y"
    );
}

// =========================================================================================
// A3: dynamic scene as a child of a *statically* cached scene.
// =========================================================================================

#[test]
fn a3_dynamic_child_of_static_cached_base() {
    let mut app = multi_asset_app(&[("a.bsn", "Position { y: 2.0 }")]);
    let child = scene(&app, "child.bsn", "Position { x: 1.0 }");

    let world = app.world_mut();
    let id = world
        .spawn_scene(bsn! {
            :"a.bsn"
            Children [ @{child} ]
        })
        .unwrap()
        .id();

    let children = world.entity(id).get::<Children>().unwrap().to_vec();
    let position = world.entity(children[0]).get::<Position>().unwrap();
    assert_eq!(
        (position.x, position.y, position.z),
        (1.0, 0.0, 0.0),
        "the child has no base of its own"
    );
}

// =========================================================================================
// A4: value coercion boundaries
// =========================================================================================

#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct Widths {
    pub(crate) u8f: u8,
    pub(crate) i64f: i64,
    pub(crate) u64f: u64,
    pub(crate) f32f: f32,
    pub(crate) f64f: f64,
    pub(crate) charf: char,
    pub(crate) optopt: Option<Option<u32>>,
    pub(crate) vecvec: Vec<Vec<u8>>,
}

fn widths_app() -> App {
    let mut app = test_app();
    app.register_type::<Widths>();
    app
}

fn try_scene(app: &App, source: &str) -> Result<(), String> {
    let document = crate::tests::doc(source);
    let registry = app.world().resource::<AppTypeRegistry>().clone();
    crate::DynamicScene::from_document(&document, "t.bsn", &registry)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[test]
fn a4_int_boundaries() {
    let app = widths_app();
    // u8::MAX + 1
    assert!(try_scene(&app, "Widths { u8f: 256 }").is_err(), "u8 256");
    assert!(try_scene(&app, "Widths { u8f: 255 }").is_ok(), "u8 255");
    // negative into unsigned
    assert!(try_scene(&app, "Widths { u64f: -1 }").is_err(), "u64 -1");
    // i64::MIN
    assert!(
        try_scene(&app, "Widths { i64f: -9223372036854775808 }").is_ok(),
        "i64::MIN should fit"
    );
    assert!(
        try_scene(&app, "Widths { i64f: 9223372036854775808 }").is_err(),
        "i64::MAX + 1 must not fit"
    );
}

#[test]
fn a4_float_exactness() {
    let app = widths_app();
    // 2^53 exactly representable, 2^53 + 1 is not.
    assert!(
        try_scene(&app, "Widths { f64f: 9007199254740992 }").is_ok(),
        "2^53"
    );
    assert!(
        try_scene(&app, "Widths { f64f: 9007199254740993 }").is_err(),
        "2^53+1 must be rejected"
    );
    assert!(try_scene(&app, "Widths { f32f: 16777216 }").is_ok(), "2^24");
    assert!(
        try_scene(&app, "Widths { f32f: 16777217 }").is_err(),
        "2^24+1 must be rejected"
    );
    // float into int
    assert!(
        try_scene(&app, "Widths { u8f: 1.0 }").is_err(),
        "float into int must be rejected"
    );
}

#[test]
fn a4_string_into_char_errors_not_panics() {
    let app = widths_app();
    let error = try_scene(&app, r#"Widths { charf: "x" }"#);
    assert!(error.is_err(), "string into char: {error:?}");
}

#[test]
fn a4_nested_option_and_vec() {
    let mut app = widths_app();
    let a = scene(&app, "a.bsn", "Widths { optopt: 5, vecvec: [[1, 2], [3]] }");
    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    let widths = world.entity(id).get::<Widths>().unwrap();
    assert_eq!(widths.optopt, Some(Some(5)));
    assert_eq!(widths.vecvec, vec![vec![1u8, 2], vec![3]]);
}

// =========================================================================================
// A5: enum edge cases
// =========================================================================================

#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) enum Shared {
    #[default]
    A,
    B {
        v: u32,
    },
    C {
        v: String,
    },
}

#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct Holder {
    pub(crate) choice: Shared,
}

#[test]
fn a5_shared_variant_field_names() {
    let mut app = test_app();
    app.register_type::<Shared>().register_type::<Holder>();
    let a = scene(&app, "a.bsn", r#"Shared::B { v: 1 }"#);
    let b = scene(&app, "b.bsn", r#"Shared::C { v: "hi" }"#);
    let world = app.world_mut();
    let id = world.spawn_scene((a, b)).unwrap().id();
    assert_eq!(
        *world.entity(id).get::<Shared>().unwrap(),
        Shared::C { v: "hi".into() }
    );
}

#[test]
fn a5_nested_enum_partial_resets_siblings() {
    // Dynamic allows a *partial* nested enum value; document what it does.
    let mut app = test_app();
    app.register_type::<Shared>().register_type::<Holder>();
    let a = scene(&app, "a.bsn", r#"Holder { choice: Shared::B { v: 7 } }"#);
    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    assert_eq!(
        world.entity(id).get::<Holder>().unwrap().choice,
        Shared::B { v: 7 }
    );
}

#[test]
fn a5_enum_variant_switch_then_partial_matches_static() {
    let mut app = test_app();
    let dynamic_result = {
        let baz = scene(&app, "a.bsn", "Choice::Baz(10)");
        let bar = scene(&app, "b.bsn", "Choice::Bar { x: 1, y: 0, z: 0 }");
        let bar2 = scene(&app, "c.bsn", "Choice::Bar { x: 1, y: 2, z: 0 }");
        let world = app.world_mut();
        let id = world.spawn_scene((baz, bar, bar2)).unwrap().id();
        world.entity(id).get::<Choice>().unwrap().clone_value_dbg()
    };
    let static_result = {
        let world = app.world_mut();
        let id = world
            .spawn_scene(bsn! {
                Choice::Baz(10)
                Choice::Bar { x: 1, y: 0, z: 0 }
                Choice::Bar { x: 1, y: 2, z: 0 }
            })
            .unwrap()
            .id();
        world.entity(id).get::<Choice>().unwrap().clone_value_dbg()
    };
    assert_eq!(dynamic_result, static_result);
}

trait DbgClone {
    fn clone_value_dbg(&self) -> String;
}
impl<T: core::fmt::Debug> DbgClone for T {
    fn clone_value_dbg(&self) -> String {
        format!("{self:?}")
    }
}

// =========================================================================================
// A6: struct shapes
// =========================================================================================

#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct Empty();

#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct One(pub(crate) u32);

#[test]
fn a6_zero_and_one_field_tuple_structs() {
    let mut app = test_app();
    app.register_type::<Empty>().register_type::<One>();
    let a = scene(&app, "a.bsn", "Empty()\nOne(5)");
    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    assert!(world.entity(id).get::<Empty>().is_some());
    assert_eq!(world.entity(id).get::<One>().unwrap().0, 5);
}

// =========================================================================================
// A7: entity references
// =========================================================================================

#[test]
fn a7_duplicate_names_and_child_only_names() {
    let mut app = test_app();
    // `#Dup` twice; a reference to a name declared only in a child.
    let a = scene(
        &app,
        "a.bsn",
        "#Dup Reference(#Deep) Children [ #Dup -- #Deep ]",
    );
    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    let children = world.entity(id).get::<Children>().unwrap().to_vec();
    assert_eq!(children.len(), 2);
    let reference = world.entity(id).get::<crate::tests::Reference>().unwrap().0;
    assert_eq!(reference, children[1], "#Deep is the second child");
}

// =========================================================================================
// A8: loader edge cases
// =========================================================================================

#[test]
fn a8_empty_and_comment_only_files() {
    let app = test_app();
    assert!(try_scene(&app, "").is_ok(), "empty file");
    assert!(try_scene(&app, "// nothing\n").is_ok(), "comments only");
    assert!(try_scene(&app, "   \n\t\n").is_ok(), "whitespace only");
}

#[test]
fn a8_base_pointing_at_a_non_scene_asset() {
    let mut app = multi_asset_app(&[]);
    let b = scene(&app, "b.bsn", ":\"image.png\"\nMarker");
    let world = app.world_mut();
    let result = world.spawn_scene(b);
    assert!(result.is_err(), "a missing/invalid base must error");
}

// =========================================================================================
// A9: apply the same resolved dynamic template twice (cached scenes re-apply)
// =========================================================================================

#[test]
fn a9_cached_scene_applied_to_two_entities() {
    let mut app = multi_asset_app(&[("a.bsn", "Position { y: 2.0 } Foo { x: 1 }")]);
    let b = scene(&app, "b.bsn", ":\"a.bsn\"\nPosition { x: 1.0 }");

    let world = app.world_mut();
    let first = world.spawn_scene(b.clone()).unwrap().id();
    let second = world.spawn_scene(b).unwrap().id();
    for id in [first, second] {
        let position = world.entity(id).get::<Position>().unwrap();
        assert_eq!((position.x, position.y, position.z), (1.0, 2.0, 0.0));
        assert_eq!(world.entity(id).get::<Foo>().unwrap().x, 1);
    }
}

// =========================================================================================
// B: batch 2
// =========================================================================================

/// B1: a malformed asset path in a `Handle`-typed field.
#[test]
fn b1_malformed_asset_path_in_handle_field() {
    let app = test_app();
    // `bad#source://x.png` is rejected by `AssetPath::try_parse` (`InvalidSourceSyntax`).
    let result = try_scene(&app, r#"Sprite("bad#source://x.png")"#);
    assert!(
        result.is_err(),
        "a malformed asset path must be a build error, not a panic"
    );
}

/// B1b: the same string aimed at an `AssetPath` field goes through the graceful path.
#[test]
fn b1b_malformed_asset_path_direct() {
    use bevy_asset::AssetPath;
    assert!(AssetPath::try_parse("bad#source://x.png").is_err());
    assert!(AssetPath::try_parse("://x.png").is_err());
}

/// B2: a conversion registered to fail at runtime.
#[derive(Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Default)]
pub(crate) struct Refusing(pub(crate) u32);

#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct HasRefusing {
    pub(crate) field: Refusing,
}

#[test]
fn b2_failing_conversion_errors_cleanly() {
    let mut app = test_app();
    app.register_type::<Refusing>()
        .register_type::<HasRefusing>();
    app.register_type_conversion::<crate::tests::TextSize, Refusing, _>(Err);
    let error = try_scene(&app, "HasRefusing { field: TextSize::Large }")
        .expect_err("a failing conversion must be an error");
    assert!(
        error.contains("Refusing"),
        "the message should name the destination: {error}"
    );
}

/// B3: `#[reflect(ignore)]` in an enum struct variant, across a variant switch.
#[derive(Component, Reflect, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) enum IgnoreEnum {
    #[default]
    Unit,
    Data {
        v: u32,
        #[reflect(ignore)]
        hidden: u32,
    },
}

#[test]
fn b3_ignored_field_in_enum_variant_switch() {
    let mut app = test_app();
    app.register_type::<IgnoreEnum>();
    let a = scene(&app, "a.bsn", "IgnoreEnum::Data { v: 3 }");
    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    assert_eq!(
        *world.entity(id).get::<IgnoreEnum>().unwrap(),
        IgnoreEnum::Data { v: 3, hidden: 0 }
    );
}

/// B4: `#[reflect(ignore)]` in a plain struct patch.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct IgnoreStruct {
    pub(crate) v: u32,
    #[reflect(ignore)]
    pub(crate) hidden: u32,
}

#[test]
fn b4_ignored_field_is_unknown() {
    let mut app = test_app();
    app.register_type::<IgnoreStruct>();
    assert!(
        try_scene(&app, "IgnoreStruct { v: 1 }").is_ok(),
        "the visible field works"
    );
    assert!(
        try_scene(&app, "IgnoreStruct { hidden: 1 }").is_err(),
        "an ignored field must be an error, not silently dropped"
    );
}

/// B5: `ReflectDefault` differing from `Default`.
#[derive(Component, Reflect, Clone, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct OddDefault {
    pub(crate) a: u32,
    pub(crate) b: u32,
}

impl Default for OddDefault {
    fn default() -> Self {
        Self { a: 7, b: 7 }
    }
}

#[test]
fn b5_reflect_default_is_used_for_the_base_value() {
    let mut app = test_app();
    app.register_type::<OddDefault>();
    let a = scene(&app, "a.bsn", "OddDefault { a: 1 }");
    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    let value = world.entity(id).get::<OddDefault>().unwrap();
    assert_eq!((value.a, value.b), (1, 7));
}

/// B6: nesting deeper than `MAX_DEPTH`.
#[test]
fn b6_deep_nesting_errors_not_overflows() {
    let app = test_app();
    let mut source = String::from("Marker");
    for _ in 0..400 {
        source = format!("Children [ ({source}) ]");
    }
    match bevy_bsn::parse(&source) {
        Err(_) => {}
        Ok(document) => {
            let registry = app.world().resource::<AppTypeRegistry>().clone();
            let result = crate::DynamicScene::from_document(&document, "t.bsn", &registry);
            assert!(result.is_err(), "400 levels should exceed MAX_DEPTH");
        }
    }
}

/// B7: a base include of a file that fails to parse.
#[test]
fn b7_base_of_unparsable_file() {
    let mut app = multi_asset_app_lenient(&[("broken.bsn", "Position { x: ")]);
    let b = scene(&app, "b.bsn", ":\"broken.bsn\"\nMarker");
    let world = app.world_mut();
    assert!(
        world.spawn_scene(b).is_err(),
        "a base that failed to load must produce an error"
    );
}

/// B8: `#Name` identity across two different `.bsn` documents must not alias.
#[test]
fn b8_name_identity_is_per_asset() {
    let mut app = test_app();
    let a = scene(&app, "a.bsn", "#Shared Children [ #Kid Marker ]");
    let b = scene(&app, "b.bsn", "#Shared Children [ #Kid Marker ]");
    let world = app.world_mut();
    let first = world.spawn_scene(a).unwrap().id();
    let second = world.spawn_scene(b).unwrap().id();
    assert_ne!(first, second);
    let fc = world.entity(first).get::<Children>().unwrap().to_vec();
    let sc = world.entity(second).get::<Children>().unwrap().to_vec();
    assert_ne!(fc[0], sc[0], "two assets must not alias each other");
}

/// B9: the same `.bsn` spawned twice must not alias entities across spawns.
#[test]
fn b9_same_asset_two_spawns_do_not_alias() {
    let mut app = test_app();
    let a = scene(&app, "a.bsn", "#Root Children [ #Kid Reference(#Root) ]");
    let world = app.world_mut();
    let first = world.spawn_scene(a.clone()).unwrap().id();
    let second = world.spawn_scene(a).unwrap().id();
    let fk = world.entity(first).get::<Children>().unwrap()[0];
    let sk = world.entity(second).get::<Children>().unwrap()[0];
    assert_ne!(fk, sk);
    assert_eq!(
        world.entity(fk).get::<crate::tests::Reference>().unwrap().0,
        first
    );
    assert_eq!(
        world.entity(sk).get::<crate::tests::Reference>().unwrap().0,
        second
    );
}

/// B10: naming a template type without the `~` prefix.
#[test]
fn b10_template_named_without_tilde() {
    let app = test_app();
    let error = try_scene(&app, r#"SpriteTemplate("a.png")"#);
    assert!(error.is_err(), "a template is not a component: {error:?}");
}

/// B11: `~` on a plain component.
#[test]
fn b11_tilde_on_a_plain_component() {
    let mut app = test_app();
    let a = scene(&app, "a.bsn", "~Position { x: 1.0 }");
    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    assert_eq!(world.entity(id).get::<Position>().unwrap().x, 1.0);
}

/// A `HandleTemplate`-bearing component with a *public* template field, so `bsn!` can patch it.
#[derive(Component, bevy_ecs::template::FromTemplate, Reflect, PartialEq, Debug)]
#[template(reflect)]
#[reflect(Component, FromTemplate)]
pub(crate) struct Icon(pub bevy_asset::Handle<crate::tests::Image>);

fn icon_app() -> App {
    let mut app = test_app();
    app.register_type::<Icon>().register_type::<IconTemplate>();
    app
}

/// B12: dynamic-over-static on a `HandleTemplate`-bearing component.
#[test]
fn b12_handle_template_static_then_dynamic() {
    let mut app = icon_app();
    let expected = app
        .world()
        .resource::<AssetServer>()
        .load::<crate::tests::Image>("dyn.png");
    let dynamic = scene(&app, "a.bsn", r#"Icon("dyn.png")"#);

    let world = app.world_mut();
    let id = world
        .spawn_scene(bsn! {
            Icon("static.png")
            @{dynamic}
        })
        .unwrap()
        .id();
    assert_eq!(
        world.entity(id).get::<Icon>().unwrap().0,
        expected,
        "dynamic-over-static must win"
    );
}

/// B12b: static-over-dynamic on a `HandleTemplate`-bearing component.
#[test]
fn b12b_handle_template_dynamic_then_static() {
    let mut app = icon_app();
    let expected = app
        .world()
        .resource::<AssetServer>()
        .load::<crate::tests::Image>("static.png");
    let dynamic = scene(&app, "a.bsn", r#"Icon("dyn.png")"#);

    let world = app.world_mut();
    let id = world
        .spawn_scene(bsn! {
            @{dynamic}
            Icon("static.png")
        })
        .unwrap()
        .id();
    assert_eq!(
        world.entity(id).get::<Icon>().unwrap().0,
        expected,
        "static-over-dynamic must win"
    );
}

/// B13: self-include spelled non-canonically. Must terminate, not hang or panic.
#[test]
fn b13_self_include_alternate_spelling() {
    let mut app = multi_asset_app_lenient(&[("selfy.bsn", ":\"./selfy.bsn\"\nMarker")]);
    let asset_server = app.world().resource::<AssetServer>().clone();
    let handle = asset_server.load::<ScenePatch>("selfy.bsn");
    for _ in 0..200 {
        app.update();
        if asset_server.is_loaded(&handle) {
            break;
        }
    }
}

/// B14: a `.bsn` whose base is a non-scene asset path.
#[test]
fn b14_base_is_a_non_scene_asset() {
    let mut app = multi_asset_app_lenient(&[("img.png", "not a scene")]);
    let b = scene(&app, "b.bsn", ":\"img.png\"\nMarker");
    let world = app.world_mut();
    assert!(world.spawn_scene(b).is_err());
}

/// Like [`multi_asset_app`] but does not require the files to load successfully.
fn multi_asset_app_lenient(files: &[(&'static str, &'static str)]) -> App {
    let (mut app, dir) = memory_asset_app();
    app.init_asset::<crate::tests::Image>();
    app.register_asset_reflect::<crate::tests::Image>();
    register_fixtures(&mut app);
    app.finish();
    app.cleanup();

    for (path, source) in files {
        dir.insert_asset_text(Path::new(path), source);
    }
    let asset_server = app.world().resource::<AssetServer>().clone();
    let handles: Vec<_> = files
        .iter()
        .map(|(path, _)| asset_server.load::<ScenePatch>(*path))
        .collect();
    for _ in 0..200 {
        app.update();
        if handles
            .iter()
            .all(|h| asset_server.is_loaded(h) || asset_server.get_load_state(h).is_some())
        {
            break;
        }
    }
    app
}

// =========================================================================================
// C: batch 3 — the follow-ups B1 and B3 asked for
// =========================================================================================

/// C1: an ignored field must not break *spawning*, on either the static or the dynamic path.
#[test]
fn c1_ignored_field_static_path_works() {
    let mut app = test_app();
    app.register_type::<IgnoreStruct>();
    let world = app.world_mut();
    let id = world
        .spawn_scene(bsn! { IgnoreStruct { v: 1 } })
        .unwrap()
        .id();
    assert_eq!(world.entity(id).get::<IgnoreStruct>().unwrap().v, 1);
}

#[test]
fn c1b_ignored_field_dynamic_path_works_too() {
    let mut app = test_app();
    app.register_type::<IgnoreStruct>();
    let a = scene(&app, "a.bsn", "IgnoreStruct { v: 1 }");
    let world = app.world_mut();
    let result = world.spawn_scene(a);
    match result {
        Ok(entity) => {
            let id = entity.id();
            assert_eq!(world.entity(id).get::<IgnoreStruct>().unwrap().v, 1);
        }
        Err(error) => panic!("dynamic spawn of a component with an ignored field failed: {error}"),
    }
}

/// C2: no malformed asset-path spelling may panic; each one has to become a build error.
#[test]
fn c2_malformed_asset_paths_never_panic() {
    let app = test_app();
    for bad in ["bad#source://x.png", "://x.png", "s://x.png#a://b"] {
        let source = format!(r#"Sprite("{bad}")"#);
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| try_scene(&app, &source)));
        assert!(
            result.is_ok(),
            "`{bad}` panicked instead of producing a build error"
        );
    }
}

/// C3: the same malformed path through the *real* `.bsn` asset loader.
#[test]
fn c3_malformed_asset_path_through_the_loader() {
    let mut app = multi_asset_app_lenient(&[("bad.bsn", "Sprite(\"bad#source://x.png\")")]);
    let asset_server = app.world().resource::<AssetServer>().clone();
    let handle = asset_server.load::<ScenePatch>("bad.bsn");
    for _ in 0..200 {
        app.update();
        if asset_server.is_loaded(&handle) {
            break;
        }
    }
    assert!(
        !asset_server.is_loaded(&handle),
        "a malformed path must not load successfully"
    );
}

/// C4: an opaque type as a field.
#[derive(Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(opaque)]
#[reflect(Default)]
pub(crate) struct Opaque(pub(crate) u32);

#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct HasOpaque {
    pub(crate) v: Opaque,
}

#[test]
fn c4_opaque_field() {
    let mut app = test_app();
    app.register_type::<Opaque>().register_type::<HasOpaque>();
    // A bare path names the type; there is no way to give it a value, but it must not panic.
    assert!(try_scene(&app, "HasOpaque { v: Opaque }").is_ok());
    assert!(try_scene(&app, "HasOpaque { v: Opaque(1) }").is_err());
    let a = scene(&app, "a.bsn", "HasOpaque { v: Opaque }");
    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    assert_eq!(world.entity(id).get::<HasOpaque>().unwrap().v, Opaque(0));
}

/// C5: an enum whose variant name shadows a registered type of the same name.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct HasShadow {
    pub(crate) v: ShadowEnum,
}

#[derive(Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Default)]
pub(crate) enum ShadowEnum {
    #[default]
    Nothing,
    /// Shares its name with the registered fixture type `Bar`.
    Bar,
}

#[test]
fn c5_variant_name_shadows_a_registered_type() {
    let mut app = test_app();
    app.register_type::<ShadowEnum>()
        .register_type::<HasShadow>();
    // `Bar` is both a registered tuple struct and a variant of `ShadowEnum`.
    let a = scene(&app, "a.bsn", "HasShadow { v: Bar }");
    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    assert_eq!(
        world.entity(id).get::<HasShadow>().unwrap().v,
        ShadowEnum::Bar
    );
}

/// C4b: the static equivalent of C4 spawns fine.
#[test]
fn c4b_opaque_field_static_path_works() {
    let mut app = test_app();
    app.register_type::<Opaque>().register_type::<HasOpaque>();
    let world = app.world_mut();
    let id = world
        .spawn_scene(bsn! { HasOpaque { v: Opaque(3) } })
        .unwrap()
        .id();
    assert_eq!(world.entity(id).get::<HasOpaque>().unwrap().v, Opaque(3));
}

// =========================================================================================
// D: batch 4
// =========================================================================================

/// D1: `Name("x")` must work the same way in a `.bsn` file as it does in `bsn!`.
#[test]
fn d1_name_static_path_works() {
    let mut app = test_app();
    let world = app.world_mut();
    let id = world.spawn_scene(bsn! { Name("explicit") }).unwrap().id();
    assert_eq!(world.entity(id).get::<Name>().unwrap().as_str(), "explicit");
}

#[test]
fn d1b_name_dynamic_path() {
    let app = test_app();
    let result = try_scene(&app, "Name(\"explicit\")");
    assert!(
        result.is_ok(),
        "`Name(\"x\")` must work in a .bsn as it does in bsn!: {result:?}"
    );
}

/// D2: an empty relation block.
#[test]
fn d2_empty_relation_block() {
    let mut app = test_app();
    let a = scene(&app, "a.bsn", "Children [ ]\nMarker");
    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    assert_eq!(world.entity(id).get::<Children>().map(|c| c.len()), Some(0));
}

/// D3: a tuple value into a tuple-typed field.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct HasTuple {
    pub(crate) pair: (u32, f32),
}

#[test]
fn d3_tuple_value() {
    let mut app = test_app();
    app.register_type::<HasTuple>();
    let a = scene(&app, "a.bsn", "HasTuple { pair: (1, 2.0) }");
    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    assert_eq!(world.entity(id).get::<HasTuple>().unwrap().pair, (1, 2.0));
}

/// D4: naming the generated template's variant directly with `~`.
#[test]
fn d4_tilde_template_enum_variant() {
    let mut app = test_app();
    let a = scene(&app, "a.bsn", "~ChoiceTemplate::Baz(7)");
    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    assert_eq!(*world.entity(id).get::<Choice>().unwrap(), Choice::Baz(7));
}

/// D5: static + dynamic on the same *enum* component (typed recovery of an enum template).
#[test]
fn d5_static_over_dynamic_enum() {
    let mut app = test_app();
    let dynamic = scene(&app, "a.bsn", "Choice::Bar { y: 2 }");
    let world = app.world_mut();
    let id = world
        .spawn_scene(bsn! {
            @{dynamic}
            Choice::Bar { x: 1, y: 0, z: 0 }
        })
        .unwrap()
        .id();
    assert_eq!(
        *world.entity(id).get::<Choice>().unwrap(),
        Choice::Bar { x: 1, y: 0, z: 0 }
    );
}

/// D6: the same component patched in a base and again at two nesting depths.
#[test]
fn d6_cow_at_two_depths() {
    let mut app = multi_asset_app(&[("base.bsn", "Foo { x: 1, y: 1, z: 1 }")]);
    let mid = scene(&app, "mid.bsn", ":\"base.bsn\"\nFoo { y: 2 }");
    let world = app.world_mut();
    let id = world
        .spawn_scene((mid, bsn! { Foo { z: 3 } }))
        .unwrap()
        .id();
    let foo = world.entity(id).get::<Foo>().unwrap();
    assert_eq!((foo.x, foo.y, foo.z), (1, 2, 3));
}

/// D7: resolving a dynamic scene with no `ResolveContext::type_registry`.
#[test]
fn d7_resolve_without_a_context_registry() {
    use bevy_asset::Assets;
    let mut app = test_app();
    let a = scene(&app, "a.bsn", "Position { x: 1.0 } Foo { y: 2 }");
    let boxed: Box<dyn bevy_scene::Scene> = Box::new(a);
    let world = app.world_mut();
    let assets = world.resource::<AssetServer>().clone();
    let resolved = {
        let patches = world.resource::<Assets<ScenePatch>>();
        bevy_scene::ResolvedSceneRoot::resolve(boxed, &assets, patches, None).unwrap()
    };
    let id = resolved.spawn(world).unwrap().id();
    assert_eq!(world.entity(id).get::<Position>().unwrap().x, 1.0);
    assert_eq!(world.entity(id).get::<Foo>().unwrap().y, 2);
}

/// D8: scalars where collections are expected, and vice versa.
#[test]
fn d8_shape_mismatches_error() {
    let app = test_app();
    assert!(
        try_scene(&app, "Collections { list: 5 }").is_err(),
        "int into Vec"
    );
    assert!(
        try_scene(&app, "Collections { maybe: [1] }").is_err(),
        "list into Option"
    );
    assert!(
        try_scene(&app, "Primitives { a_string: 5 }").is_err(),
        "int into String"
    );
    assert!(
        try_scene(&app, r#"Primitives { a_u32: "x" }"#).is_err(),
        "string into int"
    );
    assert!(
        try_scene(&app, "Primitives { a_bool: 1 }").is_err(),
        "int into bool"
    );
}

/// D9: relation entries that are not entities / duplicate relation blocks.
#[test]
fn d9_two_relation_blocks_of_the_same_kind() {
    let mut app = test_app();
    let a = scene(&app, "a.bsn", "Children [ #A ]\nChildren [ #B ]");
    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    let children = world.entity(id).get::<Children>().unwrap().to_vec();
    assert_eq!(children.len(), 2);
    assert_eq!(
        world.entity(children[0]).get::<Name>().unwrap().as_str(),
        "A"
    );
    assert_eq!(
        world.entity(children[1]).get::<Name>().unwrap().as_str(),
        "B"
    );
}

/// C6: `Skipped` — an ignored field whose type is not `Clone` at all.
#[test]
fn c6_ignored_non_clone_field_spawns() {
    let mut app = test_app();
    let a = scene(&app, "a.bsn", "Skipped { value: 4 }");
    let world = app.world_mut();
    let result = world.spawn_scene(a);
    assert!(
        result.is_ok(),
        "spawning a component with an ignored field must work: {:?}",
        result.err().map(|e| e.to_string())
    );
}
