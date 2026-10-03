//! End-to-end tests for dynamic BSN against *real* engine components.
//!
//! `bevy_scene`'s own `.bsn` integration tests deliberately use synthetic fixture components (they
//! even declare same-named fakes such as `struct Sprite`), so nothing there proves that the real
//! `Node`/`Text`/`ImageNode`/`Transform`/`Sprite` types round-trip through the dynamic pipeline.
//! This file closes that gap from the root `bevy` package, where the whole type graph is present:
//!
//! * the short type paths the shipped `.bsn` files use are unambiguous in the full registry,
//! * the `.bsn` assets shipped with the `dynamic_bsn` example actually spawn, with the values they
//!   were written with (including the example's real `.png` handle dependency), and
//! * 2D/3D components spawn from `.bsn` text served out of an in-memory asset source.
//!
//! Requires the `bsn_asset` feature (enabled by default); see the `[[test]]` block in the root
//! `Cargo.toml`.
//!
//! No plugins are needed to populate the type registry: with `reflect_auto_register` (on by default
//! through the root crate's profiles) every `#[derive(Reflect)]` type in the dependency graph is
//! registered at `App::default()`.

use bevy::{
    asset::{
        io::{
            memory::{Dir, MemoryAssetReader},
            AssetSourceBuilder, AssetSourceId,
        },
        AssetApp, AssetPlugin, AssetServer, LoadState,
    },
    color::palettes::css,
    image::{CompressedImageFormats, Image, ImageLoader, ImagePlugin},
    prelude::*,
    reflect::TypeRegistry,
    scene::{SceneInstanceState, ScenePatch, ScenePatchInstance, ScenePlugin},
    ui::widget::ImageNode,
};

/// Every short type path named by a `.bsn` file under `assets/`, plus the ones this file's own
/// in-memory sources use.
///
/// A short path only resolves when it is unique across the *whole* registry, so this list failing
/// is the early warning for "a new type shadowed one the shipped scenes depend on" — which would
/// otherwise show up as a confusing `UnknownType` error at spawn time.
const SHIPPED_SHORT_PATHS: &[&str] = &[
    // `assets/scenes/dynamic_bsn_button.bsn`
    "Node",
    "Val",
    "JustifyContent",
    "AlignItems",
    "BackgroundColor",
    "Color",
    "Srgba",
    "Text",
    "Children",
    // `assets/scenes/dynamic_bsn_example.bsn`
    "FlexDirection",
    "ImageNode",
    "TextColor",
    "TextFont",
    "FontSize",
    // used by `spawns_transform_and_sprite_from_memory_bsn`
    "Transform",
    "Sprite",
    "Vec2",
    "Vec3",
];

/// Builds a headless [`App`] that can load and spawn `.bsn` assets.
///
/// `ImagePlugin` + a real [`ImageLoader`] are what let a scene's `Handle<Image>` dependency reach
/// `Loaded`: a `ScenePatch` is only resolved once every recursive dependency has (headless-image
/// precedent: `crates/bevy_image/src/saver.rs`).
fn test_app(source: Option<Dir>) -> App {
    let mut app = App::new();

    // Asset sources must be registered *before* `AssetPlugin`, or the plugin logs an error and the
    // registration is ignored. With no override the default source is the file reader, rooted at
    // `CARGO_MANIFEST_DIR` — the repository root for this package, so `assets/` is the shipped one.
    if let Some(dir) = source {
        app.register_asset_source(
            AssetSourceId::Default,
            AssetSourceBuilder::new(move || Box::new(MemoryAssetReader { root: dir.clone() })),
        );
    }

    app.add_plugins((
        TaskPoolPlugin::default(),
        AssetPlugin::default(),
        ImagePlugin::default(),
        ScenePlugin,
        BsnAssetPlugin,
    ));
    app.register_asset_loader(ImageLoader::new(CompressedImageFormats::empty()));
    app.finish();
    app.cleanup();
    app
}

/// Loads `path`, spawns a [`ScenePatchInstance`] of it, and pumps until the scene has been applied.
///
/// The frame bound is what makes the "never resolves" failure modes (a missing dependency, an
/// include that never loads) show up as a test failure rather than a hang; the load state is
/// reported with it, so a broken `.bsn` file names its own parse or resolution error instead of
/// only producing a timeout.
fn spawn_instance(app: &mut App, path: &'static str) -> Entity {
    const MAX_FRAMES: usize = 10_000;

    let asset_server = app.world().resource::<AssetServer>().clone();
    let handle = asset_server.load::<ScenePatch>(path);
    let entity = app
        .world_mut()
        .spawn(ScenePatchInstance(handle.clone()))
        .id();

    for frame in 0..MAX_FRAMES {
        app.update();
        // Yield real time after a warmup so the IO task pool gets scheduled on slow runners.
        if frame >= 100 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        if app
            .world()
            .get::<SceneInstanceState>(entity)
            .is_some_and(|state| state.applied)
        {
            return entity;
        }
        // A failed load is terminal, so report it immediately rather than spinning to the bound.
        if let LoadState::Failed(error) = asset_server.load_state(&handle) {
            panic!("`{path}` failed to load: {error}");
        }
    }

    // A scene only resolves once every recursive dependency has, so both states are reported: a
    // broken *include* leaves the root itself `Loaded`.
    panic!(
        "`{path}` never spawned within {MAX_FRAMES} frames (load state: {:?}, recursive \
         dependency load state: {:?})",
        asset_server.load_state(&handle),
        asset_server.recursive_dependency_load_state(&handle),
    );
}

/// Returns `root`'s child at `index`, panicking with the child count if it does not exist.
fn child(app: &App, root: Entity, index: usize) -> Entity {
    let children = app
        .world()
        .get::<Children>(root)
        .unwrap_or_else(|| panic!("entity {root} has no children"));
    *children.get(index).unwrap_or_else(|| {
        panic!(
            "entity {root} has {} children, wanted index {index}",
            children.len()
        )
    })
}

/// The `Name` of `entity`, for asserting on `#Name` references.
fn name(app: &App, entity: Entity) -> String {
    app.world()
        .get::<Name>(entity)
        .unwrap_or_else(|| panic!("entity {entity} has no Name"))
        .as_str()
        .to_string()
}

/// Asserts that `color` is the sRGB color `(red, green, blue)` at full alpha.
#[track_caller]
fn assert_srgb(color: Color, red: f32, green: f32, blue: f32) {
    let actual = Srgba::from(color);
    let expected = Srgba::new(red, green, blue, 1.0);
    assert!(
        (actual.red - expected.red).abs() < 1e-5
            && (actual.green - expected.green).abs() < 1e-5
            && (actual.blue - expected.blue).abs() < 1e-5
            && (actual.alpha - expected.alpha).abs() < 1e-5,
        "expected {expected:?}, got {actual:?}",
    );
}

/// The short type paths used by the shipped `.bsn` files all resolve to exactly one type.
///
/// `.bsn` resolves a short path through [`TypeRegistry::get_with_short_type_path`], which returns
/// `None` both when nothing is registered under the name *and* when the name is ambiguous. Either
/// regression (auto-registration breaking, or a second `Node` appearing in the graph) would break
/// every shipped scene, so it is asserted directly rather than only through spawn failures.
#[test]
fn real_component_short_paths_are_unambiguous() {
    // No plugins: `reflect_auto_register` populates the registry at `App::default()`.
    let app = App::new();
    let registry = app.world().resource::<AppTypeRegistry>().read();
    let registry: &TypeRegistry = &registry;

    let mut unresolved = Vec::new();
    for short_path in SHIPPED_SHORT_PATHS {
        if registry.get_with_short_type_path(short_path).is_none() {
            unresolved.push(*short_path);
        }
    }

    assert!(
        unresolved.is_empty(),
        "these short type paths are unregistered or ambiguous, so any `.bsn` file naming them \
         fails with `UnknownType`: {unresolved:?}",
    );
}

/// The shipped reusable-button scene spawns with the values it was written with.
#[test]
fn spawns_shipped_button_bsn() {
    let mut app = test_app(None);
    let root = spawn_instance(&mut app, "scenes/dynamic_bsn_button.bsn");

    let node = app.world().get::<Node>(root).expect("root has no Node");
    assert_eq!(node.width, Val::Px(180.0));
    assert_eq!(node.height, Val::Px(56.0));
    assert_eq!(node.justify_content, JustifyContent::Center);
    assert_eq!(node.align_items, AlignItems::Center);

    let background = app
        .world()
        .get::<BackgroundColor>(root)
        .expect("root has no BackgroundColor");
    assert_srgb(background.0, 0.15, 0.15, 0.15);

    // The base deliberately has no children: children of a base and of its includer are appended,
    // not merged pairwise, so the label is left to whoever includes this scene.
    assert!(app.world().get::<Children>(root).is_none());
}

/// The shipped example scene spawns, including its real `.png` handle dependency and the
/// `.bsn`-inherits-`.bsn` styled button.
#[test]
fn spawns_shipped_example_bsn() {
    let mut app = test_app(None);
    let root = spawn_instance(&mut app, "scenes/dynamic_bsn_example.bsn");

    // `#Root` becomes a `Name`.
    assert_eq!(name(&app, root), "Root");

    let node = app.world().get::<Node>(root).expect("root has no Node");
    assert_eq!(node.width, Val::Percent(100.0));
    assert_eq!(node.height, Val::Percent(100.0));
    assert_eq!(node.flex_direction, FlexDirection::Column);
    assert_eq!(node.row_gap, Val::Px(16.0));

    // `#Logo`: a string in a `Handle` field is an asset dependency, so the scene is not applied
    // until the image itself has loaded. Reaching this line at all proves the gating works.
    let logo = child(&app, root, 0);
    assert_eq!(name(&app, logo), "Logo");
    let image_node = app
        .world()
        .get::<ImageNode>(logo)
        .expect("logo has no ImageNode");
    assert!(
        app.world()
            .resource::<Assets<Image>>()
            .contains(&image_node.image),
        "the logo's image handle did not resolve to a loaded `Image`",
    );

    // `#Title`.
    let title = child(&app, root, 1);
    assert_eq!(name(&app, title), "Title");
    assert!(app.world().get::<Text>(title).is_some());
    assert_eq!(
        app.world()
            .get::<TextFont>(title)
            .expect("title has no TextFont")
            .font_size,
        FontSize::Px(28.0),
    );
    assert_srgb(
        app.world()
            .get::<TextColor>(title)
            .expect("title has no TextColor")
            .0,
        0.95,
        0.95,
        0.95,
    );

    // `#Confirm` inherits the shipped button scene from another `.bsn` file and overrides it.
    let confirm = child(&app, root, 2);
    assert_eq!(name(&app, confirm), "Confirm");
    let confirm_node = app
        .world()
        .get::<Node>(confirm)
        .expect("confirm has no Node");
    // Inherited from `dynamic_bsn_button.bsn`, untouched by the override.
    assert_eq!(confirm_node.height, Val::Px(56.0));
    assert_eq!(confirm_node.justify_content, JustifyContent::Center);
    assert_eq!(confirm_node.align_items, AlignItems::Center);
    // Overridden by the including file, switching `Val`'s variant: match-or-reset means the base's
    // `Px(180.0)` payload is discarded rather than merged into.
    assert_eq!(confirm_node.width, Val::Percent(30.0));
    assert_srgb(
        app.world()
            .get::<BackgroundColor>(confirm)
            .expect("confirm has no BackgroundColor")
            .0,
        0.1,
        0.3,
        0.55,
    );
    // The label comes from the including file, since the base leaves the slot empty.
    let confirm_children = app
        .world()
        .get::<Children>(confirm)
        .expect("confirm has no children");
    assert_eq!(confirm_children.len(), 1);
    let confirm_label = child(&app, confirm, 0);
    assert_eq!(name(&app, confirm_label), "Label");
    assert_eq!(
        app.world()
            .get::<Text>(confirm_label)
            .expect("confirm label has no Text")
            .0,
        "Confirm",
    );
}

/// 2D/3D components — the ones with no UI machinery behind them — spawn from `.bsn` text.
///
/// Served from an in-memory `Dir` rather than a shipped file so the source and the assertions sit
/// side by side. `Sprite::image` is deliberately left at its default: a handle field only becomes a
/// load-gating dependency when the `.bsn` names a path for it.
#[test]
fn spawns_transform_and_sprite_from_memory_bsn() {
    const SOURCE: &str = r#"
        Transform {
            translation: Vec3 { x: 1.0, y: 2.0, z: 3.0 },
            scale: Vec3 { x: 2.0, y: 2.0, z: 2.0 },
        }
        Sprite {
            color: Color::Srgba(Srgba { red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0 }),
            flip_x: true,
            custom_size: Some(Vec2 { x: 32.0, y: 48.0 }),
        }
    "#;

    let dir = Dir::default();
    dir.insert_asset_text(std::path::Path::new("sprite.bsn"), SOURCE);

    let mut app = test_app(Some(dir));
    let root = spawn_instance(&mut app, "sprite.bsn");

    let transform = app
        .world()
        .get::<Transform>(root)
        .expect("root has no Transform");
    assert_eq!(transform.translation, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(transform.scale, Vec3::splat(2.0));
    // Untouched fields keep the component's own default, not `Transform::default()`-per-field
    // guesswork.
    assert_eq!(transform.rotation, Quat::IDENTITY);

    let sprite = app.world().get::<Sprite>(root).expect("root has no Sprite");
    assert_eq!(Srgba::from(sprite.color), css::RED);
    assert!(sprite.flip_x);
    assert!(!sprite.flip_y);
    assert_eq!(sprite.custom_size, Some(Vec2::new(32.0, 48.0)));
    assert_eq!(sprite.image, Handle::default());

    // `Sprite`'s `#[require(Transform, Visibility, ...)]` still runs on the dynamic path.
    assert!(app.world().get::<Visibility>(root).is_some());
}
