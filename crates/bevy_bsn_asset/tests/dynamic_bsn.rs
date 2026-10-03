//! End-to-end tests for `DynamicBsnLoader`: real `.bsn` text, served from an in-memory asset
//! source, loaded through a real `AssetServer` and spawned through the real scene pipeline.
//!
//! Everything here exercises the *asset* layer. Document lowering itself is covered by
//! this crate's unit tests in `src/`.

extern crate alloc;

use alloc::sync::Arc;
use std::{path::Path, sync::Mutex};

use core::sync::atomic::{AtomicUsize, Ordering};

use bevy_app::{App, TaskPoolPlugin, Update};
use bevy_asset::{
    io::{
        memory::{Dir, MemoryAssetReader},
        AssetSourceBuilder, AssetSourceId,
    },
    Asset, AssetApp, AssetLoadFailedEvent, AssetPlugin, AssetServer, Assets, Handle, LoadState,
};
use bevy_bsn_asset::BsnAssetPlugin;
use bevy_ecs::{
    entity::Entity,
    error::{BevyError, Result},
    hierarchy::{ChildOf, Children},
    message::MessageReader,
    name::Name,
    prelude::{Component, Resource},
    reflect::{AppTypeRegistry, ReflectComponent, ReflectFromTemplate},
    system::Res,
    template::{FromTemplate, Template, TemplateContext},
};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_scene::{
    bsn, SceneInstanceState, ScenePatch, ScenePatchInstance, ScenePlugin, WorldSceneExt,
};

/// A three-field component, so that partial patching is observable.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

/// A stand-in asset type, so that a `Handle` field has somewhere to point.
#[derive(Asset, Reflect, Default)]
struct Image;

/// A loader for [`Image`], so that a `.bsn` file's handle dependencies actually reach `Loaded` —
/// a `ScenePatch` is only resolved once every recursive dependency has.
#[derive(bevy_reflect::TypePath)]
struct ImageLoader;

impl bevy_asset::AssetLoader for ImageLoader {
    type Asset = Image;
    type Error = std::io::Error;
    type Settings = ();

    async fn load(
        &self,
        _reader: &mut dyn bevy_asset::io::Reader,
        _settings: &Self::Settings,
        _load_context: &mut bevy_asset::LoadContext<'_>,
    ) -> std::result::Result<Self::Asset, Self::Error> {
        Ok(Image)
    }

    fn extensions(&self) -> &[&str] {
        &["png"]
    }
}

/// A component with a handle field, for the asset-dependency test.
///
/// `#[template(reflect)]` generates a reflectable `SpriteTemplate` whose `image` field is a
/// `HandleTemplate<Image>`, which is what a string asset path converts into.
#[derive(Component, FromTemplate, Reflect, PartialEq, Debug)]
#[template(reflect)]
#[reflect(Component, FromTemplate)]
struct Sprite {
    image: Handle<Image>,
}

/// A component with a nested struct field, for nested partial patches.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct Foo {
    x: u32,
    y: u32,
    z: u32,
    nested: Bar,
}

/// A three-field tuple struct, used both as a component and as [`Foo`]'s nested field.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct Bar(usize, usize, usize);

/// A two-field tuple struct of mixed types, for partial tuple patches.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct TupleStruct(f32, u32);

/// An enum component with one variant of each kind.
#[derive(Component, FromTemplate, Reflect, PartialEq, Debug)]
#[template(reflect)]
#[reflect(Component, FromTemplate)]
enum Choice {
    /// A struct variant.
    #[default]
    Bar {
        /// First field.
        x: u32,
        /// Second field.
        y: u32,
        /// Third field.
        z: u32,
    },
    /// A tuple variant.
    Baz(usize),
    /// A unit variant.
    Qux,
}

/// A component with a field of every primitive family.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct Primitives {
    a_i8: i8,
    a_i16: i16,
    a_i32: i32,
    a_i64: i64,
    a_i128: i128,
    a_isize: isize,
    a_u8: u8,
    a_u16: u16,
    a_u32: u32,
    a_u64: u64,
    a_u128: u128,
    a_usize: usize,
    a_f32: f32,
    a_f64: f64,
    a_bool: bool,
    a_string: String,
}

/// A component with `Option` and `Vec` fields.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct Collections {
    maybe: Option<u32>,
    list: Vec<u8>,
}

/// A component holding an entity reference, so `#Name` values have somewhere to go.
#[derive(Component, FromTemplate, Reflect, PartialEq, Debug)]
#[template(reflect)]
#[reflect(Component, FromTemplate)]
struct Reference(Entity);

/// An enum used only as the *source* of a registered conversion.
#[derive(Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Default)]
enum TextSize {
    /// The small size.
    #[default]
    Small,
    /// The large size.
    Large,
}

/// The *destination* of the [`TextSize`] conversion.
#[derive(Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Default)]
struct FontSize(u32);

/// A component with a [`FontSize`] field, to exercise implicit conversions on field values.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct TextFont {
    font_size: FontSize,
}

/// A generic component, to check that generic type paths resolve through the registry.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct Wrapper<T: Default + Clone + PartialEq + core::fmt::Debug> {
    value: T,
}

/// A component pulled in by [`NeedsRequired`]'s `#[require]`, never named in a `.bsn` file.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct RequiredExtra(u32);

/// A component that requires [`RequiredExtra`].
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
#[require(RequiredExtra)]
struct NeedsRequired;

/// Counts drops of [`DropTracker`], so the uninserted-component drop path is observable.
static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

/// A component that counts its own drops.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct DropTracker {
    value: u32,
}

impl Drop for DropTracker {
    fn drop(&mut self) {
        DROP_COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

/// A statically-defined template that always fails to build, for the failed-apply path.
#[derive(Component, Default)]
struct Fail;

impl FromTemplate for Fail {
    type Template = Fail;
}

impl Template for Fail {
    type Output = Fail;

    fn build_template(&self, _context: &mut TemplateContext) -> Result<Self::Output> {
        Err(BevyError::error("fail!"))
    }

    fn clone_template(&self) -> Self {
        Fail
    }
}

/// Twenty distinct marker components, to prove the dynamic path has no tuple-arity limit.
macro_rules! markers {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
            #[reflect(Component, Default)]
            struct $name;
        )*

        fn register_markers(app: &mut App) {
            $( app.register_type::<$name>(); )*
        }

        /// A `.bsn` source naming every marker, one after another on the root entity.
        const MARKERS_BSN: &str = concat!($(stringify!($name), "\n"),*);

        fn assert_all_markers(app: &App, entity: Entity) {
            $(
                assert!(
                    app.world().get::<$name>(entity).is_some(),
                    concat!("missing ", stringify!($name)),
                );
            )*
        }
    };
}

markers!(
    Mark00, Mark01, Mark02, Mark03, Mark04, Mark05, Mark06, Mark07, Mark08, Mark09, Mark10, Mark11,
    Mark12, Mark13, Mark14, Mark15, Mark16, Mark17, Mark18, Mark19,
);

/// Builds an [`App`] serving `dir` as the default asset source.
///
/// The source must be registered *before* `AssetPlugin`, or the plugin logs an error and the
/// registration is ignored.
fn test_app(dir: &Dir) -> App {
    let mut app = App::new();
    let reader_dir = dir.clone();
    app.register_asset_source(
        AssetSourceId::Default,
        AssetSourceBuilder::new(move || {
            Box::new(MemoryAssetReader {
                root: reader_dir.clone(),
            })
        }),
    );
    app.add_plugins((
        TaskPoolPlugin::default(),
        AssetPlugin::default(),
        ScenePlugin,
        BsnAssetPlugin,
    ));
    app.init_asset::<Image>();
    app.register_asset_reflect::<Image>();
    app.register_asset_loader(ImageLoader);
    app.register_type::<Position>();
    app.register_type::<Sprite>();
    app.register_type::<SpriteTemplate>();
    app.register_type::<Name>();
    app.register_type::<Children>();
    app.register_type::<ChildOf>();
    app.register_type::<Foo>();
    app.register_type::<Bar>();
    app.register_type::<TupleStruct>();
    app.register_type::<Choice>();
    app.register_type::<ChoiceTemplate>();
    app.register_type::<Primitives>();
    app.register_type::<Collections>();
    app.register_type::<Reference>();
    app.register_type::<ReferenceTemplate>();
    app.register_type::<TextSize>();
    app.register_type::<FontSize>();
    app.register_type::<TextFont>();
    app.register_type::<Wrapper<u32>>();
    app.register_type::<NeedsRequired>();
    app.register_type::<RequiredExtra>();
    app.register_type::<DropTracker>();
    register_markers(&mut app);
    app.world()
        .resource::<AppTypeRegistry>()
        .write()
        .register_type_conversion::<TextSize, FontSize, _>(|size| {
            Ok(FontSize(match size {
                TextSize::Small => 12,
                TextSize::Large => 24,
            }))
        });
    app.finish();
    app.cleanup();
    app
}

/// Loads `path`, spawns a [`ScenePatchInstance`] of it, and pumps until the scene has been applied.
fn spawn_instance(app: &mut App, path: &'static str) -> Entity {
    let asset_server = app.world().resource::<AssetServer>().clone();
    let handle = asset_server.load::<ScenePatch>(path);
    let entity = app.world_mut().spawn(ScenePatchInstance(handle)).id();
    run_app_until(app, |app| {
        app.world()
            .get::<SceneInstanceState>(entity)
            .is_some_and(|state| state.applied)
    });
    entity
}

/// The names of `root`'s children, in order.
fn child_names(app: &App, root: Entity) -> Vec<String> {
    app.world()
        .get::<Children>(root)
        .map(|children| {
            children
                .iter()
                .map(|child| {
                    app.world()
                        .get::<Name>(*child)
                        .map(|name| name.as_str().to_string())
                        .unwrap_or_default()
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Pumps `app` until `predicate` holds, or panics after a bounded number of frames.
///
/// The bound is what makes the "never resolves" failure modes (self-include, a missing base)
/// show up as a test failure rather than a hang.
fn run_app_until(app: &mut App, mut predicate: impl FnMut(&mut App) -> bool) {
    const MAX_FRAMES: usize = 10_000;
    for frame in 0..MAX_FRAMES {
        app.update();
        if predicate(app) {
            return;
        }
        // After a warmup, yield real time each frame: asset loads run on the IO task pool,
        // and a hot update loop on a starved CI runner can burn every frame before the
        // loader thread is ever scheduled.
        if frame >= 100 {
            std::thread::sleep(core::time::Duration::from_millis(1));
        }
    }
    panic!("the app never reached the expected state");
}

/// Returns the asset server's rendered error for `handle`, or panics if it has not failed.
fn load_error(app: &App, handle: &Handle<ScenePatch>) -> String {
    let asset_server = app.world().resource::<AssetServer>();
    match asset_server.load_state(handle) {
        LoadState::Failed(error) => error.to_string(),
        other => panic!("expected the load to fail, but it is {other:?}"),
    }
}

/// Loads `path` and pumps the app until the load either succeeds or fails.
fn load_and_settle(app: &mut App, path: &'static str) -> Handle<ScenePatch> {
    let asset_server = app.world().resource::<AssetServer>().clone();
    let handle = asset_server.load::<ScenePatch>(path);
    let probe = handle.clone();
    run_app_until(app, |_| {
        !matches!(
            asset_server.load_state(&probe),
            LoadState::NotLoaded | LoadState::Loading
        )
    });
    handle
}

#[test]
fn loads_empty_bsn_file() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("empty.bsn"), "// nothing but a comment\n");
    let mut app = test_app(&dir);

    let handle = load_and_settle(&mut app, "empty.bsn");
    assert!(
        matches!(
            app.world().resource::<AssetServer>().load_state(&handle),
            LoadState::Loaded
        ),
        "a comment-only `.bsn` file must load: {}",
        load_error(&app, &handle)
    );

    let patches = app.world().resource::<Assets<ScenePatch>>();
    assert!(
        patches.get(&handle).unwrap().resolved.is_some(),
        "the loaded patch must have been resolved by `resolve_scene_patches`"
    );
}

#[test]
fn spawns_scene_patch_instance_from_bsn() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Position { x: 1.0, y: 2.0 }");
    let mut app = test_app(&dir);

    // Spawn *before* the load finishes, so the entity goes through `WaitingScenes`.
    let asset_server = app.world().resource::<AssetServer>().clone();
    let handle = asset_server.load::<ScenePatch>("a.bsn");
    let id = app.world_mut().spawn(ScenePatchInstance(handle)).id();

    run_app_until(&mut app, |app| app.world().get::<Position>(id).is_some());

    assert_eq!(
        *app.world().get::<Position>(id).unwrap(),
        // `z` proves the patch is partial: it keeps the component's default.
        Position {
            x: 1.0,
            y: 2.0,
            z: 0.0
        }
    );
}

#[test]
fn spawns_children_and_names_from_bsn() {
    let dir = Dir::default();
    dir.insert_asset_text(
        Path::new("a.bsn"),
        "Children [ (#X), (#Y Position { x: 3.0 }) ]",
    );
    let mut app = test_app(&dir);

    let asset_server = app.world().resource::<AssetServer>().clone();
    let handle = asset_server.load::<ScenePatch>("a.bsn");
    let id = app.world_mut().spawn(ScenePatchInstance(handle)).id();

    run_app_until(&mut app, |app| app.world().get::<Children>(id).is_some());

    let children: Vec<_> = app.world().get::<Children>(id).unwrap().iter().collect();
    assert_eq!(children.len(), 2);
    assert_eq!(app.world().get::<Name>(*children[0]).unwrap().as_str(), "X");
    assert_eq!(app.world().get::<Name>(*children[1]).unwrap().as_str(), "Y");
    assert_eq!(
        app.world().get::<Position>(*children[1]).unwrap().x,
        3.0,
        "the second child's patch must be applied"
    );
}

#[test]
fn bsn_macro_inherits_from_bsn_asset() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Position { y: 2.0 }\nChildren [ (#X) ]");
    let mut app = test_app(&dir);

    let handle = load_and_settle(&mut app, "a.bsn");
    assert!(
        matches!(
            app.world().resource::<AssetServer>().load_state(&handle),
            LoadState::Loaded
        ),
        "{}",
        load_error(&app, &handle)
    );

    let world = app.world_mut();
    let id = world
        .spawn_scene(bsn! {
            :"a.bsn"
            Position { x: 1.0 }
            Children [ #Y ]
        })
        .unwrap()
        .id();

    assert_eq!(
        *world.get::<Position>(id).unwrap(),
        // `x` from the macro, `y` from the asset, `z` from neither: field-level merging.
        Position {
            x: 1.0,
            y: 2.0,
            z: 0.0
        }
    );

    let children: Vec<_> = world.get::<Children>(id).unwrap().iter().collect();
    assert_eq!(children.len(), 2);
    assert_eq!(world.get::<Name>(*children[0]).unwrap().as_str(), "X");
    assert_eq!(world.get::<Name>(*children[1]).unwrap().as_str(), "Y");
}

#[test]
fn nested_bsn_includes_resolve_base_first() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("c.bsn"), "Position { z: 3.0 }");
    dir.insert_asset_text(Path::new("b.bsn"), ":\"c.bsn\"\nPosition { y: 2.0 }");
    dir.insert_asset_text(Path::new("a.bsn"), ":\"b.bsn\"\nPosition { x: 1.0 }");
    let mut app = test_app(&dir);

    let asset_server = app.world().resource::<AssetServer>().clone();
    let handle = asset_server.load::<ScenePatch>("a.bsn");
    let id = app.world_mut().spawn(ScenePatchInstance(handle)).id();

    run_app_until(&mut app, |app| app.world().get::<Position>(id).is_some());

    assert_eq!(
        *app.world().get::<Position>(id).unwrap(),
        Position {
            x: 1.0,
            y: 2.0,
            z: 3.0
        },
        "a three-level include chain must resolve base-first"
    );
}

#[test]
fn asset_path_field_becomes_dependency() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Sprite { image: \"logo.png\" }");
    dir.insert_asset_text(Path::new("logo.png"), "not really a png");
    let mut app = test_app(&dir);

    let handle = load_and_settle(&mut app, "a.bsn");
    let patches = app.world().resource::<Assets<ScenePatch>>();
    let patch = patches.get(&handle).expect("the patch must exist");
    assert!(
        !patch.dependencies.is_empty(),
        "a handle-valued field must become an asset dependency"
    );

    let asset_server = app.world().resource::<AssetServer>();
    assert!(
        asset_server.get_handle::<Image>("logo.png").is_some(),
        "the dependency must have been registered with the asset server"
    );
}

#[test]
fn parse_error_reports_line_and_column() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("bad.bsn"), "Position {\n  x: ,\n}\n");
    let mut app = test_app(&dir);

    let handle = load_and_settle(&mut app, "bad.bsn");
    let error = load_error(&app, &handle);
    assert!(
        error.contains("bad.bsn:2:"),
        "the error must carry `path:line:`: {error}"
    );
}

#[test]
fn unregistered_type_reports_error_with_location() {
    let dir = Dir::default();
    dir.insert_asset_text(
        Path::new("a.bsn"),
        "Position { x: 1.0 }\nmy_game::NotRegistered",
    );
    let mut app = test_app(&dir);

    let handle = load_and_settle(&mut app, "a.bsn");
    let error = load_error(&app, &handle);
    assert!(
        error.contains("my_game::NotRegistered"),
        "the error must name the offending type: {error}"
    );
    assert!(
        error.contains("a.bsn:2:1"),
        "the error must carry `path:line:column`: {error}"
    );
}

#[test]
fn multiple_roots_rejected() {
    let dir = Dir::default();
    dir.insert_asset_text(
        Path::new("a.bsn"),
        "(Position { x: 1.0 }),\n(Position { y: 2.0 })",
    );
    let mut app = test_app(&dir);

    let handle = load_and_settle(&mut app, "a.bsn");
    let error = load_error(&app, &handle);
    assert!(
        error.contains("exactly one root entity"),
        "unexpected error: {error}"
    );
}

#[test]
fn spawn_after_load_applies_immediately() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Position { x: 1.0 }");
    let mut app = test_app(&dir);

    let handle = load_and_settle(&mut app, "a.bsn");
    assert!(app
        .world()
        .resource::<Assets<ScenePatch>>()
        .get(&handle)
        .unwrap()
        .resolved
        .is_some());

    // The asset is already resolved, so this entity never enters `WaitingScenes`.
    let id = app.world_mut().spawn(ScenePatchInstance(handle)).id();
    app.update();

    assert_eq!(app.world().get::<Position>(id).unwrap().x, 1.0);
}

#[test]
fn self_include_rejected() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), ":\"a.bsn\"\nPosition { x: 1.0 }");
    let mut app = test_app(&dir);

    // `load_and_settle` is bounded: a self-including file must fail fast rather than hang
    // waiting for its own recursive dependency state to become `Loaded`.
    let handle = load_and_settle(&mut app, "a.bsn");
    let error = load_error(&app, &handle);
    assert!(
        error.contains("inherits from itself"),
        "unexpected error: {error}"
    );
}

/// Collects every `AssetLoadFailedEvent<ScenePatch>` a test observes, across all frames.
#[derive(Resource, Default, Clone)]
struct ObservedFailures(Arc<Mutex<Vec<String>>>);

fn collect_failures(
    mut failures: MessageReader<AssetLoadFailedEvent<ScenePatch>>,
    observed: Res<ObservedFailures>,
) {
    let mut observed = observed.0.lock().unwrap();
    for failure in failures.read() {
        observed.push(failure.error.to_string());
    }
}

#[test]
fn failed_load_is_reported_once() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("bad.bsn"), "Position {\n  x: ,\n}\n");
    let mut app = test_app(&dir);
    let observed = ObservedFailures::default();
    app.insert_resource(observed.clone());
    app.add_systems(Update, collect_failures);

    let handle = load_and_settle(&mut app, "bad.bsn");
    assert!(matches!(
        app.world().resource::<AssetServer>().load_state(&handle),
        LoadState::Failed(_)
    ));

    // Keep pumping: `Messages` retains events for two frames, but the reader's cursor means the
    // failure is seen — and therefore logged by `report_scene_patch_load_failures` — exactly once.
    for _ in 0..10 {
        app.update();
        std::thread::sleep(core::time::Duration::from_millis(1));
    }

    let observed = observed.0.lock().unwrap();
    assert_eq!(
        observed.len(),
        1,
        "expected exactly one load-failure event, got {observed:?}"
    );
    assert!(
        observed[0].contains("bad.bsn:"),
        "unexpected error: {}",
        observed[0]
    );
}

// ===========================================================================================
// Static -> dynamic parity matrix
//
// One test per `bsn!` feature that a `.bsn` file can also express, named `dyn_<static test>`
// after its counterpart in `crates/bevy_scene/src/lib.rs`. Rows the format deliberately cannot
// express are listed as N/A at the bottom of this section, each with the non-goal that excludes
// it, so the matrix stays auditable.
// ===========================================================================================

/// Parity with `supports_fully_qualified_component_paths`.
#[test]
fn dyn_fully_qualified_component_paths() {
    let dir = Dir::default();
    // Adaptation: unlike the `bsn!` macro, the `.bsn` grammar has no leading `::` — paths are
    // always resolved against the type registry, never against Rust's item namespace — so the
    // fully-qualified form is written without it. The rejection is asserted below.
    dir.insert_asset_text(
        Path::new("a.bsn"),
        "bevy_ecs::hierarchy::Children [ (dynamic_bsn::Position { x: 1.0 }) ]",
    );
    dir.insert_asset_text(
        Path::new("leading.bsn"),
        "::dynamic_bsn::Position { x: 1.0 }",
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    let children: Vec<_> = app.world().get::<Children>(root).unwrap().iter().collect();
    assert_eq!(children.len(), 1);
    assert_eq!(app.world().get::<Position>(*children[0]).unwrap().x, 1.0);

    let leading = load_and_settle(&mut app, "leading.bsn");
    assert!(load_error(&app, &leading).contains("may not start with `::`"));
}

/// Parity with `cached_patching`: field-level merge across a `:"base.bsn"` include.
#[test]
fn dyn_cached_patching() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Position { y: 2.0 }");
    dir.insert_asset_text(Path::new("b.bsn"), ":\"a.bsn\"\nPosition { x: 1.0 }");
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "b.bsn");
    assert_eq!(
        *app.world().get::<Position>(root).unwrap(),
        Position {
            x: 1.0,
            y: 2.0,
            z: 0.0
        }
    );
}

/// Parity with `cached_patching_order`: the dependent's value wins, because the base is applied
/// first.
#[test]
fn dyn_cached_patching_order() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Position { x: 1.0, y: 2.0 }");
    dir.insert_asset_text(Path::new("b.bsn"), ":\"a.bsn\"\nPosition { x: 3.0 }");
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "b.bsn");
    assert_eq!(
        *app.world().get::<Position>(root).unwrap(),
        Position {
            x: 3.0,
            y: 2.0,
            z: 0.0
        }
    );
}

/// Parity with `loaded_asset_cached_patching`: children of base and dependent both spawn, in
/// order, and the shared component merges.
#[test]
fn dyn_loaded_asset_cached_patching() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Position { y: 2.0 }\nChildren [ #X ]");
    dir.insert_asset_text(
        Path::new("b.bsn"),
        ":\"a.bsn\"\nPosition { x: 1.0 }\nChildren [ #Y ]",
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "b.bsn");
    assert_eq!(
        *app.world().get::<Position>(root).unwrap(),
        Position {
            x: 1.0,
            y: 2.0,
            z: 0.0
        }
    );
    assert_eq!(child_names(&app, root), ["X", "Y"]);
}

/// Parity with `inline_scene_patching`: two patches of one component on one entity land in the
/// same template slot and merge, rather than one overwriting the other.
#[test]
fn dyn_repeated_patch_same_entity() {
    let dir = Dir::default();
    dir.insert_asset_text(
        Path::new("a.bsn"),
        "Position { x: 1.0 }\nPosition { y: 2.0 }",
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    assert_eq!(
        *app.world().get::<Position>(root).unwrap(),
        Position {
            x: 1.0,
            y: 2.0,
            z: 0.0
        }
    );
}

/// Parity with `hierarchy`.
#[test]
fn dyn_hierarchy() {
    let dir = Dir::default();
    dir.insert_asset_text(
        Path::new("a.bsn"),
        "#A Children [ (#B Children [ #X ]), (#C Children [ #Y ]) ]",
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    assert_eq!(app.world().get::<Name>(root).unwrap().as_str(), "A");
    assert_eq!(child_names(&app, root), ["B", "C"]);

    let children: Vec<_> = app.world().get::<Children>(root).unwrap().iter().collect();
    assert_eq!(child_names(&app, *children[0]), ["X"]);
    assert_eq!(child_names(&app, *children[1]), ["Y"]);
}

/// Parity with `bsn_name_references`: a child referring back to the root.
#[test]
fn dyn_name_references() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "#X Children [ (Reference(#X)) ]");
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    let child = app.world().get::<Children>(root).unwrap()[0];
    assert_eq!(app.world().get::<Reference>(child).unwrap().0, root);
}

/// Parity with `bsn_reverse_reference`: a reference that is only defined *later* in the document.
#[test]
fn dyn_reverse_reference() {
    let dir = Dir::default();
    dir.insert_asset_text(
        Path::new("a.bsn"),
        "Reference(#Last)\nChildren [ #First, #Second, #Last ]",
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    let children: Vec<_> = app.world().get::<Children>(root).unwrap().iter().collect();
    assert_eq!(child_names(&app, root), ["First", "Second", "Last"]);
    assert_eq!(app.world().get::<Reference>(root).unwrap().0, *children[2]);
}

// Row 9 (`bsn_list_name_references`) is N/A: SPEC-3's grammar has no scene-list root form, so a
// `.bsn` file always describes exactly one root entity.

/// Parity with `primitive_literals`: every numeric family, `bool`, `String` and a list.
#[test]
fn dyn_primitive_literals() {
    let dir = Dir::default();
    dir.insert_asset_text(
        Path::new("a.bsn"),
        r#"Primitives {
            a_i8: -1, a_i16: -2, a_i32: -3, a_i64: -4, a_i128: -5, a_isize: -6,
            a_u8: 1, a_u16: 2, a_u32: 3, a_u64: 4, a_u128: 5, a_usize: 6,
            a_f32: 1.5, a_f64: -2.5, a_bool: true, a_string: "hello",
        }
        Collections { list: [1, 2, 3] }"#,
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    assert_eq!(
        *app.world().get::<Primitives>(root).unwrap(),
        Primitives {
            a_i8: -1,
            a_i16: -2,
            a_i32: -3,
            a_i64: -4,
            a_i128: -5,
            a_isize: -6,
            a_u8: 1,
            a_u16: 2,
            a_u32: 3,
            a_u64: 4,
            a_u128: 5,
            a_usize: 6,
            a_f32: 1.5,
            a_f64: -2.5,
            a_bool: true,
            a_string: "hello".to_string(),
        }
    );
    assert_eq!(
        app.world().get::<Collections>(root).unwrap().list,
        vec![1, 2, 3]
    );
}

/// Parity with `partial_tuple_struct`: leading fields set, trailing fields defaulted.
#[test]
fn dyn_partial_tuple_struct() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "TupleStruct(0.5)");
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    assert_eq!(
        *app.world().get::<TupleStruct>(root).unwrap(),
        TupleStruct(0.5, 0)
    );
}

/// Parity with `enum_patching`: each variant kind, and a later patch replacing an earlier variant.
#[test]
fn dyn_enum_patching() {
    let dir = Dir::default();
    dir.insert_asset_text(
        Path::new("struct_variant.bsn"),
        "Choice::Bar { x: 1, y: 2 }",
    );
    dir.insert_asset_text(Path::new("tuple_variant.bsn"), "Choice::Baz(10)");
    dir.insert_asset_text(Path::new("unit_variant.bsn"), "Choice::Qux");
    dir.insert_asset_text(
        Path::new("repatched.bsn"),
        "Choice::Baz(10)\nChoice::Bar { x: 1 }",
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "struct_variant.bsn");
    assert_eq!(
        *app.world().get::<Choice>(root).unwrap(),
        Choice::Bar { x: 1, y: 2, z: 0 }
    );

    let root = spawn_instance(&mut app, "tuple_variant.bsn");
    assert_eq!(*app.world().get::<Choice>(root).unwrap(), Choice::Baz(10));

    let root = spawn_instance(&mut app, "unit_variant.bsn");
    assert_eq!(*app.world().get::<Choice>(root).unwrap(), Choice::Qux);

    let root = spawn_instance(&mut app, "repatched.bsn");
    assert_eq!(
        *app.world().get::<Choice>(root).unwrap(),
        Choice::Bar { x: 1, y: 0, z: 0 },
        "a later patch naming a different variant replaces the whole value"
    );
}

/// Parity with `struct_patching`: nested partial patches over two files.
#[test]
fn dyn_struct_patching() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Foo { x: 1, nested: Bar(1, 1) }");
    dir.insert_asset_text(
        Path::new("b.bsn"),
        ":\"a.bsn\"\nFoo { y: 2, nested: Bar(2) }",
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "b.bsn");
    assert_eq!(
        *app.world().get::<Foo>(root).unwrap(),
        Foo {
            x: 1,
            y: 2,
            z: 0,
            nested: Bar(2, 1, 0)
        }
    );
}

/// Parity with `field_patching_with_default`: unmentioned fields keep the template's default.
#[test]
fn dyn_field_patching_with_default() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Foo { y: 2 }");
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    assert_eq!(
        *app.world().get::<Foo>(root).unwrap(),
        Foo {
            x: 0,
            y: 2,
            z: 0,
            nested: Bar(0, 0, 0)
        }
    );
}

/// Parity with `handle_template`: a string field typed as a `Handle` becomes an asset dependency,
/// and the patch does not resolve until that asset has loaded.
#[test]
fn dyn_handle_template() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Sprite { image: \"logo.png\" }");
    dir.insert_asset_text(Path::new("logo.png"), "not really a png");
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    let sprite = app.world().get::<Sprite>(root).unwrap();
    let asset_server = app.world().resource::<AssetServer>();
    assert_eq!(
        sprite.image.id(),
        asset_server.get_handle::<Image>("logo.png").unwrap().id(),
        "the string became the very handle the asset server registered as a dependency"
    );
}

/// Parity with `scene_list_children`: children spawn in document order.
#[test]
fn dyn_children_list() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Children [ #A, #B, #C ]");
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    assert_eq!(child_names(&app, root), ["A", "B", "C"]);
}

/// Parity with `generic_patching`: a generic component named by its registered type path.
#[test]
fn dyn_generic_patching() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Wrapper<u32> { value: 3 }");
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    assert_eq!(
        *app.world().get::<Wrapper<u32>>(root).unwrap(),
        Wrapper { value: 3 }
    );
}

/// Parity with `comments_in_bsn`.
#[test]
fn dyn_comments() {
    let dir = Dir::default();
    dir.insert_asset_text(
        Path::new("a.bsn"),
        "// Look ma, a comment!\n#MyName\n/*\n  Wow, a block comment now?\n*/\nPosition { x: 1.0 }",
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    assert_eq!(app.world().get::<Name>(root).unwrap().as_str(), "MyName");
    assert_eq!(app.world().get::<Position>(root).unwrap().x, 1.0);
}

/// Parity with `bsn_entry_can_surpass_tuple_limit`: the dynamic path builds a `Vec` of templates,
/// so there is no arity limit to surpass.
#[test]
fn dyn_many_patches_on_one_entity() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), MARKERS_BSN);
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    assert_all_markers(&app, root);
}

/// Parity with `scene_without_explicit_component_still_spawns_component`: `#[require]`d components
/// arrive even though the file never names them.
#[test]
fn dyn_required_components() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "NeedsRequired");
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    assert!(app.world().get::<NeedsRequired>(root).is_some());
    assert!(
        app.world().get::<RequiredExtra>(root).is_some(),
        "required components are inserted by the ECS, not by the scene"
    );
}

/// Parity with `enum_variant_field_values_use_implicit_into`: the coercion ladder, both the
/// registered-conversion rung and the lossless-numeric-widening rung.
#[test]
fn dyn_value_coercion() {
    let dir = Dir::default();
    dir.insert_asset_text(
        Path::new("a.bsn"),
        "TextFont { font_size: TextSize::Large }\nPrimitives { a_f64: 3, a_u64: 7 }",
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    assert_eq!(
        app.world().get::<TextFont>(root).unwrap().font_size,
        FontSize(24),
        "a value of the wrong type is converted through the registered conversion"
    );
    let primitives = app.world().get::<Primitives>(root).unwrap();
    assert_eq!(
        primitives.a_f64, 3.0,
        "an integer literal widens to a float"
    );
    assert_eq!(primitives.a_u64, 7);
}

/// Parity with `scene_with_optional_components`: an `Option` field, given a bare value, wrapped
/// explicitly, and omitted.
#[test]
fn dyn_optional_components() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("implicit.bsn"), "Collections { maybe: 5 }");
    dir.insert_asset_text(Path::new("explicit.bsn"), "Collections { maybe: Some(7) }");
    dir.insert_asset_text(Path::new("omitted.bsn"), "Collections { list: [1] }");
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "implicit.bsn");
    assert_eq!(app.world().get::<Collections>(root).unwrap().maybe, Some(5));

    let root = spawn_instance(&mut app, "explicit.bsn");
    assert_eq!(app.world().get::<Collections>(root).unwrap().maybe, Some(7));

    let root = spawn_instance(&mut app, "omitted.bsn");
    assert_eq!(app.world().get::<Collections>(root).unwrap().maybe, None);
}

/// Parity with `scene_nested_entity_references`: a three-deep `#Name` graph, where every reference
/// resolves to a distinct, correct entity.
#[test]
fn dyn_nested_entity_references() {
    let dir = Dir::default();
    dir.insert_asset_text(
        Path::new("a.bsn"),
        "#Root Children [ (#Middle Reference(#Root) Children [ (#Leaf Reference(#Middle)) ]) ]",
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    let middle = app.world().get::<Children>(root).unwrap()[0];
    let leaf = app.world().get::<Children>(middle).unwrap()[0];

    assert_eq!(app.world().get::<Reference>(middle).unwrap().0, root);
    assert_eq!(app.world().get::<Reference>(leaf).unwrap().0, middle);
    assert_ne!(root, middle);
    assert_ne!(middle, leaf);
}

/// Parity with `repeated_call_entity_reference`. A `.bsn` file's `SceneEntityReference`s are keyed
/// on the *asset path*, so they are identical for every instance of that file. Two instances must
/// still get two distinct entities, which holds only because `ResolvedSceneRoot::apply` builds a
/// fresh reference map per apply.
#[test]
fn dyn_two_instances_get_distinct_entities() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "#X Children [ (Reference(#X)) ]");
    let mut app = test_app(&dir);

    let first = spawn_instance(&mut app, "a.bsn");
    let second = spawn_instance(&mut app, "a.bsn");

    let first_child = app.world().get::<Children>(first).unwrap()[0];
    let second_child = app.world().get::<Children>(second).unwrap()[0];
    assert_ne!(first_child, second_child);
    assert_eq!(app.world().get::<Reference>(first_child).unwrap().0, first);
    assert_eq!(
        app.world().get::<Reference>(second_child).unwrap().0,
        second,
        "the same `#Name` in two instances must resolve to two different entities"
    );
}

/// Parity with `drop_is_called_for_uninserted_components`: when an apply fails part way, the
/// component values already built from the `.bsn` file are dropped rather than leaked.
#[test]
fn dyn_drop_on_failed_apply() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "DropTracker { value: 1 }");
    let mut app = test_app(&dir);

    let asset_server = app.world().resource::<AssetServer>().clone();
    let handle = asset_server.load::<ScenePatch>("a.bsn");
    let probe = handle.clone();
    run_app_until(&mut app, |app| {
        app.world()
            .resource::<Assets<ScenePatch>>()
            .get(&probe)
            .is_some_and(|patch| patch.resolved.is_some())
    });

    let before_drops = DROP_COUNT.load(Ordering::SeqCst);
    let before_entities = app.world().entities().count_spawned();
    // The `.bsn` component is built and staged, then `Fail` aborts the apply before the bundle is
    // written, so the staged value has to be dropped by hand.
    let result = app.world_mut().spawn_scene(bsn! { :"a.bsn" Fail });
    assert!(result.is_err());

    assert_eq!(
        DROP_COUNT.load(Ordering::SeqCst) - before_drops,
        1,
        "the component built from the `.bsn` file must be dropped exactly once"
    );
    assert_eq!(
        app.world().entities().count_spawned(),
        before_entities,
        "a failed spawn must not leak entities"
    );
    drop(handle);
}

/// Parity with `despawn_on_failed_spawn`: a `.bsn` file naming an unregistered type fails to load,
/// is logged rather than panicked, and leaves the world alone.
#[test]
fn dyn_no_entity_leak_on_failed_load() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("bad.bsn"), "my_game::NotRegistered");
    let mut app = test_app(&dir);

    let before = app.world().entities().count_spawned();
    let asset_server = app.world().resource::<AssetServer>().clone();
    let handle = asset_server.load::<ScenePatch>("bad.bsn");
    let root = app
        .world_mut()
        .spawn(ScenePatchInstance(handle.clone()))
        .id();

    run_app_until(&mut app, |app| {
        matches!(
            app.world().resource::<AssetServer>().load_state(&handle),
            LoadState::Failed(_)
        )
    });
    for _ in 0..10 {
        app.update();
        std::thread::sleep(core::time::Duration::from_millis(1));
    }

    assert!(load_error(&app, &handle).contains("my_game::NotRegistered"));
    assert!(app.world().get_entity(root).is_ok());
    assert_eq!(
        app.world().entities().count_spawned(),
        before + 1,
        "only the instance entity itself exists; the scene never spawned anything"
    );
}

// Explicitly N/A for the `.bsn` format, each excluded by a SPEC-0 §2 non-goal:
//
// - `constant_values`, `direct_macro_values_in_bsn` — Rust consts are not expressible in an asset.
// - `on_template` — observers (`on(...)`) are not expressible in an asset.
// - `children_list_expr`, `children_single_expr`, `scene_expression_passing_pointless`,
//   `empty_scene_expressions`, `scene_with_blocks`, `enum_variant_subexpressions_are_hoisted` —
//   `{ expr }` blocks are not expressible in an asset.
// - `conditional_scene` — requires Rust control flow.
// - `closures_in_bsn`, `scene_with_oneshot_system` — closures and systems are not expressible.
// - `component_scene`, `component_scene_props`, and the four `*_name_reference` scene-component
//   tests — invoking `SceneComponent::scene(props)` is out of scope (`Props` is not `Reflect`).
// - `child_of_template`, `bsn_list_name_references`, `scene_list_nested_entity_references` — a
//   `.bsn` file has no scene-list root form.
// - `field_name_shorthand` — field shorthand borrows a Rust binding.

// ===========================================================================================
// Cross-feature integration: static `bsn!` <-> dynamic `.bsn`
// ===========================================================================================

/// A `bsn!` scene inheriting a `.bsn` file merges into the *same* template slot, rather than
/// writing the component twice.
#[test]
fn bsn_macro_inherits_bsn_file() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("base.bsn"), "Position { y: 2.0 }");
    let mut app = test_app(&dir);
    load_and_settle(&mut app, "base.bsn");

    let world = app.world_mut();
    let root = world
        .spawn_scene(bsn! { :"base.bsn" Position { x: 1.0 } })
        .unwrap()
        .id();

    assert_eq!(
        *world.get::<Position>(root).unwrap(),
        Position {
            x: 1.0,
            y: 2.0,
            z: 0.0
        },
        "a single, merged `Position` — not the macro's value overwriting the asset's"
    );
}

/// A three-level `.bsn` include chain resolves base-first at every level.
#[test]
fn bsn_file_inherits_bsn_file() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("c.bsn"), "Position { z: 3.0 }\nChildren [ #C ]");
    dir.insert_asset_text(
        Path::new("b.bsn"),
        ":\"c.bsn\"\nPosition { y: 2.0 }\nChildren [ #B ]",
    );
    dir.insert_asset_text(
        Path::new("a.bsn"),
        ":\"b.bsn\"\nPosition { x: 1.0 }\nChildren [ #A ]",
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    assert_eq!(
        *app.world().get::<Position>(root).unwrap(),
        Position {
            x: 1.0,
            y: 2.0,
            z: 3.0
        }
    );
    // Only two levels of children appear. Applying a cached scene applies *that* scene's related
    // entities, but not those of the scene *it* caches in turn, so `c.bsn`'s child is dropped.
    // This is pre-existing behavior of the cached apply path (`ResolvedScene::apply_with`), not
    // something hot reload introduces; it is pinned here so the boundary is visible. The template
    // merge above is transitive because copy-on-write flattens templates at resolve time.
    assert_eq!(child_names(&app, root), ["B", "A"]);
}

/// A dynamic base and a static patch on one component produce exactly one component, merged
/// field-wise — the copy-on-write path has to skip the base's own write.
#[test]
fn dynamic_and_static_patch_merge_on_one_component() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("base.bsn"), "Position { y: 2.0 }");
    let mut app = test_app(&dir);
    load_and_settle(&mut app, "base.bsn");

    let world = app.world_mut();
    let root = world
        .spawn_scene(bsn! { :"base.bsn" Position { x: 1.0 } })
        .unwrap()
        .id();

    assert_eq!(
        *world.get::<Position>(root).unwrap(),
        Position {
            x: 1.0,
            y: 2.0,
            z: 0.0
        }
    );
    // One `Position`, not two writes: if the base's template were also applied, `x` would have
    // been reset to its default by whichever write landed last.
    let position_id = world.component_id::<Position>().unwrap();
    assert_eq!(
        world
            .entity(root)
            .archetype()
            .components()
            .iter()
            .filter(|id| **id == position_id)
            .count(),
        1
    );
}

/// Children declared dynamically and statically land in one `Children` collection, in order.
#[test]
fn dynamic_children_merge_with_static_children() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("base.bsn"), "Children [ #A ]");
    let mut app = test_app(&dir);
    load_and_settle(&mut app, "base.bsn");

    let root = app
        .world_mut()
        .spawn_scene(bsn! { :"base.bsn" Children [ #B ] })
        .unwrap()
        .id();

    assert_eq!(child_names(&app, root), ["A", "B"]);
    assert_eq!(app.world().get::<Children>(root).unwrap().len(), 2);
}

/// A `.bsn` file naming an asset path does not resolve — and its instances do not spawn — until
/// that asset has loaded.
#[test]
fn bsn_file_asset_dependency_gate() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Sprite { image: \"logo.png\" }");
    dir.insert_asset_text(Path::new("logo.png"), "not really a png");
    let mut app = test_app(&dir);

    let asset_server = app.world().resource::<AssetServer>().clone();
    let handle = asset_server.load::<ScenePatch>("a.bsn");
    let root = app
        .world_mut()
        .spawn(ScenePatchInstance(handle.clone()))
        .id();

    // Before the dependency has loaded, the patch is unresolved and the instance is untouched.
    assert!(app.world().get::<Sprite>(root).is_none());

    run_app_until(&mut app, |app| app.world().get::<Sprite>(root).is_some());
    assert!(app
        .world()
        .resource::<Assets<ScenePatch>>()
        .get(&handle)
        .unwrap()
        .resolved
        .is_some());
}

// ===========================================================================================
// Hot reload, through real `.bsn` text
//
// `Dir::insert_asset_text` overwrites, so writing the same path twice is "edit the file";
// `AssetServer::reload` is "save". Together they drive exactly the path a file watcher drives.
// ===========================================================================================

/// Overwrites `path` with `text`, then re-runs its loader.
fn edit_and_save(app: &App, dir: &Dir, path: &'static str, text: &str) {
    dir.insert_asset_text(Path::new(path), text);
    app.world().resource::<AssetServer>().reload(path);
}

#[test]
fn hot_reload_bsn_file_updates_instance() {
    let dir = Dir::default();
    dir.insert_asset_text(
        Path::new("a.bsn"),
        "Position { x: 1.0 }\nChildren [ #A, #B ]",
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    let previous: Vec<Entity> = app
        .world()
        .get::<Children>(root)
        .unwrap()
        .iter()
        .copied()
        .collect();
    let before = app.world().entities().count_spawned();

    edit_and_save(&app, &dir, "a.bsn", "Position { x: 5.0 }\nChildren [ #C ]");
    run_app_until(&mut app, |app| child_names(app, root) == ["C"]);

    assert_eq!(app.world().get::<Position>(root).unwrap().x, 5.0);
    for entity in previous {
        assert!(
            app.world().get_entity(entity).is_err(),
            "the previous generation must be despawned, not orphaned"
        );
    }
    assert_eq!(
        app.world().entities().count_spawned(),
        before - 1,
        "two children became one, and nothing else changed"
    );
}

#[test]
fn hot_reload_bsn_base_updates_dependent() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("base.bsn"), "Position { x: 1.0, y: 1.0 }");
    // `derived` patches `Position` too, so it holds a copy-on-write snapshot of the base's
    // template. That snapshot is what a base edit would otherwise leave stale.
    dir.insert_asset_text(
        Path::new("derived.bsn"),
        ":\"base.bsn\"\nPosition { x: 2.0 }",
    );
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "derived.bsn");
    assert_eq!(
        *app.world().get::<Position>(root).unwrap(),
        Position {
            x: 2.0,
            y: 1.0,
            z: 0.0
        }
    );

    edit_and_save(&app, &dir, "base.bsn", "Position { x: 1.0, y: 9.0 }");
    run_app_until(&mut app, |app| {
        app.world().get::<Position>(root).unwrap().y == 9.0
    });

    assert_eq!(
        *app.world().get::<Position>(root).unwrap(),
        Position {
            x: 2.0,
            y: 9.0,
            z: 0.0
        },
        "editing a base must rebuild every dependent file's snapshot, while its own patch wins"
    );
}

#[test]
fn hot_reload_bsn_parse_error_keeps_previous_scene() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Position { x: 1.0 }\nChildren [ #A ]");
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    let child = app.world().get::<Children>(root).unwrap()[0];

    edit_and_save(&app, &dir, "a.bsn", "Position {\n  x: ,\n}\n");
    for _ in 0..20 {
        app.update();
        std::thread::sleep(core::time::Duration::from_millis(1));
    }

    assert_eq!(app.world().get::<Position>(root).unwrap().x, 1.0);
    assert!(
        app.world().get_entity(child).is_ok(),
        "a broken edit must leave the last good version rendering"
    );

    // ...and fixing the file brings it back.
    edit_and_save(&app, &dir, "a.bsn", "Position { x: 7.0 }\nChildren [ #B ]");
    run_app_until(&mut app, |app| child_names(app, root) == ["B"]);
    assert_eq!(app.world().get::<Position>(root).unwrap().x, 7.0);
}

/// `Entity` ids that a `.bsn` scene handed out — both to the outside world and to its own
/// `Reference` components — are replaced wholesale by a reload.
#[test]
fn entity_references_across_reload() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "#X Children [ (Reference(#X)) ]");
    let mut app = test_app(&dir);

    let root = spawn_instance(&mut app, "a.bsn");
    let held = app.world().get::<Children>(root).unwrap()[0];
    assert_eq!(app.world().get::<Reference>(held).unwrap().0, root);

    edit_and_save(&app, &dir, "a.bsn", "#X Children [ (Reference(#X)) ]");
    run_app_until(&mut app, |app| {
        app.world().get::<Children>(root).unwrap()[0] != held
    });

    // Documented state loss: the scene is rebuilt, so ids held elsewhere dangle.
    assert!(app.world().get_entity(held).is_err());
    // The rebuilt scene's own references are correct again, and still point at the *same*
    // instance entity, which is never despawned.
    let rebuilt = app.world().get::<Children>(root).unwrap()[0];
    assert_eq!(app.world().get::<Reference>(rebuilt).unwrap().0, root);
}

/// Editing a base file re-resolves *every* dependent instance, not just the first one — the
/// real-text mirror of `spawn.rs`'s `hot_reload_of_base_updates_dependent_file_instances`, which
/// drives the same path through `.fakescene` closures.
#[test]
fn hot_reload_bsn_base_updates_every_dependent_instance() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("b.bsn"), "Position { x: 1.0, y: 1.0 }");
    // `a.bsn` patches `Position` too, so each of its instances holds a copy-on-write snapshot of
    // the base's template. Those snapshots are what a base edit would otherwise leave stale.
    dir.insert_asset_text(Path::new("a.bsn"), ":\"b.bsn\"\nPosition { x: 2.0 }");
    let mut app = test_app(&dir);

    let dependents = [
        spawn_instance(&mut app, "a.bsn"),
        spawn_instance(&mut app, "a.bsn"),
    ];
    let base = spawn_instance(&mut app, "b.bsn");
    for root in dependents {
        assert_eq!(
            *app.world().get::<Position>(root).unwrap(),
            Position {
                x: 2.0,
                y: 1.0,
                z: 0.0
            }
        );
    }

    edit_and_save(&app, &dir, "b.bsn", "Position { x: 1.0, y: 9.0 }");
    run_app_until(&mut app, |app| {
        [dependents[0], dependents[1], base]
            .iter()
            .all(|root| app.world().get::<Position>(*root).unwrap().y == 9.0)
    });

    for root in dependents {
        assert_eq!(
            *app.world().get::<Position>(root).unwrap(),
            Position {
                x: 2.0,
                y: 9.0,
                z: 0.0
            },
            "every instance of the dependent file is rebuilt, and its own patch still wins"
        );
    }
    assert_eq!(
        *app.world().get::<Position>(base).unwrap(),
        Position {
            x: 1.0,
            y: 9.0,
            z: 0.0
        },
        "and the instance of the edited file itself is updated too"
    );
}

/// Two saves of the same file with no frame in between collapse into a single despawn/apply
/// cycle, with the last text winning — the real-text mirror of `spawn.rs`'s
/// `hot_reload_twice_in_one_frame_applies_once`.
#[test]
fn hot_reload_bsn_twice_in_one_frame_applies_once() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Children [ #A ]");
    let mut app = test_app(&dir);

    let before = app.world().entities().count_spawned();
    let root = spawn_instance(&mut app, "a.bsn");
    assert_eq!(child_names(&app, root), ["A"]);

    // Both edits land before the next `app.update()`, so both loader runs read the *second* text.
    edit_and_save(&app, &dir, "a.bsn", "Children [ #B ]");
    edit_and_save(&app, &dir, "a.bsn", "Children [ #C ]");
    run_app_until(&mut app, |app| child_names(app, root) == ["C"]);
    for _ in 0..10 {
        app.update();
        std::thread::sleep(core::time::Duration::from_millis(1));
    }

    assert_eq!(
        child_names(&app, root),
        ["C"],
        "the last save is the one that sticks"
    );
    assert_eq!(
        app.world().entities().count_spawned(),
        before + 2,
        "the instance entity and one child, with no generation left orphaned behind them"
    );
}

/// Re-inserting `ScenePatchInstance` — the natural way to swap which scene an entity is an
/// instance of — applies the new file's content and despawns the previous generation rather than
/// orphaning it. The real-text mirror of `spawn.rs`'s ADV-3.
#[test]
fn reinserting_scene_patch_instance_swaps_the_bsn_file() {
    let dir = Dir::default();
    dir.insert_asset_text(Path::new("a.bsn"), "Children [ #A, #B ]");
    dir.insert_asset_text(Path::new("b.bsn"), "Children [ #C ]");
    let mut app = test_app(&dir);

    let before = app.world().entities().count_spawned();
    let root = spawn_instance(&mut app, "a.bsn");
    assert_eq!(child_names(&app, root), ["A", "B"]);
    let previous: Vec<Entity> = app
        .world()
        .get::<Children>(root)
        .unwrap()
        .iter()
        .copied()
        .collect();

    let b = app
        .world()
        .resource::<AssetServer>()
        .load::<ScenePatch>("b.bsn");
    app.world_mut()
        .entity_mut(root)
        .remove::<ScenePatchInstance>();
    app.world_mut()
        .entity_mut(root)
        .insert(ScenePatchInstance(b));
    run_app_until(&mut app, |app| child_names(app, root) == ["C"]);

    for entity in previous {
        assert!(
            app.world().get_entity(entity).is_err(),
            "swapping the scene must not orphan the previous generation"
        );
    }
    assert_eq!(
        app.world().entities().count_spawned(),
        before + 2,
        "the instance entity and the one child of the new scene"
    );
}
