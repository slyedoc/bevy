//! Harness shared by the crate's `#[cfg(test)]` suites.
//!
//! The lib tests in `lib.rs` and `spawn.rs` all need the
//! same two things: an [`App`] carrying the asset and scene plugins, and a bounded pump that
//! turns "never reached the expected state" into a failure rather than a hang.
//!
//! The integration tests in `tests/` keep their own copies on purpose: sharing this module across
//! the lib/integration boundary would mean exporting it publicly behind a feature, which is not
//! worth it for three functions.

use bevy_app::{App, TaskPoolPlugin};
use bevy_asset::{
    io::{
        memory::{Dir, MemoryAssetReader},
        AssetSourceBuilder, AssetSourceId,
    },
    AssetApp, AssetPlugin,
};
use bevy_ecs::{
    hierarchy::{ChildOf, Children},
    name::Name,
    reflect::AppTypeRegistry,
};

use crate::ScenePlugin;

/// The plugins every test app needs: a task pool to run load tasks on, the asset server, and the
/// scene systems themselves.
///
/// The type registry is replaced with an **empty** one before the plugins run, so the tests see
/// exactly the types they register — and nothing else. Without this, builds where
/// `reflect_auto_register` ends up enabled by feature unification (a `--workspace` test run
/// unifies against the root `bevy` crate's defaults) auto-register every `#[derive(Reflect)]`
/// fixture in this test binary; the same-named fixtures declared by different test modules
/// (`Position`, `Marker`, …) then make every short-type-path lookup ambiguous, and 28 tests fail
/// on CI while passing in a `-p bevy_scene` run. Tests must not depend on which features the
/// build happened to unify.
fn add_scene_plugins(app: &mut App) {
    app.insert_resource(AppTypeRegistry::default());
    {
        // The handful of real engine types the suites rely on; everything else is registered by
        // each suite's own fixture-registration helper.
        let registry = app.world().resource::<AppTypeRegistry>();
        let mut registry = registry.write();
        registry.register::<Name>();
        registry.register::<Children>();
        registry.register::<ChildOf>();
    }
    app.add_plugins((
        TaskPoolPlugin::default(),
        AssetPlugin::default(),
        ScenePlugin,
    ));
}

/// An [`App`] with the asset and scene plugins, and nothing else.
pub(crate) fn test_app() -> App {
    let mut app = App::new();
    add_scene_plugins(&mut app);
    app
}

/// An [`App`] like [`test_app`], serving the returned [`Dir`] as the default asset source.
///
/// The source has to be registered *before* `AssetPlugin`, or the plugin logs an error and the
/// registration is ignored — hence one function rather than a step a caller can forget.
/// `Dir::insert_asset_text` overwrites, so writing the same path twice is "edit the file".
pub(crate) fn memory_asset_app() -> (App, Dir) {
    let mut app = App::new();
    let dir = Dir::default();
    let reader_dir = dir.clone();
    app.register_asset_source(
        AssetSourceId::Default,
        AssetSourceBuilder::new(move || {
            Box::new(MemoryAssetReader {
                root: reader_dir.clone(),
            })
        }),
    );
    add_scene_plugins(&mut app);
    (app, dir)
}

/// Pumps `app` until `predicate` holds, or panics.
///
/// The bound is what makes the "never loads" and "never reloads" failure modes show up as a test
/// failure instead of a hang.
pub(crate) fn run_app_until(app: &mut App, mut predicate: impl FnMut(&mut App) -> bool) {
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
