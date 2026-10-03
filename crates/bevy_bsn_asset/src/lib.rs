#![expect(
    unsafe_code,
    reason = "`DynamicComponentTemplate` implements `ErasedTemplate`'s unsafe methods."
)]
//! The reflection-driven `.bsn` scene format: turning a parsed `.bsn` document ([`BsnDocument`])
//! into a [`Scene`](bevy_scene::Scene) that merges with statically-defined
//! [`bsn!`](bevy_scene::bsn) scenes.
//!
//! The entry point is [`DynamicScene::from_document`], which lowers a document into a
//! [`DynamicScene`] using an [`AppTypeRegistry`](bevy_ecs::reflect::AppTypeRegistry). All symbol
//! resolution, literal conversion and type-data lookup happens **once**, at that point, so
//! resolving and spawning the scene later performs no registry lookups and every failure that
//! depends on the registry is reported up front, with a source [`Span`](bevy_bsn::Span).
//!
//! ## How a document becomes components
//!
//! 1. Each `Type { … }` entry is resolved to the *template* type of the named component
//!    (through [`ReflectFromTemplate`](bevy_ecs::reflect::ReflectFromTemplate)), and the entry's
//!    fields become a partial reflected value.
//! 2. At resolve time each entry gets (or creates) the template slot keyed by that **real**
//!    template [`TypeId`](core::any::TypeId) and applies its partial value on top — the same slot
//!    a `bsn!` patch of the same component would use, which is what makes the two merge.
//! 3. At spawn time the template value is built (through
//!    [`ReflectTemplate`](bevy_ecs::reflect::ReflectTemplate) when the output type differs) and
//!    pushed into the entity's single `BundleWriter`, so an entity spawned from a `.bsn` document
//!    still performs exactly one archetype move.
//!
//! [`BsnDocument`]: bevy_bsn::BsnDocument

extern crate alloc;

mod build;
mod loader;
mod scene;
mod template;
mod value;

pub use build::{DynamicSceneBuildError, HandleProvider};
pub use loader::{report_scene_patch_load_failures, DynamicBsnLoader, DynamicBsnLoaderError};
pub use scene::DynamicScene;
pub use template::DynamicComponentTemplate;

#[cfg(test)]
mod attack;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

use bevy_app::{App, Plugin, SceneSpawnerSystems, SpawnScene};
use bevy_asset::AssetApp;
use bevy_ecs::schedule::IntoScheduleConfigs;

/// Registers [`DynamicBsnLoader`], making `.bsn` files loadable as
/// [`ScenePatch`](bevy_scene::ScenePatch) assets.
///
/// Requires [`ScenePlugin`](bevy_scene::ScenePlugin) (and therefore
/// [`AssetPlugin`](bevy_asset::AssetPlugin)) to be added first. Included in `DefaultPlugins` when
/// the `bsn_asset` cargo feature is enabled (it is by default).
#[derive(Default)]
pub struct BsnAssetPlugin;

impl Plugin for BsnAssetPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset_loader::<DynamicBsnLoader>().add_systems(
            SpawnScene,
            report_scene_patch_load_failures
                .in_set(SceneSpawnerSystems::SceneSpawn)
                // The `after` edge mirrors the one `ScenePlugin` puts on
                // `(resolve_scene_patches, spawn_queued)`: membership in the `SceneSpawn` set does
                // not order this system against `WorldInstanceSpawn`'s exclusive spawner, and
                // leaving it unordered is a schedule ambiguity (caught by the
                // `ambiguity_detection` CI example).
                .after(SceneSpawnerSystems::WorldInstanceSpawn)
                .before(bevy_scene::resolve_scene_patches),
        );
    }
}
