use crate::{
    ApplySceneError, ResolvedSceneRoot, Scene, SceneInstanceState, SceneList, SceneListPatch,
    ScenePatch, ScenePatchInstance, SpawnSceneError,
};
use alloc::sync::Arc;
use bevy_asset::{AssetEvent, AssetId, AssetPath, AssetServer, Assets, Handle};
use bevy_ecs::{
    bundle::BundleScratch, message::MessageCursor, prelude::*, reflect::AppTypeRegistry,
    relationship::Relationship,
};
use bevy_platform::collections::{HashMap, HashSet};
use tracing::error;

/// Adds scene spawning functionality to [`World`].
pub trait WorldSceneExt {
    /// Spawns the given [`Scene`] immediately. This will resolve the Scene (using [`Scene::resolve`]). If that fails (for example, if there are dependencies that have not been
    /// loaded yet), it will return a [`SpawnSceneError`]. If resolving the [`Scene`] is successful, the scene will be spawned.
    ///
    /// If resolving and spawning is successful, it will return a new [`EntityWorldMut`] containing the full contents of the spawned scene.
    ///
    /// See [`Scene`] for the features of the scene system (and how to use it).
    ///
    /// If your scene has a dependency that might not be loaded yet (for example, it includes a `.bsn` asset file), consider using [`World::queue_spawn_scene`].
    ///
    /// ```
    /// # use bevy_app::App;
    /// # use bevy_scene::{prelude::*, ScenePlugin};
    /// # use bevy_ecs::prelude::*;
    /// # use bevy_asset::AssetPlugin;
    /// # use bevy_app::TaskPoolPlugin;
    /// # let mut app = App::new();
    /// # app.add_plugins((
    /// #     TaskPoolPlugin::default(),
    /// #     AssetPlugin::default(),
    /// #     ScenePlugin::default(),
    /// # ));
    /// # let world = app.world_mut();
    /// #[derive(Component, Default, Clone)]
    /// struct Score(usize);
    ///
    /// #[derive(Component, Default, Clone)]
    /// struct Sword;
    ///
    /// #[derive(Component, Default, Clone)]
    /// struct Shield;
    ///
    /// world.spawn_scene(bsn! {
    ///     #Player
    ///     Score(0)
    ///     Children [
    ///         Sword,
    ///         Shield,
    ///     ]
    /// }).unwrap();
    /// ```
    fn spawn_scene<S: Scene>(&mut self, scene: S) -> Result<EntityWorldMut<'_>, SpawnSceneError>;

    /// Queues the `scene` to be spawned. This will evaluate the `scene`'s dependencies (via [`Scene::register_dependencies`]) and queue it to be resolved and spawned
    /// after all of the dependencies have been loaded. If a [`SpawnSceneError`] occurs, it will be logged as an error.
    ///
    /// If the dependencies are already loaded (or there are no dependencies), then the scene will be spawned this frame.
    ///
    /// See [`Scene`] for the features of the scene system (and how to use it).
    ///
    /// ```
    /// # use bevy_app::App;
    /// # use bevy_scene::{prelude::*, ScenePlugin};
    /// # use bevy_ecs::prelude::*;
    /// # use bevy_asset::AssetPlugin;
    /// # use bevy_app::TaskPoolPlugin;
    /// # let mut app = App::new();
    /// # app.add_plugins((
    /// #     TaskPoolPlugin::default(),
    /// #     AssetPlugin::default(),
    /// #     ScenePlugin::default(),
    /// # ));
    /// # let world = app.world_mut();
    /// #[derive(Component, Default, Clone)]
    /// struct Score(usize);
    ///
    /// #[derive(Component, Default, Clone)]
    /// struct Sword;
    ///
    /// #[derive(Component, Default, Clone)]
    /// struct Shield;
    ///
    /// // This scene includes the "player.bsn" asset. It will be spawned on the frame that "player.bsn"
    /// // is fully loaded.
    /// world.queue_spawn_scene(bsn! {
    ///     :"player.bsn"
    ///     #Player
    ///     Score(0)
    ///     Children [
    ///         Sword,
    ///         Shield,
    ///     ]
    /// });
    /// ```
    fn queue_spawn_scene<S: Scene>(&mut self, scene: S) -> EntityWorldMut<'_>;

    /// Spawns the given [`SceneList`] immediately. This will resolve the scene list (using [`SceneList::resolve_list`]). If that fails (for example, if there are dependencies that have not been
    /// loaded yet), it will return a [`SpawnSceneError`]. If resolving the [`SceneList`] is successful, the scene list will be spawned.
    ///
    /// If resolving and spawning is successful, it will return a [`Vec<Entity>`] containing each entity described in the [`SceneList`].
    ///
    /// See [`Scene`] for the features of the scene system (and how to use it).
    ///
    /// If your scene list has a dependency that might not be loaded yet (for example, it includes a `.bsn` asset file), consider using [`World::queue_spawn_scene_list`].
    ///
    /// ```
    /// # use bevy_app::App;
    /// # use bevy_scene::{prelude::*, ScenePlugin};
    /// # use bevy_ecs::prelude::*;
    /// # use bevy_asset::AssetPlugin;
    /// # use bevy_app::TaskPoolPlugin;
    /// # let mut app = App::new();
    /// # app.add_plugins((
    /// #     TaskPoolPlugin::default(),
    /// #     AssetPlugin::default(),
    /// #     ScenePlugin::default(),
    /// # ));
    /// # let world = app.world_mut();
    /// #[derive(Component, Default, Clone)]
    /// enum Team {
    ///     #[default]
    ///     Red,
    ///     Blue,
    /// }
    ///
    /// world.spawn_scene_list(bsn_list! {
    ///     #Player1
    ///     Team::Red
    ///     --
    ///     #Player2
    ///     Team::Blue
    /// }).unwrap();
    /// ```
    // PERF: ideally this is an iterator
    fn spawn_scene_list<L: SceneList>(&mut self, scenes: L)
        -> Result<Vec<Entity>, SpawnSceneError>;

    /// Queues the `scene_list` to be spawned. This will evaluate the `scene_list`'s dependencies (via [`Scene::register_dependencies`]) and queue it to be resolved
    /// and spawned after all of the dependencies have been loaded. If a [`SpawnSceneError`] occurs, it will be logged as an error.
    ///
    /// If the dependencies are already loaded (or there are no dependencies), then the scene list will be spawned this frame.
    /// ```
    /// # use bevy_app::App;
    /// # use bevy_scene::{prelude::*, ScenePlugin};
    /// # use bevy_ecs::prelude::*;
    /// # use bevy_asset::AssetPlugin;
    /// # use bevy_app::TaskPoolPlugin;
    /// # let mut app = App::new();
    /// # app.add_plugins((
    /// #     TaskPoolPlugin::default(),
    /// #     AssetPlugin::default(),
    /// #     ScenePlugin::default(),
    /// # ));
    /// # let world = app.world_mut();
    /// #[derive(Component, Default, Clone)]
    /// enum Team {
    ///     #[default]
    ///     Red,
    ///     Blue,
    /// }
    /// // This scene list includes the "player.bsn" asset. It will be spawned on the frame that "player.bsn"
    /// // is loaded.
    /// world.queue_spawn_scene_list(bsn_list! {
    ///     :"player.bsn"
    ///     #Player1
    ///     Team::Red
    ///     --
    ///     :"player.bsn"
    ///     #Player2
    ///     Team::Blue
    /// });
    /// ```
    fn queue_spawn_scene_list<L: SceneList>(&mut self, scenes: L);
}

/// Loads `scene` and resolves it immediately, against the world's [`AppTypeRegistry`] if it has one.
///
/// Returning the patch by value is what bounds the registry read guard: it is dropped before the
/// caller can take the `&mut World` it needs to spawn. A [`Scene::resolve`] impl must not take a
/// **write** lock on the registry while the guard is alive.
fn load_and_resolve<S: Scene>(world: &World, scene: S) -> Result<ScenePatch, SpawnSceneError> {
    let assets = world.resource::<AssetServer>();
    let mut patch = ScenePatch::load(assets, scene);
    let type_registry = world.get_resource::<AppTypeRegistry>();
    let type_registry = type_registry.map(|registry| registry.read());
    patch.resolve(
        assets,
        world.resource::<Assets<ScenePatch>>(),
        type_registry.as_deref(),
    )?;
    Ok(patch)
}

/// [`load_and_resolve`] for a [`SceneList`].
fn load_and_resolve_list<L: SceneList>(
    world: &World,
    scenes: L,
) -> Result<SceneListPatch, SpawnSceneError> {
    let assets = world.resource::<AssetServer>();
    let mut patch = SceneListPatch::load(assets, scenes);
    let type_registry = world.get_resource::<AppTypeRegistry>();
    let type_registry = type_registry.map(|registry| registry.read());
    patch.resolve(
        assets,
        world.resource::<Assets<ScenePatch>>(),
        type_registry.as_deref(),
    )?;
    Ok(patch)
}

impl WorldSceneExt for World {
    fn spawn_scene<S: Scene>(&mut self, scene: S) -> Result<EntityWorldMut<'_>, SpawnSceneError> {
        load_and_resolve(self, scene)?.spawn(self)
    }

    fn queue_spawn_scene<S: Scene>(&mut self, scene: S) -> EntityWorldMut<'_> {
        let assets = self.resource::<AssetServer>();
        let patch = ScenePatch::load(assets, scene);
        let handle = assets.add(patch);
        // Inserting the component (rather than pushing to `QueuedScenes` by hand) queues the spawn
        // through `on_insert_scene_patch_instance`, which does exactly the same push — and makes the
        // entity hot-reload-tracked, which matters for scenes that include a `.bsn` base.
        self.spawn(ScenePatchInstance(handle))
    }

    fn spawn_scene_list<L: SceneList>(
        &mut self,
        scenes: L,
    ) -> Result<Vec<Entity>, SpawnSceneError> {
        load_and_resolve_list(self, scenes)?.spawn(self)
    }

    fn queue_spawn_scene_list<L: SceneList>(&mut self, scenes: L) {
        let assets = self.resource::<AssetServer>();
        let patch = SceneListPatch::load(assets, scenes);
        let handle = assets.add(patch);
        self.resource_mut::<QueuedScenes>()
            .scene_list_spawns
            .push(handle);
    }
}

/// Adds scene spawning functionality to [`Commands`].
pub trait CommandsSceneExt {
    /// Spawns the given [`Scene`] as soon as [`Commands`] are applied. This will resolve the Scene (using [`Scene::resolve`]). If that fails (for example, if there are dependencies that have not been
    /// loaded yet), it will log a [`SpawnSceneError`] as an error. If resolving the [`Scene`] is successful, the scene will be spawned.
    ///
    /// This is essentially a [`Command`] that runs [`World::spawn_scene`].
    ///
    /// See [`Scene`] for the features of the scene system (and how to use it).
    ///
    /// If your scene has a dependency that might not be loaded yet (for example, it includes a `.bsn` asset file), consider using [`Commands::queue_spawn_scene`].
    ///
    /// ```
    /// # use bevy_scene::prelude::*;
    /// # use bevy_ecs::prelude::*;
    /// # let mut world = World::new();
    /// # let mut commands = world.commands();
    /// #[derive(Component, Default, Clone)]
    /// struct Score(usize);
    ///
    /// #[derive(Component, Default, Clone)]
    /// struct Sword;
    ///
    /// #[derive(Component, Default, Clone)]
    /// struct Shield;
    ///
    /// commands.spawn_scene(bsn! {
    ///     #Player
    ///     Score(0)
    ///     Children [
    ///         Sword,
    ///         Shield,
    ///     ]
    /// });
    /// ```
    fn spawn_scene<S: Scene>(&mut self, scene: S) -> EntityCommands<'_>;

    /// Queues the `scene` to be spawned. This will evaluate the `scene`'s dependencies (via [`Scene::register_dependencies`]) and queue it to be resolved and spawned
    /// after all of the dependencies have been loaded. If a [`SpawnSceneError`] occurs, it will be logged as an error.
    ///
    /// If the dependencies are already loaded (or there are no dependencies), then the scene will be spawned this frame.
    ///
    /// See [`Scene`] for the features of the scene system (and how to use it).
    ///
    /// ```
    /// # use bevy_scene::prelude::*;
    /// # use bevy_ecs::prelude::*;
    /// # let mut world = World::new();
    /// # let mut commands = world.commands();
    /// #[derive(Component, Default, Clone)]
    /// struct Score(usize);
    ///
    /// #[derive(Component, Default, Clone)]
    /// struct Sword;
    ///
    /// #[derive(Component, Default, Clone)]
    /// struct Shield;
    ///
    /// // This scene includes the "player.bsn" asset. It will be spawned on the frame that "player.bsn"
    /// // is fully loaded.
    /// commands.queue_spawn_scene(bsn! {
    ///     :"player.bsn"
    ///     #Player
    ///     Score(0)
    ///     Children [
    ///         Sword,
    ///         Shield,
    ///     ]
    /// });
    /// ```
    fn queue_spawn_scene<S: Scene>(&mut self, scene: S) -> EntityCommands<'_>;

    /// Spawns the given [`SceneList`] as soon as [`Commands`] are applied. This will resolve the scene list (using [`SceneList::resolve_list`]). If that fails (for example, if there are dependencies that have not been
    /// loaded yet), it will log a [`SpawnSceneError`] as an error. If resolving the [`Scene`] is successful, the scene list will be spawned.
    ///
    /// This is essentially a [`Command`] that performs [`World::spawn_scene_list`].
    ///
    /// See [`Scene`] for the features of the scene system (and how to use it).
    ///
    /// If your scene list has a dependency that might not be loaded yet (for example, it includes a `.bsn` asset file), consider using [`Commands::queue_spawn_scene_list`].
    ///
    /// ```
    /// # use bevy_scene::prelude::*;
    /// # use bevy_ecs::prelude::*;
    /// # let mut world = World::new();
    /// # let mut commands = world.commands();
    /// #[derive(Component, Default, Clone)]
    /// enum Team {
    ///     #[default]
    ///     Red,
    ///     Blue,
    /// }
    ///
    /// commands.spawn_scene_list(bsn_list! {
    ///     :"player.bsn"
    ///     #Player1
    ///     Team::Red
    ///     --
    ///     :"player.bsn"
    ///     #Player2
    ///     Team::Blue
    /// });
    /// ```
    fn spawn_scene_list<L: SceneList>(&mut self, scenes: L);

    /// Queues the `scene_list` to be spawned. This will evaluate the `scene_list`'s dependencies (via [`Scene::register_dependencies`]) and queue it to be resolved
    /// and spawned after all of the dependencies have been loaded. If a [`SpawnSceneError`] occurs, it will be logged as an error.
    ///
    /// If the dependencies are already loaded (or there are no dependencies), then the scene will be spawned this frame.
    ///
    /// ```
    /// # use bevy_scene::prelude::*;
    /// # use bevy_ecs::prelude::*;
    /// # let mut world = World::new();
    /// # let mut commands = world.commands();
    /// #[derive(Component, Default, Clone)]
    /// enum Team {
    ///     #[default]
    ///     Red,
    ///     Blue,
    /// }
    ///
    /// // This scene list includes the "player.bsn" asset. It will be spawned on the frame that "player.bsn"
    /// // is loaded.
    /// commands.queue_spawn_scene_list(bsn_list! {
    ///     :"player.bsn"
    ///     #Player1
    ///     Team::Red
    ///     --
    ///     :"player.bsn"
    ///     #Player2
    ///     Team::Blue
    /// });
    /// ```
    fn queue_spawn_scene_list<L: SceneList>(&mut self, scenes: L);
}

impl<'w, 's> CommandsSceneExt for Commands<'w, 's> {
    fn spawn_scene<S: Scene>(&mut self, scene: S) -> EntityCommands<'_> {
        let mut entity_commands = self.spawn_empty();
        let id = entity_commands.id();
        entity_commands.commands().queue(move |world: &mut World| {
            if let Ok(mut entity) = world.get_entity_mut(id)
                && let Err(err) = entity.apply_scene(scene)
            {
                error!("{err}");
            }
        });
        entity_commands
    }

    fn queue_spawn_scene<S: Scene>(&mut self, scene: S) -> EntityCommands<'_> {
        let mut entity_commands = self.spawn_empty();
        let id = entity_commands.id();
        entity_commands.commands().queue(move |world: &mut World| {
            if let Ok(mut entity) = world.get_entity_mut(id) {
                entity.queue_apply_scene(scene);
            }
        });
        entity_commands
    }

    fn spawn_scene_list<L: SceneList>(&mut self, scenes: L) {
        self.queue(move |world: &mut World| {
            if let Err(err) = world.spawn_scene_list(scenes) {
                error!("{err}");
            }
        });
    }

    fn queue_spawn_scene_list<L: SceneList>(&mut self, scenes: L) {
        self.queue(move |world: &mut World| {
            world.queue_spawn_scene_list(scenes);
        });
    }
}

/// Adds scene functionality to [`EntityWorldMut`].
pub trait EntityWorldMutSceneExt {
    /// Spawns a [`SceneList`], where each entity is related to the current entity using [`RelationshipTarget::Relationship`].
    ///
    /// This will evaluate the `scene_list`'s dependencies (via [`SceneList::register_dependencies`]) and queue it to be resolved
    /// and spawned after all of the dependencies have been loaded. If a [`SpawnSceneError`] occurs, it will be logged as an error.
    ///
    /// If the dependencies are already loaded (or there are no dependencies), then the scene list will be spawned this frame.
    ///
    /// ```
    /// # use bevy_app::App;
    /// # use bevy_scene::{prelude::*, ScenePlugin};
    /// # use bevy_ecs::prelude::*;
    /// # use bevy_asset::AssetPlugin;
    /// # use bevy_app::TaskPoolPlugin;
    /// # let mut app = App::new();
    /// # app.add_plugins((
    /// #     TaskPoolPlugin::default(),
    /// #     AssetPlugin::default(),
    /// #     ScenePlugin::default(),
    /// # ));
    /// # let world = app.world_mut();
    /// #[derive(Component, Default, Clone)]
    /// enum Team {
    ///     #[default]
    ///     Red,
    ///     Blue,
    /// }
    ///
    /// world.spawn_empty().queue_spawn_related_scenes::<Children>(bsn_list! {
    ///     #Player1
    ///     Team::Red
    ///     --
    ///     #Player2
    ///     Team::Blue
    /// });
    /// ```
    fn queue_spawn_related_scenes<T: RelationshipTarget>(self, scenes: impl SceneList) -> Self;

    /// Applies the given [`Scene`] to the current entity immediately. This will resolve the Scene (using [`Scene::resolve`]). If that fails (for example, if there are dependencies that have not been
    /// loaded yet), it will return a [`SpawnSceneError`]. If resolving the [`Scene`] is successful, the scene will be spawned.
    ///
    /// When a scene is resolved, it will replace and orphan the current entity's children.
    ///
    /// If resolving and spawning is successful, the entity will contain the full contents of the spawned scene.
    ///
    /// This will write directly on top of any existing components on the entity. [`Scene`] is generally used as a spawning mechanism, so for most things, prefer using [`World::spawn_scene`].
    ///
    /// See [`Scene`] for the features of the scene system (and how to use it).
    ///
    /// If your scene has a dependency that might not be loaded yet (for example, it includes a `.bsn` asset file), consider using [`World::queue_spawn_scene`].
    fn apply_scene<S: Scene>(&mut self, scene: S) -> Result<(), SpawnSceneError>;

    /// Queues the `scene` to be applied. This will evaluate the `scene`'s dependencies (via [`Scene::register_dependencies`]) and queue it to be resolved and spawned
    /// after all of the dependencies have been loaded. If a [`SpawnSceneError`] occurs, it will be logged as an error.
    ///
    /// See [`EntityWorldMutSceneExt::apply_scene`] for more information on what happens when a scene is resolved.
    ///
    /// If the dependencies are already loaded (or there are no dependencies), then the scene will be spawned this frame.
    /// This will write directly on top of any existing components on the entity. [`Scene`] is generally used as a spawning mechanism, so for most things, prefer using [`World::queue_spawn_scene`].
    ///
    /// See [`Scene`] for the features of the scene system (and how to use it).
    fn queue_apply_scene<S: Scene>(&mut self, scene: S);
}

impl EntityWorldMutSceneExt for EntityWorldMut<'_> {
    fn queue_spawn_related_scenes<T: RelationshipTarget>(mut self, scenes: impl SceneList) -> Self {
        let assets = self.resource::<AssetServer>();
        let patch = SceneListPatch::load(assets, scenes);
        let handle = assets.add(patch);
        let entity = self.id();
        self.resource_mut::<QueuedScenes>()
            .related_scene_list_spawns
            .push((
                RelatedSceneListSpawn {
                    entity,
                    insert: |entity, target| {
                        entity.insert(
                            <<T as RelationshipTarget>::Relationship as Relationship>::from(target),
                        );
                    },
                },
                handle,
            ));
        self
    }

    fn apply_scene<S: Scene>(&mut self, scene: S) -> Result<(), SpawnSceneError> {
        load_and_resolve(self.world(), scene)?.apply(self)
    }

    fn queue_apply_scene<S: Scene>(&mut self, scene: S) {
        let assets = self.resource::<AssetServer>();
        let patch = ScenePatch::load(assets, scene);
        let handle = assets.add(patch);
        // See `World::queue_spawn_scene` for why this inserts a component instead of pushing to
        // `QueuedScenes` directly.
        self.insert(ScenePatchInstance(handle));
    }
}

/// Adds scene functionality to [`EntityWorldMut`].
pub trait EntityCommandsSceneExt {
    /// Spawns a [`SceneList`], where each entity is related to the current entity using [`RelationshipTarget::Relationship`].
    ///
    /// This will evaluate the `scene_list`'s dependencies (via [`SceneList::register_dependencies`]) and queue it to be resolved
    /// and spawned after all of the dependencies have been loaded. If a [`SpawnSceneError`] occurs, it will be logged as an error.
    ///
    /// If the dependencies are already loaded (or there are no dependencies), then the scene list will be spawned this frame.
    ///
    /// ```
    /// # use bevy_app::App;
    /// # use bevy_scene::prelude::*;
    /// # use bevy_ecs::prelude::*;
    /// # use bevy_asset::AssetPlugin;
    /// # use bevy_app::TaskPoolPlugin;
    /// # let mut app = App::new();
    /// # let mut commands = app.world_mut().commands();
    /// #[derive(Component, Default, Clone)]
    /// enum Team {
    ///     #[default]
    ///     Red,
    ///     Blue,
    /// }
    ///
    /// commands.spawn_empty().queue_spawn_related_scenes::<Children>(bsn_list! {
    ///     #Player1
    ///     Team::Red
    ///     --
    ///     #Player2
    ///     Team::Blue
    /// });
    /// ```
    fn queue_spawn_related_scenes<T: RelationshipTarget>(
        &mut self,
        scenes: impl SceneList,
    ) -> &mut Self;

    /// Applies the given [`Scene`] to the current entity as soon as [`Commands`] are applied. This will resolve the Scene (using [`Scene::resolve`]). If that fails (for example, if there are dependencies that have not been
    /// loaded yet), it will log a [`SpawnSceneError`] as an error. If resolving the [`Scene`] is successful, the scene will be spawned.
    ///
    /// If resolving and spawning is successful, the entity will contain the full contents of the spawned scene.
    ///
    /// This will write directly on top of any existing components on the entity. [`Scene`] is generally used as a spawning mechanism, so for most things, prefer using [`Commands::spawn_scene`].
    ///
    /// See [`Scene`] for the features of the scene system (and how to use it).
    ///
    /// If your scene has a dependency that might not be loaded yet (for example, it includes a `.bsn` asset file), consider using [`Commands::spawn_scene`].
    fn apply_scene<S: Scene>(&mut self, scene: S) -> &mut Self;

    /// Queues the `scene` to be applied. This will evaluate the `scene`'s dependencies (via [`Scene::register_dependencies`]) and queue it to be resolved and spawned
    /// after all of the dependencies have been loaded. If a [`SpawnSceneError`] occurs, it will be logged as an error.
    ///
    /// If the dependencies are already loaded (or there are no dependencies), then the scene will be spawned this frame.
    /// This will write directly on top of any existing components on the entity. [`Scene`] is generally used as a spawning mechanism, so for most things, prefer using [`Commands::queue_spawn_scene`].
    ///
    /// See [`Scene`] for the features of the scene system (and how to use it).
    fn queue_apply_scene<S: Scene>(&mut self, scene: S) -> &mut Self;
}

impl EntityCommandsSceneExt for EntityCommands<'_> {
    fn queue_spawn_related_scenes<T: RelationshipTarget>(
        &mut self,
        scenes: impl SceneList,
    ) -> &mut Self {
        self.queue(move |entity: EntityWorldMut| {
            entity.queue_spawn_related_scenes::<T>(scenes);
        });
        self
    }

    fn apply_scene<S: Scene>(&mut self, scene: S) -> &mut Self {
        self.queue(move |mut entity: EntityWorldMut| entity.apply_scene(scene));
        self
    }

    fn queue_apply_scene<S: Scene>(&mut self, scene: S) -> &mut Self {
        self.queue(move |mut entity: EntityWorldMut| entity.queue_apply_scene(scene));
        self
    }
}

/// A [`System`] that resolves [`ScenePatch`] and [`SceneListPatch`] assets whose dependencies have been fully loaded.
///
/// This is also the hot-reload entry point. When a `.bsn` file changes on disk, the asset server
/// re-runs its loader and replaces the whole [`ScenePatch`] value with a fresh one — carrying a
/// fresh [`ScenePatch::scene`] and a cleared [`ScenePatch::resolved`] — then re-fires
/// [`AssetEvent::LoadedWithDependencies`]. So the same code path that resolves a first load
/// re-resolves a reload, with no retained source and no extra asset state. [`AssetEvent::Modified`]
/// is deliberately *not* used as the trigger: it reaches this schedule a frame later than
/// `LoadedWithDependencies`, and `Assets::get_mut` emits one on every resolve, so it cannot
/// distinguish a real edit from this system's own bookkeeping.
pub fn resolve_scene_patches(
    mut events: MessageReader<AssetEvent<ScenePatch>>,
    mut list_events: MessageReader<AssetEvent<SceneListPatch>>,
    assets: Res<AssetServer>,
    mut patches: ResMut<Assets<ScenePatch>>,
    mut list_patches: ResMut<Assets<SceneListPatch>>,
    mut waiting: ResMut<WaitingScenes>,
    type_registry: Option<Res<AppTypeRegistry>>,
    mut resolved_once: Local<HashSet<AssetId<ScenePatch>>>,
) {
    // Held across every `resolve` below. `Scene::resolve` impls must not write-lock the registry.
    let type_registry_guard = type_registry.as_ref().map(|registry| registry.read());
    let type_registry = type_registry_guard.as_deref();
    for event in events.read() {
        match *event {
            AssetEvent::LoadedWithDependencies { id } => {
                // `insert` returns false when the id was already present, which — since
                // `LoadedWithDependencies` fires exactly once per (re)load of an asset — means
                // this is a reload rather than a first load.
                let is_reload = !resolved_once.insert(id);
                // `get_mut_untracked` rather than `get_mut`: storing the resolved scene is
                // bookkeeping, not a change to what the asset means, and `get_mut`'s drop guard
                // would queue an `AssetEvent::Modified` on every single resolve.
                if let Some(scene) = patches.get_mut_untracked(id).and_then(|p| p.scene.take()) {
                    match ResolvedSceneRoot::resolve(scene, &assets, &patches, type_registry) {
                        Ok(resolved) => {
                            let patch = patches.get_mut_untracked(id).unwrap();
                            patch.resolved = Some(Arc::new(resolved));
                            if is_reload {
                                reload_dependents(id, &assets, &patches);
                            }
                        }
                        // A `.bsn` file that no longer parses (or no longer resolves) leaves the
                        // new value with `resolved: None`, and every live instance keeps rendering
                        // the last good version.
                        Err(err) => error!("Failed to resolve scene {id}: {err}"),
                    }
                }
            }
            AssetEvent::Removed { id } => {
                resolved_once.remove(&id);
                if let Some(waiting_entities) = waiting.scene_entities.remove(&id)
                    && !waiting_entities.is_empty()
                {
                    error!(
                        "Failed to spawn entities waiting for scene {id:?} because it was removed: {waiting_entities:?}"
                    );
                }
            }
            _ => {}
        }
    }
    for event in list_events.read() {
        match *event {
            AssetEvent::LoadedWithDependencies { id } => {
                if let Some(mut list_patch) = list_patches.get_mut(id)
                    && let Err(err) = list_patch.resolve(&assets, &patches, type_registry)
                {
                    error!("Failed to resolve scene list {id}: {err}");
                }
            }
            AssetEvent::Removed { id } => {
                if let Some(waiting_scene_lists) = waiting.scene_list_spawns.remove(&id)
                    && waiting_scene_lists > 0
                {
                    error!(
                        "Failed to spawn scene list {id:?} {waiting_scene_lists} times because it was removed."
                    );
                }

                if let Some(waiting_related) = waiting.related_list_entities.remove(&id)
                    && !waiting_related.is_empty()
                {
                    let waiting_entities =
                        waiting_related.iter().map(|r| r.entity).collect::<Vec<_>>();
                    error!(
                        "Failed to spawn related entities for scene list {id:?} because it was removed: {waiting_entities:?}"
                    );
                }
            }
            _ => {}
        }
    }
}

/// Force-reloads every [`ScenePatch`] asset that lists `changed` as a dependency — i.e. every
/// `.bsn` file that includes it as a cached base with `:"base.bsn"` — so that the copy-on-write
/// template snapshots those dependents took at resolve time are rebuilt from the new content.
///
/// This is deliberately not left to the asset server's own hot-reload ancestor walk: a cached
/// scene include is registered as a *runtime* dependency, not a *loader* dependency, and only
/// loader dependencies are walked. Editing a base file therefore produces no events at all for its
/// dependents without this.
///
/// Reloaded dependents re-enter the pipeline through the normal `LoadedWithDependencies` path a
/// frame or two later, which re-applies their instances and invalidates *their* dependents in
/// turn. The include graph is acyclic (a cycle fails at first resolve), so this terminates.
///
/// Dependents with no asset path — scenes built in code by `bsn!` — cannot be reloaded this way;
/// they are handled instead by the re-apply pass, which also matches instances whose patch merely
/// *depends on* the changed asset.
fn reload_dependents(
    changed: AssetId<ScenePatch>,
    assets: &AssetServer,
    patches: &Assets<ScenePatch>,
) {
    let changed_untyped = changed.untyped();
    // PERF: this derives the reverse-dependency map on demand — one scan of `Assets<ScenePatch>`
    // per reload event — rather than maintaining a cached index. Reloads are human-paced; a cached
    // index is a follow-up if profiling ever demands it.
    let dependents: Vec<AssetPath<'static>> = patches
        .iter()
        .filter(|(dependent_id, patch)| {
            *dependent_id != changed
                && patch
                    .dependencies
                    .iter()
                    .any(|dependency| dependency.id() == changed_untyped)
        })
        // UUID-keyed assets have no path and are skipped; the re-apply pass still covers their
        // instances.
        .filter_map(|(dependent_id, _)| assets.get_path(dependent_id).map(AssetPath::into_owned))
        .collect();

    for path in dependents {
        assets.reload(path);
    }
}

/// Returns whether the [`ScenePatch`] `dependent` lists `dependency` among its dependencies.
fn patch_depends_on(
    patches: &Assets<ScenePatch>,
    dependent: AssetId<ScenePatch>,
    dependency: AssetId<ScenePatch>,
) -> bool {
    let dependency = dependency.untyped();
    patches.get(dependent).is_some_and(|patch| {
        patch
            .dependencies
            .iter()
            .any(|handle| handle.id() == dependency)
    })
}

/// Applies `resolved` to `entity`, recording the entities it spawns into the entity's
/// [`SceneInstanceState`] and marking the instance applied.
///
/// Every application of a [`ScenePatch`] to an instance entity goes through here, so that a later
/// reload always knows what to clean up. If `entity` no longer exists this is a no-op: an instance
/// can be despawned between being queued and being applied.
fn apply_to_instance(
    world: &mut World,
    entity: Entity,
    resolved: &ResolvedSceneRoot,
    bundle_scratch: &mut BundleScratch,
) -> Result<(), ApplySceneError> {
    // Despawn the previous generation first: an instance is exactly one scene's content, so any
    // re-application — a reload, a replaced `ScenePatchInstance`, or a remove-and-re-insert —
    // must not orphan the entities the previous application spawned (the #24939 ghost class).
    // Entities already gone (e.g. despawned recursively via `Children`) fall through the
    // `get_entity_mut` check as no-ops.
    let mut spawned = world
        .get_mut::<SceneInstanceState>(entity)
        .map(|mut state| core::mem::take(&mut state.spawned))
        .unwrap_or_default();
    for previous in spawned.drain(..) {
        if let Ok(entity_mut) = world.get_entity_mut(previous) {
            // Despawning is recursive through linked-spawn relationships such as `Children`,
            // so descendants added at runtime go too. That is the documented state loss.
            entity_mut.despawn();
        }
    }

    let result = {
        let Ok(mut entity_mut) = world.get_entity_mut(entity) else {
            return Ok(());
        };
        if let Some(mut state) = entity_mut.get_mut::<SceneInstanceState>() {
            // Set before applying: if the apply fails half way, the instance is still "live" and
            // whatever landed in `spawned` must still be cleaned up by the next reload.
            state.applied = true;
        }
        // `spawned` reuses the previous application's allocation.
        resolved.apply_recording(&mut entity_mut, bundle_scratch, &mut spawned)
    };

    if let Some(mut state) = world.get_mut::<SceneInstanceState>(entity) {
        state.spawned = spawned;
    }
    result
}

/// Re-applies the (re)loaded [`ScenePatch`] `id` to every already-applied instance of it, and to
/// every already-applied instance whose own patch depends on it.
///
/// The previous generation of scene-spawned entities is despawned *first*. Applying a scene over
/// the top of an earlier application does not despawn what that application created — it only
/// unlinks it — which leaves a parentless "ghost" copy of the whole subtree behind
/// (bevyengine/bevy#24939).
fn reapply_instances(
    world: &mut World,
    id: AssetId<ScenePatch>,
    scene_patch_instances: &mut QueryState<(Entity, &ScenePatchInstance, &SceneInstanceState)>,
    bundle_scratch: &mut BundleScratch,
) {
    if world
        .resource::<Assets<ScenePatch>>()
        .get(id)
        .is_none_or(|patch| patch.resolved.is_none())
    {
        // A reload that failed to parse or to resolve leaves `resolved` empty on the new value.
        // Live instances keep the last good version.
        return;
    }

    // PERF: linear in the number of live `ScenePatchInstance` entities, but only when an asset
    // (re)loads. `QueryState::iter` updates archetypes internally, so instances spawned earlier
    // this frame are visible.
    let patches = world.resource::<Assets<ScenePatch>>();
    let instances: Vec<(Entity, AssetId<ScenePatch>)> = scene_patch_instances
        .iter(world)
        .filter(|(_, instance, state)| {
            // `applied` is what keeps this disjoint from the first-apply path: an instance still
            // waiting for its first application is not re-applied here.
            //
            // The `patch_depends_on` arm is what updates an instance of a scene that merely
            // *includes* the changed one. Such a scene reads its base's resolved scene fresh on
            // every apply, so re-applying it is all that is needed.
            state.applied
                && (instance.0.id() == id || patch_depends_on(patches, instance.0.id(), id))
        })
        .map(|(entity, instance, _)| (entity, instance.0.id()))
        .collect();

    for (entity, instance_id) in instances {
        // Each instance is re-applied from *its own* patch, which is not necessarily the asset
        // that changed: a dependent's resolved scene is what merges the new base content with the
        // dependent's own patches.
        let Some(resolved) = world
            .resource::<Assets<ScenePatch>>()
            .get(instance_id)
            .and_then(|patch| patch.resolved.clone())
        else {
            continue;
        };

        // `apply_to_instance` despawns the previous generation itself.
        if let Err(err) = apply_to_instance(world, entity, &resolved, bundle_scratch) {
            error!(
                "Failed to re-apply reloaded scene (id: {instance_id}) to entity {entity}: {err}"
            );
        }
    }
}

/// A [`Resource`] that tracks entities / scenes that have been queued to spawn.
#[derive(Resource, Default)]
pub struct QueuedScenes {
    new_scene_entities: Vec<(Entity, Handle<ScenePatch>)>,
    related_scene_list_spawns: Vec<(RelatedSceneListSpawn, Handle<SceneListPatch>)>,
    scene_list_spawns: Vec<Handle<SceneListPatch>>,
}

/// A [`Resource`] that tracks entities / scenes that are waiting for an asset to load
#[derive(Resource, Default)]
pub struct WaitingScenes {
    scene_entities: HashMap<Handle<ScenePatch>, Vec<Entity>>,
    related_list_entities: HashMap<Handle<SceneListPatch>, Vec<RelatedSceneListSpawn>>,
    scene_list_spawns: HashMap<Handle<SceneListPatch>, usize>,
}

pub(crate) struct RelatedSceneListSpawn {
    entity: Entity,
    insert: fn(&mut EntityWorldMut, target: Entity),
}

/// An [`Observer`] system that queues newly added or replaced [`ScenePatchInstance`] entities.
///
/// This watches `Insert`, not `Add`: replacing the component (for example a second
/// [`queue_apply_scene`](EntityWorldMutSceneExt::queue_apply_scene) on the same entity) must
/// re-queue the instance, and `Add` does not fire when the component is already present.
/// Each application fires a fresh bottom-up [`Ready`](crate::Ready) pass over the spawned
/// hierarchy, so `Ready` observers run again when an instance is replaced or hot reloaded.
pub fn on_insert_scene_patch_instance(
    insert: On<Insert<ScenePatchInstance>>,
    mut queued_scenes: ResMut<QueuedScenes>,
    instances: Query<&ScenePatchInstance>,
) {
    if let Ok(instance) = instances.get(insert.entity) {
        queued_scenes
            .new_scene_entities
            .push((insert.entity, instance.0.clone()));
    }
}

/// A system that spawns queued scenes when they are loaded.
pub fn spawn_queued(
    world: &mut World,
    scene_patch_instances: &mut QueryState<(Entity, &ScenePatchInstance, &SceneInstanceState)>,
    mut queued: Local<QueuedScenes>,
    mut bundle_scratch: Local<BundleScratch>,
    mut reader: Local<MessageCursor<AssetEvent<ScenePatch>>>,
    mut list_reader: Local<MessageCursor<AssetEvent<SceneListPatch>>>,
) {
    world.resource_scope(|world, mut list_patches: Mut<Assets<SceneListPatch>>| {
        world.resource_scope(|world, mut waiting: Mut<WaitingScenes>| {
            world.resource_scope(|world, events: Mut<Messages<AssetEvent<ScenePatch>>>| {
                // Collapse duplicate events for the same asset within this frame, which is what
                // two reload tasks completing in the same frame look like. We deliberately do not
                // debounce across frames the way `bevy_world_serialization`'s
                // `world_asset_spawner.rs:88` does: that exists to absorb glTF sub-asset loads,
                // and `ScenePatch` has no sub-assets, so a frame counter would only add latency
                // and could swallow a genuine second edit.
                let mut loaded: Vec<AssetId<ScenePatch>> = reader
                    .read(&events)
                    .filter_map(|event| match event {
                        AssetEvent::LoadedWithDependencies { id } => Some(*id),
                        _ => None,
                    })
                    .collect();
                loaded.sort_unstable();
                loaded.dedup();

                // The re-apply (hot reload) pass runs *before* the first-apply pass below. Both
                // are driven by the same event, and running them in this order is what keeps their
                // instance sets disjoint: at this point an instance still waiting for its first
                // application has `applied == false` and is skipped here, and the loop below then
                // applies it exactly once, from the already-refreshed `resolved`.
                for id in loaded.iter().copied() {
                    reapply_instances(world, id, scene_patch_instances, &mut bundle_scratch);
                }

                for id in loaded.iter() {
                    let patches = world.resource::<Assets<ScenePatch>>();
                    if let Some(resolved) = patches.get(*id).and_then(|p| p.resolved.clone())
                        && let Some(entities) = waiting.scene_entities.remove(id)
                    {
                        for entity in entities {
                            if let Err(err) =
                                apply_to_instance(world, entity, &resolved, &mut bundle_scratch)
                            {
                                error!(
                                    "Failed to apply scene (id: {}) to entity {entity}: {}",
                                    id, err
                                );
                            }
                        }
                    }
                }
            });
            world.resource_scope(
                |world, list_events: Mut<Messages<AssetEvent<SceneListPatch>>>| {
                    for event in list_reader.read(&list_events) {
                        if let AssetEvent::LoadedWithDependencies { id } = event
                            && let Some(list_patch) = list_patches.get_mut(*id)
                        {
                            if let Some(scene_list_spawns) =
                                waiting.related_list_entities.remove(id)
                            {
                                for scene_list_spawn in scene_list_spawns {
                                    let result = list_patch.spawn_with(world, |entity| {
                                        (scene_list_spawn.insert)(entity, scene_list_spawn.entity);
                                    });

                                    if let Err(err) = result {
                                        error!("Failed to spawn scene list (id: {}): {}", id, err);
                                    }
                                }
                            }

                            if let Some(waiting_list_spawns) = waiting.scene_list_spawns.remove(id)
                            {
                                for _ in 0..waiting_list_spawns {
                                    let result = list_patch.spawn(world);
                                    if let Err(err) = result {
                                        error!("Failed to spawn scene list (id: {}): {}", id, err);
                                    }
                                }
                            }
                        }
                    }
                },
            );

            loop {
                core::mem::swap(&mut *world.resource_mut::<QueuedScenes>(), &mut queued);
                if queued.is_empty() {
                    break;
                }
                queued.spawn_queued(world, &mut waiting, &mut bundle_scratch, &list_patches);
            }
        });
    });
}

impl QueuedScenes {
    fn is_empty(&self) -> bool {
        self.new_scene_entities.is_empty()
            && self.related_scene_list_spawns.is_empty()
            && self.scene_list_spawns.is_empty()
    }

    fn spawn_queued(
        &mut self,
        world: &mut World,
        waiting_scenes: &mut WaitingScenes,
        bundle_scratch: &mut BundleScratch,
        list_patches: &Assets<SceneListPatch>,
    ) {
        for (entity, handle) in core::mem::take(&mut self.new_scene_entities) {
            let patches = world.resource::<Assets<ScenePatch>>();
            if let Some(resolved) = patches.get(&handle).and_then(|p| p.resolved.clone()) {
                if let Err(err) = apply_to_instance(world, entity, &resolved, bundle_scratch) {
                    let id = handle.id();
                    let path = handle.path();
                    error!(
                        "Failed to apply scene (id: {id}, path: {path:?}) to \
                                    entity {entity}: {err}",
                    );
                }
            } else {
                let entities = waiting_scenes
                    .scene_entities
                    .entry(handle.clone())
                    .or_default();
                entities.push(entity);
            }
        }

        for (scene_list_spawn, handle) in core::mem::take(&mut self.related_scene_list_spawns) {
            if let Some(list_patch) = list_patches.get(&handle) {
                let result = list_patch.spawn_with(world, |entity| {
                    (scene_list_spawn.insert)(entity, scene_list_spawn.entity);
                });

                if let Err(err) = result {
                    error!(
                        "Failed to spawn scene list (id: {}, path: {:?}): {}",
                        handle.id(),
                        handle.path(),
                        err
                    );
                }
            } else {
                let entities = waiting_scenes
                    .related_list_entities
                    .entry(handle)
                    .or_default();
                entities.push(scene_list_spawn);
            }
        }

        for handle in core::mem::take(&mut self.scene_list_spawns) {
            if let Some(list_patch) = list_patches.get(&handle) {
                let result = list_patch.spawn(world);
                if let Err(err) = result {
                    error!(
                        "Failed to spawn scene list (id: {}, path: {:?}): {}",
                        handle.id(),
                        handle.path(),
                        err
                    );
                }
            } else {
                let count = waiting_scenes.scene_list_spawns.entry(handle).or_default();
                *count += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EntityCommandsSceneExt, EntityWorldMutSceneExt, WorldSceneExt};
    use crate::test_support::{memory_asset_app, run_app_until, test_app};
    use crate::{
        self as bevy_scene, bsn, Ready, Scene, SceneInstanceState, ScenePatch, ScenePatchInstance,
    };
    use alloc::sync::Arc;
    use bevy_app::{App, Last};
    use bevy_asset::{AssetApp, AssetEvent, AssetLoader, AssetServer, Assets, Handle};
    use bevy_ecs::{name::Name, prelude::*, template::FromTemplate};
    use bevy_platform::collections::HashMap;
    use bevy_reflect::TypePath;
    use std::{path::Path, sync::Mutex};

    #[derive(Component, Default, FromTemplate)]
    struct SceneChild;

    #[derive(Component)]
    struct PreExistingChild;

    /// Tests that documented behavior of [`EntityWorldMutSceneExt::apply_scene`] is correct.
    #[test]
    fn apply_scene_replaces_and_orphans_children() {
        let mut app = test_app();
        let world = app.world_mut();

        let pre_existing = world.spawn(PreExistingChild).id();
        let root = world.spawn(Name::new("root")).add_child(pre_existing).id();

        assert_eq!(
            world.entity(root).get::<Children>().map(Children::len),
            Some(1)
        );

        let scene = bsn! {
            Children [ #SceneChild SceneChild ]
        };
        world.entity_mut(root).apply_scene(scene).unwrap();

        let children: Vec<Entity> = world
            .entity(root)
            .get::<Children>()
            .map(|c| c.iter().collect())
            .unwrap_or_default();

        // Scene child is spawned and linked.
        assert_eq!(children.len(), 1);
        assert!(world.entity(children[0]).contains::<SceneChild>());

        // Pre-existing child entity still exists, but is no longer listed under root.
        assert!(world.get_entity(pre_existing).is_ok());
        assert!(!children.contains(&pre_existing));
    }

    // ---------------------------------------------------------------------------------------
    // Hot reload
    //
    // These tests drive the real reload pipeline — `AssetServer::reload` re-runs a loader, which
    // replaces the whole `ScenePatch` value and re-fires `LoadedWithDependencies` — rather than
    // hand-writing `AssetEvent`s, which could not reproduce either the event ordering or the
    // whole-value replacement that the design depends on.
    //
    // They use a fake loader rather than the `.bsn` loader, so they hold without the `bsn_asset`
    // feature and cover third-party `AssetLoader<Asset = ScenePatch>` implementations too. The
    // same scenarios are mirrored with real `.bsn` text in `tests/dynamic_bsn.rs`.
    // ---------------------------------------------------------------------------------------

    /// What a fake "file" currently contains.
    enum FakeSource {
        /// The file parses, and produces this scene.
        Scene(Box<dyn Fn() -> Box<dyn Scene> + Send + Sync>),
        /// The file no longer parses, so its loader fails — exactly what a `.bsn` syntax error
        /// does.
        ParseError,
    }

    /// The in-memory "disk" behind [`FakeSceneLoader`], shared with the test body.
    #[derive(Clone, Default)]
    struct FakeScenes(
        Arc<Mutex<HashMap<String, FakeSource>>>,
        /// Paths whose loader must stall until released. Used to control which of two concurrent
        /// reload tasks finishes first.
        Arc<Mutex<Vec<String>>>,
    );

    impl FakeScenes {
        /// Makes `path`'s loader stall until [`FakeScenes::unblock`].
        fn block(&self, path: &str) {
            self.1.lock().unwrap().push(path.to_string());
        }

        fn unblock(&self, path: &str) {
            self.1.lock().unwrap().retain(|p| p != path);
        }

        fn is_blocked(&self, path: &str) -> bool {
            self.1.lock().unwrap().iter().any(|p| p == path)
        }
    }

    impl FakeScenes {
        /// Writes `scene_fn` to `path`, overwriting whatever was there before. This is the
        /// "edit the file" primitive; [`AssetServer::reload`] is the "save" that follows it.
        fn write<S: Scene>(&self, path: &str, scene_fn: impl Fn() -> S + Send + Sync + 'static) {
            self.0.lock().unwrap().insert(
                path.to_string(),
                FakeSource::Scene(Box::new(move || Box::new(scene_fn()))),
            );
        }

        /// Replaces `path`'s contents with something that no longer parses.
        fn write_parse_error(&self, path: &str) {
            self.0
                .lock()
                .unwrap()
                .insert(path.to_string(), FakeSource::ParseError);
        }
    }

    #[derive(TypePath)]
    struct FakeSceneLoader(FakeScenes);

    impl AssetLoader for FakeSceneLoader {
        type Asset = ScenePatch;
        type Error = std::io::Error;
        type Settings = ();

        async fn load(
            &self,
            _reader: &mut dyn bevy_asset::io::Reader,
            _settings: &Self::Settings,
            load_context: &mut bevy_asset::LoadContext<'_>,
        ) -> Result<Self::Asset, Self::Error> {
            let path = load_context.path().path().to_string_lossy().into_owned();
            // Bounded so a single-threaded task pool can never hang the suite.
            for _ in 0..2000 {
                if !self.0.is_blocked(&path) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let scene = {
                let scenes = self.0 .0.lock().unwrap();
                match scenes.get(&path) {
                    Some(FakeSource::Scene(scene_fn)) => scene_fn(),
                    Some(FakeSource::ParseError) => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("{path} does not parse"),
                        ))
                    }
                    None => return Err(std::io::Error::new(std::io::ErrorKind::NotFound, path)),
                }
            };
            Ok(ScenePatch::load_with(load_context, scene))
        }

        /// A dedicated extension: `.bsn` belongs to the `bevy_bsn_asset` loader in apps that
        /// register `BsnAssetPlugin`, and these fixtures are not `.bsn` documents.
        fn extensions(&self) -> &[&str] {
            &["fakescene"]
        }
    }

    /// Builds an [`App`] whose `.fakescene` assets come from `scenes`. `files` must list every
    /// path the test will load: the asset source has to have *something* at that path, even
    /// though the fake loader ignores the bytes.
    fn hot_reload_app(scenes: &FakeScenes, files: &[&str]) -> App {
        let (mut app, dir) = memory_asset_app();
        app.finish();
        app.cleanup();
        app.register_asset_loader(FakeSceneLoader(scenes.clone()));
        for file in files {
            dir.insert_asset_text(Path::new(file), "");
        }
        app
    }

    /// Loads `path` and pumps until it has been resolved.
    fn load_and_settle(app: &mut App, path: &'static str) -> Handle<ScenePatch> {
        let asset_server = app.world().resource::<AssetServer>().clone();
        let handle = asset_server.load::<ScenePatch>(path);
        let probe = handle.clone();
        run_app_until(app, |app| {
            app.world()
                .resource::<Assets<ScenePatch>>()
                .get(&probe)
                .is_some_and(|patch| patch.resolved.is_some())
        });
        handle
    }

    /// "Saves" `path`: re-runs its loader against whatever [`FakeScenes`] now holds.
    fn reload(app: &App, path: &'static str) {
        app.world().resource::<AssetServer>().reload(path);
    }

    #[derive(Component, FromTemplate, Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Component, Default, Clone)]
    struct HotMarker;

    /// Names of `root`'s children, in order.
    fn child_names(app: &App, root: Entity) -> Vec<String> {
        app.world()
            .get::<Children>(root)
            .map(|children| {
                children
                    .iter()
                    .map(|child| {
                        app.world()
                            .get::<Name>(child)
                            .map(|name| name.as_str().to_string())
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn hot_reload_replaces_root_components() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Position { x: 1. } });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| app.world().get::<Position>(root).is_some());
        assert_eq!(app.world().get::<Position>(root).unwrap().x, 1.);

        scenes.write("a.fakescene", || bsn! { Position { x: 5. } });
        reload(&app, "a.fakescene");
        run_app_until(&mut app, |app| {
            app.world().get::<Position>(root).unwrap().x == 5.
        });
    }

    /// [`Ready`] fires per *application* of a scene: the initial spawn emits one bottom-up
    /// pass, and a hot reload emits a fresh pass over the regenerated hierarchy.
    #[test]
    fn hot_reload_refires_ready_for_regenerated_entities() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Children [ #A ] });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let ready = Arc::new(Mutex::new(Vec::new()));
        let tracker = ready.clone();
        app.add_observer(move |on: On<Ready>| tracker.lock().unwrap().push(on.entity));

        let handle = load_and_settle(&mut app, "a.fakescene");
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| child_names(app, root) == ["A"]);

        let child = app.world().get::<Children>(root).unwrap()[0];
        assert_eq!(core::mem::take(&mut *ready.lock().unwrap()), [child, root]);

        scenes.write("a.fakescene", || bsn! { Children [ #B ] });
        reload(&app, "a.fakescene");
        run_app_until(&mut app, |app| child_names(app, root) == ["B"]);

        let new_child = app.world().get::<Children>(root).unwrap()[0];
        assert_eq!(*ready.lock().unwrap(), [new_child, root]);
    }

    #[test]
    fn hot_reload_despawns_previous_children() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Children [ #A -- #B ] });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| app.world().get::<Children>(root).is_some());

        let previous: Vec<Entity> = app
            .world()
            .get::<Children>(root)
            .unwrap()
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(previous.len(), 2);

        scenes.write("a.fakescene", || bsn! { Children [ #C ] });
        reload(&app, "a.fakescene");
        run_app_until(&mut app, |app| child_names(app, root) == ["C"]);

        // The regression this whole design exists for (bevyengine/bevy#24939): re-applying a scene
        // only *unlinks* the previous children, so without an explicit despawn they survive as
        // parentless ghosts carrying all their components and observers.
        for entity in previous {
            assert!(
                app.world().get_entity(entity).is_err(),
                "the previous generation of children must be despawned, not orphaned"
            );
        }
    }

    #[test]
    fn hot_reload_no_orphaned_entities() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Children [ #A -- #B ] });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let before = app.world().entities().count_spawned();
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| app.world().get::<Children>(root).is_some());
        // The root plus its two children.
        assert_eq!(app.world().entities().count_spawned(), before + 3);

        scenes.write("a.fakescene", || bsn! { Children [ #C ] });
        reload(&app, "a.fakescene");
        run_app_until(&mut app, |app| child_names(app, root) == ["C"]);

        assert_eq!(
            app.world().entities().count_spawned(),
            before + 2,
            "a reload must leave exactly the root and the new generation of children"
        );
    }

    #[test]
    fn hot_reload_preserves_instance_entity() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Position { x: 1. } });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        let parent = app.world_mut().spawn(HotMarker).add_child(root).id();
        run_app_until(&mut app, |app| app.world().get::<Position>(root).is_some());

        scenes.write("a.fakescene", || bsn! { Position { x: 5. } });
        reload(&app, "a.fakescene");
        run_app_until(&mut app, |app| {
            app.world().get::<Position>(root).unwrap().x == 5.
        });

        assert!(app.world().get_entity(root).is_ok());
        assert_eq!(
            app.world().get::<ChildOf>(root).map(|child_of| child_of.0),
            Some(parent),
            "the instance entity itself is never despawned, so its parent link survives"
        );
    }

    #[test]
    fn hot_reload_applies_to_all_instances() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Position { x: 1. } });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let roots: Vec<Entity> = (0..3)
            .map(|_| {
                app.world_mut()
                    .spawn(ScenePatchInstance(handle.clone()))
                    .id()
            })
            .collect();
        run_app_until(&mut app, |app| {
            app.world().get::<Position>(roots[2]).is_some()
        });

        scenes.write("a.fakescene", || bsn! { Position { x: 5. } });
        reload(&app, "a.fakescene");
        run_app_until(&mut app, |app| {
            app.world().get::<Position>(roots[0]).unwrap().x == 5.
        });

        for root in roots {
            assert_eq!(
                app.world().get::<Position>(root).unwrap().x,
                5.,
                "every instance is updated in the same frame"
            );
        }
    }

    #[test]
    fn hot_reload_of_base_updates_dependent_file_instances() {
        let scenes = FakeScenes::default();
        scenes.write("base.fakescene", || bsn! { Position { x: 1., y: 1. } });
        // `derived` patches `Position` too, so it takes a copy-on-write snapshot of the base's
        // template at resolve time. That snapshot is exactly what goes stale.
        scenes.write("derived.fakescene", || {
            bsn! { :"base.fakescene" Position { x: 2. } }
        });
        let mut app = hot_reload_app(&scenes, &["base.fakescene", "derived.fakescene"]);

        let handle = load_and_settle(&mut app, "derived.fakescene");
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| app.world().get::<Position>(root).is_some());
        assert_eq!(
            *app.world().get::<Position>(root).unwrap(),
            Position { x: 2., y: 1. }
        );

        scenes.write("base.fakescene", || bsn! { Position { x: 1., y: 9. } });
        reload(&app, "base.fakescene");
        run_app_until(&mut app, |app| {
            app.world().get::<Position>(root).unwrap().y == 9.
        });

        assert_eq!(
            *app.world().get::<Position>(root).unwrap(),
            Position { x: 2., y: 9. },
            "the dependent file must be re-resolved so its copy-on-write snapshot is rebuilt, \
             while its own patch still wins"
        );
    }

    #[test]
    fn hot_reload_of_base_updates_bsn_macro_instances() {
        let scenes = FakeScenes::default();
        scenes.write("base.fakescene", || bsn! { Children [ #A ] });
        let mut app = hot_reload_app(&scenes, &["base.fakescene"]);
        load_and_settle(&mut app, "base.fakescene");

        // An in-code scene has no asset path, so it cannot be reloaded — but it reads its base's
        // resolved scene fresh on every apply, so re-applying it is enough.
        let root = app
            .world_mut()
            .queue_spawn_scene(bsn! { :"base.fakescene" HotMarker })
            .id();
        run_app_until(&mut app, |app| app.world().get::<Children>(root).is_some());
        assert_eq!(child_names(&app, root), ["A"]);

        scenes.write("base.fakescene", || bsn! { Children [ #B -- #C ] });
        reload(&app, "base.fakescene");
        run_app_until(&mut app, |app| child_names(app, root) == ["B", "C"]);
    }

    #[test]
    fn hot_reload_of_base_does_not_update_macro_overlap() {
        let scenes = FakeScenes::default();
        scenes.write("base.fakescene", || {
            bsn! { Position { x: 1., y: 1. } Children [ #A ] }
        });
        let mut app = hot_reload_app(&scenes, &["base.fakescene"]);
        load_and_settle(&mut app, "base.fakescene");

        let root = app
            .world_mut()
            .queue_spawn_scene(bsn! { :"base.fakescene" Position { x: 2. } })
            .id();
        run_app_until(&mut app, |app| app.world().get::<Position>(root).is_some());
        assert_eq!(
            *app.world().get::<Position>(root).unwrap(),
            Position { x: 2., y: 1. }
        );

        scenes.write("base.fakescene", || {
            bsn! { Position { x: 1., y: 9. } Children [ #B ] }
        });
        reload(&app, "base.fakescene");
        run_app_until(&mut app, |app| child_names(app, root) == ["B"]);

        // This pins the documented limitation. An in-code `bsn!` scene that both includes a scene
        // asset and patches a component the base also patches froze the base's values for that
        // component at its first (and only) resolve: its `Scene` was consumed by value and there
        // is no file to re-read. Everything else about the instance is live.
        assert_eq!(
            *app.world().get::<Position>(root).unwrap(),
            Position { x: 2., y: 1. },
            "the copy-on-write overlap set of an in-code scene cannot be refreshed; move the \
             overlapping patch into a scene asset to get full hot reload"
        );
    }

    #[test]
    fn hot_reload_parse_error_keeps_previous_scene() {
        let scenes = FakeScenes::default();
        scenes.write(
            "a.fakescene",
            || bsn! { Position { x: 1. } Children [ #A ] },
        );
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| app.world().get::<Position>(root).is_some());
        let child = app.world().get::<Children>(root).unwrap()[0];

        scenes.write_parse_error("a.fakescene");
        reload(&app, "a.fakescene");
        for _ in 0..20 {
            app.update();
        }

        assert_eq!(app.world().get::<Position>(root).unwrap().x, 1.);
        assert!(
            app.world().get_entity(child).is_ok(),
            "a broken edit must leave the last good version rendering, untouched"
        );
        assert_eq!(child_names(&app, root), ["A"]);
    }

    #[test]
    fn hot_reload_invalidates_external_entity_references() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Children [ #A ] });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| app.world().get::<Children>(root).is_some());
        let held = app.world().get::<Children>(root).unwrap()[0];

        scenes.write("a.fakescene", || bsn! { Children [ #A ] });
        reload(&app, "a.fakescene");
        run_app_until(&mut app, |app| {
            app.world().get::<Children>(root).unwrap()[0] != held
        });

        // Documented state loss: a reload rebuilds the scene, so `Entity` ids held outside it —
        // in a resource, another component, an observer — dangle afterwards.
        assert!(app.world().get_entity(held).is_err());
    }

    #[test]
    fn hot_reload_during_pending_spawn() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Children [ #A ] });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let asset_server = app.world().resource::<AssetServer>().clone();
        let handle = asset_server.load::<ScenePatch>("a.fakescene");
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();

        // Edit and save before the instance has ever been applied.
        scenes.write("a.fakescene", || bsn! { Children [ #B ] });
        asset_server.reload("a.fakescene");

        run_app_until(&mut app, |app| child_names(app, root) == ["B"]);
        for _ in 0..10 {
            app.update();
        }

        assert_eq!(
            child_names(&app, root),
            ["B"],
            "the instance must be applied exactly once, with the new content"
        );
    }

    #[test]
    fn hot_reload_twice_in_one_frame_applies_once() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Children [ #A ] });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let before = app.world().entities().count_spawned();
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| app.world().get::<Children>(root).is_some());

        scenes.write("a.fakescene", || bsn! { Children [ #B ] });
        reload(&app, "a.fakescene");
        reload(&app, "a.fakescene");
        run_app_until(&mut app, |app| child_names(app, root) == ["B"]);
        for _ in 0..10 {
            app.update();
        }

        assert_eq!(child_names(&app, root), ["B"]);
        assert_eq!(
            app.world().entities().count_spawned(),
            before + 2,
            "two reloads landing together must produce one despawn/apply cycle, not two"
        );
    }

    #[derive(Resource, Default)]
    struct ModifiedCount(usize);

    fn count_modified(
        mut events: MessageReader<AssetEvent<ScenePatch>>,
        mut count: ResMut<ModifiedCount>,
    ) {
        for event in events.read() {
            if matches!(event, AssetEvent::Modified { .. }) {
                count.0 += 1;
            }
        }
    }

    #[test]
    fn resolve_does_not_emit_modified() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Position { x: 1. } });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);
        app.init_resource::<ModifiedCount>();
        app.add_systems(Last, count_modified);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| app.world().get::<Position>(root).is_some());
        for _ in 0..5 {
            app.update();
        }

        // Resolution writes through `get_mut_untracked`, so `Modified` on `ScenePatch` now means
        // "the asset actually changed" rather than "the scene system did its bookkeeping".
        assert_eq!(app.world().resource::<ModifiedCount>().0, 0);
    }

    #[test]
    fn queued_scene_gains_scene_patch_instance() {
        let mut app = test_app();
        let root = app.world_mut().queue_spawn_scene(bsn! { HotMarker }).id();
        app.update();

        assert!(app.world().get::<ScenePatchInstance>(root).is_some());
        let state = app.world().get::<SceneInstanceState>(root).unwrap();
        assert!(state.applied);
        assert!(app.world().get::<HotMarker>(root).is_some());
    }

    #[test]
    fn scene_instance_state_records_all_spawned() {
        let mut app = test_app();
        let root = app
            .world_mut()
            .queue_spawn_scene(bsn! {
                Children [
                    #A Children [ #A1 -- #A2 ]
                    --
                    #B
                ]
            })
            .id();
        app.update();

        let mut descendants: Vec<Entity> = Vec::new();
        let mut stack = vec![root];
        while let Some(entity) = stack.pop() {
            if let Some(children) = app.world().get::<Children>(entity) {
                for child in children.iter() {
                    descendants.push(child);
                    stack.push(child);
                }
            }
        }
        descendants.sort_unstable();
        assert_eq!(descendants.len(), 4);

        let state = app.world().get::<SceneInstanceState>(root).unwrap();
        assert!(state.applied);
        assert_eq!(
            state.spawned, descendants,
            "every entity the scene spawned is recorded, sorted and deduplicated"
        );
        assert!(
            !state.spawned.contains(&root),
            "the instance entity itself is never recorded: it is not despawned on reload"
        );
    }

    #[test]
    fn immediate_spawn_does_not_record() {
        // `World::spawn_scene` has no asset identity and can never hot reload, so it must not pay
        // for the recording. This is the observable half of that: nothing is written down.
        let mut app = test_app();
        let root = app
            .world_mut()
            .spawn_scene(bsn! { Children [ #A ] })
            .unwrap()
            .id();
        assert!(app.world().get::<SceneInstanceState>(root).is_none());
    }

    // =======================================================================================
    // ADVERSARIAL REVIEW REPROS
    // =======================================================================================

    #[derive(Component, FromTemplate)]
    struct Reference(Entity);

    /// ADV-1 (fixed): `queue_apply_scene` twice on the same entity replaces the instance.
    ///
    /// The observer watches `Insert` (not `Add`), so replacing `ScenePatchInstance` re-queues:
    /// the second scene applies. It used to be silently dropped, leaving the recorded state
    /// describing a different asset than the handle.
    #[test]
    fn adv1_queue_apply_scene_twice_replaces_the_instance() {
        let mut app = test_app();
        let root = app.world_mut().spawn_empty().id();
        app.world_mut()
            .entity_mut(root)
            .queue_apply_scene(bsn! { HotMarker });
        app.world_mut()
            .entity_mut(root)
            .queue_apply_scene(bsn! { Position { x: 3. } });
        for _ in 0..5 {
            app.update();
        }
        assert!(
            app.world().get::<HotMarker>(root).is_some(),
            "first queued scene applied"
        );
        assert!(
            app.world().get::<Position>(root).is_some(),
            "second queued scene must also be applied"
        );
    }

    /// ADV-1b (fixed): the realistic composition — spawn an entity from one scene, then queue a
    /// second scene onto it. The second replaces the instance (components merge; the first
    /// scene's spawned descendants are despawned), per the documented replace semantics.
    #[test]
    fn adv1b_queue_apply_scene_onto_queue_spawned_entity() {
        let mut app = test_app();
        let root = app.world_mut().queue_spawn_scene(bsn! { HotMarker }).id();
        app.world_mut()
            .entity_mut(root)
            .queue_apply_scene(bsn! { Position { x: 3. } });
        for _ in 0..5 {
            app.update();
        }
        assert!(app.world().get::<HotMarker>(root).is_some());
        assert!(
            app.world().get::<Position>(root).is_some(),
            "a scene queued onto an existing instance entity must still be applied"
        );
    }

    /// ADV-2 (fixed): forward `#Name` references used only as component values are recorded and
    /// reclaimed on reload. `apply_recording` unions the `SceneEntityReferences` map into the
    /// spawn record, so the materialized entity is despawned with the rest of its generation.
    #[test]
    fn adv2_ghost_reference_entities_do_not_leak_on_reload() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Reference(#Ghost) });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let before = app.world().entities().count_spawned();
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| app.world().get::<Reference>(root).is_some());
        // root + the materialized ghost entity
        let after_first = app.world().entities().count_spawned();
        assert_eq!(after_first, before + 2, "root + one ghost entity");
        let first_ghost = app.world().get::<Reference>(root).unwrap().0;

        for i in 0..5 {
            let x = i as f32;
            scenes.write("a.fakescene", move || {
                bsn! { Reference(#Ghost) Position { x: x } }
            });
            reload(&app, "a.fakescene");
            run_app_until(&mut app, |app| {
                app.world().get::<Position>(root).map(|p| p.x) == Some(x)
            });
        }

        let second_ghost = app.world().get::<Reference>(root).unwrap().0;
        assert_ne!(first_ghost, second_ghost, "a fresh map spawns a new ghost");
        assert_eq!(
            app.world().entities().count_spawned(),
            before + 2,
            "reloading must not grow the entity count (5 reloads happened)"
        );
        assert!(
            app.world().get_entity(first_ghost).is_err(),
            "the previous generation's ghost entity must be despawned, not leaked"
        );
        assert_eq!(
            app.world().entities().count_spawned(),
            before + 2,
            "reloading must not grow the entity count"
        );
    }

    /// ADV-3 (fixed): re-adding `ScenePatchInstance` (the natural way to swap which scene an
    /// entity is an instance of) despawns the previous generation before the new scene applies —
    /// `apply_to_instance` owns the despawn, so no path can orphan a generation (#24939 class).
    #[test]
    fn adv3_reinserting_scene_patch_instance_despawns_previous_generation() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Children [ #A -- #B ] });
        scenes.write("b.fakescene", || bsn! { Children [ #C ] });
        let mut app = hot_reload_app(&scenes, &["a.fakescene", "b.fakescene"]);

        let a = load_and_settle(&mut app, "a.fakescene");
        let b = load_and_settle(&mut app, "b.fakescene");
        let before = app.world().entities().count_spawned();
        let root = app.world_mut().spawn(ScenePatchInstance(a)).id();
        run_app_until(&mut app, |app| child_names(app, root) == ["A", "B"]);
        let previous: Vec<Entity> = app
            .world()
            .get::<Children>(root)
            .unwrap()
            .iter()
            .collect::<Vec<_>>();

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
            "root + the one child of the new scene"
        );
    }

    /// ADV-4: a scene-spawned child that itself carries a `ScenePatchInstance`. Reloading the
    /// *inner* asset must not desync the outer instance's bookkeeping, and reloading the outer
    /// one must clean up everything.
    #[test]
    fn adv4_nested_instance_reload() {
        let scenes = FakeScenes::default();
        scenes.write("inner.fakescene", || bsn! { Children [ #I1 ] });
        let mut app = hot_reload_app(&scenes, &["inner.fakescene", "outer.fakescene"]);
        let inner = load_and_settle(&mut app, "inner.fakescene");

        let outer_handle = inner.clone();
        scenes.write("outer.fakescene", move || {
            let h = outer_handle.clone();
            bsn! { Children [ #Nested ScenePatchInstance(h) ] }
        });
        let outer = load_and_settle(&mut app, "outer.fakescene");

        let before = app.world().entities().count_spawned();
        let root = app.world_mut().spawn(ScenePatchInstance(outer)).id();
        run_app_until(&mut app, |app| {
            app.world()
                .get::<Children>(root)
                .and_then(|c| app.world().get::<Children>(c[0]))
                .is_some()
        });
        // root + nested child + inner's child
        assert_eq!(app.world().entities().count_spawned(), before + 3);

        // Reload the inner asset: the nested instance rebuilds itself.
        scenes.write("inner.fakescene", || bsn! { Children [ #I2 ] });
        reload(&app, "inner.fakescene");
        run_app_until(&mut app, |app| {
            let nested = app.world().get::<Children>(root).unwrap()[0];
            child_names(app, nested) == ["I2"]
        });
        assert_eq!(
            app.world().entities().count_spawned(),
            before + 3,
            "an inner reload must not leak"
        );

        // Now reload the outer asset: everything below the root is rebuilt.
        let inner2 = inner.clone();
        scenes.write("outer.fakescene", move || {
            let h = inner2.clone();
            bsn! { Children [ #Nested2 ScenePatchInstance(h) ] }
        });
        reload(&app, "outer.fakescene");
        run_app_until(&mut app, |app| {
            child_names(app, root) == ["Nested2"]
                && app
                    .world()
                    .get::<Children>(app.world().get::<Children>(root).unwrap()[0])
                    .is_some()
        });
        for _ in 0..10 {
            app.update();
        }
        assert_eq!(
            app.world().entities().count_spawned(),
            before + 3,
            "an outer reload must clean up the nested instance's entities too"
        );
    }

    /// ADV-5: manually despawning a recorded child and then reloading must not panic or leave the
    /// world inconsistent.
    #[test]
    fn adv5_manual_despawn_of_recorded_child_then_reload() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Children [ #A -- #B ] });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let before = app.world().entities().count_spawned();
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| child_names(app, root) == ["A", "B"]);
        let child = app.world().get::<Children>(root).unwrap()[0];
        app.world_mut().entity_mut(child).despawn();

        scenes.write("a.fakescene", || bsn! { Children [ #C ] });
        reload(&app, "a.fakescene");
        run_app_until(&mut app, |app| child_names(app, root) == ["C"]);
        assert_eq!(app.world().entities().count_spawned(), before + 2);
    }

    /// ADV-6: a base and its dependent reload in the same frame, with the dependent's loader
    /// finishing first (saving two files at once, a `git checkout`, an editor "save all").
    ///
    /// `resolve_scene_patches` processes `LoadedWithDependencies` in message order. If the
    /// dependent is resolved before its base, the base's freshly-replaced `ScenePatch` still has
    /// `resolved: None`, so `get_or_insert_erased_template_index` takes the `default()` branch
    /// instead of the copy-on-write clone. The overlapping component is then written twice — base's
    /// whole value, then the dependent's whole value — instead of being field-merged, and the
    /// mis-resolution is cached until the dependent is reloaded again.
    #[test]
    fn adv6_base_and_dependent_reload_out_of_order() {
        let scenes = FakeScenes::default();
        scenes.write("base.fakescene", || bsn! { Position { x: 1., y: 1. } });
        scenes.write("derived.fakescene", || {
            bsn! { :"base.fakescene" Position { x: 2. } }
        });
        let mut app = hot_reload_app(&scenes, &["base.fakescene", "derived.fakescene"]);

        let handle = load_and_settle(&mut app, "derived.fakescene");
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| app.world().get::<Position>(root).is_some());
        assert_eq!(
            *app.world().get::<Position>(root).unwrap(),
            Position { x: 2., y: 1. }
        );

        scenes.write("base.fakescene", || bsn! { Position { x: 1., y: 9. } });
        scenes.write("derived.fakescene", || {
            bsn! { :"base.fakescene" Position { x: 2. } }
        });
        // Hold the base's loader so the dependent's reload task completes (and queues its event)
        // first, then release it so both land in the same `handle_internal_asset_events` drain.
        scenes.block("base.fakescene");
        reload(&app, "base.fakescene");
        reload(&app, "derived.fakescene");
        std::thread::sleep(std::time::Duration::from_millis(300));
        scenes.unblock("base.fakescene");
        std::thread::sleep(std::time::Duration::from_millis(300));
        for _ in 0..50 {
            app.update();
        }

        assert_eq!(
            *app.world().get::<Position>(root).unwrap(),
            Position { x: 2., y: 9. },
            "the dependent must field-merge over the new base regardless of resolve order"
        );
    }

    /// ADV-7: a reload that fails to resolve, followed by one that succeeds. The failed resolve
    /// leaves `scene: None` / `resolved: None` on the new value; recovery must be clean.
    #[test]
    fn adv7_failed_reload_then_successful_reload_recovers() {
        let scenes = FakeScenes::default();
        scenes.write(
            "a.fakescene",
            || bsn! { Position { x: 1. } Children [ #A ] },
        );
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let before = app.world().entities().count_spawned();
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| app.world().get::<Position>(root).is_some());

        scenes.write_parse_error("a.fakescene");
        reload(&app, "a.fakescene");
        for _ in 0..20 {
            app.update();
        }
        assert_eq!(app.world().get::<Position>(root).unwrap().x, 1.);

        scenes.write(
            "a.fakescene",
            || bsn! { Position { x: 7. } Children [ #B ] },
        );
        reload(&app, "a.fakescene");
        run_app_until(&mut app, |app| {
            app.world().get::<Position>(root).unwrap().x == 7.
        });
        assert_eq!(child_names(&app, root), ["B"]);
        assert_eq!(
            app.world().entities().count_spawned(),
            before + 2,
            "recovery must not leak the broken generation"
        );
    }

    /// ADV-8: reloading an asset with no live instances, then spawning an instance of it.
    #[test]
    fn adv8_reload_without_instances_then_spawn() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Position { x: 1. } });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);
        let handle = load_and_settle(&mut app, "a.fakescene");

        scenes.write("a.fakescene", || bsn! { Position { x: 5. } });
        reload(&app, "a.fakescene");
        run_app_until(&mut app, |app| {
            app.world()
                .resource::<Assets<ScenePatch>>()
                .get(&handle)
                .and_then(|p| p.resolved.as_ref())
                .is_some()
        });
        for _ in 0..10 {
            app.update();
        }

        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| app.world().get::<Position>(root).is_some());
        assert_eq!(app.world().get::<Position>(root).unwrap().x, 5.);
    }

    /// ADV-9: three-level chain `top : mid : base`. Reloading `base` must update the top-level
    /// instance's merged component, transitively.
    #[test]
    fn adv9_three_level_chain_reload_of_base() {
        let scenes = FakeScenes::default();
        scenes.write("base.fakescene", || bsn! { Position { x: 1., y: 1. } });
        scenes.write("mid.fakescene", || {
            bsn! { :"base.fakescene" Position { y: 2. } }
        });
        scenes.write("top.fakescene", || {
            bsn! { :"mid.fakescene" Position { x: 3. } }
        });
        let mut app = hot_reload_app(
            &scenes,
            &["base.fakescene", "mid.fakescene", "top.fakescene"],
        );

        let handle = load_and_settle(&mut app, "top.fakescene");
        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| app.world().get::<Position>(root).is_some());
        assert_eq!(
            *app.world().get::<Position>(root).unwrap(),
            Position { x: 3., y: 2. }
        );

        // `base` no longer sets x; `mid` still sets y. The top-level patch's x must win, and the
        // whole chain must be re-resolved so nothing is stale.
        scenes.write("base.fakescene", || bsn! { Position { x: 8., y: 8. } });
        reload(&app, "base.fakescene");
        for _ in 0..60 {
            app.update();
        }
        assert_eq!(
            *app.world().get::<Position>(root).unwrap(),
            Position { x: 3., y: 2. },
            "a three-level chain must stay correctly merged after the bottom reloads"
        );
    }

    /// ADV-10: the instance entity is despawned in the same frame the reload lands.
    #[test]
    fn adv10_instance_despawned_with_reload_pending() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Children [ #A ] });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let before = app.world().entities().count_spawned();
        let keep = app
            .world_mut()
            .spawn(ScenePatchInstance(handle.clone()))
            .id();
        let doomed = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| {
            app.world().get::<Children>(doomed).is_some()
        });

        scenes.write("a.fakescene", || bsn! { Children [ #B ] });
        reload(&app, "a.fakescene");
        // Despawn one instance while the reload is in flight.
        app.world_mut().entity_mut(doomed).despawn();
        run_app_until(&mut app, |app| child_names(app, keep) == ["B"]);
        for _ in 0..10 {
            app.update();
        }
        assert_eq!(
            app.world().entities().count_spawned(),
            before + 2,
            "the surviving instance plus its one child"
        );
    }

    /// ADV-11: `AssetEvent::Removed` for a scene that still has live instances, followed by a
    /// fresh load of the same path.
    #[test]
    fn adv11_removed_then_reloaded() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Position { x: 1. } });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);

        let handle = load_and_settle(&mut app, "a.fakescene");
        let root = app
            .world_mut()
            .spawn(ScenePatchInstance(handle.clone()))
            .id();
        run_app_until(&mut app, |app| app.world().get::<Position>(root).is_some());

        // Drop every strong handle except the instance's, then remove the asset outright.
        let id = handle.id();
        drop(handle);
        app.world_mut()
            .resource_mut::<Assets<ScenePatch>>()
            .remove(id);
        for _ in 0..10 {
            app.update();
        }
        assert_eq!(
            app.world().get::<Position>(root).unwrap().x,
            1.,
            "removing the asset leaves the last applied content in place"
        );
    }

    /// ADV-12: a scene applied to an instance queues another scene from inside the apply, while a
    /// reload of the outer asset is being processed in the same `spawn_queued` call.
    #[test]
    fn adv12_queue_from_within_apply_during_reload() {
        let scenes = FakeScenes::default();
        scenes.write("a.fakescene", || bsn! { Children [ #A ] });
        let mut app = hot_reload_app(&scenes, &["a.fakescene"]);
        let handle = load_and_settle(&mut app, "a.fakescene");
        let before = app.world().entities().count_spawned();

        let root = app.world_mut().spawn(ScenePatchInstance(handle)).id();
        run_app_until(&mut app, |app| child_names(app, root) == ["A"]);

        // An observer that reacts to the new generation of children by queuing another scene.
        app.world_mut()
            .add_observer(|add: On<Add<Name>>, mut commands: Commands| {
                commands
                    .entity(add.entity)
                    .queue_apply_scene(bsn! { HotMarker });
            });

        scenes.write("a.fakescene", || bsn! { Children [ #B -- #C ] });
        reload(&app, "a.fakescene");
        run_app_until(&mut app, |app| child_names(app, root) == ["B", "C"]);
        for _ in 0..10 {
            app.update();
        }

        let children: Vec<Entity> = app
            .world()
            .get::<Children>(root)
            .unwrap()
            .iter()
            .collect::<Vec<_>>();
        for child in &children {
            assert!(
                app.world().get::<HotMarker>(*child).is_some(),
                "a scene queued from inside an apply must still be applied"
            );
        }
        assert_eq!(
            app.world().entities().count_spawned(),
            // +1 for the observer entity itself.
            before + 4,
            "root + two children (+ the observer entity)"
        );
    }
}
