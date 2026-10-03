use bevy_reflect::TypePath;
use criterion::{criterion_group, Criterion};
use glam::Mat4;
use std::{path::Path, time::Duration};

use bevy_app::App;
use bevy_asset::{
    asset_value,
    io::{
        memory::{Dir, MemoryAssetReader},
        AssetSourceBuilder, AssetSourceId,
    },
    Asset, AssetApp, AssetLoader, AssetServer, Assets, Handle,
};
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::{Reflect, ReflectDefault};
use bevy_scene::{prelude::*, ScenePatch};
use bevy_ui::prelude::*;
use bevy_ui_widgets::Button;

criterion_group!(benches, spawn);

fn spawn(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(4));
    group.bench_function("ui_immediate_function_scene", |b| {
        let mut app = bench_app(|_| {}, |_| {});
        b.iter(move || {
            app.world_mut().spawn_scene(ui()).unwrap();
        });
    });
    group.bench_function("named_entity_reference", |b| {
        let mut app = bench_app(|_| {}, |_| {});
        b.iter(move || {
            app.world_mut().spawn_scene(named_passing()).unwrap();
        });
    });
    group.bench_function("ui_immediate_loaded_scene", |b| {
        let dir = Dir::default();
        let mut app = bench_app(
            |app| {
                in_memory_asset_source(dir.clone(), app);
            },
            |app| {
                app.register_asset_loader(FakeSceneLoader::new(button));
            },
        );

        // Insert an asset that the fake loader can fake read.
        dir.insert_asset_text(Path::new("button.fakescene"), "");

        let asset_server = app.world().resource::<AssetServer>().clone();
        let handle = asset_server.load("button.fakescene");

        run_app_until(&mut app, || asset_server.is_loaded(&handle));

        let patch = app
            .world()
            .resource::<Assets<ScenePatch>>()
            .get(&handle)
            .unwrap();
        assert!(patch.resolved.is_some());

        b.iter(move || {
            app.world_mut().spawn_scene(ui_loaded_asset()).unwrap();
        });

        drop(handle);
    });
    group.bench_function("ui_raw_bundle_no_scene", |b| {
        let mut app = bench_app(|_| {}, |_| {});

        b.iter(move || {
            app.world_mut().spawn(raw_ui());
        });
    });

    group.bench_function("handle_template_handle", |b| {
        let dir = Dir::default();
        let mut app = bench_app(
            |app| {
                in_memory_asset_source(dir.clone(), app);
            },
            |app| {
                app.init_asset::<EmptyAsset>();
                let assets = app.world().resource::<AssetServer>();
                let handles = (0..10).map(|_| assets.add(EmptyAsset)).collect::<Vec<_>>();
                app.register_asset_loader(FakeSceneLoader::new(move || {
                    asset_handle_scene(handles.clone())
                }));
            },
        );

        dir.insert_asset_text(Path::new("a.fakescene"), "");

        let asset_server = app.world().resource::<AssetServer>().clone();
        let handle = asset_server.load::<ScenePatch>("a.fakescene");

        run_app_until(&mut app, || asset_server.is_loaded(&handle));

        let world = app.world_mut();
        b.iter(|| {
            for _ in 0..100 {
                world.spawn_scene(bsn! { :"a.fakescene" }).unwrap();
            }
        });
    });

    group.bench_function("handle_template_value", |b| {
        let dir = Dir::default();
        let mut app = bench_app(
            |app| {
                in_memory_asset_source(dir.clone(), app);
            },
            |app| {
                app.register_asset_loader(FakeSceneLoader::new(asset_value_scene));
                app.init_asset::<EmptyAsset>();
            },
        );

        dir.insert_asset_text(Path::new("a.fakescene"), "");

        let asset_server = app.world().resource::<AssetServer>().clone();
        let handle = asset_server.load::<ScenePatch>("a.fakescene");

        run_app_until(&mut app, || asset_server.is_loaded(&handle));

        let world = app.world_mut();
        b.iter(|| {
            for _ in 0..100 {
                world.spawn_scene(bsn! { :"a.fakescene" }).unwrap();
            }
        });
    });

    // --- SPEC-6 guardrails -------------------------------------------------------------------
    //
    // `static_node_scene_spawn` vs `dynamic_node_scene_spawn` bounds the cost of the reflection
    // driven `.bsn` path: resolution is one-shot and cached in `ScenePatch::resolved`, so the
    // *apply* cost should be within noise of the statically-defined scene. A dynamic scene more
    // than ~1.25x the static one means a dynamic template is not reaching the bundle writer and
    // is causing per-component archetype moves instead.
    group.bench_function("static_node_scene_spawn", |b| {
        let mut app = bench_app(|_| {}, register_node_types);
        b.iter(move || {
            app.world_mut().spawn_scene(node_scene()).unwrap();
        });
    });
    group.bench_function("dynamic_node_scene_spawn", |b| {
        let dir = Dir::default();
        let mut app = bench_app(
            |app| {
                in_memory_asset_source(dir.clone(), app);
            },
            register_node_types,
        );
        dir.insert_asset_text(Path::new("nodes.bsn"), &node_scene_bsn());

        let asset_server = app.world().resource::<AssetServer>().clone();
        let handle = asset_server.load::<ScenePatch>("nodes.bsn");
        for _ in 0..LARGE_ITERATION_COUNT {
            app.update();
            if app_has_resolved(&app, &handle) {
                break;
            }
        }
        assert!(app_has_resolved(&app, &handle), "nodes.bsn never resolved");

        b.iter(move || {
            app.world_mut().spawn_scene(bsn! { :"nodes.bsn" }).unwrap();
        });
    });

    // The queued/instance path, which is what SPEC-6 changed: `ScenePatchInstance` now also
    // carries a `SceneInstanceState`, and every apply records the entities it spawned into it.
    // Budget: <= 3% against the pre-SPEC-6 baseline.
    group.bench_function("queued_scene_instance_spawn", |b| {
        let dir = Dir::default();
        let mut app = bench_app(
            |app| {
                in_memory_asset_source(dir.clone(), app);
            },
            |app| {
                app.register_asset_loader(FakeSceneLoader::new(button));
            },
        );
        dir.insert_asset_text(Path::new("button.fakescene"), "");

        let asset_server = app.world().resource::<AssetServer>().clone();
        let handle = asset_server.load::<ScenePatch>("button.fakescene");
        run_app_until(&mut app, || asset_server.is_loaded(&handle));

        b.iter(move || {
            // Batched so that the fixed cost of running the schedule is amortized across ten
            // instance applications rather than dominating the sample.
            for _ in 0..10 {
                app.world_mut().spawn(ScenePatchInstance(handle.clone()));
            }
            app.update();
        });
    });

    // One despawn + re-apply cycle of a scene instance. There is nothing to compare this to yet;
    // it is recorded so that state-preserving reconciliation has a baseline to beat.
    group.bench_function("scene_hot_reload", |b| {
        let dir = Dir::default();
        let mut app = bench_app(
            |app| {
                in_memory_asset_source(dir.clone(), app);
            },
            |app| {
                app.register_asset_loader(FakeSceneLoader::new(node_scene));
            },
        );
        dir.insert_asset_text(Path::new("hot.fakescene"), "");

        let asset_server = app.world().resource::<AssetServer>().clone();
        let handle = asset_server.load::<ScenePatch>("hot.fakescene");
        run_app_until(&mut app, || asset_server.is_loaded(&handle));

        let root = app
            .world_mut()
            .spawn(ScenePatchInstance(handle.clone()))
            .id();
        for _ in 0..LARGE_ITERATION_COUNT {
            app.update();
            if app.world().get::<Children>(root).is_some() {
                break;
            }
        }

        b.iter(move || {
            let before = app.world().get::<Children>(root).unwrap()[0];
            asset_server.reload("hot.fakescene");
            // The re-apply lands the frame the reload task completes; every scene entity is
            // respawned, so a changed first child is the signal.
            for _ in 0..LARGE_ITERATION_COUNT {
                app.update();
                if app.world().get::<Children>(root).unwrap()[0] != before {
                    break;
                }
            }
        });
    });

    group.finish();
}

/// The number of children in the scenes used by the SPEC-6 guardrail benches.
const NODE_COUNT: usize = 100;

#[derive(Component, Reflect, Clone, Default)]
#[reflect(Component, Default)]
struct NodePosition {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Component, Reflect, Clone, Default)]
#[reflect(Component, Default)]
struct NodeSize {
    width: f32,
    height: f32,
}

#[derive(Component, Reflect, Clone, Default)]
#[reflect(Component, Default)]
struct NodeLabel;

fn register_node_types(app: &mut App) {
    app.register_type::<NodePosition>();
    app.register_type::<NodeSize>();
    app.register_type::<NodeLabel>();
    app.register_type::<Children>();
}

/// A `NODE_COUNT`-child scene, defined statically.
fn node_scene() -> impl Scene {
    let children = (0..NODE_COUNT)
        .map(|_| {
            bsn! {
                NodePosition { x: 1.0, y: 2.0 }
                NodeSize { width: 3.0, height: 4.0 }
                NodeLabel
            }
        })
        .collect::<Vec<_>>();
    bsn! {
        NodePosition { x: 1.0 }
        Children [{children}]
    }
}

/// The exact same scene, written as `.bsn` text.
fn node_scene_bsn() -> String {
    let mut source = String::from(
        "NodePosition { x: 1.0 }
Children [
",
    );
    for _ in 0..NODE_COUNT {
        source.push_str(
            "  (NodePosition { x: 1.0, y: 2.0 } NodeSize { width: 3.0, height: 4.0 } NodeLabel),\n",
        );
    }
    source.push_str("]\n");
    source
}

/// Whether `handle`'s patch has been resolved yet.
fn app_has_resolved(app: &App, handle: &Handle<ScenePatch>) -> bool {
    app.world()
        .resource::<Assets<ScenePatch>>()
        .get(handle)
        .is_some_and(|patch| patch.resolved.is_some())
}

#[derive(Component, FromTemplate)]
#[expect(
    unused,
    reason = "this exists to store the Entity for benchmarking #Name references"
)]
struct Reference(Entity);

fn named_passing() -> impl Scene {
    bsn! {
        #Name
        Children [
            #Name0 Reference(#Name) Reference(#Name0)
            --
            #Name1 Reference(#Name) Reference(#Name1)
            --
            #Name2 Reference(#Name) Reference(#Name2)
            --
            #Name3 Reference(#Name) Reference(#Name3)
            --
            #Name4 Reference(#Name) Reference(#Name4)
            --
            #Name5 Reference(#Name) Reference(#Name5)
            --
            #Name6 Reference(#Name) Reference(#Name6)
            --
            #Name7 Reference(#Name) Reference(#Name7)
            --
            #Name8 Reference(#Name) Reference(#Name8)
            --
            #Name9 Reference(#Name) Reference(#Name9)
        ]
    }
}

#[derive(Asset, TypePath)]
struct EmptyAsset;

#[derive(Component, FromTemplate)]
#[expect(unused, reason = "this is just used for init")]
struct AssetReference(Handle<EmptyAsset>);

fn asset_value_scene() -> impl Scene {
    let children = (0..10)
        .map(|_| {
            bsn! {AssetReference(asset_value(EmptyAsset))}
        })
        .collect::<Vec<_>>();
    bsn! {
        Children [{children}]
    }
}

fn asset_handle_scene(mut handles: Vec<Handle<EmptyAsset>>) -> impl Scene {
    let children = handles
        .drain(..)
        .map(|handle| {
            bsn! {AssetReference({handle.clone()})}
        })
        .collect::<Vec<_>>();
    bsn! {
        Children [{children}]
    }
}

fn ui() -> impl Scene {
    bsn! {
        Node
        Children [
            @button() Node { width: Val::Px(200.) }
            --
            @button() Node { width: Val::Px(200.) }
            --
            @button() Node { width: Val::Px(200.) }
            --
            @button() Node { width: Val::Px(200.) }
            --
            @button() Node { width: Val::Px(200.) }
            --
            @button() Node { width: Val::Px(200.) }
            --
            @button() Node { width: Val::Px(200.) }
            --
            @button() Node { width: Val::Px(200.) }
            --
            @button() Node { width: Val::Px(200.) }
            --
            @button() Node { width: Val::Px(200.) }
        ]
    }
}

fn ui_loaded_asset() -> impl Scene {
    bsn! {
        Node
        Children [
            :"button.fakescene" Node { width: Val::Px(200.) }
            --
            :"button.fakescene" Node { width: Val::Px(200.) }
            --
            :"button.fakescene" Node { width: Val::Px(200.) }
            --
            :"button.fakescene" Node { width: Val::Px(200.) }
            --
            :"button.fakescene" Node { width: Val::Px(200.) }
            --
            :"button.fakescene" Node { width: Val::Px(200.) }
            --
            :"button.fakescene" Node { width: Val::Px(200.) }
            --
            :"button.fakescene" Node { width: Val::Px(200.) }
            --
            :"button.fakescene" Node { width: Val::Px(200.) }
            --
            :"button.fakescene" Node { width: Val::Px(200.) }
        ]
    }
}

// A non-Node component that we add to force archetype moves, inflating their cost if/when they happen
#[derive(Component, Default, Clone)]
struct Marker;

#[derive(Component, Default, Clone)]
#[expect(unused, reason = "this exists to take up space")]
struct Marker1(Mat4);
#[derive(Component, Default, Clone)]
#[expect(unused, reason = "this exists to take up space")]
struct Marker2(Mat4);
#[derive(Component, Default, Clone)]
#[expect(unused, reason = "this exists to take up space")]
struct Marker3(Mat4);

fn button() -> impl Scene {
    bsn! {
        Button
        Node {
            width: Val::Px(150.0),
            height: Val::Px(65.0),
            border: UiRect::all(Val::Px(5.0)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
        }
        Children [
            Text("Text") Marker Marker1 Marker2 Marker3
            --
            Text("Text") Marker Marker1 Marker2 Marker3
            --
            Text("Text") Marker Marker1 Marker2 Marker3
            --
            Text("Text") Marker Marker1 Marker2 Marker3
            --
            Text("Text") Marker Marker1 Marker2 Marker3
            --
            Text("Text") Marker Marker1 Marker2 Marker3
            --
            Text("Text") Marker Marker1 Marker2 Marker3
            --
            Text("Text") Marker Marker1 Marker2 Marker3
            --
            Text("Text") Marker Marker1 Marker2 Marker3
            --
            Text("Text") Marker Marker1 Marker2 Marker3
        ]
    }
}

fn raw_button() -> impl Bundle {
    (
        Button,
        Node {
            width: Val::Px(200.0),
            height: Val::Px(65.0),
            border: UiRect::all(Val::Px(5.0)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..Default::default()
        },
        children![
            (
                Text("Text".into()),
                Marker,
                Marker1::default(),
                Marker2::default(),
                Marker3::default()
            ),
            (
                Text("Text".into()),
                Marker,
                Marker1::default(),
                Marker2::default(),
                Marker3::default()
            ),
            (
                Text("Text".into()),
                Marker,
                Marker1::default(),
                Marker2::default(),
                Marker3::default()
            ),
            (
                Text("Text".into()),
                Marker,
                Marker1::default(),
                Marker2::default(),
                Marker3::default()
            ),
            (
                Text("Text".into()),
                Marker,
                Marker1::default(),
                Marker2::default(),
                Marker3::default()
            ),
            (
                Text("Text".into()),
                Marker,
                Marker1::default(),
                Marker2::default(),
                Marker3::default()
            ),
            (
                Text("Text".into()),
                Marker,
                Marker1::default(),
                Marker2::default(),
                Marker3::default()
            ),
            (
                Text("Text".into()),
                Marker,
                Marker1::default(),
                Marker2::default(),
                Marker3::default()
            ),
            (
                Text("Text".into()),
                Marker,
                Marker1::default(),
                Marker2::default(),
                Marker3::default()
            ),
            (
                Text("Text".into()),
                Marker,
                Marker1::default(),
                Marker2::default(),
                Marker3::default()
            ),
        ],
    )
}

fn raw_ui() -> impl Bundle {
    (
        Node::default(),
        children![
            raw_button(),
            raw_button(),
            raw_button(),
            raw_button(),
            raw_button(),
            raw_button(),
            raw_button(),
            raw_button(),
            raw_button(),
            raw_button(),
        ],
    )
}

/// The frame bound used by [`run_app_until`] and the hot-reload bench.
const LARGE_ITERATION_COUNT: usize = 10000;

/// Fork of `bevy_asset::tests::run_app_until`.
fn run_app_until(app: &mut App, mut predicate: impl FnMut() -> bool) {
    for _ in 0..LARGE_ITERATION_COUNT {
        app.update();
        if predicate() {
            return;
        }
    }

    panic!("Ran out of loops to return `Some` from `predicate`");
}

fn bench_app(before: impl FnOnce(&mut App), after: impl FnOnce(&mut App)) -> App {
    let mut app = App::new();
    before(&mut app);
    app.add_plugins((
        bevy_app::TaskPoolPlugin::default(),
        bevy_asset::AssetPlugin::default(),
        bevy_scene::ScenePlugin,
        bevy_bsn_asset::BsnAssetPlugin,
    ));
    after(&mut app);
    app.finish();
    app.cleanup();
    app
}

fn in_memory_asset_source(dir: Dir, app: &mut App) {
    app.register_asset_source(
        AssetSourceId::Default,
        AssetSourceBuilder::new(move || Box::new(MemoryAssetReader { root: dir.clone() })),
    );
}

#[derive(TypePath)]
struct FakeSceneLoader(Box<dyn Fn() -> Box<dyn Scene> + Send + Sync>);

impl FakeSceneLoader {
    pub fn new<S: Scene>(scene_fn: impl (Fn() -> S) + Send + Sync + 'static) -> Self {
        Self(Box::new(move || Box::new(scene_fn())))
    }
}

impl AssetLoader for FakeSceneLoader {
    type Asset = ScenePatch;
    type Error = std::io::Error;
    type Settings = ();

    fn extensions(&self) -> &[&str] {
        // Distinct from `bsn`: the real `DynamicBsnLoader` would otherwise win the
        // extension match and these benches would measure empty scenes.
        &["fakescene"]
    }

    async fn load(
        &self,
        _reader: &mut dyn bevy_asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut bevy_asset::LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        Ok(ScenePatch::load_with(load_context, (self.0)()))
    }
}
