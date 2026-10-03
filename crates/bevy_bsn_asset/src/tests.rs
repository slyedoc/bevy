//! Fixtures shared by the `dynamic` module's tests, plus the end-to-end tests that spawn a
//! document-built scene into a real [`World`](bevy_ecs::world::World).

use std::path::Path;

use bevy_app::App;
use bevy_asset::{Asset, AssetApp, AssetServer, Assets, Handle, HandleTemplate, ReflectHandle};
use bevy_bsn::{parse, BsnDocument};
use bevy_ecs::{
    entity::Entity,
    hierarchy::{ChildOf, Children},
    name::Name,
    prelude::Component,
    reflect::{
        AppTypeRegistry, ReflectComponent, ReflectFromTemplate, ReflectRelationshipTarget,
        ReflectTemplate,
    },
    relationship::RelationshipTarget,
    template::FromTemplate,
};
use bevy_reflect::{std_traits::ReflectDefault, Reflect, TypeRegistry};
use core::any::TypeId;

use crate::{test_support::memory_asset_app, DynamicScene};
use bevy_scene::{bsn, ResolvedSceneRoot, Scene, ScenePatch, WorldSceneExt};

// ---------------------------------------------------------------------------------------
// Fixture types
// ---------------------------------------------------------------------------------------

/// A three-field component whose template is itself (the `Clone + Default` blanket).
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct Position {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
}

/// A component with a nested struct field, for nested-partial-patch tests.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct Foo {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) z: u32,
    pub(crate) nested: Bar,
}

/// A three-field tuple struct, used both as a component and as `Foo`'s nested field.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct Bar(pub(crate) usize, pub(crate) usize, pub(crate) usize);

/// A two-field tuple struct of mixed types.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct TupleStruct(pub(crate) f32, pub(crate) u32);

/// A field-less component.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct Marker;

/// An enum component with one variant of each kind.
///
/// This one goes through a generated template (rather than the `Clone + Default` blanket) because
/// the `bsn!` macro's enum patching needs the generated `default_<variant>` constructors.
#[derive(Component, FromTemplate, Reflect, PartialEq, Debug)]
#[template(reflect)]
#[reflect(Component, FromTemplate)]
pub(crate) enum Choice {
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

/// A component with a field of every primitive type.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct Primitives {
    pub(crate) a_i8: i8,
    pub(crate) a_i16: i16,
    pub(crate) a_i32: i32,
    pub(crate) a_i64: i64,
    pub(crate) a_i128: i128,
    pub(crate) a_isize: isize,
    pub(crate) a_u8: u8,
    pub(crate) a_u16: u16,
    pub(crate) a_u32: u32,
    pub(crate) a_u64: u64,
    pub(crate) a_u128: u128,
    pub(crate) a_usize: usize,
    pub(crate) a_f32: f32,
    pub(crate) a_f64: f64,
    pub(crate) a_bool: bool,
    pub(crate) a_string: String,
}

/// A component with `Option` and `Vec` fields.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct Collections {
    pub(crate) maybe: Option<u32>,
    pub(crate) list: Vec<u8>,
}

/// A component holding an entity reference, so `#Name` values have somewhere to go.
#[derive(Component, FromTemplate, Reflect, PartialEq, Debug)]
#[template(reflect)]
#[reflect(Component, FromTemplate)]
pub(crate) struct Reference(pub(crate) Entity);

/// A stand-in asset type.
#[derive(Asset, Reflect, Default)]
pub(crate) struct Image;

/// A component holding an asset handle, so string-to-handle conversion has somewhere to go.
#[derive(Component, FromTemplate, Reflect, PartialEq, Debug)]
#[template(reflect)]
#[reflect(Component, FromTemplate)]
pub(crate) struct Sprite(pub(crate) Handle<Image>);

/// The relationship of the [`Items`] custom relationship target.
#[derive(Component, Reflect)]
#[relationship(relationship_target = Items)]
#[reflect(Component)]
pub(crate) struct ItemOf(pub(crate) Entity);

/// A relationship target that is not [`Children`].
#[derive(Component, Reflect)]
#[relationship_target(relationship = ItemOf)]
#[reflect(Component, RelationshipTarget)]
pub(crate) struct Items(Vec<Entity>);

/// An enum used only as the *source* of a registered conversion.
#[derive(Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Default)]
pub(crate) enum TextSize {
    /// The small size.
    #[default]
    Small,
    /// The large size.
    Large,
}

/// The *destination* of the [`TextSize`] conversion.
#[derive(Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Default)]
pub(crate) struct FontSize(pub(crate) u32);

/// A struct that converts into a [`FontSize`], for the "different type, then converted" path.
#[derive(Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Default)]
pub(crate) struct FontSizeSource {
    pub(crate) value: u32,
}

/// A component with a [`FontSize`] field, to exercise implicit conversions on field values.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
pub(crate) struct TextFont {
    pub(crate) font_size: FontSize,
}

/// A type that is registered *without* `ReflectDefault`.
#[derive(Reflect, Clone, PartialEq, Debug)]
pub(crate) struct NoDefault(pub(crate) u32);

/// An enum whose struct variant has a field type that cannot be default-constructed reflectively.
#[derive(Component, Reflect, Clone, PartialEq, Debug)]
#[reflect(Component)]
pub(crate) enum Tricky {
    /// A variant whose field type has no `ReflectDefault`.
    A {
        /// The un-defaultable field.
        field: NoDefault,
    },
    /// A plain variant.
    B,
}

/// A non-cloneable field type, used to force `reflect_clone` to fail.
#[derive(Default)]
pub(crate) struct NotClone;

/// A component whose ignored field makes `reflect_clone` fail, exercising the `clone_template`
/// fallback ladder.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Default)]
pub(crate) struct Skipped {
    pub(crate) value: u32,
    #[reflect(ignore)]
    #[expect(
        dead_code,
        reason = "the field exists only to make `reflect_clone` fail"
    )]
    pub(crate) extra: NotClone,
}

/// A component whose template is not reflectable at all, for the unpatchable-slot error path.
#[derive(Component, Clone, Default)]
pub(crate) struct Unreflected(
    #[expect(
        dead_code,
        reason = "the field exists only to give the template a shape"
    )]
    pub(crate) u32,
);

/// Registers every fixture type in a bare [`TypeRegistry`], for unit tests that need no `World`.
pub(crate) fn test_registry() -> TypeRegistry {
    let mut registry = TypeRegistry::default();
    registry.register::<Position>();
    registry.register::<Foo>();
    registry.register::<Bar>();
    registry.register::<TupleStruct>();
    registry.register::<Marker>();
    registry.register::<Choice>();
    registry.register::<ChoiceTemplate>();
    registry.register::<Primitives>();
    registry.register::<Collections>();
    registry.register::<Reference>();
    registry.register::<ReferenceTemplate>();
    registry.register::<Sprite>();
    registry.register::<SpriteTemplate>();
    registry.register::<Items>();
    registry.register::<ItemOf>();
    registry.register::<Children>();
    registry.register::<ChildOf>();
    registry.register::<Name>();
    registry.register::<TextSize>();
    registry.register::<FontSize>();
    registry.register::<TextFont>();
    registry.register::<FontSizeSource>();
    registry.register::<NoDefault>();
    registry.register::<Tricky>();
    registry.register::<Skipped>();
    registry.register::<&'static str>();
    registry.register::<alloc::borrow::Cow<'static, str>>();
    registry.register::<Vec<Vec<u8>>>();
    register_image_asset(&mut registry);
    registry.register_type_conversion::<TextSize, FontSize, _>(|size| {
        Ok(FontSize(match size {
            TextSize::Small => 12,
            TextSize::Large => 24,
        }))
    });
    registry.register_type_conversion::<FontSizeSource, FontSize, _>(|source| {
        Ok(FontSize(source.value))
    });
    registry
}

/// Mirrors what `AssetApp::register_asset_reflect::<Image>` does, without needing an [`App`].
pub(crate) fn register_image_asset(registry: &mut TypeRegistry) {
    registry.register::<Image>();
    registry.register::<Handle<Image>>();
    registry.register::<HandleTemplate<Image>>();
    registry.register_type_data::<Handle<Image>, ReflectHandle>();
    registry.register_type_data::<HandleTemplate<Image>, ReflectTemplate>();
    registry.register_type_conversion::<String, HandleTemplate<Image>, _>(|path| Ok(path.into()));
}

/// Parses a `.bsn` source snippet, panicking with a rendered diagnostic if it does not parse.
pub(crate) fn doc(source: &str) -> BsnDocument {
    match parse(source) {
        Ok(document) => document,
        Err(error) => panic!("{}", error.render(source, None)),
    }
}

/// An [`App`] with the asset and scene plugins plus every fixture type registered.
pub(crate) fn test_app() -> App {
    let mut app = crate::test_support::test_app();
    app.init_asset::<Image>();
    app.register_asset_reflect::<Image>();
    register_fixtures(&mut app);
    app
}

/// Registers every fixture type on an [`App`].
pub(crate) fn register_fixtures(app: &mut App) {
    app.register_type::<Position>()
        .register_type::<Foo>()
        .register_type::<Bar>()
        .register_type::<TupleStruct>()
        .register_type::<Marker>()
        .register_type::<Choice>()
        .register_type::<ChoiceTemplate>()
        .register_type::<Primitives>()
        .register_type::<Collections>()
        .register_type::<Reference>()
        .register_type::<ReferenceTemplate>()
        .register_type::<Sprite>()
        .register_type::<SpriteTemplate>()
        .register_type::<Items>()
        .register_type::<ItemOf>()
        .register_type::<Children>()
        .register_type::<ChildOf>()
        .register_type::<Name>()
        .register_type::<TextSize>()
        .register_type::<FontSize>()
        .register_type::<TextFont>()
        .register_type::<FontSizeSource>()
        .register_type::<NoDefault>()
        .register_type::<Tricky>()
        .register_type::<Skipped>();
    app.register_type_conversion::<TextSize, FontSize, _>(|size| {
        Ok(FontSize(match size {
            TextSize::Small => 12,
            TextSize::Large => 24,
        }))
    });
}

/// Builds a [`DynamicScene`] from `source` using `app`'s type registry.
pub(crate) fn scene(app: &App, path: &str, source: &str) -> DynamicScene {
    let document = doc(source);
    let registry = app.world().resource::<AppTypeRegistry>().clone();
    match DynamicScene::from_document(&document, path, &registry) {
        Ok(scene) => scene,
        Err(error) => panic!("{}", error.render(source)),
    }
}

// ---------------------------------------------------------------------------------------
// End-to-end tests
// ---------------------------------------------------------------------------------------

#[test]
fn dynamic_struct_patching() {
    let mut app = test_app();
    let a = scene(&app, "a.bsn", "Foo { x: 1, nested: Bar(1, 1) }");
    let b = scene(&app, "b.bsn", "Foo { y: 2, nested: Bar(2) }");

    let world = app.world_mut();
    let id = world.spawn_scene((a, b)).unwrap().id();

    assert_eq!(
        *world.entity(id).get::<Foo>().unwrap(),
        Foo {
            x: 1,
            y: 2,
            z: 0,
            nested: Bar(2, 1, 0)
        }
    );
}

#[test]
fn dynamic_field_patching_with_default() {
    let mut app = test_app();
    let a = scene(&app, "a.bsn", "Bar(1, 1)");
    let b = scene(&app, "b.bsn", "Bar(2)");

    let world = app.world_mut();
    let id = world.spawn_scene((a, b)).unwrap().id();

    assert_eq!(*world.entity(id).get::<Bar>().unwrap(), Bar(2, 1, 0));
}

#[test]
fn dynamic_partial_tuple_struct() {
    let mut app = test_app();
    let a = scene(&app, "a.bsn", "TupleStruct(0.1)");

    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();

    let value = world.entity(id).get::<TupleStruct>().unwrap();
    assert_eq!(value.0, 0.1);
    assert_eq!(value.1, 0);
}

#[test]
fn dynamic_enum_patching() {
    let mut app = test_app();
    let baz = scene(&app, "a.bsn", "Choice::Baz(10)");
    let bar_x = scene(&app, "b.bsn", "Choice::Bar { x: 1 }");
    let bar_y = scene(&app, "c.bsn", "Choice::Bar { y: 2 }");
    let qux = scene(&app, "d.bsn", "Choice::Qux");

    let world = app.world_mut();

    // `Baz`, then a switch to `Bar` (a full variant construction), then a same-variant patch.
    let id = world
        .spawn_scene((baz.clone(), bar_x.clone(), bar_y))
        .unwrap()
        .id();
    assert_eq!(
        *world.entity(id).get::<Choice>().unwrap(),
        Choice::Bar { x: 1, y: 2, z: 0 }
    );

    let id = world.spawn_scene(baz).unwrap().id();
    assert_eq!(*world.entity(id).get::<Choice>().unwrap(), Choice::Baz(10));

    let id = world.spawn_scene((bar_x, qux)).unwrap().id();
    assert_eq!(*world.entity(id).get::<Choice>().unwrap(), Choice::Qux);
}

#[test]
fn dynamic_primitive_literals() {
    let mut app = test_app();
    let a = scene(
        &app,
        "a.bsn",
        r#"Primitives {
            a_i8: -1, a_i16: -2, a_i32: -3, a_i64: -4, a_i128: -5, a_isize: -6,
            a_u8: 1, a_u16: 2, a_u32: 3, a_u64: 4, a_u128: 5, a_usize: 6,
            a_f32: 1.5, a_f64: -2.5, a_bool: true, a_string: "hello",
        }"#,
    );

    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();

    assert_eq!(
        *world.entity(id).get::<Primitives>().unwrap(),
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
}

#[test]
fn dynamic_collection_literals() {
    let mut app = test_app();
    let a = scene(&app, "a.bsn", "Collections { maybe: 5, list: [1, 2] }");

    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();

    assert_eq!(
        *world.entity(id).get::<Collections>().unwrap(),
        Collections {
            maybe: Some(5),
            list: vec![1, 2],
        }
    );
}

#[test]
fn dynamic_handle_template() {
    let mut app = test_app();
    let expected = app.world().resource::<AssetServer>().load("image.png");
    let a = scene(&app, "a.bsn", r#"Sprite("image.png")"#);

    // The asset path became a tracked dependency of the scene.
    let patch = ScenePatch::load(app.world().resource::<AssetServer>(), a.clone());
    assert_eq!(patch.dependencies.len(), 1);

    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    assert_eq!(world.entity(id).get::<Sprite>().unwrap().0, expected);
}

#[test]
fn dynamic_hierarchy() {
    let mut app = test_app();
    let a = scene(
        &app,
        "a.bsn",
        "#A Children [ (#B Children [ #X ]), (#C Children [ #Y ]) ]",
    );

    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();

    let root = world.entity(id);
    assert_eq!(root.get::<Name>().unwrap().as_str(), "A");
    let children = root.get::<Children>().unwrap();
    assert_eq!(children.len(), 2);

    let b = world.entity(children[0]);
    let c = world.entity(children[1]);
    assert_eq!(b.get::<Name>().unwrap().as_str(), "B");
    assert_eq!(c.get::<Name>().unwrap().as_str(), "C");

    let x = world.entity(b.get::<Children>().unwrap()[0]);
    assert_eq!(x.get::<Name>().unwrap().as_str(), "X");
    let y = world.entity(c.get::<Children>().unwrap()[0]);
    assert_eq!(y.get::<Name>().unwrap().as_str(), "Y");
}

#[test]
fn dynamic_name_references() {
    let mut app = test_app();
    let a = scene(
        &app,
        "a.bsn",
        "#Root Children [ (#Child Reference(#Root)), Reference(#Child) ]",
    );

    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();

    let children = world.entity(id).get::<Children>().unwrap();
    assert_eq!(children.len(), 2);
    let child = children[0];
    let other = children[1];

    assert_eq!(world.entity(child).get::<Reference>().unwrap().0, id);
    assert_eq!(world.entity(other).get::<Reference>().unwrap().0, child);
}

#[test]
fn dynamic_custom_relationship() {
    let mut app = test_app();
    let a = scene(&app, "a.bsn", "#Root Items [ #First, #Second ]");

    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();

    let items: Vec<_> = world.entity(id).get::<Items>().unwrap().iter().collect();
    let names: Vec<_> = items
        .iter()
        .map(|item| {
            world
                .entity(*item)
                .get::<Name>()
                .unwrap()
                .as_str()
                .to_string()
        })
        .collect();
    assert_eq!(names, ["First", "Second"]);
    assert!(world.entity(id).get::<Children>().is_none());
}

#[test]
fn dynamic_and_static_children_share_one_collection() {
    let mut app = test_app();
    let dynamic = scene(&app, "a.bsn", "Children [ #X ]");

    let world = app.world_mut();
    let id = world
        .spawn_scene(bsn! {
            @{dynamic}
            Children [ #Y ]
        })
        .unwrap()
        .id();

    // Both relation blocks are keyed by `TypeId::of::<ChildOf>()`, so they land in one
    // `RelatedResolvedScenes` and spawn as one contiguous group in document order.
    let children = world.entity(id).get::<Children>().unwrap();
    assert_eq!(children.len(), 2);
    assert_eq!(
        world.entity(children[0]).get::<Name>().unwrap().as_str(),
        "X"
    );
    assert_eq!(
        world.entity(children[1]).get::<Name>().unwrap().as_str(),
        "Y"
    );
}

#[test]
fn dynamic_fully_qualified_and_short_paths() {
    let mut app = test_app();
    let a = scene(
        &app,
        "a.bsn",
        "bevy_ecs::hierarchy::Children [ #First ]\nMarker",
    );

    let world = app.world_mut();
    let id = world.spawn_scene(a).unwrap().id();
    assert_eq!(world.entity(id).get::<Children>().unwrap().len(), 1);
    assert!(world.entity(id).get::<Marker>().is_some());
}

#[test]
fn dynamic_over_static_patch() {
    let mut app = test_app();
    let dynamic = scene(&app, "a.bsn", "Position { x: 1.0 }");

    let world = app.world_mut();
    let id = world
        .spawn_scene(bsn! {
            Position { y: 2. }
            @{dynamic}
        })
        .unwrap()
        .id();

    let position = world.entity(id).get::<Position>().unwrap();
    assert_eq!((position.x, position.y, position.z), (1., 2., 0.));
}

#[test]
fn static_over_dynamic_patch_merges() {
    let mut app = test_app();
    let dynamic = scene(&app, "a.bsn", "Position { y: 2.0 }");

    let world = app.world_mut();
    let id = world
        .spawn_scene(bsn! {
            @{dynamic}
            Position { x: 1. }
        })
        .unwrap()
        .id();

    let position = world.entity(id).get::<Position>().unwrap();
    assert_eq!((position.x, position.y, position.z), (1., 2., 0.));
}

#[test]
fn static_over_dynamic_patch_without_registry_resets() {
    let mut app = test_app();
    let dynamic = scene(&app, "a.bsn", "Position { y: 2.0 }");

    // Resolving by hand without a registry is the one path that cannot recover the dynamic
    // template's values. It must degrade to `Default`, not panic.
    let boxed: Box<dyn Scene> = Box::new(bsn! {
        @{dynamic}
        Position { x: 1. }
    });
    let world = app.world_mut();
    let assets = world.resource::<AssetServer>().clone();
    let resolved = {
        let patches = world.resource::<Assets<ScenePatch>>();
        ResolvedSceneRoot::resolve(boxed, &assets, patches, None).unwrap()
    };
    let id = resolved.spawn(world).unwrap().id();

    let position = world.entity(id).get::<Position>().unwrap();
    assert_eq!((position.x, position.y, position.z), (1., 0., 0.));
}

#[test]
fn dynamic_cached_base_patching() {
    let mut app = base_asset_app("Position { y: 2.0 }\nChildren [ #X ]");
    let b = scene(
        &app,
        "b.bsn",
        ":\"a.bsn\"\nPosition { x: 1.0 }\nChildren [ #Y ]",
    );

    let world = app.world_mut();
    let id = world.spawn_scene(b).unwrap().id();

    let root = world.entity(id);
    let position = root.get::<Position>().unwrap();
    assert_eq!((position.x, position.y, position.z), (1., 2., 0.));

    // The base's `Position` template was cloned into the local scene, so applying the cached scene
    // must skip it: `Position` is present exactly once, not pushed twice with the same
    // `ComponentId`. (`ErasedTemplate::template_type_id` is what makes the skip match.)
    let position_id = world
        .components()
        .get_id(TypeId::of::<Position>())
        .expect("Position should be registered after spawning");
    assert_eq!(
        root.archetype()
            .components()
            .iter()
            .filter(|id| **id == position_id)
            .count(),
        1
    );

    let children = root.get::<Children>().unwrap();
    assert_eq!(children.len(), 2);
    assert_eq!(
        world.entity(children[0]).get::<Name>().unwrap().as_str(),
        "X"
    );
    assert_eq!(
        world.entity(children[1]).get::<Name>().unwrap().as_str(),
        "Y"
    );
}

#[test]
fn static_cached_dynamic_base() {
    let mut app = base_asset_app("Position { y: 2.0 }\nChildren [ #X ]");

    let world = app.world_mut();
    let id = world
        .spawn_scene(bsn! {
            :"a.bsn"
            Position { x: 1. }
            Children [ #Y ]
        })
        .unwrap()
        .id();

    let root = world.entity(id);
    let position = root.get::<Position>().unwrap();
    assert_eq!((position.x, position.y, position.z), (1., 2., 0.));

    let children = root.get::<Children>().unwrap();
    assert_eq!(children.len(), 2);
    assert_eq!(
        world.entity(children[0]).get::<Name>().unwrap().as_str(),
        "X"
    );
    assert_eq!(
        world.entity(children[1]).get::<Name>().unwrap().as_str(),
        "Y"
    );
}

#[test]
fn static_cached_dynamic_base_enum_component() {
    let mut app = base_asset_app("Choice::Bar { y: 2 }");

    let world = app.world_mut();
    let id = world
        .spawn_scene(bsn! {
            :"a.bsn"
            Choice::Bar { x: 1, y: 0, z: 0 }
        })
        .unwrap()
        .id();

    // A `bsn!` enum variant is a full literal, so it replaces the base's variant outright.
    assert_eq!(
        *world.entity(id).get::<Choice>().unwrap(),
        Choice::Bar { x: 1, y: 0, z: 0 }
    );
}

#[test]
fn dynamic_children_of_cached_base_are_appended() {
    let mut app = base_asset_app("Children [ #X ]");
    let b = scene(&app, "b.bsn", ":\"a.bsn\"\nChildren [ #Y ]");

    let world = app.world_mut();
    let id = world.spawn_scene(b).unwrap().id();

    let children = world.entity(id).get::<Children>().unwrap();
    assert_eq!(children.len(), 2);
    assert_eq!(
        world.entity(children[0]).get::<Name>().unwrap().as_str(),
        "X"
    );
    assert_eq!(
        world.entity(children[1]).get::<Name>().unwrap().as_str(),
        "Y"
    );
}

#[test]
fn dynamic_missing_base_dependency_errors() {
    let mut app = test_app();
    let b = scene(&app, "b.bsn", ":\"missing.bsn\"\nMarker");

    let world = app.world_mut();
    let Err(error) = world.spawn_scene(b) else {
        panic!("resolving a scene whose base has not loaded must fail");
    };
    assert!(
        format!("{error}").contains("missing.bsn"),
        "unexpected error: {error}"
    );
}

#[test]
fn dynamic_late_base_errors() {
    // A base include is only legal as an entity's first entry. The parser rejects a late one, and
    // `ResolvedScene::include_cached` rejects it again for hand-built documents.
    assert!(parse("Marker\n:\"a.bsn\"").is_err());
}

#[test]
fn dynamic_child_with_a_missing_base_errors() {
    // A failure inside a relation block has to propagate out of the whole resolve, not be
    // swallowed by the loop that restores `context.cached`.
    let mut app = test_app();
    let b = scene(
        &app,
        "b.bsn",
        "Children [ (:\"missing.bsn\" Marker), (Marker) ]",
    );

    let world = app.world_mut();
    let Err(error) = world.spawn_scene(b) else {
        panic!("resolving a child whose base has not loaded must fail");
    };
    assert!(
        format!("{error}").contains("missing.bsn"),
        "unexpected error: {error}"
    );
}

#[test]
fn dynamic_scene_is_clone_send_sync_and_scene() {
    fn assert_scene<S: Scene + Clone + Send + Sync + 'static>() {}
    assert_scene::<DynamicScene>();

    // The `Debug` impl names the source asset, which is what makes a scene identifiable in a log.
    let app = test_app();
    let text = format!("{:?}", scene(&app, "a.bsn", r#"Sprite("image.png")"#));
    assert!(text.contains("a.bsn"), "unexpected debug output: {text}");
    assert!(
        text.contains("dependencies"),
        "unexpected debug output: {text}"
    );
}

/// An [`App`] whose asset source contains an `a.bsn` that loads as a dynamic scene.
fn base_asset_app(base_source: &'static str) -> App {
    let (mut app, dir) = memory_asset_app();
    app.init_asset::<Image>();
    app.register_asset_reflect::<Image>();
    register_fixtures(&mut app);
    app.finish();
    app.cleanup();

    // The real `.bsn` loader, registered by `ScenePlugin`, parses the file below.
    dir.insert_asset_text(Path::new("a.bsn"), base_source);

    let asset_server = app.world().resource::<AssetServer>().clone();
    let handle = asset_server.load::<ScenePatch>("a.bsn");
    for frame in 0..10_000 {
        app.update();
        if frame >= 100 {
            std::thread::sleep(core::time::Duration::from_millis(1));
        }
        if asset_server.is_loaded(&handle) {
            break;
        }
    }
    assert!(
        asset_server.is_loaded(&handle),
        "the base scene asset never finished loading"
    );
    app
}
