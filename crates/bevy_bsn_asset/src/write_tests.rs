//! The writer: what it writes, and that what it writes loads back into the same entities.

use bevy_ecs::{entity::Entity, hierarchy::Children, name::Name, world::World};
use bevy_scene::WorldSceneExt;

use crate::{
    test_support::test_app,
    tests::{
        base_asset_app, register_fixtures, scene, Bar, Choice, Collections, Foo, Marker, Position,
        Reference,
    },
    write_scene_text, WriteSettings,
};

fn write(world: &mut World, root: Entity) -> String {
    write_scene_text(world, root, &WriteSettings::default()).expect("writes")
}

fn named(world: &World, parent: Entity, name: &str) -> Entity {
    world
        .get::<Children>(parent)
        .into_iter()
        .flat_map(|c| c.iter().copied())
        .find(|&c| world.get::<Name>(c).is_some_and(|n| n.as_str() == name))
        .unwrap_or_else(|| panic!("no child named {name}"))
}

#[test]
fn a_component_writes_only_the_fields_that_differ_from_its_default() {
    let mut app = test_app();
    register_fixtures(&mut app);
    let source = scene(&app, "a.bsn", "Position { y: 2.0 }\nMarker");
    let root = app.world_mut().spawn_scene(source).unwrap().id();
    let text = write(app.world_mut(), root);
    assert!(text.contains("y: 2.0"), "{text}");
    assert!(!text.contains("x:"), "{text}");
    assert!(text.contains("Marker"), "{text}");
}

#[test]
fn a_written_scene_loads_back_into_the_same_entities() {
    let mut app = test_app();
    register_fixtures(&mut app);
    let source = scene(
        &app,
        "a.bsn",
        r#"
#Root
Foo { x: 1, nested: Bar(4, 5, 6) }
Choice::Baz(3)
Collections { maybe: Some(2), list: [1, 2, 3] }
Children [
    #Target
    Position { x: 1.5, z: -2.0 }
    Marker
    --
    #Pointer
    Reference(#Target)
]
"#,
    );
    let root = app.world_mut().spawn_scene(source).unwrap().id();
    let text = write(app.world_mut(), root);
    let again = scene(&app, "b.bsn", &text);
    let copy = app.world_mut().spawn_scene(again).unwrap().id();

    let world = app.world();
    for (a, b) in [(root, copy)] {
        assert_eq!(world.get::<Foo>(a), world.get::<Foo>(b), "{text}");
        assert_eq!(world.get::<Choice>(a), world.get::<Choice>(b), "{text}");
        assert_eq!(world.get::<Collections>(a), world.get::<Collections>(b), "{text}");
    }
    let (target, target_copy) = (named(world, root, "Target"), named(world, copy, "Target"));
    assert_eq!(world.get::<Position>(target), world.get::<Position>(target_copy));
    assert!(world.get::<Marker>(target_copy).is_some());
    let pointer_copy = named(world, copy, "Pointer");
    assert_eq!(
        world.get::<Reference>(pointer_copy).map(|r| r.0),
        Some(target_copy),
        "the reference follows to the copy's own target:\n{text}"
    );
    assert_eq!(world.get::<Foo>(copy).map(|f| f.nested.clone()), Some(Bar(4, 5, 6)));
}

#[test]
fn an_inheriting_entity_writes_its_base_and_only_what_differs() {
    let mut app = base_asset_app("Position { y: 2.0, z: 3.0 }\nMarker\nChildren [ #X ]");
    // An open scene holds its base; so does this test.
    let _base = app
        .world()
        .resource::<bevy_asset::AssetServer>()
        .load::<bevy_scene::ScenePatch>("a.bsn");
    let source = scene(
        &app,
        "b.bsn",
        ":\"a.bsn\"\nPosition { x: 1.0 }\nChildren [ #Y\nPosition { x: 9.0 } ]",
    );
    let root = app.world_mut().spawn_scene(source).unwrap().id();
    let text = write(app.world_mut(), root);
    assert!(text.contains(":\"a.bsn\""), "{text}");
    assert!(text.contains("x: 1.0"), "{text}");
    assert!(!text.contains("y: 2.0"), "the base's value is not repeated:\n{text}");
    assert!(!text.contains("Marker"), "an inherited component is not repeated:\n{text}");
    assert!(!text.contains("#X"), "the base's child belongs to the base:\n{text}");
    assert!(text.contains("#Y"), "{text}");

    // Writing spawned and removed a copy of the base; nothing of it is left.
    let world = app.world_mut();
    let positions = world.query::<&Position>().iter(world).count();
    let again = scene(&app, "c.bsn", &text);
    let copy = app.world_mut().spawn_scene(again).unwrap().id();
    let world = app.world_mut();
    assert_eq!(world.get::<Position>(copy), Some(&Position { x: 1.0, y: 2.0, z: 3.0 }));
    assert_eq!(world.get::<Children>(copy).map(|c| c.len()), Some(2));
    assert_eq!(world.query::<&Position>().iter(world).count(), positions + 2);
}

mod nested_refs {
    use bevy_ecs::{entity::Entity, hierarchy::Children, name::Name, prelude::Component, reflect::{ReflectComponent, ReflectFromTemplate}, template::FromTemplate};
    use bevy_reflect::Reflect;
    use bevy_scene::WorldSceneExt;

    use crate::{test_support::test_app, tests::{register_fixtures, scene}, write_scene_text, WriteSettings};

    #[derive(FromTemplate, Reflect, Clone, Debug, PartialEq)]
    #[template(reflect)]
    pub(crate) enum Source {
        #[default]
        Nothing,
        Node(Entity, String),
    }

    #[derive(FromTemplate, Reflect, Clone, Debug, PartialEq)]
    #[template(reflect)]
    pub(crate) struct Link(pub(crate) String, pub(crate) Source);

    #[derive(Component, FromTemplate, Reflect, Clone, Debug, PartialEq)]
    #[template(reflect)]
    #[reflect(Component, FromTemplate)]
    pub(crate) struct Links(#[template(built_in)] pub(crate) Vec<Link>);

    #[test]
    fn entity_references_nested_in_a_list_resolve_and_write_back() {
        let mut app = test_app();
        register_fixtures(&mut app);
        app.register_type::<Links>().register_type::<LinksTemplate>();
        let source = scene(
            &app,
            "a.bsn",
            "#Root\nChildren [\n    #A\n    --\n    #B\n    Links([Link(\"pose\", Node(#A, \"out\"))])\n]\n",
        );
        let root = app.world_mut().spawn_scene(source).unwrap().id();
        let world = app.world();
        let kids: Vec<Entity> = world.get::<Children>(root).unwrap().iter().copied().collect();
        let (a, b) = (kids[0], kids[1]);
        assert_eq!(world.get::<Name>(a).unwrap().as_str(), "A");
        assert_eq!(
            world.get::<Links>(b),
            Some(&Links(vec![Link("pose".into(), Source::Node(a, "out".into()))]))
        );
        let text = write_scene_text(app.world_mut(), root, &WriteSettings::default()).unwrap();
        assert!(text.contains("Node(#A"), "{text}");
    }
}

mod asset_values {
    use bevy_reflect::Reflect;

    use crate::{asset_value_from_document, test_support::test_app, tests::Position, write_asset_value};

    #[test]
    fn an_asset_value_writes_what_differs_and_reads_back() {
        let mut app = test_app();
        crate::tests::register_fixtures(&mut app);
        let value = Position { x: 0.0, y: 2.5, z: 0.0 };
        let document = write_asset_value(app.world(), "spot", &value).unwrap();
        let text = document.to_bsn_string();
        assert!(text.contains("#spot") && text.contains("y: 2.5") && !text.contains("x:"), "{text}");

        let parsed = bevy_bsn::BsnDocument::parse(&text).unwrap();
        let registry = app.world().resource::<bevy_ecs::reflect::AppTypeRegistry>().read();
        let registration = registry.get(core::any::TypeId::of::<Position>()).unwrap();
        let (name, read) = asset_value_from_document(&parsed, "spot.bsn", registration, &registry, None).unwrap();
        assert_eq!(name.as_deref(), Some("spot"));
        assert_eq!(read.downcast_ref::<Position>(), Some(&value));
        let _: &dyn Reflect = read.as_ref();
    }
}
