# `bevy_bsn`

Parser, abstract syntax tree, and printer for **BSN** (Bevy Scene Notation) — a compact,
human-readable, git-diffable text format for describing entities, their components and their
relationships in an ECS.

This crate is **engine-agnostic**: it depends on nothing but `core`, `alloc` and `thiserror`,
knows nothing about Bevy's `World`, reflection or asset systems, and works on `no_std`
targets. It reads `.bsn` text into a plain data AST and writes that AST back out as canonical
`.bsn` text, so external tooling — exporters, importers, editors, language servers — can
round-trip scene files without linking a game engine.

Bevy's own `bevy_bsn_asset` crate is the reference consumer: it resolves this AST against
`bevy_reflect`'s type registry to spawn entities. Nothing in this crate assumes that consumer;
the AST refers to types only by *type path string*, and what those strings mean is entirely up
to whoever resolves them.

## The format

```text
#Root
bevy_ui::ui_node::Node {
    width: bevy_ui::geometry::Val::Px(300.0),
}
bevy_ecs::hierarchy::Children [
    #Label
    bevy_ui::widget::text::Text("hello"),
    (
        :"widgets/button.bsn"
        my_game::Follower { target: #Root }
    )
]
```

An entity is a list of *entries*: an optional `#Name`, an optional `:"other.bsn"` base to
inherit from, any number of *patches* naming a type (`Transform { … }`, `Camera3d`,
`Shape::Rect(1.0, 2.0)`) and any number of *relations* naming a relationship target
(`Children [ … ]`). Patches are partial — fields you do not mention keep the value they had
from an earlier patch or from the type's default.

## Reading

```rust
use bevy_bsn::{BsnNodeKind, parse};

let document = parse(r#"
    #Player
    my_game::Health { max: 100 }
"#).expect("valid bsn");

let root = document.node(document.roots[0]).unwrap();
let BsnNodeKind::Entity { name, patches, .. } = &root.kind else {
    unreachable!()
};
assert_eq!(name.as_deref(), Some("Player"));
assert_eq!(patches.len(), 1);
```

Errors carry a byte span and render like a compiler diagnostic:

```rust
use bevy_bsn::parse;

let source = "A { width }";
let error = parse(source).unwrap_err();
let rendered = error.render(source, Some("assets/player.bsn"));
assert!(rendered.contains("--> assets/player.bsn:1:5"));
```

## Writing

Tools that *generate* scenes build the AST directly and print it; no parsing is involved.

```rust
use bevy_bsn::{BsnDocument, BsnNodeKind, BsnPatchPrefix, BsnPath, BsnValue, PatchBody};

let mut document = BsnDocument::new();

let max = document.push_value(BsnValue::Int(100));
let health = document.push_patch(
    BsnPatchPrefix::FromTemplate,
    BsnPath::from_segments(["my_game", "Health"]),
    PatchBody::Struct(vec![("max".into(), max)]),
);

let root = document.push_node(BsnNodeKind::Entity {
    name: Some("Player".into()),
    name_span: None,
    base: None,
    base_span: None,
    patches: vec![health],
    relations: vec![],
});
document.push_root(root);

assert_eq!(
    document.to_bsn_string(),
    "#Player\nmy_game::Health { max: 100 }\n",
);
```

Printing is semantics-preserving and idempotent — `print(parse(print(d))) == print(d)` — so
the printer doubles as a formatter. It does not preserve comments, blank lines, integer radix
or raw-string delimiters, because the AST does not carry them.

## `no_std`

The crate is `#![no_std]` unconditionally and needs `alloc`. The default `std` feature only
forwards to `thiserror/std`; nothing in the library references `std`.

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
