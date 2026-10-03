---
title: "`.bsn` scene asset files"
authors: []
pull_requests: []
---

Scenes can now be authored as `.bsn` asset files and loaded at runtime, using the same Bevy Scene
Notation the `bsn!` macro compiles statically:

```text
Node { width: px(300.0), flex_direction: FlexDirection::Column }
BackgroundColor(Color::Srgba(Srgba { red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0 }))
Children [
    #Title Text("Hello from a file"),
    #Logo ImageNode { image: "branding/icon.png" },
]
```

Loading works out of the box: with the default-on `bsn_asset` cargo feature, `DefaultPlugins`
registers a loader that turns any `.bsn` file into a `ScenePatch`, spawnable with
`ScenePatchInstance(asset_server.load("scenes/menu.bsn"))`. Files resolve through reflection at
load time — component names are looked up in the type registry, values are built reflectively,
and asset-path strings become handles with real load dependencies.

Static and dynamic scenes are two views of one system, not two systems. A `.bsn` file can inherit
another file (`:"base.bsn"`), a `bsn!` scene can inherit a `.bsn` file, and patches from both
sides merge into the same per-component template slots with last-writer-wins semantics — the
static/dynamic parity is pinned by a test matrix, and every public component the `bsn!` macro can
spawn is usable from a `.bsn` file (`#[template(reflect)]` on all `FromTemplate` components, with
a source-scan test keeping future components honest). `ScenePatchInstance` itself is reflectable,
so a `.bsn` file can declare nested scene instances.

`.bsn` files are user data, so failures are diagnostics, not panics: parse and resolution errors
are reported as asset load errors with `file:line:column` locations and a message naming the type
or field involved.

The grammar itself — lexer, parser, AST, and printer — lives in the standalone `bevy_bsn` crate,
which depends on nothing but `core`, `alloc` and `thiserror` and builds on `no_std`. Exporters,
editors, and language servers can read and write `.bsn` files without linking the engine. The
loader and reflection-driven resolution live in the new `bevy_bsn_asset` crate.

See the `dynamic_bsn` example for a complete walkthrough, including live editing with hot reload.
