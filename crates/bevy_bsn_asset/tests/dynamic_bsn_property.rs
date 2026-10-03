//! Property-based spawn-oracle tests for the dynamic `.bsn` pipeline.
//!
//! The hand-written parity matrix in `tests/dynamic_bsn.rs` enumerates one row per feature. This
//! file is its systematic complement: a typed model of a scene document is generated at random and
//! interpreted **twice**.
//!
//! 1. [`to_source`] renders the model as `.bsn` text, which goes through the real pipeline —
//!    `bevy_bsn::parse` → [`DynamicScene::from_document`] → `World::spawn_scene`.
//! 2. [`oracle`] computes the expected world state with plain Rust field assignments. It never
//!    touches reflection, so "last writer wins", "a partial patch leaves untouched fields alone"
//!    and the enum "match-or-reset" rule are each re-derived independently of the code under test.
//!
//! [`compare`] then walks the spawned tree from the root through `Children`, downcasting each
//! fixture component directly and comparing it with `PartialEq`. A `None == None` comparison is a
//! real assertion here: it is what catches a component the pipeline inserted but nothing asked for.
//!
//! Two properties share that comparison:
//!
//! * `spawn_matches_oracle` (P1) — spawning the parsed document matches the oracle.
//! * `print_reparse_spawn_matches_oracle` (P2) — printing the document and re-parsing it yields a
//!   `structural_eq` document that *also* spawns to the same state.
//!
//! Each case additionally asserts that despawning the scene root leaves the world's spawned-entity
//! count exactly where it started, so a leaked reference entity fails the run.
//!
//! # Scope (phase 1)
//!
//! In: in-range integers, exactly-representable floats (quarters), strings with escapes, `Option`
//! and `Vec` fields, nested structs, partial (leading-field) tuple structs, all three enum variant
//! kinds, unit markers, entity references through `#Name`, repeated patches of the same type on one
//! entity, children up to depth 3 across multiple relation blocks, and unique `#Name`s.
//!
//! Deliberately excluded, and left for a phase 2 that needs more machinery:
//!
//! * Handles and other asset-valued fields — spawning has to be gated on a load, so the synchronous
//!   `spawn_scene` path used here cannot see them (covered by `tests/dynamic_bsn.rs`).
//! * `:"base.bsn"` includes — a second document per case, plus load ordering.
//! * `~Template` and `@SceneComponent` patch prefixes.
//! * Registered type conversions and integer→float widening — the value coercion ladder, which
//!   deserves a generator of its own rather than being folded into the tree generator.

use core::cell::RefCell;
use core::fmt::Write as _;

use bevy_app::{App, TaskPoolPlugin};
use bevy_asset::{
    io::{
        memory::{Dir, MemoryAssetReader},
        AssetSourceBuilder, AssetSourceId,
    },
    AssetApp, AssetPlugin,
};
use bevy_bsn::{parse, print_document, BsnDocument};
use bevy_bsn_asset::{BsnAssetPlugin, DynamicScene};
use bevy_ecs::{
    entity::Entity,
    hierarchy::{ChildOf, Children},
    name::Name,
    prelude::Component,
    reflect::{AppTypeRegistry, ReflectComponent, ReflectFromTemplate},
    template::FromTemplate,
    world::World,
};
use bevy_platform::collections::HashMap;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_scene::{ScenePlugin, WorldSceneExt};
use proptest::{
    prelude::*,
    test_runner::{Config, RngAlgorithm, TestCaseError, TestCaseResult, TestRng, TestRunner},
};

// ===============================================================================================
// Fixture vocabulary
//
// Self-contained on purpose: `tests/dynamic_bsn.rs` owns its own fixtures, and a shared set would
// couple two suites that want to grow in different directions.
// ===============================================================================================

/// A three-`f32` struct: the canonical partial-patch subject.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

/// A struct with two `Option` fields and one nested tuple-struct field.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct Opts {
    a: Option<u32>,
    b: Option<f32>,
    inner: Inner,
}

/// A two-field tuple struct, used both as a component and as [`Opts`]'s nested field.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct Inner(u32, u32);

/// A two-field tuple struct of mixed types, for partial *leading-field* tuple patches.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct Pair(f32, u32);

/// A unit marker component: the "ensure the slot exists, change nothing" case.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct Marker;

/// An enum component with one variant of each kind.
///
/// `#[template(reflect)]` generates the reflectable `ChoiceTemplate` that the dynamic path actually
/// patches, so both types have to be registered.
#[derive(Component, FromTemplate, Reflect, PartialEq, Debug)]
#[template(reflect)]
#[reflect(Component, FromTemplate)]
enum Choice {
    /// A struct variant, and the template's default.
    #[default]
    Alpha {
        /// First field.
        x: u32,
        /// Second field.
        y: u32,
    },
    /// A tuple variant.
    Beta(u32),
    /// A unit variant.
    Gamma,
}

/// The value `ChoiceTemplate::default()` builds into: the `#[default]` variant, fields defaulted.
fn choice_default() -> Choice {
    Choice::Alpha { x: 0, y: 0 }
}

/// A component with an `Option` field and a `Vec` field.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct Bag {
    maybe: Option<u32>,
    list: Vec<u32>,
}

/// A component with a `String` field, for escaped string literals.
#[derive(Component, Reflect, Clone, Default, PartialEq, Debug)]
#[reflect(Component, Default)]
struct Label {
    text: String,
}

/// A component holding an entity reference, so `#Name` values have somewhere to go.
///
/// The `FromTemplate` derive gives it a `LinkTemplate(EntityTemplate)`, which is the type a `#Name`
/// value is actually built for.
#[derive(Component, FromTemplate, Reflect, PartialEq, Debug)]
#[template(reflect)]
#[reflect(Component, FromTemplate)]
struct Link(Entity);

/// Every short type path the generated source uses, for the unambiguity guard.
const FIXTURE_SHORT_PATHS: &[&str] = &[
    "Position", "Opts", "Inner", "Pair", "Marker", "Choice", "Bag", "Label", "Link", "Children",
];

/// Builds the [`App`] every case is spawned into.
///
/// One app is built per property and reused across all 512 cases (and every shrink step): building
/// it is by far the most expensive part of a case, and each case despawns everything it spawned.
fn test_app() -> App {
    let mut app = App::new();
    // An empty in-memory source: nothing is ever loaded, but `AssetPlugin` still needs a reader,
    // and an in-memory one keeps the test off the file system.
    app.register_asset_source(
        AssetSourceId::Default,
        AssetSourceBuilder::new(|| {
            Box::new(MemoryAssetReader {
                root: Dir::default(),
            })
        }),
    );
    app.add_plugins((
        TaskPoolPlugin::default(),
        AssetPlugin::default(),
        ScenePlugin,
        BsnAssetPlugin,
    ));
    app.register_type::<Position>();
    app.register_type::<Opts>();
    app.register_type::<Inner>();
    app.register_type::<Pair>();
    app.register_type::<Marker>();
    app.register_type::<Choice>();
    app.register_type::<ChoiceTemplate>();
    app.register_type::<Bag>();
    app.register_type::<Label>();
    app.register_type::<Link>();
    app.register_type::<LinkTemplate>();
    app.register_type::<Name>();
    app.register_type::<Children>();
    app.register_type::<ChildOf>();
    app.finish();
    app.cleanup();
    app
}

/// Every fixture must be reachable by its *short* path, which is how the generated source names it.
///
/// A collision with a type some plugin registers would otherwise turn every case into the same
/// `UnknownType` build error, which is a much less obvious failure than this assertion.
#[test]
fn fixture_short_paths_are_unambiguous() {
    let app = test_app();
    let registry = app.world().resource::<AppTypeRegistry>().read();
    for short_path in FIXTURE_SHORT_PATHS {
        assert!(
            registry.get_with_short_type_path(short_path).is_some(),
            "`{short_path}` does not resolve by short path; another registered type may share it"
        );
    }
}

// ===============================================================================================
// The generated model
// ===============================================================================================

/// A whole generated document: one root entity plus every `#Name` it declares, in pre-order.
#[derive(Clone, Debug)]
struct GenDoc {
    root: GenEntity,
    /// Names are assigned by [`finalize`], so they are unique and always non-empty.
    names: Vec<String>,
}

/// One entity of the model.
#[derive(Clone, Debug)]
struct GenEntity {
    /// Filled in by [`finalize`]; empty while the strategy is still building the tree.
    name: String,
    patches: Vec<GenPatch>,
    /// One `Vec` per `Children [ … ]` block. Two blocks on one entity must behave like one longer
    /// block, which is why this is not flattened.
    relations: Vec<Vec<GenEntity>>,
}

/// A value written into an `Option`-typed field.
#[derive(Clone, Copy, Debug)]
enum OptU32 {
    /// `5` — the payload written bare, relying on the implicit `Some`.
    Implicit(u32),
    /// `Some(5)`.
    Explicit(u32),
    /// `None`.
    Null,
}

impl OptU32 {
    fn value(self) -> Option<u32> {
        match self {
            OptU32::Implicit(value) | OptU32::Explicit(value) => Some(value),
            OptU32::Null => None,
        }
    }

    fn render(self) -> String {
        match self {
            OptU32::Implicit(value) => value.to_string(),
            OptU32::Explicit(value) => format!("Some({value})"),
            OptU32::Null => "None".to_string(),
        }
    }
}

/// [`OptU32`] for an `Option<f32>` field.
#[derive(Clone, Copy, Debug)]
enum OptF32 {
    Implicit(f32),
    Explicit(f32),
    Null,
}

impl OptF32 {
    fn value(self) -> Option<f32> {
        match self {
            OptF32::Implicit(value) | OptF32::Explicit(value) => Some(value),
            OptF32::Null => None,
        }
    }

    fn render(self) -> String {
        match self {
            OptF32::Implicit(value) => render_f32(value),
            OptF32::Explicit(value) => format!("Some({})", render_f32(value)),
            OptF32::Null => "None".to_string(),
        }
    }
}

/// A value for an [`Inner`]-typed destination: bare, or one/two leading fields.
#[derive(Clone, Copy, Debug)]
enum InnerVal {
    Bare,
    One(u32),
    Two(u32, u32),
}

impl InnerVal {
    fn render(self) -> String {
        match self {
            InnerVal::Bare => "Inner".to_string(),
            InnerVal::One(a) => format!("Inner({a})"),
            InnerVal::Two(a, b) => format!("Inner({a}, {b})"),
        }
    }
}

/// A value for a [`Pair`]-typed destination.
#[derive(Clone, Copy, Debug)]
enum PairVal {
    Bare,
    One(f32),
    Two(f32, u32),
}

/// A patch of the [`Choice`] enum: a variant plus the fields supplied for it.
#[derive(Clone, Copy, Debug)]
enum ChoiceVal {
    /// `Choice::Alpha`, `Choice::Alpha { x: 1 }`, …
    Alpha { x: Option<u32>, y: Option<u32> },
    /// `Choice::Beta`, `Choice::Beta(1)`.
    Beta(Option<u32>),
    /// `Choice::Gamma`.
    Gamma,
}

/// One `Type { … }` / `Type(…)` / `Type` entry on an entity.
///
/// A patch whose fields are all `None` renders as the bare path form, which is the pipeline's
/// "ensure the slot exists, change nothing" case.
#[derive(Clone, Debug)]
enum GenPatch {
    Position {
        x: Option<f32>,
        y: Option<f32>,
        z: Option<f32>,
    },
    Opts {
        a: Option<OptU32>,
        b: Option<OptF32>,
        inner: Option<InnerVal>,
    },
    Inner(InnerVal),
    Pair(PairVal),
    Marker,
    Choice(ChoiceVal),
    Bag {
        maybe: Option<OptU32>,
        list: Option<Vec<u32>>,
    },
    Label(Option<String>),
    /// `Link(#Name)`. The payload is a raw index, mapped onto the document's names by
    /// [`GenDoc::link_target`], so shrinking the tree can never dangle a reference.
    Link(u32),
}

impl GenDoc {
    /// The name a `Link(raw)` patch points at. Always a name this document declares.
    fn link_target(&self, raw: u32) -> &str {
        &self.names[raw as usize % self.names.len()]
    }
}

// ===============================================================================================
// Strategies
// ===============================================================================================

/// Strings that survive a print/parse round trip, including every escape the printer emits.
const STRINGS: &[&str] = &[
    "",
    "a",
    "hello world",
    "quote\"inside",
    "back\\slash",
    "line\nbreak",
    "tab\tsep",
    "nul\0byte",
    "snowman ☃",
];

/// Floats that are exactly representable in `f32` *and* in the `f64` the parser produces: small
/// integers and quarters. Nothing here rounds, so a mismatch is always a real bug.
fn f32_value() -> impl Strategy<Value = f32> {
    (-32i32..=32i32).prop_map(|quarters| quarters as f32 / 4.0)
}

fn u32_value() -> impl Strategy<Value = u32> {
    0u32..=1000
}

fn string_value() -> impl Strategy<Value = String> {
    prop::sample::select(STRINGS.to_vec()).prop_map(str::to_string)
}

fn opt_u32() -> impl Strategy<Value = OptU32> {
    prop_oneof![
        u32_value().prop_map(OptU32::Implicit),
        u32_value().prop_map(OptU32::Explicit),
        Just(OptU32::Null),
    ]
}

fn opt_f32() -> impl Strategy<Value = OptF32> {
    prop_oneof![
        f32_value().prop_map(OptF32::Implicit),
        f32_value().prop_map(OptF32::Explicit),
        Just(OptF32::Null),
    ]
}

fn inner_value() -> impl Strategy<Value = InnerVal> {
    prop_oneof![
        Just(InnerVal::Bare),
        u32_value().prop_map(InnerVal::One),
        (u32_value(), u32_value()).prop_map(|(a, b)| InnerVal::Two(a, b)),
    ]
}

fn pair_value() -> impl Strategy<Value = PairVal> {
    prop_oneof![
        Just(PairVal::Bare),
        f32_value().prop_map(PairVal::One),
        (f32_value(), u32_value()).prop_map(|(a, b)| PairVal::Two(a, b)),
    ]
}

fn choice_value() -> impl Strategy<Value = ChoiceVal> {
    prop_oneof![
        (prop::option::of(u32_value()), prop::option::of(u32_value()))
            .prop_map(|(x, y)| ChoiceVal::Alpha { x, y }),
        prop::option::of(u32_value()).prop_map(ChoiceVal::Beta),
        Just(ChoiceVal::Gamma),
    ]
}

fn patch() -> impl Strategy<Value = GenPatch> {
    prop_oneof![
        (
            prop::option::of(f32_value()),
            prop::option::of(f32_value()),
            prop::option::of(f32_value())
        )
            .prop_map(|(x, y, z)| GenPatch::Position { x, y, z }),
        (
            prop::option::of(opt_u32()),
            prop::option::of(opt_f32()),
            prop::option::of(inner_value())
        )
            .prop_map(|(a, b, inner)| GenPatch::Opts { a, b, inner }),
        inner_value().prop_map(GenPatch::Inner),
        pair_value().prop_map(GenPatch::Pair),
        Just(GenPatch::Marker),
        choice_value().prop_map(GenPatch::Choice),
        (
            prop::option::of(opt_u32()),
            prop::option::of(prop::collection::vec(u32_value(), 0..=3))
        )
            .prop_map(|(maybe, list)| GenPatch::Bag { maybe, list }),
        prop::option::of(string_value()).prop_map(GenPatch::Label),
        any::<u32>().prop_map(GenPatch::Link),
    ]
}

/// The tree generator: depth ≤ 3, breadth ≤ 3 per relation block, ≤ 2 relation blocks per entity,
/// and a node budget of ~12 entities per document.
fn entity() -> impl Strategy<Value = GenEntity> {
    let leaf = prop::collection::vec(patch(), 0..=4).prop_map(|patches| GenEntity {
        name: String::new(),
        patches,
        relations: Vec::new(),
    });
    leaf.prop_recursive(3, 12, 3, |inner| {
        (
            prop::collection::vec(patch(), 0..=3),
            prop::collection::vec(prop::collection::vec(inner, 1..=3), 0..=2),
        )
            .prop_map(|(patches, relations)| GenEntity {
                name: String::new(),
                patches,
                relations,
            })
    })
}

fn document() -> impl Strategy<Value = GenDoc> {
    entity().prop_map(finalize)
}

/// Assigns a unique `#Name` to every entity, in pre-order, and collects them.
fn finalize(mut root: GenEntity) -> GenDoc {
    let mut names = Vec::new();
    assign_names(&mut root, &mut names);
    GenDoc { root, names }
}

fn assign_names(entity: &mut GenEntity, names: &mut Vec<String>) {
    let name = format!("E{}", names.len());
    names.push(name.clone());
    entity.name = name;
    for block in &mut entity.relations {
        for child in block {
            assign_names(child, names);
        }
    }
}

// ===============================================================================================
// Rendering the model as `.bsn` text
// ===============================================================================================

/// Formats a float so that it re-parses exactly: `{:?}` always emits a decimal point.
fn render_f32(value: f32) -> String {
    format!("{value:?}")
}

/// Escapes a string as a `.bsn` string literal, matching the printer's escape set.
fn render_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0' => out.push_str("\\0"),
            ch if ch.is_control() => {
                let _ = write!(out, "\\u{{{:x}}}", ch as u32);
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

/// Renders `Type { a: …, b: … }`, or the bare `Type` when no field was supplied.
fn render_struct(type_path: &str, fields: &[(&str, String)]) -> String {
    if fields.is_empty() {
        return type_path.to_string();
    }
    let body = fields
        .iter()
        .map(|(name, value)| format!("{name}: {value}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{type_path} {{ {body} }}")
}

fn render_patch(doc: &GenDoc, patch: &GenPatch) -> String {
    match patch {
        GenPatch::Position { x, y, z } => {
            let mut fields = Vec::new();
            if let Some(x) = x {
                fields.push(("x", render_f32(*x)));
            }
            if let Some(y) = y {
                fields.push(("y", render_f32(*y)));
            }
            if let Some(z) = z {
                fields.push(("z", render_f32(*z)));
            }
            render_struct("Position", &fields)
        }
        GenPatch::Opts { a, b, inner } => {
            let mut fields = Vec::new();
            if let Some(a) = a {
                fields.push(("a", a.render()));
            }
            if let Some(b) = b {
                fields.push(("b", b.render()));
            }
            if let Some(inner) = inner {
                fields.push(("inner", inner.render()));
            }
            render_struct("Opts", &fields)
        }
        GenPatch::Inner(value) => value.render(),
        GenPatch::Pair(value) => match value {
            PairVal::Bare => "Pair".to_string(),
            PairVal::One(a) => format!("Pair({})", render_f32(*a)),
            PairVal::Two(a, b) => format!("Pair({}, {b})", render_f32(*a)),
        },
        GenPatch::Marker => "Marker".to_string(),
        GenPatch::Choice(value) => match value {
            ChoiceVal::Alpha { x, y } => {
                let mut fields = Vec::new();
                if let Some(x) = x {
                    fields.push(("x", x.to_string()));
                }
                if let Some(y) = y {
                    fields.push(("y", y.to_string()));
                }
                render_struct("Choice::Alpha", &fields)
            }
            ChoiceVal::Beta(Some(value)) => format!("Choice::Beta({value})"),
            ChoiceVal::Beta(None) => "Choice::Beta".to_string(),
            ChoiceVal::Gamma => "Choice::Gamma".to_string(),
        },
        GenPatch::Bag { maybe, list } => {
            let mut fields = Vec::new();
            if let Some(maybe) = maybe {
                fields.push(("maybe", maybe.render()));
            }
            if let Some(list) = list {
                let items = list
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                fields.push(("list", format!("[{items}]")));
            }
            render_struct("Bag", &fields)
        }
        GenPatch::Label(text) => {
            let fields = match text {
                Some(text) => vec![("text", render_string(text))],
                None => Vec::new(),
            };
            render_struct("Label", &fields)
        }
        GenPatch::Link(raw) => format!("Link(#{})", doc.link_target(*raw)),
    }
}

/// Renders the whole document as `.bsn` text.
fn to_source(doc: &GenDoc) -> String {
    let mut out = String::new();
    render_entity(doc, &doc.root, 0, &mut out);
    out
}

fn render_entity(doc: &GenDoc, entity: &GenEntity, level: usize, out: &mut String) {
    let pad = "    ".repeat(level);
    let _ = writeln!(out, "{pad}#{}", entity.name);
    for patch in &entity.patches {
        let _ = writeln!(out, "{pad}{}", render_patch(doc, patch));
    }
    for block in &entity.relations {
        let _ = writeln!(out, "{pad}Children [");
        for (index, child) in block.iter().enumerate() {
            let _ = writeln!(out, "{pad}    (");
            render_entity(doc, child, level + 2, out);
            let separator = if index + 1 < block.len() { "," } else { "" };
            let _ = writeln!(out, "{pad}    ){separator}");
        }
        let _ = writeln!(out, "{pad}]");
    }
}

// ===============================================================================================
// The oracle: what the spawned world must look like, computed with plain Rust
// ===============================================================================================

/// The expected state of one spawned entity.
#[derive(Debug)]
struct Expected {
    name: String,
    position: Option<Position>,
    opts: Option<Opts>,
    inner: Option<Inner>,
    pair: Option<Pair>,
    marker: Option<Marker>,
    choice: Option<Choice>,
    bag: Option<Bag>,
    label: Option<Label>,
    /// The `#Name` this entity's `Link` points at, if it has one.
    link: Option<String>,
    children: Vec<Expected>,
}

impl Expected {
    fn new(name: String) -> Self {
        Expected {
            name,
            position: None,
            opts: None,
            inner: None,
            pair: None,
            marker: None,
            choice: None,
            bag: None,
            label: None,
            link: None,
            children: Vec::new(),
        }
    }
}

fn oracle(doc: &GenDoc) -> Expected {
    oracle_entity(doc, &doc.root)
}

fn oracle_entity(doc: &GenDoc, entity: &GenEntity) -> Expected {
    let mut expected = Expected::new(entity.name.clone());
    for patch in &entity.patches {
        apply_patch(doc, patch, &mut expected);
    }
    // Multiple relation blocks land in one `Children` collection, in document order.
    for block in &entity.relations {
        for child in block {
            expected.children.push(oracle_entity(doc, child));
        }
    }
    expected
}

/// Applies one patch to the expected state, with plain Rust assignment semantics.
fn apply_patch(doc: &GenDoc, patch: &GenPatch, expected: &mut Expected) {
    match patch {
        GenPatch::Position { x, y, z } => {
            let target = expected.position.get_or_insert_default();
            if let Some(x) = x {
                target.x = *x;
            }
            if let Some(y) = y {
                target.y = *y;
            }
            if let Some(z) = z {
                target.z = *z;
            }
        }
        GenPatch::Opts { a, b, inner } => {
            let target = expected.opts.get_or_insert_default();
            if let Some(a) = a {
                target.a = a.value();
            }
            if let Some(b) = b {
                target.b = b.value();
            }
            if let Some(inner) = inner {
                apply_inner_field(*inner, &mut target.inner);
            }
        }
        // A *top-level* bare `Inner` only ensures the slot exists; the same text as a nested field
        // value resets that field (see `apply_inner_field`). The asymmetry is the difference
        // between `DynamicPatchValue::Ensure` and a fully-constructed field value.
        GenPatch::Inner(value) => {
            let target = expected.inner.get_or_insert_default();
            match value {
                InnerVal::Bare => {}
                InnerVal::One(a) => target.0 = *a,
                InnerVal::Two(a, b) => {
                    target.0 = *a;
                    target.1 = *b;
                }
            }
        }
        GenPatch::Pair(value) => {
            let target = expected.pair.get_or_insert_default();
            match value {
                PairVal::Bare => {}
                PairVal::One(a) => target.0 = *a,
                PairVal::Two(a, b) => {
                    target.0 = *a;
                    target.1 = *b;
                }
            }
        }
        GenPatch::Marker => expected.marker = Some(Marker),
        GenPatch::Choice(value) => {
            let target = expected.choice.get_or_insert_with(choice_default);
            apply_choice(*value, target);
        }
        GenPatch::Bag { maybe, list } => {
            let target = expected.bag.get_or_insert_default();
            if let Some(maybe) = maybe {
                target.maybe = maybe.value();
            }
            if let Some(list) = list {
                apply_list(list, &mut target.list);
            }
        }
        GenPatch::Label(text) => {
            let target = expected.label.get_or_insert_default();
            if let Some(text) = text {
                target.text.clone_from(text);
            }
        }
        GenPatch::Link(raw) => expected.link = Some(doc.link_target(*raw).to_string()),
    }
}

/// A nested `Inner`-typed field value. Unlike a top-level patch, the bare form is a *complete*
/// value, so it resets the whole field; the positional forms patch only their leading elements.
fn apply_inner_field(value: InnerVal, target: &mut Inner) {
    match value {
        InnerVal::Bare => *target = Inner::default(),
        InnerVal::One(a) => target.0 = a,
        InnerVal::Two(a, b) => {
            target.0 = a;
            target.1 = b;
        }
    }
}

/// The enum "match-or-reset" rule: patching the variant that is already there keeps untouched
/// fields, and switching variants starts from that variant's defaults.
fn apply_choice(value: ChoiceVal, target: &mut Choice) {
    match value {
        ChoiceVal::Alpha { x, y } => {
            if !matches!(target, Choice::Alpha { .. }) {
                *target = Choice::Alpha { x: 0, y: 0 };
            }
            if let Choice::Alpha {
                x: target_x,
                y: target_y,
            } = target
            {
                if let Some(x) = x {
                    *target_x = x;
                }
                if let Some(y) = y {
                    *target_y = y;
                }
            }
        }
        ChoiceVal::Beta(value) => {
            if !matches!(target, Choice::Beta(_)) {
                *target = Choice::Beta(0);
            }
            if let (Choice::Beta(target_value), Some(value)) = (&mut *target, value) {
                *target_value = value;
            }
        }
        ChoiceVal::Gamma => *target = Choice::Gamma,
    }
}

/// Reflect's list semantics: elements are patched positionally and the source's tail is appended,
/// so a shorter list never truncates the destination.
fn apply_list(source: &[u32], target: &mut Vec<u32>) {
    for (index, value) in source.iter().enumerate() {
        match target.get_mut(index) {
            Some(slot) => *slot = *value,
            None => target.push(*value),
        }
    }
}

// ===============================================================================================
// Comparing the spawned world against the oracle
// ===============================================================================================

/// Fails a case with the offending source text attached, which is what makes a shrunk
/// counterexample readable.
fn fail(message: impl Into<String>, source: &str) -> TestCaseError {
    TestCaseError::fail(format!("{}\n--- source ---\n{source}", message.into()))
}

fn compare(world: &World, root: Entity, expected: &Expected, source: &str) -> TestCaseResult {
    let mut names = HashMap::<String, Entity>::default();
    collect_names(world, root, &mut names);
    compare_entity(world, root, expected, &names, source)
}

/// Maps every spawned entity's `Name` to its id, for `Link` comparisons.
fn collect_names(world: &World, entity: Entity, names: &mut HashMap<String, Entity>) {
    if let Some(name) = world.get::<Name>(entity) {
        names.insert(name.as_str().to_string(), entity);
    }
    if let Some(children) = world.get::<Children>(entity) {
        for child in children.iter() {
            collect_names(world, *child, names);
        }
    }
}

fn compare_entity(
    world: &World,
    entity: Entity,
    expected: &Expected,
    names: &HashMap<String, Entity>,
    source: &str,
) -> TestCaseResult {
    let name = world.get::<Name>(entity).map(Name::as_str);
    if name != Some(expected.name.as_str()) {
        return Err(fail(
            format!("`#{}`: expected that name, found {name:?}", expected.name),
            source,
        ));
    }

    // Each of these compares `Option<&T>` with `Option<&T>`: a `None == None` is the assertion that
    // the pipeline inserted nothing the document never asked for.
    macro_rules! component {
        ($ty:ty, $field:expr) => {
            let actual = world.get::<$ty>(entity);
            let expected_value = $field.as_ref();
            if actual != expected_value {
                return Err(fail(
                    format!(
                        "`#{}`: {} is {actual:?}, expected {expected_value:?}",
                        expected.name,
                        stringify!($ty),
                    ),
                    source,
                ));
            }
        };
    }
    component!(Position, expected.position);
    component!(Opts, expected.opts);
    component!(Inner, expected.inner);
    component!(Pair, expected.pair);
    component!(Marker, expected.marker);
    component!(Choice, expected.choice);
    component!(Bag, expected.bag);
    component!(Label, expected.label);

    let actual_link = world.get::<Link>(entity).map(|link| link.0);
    let expected_link = match &expected.link {
        Some(target) => Some(*names.get(target).ok_or_else(|| {
            fail(
                format!(
                    "`#{}`: no spawned entity is named `#{target}`",
                    expected.name
                ),
                source,
            )
        })?),
        None => None,
    };
    if actual_link != expected_link {
        return Err(fail(
            format!(
                "`#{}`: Link is {actual_link:?}, expected {expected_link:?}",
                expected.name
            ),
            source,
        ));
    }

    let children: Vec<Entity> = world
        .get::<Children>(entity)
        .map(|children| children.iter().copied().collect())
        .unwrap_or_default();
    if children.len() != expected.children.len() {
        return Err(fail(
            format!(
                "`#{}`: {} children, expected {}",
                expected.name,
                children.len(),
                expected.children.len()
            ),
            source,
        ));
    }
    for (child, expected_child) in children.iter().zip(&expected.children) {
        compare_entity(world, *child, expected_child, names, source)?;
    }

    Ok(())
}

// ===============================================================================================
// The properties
// ===============================================================================================

/// Parses `source`, or fails the case with the parse error rendered against it.
fn parse_source(source: &str) -> Result<BsnDocument, TestCaseError> {
    parse(source).map_err(|error| {
        fail(
            format!("parse failed: {}", error.render(source, None)),
            source,
        )
    })
}

/// Builds `document`, spawns it, compares it with `expected`, then despawns it and checks that the
/// world is back where it started.
///
/// Despawning happens even when the comparison fails, because the same `App` is reused by every
/// later case and every shrink step.
fn spawn_and_compare(
    app: &mut App,
    document: &BsnDocument,
    expected: &Expected,
    source: &str,
) -> TestCaseResult {
    let registry = app.world().resource::<AppTypeRegistry>().clone();
    let scene = DynamicScene::from_document(document, "property.bsn", &registry)
        .map_err(|error| fail(format!("building the scene failed: {error}"), source))?;

    let world = app.world_mut();
    let before = world.entities().count_spawned();
    let root = world
        .spawn_scene(scene)
        .map_err(|error| fail(format!("spawning failed: {error}"), source))?
        .id();

    let result = compare(world, root, expected, source);

    // `Children` is a linked-spawn relationship, so this takes the whole tree with it.
    world.despawn(root);
    let after = world.entities().count_spawned();

    result?;
    if before != after {
        return Err(fail(
            format!(
                "despawning the scene root did not restore the entity count \
                 (before: {before}, after: {after})"
            ),
            source,
        ));
    }
    Ok(())
}

/// 512 cases, a fixed seed and no on-disk failure persistence, so a run is reproducible in CI.
fn config() -> Config {
    Config {
        cases: 512,
        max_shrink_iters: 512,
        failure_persistence: None,
        ..Config::default()
    }
}

/// Runs `check` over 512 generated documents against one shared [`App`].
fn run_property(check: impl Fn(&mut App, &GenDoc) -> TestCaseResult) {
    let app = RefCell::new(test_app());
    let mut runner =
        TestRunner::new_with_rng(config(), TestRng::deterministic_rng(RngAlgorithm::ChaCha));
    let result = runner.run(&document(), |doc| {
        let mut app = app.borrow_mut();
        check(&mut app, &doc)
    });
    if let Err(error) = result {
        panic!("{error}");
    }
}

/// P1: parsing and spawning a generated document reproduces the oracle exactly.
#[test]
fn spawn_matches_oracle() {
    run_property(|app, doc| {
        let source = to_source(doc);
        let document = parse_source(&source)?;
        spawn_and_compare(app, &document, &oracle(doc), &source)
    });
}

/// P2: printing a parsed document and re-parsing it yields a structurally identical document that
/// spawns to the same state — the printer, the parser and the lowering all agree.
#[test]
fn print_reparse_spawn_matches_oracle() {
    run_property(|app, doc| {
        let source = to_source(doc);
        let first = parse_source(&source)?;
        let printed = print_document(&first);
        let second = parse_source(&printed)?;
        if !first.structural_eq(&second) {
            return Err(fail(
                format!("printing and re-parsing changed the document\n--- printed ---\n{printed}"),
                &source,
            ));
        }
        spawn_and_compare(app, &second, &oracle(doc), &printed)
    });
}
