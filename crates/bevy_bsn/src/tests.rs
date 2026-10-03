//! The crate's test suite: the worked corpus, table-driven lexer tests, printer round-trip
//! tests, the rejected-construct diagnostics, and the hygiene tests that keep the crate
//! engine-independent and deterministic.

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use crate::ast::MAX_WALK_DEPTH;
use crate::lexer::{decode_float, decode_int, decode_string};
use crate::{
    parse, print_document, unsupported, write_document_with, BsnDocument, BsnNodeId, BsnNodeKind,
    BsnParseError, BsnParseErrorKind, BsnPatchPrefix, BsnPath, BsnValue, BsnValueId, LexError,
    Lexer, PatchBody, PrintOptions, Span, TokenKind, MAX_NESTING_DEPTH,
};

// ---------------------------------------------------------------------------------------
// §12 worked corpus
// ---------------------------------------------------------------------------------------

/// §12.1 Minimal entity.
const CORPUS_1: &str = "bevy_transform::components::transform::Transform\n";

const TREE_1: &str = "\
Entity#0 name=- base=-
  Patch#1 patch `bevy_transform::components::transform::Transform` value=$0
values:
$0 Path(bevy_transform::components::transform::Transform)
";

/// §12.2 Struct patch with nested struct value, negatives, partial fields.
const CORPUS_2: &str = r#"#Camera
bevy_camera::components::Camera3d
bevy_transform::components::transform::Transform {
    translation: glam::Vec3 { x: 0.0, y: 6.0, z: -12.5 },
    scale: glam::Vec3 { x: 1.0 },
}
"#;

const TREE_2: &str = "\
Entity#0 name=\"Camera\" base=-
  Patch#1 patch `bevy_camera::components::Camera3d` value=$0
  Patch#2 patch `bevy_transform::components::transform::Transform` value=$1
values:
$0 Path(bevy_camera::components::Camera3d)
$1 Struct(bevy_transform::components::transform::Transform)
  field translation = $2
  field scale = $6
$2 Struct(glam::Vec3)
  field x = $3
  field y = $4
  field z = $5
$3 Float(0.0)
$4 Float(6.0)
$5 Float(-12.5)
$6 Struct(glam::Vec3)
  field x = $7
$7 Float(1.0)
";

/// §12.3 Nested children, relations, entity references.
const CORPUS_3: &str = r#"#Root
bevy_ui::ui_node::Node
bevy_ecs::hierarchy::Children [
    #Label
    bevy_ui::widget::text::Text("hello"),
    my_game::Follower { target: #Root }
]
"#;

const TREE_3: &str = "\
Entity#0 name=\"Root\" base=-
  Patch#1 patch `bevy_ui::ui_node::Node` value=$0
  Relation#2 `bevy_ecs::hierarchy::Children`
    Entity#3 name=\"Label\" base=-
      Patch#4 patch `bevy_ui::widget::text::Text` value=$1
    Entity#5 name=- base=-
      Patch#6 patch `my_game::Follower` value=$3
values:
$0 Path(bevy_ui::ui_node::Node)
$1 NamedTuple(bevy_ui::widget::text::Text)
  $2
$2 Str(\"hello\")
$3 Struct(my_game::Follower)
  field target = $4
$4 EntityRef(Root)
";

/// §12.4 Inheritance (`:base`) plus overriding patches.
const CORPUS_4: &str = r#":"enemies/orc.bsn"
my_game::Health { max: 200 }
bevy_ecs::hierarchy::Children [
    my_game::Weapon
]
"#;

const TREE_4: &str = "\
Entity#0 name=- base=\"enemies/orc.bsn\"
  Patch#1 patch `my_game::Health` value=$0
  Relation#2 `bevy_ecs::hierarchy::Children`
    Entity#3 name=- base=-
      Patch#4 patch `my_game::Weapon` value=$2
values:
$0 Struct(my_game::Health)
  field max = $1
$1 Int(200)
$2 Path(my_game::Weapon)
";

/// §12.5 Enum variants — all three kinds.
const CORPUS_5: &str = r#"bevy_camera::visibility::Visibility::Visible
my_game::Shape::Circle { radius: 2.5 }
my_game::Shape::Rect(1.0, 2.0)
"#;

const TREE_5: &str = "\
Entity#0 name=- base=-
  Patch#1 patch `bevy_camera::visibility::Visibility::Visible` value=$0
  Patch#2 patch `my_game::Shape::Circle` value=$1
  Patch#3 patch `my_game::Shape::Rect` value=$3
values:
$0 Path(bevy_camera::visibility::Visibility::Visible)
$1 Struct(my_game::Shape::Circle)
  field radius = $2
$2 Float(2.5)
$3 NamedTuple(my_game::Shape::Rect)
  $4
  $5
$4 Float(1.0)
$5 Float(2.0)
";

/// §12.6 Partial tuple patch, list value, string→handle, unit and tuple values.
const CORPUS_6: &str = r#"my_game::Rgba(1.0, 0.5)
bevy_render::mesh::components::Mesh3d("models/tree.gltf#Mesh0/Primitive0")
my_game::Waypoints([1.0, 2.0, 3.0])
my_game::Marker(())
my_game::Pair((1, 2))
"#;

const TREE_6: &str = "\
Entity#0 name=- base=-
  Patch#1 patch `my_game::Rgba` value=$0
  Patch#2 patch `bevy_render::mesh::components::Mesh3d` value=$3
  Patch#3 patch `my_game::Waypoints` value=$5
  Patch#4 patch `my_game::Marker` value=$10
  Patch#5 patch `my_game::Pair` value=$12
values:
$0 NamedTuple(my_game::Rgba)
  $1
  $2
$1 Float(1.0)
$2 Float(0.5)
$3 NamedTuple(bevy_render::mesh::components::Mesh3d)
  $4
$4 Str(\"models/tree.gltf#Mesh0/Primitive0\")
$5 NamedTuple(my_game::Waypoints)
  $6
$6 List
  $7
  $8
  $9
$7 Float(1.0)
$8 Float(2.0)
$9 Float(3.0)
$10 NamedTuple(my_game::Marker)
  $11
$11 Unit
$12 NamedTuple(my_game::Pair)
  $13
$13 Tuple
  $14
  $15
$14 Int(1)
$15 Int(2)
";

/// §12.7 Template (`~`) and scene-component (`@`) patches.
const CORPUS_7: &str = r#"~bevy_asset::handle::HandleTemplate<bevy_image::image::Image>("icon.png")
@my_game::HealthBar { width: 100 }
"#;

const TREE_7: &str = "\
Entity#0 name=- base=-
  Patch#1 template `bevy_asset::handle::HandleTemplate<bevy_image::image::Image>` value=$0
  Patch#2 scene `my_game::HealthBar` value=$2
values:
$0 NamedTuple(bevy_asset::handle::HandleTemplate<bevy_image::image::Image>)
  $1
$1 Str(\"icon.png\")
$2 Struct(my_game::HealthBar)
  field width = $3
$3 Int(100)
";

/// §12.8 Multi-root document, parentheses, comments, non-finite floats, literal forms.
const CORPUS_8: &str = r##"// Two roots. The first is parenthesized, the second is flat.
(
    #Left
    my_game::Link { other: #Right }
),
#Right
my_game::Link { other: #Left }
my_game::Sensor {
    enabled: true,            /* block comment */
    threshold: inf,
    bias: -inf,
    seed: 0xFF,
    label: r#"raw "quoted" text"#,
    fallback: NaN,
}
"##;

const TREE_8: &str = "\
Entity#0 name=\"Left\" base=-
  Patch#1 patch `my_game::Link` value=$0
Entity#2 name=\"Right\" base=-
  Patch#3 patch `my_game::Link` value=$2
  Patch#4 patch `my_game::Sensor` value=$4
values:
$0 Struct(my_game::Link)
  field other = $1
$1 EntityRef(Right)
$2 Struct(my_game::Link)
  field other = $3
$3 EntityRef(Left)
$4 Struct(my_game::Sensor)
  field enabled = $5
  field threshold = $6
  field bias = $7
  field seed = $8
  field label = $9
  field fallback = $10
$5 Bool(true)
$6 Float(inf)
$7 Float(-inf)
$8 Int(255)
$9 Str(\"raw \\\"quoted\\\" text\")
$10 Float(NaN)
";

/// Every corpus document, with its name and expected [`BsnDocument::debug_tree`].
const CORPUS: [(&str, &str, &str); 8] = [
    ("1_minimal_entity", CORPUS_1, TREE_1),
    ("2_struct_patch", CORPUS_2, TREE_2),
    ("3_children", CORPUS_3, TREE_3),
    ("4_inheritance", CORPUS_4, TREE_4),
    ("5_enum_variants", CORPUS_5, TREE_5),
    ("6_tuples_and_lists", CORPUS_6, TREE_6),
    ("7_template_and_scene", CORPUS_7, TREE_7),
    ("8_multi_root", CORPUS_8, TREE_8),
];

fn corpus_documents() -> Vec<(&'static str, BsnDocument)> {
    CORPUS
        .iter()
        .map(|(name, source, _)| {
            (
                *name,
                parse(source).unwrap_or_else(|error| {
                    panic!(
                        "corpus {name} failed to parse: {}",
                        error.render(source, None)
                    )
                }),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------------------
// §11.1 Table-driven lexer tests
// ---------------------------------------------------------------------------------------

fn lex_kinds(source: &str) -> Vec<TokenKind> {
    Lexer::tokenize(source)
        .into_iter()
        .filter(|token| token.kind != TokenKind::Eof)
        .map(|token| token.kind)
        .collect()
}

fn lex_spanned(source: &str) -> Vec<(TokenKind, u32, u32)> {
    Lexer::tokenize(source)
        .into_iter()
        .filter(|token| token.kind != TokenKind::Eof)
        .map(|token| (token.kind, token.span.start, token.span.end))
        .collect()
}

#[test]
fn lex_punctuation_table() {
    let table = [
        ("::", TokenKind::ColonColon),
        (":", TokenKind::Colon),
        (",", TokenKind::Comma),
        ("#", TokenKind::Hash),
        ("@", TokenKind::At),
        ("~", TokenKind::Tilde),
        ("-", TokenKind::Minus),
        ("<", TokenKind::Lt),
        (">", TokenKind::Gt),
        ("(", TokenKind::LParen),
        (")", TokenKind::RParen),
        ("{", TokenKind::LBrace),
        ("}", TokenKind::RBrace),
        ("[", TokenKind::LBracket),
        ("]", TokenKind::RBracket),
    ];
    for (source, kind) in table {
        assert_eq!(
            lex_spanned(source),
            vec![(kind, 0, source.len() as u32)],
            "lexing {source:?}"
        );
    }
}

#[test]
fn lex_colon_vs_coloncolon() {
    assert_eq!(
        lex_kinds(":: :"),
        vec![TokenKind::ColonColon, TokenKind::Colon]
    );
    assert_eq!(
        lex_kinds("a::b"),
        vec![TokenKind::Ident, TokenKind::ColonColon, TokenKind::Ident]
    );
}

#[test]
fn lex_idents_table() {
    for source in ["_a", "A1", "étoile", "r"] {
        assert_eq!(
            lex_spanned(source),
            vec![(TokenKind::Ident, 0, source.len() as u32)],
            "lexing {source:?}"
        );
    }
}

#[test]
fn lex_int_table() {
    let table = [
        ("0", 0i128),
        ("12", 12),
        ("1_000", 1000),
        ("0xFF", 255),
        ("0b1010", 10),
        ("0o17", 15),
    ];
    for (source, expected) in table {
        assert_eq!(
            lex_spanned(source),
            vec![(TokenKind::Int, 0, source.len() as u32)],
            "lexing {source:?}"
        );
        let span = Span::new(0, source.len() as u32);
        assert_eq!(decode_int(source, span).unwrap(), expected);
    }
}

#[test]
fn lex_float_table() {
    let table = [
        ("1.0", 1.0f64),
        ("1.", 1.0),
        ("1e5", 100000.0),
        ("2.5E-3", 0.0025),
        ("1_0.5", 10.5),
    ];
    for (source, expected) in table {
        assert_eq!(
            lex_spanned(source),
            vec![(TokenKind::Float, 0, source.len() as u32)],
            "lexing {source:?}"
        );
        let span = Span::new(0, source.len() as u32);
        assert_eq!(decode_float(source, span).unwrap(), expected);
    }
}

#[test]
fn lex_number_suffix_rejected() {
    for source in ["1u8", "1.0f32", "0x1g"] {
        assert_eq!(
            lex_spanned(source),
            vec![(
                TokenKind::Error(LexError::NumericSuffix),
                0,
                source.len() as u32
            )],
            "lexing {source:?}"
        );
    }
}

#[test]
fn lex_leading_dot_is_not_float() {
    assert_eq!(
        lex_kinds(".5"),
        vec![TokenKind::Error(LexError::Unknown), TokenKind::Int]
    );
}

#[test]
fn lex_string_table() {
    let table: [(&str, &str); 8] = [
        ("\"a\"", "a"),
        ("\"a\\nb\"", "a\nb"),
        ("\"\\u{1F600}\"", "\u{1F600}"),
        ("\"\\x41\"", "A"),
        ("\"\\'\"", "'"),
        ("\"\\\\\\\"\\r\\t\\0\"", "\\\"\r\t\0"),
        ("r\"a\\b\"", "a\\b"),
        ("r#\"a\"b\"#", "a\"b"),
    ];
    for (source, expected) in table {
        assert_eq!(
            lex_spanned(source),
            vec![(TokenKind::Str, 0, source.len() as u32)],
            "lexing {source:?}"
        );
        let span = Span::new(0, source.len() as u32);
        assert_eq!(decode_string(source, span).unwrap(), expected);
    }
}

#[test]
fn lex_string_errors() {
    assert_eq!(
        lex_spanned("\"abc"),
        vec![(TokenKind::Error(LexError::UnterminatedString), 0, 4)]
    );
    let tokens = lex_spanned("\"\\q\"");
    assert_eq!(tokens[0], (TokenKind::Error(LexError::InvalidEscape), 1, 3));
    assert_eq!(
        lex_spanned("r#\"x\""),
        vec![(TokenKind::Error(LexError::UnterminatedRawString), 0, 5)]
    );
}

#[test]
fn decode_string_rejects_malformed_escapes() {
    // The lexer rejects every one of these before the parser can reach the decoder, but
    // `decode_string` is public API: a consumer that decodes a span of its own must get an error
    // rather than a panic or a silently mangled string.
    let table = [
        "\"\\\"",          // a backslash with nothing after it
        "\"\\q\"",         // an unknown escape
        "\"\\xzz\"",       // `\x` without hex digits
        "\"\\x4\"",        // `\x` with only one digit
        "\"\\uABCD\"",     // `\u` without a brace
        "\"\\u{zz}\"",     // `\u{…}` with a non-hex digit
        "\"\\u{41\"",      // `\u{…}` left unclosed
        "\"\\u{D800}\"",   // a surrogate, which is not a `char`
        "\"\\u{110000}\"", // past the last code point
    ];
    for source in table {
        let span = Span::new(0, source.len() as u32);
        let error = decode_string(source, span).expect_err("decoding should fail");
        assert!(
            matches!(error.kind, BsnParseErrorKind::InvalidEscape),
            "decoding {source:?} produced {:?}",
            error.kind
        );
        assert_eq!(error.span, span, "decoding {source:?}");
    }
}

#[test]
fn lex_comments_table() {
    let table = [
        ("// x\nA", 5u32),
        ("/* x */A", 7),
        ("/* /* */ */A", 11),
        ("///doc\nA", 7),
    ];
    for (source, start) in table {
        assert_eq!(
            lex_spanned(source),
            vec![(TokenKind::Ident, start, start + 1)],
            "lexing {source:?}"
        );
    }
    assert_eq!(
        lex_spanned("/*"),
        vec![(TokenKind::Error(LexError::UnterminatedBlockComment), 0, 2)]
    );
}

#[test]
fn lex_trivia_and_bom() {
    assert_eq!(lex_spanned("\u{FEFF}A"), vec![(TokenKind::Ident, 3, 4)]);
    assert_eq!(lex_kinds("\r\n\t  A\r\n"), vec![TokenKind::Ident]);
}

#[test]
fn lex_rejected_chars_table() {
    let table = [
        ("'", LexError::CharLiteral, 1u32),
        ("|", LexError::Closure, 1),
        ("!", LexError::Macro, 1),
        ("r#x", LexError::RawIdentifier, 2),
        ("=", LexError::Unknown, 1),
        (";", LexError::Unknown, 1),
        ("&", LexError::Unknown, 1),
        (".", LexError::Unknown, 1),
    ];
    for (source, error, consumed) in table {
        let tokens = lex_spanned(source);
        assert_eq!(tokens[0].0, TokenKind::Error(error), "lexing {source:?}");
        assert_eq!(tokens[0].1, 0);
        assert_eq!(
            tokens[0].2, consumed,
            "lexing {source:?} consumed too little"
        );
        assert!(tokens[0].2 > 0);
    }
}

#[test]
fn lex_never_loops() {
    const SAMPLE: &str = "#Root a::B { c: 1.0, d: \"x\\ny\" } E [ F(1,) ] ~G @H -0x1 /* c */ // l\n'|!=;&.\u{FEFF}étoile r#\"raw\"# r#id 1u8 .5 :: :";
    let sample: String = SAMPLE.chars().cycle().take(200).collect();
    let tokens = Lexer::tokenize(&sample);
    assert_eq!(tokens.last().map(|token| token.kind), Some(TokenKind::Eof));
    for ch in sample.chars() {
        let text = ch.to_string();
        let tokens = Lexer::tokenize(&text);
        assert_eq!(tokens.last().map(|token| token.kind), Some(TokenKind::Eof));
    }
}

#[test]
fn lex_eof_span_is_end_of_file() {
    let source = "A { x: 1 }";
    let tokens = Lexer::tokenize(source);
    let eof = tokens.last().copied().unwrap();
    assert_eq!(eof.kind, TokenKind::Eof);
    assert_eq!(
        eof.span,
        Span::new(source.len() as u32, source.len() as u32)
    );
}

// ---------------------------------------------------------------------------------------
// §11.2 Parser corpus / structural tests
// ---------------------------------------------------------------------------------------

macro_rules! corpus_test {
    ($name:ident, $index:expr) => {
        #[test]
        fn $name() {
            let (_, source, expected) = CORPUS[$index];
            let document =
                parse(source).unwrap_or_else(|error| panic!("{}", error.render(source, None)));
            assert_eq!(document.debug_tree(), expected);
        }
    };
}

corpus_test!(parse_corpus_1_minimal_entity, 0);
corpus_test!(parse_corpus_2_struct_patch, 1);
corpus_test!(parse_corpus_3_children, 2);
corpus_test!(parse_corpus_4_inheritance, 3);
corpus_test!(parse_corpus_5_enum_variants, 4);
corpus_test!(parse_corpus_6_tuples_and_lists, 5);
corpus_test!(parse_corpus_7_template_and_scene, 6);
corpus_test!(parse_corpus_8_multi_root, 7);

#[test]
fn parse_empty_document() {
    for source in ["", "// only a comment", "\n\n   \n"] {
        let document = parse(source).unwrap();
        assert!(document.roots.is_empty(), "parsing {source:?}");
        assert!(document.nodes.is_empty());
        assert!(document.values.is_empty());
    }
}

#[test]
fn parse_node_ids_are_preorder() {
    let document = parse(CORPUS_3).unwrap();
    assert_eq!(document.roots, vec![BsnNodeId(0)]);
    let entity_ids: Vec<u32> = document.entities().map(|node| node.id.0).collect();
    assert_eq!(entity_ids, vec![0, 3, 5]);
    for window in entity_ids.windows(2) {
        assert!(window[0] < window[1]);
    }
}

#[test]
fn parse_node_ids_stable_across_reparse() {
    for (name, source, _) in CORPUS {
        let first = parse(source).unwrap();
        let second = parse(source).unwrap();
        assert_eq!(first.debug_tree(), second.debug_tree(), "corpus {name}");
        let first_ids: Vec<u32> = first.entities().map(|node| node.id.0).collect();
        let second_ids: Vec<u32> = second.entities().map(|node| node.id.0).collect();
        assert_eq!(first_ids, second_ids, "corpus {name}");
    }
}

#[test]
fn parse_patch_value_invariant() {
    for (name, document) in corpus_documents() {
        for node in &document.nodes {
            let BsnNodeKind::Patch { symbol, value, .. } = &node.kind else {
                continue;
            };
            let value = document.value(*value).expect("patch value exists");
            let path = match &value.value {
                BsnValue::Path(path)
                | BsnValue::Struct(path, _)
                | BsnValue::NamedTuple(path, _) => path,
                other => panic!("corpus {name}: patch value is {other:?}"),
            };
            assert!(
                path.structural_eq(symbol),
                "corpus {name}: {} != {}",
                path.to_type_path(),
                symbol.to_type_path()
            );
        }
    }
}

#[test]
fn parse_multiple_patches_same_component() {
    let document = parse("A A { x: 1 }").unwrap();
    let symbols: Vec<String> = document
        .nodes
        .iter()
        .filter_map(|node| match &node.kind {
            BsnNodeKind::Patch { symbol, .. } => Some(symbol.to_type_path()),
            _ => None,
        })
        .collect();
    assert_eq!(symbols, vec!["A".to_string(), "A".to_string()]);
}

#[test]
fn parse_spans_cover_source() {
    for (name, source, _) in CORPUS {
        let document = parse(source).unwrap();
        let length = source.len() as u32;
        for node in &document.nodes {
            assert!(node.span.start <= node.span.end, "corpus {name}");
            assert!(node.span.end <= length, "corpus {name}");
        }
        for value in &document.values {
            assert!(value.span.start <= value.span.end, "corpus {name}");
            assert!(value.span.end <= length, "corpus {name}");
        }
        for node in &document.nodes {
            match &node.kind {
                BsnNodeKind::Entity {
                    patches, relations, ..
                } => {
                    for child in patches.iter().chain(relations) {
                        let child = document.node(*child).unwrap();
                        assert!(
                            node.span.start <= child.span.start && child.span.end <= node.span.end,
                            "corpus {name}: entity {:?} does not contain {:?}",
                            node.span,
                            child.span
                        );
                    }
                }
                BsnNodeKind::Relation { entities, .. } => {
                    for child in entities {
                        let child = document.node(*child).unwrap();
                        assert!(
                            node.span.start <= child.span.start && child.span.end <= node.span.end,
                            "corpus {name}"
                        );
                    }
                }
                BsnNodeKind::Patch { .. } => {}
            }
        }
    }
}

#[test]
fn parse_relation_and_patch_source_order() {
    let document = parse(CORPUS_3).unwrap();
    let BsnNodeKind::Entity {
        patches, relations, ..
    } = &document.node(BsnNodeId(0)).unwrap().kind
    else {
        panic!("root is not an entity");
    };
    let mut merged: Vec<BsnNodeId> = patches.iter().chain(relations).copied().collect();
    merged.sort_by_key(|id| document.node(*id).unwrap().span.start);
    assert_eq!(merged, vec![BsnNodeId(1), BsnNodeId(2)]);
}

#[test]
fn structural_eq_compares_floats_by_bits() {
    let mut left = BsnDocument::new();
    let value = left.push_value(BsnValue::Float(f64::NAN));
    let patch = left.push_patch(
        BsnPatchPrefix::FromTemplate,
        BsnPath::from_segments(["A"]),
        PatchBody::Struct(vec![("x".to_string(), value)]),
    );
    let root = left.push_node(entity(vec![patch], vec![]));
    left.push_root(root);

    let same = left.clone();
    assert!(left.structural_eq(&same), "NaN must equal NaN");

    let mut different = left.clone();
    different.values[0].value = BsnValue::Float(0.0);
    assert!(!left.structural_eq(&different), "NaN must not equal 0.0");

    let mut negative_zero = left.clone();
    negative_zero.values[0].value = BsnValue::Float(-0.0);
    assert!(!different.structural_eq(&negative_zero), "0.0 != -0.0");
}

#[test]
fn document_helpers_match_the_free_functions() {
    let document = BsnDocument::parse(CORPUS_2).expect("the corpus parses");
    assert_eq!(document.to_bsn_string(), print_document(&document));
    assert!(document.structural_eq(&parse(CORPUS_2).unwrap()));
}

#[test]
fn structural_eq_rejects_mismatched_documents() {
    /// A document holding one root entity with a single `A` patch whose value is `value`.
    fn document_with(value: BsnValue) -> BsnDocument {
        let mut document = BsnDocument::new();
        let value = document.push_value(value);
        let patch = document.push_node(BsnNodeKind::Patch {
            symbol: BsnPath::from_segments(["A"]),
            prefix: BsnPatchPrefix::FromTemplate,
            value,
        });
        let root = document.push_node(entity(vec![patch], vec![]));
        document.push_root(root);
        document
    }

    let left = document_with(BsnValue::Int(1));
    assert!(left.structural_eq(&document_with(BsnValue::Int(1))));

    // A different number of roots.
    assert!(!left.structural_eq(&BsnDocument::new()));

    // A root of a different kind.
    let mut patch_root = BsnDocument::new();
    let only = patch_root.push_patch(
        BsnPatchPrefix::FromTemplate,
        BsnPath::from_segments(["A"]),
        PatchBody::Unit,
    );
    patch_root.push_root(only);
    assert!(!patch_root.structural_eq(&left));

    // A value of a different kind.
    assert!(!left.structural_eq(&document_with(BsnValue::String("1".to_string()))));

    // A dangling node id.
    let mut dangling_node = BsnDocument::new();
    dangling_node.push_root(BsnNodeId(9));
    assert!(!dangling_node.structural_eq(&left));

    // A dangling value id.
    let mut dangling_value = left.clone();
    let BsnNodeKind::Patch { value, .. } = &mut dangling_value.nodes[0].kind else {
        panic!("the first node is the patch");
    };
    *value = BsnValueId(9);
    assert!(!left.structural_eq(&dangling_value));
}

#[test]
fn structural_eq_stops_on_cycles_and_over_deep_documents() {
    // Comparing two hand-built documents has to terminate even when they are cyclic or nested
    // deeper than the walk guard; the comparison gives up and reports "not equal".
    let mut cyclic = BsnDocument::new();
    let relation = cyclic.push_node(BsnNodeKind::Relation {
        target_symbol: BsnPath::from_segments(["Children"]),
        entities: Vec::new(),
    });
    let root = cyclic.push_node(entity(vec![], vec![relation]));
    cyclic.nodes[relation.0 as usize].kind = BsnNodeKind::Relation {
        target_symbol: BsnPath::from_segments(["Children"]),
        entities: vec![root],
    };
    cyclic.push_root(root);
    assert!(!cyclic.structural_eq(&cyclic.clone()));

    let mut deep = BsnDocument::new();
    let mut value = deep.push_value(BsnValue::Int(1));
    for _ in 0..=MAX_WALK_DEPTH {
        value = deep.push_value(BsnValue::List(vec![value]));
    }
    let patch = deep.push_node(BsnNodeKind::Patch {
        symbol: BsnPath::from_segments(["A"]),
        prefix: BsnPatchPrefix::FromTemplate,
        value,
    });
    let root = deep.push_node(entity(vec![patch], vec![]));
    deep.push_root(root);
    assert!(!deep.structural_eq(&deep.clone()));
}

/// Builds an `Entity` node kind with no name and no base.
fn entity(patches: Vec<BsnNodeId>, relations: Vec<BsnNodeId>) -> BsnNodeKind {
    BsnNodeKind::Entity {
        name: None,
        name_span: None,
        base: None,
        base_span: None,
        patches,
        relations,
    }
}

// ---------------------------------------------------------------------------------------
// §11.3 Printer and round-trip tests
// ---------------------------------------------------------------------------------------

#[test]
fn roundtrip_corpus_structural() {
    for (name, document) in corpus_documents() {
        let text = print_document(&document);
        let reparsed = parse(&text)
            .unwrap_or_else(|error| panic!("corpus {name}: {}", error.render(&text, None)));
        assert!(
            reparsed.structural_eq(&document),
            "corpus {name} did not survive a print/parse round trip:\n{text}"
        );
    }
}

#[test]
fn roundtrip_corpus_debug_tree() {
    for (name, document) in corpus_documents() {
        let text = print_document(&document);
        let reparsed = parse(&text).unwrap();
        assert_eq!(
            reparsed.debug_tree(),
            document.debug_tree(),
            "corpus {name}\n{text}"
        );
    }
}

#[test]
fn print_is_idempotent() {
    for (name, source, _) in CORPUS {
        let once = print_document(&parse(source).unwrap());
        let twice = print_document(&parse(&once).unwrap());
        assert_eq!(once, twice, "corpus {name}");
    }
}

const PRINT_1: &str = "bevy_transform::components::transform::Transform\n";

const PRINT_2: &str = "\
#Camera
bevy_camera::components::Camera3d
bevy_transform::components::transform::Transform {
    translation: glam::Vec3 { x: 0.0, y: 6.0, z: -12.5 },
    scale: glam::Vec3 { x: 1.0 },
}
";

const PRINT_3: &str = "\
#Root
bevy_ui::ui_node::Node
bevy_ecs::hierarchy::Children [
    #Label
    bevy_ui::widget::text::Text(\"hello\"),
    my_game::Follower { target: #Root }
]
";

const PRINT_4: &str = "\
:\"enemies/orc.bsn\"
my_game::Health { max: 200 }
bevy_ecs::hierarchy::Children [
    my_game::Weapon
]
";

const PRINT_5: &str = "\
bevy_camera::visibility::Visibility::Visible
my_game::Shape::Circle { radius: 2.5 }
my_game::Shape::Rect(1.0, 2.0)
";

const PRINT_6: &str = "\
my_game::Rgba(1.0, 0.5)
bevy_render::mesh::components::Mesh3d(\"models/tree.gltf#Mesh0/Primitive0\")
my_game::Waypoints([1.0, 2.0, 3.0])
my_game::Marker(())
my_game::Pair((1, 2))
";

const PRINT_7: &str = "\
~bevy_asset::handle::HandleTemplate<bevy_image::image::Image>(\"icon.png\")
@my_game::HealthBar { width: 100 }
";

const PRINT_8: &str = "\
#Left
my_game::Link { other: #Right },

#Right
my_game::Link { other: #Left }
my_game::Sensor {
    enabled: true,
    threshold: inf,
    bias: -inf,
    seed: 255,
    label: \"raw \\\"quoted\\\" text\",
    fallback: NaN,
}
";

const PRINTED: [&str; 8] = [
    PRINT_1, PRINT_2, PRINT_3, PRINT_4, PRINT_5, PRINT_6, PRINT_7, PRINT_8,
];

#[test]
fn print_snapshot_corpus() {
    for (index, (name, document)) in corpus_documents().into_iter().enumerate() {
        assert_eq!(print_document(&document), PRINTED[index], "corpus {name}");
    }
}

#[test]
fn print_builder_document() {
    let mut document = BsnDocument::new();

    let x = document.push_value(BsnValue::Float(1.0));
    let y = document.push_value(BsnValue::Float(-2.0));
    let translation = document.push_value(BsnValue::Struct(
        BsnPath::from_segments(["glam", "Vec3"]),
        vec![("x".to_string(), x), ("y".to_string(), y)],
    ));
    let transform = document.push_patch(
        BsnPatchPrefix::FromTemplate,
        BsnPath::from_segments(["my_game", "Transform"]),
        PatchBody::Struct(vec![("translation".to_string(), translation)]),
    );
    let marker = document.push_patch(
        BsnPatchPrefix::Template,
        BsnPath::from_segments(["my_game", "MarkerTemplate"]),
        PatchBody::Unit,
    );
    let child_patch = document.push_patch(
        BsnPatchPrefix::FromTemplate,
        BsnPath::from_segments(["my_game", "Child"]),
        PatchBody::Tuple(vec![]),
    );
    let child = document.push_node(entity(vec![child_patch], vec![]));
    let relation = document.push_node(BsnNodeKind::Relation {
        target_symbol: BsnPath::from_segments(["bevy_ecs", "hierarchy", "Children"]),
        entities: vec![child],
    });
    let root = document.push_node(BsnNodeKind::Entity {
        name: Some("Root".to_string()),
        name_span: None,
        base: Some("base.bsn".to_string()),
        base_span: None,
        patches: vec![transform, marker],
        relations: vec![relation],
    });
    document.push_root(root);

    let text = print_document(&document);
    assert_eq!(
        text,
        "\
:\"base.bsn\"
#Root
my_game::Transform { translation: glam::Vec3 { x: 1.0, y: -2.0 } }
~my_game::MarkerTemplate
bevy_ecs::hierarchy::Children [
    my_game::Child()
]
"
    );
    let reparsed = parse(&text).unwrap();
    assert!(reparsed.structural_eq(&document));
}

#[test]
fn print_entry_order_from_spans() {
    let source = "A\nbevy_ecs::hierarchy::Children [\n    B\n]\nC\n";
    let document = parse(source).unwrap();
    assert_eq!(print_document(&document), source);
}

#[test]
fn print_entry_order_from_ids() {
    let mut document = BsnDocument::new();
    let a = document.push_patch(
        BsnPatchPrefix::FromTemplate,
        BsnPath::from_segments(["A"]),
        PatchBody::Unit,
    );
    let b = document.push_patch(
        BsnPatchPrefix::FromTemplate,
        BsnPath::from_segments(["B"]),
        PatchBody::Unit,
    );
    let child = document.push_node(entity(vec![b], vec![]));
    let relation = document.push_node(BsnNodeKind::Relation {
        target_symbol: BsnPath::from_segments(["Children"]),
        entities: vec![child],
    });
    let c = document.push_patch(
        BsnPatchPrefix::FromTemplate,
        BsnPath::from_segments(["C"]),
        PatchBody::Unit,
    );
    let root = document.push_node(BsnNodeKind::Entity {
        name: None,
        name_span: None,
        base: None,
        base_span: None,
        patches: vec![a, c],
        relations: vec![relation],
    });
    document.push_root(root);
    assert_eq!(
        print_document(&document),
        "A\nChildren [\n    B\n]\nC\n",
        "builder order is id order"
    );
}

#[test]
fn print_inline_vs_multiline() {
    let narrow = "A { x: 1, y: 2 }\n";
    let document = parse(narrow).unwrap();
    assert_eq!(print_document(&document), narrow);

    let wide = "A { a_really_quite_extremely_long_field_name_number_one: 1, a_really_quite_extremely_long_field_name_number_two: 2 }\n";
    assert!(wide.len() > 100);
    let document = parse(wide).unwrap();
    assert_eq!(
        print_document(&document),
        "\
A {
    a_really_quite_extremely_long_field_name_number_one: 1,
    a_really_quite_extremely_long_field_name_number_two: 2,
}
"
    );

    let document = parse(narrow).unwrap();
    let mut out = String::new();
    write_document_with(
        &document,
        &mut out,
        &PrintOptions {
            max_inline_width: 0,
            ..PrintOptions::default()
        },
    )
    .unwrap();
    assert_eq!(out, "A {\n    x: 1,\n    y: 2,\n}\n");
}

#[test]
fn print_string_escapes() {
    let source = "A { x: \"a\\\"b\\\\c\\n\\t\\0\\u{7}\\u{1F600}\" }\n";
    let document = parse(source).unwrap();
    let text = print_document(&document);
    assert!(text.contains("\\u{7}"), "{text}");
    assert!(text.contains('\u{1F600}'), "{text}");
    assert!(!text.contains("\\u{1f600}"), "{text}");
    let reparsed = parse(&text).unwrap();
    assert!(reparsed.structural_eq(&document));
}

#[test]
fn print_non_finite_floats() {
    let document = parse("A { a: inf, b: -inf, c: NaN, d: -0.0 }\n").unwrap();
    let text = print_document(&document);
    assert_eq!(text, "A { a: inf, b: -inf, c: NaN, d: -0.0 }\n");
    let reparsed = parse(&text).unwrap();
    let floats: Vec<f64> = reparsed
        .values
        .iter()
        .filter_map(|value| match value.value {
            BsnValue::Float(float) => Some(float),
            _ => None,
        })
        .collect();
    assert_eq!(floats[0], f64::INFINITY);
    assert_eq!(floats[1], f64::NEG_INFINITY);
    assert!(floats[2].is_nan());
    assert!(floats[3].is_sign_negative() && floats[3] == 0.0);
}

#[test]
fn print_float_keeps_decimal_point() {
    let document = parse("A { x: 1.0 }").unwrap();
    let text = print_document(&document);
    assert_eq!(text, "A { x: 1.0 }\n");
    let reparsed = parse(&text).unwrap();
    assert!(matches!(reparsed.values[1].value, BsnValue::Float(_)));
}

#[test]
fn print_int_normalizes_radix() {
    let document = parse("A { x: 0xFF, y: 1_000 }").unwrap();
    assert_eq!(print_document(&document), "A { x: 255, y: 1000 }\n");
}

#[test]
fn print_generic_paths() {
    let document = parse("alloc::vec::Vec<f32>\n").unwrap();
    assert_eq!(print_document(&document), "alloc::vec::Vec<f32>\n");

    let path = BsnPath::from_type_path("A<B<C>, D>").expect("valid path");
    assert_eq!(path.to_type_path(), "A<B<C>, D>");
    assert_eq!(BsnPath::from_type_path("::A"), None);
    assert_eq!(BsnPath::from_type_path("A B"), None);

    let document = parse("A<B<C>, D>\n").unwrap();
    assert_eq!(print_document(&document), "A<B<C>, D>\n");
}

#[test]
fn print_empty_document() {
    let document = BsnDocument::new();
    assert_eq!(print_document(&document), "");
    assert!(parse(&print_document(&document)).unwrap().roots.is_empty());
}

#[test]
fn print_multi_root() {
    let text = print_document(&parse(CORPUS_8).unwrap());
    assert!(text.contains("},\n\n#Right"), "{text}");
    assert!(!text.trim_end().ends_with(','), "{text}");
}

#[test]
fn print_nested_relations() {
    let source = "\
A
Children [
    B
    Children [
        C
        Children [
            D
        ]
    ]
]
";
    let document = parse(source).unwrap();
    assert_eq!(print_document(&document), source);
    assert!(source.contains("\n            D\n"));
}

#[test]
fn print_never_panics_on_dangling_id() {
    let mut document = BsnDocument::new();
    let patch = document.push_node(BsnNodeKind::Patch {
        symbol: BsnPath::from_segments(["A"]),
        prefix: BsnPatchPrefix::FromTemplate,
        value: BsnValueId(99),
    });
    let root = document.push_node(entity(vec![patch], vec![]));
    document.push_root(root);
    let text = print_document(&document);
    assert!(text.contains("/* <invalid node id 99> */"), "{text}");

    let mut document = BsnDocument::new();
    let root = document.push_node(entity(vec![BsnNodeId(42)], vec![]));
    document.push_root(root);
    assert!(print_document(&document).contains("/* <invalid node id 42> */"));
}

#[test]
fn print_stops_on_cycles_and_over_deep_documents() {
    const TOO_DEEP: &str = "/* <nesting too deep> */";
    let children = || BsnPath::from_segments(["Children"]);

    // An entity whose relation block contains the entity itself.
    let mut document = BsnDocument::new();
    let relation = document.push_node(BsnNodeKind::Relation {
        target_symbol: children(),
        entities: Vec::new(),
    });
    let root = document.push_node(entity(vec![], vec![relation]));
    document.nodes[relation.0 as usize].kind = BsnNodeKind::Relation {
        target_symbol: children(),
        entities: vec![root],
    };
    document.push_root(root);
    let text = print_document(&document);
    assert!(text.contains(TOO_DEEP), "{text}");

    // A chain of entities nested deeper than the walk guard.
    let mut document = BsnDocument::new();
    let mut node = document.push_node(entity(vec![], vec![]));
    for _ in 0..=MAX_WALK_DEPTH {
        let relation = document.push_node(BsnNodeKind::Relation {
            target_symbol: children(),
            entities: vec![node],
        });
        node = document.push_node(entity(vec![], vec![relation]));
    }
    document.push_root(node);
    let text = print_document(&document);
    assert!(text.contains(TOO_DEEP), "the entity walk must stop");

    // A chain of values nested deeper than the walk guard.
    let mut document = BsnDocument::new();
    let mut value = document.push_value(BsnValue::Int(1));
    for _ in 0..=MAX_WALK_DEPTH {
        value = document.push_value(BsnValue::List(vec![value]));
    }
    let patch = document.push_node(BsnNodeKind::Patch {
        symbol: BsnPath::from_segments(["A"]),
        prefix: BsnPatchPrefix::FromTemplate,
        value,
    });
    let root = document.push_node(entity(vec![patch], vec![]));
    document.push_root(root);
    let text = print_document(&document);
    assert!(text.contains(TOO_DEEP), "the value walk must stop");
}

#[test]
fn print_output_ends_with_newline() {
    for (name, document) in corpus_documents() {
        let text = print_document(&document);
        assert!(text.ends_with('\n'), "corpus {name}");
        assert!(!text.ends_with("\n\n"), "corpus {name}");
        assert!(!text.contains('\r'), "corpus {name}");
    }
}

// ---------------------------------------------------------------------------------------
// §11.4 Rejected-construct diagnostics
// ---------------------------------------------------------------------------------------

fn err(source: &str) -> BsnParseError {
    parse(source).expect_err("expected a parse error")
}

fn assert_unsupported(source: &str, message: &'static str) -> BsnParseError {
    let error = err(source);
    match error.kind {
        BsnParseErrorKind::Unsupported(actual) => assert_eq!(
            actual, message,
            "parsing {source:?} produced the wrong diagnostic"
        ),
        ref other => {
            panic!("parsing {source:?} produced {other:?}, expected an unsupported construct")
        }
    }
    error
}

#[test]
fn reject_expr_entry() {
    let error = assert_unsupported("{ my_scene() }", unsupported::EXPR);
    assert_eq!(error.span, Span::new(0, 1));
}

#[test]
fn reject_expr_value() {
    assert_unsupported("A { x: { 1 + 2 } }", unsupported::EXPR);
}

#[test]
fn reject_expr_const_block() {
    assert_unsupported("A { x: const { 1 } }", unsupported::EXPR);
    assert_unsupported("A { x: unsafe { 1 } }", unsupported::EXPR);
}

#[test]
fn reject_closure() {
    assert_unsupported("A { x: |c| { 1 } }", unsupported::CLOSURE);
}

#[test]
fn reject_observer() {
    assert_unsupported("on(my_obs)", unsupported::OBSERVER);
}

#[test]
fn reject_scene_fn() {
    assert_unsupported("my_scene()", unsupported::FN);
    assert_unsupported("my_scene", unsupported::FN);
}

#[test]
fn reject_ctor() {
    assert_unsupported("A { x: Color::srgb(1.0, 0.0, 0.0) }", unsupported::CTOR);
}

#[test]
fn reject_const_bare() {
    assert_unsupported("A { x: PI }", unsupported::CONST);
}

#[test]
fn reject_const_assoc() {
    assert_unsupported("A { x: f32::MAX }", unsupported::CONST);
}

#[test]
fn reject_shorthand() {
    let error = assert_unsupported("A { width }", unsupported::SHORTHAND);
    assert_eq!(error.span, Span::new(4, 9));
}

#[test]
fn reject_prop() {
    assert_unsupported("@W { @prop: 1 }", unsupported::PROP);
}

#[test]
fn reject_macro() {
    assert_unsupported("A { x: vec![1] }", unsupported::MACRO);
}

#[test]
fn reject_use() {
    assert_unsupported("use a::B;", unsupported::USE);
}

#[test]
fn reject_char() {
    assert_unsupported("A { x: 'c' }", unsupported::CHAR);
}

#[test]
fn reject_suffix() {
    assert_unsupported("A { x: 1.0f32 }", unsupported::SUFFIX);
}

#[test]
fn reject_raw_ident() {
    assert_unsupported("r#type", unsupported::RAW_IDENT);
}

#[test]
fn reject_lowercase_value_path() {
    assert_unsupported("A { x: foo::bar }", unsupported::PATH_CASE);
}

#[test]
fn reject_base_not_first() {
    let error = err("A :\"b.bsn\"");
    assert!(matches!(error.kind, BsnParseErrorKind::BaseNotFirst));
}

#[test]
fn reject_base_not_string() {
    let error = err(":enemy()");
    assert!(matches!(error.kind, BsnParseErrorKind::BaseNotString));
    assert!(matches!(err(":@W").kind, BsnParseErrorKind::BaseNotString));
    assert!(matches!(err(":Foo").kind, BsnParseErrorKind::BaseNotString));
}

#[test]
fn reject_duplicate_name() {
    assert!(matches!(
        err("#A #B").kind,
        BsnParseErrorKind::DuplicateName
    ));
}

#[test]
fn reject_duplicate_field() {
    let error = err("A { x: 1, x: 2 }");
    match error.kind {
        BsnParseErrorKind::DuplicateField(ref name) => assert_eq!(name, "x"),
        ref other => panic!("expected DuplicateField, got {other:?}"),
    }
}

#[test]
fn reject_neg_operand() {
    assert!(matches!(
        err("A { x: -B }").kind,
        BsnParseErrorKind::NegOperand
    ));
}

#[test]
fn reject_leading_path_sep() {
    assert!(matches!(
        err("::a::B").kind,
        BsnParseErrorKind::LeadingPathSeparator
    ));
    assert!(matches!(
        err("A { x: ::a::B }").kind,
        BsnParseErrorKind::LeadingPathSeparator
    ));
}

#[test]
fn reject_int_out_of_range() {
    let nines: String = core::iter::repeat_n('9', 40).collect();
    let source = format!("A {{ x: {nines} }}");
    assert!(matches!(
        err(&source).kind,
        BsnParseErrorKind::NumberOutOfRange
    ));
}

#[test]
fn reject_nesting_too_deep() {
    let source = format!("A {{ x: {} }}", "[".repeat(200));
    assert!(matches!(
        err(&source).kind,
        BsnParseErrorKind::NestingTooDeep
    ));
    const { assert!(MAX_NESTING_DEPTH < 200) };
}

#[test]
fn reject_unclosed_brace() {
    let error = err("A { x: 1");
    assert!(matches!(error.kind, BsnParseErrorKind::UnexpectedEof));
    assert!(error.expected.contains(&"`,`"));
    assert!(error.expected.contains(&"`}`"));
}

#[test]
fn reject_tilde_relation() {
    let error = err("~Children [ ]");
    assert!(matches!(
        error.kind,
        BsnParseErrorKind::UnexpectedToken { .. }
    ));
    assert!(
        error
            .expected
            .iter()
            .any(|text| text.contains('~') && text.contains('@')),
        "{:?}",
        error.expected
    );
}

#[test]
fn error_messages_end_with_a_remedy() {
    let messages = [
        unsupported::EXPR,
        unsupported::CLOSURE,
        unsupported::OBSERVER,
        unsupported::FN,
        unsupported::CTOR,
        unsupported::CONST,
        unsupported::SHORTHAND,
        unsupported::PROP,
        unsupported::MACRO,
        unsupported::USE,
        unsupported::CHAR,
        unsupported::SUFFIX,
        unsupported::RAW_IDENT,
        unsupported::PATH_CASE,
    ];
    for message in messages {
        assert!(message.contains("`.bsn`"), "{message}");
        assert!(message.len() >= 40, "{message}");
    }
}

// ---------------------------------------------------------------------------------------
// §11.2 error rendering (step 2 of the implementation plan)
// ---------------------------------------------------------------------------------------

#[test]
fn span_line_col() {
    let source = "abc\ndef\u{1F600}ghi";
    assert_eq!(Span::new(0, 1).line_col(source), (1, 1));
    assert_eq!(Span::new(2, 3).line_col(source), (1, 3));
    assert_eq!(Span::new(4, 5).line_col(source), (2, 1));
    // The emoji occupies four bytes but a single column.
    assert_eq!(Span::new(11, 12).line_col(source), (2, 5));
    assert_eq!(Span::new(999, 999).line_col(source), (2, 8));
    assert_eq!(Span::new(0, 3).text(source), "abc");
    assert_eq!(Span::new(0, 999).text(source), "");
    assert_eq!(Span::new(1, 2).join(Span::new(5, 6)), Span::new(1, 6));
    assert!(Span::NONE.is_none());
}

#[test]
fn render_points_at_the_right_column() {
    let source = "A { width }";
    let error = err(source);
    let rendered = error.render(source, Some("assets/player.bsn"));
    let expected = "\
error: Field shorthand (`{ name }`) is not supported in `.bsn` assets, because there are no variables to capture. Write `name: <value>` instead.
  --> assets/player.bsn:1:5
   |
 1 | A { width }
   |     ^^^^^
";
    assert_eq!(rendered, expected);

    let error = err("A\nB\nC { x }");
    let rendered = error.render("A\nB\nC { x }", None);
    assert!(rendered.contains("--> <bsn>:3:5"), "{rendered}");
}

// ---------------------------------------------------------------------------------------
// §11.5 Hygiene tests
// ---------------------------------------------------------------------------------------

/// Every shipped library source file: `lib.rs` plus each module it declares outside
/// `#[cfg(test)]`. [`sources_have_no_bevy_or_std_references`] checks that this list is complete,
/// so a new module cannot silently escape the scan.
const SOURCES: [(&str, &str); 6] = [
    ("lib.rs", include_str!("lib.rs")),
    ("ast.rs", include_str!("ast.rs")),
    ("error.rs", include_str!("error.rs")),
    ("lexer.rs", include_str!("lexer.rs")),
    ("parser.rs", include_str!("parser.rs")),
    ("printer.rs", include_str!("printer.rs")),
];

/// The modules exempt from [`SOURCES`], because they exist only under `#[cfg(test)]` and so are
/// never compiled into the shipped crate. Anything added here has to be a deliberate decision.
const TEST_ONLY_MODULES: [&str; 2] = ["adversarial", "tests"];

/// Splits `lib.rs` into the module names it declares, as `(shipped, test only)`.
fn declared_modules(lib: &str) -> (Vec<String>, Vec<String>) {
    let mut shipped = Vec::new();
    let mut test_only = Vec::new();
    let mut under_cfg_test = false;
    for line in lib.lines() {
        let line = line.trim();
        if line == "#[cfg(test)]" {
            under_cfg_test = true;
            continue;
        }
        let declaration = line
            .strip_prefix("pub(crate) ")
            .or_else(|| line.strip_prefix("pub "))
            .unwrap_or(line);
        if let Some(name) = declaration
            .strip_prefix("mod ")
            .and_then(|rest| rest.strip_suffix(';'))
        {
            if under_cfg_test {
                test_only.push(name.to_string());
            } else {
                shipped.push(name.to_string());
            }
        }
        under_cfg_test = false;
    }
    (shipped, test_only)
}

#[test]
fn manifest_has_no_bevy_dependencies() {
    let manifest = include_str!("../Cargo.toml");
    for line in manifest.lines() {
        if line.starts_with("name = ") {
            continue;
        }
        assert!(
            !line.contains("bevy_"),
            "the manifest may not mention another bevy crate: {line}"
        );
    }
    assert!(!manifest.contains("[dev-dependencies]"));
    assert!(!manifest.contains("[build-dependencies]"));

    let mut in_dependencies = false;
    let mut dependencies: Vec<&str> = Vec::new();
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_dependencies = line == "[dependencies]";
            continue;
        }
        if in_dependencies && !line.is_empty() && !line.starts_with('#') {
            dependencies.push(line);
        }
    }
    assert_eq!(dependencies.len(), 1, "{dependencies:?}");
    assert!(dependencies[0].starts_with("thiserror"), "{dependencies:?}");
}

#[test]
fn sources_have_no_bevy_or_std_references() {
    // The scan is only as good as its list, so derive that list from `lib.rs` itself: every
    // module declared outside `#[cfg(test)]` must appear in `SOURCES`, in the same order.
    let (shipped, test_only) = declared_modules(SOURCES[0].1);
    let expected: Vec<String> = core::iter::once("lib.rs".to_string())
        .chain(shipped.iter().map(|name| format!("{name}.rs")))
        .collect();
    let scanned: Vec<String> = SOURCES.iter().map(|(name, _)| name.to_string()).collect();
    assert_eq!(
        scanned, expected,
        "`SOURCES` must list `lib.rs` and every module it declares outside `cfg(test)`"
    );
    assert_eq!(
        test_only, TEST_ONLY_MODULES,
        "only `cfg(test)`-only modules may be left out of the scan"
    );

    for (name, source) in SOURCES {
        // The crate is allowed to name itself, and the `use`-import diagnostic quotes a
        // fully-qualified type path as an example. Neither is a dependency: no other
        // mention of the bevy namespace may appear, because the crate must build without
        // any of it.
        let scrubbed = source.replace("bevy_bsn", "").replace(unsupported::USE, "");
        assert!(
            !scrubbed.contains("bevy_"),
            "{name} references a bevy crate"
        );
        assert!(!source.contains("std::"), "{name} references std");
        assert!(!source.contains("use std"), "{name} references std");
    }
    // `extern crate std;` is the sole permitted mention, and only in `lib.rs`.
    assert_eq!(SOURCES[0].1.matches("extern crate std;").count(), 1);
    for (name, source) in &SOURCES[1..] {
        assert!(!source.contains("extern crate std"), "{name}");
    }
}

#[test]
fn parse_is_deterministic() {
    let first = parse(CORPUS_8).unwrap().debug_tree();
    for _ in 0..10 {
        assert_eq!(parse(CORPUS_8).unwrap().debug_tree(), first);
    }
}

#[test]
fn parse_never_panics_on_truncation() {
    for (index, _) in CORPUS_8.char_indices() {
        let _ = parse(&CORPUS_8[..index]);
    }
    let _ = parse(CORPUS_8);
}

/// A tiny deterministic xorshift generator, so the fuzz-lite tests are reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.0 = state;
        state
    }

    fn index(&mut self, len: usize) -> usize {
        if len == 0 {
            0
        } else {
            (self.next() % len as u64) as usize
        }
    }
}

#[test]
fn parse_never_panics_on_byte_flip() {
    const ALPHABET: [char; 28] = [
        '{', '}', '[', ']', '(', ')', ',', ':', '#', '@', '~', '-', '"', '\\', '\'', '|', '!', '0',
        'a', 'A', '.', '<', '>', 'r', '\n', '/', '*', '_',
    ];
    let mut rng = Rng(0x5eed_1234_9876_abcd);
    let original: Vec<char> = CORPUS_8.chars().collect();
    for _ in 0..500 {
        let mut mutated = original.clone();
        let index = rng.index(mutated.len());
        mutated[index] = ALPHABET[rng.index(ALPHABET.len())];
        let source: String = mutated.into_iter().collect();
        let _ = parse(&source);
    }
}

#[test]
fn print_never_panics_on_fuzzed_ast() {
    let mut rng = Rng(0xdead_beef_0bad_f00d);
    for _ in 0..200 {
        let mut document = parse(CORPUS_3).unwrap();
        for _ in 0..3 {
            mutate(&mut document, &mut rng);
        }
        let text = print_document(&document);
        assert!(text.len() < 1_000_000);
    }
}

/// Corrupts a document the way a buggy tool might: dangling ids, cleared lists, extra roots.
fn mutate(document: &mut BsnDocument, rng: &mut Rng) {
    let value_bound = document.values.len() as u64 + 4;
    let node_bound = document.nodes.len() as u64 + 4;
    match rng.next() % 4 {
        0 => {
            let index = rng.index(document.nodes.len());
            if let Some(node) = document.nodes.get_mut(index)
                && let BsnNodeKind::Patch { value, .. } = &mut node.kind
            {
                *value = BsnValueId((rng.next() % value_bound) as u32);
            }
        }
        1 => {
            let index = rng.index(document.values.len());
            let replacement = BsnValueId((rng.next() % value_bound) as u32);
            let choice = rng.next();
            if let Some(node) = document.values.get_mut(index) {
                match &mut node.value {
                    BsnValue::Tuple(items)
                    | BsnValue::List(items)
                    | BsnValue::NamedTuple(_, items)
                        if !items.is_empty() =>
                    {
                        let slot = (choice % items.len() as u64) as usize;
                        items[slot] = replacement;
                    }
                    BsnValue::Struct(_, fields) if !fields.is_empty() => {
                        let slot = (choice % fields.len() as u64) as usize;
                        fields[slot].1 = replacement;
                    }
                    _ => {}
                }
            }
        }
        2 => {
            let index = rng.index(document.nodes.len());
            if let Some(node) = document.nodes.get_mut(index) {
                match &mut node.kind {
                    BsnNodeKind::Entity {
                        patches, relations, ..
                    } => {
                        patches.clear();
                        relations.clear();
                    }
                    BsnNodeKind::Relation { entities, .. } => entities.clear(),
                    BsnNodeKind::Patch { .. } => {}
                }
            }
        }
        _ => document.push_root(BsnNodeId((rng.next() % node_bound) as u32)),
    }
}

#[test]
fn walk_depth_guard_is_documented() {
    // The printer and the structural comparison both stop at this depth, which must be
    // greater than the parser's own limit so that every parseable document prints.
    const { assert!(MAX_WALK_DEPTH > MAX_NESTING_DEPTH) };
}

/// §8.4 — the edge-case table, each row with its defined behavior.
#[test]
fn parse_edge_cases_table() {
    // `()` is a complete entity with nothing in it.
    let document = parse("()").unwrap();
    assert_eq!(document.roots.len(), 1);
    let BsnNodeKind::Entity {
        name,
        base,
        patches,
        relations,
        ..
    } = &document.node(BsnNodeId(0)).unwrap().kind
    else {
        panic!("not an entity");
    };
    assert!(name.is_none() && base.is_none() && patches.is_empty() && relations.is_empty());
    assert_eq!(print_document(&document), "()\n");

    // An empty relation body, and empty struct/tuple bodies.
    assert!(parse("A\nChildren []").is_ok());
    assert!(parse("A {}").is_ok());
    assert!(parse("A()").is_ok());

    // Trailing commas at every list level.
    assert!(parse("A(1, 2,)").is_ok());
    assert!(parse("A { x: [1, 2,], }").is_ok());
    assert!(parse("A,\nB,").is_ok());
    assert!(parse("A\nChildren [ B, ]").is_ok());

    // A comma with nothing before it is not an entity.
    assert!(matches!(
        err(", A").kind,
        BsnParseErrorKind::UnexpectedToken { .. }
    ));

    // Unpaired closing delimiters.
    assert!(matches!(
        err("A { x: 1 } }").kind,
        BsnParseErrorKind::UnexpectedToken { .. }
    ));
    assert!(matches!(
        err("A(1))").kind,
        BsnParseErrorKind::UnexpectedToken { .. }
    ));

    // Float overflow saturates, following Rust's `FromStr`.
    let document = parse("A { x: 1e400 }").unwrap();
    assert!(matches!(
        document.values[1].value,
        BsnValue::Float(f) if f == f64::INFINITY
    ));

    // CRLF line endings are whitespace, and spans stay byte-accurate.
    let document = parse("A\r\nB\r\n").unwrap();
    assert_eq!(document.nodes.len(), 3);
    assert_eq!(document.node(BsnNodeId(2)).unwrap().span, Span::new(3, 4));

    // A leading BOM is skipped.
    assert!(parse("\u{FEFF}A").is_ok());

    // Numeric field names are not supported.
    assert!(matches!(
        err("A { 0: 1.0 }").kind,
        BsnParseErrorKind::UnexpectedToken { .. }
    ));

    // Grouping parentheses are not one-element tuples.
    let document = parse("A((1))").unwrap();
    assert!(matches!(document.values[1].value, BsnValue::Int(1)));
    let document = parse("A((1,))").unwrap();
    assert!(matches!(document.values[1].value, BsnValue::Tuple(_)));
    assert_eq!(print_document(&document), "A((1,))\n");
}

// --- Mutation-audit tests -----------------------------------------------------------------
//
// Each test below kills mutants that survived the cargo-mutants audit: code that was
// *executed* by the suite but whose result was never *asserted* strongly enough to notice a
// wrong answer.

/// Finds the first `Patch` node of a parsed document (node 0 is always the root entity).
fn first_patch(document: &BsnDocument) -> (&BsnPath, &BsnPatchPrefix) {
    document
        .nodes
        .iter()
        .find_map(|node| match &node.kind {
            BsnNodeKind::Patch { symbol, prefix, .. } => Some((symbol, prefix)),
            _ => None,
        })
        .expect("document should contain a patch")
}

/// `true`/`false` must parse as boolean values, not as (rejected lowercase) paths.
#[test]
fn bool_literals_parse_as_bools() {
    let document = parse("A(true, false)").unwrap();
    assert!(matches!(document.values[1].value, BsnValue::Bool(true)));
    assert!(matches!(document.values[2].value, BsnValue::Bool(false)));
}

/// `BsnPath` accessors, asserted directly (their in-crate uses never checked the results).
#[test]
fn path_accessors_report_exact_results() {
    let document = parse("a::b::C { x: 1 }").unwrap();
    let (symbol, prefix) = first_patch(&document);
    assert_eq!(symbol.to_type_path(), "a::b::C");
    assert_eq!(symbol.parent_type_path().as_deref(), Some("a::b"));
    assert!(!symbol.is_single_segment());
    assert!(!prefix.is_template());

    let single = parse("C").unwrap();
    let (s, _) = first_patch(&single);
    assert_eq!(s.parent_type_path(), None);
    assert!(s.is_single_segment());

    let two = parse("a::B").unwrap();
    let (s2, _) = first_patch(&two);
    assert_eq!(s2.parent_type_path().as_deref(), Some("a"));

    let template = parse("~T(1)").unwrap();
    let (_, p) = first_patch(&template);
    assert!(p.is_template());
}

/// Every clause of `BsnPath::structural_eq` is load-bearing: paths differing in exactly one
/// dimension are unequal, and spans are ignored.
#[test]
fn path_structural_eq_distinguishes_each_clause() {
    let path = |source: &str| -> BsnPath {
        let document = parse(source).unwrap();
        first_patch(&document).0.clone()
    };
    let base = path("a::B<C>");
    assert!(base.structural_eq(&path("a::B<C>")));
    assert!(base.structural_eq(&path("  a::B<C>"))); // span differs, still equal
    assert!(!base.structural_eq(&path("a::B<C, D>"))); // generics len
    assert!(!base.structural_eq(&path("a::B<D>"))); // generic ident
    assert!(!base.structural_eq(&path("a::X<C>"))); // ident
    assert!(!base.structural_eq(&path("B<C>"))); // segment count
}

/// The rendered `expected …` suffix, exact text for every list shape (also pins
/// `token_desc`'s human-readable names).
#[test]
fn expected_suffix_renders_exact_text() {
    let none = parse("A { x: }").unwrap_err();
    let one_or_more = parse("A {").unwrap_err();
    assert!(!one_or_more.expected.is_empty());
    assert!(one_or_more.expected_suffix().starts_with(" expected "));
    // Exact suffix for a known two-alternative site: a struct body after a field value
    // (`5` is a valid token in an invalid position, so the parser reports alternatives).
    let error = parse("A { x: 1 5").unwrap_err();
    assert_eq!(error.expected_suffix(), " expected `,` or `}`");
    let _ = none;
}

/// `Span::is_none` is a real predicate, not a constant.
#[test]
fn span_is_none_is_exact() {
    assert!(Span::NONE.is_none());
    assert!(!Span::new(0, 1).is_none());
}

/// Unicode escape validation boundaries: the largest scalar passes, one past it and
/// seven-digit forms fail, and surrogate-range values fail via `char::from_u32`.
#[test]
fn unicode_escape_boundaries() {
    assert!(parse(r#"A("\u{10FFFF}")"#).is_ok());
    assert!(parse(r#"A("\u{110000}")"#).is_err());
    assert!(parse(r#"A("\u{0010FFFF}")"#).is_err()); // 8 digits
    assert!(parse(r#"A("\u{D800}")"#).is_err()); // surrogate
    assert!(parse(r#"A("\x7F")"#).is_ok());
    assert!(parse(r#"A("\x80")"#).is_err());
    // The decoded value must be the right character, not merely accepted.
    let document = parse(r#"A("\u{1F600}")"#).unwrap();
    let BsnValue::String(text) = &document.values[1].value else {
        panic!("expected a string");
    };
    assert_eq!(text, "\u{1F600}");
}

/// A multi-segment path value's span covers the whole path (the span-merge arm in
/// `parse_path_inner`).
#[test]
fn path_value_span_covers_all_segments() {
    let source = "A { x: some::path::C }";
    let document = parse(source).unwrap();
    let BsnValue::Path(path) = &document.values[1].value else {
        panic!("expected a path value");
    };
    assert_eq!(path.span.text(source), "some::path::C");
}

/// Angle-bracket depth in the grouping-vs-tuple scan: a comma inside generics does not make
/// a tuple; a comma after a generic path does; nested closers unwind correctly.
#[test]
fn grouping_scan_tracks_generic_and_bracket_depth() {
    let tuple = parse("A { x: (p::Q<R, S>, 1) }").unwrap();
    let BsnValue::Tuple(items) = &tuple.values[1].value else {
        panic!("expected a tuple");
    };
    assert_eq!(items.len(), 2);

    let grouped = parse("A { x: (p::Q<R, S>) }").unwrap();
    assert!(
        matches!(&grouped.values[1].value, BsnValue::Path(_)),
        "grouping parens around a generic path stay a grouping"
    );

    let nested = parse("A { x: ((1, 2)) }").unwrap();
    let BsnValue::Tuple(inner) = &nested.values[1].value else {
        panic!("nested closers must unwind: the outer parens are a grouping");
    };
    assert_eq!(inner.len(), 2);

    // Closers must unwind depth during the scan: the comma between the two groups is
    // top-level, so the outer parens are a 2-tuple of groupings.
    let two_groups = parse("A { x: ((1), (2)) }").unwrap();
    let BsnValue::Tuple(outer) = &two_groups.values[1].value else {
        panic!("expected a 2-tuple of grouped values");
    };
    assert_eq!(outer.len(), 2);
}

/// `token_desc` supplies the human-readable name in "unexpected …" diagnostics.
#[test]
fn unexpected_token_names_the_token() {
    let error = parse("A { x: 1 5").unwrap_err();
    assert_eq!(error.to_string(), "unexpected number");
    let error = parse("A { x: ! }").unwrap_err();
    assert!(!error.to_string().trim_end().ends_with("unexpected"));
}
