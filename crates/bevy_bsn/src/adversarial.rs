//! Adversarial regression suite for the lexer, parser and printer.
//!
//! Every test in the first section comes from the adversarial correctness review of this crate
//! and pins a defect that review found: each one failed before the fix and asserts the fixed
//! behavior now, so a regression shows up here first.
//!
//! The rest of the module is a set of bounded, deterministic property loops — seed mutation,
//! fragment soup, a grammar-directed document generator, and the shared [`exercise`] invariant
//! battery (span validity, pre-order ids, print/parse round trip, printer idempotence). They
//! run on every platform in a couple of seconds.

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use crate::{
    parse, print_document, BsnDocument, BsnNodeKind, BsnPatchPrefix, BsnPath, BsnValue, BsnValueId,
    Lexer, Span, TokenKind,
};

fn lex_kinds(source: &str) -> Vec<TokenKind> {
    Lexer::tokenize(source)
        .into_iter()
        .map(|token| token.kind)
        .collect()
}

/// Wraps `value` as the sole item of an `A( … )` patch on a single root entity.
fn wrap(value: BsnValue) -> BsnDocument {
    let mut document = BsnDocument::new();
    let inner = document.push_value(value);
    wrap_id(document, inner)
}

fn wrap_id(mut document: BsnDocument, inner: BsnValueId) -> BsnDocument {
    let body = document.push_value(BsnValue::NamedTuple(
        BsnPath::from_segments(["A"]),
        vec![inner],
    ));
    let patch = document.push_node(BsnNodeKind::Patch {
        symbol: BsnPath::from_segments(["A"]),
        prefix: BsnPatchPrefix::FromTemplate,
        value: body,
    });
    let entity = document.push_node(BsnNodeKind::Entity {
        name: None,
        name_span: None,
        base: None,
        base_span: None,
        patches: vec![patch],
        relations: vec![],
    });
    document.push_root(entity);
    document
}

// ---------------------------------------------------------------------------------------
// Fixed defects, each pinned by the test that found it
// ---------------------------------------------------------------------------------------

/// `( v )` is a grouping per SPEC-3 §5.3. A path with generic arguments used to make
/// `paren_has_top_level_comma` see the generic argument's comma at "top level", silently
/// turning the value into a one-element tuple; the comma scan now tracks angle-bracket depth.
#[test]
fn grouping_parens_around_generic_paths_stay_groupings() {
    let grouped = parse("A { x: (my_game::Pair<f32, f32>) }").unwrap();
    let bare = parse("A { x: my_game::Pair<f32, f32> }").unwrap();
    assert!(
        grouped.structural_eq(&bare),
        "`(P<a, b>)` must be a grouping, not a 1-tuple:\n{}",
        grouped.debug_tree()
    );
}

/// A builder-constructed `Int(i128::MIN)` used to print text that did not parse, violating the
/// §7.9 round-trip property.
#[test]
fn i128_min_round_trips_through_the_printer() {
    let document = wrap(BsnValue::Int(i128::MIN));
    let text = print_document(&document);
    let reparsed = parse(&text).expect("printed text must re-parse");
    assert!(document.structural_eq(&reparsed));
}

/// The same value used to be unwritable in source at all: the negation path now decodes the
/// magnitude as `u128`, so `2^127` is reachable. Both the decimal and the radix spellings must
/// work, and one past it must still be rejected.
#[test]
fn i128_min_literal_is_accepted_in_source() {
    for source in [
        "A(-170141183460469231731687303715884105728)",
        "A(-0x80000000000000000000000000000000)",
        "A(-0o2000000000000000000000000000000000000000000)",
    ] {
        let document = parse(source).expect("i128::MIN is a valid i128 literal");
        // `values[0]` is the `A(…)` named tuple; the literal is its only item.
        assert!(
            matches!(document.values[1].value, BsnValue::Int(i128::MIN)),
            "{source}:\n{}",
            document.debug_tree()
        );
    }
    // One past `i128::MIN` is still out of range.
    assert!(matches!(
        parse("A(-170141183460469231731687303715884105729)")
            .unwrap_err()
            .kind,
        crate::BsnParseErrorKind::NumberOutOfRange
    ));
    // And the unnegated magnitude remains out of range.
    assert!(matches!(
        parse("A(170141183460469231731687303715884105728)")
            .unwrap_err()
            .kind,
        crate::BsnParseErrorKind::NumberOutOfRange
    ));
}

/// An empty `Tuple` prints as `()` and re-parses as `Unit`, which used to break the round trip.
///
/// The distinction is unrepresentable in text, so `structural_eq` treats `Unit` and an empty
/// `Tuple` as equal (in both directions) and the round trip holds. The printer is unchanged —
/// both still print as `()`.
#[test]
fn empty_tuple_and_unit_are_round_trip_equal() {
    let document = wrap(BsnValue::Tuple(vec![]));
    let text = print_document(&document);
    assert_eq!(text, "A(())\n");
    let reparsed = parse(&text).unwrap();
    assert!(
        document.structural_eq(&reparsed),
        "empty tuple printed as {text:?}"
    );
    // The equivalence holds in the other direction too, and only for the *empty* tuple.
    let unit = wrap(BsnValue::Unit);
    assert!(unit.structural_eq(&document));
    assert!(document.structural_eq(&unit));
    let inner = {
        let mut document = BsnDocument::new();
        let item = document.push_value(BsnValue::Int(1));
        let tuple = document.push_value(BsnValue::Tuple(vec![item]));
        wrap_id(document, tuple)
    };
    assert!(!unit.structural_eq(&inner));
    assert!(!inner.structural_eq(&unit));
}

/// A negative NaN used to print as `NaN` and re-parse with a different bit pattern, which
/// `structural_eq` (documented to compare floats by `to_bits`) rejects.
#[test]
fn negative_nan_round_trips() {
    let document = wrap(BsnValue::Float(-f64::NAN));
    let text = print_document(&document);
    let reparsed = parse(&text).unwrap();
    assert!(
        document.structural_eq(&reparsed),
        "-NaN printed as {text:?}"
    );
}

/// A binary/octal literal with an out-of-radix digit used to be split into two integer tokens
/// and silently accepted. The out-of-radix run is now folded into a single invalid-number
/// token, so the literal is diagnosed as one bad literal rather than accepted as two good ones.
#[test]
fn out_of_radix_digit_is_one_invalid_number_token() {
    let invalid = vec![
        TokenKind::Error(crate::LexError::InvalidNumber),
        TokenKind::Eof,
    ];
    assert_eq!(lex_kinds("0b12"), invalid);
    assert_eq!(lex_kinds("0o19"), invalid);
    assert_eq!(lex_kinds("0b1_2"), invalid);
    assert_eq!(lex_kinds("0b102030"), invalid);
    for source in ["A(0b12)", "A(0o19)"] {
        let error = parse(source).unwrap_err();
        assert!(
            matches!(error.kind, crate::BsnParseErrorKind::InvalidNumber),
            "{source} -> {error}"
        );
    }
    // Legal radix literals and alphabetic suffixes are untouched.
    assert_eq!(lex_kinds("0b1010"), vec![TokenKind::Int, TokenKind::Eof]);
    assert_eq!(lex_kinds("0o17"), vec![TokenKind::Int, TokenKind::Eof]);
    assert_eq!(lex_kinds("0xFF"), vec![TokenKind::Int, TokenKind::Eof]);
    assert_eq!(
        lex_kinds("0x1g"),
        vec![
            TokenKind::Error(crate::LexError::NumericSuffix),
            TokenKind::Eof
        ]
    );
}

// ---------------------------------------------------------------------------------------
// Invariant fuzzing
// ---------------------------------------------------------------------------------------

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

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.index(items.len())]
    }
}

const SEEDS: [&str; 10] = [
    "#Root\nA { x: 1.0, y: -2 }\nChildren [ B, (C #Kid D(1, \"s\")) ]",
    ":\"base.bsn\"\n@W { f: [1, 2, (3, 4)] }\n~T<a::B>(1.0)",
    "A((1))\nB((1,))\nC(())",
    "A { x: \"\\u{1f600}\\n\\t\\\\\\\"\" }",
    "A(0x10, 0o7, 0b1, 1_000, 1e5, -inf, inf, NaN, true, false, #Ref)",
    "a::b::C::D\nE [ F [ G [ H ] ] ]",
    "A(1.0)/* c */ // l\nB",
    "()",
    "A,\nB,\nC",
    "Ä { é: 1 }",
];

/// Token-ish fragments used to synthesize random inputs.
const FRAGMENTS: [&str; 40] = [
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    ",",
    ":",
    "::",
    "#",
    "@",
    "~",
    "-",
    "<",
    ">",
    "A",
    "b",
    "a::B",
    "C<d>",
    "1",
    "1.0",
    "0x1f",
    "1e9",
    "\"s\"",
    "\"\\n\"",
    "r#\"raw\"#",
    "true",
    "false",
    "inf",
    "NaN",
    " ",
    "\n",
    "\t",
    "\r\n",
    "//x\n",
    "/*y*/",
    "é",
    "\u{1f600}",
    "_",
    ".",
];

/// Every span the document exposes must be a valid, non-inverted slice of the source and
/// must land on `char` boundaries.
fn check_spans(source: &str, document: &BsnDocument) {
    let check = |span: Span, what: &str| {
        assert!(
            span.start <= span.end,
            "inverted span {span:?} ({what}) in {source:?}"
        );
        assert!(
            span.end as usize <= source.len(),
            "out-of-bounds span {span:?} ({what}) in {source:?}"
        );
        assert!(
            source.is_char_boundary(span.start as usize)
                && source.is_char_boundary(span.end as usize),
            "span {span:?} ({what}) is not on char boundaries in {source:?}"
        );
    };
    for node in &document.nodes {
        check(node.span, "node");
        if let BsnNodeKind::Entity {
            name_span,
            base_span,
            ..
        } = &node.kind
        {
            if let Some(span) = name_span {
                check(*span, "name");
            }
            if let Some(span) = base_span {
                check(*span, "base");
            }
        }
    }
    for value in &document.values {
        check(value.span, "value");
    }
}

fn check_error_span(source: &str, error: &crate::BsnParseError) {
    let span = error.span;
    assert!(span.start <= span.end, "inverted error span in {source:?}");
    assert!(
        span.end as usize <= source.len(),
        "out-of-bounds error span {span:?} in {source:?}"
    );
    assert!(
        source.is_char_boundary(span.start as usize) && source.is_char_boundary(span.end as usize),
        "error span {span:?} not on char boundaries in {source:?}"
    );
    let (line, column) = span.line_col(source);
    assert!(line >= 1 && column >= 1);
    // Must not panic.
    let _ = error.render(source, Some("fuzz.bsn"));
}

/// The core property battery for one input.
fn exercise(source: &str) {
    match parse(source) {
        Err(error) => check_error_span(source, &error),
        Ok(document) => {
            check_spans(source, &document);
            // §7.5 patch/value invariant.
            for node in &document.nodes {
                if let BsnNodeKind::Patch { symbol, value, .. } = &node.kind {
                    let value = document.value(*value).expect("patch value must exist");
                    let path = match &value.value {
                        BsnValue::Path(path)
                        | BsnValue::Struct(path, _)
                        | BsnValue::NamedTuple(path, _) => path,
                        other => panic!("patch value is {other:?} in {source:?}"),
                    };
                    assert!(path.structural_eq(symbol), "{source:?}");
                }
            }
            // Ids are pre-order indices into their arena.
            for (index, node) in document.nodes.iter().enumerate() {
                assert_eq!(node.id.0 as usize, index);
            }
            for (index, value) in document.values.iter().enumerate() {
                assert_eq!(value.id.0 as usize, index);
            }
            // Re-parsing identical text yields identical ids and structure.
            let again = parse(source).expect("re-parse of the same text must succeed");
            assert_eq!(document.debug_tree(), again.debug_tree(), "{source:?}");

            // Round trip: print → parse → structurally equal, and printing is a fixed point.
            let printed = print_document(&document);
            let reparsed = match parse(&printed) {
                Ok(reparsed) => reparsed,
                Err(error) => panic!(
                    "printed text does not re-parse\n source: {source:?}\nprinted: {printed:?}\n  error: {error}"
                ),
            };
            assert!(
                document.structural_eq(&reparsed),
                "round trip changed the document\n source: {source:?}\nprinted: {printed:?}\n   from: {}\n     to: {}",
                document.debug_tree(),
                reparsed.debug_tree()
            );
            let printed_again = print_document(&reparsed);
            assert_eq!(
                printed, printed_again,
                "printer is not idempotent for {source:?}"
            );
        }
    }
}

#[test]
fn fuzz_mutations_of_the_seed_corpus() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for _ in 0..30_000 {
        let seed: Vec<char> = rng.pick(&SEEDS).chars().collect();
        let mut text: String = String::new();
        let edits = 1 + rng.index(4);
        let mut chars = seed;
        for _ in 0..edits {
            if chars.is_empty() {
                break;
            }
            let at = rng.index(chars.len());
            match rng.next() % 3 {
                0 => {
                    chars.remove(at);
                }
                1 => {
                    let fragment = rng.pick(&FRAGMENTS);
                    for (offset, ch) in fragment.chars().enumerate() {
                        chars.insert(at + offset, ch);
                    }
                }
                _ => {
                    let fragment = rng.pick(&FRAGMENTS);
                    if let Some(ch) = fragment.chars().next() {
                        chars[at] = ch;
                    }
                }
            }
        }
        text.extend(chars);
        exercise(&text);
    }
}

#[test]
fn fuzz_random_fragment_soup() {
    let mut rng = Rng(0xfeed_face_dead_beef);
    for _ in 0..30_000 {
        let len = 1 + rng.index(24);
        let mut text = String::new();
        for _ in 0..len {
            text.push_str(FRAGMENTS[rng.index(FRAGMENTS.len())]);
        }
        exercise(&text);
    }
}

#[test]
fn fuzz_truncations_and_prefixes() {
    for seed in SEEDS {
        for (index, _) in seed.char_indices() {
            exercise(&seed[..index]);
        }
        exercise(seed);
    }
}

// ---------------------------------------------------------------------------------------
// Targeted torture
// ---------------------------------------------------------------------------------------

#[test]
fn deep_but_legal_nesting_does_not_overflow() {
    // One below the limit in every recursive production.
    let depth = (crate::MAX_NESTING_DEPTH - 2) as usize;

    let list = format!("A({}1{})", "[".repeat(depth), "]".repeat(depth));
    exercise(&list);

    let group = format!("A({}1{})", "(".repeat(depth), ")".repeat(depth));
    let _ = parse(&group).unwrap();

    let relations = format!("A\n{}B{}", "R [ ".repeat(depth - 1), " ]".repeat(depth - 1));
    exercise(&relations);

    let generics = format!("A<{}B{}>", "C<".repeat(depth - 2), ">".repeat(depth - 2));
    let _ = parse(&generics);
}

#[test]
fn over_limit_nesting_is_an_error_not_a_crash() {
    for count in [200usize, 1000, 10_000] {
        let source = format!("A({}1{})", "[".repeat(count), "]".repeat(count));
        assert!(matches!(
            parse(&source).unwrap_err().kind,
            crate::BsnParseErrorKind::NestingTooDeep
        ));
        let source = format!("A({}", "[".repeat(count));
        let _ = parse(&source);
        let source = format!("{}A", "R [ ".repeat(count));
        assert!(parse(&source).is_err());
    }
}

#[test]
fn float_and_int_literal_edges_round_trip() {
    let cases = [
        "1e308",
        "1e-308",
        "5e-324",
        "1e400",
        "-1e400",
        "0.0",
        "-0.0",
        "1.",
        "1e16",
        "1e15",
        "123456789012345678901234567890.5",
        "0.00001",
    ];
    for case in cases {
        let source = format!("A({case})");
        exercise(&source);
    }
    let ints = [
        "170141183460469231731687303715884105727",
        "0",
        "-0",
        "0x7fffffffffffffffffffffffffffffff",
        "1_0_0",
        "0b1111",
    ];
    for case in ints {
        let source = format!("A({case})");
        exercise(&source);
    }
}

#[test]
fn string_escape_edges_round_trip() {
    let cases = [
        r#""\u{0}""#,
        r#""\x7f""#,
        r#""\0\n\r\t\\\"""#,
        r#""\u{10ffff}""#,
        "\"\u{2028}\u{feff}\u{0}\"",
        "r\"\\n\"",
        "r##\"a\"#b\"##",
        "\"\r\n\"",
    ];
    for case in cases {
        let source = format!("A({case})");
        exercise(&source);
    }
}

#[test]
fn multibyte_line_col_is_char_accurate() {
    let source = "A {\n x: \"\u{1f600}\u{1f600}\" y }\n";
    let error = parse(source).unwrap_err();
    let (line, column) = error.span.line_col(source);
    assert_eq!((line, column), (2, 10), "{}", error.render(source, None));
}

// ---------------------------------------------------------------------------------------
// Grammar-directed generator: high-yield valid documents for the round-trip property
// ---------------------------------------------------------------------------------------

fn gen_path(rng: &mut Rng) -> String {
    let heads = ["Aa", "Bb", "Cc0", "mm::Dd", "a::b::Ee::Ff", "Gg"];
    let mut path = String::from(heads[rng.index(heads.len())]);
    if rng.next() % 5 == 0 {
        let count = 1 + rng.index(2);
        path.push('<');
        for index in 0..count {
            if index > 0 {
                path.push_str(", ");
            }
            path.push_str(["Hh", "ii::Jj", "Kk<Ll>", "f32"][rng.index(4)]);
        }
        path.push('>');
    }
    path
}

fn gen_value(rng: &mut Rng, depth: u32) -> String {
    let leaf = depth >= 4;
    let choice = rng.next() % if leaf { 9 } else { 16 };
    match choice {
        0 => format!("{}", rng.next() as i64),
        1 => format!("-{}", rng.next() % 1000),
        2 => ["1.0", "-0.0", "1e9", "3.5e-7", "0.5", "1.", "1e400"][rng.index(7)].to_string(),
        3 => [
            "\"\"",
            "\"a\\nb\"",
            "\"\u{1f600}\"",
            "r#\"raw \" x\"#",
            "\"\\u{7}\"",
            "\"0123456789012345678901234567890123456789\"",
            "\"012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789\"",
        ][rng.index(7)]
        .to_string(),
        4 => ["true", "false"][rng.index(2)].to_string(),
        5 => ["inf", "-inf", "NaN", "nan"][rng.index(4)].to_string(),
        6 => "()".to_string(),
        7 => "#Ref1".to_string(),
        8 => gen_path(rng),
        9 => {
            let count = rng.index(4);
            let items: Vec<String> = (0..count).map(|_| gen_value(rng, depth + 1)).collect();
            format!(
                "[{}{}]",
                items.join(", "),
                if count > 0 && rng.next() % 4 == 0 {
                    ","
                } else {
                    ""
                }
            )
        }
        10 => {
            // 1-tuple, which must keep its trailing comma.
            format!("({},)", gen_value(rng, depth + 1))
        }
        11 => {
            let count = 2 + rng.index(3);
            let items: Vec<String> = (0..count).map(|_| gen_value(rng, depth + 1)).collect();
            format!("({})", items.join(", "))
        }
        12 => {
            // Grouping parentheses.
            format!("({})", gen_value(rng, depth + 1))
        }
        13 => {
            let count = rng.index(4);
            let fields: Vec<String> = (0..count)
                .map(|index| format!("f{index}: {}", gen_value(rng, depth + 1)))
                .collect();
            format!("{} {{ {} }}", gen_path(rng), fields.join(", "))
        }
        _ => {
            let count = rng.index(4);
            let items: Vec<String> = (0..count).map(|_| gen_value(rng, depth + 1)).collect();
            format!("{}({})", gen_path(rng), items.join(", "))
        }
    }
}

fn gen_entity(rng: &mut Rng, depth: u32, parenthesized: bool) -> String {
    let mut entries: Vec<String> = Vec::new();
    if rng.next() % 6 == 0 {
        entries.push("\"base.bsn\"".to_string());
    }
    let mut body = String::new();
    if let Some(base) = entries.first() {
        body.push_str(&format!(":{base} "));
    }
    if rng.next() % 3 == 0 {
        body.push_str("#N1 ");
    }
    let count = rng.index(3) + usize::from(body.is_empty());
    for _ in 0..count {
        match rng.next() % 6 {
            0 if depth < 3 => {
                let children = rng.index(3);
                let mut list: Vec<String> = Vec::new();
                for _ in 0..children {
                    let paren = rng.next() % 2 == 0;
                    list.push(gen_entity(rng, depth + 1, paren));
                }
                body.push_str(&format!("{} [ {} ] ", gen_path(rng), list.join(", ")));
            }
            1 => body.push_str(&format!("~{} ", gen_path(rng))),
            2 => body.push_str(&format!("@{} {{ f0: 1 }} ", gen_path(rng))),
            3 => body.push_str(&format!("{} ", gen_path(rng))),
            4 => body.push_str(&format!("{}({}) ", gen_path(rng), gen_value(rng, 2))),
            _ => body.push_str(&format!(
                "{} {{ f0: {} }} ",
                gen_path(rng),
                gen_value(rng, 2)
            )),
        }
    }
    if body.trim().is_empty() {
        return "()".to_string();
    }
    if parenthesized {
        format!("({body})")
    } else {
        body
    }
}

#[test]
fn fuzz_generated_valid_documents() {
    let mut rng = Rng(0x0bad_c0de_1337_4242);
    let mut ok = 0usize;
    for _ in 0..20_000 {
        let roots = 1 + rng.index(3);
        let mut source = String::new();
        for index in 0..roots {
            if index > 0 {
                source.push_str(",\n");
            }
            let paren = roots > 1 || rng.next() % 2 == 0;
            source.push_str(&gen_entity(&mut rng, 0, paren));
        }
        if parse(&source).is_ok() {
            ok += 1;
        }
        exercise(&source);
    }
    assert!(
        ok > 15_000,
        "generator produced too few valid documents: {ok}/20000 parsed"
    );
}

// ---------------------------------------------------------------------------------------
// Pre-order id contract, path inversion, public decode API, print options
// ---------------------------------------------------------------------------------------

/// Walks the document in source order and returns the node ids in visit order.
fn preorder_nodes(document: &BsnDocument) -> Vec<u32> {
    fn walk(document: &BsnDocument, id: crate::BsnNodeId, out: &mut Vec<u32>) {
        let Some(node) = document.node(id) else {
            return;
        };
        out.push(id.0);
        match &node.kind {
            BsnNodeKind::Entity {
                patches, relations, ..
            } => {
                let mut entries: Vec<crate::BsnNodeId> =
                    patches.iter().chain(relations).copied().collect();
                entries.sort_by_key(|id| {
                    let start = document.node(*id).map_or(0, |node| node.span.start);
                    (start, id.0)
                });
                for entry in entries {
                    walk(document, entry, out);
                }
            }
            BsnNodeKind::Relation { entities, .. } => {
                for entity in entities {
                    walk(document, *entity, out);
                }
            }
            BsnNodeKind::Patch { .. } => {}
        }
    }
    let mut out = Vec::new();
    for root in &document.roots {
        walk(document, *root, &mut out);
    }
    out
}

/// Walks the value arena in source order and returns the value ids in visit order.
fn preorder_values(document: &BsnDocument) -> Vec<u32> {
    fn walk_value(document: &BsnDocument, id: BsnValueId, out: &mut Vec<u32>) {
        let Some(node) = document.value(id) else {
            return;
        };
        out.push(id.0);
        match &node.value {
            BsnValue::Tuple(items) | BsnValue::List(items) | BsnValue::NamedTuple(_, items) => {
                for item in items {
                    walk_value(document, *item, out);
                }
            }
            BsnValue::Struct(_, fields) => {
                for (_, item) in fields {
                    walk_value(document, *item, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    for id in preorder_nodes(document) {
        if let Some(node) = document.node(crate::BsnNodeId(id))
            && let BsnNodeKind::Patch { value, .. } = &node.kind
        {
            walk_value(document, *value, &mut out);
        }
    }
    out
}

#[test]
fn ids_are_assigned_in_source_pre_order() {
    let mut rng = Rng(0xabcd_0001_0002_0003);
    for _ in 0..5_000 {
        let roots = 1 + rng.index(3);
        let mut source = String::new();
        for index in 0..roots {
            if index > 0 {
                source.push_str(",\n");
            }
            let paren = roots > 1 || rng.next() % 2 == 0;
            source.push_str(&gen_entity(&mut rng, 0, paren));
        }
        let Ok(document) = parse(&source) else {
            continue;
        };
        let nodes = preorder_nodes(&document);
        let expected: Vec<u32> = (0..document.nodes.len() as u32).collect();
        assert_eq!(nodes, expected, "node ids not pre-order for {source:?}");
        let values = preorder_values(&document);
        let expected: Vec<u32> = (0..document.values.len() as u32).collect();
        assert_eq!(values, expected, "value ids not pre-order for {source:?}");
    }
}

#[test]
fn from_type_path_inverts_to_type_path() {
    let mut rng = Rng(0x5151_2222_3333_4444);
    for _ in 0..3_000 {
        let source = format!("A({})", gen_path(&mut rng));
        let Ok(document) = parse(&source) else {
            continue;
        };
        for value in &document.values {
            if let BsnValue::Path(path)
            | BsnValue::Struct(path, _)
            | BsnValue::NamedTuple(path, _) = &value.value
            {
                let text = path.to_type_path();
                let round = BsnPath::from_type_path(&text)
                    .unwrap_or_else(|| panic!("{text:?} did not re-parse as a path"));
                assert!(
                    path.structural_eq(&round),
                    "{text:?} did not survive from_type_path"
                );
                assert_eq!(round.to_type_path(), text);
            }
        }
    }
}

#[test]
fn public_decoders_never_panic_on_arbitrary_spans() {
    let mut rng = Rng(0x9999_8888_7777_6666);
    let samples = [
        "\"abc\"",
        "r#\"a\"#",
        "0x",
        "1e",
        "\u{1f600}\"\u{1f600}\"",
        "r\"\"",
        "r",
        "",
        "0b1_",
        "\\u{",
        "1.0e-",
        "999999999999999999999999999999999999999999",
    ];
    for _ in 0..20_000 {
        let source = samples[rng.index(samples.len())];
        let start = rng.index(source.len() + 3) as u32;
        let end = rng.index(source.len() + 3) as u32;
        let span = Span::new(start, end);
        let _ = crate::decode_int(source, span);
        let _ = crate::decode_float(source, span);
        let _ = crate::decode_string(source, span);
        let _ = span.text(source);
        let _ = span.line_col(source);
    }
}

#[test]
fn from_type_path_never_panics() {
    let mut rng = Rng(0x1010_2020_3030_4040);
    for _ in 0..20_000 {
        let len = 1 + rng.index(10);
        let mut text = String::new();
        for _ in 0..len {
            text.push_str(FRAGMENTS[rng.index(FRAGMENTS.len())]);
        }
        let _ = BsnPath::from_type_path(&text);
    }
}

#[test]
fn non_default_print_options_still_re_parse() {
    let options = [
        crate::PrintOptions {
            indent: 0,
            max_inline_width: 0,
            trailing_commas: false,
            blank_line_between_roots: false,
        },
        crate::PrintOptions {
            indent: 1,
            max_inline_width: u16::MAX,
            trailing_commas: false,
            blank_line_between_roots: true,
        },
        crate::PrintOptions {
            indent: 8,
            max_inline_width: 1,
            trailing_commas: true,
            blank_line_between_roots: false,
        },
    ];
    let mut rng = Rng(0x7777_1111_2222_3333);
    for _ in 0..2_000 {
        let roots = 1 + rng.index(3);
        let mut source = String::new();
        for index in 0..roots {
            if index > 0 {
                source.push_str(",\n");
            }
            let paren = roots > 1 || rng.next() % 2 == 0;
            source.push_str(&gen_entity(&mut rng, 0, paren));
        }
        let Ok(document) = parse(&source) else {
            continue;
        };
        for option in &options {
            let mut text = String::new();
            crate::write_document_with(&document, &mut text, option).unwrap();
            let reparsed = match parse(&text) {
                Ok(reparsed) => reparsed,
                Err(error) => panic!("{option:?} produced unparsable text {text:?}: {error}"),
            };
            assert!(
                document.structural_eq(&reparsed),
                "{option:?} changed the document\n{text}"
            );
        }
    }
}

/// A shared sub-value (a DAG, which the arena API makes trivial to build) used to make the
/// printer produce output exponential in the number of value nodes. A visit budget now bounds
/// the output and marks where it was cut.
#[test]
fn shared_value_ids_hit_the_print_budget() {
    let mut document = BsnDocument::new();
    let mut current = document.push_value(BsnValue::Int(123_456));
    for _ in 0..20 {
        current = document.push_value(BsnValue::Tuple(vec![current, current]));
    }
    let document = wrap_id(document, current);
    assert!(document.values.len() < 25);
    let text = print_document(&document);
    assert!(
        text.len() < 100_000,
        "{} value nodes produced {} bytes of output",
        document.values.len(),
        text.len()
    );
    // The output is truncated by the visit budget, and says so.
    assert!(
        text.contains("/* <print budget exceeded> */"),
        "expected the budget marker in:\n{text}"
    );
    // Deterministic: printing the same document twice gives the same text.
    assert_eq!(text, print_document(&document));
}

/// Lexer invariants over hostile Unicode: no empty tokens (which would let the parser spin),
/// monotonic non-overlapping spans on `char` boundaries, exactly one trailing `Eof`.
#[test]
fn lexer_invariants_over_hostile_unicode() {
    const CHARS: [char; 40] = [
        '"',
        '\\',
        '\'',
        '`',
        '#',
        '@',
        '~',
        '-',
        '<',
        '>',
        '(',
        ')',
        '{',
        '}',
        '[',
        ']',
        ',',
        ':',
        '.',
        '/',
        '*',
        '!',
        '|',
        '0',
        '9',
        'a',
        'r',
        'e',
        'x',
        'u',
        '_',
        '\n',
        '\r',
        '\t',
        ' ',
        '\u{0}',
        '\u{feff}',
        '\u{1f600}',
        '\u{300}',
        '\u{2028}',
    ];
    let mut rng = Rng(0x2468_1357_9bdf_0246);
    for _ in 0..20_000 {
        let len = 1 + rng.index(30);
        let mut source = String::new();
        for _ in 0..len {
            source.push(CHARS[rng.index(CHARS.len())]);
        }
        let tokens = Lexer::tokenize(&source);
        assert!(!tokens.is_empty());
        let mut previous_end = 0u32;
        for (index, token) in tokens.iter().enumerate() {
            let last = index + 1 == tokens.len();
            assert_eq!(
                token.kind == TokenKind::Eof,
                last,
                "stray Eof in {source:?}: {tokens:?}"
            );
            assert!(
                source.is_char_boundary(token.span.start as usize)
                    && source.is_char_boundary(token.span.end as usize),
                "token {token:?} off a char boundary in {source:?}"
            );
            assert!(
                token.span.end as usize <= source.len(),
                "token {token:?} past the end of {source:?}"
            );
            assert!(
                token.span.start >= previous_end,
                "token {token:?} overlaps the previous one in {source:?}"
            );
            assert!(token.span.start <= token.span.end, "{token:?}");
            if last {
                assert_eq!(
                    token.span,
                    Span::new(source.len() as u32, source.len() as u32),
                    "Eof span for {source:?}"
                );
            } else {
                assert!(
                    token.span.end > token.span.start,
                    "empty token {token:?} in {source:?}"
                );
            }
            previous_end = token.span.end;
        }
        // The parser must not hang or panic on the same input.
        exercise(&source);
    }
}

#[test]
fn comments_in_awkward_places() {
    let cases = [
        "A/*x*/{/*y*/f0:/*z*/1/*w*/}",
        "A // trailing, no newline",
        "/*a/*b*/c*/A",
        "A\n// comment at eof",
        "A(/**/1/**/,/**/2/**/)",
        "/**/",
        "//",
        "A/*",
    ];
    for case in cases {
        exercise(case);
    }
    assert!(parse("A/*x*/{/*y*/f0:/*z*/1/*w*/}").is_ok());
    assert!(parse("/*a/*b*/c*/A").is_ok());
    assert!(parse("/**/").unwrap().roots.is_empty());
    assert!(matches!(
        parse("A/*").unwrap_err().kind,
        crate::BsnParseErrorKind::UnterminatedBlockComment
    ));
}

/// A patch appended to a *parsed* entity has `Span::NONE`, and the span-ordered merge used to
/// print it before every parsed entry — patch order is semantically significant. An entry with
/// no span now sorts after every spanned one, so an append prints last.
#[test]
fn appending_a_patch_to_a_parsed_entity_prints_it_last() {
    let mut document = parse("First\nSecond").unwrap();
    let patch = document.push_patch(
        BsnPatchPrefix::FromTemplate,
        BsnPath::from_segments(["Third"]),
        crate::PatchBody::Unit,
    );
    let BsnNodeKind::Entity { patches, .. } = &mut document.nodes[0].kind else {
        panic!()
    };
    patches.push(patch);
    let text = print_document(&document);
    assert_eq!(text, "First\nSecond\nThird\n", "appended patch moved");
}

#[test]
fn bom_does_not_shift_the_reported_column() {
    let source = "\u{feff}A { x }";
    let error = parse(source).unwrap_err();
    assert_eq!(
        error.span.line_col(source),
        (1, 6),
        "{}",
        error.render(source, None)
    );
}

/// The duplicate-field check used to compare every field against every earlier one, so a wide
/// struct cost O(n²) string comparisons. It is O(n log n) now, which the correctness tests for
/// [`crate::BsnParseErrorKind::DuplicateField`] pin; this one is the performance tripwire.
///
/// The bound is deliberately enormous — a debug build parses this in single-digit milliseconds,
/// while the quadratic version took seconds — so it can only fire on a real complexity
/// regression, never on a slow or loaded machine.
#[test]
fn a_wide_struct_body_does_not_parse_quadratically() {
    const FIELDS: usize = 4000;
    let mut source = String::from("A {");
    for index in 0..FIELDS {
        source.push_str(&format!("f{index}: 1,"));
    }
    source.push('}');

    let start = std::time::Instant::now();
    let document = parse(&source).unwrap();
    let elapsed = start.elapsed();

    // `values[0]` is the `A { … }` struct itself; every field is a value of its own.
    let BsnValue::Struct(_, fields) = &document.values[0].value else {
        panic!("expected a struct value:\n{}", document.debug_tree());
    };
    assert_eq!(fields.len(), FIELDS);
    assert!(
        elapsed < core::time::Duration::from_secs(2),
        "parsing {FIELDS} distinct fields took {elapsed:?}"
    );
}
