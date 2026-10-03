//! Parse errors, the catalogue of "unsupported construct" diagnostics, and rustc-style
//! error rendering.

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt::Write as _;

use crate::lexer::Span;
use crate::parser::MAX_NESTING_DEPTH;

/// The exact diagnostic messages emitted for constructs that exist in the `bsn!` macro but
/// cannot exist in a `.bsn` asset.
///
/// Each message names the construct, says why it cannot work in an asset, and ends with a
/// remedy. Downstream tools can compare a [`BsnParseErrorKind::Unsupported`] payload against
/// these constants instead of matching on message text.
pub mod unsupported {
    /// A Rust expression block, `{ … }`.
    pub const EXPR: &str = "Rust expressions (`{ ... }`) are not supported in `.bsn` assets. Only literal values are allowed here; move the computation into a `bsn!` macro or a scene function.";
    /// A closure, `|x| { … }`.
    pub const CLOSURE: &str = "Closures are not supported in `.bsn` assets. Observers and template functions must be written in Rust.";
    /// An observer entry, `on(…)`.
    pub const OBSERVER: &str = "Observers (`on(...)`) are not supported in `.bsn` assets. Attach them from Rust, e.g. with a `bsn!` scene or an `Observer` entity.";
    /// A scene function or function call in entry position.
    pub const FN: &str = "Scene functions and function calls are not supported in `.bsn` assets. Expected a type path; type names start with an uppercase letter.";
    /// A constructor call, `Type::function(…)`.
    pub const CTOR: &str = "Constructor calls (`Type::function(...)`) are not supported in `.bsn` assets. Write the resulting value out in full instead.";
    /// A Rust constant, such as `PI` or `Type::MAX`.
    pub const CONST: &str = "Constants are not supported in `.bsn` assets. Write the literal value instead (`.bsn` has no access to Rust items).";
    /// Struct field shorthand, `Comp { name }`.
    pub const SHORTHAND: &str = "Field shorthand (`{ name }`) is not supported in `.bsn` assets, because there are no variables to capture. Write `name: <value>` instead.";
    /// A scene component prop, `@prop: …`.
    pub const PROP: &str = "Scene component props (`@prop: ...`) are not supported in `.bsn` assets. Props are evaluated by Rust code when the scene is included and cannot be expressed in an asset.";
    /// A macro invocation, `vec![…]`.
    pub const MACRO: &str = "Macro invocations are not supported in `.bsn` assets. Use a list literal `[ ... ]` instead of `vec![ ... ]`.";
    /// A `use` import.
    pub const USE: &str = "`use` imports are not supported in `.bsn` assets. Write fully-qualified type paths, e.g. `bevy_transform::components::transform::Transform`.";
    /// A character literal, `'a'`.
    pub const CHAR: &str =
        "Character literals are not supported in `.bsn` assets. Use a string literal instead.";
    /// A numeric literal suffix, `1u8` or `1.0f32`.
    pub const SUFFIX: &str = "Numeric literal suffixes are not supported in `.bsn` assets. The field's declared type determines the literal's type.";
    /// A raw identifier, `r#type`.
    pub const RAW_IDENT: &str = "Raw identifiers (`r#name`) are not supported in `.bsn` assets.";
    /// A lowercase path where a value was expected.
    pub const PATH_CASE: &str = "Expected a value. Type paths start with an uppercase letter; this path looks like a function or variable, and `.bsn` assets can only contain literal values.";
}

/// An error produced while parsing a `.bsn` document.
///
/// Parsing is fail-fast: the first problem aborts the parse and is returned. Use
/// [`BsnParseError::render`] to turn the error into a rustc-style, multi-line report.
#[derive(Clone, Debug, thiserror::Error)]
#[error("{kind}")]
pub struct BsnParseError {
    /// Byte range of the offending text.
    pub span: Span,
    /// What went wrong.
    pub kind: BsnParseErrorKind,
    /// Token descriptions that would have been accepted here, e.g. ``["`,`", "`]`"]``.
    ///
    /// Empty for errors where an "expected" list is meaningless.
    pub expected: Vec<&'static str>,
}

impl BsnParseError {
    /// The rendered `expected …` suffix for this error, or an empty string when the parser
    /// recorded no expectations. Begins with a leading space, so it can be appended directly
    /// to the [`Display`](core::fmt::Display) output when composing one-line diagnostics.
    pub fn expected_suffix(&self) -> String {
        render_expected(&self.expected)
    }

    /// Creates an error with no "expected" list.
    pub fn new(span: Span, kind: BsnParseErrorKind) -> Self {
        BsnParseError {
            span,
            kind,
            expected: Vec::new(),
        }
    }

    /// Creates an error with an "expected" list.
    pub fn expected(span: Span, kind: BsnParseErrorKind, expected: &[&'static str]) -> Self {
        BsnParseError {
            span,
            kind,
            expected: expected.to_vec(),
        }
    }

    /// Renders the error in rustc's style, quoting the offending source line.
    ///
    /// `path` is the file name shown in the location line; `<bsn>` is used when it is `None`.
    ///
    /// ```text
    /// error: unexpected `}`
    ///   --> assets/player.bsn:3:22
    ///    |
    ///  3 |     Transform { x: 1.0 }
    ///    |                      ^ expected `,` or `:`
    /// ```
    pub fn render(&self, source: &str, path: Option<&str>) -> String {
        const MAX_LINE: usize = 200;

        let (line, column) = self.span.line_col(source);
        let line_text = source.lines().nth(line as usize - 1).unwrap_or("");
        let mut display_line: String = line_text
            .chars()
            .take(MAX_LINE)
            .map(|c| if c == '\t' { ' ' } else { c })
            .collect();
        if line_text.chars().nth(MAX_LINE).is_some() {
            display_line.push_str("...");
        }

        let number = line.to_string();
        let gutter = " ".repeat(number.len() + 1);

        let mut out = String::new();
        let _ = writeln!(out, "error: {}", self.kind);
        let _ = writeln!(
            out,
            "{gutter}--> {}:{line}:{column}",
            path.unwrap_or("<bsn>")
        );
        let _ = writeln!(out, "{gutter} |");
        let _ = writeln!(out, " {number} | {display_line}");

        let caret_count = {
            let text = self.span.text(source);
            let count = text.chars().take_while(|c| *c != '\n').count();
            count.max(1)
        };
        let pad = " ".repeat((column as usize).saturating_sub(1).min(MAX_LINE));
        let carets = "^".repeat(caret_count.min(MAX_LINE));
        let expected = render_expected(&self.expected);
        let _ = write!(out, "{gutter} | {pad}{carets}{expected}");
        out.push('\n');
        out
    }
}

/// Formats an "expected" list as ` expected a, b, or c`, or `""` when empty.
fn render_expected(expected: &[&'static str]) -> String {
    match expected {
        [] => String::new(),
        [one] => format!(" expected {one}"),
        [one, two] => format!(" expected {one} or {two}"),
        [rest @ .., last] => format!(" expected {}, or {last}", rest.join(", ")),
    }
}

/// The kind of a [`BsnParseError`].
///
/// `#[non_exhaustive]` because new diagnostics may be added in a minor release; consumers
/// should always keep a fallback arm.
#[derive(Clone, Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BsnParseErrorKind {
    /// A token appeared where the grammar does not allow it.
    #[error("unexpected {found}")]
    UnexpectedToken {
        /// A human-readable description of the offending token, e.g. ``"`}`"``.
        found: &'static str,
    },
    /// The document ended in the middle of a construct.
    #[error("unexpected end of file")]
    UnexpectedEof,
    /// A string literal was not closed.
    #[error("unterminated string literal")]
    UnterminatedString,
    /// A `/* … */` comment was not closed.
    #[error("unterminated block comment")]
    UnterminatedBlockComment,
    /// A string literal contains an unknown escape sequence.
    #[error("invalid escape sequence")]
    InvalidEscape,
    /// A numeric literal could not be interpreted.
    #[error("invalid numeric literal")]
    InvalidNumber,
    /// An integer literal does not fit in an `i128`.
    #[error("integer literal out of range for i128")]
    NumberOutOfRange,
    /// A character with no meaning in BSN.
    #[error("unexpected character `{0}`")]
    UnknownCharacter(char),
    /// A `:"…"` scene include appeared after another entry.
    #[error("a `:\"…\"` scene include must be the first entry of an entity")]
    BaseNotFirst,
    /// A `:` include was followed by something other than a string literal.
    #[error(
        "only scene assets can be included with `:`; expected a string literal, e.g. `:\"player.bsn\"`"
    )]
    BaseNotString,
    /// An entity carries more than one `#Name`.
    #[error("duplicate entity name; an entity may have at most one `#Name`")]
    DuplicateName,
    /// A struct body sets the same field twice.
    #[error("duplicate field `{0}`")]
    DuplicateField(String),
    /// `-` was applied to something that is not a number.
    #[error("`-` may only be applied to a number or `inf`")]
    NegOperand,
    /// A path started with `::`.
    #[error("paths may not start with `::`")]
    LeadingPathSeparator,
    /// The document nests deeper than [`MAX_NESTING_DEPTH`].
    #[error("nesting is too deep (limit {MAX_NESTING_DEPTH})")]
    NestingTooDeep,
    /// A `bsn!` construct that cannot exist in an asset. The payload is one of the
    /// [`unsupported`] constants.
    #[error("{0}")]
    Unsupported(&'static str),
    /// The parser reached a state it believes to be impossible. Always a bug in this crate.
    #[error("internal parser error: {0}")]
    Internal(&'static str),
}
