//! The canonical printer: the write side of the format.
//!
//! [`print_document`] renders a [`BsnDocument`] back to `.bsn` text in a fixed, stable
//! style, so tools can emit files that diff cleanly. Printing is *semantics preserving* but
//! not *source preserving*: comments, layout, integer radix and raw-string delimiters are
//! not carried by the AST and therefore do not survive a parse/print cycle.
//!
//! Two AST distinctions likewise have no text to survive in, and
//! [`BsnDocument::structural_eq`] treats each pair as equal so that the round-trip property
//! still holds for documents a builder can construct:
//!
//! - [`BsnValue::Unit`] and an empty [`BsnValue::Tuple`] both print as `()`.
//! - Every `NaN` of a given sign prints as `NaN` or `-NaN`. The *sign* round-trips; a
//!   non-canonical `NaN` *payload* does not, and comes back as the platform's quiet `NaN`.

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt::Write as _;

use crate::ast::{
    merge_entries, BsnDocument, BsnNodeId, BsnNodeKind, BsnValue, BsnValueId, BsnValueNode,
    MAX_WALK_DEPTH,
};

/// Emitted in place of a node or value whose id is not present in the document.
const INVALID: &str = "/* <invalid node id ";
/// Emitted when a (necessarily hand-built) document nests deeper than [`MAX_WALK_DEPTH`].
const TOO_DEEP: &str = "/* <nesting too deep> */";
/// Emitted when a (necessarily hand-built) document exhausts the printer's visit budget.
const OVER_BUDGET: &str = "/* <print budget exceeded> */";

/// Value visits allowed per value node, on top of [`BUDGET_BASE`].
///
/// A legal tree is visited at most once per ancestor — the printer renders each body inline
/// first and re-renders it broken across lines if it does not fit — so `depth + 1` visits per
/// node bounds it, and the parser caps nesting at [`crate::MAX_NESTING_DEPTH`] (128).
const BUDGET_PER_VALUE: usize = 256;
/// Visits allowed regardless of document size, so tiny documents are never constrained.
const BUDGET_BASE: usize = 1024;

/// Formatting knobs for the printer.
///
/// [`Default`] is the canonical style, and is what [`print_document`] uses.
#[derive(Clone, Debug)]
pub struct PrintOptions {
    /// Spaces per indent level. Default 4.
    pub indent: u8,
    /// Soft line-width budget for keeping a struct, tuple or list body on one line.
    /// Default 100.
    pub max_inline_width: u16,
    /// Emit a trailing comma on multi-line bodies. Default `true`.
    pub trailing_commas: bool,
    /// Emit a blank line between top-level roots. Default `true`.
    pub blank_line_between_roots: bool,
}

impl Default for PrintOptions {
    fn default() -> Self {
        PrintOptions {
            indent: 4,
            max_inline_width: 100,
            trailing_commas: true,
            blank_line_between_roots: true,
        }
    }
}

/// Renders `document` as canonical `.bsn` text.
///
/// Never fails and never panics: a malformed document (a dangling id, or a cycle) prints a
/// `/* <invalid node id N> */` marker instead of aborting.
///
/// # Output is bounded
///
/// [`parse`](crate::parse) always produces a tree, but the builder API hands out
/// [`BsnValueId`]s, so a hand-built document may share a value between several parents.
/// Printing such a DAG expands it back into a tree, which is exponential in the number of
/// value nodes. The printer therefore spends from a fixed budget of value visits —
/// `values.len() * 256 + 1024`, computed once — and emits a
/// `/* <print budget exceeded> */` marker instead of descending once it runs out. The budget
/// is generous enough that no legal tree can reach it (a tree costs at most one visit per
/// node per ancestor, and nesting is capped at [`MAX_NESTING_DEPTH`](crate::MAX_NESTING_DEPTH)),
/// and it is consumed in a fixed traversal order, so the output stays deterministic.
pub fn print_document(document: &BsnDocument) -> String {
    let mut out = String::new();
    let _ = write_document(document, &mut out);
    out
}

/// Streams `document` into `out` using the canonical style.
pub fn write_document<W: core::fmt::Write>(
    document: &BsnDocument,
    out: &mut W,
) -> core::fmt::Result {
    write_document_with(document, out, &PrintOptions::default())
}

/// Streams `document` into `out` using explicit [`PrintOptions`].
pub fn write_document_with<W: core::fmt::Write>(
    document: &BsnDocument,
    out: &mut W,
    options: &PrintOptions,
) -> core::fmt::Result {
    let mut printer = Printer {
        document,
        options,
        value_stack: Vec::new(),
        node_stack: Vec::new(),
        budget: document
            .values
            .len()
            .saturating_mul(BUDGET_PER_VALUE)
            .saturating_add(BUDGET_BASE),
    };
    out.write_str(&printer.document_text())
}

/// Escapes `value` as a double-quoted BSN string literal, including the quotes.
///
/// Only `\\`, `\"`, `\n`, `\r`, `\t`, `\0` and other control characters are escaped; all
/// other text, including non-ASCII characters, is emitted verbatim.
pub(crate) fn escape_string(value: &str) -> String {
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

/// Formats a float so that it re-parses as a float: `1.0`, `inf`, `-inf`, `NaN`, `-NaN`.
///
/// The sign of a `NaN` is preserved, but its payload is not: every `NaN` with a given sign
/// prints the same way and re-parses as the platform's canonical quiet `NaN`.
pub(crate) fn format_float(value: f64) -> String {
    if value.is_nan() {
        if value.is_sign_negative() {
            "-NaN".to_string()
        } else {
            "NaN".to_string()
        }
    } else if value == f64::INFINITY {
        "inf".to_string()
    } else if value == f64::NEG_INFINITY {
        "-inf".to_string()
    } else {
        format!("{value:?}")
    }
}

struct Printer<'a> {
    document: &'a BsnDocument,
    options: &'a PrintOptions,
    /// Ids on the current value recursion path, so a cyclic hand-built document terminates.
    value_stack: Vec<BsnValueId>,
    /// Ids on the current entity recursion path, for the same reason.
    node_stack: Vec<BsnNodeId>,
    /// Remaining value visits, so a shared (DAG) value cannot expand exponentially.
    budget: usize,
}

impl Printer<'_> {
    /// Charges one value visit to the budget, returning `false` once it is exhausted.
    fn spend(&mut self) -> bool {
        match self.budget.checked_sub(1) {
            Some(remaining) => {
                self.budget = remaining;
                true
            }
            None => false,
        }
    }

    /// The whole document, ending with exactly one newline when non-empty.
    fn document_text(&mut self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for root in &self.document.roots {
            let mut text = String::new();
            self.entity(&mut text, *root, 0);
            while text.ends_with('\n') {
                text.pop();
            }
            parts.push(text);
        }
        if parts.is_empty() {
            return String::new();
        }
        let separator = if self.options.blank_line_between_roots {
            ",\n\n"
        } else {
            ",\n"
        };
        let mut out = parts.join(separator);
        out.push('\n');
        out
    }

    fn indent(&self, level: usize) -> String {
        " ".repeat(level * self.options.indent as usize)
    }

    /// Appends the lines of an entity, each terminated by `\n`.
    fn entity(&mut self, out: &mut String, id: BsnNodeId, level: usize) {
        if level > MAX_WALK_DEPTH as usize {
            let _ = writeln!(out, "{}{TOO_DEEP}", self.indent(level));
            return;
        }
        let pad = self.indent(level);
        if self.node_stack.contains(&id) {
            let _ = writeln!(out, "{pad}{TOO_DEEP}");
            return;
        }
        let document = self.document;
        let Some(node) = document.node(id) else {
            let _ = writeln!(out, "{pad}{INVALID}{}> */", id.0);
            return;
        };
        let BsnNodeKind::Entity {
            name,
            base,
            patches,
            relations,
            ..
        } = &node.kind
        else {
            let _ = writeln!(out, "{pad}{INVALID}{}> */", id.0);
            return;
        };
        self.node_stack.push(id);
        let start = out.len();
        if let Some(base) = base {
            let _ = writeln!(out, "{pad}:{}", escape_string(base));
        }
        if let Some(name) = name {
            let _ = writeln!(out, "{pad}#{name}");
        }
        for entry in merge_entries(document, patches, relations) {
            match document.node(entry).map(|node| &node.kind) {
                Some(BsnNodeKind::Patch { prefix, value, .. }) => {
                    let (prefix, value) = (*prefix, *value);
                    out.push_str(&pad);
                    out.push_str(prefix.sigil());
                    self.value(out, value, level, 0);
                    out.push('\n');
                }
                Some(BsnNodeKind::Relation {
                    target_symbol,
                    entities,
                }) => {
                    let (path, entities) = (target_symbol.to_type_path(), entities.clone());
                    self.relation(out, &path, &entities, level);
                }
                _ => {
                    let _ = writeln!(out, "{pad}{INVALID}{}> */", entry.0);
                }
            }
        }
        if out.len() == start {
            let _ = writeln!(out, "{pad}()");
        }
        self.node_stack.pop();
    }

    /// Appends a relation entry: its type path, then each child entity one level in.
    fn relation(&mut self, out: &mut String, path: &str, entities: &[BsnNodeId], level: usize) {
        let pad = self.indent(level);
        if entities.is_empty() {
            let _ = writeln!(out, "{pad}{path} []");
            return;
        }
        let _ = writeln!(out, "{pad}{path} [");
        for (index, entity) in entities.iter().enumerate() {
            self.entity(out, *entity, level + 1);
            if index + 1 < entities.len() && out.ends_with('\n') {
                out.pop();
                out.push_str(",\n");
            }
        }
        let _ = writeln!(out, "{pad}]");
    }

    /// Appends a value, continuing the current line. Multi-line bodies indent their contents
    /// relative to `level` and leave the cursor after the closing delimiter.
    fn value(&mut self, out: &mut String, id: BsnValueId, level: usize, depth: u32) {
        if depth > MAX_WALK_DEPTH {
            out.push_str(TOO_DEEP);
            return;
        }
        if !self.spend() {
            out.push_str(OVER_BUDGET);
            return;
        }
        if self.value_stack.contains(&id) {
            out.push_str(TOO_DEEP);
            return;
        }
        let document = self.document;
        let Some(node) = document.value(id) else {
            let _ = write!(out, "{INVALID}{}> */", id.0);
            return;
        };
        let inline = self.inline_value(id, depth);
        let breakable = is_breakable(&node.value);
        if let Some(text) = &inline {
            let width = last_line_width(out) + text.chars().count();
            if !breakable || width <= self.options.max_inline_width as usize {
                out.push_str(text);
                return;
            }
        }
        self.value_stack.push(id);
        match &node.value {
            BsnValue::Struct(path, fields) if !fields.is_empty() => {
                out.push_str(&path.to_type_path());
                out.push_str(" {\n");
                for (index, (name, field)) in fields.iter().enumerate() {
                    let _ = write!(out, "{}{name}: ", self.indent(level + 1));
                    self.value(out, *field, level + 1, depth + 1);
                    self.item_separator(out, index + 1 == fields.len(), false);
                }
                let _ = write!(out, "{}}}", self.indent(level));
            }
            BsnValue::NamedTuple(path, items) if !items.is_empty() => {
                out.push_str(&path.to_type_path());
                self.multiline_items(out, items, level, depth, "(", ")");
            }
            BsnValue::Tuple(items) if !items.is_empty() => {
                self.multiline_items(out, items, level, depth, "(", ")");
            }
            BsnValue::List(items) if !items.is_empty() => {
                self.multiline_items(out, items, level, depth, "[", "]");
            }
            // Every remaining value is a leaf or an empty body, which `inline_value` always
            // renders: the dangling-id, depth and cycle cases were all handled above.
            _ => {
                if let Some(text) = inline {
                    out.push_str(&text);
                }
            }
        }
        self.value_stack.pop();
    }

    fn multiline_items(
        &mut self,
        out: &mut String,
        items: &[BsnValueId],
        level: usize,
        depth: u32,
        open: &str,
        close: &str,
    ) {
        out.push_str(open);
        out.push('\n');
        let force_comma = open == "(" && items.len() == 1;
        for (index, item) in items.iter().enumerate() {
            out.push_str(&self.indent(level + 1));
            self.value(out, *item, level + 1, depth + 1);
            self.item_separator(out, index + 1 == items.len(), force_comma);
        }
        let _ = write!(out, "{}{close}", self.indent(level));
    }

    fn item_separator(&self, out: &mut String, last: bool, force_comma: bool) {
        if !last || self.options.trailing_commas || force_comma {
            out.push(',');
        }
        out.push('\n');
    }

    /// Renders a value on a single line, or `None` if it contains a dangling id.
    fn inline_value(&mut self, id: BsnValueId, depth: u32) -> Option<String> {
        if depth > MAX_WALK_DEPTH || self.value_stack.contains(&id) {
            return None;
        }
        if !self.spend() {
            // Not `None`: falling back to the multi-line path would keep descending, and the
            // whole point of the budget is to stop.
            return Some(OVER_BUDGET.to_string());
        }
        let document = self.document;
        let node = document.value(id)?;
        self.value_stack.push(id);
        let result = self.inline_value_inner(node, depth);
        self.value_stack.pop();
        result
    }

    fn inline_value_inner(&mut self, node: &BsnValueNode, depth: u32) -> Option<String> {
        Some(match &node.value {
            BsnValue::Unit => "()".to_string(),
            BsnValue::Bool(value) => value.to_string(),
            BsnValue::Int(value) => value.to_string(),
            BsnValue::Float(value) => format_float(*value),
            BsnValue::String(value) => escape_string(value),
            BsnValue::EntityRef(name) => format!("#{name}"),
            BsnValue::Path(path) => path.to_type_path(),
            BsnValue::Tuple(values) => match self.inline_items(values, depth)?.as_slice() {
                [] => "()".to_string(),
                [one] => format!("({one},)"),
                parts => format!("({})", parts.join(", ")),
            },
            BsnValue::List(values) => format!("[{}]", self.inline_items(values, depth)?.join(", ")),
            BsnValue::NamedTuple(path, values) => format!(
                "{}({})",
                path.to_type_path(),
                self.inline_items(values, depth)?.join(", ")
            ),
            BsnValue::Struct(path, fields) if fields.is_empty() => {
                format!("{} {{}}", path.to_type_path())
            }
            BsnValue::Struct(path, fields) => {
                let mut parts = Vec::with_capacity(fields.len());
                for (name, field) in fields {
                    parts.push(format!("{name}: {}", self.inline_value(*field, depth + 1)?));
                }
                format!("{} {{ {} }}", path.to_type_path(), parts.join(", "))
            }
        })
    }

    /// Renders each item on a single line, or `None` if any of them cannot be rendered.
    fn inline_items(&mut self, items: &[BsnValueId], depth: u32) -> Option<Vec<String>> {
        let mut parts = Vec::with_capacity(items.len());
        for item in items {
            parts.push(self.inline_value(*item, depth + 1)?);
        }
        Some(parts)
    }
}

/// Returns `true` if a value has a body that can be broken across lines.
fn is_breakable(value: &BsnValue) -> bool {
    match value {
        BsnValue::Struct(_, fields) => !fields.is_empty(),
        BsnValue::NamedTuple(_, items) | BsnValue::Tuple(items) | BsnValue::List(items) => {
            !items.is_empty()
        }
        _ => false,
    }
}

/// The width, in `char`s, of the text after the last newline in `out`.
fn last_line_width(out: &str) -> usize {
    match out.rfind('\n') {
        Some(index) => out[index + 1..].chars().count(),
        None => out.chars().count(),
    }
}
