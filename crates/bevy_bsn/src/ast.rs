//! The plain-value abstract syntax tree: two flat arenas of nodes and values, addressed by
//! stable [`BsnNodeId`]s and [`BsnValueId`]s.

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt::Write as _;

use crate::error::BsnParseError;
use crate::lexer::Span;
use crate::printer::{escape_string, print_document};

/// The identity of a node (entity, patch or relation) inside a [`BsnDocument`].
///
/// Ids are indices into [`BsnDocument::nodes`] and are assigned in pre-order of the source
/// text, so they are identical across re-parses of identical text.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BsnNodeId(pub u32);

/// The identity of a value inside a [`BsnDocument`].
///
/// Ids are indices into [`BsnDocument::values`], assigned in pre-order of the source text.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BsnValueId(pub u32);

/// A parsed `.bsn` document.
///
/// The document owns two flat arenas — [`BsnDocument::nodes`] and [`BsnDocument::values`] —
/// and refers to their entries by index. All fields are public so that tools can build a
/// document from scratch; [`BsnDocument::push_node`] and friends keep the indices consistent.
#[derive(Clone, Debug, Default)]
pub struct BsnDocument {
    /// Root entities, in source order.
    pub roots: Vec<BsnNodeId>,
    /// Entity, patch and relation nodes, indexed by [`BsnNodeId`].
    pub nodes: Vec<BsnNode>,
    /// Value nodes, indexed by [`BsnValueId`].
    pub values: Vec<BsnValueNode>,
}

/// A node in the [`BsnDocument`] node arena.
#[derive(Clone, Debug)]
pub struct BsnNode {
    /// This node's own id; equal to its index in [`BsnDocument::nodes`].
    pub id: BsnNodeId,
    /// The source text this node covers.
    pub span: Span,
    /// What kind of node this is.
    pub kind: BsnNodeKind,
}

/// A node in the [`BsnDocument`] value arena.
#[derive(Clone, Debug)]
pub struct BsnValueNode {
    /// This value's own id; equal to its index in [`BsnDocument::values`].
    pub id: BsnValueId,
    /// The source text this value covers.
    pub span: Span,
    /// The value itself.
    pub value: BsnValue,
}

/// The three kinds of node: entities, the patches applied to them, and their relations.
#[derive(Clone, Debug)]
pub enum BsnNodeKind {
    /// An entity: an optional `#Name`, an optional `:"base.bsn"` include, and its entries.
    Entity {
        /// The `#Name` of the entity, without the `#`.
        name: Option<String>,
        /// The source range of the `#Name`, if any.
        name_span: Option<Span>,
        /// The `:"path.bsn"` include, unescaped and without the quotes.
        base: Option<String>,
        /// The source range of the `:"path.bsn"` include, if any.
        base_span: Option<Span>,
        /// The entity's patches, in source order. Every id refers to a
        /// [`BsnNodeKind::Patch`].
        patches: Vec<BsnNodeId>,
        /// The entity's relations, in source order. Every id refers to a
        /// [`BsnNodeKind::Relation`].
        relations: Vec<BsnNodeId>,
    },
    /// A patch applied to the enclosing entity, such as `Transform { … }`.
    Patch {
        /// The type path named by the patch.
        symbol: BsnPath,
        /// Which sigil, if any, preceded the path.
        prefix: BsnPatchPrefix,
        /// The patch's value. Always a [`BsnValue::Path`], [`BsnValue::Struct`] or
        /// [`BsnValue::NamedTuple`] whose path equals `symbol`.
        value: BsnValueId,
    },
    /// A relation block, such as `Children [ … ]`.
    Relation {
        /// The type path of the relationship target named by the block.
        target_symbol: BsnPath,
        /// The related entities, in source order. Every id refers to a
        /// [`BsnNodeKind::Entity`].
        entities: Vec<BsnNodeId>,
    },
}

/// Which sigil (if any) preceded a patch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BsnPatchPrefix {
    /// `Transform { … }` — the symbol is a component; resolve it through `FromTemplate`.
    FromTemplate,
    /// `~MyTemplate { … }` — the symbol is already a template.
    Template,
    /// `@MyWidget { … }` — the symbol is a scene component.
    SceneComponent,
}

impl BsnPatchPrefix {
    /// Returns `true` for `~`-prefixed patches.
    pub fn is_template(self) -> bool {
        matches!(self, Self::Template)
    }

    /// The sigil that produced this prefix: `""`, `"~"` or `"@"`.
    pub fn sigil(self) -> &'static str {
        match self {
            BsnPatchPrefix::FromTemplate => "",
            BsnPatchPrefix::Template => "~",
            BsnPatchPrefix::SceneComponent => "@",
        }
    }
}

/// A value in the value arena.
///
/// This enum is deliberately exhaustive: a new value form is a format change that every
/// resolver must consciously handle.
#[derive(Clone, Debug)]
pub enum BsnValue {
    /// `()`
    Unit,
    /// `true` or `false`.
    Bool(bool),
    /// An integer literal. The radix and any `_` separators of the source are not preserved.
    Int(i128),
    /// A floating-point literal, including `inf`, `-inf` and `NaN`.
    Float(f64),
    /// A string literal, unescaped and without its delimiters.
    String(String),
    /// A bare path, such as `Foo` or `a::b::Foo::Bar`. The parser does not decide whether
    /// this names a unit struct or an enum variant.
    Path(BsnPath),
    /// A tuple value, such as `(1, 2)`. A one-element tuple is written `(1,)`.
    Tuple(Vec<BsnValueId>),
    /// A list value, such as `[1, 2]`.
    List(Vec<BsnValueId>),
    /// A struct value, such as `a::B { x: 1 }`. Fields are in source order.
    Struct(BsnPath, Vec<(String, BsnValueId)>),
    /// A tuple-struct or tuple-variant value, such as `a::B(1, 2)`.
    NamedTuple(BsnPath, Vec<BsnValueId>),
    /// A reference to a named entity, such as `#Root`, without the `#`.
    EntityRef(String),
}

/// A type path, such as `a::b::C<d::E>`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BsnPath {
    /// The `::`-separated segments. Always at least one.
    pub segments: Vec<BsnPathSegment>,
    /// The source text the whole path covers.
    pub span: Span,
}

/// One segment of a [`BsnPath`], with its optional generic arguments.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BsnPathSegment {
    /// The segment's identifier.
    pub ident: String,
    /// Generic arguments, e.g. the `f32` in `Vec<f32>`. Empty when there are none.
    pub generics: Vec<BsnPath>,
    /// The source text the segment covers, including its generic arguments.
    pub span: Span,
}

impl BsnPath {
    /// Renders the path as a canonical type-path string: `::` between segments and `", "`
    /// between generic arguments.
    ///
    /// This is byte-identical to the type-path strings a reflection type registry uses.
    pub fn to_type_path(&self) -> String {
        let mut out = String::new();
        for (index, segment) in self.segments.iter().enumerate() {
            if index > 0 {
                out.push_str("::");
            }
            out.push_str(&segment.ident);
            if !segment.generics.is_empty() {
                out.push('<');
                for (generic_index, generic) in segment.generics.iter().enumerate() {
                    if generic_index > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&generic.to_type_path());
                }
                out.push('>');
            }
        }
        out
    }

    /// Everything but the last segment, or `None` if there is only one segment.
    ///
    /// Resolvers use this for the enum-variant fallback: if `a::B::C` is not a type, try
    /// `a::B` with variant `C`.
    pub fn parent_type_path(&self) -> Option<String> {
        if self.segments.len() < 2 {
            return None;
        }
        let parent = BsnPath {
            segments: self.segments[..self.segments.len() - 1].to_vec(),
            span: self.span,
        };
        Some(parent.to_type_path())
    }

    /// The identifier of the last segment.
    pub fn last_ident(&self) -> &str {
        self.segments.last().map_or("", |s| s.ident.as_str())
    }

    /// Returns `true` if the path has exactly one segment.
    pub fn is_single_segment(&self) -> bool {
        self.segments.len() == 1
    }

    /// Parses a canonical type-path string, the inverse of [`BsnPath::to_type_path`].
    ///
    /// Returns `None` if `s` is not a syntactically valid path. All spans in the result
    /// refer to `s`, not to any document.
    pub fn from_type_path(s: &str) -> Option<BsnPath> {
        crate::parser::parse_path_str(s)
    }

    /// Builds a path from plain segment identifiers, without generic arguments.
    pub fn from_segments(idents: impl IntoIterator<Item = impl Into<String>>) -> BsnPath {
        BsnPath {
            segments: idents
                .into_iter()
                .map(|ident| BsnPathSegment {
                    ident: ident.into(),
                    generics: Vec::new(),
                    span: Span::NONE,
                })
                .collect(),
            span: Span::NONE,
        }
    }

    /// Structural equality: like `==`, but ignoring every [`Span`].
    pub fn structural_eq(&self, other: &BsnPath) -> bool {
        self.segments.len() == other.segments.len()
            && self.segments.iter().zip(&other.segments).all(|(a, b)| {
                a.ident == b.ident && a.generics.len() == b.generics.len() && {
                    a.generics
                        .iter()
                        .zip(&b.generics)
                        .all(|(a, b)| a.structural_eq(b))
                }
            })
    }
}

/// What follows a patch's path when building one with [`BsnDocument::push_patch`].
#[derive(Clone, Debug)]
pub enum PatchBody {
    /// No body at all: `Foo`.
    Unit,
    /// A struct body: `Foo { x: 1 }`.
    Struct(Vec<(String, BsnValueId)>),
    /// A tuple body: `Foo(1, 2)`.
    Tuple(Vec<BsnValueId>),
}

impl BsnDocument {
    /// Parses `source` into a document. Identical to [`crate::parse`].
    pub fn parse(source: &str) -> Result<BsnDocument, BsnParseError> {
        crate::parser::parse(source)
    }

    /// An empty document. Identical to [`Default::default`].
    pub fn new() -> Self {
        BsnDocument::default()
    }

    /// Looks up a node by id, or `None` if the id is out of bounds.
    pub fn node(&self, id: BsnNodeId) -> Option<&BsnNode> {
        self.nodes.get(id.0 as usize)
    }

    /// Looks up a value by id, or `None` if the id is out of bounds.
    pub fn value(&self, id: BsnValueId) -> Option<&BsnValueNode> {
        self.values.get(id.0 as usize)
    }

    /// Iterates every entity node in document order (that is, in ascending id).
    pub fn entities(&self) -> impl Iterator<Item = &BsnNode> {
        self.nodes
            .iter()
            .filter(|node| matches!(node.kind, BsnNodeKind::Entity { .. }))
    }

    /// Appends a value node with the [`Span::NONE`] placeholder span and returns its id.
    pub fn push_value(&mut self, value: BsnValue) -> BsnValueId {
        self.push_value_spanned(value, Span::NONE)
    }

    /// Appends a value node with an explicit span and returns its id.
    pub fn push_value_spanned(&mut self, value: BsnValue, span: Span) -> BsnValueId {
        let id = BsnValueId(self.values.len() as u32);
        self.values.push(BsnValueNode { id, span, value });
        id
    }

    /// Appends a node with the [`Span::NONE`] placeholder span and returns its id.
    pub fn push_node(&mut self, kind: BsnNodeKind) -> BsnNodeId {
        self.push_node_spanned(kind, Span::NONE)
    }

    /// Appends a node with an explicit span and returns its id.
    pub fn push_node_spanned(&mut self, kind: BsnNodeKind, span: Span) -> BsnNodeId {
        let id = BsnNodeId(self.nodes.len() as u32);
        self.nodes.push(BsnNode { id, span, kind });
        id
    }

    /// Marks an existing node as a document root by appending it to [`BsnDocument::roots`].
    pub fn push_root(&mut self, id: BsnNodeId) {
        self.roots.push(id);
    }

    /// Appends a patch node, building the matching value for you so that the patch/value
    /// invariant holds.
    pub fn push_patch(
        &mut self,
        prefix: BsnPatchPrefix,
        path: BsnPath,
        body: PatchBody,
    ) -> BsnNodeId {
        let value = match body {
            PatchBody::Unit => BsnValue::Path(path.clone()),
            PatchBody::Struct(fields) => BsnValue::Struct(path.clone(), fields),
            PatchBody::Tuple(items) => BsnValue::NamedTuple(path.clone(), items),
        };
        let value = self.push_value(value);
        self.push_node(BsnNodeKind::Patch {
            symbol: path,
            prefix,
            value,
        })
    }

    /// Canonical `.bsn` text for this document. Convenience alias for
    /// [`crate::print_document`].
    pub fn to_bsn_string(&self) -> String {
        print_document(self)
    }

    /// Structural equality, ignoring every [`Span`].
    ///
    /// Floats are compared by `to_bits`, so `NaN` equals `NaN` and `0.0` does not equal
    /// `-0.0`. Ids are compared by the position they occupy in the traversal rather than by
    /// value, so two documents whose arenas are ordered differently can still be equal.
    ///
    /// [`BsnValue::Unit`] and an empty [`BsnValue::Tuple`] compare equal in both directions.
    /// A builder can construct both, but `()` is the only text that denotes either, so the
    /// distinction cannot survive a print/parse round trip and treating them as different
    /// would break the round-trip property for documents the builder can legitimately build.
    pub fn structural_eq(&self, other: &BsnDocument) -> bool {
        if self.roots.len() != other.roots.len() {
            return false;
        }
        self.roots
            .iter()
            .zip(&other.roots)
            .all(|(a, b)| self.node_eq(*a, other, *b, 0))
    }

    fn node_eq(&self, id: BsnNodeId, other: &BsnDocument, other_id: BsnNodeId, depth: u32) -> bool {
        if depth > MAX_WALK_DEPTH {
            return false;
        }
        let (Some(a), Some(b)) = (self.node(id), other.node(other_id)) else {
            return false;
        };
        match (&a.kind, &b.kind) {
            (
                BsnNodeKind::Entity {
                    name,
                    base,
                    patches,
                    relations,
                    ..
                },
                BsnNodeKind::Entity {
                    name: other_name,
                    base: other_base,
                    patches: other_patches,
                    relations: other_relations,
                    ..
                },
            ) => {
                name == other_name
                    && base == other_base
                    && patches.len() == other_patches.len()
                    && relations.len() == other_relations.len()
                    && patches
                        .iter()
                        .zip(other_patches)
                        .all(|(a, b)| self.node_eq(*a, other, *b, depth + 1))
                    && relations
                        .iter()
                        .zip(other_relations)
                        .all(|(a, b)| self.node_eq(*a, other, *b, depth + 1))
            }
            (
                BsnNodeKind::Patch {
                    symbol,
                    prefix,
                    value,
                },
                BsnNodeKind::Patch {
                    symbol: other_symbol,
                    prefix: other_prefix,
                    value: other_value,
                },
            ) => {
                prefix == other_prefix
                    && symbol.structural_eq(other_symbol)
                    && self.value_eq(*value, other, *other_value, depth + 1)
            }
            (
                BsnNodeKind::Relation {
                    target_symbol,
                    entities,
                },
                BsnNodeKind::Relation {
                    target_symbol: other_symbol,
                    entities: other_entities,
                },
            ) => {
                target_symbol.structural_eq(other_symbol)
                    && entities.len() == other_entities.len()
                    && entities
                        .iter()
                        .zip(other_entities)
                        .all(|(a, b)| self.node_eq(*a, other, *b, depth + 1))
            }
            _ => false,
        }
    }

    fn value_eq(
        &self,
        id: BsnValueId,
        other: &BsnDocument,
        other_id: BsnValueId,
        depth: u32,
    ) -> bool {
        if depth > MAX_WALK_DEPTH {
            return false;
        }
        let (Some(a), Some(b)) = (self.value(id), other.value(other_id)) else {
            return false;
        };
        let items_eq = |a: &Vec<BsnValueId>, b: &Vec<BsnValueId>| {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(a, b)| self.value_eq(*a, other, *b, depth + 1))
        };
        match (&a.value, &b.value) {
            (BsnValue::Unit, BsnValue::Unit) => true,
            // `()` is the only text for both, so the printer cannot tell them apart and the
            // comparison must not either. See the note on `structural_eq`.
            (BsnValue::Unit, BsnValue::Tuple(items)) | (BsnValue::Tuple(items), BsnValue::Unit) => {
                items.is_empty()
            }
            (BsnValue::Bool(a), BsnValue::Bool(b)) => a == b,
            (BsnValue::Int(a), BsnValue::Int(b)) => a == b,
            (BsnValue::Float(a), BsnValue::Float(b)) => a.to_bits() == b.to_bits(),
            (BsnValue::String(a), BsnValue::String(b))
            | (BsnValue::EntityRef(a), BsnValue::EntityRef(b)) => a == b,
            (BsnValue::Path(a), BsnValue::Path(b)) => a.structural_eq(b),
            (BsnValue::Tuple(a), BsnValue::Tuple(b)) | (BsnValue::List(a), BsnValue::List(b)) => {
                items_eq(a, b)
            }
            (BsnValue::NamedTuple(path, a), BsnValue::NamedTuple(other_path, b)) => {
                path.structural_eq(other_path) && items_eq(a, b)
            }
            (BsnValue::Struct(path, a), BsnValue::Struct(other_path, b)) => {
                path.structural_eq(other_path)
                    && a.len() == b.len()
                    && a.iter().zip(b).all(|((name, a), (other_name, b))| {
                        name == other_name && self.value_eq(*a, other, *b, depth + 1)
                    })
            }
            _ => false,
        }
    }

    /// A deterministic, indented dump of the document, used by tests and for debugging.
    ///
    /// Nodes are printed as a tree, followed by a flat `values:` section in ascending id.
    pub fn debug_tree(&self) -> String {
        let mut out = String::new();
        for root in &self.roots {
            self.debug_node(*root, 0, &mut out);
        }
        out.push_str("values:\n");
        for value in &self.values {
            let _ = writeln!(out, "${} {}", value.id.0, debug_value_head(&value.value));
            match &value.value {
                BsnValue::Tuple(items) | BsnValue::List(items) | BsnValue::NamedTuple(_, items) => {
                    for item in items {
                        let _ = writeln!(out, "  ${}", item.0);
                    }
                }
                BsnValue::Struct(_, fields) => {
                    for (name, item) in fields {
                        let _ = writeln!(out, "  field {name} = ${}", item.0);
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn debug_node(&self, id: BsnNodeId, indent: usize, out: &mut String) {
        if indent > MAX_WALK_DEPTH as usize {
            return;
        }
        let pad = " ".repeat(indent);
        let Some(node) = self.node(id) else {
            let _ = writeln!(out, "{pad}<invalid node id {}>", id.0);
            return;
        };
        match &node.kind {
            BsnNodeKind::Entity {
                name,
                base,
                patches,
                relations,
                ..
            } => {
                let name = name
                    .as_deref()
                    .map_or_else(|| "-".to_string(), escape_string);
                let base = base
                    .as_deref()
                    .map_or_else(|| "-".to_string(), escape_string);
                let _ = writeln!(out, "{pad}Entity#{} name={name} base={base}", node.id.0);
                for entry in merge_entries(self, patches, relations) {
                    self.debug_node(entry, indent + 2, out);
                }
            }
            BsnNodeKind::Patch {
                symbol,
                prefix,
                value,
            } => {
                let kind = match prefix {
                    BsnPatchPrefix::FromTemplate => "patch",
                    BsnPatchPrefix::Template => "template",
                    BsnPatchPrefix::SceneComponent => "scene",
                };
                let _ = writeln!(
                    out,
                    "{pad}Patch#{} {kind} `{}` value=${}",
                    node.id.0,
                    symbol.to_type_path(),
                    value.0
                );
            }
            BsnNodeKind::Relation {
                target_symbol,
                entities,
            } => {
                let _ = writeln!(
                    out,
                    "{pad}Relation#{} `{}`",
                    node.id.0,
                    target_symbol.to_type_path()
                );
                for entity in entities {
                    self.debug_node(*entity, indent + 2, out);
                }
            }
        }
    }
}

/// The head line of a value in [`BsnDocument::debug_tree`] output.
fn debug_value_head(value: &BsnValue) -> String {
    match value {
        BsnValue::Unit => "Unit".to_string(),
        BsnValue::Bool(value) => format!("Bool({value})"),
        BsnValue::Int(value) => format!("Int({value})"),
        BsnValue::Float(value) => format!("Float({})", crate::printer::format_float(*value)),
        BsnValue::String(value) => format!("Str({})", escape_string(value)),
        BsnValue::Path(path) => format!("Path({})", path.to_type_path()),
        BsnValue::EntityRef(name) => format!("EntityRef({name})"),
        BsnValue::Tuple(_) => "Tuple".to_string(),
        BsnValue::List(_) => "List".to_string(),
        BsnValue::Struct(path, _) => format!("Struct({})", path.to_type_path()),
        BsnValue::NamedTuple(path, _) => format!("NamedTuple({})", path.to_type_path()),
    }
}

/// A guard against pathological or cyclic hand-built documents.
pub(crate) const MAX_WALK_DEPTH: u32 = 256;

/// Merges an entity's patches and relations back into source order.
///
/// Entries are ordered by [`Span::start`], with ties broken by ascending id, so a
/// synthesized document (all spans [`Span::NONE`]) prints in builder order.
///
/// An entry with no span sorts *after* every spanned one rather than at offset 0, so
/// appending to an entity that came from [`parse`](crate::parse) appends to its printed form
/// too — patch order is semantically significant, and "appended last" is what the builder
/// asked for. A dangling id has no span either and is ordered the same way.
pub(crate) fn merge_entries(
    document: &BsnDocument,
    patches: &[BsnNodeId],
    relations: &[BsnNodeId],
) -> Vec<BsnNodeId> {
    let mut entries: Vec<BsnNodeId> = patches.iter().chain(relations).copied().collect();
    entries.sort_by_key(|id| {
        let start = document.node(*id).map_or(u32::MAX, |node| {
            if node.span.is_none() {
                u32::MAX
            } else {
                node.span.start
            }
        });
        (start, id.0)
    });
    entries
}
