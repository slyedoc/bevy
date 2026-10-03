//! The recursive-descent parser.
//!
//! [`parse`] turns BSN source text into a [`BsnDocument`]. Parsing is fail-fast: the first
//! problem aborts and is returned as a [`BsnParseError`].

use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};

use crate::ast::{
    BsnDocument, BsnNode, BsnNodeId, BsnNodeKind, BsnPatchPrefix, BsnPath, BsnPathSegment,
    BsnValue, BsnValueId, BsnValueNode,
};
use crate::error::{unsupported, BsnParseError, BsnParseErrorKind};
use crate::lexer::{
    decode_float, decode_int, decode_int_magnitude, decode_string, lex_error_to_parse_error, Lexer,
    Span, Token, TokenKind,
};

/// The maximum nesting depth of entities, values and paths.
///
/// `.bsn` files are untrusted input to an asset loader, so the recursive-descent parser
/// refuses to descend further rather than risk exhausting the stack. Exceeding the limit is
/// reported as [`BsnParseErrorKind::NestingTooDeep`].
pub const MAX_NESTING_DEPTH: u32 = 128;

/// Parses `source` into a [`BsnDocument`].
///
/// # Errors
///
/// Returns the first [`BsnParseError`] encountered. Use [`BsnParseError::render`] to render
/// it against `source`.
pub fn parse(source: &str) -> Result<BsnDocument, BsnParseError> {
    Parser::new(source).parse_document()
}

/// Parses a bare type path, used by [`BsnPath::from_type_path`].
pub(crate) fn parse_path_str(source: &str) -> Option<BsnPath> {
    let mut parser = Parser::new(source);
    let path = parser.parse_path().ok()?;
    if parser.peek() != TokenKind::Eof {
        return None;
    }
    Some(path)
}

/// The parts of an entity collected while its entries are parsed.
#[derive(Default)]
struct EntityBuilder {
    name: Option<String>,
    name_span: Option<Span>,
    base: Option<String>,
    base_span: Option<Span>,
    patches: Vec<BsnNodeId>,
    relations: Vec<BsnNodeId>,
}

struct Parser<'src> {
    source: &'src str,
    tokens: Vec<Token>,
    pos: usize,
    depth: u32,
    nodes: Vec<Option<BsnNode>>,
    values: Vec<Option<BsnValueNode>>,
}

impl<'src> Parser<'src> {
    fn new(source: &'src str) -> Self {
        Parser {
            source,
            tokens: Lexer::tokenize(source),
            pos: 0,
            depth: 0,
            nodes: Vec::new(),
            values: Vec::new(),
        }
    }

    // -- token primitives ---------------------------------------------------------------

    fn peek_token(&self) -> Token {
        let end = self.source.len() as u32;
        self.tokens.get(self.pos).copied().unwrap_or(Token {
            kind: TokenKind::Eof,
            span: Span::new(end, end),
        })
    }

    fn peek(&self) -> TokenKind {
        self.peek_token().kind
    }

    fn peek_at(&self, offset: usize) -> TokenKind {
        self.tokens
            .get(self.pos + offset)
            .map_or(TokenKind::Eof, |token| token.kind)
    }

    fn bump(&mut self) -> Token {
        let token = self.peek_token();
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        token
    }

    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.peek() == kind {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(
        &mut self,
        kind: TokenKind,
        expected: &[&'static str],
    ) -> Result<Token, BsnParseError> {
        self.check_error_token()?;
        let token = self.peek_token();
        if token.kind == kind {
            self.bump();
            return Ok(token);
        }
        Err(self.unexpected(token, expected))
    }

    /// The end offset of the most recently consumed token.
    fn prev_end(&self) -> u32 {
        self.pos
            .checked_sub(1)
            .and_then(|index| self.tokens.get(index))
            .map_or(0, |token| token.span.end)
    }

    /// Builds the error for a token that the grammar does not allow here.
    fn unexpected(&self, token: Token, expected: &[&'static str]) -> BsnParseError {
        let kind = if token.kind == TokenKind::Eof {
            BsnParseErrorKind::UnexpectedEof
        } else {
            BsnParseErrorKind::UnexpectedToken {
                found: token_desc(token.kind),
            }
        };
        BsnParseError::expected(token.span, kind, expected)
    }

    /// Turns a [`TokenKind::Error`] at the cursor into the parse error it stands for.
    fn check_error_token(&self) -> Result<(), BsnParseError> {
        let token = self.peek_token();
        if let TokenKind::Error(error) = token.kind {
            return Err(lex_error_to_parse_error(error, token.span, self.source));
        }
        Ok(())
    }

    fn unsupported(&self, message: &'static str, span: Span) -> BsnParseError {
        BsnParseError::new(span, BsnParseErrorKind::Unsupported(message))
    }

    fn enter(&mut self) -> Result<(), BsnParseError> {
        self.depth += 1;
        if self.depth > MAX_NESTING_DEPTH {
            return Err(BsnParseError::new(
                self.peek_token().span,
                BsnParseErrorKind::NestingTooDeep,
            ));
        }
        Ok(())
    }

    fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Runs `inner` one nesting level deeper, enforcing [`MAX_NESTING_DEPTH`].
    fn nested<T>(
        &mut self,
        inner: impl FnOnce(&mut Self) -> Result<T, BsnParseError>,
    ) -> Result<T, BsnParseError> {
        self.enter()?;
        let result = inner(self);
        self.leave();
        result
    }

    /// Comma-separated items with an optional trailing comma, up to (not including) `term`.
    fn parse_list<T>(
        &mut self,
        term: TokenKind,
        mut item: impl FnMut(&mut Self) -> Result<T, BsnParseError>,
    ) -> Result<Vec<T>, BsnParseError> {
        let mut items = Vec::new();
        loop {
            self.check_error_token()?;
            if self.peek() == term {
                break;
            }
            items.push(item(self)?);
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        Ok(items)
    }

    // -- arenas -------------------------------------------------------------------------

    fn alloc_node(&mut self) -> BsnNodeId {
        let id = BsnNodeId(self.nodes.len() as u32);
        self.nodes.push(None);
        id
    }

    fn finish_node(&mut self, id: BsnNodeId, span: Span, kind: BsnNodeKind) {
        if let Some(slot) = self.nodes.get_mut(id.0 as usize) {
            *slot = Some(BsnNode { id, span, kind });
        }
    }

    fn alloc_value(&mut self) -> BsnValueId {
        let id = BsnValueId(self.values.len() as u32);
        self.values.push(None);
        id
    }

    fn finish_value(&mut self, id: BsnValueId, span: Span, value: BsnValue) {
        if let Some(slot) = self.values.get_mut(id.0 as usize) {
            *slot = Some(BsnValueNode { id, span, value });
        }
    }

    /// Allocates and finishes a leaf value in one step.
    fn push_value(&mut self, span: Span, value: BsnValue) -> BsnValueId {
        let id = self.alloc_value();
        self.finish_value(id, span, value);
        id
    }

    fn finish(self, roots: Vec<BsnNodeId>) -> Result<BsnDocument, BsnParseError> {
        // Every reserved slot is filled in by construction; a `None` is a bug in this crate.
        let internal = |what| BsnParseError::new(Span::NONE, BsnParseErrorKind::Internal(what));
        let nodes = self
            .nodes
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| internal("a reserved node was never finished"))?;
        let values = self
            .values
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| internal("a reserved value was never finished"))?;
        Ok(BsnDocument {
            roots,
            nodes,
            values,
        })
    }

    // -- productions --------------------------------------------------------------------

    /// `document = [ entity_list ] EOF`
    fn parse_document(mut self) -> Result<BsnDocument, BsnParseError> {
        let roots = if self.peek() == TokenKind::Eof {
            Vec::new()
        } else {
            self.parse_list(TokenKind::Eof, Self::parse_entity)?
        };
        self.expect(TokenKind::Eof, &["`,`", "end of file"])?;
        self.finish(roots)
    }

    /// `entity = "(" entity_body ")" | entity_body`
    fn parse_entity(&mut self) -> Result<BsnNodeId, BsnParseError> {
        self.nested(Self::parse_entity_inner)
    }

    fn parse_entity_inner(&mut self) -> Result<BsnNodeId, BsnParseError> {
        self.check_error_token()?;
        let id = self.alloc_node();
        let start = self.peek_token().span.start;
        let start_pos = self.pos;
        let parenthesized = self.eat(TokenKind::LParen);
        let mut builder = EntityBuilder::default();
        self.parse_entity_body(&mut builder, parenthesized)?;
        if parenthesized {
            self.expect(TokenKind::RParen, &["`)`", "type path"])?;
        } else if self.pos == start_pos {
            let token = self.peek_token();
            return Err(self.unexpected(token, &["type path", "`#`", "`~`", "`@`", "`(`", "`:`"]));
        }
        let span = Span::new(start, self.prev_end());
        self.finish_node(
            id,
            span,
            BsnNodeKind::Entity {
                name: builder.name,
                name_span: builder.name_span,
                base: builder.base,
                base_span: builder.base_span,
                patches: builder.patches,
                relations: builder.relations,
            },
        );
        Ok(id)
    }

    /// `entity_body = [ base ] { entry }`
    fn parse_entity_body(
        &mut self,
        builder: &mut EntityBuilder,
        parenthesized: bool,
    ) -> Result<(), BsnParseError> {
        if self.peek() == TokenKind::Colon {
            let colon = self.bump();
            self.check_error_token()?;
            let token = self.peek_token();
            if token.kind != TokenKind::Str {
                return Err(BsnParseError::new(
                    token.span,
                    BsnParseErrorKind::BaseNotString,
                ));
            }
            self.bump();
            builder.base = Some(decode_string(self.source, token.span)?);
            builder.base_span = Some(colon.span.join(token.span));
        }
        loop {
            if is_entity_stop(self.peek(), parenthesized) {
                break;
            }
            self.parse_entry(builder)?;
        }
        Ok(())
    }

    /// `entry = name_entry | relation_entry | patch_entry`
    fn parse_entry(&mut self, builder: &mut EntityBuilder) -> Result<(), BsnParseError> {
        self.check_error_token()?;
        let token = self.peek_token();
        match token.kind {
            TokenKind::Colon => Err(BsnParseError::new(
                token.span,
                BsnParseErrorKind::BaseNotFirst,
            )),
            TokenKind::ColonColon => Err(BsnParseError::new(
                token.span,
                BsnParseErrorKind::LeadingPathSeparator,
            )),
            TokenKind::LBrace => Err(self.unsupported(unsupported::EXPR, token.span)),
            TokenKind::Hash => {
                self.bump();
                let ident = self.expect(TokenKind::Ident, &["identifier"])?;
                if builder.name.is_some() {
                    return Err(BsnParseError::new(
                        ident.span,
                        BsnParseErrorKind::DuplicateName,
                    ));
                }
                builder.name = Some(ident.span.text(self.source).to_string());
                builder.name_span = Some(token.span.join(ident.span));
                Ok(())
            }
            TokenKind::Tilde => {
                self.bump();
                self.parse_patch(builder, BsnPatchPrefix::Template, token.span)
            }
            TokenKind::At => {
                self.bump();
                self.parse_patch(builder, BsnPatchPrefix::SceneComponent, token.span)
            }
            TokenKind::Ident => {
                let text = token.span.text(self.source);
                if text == "use" {
                    return Err(self.unsupported(unsupported::USE, token.span));
                }
                if text == "on" && self.peek_at(1) == TokenKind::LParen {
                    return Err(self.unsupported(unsupported::OBSERVER, token.span));
                }
                self.parse_patch(builder, BsnPatchPrefix::FromTemplate, token.span)
            }
            _ => Err(self.unexpected(token, &["type path", "`#`", "`~`", "`@`"])),
        }
    }

    /// `patch_entry` and `relation_entry`, which share a leading path.
    fn parse_patch(
        &mut self,
        builder: &mut EntityBuilder,
        prefix: BsnPatchPrefix,
        start: Span,
    ) -> Result<(), BsnParseError> {
        let id = self.alloc_node();
        let path = self.parse_path()?;
        self.check_error_token()?;
        self.classify_path(&path, unsupported::FN)?;
        if self.peek() == TokenKind::LBracket {
            if prefix != BsnPatchPrefix::FromTemplate {
                let token = self.peek_token();
                return Err(self.unexpected(
                    token,
                    &["a patch body (relationships cannot be prefixed with `~` or `@`)"],
                ));
            }
            self.bump();
            let entities = self.parse_list(TokenKind::RBracket, Self::parse_entity)?;
            self.expect(TokenKind::RBracket, &["`,`", "`]`"])?;
            let span = Span::new(start.start, self.prev_end());
            self.finish_node(
                id,
                span,
                BsnNodeKind::Relation {
                    target_symbol: path,
                    entities,
                },
            );
            builder.relations.push(id);
            return Ok(());
        }
        let value = self.parse_path_body(path.clone())?;
        let span = Span::new(start.start, self.prev_end());
        self.finish_node(
            id,
            span,
            BsnNodeKind::Patch {
                symbol: path,
                prefix,
                value,
            },
        );
        builder.patches.push(id);
        Ok(())
    }

    /// What follows a path in patch or value position: nothing, a struct body or a tuple body.
    fn parse_path_body(&mut self, path: BsnPath) -> Result<BsnValueId, BsnParseError> {
        match self.peek() {
            TokenKind::LBrace => {
                let id = self.alloc_value();
                let fields = self.parse_struct_body()?;
                let span = Span::new(path.span.start, self.prev_end());
                self.finish_value(id, span, BsnValue::Struct(path, fields));
                Ok(id)
            }
            TokenKind::LParen => {
                let id = self.alloc_value();
                let items = self.parse_tuple_body()?;
                let span = Span::new(path.span.start, self.prev_end());
                self.finish_value(id, span, BsnValue::NamedTuple(path, items));
                Ok(id)
            }
            _ => {
                let span = path.span;
                Ok(self.push_value(span, BsnValue::Path(path)))
            }
        }
    }

    /// `struct_body = "{" [ field { "," field } [ "," ] ] "}"`
    fn parse_struct_body(&mut self) -> Result<Vec<(String, BsnValueId)>, BsnParseError> {
        self.expect(TokenKind::LBrace, &["`{`"])?;
        let mut fields: Vec<(String, BsnValueId)> = Vec::new();
        let mut name_spans: Vec<Span> = Vec::new();
        loop {
            self.check_error_token()?;
            if self.peek() == TokenKind::RBrace {
                break;
            }
            if self.peek() == TokenKind::At {
                let token = self.peek_token();
                return Err(self.unsupported(unsupported::PROP, token.span));
            }
            let ident = self.expect(TokenKind::Ident, &["identifier", "`}`"])?;
            if self.peek() != TokenKind::Colon {
                return Err(self.unsupported(unsupported::SHORTHAND, ident.span));
            }
            self.bump();
            let name = ident.span.text(self.source).to_string();
            let value = self.parse_value()?;
            fields.push((name, value));
            name_spans.push(ident.span);
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        if let Some(duplicate) = first_duplicate_field(&fields) {
            return Err(BsnParseError::new(
                name_spans[duplicate],
                BsnParseErrorKind::DuplicateField(fields[duplicate].0.clone()),
            ));
        }
        self.expect(TokenKind::RBrace, &["`,`", "`}`"])?;
        Ok(fields)
    }

    /// `tuple_body = "(" [ value { "," value } [ "," ] ] ")"`
    fn parse_tuple_body(&mut self) -> Result<Vec<BsnValueId>, BsnParseError> {
        self.expect(TokenKind::LParen, &["`(`"])?;
        let items = self.parse_list(TokenKind::RParen, Self::parse_value)?;
        self.expect(TokenKind::RParen, &["`,`", "`)`"])?;
        Ok(items)
    }

    /// `value`
    fn parse_value(&mut self) -> Result<BsnValueId, BsnParseError> {
        self.nested(Self::parse_value_inner)
    }

    fn parse_value_inner(&mut self) -> Result<BsnValueId, BsnParseError> {
        self.check_error_token()?;
        let token = self.peek_token();
        match token.kind {
            TokenKind::Int => {
                self.bump();
                let value = decode_int(self.source, token.span)?;
                Ok(self.push_value(token.span, BsnValue::Int(value)))
            }
            TokenKind::Float => {
                self.bump();
                let value = decode_float(self.source, token.span)?;
                Ok(self.push_value(token.span, BsnValue::Float(value)))
            }
            TokenKind::Str => {
                self.bump();
                let value = decode_string(self.source, token.span)?;
                Ok(self.push_value(token.span, BsnValue::String(value)))
            }
            TokenKind::Minus => {
                self.bump();
                self.check_error_token()?;
                let next = self.peek_token();
                match next.kind {
                    TokenKind::Int => {
                        self.bump();
                        // The magnitude of `i128::MIN` is one past `i128::MAX`, so the
                        // literal only fits while it is still unsigned.
                        let magnitude = decode_int_magnitude(self.source, next.span)?;
                        let value = if magnitude == i128::MIN.unsigned_abs() {
                            i128::MIN
                        } else {
                            i128::try_from(magnitude)
                                .ok()
                                .and_then(i128::checked_neg)
                                .ok_or_else(|| {
                                    BsnParseError::new(
                                        token.span.join(next.span),
                                        BsnParseErrorKind::NumberOutOfRange,
                                    )
                                })?
                        };
                        Ok(self.push_value(token.span.join(next.span), BsnValue::Int(value)))
                    }
                    TokenKind::Float => {
                        self.bump();
                        let value = decode_float(self.source, next.span)?;
                        Ok(self.push_value(token.span.join(next.span), BsnValue::Float(-value)))
                    }
                    TokenKind::Ident if next.span.text(self.source) == "inf" => {
                        self.bump();
                        Ok(self.push_value(
                            token.span.join(next.span),
                            BsnValue::Float(f64::NEG_INFINITY),
                        ))
                    }
                    TokenKind::Ident if matches!(next.span.text(self.source), "NaN" | "nan") => {
                        self.bump();
                        Ok(self.push_value(token.span.join(next.span), BsnValue::Float(-f64::NAN)))
                    }
                    _ => Err(BsnParseError::new(next.span, BsnParseErrorKind::NegOperand)),
                }
            }
            TokenKind::Hash => {
                self.bump();
                let ident = self.expect(TokenKind::Ident, &["identifier"])?;
                let name = ident.span.text(self.source).to_string();
                Ok(self.push_value(token.span.join(ident.span), BsnValue::EntityRef(name)))
            }
            TokenKind::LBracket => {
                let id = self.alloc_value();
                self.bump();
                let items = self.parse_list(TokenKind::RBracket, Self::parse_value)?;
                self.expect(TokenKind::RBracket, &["`,`", "`]`"])?;
                let span = Span::new(token.span.start, self.prev_end());
                self.finish_value(id, span, BsnValue::List(items));
                Ok(id)
            }
            TokenKind::LParen => self.parse_paren_value(),
            TokenKind::LBrace => Err(self.unsupported(unsupported::EXPR, token.span)),
            TokenKind::ColonColon => Err(BsnParseError::new(
                token.span,
                BsnParseErrorKind::LeadingPathSeparator,
            )),
            TokenKind::Ident => {
                let text = token.span.text(self.source);
                match text {
                    "true" => {
                        self.bump();
                        Ok(self.push_value(token.span, BsnValue::Bool(true)))
                    }
                    "false" => {
                        self.bump();
                        Ok(self.push_value(token.span, BsnValue::Bool(false)))
                    }
                    "inf" => {
                        self.bump();
                        Ok(self.push_value(token.span, BsnValue::Float(f64::INFINITY)))
                    }
                    "NaN" | "nan" => {
                        self.bump();
                        Ok(self.push_value(token.span, BsnValue::Float(f64::NAN)))
                    }
                    "const" | "unsafe" if self.peek_at(1) == TokenKind::LBrace => {
                        Err(self.unsupported(unsupported::EXPR, token.span))
                    }
                    _ => self.parse_path_value(),
                }
            }
            _ => Err(self.unexpected(
                token,
                &[
                    "a value",
                    "number",
                    "string literal",
                    "`true`",
                    "`false`",
                    "type path",
                    "`#`",
                    "`[`",
                    "`(`",
                ],
            )),
        }
    }

    /// `paren_value` — unit, grouping or tuple.
    fn parse_paren_value(&mut self) -> Result<BsnValueId, BsnParseError> {
        let open = self.bump();
        if self.peek() == TokenKind::RParen {
            let close = self.bump();
            return Ok(self.push_value(open.span.join(close.span), BsnValue::Unit));
        }
        if !self.paren_has_top_level_comma() {
            let value = self.parse_value()?;
            self.expect(TokenKind::RParen, &["`)`", "`,`"])?;
            return Ok(value);
        }
        let id = self.alloc_value();
        let items = self.parse_list(TokenKind::RParen, Self::parse_value)?;
        self.expect(TokenKind::RParen, &["`,`", "`)`"])?;
        let span = Span::new(open.span.start, self.prev_end());
        self.finish_value(id, span, BsnValue::Tuple(items));
        Ok(id)
    }

    /// Looks ahead for a comma at the top level of the parenthesized group the cursor is
    /// currently inside, which distinguishes `(v)` (grouping) from `(v,)` (a 1-tuple).
    ///
    /// Generic argument lists count as nesting too, so the separator in `(Pair<f32, f32>)`
    /// does not make the group look like a tuple. A `>` only closes an angle group when one
    /// is open, so a stray `>` cannot drive the depth negative and hide a real comma.
    fn paren_has_top_level_comma(&self) -> bool {
        let mut depth = 0u32;
        let mut angle_depth = 0u32;
        let mut index = self.pos;
        while let Some(token) = self.tokens.get(index) {
            match token.kind {
                TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace => depth += 1,
                TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace => {
                    if depth == 0 {
                        return false;
                    }
                    depth -= 1;
                }
                TokenKind::Lt => angle_depth += 1,
                TokenKind::Gt => angle_depth = angle_depth.saturating_sub(1),
                TokenKind::Comma if depth == 0 && angle_depth == 0 => return true,
                TokenKind::Eof => return false,
                _ => {}
            }
            index += 1;
        }
        false
    }

    /// `path_value = path [ struct_body | tuple_body ]`
    fn parse_path_value(&mut self) -> Result<BsnValueId, BsnParseError> {
        let path = self.parse_path()?;
        self.check_error_token()?;
        self.classify_path(&path, unsupported::PATH_CASE)?;
        self.parse_path_body(path)
    }

    /// `path = path_segment { "::" path_segment }`
    fn parse_path(&mut self) -> Result<BsnPath, BsnParseError> {
        self.nested(Self::parse_path_inner)
    }

    fn parse_path_inner(&mut self) -> Result<BsnPath, BsnParseError> {
        self.check_error_token()?;
        if self.peek() == TokenKind::ColonColon {
            return Err(BsnParseError::new(
                self.peek_token().span,
                BsnParseErrorKind::LeadingPathSeparator,
            ));
        }
        let mut segments = vec![self.parse_path_segment()?];
        while self.peek() == TokenKind::ColonColon {
            self.bump();
            segments.push(self.parse_path_segment()?);
        }
        let span = match (segments.first(), segments.last()) {
            (Some(first), Some(last)) => first.span.join(last.span),
            _ => Span::NONE,
        };
        Ok(BsnPath { segments, span })
    }

    /// `path_segment = IDENT [ "<" path { "," path } [ "," ] ">" ]`
    fn parse_path_segment(&mut self) -> Result<BsnPathSegment, BsnParseError> {
        let ident = self.expect(TokenKind::Ident, &["identifier"])?;
        let mut generics = Vec::new();
        if self.eat(TokenKind::Lt) {
            generics = self.parse_list(TokenKind::Gt, Self::parse_path)?;
            self.expect(TokenKind::Gt, &["`,`", "`>`"])?;
        }
        Ok(BsnPathSegment {
            ident: ident.span.text(self.source).to_string(),
            generics,
            span: Span::new(ident.span.start, self.prev_end()),
        })
    }

    /// Rejects paths that name Rust items an asset cannot reach: functions, constructors and
    /// constants. Casing is the same signal the `bsn!` macro uses.
    ///
    /// `lowercase` is the diagnostic for a lowercase-initial path that is not a constructor:
    /// [`unsupported::FN`] in entry position, [`unsupported::PATH_CASE`] in value position.
    fn classify_path(&self, path: &BsnPath, lowercase: &'static str) -> Result<(), BsnParseError> {
        let last = path.last_ident();
        let Some(first_char) = last.chars().next() else {
            return Ok(());
        };
        if first_char.is_uppercase() {
            if is_const_ident(last) {
                return Err(self.unsupported(unsupported::CONST, path.span));
            }
            return Ok(());
        }
        let previous_is_type = path.segments.len() >= 2
            && path.segments[path.segments.len() - 2]
                .ident
                .chars()
                .next()
                .is_some_and(char::is_uppercase);
        if previous_is_type {
            return Err(self.unsupported(unsupported::CTOR, path.span));
        }
        Err(self.unsupported(lowercase, path.span))
    }
}

/// Finds the earliest *repeat* of a field name in `fields`, as an index into `fields`.
///
/// This is the index the eager per-field scan would have flagged — the first position whose
/// name already appeared earlier — but found in `O(n log n)` rather than `O(n²)`, without a
/// hash map (the crate is `no_std` and its diagnostics must be deterministic).
fn first_duplicate_field(fields: &[(String, BsnValueId)]) -> Option<usize> {
    let mut order: Vec<usize> = (0..fields.len()).collect();
    order.sort_unstable_by(|a, b| fields[*a].0.cmp(&fields[*b].0).then(a.cmp(b)));
    // In name-then-index order, the later element of every adjacent equal-name pair is a
    // repeat, and every repeat shows up as one; the earliest of them is the reported one.
    order
        .windows(2)
        .filter(|pair| fields[pair[0]].0 == fields[pair[1]].0)
        .map(|pair| pair[1])
        .min()
}

/// Returns `true` if a flat entity body ends at `kind`.
fn is_entity_stop(kind: TokenKind, parenthesized: bool) -> bool {
    match kind {
        TokenKind::Eof | TokenKind::RParen => true,
        TokenKind::Comma | TokenKind::RBracket => !parenthesized,
        _ => false,
    }
}

/// The `bsn!` macro's constant heuristic: at least two characters and no lowercase letter.
fn is_const_ident(ident: &str) -> bool {
    ident.chars().count() > 1 && !ident.chars().any(char::is_lowercase)
}

/// A human-readable description of a token, used in "unexpected …" messages.
fn token_desc(kind: TokenKind) -> &'static str {
    match kind {
        TokenKind::Ident => "identifier",
        TokenKind::Int | TokenKind::Float => "number",
        TokenKind::Str => "string literal",
        TokenKind::ColonColon => "`::`",
        TokenKind::Colon => "`:`",
        TokenKind::Comma => "`,`",
        TokenKind::Hash => "`#`",
        TokenKind::At => "`@`",
        TokenKind::Tilde => "`~`",
        TokenKind::Minus => "`-`",
        TokenKind::Lt => "`<`",
        TokenKind::Gt => "`>`",
        TokenKind::LParen => "`(`",
        TokenKind::RParen => "`)`",
        TokenKind::LBrace => "`{`",
        TokenKind::RBrace => "`}`",
        TokenKind::LBracket => "`[`",
        TokenKind::RBracket => "`]`",
        TokenKind::Eof => "end of file",
        TokenKind::Error(_) => "invalid token",
    }
}
