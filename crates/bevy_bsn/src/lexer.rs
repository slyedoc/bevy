//! The hand-written, allocation-free lexer.
//!
//! [`Lexer`] turns BSN source text into a flat stream of [`Token`]s. Tokens are [`Copy`] and
//! own no data: the text of a token is recovered from its [`Span`], and literal values are
//! decoded on demand with [`decode_int`], [`decode_float`] and [`decode_string`].
//!
//! The lexer never fails. Lexical problems are reported as [`TokenKind::Error`] tokens that
//! always consume at least one character, so tokenizing always terminates.

use alloc::{string::String, vec::Vec};

use crate::error::{unsupported, BsnParseError, BsnParseErrorKind};

/// A half-open byte range into the source text.
///
/// `start` and `end` are byte offsets, so a span can be used to slice the original `&str`
/// directly (see [`Span::text`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Span {
    /// Byte offset of the first byte of the span.
    pub start: u32,
    /// Byte offset one past the last byte of the span.
    pub end: u32,
}

impl Span {
    /// The span of a synthesized node that has no source text.
    pub const NONE: Span = Span { start: 0, end: 0 };

    /// Creates a new span from a half-open byte range.
    pub const fn new(start: u32, end: u32) -> Self {
        Span { start, end }
    }

    /// Returns `true` if this is the [`Span::NONE`] placeholder.
    pub const fn is_none(&self) -> bool {
        self.start == 0 && self.end == 0
    }

    /// The 1-based `(line, column)` of [`Span::start`] within `source`.
    ///
    /// Columns count `char`s, not bytes. An out-of-bounds span reports the position of the
    /// end of `source`.
    pub fn line_col(&self, source: &str) -> (u32, u32) {
        let offset = (self.start as usize).min(source.len());
        let mut line = 1u32;
        let mut col = 1u32;
        for (index, ch) in source.char_indices() {
            if index >= offset {
                break;
            }
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        (line, col)
    }

    /// The exact source text covered by this span, or `""` if the span is out of bounds or
    /// does not fall on `char` boundaries.
    pub fn text<'s>(&self, source: &'s str) -> &'s str {
        source
            .get(self.start as usize..self.end as usize)
            .unwrap_or("")
    }

    /// The smallest span covering both `self` and `other`.
    pub fn join(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// A single lexical token: a [`TokenKind`] plus the [`Span`] it covers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Token {
    /// What kind of token this is.
    pub kind: TokenKind,
    /// The byte range the token covers in the source text.
    pub span: Span,
}

/// The kind of a [`Token`].
///
/// This is a lower-level API than [`crate::parse`]; it exists for tooling such as syntax
/// highlighters and language servers. It is `#[non_exhaustive]` because new token kinds may
/// be added in a minor release.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum TokenKind {
    /// An identifier or keyword-like word, such as `Transform`, `true`, `inf` or `use`.
    Ident,
    /// An integer literal: `12`, `0xFF`, `1_000`, `0b1010`, `0o17`.
    Int,
    /// A floating-point literal: `1.0`, `1.`, `1e5`, `2.5E-3`.
    Float,
    /// A string literal. The span **includes** the quotes and any raw-string delimiters.
    Str,
    /// `::`
    ColonColon,
    /// `:`
    Colon,
    /// `,`
    Comma,
    /// `#`
    Hash,
    /// `@`
    At,
    /// `~`
    Tilde,
    /// `-`
    Minus,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// The end of the source text. Its span is `(len, len)`.
    Eof,
    /// A lexical problem. The parser turns this into a [`BsnParseError`] verbatim.
    Error(LexError),
}

/// A lexical problem, carried by [`TokenKind::Error`].
///
/// `#[non_exhaustive]` because new lexical diagnostics may be added in a minor release.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum LexError {
    /// A string literal was not closed before the end of the file.
    UnterminatedString,
    /// A `/* … */` comment was not closed before the end of the file.
    UnterminatedBlockComment,
    /// A raw string literal was not closed before the end of the file.
    UnterminatedRawString,
    /// An unknown or malformed escape sequence inside a string literal.
    InvalidEscape,
    /// A numeric literal that cannot be interpreted, such as `0x` with no digits.
    InvalidNumber,
    /// A numeric literal with a type suffix, such as `1u8` or `1.0f32`.
    NumericSuffix,
    /// A raw identifier, such as `r#type`.
    RawIdentifier,
    /// A character literal, such as `'a'`.
    CharLiteral,
    /// A `|`, which can only start a closure.
    Closure,
    /// A `!`, which can only be a macro invocation.
    Macro,
    /// A character that has no meaning in BSN.
    Unknown,
}

/// The BSN lexer.
///
/// Create one with [`Lexer::new`] and pull tokens with [`Lexer::next_token`], or lex a whole
/// document at once with [`Lexer::tokenize`].
pub struct Lexer<'src> {
    source: &'src str,
    pos: usize,
}

impl<'src> Lexer<'src> {
    /// Creates a lexer over `source`, skipping a leading UTF-8 BOM (`\u{FEFF}`) if present.
    pub fn new(source: &'src str) -> Self {
        let pos = if source.starts_with('\u{FEFF}') { 3 } else { 0 };
        Lexer { source, pos }
    }

    /// Lexes `source` completely, returning every token followed by exactly one
    /// [`TokenKind::Eof`].
    pub fn tokenize(source: &'src str) -> Vec<Token> {
        let mut lexer = Lexer::new(source);
        let mut tokens = Vec::new();
        loop {
            let token = lexer.next_token();
            tokens.push(token);
            if token.kind == TokenKind::Eof {
                return tokens;
            }
        }
    }

    /// Produces the next token.
    ///
    /// Never fails: lexical problems become [`TokenKind::Error`] tokens. Once the source is
    /// exhausted this returns [`TokenKind::Eof`] forever.
    pub fn next_token(&mut self) -> Token {
        if let Some(error) = self.skip_trivia() {
            return error;
        }
        let start = self.pos;
        let Some(ch) = self.first() else {
            let end = self.source.len() as u32;
            return Token {
                kind: TokenKind::Eof,
                span: Span::new(end, end),
            };
        };

        let kind = match ch {
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '{' => TokenKind::LBrace,
            '}' => TokenKind::RBrace,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            ',' => TokenKind::Comma,
            '#' => TokenKind::Hash,
            '@' => TokenKind::At,
            '~' => TokenKind::Tilde,
            '<' => TokenKind::Lt,
            '>' => TokenKind::Gt,
            '-' => TokenKind::Minus,
            '\'' => TokenKind::Error(LexError::CharLiteral),
            '|' => TokenKind::Error(LexError::Closure),
            '!' => TokenKind::Error(LexError::Macro),
            ':' => {
                self.pos += 1;
                if self.first() == Some(':') {
                    self.pos += 1;
                    return self.token(TokenKind::ColonColon, start);
                }
                return self.token(TokenKind::Colon, start);
            }
            '"' => return self.string(start),
            'r' if matches!(self.second(), Some('"') | Some('#')) => {
                return self.raw_prefixed(start)
            }
            c if c.is_ascii_digit() => return self.number(start),
            c if is_ident_start(c) => {
                self.eat_while(is_ident_continue);
                return self.token(TokenKind::Ident, start);
            }
            _ => TokenKind::Error(LexError::Unknown),
        };
        self.pos += ch.len_utf8();
        self.token(kind, start)
    }

    /// Skips whitespace and comments. Returns an error token if a block comment is unclosed.
    fn skip_trivia(&mut self) -> Option<Token> {
        loop {
            let ch = self.first()?;
            if ch.is_whitespace() {
                self.pos += ch.len_utf8();
                continue;
            }
            if ch == '/' {
                match self.second() {
                    Some('/') => {
                        self.pos += 2;
                        while let Some(c) = self.first() {
                            self.pos += c.len_utf8();
                            if c == '\n' {
                                break;
                            }
                        }
                        continue;
                    }
                    Some('*') => {
                        let start = self.pos;
                        self.pos += 2;
                        let mut depth = 1usize;
                        while depth > 0 {
                            match (self.first(), self.second()) {
                                (Some('/'), Some('*')) => {
                                    self.pos += 2;
                                    depth += 1;
                                }
                                (Some('*'), Some('/')) => {
                                    self.pos += 2;
                                    depth -= 1;
                                }
                                (Some(c), _) => self.pos += c.len_utf8(),
                                (None, _) => {
                                    return Some(Token {
                                        kind: TokenKind::Error(LexError::UnterminatedBlockComment),
                                        span: Span::new(start as u32, self.source.len() as u32),
                                    });
                                }
                            }
                        }
                        continue;
                    }
                    _ => return None,
                }
            }
            return None;
        }
    }

    /// Lexes a normal (non-raw) string literal, starting at the opening quote.
    fn string(&mut self, start: usize) -> Token {
        self.pos += 1;
        loop {
            let escape_start = self.pos;
            let Some(ch) = self.bump_char() else {
                return Token {
                    kind: TokenKind::Error(LexError::UnterminatedString),
                    span: Span::new(start as u32, self.source.len() as u32),
                };
            };
            match ch {
                '"' => return self.token(TokenKind::Str, start),
                '\\' => {
                    let valid = self.escape();
                    if !valid {
                        return Token {
                            kind: TokenKind::Error(LexError::InvalidEscape),
                            span: Span::new(escape_start as u32, self.pos as u32),
                        };
                    }
                }
                _ => {}
            }
        }
    }

    /// Consumes one escape sequence (after the backslash). Returns `false` if it is invalid.
    fn escape(&mut self) -> bool {
        let Some(ch) = self.bump_char() else {
            return false;
        };
        match ch {
            '\\' | '"' | '\'' | 'n' | 'r' | 't' | '0' => true,
            'x' => {
                let mut value = 0u32;
                for _ in 0..2 {
                    let Some(digit) = self.first().and_then(|c| c.to_digit(16)) else {
                        return false;
                    };
                    self.pos += 1;
                    value = value * 16 + digit;
                }
                value <= 0x7F
            }
            'u' => {
                if self.first() != Some('{') {
                    return false;
                }
                self.pos += 1;
                let mut value = 0u32;
                let mut digits = 0;
                while let Some(digit) = self.first().and_then(|c| c.to_digit(16)) {
                    self.pos += 1;
                    value = value.saturating_mul(16).saturating_add(digit);
                    digits += 1;
                    if digits > 6 {
                        return false;
                    }
                }
                if digits == 0 || self.first() != Some('}') {
                    return false;
                }
                self.pos += 1;
                char::from_u32(value).is_some()
            }
            _ => false,
        }
    }

    /// Lexes `r"…"`, `r#"…"#` or the rejected raw identifier `r#name`.
    fn raw_prefixed(&mut self, start: usize) -> Token {
        // `self.first()` is `r`, `self.second()` is `"` or `#`.
        let mut probe = self.pos + 1;
        let mut hashes = 0usize;
        while self.source[probe..].starts_with('#') {
            probe += 1;
            hashes += 1;
        }
        if !self.source[probe..].starts_with('"') {
            // `r#name` — a raw identifier. Consume `r#` so the lexer makes progress.
            self.pos += 2;
            return Token {
                kind: TokenKind::Error(LexError::RawIdentifier),
                span: Span::new(start as u32, self.pos as u32),
            };
        }
        self.pos = probe + 1;
        loop {
            let Some(ch) = self.bump_char() else {
                return Token {
                    kind: TokenKind::Error(LexError::UnterminatedRawString),
                    span: Span::new(start as u32, self.source.len() as u32),
                };
            };
            if ch == '"' {
                let mut seen = 0usize;
                while seen < hashes && self.source[self.pos..].starts_with('#') {
                    self.pos += 1;
                    seen += 1;
                }
                if seen == hashes {
                    return self.token(TokenKind::Str, start);
                }
            }
        }
    }

    /// Lexes an integer or float literal.
    fn number(&mut self, start: usize) -> Token {
        let mut is_float = false;
        let radix_prefixed =
            self.first() == Some('0') && matches!(self.second(), Some('x') | Some('o') | Some('b'));
        if radix_prefixed {
            let radix = match self.second() {
                Some('x') => 16,
                Some('o') => 8,
                _ => 2,
            };
            self.pos += 2;
            let mut any = false;
            self.eat_while(|c| {
                let digit = c.is_digit(radix);
                any |= digit;
                digit || c == '_'
            });
            // A digit that is out of range for the radix (the `2` in `0b12`) would otherwise
            // start a *second* number token, turning one bad literal into two good ones. An
            // alphabetic tail is not split — it is a numeric suffix, reported below.
            let out_of_radix_digit =
                |c: char| c.is_alphanumeric() && !is_ident_start(c) && !c.is_digit(radix);
            if !any || self.first().is_some_and(out_of_radix_digit) {
                self.eat_ident_tail();
                return Token {
                    kind: TokenKind::Error(LexError::InvalidNumber),
                    span: Span::new(start as u32, self.pos as u32),
                };
            }
        } else {
            self.eat_digits();
            if self.first() == Some('.') && self.second() != Some('.') {
                is_float = true;
                self.pos += 1;
                self.eat_digits();
            }
            if matches!(self.first(), Some('e') | Some('E')) {
                let mut probe = self.pos + 1;
                if self.source[probe..].starts_with(['+', '-']) {
                    probe += 1;
                }
                if self.source[probe..].starts_with(|c: char| c.is_ascii_digit()) {
                    self.pos = probe;
                    self.eat_digits();
                    is_float = true;
                }
            }
        }
        if self.first().is_some_and(is_ident_start) {
            self.eat_ident_tail();
            return Token {
                kind: TokenKind::Error(LexError::NumericSuffix),
                span: Span::new(start as u32, self.pos as u32),
            };
        }
        let kind = if is_float {
            TokenKind::Float
        } else {
            TokenKind::Int
        };
        self.token(kind, start)
    }

    /// Consumes characters while `accept` returns `true`.
    fn eat_while(&mut self, mut accept: impl FnMut(char) -> bool) {
        while let Some(c) = self.first() {
            if !accept(c) {
                break;
            }
            self.pos += c.len_utf8();
        }
    }

    /// Consumes ASCII digits and `_` separators.
    fn eat_digits(&mut self) {
        self.eat_while(|c| c.is_ascii_digit() || c == '_');
    }

    /// Consumes an identifier-shaped run of characters (used to widen error spans).
    fn eat_ident_tail(&mut self) {
        self.eat_while(is_ident_continue);
    }

    /// Builds a token spanning `start..self.pos`.
    fn token(&self, kind: TokenKind, start: usize) -> Token {
        Token {
            kind,
            span: Span::new(start as u32, self.pos as u32),
        }
    }

    fn first(&self) -> Option<char> {
        self.source[self.pos..].chars().next()
    }

    fn second(&self) -> Option<char> {
        self.source[self.pos..].chars().nth(1)
    }

    fn bump_char(&mut self) -> Option<char> {
        let ch = self.first()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }
}

/// Returns `true` if `c` may start an identifier.
fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

/// Returns `true` if `c` may continue an identifier.
fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Decodes an integer literal token.
///
/// Handles the `0x`, `0o` and `0b` prefixes and strips `_` separators. Returns
/// [`BsnParseErrorKind::NumberOutOfRange`] if the value does not fit in an `i128`.
///
/// A literal token never carries a sign — `-1` lexes as [`TokenKind::Minus`] followed by
/// [`TokenKind::Int`] — so this decodes the *magnitude*, and the magnitude of `i128::MIN`
/// (`2^127`) is one past `i128::MAX`. Decoding the text of a negated `i128::MIN` therefore
/// reports `NumberOutOfRange` here; the parser negates via `decode_int_magnitude` instead,
/// so `-170141183460469231731687303715884105728` in a source file is accepted.
pub fn decode_int(source: &str, span: Span) -> Result<i128, BsnParseError> {
    let magnitude = decode_int_magnitude(source, span)?;
    i128::try_from(magnitude)
        .map_err(|_| BsnParseError::new(span, BsnParseErrorKind::NumberOutOfRange))
}

/// Decodes an integer literal token as an unsigned magnitude.
///
/// Same syntax as [`decode_int`], but the result is only bounded by `u128`, which lets the
/// parser's negation path reach `i128::MIN` (whose magnitude is `i128::MAX as u128 + 1`).
pub(crate) fn decode_int_magnitude(source: &str, span: Span) -> Result<u128, BsnParseError> {
    let text = span.text(source);
    let (radix, digits) = if let Some(rest) = text.strip_prefix("0x") {
        (16, rest)
    } else if let Some(rest) = text.strip_prefix("0o") {
        (8, rest)
    } else if let Some(rest) = text.strip_prefix("0b") {
        (2, rest)
    } else {
        (10, text)
    };
    let mut cleaned = String::with_capacity(digits.len());
    cleaned.extend(digits.chars().filter(|c| *c != '_'));
    if cleaned.is_empty() {
        return Err(BsnParseError::new(span, BsnParseErrorKind::InvalidNumber));
    }
    u128::from_str_radix(&cleaned, radix)
        .map_err(|_| BsnParseError::new(span, BsnParseErrorKind::NumberOutOfRange))
}

/// Decodes a floating-point literal token, stripping `_` separators.
pub fn decode_float(source: &str, span: Span) -> Result<f64, BsnParseError> {
    let text = span.text(source);
    let mut cleaned = String::with_capacity(text.len());
    cleaned.extend(text.chars().filter(|c| *c != '_'));
    cleaned
        .parse::<f64>()
        .map_err(|_| BsnParseError::new(span, BsnParseErrorKind::InvalidNumber))
}

/// Decodes a string literal token, removing the delimiters and resolving escapes.
///
/// Raw strings (`r"…"`, `r#"…"#`) are returned verbatim, without escape processing.
pub fn decode_string(source: &str, span: Span) -> Result<String, BsnParseError> {
    let text = span.text(source);
    if let Some(rest) = text.strip_prefix('r') {
        let hashes = rest.chars().take_while(|c| *c == '#').count();
        let body = rest
            .get(hashes + 1..rest.len().saturating_sub(hashes + 1))
            .unwrap_or("");
        return Ok(String::from(body));
    }
    let body = text
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or("");
    let mut out = String::with_capacity(body.len());
    let mut chars = body.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        let Some(escape) = chars.next() else {
            return Err(BsnParseError::new(span, BsnParseErrorKind::InvalidEscape));
        };
        match escape {
            '\\' => out.push('\\'),
            '"' => out.push('"'),
            '\'' => out.push('\''),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            '0' => out.push('\0'),
            'x' => {
                let mut value = 0u32;
                for _ in 0..2 {
                    let Some(digit) = chars.next().and_then(|c| c.to_digit(16)) else {
                        return Err(BsnParseError::new(span, BsnParseErrorKind::InvalidEscape));
                    };
                    value = value * 16 + digit;
                }
                let Some(decoded) = char::from_u32(value) else {
                    return Err(BsnParseError::new(span, BsnParseErrorKind::InvalidEscape));
                };
                out.push(decoded);
            }
            'u' => {
                if chars.next() != Some('{') {
                    return Err(BsnParseError::new(span, BsnParseErrorKind::InvalidEscape));
                }
                let mut value = 0u32;
                let mut closed = false;
                for c in chars.by_ref() {
                    if c == '}' {
                        closed = true;
                        break;
                    }
                    let Some(digit) = c.to_digit(16) else {
                        return Err(BsnParseError::new(span, BsnParseErrorKind::InvalidEscape));
                    };
                    value = value.saturating_mul(16).saturating_add(digit);
                }
                let decoded = if closed { char::from_u32(value) } else { None };
                let Some(decoded) = decoded else {
                    return Err(BsnParseError::new(span, BsnParseErrorKind::InvalidEscape));
                };
                out.push(decoded);
            }
            _ => return Err(BsnParseError::new(span, BsnParseErrorKind::InvalidEscape)),
        }
    }
    Ok(out)
}

/// Maps a [`LexError`] to the parse error it produces at `span`.
pub(crate) fn lex_error_to_parse_error(error: LexError, span: Span, source: &str) -> BsnParseError {
    let kind = match error {
        LexError::UnterminatedString | LexError::UnterminatedRawString => {
            BsnParseErrorKind::UnterminatedString
        }
        LexError::UnterminatedBlockComment => BsnParseErrorKind::UnterminatedBlockComment,
        LexError::InvalidEscape => BsnParseErrorKind::InvalidEscape,
        LexError::InvalidNumber => BsnParseErrorKind::InvalidNumber,
        LexError::NumericSuffix => BsnParseErrorKind::Unsupported(unsupported::SUFFIX),
        LexError::RawIdentifier => BsnParseErrorKind::Unsupported(unsupported::RAW_IDENT),
        LexError::CharLiteral => BsnParseErrorKind::Unsupported(unsupported::CHAR),
        LexError::Closure => BsnParseErrorKind::Unsupported(unsupported::CLOSURE),
        LexError::Macro => BsnParseErrorKind::Unsupported(unsupported::MACRO),
        LexError::Unknown => {
            let ch = span.text(source).chars().next().unwrap_or('\u{FFFD}');
            BsnParseErrorKind::UnknownCharacter(ch)
        }
    };
    BsnParseError::new(span, kind)
}
