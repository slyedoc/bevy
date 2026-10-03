#![doc = include_str!("../README.md")]
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![doc(
    html_logo_url = "https://bevy.org/assets/icon.png",
    html_favicon_url = "https://bevy.org/assets/icon.png"
)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[cfg(test)]
mod adversarial;
mod ast;
mod error;
mod lexer;
mod parser;
mod printer;
#[cfg(test)]
mod tests;

pub use ast::{
    BsnDocument, BsnNode, BsnNodeId, BsnNodeKind, BsnPatchPrefix, BsnPath, BsnPathSegment,
    BsnValue, BsnValueId, BsnValueNode, PatchBody,
};
pub use error::{unsupported, BsnParseError, BsnParseErrorKind};
pub use lexer::{decode_float, decode_int, decode_string, LexError, Lexer, Span, Token, TokenKind};
pub use parser::{parse, MAX_NESTING_DEPTH};
pub use printer::{print_document, write_document, write_document_with, PrintOptions};
