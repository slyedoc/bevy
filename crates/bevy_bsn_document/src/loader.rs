//! BSN text loader: parse `.bsn` text into a [`SceneBsnAst`].

use crate::document::SceneBsnAst;
use crate::parse::ParseError;

/// Errors that can occur when loading BSN text.
#[derive(Debug)]
pub enum BsnLoadError {
    /// The source text could not be parsed.
    Parse(ParseError),
    /// A referenced AST node was missing from the parsed world.
    NoAstNode,
}

impl std::fmt::Display for BsnLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BsnLoadError::Parse(err) => write!(f, "BSN parse error: {err}"),
            BsnLoadError::NoAstNode => write!(f, "No AST node found"),
        }
    }
}

impl std::error::Error for BsnLoadError {}

/// Parse BSN text into a document [`SceneBsnAst`], one root per top-level entity.
pub fn parse_bsn_text(text: &str) -> Result<SceneBsnAst, BsnLoadError> {
    crate::parse::parse_bsn(text).map_err(BsnLoadError::Parse)
}
