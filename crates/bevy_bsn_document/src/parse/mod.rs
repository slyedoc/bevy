//! `.bsn` source text to the editor document.
//!
//! Parsing is `bevy_bsn`'s: this module lowers its arena [`BsnDocument`] into a
//! [`SceneBsnAst`]. The document holds `Children` as its only relation. A list whose
//! every item is a two-element tuple, `[(key, value), ...]`, is a map; any other tuple
//! is a list.

use bevy_bsn::{BsnDocument, BsnNodeId, BsnNodeKind, BsnPatchPrefix, BsnValueId, Span};
use bevy_ecs::entity::Entity;
use thiserror::Error;

use crate::document::{
    BsnField, BsnPatch, BsnPatches, BsnStructData, BsnStructFields, BsnTupleStructData, BsnValue,
    SceneBsnAst,
};

/// An error produced while parsing `.bsn` source text.
#[derive(Error, Debug)]
#[error("{line}:{column}: {message}")]
pub struct ParseError {
    /// 1-based line of the offending text.
    pub line: usize,
    /// 1-based column of the offending text.
    pub column: usize,
    /// What is wrong.
    pub message: String,
}

impl ParseError {
    fn at(text: &str, span: Span, message: impl Into<String>) -> Self {
        let (line, column) = span.line_col(text);
        Self {
            line: line as usize,
            column: column as usize,
            message: message.into(),
        }
    }
}

/// Parse `.bsn` source text into a document, one root per top-level entity.
pub fn parse_bsn(text: &str) -> Result<SceneBsnAst, ParseError> {
    let doc = bevy_bsn::parse(text)
        .map_err(|error| ParseError::at(text, error.span, error.to_string()))?;
    let mut lower = Lower {
        text,
        doc: &doc,
        ast: SceneBsnAst::default(),
    };
    for &root in &doc.roots {
        let entity = lower.entity(root)?;
        lower.ast.add_to_roots(entity);
    }
    Ok(lower.ast)
}

struct Lower<'a> {
    text: &'a str,
    doc: &'a BsnDocument,
    ast: SceneBsnAst,
}

impl Lower<'_> {
    fn error(&self, span: Span, message: impl Into<String>) -> ParseError {
        ParseError::at(self.text, span, message)
    }

    fn entity(&mut self, id: BsnNodeId) -> Result<Entity, ParseError> {
        let doc = self.doc;
        let node = doc.node(id).expect("parsed node ids are valid");
        let BsnNodeKind::Entity {
            name,
            base,
            patches,
            relations,
            ..
        } = &node.kind
        else {
            unreachable!("roots and relation items are entities");
        };
        let mut out = Vec::new();
        if let Some(base) = base {
            out.push(self.ast.world.spawn(BsnPatch::Base(base.clone())).id());
        }
        if let Some(name) = name {
            out.push(self.ast.world.spawn(BsnPatch::Name(name.clone())).id());
        }
        // Patches and relations interleave in source order.
        let mut entries: Vec<BsnNodeId> = patches.iter().chain(relations).copied().collect();
        entries.sort_by_key(|entry| doc.node(*entry).map_or(0, |node| node.span.start));
        for entry in entries {
            let patch = self.patch(entry)?;
            out.push(self.ast.world.spawn(patch).id());
        }
        Ok(self.ast.world.spawn(BsnPatches(out)).id())
    }

    fn patch(&mut self, id: BsnNodeId) -> Result<BsnPatch, ParseError> {
        let doc = self.doc;
        let node = doc.node(id).expect("parsed node ids are valid");
        match &node.kind {
            BsnNodeKind::Relation {
                target_symbol,
                entities,
            } => {
                if target_symbol.last_ident() != "Children" {
                    return Err(self.error(
                        node.span,
                        format!(
                            "`{}`: the document holds `Children` as its only relation",
                            target_symbol.to_type_path()
                        ),
                    ));
                }
                let children = entities
                    .iter()
                    .map(|&child| self.entity(child))
                    .collect::<Result<_, _>>()?;
                Ok(BsnPatch::Children(children))
            }
            BsnNodeKind::Patch {
                symbol,
                prefix,
                value,
            } => {
                let type_path = symbol.to_type_path();
                let value = &doc.value(*value).expect("parsed value ids are valid").value;
                match (prefix, value) {
                    (BsnPatchPrefix::SceneComponent, _) => Err(self.error(
                        node.span,
                        "`@` scene components are not supported in a document",
                    )),
                    (BsnPatchPrefix::Template, bevy_bsn::BsnValue::Path(_)) => {
                        Ok(BsnPatch::Template(type_path, None))
                    }
                    (BsnPatchPrefix::Template, bevy_bsn::BsnValue::Struct(_, fields)) => {
                        Ok(BsnPatch::Template(type_path, Some(self.fields(fields)?)))
                    }
                    (BsnPatchPrefix::Template, _) => Err(self.error(
                        node.span,
                        "a `~` template takes no body or a struct body",
                    )),
                    (_, bevy_bsn::BsnValue::Path(_)) => Ok(BsnPatch::Type(type_path)),
                    (_, bevy_bsn::BsnValue::Struct(_, fields)) => {
                        Ok(BsnPatch::Struct(BsnStructData {
                            type_path,
                            fields: self.fields(fields)?,
                        }))
                    }
                    (_, bevy_bsn::BsnValue::NamedTuple(_, items)) => {
                        Ok(BsnPatch::TupleStruct(BsnTupleStructData {
                            type_path,
                            values: self.values(items)?,
                        }))
                    }
                    _ => unreachable!("a patch value is a path, struct or named tuple"),
                }
            }
            BsnNodeKind::Entity { .. } => unreachable!("entity entries are patches or relations"),
        }
    }

    fn fields(&self, fields: &[(String, BsnValueId)]) -> Result<BsnStructFields, ParseError> {
        fields
            .iter()
            .map(|(name, value)| {
                Ok(BsnField {
                    name: name.clone(),
                    value: self.value(*value)?,
                })
            })
            .collect::<Result<_, _>>()
            .map(BsnStructFields)
    }

    fn values(&self, items: &[BsnValueId]) -> Result<Vec<BsnValue>, ParseError> {
        items.iter().map(|&item| self.value(item)).collect()
    }

    fn value(&self, id: BsnValueId) -> Result<BsnValue, ParseError> {
        let node = self.doc.value(id).expect("parsed value ids are valid");
        if let Some(pairs) = self.map_entries(&node.value) {
            return Ok(BsnValue::Map(pairs?));
        }
        Ok(match &node.value {
            bevy_bsn::BsnValue::Bool(b) => BsnValue::Bool(*b),
            bevy_bsn::BsnValue::Int(i) => BsnValue::Int(*i),
            bevy_bsn::BsnValue::Float(f) => BsnValue::Float(*f),
            bevy_bsn::BsnValue::String(s) => BsnValue::String(s.clone()),
            bevy_bsn::BsnValue::Path(path) => BsnValue::Type(path.to_type_path()),
            bevy_bsn::BsnValue::Struct(path, fields) => BsnValue::Struct(BsnStructData {
                type_path: path.to_type_path(),
                fields: self.fields(fields)?,
            }),
            bevy_bsn::BsnValue::NamedTuple(path, items) => {
                BsnValue::TupleStruct(BsnTupleStructData {
                    type_path: path.to_type_path(),
                    values: self.values(items)?,
                })
            }
            bevy_bsn::BsnValue::Tuple(items) | bevy_bsn::BsnValue::List(items) => {
                BsnValue::List(self.values(items)?)
            }
            bevy_bsn::BsnValue::Unit => BsnValue::List(Vec::new()),
            bevy_bsn::BsnValue::EntityRef(_) => {
                return Err(self.error(
                    node.span,
                    "`#Name` entity references are not supported in a document",
                ));
            }
        })
    }

    /// A non-empty list of two-element tuples, as map entries.
    fn map_entries(
        &self,
        value: &bevy_bsn::BsnValue,
    ) -> Option<Result<Vec<(BsnValue, BsnValue)>, ParseError>> {
        let bevy_bsn::BsnValue::List(items) = value else {
            return None;
        };
        let pairs: Vec<_> = items
            .iter()
            .map(|&item| match &self.doc.value(item)?.value {
                bevy_bsn::BsnValue::Tuple(pair) if pair.len() == 2 => Some((pair[0], pair[1])),
                _ => None,
            })
            .collect::<Option<_>>()?;
        if pairs.is_empty() {
            return None;
        }
        Some(
            pairs
                .into_iter()
                .map(|(key, value)| Ok((self.value(key)?, self.value(value)?)))
                .collect(),
        )
    }
}
