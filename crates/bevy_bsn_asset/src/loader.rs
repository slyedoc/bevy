//! The `.bsn` [`AssetLoader`]: reading a scene asset file, parsing it with
//! [`bevy_bsn`](bevy_bsn::parse), lowering it with [`DynamicScene::from_document`], and handing
//! the result to [`ScenePatch::load_with`] so that its asset dependencies are registered.
//!
//! Every failure is an error carrying `path:line:column`, never a panic: `.bsn` files are user
//! data, and a malformed one must not take the app down.

use alloc::{
    string::{String, ToString},
    vec::Vec,
};

use bevy_asset::{io::Reader, AssetLoadFailedEvent, AssetLoader, AssetPath, LoadContext};
use bevy_bsn::{BsnDocument, BsnNodeKind, BsnParseError};
use bevy_ecs::{
    message::MessageReader,
    reflect::AppTypeRegistry,
    world::{FromWorld, World},
};
use bevy_reflect::TypePath;
use thiserror::Error;
use tracing::error;

use crate::{DynamicScene, DynamicSceneBuildError};
use bevy_scene::ScenePatch;

/// An [`AssetLoader`] for `.bsn` files, producing a [`ScenePatch`].
///
/// Registered automatically by [`BsnAssetPlugin`](crate::BsnAssetPlugin) when the `bsn_asset` cargo
/// feature is enabled (it is on by default), so
/// `asset_server.load::<ScenePatch>("scenes/player.bsn")` works out of the box.
///
/// Every type named in a `.bsn` file must be registered in the [`AppTypeRegistry`]
/// (`#[derive(Reflect)]` types are registered automatically when the `reflect_auto_register`
/// feature is on; otherwise call [`App::register_type`](bevy_app::App)). The registry handle is
/// cloned from the world when the loader is created, so types registered *after*
/// [`BsnAssetPlugin`](crate::BsnAssetPlugin) is built are still visible to it.
#[derive(TypePath)]
pub struct DynamicBsnLoader {
    /// A clone of the app's [`AppTypeRegistry`]. Cloning aliases the same lock, so later
    /// registrations are picked up.
    type_registry: AppTypeRegistry,
}

impl core::fmt::Debug for DynamicBsnLoader {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DynamicBsnLoader").finish_non_exhaustive()
    }
}

impl DynamicBsnLoader {
    /// Creates a loader that resolves `.bsn` type paths against `type_registry`.
    ///
    /// Prefer letting [`BsnAssetPlugin`](crate::BsnAssetPlugin) register the loader; this constructor
    /// exists for apps that build their schedules by hand.
    pub fn new(type_registry: AppTypeRegistry) -> Self {
        Self { type_registry }
    }
}

impl FromWorld for DynamicBsnLoader {
    fn from_world(world: &mut World) -> Self {
        DynamicBsnLoader {
            type_registry: world.resource::<AppTypeRegistry>().clone(),
        }
    }
}

impl AssetLoader for DynamicBsnLoader {
    type Asset = ScenePatch;
    type Settings = ();
    type Error = DynamicBsnLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let path = load_context.path().to_string();

        // (1) Bytes. This is the only `.await` in the function; nothing that follows may hold a
        //     registry guard across a suspension point.
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .await
            .map_err(|source| DynamicBsnLoaderError::Io {
                path: path.clone(),
                source,
            })?;

        // (2) UTF-8, with the BOM policy of `decode_bsn_source`.
        let source = decode_bsn_source(&bytes, &path)?;

        // (3) Parse.
        let document = bevy_bsn::parse(source).map_err(|error| {
            let (line, column) = error.span.line_col(source);
            DynamicBsnLoaderError::Parse {
                path: path.clone(),
                line,
                column,
                message: parse_error_message(&error),
            }
        })?;

        // (4) The one document-level check the builder cannot make, because it does not know our
        //     path. (The single-root rule *is* checked by the builder, which reports it with a
        //     span that `from_build_error` turns into `path:line:column`.)
        check_no_self_include(&document, source, &path, load_context.path())?;

        // (5) Lower the document. `source` here is the asset path string: it gives the document's
        //     `#Name` references a stable identity across reloads of the same file.
        let scene = DynamicScene::from_document(&document, path.clone(), &self.type_registry)
            .map_err(|error| DynamicBsnLoaderError::from_build_error(&path, source, error))?;

        // (6) Register the scene's asset dependencies and build the asset value. Dependencies are
        //     registered exactly once, here, so `ScenePatch::dependencies` stays authoritative.
        Ok(ScenePatch::load_with(load_context, scene))
    }

    fn extensions(&self) -> &[&str] {
        &["bsn"]
    }
}

/// Renders a [`BsnParseError`] as a single line, appending its "expected" list when it has one.
fn parse_error_message(error: &BsnParseError) -> String {
    let suffix = error.expected_suffix();
    if suffix.is_empty() {
        error.to_string()
    } else {
        format!("{error};{suffix}")
    }
}

/// Rejects `a.bsn` whose base is `a.bsn`.
///
/// Such a file could never finish loading: its own recursive-dependency state can never reach
/// `Loaded`, so it would never emit `LoadedWithDependencies` and every entity waiting on it would
/// hang silently.
///
/// The comparison is on the literal [`AssetPath`], which catches the common copy-paste mistake
/// (including source-qualified spellings, because `AssetPath` parses `source://path#label` on both
/// sides). Non-canonical spellings of the same file (`"./a.bsn"`) and cycles spanning several
/// files are not detected.
fn check_no_self_include(
    document: &BsnDocument,
    source: &str,
    path: &str,
    own_path: &AssetPath<'static>,
) -> Result<(), DynamicBsnLoaderError> {
    for node in document.entities() {
        let BsnNodeKind::Entity {
            base: Some(base),
            base_span,
            ..
        } = &node.kind
        else {
            continue;
        };
        // `bevy_bsn` knows nothing about `bevy_asset`, so a base is a plain `String` there; the
        // loader is the layer that gives it asset-path meaning. An unparsable path is left to the
        // builder, which reports it as `InvalidAssetPath` with a span.
        let Ok(base_path) = AssetPath::try_parse(base.as_str()) else {
            continue;
        };
        if base_path == *own_path {
            let (line, column) = base_span.unwrap_or(node.span).line_col(source);
            return Err(DynamicBsnLoaderError::SelfInclude {
                path: path.to_string(),
                line,
                column,
            });
        }
    }
    Ok(())
}

/// Decodes `.bsn` source bytes as UTF-8, stripping a leading UTF-8 byte order mark.
///
/// `.bsn` files are UTF-8. A UTF-8 BOM is silently stripped (Windows editors emit one) and every
/// span is computed against the *stripped* source, so a diagnostic on line 1 reports the column the
/// user sees in their editor. UTF-16 BOMs get a dedicated message rather than a confusing
/// "invalid byte sequence at offset 1". No other encodings are accepted, and no `\r\n`
/// normalization is performed (the lexer treats `\r` as whitespace).
fn decode_bsn_source<'a>(bytes: &'a [u8], path: &str) -> Result<&'a str, DynamicBsnLoaderError> {
    if bytes.starts_with(&[0xFF, 0xFE]) || bytes.starts_with(&[0xFE, 0xFF]) {
        return Err(DynamicBsnLoaderError::UnsupportedEncoding {
            path: path.to_string(),
        });
    }
    let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    core::str::from_utf8(bytes).map_err(|error| DynamicBsnLoaderError::InvalidUtf8 {
        path: path.to_string(),
        valid_up_to: error.valid_up_to(),
    })
}

/// Errors produced by [`DynamicBsnLoader`].
///
/// Messages are formatted as `path:line:column: message` so that editors and terminals can jump to
/// the offending location. Lines and columns are 1-based; columns count `char`s.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DynamicBsnLoaderError {
    /// The asset bytes could not be read.
    #[error("{path}: failed to read `.bsn` file: {source}")]
    Io {
        /// The asset path that was being loaded.
        path: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The file is not valid UTF-8.
    #[error("{path}: `.bsn` files must be UTF-8; invalid byte sequence at offset {valid_up_to}")]
    InvalidUtf8 {
        /// The asset path that was being loaded.
        path: String,
        /// How many bytes were valid UTF-8 before the offending sequence.
        valid_up_to: usize,
    },

    /// The file begins with a UTF-16 byte order mark.
    #[error(
        "{path}: `.bsn` files must be UTF-8, but this file starts with a UTF-16 byte order mark"
    )]
    UnsupportedEncoding {
        /// The asset path that was being loaded.
        path: String,
    },

    /// The file could not be parsed.
    #[error("{path}:{line}:{column}: {message}")]
    Parse {
        /// The asset path that was being loaded.
        path: String,
        /// 1-based line of the offending text.
        line: u32,
        /// 1-based, `char`-counted column of the offending text.
        column: u32,
        /// The parser's rendered message.
        message: String,
    },

    /// The parsed document could not be turned into a scene (unregistered type, missing
    /// `ReflectDefault`, unknown field, un-coercible value, ...).
    #[error("{path}:{line}:{column}: {source}")]
    Build {
        /// The asset path that was being loaded.
        path: String,
        /// 1-based line of the offending text.
        line: u32,
        /// 1-based, `char`-counted column of the offending text.
        column: u32,
        /// The underlying build error.
        #[source]
        source: DynamicSceneBuildError,
    },

    /// As [`Self::Build`], for errors whose span does not point at any source text.
    #[error("{path}: {source}")]
    BuildNoSpan {
        /// The asset path that was being loaded.
        path: String,
        /// The underlying build error.
        #[source]
        source: DynamicSceneBuildError,
    },

    /// The file inherits from itself (`a.bsn` containing `:"a.bsn"`).
    #[error(
        "{path}:{line}:{column}: this `.bsn` file inherits from itself. A scene cannot be its own \
         base; remove the `:\"{path}\"` include."
    )]
    SelfInclude {
        /// The asset path that was being loaded.
        path: String,
        /// 1-based line of the offending include.
        line: u32,
        /// 1-based, `char`-counted column of the offending include.
        column: u32,
    },
}

impl DynamicBsnLoaderError {
    /// Wraps a [`DynamicSceneBuildError`] with the file it came from, resolving its span to a
    /// line and column within `source`.
    fn from_build_error(path: &str, source: &str, error: DynamicSceneBuildError) -> Self {
        if error.span().is_none() {
            return Self::BuildNoSpan {
                path: path.to_string(),
                source: error,
            };
        }
        let (line, column) = error.span().line_col(source);
        Self::Build {
            path: path.to_string(),
            line,
            column,
            source: error,
        }
    }
}

/// Logs an error for every [`ScenePatch`] asset that fails to load, including `.bsn` parse and
/// resolution diagnostics.
///
/// Without this, a `.bsn` that fails to load is reported once by the asset server and then every
/// [`ScenePatchInstance`](bevy_scene::ScenePatchInstance) pointing at it waits forever with no
/// indication that *scene spawning* is what broke.
///
/// Added by [`BsnAssetPlugin`](crate::BsnAssetPlugin) when the `bsn_asset` feature is enabled. The
/// [`MessageReader`] cursor means each failure is logged exactly once, not once per frame; a retry
/// or a hot-reload attempt emits a new event and is logged again, which is intended.
pub fn report_scene_patch_load_failures(
    mut failures: MessageReader<AssetLoadFailedEvent<ScenePatch>>,
) {
    for failure in failures.read() {
        error!(
            "Failed to load scene asset \"{}\": {}. Entities whose `ScenePatchInstance` points at \
             this asset will never be spawned.",
            failure.path, failure.error
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_strips_utf8_bom() {
        let bytes = b"\xEF\xBB\xBFHello";
        assert_eq!(decode_bsn_source(bytes, "a.bsn").unwrap(), "Hello");
    }

    #[test]
    fn decode_rejects_utf16_bom() {
        let error = decode_bsn_source(b"\xFF\xFENothing", "a.bsn").unwrap_err();
        assert!(
            matches!(error, DynamicBsnLoaderError::UnsupportedEncoding { .. }),
            "unexpected error: {error}"
        );
        assert!(
            error.to_string().contains("must be UTF-8"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn decode_rejects_invalid_utf8() {
        let error = decode_bsn_source(b"ok\xC3", "a.bsn").unwrap_err();
        let DynamicBsnLoaderError::InvalidUtf8 { valid_up_to, .. } = error else {
            panic!("unexpected error: {error}");
        };
        assert_eq!(valid_up_to, 2);
    }

    #[test]
    fn parse_error_message_is_file_line_column() {
        let error = DynamicBsnLoaderError::Parse {
            path: "scenes/x.bsn".to_string(),
            line: 12,
            column: 5,
            message: "expected `}`".to_string(),
        };
        assert_eq!(error.to_string(), "scenes/x.bsn:12:5: expected `}`");
    }

    /// Lowers `source` the way [`DynamicBsnLoader::load`] does, and wraps any build error with the
    /// file it came from. An empty registry is enough: document-shape errors are reported before
    /// any type is resolved.
    fn load_errors(path: &'static str, source: &str) -> DynamicBsnLoaderError {
        let document = bevy_bsn::parse(source).expect("the fixture should parse");
        let error = DynamicScene::from_document(&document, path, &AppTypeRegistry::default())
            .expect_err("the document should be rejected");
        DynamicBsnLoaderError::from_build_error(path, source, error)
    }

    #[test]
    fn multiple_roots_error_mentions_count() {
        // Multi-root documents are rejected by the builder; the loader is what turns the span into
        // a `path:line:column` prefix.
        let message = load_errors("scenes/x.bsn", "(Foo),\n(Bar),\n(Baz)").to_string();
        assert!(
            message.contains("exactly one root entity"),
            "unexpected message: {message}"
        );
        assert!(message.contains("found 3"), "unexpected message: {message}");
        assert!(
            message.starts_with("scenes/x.bsn:2:"),
            "the message should locate the second root: {message}"
        );
    }

    #[test]
    fn self_include_error_mentions_the_path() {
        let error = DynamicBsnLoaderError::SelfInclude {
            path: "scenes/a.bsn".to_string(),
            line: 1,
            column: 1,
        };
        let message = error.to_string();
        assert!(
            message.contains("inherits from itself"),
            "unexpected message: {message}"
        );
        assert!(
            message.contains("scenes/a.bsn"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn loader_extensions() {
        let loader = DynamicBsnLoader::new(AppTypeRegistry::default());
        assert_eq!(loader.extensions(), &["bsn"]);
    }

    #[test]
    fn parse_message_appends_expected_list() {
        let error = bevy_bsn::parse("Foo { x: 1 ").unwrap_err();
        let message = parse_error_message(&error);
        assert!(
            message.contains("expected"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn build_error_without_span_renders_without_line_and_column() {
        let error = DynamicSceneBuildError::MalformedDocument {
            message: "dangling node id".to_string(),
            span: bevy_bsn::Span::NONE,
        };
        let error = DynamicBsnLoaderError::from_build_error("a.bsn", "", error);
        assert!(
            matches!(error, DynamicBsnLoaderError::BuildNoSpan { .. }),
            "unexpected error: {error}"
        );
        assert!(
            error.to_string().starts_with("a.bsn: "),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn single_root_check_accepts_one_root_and_rejects_two() {
        // A one-root document is never rejected *for its root count* (it still fails later here,
        // on the unregistered `Foo`, because the fixture registry is empty).
        let error = load_errors("a.bsn", "Foo\n");
        assert!(
            !matches!(
                error,
                DynamicBsnLoaderError::Build {
                    source: DynamicSceneBuildError::MultipleRoots { .. },
                    ..
                }
            ),
            "a single-root document must not be rejected for its root count: {error}"
        );

        let error = load_errors("a.bsn", "(Foo),\n(Bar)");
        assert!(
            matches!(
                error,
                DynamicBsnLoaderError::Build {
                    source: DynamicSceneBuildError::MultipleRoots { count: 2, .. },
                    ..
                }
            ),
            "a two-root document must be rejected: {error}"
        );
        assert!(
            error.to_string().contains("exactly one root entity"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn self_include_check_matches_only_the_own_path() {
        let source = ":\"a.bsn\"\nFoo";
        let document = bevy_bsn::parse(source).unwrap();

        let own = AssetPath::from("a.bsn").into_owned();
        let error = check_no_self_include(&document, source, "a.bsn", &own)
            .expect_err("a self-including document must be rejected");
        assert!(
            error.to_string().contains("inherits from itself"),
            "unexpected message: {error}"
        );

        let other = AssetPath::from("b.bsn").into_owned();
        assert!(check_no_self_include(&document, source, "b.bsn", &other).is_ok());
    }

    /// The hand-written `Debug` impl names the type (it exists because `AppTypeRegistry`
    /// is not `Debug`).
    #[test]
    fn loader_debug_names_the_type() {
        let loader = DynamicBsnLoader::new(AppTypeRegistry::default());
        assert!(format!("{loader:?}").contains("DynamicBsnLoader"));
    }
}
