//! Live Slang source registry: hot reload for the RT stages.
//!
//! Every built-in stage and module source is embedded at compile time
//! (`include_str!`), but each also carries its checkout path. When the file
//! exists on disk — a dev checkout, or the cargo registry's source copy —
//! the on-disk content wins and the file is watched: an edit bumps
//! [`generation`](SlangSources::generation), the dispatch tears down the
//! RT library cache, and the next pipeline build recompiles every stage
//! from the new source. Reading from disk at startup also means a shader
//! edited after the binary was built takes effect without a recompile.
//!
//! Reloaded content is leaked (`Box::leak`) so it can flow through the same
//! `&'static str` plumbing as the embedded sources — a few KB per edit,
//! bounded by the session's edit count.

use bevy_ecs::resource::Resource;
use std::path::PathBuf;
use std::time::SystemTime;

struct SourceEntry {
    /// File name, the lookup key (`"raygen.slang"`).
    file: &'static str,
    /// The checkout path baked at compile time; watched when it exists.
    path: PathBuf,
    current: &'static str,
    /// `Some` while the on-disk file is being watched (mtime of the last
    /// successful read).
    mtime: Option<SystemTime>,
}

/// The watched RT stage + module sources. Render-world resource; polled once
/// per frame ([`poll_slang_sources`]).
#[derive(Resource)]
pub struct SlangSources {
    entries: Vec<SourceEntry>,
    generation: u64,
}

macro_rules! watched {
    ($file:literal) => {
        (
            $file,
            concat!(env!("CARGO_MANIFEST_DIR"), "/src/render/rt_pipeline/", $file),
            include_str!(concat!("../render/rt_pipeline/", $file)),
        )
    };
}

impl SlangSources {
    pub fn new() -> Self {
        let mut sources = Self {
            entries: [
                watched!("raygen.slang"),
                watched!("miss.slang"),
                watched!("miss_shadow.slang"),
                watched!("ahit_alpha.slang"),
                watched!("chit_opaque.slang"),
                watched!("chit_glass.slang"),
                watched!("chit_hair.slang"),
                watched!("chit_portal.slang"),
                watched!("rt_payload.slang"),
                watched!("scene_resolve.slang"),
                watched!("brdf.slang"),
                watched!("sampling.slang"),
                watched!("hair.slang"),
            ]
            .into_iter()
            .map(|(file, path, embedded)| SourceEntry {
                file,
                path: PathBuf::from(path),
                current: embedded,
                mtime: None,
            })
            .collect(),
            generation: 0,
        };
        for entry in &mut sources.entries {
            entry.refresh();
        }
        sources
    }

    /// Bumped whenever any watched file's content changes.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Current content of a watched file, by file name — `None` for files
    /// outside the built-in set (a downstream hit group's own source).
    pub fn get(&self, file: &str) -> Option<&'static str> {
        self.entries
            .iter()
            .find(|entry| entry.file == file)
            .map(|entry| entry.current)
    }

    /// [`get`](Self::get) for the built-in set, where absence is a bug.
    #[track_caller]
    pub fn source(&self, file: &str) -> &'static str {
        self.get(file)
            .unwrap_or_else(|| panic!("{file} is not a watched slang source"))
    }

    /// Re-stat every watched file and re-read the changed ones; bumps the
    /// generation if any content changed.
    pub fn poll(&mut self) {
        let mut changed = false;
        for entry in &mut self.entries {
            changed |= entry.refresh();
        }
        if changed {
            self.generation += 1;
        }
    }
}

impl SourceEntry {
    /// Sync `current` with the on-disk file if it exists and its mtime moved;
    /// returns whether the content changed. A transient read failure (editor
    /// mid-save) leaves the stored mtime alone, so the next poll retries.
    fn refresh(&mut self) -> bool {
        let Ok(mtime) = std::fs::metadata(&self.path).and_then(|m| m.modified()) else {
            return false;
        };
        if self.mtime == Some(mtime) {
            return false;
        }
        let Ok(content) = std::fs::read_to_string(&self.path) else {
            return false;
        };
        self.mtime = Some(mtime);
        if content == self.current {
            return false;
        }
        self.current = Box::leak(content.into_boxed_str());
        true
    }
}

impl Default for SlangSources {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-frame mtime poll (a dozen `stat` calls; microseconds).
pub fn poll_slang_sources(mut sources: bevy_ecs::system::ResMut<SlangSources>) {
    sources.poll();
}
