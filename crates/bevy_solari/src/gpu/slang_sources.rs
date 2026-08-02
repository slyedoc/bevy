//! Live Slang source registry: hot reload for the RT stages, driven by the
//! bevy asset server.
//!
//! Every built-in stage and module source is registered as an embedded asset
//! (`embedded_asset!`, in `render/mod.rs`) and loaded through
//! [`SlangSourceLoader`]; the handles are pinned by [`SlangSourceHandles`].
//! Asset events extract into the render world
//! ([`extract_slang_sources`], the same pattern bevy's `PipelineCache` uses
//! for WGSL), where [`SlangSources`] holds the current content: a changed
//! source bumps [`generation`](SlangSources::generation), the dispatch tears
//! down the RT library cache, and the next pipeline build recompiles from
//! the new content.
//!
//! With bevy_asset's `embedded_watcher` feature enabled (the app's opt-in,
//! exactly as for bevy's own shaders), saving a `.slang` file in the
//! checkout fires a `Modified` event and the edit shows up live. Without
//! it, the embedded bytes are simply static — a shipped build carries no
//! watcher.
//!
//! Reloaded content is leaked (`Box::leak`) so it can flow through the same
//! `&'static str` plumbing as the embedded sources — a few KB per edit,
//! bounded by the session's edit count.

use bevy_asset::{io::Reader, Asset, AssetEvent, AssetLoader, Assets, Handle, LoadContext};
use bevy_ecs::{
    message::MessageReader,
    resource::Resource,
    system::{Res, ResMut},
};
use bevy_reflect::TypePath;
use bevy_render::Extract;

/// One raw `.slang` source file.
#[derive(Asset, TypePath)]
pub struct SlangSource(pub String);

#[derive(Default, TypePath)]
pub struct SlangSourceLoader;

impl AssetLoader for SlangSourceLoader {
    type Asset = SlangSource;
    type Settings = ();
    type Error = std::io::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<SlangSource, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        String::from_utf8(bytes)
            .map(SlangSource)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    fn extensions(&self) -> &[&str] {
        &["slang"]
    }
}

/// Pins the embedded `.slang` assets loaded (dropping a handle would unload
/// its asset) and maps asset ids back to file names for the extract sync.
/// Main-world resource.
#[derive(Resource)]
pub struct SlangSourceHandles(pub Vec<(&'static str, Handle<SlangSource>)>);

struct SourceEntry {
    /// File name, the lookup key (`"raygen.slang"`).
    file: &'static str,
    current: &'static str,
}

/// The current RT stage + module sources. Render-world resource, updated by
/// [`extract_slang_sources`]; starts as the embedded copies, so the first
/// pipeline build never waits on asset loads.
#[derive(Resource)]
pub struct SlangSources {
    entries: Vec<SourceEntry>,
    generation: u64,
}

macro_rules! embedded {
    ($file:literal) => {
        ($file, include_str!(concat!("../render/rt_pipeline/", $file)))
    };
}

/// The built-in watched set; `render/mod.rs` registers each as an embedded
/// asset from the same list shape.
const SOURCE_FILES: [(&str, &str); 13] = [
    embedded!("raygen.slang"),
    embedded!("miss.slang"),
    embedded!("miss_shadow.slang"),
    embedded!("ahit_alpha.slang"),
    embedded!("chit_opaque.slang"),
    embedded!("chit_glass.slang"),
    embedded!("chit_hair.slang"),
    embedded!("chit_portal.slang"),
    embedded!("rt_payload.slang"),
    embedded!("scene_resolve.slang"),
    embedded!("brdf.slang"),
    embedded!("sampling.slang"),
    embedded!("hair.slang"),
];

impl SlangSources {
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

    /// Replace a file's content; bumps the generation when it actually
    /// changed (initial asset loads deliver the embedded bytes — a no-op).
    fn apply(&mut self, file: &str, content: &str) {
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.file == file) else {
            return;
        };
        if entry.current == content {
            return;
        }
        entry.current = Box::leak(content.to_owned().into_boxed_str());
        self.generation += 1;
    }
}

impl Default for SlangSources {
    fn default() -> Self {
        Self {
            entries: SOURCE_FILES
                .into_iter()
                .map(|(file, embedded)| SourceEntry {
                    file,
                    current: embedded,
                })
                .collect(),
            generation: 0,
        }
    }
}

/// `ExtractSchedule`: apply main-world `.slang` asset events into the
/// render-world registry.
pub fn extract_slang_sources(
    mut sources: ResMut<SlangSources>,
    assets: Extract<Res<Assets<SlangSource>>>,
    handles: Extract<Res<SlangSourceHandles>>,
    mut events: Extract<MessageReader<AssetEvent<SlangSource>>>,
) {
    for event in events.read() {
        let (AssetEvent::Added { id } | AssetEvent::Modified { id }) = event else {
            continue;
        };
        let Some((file, _)) = handles.0.iter().find(|(_, handle)| handle.id() == *id) else {
            continue;
        };
        if let Some(asset) = assets.get(*id) {
            sources.apply(file, &asset.0);
        }
    }
}
