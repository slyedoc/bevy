//! A `.bsn` file holding one asset value rather than a scene: the root's `#name`, and the value
//! as the root's one patch.
//!
//! ```bsn
//! #slate
//! bevy_aurora::material::AuroraMaterial { base_color: Srgba(Srgba { red: 0.3, .. }) }
//! ```
//!
//! [`BsnValueLoader`] loads such a file as the asset; [`write_asset_value`] writes one, with only
//! the fields that differ from the type's default.

use alloc::{boxed::Box, format, string::String, vec::Vec};
use core::marker::PhantomData;

use bevy_app::App;
use bevy_asset::{io::Reader, Asset, AssetApp, AssetLoader, LoadContext};
use bevy_bsn::{BsnDocument, BsnNodeKind};
use bevy_ecs::{
    reflect::AppTypeRegistry,
    world::{FromWorld, World},
};
use bevy_reflect::{
    std_traits::ReflectDefault, FromReflect, PartialReflect, Reflect, ReflectFromReflect,
    TypePath, TypeRegistration, TypeRegistry, TypeRegistryArc,
};
use thiserror::Error;

use crate::{
    build::{resolve_symbol, BuildCx, DynamicSceneBuildError, HandleProvider},
    value::build_value,
    write::{BsnWriteError, Writer},
};

/// Why a document did not hold the asset value asked for.
#[derive(Debug, Error)]
pub enum BsnAssetValueError {
    /// The document has no root entity, or more than one.
    #[error("an asset file has one root entity, not {0}")]
    Roots(usize),
    /// The root holds no patch of the asked-for type.
    #[error("the root holds no `{0}`")]
    Missing(String),
    /// The value did not build.
    #[error(transparent)]
    Build(#[from] DynamicSceneBuildError),
    /// The value built but would not become the type.
    #[error("`{0}` could not be built from the value written")]
    NotTheType(String),
}

/// The name a document's root carries, and the value of `registration`'s type it holds.
pub fn asset_value_from_document(
    document: &BsnDocument,
    source: &str,
    registration: &TypeRegistration,
    registry: &TypeRegistry,
    handles: Option<HandleProvider>,
) -> Result<(Option<String>, Box<dyn Reflect>), BsnAssetValueError> {
    let [root] = document.roots[..] else {
        return Err(BsnAssetValueError::Roots(document.roots.len()));
    };
    let type_path = registration.type_info().type_path();
    let Some(BsnNodeKind::Entity { name, patches, .. }) = document.node(root).map(|n| &n.kind)
    else {
        return Err(BsnAssetValueError::Roots(0));
    };
    let mut cx = BuildCx::new(registry, document, source);
    cx.handles = handles;
    let value = patches.iter().find_map(|&patch| {
        let Some(BsnNodeKind::Patch { symbol, value, .. }) = document.node(patch).map(|n| &n.kind)
        else {
            return None;
        };
        let resolved = resolve_symbol(registry, symbol, bevy_bsn::Span::NONE).ok()?;
        (resolved.registration.type_id() == registration.type_id()).then_some(*value)
    });
    let Some(value) = value else {
        return Err(BsnAssetValueError::Missing(type_path.into()));
    };
    let partial = build_value(&mut cx, value, registration)?;
    let concrete = registration
        .data::<ReflectFromReflect>()
        .and_then(|from| from.from_reflect(partial.as_ref()))
        .or_else(|| {
            let mut value = registration.data::<ReflectDefault>()?.default();
            value.try_apply(partial.as_ref()).ok()?;
            Some(value)
        })
        .ok_or_else(|| BsnAssetValueError::NotTheType(type_path.into()))?;
    Ok((name.clone(), concrete))
}

/// Writes `value` as an asset file named `name`: one patch, holding the fields that differ from
/// the type's default.
pub fn write_asset_value(
    world: &World,
    name: &str,
    value: &dyn PartialReflect,
) -> Result<BsnDocument, BsnWriteError> {
    write_asset_value_with(world, name, value, &crate::write::WriteSettings::default())
}

/// [`write_asset_value`] with the settings a scene write takes (paths for pathless handles, say).
pub fn write_asset_value_with(
    world: &World,
    name: &str,
    value: &dyn PartialReflect,
    settings: &crate::write::WriteSettings,
) -> Result<BsnDocument, BsnWriteError> {
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let writer = Writer::new(&registry, settings);
    let type_path = value.reflect_type_path();
    let default = value
        .get_represented_type_info()
        .and_then(|info| registry.get_type_data::<ReflectDefault>(info.type_id()))
        .map(|d| d.default());
    let (symbol, body) = writer.patch(
        world,
        type_path,
        value,
        default.as_deref().map(|d| d.as_partial_reflect()),
    )?;
    let mut document = BsnDocument::new();
    let patch = writer.emit_patch(&mut document, &symbol, body);
    let root = document.push_node(BsnNodeKind::Entity {
        name: Some(name.into()),
        name_span: None,
        base: None,
        base_span: None,
        patches: Vec::from([patch]),
        relations: Vec::new(),
    });
    document.push_root(root);
    Ok(document)
}

/// Loads a `.bsn` asset file as an `A` (see the module docs). Register it with
/// [`BsnAssetAppExt::register_bsn_asset`].
#[derive(TypePath)]
pub struct BsnValueLoader<A> {
    registry: TypeRegistryArc,
    #[type_path(ignore)]
    _asset: PhantomData<fn() -> A>,
}

impl<A> FromWorld for BsnValueLoader<A> {
    fn from_world(world: &mut World) -> Self {
        Self {
            registry: world.resource::<AppTypeRegistry>().0.clone(),
            _asset: PhantomData,
        }
    }
}

/// Why a `.bsn` asset file did not load.
#[derive(Debug, Error)]
pub enum BsnValueLoaderError {
    /// Reading the file failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The file is not UTF-8.
    #[error("not UTF-8: {0}")]
    Utf8(#[from] core::str::Utf8Error),
    /// The file did not parse.
    #[error("does not parse: {0}")]
    Parse(String),
    /// The file did not hold the value.
    #[error(transparent)]
    Value(#[from] BsnAssetValueError),
}

impl<A: Asset + FromReflect> AssetLoader for BsnValueLoader<A> {
    type Asset = A;
    type Settings = ();
    type Error = BsnValueLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<A, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let text = core::str::from_utf8(&bytes)?;
        let document = BsnDocument::parse(text).map_err(|e| BsnValueLoaderError::Parse(format!("{e:?}")))?;
        let source = load_context.path().to_string();
        let registry = self.registry.read();
        let registration = registry
            .get(core::any::TypeId::of::<A>())
            .ok_or_else(|| BsnAssetValueError::Missing(A::type_path().into()))?;
        let mut handles =
            |type_id, path| load_context.load_builder().load_erased(type_id, path);
        let (_, value) = asset_value_from_document(
            &document,
            &source,
            registration,
            &registry,
            Some(&mut handles),
        )?;
        value
            .take::<A>()
            .map_err(|_| BsnAssetValueError::NotTheType(A::type_path().into()).into())
    }

    fn extensions(&self) -> &[&str] {
        &["bsn"]
    }
}

/// Registers `.bsn` asset files for an asset type.
pub trait BsnAssetAppExt {
    /// Load `.bsn` files holding one `A` as `A` assets (a typed load picks this loader over
    /// the scene loader that shares the extension).
    fn register_bsn_asset<A: Asset + FromReflect>(&mut self) -> &mut Self;
}

impl BsnAssetAppExt for App {
    fn register_bsn_asset<A: Asset + FromReflect>(&mut self) -> &mut Self {
        let loader = BsnValueLoader::<A>::from_world(self.world_mut());
        self.register_asset_loader(loader)
    }
}
