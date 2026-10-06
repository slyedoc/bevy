//! Writing live entities back to `.bsn`: the inverse of the loader.
//!
//! [`write_scene`] walks an entity tree by reflection and builds a [`BsnDocument`] that loads back
//! into the same entities. It writes what an author would:
//!
//! * a component's fields only where they differ from the type's default (a component equal to
//!   its default is written as its bare path, which still adds it);
//! * an entity that inherits a `:"base.bsn"` (it carries [`SceneBase`]) as the include plus only
//!   what differs from a fresh copy of the base. The base's own descendants are not written: they
//!   belong to the base file, which is where they are edited;
//! * a handle as its asset path, or, for a handle with no path, the asset's value inline;
//! * an [`Entity`] pointing inside the tree as `#name`;
//! * `Name` as the entity's `#name`, and `Children` as its relation block.
//!
//! Components that are derived at runtime (global transforms, computed visibility) are skipped,
//! as is a component another component on the entity requires while it still holds its default.

use alloc::{
    borrow::Cow,
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::any::TypeId;

use bevy_asset::{AssetPath, AssetServer, Assets, ReflectAsset, ReflectHandle, UntypedAssetId};
use bevy_bsn::{BsnDocument, BsnNodeKind, BsnPatchPrefix, BsnPath, BsnValue, BsnValueId, PatchBody};
use bevy_ecs::{
    entity::Entity,
    entity_disabling::Disabled,
    hierarchy::{ChildOf, Children},
    name::Name,
    reflect::{AppTypeRegistry, ReflectComponent},
    world::World,
};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_reflect::{
    enums::{Enum, VariantType}, std_traits::ReflectDefault, structs::Struct,
    tuple_struct::TupleStruct, PartialReflect, ReflectRef,
    TypeRegistry,
};
use bevy_scene::{SceneBase, ScenePatch};
use thiserror::Error;

/// Components [`write_scene`] never writes, by type path: values the engine derives at runtime.
pub const DERIVED_COMPONENTS: &[&str] = &[
    "bevy_transform::components::global_transform::GlobalTransform",
    "bevy_transform::components::transform::TransformTreeChanged",
    "bevy_camera::visibility::InheritedVisibility",
    "bevy_camera::visibility::ViewVisibility",
    "bevy_scene::scene_patch::SceneInstanceState",
];

/// Decides, per component value, that it is derived rather than authored (a generated mesh
/// handle, say) and is not written.
pub type SkipValue = alloc::sync::Arc<dyn Fn(TypeId, &dyn PartialReflect) -> bool + Send + Sync>;

/// What [`write_scene`] leaves out, beyond [`DERIVED_COMPONENTS`].
#[derive(Default, Clone)]
pub struct WriteSettings {
    /// Component types never written (an editor's own bookkeeping, say).
    pub skip_components: HashSet<TypeId>,
    /// Entities never written, nor anything below them (an editor's gizmos, say).
    pub skip_entities: HashSet<Entity>,
    /// Component values not written, by what they hold.
    pub skip_value: Option<SkipValue>,
    /// The path to write for a handle that has none of its own (an asset an editor read from a
    /// file into its store, say).
    pub asset_paths: HashMap<UntypedAssetId, String>,
}

/// Why an entity tree could not be written.
#[derive(Debug, Error)]
pub enum BsnWriteError {
    /// A value of a type `.bsn` has no literal for.
    #[error("`{component}` holds a `{value_type}`, which a .bsn cannot write")]
    Unsupported {
        /// The component holding it.
        component: String,
        /// The value's type.
        value_type: String,
    },
    /// A handle with neither a path nor a loaded asset to write inline.
    #[error("`{component}` holds a handle with no path and no loaded asset")]
    EmptyHandle {
        /// The component holding it.
        component: String,
    },
    /// An [`Entity`] value pointing at an entity that is not written, or not uniquely named.
    #[error("`{component}` points at an entity outside the scene, or one whose name is not unique")]
    ForeignEntity {
        /// The component holding it.
        component: String,
    },
    /// A `:"base.bsn"` whose scene is not loaded, so the difference cannot be taken.
    #[error("base `{0}` is not loaded")]
    BaseNotLoaded(String),
    /// A `:"base.bsn"` that failed to spawn for comparison.
    #[error("base `{0}`: {1}")]
    BaseSpawn(String, String),
}

/// Writes `root` and the entities below it as a `.bsn` document (see the module docs).
pub fn write_scene(
    world: &mut World,
    root: Entity,
    settings: &WriteSettings,
) -> Result<BsnDocument, BsnWriteError> {
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let mut bases = Bases::default();
    let result = Writer::new(&registry, settings).write(world, root, &mut bases);
    bases.despawn(world);
    result
}

/// Writes `roots` as a scene file: an unnamed root holding them as its children.
pub fn write_scene_roots(
    world: &mut World,
    roots: &[Entity],
    settings: &WriteSettings,
) -> Result<BsnDocument, BsnWriteError> {
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let mut bases = Bases::default();
    let writer = Writer::new(&registry, settings);
    let result = (|| {
        let mut children = Vec::new();
        for &root in roots {
            if !settings.skip_entities.contains(&root) {
                children.push(writer.node(world, root, &mut bases)?);
            }
        }
        writer.finish(Node {
            entity: Entity::PLACEHOLDER,
            name: None,
            base: None,
            patches: Vec::new(),
            children,
        })
    })();
    bases.despawn(world);
    result
}

/// [`write_scene_roots`], printed.
pub fn write_scene_roots_text(
    world: &mut World,
    roots: &[Entity],
    settings: &WriteSettings,
) -> Result<String, BsnWriteError> {
    write_scene_roots(world, roots, settings).map(|document| document.to_bsn_string())
}

/// [`write_scene`], printed.
pub fn write_scene_text(
    world: &mut World,
    root: Entity,
    settings: &WriteSettings,
) -> Result<String, BsnWriteError> {
    write_scene(world, root, settings).map(|document| document.to_bsn_string())
}

/// A value on its way to the document; entity references are resolved to names at the end.
pub(crate) enum W {
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Path(String),
    Tuple(Vec<W>),
    List(Vec<W>),
    Struct(String, Vec<(String, W)>),
    Named(String, Vec<W>),
    Ref(Entity, String),
}

pub(crate) enum Body {
    Unit,
    Struct(Vec<(String, W)>),
    Tuple(Vec<W>),
}

struct Node {
    entity: Entity,
    name: Option<String>,
    base: Option<String>,
    patches: Vec<(String, Body)>,
    children: Vec<Node>,
}

/// Fresh copies of bases, spawned disabled for comparison and despawned when done.
#[derive(Default)]
struct Bases {
    spawned: HashMap<String, Entity>,
}

impl Bases {
    fn get(&mut self, world: &mut World, path: &str) -> Result<Entity, BsnWriteError> {
        if let Some(&root) = self.spawned.get(path) {
            return Ok(root);
        }
        let asset_path = AssetPath::parse(path).into_owned();
        let resolved = world
            .get_resource::<AssetServer>()
            .and_then(|server| server.get_handle::<ScenePatch>(&asset_path))
            .and_then(|handle| world.resource::<Assets<ScenePatch>>().get(&handle))
            .and_then(|patch| patch.resolved.clone())
            .ok_or_else(|| BsnWriteError::BaseNotLoaded(path.to_string()))?;
        let root = resolved
            .spawn(world)
            .map_err(|err| BsnWriteError::BaseSpawn(path.to_string(), format!("{err:?}")))?
            .id();
        let mut stack = vec![root];
        while let Some(entity) = stack.pop() {
            if let Some(children) = world.get::<Children>(entity) {
                stack.extend(children.iter());
            }
            world.entity_mut(entity).insert(Disabled);
        }
        self.spawned.insert(path.to_string(), root);
        Ok(root)
    }

    fn despawn(self, world: &mut World) {
        for root in self.spawned.into_values() {
            if let Ok(entity) = world.get_entity_mut(root) {
                entity.despawn();
            }
        }
    }
}

pub(crate) struct Writer<'a> {
    registry: &'a TypeRegistry,
    settings: &'a WriteSettings,
    derived: HashSet<TypeId>,
}

impl<'a> Writer<'a> {
    pub(crate) fn new(registry: &'a TypeRegistry, settings: &'a WriteSettings) -> Self {
        Self {
            registry,
            settings,
            derived: DERIVED_COMPONENTS
                .iter()
                .filter_map(|path| registry.get_with_type_path(path).map(|r| r.type_id()))
                .collect(),
        }
    }

    /// One patch node in `document`, from what [`Writer::patch`] made.
    pub(crate) fn emit_patch(
        &self,
        document: &mut BsnDocument,
        symbol: &str,
        body: Body,
    ) -> bevy_bsn::BsnNodeId {
        emit_patch(document, symbol, &body, &HashMap::default())
    }

    fn write(
        &self,
        world: &mut World,
        root: Entity,
        bases: &mut Bases,
    ) -> Result<BsnDocument, BsnWriteError> {
        let node = self.node(world, root, bases)?;
        self.finish(node)
    }

    /// The document for a written tree: names for every entity a reference points at.
    fn finish(&self, node: Node) -> Result<BsnDocument, BsnWriteError> {

        // `#name` per written entity; references need their target named, and uniquely.
        let mut names: HashMap<Entity, String> = HashMap::default();
        let mut taken: HashMap<String, usize> = HashMap::default();
        collect_names(&node, &mut names, &mut taken);
        let mut refs = Vec::new();
        collect_refs(&node, &mut refs);
        let mut generated = 0;
        for (target, component) in refs {
            match names.get(&target) {
                Some(name) if taken.get(name) == Some(&1) => {}
                Some(_) => return Err(BsnWriteError::ForeignEntity { component }),
                None if contains(&node, target) => {
                    let name = loop {
                        generated += 1;
                        let name = format!("entity_{generated}");
                        if !taken.contains_key(&name) {
                            break name;
                        }
                    };
                    taken.insert(name.clone(), 1);
                    names.insert(target, name);
                }
                None => return Err(BsnWriteError::ForeignEntity { component }),
            }
        }

        let mut document = BsnDocument::new();
        let root = emit_node(&mut document, &node, &names);
        document.push_root(root);
        Ok(document)
    }

    fn node(
        &self,
        world: &mut World,
        entity: Entity,
        bases: &mut Bases,
    ) -> Result<Node, BsnWriteError> {
        let base = world.get::<SceneBase>(entity).map(|b| b.0.clone());
        let base_root = match &base {
            Some(path) => Some(bases.get(world, path)?),
            None => None,
        };
        let patches = self.patches(world, entity, base_root)?;
        let name = world.get::<Name>(entity).map(|n| n.as_str().to_string());

        // A base's own children come first; only those after them are this file's.
        let skip = base_root
            .and_then(|b| world.get::<Children>(b))
            .map_or(0, |c| c.len());
        let kids: Vec<Entity> = world
            .get::<Children>(entity)
            .map(|c| c.iter().skip(skip).copied().collect())
            .unwrap_or_default();
        let mut children = Vec::new();
        for kid in kids {
            if self.settings.skip_entities.contains(&kid) {
                continue;
            }
            children.push(self.node(world, kid, bases)?);
        }
        Ok(Node {
            entity,
            name,
            base,
            patches,
            children,
        })
    }

    /// The entity's component patches, sorted by type path for stable output.
    fn patches(
        &self,
        world: &World,
        entity: Entity,
        base: Option<Entity>,
    ) -> Result<Vec<(String, Body)>, BsnWriteError> {
        let entity_ref = world.entity(entity);
        let components = world.components();
        let present: Vec<_> = entity_ref.archetype().components().to_vec();
        let required: HashSet<_> = present
            .iter()
            .filter_map(|id| components.get_info(*id))
            .flat_map(|info| info.required_components().iter_ids().collect::<Vec<_>>())
            .collect();
        let mut out = Vec::new();
        for id in present {
            let Some(type_id) = components.get_info(id).and_then(|i| i.type_id()) else {
                continue;
            };
            if self.skipped(type_id) {
                continue;
            }
            let Some(registration) = self.registry.get(type_id) else {
                continue;
            };
            let Some(reflect_component) = registration.data::<ReflectComponent>() else {
                continue;
            };
            let Some(value) = reflect_component.reflect(entity_ref) else {
                continue;
            };
            if self
                .settings
                .skip_value
                .as_ref()
                .is_some_and(|skip| skip(type_id, value.as_partial_reflect()))
            {
                continue;
            }
            let type_path = registration.type_info().type_path();
            let default = registration.data::<ReflectDefault>().map(|d| d.default());
            if required.contains(&id)
                && default
                    .as_ref()
                    .is_some_and(|d| equal(d.as_partial_reflect(), value.as_partial_reflect()))
            {
                continue;
            }
            let baseline = match base {
                Some(base) => match world
                    .get_entity(base)
                    .ok()
                    .and_then(|b| reflect_component.reflect(b))
                {
                    // Inherited unchanged: the base supplies it.
                    Some(inherited) if equal(inherited.as_partial_reflect(), value.as_partial_reflect()) => {
                        continue
                    }
                    Some(inherited) => Some(inherited.as_partial_reflect()),
                    None => default.as_deref().map(|d| d.as_partial_reflect()),
                },
                None => default.as_deref().map(|d| d.as_partial_reflect()),
            };
            out.push(self.patch(world, type_path, value.as_partial_reflect(), baseline)?);
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    fn skipped(&self, type_id: TypeId) -> bool {
        type_id == TypeId::of::<Name>()
            || type_id == TypeId::of::<Children>()
            || type_id == TypeId::of::<ChildOf>()
            || type_id == TypeId::of::<SceneBase>()
            || type_id == TypeId::of::<Disabled>()
            || self.derived.contains(&type_id)
            || self.settings.skip_components.contains(&type_id)
    }

    /// A component as a patch: the fields that differ from `baseline` (its default, or what the
    /// base gives it), since a patch applies on top of exactly that.
    pub(crate) fn patch(
        &self,
        world: &World,
        type_path: &str,
        value: &dyn PartialReflect,
        baseline: Option<&dyn PartialReflect>,
    ) -> Result<(String, Body), BsnWriteError> {
        let cx = Cx {
            world,
            component: type_path,
        };
        Ok(match value.reflect_ref() {
            ReflectRef::Struct(value) => {
                let fields = self.struct_fields(&cx, value, baseline.and_then(as_struct))?;
                let body = if fields.is_empty() {
                    Body::Unit
                } else {
                    Body::Struct(fields)
                };
                (type_path.to_string(), body)
            }
            ReflectRef::TupleStruct(value) => {
                let items = self.leading_fields(&cx, value, baseline.and_then(as_tuple_struct))?;
                let body = if items.is_empty() {
                    Body::Unit
                } else {
                    Body::Tuple(items)
                };
                (type_path.to_string(), body)
            }
            ReflectRef::Enum(value) => {
                let symbol = format!("{type_path}::{}", value.variant_name());
                // The same variant as the baseline patches its fields; a switch is written whole.
                let same = baseline
                    .and_then(as_enum)
                    .filter(|b| b.variant_name() == value.variant_name());
                let body = match value.variant_type() {
                    VariantType::Unit => Body::Unit,
                    VariantType::Tuple => {
                        let last = (0..value.field_len()).rev().find(|&i| {
                            !same
                                .and_then(|b| b.field_at(i))
                                .zip(value.field_at(i))
                                .is_some_and(|(b, v)| equal(b, v))
                        });
                        match last {
                            None => Body::Unit,
                            Some(last) => Body::Tuple(
                                (0..=last)
                                    .filter_map(|i| value.field_at(i))
                                    .map(|f| self.value(&cx, f, None))
                                    .collect::<Result<_, _>>()?,
                            ),
                        }
                    }
                    VariantType::Struct => {
                        let mut fields = Vec::new();
                        for field in value.iter_fields() {
                            let name = field.name().unwrap_or_default();
                            let before = same.and_then(|b| b.field(name));
                            if before.is_some_and(|b| equal(b, field.value())) {
                                continue;
                            }
                            fields.push((name.to_string(), self.value(&cx, field.value(), before)?));
                        }
                        if fields.is_empty() && same.is_some() {
                            Body::Unit
                        } else {
                            Body::Struct(fields)
                        }
                    }
                };
                (symbol, body)
            }
            _ => {
                return Err(BsnWriteError::Unsupported {
                    component: type_path.to_string(),
                    value_type: type_path.to_string(),
                })
            }
        })
    }

    /// The fields of `value` that differ from `baseline`'s, each written against its baseline
    /// field (a nested struct patches onto the field it replaces). No baseline: every field.
    fn struct_fields(
        &self,
        cx: &Cx,
        value: &dyn Struct,
        baseline: Option<&dyn Struct>,
    ) -> Result<Vec<(String, W)>, BsnWriteError> {
        let mut fields = Vec::new();
        for (name, field) in value.iter_fields() {
            let before = baseline.and_then(|b| b.field(name));
            if before.is_some_and(|b| equal(b, field)) {
                continue;
            }
            fields.push((name.to_string(), self.value(cx, field, before)?));
        }
        Ok(fields)
    }

    /// The leading fields of `value` up to the last that differs from `baseline`: a tuple patch
    /// sets fields `0..n`.
    fn leading_fields(
        &self,
        cx: &Cx,
        value: &dyn TupleStruct,
        baseline: Option<&dyn TupleStruct>,
    ) -> Result<Vec<W>, BsnWriteError> {
        let last = (0..value.field_len()).rev().find(|&i| {
            !baseline
                .and_then(|b| b.field(i))
                .zip(value.field(i))
                .is_some_and(|(b, v)| equal(b, v))
        });
        let Some(last) = last else {
            return Ok(Vec::new());
        };
        (0..=last)
            .filter_map(|i| value.field(i).map(|f| (f, baseline.and_then(|b| b.field(i)))))
            .map(|(f, before)| self.value(cx, f, before))
            .collect()
    }

    /// `value` as a `.bsn` value. A struct or tuple struct writes only what differs from
    /// `baseline`, the value it will be patched onto; enums are written whole, since a variant
    /// value is built from that variant's defaults.
    fn value(
        &self,
        cx: &Cx,
        value: &dyn PartialReflect,
        baseline: Option<&dyn PartialReflect>,
    ) -> Result<W, BsnWriteError> {
        let unsupported = || BsnWriteError::Unsupported {
            component: cx.component.to_string(),
            value_type: value.reflect_type_path().to_string(),
        };
        if let Some(entity) = value.try_downcast_ref::<Entity>() {
            return Ok(W::Ref(*entity, cx.component.to_string()));
        }
        if let Some(info) = value.get_represented_type_info()
            && let Some(reflect_handle) = self.registry.get_type_data::<ReflectHandle>(info.type_id())
        {
            let handle = value
                .try_as_reflect()
                .and_then(|v| reflect_handle.downcast_handle_untyped(v.as_any()))
                .ok_or_else(unsupported)?;
            if let Some(path) = handle.path() {
                return Ok(W::Str(path.to_string()));
            }
            if let Some(path) = self.settings.asset_paths.get(&handle.id()) {
                return Ok(W::Str(path.clone()));
            }
            // No path: the asset was written inline, so write it inline again, against the
            // asset type's default (an inline asset is built from it).
            let asset_type = reflect_handle.asset_type_id();
            let asset = self
                .registry
                .get_type_data::<ReflectAsset>(asset_type)
                .and_then(|asset| asset.get(cx.world, handle.id()))
                .ok_or_else(|| BsnWriteError::EmptyHandle {
                    component: cx.component.to_string(),
                })?;
            let default = self
                .registry
                .get_type_data::<ReflectDefault>(asset_type)
                .map(|d| d.default());
            return self.value(
                cx,
                asset.as_partial_reflect(),
                default.as_deref().map(|d| d.as_partial_reflect()),
            );
        }
        Ok(match value.reflect_ref() {
            ReflectRef::Struct(s) => W::Struct(
                s.reflect_type_path().to_string(),
                self.struct_fields(cx, s, baseline.and_then(as_struct))?,
            ),
            ReflectRef::TupleStruct(t) => {
                let mut items = self.leading_fields(cx, t, baseline.and_then(as_tuple_struct))?;
                if items.is_empty() {
                    items = self.leading_fields(cx, t, None)?;
                }
                W::Named(t.reflect_type_path().to_string(), items)
            }
            ReflectRef::Tuple(t) => W::Tuple(
                t.iter_fields()
                    .map(|f| self.value(cx, f, None))
                    .collect::<Result<_, _>>()?,
            ),
            ReflectRef::List(l) => W::List(
                l.iter().map(|f| self.value(cx, f, None)).collect::<Result<_, _>>()?,
            ),
            ReflectRef::Array(a) => W::List(
                a.iter().map(|f| self.value(cx, f, None)).collect::<Result<_, _>>()?,
            ),
            ReflectRef::Set(s) => W::List(
                s.iter().map(|f| self.value(cx, f, None)).collect::<Result<_, _>>()?,
            ),
            ReflectRef::Map(m) => W::List(
                m.iter()
                    .map(|(k, v)| {
                        Ok(W::Tuple(vec![self.value(cx, k, None)?, self.value(cx, v, None)?]))
                    })
                    .collect::<Result<_, BsnWriteError>>()?,
            ),
            // `Some(x)` as `x`: an `Option` field takes its payload directly, which also spares
            // building `Some` from a default payload a handle does not have.
            ReflectRef::Enum(e) if is_some(e) => match e.field_at(0) {
                Some(payload) => self.value(cx, payload, None)?,
                None => return Err(unsupported()),
            },
            // Nested enums are written by variant: the destination type names the enum.
            ReflectRef::Enum(e) => match e.variant_type() {
                VariantType::Unit => W::Path(e.variant_name().to_string()),
                VariantType::Tuple => W::Named(
                    e.variant_name().to_string(),
                    e.iter_fields()
                        .map(|f| self.value(cx, f.value(), None))
                        .collect::<Result<_, _>>()?,
                ),
                VariantType::Struct => W::Struct(
                    e.variant_name().to_string(),
                    e.iter_fields()
                        .map(|f| {
                            Ok((
                                f.name().unwrap_or_default().to_string(),
                                self.value(cx, f.value(), None)?,
                            ))
                        })
                        .collect::<Result<_, BsnWriteError>>()?,
                ),
            },
            ReflectRef::Opaque(o) => opaque(o).ok_or_else(unsupported)?,
            #[expect(
                clippy::allow_attributes,
                reason = "`unreachable_patterns` may not always lint"
            )]
            #[allow(
                unreachable_patterns,
                reason = "ReflectRef::Function exists only with bevy_reflect/functions"
            )]
            _ => return Err(unsupported()),
        })
    }
}

/// `Some(_)` of an `Option`-shaped enum: a unit `None` beside a one-field tuple `Some`.
fn is_some(value: &dyn Enum) -> bool {
    use bevy_reflect::TypeInfo;
    value.variant_name() == "Some"
        && value.variant_type() == VariantType::Tuple
        && value.field_len() == 1
        && matches!(
            value.get_represented_type_info(),
            Some(TypeInfo::Enum(info)) if info.variant_len() == 2 && info.contains_variant("None")
        )
}

fn as_struct(value: &dyn PartialReflect) -> Option<&dyn Struct> {
    match value.reflect_ref() {
        ReflectRef::Struct(s) => Some(s),
        _ => None,
    }
}

fn as_tuple_struct(value: &dyn PartialReflect) -> Option<&dyn TupleStruct> {
    match value.reflect_ref() {
        ReflectRef::TupleStruct(t) => Some(t),
        _ => None,
    }
}

fn as_enum(value: &dyn PartialReflect) -> Option<&dyn Enum> {
    match value.reflect_ref() {
        ReflectRef::Enum(e) => Some(e),
        _ => None,
    }
}

struct Cx<'a> {
    world: &'a World,
    component: &'a str,
}

/// Primitive and text values.
fn opaque(value: &dyn PartialReflect) -> Option<W> {
    macro_rules! ints {
        ($($t:ty),*) => {$(
            if let Some(v) = value.try_downcast_ref::<$t>() {
                return Some(W::Int(*v as i128));
            }
        )*};
    }
    ints!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, usize);
    if let Some(v) = value.try_downcast_ref::<u128>() {
        return i128::try_from(*v).ok().map(W::Int);
    }
    if let Some(v) = value.try_downcast_ref::<f32>() {
        // The f32's own shortest decimal, not the digits its f64 widening shows.
        return Some(W::Float(v.to_string().parse().unwrap_or(*v as f64)));
    }
    if let Some(v) = value.try_downcast_ref::<f64>() {
        return Some(W::Float(*v));
    }
    if let Some(v) = value.try_downcast_ref::<bool>() {
        return Some(W::Bool(*v));
    }
    if let Some(v) = value.try_downcast_ref::<String>() {
        return Some(W::Str(v.clone()));
    }
    if let Some(v) = value.try_downcast_ref::<&'static str>() {
        return Some(W::Str(v.to_string()));
    }
    if let Some(v) = value.try_downcast_ref::<Cow<'static, str>>() {
        return Some(W::Str(v.to_string()));
    }
    if let Some(v) = value.try_downcast_ref::<char>() {
        return Some(W::Str(v.to_string()));
    }
    if let Some(v) = value.try_downcast_ref::<AssetPath<'static>>() {
        return Some(W::Str(v.to_string()));
    }
    if let Some(v) = value.try_downcast_ref::<Name>() {
        return Some(W::Str(v.as_str().to_string()));
    }
    None
}

fn equal(a: &dyn PartialReflect, b: &dyn PartialReflect) -> bool {
    a.reflect_partial_eq(b).unwrap_or(false)
}

fn collect_names(node: &Node, names: &mut HashMap<Entity, String>, taken: &mut HashMap<String, usize>) {
    if let Some(name) = &node.name {
        names.insert(node.entity, name.clone());
        *taken.entry(name.clone()).or_default() += 1;
    }
    for child in &node.children {
        collect_names(child, names, taken);
    }
}

fn collect_refs(node: &Node, refs: &mut Vec<(Entity, String)>) {
    fn walk(w: &W, refs: &mut Vec<(Entity, String)>) {
        match w {
            W::Ref(e, component) => refs.push((*e, component.clone())),
            W::Tuple(items) | W::List(items) | W::Named(_, items) => {
                items.iter().for_each(|i| walk(i, refs));
            }
            W::Struct(_, fields) => fields.iter().for_each(|(_, f)| walk(f, refs)),
            _ => {}
        }
    }
    for (_, body) in &node.patches {
        match body {
            Body::Unit => {}
            Body::Struct(fields) => fields.iter().for_each(|(_, f)| walk(f, refs)),
            Body::Tuple(items) => items.iter().for_each(|i| walk(i, refs)),
        }
    }
    for child in &node.children {
        collect_refs(child, refs);
    }
}

fn contains(node: &Node, entity: Entity) -> bool {
    node.entity == entity || node.children.iter().any(|c| contains(c, entity))
}

fn path(type_path: &str) -> BsnPath {
    BsnPath::from_type_path(type_path).unwrap_or_else(|| BsnPath::from_segments([type_path]))
}

fn emit_value(document: &mut BsnDocument, w: &W, names: &HashMap<Entity, String>) -> BsnValueId {
    let value = match w {
        W::Bool(v) => BsnValue::Bool(*v),
        W::Int(v) => BsnValue::Int(*v),
        W::Float(v) => BsnValue::Float(*v),
        W::Str(v) => BsnValue::String(v.clone()),
        W::Path(p) => BsnValue::Path(path(p)),
        W::Tuple(items) => {
            BsnValue::Tuple(items.iter().map(|i| emit_value(document, i, names)).collect())
        }
        W::List(items) => {
            BsnValue::List(items.iter().map(|i| emit_value(document, i, names)).collect())
        }
        W::Struct(p, fields) => BsnValue::Struct(
            path(p),
            fields
                .iter()
                .map(|(n, f)| (n.clone(), emit_value(document, f, names)))
                .collect(),
        ),
        W::Named(p, items) => BsnValue::NamedTuple(
            path(p),
            items.iter().map(|i| emit_value(document, i, names)).collect(),
        ),
        W::Ref(e, _) => BsnValue::EntityRef(names.get(e).cloned().unwrap_or_default()),
    };
    document.push_value(value)
}

fn emit_patch(
    document: &mut BsnDocument,
    type_path: &str,
    body: &Body,
    names: &HashMap<Entity, String>,
) -> bevy_bsn::BsnNodeId {
    let body = match body {
        Body::Unit => PatchBody::Unit,
        Body::Struct(fields) => PatchBody::Struct(
            fields
                .iter()
                .map(|(n, f)| (n.clone(), emit_value(document, f, names)))
                .collect(),
        ),
        Body::Tuple(items) => {
            PatchBody::Tuple(items.iter().map(|i| emit_value(document, i, names)).collect())
        }
    };
    document.push_patch(BsnPatchPrefix::FromTemplate, path(type_path), body)
}

fn emit_node(document: &mut BsnDocument, node: &Node, names: &HashMap<Entity, String>) -> bevy_bsn::BsnNodeId {
    let patches = node
        .patches
        .iter()
        .map(|(type_path, body)| emit_patch(document, type_path, body, names))
        .collect();
    let children: Vec<_> = node
        .children
        .iter()
        .map(|c| emit_node(document, c, names))
        .collect();
    let relations = if children.is_empty() {
        Vec::new()
    } else {
        vec![document.push_node(BsnNodeKind::Relation {
            target_symbol: path("bevy_ecs::hierarchy::Children"),
            entities: children,
        })]
    };
    document.push_node(BsnNodeKind::Entity {
        name: names.get(&node.entity).cloned().or_else(|| node.name.clone()),
        name_span: None,
        base: node.base.clone(),
        base_span: None,
        patches,
        relations,
    })
}
