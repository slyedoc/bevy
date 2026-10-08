//! Root addressing and the writeback observers.
//!
//! Every leaf widget spawned by the inspector carries an [`InspectorBinding`] that records how to
//! get from the widget back to the data it edits: an [`InspectorRoot`] (which component, on which
//! entity) plus a reflection [`ParsedPath`] from that root down to the edited field. When a widget
//! emits a [`ValueChange<T>`], a writeback observer resolves the binding and hands the new value to
//! [`apply_edit`], which writes it through reflection (driving change detection on the target) or
//! passes it to the app's [`InspectorEdits`] handler, an editor's undo history say.

use core::any::TypeId;

use bevy_asset::{ReflectAsset, UntypedAssetId};
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_reflect::{GetPath, ParsedPath, PartialReflect, Reflect, ReflectMut, ReflectRef};
use bevy_ui_widgets::ValueChange;

use crate::widget::SliderScalar;

/// An app's handler for inspector edits, called with the root, the path to the edited value, the
/// new value and whether the gesture is over (`is_final`: a released drag, a committed text, a
/// click). It returns whether it took the edit; one it leaves is written straight through.
///
/// A drag sends many non-final edits and then one final one, so an undo history can record one
/// entry per gesture.
#[derive(Resource, Clone, Copy)]
pub struct InspectorEdits(
    pub fn(&mut World, &InspectorRoot, &ParsedPath, &dyn PartialReflect, bool) -> bool,
);

/// Apply an edit of the value at `root` / `path`: the [`InspectorEdits`] handler's to make if it
/// takes it, else written through reflection.
pub fn apply_edit(
    world: &mut World,
    root: &InspectorRoot,
    path: &ParsedPath,
    value: &dyn PartialReflect,
    is_final: bool,
) {
    if let Some(InspectorEdits(handler)) = world.get_resource::<InspectorEdits>().copied()
        && handler(world, root, path, value, is_final)
    {
        return;
    }
    with_field_reflect_mut(world, root, path, |target| {
        if let Err(err) = assign(target, value) {
            bevy_log::warn!("inspector: {path}: {err}");
        }
    });
}

/// `target.try_apply(value)`, except that a list ends up exactly as long as `value`: reflection's
/// own list apply only overwrites and appends, so a removal would leave the old tail behind.
pub fn assign(
    target: &mut dyn PartialReflect,
    value: &dyn PartialReflect,
) -> Result<(), bevy_reflect::ApplyError> {
    if let (ReflectMut::List(list), ReflectRef::List(new)) =
        (target.reflect_mut(), value.reflect_ref())
    {
        while list.len() > new.len() {
            list.pop();
        }
    }
    target.try_apply(value)
}

/// The value at `root` / `path` as an owned copy, for an edit that rebuilds it (a list push, an
/// enum variant switch) before handing it to [`apply_edit`].
pub fn clone_field(
    world: &World,
    root: &InspectorRoot,
    path: &ParsedPath,
) -> Option<Box<dyn PartialReflect>> {
    read_field(world, root, path, |value| value.to_dynamic().ok()).flatten()
}

/// Queue an edit of the field `binding` addresses; the observers' common tail.
pub fn queue_edit(
    commands: &mut Commands,
    binding: &InspectorBinding,
    value: Box<dyn PartialReflect>,
    is_final: bool,
) {
    let Some(root) = binding.root.clone() else {
        return;
    };
    let path = binding.path.clone();
    commands.queue(move |world: &mut World| {
        apply_edit(world, &root, &path, value.as_ref(), is_final);
    });
}

/// Resolve `root` + `path` to an immutable reflected reference and hand it to `f`.
///
/// Used by the external-sync systems to read the current source value back out.
pub fn read_field<R>(
    world: &World,
    root: &InspectorRoot,
    path: &ParsedPath,
    f: impl FnOnce(&dyn PartialReflect) -> R,
) -> Option<R> {
    // Resolved entirely by the app; nothing here knows what it reached.
    if let InspectorRoot::Custom { read, .. } = root {
        // `f` is `FnOnce` but the resolver takes `FnMut`, so it moves out on first call.
        let mut f = Some(f);
        let mut out = None;
        let mut visit = |value: &dyn Reflect| {
            let Ok(target) = value.reflect_path(path) else {
                return;
            };
            let Some(f) = f.take() else { return };
            out = Some(f(target));
        };
        read(world, &mut visit);
        return out;
    }
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    // Assets do not live on an entity, so they resolve through `ReflectAsset` instead.
    if let InspectorRoot::Asset { asset_id, type_id } = root {
        let reflect_asset = registry.get(*type_id)?.data::<ReflectAsset>()?;
        let reflected = reflect_asset.get(world, *asset_id)?;
        let target = reflected.reflect_path(path).ok()?;
        return Some(f(target));
    }
    let entity = match root {
        InspectorRoot::Component { entity, .. } => *entity,
        InspectorRoot::Resource { type_id } => world
            .components()
            .get_id(*type_id)
            .and_then(|id| world.resource_entities().get(id))?,
        InspectorRoot::Asset { .. } | InspectorRoot::Custom { .. } => {
            unreachable!("handled above")
        }
    };
    let type_id = match root {
        InspectorRoot::Component { type_id, .. }
        | InspectorRoot::Resource { type_id }
        | InspectorRoot::Asset { type_id, .. } => *type_id,
        InspectorRoot::Custom { .. } => unreachable!("handled above"),
    };
    let reflect_component = registry.get(type_id)?.data::<ReflectComponent>()?;
    let reflected = reflect_component.reflect(world.get_entity(entity).ok()?)?;
    let target = reflected.reflect_path(path).ok()?;
    Some(f(target))
}

/// Best-effort read of a reflected numeric value as `f32`.
pub(crate) fn reflect_to_f32(value: &dyn PartialReflect) -> Option<f32> {
    let reflect = value.try_as_reflect()?;
    if let Some(v) = reflect.downcast_ref::<f32>() {
        return Some(*v);
    }
    if let Some(v) = reflect.downcast_ref::<f64>() {
        return Some(*v as f32);
    }
    if let Some(v) = reflect.downcast_ref::<i32>() {
        return Some(*v as f32);
    }
    if let Some(v) = reflect.downcast_ref::<i64>() {
        return Some(*v as f32);
    }
    if let Some(v) = reflect.downcast_ref::<u32>() {
        return Some(*v as f32);
    }
    if let Some(v) = reflect.downcast_ref::<u64>() {
        return Some(*v as f32);
    }
    if let Some(v) = reflect.downcast_ref::<usize>() {
        return Some(*v as f32);
    }
    None
}

/// Best-effort read of a reflected `bool`.
pub(crate) fn reflect_to_bool(value: &dyn PartialReflect) -> Option<bool> {
    value.try_as_reflect()?.downcast_ref::<bool>().copied()
}

/// Hands `f` an immutable reference to a value reached from the world however the app likes.
///
/// A plain `fn` rather than a boxed closure so an [`InspectorRoot`] stays `Clone + Eq + Hash`;
/// anything the resolver needs to know (which item is selected, say) it reads from the world.
pub type CustomRead = fn(&World, &mut dyn FnMut(&dyn Reflect));

/// The mutable counterpart of [`CustomRead`]. Reach the value through whatever marks its owner
/// changed — `ResMut`, `Assets::get_mut` — or edits will not be seen.
pub type CustomWrite = fn(&mut World, &mut dyn FnMut(&mut dyn Reflect));

/// Identifies the reflected value that an inspector widget edits.
///
/// `PartialEq`/`Hash` are written out rather than derived because of [`InspectorRoot::Custom`]:
/// a derive would compare its `fn` fields directly, which Rust warns about because a function's
/// address is not a stable identity — the same function can have different addresses in different
/// codegen units, and distinct functions can be merged to one address.
#[derive(Clone, Copy)]
pub enum InspectorRoot {
    /// A component on a specific entity.
    Component {
        /// The entity holding the component.
        entity: Entity,
        /// The component's registered type.
        type_id: TypeId,
    },
    /// A resource.
    Resource {
        /// The resource's registered type.
        type_id: TypeId,
    },
    /// A loaded asset, addressed by id so the binding outlives any particular handle.
    ///
    /// Reaching one needs [`ReflectAsset`], which a type carries only if it was registered with
    /// `register_asset_reflect` rather than a plain `init_asset`.
    Asset {
        /// Which asset, within its collection.
        asset_id: UntypedAssetId,
        /// The asset's registered type.
        type_id: TypeId,
    },
    /// Anything else, reached by a pair of resolver functions.
    ///
    /// The escape hatch for data the other three roots cannot address: a value behind a
    /// `#[reflect(ignore)]` field, an element of a collection keyed by something a
    /// [`ParsedPath`] cannot spell, or a `dyn` trait object whose concrete type is only known
    /// at runtime. The path is still applied to whatever the resolver yields, so nesting,
    /// enums and lists all work below it as usual.
    Custom {
        /// Resolves the value for reading (drawing the UI, and the external-sync refresh).
        read: CustomRead,
        /// Resolves it for writing. Must go through something that marks the owner changed.
        write: CustomWrite,
    },
}

// Comparing two `Custom` roots means comparing function addresses, which the language does not
// promise anything about. `fn_addr_eq` is the sanctioned way to ask anyway, and hashing casts the
// same pointers — so `a == b` still implies `hash(a) == hash(b)`, which is the invariant that
// matters for using a root as a map key.
//
// What this does NOT promise is that two roots built from the same function always compare equal.
// The consequence is bounded: a panel may rebuild when it could have been reused. If a caller ever
// needs custom roots distinguished reliably, give them an explicit id field and compare on that
// rather than leaning on the addresses.
impl PartialEq for InspectorRoot {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Component { entity, type_id },
                Self::Component {
                    entity: other_entity,
                    type_id: other_type,
                },
            ) => entity == other_entity && type_id == other_type,
            (Self::Resource { type_id }, Self::Resource { type_id: other }) => type_id == other,
            (
                Self::Asset { asset_id, type_id },
                Self::Asset {
                    asset_id: other_asset,
                    type_id: other_type,
                },
            ) => asset_id == other_asset && type_id == other_type,
            (
                Self::Custom { read, write },
                Self::Custom {
                    read: other_read,
                    write: other_write,
                },
            ) => {
                core::ptr::fn_addr_eq(*read, *other_read)
                    && core::ptr::fn_addr_eq(*write, *other_write)
            }
            _ => false,
        }
    }
}

impl Eq for InspectorRoot {}

impl core::hash::Hash for InspectorRoot {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Component { entity, type_id } => {
                entity.hash(state);
                type_id.hash(state);
            }
            Self::Resource { type_id } => type_id.hash(state),
            Self::Asset { asset_id, type_id } => {
                asset_id.hash(state);
                type_id.hash(state);
            }
            Self::Custom { read, write } => {
                (*read as *const ()).hash(state);
                (*write as *const ()).hash(state);
            }
        }
    }
}

/// Placed on every leaf widget entity so its change observer can write back to the source data.
#[derive(Component, Clone)]
pub struct InspectorBinding {
    /// Where the edited value lives, or `None` for an unbound/placeholder widget.
    pub root: Option<InspectorRoot>,
    /// Reflection path from the root down to the edited field.
    pub path: ParsedPath,
}

impl Default for InspectorBinding {
    fn default() -> Self {
        Self {
            root: None,
            path: ParsedPath(Vec::new()),
        }
    }
}

/// Writeback observer for a `bool` field (from a checkbox's `ValueChange<bool>`).
pub fn inspector_writeback_bool(
    event: On<ValueChange<bool>>,
    bindings: Query<&InspectorBinding>,
    mut commands: Commands,
) {
    if let Ok(binding) = bindings.get(event.source) {
        queue_edit(&mut commands, binding, Box::new(event.value), true);
    }
}

/// Writeback observer for any numeric field edited by a slider (`ValueChange<f32>`).
///
/// The slider always emits `f32`; `T::from_slider_f32` converts it back to the field's real type
/// (rounding for integers) so the reflected value keeps its original type.
pub fn inspector_writeback_slider<T: SliderScalar>(
    event: On<ValueChange<f32>>,
    bindings: Query<&InspectorBinding>,
    mut commands: Commands,
) {
    if let Ok(binding) = bindings.get(event.source) {
        let value = T::from_slider_f32(event.value);
        queue_edit(&mut commands, binding, Box::new(value), event.is_final);
    }
}

/// Resolve `root` + `path` to a mutable reflected reference and hand it to `f`.
///
/// Going through `Mut`'s `DerefMut` marks the component/resource changed, so `Changed<T>` fires.
/// Shared by every writeback path (scalars, enum variant switches, list edits).
pub(crate) fn with_field_reflect_mut(
    world: &mut World,
    root: &InspectorRoot,
    path: &ParsedPath,
    f: impl FnOnce(&mut dyn PartialReflect),
) {
    // Resolved by the app, and it needs `&mut World`, so take it before the registry guard.
    if let InspectorRoot::Custom { write, .. } = root {
        // Same `FnOnce` into `FnMut` hand-off as the read side.
        let mut f = Some(f);
        let mut visit = |value: &mut dyn Reflect| {
            let Ok(target) = value.reflect_path_mut(path) else {
                return;
            };
            let Some(f) = f.take() else { return };
            f(target);
        };
        write(world, &mut visit);
        return;
    }
    // Clone the `Arc` so the read guard does not borrow `world`, leaving it free for `entity_mut`.
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();

    match root {
        InspectorRoot::Component { entity, type_id } => {
            let Some(registration) = registry.get(*type_id) else {
                return;
            };
            let Some(reflect_component) = registration.data::<ReflectComponent>() else {
                return;
            };
            let Ok(entity_mut) = world.get_entity_mut(*entity) else {
                return;
            };
            let Some(mut reflected) = reflect_component.reflect_mut(entity_mut) else {
                return;
            };
            if let Ok(target) = reflected.reflect_path_mut(path) {
                f(target);
            }
        }
        InspectorRoot::Resource { type_id } => {
            // Resources are stored on their own entity; reuse `ReflectComponent` against it.
            let Some(registration) = registry.get(*type_id) else {
                return;
            };
            let Some(reflect_component) = registration.data::<ReflectComponent>() else {
                return;
            };
            let Some(resource_entity) = world
                .components()
                .get_id(*type_id)
                .and_then(|id| world.resource_entities().get(id))
            else {
                return;
            };
            let Ok(entity_mut) = world.get_entity_mut(resource_entity) else {
                return;
            };
            let Some(mut reflected) = reflect_component.reflect_mut(entity_mut) else {
                return;
            };
            if let Ok(target) = reflected.reflect_path_mut(path) {
                f(target);
            }
        }
        InspectorRoot::Asset { asset_id, type_id } => {
            // `ReflectAsset::get_mut` borrows the world, so the registration is cloned out
            // first — the same dance the component branch does with the registry guard.
            let Some(reflect_asset) = registry
                .get(*type_id)
                .and_then(|r| r.data::<ReflectAsset>())
            else {
                return;
            };
            let reflect_asset = reflect_asset.clone();
            drop(registry);
            let Some(reflected) = reflect_asset.get_mut(world, *asset_id) else {
                return;
            };
            if let Ok(target) = reflected.reflect_path_mut(path) {
                f(target);
            }
        }
        InspectorRoot::Custom { .. } => unreachable!("handled above"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::resource::Resource;
    use bevy_reflect::Reflect;

    #[derive(Reflect, Default, PartialEq, Debug)]
    struct Inner {
        gain: f32,
        on: bool,
    }

    /// The shape `InspectorRoot::Custom` exists for: the value is behind a field no reflection
    /// path can traverse, so only hand-written resolvers can reach it.
    #[derive(Resource, Reflect, Default)]
    #[reflect(Resource)]
    struct Owner {
        #[reflect(ignore)]
        hidden: Inner,
    }

    fn read(world: &World, visit: &mut dyn FnMut(&dyn Reflect)) {
        if let Some(owner) = world.get_resource::<Owner>() {
            visit(&owner.hidden);
        }
    }

    fn write(world: &mut World, visit: &mut dyn FnMut(&mut dyn Reflect)) {
        if let Some(mut owner) = world.get_resource_mut::<Owner>() {
            visit(&mut owner.hidden);
        }
    }

    fn root() -> InspectorRoot {
        InspectorRoot::Custom { read, write }
    }

    #[test]
    fn a_custom_root_reads_and_writes_through_a_path() {
        let mut world = World::new();
        world.insert_resource(AppTypeRegistry::default());
        world.insert_resource(Owner {
            hidden: Inner {
                gain: 0.5,
                on: false,
            },
        });

        let path = ParsedPath::parse("gain").unwrap();
        let before = read_field(&world, &root(), &path, |value| reflect_to_f32(value));
        assert_eq!(before, Some(Some(0.5)));

        with_field_reflect_mut(&mut world, &root(), &path, |target| {
            let _ = target.try_apply(2.5f32.as_partial_reflect());
        });
        assert_eq!(world.resource::<Owner>().hidden.gain, 2.5);
    }

    #[test]
    fn a_custom_root_that_resolves_to_nothing_is_not_an_error() {
        // No `Owner` in the world: the resolver never calls back, and both sides no-op rather
        // than unwrapping something absent.
        let mut world = World::new();
        world.insert_resource(AppTypeRegistry::default());
        let path = ParsedPath::parse("gain").unwrap();
        assert!(read_field(&world, &root(), &path, |_| ()).is_none());
        with_field_reflect_mut(&mut world, &root(), &path, |_| {
            panic!("must not be called when the value cannot be reached");
        });
    }

    #[test]
    fn a_bad_path_does_not_reach_the_writer() {
        let mut world = World::new();
        world.insert_resource(AppTypeRegistry::default());
        world.insert_resource(Owner::default());
        let path = ParsedPath::parse("nope").unwrap();
        assert!(read_field(&world, &root(), &path, |_| ()).is_none());
        with_field_reflect_mut(&mut world, &root(), &path, |_| {
            panic!("a path that does not resolve must not yield a target");
        });
    }
}
