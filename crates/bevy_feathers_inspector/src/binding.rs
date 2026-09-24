//! Root addressing and the writeback observers.
//!
//! Every leaf widget spawned by the inspector carries an [`InspectorBinding`] that records how to
//! get from the widget back to the data it edits: an [`InspectorRoot`] (which component, on which
//! entity) plus a reflection [`ParsedPath`] from that root down to the edited field. When a widget
//! emits a [`ValueChange<T>`], a writeback observer resolves the binding and applies the new value
//! through reflection, driving change detection on the target component.

use core::any::TypeId;

use bevy_ecs::prelude::*;
use bevy_asset::{ReflectAsset, UntypedAssetId};
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_reflect::{GetPath, ParsedPath, PartialReflect};
use bevy_ui_widgets::ValueChange;

use crate::widget::SliderScalar;

/// Resolve `root` + `path` to an immutable reflected reference and hand it to `f`.
///
/// Used by the external-sync systems to read the current source value back out.
pub(crate) fn read_field<R>(
    world: &World,
    root: &InspectorRoot,
    path: &ParsedPath,
    f: impl FnOnce(&dyn PartialReflect) -> R,
) -> Option<R> {
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
        InspectorRoot::Asset { .. } => unreachable!("handled above"),
    };
    let type_id = match root {
        InspectorRoot::Component { type_id, .. }
        | InspectorRoot::Resource { type_id }
        | InspectorRoot::Asset { type_id, .. } => *type_id,
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

/// Identifies the reflected value that an inspector widget edits.
#[derive(Clone, PartialEq, Eq, Hash)]
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
    let Ok(binding) = bindings.get(event.source) else {
        return;
    };
    let Some(root) = binding.root.clone() else {
        return;
    };
    let path = binding.path.clone();
    let value = event.value;
    commands.queue(move |world: &mut World| {
        with_field_reflect_mut(world, &root, &path, |target| {
            let _ = target.try_apply(value.as_partial_reflect());
        });
    });
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
    let Ok(binding) = bindings.get(event.source) else {
        return;
    };
    let Some(root) = binding.root.clone() else {
        return;
    };
    let path = binding.path.clone();
    let value = T::from_slider_f32(event.value);
    commands.queue(move |world: &mut World| {
        with_field_reflect_mut(world, &root, &path, |target| {
            let _ = target.try_apply(value.as_partial_reflect());
        });
    });
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
            let Some(reflect_asset) = registry.get(*type_id).and_then(|r| r.data::<ReflectAsset>())
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
    }
}
