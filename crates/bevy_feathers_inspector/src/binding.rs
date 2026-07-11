//! Root addressing and the generic writeback observer.
//!
//! Every leaf widget spawned by the inspector carries an [`InspectorBinding`] that records how to
//! get from the widget back to the data it edits: an [`InspectorRoot`] (which component, on which
//! entity) plus a reflection [`ParsedPath`] from that root down to the edited field. When a widget
//! emits a [`ValueChange<T>`], [`inspector_writeback`] resolves the binding and applies the new
//! value through reflection, driving change detection on the target component.

use core::any::TypeId;

use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_reflect::{GetPath, ParsedPath, PartialReflect};
use bevy_ui_widgets::ValueChange;

/// Identifies the reflected value that an inspector widget edits.
#[derive(Clone)]
pub enum InspectorRoot {
    /// A component on a specific entity.
    Component {
        /// The entity holding the component.
        entity: Entity,
        /// The component's registered type.
        type_id: TypeId,
    },
    // `Resource { type_id }` is added in Phase 3.
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

/// Generic observer that writes a widget's new value back into the reflected source field.
///
/// One monomorphization is attached per leaf widget type (e.g. `inspector_writeback::<f32>` for a
/// number input, `inspector_writeback::<bool>` for a checkbox). The actual reflection mutation is
/// deferred into a command so it can take exclusive `&mut World` access.
pub fn inspector_writeback<T: PartialReflect + Clone>(
    event: On<ValueChange<T>>,
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
    let value = event.value.clone();
    commands.queue(move |world: &mut World| {
        apply_writeback(world, &root, &path, &value);
    });
}

/// Resolve `root` to a mutable reflected reference and apply `value` at `path`.
fn apply_writeback(
    world: &mut World,
    root: &InspectorRoot,
    path: &ParsedPath,
    value: &dyn PartialReflect,
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
            // `reflect_path_mut` goes through `Mut`'s `DerefMut`, so change detection fires.
            if let Ok(target) = reflected.reflect_path_mut(path) {
                let _ = target.try_apply(value);
            }
        }
    }
}
