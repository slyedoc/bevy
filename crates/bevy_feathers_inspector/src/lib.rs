//! `bevy_feathers_inspector` builds reflection-driven property editors out of
//! [`bevy_feathers`](https://docs.rs/bevy_feathers) widgets — a feathers-native counterpart to
//! `bevy-inspector-egui`.
//!
//! Derive [`Reflect`](bevy_reflect::Reflect) on a component, register it, and the inspector can
//! generate a live editing UI for it. The design has two open extension channels keyed in the
//! [`TypeRegistry`](bevy_reflect::TypeRegistry):
//!
//! * [`ReflectInspectorWidget`] — a per-type widget builder (the leaf override), and
//! * the structural recursion in [`build_value`], which emits a labeled row per field.
//!
//! Edits are written back through reflection paths by a generic observer
//! ([`inspector_writeback`]), so `Changed<T>` fires on real edits.
//!
//! ## Warning: Experimental!
//! Like `bevy_feathers` itself, this crate is early and will change in breaking ways.

extern crate alloc;

pub mod binding;
pub mod entry;
pub mod recurse;
pub mod widget;

pub use binding::{inspector_writeback, InspectorBinding, InspectorRoot};
pub use entry::{build_entity_inspector, BuildEntityInspector};
pub use recurse::build_value;
pub use widget::{
    DefaultInspectorWidgetsPlugin, FeathersInspectorPlugins, ReflectInspectorWidget,
};
