//! `bevy_feathers_inspector` builds reflection-driven property editors out of
//! [`bevy_feathers`](https://docs.rs/bevy_feathers) widgets — a feathers-native counterpart to
//! `bevy-inspector-egui`.
//!
//! Derive [`Reflect`](bevy_reflect::Reflect) on a component, register it, and the inspector can
//! generate a live editing UI for it. The design has two open extension channels keyed in the
//! [`TypeRegistry`](bevy_reflect::TypeRegistry):
//!
//! * [`ReflectInspectorWidget`] — a per-type widget builder (the leaf override), and
//! * the structural recursion in [`build_value`], which emits a labeled row per field, plus
//!   dedicated handling for enums (variant selector) and lists (add/remove).
//!
//! Edits are written back through reflection paths by writeback observers, so `Changed<T>` fires
//! on real edits. Per-field configuration reuses native `#[reflect(@...)]` attributes: a numeric
//! range gives a bounded slider, and the [`ReadOnly`]/[`Hidden`] markers control fields.
//!
//! ## Warning: Experimental!
//! Like `bevy_feathers` itself, this crate is early and will change in breaking ways.

extern crate alloc;

pub mod attributes;
pub mod binding;
pub mod entry;
pub mod enums;
pub mod lists;
pub mod recurse;
pub mod sync;
pub mod widget;
pub mod world_inspector;

pub use attributes::{FieldCtx, Hidden, ReadOnly};
pub use binding::{
    inspector_writeback_bool, inspector_writeback_slider, InspectorBinding, InspectorRoot,
};
pub use entry::{
    build_entity_inspector, build_resource_inspector, find_ancestor_panel, rebuild_panel,
    BuildEntityInspector, BuildResourceInspector, InspectorPanel,
};
pub use recurse::{build_value, BuildCx};
pub use widget::{
    DefaultInspectorWidgetsPlugin, FeathersInspectorPlugins, ReflectInspectorWidget, SliderScalar,
};
pub use world_inspector::{WorldInspectorPlugin, WorldInspectorState};
