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
//! Every card — a component section, or a nested struct/enum/list field — has a disclosure chevron
//! that hides its body, so a deep component can be narrowed to the part you care about. See
//! [`InspectorCollapsed`].
//!
//! ## Warning: Experimental!
//! Like `bevy_feathers` itself, this crate is early and will change in breaking ways.

extern crate alloc;

pub mod attributes;
pub mod binding;
pub mod collapse;
pub mod color;
pub mod entry;
pub mod enums;
pub mod lists;
pub mod number;
pub mod recurse;
pub mod sync;
pub mod text;
pub mod val;
pub mod widget;

pub use attributes::{FieldCtx, Hidden, ReadOnly, Step};
pub use binding::{
    apply_edit, assign, clone_field, inspector_writeback_bool, inspector_writeback_slider,
    queue_edit, read_field, CustomRead, CustomWrite, InspectorBinding, InspectorEdits,
    InspectorRoot,
};
pub use collapse::InspectorCollapsed;
pub use color::{color_picker, ColorPicker};
pub use entry::{
    build_asset_inspector, build_body, build_body_at, build_component_inspector, build_field, build_custom_inspector,
    build_entity_inspector, build_resource_inspector, find_ancestor_panel, rebuild_panel,
    BuildAssetInspector, BuildComponentInspector, BuildCustomInspector, BuildEntityInspector,
    BuildResourceInspector, InspectorPanel,
};
pub use enums::set_variant;
pub use recurse::{build_value, BuildCx, InspectorField};
pub use widget::{
    DefaultInspectorWidgetsPlugin, FeathersInspectorPlugins, ReflectInspectorWidget, SliderScalar,
};
