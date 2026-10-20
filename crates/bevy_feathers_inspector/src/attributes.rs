//! Per-field configuration read from native `#[reflect(@...)]` custom attributes.
//!
//! Rather than a bespoke derive, the inspector reuses Bevy's own reflection attributes. A field
//! annotated `#[reflect(@0.0..=1.0)]` gets a bounded slider, `#[reflect(@Step(0.01))]` its step
//! (dragging and nudging a number by that much); `#[reflect(@ReadOnly)]` /
//! `#[reflect(@Hidden)]` (the marker types below) control visibility and editability.

use core::ops::RangeInclusive;

use bevy_reflect::{NamedField, Reflect};

/// Marker attribute: render the field read-only (display its value, no editing widget).
///
/// ```ignore
/// #[reflect(@ReadOnly)]
/// id: u32,
/// ```
#[derive(Reflect)]
pub struct ReadOnly;

/// Marker attribute: omit the field from the inspector entirely.
///
/// ```ignore
/// #[reflect(@Hidden)]
/// internal: f32,
/// ```
#[derive(Reflect)]
pub struct Hidden;

/// The step a number moves by when dragged or nudged, and a slider snaps to.
///
/// ```ignore
/// #[reflect(@Step(0.01))]
/// scattering: Vec3,
/// ```
#[derive(Reflect, Clone, Copy)]
pub struct Step(pub f32);

/// Per-field context threaded through the recursion, derived from a field's custom attributes.
#[derive(Default, Clone)]
pub struct FieldCtx {
    /// An explicit numeric range from `#[reflect(@min..=max)]`, applied to sliders.
    pub range: Option<(f32, f32)>,
    /// The step from `#[reflect(@Step(..))]`: number inputs (each axis of a vector) and sliders.
    pub step: Option<f32>,
    /// Whether the field is read-only.
    pub read_only: bool,
}

impl FieldCtx {
    /// Extract a [`FieldCtx`] from a struct/tuple field's reflection metadata.
    pub fn from_field(field: &NamedField) -> Self {
        Self {
            range: range_from_field(field),
            step: field.get_attribute::<Step>().map(|s| s.0),
            read_only: field.has_attribute::<ReadOnly>(),
        }
    }

    /// Whether a field with these attributes should be skipped entirely.
    pub fn is_hidden(field: &NamedField) -> bool {
        field.has_attribute::<Hidden>()
    }
}

/// Read an inclusive numeric range attribute (`f32` or `f64` form) into an `(f32, f32)` pair.
fn range_from_field(field: &NamedField) -> Option<(f32, f32)> {
    if let Some(range) = field.get_attribute::<RangeInclusive<f32>>() {
        return Some((*range.start(), *range.end()));
    }
    if let Some(range) = field.get_attribute::<RangeInclusive<f64>>() {
        return Some((*range.start() as f32, *range.end() as f32));
    }
    None
}
