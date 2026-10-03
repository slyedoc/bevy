//! Turning a `.bsn` value into a reflected value for a known destination type.
//!
//! Everything here runs at build time. The produced values are only ever fed to
//! [`PartialReflect::try_apply`], which visits the *incoming* value's fields, so a partial value —
//! a `DynamicStruct` with two of five fields, a `DynamicTupleStruct` with only its leading field —
//! patches exactly those fields and leaves the rest alone. That is what makes a `.bsn` patch behave
//! like the equivalent `bsn!` patch.

use alloc::borrow::Cow;
use core::any::TypeId;

use bevy_asset::{AssetPath, ReflectHandle};
use bevy_bsn::{BsnPath, BsnValue, BsnValueId, Span};
use bevy_ecs::{
    reflect::ReflectTemplate,
    template::{EntityTemplate, SceneEntityReference},
};
use bevy_platform::collections::HashSet;
use bevy_reflect::{
    convert::ReflectConvert,
    enums::{DynamicEnum, DynamicVariant, VariantInfo},
    list::DynamicList,
    std_traits::ReflectDefault,
    structs::DynamicStruct,
    tuple::DynamicTuple,
    tuple_struct::DynamicTupleStruct,
    PartialReflect, Reflect, TypeInfo, TypeRegistration, TypeRegistry,
};

use crate::build::{malformed, resolve_symbol, BuildCx, DynamicSceneBuildError};

/// The body that followed a path in the source: for an enum variant, the fields supplied for it.
#[derive(Clone, Copy)]
pub(crate) enum EnumInput<'a> {
    /// `Foo::Qux`
    Unit,
    /// `Foo::Bar { x: 1 }`
    Named(&'a [(String, BsnValueId)]),
    /// `Foo::Baz(1)`
    Tuple(&'a [BsnValueId]),
}

/// Builds a reflected value for `value` that can be applied to a destination of type `expected`.
///
/// `expected` is the *template-side* type of the destination (e.g. `HandleTemplate<Image>` rather
/// than `Handle<Image>`), because values are always built for a template.
pub(crate) fn build_value(
    cx: &mut BuildCx,
    value: BsnValueId,
    expected: &TypeRegistration,
) -> Result<Box<dyn PartialReflect>, DynamicSceneBuildError> {
    let node = cx
        .document
        .value(value)
        .ok_or_else(|| malformed(format!("no value with id {}", value.0), Span::NONE))?;
    let span = node.span;
    cx.enter(span)?;
    let result = build_value_direct(cx, &node.value, expected, span);

    // "Optionish" destinations accept their payload directly, mirroring `impl<T> From<T> for
    // Option<T>` and `impl<T> From<T> for OptionTemplate<T>` plus the `bsn!` macro's implicit
    // `.into()`. Explicit `None` / `Some(x)` never reach this: they match by variant name first.
    let built = match result {
        Err(err @ DynamicSceneBuildError::ValueTypeMismatch { .. }) => {
            match optionish_payload(cx.registry, expected) {
                Some(payload) => match build_value(cx, value, payload) {
                    Ok(inner) => {
                        let mut tuple = DynamicTuple::default();
                        tuple.insert_boxed(inner);
                        let mut dynamic = DynamicEnum::new("Some", DynamicVariant::Tuple(tuple));
                        dynamic.set_represented_type(Some(expected.type_info()));
                        Ok(Box::new(dynamic) as Box<dyn PartialReflect>)
                    }
                    Err(_) => Err(err),
                },
                None => Err(err),
            }
        }
        other => other,
    }?;

    cx.exit();
    Ok(built)
}

/// [`build_value`] without the "optionish" fallback.
fn build_value_direct(
    cx: &mut BuildCx,
    node: &BsnValue,
    expected: &TypeRegistration,
    span: Span,
) -> Result<Box<dyn PartialReflect>, DynamicSceneBuildError> {
    match node {
        BsnValue::Unit => {
            if expected.type_id() == TypeId::of::<()>() {
                Ok(Box::new(()))
            } else {
                Err(mismatch("()", expected, span))
            }
        }
        BsnValue::Bool(literal) => {
            if expected.type_id() == TypeId::of::<bool>() {
                Ok(Box::new(*literal))
            } else {
                coerce(Box::new(*literal), expected, span)
            }
        }
        BsnValue::Int(literal) => build_int(*literal, expected, span),
        BsnValue::Float(literal) => build_float(*literal, expected, span),
        BsnValue::String(literal) => build_string(cx, literal, expected, span),
        // A bare path is a unit struct or a unit enum variant.
        BsnValue::Path(path) => build_named(cx, path, EnumInput::Unit, expected, span),
        BsnValue::Tuple(items) => build_tuple(cx, items, expected, span),
        BsnValue::List(items) => build_list(cx, items, expected, span),
        BsnValue::Struct(path, fields) => {
            build_named(cx, path, EnumInput::Named(fields), expected, span)
        }
        BsnValue::NamedTuple(path, items) => {
            build_named(cx, path, EnumInput::Tuple(items), expected, span)
        }
        BsnValue::EntityRef(name) => build_entity_ref(cx, name, expected, span),
    }
}

/// The shared coercion tail: exact type, then a registered conversion, then failure.
///
/// A registered conversion is what turns `"image.png"` into `HandleTemplate::Path(…)` and
/// `TextSize::Large` into `FontSize(24)`, the reflection equivalent of the `bsn!` macro's implicit
/// `.into()` on every value.
fn coerce(
    produced: Box<dyn Reflect>,
    expected: &TypeRegistration,
    span: Span,
) -> Result<Box<dyn PartialReflect>, DynamicSceneBuildError> {
    if (*produced).type_id() == expected.type_id() {
        return Ok(produced.into_partial_reflect());
    }

    let mut produced = produced;
    if let Some(convert) = expected.data::<ReflectConvert>() {
        match convert.try_convert_from(produced) {
            Ok(converted) => return Ok(converted.into_partial_reflect()),
            Err(original) => produced = original,
        }
    }

    Err(mismatch(produced.reflect_type_path(), expected, span))
}

/// Builds a [`DynamicSceneBuildError::ValueTypeMismatch`].
fn mismatch(found: &str, expected: &TypeRegistration, span: Span) -> DynamicSceneBuildError {
    DynamicSceneBuildError::ValueTypeMismatch {
        found: found.to_string(),
        expected: expected.type_info().type_path().to_string(),
        span,
    }
}

/// If `expected` is an `Option`-shaped enum — exactly a unit `None` and a one-field tuple `Some` —
/// returns the registration of `Some`'s payload type.
///
/// The test is deliberately structural rather than `Option`-specific, so that it also covers
/// `OptionTemplate<T>`, which is what a `#[template(built_in)]` `Option` field uses.
fn optionish_payload<'a>(
    registry: &'a TypeRegistry,
    expected: &TypeRegistration,
) -> Option<&'a TypeRegistration> {
    let TypeInfo::Enum(info) = expected.type_info() else {
        return None;
    };
    if info.variant_len() != 2 {
        return None;
    }
    if !matches!(info.variant("None")?, VariantInfo::Unit(_)) {
        return None;
    }
    let VariantInfo::Tuple(some) = info.variant("Some")? else {
        return None;
    };
    if some.field_len() != 1 {
        return None;
    }
    registry.get(some.field_at(0)?.ty().id())
}

/// Integer literals, range-checked against the destination type.
///
/// Integer-to-float is allowed but only when the value is exactly representable, so writing `1`
/// where `1.0` was meant never silently loses precision. Float-to-integer is never allowed.
fn build_int(
    literal: i128,
    expected: &TypeRegistration,
    span: Span,
) -> Result<Box<dyn PartialReflect>, DynamicSceneBuildError> {
    let type_id = expected.type_id();

    macro_rules! integer {
        ($($ty:ty),*) => {$(
            if type_id == TypeId::of::<$ty>() {
                return <$ty>::try_from(literal)
                    .map(|value| Box::new(value) as Box<dyn PartialReflect>)
                    .map_err(|_| DynamicSceneBuildError::IntegerOutOfRange {
                        value: literal,
                        type_path: expected.type_info().type_path().to_string(),
                        span,
                    });
            }
        )*};
    }
    integer!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

    macro_rules! float {
        ($ty:ty, $exact:expr) => {
            if type_id == TypeId::of::<$ty>() {
                if literal.unsigned_abs() > $exact {
                    return Err(DynamicSceneBuildError::LiteralNotRepresentable {
                        value: literal,
                        type_path: expected.type_info().type_path().to_string(),
                        span,
                    });
                }
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "the magnitude was just checked to be exactly representable"
                )]
                return Ok(Box::new(literal as $ty));
            }
        };
    }
    // 2^24 and 2^53: the largest magnitudes with an exact representation in each format.
    float!(f32, 1u128 << 24);
    float!(f64, 1u128 << 53);

    // Anything else has to go through a registered conversion. The literal is boxed as its natural
    // type so that the conversion is looked up by the source type an author would expect.
    let natural: Box<dyn Reflect> = match i64::try_from(literal) {
        Ok(literal) => Box::new(literal),
        Err(_) => Box::new(literal),
    };
    coerce(natural, expected, span)
}

/// Floating-point literals. Rounding `f64` to `f32` is allowed (it is what a Rust literal does);
/// only rounding a finite value to a non-finite one is rejected.
fn build_float(
    literal: f64,
    expected: &TypeRegistration,
    span: Span,
) -> Result<Box<dyn PartialReflect>, DynamicSceneBuildError> {
    let type_id = expected.type_id();
    if type_id == TypeId::of::<f64>() {
        return Ok(Box::new(literal));
    }
    if type_id == TypeId::of::<f32>() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "rounding a float literal to f32 is intended; only overflow to infinity is rejected"
        )]
        let value = literal as f32;
        if literal.is_finite() && !value.is_finite() {
            return Err(mismatch("f64", expected, span));
        }
        return Ok(Box::new(value));
    }
    coerce(Box::new(literal), expected, span)
}

/// String literals: text types directly, everything else through a registered conversion — which
/// is how a `Handle<A>` field's `HandleTemplate<A>` is produced from an asset path.
fn build_string(
    cx: &mut BuildCx,
    literal: &str,
    expected: &TypeRegistration,
    span: Span,
) -> Result<Box<dyn PartialReflect>, DynamicSceneBuildError> {
    let type_id = expected.type_id();
    if type_id == TypeId::of::<String>() {
        return Ok(Box::new(literal.to_string()));
    }
    if type_id == TypeId::of::<Cow<'static, str>>() {
        return Ok(Box::new(Cow::<'static, str>::Owned(literal.to_string())));
    }
    if type_id == TypeId::of::<AssetPath<'static>>() {
        return Ok(Box::new(asset_path(literal, span)?));
    }

    // The destination is a template whose output is a `Handle<A>`. Validate the path *before*
    // coercing — the registered `String → HandleTemplate` conversion rejects malformed paths, and
    // reporting `InvalidAssetPath` with this literal's span beats the generic coercion mismatch.
    let handle_asset_type = expected
        .data::<ReflectTemplate>()
        .and_then(|reflect_template| cx.registry.get(reflect_template.output_type_id))
        .and_then(|output| output.data::<ReflectHandle>())
        .map(ReflectHandle::asset_type_id);
    let path = match handle_asset_type {
        Some(_) => Some(asset_path(literal, span)?),
        None => None,
    };

    let converted = coerce(Box::new(literal.to_string()), expected, span)?;

    // Record the asset dependency so that the `.bsn`'s assets are load-context dependencies of
    // the scene rather than being loaded lazily at spawn time.
    if let (Some(asset_type_id), Some(path)) = (handle_asset_type, path) {
        cx.dependencies.push((asset_type_id, path));
    }

    Ok(converted)
}

/// Parses an asset path, reporting a build error instead of panicking on a malformed one.
fn asset_path(literal: &str, span: Span) -> Result<AssetPath<'static>, DynamicSceneBuildError> {
    AssetPath::try_parse(literal)
        .map(AssetPath::into_owned)
        .map_err(|_| DynamicSceneBuildError::InvalidAssetPath {
            path: literal.to_string(),
            span,
        })
}

/// A value written as a path plus an optional body: `Marker`, `Foo::Qux`, `Bar { a: 1 }`, `Bar(2)`.
fn build_named(
    cx: &mut BuildCx,
    path: &BsnPath,
    body: EnumInput,
    expected: &TypeRegistration,
    span: Span,
) -> Result<Box<dyn PartialReflect>, DynamicSceneBuildError> {
    // An enum-typed destination accepts a bare variant name, which is what makes `None` and
    // `Some(5)` work without naming `core::option::Option<…>`.
    if path.is_single_segment()
        && let TypeInfo::Enum(info) = expected.type_info()
        && info.variant(path.last_ident()).is_some()
    {
        let (full, _partial) = build_enum_forms(cx, expected, path.last_ident(), body, span)?;
        return Ok(full);
    }

    let symbol = resolve_symbol(cx.registry, path, span)?;
    let named = symbol.registration;

    let partial = match symbol.variant {
        // An enum variant. `full` — every field of the variant defaulted, then the supplied fields
        // overlaid — is the right form for a *value*: a variant switch has to produce a complete
        // value. (The two-form treatment is only needed for top-level patches.)
        Some(variant) => Some(build_enum_forms(cx, named, variant, body, span)?.0),
        None => match body {
            EnumInput::Unit => None,
            EnumInput::Named(fields) => Some(build_partial_struct(cx, named, fields, span)?),
            EnumInput::Tuple(items) => Some(build_partial_tuple_struct(cx, named, items, span)?),
        },
    };

    // Same type as the destination: hand back the *partial* value, so that
    // `Foo { nested: Bar(2) }` patches only `Bar`'s first field and leaves the rest of the nested
    // value alone. This is the reflection equivalent of the macro's nested path assignment.
    if named.type_id() == expected.type_id()
        && let Some(partial) = partial
    {
        return Ok(partial);
    }

    // A different type: construct it fully, then convert. This mirrors the macro's `(#ty).into()`.
    let mut value = default_value(named, span)?;
    if let Some(partial) = &partial {
        value
            .try_apply(&**partial)
            .map_err(|error| DynamicSceneBuildError::ValueApplyFailed {
                type_path: named.type_info().type_path().to_string(),
                error: error.to_string(),
                span,
            })?;
    }
    coerce(value, expected, span)
}

/// Default-constructs a value of a registered type.
fn default_value(
    registration: &TypeRegistration,
    span: Span,
) -> Result<Box<dyn Reflect>, DynamicSceneBuildError> {
    registration
        .data::<ReflectDefault>()
        .map(ReflectDefault::default)
        .ok_or_else(|| DynamicSceneBuildError::MissingReflectDefault {
            type_path: registration.type_info().type_path().to_string(),
            span,
        })
}

/// Builds a partial [`DynamicStruct`] holding exactly the supplied fields.
///
/// Unknown and duplicated field names are hard errors, matching what the `bsn!` macro reports at
/// compile time — and deliberately *not* `try_apply`'s silent skip, which would swallow typos.
pub(crate) fn build_partial_struct(
    cx: &mut BuildCx,
    registration: &TypeRegistration,
    fields: &[(String, BsnValueId)],
    span: Span,
) -> Result<Box<dyn PartialReflect>, DynamicSceneBuildError> {
    let type_path = registration.type_info().type_path();
    let TypeInfo::Struct(info) = registration.type_info() else {
        return Err(DynamicSceneBuildError::TypeNotStruct {
            type_path: type_path.to_string(),
            span,
        });
    };

    let mut dynamic = DynamicStruct::default();
    dynamic.set_represented_type(Some(registration.type_info()));
    let mut seen = HashSet::<&str>::default();
    for (name, value) in fields {
        let field = info
            .field(name)
            .ok_or_else(|| DynamicSceneBuildError::UnknownField {
                type_path: type_path.to_string(),
                field: name.clone(),
                span,
            })?;
        if !seen.insert(name.as_str()) {
            return Err(DynamicSceneBuildError::DuplicateField {
                type_path: type_path.to_string(),
                field: name.clone(),
                span,
            });
        }
        let field_registration = cx.registration(field.ty().id(), field.type_path(), span)?;
        let value = build_value(cx, *value, field_registration)?;
        dynamic.insert_boxed(name.clone(), value);
    }

    Ok(Box::new(dynamic))
}

/// Builds a partial [`DynamicTupleStruct`] holding the supplied *leading* fields.
///
/// Trailing fields are simply absent, so `try_apply` leaves them at whatever they were — the
/// "partial leading fields" rule the `bsn!` macro gets from assigning indices `0..n`.
pub(crate) fn build_partial_tuple_struct(
    cx: &mut BuildCx,
    registration: &TypeRegistration,
    items: &[BsnValueId],
    span: Span,
) -> Result<Box<dyn PartialReflect>, DynamicSceneBuildError> {
    let type_path = registration.type_info().type_path();
    let TypeInfo::TupleStruct(info) = registration.type_info() else {
        return Err(DynamicSceneBuildError::TypeNotTupleStruct {
            type_path: type_path.to_string(),
            span,
        });
    };
    if items.len() > info.field_len() {
        return Err(DynamicSceneBuildError::TooManyTupleFields {
            type_path: type_path.to_string(),
            given: items.len(),
            expected: info.field_len(),
            span,
        });
    }

    // `zip` stops at the shorter side, which is `items`: its length was just bounds-checked.
    let mut dynamic = DynamicTupleStruct::default();
    dynamic.set_represented_type(Some(registration.type_info()));
    for (field, value) in info.iter().zip(items) {
        let field_registration = cx.registration(field.ty().id(), field.type_path(), span)?;
        dynamic.insert_boxed(build_value(cx, *value, field_registration)?);
    }

    Ok(Box::new(dynamic))
}

/// Builds the two forms of an enum-variant value.
///
/// * `full` has every non-ignored field of the variant defaulted, then the supplied fields
///   overlaid. It reproduces the generated `T::default_<variant>()` followed by the assignments,
///   and is what a variant *switch* needs: the derived enum `try_apply` requires a complete field
///   set when the variant name changes.
/// * `partial` has only the supplied fields, and is what a patch of the *same* variant needs so
///   that untouched fields survive.
///
/// Fields marked `#[reflect(ignore)]` do not appear in the variant's info at all and are
/// auto-defaulted by the generated `try_apply`, so leaving them out of `full` is correct.
pub(crate) fn build_enum_forms(
    cx: &mut BuildCx,
    registration: &TypeRegistration,
    variant: &str,
    input: EnumInput,
    span: Span,
) -> Result<(Box<dyn PartialReflect>, Box<dyn PartialReflect>), DynamicSceneBuildError> {
    let type_path = registration.type_info().type_path();
    let TypeInfo::Enum(info) = registration.type_info() else {
        return Err(DynamicSceneBuildError::UnknownVariant {
            type_path: type_path.to_string(),
            variant: variant.to_string(),
            span,
        });
    };
    let variant_info =
        info.variant(variant)
            .ok_or_else(|| DynamicSceneBuildError::UnknownVariant {
                type_path: type_path.to_string(),
                variant: variant.to_string(),
                span,
            })?;
    let variant_name = variant_info.name();

    let (full, partial) = match (variant_info, input) {
        (VariantInfo::Unit(_), EnumInput::Unit) => (DynamicVariant::Unit, DynamicVariant::Unit),
        (VariantInfo::Struct(variant_info), input) => {
            let fields: &[(String, BsnValueId)] = match input {
                EnumInput::Named(fields) => fields,
                EnumInput::Unit => &[],
                EnumInput::Tuple(_) => {
                    return Err(DynamicSceneBuildError::TypeNotTupleStruct {
                        type_path: format!("{type_path}::{variant_name}"),
                        span,
                    })
                }
            };

            let mut partial = DynamicStruct::default();
            let mut seen = HashSet::<&str>::default();
            for (name, value) in fields {
                let field = variant_info.field(name).ok_or_else(|| {
                    DynamicSceneBuildError::UnknownField {
                        type_path: format!("{type_path}::{variant_name}"),
                        field: name.clone(),
                        span,
                    }
                })?;
                if !seen.insert(name.as_str()) {
                    return Err(DynamicSceneBuildError::DuplicateField {
                        type_path: format!("{type_path}::{variant_name}"),
                        field: name.clone(),
                        span,
                    });
                }
                let field_registration =
                    cx.registration(field.ty().id(), field.type_path(), span)?;
                let value = build_value(cx, *value, field_registration)?;
                partial.insert_boxed(name.clone(), value);
            }

            let mut full = DynamicStruct::default();
            for field in variant_info.iter() {
                let field_registration =
                    cx.registration(field.ty().id(), field.type_path(), span)?;
                full.insert_boxed(
                    field.name(),
                    default_value(field_registration, span)?.into_partial_reflect(),
                );
            }
            apply_overlay(&mut full, &partial, type_path, variant_name, span)?;

            (
                DynamicVariant::Struct(full),
                DynamicVariant::Struct(partial),
            )
        }
        (VariantInfo::Tuple(variant_info), input) => {
            let items: &[BsnValueId] = match input {
                EnumInput::Tuple(items) => items,
                EnumInput::Unit => &[],
                EnumInput::Named(_) => {
                    return Err(DynamicSceneBuildError::TypeNotStruct {
                        type_path: format!("{type_path}::{variant_name}"),
                        span,
                    })
                }
            };
            if items.len() > variant_info.field_len() {
                return Err(DynamicSceneBuildError::TooManyTupleFields {
                    type_path: format!("{type_path}::{variant_name}"),
                    given: items.len(),
                    expected: variant_info.field_len(),
                    span,
                });
            }

            // `zip` stops at the shorter side, which is `items`: its length was just
            // bounds-checked.
            let mut partial = DynamicTuple::default();
            for (field, value) in variant_info.iter().zip(items) {
                let field_registration =
                    cx.registration(field.ty().id(), field.type_path(), span)?;
                partial.insert_boxed(build_value(cx, *value, field_registration)?);
            }

            let mut full = DynamicTuple::default();
            for field in variant_info.iter() {
                let field_registration =
                    cx.registration(field.ty().id(), field.type_path(), span)?;
                full.insert_boxed(default_value(field_registration, span)?.into_partial_reflect());
            }
            apply_overlay(&mut full, &partial, type_path, variant_name, span)?;

            (DynamicVariant::Tuple(full), DynamicVariant::Tuple(partial))
        }
        (VariantInfo::Unit(_), _) => {
            return Err(DynamicSceneBuildError::TooManyTupleFields {
                type_path: format!("{type_path}::{variant_name}"),
                given: 1,
                expected: 0,
                span,
            })
        }
    };

    let mut full_enum = DynamicEnum::new(variant_name, full);
    full_enum.set_represented_type(Some(registration.type_info()));
    let mut partial_enum = DynamicEnum::new(variant_name, partial);
    partial_enum.set_represented_type(Some(registration.type_info()));

    Ok((Box::new(full_enum), Box::new(partial_enum)))
}

/// Overlays the supplied fields of a variant onto the fully-defaulted form.
fn apply_overlay(
    full: &mut dyn PartialReflect,
    partial: &dyn PartialReflect,
    type_path: &str,
    variant: &str,
    span: Span,
) -> Result<(), DynamicSceneBuildError> {
    full.try_apply(partial)
        .map_err(|error| DynamicSceneBuildError::ValueApplyFailed {
            type_path: format!("{type_path}::{variant}"),
            error: error.to_string(),
            span,
        })
}

/// Tuple values, such as `(1, 2)`.
fn build_tuple(
    cx: &mut BuildCx,
    items: &[BsnValueId],
    expected: &TypeRegistration,
    span: Span,
) -> Result<Box<dyn PartialReflect>, DynamicSceneBuildError> {
    let TypeInfo::Tuple(info) = expected.type_info() else {
        return Err(mismatch("a tuple", expected, span));
    };
    if items.len() > info.field_len() {
        return Err(DynamicSceneBuildError::TooManyTupleFields {
            type_path: expected.type_info().type_path().to_string(),
            given: items.len(),
            expected: info.field_len(),
            span,
        });
    }

    // `zip` stops at the shorter side, which is `items`: its length was just bounds-checked.
    let mut dynamic = DynamicTuple::default();
    for (field, value) in info.iter().zip(items) {
        let field_registration = cx.registration(field.ty().id(), field.type_path(), span)?;
        dynamic.insert_boxed(build_value(cx, *value, field_registration)?);
    }
    dynamic.set_represented_type(Some(expected.type_info()));

    Ok(Box::new(dynamic))
}

/// List values, such as `[1, 2]`.
///
/// A single-field tuple struct whose field is a list is unwrapped first, so that `VecTemplate<T>` —
/// what a `#[template(built_in)]` `Vec` field uses — accepts a list directly.
fn build_list(
    cx: &mut BuildCx,
    items: &[BsnValueId],
    expected: &TypeRegistration,
    span: Span,
) -> Result<Box<dyn PartialReflect>, DynamicSceneBuildError> {
    if let TypeInfo::TupleStruct(info) = expected.type_info()
        && info.field_len() == 1
        && let Some(field) = info.field_at(0)
        && let Some(inner) = cx.registry.get(field.ty().id())
        && matches!(inner.type_info(), TypeInfo::List(_))
    {
        let list = build_list(cx, items, inner, span)?;
        let mut dynamic = DynamicTupleStruct::default();
        dynamic.set_represented_type(Some(expected.type_info()));
        dynamic.insert_boxed(list);
        return Ok(Box::new(dynamic));
    }

    let TypeInfo::List(info) = expected.type_info() else {
        return Err(mismatch("a list", expected, span));
    };
    let item_registration = cx.registration(info.item_ty().id(), info.item_ty().path(), span)?;

    let mut dynamic = DynamicList::default();
    dynamic.set_represented_type(Some(expected.type_info()));
    for value in items {
        dynamic.push_box(build_value(cx, *value, item_registration)?);
    }

    Ok(Box::new(dynamic))
}

/// A `#Name` reference used as a value.
fn build_entity_ref(
    cx: &mut BuildCx,
    name: &str,
    expected: &TypeRegistration,
    span: Span,
) -> Result<Box<dyn PartialReflect>, DynamicSceneBuildError> {
    let node_id = *cx
        .names
        .get(name)
        .ok_or_else(|| DynamicSceneBuildError::UnknownEntityName {
            name: name.to_string(),
            span,
        })?;
    let reference = SceneEntityReference::from_asset_hashed(cx.source_path_hash, node_id);
    let template = EntityTemplate::SceneEntityReference(reference);

    if expected.type_id() == TypeId::of::<EntityTemplate>() {
        return Ok(Box::new(template));
    }
    coerce(Box::new(template), expected, span)
}

#[cfg(test)]
mod tests {
    use bevy_asset::{AssetPath, HandleTemplate};
    use bevy_bsn::{BsnDocument, BsnNodeKind, BsnPath, BsnValue, BsnValueId};
    use bevy_ecs::template::{EntityTemplate, SceneEntityReference};
    use bevy_reflect::{Reflect, ReflectRef, TypeRegistration};

    use super::*;
    use crate::tests::{
        test_registry, Bar, Choice, Collections, FontSize, Foo, Image, Marker, TextFont, Tricky,
    };

    /// Runs `f` against a build context over `document` and a registry holding every fixture type.
    fn with_cx<R>(document: &BsnDocument, f: impl FnOnce(&mut BuildCx) -> R) -> R {
        let registry = test_registry();
        let mut cx = BuildCx::new(&registry, document, "test.bsn");
        f(&mut cx)
    }

    /// The registration of `T`, which must be in the fixture registry.
    fn reg<'a, T: 'static>(cx: &BuildCx<'a>) -> &'a TypeRegistration {
        cx.registry
            .get(TypeId::of::<T>())
            .expect("the fixture type should be registered")
    }

    /// Applies a built value onto a default `T`.
    fn applied<T: Reflect + Default>(value: &dyn PartialReflect) -> T {
        let mut target = T::default();
        target
            .try_apply(value)
            .expect("the built value should apply cleanly");
        target
    }

    /// A document holding a single value.
    fn value_doc(value: BsnValue) -> (BsnDocument, BsnValueId) {
        let mut document = BsnDocument::new();
        let id = document.push_value(value);
        (document, id)
    }

    #[test]
    fn unit_value_only_fits_the_unit_type() {
        let (document, id) = value_doc(BsnValue::Unit);
        with_cx(&document, |cx| {
            assert!(build_value(cx, id, reg::<()>(cx))
                .unwrap()
                .try_downcast_ref::<()>()
                .is_some());
            let error = build_value(cx, id, reg::<u32>(cx)).unwrap_err();
            assert!(
                matches!(&error, DynamicSceneBuildError::ValueTypeMismatch { found, .. }
                    if found == "()"),
                "unexpected error: {error}"
            );
        });
    }

    #[test]
    fn int_literal_exact_widths() {
        let (document, id) = value_doc(BsnValue::Int(1));
        with_cx(&document, |cx| {
            macro_rules! check {
                ($($ty:ty),*) => {$({
                    let value = build_value(cx, id, reg::<$ty>(cx)).unwrap();
                    assert_eq!(
                        *value.try_downcast_ref::<$ty>().unwrap(),
                        1 as $ty,
                        "{}",
                        core::any::type_name::<$ty>()
                    );
                })*};
            }
            check!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
        });
    }

    #[test]
    fn int_literal_out_of_range_errors() {
        let (document, id) = value_doc(BsnValue::Int(300));
        with_cx(&document, |cx| {
            assert!(matches!(
                build_value(cx, id, reg::<u8>(cx)),
                Err(DynamicSceneBuildError::IntegerOutOfRange { value: 300, .. })
            ));
        });

        let (document, id) = value_doc(BsnValue::Int(-1));
        with_cx(&document, |cx| {
            assert!(matches!(
                build_value(cx, id, reg::<u32>(cx)),
                Err(DynamicSceneBuildError::IntegerOutOfRange { value: -1, .. })
            ));
        });
    }

    #[test]
    fn int_literal_into_float_exact() {
        let (document, id) = value_doc(BsnValue::Int(1));
        with_cx(&document, |cx| {
            let value = build_value(cx, id, reg::<f32>(cx)).unwrap();
            assert_eq!(*value.try_downcast_ref::<f32>().unwrap(), 1.0);
            let value = build_value(cx, id, reg::<f64>(cx)).unwrap();
            assert_eq!(*value.try_downcast_ref::<f64>().unwrap(), 1.0);
        });
    }

    #[test]
    fn int_literal_into_float_inexact_errors() {
        let (document, id) = value_doc(BsnValue::Int(1 << 30));
        with_cx(&document, |cx| {
            assert!(matches!(
                build_value(cx, id, reg::<f32>(cx)),
                Err(DynamicSceneBuildError::LiteralNotRepresentable { .. })
            ));
            // The same literal is exactly representable as an `f64`.
            assert!(build_value(cx, id, reg::<f64>(cx)).is_ok());
        });
    }

    #[test]
    fn int_literal_beyond_i64_keeps_its_natural_type() {
        // A literal that does not fit an `i64` is offered to a registered conversion as an `i128`,
        // so the failure names the type an author would expect.
        let (document, id) = value_doc(BsnValue::Int(i128::from(u64::MAX) + 1));
        with_cx(&document, |cx| {
            let error = build_value(cx, id, reg::<Marker>(cx)).unwrap_err();
            assert!(
                matches!(&error, DynamicSceneBuildError::ValueTypeMismatch { found, .. }
                    if found == "i128"),
                "unexpected error: {error}"
            );
        });
    }

    #[test]
    fn float_literal_into_f32_and_f64() {
        let (document, id) = value_doc(BsnValue::Float(1.5));
        with_cx(&document, |cx| {
            assert_eq!(
                *build_value(cx, id, reg::<f32>(cx))
                    .unwrap()
                    .try_downcast_ref::<f32>()
                    .unwrap(),
                1.5f32
            );
            assert_eq!(
                *build_value(cx, id, reg::<f64>(cx))
                    .unwrap()
                    .try_downcast_ref::<f64>()
                    .unwrap(),
                1.5f64
            );
        });

        let (document, id) = value_doc(BsnValue::Float(1.1));
        with_cx(&document, |cx| {
            assert_eq!(
                *build_value(cx, id, reg::<f32>(cx))
                    .unwrap()
                    .try_downcast_ref::<f32>()
                    .unwrap(),
                1.1f32
            );
        });
    }

    #[test]
    fn float_literal_overflowing_f32_errors() {
        // Rounding an `f64` literal to `f32` is fine, but rounding a finite value to infinity is
        // not: that is a value the author cannot have meant.
        let (document, id) = value_doc(BsnValue::Float(1e39));
        with_cx(&document, |cx| {
            let error = build_value(cx, id, reg::<f32>(cx)).unwrap_err();
            assert!(
                matches!(&error, DynamicSceneBuildError::ValueTypeMismatch { found, .. }
                    if found == "f64"),
                "unexpected error: {error}"
            );
            // The same literal is fine as an `f64`.
            assert!(build_value(cx, id, reg::<f64>(cx)).is_ok());
        });
    }

    #[test]
    fn float_literal_into_int_errors() {
        let (document, id) = value_doc(BsnValue::Float(1.0));
        with_cx(&document, |cx| {
            assert!(matches!(
                build_value(cx, id, reg::<u32>(cx)),
                Err(DynamicSceneBuildError::ValueTypeMismatch { .. })
            ));
        });
    }

    #[test]
    fn bool_literal() {
        let (document, id) = value_doc(BsnValue::Bool(true));
        with_cx(&document, |cx| {
            assert!(*build_value(cx, id, reg::<bool>(cx))
                .unwrap()
                .try_downcast_ref::<bool>()
                .unwrap());
            assert!(matches!(
                build_value(cx, id, reg::<u32>(cx)),
                Err(DynamicSceneBuildError::ValueTypeMismatch { .. })
            ));
        });
    }

    #[test]
    fn string_into_string_and_cow() {
        let (document, id) = value_doc(BsnValue::String("x".to_string()));
        with_cx(&document, |cx| {
            assert_eq!(
                build_value(cx, id, reg::<String>(cx))
                    .unwrap()
                    .try_downcast_ref::<String>()
                    .unwrap(),
                "x"
            );
            assert_eq!(
                build_value(cx, id, reg::<Cow<'static, str>>(cx))
                    .unwrap()
                    .try_downcast_ref::<Cow<'static, str>>()
                    .unwrap(),
                "x"
            );
        });
    }

    #[test]
    fn string_into_static_str_errors() {
        let (document, id) = value_doc(BsnValue::String("x".to_string()));
        with_cx(&document, |cx| {
            assert!(matches!(
                build_value(cx, id, reg::<&'static str>(cx)),
                Err(DynamicSceneBuildError::ValueTypeMismatch { .. })
            ));
        });
    }

    #[test]
    fn string_into_asset_path() {
        let (document, id) = value_doc(BsnValue::String("a.png".to_string()));
        with_cx(&document, |cx| {
            assert_eq!(
                *build_value(cx, id, reg::<AssetPath<'static>>(cx))
                    .unwrap()
                    .try_downcast_ref::<AssetPath<'static>>()
                    .unwrap(),
                AssetPath::parse("a.png").into_owned()
            );
        });
    }

    #[test]
    fn string_into_handle_template_via_convert() {
        let (document, id) = value_doc(BsnValue::String("a.png".to_string()));
        let dependencies = with_cx(&document, |cx| {
            let value = build_value(cx, id, reg::<HandleTemplate<Image>>(cx)).unwrap();
            let template = value.try_downcast_ref::<HandleTemplate<Image>>().unwrap();
            assert!(
                matches!(template, HandleTemplate::Path(path) if *path == AssetPath::parse("a.png"))
            );
            core::mem::take(&mut cx.dependencies)
        });
        assert_eq!(dependencies.len(), 1);
        assert_eq!(dependencies[0].0, TypeId::of::<Image>());
        assert_eq!(dependencies[0].1, AssetPath::parse("a.png").into_owned());
    }

    #[test]
    fn unit_struct_value() {
        let (document, id) = value_doc(BsnValue::Path(BsnPath::from_segments(["Marker"])));
        with_cx(&document, |cx| {
            let value = build_value(cx, id, reg::<Marker>(cx)).unwrap();
            assert_eq!(applied::<Marker>(&*value), Marker);
        });
    }

    #[test]
    fn unit_enum_variant_value() {
        let (document, id) = value_doc(BsnValue::Path(BsnPath::from_segments(["Choice", "Qux"])));
        with_cx(&document, |cx| {
            let value = build_value(cx, id, reg::<Choice>(cx)).unwrap();
            // jbuehler23's fix: the value is a `DynamicEnum` naming the variant, not a bare
            // `DynamicStruct`/`DynamicTupleStruct`.
            let ReflectRef::Enum(dynamic) = value.reflect_ref() else {
                panic!("expected an enum value");
            };
            assert_eq!(dynamic.variant_name(), "Qux");
            let mut target = Choice::Baz(1);
            target.try_apply(&*value).unwrap();
            assert_eq!(target, Choice::Qux);
        });
    }

    #[test]
    fn enum_variant_value_via_reflect_convert() {
        let (document, id) = value_doc(BsnValue::Path(BsnPath::from_segments([
            "TextSize", "Large",
        ])));
        with_cx(&document, |cx| {
            let value = build_value(cx, id, reg::<FontSize>(cx)).unwrap();
            assert_eq!(*value.try_downcast_ref::<FontSize>().unwrap(), FontSize(24));
        });
    }

    #[test]
    fn struct_value_same_type_is_partial() {
        let mut document = BsnDocument::new();
        let one = document.push_value(BsnValue::Int(1));
        let id = document.push_value(BsnValue::Struct(
            BsnPath::from_segments(["Foo"]),
            vec![("x".to_string(), one)],
        ));
        with_cx(&document, |cx| {
            let value = build_value(cx, id, reg::<Foo>(cx)).unwrap();
            let ReflectRef::Struct(dynamic) = value.reflect_ref() else {
                panic!("expected a struct value");
            };
            assert_eq!(dynamic.field_len(), 1);
            // Applying it patches only `x`.
            assert_eq!(
                applied::<Foo>(&*value),
                Foo {
                    x: 1,
                    ..Default::default()
                }
            );
        });
    }

    #[test]
    fn struct_value_other_type_is_full_then_converted() {
        let mut document = BsnDocument::new();
        let thirty = document.push_value(BsnValue::Int(30));
        let id = document.push_value(BsnValue::Struct(
            BsnPath::from_segments(["FontSizeSource"]),
            vec![("value".to_string(), thirty)],
        ));
        with_cx(&document, |cx| {
            let value = build_value(cx, id, reg::<FontSize>(cx)).unwrap();
            assert_eq!(*value.try_downcast_ref::<FontSize>().unwrap(), FontSize(30));
        });
    }

    #[test]
    fn tuple_struct_partial_leading_fields() {
        let mut document = BsnDocument::new();
        let two = document.push_value(BsnValue::Int(2));
        let id = document.push_value(BsnValue::NamedTuple(
            BsnPath::from_segments(["Bar"]),
            vec![two],
        ));
        with_cx(&document, |cx| {
            let value = build_value(cx, id, reg::<Bar>(cx)).unwrap();
            let ReflectRef::TupleStruct(dynamic) = value.reflect_ref() else {
                panic!("expected a tuple-struct value");
            };
            assert_eq!(dynamic.field_len(), 1);

            let mut target = Bar(1, 1, 0);
            target.try_apply(&*value).unwrap();
            assert_eq!(target, Bar(2, 1, 0));
        });
    }

    #[test]
    fn tuple_struct_too_many_fields_errors() {
        let mut document = BsnDocument::new();
        let one = document.push_value(BsnValue::Int(1));
        let id = document.push_value(BsnValue::NamedTuple(
            BsnPath::from_segments(["Bar"]),
            vec![one, one, one, one],
        ));
        with_cx(&document, |cx| {
            assert!(matches!(
                build_value(cx, id, reg::<Bar>(cx)),
                Err(DynamicSceneBuildError::TooManyTupleFields {
                    given: 4,
                    expected: 3,
                    ..
                })
            ));
        });
    }

    #[test]
    fn unknown_field_errors() {
        let mut document = BsnDocument::new();
        let one = document.push_value(BsnValue::Int(1));
        let id = document.push_value(BsnValue::Struct(
            BsnPath::from_segments(["Foo"]),
            vec![("nope".to_string(), one)],
        ));
        with_cx(&document, |cx| {
            let error = build_value(cx, id, reg::<Foo>(cx)).unwrap_err();
            assert!(
                matches!(&error, DynamicSceneBuildError::UnknownField { field, .. } if field == "nope"),
                "unexpected error: {error}"
            );
        });
    }

    #[test]
    fn duplicate_field_errors() {
        let mut document = BsnDocument::new();
        let one = document.push_value(BsnValue::Int(1));
        let two = document.push_value(BsnValue::Int(2));
        let id = document.push_value(BsnValue::Struct(
            BsnPath::from_segments(["Foo"]),
            vec![("x".to_string(), one), ("x".to_string(), two)],
        ));
        with_cx(&document, |cx| {
            assert!(matches!(
                build_value(cx, id, reg::<Foo>(cx)),
                Err(DynamicSceneBuildError::DuplicateField { .. })
            ));
        });
    }

    #[test]
    fn struct_body_on_a_tuple_struct_errors() {
        let mut document = BsnDocument::new();
        let one = document.push_value(BsnValue::Int(1));
        let id = document.push_value(BsnValue::Struct(
            BsnPath::from_segments(["Bar"]),
            vec![("x".to_string(), one)],
        ));
        with_cx(&document, |cx| {
            let error = build_value(cx, id, reg::<Bar>(cx)).unwrap_err();
            assert!(
                matches!(&error, DynamicSceneBuildError::TypeNotStruct { type_path, .. }
                    if type_path.ends_with("Bar")),
                "unexpected error: {error}"
            );
        });
    }

    #[test]
    fn enum_forms_reject_non_enums_and_unknown_variants() {
        // Both guards protect against a component and its generated template disagreeing about
        // their shape; neither is reachable from a document built against a matching registry.
        let document = BsnDocument::new();
        with_cx(&document, |cx| {
            let error = build_enum_forms(cx, reg::<Foo>(cx), "Bar", EnumInput::Unit, Span::NONE)
                .unwrap_err();
            assert!(
                matches!(&error, DynamicSceneBuildError::UnknownVariant { type_path, .. }
                    if type_path.ends_with("Foo")),
                "unexpected error: {error}"
            );

            let error =
                build_enum_forms(cx, reg::<Choice>(cx), "Nope", EnumInput::Unit, Span::NONE)
                    .unwrap_err();
            assert!(
                matches!(&error, DynamicSceneBuildError::UnknownVariant { variant, .. }
                    if variant == "Nope"),
                "unexpected error: {error}"
            );
        });
    }

    #[test]
    fn enum_variant_body_shape_mismatches_error() {
        let mut document = BsnDocument::new();
        let one = document.push_value(BsnValue::Int(1));
        // `Choice::Bar` is a struct variant, written with positional fields.
        let tuple_on_struct = document.push_value(BsnValue::NamedTuple(
            BsnPath::from_segments(["Choice", "Bar"]),
            vec![one],
        ));
        // `Choice::Baz` is a tuple variant, written with named fields.
        let named_on_tuple = document.push_value(BsnValue::Struct(
            BsnPath::from_segments(["Choice", "Baz"]),
            vec![("x".to_string(), one)],
        ));
        // More positional values than `Choice::Baz` has fields.
        let too_many = document.push_value(BsnValue::NamedTuple(
            BsnPath::from_segments(["Choice", "Baz"]),
            vec![one, one],
        ));
        // A body on the unit variant `Choice::Qux`.
        let body_on_unit = document.push_value(BsnValue::NamedTuple(
            BsnPath::from_segments(["Choice", "Qux"]),
            vec![one],
        ));

        with_cx(&document, |cx| {
            let error = build_value(cx, tuple_on_struct, reg::<Choice>(cx)).unwrap_err();
            assert!(
                matches!(&error, DynamicSceneBuildError::TypeNotTupleStruct { type_path, .. }
                    if type_path.ends_with("Choice::Bar")),
                "unexpected error: {error}"
            );

            let error = build_value(cx, named_on_tuple, reg::<Choice>(cx)).unwrap_err();
            assert!(
                matches!(&error, DynamicSceneBuildError::TypeNotStruct { type_path, .. }
                    if type_path.ends_with("Choice::Baz")),
                "unexpected error: {error}"
            );

            assert!(matches!(
                build_value(cx, too_many, reg::<Choice>(cx)),
                Err(DynamicSceneBuildError::TooManyTupleFields {
                    given: 2,
                    expected: 1,
                    ..
                })
            ));

            assert!(matches!(
                build_value(cx, body_on_unit, reg::<Choice>(cx)),
                Err(DynamicSceneBuildError::TooManyTupleFields { expected: 0, .. })
            ));
        });
    }

    #[test]
    fn enum_variant_field_name_errors() {
        let mut document = BsnDocument::new();
        let one = document.push_value(BsnValue::Int(1));
        let two = document.push_value(BsnValue::Int(2));
        let unknown = document.push_value(BsnValue::Struct(
            BsnPath::from_segments(["Choice", "Bar"]),
            vec![("nope".to_string(), one)],
        ));
        let duplicate = document.push_value(BsnValue::Struct(
            BsnPath::from_segments(["Choice", "Bar"]),
            vec![("x".to_string(), one), ("x".to_string(), two)],
        ));

        with_cx(&document, |cx| {
            let error = build_value(cx, unknown, reg::<Choice>(cx)).unwrap_err();
            assert!(
                matches!(&error, DynamicSceneBuildError::UnknownField { type_path, field, .. }
                    if type_path.ends_with("Choice::Bar") && field == "nope"),
                "unexpected error: {error}"
            );

            let error = build_value(cx, duplicate, reg::<Choice>(cx)).unwrap_err();
            assert!(
                matches!(&error, DynamicSceneBuildError::DuplicateField { type_path, field, .. }
                    if type_path.ends_with("Choice::Bar") && field == "x"),
                "unexpected error: {error}"
            );
        });
    }

    #[test]
    fn enum_struct_variant_fills_defaults() {
        let mut document = BsnDocument::new();
        let one = document.push_value(BsnValue::Int(1));
        let id = document.push_value(BsnValue::Struct(
            BsnPath::from_segments(["Choice", "Bar"]),
            vec![("x".to_string(), one)],
        ));
        with_cx(&document, |cx| {
            let (full, partial) = build_enum_forms(
                cx,
                reg::<Choice>(cx),
                "Bar",
                EnumInput::Named(&[("x".to_string(), one)]),
                Span::NONE,
            )
            .unwrap();

            let ReflectRef::Enum(full_enum) = full.reflect_ref() else {
                panic!("expected an enum");
            };
            assert_eq!(full_enum.variant_name(), "Bar");
            assert_eq!(full_enum.field_len(), 3);

            let ReflectRef::Enum(partial_enum) = partial.reflect_ref() else {
                panic!("expected an enum");
            };
            assert_eq!(partial_enum.field_len(), 1);

            // The value form is the full one, so a variant switch always succeeds.
            let value = build_value(cx, id, reg::<Choice>(cx)).unwrap();
            let mut target = Choice::Qux;
            target.try_apply(&*value).unwrap();
            assert_eq!(target, Choice::Bar { x: 1, y: 0, z: 0 });
        });
    }

    #[test]
    fn enum_tuple_variant_fills_defaults() {
        let mut document = BsnDocument::new();
        let ten = document.push_value(BsnValue::Int(10));
        let id = document.push_value(BsnValue::NamedTuple(
            BsnPath::from_segments(["Choice", "Baz"]),
            vec![ten],
        ));
        with_cx(&document, |cx| {
            let value = build_value(cx, id, reg::<Choice>(cx)).unwrap();
            let mut target = Choice::Qux;
            target.try_apply(&*value).unwrap();
            assert_eq!(target, Choice::Baz(10));
        });
    }

    #[test]
    fn enum_variant_without_a_body_fills_defaults() {
        // `Choice::Bar` and `Choice::Baz` name a variant with no body at all, which selects the
        // variant with every field defaulted.
        let mut document = BsnDocument::new();
        let struct_variant =
            document.push_value(BsnValue::Path(BsnPath::from_segments(["Choice", "Bar"])));
        let tuple_variant =
            document.push_value(BsnValue::Path(BsnPath::from_segments(["Choice", "Baz"])));

        with_cx(&document, |cx| {
            let value = build_value(cx, struct_variant, reg::<Choice>(cx)).unwrap();
            let mut target = Choice::Qux;
            target.try_apply(&*value).unwrap();
            assert_eq!(target, Choice::Bar { x: 0, y: 0, z: 0 });

            let value = build_value(cx, tuple_variant, reg::<Choice>(cx)).unwrap();
            let mut target = Choice::Qux;
            target.try_apply(&*value).unwrap();
            assert_eq!(target, Choice::Baz(0));
        });
    }

    #[test]
    fn enum_variant_field_missing_reflect_default_errors() {
        let mut document = BsnDocument::new();
        let one = document.push_value(BsnValue::Int(1));
        let inner = document.push_value(BsnValue::NamedTuple(
            BsnPath::from_segments(["NoDefault"]),
            vec![one],
        ));
        let id = document.push_value(BsnValue::Struct(
            BsnPath::from_segments(["Tricky", "A"]),
            vec![("field".to_string(), inner)],
        ));
        with_cx(&document, |cx| {
            let error = build_value(cx, id, reg::<Tricky>(cx)).unwrap_err();
            assert!(
                matches!(
                    &error,
                    DynamicSceneBuildError::MissingReflectDefault { type_path, .. }
                        if type_path.ends_with("NoDefault")
                ),
                "unexpected error: {error}"
            );
            // The diagnostic has to tell the user how to fix it.
            assert!(
                error.to_string().contains("#[reflect(Default)]"),
                "unexpected message: {error}"
            );
        });
    }

    #[test]
    fn option_field_implicit_some() {
        let mut document = BsnDocument::new();
        let five = document.push_value(BsnValue::Int(5));
        let none = document.push_value(BsnValue::Path(BsnPath::from_segments(["None"])));
        with_cx(&document, |cx| {
            let value = build_value(cx, five, reg::<Option<u32>>(cx)).unwrap();
            assert_eq!(applied::<Collections>(&wrap_maybe(value)).maybe, Some(5));

            let value = build_value(cx, none, reg::<Option<u32>>(cx)).unwrap();
            let mut target = Some(7u32);
            target.try_apply(&*value).unwrap();
            assert_eq!(target, None);
        });
    }

    /// An `Option` lookalike whose `None` carries a field.
    #[derive(Reflect, Clone, PartialEq, Debug)]
    enum NoneIsNotUnit {
        None(u32),
        Some(u32),
    }

    /// An `Option` lookalike whose `Some` is a struct variant.
    #[derive(Reflect, Clone, PartialEq, Debug)]
    enum SomeIsNotATuple {
        None,
        Some { value: u32 },
    }

    /// An `Option` lookalike whose `Some` carries two fields.
    #[derive(Reflect, Clone, PartialEq, Debug)]
    enum SomeTakesTwo {
        None,
        Some(u32, u32),
    }

    #[test]
    fn implicit_some_only_applies_to_option_shaped_enums() {
        let (document, id) = value_doc(BsnValue::Int(5));
        let mut registry = test_registry();
        registry.register::<NoneIsNotUnit>();
        registry.register::<SomeIsNotATuple>();
        registry.register::<SomeTakesTwo>();
        let mut cx = BuildCx::new(&registry, &document, "test.bsn");

        macro_rules! check {
            ($($ty:ty),*) => {$({
                let expected = reg::<$ty>(&cx);
                let error = build_value(&mut cx, id, expected).unwrap_err();
                assert!(
                    matches!(&error, DynamicSceneBuildError::ValueTypeMismatch { .. }),
                    "{}: unexpected error: {error}",
                    core::any::type_name::<$ty>()
                );
            })*};
        }
        // Three variants; `None` with a field; `Some` as a struct variant; `Some` with two fields.
        check!(Choice, NoneIsNotUnit, SomeIsNotATuple, SomeTakesTwo);
    }

    #[test]
    fn option_field_explicit_some() {
        let mut document = BsnDocument::new();
        let five = document.push_value(BsnValue::Int(5));
        let some = document.push_value(BsnValue::NamedTuple(
            BsnPath::from_segments(["Some"]),
            vec![five],
        ));
        with_cx(&document, |cx| {
            let value = build_value(cx, some, reg::<Option<u32>>(cx)).unwrap();
            let mut target: Option<u32> = None;
            target.try_apply(&*value).unwrap();
            assert_eq!(target, Some(5));
        });
    }

    /// Wraps a built `Option<u32>` value in a `Collections`-shaped partial struct.
    fn wrap_maybe(value: Box<dyn PartialReflect>) -> DynamicStruct {
        let mut dynamic = DynamicStruct::default();
        dynamic.insert_boxed("maybe", value);
        dynamic
    }

    #[test]
    fn list_value() {
        let mut document = BsnDocument::new();
        let one = document.push_value(BsnValue::Int(1));
        let two = document.push_value(BsnValue::Int(2));
        let id = document.push_value(BsnValue::List(vec![one, two]));
        with_cx(&document, |cx| {
            let value = build_value(cx, id, reg::<Vec<u8>>(cx)).unwrap();
            let mut target: Vec<u8> = Vec::new();
            target.try_apply(&*value).unwrap();
            assert_eq!(target, vec![1, 2]);
        });
    }

    #[test]
    fn tuple_value_into_a_tuple_destination() {
        let mut document = BsnDocument::new();
        let one = document.push_value(BsnValue::Int(1));
        let pair = document.push_value(BsnValue::Tuple(vec![one, one]));
        let triple = document.push_value(BsnValue::Tuple(vec![one, one, one]));

        // `(u32, f32)` is only registered as a field type of a fixture component, so the bare
        // fixture registry needs it added.
        let mut registry = test_registry();
        registry.register::<(u32, f32)>();
        let mut cx = BuildCx::new(&registry, &document, "test.bsn");
        let expected = reg::<(u32, f32)>(&cx);

        let value = build_value(&mut cx, pair, expected).unwrap();
        let mut target = (0u32, 0f32);
        target.try_apply(&*value).unwrap();
        assert_eq!(target, (1, 1.0));

        assert!(matches!(
            build_value(&mut cx, triple, expected),
            Err(DynamicSceneBuildError::TooManyTupleFields {
                given: 3,
                expected: 2,
                ..
            })
        ));

        // A tuple written where a scalar was expected.
        let scalar = reg::<u32>(&cx);
        let error = build_value(&mut cx, pair, scalar).unwrap_err();
        assert!(
            matches!(&error, DynamicSceneBuildError::ValueTypeMismatch { found, .. }
                if found == "a tuple"),
            "unexpected error: {error}"
        );
    }

    /// A single-field tuple struct wrapping a list — the shape a `#[template(built_in)]` `Vec`
    /// field's `VecTemplate<T>` has.
    #[derive(Reflect, Clone, Default, PartialEq, Debug)]
    #[reflect(Default)]
    struct ListWrapper(Vec<u8>);

    #[test]
    fn list_into_a_list_wrapping_tuple_struct() {
        let mut document = BsnDocument::new();
        let one = document.push_value(BsnValue::Int(1));
        let two = document.push_value(BsnValue::Int(2));
        let id = document.push_value(BsnValue::List(vec![one, two]));

        let mut registry = test_registry();
        registry.register::<ListWrapper>();
        let mut cx = BuildCx::new(&registry, &document, "test.bsn");
        let expected = reg::<ListWrapper>(&cx);

        let value = build_value(&mut cx, id, expected).unwrap();
        assert_eq!(applied::<ListWrapper>(&*value), ListWrapper(vec![1, 2]));
    }

    #[test]
    fn list_item_type_mismatch_errors() {
        let mut document = BsnDocument::new();
        let text = document.push_value(BsnValue::String("a".to_string()));
        let id = document.push_value(BsnValue::List(vec![text]));
        with_cx(&document, |cx| {
            assert!(matches!(
                build_value(cx, id, reg::<Vec<u8>>(cx)),
                Err(DynamicSceneBuildError::ValueTypeMismatch { .. })
            ));
        });
    }

    #[test]
    fn entity_ref_value() {
        let mut document = BsnDocument::new();
        let root = document.push_node(BsnNodeKind::Entity {
            name: Some("Root".to_string()),
            name_span: None,
            base: None,
            base_span: None,
            patches: Vec::new(),
            relations: Vec::new(),
        });
        document.push_root(root);
        let id = document.push_value(BsnValue::EntityRef("Root".to_string()));

        with_cx(&document, |cx| {
            let value = build_value(cx, id, reg::<EntityTemplate>(cx)).unwrap();
            let template = value.try_downcast_ref::<EntityTemplate>().unwrap();
            let expected = SceneEntityReference::from_asset("test.bsn", root.0);
            assert!(
                matches!(template, EntityTemplate::SceneEntityReference(reference) if *reference == expected)
            );
        });
    }

    #[test]
    fn entity_ref_into_a_foreign_type_errors() {
        let mut document = BsnDocument::new();
        let root = document.push_node(BsnNodeKind::Entity {
            name: Some("Root".to_string()),
            name_span: None,
            base: None,
            base_span: None,
            patches: Vec::new(),
            relations: Vec::new(),
        });
        document.push_root(root);
        let id = document.push_value(BsnValue::EntityRef("Root".to_string()));

        with_cx(&document, |cx| {
            // A destination that is neither an `EntityTemplate` nor convertible from one.
            let error = build_value(cx, id, reg::<u32>(cx)).unwrap_err();
            assert!(
                matches!(&error, DynamicSceneBuildError::ValueTypeMismatch { expected, .. }
                    if expected == "u32"),
                "unexpected error: {error}"
            );
        });
    }

    #[test]
    fn entity_ref_to_unknown_name_errors() {
        let (document, id) = value_doc(BsnValue::EntityRef("Nope".to_string()));
        with_cx(&document, |cx| {
            assert!(matches!(
                build_value(cx, id, reg::<EntityTemplate>(cx)),
                Err(DynamicSceneBuildError::UnknownEntityName { .. })
            ));
        });
    }

    #[test]
    fn nested_field_values_use_registered_conversions() {
        let mut document = BsnDocument::new();
        let large = document.push_value(BsnValue::Path(BsnPath::from_segments([
            "TextSize", "Large",
        ])));
        let id = document.push_value(BsnValue::Struct(
            BsnPath::from_segments(["TextFont"]),
            vec![("font_size".to_string(), large)],
        ));
        with_cx(&document, |cx| {
            let value = build_value(cx, id, reg::<TextFont>(cx)).unwrap();
            assert_eq!(
                applied::<TextFont>(&*value),
                TextFont {
                    font_size: FontSize(24)
                }
            );
        });
    }

    #[test]
    fn dangling_value_ids_error_rather_than_panic() {
        let document = BsnDocument::new();
        with_cx(&document, |cx| {
            assert!(matches!(
                build_value(cx, BsnValueId(7), reg::<u32>(cx)),
                Err(DynamicSceneBuildError::MalformedDocument { .. })
            ));
        });
    }

    #[test]
    fn cyclic_values_error_rather_than_overflow() {
        // A hand-built cyclic document: value 0 is a list containing itself. Whether the walk is
        // stopped by the depth guard or by a type mismatch, it must stop with an `Err`.
        let mut document = BsnDocument::new();
        let id = document.push_value(BsnValue::List(Vec::new()));
        document.values[0].value = BsnValue::List(vec![id]);
        with_cx(&document, |cx| {
            assert!(build_value(cx, id, reg::<Vec<Vec<u8>>>(cx)).is_err());
        });
    }
}
