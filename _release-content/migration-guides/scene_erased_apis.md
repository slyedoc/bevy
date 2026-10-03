---
title: "Type-erased scene APIs"
pull_requests: []
---

`ResolveContext` gained a `type_registry: Option<&TypeRegistry>` field, and
`ResolvedSceneRoot::resolve`, `ResolvedSceneListRoot::resolve`, `ScenePatch::resolve` and
`SceneListPatch::resolve` gained a matching trailing parameter. Pass
`world.get_resource::<AppTypeRegistry>().map(|r| r.read())` (keeping the guard alive across the
call, and never write-locking the registry during resolution) if you have a `World`, or `None` if
you do not — reflection-driven scenes such as those loaded from `.bsn` files require it,
statically-defined scenes ignore it.

`ResolvedScene::get_or_insert_erased_template`'s `default` parameter changed from
`fn() -> Box<dyn ErasedComponentTemplate>` to
`impl FnOnce() -> Box<dyn ErasedComponentTemplate>`. Function items, function pointers and
non-capturing closures continue to work unchanged; capturing closures are now accepted too. The
only source-incompatible case is an explicit lifetime turbofish
(`get_or_insert_erased_template::<'a>(..)`), which is no longer permitted.

`ResolveSceneError` gained the variants `TypeNotRegistered`, `TypeNotReflectable`,
`MissingReflectDefault`, `MissingReflectComponent`, `UnsupportedRelationship`, `ApplyFailed`
and `UnpatchableTemplate`. Exhaustive `match`es over it need new arms (or a `_` arm).

`ErasedComponentTemplate` gained three provided methods — `try_as_partial_reflect`,
`try_as_partial_reflect_mut` and `template_type_id` — all with defaults. Existing implementations
need no changes, **unless** your implementation stores its template value type-erased and is
filed under a `TypeId` other than its own concrete type: it must then override
`template_type_id` to return that `TypeId`, or the cached copy-on-write duplicate check will
write its component twice.

`ResolvedScene::get_or_insert_template` no longer panics when the template slot holds a value of
a different concrete type. It now converts that value back to `T` via reflection (preserving
field values) or, failing that, resets the slot to `T::default()` and logs an error.
