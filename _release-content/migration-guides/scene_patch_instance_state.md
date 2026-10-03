---
title: "`ScenePatchInstance` now requires `SceneInstanceState`"
pull_requests: []
---

`ScenePatchInstance` gained `#[require(SceneInstanceState)]`, so entities carrying it also carry a
`SceneInstanceState` component recording whether the scene has been applied and which entities the
most recent application spawned. This is what lets hot reload clean up the previous generation of
scene entities instead of orphaning them. It is added in the same archetype move as
`ScenePatchInstance` itself, so spawning an instance costs no extra archetype move.

`Commands::queue_spawn_scene`, `World::queue_spawn_scene` and `EntityWorldMut::queue_apply_scene`
now insert `ScenePatchInstance` on the target entity rather than registering the queued spawn
out-of-band. Single-scene spawning behavior is unchanged, and these entities become
hot-reload-tracked as a result, but two things follow:

- Code that asserted on an exact archetype or component count for these entities must account for
  the two additional components.
- An entity is an instance of exactly *one* scene. Calling `queue_apply_scene` again on the same
  entity (or replacing its `ScenePatchInstance`) now **replaces** the instance: the previous
  application's spawned descendants are despawned before the new scene applies, exactly as on a
  hot reload. Components the earlier scene added to the entity itself are not removed. Apply the
  scenes to different entities, or compose them into one scene, if you relied on the old additive
  behavior.

`resolve_scene_patches` now writes through `Assets::get_mut_untracked`, so resolving a scene no
longer emits `AssetEvent::Modified` for its `ScenePatch`. `Modified` on a `ScenePatch` now means the
asset actually changed. If you were reading `AssetEvent::<ScenePatch>::Modified` to detect that a
scene had been resolved, read `AssetEvent::LoadedWithDependencies` instead, or check
`ScenePatch::resolved`.

`ResolvedSceneRoot::apply` is unchanged in both signature and behavior. If you need to know which
entities an application created, use the new `ResolvedSceneRoot::apply_recording`.
