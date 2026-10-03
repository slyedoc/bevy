---
title: "Hot reloading `.bsn` scenes"
authors: []
pull_requests: []
---

Entities spawned from a `.bsn` file via `ScenePatchInstance` now update live when the file changes
on disk. Enable asset watching (the `file_watcher` feature, or
`AssetPlugin { watch_for_changes_override: Some(true), .. }`), edit a `.bsn` file, and every live
instance is rebuilt from the new definition — including instances of other `.bsn` files that
inherit it with `:"base.bsn"`, and in-code `bsn!` scenes that inherit it. Calling
`AssetServer::reload` directly does the same thing without a watcher.

This needed no new asset API: the asset server already replaces the whole `ScenePatch` value on
reload and re-fires `AssetEvent::LoadedWithDependencies`, so the system that resolves a scene on
first load is the same one that re-resolves it on every edit. What is new is the *re-application*
half, and the bookkeeping that makes it correct.

Reloading is a rebuild, not a reconciliation: the scene's descendants are despawned and respawned,
so runtime state on scene-spawned entities — and `Entity` ids held elsewhere — does not survive a
reload. `Ready` fires per application, so each reload emits a fresh bottom-up `Ready` pass over
the regenerated hierarchy — observers doing per-spawn setup run again, which is usually exactly
what a rebuilt scene wants. Despawning first is what stops the previous generation of children from being left behind
as parentless ghosts. State-preserving reconciliation is planned as follow-up work. Components that
a `.bsn` file *stops* declaring are also not removed from the live root entity; respawn the
instance to pick that up.

A file that stops parsing leaves every live instance rendering the last good version and logs the
error rather than panicking, so a typo mid-edit does not blank your UI.

The set of entities a scene created is now visible on each instance as
`SceneInstanceState::spawned`, and `ResolvedSceneRoot::apply_recording` exposes the same
information to code that applies scenes itself.

One boundary is worth knowing about: an in-code `bsn!` scene that both includes `:"base.bsn"` and
patches a component the base also patches keeps that component's base values as of the first
resolve. Its `Scene` was consumed by value when it resolved and there is no file to re-read.
Everything else about such a scene — children, other components, entity references — reloads
correctly; move the overlapping patch into a `.bsn` file for full hot reload.
