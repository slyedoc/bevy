# GPU transform propagation

Moving Bevy's transform propagation (and instance movement) off the CPU and onto
the GPU, so the pure-RT Solari path can scale to **~2M objects**. This doc is both
the design rationale and the as-built record; the original plan (Jacobi +
async readback) is preserved in the "Plan vs. as-built" section because the
shipped design diverged from it.

## Result

`bevy_city --size 100` = **2,080,362 entities with `GlobalTransform`** (RTX 5090).
Before this work the scene ran at **~20 fps** (CPU-bound, `main app` ≈ 18.9 ms,
dominated by `propagate_parent_transforms` + the visibility hierarchy + the
per-instance cluster-BLAS rebuild). After: **~180 fps**.

Main-thread frame (`update`) arc, mean/frame: **24 ms → ~7.1 ms**. GPU
**3.87 → 1.17 ms**. Cars verified moving correctly; static scene stable.

The wins, in order they landed this session:

| change | effect |
|---|---|
| P0 `set_if_neq` on the dirty root | propagate 15.7 → 8.1 ms (CPU, pre-port) |
| GPU transform table + ancestor-walk propagation | transform hierarchy → GPU |
| strip `Visibility`/`InheritedVisibility` from RT meshes + ancestors | visibility tax 3.4 → 0.04 ms |
| GPU move detection (PTLAS `fill_incremental`) | dropped `Changed<Transform>` from the instance extract; killed per-car pack/flush churn |
| changed-only ancestor-walk (was 16-pass Jacobi over all 2M) | `transform_propagate` 2.68 → 0.0095 ms GPU |

## Why the CPU couldn't do it

240 fps = **4.16 ms for the entire frame**. The transform hierarchy alone
(`propagate_parent_transforms` + `mark_dirty_trees`) was ~9.9 ms — 2.4× the whole
budget — and no CPU rearrangement fits 2.08M affine propagations into it. The GPU
is nearly idle (pathtracer ~0.69 ms): 2.08M affine multiplies × ~6 hierarchy
levels is single-digit µs on the 5090. The cost was never the math — it's getting
data to the GPU and **not reading it back per frame**. That framing drove the
whole design.

### Three structural taxes, and they rhyme

1. **Transform hierarchy ~9.9 ms** — this doc. → ~0 on GPU. **DONE.**
2. **Visibility hierarchy ~5.2 ms** (`visibility_propagate` + `check_visibility`).
   The *same graph* walked again. **Largely addressed** by stripping the
   visibility components off RT meshes + their mesh-less ancestors (see
   `helper.rs`) rather than a GPU visibility pass — pure-RT meshes are never
   frustum-culled (BVH does it) and Solari extracts them regardless of
   `ViewVisibility`. The remaining design notes for a *proper* RT visibility
   model live in `src/camera/mod.rs`.
3. **PBR raster vestiges ~1.5 ms** (`extract_mesh_materials`, etc.). Dead weight
   in pure-RT; the `PbrPlugin` decoupling work is orthogonal and ongoing.

## P0 (DONE, upstreamable) — `set_if_neq` on the dirty root

`crates/bevy_transform/src/systems.rs` (parallel `propagate_parent_transforms`,
root seed): a root is visited whenever *any* descendant moved (the dirty bit
climbs to the root). The old direct write `*parent_transform =
GlobalTransform::from(*transform)` marked the root changed even when it never
moved, defeating the child prune test (`!p_global_transform.is_changed()`) and
force-visiting **every** direct child. `set_if_neq` marks changed only on a real
move, so static direct-children prune.

size 100: propagate 15.66 → 8.09 ms. All `bevy_transform` tests pass. **Standalone
upstream bevy bug — worth a PR to `main` independently of this work.** It caps out
(residual scales with movers × root-breadth), so it sized the prize, it isn't the
fix.

## Architecture (as built)

A **GPU transform table** keyed by its own dense node-slot space covering **every
entity with `GlobalTransform`**, built on the existing `GpuColumn` scatter infra,
declared via the `gpu_table!` macro. Two tables:

- **Transform table** (`src/transform/`, `gpu_table! TransformGraph`): every
  `GlobalTransform` entity. GPU columns `local` (`Affine3x4` from `Transform`) +
  `parent` (node-slot, `ROOT_PARENT` sentinel for roots). Own component-indexed
  slot allocator (`GpuSlot<TransformGraph>` — a free array-indexed component read,
  no `EntityHashMap`). Propagation pass → persistent `world` buffer.
- **Instance table** (`instance/`): RT meshes. Carries a `NodeSlotColumn`
  (foreign key into the transform table). Its `TransformColumn` is filled by the
  gather, not the CPU: `transform_column[i] = world[node_slot[i]]`.

### `SolariTransformPlugin` — own the systems, reuse the components

Solari carries **no** new per-entity data, so it reuses Bevy's
`Transform`/`GlobalTransform`/`ChildOf`/`Children` verbatim and replaces only the
propagation *systems*. `bevy_city` does `DefaultPlugins.disable::<TransformPlugin>()`
under the `solari` feature, keeping the crate + components so Avian/UI/gizmos/
picking still read `GlobalTransform`, with a CPU carve-out in `PostUpdate`
(`sync_simple_transforms` + `propagate_transforms_for::<With<Node>>`) for the
small set (camera, sun, UI) that needs same-frame CPU `GlobalTransform`. The 2.08M
render bulk gets nothing on the CPU.

### Propagation pass — single-pass ancestor-walk over the changed set

**This is the key divergence from the original plan** (which was a fixed
16-pass Jacobi ping-pong over all nodes). `world[node]` is a pure function of the
node's own `local`/`parent` ancestor chain — independent of any other node's
world. So one thread walks a node's parent chain composing locals and writes its
world in a **single pass**: no iteration, no ping-pong, no read/write hazard
(reads the read-only `local`/`parent` columns, writes only its own `world` slot).

The buffer is **persistent**, and the dispatch runs **one thread per changed
node** — the `LocalColumn` delta (`GpuColumn::delta_buffer()`/`pending()`/
`record_stride()`; slot at `changed[k*stride]`). Static nodes keep last frame's
world. On a capacity growth, `full_rebuild` walks every node once to repopulate
the new buffer; steady state is changed-only.

```wgsl
// one thread per changed node (full_rebuild: per node, id = slot)
var m = local[node];
var p = parent[node];
for (...; p != ROOT_PARENT; p = parent[p]) { m = local[p] ∘ m; }  // ≤ MAX_DEPTH
world[node] = m;
```

- Result: `transform_propagate` 2.68 ms (Jacobi) → **0.0095 ms** (changed-only) —
  only the moving cars get walked instead of 2M × 16 passes.
- **Boundary:** updates a *changed* node's own world. Animating a node's
  DESCENDANTS without changing their own `local` (animating a parent, or a bare
  re-parent) won't re-walk those descendants. **`bevy_city` is exact** — cars are
  single-mesh LEAVES (`merge_car_meshes` merges body+tires into one mesh on
  purpose; roots/roads are static, walked once at first-sight). A general
  hierarchy needs descendant dirtying (a 1-bit downward propagation seeded from
  the delta) — deferred, see Open.

### Output — the gather drives the RT scene

`gather` (`transform_gather.wgsl`) copies each instance's world into the instance
`TransformColumn` every RT consumer already reads (PTLAS fill, blas sharing, the
raytracing scene), indexed by `node_slot[i]`, and shifts current → previous
(motion vectors / ReSTIR temporal **and** the PTLAS-fill move detection below).
There is no CPU instance-transform path.

### Instance movement — GPU move detection (no CPU `Changed<Transform>`)

The original plan kept a CPU `Changed<Transform>` scan to tell the PTLAS which
instances moved. **Removed.** `ptlas_fill.wgsl::fill_incremental` (one thread per
active instance) detects moves by comparing this frame's gathered world
(`cluster_instance_transforms[slot]`) against last frame's (the gather's
`previous` buffer, bound at fill group(1) binding(10)); differ ⇒ append a WRITE
record. This:

- dropped `Changed<Transform>`/`Changed<GlobalTransform>` from the instance
  extract filter (`ExtractChangeFilter` is now `Added ∪ Changed<Material> ∪
  Changed<RenderLayers>`), eliminating the per-car `RtSlotMap` lookups + `flush`
  churn (`pack` 1.87 → 0.84, `flush` 1.07 → ~0 ms);
- **closes the parent-driven-move gap** the CPU path had (it compares actual
  world transforms);
- `InstanceManager::moved_slots` → renamed `rewrite_slots`: now ONLY cull-mask
  (`RenderLayers`) changes, which `fill_incremental` can't see (the mask is baked
  into the TLAS record). Material changes need no re-write (stable `instance_id`,
  resolved at trace time). CPU PTLAS seed = added ∪ rewrite(mask) ∪ disabled;
  moves come from the GPU.

## Plan vs. as-built

What the original plan got right, and where reality differed:

| planned | as-built |
|---|---|
| Jacobi ping-pong, `max_depth` passes, brute-force all nodes every frame | single-pass ancestor-walk, **changed nodes only**, persistent buffer |
| moved-set via CPU `Changed<Transform>` → PTLAS | GPU current-vs-previous compare in `fill_incremental` (also fixes parent-move gap) |
| async readback + `node→Entity` map to reconstruct `GlobalTransform` (P3) | **not built yet, but required** — `bevy_city` has no physics so nothing reads the bulk's `GlobalTransform` on the CPU; **Avian will need it** (see below). Deferred, not abandoned. |
| `NoGpuGlobalTransformReadback` opt-out marker (P4) | deferred with P3 — the carve-out + component-stripping cover the no-physics case, but the opt-out is the natural partner to the readback (mark what *doesn't* need CPU writeback) |
| visibility as a second GPU pass over `parent[]` | **stripped** visibility components off RT meshes + ancestors instead (simpler, exact for pure-RT) |

## Rollout status

- **P0** ✅ `set_if_neq` (working tree; PR upstream separately).
- **P1–P2** ✅ `src/transform/` table + propagation, gather wired into the RT
  `TransformColumn` — the world buffer drives the rendered scene. Verified: cars
  move, scene stable.
- **Move detection** ✅ GPU-side in the PTLAS fill; CPU instance extract no longer
  scans transforms.
- **Changed-only propagation** ✅ ancestor-walk over the `LocalColumn` delta.
- **P3 readback** ✅ **first cut built + validated.** Default-on, change-driven
  GPU→CPU `GlobalTransform` writeback with `NoGpuGlobalTransformReadback` opt-out — see the
  next section. Validated in bevy_city (a tagged car's CPU `GlobalTransform`
  tracks its GPU-driven motion, ancestor composition included). Cost follow-ups
  below remain (whole-buffer readback, CPU-side opt-out, scale).
- **P4 opt-out** ✅ `NoGpuGlobalTransformReadback` marker (parallels `NoCpuCulling`),
  enforced **GPU-side**: a `no_readback` flag column on the transform table
  (scattered alongside `local`) makes the readback gather skip opted-out nodes —
  they never enter the buffer, so no per-mover CPU writeback. The transfer is
  **count-scoped**: the observer drives bevy's `Readback` range from the lagged
  record count (+headroom), so a mostly-opted-out scene transfers ~KB not the
  full capacity. Opt-out + count-scope → readback transfer ∝ non-opted movers
  (≈0 when everything's opted out).

## Readback — required for Avian (P3, not yet built)

Once the GPU is the source of truth for movement, any CPU system that reads a
GPU-moved entity's `GlobalTransform` sees a stale value unless we write it back.
`bevy_city` dodges this (no physics, and the camera/UI/sun carve-out propagates on
the CPU). **Avian does not** — so this is the gating work before Solari + Avian
compose.

### The Avian wrinkle — who owns movement matters

Avian doesn't read `GlobalTransform` for its core solve: `Position`/`Rotation` are
**world-space**, and the physics integrator owns them. The dependency is at the
boundaries, and it splits by who moves the entity:

- **Avian moves it (dynamic bodies):** the CPU is already the source of truth.
  Avian writes `Position`/`Rotation` → synced to `Transform` → scattered into the
  `local` column like any other mover. These need **no readback** — the GPU is
  downstream of them, not upstream. They're effectively `NoGpuGlobalTransformReadback`'s
  opposite: CPU-authored, GPU-consumed.
- **The GPU moves it, Avian reads it:** a collider/body parented under a
  GPU-propagated hierarchy (e.g. a static-mesh platform animated on the GPU with a
  physics body riding it), nested `ColliderParent` chains, or a raycast/query
  against a GPU-moved entity's world pose. **These need the readback** — the
  body's `GlobalTransform` (or its parent's world) is computed on the GPU and the
  CPU must see it.

So the readback set is narrower than "all 2M": it's the entities the **GPU moves
that the CPU also reads** — typically small. The opt-out marker (P4) is the
inverse tag: CPU-authored entities (Avian dynamic bodies, the carve-out) skip
writeback entirely.

### Design (carried from the original plan, still the approach)

Change-driven **async readback**, O(moved), no stall:

1. After propagation, a compaction pass diffs `world != world_prev` (the buffer
   the gather already maintains) and appends `(node_slot, world)` for changed
   nodes — the same delta-append pattern as `fill_incremental`.
2. A `node_slot → Entity` reverse map (a parallel column, or reuse the
   `GpuSlot<TransformGraph>` inverse) turns slots into entities.
3. Triple-buffered staging buffer, mapped 1–3 frames later; a CPU system writes
   `GlobalTransform` for those entries.

**Coherence caveat to validate against real Avian:** a CPU system that writes
`Transform` and reads `GlobalTransform` the *same* frame sees a 1–3-frame-stale
global under async readback. For the GPU-moved-Avian-reads case this is usually
fine (physics maintains its own world state and reads `GlobalTransform` at
collider init / nested-body setup, not per-step), but a same-frame
write-then-read body is the failure mode. The escape hatch is the existing
synchronous `propagate_transforms_for::<F>` (`systems.rs`) — per-entity ancestor
walk, same-frame, zero latency — reserved for the small latency-critical set.

## Residual cost & open levers

`update` ≈ 7.1 ms is now led by:

- `extract_transform_graph` ~1.9 ms — the transform table's own extract. Measured
  split (size 100): **scan 0.58 + build 0.45 + merge 0.85 ms**. The merge (serial
  drain of the parallel thread-locals into the column Vec) is the biggest piece
  and is **memory-bandwidth-bound** (a parallel-scatter rewrite did not beat it).
  The only real win is to *relocate* it off the main thread (chunked
  `write_buffer` per thread-local in the render-schedule prepare) — deferred,
  needs `gpu_table!`/`GpuColumnDesc` machinery.
- the render-thread recv-wait (~1.9 ms) — main thread blocks on the render thread
  overrunning the main schedule; reduce render-thread time and it shrinks.
- `main app` schedule ~3.0 ms.

## Follow-on — skeletal animation on the same table

Skeletal animation *is* transform-hierarchy propagation: bone/joint entities are
just `Transform` + `ChildOf` nodes, **already in the transform table** — no
special hierarchy handling. `AnimationPlayer` writes animated joint *local*
`Transform`s (CPU, cheap, change-driven) → `local` column → the same ancestor-walk
produces joint worlds. Skinning becomes a **consumer of the table**, like RT
instances: `skin_matrix[j] = world[joint_node_slot[j]] * inverse_bind[j]`. The
animated part (joint poses) lives in the table; only the static part (inverse-bind
+ `joint_index → node_slot`) rides with the skinned mesh. `SkinnedMesh.joints`
references joints by `Entity` (often siblings, not ancestors) — captured for free
because the table covers *all* `GlobalTransform` entities.

**Why it matters at scale:** animation is the "many movers" regime — a crowd of
animated characters is thousands of joints changing every frame, where CPU
propagation collapses and the changed-only GPU walk doesn't care. The descendant-
dirtying boundary above must be solved first (animated joints have child joints).

## Open / to decide

- **Readback for Avian** (P3) — required before Solari + Avian compose; design
  above. The next major piece of work.
- **Descendant dirtying** — the changed-only walk's one assumption (movers are
  leaves). A 1-bit downward dirty-propagation seeded from the `LocalColumn` /
  `ParentColumn` delta generalizes it (and is the prerequisite for skeletal
  animation). Cheap (flags, not mat3x4).
- **Relocate the extract merge** off the main thread (chunked upload).
- Whether to upstream the transform table into `bevy_transform` (it's a general
  GPU transform backend; Solari is just the first consumer).
- Proper RT visibility model (user-`Hidden` → PTLAS cull mask, event-driven) —
  see `src/camera/mod.rs`.
