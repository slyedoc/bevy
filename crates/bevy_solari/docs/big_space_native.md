# big_space, native in bevy_solari — design & Vulkan feasibility

> Status: design + feasibility, **net-new** (the working tree has no floating-origin state — verified clean). All ash/Vulkan facts below were read from `/mnt/code/f/ash` and the live solari tree at `/mnt/code/f/bevy/crates/bevy_solari`; each load-bearing claim is marked **confirmed** / **uncertain (RTX-gated)**.

---

## 1. Problem — the TransformPlugin collision

`big_space` (aevyrie, MIT/Apache-2.0) gives Bevy a floating-origin coordinate system: every spatial entity carries an integer grid coordinate (`CellCoord`, `GridPrecision = i64` by default) **plus** a normal f32 `Transform`. Its high-precision position is `cell.xyz * cell_edge_length + transform.translation`. To render, `big_space` **replaces Bevy's `TransformPlugin`** with its own CPU propagation (`Grid::propagate_high_precision` / `propagate_low_precision`), expressing every `GlobalTransform` *camera-relative* (origin-relative) so the f32 fed to the GPU is always small/sub-meter near the camera regardless of absolute galactic distance. `BigSpaceCorePlugin` **panics if Bevy's `TransformPlugin` is still present** (`big_space/src/plugin.rs:90-91`).

`bevy_solari` **also** replaces transform propagation — but on the GPU. Its `TransformGraph` (`transform/graph.rs:53`) mirrors the hierarchy into a `gpu_table!`, a single-pass GPU ancestor-walk (`transform/transform_propagate.wgsl`) computes each node's `world`, and a gather (`transform/gather.rs`) writes per-instance `transforms[i]` for the PTLAS build. This GPU table is a large perf win at the 1M-instance / 240 fps target.

**Both crates cannot run their propagation simultaneously**: `big_space` needs its CPU `Grid::propagate_*` systems; solari needs its GPU walk; and `big_space` panics if Bevy's CPU `TransformPlugin` is around at all. The resolution is **not** to run `big_space` — it is to **re-implement `big_space`'s floating-origin *model* natively inside solari's GPU tables + PTLAS**, with `big_space` as the credited reference algorithm.

---

## 2. The core idea

`big_space`'s per-entity render formula (`big_space/src/grid/mod.rs:143-166`) is:

```
cell_origin_relative = local_cell − local_floating_origin.cell      // INTEGER subtract (i64)
grid_offset          = cell_origin_relative * cell_edge_length      // then ×edge, in f64
global               = grid_transform_origin * (local + grid_offset) // grid_transform = identity for single-grid
```

For solari's primary case (one grid, the origin in that grid — bevy_city, a single ship) `grid_transform = identity`, so the formula collapses to **`local + (cell − origin) * cell_edge`**.

The hardware lever is **`VK_NV_partitioned_acceleration_structure`** (PTLAS), which solari already builds via raw ash (`accel/ptlas.rs`). A PTLAS partition can carry a **per-partition `[f32;3]` translation** that the driver adds to every instance in that partition at traversal. So:

- **Map a coarse cell-BLOCK (N×N×N cells) → one PTLAS partition**, with `partition_translation = (block_origin_cell − origin_cell) * cell_edge` (computed CPU-side in i64→f64→f32).
- **Instances store only their block-LOCAL (sub-meter-precise f32) transform.** The hardware adds the partition translation.
- **A floating-origin recenter becomes `O(partitions)`** `WRITE_PARTITION_TRANSLATION` ops — *not* `O(instances)` rewrites. Block-local instance transforms never change on recenter.

This **supersedes** the simpler "add `(cell−origin)*edge` in the propagate shader" prototype described in the design brief: that prototype bakes the offset into each node's f32 world during propagation and forces an `O(all-nodes)` re-walk on every recenter. The block-partition approach moves the offset into the hardware AS traversal and makes recenter `O(partitions)`. The shader-add path is **kept only as the unconditional fallback** (it has no partition-count limit).

---

## 3. Vulkan feasibility (verified)

| Fact | Verdict | Evidence |
|---|---|---|
| `WRITE_PARTITION_TRANSLATION` op exists, `op_type = 2` | **confirmed** | `ash/src/vk/enums.rs:2681` (`WRITE_INSTANCE=0, UPDATE_INSTANCE=1, WRITE_PARTITION_TRANSLATION=2`) |
| Translation record is **16 bytes**, `{ partition_index: u32, partition_translation: [f32;3] }`, **no `p_next`** | **confirmed** (digest's "8 B" was wrong → `arg_data_stride` must be **16**) | `ash/src/vk/definitions.rs:38243-38246` — plain `#[repr(C)]`, no s_type/p_next |
| `enable_partition_translation` is a `Bool32` on `PartitionedAccelerationStructureFlagsNV`, chained via `p_next` onto `…InstancesInputNV` (`Extends`), no `push_next` builder | **confirmed** | `definitions.rs:38089` (field), `definitions.rs:38112-38115` (`Extends<PartitionedAccelerationStructureInstancesInputNV>`); `InstancesInputNV` has field setters only — `p_next` must be assigned manually |
| Per-instance `partition_index` field already in the write-instance record | **confirmed** | `definitions.rs:38128-38138`; solari's WGSL mirror already writes it (`accel/ptlas_fill.wgsl` `make_record`, `instance_index`/`part`) |
| `PARTITION_INDEX_GLOBAL_NV = !0` | **confirmed** | `ash/src/vk/constants.rs:32`; matches solari's `PTLAS_GLOBAL_PARTITION` |
| `max_partition_count` is a queryable device property | **confirmed (value NOT read)** | `PhysicalDevicePartitionedAccelerationStructurePropertiesNV.max_partition_count`, `definitions.rs:38024`. **solari does not query it today** (must add). |
| The translation is **added at traversal to all instances in the partition** | **uncertain — INFERRED, not in this checkout** | The runtime-add prose lives in the absent man-page (`VkPartitionedAccelerationStructureWritePartitionTranslationDataNV.html`). The struct name, per-partition keying, the enable flag, and the GLOBAL exclusion are all strongly consistent with it, and it is the universally documented NV behavior — but it is **not** locally verifiable. **RTX-validate.** Cite as "per NV `VK_NV_partitioned_acceleration_structure` spec." |
| The GLOBAL partition (`!0`) **cannot** receive a translation | **uncertain (RTX-gated) — but design is robust either way** | VUID-10574 bounds `partition_index < input.partitionCount`; `!0` is never `<`. **Caveat:** the *instance* VUID-10569 reads identically yet solari legally routes instances to GLOBAL on RTX today — so the carve-out is in spec prose, not the VUID string. The reason to never translate GLOBAL is **semantic** (a translation on the spans-all partition is meaningless) + RTX behavior, not the bare VUID wording. |
| `src==dst` in-place / incremental builds; size once for worst case | **confirmed** | solari already does this (`accel/ptlas.rs:830-841`). Per-build counts must be `≤` sized counts. |
| ≤ one op of each type per build (max 3 indirect commands) | **confirmed (spec VUID-10564)** | Batch all translations into one strided array under one `IndirectCommand`. |

**Net:** every load-bearing struct/op/flag the design uses is present and correctly shaped in the ash fork. The two genuinely open items are the *runtime-add semantics* and the *GLOBAL-translation exclusion* — both **RTX-gated**, and the architecture is correct under either resolution because **movers live in GLOBAL and get full `WRITE_INSTANCE` on recenter regardless** (§6).

---

## 4. GPU-table representation

Extend the existing `TransformGraph` `gpu_table!` (`transform/graph.rs:53` — today: `LocalColumn`, `ParentColumn`, `NoReadbackColumn`, `NodeEntityColumn`, plus the `StaticColumn` presence column). Each new member auto-generates a `GpuColumnDesc` + scatter pipeline; only `extract_transform_graph` is hand-written.

| New column | Type | Role (big_space analog) |
|---|---|---|
| `CellColumn` | `[i32;4]` = `[cell.x, cell.y, cell.z, has_cell]` | mirror of `CellCoord` on every node with a `SolariGridCell`. **v1 = i32** (≈4× solar-system at edge 2000); i64 (`[i32;6]`) is a deferred **v2** width change. Only `(cell−origin)` is ever used, and that delta stays i32-safe by construction. |
| `GridParentColumn` | `u32` | node-slot of the enclosing **grid frame** (`ROOT_PARENT` for roots). Distinct from the transform-hierarchy `ParentColumn`. Degenerate (every leaf → one frame → root) in the flat case. |
| `GridOriginColumn` | `mat3x4` | per grid frame = big_space `LocalFloatingOrigin.grid_transform`. **Identity** for single-grid + axis-aligned nesting; non-identity only for a rotated/offset nested grid. Computed CPU-side (grids are few). |
| `FarCellColumn` | `u32` flag (or a presence column) | the single source-of-truth tag: instance is block-partition-routed. Drives **both** propagate-suppression **and** `make_record`'s block-local write. Default `0` → near-field shader-add path. |

New components/resources (exported from `transform/graph.rs`):

- `SolariGridCell { cell: [i32;3] }` — game-side mirror of `CellCoord`. Mirror **only onto leaf spatial entities**, never onto a grid frame plus its children (double-count).
- `SolariFloatingOrigin { origin_cell: [i32;3], cell_edge: f32 }` as `ExtractResource`. **Default `origin = 0, edge = 0` ⇒ every offset identically zero ⇒ non-big_space scenes byte-identical.** This no-op-by-default property is mandatory (CI guard).
- `TransformChangeFilter` (`graph.rs:115`, today `Or<(Changed<Transform>, Changed<ChildOf>)>`) → add `Changed<SolariGridCell>`.

**Propagate shader** (`transform/transform_propagate.wgsl`): in `load_local(node)` (currently lines 59-79, no offset), after building the node's TRS `Mat3x4`, gate on `has_cell != 0 && far_cell == 0`:

```wgsl
let icell       = cell[node].xyz - origin_cell;          // i32 subtract FIRST
let grid_offset = vec3<f32>(icell) * cell_edge;          // then ×edge in f32
m.r0.w += grid_offset.x; m.r1.w += grid_offset.y; m.r2.w += grid_offset.z;
```

When the ancestor walk steps through a grid frame whose `GridOriginColumn` is non-identity, left-compose that `mat3x4` (identity-skip in the flat case). **FarCell-tagged statics get NO cell add** — their offset lives entirely in the partition translation.

**Partition assignment** (`accel/ptlas_fill.wgsl`, `resolve_partition`, currently lines 119-126 — verified the exact binary static/mover routing point). Add a third branch:

```
FarCellColumn[node] != 0  →  numbered block partition = stable_block_index(floor(cell / N))
static_flags[node] != 0   →  regular partition 0          (today's static)
else                      →  PTLAS_GLOBAL_PARTITION = !0  (movers / node-less)
```

`stable_block_index` = a dense/Morton hash of `floor(cell/N)` — **stable for static geometry**, never `big_space`'s `PartitionId` (a monotonic counter reused on merge / minted on split that drifts under a stationary entity — disqualifying). `make_record` already writes `partition_index` (verified: it stamps `part = resolve_partition(slot)` into both `instance_written_partition[slot]` and the record), so **no record layout change** is needed — only FarCell-aware behavior.

---

## 5. Per-frame data flow

1. **Extract** (`extract_transform_graph`, main→render). Parallel `par_iter` over `Or<(Changed<Transform>, Changed<ChildOf>, Changed<SolariGridCell>)> + Without<TransformStatic>`. Fills thread-local `(local, parent, no_readback, entity, cell, grid_parent, grid_origin, far_cell)`. `local` (the big mover delta) → `write_delta_direct`; small columns serial-merged (cell on first-sight/cell-change, grid_parent/grid_origin/far_cell on first-sight/grid-change only). `SolariFloatingOrigin` extracted as a resource.
2. **GPU propagate** (`transform_propagate.wgsl`). Change-driven dispatch over the `LocalColumn` delta (`dispatch_count = full_rebuild ? node_count : local.pending()`, verified `propagate.rs:204-210`) **plus** the new persistent celled-slot dispatch on a recenter frame (§6). One thread per node walks its parent chain into a `mat3x4`; the cell offset is added per §4 for near-field celled nodes, suppressed for FarCell.
3. **Gather** (`transform_gather.wgsl`). `transforms[i] = world[node_slot[i]]` into the instance `TransformColumn` (KEEP_PREVIOUS, scene 0/1), shifting current→previous. Near-field → origin-relative world (unchanged consumer contract). FarCell → block-local transform (cell add was suppressed). **Single source of truth: gather emits exactly what `make_record` writes** (verified: move-detect compares `cluster_instance_transforms[slot]` vs `instance_previous_transforms[slot]`, and `make_record` copies that same `cluster_instance_transforms[slot]`).
4. **PTLAS op build** (`ptlas_fill.wgsl` + `ptlas.rs`).
   - `fill_seed` / `fill_incremental` write `WRITE_INSTANCE` records via `make_record`, `partition_index = resolve_partition(slot)`. Move detection is consistent because gather and make_record share the buffer (verified). Note `fill_incremental` also keys off `geometry_dirty[geom]` and the `force_all` full-rebuild path — a recenter must not silently flip `force_all`.
   - **On a recenter frame**, a small compute pass fills the partition-translation buffer (`[{partition_index, [f32;3]}]` per live block), and the build emits a **second** `IndirectCommand` (`op_type=2`, `arg_count = block_count`, `arg_data_stride = 16`) in the already-reserved `MAX_OPS=2` slot; `src_infos_count` set to `2` (else `1`). The `finalize` pass already writes `src_infos[1]` (the WRITE op's arg_count) — verified.
   - Raw-VK `cmd_build_partitioned_acceleration_structures` consumes both ops in-place (`src==dst`).

---

## 6. Recenter + partition lifecycle

A recenter = the `FloatingOrigin` (the camera) crosses a cell boundary ⇒ `SolariFloatingOrigin.origin_cell` changes, detected by a new `last_origin` field on `TransformPropagate`. This is **coarser** than big_space's full `LocalFloatingOrigin.eq` latch — it fires only on integer cell crossing; intra-cell origin drift is absorbed by the camera's own sub-cell `GlobalTransform`, not by any rewrite. Use big_space's **hysteresis band** `maximum_distance_from_origin = cell_edge/2 + switching_threshold` so the origin (and movers) don't thrash at boundaries.

Costs by tier:

- **Static bulk (FarCell / block partitions) — `O(block partitions)`. CONFIRMED SOUND.** `make_record` copies block-local transforms that do **not** change on recenter; propagate does not re-run for them (no `local` delta), so gather emits bit-identical worlds, `fill_incremental` sees `cur == prev` ⇒ no WRITE ⇒ carried from `src`. The whole recenter is one `WRITE_PARTITION_TRANSLATION` op over `block_count`. **Zero per-instance rewrites.**
- **Movers (GLOBAL) — `O(movers)`. CONFIRMED.** GLOBAL can't be translated, so each mover re-emits full origin-relative world via `WRITE_INSTANCE`. But gather already re-emits every moved instance every frame, and the gather pass is *already* `O(all instances)` unconditionally — a recenter just marks all movers "moved" for one frame. Bounded by the (small) mover count. **This is why the design is robust to the §3 GLOBAL-translation unknown.**
- **Near-field statics on the shader-add fallback — `O(celled changed nodes)`, and this needs NEW machinery.** ⚠️ **Verified gap:** propagation today dispatches only over `local.pending()` (`propagate.rs:205`); a recenter changes no node's `local`, so a shader-add static would keep a **stale world after a cell crossing = hard visual freeze**. This path requires a **net-new persistent celled-slot dispatch list** (built once, appended as cells are added), driving an `O(celled)` recenter dispatch. **Do NOT re-arm `needs_full_rebuild`** (it is the `O(all-nodes)` cold-start latch, `propagate.rs:162/320`). Fallback: if `celled_count` is a large fraction of `node_count`, fall back to `needs_full_rebuild`. **Completeness:** a non-celled static leaf under a celled grid frame must still get the offset — re-walk the grid frame's subtree on recenter, or use the full-rebuild fallback. A miss here is a hard freeze; **test at an actual cell boundary.**

### Partition lifecycle — the churn trap

⚠️ **Verified gap (major).** `partition_count` is passed to **both** the sizing query (`ptlas.rs:552`) and the build (`ptlas.rs:807`), and solari forces a full `O(all-instances)` rebuild whenever capacity/sizing grows (`ptlas.rs:423/433`). If `partition_count` tracked the *live* `block_count`, a **new block entering the active set would change the sizing and retrigger the full rebuild** — defeating the whole win.

**Mandatory mitigation:** pre-size `partition_count` to a **fixed, clamped maximum** queried at startup from `PhysicalDevicePartitionedAccelerationStructurePropertiesNV.max_partition_count` (not queried today) and **hold it constant across recenters**. With a dense stable block hash and a fixed maximum, static instances never change `partition_index` (their block is a pure function of `floor(cell/N)`), so block churn alone forces **no** instance rewrites and **no** re-size. `max_instance_per_partition_count` / `max_instance_in_global_partition_count` move from "= capacity" to conservative per-partition occupancy bounds, also fixed. The reverted spatial-grid-of-partitions (`ptlas.rs:50-64`) failed on **scene-spanning AABB overlap**; block partition *translation* is the opposite mechanism — block-local-only transforms give **cell-sized AABBs** (mechanism argument, **not yet RTX-measured**).

---

## 7. Precision analysis

Three tiers, near→far. The honest framing: **sub-meter precise for geometry NEAR THE CAMERA at any galactic camera position** — *not* "sub-meter at light-year scale" in absolute terms.

1. **Camera-relative frame (the core win, free).** The camera IS the `FloatingOrigin`, so `origin_cell` tracks it; `(cell − origin_cell)` is small near the camera ⇒ `vec3<f32>(icell)*edge + sub-cell local` keeps full f32 mantissa (~24 bits). `RtCamera.camera_position` (`render/rt_pipeline/mod.rs`, from `view.world_from_view.translation()`) is sub-cell, so ray origins and instance worlds share the same origin-relative frame and align sub-mm regardless of absolute coordinates. **A system must make the camera node's effective world its sub-cell transform** (origin cell subtracts to zero for the origin's own cell). Without this, ray origins and instance worlds diverge.
2. **Integer-first subtraction (the cancellation guard, non-negotiable).** `(cell − origin)` in i32/i64 **before** `× edge`, on **both** the GPU shader-add path **and** the CPU partition-translation builder `(block_origin − origin)` in i64 → f64 → f32-downcast. **NEVER** `cell*edge − origin*edge` in f32: at a 1-AU cell index (~7.5e7) two adjacent cells collapse to `0.0 m` in f32 (full 2000 m loss); the integer-first form is exact.
3. **Far-field partition translation (the only new precision machinery).** For blocks so distant that even cell-relative `icell*edge` would crush sub-cell bits, the magnitude lives in the f32 `partition_translation` `(block_origin − origin)*edge` (CPU i64→f64→f32), while the instance keeps a sub-cell block-local f32 transform. Bits lost: (a) the f32 translation has meter-ish error at extreme distance, but it is a **rigid shift common to the whole block** — intra-block relative geometry is exact (and a 1-ly block subtends ~0 px); (b) i32 absolute cells cap volume; (c) the coarse origin-cell compare ignores sub-cell origin drift (absorbed by the camera).

**The true ceiling is NOT f32 mantissa or f32 range** (4 ly = 4e16 m ≪ f32 max 3.4e38). It is **populated-block count vs queried `max_partition_count`** — partitions must be **sparse** (one per block that actually contains geometry). Dense space-tiling is hopeless (2M partitions at N=1 reach only ~1.3e5 m). If live blocks would exceed `max_partition_count`, **grow N** (coarsening the snapping) or fall to the shader-add path (no partition limit).

**i32 vs i64.** i32 cells **overflow at ~4 ly** (index 2e13 > i32 max 2.1e9). **v1 ships i32 = solar-system scale only.** The "light-year" reach **requires the v2 i64 cell column** (`[i32;6]` lo/hi). The `(cell − origin)` *delta* stays i32-safe even at v2 because camera-relative recentering keeps it small; only the CPU partition builder does the i64-subtract.

---

## 8. Nested grids

big_space: `global = grid_transform_origin * (local + (cell − origin)*edge)`. Single-grid (origin in-grid) ⇒ `grid_transform = identity`.

- **v1 (fully implemented + RTX-validated): single-grid** (bevy_city, one-grid ship). `GridOriginColumn` identity; compose step skipped. Block-partition translation is **exact** (pure translation).
- **Wired, axis-aligned-validated only in v1: 2-level nesting** via `GridParentColumn` + `GridOriginColumn`. The shader walks the grid-parent chain and left-composes each frame's `mat3x4`.
- **Rotated nested grids (spinning/orbiting planet) — post-v1, but this architecture is the only viable home.** **PTLAS `partition_translation` is TRANSLATION-ONLY.** A rotated grid's rotation **cannot** be a partition translation; it must be folded into the per-instance `mat3x4` on the free GPU walk (`GridOriginColumn` compose), while only the integer cell offset becomes the partition translation. Wiring the columns now (identity-skip, zero cost in the flat case) future-proofs this.

Double-subtract trap (two celled nodes in one chain both subtracting origin) is avoided: each node adds only its own cell offset; a grid frame contributes its `grid_transform`, not another origin subtraction. **v1 enforces exactly one celled node per `ChildOf` chain.**

---

## 9. Concrete change list (checklist)

**`transform/graph.rs`**
- [ ] Add `CellColumn [i32;4]`, `GridParentColumn u32`, `GridOriginColumn mat3x4`, `FarCellColumn u32` to the `gpu_table!` columns (line 53).
- [ ] Add `SolariGridCell { cell:[i32;3] }` component + `SolariFloatingOrigin { origin_cell:[i32;3], cell_edge:f32 }` `ExtractResource` (default `origin 0 / edge 0` = no-op).
- [ ] Extend `TransformChangeFilter` (line 115) with `Changed<SolariGridCell>`.
- [ ] Extend `extract_transform_graph` (line 161) to fill the new thread-local buffers (cell on cell-change/first-sight; grid_parent/grid_origin/far_cell on grid-change/first-sight).

**`transform/propagate.rs`**
- [ ] Extend `PropagateParams` (line 56, today 4 fields) with `{ cell_edge: f32, origin_x/y/z: i32 }`.
- [ ] Grow `transform_propagate_bind_group_layout` (line 99) from 5 → ~8 bindings (add cell, grid_parent, grid_origin, far_cell).
- [ ] Add `last_origin` field + recenter detection in `prepare_transform_propagate` (~line 184-210).
- [ ] **Add the persistent celled-slot dispatch list + the `O(celled)` recenter dispatch — NOT `needs_full_rebuild`** (leave it as the cold-start latch); large-fraction full-rebuild fallback.

**`transform/transform_propagate.wgsl`**
- [ ] In `load_local` (lines 59-79) add the integer-first cell offset gated on `has_cell && !far_cell`.
- [ ] Add `GridOriginColumn` left-compose in the ancestor walk (after `load_local`, identity-skip).
- [ ] Add the new bindings (today @binding 0-4).

**`transform/gather.rs` + `transform_gather.wgsl`**
- [ ] Near-field unchanged; for FarCell-tagged statics emit block-local transforms (read `FarCellColumn`).

**`accel/ptlas_fill.wgsl`**
- [ ] Extend `resolve_partition` (lines 119-126) with the FarCell → `stable_block_index(floor(cell/N))` branch. `make_record` (~line 144) already writes `partition_index` + transform — no struct change. `instance_written_partition` self-heal (line ~112, the `partition_changed` drift check, verified) handles drift automatically.

**`accel/ptlas.rs`**
- [ ] Chain `vk::PartitionedAccelerationStructureFlagsNV::default().enable_partition_translation(true)` via manual `input.p_next` at **BOTH** the sizing query (`size_input`, lines 551-553) **AND** the build input (lines 806-808). The flag struct must **outlive** the build call (no generated `push_next` on `InstancesInputNV`, verified — assign `p_next` manually).
- [ ] Replace `PTLAS_PARTITION_COUNT = 1` (line 160) with a **fixed clamped maximum** `partition_count = block_count_max`, applied identically at lines 552 **and** 807. **Query `PhysicalDevicePartitionedAccelerationStructurePropertiesNV.max_partition_count` at startup** (not done today) and clamp; assert `block_count_max + max_instance_in_global_partition_count ≤ max_partition_count` (VUID-10535).
- [ ] Add a **net-new 16-B-stride partition-translation buffer** (cannot reuse the 104-B `write_data`); fill it in a compute pass on recenter frames.
- [ ] Write the **second `IndirectCommand`** (op_type=2, `arg_data_stride=16`) at `src_infos` byte offset 24 (the reserved `MAX_OPS=2` slot, line ~515) pointing at that buffer.
- [ ] Set `src_infos_count = 2` on recenter frames, `1` otherwise (line 522, currently hardcoded `1`).

**`transform/mod.rs`**
- [ ] Register `SolariFloatingOrigin` `ExtractResource` + the new columns in the plugin. **Do NOT add any `big_space` CPU plugins.**

**`render/rt_pipeline/mod.rs`**
- [ ] Ensure `RtCamera.camera_position` derives from the camera's origin-relative `GlobalTransform` (camera = FloatingOrigin; its world is its sub-cell offset).

**ash ops added (all confirmed present):** `PartitionedAccelerationStructureFlagsNV.enable_partition_translation` (`definitions.rs:38089`, `Extends …InstancesInputNV` `:38112`); `PartitionedAccelerationStructureOpTypeNV::WRITE_PARTITION_TRANSLATION = 2` (`enums.rs:2681`); `PartitionedAccelerationStructureWritePartitionTranslationDataNV` **16 B** (`definitions.rs:38243`); `PARTITION_INDEX_GLOBAL_NV = !0` (`constants.rs:32`); `PhysicalDevicePartitionedAccelerationStructurePropertiesNV.max_partition_count` (`definitions.rs:38024`). The build wrapper `cmd_build_partitioned_acceleration_structures` (`gpu/extension.rs:508`) is **unchanged** — only `input.p_next`, `partition_count`, and `src_infos_count` differ.

---

## 10. What big_space code is replaced + attribution

| big_space (CPU) | solari native (GPU) |
|---|---|
| `LocalFloatingOrigin::compute_all` (per-BigSpace tree walk) | GPU grid-parent walk + `GridOriginColumn` compose in `transform_propagate.wgsl`. The "origin's grid = identity" seed becomes "origin_cell subtracts to zero for its own cell." |
| `Grid::propagate_high_precision` (CPU par_iter of `GlobalTransform`) | GPU ancestor-walk emitting origin-relative worlds. **The head-to-head conflict** — big_space's CPU propagation is **not** added. |
| `Grid::propagate_low_precision` + `tag_low_precision_roots` | solari's ancestor walk already composes plain Transform chains; non-celled leaves inherit the offset via composition. (UI keeps Bevy's own `sync_simple_transforms`.) |
| `big_space::partition/*` (Partition, PartitionLookup, merge/split, the O(cells) `reverse_map` clone) | **Not used.** Block partitions are stable cell-block Morton indices, **not** big_space connectivity `Partition`s. |
| `big_space::hash/*` (CellId, CellLookup, **ChangedCells**) | Not reused as code, but the **ChangedCells sparse-delta pattern** is mirrored by solari's change-driven extract + presence column + the new celled-slot recenter list. |
| `BigSpaceCorePlugin` (panics if Bevy `TransformPlugin` present) | **Not added.** solari's RenderApp propagate→gather schedule is the GPU analog of `LocalFloatingOrigins → PropagateHighPrecision`. |

**Kept as reference (credited):** the `CellCoord` integer-grid concept (mirrored as `SolariGridCell`), `Grid::cell_edge_length` / `maximum_distance_from_origin` (the recenter hysteresis band), `Grid::translation_to_grid` as a CPU helper for mover recentering. `RecenterLargeTransforms` (keeping each entity within half a cell) is the **precondition** the block-local-transform invariant relies on — replicate it CPU-side for movers, or accept it as authored.

> **Attribution.** The floating-origin algorithm — integer grid coordinate + within-cell f32 transform, camera-relative `local_floating_origin`, integer-first `(cell − origin)` before `× edge`, the recenter hysteresis band — is **`big_space` by Aevyrie (`https://github.com/aevyrie/big_space`, dual MIT / Apache-2.0)**. This design re-implements that model natively on solari's GPU tables + PTLAS; credit and the license notice must be carried in the solari source.

---

## 11. Open questions / RTX-validation-gated unknowns

1. **Runtime-add semantics (uncertain — INFERRED).** "The partition translation is added to every instance in the partition at traversal" is the NV man-page description, **not present in this ash/Vulkan-Headers checkout**. Strongly consistent with the struct shape + enable flag, but must be **observed on RTX**. Cite as "per NV spec," not as locally verified.
2. **Can the GLOBAL partition (`!0`) be translated? (uncertain — RTX-gated.)** VUID-10574 wording is identical to the instance VUID-10569 that GLOBAL legally violates today, so the wording alone doesn't decide it; the semantic + RTX behavior do. **The design never routes a translation to GLOBAL** and re-emits movers via `WRITE_INSTANCE`, so it is correct either way — but confirm on hardware.
3. **`max_partition_count` actual value (unverified).** Not read from hardware here (NV docs cite ~2^21-order). If small, the populated-block budget shrinks and N must grow. **Query at startup, clamp, grow N on exhaustion.**
4. **Partition-count churn → full rebuild (§6).** Verified failure mode: if `partition_count` tracks live `block_count`, a new block re-sizes and forces an `O(all-instances)` rebuild. **Must pre-size to a fixed clamped max.** Confirm with a streaming stress test crossing block boundaries.
5. **Block-local AABB ⇒ cell-sized AABB (claim 13, mechanism only — not RTX-measured).** The argument that this avoids the reverted spatial-grid traversal inflation (`ptlas.rs:50-64`) is sound in principle; **measure traversal cost on RTX** before trusting it at 1M instances.
6. **Near-field shader-add recenter completeness (§6, hard-freeze risk).** Every renderable static must be either celled (in the new dispatch list) or in the re-walk set; a non-celled leaf under a celled grid frame must be covered. **Test at an actual camera cell crossing** — a miss is a silent visual freeze, not a crash.
7. **Camera = FloatingOrigin coherence (§7).** `RtCamera.camera_position` and instance worlds must share the origin-relative frame. Verify ray origins and instance worlds stay sub-mm-aligned after a recenter; otherwise the near-camera precision guarantee breaks.
8. **i32 → i64 cell widening (v2).** v1 (i32) is solar-system-scale only; star-to-star (Tau Zero) needs the `[i32;6]` i64 column. Deferred, but the `(cell−origin)` delta stays i32-safe throughout.

---

## 12. Resolved gaps (post-review)

The seven review gaps are resolved below. The **block-index free-list allocator (gaps 3+4)** is the keystone and leads; everything else (op layout, recenter cost, motion-vector compensation, camera frame, compose-order) hangs off it. All file:line citations were re-verified against the working tree; where the tree has drifted ahead of the original review (it already ships bindings 15/16 and the `partition_changed` self-heal), the deltas below are written against the *current* code.

---

### 12.1 Gaps 3 + 4 (KEYSTONE) — the block→partition allocator

> **Gap 3:** the block→partition-index mapping must be both *stable* (a resident block keeps its index across frames so resident instances never re-WRITE) and *dense* (`≤ max_partition_count`), without inheriting big_space's PartitionId churn.
> **Gap 4:** eviction must reclaim partition indices without perturbing resident instances.

#### Design: `SlotPool<IVec3>` — first-touch alloc, free-list eviction

solari already ships the exact data structure — `SlotPool<K>` (`/mnt/code/f/bevy/crates/bevy_solari/src/ecs_gpu/slot.rs:179-265`): a `HashMap<K,u32>` + `Vec<u32>` free-list + monotonic `next` + a `generation` counter. It is the key-indexed sibling of `GpuSlotAllocator`'s `SlotFreeList` (slot.rs:81-92). Wrap it in a new render-world resource:

```rust
/// CPU-owned block-cell → dense PTLAS partition-index allocator. First-touch:
/// a block gets its partition index when its first instance streams in; freed
/// to the pool on eviction; reused only by newly-streamed blocks.
#[derive(Resource, Default)]
pub struct PartitionAllocator {
    pool: SlotPool<IVec3>,   // block cell-coord (cell >> BLOCK_SHIFT) → partition 0..N
}
```

`IVec3` is the **block** cell-coord (a coarsened big_space `GridCell`, `cell >> BLOCK_SHIFT`). The four required operations all already exist on `SlotPool`:

| operation | method | line | semantics |
|---|---|---|---|
| first-touch alloc | `allocate(key)` | slot.rs:227-241 | returns the **existing** slot if present (`map.get` early-return, slot.rs:229-231), else drains `free` before bumping `next` |
| eviction | `free(key)` | slot.rs:243-248 | pushes the index onto `free`, bumps `generation` |
| mark-and-sweep | `reconcile(present, is_present)` | slot.rs:252-264 | alloc every present block, free any tracked block no longer present |
| build sizing | `len()` | slot.rs:213-215 | returns `next` (the high-water = `partition_count` for the build) |
| translation payload | `iter()` | slot.rs:223-225 | `(block_coord, partition_index)` for every live block |

**STABLE for resident blocks.** `allocate` early-returns the existing index when the key is already mapped (slot.rs:229-231). A block that stays resident is re-`allocate`d every frame (an idempotent presence touch) and keeps the same index forever — `generation` does not even bump. A stationary static in that block therefore keeps `partition_index` constant across all frames, so in `fill_incremental` the `partition_changed` test (`/mnt/code/f/bevy/crates/bevy_solari/src/accel/ptlas_fill.wgsl:259`) stays *false* → the record is not re-WRITTEN, the driver carries the instance from `src` untouched. **Stable index ⇒ zero resident instance rewrite** — the whole point, since instances store block-LOCAL transforms that never move relative to their block.

**DENSE, `≤ max_partition_count`.** Indices come from `next` (slot.rs:233-236), reused from `free` before growing, so the high-water is bounded by the *peak* simultaneously-live block count. At steady state that is the streaming ball around the floating-origin camera — O(radius³) blocks, tens not millions. Query `max_partition_count` once (ash `definitions.rs:38024`) as the hard ceiling and assert `pool.len() ≤ max_partition_count`.

**NOT big_space PartitionId churn.** big_space reassigns/merges/splits PartitionIds as the spatial hierarchy reorganizes, so an existing entity's partition can change identity under it. Here the block↔index binding is *monotone*: created on first-touch, destroyed only on eviction, **never re-keyed, never coalesced**. A freed index (slot.rs:243-248) is handed back out *only* to a block that streams in later (slot.rs:232) — and that block's instances are brand-new to the PTLAS, so they must be WRITTEN anyway. **The reuse cost is fully absorbed by unavoidable add-work; no resident instance ever sees its index change.** That is the crisp distinction: big_space churn perturbs *resident* entities; this allocator perturbs *only* entities being added regardless.

#### GPU instance → block index: a per-node column, not a per-frame lookup

Add a node-indexed `BlockIndexColumn` (a `GpuColumn<u32>`, identical machinery to `NodeSlotColumn`/`Presence<StaticColumn>` already bound at ptlas.rs:728-733). When a static's block is (re)assigned CPU-side, scatter its dense partition index into this column keyed by the same node slot `resolve_partition` already dereferences. `resolve_partition` (currently ptlas_fill.wgsl:148-156, returning the hardcoded `0u` for statics) becomes a single array read:

```wgsl
fn resolve_partition(slot: u32) -> u32 {
    let node = node_slots[slot];
    if node == PTLAS_GLOBAL_PARTITION || static_flags[node] == 0u {
        return PTLAS_GLOBAL_PARTITION;   // movers / node-less → untranslated GLOBAL
    }
    return block_index[node];            // static → its block's dense partition
}
```

This reuses the existing `partition_index` record field (ptlas_fill.wgsl:45 — zero new per-record bytes) and the `instance_written_partition` self-heal mirror (ptlas_fill.wgsl:163-164 stamp, :259 compare). The self-heal **already in the tree** now also covers the block-index cold-start window: a static placed in GLOBAL before its `block_index` column scatters migrates into its translated partition the frame the column lands — exactly the window the self-heal was built for.

#### Eviction (gap 4), step by step

1. **CPU**: `partition_allocator.pool.free(block_coord)` (slot.rs:243-248) — index returns to `free`, `generation` bumps, `len()` holds (a hole) or shrinks. Driven by a block-despawn observer or `pool.reconcile(present, is_present)` (slot.rs:252-264) over the live block set. This is the `SlotPool` analog of `free_gpu_slot` (slot.rs:151-159).
2. **Resident instances of *other* blocks are untouched** — their keys are still mapped to the same indices (slot.rs:229-231); `partition_changed` stays false → carried from `src`. (`freed_slot_is_reused_immediately` test, slot.rs:281-294, proves freeing one slot does not perturb the held ones.)
3. **The evicted block's own instances** are removed the standard despawn way: they hit `disabled_slots` → seeded `PAIR_NULL` (ptlas.rs ~491) → `fill_seed` writes a null AS address. The index just becomes vacant.
4. **Only newly-streamed instances pay a WRITE** when a later block draws the recycled index — they are in `added_slots` and WRITTEN regardless.

Net eviction cost: O(evicted-block instances) WRITEs (unavoidable removals) + O(0) resident perturbation.

---

### 12.2 Gap 1 — PTLAS op layout (no `finalize` collision)

> **Gap 1:** adding a `WRITE_PARTITION_TRANSLATION` op must not collide with the GPU-driven `arg_count` write that `finalize` performs.

**The hazard.** `finalize` writes `src_infos[1] = atomicLoad(write_count[0])` (ptlas_fill.wgsl:276-278). Word index `1` is the **second u32 of the first 24-B record** = op[0]`.arg_count` (the GPU-driven WRITE_INSTANCE count). The translation op's `arg_count` is CPU-known (gap 2) and must never be touched by the GPU.

**Layout decision (order is moot):**
- **op[0] = WRITE_INSTANCE** — keep exactly where it is (written at byte 0, ptlas.rs:515-522). `finalize`'s `src_infos[1]` keeps hitting op[0].arg_count. **Zero change to `finalize`, zero change to the GPU write_count path.**
- **op[1] = WRITE_PARTITION_TRANSLATION** — lives at byte 24 (`1 * size_of::<IndirectCommand>()`, the 24-B mirror confirmed at ptlas.rs:130-146). Its `arg_count` is word **7** (`src_infos[7]`). `finalize` never writes word 7 → **no collision.**

`MAX_OPS = 2` (ptlas.rs:123) and `src_infos` is already sized `2 * 24 = 48 B` — the "reserved headroom for a future op" the comment at ptlas.rs:120-122 names. **This is that op. No buffer resize.**

**Why order is moot.** The two ops touch disjoint state (WRITE_INSTANCE → per-instance `write_data`; WRITE_PARTITION_TRANSLATION → per-partition translation array) and disjoint partitions. By the content invariant (gap 3), translated partitions hold *only* statics whose records are write-once, so on a recenter frame **statics emit no WRITE_INSTANCE op at all** (nothing moved, no flags drifted, the *index* is stable — only the partition's *translation* changed, which is not a per-instance property). The only WRITE_INSTANCE records on a recenter frame are movers in untranslated GLOBAL (which carry no translation). Disjoint partitions ⇒ neither order changes the result. (Belt-and-suspenders: place the translation op first; not required.)

---

### 12.3 Gap 2 — the translation op's `arg_count` is CPU-known

> **Gap 2:** the translation op's `arg_count` must be sourced without a GPU writeback (unlike WRITE_INSTANCE's count).

`arg_count` for WRITE_PARTITION_TRANSLATION = number of `WritePartitionTranslationDataNV` records = number of partitions whose translation we rewrite this frame = **live block count = `partition_allocator.pool.len()`** (slot.rs:213-215). The CPU free-list owns this number; there is no GPU dependency. Written CPU-side like every other non-`arg_count` field; `finalize` (the only GPU `src_infos` writer) never touches word 7.

**Payload** (`{partition_index: u32, partition_translation: [f32;3]}`, 16 B — matches ash `WritePartitionTranslationDataNV`, `definitions.rs:38243-38245`) is CPU-built: for each live block from `pool.iter()` (slot.rs:223-225), `translation = (block_origin − origin) * edge`. Write into a small host-visible buffer, ring-buffered if it changes per frame (per the raw-UBO-ring note — the untracked NV build reads it by device address).

**Exact ptlas.rs insertion** (right after the op[0] write, replacing the unconditional `src_infos_count = 1` at ptlas.rs:521-523):

```rust
// op[1] = WRITE_PARTITION_TRANSLATION. arg_count is CPU-known (= live block
// count); finalize only writes op[0].arg_count (src_infos word 1), never word 7.
let mut op_count = 1u32;
if recenter {
    let translation_op = IndirectCommand {
        op_type: vk::PartitionedAccelerationStructureOpTypeNV::WRITE_PARTITION_TRANSLATION
            .as_raw() as u32,                            // == 2 (ash enums.rs:2681)
        arg_count: resources.live_block_count,           // pool.len(), CPU-owned
        arg_data_start_address: resources.partition_translation_data.address,
        arg_data_stride: 16,                             // WritePartitionTranslationDataNV
    };
    render_queue.write_buffer(
        &resources.src_infos,
        size_of::<IndirectCommand>() as u64,             // byte offset 24 → op[1]
        bytemuck::bytes_of(&translation_op),
    );
    op_count = 2u32;
}
render_queue.write_buffer(&resources.src_infos_count, 0, &op_count.to_le_bytes());
```

Carry `op_count` into `resources.op_count` (the dynamic op count the build reads, ~ptlas.rs:640).

**One required NV-input change:** set `.enable_partition_translation(true)` on the `PartitionedAccelerationStructureInstancesInputNV` at **both** the sizing query (ptlas.rs:548-553) and the build (ptlas.rs:803-808) — the flag (ash `definitions.rs:38092-38119`) that makes the driver honor per-partition translations. And raise `PTLAS_PARTITION_COUNT` (the constant `1` at ptlas.rs:160) to a dynamic `partition_allocator.pool.len().max(1)` fed to `.partition_count()` at ptlas.rs:552 **and** ptlas.rs:807 (they must match — the build comment at ptlas.rs:802 already insists).

---

### 12.4 Gap 5 — partition-translation composition under rotation/scale

> **Gap 5:** does a pure-translation partition offset compose correctly when instances carry rotation/scale?

The NV partition translation is a **world-space post-translation**, applied to the instance's already-transformed geometry at traversal — NOT an object-space pre-translation folded into the instance matrix. This is derivable (no RTX needed) from the headers: `WritePartitionTranslationDataNV` carries a `[f32;3]` **only** (`definitions.rs:38245`), no 3×3 — so it cannot live inside the instance's object space (an object-space offset of a rotated/scaled instance would need the offset rotated/scaled, impossible for a bare vec3). It is separate from the per-instance `transform` (the full 3×4 affine solari already writes in `make_record`, ptlas_fill.wgsl:148-152).

**Composition.** Instance carries block-LOCAL affine `T(x) = R·S·x + b`; the partition carries `t_partition`. The driver evaluates `world(x) = t_partition + T(x) = t_partition + R·S·x + b`. The big_space target under grid transform `G` is `world_target(x) = G·R_local·S·x + G·b_local + G·(cell_origin·edge)`.

**Exact for the single axis-aligned grid (`G = I`, camera == FloatingOrigin):** rotation/scale `R·S = R_local·S` ✓; local translation `b = b_local` ✓; cell offset `t_partition = cell_origin·edge` ✓ — and crucially exact for a *rotated/scaled* instance too, because the partition term is added in WORLD space **after** `T`, matching `world_target`'s `+ (cell_origin·edge)` which is also outside `R_local·S`. **So "local + cell_origin·edge" is exact regardless of instance rotation/scale**; rotation/scale ride inside `T`, only the integer cell offset rides `t_partition`. This is the load-bearing reason the scheme works.

**Out of scope (document as a constraint):** a rotated/scaled *nested* grid (`G ≠ I`). Then `t_partition = G·(cell_origin·edge)` (still a pure world translation, exact) but G's 3×3 must be folded into the instance matrix — `cluster_instance_transforms[slot]` must emit `G·T_local`, not `T_local`. solari targets `G = I` today, so this is the constraint to record if nested rotated grids are ever added.

**RTX-gated:** that the driver applies `t_partition` as `world = t_partition + T(x)` at traversal (the headers give layout, not application-order prose). Validate with a 45°-rotated, non-uniformly-scaled instance in a partition with `t_partition = (D,0,0)`: the hit must equal `(D,0,0) + R·S·x + b` (rigid shift), not a sheared/pre-rotated offset.

---

### 12.5 Gap 6 — recenter motion-vector spike (movers)

> **Gap 6:** on an origin recenter, GLOBAL movers' `previous` transform is still in the old-origin frame, producing a one-frame MV spike (~cell_edge) → DLSS/ReSTIR ghost.

**Single buffer feeds all three consumers.** `TransformColumn::previous` (`/mnt/code/f/bevy/crates/bevy_solari/src/ecs_gpu/column.rs:126-127,182`) is read by:
1. **DLSS/ReSTIR MV** — scene binding 1 `previous_frame_transforms` (`bindings/raytracing_scene_bindings.wgsl:94`; consumed at `chit_opaque.wgsl:190-198`, `chit_glass.wgsl:115-119` as `cur_uv − prev_uv`).
2. **PTLAS move-detect** — `instance_previous_transforms` (ptlas_fill.wgsl binding 10, compared at :249-251).
3. The gather writes it — `transform_gather.wgsl:45-47` shifts current→previous.

All three bind the same `previous_buffer()` (gather.rs ~143; `ecs_gpu/scene_columns.rs:120`; ptlas.rs:689,723). One buffer to patch.

**Root cause.** `TransformColumn` holds the **origin-relative world** PTLAS transform. On recenter (`origin: old→new`, `Δ = (new_origin − old_origin)·edge`): FarCell statics are bit-identical (their effective shift is carried by the partition translation, symmetric with the camera) — no spurious move. But GLOBAL movers get new-origin world in *current* while *previous* still holds old-origin world → `motion ≈ true_motion − Δ`, a ~`|Δ|` spike on every cell crossing. Standard floating-origin + TAA/DLSS camera-relative MV problem.

**Fix — gather-shader branch, movers-only, O(movers).** On a recenter frame apply `−Δ` to `previous` for GLOBAL instances only, **before** the current→previous shift. Statics are skipped (their column-world is block-local and must NOT be shifted). Gate on the same predicate `resolve_partition` uses (`static_flags[ns] == 0u`). Translation-only — the 4th column of the `mat3x4` rows; never touch the 3×3 linear part.

**Edits:**
1. **`transform/gather.rs` — `GatherParams`** (gather.rs:39-44, currently `_pad0`/`_pad1`): replace the pads with `recenter: u32` + `origin_delta: [f32;3]`, keeping 16-B alignment.
2. **`transform/gather.rs` — `prepare_transform_gather`** (gather.rs:90-110): read the floating-origin recenter signal (the big_space→solari bridge resource owning `origin`/`edge`); set `recenter`+`origin_delta`, else `recenter = 0`.
3. **`transform/gather.rs` — layout + bind group**: add binding **5** `storage_buffer_read_only_sized(false, None)` for `static_flags` in `transform_gather_bind_group_layout()` (gather.rs:56-72, after the `uniform_buffer` at 4); add a `Res<GpuColumn<Presence<StaticColumn>>>` param to `prepare_transform_gather_bind_group` (gather.rs:120-148) and bind it. The once-built bind group (early-return at gather.rs:131-133) stays — `static_flags` has a stable sparse handle; the recenter signal rides the already-per-frame `params` UBO.
4. **`transform/transform_gather.wgsl`** (the consts at :10-15, the binding block ending at :24, the shift at :45-47): add the new `GatherParams` fields + `@group(0) @binding(5) var<storage, read> static_flags: array<u32>;`. **Before** the shift:
   ```wgsl
   if params.recenter != 0u && static_flags[ns] == 0u {  // movers only — O(movers)
       previous[dst].w      -= params.origin_delta.x;
       previous[dst + 1u].w -= params.origin_delta.y;
       previous[dst + 2u].w -= params.origin_delta.z;
   }
   ```
   (`ns = node_slot[i]` already computed at :35.) The existing shift (:45-47) and new-current write (:48-50) proceed unchanged.
5. **No consumer changes.** chit MV math, scene binding 1, and ptlas move-detect all read the corrected `previous`; the gather runs in the `Propagate` graph node before PTLAS fill and shading, so the fix is visible to all three the same frame. `ptlas_fill`'s `moved` test stays correct (movers are re-specified that frame regardless). Sign convention: pin `Δ = (new − old)·edge` in the bridge resource and flip in one place if the bridge defines it as old−new.

---

### 12.6 Gap 7 — origin-relative camera transform source

> **Gap 7:** `RtCamera` is built from bevy's **absolute** `GlobalTransform`, while instance worlds are **origin-relative** → f32 catastrophe at AU scale.

**Root cause (verified).** rt_pipeline/mod.rs:457 `let world_from_view = view.world_from_view.to_matrix();` and :473 `camera_position: view.world_from_view.translation()…` both read the absolute `GlobalTransform` (the only transform `ExtractedView` carries). raygen consumes `camera.camera_position.xyz` (ray origin) and `camera.inverse_view_proj` (per-pixel direction, raygen.wgsl:59-60). Every instance world is GPU origin-relative (composed from `LocalColumn` TRS in `transform_propagate.wgsl`). Under big_space, instance worlds are bounded near the origin while the camera position is AU-scale → divergence.

**The camera is the zero-offset node.** It carries a `GlobalTransform`, so `gpu_table!` (graph.rs:57-58) already allocates it a node slot and scatters its **intra-cell** `Transform` into `LocalColumn`. Since the camera IS the FloatingOrigin (`cam_cell == origin_cell`), its cell offset is exactly zero ⇒ its origin-relative world equals its intra-cell LOCAL transform.

**Fix — option (b): CPU view from the camera's intra-cell `Transform`** (not (a) GPU readback). (a) needs a per-frame GPU→CPU readback of `world[cam_slot]` (`propagate.rs:119` `current_world() -> &Buffer`, GPU-only) plus a frame of latency — exactly what `NoGpuGlobalTransformReadback` (`readback.rs:81-92`) exists to avoid for the camera. (b) is zero-latency, zero-readback, and provably identical: the camera's intra-cell `Transform` is the same `LocalTRS` the GPU composes, and `world == local` for the floating-origin camera by construction.

**Edits:**
1. **`render/mod.rs`** (~:152-155): extract the camera's **cell-local** transform — a field on `SolariCamera`'s extract or a sibling `SolariCameraOriginRelative(GlobalTransform)`, populated from the main-world camera's cell-local pose (big_space `propagated_transform`; plain `GlobalTransform` when big_space is absent). Register its `ExtractComponentPlugin` alongside the existing one (:50).
2. **`render/rt_pipeline/mod.rs`** (~:269-307): add the extracted component to the `ViewQuery` tuple and destructure it.
3. **`render/rt_pipeline/mod.rs:457`**: `world_from_view` ← the extracted origin-relative transform (`solari_cam.origin_relative.to_matrix()`). Downstream `view_from_world` (:458), `world_from_clip` (:459), `clip_from_world` (:464), DLSS `prev_clip_from_world` (:465) inherit the fix; `view.clip_from_view` (projection) is origin-independent, unchanged.
4. **`render/rt_pipeline/mod.rs:473`**: `camera_position` ← same origin-relative translation, keep `.extend(camera.exposure)`.
5. **`render/rt_pipeline/mod.rs:465,498-500`**: on a recenter frame, rebase `RtPrevViewProj.clip_from_world` by the recenter delta (or mark motion invalid that frame). Pure recenter concern; no steady-state change.
6. **No change** to `transform_propagate.wgsl`, `transform_gather.wgsl`, `graph.rs`, `slot.rs` — the camera already has a node slot and its `LocalColumn` is its intra-cell pose; only *which* CPU transform feeds the view matrices changes. Keep `NoGpuGlobalTransformReadback`.

**Sub-mm proof across recenter.** Instance world `W_i = T_block_local(i) + (c_block(i) − c_o)·e` (cell term as partition translation); camera ray origin `O = T_cam_local`, and `c_cam == c_o` ⇒ camera cell term = 0 ⇒ `O` equals the camera's origin-relative world exactly — same arithmetic as any node with a zero cell term. Both in the origin frame, magnitudes bounded by cell + scene-span, never AU. On recenter (`c_o → c_o'`): partition translations rewrite O(partitions), the camera's `c_cam` advances with `c_o'` so its cell term stays 0 — `O` re-expresses in the new frame automatically; both sides shift by the identical integer-cell delta → bit-for-bit co-located. Only per-recenter fixup: rebasing `prev_clip_from_world` (edit 5).

**Sanity guard:** with big_space absent the extracted transform == absolute `GlobalTransform`, so the change is a numerical no-op — verify a non-big_space example (e.g. `examples/3d/solari/cathedral.rs`) renders byte-identical before merging.

---

### Checklist deltas (supersedes/augments §9)

**New module: `accel/partition.rs`** (or in `accel/ptlas.rs`)
- `PartitionAllocator(SlotPool<IVec3>)` resource. `allocate(block)` first-touch; `free`/`reconcile` on eviction (mirror `assign_gpu_slots`/`free_gpu_slot`, slot.rs:135-159, driven from the live block set). Expose `len()` → `live_block_count`, `iter()` → translation payload. Assert `len() ≤ max_partition_count` (ash `definitions.rs:38024`, queried once).
- **No change to `slot.rs`** — `SlotPool<K>` (slot.rs:179-265) reused verbatim.

**New: `BlockIndexColumn`** — node-indexed `GpuColumn<u32>` (clone `NodeSlotColumn` wiring). Scatter a static's dense partition index on (re)assignment. Bind in `prepare_ptlas_fill_bind_group` (append to the `BindGroupEntries::sequential` tuple, ptlas.rs:712-734, after the current binding 16 `instance_written_partition`) → new binding **17**; add the matching entry to `SolariResourceManager.ptlas`.

**`accel/ptlas.rs`**
1. Add to `Ptlas`: `partition_translation_data` (host-visible/ring buffer, 16 B × max_partition_count), `live_block_count: u32`, `recenter: bool`. Allocate in `init_ptlas`.
2. `PTLAS_PARTITION_COUNT` const (ptlas.rs:160) → dynamic `partition_allocator.pool.len().max(1)`, fed to `.partition_count()` at **ptlas.rs:552 and ptlas.rs:807** (keep equal).
3. `.enable_partition_translation(true)` on the `InstancesInputNV` at **ptlas.rs:548-553 and ptlas.rs:803-808**.
4. In `prepare_ptlas_params`: detect recenter (origin cell changed); build the 16-B translation payload from `pool.iter()` (`(block_origin − origin)·edge`); write op[1] at byte 24 + set `src_infos_count = 2` on recenter, else 1 (insertion in §12.3, replacing the unconditional write at ptlas.rs:521-522).
5. Carry the dynamic op count into `resources.op_count` (~ptlas.rs:640).

**`accel/ptlas_fill.wgsl`**
1. `finalize` (:276-278): **unchanged** — `src_infos[1]` targets op[0].arg_count; the translation op's `arg_count` (word 7) is CPU-written.
2. Add `block_index` (node-indexed `array<u32>`) binding at the next free `@group(1)` slot (**17**).
3. `resolve_partition` (:148-156): statics return `block_index[node]` instead of hardcoded `0u`; movers/node-less stay `PTLAS_GLOBAL_PARTITION`. Existing `partition_index` record field (:45) + `instance_written_partition` self-heal (:163-164 stamp, :259 compare) carry it unchanged — and now also cover the block-index cold-start window.

**`transform/gather.rs` + `transform/transform_gather.wgsl`** (gap 6)
- `GatherParams`: pads → `recenter: u32` + `origin_delta: [f32;3]` (gather.rs:39-44 + wgsl :10-15).
- `prepare_transform_gather` (gather.rs:90-110): set `recenter`/`origin_delta` from the bridge.
- Layout + bind group: add binding **5** `static_flags` = `Presence<StaticColumn>` (gather.rs:56-72, :120-148); once-built bind group preserved.
- `transform_gather.wgsl` (:24, before :45): movers-only `previous[dst+k].w -= origin_delta` translate, gated `params.recenter != 0u && static_flags[ns] == 0u`.

**`render/rt_pipeline/mod.rs` + `render/mod.rs`** (gap 7)
- Extract camera cell-local transform (render/mod.rs ~:152-155 + plugin reg :50).
- rt_pipeline `ViewQuery` (~:269-307): add the component.
- Source `world_from_view` (mod.rs:457) and `camera_position` (mod.rs:473) from it.
- Rebase `RtPrevViewProj.clip_from_world` (mod.rs:465,498-500) on recenter frames.
- Keep `NoGpuGlobalTransformReadback` (readback.rs:81-92). Verify a non-big_space example renders byte-identical.

**Verification gates**
- RTX-gated: driver applies `t_partition` as world-space post-translation at traversal under `enable_partition_translation` + op_type=2 (gap 5 test: 45°-rotated non-uniform-scale instance, `t_partition=(D,0,0)`, hit == rigid shift).
- Numerical: non-big_space example byte-identical (gaps 6+7 are no-ops without a recenter signal).

---

## 13. Allocator invariants the adversarial re-check requires

The block→partition `SlotPool` allocator (§12.1) was adversarially re-checked against `ecs_gpu/slot.rs`. All four
sub-points held — a stationary static never changes index; the live index set is bounded; eviction never perturbs
resident instances of other blocks; the GPU lookup stays consistent via the existing `partition_changed` self-heal —
**conditional on three invariants that must be made explicit in code, not assumed:**

1. **Block presence ≡ "≥1 resident instance," never a separate streaming flag.** `SlotPool::allocate` is a no-op for a
   mapped key (`slot.rs:229-231`, before the `generation` bump at `:238`), so a resident block keeps its index forever —
   *unless* `reconcile`'s `is_present` predicate flickers false for a frame in which a resident static still exists. If
   presence is derived from a streaming signal that can lag, `free` runs and a later `allocate` may hand the block a
   different index from the LIFO free-list, silently migrating a resident static. **Derive presence only from live
   instance count, and assert it in the bridge.**
2. **The `max_partition_count` guard must target a *compacted* high-water, or be a soft throttle — not a panic on `next`.**
   `SlotPool::len()` returns `next`, a high-water that never shrinks (no compaction today). A transient streaming spike
   (fast fly-through, teleport, an eviction-lags-ingest frame) ratchets `next` up permanently, so a hard
   `assert(len() ≤ max_partition_count)` would brick the build even after the active ball shrinks again. **This is the
   one real `SlotPool` addition gaps 3/4 need: high-water compaction when `free` covers the tail** (alternatively, make
   the ceiling a streaming throttle rather than a panic).
3. **Eviction removal + `free` + translation-op rebuild must be sequenced in one extract (atomic per-frame).** The
   self-heal detects a change in the *resolved partition value*, not in which physical partition an index points at. An
   evicted block's index recycled to a new block while a stale static still carries it (only reachable if invariant 1 is
   violated) would leave the self-heal blind. Sequencing the evicted block's instance removal
   (`disabled_slots`→`PAIR_NULL`), the `pool.free`, and the translation-op rebuild in the same extract closes the hole.

No sub-point was found broken: the scheme is buildable. These three invariants are the difference between robust and
fragile, and belong in the implementation as asserts/ordering constraints.

---

## 14. Implementation order (RTX gate first)

The whole design rests on one hardware-observable behavior (§3, §11.1): that `WRITE_PARTITION_TRANSLATION` adds the
per-partition `[f32;3]` to every instance in that partition at traversal. That is the **gate** — everything in the
PTLAS op-wiring path assumes it. So the build order front-loads validating it, and lets the RTX-*independent*
foundation proceed in parallel without betting on it.

### Stage 0 — RTX validation gate *(do first; blocks Stages 2–3)*
- Minimal raw-VK probe on the RTX box: build a 2-partition PTLAS, set `enable_partition_translation`, write one
  `WRITE_PARTITION_TRANSLATION` to partition 1, and confirm instances in partition 1 render shifted by the translation
  while partition 0 / the GLOBAL partition (`!0`) are unaffected. Read back `max_partition_count`.
- Resolves §11.1 (traversal-add semantics) and §11.2 (can GLOBAL be translated). If it behaves differently, only the
  op-wiring changes — the Stage 1 foundation and the precision model stand regardless.

### Stage 1 — RTX-independent CPU/GPU-table foundation *(safe to build now, no gate)*
- [x] `SlotPool::compact()` + tests — the high-water reclaim (§13 invariant 2). **Done, green.**
- [ ] `PartitionAllocator` resource = `SlotPool<IVec3>` (block cell-coord → dense partition index), with the §13
  invariants enforced: presence ≡ ≥1 resident instance; `compact()` before any `len() ≤ max_partition_count` guard.
- [ ] `SolariGridCell` / `SolariFloatingOrigin` types + extract (already prototyped, uncommitted) — the cell mirror.
- [ ] Camera origin-relative frame (§12.6): feed `RtCamera` from the camera's own origin-relative propagated world.
  Pure transform sourcing; verifiable by a CPU assert that ray-origin and a known instance world stay aligned.
- All of Stage 1 is cargo-checkable + unit-testable without a GPU and commits to nothing the gate could invalidate.

### Stage 2 — GPU routing *(gated on Stage 0)*
- [ ] `BlockIndexColumn` (node-keyed `GpuColumn<u32>`) + the `resolve_partition` FarCell→`block_index[node]` branch
  (`ptlas_fill.wgsl`). Inert until Stage 3 writes translations, so land it with Stage 3.

### Stage 3 — PTLAS op wiring *(gated on Stage 0)*
- [ ] `enable_partition_translation` flag chained on the sizing + build inputs; fixed clamped `partition_count`;
  the second `IndirectCommand` (op=2, stride 16) + `src_infos_count` toggle; the translation buffer fill.

### Stage 4 — recenter dynamics *(after 1–3 land)*
- [ ] Recenter detection (`last_origin`) + the previous-buffer origin-shift for movers (§12.5, the DLSS MV-spike fix);
  partition lifecycle (eviction sequencing, §13 invariant 3).

**Net:** Stage 1 is the safe forward progress available pre-hardware; Stages 2–4 wait on the Stage 0 probe so no
op-wiring is written against an unverified assumption.
