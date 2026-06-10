# bevy_solari — profiling notes

Working notes for the cluster ray-tracing / pathtracer work in `bevy_city`.
How we capture, how to read it, and what we've found. Scene = `bevy_city`
(default `--size 30`, ~900 blocks), GPU = RTX 5090, Vulkan.

## Execution model (read this first)

Pipelined. With `PipelinedRenderingPlugin` (default), the main world's frame
N+1 runs concurrently with the render world's frame N, and the GPU runs async
behind the render thread's submits. So frame time is
`max(main-thread, render-thread, GPU) + extract-sync`, NOT the sum.

> **Status (2026-06-08, size 100):** ~250–280 fps **uninstrumented**, ~5.3 ms
> *traced* (tracy's per-span cost makes traced fps pessimistic — trust the
> uninstrumented number). The transform extract (`extract_transform_graph`) is
> ~38% lighter than the baseline below after the bandwidth work (observer-driven
> instance extract, `TransformStatic`, TRS-in-local, parallel delta merge). The
> baseline numbers in the rest of this section are **pre-optimization** — kept
> for the methodology, not as current targets. Re-measure before trusting them.

**This scene is CPU-bound, render thread is the long pole.** Baseline measured
(`--size 30`, RTX 5090, NVIDIA 580, steady state, camera still): frame ≈ 5.74 ms
(~174 fps); `schedule{Render}` ≈ 4.69 ms; `schedule{Main}` ≈ 4.25 ms (parallel
half); GPU ≈ 0.8 ms *visible* (pathtracer 608 µs + bloom/clustering/…). The GPU
sits idle most of the frame — spend optimization budget on render-thread CPU,
not GPU passes. (AS builds are raw-VK and invisible to wgpu timestamps, see
"Reading the GPU side"; even +several-hundred-µs for them, GPU ≪ frame.)

A load-bearing lesson from the extract work: `extract_transform_graph` was
**bandwidth-bound** (building + uploading the per-mover delta), not scan- or
compute-bound. The wins came from shrinking bytes (TRS-in-local: 40 B vs a
packed 48 B mat3x4, and no CPU `compute_affine`) and killing the serial merge
(parallel disjoint copy). Re-homing the table to the render world, `set_if_neq`
+ `SyncToRenderWorld` change detection, and offloading the affine pack to a
pipelined render system were all tried and **regressed** — they moved or
duplicated bandwidth. Measure the byte path, not the instruction count.


## Capturing (`~/.cargo/bin/capture.sh`)

One-shot Tracy capture of `bevy_city` (release + solari + tracy). Run from a
session WITH a display: `! capture.sh`.

Outputs to `~/bevy-traces/bevy_cityNN.{tracy,csv,gpu.csv,log}` (next free index).
- `.tracy` — full trace (open in Tracy GUI for range selection).
- `.csv` — CPU zone stats (per-zone total/mean/count).
- `.gpu.csv` — **GPU zone events** (`tracy-csvexport -g`); pathtracer/bloom/etc.
  durations. Always exported now — essential to tell CPU-encode from GPU-execute.
- `.log` — app stdout/stderr.

### Steady-state only (no load/bake spike)
Two mechanisms, both required:
1. **`trace_tracy_ondemand`** feature (bevy_log→internal→root) — Tracy records
   ONLY while a profiler is connected. Without it Tracy back-fills its entire
   buffered history on connect, so gating connect-time alone doesn't exclude load.
2. **Connect at steady state.** `bevy_city`'s `signal_capture_ready` logs
   `CAPTURE_READY` once spawned + (solari) all meshes converted; capture.sh waits
   for it, else falls back to `READY_TIMEOUT` (default 10s — past this scene's
   ~7s load/bake).

### Tunables (env)
- `EXTRA_FEATURES` — extra cargo features.
- `READY_TIMEOUT` (default 10), `READY_MARKER`, `SETTLE`, `SECONDS_CAP`,
  `OUTDIR`, `PREFIX`, `OUT`, `FILTER`.

### Methodology gotchas (learned the hard way)
- **Summed per-system µs OVERCOUNTS.** Systems run `par_iter` across worker
  threads and pipeline; only the **frame-time delta** + **which thread** the work
  is on are meaningful.

### Reading the GPU side
`tracy-csvexport -g <trace>` (now auto in `.gpu.csv`). NB: the **AS builds use
raw-VK `as_hal_mut`** and are INVISIBLE to wgpu's GPU timestamps. The old
per-pass `gpu_timing` fence-waits (CPU `*.gpu_wait` spans) were removed for the
single-submit migration — a per-pass `poll(Wait)` can't coexist with the graph's
deferred submit. The `gpu_timing` cargo feature still exists but is now **inert**
(no code reads it); AS-pass GPU timing needs timestamp queries
(`RenderContext::diagnostics_recorder`) instead — see `docs/plan_systems.md`.

## Findings

### Render-graph single-submit migration (commit 91e06d7011, traces 44→43)
Moved the cluster AS dispatch chain onto render-graph-as-systems so the graph
does one submit at `RenderGraphSystems::Submit`.

- **Submit count: ~6 → 1 `vkQueueSubmit`/frame.** Was: main render submit
  (`count≈13`) + `instance_columns.submit` + selector + blas_sharing +
  ptlas.fill + `ptlas.build_submit`. Now: one submit, `count≈19`.
- **Frame time: unchanged** (5.745 → 5.737 ms). The submit cost was
  *consolidated, not eliminated* — 44's `153 µs (main) + 297 µs
  (instance_columns.submit) ≈ 450 µs` ≈ 43's `434 µs` single submit. On NVIDIA
  580 the `vkQueueSubmit` cost scales with command-buffer content + queue-sync,
  not per-call fixed overhead, so 6→1 calls bought ~nothing on the clock. The
  `+750 µs` that shows up in `render_system`/`schedule{RenderGraph}` in 43 is
  just the AS dispatch work relocating into the graph sub-schedule
  (`schedule{Render}` total is flat → work is conserved, not added).

**Correction to the earlier hypothesis (submit overhead = the long pole): it is
not.** The migration is still worth keeping — idiomatic render-graph systems, and
one shared `RenderContext` is the foundation ReSTIR PT records into — but it is
NOT a perf win. Don't measure or claim it as one.

### Where the render-thread time actually goes (trace 43, mean µs/frame)
| system | µs | note |
|---|---|---|
| `camera_driver` (Core3d encode) | 1014 | pathtracer encode |
| `check_visibility_cpu_culling` | 528 | raster cull — RT path may not need it |
| `prepare_cluster_scene_bind_group` | 437 | solari: rebuilt EVERY frame ← cache it |
| `submit_pending_command_buffers` | 434 | the one submit |
| `extract_cluster_instances` | 421 | solari |
| `prepare_instance_columns` | 313 | solari |
| `simulate_cars` | 203 | app sim |
| `prepare_clusters_for_gpu_clustering` | 192 | PBR raster clustering — still on |
| `dispatch_ptlas` | 120 | solari |

Next levers, in order: (1) diff-cache `prepare_cluster_scene_bind_group` (it
rebuilds the whole scene bind group + re-resolves the layout every frame — the
same waste the instance-columns refactor killed); (2) drop the raster
freeloaders the full-RT path shouldn't pay for (`check_visibility_cpu_culling`,
`prepare_clusters_for_gpu_clustering`).

## Branches
- `solari-pt` — baseline (PbrPlugin on, StandardMaterial).
- `solari-no-pbr` — PbrPlugin disabled; bevy_solari owns material/lights/DfgLut +
  vendored shaders (see the PbrPlugin-decoupling work).
- `solari-no-vis` — `solari-pt` + strips raster visibility from RT instances (didn't help).



● Verified. The CLAS-queue clear (clas_arena 534-543) is actually fine — those None/is_none() paths are permanent "RT unsupported"
  conditions, not transient, so clearing is correct (just missing a warn!). Agent A overstated that one. The two instance_manager
  findings are real.
     
  Here's my synthesized assessment, ranked by impact. I verified the Tier 1 items against the source; the unsafe ones in Tier 4 are
  plausible-but-I'd-confirm-with-validation-layers.

  Tier 1 — Latent correctness bugs (verified, fix these)

  1. AssetEvent::Modified evicts GPU ranges without re-binding the instances that point at them — instance/instance_manager.rs:551-554.
  remove(&id) frees the mesh's vertex/cluster slices back to the RangeAllocator, but instances keep their cached SlotMeshPointers
  (cluster_base, …) and nothing re-binds them (a Modified asset doesn't flag the entity Changed). The next upload reuses those ranges →
  instances silently sample another mesh's geometry. Fix: drop Modified from the match (re-uploading an unchanged asset id is a no-op
  anyway; the eviction is the bug), or collect affected entities and force a re-bind.
  2. despawn→bind in the same frame can reuse a slot that's simultaneously "disabled" and "added" — instance_manager.rs:561-617. Despawn
  pushes the slot to free_slots and disabled_slots; the bind loop pops free_slots (LIFO) and pushes the same slot to
  added_slots/column_dirty. That slot is now in both the PTLAS null-write delta and the full-write delta — order-dependent; if null
  wins, the new instance vanishes. Fix: defer the free_slots recycle by one frame, or dedup disabled ∩ added in the PTLAS seed.

  Tier 2 — "Must stay in lockstep" invariants guarded only by comments (highest quality leverage)

  These are GPU-memory-safety relationships (OOB writes if they drift), currently enforced by twin comments instead of shared code:

  3. max_per (= max_cluster_count().max(1)) computed independently in selector.rs:254, blas_rebuild.rs:161, and a third variant in 
  blas_sharing.rs:331 — all carry "MUST match" comments. Hoist to one InstanceManager::max_clusters_per_bucket() and call it everywhere.
  4. bucket_capacity.min(MAX_BUCKETS) re-clamped at 4 sites (selector ×2, blas_rebuild, blas_sharing) — but blas_sharing already
  pre-clamps at assignment. Expose one accessor returning the clamped value; delete the downstream .min()s + debug_assert!.

  Tier 3 — Drifted naming (real semantic drift, not preference)

  5. The "bucket" vocabulary now lies: blas_sharing.rs:123 live_bucket_count actually holds the dirty build count (its own backing
  buffer is labeled "blas_sharing.dirty_count"); bucket_capacity actually holds resident geometry count. The WGSL side already uses
  dirty_count/geometry_*, so the Rust is the inconsistent outlier. At minimum rename the two whose name inverts their meaning. The
  comments admitting "name kept for source compatibility" are a tell that this was deferred.

  Tier 4 — unsafe / soundness hygiene

  6. SparseBuffer::commit sets the committed-pages bitset before the vkQueueBindSparse it depends on (allocator.rs ~505-582). On a bind
  failure the bitset says "committed" while no memory is bound → a retry skips those pages → GPU fault. Set bits only after Ok.
  7. Mutex-poison .unwrap()/expect() in the commit hot path and in Drop (allocator.rs) — a poisoned lock makes Drop panic-in-drop
  (process abort + leak) for state where poison breaks no invariant. Use unwrap_or_else(PoisonError::into_inner).
  8. Orphaned # Safety doc in extension.rs:307-324 — the partitioned-build fn's safety contract is detached and sits above
  cmd_global_as_barrier; the actual cmd_build_partitioned_acceleration_structures (line 382) is undocumented. Move it.
  9. (Lower confidence) SparseBuffer::Drop may free_memory while the buffer is still bound — VUID-vkFreeMemory-00677; worth a
  validation-layer run. Destroy the buffer handle before freeing the chunks.

  Tier 5 — Duplication → single helper each

  10. #[cfg(feature="gpu_timing")] fence-wait block copy-pasted 4× verbatim (selector, blas_sharing, blas_rebuild, ptlas) + a stale
  orphan comment left above one. → fn gpu_timing_wait(device, span).
  11. Device-address lo/hi split (a & 0xFFFF_FFFF) as u32 / (a >> 32) as u32 open-coded 4×. → fn split_device_address(u64) -> [u32; 2]
  in raw_vulkan.
  12. Align-up arithmetic duplicated (blas_rebuild, clas_arena). → fn align_up(addr, align).
  13. The "empty stream → 0..0" guard (6×) + two dummy-element writes leak PersistentGpuBuffer's "empty input panics / empty buffer is
  an illegal binding" contract to every caller. Push both into the API: queue_write no-ops empty input; binding() guarantees a 1-element
  minimum. Removes the panic surface and the duplication.
  14. Missing GPU-layout const _: () = assert!(size_of...) on PtlasFillParamsGpu and PtlasWritePair — their siblings (SelectorParamsGpu,
  SharingParamsGpu, IndirectCommand) all have them. A field-add silently desyncs WGSL.

  Tier 6 — binder.rs per-frame waste & panics

  15. The material/light/texture Vecs and the bind group are rebuilt from scratch every frame — the exact thing the instance-columns
  refactor existed to avoid (its own header comment says so). Plus it re-resolves the immutable bind-group layout per frame
  (pipeline_cache.get_bind_group_layout = mutex lock + full descriptor clone). Persist the buffers on the resource, .clear() not
  re-default(), dirty-track materials, resolve the layout once at startup.
  16. .unwrap() on light_sources.binding() / directional_lights.binding() (binder.rs:267-269) panics for a scene with geometry but no 
  lights (empty storage buffer → binding() is None) — the cluster_ranges path right above handles None correctly; these don't. Guard or
  always push a sentinel.
  17. Dead let _ = (...) no-op at binder.rs:275 and #[allow(dead_code)] tlas_transform — delete (git keeps them).

  Tier 7 — Minor robustness (compressed)

  - mesh_manager.rs:185 remove_untracked().expect(...) is reachable for unmanaged/raced handles → return Option/re-park instead of
  panicking the render app.
  - Unchecked as u32 narrowing on usize/u64 pool-base + total_triangle_count sums (mesh_manager, instance_manager) →
  debug_assert!/try_from.
  - upload_pending_clas clears the queue silently on the unsupported path → add a one-time warn!.
  - Cross-plugin RenderStartup init order relies on insertion order, not .chain()/.after() (cluster.rs tuple is unordered while scene.rs
  chains) — document the independence or constrain it.
  - @group(0) binding indices 0–18 are coupled to the WGSL by comment only; bind_groups.rs:37 doc even still says "All 16 entries" (it's
  13/19) — already drifted. Consider index-tagged entries or shared consts.
  - material_slots does a full O(n) map sweep every frame regardless of change.

  ---
  The highest-value work is Tier 1 (two real bugs) and Tier 2 (turn the "MUST match" comments into shared functions — that's where a
  future edit silently corrupts GPU memory). Tier 3/4 are correctness-adjacent; Tier 5 is pure cleanup with good ROI. 
  
  Want me to fix a tier? I'd suggest Tier 1 + Tier 2 first (correctness), then the Tier 5 helpers as a quick mechanical sweep.