# Moonshot plan — RT-native bindless solari (Slang + descriptor heap + SBT)

The operative plan for the next sessions. Successor to `heap_plan.md` (whose
H0 findings + support matrix REMAIN BINDING — read that section first) and
`slang_plan.md` (complete; its laws still apply). Scope: **RT is the
product** — every design choice optimizes the ray-tracing path; compute
exists to feed it.

## The endgame in one paragraph

One RT pipeline built by LINKING pipeline libraries (raygen, misses, one per
hit group — `VK_KHR_pipeline_library` + deferred host ops; registering a
downstream material = compile one library + relink, never a full rebuild).
No descriptor sets except one static coopvec-matmul set. The scene is: TLAS
as a device address in push data, geometry/attributes via BDA (done since
slang_plan), textures/samplers via the descriptor heap, and **each
material's parameters living in its SBT record** — the SBT is RT's dynamic
binding table, GPU-writable by the diff-driven column pipeline. Shaders keep
plain `[[vk::binding]]` declarations; the heap wiring happens HOST-SIDE via
the mapping API, so shader source never re-targets again.

## The seam (build this FIRST — one Rust module, e.g. `gpu/binding_seam.rs`)

Three functions; everything in solari (column pipeline, material upload, SBT
writer, view bindings) talks ONLY to these:

- `alloc_heap_index(resource) -> u32` — writes the descriptor bytes, returns
  the heap slot (free-list; per-TYPE regions — buffer/image/sampler strides
  differ: 16/32/32 B on the 5090).
- `device_address(buffer) -> u64` — BDA of any solari/wgpu buffer (fork
  gives all storage buffers `SHADER_DEVICE_ADDRESS`).
- `write_record(slot, fields)` — one SBT record: hit-group handle + the
  material's pointers/heap indices/constants.

Behind the seam: `VkShaderDescriptorSetAndBindingMappingInfoEXT`
construction, the `DescriptorMappingSourceEXT` union arms
(HEAP_WITH_PUSH_INDEX / HEAP_WITH_CONSTANT_OFFSET /
HEAP_WITH_SHADER_RECORD_INDEX / SHADER_RECORD_DATA / SHADER_RECORD_ADDRESS),
resource-type masks, heap alloc/bind, reserved ranges. When the KHR version
lands (Vulkan Roadmap 2026 direction), ONE file changes.

## M0 VERDICT (2026-08-01): mapping modes DODGE the coopvec wall — CONFIRMED

`heap_probe --mode mapped` (heap_probe_mapped.slang: identical coopvec math,
classic `[[vk::binding]]` declarations, NO heap capability; host maps set 0
→ heap via HEAP_WITH_CONSTANT_OFFSET on the stage create info): **COMPLETED,
bit-exact** — while the same math via `.Handle()` hangs both dispatch modes.
The H0 wall was never "coopvec × heap"; it was "coopvec × untyped-pointer
SPIR-V". Consequences, now binding:
- DirectAccess (`.Handle()`) is DEAD to us. All heap access goes through
  host-side mappings; .slang files keep plain bindings, zero rewrites.
- The H0 fallback (static matmul set) is unnecessary — M2's coopvec caveat
  is void; raygen's inline NRC query maps like everything else.
- Extra facts proven by the probe: mapping offsets are BYTES (per-type
  descriptor sizes stay host-side — the seam's job); several bindings may
  alias ONE heap descriptor (float/half/BAB views of one buffer); a classic
  `[[vk::push_constant]]` block is fed by `vkCmdPushDataEXT` unmodified;
  correctness verified to the level of an FTZ'd denormal in the aliased
  accumulate region.

## Phases (each independently shippable; session ritual per slang_plan.md)

### M0. Seam + mapping probe — DONE (2026-08-01)
- `heap_probe --mode mapped` — verdict above (M2 takes the
  zero-shader-edit branch everywhere).
- `gpu/binding_seam.rs` EXISTS: `BindingSeam` resource with
  alloc_heap_index / rewrite_heap_index / free_heap_index,
  device_address, write_record (+ record_region for trace), map_binding
  (the HEAP_WITH_CONSTANT_OFFSET arm), bind_heaps, push_data. Per-type
  regions (buffer 4096 / image 2048 / sampler 256 slots); records =
  4096 × (handle + 96 B fields). Host-visible; consumers arrive in M2.
- Keep the existing probe modes as the driver/slang regression net (retest
  on every driver or Slang bump; `--mode so`/`pipeline` flipping to
  COMPLETED = heap handles fixed upstream).

### M1. RT pipeline libraries — CODE DONE (2026-08-01), awaiting RTX run
- `RtLibraryCache` (gpu/rt_pipeline.rs): render-world resource caching one
  `VK_KHR_pipeline_library` per stage — raygen, primary miss (keyed by
  custom-sky generation), shadow miss, one per registry hit group
  (append-only ⇒ only new entries compile). Owns the shared set-1 DSL +
  pipeline layout; recreated when the wgpu raw layouts change
  (`layout_key`). `RtPipeline::new(allocator, &mut cache, ...)` =
  ensure_* + `link()`; linked group order raygen(0)/miss(1)/hits(2..)/
  shadow-last keeps the SBT bake byte-identical. Sky swap now recompiles
  ONLY the miss library; class churn/SBT growth is link-only. Fork hal
  auto-enables `VK_KHR_pipeline_library`. Interface: payload 160 B,
  attributes 8 B, shared by every library + link.
- Deferred host ops for parallel library compiles: SKIPPED — each library
  compiles once per app run now; the recurring cost is gone.
- Registry `register()` keeps its eager slangc validation; the vk library
  compiles on first pipeline build (register has no device access).
- Gate (USER): tessellation + many_lights visually identical, no VVL
  errors; sky hot-swap noticeably faster (link-only).
- VALIDATED (2026-08-01, tessellation + many_lights render clean through
  the library link). The validation cost a full debugging session whose
  REAL root cause was NOT M1: see the descriptor-heap driver wall below.
  Two keepers from the hunt: (1) the linked pipeline sets an explicit
  stack size (`VK_DYNAMIC_STATE_RAY_TRACING_PIPELINE_STACK_SIZE_KHR` +
  per-group `vkGetRayTracingShaderGroupStackSizeKHR`, formula in
  `RtPipeline::new`) — precautionary, spec-recommended for linked
  pipelines, not the bug's fix; (2) debugging traps: a `map_async`
  BEFORE the frame's submit makes wgpu drop the whole submit ("buffer
  still mapped") and readbacks fabricate zeros — defer the map one
  frame; and a forced debug view left in an example poisons every
  visual observation after it.

## DRIVER WALL №2 (2026-08-01): VK_EXT_descriptor_heap × ray tracing —
## FIXED BY R610; R610 IS THE HEAP DRIVER FLOOR

On R595 (595.84), merely ENABLING `VK_EXT_descriptor_heap` (+ its
feature struct) at device creation made the RT pipeline execute with
quad-scrambled shading — a 2×2 checkerboard of misrouted payloads that
accumulation averages into uniform grey. No validation error, no
device-lost, nothing needs to USE a heap. Bisected across the fork's
ride-along set: descriptor_heap alone guilty; untyped_pointers / sync2 /
maintenance5 / push_descriptor / pipeline_library innocent.

**R610 retest (610.43.02, same day): FIXED.** Heap enabled +
tessellation renders clean; the fork auto-enables descriptor_heap again
with the R610 floor documented in adapter.rs. Probe matrix on 610:
`--mode so`/`pipeline` (coopvec over heap HANDLES) STILL HANG — wall №1
survives, DirectAccess stays dead; `--mode mapped` bit-exact and
`--mode ptr` correct — the seam's mapping-mode strategy is confirmed on
both driver series. **M2's heap half is UNBLOCKED.** Standing nuance
from the NVIDIA forum (610.74, July 2026): shader-side heap access to
ACCELERATION STRUCTURES device-losts even on 610 while
HEAP_WITH_CONSTANT_OFFSET mapping of the same AS works — M2's TLAS
address must come from PUSH DATA, never a heap load.

Driver intel (researched 2026-08-01):
- **R610 (610.43.02, May 2026; in Pop noble-updates now) is the first
  driver with FULL VK_EXT_descriptor_heap support** (plus Nsight 2026.2
  capture/replay). R595's heap support was explicitly still taking
  "various fixes" through its beta line (595.44.06) — our 595.84 predates
  them; the scramble wall is plausibly in that churn. Upgrade + retest:
  (1) re-add descriptor_heap to the fork ride-alongs, run
  solari_tessellation → is the quad-scramble gone? (2) heap_probe
  --mode so/pipeline → coopvec-over-handles wall? (3) --mode mapped
  still bit-exact?
- **Independent validation of the seam strategy** (NVIDIA forum, July
  2026, driver 610.74): shader-side heap access to ACCELERATION
  STRUCTURES (Slang u64 heap load + OpConvertUToAccelerationStructureKHR,
  and GLSL untyped-pointer loads) device-losts even on 610 — while the
  SAME heap-resident AS through HEAP_WITH_CONSTANT_OFFSET **mapping**
  works. DirectAccess is broken for RT on NVIDIA across TWO driver
  series; the mapping modes are the reliable path. Nuance for M2's
  "TLAS by address": the failing repro loads the u64 FROM THE HEAP —
  push-data-supplied addresses are a different path, verify separately.
- R610 also ships `VK_KHR_internally_synchronized_queues` (could retire
  the as_hal_locked queue-external-sync discipline) and
  `VK_NV_push_constant_bank` (interesting for push-data-heavy binding).

### M2. RT goes heap + record-bindless (through the seam)
- **STAGING COMPLETE (2026-08-01): every RT-visible resource is mirrored
  into the heap, unread, classic path still driving.** Set 1 per view
  (`RtViewHeapSlots`: 15 buffers + env cube image + env sampler), scene set
  0 (`SceneHeapSlots`: 8 storage buffers + dense image blocks for
  `textures[]`/`texture_arrays[]` + parallel sampler block + DFG LUT pair +
  array sampler), columns set 2 (`SceneColumns::heap_slots`, signature
  cadence). Fork additions that made it verbatim: hal `TextureView` records
  its `ImageViewCreateInfo` (+`image_view_create_info()`), hal
  `Sampler::create_info()`, and `Sampler::as_hal` (new, all three layers).
- **Hardware datum**: NVIDIA sampler heap max = 128 KB ⇒ 4096 sampler
  descriptors (4080 usable). The `samplers[5000]` parallel array can never
  mirror 1:1 — block clamps to capacity (4016) with a loud assert. Classic
  path has the same wall disguised (`maxSamplerAllocationCount` 4000). At
  flip, shrink the WGSL sampler array bound or decouple sampler ids from
  texture ids (records). Resource heap max 32 MB — buffers/images roomy.
- **M2c FLIP DONE (2026-08-02, first run clean — no VVL, no device loss,
  NRC converging on real hit data).** Libraries + link carry
  `PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT` (flags moved to flags2), NO
  pipeline layout, NO set-1 DSL; every stage chains one shared mapping
  table (`build_heap_mappings`): scene set 0 + columns set 2 at constant
  heap offsets, all 17 set-1 bindings `HEAP_WITH_PUSH_INDEX` (slot indices
  from push data — one pipeline serves every view, resize/skybox rebuilds
  don't relink), TLAS as `PUSH_ADDRESS`. Trace = `bind_heaps` + one 76-B
  `push_data` blob + bind + trace rays. DELETED: per-view descriptor
  pool/set/env-sampler objects, `cmd_bind_descriptor_sets`, camera dynamic
  offset, `raw_bgl`/`raw_set`, both `keep_bind_group_alive` calls, the
  layout_key cache invalidation. Sampler mappings use the `sampler_*`
  union fields (separate sampler heap bind point). R610+ is REQUIRED:
  seam-absent now disables solari wholesale (no classic fallback path).
  The wgpu scene/columns bind groups still exist — the ReSTIR spatial
  wgpu pass consumes them (they leave the RT trace's world, not the
  renderer).
- Materials: `[[vk::shader_record]]` manual unpack in chit_opaque/chit_glass
  is volatility in shader source — replace with mapped record sources
  (SHADER_RECORD_DATA / HEAP_WITH_SHADER_RECORD_INDEX) via `write_record`.
- Post-flip: texture/sampler heap writes move off the per-frame mirror onto
  the diff-driven upload path (write descriptors at register/evict, material
  records carry heap indices; steady-state frames write zero descriptors).
- Materials: `[[vk::shader_record]]` manual unpack in chit_opaque/chit_glass
  is volatility in shader source — replace with mapped record sources
  (SHADER_RECORD_DATA / HEAP_WITH_SHADER_RECORD_INDEX) via `write_record`.
- Columns set 2: plain storage → BDA pointers in push data.
- Coopvec caveat RESOLVED by M0: raygen's inline NRC query maps like every
  other binding; no static matmul set needed. (BDA pointer forms remain
  verified via `--mode ptr` if a bufferless spot wants them.)
- Gate: full example sweep (tessellation, many_lights, refraction, hair,
  bistro viewer) + no VVL errors.

### M3. NRC + compute follow the seam
- H1 from heap_plan, reshaped: training chain on raw dispatch, matmul per
  M0's verdict, everything else BDA/heap; one raw encoder, fills + converts
  inline (Law 9 stops applying); dedicated aligned buffers for the convert
  (VUID 10084/10085 — stop relying on driver tolerance).
- Gym certifies (0.545 gradcheck / ≥2900 steps/s baseline).

### M4. Slang module linking (replaces the blob workflow)
- Precompiled `.slang-module` IR per stage; libslang links + specializes at
  pipeline-library build (link-time constants replace the
  raygen/raygen_clock two-blob axis and any future variant axis).
- Regen commands shrink to "slangc -o *.slang-module"; `rt_shaders_compile`
  links everything headlessly.

### M5. Long tail + teardown
- Remaining wgpu passes → Slang + raw dispatch through the seam (order per
  heap_plan H3); delete naga_oil from bevy_solari entirely.
- Fork shrink round 2: `keep_bind_group_alive`, BindGroup/BGL `as_hal`,
  `Tlas::from_hal` go dead → drop.

## Upstream actions (file early, they gate the endgame)

- **Slang**: expose a public `Ptr` overload for `coopVecMatMulAdd` (the
  internal `Ptr<T[]>` path exists; load/store + both training accumulates
  already have accessible pointer forms). Also report the
  `ByteAddressBuffer.Handle` typed-access miscompile.
- **wgpu**: `solari-slang-bda` PR (storage-buffer SHADER_DEVICE_ADDRESS) is
  ready to open.
- Watch: KHR descriptor heap (Roadmap 2026) — adoption = seam-only change;
  NVIDIA "End-to-End Descriptor Heaps" post tracks driver maturity.

## Avoid (scar tissue, do not relearn)

- VK_EXT_shader_object: NO ray-tracing stages, by design. The SBT is RT's
  shader object; pipeline libraries are RT's fast-relink story.
- `.Handle()` (DirectAccess) in any kernel that touches coopvec; typed
  access through `ByteAddressBuffer.Handle` anywhere (slangc miscompile).
- Mixed-type heap regions without per-type strides/bases.
- Unaligned convert addresses; `vkCmdFillBuffer` is CLEAR-stage.
- Heap kernels under wgpu-tracked dispatch (H-LAW 1/2 in heap_plan.md);
  wgpu lazy zero-init wipes untracked writes (slang_plan LAW 9).
