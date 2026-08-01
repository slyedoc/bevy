# Descriptor-heap migration plan — bevy_solari

Goal: every GPU pass in bevy_solari runs on **raw-VK pipelines + Slang +
`VK_EXT_descriptor_heap`** — no wgpu-driven passes, no descriptor set layouts,
no bind groups. wgpu remains only as the device/queue/swapchain/UI host and
the buffer/texture allocator; the Law-8 / binding-contiguity bug class dies
because there is no layout left to desync. Follow-on to `slang_plan.md`
(which put every shader on Slang; this puts every *dispatch* on solari).

Reference: `/mnt/code/f/vk_mini_samples` — `samples/descriptor_heap` (the
three heap-index modes; we use DirectAccess) and `samples/gltf_raytrace`
(a full Slang ray tracer on heaps, incl. TLAS-from-address).

## The model

- Two heaps per device, bound once per command buffer: **sampler heap** +
  **resource heap**. Each is a plain BDA buffer with
  `VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT`; descriptors are BYTES at
  `index * descriptorSize` (sizes/alignments/reserved-range from
  `VkPhysicalDeviceDescriptorHeapPropertiesEXT`; the reserved range sits
  after the descriptors and belongs to the driver).
- Writes: `vkWriteResourceDescriptorsEXT` / `vkWriteSamplerDescriptorsEXT`
  emit descriptor bytes to any HOST address (host-visible heap directly, or
  staging → copy). Image descriptors take a `VkImageViewCreateInfo` INLINE —
  no `VkImageView` object exists. Buffer descriptors take address+range.
- Bind: `vkCmdBindSamplerHeapEXT` / `vkCmdBindResourceHeapEXT` (device
  address + size + reserved range). Push data via `vkCmdPushDataEXT` —
  NO pipeline layout involved (`VkPushDataInfoEXT` is just offset+bytes).
- Slang: `[require(spvDescriptorHeapEXT)]` on the entry; **no resource
  declarations** — handles are constructed at the use site:
  `Texture2D.Handle(uint2(idx, 0))`, `SamplerState.Handle(uint2(idx, 0))`,
  `StructuredBuffer<T>.Handle(...)`, `RWTexture2D<T>.Handle(...)`,
  `RaytracingAccelerationStructure(device_address)` (TLAS needs NO
  descriptor at all — it rides push data as a u64). Push block stays
  `[[vk::push_constant]]`. Emits SPV_EXT_descriptor_heap via
  SPV_KHR_untyped_pointers.
- Pipelines: `VkPipelineLayout` with ZERO set layouts (compute + RT alike).

## Stack status (verified 2026-08-01)

- Driver 595.84: `VK_EXT_descriptor_heap` + `VK_KHR_shader_untyped_pointers` ✓
- ash master (pinned): `ext::descriptor_heap` fn table
  (write_*_descriptors, cmd_bind_*_heap, cmd_push_data) ✓
- Slang v2026.14.1: `DescriptorHeapEXT` capability + `.Handle()` /
  `ResourceDescriptorHeap[i]` surface ✓
- BDA on all storage buffers: fork commit cfd885f6e (upstream PR branch
  `solari-slang-bda`) ✓

## What dies

| Today | After |
|---|---|
| Law 8 + RT binding-contiguity trap (2 lost sessions) | no layouts exist |
| `[5000]` sized binding arrays (`MAX_TEXTURE_COUNT`) | open-ended heap indices |
| scene/columns bind groups rebuilt per frame + `as_hal` raw-handle dance | descriptor bytes written once per CHANGE (diff-driven, like the columns) |
| `keep_bind_group_alive` teardown keepalive | heaps are buffers solari owns |
| wgpu `PipelineCache`/naga runtime compiles (restir_spatial etc.) | Slang blobs on raw pipelines |
| 3-encoder Law-9 dance in NRC training | one raw encoder, own barriers |
| `Tlas::from_hal` fork interop | TLAS device address in push data |

## Phases

### H0. Foundation + gym beachhead — ATTEMPTED 2026-08-01, PARTIAL (see findings)

**Findings from the first H0 attempt (all reproducible on driver 595.84):**
- slangc flavor trap: without `-capability spvDescriptorHeapEXT` on the CLI,
  `[require(spvDescriptorHeapEXT)]` code silently compiles to the
  RuntimeDescriptorArray FALLBACK (real descriptor arrays, not the heap).
- slangc BUG (v2026.14.1): a typed `.Load<T>`/`.Store<T>` through a
  `ByteAddressBuffer.Handle(...)` emits INVALID SPIR-V (BAB heap pointees
  dedup to one element type; spirv-val catches it). Workaround: every typed
  access via `StructuredBuffer<T>.Handle`; BAB handles only for opaque ops.
- DRIVER WALL: a compute pipeline created with
  `PIPELINE_CREATE_2_DESCRIPTOR_HEAP_EXT` that runs **coopvec ops over heap
  handles HANGS the queue** (fence never signals, GPU idle, no device-lost,
  no validation errors). BAB handles: deterministic hang. Typed
  `StructuredBuffer` handles: hung in most runs (one anomalous success).
  Non-coopvec heap dispatch (`gym_gen`: typed handles, push data, heap
  bind) ran CORRECTLY and repeatedly. The layout-convert cmd also hung when
  recorded after heap dispatches in the same cb.
- KEY UNTESTED LEAD: the NV samples dispatch heap shaders via
  **VK_EXT_shader_object**, NOT compute pipelines. Heap+pipeline may be the
  under-tested driver path; try shader objects next attempt.
- Descriptor stride: the shader-side heap index unit is the PER-TYPE
  descriptor size (5090: buffer 16 B, image 32 B, sampler 32 B) — a mixed
  heap needs per-type regions (the samples' `bufferHeapBase` etc.). Using
  max() strides desyncs indexing → garbage descriptors → hang.
- vkCmdConvertCooperativeVectorMatrixNV VUIDs 10084/10085: src/dst device
  addresses must be 64 B aligned — wgpu-suballocated buffers are NOT
  (0x...120 offsets); §6 had silently relied on driver tolerance. Dedicated
  raw allocations (offset-0 bind) fix it; production nrc should adopt this.
- vkCmdFillBuffer executes in the CLEAR stage, not COPY — barrier srcStage
  accordingly.
- Fork commits on wgpu-slang: BDA usage on storage buffers (cfd885f6e, also
  on the upstream PR branch solari-slang-bda), auto-enable of
  descriptor_heap + untyped_pointers + synchronization2 + maintenance5 +
  push_descriptor when supported (ec603bc30 + follow-up).
- Kernel heap ports (nrc_train/adam/encode/query_infer/infer_coopvec/
  gym_gen with push-data indices + `.Handle()` accessors) compile and
  spirv-val clean — archived in git history for the next attempt; the
  branch was RESTORED to the certified §6 binding/wgpu-dispatch state until
  the dispatch path is stable.

**Verdict (updated after the shader-object exploration, same day):** the
dispatch mode is IRRELEVANT — `examples/3d/solari/heap_probe.rs` (+ the two
kernels beside it) A/Bs shader-object vs pipeline: plain heap access
completes correctly BOTH ways; coopvec-over-heap-handles hangs BOTH ways.
The wall is the driver's coopvec implementation not accepting heap-sourced
(untyped-pointer) buffer operands. The full support matrix on 595.84 +
slang v2026.14.1:

| coopvec operand source | compile | runtime |
|---|---|---|
| classic bindings (BAB/SB) | ✓ | ✓ (production today) |
| classic bindings MAPPED to heap (`--mode mapped`) | ✓ (no heap capability in SPIR-V at all) | ✓ **verified bit-exact** — THE ANSWER |
| heap `ByteAddressBuffer.Handle` | invalid SPIR-V (slang bug) | — |
| heap `StructuredBuffer<T>.Handle` | ✓ | HANG (so + pipeline) |
| BDA pointers | load/store: public `Ptr<T>` overloads; outer-product/reduce-sum: ACCESSIBLE `__coopVecOuterProductAccumulateFromPointer<T,M,N>(Ptr<void>, offset, a, b, layout, type, stride)` / `__coopVecReduceSumAccumulateFromPointer<T,N>(Ptr<void>, offset, v)`; **matMulAdd: buffers-only public API** (the internal impl HAS a `Ptr<T[]>` path — slang exposure gap, worth an upstream issue) | ✓ **verified correct** (`--mode ptr`) |

DRIVER WALL №2 (2026-08-01): on R595 (595.84), merely ENABLING
`VK_EXT_descriptor_heap` (+ feature) at device creation made RAY-TRACING
pipelines execute with quad-scrambled shading (2×2 checkerboard of
misrouted payloads; accumulates to uniform grey; no VVL error; nothing
needs to use a heap). Bisected across the fork ride-along set —
descriptor_heap ALONE was guilty. **FIXED by R610 (610.43.02, retested
same day: heap enabled + tessellation renders clean) — R610 is the
driver FLOOR for the heap; the fork auto-enables it again.** Wall №1
(coopvec × heap HANDLES) is NOT fixed by 610: `--mode so`/`pipeline`
still hang; `--mode mapped` stays bit-exact and `--mode ptr` correct —
DirectAccess remains dead, the mapping modes remain the path. Retest
both walls on every driver bump: this file's probe for the coopvec
wall, solari_tessellation for the RT-scramble wall.

M0 UPDATE (2026-08-01, same day): the mapped row makes the "workable shape"
below OBSOLETE — no static matmul set needed. Shaders keep classic
bindings; the host chains `VkShaderDescriptorSetAndBindingMappingInfoEXT`
(HEAP_WITH_CONSTANT_OFFSET, byte offsets, several bindings may alias one
descriptor) onto the stage create info of a heap-flagged pipeline. The wall
was "coopvec × untyped-pointer SPIR-V", and mapping mode never emits
untyped pointers. See moonshot_plan.md M0 VERDICT. VK_EXT_shader_object
is enabled in the fork hal now (auto-when-supported) and works — the probe
dispatches through it.

Original next-attempt options for reference: (1) ~~shader objects instead of
pipelines~~ tested, not the issue, (2) classic bindings for coopvec matmul +
pointers/heap for the rest ← CURRENT PLAN, (3) retest heap-handle coopvec on
each new driver (the probe is the regression test: `--mode so`/`pipeline`
should flip from HANG to COMPLETED).

#### Original H0 plan
- `SolariInitPlugin`: enable `VK_EXT_descriptor_heap` (+ feature struct) and
  `VK_KHR_shader_untyped_pointers` at device creation.
- `gpu/heap.rs`: `DescriptorHeap` — property query, heap buffer alloc
  (host-visible first; device-local staging later if profiling demands),
  index allocation (free-list), typed write helpers (storage buffer,
  sampled image, storage image, sampler), `cmd_bind(cb)`.
- `gpu/raw_compute.rs`: `RawComputePipeline` — Slang blob → VkShaderModule →
  compute pipeline on an empty layout; dispatch helper recording into a raw
  encoder (raw_trace.rs pattern) with sync2 barriers + `cmd_push_data`.
- **Beachhead = the NRC gym**, because it certifies: port every gym kernel
  (gen, learn, adam, infer) to heap handles + push data + raw dispatch.
  RETIRES THE TOP RISK: do coopvec/coopmat ops accept heap-handle buffers?
  (If not: fallback is BDA pointers for coopvec operands — offsets are
  already explicit — and heap for everything else.)
- Gate: full gym certification (forward 0.195, gradcheck 0.545, parity
  0.000, ≥2900 steps/s).

### H1. Production NRC chain → raw dispatch + heap
encode/learn/adam/query_infer off wgpu dispatch; one raw encoder for the
whole training step (clears via vkCmdFillBuffer, converts inline — the
encoder-split and the Law-9 tracked-write workaround both evaporate).
Gate: tessellation/many_lights loss parity.

### H2. RT pipeline → heap
- Set 1 (solari-private per-view) → heap indices in push data; delete the
  per-view descriptor pool/sets.
- Scene set 0: textures/samplers → heap indices carried in the Material
  table (the CPU column already knows them); TLAS → device address.
- Columns set 2 → heap (or BDA — the columns are all storage buffers).
- Kills: `RtViewBindings` descriptor machinery, scene bind group `as_hal`,
  `Tlas::from_hal`. SBT untouched.

### H3. The long tail: every remaining wgpu pass → Slang + raw dispatch
Groups, roughly in order of coupling:
1. restir_spatial + rt_camera + blit + gizmo_depth + dlss_resolve (the
   per-frame render chain around the trace)
2. light_resolve + atmosphere_bake/lut (lights/sky)
3. transforms ×5 + scatter/reconcile (ecs_gpu)
4. tess chain + accel chain (selector/blas_sharing/instantiate/ptlas_fill/
   deform/hair_write — already raw-adjacent, share encoders with AS builds)
5. ray_query service, instance_mask, view_cull
Each port: WGSL → Slang (Law-2 discipline vs a naga dump), heap/BDA
bindings, raw dispatch, live validation. Delete the WGSL + its
`load_shader_library`/`embedded_asset` as it goes.

### H4. Cleanup
- Delete `pipelines.rs` embedded-asset registry, bindings/binder bind-group
  builders, naga_oil from bevy_solari entirely (test composer included).
- Fork shrink: `keep_bind_group_alive`, BindGroup/BGL `as_hal`,
  `Tlas::from_hal` become dead → drop from the fork.
- Revisit: DLSS + UI + swapchain remain the only wgpu consumers (phase-E
  decision point).

## Laws (inherited + new)

- Slang laws 1–9 from `slang_plan.md` still bind where applicable; Law 8/9
  stop applying to heap-based kernels (no wgpu layouts, no wgpu tracking).
- **H-LAW 1: wgpu must never see a heap kernel.** Heap binding is raw
  cmd-buffer state; raw and wgpu commands cannot share an encoder
  (wgpu-core panics). Every heap dispatch lives in a raw encoder with its
  own sync2 barriers.
- **H-LAW 2: descriptors for wgpu-owned resources are STILL wgpu-lifetime.**
  Writing a heap descriptor for a wgpu texture/buffer does not keep it
  alive; the heap index must be dropped/rewritten when the resource is —
  route every such write through the diff-driven reconcile that already
  tracks the resource.
- (expected) Aim heap writes at host-visible memory first — the NV sample's
  device-local staging path is an optimization, not a requirement.

## Session ritual

Same as slang_plan.md: after every ported pass — spirv-val, offset-diff
where a CPU-written struct is involved, `cargo test -p bevy_solari`,
live-run the exercising example, screenshot-verify. One pass at a time;
the branch stays shippable between passes.
