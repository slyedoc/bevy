# Slang migration plan — bevy_solari

Goal: bevy_solari's entire GPU surface authored in Slang, compiled to SPIR-V,
consumed by the raw-VK pipeline layer — retiring the naga fork (~3,900 lines),
naga_oil, and eventually most of the wgpu fork. Slang ships every naga feature
we hand-built (SER/HitObject, cluster_id, position fetch, BDA pointers, coopvec
incl. the training ops, shader clock) and `spirv_asm` + `[[vk::ext_*]]` end the
fork-per-NV-extension treadmill.

## Toolchain

- Slang **v2026.14.1** pinned at `/mnt/code/f/slang` (`bin/slangc`). Weekly
  upstream releases; bump deliberately, re-run every certification below.
- Compile pattern: **precompiled `.spv` checked in next to its `.slang`**, regen
  command in each file header, `include_bytes!` + passthrough/raw module create.
  `spirv-val` after every regen. Build-system integration is deliberately
  deferred until the libslang step (below).
- Device features: everything arrives via `SolariInitPlugin`'s raw_vulkan_init
  callback + `SolariPlugin::required_wgpu_features` (incl. `PASSTHROUGH_SHADERS`).
  This IS the device-ownership story — do **not** build a `device_from_raw` path.

## State (all live-validated on RTX 5090)

| Area | State |
|---|---|
| NRC training (gym + production) | Slang fused kernel; 38→16 dispatches; certification suite in `nrc_gym.rs` |
| `miss_shadow`, `ahit_alpha`, `chit_portal` | Slang `.spv`, WGSL deleted; mixed per-stage with naga stages in one RT pipeline |
| `SolariChitSource::{Wgsl, SpirV}` | per-stage migration enum on the hit-group registry (zero's registrars migrate on touch) |
| `rt_payload.slang` | the shared payload module — see LAW 1 |

## The laws (each one cost a debugging session — do not relearn them)

1. **Payloads cross the naga/Slang boundary MEMBER-WISE.** The Slang struct
   must match the WGSL declaration's member shape exactly — same count, types
   (`float3` stays `float3`), order. A byte-identical but reshaped struct
   (scalarized/padded) silently drops every write on SER-executed chits: black
   output, zero errors. `[[vk::offset]]` is silently ignored on payload storage.
2. **Verify every buffer-struct mirror against a naga dump.** Dump the WGSL
   stage's SPIR-V (scratch test → `try_compile_rt_wgsl` → file), diff
   `OpMemberDecorate ... Offset` + `ArrayStride` against slangc's output.
   Known-good: Cluster stride 48 (explicit `pad_a/pad_b` in Slang), Material
   `base_color_texture_id`@4 / `alpha_mask`@76, SolariGeometryAddresses std140
   u64s @0,8,16 + u32@24 then 32,40,48,56,64, Affine stride 48, Portal stride 16.
3. **Never splat-construct a CoopVec.** `CoopVec(x)` and zero-init lower to
   `OpCompositeConstructReplicateEXT` → needs `VK_EXT_shader_replicated_composites`
   the device doesn't enable. Load zeros from a zeros buffer (`HVec.load(zeros, 0)`).
4. **f16 A/B are the only working coopvec matmul configs.** The (f32 vector ×
   f16 matrix) config SEGFAULTS the driver's pipeline compiler at
   vkCreateComputePipelines — exit 139, no validation error. Backward-chain
   precision comes from scalar f32 loops (warp-uniform weight reads), not configs.
5. **slangc names every entry point `main`** regardless of the function name.
6. Bindless arrays are sized (`[5000]` = `MAX_TEXTURE_COUNT`) with
   `NonUniformResourceIndex`; BDA is `*(T*)(u64_addr)`; `cluster_id()` via
   `spirv_asm { OpCapability RayTracingClusterAccelerationStructureNV; ... 
   OpLoad builtin(ClusterIDNV:uint) }`; `InstanceIndex()`→InstanceId,
   `PrimitiveIndex()`, `IgnoreHit()`, `[vk::shader_record]` for SBT records.
7. IDifferentiable impls need `[Differentiable]` on dadd/dmul/dzero with
   `no_diff` bodies; coopvec buffer operands cast `(void*)`; autodiff replays
   the primal (expect ~1.5× the theoretical matmul count).

## Debug workflow (proven)

- Screenshot loop without user eyes: run example with `env -u WAYLAND_DISPLAY`
  (forces XWayland), find window via `xwininfo -root -tree`, capture with
  `import -window $WID out.png`, Read the png. Watch window tiling: a portrait
  window narrows horizontal FOV — reframe the camera temporarily if the target
  is off-screen.
- Black-output triage: color-code the stage's branches (blue/red discriminators);
  if colors don't appear, writes aren't landing (payload law) — confirm routing
  separately with a temp naga-compiled color-emitter chit.

## Remaining work, in order

### 1. `scene_resolve.slang` — the shared module (gates all chit ports)
Port `resolve_triangle_data_full_mat_fetch` + callees from
`raytracing_scene_bindings.wgsl`: PackedVertex decode (16-B record:
normal@0 tangent@4 uv@8), position fetch, TessCluster path, deform-pool +
animated-table lookups, previous-frame transforms, full Material mirror +
`ResolvedMaterial` texture resolve (ray-cone LOD via `texel_lod_bias`).
Every BDA record gets the Law-2 offset diff. Seed exists in `ahit_alpha.slang`
(bindings + partial Material); grow it into the module and make ahit import it.

### 2. `chit_glass` (183 lines)
Needs scene_resolve + `sample_glass_bsdf` (+ `fresnel_dielectric`) from
`brdf.wgsl` + the position-fetch builtin (`HitTriangleVertexPositionsKHR` via
`[[vk::ext_builtin_input]]` or spirv_asm) + the **SOLARI_DLSS axis**: two
checked-in variants (`slangc -D SOLARI_DLSS`), `#[cfg(feature = "dlss")]`
selecting the `include_bytes!`. Port the TIR guard and the zero-length-ray
guard exactly — both are GPU-hang protections.

### 3. `chit_hair` (255) then `chit_opaque` (573)
Hair adds LSS intersection attrs + hair pools. Opaque adds NEE + shadow-ray
trace (`TraceRay` from a chit — first Slang→Slang payload pair), ReSTIR
reservoir writes (`bevy_solari::sampling` struct mirrors — Law 2), DLSS axis
again. After opaque, delete `SolariChitSource::Wgsl`? NO — downstream crates
(zero) still register WGSL chits; keep the enum.

### 4. `raygen` (1,122) — the boss
SER (`HitObject`/`MaybeReorderThread` are first-class Slang), the bounce loop,
ReSTIR DI/GI, NRC inline coopvec query + record writes, DLSS guides, debug
views, shader clock (`getRealtimeClock`, gate stays a spec constant or two
variants). Port LAST — every contract it touches will already be exercised by
the chit ports. Payload pairs become Slang↔Slang; the member-shape law then
relaxes to ordinary care.

### 5. `miss` + custom sky — the libslang step
`miss.wgsl` composes the runtime-swappable `custom_sky` module → needs
runtime compilation. Bindgen the minimal slang C API (global session, load
module from source string, compose, entry-point code) — ~5 entry points,
pinned to /mnt/code/f/slang. The slang-rs crate is not production-grade
(0.1.0, FFI bugs); own the surface. This also unlocks deleting the regen
commands in favor of build.rs compilation if desired.
`SolariSky::Shader` user modules become Slang modules implementing an `ISky`
interface — nicer than naga_oil text splicing.

### 6. NRC endgame: TrainingOptimal outer-product
When the NRC dispatch path owns raw command encoding: swap the dz-store in
`nrc_train.slang`'s `evalBwd` for `coopVecOuterProductAccumulate` /
`coopVecReduceSumAccumulate` (TrainingOptimal layout), add
`vkConvertCooperativeVectorMatrixNV` (device cmd) to convert grads for adam,
delete `mm_tn`/`bias_grad` and the acts/dz buffers → 16→~4 dispatches.
The gym must grow the convert step to keep certifying.

### 7. Cleanup / fork shrink checklist (after 1–6)
- `nrc_infer_coopvec`/`nrc_query_infer`/`nrc_encode_records` (WGSL coopvec) →
  Slang, deleting the last `wgpu_cooperative_*` WGSL → fork's coopvec feature
  mapping + naga coopvec/coopmat code become dead.
- `restir_spatial`/`rt_camera`/`gizmo_depth`/`blit` + transform/tess/accel
  compute: **leave on WGSL** (stock-naga compatible; no fork features). Migrate
  only if/when phase-E (own swapchain) lands.
- Then: rebase the wgpu fork dropping naga RT/SER/coopvec commits; the
  remaining fork = hal extension plumbing + queue-sync micro-surface.
- `rt_shaders_compile` keeps covering both: WGSL composes, `.spv` blobs get
  magic/alignment checks; consider asserting Law-2 offsets mechanically.

## Session ritual

Start: read this file + the `solari-slang-rawvk-direction` memory. After every
ported stage: `spirv-val`, offset-diff vs the naga dump, `cargo test -p
bevy_solari`, live-run the example that exercises the stage, screenshot-verify.
One stage at a time; the branch must stay shippable between stages.
