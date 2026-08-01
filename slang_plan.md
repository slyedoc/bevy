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

## State

| Area | State |
|---|---|
| NRC training (gym + production) | Slang fused kernel; 38→16 dispatches; certification suite in `nrc_gym.rs` (live-validated RTX 5090) |
| `miss_shadow`, `ahit_alpha`, `chit_portal` | Slang `.spv`, WGSL deleted (live-validated RTX 5090) |
| `scene_resolve.slang` | DONE — full shared resolve module (bindings, Material/Cluster mirrors, fetch+full resolve, tess paths, deform, hair/LSS, alpha test, pack/unpack helpers); every mirror offset-diffed vs naga dumps |
| `brdf.slang` / `sampling.slang` / `hair.slang` | DONE — full BRDF stack (GGX VNDF, DFG LUT, glass), light sampling + ReSTIR structs (strides 32/48/64 verified), Chiang fiber BSDF |
| `chit_glass`, `chit_hair`, `chit_opaque` | DONE — Slang `.spv`, WGSL deleted; compile-validated (spirv-val + offset diffs), **awaiting live validation** |
| `raygen` | DONE — Slang, two variants (`raygen.spv` + `raygen_clock.spv`, selected by `shader_clock_available()`); SER via `ReorderThread` (NV flavor, `-capability spvShaderInvocationReorderNV`), inline coopvec NRC query; **awaiting live validation** |
| `SolariChitSource::{Wgsl, SpirV}` | per-stage migration enum on the hit-group registry (zero's registrars migrate on touch) |
| `rt_payload.slang` | shared payload module (+ RtCamera/Mat4 mirror) — see LAW 1; still binding: naga `miss.wgsl` reads the payload |

Port findings (2026-08-01):
- `SOLARI_DLSS` was already unconditionally true in `try_compile_rt_wgsl` (the
  set-1 layout always has the G-buffer bindings), so the ported chits compile
  the guide writes unconditionally — no two-variant DLSS axis exists.
- Slang has `HitTriangleVertexPosition(i)` as a first-class intrinsic (emits
  the builtin + capability); no spirv_asm needed.
- SER: `MaybeReorderThread` is HLSL-target-only in v2026.14.1 — use
  `ReorderThread(hit, hint, bits)`; default emission is the EXT flavor, force
  NV (the device extension) with `-capability spvShaderInvocationReorderNV`.
- Zero CoopVec without a zeros buffer: `HVec.load(any_buffer, 0)` then
  overwrite ALL elements — no `OpCompositeConstructReplicateEXT` emitted.
- Matrix mirrors: `Mat4` (four explicit `float4` columns) + `mat4_mul`
  sidesteps the Slang/HLSL row/column-major mapping entirely; offsets match
  naga's ColMajor stride-16 exactly.
- `GpuHairInstance` needs explicit tail pads to hit naga's stride 64.

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
8. **Passthrough blobs on wgpu pipelines need contiguous-from-0 bindings.**
   wgpu-hal's Vulkan backend numbers descriptor bindings SEQUENTIALLY by
   layout-entry order (`binding: next_binding`), ignoring sparse wgpu binding
   numbers — a blob with sparse `[[vk::binding]]` slots desyncs SILENTLY (no
   validation error; reads garbage, writes nowhere). The raw-VK RT path is
   immune (solari owns those DSLs). Same failure surface as the RT
   binding-contiguity session. Per-kernel entry files with their own 0..N-1
   tables + a bindings-free shared lib (buffers passed as parameters) is the
   working pattern — see nrc_mlp.slang + nrc_*.slang.

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

### 0. LIVE-VALIDATE the 2026-08-01 ports — MOSTLY DONE (same day, RTX 5090)
- Gym: full certification (forward 0.195 err/tol, gradcheck worst 0.545,
  coopvec parity EXACT 0.000) + training 17.22→0.289 in 2000 steps, 754
  steps/s. (First run caught LAW 8 — sparse passthrough bindings silently
  no-op'd every new kernel.)
- solari_many_lights: raygen (SER) + chit_opaque + miss_shadow + NRC
  production chain, 4000+ training steps, zero NaN, no device loss; loss band
  ~5–58 centered ~20 at ~1200/16384 records (old baseline 5–20 — slightly
  noisier, grade with solari_grader if it matters). Debug views verified:
  NormalFacing (all blue), Clusters, NrcCache (live learned field).
- solari_refraction: chit_glass refract/reflect/absorption at 9 bounces, no
  hang (TIR + zero-length-ray guards hold); naga miss.wgsl sky renders — the
  Slang↔naga payload member-wise contract holds.
- solari_hair (BEVY_ASSET_ROOT=/mnt/code/f/bevy for models): wCurly renders
  correctly via chit_hair (Chiang BSDF + Slang→Slang shadow payload pair).
- Bistro (`/mnt/code/p/solari_files` viewer, bevy dep repointed at this
  worktree, `solari_view assets/bistro/bistro.bsn`): full scene renders
  correctly at ~38 FPS debug — foliage cutout silhouettes correct. The
  AnyHitCount heatmap shows ZERO any-hit invocations: this bake resolves
  cutouts via embedded OMMs in hardware (the expected "OMM-covered foliage
  reads cold" signature), so `ahit_alpha`'s cutout path itself still wants a
  non-OMM foliage asset.
Still unvalidated: ahit_alpha on a non-OMM cutout asset, a DLSS-RR-active run
(bistro viewer builds with dlss but RR reported unsupported in this env),
tess smooth/facet branches, portals (pre-session validation stands), ReSTIR
spatial mode.

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

### 7. Cleanup / fork shrink checklist (after 0, 5, 6)
- ~~`nrc_infer_coopvec`/`nrc_query_infer`/`nrc_encode_records` → Slang~~ DONE
  (2026-08-01): ALL of nrc_mlp.wgsl (incl. `mm_tn`/`bias_grad`/`adam`) +
  the gym's `gym_gen` are Slang per-entry blobs, passthrough-loaded with
  explicit layouts (nothing NRC left on the PipelineCache); WGSL deleted.
  Zero `wgpu_cooperative_*` WGSL remains → the fork's coopvec/coopmat feature
  mapping + naga coop code are now DEAD (pending live re-certification via
  the gym + solari_many_lights loss band). Slang coopmat notes: types live in
  `namespace linalg` (`using namespace linalg;`), `coopMatMulAdd<float,
  false>(a, b, acc)` needs the explicit specialization, KHR flavor emitted;
  load/store offsets+strides are in buffer-element units over
  StructuredBuffer<T> (exact naga operand parity — verified against a naga
  dump); coopvec offsets stay BYTES. Gym's `GenParams` moved binding 23→25
  (adam's `ema_master` owns 23 in the shared module).
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
