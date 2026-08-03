# Slang integration: current state + the RT debug-info attribution problem

Written 2026-08-02 at the end of the slang-modernization arc; updated
2026-08-03 at the completion of the WGSL→Slang COMPUTE migration: **every
solari compute dispatch is a layout-free Slang heap kernel** — zero wgpu
compute pipelines, zero naga_oil, the `naga`/`naga_oil` dependencies are
gone from the crate. The shared pattern is `gpu/heap_kernel.rs`
(`HeapKernel` + `KernelSlots`: params ride push data as `[params | slot
array]`, slot arrays assembled by reflected parameter name intersected with
the SPIR-V's surviving bindings, descriptor kinds classified from the
module's type graph). Multi-set kernels (restir_spatial, ray_query, the
cluster passes, tess gen_verts with its 256-texture displacement array) use
`new_with_mappings` over the mirrored scene/cluster/columns heap surfaces;
the TLAS is always push-address-sourced. The only WGSL left is
`gizmo_depth.wgsl` (the sole raster pass — kept wgpu per
`docs/wgpu_inventory.md`, the drop-wgpu decision document). Encoder rule
that cost a runtime panic: the fork forbids mixing wgpu passes with raw
`as_hal_mut` on one encoder — raw segments get their own encoders, spliced
via `add_command_buffer` (RenderContext is per-system).

## Current state

Every solari GPU stage compiles from `include_str!` Slang source at pipeline
build. There are no precompiled `.spv` blobs anywhere.

- **Compiler** (`src/gpu/slang.rs`): the modern COM API, dlopen'd from the
  pinned toolchain (`$SLANG_DIR/lib/libslang.so`, default
  `/mnt/code/f/slang`). Per compile: `IGlobalSession` → `ISession` (targets,
  preprocessor defines, target capabilities in the `SessionDesc`/
  `TargetDesc`) → `loadModuleFromSourceString` for every module (imports
  resolve in-session, fixpoint order, no files touched) →
  `createCompositeComponentType` → `link` → `getEntryPointCode` +
  `getLayout`. Vtable slots are transcribed from the pinned `slang.h` —
  note `IModule`/`IEntryPoint` methods start at slot 17 (they extend
  `IComponentType`, whose 14 methods occupy 3..=16). Reflection uses the
  plain-C `spReflection_*` symbols (not deprecated) on `getLayout`'s
  `ProgramLayout`; `SlangSession` is a typedef of `IGlobalSession`, so the
  C helpers take the COM pointer directly.
- **Stages**: the entry stage comes from the `[shader("...")]` attribute
  (there is no stage parameter). Entries keep their REAL names in
  `OpEntryPoint` (`CompilerOptionName::VulkanUseEntryPointName = 52`;
  `renameEntryPoint` does NOT affect the emitted name — dead end), and
  pipeline stages pass the entry name as `pName`. Asserted in
  `rt_shaders_compile`.
- **Capabilities**: passed as target compiler options
  (`CompilerOptionName::Capability`, id via `spFindCapability`). Raygen pins
  `spvShaderInvocationReorderNV` + `spvCooperativeVectorNV` +
  `spvShaderClockKHR` — declaring any atom makes the set the whole target
  profile, so it must cover everything the entry uses. TRAP: without the
  SER atom, slang free-chooses the **EXT** SER flavor
  (`SPV_EXT_shader_invocation_reorder`), which the device does not enable
  (VUID 08740, device-lost); in-source `[require(...)]` has the same
  problem AND turns on strict checking. The test asserts the EXT flavor
  never appears.
- **Debug info**: every compile carries
  `CompilerOptionName::DebugInformation = MAXIMAL` (44 → 3) +
  `DebugInfoIncludeSource = 157` on the target →
  `NonSemantic.Shader.DebugInfo.100` with full line tables and the `.slang`
  source text embedded in the SPIR-V. Verified present (a cached raygen
  carries ~3.9k `DebugLine`s + `DebugSource` with source). NonSemantic
  instructions are stripped by the driver's final compile — no runtime
  cost, only SPIR-V bytes (~2.5× size).
- **Disk cache**: `~/.cache/bevy_solari/slang/` — key = compiler build tag
  (`spGetBuildTagString`) + entry source + all module sources + defines +
  capabilities; the full key is stored in the file and byte-compared on
  load (filename FNV hash is only a lookup hint), and a fixed-options
  fingerprint (`CACHE_OPTIONS_TAG`) rides in the key so compiler-option
  changes invalidate too. Warm startups compile nothing. Format magic
  `SLN3`.
- **Hot reload**: the `.slang` sources are `embedded_asset!`s
  (`src/gpu/slang_sources.rs`); asset events extract into the render-world
  `SlangSources` (the `PipelineCache` pattern); a generation bump drains
  and invalidates the RT library cache. Live editing = bevy_asset's
  `embedded_watcher` feature (solari_files enables it).
- **Reflection-driven dispatch**: `compile_rt_slang` returns
  `CompiledShader { spirv, bindings }`; heap compute mapping tables are
  derived from the module's own SPIR-V decorations
  (`spirv_descriptor_bindings`), and the NRC dispatch slot arrays are
  assembled by parameter *name* against the reflected layout
  (`NrcKernel::push_slots`) — nothing about kernel bindings is
  hand-maintained.
- **RT pipeline shape**: compiled stages (shader modules + entry names)
  cached in `RtShaderCache` (raygen / miss keyed by sky generation /
  shadow / per-hit-group), assembled into ONE MONOLITHIC
  `vkCreateRayTracingPipelinesKHR`. Layout-free (`DESCRIPTOR_HEAP_EXT`
  flags2), stages chained with the shared descriptor-mapping table.
  Monolithic on purpose: the `VK_KHR_pipeline_library` link shape loses all
  Nsight shader-profiler attribution (the driver/tool gap below), so the
  ~1.25s create is paid on every rebuild axis (sky swap, new hit group,
  source edit) in exchange for per-line RT profiling. Switch back to
  libraries+link (~7.5ms, shape in git history before the monolithic flip)
  once the gap is fixed.

## The open problem: RT stages are Unattributed in Nsight

With full debug info in every module, the Nsight Graphics GPU Trace shader
profiler attributes some shaders and not others. The evidence matrix
(bevy_city captures, R610 / 610.43.02, RTX 5090):

| Pipeline shape                                | Attribution |
| --------------------------------------------- | ----------- |
| Heap compute (NRC kernels, layout-free flags2) | ✅ full — `Layer.evalBwd nrc_train.slang:161`, per-line stalls |
| RT **monolithic** (all stages, one create)     | ✅ full — raygen/chits/misses, even cross-module inlines (`ggx_vndf_sample_invalid brdf.slang:174`) |
| RT **libraries + link** (the shipping shape)   | ❌ nothing — all RT samples land in `[Unattributed]` (~66% of the frame) |

Identical SPIR-V in every case. The correlation is lost specifically across
the `VK_KHR_pipeline_library` link.

### Ruled out (each verified by a bevy_city capture)

- Missing SPIR-V debug info (present + verified; compute attributes).
- Nsight version: reproduced identically on Nsight Graphics **2026.3**.
- `VkDeviceDiagnosticsConfigCreateInfoNV` with `ENABLE_SHADER_DEBUG_INFO`
  (+resource tracking/checkpoints/error reporting) at device creation — the
  fork wires it behind `WGPU_AFTERMATH=1`. No change.
- `VK_KHR_pipeline_executable_properties` +
  `CAPTURE_INTERNAL_REPRESENTATIONS` on libraries and link. No change.
- **Distinct entry-point names.** Shipped permanently (see Current state):
  `OpEntryPoint` carries raygen / miss_primary / chit_opaque / … with
  matching stage `pName`s, so the linked pipeline no longer aggregates
  five libraries all exporting `main`. Kept for its own sake — but
  attribution did NOT change. The collision theory is dead.
- Keeping monolithic permanently: a monolithic create costs **1.25s** (cold
  AND warm — the driver reuses nothing in-process) vs **7.5ms** for the
  library link. Every sky swap / hot reload / material rebuild would hitch.

### Research conclusions (2026-08-02 deep dive)

A documentation/ecosystem sweep (Nsight Graphics docs + release notes,
Aftermath SDK + samples, NVIDIA forums, GitHub, the DXR/PIX analogue)
settled the remaining open questions. The verdict: **this is a driver/tool
gap, not a missing host-side step.** Details:

- **How attribution actually works** (frames everything): correlation is
  two-stage. Stage 1, SASS↔IL (SPIR-V), is produced by the *driver's*
  shader compiler and consumed by the profiler directly from the driver at
  trace time — the app cannot author or supply it. Stage 2, IL↔source,
  comes from the NonSemantic debug info embedded in the SPIR-V — which we
  already emit (and which is proven good by compute + monolithic). A
  linked-only failure therefore lives in stage 1 / the sample→pipeline
  join: the SASS was compiled at *library* create, the samples land on the
  *linked* pipeline, and the driver/profiler interface fails to join them.
- **The Aftermath `.nvdbg` channel is a dead end for the profiler.** The
  `ENABLE_SHADER_DEBUG_INFO` → `shaderDebugInfoCb` → `shader-<id>.nvdbg`
  → "NVIDIA Shader Debug Information" search-path flow exists solely for
  the **crash-dump inspector** (mapping faulted-warp addresses). The
  Shader Profiler docs never mention it, its options tab has no `.nvdbg`
  knob, and NVIDIA's blog states the profiler gets SASS↔IL from the driver
  during the trace. `WGPU_AFTERMATH` is irrelevant to attribution —
  consistent with compute/monolithic attributing fine without it. (Since
  R615 the driver embeds this info into crash dumps directly, phasing out
  `.nvdbg` even for its real purpose.)
- **Nobody has ever demonstrated linked-RT attribution.** Zero uses of
  `VK_PIPELINE_CREATE_LIBRARY_BIT_KHR` for RT across nvpro-samples /
  NVIDIAGameWorks / NVIDIA-RTX — every NVIDIA sample builds monolithic, so
  the path likely has no internal QA coverage. Nsight release notes only
  ever claimed *graphics* pipeline-library support (2022.3), never RT.
  Nobody has filed this bug either — we would be first.
- **Precedent says driver-fix-only.** Release notes document a prior R495
  issue: "samples may not be attributed and will be classified as
  'Unattributed'… addressed in a future driver release." A 2022 raygen-only
  correlation loss was likewise fixed in-driver (512.95). 2025.1 known
  issue: "the debug symbolic information provided by the driver is limited
  for ray tracing shaders"; R575 improved it — RT symbolics were being
  fixed branch-by-branch through 2025–2026.
- **The API has no channel to pass anything.** `VkPipelineLibraryCreateInfoKHR`
  is `libraryCount` + `pLibraries` with zero pNext extensibility, and the RT
  create's allowed pNext chain (registry-verified) is flags2 / creation
  feedback (output-only) / pipeline-binary / offline (SC) / robustness /
  CLAS — no metadata channel. `LINK_TIME_OPTIMIZATION_BIT_EXT` and
  `RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT` are graphics-pipeline-library
  only (all VUs on `VkGraphicsPipelineCreateInfo`); the RT library design
  never got a retain-info-across-the-link affordance. Carrying line tables
  across the link is entirely the driver's job.
- **The DXR analogue proves the intended design.** PIX keys attribution to
  each *library's* shader binary (PDB hash), so `AddToStateObject` linking
  is attribution-neutral by construction; nobody in DX land reports losing
  correlation across links. The Vulkan equivalent requires the NV driver
  to preserve module↔SASS tables through the link — exactly the
  undocumented step that's failing.

### Actionable next steps (ranked)

1. **Read the Correlation column** in the Shader Pipelines view of a
   linked-shape capture. It reports per-shader correlation success/failure
   and its hover tooltip names the failing leg and the files consulted
   (known strings: "HL: Unable to locate source referenced by debug line
   tables", "IL: The driver only provided a high level line table").
   Free, and it turns the bug report from speculation into an exact error
   string.
2. **Try an R615+ driver.** We're on R610 — one branch below a documented
   rework of exactly this shader-debug-info plumbing, in a series where RT
   symbolics fixes have landed branch-by-branch.
3. Confirm the GPU Trace activity has **"Collect Shader Pipelines"** and
   **"Collect External Shader Debug Info"** enabled (defaults, but they
   gate the profiler's pipeline view).
4. **File it.** The Shader Profiler docs explicitly instruct: for a high
   quantity of Unattributed samples, "save this report to communicate this
   issue to the Nsight Graphics team" (forums.developer.nvidia.com, Nsight
   Graphics category). Attach both captures — monolithic (attributes) and
   linked (doesn't) — same SPIR-V, plus the Correlation-column tooltip
   text from step 1.
5. ~~Profiling workaround: re-run monolithic~~ — **done, monolithic is now
   the shipping shape** (`RtShaderCache::create`). The libraries+link code
   lives in git history immediately before the flip; restore it when the
   driver/tool gap is fixed to get the 7.5ms rebuild back.

Dropped leads: Aftermath `.nvdbg` plumbing (crash-dump-only channel, see
above); `vkSetDebugUtilsObjectNameEXT` (no tool uses object names for
correlation joins per the docs — still nice for tool readability, but not
an attribution fix); library-lifetime experiment (superseded by the
two-stage model — the join failure is in driver metadata, not object
liveness; cheap to try if filing stalls, but expectations are low).

### Numbers worth keeping

- Library link: **7.5ms**. Monolithic create: **1.25s cold, 1.25s warm**
  (same process, same modules — no driver reuse).
- If library attribution is ever fixed and monolithic still tempts for
  perf (cross-stage inlining), consider tiered compilation: link instantly,
  swap in a background-compiled monolithic pipeline (~1.2s later, re-bake
  the SBT handles at swap). Rejected for now as complexity without a
  proven perf win.
