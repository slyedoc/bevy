# Slang integration: current state + the RT debug-info attribution problem

Written 2026-08-02, at the end of the slang-modernization arc
(`chore/research-slang`, through commit `dab549340c`). The open problem at
the bottom — Nsight shader-profiler attribution for the RT stages — is
unresolved and is the subject for a follow-up session.

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
- **RT pipeline shape**: per-stage `VK_KHR_pipeline_library` pipelines
  cached in `RtLibraryCache` (raygen / miss keyed by sky generation /
  shadow / per-hit-group), linked into the executable pipeline. All
  layout-free (`DESCRIPTOR_HEAP_EXT` flags2), stages chained with the
  shared descriptor-mapping table.

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

### Ruled out

- Missing SPIR-V debug info (present + verified; compute attributes).
- Nsight version: reproduced identically on Nsight Graphics **2026.3**.
- `VkDeviceDiagnosticsConfigCreateInfoNV` with `ENABLE_SHADER_DEBUG_INFO`
  (+resource tracking/checkpoints/error reporting) at device creation — the
  fork wires it behind `WGPU_AFTERMATH=1`. No change.
- `VK_KHR_pipeline_executable_properties` +
  `CAPTURE_INTERNAL_REPRESENTATIONS` on libraries and link. No change.
- Keeping monolithic permanently: a monolithic create costs **1.25s** (cold
  AND warm — the driver reuses nothing in-process) vs **7.5ms** for the
  library link. Every sky swap / hot reload / material rebuild would hitch.

### Leads for the next session (most promising first)

1. **Distinct entry-point names — IMPLEMENTED, awaiting a capture.** Every
   stage now emits `OpEntryPoint` under its real name (raygen /
   miss_primary / chit_opaque / …) instead of five libraries all exporting
   `main`; stage `pName`s match. If the driver keyed per-pipeline debug
   records by entry name, the collision is gone. The next Nsight capture
   answers this.
2. **Separate shader debug info via the Aftermath channel.** Nsight
   Graphics can load shader debug info from configured *search paths*
   (`.nvdbg` blobs) instead of relying on live driver metadata. With
   `ENABLE_SHADER_DEBUG_INFO` on (the `WGPU_AFTERMATH` path), the Aftermath
   SDK's `GetShaderDebugInfo` callback receives per-shader blobs — dump
   them to a directory and point Nsight's shader debug info search path at
   it. The zero project already integrates the Aftermath SDK (see the
   aftermath-debug workflow), so most of the plumbing exists. This is the
   most likely "we're doing something wrong" fix: the driver may generate
   the metadata but Nsight may need to be *handed* it for linked pipelines.
3. **`vkSetDebugUtilsObjectNameEXT`** on the library pipelines, the linked
   pipeline, and the shader modules. Cheap, improves tool bookkeeping
   regardless, and some tools use object identity for correlation joins.
5. **Library lifetime experiment.** The cache keeps the library pipelines
   alive alongside the linked pipeline. Try destroying them right after
   the link (spec-legal): if the driver's sample→pipeline mapping is
   confused by never-bound pipelines that own the SASS, this changes the
   picture.
6. **Reference check.** Find any NVIDIA sample (nvpro-samples,
   vk_mini_samples) that uses RT pipeline libraries AND demonstrates
   shader-profiler attribution. If none exists, that is soft evidence for
   a driver/tool gap → file the report. The repro here is minimal and
   airtight either way: same modules, monolithic attributes, linked
   doesn't.
7. **Re-run the monolithic experiment** when needed: replace
   `RtLibraryCache::link` with a monolithic create over the cached
   libraries' `modules` (stages: raygen, miss, per-group chit[+ahit],
   shadow; groups in that order to keep the SBT layout; chain the mapping
   table per stage; flags2 = OMM | DESCRIPTOR_HEAP, no LIBRARY bit, no
   `library_info`/`library_interface`). The exact code shape exists in
   this branch's reflog around `dab549340c`.

### Numbers worth keeping

- Library link: **7.5ms**. Monolithic create: **1.25s cold, 1.25s warm**
  (same process, same modules — no driver reuse).
- If library attribution is ever fixed and monolithic still tempts for
  perf (cross-stage inlining), consider tiered compilation: link instantly,
  swap in a background-compiled monolithic pipeline (~1.2s later, re-bake
  the SBT handles at swap). Rejected for now as complexity without a
  proven perf win.
