# `bevy_aurora` — pure ray-traced rendering for clustered geometry

> "Solari is the sun, Aurora is what light does at the edge of the atmosphere."

A standalone bevy crate that renders meshlet geometry through a fully ray-traced
pipeline: cluster acceleration structures for geometry, ReSTIR for lighting, and
DLSS Ray Reconstruction for upscale. Designed to mirror NVIDIA's Zorah / RTX
MegaGeometry architecture inside bevy.

## 1. Goal

Replicate the shape of the Zorah pipeline for bevy:

```
Meshlet geometry ─→ cluster_AS ─→ BLAS ─→ TLAS
                                            │
                                            ▼
                            ┌──────────── Ray Tracing ────────────┐
                            ▼                                       ▼
                Direct Lighting (full res)            Indirect GI (quarter-res)
                ReSTIR DI                              ReSTIR PT (later) / GI (initially)
                            │                                       │
                            └────────── Compose ────────────────────┘
                                            │
                                            ▼
                                   DLSS Ray Reconstruction
                                            │
                                            ▼
                                       Final 4K
```

Three things we want to be able to test end-to-end on a single bunny scene:
1. **Direct lighting** through ReSTIR DI on cluster geometry.
2. **Indirect lighting** through ReSTIR GI (later: ReSTIR PT) on cluster geometry.
3. **DLSS Ray Reconstruction** upscale producing a final clean 4K frame.

## 2. Why not extend `bevy_solari`?

`bevy_solari` is a hybrid renderer by design. `SolariLighting` requires
`DeferredPrepass`, `DepthPrepass`, and `MotionVectorPrepass` — raster passes that
fill a G-buffer the lighting compute then reads. Every realtime shader in solari
calls `gpixel_resolve(gbuffer, depth, ...)` to recover world-space position and
material. Material binding indexes by raster instance ID. Temporal reprojection
assumes raster motion vectors.

If we drop the raster G-buffer in favour of RT primary visibility (which is
required to consume meshlet geometry through cluster_AS without a proxy mesh),
we're rewriting the foundation of every solari shader. At that point we're using
maybe 20% of solari (the BRDF/sampling/world-cache math) and reimplementing the
other 80% on top of a different rendering model.

The pragmatic move is **a new crate that copies algorithms wholesale** (under
the same MIT/Apache license, attribution preserved) and rebinds them to its own
G-buffer source. Solari stays the canonical hybrid path; `bevy_aurora` is the
pure-RT sibling.

## 3. Hybrid-blit integration with bevy

`bevy_solari` runs its lighting compute `.before(main_opaque_pass_3d)`, writes
directly into the camera's `view_target`, and lets the rest of bevy's render
graph (transparent pass, skybox, post-process, tonemap, TAA, UI, swapchain blit)
run normally. That pattern works for us too.

Aurora's pipeline slots into the same architectural seam:

```
[bevy raster prepass]            ← fills raster G-buffer for non-meshlet entities
    │
    ▼
[aurora primary visibility]       ← traces camera rays against TLAS,
                                    writes Aurora's primary G-buffer (PGBuffer)
                                    for meshlet entities. Compute, no fragment shader.
    │
    ▼
[aurora lighting compute]         ← reads PGBuffer (+ raster G-buffer for hybrid scenes),
                                    runs ReSTIR DI / GI / specular,
                                    writes view_target
    │
    ▼
[bevy main opaque (skipped if all surfaces handled by aurora)]
[bevy transparent pass]           ← still works on traced surfaces if hit info exposed
[bevy skybox]
[bevy post-process / tonemap / bloom / TAA / UI]
    │
    ▼
[swapchain]
```

This means `bevy_aurora` cooperates with bevy's existing rendering ecosystem
"for free" — particles, UI, decals on transparent surfaces, post-process all
keep working without aurora knowing about them.

## 4. What to crib from `bevy_solari`

Files that are pure algorithms (no G-buffer coupling) — copied wholesale with
attribution:

| Solari file | LOC | Aurora destination | Notes |
|---|---|---|---|
| `scene/brdf.wgsl` | ~150 | `lighting/brdf.wgsl` | Diffuse + GGX VNDF, no changes |
| `scene/sampling.wgsl` | ~250 | `lighting/sampling.wgsl` | Light sampling, MIS, RNG |
| `realtime/restir_di.wgsl` | 343 | `lighting/restir_di.wgsl` | Rebind to PGBuffer |
| `realtime/restir_gi.wgsl` | 320 | `lighting/restir_gi.wgsl` | Rebind to PGBuffer |
| `realtime/specular_gi.wgsl` | 250 | `lighting/specular_gi.wgsl` | Glossy + world cache |
| `realtime/world_cache_*.wgsl` | ~600 | `lighting/world_cache_*.wgsl` | Hash-grid radiance cache |
| `realtime/presample_light_tiles.wgsl` | ~150 | `lighting/light_tiles.wgsl` | Light-tile presampling |
| `realtime/resolve_dlss_rr_textures.wgsl` | ~80 | `dlss_rr/resolve.wgsl` | DLSS-RR guide-buffer fill |

Files **not** worth cribbing — Aurora replaces them:

| Solari file | Why we replace |
|---|---|
| `realtime/gbuffer_utils.wgsl` | Raster G-buffer resolve. Aurora generates PGBuffer from primary RT, exposes its own resolve helpers. |
| `scene/blas.rs` | Per-mesh BLAS via wgpu. Aurora uses cluster BLAS via raw vk. |
| `scene/binder.rs` | Material binding via instance ID. Aurora needs cluster-id → meshlet-id → material lookup. |
| `realtime/prepare.rs` / `node.rs` | Texture lifecycle + node wiring. Aurora has its own. |
| `realtime/extract.rs` | Extracts SolariLighting components. Aurora extracts AuroraCamera. |

## 5. Milestones

Implementation order. Each milestone ships an end-to-end runnable example so
progress is observable.

### M-A — "Cluster AS pipeline produces a traceable TLAS" *(2-3 weeks)*

End state: `bunny.meshlet_mesh` loaded, every meshlet's positions dequantized,
one indirect cluster_AS build produces all CLAS, one BLAS contains all CLAS
via `BUILD_CLUSTERS_BOTTOM_LEVEL`, one TLAS instances the BLAS. Raw vk compute
shader traces fixed rays, writes hit/miss to a storage texture, blit to
swapchain.

**No lighting yet.** Just proves the geometry pipeline.

Architecture: a single `ClusterAsManager` bevy `Resource` owns the persistent
CLAS storage buffer, the BLAS handles, and the TLAS handle for the entire
lifetime of the app. Per-frame systems mutate them via change detection on
`AuroraMeshlet3d` components. No buffer recreation on the hot path.

This mirrors `bevy_pbr::meshlet::meshlet_mesh_manager`'s design exactly --
the rasterizer side already has the persistent-buffer + per-frame-update
pattern; Aurora is the same shape for the RT side. The committed-vs-sparse
backing decision (M-A vs M-A.streaming) is hidden inside `PersistentBuffer`
behind a `device_address(offset)` API, so callers don't change.

```rust
// Sketch:
#[derive(Resource)]
struct ClusterAsManager {
    clas_storage: PersistentBuffer,            // CLAS payload, fixed virtual addr
    clas_index: HashMap<MeshletId, u64>,       // meshlet → device addr in clas_storage
    free_ranges: RangeAllocator,               // M-A: bump allocator; later: real free-list
    blases: HashMap<MeshHandle, ClusterBlas>,
    tlas: Option<KhrTlas>,
}

fn process_meshlet_uploads(
    mut mgr: ResMut<ClusterAsManager>,
    new: Query<&AuroraMeshlet3d, Added<AuroraMeshlet3d>>,
    /* device, queue, etc. */
) { /* allocate + cmd_build_indirect */ }

fn update_tlas(/* ... */) { /* rebuild instance array, update TLAS */ }
```

Deliverables:
- `scene/raw_vk.rs` — `RawBuffer`, `PersistentBuffer`, `RangeAllocator` allocator primitives
- `scene/cluster_as.rs` — `ClusterAsManager` resource + per-frame systems
- `scene/blas.rs` — cluster-bottom-level BLAS build helper
- `scene/tlas.rs` — KHR TLAS build helper
- `scene/meshlet_loader.rs` — `MeshletMesh` dequant + descriptor preparation
- `examples/m_a_traceable.rs` — render hit/miss image of bunny

**Risks**:
- CLAS array sizing — committed 256 MiB cap initially; M-A.streaming lifts this
- Material binding deferred (M-A traces only — no shading)
- Per-mesh static build only (no LOD, no eviction in M-A; M-A.streaming adds them)

### M-B — "RT primary visibility writes PGBuffer" *(2-3 weeks)*

End state: replace M-A's "hit/miss image" with a compute shader that traces
camera rays and writes a full primary G-buffer (PGBuffer): world position, world
normal, geometric normal, material id, motion vector, depth.

This is the layer that replaces `DeferredPrepass` for meshlet entities. Format
must be DLSS-RR-compatible (Aurora's eventual upscale consumer):

```
PGBuffer:
  world_position : Rgba32Float   (xyz position, w optional flags)
  world_normal   : Rg16Snorm     (octahedral encoded)
  motion_vector  : Rg16Snorm     (screen-space, like raster motion)
  material_id    : R32Uint       (cluster_id → meshlet_id → material slot)
  depth          : Depth32Float  (linearized for ray query consumers)
```

Deliverables:
- `primary/pgbuffer.rs` — PGBuffer texture allocation + lifecycle
- `primary/visibility.wgsl` — primary ray trace → PGBuffer
- `primary/material_lut.rs` — cluster-id → material lookup table
- `examples/m_b_pgbuffer_vis.rs` — visualize PGBuffer normals as colors

**Risks**:
- Material lookup is novel — no solari precedent.
- Motion vectors from RT primary need camera-jitter awareness (TAA / DLSS).
- Hybrid scenes (meshlet + raster entities) need a write strategy: aurora-priority
  inside meshlet pixels, raster fallback elsewhere. Defer to M-D's compose.

### M-C — "ReSTIR DI shadows on meshlet geometry" *(1-2 weeks)*

End state: bunny + light + ground plane. Direct lighting only. ReSTIR DI traces
shadow rays through the cluster TLAS. Plane is a meshlet too (so we exercise
real shadow contact, not just bunny-on-itself).

This is where `restir_di.wgsl` gets cribbed and rebound. The original reads
`gpixel_resolve(gbuffer, depth, ...)`; the rebound version reads PGBuffer.

Deliverables:
- `lighting/restir_di.wgsl` — copied from solari, rebound
- `lighting/pgbuffer_utils.wgsl` — replaces solari's `gbuffer_utils`
- `lighting/light_tiles.wgsl` — copied from solari
- `examples/m_c_direct_light.rs` — bunny under one directional light, shadow visible

**Risks**:
- PGBuffer permute_pixel for temporal reprojection needs equivalent of solari's
  raster-motion path. May need to rebuild from camera motion + hit world pos.
- Validating "shadow looks right" — compare against bevy's regular renderer
  doing the same scene with raster shadow.

### M-D — "ReSTIR GI bounce + compose" *(1-2 weeks)*

End state: same scene, indirect bounce works. Light hits ground, scatters,
illuminates underside of bunny.

Compose pass: aurora's lighting compute writes view_target with both DI + GI
combined (pre-tonemapping, pre-post-process). Bevy's main opaque pass becomes
a no-op for meshlet entities. Hybrid scenes still composite with raster
opaque/transparent.

Deliverables:
- `lighting/restir_gi.wgsl` — copied from solari, rebound
- `lighting/specular_gi.wgsl` — copied from solari
- `lighting/world_cache_*.wgsl` — copied from solari
- `compose/compose.wgsl` — DI + GI + specular combine, write view_target
- `examples/m_d_indirect_light.rs` — visible color bleed under bunny

**Risks**:
- World cache memory budget on large meshlet scenes
- Specular GI's per-pixel path tracing wants a real material at hit — needs M-B's
  material lookup proven solid

### M-E — "Quarter-res GI + DLSS-RR" *(1-2 weeks)*

End state: GI runs at 1/4 viewport, stochastic upsample composes with DI at
full res, DLSS-RR upscales the final view_target to display resolution.

The Zorah-shape pipeline. Mirrors what the slide shows: 1080p direct + 512p
indirect → DLSS-RR → 4K.

Deliverables:
- `compose/quarter_res.wgsl` — stochastic upsample with 4× radiance multiplier
- `dlss_rr/` — DLSS-RR plumbing (lift from solari's `resolve_dlss_rr_textures`)
- `examples/m_e_dlss_4k.rs` — same scene, 4K output

**Risks**:
- DLSS-RR guide-buffer expectations need exact match (motion, normals, roughness,
  depth, albedo) — verify shape-by-shape against solari's working version
- Quarter-res introduces aliasing on high-frequency GI — may need a small
  variance-aware filter before DLSS-RR

### M-A.streaming — "Sparse CLAS array + meshlet page lifecycle" *(2-3 weeks)*

End state: CLAS array lives in a sparse-residency `VkBuffer` whose pages are
bound on demand as meshlet streamer adds/evicts geometry. No reallocation +
copy when the array grows. Hooks into `bevy_pbr::meshlet`'s page lifecycle.

Mirrors what NVIDIA gets from D3D12 Reserved Resources in Zorah. Required
before Aurora can render scenes that don't fit in a fixed CLAS budget at
build time.

Deliverables:
- `scene/raw_vk.rs::SparseBuffer` wrapper with `bind_pages` / `unbind_pages`
- Sparse-binding queue selection at adapter init
- CLAS allocator that hands out offset ranges as meshlet pages stream in
- Eviction path that frees ranges when meshlet pages stream out
- Defrag pass (Adam's "move ops" from Zorah) — periodic compaction so the
  bound range stays contiguous

**Risks**:
- Sparse-binding queue ordering vs the regular graphics queue — bind ops are
  asynchronous; need fences to keep CLAS reads ordered after binds.
- Page granularity (typically 64 KiB on NVIDIA, larger on some IHVs) — wastes
  memory if individual CLASes are small.
- Defrag is intrusive; needs to coordinate with in-flight ray traces.

This is a separate milestone, **not a prerequisite** for M-B/M-C/M-D — those
all work with base M-A's committed buffers as long as the scene fits in the
fixed CLAS budget. M-A.streaming becomes a prerequisite for "render scenes
larger than 1 GB of clusters" and for matching Zorah's stutter-free
streaming behaviour.

### M-F — "ReSTIR PT (Lin 2022)" *(4-8 weeks, separate project)*

End state: ReSTIR GI replaced with full ReSTIR PT. Multi-bounce reuse via
hybrid shift. Optional Lin 2026 Enhanced techniques layered in.

Major implementation:
- New reservoir layout (path representation in primary sample space)
- Random replay through TLAS during spatial reuse
- Path-tree initial sampling
- Pairwise MIS, hybrid shift mapping
- Optional: paired spatial reuse, dup map, footprint thresholds (Enhanced paper)

This is its own multi-week chunk and is **not required** for the stated test goal
(direct + indirect + DLSS). M-D's ReSTIR GI is sufficient. M-F is the path to
matching Zorah's quality bar specifically.

## 6. Crate layout

```
crates/bevy_aurora/
├── Cargo.toml
├── plan.md                       (this file)
├── README.md                     (when there's something to say)
├── src/
│   ├── lib.rs                    AuroraPlugins, feature-gated
│   ├── scene/                    AS pipeline (M-A)
│   │   ├── mod.rs
│   │   ├── cluster_as.rs         CLAS array, dequant, indirect build
│   │   ├── blas.rs               cluster-bottom-level BLAS
│   │   ├── tlas.rs               KHR TLAS
│   │   ├── meshlet_loader.rs     MeshletMesh → upload + descriptor
│   │   └── streaming.rs          (deferred) CLAS lifetime / eviction / defrag
│   ├── primary/                  RT primary visibility (M-B)
│   │   ├── mod.rs
│   │   ├── pgbuffer.rs           PGBuffer texture set + lifecycle
│   │   ├── visibility.wgsl       camera ray trace → PGBuffer
│   │   └── material_lut.rs       cluster_id → material slot
│   ├── lighting/                 ReSTIR + world cache (M-C, M-D, M-F)
│   │   ├── mod.rs
│   │   ├── pgbuffer_utils.wgsl   replaces solari's gbuffer_utils
│   │   ├── brdf.wgsl             from solari
│   │   ├── sampling.wgsl         from solari
│   │   ├── light_tiles.wgsl      from solari (presample_light_tiles)
│   │   ├── restir_di.wgsl        from solari, rebound
│   │   ├── restir_gi.wgsl        from solari, rebound (until M-F)
│   │   ├── specular_gi.wgsl      from solari, rebound
│   │   ├── world_cache_query.wgsl     from solari
│   │   ├── world_cache_update.wgsl    from solari
│   │   └── world_cache_compact.wgsl   from solari
│   ├── compose/                  combine + upscale (M-D, M-E)
│   │   ├── mod.rs
│   │   ├── compose.wgsl          DI + GI + specular → view_target
│   │   └── quarter_res.wgsl      stochastic upsample (M-E)
│   ├── dlss_rr/                  DLSS Ray Reconstruction (M-E)
│   │   ├── mod.rs
│   │   └── resolve.wgsl          (lift from solari)
│   └── components.rs             AuroraCamera, AuroraMeshlet3d, etc.
└── examples/
    ├── m_a_traceable.rs          M-A: hit/miss image
    ├── m_b_pgbuffer_vis.rs       M-B: visualize PGBuffer
    ├── m_c_direct_light.rs       M-C: shadows
    ├── m_d_indirect_light.rs     M-D: GI bounce
    ├── m_e_dlss_4k.rs            M-E: full Zorah pipeline
    └── m_f_restir_pt.rs          M-F: ReSTIR PT (later)
```

Roughly 5000-8000 LOC at completion through M-E, mostly WGSL + raw-vk plumbing.

## 7. Plugin shape

```rust
pub struct AuroraPlugins;

impl PluginGroup for AuroraPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(AuroraScenePlugin)            // M-A
            .add(AuroraPrimaryVisibilityPlugin) // M-B
            .add(AuroraLightingPlugin)          // M-C, M-D
            .add(AuroraComposePlugin)           // M-D, M-E
            .add(AuroraDlssRrPlugin)            // M-E (optional feature)
    }
}

#[derive(Component)]
pub struct AuroraCamera {
    /// If true, run RT primary visibility for meshlet entities (M-B+).
    /// If false, lighting still works but expects an external G-buffer.
    pub rt_primary: bool,
    /// Resolution scale for indirect GI dispatch (M-E). 1 = full, 2 = quarter.
    pub gi_scale: u32,
    /// Use DLSS-RR for final upscale (M-E).
    pub dlss_rr: bool,
}

#[derive(Component)]
pub struct AuroraMeshlet3d(pub Handle<MeshletMesh>);
```

User opt-in: add `AuroraPlugins`, attach `AuroraCamera` to the camera, replace
`MeshletMesh3d` with `AuroraMeshlet3d` (or keep both — Aurora can read the same
asset). No `Mesh3d`, no raster pipeline for those entities — pure RT.

## 8. Architectural decisions to revisit later

These are unresolved and will surface during M-A or M-B:

- **wgpu integration scope.** Currently raw vk via `as_hal` for cluster_AS.
  Long-term: extend `wgpu-hal` with cluster-AS resource types + builder methods.
  Multi-week wgpu work; deferred until the API surface is settled.
- **CLAS lifetime / streaming.** Production Nanite has CLAS array eviction +
  defrag. Prototype scope: static meshes only, build once, never evict.
  Production scope: integrate with `bevy_pbr::meshlet`'s page lifecycle.
- **Sparse-residency buffers for streaming (D3D12 "Reserved Resources" equivalent).**
  Zorah uses D3D12 Reserved Resources extensively to avoid CPU-side stutter
  from CLAS-array resizes — every meshlet page streamed in extends the array,
  and reallocating + copying multi-MB buffers on a frame-hot path stalls.
  Vulkan equivalent: `VkBufferCreateInfo::flags = SPARSE_BINDING_BIT |
  SPARSE_RESIDENCY_BIT` + `vkQueueBindSparse` to wire pages into virtual ranges.
  Needs `sparseBinding` + `sparseResidencyBuffer` device features (universal on
  discrete GPUs) plus a sparse-binding queue. wgpu has zero coverage for
  sparse buffers — this is more raw-vk territory.

  **Scope**: deferred to a separate milestone "M-A.streaming". Base M-A uses
  committed (`vkAllocateMemory + vkBindBufferMemory`) buffers since it's
  static-only. Migrating committed → sparse later means swapping `RawBuffer`'s
  allocation backend; the device-address API is unchanged so call sites don't
  move.
- **Material model.** Per-mesh material (one StandardMaterial per AuroraMeshlet3d
  entity, applied to all clusters of that mesh). Per-cluster material slots
  (Nanite-style multi-material meshlets) deferred — not required for prototype.
- **Hybrid scene compositing.** When raster entities and meshlet entities coexist
  in the same camera, Aurora's PGBuffer overrides raster G-buffer per-pixel where
  a primary ray hit a meshlet. Need a clean boolean mask. Deferred to M-D.
- **TAA jitter compatibility.** Camera ray jitter for TAA / DLSS-RR motion-vector
  generation needs aurora-aware sequencing. Deferred to M-B / M-E.

## 9. Open questions

- **wgpu version coupling.** Aurora pulls in our slyedoc/wgpu fork for cluster_AS.
  Bevy main moves regularly; rebasing is a recurring tax. Explicit decision: stay
  on the fork until cluster_AS lands in mainline wgpu (years out).
- **Solari as a dep, sibling, or unrelated?** Current plan: unrelated, copy WGSL
  with attribution. Alternative: depend on solari for non-G-buffer-coupled bits.
  Pragmatic argument for "unrelated": shader patches are small + we don't want
  Aurora gated on solari API stability.
- **Demo content.** `bunny.meshlet_mesh` + ground plane meshlet for M-A through
  M-D. Sponza / Crytek for M-E onward. Decide whether to ship `.meshlet_mesh`
  conversions of standard test scenes or always preprocess at runtime.

## 10. Risks I'm tracking

- **Material lookup novelty.** Cluster-id → meshlet-id → material slot is new
  ground; solari's per-instance-id model doesn't translate. M-B is the riskiest
  milestone for this reason.
- **PGBuffer ↔ DLSS-RR fit.** DLSS-RR has tight expectations on guide-buffer
  shape. Verify against solari's working DLSS-RR call site before committing
  PGBuffer formats.
- **wgpu API drift.** Aurora's raw-vk reaches assume a specific wgpu trunk shape.
  Wgpu trunk changes may require touch-ups; budget rebases.
- **Meshlet asset format stability.** `MESHLET_MESH_ASSET_VERSION` bumps occasionally.
  Aurora's M-A loader codes against the current version (3); upgrade path is a
  reload, not a migration.
- **Validation layer noise.** As scope grows, more VUIDs surface. Budget time for
  fixing. Two non-obvious ones already documented in the example:
  `srcInfosCount` is a device address (not a count) in IMPLICIT/EXPLICIT modes;
  `dst_addresses` buffer needs `ACCELERATION_STRUCTURE_STORAGE_KHR` usage.

## 11. Reference material

Already read and digested:
- Lin et al. 2022 ReSTIR PT, Lin et al. 2026 Enhanced ReSTIR PT (papers in `/p/restir`)
- Zorah talk transcript (`/p/restir/Path_Tracing_Nanite_in_NVIDIA__transcript_en-us.txt`)
- NVIDIA `VK_NV_cluster_acceleration_structure` extension spec
- Unreal NVRTX `Engine/Shaders/Private/Nanite/RayTracing/` (~4300 LOC across 22 files)
  — covers the producer side of the geometry pipeline; informs M-A's shape
- bevy_solari (entire codebase)

To consult during implementation:
- Lin 2022 supplementary material — hybrid shift Jacobian derivations
- Vulkan 1.4 ray tracing chapter — KHR-AS specifics
- DLSS 4 SDK docs — Ray Reconstruction guide-buffer formats

## 12. Status

**Pre-M-A.** Cluster_AS bottom-of-the-stack proven on a synthetic 1-triangle test
in `examples/3d/cluster_acceleration_structure.rs` (M-A scratch / smoke test).
Real implementation begins when `bevy_aurora/src/lib.rs` exists.

Forks involved (all on slyedoc):
- `slyedoc/wgpu#cluster-acceleration-structure`
- `slyedoc/ash#cluster-acceleration-structure`
- `slyedoc/gpu-allocator#ash-push-rename`
- `slyedoc/bevy#restir_primary` (this branch)
