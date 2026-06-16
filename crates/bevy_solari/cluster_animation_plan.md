# Cluster animation plan (skeletal deform → per-instance CLAS/BLAS)

Port of aurora's animated-cluster proof-of-concept to the bevy_solari
branch, rebuilt on top of the GPU transform table. Goal: ray-traced
skeletal animation (linear-blend skinning) of cluster meshes, fully
GPU-driven, no per-frame CPU geometry work.

Status: **DONE — working + hardware-validated** (RTX 5090, NV cluster-AS). The
glTF fox skins, instantiates, builds per-instance BLAS, and ray-traces with the
run animation deforming the traced geometry. Example:
`examples/3d/solari/animation.rs`.

Pipeline (per frame), all GPU-driven:
- **CLAS templates** (`geometry/clas_template.rs`): topology-only per-cluster
  templates built once at upload for animated meshes (those with bloat AABBs);
  per-cluster template addresses in a global `cluster_template_addresses` table
  parallel to `ClasArena::cluster_clas_addresses`.
- **Deform** (`accel/deform.rs` + `deform.wgsl`, `Deform` stage):
  `extract_animated_skins` gathers active animated instances (cap
  `MAX_ANIMATED_INSTANCES=8`) — skinning palette (joint →
  `GpuSlot<TransformGraph>` node slot), inverse-bind poses
  (`SkinnedMeshInverseBindposes` → `mat3x4`), mesh pool bases
  (`ClusterMeshManager::animated_mesh_pointers`). The compute LBS-skins rest
  verts using `joint_world(palette[j]) · inverse_bind[j]` and premultiplies the
  inverse instance world (mesh-local output). **`joint_world` walks the live
  `local`/`parent` columns** rather than reading the change-driven `world[]`
  buffer — that buffer only re-walks nodes whose own local changed, so a joint
  whose parent animates but whose own local is static (e.g. a toe bone) would be
  stale and its verts pinned. Walking `local`/`parent` is always correct.
- **Instantiate + per-instance BLAS** (`accel/animated_blas.rs` +
  `instantiate.wgsl`, `BuildAnimatedBlas` stage, after `BuildBlas`): instantiate
  compute → `INSTANTIATE_TRIANGLE_CLUSTER` (finest-LOD clusters → templates fed
  the deformed verts) → per-instance CLAS; raw-VK `BUILD_CLUSTERS_BOTTOM_LEVEL`
  (explicit dst) → one BLAS per active slot in a stable pool (`base+slot*stride`).
  The compute also repoints `instance_blas_address[slot]` at the per-instance
  BLAS (runs after `assign_address`, so it overwrites the static shared address).
- **PTLAS**: `prepare_ptlas_params` seeds active animated slots as per-frame
  rewrites (their BLAS content changes in place).
- **Resolve**: `raytracing_scene_bindings.wgsl` branches
  `resolve_triangle_data_full` on the slot-indexed `instance_animated` table
  (scene-group bindings 14/15/16: deform positions/normals + table, `Deform`-owned
  and diff-maintained so non-animated scenes pay nothing), reading the deform pool
  for animated hits.

Example notes: runs `PbrPlugin`-off / `TransformPlugin`-off (the sun is a
`SolariDirectionLight`; `init_asset::<StandardMaterial>()` re-registers the asset
the glTF loader + material convert need). `SolariTransformPlugin` now runs the
CPU `sync_simple_transforms` + `propagate_transforms_for::<With<Node>>` (roots +
UI) so examples don't need bevy's `TransformPlugin`; `bevy_ui` is a required
bevy_solari dep.

---

## 1. What aurora does (reference architecture)

Per animated instance, every frame:

1. **Deform** — compute shader skins rest-pose vertices into a
   per-instance position/normal pool (LBS: blend 4 joint matrices,
   write mesh-local positions).
2. **Instantiate** — feed deformed positions into per-cluster **CLAS
   templates** (topology-only, built once at upload) via NV's
   `INSTANTIATE_TRIANGLE_CLUSTER` op → fresh per-instance CLAS each
   frame. Per-cluster bloat AABB bounds the deform envelope.
3. **BLAS** — build one BLAS per animated instance over its
   instantiated CLAS (`BUILD_CLUSTERS_BOTTOM_LEVEL`).
4. **TLAS** — instance references its per-instance BLAS; transform is
   the skinned-mesh world transform.
5. **Resolve** — hit shader branches on `is_animated` and fetches the
   vertex from the deform pool instead of the static pool.

Aurora drove skinning off Bevy's CPU `prepare_skins` → `SkinUniforms`
(per-frame CPU upload of joint matrices) and pre-multiplied each
instance's `inverse(world_from_local)` in the deform shader so the
TLAS transform path stayed identical to static instances.

Key aurora files (for reference): `src/accel/deform.{rs,wgsl}`,
`src/accel/clas_template.rs`, `src/accel/clas_instantiate.rs` +
`instantiate_lod_aware.wgsl`, `src/accel/animated_storage.rs`,
`src/realtime/resolve.wgsl` (conditional fetch).

---

## 2. The simplification: skin from the GPU transform table

This branch has the GPU transform table (`src/transform/`): every
entity with `GlobalTransform` is a node, propagation writes
`world[node_slot]` (mat3x4) each frame, and skeleton joints are
already nodes in that table. **We do not need Bevy's `SkinUniforms`
or `prepare_skins`.**

Skin matrix for joint `j` of an instance:

```
skin_mat[j] = world[ palette[j] ] * inverse_bind[j]
```

- `world[palette[j]]` — joint's world transform, already on GPU from
  propagation. `palette[j]` is the transform-table node slot of the
  instance's `SkinnedMesh.joints[j]`.
- `inverse_bind[j]` — the "known ref state" the user flagged: must be
  copied to the GPU once per skinned mesh (from Bevy's
  `SkinnedMeshInverseBindposes` asset). This is the **only** new
  ref-state upload.

So two new per-skin inputs replace aurora's CPU skin-matrix upload:
1. **inverse-bind-pose buffer** (per mesh/skeleton, ref state) — upload once.
2. **joint palette** (per instance: joint-index → node-slot) — extract
   on skeleton bind / change, scatter as a GPU column.

Everything else stays on GPU. The deform shader derives joint worlds from the
transform table's `local`/`parent` columns (ancestor-walk — see the status
block on why, not the change-driven `world[]` buffer); no `world_from_local`
round-trip through the CPU.

---

## 3. Pipeline integration

Current `SolariClusterSystems` order (`src/lib.rs:62`,
`src/accel/mod.rs:71`):

```
Scatter → Propagate → Classify → Select → BuildBlas → BuildTlas → Cleanup
```

New order with a `Deform` stage and an animated CLAS/BLAS path:

```
Scatter
  → Propagate            (world[] for all nodes incl. joints; instance gather)
  → Deform               (NEW: skin animated instances → deform pool)
  → InstantiateClas      (NEW: templates + deform pool → per-instance CLAS)
  → Classify             (static buckets; unchanged)
  → Select               (static DAG cut; unchanged)
  → BuildBlas            (static shared buckets; unchanged)
  → BuildAnimatedBlas    (NEW: per-instance BLAS over instantiated CLAS)
  → BuildTlas            (PTLAS; animated instances always re-specified)
  → Cleanup
```

`Deform`/`InstantiateClas`/`BuildAnimatedBlas` are no-ops when no
animated instances exist (skip on empty active list), so static scenes
pay nothing.

The static path (Classify/Select/BuildBlas with **shared** BLAS by
geometry×LOD-band) is **untouched**. Animated instances opt out of
BLAS sharing entirely — they cannot share, every pose is unique.

### Why this slots in cleanly

The PTLAS fill (`ptlas_fill.wgsl`) already keys each instance off a
per-slot `instance_blas_address[slot]` and re-specifies any instance
that "moved" or whose geometry is `dirty`. It does **not** care whether
a BLAS is shared or per-instance. So an animated instance just needs:

- `instance_blas_address[slot]` → its own per-instance BLAS address
  (written by `BuildAnimatedBlas` instead of `blas_sharing`), and
- to be force-rewritten every frame (its geometry changes each frame).

That's a small extension to `fill_incremental`, not a new TLAS path.

---

## 4. New data (GPU buffers / columns)

### Per-skinned-mesh (ref state, upload once)

- `inverse_bind_poses: PersistentGpuBuffer<mat3x4>` in
  `ClusterMeshManager` (or a sibling skin manager). Source: Bevy
  `SkinnedMeshInverseBindposes`. The bake already stamps
  `ClusterMesh::inverse_bind_count` for validation. Per-mesh base
  offset tracked alongside the existing joint streams.

Joint streams already uploaded: `vertex_joint_indices`,
`vertex_joint_weights` (see `mesh_manager.rs:119,142`).

### Per-animated-instance (new GPU columns, via `gpu_table!`/`GpuColumn`)

Follow the existing column pattern in `src/instance/gpu_instances.rs`:

- `AnimatedFlagColumn: u32` — 0 static / 1 animated. Read by resolve +
  ptlas fill.
- `DeformPoolBaseColumn: u32` — base offset into the deform pool for
  this instance (resolve uses it to bias vertex fetch).
- `JointPaletteBaseColumn: u32` — base into the joint-palette pool.
- `JointPaletteColumn` (variable) — the palette itself
  (`joint-index → node_slot`), one block per animated instance. Either
  a dedicated pool keyed by base, or a sub-allocated persistent buffer.

Palette extraction: on `SkinnedMesh` add/change, map each
`joints[k]: Entity` → its `GpuSlot<TransformGraph>` and scatter.

### Deform pool (per-frame, per-active-instance)

- `deform_positions: array<f32>` (stride 3) — slot-indexed by active
  animated instance: `base = active_index * MAX_VERTS_PER_ANIMATED_MESH`.
- `deform_normals: array<u32>` (octahedral, to match
  `vertex_normals`) — parallel.

Size = `MAX_ANIMATED_INSTANCES * MAX_VERTS_PER_ANIMATED_MESH`. Start
with aurora's caps (8 instances, 65536 verts) as consts; make them
plugin config later. Sparse-back it like the other pools so the cap
costs no committed memory until used.

### Animated CLAS + BLAS storage

- **CLAS templates**: per-cluster topology-only CLAS built once at
  upload for animated meshes (new build in `clas_arena.rs`, op
  `BUILD_TEMPLATE` / `TEMPLATE_INSTANTIATE` path; uses
  `cluster_bloat_aabbs` as the instantiation bounding-box limit).
  Stored in a template arena; per-cluster template addresses in a
  table parallel to `cluster_clas_addresses`.
- **Instantiated CLAS**: per-frame arena (sparse, aliased each frame)
  holding this frame's instantiated CLAS for all active animated
  instances. Addresses written by the instantiate build into a
  per-instance ref list.
- **Animated BLAS pool**: per-instance stable BLAS regions (one slot
  per active animated instance), `EXPLICIT_DESTINATIONS`, mirroring
  `blas_sharing.geometry_blas_pool` but keyed by animated-instance slot
  rather than bucket. Address → `instance_blas_address[slot]`.

---

## 5. Shaders

### `deform.wgsl` (NEW) — `Deform` stage

One workgroup per active animated instance (y), threads over vertices (x).

Inputs: active-slot table, `world[]` (transform table), per-mesh
`inverse_bind_poses`, `vertex_joint_indices`, `vertex_joint_weights`,
rest `vertex_positions`/`vertex_normals`, per-instance `world` (the
instance's own node, for the local-space premultiply).

Per vertex:
```
m = Σ_k w[k] * (world[palette[base+jidx[k]]] * inverse_bind[ibp_base+jidx[k]])
pos_world  = m * vec4(rest_pos, 1)
pos_local  = inverse(instance_world) * pos_world      // keep TLAS path identical
nrm_local  = normalize(linear(inverse(instance_world)) * linear(m) * rest_nrm)
deform_positions[out_base + v] = pos_local
deform_normals[out_base + v]   = octahedral_encode(nrm_local)
```

Notes:
- `instance_world` = `cluster_instance_transforms[slot]` (already
  gathered). Its inverse can be computed in-shader (mat3x4 affine
  inverse) or supplied. Premultiplying keeps resolve's
  `transforms[instance_id]` application unchanged — same trick as
  aurora, but the transform now comes from the GPU table, not CPU.
- Alternative (simpler, revisit): skin straight to world space and set
  the animated instance's transform column to identity. Costs a
  separate motion-vector story; defer.
- Normals: LBS-on-normals (rotate by blended 3×3, renormalize). Exact
  for uniform-scale rigs; matches aurora's documented approximation.

### `instantiate.wgsl` (NEW) — `InstantiateClas` stage

Port aurora's `instantiate_lod_aware.wgsl`. One workgroup per active
animated instance. Emits
`VkClusterAccelerationStructureInstantiateClusterInfoNV` records
pointing each cluster's template at its deformed-position slice in
`deform_positions`. GPU atomic counter drives the indirect instantiate
build count. Writes a per-instance CLAS ref list for the BLAS build.

LOD: aurora picks a discrete LOD band per instance (`last_lod[slot]`).
**Milestone 1: skip LOD — instantiate the finest level only.** Add
per-instance band selection in a later milestone (a lightweight
per-instance version of `selector.wgsl`'s DAG cut, or reuse the band
from `blas_sharing` classify input).

### `raytracing_scene_bindings.wgsl` — conditional vertex fetch

`load_cluster_vertex` / `resolve_triangle_data_full`
(`raytracing_scene_bindings.wgsl:199,251`) currently always read the
static `vertex_positions`. Add the animated branch:

```
// in resolve_triangle_data_full, we have instance_id + global vertex slot
if animated_flag[instance_id] == 1u {
    let local = (cluster.vertex_offset + cluster_indices[...]) ; // mesh-local-ish
    let p = deform_positions[ deform_pool_base[instance_id] + (slot - mesh_vertex_base) ];
    ...
}
```

Mirror aurora's `animated_pool_offset` bias so the same
`cluster.vertex_offset + local_index` arithmetic indexes the deform
pool (store the bias in `DeformPoolBaseColumn` precomputed as
`pool_base - mesh_vertex_base`). Positions + normals get the branch;
uv/tangent stay from the rest streams for milestone 1 (uv is
deform-invariant; tangent approximation acceptable, deform later).

### Raw-VK builds (Rust, in `accel/`)

- `BuildAnimatedBlas`: clone `blas_rebuild.rs`'s
  `cmd_build_cluster_acceleration_structures_indirect` call with op
  `BUILD_CLUSTERS_BOTTOM_LEVEL`, `EXPLICIT_DESTINATIONS`, count =
  active animated instances, src = per-instance CLAS ref lists, dst =
  animated BLAS pool addresses. Same barrier discipline (pre/post
  global AS barrier) as the existing rebuild.
- Template build + instantiate: new raw-VK calls in `clas_arena.rs`
  (template build at upload) and a new per-frame instantiate dispatch
  (indirect, GPU count).

---

## 6. PTLAS integration

`ptlas_fill.wgsl` `fill_incremental`: an animated instance's geometry
changes every frame, so it must always be re-specified. Add an
`animated_flag` binding and force the write:

```
if animated_flag[slot] == 1u { /* always write record */ }
else if force_all == 0 && geometry_dirty[geom] == 0 { move-check ... }
```

`instance_blas_address[slot]` for animated instances is written by
`BuildAnimatedBlas` (per-instance BLAS), not `blas_sharing`. Keep a
single slot-indexed address buffer; the animated path writes its
slots, the sharing path writes the rest.

---

## 7. CPU / extract work

- **Skin extract** (main thread, `ExtractSchedule`): for each
  `RaytracingMesh3d` with `SkinnedMesh`, on add/change emit the joint
  palette (`joints[k] → GpuSlot<TransformGraph>`) and mark the instance
  animated. Reuse the `RtInstanceChanges` delta machinery in
  `instance/instance_manager.rs`.
- **Inverse-bind upload**: when an animated `ClusterMesh` first
  uploads, also upload its `SkinnedMeshInverseBindposes` (validate
  `len == inverse_bind_count`).
- **Active-slot list** (render world, before Deform): collect active
  animated instances (cap `MAX_ANIMATED_INSTANCES`), assign deform-pool
  slots, build the per-slot GPU table the deform/instantiate dispatches
  read. Mirrors aurora's `prepare_animated_active_slots`.

---

## 8. Milestones

**M1 — one animated mesh, finest LOD, on screen. ✅ DONE + RTX-validated.**

**M2 — robustness + multiple instances. ✅ DONE.**
- N animated instances up to the cap: works (cap 8).
- Skeleton-change re-extract: `extract_animated_skins` rebuilds every frame, so
  this is automatic. Despawn cleanup: the animated path holds no per-slot
  persistent allocation (deform-pool slot = active index, reused; `animated_table`
  diff-maintained), so nothing to free.
- Graceful fallback + validation: a mesh over `MAX_VERTS_PER_ANIMATED_MESH`, or
  instances over `MAX_ANIMATED_INSTANCES`, fall back to the static rest-pose path
  with a one-shot `warn!` (no silent corruption). [Remaining: a per-mesh
  finest-LOD-cluster cap guard vs `MAX_CLUSTERS_PER_ANIMATED_MESH` — low risk;
  fox-scale meshes are well under.]

**M3 — LOD for animated instances. ✅ DONE (reuses the static LOD logic).**
- The static cut is pose-independent (a function of instance transform + rest-pose
  root sphere + camera, not the skin), so animated instances reuse it directly:
  `blas_sharing::classify` writes each instance's object-space error budget
  (`e_ideal`, unbanded) into a slot-indexed `instance_e_build` buffer (sharing
  binding 14), and `instantiate.wgsl` runs the **same** `selector.wgsl` accept rule
  (`own_fits ∧ ¬parent_fits`, leaf fallback) over the geometry's groups to pick
  which cluster templates to instantiate — instead of always finest. The deform
  skins all LOD levels' verts, so any selected cluster's verts are already in the
  pool; indexing stays consistent through resolve. Animated instances still get
  per-instance CLAS/BLAS (no bucket sharing) — only the *selection* is reused.
- Bake fix (`from_mesh.rs` orphan repair): a group with no coarser-level parent
  (single-LOD mesh, or a coarsest-level group) now keeps `parent_group = NONE`
  instead of being back-filled to a SAME-level group. The old `unwrap_or(root)`
  fallback made the runtime cut treat that same-level group as a coarser parent
  and cull everything else → single-LOD meshes (the glTF fox) rendered as one
  cluster. The cut also defensively ignores any parent that isn't strictly
  coarser (`parent.lod_level > group.lod_level`). This also hardens the static
  selector, which had the same latent coarse-end hole (never exposed because the
  fox's static shared BLAS is unused).
- ⬜ Remaining: tighten per-cluster bloat AABBs (30%-diagonal default in
  `from_mesh.rs`); per-mesh selected-cluster-count cap guard.

**M4 — quality + perf.**
- ✅ Tangent deform — deform skins the tangent (rotate by the blended skin matrix,
  premultiply inverse instance world), writes `deform.tangents`; resolve reads it
  for animated hits (scene-group binding 17). Correct for normal-mapped skinned
  meshes.
- ⬜ Motion vectors — needs a previous-frame deform pool (ping-pong) so resolve's
  `previous_frame_world_position` uses last frame's deformed verts, not current
  verts × previous transform. Subtlety: `deform_pool_base` = active index, which
  can change frame-to-frame if the active set reorders — the previous-frame table
  must carry the instance's previous base. Most invasive remaining item; without
  it, animated geometry ghosts under temporal denoising (DLSS RR).
- ⬜ Motion-threshold skip — skip re-instantiate/BLAS when an instance's joints
  didn't move this frame.

---

## 9. Risks / open questions

- **Local-space premultiply vs world-space + identity transform.**
  Premultiply keeps resolve unchanged and gives a clean motion-vector
  path later, at the cost of an in-shader affine inverse. Recommend
  premultiply (matches aurora); revisit if the inverse is a problem.
- **Instantiate arena sizing / aliasing.** Per-frame instantiated CLAS
  arena must be sized for worst-case active clusters; sparse-back and
  alias each frame like `blas_rebuild.scratch`.
- **Barrier correctness across raw-VK builds.** Deform (compute) →
  instantiate (raw VK) → animated BLAS (raw VK) → PTLAS (raw VK) all
  need global AS/compute barriers between them; follow the existing
  pre/post-barrier pattern in `blas_rebuild.rs` and `clas_arena.rs`.
- **Joint world transforms in table space.** Confirm Bevy's joint
  entities carry world-space `GlobalTransform` consistent with the
  inverse-bind convention (skin = jointWorld · invBind). They should,
  since the table propagates the same hierarchy Bevy skinning reads.
- **Caps.** `MAX_ANIMATED_INSTANCES` / `MAX_VERTS_PER_ANIMATED_MESH`
  are hard limits; log on overflow (don't silently drop).

---

## 10. File-level work summary

New:
- `src/accel/deform.rs` + `deform.wgsl`
- `src/accel/instantiate.rs` + `instantiate.wgsl`
- `src/accel/animated_blas.rs` (per-instance BLAS build)
- skin/active-slot extract (extend `instance/`)

Modified:
- `src/lib.rs` — `SolariClusterSystems` (+Deform, +InstantiateClas, +BuildAnimatedBlas)
- `src/accel/mod.rs` — stage ordering + dispatch registration
- `src/geometry/clas_arena.rs` — template build for animated meshes
- `src/geometry/mesh_manager.rs` — inverse-bind upload
- `src/instance/gpu_instances.rs` — animated/deform/palette columns
- `src/instance/instance_manager.rs` — skin extract + active-slot list
- `src/bindings/raytracing_scene_bindings.wgsl` — conditional fetch + new bindings
- `src/accel/ptlas_fill.wgsl` — animated force-rewrite
- `src/bindings/binder.rs` / `bind_groups.rs` — bind the new buffers

Already done (asset/bake/upload): `ClusterMesh` joint streams +
`cluster_bloat_aabbs` + `inverse_bind_count` (`geometry/asset.rs`),
bake extraction (`geometry/from_mesh.rs`), GPU joint-stream upload
(`geometry/mesh_manager.rs`).
