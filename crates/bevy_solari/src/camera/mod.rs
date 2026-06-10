//! Notes on RT-correct visibility (no plugin yet — see findings).
//!
//! `RaytracingMesh3d` already `#[require(NoCpuCulling)]` (see `bindings/types.rs`),
//! so the RT meshes are *already* excluded from `check_visibility_cpu_culling`
//! (its query is `Without<NoCpuCulling>`). The visibility cost is therefore NOT
//! mesh frustum culling — it's elsewhere, and trickier than transforms:
//!
//! - `check_visibility_cpu_culling` (~1.2ms) iterates the `Without<NoCpuCulling>`
//!   set — the camera, UI, lights, and the (mesh-less) intermediate hierarchy
//!   nodes — unconditionally each frame, setting `ViewVisibility`. The camera's
//!   `ViewVisibility` here is load-bearing: `SolariCamera` is extracted via
//!   `ExtractComponentPlugin`, which only extracts entities whose
//!   `ViewVisibility` is true — so naively disabling this system blanks the
//!   screen.
//! - `check_visibility_gpu_culling` (~1.1ms) and `visibility_propagate_system`
//!   (~3ms) are change-driven (`Or<(Changed<…>, …)>`) yet cost real time in a
//!   static scene — strongly suggesting their `Or<Changed>` filters scan all
//!   ~2M entities per-frame without table-level change-skip.
//!
//! So a correct fix must (a) keep the camera's `ViewVisibility`, and (b) cut the
//! per-frame iteration/scan over millions of nodes — likely by splitting the
//! `Or<Changed>` filters and/or taking the mesh-less intermediate nodes out of
//! the culling iteration. Deferred until scoped properly.
