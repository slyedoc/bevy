# What's left of wgpu — the drop-wgpu decision document

Written 2026-08-03, at the completion of the WGSL→Slang compute migration.
Every solari compute dispatch is a layout-free Slang heap kernel; **zero
compute passes or compute pipelines go through wgpu/naga_oil** (the `naga` +
`naga_oil` dependencies are gone from the crate). This is the complete
inventory of what wgpu still does, as input to the shell-vs-guest-vs-drop
decision.

## Inside bevy_solari

1. **One raster pipeline: `gizmo_depth`** — the fullscreen frag-depth bridge
   (RT G-buffer → hardware depth so gizmos occlude). The only remaining
   `PipelineCache` entry and the only reason `SolariPipelineRegistry` /
   `SolariPipelines` still exist. Plus one empty attachment pass
   (`rt_blit_init`) that exists solely to pull wgpu's lazy zero-init ahead
   of the raw blit.
2. **Buffer staging + lifetime** — `RawBufferVec` / `StorageBuffer` /
   `write_buffer_with` uploads (column deltas, work lists, params UBOs), and
   wgpu's deferred-destruction tracking, which keeps staged-written buffers
   alive across the submissions that raw-read them. Going raw means owning a
   universal retire discipline (`gpu/retire.rs` is the seed).
3. **Texture upload + residency** — `Image` assets → `RenderAssets<GpuImage>`
   for scene/material textures, displacement maps, skyboxes. The heap mirrors
   *descriptors* from these; the upload pipeline is wgpu's.
4. **Image-layout transitions via the tracker** — 6 `transition_resources`
   sites (skybox, view target, DLSS guides, atmosphere cube, displacement
   maps). Known gap: nothing transitions scene MATERIAL textures for the raw
   trace since the M2c flip (they sit in wgpu's COPY_DST state post-upload;
   renders correctly on NVIDIA today — latent, flagged).
5. **Readbacks** — `map_async` staging (transform readback, picking, PTLAS
   validate/nulls) and bevy's `Readback` machinery.
6. **Readiness signals** — the scene / cluster-scene / columns wgpu bind
   groups are still *built* but never bound by any pass; dispatches use their
   existence as "scene ready". Deletable with a small readiness refactor.
7. **The device itself** — instance/adapter/device/queue creation, and every
   raw handle obtained through `as_hal` seams.

## Outside bevy_solari (the real anchors)

8. **Swapchain/present + winit** — bevy_render's window pipeline.
9. **bevy_feathers / bevy_ui / gizmos** — the inspector, debug cards, and the
   sly_tree editor. The investment that decides the question.
10. **The fork (wgpu-slang)** — permanent maintenance: naga builtins,
    `Tlas::from_hal`, sampler/view `as_hal` accessors, aftermath wiring,
    auto-enabled extensions; trunk rebase blocked by the ash split.

## The options

- **A — shell wgpu** (today's end-state): wgpu keeps 1–10. Cheapest; fork
  maintenance continues.
- **B — full drop**: own device/swapchain/uploader/staging/retire; keep bevy
  ECS/app/window; **loses feathers/gizmos** (9) unless a replacement UI path
  is built.
- **C — invert (leading)**: solari owns device/queue/frame-loop/present raw;
  a wgpu guest device adopts the raw VkDevice (wgpu-hal supports external
  handles) purely so 9 can raster into an overlay we composite. Items 2–6
  migrate to our own staging/upload/retire at whatever pace; the fork shrinks
  toward what UI raster needs.

The deciding question is item 9: how much is feathers worth keeping.
