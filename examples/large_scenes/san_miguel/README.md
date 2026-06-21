# San Miguel Example

A large, foliage-heavy courtyard scene — Guillermo M. Leal Llaguno's **San
Miguel**, the classic ray-tracing benchmark for *alpha-tested geometry*. Its
dense cutout leaves are exactly the workload that exercises the ray tracer's
any-hit alpha-test path, so it makes a far better stress test than a mostly
opaque scene that renders too fast to profile.

## Download

Get the scene from Morgan McGuire's Computer Graphics Archive:
<https://casual-effects.com/data/> — listed as **"San Miguel"** (by Guillermo M.
Leal Llaguno). It ships as `san-miguel.obj` (+ `.mtl` + `textures/`), roughly
10M triangles. A lighter `san-miguel-low-poly.obj` is bundled if the full mesh
is too heavy for your GPU.

## Asset prep (ray-traced `solari` path)

bevy can't load OBJ directly, so convert to glTF and run two post-processing
passes that matter for performance. Run them in order:

The prep scripts (`prepare_san_miguel.py`, `reclassify_alpha.py`) live with the
assets, outside this repo — `$BEVY_ASSET_ROOT/scripts/`. Below, `$SCRIPTS` is
that directory and `$DIR` is `$BEVY_ASSET_ROOT/assets/san_miguel` (create it
first). All three passes write into `$DIR`.

1. **OBJ → GLB** with headless Blender:
   ```
   blender --background --python $SCRIPTS/prepare_san_miguel.py -- /path/to/san-miguel.obj $DIR/SanMiguel.glb
   ```
   Imports the OBJ + materials, adds planar UVs to any mesh missing them
   (needed for tangents), drops cameras/lights, and exports one `SanMiguel.glb`.

2. **Texture compression** — Blender embeds textures uncompressed (4 B/texel, no
   mips). Transcode to UASTC `KHR_texture_basisu` KTX2 (→ BC7 ~1 B/texel + mips
   on the GPU). Requires KTX-Software (`toktx`) on PATH and the `basis-universal`
   cargo feature. **Use UASTC, not ETC1S — bevy's ktx2 loader does not support
   ETC1S/BasisLZ supercompression.**
   ```
   npx @gltf-transform/cli uastc $DIR/SanMiguel.glb $DIR/SanMiguel_ktx2.glb --level 0 --zstd 18
   ```

3. **Alpha-mode reclassification** — restores `OPAQUE` on materials the
   OBJ→glTF roundtrip left non-opaque, while keeping the real foliage cutouts as
   `MASK`. Without it, solid stucco/stone pays any-hit alpha-test traversal it
   doesn't need. Re-run after each re-encode (it patches in place):
   ```
   python $SCRIPTS/reclassify_alpha.py $DIR/SanMiguel.glb $DIR/SanMiguel_ktx2.glb
   ```

## Run

The example loads `san_miguel/SanMiguel_ktx2.glb`. Keep the large assets out of
the bevy repo and point bevy at them with `BEVY_ASSET_ROOT` (bevy resolves assets
under `$BEVY_ASSET_ROOT/assets/`), so the final file lives at:

```
$BEVY_ASSET_ROOT/assets/san_miguel/SanMiguel_ktx2.glb
```

Then:

```
BEVY_ASSET_ROOT=/path/to/solari_files cargo run -p san_miguel --release --features solari
```

(Add `dlss` for DLSS upscaling/denoising if available.)

San Miguel has ~1000 unique high-poly meshes, and the ray-tracing cluster bake
runs on the CPU. Baking them all at once would freeze the window for seconds, so
the example streams the bake in over several frames behind a loading overlay that
reports progress. Tune the rate with `--bake-per-frame N` (default 16) — lower
keeps the window more responsive, higher finishes faster. If the bake is too slow
or memory-heavy, convert `san-miguel-low-poly.obj` instead.

- Press `1`, `2`, `3` for preset camera positions.
- Press `I` to print the current camera transform (record your own positions and
  paste them into `CameraPositions::default` / `ANIM_CAM` in `src/main.rs`).
- Press `Space` to toggle the fly-through path.
- Press `B` to run the benchmark.
- `--count N` tiles N copies of the scene in a grid to drive the frame rate down
  for profiling; `--spin` orbits the scene/camera.

The light panel (top right) steers the sun azimuth/elevation and an emissive
boost (defaults to 1.0 — San Miguel is daylit with little authored emissive).
