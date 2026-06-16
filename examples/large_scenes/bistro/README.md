# Bistro Example

Download Amazon Lumberyard version of bistro [from developer.nvidia.com](<https://developer.nvidia.com/orca/amazon-lumberyard-bistro>) (or see link below for processed glTF files with instancing)

Reexport BistroExterior.fbx and BistroInterior_Wine.fbx as GLTF files (in .gltf + .bin + textures format). Move the files into the respective bistro_exterior and bistro_interior_wine folders.

## Ray-traced (`solari`) asset prep

`prepare_bistro.py` (headless Blender) combines the two FBX files into one
`bistro/Bistro.glb`. For the full ray-traced path, two post-processing passes
on that GLB matter for performance — run them in order:

1. **Texture compression** — Blender embeds textures as PNG, which upload as
   uncompressed RGBA8 (4 B/texel, no mips). Transcode to UASTC `KHR_texture_basisu`
   KTX2 (→ BC7 ~1 B/texel + mips on the GPU). Requires KTX-Software (`toktx`) on
   PATH and the `basis-universal` cargo feature. **Use UASTC, not ETC1S — bevy's
   ktx2 loader does not support ETC1S/BasisLZ supercompression.**
   ```
   npx @gltf-transform/cli uastc Bistro.glb Bistro_ktx2.glb --level 0 --zstd 18
   ```
2. **Alpha-mode reclassification** — the FBX→glTF roundtrip flattens almost every
   material to `BLEND`, which forces the ray tracer through the any-hit alpha-test
   loop for ~95% of the (actually opaque) scene. `reclassify_alpha.py` restores
   `OPAQUE` from the real base-color alpha content, leaving cutout foliage/decals
   untouched. Re-run it after each re-encode (it patches in place):
   ```
   python reclassify_alpha.py Bistro.glb Bistro_ktx2.glb
   ```

The example loads `bistro/Bistro_ktx2.glb`.

- Press 1, 2 or 3 for various camera positions.
- Press B for benchmark.
- Press to animate camera along path.

Run with texture compression while caching compressed images to disk for faster startup times:
`cargo run -p bistro --release --features mipmap_generator/compress -- --cache`

```console
Options:
--no-gltf-lights  disable glTF lights
--minimal         disable bloom, AO, AA, shadows
--compress        compress textures (if they are not already, requires
                  compress feature)
--low-quality-compression
                  if low_quality_compression is set, only 0.5 byte/px formats
                  will be used (BC1, BC4) unless the alpha channel is in use,
                  then BC3 will be used. When low quality is set, compression
                  is generally faster than CompressionSpeed::UltraFast and
                  CompressionSpeed is ignored.
--cache           compressed texture cache (requires compress feature)
--count           quantity of bistros
--spin            spin the bistros and camera
--hide-frame-time don't show frame time
--deferred        use deferred shading
--no-frustum-culling
                  disable all frustum culling. Stresses queuing and batching
                  as all mesh material entities in the scene are always drawn.
--no-automatic-batching
                  disable automatic batching. Skips batching resulting in
                  heavy stress on render pass draw command encoding.
--no-view-occlusion-culling
                  disable gpu occlusion culling for the camera
--no-shadow-occlusion-culling
                  disable gpu occlusion culling for the directional light
--no-indirect-drawing
                  disable indirect drawing.
--no-cpu-culling  disable CPU culling.
--help, help      display usage information
```

[Alternate processed files with instancing (glTF files on discord):](https://discord.com/channels/691052431525675048/1237853896471220314/1237859248067575910)

- Fixed most of the metallic from fbx issue by using a script that makes everything dielectric unless it has metal in the name of the material
- Made the plants alpha clip instead of blend
- Setup the glassware/liquid materials correctly
- Mesh origins are at individual bounding box center instead of world origin
- Removed duplicate vertices (There were lots of odd cases, often making one instance not match another that would otherwise exactly match)
- Made the scene use instances (unique mesh count 3880 -> 1188)
- Removed 2 cases where duplicated meshes were overlapping
- Setup some of the interior/exterior lights with actual sources
- Setup some basic fake GI
- Use included scene HDRI for IBL
