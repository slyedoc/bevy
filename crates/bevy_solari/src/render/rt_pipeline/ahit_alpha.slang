// Any-hit alpha-cutout test — the RT-pipeline analog of the inline rayQuery
// candidate test (`scene_bindings::alpha_test`), for foliage / fences / cutouts.
//
// Attached to the OPAQUE triangle hit group. The hardware runs an any-hit ONLY for
// non-opaque geometry, and PTLAS marks exactly the alpha-masked instances
// `FORCE_NO_OPAQUE` (`material.traversal_alpha_cutoff() >= 0`), so opaque geometry
// commits in hardware and pays nothing here. When the base-color alpha at the hit
// UV is below the material's cutoff the texel is a hole, so `IgnoreHit()`
// rejects the candidate and the ray continues through it — for primary, bounce, AND
// shadow rays (shadow rays are not force-opaque, so foliage casts cutout
// shadows; see chit_opaque `SHADOW_RAY_FLAGS`).
//
// Regenerate SPIR-V (from crates/bevy_solari/src/render/rt_pipeline/):
//   slangc ahit_alpha.slang -target spirv -o ahit_alpha.spv

import scene_resolve;

// Single-word payload alias: `RtPayload` and `ShadowPayload` both keep
// `anyhit_count` as their first word, so this safely targets the counter whichever
// ray invoked the any-hit. Incrementing it per invocation drives the
// OMM-effectiveness heatmap (raygen colormaps the per-pixel total) — with OMM, the
// RT cores resolve opaque/transparent micro-regions in hardware and the any-hit
// never runs there, so OMM-covered foliage reads "cold".
struct AlphaHitPayload
{
    uint anyhit_count;
}

[shader("anyhit")]
void ahit_alpha(inout AlphaHitPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    // Count every any-hit invocation (the cost OMM removes), pass or fail.
    payload.anyhit_count += 1;
    if (!alpha_passes(
            material_ids[InstanceIndex()], cluster_id(), PrimitiveIndex(),
            attribs.barycentrics))
    {
        IgnoreHit();
    }
}
