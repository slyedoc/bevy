// Any-hit alpha-cutout test — the RT-pipeline analog of the inline rayQuery
// candidate test (`scene_bindings::alpha_test`), for foliage / fences / cutouts.
//
// Attached to the OPAQUE triangle hit group. The hardware runs an any-hit ONLY for
// non-opaque geometry, and PTLAS marks exactly the alpha-masked instances
// `FORCE_NO_OPAQUE` (`material.traversal_alpha_cutoff() >= 0`), so opaque geometry
// commits in hardware and pays nothing here. When the base-color alpha at the hit
// UV is below the material's cutoff the texel is a hole, so `ignoreIntersection()`
// rejects the candidate and the ray continues through it — for primary, bounce, AND
// shadow rays (shadow rays are no longer force-opaque, so foliage casts cutout
// shadows; see chit_opaque `SHADOW_RAY_FLAGS`).
enable wgpu_ray_tracing_pipeline;
enable primitive_index;

#import bevy_solari::scene_bindings::{alpha_passes, material_ids}

// Single-word payload alias: `RtPayload` and `ShadowPayload` both keep
// `anyhit_count` as their first word, so this safely targets the counter whichever
// ray invoked the any-hit. Incrementing it per invocation drives the
// OMM-effectiveness heatmap (raygen colormaps the per-pixel total) — with OMM, the
// RT cores resolve opaque/transparent micro-regions in hardware and the any-hit
// never runs there, so OMM-covered foliage reads "cold".
struct AlphaHitPayload {
    anyhit_count: u32,
}
var<incoming_ray_payload> payload: AlphaHitPayload;
// Driver-provided triangle barycentrics (the fixed-function intersection's u, v).
var<hit_attribute> bary: vec2<f32>;

@any_hit
@incoming_payload(payload)
fn ahit_alpha(
    // InstanceId (TLAS instance index) — the key `material_ids` is indexed by, same
    // as the inline path's `hit.instance_index`.
    @builtin(instance_id) instance_id: u32,
    // ClusterIDNV — the canonical cluster index into `clusters[]`, exactly as the
    // closest-hit uses it. NOT `geometry_index`: in the pipeline that's the raw
    // per-CLAS geometry index (≈0), which would read the wrong cluster's UVs and
    // fail the alpha test everywhere. (The inline rayQuery's `geometry_index`
    // returns the baked cluster id — a different mechanism.)
    @builtin(cluster_id) cluster_id: u32,
    // Cluster-local triangle index, pairing with `cluster_id` (as in the chit).
    @builtin(primitive_index) primitive_index: u32,
) {
    // Count every any-hit invocation (the cost OMM removes), pass or fail.
    payload.anyhit_count += 1u;
    if !alpha_passes(material_ids[instance_id], cluster_id, primitive_index, bary) {
        ignoreIntersection();
    }
}
