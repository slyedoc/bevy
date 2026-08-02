// Shadow-ray miss program. A visibility ray traced from a closest-hit toward a
// light (with RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_TERMINATE_ON_FIRST_HIT)
// invokes NO closest-hit; it either hits an occluder (terminates, no shader) or
// reaches the light unobstructed and lands here. So "this program ran" == "the
// light is visible": clear the occlusion flag. This is the NVIDIA-canonical DI
// visibility test — the traversal lives in the fixed-function unit, not in the
// closest-hit's registers.
//

// MUST MATCH ShadowPayload in rt_payload.wgsl: two u32 words at offsets 0/4.
// Payload linkage across the naga/Slang stage boundary is by trace-call
// operand + struct layout only (naga emits no payload Location decorations).
struct ShadowPayload
{
    // First word mirrors RtPayload.anyhit_count so ahit_alpha's single-word
    // alias targets the counter whichever ray type invoked it.
    uint anyhit_count;
    uint occluded;
}

[shader("miss")]
void miss_shadow(inout ShadowPayload payload)
{
    payload.occluded = 0;
}
