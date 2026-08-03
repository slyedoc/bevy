// Directional-light direction resolve.
//
// Each directional light is a node in the GPU transform table, so its world
// transform already lives in the propagation `world` buffer. This pass fills the
// only transform-derived field of `GpuDirectionalLight` — `direction_to_light` —
// from `world[slot]`, so the CPU never needs the light's `GlobalTransform`
// (the settings fields are filled CPU-side; this only patches the direction).
//
// One thread per directional light. Runs after transform propagation (world
// ready) and before the path tracer reads `directional_lights`.

struct GpuDirectionalLight {
    float3 direction_to_light;
    float cos_theta_max;
    float3 luminance;
    float inverse_pdf;
}

struct ResolveParams {
    uint light_count;
    uint node_count;
    // X workgroup count of the 2D-split dispatch (X capped at 65535, rest in
    // Y) — flat-index reconstruction below.
    uint groups_x;
    uint _pad;
}

[[vk::push_constant]] ResolveParams params;

[[vk::binding(0, 0)]] StructuredBuffer<float4> world; // 3 per node (mat3x4 rows)
[[vk::binding(1, 0)]] StructuredBuffer<uint> slots;   // transform slot per light
[[vk::binding(2, 0)]] RWStructuredBuffer<GpuDirectionalLight> lights;

[shader("compute")]
[numthreads(64, 1, 1)]
void resolve(uint3 gid: SV_DispatchThreadID) {
    // Flat index across the 2D-split dispatch.
    let i = gid.x + gid.y * params.groups_x * 64;
    if (i >= params.light_count) {
        return;
    }
    let slot = slots[i];
    if (slot >= params.node_count) {
        return; // unassigned / out-of-range node — leave last value.
    }
    let s = slot * 3;
    // `GlobalTransform::back()` = +Z basis = the z column = (row0.z, row1.z,
    // row2.z) in the mat3x4 row layout. Normalize in case of scale.
    let z_axis = float3(world[s].z, world[s + 1].z, world[s + 2].z);
    lights[i].direction_to_light = normalize(z_axis);
}
