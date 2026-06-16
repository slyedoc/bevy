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
    direction_to_light: vec3<f32>,
    cos_theta_max: f32,
    luminance: vec3<f32>,
    inverse_pdf: f32,
}

struct ResolveParams {
    light_count: u32,
    node_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> world: array<vec4<f32>>;        // 3 per node (mat3x4 rows)
@group(0) @binding(1) var<storage, read> slots: array<u32>;             // transform slot per light
@group(0) @binding(2) var<storage, read_write> lights: array<GpuDirectionalLight>;
@group(0) @binding(3) var<uniform> params: ResolveParams;

@compute @workgroup_size(64)
fn resolve(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    // Flat index across a 2D-split dispatch (X capped at 65535, rest in Y).
    let i = gid.x + gid.y * num_workgroups.x * 64u;
    if i >= params.light_count {
        return;
    }
    let slot = slots[i];
    if slot >= params.node_count {
        return; // unassigned / out-of-range node — leave last value.
    }
    let s = slot * 3u;
    // `GlobalTransform::back()` = +Z basis = the z column = (row0.z, row1.z, row2.z)
    // in the mat3x4 row layout. Normalize in case of scale.
    let z_axis = vec3<f32>(world[s].z, world[s + 1u].z, world[s + 2u].z);
    lights[i].direction_to_light = normalize(z_axis);
}
