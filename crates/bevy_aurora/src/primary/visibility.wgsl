//! Aurora primary visibility compute shader.
//!
//! For each pixel, traces a camera ray against Aurora's wrapped TLAS and
//! writes a debug colour straight into the camera's view target. Hit pixels
//! get a colour derived from the world-space hit position (so the bunny's
//! shape + a sense of depth is visible without any real lighting); miss
//! pixels get a dim "sky" colour.

enable wgpu_ray_query;

#import bevy_render::view::View

@group(0) @binding(0) var view_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var tlas: acceleration_structure;

const RAY_T_MIN: f32 = 0.001;
const RAY_T_MAX: f32 = 100000.0;
const RAY_NO_CULL: u32 = 0xFFu;

@compute @workgroup_size(8, 8, 1)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(view_output);
    if (gid.x >= dims.x) || (gid.y >= dims.y) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);

    // NDC: x in [-1, 1] left→right, y in [-1, 1] bottom→top. UV's y grows
    // downward, so flip.
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    // Unproject the far plane to get a world-space point; ray = point - origin.
    let world_h = view.world_from_clip * vec4<f32>(ndc, 1.0, 1.0);
    let world_pt = world_h.xyz / world_h.w;
    let origin = view.world_position;
    let dir = normalize(world_pt - origin);

    let ray = RayDesc(0u, RAY_NO_CULL, RAY_T_MIN, RAY_T_MAX, origin, dir);
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, ray);
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);

    var color: vec4<f32>;
    if hit.kind != RAY_QUERY_INTERSECTION_NONE {
        let hit_pos = origin + dir * hit.t;
        // Crude debug shading: world-position fractional bits → R/G/B. Lets
        // us see the bunny's shape + per-cluster bands without lighting.
        let c = abs(fract(hit_pos * 0.5));
        color = vec4<f32>(c, 1.0);
    } else {
        color = vec4<f32>(0.04, 0.06, 0.10, 1.0);
    }
    textureStore(view_output, pixel, color);
}
