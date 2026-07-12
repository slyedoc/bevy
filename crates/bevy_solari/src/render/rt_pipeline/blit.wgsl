// Copies the RT-pipeline's per-pixel output storage buffer (written by the raw
// cmd_trace_rays raygen shader, row-major `y*width+x`) into the view's HDR
// storage texture. A normal wgpu compute pass, so the view target stays fully
// wgpu-layout-tracked — the raw write only ever touches the buffer.
@group(0) @binding(0) var<storage, read> rt_output: array<vec4<f32>>;
@group(0) @binding(1) var view_output: texture_storage_2d<rgba16float, write>;
// Frozen reference snapshot (freeze/diff harness). Bound to `rt_output` itself
// when nothing is frozen; `params.x` gates the mode so that alias is never read.
@group(0) @binding(2) var<storage, read> frozen: array<vec4<f32>>;
// .x = mode (0 passthrough, 1 diff heatmap), .y = diff scale, .z = camera
// exposure (the output buffer is physical radiance; 1.0 while a debug view
// paints raw non-radiance values).
@group(0) @binding(3) var<uniform> params: vec4<f32>;

// Black -> blue -> green -> yellow -> red ramp for |diff| luminance.
fn diff_ramp(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let blue = smoothstep(0.0, 0.25, x) - smoothstep(0.25, 0.5, x);
    let green = smoothstep(0.25, 0.5, x) - smoothstep(0.75, 1.0, x);
    let red = smoothstep(0.5, 0.75, x);
    return vec3<f32>(red, green, blue);
}

@compute @workgroup_size(8, 8, 1)
fn blit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(view_output);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let index = gid.y * dims.x + gid.x;
    // `.w` carries the primary-hit depth (gizmo-depth bridge), not alpha — force
    // alpha to 1.0 so it never tints the displayed image.
    let c = rt_output[index];
    var rgb = c.rgb * params.z;
    if params.x == 1.0 {
        let d = abs(c.rgb - frozen[index].rgb) * params.z;
        let lum = dot(d, vec3<f32>(0.2126, 0.7152, 0.0722));
        rgb = diff_ramp(lum * params.y);
    }
    textureStore(view_output, vec2<i32>(gid.xy), vec4<f32>(rgb, 1.0));
}
