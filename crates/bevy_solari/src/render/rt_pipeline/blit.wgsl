// Copies the RT-pipeline's per-pixel output storage buffer (written by the raw
// cmd_trace_rays raygen shader, row-major `y*width+x`) into the view's HDR
// storage texture. A normal wgpu compute pass, so the view target stays fully
// wgpu-layout-tracked — the raw write only ever touches the buffer.
@group(0) @binding(0) var<storage, read> rt_output: array<vec4<f32>>;
@group(0) @binding(1) var view_output: texture_storage_2d<rgba16float, write>;

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
    textureStore(view_output, vec2<i32>(gid.xy), vec4<f32>(c.rgb, 1.0));
}
