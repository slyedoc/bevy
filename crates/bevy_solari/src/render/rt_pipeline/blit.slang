// Copies the RT-pipeline's per-pixel output storage buffer (written by the raw
// cmd_trace_rays raygen shader, row-major `y*width+x`) into the view's HDR
// storage texture. A raw heap dispatch: the dispatch transitions the view
// target to its storage state (GENERAL) through wgpu's tracker first, so the
// rest of the frame stays wgpu-layout-tracked.

[[vk::binding(0, 0)]] StructuredBuffer<float4> rt_output;
[[vk::image_format("rgba16f")]]
[[vk::binding(1, 0)]] RWTexture2D<float4> view_output;
// Frozen reference snapshot (freeze/diff harness). Bound to `rt_output` itself
// when nothing is frozen; `params.x` gates the mode so that alias is never read.
[[vk::binding(2, 0)]] StructuredBuffer<float4> frozen;

// .x = mode (0 passthrough, 1 diff heatmap), .y = diff scale, .z = camera
// exposure (the output buffer is physical radiance; 1.0 while a debug view
// paints raw non-radiance values).
[[vk::push_constant]] float4 params;

// Black -> blue -> green -> yellow -> red ramp for |diff| luminance.
float3 diff_ramp(float t) {
    let x = clamp(t, 0.0, 1.0);
    let blue = smoothstep(0.0, 0.25, x) - smoothstep(0.25, 0.5, x);
    let green = smoothstep(0.25, 0.5, x) - smoothstep(0.75, 1.0, x);
    let red = smoothstep(0.5, 0.75, x);
    return float3(red, green, blue);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void blit(uint3 gid: SV_DispatchThreadID) {
    uint width;
    uint height;
    view_output.GetDimensions(width, height);
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let index = gid.y * width + gid.x;
    // `.w` carries the primary-hit depth (gizmo-depth bridge), not alpha — force
    // alpha to 1.0 so it never tints the displayed image.
    let c = rt_output[index];
    var rgb = c.rgb * params.z;
    if (params.x == 1.0) {
        let d = abs(c.rgb - frozen[index].rgb) * params.z;
        let lum = dot(d, float3(0.2126, 0.7152, 0.0722));
        rgb = diff_ramp(lum * params.y);
    }
    view_output[gid.xy] = float4(rgb, 1.0);
}
