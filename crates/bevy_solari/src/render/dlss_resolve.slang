// DLSS Ray Reconstruction guide resolve: unpacks the rt_pipeline's per-pixel guide
// STORAGE BUFFERS (written chit-direct by the trace) into the TEXTURES the
// `dlss_wgpu` ray-reconstruction API consumes. A raw heap dispatch — the
// trace's trailing SHADER_WRITE→SHADER_READ barrier makes the buffer writes visible
// here, and the dispatch transitions the guide textures to their storage state
// (GENERAL) through wgpu's tracker first, so RR's reads stay wgpu-layout-tracked.
//
// Packing produced by the trace (each `float4` per pixel, row-major y*width+x):
//   normal_roughness : world normal.xyz + linear roughness (.w)   [Packed]
//   diffuse          : diffuse albedo.xyz + linear view depth (.w)
//   specular         : specular albedo (F0).xyz + spec hit dist (.w)
//   motion           : screen-space motion.xy

[[vk::binding(0, 0)]] StructuredBuffer<float4> nr_buf;
[[vk::binding(1, 0)]] StructuredBuffer<float4> diffuse_buf;
[[vk::binding(2, 0)]] StructuredBuffer<float4> specular_buf;
[[vk::binding(3, 0)]] StructuredBuffer<float4> motion_buf;
[[vk::image_format("r32f")]]
[[vk::binding(4, 0)]] RWTexture2D<float4> out_depth;
[[vk::image_format("rgba16f")]]
[[vk::binding(5, 0)]] RWTexture2D<float4> out_normal_roughness;
[[vk::image_format("rgba8")]]
[[vk::binding(6, 0)]] RWTexture2D<float4> out_diffuse;
[[vk::image_format("rgba8")]]
[[vk::binding(7, 0)]] RWTexture2D<float4> out_specular;
[[vk::image_format("rg16f")]]
[[vk::binding(8, 0)]] RWTexture2D<float4> out_motion;
[[vk::image_format("r32f")]]
[[vk::binding(9, 0)]] RWTexture2D<float4> out_spec_hit_distance;

[shader("compute")]
[numthreads(8, 8, 1)]
void resolve(uint3 gid: SV_DispatchThreadID) {
    uint width;
    uint height;
    out_depth.GetDimensions(width, height);
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let p = gid.xy;
    let idx = gid.y * width + gid.x;

    let nr = nr_buf[idx];
    let diffuse = diffuse_buf[idx];
    let specular = specular_buf[idx];
    let motion = motion_buf[idx];

    out_normal_roughness[p] = nr;
    out_diffuse[p] = float4(diffuse.xyz, 1.0);
    out_depth[p] = float4(diffuse.w, 0.0, 0.0, 0.0);
    out_specular[p] = float4(specular.xyz, 1.0);
    out_spec_hit_distance[p] = float4(specular.w, 0.0, 0.0, 0.0);
    out_motion[p] = float4(motion.xy, 0.0, 0.0);
}
