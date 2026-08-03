// Bake a single-scattering atmosphere into a sky cubemap.
//
// One thread per cube texel: for the texel's world direction, sample the shared
// atmosphere physics (the `atmosphere` module) and write the resulting sky
// radiance into the cube. The pathtracer then samples this cube on a ray miss —
// so the sky both lights the scene (escaped rays) and is the background, no
// separate IBL needed.

import atmosphere;

// The `Atmosphere` params ride the push block (96 B std140/std430-identical
// layout — the same bytes `GpuSolariAtmosphere`'s encase encoding produces).
[[vk::push_constant]] Atmosphere atmo;

[[vk::image_format("rgba16f")]]
[[vk::binding(0, 0)]] RWTexture2DArray<float4> sky;

// Cubemap face + pixel → world direction (matches `textureSample(cube, dir)`).
float3 cube_direction(uint face, float2 uv) {
    float3 dir;
    switch (face) {
        case 0: dir = float3( 1.0, -uv.y, -uv.x); break; // +X
        case 1: dir = float3(-1.0, -uv.y,  uv.x); break; // -X
        case 2: dir = float3( uv.x,  1.0,  uv.y); break; // +Y
        case 3: dir = float3( uv.x, -1.0, -uv.y); break; // -Y
        case 4: dir = float3( uv.x, -uv.y,  1.0); break; // +Z
        default: dir = float3(-uv.x, -uv.y, -1.0); break; // -Z
    }
    return normalize(dir);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void bake(uint3 gid: SV_DispatchThreadID) {
    uint width;
    uint height;
    uint layers;
    sky.GetDimensions(width, height, layers);
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let uv = (float2(gid.xy) + 0.5) / float2(width, height) * 2.0 - 1.0;
    let dir = cube_direction(gid.z, uv);

    let radiance = atmosphere_sky_radiance(atmo, dir);
    sky[uint3(gid.xy, gid.z)] = float4(radiance, 1.0);
}
