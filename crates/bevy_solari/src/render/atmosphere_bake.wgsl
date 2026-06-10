// Bake a single-scattering atmosphere into a sky cubemap.
//
// One thread per cube texel: for the texel's world direction, sample the shared
// atmosphere physics ([`bevy_solari::atmosphere`]) and write the resulting sky
// radiance into the cube. The pathtracer then samples this cube on a ray miss —
// so the sky both lights the scene (escaped rays) and is the background, no
// separate IBL needed.

#import bevy_solari::atmosphere::{Atmosphere, atmosphere_sky_radiance}

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var sky: texture_storage_2d_array<rgba16float, write>;

// Cubemap face + pixel → world direction (matches `textureSample(cube, dir)`).
fn cube_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    var dir: vec3<f32>;
    switch face {
        case 0u: { dir = vec3<f32>( 1.0, -uv.y, -uv.x); } // +X
        case 1u: { dir = vec3<f32>(-1.0, -uv.y,  uv.x); } // -X
        case 2u: { dir = vec3<f32>( uv.x,  1.0,  uv.y); } // +Y
        case 3u: { dir = vec3<f32>( uv.x, -1.0, -uv.y); } // -Y
        case 4u: { dir = vec3<f32>( uv.x, -uv.y,  1.0); } // +Z
        default: { dir = vec3<f32>(-uv.x, -uv.y, -1.0); } // -Z
    }
    return normalize(dir);
}

@compute @workgroup_size(8, 8, 1)
fn bake(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(sky).xy;
    if any(gid.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size) * 2.0 - 1.0;
    let dir = cube_direction(gid.z, uv);

    let radiance = atmosphere_sky_radiance(atmosphere, dir);
    textureStore(sky, gid.xy, gid.z, vec4<f32>(radiance, 1.0));
}
