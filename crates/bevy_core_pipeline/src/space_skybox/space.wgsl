#import bevy_render::view::View
#import bevy_pbr::utils::coords_to_viewport_uv

struct SkyboxUniforms {
	brightness: f32,
}

@group(0) @binding(0) var skybox: texture_cube<f32>;
@group(0) @binding(1) var skybox_sampler: sampler;
@group(0) @binding(2) var<uniform> view: View;
@group(0) @binding(3) var<uniform> uniforms: SkyboxUniforms;

// struct View {
//     view_proj: mat4x4<f32>,
//     unjittered_view_proj: mat4x4<f32>,
//     inverse_view_proj: mat4x4<f32>,
//     view: mat4x4<f32>,
//     inverse_view: mat4x4<f32>,
//     projection: mat4x4<f32>,
//     inverse_projection: mat4x4<f32>,
//     world_position: vec3<f32>,
//     exposure: f32,
//     // viewport(x_origin, y_origin, width, height)
//     viewport: vec4<f32>,
//     frustum: array<vec4<f32>, 6>,
//     color_grading: ColorGrading,
//     mip_bias: f32,
//     render_layers: u32,
// };


fn coords_to_ray_direction(position: vec2<f32>, viewport: vec4<f32>) -> vec3<f32> {
    // Using world positions of the fragment and camera to calculate a ray direction
    // breaks down at large translations. This code only needs to know the ray direction.
    // The ray direction is along the direction from the camera to the fragment position.
    // In view space, the camera is at the origin, so the view space ray direction is
    // along the direction of the fragment position - (0,0,0) which is just the
    // fragment position.
    // Use the position on the near clipping plane to avoid -inf world position
    // because the far plane of an infinite reverse projection is at infinity.
    let view_position_homogeneous = view.inverse_projection * vec4(
        coords_to_viewport_uv(position, viewport) * vec2(2.0, -2.0) + vec2(-1.0, 1.0),
        1.0,
        1.0,
    );
    let view_ray_direction = view_position_homogeneous.xyz / view_position_homogeneous.w;
    // Transforming the view space ray direction by the view matrix, transforms the
    // direction to world space. Note that the w element is set to 0.0, as this is a
    // vector direction, not a position, That causes the matrix multiplication to ignore
    // the translations from the view matrix.
    let ray_direction = (view.view * vec4(view_ray_direction, 0.0)).xyz;

    return normalize(ray_direction);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

//  3 |  2.
//  2 |  :  `.
//  1 |  x-----x.
//  0 |  |  s  |  `.
// -1 |  0-----x.....1
//    +---------------
//      -1  0  1  2  3
//
// The axes are clip-space x and y. The region marked s is the visible region.
// The digits in the corners of the right-angled triangle are the vertex
// indices.
@vertex
fn skybox_vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // See the explanation above for how this works.
    let clip_position = vec4(
        f32(vertex_index & 1u),
        f32((vertex_index >> 1u) & 1u),
        0.25,
        0.5
    ) * 4.0 - vec4(1.0);

    return VertexOutput(clip_position);
}

//Basic hash function for pseudo-random noise
fn hash2(p: vec2<f32>) -> vec2<f32> {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(vec2<f32>(h, h + 1.0)) * 43758.5453);
}

fn hash3(n: vec3<f32>) -> f32 {
    let x = dot(n, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(x) * 43758.5453123);
}

fn random2(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(12.9898, 78.233));
    return fract(sin(h) * 43758.5453);
}

// Function to generate starry sky
fn star_intensity(direction: vec3<f32>) -> f32 {
    let large_scale = 1000.0; // Use a large scale to simulate a large sphere
    let noise_value = hash3(direction * large_scale);
    let star_threshold = 0.998; // Adjust threshold for density of stars
    return step(star_threshold, noise_value);
}

// Smoother step function for more gradual transitions
fn smooth_star_intensity(direction: vec3<f32>, scale: f32) -> f32 {
    let cell = floor(direction * scale);
    let fractional_part = fract(direction * scale);
    let h = hash3(cell);
    return smoothstep(0.98, 0.99, h); // Adjust for finer control
}

@fragment
fn skybox_fragment(in: VertexOutput) -> @location(0) vec4<f32> {


    let ray_direction = coords_to_ray_direction(in.position.xy, view.viewport);
    
    //let uv = mesh.uv * 100.0; // Scale UV coordinates
    //let distance = voronoi(ray_direction.xy * 10.0); // Generate a distance field

    // Color ramp: white if distance is low (star), black otherwise (space)
    let star_color = vec3<f32>(1.0, 1.0, 1.0);
    let space_color = vec3<f32>(0.0, 0.0, 0.0);

    // Generate a random value for the current UV coordinates
    let rand_value = random2(ray_direction.xz);

    // Map the random value to a star size threshold
    let star_size_threshold = 0.01 + rand_value * 0.03; // Random size between 0.001 and 0.003

    // Simple threshold to create stars
    let color = mix(star_color, space_color, smoothstep(0.0, star_size_threshold, distance));

    // var color = vec3<f32>(0.0, 0.0, 0.0); // base color of the sky
    // let intensity = star_intensity(ray_direction);
    // if (intensity > 0.0) {
    //     color = vec3<f32>(1.0, 1.0, 1.0) * intensity; // color of the stars
    // }

//    let scale = 20.0; // Adjust scale for finer granularity
//     let intensity = smooth_star_intensity(normalize(ray_direction), scale);
//     var color = vec3<f32>(0.0, 0.0, 0.0); // Base color of the sky
//     if (intensity > 0.0) {
//         color = vec3<f32>(1.0, 1.0, 1.0) * intensity; // Star color
//     }

    return vec4<f32>(color, 1.0);
}



fn voronoi(p: vec2<f32>) -> f32 {
    let n = floor(p);
    let f = fract(p);

    var min_dist = 8.0;

        {
        let b = vec2<f32>(-1.0, -1.0);
        let r = b + hash2(n + b) - f;
        let d = dot(r, r);
        min_dist = min(min_dist, d);
    }

        {
        let b = vec2<f32>(0.0, -1.0);
        let r = b + hash2(n + b) - f;
        let d = dot(r, r);
        min_dist = min(min_dist, d);
    }

        {
        let b = vec2<f32>(1.0, -1.0);
        let r = b + hash2(n + b) - f;
        let d = dot(r, r);
        min_dist = min(min_dist, d);
    }

        {
        let b = vec2<f32>(-1.0, 0.0);
        let r = b + hash2(n + b) - f;
        let d = dot(r, r);
        min_dist = min(min_dist, d);
    }

        {
        let b = vec2<f32>(0.0, 0.0);
        let r = b + hash2(n + b) - f;
        let d = dot(r, r);
        min_dist = min(min_dist, d);
    }

        {
        let b = vec2<f32>(1.0, 0.0);
        let r = b + hash2(n + b) - f;
        let d = dot(r, r);
        min_dist = min(min_dist, d);
    }

        {
        let b = vec2<f32>(-1.0, 1.0);
        let r = b + hash2(n + b) - f;
        let d = dot(r, r);
        min_dist = min(min_dist, d);
    }

        {
        let b = vec2<f32>(0.0, 1.0);
        let r = b + hash2(n + b) - f;
        let d = dot(r, r);
        min_dist = min(min_dist, d);
    }

        {
        let b = vec2<f32>(1.0, 1.0);
        let r = b + hash2(n + b) - f;
        let d = dot(r, r);
        min_dist = min(min_dist, d);
    }

    return sqrt(min_dist);
}