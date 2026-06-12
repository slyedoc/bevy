// Solari debug-view overlay. One specialized variant per view (selected by a
// `VIEW_*` shader_def), overwriting the lit color. Cluster views trace their own
// primary ray and read cluster data from the scene bind group (group 0); buffer
// views read the bound `selected` texture.

enable wgpu_ray_query;

#import bevy_render::view::View
#ifdef VIEW_CLUSTER_FAMILY
#import bevy_solari::scene_bindings::{trace_ray, set_view_cull_mask, clusters, instance_cluster_ranges, RAY_T_MAX}
#endif

@group(1) @binding(0) var selected: texture_2d<f32>;
@group(1) @binding(1) var view_output: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(2) var<uniform> view: View;
#ifdef VIEW_CLUSTER_FAMILY
// Per-view RT cull mask — only the cluster-family views trace primary rays.
struct SolariView { cull_mask: vec4<u32> }
@group(1) @binding(3) var<uniform> solari_view: SolariView;
#endif

@compute @workgroup_size(8, 8, 1)
fn debug_overlay(@builtin(global_invocation_id) global_id: vec3<u32>) {
#ifdef VIEW_CLUSTER_FAMILY
    set_view_cull_mask(solari_view.cull_mask.x);
#endif
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }
    let pixel = global_id.xy;
    // `selected` may be smaller than the output (the DLSS guide textures stay
    // at render resolution while debug views render full-res) — nearest-scale
    // the sample coordinate; identical dims degenerate to `pixel`.
    let src = vec2<i32>(
        pixel * textureDimensions(selected) / vec2u(view.main_pass_viewport.zw),
    );
    var color = vec3(0.0);

#ifdef VIEW_COLOR
    color = textureLoad(selected, src, 0).rgb;
#endif
#ifdef VIEW_GRAYSCALE
    color = vec3(textureLoad(selected, src, 0).r);
#endif
#ifdef VIEW_NORMAL
    color = textureLoad(selected, src, 0).xyz * 0.5 + 0.5;
#endif
#ifdef VIEW_MOTION
    color = vec3(abs(textureLoad(selected, src, 0).xy) * 20.0, 0.0);
#endif
#ifdef VIEW_MATERIAL_ID
    let id_texel = textureLoad(selected, src, 0);
    if id_texel.w >= 0.0 {
        color = hash_color(u32(id_texel.w));
    }
#endif
#ifdef VIEW_CLUSTER_FAMILY
    color = cluster_debug(pixel);
#endif

    textureStore(view_output, pixel, vec4(color, 1.0));
}

fn hash_color(seed: u32) -> vec3<f32> {
    var x = seed;
    x = (x ^ 61u) ^ (x >> 16u);
    x = x + (x << 3u);
    x = x ^ (x >> 4u);
    x = x * 0x27d4eb2du;
    x = x ^ (x >> 15u);
    let r = f32(x & 0xFFu) / 255.0;
    let g = f32((x >> 8u) & 0xFFu) / 255.0;
    let b = f32((x >> 16u) & 0xFFu) / 255.0;
    let m = max(r, max(g, b));
    return vec3<f32>(r, g, b) / max(m, 0.001) * 0.9 + vec3<f32>(0.05);
}

#ifdef VIEW_CLUSTER_FAMILY
fn cluster_debug(pixel: vec2<u32>) -> vec3<f32> {
    let pixel_uv = (vec2<f32>(pixel) + 0.5) / view.main_pass_viewport.zw;
    let pixel_ndc = pixel_uv * 2.0 - 1.0;
    let ray_target = view.world_from_clip * vec4(pixel_ndc.x, -pixel_ndc.y, 1.0, 1.0);
    let origin = view.world_position;
    let dir = normalize(ray_target.xyz / ray_target.w - origin);
    let ray = trace_ray(origin, dir, 0.0, RAY_T_MAX, RAY_FLAG_NONE);
    if ray.kind == RAY_QUERY_INTERSECTION_NONE {
        return vec3(0.0);
    }
#ifdef VIEW_LOD
    return lod_palette(clusters[ray.geometry_index].lod_level);
#endif
#ifdef VIEW_CLUSTER
    return hash_color(ray.geometry_index);
#endif
#ifdef VIEW_TRIANGLE
    return hash_color(ray.geometry_index * 31u + ray.primitive_index);
#endif
#ifdef VIEW_GEOMETRY_CHECK
    let range = instance_cluster_ranges[ray.instance_index];
    let lo = range.cluster_base;
    let hi = range.cluster_base + range.cluster_count;
    if ray.geometry_index >= lo && ray.geometry_index < hi {
        return vec3<f32>(0.05, 0.8, 0.05);
    }
    return vec3<f32>(1.0, 0.0, 0.0);
#endif
    return vec3(0.0);
}

fn lod_palette(lod: u32) -> vec3<f32> {
    switch lod {
        case 0u:  { return vec3<f32>(0.20, 0.60, 1.00); }
        case 1u:  { return vec3<f32>(0.20, 0.90, 0.90); }
        case 2u:  { return vec3<f32>(0.20, 0.90, 0.40); }
        case 3u:  { return vec3<f32>(0.70, 0.95, 0.20); }
        case 4u:  { return vec3<f32>(1.00, 0.90, 0.20); }
        case 5u:  { return vec3<f32>(1.00, 0.60, 0.10); }
        case 6u:  { return vec3<f32>(1.00, 0.30, 0.10); }
        case 7u:  { return vec3<f32>(0.90, 0.10, 0.40); }
        case 8u:  { return vec3<f32>(0.70, 0.10, 0.70); }
        case 9u:  { return vec3<f32>(0.40, 0.10, 0.80); }
        case 10u: { return vec3<f32>(0.20, 0.20, 0.80); }
        default:  { return vec3<f32>(1.00, 1.00, 1.00); }
    }
}
#endif
