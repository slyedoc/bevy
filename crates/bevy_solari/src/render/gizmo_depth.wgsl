// Bridges the ReSTIR primary-hit G-buffer into the hardware depth buffer.
//
// Solari's passes are compute and write storage textures; a depth-format
// texture can only be written by the fixed-function depth unit inside a raster
// pass. This fullscreen pass is that bridge: it reads the primary-hit world
// position (a storage texture the visibility pass wrote) and emits NDC depth
// via `@builtin(frag_depth)`, so rasterized overlays drawn in the Transparent3d
// phase (e.g. gizmos) depth-test against the ray-traced scene.
//
// Runs after `main_opaque_pass_3d` (which clears depth to the far plane) and
// before `main_transparent_pass_3d`. The pipeline uses `depth_compare = Always`
// + depth write, so every hit overwrites the cleared value; misses `discard` to
// keep the far plane, letting overlays draw over the background.
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View

@group(0) @binding(0) var gbuffer_position: texture_2d<f32>;
@group(0) @binding(1) var<uniform> view: View;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @builtin(frag_depth) f32 {
    // Framebuffer pixel → G-buffer texel. The Solari view renders to the
    // top-left of its target (origin 0), but under DLSS upscaling the
    // G-buffer is RENDER resolution while this pass covers the full-res
    // depth texture — nearest-scale the coordinate (1:1 when dims match).
    let gbuffer_dims = textureDimensions(gbuffer_position);
    let coords = vec2<i32>(vec2<u32>(in.position.xy) * gbuffer_dims / vec2<u32>(view.viewport.zw));
    let hit = textureLoad(gbuffer_position, coords, 0);

    // Miss: w < 0 marks "no geometry". Discard so the cleared far-plane depth
    // survives and overlays render over the background.
    if hit.w < 0.0 {
        discard;
    }

    // Project with the same (jittered) clip matrix the gizmo vertex shader uses,
    // so the reconstructed depth lines up with the rasterized lines. Reverse-Z.
    let clip = view.clip_from_world * vec4<f32>(hit.xyz, 1.0);
    return clip.z / clip.w;
}
