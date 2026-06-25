// Bridges the RT pipeline's primary-hit depth into the hardware depth buffer so
// rasterized overlays drawn in the Transparent3d phase (e.g. gizmos) depth-test
// against the ray-traced scene.
//
// Solari's trace is a raw compute pass writing a per-pixel storage buffer; a
// depth-format texture can only be written by the fixed-function depth unit in a
// raster pass. This fullscreen pass is that bridge: it reads the reverse-Z NDC
// depth the raygen shader packed into the output buffer's `.w` (always, no DLSS
// needed) and emits it via `@builtin(frag_depth)`.
//
// Runs after `main_opaque_pass_3d` (which clears depth to the far plane) and
// before `main_transparent_pass_3d`. The pipeline uses `depth_compare = Always`
// + depth write, so every hit overwrites the cleared value; misses (`w < 0`)
// `discard` to keep the far plane, letting overlays draw over the background.
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View

@group(0) @binding(0) var<storage, read> rt_output: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> view: View;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @builtin(frag_depth) f32 {
    // Framebuffer pixel → row-major output index (`y*width + x`), matching the
    // raygen launch grid. Non-DLSS: render res == viewport, so this is 1:1.
    let coord = vec2<u32>(in.position.xy);
    let width = u32(view.viewport.z);
    let depth = rt_output[coord.y * width + coord.x].w;

    // Miss (w < 0): discard so the cleared far-plane depth survives and overlays
    // render over the ray-traced background.
    if depth < 0.0 {
        discard;
    }
    return depth;
}
