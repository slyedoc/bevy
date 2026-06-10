// Pass 5 — compose.
//
// Reads the raw linear radiance written to `view_output` by the path-trace
// pass, applies camera exposure, and writes it back. Downstream bloom +
// tonemapping (bevy's post chain) take it from there.
//
// Debug visualization deliberately does NOT live here (no baked view-mode
// switch) — it belongs in a separate render-debug-style overlay that reads the
// G-buffer / reservoir buffers.

#import bevy_solari::restir_bindings::{view, view_output}

@compute @workgroup_size(8, 8, 1)
fn compose(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }

    let radiance = textureLoad(view_output, global_id.xy).rgb;
    textureStore(view_output, global_id.xy, vec4(radiance * view.exposure, 1.0));
}
