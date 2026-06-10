// Pass 2 — light-tile presampling.
//
// Stratified RIS over analytic + emissive lights, resolved into the
// `light_tiles` pool once per frame so the per-pixel ReSTIR passes draw cheap
// pre-resolved samples instead of re-sampling the whole light set.
//
// Dispatched as LIGHT_TILE_BLOCKS workgroups. Workgroup size is kept at 256
// (within the default `max_compute_invocations_per_workgroup`); retune when
// porting aurora's presample (which strided 1024 samples per block).
//
// TODO(restir): port aurora's presample.wgsl (analytic light RIS + emissive
// triangle sampling, RGB9E5 + octahedral packing).

#import bevy_solari::restir_bindings::{light_tiles, ResolvedLightSample}

@compute @workgroup_size(256, 1, 1)
fn presample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&light_tiles) {
        return;
    }

    // Stub: write an empty resolved sample.
    light_tiles[i] = ResolvedLightSample(vec3(0.0), 0.0, vec3(0.0), 0u);
}
