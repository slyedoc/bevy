// Pass 2 — light-tile presampling.
//
// Fills the `light_tiles` pool with uniformly random, fully resolved light
// samples (analytic + emissive triangles), once per frame. Per-pixel initial
// candidates then draw from one workgroup-shared tile — coherent 32-byte
// reads — instead of resolving lights themselves: the emissive resolve walks
// the owning instance's cluster list to find the picked triangle, which is far
// too heavy to run per candidate per pixel.
//
// One thread resolves and packs one pool entry. The pool is plain uniform
// sampling over the light set, so a tile is an unbiased sample stream for any
// receiver; the per-pixel WRS over its candidates does the importance part.

#import bevy_solari::restir_bindings::{view, light_tiles, pack_light_tile_sample, LightTileSample}
#import bevy_solari::sampling::{generate_random_light_sample, NULL_LIGHT_ID}
#import bevy_solari::scene_bindings::light_sources

@compute @workgroup_size(256, 1, 1)
fn presample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= arrayLength(&light_tiles) {
        return;
    }

    // No lights: a null entry has zero radiance, so consumers' target
    // functions reject it and their reservoirs stay invalid.
    if arrayLength(&light_sources) == 0u {
        light_tiles[i] = LightTileSample(vec3(0.0), 0.0, 0u, 0u, NULL_LIGHT_ID, 0u);
        return;
    }

    var rng = i + view.frame_count * 5782582u;
    let sample = generate_random_light_sample(&rng);
    light_tiles[i] = pack_light_tile_sample(
        sample.resolved_light_sample,
        sample.light_sample.light_id,
        sample.light_sample.seed,
    );
}
