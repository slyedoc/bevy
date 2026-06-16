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
// sampling over the EMISSIVE set (directional lights are shaded outside
// ReSTIR, deterministically), so a tile is an unbiased sample stream for any
// receiver; the per-pixel WRS over its candidates does the importance part.

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_solari::pbr::{rand_f, rand_range_u}
#import bevy_solari::restir_bindings::{view, light_tiles, pack_light_tile_sample, unpack_light_tile_sample, LightTileSample, regir_checksums, regir_life, regir_cell_data, regir_samples, REGIR_TABLE_SIZE, REGIR_ENTRIES_PER_CELL}
#import bevy_solari::sampling::{generate_random_emissive_light_sample, NULL_LIGHT_ID}
#import bevy_solari::scene_bindings::{light_sources, active_light_list, LIGHT_SOURCE_KIND_NONE}

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
    let sample = generate_random_emissive_light_sample(&rng);
    light_tiles[i] = pack_light_tile_sample(
        sample.resolved_light_sample,
        sample.light_sample.light_id,
        sample.light_sample.seed,
    );
}

// ── ReGIR grid maintenance ───────────────────────────────────────────────────

/// A cell's preference for a light sample: emitted luminance over (clamped)
/// squared distance to the cell, weighted by how much the emitting surface
/// FACES the cell. Light samples are points on emissive geometry (e.g. one
/// spot on a bulb sphere), and a point only lights receivers in front of it —
/// without the facing term, cells stock back-facing points half their
/// receivers can't use, and "which way the lamp shines" strobes with the
/// per-frame entry roll. The cosine is floored, not zeroed, so every sample
/// keeps non-zero probability (full support ⇒ the chained RIS stays unbiased
/// for receivers that CAN see a point the cell center can't).
fn regir_target_function(entry: LightTileSample, cell_center: vec3<f32>, min_distance_squared: f32) -> f32 {
    let resolved = unpack_light_tile_sample(entry);
    let to_cell = cell_center - resolved.world_position.xyz;
    let distance_squared = max(dot(to_cell, to_cell), min_distance_squared);
    let facing = clamp(
        dot(resolved.world_normal, to_cell / sqrt(distance_squared)),
        0.1,
        1.0,
    );
    return luminance(resolved.radiance) * facing / distance_squared;
}

/// Uniform tile-pool candidates RIS-reduced into each cell entry per frame.
const REGIR_FILL_CANDIDATES = 32u;
/// Temporal confidence cap for a cell entry (M, stored in the entry's `seed`
/// field). Bounds the smoothing lag: the entry's weight converges to an
/// exponential average with factor ~1/cap, and a lighting change fully
/// re-resolves within ~cap frames.
const REGIR_ENTRY_CONFIDENCE_CAP = 32u;

/// Age every cell; a cell unqueried for its whole lifetime frees its slot.
/// Runs before [`regir_fill`] and the queries, so a freed slot can be
/// re-claimed the same frame.
@compute @workgroup_size(256, 1, 1)
fn regir_decay(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if cell >= REGIR_TABLE_SIZE {
        return;
    }
    let life = atomicLoad(&regir_life[cell]);
    if life == 0u {
        return;
    }
    atomicStore(&regir_life[cell], life - 1u);
    if life == 1u {
        atomicStore(&regir_checksums[cell], 0u);
        // Wipe the entries so a same-frame re-claim of this slot can't serve
        // the previous occupant's lights.
        let base = cell * REGIR_ENTRIES_PER_CELL;
        for (var e = 0u; e < REGIR_ENTRIES_PER_CELL; e += 1u) {
            regir_samples[base + e] = LightTileSample(vec3(0.0), 0.0, 0u, 0u, NULL_LIGHT_ID, 0u);
        }
    }
}

/// Refill every live cell's light samples: each entry RIS-selects from
/// `REGIR_FILL_CANDIDATES` uniform tile-pool candidates with a
/// distance-to-cell target. The target is position-only (`luminance / d²`, no
/// emitter cosine) so every emissive keeps non-zero probability in every cell
/// — the downstream per-pixel RIS stays unbiased — and `d²` is clamped to
/// half a cell so in-cell lights don't explode the weights. The stored
/// `inverse_pdf` becomes the chained-RIS contribution weight
/// (`weight_sum / target`), exactly what the per-pixel WRS multiplies by.
@compute @workgroup_size(256, 1, 1)
fn regir_fill(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= REGIR_TABLE_SIZE * REGIR_ENTRIES_PER_CELL {
        return;
    }
    let cell = i / REGIR_ENTRIES_PER_CELL;
    if atomicLoad(&regir_life[cell]) == 0u {
        return;
    }
    let cell_data = regir_cell_data[cell];
    let cell_center = cell_data.xyz;
    let min_distance_squared = cell_data.w * cell_data.w * 0.25;

    var rng = i + view.frame_count * 0x9e3779b9u;

    // The cell RIS runs in LIGHT space: an entry selects a LIGHT (its stored
    // point is only the proxy the target evaluates at), and the per-pixel
    // candidates sample a FRESH point on that light each frame. In light
    // space a uniform tile-pool candidate's inverse pick pdf is just
    // `emissive_count`; the stored entry weight is the light's chained-RIS
    // contribution weight, which the pixel multiplies by its fresh point's
    // area inverse-pdf.
    let emissive_count = f32(active_light_list[0]);

    // Fresh light selection: RIS over the tile-pool candidates.
    var fresh = LightTileSample(vec3(0.0), 0.0, 0u, 0u, NULL_LIGHT_ID, 0u);
    var fresh_target_function = 0.0;
    var fresh_weight_sum = 0.0;
    let pool_size = arrayLength(&light_tiles);
    for (var k = 0u; k < REGIR_FILL_CANDIDATES; k += 1u) {
        let candidate = light_tiles[rand_range_u(pool_size, &rng)];
        if candidate.light_id == NULL_LIGHT_ID {
            continue;
        }
        let target_function = regir_target_function(candidate, cell_center, min_distance_squared);
        let resampling_weight = target_function * emissive_count / f32(REGIR_FILL_CANDIDATES);
        fresh_weight_sum += resampling_weight;
        if rand_f(&rng) < resampling_weight / fresh_weight_sum {
            fresh = candidate;
            fresh_target_function = target_function;
        }
    }
    var fresh_lambda = 0.0;
    if fresh_target_function > 0.0 {
        fresh_lambda = fresh_weight_sum / fresh_target_function;
    }

    // Temporal accumulation, confidence-weighted exactly like the per-pixel
    // reservoirs (the entry's M count lives in its `seed` field — the pixel
    // side samples fresh points and never reads it). Without this, the
    // entry's weight is a FRESH roll of `REGIR_FILL_CANDIDATES` random tile
    // draws every frame: every pixel in the cell multiplies by that same
    // re-rolled, heavy-tailed weight, so whole light pools visibly pump in
    // unison even in a static scene. Confidence-MIS smoothing converges the
    // weight to its mean while staying unbiased and responsive (cap below).
    var selected = fresh;
    var selected_target_function = fresh_target_function;
    var selected_lambda = fresh_lambda;
    var selected_m = 1u;

    let prior = regir_samples[i];
    let prior_m = min(prior.seed, REGIR_ENTRY_CONFIDENCE_CAP);
    if prior.light_id != NULL_LIGHT_ID
        && prior.inverse_pdf > 0.0
        && prior_m > 0u
        && light_sources[prior.light_id >> 16u].kind != LIGHT_SOURCE_KIND_NONE
    {
        let prior_target_function = regir_target_function(prior, cell_center, min_distance_squared);
        let m_total = f32(prior_m + 1u);
        let w_prior = (f32(prior_m) / m_total) * prior_target_function * prior.inverse_pdf;
        let w_fresh = (1.0 / m_total) * fresh_target_function * fresh_lambda;
        let weight_sum = w_prior + w_fresh;
        if weight_sum > 0.0 {
            if rand_f(&rng) >= w_fresh / weight_sum {
                selected = prior;
                selected_target_function = prior_target_function;
            }
            selected_lambda = weight_sum / selected_target_function;
            selected_m = prior_m + 1u;
        }
    }

    selected.inverse_pdf = selected_lambda;
    selected.seed = selected_m;
    regir_samples[i] = selected;
}
