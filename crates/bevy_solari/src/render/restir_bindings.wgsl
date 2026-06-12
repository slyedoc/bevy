#define_import_path bevy_solari::restir_bindings

#import bevy_render::view::View
#import bevy_render::utils::{octahedral_encode, octahedral_decode}
#import bevy_solari::pbr::{vec3_to_rgb9e5_, rgb9e5_to_vec3_, rand_f, rand_vec2f}
#import bevy_solari::sampling::ResolvedLightSample
#import bevy_solari::atmosphere::Atmosphere

// Shared group(1) bindings + data structures for the full-RT ReSTIR path
// tracer. The scene (TLAS, instances, materials, lights) lives in group(0),
// supplied by `bevy_solari::scene_bindings`.
//
// Structs are declared before the bindings that reference them so the
// composed module type-checks regardless of pass.
//
// Reservoirs are stored packed as `vec4<u32>` (the consuming pass defines the
// reservoir struct + pack/unpack). 3b is ReSTIR DI (a small light-sample
// reservoir); the unified ReSTIR PT `PathReservoir` returns in 3b-B.

// Light-tile pool shape: `LIGHT_TILE_BLOCKS` tiles of
// `LIGHT_TILE_SAMPLES_PER_BLOCK` presampled light samples each. An 8×8
// initial-candidates workgroup picks ONE random tile and draws all its
// candidates from it — coherent 32-byte reads instead of per-pixel light
// resolves (the emissive resolve walks the instance's cluster list, far too
// heavy per candidate per pixel). Keep in sync with `prepare.rs`.
const LIGHT_TILE_BLOCKS = 128u;
const LIGHT_TILE_SAMPLES_PER_BLOCK = 1024u;

/// One presampled light-tile entry: the resolved world-space payload (for the
/// initial candidates' target function) plus the originating `LightSample`
/// identity (`light_id` + `seed`) that a reservoir stores for exact
/// re-resolution at merge/shade time. 32-byte stride — keep in sync with
/// `LIGHT_TILE_SAMPLE_STRUCT_SIZE` in `prepare.rs`.
struct LightTileSample {
    world_position: vec3<f32>,
    // |inverse_pdf|; the sign bit carries the directional flag (a directional
    // sample's `world_position` is a direction, not a point).
    inverse_pdf: f32,
    // Octahedral-encoded world normal (2x16unorm).
    packed_world_normal: u32,
    // RGB9E5-packed `log2(radiance + 1)` — log-encoded so the shared exponent
    // spans sun-scale luminance; lossiness only perturbs the RIS target
    // function (shading re-resolves the light exactly from `light_id`/`seed`).
    packed_radiance: u32,
    light_id: u32,
    seed: u32,
}

fn pack_light_tile_sample(resolved: ResolvedLightSample, light_id: u32, seed: u32) -> LightTileSample {
    let directional = resolved.world_position.w == 0.0;
    return LightTileSample(
        resolved.world_position.xyz,
        select(resolved.inverse_pdf, -resolved.inverse_pdf, directional),
        pack2x16unorm(octahedral_encode(resolved.world_normal)),
        vec3_to_rgb9e5_(log2(resolved.radiance + 1.0)),
        light_id,
        seed,
    );
}

fn unpack_light_tile_sample(tile: LightTileSample) -> ResolvedLightSample {
    let directional = tile.inverse_pdf < 0.0;
    return ResolvedLightSample(
        vec4(tile.world_position, select(1.0, 0.0, directional)),
        octahedral_decode(unpack2x16unorm(tile.packed_world_normal)),
        exp2(rgb9e5_to_vec3_(tile.packed_radiance)) - 1.0,
        abs(tile.inverse_pdf),
    );
}

/// GI (one-bounce indirect) reservoir. 48-byte std430 stride — keep in sync
/// with `GI_RESERVOIR_STRUCT_SIZE` in `prepare.rs`. Mirrors realtime's GI
/// `Reservoir`: a reconnection vertex (sample point) + its incoming radiance.
struct GiReservoir {
    sample_point_world_position: vec3<f32>,
    weight_sum: f32,
    sample_point_world_normal: vec3<f32>,
    confidence_weight: f32,
    radiance: vec3<f32>,
    unbiased_contribution_weight: f32,
}

@group(1) @binding(0) var view_output: texture_storage_2d<rgba16float, read_write>;
// Binding names are prefixed `gbuffer_` so they don't collide with struct
// field names like `world_position` (naga_oil flattens both into one
// namespace and a global/field name clash breaks field accessors).
@group(1) @binding(1) var gbuffer_position: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(2) var previous_gbuffer_position: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(3) var gbuffer_normal: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(4) var previous_gbuffer_normal: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(5) var motion_vectors: texture_storage_2d<rgba16float, read_write>;
// Fixed-role reservoir buffers (not frame-parity ping-pong): `a` = temporal
// history (persists across frames), `b` = this-frame intermediate.
@group(1) @binding(6) var<storage, read_write> reservoir_a: array<vec4<u32>>;
@group(1) @binding(7) var<storage, read_write> reservoir_b: array<vec4<u32>>;
@group(1) @binding(8) var<storage, read_write> light_tiles: array<LightTileSample>;
@group(1) @binding(9) var<uniform> view: View;
@group(1) @binding(10) var gbuffer_uv: texture_storage_2d<rgba32float, read_write>;
// Self-owned previous-frame clip_from_world: slot [frame&1] = current (written
// by one thread in visibility), slot [1-(frame&1)] = previous (read for
// temporal reprojection). `view.frame_count` provides the parity.
@group(1) @binding(11) var<storage, read_write> view_clip_from_world: array<mat4x4<f32>, 2>;
// GI reservoir buffers (fixed roles: a = history, b = intermediate).
@group(1) @binding(12) var<storage, read_write> gi_reservoir_a: array<GiReservoir>;
@group(1) @binding(13) var<storage, read_write> gi_reservoir_b: array<GiReservoir>;
/// Per-view RT cull mask (camera `RenderLayers` → low 8 bits, `.x`), the
/// no-skybox primary-miss background (`clear_color`, linear RGB), and the sky
/// brightness in raw cd/m² (`0.0` ⇒ no skybox). Seeded into the scene-bindings
/// `view_cull_mask` at each tracing entry.
struct SolariView {
    cull_mask: vec4<u32>,
    clear_color: vec3<f32>,
    environment_brightness: f32,
    // `restir_debug` visualization mode (0 = none).
    debug_mode: u32,
    // Thin-lens DoF (`SolariLens`); aperture 0 = pinhole. Pathtracer-only.
    focus_distance: f32,
    aperture_radius: f32,
}
@group(1) @binding(14) var<uniform> solari_view: SolariView;
// Environment map (sky), sampled in the ray direction on a miss. Bound to the
// view's baked-atmosphere / `Skybox.image` cube (or a fallback when absent).
@group(1) @binding(15) var environment_map: texture_cube<f32>;
@group(1) @binding(16) var environment_map_sampler: sampler;
// Atmosphere params + sun for primary-ray aerial perspective (distance haze).
// Disabled (`aerial_enabled == 0`) when the view has no `SolariAtmosphere`.
@group(1) @binding(17) var<uniform> atmosphere: Atmosphere;
// First-hit distance of the specular reflection ray (specular-GI pass writes,
// the DLSS guide resolve reads; `RAY_T_MAX` = environment miss).
@group(1) @binding(18) var specular_hit_distance: texture_storage_2d<r32float, read_write>;
// ── ReGIR world-space light grid ─────────────────────────────────────────────
//
// A spatial hash of cells (same scheme as the GI world cache: quantized world
// position + distance LOD, PCG key + IQ checksum, linear probing). Each LIVE
// cell holds `REGIR_ENTRIES_PER_CELL` light samples, RIS-selected from the
// uniform tile pool with a distance-to-cell target — so a pixel's initial DI
// candidates draw from lights that matter NEAR its surface point. With many
// thousands of small local lights, uniform tiles alone starve every pixel of
// its dominant light (present in a given tile with probability lights/pool),
// and the reservoir's rare huge-weight catches render as morphing bright
// spots.
//
// Flow per frame: `regir_decay` (age cells, free the dead) → `regir_fill`
// (refill live cells) → `initial_and_temporal` queries (insert + mark; a cell
// inserted this frame serves candidates from NEXT frame — callers fall back
// to the uniform tiles while it's cold).
@group(1) @binding(19) var<storage, read_write> regir_checksums: array<atomic<u32>>;
@group(1) @binding(20) var<storage, read_write> regir_life: array<atomic<u32>>;
// Per-cell RIS target point: `xyz` = cell center, `w` = cell size.
@group(1) @binding(21) var<storage, read_write> regir_cell_data: array<vec4<f32>>;
@group(1) @binding(22) var<storage, read_write> regir_samples: array<LightTileSample>;

// ── Caustic photon grid ──────────────────────────────────────────────────────
//
// Photon-mapped light → specular chain → diffuse transport, deposited by the
// `caustic_emit` pass (`caustics.wgsl`) and gathered at diffuse shading —
// the transport the reservoirs structurally can't find (NEE can't thread a
// delta chain). World-space, so the caustic is stable under camera motion;
// a 16-frame exponential running average (the decay pass), so light/glass
// changes re-converge without explicit invalidation. Each cell is 4 words:
// `[identity checksum, r, g, b]` with flux in fixed point.
@group(1) @binding(23) var<storage, read_write> caustic_cells: array<atomic<u32>>;
// The photon-emission parameters, computed ENTIRELY on the GPU by the
// `caustic_prepare_*` passes (reset → reduce over instances → finalize) from
// the live transforms / material ids / instance AABBs / light columns — no
// CPU mirror of the scene is consulted. Word layout (f32s bitcast except
// where noted):
//   0-4   reduce scratch: min_u, max_u, min_v, max_v, min_depth
//         (orderable-encoded u32 atomics)
//   5     reduce scratch: transmissive instance count (u32)
//   8-10  rect center xyz        11 half extent along u
//   12-14 rect basis u           15 half extent along v
//   16-18 rect basis v           19 unused
//   20-22 photon travel dir      23 up-beam start distance
//   24-26 per-photon power rgb   27 enabled (1.0 / 0.0)
@group(1) @binding(24) var<storage, read_write> caustic_emitter: array<atomic<u32>, 32>;

fn caustic_emitter_load(i: u32) -> f32 {
    return bitcast<f32>(atomicLoad(&caustic_emitter[i]));
}

fn caustic_emitter_store(i: u32, value: f32) {
    atomicStore(&caustic_emitter[i], bitcast<u32>(value));
}

// Monotonic (orderable) encoding of an f32 into a u32, so atomicMin/Max
// order floats correctly across signs.
fn caustic_float_to_orderable(f: f32) -> u32 {
    let b = bitcast<u32>(f);
    return select(~b, b ^ 0x80000000u, (b >> 31u) == 0u);
}

fn caustic_orderable_to_float(o: u32) -> f32 {
    return bitcast<f32>(select(~o, o ^ 0x80000000u, (o >> 31u) == 1u));
}

/// Hash-table cell count (power of two). Keep in sync with `prepare.rs`.
const CAUSTIC_TABLE_SIZE = 65536u;
/// Cell edge length, world units. Fixed (no LOD): caustic sharpness is a
/// world-space property, and a 2 cm texel resolves a spectrum band.
const CAUSTIC_CELL_SIZE = 0.02;
/// Fixed-point scale for the flux atomics.
const CAUSTIC_FIXED_SCALE = 4096.0;
/// The decay pass's exponential window (×15/16 per frame ⇒ ~16 frames).
const CAUSTIC_EMA_FRAMES = 16.0;
/// Linear-probe attempts after a hash collision.
const CAUSTIC_PROBE_STEPS = 8u;

fn caustic_cell_key(quantized: vec3<u32>) -> u32 {
    var key = regir_pcg_hash(quantized.x);
    key = regir_pcg_hash(key + quantized.y);
    key = regir_pcg_hash(key + quantized.z);
    return key & (CAUSTIC_TABLE_SIZE - 1u);
}

// The cell checksum carries a KIND bit (bit 31): surface deposits store
// irradiance (flux/area), volume deposits store in-scattered radiance —
// different units, so the same world position hashes to DISTINCT cells per
// kind and each gather sees only its own.
fn caustic_cell_checksum(quantized: vec3<u32>, volume: bool) -> u32 {
    var checksum = regir_iqint_hash(quantized.x);
    checksum = regir_iqint_hash(checksum + quantized.y);
    checksum = regir_iqint_hash(checksum + quantized.z);
    checksum = max(checksum & 0x7FFFFFFFu, 1u); // 0 marks an empty slot
    return select(checksum, checksum | 0x80000000u, volume);
}

/// The table size behind a function — naga_oil resolves only functions and
/// structs across modules, so `caustics.wgsl` can't import the const.
fn caustic_table_size() -> u32 {
    return CAUSTIC_TABLE_SIZE;
}

/// One decay step for one cell (the `caustic_decay` pass body): age the flux
/// toward zero (×15/16, but at least −1 so residues die) and free the cell
/// when it empties, so the table never silts up with stale claims.
fn caustic_decay_cell(cell: u32) {
    let base = cell * 4u;
    if atomicLoad(&caustic_cells[base]) == 0u {
        return;
    }
    var live = false;
    for (var c = 1u; c <= 3u; c += 1u) {
        var e = atomicLoad(&caustic_cells[base + c]);
        if e > 0u {
            e -= max(e >> 4u, 1u);
        }
        atomicStore(&caustic_cells[base + c], e);
        live = live || e > 0u;
    }
    if !live {
        atomicStore(&caustic_cells[base], 0u);
    }
}

fn caustic_deposit_kind(p: vec3<f32>, flux: vec3<f32>, volume: bool) {
    let quantized = bitcast<vec3<u32>>(floor(p / CAUSTIC_CELL_SIZE + 0.0001));
    var key = caustic_cell_key(quantized);
    let checksum = caustic_cell_checksum(quantized, volume);
    for (var i = 0u; i < CAUSTIC_PROBE_STEPS; i += 1u) {
        let existing = atomicCompareExchangeWeak(&caustic_cells[key * 4u], 0u, checksum).old_value;
        if existing == 0u || existing == checksum {
            let e = vec3<u32>(flux * CAUSTIC_FIXED_SCALE + 0.5);
            atomicAdd(&caustic_cells[key * 4u + 1u], e.x);
            atomicAdd(&caustic_cells[key * 4u + 2u], e.y);
            atomicAdd(&caustic_cells[key * 4u + 3u], e.z);
            return;
        }
        key = (key + 1u) & (CAUSTIC_TABLE_SIZE - 1u);
    }
}

/// Claim-or-match the cell for `p` and accumulate fixed-point photon flux
/// (the emit pass's surface deposit).
fn caustic_deposit(p: vec3<f32>, flux: vec3<f32>) {
    caustic_deposit_kind(p, flux, false);
}

/// One volumetric photon-march step: `flux_step` = photon power × σ_s × ds.
/// The 1/(4π·cell) here folds the isotropic phase and the volume↔area
/// normalization difference, so the gather's surface-style estimate returns
/// in-scattered RADIANCE directly — the fog march multiplies by its own
/// transmittance × ds and nothing else.
fn caustic_deposit_volume(p: vec3<f32>, flux_step: vec3<f32>) {
    caustic_deposit_kind(p, flux_step / (12.566371 * CAUSTIC_CELL_SIZE), true);
}

/// Irradiance estimate (lm/m²) at a surface point from the photon grid:
/// flux ÷ (EMA window × cell area). The query position is jittered by half a
/// cell so the 2 cm texels dither instead of showing as blocks — the
/// temporal pass integrates the dither out. Photon density already encodes
/// the incidence cosine (grazing flux spreads over more cells), so the
/// caller multiplies by the diffuse albedo / π only.
fn caustic_gather_kind(world_position: vec3<f32>, rng: ptr<function, u32>, volume: bool) -> vec3<f32> {
    let jitter = vec3(rand_f(rng), rand_f(rng), rand_f(rng)) - 0.5;
    let p = world_position + jitter * CAUSTIC_CELL_SIZE;
    let quantized = bitcast<vec3<u32>>(floor(p / CAUSTIC_CELL_SIZE + 0.0001));
    var key = caustic_cell_key(quantized);
    let checksum = caustic_cell_checksum(quantized, volume);
    for (var i = 0u; i < CAUSTIC_PROBE_STEPS; i += 1u) {
        let existing = atomicLoad(&caustic_cells[key * 4u]);
        if existing == checksum {
            let flux = vec3(
                f32(atomicLoad(&caustic_cells[key * 4u + 1u])),
                f32(atomicLoad(&caustic_cells[key * 4u + 2u])),
                f32(atomicLoad(&caustic_cells[key * 4u + 3u])),
            ) / CAUSTIC_FIXED_SCALE;
            return flux / (CAUSTIC_EMA_FRAMES * CAUSTIC_CELL_SIZE * CAUSTIC_CELL_SIZE);
        }
        if existing == 0u {
            return vec3(0.0);
        }
        key = (key + 1u) & (CAUSTIC_TABLE_SIZE - 1u);
    }
    return vec3(0.0);
}

fn caustic_gather(world_position: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    return caustic_gather_kind(world_position, rng, false);
}

/// In-scattered radiance (cd/m²) at a fog point from volumetric photon
/// deposits — the dispersed beams glowing in the dust. See
/// [`caustic_deposit_volume`] for the folded normalization.
fn caustic_gather_volume(world_position: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    return caustic_gather_kind(world_position, rng, true);
}

/// Hash-table cell count (power of two). Keep in sync with `prepare.rs`.
const REGIR_TABLE_SIZE = 65536u;
/// Light samples per cell. Keep in sync with `prepare.rs`. Sized against
/// intra-cell correlation: pixels in a cell draw from the same entries, so
/// too few entries strobe the whole cell in unison.
const REGIR_ENTRIES_PER_CELL = 32u;
/// Cell edge length at the closest LOD, in world units.
const REGIR_BASE_CELL_SIZE = 0.25;
/// How fast cells grow with distance to the camera.
const REGIR_LOD_SCALE = 15.0;
/// Frames a cell lives without being queried.
const REGIR_CELL_LIFETIME = 10u;
/// Linear-probe attempts after a hash collision.
const REGIR_MAX_SEARCH_STEPS = 3u;
/// "No cell" sentinel returned by [`regir_query`].
const REGIR_CELL_NONE = 0xFFFFFFFFu;

fn regir_pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn regir_iqint_hash(input: u32) -> u32 {
    let n = (input << 13u) ^ input;
    return n * (n * n * 15731u + 789221u) + 1376312589u;
}

/// Stochastically-rounded distance LOD (cube-fract bias toward the finer
/// level, like the world cache) — returns the cell size for this lookup.
fn regir_cell_size(world_position: vec3<f32>, view_position: vec3<f32>, rng: ptr<function, u32>) -> f32 {
    let camera_distance = distance(view_position, world_position) / REGIR_LOD_SCALE;
    let lod_f = log2(1.0 + camera_distance);
    let lod_fract = fract(lod_f);
    let lod = floor(lod_f) + select(0.0, 1.0, rand_f(rng) < lod_fract * lod_fract * lod_fract);
    return REGIR_BASE_CELL_SIZE * exp2(lod);
}

/// The live cell index for this surface point, inserting + marking it on the
/// way. Returns `REGIR_CELL_NONE` when the cell is cold (inserted this frame,
/// or probing failed) — the caller falls back to the uniform light tiles. The
/// query point is jittered in the surface's tangent plane by half a cell so
/// the grid structure doesn't imprint on the lighting.
fn regir_query(world_position_in: vec3<f32>, world_normal: vec3<f32>, view_position: vec3<f32>, rng: ptr<function, u32>) -> u32 {
    var world_position = world_position_in;
    var cell_size = regir_cell_size(world_position, view_position, rng);

    // https://tomclabault.github.io/blog/2025/regir (tangent-plane jitter)
    let tangent = normalize(select(
        vec3(0.0, -world_normal.z, world_normal.y),
        vec3(-world_normal.y, world_normal.x, 0.0),
        abs(world_normal.x) > abs(world_normal.z),
    ));
    let bitangent = cross(world_normal, tangent);
    let offset = (rand_vec2f(rng) * 2.0 - 1.0) * cell_size * 0.5;
    world_position += offset.x * tangent + offset.y * bitangent;
    cell_size = regir_cell_size(world_position, view_position, rng);

    let quantized = vec3<u32>(bitcast<vec3<u32>>(floor(world_position / cell_size + 0.0001)));
    // The LOD is part of the cell identity: the same integer coords at two
    // cell sizes are different regions of space.
    let lod_bits = bitcast<u32>(cell_size);

    var key = regir_pcg_hash(quantized.x);
    key = regir_pcg_hash(key + quantized.y);
    key = regir_pcg_hash(key + quantized.z);
    key = regir_pcg_hash(key + lod_bits);
    key = key & (REGIR_TABLE_SIZE - 1u);

    var checksum = regir_iqint_hash(quantized.x);
    checksum = regir_iqint_hash(checksum + quantized.y);
    checksum = regir_iqint_hash(checksum + quantized.z);
    checksum = regir_iqint_hash(checksum + lod_bits);
    checksum = max(checksum, 1u); // 0 marks an empty slot

    for (var i = 0u; i < REGIR_MAX_SEARCH_STEPS; i += 1u) {
        let existing = atomicCompareExchangeWeak(&regir_checksums[key], 0u, checksum).old_value;
        if existing == checksum {
            atomicStore(&regir_life[key], REGIR_CELL_LIFETIME);
            return key;
        }
        if existing == 0u {
            // Claimed an empty slot: record the RIS target point. Racing
            // inserters of the same cell write identical values. Cold until
            // the fill pass runs next frame.
            atomicStore(&regir_life[key], REGIR_CELL_LIFETIME);
            let center = (floor(world_position / cell_size + 0.0001) + 0.5) * cell_size;
            regir_cell_data[key] = vec4(center, cell_size);
            return REGIR_CELL_NONE;
        }
        key = (key + 1u) & (REGIR_TABLE_SIZE - 1u);
    }
    return REGIR_CELL_NONE;
}
// Per-frame scalars come from `view` — `view.frame_count` seeds the RNG, so no
// push-constant block is needed.

/// Read-only probe: the live cell index for this surface point, WITHOUT
/// inserting or marking it (the `restir_debug` cell view must not perturb the
/// grid it's visualizing). Same jittered quantization as [`regir_query`].
fn regir_find(world_position_in: vec3<f32>, world_normal: vec3<f32>, view_position: vec3<f32>, rng: ptr<function, u32>) -> u32 {
    var world_position = world_position_in;
    var cell_size = regir_cell_size(world_position, view_position, rng);
    let tangent = normalize(select(
        vec3(0.0, -world_normal.z, world_normal.y),
        vec3(-world_normal.y, world_normal.x, 0.0),
        abs(world_normal.x) > abs(world_normal.z),
    ));
    let bitangent = cross(world_normal, tangent);
    let offset = (rand_vec2f(rng) * 2.0 - 1.0) * cell_size * 0.5;
    world_position += offset.x * tangent + offset.y * bitangent;
    cell_size = regir_cell_size(world_position, view_position, rng);

    let quantized = vec3<u32>(bitcast<vec3<u32>>(floor(world_position / cell_size + 0.0001)));
    let lod_bits = bitcast<u32>(cell_size);

    var key = regir_pcg_hash(quantized.x);
    key = regir_pcg_hash(key + quantized.y);
    key = regir_pcg_hash(key + quantized.z);
    key = regir_pcg_hash(key + lod_bits);
    key = key & (REGIR_TABLE_SIZE - 1u);

    var checksum = regir_iqint_hash(quantized.x);
    checksum = regir_iqint_hash(checksum + quantized.y);
    checksum = regir_iqint_hash(checksum + quantized.z);
    checksum = regir_iqint_hash(checksum + lod_bits);
    checksum = max(checksum, 1u);

    for (var i = 0u; i < REGIR_MAX_SEARCH_STEPS; i += 1u) {
        let existing = atomicLoad(&regir_checksums[key]);
        if existing == checksum {
            return key;
        }
        if existing == 0u {
            return REGIR_CELL_NONE;
        }
        key = (key + 1u) & (REGIR_TABLE_SIZE - 1u);
    }
    return REGIR_CELL_NONE;
}
