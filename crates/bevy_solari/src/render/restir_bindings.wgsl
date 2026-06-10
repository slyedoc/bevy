#define_import_path bevy_solari::restir_bindings

#import bevy_render::view::View
#import bevy_render::utils::{octahedral_encode, octahedral_decode}
#import bevy_solari::pbr::{vec3_to_rgb9e5_, rgb9e5_to_vec3_}
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
}
@group(1) @binding(14) var<uniform> solari_view: SolariView;
// Environment map (sky), sampled in the ray direction on a miss. Bound to the
// view's baked-atmosphere / `Skybox.image` cube (or a fallback when absent).
@group(1) @binding(15) var environment_map: texture_cube<f32>;
@group(1) @binding(16) var environment_map_sampler: sampler;
// Atmosphere params + sun for primary-ray aerial perspective (distance haze).
// Disabled (`aerial_enabled == 0`) when the view has no `SolariAtmosphere`.
@group(1) @binding(17) var<uniform> atmosphere: Atmosphere;
// Per-frame scalars come from `view` — `view.frame_count` seeds the RNG, so no
// push-constant block is needed.
