#define_import_path bevy_solari::restir_bindings

#import bevy_render::view::View

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

/// A resolved (world-space) light-tile sample. 32-byte stride — keep in sync
/// with `RESOLVED_LIGHT_SAMPLE_STRUCT_SIZE` in `prepare.rs`. Placeholder until
/// `presample.wgsl` is ported.
struct ResolvedLightSample {
    world_position: vec3<f32>,
    inverse_pdf: f32,
    world_normal: vec3<f32>,
    radiance_packed: u32,
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
@group(1) @binding(8) var<storage, read_write> light_tiles: array<ResolvedLightSample>;
@group(1) @binding(9) var<uniform> view: View;
@group(1) @binding(10) var gbuffer_uv: texture_storage_2d<rgba32float, read_write>;
// Self-owned previous-frame clip_from_world: slot [frame&1] = current (written
// by one thread in visibility), slot [1-(frame&1)] = previous (read for
// temporal reprojection). `view.frame_count` provides the parity.
@group(1) @binding(11) var<storage, read_write> view_clip_from_world: array<mat4x4<f32>, 2>;
// GI reservoir buffers (fixed roles: a = history, b = intermediate).
@group(1) @binding(12) var<storage, read_write> gi_reservoir_a: array<GiReservoir>;
@group(1) @binding(13) var<storage, read_write> gi_reservoir_b: array<GiReservoir>;
/// Per-view RT cull mask (camera `RenderLayers` → low 8 bits, `.x`). Seeded
/// into the scene-bindings `view_cull_mask` at each tracing entry.
struct SolariView { cull_mask: vec4<u32> }
@group(1) @binding(14) var<uniform> solari_view: SolariView;
// Per-frame scalars come from `view` — `view.frame_count` seeds the RNG, so no
// push-constant block is needed.
