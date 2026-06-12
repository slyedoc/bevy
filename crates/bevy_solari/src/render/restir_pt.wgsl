enable wgpu_ray_query;

// Passes 3 & 4 — ReSTIR DI (direct) + ReSTIR GI (one-bounce indirect).
//
// `initial_and_temporal`: per primary-hit pixel, build + temporally reuse both
//   a DI reservoir (a light sample) and a GI reservoir (a hemisphere
//   reconnection vertex with its incoming radiance). Reprojection is shared:
//   the current world position is projected with the previous-frame clip matrix
//   and validated against the previous-frame G-buffer. Results → the `_b`
//   (intermediate) reservoir buffers.
// `spatial_and_shade`: merge one validated spatial neighbour into both
//   reservoirs, shade direct (DI reservoir, one final visibility ray) +
//   indirect (GI reservoir, one visibility ray to the reconnection vertex),
//   sum → view_output, and persist the result as next frame's history (`_a`).
//
// DI: light samples are world-space → no shift/Jacobian. GI: the reconnection
// vertex needs a geometry Jacobian on reuse. Both use confidence-weighted
// (M-based) MIS. No world cache → GI is a single indirect bounce (direct
// lighting at the reconnection vertex); multi-bounce returns with the unified
// PT reservoir later.

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_solari::pbr::{rand_f, rand_u, rand_range_u, sample_uniform_hemisphere, uniform_hemisphere_inverse_pdf, sample_disk}
#import bevy_render::maths::{PI, orthonormalize}
#import bevy_solari::brdf::{evaluate_brdf, evaluate_diffuse_brdf, evaluate_specular_brdf, F_AB, bend_shading_normal, fresnel_dielectric, dispersive_ior, spectral_lambda_rgb}
#import bevy_solari::sampling::{LightSample, ResolvedLightSample, generate_random_light_sample, resolve_light_sample, calculate_resolved_light_contribution, trace_light_visibility, trace_point_visibility, sample_random_light, sample_ggx_vndf, ggx_vndf_pdf, ggx_vndf_sample_invalid, random_emissive_light_pdf, power_heuristic, isnan, NULL_LIGHT_ID}
#import bevy_solari::scene_bindings::{trace_ray, trace_ray_traversal, set_view_cull_mask, resolve_ray_hit_full, resolve_material, materials, light_sources, active_light_list, directional_lights, ResolvedMaterial, LIGHT_SOURCE_KIND_NONE, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD, offset_ray_origin}
#import bevy_solari::restir_bindings::{view, view_output, gbuffer_position, gbuffer_normal, previous_gbuffer_position, previous_gbuffer_normal, gbuffer_uv, motion_vectors, reservoir_a, reservoir_b, gi_reservoir_a, gi_reservoir_b, GiReservoir, solari_view, light_tiles, unpack_light_tile_sample, LightTileSample, LIGHT_TILE_BLOCKS, LIGHT_TILE_SAMPLES_PER_BLOCK, environment_map, environment_map_sampler, specular_hit_distance, regir_query, regir_find, regir_samples, REGIR_CELL_NONE, REGIR_ENTRIES_PER_CELL}

const INITIAL_SAMPLES = 8u;
const DI_CONFIDENCE_WEIGHT_CAP = 20.0;
const GI_CONFIDENCE_WEIGHT_CAP = 8.0;
const MAX_JACOBIAN = 1.2;
const SPATIAL_ATTEMPTS = 5u;
const SPATIAL_RADIUS_PIXELS = 30.0;
// Below this roughness the DI reservoir stops shading the specular lobe and
// hands it to the BRDF-sampled specular GI pass (which then owns first-bounce
// emissive). Above it, DI does specular NEE (needed for directional/analytic
// lights a reflection ray can't hit) and specular GI zeroes first-bounce
// emissive to avoid double-counting reflected emissive geometry.
const SPECULAR_GI_FOR_DI_ROUGHNESS_THRESHOLD = 0.0225;
// Within this range of a candidate light, the initial-candidate target
// includes a real visibility trace. At point-blank range an area light
// self-occludes most of its own surface (a receiver 2 radii from a bulb sees
// ~25% of it), so an unshadowed target keeps electing occluded points and the
// light's contribution gates on/off with each frame's pick. The rays are
// short and only contact-range pixels pay for them.
const SHADOWED_TARGET_DISTANCE_SQUARED = 1.0;

struct Reservoir {
    light_id: u32,
    seed: u32,
    confidence_weight: f32,            // M
    unbiased_contribution_weight: f32, // W (unshadowed)
}

struct Surface {
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    material: ResolvedMaterial,
    material_id: u32,
    valid: bool,
}

struct Reprojection {
    valid: bool,
    pixel: vec2<u32>,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
}

struct ReservoirContribution {
    radiance: vec3<f32>,
    target_function: f32,
    wi: vec3<f32>,
    world_position: vec4<f32>,
}

struct MergeResult {
    reservoir: Reservoir,
    radiance: vec3<f32>,
    wi: vec3<f32>,
    world_position: vec4<f32>,
}

struct GiNeighbor {
    reservoir: GiReservoir,
    world_position: vec3<f32>,
}

struct GiMergeResult {
    reservoir: GiReservoir,
    radiance: vec3<f32>,
    wi: vec3<f32>,
}

// ---------------------------------------------------------------- entry points

// Boiling filter (cf. RTXDI): a rare candidate caught with a huge
// compensation weight survives merges on raw intensity and glows for its
// whole confidence lifetime — the "spot that appears then slowly fades".
// Kill any reservoir whose weight exceeds this multiple of its 8×8
// workgroup's average. Slightly biased (drops legitimate extreme energy),
// hugely stabilizing.
const BOILING_FILTER_STRENGTH = 20.0;

var<workgroup> boiling_energy: array<f32, 64>;

@compute @workgroup_size(8, 8, 1)
fn initial_and_temporal(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    set_view_cull_mask(solari_view.cull_mask.x);
    let pixel = global_id.xy;
    let index = reservoir_index(pixel);
    // No early returns: the boiling filter's workgroup barriers below need
    // uniform control flow, so inactive threads carry empty reservoirs
    // through instead of bailing.
    let in_viewport = all(pixel < vec2u(view.main_pass_viewport.zw));

    // Workgroup-uniform tile pick: all 64 threads draw their initial
    // candidates from the same 1024-entry tile, so the pool reads stay
    // coherent across the workgroup.
    var workgroup_rng = (workgroup_id.x * 7919u + workgroup_id.y) + view.frame_count * 5782582u;
    let tile_start = rand_range_u(LIGHT_TILE_BLOCKS, &workgroup_rng) * LIGHT_TILE_SAMPLES_PER_BLOCK;

    var surface: Surface;
    surface.valid = false;
    if in_viewport {
        surface = load_surface(pixel);
    }

    var di_reservoir = empty_reservoir();
    var gi_reservoir = gi_empty();
    if surface.valid {
        var rng = index + view.frame_count * 5782582u;
        let diffuse_brdf = surface.material.base_color / PI;

        // ReGIR cell for this surface point (inserting + marking it). Cold cell
        // (just inserted / probe failed) → the initial candidates fall back to
        // the workgroup's uniform tile.
        let regir_cell = regir_query(surface.world_position, surface.world_normal, view.world_position, &rng);

        // Direct (DI).
        let di_initial = generate_initial_reservoir(tile_start, regir_cell, surface.world_position, surface.world_normal, diffuse_brdf, &rng);
        let di_temporal = load_temporal_reservoir(pixel, surface.world_position, surface.world_normal);
        di_reservoir = merge_reservoirs(di_initial, di_temporal, surface.world_position, surface.world_normal, diffuse_brdf, &rng).reservoir;

        // Indirect (GI).
        let gi_initial = gi_generate_initial(surface.world_position, surface.world_normal, &rng);
        let gi_temporal = gi_load_temporal(pixel, surface.world_position, surface.world_normal);
        gi_reservoir = gi_merge(gi_initial, gi_temporal.reservoir, gi_temporal.world_position, surface.world_position, surface.world_normal, diffuse_brdf, &rng).reservoir;
    }

    // DI boiling filter.
    var energy = 0.0;
    if reservoir_valid(di_reservoir) {
        energy = di_reservoir.unbiased_contribution_weight;
    }
    boiling_energy[local_index] = energy;
    workgroupBarrier();
    var sum = 0.0;
    var live = 0u;
    for (var k = 0u; k < 64u; k += 1u) {
        let v = boiling_energy[k];
        sum += v;
        live += u32(v > 0.0);
    }
    if live > 0u && energy > BOILING_FILTER_STRENGTH * (sum / f32(live)) {
        di_reservoir = empty_reservoir();
    }

    // GI boiling filter (energy = expected contribution scale).
    workgroupBarrier();
    energy = 0.0;
    if gi_reservoir.confidence_weight > 0.0 {
        energy = luminance(gi_reservoir.radiance) * gi_reservoir.unbiased_contribution_weight;
    }
    boiling_energy[local_index] = energy;
    workgroupBarrier();
    sum = 0.0;
    live = 0u;
    for (var k = 0u; k < 64u; k += 1u) {
        let v = boiling_energy[k];
        sum += v;
        live += u32(v > 0.0);
    }
    if live > 0u && energy > BOILING_FILTER_STRENGTH * (sum / f32(live)) {
        gi_reservoir = gi_empty();
    }

    if in_viewport {
        store_reservoir_b(pixel, di_reservoir);
        gi_reservoir_b[index] = gi_reservoir;
    }
}

@compute @workgroup_size(8, 8, 1)
fn spatial_and_shade(@builtin(global_invocation_id) global_id: vec3<u32>) {
    set_view_cull_mask(solari_view.cull_mask.x);
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }
    let pixel = global_id.xy;
    let index = reservoir_index(pixel);

    let surface = load_surface(pixel);
    if !surface.valid {
        store_reservoir_a(pixel, empty_reservoir());
        gi_reservoir_a[index] = gi_empty();
        // Captured by a black-hole horizon (`w = -2` sentinel from the
        // visibility pass): black, NOT the sky.
        if textureLoad(gbuffer_position, pixel).w <= -1.5 {
            textureStore(view_output, pixel, vec4(0.0, 0.0, 0.0, 1.0));
            return;
        }
        // Primary miss: the sky (or the camera clear color) is the background.
        // Raw radiance — compose applies aerial perspective + exposure; the
        // clear color is a fixed framebuffer value, so ÷ exposure pre-cancels.
        let ray_direction = primary_ray_direction(pixel);
        var background = vec3(0.0);
        if solari_view.environment_brightness > 0.0 {
            background = sky_radiance(ray_direction);
        } else {
            background = solari_view.clear_color / max(view.exposure, 1e-6);
        }
        // Each ACTIVE directional light as a disk of its angular radius — the
        // sky bake's Mie halo doesn't draw the disk itself. Walked via the
        // active list: the slot-indexed column keeps stale luminance in freed
        // slots, which would draw a ghost sun.
        let emissive_count = active_light_list[0];
        let num_directional = active_light_list[1];
        for (var i = 0u; i < num_directional; i = i + 1u) {
            let source = light_sources[active_light_list[2u + emissive_count + i]];
            let sun = directional_lights[source.id];
            if dot(ray_direction, sun.direction_to_light) >= sun.cos_theta_max {
                background += sun.luminance;
            }
        }
        textureStore(view_output, pixel, vec4(background, 1.0));
        return;
    }

    var rng = index + view.frame_count * 5782582u + 0x9e3779b9u;
    let diffuse_brdf = surface.material.base_color / PI;

    // Spatial reuse: merge one validated neighbour's reservoir (this frame's
    // intermediate `_b` buffers), then persist the combined result as next
    // frame's temporal history (`_a`).
    let neighbor = find_spatial_neighbor(pixel, surface, &rng);

    var di_reservoir = load_reservoir_b(pixel);
    if neighbor.valid {
        let neighbor_reservoir = load_reservoir_b(neighbor.pixel);
        di_reservoir = merge_reservoirs(di_reservoir, neighbor_reservoir, surface.world_position, surface.world_normal, diffuse_brdf, &rng).reservoir;
    }
    store_reservoir_a(pixel, di_reservoir);

    var gi_reservoir = gi_reservoir_b[index];
    if neighbor.valid {
        let neighbor_gi_reservoir = gi_reservoir_b[reservoir_index(neighbor.pixel)];
        gi_reservoir = gi_merge(gi_reservoir, neighbor_gi_reservoir, neighbor.world_position, surface.world_position, surface.world_normal, diffuse_brdf, &rng).reservoir;
    }
    gi_reservoir_a[index] = gi_reservoir;

    let wo = normalize(view.world_position - surface.world_position);
    // Bend the smooth shading normal into the view hemisphere so silhouette
    // edges (where the interpolated normal dips past 90°) don't black out.
    // The raw G-buffer normal stays in `surface` — reuse validation compares
    // stored normals across pixels/frames, which must stay un-bent.
    let shading_normal = bend_shading_normal(surface.world_normal, wo);
    let NdotV = max(dot(shading_normal, wo), 0.0001);
    let F_ab = F_AB(surface.material.perceptual_roughness, NdotV);

    var radiance = surface.material.emissive;

    // Transmissive surfaces have no diffuse lobe to that extent — light passes
    // through instead of scattering. The specular GI pass owns everything that
    // remains (reflection + refraction), so scale the diffuse passes out here.
    let diffuse_fraction = 1.0 - surface.material.specular_transmission;

    // Directional lights (the sun) are shaded deterministically every frame —
    // one cone sample + shadow ray each — OUTSIDE the reservoir. A reservoir
    // holds one light sample per pixel; making the sun compete with a nearby
    // emissive for that slot patchworks the screen into per-light winners,
    // which reuse correlates into morphing light blobs. The light tiles feed
    // the reservoir emissive samples only.
    let emissive_count = active_light_list[0];
    let directional_count = active_light_list[1];
    for (var i = 0u; i < directional_count; i = i + 1u) {
        let slot = active_light_list[2u + emissive_count + i];
        let light_sample = LightSample(slot << 16u, rand_u(&rng));
        let resolved = resolve_light_sample(light_sample, light_sources[slot]);
        let contribution = calculate_resolved_light_contribution(resolved, surface.world_position, surface.world_normal);
        let visibility = trace_light_visibility(surface.world_position + surface.world_normal * RAY_T_MIN, resolved.world_position);
        var brdf = evaluate_diffuse_brdf(wo, contribution.wi, shading_normal, surface.material, F_ab);
        if surface.material.roughness > SPECULAR_GI_FOR_DI_ROUGHNESS_THRESHOLD {
            brdf += evaluate_specular_brdf(wo, contribution.wi, shading_normal, surface.material, F_ab);
        }
        radiance += diffuse_fraction * contribution.radiance * resolved.inverse_pdf * visibility * brdf;
    }

    // Direct lighting from the DI reservoir (final visibility traced once).
    // Diffuse always; specular NEE too unless the surface is near-mirror, where
    // the specular GI pass takes over via BRDF sampling.
    if reservoir_valid(di_reservoir) {
        let contribution = reservoir_contribution(di_reservoir, surface.world_position, surface.world_normal, diffuse_brdf);
        let visibility = trace_light_visibility(surface.world_position + surface.world_normal * RAY_T_MIN, contribution.world_position);
        var brdf = evaluate_diffuse_brdf(wo, contribution.wi, shading_normal, surface.material, F_ab);
        if surface.material.roughness > SPECULAR_GI_FOR_DI_ROUGHNESS_THRESHOLD {
            brdf += evaluate_specular_brdf(wo, contribution.wi, shading_normal, surface.material, F_ab);
        }
        radiance += diffuse_fraction * contribution.radiance * di_reservoir.unbiased_contribution_weight * visibility * brdf;
    }

    // Indirect lighting from the GI reservoir (visibility to the reconnection
    // vertex traced once).
    if gi_reservoir.confidence_weight > 0.0 {
        let wi = normalize(gi_reservoir.sample_point_world_position - surface.world_position);
        let visibility = trace_point_visibility(surface.world_position + surface.world_normal * RAY_T_MIN, gi_reservoir.sample_point_world_position);
        let brdf = evaluate_diffuse_brdf(wo, wi, shading_normal, surface.material, F_ab);
        radiance += diffuse_fraction * gi_reservoir.radiance * gi_reservoir.unbiased_contribution_weight * visibility * brdf;
    }

    // Raw linear radiance — `compose` applies exposure.
    textureStore(view_output, pixel, vec4(radiance, 1.0));
}

// Pass 4b — specular GI. Trace a fresh BSDF-sampled specular path each frame
// and add it on top of the diffuse passes' output. Not ReSTIR'd (single noisy
// sample, meant to be denoised downstream). Owns ALL specular, including
// direct-light highlights (reflected emissives), since DI shades diffuse only —
// and owns transmission: glass pixels are shaded almost entirely by this pass.
@compute @workgroup_size(8, 8, 1)
fn specular_gi(@builtin(global_invocation_id) global_id: vec3<u32>) {
    set_view_cull_mask(solari_view.cull_mask.x);
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }
    let pixel = global_id.xy;

    let surface = load_surface(pixel);
    if !surface.valid {
        // Sky: the reflection IS the environment — virtual point at infinity
        // for the DLSS specular-motion guide.
        textureStore(specular_hit_distance, pixel, vec4(RAY_T_MAX, 0.0, 0.0, 0.0));
        return; // diffuse passes already wrote view_output (black on miss).
    }
    // Zero distance = "no reflection data" — the guide resolve falls back to
    // the surface motion. Overwritten below once the reflection ray reports.
    textureStore(specular_hit_distance, pixel, vec4(0.0));

    var rng = reservoir_index(pixel) + view.frame_count * 5782582u + 0x68bc21ebu;

    // Transmissive primaries SPLIT deterministically: both the reflection and
    // the refraction path are traced and blended by exact Fresnel. A
    // stochastic lobe pick here flickers the pixel between "mirror" and
    // "through the glass" every frame — temporally stable when still, but the
    // moment the camera moves the denoiser loses its history and smears all
    // glass into a blur.
    var radiance = vec3(0.0);
    var first_hit_t = RAY_T_MAX; // environment miss = reflection at infinity
    let transmission = surface.material.specular_transmission;
    if transmission > 0.0 {
        let wo = normalize(view.world_position - surface.world_position);
        let primary_normal = bend_shading_normal(surface.world_normal, wo);
        let reflectance = fresnel_dielectric(
            min(dot(wo, primary_normal), 1.0), 1.0 / surface.material.ior);
        var reflect_hit_t = RAY_T_MAX;
        if reflectance < 1.0 {
            radiance += transmission * (1.0 - reflectance)
                * trace_specular_path_spectral(surface, PRIMARY_LOBE_REFRACT, &first_hit_t, &rng);
        }
        radiance += transmission * reflectance
            * trace_specular_path_spectral(surface, PRIMARY_LOBE_REFLECT, &reflect_hit_t, &rng);
        // The DLSS guide gets the dominant branch's content distance (for a
        // refracted branch that's the accumulated path length to the first
        // opaque hit — RTXPT's virtual-point scheme).
        if reflectance >= 0.5 {
            first_hit_t = reflect_hit_t;
        }
    }
    if transmission < 1.0 {
        var ggx_hit_t = RAY_T_MAX;
        radiance += (1.0 - transmission)
            * trace_specular_path_spectral(surface, PRIMARY_LOBE_GGX, &ggx_hit_t, &rng);
        if transmission == 0.0 {
            first_hit_t = ggx_hit_t;
        }
    }
    textureStore(specular_hit_distance, pixel, vec4(first_hit_t, 0.0, 0.0, 0.0));

    let existing = textureLoad(view_output, pixel).rgb;
    textureStore(view_output, pixel, vec4(existing + radiance, 1.0));
}

const PRIMARY_LOBE_GGX = 0u;
const PRIMARY_LOBE_REFLECT = 1u;
const PRIMARY_LOBE_REFRACT = 2u;

// Glossy-bounce budget, transmissive-crossing budget (a window is already 2
// interfaces, a wine glass with liquid 4+), and the nested-medium stack depth
// (see the pathtracer's twin).
const SPECULAR_GLOSSY_BOUNCES = 3u;
// Concave glass (the attenuation dragon) drives long total-internal-reflection
// cascades; paths that exhaust this budget return no radiance, so a tight cap
// shows as view-swimming dark patches.
const SPECULAR_GLASS_CROSSINGS = 16u;
// Above this roughness a specular-path hit takes its NEE light and ends the
// chain instead of continuing (see the loop).
const SPECULAR_PATH_CONTINUE_ROUGHNESS = 0.25;
const MEDIUM_STACK_SIZE = 4u;
const MEDIUM_NOT_FOUND = 0xFFFFFFFFu;

// Spectral wrapper around [`trace_specular_path`]: green traces the chain
// first carrying full RGB; only if it crossed a dispersive interface do red
// and blue re-trace from the SAME rng state — identical lobe decisions and
// light picks, different n(λ) — so the split is deterministic (the denoiser
// never sees per-frame color flicker) and the 3× cost lands only on pixels
// whose chain actually touches dispersive glass. Pre-dispersion contributions
// are gated to the green call (full RGB, counted once); post-collapse each
// chain carries exactly its own channel.
fn trace_specular_path_spectral(primary_surface: Surface, primary_lobe: u32, first_hit_t: ptr<function, f32>, rng: ptr<function, u32>) -> vec3<f32> {
    // ONE call site in a channel loop, not three inlined calls: WGSL inlines
    // every call and the chain is the biggest function in this shader —
    // three static calls per lobe site would triple the inlined shader body
    // (register pressure, occupancy) and tax every pixel, dispersive or not.
    let rng_start = *rng;
    let lambda_rgb = spectral_lambda_rgb();
    // Green first: it carries full RGB until a dispersive interface, and if
    // it never crosses one the loop ends after one pass with the chain
    // identical to the pre-spectral path.
    let lambdas = vec3(lambda_rgb.y, lambda_rgb.x, lambda_rgb.z);
    let masks = mat3x3<f32>(
        vec3(0.0, 1.0, 0.0),
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 0.0, 1.0),
    );
    var radiance = vec3(0.0);
    var rng_out = rng_start;
    for (var c = 0u; c < 3u; c += 1u) {
        // Every channel restarts from the same rng state — identical lobe
        // decisions and light picks, different n(λ).
        var rng_c = rng_start;
        var dispersed = false;
        var t_c = *first_hit_t;
        radiance += trace_specular_path(
            primary_surface, primary_lobe, &t_c, &rng_c,
            lambdas[c], masks[c], select(0.0, 1.0, c == 0u), &dispersed);
        if c == 0u {
            rng_out = rng_c;
            *first_hit_t = t_c;
            if !dispersed {
                break;
            }
        }
    }
    *rng = rng_out;
    return radiance;
}

// BSDF-sampled specular path from the primary surface: GGX reflection lobes
// with NEE at glossy hits, and exact-Fresnel reflect/refract delta lobes with
// Beer-Lambert absorption + a nested-dielectric medium stack at transmissive
// hits (the pathtracer's scheme, bounded for realtime). No world cache, so a
// glossy chain just terminates (no diffuse-cache fallback at the end).
//
// Spectral arguments (see [`trace_specular_path_spectral`]): `lambda_nm` is
// this chain's wavelength for dispersive etas; on the first dispersive
// interface the throughput collapses to `channel_mask` and `prefix_gate`
// (0 on the red/blue re-traces) stops gating the radiance sums — pre-collapse
// light belongs to the green call alone.
fn trace_specular_path(primary_surface: Surface, primary_lobe: u32, first_hit_t: ptr<function, f32>, rng: ptr<function, u32>, lambda_nm: f32, channel_mask: vec3<f32>, prefix_gate: f32, dispersed: ptr<function, bool>) -> vec3<f32> {
    var radiance = vec3(0.0);
    var throughput = vec3(1.0);
    var gate = prefix_gate;
    var spectral_masked = false;

    let wo_primary = normalize(view.world_position - primary_surface.world_position);
    let primary_normal = bend_shading_normal(primary_surface.world_normal, wo_primary);

    // Nested-dielectric medium stack (see the pathtracer for the full notes):
    // the highest-priority entry is the active medium; boundaries of
    // lower-priority volumes inside it are false interfaces.
    var medium_id: array<u32, MEDIUM_STACK_SIZE>;
    var medium_priority: array<u32, MEDIUM_STACK_SIZE>;
    var medium_ior: array<f32, MEDIUM_STACK_SIZE>;
    var medium_dispersion: array<f32, MEDIUM_STACK_SIZE>;
    var medium_extinction_entry: array<vec3<f32>, MEDIUM_STACK_SIZE>;
    var medium_count = 0u;
    var medium_extinction = vec3(0.0);

    var wi: vec3<f32>;
    var p_bounce = 0.0;
    var ray_origin: vec3<f32>;

    // The caller selected the primary lobe (Fresnel-weighted split for
    // transmissive surfaces — deterministic, the Fresnel factors are applied
    // outside).
    if primary_lobe == PRIMARY_LOBE_REFRACT {
        // The camera is in air, so the far side of the interface is the material.
        var primary_ior = primary_surface.material.ior;
        if primary_surface.material.dispersion > 0.0 {
            primary_ior = dispersive_ior(primary_ior, primary_surface.material.dispersion, lambda_nm);
            throughput *= channel_mask;
            gate = 1.0;
            spectral_masked = true;
            *dispersed = true;
        }
        wi = refract(-wo_primary, primary_normal, 1.0 / primary_ior);
        p_bounce = bitcast<f32>(0x7F800000u); // INF: delta lobe
        medium_id[0] = primary_surface.material_id;
        medium_priority[0] = primary_surface.material.nested_priority;
        medium_ior[0] = primary_surface.material.ior;
        medium_dispersion[0] = primary_surface.material.dispersion;
        medium_extinction_entry[0] = primary_surface.material.extinction;
        medium_count = 1u;
        medium_extinction = primary_surface.material.extinction;
        ray_origin = offset_ray_origin(primary_surface.world_position, -primary_surface.world_normal);
    } else if primary_lobe == PRIMARY_LOBE_REFLECT {
        wi = reflect(-wo_primary, primary_normal);
        p_bounce = bitcast<f32>(0x7F800000u); // INF: delta lobe
        ray_origin = offset_ray_origin(primary_surface.world_position, primary_surface.world_normal);
    } else {
        // Opaque primary: sample the GGX specular lobe in tangent space; its
        // BRDF/pdf weight folds into the path throughput up front.
        let TBN = orthonormalize(primary_normal);
        let wo_tangent = vec3(dot(wo_primary, TBN[0]), dot(wo_primary, TBN[1]), dot(wo_primary, TBN[2]));
        let wi_tangent = sample_ggx_vndf(wo_tangent, primary_surface.material.roughness, rng);
        if ggx_vndf_sample_invalid(wi_tangent) {
            return vec3(0.0);
        }
        wi = wi_tangent.x * TBN[0] + wi_tangent.y * TBN[1] + wi_tangent.z * TBN[2];
        p_bounce = ggx_vndf_pdf(wo_tangent, wi_tangent, primary_surface.material.roughness);
        let NdotV = max(dot(primary_normal, wo_primary), 0.0001);
        let F_ab = F_AB(primary_surface.material.perceptual_roughness, NdotV);
        throughput = evaluate_specular_brdf(wo_primary, wi, primary_normal, primary_surface.material, F_ab);
        if primary_surface.material.roughness > MIRROR_ROUGHNESS_THRESHOLD {
            throughput /= p_bounce;
        }
        ray_origin = offset_ray_origin(primary_surface.world_position, primary_surface.world_normal);
    }

    var glossy_bounces = 0u;
    var crossings = 0u;
    var first_event = true;
    // Path length until the first NON-transmissive event — what the pixel is
    // actually showing. Reporting the first interface instead (a glass inner
    // wall is sub-mm away) puts the DLSS virtual point on the glass while the
    // visible content is meters beyond it, smearing refractions in motion.
    var path_t = 0.0;
    var first_interface_t = 0.0;
    var hit_distance_pending = true;
    loop {
        if glossy_bounces >= SPECULAR_GLOSSY_BOUNCES || crossings >= SPECULAR_GLASS_CROSSINGS {
            break;
        }
        // Portals teleport and black holes bend the ray mid-trace (so both
        // show up in reflections and through glass); origin/direction updated
        // in place. Disk emission rides back; captured rays end the chain
        // black (no sky).
        var traversal_emitted = vec3(0.0);
        var traversal_captured = 0u;
        let ray = trace_ray_traversal(&ray_origin, &wi, 0.0, RAY_FLAG_NONE, &traversal_emitted, &traversal_captured);
        radiance += gate * throughput * traversal_emitted;
        if traversal_captured != 0u {
            hit_distance_pending = false;
            break;
        }
        if ray.kind == RAY_QUERY_INTERSECTION_NONE {
            // Sky reflection — full weight, the sky isn't in next-event estimation.
            radiance += gate * throughput * sky_radiance(wi);
            hit_distance_pending = false; // content at infinity (caller's RAY_T_MAX init)
            break;
        }
        let hit = resolve_ray_hit_full(ray);
        let segment_length = length(hit.world_position - ray_origin);
        throughput *= exp(-medium_extinction * segment_length);
        path_t += segment_length;
        if first_event {
            first_interface_t = segment_length;
        }

        let wo = -wi;

        // Emission MIS. Later bounces weight against the BSDF sample that found
        // the hit (delta lobes have p = INF → weight 1). First event: full
        // weight only if DI did NOT do specular NEE (near-mirror or transmissive
        // primary); otherwise 0 to avoid double-counting reflected emissive
        // geometry that DI's light sampling already caught.
        var emissive_mis = 0.0;
        if !first_event {
            emissive_mis = power_heuristic(p_bounce, random_emissive_light_pdf(hit));
        } else if primary_surface.material.roughness <= SPECULAR_GI_FOR_DI_ROUGHNESS_THRESHOLD
            || primary_surface.material.specular_transmission > 0.0 {
            emissive_mis = 1.0;
        }
        radiance += gate * throughput * emissive_mis * hit.material.emissive;
        first_event = false;

        if rand_f(rng) < hit.material.specular_transmission {
            // Transmissive hit: cross the interface (no NEE at delta lobes).
            // Entering is tested against the true triangle normal — see the
            // pathtracer's twin of this branch for the reasoning.
            let entering = dot(wi, hit.geometric_world_normal) < 0.0;
            let oriented_geometric_normal = select(
                -hit.geometric_world_normal, hit.geometric_world_normal, entering);
            let oriented_normal = bend_shading_normal(
                select(-hit.world_normal, hit.world_normal, entering), wo);

            var self_index = MEDIUM_NOT_FOUND;
            var other_priority = 0u;
            var other_ior = 1.0;
            var other_dispersion = 0.0;
            for (var i = 0u; i < medium_count; i += 1u) {
                if medium_id[i] == hit.material_id {
                    self_index = i;
                    continue;
                }
                if medium_priority[i] >= other_priority {
                    other_priority = medium_priority[i];
                    other_ior = medium_ior[i];
                    other_dispersion = medium_dispersion[i];
                }
            }
            let true_interface = hit.material.nested_priority >= other_priority;

            var crossed = true;
            if true_interface {
                var self_ior = hit.material.ior;
                var far_ior = other_ior;
                if hit.material.dispersion > 0.0 || other_dispersion > 0.0 {
                    // Dispersive interface: this chain collapses to its
                    // channel (once) and refracts at n(λ) on both sides.
                    if !spectral_masked {
                        throughput *= channel_mask;
                        gate = 1.0;
                        spectral_masked = true;
                    }
                    *dispersed = true;
                    self_ior = dispersive_ior(self_ior, hit.material.dispersion, lambda_nm);
                    far_ior = dispersive_ior(far_ior, other_dispersion, lambda_nm);
                }
                let eta = select(
                    self_ior / far_ior,
                    far_ior / self_ior,
                    entering,
                );
                // Deterministic dominant lobe — no per-frame coin flip (the
                // denoiser needs a temporally stable signal): follow refract
                // unless Fresnel favors reflection (incl. TIR), weighted by
                // the followed lobe's Fresnel factor. The minority lobe's
                // energy is dropped: slight darkening at grazing interior
                // angles, traded for stability in motion.
                let reflectance = fresnel_dielectric(min(dot(wo, oriented_normal), 1.0), eta);
                if reflectance > 0.5 {
                    wi = reflect(-wo, oriented_normal);
                    crossed = false;
                    throughput *= reflectance;
                } else {
                    wi = refract(-wo, oriented_normal, eta);
                    throughput *= 1.0 - reflectance;
                }
                p_bounce = bitcast<f32>(0x7F800000u);
            }

            if crossed {
                if entering {
                    if self_index == MEDIUM_NOT_FOUND && medium_count < MEDIUM_STACK_SIZE {
                        medium_id[medium_count] = hit.material_id;
                        medium_priority[medium_count] = hit.material.nested_priority;
                        medium_ior[medium_count] = hit.material.ior;
                        medium_dispersion[medium_count] = hit.material.dispersion;
                        medium_extinction_entry[medium_count] = hit.material.extinction;
                        medium_count += 1u;
                    }
                } else if self_index != MEDIUM_NOT_FOUND {
                    medium_count -= 1u;
                    medium_id[self_index] = medium_id[medium_count];
                    medium_priority[self_index] = medium_priority[medium_count];
                    medium_ior[self_index] = medium_ior[medium_count];
                    medium_dispersion[self_index] = medium_dispersion[medium_count];
                    medium_extinction_entry[self_index] = medium_extinction_entry[medium_count];
                }
                medium_extinction = vec3(0.0);
                var active_priority = 0u;
                var found = false;
                for (var i = 0u; i < medium_count; i += 1u) {
                    if !found || medium_priority[i] >= active_priority {
                        active_priority = medium_priority[i];
                        medium_extinction = medium_extinction_entry[i];
                        found = true;
                    }
                }
            }

            let offset_normal = select(oriented_geometric_normal, -oriented_geometric_normal, crossed);
            ray_origin = offset_ray_origin(hit.world_position, offset_normal);
            crossings += 1u;
            continue;
        }

        if hit_distance_pending {
            *first_hit_t = path_t;
            hit_distance_pending = false;
        }

        let hit_normal = bend_shading_normal(hit.world_normal, wo);
        let TBN = orthonormalize(hit_normal);
        let T = TBN[0];
        let B = TBN[1];
        let N = TBN[2];
        let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));
        let NdotV = max(dot(hit_normal, wo), 0.0001);
        let F_ab = F_AB(hit.material.perceptual_roughness, NdotV);

        let is_mirror = hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && hit.material.metallic > 0.9999;
        if !is_mirror {
            let light = sample_random_light(hit.world_position, hit_normal, rng);
            let mis = specular_nee_mis_weight(light.inverse_pdf, light.brdf_rays_can_hit, light.wi, wo_tangent, hit.material.roughness, TBN);
            radiance += gate * throughput * mis * light.radiance * light.inverse_pdf * evaluate_brdf(wo, light.wi, hit_normal, hit.material, F_ab);
        }

        // End the chain at rough hits (after their NEE): continuing means a
        // near-random GGX direction each frame — sky one frame, wall the
        // next — and through glass this single-sample variance is the whole
        // signal, flickering pixels the denoiser then smears in motion.
        // Mirror-like hits continue (their continuation is deterministic).
        if hit.material.roughness > SPECULAR_PATH_CONTINUE_ROUGHNESS {
            // A surface reached THROUGH glass has no other lighting channel
            // (the diffuse passes shade directly-visible surfaces only), so
            // where its sun NEE is occluded — under a glass object's own base
            // — it would go pitch black; the pathtracer fills the same spot
            // with skylight via path continuation. Approximate that with one
            // deterministic env sample along the normal: irradiance ≈ π·L_sky,
            // diffuse brdf = albedo/π ⇒ albedo · L_sky.
            if crossings > 0u {
                radiance += gate * throughput * hit.material.base_color * sky_radiance(hit_normal);
            }
            break;
        }
        let next_tangent = sample_ggx_vndf(wo_tangent, hit.material.roughness, rng);
        if ggx_vndf_sample_invalid(next_tangent) {
            break;
        }
        wi = next_tangent.x * T + next_tangent.y * B + next_tangent.z * N;
        ray_origin = offset_ray_origin(hit.world_position, hit.geometric_world_normal);
        p_bounce = ggx_vndf_pdf(wo_tangent, next_tangent, hit.material.roughness);
        throughput *= evaluate_brdf(wo, wi, N, hit.material, F_ab);
        if hit.material.roughness > MIRROR_ROUGHNESS_THRESHOLD {
            throughput /= p_bounce;
        }
        glossy_bounces += 1u;

        let p = min(luminance(throughput), 0.95);
        if rand_f(rng) > p {
            break;
        }
        throughput /= p;
    }

    // Path ended without reaching opaque content (TIR cascade exhausted the
    // crossing budget): report the first interface, not RAY_T_MAX — pixels
    // flickering between a finite content distance and infinity destabilize
    // the DLSS hit-distance guide and smear the whole object in motion.
    if hit_distance_pending {
        *first_hit_t = first_interface_t;
    }

    return radiance;
}

fn specular_nee_mis_weight(inverse_p_light: f32, brdf_rays_can_hit: bool, light_wi: vec3<f32>, wo_tangent: vec3<f32>, roughness: f32, TBN: mat3x3<f32>) -> f32 {
    if !brdf_rays_can_hit {
        return 1.0;
    }
    let wi_tangent = vec3(dot(light_wi, TBN[0]), dot(light_wi, TBN[1]), dot(light_wi, TBN[2]));
    let p_light = 1.0 / inverse_p_light;
    let p_bounce = ggx_vndf_pdf(wo_tangent, wi_tangent, roughness);
    return power_heuristic(p_light, p_bounce);
}

// --------------------------------------------------------------- shared reuse

// Reproject the current world position to the previous frame and validate the
// previous-frame surface. The reservoir load is left to the caller (DI/GI use
// different buffers).
fn reproject_previous(pixel: vec2<u32>, world_position: vec3<f32>, world_normal: vec3<f32>) -> Reprojection {
    var result: Reprojection;
    result.valid = false;

    // Follow the motion vector back to the previous-frame pixel.
    let motion = textureLoad(motion_vectors, pixel).xy;
    let previous_pixel_f = vec2<f32>(pixel) - motion * view.main_pass_viewport.zw;
    if any(previous_pixel_f < vec2(0.0)) || any(previous_pixel_f >= view.main_pass_viewport.zw) {
        return result;
    }
    let previous_pixel = vec2<u32>(previous_pixel_f);

    let prev_gpos = textureLoad(previous_gbuffer_position, previous_pixel);
    if prev_gpos.w < 0.0 {
        return result;
    }
    let prev_normal = textureLoad(previous_gbuffer_normal, previous_pixel).xyz;
    if !surface_similar(world_position, world_normal, prev_gpos.xyz, prev_normal) {
        return result;
    }

    result.valid = true;
    result.pixel = previous_pixel;
    result.world_position = prev_gpos.xyz;
    result.world_normal = prev_normal;
    return result;
}

// Pick one current-frame neighbour pixel (disk around the center) whose surface
// matches the center — up to SPATIAL_ATTEMPTS tries. Used by both DI and GI
// spatial reuse.
fn find_spatial_neighbor(center_pixel: vec2<u32>, surface: Surface, rng: ptr<function, u32>) -> Reprojection {
    var result: Reprojection;
    result.valid = false;

    for (var i = 0u; i < SPATIAL_ATTEMPTS; i++) {
        let offset = sample_disk(SPATIAL_RADIUS_PIXELS, rng);
        let neighbor_f = clamp(vec2<f32>(center_pixel) + offset, vec2(0.0), view.main_pass_viewport.zw - 1.0);
        let pixel = vec2<u32>(neighbor_f);

        let neighbor_gpos = textureLoad(gbuffer_position, pixel);
        if neighbor_gpos.w < 0.0 {
            continue;
        }
        let neighbor_normal = textureLoad(gbuffer_normal, pixel).xyz;
        if !surface_similar(surface.world_position, surface.world_normal, neighbor_gpos.xyz, neighbor_normal) {
            continue;
        }

        result.valid = true;
        result.pixel = pixel;
        result.world_position = neighbor_gpos.xyz;
        result.world_normal = neighbor_normal;
        return result;
    }
    return result;
}

// --------------------------------------------------------------------- DI

// WRS over INITIAL_SAMPLES candidates drawn from the surface point's ReGIR
// cell (LIGHTS pre-selected for proximity; a fresh point is sampled per
// candidate — see the cell branch), or from one presampled light tile while
// the cell is cold (tile entries are points, used as-is). The selected
// `LightSample` identity is what the reservoir stores for exact re-resolution
// at merge/shade time. Note the cell path re-resolves the light per candidate
// (a cluster walk for the picked triangle) — cheap for lamp-sized emissive
// meshes; a scene with huge emissive meshes would want this bounded.
fn generate_initial_reservoir(tile_start: u32, regir_cell: u32, world_position: vec3<f32>, world_normal: vec3<f32>, diffuse_brdf: vec3<f32>, rng: ptr<function, u32>) -> Reservoir {
    var reservoir = empty_reservoir();
    var weight_sum = 0.0;
    var reservoir_target_function = 0.0;
    let mis_weight = 1.0 / f32(INITIAL_SAMPLES);

    for (var i = 0u; i < INITIAL_SAMPLES; i++) {
        var candidate_light_id: u32;
        var candidate_seed: u32;
        var resolved: ResolvedLightSample;
        if regir_cell != REGIR_CELL_NONE {
            // The cell entry selects a LIGHT (light-space contribution weight);
            // sample a FRESH point on it for this pixel. Pinning the entry's
            // stored point would correlate every receiver in the cell onto one
            // spot of the emitter — the lamp's illumination (and the shadows
            // it casts) would swing direction with each frame's entry roll
            // instead of resolving the penumbra spatially across pixels.
            let entry = regir_samples[regir_cell * REGIR_ENTRIES_PER_CELL + rand_range_u(REGIR_ENTRIES_PER_CELL, rng)];
            if entry.light_id == NULL_LIGHT_ID {
                continue;
            }
            let slot = entry.light_id >> 16u;
            let light_source = light_sources[slot];
            let triangle_id = rand_range_u(light_source.kind >> 1u, rng);
            candidate_light_id = (slot << 16u) | triangle_id;
            candidate_seed = rand_u(rng);
            resolved = resolve_light_sample(LightSample(candidate_light_id, candidate_seed), light_source);
            // light weight × fresh point's area inverse-pdf
            resolved.inverse_pdf *= entry.inverse_pdf;
        } else {
            let entry = light_tiles[tile_start + rand_range_u(LIGHT_TILE_SAMPLES_PER_BLOCK, rng)];
            if entry.light_id == NULL_LIGHT_ID {
                continue;
            }
            candidate_light_id = entry.light_id;
            candidate_seed = entry.seed;
            resolved = unpack_light_tile_sample(entry);
        }
        let contribution = calculate_resolved_light_contribution(resolved, world_position, world_normal);
        var target_function = luminance(contribution.radiance * diffuse_brdf * saturate(dot(contribution.wi, world_normal)));
        // Shadowed target at contact range (see SHADOWED_TARGET_DISTANCE_SQUARED).
        // Valid RIS: the contribution weight divides by this same target.
        let to_light = resolved.world_position.xyz - world_position;
        if target_function > 0.0 && dot(to_light, to_light) < SHADOWED_TARGET_DISTANCE_SQUARED {
            target_function *= trace_light_visibility(world_position + world_normal * RAY_T_MIN, resolved.world_position);
        }
        let resampling_weight = mis_weight * (target_function * contribution.inverse_pdf);

        weight_sum += resampling_weight;
        if rand_f(rng) < resampling_weight / weight_sum {
            reservoir.light_id = candidate_light_id;
            reservoir.seed = candidate_seed;
            reservoir_target_function = target_function;
        }
    }

    if reservoir_target_function > 0.0 {
        reservoir.unbiased_contribution_weight = weight_sum / reservoir_target_function;
    }
    reservoir.confidence_weight = 1.0;
    return reservoir;
}

fn load_temporal_reservoir(pixel: vec2<u32>, world_position: vec3<f32>, world_normal: vec3<f32>) -> Reservoir {
    let reprojection = reproject_previous(pixel, world_position, world_normal);
    if !reprojection.valid {
        return empty_reservoir();
    }

    var reservoir = load_reservoir_a(reprojection.pixel);
    // `light_id` packs the light's STABLE slot, so history survives lights
    // being added/removed; a dead light's slot resolves to a `NONE` hole and
    // the reservoir is rejected in `reservoir_contribution`. (A freed slot's
    // reuse can briefly misattribute history to the new occupant — bounded by
    // the confidence cap.)
    reservoir.confidence_weight = min(reservoir.confidence_weight, DI_CONFIDENCE_WEIGHT_CAP);

    // Visibility recheck (one shadow ray): the merge's target function is
    // unshadowed, so an occluded history sample would keep winning merges on
    // pure intensity and smear a bright spot for its whole confidence
    // lifetime. Discarding it here lets this frame's initial candidates take
    // the pixel immediately.
    if reservoir_valid(reservoir) {
        let light_source = light_sources[reservoir.light_id >> 16u];
        if light_source.kind == LIGHT_SOURCE_KIND_NONE {
            return empty_reservoir();
        }
        let sample = LightSample(reservoir.light_id, reservoir.seed);
        let resolved = resolve_light_sample(sample, light_source);
        let visibility = trace_light_visibility(world_position + world_normal * RAY_T_MIN, resolved.world_position);
        if visibility == 0.0 {
            return empty_reservoir();
        }
    }
    return reservoir;
}

fn merge_reservoirs(
    canonical: Reservoir,
    other: Reservoir,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    diffuse_brdf: vec3<f32>,
    rng: ptr<function, u32>,
) -> MergeResult {
    let canonical_contribution = reservoir_contribution(canonical, world_position, world_normal, diffuse_brdf);
    let other_contribution = reservoir_contribution(other, world_position, world_normal, diffuse_brdf);

    var combined = empty_reservoir();
    combined.confidence_weight = canonical.confidence_weight + other.confidence_weight;

    let m_total = combined.confidence_weight;
    let m_canonical = select(0.0, canonical.confidence_weight / m_total, m_total > 0.0);
    let m_other = select(0.0, other.confidence_weight / m_total, m_total > 0.0);
    let canonical_weight = m_canonical * canonical_contribution.target_function * canonical.unbiased_contribution_weight;
    let other_weight = m_other * other_contribution.target_function * other.unbiased_contribution_weight;
    let weight_sum = canonical_weight + other_weight;

    if rand_f(rng) < other_weight / weight_sum {
        combined.light_id = other.light_id;
        combined.seed = other.seed;
        combined.unbiased_contribution_weight = select(0.0, weight_sum / other_contribution.target_function, other_contribution.target_function > 0.0);
        return MergeResult(combined, other_contribution.radiance, other_contribution.wi, other_contribution.world_position);
    }
    combined.light_id = canonical.light_id;
    combined.seed = canonical.seed;
    combined.unbiased_contribution_weight = select(0.0, weight_sum / canonical_contribution.target_function, canonical_contribution.target_function > 0.0);
    return MergeResult(combined, canonical_contribution.radiance, canonical_contribution.wi, canonical_contribution.world_position);
}

fn reservoir_contribution(reservoir: Reservoir, world_position: vec3<f32>, world_normal: vec3<f32>, diffuse_brdf: vec3<f32>) -> ReservoirContribution {
    if !reservoir_valid(reservoir) {
        return ReservoirContribution(vec3(0.0), 0.0, vec3(0.0), vec4(0.0));
    }
    // The stored slot's light died (slot holed out as `NONE`): resolving it
    // would read another instance's geometry — reject the reservoir instead.
    let light_source = light_sources[reservoir.light_id >> 16u];
    if light_source.kind == LIGHT_SOURCE_KIND_NONE {
        return ReservoirContribution(vec3(0.0), 0.0, vec3(0.0), vec4(0.0));
    }
    let sample = LightSample(reservoir.light_id, reservoir.seed);
    let resolved = resolve_light_sample(sample, light_source);
    let contribution = calculate_resolved_light_contribution(resolved, world_position, world_normal);
    let target_function = luminance(contribution.radiance * diffuse_brdf * saturate(dot(contribution.wi, world_normal)));
    return ReservoirContribution(contribution.radiance, target_function, contribution.wi, resolved.world_position);
}

// --------------------------------------------------------------------- GI

fn gi_generate_initial(world_position: vec3<f32>, world_normal: vec3<f32>, rng: ptr<function, u32>) -> GiReservoir {
    var reservoir = gi_empty();

    let ray_direction = sample_uniform_hemisphere(world_normal, rng);
    let ray = trace_ray(world_position + world_normal * RAY_T_MIN, ray_direction, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
    if ray.kind == RAY_QUERY_INTERSECTION_NONE {
        // Sky miss → skylight as a reconnection vertex at a far virtual
        // distance. Reuse stays consistent: the shade visibility ray re-traces
        // toward (effectively) the same direction, and the spatial/temporal
        // jacobian degenerates to ~1 at this range.
        if solari_view.environment_brightness > 0.0 {
            reservoir.sample_point_world_position = world_position + ray_direction * SKY_VERTEX_DISTANCE;
            reservoir.sample_point_world_normal = -ray_direction;
            reservoir.confidence_weight = 1.0;
            reservoir.radiance = sky_radiance(ray_direction);
            reservoir.unbiased_contribution_weight = uniform_hemisphere_inverse_pdf();
        }
        return reservoir;
    }

    let sample_point = resolve_ray_hit_full(ray);
    // Emissive reconnection vertices are covered by direct lighting; skip them
    // here to avoid double counting.
    if any(sample_point.material.emissive != vec3(0.0)) {
        return reservoir;
    }

    reservoir.sample_point_world_position = sample_point.world_position;
    reservoir.sample_point_world_normal = sample_point.world_normal;
    reservoir.confidence_weight = 1.0;

    // One bounce of indirect: direct lighting at the reconnection vertex.
    let direct_lighting = sample_random_light(sample_point.world_position, sample_point.world_normal, rng);
    reservoir.radiance = direct_lighting.radiance * saturate(dot(direct_lighting.wi, sample_point.world_normal));
    reservoir.unbiased_contribution_weight = direct_lighting.inverse_pdf * uniform_hemisphere_inverse_pdf();
    reservoir.radiance *= sample_point.material.base_color / PI;
    return reservoir;
}

fn gi_load_temporal(pixel: vec2<u32>, world_position: vec3<f32>, world_normal: vec3<f32>) -> GiNeighbor {
    var neighbor: GiNeighbor;
    neighbor.reservoir = gi_empty();
    neighbor.world_position = vec3(0.0);

    let reprojection = reproject_previous(pixel, world_position, world_normal);
    if !reprojection.valid {
        return neighbor;
    }

    var reservoir = gi_reservoir_a[reservoir_index(reprojection.pixel)];
    reservoir.confidence_weight = min(reservoir.confidence_weight, GI_CONFIDENCE_WEIGHT_CAP);
    neighbor.reservoir = reservoir;
    neighbor.world_position = reprojection.world_position;
    return neighbor;
}

fn gi_merge(
    canonical: GiReservoir,
    other: GiReservoir,
    other_world_position: vec3<f32>,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    diffuse_brdf: vec3<f32>,
    rng: ptr<function, u32>,
) -> GiMergeResult {
    let canonical_wi = normalize(canonical.sample_point_world_position - world_position);
    let other_wi = normalize(other.sample_point_world_position - world_position);

    let canonical_tf = luminance(canonical.radiance * saturate(dot(canonical_wi, world_normal)) * diffuse_brdf);
    let other_tf = luminance(other.radiance * saturate(dot(other_wi, world_normal)) * diffuse_brdf);

    // Reconnect the other sample's vertex to this pixel (change of measure).
    let jacobian = gi_jacobian(world_position, other_world_position, other.sample_point_world_position, other.sample_point_world_normal);
    // Huge jacobians explode the variance — keep the canonical sample instead.
    if jacobian > MAX_JACOBIAN {
        return GiMergeResult(canonical, canonical.radiance, canonical_wi);
    }

    var combined = gi_empty();
    combined.confidence_weight = canonical.confidence_weight + other.confidence_weight;

    let m_total = combined.confidence_weight;
    let m_canonical = select(0.0, canonical.confidence_weight / m_total, m_total > 0.0);
    let m_other = select(0.0, other.confidence_weight / m_total, m_total > 0.0);
    let canonical_weight = m_canonical * canonical_tf * canonical.unbiased_contribution_weight;
    let other_weight = m_other * (other_tf * jacobian) * other.unbiased_contribution_weight;
    combined.weight_sum = canonical_weight + other_weight;

    if rand_f(rng) < other_weight / combined.weight_sum {
        combined.sample_point_world_position = other.sample_point_world_position;
        combined.sample_point_world_normal = other.sample_point_world_normal;
        combined.radiance = other.radiance;
        combined.unbiased_contribution_weight = select(0.0, combined.weight_sum / other_tf, other_tf > 0.0);
        return GiMergeResult(combined, other.radiance, other_wi);
    }
    combined.sample_point_world_position = canonical.sample_point_world_position;
    combined.sample_point_world_normal = canonical.sample_point_world_normal;
    combined.radiance = canonical.radiance;
    combined.unbiased_contribution_weight = select(0.0, combined.weight_sum / canonical_tf, canonical_tf > 0.0);
    return GiMergeResult(combined, canonical.radiance, canonical_wi);
}

fn gi_jacobian(new_world_position: vec3<f32>, original_world_position: vec3<f32>, sample_point_world_position: vec3<f32>, sample_point_world_normal: vec3<f32>) -> f32 {
    let r = new_world_position - sample_point_world_position;
    let q = original_world_position - sample_point_world_position;
    let rl = length(r);
    let ql = length(q);
    let phi_r = saturate(dot(r / rl, sample_point_world_normal));
    let phi_q = saturate(dot(q / ql, sample_point_world_normal));
    let jacobian = (phi_r * ql * ql) / (phi_q * rl * rl);
    return select(jacobian, 0.0, isinf(jacobian) || isnan(jacobian));
}

// --------------------------------------------------------------------- helpers

/// Virtual reconnection-vertex distance for skylight GI samples — far enough
/// that the reuse jacobian is ~1, well inside `RAY_T_MAX` so the shade pass's
/// visibility ray still traces.
const SKY_VERTEX_DISTANCE = 10000.0;

/// The sky's raw radiance in `direction` (the baked atmosphere / skybox cube ×
/// its brightness). Zero when the view has neither.
fn sky_radiance(direction: vec3<f32>) -> vec3<f32> {
    if solari_view.environment_brightness == 0.0 {
        return vec3(0.0);
    }
    let sky = textureSampleLevel(environment_map, environment_map_sampler, direction, 0.0).rgb;
    return sky * solari_view.environment_brightness;
}

/// The camera ray through `pixel` (jittered, like the visibility pass's).
fn primary_ray_direction(pixel: vec2<u32>) -> vec3<f32> {
    let pixel_center = vec2<f32>(pixel) + 0.5;
    let pixel_uv = pixel_center / view.main_pass_viewport.zw;
    let pixel_ndc = pixel_uv * 2.0 - 1.0;
    let ray_target = view.world_from_clip * vec4(pixel_ndc.x, -pixel_ndc.y, 1.0, 1.0);
    return normalize((ray_target.xyz / ray_target.w) - view.world_position);
}

fn load_surface(pixel: vec2<u32>) -> Surface {
    var surface: Surface;
    let gpos = textureLoad(gbuffer_position, pixel);
    surface.valid = gpos.w >= 0.0;
    if !surface.valid {
        return surface;
    }
    surface.world_position = gpos.xyz;
    surface.world_normal = textureLoad(gbuffer_normal, pixel).xyz;
    let uv = textureLoad(gbuffer_uv, pixel).xy;
    surface.material_id = u32(gpos.w);
    surface.material = resolve_material(materials[surface.material_id], uv);
    return surface;
}

fn surface_similar(p0: vec3<f32>, n0: vec3<f32>, p1: vec3<f32>, n1: vec3<f32>) -> bool {
    let camera_distance = length(view.world_position - p0);
    // Plane distance, not point distance: a translating surface (a moving
    // car, the road sliding under it) stays within its own tangent plane, so
    // its history survives — while a depth disocclusion still rejects. A
    // point-distance test invalidates movers every frame, which strobes their
    // lighting and leaves denoiser afterimages where they were.
    if abs(dot(p0 - p1, n1)) > 0.01 * camera_distance {
        return false;
    }
    return dot(n0, n1) > 0.9;
}

fn isinf(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u;
}

fn reservoir_index(pixel: vec2<u32>) -> u32 {
    return pixel.x + pixel.y * u32(view.main_pass_viewport.z);
}

fn empty_reservoir() -> Reservoir {
    return Reservoir(NULL_LIGHT_ID, 0u, 0.0, 0.0);
}

fn reservoir_valid(reservoir: Reservoir) -> bool {
    return reservoir.light_id != NULL_LIGHT_ID;
}

fn gi_empty() -> GiReservoir {
    return GiReservoir(vec3(0.0), 0.0, vec3(0.0), 0.0, vec3(0.0), 0.0);
}

fn pack_reservoir(reservoir: Reservoir) -> vec4<u32> {
    return vec4(
        reservoir.light_id,
        reservoir.seed,
        bitcast<u32>(reservoir.confidence_weight),
        bitcast<u32>(reservoir.unbiased_contribution_weight),
    );
}

fn unpack_reservoir(packed: vec4<u32>) -> Reservoir {
    return Reservoir(packed.x, packed.y, bitcast<f32>(packed.z), bitcast<f32>(packed.w));
}

fn load_reservoir_a(pixel: vec2<u32>) -> Reservoir {
    return unpack_reservoir(reservoir_a[reservoir_index(pixel)]);
}

fn load_reservoir_b(pixel: vec2<u32>) -> Reservoir {
    return unpack_reservoir(reservoir_b[reservoir_index(pixel)]);
}

fn store_reservoir_a(pixel: vec2<u32>, reservoir: Reservoir) {
    reservoir_a[reservoir_index(pixel)] = pack_reservoir(reservoir);
}

fn store_reservoir_b(pixel: vec2<u32>, reservoir: Reservoir) {
    reservoir_b[reservoir_index(pixel)] = pack_reservoir(reservoir);
}

// ------------------------------------------------------------------ debug view
//
// Reservoir / ReGIR visualizations (`SolariDebugView::{DiWeight, DiConfidence,
// DiLight, RegirCells}`). Dispatched by the restir node after compose when
// `solari_view.debug_mode != 0` — these read buffers only this bind group
// sees, so they can't live in the generic overlay pass.

fn debug_hash_color(id: u32) -> vec3<f32> {
    let h = id * 747796405u + 2891336453u;
    return vec3(
        f32((h >> 0u) & 1023u),
        f32((h >> 10u) & 1023u),
        f32((h >> 20u) & 1023u),
    ) / 1023.0;
}

/// Blue → green → red heat ramp over `x` in [0, 1].
fn debug_heat(x: f32) -> vec3<f32> {
    let t = clamp(x, 0.0, 1.0);
    return vec3(smoothstep(0.5, 1.0, t), 1.0 - abs(t - 0.5) * 2.0, 1.0 - smoothstep(0.0, 0.5, t));
}

@compute @workgroup_size(8, 8, 1)
fn restir_debug(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }
    let pixel = global_id.xy;

    let surface = load_surface(pixel);
    if !surface.valid {
        textureStore(view_output, pixel, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    var color = vec3(0.0);
    let reservoir = load_reservoir_a(pixel);
    switch solari_view.debug_mode {
        case 1u: { // DI contribution weight: log heat; NaN = magenta, invalid = black.
            let w = reservoir.unbiased_contribution_weight;
            if isnan(w) {
                color = vec3(1.0, 0.0, 1.0);
            } else if reservoir_valid(reservoir) {
                color = debug_heat(log2(1.0 + w) / 16.0);
            }
        }
        case 2u: { // DI confidence (M) over its cap.
            color = vec3(reservoir.confidence_weight / DI_CONFIDENCE_WEIGHT_CAP);
        }
        case 3u: { // DI light identity (stable slot), hashed.
            if reservoir_valid(reservoir) {
                color = debug_hash_color(reservoir.light_id >> 16u);
            }
        }
        case 4u: { // ReGIR cell, hashed; red = cold/none.
            var rng = reservoir_index(pixel) + view.frame_count * 5782582u;
            let cell = regir_find(surface.world_position, surface.world_normal, view.world_position, &rng);
            if cell == REGIR_CELL_NONE {
                color = vec3(1.0, 0.0, 0.0);
            } else {
                color = debug_hash_color(cell);
            }
        }
        default: {}
    }
    textureStore(view_output, pixel, vec4(color, 1.0));
}
