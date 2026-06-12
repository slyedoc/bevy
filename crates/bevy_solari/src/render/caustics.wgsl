enable wgpu_ray_query;

// Caustic photon passes (phase B).
//
// Everything is GPU-driven, in scene-column terms: `caustic_prepare_reduce`
// scans the live instance columns (material ids → transmissive?, mesh-local
// AABBs × LIVE transforms → world bounds) and reduces the union projected
// perpendicular to the first directional light into an emission rect;
// `caustic_prepare_finalize` turns the reduction into per-photon emission
// parameters (power = illuminance × rect area / photon count, so the
// estimate is unbiased however tight the rect is). No CPU mirror of the
// scene is consulted — material swaps, GPU-propagated transforms, and sun
// motion are all picked up the frame they happen.
//
// `caustic_decay` ages the world-space photon grid (a 16-frame exponential
// running average — changes re-converge with no explicit invalidation).
// `caustic_emit` then fires photons over the rect, traces them through the
// glass code — exact Fresnel delta lobes, nested-dielectric media,
// Beer-Lambert absorption, hero-wavelength dispersion — and deposits the
// survivors that land on a DIFFUSE surface after crossing at least one real
// glass interface. Direct photons deposit nothing (NEE already shades
// direct light); the grid holds exactly the transport the reservoirs can't
// find. The shade pass gathers via `caustic_gather` in
// `restir_bindings.wgsl`.

#import bevy_solari::pbr::{rand_f, rand_vec2f}
#import bevy_solari::brdf::{sample_glass_bsdf, dispersive_ior, spectral_rgb_weight, sample_hero_wavelength}
#import bevy_solari::scene_bindings::{trace_ray, resolve_ray_hit_full, offset_ray_origin, affine_transform_point, transforms, material_ids, materials, instance_aabbs, light_sources, active_light_list, directional_lights, fog_volumes_range, fog_volumes_sample, RAY_T_MAX}
#import bevy_solari::restir_bindings::{view, caustic_deposit, caustic_deposit_volume, caustic_decay_cell, caustic_table_size, caustic_emitter, caustic_emitter_load, caustic_emitter_store, caustic_float_to_orderable, caustic_orderable_to_float}

// Keep in sync with `CAUSTIC_PHOTONS` in `render/caustics.rs`.
const CAUSTIC_PHOTONS = 262144u;
// Specular events per photon: bounded instead of rouletted — a photon deep
// in total internal reflection is exactly the energy caustics are made of.
const PHOTON_MAX_EVENTS = 24u;

const MEDIUM_STACK_SIZE = 4u;
const MEDIUM_NOT_FOUND = 0xFFFFFFFFu;

const PI_2 = 6.283185307;

/// The first ACTIVE directional light's photon-travel direction, or w = 0
/// when the scene has none. Shared by reduce + finalize so the projection
/// basis is bit-identical in both.
fn caustic_sun_direction() -> vec4<f32> {
    let emissive_count = active_light_list[0];
    if active_light_list[1] == 0u {
        return vec4(0.0);
    }
    let slot = active_light_list[2u + emissive_count];
    let sun = directional_lights[light_sources[slot].id];
    return vec4(-sun.direction_to_light, 1.0);
}

/// Deterministic basis perpendicular to `direction` (matches nothing else —
/// just stable per frame).
fn caustic_rect_u(direction: vec3<f32>) -> vec3<f32> {
    if abs(direction.x) > abs(direction.z) {
        return normalize(vec3(-direction.y, direction.x, 0.0));
    }
    return normalize(vec3(0.0, -direction.z, direction.y));
}

/// Reset the reduction scratch (one thread).
@compute @workgroup_size(1, 1, 1)
fn caustic_prepare_reset(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let max_orderable = caustic_float_to_orderable(3.4e38);
    let min_orderable = caustic_float_to_orderable(-3.4e38);
    atomicStore(&caustic_emitter[0], max_orderable); // min_u
    atomicStore(&caustic_emitter[1], min_orderable); // max_u
    atomicStore(&caustic_emitter[2], max_orderable); // min_v
    atomicStore(&caustic_emitter[3], min_orderable); // max_v
    atomicStore(&caustic_emitter[4], max_orderable); // min_depth
    atomicStore(&caustic_emitter[5], 0u);            // transmissive count
}

/// One thread per instance slot: transmissive instances project their world
/// bounds (mesh-local AABB × live transform) onto the sun-perpendicular
/// basis and reduce the union. Freed slots keep stale columns until reuse —
/// a stale glass entry only inflates the rect (wasted photons, dimmer
/// caustics), never wrong light.
@compute @workgroup_size(256, 1, 1)
fn caustic_prepare_reduce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot = global_id.x;
    if slot >= arrayLength(&material_ids) {
        return;
    }
    let sun = caustic_sun_direction();
    if sun.w == 0.0 {
        return;
    }
    let material = materials[material_ids[slot]];
    if material.specular_transmission <= 0.0 {
        return;
    }
    let aabb = instance_aabbs[slot];
    // Never-bound slots / grown-buffer tail: zero half extents span nothing.
    if all(aabb.half_extent.xyz == vec3(0.0)) {
        return;
    }
    let direction = sun.xyz;
    let u_axis = caustic_rect_u(direction);
    let v_axis = cross(direction, u_axis);
    let transform = transforms[slot];
    for (var i = 0u; i < 8u; i += 1u) {
        let signs = vec3(
            select(-1.0, 1.0, (i & 1u) != 0u),
            select(-1.0, 1.0, (i & 2u) != 0u),
            select(-1.0, 1.0, (i & 4u) != 0u),
        );
        let corner = affine_transform_point(
            transform, aabb.center.xyz + signs * aabb.half_extent.xyz);
        atomicMin(&caustic_emitter[0], caustic_float_to_orderable(dot(corner, u_axis)));
        atomicMax(&caustic_emitter[1], caustic_float_to_orderable(dot(corner, u_axis)));
        atomicMin(&caustic_emitter[2], caustic_float_to_orderable(dot(corner, v_axis)));
        atomicMax(&caustic_emitter[3], caustic_float_to_orderable(dot(corner, v_axis)));
        atomicMin(&caustic_emitter[4], caustic_float_to_orderable(dot(corner, direction)));
    }
    atomicAdd(&caustic_emitter[5], 1u);
}

/// Turn the reduction into emission parameters (one thread).
@compute @workgroup_size(1, 1, 1)
fn caustic_prepare_finalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sun = caustic_sun_direction();
    if sun.w == 0.0 || atomicLoad(&caustic_emitter[5]) == 0u {
        caustic_emitter_store(27u, 0.0);
        return;
    }
    let min_u = caustic_orderable_to_float(atomicLoad(&caustic_emitter[0]));
    let max_u = caustic_orderable_to_float(atomicLoad(&caustic_emitter[1]));
    let min_v = caustic_orderable_to_float(atomicLoad(&caustic_emitter[2]));
    let max_v = caustic_orderable_to_float(atomicLoad(&caustic_emitter[3]));
    let min_depth = caustic_orderable_to_float(atomicLoad(&caustic_emitter[4]));

    let direction = sun.xyz;
    let u_axis = caustic_rect_u(direction);
    let v_axis = cross(direction, u_axis);
    // 5 cm margin so jittered edges still cover silhouettes; rect anchored
    // just up-beam of the nearest glass, photons start far further up so
    // occluders between the sun and the glass (a window shutter, a roof)
    // block them exactly like light.
    let half = vec2(max_u - min_u, max_v - min_v) * 0.5 + 0.05;
    let center2 = vec2(max_u + min_u, max_v + min_v) * 0.5;
    let center = u_axis * center2.x + v_axis * center2.y + direction * (min_depth - 1.0);

    // Per-photon power: sun illuminance (disk radiance × its solid angle)
    // over the rect area, split across the photon budget.
    let emissive_count = active_light_list[0];
    let sun_light = directional_lights[light_sources[active_light_list[2u + emissive_count]].id];
    let solid_angle = PI_2 * (1.0 - sun_light.cos_theta_max);
    let area = 4.0 * half.x * half.y;
    let power = sun_light.luminance * solid_angle * area / f32(CAUSTIC_PHOTONS);

    caustic_emitter_store(8u, center.x);
    caustic_emitter_store(9u, center.y);
    caustic_emitter_store(10u, center.z);
    caustic_emitter_store(11u, half.x);
    caustic_emitter_store(12u, u_axis.x);
    caustic_emitter_store(13u, u_axis.y);
    caustic_emitter_store(14u, u_axis.z);
    caustic_emitter_store(15u, half.y);
    caustic_emitter_store(16u, v_axis.x);
    caustic_emitter_store(17u, v_axis.y);
    caustic_emitter_store(18u, v_axis.z);
    caustic_emitter_store(20u, direction.x);
    caustic_emitter_store(21u, direction.y);
    caustic_emitter_store(22u, direction.z);
    caustic_emitter_store(23u, 50.0);
    caustic_emitter_store(24u, power.x);
    caustic_emitter_store(25u, power.y);
    caustic_emitter_store(26u, power.z);
    caustic_emitter_store(27u, 1.0);
}

/// Age every cell toward zero and free the ones that die (the body lives in
/// `restir_bindings.wgsl` with the grid's constants — naga_oil can't import
/// consts across modules).
@compute @workgroup_size(256, 1, 1)
fn caustic_decay(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= caustic_table_size() {
        return;
    }
    caustic_decay_cell(global_id.x);
}

@compute @workgroup_size(256, 1, 1)
fn caustic_emit(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= CAUSTIC_PHOTONS || caustic_emitter_load(27u) == 0.0 {
        return;
    }
    var rng = i + view.frame_count * 0x68bc21ebu;

    // Jittered point on the emitter rect; the photon starts far up-beam of
    // it so occluders between the sun and the glass apply.
    let rect_center = vec3(caustic_emitter_load(8u), caustic_emitter_load(9u), caustic_emitter_load(10u));
    let u_axis = vec3(caustic_emitter_load(12u), caustic_emitter_load(13u), caustic_emitter_load(14u));
    let v_axis = vec3(caustic_emitter_load(16u), caustic_emitter_load(17u), caustic_emitter_load(18u));
    let r = rand_vec2f(&rng) * 2.0 - 1.0;
    let aim = rect_center
        + (r.x * caustic_emitter_load(11u)) * u_axis
        + (r.y * caustic_emitter_load(15u)) * v_axis;
    var ray_direction = vec3(caustic_emitter_load(20u), caustic_emitter_load(21u), caustic_emitter_load(22u));
    var ray_origin = aim - ray_direction * caustic_emitter_load(23u);

    var throughput = vec3(caustic_emitter_load(24u), caustic_emitter_load(25u), caustic_emitter_load(26u));
    // Hero wavelength: photons collapse at their first dispersive interface
    // exactly like camera paths — the deposited grid carries the rainbow.
    var lambda = 0.0;
    var crossed_glass = false;

    // Nested-dielectric medium stack (the pathtracer's scheme).
    var medium_id: array<u32, MEDIUM_STACK_SIZE>;
    var medium_priority: array<u32, MEDIUM_STACK_SIZE>;
    var medium_ior: array<f32, MEDIUM_STACK_SIZE>;
    var medium_dispersion: array<f32, MEDIUM_STACK_SIZE>;
    var medium_extinction_entry: array<vec3<f32>, MEDIUM_STACK_SIZE>;
    var medium_count = 0u;
    var medium_extinction = vec3(0.0);

    for (var bounce = 0u; bounce < PHOTON_MAX_EVENTS; bounce += 1u) {
        let ray = trace_ray(ray_origin, ray_direction, 0.0, RAY_T_MAX, RAY_FLAG_NONE);
        if ray.kind == RAY_QUERY_INTERSECTION_NONE {
            return;
        }
        let hit = resolve_ray_hit_full(ray);
        let segment_length = length(hit.world_position - ray_origin);
        throughput *= exp(-medium_extinction * segment_length);

        // Volumetric caustics: a POST-glass segment crossing a fog volume
        // deposits its in-scatter mid-air — the dispersed beams glow in the
        // dust as colored shafts. Gated on `crossed_glass`, so the direct
        // (unrefracted) shaft stays the fog march's analytic sun term and is
        // never double-counted. Jittered coarse steps; the grid's temporal
        // average fills the gaps.
        if crossed_glass {
            let span = fog_volumes_range(ray_origin, ray_direction, segment_length);
            if span.y > span.x {
                let step = 0.05;
                var t = span.x + rand_f(&rng) * step;
                for (var k = 0u; k < 128u && t < span.y; k += 1u) {
                    let p = ray_origin + ray_direction * t;
                    let s = fog_volumes_sample(p, 0.0);
                    if s.phase_weight > 1e-7 {
                        caustic_deposit_volume(p, throughput * s.sigma_s * step);
                    }
                    t += step;
                }
            }
        }

        if rand_f(&rng) >= hit.material.specular_transmission {
            // Diffuse (or opaque) surface: deposit if the photon came through
            // glass — that flux is the caustic — and die either way (direct
            // light is NEE's job, multi-bounce is the path tracers').
            if crossed_glass {
                caustic_deposit(hit.world_position, throughput);
            }
            return;
        }

        // Transmissive surface: cross the interface, exactly like the
        // pathtracer's branch (entering tested against the true triangle
        // normal; see there for the reasoning).
        let entering = dot(ray_direction, hit.geometric_world_normal) < 0.0;
        let oriented_geometric_normal = select(
            -hit.geometric_world_normal, hit.geometric_world_normal, entering);
        let wo = -ray_direction;
        let oriented_normal = select(-hit.world_normal, hit.world_normal, entering);

        var self_index = MEDIUM_NOT_FOUND;
        var other_priority = 0u;
        var other_ior = 1.0;
        var other_dispersion = 0.0;
        for (var m = 0u; m < medium_count; m += 1u) {
            if medium_id[m] == hit.material_id {
                self_index = m;
                continue;
            }
            if medium_priority[m] >= other_priority {
                other_priority = medium_priority[m];
                other_ior = medium_ior[m];
                other_dispersion = medium_dispersion[m];
            }
        }
        let true_interface = hit.material.nested_priority >= other_priority;

        var crossed = true;
        if true_interface {
            var self_ior = hit.material.ior;
            var far_ior = other_ior;
            if hit.material.dispersion > 0.0 || other_dispersion > 0.0 {
                if lambda == 0.0 {
                    lambda = sample_hero_wavelength(&rng);
                    throughput *= spectral_rgb_weight(lambda);
                }
                self_ior = dispersive_ior(self_ior, hit.material.dispersion, lambda);
                far_ior = dispersive_ior(far_ior, other_dispersion, lambda);
            }
            let eta = select(
                self_ior / far_ior,
                far_ior / self_ior,
                entering,
            );
            let glass = sample_glass_bsdf(wo, oriented_normal, eta, &rng);
            ray_direction = glass.wi;
            crossed = glass.refracted;
            crossed_glass = true;
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
            for (var m = 0u; m < medium_count; m += 1u) {
                if !found || medium_priority[m] >= active_priority {
                    active_priority = medium_priority[m];
                    medium_extinction = medium_extinction_entry[m];
                    found = true;
                }
            }
        }

        let offset_normal = select(oriented_geometric_normal, -oriented_geometric_normal, crossed);
        ray_origin = offset_ray_origin(hit.world_position, offset_normal);
    }
}
