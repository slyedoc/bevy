// ReSTIR DI spatial reuse + shade (rung 3, session 2). Runs AFTER the trace:
// every pixel's post-temporal reservoir exists, so neighbors can merge — the
// barrier the in-chit path can't provide. Merges K disk-sampled neighbors
// (depth/normal-validated), traces ONE ray-query visibility ray for the winner,
// and adds the DI into the accumulated output with the SAME blend weight the
// raygen used this frame (`output += blend_w · DI`, physical radiance) —
// accumulation composes without a history buffer. Shade-only: reservoirs are NOT written
// back, so the bias study isolates the spatial combiner (no feedback loop).
//
// Two combiners (`params.unbiased`):
//   0 = naive Algorithm-4 M-sum — BIASED: neighbors whose domain can't produce
//       the winner still inflate M → the classic darkening (the exam wants to
//       SEE this in the freeze-diff before fixing it).
//   1 = Z-count — re-evaluate the winner's p̂ at each contributor's surface and
//       count only the M that could have produced it (Bitterli Alg. 6).
enable wgpu_ray_query;

#import bevy_solari::sampling::{Reservoir, SurfaceGbuf, GiSample, Surf, unpack_surface, StoredLight, ResolvedLightSample, unpack_stored_light, NULL_LIGHT_ID, calculate_resolved_light_contribution, power_heuristic, pick_luminance}
#import bevy_solari::brdf::{evaluate_brdf, brdf_pdf, F_AB, gi_shade, gi_phat}
#import bevy_solari::scene_bindings::{offset_ray_origin, ResolvedMaterial, tlas, RAY_T_MIN, RAY_T_MAX, RAY_NO_CULL}
#import bevy_solari::pbr::rand_f
#import bevy_render::utils::octahedral_decode_signed

struct SpatialParams {
    width: u32,
    height: u32,
    parity: u32,      // this frame's CURRENT reservoir slot (frame & 1)
    frame: u32,       // RNG decorrelation
    taps: u32,        // neighbor count (≤ MAX_TAPS)
    radius: f32,      // disk radius, pixels
    blend_w: f32,     // raygen's accumulation weight this frame (1 = no accum)
    firefly_clamp: f32, // max GI luminance per frame (physical; 0 = off)
    unbiased: u32,    // 0 = naive M-sum, 1 = Z-count
    pad_a: u32,       // debug paint
    di_on: u32,
    gi_on: u32,
}

@group(1) @binding(0) var<storage, read_write> reservoirs: array<Reservoir>;
@group(1) @binding(1) var<storage, read> surfaces: array<SurfaceGbuf>;
@group(1) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
@group(1) @binding(3) var<uniform> params: SpatialParams;
// Chit-resolved winner sample per reservoir slot — the pass reshades from this
// instead of `resolve_emissive_for_restir` (bindless loads a wgpu pass can't do).
// Binding 5: scene_bindings hard-codes `geometry_addresses` at group(1) binding(4)
// (pulled in transitively via brdf), so this slot stays clear of it.
@group(1) @binding(5) var<storage, read> light_samples: array<StoredLight>;
// Raygen-written GI reservoirs (2 slots/pixel by parity, see `GiSample`).
@group(1) @binding(6) var<storage, read> gi_samples: array<GiSample>;

const MAX_TAPS: u32 = 8u;

// Unpacked shading state for one pixel's surface.
fn load_surf(px: u32) -> Surf {
    var out = unpack_surface(surfaces[px]);
    out.f_ab = F_AB(out.mat.perceptual_roughness, max(dot(out.ns, out.wo), 1.0e-4));
    return out;
}

// The chit's target function, re-evaluated at `surf`: p̂ = luminance(w_mis·L·G·BRDF)
// with the NEE-vs-BSDF MIS weight folded in — MUST stay one function everywhere.
struct TargetEval {
    f: vec3<f32>,
    phat: f32,
    light_pos: vec4<f32>,
}

fn eval_target(surf: Surf, resolved: ResolvedLightSample) -> TargetEval {
    let lc = calculate_resolved_light_contribution(resolved, surf.pos, surf.ns);
    if lc.inverse_pdf <= 0.0 {
        return TargetEval(vec3<f32>(0.0), 0.0, resolved.world_position);
    }
    var w_mis = 1.0;
    if lc.brdf_rays_can_hit {
        w_mis = power_heuristic(lc.pdf_solid, brdf_pdf(surf.wo, lc.wi, surf.ns, surf.mat, surf.f_ab));
    }
    let f = w_mis * lc.radiance * saturate(dot(surf.ns, lc.wi)) * evaluate_brdf(surf.wo, lc.wi, surf.ns, surf.mat, surf.f_ab);
    return TargetEval(f, pick_luminance(f), resolved.world_position);
}

// Self-contained opaque visibility (no `physical_load`): the shared
// `trace_light_visibility` alpha-tests cutouts, which needs bindless material
// loads a wgpu compute pass can't do. Alpha-masked candidates are confirmed as
// opaque (conservative — a cutout occludes rather than leaks light), matching
// `ray_query.wgsl`. `light_pos.w == 1` = area point; `w == 0` = directional dir.
fn spatial_visibility(ray_origin: vec3<f32>, light_pos: vec4<f32>) -> f32 {
    var dir = light_pos.xyz;
    var t_max = RAY_T_MAX;
    if light_pos.w == 1.0 {
        let to = dir - ray_origin;
        let dist = length(to);
        dir = to / dist;
        t_max = dist - RAY_T_MIN;
    }
    if t_max < RAY_T_MIN {
        return 0.0;
    }
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, RAY_NO_CULL, RAY_T_MIN, t_max, ray_origin, dir));
    while rayQueryProceed(&rq) {
        let c = rayQueryGetCandidateIntersection(&rq);
        if c.kind == RAY_QUERY_INTERSECTION_TRIANGLE {
            rayQueryConfirmIntersection(&rq);
        }
    }
    return f32(rayQueryGetCommittedIntersection(&rq).kind == RAY_QUERY_INTERSECTION_NONE);
}

@compute @workgroup_size(8, 8, 1)
fn spatial(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let px = gid.y * params.width + gid.x;
    if params.di_on == 1u {
        di_spatial(px, gid.xy);
    }
    if params.gi_on == 1u {
        gi_spatial(px, gid.xy);
    }
}

fn di_spatial(px: u32, gid: vec2<u32>) {
    let own = reservoirs[px * 2u + params.parity];
    let debug = params.pad_a == 1u;
    // m == 0: sky / mirror / restir-off pixel (raygen cleared the slot).
    if own.m <= 0.0 {
        // Debug: RED = dead/unseen reservoir (raw-VK write not visible? sky?).
        if debug {
            output[px] = vec4<f32>(1.0, 0.0, 0.0, output[px].a);
        }
        return;
    }
    let surf = load_surf(px);
    var rng = (px + params.frame * 5782582u) * 0x9e3779b1u + 0x68bc21ebu;

    // Contributor list: self + accepted neighbors (for the Z-count pass).
    var src_px: array<u32, 9>;
    var src_m: array<f32, 9>;
    var src_n = 0u;

    // Stream self first.
    var w_sum = 0.0;
    var sel_ls = ResolvedLightSample(vec4<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0);
    var sel_f = vec3<f32>(0.0);
    var sel_pos = vec4<f32>(0.0);
    var sel_phat = 0.0;
    var m_total = 0.0;
    if own.light_id != NULL_LIGHT_ID && own.w > 0.0 {
        let own_stored = light_samples[px * 2u + params.parity];
        let own_ls = unpack_stored_light(own_stored);
        // Own pixel: the chit's EXACT f/p̂ (same surface it was computed on) — a
        // G-buffer recompute here drifts ~6% dark on dim pixels via w_mis.
        // Own pixel: the chit's EXACT f/p̂ (computed on this same surface) — a
        // G-buffer recompute here drifts ~6% dark on dim pixels via w_mis.
        let phat = own_stored.phat;
        let w = phat * own.w * own.m;
        if w > 0.0 {
            w_sum = w;
            sel_ls = own_ls;
            sel_f = vec3<f32>(own_stored.fr, own_stored.fg, own_stored.fb);
            sel_pos = own_ls.world_position;
            sel_phat = phat;
        }
    }
    m_total = own.m;
    src_px[0] = px;
    src_m[0] = own.m;
    src_n = 1u;

    // Neighbors: uniform disk, geometry-validated (depth 10%, normal 25°).
    let taps = min(params.taps, MAX_TAPS);
    for (var t = 0u; t < taps; t += 1u) {
        let ang = rand_f(&rng) * 6.2831853;
        let rad = sqrt(rand_f(&rng)) * params.radius;
        let nx = i32(gid.x) + i32(round(cos(ang) * rad));
        let ny = i32(gid.y) + i32(round(sin(ang) * rad));
        if nx < 0 || ny < 0 || nx >= i32(params.width) || ny >= i32(params.height) {
            continue;
        }
        let npx = u32(ny) * params.width + u32(nx);
        if npx == px {
            continue;
        }
        let nres = reservoirs[npx * 2u + params.parity];
        if nres.m <= 0.0 || nres.light_id == NULL_LIGHT_ID {
            continue;
        }
        let nsurf_raw = surfaces[npx];
        let n_ns = octahedral_decode_signed(unpack2x16snorm(nsurf_raw.normal_oct));
        let depth_ok = abs(nsurf_raw.view_z - surf.view_z) <= 0.1 * max(nsurf_raw.view_z, surf.view_z);
        if !depth_ok || dot(n_ns, surf.ns) < 0.9 {
            continue;
        }
        // Re-target the neighbor's sample at OUR surface and stream it in.
        if nres.w > 0.0 {
            let nls = unpack_stored_light(light_samples[npx * 2u + params.parity]);
            let e = eval_target(surf, nls);
            let w = e.phat * nres.w * nres.m;
            w_sum += w;
            if w > 0.0 && rand_f(&rng) * w_sum < w {
                sel_ls = nls;
                sel_f = e.f;
                sel_pos = e.light_pos;
                sel_phat = e.phat;
            }
        }
        m_total += nres.m;
        src_px[src_n] = npx;
        src_m[src_n] = nres.m;
        src_n += 1u;
    }

    if sel_phat <= 0.0 || w_sum <= 0.0 {
        // Debug: YELLOW = reservoir seen but every target re-eval came out zero
        // (garbage surfaces / broken resolve in compute).
        if debug {
            output[px] = vec4<f32>(1.0, 1.0, 0.0, output[px].a);
        }
        return;
    }

    // Winner visibility from the target — also the own contributor's visibility.
    let origin = offset_ray_origin(surf.pos, surf.ng);
    let visible = spatial_visibility(origin, sel_pos);

    // Denominator: naive M-sum (biased dark where a contributor's domain can't
    // produce the winner) or the visibility-aware 1/Z — count a contributor's M
    // only if the winner is in its unshadowed domain AND actually VISIBLE from it
    // (the shadow-boundary fix; one shadow ray per contributor).
    var m_denom = m_total;
    if params.unbiased == 1u {
        var z = 0.0;
        for (var s = 0u; s < src_n; s += 1u) {
            var can_produce = visible > 0.0; // own: reuse the target's trace
            if src_px[s] != px {
                let ssurf = load_surf(src_px[s]);
                can_produce = eval_target(ssurf, sel_ls).phat > 0.0
                    && spatial_visibility(offset_ray_origin(ssurf.pos, ssurf.ng), sel_pos) > 0.0;
            }
            if can_produce {
                z += src_m[s];
            }
        }
        m_denom = max(z, 1.0e-4);
    }

    let big_w = w_sum / max(m_denom * sel_phat, 1.0e-12);
    // Debug: BLUE = winner occluded (all-blue floor = ray query broken);
    // GREEN = visible, DI would land. Numbers via the probe readback.
    if debug {
        output[px] = vec4<f32>(0.0, visible, 1.0 - visible, output[px].a);
        return;
    }
    if visible <= 0.0 {
        return;
    }
    let di = sel_f * big_w * visible * params.blend_w;
    // Alpha carries the gizmo depth — leave it untouched.
    output[px] = vec4<f32>(output[px].rgb + di, output[px].a);
}

// Spatial GI reuse: merge neighbors' reservoirs at this surface. Streams are
// accepted only if geometry-compatible, Jacobian-sane, AND their sample is
// visible from here (one ray per candidate — an occluded stream must not count,
// or its dead M darkens shadow boundaries). `unbiased` = balance-heuristic MIS
// over the surviving generating streams (per-pixel BSDF pdfs, so 1/M is not a
// balance heuristic for GI); 0 = naive M-sum over the survivors.
fn gi_spatial(px: u32, gid: vec2<u32>) {
    let own = gi_samples[px * 2u + params.parity];
    let debug = params.pad_a == 1u;
    // Debug paint at EVERY exit: R = pass shade lum, G = raygen's stashed
    // would-be shade lum, B = exit code (1 dead own, 2 no streams, 3 zero
    // target, 4 shaded). Deficit lives where G > R.
    let dbg_ref = bitcast<f32>(own.pad_a);
    if own.m <= 0.0 || own.surf_view_z <= 0.0 {
        if debug {
            output[px] = vec4<f32>(0.0, dbg_ref, 1.0, output[px].a);
        }
        return;
    }
    let surf = load_surf(px);
    let my_origin = offset_ray_origin(surf.pos, surf.ng);
    var rng = (px + params.frame * 5782582u) * 0x9e3779b1u + 0x1b873593u;

    // Survivor streams: candidate + its W in MY measure + its M.
    var s_px: array<u32, 9>;
    var s_m: array<f32, 9>;
    var s_w: array<f32, 9>;
    var s_n = 0u;
    if own.w > 0.0 {
        s_px[0] = px;
        s_m[0] = own.m;
        s_w[0] = own.w;
        s_n = 1u;
    }
    let taps = min(params.taps, MAX_TAPS);
    for (var t = 0u; t < taps; t += 1u) {
        let ang = rand_f(&rng) * 6.2831853;
        let rad = sqrt(rand_f(&rng)) * params.radius;
        let nx = i32(gid.x) + i32(round(cos(ang) * rad));
        let ny = i32(gid.y) + i32(round(sin(ang) * rad));
        if nx < 0 || ny < 0 || nx >= i32(params.width) || ny >= i32(params.height) {
            continue;
        }
        let npx = u32(ny) * params.width + u32(nx);
        if npx == px {
            continue;
        }
        let ns = gi_samples[npx * 2u + params.parity];
        if ns.m <= 0.0 || ns.w <= 0.0 || ns.surf_view_z <= 0.0 {
            continue;
        }
        let nn = octahedral_decode_signed(unpack2x16snorm(ns.surf_normal_oct));
        let depth_ok = abs(ns.surf_view_z - surf.view_z) <= 0.1 * max(ns.surf_view_z, surf.view_z);
        if !depth_ok || dot(nn, surf.ns) < 0.9 {
            continue;
        }
        let xj = vec3<f32>(ns.pos_x, ns.pos_y, ns.pos_z);
        let n_s = octahedral_decode_signed(unpack2x16snorm(ns.normal_oct));
        let jsurf_raw = surfaces[npx];
        let to_me = surf.pos - xj;
        let to_j = vec3<f32>(jsurf_raw.pos_x, jsurf_raw.pos_y, jsurf_raw.pos_z) - xj;
        let d2_me = dot(to_me, to_me);
        let d2_j = dot(to_j, to_j);
        if d2_me < 1.0e-8 || d2_j < 1.0e-8 {
            continue;
        }
        let cos_me = abs(dot(n_s, to_me)) * inverseSqrt(d2_me);
        let cos_j = abs(dot(n_s, to_j)) * inverseSqrt(d2_j);
        let jac = (cos_me / max(cos_j, 1.0e-4)) * (d2_j / max(d2_me, 1.0e-8));
        if jac < 0.1 || jac > 10.0 {
            continue;
        }
        if spatial_visibility(my_origin, vec4<f32>(xj, 1.0)) <= 0.0 {
            continue;
        }
        s_px[s_n] = npx;
        s_m[s_n] = ns.m;
        s_w[s_n] = ns.w * jac;
        s_n += 1u;
    }
    if s_n == 0u {
        if debug {
            output[px] = vec4<f32>(0.0, dbg_ref, 2.0, output[px].a);
        }
        return;
    }

    var w_sum = 0.0;
    var sel = own;
    var sel_phat = 0.0;
    var m_total = 0.0;
    for (var j = 0u; j < s_n; j += 1u) {
        m_total += s_m[j];
        let cand = gi_samples[s_px[j] * 2u + params.parity];
        let ph_me = gi_phat(surf, cand);
        if ph_me <= 0.0 {
            continue;
        }
        var w = 0.0;
        if params.unbiased == 1u {
            // Balance-heuristic MIS: p̂ proxies at each stream's surface, in a
            // common area measure at x_s (× cosθ/d²).
            let xj = vec3<f32>(cand.pos_x, cand.pos_y, cand.pos_z);
            let n_s = octahedral_decode_signed(unpack2x16snorm(cand.normal_oct));
            var num = 0.0;
            var denom = 0.0;
            for (var k = 0u; k < s_n; k += 1u) {
                let ksurf = load_surf(s_px[k]);
                let to_k = ksurf.pos - xj;
                let d2_k = dot(to_k, to_k);
                if d2_k < 1.0e-8 {
                    continue;
                }
                let conv = abs(dot(n_s, to_k)) * inverseSqrt(d2_k) / d2_k;
                let pk = gi_phat(ksurf, cand) * conv * s_m[k];
                denom += pk;
                if k == j {
                    num = pk;
                }
            }
            if num <= 0.0 || denom <= 0.0 {
                continue;
            }
            w = (num / denom) * ph_me * s_w[j];
        } else {
            w = ph_me * s_w[j] * s_m[j];
        }
        w_sum += w;
        if w > 0.0 && rand_f(&rng) * w_sum < w {
            sel = cand;
            sel_phat = ph_me;
        }
    }

    if sel_phat <= 0.0 || w_sum <= 0.0 {
        if debug {
            output[px] = vec4<f32>(0.0, dbg_ref, 3.0, output[px].a);
        }
        return;
    }
    // Survivors are pre-validated visible; MIS weights partition unity so the
    // naive 1/M division drops out in unbiased mode.
    var denom_final = m_total * sel_phat;
    if params.unbiased == 1u {
        denom_final = sel_phat;
    }
    let big_w = w_sum / max(denom_final, 1.0e-12);
    if debug {
        let mine = pick_luminance(gi_shade(surf, sel) * big_w);
        // B carries the own reservoir's m — the chain-maturity readout.
        output[px] = vec4<f32>(mine, dbg_ref, own.m, 4.0);
        return;
    }
    var gi = gi_shade(surf, sel) * big_w;
    // Realtime firefly filter (reference passes 0 = off): scale reservoir
    // spikes down luminance-preserving before they reach the frame.
    if params.firefly_clamp > 0.0 {
        let lum = pick_luminance(gi);
        if lum > params.firefly_clamp {
            gi *= params.firefly_clamp / lum;
        }
    }
    output[px] = vec4<f32>(output[px].rgb + gi * params.blend_w, output[px].a);
}
