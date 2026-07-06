// ReSTIR DI spatial reuse + shade (rung 3, session 2). Runs AFTER the trace:
// every pixel's post-temporal reservoir exists, so neighbors can merge — the
// barrier the in-chit path can't provide. Merges K disk-sampled neighbors
// (depth/normal-validated), traces ONE ray-query visibility ray for the winner,
// and adds the DI into the accumulated output with the SAME blend weight the
// raygen used this frame (`output += blend_w · DI · exposure`) — accumulation
// composes without a history buffer. Shade-only: reservoirs are NOT written
// back, so the bias study isolates the spatial combiner (no feedback loop).
//
// Two combiners (`params.unbiased`):
//   0 = naive Algorithm-4 M-sum — BIASED: neighbors whose domain can't produce
//       the winner still inflate M → the classic darkening (the exam wants to
//       SEE this in the freeze-diff before fixing it).
//   1 = Z-count — re-evaluate the winner's p̂ at each contributor's surface and
//       count only the M that could have produced it (Bitterli Alg. 6).
enable wgpu_ray_query;

#import bevy_solari::sampling::{Reservoir, SurfaceGbuf, LightSample, NULL_LIGHT_ID, resolve_emissive_for_restir, calculate_resolved_light_contribution, trace_light_visibility, power_heuristic, pick_luminance}
#import bevy_solari::brdf::{evaluate_brdf, brdf_pdf, F_AB}
#import bevy_solari::scene_bindings::{offset_ray_origin, ResolvedMaterial}
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
    exposure: f32,
    unbiased: u32,    // 0 = naive M-sum, 1 = Z-count
    pad_a: u32,
    pad_b: u32,
    pad_c: u32,
}

@group(1) @binding(0) var<storage, read_write> reservoirs: array<Reservoir>;
@group(1) @binding(1) var<storage, read> surfaces: array<SurfaceGbuf>;
@group(1) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
@group(1) @binding(3) var<uniform> params: SpatialParams;

const MAX_TAPS: u32 = 8u;

// Unpacked shading state for one pixel's surface.
struct Surf {
    pos: vec3<f32>,
    view_z: f32,
    ns: vec3<f32>,
    ng: vec3<f32>,
    mat: ResolvedMaterial,
    wo: vec3<f32>,
    f_ab: vec2<f32>,
}

fn load_surf(px: u32) -> Surf {
    let s = surfaces[px];
    var out: Surf;
    out.pos = vec3<f32>(s.pos_x, s.pos_y, s.pos_z);
    out.view_z = s.view_z;
    out.ns = octahedral_decode_signed(unpack2x16snorm(s.normal_oct));
    out.ng = octahedral_decode_signed(unpack2x16snorm(s.geo_normal_oct));
    let c_rg = unpack2x16float(s.color_rg);
    let c_bm = unpack2x16float(s.color_b_metallic);
    let r_pr = unpack2x16float(s.rough_prough);
    let refl = unpack2x16float(s.reflectance);
    var m: ResolvedMaterial;
    m.base_color = vec3<f32>(c_rg, c_bm.x);
    m.emissive = vec3<f32>(0.0);
    m.reflectance = refl.x;
    m.roughness = r_pr.x;
    m.perceptual_roughness = r_pr.y;
    m.metallic = c_bm.y;
    m.specular_transmission = 0.0;
    m.ior = 1.5;
    m.dispersion = 0.0;
    m.extinction = vec3<f32>(0.0);
    m.nested_priority = 0u;
    out.mat = m;
    // Camera-relative trace space: the eye is the origin.
    out.wo = normalize(-out.pos);
    out.f_ab = F_AB(m.perceptual_roughness, max(dot(out.ns, out.wo), 1.0e-4));
    return out;
}

// The chit's target function, re-evaluated at `surf`: p̂ = luminance(w_mis·L·G·BRDF)
// with the NEE-vs-BSDF MIS weight folded in — MUST stay one function everywhere.
struct TargetEval {
    f: vec3<f32>,
    phat: f32,
    light_pos: vec4<f32>,
}

fn eval_target(surf: Surf, ls: LightSample) -> TargetEval {
    let resolved = resolve_emissive_for_restir(ls);
    let lc = calculate_resolved_light_contribution(resolved, surf.pos, surf.ns);
    if lc.inverse_pdf <= 0.0 {
        return TargetEval(vec3<f32>(0.0), 0.0, resolved.world_position);
    }
    var w_mis = 1.0;
    if lc.brdf_rays_can_hit {
        w_mis = power_heuristic(lc.pdf_solid, brdf_pdf(surf.wo, lc.wi, surf.ns, surf.mat, surf.f_ab));
    }
    let f = w_mis * lc.radiance * evaluate_brdf(surf.wo, lc.wi, surf.ns, surf.mat, surf.f_ab);
    return TargetEval(f, pick_luminance(f), resolved.world_position);
}

@compute @workgroup_size(8, 8, 1)
fn spatial(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let px = gid.y * params.width + gid.x;
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
    var sel = LightSample(NULL_LIGHT_ID, 0u);
    var sel_f = vec3<f32>(0.0);
    var sel_pos = vec4<f32>(0.0);
    var sel_phat = 0.0;
    var m_total = 0.0;
    if own.light_id != NULL_LIGHT_ID && own.w > 0.0 {
        let e = eval_target(surf, LightSample(own.light_id, own.seed));
        let w = e.phat * own.w * own.m;
        if w > 0.0 {
            w_sum = w;
            sel = LightSample(own.light_id, own.seed);
            sel_f = e.f;
            sel_pos = e.light_pos;
            sel_phat = e.phat;
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
            let e = eval_target(surf, LightSample(nres.light_id, nres.seed));
            let w = e.phat * nres.w * nres.m;
            w_sum += w;
            if w > 0.0 && rand_f(&rng) * w_sum < w {
                sel = LightSample(nres.light_id, nres.seed);
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

    // Denominator: naive M-sum (biased dark where a contributor's domain can't
    // produce the winner) or the Z-count of M that actually could.
    var m_denom = m_total;
    if params.unbiased == 1u {
        var z = 0.0;
        for (var s = 0u; s < src_n; s += 1u) {
            if src_px[s] == px {
                z += src_m[s]; // own p̂(winner) > 0 by construction
                continue;
            }
            let e = eval_target(load_surf(src_px[s]), sel);
            if e.phat > 0.0 {
                z += src_m[s];
            }
        }
        m_denom = max(z, src_m[0]);
    }

    let big_w = w_sum / max(m_denom * sel_phat, 1.0e-12);
    let origin = offset_ray_origin(surf.pos, surf.ng);
    let visible = trace_light_visibility(origin, sel_pos);
    // Debug: BLUE = winner occluded (all-blue floor = ray query broken);
    // GREEN = visible, DI would land. Numbers via the probe readback.
    if debug {
        output[px] = vec4<f32>(0.0, visible, 1.0 - visible, output[px].a);
        return;
    }
    if visible <= 0.0 {
        return;
    }
    let di = sel_f * big_w * visible * params.exposure * params.blend_w;
    // Alpha carries the gizmo depth — leave it untouched.
    output[px] = vec4<f32>(output[px].rgb + di, output[px].a);
}
