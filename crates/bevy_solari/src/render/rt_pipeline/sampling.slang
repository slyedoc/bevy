// Light sampling + ReSTIR shared types for the RT-pipeline Slang stages — the
// Slang mirror of `bindings/sampling.wgsl` + `bindings/light_sampling.wgsl`
// (one module here: Slang links per-function, so the WGSL's physical_load
// module split doesn't apply). Numerics are kept bit-faithful to the WGSL.
//
// Layout contract: `Reservoir` / `SurfaceGbuf` / `StoredLight` are SHARED with
// the wgpu `restir_spatial` pass — all-scalar structs, sizes 16-byte multiples
// (32/48/64 B), so every layout rule agrees. Offset-diff against a naga dump
// when touching them (slang_plan.md, Law 2).
//
// Not compiled standalone: `import sampling;` (slangc resolves same-dir imports).
module sampling;

import scene_resolve;
import brdf;

public static const uint NULL_LIGHT_ID = 0xFFFFFFFF;

public struct LightSample
{
    public uint light_id;
    public uint seed;
}

// One ReSTIR DI reservoir (32 B). Two slots per pixel, INTERLEAVED by frame
// parity (`pixel*2 + (frame&1)` = current). `light_id`/`seed` reconstruct the
// light sample via `resolve_light_sample` (stable light slots); `w` is the
// unbiased contribution weight W; `normal_oct`/`depth` validate temporal
// reprojection.
public struct Reservoir
{
    public uint light_id;
    public uint seed;
    public float m;
    public float w;
    public uint normal_oct;
    public float depth;
    public uint pad_a;
    public uint pad_b;
}

// Primary-hit surface attributes for the ReSTIR spatial pass (48 B/pixel): the
// exact shading inputs `evaluate_brdf`/`brdf_pdf`/`F_AB` need, f16-packed.
public struct SurfaceGbuf
{
    public float pos_x;
    public float pos_y;
    public float pos_z;
    public float view_z;
    public uint normal_oct;     // SHADING normal (bent), snorm-oct
    public uint geo_normal_oct; // geometric normal (shadow-ray origin offset)
    public uint color_rg;       // pack2x16float(base_color.rg)
    public uint color_b_metallic;
    public uint rough_prough;   // pack2x16float(roughness, perceptual_roughness)
    public uint reflectance;    // pack2x16float(reflectance, 0)
    public uint wo_oct;         // view dir — world_position is ABSOLUTE, so the
                                // pass can't reconstruct wo from position alone
    public uint f_ab_packed;    // pack2x16float(F_AB) — consumers never
                                // re-sample the DFG LUT
}

public struct ResolvedLightSample
{
    public float4 world_position;
    public float3 world_normal;
    public float3 radiance;
    public float inverse_pdf;
}

// Canonical ReSTIR GI sample: the first-bounce reconnection vertex + the
// suffix radiance through it. Size must stay a 16-byte multiple (64 B).
public struct GiSample
{
    public float pos_x;      // x_s (bounce-1 offset ray origin)
    public float pos_y;
    public float pos_z;
    public uint normal_oct;  // n_s, snorm-oct
    public float l_r;        // reusable suffix radiance (bounce-1 emission excluded)
    public float l_g;
    public float l_b;
    public float w;          // unbiased contribution weight (1/pdf when canonical)
    public float a0_r;       // exact primary BSDF weight f·cos/pdf (incl. its RR share)
    public float a0_g;
    public float a0_b;
    public float m;          // sample count; 0 = dead sample
    public uint surf_normal_oct; // the GENERATING surface, for temporal validation
    public float surf_view_z;
    public uint pad_a;
    public uint pad_b;
}

// Per-reservoir-slot winner sample (64 B), written by the chit (which has the
// bindless resolve) so the wgpu spatial pass reshades from stored data alone.
public struct StoredLight
{
    public float px; public float py; public float pz; public float pw;
    public float nx; public float ny; public float nz;
    public float rr; public float rg; public float rb;
    public float inv_pdf;
    public float fr; public float fg; public float fb; // chit's exact winner f
    public float phat;                                 // chit's exact p̂ = luminance(f)
    public float pad; // pad to 64 B — a struct shared with the wgpu pass MUST
                      // be 16-byte-aligned in size
}

public StoredLight pack_stored_light(ResolvedLightSample r, float3 f, float phat)
{
    StoredLight s;
    s.px = r.world_position.x; s.py = r.world_position.y;
    s.pz = r.world_position.z; s.pw = r.world_position.w;
    s.nx = r.world_normal.x; s.ny = r.world_normal.y; s.nz = r.world_normal.z;
    s.rr = r.radiance.x; s.rg = r.radiance.y; s.rb = r.radiance.z;
    s.inv_pdf = r.inverse_pdf;
    s.fr = f.x; s.fg = f.y; s.fb = f.z;
    s.phat = phat;
    s.pad = 0.0;
    return s;
}

public struct LightContribution
{
    public float3 radiance;
    public float inverse_pdf;
    public float3 wi;
    public bool brdf_rays_can_hit;
    /// The sample's pdf in SOLID-ANGLE measure (area pdf × d²/cosθ_light) —
    /// the value MIS weights compare against a BRDF pdf. `inverse_pdf` stays
    /// area-measure (the estimator's `cosθ/d²` is folded into `radiance`).
    public float pdf_solid;
}

public struct GenerateRandomLightSampleResult
{
    public LightSample light_sample;
    public ResolvedLightSample resolved_light_sample;
}

// ── MIS ─────────────────────────────────────────────────────────────────────

public float balance_heuristic(float f, float g)
{
    // Need to guard against NaNs since ReSTIR reservoirs can have UCW=0
    if (f == 0.0)
    {
        return 0.0;
    }
    return max(0.0, 1.0 / (1.0 + (g / f)));
}

public float power_heuristic(float f, float g)
{
    return balance_heuristic(f * f, g * g);
}

// ── Light-list helpers ──────────────────────────────────────────────────────

/// Number of active emissive-mesh lights (the `active_light_list` header).
public uint emissive_light_count()
{
    return active_light_list[0];
}

/// Number of active directional lights (the `active_light_list` header).
public uint directional_light_count()
{
    return active_light_list[1];
}

// Rec. 709 luminance — the CPU flux basis (`prepare_light_sources`) uses the
// same coefficients; the two MUST stay identical or the pick pdf de-mirrors.
public float pick_luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

public struct WeightedPick
{
    public uint index;
    public float prob;
}

/// Power-weighted emissive pick: binary-search the normalized flux CDF
/// appended (as f32 bits) after the slot lists in `active_light_list`,
/// followed by the total flux (0 = uniform-pick sentinel).
public WeightedPick pick_emissive_weighted(
    uint emissive_count, uint directional_count, inout uint rng)
{
    let cdf_base = 2 + emissive_count + directional_count;
    let total_flux = asfloat(active_light_list[cdf_base + emissive_count]);
    WeightedPick pick;
    if (total_flux <= 0.0)
    {
        pick.index = rand_range_u(emissive_count, rng);
        pick.prob = 1.0 / float(emissive_count);
        return pick;
    }
    let u = rand_f(rng);
    var lo = 0u;
    var hi = emissive_count - 1;
    while (lo < hi)
    {
        let mid = (lo + hi) >> 1;
        if (u < asfloat(active_light_list[cdf_base + mid]))
        {
            hi = mid;
        }
        else
        {
            lo = mid + 1;
        }
    }
    let c1 = asfloat(active_light_list[cdf_base + lo]);
    var c0 = 0.0;
    if (lo > 0)
    {
        c0 = asfloat(active_light_list[cdf_base + lo - 1]);
    }
    pick.index = lo;
    pick.prob = max(c1 - c0, 1e-9);
    return pick;
}

public LightContribution calculate_resolved_light_contribution(
    ResolvedLightSample resolved_light_sample, float3 ray_origin, float3 origin_world_normal)
{
    let ray = resolved_light_sample.world_position.xyz
        - (resolved_light_sample.world_position.w * ray_origin);
    // Clamp the inverse-square at contact range: a sample point numerically on
    // the receiver explodes `1/d²` to inf, and `inf × 0` from the visibility
    // trace is NaN — one NaN permanently poisons a progressive accumulator.
    let light_distance = max(length(ray), RAY_T_MIN);
    let wi = ray / light_distance;

    let cos_theta_light = saturate(dot(-wi, resolved_light_sample.world_normal));
    let light_distance_squared = light_distance * light_distance;

    let radiance = resolved_light_sample.radiance * (cos_theta_light / light_distance_squared);

    // Solid-angle pdf for MIS: area pdf × d²/cosθ. For a directional light
    // (w == 0) d = 1 and cosθ = 1, so this is already its cone pdf.
    let pdf_area = resolved_light_sample.inverse_pdf > 0.0
        ? 1.0 / resolved_light_sample.inverse_pdf
        : 0.0;
    let pdf_solid = pdf_area * light_distance_squared / max(cos_theta_light, 1e-4);

    LightContribution lc;
    lc.radiance = radiance;
    lc.inverse_pdf = resolved_light_sample.inverse_pdf;
    lc.wi = wi;
    lc.brdf_rays_can_hit = resolved_light_sample.world_position.w == 1.0;
    lc.pdf_solid = pdf_solid;
    return lc;
}

// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec22%3A297
public float3 triangle_barycentrics(uint seed)
{
    var rng = seed;
    var barycentrics = rand_vec2f(rng);
    if (barycentrics.x + barycentrics.y > 1.0)
    {
        barycentrics = 1.0 - barycentrics;
    }
    return float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
}

// ── Light sample resolve + generation (bindless path) ───────────────────────

public ResolvedLightSample resolve_light_sample(LightSample light_sample, LightSource light_source)
{
    if (light_source.kind == LIGHT_SOURCE_KIND_DIRECTIONAL)
    {
        let directional_light = directional_lights[light_source.id];

        // Sample a random direction within a cone whose base is the sun
        // approximated as a disk.
        // https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec30%3A305
        var rng = light_sample.seed;
        let random = rand_vec2f(rng);
        let cos_theta = (1.0 - random.x) + random.x * directional_light.cos_theta_max;
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        let phi = random.y * PI_2;
        var direction_to_light = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

        // Rotate the ray so that the cone it was sampled from is aligned with
        // the light direction.
        direction_to_light = basis_mul(
            orthonormalize(directional_light.direction_to_light), direction_to_light);

        ResolvedLightSample resolved;
        resolved.world_position = float4(direction_to_light, 0.0);
        resolved.world_normal = -direction_to_light;
        resolved.radiance = directional_light.luminance;
        resolved.inverse_pdf = directional_light.inverse_pdf;
        return resolved;
    }

    let triangle_count = light_source.kind >> 1;
    let instance_id = light_source.id;
    let instance_triangle_id = light_sample.light_id & 0xFFFF;
    let barycentrics = triangle_barycentrics(light_sample.seed);

    // Walk this instance's clusters in pool order, prefix-summing
    // `cluster.triangle_count` until we cover `instance_triangle_id`. The
    // cluster that crosses the threshold owns the chosen triangle.
    let range = instance_cluster_ranges[instance_id];
    let cluster_base = range.cluster_base;
    let cluster_count = range.cluster_count;

    var picked_cluster = cluster_base;
    var triangle_in_cluster = 0u;
    var running = 0u;
    for (uint i = 0; i < cluster_count; i++)
    {
        let cluster_global = cluster_base + i;
        let tris = clusters[cluster_global].triangle_count;
        let next = running + tris;
        if (instance_triangle_id < next)
        {
            picked_cluster = cluster_global;
            triangle_in_cluster = instance_triangle_id - running;
            break;
        }
        running = next;
    }

    let triangle_data = resolve_triangle_data_full(
        instance_id, picked_cluster, triangle_in_cluster, barycentrics);

    ResolvedLightSample resolved;
    resolved.world_position = float4(triangle_data.world_position, 1.0);
    resolved.world_normal = triangle_data.world_normal;
    resolved.radiance = triangle_data.material.emissive;
    resolved.inverse_pdf = float(triangle_count) * triangle_data.triangle_area;
    return resolved;
}

/// The pdf with which the NEE picker would have generated a sample on this
/// emissive hit — the BSDF-vs-NEE MIS counterpart. Must mirror the stratified
/// pick in `generate_random_light_sample` exactly.
public float random_emissive_light_pdf(ResolvedRayHitFull hit)
{
    let stratum_probability = directional_light_count() > 0 ? 0.5 : 1.0;
    return stratum_probability * random_emissive_light_pdf_flux(hit);
}

/// The same mirror WITHOUT the directional stratum factor — the restir
/// estimator samples emissives and directionals as separate techniques.
public float random_emissive_light_pdf_flux(ResolvedRayHitFull hit)
{
    let emissive_count = emissive_light_count();
    let cdf_base = 2 + emissive_count + directional_light_count();
    let total_flux = asfloat(active_light_list[cdf_base + emissive_count]);
    var pick_prob = 1.0 / float(emissive_count);
    if (total_flux > 0.0)
    {
        // Flux from the BASE (untextured) material emissive — the CPU CDF has
        // no access to textures, so the mirror must ignore them too.
        let base = load_material_bindless(hit.material_id);
        pick_prob = pick_luminance(base.emissive) * float(hit.triangle_count) / total_flux;
    }
    return pick_prob / (float(hit.triangle_count) * hit.triangle_area);
}

/// Re-resolve a stored reservoir sample at shading time with the FULL
/// flux-weighted technique pdf folded into `inverse_pdf` — must produce the
/// same pdf `generate_random_emissive_light_sample` gave the sample when it
/// was a candidate.
public ResolvedLightSample resolve_emissive_for_restir(LightSample ls)
{
    let light_source = light_sources[ls.light_id >> 16];
    var resolved = resolve_light_sample(ls, light_source);
    let emissive_count = emissive_light_count();
    let cdf_base = 2 + emissive_count + directional_light_count();
    let total_flux = asfloat(active_light_list[cdf_base + emissive_count]);
    var pick_prob = 1.0 / float(emissive_count);
    if (total_flux > 0.0)
    {
        let triangle_count = light_source.kind >> 1;
        let base = load_material_bindless(material_ids[light_source.id]);
        pick_prob = pick_luminance(base.emissive) * float(triangle_count) / total_flux;
    }
    resolved.inverse_pdf *= 1.0 / max(pick_prob, 1e-12);
    return resolved;
}

/// One uniformly random EMISSIVE light sample — the restir DI reservoirs
/// manage emissive lights only; directional lights are a deterministic
/// handful, shaded every frame outside ReSTIR.
public GenerateRandomLightSampleResult generate_random_emissive_light_sample(inout uint rng)
{
    GenerateRandomLightSampleResult result;
    let emissive_count = emissive_light_count();
    if (emissive_count == 0)
    {
        result.light_sample = { NULL_LIGHT_ID, 0 };
        result.resolved_light_sample =
            { float4(0.0, 1.0, 0.0, 0.0), float3(0.0, -1.0, 0.0), float3(0.0), 0.0 };
        return result;
    }

    let picked = pick_emissive_weighted(emissive_count, directional_light_count(), rng);
    let light_id = active_light_list[2 + picked.index];
    let light_source = light_sources[light_id];

    let triangle_count = light_source.kind >> 1;
    let triangle_id = rand_range_u(triangle_count, rng);

    let seed = rand_u(rng);
    LightSample light_sample = { (light_id << 16) | triangle_id, seed };

    var resolved_light_sample = resolve_light_sample(light_sample, light_source);
    resolved_light_sample.inverse_pdf *= 1.0 / picked.prob;

    result.light_sample = light_sample;
    result.resolved_light_sample = resolved_light_sample;
    return result;
}

/// One stratified random light sample: pick the directional stratum (the sun)
/// or the emissive stratum with probability ½ each (when both exist), then
/// within the stratum; `inverse_pdf` carries the full pick pdf. The pick
/// indexes `active_light_list` (dense); the resulting `light_id` packs the
/// picked light's **stable slot** into `light_sources`. Returns a
/// `NULL_LIGHT_ID` sample (zero radiance) when the scene has no lights.
public GenerateRandomLightSampleResult generate_random_light_sample(inout uint rng)
{
    GenerateRandomLightSampleResult result;
    let emissive_count = emissive_light_count();
    let directional_count = directional_light_count();

    var emissive_stratum = emissive_count > 0;
    var stratum_probability = 1.0;
    if (directional_count > 0 && emissive_count > 0)
    {
        stratum_probability = 0.5;
        emissive_stratum = rand_f(rng) >= 0.5;
    }
    else if (emissive_count == 0 && directional_count == 0)
    {
        result.light_sample = { NULL_LIGHT_ID, 0 };
        result.resolved_light_sample =
            { float4(0.0, 1.0, 0.0, 0.0), float3(0.0, -1.0, 0.0), float3(0.0), 0.0 };
        return result;
    }

    uint pick;
    float pick_prob;
    if (emissive_stratum)
    {
        let picked = pick_emissive_weighted(emissive_count, directional_count, rng);
        pick = picked.index;
        pick_prob = picked.prob;
    }
    else
    {
        pick = emissive_count + rand_range_u(directional_count, rng);
        pick_prob = 1.0 / float(directional_count);
    }
    let light_id = active_light_list[2 + pick];
    let light_source = light_sources[light_id];

    var triangle_id = 0u;
    if (light_source.kind != LIGHT_SOURCE_KIND_DIRECTIONAL)
    {
        let triangle_count = light_source.kind >> 1;
        triangle_id = rand_range_u(triangle_count, rng);
    }

    let seed = rand_u(rng);
    LightSample light_sample = { (light_id << 16) | triangle_id, seed };

    var resolved_light_sample = resolve_light_sample(light_sample, light_source);
    resolved_light_sample.inverse_pdf *= 1.0 / (pick_prob * stratum_probability);

    result.light_sample = light_sample;
    result.resolved_light_sample = resolved_light_sample;
    return result;
}
