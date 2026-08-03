// Shared single-scattering atmosphere physics, used by the sky bake
// (`atmosphere_bake.slang`) — full-length march per cube texel. (The
// pathtracer's aerial-perspective march re-implements the same physics
// in raygen.slang against the world-space atmosphere volumes.)
//
// Pure functions: each takes the `Atmosphere` params by value, so the binding
// (push block / buffer) is the caller's concern — no global binding here.
// Units are kilometres, planet centre at the origin, world up = +Y; output
// radiance is scaled by the sun illuminance so it lands in the scene's units
// (exposed once).
module atmosphere;

// Physical atmosphere parameters + sun. Mirrors `GpuSolariAtmosphere` (Rust).
public struct Atmosphere {
    // Planet surface + atmosphere-top radii (km).
    public float bottom_radius;
    public float top_radius;
    // Rayleigh (air) scattering per channel (1/km) + density scale height (km).
    public float3 rayleigh_scattering;
    public float rayleigh_scale_height;
    // Mie (aerosol) scattering + extinction (1/km), scale height (km), phase g.
    public float mie_scattering;
    public float mie_extinction;
    public float mie_scale_height;
    public float mie_phase_g;
    // Sun: world-space direction TO the sun, and illuminance (lux).
    public float3 sun_direction;
    public float sun_illuminance;
    // Camera altitude above the surface (km).
    public float camera_altitude;
    // Aerial perspective: Koschmieder ground-level visibility in WORLD units (the
    // distance at which a surface in the densest fog fades to ~2 % contrast), plus
    // a low-lying exponential fog layer — `aerial_fog_height` scale height and
    // `aerial_fog_base` ground level (world units). `aerial_enabled` is 1.0 on an
    // atmosphere view.
    public float aerial_visibility;
    public float aerial_fog_height;
    public float aerial_fog_base;
    // Henyey-Greenstein asymmetry for the volumetric sun shafts (god rays):
    // forward-scatter sharpness of the glow toward the sun, in (-1, 1).
    public float aerial_phase_g;
    public float aerial_enabled;
}

static const float ATMOSPHERE_PI = 3.14159265358979;
static const uint ATMOSPHERE_SUN_STEPS = 8;
static const uint ATMOSPHERE_SKY_STEPS = 32;

// Result of a march: pre-illuminance in-scatter + segment transmittance.
public struct AtmosphereScatter {
    public float3 inscatter;
    public float3 transmittance;
}

// Distance to the far intersection of a ray (origin `o`, dir `d`) with a sphere
// of radius `r` centred at the origin. Negative if no positive hit.
public float atmosphere_ray_sphere_far(float3 o, float3 d, float r) {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    return -b + sqrt(disc);
}

// Distance to the near intersection (entry) with a sphere, or -1 if none ahead.
public float atmosphere_ray_sphere_near(float3 o, float3 d, float r) {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    return -b - sqrt(disc);
}

// Rayleigh / Mie density at height `h` above the surface (km).
public float2 atmosphere_densities(Atmosphere atm, float h) {
    return float2(
        exp(-h / atm.rayleigh_scale_height),
        exp(-h / atm.mie_scale_height));
}

public float atmosphere_rayleigh_phase(float cos_theta) {
    return (3.0 / (16.0 * ATMOSPHERE_PI)) * (1.0 + cos_theta * cos_theta);
}

public float atmosphere_mie_phase(float g, float cos_theta) {
    return (1.0 / (4.0 * ATMOSPHERE_PI)) * ((1.0 - g * g)
        / pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1e-4), 1.5));
}

// Optical depth from `p` toward the sun (to the atmosphere top), as
// (rayleigh, mie) density integrals. Used for sun transmittance.
public float2 atmosphere_sun_optical_depth(Atmosphere atm, float3 p) {
    let to_sun = atm.sun_direction;
    let t_top = atmosphere_ray_sphere_far(p, to_sun, atm.top_radius);
    if (t_top <= 0.0) {
        return float2(0.0);
    }
    // If the ground occludes the sun, treat as fully shadowed (large depth).
    let t_ground = atmosphere_ray_sphere_near(p, to_sun, atm.bottom_radius);
    if (t_ground > 0.0) {
        return float2(1e9);
    }
    let ds = t_top / float(ATMOSPHERE_SUN_STEPS);
    var od = float2(0.0);
    for (var i = 0u; i < ATMOSPHERE_SUN_STEPS; i += 1) {
        let s = (float(i) + 0.5) * ds;
        let h = length(p + to_sun * s) - atm.bottom_radius;
        od += atmosphere_densities(atm, h) * ds;
    }
    return od;
}

// Ray-march single-scattered sunlight from `origin` along `dir` for `t_max` km.
// Returns the (pre-illuminance) in-scatter and the camera→end transmittance.
public AtmosphereScatter atmosphere_march(
    Atmosphere atm, float3 origin, float3 dir, float t_max, uint steps)
{
    let cos_theta = dot(dir, atm.sun_direction);
    let phase_r = atmosphere_rayleigh_phase(cos_theta);
    let phase_m = atmosphere_mie_phase(atm.mie_phase_g, cos_theta);

    let ds = t_max / float(steps);
    var od_r = 0.0; // accumulated Rayleigh density along the view ray
    var od_m = 0.0; // accumulated Mie density along the view ray
    var inscatter = float3(0.0);

    for (var i = 0u; i < steps; i += 1) {
        let s = (float(i) + 0.5) * ds;
        let p = origin + dir * s;
        let h = length(p) - atm.bottom_radius;
        let d = atmosphere_densities(atm, h);
        od_r += d.x * ds;
        od_m += d.y * ds;

        // Transmittance camera→sample.
        let t_view = exp(-(atm.rayleigh_scattering * od_r
            + float3(atm.mie_extinction) * od_m));
        // Transmittance sample→sun.
        let od_sun = atmosphere_sun_optical_depth(atm, p);
        let t_sun = exp(-(atm.rayleigh_scattering * od_sun.x
            + float3(atm.mie_extinction) * od_sun.y));

        let scatter = atm.rayleigh_scattering * (d.x * phase_r)
            + float3(atm.mie_scattering) * (d.y * phase_m);
        inscatter += t_view * t_sun * scatter * ds;
    }

    AtmosphereScatter result;
    result.inscatter = inscatter;
    result.transmittance = exp(-(atm.rayleigh_scattering * od_r
        + float3(atm.mie_extinction) * od_m));
    return result;
}

// Sky radiance for a view direction `dir` from the camera — the bake's per-texel
// value, and the background/IBL on a ray miss. Marches to the atmosphere top (or
// to the ground if the ray dips below the horizon).
public float3 atmosphere_sky_radiance(Atmosphere atm, float3 dir) {
    let origin = float3(0.0, atm.bottom_radius + atm.camera_altitude, 0.0);
    var t_max = atmosphere_ray_sphere_far(origin, dir, atm.top_radius);
    let t_ground = atmosphere_ray_sphere_near(origin, dir, atm.bottom_radius);
    if (t_ground > 0.0) {
        t_max = min(t_max, t_ground);
    }
    if (t_max <= 0.0) {
        return float3(0.0);
    }
    let r = atmosphere_march(atm, origin, dir, t_max, ATMOSPHERE_SKY_STEPS);
    return r.inscatter * atm.sun_illuminance;
}

// Fog scattering/extinction coefficient (per world unit) at world height `y`: the
// ground-level Koschmieder extinction (`3.912 / aerial_visibility`) scaled by an
// exponential height falloff — densest at `aerial_fog_base`, thinning over
// `aerial_fog_height`. So the fog is a low-lying layer the elevated camera shoots
// over (crisp foreground) while distant near-ground paths accumulate haze.
// Non-absorbing fog ⇒ scattering coefficient = extinction coefficient. The
// per-step value for the volumetric aerial-perspective march (see the pathtracer).
public float atmosphere_fog_extinction(Atmosphere atm, float y) {
    let beta0 = 3.912 / max(atm.aerial_visibility, 1e-4);
    // Clamp below-`base` heights so the density can't blow up underground.
    let density = exp(-max(y - atm.aerial_fog_base, 0.0) / max(atm.aerial_fog_height, 1e-4));
    return beta0 * density;
}
