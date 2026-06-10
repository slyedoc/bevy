#define_import_path bevy_solari::atmosphere

// Shared single-scattering atmosphere physics, used by two passes:
//   * the sky bake (`atmosphere_bake.wgsl`) — full-length march per cube texel,
//   * the pathtracer (`pathtracer.wgsl`) — short march over the camera→primary-hit
//     segment for aerial perspective (distance haze), primary rays only.
//
// Pure functions: each takes the `Atmosphere` uniform by value, so the binding
// (group/slot) is the caller's concern — no global binding here. Units are
// kilometres, planet centre at the origin, world up = +Y; output radiance is
// scaled by the sun illuminance so it lands in the scene's units (exposed once).

// Physical atmosphere parameters + sun. Mirrors `GpuSolariAtmosphere` (Rust).
struct Atmosphere {
    // Planet surface + atmosphere-top radii (km).
    bottom_radius: f32,
    top_radius: f32,
    // Rayleigh (air) scattering per channel (1/km) + density scale height (km).
    rayleigh_scattering: vec3<f32>,
    rayleigh_scale_height: f32,
    // Mie (aerosol) scattering + extinction (1/km), scale height (km), phase g.
    mie_scattering: f32,
    mie_extinction: f32,
    mie_scale_height: f32,
    mie_phase_g: f32,
    // Sun: world-space direction TO the sun, and illuminance (lux).
    sun_direction: vec3<f32>,
    sun_illuminance: f32,
    // Camera altitude above the surface (km).
    camera_altitude: f32,
    // Aerial perspective: Koschmieder ground-level visibility in WORLD units (the
    // distance at which a surface in the densest fog fades to ~2 % contrast), plus
    // a low-lying exponential fog layer — `aerial_fog_height` scale height and
    // `aerial_fog_base` ground level (world units). `aerial_enabled` is 1.0 on an
    // atmosphere view.
    aerial_visibility: f32,
    aerial_fog_height: f32,
    aerial_fog_base: f32,
    // Henyey-Greenstein asymmetry for the volumetric sun shafts (god rays):
    // forward-scatter sharpness of the glow toward the sun, in (-1, 1).
    aerial_phase_g: f32,
    aerial_enabled: f32,
}

const ATMOSPHERE_PI: f32 = 3.14159265358979;
const ATMOSPHERE_SUN_STEPS: u32 = 8u;
const ATMOSPHERE_SKY_STEPS: u32 = 32u;

// Result of a march: pre-illuminance in-scatter + segment transmittance.
struct AtmosphereScatter {
    inscatter: vec3<f32>,
    transmittance: vec3<f32>,
}

// Distance to the far intersection of a ray (origin `o`, dir `d`) with a sphere
// of radius `r` centred at the origin. Negative if no positive hit.
fn atmosphere_ray_sphere_far(o: vec3<f32>, d: vec3<f32>, r: f32) -> f32 {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return -1.0;
    }
    return -b + sqrt(disc);
}

// Distance to the near intersection (entry) with a sphere, or -1 if none ahead.
fn atmosphere_ray_sphere_near(o: vec3<f32>, d: vec3<f32>, r: f32) -> f32 {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return -1.0;
    }
    return -b - sqrt(disc);
}

// Rayleigh / Mie density at height `h` above the surface (km).
fn atmosphere_densities(atm: Atmosphere, h: f32) -> vec2<f32> {
    return vec2<f32>(
        exp(-h / atm.rayleigh_scale_height),
        exp(-h / atm.mie_scale_height),
    );
}

fn atmosphere_rayleigh_phase(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * ATMOSPHERE_PI)) * (1.0 + cos_theta * cos_theta);
}

fn atmosphere_mie_phase(g: f32, cos_theta: f32) -> f32 {
    return (1.0 / (4.0 * ATMOSPHERE_PI)) * ((1.0 - g * g)
        / pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1e-4), 1.5));
}

// Optical depth from `p` toward the sun (to the atmosphere top), as
// (rayleigh, mie) density integrals. Used for sun transmittance.
fn atmosphere_sun_optical_depth(atm: Atmosphere, p: vec3<f32>) -> vec2<f32> {
    let to_sun = atm.sun_direction;
    let t_top = atmosphere_ray_sphere_far(p, to_sun, atm.top_radius);
    if t_top <= 0.0 {
        return vec2<f32>(0.0);
    }
    // If the ground occludes the sun, treat as fully shadowed (large depth).
    let t_ground = atmosphere_ray_sphere_near(p, to_sun, atm.bottom_radius);
    if t_ground > 0.0 {
        return vec2<f32>(1e9);
    }
    let ds = t_top / f32(ATMOSPHERE_SUN_STEPS);
    var od = vec2<f32>(0.0);
    for (var i = 0u; i < ATMOSPHERE_SUN_STEPS; i = i + 1u) {
        let s = (f32(i) + 0.5) * ds;
        let h = length(p + to_sun * s) - atm.bottom_radius;
        od += atmosphere_densities(atm, h) * ds;
    }
    return od;
}

// Ray-march single-scattered sunlight from `origin` along `dir` for `t_max` km.
// Returns the (pre-illuminance) in-scatter and the camera→end transmittance.
fn atmosphere_march(
    atm: Atmosphere, origin: vec3<f32>, dir: vec3<f32>, t_max: f32, steps: u32,
) -> AtmosphereScatter {
    let cos_theta = dot(dir, atm.sun_direction);
    let phase_r = atmosphere_rayleigh_phase(cos_theta);
    let phase_m = atmosphere_mie_phase(atm.mie_phase_g, cos_theta);

    let ds = t_max / f32(steps);
    var od_r = 0.0; // accumulated Rayleigh density along the view ray
    var od_m = 0.0; // accumulated Mie density along the view ray
    var inscatter = vec3<f32>(0.0);

    for (var i = 0u; i < steps; i = i + 1u) {
        let s = (f32(i) + 0.5) * ds;
        let p = origin + dir * s;
        let h = length(p) - atm.bottom_radius;
        let d = atmosphere_densities(atm, h);
        od_r += d.x * ds;
        od_m += d.y * ds;

        // Transmittance camera→sample.
        let t_view = exp(-(atm.rayleigh_scattering * od_r
            + vec3<f32>(atm.mie_extinction) * od_m));
        // Transmittance sample→sun.
        let od_sun = atmosphere_sun_optical_depth(atm, p);
        let t_sun = exp(-(atm.rayleigh_scattering * od_sun.x
            + vec3<f32>(atm.mie_extinction) * od_sun.y));

        let scatter = atm.rayleigh_scattering * (d.x * phase_r)
            + vec3<f32>(atm.mie_scattering) * (d.y * phase_m);
        inscatter += t_view * t_sun * scatter * ds;
    }

    let transmittance = exp(-(atm.rayleigh_scattering * od_r
        + vec3<f32>(atm.mie_extinction) * od_m));
    return AtmosphereScatter(inscatter, transmittance);
}

// Sky radiance for a view direction `dir` from the camera — the bake's per-texel
// value, and the background/IBL on a ray miss. Marches to the atmosphere top (or
// to the ground if the ray dips below the horizon).
fn atmosphere_sky_radiance(atm: Atmosphere, dir: vec3<f32>) -> vec3<f32> {
    let origin = vec3<f32>(0.0, atm.bottom_radius + atm.camera_altitude, 0.0);
    var t_max = atmosphere_ray_sphere_far(origin, dir, atm.top_radius);
    let t_ground = atmosphere_ray_sphere_near(origin, dir, atm.bottom_radius);
    if t_ground > 0.0 {
        t_max = min(t_max, t_ground);
    }
    if t_max <= 0.0 {
        return vec3<f32>(0.0);
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
fn atmosphere_fog_extinction(atm: Atmosphere, y: f32) -> f32 {
    let beta0 = 3.912 / max(atm.aerial_visibility, 1e-4);
    // Clamp below-`base` heights so the density can't blow up underground.
    let density = exp(-max(y - atm.aerial_fog_base, 0.0) / max(atm.aerial_fog_height, 1e-4));
    return beta0 * density;
}
