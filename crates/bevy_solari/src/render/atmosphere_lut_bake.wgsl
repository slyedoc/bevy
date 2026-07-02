// Transmittance-LUT bake for the atmosphere VOLUMES: per volume, per texel
// (mu = cos zenith, r = radius), integrate optical depth from a point at
// radius r along the mu-direction to the atmosphere top and store
// T = exp(-(rayleigh·od_r + mie_ext·od_m)). Raygen's volume march then does
// one bilinear buffer lookup per step instead of an 8-step sun integral.
//
// One binding: THE volume buffer (header 32 B + 4×64 B volume records +
// LUT region at byte 512 — mirrors `atmosphere.rs` layout constants), viewed
// as vec4s. Self-contained math (no imports) — bake cost is irrelevant.

@group(0) @binding(0) var<storage, read_write> buf: array<vec4<f32>>;

const LUT_W: u32 = 256u;
const LUT_H: u32 = 64u;
// vec4 indices: header = [0,2), volumes = [2, 2+4·4), LUT layers at 512 B.
const VOLUMES_BASE: u32 = 2u;
const VOLUME_VEC4S: u32 = 4u;
const LUT_BASE: u32 = 32u; // 512 B / 16
const LUT_LAYER_VEC4S: u32 = LUT_W * LUT_H;
const SUN_STEPS: u32 = 64u;

fn ray_sphere_far(o: vec3<f32>, d: vec3<f32>, r: f32) -> f32 {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return -1.0;
    }
    return -b + sqrt(disc);
}

fn ray_sphere_near(o: vec3<f32>, d: vec3<f32>, r: f32) -> f32 {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return -1.0;
    }
    return -b - sqrt(disc);
}

@compute @workgroup_size(8, 8, 1)
fn bake(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= LUT_W || gid.y >= LUT_H {
        return;
    }
    let layer = gid.z;
    let vbase = VOLUMES_BASE + layer * VOLUME_VEC4S;
    // v0 = center.xyz | bottom, v1 = rayleigh.xyz | rayleigh_h,
    // v2 = top | mie_s | mie_e | mie_h, v3 = mie_g | pads.
    let v0 = buf[vbase];
    let v1 = buf[vbase + 1u];
    let v2 = buf[vbase + 2u];
    let bottom = v0.w;
    let top = v2.x;
    let rayleigh = v1.xyz;
    let rayleigh_h = v1.w;
    let mie_ext = v2.z;
    let mie_h = v2.w;

    // Texel → (radius, zenith cosine). A hair above the surface so the
    // r = bottom row isn't self-shadowed by the ground test.
    let mu = (f32(gid.x) / f32(LUT_W - 1u)) * 2.0 - 1.0;
    let r = mix(bottom + 1e-3, top, f32(gid.y) / f32(LUT_H - 1u));
    let p = vec3<f32>(0.0, r, 0.0);
    let dir = vec3<f32>(sqrt(max(1.0 - mu * mu, 0.0)), mu, 0.0);

    var t = vec3<f32>(0.0);
    // Ground occlusion: the sun is below the horizon of this point.
    if ray_sphere_near(p, dir, bottom) <= 0.0 {
        let t_top = ray_sphere_far(p, dir, top);
        if t_top > 0.0 {
            let ds = t_top / f32(SUN_STEPS);
            var od = vec2<f32>(0.0);
            for (var i = 0u; i < SUN_STEPS; i += 1u) {
                let h = length(p + dir * ((f32(i) + 0.5) * ds)) - bottom;
                od += vec2<f32>(exp(-h / rayleigh_h), exp(-h / mie_h)) * ds;
            }
            t = exp(-(rayleigh * od.x + vec3<f32>(mie_ext) * od.y));
        }
    }
    buf[LUT_BASE + layer * LUT_LAYER_VEC4S + gid.y * LUT_W + gid.x] = vec4<f32>(t, 1.0);
}
