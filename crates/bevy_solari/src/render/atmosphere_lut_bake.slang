// Transmittance-LUT bake for the atmosphere VOLUMES: per volume, per texel
// (mu = cos zenith, r = radius), integrate optical depth from a point at
// radius r along the mu-direction to the atmosphere top and store
// T = exp(-(rayleigh·od_r + mie_ext·od_m)). Raygen's volume march then does
// one bilinear buffer lookup per step instead of an 8-step sun integral.
//
// One binding: THE volume buffer (header 32 B + 4×64 B volume records +
// LUT region at byte 512 — mirrors `atmosphere.rs` layout constants), viewed
// as float4s. Self-contained math (no imports) — bake cost is irrelevant.

[[vk::binding(0, 0)]] RWStructuredBuffer<float4> buf;

static const uint LUT_W = 256;
static const uint LUT_H = 64;
// float4 indices: header = [0,2), volumes = [2, 2+4·4), LUT layers at 512 B.
static const uint VOLUMES_BASE = 2;
static const uint VOLUME_VEC4S = 4;
static const uint LUT_BASE = 32; // 512 B / 16
static const uint LUT_LAYER_VEC4S = LUT_W * LUT_H;
static const uint SUN_STEPS = 64;

float ray_sphere_far(float3 o, float3 d, float r) {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    return -b + sqrt(disc);
}

float ray_sphere_near(float3 o, float3 d, float r) {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    return -b - sqrt(disc);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void bake(uint3 gid: SV_DispatchThreadID) {
    if (gid.x >= LUT_W || gid.y >= LUT_H) {
        return;
    }
    let layer = gid.z;
    let vbase = VOLUMES_BASE + layer * VOLUME_VEC4S;
    // v0 = center.xyz | bottom, v1 = rayleigh.xyz | rayleigh_h,
    // v2 = top | mie_s | mie_e | mie_h, v3 = mie_g | pads.
    let v0 = buf[vbase];
    let v1 = buf[vbase + 1];
    let v2 = buf[vbase + 2];
    let bottom = v0.w;
    let top = v2.x;
    let rayleigh = v1.xyz;
    let rayleigh_h = v1.w;
    let mie_ext = v2.z;
    let mie_h = v2.w;

    // Texel → (radius, zenith cosine). A hair above the surface so the
    // r = bottom row isn't self-shadowed by the ground test.
    let mu = (float(gid.x) / float(LUT_W - 1)) * 2.0 - 1.0;
    let r = lerp(bottom + 1e-3, top, float(gid.y) / float(LUT_H - 1));
    let p = float3(0.0, r, 0.0);
    let dir = float3(sqrt(max(1.0 - mu * mu, 0.0)), mu, 0.0);

    var t = float3(0.0);
    // Ground occlusion: the sun is below the horizon of this point.
    if (ray_sphere_near(p, dir, bottom) <= 0.0) {
        let t_top = ray_sphere_far(p, dir, top);
        if (t_top > 0.0) {
            let ds = t_top / float(SUN_STEPS);
            var od = float2(0.0);
            for (var i = 0u; i < SUN_STEPS; i += 1) {
                let h = length(p + dir * ((float(i) + 0.5) * ds)) - bottom;
                od += float2(exp(-h / rayleigh_h), exp(-h / mie_h)) * ds;
            }
            t = exp(-(rayleigh * od.x + float3(mie_ext) * od.y));
        }
    }
    buf[LUT_BASE + layer * LUT_LAYER_VEC4S + gid.y * LUT_W + gid.x] = float4(t, 1.0);
}
