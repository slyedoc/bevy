// Double-single (df64) primitives — the WGSL mirror of `transform/df64.rs`.
//
// A high-precision value is carried as two f32 lanes (hi + lo). This lets the GPU
// hold frame world translations at AU/interstellar magnitude and still emit a
// crisp origin-relative f32, with NO native f64 (WGSL/naga has no f64 scalar) and
// NO naga fork — every op below is plain f32 arithmetic plus `fma` (SPIR-V OpFma).
//
// These are INLINED into the shaders that use them (the transform compute shaders
// deliberately avoid naga_oil imports — see `rt_camera.wgsl`). Keep in lockstep
// with `df64.rs`; the render subtract in particular must match `Df64Vec3::sub_to_f32`.
//
// Storage layout of a df64 vec3 on the wire: `[hi.x, hi.y, hi.z, lo.x, lo.y, lo.z]`.

// Error-free transform of a sum: returns (s, e) with s = fl(a+b) and a+b = s+e
// exactly (Knuth/Møller TwoSum, no assumption on |a| vs |b|).
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    return vec2<f32>(s, e);
}

// Error-free transform of a product via fused multiply-add: p = fl(a*b),
// e = a*b - p exactly. `fma` is a single rounding, so `fma(a,b,-p)` is the
// exact residual.
fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let e = fma(a, b, -p);
    return vec2<f32>(p, e);
}

// Renormalize a hi/lo pair so |lo| <= 1/2 ulp(hi).
fn df_renorm(hi: f32, lo: f32) -> vec2<f32> {
    return two_sum(hi, lo);
}

// df + df  (a.x/a.y = hi/lo, likewise b). Accurate ("sloppy" variant is enough
// for chain composition at our magnitudes; upgrade to the Bailey 2-2 sum if a
// stress case ever needs it).
fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum(a.x, b.x);
    let e = s.y + (a.y + b.y);
    return df_renorm(s.x, e);
}

// df - df
fn df_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return df_add(a, vec2<f32>(-b.x, -b.y));
}

// df * f32
fn df_mul_f32(a: vec2<f32>, b: f32) -> vec2<f32> {
    let p = two_prod(a.x, b);
    let e = fma(a.y, b, p.y);
    return df_renorm(p.x, e);
}

// --- vec3 wrappers: a df64 vec3 is (hi: vec3, lo: vec3) ---

// Origin-relative render translation: (self - origin) collapsed to a small f32
// vec3. MUST match `Df64Vec3::sub_to_f32` in df64.rs: (hi-hi) + (lo-lo). The hi
// subtraction is exact when self~origin (Sterbenz); the lo term restores the
// dropped sub-ulp part. This is the only df64 op on the hot render path.
fn df3_sub_to_f32(self_hi: vec3<f32>, self_lo: vec3<f32>, o_hi: vec3<f32>, o_lo: vec3<f32>) -> vec3<f32> {
    return (self_hi - o_hi) + (self_lo - o_lo);
}
