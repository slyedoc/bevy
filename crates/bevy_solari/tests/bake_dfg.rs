//! DFG LUT bake: Monte-Carlo integrates the runtime's EXACT specular estimator
//! (bounded-VNDF sampler + D_GGX·V_SmithGGXCorrelated eval, guard clamps and dead
//! samples included) into the split-sum table `E = F0·A + B`, texel centers matching
//! `F_AB`'s linear sampling. Regenerate after any BRDF change:
//!   cargo test -p bevy_solari --release bake_dfg -- --ignored --nocapture
//! Writes src/material/dfg_baked.bin (64×64 RG f16), consumed by `insert_dfg_lut`.

const SIZE: usize = 64;
const SAMPLES: u32 = 131_072;

fn d_ggx(roughness: f32, ndoth: f32) -> f32 {
    let one_minus = 1.0 - ndoth * ndoth;
    let a = ndoth * roughness;
    let k = roughness / (one_minus + a * a);
    k * k * (1.0 / std::f32::consts::PI)
}

fn v_smith_ggx_correlated(roughness: f32, ndotv: f32, ndotl: f32) -> f32 {
    let a2 = roughness * roughness;
    let lambda_v = ndotl * ((ndotv - a2 * ndotv) * ndotv + a2).sqrt();
    let lambda_l = ndotv * ((ndotl - a2 * ndotl) * ndotl + a2).sqrt();
    0.5 / (lambda_v + lambda_l)
}

/// Port of `sampling.wgsl::sample_ggx_vndf` (bounded VNDF, general branch).
fn sample_ggx_vndf(i: [f32; 3], roughness: f32, u: [f32; 2]) -> [f32; 3] {
    let i_std = normalize([i[0] * roughness, i[1] * roughness, i[2]]);
    let phi = 2.0 * std::f32::consts::PI * u[0];
    let a = roughness;
    let s = 1.0 + (i[0] * i[0] + i[1] * i[1]).sqrt();
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * i[2] * i[2]);
    let b = if i[2] > 0.0 { k * i_std[2] } else { i_std[2] };
    let z = (1.0 - u[1]) * (1.0 + b) - b;
    let sin_theta = (1.0f32 - z * z).max(0.0).sqrt();
    let o_std = [sin_theta * phi.cos(), sin_theta * phi.sin(), z];
    let m_std = [i_std[0] + o_std[0], i_std[1] + o_std[1], i_std[2] + o_std[2]];
    let m = normalize([m_std[0] * roughness, m_std[1] * roughness, m_std[2]]);
    let idotm = dot(i, m);
    [
        2.0 * idotm * m[0] - i[0],
        2.0 * idotm * m[1] - i[1],
        2.0 * idotm * m[2] - i[2],
    ]
}

/// Port of `sampling.wgsl::ggx_vndf_pdf` (general branch; `i.z >= 0` in the bake).
fn ggx_vndf_pdf(i: [f32; 3], o: [f32; 3], roughness: f32) -> f32 {
    let m = normalize([i[0] + o[0], i[1] + o[1], i[2] + o[2]]);
    let ndf = d_ggx(roughness, m[2].clamp(0.0, 1.0));
    let ai = [roughness * i[0], roughness * i[1]];
    let len2 = ai[0] * ai[0] + ai[1] * ai[1];
    let t = (len2 + i[2] * i[2]).sqrt();
    let a = roughness;
    let s = 1.0 + (i[0] * i[0] + i[1] * i[1]).sqrt();
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * i[2] * i[2]);
    let pdf = ndf / (2.0 * (k * i[2] + t));
    if pdf.is_nan() { 0.0 } else { pdf }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = dot(v, v).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

fn pcg(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((*state >> ((*state >> 28) + 4)) ^ *state).wrapping_mul(277803737);
    ((word >> 22) ^ word) as f32 / u32::MAX as f32
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let frac = b & 0x007f_ffff;
    if exp <= 0 {
        return sign; // flush tiny to zero — LUT values are well within range
    }
    sign | ((exp as u16) << 10) | (frac >> 13) as u16
}

/// One texel's (A, B): the estimator's Fresnel-split directional albedo. Mirrors
/// `evaluate_and_sample_brdf`'s specular branch exactly — invalid samples and the
/// eval's grazing guards contribute 0, so the LUT bakes THEIR energy loss too.
fn bake_texel(ndotv: f32, perceptual_roughness: f32, seed: u32) -> (f32, f32) {
    // Floor: at α = 0 exactly D degenerates and every sample skips (an all-zero row
    // would NaN the mirror lobe weights); α ≈ 6e-5 is numerically the mirror limit.
    let roughness = (perceptual_roughness * perceptual_roughness).max(6.0e-5);
    let wo = [(1.0f32 - ndotv * ndotv).max(0.0).sqrt(), 0.0, ndotv];
    let mut rng = seed;
    let (mut a_sum, mut b_sum) = (0.0f64, 0.0f64);
    for _ in 0..SAMPLES {
        let u = [pcg(&mut rng), pcg(&mut rng)];
        let wi = sample_ggx_vndf(wo, roughness, u);
        if !(wi[2] > 0.0) {
            continue; // ggx_vndf_sample_invalid — dead path at runtime
        }
        let h = normalize([wo[0] + wi[0], wo[1] + wi[1], wo[2] + wi[2]]);
        let ndotl = wi[2];
        let ndoth = h[2];
        let ldoth = dot(wi, h);
        // Runtime guard clamps (evaluate_specular_brdf).
        if ndotl < 0.0001 || ndoth < 0.0001 || ldoth < 0.0001 || ndotv < 0.0001 {
            continue;
        }
        let pdf = ggx_vndf_pdf(wo, wi, roughness);
        if pdf <= 0.0 {
            continue;
        }
        let d = d_ggx(roughness, ndoth);
        let v = v_smith_ggx_correlated(roughness, ndotv, ndotl);
        let val = (d * v * ndotl / pdf) as f64;
        let fc = (1.0 - ldoth).powi(5) as f64;
        a_sum += val * (1.0 - fc);
        b_sum += val * fc;
    }
    let n = SAMPLES as f64;
    ((a_sum / n) as f32, (b_sum / n) as f32)
}

#[test]
#[ignore = "regenerates src/material/dfg_baked.bin — run explicitly in release"]
fn bake_dfg() {
    let mut rows: Vec<Vec<(f32, f32)>> = Vec::with_capacity(SIZE);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..SIZE)
            .map(|j| {
                scope.spawn(move || {
                    // INCLUSIVE grid: texel j holds E at exactly j/(N-1), endpoints
                    // included — F_AB's uv remap lands bilinear taps on these values.
                    // (Half-texel clamp grids can't represent the edges, and E drops
                    // ~3% inside the last half-texel toward roughness 1.)
                    let perceptual = j as f32 / (SIZE - 1) as f32;
                    (0..SIZE)
                        .map(|i| {
                            let ndotv = i as f32 / (SIZE - 1) as f32;
                            bake_texel(ndotv.max(1e-3), perceptual, (j * SIZE + i) as u32 * 9781 + 1)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            rows.push(h.join().unwrap());
        }
    });

    let mut bytes = Vec::with_capacity(SIZE * SIZE * 4);
    for row in &rows {
        for &(a, b) in row {
            bytes.extend_from_slice(&f32_to_f16_bits(a).to_le_bytes());
            bytes.extend_from_slice(&f32_to_f16_bits(b).to_le_bytes());
        }
    }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/material/dfg_baked.bin");
    std::fs::write(path, &bytes).unwrap();

    // Corner diagnostics: Ess = A + B (F0 = 1); the furnace's metal spheres read these.
    for &(i, j, label) in &[
        (SIZE - 1, SIZE - 1, "NdotV~1, r~1  "),
        (SIZE / 2, SIZE - 1, "NdotV~.5, r~1 "),
        (SIZE - 1, SIZE / 2, "NdotV~1, r~.5 "),
        (SIZE - 1, 0, "NdotV~1, r~0  "),
    ] {
        let (a, b) = rows[j][i];
        println!("{label} A={a:.4} B={b:.4} Ess={:.4}", a + b);
    }
    println!("wrote {path} ({} bytes)", bytes.len());
}
