// NRC MLP kernels (the fused forward/loss/backward lives in
// nrc_train.slang; these are the coopmat dW/db reductions, adam, the
// batched coopvec inference, and the record encode).
// Convention: every matrix is ROW-major. Plain products use the T-suffixed
// coop builtins; a transposed operand falls out of the unsuffixed
// (column-major) load of the same row-major data — no transpose kernels.
// A/B operands are f16 (the only NV configs), accumulation is f32.

enable f16;
enable wgpu_cooperative_matrix;
enable wgpu_cooperative_vector;

const TILE: u32 = 16u;
const WIDTH: u32 = 64u;
const OUT_CH: u32 = 3u;
const REL_EPS: f32 = 0.01;
const ONE_BLOB_K: u32 = 4u;
const FREQ_OCTAVES: u32 = 6u;

// ---- matmul family: bindings 0-4 ----------------------------------------

struct MmDims {
    m: u32,
    n: u32,
    k: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    pad_a: u32,
    pad_b: u32,
}

@group(0) @binding(0) var<storage, read> mat_a: array<f16>;
@group(0) @binding(1) var<storage, read> mat_b: array<f16>;
@group(0) @binding(2) var<storage, read_write> mat_c: array<f32>;
@group(0) @binding(3) var<storage, read> zero_tile: array<f32>;
@group(0) @binding(4) var<uniform> dims: MmDims;

// C[m×n] = Aᵀ · B[k×n], A stored row-major [k×m]
@compute @workgroup_size(64, 1, 1)
fn mm_tn(@builtin(workgroup_id) wg: vec3<u32>) {
    let row = wg.x * TILE;
    let col = wg.y * TILE;
    var acc = coopLoadT<coop_mat16x16<f32, C>>(&zero_tile[0], TILE);
    for (var k = 0u; k < dims.k; k += TILE) {
        let a = coopLoad<coop_mat16x16<f16, A>>(&mat_a[dims.a_off + k * dims.m + row], dims.m);
        let b = coopLoadT<coop_mat16x16<f16, B>>(&mat_b[dims.b_off + k * dims.n + col], dims.n);
        acc = coopMultiplyAdd(a, b, acc);
    }
    coopStoreT(acc, &mat_c[dims.c_off + row * dims.n + col], dims.n);
}

// ---- elementwise family: bindings 5-12 -----------------------------------

struct EwParams {
    batch: u32,
    layer_off: u32,
    relu: u32,
    loss_scale: f32,
}

@group(0) @binding(6) var<storage, read_write> act_out: array<f16>;
@group(0) @binding(8) var<uniform> ew: EwParams;
@group(0) @binding(12) var<storage, read> act_mask: array<f16>;

// db[j] = Σ_batch dZ[·,j]; one thread per column, dims.k = batch
@compute @workgroup_size(64, 1, 1)
fn bias_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if col >= WIDTH { return; }
    var sum = 0.0;
    for (var r = 0u; r < dims.k; r += 1u) {
        sum += f32(act_mask[r * WIDTH + col]);
    }
    mat_c[dims.c_off + col] = sum;
}

// ---- adam: bindings 13-18 -------------------------------------------------

struct AdamParams {
    count: u32,
    step: u32,
    mirror_mode: u32,
    lr: f32,
    beta_one: f32,
    beta_two: f32,
    eps: f32,
    inv_grad_scale: f32,
    // Inference-mirror EMA weight (1 = mirrors track the live master).
    ema_alpha: f32,
    pad_a: u32,
    pad_b: u32,
    pad_c: u32,
}

@group(0) @binding(13) var<storage, read_write> master: array<f32>;
@group(0) @binding(14) var<storage, read> grad_in: array<f32>;
@group(0) @binding(15) var<storage, read_write> moment_m: array<f32>;
@group(0) @binding(16) var<storage, read_write> moment_v: array<f32>;
@group(0) @binding(17) var<storage, read_write> mirror_f16: array<f16>;
@group(0) @binding(18) var<uniform> adam_u: AdamParams;
@group(0) @binding(21) var<storage, read_write> mirror_alt: array<f16>;
@group(0) @binding(23) var<storage, read_write> ema_master: array<f32>;
@group(0) @binding(24) var<storage, read_write> mirror_ema: array<f16>;

@compute @workgroup_size(64, 1, 1)
fn adam(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= adam_u.count { return; }
    var g = grad_in[i] * adam_u.inv_grad_scale;
    if g != g || abs(g) > 1.0e6 {
        g = 0.0;
    }
    let m = adam_u.beta_one * moment_m[i] + (1.0 - adam_u.beta_one) * g;
    let v = adam_u.beta_two * moment_v[i] + (1.0 - adam_u.beta_two) * g * g;
    moment_m[i] = m;
    moment_v[i] = v;
    let t = f32(adam_u.step);
    let m_hat = m / (1.0 - pow(adam_u.beta_one, t));
    let v_hat = v / (1.0 - pow(adam_u.beta_two, t));
    let p = master[i] - adam_u.lr * m_hat / (sqrt(v_hat) + adam_u.eps);
    master[i] = p;
    // The RENDER path reads an exponential moving average of the master (the
    // optimizer's per-step wiggle stays out of the image); everything the
    // TRAINING loop touches — the forward mirror, the transposed mirror the
    // TD targets query through — stays live. Feeding the TD its own lagged
    // EMA destabilizes the bootstrap into a slow monotonic drift.
    let e = ema_master[i] + adam_u.ema_alpha * (p - ema_master[i]);
    ema_master[i] = e;
    if adam_u.mirror_mode == 1u {
        mirror_f16[i] = f16(p);
        let layer = i / (WIDTH * WIDTH);
        let row = (i % (WIDTH * WIDTH)) / WIDTH;
        let col = i % WIDTH;
        let t = layer * WIDTH * WIDTH + col * WIDTH + row;
        mirror_alt[t] = f16(p);
        mirror_ema[t] = f16(e);
    } else if adam_u.mirror_mode == 2u {
        mirror_f16[i] = f16(p);
        mirror_ema[i] = f16(e);
    }
}

// coopvec inference: one thread = one sample = one full MLP evaluation.
// mat_a = encoded inputs [batch x WIDTH], mat_b = TRANSPOSED weight mirror
// [out x in] row-major per layer, act_mask = f16 bias mirror, act_out = preds.
@compute @workgroup_size(64, 1, 1)
fn nrc_infer_coopvec(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sample = gid.x;
    if sample >= ew.batch { return; }
    var v = coopVecLoad<coop_vec64<f16>>(&mat_a, sample * WIDTH);
    let zero_vec = coopVecSplat<coop_vec64<f16>>(0.0h);
    for (var l = 0u; l < 6u; l += 1u) {
        v = coopVecMatMulAdd<coop_vec64<f16>>(
            v, &mat_b, l * WIDTH * WIDTH, &act_mask, l * WIDTH);
        if l < 5u {
            v = coopVecMax(v, zero_vec);
        }
    }
    coopVecStore(v, &act_out, sample * WIDTH);
}

// ---- input encoding --------------------------------------------------------

fn one_blob(x: f32, bin: u32) -> f32 {
    let center = (f32(bin) + 0.5) / f32(ONE_BLOB_K);
    let d = x - center;
    return exp(-d * d * 32.0);
}

// Wrap-aware one-blob for the cylindrical phi coordinate (0 and 1 are the
// same direction — blob distance wraps so the seam is invisible).
fn one_blob_wrap(x: f32, bin: u32) -> f32 {
    let center = (f32(bin) + 0.5) / f32(ONE_BLOB_K);
    var d = x - center;
    d -= round(d);
    return exp(-d * d * 32.0);
}

// Cylindrical equal-area (Lambert) mapping of a unit vector: (phi in [0,1)
// wrapped, z in [0,1]). Continuous everywhere except the poles (which the
// blob smoothness covers) — an octahedral map folds the -z hemisphere and
// its seams show as hard lines in the learned field.
fn cyl_encode(v: vec3<f32>) -> vec2<f32> {
    // Y is the cylinder axis: view directions are mostly horizontal (far
    // from the poles, where phi from tiny components is pure noise), and the
    // axis-aligned floor/ceiling normals land exactly ON the pole where phi
    // is constant.
    return vec2<f32>(
        atan2(v.x, v.z) / (2.0 * PI) + 0.5,
        v.y * 0.5 + 0.5,
    );
}

const PI: f32 = 3.14159265;

// 62 used features, padded to WIDTH.
// MUST MATCH the inline copy in rt_pipeline/raygen.wgsl (nrc_query).
fn nrc_encode_cyl(
    pos: vec3<f32>,
    dir_cs: vec2<f32>,
    nrm_cs: vec2<f32>,
    roughness: f32,
    diff_albedo: vec3<f32>,
    spec_albedo: vec3<f32>,
    out_base: u32,
) {
    var pos_v = pos;
    var diff_v = diff_albedo;
    var spec_v = spec_albedo;
    for (var d = 0u; d < 3u; d += 1u) {
        for (var oct = 0u; oct < FREQ_OCTAVES; oct += 1u) {
            let phase = pos_v[d] * PI * f32(1u << oct);
            act_out[out_base + d * 12u + oct * 2u] = f16(sin(phase));
            act_out[out_base + d * 12u + oct * 2u + 1u] = f16(cos(phase));
        }
    }
    let rough_in = 1.0 - exp(-roughness);
    for (var b = 0u; b < ONE_BLOB_K; b += 1u) {
        act_out[out_base + 36u + b] = f16(one_blob_wrap(dir_cs.x, b));
        act_out[out_base + 40u + b] = f16(one_blob(dir_cs.y, b));
        act_out[out_base + 44u + b] = f16(one_blob_wrap(nrm_cs.x, b));
        act_out[out_base + 48u + b] = f16(one_blob(nrm_cs.y, b));
        act_out[out_base + 52u + b] = f16(one_blob(rough_in, b));
    }
    for (var c = 0u; c < 3u; c += 1u) {
        act_out[out_base + 56u + c] = f16(diff_v[c]);
        act_out[out_base + 59u + c] = f16(spec_v[c]);
    }
    act_out[out_base + 62u] = 0.0h;
    act_out[out_base + 63u] = 0.0h;
}

fn nrc_encode(
    pos: vec3<f32>,
    dir: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    diff_albedo: vec3<f32>,
    spec_albedo: vec3<f32>,
    out_base: u32,
) {
    nrc_encode_cyl(
        pos, cyl_encode(dir), cyl_encode(normal), roughness,
        diff_albedo, spec_albedo, out_base);
}

// ---- production record encode: bindings 19-20, 22 -------------------------
// MUST MATCH NrcRecord in raygen.wgsl and nrc/mod.rs.

struct NrcRecord {
    pos_rough: vec4<f32>,
    dir_normal_cs: vec4<f32>,
    diff_target_r: vec4<f32>,
    spec_target_g: vec4<f32>,
    target_b_valid: vec4<f32>,
}

struct NrcTrainParams {
    count: u32,
    inv_scene_scale: f32,
    pad_a: u32,
    pad_b: u32,
}

@group(0) @binding(19) var<uniform> train_params: NrcTrainParams;
@group(0) @binding(20) var<storage, read_write> targets_out: array<f32>;
@group(0) @binding(22) var<storage, read> records: array<NrcRecord>;

// ---- batched termination-query inference: bindings 26-28 -------------------
// Raygen appends one query per terminating path (see raygen.wgsl NrcQueryBuf);
// this pass batch-evaluates the MLP coherently (uniform weight reads, coopvec
// in convergent control flow) and composites the de-factorized radiance into
// the per-pixel output buffer with the raygen accumulation blend pre-folded
// into `scale`. Pixels are unique per frame (sample-0-only append), so the
// read-modify-write needs no atomics.

// MUST MATCH NrcQueryBuf/NrcQueryGpu in raygen.wgsl and NRC_QUERY_* in nrc/mod.rs.
struct NrcQueryGpu {
    // [pos_unit.xyz (f32 bits), packed material r5g6b5+m8+r8]
    v0: vec4<u32>,
    // [normal cyl (unorm2x16), -wo cyl (unorm2x16), throughput.rg (f16x2),
    //  throughput.b (f16x2, y unused)]
    v1: vec4<u32>,
    // [pixel index, unused ×3]
    v2: vec4<u32>,
}

struct NrcQueryBuf {
    count: u32,
    train_count: u32,
    pad_b: u32,
    pad_c: u32,
    q: array<NrcQueryGpu>,
}

struct NrcQueryParams {
    // Raygen output-blend weight for a deferred contribution:
    // accum_blend / rounds (1 when not accumulating).
    scale: f32,
    // 1 / camera exposure — the cache trains exposure-scaled.
    inv_exposure: f32,
    cap: u32,
    pad: u32,
}

@group(0) @binding(26) var<storage, read> nrc_queries: NrcQueryBuf;
@group(0) @binding(27) var<storage, read_write> out_radiance: array<vec4<f32>>;
@group(0) @binding(28) var<uniform> qparams: NrcQueryParams;

@compute @workgroup_size(64, 1, 1)
fn nrc_query_infer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    // The raygen counter keeps climbing past the cap (append attempts); only
    // the first `cap` slots hold queries.
    if i >= min(nrc_queries.count, qparams.cap) { return; }
    let q = nrc_queries.q[i];

    let pos_unit = vec3<f32>(
        bitcast<f32>(q.v0.x), bitcast<f32>(q.v0.y), bitcast<f32>(q.v0.z));
    let m = q.v0.w;
    let base_color = vec3<f32>(
        f32((m >> 27u) & 0x1fu) / 31.0,
        f32((m >> 21u) & 0x3fu) / 63.0,
        f32((m >> 16u) & 0x1fu) / 31.0);
    let metallic = f32((m >> 8u) & 0xffu) / 255.0;
    let rough = f32(m & 0xffu) / 255.0;
    let diff_alb = base_color * (1.0 - metallic);
    let spec_alb = mix(vec3(0.04), base_color, metallic);
    let nrm_cs = unpack2x16unorm(q.v1.x);
    let dir_cs = unpack2x16unorm(q.v1.y);
    let throughput = vec3<f32>(
        unpack2x16float(q.v1.z), unpack2x16float(q.v1.w).x);
    let pixel = q.v2.x;

    // Encode + evaluate — the same feature layout as nrc_encode_cyl, built
    // in-register for coopvec.
    var v = coopVecSplat<coop_vec64<f16>>(0.0h);
    for (var d = 0u; d < 3u; d += 1u) {
        for (var oct = 0u; oct < FREQ_OCTAVES; oct += 1u) {
            let phase = pos_unit[d] * PI * f32(1u << oct);
            v = coopVecInsert(v, d * 12u + oct * 2u, f16(sin(phase)));
            v = coopVecInsert(v, d * 12u + oct * 2u + 1u, f16(cos(phase)));
        }
    }
    let rough_in = 1.0 - exp(-rough);
    for (var b = 0u; b < ONE_BLOB_K; b += 1u) {
        v = coopVecInsert(v, 36u + b, f16(one_blob_wrap(dir_cs.x, b)));
        v = coopVecInsert(v, 40u + b, f16(one_blob(dir_cs.y, b)));
        v = coopVecInsert(v, 44u + b, f16(one_blob_wrap(nrm_cs.x, b)));
        v = coopVecInsert(v, 48u + b, f16(one_blob(nrm_cs.y, b)));
        v = coopVecInsert(v, 52u + b, f16(one_blob(rough_in, b)));
    }
    for (var c = 0u; c < 3u; c += 1u) {
        v = coopVecInsert(v, 56u + c, f16(diff_alb[c]));
        v = coopVecInsert(v, 59u + c, f16(spec_alb[c]));
    }
    let zero_vec = coopVecSplat<coop_vec64<f16>>(0.0h);
    for (var l = 0u; l < 6u; l += 1u) {
        v = coopVecMatMulAdd<coop_vec64<f16>>(
            v, &mat_b, l * WIDTH * WIDTH, &act_mask, l * WIDTH);
        if l < 5u {
            v = coopVecMax(v, zero_vec);
        }
    }
    var cache = vec3<f32>(
        f32(coopVecExtract(v, 0u)),
        f32(coopVecExtract(v, 1u)),
        f32(coopVecExtract(v, 2u)),
    );
    cache = select(cache, vec3(0.0), cache != cache);
    cache = clamp(cache, vec3(0.0), vec3(256.0));

    // De-factorize + de-expose, weight by the path throughput into the
    // terminated vertex, fold in the raygen blend. Alpha carries depth —
    // leave it.
    let contribution = throughput * cache
        * (diff_alb + spec_alb + vec3(1.0e-2)) * qparams.inv_exposure;
    let prev = out_radiance[pixel];
    out_radiance[pixel] = vec4<f32>(prev.rgb + qparams.scale * contribution, prev.a);
}

@compute @workgroup_size(64, 1, 1)
fn nrc_encode_records(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= train_params.count { return; }
    let rec = records[idx];
    let pos = clamp(
        rec.pos_rough.xyz * train_params.inv_scene_scale + 0.5,
        vec3(0.0), vec3(1.0));
    nrc_encode_cyl(
        pos,
        rec.dir_normal_cs.xy,
        rec.dir_normal_cs.zw,
        rec.pos_rough.w,
        rec.diff_target_r.xyz,
        rec.spec_target_g.xyz,
        idx * WIDTH,
    );
    targets_out[idx * 4u] = rec.diff_target_r.w;
    targets_out[idx * 4u + 1u] = rec.spec_target_g.w;
    targets_out[idx * 4u + 2u] = rec.target_b_valid.x;
    targets_out[idx * 4u + 3u] = rec.target_b_valid.y;
}
