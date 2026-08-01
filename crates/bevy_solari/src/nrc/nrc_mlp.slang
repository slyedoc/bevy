// Shared NRC MLP library (no bindings): the input encode + the coopvec MLP
// evaluation used by the per-kernel entry files (nrc_*.slang in this
// directory + the gym's generator). Buffer operands are passed as parameters
// so every kernel can declare its OWN contiguous-from-0 binding table —
// passthrough SPIR-V bindings MUST be contiguous per set: wgpu-hal's Vulkan
// backend numbers descriptor bindings sequentially by layout-entry order,
// ignoring sparse wgpu binding numbers (same trap as the raw RT bind groups).
//
// Convention: every matrix is ROW-major; coopvec matrix/bias offsets and
// strides are BYTES. A/B operands are f16 (the only NV configs).
module nrc_mlp;

public static const uint WIDTH = 64;
public static const uint ONE_BLOB_K = 4;
public static const uint FREQ_OCTAVES = 6;
public static const float PI = 3.14159265;

public typealias HVec = CoopVec<half, WIDTH>;

public float one_blob(float x, uint bin)
{
    let center = (float(bin) + 0.5) / float(ONE_BLOB_K);
    let d = x - center;
    return exp(-d * d * 32.0);
}

// Wrap-aware one-blob for the cylindrical phi coordinate (0 and 1 are the
// same direction — blob distance wraps so the seam is invisible).
public float one_blob_wrap(float x, uint bin)
{
    let center = (float(bin) + 0.5) / float(ONE_BLOB_K);
    var d = x - center;
    d -= round(d);
    return exp(-d * d * 32.0);
}

// Cylindrical equal-area (Lambert) mapping of a unit vector: (phi in [0,1)
// wrapped, z in [0,1]). Continuous everywhere except the poles (which the
// blob smoothness covers).
public float2 cyl_encode(float3 v)
{
    // Y is the cylinder axis: view directions are mostly horizontal, and the
    // axis-aligned floor/ceiling normals land exactly ON the pole where phi
    // is constant.
    return float2(
        atan2(v.x, v.z) / (2.0 * PI) + 0.5,
        v.y * 0.5 + 0.5);
}

// 62 used features, padded to WIDTH, written to `out_buf` at `out_base`.
// MUST MATCH the inline copy in rt_pipeline/raygen.slang (nrc_query).
public void nrc_encode_cyl(
    RWStructuredBuffer<half> out_buf,
    float3 pos,
    float2 dir_cs,
    float2 nrm_cs,
    float roughness,
    float3 diff_albedo,
    float3 spec_albedo,
    uint out_base)
{
    for (uint d = 0; d < 3; d++)
    {
        for (uint oct = 0; oct < FREQ_OCTAVES; oct++)
        {
            let phase = pos[d] * PI * float(1u << oct);
            out_buf[out_base + d * 12 + oct * 2] = half(sin(phase));
            out_buf[out_base + d * 12 + oct * 2 + 1] = half(cos(phase));
        }
    }
    let rough_in = 1.0 - exp(-roughness);
    for (uint b = 0; b < ONE_BLOB_K; b++)
    {
        out_buf[out_base + 36 + b] = half(one_blob_wrap(dir_cs.x, b));
        out_buf[out_base + 40 + b] = half(one_blob(dir_cs.y, b));
        out_buf[out_base + 44 + b] = half(one_blob_wrap(nrm_cs.x, b));
        out_buf[out_base + 48 + b] = half(one_blob(nrm_cs.y, b));
        out_buf[out_base + 52 + b] = half(one_blob(rough_in, b));
    }
    for (uint c = 0; c < 3; c++)
    {
        out_buf[out_base + 56 + c] = half(diff_albedo[c]);
        out_buf[out_base + 59 + c] = half(spec_albedo[c]);
    }
    out_buf[out_base + 62] = 0.0h;
    out_buf[out_base + 63] = 0.0h;
}

public void nrc_encode(
    RWStructuredBuffer<half> out_buf,
    float3 pos,
    float3 dir,
    float3 normal,
    float roughness,
    float3 diff_albedo,
    float3 spec_albedo,
    uint out_base)
{
    nrc_encode_cyl(
        out_buf, pos, cyl_encode(dir), cyl_encode(normal), roughness,
        diff_albedo, spec_albedo, out_base);
}

// The full 6-layer MLP over the TRANSPOSED [out x in] row-major weight
// mirror + f16 bias mirror (ReLU between hidden layers, elementwise — never
// a zero splat).
public HVec mlp_eval(StructuredBuffer<half> weights_t, StructuredBuffer<half> biases, HVec v)
{
    var x = v;
    for (uint l = 0; l < 6; l++)
    {
        x = coopVecMatMulAdd<half, WIDTH>(
            x, CoopVecComponentType.Float16,
            weights_t, l * WIDTH * WIDTH * 2, CoopVecComponentType.Float16,
            biases, l * WIDTH * 2, CoopVecComponentType.Float16,
            CoopVecMatrixLayout.RowMajor, false, WIDTH * 2);
        if (l < 5)
        {
            [ForceUnroll]
            for (int i = 0; i < WIDTH; i++)
                x[i] = max(x[i], 0.0h);
        }
    }
    return x;
}

// WGSL `unpack2x16unorm`.
public float2 unpack_unorm2x16(uint v)
{
    return float2(float(v & 0xffff), float(v >> 16)) / 65535.0;
}

public float2 unpack_half2x16(uint v)
{
    return float2(f16tof32(v & 0xffff), f16tof32(v >> 16));
}
