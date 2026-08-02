// Batched termination-query inference: raygen appends one query per
// terminating path (see raygen.slang); this pass batch-evaluates the MLP
// coherently (uniform weight reads, coopvec in convergent control flow) and
// composites the de-factorized radiance into the per-pixel output buffer with
// the raygen accumulation blend pre-folded into `scale`. Pixels are unique
// per frame, so the read-modify-write needs no atomics.
//
// Query-ring byte layout MUST MATCH raygen.slang and NRC_QUERY_* in
// nrc/mod.rs: count@0, train_count@4, queries at 16, 48 B each
// ([pos.xyz bits, material], [normal cyl, -wo cyl, throughput f16x2 ×2],
// [pixel, unused ×3]).
//
import nrc_mlp;

struct NrcQueryParams
{
    // Raygen output-blend weight for a deferred contribution:
    // accum_blend / rounds (1 when not accumulating).
    float scale;
    // 1 / camera exposure — the cache trains exposure-scaled.
    float inv_exposure;
    uint cap;
    uint pad;
}

[[vk::binding(0, 0)]] StructuredBuffer<half> weights_t;
[[vk::binding(1, 0)]] StructuredBuffer<half> biases;
[[vk::binding(2, 0)]] ByteAddressBuffer nrc_queries;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> out_radiance;
[[vk::binding(4, 0)]] ConstantBuffer<NrcQueryParams> qparams;

[shader("compute")]
[numthreads(64, 1, 1)]
void nrc_query_infer(uint3 gid: SV_DispatchThreadID)
{
    let i = gid.x;
    // The raygen counter keeps climbing past the cap (append attempts); only
    // the first `cap` slots hold queries.
    if (i >= min(nrc_queries.Load(0), qparams.cap))
    {
        return;
    }
    let q_off = 16 + i * 48;
    let v0 = nrc_queries.Load<uint4>(q_off);
    let v1 = nrc_queries.Load<uint4>(q_off + 16);
    let v2 = nrc_queries.Load<uint4>(q_off + 32);

    let pos_unit = float3(asfloat(v0.x), asfloat(v0.y), asfloat(v0.z));
    let m = v0.w;
    let base_color = float3(
        float((m >> 27) & 0x1f) / 31.0,
        float((m >> 21) & 0x3f) / 63.0,
        float((m >> 16) & 0x1f) / 31.0);
    let metallic = float((m >> 8) & 0xff) / 255.0;
    let rough = float(m & 0xff) / 255.0;
    let diff_alb = base_color * (1.0 - metallic);
    let spec_alb = lerp(float3(0.04), base_color, metallic);
    let nrm_cs = unpack_unorm2x16(v1.x);
    let dir_cs = unpack_unorm2x16(v1.y);
    let throughput = float3(unpack_half2x16(v1.z), unpack_half2x16(v1.w).x);
    let pixel = v2.x;

    // Encode + evaluate — the same feature layout as nrc_encode_cyl, built
    // in-register for coopvec. Seed with arbitrary buffer bytes, then write
    // EVERY element (a zero splat would lower to
    // OpCompositeConstructReplicateEXT, which the device doesn't enable).
    var v = HVec.load(weights_t, 0);
    for (uint d = 0; d < 3; d++)
    {
        for (uint oct = 0; oct < FREQ_OCTAVES; oct++)
        {
            let phase = pos_unit[d] * PI * float(1u << oct);
            v[d * 12 + oct * 2] = half(sin(phase));
            v[d * 12 + oct * 2 + 1] = half(cos(phase));
        }
    }
    let rough_in = 1.0 - exp(-rough);
    for (uint b = 0; b < ONE_BLOB_K; b++)
    {
        v[36 + b] = half(one_blob_wrap(dir_cs.x, b));
        v[40 + b] = half(one_blob(dir_cs.y, b));
        v[44 + b] = half(one_blob_wrap(nrm_cs.x, b));
        v[48 + b] = half(one_blob(nrm_cs.y, b));
        v[52 + b] = half(one_blob(rough_in, b));
    }
    for (uint c = 0; c < 3; c++)
    {
        v[56 + c] = half(diff_alb[c]);
        v[59 + c] = half(spec_alb[c]);
    }
    v[62] = 0.0h;
    v[63] = 0.0h;
    v = mlp_eval(weights_t, biases, v);
    var cache = float3(float(v[0]), float(v[1]), float(v[2]));
    cache.x = cache.x != cache.x ? 0.0 : cache.x;
    cache.y = cache.y != cache.y ? 0.0 : cache.y;
    cache.z = cache.z != cache.z ? 0.0 : cache.z;
    cache = clamp(cache, float3(0.0), float3(256.0));

    // De-factorize + de-expose, weight by the path throughput into the
    // terminated vertex, fold in the raygen blend. Alpha carries depth —
    // leave it.
    let contribution = throughput * cache
        * (diff_alb + spec_alb + float3(1.0e-2)) * qparams.inv_exposure;
    let prev = out_radiance[pixel];
    out_radiance[pixel] = float4(prev.rgb + qparams.scale * contribution, prev.a);
}
