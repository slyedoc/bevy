// Batched coopvec inference: one thread = one sample = one full MLP
// evaluation. `inputs` = encoded batch [batch x WIDTH], `weights_t` =
// TRANSPOSED weight mirror [out x in] row-major per layer, `biases` = f16
// bias mirror, `preds_out` = predictions.
//
import nrc_mlp;

struct EwParams
{
    uint batch;
    uint layer_off;
    uint relu;
    float loss_scale;
}

[[vk::binding(0, 0)]] StructuredBuffer<half> inputs;
[[vk::binding(1, 0)]] StructuredBuffer<half> weights_t;
[[vk::binding(2, 0)]] StructuredBuffer<half> biases;
[[vk::binding(3, 0)]] RWStructuredBuffer<half> preds_out;
[[vk::binding(4, 0)]] ConstantBuffer<EwParams> ew;

[shader("compute")]
[numthreads(64, 1, 1)]
void nrc_infer_coopvec(uint3 gid: SV_DispatchThreadID)
{
    let sample = gid.x;
    if (sample >= ew.batch)
    {
        return;
    }
    var v = HVec.load(inputs, sample * WIDTH * 2);
    v = mlp_eval(weights_t, biases, v);
    v.store(preds_out, sample * WIDTH * 2);
}
