// Batched coopvec inference: one thread = one sample = one full MLP
// evaluation. `inputs` = encoded batch [batch x WIDTH], `weights_t` =
// TRANSPOSED weight mirror [out x in] row-major per layer, `biases` = f16
// bias mirror, `preds_out` = predictions.
//
// Regenerate SPIR-V (from crates/bevy_solari/src/nrc/):
//   slangc nrc_infer_coopvec.slang -target spirv -entry nrc_infer_coopvec -capability spvDescriptorHeapEXT -o nrc_infer_coopvec.spv
import nrc_mlp;

// Descriptor-heap kernel: buffers are resource-heap indices in push data.
struct InferPush
{
    uint inputs_idx;
    uint weights_t_idx;
    uint biases_idx;
    uint preds_out_idx;
    uint batch;
}
[[vk::push_constant]] ConstantBuffer<InferPush> ew;

StructuredBuffer<half> inputs() { return StructuredBuffer<half>.Handle(uint2(ew.inputs_idx, 0)); }
StructuredBuffer<half> weights_t() { return StructuredBuffer<half>.Handle(uint2(ew.weights_t_idx, 0)); }
StructuredBuffer<half> biases() { return StructuredBuffer<half>.Handle(uint2(ew.biases_idx, 0)); }
RWStructuredBuffer<half> preds_out() { return RWStructuredBuffer<half>.Handle(uint2(ew.preds_out_idx, 0)); }

[shader("compute")]
[numthreads(64, 1, 1)]
[require(spvDescriptorHeapEXT)]
[require(spvCooperativeVectorNV)]
void nrc_infer_coopvec(uint3 gid: SV_DispatchThreadID)
{
    let sample = gid.x;
    if (sample >= ew.batch)
    {
        return;
    }
    var v = HVec.load(inputs(), sample * WIDTH * 2);
    v = mlp_eval(weights_t(), biases(), v);
    v.store(preds_out(), sample * WIDTH * 2);
}
