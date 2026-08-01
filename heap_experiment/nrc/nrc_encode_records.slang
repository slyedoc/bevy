// Production record encode: raygen's captured path vertices → encoded MLP
// inputs + training targets.
// `NrcRecord` MUST MATCH raygen.slang and nrc/mod.rs.
//
// Regenerate SPIR-V (from crates/bevy_solari/src/nrc/):
//   slangc nrc_encode_records.slang -target spirv -entry nrc_encode_records -capability spvDescriptorHeapEXT -o nrc_encode_records.spv
import nrc_mlp;

struct NrcRecord
{
    float4 pos_rough;
    float4 dir_normal_cs;
    float4 diff_target_r;
    float4 spec_target_g;
    float4 target_b_valid;
}

// Descriptor-heap kernel: buffers are resource-heap indices in push data.
struct EncodePush
{
    uint records_idx;
    uint act_out_idx;
    uint targets_out_idx;
    uint count;
    float inv_scene_scale;
}
[[vk::push_constant]] ConstantBuffer<EncodePush> train_params;

StructuredBuffer<NrcRecord> records() { return StructuredBuffer<NrcRecord>.Handle(uint2(train_params.records_idx, 0)); }
RWStructuredBuffer<half> act_out() { return RWStructuredBuffer<half>.Handle(uint2(train_params.act_out_idx, 0)); }
RWStructuredBuffer<float> targets_out() { return RWStructuredBuffer<float>.Handle(uint2(train_params.targets_out_idx, 0)); }

[shader("compute")]
[numthreads(64, 1, 1)]
[require(spvDescriptorHeapEXT)]
void nrc_encode_records(uint3 gid: SV_DispatchThreadID)
{
    let idx = gid.x;
    if (idx >= train_params.count)
    {
        return;
    }
    let rec = records()[idx];
    let pos = clamp(
        rec.pos_rough.xyz * train_params.inv_scene_scale + 0.5,
        float3(0.0), float3(1.0));
    nrc_encode_cyl(
        act_out(),
        pos,
        rec.dir_normal_cs.xy,
        rec.dir_normal_cs.zw,
        rec.pos_rough.w,
        rec.diff_target_r.xyz,
        rec.spec_target_g.xyz,
        idx * WIDTH);
    targets_out()[idx * 4] = rec.diff_target_r.w;
    targets_out()[idx * 4 + 1] = rec.spec_target_g.w;
    targets_out()[idx * 4 + 2] = rec.target_b_valid.x;
    targets_out()[idx * 4 + 3] = rec.target_b_valid.y;
}
