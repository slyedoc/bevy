// Minimal reproducer for the coopvec × descriptor-heap × dispatch-mode hang
// (heap_plan.md H0 findings): one coopVecMatMulAdd + the two training
// accumulate ops over TYPED StructuredBuffer heap handles. The host probe
// dispatches this as a VK_EXT_shader_object AND as a heap-flagged compute
// pipeline and reports which completes.
//
// Regenerate SPIR-V (from examples/3d/solari/heap_probe/):
//   slangc heap_probe.slang -target spirv -entry probe -capability spvDescriptorHeapEXT -o heap_probe.spv
struct ProbePush
{
    uint w_idx;     // f16 weights: [64 x 64] row-major + 64 bias at 8192 B
    uint io_idx;    // f32 out: preds [batch x 64] then accumulators
    uint zeros_idx; // >= 64 f32 zeros
    uint batch;
    uint use_coopvec; // 0 = plain elementwise (control), 1 = coopvec
}
[[vk::push_constant]] ConstantBuffer<ProbePush> push;

StructuredBuffer<half> weights() { return StructuredBuffer<half>.Handle(uint2(push.w_idx, 0)); }
RWStructuredBuffer<float> io_f() { return RWStructuredBuffer<float>.Handle(uint2(push.io_idx, 0)); }
RWStructuredBuffer<half> io_h() { return RWStructuredBuffer<half>.Handle(uint2(push.io_idx, 0)); }
StructuredBuffer<half> zeros_h() { return StructuredBuffer<half>.Handle(uint2(push.zeros_idx, 0)); }
StructuredBuffer<float> zeros_f() { return StructuredBuffer<float>.Handle(uint2(push.zeros_idx, 0)); }
RWByteAddressBuffer io_bab() { return RWByteAddressBuffer.Handle(uint2(push.io_idx, 0)); }

static const int WIDTH = 64;

[require(spvDescriptorHeapEXT)]
[require(spvCooperativeVectorNV)]
[shader("compute")]
[numthreads(64, 1, 1)]
void probe(uint3 tid: SV_DispatchThreadID)
{
    let sample = tid.x;
    if (sample >= push.batch)
    {
        return;
    }
    if (push.use_coopvec == 0)
    {
        // Control path: typed heap loads/stores only, no coopvec.
        var acc = 0.0;
        for (int i = 0; i < WIDTH; i++)
            acc += float(weights()[i]);
        io_f()[sample] = acc;
        return;
    }
    var v = CoopVec<half, WIDTH>.load(zeros_h(), 0);
    var o = coopVecMatMulAdd<half, WIDTH>(
        v, CoopVecComponentType.Float16,
        weights(), 0, CoopVecComponentType.Float16,
        weights(), 8192, CoopVecComponentType.Float16,
        CoopVecMatrixLayout.RowMajor, false, WIDTH * 2);
    o.store(io_h(), sample * WIDTH * 2);
    coopVecOuterProductAccumulate(
        v, o, io_bab(), 0, 0,
        CoopVecMatrixLayout.TrainingOptimal, CoopVecComponentType.Float32);
    var g = CoopVec<float, WIDTH>.load(zeros_f(), 0);
    coopVecReduceSumAccumulate(g, io_bab(), 1024);
}
