// The M0 mapping-mode probe (moonshot_plan.md): the SAME coopvec math as
// heap_probe.slang, but over CLASSIC [[vk::binding]] declarations — no heap
// capability, no untyped pointers. The host maps set 0 to heap descriptors
// via VkShaderDescriptorSetAndBindingMappingInfoEXT (HEAP_WITH_CONSTANT_OFFSET),
// aliasing bindings 1/2/3 onto one io descriptor and 4/5 onto zeros.
// COMPLETED here while --mode pipeline hangs = the H0 wall is untyped-pointer
// heap access, not the heap itself, and no .slang file ever needs .Handle().
//
// Regenerate SPIR-V (from examples/3d/solari/heap_probe/):
//   slangc heap_probe_mapped.slang -target spirv -entry probe -o heap_probe_mapped.spv
struct ProbePush
{
    uint w_idx;     // unused here (descriptors come from the mapping)
    uint io_idx;    // unused
    uint zeros_idx; // unused
    uint batch;
    uint use_coopvec; // 0 = plain elementwise (control), 1 = coopvec
}
[[vk::push_constant]] ConstantBuffer<ProbePush> push;

// f16 weights: [64 x 64] row-major + 64 bias at 8192 B
[[vk::binding(0, 0)]] StructuredBuffer<half> weights;
// f32 out: preds [batch x 64] then accumulators — three views, one descriptor
[[vk::binding(1, 0)]] RWStructuredBuffer<float> io_f;
[[vk::binding(2, 0)]] RWStructuredBuffer<half> io_h;
[[vk::binding(3, 0)]] RWByteAddressBuffer io_bab;
// >= 64 f32 zeros — two views, one descriptor
[[vk::binding(4, 0)]] StructuredBuffer<half> zeros_h;
[[vk::binding(5, 0)]] StructuredBuffer<float> zeros_f;

static const int WIDTH = 64;

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
        // Control path: mapped loads/stores only, no coopvec.
        var acc = 0.0;
        for (int i = 0; i < WIDTH; i++)
            acc += float(weights[i]);
        io_f[sample] = acc;
        return;
    }
    var v = CoopVec<half, WIDTH>.load(zeros_h, 0);
    var o = coopVecMatMulAdd<half, WIDTH>(
        v, CoopVecComponentType.Float16,
        weights, 0, CoopVecComponentType.Float16,
        weights, 8192, CoopVecComponentType.Float16,
        CoopVecMatrixLayout.RowMajor, false, WIDTH * 2);
    o.store(io_h, sample * WIDTH * 2);
    coopVecOuterProductAccumulate(
        v, o, io_bab, 0, 0,
        CoopVecMatrixLayout.TrainingOptimal, CoopVecComponentType.Float32);
    var g = CoopVec<float, WIDTH>.load(zeros_f, 0);
    coopVecReduceSumAccumulate(g, io_bab, 1024);
}
