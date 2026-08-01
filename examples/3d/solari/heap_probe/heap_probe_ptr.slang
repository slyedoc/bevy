// Pointer-only coopvec probe: CoopVec load/store + the training accumulates
// over raw BDA pointers (no bindings, no heap). Matmul intentionally absent —
// its public API is buffers-only.
struct P
{
    uint64_t w_addr;
    uint64_t out_addr;
    uint64_t zeros_addr;
}
[[vk::push_constant]] ConstantBuffer<P> push;

[require(spvCooperativeVectorNV)]
[require(spvCooperativeVectorTrainingNV)]
[shader("compute")]
[numthreads(64, 1, 1)]
void probe(uint3 tid: SV_DispatchThreadID)
{
    var v = CoopVec<half, 64>.load((half*)push.w_addr, 0);
    v.store((half*)push.out_addr, 16384 + tid.x * 128);
    void* acc = (void*)push.out_addr;
    __coopVecOuterProductAccumulateFromPointer<half, 64, 64>(
        acc, 0, v, v,
        CoopVecMatrixLayout.TrainingOptimal, CoopVecComponentType.Float32, 0);
    var g = CoopVec<float, 64>.load((float*)push.zeros_addr, 0);
    var one = g;
    for (int i = 0; i < 64; i++)
        one[i] = 1.0;
    __coopVecReduceSumAccumulateFromPointer<float, 64>(acc, 8192, one);
}
