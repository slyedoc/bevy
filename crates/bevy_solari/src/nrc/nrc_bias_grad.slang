// db[j] = Σ_batch dZ[·,j]; one thread per column, dims.k = batch. `dz` is the
// caller-bound layer slice of the packed dZ buffer.
//
// Regenerate SPIR-V (from crates/bevy_solari/src/nrc/):
//   slangc nrc_bias_grad.slang -target spirv -entry bias_grad -o nrc_bias_grad.spv
import nrc_mlp;

struct MmDims
{
    uint m;
    uint n;
    uint k;
    uint a_off;
    uint b_off;
    uint c_off;
    uint pad_a;
    uint pad_b;
}

[[vk::binding(0, 0)]] StructuredBuffer<half> dz;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> db_out;
[[vk::binding(2, 0)]] ConstantBuffer<MmDims> dims;

[shader("compute")]
[numthreads(64, 1, 1)]
void bias_grad(uint3 gid: SV_DispatchThreadID)
{
    let col = gid.x;
    if (col >= WIDTH)
    {
        return;
    }
    var sum = 0.0;
    for (uint r = 0; r < dims.k; r++)
    {
        sum += float(dz[r * WIDTH + col]);
    }
    db_out[dims.c_off + col] = sum;
}
