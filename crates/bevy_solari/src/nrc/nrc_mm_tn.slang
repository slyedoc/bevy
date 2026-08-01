// C[m×n] = Aᵀ · B[k×n], A stored row-major [k×m] — the coopmat dW reduction.
// Layout mapping mirrors the retired naga kernel exactly: the plain-product
// accesses are RowMajor, the transposed A operand is the ColumnMajor load;
// load/store offsets + strides are in buffer-element units.
//
// Regenerate SPIR-V (from crates/bevy_solari/src/nrc/):
//   slangc nrc_mm_tn.slang -target spirv -entry mm_tn -o nrc_mm_tn.spv
using namespace linalg;

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

[[vk::binding(0, 0)]] StructuredBuffer<half> mat_a;
[[vk::binding(1, 0)]] StructuredBuffer<half> mat_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> mat_c;
[[vk::binding(3, 0)]] StructuredBuffer<float> zero_tile;
[[vk::binding(4, 0)]] ConstantBuffer<MmDims> dims;

static const uint TILE = 16;

typealias MatA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias MatAcc = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

[shader("compute")]
[numthreads(64, 1, 1)]
void mm_tn(uint3 wg: SV_GroupID)
{
    let row = wg.x * TILE;
    let col = wg.y * TILE;
    var acc = MatAcc.Load<CoopMatMatrixLayout.RowMajor>(zero_tile, 0, TILE);
    for (uint k = 0; k < dims.k; k += TILE)
    {
        let a = MatA.Load<CoopMatMatrixLayout.ColumnMajor>(
            mat_a, dims.a_off + k * dims.m + row, dims.m);
        let b = MatB.Load<CoopMatMatrixLayout.RowMajor>(
            mat_b, dims.b_off + k * dims.n + col, dims.n);
        acc = coopMatMulAdd<float, false>(a, b, acc);
    }
    acc.Store<CoopMatMatrixLayout.RowMajor>(mat_c, dims.c_off + row * dims.n + col, dims.n);
}
