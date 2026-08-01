// NRC fused training kernel: forward + relative-L2 loss + full backward in
// one dispatch (one thread = one sample). The dW/db reductions happen HERE
// too: each layer's backward outer-product-accumulates dz ⊗ x into the
// TrainingOptimal-layout gradient buffer and reduce-sum-accumulates dz into
// the bias gradients — no dZ/activation stores, no separate reduction
// kernels. The dispatch chain then runs vkCmdConvertCooperativeVectorMatrixNV
// per layer to convert the opaque-layout dW to the row-major f32 adam reads.
//
// Weights are the TRANSPOSED mirror ([out x in] row-major per layer, the
// same buffer coopvec inference reads); the backward W^T·dZ product reads
// the identical bytes as ColumnMajor. Biases are the f16 mirror.
//
// Regenerate SPIR-V (from crates/bevy_solari/src/nrc/):
//   slangc nrc_train.slang -target spirv -capability spvDescriptorHeapEXT -o nrc_train.spv
//
// The layer chain is composed by Slang autodiff (bwd_diff through
// [Differentiable] evals); each layer's hand-written backward recomputes
// its pre-activation, masks the incoming gradient, records dZ for the
// coopmat dW reduction, and returns W^T·dZ. The loss gradient seed is
// hand-built to keep the production numerics exact: per-sample weight in
// targets.w, non-finite prediction kills the whole sample (0*NaN = NaN —
// the weight alone can't save the sum), the denominator is the shared
// PREDICTION LUMINANCE (paper §4: per-channel denominators decouple the
// channels into init-dependent equilibria), and the f16 output gradient is
// clamped to ±3.2e4 or adam turns inf-inf into permanent NaN weights.
// 1/batch is deliberately NOT folded in (it would underflow f16 gradients);
// adam divides it out via inv_grad_scale.

static const int WIDTH = 64;
static const int LAYERS = 6;
static const int OUT_CH = 3;
static const float REL_EPS = 0.01;

// Descriptor-heap kernel: NO bindings — every buffer is a resource-heap
// index carried in push data, resolved at the use site (see heap_plan.md).
struct LearnPush
{
    // Resource-heap indices.
    uint acts_idx;      // encoded batch, f16 [batch][WIDTH] (written by encode)
    uint weights_t_idx; // w16_t: [out x in] row-major f16 per layer
    uint biases_idx;    // b16: f16 bias mirror, [LAYERS][WIDTH]
    uint targets_idx;   // [batch][4]: rgb target + sample weight
    uint preds_idx;     // [batch][WIDTH] f32 predictions (f16-grade, stored wide)
    uint loss_idx;      // [batch] per-sample relative-L2
    // dW accumulator: [LAYERS] TrainingOptimal-layout f32 blocks of
    // `opt_size` bytes each (opaque device layout; sized by the host
    // vkConvertCooperativeVectorMatrixNV query). Cleared each step; the
    // outer-product accumulate is atomic across all samples.
    uint dw_opt_idx;
    uint db_idx; // db accumulator: f32 [LAYERS][WIDTH]. Cleared each step.
    // >= WIDTH f32 zeros. A zero CoopVec is LOADED, not splat-constructed:
    // the splat lowers to OpCompositeConstructReplicateEXT, and the device
    // won't have VK_EXT_shader_replicated_composites enabled.
    uint zeros_idx;
    uint batch;
    float loss_scale;
    // Bytes per layer's TrainingOptimal dW block (host convert query).
    uint opt_size;
}
[[vk::push_constant]] ConstantBuffer<LearnPush> push;

// ByteAddressBuffer handles feed ONLY the opaque coopvec operands; every
// TYPED element access goes through a StructuredBuffer<T> handle instead —
// slangc v2026.14.1 dedups BAB heap pointees to one element type, so a
// typed .Load<T> through a BAB handle emits invalid SPIR-V (heap_plan.md).
StructuredBuffer<half> acts() { return StructuredBuffer<half>.Handle(uint2(push.acts_idx, 0)); }
StructuredBuffer<half> weights_t() { return StructuredBuffer<half>.Handle(uint2(push.weights_t_idx, 0)); }
StructuredBuffer<half> biases() { return StructuredBuffer<half>.Handle(uint2(push.biases_idx, 0)); }
StructuredBuffer<float> targets_f() { return StructuredBuffer<float>.Handle(uint2(push.targets_idx, 0)); }
RWStructuredBuffer<float> preds_f() { return RWStructuredBuffer<float>.Handle(uint2(push.preds_idx, 0)); }
RWStructuredBuffer<float> loss_f() { return RWStructuredBuffer<float>.Handle(uint2(push.loss_idx, 0)); }
RWStructuredBuffer<float> dw_opt() { return RWStructuredBuffer<float>.Handle(uint2(push.dw_opt_idx, 0)); }
RWStructuredBuffer<float> db() { return RWStructuredBuffer<float>.Handle(uint2(push.db_idx, 0)); }
StructuredBuffer<half> zeros_h() { return StructuredBuffer<half>.Handle(uint2(push.zeros_idx, 0)); }
StructuredBuffer<float> zeros_f() { return StructuredBuffer<float>.Handle(uint2(push.zeros_idx, 0)); }

typealias HVec = CoopVec<half, WIDTH>;

// The gradient chain vector: a plain f32 array, NOT a CoopVec. The dZ
// chain crosses up to five W^T products, and carrying it in f16 loses
// enough precision that early-layer dW drifts past the FD certification
// (worst err/tol 1.13); the f32-input coopvec matmul config is not
// supported (the driver's pipeline compiler segfaults on it), so the
// backward products run as scalar f32 loops — the same accumulation the
// coopmat path had, and the per-layer weight reads are warp-uniform.
// dZ still QUANTIZES to f16 at the store, matching the coopmat path.
struct MLGrad : IDifferentiable
{
    float g[WIDTH];
    typealias Differential = MLGrad;

    static MLGrad zero()
    {
        MLGrad z;
        for (int i = 0; i < WIDTH; i++)
            z.g[i] = 0.0;
        return z;
    }

    [Differentiable]
    static Differential dadd(Differential a, Differential b)
    {
        var r = a;
        for (int i = 0; i < WIDTH; i++)
            r.g[i] += b.g[i];
        return r;
    }
    [Differentiable]
    static Differential dmul<U : __BuiltinRealType>(U s, Differential d)
    {
        var r = d;
        for (int i = 0; i < WIDTH; i++)
            r.g[i] *= __realCast<float>(s);
        return r;
    }
    [Differentiable]
    static Differential dzero() { return no_diff(zero()); }
}

// CoopVec wrapped for differentiable contexts (after the SIGGRAPH 2025
// neural-shading course MLVec); its differential is the f32 MLGrad.
struct MLVec : IDifferentiable
{
    no_diff HVec data;
    typealias Differential = MLGrad;
}

// One 64x64 feed-forward layer over the shared mirror buffers. All fields
// are plain offsets/ids (non-differentiable), so autodiff carries no state
// through `this`.
struct Layer
{
    uint w_byte_off; // into weights_t
    uint b_byte_off; // into biases
    uint layer;      // 0-based; < LAYERS-1 applies ReLU
    uint sample;

    [Differentiable]
    MLVec eval(MLVec input)
    {
        var o = coopVecMatMulAdd<half, WIDTH>(
            input.data, CoopVecComponentType.Float16,
            weights_t(), w_byte_off, CoopVecComponentType.Float16,
            biases(), b_byte_off, CoopVecComponentType.Float16,
            CoopVecMatrixLayout.RowMajor, false, WIDTH * 2);
        if (layer < LAYERS - 1)
        {
            [ForceUnroll]
            for (int i = 0; i < WIDTH; i++)
                o[i] = max(o[i], 0.0h);
        }
        return { o };
    }

    [BackwardDerivativeOf(eval)]
    void evalBwd(inout DifferentialPair<MLVec> input, MLGrad result_grad)
    {
        // Recompute the pre-activation for the ReLU mask (activations live
        // in registers; re-running one matmul beats spilling them).
        var pre = coopVecMatMulAdd<half, WIDTH>(
            input.p.data, CoopVecComponentType.Float16,
            weights_t(), w_byte_off, CoopVecComponentType.Float16,
            biases(), b_byte_off, CoopVecComponentType.Float16,
            CoopVecMatrixLayout.RowMajor, false, WIDTH * 2);
        var g = result_grad.g;
        if (layer < LAYERS - 1)
        {
            [ForceUnroll]
            for (int i = 0; i < WIDTH; i++)
                g[i] = pre[i] > 0.0h ? g[i] : 0.0;
        }
        // The dW accumulate takes f16 dZ — quantize once, here (the same
        // f16-products-summed-in-f32 numerics the coopmat reduction had).
        var g16 = HVec.load(zeros_h(), 0);
        [ForceUnroll]
        for (int i = 0; i < WIDTH; i++)
            g16[i] = half(g[i]);
        // dW[i][j] += x[i] * dz[j] — [in x out], matching the row-major f32
        // gradient adam consumes after the per-layer layout convert.
        coopVecOuterProductAccumulate(
            input.p.data,
            g16,
            dw_opt(),
            layer * push.opt_size,
            0,
            CoopVecMatrixLayout.TrainingOptimal,
            CoopVecComponentType.Float32);
        // db[j] += dz[j], f32 accumulation.
        var g32 = CoopVec<float, WIDTH>.load(zeros_f(), 0);
        [ForceUnroll]
        for (int i = 0; i < WIDTH; i++)
            g32[i] = g[i];
        coopVecReduceSumAccumulate(g32, db(), layer * WIDTH * 4);
        // dX = W^T · dZ, scalar f32: dx[i] = sum_j W[j,i] * dz[j] over the
        // [out x in] row-major mirror.
        MLGrad dxv;
        for (int i = 0; i < WIDTH; i++)
        {
            var sum = 0.0;
            for (int j = 0; j < WIDTH; j++)
                sum += float(weights_t()[(w_byte_off >> 1) + j * WIDTH + i]) * g[j];
            dxv.g[i] = sum;
        }
        input = diffPair(input.p, dxv);
    }
}

struct Net
{
    Layer layers[LAYERS];
}

[Differentiable]
MLVec eval_net(Net net, MLVec x)
{
    var v = x;
    [ForceUnroll]
    for (int l = 0; l < LAYERS; l++)
        v = net.layers[l].eval(v);
    return v;
}

[shader("compute")]
[numthreads(64, 1, 1)]
[require(spvCooperativeVectorNV)]
[require(spvDescriptorHeapEXT)]
void learn_gradient(uint tid: SV_DispatchThreadID)
{
    let sample = tid;
    if (sample >= push.batch)
        return;

    Net net;
    [ForceUnroll]
    for (uint l = 0; l < LAYERS; l++)
        net.layers[l] = { l * WIDTH * WIDTH * 2, l * WIDTH * 2, l, sample };

    let x = MLVec({ HVec.load(acts(), sample * WIDTH * 2) });
    let out = eval_net(net, x);

    [ForceUnroll]
    for (int i = 0; i < WIDTH; i++)
        preds_f()[sample * WIDTH + i] = float(out.data[i]);

    // Loss + output-layer gradient seed, numerics per nrc_mlp.slang loss_grad.
    var weight = targets_f()[sample * 4 + 3];
    [ForceUnroll]
    for (int c = 0; c < OUT_CH; c++)
    {
        let p = float(out.data[c]);
        if (isnan(p) || abs(p) > 6.0e4)
            weight = 0.0;
    }
    let lum = 0.2126 * float(out.data[0])
        + 0.7152 * float(out.data[1])
        + 0.0722 * float(out.data[2]);
    let denom = lum * lum + REL_EPS;
    var loss = 0.0;
    var g0 = MLGrad.zero();
    if (weight > 0.0)
    {
        [ForceUnroll]
        for (int c = 0; c < OUT_CH; c++)
        {
            let diff = float(out.data[c]) - targets_f()[sample * 4 + c];
            loss += weight * diff * diff / denom;
            // clamped to the f16 range the dZ STORE quantizes into
            g0.g[c] = clamp(
                weight * push.loss_scale * 2.0 * diff / denom, -3.2e4, 3.2e4);
        }
    }
    loss_f()[sample] = loss;

    var dp = diffPair(x);
    bwd_diff(eval_net)(net, dp, g0);
}
