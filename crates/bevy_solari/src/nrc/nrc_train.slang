// NRC fused training kernel: forward + relative-L2 loss + full dZ backward
// chain in one dispatch (one thread = one sample), replacing the tiled
// coopmat mm_nn/bias_act/loss_grad/mm_nt/relu_bwd ladder. The dW/db
// reductions stay on the coopmat kernels (nrc_mlp.wgsl mm_tn/bias_grad):
// coopVecOuterProductAccumulate needs the TrainingOptimal opaque layout,
// whose gradient read-back requires vkConvertCooperativeVectorMatrixNV —
// adopted when the NRC dispatch path owns a raw device.
//
// Weights are the TRANSPOSED mirror ([out x in] row-major per layer, the
// same buffer coopvec inference reads); the backward W^T·dZ product reads
// the identical bytes as ColumnMajor. Biases are the f16 mirror.
//
// Regenerate SPIR-V (from crates/bevy_solari/src/nrc/):
//   slangc nrc_train.slang -target spirv -o nrc_train.spv
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

// acts[l] = layer-l INPUT activations, f16 [LAYERS][batch][WIDTH]:
// [0] is the encoded batch (read); [1..5] are written for the dW reduction.
[[vk::binding(0, 0)]] RWByteAddressBuffer acts;
// w16_t: [out x in] row-major f16 per layer.
[[vk::binding(1, 0)]] ByteAddressBuffer weights_t;
// b16: f16 bias mirror, [LAYERS][WIDTH].
[[vk::binding(2, 0)]] ByteAddressBuffer biases;
// [batch][4]: rgb target + sample weight.
[[vk::binding(3, 0)]] ByteAddressBuffer targets;
// [batch][WIDTH] f32 predictions (f16-grade values, stored wide).
[[vk::binding(4, 0)]] RWByteAddressBuffer preds;
// [batch] per-sample relative-L2.
[[vk::binding(5, 0)]] RWByteAddressBuffer loss_out;
// dZ = dL/d(pre-activation), f16 [LAYERS][batch][WIDTH].
[[vk::binding(6, 0)]] RWByteAddressBuffer dz;

struct TrainParams
{
    uint batch;
    float loss_scale;
    uint pad_a;
    uint pad_b;
}
[[vk::binding(7, 0)]] ConstantBuffer<TrainParams> params;
// >= WIDTH f32 zeros. A zero CoopVec is LOADED, not splat-constructed:
// the splat lowers to OpCompositeConstructReplicateEXT, and the device
// won't have VK_EXT_shader_replicated_composites enabled.
[[vk::binding(8, 0)]] ByteAddressBuffer zeros_buf;

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
            weights_t, w_byte_off, CoopVecComponentType.Float16,
            biases, b_byte_off, CoopVecComponentType.Float16,
            CoopVecMatrixLayout.RowMajor, false, WIDTH * 2);
        if (layer < LAYERS - 1)
        {
            [ForceUnroll]
            for (int i = 0; i < WIDTH; i++)
                o[i] = max(o[i], 0.0h);
            // The dW reduction (mm_tn) reads next layer's input from global.
            o.store(acts, ((layer + 1) * params.batch + sample) * WIDTH * 2);
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
            weights_t, w_byte_off, CoopVecComponentType.Float16,
            biases, b_byte_off, CoopVecComponentType.Float16,
            CoopVecMatrixLayout.RowMajor, false, WIDTH * 2);
        var g = result_grad.g;
        if (layer < LAYERS - 1)
        {
            [ForceUnroll]
            for (int i = 0; i < WIDTH; i++)
                g[i] = pre[i] > 0.0h ? g[i] : 0.0;
        }
        // The coopmat dW reduction reads f16 dZ — quantize once, at the store.
        var g16 = HVec.load(zeros_buf, 0);
        [ForceUnroll]
        for (int i = 0; i < WIDTH; i++)
            g16[i] = half(g[i]);
        g16.store(dz, (layer * params.batch + sample) * WIDTH * 2);
        // dX = W^T · dZ, scalar f32: dx[i] = sum_j W[j,i] * dz[j] over the
        // [out x in] row-major mirror.
        MLGrad dxv;
        for (int i = 0; i < WIDTH; i++)
        {
            var sum = 0.0;
            for (int j = 0; j < WIDTH; j++)
                sum += float(weights_t.Load<half>(w_byte_off + (j * WIDTH + i) * 2)) * g[j];
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
void learn_gradient(uint tid: SV_DispatchThreadID)
{
    let sample = tid;
    if (sample >= params.batch)
        return;

    Net net;
    [ForceUnroll]
    for (uint l = 0; l < LAYERS; l++)
        net.layers[l] = { l * WIDTH * WIDTH * 2, l * WIDTH * 2, l, sample };

    let x = MLVec({ HVec.load(acts, sample * WIDTH * 2) });
    let out = eval_net(net, x);

    [ForceUnroll]
    for (int i = 0; i < WIDTH; i++)
        preds.Store<float>((sample * WIDTH + i) * 4, float(out.data[i]));

    // Loss + output-layer gradient seed, numerics per nrc_mlp.wgsl loss_grad.
    var weight = targets.Load<float>((sample * 4 + 3) * 4);
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
            let diff = float(out.data[c]) - targets.Load<float>((sample * 4 + c) * 4);
            loss += weight * diff * diff / denom;
            // clamped to the f16 range the dZ STORE quantizes into
            g0.g[c] = clamp(
                weight * params.loss_scale * 2.0 * diff / denom, -3.2e4, 3.2e4);
        }
    }
    loss_out.Store<float>(sample * 4, loss);

    var dp = diffPair(x);
    bwd_diff(eval_net)(net, dp, g0);
}
