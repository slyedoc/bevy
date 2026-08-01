// Adam optimizer step + the f16 inference mirrors (forward, transposed, EMA).
//
// Regenerate SPIR-V (from crates/bevy_solari/src/nrc/):
//   slangc nrc_adam.slang -target spirv -entry adam -capability spvDescriptorHeapEXT -o nrc_adam.spv
import nrc_mlp;

// Descriptor-heap kernel: buffers are resource-heap indices in push data.
struct AdamPush
{
    uint master_idx;
    uint grad_idx;
    uint m_idx;
    uint v_idx;
    uint mirror_f16_idx;
    uint mirror_alt_idx;
    uint ema_idx;
    uint mirror_ema_idx;
    uint count;
    uint step;
    uint mirror_mode;
    float lr;
    float beta_one;
    float beta_two;
    float eps;
    float inv_grad_scale;
    // Inference-mirror EMA weight (1 = mirrors track the live master).
    float ema_alpha;
}
[[vk::push_constant]] ConstantBuffer<AdamPush> adam_u;

RWStructuredBuffer<float> master() { return RWStructuredBuffer<float>.Handle(uint2(adam_u.master_idx, 0)); }
StructuredBuffer<float> grad_in() { return StructuredBuffer<float>.Handle(uint2(adam_u.grad_idx, 0)); }
RWStructuredBuffer<float> moment_m() { return RWStructuredBuffer<float>.Handle(uint2(adam_u.m_idx, 0)); }
RWStructuredBuffer<float> moment_v() { return RWStructuredBuffer<float>.Handle(uint2(adam_u.v_idx, 0)); }
RWStructuredBuffer<half> mirror_f16() { return RWStructuredBuffer<half>.Handle(uint2(adam_u.mirror_f16_idx, 0)); }
RWStructuredBuffer<half> mirror_alt() { return RWStructuredBuffer<half>.Handle(uint2(adam_u.mirror_alt_idx, 0)); }
RWStructuredBuffer<float> ema_master() { return RWStructuredBuffer<float>.Handle(uint2(adam_u.ema_idx, 0)); }
RWStructuredBuffer<half> mirror_ema() { return RWStructuredBuffer<half>.Handle(uint2(adam_u.mirror_ema_idx, 0)); }

[shader("compute")]
[numthreads(64, 1, 1)]
[require(spvDescriptorHeapEXT)]
void adam(uint3 gid: SV_DispatchThreadID)
{
    let i = gid.x;
    if (i >= adam_u.count)
    {
        return;
    }
    var g = grad_in()[i] * adam_u.inv_grad_scale;
    if (g != g || abs(g) > 1.0e6)
    {
        g = 0.0;
    }
    let m = adam_u.beta_one * moment_m()[i] + (1.0 - adam_u.beta_one) * g;
    let v = adam_u.beta_two * moment_v()[i] + (1.0 - adam_u.beta_two) * g * g;
    moment_m()[i] = m;
    moment_v()[i] = v;
    let t = float(adam_u.step);
    let m_hat = m / (1.0 - pow(adam_u.beta_one, t));
    let v_hat = v / (1.0 - pow(adam_u.beta_two, t));
    let p = master()[i] - adam_u.lr * m_hat / (sqrt(v_hat) + adam_u.eps);
    master()[i] = p;
    // The RENDER path reads an exponential moving average of the master (the
    // optimizer's per-step wiggle stays out of the image); everything the
    // TRAINING loop touches — the forward mirror, the transposed mirror the
    // TD targets query through — stays live. Feeding the TD its own lagged
    // EMA destabilizes the bootstrap into a slow monotonic drift.
    let e = ema_master()[i] + adam_u.ema_alpha * (p - ema_master()[i]);
    ema_master()[i] = e;
    if (adam_u.mirror_mode == 1)
    {
        mirror_f16()[i] = half(p);
        let layer = i / (WIDTH * WIDTH);
        let row = (i % (WIDTH * WIDTH)) / WIDTH;
        let col = i % WIDTH;
        let ti = layer * WIDTH * WIDTH + col * WIDTH + row;
        mirror_alt()[ti] = half(p);
        mirror_ema()[ti] = half(e);
    }
    else if (adam_u.mirror_mode == 2)
    {
        mirror_f16()[i] = half(p);
        mirror_ema()[i] = half(e);
    }
}
