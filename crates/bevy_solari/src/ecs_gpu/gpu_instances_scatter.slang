// Generic byte-wise scatter: places a compact `(slot, value)` delta into a
// persistent slot-indexed column. One kernel serves every `GpuColumn<T>` —
// the per-value word count is a runtime param, so a `u32` column and a
// `mat3x4` column run the same code, and the buffers are per-dispatch heap
// slots, so every column shares the same two pipelines. One thread per delta
// record. See `ecs_gpu::column::GpuColumn`.

struct ScatterParams {
    // Number of delta records this dispatch.
    uint count;
    // `size_of::<T>() / 4` — words copied per record.
    uint words_per_value;
    // X workgroup count of the 2D-split dispatch (X capped at 65535, rest in Y).
    uint groups_x;
    uint _pad;
}

[[vk::push_constant]] ScatterParams params;

// Delta: `count` tightly-packed records, each `[slot, value_word_0, …]`
// (`words_per_value + 1` u32 words).
[[vk::binding(0, 0)]] StructuredBuffer<uint> delta;
// Target column, viewed as raw u32 words (`slot * words_per_value` base).
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> column;
// History columns only: the previous-frame buffer.
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> previous;

// Flat index across the 2D-split dispatch.
uint flat_index(uint3 gid) {
    return gid.x + gid.y * params.groups_x * 64;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void scatter(uint3 gid: SV_DispatchThreadID) {
    let i = flat_index(gid);
    if (i >= params.count) {
        return;
    }
    let record_stride = params.words_per_value + 1;
    let base = i * record_stride;
    let slot = delta[base];
    let dst = slot * params.words_per_value;
    for (uint w = 0; w < params.words_per_value; w++) {
        column[dst + w] = delta[base + 1 + w];
    }
}

// Double-buffered scatter: shift the current value into `previous` before
// overwriting it (the GPU already holds last frame's value — no separate
// upload).
[shader("compute")]
[numthreads(64, 1, 1)]
void scatter_with_history(uint3 gid: SV_DispatchThreadID) {
    let i = flat_index(gid);
    if (i >= params.count) {
        return;
    }
    let record_stride = params.words_per_value + 1;
    let base = i * record_stride;
    let slot = delta[base];
    let dst = slot * params.words_per_value;
    for (uint w = 0; w < params.words_per_value; w++) {
        let new_word = delta[base + 1 + w];
        previous[dst + w] = column[dst + w];
        column[dst + w] = new_word;
    }
}
