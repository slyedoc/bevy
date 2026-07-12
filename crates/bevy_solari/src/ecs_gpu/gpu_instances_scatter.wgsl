// Generic byte-wise scatter: places a compact `(slot, value)` delta into a
// persistent slot-indexed column. One shader serves every `GpuColumn<T>` —
// the per-value word count is a runtime param, so a `u32` column and a
// `mat3x4` column run the same code. One thread per delta record.
// See `ecs_gpu::column::GpuColumn`.

struct ScatterParams {
    // Number of delta records this dispatch.
    count: u32,
    // `size_of::<T>() / 4` — words copied per record.
    words_per_value: u32,
}

// Delta: `count` tightly-packed records, each `[slot, value_word_0, …]`
// (`words_per_value + 1` u32 words).
@group(0) @binding(0) var<storage, read> delta: array<u32>;
// Target column, viewed as raw u32 words (`slot * words_per_value` base).
@group(0) @binding(1) var<storage, read_write> column: array<u32>;
@group(0) @binding(2) var<uniform> params: ScatterParams;
// History columns only: the previous-frame buffer.
@group(0) @binding(3) var<storage, read_write> previous: array<u32>;

@compute @workgroup_size(64)
fn scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    // Flat index across a 2D-split dispatch (X capped at 65535, rest in Y).
    let i = gid.x + gid.y * num_workgroups.x * 64u;
    if i >= params.count {
        return;
    }
    let record_stride = params.words_per_value + 1u;
    let base = i * record_stride;
    let slot = delta[base];
    let dst = slot * params.words_per_value;
    for (var w = 0u; w < params.words_per_value; w = w + 1u) {
        column[dst + w] = delta[base + 1u + w];
    }
}

// Double-buffered scatter: shift the current value into `previous` before
// overwriting it (the GPU already holds last frame's value — no separate
// upload).
@compute @workgroup_size(64)
fn scatter_with_history(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    // Flat index across a 2D-split dispatch (X capped at 65535, rest in Y).
    let i = gid.x + gid.y * num_workgroups.x * 64u;
    if i >= params.count {
        return;
    }
    let record_stride = params.words_per_value + 1u;
    let base = i * record_stride;
    let slot = delta[base];
    let dst = slot * params.words_per_value;
    for (var w = 0u; w < params.words_per_value; w = w + 1u) {
        let new_word = delta[base + 1u + w];
        previous[dst + w] = column[dst + w];
        column[dst + w] = new_word;
    }
}
