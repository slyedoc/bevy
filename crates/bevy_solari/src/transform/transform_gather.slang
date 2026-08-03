// Transform gather — copy each RT instance's GPU-propagated world transform into
// the instance `TransformColumn` buffer that every RT consumer already reads
// (PTLAS fill, blas sharing, the raytracing scene). One thread per instance slot:
//   transform_column[i] = world[node_slot[i]]
// This is the bridge that makes GPU propagation drive the render with no change
// to any consumer — they keep indexing transforms by instance slot.
//
// mat3x4 = 3 float4 rows per entry, matching `Affine3x4` packing.

struct GatherParams {
    // Number of instance slots to gather (instance-table high-water).
    uint instance_count;
    // Number of transform nodes the world buffer covers (out-of-range guard).
    uint node_count;
    // X workgroup count of the 2D-split dispatch (X capped at 65535, rest in Y).
    uint groups_x;
    uint _pad;
}

[[vk::push_constant]] GatherParams params;

[[vk::binding(0, 0)]] StructuredBuffer<uint> node_slot;      // per instance slot
[[vk::binding(1, 0)]] StructuredBuffer<float4> world;        // 3 per node
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> transforms; // 3 per instance (current)
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> previous;   // 3 per instance (prev frame)

[shader("compute")]
[numthreads(64, 1, 1)]
void gather(uint3 gid: SV_DispatchThreadID) {
    // Flat index across the 2D-split dispatch.
    let i = gid.x + gid.y * params.groups_x * 64;
    if (i >= params.instance_count) {
        return;
    }
    let ns = node_slot[i];
    // Unassigned / stale slot → leave this instance's transform untouched.
    if (ns >= params.node_count) {
        return;
    }
    let dst = i * 3;
    let src = ns * 3;
    // Each thread owns index `i`, so the read-then-write of `transforms[dst]`
    // is hazard-free. Shift current → previous (motion vectors / ReSTIR temporal),
    // then write this frame's propagated world as the new current.
    previous[dst] = transforms[dst];
    previous[dst + 1] = transforms[dst + 1];
    previous[dst + 2] = transforms[dst + 2];
    transforms[dst] = world[src];
    transforms[dst + 1] = world[src + 1];
    transforms[dst + 2] = world[src + 2];
}
