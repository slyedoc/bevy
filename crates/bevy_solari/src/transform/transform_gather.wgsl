// Transform gather — copy each RT instance's GPU-propagated world transform into
// the instance `TransformColumn` buffer that every RT consumer already reads
// (PTLAS fill, blas sharing, the raytracing scene). One thread per instance slot:
//   transform_column[i] = world[node_slot[i]]
// This is the bridge that makes GPU propagation drive the render with no change
// to any consumer — they keep indexing transforms by instance slot.
//
// mat3x4<f32> = 3 vec4 rows per entry, matching `Affine3x4` packing.

struct GatherParams {
    // Number of instance slots to gather (instance-table high-water).
    instance_count: u32,
    // Number of transform nodes the world buffer covers (out-of-range guard).
    node_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> node_slot: array<u32>;          // per instance slot
@group(0) @binding(1) var<storage, read> world: array<vec4<f32>>;        // 3 per node
@group(0) @binding(2) var<storage, read_write> transforms: array<vec4<f32>>; // 3 per instance (current)
@group(0) @binding(3) var<storage, read_write> previous: array<vec4<f32>>;   // 3 per instance (prev frame)
@group(0) @binding(4) var<uniform> params: GatherParams;

@compute @workgroup_size(64)
fn gather(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    // Flat index across a 2D-split dispatch (X capped at 65535, rest in Y).
    let i = gid.x + gid.y * num_workgroups.x * 64u;
    if i >= params.instance_count {
        return;
    }
    let ns = node_slot[i];
    // Unassigned / stale slot → leave this instance's transform untouched.
    if ns >= params.node_count {
        return;
    }
    let dst = i * 3u;
    let src = ns * 3u;
    // Each thread owns index `i`, so the read-then-write of `transforms[dst]`
    // is hazard-free. Shift current → previous (motion vectors / ReSTIR temporal),
    // then write this frame's propagated world as the new current.
    previous[dst] = transforms[dst];
    previous[dst + 1u] = transforms[dst + 1u];
    previous[dst + 2u] = transforms[dst + 2u];
    transforms[dst] = world[src];
    transforms[dst + 1u] = world[src + 1u];
    transforms[dst + 2u] = world[src + 2u];
}
