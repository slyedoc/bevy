// Reusable inline-`RayQuery` batch trace. One thread per ray: trace a buffer of
// rays against the scene TLAS and write a buffer of hits.
//
// Self-contained on purpose: it declares `tlas` locally and traces inline,
// importing NOTHING from `scene_resolve` — the pass needs no scene mappings, so
// its kernel builds at startup with just the TLAS push-address row. The TLAS
// keeps the scene surface's (0,4) binding number so it reuses the same
// push-address convention (`[tlas address u64 @0 | set-1 slot array @8]`).
//
// This pass has no built-in producer: `params.ray_count` defaults to 0, so until
// something fills `rays` and sets `ray_count` it traces nothing. A producer writes
// `rays[0..ray_count]`, sets `ray_count`, and reads back `hits[0..ray_count]`.

[[vk::binding(4, 0)]] RaytracingAccelerationStructure tlas;

// One ray to trace (std430, 32 B). `t_min` / `t_max` pack into the `float3` tails
// so origin+direction stay 16-byte aligned.
struct Ray
{
    float3 origin;
    float t_min;
    float3 direction;
    float t_max;
}

// One resolved hit (std430, 48 B). A miss writes `t = -1.0` and leaves the rest
// zeroed; the indices + entity bits are only meaningful on a hit. `entity_lo` /
// `entity_hi` are the picked entity's `Entity::to_bits` as `[lo, hi]`, resolved via
// the two-level instance-slot → node-slot → entity indirection below.
struct Hit
{
    float3 world_position;
    float t;
    float3 world_normal;
    uint instance_index;
    uint primitive_index;
    uint geometry_index;
    uint entity_lo;
    uint entity_hi;
}

// Dispatch parameters (uniform, 16 B). `ray_flags` is the per-ray flag word
// (e.g. `RAY_FLAG_NONE`, `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH`). A uniform
// buffer, not `[[vk::push_constant]]`: a push block reads from offset 0, where
// the mapping table sources the TLAS device address.
struct Params
{
    uint ray_count;
    uint ray_flags;
    uint _pad0;
    uint _pad1;
}

// I/O set — set 0 is the scene surface (it owns `tlas`), so this lands at 1.
[[vk::binding(0, 1)]] StructuredBuffer<Ray> rays;
[[vk::binding(1, 1)]] RWStructuredBuffer<Hit> hits;
[[vk::binding(2, 1)]] ConstantBuffer<Params> params;

// Picked-entity indirection. `hit.instance_index` is the PTLAS instanceIndex
// = the InstanceManager INSTANCE slot (`ptlas_fill.slang` writes `instance_id = slot`).
// The owning entity's bits live in the transform table's per-NODE entity column, so
// resolving them needs two hops:
//   node_entity[ node_slots[hit.instance_index] ]
// `node_slots` is the instance-slot-indexed `NodeSlotColumn` (instance → transform
// node slot; `0xffffffff` = no node), `node_entity` the node-slot-indexed
// `NodeEntityColumn` (`Entity::to_bits` as `[lo, hi]`). Both are plain read-only
// storage buffers, bound at their committed sizes.
[[vk::binding(3, 1)]] StructuredBuffer<uint> node_slots;
[[vk::binding(4, 1)]] StructuredBuffer<uint2> node_entity;

// No-cull mask: the batch trace is scene-global, not per-view (the RT analog of
// every render layer).
static const uint RAY_NO_CULL = 0xFF;

[shader("compute")]
[numthreads(64, 1, 1)]
void query_rays(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= params.ray_count)
    {
        return;
    }

    let r = rays[id.x];

    RayDesc ray;
    ray.Origin = r.origin;
    ray.TMin = r.t_min;
    ray.Direction = r.direction;
    ray.TMax = r.t_max;
    RayQuery<RAY_FLAG_NONE> rq;
    rq.TraceRayInline(tlas, params.ray_flags, RAY_NO_CULL, ray);
    while (rq.Proceed())
    {
        // Treat alpha-masked candidates as opaque: confirming a cutout would
        // need the hit's material + uv, which only the chit-side resolve can
        // fetch — deferred.
        if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            rq.CommitNonOpaqueTriangleHit();
        }
    }

    if (rq.CommittedStatus() == COMMITTED_NOTHING)
    {
        hits[id.x] = { float3(0.0), -1.0, float3(0.0), 0, 0, 0, 0, 0 };
        return;
    }

    // Resolve the picked entity: instance slot → node slot → entity bits. A null
    // node (`0xffffffff`, e.g. a node-less instance) leaves the bits 0 → the CPU
    // backend skips it (`Entity::from_bits(0)` is invalid).
    var entity_lo = 0u;
    var entity_hi = 0u;
    let instance_index = rq.CommittedInstanceIndex();
    let node = node_slots[instance_index];
    if (node != 0xffffffff)
    {
        let e = node_entity[node];
        entity_lo = e.x;
        entity_hi = e.y;
    }

    let t = rq.CommittedRayT();
    hits[id.x] = {
        r.origin + r.direction * t,
        t,
        // Normal deferred — needs vertex data via the scene's BDA pools (a
        // scene_resolve import this pass deliberately avoids).
        float3(0.0),
        instance_index,
        rq.CommittedPrimitiveIndex(),
        rq.CommittedGeometryIndex(),
        entity_lo,
        entity_hi,
    };
}
