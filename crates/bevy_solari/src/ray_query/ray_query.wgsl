// Reusable inline-`rayQuery` batch trace. One thread per ray: trace a buffer of
// rays against the scene TLAS and write a buffer of hits.
//
// Self-contained on purpose: it declares `tlas` locally and traces inline, pulling
// in NOTHING from the big `scene_bindings` module. `#import`ing even one item makes
// naga_oil preprocess that whole module, which tripped two ways: (1) the bindless
// resolve path there (`resolve_ray_hit_full` / `load_material_bindless` /
// `alpha_test` / the shared `trace_ray`) reaches `physical_load` buffer-device-address
// loads, needing the `PhysicalStorageBufferAddresses` capability that only the
// rt_pipeline's raw-VK WGSL→SPIR-V (`rt_capabilities`) enables — a plain wgpu compute
// pipeline can't; and (2) its `@group(#{SOLARI_SCENE_COLUMNS_GROUP})` preprocessor
// defs must then be supplied too. The `ray_query` type, `RayDesc`, the `rayQuery*`
// intrinsics, and the `RAY_QUERY_INTERSECTION_*` constants are builtins from
// `enable wgpu_ray_query` — no import needed.
//
// This pass has no built-in producer: `params.ray_count` defaults to 0, so until
// something fills `rays` and sets `ray_count` it traces nothing. A producer writes
// `rays[0..ray_count]`, sets `ray_count`, and reads back `hits[0..ray_count]`.

enable wgpu_ray_query;

#define_import_path bevy_solari::ray_query

// `@group(0)` is the full scene bind group (`RaytracingSceneBindings`); its binding 4
// is the TLAS. Declared here rather than imported so this shader stays decoupled from
// `scene_bindings`' module-level defs (see above). Only this one binding is used, so
// the other scene-group entries in the layout are simply unreferenced.
@group(0) @binding(4) var tlas: acceleration_structure;

// One ray to trace (std430, 32 B). `t_min` / `t_max` pack into the `vec3` tails so
// origin+direction stay 16-byte aligned.
struct Ray {
    origin: vec3<f32>,
    t_min: f32,
    direction: vec3<f32>,
    t_max: f32,
}

// One resolved hit (std430, 48 B). A miss writes `t = -1.0` and leaves the rest
// zeroed; the indices + entity bits are only meaningful on a hit. `entity_lo` /
// `entity_hi` are the picked entity's `Entity::to_bits` as `[lo, hi]`, resolved via
// the two-level instance-slot → node-slot → entity indirection below.
struct Hit {
    world_position: vec3<f32>,
    t: f32,
    world_normal: vec3<f32>,
    instance_index: u32,
    primitive_index: u32,
    geometry_index: u32,
    entity_lo: u32,
    entity_hi: u32,
}

// Dispatch parameters (uniform, 16 B). `ray_flags` is the `RayDesc` flag word
// (e.g. `RAY_FLAG_NONE`, `RAY_FLAG_TERMINATE_ON_FIRST_HIT`).
struct Params {
    ray_count: u32,
    ray_flags: u32,
    _pad0: u32,
    _pad1: u32,
}

// I/O group — `@group(0)` is the scene group (it owns `tlas`), so this lands at 1.
@group(1) @binding(0) var<storage, read> rays: array<Ray>;
@group(1) @binding(1) var<storage, read_write> hits: array<Hit>;
@group(1) @binding(2) var<uniform> params: Params;

// Picked-entity indirection. `hit.instance_index` is the PTLAS instanceCustomIndex
// = the InstanceManager INSTANCE slot (`ptlas_fill.wgsl` writes `instance_id = slot`).
// The owning entity's bits live in the transform table's per-NODE entity column, so
// resolving them needs two hops:
//   node_entity[ node_slots[hit.instance_index] ]
// `node_slots` is the instance-slot-indexed `NodeSlotColumn` (instance → transform
// node slot; `0xffffffff` = no node), `node_entity` the node-slot-indexed
// `NodeEntityColumn` (`Entity::to_bits` as `[lo, hi]`). Both are plain read-only
// storage buffers (no `physical_load`), bound at their committed sizes.
@group(1) @binding(3) var<storage, read> node_slots: array<u32>;
@group(1) @binding(4) var<storage, read> node_entity: array<vec2<u32>>;

// No-cull mask: the batch trace is scene-global, not per-view (the RT analog of
// every render layer).
const RAY_NO_CULL: u32 = 0xFFu;

@compute @workgroup_size(64, 1, 1)
fn query_rays(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.ray_count {
        return;
    }

    let r = rays[id.x];

    var rq: ray_query;
    rayQueryInitialize(
        &rq,
        tlas,
        RayDesc(params.ray_flags, RAY_NO_CULL, r.t_min, r.t_max, r.origin, r.direction),
    );
    while rayQueryProceed(&rq) {
        let c = rayQueryGetCandidateIntersection(&rq);
        // Treat alpha-masked candidates as opaque: confirming a cutout would need
        // the hit's material + uv, which only the `physical_load` resolve path can
        // fetch — deferred (see the module note).
        if c.kind == RAY_QUERY_INTERSECTION_TRIANGLE {
            rayQueryConfirmIntersection(&rq);
        }
    }
    let hit = rayQueryGetCommittedIntersection(&rq);

    if hit.kind == RAY_QUERY_INTERSECTION_NONE {
        hits[id.x] = Hit(vec3(0.0), -1.0, vec3(0.0), 0u, 0u, 0u, 0u, 0u);
        return;
    }

    // Resolve the picked entity: instance slot → node slot → entity bits. A null
    // node (`0xffffffff`, e.g. a node-less instance) leaves the bits 0 → the CPU
    // backend skips it (`Entity::from_bits(0)` is invalid).
    var entity_lo = 0u;
    var entity_hi = 0u;
    let node = node_slots[hit.instance_index];
    if node != 0xffffffffu {
        let e = node_entity[node];
        entity_lo = e.x;
        entity_hi = e.y;
    }

    hits[id.x] = Hit(
        r.origin + r.direction * hit.t,
        hit.t,
        // Normal deferred — needs vertex data via physical_load (rt_capabilities,
        // raw-VK compute) or a vertex-pool storage binding.
        vec3(0.0),
        hit.instance_index,
        hit.primitive_index,
        hit.geometry_index,
        entity_lo,
        entity_hi,
    );
}
