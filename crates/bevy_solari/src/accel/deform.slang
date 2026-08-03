// Skeletal deform — linear-blend skinning of animated cluster meshes on the GPU.
//
// One thread per (active animated slot, vertex). For each vertex it blends the
// 4 influencing joints' skin matrices (`joint_world(palette[j]) * inverse_bind[j]`,
// both `mat3x4` affines), transforms the rest-pose position/normal, then
// pre-multiplies the inverse of the instance's own world transform so the
// output pool holds MESH-LOCAL skinned positions — keeping the resolve path's
// `transforms[instance_id]` application identical to static instances.
//
// Inputs are read straight from the GPU transform table (joints are nodes) and
// the shared cluster pools; no CPU skin-matrix upload.

// Same octahedral normal codec the cluster pool + resolve path use
// (`bevy_render::utils` parity), so deformed normals read back consistently
// with the static `vertex_normals`.
import octahedral;

// Per active animated instance. Mirrors `deform.rs::AnimatedSlotGpu` (36 B).
struct AnimatedSlot {
    uint instance_slot;      // GpuEntity slot of the skinned instance
    uint mesh_vertex_base;   // global vertex-pool slot of vertex 0 (rest pos/normal)
    uint mesh_vertex_count;  // vertices to deform
    uint deform_pool_base;   // per-slot base (in vertices) into the deform pool
    uint joint_count;
    uint palette_base;       // base into `palette` (node slot per joint)
    uint inverse_bind_base;  // base into `inverse_bind` (mat3x4 per joint)
    uint joint_base;         // global slot of vertex 0 in the joint streams
    // The instance's own transform-table node — its ABSOLUTE world is recomputed
    // here with the same local/parent walk the joints use, so the skin and the
    // inverse-instance transform live in the same (absolute) space. The gathered
    // per-instance transforms are ORIGIN-RELATIVE and would displace the deformed
    // geometry by the camera's world position.
    uint node_slot;
}

struct DeformParams {
    uint num_slots;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}

[[vk::push_constant]] DeformParams params;

// A `mat3x4` affine: `r[k]` is row k of the 4x4 (matches `Affine3x4` packing —
// 3 float4 rows, 48 B).
struct Affine3x4 {
    float4 r0;
    float4 r1;
    float4 r2;
}

[[vk::binding(0, 0)]] StructuredBuffer<AnimatedSlot> active_slots;
// Transform-table `local` column (f64 translation, 3 per node). We walk each
// joint's ancestor chain here to get its CURRENT world, rather than reading the
// propagated `world[]` buffer — that buffer only re-walks nodes whose own local
// changed, so a joint whose parent animates but whose own local is static (e.g.
// a toe bone) would be stale. `local` always holds every node's current local.
[[vk::binding(1, 0)]] StructuredBuffer<double> local_t;
// Per-joint inverse-bind poses (mat3x4), concatenated across active skeletons.
[[vk::binding(2, 0)]] StructuredBuffer<Affine3x4> inverse_bind;
// Per-joint node slot (skinning palette), concatenated across active skeletons.
[[vk::binding(3, 0)]] StructuredBuffer<uint> palette;
// Rest-pose positions (shared cluster pool, stride 3 f32) + octahedral normals.
[[vk::binding(4, 0)]] StructuredBuffer<float> rest_positions;
[[vk::binding(5, 0)]] StructuredBuffer<uint> rest_normals;
// Per-vertex joint influences: 4 indices packed as [u16;4] -> uint2, + weights.
[[vk::binding(6, 0)]] StructuredBuffer<uint2> joint_indices;
[[vk::binding(7, 0)]] StructuredBuffer<float4> joint_weights;
// Outputs: per-slot deformed positions (stride 3 f32) + octahedral normals.
[[vk::binding(8, 0)]] RWStructuredBuffer<float> deform_positions;
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> deform_normals;
// Transform-table `parent` column (node-slot per node) — for the ancestor walk.
[[vk::binding(10, 0)]] StructuredBuffer<uint> parent;
// Rest tangents (vec4: xyz + w bitangent sign) + deformed-tangent output.
[[vk::binding(11, 0)]] StructuredBuffer<float4> rest_tangents;
[[vk::binding(12, 0)]] RWStructuredBuffer<float4> deform_tangents;
// Transform-table `local_rs` column (quat + scale, 7 f32 per node).
[[vk::binding(13, 0)]] StructuredBuffer<float> local_rs;

static const uint ROOT_PARENT = 0xffffffff;
static const uint MAX_DEPTH = 64;

// Apply a `mat3x4` affine (row k of the 4x4 in `r[k]`) to a point / direction.
float3 affine_point(Affine3x4 m, float3 p) {
    return float3(
        dot(m.r0.xyz, p) + m.r0.w,
        dot(m.r1.xyz, p) + m.r1.w,
        dot(m.r2.xyz, p) + m.r2.w);
}
float3 affine_dir(Affine3x4 m, float3 v) {
    return float3(dot(m.r0.xyz, v), dot(m.r1.xyz, v), dot(m.r2.xyz, v));
}
// Transform a NORMAL by `m`'s linear part — the inverse-transpose (cofactor matrix),
// NOT `affine_dir`. Blending skin matrices shears/scales the linear part, and only
// the cofactor keeps normals perpendicular to the deformed surface. Caller
// normalizes. `cof(R) = R` for a rotation, so rigid single-joint verts are unchanged.
float3 affine_normal(Affine3x4 m, float3 n) {
    let c0 = float3(m.r0.x, m.r1.x, m.r2.x);
    let c1 = float3(m.r0.y, m.r1.y, m.r2.y);
    let c2 = float3(m.r0.z, m.r1.z, m.r2.z);
    return cross(c1, c2) * n.x + cross(c2, c0) * n.y + cross(c0, c1) * n.z;
}

// Invert a `mat3x4` affine (rigid+scale). Returns the inverse in the same packing.
Affine3x4 affine_inverse(Affine3x4 m) {
    let a = m.r0.xyz;
    let b = m.r1.xyz;
    let c = m.r2.xyz;
    let t = float3(m.r0.w, m.r1.w, m.r2.w);
    // Inverse of the 3x3 linear part (a,b,c are its rows; r0..r2 its columns).
    let r0 = float3(a.x, b.x, c.x);
    let r1 = float3(a.y, b.y, c.y);
    let r2 = float3(a.z, b.z, c.z);
    let det = dot(r0, cross(r1, r2));
    let inv_det = abs(det) > 1e-12 ? 1.0 / det : 0.0;
    // inv linear rows (adjugate / det)
    let i0 = cross(r1, r2) * inv_det;
    let i1 = cross(r2, r0) * inv_det;
    let i2 = cross(r0, r1) * inv_det;
    let inv_t = float3(-dot(i0, t), -dot(i1, t), -dot(i2, t));
    Affine3x4 r;
    r.r0 = float4(i0, inv_t.x);
    r.r1 = float4(i1, inv_t.y);
    r.r2 = float4(i2, inv_t.z);
    return r;
}

// Compose two `mat3x4` affines: result = a ∘ b (apply b, then a).
Affine3x4 affine_mul(Affine3x4 a, Affine3x4 b) {
    let bx = float3(b.r0.x, b.r1.x, b.r2.x);
    let by = float3(b.r0.y, b.r1.y, b.r2.y);
    let bz = float3(b.r0.z, b.r1.z, b.r2.z);
    let bt = float3(b.r0.w, b.r1.w, b.r2.w);
    Affine3x4 r;
    r.r0 = float4(dot(a.r0.xyz, bx), dot(a.r0.xyz, by), dot(a.r0.xyz, bz), dot(a.r0.xyz, bt) + a.r0.w);
    r.r1 = float4(dot(a.r1.xyz, bx), dot(a.r1.xyz, by), dot(a.r1.xyz, bz), dot(a.r1.xyz, bt) + a.r1.w);
    r.r2 = float4(dot(a.r2.xyz, bx), dot(a.r2.xyz, by), dot(a.r2.xyz, bz), dot(a.r2.xyz, bt) + a.r2.w);
    return r;
}

Affine3x4 scale_affine(float s, Affine3x4 m) {
    Affine3x4 r;
    r.r0 = s * m.r0;
    r.r1 = s * m.r1;
    r.r2 = s * m.r2;
    return r;
}
Affine3x4 add_affine(Affine3x4 a, Affine3x4 b) {
    Affine3x4 r;
    r.r0 = a.r0 + b.r0;
    r.r1 = a.r1 + b.r1;
    r.r2 = a.r2 + b.r2;
    return r;
}

// A node's local transform as a `mat3x4` affine (quat→3x3, scaled columns,
// translation in `.w`). Matches `transform_propagate.slang::load_local`.
Affine3x4 load_local_affine(uint node) {
    // The f64 local translation narrows to f32 here: the joint chain and the skinned
    // instance's world are composed in the SAME space, so the shared magnitude cancels
    // in their relative product (skinning at AU distances would need the f64 walk).
    let tb = node * 3;
    let t = float3(float(local_t[tb]), float(local_t[tb + 1]), float(local_t[tb + 2]));
    let b = node * 7;
    let qx = local_rs[b]; let qy = local_rs[b + 1]; let qz = local_rs[b + 2]; let qw = local_rs[b + 3];
    let s = float3(local_rs[b + 4], local_rs[b + 5], local_rs[b + 6]);
    let xx = qx * qx; let yy = qy * qy; let zz = qz * qz;
    let xy = qx * qy; let xz = qx * qz; let yz = qy * qz;
    let wx = qw * qx; let wy = qw * qy; let wz = qw * qz;
    let rot0 = float3(1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy));
    let rot1 = float3(2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx));
    let rot2 = float3(2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy));
    Affine3x4 m;
    m.r0 = float4(rot0 * s, t.x);
    m.r1 = float4(rot1 * s, t.y);
    m.r2 = float4(rot2 * s, t.z);
    return m;
}

// Fresh world transform of `node`: M = local[root] ∘ … ∘ local[parent] ∘ local[node].
// Walks the live `local`/`parent` columns, so a joint whose own local is static
// but whose parent animates gets the correct world (the propagated `world[]`
// buffer would be stale for it).
Affine3x4 joint_world(uint node) {
    var m = load_local_affine(node);
    var p = parent[node];
    for (uint step = 0; step < MAX_DEPTH; step++) {
        if (p == ROOT_PARENT) {
            break;
        }
        m = affine_mul(load_local_affine(p), m);
        p = parent[p];
    }
    return m;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void deform(uint3 gid: SV_DispatchThreadID) {
    let slot_idx = gid.y;
    if (slot_idx >= params.num_slots) {
        return;
    }
    let s = active_slots[slot_idx];
    let vid = gid.x;
    if (vid >= s.mesh_vertex_count) {
        return;
    }

    // Rest-pose position + normal at this mesh-local vertex.
    let pos_slot = s.mesh_vertex_base + vid;
    let rest_pos = float3(
        rest_positions[pos_slot * 3 + 0],
        rest_positions[pos_slot * 3 + 1],
        rest_positions[pos_slot * 3 + 2]);
    let rest_nrm = octahedral_decode_signed(unpack_snorm2x16(rest_normals[pos_slot]));

    // Per-vertex joint influences (joint stream may sit at a different pool base).
    let joint_slot = s.joint_base + vid;
    let packed = joint_indices[joint_slot];
    let js = uint4(
        packed.x & 0xffff,
        packed.x >> 16,
        packed.y & 0xffff,
        packed.y >> 16);
    let w = joint_weights[joint_slot];

    // Blend the 4 skin matrices: skin_k = joint_world(palette[jk]) * inverse_bind[jk].
    Affine3x4 m;
    m.r0 = float4(0.0);
    m.r1 = float4(0.0);
    m.r2 = float4(0.0);
    for (uint k = 0; k < 4; k++) {
        let weight = w[k];
        if (weight == 0.0) {
            continue;
        }
        let j = js[k];
        if (j >= s.joint_count) {
            continue;
        }
        let node = palette[s.palette_base + j];
        if (node == ROOT_PARENT) {
            continue; // unresolved joint slot — skip rather than read OOB
        }
        let skin = affine_mul(joint_world(node), inverse_bind[s.inverse_bind_base + j]);
        m = add_affine(m, scale_affine(weight, skin));
    }

    // Skin to (absolute) world, then pre-multiply inverse(instance world) -> mesh-local.
    // The instance world is walked from the SAME local columns as the joints, so the
    // two are in the same absolute space and the huge shared translation cancels.
    let pos_world = affine_point(m, rest_pos);
    let nrm_world = affine_normal(m, rest_nrm);
    let rest_tan = rest_tangents[pos_slot];
    let tan_world = affine_dir(m, rest_tan.xyz);
    let inv_instance = affine_inverse(joint_world(s.node_slot));
    let pos_local = affine_point(inv_instance, pos_world);
    let nrm_local = normalize(affine_normal(inv_instance, nrm_world));
    let tan_local = normalize(affine_dir(inv_instance, tan_world));

    let out_v = s.deform_pool_base + vid;
    deform_positions[out_v * 3 + 0] = pos_local.x;
    deform_positions[out_v * 3 + 1] = pos_local.y;
    deform_positions[out_v * 3 + 2] = pos_local.z;
    // octahedral_encode returns [0,1]; the pool + octahedral_decode_signed use [-1,1].
    deform_normals[out_v] = pack_snorm2x16(octahedral_encode(nrm_local) * 2.0 - 1.0);
    deform_tangents[out_v] = float4(tan_local, rest_tan.w); // w = bitangent sign (deform-invariant)
}
