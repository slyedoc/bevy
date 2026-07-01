// Skeletal deform — linear-blend skinning of animated cluster meshes on the GPU.
//
// One thread per (active animated slot, vertex). For each vertex it blends the
// 4 influencing joints' skin matrices (`world[palette[j]] * inverse_bind[j]`,
// both `mat3x4` affines), transforms the rest-pose position/normal, then
// pre-multiplies the inverse of the instance's own world transform so the
// output pool holds MESH-LOCAL skinned positions — keeping the resolve path's
// `transforms[instance_id]` application identical to static instances.
//
// Inputs are read straight from the GPU transform table (`world[]`, produced by
// the propagate pass — joints are nodes) and the shared cluster pools; no CPU
// skin-matrix upload. See `crates/bevy_solari/cluster_animation_plan.md`.

// Same octahedral normal codec the cluster pool + resolve path use, so deformed
// normals read back consistently with the static `vertex_normals`.
#import bevy_render::utils::{octahedral_encode, octahedral_decode_signed}

// Per active animated instance. Mirrors `deform.rs::AnimatedSlotGpu` (32 B).
struct AnimatedSlot {
    instance_slot: u32,      // GpuEntity slot — indexes instance_transforms
    mesh_vertex_base: u32,   // global vertex-pool slot of vertex 0 (rest pos/normal)
    mesh_vertex_count: u32,  // vertices to deform
    deform_pool_base: u32,   // per-slot base (in vertices) into the deform pool
    joint_count: u32,
    palette_base: u32,       // base into `palette` (node slot per joint)
    inverse_bind_base: u32,  // base into `inverse_bind` (mat3x4 per joint)
    joint_base: u32,         // global slot of vertex 0 in the joint streams
}

struct DeformParams {
    num_slots: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> active_slots: array<AnimatedSlot>;
// Transform-table `local` column (raw TRS, 10 f32 / node). We walk each joint's
// ancestor chain here to get its CURRENT world, rather than reading the
// propagated `world[]` buffer — that buffer only re-walks nodes whose own local
// changed, so a joint whose parent animates but whose own local is static (e.g.
// a toe bone) would be stale. `local` always holds every node's current local.
@group(0) @binding(1) var<storage, read> local_t: array<f64>;
// Per-joint inverse-bind poses (mat3x4), concatenated across active skeletons.
@group(0) @binding(2) var<storage, read> inverse_bind: array<mat3x4<f32>>;
// Per-joint node slot (skinning palette), concatenated across active skeletons.
@group(0) @binding(3) var<storage, read> palette: array<u32>;
// Rest-pose positions (shared cluster pool, stride 3 f32) + octahedral normals.
@group(0) @binding(4) var<storage, read> rest_positions: array<f32>;
@group(0) @binding(5) var<storage, read> rest_normals: array<u32>;
// Per-vertex joint influences: 4 indices packed as [u16;4] -> vec2<u32>, + weights.
@group(0) @binding(6) var<storage, read> joint_indices: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> joint_weights: array<vec4<f32>>;
// Per-instance world transforms (the gather output the resolve path also reads).
@group(0) @binding(8) var<storage, read> instance_transforms: array<mat3x4<f32>>;
@group(0) @binding(9) var<uniform> params: DeformParams;
// Outputs: per-slot deformed positions (stride 3 f32) + octahedral normals.
@group(0) @binding(10) var<storage, read_write> deform_positions: array<f32>;
@group(0) @binding(11) var<storage, read_write> deform_normals: array<u32>;
// Transform-table `parent` column (node-slot per node) — for the ancestor walk.
@group(0) @binding(12) var<storage, read> parent: array<u32>;
// Rest tangents (vec4: xyz + w bitangent sign) + deformed-tangent output.
@group(0) @binding(13) var<storage, read> rest_tangents: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read_write> deform_tangents: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read> local_rs: array<f32>;

const ROOT_PARENT: u32 = 0xffffffffu;
const MAX_DEPTH: u32 = 64u;

// Apply a `mat3x4` affine (column k = 4x4 row k) to a point / direction.
fn affine_point(m: mat3x4<f32>, p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(m[0].xyz, p) + m[0].w,
        dot(m[1].xyz, p) + m[1].w,
        dot(m[2].xyz, p) + m[2].w,
    );
}
fn affine_dir(m: mat3x4<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(m[0].xyz, v), dot(m[1].xyz, v), dot(m[2].xyz, v));
}

// Invert a `mat3x4` affine (rigid+scale). Returns the inverse in the same packing.
fn affine_inverse(m: mat3x4<f32>) -> mat3x4<f32> {
    let a = m[0].xyz;
    let b = m[1].xyz;
    let c = m[2].xyz;
    let t = vec3<f32>(m[0].w, m[1].w, m[2].w);
    // Inverse of the 3x3 linear part (columns a,b,c are its ROWS here).
    let r0 = vec3<f32>(a.x, b.x, c.x);
    let r1 = vec3<f32>(a.y, b.y, c.y);
    let r2 = vec3<f32>(a.z, b.z, c.z);
    let det = dot(r0, cross(r1, r2));
    let inv_det = select(0.0, 1.0 / det, abs(det) > 1e-12);
    // inv linear rows (adjugate / det)
    let i0 = cross(r1, r2) * inv_det;
    let i1 = cross(r2, r0) * inv_det;
    let i2 = cross(r0, r1) * inv_det;
    let inv_t = vec3<f32>(-dot(i0, t), -dot(i1, t), -dot(i2, t));
    return mat3x4<f32>(
        vec4<f32>(i0, inv_t.x),
        vec4<f32>(i1, inv_t.y),
        vec4<f32>(i2, inv_t.z),
    );
}

// Compose two `mat3x4` affines: result = a ∘ b (apply b, then a).
fn affine_mul(a: mat3x4<f32>, b: mat3x4<f32>) -> mat3x4<f32> {
    let bx = vec3<f32>(b[0].x, b[1].x, b[2].x);
    let by = vec3<f32>(b[0].y, b[1].y, b[2].y);
    let bz = vec3<f32>(b[0].z, b[1].z, b[2].z);
    let bt = vec3<f32>(b[0].w, b[1].w, b[2].w);
    return mat3x4<f32>(
        vec4<f32>(dot(a[0].xyz, bx), dot(a[0].xyz, by), dot(a[0].xyz, bz), dot(a[0].xyz, bt) + a[0].w),
        vec4<f32>(dot(a[1].xyz, bx), dot(a[1].xyz, by), dot(a[1].xyz, bz), dot(a[1].xyz, bt) + a[1].w),
        vec4<f32>(dot(a[2].xyz, bx), dot(a[2].xyz, by), dot(a[2].xyz, bz), dot(a[2].xyz, bt) + a[2].w),
    );
}

fn scale_affine(s: f32, m: mat3x4<f32>) -> mat3x4<f32> {
    return mat3x4<f32>(s * m[0], s * m[1], s * m[2]);
}
fn add_affine(a: mat3x4<f32>, b: mat3x4<f32>) -> mat3x4<f32> {
    return mat3x4<f32>(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

// A node's local transform as a `mat3x4` affine (quat→3x3, scaled columns,
// translation in `.w`). Matches `transform_propagate.wgsl::load_local`.
fn load_local_affine(node: u32) -> mat3x4<f32> {
    // The f64 local translation narrows to f32 here: the joint chain and the skinned
    // instance's world are composed in the SAME space, so the shared magnitude cancels
    // in their relative product (skinning at AU distances would need the f64 walk).
    let tb = node * 3u;
    let t = vec3<f32>(f32(local_t[tb]), f32(local_t[tb + 1u]), f32(local_t[tb + 2u]));
    let b = node * 7u;
    let qx = local_rs[b]; let qy = local_rs[b + 1u]; let qz = local_rs[b + 2u]; let qw = local_rs[b + 3u];
    let s = vec3<f32>(local_rs[b + 4u], local_rs[b + 5u], local_rs[b + 6u]);
    let xx = qx * qx; let yy = qy * qy; let zz = qz * qz;
    let xy = qx * qy; let xz = qx * qz; let yz = qy * qz;
    let wx = qw * qx; let wy = qw * qy; let wz = qw * qz;
    let rot0 = vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy));
    let rot1 = vec3<f32>(2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx));
    let rot2 = vec3<f32>(2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy));
    return mat3x4<f32>(
        vec4<f32>(rot0 * s, t.x),
        vec4<f32>(rot1 * s, t.y),
        vec4<f32>(rot2 * s, t.z),
    );
}

// Fresh world transform of `node`: M = local[root] ∘ … ∘ local[parent] ∘ local[node].
// Walks the live `local`/`parent` columns, so a joint whose own local is static
// but whose parent animates gets the correct world (the propagated `world[]`
// buffer would be stale for it).
fn joint_world(node: u32) -> mat3x4<f32> {
    var m = load_local_affine(node);
    var p = parent[node];
    for (var step = 0u; step < MAX_DEPTH; step = step + 1u) {
        if p == ROOT_PARENT {
            break;
        }
        m = affine_mul(load_local_affine(p), m);
        p = parent[p];
    }
    return m;
}

@compute @workgroup_size(64, 1, 1)
fn deform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot_idx = gid.y;
    if slot_idx >= params.num_slots {
        return;
    }
    let s = active_slots[slot_idx];
    let vid = gid.x;
    if vid >= s.mesh_vertex_count {
        return;
    }

    // Rest-pose position + normal at this mesh-local vertex.
    let pos_slot = s.mesh_vertex_base + vid;
    let rest_pos = vec3<f32>(
        rest_positions[pos_slot * 3u + 0u],
        rest_positions[pos_slot * 3u + 1u],
        rest_positions[pos_slot * 3u + 2u],
    );
    let rest_nrm = octahedral_decode_signed(unpack2x16snorm(rest_normals[pos_slot]));

    // Per-vertex joint influences (joint stream may sit at a different pool base).
    let joint_slot = s.joint_base + vid;
    let packed = joint_indices[joint_slot];
    let j0 = packed.x & 0xFFFFu;
    let j1 = packed.x >> 16u;
    let j2 = packed.y & 0xFFFFu;
    let j3 = packed.y >> 16u;
    let w = joint_weights[joint_slot];

    // Blend the 4 skin matrices: skin_k = joint_world(palette[jk]) * inverse_bind[jk].
    var m = mat3x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)); // zero affine
    let js = array<u32, 4>(j0, j1, j2, j3);
    let ws = array<f32, 4>(w.x, w.y, w.z, w.w);
    for (var k = 0u; k < 4u; k++) {
        let weight = ws[k];
        if weight == 0.0 {
            continue;
        }
        let j = js[k];
        if j >= s.joint_count {
            continue;
        }
        let node = palette[s.palette_base + j];
        if node == ROOT_PARENT {
            continue; // unresolved joint slot — skip rather than read OOB
        }
        let skin = affine_mul(joint_world(node), inverse_bind[s.inverse_bind_base + j]);
        m = add_affine(m, scale_affine(weight, skin));
    }

    // Skin to world, then pre-multiply inverse(instance world) -> mesh-local.
    let pos_world = affine_point(m, rest_pos);
    let nrm_world = affine_dir(m, rest_nrm);
    let rest_tan = rest_tangents[pos_slot];
    let tan_world = affine_dir(m, rest_tan.xyz);
    let inv_instance = affine_inverse(instance_transforms[s.instance_slot]);
    let pos_local = affine_point(inv_instance, pos_world);
    let nrm_local = normalize(affine_dir(inv_instance, nrm_world));
    let tan_local = normalize(affine_dir(inv_instance, tan_world));

    let out_v = s.deform_pool_base + vid;
    deform_positions[out_v * 3u + 0u] = pos_local.x;
    deform_positions[out_v * 3u + 1u] = pos_local.y;
    deform_positions[out_v * 3u + 2u] = pos_local.z;
    deform_normals[out_v] = pack2x16snorm(octahedral_encode(nrm_local));
    deform_tangents[out_v] = vec4<f32>(tan_local, rest_tan.w); // w = bitangent sign (deform-invariant)
}
