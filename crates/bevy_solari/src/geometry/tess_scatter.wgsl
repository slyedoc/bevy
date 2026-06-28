// Phase C step 3a: lay out the flat per-part CLAS addresses as per-instance BLAS
// reference lists.
//
// `classify` assigns each base triangle a DETERMINISTIC part index
// (`part_base + triangle`), and work clusters are pushed grouped by instance, so the
// parts of instance `i` already occupy the CONTIGUOUS global range
// `[base_offsets[i], base_offsets[i] + count_i)`. Therefore `clas_addresses` is
// already grouped by instance and `references[p] = clas_addresses[p]` is the correct
// layout — instance `i`'s BLAS reads `references[base_offsets[i] ..]` unchanged.
//
// Deterministic ON PURPOSE: a per-frame-varying CLAS order inside a BLAS makes the
// traversal return a different first-hit each frame on coplanar/overlapping displaced
// micro-geometry → the hit flickers even with a still camera. A fixed order is stable.

// One CLAS device address (u64 as lo,hi) per emitted part, written by INSTANTIATE.
@group(0) @binding(0) var<storage, read> clas_addresses: array<vec2<u32>>;
// Output: per-instance contiguous CLAS-address array (cluster_references for the BLAS).
@group(0) @binding(1) var<storage, read_write> references: array<vec2<u32>>;
// counts[0] = emitted part count.
@group(0) @binding(2) var<storage, read> counts: array<u32>;

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= counts[0] {
        return;
    }
    references[p] = clas_addresses[p];
}
