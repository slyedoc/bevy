//! `Mesh → ClusterMesh` bake. Calls `meshopt` directly to build
//! clusters + stacked LOD levels.
//!
//! **v1 strategy: stacked discrete LOD levels.** Each level is built
//! bottom-up by clustering + partitioning the current triangle
//! stream, then simplifying each partition under boundary-locked
//! simplification to produce the next coarser stream.
//! [`ClusterLodGroup`]s at level N carry the simplification error
//! metric so the runtime selector can pick the coarsest level whose
//! error projects below the on-screen pixel threshold.
//!
//! Per-region LOD variation (proper Nanite-style DAG cut where
//! adjacent regions of one instance use different LODs) needs
//! per-triangle source-group provenance threaded through meshopt's
//! reorderings. The asset format already carries
//! [`ClusterLodGroup::children_offset`] / `children_count` for future
//! DAG-cut consumers; v1 leaves them at 0 (treat each group as a
//! leaf) and the selector evaluates groups by LOD level + error
//! against the camera.

use alloc::sync::Arc;
use bevy_math::{Vec2, Vec3, Vec3Swizzles, Vec4};
use bevy_mesh::{Mesh, MeshVertexAttributeId, VertexAttributeValues};
use bevy_platform::collections::HashMap;
use meshopt::{
    build_meshlets_spatial, generate_position_remap, partition_clusters, simplify_with_locks,
    SimplifyOptions, VertexDataAdapter,
};
use thiserror::Error;
use tracing::debug_span;

use super::asset::{Cluster, ClusterLodGroup, ClusterMesh, ClusterMeshAabb};

/// Max vertices per cluster. 128 is comfortably below NV's per-CLAS
/// cap (256) while letting meshopt produce dense clusters.
pub const MAX_CLUSTER_VERTICES: usize = 128;
/// Max triangles per cluster. Must be divisible by 4 per meshopt;
/// 124 is the largest valid value ≤ 128.
pub const MAX_CLUSTER_TRIANGLES: usize = 124;
/// Min triangles per cluster for `build_meshlets_spatial` (divisible by 4). A floor
/// that keeps spatial clusters from collapsing into tiny CLASes; `SPATIAL_FILL_WEIGHT`
/// trades cluster fullness against locality above it.
pub const MIN_CLUSTER_TRIANGLES: usize = 60;
/// Spatial-meshlet fill weight: `0.0` = purest spatial locality (tightest cluster
/// bounds, more clusters), `1.0` = fullest clusters. `0.5` balances tight bounds (faster
/// RT/CLAS traversal) against cluster count.
pub const SPATIAL_FILL_WEIGHT: f32 = 0.5;
/// Target clusters per group at partition time. Balances
/// simplification quality against group count.
pub const TARGET_GROUP_SIZE: usize = 8;
/// Per-LOD-level simplification ratio. Half the triangles each level.
pub const SIMPLIFY_TARGET_FRACTION: f32 = 0.5;
/// Monotone-enlarges parent error so DAG cuts can't pick a parent
/// that's "smaller" than a child.
pub const MERGE_PREV_FACTOR: f32 = 1.1;
/// Slight additive bump on top of the multiplier to handle
/// equal-error edge cases.
pub const MERGE_ADDITIVE_FACTOR: f32 = 0.05;
/// Cap LOD recursion depth — safety against pathological meshes.
pub const MAX_LOD_LEVELS: u32 = 12;

// Process a [`Mesh`] into a [`ClusterMesh`]. Very slow — meant to run
// offline (gated on the `cluster_processor` feature), not at runtime.
// The input mesh must use `PrimitiveTopology::TriangleList` with indices
// and carry `{POSITION, NORMAL, UV_0}`; `TANGENT` is generated via
// mikktspace on a local clone when missing (the input is not mutated).
impl TryFrom<&Mesh> for ClusterMesh {
    type Error = MeshToClusterMeshConversionError;

    fn try_from(mesh: &Mesh) -> Result<Self, Self::Error> {
        
         let s = debug_span!("build cluster mesh");
        let _e = s.enter();

        // Auto-generate tangents on a local clone if missing — keeps
        // the input mesh untouched while satisfying the bake's
        // tangent requirement.
        let owned_with_tangents;
        let mesh = if mesh.attribute(Mesh::ATTRIBUTE_TANGENT.id).is_none() {
            owned_with_tangents = mesh.clone().with_generated_tangents()?;
            &owned_with_tangents
        } else {
            mesh
        };

        let mesh_positions = extract_vec3(mesh, Mesh::ATTRIBUTE_POSITION.id, "POSITION")?;
        let mesh_normals = extract_vec3(mesh, Mesh::ATTRIBUTE_NORMAL.id, "NORMAL")?;
        let mesh_tangents = extract_vec4(mesh, Mesh::ATTRIBUTE_TANGENT.id, "TANGENT")?;
        let mesh_uvs = extract_vec2(mesh, Mesh::ATTRIBUTE_UV_0.id, "UV_0")?;
        // Non-indexed triangle lists (e.g. the glTF fox) get a trivial sequential
        // index buffer; the position remap below re-shares coincident positions
        // so the DAG cut / simplifier still work.
        let mesh_indices: Vec<u32> = match mesh.indices() {
            Some(indices) => indices.iter().map(|i| i as u32).collect(),
            None => (0..mesh_positions.len() as u32).collect(),
        };

        // meshopt VertexDataAdapter wants the position buffer as a `&[u8]`
        // with explicit stride. Vec3 is `#[repr(C)]` with 12 B size, no
        // padding — bytemuck slice cast is direct.
        let pos_bytes = bytemuck::cast_slice::<Vec3, u8>(&mesh_positions);
        let vert_adapter = VertexDataAdapter::new(pos_bytes, 12, 0)
            .map_err(|_| MeshToClusterMeshConversionError::VertexAdapter)?;

        // Position-only canonical vertex ids. Two distinct vertex
        // indices that share a position (UV / normal "wedges" at
        // attribute discontinuities) get the same canonical id here.
        // Required for crack-free DAG cuts: the boundary lock must be
        // decided per *position*, not per *vertex index*. If we lock
        // by raw index, adjacent partitions that use different wedges
        // for the same boundary position fail to share a lock — the
        // simplifier moves one wedge while the other stays put,
        // tearing the surface open at LOD transitions.
        let position_remap = generate_position_remap(&vert_adapter);

        let aabb = mesh_aabb(&mesh_positions);

        let mut out_positions: Vec<Vec3> = Vec::with_capacity(mesh_positions.len() * 2);
        let mut out_normals: Vec<u32> = Vec::with_capacity(mesh_normals.len() * 2);
        let mut out_tangents: Vec<Vec4> = Vec::with_capacity(mesh_tangents.len() * 2);
        let mut out_uvs: Vec<Vec2> = Vec::with_capacity(mesh_uvs.len() * 2);
        let mut out_indices: Vec<u32> = Vec::with_capacity(mesh_indices.len() * 2);
        let mut out_clusters: Vec<Cluster> = Vec::new();
        let mut out_groups: Vec<ClusterLodGroup> = Vec::new();
        let mut out_cluster_to_group: Vec<u32> = Vec::new();

        let mut current_indices: Vec<u32> = mesh_indices.clone();
        let mut prev_level_max_error: f32 = 0.0;
        let mut lod_level: u32 = 0;
        let mut root_group_id: u32 = u32::MAX;
        let mut mesh_max_error: f32 = 0.0;

        // Parent-linkage state, rotated each iteration:
        // - `vertex_origin_groups`: which previous-level groups
        //   contributed each vertex into `current_indices`. Empty at
        //   level 0 (no parents to assign yet).
        // - `prev_level_groups`: group ids from the previous
        //   iteration. We set their `parent_group` once we know which
        //   group at the current level they fed into.
        let mut vertex_origin_groups: HashMap<u32, Vec<u32>> = HashMap::default();
        let mut prev_level_groups: Vec<u32> = Vec::new();

        loop {
            if current_indices.len() < 3 {
                break;
            }

            // 1. Cluster the current triangle stream SPATIALLY: tight cluster bounding
            //    volumes speed up ray-tracing CLAS/BLAS traversal. `build_meshlets`
            //    instead optimizes vertex-cache locality — a rasterization metric the
            //    RT path doesn't benefit from. `SPATIAL_FILL_WEIGHT` balances bounds
            //    tightness against cluster count.
            let meshlets = build_meshlets_spatial(
                &current_indices,
                &vert_adapter,
                MAX_CLUSTER_VERTICES,
                MIN_CLUSTER_TRIANGLES,
                MAX_CLUSTER_TRIANGLES,
                SPATIAL_FILL_WEIGHT,
            );
            if meshlets.len() == 0 {
                break;
            }

            // 2. Partition clusters into groups (simplification + runtime
            //    DAG-cut units).
            let mut cluster_indices_flat: Vec<u32> = Vec::new();
            let mut cluster_index_counts: Vec<u32> = Vec::new();
            for mlet in meshlets.iter() {
                for &local_idx in mlet.triangles {
                    cluster_indices_flat.push(mlet.vertices[local_idx as usize]);
                }
                cluster_index_counts.push(mlet.triangles.len() as u32);
            }
            let mut partition_dest = vec![0u32; meshlets.len()];
            let partition_count = if meshlets.len() == 1 {
                partition_dest[0] = 0;
                1
            } else {
                partition_clusters(
                    &mut partition_dest,
                    &cluster_indices_flat,
                    &cluster_index_counts,
                    mesh_positions.len(),
                    TARGET_GROUP_SIZE,
                )
            };

            // 3. Bucket meshlet ids by partition.
            let mut partition_meshlets: Vec<Vec<usize>> = vec![Vec::new(); partition_count];
            for (mlet_id, &part_id) in partition_dest.iter().enumerate() {
                partition_meshlets[part_id as usize].push(mlet_id);
            }

            // 4. Parent linkage: each level-(L-1) group fed its
            //    simplified output into `current_indices`; we tracked
            //    per-vertex origin groups via `vertex_origin_groups`.
            //    Map previous-level groups to this level's partitions
            //    by majority vertex vote, then back-fill `parent_group`
            //    on those previous-level entries. Skipped at level 0
            //    (no previous level to link).
            //
            //    Group ids assigned to non-empty partitions in
            //    iteration order; predict here so we can write parents
            //    without an extra pass.
            let level_group_start = out_groups.len() as u32;
            let mut partition_to_group_id: Vec<Option<u32>> =
                vec![None; partition_meshlets.len()];
            {
                let mut running = level_group_start;
                for (part_idx, mlet_ids) in partition_meshlets.iter().enumerate() {
                    if !mlet_ids.is_empty() {
                        partition_to_group_id[part_idx] = Some(running);
                        running += 1;
                    }
                }
            }
            if !prev_level_groups.is_empty() {
                // `votes[prev_group][partition_idx] = vertex_count`
                let mut votes: HashMap<u32, HashMap<u32, u32>> = HashMap::default();
                for (part_idx, mlet_ids) in partition_meshlets.iter().enumerate() {
                    if mlet_ids.is_empty() {
                        continue;
                    }
                    for &mlet_id in mlet_ids {
                        let mlet = meshlets.get(mlet_id);
                        for &vert in mlet.vertices {
                            if let Some(origins) = vertex_origin_groups.get(&vert) {
                                for &prev_group in origins {
                                    *votes
                                        .entry(prev_group)
                                        .or_default()
                                        .entry(part_idx as u32)
                                        .or_insert(0) += 1;
                                }
                            }
                        }
                    }
                }
                for (prev_group, partition_votes) in votes {
                    if let Some((&best_part, _)) = partition_votes
                        .iter()
                        .max_by_key(|(_, count)| **count)
                    {
                        if let Some(parent_id) =
                            partition_to_group_id[best_part as usize]
                        {
                            out_groups[prev_group as usize].parent_group = parent_id;
                        }
                    }
                }
            }

            // 5. Emit clusters + groups for this LOD level. Each group's
            //    clusters live contiguously in `out_clusters` (selector
            //    emits them as a `[start, start+count)` range); iterating
            //    partitions in order satisfies this.
            let mut this_level_groups: Vec<u32> = Vec::with_capacity(partition_count);
            let mut this_level_max_error: f32 = prev_level_max_error;
            let mut next_indices: Vec<u32> = Vec::new();
            // Vertex origins emitted THIS iteration — feeds parent
            // linkage at the next iteration. Rotates into
            // `vertex_origin_groups` at loop bottom.
            let mut next_vertex_origin_groups: HashMap<u32, Vec<u32>> = HashMap::default();

            for (partition_idx, mlet_ids) in partition_meshlets.iter().enumerate() {
                if mlet_ids.is_empty() {
                    continue;
                }
                let group_id = out_groups.len() as u32;
                let cluster_start = out_clusters.len() as u32;

                for &mlet_id in mlet_ids {
                    let mlet = meshlets.get(mlet_id);
                    let vertex_offset = out_positions.len() as u32;
                    let vertex_count = mlet.vertices.len() as u32;
                    let index_offset = out_indices.len() as u32;
                    let triangle_count = (mlet.triangles.len() / 3) as u32;

                    for &global_v in mlet.vertices {
                        let v = global_v as usize;
                        out_positions.push(mesh_positions[v]);
                        out_normals.push(pack2x16snorm(octahedral_encode(mesh_normals[v])));
                        out_tangents.push(mesh_tangents[v]);
                        out_uvs.push(mesh_uvs[v]);
                    }

                    // NV CLAS expects 32-bit indices; widen from meshopt's u8 locals.
                    for &local_idx in mlet.triangles {
                        out_indices.push(local_idx as u32);
                    }

                    let cluster_positions =
                        mlet.vertices.iter().map(|&v| mesh_positions[v as usize]);
                    let (center, radius) = bounding_sphere(cluster_positions);

                    out_clusters.push(Cluster {
                        vertex_offset,
                        vertex_count,
                        index_offset,
                        triangle_count,
                        bounds_sphere: [center.x, center.y, center.z, radius],
                        local_material_id: 0,
                        lod_level,
                        _pad: [0; 2],
                    });
                    out_cluster_to_group.push(group_id);
                }
                let cluster_end = out_clusters.len() as u32;

                // Simplify this group's geometry. Boundary-locked: vertices
                // shared with adjacent partitions stay put, so the LOD
                // transition doesn't introduce cracks. The simplified
                // output both seeds the next LOD level's triangle stream
                // and supplies the error metric for this group.
                let group_global_indices: Vec<u32> = mlet_ids
                    .iter()
                    .flat_map(|&mid| {
                        let mlet = meshlets.get(mid);
                        mlet.triangles
                            .iter()
                            .map(move |&l| mlet.vertices[l as usize])
                    })
                    .collect();
                let target_count =
                    ((group_global_indices.len() as f32) * SIMPLIFY_TARGET_FRACTION) as usize;
                let vertex_lock = compute_group_vertex_lock(
                    &cluster_index_counts,
                    mesh_positions.len(),
                    partition_idx as u32,
                    &partition_dest,
                    &cluster_indices_flat,
                    &position_remap,
                );
                let mut group_error_result: f32 = 0.0;
                let simplified = simplify_with_locks(
                    &group_global_indices,
                    &vert_adapter,
                    &vertex_lock,
                    target_count,
                    f32::MAX,
                    // `ErrorAbsolute`: meshopt returns error in
                    // absolute mesh-space units (multiplied by mesh
                    // extent), so per-level accumulation lines up
                    // across partitions of different sizes.
                    //
                    // NOTE: we intentionally do NOT pass `Sparse`.
                    // Sparse mode normalizes error by the SUBSET
                    // extent rather than the full mesh extent — that
                    // makes small partitions return tiny errors that
                    // never project past a 1-pixel threshold, so the
                    // selector picks the coarsest LOD everywhere.
                    // Locks still pin boundary verts without Sparse;
                    // we only lose a small simplifier performance win.
                    SimplifyOptions::ErrorAbsolute,
                    Some(&mut group_error_result),
                );

                // Nanite-style group error semantics: each group's
                // `max_quadric_error` is the REPRESENTATION error of
                // THIS LOD level — the cumulative deviation introduced
                // by every simplification step from the original mesh
                // down to this level. Lod 0 (original mesh) therefore
                // has error 0; lod_(k+1) has at least the error of the
                // simplification we're about to perform plus whatever
                // earlier simplifications cost.
                //
                // We do NOT fold the *current* simplification's
                // `group_error_result` into this group's error
                // (that's the error of going from THIS level to the
                // NEXT). Instead it gets accumulated below into
                // `this_level_max_error`, which becomes the next
                // iteration's `prev_level_max_error`.
                let group_error = prev_level_max_error;

                // Contribution this group's simplification adds to the
                // next-level's representation error. Monotone parent
                // ≥ child guaranteed because we take a max over
                // prev_level_max_error * MERGE_PREV_FACTOR and the new
                // simplification error, plus a small additive bump
                // for tie-breaking.
                let next_level_contribution = (prev_level_max_error * MERGE_PREV_FACTOR)
                    .max(group_error_result)
                    + group_error_result * MERGE_ADDITIVE_FACTOR;

                let group_positions = (cluster_start..cluster_end).flat_map(|cid| {
                    let c = &out_clusters[cid as usize];
                    (c.vertex_offset..c.vertex_offset + c.vertex_count)
                        .map(|v| out_positions[v as usize])
                });
                let (g_center, g_radius) = bounding_sphere(group_positions);

                out_groups.push(ClusterLodGroup {
                    cluster_start,
                    cluster_count: cluster_end - cluster_start,
                    children_offset: 0,
                    children_count: 0,
                    traversal_sphere: [g_center.x, g_center.y, g_center.z, g_radius],
                    max_quadric_error: group_error,
                    parent_group: u32::MAX,
                    lod_level,
                    _pad: 0,
                });
                this_level_groups.push(group_id);
                this_level_max_error = this_level_max_error.max(next_level_contribution);

                // Track which vertices this group contributed to the
                // NEXT level's input stream — vote target for parent
                // linkage one iteration later.
                for &v in simplified.iter() {
                    next_vertex_origin_groups
                        .entry(v)
                        .or_default()
                        .push(group_id);
                }
                next_indices.extend_from_slice(&simplified);
            }

            // Termination: single group at this level = root.
            if this_level_groups.len() == 1 {
                root_group_id = this_level_groups[0];
                mesh_max_error = this_level_max_error;
                break;
            }

            if lod_level + 1 >= MAX_LOD_LEVELS {
                // Recursion cap: pick the largest-error group as root.
                // Selector treats every group as a leaf cut candidate so
                // the root choice is just an "always-emit" fallback.
                root_group_id = *this_level_groups
                    .iter()
                    .max_by(|&&a, &&b| {
                        out_groups[a as usize]
                            .max_quadric_error
                            .partial_cmp(&out_groups[b as usize].max_quadric_error)
                            .unwrap_or(core::cmp::Ordering::Equal)
                    })
                    .expect("non-empty group list");
                mesh_max_error = this_level_max_error;
                break;
            }

            // Sanity: stop if simplification couldn't reduce (avoids
            // infinite loop on geometry that can't simplify further).
            if next_indices.len() >= current_indices.len() {
                root_group_id = this_level_groups[0];
                mesh_max_error = this_level_max_error;
                break;
            }

            current_indices = next_indices;
            vertex_origin_groups = next_vertex_origin_groups;
            prev_level_groups = this_level_groups;
            prev_level_max_error = this_level_max_error;
            lod_level += 1;
        }

        if root_group_id == u32::MAX {
            // Empty mesh — degenerate. Emit a single empty group so the
            // asset is loadable; runtime treats `cluster_count == 0` as
            // a no-op instance.
            out_groups.push(ClusterLodGroup::default());
            root_group_id = 0;
        }

        // Orphan repair: any non-root group whose vertices got
        // entirely simplified out (no votes propagated forward in
        // the per-level loop) ends up with `parent_group == MAX`.
        // We need to plug them back into the DAG so the runtime
        // cut is single-valued — leaving them at MAX makes the
        // selector treat them as roots (overlap), and suppressing
        // them via `error = +inf` leaves holes for their spatial
        // region (their nominal-parent's other children don't
        // cover the orphan's surface area). Link each orphan to
        // the level-(N+1) group whose `traversal_sphere` is
        // closest to the orphan's — geometrically that group
        // contains the orphan's region, so the cut at that subtree
        // descends through the orphan rather than skipping it.
        let mut orphan_count: usize = 0;
        let total_groups = out_groups.len();
        let snapshot: Vec<(u32, [f32; 4])> = out_groups
            .iter()
            .map(|g| (g.lod_level, g.traversal_sphere))
            .collect();
        for (gid, g) in out_groups.iter_mut().enumerate() {
            if gid as u32 == root_group_id || g.parent_group != u32::MAX {
                continue;
            }
            let target_level = g.lod_level + 1;
            let oc = g.traversal_sphere;
            let mut best: Option<(u32, f32)> = None;
            for (cand_id, (lvl, sph)) in snapshot.iter().enumerate() {
                if *lvl != target_level {
                    continue;
                }
                let dx = sph[0] - oc[0];
                let dy = sph[1] - oc[1];
                let dz = sph[2] - oc[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if best.map_or(true, |(_, bd)| d2 < bd) {
                    best = Some((cand_id as u32, d2));
                }
            }
            // Link to the closest coarser-level (level N+1) group if one exists.
            // If none does, this group has no coarser level above it — a single-LOD
            // mesh, or a coarsest-level group — so it's a genuine top-level group:
            // leave `parent_group == NONE`. Linking it to a SAME-level group (the
            // old `unwrap_or(root_group_id)`) makes the runtime DAG cut treat that
            // same-level group as a coarser parent and cull this one, leaving holes
            // (this is what broke single-LOD meshes like the glTF fox).
            if let Some((id, _)) = best {
                g.parent_group = id;
                orphan_count += 1;
            }
        }
        if orphan_count > 0 {
            tracing::debug!(
                "cluster bake: re-linked {orphan_count} orphan group(s) \
                 by geometric proximity (out of {total_groups} total groups)"
            );
        }

        // DAG-cut invariant: every non-root group's parent has
        // error >= this group's error. Construction guarantees this
        // because `group_error = max(prev_level_max_error *
        // MERGE_PREV_FACTOR, group_error_result) + ...` with
        // `MERGE_PREV_FACTOR > 1.0` — any level-L+1 group's error is
        // strictly greater than any level-L group's, regardless of
        // the spatial parent picked by the vertex-vote linkage or
        // the geometric-proximity repair above (geometric repair
        // also picks a level-L+1 parent, so the strict-greater
        // property still holds).
        debug_assert!(out_groups.iter().enumerate().all(|(gid, g)| {
            // Skip the root and any top-level group with no coarser parent (NONE).
            if gid as u32 == root_group_id || g.parent_group == u32::MAX {
                true
            } else {
                out_groups[g.parent_group as usize].max_quadric_error >= g.max_quadric_error
            }
        }));

        Ok(ClusterMesh {
            vertex_positions: out_positions.into(),
            vertex_normals: out_normals.into(),
            vertex_tangents: out_tangents.into(),
            vertex_uvs: out_uvs.into(),
            vertex_custom: Arc::from(&[][..]),
            indices: out_indices.into(),
            clusters: out_clusters.into(),
            groups: out_groups.into(),
            nodes: Arc::from(&[][..]),
            child_table: Arc::from(&[][..]),
            cluster_to_group: out_cluster_to_group.into(),
            aabb,
            mesh_max_error,
            root_group_id,
            root_node_id: u32::MAX,
            lod_levels: lod_level + 1,
            // OMM is baked separately (offline, from the alpha texture) and
            // attached via `ClusterMesh::set_opacity_micromap`.
            omm_array_data: Arc::from(&[][..]),
            omm_descs: Arc::from(&[][..]),
            omm_index: Arc::from(&[][..]),
            omm_usage: Arc::from(&[][..]),
            omm_index_usage: Arc::from(&[][..]),
        })

    }
}

#[derive(Error, Debug)]
pub enum MeshToClusterMeshConversionError {
    #[error("mesh missing required attribute: {0}")]
    MissingAttribute(&'static str),
    #[error("mesh has no indices")]
    NoIndices,
    #[error("unexpected attribute format for {0}")]
    BadAttributeFormat(&'static str),
    #[error("meshopt VertexDataAdapter rejected position buffer")]
    VertexAdapter,
    #[error("mikktspace tangent generation failed: {0}")]
    TangentGen(#[from] bevy_mesh::GenerateTangentsError),
}

/// Decide which vertices `simplify_with_locks` must keep in place
/// during this partition's simplification.
///
/// The lock decision is per *position*, not per vertex index — wedges
/// (multiple vertex indices at the same position, used to split
/// normals / UVs across attribute seams) all share the same lock
/// state. Each `position_remap[v]` gives the canonical id for the
/// position; we tag canonical ids that appear in clusters *outside*
/// the target partition, then propagate the tag back to every vertex
/// index that maps to a tagged canonical id.
///
/// Position-based locking is load-bearing for crack-free DAG cuts:
/// if we locked by raw index instead, adjacent partitions using
/// different wedges for a shared boundary position would each see
/// "the OTHER wedge isn't in my partition either" and both ends could
/// move — opening a tear at the LOD seam.
fn compute_group_vertex_lock(
    cluster_index_counts: &[u32],
    mesh_vertex_count: usize,
    target_partition: u32,
    partition_dest: &[u32],
    cluster_indices_flat: &[u32],
    position_remap: &[u32],
) -> Vec<bool> {
    // Pass 1: tag canonical position ids used by OTHER partitions.
    let mut canonical_locked = vec![false; mesh_vertex_count];
    let mut cursor = 0usize;
    for (cluster_id, &count) in cluster_index_counts.iter().enumerate() {
        let range_end = cursor + count as usize;
        if partition_dest[cluster_id] != target_partition {
            for &v in &cluster_indices_flat[cursor..range_end] {
                canonical_locked[position_remap[v as usize] as usize] = true;
            }
        }
        cursor = range_end;
    }
    // Pass 2: lock every vertex index whose canonical id was tagged.
    let mut locked = vec![false; mesh_vertex_count];
    for v in 0..mesh_vertex_count {
        locked[v] = canonical_locked[position_remap[v] as usize];
    }
    locked
}

/// AABB-center-based bounding sphere — conservative approximation,
/// sufficient for the screen-space-error metric.
fn bounding_sphere(points: impl Iterator<Item = Vec3>) -> (Vec3, f32) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut count = 0usize;
    for p in points {
        min = min.min(p);
        max = max.max(p);
        count += 1;
    }
    if count == 0 {
        return (Vec3::ZERO, 0.0);
    }
    let center = (min + max) * 0.5;
    let half = (max - min) * 0.5;
    (center, half.length())
}

fn mesh_aabb(positions: &[Vec3]) -> ClusterMeshAabb {
    if positions.is_empty() {
        return ClusterMeshAabb::default();
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for &p in positions {
        min = min.min(p);
        max = max.max(p);
    }
    let center = (min + max) * 0.5;
    let half = (max - min) * 0.5;
    ClusterMeshAabb {
        center: [center.x, center.y, center.z, 0.0],
        half_extent: [half.x, half.y, half.z, 0.0],
    }
}

// Octahedral encoding for unit-length normals → 2x f32 in [-1, 1].
// Same construction as `bevy_pbr::meshlet::from_mesh`.
fn octahedral_encode(v: Vec3) -> Vec2 {
    let n = v / (v.x.abs() + v.y.abs() + v.z.abs());
    let octahedral_wrap = (1.0 - n.yx().abs())
        * Vec2::new(
            if n.x >= 0.0 { 1.0 } else { -1.0 },
            if n.y >= 0.0 { 1.0 } else { -1.0 },
        );
    if n.z >= 0.0 {
        n.xy()
    } else {
        octahedral_wrap
    }
}

// WGSL pack2x16snorm equivalent — two f32s in [-1, 1] packed into a u32.
// https://www.w3.org/TR/WGSL/#pack2x16snorm-builtin
fn pack2x16snorm(v: Vec2) -> u32 {
    let v = v.clamp(Vec2::NEG_ONE, Vec2::ONE);
    let v = (v * 32767.0 + 0.5).floor().as_i16vec2();
    bytemuck::cast(v)
}

fn extract_vec3(
    mesh: &Mesh,
    id: MeshVertexAttributeId,
    name: &'static str,
) -> Result<Vec<Vec3>, MeshToClusterMeshConversionError> {
    match mesh.attribute(id) {
        Some(VertexAttributeValues::Float32x3(v)) => Ok(v.iter().map(|&a| Vec3::from(a)).collect()),
        Some(_) => Err(MeshToClusterMeshConversionError::BadAttributeFormat(name)),
        None => Err(MeshToClusterMeshConversionError::MissingAttribute(name)),
    }
}

fn extract_vec4(
    mesh: &Mesh,
    id: MeshVertexAttributeId,
    name: &'static str,
) -> Result<Vec<Vec4>, MeshToClusterMeshConversionError> {
    match mesh.attribute(id) {
        Some(VertexAttributeValues::Float32x4(v)) => Ok(v.iter().map(|&a| Vec4::from(a)).collect()),
        Some(_) => Err(MeshToClusterMeshConversionError::BadAttributeFormat(name)),
        None => Err(MeshToClusterMeshConversionError::MissingAttribute(name)),
    }
}

fn extract_vec2(
    mesh: &Mesh,
    id: MeshVertexAttributeId,
    name: &'static str,
) -> Result<Vec<Vec2>, MeshToClusterMeshConversionError> {
    match mesh.attribute(id) {
        Some(VertexAttributeValues::Float32x2(v)) => Ok(v.iter().map(|&a| Vec2::from(a)).collect()),
        Some(_) => Err(MeshToClusterMeshConversionError::BadAttributeFormat(name)),
        None => Err(MeshToClusterMeshConversionError::MissingAttribute(name)),
    }
}
