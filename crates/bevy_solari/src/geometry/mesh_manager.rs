use super::asset::{
    Cluster, ClusterBvhNode, ClusterLodGroup, ClusterMesh, ClusterMeshAabb, OmmDesc, OmmUsage,
    PackedVertex,
};
use super::indices::{ClusterIndex, GroupIndex, NodeIndex};
use crate::gpu::extension::opacity_micromap_available;
use crate::gpu::allocator::Allocator;
use crate::gpu::persistent_buffer::PersistentGpuBuffer;
use alloc::sync::Arc;
use bevy_asset::{AssetId, Assets};
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_math::{Vec2, Vec3, Vec4};
use bevy_platform::collections::HashMap;
use bevy_render::{
    render_resource::BufferAddress,
    renderer::{RenderDevice, RenderQueue},
};
use core::ops::Range;

/// Per-cluster snapshot the tessellation showcase consumes: global vertex / index
/// offsets + triangle count (for the displace dispatch) plus the cluster's
/// object-space bounds-sphere center, which drives **per-cluster** view-dependent
/// LOD (so detail follows the camera across one surface, not the whole instance).
#[derive(Clone, Copy, Debug)]
pub struct TessCluster {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub triangle_count: u32,
    /// Object-space cluster centroid (`Cluster::bounds_sphere` xyz).
    pub center: [f32; 3],
}

/// Per-asset slice ownership inside [`ClusterMeshManager`]'s buffers.
#[derive(Clone)]
struct ClusterMeshSlices {
    vertex_positions: Range<BufferAddress>,
    /// Freed in lockstep with `vertex_positions` to keep both pools' per-mesh
    /// element index identical (the resolve indexes both with one global index).
    vertex_packed: Range<BufferAddress>,
    vertex_normals: Range<BufferAddress>,
    vertex_tangents: Range<BufferAddress>,
    vertex_uvs: Range<BufferAddress>,
    indices: Range<BufferAddress>,
    child_table: Range<BufferAddress>,
    clusters: Range<BufferAddress>,
    groups: Range<BufferAddress>,
    nodes: Range<BufferAddress>,
    cluster_to_group: Range<BufferAddress>,
    aabb: ClusterMeshAabb,
    root_group: GroupIndex,
    root_node: NodeIndex,
    lod_levels: u32,
    mesh_max_error: f32,
    cluster_count: u32,
    total_triangle_count: u32,
    /// Per-cluster [`TessCluster`] (global offsets + triangle count + centroid) for
    /// every cluster in this mesh, retained for the tessellation showcase (which
    /// subdivides a real instance's geometry in place; the per-cluster `Cluster`
    /// records are consumed when the asset uploads, so this snapshot is the only
    /// post-upload CPU access to them).
    tess_clusters: Vec<TessCluster>,
    /// Dense, stable per-unique-geometry id assigned on first upload.
    /// BLAS sharing keys one shared BLAS per geometry on this (NOT on
    /// camera distance), so the BLAS count is bounded by the resident
    /// geometry universe. Monotonic for now (not recycled on remove —
    /// eviction is a follow-up).
    geometry_id: u32,
}

/// Per-asset metadata returned from [`ClusterMeshManager::queue_upload_if_needed`].
/// Global slot ids (post-rebase) for downstream selector / shader binding.
#[derive(Copy, Clone, Debug)]
pub struct ClusterMeshUpload {
    /// Global slot of this mesh's first cluster in the cluster pool.
    pub cluster_base: ClusterIndex,
    /// Global slot of this mesh's first group in the group pool.
    pub group_base: GroupIndex,
    /// Global slot of this mesh's root node, or [`NodeIndex::NONE`]
    /// if the mesh has no interior node tree.
    pub root_node: NodeIndex,
    /// Global slot of this mesh's root group.
    pub root_group: GroupIndex,
    pub aabb: ClusterMeshAabb,
    pub lod_levels: u32,
    pub mesh_max_error: f32,
    /// Cluster count in this mesh — exposed for the selector's
    /// per-instance dispatch (`for cid in 0..cluster_count`).
    pub cluster_count: u32,
    /// Total triangle count across every cluster in this mesh —
    /// used by emissive-mesh light sampling to pick a uniform random
    /// triangle within the instance.
    pub total_triangle_count: u32,
    /// Dense, stable per-unique-geometry id. BLAS sharing builds one
    /// shared BLAS per geometry keyed on this; every instance of this
    /// mesh references `geometry_blas_pool.base + geometry_id * stride`.
    pub geometry_id: u32,
}

/// One entry in [`ClusterMeshManager::pending_clas_uploads`] —
/// metadata + cluster data that downstream CLAS-build code needs.
/// The manager queues a [`PendingClasUpload`] on every fresh mesh
/// upload; [`crate::geometry::clas_arena`] drains the queue after the
/// per-mesh vertex / index writes have landed.
/// Opacity micro-map payload threaded from a [`ClusterMesh`] to the CLAS build
/// (cheap `Arc` clones). Present only when the mesh carries a baked OMM *and*
/// `VK_EXT_opacity_micromap` is available. The CLAS build (`clas_arena`) builds
/// one `VkMicromapEXT` from `array_data` + `descs` and references it per cluster
/// via the per-triangle `index`.
#[derive(Clone, Debug)]
pub struct OmmUploadData {
    pub array_data: Arc<[u8]>,
    pub descs: Arc<[OmmDesc]>,
    /// Per-triangle OMM index, parallel to the mesh's triangles in cluster order.
    pub index: Arc<[i32]>,
    pub usage: Arc<[OmmUsage]>,
}

#[derive(Clone, Debug)]
pub struct PendingClasUpload {
    pub asset_id: AssetId<ClusterMesh>,
    pub cluster_base: ClusterIndex,
    /// Mesh-pool vertex slot where this mesh's vertex data starts.
    /// The CLAS build needs this to compute the absolute
    /// `vertex_buffer` address per cluster — `cluster.vertex_offset`
    /// in `clusters` is still MESH-LOCAL at this point (the
    /// global-slot rebase happens GPU-side in
    /// [`crate::geometry::persistent_buffer_impls`]).
    pub vertex_base: u32,
    /// Same idea for the index pool.
    pub index_base: u32,
    /// Shared with the mesh-pool upload (`Arc` clone). Holding it
    /// keeps the cluster data alive past the `remove_untracked` that
    /// fires inside [`ClusterMeshManager::queue_upload_if_needed`].
    pub clusters: Arc<[Cluster]>,
    /// Baked opacity micro-map for this mesh, or `None` (no OMM / extension
    /// unavailable). Consumed by the CLAS build to attach the OMM per cluster.
    pub omm: Option<OmmUploadData>,
}

/// Manages uploading [`ClusterMesh`] asset data to the GPU.
#[derive(Resource)]
pub struct ClusterMeshManager {
    pub vertex_positions: PersistentGpuBuffer<Arc<[Vec3]>>,
    pub vertex_normals: PersistentGpuBuffer<Arc<[u32]>>,
    pub vertex_tangents: PersistentGpuBuffer<Arc<[Vec4]>>,
    pub vertex_uvs: PersistentGpuBuffer<Arc<[Vec2]>>,
    /// Interleaved (AoS) copy of the four vertex streams above, parallel to
    /// `vertex_positions` (same global vertex index). One contiguous 28-byte
    /// [`PackedVertex`] per vertex for the bindless RT-pipeline resolve's
    /// cache-friendly `physical_load`; the SoA pools stay for the CLAS build
    /// and the AS-selector's position/normal reads.
    pub vertex_packed: PersistentGpuBuffer<Arc<[PackedVertex]>>,
    pub indices: PersistentGpuBuffer<Arc<[u32]>>,
    pub child_table: PersistentGpuBuffer<Arc<[u32]>>,
    pub clusters: PersistentGpuBuffer<Arc<[Cluster]>>,
    pub groups: PersistentGpuBuffer<Arc<[ClusterLodGroup]>>,
    pub nodes: PersistentGpuBuffer<Arc<[ClusterBvhNode]>>,
    /// Parallel to `clusters` — per-cluster group id, rebased to
    /// global group ids before queueing.
    pub cluster_to_group: PersistentGpuBuffer<Arc<[u32]>>,
    cluster_mesh_slices: HashMap<AssetId<ClusterMesh>, ClusterMeshSlices>,
    /// Next dense geometry id to hand out. Monotonic (geometry ids are
    /// not recycled on `remove` yet); `geometry_count` == this value ==
    /// the geometry-pool high-water BLAS sharing sizes against.
    next_geometry_id: u32,
    pub pending_clas_uploads: Vec<PendingClasUpload>,
}

pub fn init_cluster_mesh_manager(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
) {
    // The pools are sparse buffers (stable handle across growth); they need the
    // cluster allocator, created in `SolariSetup` (this runs after). Absent →
    // device lacks the support solari needs; skip (consumers read it via Option).
    let Some(allocator) = allocator else {
        return;
    };
    let mut manager = ClusterMeshManager {
        vertex_positions: PersistentGpuBuffer::new("cluster_vertex_positions", &render_device, &allocator),
        vertex_normals: PersistentGpuBuffer::new("cluster_vertex_normals", &render_device, &allocator),
        vertex_tangents: PersistentGpuBuffer::new("cluster_vertex_tangents", &render_device, &allocator),
        vertex_uvs: PersistentGpuBuffer::new("cluster_vertex_uvs", &render_device, &allocator),
        vertex_packed: PersistentGpuBuffer::new("cluster_vertex_packed", &render_device, &allocator),
        indices: PersistentGpuBuffer::new("cluster_indices", &render_device, &allocator),
        child_table: PersistentGpuBuffer::new("cluster_child_table", &render_device, &allocator),
        clusters: PersistentGpuBuffer::new("clusters", &render_device, &allocator),
        groups: PersistentGpuBuffer::new("cluster_groups", &render_device, &allocator),
        nodes: PersistentGpuBuffer::new("cluster_nodes", &render_device, &allocator),
        cluster_to_group: PersistentGpuBuffer::new("cluster_to_group", &render_device, &allocator),
        cluster_mesh_slices: HashMap::default(),
        next_geometry_id: 0,
        pending_clas_uploads: Vec::new(),
    };

    // Queue a placeholder write to `child_table` + `nodes` so the
    // underlying GPU buffer has non-zero size on first frame — the
    // current bake leaves both empty (no interior-node tree yet),
    // and wgpu rejects size-zero bind-group entries. Contents are
    // unused; both bindings are declared in the cluster scene bind
    // group layout for future DAG-tree consumers.
    let _ = manager
        .child_table
        .queue_write(Arc::from([0u32].as_slice()), ());
    let _ = manager
        .nodes
        .queue_write(Arc::from([ClusterBvhNode::default()].as_slice()), 0);

    commands.insert_resource(manager);
}

impl ClusterMeshManager {
    /// Queue a [`ClusterMesh`] for GPU upload if not already
    /// uploaded. Returns the global slot indices selectors and
    /// shaders need to bind into the persistent buffers.
    pub fn queue_upload_if_needed(
        &mut self,
        asset_id: AssetId<ClusterMesh>,
        assets: &mut Assets<ClusterMesh>,
    ) -> ClusterMeshUpload {
        if let Some(slices) = self.cluster_mesh_slices.get(&asset_id) {
            return upload_from_slices(slices);
        }

        let mesh = assets.remove_untracked(asset_id).expect(
            "ClusterMesh asset was already unloaded but is not registered with ClusterMeshManager",
        );

        // 1. Queue independent streams first; their `.start` becomes
        //    the base for downstream streams' rebase metadata.
        let vertex_positions = self
            .vertex_positions
            .queue_write(Arc::clone(&mesh.vertex_positions), ());
        let vertex_normals = self
            .vertex_normals
            .queue_write(Arc::clone(&mesh.vertex_normals), ());
        let vertex_tangents = self
            .vertex_tangents
            .queue_write(Arc::clone(&mesh.vertex_tangents), ());
        let vertex_uvs = self
            .vertex_uvs
            .queue_write(Arc::clone(&mesh.vertex_uvs), ());
        // Interleaved AoS copy for the bindless RT resolve. Built parallel to the
        // SoA streams above; since every pool gets exactly one bump-allocated
        // append per mesh in the same order, the packed pool's element base equals
        // `vertex_base` (asserted below), so the resolve indexes it with the same
        // global vertex index.
        let packed: Arc<[PackedVertex]> = (0..mesh.vertex_positions.len())
            .map(|i| PackedVertex {
                normal: mesh.vertex_normals[i],
                tangent: super::asset::pack_tangent(mesh.vertex_tangents[i]),
                uv: mesh.vertex_uvs[i].to_array(),
            })
            .collect();
        let vertex_packed = self.vertex_packed.queue_write(packed, ());
        let indices = self.indices.queue_write(Arc::clone(&mesh.indices), ());
        // `child_table` is empty for current bakes (interior-node
        // tree isn't built yet); guard against the zero-byte
        // allocate that `range-alloc` panics on.
        let child_table = if mesh.child_table.is_empty() {
            0..0
        } else {
            self.child_table
                .queue_write(Arc::clone(&mesh.child_table), ())
        };

        let vertex_base = (vertex_positions.start / size_of::<Vec3>() as u64) as u32;
        // The packed pool must share the SoA global vertex index (the resolve uses
        // one index for both). Holds while both pools bump-allocate one append per
        // mesh in lockstep.
        debug_assert_eq!(
            vertex_base,
            (vertex_packed.start / size_of::<PackedVertex>() as u64) as u32,
            "vertex_packed pool diverged from vertex_positions indexing",
        );
        let index_base = (indices.start / size_of::<u32>() as u64) as u32;
        let child_table_base = (child_table.start / size_of::<u32>() as u64) as u32;

        // 2. Clusters need vertex + index bases; queue them next.
        let clusters = self
            .clusters
            .queue_write(Arc::clone(&mesh.clusters), (vertex_base, index_base));
        let cluster_base =
            ClusterIndex((clusters.start / size_of::<Cluster>() as u64) as u32);

        // 3. Groups need cluster + child_table bases.
        let groups = self
            .groups
            .queue_write(Arc::clone(&mesh.groups), (cluster_base, child_table_base));
        let group_base =
            GroupIndex((groups.start / size_of::<ClusterLodGroup>() as u64) as u32);

        // 4. Nodes need child_table base. Empty for current bakes
        //    (no interior-node tree yet).
        let nodes = if mesh.nodes.is_empty() {
            0..0
        } else {
            self.nodes
                .queue_write(Arc::clone(&mesh.nodes), child_table_base)
        };

        // 5. cluster_to_group's values are group ids — rebase
        //    CPU-side to global ids before queueing. Done here
        //    rather than in the impl because `Arc<[u32]>` is also
        //    used by `child_table` (no rebase), so the impl can't
        //    take group_base as metadata without a newtype split.
        let cluster_to_group_rebased: Arc<[u32]> = mesh
            .cluster_to_group
            .iter()
            .map(|g| g + group_base.0)
            .collect();
        let cluster_to_group = self
            .cluster_to_group
            .queue_write(cluster_to_group_rebased, ());

        // 6. Rebase the asset's root pointers from mesh-local to
        //    global slots so downstream consumers don't need to
        //    track per-instance base indices.
        let root_group = GroupIndex(mesh.root_group_id + group_base.0);
        let root_node = if mesh.root_node_id == u32::MAX {
            NodeIndex::NONE
        } else {
            NodeIndex(
                mesh.root_node_id
                    + (nodes.start / size_of::<ClusterBvhNode>() as u64) as u32,
            )
        };

        let cluster_count = mesh.clusters.len() as u32;
        let total_triangle_count: u32 = mesh.clusters.iter().map(|c| c.triangle_count).sum();

        // Hand out a dense geometry id for this brand-new mesh.
        let geometry_id = self.next_geometry_id;
        self.next_geometry_id += 1;

        let slices = ClusterMeshSlices {
            vertex_positions,
            vertex_packed,
            vertex_normals,
            vertex_tangents,
            vertex_uvs,
            indices,
            child_table,
            clusters,
            groups,
            nodes,
            cluster_to_group,
            aabb: mesh.aabb,
            root_group,
            root_node,
            lod_levels: mesh.lod_levels,
            mesh_max_error: mesh.mesh_max_error,
            cluster_count,
            total_triangle_count,
            // Rebase each cluster's mesh-local vertex/index offsets to global pool
            // offsets (the same +base the GPU `Cluster` records get on upload).
            tess_clusters: mesh
                .clusters
                .iter()
                .map(|c| TessCluster {
                    vertex_offset: vertex_base + c.vertex_offset,
                    index_offset: index_base + c.index_offset,
                    triangle_count: c.triangle_count,
                    center: [c.bounds_sphere[0], c.bounds_sphere[1], c.bounds_sphere[2]],
                })
                .collect(),
            geometry_id,
        };

        let upload = ClusterMeshUpload {
            cluster_base,
            group_base,
            root_node,
            root_group,
            aabb: mesh.aabb,
            lod_levels: mesh.lod_levels,
            mesh_max_error: mesh.mesh_max_error,
            cluster_count,
            total_triangle_count,
            geometry_id,
        };

        self.cluster_mesh_slices.insert(asset_id, slices);
        // Thread the baked OMM through to the CLAS build, but only if the device
        // can actually consume it — otherwise alpha cutouts fall back to any-hit.
        let omm = (opacity_micromap_available() && mesh.has_opacity_micromap()).then(|| {
            OmmUploadData {
                array_data: Arc::clone(&mesh.omm_array_data),
                descs: Arc::clone(&mesh.omm_descs),
                index: Arc::clone(&mesh.omm_index),
                usage: Arc::clone(&mesh.omm_usage),
            }
        });
        let pending = PendingClasUpload {
            asset_id,
            cluster_base,
            vertex_base,
            index_base,
            clusters: Arc::clone(&mesh.clusters),
            omm,
        };
        self.pending_clas_uploads.push(pending);
        upload
    }

    /// Number of distinct [`ClusterMesh`] assets currently resident on
    /// the GPU.
    #[inline]
    pub fn resident_mesh_count(&self) -> usize {
        self.cluster_mesh_slices.len()
    }

    /// Per-cluster global `[vertex_offset, index_offset, triangle_count]` for every
    /// cluster of a resident mesh — the tessellation showcase subdivides this real
    /// geometry in place. `None` if the mesh isn't resident yet.
    #[inline]
    pub fn tess_clusters(&self, asset_id: AssetId<ClusterMesh>) -> Option<&[TessCluster]> {
        self.cluster_mesh_slices
            .get(&asset_id)
            .map(|s| s.tess_clusters.as_slice())
    }

    /// Object-space `(min, max)` AABB of a resident mesh — the tessellation showcase
    /// derives its world bounds from this (+ the displacement margin) instead of a
    /// per-frame readback of every displaced position. `None` if not resident.
    #[inline]
    pub fn mesh_aabb(&self, asset_id: AssetId<ClusterMesh>) -> Option<([f32; 3], [f32; 3])> {
        self.cluster_mesh_slices.get(&asset_id).map(|s| {
            let c = s.aabb.center;
            let h = s.aabb.half_extent;
            (
                [c[0] - h[0], c[1] - h[1], c[2] - h[2]],
                [c[0] + h[0], c[1] + h[1], c[2] + h[2]],
            )
        })
    }

    /// High-water count of dense geometry ids handed out — the size the
    /// per-geometry BLAS pool / desired-level / built-level buffers must
    /// cover. Monotonic (ids not recycled on `remove` yet), so any live
    /// `geometry_id` is in `0..geometry_count()`.
    #[inline]
    pub fn geometry_count(&self) -> u32 {
        self.next_geometry_id
    }

    pub fn remove(&mut self, asset_id: &AssetId<ClusterMesh>) {
        let Some(slices) = self.cluster_mesh_slices.remove(asset_id) else {
            return;
        };
        self.vertex_positions.mark_slice_unused(slices.vertex_positions);
        self.vertex_packed.mark_slice_unused(slices.vertex_packed);
        self.vertex_normals.mark_slice_unused(slices.vertex_normals);
        self.vertex_tangents.mark_slice_unused(slices.vertex_tangents);
        self.vertex_uvs.mark_slice_unused(slices.vertex_uvs);
        self.indices.mark_slice_unused(slices.indices);
        self.child_table.mark_slice_unused(slices.child_table);
        self.clusters.mark_slice_unused(slices.clusters);
        self.groups.mark_slice_unused(slices.groups);
        self.nodes.mark_slice_unused(slices.nodes);
        self.cluster_to_group.mark_slice_unused(slices.cluster_to_group);
    }
}

fn upload_from_slices(slices: &ClusterMeshSlices) -> ClusterMeshUpload {
    ClusterMeshUpload {
        cluster_base: ClusterIndex(
            (slices.clusters.start / size_of::<Cluster>() as u64) as u32,
        ),
        group_base: GroupIndex(
            (slices.groups.start / size_of::<ClusterLodGroup>() as u64) as u32,
        ),
        root_node: slices.root_node,
        root_group: slices.root_group,
        aabb: slices.aabb,
        lod_levels: slices.lod_levels,
        mesh_max_error: slices.mesh_max_error,
        cluster_count: slices.cluster_count,
        total_triangle_count: slices.total_triangle_count,
        geometry_id: slices.geometry_id,
    }
}

/// Upload all newly queued [`ClusterMesh`] asset data to the GPU.
pub fn perform_pending_cluster_mesh_writes(
    mut manager: ResMut<ClusterMeshManager>,
    render_queue: Res<RenderQueue>,
) {
    manager
        .vertex_positions
        .perform_writes(&render_queue);
    manager
        .vertex_normals
        .perform_writes(&render_queue);
    manager
        .vertex_tangents
        .perform_writes(&render_queue);
    manager
        .vertex_uvs
        .perform_writes(&render_queue);
    manager
        .vertex_packed
        .perform_writes(&render_queue);
    manager.indices.perform_writes(&render_queue);
    manager
        .child_table
        .perform_writes(&render_queue);
    manager.clusters.perform_writes(&render_queue);
    manager.groups.perform_writes(&render_queue);
    manager.nodes.perform_writes(&render_queue);
    manager
        .cluster_to_group
        .perform_writes(&render_queue);
}
