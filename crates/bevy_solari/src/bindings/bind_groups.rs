//! Bind group layout + cached bind group for the cluster
//! scene state. Matches the `@group(0)` declarations in
//! `cluster_bindings.slang`.
//!
//! Skips bind-group creation while any required buffer hasn't been
//! populated yet (scene has no `RaytracingMesh3d` entities, or no
//! `ClusterMesh` asset has finished loading). Consumers should
//! treat [`ClusterSceneBindGroup`] as `Option<...>`.

use crate::geometry::ClusterMeshManager;
use crate::ecs_gpu::GpuColumn;
use crate::instance::{GroupBaseColumn, LodInputColumn, TransformColumn};
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::{
    render_resource::{
        binding_types::storage_buffer_read_only_sized, BindGroup, BindGroupEntries,
        BindGroupLayoutDescriptor, BindGroupLayoutEntries, BufferId, PipelineCache, ShaderStages,
    },
    renderer::RenderDevice,
};

/// Bind group layout descriptor for the cluster-scene set 0
/// (`cluster_bindings.slang`), materialized at bind-group creation time via
/// [`PipelineCache::get_bind_group_layout`].
///
/// All [`BIND_GROUP_BUFFER_COUNT`] entries are read-only storage buffers.
#[derive(Resource, Clone)]
pub struct ClusterSceneBindGroupLayout(pub BindGroupLayoutDescriptor);

/// Number of buffer-backed entries in the scene bind group. Doubles as
/// the length of the rebuild-skip signature in [`ClusterSceneBindGroup`].
pub const BIND_GROUP_BUFFER_COUNT: usize = 13;

pub fn init_cluster_scene_bind_group_layout(mut commands: Commands) {
    let descriptor = BindGroupLayoutDescriptor::new(
        "cluster_scene_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
            (
                // Mesh-pool buffers (bindings 0..=9)
                storage_buffer_read_only_sized(false, None), // vertex_positions
                storage_buffer_read_only_sized(false, None), // vertex_normals
                storage_buffer_read_only_sized(false, None), // vertex_tangents
                storage_buffer_read_only_sized(false, None), // vertex_uvs
                storage_buffer_read_only_sized(false, None), // cluster_indices
                storage_buffer_read_only_sized(false, None), // child_table
                storage_buffer_read_only_sized(false, None), // clusters
                storage_buffer_read_only_sized(false, None), // cluster_groups
                storage_buffer_read_only_sized(false, None), // cluster_nodes
                storage_buffer_read_only_sized(false, None), // cluster_to_group
                // Per-instance buffers (bindings 10..=12)
                storage_buffer_read_only_sized(false, None), // 10 instance_transforms (column)
                storage_buffer_read_only_sized(false, None), // 11 instance_group_bases
                storage_buffer_read_only_sized(false, None), // 12 instance_lod_inputs
            ),
        ),
    );
    commands.insert_resource(ClusterSceneBindGroupLayout(descriptor));
}

/// Cached bind group for the cluster scene `@group(0)`, plus its descriptor-
/// heap mirror (the set-0 surface the cluster AS heap kernels map at constant
/// offsets — see `gpu::rt_pipeline::cluster_heap_mappings`).
///
/// The bind group is byte-identical frame-to-frame unless one of its
/// buffer *handles* changes — and a handle only changes when that buffer
/// is reallocated on growth (a mesh-pool upload, the transform column, or
/// a slot-buffer resize), which is rare. So we cache the materialized
/// group + a signature of its handle ids and rebuild only when the
/// signature moves; a static frame skips both the rebuild and the
/// `PipelineCache` layout lock entirely. The heap mirror is rewritten on
/// the same cadence.
///
/// `bind_group` is `None` when the scene has no `RaytracingMesh3d`
/// instances or the asset hasn't loaded yet (a required buffer wasn't
/// allocated).
#[derive(Resource, Default)]
pub struct ClusterSceneBindGroup {
    /// The materialized `@group(0)` — what consumers bind.
    pub bind_group: Option<BindGroup>,
    /// Handle ids of the [`BIND_GROUP_BUFFER_COUNT`] entries at last
    /// build. `None` forces a rebuild (and is reset on every early-out so
    /// a later valid frame rebuilds).
    buffer_ids: Option<[BufferId; BIND_GROUP_BUFFER_COUNT]>,
    /// The buffers mirrored into the descriptor heap as
    /// `(binding, heap buffer-region slot)` pairs, rewritten on the same
    /// signature cadence as the bind group. The slots are allocated once
    /// (the binding count is fixed) and rewritten in place, so the mapping
    /// tables the AS kernels baked from them never go stale — this field
    /// survives the early-out resets above.
    pub heap_slots: Option<Vec<(u32, u32)>>,
}

/// Rebuild the scene bind group when a buffer handle changed. Runs in
/// `Render::PrepareBindGroups` after the cluster-mesh-pool writes complete
/// in `Render::PrepareAssets`. A static frame does nothing — see
/// [`ClusterSceneBindGroup`].
pub fn prepare_cluster_scene_bind_group(
    mut bind_group: ResMut<ClusterSceneBindGroup>,
    layout: Res<ClusterSceneBindGroupLayout>,
    pipeline_cache: Res<PipelineCache>,
    mesh_manager: Res<ClusterMeshManager>,
    transforms: Option<Res<GpuColumn<TransformColumn>>>,
    group_bases: Option<Res<GpuColumn<GroupBaseColumn>>>,
    lod_inputs: Option<Res<GpuColumn<LodInputColumn>>>,
    seam: Option<Res<crate::gpu::binding_seam::BindingSeam>>,
    render_device: Res<RenderDevice>,
) {
    // Early-outs clear only the bind group + signature: `heap_slots` must
    // survive (the AS kernels' mapping tables bake those slot offsets).
    let (Some(transforms), Some(group_bases), Some(lod_inputs)) =
        (transforms, group_bases, lod_inputs)
    else {
        bind_group.bind_group = None;
        bind_group.buffer_ids = None;
        return;
    };
    // Mesh-pool buffers have no committed sparse pages until the first
    // `perform_pending_cluster_mesh_writes` runs — binding/reading them then would
    // fault on unbacked memory. Bail until something is uploaded.
    if mesh_manager.vertex_positions.is_empty() {
        bind_group.bind_group = None;
        bind_group.buffer_ids = None;
        return;
    }

    // The slot-indexed instance columns (`transforms` / `group_bases` /
    // `lod_inputs`) are GPU-scattered by `gpu_instances`; here we only bind
    // them. Their handles change only on a `GpuColumn` reallocation (growth),
    // which the id signature below catches.

    // Handle-id signature, same order as the entries below. Unchanged ids
    // ⇒ the live bind group is byte-identical ⇒ skip the rebuild (and the
    // `PipelineCache` layout lock / descriptor-set churn it costs).
    let buffer_ids = [
        mesh_manager.vertex_positions.buffer().id(),
        mesh_manager.vertex_normals.buffer().id(),
        mesh_manager.vertex_tangents.buffer().id(),
        mesh_manager.vertex_uvs.buffer().id(),
        mesh_manager.indices.buffer().id(),
        mesh_manager.child_table.buffer().id(),
        mesh_manager.clusters.buffer().id(),
        mesh_manager.groups.buffer().id(),
        mesh_manager.nodes.buffer().id(),
        mesh_manager.cluster_to_group.buffer().id(),
        transforms.buffer().id(),
        group_bases.buffer().id(),
        lod_inputs.buffer().id(),
    ];
    if bind_group.bind_group.is_some() && bind_group.buffer_ids == Some(buffer_ids) {
        return;
    }

    let group = render_device.create_bind_group(
        "cluster_scene_bind_group",
        &pipeline_cache.get_bind_group_layout(&layout.0),
        &BindGroupEntries::sequential((
            mesh_manager.vertex_positions.binding(),
            mesh_manager.vertex_normals.binding(),
            mesh_manager.vertex_tangents.binding(),
            mesh_manager.vertex_uvs.binding(),
            mesh_manager.indices.binding(),
            mesh_manager.child_table.binding(),
            mesh_manager.clusters.binding(),
            mesh_manager.groups.binding(),
            mesh_manager.nodes.binding(),
            mesh_manager.cluster_to_group.binding(),
            transforms.buffer().as_entire_binding(),
            group_bases.buffer().as_entire_binding(),
            lod_inputs.buffer().as_entire_binding(),
        )),
    );

    // Mirror the same 13 buffers into the descriptor heap on the same
    // signature cadence as the bind group — the cluster AS heap kernels map
    // set 0 from these slots at constant offsets. Rewritten in place so the
    // baked mapping tables stay valid across buffer reallocation; slots are
    // re-allocated only if the binding count itself changes. Each descriptor
    // covers the entire (sparse, stable-address) buffer, matching the
    // `as_entire` bind-group entries above.
    if let Some(seam) = seam.as_deref() {
        use crate::gpu::binding_seam::{HeapKind, HeapResource};
        let buffers: [&bevy_render::render_resource::Buffer; BIND_GROUP_BUFFER_COUNT] = [
            mesh_manager.vertex_positions.buffer(),
            mesh_manager.vertex_normals.buffer(),
            mesh_manager.vertex_tangents.buffer(),
            mesh_manager.vertex_uvs.buffer(),
            mesh_manager.indices.buffer(),
            mesh_manager.child_table.buffer(),
            mesh_manager.clusters.buffer(),
            mesh_manager.groups.buffer(),
            mesh_manager.nodes.buffer(),
            mesh_manager.cluster_to_group.buffer(),
            transforms.buffer(),
            group_bases.buffer(),
            lod_inputs.buffer(),
        ];
        let slots: Vec<(u32, u32)> = match bind_group.heap_slots.take() {
            Some(slots) if slots.len() == buffers.len() => slots,
            stale => {
                for (_, slot) in stale.into_iter().flatten() {
                    seam.free_heap_index(HeapKind::Buffer, slot);
                }
                (0..buffers.len() as u32)
                    .map(|binding| (binding, seam.alloc_heap_block(HeapKind::Buffer, 1)))
                    .collect()
            }
        };
        for (buffer, &(_, slot)) in buffers.iter().zip(&slots) {
            seam.rewrite_heap_index(
                HeapKind::Buffer,
                slot,
                HeapResource::Buffer {
                    address: seam.device_address(buffer).get(),
                    size: buffer.size(),
                },
            );
        }
        bind_group.heap_slots = Some(slots);
    }

    bind_group.bind_group = Some(group);
    bind_group.buffer_ids = Some(buffer_ids);
}
