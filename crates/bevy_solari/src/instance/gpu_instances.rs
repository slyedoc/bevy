//! The concrete per-instance GPU columns — the GPU-side "components" of a
//! [`GpuEntity`]. Each is a [`GpuColumnDesc`] (value type + source + dirty
//! set); [`GpuColumnPlugin`] builds the scatter pipeline. [`GpuInstancesPlugin`]
//! registers all of them.
//!
//! One mechanism, a per-column dirty set: `transforms` scatters its pre-built
//! `transform_delta` (and keeps its previous frame GPU-side, `KEEP_PREVIOUS`);
//! `material_ids` on `material_dirty` (bind / asset swap / re-resolve);
//! `group_bases` / `lod_inputs` / `geometry_ids` on `added_slots` (bind only).
//! Every column scatters through the same `GpuColumn`.

use bevy_app::{App, Plugin};
use bevy_ecs::system::{Res, SystemParam};
use bevy_math::{Affine3, Affine3A, Affine3Ext, Vec4};
use bytemuck::{Pod, Zeroable};

use crate::ecs_gpu::{GpuColumn, GpuColumnDesc, GpuColumnPlugin, GpuTable};
use super::instance_manager::{InstanceLodInputGpu, InstanceManager};

/// The per-instance columns all live in the [`InstanceManager`] table; its slot
/// high-water bounds every one of them.
impl GpuTable for InstanceManager {
    fn high_water(&self) -> u32 {
        self.slot_high_water()
    }
}

/// Bytes per transform-column entry: `mat3x4<f32>` = 3 × `vec4` = 48 B.
const AFFINE_STRIDE: u64 = 48;

/// `mat3x4<f32>` affine, GPU layout matching the WGSL the selector already
/// uses: column `k` (a `vec4`) is row `k` of the standard 4×4 (`.xyz` = linear
/// row, `.w` = translation component `k`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct Affine3x4 {
    /// `rows[k]` = the WGSL `mat3x4` column `k` = 4×4 row `k`. Same packing as
    /// `bevy_pbr`'s `MeshUniform::world_from_local`.
    pub rows: [Vec4; 3],
}

const _: () = assert!(size_of::<Affine3x4>() as u64 == AFFINE_STRIDE);

impl Affine3x4 {
    /// Pack a `GlobalTransform`'s affine into the GPU `mat3x4`, reusing
    /// bevy_math's canonical transpose-for-GPU-packing ([`Affine3Ext::to_transpose`])
    /// — the same path `MeshUniform::world_from_local` takes. Goes straight
    /// `Affine3A` → packed, with no `Mat4` intermediate.
    pub fn from_transform(world_from_local: Affine3A) -> Self {
        Self {
            rows: Affine3::from(world_from_local).to_transpose(),
        }
    }
}

/// Current world transforms (cluster scene + raytracing groups read this).
/// `KEEP_PREVIOUS`: the previous-frame transform (for ReSTIR temporal reuse)
/// is shifted in on the GPU from the current buffer — no separate upload.
///
/// Delta-direct: the per-frame `(slot, world)` records are pre-built by the
/// extract straight from the packed move data ([`InstanceManager::transform_delta`]),
/// so this column never gathers values out of a CPU mirror — `prebuilt_delta`
/// is uploaded verbatim. The GPU buffer is the source of truth across a
/// capacity growth (`GpuColumn` copies it old→new), so there is no CPU mirror.
pub struct TransformColumn;
impl GpuColumnDesc for TransformColumn {
    type Value = Affine3x4;
    type Table = InstanceManager;
    const LABEL: &'static str = "gpu_instances.transforms";
    const KEEP_PREVIOUS: bool = true;
    // Scene-columns group: `transforms` @binding(0), `previous_frame_transforms` @binding(1).
    const SCENE_BINDING: Option<u32> = Some(0);
    const SCENE_BINDING_PREVIOUS: Option<u32> = Some(1);
    /// No CPU delta — this column is written GPU-side by the transform-gather
    /// pass (`world[node_slot[i]]` → current, shifting current → previous). The
    /// `GpuColumn` still owns the current + previous buffers and grows them with
    /// the instance high-water; the empty delta just means it never scatters.
    fn delta_records(_: &InstanceManager) -> &[u32] {
        &[]
    }
}

/// Per-instance resolved `material_id` (the index into `materials[]`). Its
/// delta is built when [`InstanceManager::resolve_material_ids`] resolves a
/// changed slot — not on every move (a plain transform move keeps the same
/// material).
pub struct MaterialColumn;
impl GpuColumnDesc for MaterialColumn {
    type Value = u32;
    type Table = InstanceManager;
    const LABEL: &'static str = "gpu_instances.material_ids";
    /// Scene-columns group: `material_ids` @binding(2).
    const SCENE_BINDING: Option<u32> = Some(2);
    fn delta_records(m: &InstanceManager) -> &[u32] {
        m.material_delta()
    }
}

/// Per-instance group base (cluster scene group). Bind-only, **reconcile-written**:
/// the GPU reconcile pass ([`crate::ecs_gpu::reconcile`]) is its sole writer (fed by
/// the journal `UPSERT`), so the CPU delta is empty — `GpuColumn` still owns/grows the
/// buffer, the empty delta just means it never CPU-scatters (like [`TransformColumn`]).
pub struct GroupBaseColumn;
impl GpuColumnDesc for GroupBaseColumn {
    type Value = u32;
    type Table = InstanceManager;
    const LABEL: &'static str = "gpu_instances.group_bases";
    fn delta_records(_: &InstanceManager) -> &[u32] {
        &[]
    }
}

/// Per-instance packed LOD inputs (selector reads these). Bind-only,
/// **reconcile-written** (see [`GroupBaseColumn`]).
pub struct LodInputColumn;
impl GpuColumnDesc for LodInputColumn {
    type Value = InstanceLodInputGpu;
    type Table = InstanceManager;
    const LABEL: &'static str = "gpu_instances.lod_inputs";
    /// Scene-columns group: `instance_cluster_ranges` @binding(4) (the path tracer
    /// reads `(cluster_base, cluster_count)` from the LOD-input struct).
    const SCENE_BINDING: Option<u32> = Some(4);
    fn delta_records(_: &InstanceManager) -> &[u32] {
        &[]
    }
}

/// Per-instance geometry id (BLAS sharing + PTLAS fill read these). Bind-only,
/// **reconcile-written** (see [`GroupBaseColumn`]).
pub struct GeometryIdColumn;
impl GpuColumnDesc for GeometryIdColumn {
    type Value = u32;
    type Table = InstanceManager;
    const LABEL: &'static str = "gpu_instances.geometry_ids";
    fn delta_records(_: &InstanceManager) -> &[u32] {
        &[]
    }
}

/// Per-instance 8-bit ray-tracing cull mask (low byte used), derived from the
/// entity's `RenderLayers`. PTLAS fill writes it into the instance `mask`, so a
/// camera's `cullMask` hides non-matching instances for free during BVH
/// traversal. Written at bind and whenever `RenderLayers` changes.
pub struct InstanceMaskColumn;
impl GpuColumnDesc for InstanceMaskColumn {
    type Value = u32;
    type Table = InstanceManager;
    const LABEL: &'static str = "gpu_instances.instance_masks";
    fn delta_records(m: &InstanceManager) -> &[u32] {
        m.instance_mask_delta()
    }
}

/// Per-instance PTLAS regular-partition hint (`PARTITION_HINT_NONE` = derive
/// from the static flag). Bind-only, **reconcile-written** — the CPU assigns
/// spatially-tight ids (one per streamed cell) via [`super::SolariPartition`].
pub struct PartitionColumn;
impl GpuColumnDesc for PartitionColumn {
    type Value = u32;
    type Table = InstanceManager;
    const LABEL: &'static str = "gpu_instances.partition_hints";
    fn delta_records(_: &InstanceManager) -> &[u32] {
        &[]
    }
}

/// Per-instance transform-table node slot — the foreign key bridging an RT
/// instance to its node in the [`TransformGraph`](crate::transform::TransformGraph).
/// Bind-only (an entity's `GpuSlot` is stable). The transform-gather pass reads it
/// to copy `world[node_slot]` into this instance's `TransformColumn` entry.
pub struct NodeSlotColumn;
impl GpuColumnDesc for NodeSlotColumn {
    type Value = u32;
    type Table = InstanceManager;
    const LABEL: &'static str = "gpu_instances.node_slots";
    /// Bind-only, **reconcile-written** (see [`GroupBaseColumn`]).
    fn delta_records(_: &InstanceManager) -> &[u32] {
        &[]
    }
}

/// Ergonomic read-only view of all instance columns for a consumer system —
/// a `SystemParam` grouping, not a resource, so the columns stay separate
/// (per-column prepare parallelism preserved) and one param fetches them all.
#[derive(SystemParam)]
pub struct InstanceColumns<'w> {
    /// Current transforms; `previous_buffer()` is the previous frame's (GPU
    /// double-buffered, no separate column).
    pub transforms: Res<'w, GpuColumn<TransformColumn>>,
    pub material_ids: Res<'w, GpuColumn<MaterialColumn>>,
    pub group_bases: Res<'w, GpuColumn<GroupBaseColumn>>,
    pub lod_inputs: Res<'w, GpuColumn<LodInputColumn>>,
    pub geometry_ids: Res<'w, GpuColumn<GeometryIdColumn>>,
    pub instance_masks: Res<'w, GpuColumn<InstanceMaskColumn>>,
    /// Per-instance transform-table node slot (transform-gather reads this).
    pub node_slots: Res<'w, GpuColumn<NodeSlotColumn>>,
}

/// Registers every per-instance GPU column as its own `GpuColumnPlugin` —
/// each gets an independent resource + parallel prepare/scatter systems.
pub struct GpuInstancesPlugin;

impl Plugin for GpuInstancesPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            GpuColumnPlugin::<TransformColumn>::default(),
            GpuColumnPlugin::<MaterialColumn>::default(),
            GpuColumnPlugin::<GroupBaseColumn>::default(),
            GpuColumnPlugin::<LodInputColumn>::default(),
            GpuColumnPlugin::<GeometryIdColumn>::default(),
            GpuColumnPlugin::<InstanceMaskColumn>::default(),
            GpuColumnPlugin::<NodeSlotColumn>::default(),
            GpuColumnPlugin::<PartitionColumn>::default(),
        ));
    }
}
