//! Bind-group **layout** builders for the acceleration-structure passes
//! (selector, BLAS sharing, PTLAS fill).
//!
//! The layouts span three pass-param types, so they're grouped here rather than
//! co-located one-per-file. They're owned by
//! [`SolariResourceManager`](crate::resource_manager::SolariResourceManager) (which
//! calls these), and the matching pipeline ids by
//! [`SolariPipelines`](crate::pipelines::SolariPipelines) — the meshlet split.

use bevy_render::{
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
        BindGroupLayoutDescriptor, BindGroupLayoutEntries, ShaderStages,
    },
    view::ViewUniform,
};

use super::blas_sharing::SharingParamsGpu;
use super::ptlas::PtlasFillParamsGpu;
use super::selector::SelectorParamsGpu;

/// The cluster-selector `@group(1)` layout.
pub fn selector_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "cluster_selector_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 cluster_clas_addresses
                storage_buffer_sized(false, None),           // 1 selected_clas_refs
                storage_buffer_sized(false, None),           // 2 args_buf
                storage_buffer_sized(false, None),           // 3 per_bucket_counts (atomic)
                uniform_buffer::<SelectorParamsGpu>(false),  // 4 params
                storage_buffer_read_only_sized(false, None), // 5 build_desc
                storage_buffer_read_only_sized(false, None), // 6 dirty_build_count
            ),
        ),
    )
}

/// The BLAS-sharing `@group(1)` layout (shared by all five sharing passes).
pub fn blas_sharing_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "blas_sharing_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<ViewUniform>(true),         // 0 view (dynamic)
                uniform_buffer::<SharingParamsGpu>(false),   // 1 params
                storage_buffer_read_only_sized(false, None), // 2 active_to_slot
                storage_buffer_read_only_sized(false, None), // 3 instance_geometry_ids
                storage_buffer_sized(false, None),           // 4 geometry_desired_level (atomic)
                storage_buffer_sized(false, None),           // 5 geometry_built_level
                storage_buffer_sized(false, None),           // 6 geometry_dirty
                storage_buffer_sized(false, None),           // 7 dirty_count (atomic)
                storage_buffer_sized(false, None),           // 8 dirty_gid
                storage_buffer_sized(false, None),           // 9 build_desc
                storage_buffer_sized(false, None),           // 10 geometry_dst_addresses
                storage_buffer_sized(false, None),           // 11 instance_blas_address
                storage_buffer_sized(false, None),           // 12 build_count
                storage_buffer_sized(false, None),           // 13 geometry_desc
                storage_buffer_sized(false, None),           // 14 instance_e_build
            ),
        ),
    )
}

/// The PTLAS-fill `@group(1)` layout (shared by seed / incremental / finalize).
pub fn ptlas_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "ptlas_fill_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // 0 instance_blas_address
                storage_buffer_read_only_sized(false, None), // 1 geometry_dirty
                storage_buffer_sized(false, None),           // 2 write_count (atomic)
                storage_buffer_sized(false, None),           // 3 write_data
                storage_buffer_read_only_sized(false, None), // 4 write_slots_cpu
                storage_buffer_read_only_sized(false, None), // 5 active_to_slot
                storage_buffer_sized(false, None),           // 6 src_infos
                uniform_buffer::<PtlasFillParamsGpu>(false), // 7 params
                storage_buffer_read_only_sized(false, None), // 8 instance_geometry_ids
                storage_buffer_read_only_sized(false, None), // 9 instance_masks
                storage_buffer_read_only_sized(false, None), // 10 instance_previous_transforms
            ),
        ),
    )
}
