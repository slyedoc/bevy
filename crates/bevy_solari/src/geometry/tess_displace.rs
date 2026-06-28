//! Main-world discovery of displacement-mapped instances + the PTLAS-write pass
//! that injects their GPU-tessellated versions into the trace.
//!
//! [`find_tess_showcase_instances`] collects every displacement-mapped instance
//! in the main world (mesh + world transform + its own displacement map) and
//! extracts it to the render world; [`hide_tessellated_base_instances`] masks the
//! original flat cluster instance so only the tessellated version renders. The
//! GPU tessellation itself lives in `tess_classify`; here the PTLAS-write
//! ([`prepare_tess_ptlas_write`] / [`prepare_tess_ptlas_write_bind_group`]) reads
//! the per-instance BLAS addresses it built and appends one PTLAS record each.

use bevy_asset::AssetId;
use bevy_ecs::{
    entity::{Entity, EntityHashMap},
    query::With,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_image::Image;
use bevy_transform::components::Transform;
use bevy_render::{
    extract_resource::ExtractResource,
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
};
use bevy_asset::{Assets, Handle};
use bevy_math::Vec4;

use crate::bindings::RaytracingMesh3d;
use crate::gpu::allocator::Allocator;
use crate::material::{SolariMaterial, SolariMaterial3d};
use super::asset::ClusterMesh;

/// One real displacement-mapped scene instance the tessellation path tessellates
/// in place: its mesh, its own displacement map, and its world transform.
#[derive(Clone)]
pub struct TessShowcaseInstanceData {
    pub mesh: AssetId<ClusterMesh>,
    /// The instance's material — its SBT hit-group record (= material slot) so the
    /// tessellated hit shades with this surface's textures, not material 0's.
    pub material: AssetId<SolariMaterial>,
    /// The source entity — resolved to its cluster slot at PTLAS-write time so the
    /// tess hit reads that entity's real previous-frame transform (correct motion
    /// vectors / no DLSS flicker), rather than assuming the surface is static.
    pub entity: Entity,
    pub displacement: Handle<Image>,
    /// World-from-local affine, row-major 3×4 (`TransformMatrixKHR` layout).
    pub world_from_local: [[f32; 4]; 3],
    /// Object-space AABB (from the `ClusterMesh`) — the PTLAS write derives the
    /// instance's explicit world AABB from it (`world_aabb`).
    pub local_aabb_center: [f32; 3],
    pub local_aabb_half: [f32; 3],
}

/// All displacement-mapped instances the tessellation path tessellates, found once
/// in the main world by [`find_tess_showcase_instances`] and extracted to the
/// render world. Each is subdivided at its own world transform with its own map.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct TessShowcaseInstances {
    pub found: bool,
    pub instances: Vec<TessShowcaseInstanceData>,
}

/// `Update` (main world): collect EVERY displacement-mapped instance as a
/// tessellation subject (mesh + world transform + its own displacement map),
/// latched once all materials are loaded. Gated on `SOLARI_TESS`.
pub fn find_tess_showcase_instances(
    mut found: ResMut<TessShowcaseInstances>,
    materials: Res<Assets<SolariMaterial>>,
    cluster_meshes: Res<Assets<ClusterMesh>>,
    query: Query<(Entity, &SolariMaterial3d, &RaytracingMesh3d, &Transform)>,
) {
    if found.found {
        return;
    }
    // Wait until the scene has spawned (an empty query is "not loaded yet", NOT "no
    // displacement") and every instance's material + cluster mesh is loaded, so the
    // full set (and its AABBs) is collected at once (the `.bsn` spawns atomically).
    if query.is_empty()
        || query.iter().any(|(_, m, mesh, _)| {
            materials.get(&m.0).is_none() || cluster_meshes.get(&mesh.0).is_none()
        })
    {
        return;
    }

    let mut instances = Vec::new();
    for (entity, mat3d, mesh3d, transform) in &query {
        let mat = materials.get(&mat3d.0).expect("checked loaded above");
        let Some(displacement) = mat.depth_map.clone() else {
            continue;
        };
        // Solari disables `TransformPlugin`, so CPU `GlobalTransform` is dead
        // (identity) — the real placement lives in `Transform`. The `.bsn` is a flat
        // root→children hierarchy with an identity root, so the local `Transform`
        // IS the world transform. Row-major 3×4: world = matrix3 * local +
        // translation; each row is (basis_row_r, translation_r).
        let a = transform.compute_affine();
        let (m, t) = (a.matrix3, a.translation);
        let world_from_local = [
            [m.x_axis.x, m.y_axis.x, m.z_axis.x, t.x],
            [m.x_axis.y, m.y_axis.y, m.z_axis.y, t.y],
            [m.x_axis.z, m.y_axis.z, m.z_axis.z, t.z],
        ];
        // Object-space AABB (checked loaded above) → drives the explicit world AABB.
        let aabb = cluster_meshes.get(&mesh3d.0).expect("checked loaded above").aabb();
        let local_aabb_center = [aabb.center[0], aabb.center[1], aabb.center[2]];
        let local_aabb_half = [aabb.half_extent[0], aabb.half_extent[1], aabb.half_extent[2]];
        instances.push(TessShowcaseInstanceData {
            mesh: mesh3d.0.id(),
            material: mat3d.0.id(),
            entity,
            displacement,
            world_from_local,
            local_aabb_center,
            local_aabb_half,
        });
    }
    if instances.is_empty() {
        // No displacement-mapped instances in this scene — latch so we stop scanning
        // every frame (the `.bsn` spawns atomically, so the set won't grow later).
        found.found = true;
        return;
    }
    bevy_log::debug!(
        "tess showcase: {} displacement-mapped instances found",
        instances.len(),
    );
    found.instances = instances;
    found.found = true;
}

/// Marker: this entity's original (flat) cluster instance has been masked out of the
/// trace — its tessellated version (injected separately) renders instead.
#[derive(bevy_ecs::component::Component)]
pub struct TessBaseHidden;

/// `Update` (main world): hide the original flat cluster instance of every
/// displacement-mapped entity by setting its RT cull mask to 0 (`RenderLayers::none()`),
/// so only the tessellated, displaced version renders. Without this the flat base mesh
/// occludes the (recessed) tessellation. The `ClusterMesh` asset still uploads, so the
/// tessellation can read the base geometry.
pub fn hide_tessellated_base_instances(
    mut commands: Commands,
    materials: Res<Assets<SolariMaterial>>,
    query: Query<
        (Entity, &SolariMaterial3d),
        (With<RaytracingMesh3d>, bevy_ecs::query::Without<TessBaseHidden>),
    >,
) {
    for (entity, mat3d) in &query {
        let Some(mat) = materials.get(&mat3d.0) else {
            continue;
        };
        if mat.depth_map.is_some() {
            commands
                .entity(entity)
                .insert((bevy_camera::visibility::RenderLayers::none(), TessBaseHidden));
        }
    }
}

// ── PTLAS injection (mirrors `hair/ptlas_hair.rs`) ────────────────────────────

/// Shared PTLAS-write params (mirrors `tess_ptlas_write.wgsl::TessWriteParams`);
/// the per-instance transform / AABB / BLAS ride in the `instances` buffer.
#[repr(C)]
#[derive(Copy, Clone, Default, ShaderType)]
pub struct TessWriteParams {
    pub tess_count: u32,
    pub tess_base: u32,
    pub sbt_record: u32,
    pub mask: u32,
    /// PTLAS partition: `0` = static bulk, `0xffffffff` = global (diagnostic).
    pub partition_index: u32,
}

/// One injected tessellated instance (mirrors `tess_ptlas_write.wgsl::TessInstance`).
/// Transform as 3 `vec4` rows (mat3x4); explicit world AABB so the partitioned
/// build never derives bounds from the BLAS (a zero/NaN derived AABB hangs it).
#[derive(Copy, Clone, Default, ShaderType)]
pub struct TessInstanceGpu {
    pub transform_r0: Vec4,
    pub transform_r1: Vec4,
    pub transform_r2: Vec4,
    pub aabb_min: Vec4,
    pub aabb_max: Vec4,
    /// Index into the GPU `blas_addresses` buffer — the PTLAS write reads the BLAS
    /// device address there, GPU-side (no CPU readback).
    pub blas_slot: u32,
    /// SBT hit-group record (= this instance's material slot) so the tessellated hit
    /// shades with its own material's textures.
    pub sbt_record: u32,
    /// Cluster slot of the source entity — the closest-hit indexes
    /// `previous_frame_transforms` by this for correct motion vectors (the surface's
    /// real previous transform), not a guessed-static 0.
    pub instance_id: u32,
}

/// Render-world resource: the tessellation PTLAS-write pipeline, shared params +
/// the per-instance buffer + bind group. Self-contained (its own pipeline id), so
/// it touches neither `SolariPipelines` nor `SolariResourceManager`.
#[derive(Resource)]
pub struct TessPtlasWrite {
    pub pipeline: CachedComputePipelineId,
    pub layout: BindGroupLayoutDescriptor,
    pub params: UniformBuffer<TessWriteParams>,
    pub instances: StorageBuffer<Vec<TessInstanceGpu>>,
    pub tess_count: u32,
    pub bind_group: Option<BindGroup>,
}

/// The tessellation PTLAS-write `@group(0)` layout: the shared PTLAS record
/// buffers + the params UBO + the per-instance buffer (one record per tessellated
/// instance, like hair's per-instance buffer).
fn tess_ptlas_write_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "tess_ptlas_write",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_sized(false, None),           // 0 write_count (rw atomic)
                storage_buffer_sized(false, None),           // 1 write_data (rw)
                uniform_buffer::<TessWriteParams>(false),    // 2 params
                storage_buffer_read_only_sized(false, None), // 3 instances
                storage_buffer_read_only_sized(false, None), // 4 blas_addresses
            ),
        ),
    )
}

/// `RenderStartup`: queue the PTLAS-write pipeline + the params buffer.
pub fn init_tess_ptlas_write(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
    allocator: Option<Res<Allocator>>,
) {
    if allocator.is_none() {
        return;
    }
    let layout = tess_ptlas_write_layout();
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("tess_ptlas_write".into()),
        layout: vec![layout.clone()],
        // Path is relative to THIS file's dir (`src/geometry/`); the
        // `embedded_asset!` in `pipelines.rs` (at `src/`) registered it as
        // `geometry/tess_ptlas_write.wgsl`, so loading from here uses the bare name.
        shader: bevy_asset::load_embedded_asset!(asset_server.as_ref(), "tess_ptlas_write.wgsl"),
        shader_defs: vec![],
        entry_point: Some("tess_write".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    let mut params = UniformBuffer::<TessWriteParams>::default();
    params.set_label(Some("tess_ptlas_write"));
    let mut instances = StorageBuffer::<Vec<TessInstanceGpu>>::default();
    instances.set_label(Some("tess_ptlas_write.instances"));
    commands.insert_resource(TessPtlasWrite {
        pipeline,
        layout,
        params,
        instances,
        tess_count: 0,
        bind_group: None,
    });
}

/// World AABB of a local center/half-extent under a row-major 3×4 transform (the 8
/// transformed corners' min/max). Explicit bounds the partitioned PTLAS build requires.
fn world_aabb(
    world_from_local: &[[f32; 4]; 3],
    center: [f32; 3],
    half: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let t = world_from_local;
    let mut wmin = [f32::INFINITY; 3];
    let mut wmax = [f32::NEG_INFINITY; 3];
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                let p = [
                    center[0] + sx * half[0],
                    center[1] + sy * half[1],
                    center[2] + sz * half[2],
                ];
                for r in 0..3 {
                    let w = t[r][0] * p[0] + t[r][1] * p[1] + t[r][2] * p[2] + t[r][3];
                    wmin[r] = wmin[r].min(w);
                    wmax[r] = wmax[r].max(w);
                }
            }
        }
    }
    (wmin, wmax)
}

/// `Render::Prepare`: set the PTLAS-write params from the armed tessellation set. The
/// tess instance occupies PTLAS index `cluster_high_water + hair_count` (above
/// the cluster slots + hair), matching the `high_water` fold in
/// [`crate::accel::ptlas::prepare_ptlas_params`].
pub fn prepare_tess_ptlas_write(
    write: Option<ResMut<TessPtlasWrite>>,
    found: Option<Res<TessShowcaseInstances>>,
    classify: Option<Res<super::tess_classify::TessClassify>>,
    instances: Option<Res<crate::instance::InstanceManager>>,
    hair: Option<Res<crate::hair::HairInstances>>,
    material_slots: Option<Res<crate::material::MaterialSlots>>,
    // Synced render entities tagged with their cluster slot + the main entity they came
    // from — lets us map each tess instance's source entity to its slot.
    instance_slots: Query<(
        &bevy_render::sync_world::MainEntity,
        &crate::instance::RaytracingGpuEntity,
    )>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(mut write) = write else {
        return;
    };
    // GPU tessellation path: one PTLAS record per found displacement instance, injected
    // only once `tess_classify` has built the per-instance BLAS (`blas_ready`). Gated the
    // same way as `prepare_ptlas_params`'s `tess_count` so the slot reservation matches.
    let blas_ready = classify.as_ref().is_some_and(|c| c.blas_ready);
    let tess_count = match (&found, blas_ready) {
        (Some(f), true) if f.found => f.instances.len() as u32,
        _ => 0,
    };
    write.tess_count = tess_count;
    if tess_count == 0 {
        return;
    }
    let found = found.unwrap();
    let cluster_high_water = instances.as_ref().map_or(0, |i| i.slot_high_water());
    let hair_count = hair.as_ref().map_or(0, |h| h.count);

    // Main entity → cluster slot, so each tess instance reads its own previous-frame
    // transform (correct motion vectors) instead of slot 0's.
    let slot_of: EntityHashMap<u32> = instance_slots
        .iter()
        .map(|(main, slot)| (main.id(), slot.0 .0))
        .collect();

    let gpu_instances: Vec<TessInstanceGpu> = found
        .instances
        .iter()
        .enumerate()
        .map(|(i, inst)| {
            // The GPU gen pass bakes WORLD-space micro-vertices, so each per-instance BLAS
            // is already world-space → inject at IDENTITY (don't re-apply world_from_local).
            // The world AABB is `world_from_local · local_aabb` (8 corners) — the
            // partitioned PTLAS build needs explicit bounds (BLAS-derived hangs it).
            let (wmin, wmax) = world_aabb(
                &inst.world_from_local,
                inst.local_aabb_center,
                inst.local_aabb_half,
            );
            // DIAGNOSTIC: huge AABB to rule the explicit-bounds-mismatch hypothesis in/out.
            // If the GPU tess geometry appears with this, the per-instance world AABB
            // (wmin/wmax) was wrong; revert to wmin/wmax once confirmed.
            let _ = (wmin, wmax);
            TessInstanceGpu {
                transform_r0: Vec4::new(1.0, 0.0, 0.0, 0.0),
                transform_r1: Vec4::new(0.0, 1.0, 0.0, 0.0),
                transform_r2: Vec4::new(0.0, 0.0, 1.0, 0.0),
                aabb_min: Vec4::new(-10000.0, -10000.0, -10000.0, 0.0),
                aabb_max: Vec4::new(10000.0, 10000.0, 10000.0, 0.0),
                // BLAS address read GPU-side from `blas_addresses[i]` (same instance order
                // the step-3b loop built them in).
                blas_slot: i as u32,
                sbt_record: material_slots
                    .as_ref()
                    .and_then(|ms| ms.slot_of(inst.material))
                    .unwrap_or(0),
                instance_id: slot_of.get(&inst.entity).copied().unwrap_or(0),
            }
        })
        .collect();
    write.instances.set(gpu_instances);
    write.instances.write_buffer(&render_device, &render_queue);

    *write.params.get_mut() = TessWriteParams {
        tess_count,
        tess_base: cluster_high_water + hair_count,
        // Route to material 0's `chit_opaque` — geometry correct via position-fetch;
        // shading uses material 0's textures (smooth tess normals are recovered in the
        // chit; UV is fixed until textured tess shading lands).
        sbt_record: 0,
        mask: 0xff,
        // Static partition (0) — what `ptlas_fill::resolve_partition` assigns static
        // cluster instances (the global partition hangs the build with a lone occupant).
        partition_index: 0,
    };
    write.params.write_buffer(&render_device, &render_queue);
}

/// `Render::PrepareBindGroups`: build the PTLAS-write bind group from the shared
/// PTLAS record buffers + the params UBO.
pub fn prepare_tess_ptlas_write_bind_group(
    write: Option<ResMut<TessPtlasWrite>>,
    ptlas: Option<Res<crate::accel::ptlas::Ptlas>>,
    classify: Option<Res<super::tess_classify::TessClassify>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let Some(mut write) = write else {
        return;
    };
    let (fp, fi, fpt, fc) = (
        write.params.binding().is_some(),
        write.instances.binding().is_some(),
        ptlas.is_some(),
        classify.is_some(),
    );
    let (Some(ptlas), Some(params), Some(instances), Some(classify)) = (
        ptlas,
        write.params.binding(),
        write.instances.binding(),
        classify.as_ref(),
    ) else {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) % 180 == 0 {
            tracing::debug!(
                "tess_ptlas_write_bind_group: bail A — ptlas={fpt} params={fp} instances={fi} classify={fc}",
            );
        }
        write.bind_group = None;
        return;
    };
    // The GPU per-instance BLAS-address buffer (`tess_classify` step 3b); absent until the
    // CLAS pool is first sized, in which case there's nothing to inject yet.
    let Some(blas_addresses) = classify.blas_addresses.as_ref() else {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) % 180 == 0 {
            tracing::debug!("tess_ptlas_write_bind_group: bail B (classify.blas_addresses None)");
        }
        write.bind_group = None;
        return;
    };
    {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) % 180 == 0 {
            tracing::debug!("tess_ptlas_write_bind_group: BUILT (tess write dispatches)");
        }
    }
    let layout = pipeline_cache.get_bind_group_layout(&write.layout);
    write.bind_group = Some(render_device.create_bind_group(
        "tess_ptlas_write",
        &layout,
        &BindGroupEntries::sequential((
            ptlas.write_count.as_entire_binding(),
            ptlas.write_data.wgpu_buffer.as_entire_binding(),
            params,
            instances,
            blas_addresses.as_entire_binding(),
        )),
    ));
}
