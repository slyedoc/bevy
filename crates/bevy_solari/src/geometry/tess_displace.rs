//! Main-world discovery of displacement-mapped instances + the PTLAS-write pass
//! that injects their GPU-tessellated versions into the trace.
//!
//! [`find_tess_showcase_instances`] collects every displacement-mapped instance
//! in the main world (mesh + world transform + its own displacement map) and
//! extracts it to the render world; [`hide_tessellated_base_instances`] masks the
//! original flat cluster instance so only the tessellated version renders. The
//! GPU tessellation itself lives in `tess_classify`; here the PTLAS-write
//! ([`prepare_tess_ptlas_write`]) reads the per-instance BLAS addresses it
//! built and appends one PTLAS record each.

#![allow(unsafe_code)]

use bevy_asset::AssetId;
use bevy_asset::{Assets, Handle};
use bevy_ecs::{
    entity::{Entity, EntityHashMap},
    query::With,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_image::Image;
use bevy_math::Vec4;
use bevy_render::{
    extract_resource::ExtractResource,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
};
use bytemuck::{Pod, Zeroable};

use super::asset::ClusterMesh;
use crate::bindings::RaytracingMesh3d;
use crate::gpu::allocator::Allocator;
use crate::gpu::binding_seam::BindingSeam;
use crate::gpu::heap_kernel::{HeapKernel, KernelSlots};
use crate::material::{SolariMaterial3d, StandardSolariMaterial};

/// One real displacement-mapped scene instance the tessellation path tessellates
/// in place: its mesh, its own displacement map, and its world transform.
#[derive(Clone)]
pub struct TessShowcaseInstanceData {
    pub mesh: AssetId<ClusterMesh>,
    /// The instance's material — its SBT hit-group record (= material slot) so the
    /// tessellated hit shades with this surface's textures, not material 0's.
    pub material: AssetId<StandardSolariMaterial>,
    /// The source entity — resolved to its cluster slot at PTLAS-write time so the
    /// tess hit reads that entity's real previous-frame transform (correct motion
    /// vectors / no DLSS flicker), rather than assuming the surface is static.
    pub entity: Entity,
    pub displacement: Handle<Image>,
    /// Object-space AABB (from the `ClusterMesh`) — the PTLAS write derives the
    /// instance's explicit world AABB from it via `transforms[instance_id]`.
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
/// latched once all materials are loaded.
pub fn find_tess_showcase_instances(
    mut found: ResMut<TessShowcaseInstances>,
    materials: Res<Assets<StandardSolariMaterial>>,
    cluster_meshes: Res<Assets<ClusterMesh>>,
    query: Query<(Entity, &SolariMaterial3d, &RaytracingMesh3d)>,
) {
    if found.found {
        return;
    }
    // Wait until the scene has spawned (an empty query is "not loaded yet", NOT "no
    // displacement") and every instance's material + cluster mesh is loaded, so the
    // full set (and its AABBs) is collected at once (the `.bsn` spawns atomically).
    if query.is_empty()
        || query.iter().any(|(_, m, mesh)| {
            materials.get(&m.0).is_none() || cluster_meshes.get(&mesh.0).is_none()
        })
    {
        return;
    }

    let mut instances = Vec::new();
    for (entity, mat3d, mesh3d) in &query {
        let mat = materials.get(&mat3d.0).expect("checked loaded above");
        let Some(displacement) = mat.depth_map.clone() else {
            continue;
        };
        // Object-space AABB (checked loaded above) → the write shader derives the
        // explicit world AABB from it via `transforms[instance_id]`.
        let aabb = cluster_meshes
            .get(&mesh3d.0)
            .expect("checked loaded above")
            .aabb();
        let local_aabb_center = [aabb.center[0], aabb.center[1], aabb.center[2]];
        let local_aabb_half = [
            aabb.half_extent[0],
            aabb.half_extent[1],
            aabb.half_extent[2],
        ];
        instances.push(TessShowcaseInstanceData {
            mesh: mesh3d.0.id(),
            material: mat3d.0.id(),
            entity,
            displacement,
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

/// Marker: material inspected once — keeps the scan query archetype-empty in
/// steady state. A `depth_map` added at runtime won't re-hide the instance.
#[derive(bevy_ecs::component::Component)]
pub struct TessBaseChecked;

/// `Update` (main world): hide the original flat cluster instance of every
/// displacement-mapped entity by setting its RT cull mask to 0 (`RenderLayers::none()`),
/// so only the tessellated, displaced version renders. Without this the flat base mesh
/// occludes the (recessed) tessellation. The `ClusterMesh` asset still uploads, so the
/// tessellation can read the base geometry.
pub fn hide_tessellated_base_instances(
    mut commands: Commands,
    materials: Res<Assets<StandardSolariMaterial>>,
    query: Query<
        (Entity, &SolariMaterial3d),
        (
            With<RaytracingMesh3d>,
            bevy_ecs::query::Without<TessBaseChecked>,
        ),
    >,
) {
    for (entity, mat3d) in &query {
        // Not loaded yet — stays unmarked, retried next frame.
        let Some(mat) = materials.get(&mat3d.0) else {
            continue;
        };
        if mat.depth_map.is_some() {
            commands.entity(entity).insert((
                bevy_camera::visibility::RenderLayers::none(),
                TessBaseHidden,
                TessBaseChecked,
            ));
        } else {
            commands.entity(entity).insert(TessBaseChecked);
        }
    }
}

// ── PTLAS injection (mirrors `hair/ptlas_hair.rs`) ────────────────────────────

/// Push params shared with `tess_ptlas_write.slang::TessWriteParams`;
/// the per-instance transform / AABB / BLAS ride in the `instances` buffer.
#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
pub struct TessWriteParams {
    pub tess_count: u32,
    pub tess_base: u32,
    pub sbt_record: u32,
    pub mask: u32,
    /// PTLAS partition: `0` = static bulk, `0xffffffff` = global (diagnostic).
    pub partition_index: u32,
}

/// One injected tessellated instance (mirrors `tess_ptlas_write.slang::TessInstance`).
/// The BLAS is baked in OBJECT space, so both the PTLAS instance transform and the
/// explicit AABB the partitioned build requires (a zero/NaN BLAS-derived AABB hangs
/// it) are computed GPU-side in the write shader from `transforms[instance_id]` (the
/// origin-relative `world_rel` column) — so they track the floating origin, rather
/// than being baked at absolute CPU coords.
#[derive(Copy, Clone, Default, ShaderType)]
pub struct TessInstanceGpu {
    /// Object-space AABB center / half-extent (half is pre-inflated by the
    /// displacement margin). `.w` unused.
    pub aabb_center: Vec4,
    pub aabb_half: Vec4,
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

/// Render-world resource: the tessellation PTLAS-write heap kernel, shared
/// params + the per-instance buffer. Self-contained, so it touches neither
/// `SolariPipelines` nor `SolariResourceManager`.
#[derive(Resource)]
pub struct TessPtlasWrite {
    pub params: TessWriteParams,
    pub instances: StorageBuffer<Vec<TessInstanceGpu>>,
    pub tess_count: u32,
    pub kernel: HeapKernel,
    pub slots: KernelSlots,
    pub raw_device: ash::Device,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for TessPtlasWrite {
    fn drop(&mut self) {
        self._device_keepalive.quiesce_before_raw_destroy();
        // SAFETY: quiesced; handles exclusively owned here.
        unsafe { self.kernel.destroy(&self.raw_device) };
    }
}

// SAFETY: plain Vulkan handles; used solely from the render schedule.
unsafe impl Send for TessPtlasWrite {}
unsafe impl Sync for TessPtlasWrite {}

/// `RenderStartup` (after `SolariSetup`): compile the PTLAS-write kernel — a
/// layout-free heap pipeline ([`HeapKernel`]), Slang from source.
pub fn init_tess_ptlas_write(
    mut commands: Commands,
    seam: Option<Res<BindingSeam>>,
    allocator: Option<Res<Allocator>>,
) {
    let (Some(seam), Some(allocator)) = (seam, allocator) else {
        return;
    };
    let Some(kernel) = HeapKernel::new(
        &seam,
        "tess_ptlas_write.slang",
        include_str!("tess_ptlas_write.slang"),
        "tess_write",
        &[],
        &[],
        "tess_ptlas_write",
        size_of::<TessWriteParams>() as u32,
    ) else {
        return;
    };
    let slots = KernelSlots::new(&seam, 5);
    let mut instances = StorageBuffer::<Vec<TessInstanceGpu>>::default();
    instances.set_label(Some("tess_ptlas_write.instances"));
    commands.insert_resource(TessPtlasWrite {
        params: TessWriteParams::default(),
        instances,
        tess_count: 0,
        kernel,
        slots,
        raw_device: allocator.device().clone(),
        _device_keepalive: allocator.clone(),
    });
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
    settings: Res<crate::SolariSettings>,
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

    // Displaced micro-verts recede up to `displacement_scale` along the normal
    // (object units, applied pre-transform in the gen pass) — inflate the local box
    // by that margin so the explicit bounds stay conservative.
    let margin = settings.tess_displacement_scale;

    let gpu_instances: Vec<TessInstanceGpu> = found
        .instances
        .iter()
        .enumerate()
        .map(|(i, inst)| {
            // The GPU gen pass bakes micro-vertices in OBJECT space, so the PTLAS
            // record places each per-instance BLAS with `transforms[instance_id]`
            // (the origin-relative `world_rel` column), read GPU-side in the write
            // shader. The explicit world AABB is derived there from that same
            // transform (8 corners of the local box), so geometry and bounds stay
            // in one frame.
            let c = inst.local_aabb_center;
            let h = inst.local_aabb_half;
            TessInstanceGpu {
                aabb_center: Vec4::new(c[0], c[1], c[2], 0.0),
                // Pre-inflate the half-extent by the displacement margin (micro-verts
                // recede up to `displacement_scale` along the normal).
                aabb_half: Vec4::new(h[0] + margin, h[1] + margin, h[2] + margin, 0.0),
                // BLAS address read GPU-side from `blas_addresses[i]` (same instance order
                // the BLAS-build loop used).
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

    write.params = TessWriteParams {
        tess_count,
        tess_base: cluster_high_water + hair_count,
        // Unused by the write shader — each instance carries its own `sbt_record`.
        sbt_record: 0,
        mask: 0xff,
        // Static partition (0) — what `ptlas_fill::resolve_partition` assigns static
        // cluster instances (the global partition hangs the build with a lone occupant).
        partition_index: 0,
    };
}
