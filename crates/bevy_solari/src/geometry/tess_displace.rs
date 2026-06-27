//! Phase 2c — the displacement compute pass (pipeline + dispatch).
//!
//! Owns the `geometry/tess_displace.wgsl` compute pipeline (a self-contained
//! `@group(0)`: cluster pools as plain wgpu storage + one displacement texture)
//! and a [`dispatch_tess_displace`] helper that fills a buffer of displaced
//! micro-vertex positions for `INSTANTIATE_TRIANGLE_CLUSTER` to consume.
//!
//! It also runs a one-shot **self-test** under the `SOLARI_TESS` env var: it
//! displaces a known unit triangle at a fixed subdivision level and submits the
//! dispatch, so the pass (shader compilation, the self-contained bind group,
//! `textureSampleLevel` in compute) can be validated on hardware in isolation —
//! before it's wired to real cluster geometry + the per-instance BLAS.

use bevy_asset::{AssetId, Assets, Handle};
use bevy_camera::Camera;
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
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::GpuImage,
};
use bytemuck::{Pod, Zeroable};
use ash::vk;
use bevy_math::{Vec3, Vec4};

use crate::bindings::RaytracingMesh3d;
use crate::gpu::allocator::{Allocator, MemoryLocation};
use crate::gpu::extension::ClusterExtensionFns;
use crate::material::{SolariMaterial, SolariMaterial3d};
use super::asset::ClusterMesh;
use super::mesh_manager::ClusterMeshManager;
use super::tess_template::{
    record_build_per_instance_blas, TessellationTemplates, MAX_TESS_LEVEL, TESS_CLUSTER_ID_BASE,
};
use super::tessellation::SubdividedTriangle;


/// The displacement (height) map the showcase patch samples. The app points this
/// at a real texture (san_miguel uses a floor displacement map); `None` falls back
/// to the built-in synthetic checker in the self-test. Extracted to the render
/// world so [`tess_displace_selftest`] can resolve its [`GpuImage`].
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct TessShowcaseDisplacement(pub Option<Handle<Image>>);

/// One real displacement-mapped scene instance the showcase tessellates in place:
/// its mesh, its own displacement map, and its world transform (natural scale).
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
    /// View-dependent subdivision level (1..=`MAX_TESS_LEVEL`), chosen from the
    /// instance's distance to the camera — closer surfaces tessellate denser. The
    /// "classify" step; computed once at the start camera for now (per-frame
    /// re-classification is a later brick).
    pub level: u32,
}

/// All displacement-mapped instances the showcase tessellates, found once in the
/// main world by [`find_tess_showcase_instances`] and extracted to the render
/// world. Each is subdivided at its own world transform with its own map.
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
    query: Query<(Entity, &SolariMaterial3d, &RaytracingMesh3d, &Transform)>,
    cameras: Query<&Transform, With<Camera>>,
) {
    if found.found {
        return;
    }
    // Wait until the scene has spawned (an empty query is "not loaded yet", NOT "no
    // displacement") and every instance's material is loaded, so the full set is
    // collected at once (the `.bsn` spawns all instances atomically; materials stream in).
    if query.is_empty() || query.iter().any(|(_, m, _, _)| materials.get(&m.0).is_none()) {
        return;
    }
    // View-dependent subdivision level: `SOLARI_TESS_LEVEL` (the cap) up close, falling
    // off with distance past `SOLARI_TESS_REFDIST`. `reclassify_tess_levels` keeps it
    // live as the camera moves. (No camera ⇒ the flat LEVEL.)
    let base_level = env_u32("SOLARI_TESS_LEVEL", 12).clamp(1, MAX_TESS_LEVEL);
    let ref_dist = tess_ref_dist();
    let cam_pos = cameras.iter().next().map(|t| t.translation);

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
        let level = match cam_pos {
            Some(c) => tess_level_continuous(
                base_level,
                ref_dist,
                (c - transform.translation).length(),
            )
            .round() as u32,
            None => base_level,
        };
        instances.push(TessShowcaseInstanceData {
            mesh: mesh3d.0.id(),
            material: mat3d.0.id(),
            entity,
            displacement,
            world_from_local: [
                [m.x_axis.x, m.y_axis.x, m.z_axis.x, t.x],
                [m.x_axis.y, m.y_axis.y, m.z_axis.y, t.y],
                [m.x_axis.z, m.y_axis.z, m.z_axis.z, t.z],
            ],
            level,
        });
    }
    if instances.is_empty() {
        // No displacement-mapped instances in this scene — latch so we stop scanning
        // every frame (the `.bsn` spawns atomically, so the set won't grow later).
        found.found = true;
        return;
    }
    bevy_log::info!(
        "tess showcase: {} displacement-mapped instances found (view-dependent, max level {}, ref_dist {})",
        instances.len(),
        base_level,
        ref_dist,
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

/// View-dependent reference distance (`SOLARI_TESS_REFDIST`): the distance at which a
/// surface reaches its full `SOLARI_TESS_LEVEL`. Within it a surface stays at LEVEL
/// (the cap); beyond it the level falls off. Raise it toward "uniform LEVEL
/// everywhere"; default 20.
fn tess_ref_dist() -> f32 {
    std::env::var("SOLARI_TESS_REFDIST")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(20.0)
}

/// Continuous view-dependent level: `LEVEL` up close (`distance <= ref_dist`), then
/// `LEVEL * ref_dist / distance` beyond — clamped to `[1, LEVEL]` so LEVEL is the
/// achievable maximum (close-up), not a value far surfaces can exceed.
fn tess_level_continuous(base: u32, ref_dist: f32, distance: f32) -> f32 {
    ((base as f32) * ref_dist / distance.max(0.1)).clamp(1.0, base as f32)
}

/// `Update` (main world): RE-CLASSIFY per-instance subdivision levels from the
/// CURRENT camera each frame, with hysteresis so a surface near a level boundary
/// doesn't thrash. The render-world self-test re-tessellates only when a level
/// actually changes — so this is the "live" half of view-dependent tessellation.
/// Runs once the instance set is found.
pub fn reclassify_tess_levels(
    mut found: ResMut<TessShowcaseInstances>,
    cameras: Query<&Transform, With<Camera>>,
) {
    if !found.found {
        return;
    }
    let Some(cam) = cameras.iter().next().map(|t| t.translation) else {
        return;
    };
    let base = env_u32("SOLARI_TESS_LEVEL", 12).clamp(1, MAX_TESS_LEVEL);
    let ref_dist = tess_ref_dist();
    for inst in &mut found.instances {
        let pos = Vec3::new(
            inst.world_from_local[0][3],
            inst.world_from_local[1][3],
            inst.world_from_local[2][3],
        );
        let continuous = tess_level_continuous(base, ref_dist, (cam - pos).length());
        // Hysteresis: switch only when `continuous` is clearly past the current
        // level (±0.6) so a surface hovering at a boundary doesn't flip-flop (and
        // doesn't re-tessellate every frame while the camera barely moves).
        let want = continuous.round() as u32;
        if want != inst.level && (continuous - inst.level as f32).abs() > 0.6 {
            inst.level = want;
        }
    }
}

/// One tessellated instance armed for PTLAS injection: the index of its per-instance
/// BLAS address in the GPU `blas_addresses` buffer (the PTLAS write reads the address
/// there, GPU-side — no CPU readback), its world transform, and its world-space AABB
/// (from the mesh AABB + displacement margin, so the build never derives bounds from
/// the BLAS).
#[derive(Clone, Copy)]
pub struct TessShowcaseEntry {
    pub blas_slot: u32,
    /// This instance's material — resolved to its SBT record at PTLAS-write time.
    pub material: AssetId<SolariMaterial>,
    /// This instance's source entity — resolved to its cluster slot (for the correct
    /// previous-frame transform) at PTLAS-write time.
    pub entity: Entity,
    pub transform: [[f32; 4]; 3],
    pub world_aabb_min: [f32; 3],
    pub world_aabb_max: [f32; 3],
}

/// One armed showcase instance: its PTLAS entry, the subdivision level it was
/// last built at (so a level change re-tessellates only this instance), and the
/// buffers backing its displaced positions / CLAS / BLAS (held alive for the
/// trace — each BLAS references its CLAS storage references its positions).
struct TessShowcaseSlot {
    entry: TessShowcaseEntry,
    level: u32,
    /// Held only to keep the BLAS/CLAS/positions buffers alive for the trace;
    /// never read. Mixed `wgpu::Buffer` (allocator) + bevy `Buffer`, so boxed.
    _keepalive: Vec<Box<dyn core::any::Any + Send + Sync>>,
}

/// Persistent result of the showcase build (displace → instantiate → per-instance
/// BLAS) for EVERY displacement instance. One [`TessShowcaseSlot`] per instance,
/// indexed to match [`TessShowcaseInstances`]; re-tessellation replaces a single
/// slot, leaving the others — so one surface crossing a level boundary rebuilds
/// only itself, not the whole set. `entries` is the flat PTLAS-write view,
/// rebuilt from the live slots after any change.
#[derive(Resource, Default)]
pub struct TessShowcase {
    pub entries: Vec<TessShowcaseEntry>,
    /// `None` until the corresponding instance first tessellates.
    slots: Vec<Option<TessShowcaseSlot>>,
    /// Persistent GPU buffer of per-instance BLAS device addresses (a `u64` each),
    /// written by the BLAS builds (IMPLICIT dst at `slot * 8`) and read GPU-side by
    /// the PTLAS write. `0` = not (yet) built → inactive PTLAS instance. Sized to the
    /// instance count; never freed while the showcase lives.
    pub blas_addresses: Option<wgpu::Buffer>,
    /// Per-CLAS smooth-normal metadata ([`TessClusterMeta`] records) the closest-hit
    /// reaches via `geometry_addresses.tess_clusters`. `Some` only when the smooth-tess
    /// path (`SOLARI_TESS_SMOOTH`) is on. Per-instance fixed CLAS regions keep each
    /// CLAS's index stable across per-instance rebuilds. Allocator-backed for a stable
    /// trace device address.
    pub tess_clusters_meta: Option<wgpu::Buffer>,
    /// Deferred-drop ring for buffers an in-flight build or trace may still
    /// reference: each rebuild's replaced-slot trace buffers + its build transients.
    /// The old per-instance build's single `poll(Wait)` used to drain the GPU and
    /// make immediate frees safe; the async path drops that poll, so frees wait a few
    /// build generations (> frames-in-flight) instead. Index 0 = oldest.
    retiring: std::collections::VecDeque<Vec<Box<dyn core::any::Any + Send + Sync>>>,
}

/// Build generations to defer a buffer's drop by — comfortably above wgpu's default
/// maximum frame latency (2), so a buffer last used in build/trace generation `g` is
/// freed only once generation `g + RETIRE_GENERATIONS` is recorded, by which point
/// the GPU has long finished with it.
const RETIRE_GENERATIONS: usize = 4;

/// Mirrors `tess_displace.wgsl::TessDisplaceParams` (32 B). The compute is
/// dispatched per cluster, so the base mesh pool offsets ride here rather than
/// a `clusters[]` binding.
#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
pub struct TessDisplaceParams {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub base_triangle_count: u32,
    pub micro_vertex_count: u32,
    pub displacement_scale: f32,
    pub displacement_bias: f32,
    pub out_vertex_base: u32,
    pub has_displacement: u32,
    /// Recursive split factor `K²` (sub-triangles per base triangle); 1 = no split.
    pub split_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Render-world resource: the displacement compute pipeline + its self-contained
/// bind-group layout. Present only on a cluster-AS-capable device (gated on
/// [`Allocator`](crate::gpu::allocator::Allocator) at init, like the rest of the
/// geometry domain).
#[derive(Resource)]
pub struct TessDisplace {
    pub pipeline: CachedComputePipelineId,
    pub layout: BindGroupLayoutDescriptor,
}

/// `RenderStartup`: create the displacement layout + queue its compute pipeline.
pub fn init_tess_displace(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
    allocator: Option<Res<Allocator>>,
) {
    if allocator.is_none() {
        return;
    }
    let layout = BindGroupLayoutDescriptor::new(
        "tess_displace_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, None),         // 0 params
                storage_buffer_read_only_sized(false, None), // 1 cluster_indices
                storage_buffer_read_only_sized(false, None), // 2 vertex_positions
                storage_buffer_read_only_sized(false, None), // 3 vertex_packed
                storage_buffer_read_only_sized(false, None), // 4 tess_barycentrics
                storage_buffer_sized(false, None),         // 5 out_positions (rw)
                texture_2d(TextureSampleType::Float { filterable: true }), // 6
                sampler(SamplerBindingType::Filtering),    // 7
                storage_buffer_read_only_sized(false, None), // 8 split_corner_barys
            ),
        ),
    );

    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("tess_displace".into()),
        layout: vec![layout.clone()],
        // Path is relative to THIS file's dir (`src/geometry/`); the
        // `embedded_asset!` in `pipelines.rs` (at `src/`) registered it as
        // `geometry/tess_displace.wgsl`, so loading from here uses the bare
        // name — `"geometry/tess_displace.wgsl"` here would double the segment.
        shader: bevy_asset::load_embedded_asset!(asset_server.as_ref(), "tess_displace.wgsl"),
        shader_defs: vec![],
        entry_point: Some("displace".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    commands.insert_resource(TessDisplace { pipeline, layout });
    commands.insert_resource(TessShowcase::default());
}

/// Record a displacement dispatch into `encoder`: one thread per micro-vertex
/// (`base_triangle_count * micro_vertex_count`), `@workgroup_size(64)`.
///
/// All buffers are caller-owned: `params` (uniform), the base mesh
/// `cluster_indices` / `vertex_positions` / `vertex_packed` pools, the level's
/// `tess_barycentrics`, the displaced-position `out_positions` (read-write), and
/// the displacement `texture` + `sampler`. Returns `false` (no-op) if the
/// pipeline hasn't compiled yet.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tess_displace(
    encoder: &mut CommandEncoder,
    render_device: &RenderDevice,
    pipeline_cache: &PipelineCache,
    tess: &TessDisplace,
    params: &Buffer,
    cluster_indices: &Buffer,
    vertex_positions: &Buffer,
    vertex_packed: &Buffer,
    tess_barycentrics: &Buffer,
    // Allocator-backed (`wgpu::Buffer`) so its device address is 256-aligned for
    // the downstream instantiate; the other inputs are bevy `Buffer`s.
    out_positions: &wgpu::Buffer,
    texture: &TextureView,
    sampler: &Sampler,
    split_corner_barys: &Buffer,
    thread_count: u32,
) -> bool {
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(tess.pipeline) else {
        return false;
    };
    let bind_group = render_device.create_bind_group(
        "tess_displace_bind_group",
        &pipeline_cache.get_bind_group_layout(&tess.layout),
        &BindGroupEntries::sequential((
            params.as_entire_binding(),
            cluster_indices.as_entire_binding(),
            vertex_positions.as_entire_binding(),
            vertex_packed.as_entire_binding(),
            tess_barycentrics.as_entire_binding(),
            out_positions.as_entire_binding(),
            texture,
            sampler,
            split_corner_barys.as_entire_binding(),
        )),
    );
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("tess_displace"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(thread_count.div_ceil(64), 1, 1);
    true
}

/// Mirrors `tess_normals.wgsl::TessNormalsParams` (48 B). One dispatch per cluster,
/// like the displace pass, so the base mesh pool offsets ride here.
#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
pub struct TessNormalsParams {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub base_triangle_count: u32,
    /// Micro-triangles per CLAS (= level²); the per-CLAS primitive stride.
    pub micro_triangle_count: u32,
    pub normal_strength: f32,
    pub has_displacement: u32,
    /// Running CLAS index for this cluster (prior base triangles × `split_count`).
    pub out_clas_base: u32,
    pub split_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// One per-CLAS smooth-normal metadata record (mirrors `scene_bindings::TessCluster`,
/// 16 B): the device address of this CLAS's instance normal buffer + the CLAS's first
/// micro-triangle within it. The closest-hit reads it at `cluster_id - TESS_CLUSTER_ID_BASE`.
#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
pub struct TessClusterMeta {
    pub normals_lo: u32,
    pub normals_hi: u32,
    pub primitive_base: u32,
    pub _pad: u32,
}

/// Render-world resource: the smooth-normal compute pipeline + its bind-group layout.
#[derive(Resource)]
pub struct TessNormals {
    pub pipeline: CachedComputePipelineId,
    pub layout: BindGroupLayoutDescriptor,
}

/// `RenderStartup`: create the smooth-normal layout + queue its compute pipeline.
pub fn init_tess_normals(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
    allocator: Option<Res<Allocator>>,
) {
    if allocator.is_none() {
        return;
    }
    let layout = BindGroupLayoutDescriptor::new(
        "tess_normals_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, None),         // 0 params
                storage_buffer_read_only_sized(false, None), // 1 cluster_indices
                storage_buffer_read_only_sized(false, None), // 2 vertex_packed
                storage_buffer_read_only_sized(false, None), // 3 tess_barycentrics
                storage_buffer_read_only_sized(false, None), // 4 micro_indices
                storage_buffer_read_only_sized(false, None), // 5 split_corner_barys
                texture_2d(TextureSampleType::Float { filterable: true }), // 6
                sampler(SamplerBindingType::Filtering),    // 7
                storage_buffer_sized(false, None),         // 8 normals_out (rw)
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("tess_normals".into()),
        layout: vec![layout.clone()],
        shader: bevy_asset::load_embedded_asset!(asset_server.as_ref(), "tess_normals.wgsl"),
        shader_defs: vec![],
        entry_point: Some("compute_normals".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    commands.insert_resource(TessNormals { pipeline, layout });
}

/// Record a smooth-normal dispatch into `encoder`: one thread per micro-triangle
/// (`base_triangle_count * split_count * micro_triangle_count`). Returns `false`
/// (no-op) if the pipeline hasn't compiled yet.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tess_normals(
    encoder: &mut CommandEncoder,
    render_device: &RenderDevice,
    pipeline_cache: &PipelineCache,
    normals: &TessNormals,
    params: &Buffer,
    cluster_indices: &Buffer,
    vertex_packed: &Buffer,
    tess_barycentrics: &Buffer,
    micro_indices: &Buffer,
    split_corner_barys: &Buffer,
    texture: &TextureView,
    sampler: &Sampler,
    normals_out: &wgpu::Buffer,
    thread_count: u32,
) -> bool {
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(normals.pipeline) else {
        return false;
    };
    let bind_group = render_device.create_bind_group(
        "tess_normals_bind_group",
        &pipeline_cache.get_bind_group_layout(&normals.layout),
        &BindGroupEntries::sequential((
            params.as_entire_binding(),
            cluster_indices.as_entire_binding(),
            vertex_packed.as_entire_binding(),
            tess_barycentrics.as_entire_binding(),
            micro_indices.as_entire_binding(),
            split_corner_barys.as_entire_binding(),
            texture,
            sampler,
            normals_out.as_entire_binding(),
        )),
    );
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("tess_normals"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(thread_count.div_ceil(64), 1, 1);
    true
}

/// `Render` system: one-shot `SOLARI_TESS` self-test / in-situ showcase. Subdivides
/// a real displacement-mapped instance's first cluster (level + triangle count
/// live-tunable via env) and injects the displaced result at the instance's
/// transform. Gated so it runs once everything it needs is resident.
pub fn tess_displace_selftest(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    tess: Option<Res<TessDisplace>>,
    tess_normals: Option<Res<TessNormals>>,
    allocator: Option<Res<Allocator>>,
    fns: Option<Res<ClusterExtensionFns>>,
    templates: Option<Res<TessellationTemplates>>,
    showcase: Option<ResMut<TessShowcase>>,
    showcase_instances: Option<Res<TessShowcaseInstances>>,
    render_images: Res<RenderAssets<GpuImage>>,
    mesh_manager: Option<Res<ClusterMeshManager>>,
) {
    let Some(tess) = tess else {
        return;
    };
    // Wait for the pipeline to finish compiling before the build.
    if pipeline_cache.get_compute_pipeline(tess.pipeline).is_none() {
        return;
    }
    // The whole showcase needs the cluster-AS resources; unwrap up front so the
    // AS-input buffers below can be allocator-backed (256-aligned addresses — a
    // wgpu buffer's device address may not be, and the NV cluster-AS build
    // requires it).
    let (Some(allocator), Some(fns), Some(templates), Some(mut showcase)) =
        (allocator, fns, templates, showcase)
    else {
        return;
    };

    // In-situ subjects: every displacement-mapped instance found in the main world.
    // Wait until they're found, every mesh is resident, and every displacement map
    // has uploaded (so the latched set is complete).
    let Some(showcase_instances) = showcase_instances.as_ref() else {
        return;
    };
    if !showcase_instances.found || showcase_instances.instances.is_empty() {
        return;
    }
    // Per-instance DIRTY set: an instance is re-tessellated only when its
    // (view-dependent) level changed — `reclassify` updates levels each frame from
    // the camera; an instance with no slot yet (first build / not-yet-ready) is
    // dirty too. So one surface crossing a level boundary rebuilds only itself, not
    // the whole set; steady-state frames find nothing dirty and return.
    let n = showcase_instances.instances.len();
    if showcase.slots.len() != n {
        showcase.slots.resize_with(n, || None);
    }
    let dirty: Vec<usize> = (0..n)
        .filter(|&i| {
            showcase.slots[i]
                .as_ref()
                .map_or(true, |s| s.level != showcase_instances.instances[i].level)
        })
        .collect();
    if dirty.is_empty() {
        return;
    }
    let Some(mesh_manager) = mesh_manager.as_ref() else {
        return;
    };

    // Persistent per-instance BLAS-address buffer (a u64 per instance): the BLAS
    // builds write addresses here GPU-side and the PTLAS write reads them, so the
    // trace never stalls on a CPU readback. Created once (the instance count is
    // fixed) and zero-initialised — an unbuilt slot reads 0 → inactive PTLAS
    // instance. ACCELERATION_STRUCTURE_STORAGE: the cluster-AS build writes here.
    if showcase.blas_addresses.is_none() {
        let buf = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            (n as u64) * 8,
            MemoryLocation::GpuOnly,
            "tess_showcase.blas_addresses",
        );
        // One-time zero-init (not the per-frame path, so a poll here is fine).
        let mut enc = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("tess_showcase.blas_addresses_clear"),
        });
        enc.clear_buffer(&buf, 0, None);
        let clear_idx = render_queue.submit([enc.finish()]);
        let _ = render_device.wgpu_device().poll(PollType::Wait {
            submission_index: Some(clear_idx),
            timeout: None,
        });
        showcase.blas_addresses = Some(buf);
    }
    let blas_addresses_addr =
        allocator.wgpu_buffer_device_address(showcase.blas_addresses.as_ref().unwrap());

    // Shared density knobs (live-tunable, no rebuild). The subdivision LEVEL is
    // per-instance (view-dependent, chosen in `find_tess_showcase_instances`).
    //   SOLARI_TESS_TRIS  — TOTAL base triangles PER INSTANCE (bounds memory).
    //   SOLARI_TESS_SCALE — displacement height (mesh-local ≈ world units).
    //   SOLARI_TESS_SPLIT — recursive split factor K (each base triangle → K²
    //                       sub-triangles, each its own CLAS) for detail beyond the
    //                       per-CLAS budget; 1 = none. Multiplies the CLAS count K².
    let max_total_tris = env_u32("SOLARI_TESS_TRIS", 8192);
    let displacement_scale = std::env::var("SOLARI_TESS_SCALE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.2);
    let mk = |label: &'static str, contents: &[u8], usage: BufferUsages| {
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        })
    };

    // Recursive-split table (shared, K-only — the level is per-instance). One CLAS
    // per sub-triangle; `split_corner_barys` holds each sub-triangle's 3 corner
    // barycentrics within the base triangle (the compute composes them with the
    // per-sub-triangle level barys). K=1 ⇒ one sub == the base (identity).
    let split_k = env_u32("SOLARI_TESS_SPLIT", 1).clamp(1, 8);
    let split = SubdividedTriangle::new(split_k);
    let split_count = split.triangle_count() as u32;
    let mut split_corner_barys: Vec<f32> = Vec::with_capacity(split.triangle_count() * 9);
    for tri in 0..split.triangle_count() {
        for k in 0..3 {
            let vi = split.indices[tri * 3 + k] as usize;
            split_corner_barys.extend_from_slice(&split.barycentrics[vi]);
        }
    }
    let split_corner_buf = mk(
        "tess_selftest.split_corners",
        bytemuck::cast_slice(&split_corner_barys),
        BufferUsages::STORAGE,
    );

    // Smooth-shading path (`SOLARI_TESS_SMOOTH`): compute per-micro-vertex smooth,
    // displacement-aware normals into per-instance buffers + a per-CLAS metadata table
    // the closest-hit reads, so tess hits shade without micro-triangle faceting. Off →
    // the chit keeps the facet normal. `SOLARI_TESS_NORMAL` tunes the bump strength.
    let smooth = env_truthy("SOLARI_TESS_SMOOTH")
        && tess_normals
            .as_ref()
            .is_some_and(|nrm| pipeline_cache.get_compute_pipeline(nrm.pipeline).is_some());
    let normal_strength = std::env::var("SOLARI_TESS_NORMAL")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.0);
    // Stable per-instance CLAS-index region so a per-instance rebuild never renumbers
    // another instance's CLASes (their `cluster_id`s + metadata slots stay fixed).
    let clas_per_region = max_total_tris.saturating_mul(split_count).max(1);
    // Persistent per-CLAS metadata table (created once when smooth is on).
    if smooth && showcase.tess_clusters_meta.is_none() {
        let buf = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            (n as u64) * (clas_per_region as u64) * 16,
            MemoryLocation::GpuOnly,
            "tess_showcase.tess_clusters_meta",
        );
        showcase.tess_clusters_meta = Some(buf);
    }
    // Cloned handle (Arc) so the meta copies below don't hold a `showcase` borrow.
    let meta_buf = showcase.tess_clusters_meta.clone();

    let mut rebuilt = 0usize;
    let mut rebuilt_micro_tris = 0u64;

    // One displace (wgpu compute) encoder + one raw-VK AS-build encoder for the WHOLE
    // dirty batch — chained GPU-side and submitted ONCE, so a re-tessellation costs a
    // single poll instead of the former ~5 submit+poll round-trips per instance. The
    // displace passes (wgpu) and the cluster-AS builds (raw `as_hal_mut`) must live in
    // separate encoders — the wgpu fork panics if one encoder mixes them.
    let mut displace_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("tess_selftest.displace"),
    });
    let mut as_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("tess_selftest.as_build"),
    });

    // This generation's deferred-drop bucket: build-only buffers (params, barys,
    // scratch, src-infos, CLAS-address lists) the in-flight submit reads, plus each
    // replaced slot's old trace buffers. Enqueued into `retiring` and dropped a few
    // generations later (> frames-in-flight) since the async path has no poll to drain.
    let mut retire_now: Vec<Box<dyn core::any::Any + Send + Sync>> = Vec::new();
    let mut recorded = false;

    for &i in &dirty {
        let inst = &showcase_instances.instances[i];
        let Some(clusters) = mesh_manager.tess_clusters(inst.mesh) else {
            continue;
        };
        let Some(mesh_aabb) = mesh_manager.mesh_aabb(inst.mesh) else {
            continue;
        };
        let Some(gpu_disp) = render_images.get(&inst.displacement) else {
            continue;
        };
        let total_base_tris = clusters.iter().map(|c| c[2]).sum::<u32>().min(max_total_tris);
        if total_base_tris == 0 {
            continue;
        }

        // Per-instance (view-dependent) subdivision — its own level, barycentrics,
        // and micro-vertex count.
        let level = inst.level.clamp(1, MAX_TESS_LEVEL);
        let subdiv = SubdividedTriangle::new(level);
        let micro = subdiv.vertex_count() as u32;
        let micro_tris_per_base = subdiv.triangle_count() as u64;
        let barys: Vec<f32> = subdiv.barycentrics.iter().flatten().copied().collect();
        let bary_buf = mk(
            "tess_selftest.barys",
            bytemuck::cast_slice(&barys),
            BufferUsages::STORAGE,
        );

        let disp_view = &gpu_disp.texture_view;
        let disp_sampler = &gpu_disp.sampler;
        // Allocator-backed ⇒ 256-aligned device address the NV cluster-AS build needs;
        // STORAGE (compute write) + AS build input.
        let out_buf = allocator.create_buffer(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            BufferUsages::STORAGE,
            (micro as u64) * (total_base_tris as u64) * (split_count as u64) * 12,
            MemoryLocation::GpuOnly,
            "tess_selftest.out",
        );

        let tri_count = micro_tris_per_base as u32;
        let total_clas = total_base_tris * split_count;
        // Smooth path: this instance's denormalized per-CLAS normal buffer (read by the
        // closest-hit via the metadata table) + the level's micro-triangle topology
        // (primitive → 3 micro-vertex indices, for the normal compute). `None` off.
        let (normal_buf, micro_indices_buf) = if smooth {
            let nbuf = allocator.create_buffer(
                &render_device,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                BufferUsages::STORAGE,
                // Per micro-triangle: 3 vertices × (packed normal + UV) = 3 × 12 B.
                (total_clas as u64) * (tri_count as u64) * 36,
                MemoryLocation::GpuOnly,
                "tess_selftest.normals",
            );
            let micro_indices: Vec<u32> = subdiv.indices.iter().map(|&x| x as u32).collect();
            let ibuf = mk(
                "tess_selftest.micro_indices",
                bytemuck::cast_slice(&micro_indices),
                BufferUsages::STORAGE,
            );
            (Some(nbuf), Some(ibuf))
        } else {
            (None, None)
        };

        // One displace dispatch PER CLUSTER into disjoint regions of `out_buf`
        // (`out_vertex_base` = running micro-vertex count → contiguous by base tri),
        // recorded into the shared displace encoder. The smooth-normal dispatch (if on)
        // runs per cluster too, sharing the `emitted` running base-triangle count.
        let mut emitted = 0u32;
        for c in clusters {
            if emitted >= total_base_tris {
                break;
            }
            let cluster_tris = c[2].min(total_base_tris - emitted);
            if cluster_tris == 0 {
                continue;
            }
            let params = TessDisplaceParams {
                vertex_offset: c[0],
                index_offset: c[1],
                base_triangle_count: cluster_tris,
                micro_vertex_count: micro,
                displacement_scale,
                displacement_bias: 0.0,
                // Running micro-vertex offset, accounting for the K² sub-CLASes/base.
                out_vertex_base: emitted * split_count * micro,
                has_displacement: 1,
                split_count,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let params_buf = mk(
                "tess_selftest.params",
                bytemuck::bytes_of(&params),
                BufferUsages::UNIFORM,
            );
            let dispatched = dispatch_tess_displace(
                &mut displace_encoder,
                &render_device,
                &pipeline_cache,
                &tess,
                &params_buf,
                mesh_manager.indices.buffer(),
                mesh_manager.vertex_positions.buffer(),
                mesh_manager.vertex_packed.buffer(),
                &bary_buf,
                &out_buf,
                disp_view,
                disp_sampler,
                &split_corner_buf,
                cluster_tris * split_count * micro,
            );
            if !dispatched {
                return;
            }
            retire_now.push(Box::new(params_buf));

            // Smooth normals for this cluster's CLASes (denormalized per micro-triangle).
            if let (Some(normals), Some(normal_buf), Some(micro_indices_buf)) =
                (tess_normals.as_deref(), normal_buf.as_ref(), micro_indices_buf.as_ref())
            {
                let nparams = TessNormalsParams {
                    vertex_offset: c[0],
                    index_offset: c[1],
                    base_triangle_count: cluster_tris,
                    micro_triangle_count: tri_count,
                    normal_strength,
                    has_displacement: 1,
                    // Running CLAS index base for this cluster.
                    out_clas_base: emitted * split_count,
                    split_count,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    _pad3: 0,
                };
                let nparams_buf = mk(
                    "tess_selftest.nparams",
                    bytemuck::bytes_of(&nparams),
                    BufferUsages::UNIFORM,
                );
                dispatch_tess_normals(
                    &mut displace_encoder,
                    &render_device,
                    &pipeline_cache,
                    normals,
                    &nparams_buf,
                    mesh_manager.indices.buffer(),
                    mesh_manager.vertex_packed.buffer(),
                    &bary_buf,
                    micro_indices_buf,
                    &split_corner_buf,
                    disp_view,
                    disp_sampler,
                    normal_buf,
                    cluster_tris * split_count * tri_count,
                );
                retire_now.push(Box::new(nparams_buf));
            }
            emitted += cluster_tris;
        }
        retire_now.push(Box::new(bary_buf));
        if let Some(b) = micro_indices_buf {
            retire_now.push(Box::new(b));
        }

        // Placed world AABB from the mesh's object-space AABB + the displacement
        // margin (height ≤ 1 along the normal ⇒ ±displacement_scale), pushed through
        // the instance transform. No per-frame megabyte position readback — the
        // per-frame path can't afford it; this is conservative but cheap (8 corners).
        let (transform, world_aabb_min, world_aabb_max) = {
            let (amin, amax) = mesh_aabb;
            let margin = displacement_scale.abs();
            let obj_min = [amin[0] - margin, amin[1] - margin, amin[2] - margin];
            let obj_max = [amax[0] + margin, amax[1] + margin, amax[2] + margin];
            let t = inst.world_from_local;
            let mut w_min = [f32::INFINITY; 3];
            let mut w_max = [f32::NEG_INFINITY; 3];
            for &x in &[obj_min[0], obj_max[0]] {
                for &y in &[obj_min[1], obj_max[1]] {
                    for &z in &[obj_min[2], obj_max[2]] {
                        for r in 0..3 {
                            let w = t[r][0] * x + t[r][1] * y + t[r][2] * z + t[r][3];
                            w_min[r] = w_min[r].min(w);
                            w_max[r] = w_max[r].max(w);
                        }
                    }
                }
            }
            (t, w_min, w_max)
        };

        // Record instantiate (CLAS) → per-instance BLAS into the shared AS encoder,
        // chained GPU-side: the instantiate's CLAS-address buffer feeds straight into
        // the BLAS as `cluster_references`, and the BLAS device address is written
        // directly into this instance's slot of the persistent `blas_addresses` buffer
        // — read GPU-side by the PTLAS write, so nothing is read back to the CPU.
        // `split_count` sub-CLASes/base. Each CLAS bakes `cluster_id_base + c` so the
        // closest-hit recovers a unique per-CLAS index into the smooth-normal metadata.
        let cluster_id_base = TESS_CLUSTER_ID_BASE + (i as u32) * clas_per_region;
        let out_addr = allocator.wgpu_buffer_device_address(&out_buf);
        let clas = templates.record_instantiate_displaced_batch(
            &render_device,
            &render_queue,
            &allocator,
            &fns,
            &mut as_encoder,
            level,
            out_addr,
            total_clas,
            cluster_id_base,
        );
        let clas_refs_addr = allocator.wgpu_buffer_device_address(&clas.clas_addresses);
        let blas = record_build_per_instance_blas(
            &render_device,
            &render_queue,
            &allocator,
            &fns,
            &mut as_encoder,
            clas_refs_addr,
            total_clas,
            blas_addresses_addr + (i as u64) * 8,
        );

        // Smooth path: write this instance's per-CLAS metadata (its normal buffer
        // address + each CLAS's first micro-triangle) into its fixed region of the
        // metadata table, GPU-side via a staging copy in the displace encoder.
        let normal_keep: Option<Box<dyn core::any::Any + Send + Sync>> =
            match (normal_buf, meta_buf.as_ref()) {
                (Some(nbuf), Some(mbuf)) => {
                    let n_addr = allocator.wgpu_buffer_device_address(&nbuf);
                    let metas: Vec<TessClusterMeta> = (0..total_clas)
                        .map(|c| TessClusterMeta {
                            normals_lo: n_addr as u32,
                            normals_hi: (n_addr >> 32) as u32,
                            primitive_base: c * tri_count,
                            _pad: 0,
                        })
                        .collect();
                    let staging = mk(
                        "tess_selftest.meta_staging",
                        bytemuck::cast_slice(&metas),
                        BufferUsages::COPY_SRC,
                    );
                    displace_encoder.copy_buffer_to_buffer(
                        &staging,
                        0,
                        mbuf,
                        (i as u64) * (clas_per_region as u64) * 16,
                        (total_clas as u64) * 16,
                    );
                    retire_now.push(Box::new(staging));
                    Some(Box::new(nbuf))
                }
                _ => None,
            };

        // Slot keepalive: each BLAS references its CLAS storage references its
        // `out_buf` — all three must outlive the trace; the smooth-normal buffer (if
        // any) is read by the closest-hit, so it must too.
        let mut keepalive: Vec<Box<dyn core::any::Any + Send + Sync>> =
            vec![Box::new(out_buf), Box::new(clas.storage), Box::new(blas.storage)];
        if let Some(nk) = normal_keep {
            keepalive.push(nk);
        }
        // The CLAS-address list + both builds' scratch/src-infos are read only during
        // this frame's submit (GPU-side) → deferred-drop.
        retire_now.push(Box::new(clas.clas_addresses));
        retire_now.extend(clas.transient);
        retire_now.extend(blas.transient);

        // Claim the slot now — the BLAS address arrives GPU-side, so there's nothing to
        // wait for. The replaced slot's trace buffers go to the deferred-drop ring (an
        // in-flight trace may still read its BLAS).
        let old = showcase.slots[i].replace(TessShowcaseSlot {
            entry: TessShowcaseEntry {
                blas_slot: i as u32,
                material: inst.material,
                entity: inst.entity,
                transform,
                world_aabb_min,
                world_aabb_max,
            },
            level,
            _keepalive: keepalive,
        });
        if let Some(old) = old {
            retire_now.extend(old._keepalive);
        }
        rebuilt += 1;
        rebuilt_micro_tris += (total_clas as u64) * micro_tris_per_base;
        recorded = true;
    }

    // Every dirty instance skipped (resources not resident) → nothing recorded; leave
    // them dirty and retry next frame (no submit).
    if !recorded {
        return;
    }

    // ONE async submit for the whole dirty batch: displace (compute) → AS builds
    // (instantiate + per-instance BLAS, chained GPU-side with AS barriers; the BLAS
    // addresses land straight in `blas_addresses`). NO poll — the render thread never
    // blocks on the build. The graph's PTLAS write (later this frame, same queue) reads
    // the fresh addresses; the build's trailing AS barrier makes them visible across
    // the submit boundary.
    render_queue.submit([displace_encoder.finish(), as_encoder.finish()]);

    // Defer this generation's buffer drops past the frames-in-flight window — the
    // async path has no poll to drain the GPU before freeing.
    showcase.retiring.push_back(retire_now);
    while showcase.retiring.len() > RETIRE_GENERATIONS {
        showcase.retiring.pop_front();
    }

    // Rebuild the flat PTLAS-write view from the live slots — re-tessellated this
    // frame or held from a previous one. A still-`None` slot (not yet resident)
    // leaves no entry, naturally excluded.
    showcase.entries = showcase
        .slots
        .iter()
        .filter_map(|s| s.as_ref().map(|s| s.entry))
        .collect();
    if rebuilt > 0 {
        bevy_log::info!(
            "tess showcase: re-tessellated {}/{} dirty instances ({} armed, \
             {} micro-triangles rebuilt).",
            rebuilt,
            dirty.len(),
            showcase.entries.len(),
            rebuilt_micro_tris,
        );
    }
}

// ── Brick B4: PTLAS injection (mirrors `hair/ptlas_hair.rs`) ──────────────────

/// Truthy env check — false for unset / `""` / `"0"` / `"false"`.
fn env_truthy(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(v.as_str(), "" | "0" | "false"),
        Err(_) => false,
    }
}

/// Parse a `u32` env var, falling back to `default` if unset or unparseable.
fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}


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
        // Relative to this file's dir (`src/geometry/`); see `init_tess_displace`.
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

/// `Render::Prepare`: set the PTLAS-write params from the armed showcase. The
/// tess instance occupies PTLAS index `cluster_high_water + hair_count` (above
/// the cluster slots + hair), matching the `high_water` fold in
/// [`crate::accel::ptlas::prepare_ptlas_params`].
pub fn prepare_tess_ptlas_write(
    write: Option<ResMut<TessPtlasWrite>>,
    showcase: Option<Res<TessShowcase>>,
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
    // One PTLAS record per armed entry (matches the `high_water` fold in
    // `prepare_ptlas_params`).
    let tess_count = showcase.as_ref().map_or(0, |s| s.entries.len() as u32);
    write.tess_count = tess_count;
    if tess_count == 0 {
        return;
    }
    let showcase = showcase.unwrap();
    let cluster_high_water = instances.as_ref().map_or(0, |i| i.slot_high_water());
    let hair_count = hair.as_ref().map_or(0, |h| h.count);

    // Main entity → cluster slot, so each tess instance reads its own previous-frame
    // transform (correct motion vectors) instead of slot 0's.
    let slot_of: EntityHashMap<u32> = instance_slots
        .iter()
        .map(|(main, slot)| (main.id(), slot.0 .0))
        .collect();

    let gpu_instances: Vec<TessInstanceGpu> = showcase
        .entries
        .iter()
        .map(|e| {
            let t = e.transform;
            TessInstanceGpu {
                transform_r0: Vec4::new(t[0][0], t[0][1], t[0][2], t[0][3]),
                transform_r1: Vec4::new(t[1][0], t[1][1], t[1][2], t[1][3]),
                transform_r2: Vec4::new(t[2][0], t[2][1], t[2][2], t[2][3]),
                aabb_min: Vec4::new(e.world_aabb_min[0], e.world_aabb_min[1], e.world_aabb_min[2], 0.0),
                aabb_max: Vec4::new(e.world_aabb_max[0], e.world_aabb_max[1], e.world_aabb_max[2], 0.0),
                // The BLAS address itself is read GPU-side from `blas_addresses` at
                // this slot; the record only carries the index.
                blas_slot: e.blas_slot,
                // The material's slot IS its SBT hit-group record (`derived_hit_group`);
                // route the tess hit to this surface's own material, not material 0.
                sbt_record: material_slots
                    .as_ref()
                    .and_then(|ms| ms.slot_of(e.material))
                    .unwrap_or(0),
                // The source entity's cluster slot → its real previous-frame transform.
                instance_id: slot_of.get(&e.entity).copied().unwrap_or(0),
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
    showcase: Option<Res<TessShowcase>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let Some(mut write) = write else {
        return;
    };
    let (Some(ptlas), Some(params), Some(instances), Some(showcase)) = (
        ptlas,
        write.params.binding(),
        write.instances.binding(),
        showcase.as_ref(),
    ) else {
        write.bind_group = None;
        return;
    };
    // The per-instance BLAS-address buffer the build writes GPU-side; absent until
    // the showcase first tessellates, in which case there's nothing to inject yet.
    let Some(blas_addresses) = showcase.blas_addresses.as_ref() else {
        write.bind_group = None;
        return;
    };
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
