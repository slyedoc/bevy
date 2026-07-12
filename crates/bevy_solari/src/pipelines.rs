//! Centralized pipelines for `bevy_solari` — one [`SolariPipelines`] resource
//! holding every pass's compiled-pipeline id, built once at `RenderStartup`, so a
//! dispatch reads its id from **one** place instead of N scattered per-resource
//! accessors. The companion to [`SolariResourceManager`](crate::resource_manager)
//! (the layouts): together they are `bevy_pbr`'s meshlet split — `MeshletPipelines`
//! (ids) + `ResourceManager` (layouts).
//!
//! Field names mirror [`SolariResourceManager`] 1:1, so a pass's id
//! (`pipelines.selector`) and its layout (`resource_manager.selector`) share a key:
//! [`init_solari_pipelines`] reads each layout from the manager to queue the matching
//! pipeline.
//!
//! The generic per-column **scatter** pipeline is the one exception: it's created
//! by the generic [`GpuColumnPlugin`](crate::ecs_gpu) per column type, so it stays
//! with that infrastructure rather than being enumerated here. The shading
//! integrator is the raw-VK ray-tracing pipeline (`gpu::rt_pipeline`), built on
//! its own — not a compute pipeline, so not enumerated here either.

use bevy_app::App;
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer};
use bevy_core_pipeline::FullscreenShader;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_render::render_resource::{
    CachedComputePipelineId, CachedRenderPipelineId, ComputePipelineDescriptor, PipelineCache,
};

use crate::bindings::{ClusterSceneBindGroupLayout, RaytracingSceneBindings};
use crate::resource_manager::SolariResourceManager;

/// Every solari compute-pass pipeline id, queued once at `RenderStartup`. Fields
/// are `pub`: a dispatch reads its id directly and resolves it through
/// `PipelineCache`. Absent only on devices lacking the cluster/sparse support
/// solari needs (the manager gates on the allocator).
#[derive(Resource)]
pub struct SolariPipelines {
    pub transform_frontier_seed: CachedComputePipelineId,
    pub transform_frontier_expand: CachedComputePipelineId,
    pub transform_frontier_finalize: CachedComputePipelineId,
    pub transform_propagate: CachedComputePipelineId,
    pub transform_subtract: CachedComputePipelineId,
    pub transform_gather: CachedComputePipelineId,
    pub transform_readback: CachedComputePipelineId,
    pub rt_camera: CachedComputePipelineId,
    pub light_resolve: CachedComputePipelineId,
    pub atmosphere: CachedComputePipelineId,
    pub atmosphere_lut: CachedComputePipelineId,

    pub selector_reset: CachedComputePipelineId,
    pub selector_main: CachedComputePipelineId,

    pub blas_sharing_geom_reset: CachedComputePipelineId,
    pub blas_sharing_classify: CachedComputePipelineId,
    pub blas_sharing_elect_dirty: CachedComputePipelineId,
    pub blas_sharing_finalize_count: CachedComputePipelineId,
    pub blas_sharing_assign_address: CachedComputePipelineId,
    /// Commits `geometry_built_level` after the BLAS build chain truly records
    /// (dispatched from `dispatch_blas_rebuild`) — the rebuild-until-built heal.
    pub blas_sharing_commit_built: CachedComputePipelineId,

    pub ptlas_seed: CachedComputePipelineId,
    pub ptlas_incremental: CachedComputePipelineId,
    pub ptlas_finalize: CachedComputePipelineId,
    /// Debug (SOLARI_PTLAS_VALIDATE): flags + nulls corrupt BLAS addresses in
    /// the WRITE stream before the partitioned-AS build consumes them.
    pub ptlas_validate: CachedComputePipelineId,

    /// Appends hair instances to the PTLAS WRITE stream (`hair/ptlas_hair_write.wgsl`).
    pub ptlas_hair_write: CachedComputePipelineId,

    /// Reusable inline-`rayQuery` batch trace (`ray_query/ray_query.wgsl`).
    pub ray_query: CachedComputePipelineId,
    /// Skeletal LBS deform of animated cluster meshes (`accel/deform.wgsl`).
    pub deform: CachedComputePipelineId,
    /// Per-instance animated CLAS instantiate + BLAS args (`accel/instantiate.wgsl`).
    pub animated_blas: CachedComputePipelineId,
    /// Fullscreen depth-write bridging RT primary-hit depth into the hardware depth
    /// buffer so gizmos occlude against the ray-traced scene (`render/gizmo_depth.wgsl`).
    pub gizmo_depth: CachedRenderPipelineId,
}

/// Register every solari compute shader as an embedded asset. Called from
/// [`crate::SolariPlugin::build`] so the `embedded_asset!` here and the
/// `load_embedded_asset!` in [`init_solari_pipelines`] are co-located — the
/// embedded path matches by construction, no cross-module path drift.
pub fn embed_solari_shaders(app: &mut App) {
    embedded_asset!(app, "transform/transform_frontier.wgsl");
    embedded_asset!(app, "transform/transform_propagate.wgsl");
    embedded_asset!(app, "transform/transform_subtract.wgsl");
    embedded_asset!(app, "transform/transform_gather.wgsl");
    embedded_asset!(app, "transform/transform_readback.wgsl");
    embedded_asset!(app, "render/rt_pipeline/rt_camera.wgsl");
    // rt_payload.wgsl is composed into the raw-VK RT pipeline directly, but
    // restir_spatial.wgsl (a wgpu compute pass) #imports its RtCamera struct —
    // the wgpu pipeline cache resolves that only for registered libraries.
    bevy_shader::load_shader_library!(app, "render/rt_pipeline/rt_payload.wgsl");
    embedded_asset!(app, "lights/light_resolve.wgsl");
    embedded_asset!(app, "render/atmosphere_bake.wgsl");
    embedded_asset!(app, "render/atmosphere_lut_bake.wgsl");
    embedded_asset!(app, "render/rt_pipeline/blit.wgsl");
    embedded_asset!(app, "render/rt_pipeline/restir_spatial.wgsl");
    embedded_asset!(app, "nrc/nrc_mlp.wgsl");
    embedded_asset!(app, "render/dlss_resolve.wgsl");
    embedded_asset!(app, "geometry/tess_classify.wgsl");
    embedded_asset!(app, "geometry/tess_gen_verts.wgsl");
    embedded_asset!(app, "geometry/tess_gen_attrs.wgsl");
    embedded_asset!(app, "geometry/tess_instantiate.wgsl");
    embedded_asset!(app, "geometry/tess_scatter.wgsl");
    embedded_asset!(app, "geometry/tess_ptlas_write.wgsl");
    embedded_asset!(app, "accel/deform.wgsl");
    embedded_asset!(app, "accel/instantiate.wgsl");
    embedded_asset!(app, "accel/selector.wgsl");
    embedded_asset!(app, "accel/blas_sharing.wgsl");
    embedded_asset!(app, "accel/ptlas_fill.wgsl");
    embedded_asset!(app, "hair/ptlas_hair_write.wgsl");
    embedded_asset!(app, "ray_query/ray_query.wgsl");
    embedded_asset!(app, "render/gizmo_depth.wgsl");
}

/// `RenderStartup`, after [`SolariResourceManager`] is built: queue every pass's
/// pipeline against the layout the manager owns, and store the ids. Gated on the
/// manager (absent on unsupported devices), so [`SolariPipelines`] — and every
/// `run_if`-gated dispatch — never exists there.
pub fn init_solari_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
    // The single source of every pass layout. `Option` — absent on devices without
    // the cluster/sparse support solari needs (the manager gates on the allocator).
    resource_manager: Option<Res<SolariResourceManager>>,
    cluster_scene_layout: Res<ClusterSceneBindGroupLayout>,
    // The raytracing scene group (`@group(0)`, holding the TLAS) — the batch
    // ray-query pipeline's layout pairs it with the I/O group (`@group(1)`).
    scene_bindings: Res<RaytracingSceneBindings>,
    // Fullscreen vertex for the gizmo-depth bridge raster pass.
    fullscreen_shader: Res<FullscreenShader>,
    mut registry: ResMut<crate::ecs_gpu::SolariPipelineRegistry>,
) {
    let Some(resource_manager) = resource_manager else {
        return;
    };
    let rt_camera = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("rt_camera".into()),
        layout: vec![resource_manager.rt_camera.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "render/rt_pipeline/rt_camera.wgsl"),
        shader_defs: vec![],
        entry_point: Some("rt_camera".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    let frontier_shader =
        load_embedded_asset!(asset_server.as_ref(), "transform/transform_frontier.wgsl");
    let transform_frontier_seed = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("transform_frontier_seed".into()),
        layout: vec![resource_manager.transform_frontier.clone()],
        shader: frontier_shader.clone(),
        shader_defs: vec![],
        entry_point: Some("seed".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    let transform_frontier_expand =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("transform_frontier_expand".into()),
            layout: vec![resource_manager.transform_frontier.clone()],
            shader: frontier_shader.clone(),
            shader_defs: vec![],
            entry_point: Some("expand".into()),
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        });
    let transform_frontier_finalize =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("transform_frontier_finalize".into()),
            layout: vec![resource_manager.transform_frontier.clone()],
            shader: frontier_shader,
            shader_defs: vec![],
            entry_point: Some("finalize".into()),
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        });
    let transform_propagate = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("transform_propagate".into()),
        layout: vec![resource_manager.transform_propagate.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "transform/transform_propagate.wgsl"),
        shader_defs: vec![],
        entry_point: Some("propagate".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    let transform_subtract = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("transform_subtract".into()),
        layout: vec![resource_manager.transform_subtract.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "transform/transform_subtract.wgsl"),
        shader_defs: vec![],
        entry_point: Some("subtract".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    let transform_gather = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("transform_gather".into()),
        layout: vec![resource_manager.transform_gather.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "transform/transform_gather.wgsl"),
        shader_defs: vec![],
        entry_point: Some("gather".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    let transform_readback = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("transform_readback".into()),
        layout: vec![resource_manager.transform_readback.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "transform/transform_readback.wgsl"),
        shader_defs: vec![],
        entry_point: Some("readback".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    let light_resolve = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("light_resolve".into()),
        layout: vec![resource_manager.light_resolve.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "lights/light_resolve.wgsl"),
        shader_defs: vec![],
        entry_point: Some("resolve".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    let atmosphere_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("solari_atmosphere_lut_bake".into()),
        layout: vec![resource_manager.atmosphere_lut.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "render/atmosphere_lut_bake.wgsl"),
        shader_defs: vec![],
        entry_point: Some("bake".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    let atmosphere = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("solari_atmosphere_bake".into()),
        layout: vec![resource_manager.atmosphere.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "render/atmosphere_bake.wgsl"),
        shader_defs: vec![],
        entry_point: Some("bake".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    // ── AS passes: each pairs the cluster-scene group with its `@group(1)` layout. ──
    let scene = cluster_scene_layout.0.clone();

    let selector_shader = load_embedded_asset!(asset_server.as_ref(), "accel/selector.wgsl");
    let selector_pl = vec![scene.clone(), resource_manager.selector.clone()];
    let selector_reset = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("cluster_selector_reset".into()),
        layout: selector_pl.clone(),
        shader: selector_shader.clone(),
        shader_defs: vec![],
        entry_point: Some("select_reset".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });
    let selector_main = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("cluster_selector_main".into()),
        layout: selector_pl,
        shader: selector_shader,
        shader_defs: vec![],
        entry_point: Some("select_main".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    let sharing_shader = load_embedded_asset!(asset_server.as_ref(), "accel/blas_sharing.wgsl");
    let sharing_pl = vec![scene.clone(), resource_manager.blas_sharing.clone()];
    let sharing_pipeline = |entry: &'static str| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("blas_sharing_{entry}").into()),
            layout: sharing_pl.clone(),
            shader: sharing_shader.clone(),
            shader_defs: vec![],
            entry_point: Some(entry.into()),
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        })
    };
    let blas_sharing_geom_reset = sharing_pipeline("geom_reset");
    let blas_sharing_classify = sharing_pipeline("classify");
    let blas_sharing_elect_dirty = sharing_pipeline("elect_dirty");
    let blas_sharing_finalize_count = sharing_pipeline("finalize_count");
    let blas_sharing_assign_address = sharing_pipeline("assign_address");
    let blas_sharing_commit_built = sharing_pipeline("commit_built");

    let ptlas_shader = load_embedded_asset!(asset_server.as_ref(), "accel/ptlas_fill.wgsl");
    let ptlas_pl = vec![scene, resource_manager.ptlas.clone()];
    let ptlas_make = |entry: &'static str| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("ptlas_{entry}").into()),
            layout: ptlas_pl.clone(),
            shader: ptlas_shader.clone(),
            shader_defs: vec![],
            entry_point: Some(entry.into()),
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        })
    };
    let ptlas_seed = ptlas_make("fill_seed");
    let ptlas_incremental = ptlas_make("fill_incremental");
    let ptlas_finalize = ptlas_make("finalize");
    let ptlas_validate = ptlas_make("validate");

    // Hair PTLAS-write — a single self-contained `@group(0)` (no scene group).
    let ptlas_hair_write = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("ptlas_hair_write".into()),
        layout: vec![resource_manager.ptlas_hair_write.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "hair/ptlas_hair_write.wgsl"),
        shader_defs: vec![],
        entry_point: Some("hair_write".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    // Batch ray-query: the self-contained shader needs only the scene group (0, for
    // the TLAS) + this pass's I/O group (1).
    let ray_query = crate::ray_query::queue_ray_query_pipeline(
        &pipeline_cache,
        asset_server.as_ref(),
        vec![
            scene_bindings.bind_group_layout.clone(),
            resource_manager.ray_query_io.clone(),
        ],
    );

    let gizmo_depth = crate::render::gizmo_depth::gizmo_depth_pipeline(
        &pipeline_cache,
        &fullscreen_shader,
        load_embedded_asset!(asset_server.as_ref(), "render/gizmo_depth.wgsl"),
        resource_manager.gizmo_depth.clone(),
    );

    let deform = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("deform".into()),
        layout: vec![resource_manager.deform.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "accel/deform.wgsl"),
        shader_defs: vec![],
        entry_point: Some("deform".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    let animated_blas = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("animated_instantiate".into()),
        layout: vec![resource_manager.animated_blas.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "accel/instantiate.wgsl"),
        shader_defs: vec![],
        entry_point: Some("instantiate".into()),
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    // Every id joins the one readiness gate ([`solari_pipelines_ready`]) —
    // subset gates are the recurring startup-race class.
    for (label, id) in [
        ("transform_frontier_seed", transform_frontier_seed),
        ("transform_frontier_expand", transform_frontier_expand),
        ("transform_frontier_finalize", transform_frontier_finalize),
        ("transform_propagate", transform_propagate),
        ("transform_subtract", transform_subtract),
        ("transform_gather", transform_gather),
        ("transform_readback", transform_readback),
        ("rt_camera", rt_camera),
        ("light_resolve", light_resolve),
        ("atmosphere", atmosphere),
        ("atmosphere_lut", atmosphere_lut),
        ("selector_reset", selector_reset),
        ("selector_main", selector_main),
        ("blas_sharing_geom_reset", blas_sharing_geom_reset),
        ("blas_sharing_classify", blas_sharing_classify),
        ("blas_sharing_elect_dirty", blas_sharing_elect_dirty),
        ("blas_sharing_finalize_count", blas_sharing_finalize_count),
        ("blas_sharing_assign_address", blas_sharing_assign_address),
        ("blas_sharing_commit_built", blas_sharing_commit_built),
        ("ptlas_seed", ptlas_seed),
        ("ptlas_incremental", ptlas_incremental),
        ("ptlas_finalize", ptlas_finalize),
        ("ptlas_validate", ptlas_validate),
        ("ptlas_hair_write", ptlas_hair_write),
        ("ray_query", ray_query),
        ("deform", deform),
        ("animated_blas", animated_blas),
    ] {
        registry.register(label, id);
    }
    registry.register_render("gizmo_depth", gizmo_depth);

    commands.insert_resource(SolariPipelines {
        transform_frontier_seed,
        transform_frontier_expand,
        transform_frontier_finalize,
        transform_propagate,
        transform_subtract,
        transform_gather,
        transform_readback,
        rt_camera,
        light_resolve,
        atmosphere,
        atmosphere_lut,
        selector_reset,
        selector_main,
        blas_sharing_geom_reset,
        blas_sharing_classify,
        blas_sharing_elect_dirty,
        blas_sharing_finalize_count,
        blas_sharing_assign_address,
        blas_sharing_commit_built,
        ptlas_seed,
        ptlas_incremental,
        ptlas_finalize,
        ptlas_validate,
        ptlas_hair_write,
        ray_query,
        deform,
        animated_blas,
        gizmo_depth,
    });
}
