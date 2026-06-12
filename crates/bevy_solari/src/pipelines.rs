//! Centralized pipelines for `bevy_solari` — one [`SolariPipelines`] resource
//! holding every pass's compiled-pipeline id, built once at `RenderStartup`, so a
//! dispatch reads its id from **one** place instead of N scattered per-resource
//! accessors. The companion to [`SolariResourceManager`](crate::resource_manager)
//! (the layouts): together they are `bevy_pbr`'s meshlet split — `MeshletPipelines`
//! (ids) + `ResourceManager` (layouts).
//!
//! Field names mirror [`SolariResourceManager`] 1:1, so a pass's id
//! (`pipelines.deform`) and its layout (`resource_manager.deform`) share a key:
//! [`init_solari_pipelines`] reads each layout from the manager to queue the matching
//! pipeline. Fields are `pub` — the dispatch is registered with
//! `run_if(resource_exists::<SolariPipelines>)`, so it takes a non-optional
//! `Res<SolariPipelines>` and reads its id directly.
//!
//! The generic per-column **scatter** pipeline is the one exception: it's created
//! by the generic [`GpuColumnPlugin`](crate::ecs_gpu) per column type, so it stays
//! with that infrastructure rather than being enumerated here.

use bevy_app::App;
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle};
use bevy_core_pipeline::FullscreenShader;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res},
};
use bevy_render::render_resource::{
    CachedComputePipelineId, CachedRenderPipelineId, ComputePipelineDescriptor, PipelineCache,
};
use bevy_shader::{Shader, ShaderDefVal};

use crate::bindings::{ClusterSceneBindGroupLayout, RaytracingSceneBindings};
use crate::ecs_gpu::{SceneColumns, SCENE_COLUMNS_GROUP_DEF};
use crate::render::gizmo_depth_pipeline;
use crate::render::pathtracer::pipelines::SCENE_COLUMNS_GROUP;
use crate::resource_manager::SolariResourceManager;

/// Every solari pass pipeline id, queued once at `RenderStartup`. Fields are `pub`:
/// a dispatch reads its id directly and resolves it through `PipelineCache`. The
/// dispatch is registered with `run_if(resource_exists::<SolariPipelines>)`, so it
/// takes a non-optional `Res<SolariPipelines>` (this resource is absent only on
/// devices lacking the cluster/sparse support solari needs, where it never runs).
#[derive(Resource)]
pub struct SolariPipelines {
    pub transform_propagate: CachedComputePipelineId,
    pub transform_gather: CachedComputePipelineId,
    pub transform_readback: CachedComputePipelineId,
    pub light_resolve: CachedComputePipelineId,
    pub deform: CachedComputePipelineId,
    pub animated_blas: CachedComputePipelineId,
    pub atmosphere: CachedComputePipelineId,
    pub pathtracer: CachedComputePipelineId,

    pub selector_reset: CachedComputePipelineId,
    pub selector_main: CachedComputePipelineId,

    pub blas_sharing_geom_reset: CachedComputePipelineId,
    pub blas_sharing_classify: CachedComputePipelineId,
    pub blas_sharing_elect_dirty: CachedComputePipelineId,
    pub blas_sharing_finalize_count: CachedComputePipelineId,
    pub blas_sharing_assign_address: CachedComputePipelineId,

    pub ptlas_seed: CachedComputePipelineId,
    pub ptlas_incremental: CachedComputePipelineId,
    pub ptlas_finalize: CachedComputePipelineId,

    pub restir_visibility: CachedComputePipelineId,
    pub restir_presample: CachedComputePipelineId,
    pub restir_regir_decay: CachedComputePipelineId,
    pub restir_regir_fill: CachedComputePipelineId,
    pub restir_initial_and_temporal: CachedComputePipelineId,
    pub restir_spatial_and_shade: CachedComputePipelineId,
    pub restir_specular_gi: CachedComputePipelineId,
    pub restir_compose: CachedComputePipelineId,
    pub restir_caustic_decay: CachedComputePipelineId,
    pub restir_caustic_emit: CachedComputePipelineId,
    pub restir_caustic_prepare_reset: CachedComputePipelineId,
    pub restir_caustic_prepare_reduce: CachedComputePipelineId,
    pub restir_caustic_prepare_finalize: CachedComputePipelineId,
    pub restir_debug: CachedComputePipelineId,

    /// Fullscreen depth-write (RT G-buffer → hardware depth); a **render** pipeline.
    pub gizmo_depth: CachedRenderPipelineId,

    /// DLSS-guide resolve (restir G-buffer → DLSS guide buffers).
    #[cfg(feature = "dlss")]
    pub dlss_resolve: CachedComputePipelineId,
}

/// Register every solari shader as an embedded asset. Called from
/// [`crate::SolariPlugin::build`] so the `embedded_asset!` here and the
/// `load_embedded_asset!` in [`init_solari_pipelines`] are co-located — the
/// embedded path matches by construction, no cross-module path drift.
pub fn embed_solari_shaders(app: &mut App) {
    embedded_asset!(app, "transform/transform_propagate.wgsl");
    embedded_asset!(app, "transform/transform_gather.wgsl");
    embedded_asset!(app, "transform/transform_readback.wgsl");
    embedded_asset!(app, "lights/light_resolve.wgsl");
    embedded_asset!(app, "accel/deform.wgsl");
    embedded_asset!(app, "accel/instantiate.wgsl");
    embedded_asset!(app, "render/atmosphere_bake.wgsl");
    embedded_asset!(app, "render/pathtracer/pathtracer.wgsl");
    embedded_asset!(app, "accel/selector.wgsl");
    embedded_asset!(app, "accel/blas_sharing.wgsl");
    embedded_asset!(app, "accel/ptlas_fill.wgsl");
    embedded_asset!(app, "render/visibility.wgsl");
    embedded_asset!(app, "render/presample.wgsl");
    embedded_asset!(app, "render/restir_pt.wgsl");
    embedded_asset!(app, "render/compose.wgsl");
    embedded_asset!(app, "render/caustics.wgsl");
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
    // Foundational, inserted at plugin build / `SolariSetup` — always present (the
    // pathtracer + AS pipelines composite their `@group(1)` layout with one of these).
    scene_bindings: Res<RaytracingSceneBindings>,
    scene_columns: Res<SceneColumns>,
    cluster_scene_layout: Res<ClusterSceneBindGroupLayout>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    let Some(resource_manager) = resource_manager else {
        return;
    };
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
    let pathtracer = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("pathtracer_pipeline".into()),
        layout: vec![
            scene_bindings.bind_group_layout.clone(),
            resource_manager.pathtracer.clone(),
            scene_columns
                .layout()
                .expect("scene-columns layout is built before RenderStartup")
                .clone(),
        ],
        shader: load_embedded_asset!(asset_server.as_ref(), "render/pathtracer/pathtracer.wgsl"),
        shader_defs: vec![ShaderDefVal::UInt(
            SCENE_COLUMNS_GROUP_DEF.into(),
            SCENE_COLUMNS_GROUP,
        )],
        entry_point: None,
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

    // ── ReSTIR realtime path: scene-bindings group + restir `@group(1)` + columns. ──
    let restir_columns = scene_columns
        .layout()
        .expect("scene-columns layout is built before RenderStartup")
        .clone();
    let restir_pl = vec![
        scene_bindings.bind_group_layout.clone(),
        resource_manager.restir.clone(),
        restir_columns,
    ];
    let restir_defs = vec![ShaderDefVal::UInt(
        SCENE_COLUMNS_GROUP_DEF.into(),
        SCENE_COLUMNS_GROUP,
    )];
    let visibility_shader = load_embedded_asset!(asset_server.as_ref(), "render/visibility.wgsl");
    let presample_shader = load_embedded_asset!(asset_server.as_ref(), "render/presample.wgsl");
    let restir_pt_shader = load_embedded_asset!(asset_server.as_ref(), "render/restir_pt.wgsl");
    let compose_shader = load_embedded_asset!(asset_server.as_ref(), "render/compose.wgsl");
    let restir_pipeline = |label: &'static str, entry: &'static str, shader: Handle<Shader>| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: restir_pl.clone(),
            shader,
            shader_defs: restir_defs.clone(),
            entry_point: Some(entry.into()),
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        })
    };
    let restir_visibility =
        restir_pipeline("restir_visibility_pipeline", "visibility", visibility_shader);
    let restir_presample =
        restir_pipeline("restir_presample_pipeline", "presample", presample_shader.clone());
    let restir_regir_decay =
        restir_pipeline("restir_regir_decay_pipeline", "regir_decay", presample_shader.clone());
    let restir_regir_fill =
        restir_pipeline("restir_regir_fill_pipeline", "regir_fill", presample_shader);
    let restir_initial_and_temporal = restir_pipeline(
        "restir_initial_and_temporal_pipeline",
        "initial_and_temporal",
        restir_pt_shader.clone(),
    );
    let restir_spatial_and_shade = restir_pipeline(
        "restir_spatial_and_shade_pipeline",
        "spatial_and_shade",
        restir_pt_shader.clone(),
    );
    let restir_debug = restir_pipeline(
        "restir_debug_pipeline",
        "restir_debug",
        restir_pt_shader.clone(),
    );
    let restir_specular_gi =
        restir_pipeline("restir_specular_gi_pipeline", "specular_gi", restir_pt_shader);
    let restir_compose = restir_pipeline("restir_compose_pipeline", "compose", compose_shader);
    let caustics_shader = load_embedded_asset!(asset_server.as_ref(), "render/caustics.wgsl");
    let restir_caustic_decay = restir_pipeline(
        "restir_caustic_decay_pipeline",
        "caustic_decay",
        caustics_shader.clone(),
    );
    let restir_caustic_emit = restir_pipeline(
        "restir_caustic_emit_pipeline",
        "caustic_emit",
        caustics_shader.clone(),
    );
    let restir_caustic_prepare_reset = restir_pipeline(
        "restir_caustic_prepare_reset_pipeline",
        "caustic_prepare_reset",
        caustics_shader.clone(),
    );
    let restir_caustic_prepare_reduce = restir_pipeline(
        "restir_caustic_prepare_reduce_pipeline",
        "caustic_prepare_reduce",
        caustics_shader.clone(),
    );
    let restir_caustic_prepare_finalize = restir_pipeline(
        "restir_caustic_prepare_finalize_pipeline",
        "caustic_prepare_finalize",
        caustics_shader,
    );

    // Fullscreen depth-write — a render pipeline; its bulky descriptor lives in
    // `render::gizmo_depth`, only the id lands here.
    let gizmo_depth = gizmo_depth_pipeline(
        &pipeline_cache,
        &fullscreen_shader,
        asset_server.as_ref(),
        resource_manager.gizmo_depth.clone(),
    );

    // DLSS-guide resolve — compute, but cfg-gated; its builder lives in `render::dlss`.
    #[cfg(feature = "dlss")]
    let dlss_resolve = crate::render::restir_dlss_resolve_pipeline(
        &pipeline_cache,
        asset_server.as_ref(),
        scene_bindings.bind_group_layout.clone(),
        resource_manager.dlss_resolve.clone(),
        scene_columns
            .layout()
            .expect("scene-columns layout is built before RenderStartup")
            .clone(),
    );

    commands.insert_resource(SolariPipelines {
        transform_propagate,
        transform_gather,
        transform_readback,
        light_resolve,
        deform,
        animated_blas,
        atmosphere,
        pathtracer,
        selector_reset,
        selector_main,
        blas_sharing_geom_reset,
        blas_sharing_classify,
        blas_sharing_elect_dirty,
        blas_sharing_finalize_count,
        blas_sharing_assign_address,
        ptlas_seed,
        ptlas_incremental,
        ptlas_finalize,
        restir_visibility,
        restir_presample,
        restir_regir_decay,
        restir_regir_fill,
        restir_initial_and_temporal,
        restir_spatial_and_shade,
        restir_specular_gi,
        restir_compose,
        restir_caustic_decay,
        restir_caustic_emit,
        restir_caustic_prepare_reset,
        restir_caustic_prepare_reduce,
        restir_caustic_prepare_finalize,
        restir_debug,
        gizmo_depth,
        #[cfg(feature = "dlss")]
        dlss_resolve,
    });
}
