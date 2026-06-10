//! Solari debug-view overlays (the [`SolariDebugView`](crate::debug::SolariDebugView)
//! visualizations), rendered over the Solari output.
//!
//! Each debug view is its **own run-condition-gated system** — `overlay::<V>`,
//! generic over an [`OverlayView`] marker, registered via [`add_overlay_view`].
//! A view's whole config — its `VIEW_*` shader_def(s) and which texture it
//! samples — lives in one [`OverlayView`] impl (declared with the
//! [`overlay_view!`] macro), so the shader variant and the bound texture can't
//! drift apart. Cluster-family views sample nothing: they trace their own
//! primary ray from the scene bind group (group 0).
//!
//! The render shaders (restir, pathtracer) carry no debug code. Overlays run in
//! `EarlyPostProcess`, after `restir`, before DLSS, so the chosen visualization
//! is overwritten onto the lit color and upscaled with the rest.

use core::marker::PhantomData;

#[cfg(feature = "dlss")]
use super::dlss::ViewRestirDlssTextures;
use super::prepare::RestirResources;
use crate::bindings::RaytracingSceneBindings;
use crate::render::view::{debug_is, SolariDebugView};
use crate::render::view_cull::{SolariViewOffset, SolariViewUniform, SolariViewUniforms};
use bevy_app::SubApp;
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_core_pipeline::schedule::{Core3d, Core3dSystems};
use bevy_diagnostic::FrameCount;
use bevy_ecs::{
    resource::Resource,
    schedule::{IntoScheduleConfigs, SystemSet},
    system::{Commands, Res},
};
use bevy_render::{
    render_resource::{
        binding_types::{texture_2d, texture_storage_2d, uniform_buffer},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
        PipelineCache,
        ShaderStages, StorageTextureAccess, TextureFormat, TextureSampleType, TextureView,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
    RenderStartup,
};
use bevy_shader::ShaderDefVal;
use bevy_utils::default;

/// All Solari debug-view overlays share this set so DLSS can be ordered after
/// them.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SolariDebugOverlay;

/// Inputs an [`OverlayView`] may read to choose its source texture.
pub struct OverlayInputs<'a> {
    res: &'a RestirResources,
    /// Current ping-pong slot (`frame_count & 1`).
    curr: usize,
    #[cfg(feature = "dlss")]
    dlss: Option<&'a ViewRestirDlssTextures>,
}

/// One debug view's overlay config. Implemented by a zero-size marker per view
/// (see the [`overlay_view!`] invocations); `overlay::<V>` is generic over it,
/// keeping a single shared body instead of one copy per view.
pub trait OverlayView: Send + Sync + 'static {
    /// The [`SolariDebugView`] this overlay renders — its run-condition key.
    const VIEW: SolariDebugView;
    /// `VIEW_*` shader_defs selecting the variant in `debug_overlay.wgsl`.
    const DEFS: &'static [&'static str];
    /// Texture this view samples, or `None` for cluster-family views (which
    /// trace their own ray and ignore the bound `selected` texture).
    fn selected<'a>(inputs: &OverlayInputs<'a>) -> Option<&'a TextureView>;
}

/// Declares an [`OverlayView`] marker: a view's shader_def(s) and its source
/// texture, colocated so they can't desync.
macro_rules! overlay_view {
    ($name:ident, $view:expr, [$($def:literal),*], |$i:ident| $sel:expr) => {
        pub struct $name;
        impl OverlayView for $name {
            const VIEW: SolariDebugView = $view;
            const DEFS: &'static [&'static str] = &[$($def),*];
            fn selected<'a>($i: &OverlayInputs<'a>) -> Option<&'a TextureView> {
                $sel
            }
        }
    };
}

// Buffer views: sample one G-buffer texture.
overlay_view!(WorldPositionView, SolariDebugView::WorldPosition, ["VIEW_COLOR"],
    |i| Some(&i.res.world_position[i.curr]));
overlay_view!(MaterialIdView, SolariDebugView::MaterialId, ["VIEW_MATERIAL_ID"],
    |i| Some(&i.res.world_position[i.curr])); // material id is packed in world_position.w
overlay_view!(WorldNormalView, SolariDebugView::WorldNormal, ["VIEW_NORMAL"],
    |i| Some(&i.res.world_normal[i.curr]));
overlay_view!(UvView, SolariDebugView::Uv, ["VIEW_COLOR"],
    |i| Some(&i.res.uv));
overlay_view!(MotionVectorsView, SolariDebugView::MotionVectors, ["VIEW_MOTION"],
    |i| Some(&i.res.motion_vectors));

// Cluster family: trace their own primary ray; sample nothing.
overlay_view!(LodView, SolariDebugView::Lod,
    ["VIEW_CLUSTER_FAMILY", "VIEW_LOD"], |_i| None);
overlay_view!(ClusterView, SolariDebugView::Cluster,
    ["VIEW_CLUSTER_FAMILY", "VIEW_CLUSTER"], |_i| None);
overlay_view!(TriangleView, SolariDebugView::Triangle,
    ["VIEW_CLUSTER_FAMILY", "VIEW_TRIANGLE"], |_i| None);
overlay_view!(GeometryCheckView, SolariDebugView::GeometryCheck,
    ["VIEW_CLUSTER_FAMILY", "VIEW_GEOMETRY_CHECK"], |_i| None);

// DLSS guide buffers (only registered when DLSS is active).
#[cfg(feature = "dlss")]
overlay_view!(DlssDepthView, SolariDebugView::DlssDepth, ["VIEW_GRAYSCALE"],
    |i| i.dlss.map(|d| &d.depth));
#[cfg(feature = "dlss")]
overlay_view!(DlssNormalRoughnessView, SolariDebugView::DlssNormalRoughness, ["VIEW_NORMAL"],
    |i| i.dlss.map(|d| &d.normal_roughness));
#[cfg(feature = "dlss")]
overlay_view!(DlssDiffuseAlbedoView, SolariDebugView::DlssDiffuseAlbedo, ["VIEW_COLOR"],
    |i| i.dlss.map(|d| &d.diffuse_albedo));
#[cfg(feature = "dlss")]
overlay_view!(DlssSpecularAlbedoView, SolariDebugView::DlssSpecularAlbedo, ["VIEW_COLOR"],
    |i| i.dlss.map(|d| &d.specular_albedo));
#[cfg(feature = "dlss")]
overlay_view!(DlssSpecularMotionView, SolariDebugView::DlssSpecularMotion, ["VIEW_MOTION"],
    |i| i.dlss.map(|d| &d.specular_motion_vectors));

/// The overlay compute pipeline for view `V` (one per view, keyed by type).
#[derive(Resource)]
pub struct OverlayPipeline<V: OverlayView>(CachedComputePipelineId, PhantomData<V>);

impl<V: OverlayView> OverlayPipeline<V> {
    /// View `V`'s overlay compute pipeline, or `None` until it has compiled (the
    /// dispatch bails that frame). Fetch-or-bail, cf. `MeshletPipelines::get`.
    #[inline]
    pub fn get<'a>(&self, cache: &'a PipelineCache) -> Option<&'a ComputePipeline> {
        cache.get_compute_pipeline(self.0)
    }
}

/// Group-1 layout shared by every overlay pipeline: `selected` texture, view
/// output, view uniform. Cheap to reconstruct (the cache dedups), so init and
/// the render system both call it rather than storing a resource.
fn overlay_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "solari_debug_overlay_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // 0: selected buffer (cluster views ignore it and trace instead)
                texture_2d(TextureSampleType::Float { filterable: false }),
                // 1: view output (the restir lit color, render res)
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite),
                // 2: view uniform
                uniform_buffer::<ViewUniform>(true),
                // 3: per-view RT cull mask (cluster-family views trace)
                uniform_buffer::<SolariViewUniform>(true),
            ),
        ),
    )
}

/// Queues view `V`'s overlay pipeline (with its `VIEW_*` shader_defs) at render
/// startup and stores it in an [`OverlayPipeline<V>`] resource.
pub fn init_overlay_pipeline<V: OverlayView>(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    scene_bindings: Res<RaytracingSceneBindings>,
    scene_columns: Res<crate::ecs_gpu::SceneColumns>,
    asset_server: Res<AssetServer>,
) {
    let mut shader_defs: Vec<ShaderDefVal> = V::DEFS.iter().map(|d| (*d).into()).collect();
    // group(2) = the shared scene-columns group (debug_overlay reads instance_cluster_ranges).
    shader_defs.push(ShaderDefVal::UInt(
        crate::ecs_gpu::SCENE_COLUMNS_GROUP_DEF.into(),
        2,
    ));
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("solari_debug_overlay_pipeline".into()),
        // group(0) = scene bindings (cluster views' ray trace + cluster data).
        layout: vec![
            scene_bindings.bind_group_layout.clone(),
            overlay_layout(),
            scene_columns
                .layout()
                .expect("scene-columns layout is built before RenderStartup")
                .clone(),
        ],
        shader: load_embedded_asset!(asset_server.as_ref(), "debug_overlay.wgsl"),
        shader_defs,
        entry_point: Some("debug_overlay".into()),
        ..default()
    });
    commands.insert_resource(OverlayPipeline::<V>(pipeline, PhantomData));
}

#[cfg(feature = "dlss")]
type OverlayViewData = (
    &'static RestirResources,
    &'static ViewTarget,
    &'static ViewUniformOffset,
    &'static SolariViewOffset,
    Option<&'static ViewRestirDlssTextures>,
);
#[cfg(not(feature = "dlss"))]
type OverlayViewData = (
    &'static RestirResources,
    &'static ViewTarget,
    &'static ViewUniformOffset,
    &'static SolariViewOffset,
);

/// Overwrites the lit color with view `V`'s visualization (render res). Gated by
/// `run_if(view_is(V::VIEW))`, so it only runs when `V` is the selected view.
pub fn overlay<V: OverlayView>(
    view: ViewQuery<OverlayViewData>,
    pipeline: Option<Res<OverlayPipeline<V>>>,
    pipeline_cache: Res<PipelineCache>,
    scene_bindings: Res<RaytracingSceneBindings>,
    scene_columns: Res<crate::ecs_gpu::SceneColumns>,
    view_uniforms: Res<ViewUniforms>,
    solari_view_uniforms: Res<SolariViewUniforms>,
    frame_count: Res<FrameCount>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    #[cfg(feature = "dlss")]
    let (resources, view_target, view_uniform_offset, solari_view_offset, dlss) = view.into_inner();
    #[cfg(not(feature = "dlss"))]
    let (resources, view_target, view_uniform_offset, solari_view_offset) = view.into_inner();

    let Some(pipeline) = pipeline else {
        return;
    };
    let (
        Some(pipeline),
        Some(scene_bind_group),
        Some(scene_columns_bind_group),
        Some(view_uniforms_binding),
        Some(solari_view_binding),
    ) = (
        pipeline.get(&pipeline_cache),
        &scene_bindings.bind_group,
        &scene_columns.bind_group,
        view_uniforms.uniforms.binding(),
        solari_view_uniforms.uniforms.binding(),
    ) else {
        return;
    };

    let curr = (frame_count.0 & 1) as usize;
    let inputs = OverlayInputs {
        res: resources,
        curr,
        #[cfg(feature = "dlss")]
        dlss,
    };
    // Cluster-family views trace their own ray — bind any texture as a
    // placeholder to satisfy the layout.
    let selected = V::selected(&inputs).unwrap_or(&resources.world_position[curr]);

    let bind_group = render_device.create_bind_group(
        "solari_debug_overlay_bind_group",
        &pipeline_cache.get_bind_group_layout(&overlay_layout()),
        &BindGroupEntries::sequential((
            selected,
            view_target.get_unsampled_color_attachment().view,
            view_uniforms_binding,
            solari_view_binding,
        )),
    );

    let dx = resources.view_size.x.div_ceil(8);
    let dy = resources.view_size.y.div_ceil(8);
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("solari_debug_overlay"),
            timestamp_writes: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, scene_bind_group, &[]);
    pass.set_bind_group(
        1,
        &bind_group,
        &[view_uniform_offset.offset, solari_view_offset.0],
    );
    pass.set_bind_group(2, scene_columns_bind_group, &[]);
    pass.dispatch_workgroups(dx, dy, 1);
}

/// Registers view `V`'s overlay: queues its pipeline at startup and adds its
/// run-condition-gated render system in `EarlyPostProcess`.
pub fn add_overlay_view<V: OverlayView>(render_app: &mut SubApp) {
    render_app
        .add_systems(RenderStartup, init_overlay_pipeline::<V>)
        .add_systems(
            Core3d,
            overlay::<V>
                .run_if(debug_is(V::VIEW))
                .in_set(Core3dSystems::EarlyPostProcess)
                .in_set(SolariDebugOverlay),
        );
}

/// Registers every always-available overlay (buffer + cluster families).
pub fn register_overlays(render_app: &mut SubApp) {
    add_overlay_view::<WorldPositionView>(render_app);
    add_overlay_view::<MaterialIdView>(render_app);
    add_overlay_view::<WorldNormalView>(render_app);
    add_overlay_view::<UvView>(render_app);
    add_overlay_view::<MotionVectorsView>(render_app);
    add_overlay_view::<LodView>(render_app);
    add_overlay_view::<ClusterView>(render_app);
    add_overlay_view::<TriangleView>(render_app);
    add_overlay_view::<GeometryCheckView>(render_app);
}

/// Registers the DLSS-guide overlays (only call when DLSS is active).
#[cfg(feature = "dlss")]
pub fn register_dlss_overlays(render_app: &mut SubApp) {
    add_overlay_view::<DlssDepthView>(render_app);
    add_overlay_view::<DlssNormalRoughnessView>(render_app);
    add_overlay_view::<DlssDiffuseAlbedoView>(render_app);
    add_overlay_view::<DlssSpecularAlbedoView>(render_app);
    add_overlay_view::<DlssSpecularMotionView>(render_app);
}
