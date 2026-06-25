use crate::{
    init_line_gizmo_uniform_bind_group_layout, line_gizmo_vertex_buffer_layouts,
    line_joint_gizmo_vertex_buffer_layouts, DrawLineGizmo, DrawLineJointGizmo, GizmoRenderSystems,
    GpuLineGizmo, LineGizmoEntities, LineGizmoUniformBindgroupLayout, SetLineGizmoBindGroup,
};
use bevy_app::{App, Plugin};
use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_camera::visibility::RenderLayers;
use bevy_core_pipeline::core_3d::{Transparent3d, TransparentSortingInfo3d, CORE_3D_DEPTH_FORMAT};
use bevy_gizmos::config::{GizmoLineJoint, GizmoLineStyle, GizmoMeshConfig};

use bevy_ecs::{
    error::BevyError,
    prelude::Entity,
    query::ROQueryItem,
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{
        lifetimeless::{Read, SRes},
        Commands, Query, Res, ResMut, SystemParamItem,
    },
};
use bevy_render::{
    render_asset::{prepare_assets, RenderAssets},
    render_phase::{
        AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
        RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
    },
    render_resource::{binding_types::uniform_buffer, *},
    renderer::RenderDevice,
    view::{ExtractedView, Msaa, ViewUniform, ViewUniformOffset, ViewUniforms},
    Render, RenderApp, RenderSystems,
};
use bevy_render::{sync_world::MainEntity, GpuResourceAppExt, RenderStartup};
use bevy_shader::Shader;
use bevy_utils::default;
use tracing::error;

pub struct LineGizmo3dPlugin;
impl Plugin for LineGizmo3dPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_command::<Transparent3d, DrawLineGizmo3d>()
            .add_render_command::<Transparent3d, DrawLineGizmo3dStrip>()
            .add_render_command::<Transparent3d, DrawLineJointGizmo3d>()
            .init_gpu_resource::<SpecializedRenderPipelines<LineJointGizmoPipeline>>()
            .configure_sets(
                Render,
                GizmoRenderSystems::QueueLineGizmos3d.in_set(RenderSystems::Queue),
            )
            .add_systems(
                RenderStartup,
                init_line_gizmo_pipelines.after(init_line_gizmo_uniform_bind_group_layout),
            )
            .add_systems(
                Render,
                prepare_gizmo_view_bind_group.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                (queue_line_gizmos_3d, queue_line_joint_gizmos_3d)
                    .in_set(GizmoRenderSystems::QueueLineGizmos3d)
                    .after(prepare_assets::<GpuLineGizmo>),
            );
    }
}

/// The gizmo view bind-group layout: a single `View` uniform at `@binding(0)`.
///
/// 3D gizmos only read `View` (clip/view matrices + viewport); they do NOT need
/// the full mesh view bind group (lights/shadows/clusters), so they no longer
/// depend on `bevy_pbr`'s `MeshPipeline`. This lets gizmos render with `PbrPlugin`
/// disabled (e.g. a pure ray-traced view), and is identical for the PBR path.
#[derive(Resource)]
struct GizmoViewLayout {
    layout: BindGroupLayoutDescriptor,
}

/// The per-frame gizmo view bind group (the `View` uniform), bound with each
/// view's dynamic offset by [`SetGizmoViewBindGroup`].
#[derive(Resource)]
struct GizmoViewBindGroup {
    bindgroup: BindGroup,
}

fn init_line_gizmo_pipelines(
    mut commands: Commands,
    uniform_bind_group_layout: Res<LineGizmoUniformBindgroupLayout>,
    asset_server: Res<AssetServer>,
) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "GizmoView layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    let line_shader = load_embedded_asset!(asset_server.as_ref(), "lines.wgsl");
    let variants_line = Variants::new(
        LineGizmoPipelineSpecializer,
        RenderPipelineDescriptor {
            label: Some("LineGizmo 3d Pipeline".into()),
            vertex: VertexState {
                shader: line_shader.clone(),
                ..default()
            },
            fragment: Some(FragmentState {
                shader: line_shader,
                ..default()
            }),
            layout: vec![view_layout.clone(), uniform_bind_group_layout.layout.clone()],
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Greater),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            ..default()
        },
    );

    commands.insert_resource(LineGizmoPipeline {
        variants: variants_line,
    });
    commands.insert_resource(LineJointGizmoPipeline {
        view_layout: view_layout.clone(),
        uniform_layout: uniform_bind_group_layout.layout.clone(),
        shader: load_embedded_asset!(asset_server.as_ref(), "line_joints.wgsl"),
    });
    commands.insert_resource(GizmoViewLayout {
        layout: view_layout,
    });
}

/// `PrepareBindGroups`: build the gizmo view bind group from [`ViewUniforms`]
/// (core to `bevy_render`, present with or without `bevy_pbr`).
fn prepare_gizmo_view_bind_group(
    mut commands: Commands,
    view_layout: Res<GizmoViewLayout>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    view_uniforms: Res<ViewUniforms>,
) {
    if let Some(binding) = view_uniforms.uniforms.binding() {
        commands.insert_resource(GizmoViewBindGroup {
            bindgroup: render_device.create_bind_group(
                "GizmoView bindgroup",
                &pipeline_cache.get_bind_group_layout(&view_layout.layout),
                &BindGroupEntries::single(binding),
            ),
        });
    }
}

#[derive(Resource)]
struct LineGizmoPipeline {
    variants: Variants<RenderPipeline, LineGizmoPipelineSpecializer>,
}

struct LineGizmoPipelineSpecializer;

#[derive(PartialEq, Eq, Hash, Clone, SpecializerKey)]
struct LineGizmoPipelineKey {
    msaa_samples: u32,
    format: TextureFormat,
    strip: bool,
    perspective: bool,
    line_style: GizmoLineStyle,
}

impl Specializer<RenderPipeline> for LineGizmoPipelineSpecializer {
    type Key = LineGizmoPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        descriptor: &mut RenderPipelineDescriptor,
    ) -> Result<Canonical<Self::Key>, BevyError> {
        descriptor.vertex.buffers = line_gizmo_vertex_buffer_layouts(key.strip);
        descriptor.multisample.count = key.msaa_samples;

        let fragment = descriptor.fragment_mut()?;

        #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
        fragment.shader_defs.push("SIXTEEN_BYTE_ALIGNMENT".into());

        if key.perspective {
            fragment.shader_defs.push("PERSPECTIVE".into());
        }

        let fragment_entry_point = match key.line_style {
            GizmoLineStyle::Solid => "fragment_solid",
            GizmoLineStyle::Dotted => "fragment_dotted",
            GizmoLineStyle::Dashed { .. } => "fragment_dashed",
            _ => unimplemented!(),
        };

        fragment.entry_point = Some(fragment_entry_point.into());

        fragment.set_target(
            0,
            ColorTargetState {
                format: key.format,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            },
        );

        Ok(key)
    }
}

#[derive(Clone, Resource)]
struct LineJointGizmoPipeline {
    view_layout: BindGroupLayoutDescriptor,
    uniform_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct LineJointGizmoPipelineKey {
    msaa_samples: u32,
    format: TextureFormat,
    perspective: bool,
    joints: GizmoLineJoint,
}

impl SpecializedRenderPipeline for LineJointGizmoPipeline {
    type Key = LineJointGizmoPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = vec![
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            "SIXTEEN_BYTE_ALIGNMENT".into(),
        ];

        if key.perspective {
            shader_defs.push("PERSPECTIVE".into());
        }

        let layout = vec![self.view_layout.clone(), self.uniform_layout.clone()];

        if key.joints == GizmoLineJoint::None {
            error!("There is no entry point for line joints with GizmoLineJoints::None. Please consider aborting the drawing process before reaching this stage.");
        };

        let entry_point = match key.joints {
            GizmoLineJoint::Miter => "vertex_miter",
            GizmoLineJoint::Round(_) => "vertex_round",
            GizmoLineJoint::None | GizmoLineJoint::Bevel => "vertex_bevel",
        };

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                entry_point: Some(entry_point.into()),
                shader_defs: shader_defs.clone(),
                buffers: line_joint_gizmo_vertex_buffer_layouts(),
                constants: vec![],
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            layout,
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Greater),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: key.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            label: Some("LineJointGizmo 3d Pipeline".into()),
            ..default()
        }
    }
}

/// Binds the gizmo [`View`](bevy_render::view::View) uniform at `@group(I)` with
/// the view's dynamic offset. Replaces `bevy_pbr`'s `SetMeshViewBindGroup` so 3D
/// gizmos don't pull in the mesh view bind group (and thus `PbrPlugin`).
struct SetGizmoViewBindGroup<const I: usize>;

impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetGizmoViewBindGroup<I> {
    type Param = SRes<GizmoViewBindGroup>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        view_uniform_offset: ROQueryItem<'w, '_, Self::ViewQuery>,
        _entity: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(
            I,
            &bind_group.into_inner().bindgroup,
            &[view_uniform_offset.offset],
        );
        RenderCommandResult::Success
    }
}

type DrawLineGizmo3d = (
    SetItemPipeline,
    SetGizmoViewBindGroup<0>,
    SetLineGizmoBindGroup<1>,
    DrawLineGizmo<false>,
);
type DrawLineGizmo3dStrip = (
    SetItemPipeline,
    SetGizmoViewBindGroup<0>,
    SetLineGizmoBindGroup<1>,
    DrawLineGizmo<true>,
);
type DrawLineJointGizmo3d = (
    SetItemPipeline,
    SetGizmoViewBindGroup<0>,
    SetLineGizmoBindGroup<1>,
    DrawLineJointGizmo,
);

fn queue_line_gizmos_3d(
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    mut pipeline: ResMut<LineGizmoPipeline>,
    pipeline_cache: Res<PipelineCache>,
    line_gizmos: Query<(Entity, &GizmoMeshConfig)>,
    line_gizmo_assets: Res<RenderAssets<GpuLineGizmo>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<(&ExtractedView, &Msaa, Option<&RenderLayers>)>,
    line_gizmo_entities: Res<LineGizmoEntities>,
) -> Result<(), BevyError> {
    let draw_function = draw_functions.read().get_id::<DrawLineGizmo3d>().unwrap();
    let draw_function_strip = draw_functions
        .read()
        .get_id::<DrawLineGizmo3dStrip>()
        .unwrap();

    for (view, msaa, render_layers) in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let render_layers = render_layers.unwrap_or_default();
        let msaa_samples = msaa.samples();
        let format = view.target_format;

        for (entity, config) in &line_gizmos {
            if !config.render_layers.intersects(render_layers) {
                continue;
            }

            let Some(line_gizmo) = line_gizmo_assets.get(&config.handle) else {
                continue;
            };

            if line_gizmo.list_vertex_count > 0 {
                let pipeline = pipeline.variants.specialize(
                    &pipeline_cache,
                    LineGizmoPipelineKey {
                        msaa_samples,
                        format,
                        strip: false,
                        perspective: config.line_perspective,
                        line_style: config.line_style,
                    },
                )?;
                transparent_phase.add_transient(Transparent3d {
                    sorting_info: TransparentSortingInfo3d::AlwaysOnTop,
                    entity: (entity, line_gizmo_entities.line_gizmo_renderer),
                    draw_function,
                    pipeline,
                    distance: 0.,
                    batch_range: 0..1,
                    extra_index: PhaseItemExtraIndex::None,
                    indexed: true,
                });
            }

            if line_gizmo.strip_vertex_count >= 2 {
                let pipeline = pipeline.variants.specialize(
                    &pipeline_cache,
                    LineGizmoPipelineKey {
                        msaa_samples,
                        format,
                        strip: true,
                        perspective: config.line_perspective,
                        line_style: config.line_style,
                    },
                )?;
                transparent_phase.add_transient(Transparent3d {
                    sorting_info: TransparentSortingInfo3d::AlwaysOnTop,
                    entity: (entity, line_gizmo_entities.line_strip_gizmo_renderer),
                    draw_function: draw_function_strip,
                    pipeline,
                    distance: 0.,
                    batch_range: 0..1,
                    extra_index: PhaseItemExtraIndex::None,
                    indexed: true,
                });
            }
        }
    }

    Ok(())
}

fn queue_line_joint_gizmos_3d(
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    pipeline: Res<LineJointGizmoPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<LineJointGizmoPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    line_gizmos: Query<(Entity, &MainEntity, &GizmoMeshConfig)>,
    line_gizmo_assets: Res<RenderAssets<GpuLineGizmo>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<(&ExtractedView, &Msaa, Option<&RenderLayers>)>,
    line_gizmo_entities: Res<LineGizmoEntities>,
) {
    let draw_function = draw_functions
        .read()
        .get_id::<DrawLineJointGizmo3d>()
        .unwrap();

    for (view, msaa, render_layers) in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let render_layers = render_layers.unwrap_or_default();
        let msaa_samples = msaa.samples();
        let format = view.target_format;

        for (entity, _, config) in &line_gizmos {
            if !config.render_layers.intersects(render_layers) {
                continue;
            }

            let Some(line_gizmo) = line_gizmo_assets.get(&config.handle) else {
                continue;
            };

            if line_gizmo.strip_vertex_count < 3 || config.line_joints == GizmoLineJoint::None {
                continue;
            }

            let pipeline = pipelines.specialize(
                &pipeline_cache,
                &pipeline,
                LineJointGizmoPipelineKey {
                    msaa_samples,
                    format,
                    perspective: config.line_perspective,
                    joints: config.line_joints,
                },
            );

            transparent_phase.add_transient(Transparent3d {
                sorting_info: TransparentSortingInfo3d::AlwaysOnTop,
                entity: (entity, line_gizmo_entities.line_joint_gizmo_renderer),
                draw_function,
                pipeline,
                distance: 0.,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
        }
    }
}
