#![allow(
    clippy::doc_markdown,
    reason = "Aurora identifiers (TLAS, PGBuffer, etc.) need not be backticked."
)]
//! Aurora's RT primary visibility compute pass.
//!
//! For cameras tagged with [`AuroraCamera`], dispatches a compute shader that
//! traces a camera ray per pixel against Aurora's wrapped TLAS and writes a
//! debug colour directly into the camera's view target.
//!
//! M-B sub-2 scope: validates the wrapped-TLAS binding works end-to-end.
//! Real PGBuffer fill (proper world_position / normal / motion / material_id /
//! depth channels) lands in M-B sub-3 once we know the shape of the
//! material LUT (see plan.md §10).

use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer};
use bevy_camera::CameraMainTextureUsages;
use bevy_core_pipeline::{
    core_3d::main_opaque_pass_3d,
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::{
    component::Component,
    query::With,
    reflect::ReflectComponent,
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res},
};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    render_resource::{
        binding_types::{acceleration_structure, texture_storage_2d, uniform_buffer},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
        ShaderStages, StorageTextureAccess, TextureFormat, TextureUsages,
    },
    renderer::{RenderContext, RenderDevice},
    sync_world::RenderEntity,
    view::{ExtractedView, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
    ExtractSchedule, MainWorld, RenderApp, RenderStartup,
};
use bevy_ecs::system::ResMut;

use crate::scene::TlasManager;

/// Component that opts a camera into Aurora's RT primary visibility pass.
///
/// Requires the camera's view target to be writable as a storage texture, so
/// adding this also adds [`CameraMainTextureUsages`] with `STORAGE_BINDING`.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default, Clone)]
#[require(CameraMainTextureUsages = CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING))]
pub struct AuroraCamera;


/// Cached compute pipeline + bind group layout for the trace pass.
#[derive(Resource)]
pub struct AuroraTracePipeline {
    pub layout: BindGroupLayoutDescriptor,
    pub pipeline_id: CachedComputePipelineId,
}

pub struct AuroraPrimaryVisibilityPlugin;

impl Plugin for AuroraPrimaryVisibilityPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<AuroraCamera>();
        embedded_asset!(app, "visibility.wgsl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_trace_pipeline)
            .add_systems(ExtractSchedule, extract_aurora_camera)
            .add_systems(
                Core3d,
                aurora_trace
                    .in_set(Core3dSystems::MainPass)
                    .before(main_opaque_pass_3d),
            );
    }
}

fn extract_aurora_camera(mut main_world: ResMut<MainWorld>, mut commands: Commands) {
    let mut q = main_world.query::<(RenderEntity, &AuroraCamera)>();
    for (render_entity, aurora) in q.iter_mut(&mut main_world) {
        if let Ok(mut e) = commands.get_entity(render_entity) {
            e.insert(*aurora);
        }
    }
}

fn init_trace_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "aurora_trace_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<ViewUniform>(true),
                acceleration_structure(),
            ),
        ),
    );

    let shader = load_embedded_asset!(asset_server.as_ref(), "visibility.wgsl");
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("aurora_trace_pipeline".into()),
        layout: vec![layout.clone()],
        immediate_size: 0,
        shader,
        shader_defs: vec![],
        entry_point: Some("trace".into()),
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(AuroraTracePipeline {
        layout,
        pipeline_id,
    });
}

#[allow(clippy::too_many_arguments)]
fn aurora_trace(
    cameras: Query<
        (&ViewTarget, &ViewUniformOffset, &ExtractedView),
        With<AuroraCamera>,
    >,
    pipeline: Option<Res<AuroraTracePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    tlas_manager: Option<Res<TlasManager>>,
    view_uniforms: Res<ViewUniforms>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    let Some(tlas_manager) = tlas_manager else {
        return;
    };
    let Some(tlas) = tlas_manager.tlas() else {
        // TLAS not yet built (mesh upload still in flight, etc.).
        return;
    };
    let Some(trace_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline_id) else {
        return;
    };
    let Some(view_uniforms_binding) = view_uniforms.uniforms.binding() else {
        return;
    };

    for (view_target, view_uniform_offset, _extracted_view) in cameras.iter() {
        let attachment = view_target.get_unsampled_color_attachment();

        let bind_group = render_device.create_bind_group(
            "aurora_trace_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::sequential((
                attachment.view,
                view_uniforms_binding.clone(),
                tlas.as_binding(),
            )),
        );

        let target_size = view_target.main_texture().size();
        let group_count_x = target_size.width.div_ceil(8);
        let group_count_y = target_size.height.div_ceil(8);

        let command_encoder = ctx.command_encoder();
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("aurora_trace_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(trace_pipeline);
        pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        pass.dispatch_workgroups(group_count_x, group_count_y, 1);
    }

}
