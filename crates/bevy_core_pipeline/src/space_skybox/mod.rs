use bevy::{
    asset::load_internal_asset,
    ecs::query::QueryItem,
    prelude::*,
    reflect::TypePath,
    render::{
        camera::Exposure,
        extract_component::*,
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::RenderDevice,
        texture::BevyDefault,
        view::*,
        Render, RenderApp, RenderSet,
    },
};


const SKYBOX_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(5052371412819132759);

pub struct SpaceSkyboxPlugin;

impl Plugin for SpaceSkyboxPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SKYBOX_SHADER_HANDLE, "space.wgsl", Shader::from_wgsl);

        app.add_plugins((
            ExtractComponentPlugin::<SpaceSkybox>::default(),
            UniformComponentPlugin::<SpaceSkyboxUniforms>::default(),
        ));

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<SpaceSkyboxPipeline>>()
            .add_systems(
                Render,
                (
                    prepare_skybox_pipelines.in_set(RenderSet::Prepare),
                    prepare_skybox_bind_groups.in_set(RenderSet::PrepareBindGroups),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let render_device = render_app.world.resource::<RenderDevice>().clone();

        render_app.insert_resource(SpaceSkyboxPipeline::new(&render_device));
    }
}

#[derive(Component, Clone)]
pub struct SpaceSkybox {
    pub image: Handle<Image>,
    /// Scale factor applied to the skybox image.
    /// After applying this multiplier to the image samples, the resulting values should
    /// be in units of [cd/m^2](https://en.wikipedia.org/wiki/Candela_per_square_metre).
    pub brightness: f32,
}

// TODO: Replace with a push constant once WebGPU gets support for that
#[derive(Component, ShaderType, Clone)]
pub struct SpaceSkyboxUniforms {
    brightness: f32,
}

impl ExtractComponent for SpaceSkybox {
    type QueryData = (&'static Self, Option<&'static Exposure>);
    type QueryFilter = ();
    type Out = (Self, SpaceSkyboxUniforms);

    fn extract_component((skybox, exposure): QueryItem<'_, Self::QueryData>) -> Option<Self::Out> {
        let exposure = exposure
            .map(|e| e.exposure())
            .unwrap_or_else(|| Exposure::default().exposure());

        Some((
            skybox.clone(),
            SpaceSkyboxUniforms {
                brightness: skybox.brightness * exposure,
            },
        ))
    }
}

#[derive(Resource)]
pub(super) struct SpaceSkyboxPipeline {
    bind_group_layout: BindGroupLayout,
}

impl SpaceSkyboxPipeline {
    pub fn new(render_device: &RenderDevice) -> Self {
        Self {
            bind_group_layout: render_device.create_bind_group_layout(
                "skybox_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::FRAGMENT,
                    (
                        texture_cube(TextureSampleType::Float { filterable: true }),
                        sampler(SamplerBindingType::Filtering),
                        uniform_buffer::<ViewUniform>(true)
                            .visibility(ShaderStages::VERTEX_FRAGMENT),
                        uniform_buffer::<SpaceSkyboxUniforms>(true),
                    ),
                ),
            ),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub(super) struct SpaceSkyboxPipelineKey {
    hdr: bool,
    samples: u32,
    depth_format: TextureFormat,
}

impl SpecializedRenderPipeline for SpaceSkyboxPipeline {
    type Key = SpaceSkyboxPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("skybox_pipeline".into()),
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            vertex: VertexState {
                shader: SKYBOX_SHADER_HANDLE,
                shader_defs: Vec::new(),
                entry_point: "skybox_vertex".into(),
                buffers: Vec::new(),
            },
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: key.depth_format,
                depth_write_enabled: false,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: key.samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                shader: SKYBOX_SHADER_HANDLE,
                shader_defs: Vec::new(),
                entry_point: "skybox_fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: if key.hdr {
                        ViewTarget::TEXTURE_FORMAT_HDR
                    } else {
                        TextureFormat::bevy_default()
                    },
                    // BlendState::REPLACE is not needed here, and None will be potentially much faster in some cases.
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
        }
    }
}

#[derive(Component)]
pub struct SpaceSkyboxPipelineId(pub CachedRenderPipelineId);

fn prepare_skybox_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SpaceSkyboxPipeline>>,
    pipeline: Res<SpaceSkyboxPipeline>,
    msaa: Res<Msaa>,
    views: Query<(Entity, &ExtractedView), With<SpaceSkybox>>,
) {
    for (entity, view) in &views {
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            SpaceSkyboxPipelineKey {
                hdr: view.hdr,
                samples: msaa.samples(),
                depth_format: bevy::core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT,
            },
        );

        commands
            .entity(entity)
            .insert(SpaceSkyboxPipelineId(pipeline_id));
    }
}

#[derive(Component)]
pub struct SpaceSkyboxBindGroup(pub (BindGroup, u32));

fn prepare_skybox_bind_groups(
    mut commands: Commands,
    pipeline: Res<SpaceSkyboxPipeline>,
    view_uniforms: Res<ViewUniforms>,
    skybox_uniforms: Res<ComponentUniforms<SpaceSkyboxUniforms>>,
    images: Res<RenderAssets<Image>>,
    render_device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &SpaceSkybox,
        &DynamicUniformIndex<SpaceSkyboxUniforms>,
    )>,
) {
    for (entity, skybox, skybox_uniform_index) in &views {
        if let (Some(skybox), Some(view_uniforms), Some(skybox_uniforms)) = (
            images.get(&skybox.image),
            view_uniforms.uniforms.binding(),
            skybox_uniforms.binding(),
        ) {
            let bind_group = render_device.create_bind_group(
                "skybox_bind_group",
                &pipeline.bind_group_layout,
                &BindGroupEntries::sequential((
                    &skybox.texture_view,
                    &skybox.sampler,
                    view_uniforms,
                    skybox_uniforms,
                )),
            );

            commands.entity(entity).insert(SpaceSkyboxBindGroup((
                bind_group,
                skybox_uniform_index.index(),
            )));
        }
    }
}
