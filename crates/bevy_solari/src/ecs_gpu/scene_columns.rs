//! The shared **scene-columns bind group** — `ecs_gpu` assembles one bind group
//! from every [`GpuColumn`] that declares a [`GpuColumnDesc::SCENE_BINDING`], so
//! the RT scene shaders read columns without the scene binder hand-wiring each
//! buffer. Adding a scene column = set its `SCENE_BINDING`; it then appears in the
//! group automatically.
//!
//! The group is **relocatable**: its WGSL block uses `@group(#{SOLARI_SCENE_COLUMNS_GROUP})`
//! (a naga_oil shader-def token), so each consumer pipeline assigns the group index
//! it has room for via [`SCENE_COLUMNS_GROUP_DEF`] and binds it there. One group for
//! all columns → no `maxBindGroups` pressure.

use bevy_app::{App, Plugin};
use bevy_ecs::{resource::Resource, schedule::IntoScheduleConfigs, world::World};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
        BindingType, Buffer, BufferBinding, BufferBindingType, PipelineCache, ShaderStages,
    },
    renderer::RenderDevice,
    Render, RenderApp, RenderSystems,
};
use core::num::NonZeroU64;

use super::column::{GpuColumn, GpuColumnDesc};

/// Shader-def name a consumer pipeline sets (`ShaderDefVal::UInt`) to the group
/// index it binds the scene-columns group at — substituted into
/// `@group(#{SOLARI_SCENE_COLUMNS_GROUP})` in the WGSL.
pub const SCENE_COLUMNS_GROUP_DEF: &str = "SOLARI_SCENE_COLUMNS_GROUP";

/// One column's contribution to the scene-columns group: its binding index and a
/// type-erased accessor that reads the column's current GPU buffer + the committed
/// byte size to bind (a range of exactly that size — see [`GpuColumn::committed_bytes`]).
struct SceneColumnEntry {
    binding: u32,
    get_buffer: fn(&World) -> Option<(Buffer, u64)>,
}

/// Render-world resource: the registered scene columns, their shared layout, and
/// the per-frame bind group consumers bind.
#[derive(Resource, Default)]
pub struct SceneColumns {
    entries: Vec<SceneColumnEntry>,
    /// Built from the registered entries (stable once all columns register); the
    /// layout consumer pipelines list and the bind group is created against.
    layout: Option<BindGroupLayoutDescriptor>,
    /// (Re)built each frame from the columns' current buffers.
    pub bind_group: Option<BindGroup>,
}

impl SceneColumns {
    /// The shared layout (registered columns as read-only storage at their
    /// `SCENE_BINDING`). Consumers list this in their pipeline layout.
    pub fn layout(&self) -> Option<&BindGroupLayoutDescriptor> {
        self.layout.as_ref()
    }

    fn rebuild_layout(&mut self) {
        self.entries.sort_by_key(|e| e.binding);
        let entries: Vec<BindGroupLayoutEntry> = self
            .entries
            .iter()
            .map(|e| BindGroupLayoutEntry {
                binding: e.binding,
                visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        self.layout = Some(BindGroupLayoutDescriptor::new(
            "solari_scene_columns_layout",
            &entries,
        ));
    }
}

/// Register column `C`'s scene binding(s) into [`SceneColumns`] — called from
/// [`GpuColumnPlugin<C>::build`](super::GpuColumnPlugin) when `C::SCENE_BINDING`
/// is set. Pushes a type-erased buffer accessor (and a previous-frame one for a
/// `KEEP_PREVIOUS` column with `SCENE_BINDING_PREVIOUS`) and rebuilds the layout.
pub(super) fn register_scene_column<C: GpuColumnDesc>(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app.init_resource::<SceneColumns>();
    let mut scene = render_app.world_mut().resource_mut::<SceneColumns>();
    if let Some(binding) = C::SCENE_BINDING {
        scene.entries.push(SceneColumnEntry {
            binding,
            get_buffer: |world| {
                world
                    .get_resource::<GpuColumn<C>>()
                    .map(|c| (c.buffer().clone(), c.committed_bytes()))
            },
        });
    }
    if let Some(binding) = C::SCENE_BINDING_PREVIOUS {
        scene.entries.push(SceneColumnEntry {
            binding,
            get_buffer: |world| {
                world
                    .get_resource::<GpuColumn<C>>()
                    .and_then(|c| c.previous_buffer().cloned().map(|b| (b, c.committed_bytes())))
            },
        });
    }
    scene.rebuild_layout();
}

/// `Render::PrepareBindGroups`: (re)build the scene-columns bind group from the
/// registered columns' current buffers. Exclusive because the accessors read
/// arbitrary `GpuColumn<C>` resources by type.
pub fn prepare_scene_columns_bind_group(world: &mut World) {
    // Snapshot the accessors so the `SceneColumns` borrow is dropped before we
    // read the (arbitrary) column resources.
    let accessors: Vec<(u32, fn(&World) -> Option<(Buffer, u64)>)> = {
        let scene = world.resource::<SceneColumns>();
        if scene.layout.is_none() || scene.entries.is_empty() {
            return;
        }
        scene.entries.iter().map(|e| (e.binding, e.get_buffer)).collect()
    };

    let mut buffers: Vec<(u32, Buffer, u64)> = Vec::with_capacity(accessors.len());
    for (binding, get) in &accessors {
        let Some((buffer, bytes)) = get(world) else {
            world.resource_mut::<SceneColumns>().bind_group = None;
            return; // a column's buffer isn't ready yet — retry next frame.
        };
        if bytes == 0 {
            // No pages committed yet (column not sized this frame) — retry.
            world.resource_mut::<SceneColumns>().bind_group = None;
            return;
        }
        buffers.push((*binding, buffer, bytes));
    }

    let device = world.resource::<RenderDevice>().clone();
    let layout = {
        let cache = world.resource::<PipelineCache>();
        let scene = world.resource::<SceneColumns>();
        cache.get_bind_group_layout(scene.layout.as_ref().unwrap())
    };
    let entries: Vec<BindGroupEntry> = buffers
        .iter()
        .map(|(binding, buffer, bytes)| BindGroupEntry {
            binding: *binding,
            // Bind exactly the committed range, NOT the whole sparse virtual buffer
            // — so the RT shaders' `arrayLength()` is the real slot count (binding
            // the full 1 GiB made `arrayLength(&directional_lights)` ~33M and hung
            // the path tracer's light loop).
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &**buffer,
                offset: 0,
                size: NonZeroU64::new(*bytes),
            }),
        })
        .collect();
    let bind_group = device.create_bind_group("solari_scene_columns", &layout, &entries);
    world.resource_mut::<SceneColumns>().bind_group = Some(bind_group);
}

/// Inits [`SceneColumns`] and schedules its per-frame bind-group build. Added once
/// by [`crate::SolariPlugin`]; columns register themselves via their column plugin.
pub struct SceneColumnsPlugin;

impl Plugin for SceneColumnsPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<SceneColumns>()
            .add_systems(
                Render,
                prepare_scene_columns_bind_group.in_set(RenderSystems::PrepareBindGroups),
            );
    }
}
