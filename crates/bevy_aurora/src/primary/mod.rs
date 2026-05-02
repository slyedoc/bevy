//! Aurora's RT primary visibility pass — the layer that replaces
//! [`DeferredPrepass`] / [`DepthPrepass`] for meshlet-rendered entities.
//!
//! M-B sub-1 scaffolded the [`PgbufferTextures`] resource. M-B sub-2 (this
//! module) adds the actual compute pass: a camera ray per pixel against
//! Aurora's wrapped TLAS, written straight into the camera's view target as
//! a debug shading. M-B sub-3 will swap that simple write for a proper
//! PGBuffer fill (world_position / normal / motion / material_id / depth).

pub mod pgbuffer;
pub mod visibility;

use bevy_app::{App, Plugin};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    render_resource::WgpuFeatures, renderer::RenderDevice, Render, RenderApp, RenderStartup,
    RenderSystems,
};

pub use pgbuffer::{PgbufferExtent, PgbufferTextures};
pub use visibility::{AuroraCamera, AuroraPrimaryVisibilityPlugin};

/// Allocates [`PgbufferTextures`] at render startup. Per-camera sizing comes
/// when [`AuroraCamera`] gains a viewport tracker; for now the fixed
/// [`PgbufferExtent`] is used.
pub struct AuroraPgbufferPlugin;

impl Plugin for AuroraPgbufferPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<PgbufferExtent>()
            .add_systems(RenderStartup, init_pgbuffer)
            .add_systems(
                Render,
                resize_pgbuffer.in_set(RenderSystems::PrepareResources),
            );
    }
}

fn init_pgbuffer(
    mut commands: bevy_ecs::system::Commands,
    device: bevy_ecs::system::Res<RenderDevice>,
    extent: bevy_ecs::system::Res<PgbufferExtent>,
) {
    if !device
        .features()
        .contains(WgpuFeatures::EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE)
    {
        return;
    }
    let textures = PgbufferTextures::allocate(&device, extent.0);
    tracing::info!(
        target: "bevy_aurora",
        width = extent.0.x,
        height = extent.0.y,
        "PGBuffer textures allocated",
    );
    commands.insert_resource(textures);
}

fn resize_pgbuffer(
    mut commands: bevy_ecs::system::Commands,
    device: bevy_ecs::system::Res<RenderDevice>,
    extent: bevy_ecs::system::Res<PgbufferExtent>,
    pgbuffer: Option<bevy_ecs::system::Res<PgbufferTextures>>,
) {
    let needs_resize = match pgbuffer.as_deref() {
        Some(pg) => pg.extent != extent.0,
        None => false,
    };
    if !needs_resize {
        return;
    }
    let new_textures = PgbufferTextures::allocate(&device, extent.0);
    tracing::info!(
        target: "bevy_aurora",
        width = extent.0.x,
        height = extent.0.y,
        "PGBuffer textures reallocated",
    );
    commands.insert_resource(new_textures);
}
