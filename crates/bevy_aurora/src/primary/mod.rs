//! Aurora's RT primary visibility pass — the layer that replaces
//! [`DeferredPrepass`] / [`DepthPrepass`] for meshlet-rendered entities.
//!
//! Traces a camera ray per pixel against the TLAS managed by
//! [`crate::scene::TlasManager`] and writes a "primary G-buffer" (`PGBuffer`) of
//! world position / world normal / screen motion / material id / linear depth.
//! Downstream Aurora passes (lighting, compose) read `PGBuffer` the same way
//! they would read raster G-buffer in solari.
//!
//! # Milestone state
//!
//! M-B sub-1 (this commit): allocate `PGBuffer` textures + plugin scaffolding.
//! No compute shader, no ray query, no dispatch yet — those land in
//! M-B sub-2 (compute pipeline) and M-B sub-3 (visibility WGSL).

pub mod pgbuffer;

use bevy_app::{App, Plugin};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    render_resource::WgpuFeatures, renderer::RenderDevice, Render, RenderApp, RenderStartup,
    RenderSystems,
};

pub use pgbuffer::{PgbufferExtent, PgbufferTextures};

/// Allocates [`PgbufferTextures`] at render startup and (later) drives the RT
/// primary visibility compute pass.
pub struct AuroraPrimaryVisibilityPlugin;

impl Plugin for AuroraPrimaryVisibilityPlugin {
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

/// Allocate `PGBuffer` textures at render startup, sized to the default extent.
/// Real per-camera sizing comes when [`crate::scene::components::AuroraCamera`]
/// lands in the next sub-milestone.
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

/// React to changes in [`PgbufferExtent`] by reallocating the `PGBuffer`
/// textures. M-B sub-1 only fires when an external system mutates the extent;
/// the auto-resize-on-camera-viewport-change path comes with `AuroraCamera`.
fn resize_pgbuffer(
    mut commands: bevy_ecs::system::Commands,
    device: bevy_ecs::system::Res<RenderDevice>,
    extent: bevy_ecs::system::Res<PgbufferExtent>,
    pgbuffer: Option<bevy_ecs::system::Res<PgbufferTextures>>,
) {
    let needs_resize = match pgbuffer.as_deref() {
        Some(pg) => pg.extent != extent.0,
        None => false, // init_pgbuffer handles first allocation
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
