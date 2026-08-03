//! Centralized bind-group **layouts** for `bevy_solari` — one
//! [`SolariResourceManager`] resource owning every pass's
//! [`BindGroupLayoutDescriptor`], built once at `RenderStartup`. The companion to
//! [`crate::pipelines::SolariPipelines`] (which owns the compiled pipeline *ids*):
//! together they are the two halves of `bevy_pbr`'s meshlet split —
//! `MeshletPipelines` (ids) + `ResourceManager` (layouts + shared resources).
//!
//! Only the **layouts** are centralized here. A layout is an immutable descriptor
//! with no per-frame state, so pooling them in one resource removes the per-pass
//! `layout` scatter with zero contention. Each pass's *mutable* buffers (with their
//! growth, staging, and cold-start latches) deliberately stay on the pass resource:
//! folding those into one `ResMut` god-resource would serialize the prepare systems
//! and lose per-pass change detection — and meshlet doesn't do it either (its
//! mutable state lives in `InstanceManager` / `MeshletViewResources`).
//!
//! Each layout's descriptor is built by a `*_bind_group_layout()` free function
//! co-located with its pass (next to the param type and the bind-group code that
//! must agree with it); this resource just calls them and owns the results.
//! [`init_solari_pipelines`](crate::pipelines::init_solari_pipelines) reads these
//! layouts to queue each pipeline.

use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res},
};
use bevy_render::render_resource::BindGroupLayoutDescriptor;

use crate::gpu::allocator::Allocator;

/// Every solari pass bind-group layout, built once at `RenderStartup`. Field names
/// mirror [`SolariPipelines`](crate::pipelines::SolariPipelines) 1:1, so a pass's
/// id (`pipelines.gizmo_depth`) and its layout (`resource_manager.gizmo_depth`)
/// share a key.
///
/// Absent on devices lacking the cluster support solari needs (the builder gates
/// on the [`Allocator`], exactly as the pass resources do), so the pipelines that
/// read it — and every dispatch's `run_if(resource_exists::<SolariResourceManager>)`
/// — never come into existence there.
#[derive(Resource)]
pub struct SolariResourceManager {
    /// The fullscreen gizmo-depth bridge `@group(0)` layout (rt output + view).
    pub gizmo_depth: BindGroupLayoutDescriptor,
}

/// `RenderStartup` (after `SolariSetup`): build every pass's bind-group layout.
/// Layout descriptors need no GPU device — the [`Allocator`] is read only as the
/// device-support gate (its absence means the cluster features are missing, so
/// solari produces nothing on this device).
pub fn init_solari_resource_manager(mut commands: Commands, allocator: Option<Res<Allocator>>) {
    if allocator.is_none() {
        return;
    }
    commands.insert_resource(SolariResourceManager {
        gizmo_depth: crate::render::gizmo_depth::gizmo_depth_bind_group_layout(),
    });
}
