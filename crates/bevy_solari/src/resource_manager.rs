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

use crate::accel::{
    blas_sharing_bind_group_layout, ptlas_bind_group_layout, selector_bind_group_layout,
};
use crate::gpu::allocator::Allocator;
use crate::lights::light_resolve_bind_group_layout;
use crate::render::atmosphere::atmosphere_bind_group_layout;
use crate::transform::{
    transform_gather_bind_group_layout, transform_propagate_bind_group_layout,
    transform_readback_bind_group_layout,
};

/// Every solari pass bind-group layout, built once at `RenderStartup`. Field names
/// mirror [`SolariPipelines`](crate::pipelines::SolariPipelines) 1:1, so a pass's
/// id (`pipelines.selector`) and its layout (`resource_manager.selector`) share a key.
///
/// Absent on devices lacking the cluster support solari needs (the builder gates
/// on the [`Allocator`], exactly as the pass resources do), so the pipelines that
/// read it — and every dispatch's `run_if(resource_exists::<SolariResourceManager>)`
/// — never come into existence there.
#[derive(Resource)]
pub struct SolariResourceManager {
    pub transform_propagate: BindGroupLayoutDescriptor,
    pub transform_gather: BindGroupLayoutDescriptor,
    pub transform_readback: BindGroupLayoutDescriptor,
    pub light_resolve: BindGroupLayoutDescriptor,
    pub atmosphere: BindGroupLayoutDescriptor,
    // AS passes — each is the `@group(1)` layout; the full pipeline layout pairs it
    // with the cluster-scene group (composited in `init_solari_pipelines`).
    pub selector: BindGroupLayoutDescriptor,
    pub blas_sharing: BindGroupLayoutDescriptor,
    pub ptlas: BindGroupLayoutDescriptor,
    /// The hair PTLAS-write `@group(0)` layout.
    pub ptlas_hair_write: BindGroupLayoutDescriptor,
    /// The batch ray-query `@group(1)` I/O layout (rays / hits / params).
    pub ray_query_io: BindGroupLayoutDescriptor,
    /// The fullscreen gizmo-depth bridge `@group(0)` layout (rt output + view).
    pub gizmo_depth: BindGroupLayoutDescriptor,
    /// The skeletal-deform compute `@group(0)` layout.
    pub deform: BindGroupLayoutDescriptor,
    /// The animated CLAS-instantiate compute `@group(0)` layout.
    pub animated_blas: BindGroupLayoutDescriptor,
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
        transform_propagate: transform_propagate_bind_group_layout(),
        transform_gather: transform_gather_bind_group_layout(),
        transform_readback: transform_readback_bind_group_layout(),
        light_resolve: light_resolve_bind_group_layout(),
        atmosphere: atmosphere_bind_group_layout(),
        selector: selector_bind_group_layout(),
        blas_sharing: blas_sharing_bind_group_layout(),
        ptlas: ptlas_bind_group_layout(),
        ptlas_hair_write: crate::hair::ptlas_hair::ptlas_hair_write_bind_group_layout(),
        ray_query_io: crate::ray_query::ray_query_io_bind_group_layout(),
        gizmo_depth: crate::render::gizmo_depth::gizmo_depth_bind_group_layout(),
        deform: crate::accel::deform::deform_bind_group_layout(),
        animated_blas: crate::accel::animated_blas::animated_blas_bind_group_layout(),
    });
}
