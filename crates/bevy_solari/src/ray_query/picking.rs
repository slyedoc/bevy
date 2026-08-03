//! A [`bevy_picking`] backend over the [`SolariRayQuery`] service: pointer rays →
//! PTLAS → [`PointerHits`]. Adding [`SolariPickingPlugin`] makes solari a GPU
//! picking backend; not adding it costs nothing (the ray_query service stays a
//! no-op with `ray_count == 0`).
//!
//! Flow (mirrors `mesh_picking`'s `update_hits` CPU template, but async GPU):
//! 1. **Extract** ([`extract_picking_rays`]): clone `bevy_picking`'s [`RayMap`]
//!    into the render world, ordered deterministically by `(camera, pointer)` so a
//!    ray's index is stable across frames (pointer sets are tiny + stable).
//! 2. **Produce** ([`prepare_picking_rays`], `Render::Prepare` before the service's
//!    `prepare_ray_query`): write the ordered rays into [`SolariRayQuery::rays`] and
//!    set `ray_count`.
//! 3. **Dispatch + copy**: the service traces; [`copy_picking_hits`] copies
//!    `hits[0..ray_count]` into a main-world [`ShaderBuffer`] that bevy's [`Readback`]
//!    streams back, and [`receive_picking_hits`] lands the latest `Vec<Hit>` in
//!    [`SolariPickingHits`].
//! 4. **Backend** ([`emit_pointer_hits`], `PreUpdate` in [`PickingSystems::Backend`]):
//!    re-derive the same `(camera, pointer)` ordering from the current [`RayMap`],
//!    zip it index-for-index with the latest hits, and emit one [`PointerHits`] per
//!    pointer.
//!
//! **Async latency**: the readback is a couple frames behind the dispatch (GPU →
//! CPU streaming), so a hit's *position/entity* reflects the ray cast ~2 frames ago.
//! The pointer/camera *identity* stays correct because the ordering is stable, which
//! is what hover/click need; the small spatial lag is acceptable for picking.

use bevy_app::{App, Plugin, PreUpdate};
use bevy_asset::{Assets, Handle, RenderAssetUsages};
use bevy_camera::Camera;
use bevy_ecs::{
    entity::Entity,
    observer::On,
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::Vec3;
use bevy_picking::{
    backend::{
        ray::{RayId, RayMap},
        HitData, PointerHits,
    },
    pointer::PointerId,
    PickingSystems,
};
use bevy_ecs::message::MessageWriter;
use bevy_render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    gpu_readback::{Readback, ReadbackComplete},
    render_asset::RenderAssets,
    render_resource::BufferUsages,
    renderer::{RenderContext, RenderQueue},
    storage::{GpuShaderBuffer, ShaderBuffer},
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
};

use super::{prepare_ray_query, Hit, Ray, SolariRayQuery, RAY_T_MAX_DEFAULT, RAY_T_MIN_DEFAULT};

/// Deterministic, frame-stable sort key for a [`RayId`]: cameras by entity bits,
/// pointers by a `(variant, payload)` tuple (`PointerId` itself isn't `Ord` because
/// of its `Custom(Uuid)` arm). Both the producer (writes rays in this order) and the
/// backend (matches hits in this order) sort by this, so index i ↔ the same RayId.
fn ray_sort_key(id: &RayId) -> (u64, u8, u128) {
    let pointer = match id.pointer {
        PointerId::Mouse => (0u8, 0u128),
        PointerId::Touch(n) => (1, n as u128),
        PointerId::Custom(uuid) => (2, uuid.as_u128()),
    };
    (id.camera.to_bits(), pointer.0, pointer.1)
}

/// The current [`RayMap`] ordered by [`ray_sort_key`] — the canonical per-frame ray
/// order. Built once and reused by both the render-world producer and (re-derived)
/// the main-world backend.
fn ordered_rays(ray_map: &RayMap) -> Vec<(RayId, bevy_math::Ray3d)> {
    let mut rays: Vec<(RayId, bevy_math::Ray3d)> =
        ray_map.map.iter().map(|(id, ray)| (*id, *ray)).collect();
    rays.sort_unstable_by_key(|(id, _)| ray_sort_key(id));
    rays
}

/// Render-world resource holding the ordered picking rays, built each frame by
/// [`extract_picking_rays`] (the small `(RayId, Ray3d)` list).
#[derive(Resource, Default)]
pub struct ExtractedPickingRays {
    rays: Vec<(RayId, bevy_math::Ray3d)>,
}

/// Handle to the readback buffer (the render-graph copy fills it from `hits`; bevy's
/// [`Readback`] streams it back). `ExtractResource`d so the render-world copy system
/// reads the same handle.
#[derive(Resource, Clone, ExtractResource)]
struct PickingReadbackTarget {
    buffer: Handle<ShaderBuffer>,
    /// Capacity (in [`Hit`]s) the buffer was sized for; the copy clamps to it.
    capacity: u32,
}

/// Latest hits streamed back from the GPU, landed in the main world. Indexed the
/// same as the dispatch's ordered ray list.
#[derive(Resource, Default)]
pub struct SolariPickingHits {
    hits: Vec<Hit>,
}

/// Max picking rays per frame (pointer × camera). Pointer sets are tiny; this caps
/// the readback buffer + the dispatch.
const MAX_PICKING_RAYS: u32 = 256;

/// `ExtractSchedule`: clone the ordered [`RayMap`] into the render world. `RayMap`
/// is `Option` so a misconfiguration (no `PickingPlugin`) leaves an empty list
/// rather than panicking.
fn extract_picking_rays(ray_map: Extract<Option<Res<RayMap>>>, mut commands: Commands) {
    let rays = ray_map.as_ref().map_or_else(Vec::new, |map| {
        let mut rays = ordered_rays(map);
        rays.truncate(MAX_PICKING_RAYS as usize);
        rays
    });
    commands.insert_resource(ExtractedPickingRays { rays });
}

/// `Render::Prepare` (before [`prepare_ray_query`]): write the extracted rays into
/// the service's `rays` buffer and set `ray_count`.
fn prepare_picking_rays(
    extracted: Res<ExtractedPickingRays>,
    ray_query: Option<ResMut<SolariRayQuery>>,
    render_queue: Res<RenderQueue>,
) {
    let Some(mut ray_query) = ray_query else {
        return; // unsupported device — no batch ray-query service.
    };
    let count = extracted.rays.len().min(MAX_PICKING_RAYS as usize);
    if count == 0 {
        ray_query.ray_count = 0;
        return;
    }
    let rays: Vec<Ray> = extracted.rays[..count]
        .iter()
        .map(|(_, ray)| Ray {
            origin: ray.origin.to_array(),
            t_min: RAY_T_MIN_DEFAULT,
            direction: Vec3::from(ray.direction).to_array(),
            t_max: RAY_T_MAX_DEFAULT,
        })
        .collect();
    render_queue.write_buffer(&ray_query.rays.wgpu_buffer, 0, bytemuck::cast_slice(&rays));
    ray_query.ray_count = count as u32;
    // RAY_FLAG_NONE (0): closest hit against all geometry, like a primary ray.
    ray_query.ray_flags = 0;
}

/// `RenderGraph` (`RayQueries`, after the dispatch): copy `hits[0..ray_count]` into
/// the readback [`ShaderBuffer`] bevy's [`Readback`] streams back.
fn copy_picking_hits(
    ray_query: Option<Res<SolariRayQuery>>,
    target: Option<Res<PickingReadbackTarget>>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    mut ctx: RenderContext,
) {
    let Some(target) = target else {
        return;
    };
    let Some(ray_query) = ray_query else {
        return;
    };
    if ray_query.ray_count == 0 {
        return;
    }
    let Some(out) = gpu_buffers.get(&target.buffer) else {
        return;
    };
    let count = ray_query.ray_count.min(target.capacity);
    let bytes = count as u64 * size_of::<Hit>() as u64;
    ctx.command_encoder().copy_buffer_to_buffer(
        &ray_query.hits.wgpu_buffer,
        0,
        &out.buffer,
        0,
        bytes,
    );
}

/// Main world: a delivered readback buffer → [`SolariPickingHits`]. The buffer holds
/// whole [`Hit`] records; a short / empty delivery just yields fewer hits.
fn receive_picking_hits(event: On<ReadbackComplete>, mut hits: ResMut<SolariPickingHits>) {
    let decoded: &[Hit] = bytemuck::cast_slice(&event.data);
    hits.hits = decoded.to_vec();
}

/// `PreUpdate` ([`PickingSystems::Backend`]): zip the latest hits with the current
/// frame's ordered ray list (same `(camera, pointer)` ordering as the dispatch) and
/// emit one [`PointerHits`] per pointer. Skips misses (`t < 0`) and unresolved
/// entities. The hit position/entity lags the dispatch by a couple frames (async
/// readback); the pointer/camera identity is exact because the ordering is stable.
fn emit_pointer_hits(
    ray_map: Res<RayMap>,
    hits: Res<SolariPickingHits>,
    cameras: Query<&Camera>,
    mut pointer_hits: MessageWriter<PointerHits>,
) {
    if hits.hits.is_empty() {
        return;
    }
    let rays = ordered_rays(&ray_map);

    // Group picks by pointer (a pointer over multiple cameras yields several rays;
    // coalesce them into one `PointerHits`). `order` is the picking camera's order.
    let mut by_pointer: bevy_platform::collections::HashMap<PointerId, (Vec<(Entity, HitData)>, f32)> =
        bevy_platform::collections::HashMap::default();

    for (i, (ray_id, _)) in rays.iter().enumerate() {
        let Some(hit) = hits.hits.get(i) else {
            break; // fewer hits delivered than rays this frame — stop.
        };
        if hit.t < 0.0 {
            continue; // miss.
        }
        let bits = hit.entity_lo as u64 | ((hit.entity_hi as u64) << 32);
        let Some(entity) = Entity::try_from_bits(bits) else {
            continue; // null / torn entity bits.
        };
        let Ok(camera) = cameras.get(ray_id.camera) else {
            continue;
        };
        let hit_data = HitData::new(
            ray_id.camera,
            hit.t,
            Some(Vec3::from_array(hit.world_position)),
            None, // normal not resolved GPU-side (see ray_query.slang).
        );
        let entry = by_pointer
            .entry(ray_id.pointer)
            .or_insert_with(|| (Vec::new(), camera.order as f32));
        entry.0.push((entity, hit_data));
        // Use the highest camera order seen for this pointer (front-most layer).
        entry.1 = entry.1.max(camera.order as f32);
    }

    for (pointer, (picks, order)) in by_pointer {
        if !picks.is_empty() {
            pointer_hits.write(PointerHits::new(pointer, picks, order));
        }
    }
}

/// Main world: create the readback buffer + spawn the [`Readback`] with the
/// receive observer, and insert the hits resource. Runs once at startup.
fn setup_picking_readback(mut commands: Commands, mut buffers: ResMut<Assets<ShaderBuffer>>) {
    let mut buffer = ShaderBuffer::with_size(
        MAX_PICKING_RAYS as usize * size_of::<Hit>(),
        RenderAssetUsages::RENDER_WORLD,
    );
    // COPY_DST: the render-graph copy fills it. COPY_SRC: `Readback` streams it.
    buffer.buffer_description.usage =
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let handle = buffers.add(buffer);

    commands.insert_resource(PickingReadbackTarget {
        buffer: handle.clone(),
        capacity: MAX_PICKING_RAYS,
    });
    commands.init_resource::<SolariPickingHits>();
    commands
        .spawn(Readback::buffer(handle))
        .observe(receive_picking_hits);
}

/// Makes solari a [`bevy_picking`] backend over the ray_query service. Additive:
/// adding it enables GPU picking; omitting it leaves the service idle. Requires
/// `bevy_picking`'s core `PickingPlugin` (registers the [`PointerHits`] message) —
/// present in `DefaultPlugins`.
pub struct SolariPickingPlugin;

impl Plugin for SolariPickingPlugin {
    fn build(&self, app: &mut App) {
        // `PickingReadbackTarget` is created at startup in the main world, then
        // `ExtractResource`d so the render-world copy system reads the same handle.
        app.add_plugins(ExtractResourcePlugin::<PickingReadbackTarget>::default())
            .add_systems(bevy_app::Startup, setup_picking_readback)
            .add_systems(PreUpdate, emit_pointer_hits.in_set(PickingSystems::Backend));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ExtractedPickingRays>()
            .add_systems(ExtractSchedule, extract_picking_rays)
            .add_systems(
                Render,
                prepare_picking_rays
                    .in_set(RenderSystems::PrepareResources)
                    .before(prepare_ray_query),
            )
            .add_systems(
                bevy_render::renderer::RenderGraph,
                copy_picking_hits
                    .after(super::dispatch_ray_query)
                    .in_set(crate::SolariClusterSystems::RayQueries),
            );
    }
}
