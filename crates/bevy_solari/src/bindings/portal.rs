//! Ray portals — surfaces that teleport rays.
//!
//! Add [`SolariPortal`] to a [`RaytracingMesh3d`](super::RaytracingMesh3d)
//! surface (a quad, an arch, anything) and every ray that hits it continues
//! from the paired portal instead: origin and direction mapped by
//! `target_world × R_y(π) × portal_world⁻¹`, so looking INTO this surface
//! shows the view OUT of the target's front (+Z). Pair two portals by
//! pointing their `target`s at each other; a one-way portal is just an
//! unpaired one.
//!
//! The GPU table carries only the instance-slot PAIRING — the ray map
//! derives at hit time from the live GPU transform column, so portals on
//! moving or GPU-propagated parents stay exact with zero CPU transform
//! reads. Because the redirect happens during ray traversal
//! (`portal_redirect` in `raytracing_scene_bindings.wgsl`), recursion is
//! free: portals seen through portals, portals in reflections, portals
//! through glass. Light is NOT transported — portals carry the view, not
//! next-event estimation, so each side is lit by its own surroundings.

use bevy_ecs::{
    component::Component,
    entity::Entity,
    prelude::ReflectComponent,
    resource::Resource,
    system::{Query, Res, ResMut},
};
use bevy_reflect::Reflect;
use bevy_render::{
    render_resource::{ShaderType, StorageBuffer},
    renderer::{RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    Extract,
};

use crate::instance::RaytracingGpuEntity;

/// Marks a ray-traced surface as a portal showing the view out of `target`'s
/// front face (see the module docs for the exact mapping). The surface's
/// material is never shaded; rays redirect on hit.
#[derive(Component, Reflect, Clone, Copy)]
#[reflect(Component)]
pub struct SolariPortal {
    /// The portal entity this surface looks out of.
    pub target: Entity,
}

/// One GPU portal-table entry: the instance-slot pairing. Mirror of `Portal`
/// in `raytracing_scene_bindings.wgsl`.
#[derive(ShaderType, Clone, Copy)]
pub struct GpuPortal {
    /// Instance slot whose surface triggers this entry (`u32::MAX` = dummy).
    slot: u32,
    /// Instance slot of the paired exit portal.
    target_slot: u32,
}

/// The frame's portal table (scene binding 18). Rebuilt every frame — portal
/// counts are tiny, and the entries are pure slot pairs (the transforms live
/// in the GPU column).
#[derive(Resource, Default)]
pub struct PortalTable {
    pub buffer: StorageBuffer<Vec<GpuPortal>>,
}

/// `ExtractSchedule`: rebuild the portal table from main-world
/// [`SolariPortal`]s. A portal whose endpoints haven't bound their instance
/// slots yet (assets still streaming) is skipped until they have.
pub fn extract_solari_portals(
    portals: Extract<Query<(RenderEntity, &SolariPortal)>>,
    target_render_entities: Extract<Query<RenderEntity>>,
    slots: Query<&RaytracingGpuEntity>,
    mut table: ResMut<PortalTable>,
) {
    let list = table.buffer.get_mut();
    list.clear();
    for (render_entity, portal) in &portals {
        let Ok(target_render_entity) = target_render_entities.get(portal.target) else {
            continue;
        };
        let (Ok(slot), Ok(target_slot)) = (
            slots.get(render_entity),
            slots.get(target_render_entity),
        ) else {
            continue;
        };
        list.push(GpuPortal {
            slot: slot.0.0,
            target_slot: target_slot.0.0,
        });
    }
    // ≥1 element keeps the binding valid; the sentinel slot never matches.
    if list.is_empty() {
        list.push(GpuPortal {
            slot: u32::MAX,
            target_slot: u32::MAX,
        });
    }
}

/// `Render::Prepare`: upload the table (before the scene bind group builds).
pub fn prepare_solari_portals(
    mut table: ResMut<PortalTable>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    table.buffer.write_buffer(&render_device, &render_queue);
}
