//! Ray portals — surfaces that teleport rays.
//!
//! Add [`SolariPortal`] to a ray-traced surface (a quad, an arch, anything)
//! and every ray that hits it continues from the paired portal instead: into
//! portal-local space, a half-turn about local Y, out through the target's
//! frame — so looking INTO this surface shows the view OUT of the target's
//! front (+Z). Pair two portals by pointing their `target`s at each other; a
//! one-way portal is just an unpaired one.
//!
//! The GPU table is a [`gpu_table!`](crate::gpu_table) column of instance-slot
//! PAIRS, slot-indexed by `GpuSlot<SolariPortals>`; the ray map derives at hit
//! time from the live GPU transform column, so portals on moving or
//! GPU-propagated parents stay exact with zero CPU transform reads. The portal
//! surface's material is flagged `portal`, so the rt_pipeline SBT routes its
//! hits to the dedicated `chit_portal` program — the scan + redirect math stays
//! off the opaque chit's register budget. Because the redirect happens in the
//! continued ray, recursion is free: portals seen through portals, portals in
//! reflections, portals through glass. Light is NOT transported — portals carry
//! the view, not next-event estimation, so each side is lit by its surroundings.

use bevy_ecs::{
    component::Component,
    entity::{Entity, EntityHashMap, EntityHashSet},
    prelude::ReflectComponent,
    system::{Local, Query, ResMut},
};
use bevy_reflect::Reflect;
use bevy_render::{sync_world::RenderEntity, Extract};
use bytemuck::{Pod, Zeroable};

use crate::ecs_gpu::{push_record, GpuSlot};
use crate::instance::RaytracingGpuEntity;

/// Marks a ray-traced surface as a portal showing the view out of `target`'s
/// front face (see the module docs for the exact mapping). The surface's
/// material should be flagged `portal` ([`StandardSolariMaterial::portal`](crate::material::StandardSolariMaterial))
/// so the SBT routes its hits to `chit_portal`; rays redirect on hit, never shade.
#[derive(Component, Reflect, Clone, Copy)]
#[reflect(Component)]
pub struct SolariPortal {
    /// The portal entity this surface looks out of.
    pub target: Entity,
}

/// One portal-table column entry: the instance-slot pairing. Mirror of
/// `Portal` in `raytracing_scene_bindings.wgsl`. `valid = 0` covers both
/// tombstoned (removed) entries and the zero-initialized tail of a grown
/// column buffer — instance slot 0 is real, so zeros must read as inert.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Pod, Zeroable)]
pub struct GpuPortal {
    /// Instance slot whose surface triggers this entry.
    instance_slot: u32,
    /// Instance slot of the paired exit portal.
    target_slot: u32,
    valid: u32,
    _pad: u32,
}

impl GpuPortal {
    const INVALID: Self = Self {
        instance_slot: 0,
        target_slot: 0,
        valid: 0,
        _pad: 0,
    };
}

crate::gpu_table! {
    /// The portal table: one slot per [`SolariPortal`] entity, carrying the
    /// (portal instance, target instance) pairing `portal_redirect` matches
    /// a hit against.
    pub table SolariPortals as SolariPortalsTablePlugin {
        members: bevy_ecs::query::With<SolariPortal>,
        extract: extract_solari_portals,
        columns {
            PortalPairColumn => pair: GpuPortal = "portals.pair" @ scene 5,
        }
    }
}

/// `ExtractSchedule` (the table's extract): scatter each portal's pairing
/// when it differs from what the GPU holds. The last-written memo makes this
/// change-driven against EVERYTHING the pairing depends on — component edits,
/// instance slots binding late (meshes stream in), retargeting — and
/// tombstones the freed slot when a portal goes away (a stale entry would be
/// a ghost portal on a reused instance slot).
pub fn extract_solari_portals(
    portals: Extract<Query<(Entity, RenderEntity, &SolariPortal, &GpuSlot<SolariPortals>)>>,
    target_render_entities: Extract<Query<RenderEntity>>,
    instance_slots: Query<&RaytracingGpuEntity>,
    mut table: ResMut<SolariPortals>,
    mut written: Local<EntityHashMap<(u32, GpuPortal)>>,
    mut seen: Local<EntityHashSet>,
) {
    seen.clear();
    for (entity, render_entity, portal, slot) in &portals {
        seen.insert(entity);
        let slot = slot.index();
        // Unresolved endpoints (assets still streaming) stay invalid until
        // both instance slots exist — the memo re-checks every frame.
        let pair = match (
            instance_slots.get(render_entity),
            target_render_entities
                .get(portal.target)
                .and_then(|target| instance_slots.get(target)),
        ) {
            (Ok(instance), Ok(target)) => GpuPortal {
                instance_slot: instance.0 .0,
                target_slot: target.0 .0,
                valid: 1,
                _pad: 0,
            },
            _ => GpuPortal::INVALID,
        };
        if written.get(&entity) != Some(&(slot, pair)) {
            push_record(&mut table.pair, slot, pair);
            written.insert(entity, (slot, pair));
        }
    }
    let pair_column = &mut table.pair;
    written.retain(|entity, (slot, _)| {
        let live = seen.contains(entity);
        if !live {
            push_record(pair_column, *slot, GpuPortal::INVALID);
        }
        live
    });
}
