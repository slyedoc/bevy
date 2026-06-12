//! Black holes — regions that gravitationally bend rays.
//!
//! Add [`SolariBlackHole`] to an entity with a `Transform` (no mesh — it's a
//! field, not geometry) and rays crossing its influence sphere stop flying
//! straight: they march the Schwarzschild photon bend
//! (`d̂' ∝ −1.5·r_s·h²·r⃗/r⁵`, the GR factor that produces the photon ring),
//! get captured inside the horizon (those pixels are black, not sky), and
//! sample a procedural emissive accretion disk in the entity's local XZ
//! plane along the way. Einstein rings, doubled images, and the
//! light-wrapped disk all EMERGE from the bend — nothing is painted.
//!
//! The GPU table is a [`gpu_table!`](crate::gpu_table) column slot-indexed
//! by `GpuSlot<SolariBlackHoles>`, scattered change-driven (component edits
//! and transform motion). Geometry inside the influence sphere is not
//! intersected during the march (the region is assumed to be vacuum apart
//! from the built-in disk); keep meshes outside it. Lighting is not bent
//! either — like portals, the field carries the view, not next-event
//! estimation.

use bevy_ecs::{
    change_detection::DetectChanges,
    component::Component,
    entity::{Entity, EntityHashMap, EntityHashSet},
    prelude::ReflectComponent,
    system::{Local, Query, ResMut},
    world::Ref,
};
use bevy_math::Vec3;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::Extract;
use bevy_transform::components::GlobalTransform;
use bytemuck::{Pod, Zeroable};

use crate::ecs_gpu::{push_record, GpuSlot};

/// A gravitational lens with an event horizon and an optional accretion
/// disk. The disk lies in the entity's local XZ plane (tilt it with the
/// `Transform`); rays bend within [`Self::influence_radius`] of the entity.
#[derive(Component, Reflect, Clone, Copy)]
#[reflect(Component, Default)]
pub struct SolariBlackHole {
    /// Schwarzschild radius (world units): rays inside are captured (black).
    /// The photon ring forms at 1.5× this.
    pub schwarzschild_radius: f32,
    /// Radius of the bend region. Larger = lensing visible from farther
    /// away, more march steps spent. ~10–20× the Schwarzschild radius reads
    /// well.
    pub influence_radius: f32,
    /// Accretion-disk annulus (world units); set `disk_emission` to 0 for a
    /// bare lens.
    pub disk_inner_radius: f32,
    pub disk_outer_radius: f32,
    /// Emitted radiance scale of the disk's hot inner edge.
    pub disk_emission: f32,
}

impl Default for SolariBlackHole {
    fn default() -> Self {
        Self {
            schwarzschild_radius: 0.1,
            influence_radius: 1.5,
            disk_inner_radius: 0.18,
            disk_outer_radius: 0.7,
            disk_emission: 50.0,
        }
    }
}

/// One black-hole-table column entry. Mirror of `BlackHole` in
/// `raytracing_scene_bindings.wgsl`. All-zero (tombstone, or the
/// zero-initialized tail of a grown column buffer) is inert: a zero
/// influence radius never matches a ray.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Pod, Zeroable)]
pub struct GpuBlackHole {
    /// xyz = world center, w = influence radius.
    center_influence: [f32; 4],
    /// xyz = disk plane normal (entity local +Y), w = Schwarzschild radius.
    normal_rs: [f32; 4],
    /// x = disk inner radius, y = outer radius, z = emission scale.
    disk: [f32; 4],
}

impl GpuBlackHole {
    const INVALID: Self = Self {
        center_influence: [0.0; 4],
        normal_rs: [0.0; 4],
        disk: [0.0; 4],
    };

    fn new(hole: &SolariBlackHole, transform: &GlobalTransform) -> Self {
        let center = transform.translation();
        let normal = transform.affine().transform_vector3(Vec3::Y).normalize();
        Self {
            center_influence: [center.x, center.y, center.z, hole.influence_radius.max(0.0)],
            normal_rs: [
                normal.x,
                normal.y,
                normal.z,
                hole.schwarzschild_radius.max(0.0),
            ],
            disk: [
                hole.disk_inner_radius,
                hole.disk_outer_radius,
                hole.disk_emission,
                0.0,
            ],
        }
    }
}

crate::gpu_table! {
    /// The black-hole table: one slot per [`SolariBlackHole`] entity,
    /// carrying the field parameters the traversal march reads.
    pub table SolariBlackHoles as SolariBlackHolesTablePlugin {
        members: bevy_ecs::query::With<SolariBlackHole>,
        extract: extract_solari_black_holes,
        columns {
            BlackHoleColumn => hole: GpuBlackHole = "black_holes.hole" @ scene 6,
        }
    }
}

/// `ExtractSchedule` (the table's extract): scatter on component edits and
/// transform motion (`Ref` change detection covers both, including first
/// sight); tombstone freed slots so a removed hole doesn't keep bending rays
/// out of a reused slot.
pub fn extract_solari_black_holes(
    black_holes: Extract<
        Query<(
            Entity,
            Ref<SolariBlackHole>,
            Ref<GlobalTransform>,
            &GpuSlot<SolariBlackHoles>,
        )>,
    >,
    mut table: ResMut<SolariBlackHoles>,
    mut written: Local<EntityHashMap<u32>>,
    mut seen: Local<EntityHashSet>,
) {
    seen.clear();
    for (entity, hole, transform, slot) in &black_holes {
        seen.insert(entity);
        let slot = slot.index();
        // `is_changed()` includes the frame the component was added.
        if hole.is_changed() || transform.is_changed() || !written.contains_key(&entity) {
            push_record(&mut table.hole, slot, GpuBlackHole::new(&hole, &transform));
            written.insert(entity, slot);
        }
    }
    let hole_column = &mut table.hole;
    written.retain(|entity, slot| {
        let live = seen.contains(entity);
        if !live {
            push_record(hole_column, *slot, GpuBlackHole::INVALID);
        }
        live
    });
}
