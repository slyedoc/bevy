//! Fog volumes — local participating-media regions.
//!
//! Add [`SolariFogVolume`] to an entity with a `Transform` (no mesh — it's a
//! medium, not geometry) and the aerial-perspective marches (compose pass and
//! pathtracer) sample it on top of the global height fog: a unit box (or
//! sphere) in entity-local space, scaled/rotated/placed by the transform,
//! holding a homogeneous scattering medium with a soft edge. Sun shafts
//! through the volume, self-shadowing from scene geometry, and sky-ambient
//! glow all come from the same shadowed march that drives the global fog —
//! a smoke column or a pool of ground mist is just a denser, bounded patch
//! of the same field.
//!
//! Lighting comes from the view's [`SolariAtmosphere`](crate::render::atmosphere::SolariAtmosphere)
//! sun plus the environment ambient; without an atmosphere the volume is
//! ambient-lit only (no sun term to scatter).
//!
//! The GPU table is a [`gpu_table!`](crate::gpu_table) column slot-indexed by
//! `GpuSlot<SolariFogVolumes>`, scattered change-driven (component edits and
//! transform motion) — the portal / black-hole pattern.

use bevy_ecs::{
    change_detection::DetectChanges,
    component::Component,
    entity::{Entity, EntityHashMap, EntityHashSet},
    prelude::ReflectComponent,
    system::{Local, Query, ResMut},
    world::Ref,
};
use bevy_math::{DVec3, ToRender, Vec3};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::Extract;
use bevy_transform::components::GlobalTransform;
use bytemuck::{Pod, Zeroable};

use crate::ecs_gpu::{push_record, GpuSlot};

/// A bounded participating medium: a unit box (half-extents 1) or unit sphere
/// (radius 1) in entity-local space — scale the `Transform` to size it. The
/// aerial marches accumulate sun in-scatter (shadowed by scene geometry) and
/// sky ambient through it, on top of the global height fog.
#[derive(Component, Reflect, Clone, Copy)]
#[reflect(Component, Default)]
pub struct SolariFogVolume {
    /// Extinction coefficient σ_t per world unit at the volume's core. The
    /// mean free path is `1 / density`; a volume much thicker than that reads
    /// as opaque.
    pub density: f32,
    /// Single-scatter albedo (σ_s / σ_t) per channel in `[0, 1]` — the
    /// medium's color. Below 1, the remainder is absorbed (smoke darkens,
    /// mist doesn't).
    pub albedo: Vec3,
    /// Henyey-Greenstein phase asymmetry `g` in `(-1, 1)`: higher ⇒ a
    /// tighter, brighter glow toward the sun.
    pub phase_g: f32,
    /// Fraction of the shape over which density fades to zero at the
    /// boundary: 0 = hard edge, 1 = fades from the center out.
    pub softness: f32,
    /// Use the unit sphere instead of the unit box.
    pub spherical: bool,
}

impl Default for SolariFogVolume {
    fn default() -> Self {
        Self {
            density: 0.3,
            albedo: Vec3::splat(0.85),
            phase_g: 0.5,
            softness: 0.35,
            spherical: false,
        }
    }
}

/// One fog-volume-table column entry. Mirror of `FogVolume` in
/// `raytracing_scene_bindings.wgsl`. All-zero (tombstone, or the
/// zero-initialized tail of a grown column buffer) is inert: zero extinction
/// is skipped before the transform is even read.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Pod, Zeroable)]
pub struct GpuFogVolume {
    /// World → entity-local affine, row-packed (WGSL `mat3x4` column k = the
    /// 4x4's row k — the `transforms` column convention).
    local_from_world: [[f32; 4]; 3],
    /// xyz = scattering σ_s (albedo × density), w = extinction σ_t (density).
    scattering: [f32; 4],
    /// x = phase g, y = edge softness, z = 1 for sphere / 0 for box.
    params: [f32; 4],
}

impl GpuFogVolume {
    const INVALID: Self = Self {
        local_from_world: [[0.0; 4]; 3],
        scattering: [0.0; 4],
        params: [0.0; 4],
    };

    /// `camera_translation` is the primary camera's absolute world translation — the
    /// floating origin. The fog uniform must be ORIGIN-RELATIVE (the aerial marches
    /// intersect origin-relative rays), and `GlobalTransform` is absolute, so the
    /// subtraction happens here, in f64, before the f32 narrowing.
    fn new(
        volume: &SolariFogVolume,
        transform: &GlobalTransform,
        camera_translation: DVec3,
    ) -> Self {
        let mut affine_full = transform.affine_full();
        affine_full.translation -= camera_translation;
        let affine = affine_full.to_render();
        let det = affine.matrix3.determinant();
        // A degenerate (zero-scale) transform has no inverse — inert entry.
        if !det.is_finite() || det.abs() < 1e-12 {
            return Self::INVALID;
        }
        let inv = affine.inverse();
        let (m, t) = (inv.matrix3, inv.translation);
        let density = volume.density.max(0.0);
        let sigma_s = volume.albedo.clamp(Vec3::ZERO, Vec3::ONE) * density;
        Self {
            local_from_world: [
                [m.x_axis.x, m.y_axis.x, m.z_axis.x, t.x],
                [m.x_axis.y, m.y_axis.y, m.z_axis.y, t.y],
                [m.x_axis.z, m.y_axis.z, m.z_axis.z, t.z],
            ],
            scattering: [sigma_s.x, sigma_s.y, sigma_s.z, density],
            params: [
                volume.phase_g.clamp(-0.99, 0.99),
                volume.softness.clamp(0.0, 1.0),
                if volume.spherical { 1.0 } else { 0.0 },
                0.0,
            ],
        }
    }
}

crate::gpu_table! {
    /// The fog-volume table: one slot per [`SolariFogVolume`] entity,
    /// carrying the medium parameters the aerial marches sample.
    pub table SolariFogVolumes as SolariFogVolumesTablePlugin {
        members: bevy_ecs::query::With<SolariFogVolume>,
        extract: extract_solari_fog_volumes,
        columns {
            FogVolumeColumn => volume: GpuFogVolume = "fog_volumes.volume" @ scene 7,
        }
    }
}

/// `ExtractSchedule` (the table's extract): scatter on component edits and
/// transform motion (`Ref` change detection covers both, including first
/// sight); tombstone freed slots so a removed volume doesn't keep fogging a
/// reused slot.
pub fn extract_solari_fog_volumes(
    volumes: Extract<
        Query<(
            Entity,
            Ref<SolariFogVolume>,
            Ref<GlobalTransform>,
            &GpuSlot<SolariFogVolumes>,
        )>,
    >,
    // The floating origin: fog uniforms are origin-relative, so a camera move
    // re-relativizes every volume (the readback marks the camera Changed).
    camera: Extract<
        Query<Ref<GlobalTransform>, bevy_ecs::query::With<crate::render::SolariCamera>>,
    >,
    mut table: ResMut<SolariFogVolumes>,
    mut written: Local<EntityHashMap<u32>>,
    mut seen: Local<EntityHashSet>,
) {
    seen.clear();
    let (camera_translation, camera_moved) = camera
        .iter()
        .next()
        .map(|c| (c.translation(), c.is_changed()))
        .unwrap_or((DVec3::ZERO, false));
    for (entity, volume, transform, slot) in &volumes {
        seen.insert(entity);
        let slot = slot.index();
        // `is_changed()` includes the frame the component was added.
        if volume.is_changed()
            || transform.is_changed()
            || camera_moved
            || !written.contains_key(&entity)
        {
            push_record(
                &mut table.volume,
                slot,
                GpuFogVolume::new(&volume, &transform, camera_translation),
            );
            written.insert(entity, slot);
        }
    }
    let volume_column = &mut table.volume;
    written.retain(|entity, slot| {
        let live = seen.contains(entity);
        if !live {
            push_record(volume_column, *slot, GpuFogVolume::INVALID);
        }
        live
    });
}
