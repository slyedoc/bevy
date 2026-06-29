//! Stable material-slot allocation.
//!
//! Assigns each [`SolariMaterial`] asset a **stable** `u32` slot that persists
//! across frames (append-only, reused via a free-list on unload) — the asset-keyed
//! use of [`SlotPool`](crate::ecs_gpu::SlotPool). The `materials[]` GPU array is
//! indexed by this slot, and the per-instance `material_id` column stores it per
//! slot.
//!
//! Stability is the prerequisite for `material_id` being a slot-indexed delta
//! column: a reordering array (as the old iteration-order map was) would change
//! every instance's `material_id` every frame and could never be a clean delta.

use bevy_asset::AssetId;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};

use crate::bindings::SolariMaterialAssets;
use crate::ecs_gpu::SlotPool;
use crate::material::SolariMaterial;

/// Persistent `material asset → stable slot` map — an asset-keyed
/// [`SlotPool`]. Slot `0..len()` index the `materials[]` GPU array; freed slots
/// are reused before `len` grows.
#[derive(Resource, Default)]
pub struct MaterialSlots(SlotPool<AssetId<SolariMaterial>>);

impl MaterialSlots {
    /// Stable slot for `asset`, if it has one this frame.
    #[inline]
    pub fn slot_of(&self, asset: AssetId<SolariMaterial>) -> Option<u32> {
        self.0.slot_of(asset)
    }

    /// Monotonic version of the `asset → slot` map; advances whenever a material is
    /// assigned a slot or freed, so consumers caching resolved slots re-resolve.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.0.generation()
    }

    /// Number of material slots in use (== required `materials[]` length; freed
    /// slots leave holes filled with a default).
    #[inline]
    pub fn len(&self) -> u32 {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterate `(asset, slot)` for every currently-bound material.
    pub fn iter(&self) -> impl Iterator<Item = (AssetId<SolariMaterial>, u32)> + '_ {
        self.0.iter()
    }
}

pub fn init_material_slots(mut commands: Commands) {
    commands.insert_resource(MaterialSlots::default());
}

/// `Render::Prepare`: reconcile the stable slot map against this frame's extracted
/// material assets — assign slots to new materials, free the slots of materials
/// that are gone. Runs before any consumer (binder, instance-column scatter).
pub fn prepare_material_slots(
    mut slots: Option<ResMut<MaterialSlots>>,
    material_assets: Res<SolariMaterialAssets>,
) {
    let Some(slots) = slots.as_deref_mut() else {
        return;
    };
    slots.0.reconcile(
        material_assets.iter().map(|(id, _)| *id),
        |id| material_assets.get(&id).is_some(),
    );
}

/// [`MaterialTraversalFlags`] bit: the material alpha-tests (cutout foliage),
/// so its instances traverse `FORCE_NO_OPAQUE` to surface candidate hits.
pub const MATERIAL_TRAVERSAL_ALPHA_TESTED: u32 = 0x1;
/// [`MaterialTraversalFlags`] bit: the material is glass/transmissive, so its
/// instances route to the glass RT-pipeline hit group.
pub const MATERIAL_TRAVERSAL_GLASS: u32 = 0x2;
/// [`MaterialTraversalFlags`] bit: the material is a ray-portal surface, so its
/// instances route to the `chit_portal` hit group (teleport, no shading).
pub const MATERIAL_TRAVERSAL_PORTAL: u32 = 0x4;
/// [`MaterialTraversalFlags`] bit: the material is a planet surface, so its instances
/// route to the `chit_planet` hit group (biome albedo from `vertex_custom`).
pub const MATERIAL_TRAVERSAL_PLANET: u32 = 0x8;

/// The RT-pipeline SBT hit-group CLASS an instance's material selects, from its
/// [`MaterialTraversalFlags`] word: portal → 3 (`chit_portal`), glass → 1
/// (`chit_glass`), else 0 (`chit_opaque`). The SBT bakes `handle(2 + class)`
/// into that material's hit record (class 2 = hair is reached via a reserved
/// record, not a material), so portal/glass instances reach their dedicated
/// program instead of the opaque one. This is the single CPU-side surface-class
/// → shader routing key; add a class by extending this and the pipeline's hit
/// groups in lockstep.
pub fn material_sbt_class(traversal_flags: u32) -> u32 {
    if traversal_flags & MATERIAL_TRAVERSAL_PLANET != 0 {
        return 4;
    }
    if traversal_flags & MATERIAL_TRAVERSAL_PORTAL != 0 {
        return 3;
    }
    u32::from(traversal_flags & MATERIAL_TRAVERSAL_GLASS != 0)
}

/// Per-material-slot traversal flags consumed by the PTLAS fill (bit 0 =
/// the material needs candidate-hit inspection, i.e. it alpha-tests — see
/// [`SolariMaterial::traversal_alpha_cutoff`]). Slot-aligned with
/// `materials[]`. The fill derives each instance's `FORCE_NO_OPAQUE` flag
/// from this **on the GPU** (via the instance's `material_id` column) and
/// detects changes by comparing against the record it last wrote — so a
/// late-loading material, a runtime material swap, or a live asset edit all
/// self-heal without any CPU change tracking.
#[derive(Resource, Default)]
pub struct MaterialTraversalFlags {
    pub buffer: bevy_render::render_resource::StorageBuffer<Vec<u32>>,
}

/// `Render::Prepare` (after [`prepare_material_slots`]): rebuild the
/// slot-aligned flag list. O(materials) per frame — materials are few and the
/// payload is one `u32` each.
pub fn prepare_material_traversal_flags(
    mut flags: ResMut<MaterialTraversalFlags>,
    slots: Option<Res<MaterialSlots>>,
    material_assets: Res<SolariMaterialAssets>,
    render_device: Res<bevy_render::renderer::RenderDevice>,
    render_queue: Res<bevy_render::renderer::RenderQueue>,
) {
    let Some(slots) = slots.as_deref() else {
        return;
    };
    // ≥1 element so the PTLAS fill bind group always has a live buffer
    // (instances can exist before any non-default material does).
    let len = (slots.len() as usize).max(1);
    let list = flags.buffer.get_mut();
    list.clear();
    list.resize(len, 0);
    for (asset_id, slot) in slots.iter() {
        if let Some(material) = material_assets.get(&asset_id) {
            // bit 0: alpha-tested (FORCE_NO_OPAQUE during traversal).
            // bit 1: glass/transmissive — selects the glass RT-pipeline hit group.
            let alpha =
                u32::from(material.traversal_alpha_cutoff() >= 0.0) * MATERIAL_TRAVERSAL_ALPHA_TESTED;
            let glass = u32::from(material.specular_transmission > 0.0) * MATERIAL_TRAVERSAL_GLASS;
            let portal = u32::from(material.portal) * MATERIAL_TRAVERSAL_PORTAL;
            let planet = u32::from(material.planet) * MATERIAL_TRAVERSAL_PLANET;
            list[slot as usize] = alpha | glass | portal | planet;
        }
    }
    flags.buffer.write_buffer(&render_device, &render_queue);
}
