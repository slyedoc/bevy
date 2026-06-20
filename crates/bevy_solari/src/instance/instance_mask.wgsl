// Named bit values for the PTLAS fill: the `InstanceMaskColumn` layout (the
// 8-bit hardware cull mask from `RenderLayers`), the per-material traversal
// flags, and the Vulkan instance-flag bits. Import this instead of hardcoding
// bit values.

#define_import_path bevy_solari::instance_mask

// Low byte: the 8-bit hardware TLAS cull mask (`RenderLayers` 0-7).
const INSTANCE_MASK_HARDWARE_BITS: u32 = 0xFFu;

// Per-material traversal flags (`MaterialTraversalFlags`, slot-aligned with
// `materials[]`): bit 0 = the material alpha-tests, so its instances need
// candidate-hit inspection.
const MATERIAL_TRAVERSAL_ALPHA_TESTED: u32 = 0x1u;
// bit 1 = the material is glass/transmissive — routes its instances to the glass
// RT-pipeline hit group (ignored by the inline-rayQuery path).
const MATERIAL_TRAVERSAL_GLASS: u32 = 0x2u;

// RT-pipeline SBT hit-group indices (instance_contribution_to_hit_group_index).
const HIT_GROUP_OPAQUE: u32 = 0u;
const HIT_GROUP_GLASS: u32 = 1u;
const HIT_GROUP_HAIR: u32 = 2u;

// `VkGeometryInstanceFlagBitsKHR` — the PTLAS record's `instance_flags`.
const VK_INSTANCE_TRIANGLE_FACING_CULL_DISABLE: u32 = 0x1u;
const VK_INSTANCE_TRIANGLE_FLIP_FACING: u32 = 0x2u;
const VK_INSTANCE_FORCE_OPAQUE: u32 = 0x4u;
const VK_INSTANCE_FORCE_NO_OPAQUE: u32 = 0x8u;

/// The hardware cull mask carried in a column entry.
fn instance_hardware_mask(mask_bits: u32) -> u32 {
    return mask_bits & INSTANCE_MASK_HARDWARE_BITS;
}

/// The Vulkan instance flags an instance's MATERIAL implies: alpha-tested
/// materials force their instances non-opaque so traversal surfaces
/// candidate hits to `trace_ray`'s alpha test.
fn material_vk_flags(traversal_flags: u32) -> u32 {
    return select(
        0u,
        VK_INSTANCE_FORCE_NO_OPAQUE,
        (traversal_flags & MATERIAL_TRAVERSAL_ALPHA_TESTED) != 0u,
    );
}

/// The RT-pipeline SBT hit-group index a cluster instance's MATERIAL selects:
/// glass → glass hit group, else opaque. (Hair instances are routed to
/// `HIT_GROUP_HAIR` in `ptlas_hair_write.wgsl`, not here.)
fn material_hit_group(traversal_flags: u32) -> u32 {
    return select(
        HIT_GROUP_OPAQUE,
        HIT_GROUP_GLASS,
        (traversal_flags & MATERIAL_TRAVERSAL_GLASS) != 0u,
    );
}
