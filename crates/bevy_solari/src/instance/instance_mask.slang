// Named bit values for the PTLAS fill: the `InstanceMaskColumn` layout (the
// 8-bit hardware cull mask from `RenderLayers`), the per-material traversal
// flags, and the Vulkan instance-flag bits. Import this instead of hardcoding
// bit values.
module instance_mask;

// Low byte: the 8-bit hardware TLAS cull mask (`RenderLayers` 0-7).
public static const uint INSTANCE_MASK_HARDWARE_BITS = 0xFF;

// Per-material traversal flags (`MaterialTraversalFlags`, slot-aligned with
// `materials[]`): bit 0 = the material alpha-tests, so its instances need
// candidate-hit inspection.
public static const uint MATERIAL_TRAVERSAL_ALPHA_TESTED = 0x1;
// bit 1 = the material is glass/transmissive — routes its instances to the glass
// RT-pipeline hit group (ignored by the inline-rayQuery path).
public static const uint MATERIAL_TRAVERSAL_GLASS = 0x2;

// RT-pipeline SBT hit-group indices (instance_contribution_to_hit_group_index).
public static const uint HIT_GROUP_OPAQUE = 0;
public static const uint HIT_GROUP_GLASS = 1;
public static const uint HIT_GROUP_HAIR = 2;

// `VkGeometryInstanceFlagBitsKHR` — the PTLAS record's `instance_flags`.
public static const uint VK_INSTANCE_TRIANGLE_FACING_CULL_DISABLE = 0x1;
public static const uint VK_INSTANCE_TRIANGLE_FLIP_FACING = 0x2;
public static const uint VK_INSTANCE_FORCE_OPAQUE = 0x4;
public static const uint VK_INSTANCE_FORCE_NO_OPAQUE = 0x8;

/// The hardware cull mask carried in a column entry.
public uint instance_hardware_mask(uint mask_bits) {
    return mask_bits & INSTANCE_MASK_HARDWARE_BITS;
}

/// The Vulkan instance flags an instance's MATERIAL implies: alpha-tested
/// materials force their instances non-opaque so traversal surfaces
/// candidate hits to `trace_ray`'s alpha test.
public uint material_vk_flags(uint traversal_flags) {
    return (traversal_flags & MATERIAL_TRAVERSAL_ALPHA_TESTED) != 0
        ? VK_INSTANCE_FORCE_NO_OPAQUE
        : 0;
}

/// The RT-pipeline SBT hit-group index a cluster instance's MATERIAL selects:
/// glass → glass hit group, else opaque. (Hair instances are routed to
/// `HIT_GROUP_HAIR` in `ptlas_hair_write.slang`, not here.)
public uint material_hit_group(uint traversal_flags) {
    return (traversal_flags & MATERIAL_TRAVERSAL_GLASS) != 0
        ? HIT_GROUP_GLASS
        : HIT_GROUP_OPAQUE;
}
