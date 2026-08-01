enable wgpu_ray_query;

#define_import_path bevy_solari::scene_bindings

// The wgpu-side view of the raytracing scene bind group (set 0) — the subset
// the remaining WGSL compute passes (`restir_spatial` and the modules it
// imports) actually touch. The RT-pipeline stages see the full bind group
// through `scene_resolve.slang`; the Rust layout in `bindings/binder.rs` is
// the single source of truth for both.

// The scene TLAS (`restir_spatial` traces opaque visibility rays inline).
@group(0) @binding(4) var tlas: acceleration_structure;

// Split-sum DFG LUT (`brdf.wgsl`'s `F_AB`).
@group(0) @binding(6) var brdf_dfg_lut: texture_2d<f32>;
@group(0) @binding(7) var brdf_dfg_lut_sampler: sampler;

// Roughness at/below which a surface is treated as a perfect mirror (delta
// specular): sampling returns the exact reflection, pdf handling switches to
// the delta case.
const MIRROR_ROUGHNESS_THRESHOLD = 0.001f;

// Self-intersection guard for secondary rays; visibility traces return 0
// inside it (see `calculate_resolved_light_contribution`'s NaN note).
const RAY_T_MIN = 0.001f;
// Effectively infinite ray length (primary, shadow, GI). Finite, not `inf`: an `inf`
// tmax produces NaN in the slab test (`inf * 0`) and silently drops hits.
const RAY_T_MAX = 1.0e30f;

const RAY_NO_CULL = 0xFFu;

struct ResolvedMaterial {
    base_color: vec3<f32>,
    emissive: vec3<f32>,
    reflectance: f32,
    perceptual_roughness: f32,
    roughness: f32,
    metallic: f32,
    specular_transmission: f32,
    ior: f32,
    dispersion: f32,
    extinction: vec3<f32>,
    nested_priority: u32,
}

// Self-intersection-free ray origin for continuation rays (Wächter & Binder,
// "A Fast and Robust Method for Avoiding Self-Intersection", Ray Tracing
// Gems ch. 6). Offsets the hit point along the geometric normal by a few ULPs
// of its own float representation — exactly as much as precision requires, so
// it can neither re-hit the surface it left nor skip real geometry (no fixed
// world-space epsilon to outgrow a millimeter-scale wine glass or underflow a
// kilometer-scale city). Trace from the result with `t_min = 0`.
fn offset_ray_origin(p: vec3<f32>, geometric_normal: vec3<f32>) -> vec3<f32> {
    let int_offset = vec3<i32>(geometric_normal * 256.0);
    let p_int = vec3<f32>(
        bitcast<f32>(bitcast<i32>(p.x) + select(int_offset.x, -int_offset.x, p.x < 0.0)),
        bitcast<f32>(bitcast<i32>(p.y) + select(int_offset.y, -int_offset.y, p.y < 0.0)),
        bitcast<f32>(bitcast<i32>(p.z) + select(int_offset.z, -int_offset.z, p.z < 0.0)),
    );
    // Near zero a fixed float offset replaces the integer bump (the ULP size
    // collapses as the exponent does).
    let near_origin = abs(p) < vec3(1.0 / 32.0);
    return select(p_int, p + geometric_normal * (1.0 / 65536.0), near_origin);
}
