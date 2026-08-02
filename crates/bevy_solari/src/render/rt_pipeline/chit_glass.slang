// Glass/transmissive closest-hit — a SEPARATE SBT program from `chit_opaque`,
// reached by instances whose material is glass-class: the SBT bakes this
// group's handle into those materials' hit records (see the SBT build in
// `gpu/rt_pipeline.rs` + `material::material_sbt_class`). Keeping it off the
// opaque program is the RT-pipeline win — the transmissive path's registers
// don't bloat the (far more common) opaque chit's occupancy.
//
// A smooth dielectric: one delta lobe (reflect or refract) chosen by exact
// Fresnel (`brdf::sample_glass_bsdf`), with Beer-Lambert absorption applied
// over each segment travelled INSIDE the medium. No NEE — a delta lobe can't
// be importance-sampled by area-light sampling — so this only fills the
// payload's continuation ray; the raygen drives the bounce loop.
// Nested-dielectric priority (overlapping volumes) is a follow-up: this
// handles enter/exit by face orientation, correct for non-overlapping glass.
//
// The DLSS ray-reconstruction guide writes are unconditional: the set-1
// layout always carries the G-buffer bindings and the WGSL twin compiled with
// `SOLARI_DLSS` always set.
//
// Regenerate SPIR-V (from crates/bevy_solari/src/render/rt_pipeline/):
//   slangc chit_glass.slang -target spirv -o chit_glass.spv

import rt_payload;
import scene_resolve;
import brdf;

// Camera UBO (set 1 binding 1): origin delta for the tess reprojection seed +
// the DLSS guide matrices.
[[vk::binding(1, 1)]] ConstantBuffer<RtCamera> camera;
// DLSS Ray Reconstruction guide G-buffer (set 1) — same bindings/packing as
// raygen and chit_opaque. Written for the primary glass hit so RR reprojects
// glass under camera motion instead of smearing the sky default.
[[vk::binding(5, 1)]] RWStructuredBuffer<float4> gbuffer_normal_roughness;
[[vk::binding(6, 1)]] RWStructuredBuffer<float4> gbuffer_diffuse;
[[vk::binding(7, 1)]] RWStructuredBuffer<float4> gbuffer_specular;
[[vk::binding(8, 1)]] RWStructuredBuffer<float4> gbuffer_motion;
static const uint NO_GBUFFER = 0xffffffff;

// Per-material SBT record — the material id baked into this record (same as
// `chit_opaque`): a plain binding on the record-sourced set (3), mapped
// host-side to the record's data bytes (`SHADER_RECORD_DATA`). Uniform per
// record, so material + texture fetches stay uniform per warp after SER.
struct SbtRecord
{
    uint material_id;
}
[[vk::binding(0, 3)]] ConstantBuffer<SbtRecord> sbt;

[shader("closesthit")]
void chit_glass(inout RtPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    var rng = payload.rng;
    // Tess hits synthesize their previous position from the current one; seed
    // the delta so it lands in the previous frame's origin like the mesh path.
    set_reproj_origin_delta(camera.origin_delta.xyz);
    let bary = attribs.barycentrics;
    let barycentrics = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    let ray_direction = WorldRayDirection();
    // Distance to this hit — the length of the segment just travelled (inside
    // the medium when this is the exit/back face), for Beer-Lambert absorption.
    let ray_t = RayTCurrent();
    // `ObjectToWorld3x4()` rows ARE the row-form affine the resolve expects.
    let o2w = ObjectToWorld3x4();
    Affine transform;
    transform.r0 = float4(o2w[0][0], o2w[0][1], o2w[0][2], o2w[0][3]);
    transform.r1 = float4(o2w[1][0], o2w[1][1], o2w[1][2], o2w[1][3]);
    transform.r2 = float4(o2w[2][0], o2w[2][1], o2w[2][2], o2w[2][3]);
    // Hit triangle's object-space vertex positions from the CLAS (position
    // fetch); skips the vertex-pool position loads.
    float3 hit_positions[3] = {
        HitTriangleVertexPosition(0),
        HitTriangleVertexPosition(1),
        HitTriangleVertexPosition(2),
    };
    let ray_hit = resolve_triangle_data_full_mat_fetch(
        InstanceIndex(), sbt.material_id, transform, cluster_id(), PrimitiveIndex(),
        barycentrics, hit_positions);

    let wo = -ray_direction;
    // Geometric normal: robust for the front/back-face test and the
    // self-intersection offset. `eta` = n_incident / n_transmitted — a front
    // face is air→glass (1/ior), a back face is glass→air (ray leaving).
    let geo_normal = ray_hit.geometric_world_normal;
    let entering = dot(ray_direction, geo_normal) < 0.0;
    let eta = entering ? 1.0 / ray_hit.material.ior : ray_hit.material.ior;
    // Geometric normal oriented into `wo`'s hemisphere (the safe interface normal).
    let safe_normal = entering ? geo_normal : -geo_normal;

    // The SMOOTH interpolated shading normal drives the refraction, so a
    // curved lens refracts smoothly instead of faceting per triangle. Align it
    // to the geometric side, orient it to `wo`, and fall back to the geometric
    // normal at silhouettes — there the interpolated normal tilts past the
    // view horizon, giving a negative cosine and a broken refraction.
    var normal = ray_hit.world_normal;
    if (dot(normal, geo_normal) < 0.0) { normal = -normal; } // align to geo side
    if (!entering) { normal = -normal; }                     // orient to wo's side
    if (dot(wo, normal) <= 1.0e-3) { normal = safe_normal; } // silhouette fallback

    // Primary-hit ray-reconstruction guide (chit-direct): give glass pixels a
    // real surface — depth, motion, and a smooth-specular normal — so RR
    // reprojects them under motion instead of accumulating the sky default
    // (the cause of glass smearing). Only the primary bounce carries a pixel;
    // secondary glass bounces pass NO_GBUFFER and skip this.
    if (payload.gbuffer_pixel != NO_GBUFFER)
    {
        let px = payload.gbuffer_pixel;
        let cur_clip = mat4_mul(camera.clip_from_world, float4(ray_hit.world_position, 1.0));
        // The camera passing THROUGH the glass puts the hit at the eye, so
        // clip.w → 0 and the perspective divides below would emit NaN/Inf into
        // the guide — which hangs DLSS. Skip the guide for such a pixel and
        // leave the raygen sky default (finite, safe).
        if (cur_clip.w > 1.0e-4)
        {
            // `normal` already faces the camera (oriented to `wo` above).
            gbuffer_normal_roughness[px] = float4(normal, ray_hit.material.roughness);
            let view_pos = mat4_mul(camera.view_from_world, float4(ray_hit.world_position, 1.0));
            // Glass has no diffuse albedo; depth rides in `.w` (clamped positive).
            gbuffer_diffuse[px] = float4(0.0, 0.0, 0.0, max(-view_pos.z, 1.0e-4));
            // Dielectric F0 from the IOR (≈0.04 at 1.5). `.w` (specular hit
            // distance) is filled by raygen after the continuation ray.
            let f0 = pow((ray_hit.material.ior - 1.0) / (ray_hit.material.ior + 1.0), 2.0);
            gbuffer_specular[px] = float4(f0, f0, f0, 0.0);
            // Screen-space motion of the glass surface (current vs previous
            // unjittered clip), UV space y-flipped. previous_frame_world_position
            // is in the PREVIOUS frame's origin; the rebased prev matrix expects
            // CURRENT-origin input — convert, or static MVs are off by the
            // camera delta. Guard last frame's divide too.
            let prev_clip = mat4_mul(camera.prev_clip_from_world,
                float4(ray_hit.previous_frame_world_position - camera.origin_delta.xyz, 1.0));
            var motion = float2(0.0);
            if (prev_clip.w > 1.0e-4)
            {
                let cur_uv = (cur_clip.xy / cur_clip.w) * float2(0.5, -0.5);
                let prev_uv = (prev_clip.xy / prev_clip.w) * float2(0.5, -0.5);
                motion = cur_uv - prev_uv;
            }
            gbuffer_motion[px] = float4(motion, 0.0, 0.0);
        }
    }

    let sample = sample_glass_bsdf(wo, normal, eta, rng);

    // `refract()` returns the ZERO vector at total internal reflection, and
    // `fresnel_dielectric` (which chose the lobe) computes the critical-angle
    // test as a separate float expression — so at grazing exit angles a 1-ULP
    // slip can make Fresnel pick "refract" while `refract` degenerated.
    // Tracing a zero-length ray hangs the RT core. Treat a degenerate
    // refraction as the TIR it really is (reflect); the flag flips so the
    // offset stays on the right side.
    var wi = sample.wi;
    var refracted = sample.refracted;
    if (refracted && dot(wi, wi) < 1.0e-8)
    {
        wi = reflect(ray_direction, normal);
        refracted = false;
    }
    // Last-resort guard (a degenerate normal could zero the reflection too):
    // terminate rather than emit a bad ray.
    if (!(dot(wi, wi) > 1.0e-8))
    {
        payload.bounce = 0;
        payload.rng = rng;
        return;
    }

    // Beer-Lambert absorption over the segment just travelled inside the
    // medium — only when leaving (the entry→exit distance is this hit's
    // `ray_t`). The interface itself is lossless (Fresnel selection cancels
    // the lobe weight); colour comes from this volume term.
    var absorption = float3(1.0);
    if (!entering)
    {
        absorption = exp(-ray_hit.material.extinction * ray_t);
    }

    payload.emitted = float3(0.0);
    payload.attenuation = sample.throughput * absorption;
    // Offset along the GEOMETRIC normal (robust self-intersection avoidance),
    // toward the side the next ray leaves: refraction crosses to the far side,
    // reflection stays on the incoming (`wo`) side.
    let push_normal = refracted ? -safe_normal : safe_normal;
    payload.next_origin = offset_ray_origin(ray_hit.world_position, push_normal);
    payload.next_direction = normalize(wi);
    // Delta lobe → no MIS against NEE at the next vertex (it can't be reached
    // by light sampling); the next hit takes full emissive.
    payload.p_bounce = 0.0;
    payload.bounce = 1;
    payload.rng = rng;
}
