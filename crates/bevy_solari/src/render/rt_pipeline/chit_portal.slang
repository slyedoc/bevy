// Portal closest-hit — the leanest SBT program. SBT-routed for portal-flagged
// materials so the portal scan + redirect math stay OFF the opaque chit's
// register budget (the occupancy win of per-class routing). On a portal hit it
// rewrites the ray to continue from the paired portal (`portal_redirect`) and
// returns; the raygen loop traces the continued ray, so portals work in
// reflections / through glass, bounded by `MAX_BOUNCES`. No shading, no NEE, no
// G-buffer — portals carry the view, lit by the far side's own surroundings.
//
// Regenerate SPIR-V (from crates/bevy_solari/src/render/rt_pipeline/):
//   slangc chit_portal.slang -target spirv -o chit_portal.spv

import rt_payload;

// Row-packed affine (mat3x4: r_k = the 4x4's row k), std430 stride 48 —
// mirrors `transforms` in raytracing_scene_bindings.wgsl.
struct Affine
{
    float4 r0;
    float4 r1;
    float4 r2;
}

// Mirrors `Portal` (stride 16).
struct Portal
{
    uint instance_slot;
    uint target_slot;
    uint valid;
    uint _pad;
}

// GPU scene columns (set 2 = SOLARI_SCENE_COLUMNS_GROUP).
[[vk::binding(0, 2)]] StructuredBuffer<Affine> transforms;
[[vk::binding(5, 2)]] StructuredBuffer<Portal> portals;

// Nudge past the exit portal's surface so the continued ray doesn't immediately
// re-hit it.
static const float PORTAL_SURFACE_OFFSET = 1e-4;

// k = dot(linear row k, p) + translation k.
float3 affine_transform_point(Affine m, float3 p)
{
    return float3(
        dot(m.r0.xyz, p) + m.r0.w,
        dot(m.r1.xyz, p) + m.r1.w,
        dot(m.r2.xyz, p) + m.r2.w);
}

// Apply the linear part only (for directions).
float3 affine_transform_direction(Affine m, float3 v)
{
    return float3(dot(m.r0.xyz, v), dot(m.r1.xyz, v), dot(m.r2.xyz, v));
}

// Inverse of a row-packed affine: 3x3 adjugate for the linear part,
// `-L⁻¹·t` for the translation.
Affine affine_inverse(Affine m)
{
    let c0 = float3(m.r0.x, m.r1.x, m.r2.x);
    let c1 = float3(m.r0.y, m.r1.y, m.r2.y);
    let c2 = float3(m.r0.z, m.r1.z, m.r2.z);
    let t = float3(m.r0.w, m.r1.w, m.r2.w);
    let inv_det = 1.0 / dot(c0, cross(c1, c2));
    let r0 = cross(c1, c2) * inv_det;
    let r1 = cross(c2, c0) * inv_det;
    let r2 = cross(c0, c1) * inv_det;
    Affine inv;
    inv.r0 = float4(r0, -dot(r0, t));
    inv.r1 = float4(r1, -dot(r1, t));
    inv.r2 = float4(r2, -dot(r2, t));
    return inv;
}

// If `hit_instance` is a portal surface, rewrite the ray to continue from the
// paired portal and return true; else leave the ray and return false.
bool portal_redirect(
    uint hit_instance,
    float3 hit_position,
    inout float3 ray_origin,
    inout float3 ray_direction)
{
    uint count;
    uint stride;
    portals.GetDimensions(count, stride);
    for (uint i = 0; i < count; i++)
    {
        if (portals[i].valid == 0 || portals[i].instance_slot != hit_instance)
        {
            continue;
        }
        let into_portal = affine_inverse(transforms[portals[i].instance_slot]);
        let out_of_target = transforms[portals[i].target_slot];
        var p = affine_transform_point(into_portal, hit_position);
        var d = affine_transform_direction(into_portal, ray_direction);
        p = float3(-p.x, p.y, -p.z);
        d = float3(-d.x, d.y, -d.z);
        let direction = normalize(affine_transform_direction(out_of_target, d));
        ray_origin = affine_transform_point(out_of_target, p)
            + direction * PORTAL_SURFACE_OFFSET;
        ray_direction = direction;
        return true;
    }
    return false;
}

[shader("closesthit")]
void chit_portal(inout RtPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    let hit_position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    var origin = WorldRayOrigin();
    var direction = WorldRayDirection();
    if (portal_redirect(InstanceIndex(), hit_position, origin, direction))
    {
        // Continue from the paired portal; lossless (the view, not light).
        payload.emitted = float3(0.0);
        payload.attenuation = float3(1.0);
        payload.next_origin = origin;
        payload.next_direction = direction;
        // Not a BSDF sample (a delta teleport) → the next vertex takes full emissive.
        payload.p_bounce = 0.0;
        payload.bounce = 1;
        return;
    }
    // Defensive: a portal-routed hit whose pairing hasn't resolved yet (an
    // endpoint's instance slot still streaming) — terminate, don't emit a stale ray.
    payload.emitted = float3(0.0);
    payload.bounce = 0;
}
