// Portal closest-hit — the leanest SBT program. SBT-routed for portal-flagged
// materials so the portal scan + redirect math stay OFF the opaque chit's
// register budget (the occupancy win of per-class routing). On a portal hit it
// rewrites the ray to continue from the paired portal (`portal_redirect`) and
// returns; the raygen loop traces the continued ray, so portals work in
// reflections / through glass, bounded by `MAX_BOUNCES`. No shading, no NEE, no
// G-buffer — portals carry the view, lit by the far side's own surroundings.
enable wgpu_ray_tracing_pipeline;

#import bevy_solari::rt_payload::RtPayload
#import bevy_solari::scene_bindings::portal_redirect

var<incoming_ray_payload> payload: RtPayload;

@closest_hit
@incoming_payload(payload)
fn chit_portal(
    // PTLAS instance slot — the key `portal_redirect` matches against the table.
    @builtin(instance_id) instance_id: u32,
    @builtin(world_ray_origin) ray_origin: vec3<f32>,
    @builtin(world_ray_direction) ray_direction: vec3<f32>,
    @builtin(ray_t_current_max) ray_t: f32,
) {
    let hit_position = ray_origin + ray_direction * ray_t;
    var origin = ray_origin;
    var direction = ray_direction;
    if portal_redirect(instance_id, hit_position, &origin, &direction) {
        // Continue from the paired portal; lossless (the view, not light).
        payload.emitted = vec3<f32>(0.0);
        payload.attenuation = vec3<f32>(1.0);
        payload.next_origin = origin;
        payload.next_direction = direction;
        // Not a BSDF sample (a delta teleport) → the next vertex takes full emissive.
        payload.p_bounce = 0.0;
        payload.bounce = 1u;
        return;
    }
    // Defensive: a portal-routed hit whose pairing hasn't resolved yet (an
    // endpoint's instance slot still streaming) — terminate, don't emit a stale ray.
    payload.emitted = vec3<f32>(0.0);
    payload.bounce = 0u;
}
