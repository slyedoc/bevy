// Glass/transmissive closest-hit — a SEPARATE SBT program from `chit_opaque`
// (hit group 3), reached by instances whose material is glass-class: the SBT
// bakes this group's handle into those materials' hit records (see the SBT build
// in `gpu/rt_pipeline.rs` + `material::material_sbt_class`). Keeping it off the
// opaque program is the RT-pipeline win — the transmissive path's registers
// don't bloat the (far more common) opaque chit's occupancy.
//
// A smooth dielectric: one delta lobe (reflect or refract) chosen by exact
// Fresnel (`brdf::sample_glass_bsdf`), with Beer-Lambert absorption applied over
// each segment travelled INSIDE the medium. No NEE — a delta lobe can't be
// importance-sampled by area-light sampling — so this only fills the payload's
// continuation ray; the raygen drives the bounce loop. Nested-dielectric
// priority (overlapping volumes) is a follow-up: this handles enter/exit by
// face orientation, correct for non-overlapping glass.
enable wgpu_ray_tracing_pipeline;
enable primitive_index;

#import bevy_solari::rt_payload::{RtPayload, RtCamera}
#import bevy_solari::brdf::sample_glass_bsdf
#import bevy_solari::scene_bindings::{resolve_triangle_data_full_mat_fetch, offset_ray_origin}

var<incoming_ray_payload> payload: RtPayload;
// Driver-provided triangle barycentrics (the fixed-function intersection's u, v).
var<hit_attribute> bary: vec2<f32>;

#ifdef SOLARI_DLSS
// DLSS Ray Reconstruction guide G-buffer (set 1) — same bindings/packing as raygen
// and chit_opaque. Written for the primary glass hit so RR reprojects glass under
// camera motion instead of smearing the sky default. Reachable via the existing
// CLOSEST_HIT_KHR binding visibility, so no descriptor-layout change.
@group(1) @binding(1) var<uniform> camera: RtCamera;
@group(1) @binding(5) var<storage, read_write> gbuffer_normal_roughness: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> gbuffer_diffuse: array<vec4<f32>>;
@group(1) @binding(7) var<storage, read_write> gbuffer_specular: array<vec4<f32>>;
@group(1) @binding(8) var<storage, read_write> gbuffer_motion: array<vec4<f32>>;
const NO_GBUFFER: u32 = 0xffffffffu;
#endif

// Per-material SBT shader record — the material id baked into this record (same
// as `chit_opaque`). Uniform per record, so material + texture fetches stay
// uniform per warp after SER.
struct SbtRecord {
    material_id: u32,
}
var<shader_record> sbt: SbtRecord;

@closest_hit
@incoming_payload(payload)
fn chit_glass(
    @builtin(instance_id) instance_id: u32,
    @builtin(cluster_id) cluster_id: u32,
    @builtin(primitive_index) primitive_index: u32,
    @builtin(world_ray_direction) ray_direction: vec3<f32>,
    // Distance to this hit — the length of the segment just travelled (inside the
    // medium when this is the exit/back face), for Beer-Lambert absorption.
    @builtin(ray_t_current_max) ray_t: f32,
    @builtin(object_to_world) object_to_world: mat4x3<f32>,
    // Hit triangle's object-space vertex positions from the CLAS (position fetch);
    // skips the vertex-pool position loads. See `chit_opaque` / `clas_arena`.
    @builtin(hit_triangle_vertex_positions) hit_positions: array<vec3<f32>, 3>,
) {
    var rng = payload.rng;
    let barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    // Row-form affine (m[r] = (basis_row_r, translation_r)) the resolve expects.
    let transform = mat3x4<f32>(
        vec4<f32>(object_to_world[0].x, object_to_world[1].x, object_to_world[2].x, object_to_world[3].x),
        vec4<f32>(object_to_world[0].y, object_to_world[1].y, object_to_world[2].y, object_to_world[3].y),
        vec4<f32>(object_to_world[0].z, object_to_world[1].z, object_to_world[2].z, object_to_world[3].z),
    );
    let ray_hit = resolve_triangle_data_full_mat_fetch(instance_id, sbt.material_id, transform, cluster_id, primitive_index, barycentrics, hit_positions);

    let wo = -ray_direction;
    // Geometric normal: robust for the front/back-face test and the
    // self-intersection offset. `eta` = n_incident / n_transmitted — a front face
    // is air→glass (1/ior), a back face is glass→air (ior, ray leaving the medium).
    let geo_normal = ray_hit.geometric_world_normal;
    let entering = dot(ray_direction, geo_normal) < 0.0;
    let eta = select(ray_hit.material.ior, 1.0 / ray_hit.material.ior, entering);
    // Geometric normal oriented into `wo`'s hemisphere (the safe interface normal).
    let safe_normal = select(-geo_normal, geo_normal, entering);

    // The SMOOTH interpolated shading normal drives the refraction, so a curved
    // lens refracts smoothly instead of faceting per triangle (the geometric
    // normal is flat per triangle → a blocky lens). Align it to the geometric
    // side, orient it to `wo`, and fall back to the geometric normal at
    // silhouettes — there the interpolated normal tilts past the view horizon,
    // giving a negative cosine and a broken refraction.
    var normal = ray_hit.world_normal;
    if dot(normal, geo_normal) < 0.0 { normal = -normal; } // align to geo side
    if !entering { normal = -normal; }                     // orient to wo's side
    if dot(wo, normal) <= 1.0e-3 { normal = safe_normal; } // silhouette fallback

#ifdef SOLARI_DLSS
    // Primary-hit ray-reconstruction guide (chit-direct): give glass pixels a real
    // surface — depth, motion, and a smooth-specular normal — so RR reprojects them
    // under motion instead of accumulating the sky default (the cause of glass
    // smearing). Only the primary bounce carries a pixel; secondary glass bounces
    // pass NO_GBUFFER and skip this. The visible colour is the refraction/reflection,
    // but RR tracks the glass SURFACE here; the residual (refracted content moving
    // differently from the surface) is the separate refraction-motion problem.
    if payload.gbuffer_pixel != NO_GBUFFER {
        let px = payload.gbuffer_pixel;
        let cur_clip = camera.clip_from_world * vec4<f32>(ray_hit.world_position, 1.0);
        // The camera passing THROUGH the glass puts the hit at the eye, so clip.w → 0
        // and the perspective divides below would emit NaN/Inf into the guide — which
        // hangs DLSS. Skip the guide for such a pixel and leave the raygen sky default
        // (finite, safe); it's a transient frame while passing through anyway.
        if cur_clip.w > 1.0e-4 {
            // `normal` already faces the camera (oriented to `wo` above); glass is smooth.
            gbuffer_normal_roughness[px] = vec4<f32>(normal, ray_hit.material.roughness);
            let view_pos = camera.view_from_world * vec4<f32>(ray_hit.world_position, 1.0);
            // Glass has no diffuse albedo; depth rides in `.w` (clamped positive).
            gbuffer_diffuse[px] = vec4<f32>(0.0, 0.0, 0.0, max(-view_pos.z, 1.0e-4));
            // Dielectric F0 from the IOR (≈0.04 at 1.5). `.w` (specular hit distance) is
            // filled by raygen after the continuation ray.
            let f0 = pow((ray_hit.material.ior - 1.0) / (ray_hit.material.ior + 1.0), 2.0);
            gbuffer_specular[px] = vec4<f32>(f0, f0, f0, 0.0);
            // Screen-space motion of the glass surface (current vs previous unjittered
            // clip), UV space y-flipped. Guard last frame's divide too; zero motion if
            // the surface was at the eye then.
            let prev_clip =
                camera.prev_clip_from_world * vec4<f32>(ray_hit.previous_frame_world_position, 1.0);
            var motion = vec2<f32>(0.0);
            if prev_clip.w > 1.0e-4 {
                let cur_uv = (cur_clip.xy / cur_clip.w) * vec2<f32>(0.5, -0.5);
                let prev_uv = (prev_clip.xy / prev_clip.w) * vec2<f32>(0.5, -0.5);
                motion = cur_uv - prev_uv;
            }
            gbuffer_motion[px] = vec4<f32>(motion, 0.0, 0.0);
        }
    }
#endif

    let sample = sample_glass_bsdf(wo, normal, eta, &rng);

    // `refract()` returns the ZERO vector at total internal reflection, and
    // `fresnel_dielectric` (which chose the lobe) computes the critical-angle test
    // as a separate float expression — so at grazing exit angles a 1-ULP slip can
    // make Fresnel pick "refract" while `refract` degenerated. Tracing a
    // zero-length ray hangs the RT core. Treat a degenerate refraction as the TIR
    // it really is (reflect); the flag flips so the offset stays on the right side.
    var wi = sample.wi;
    var refracted = sample.refracted;
    if refracted && dot(wi, wi) < 1.0e-8 {
        wi = reflect(ray_direction, normal);
        refracted = false;
    }
    // Last-resort guard (a degenerate normal could zero the reflection too):
    // terminate rather than emit a bad ray.
    if !(dot(wi, wi) > 1.0e-8) {
        payload.bounce = 0u;
        payload.rng = rng;
        return;
    }

    // Beer-Lambert absorption over the segment just travelled inside the medium —
    // only when leaving (the entry→exit distance is this hit's `ray_t`). The
    // interface itself is lossless (Fresnel selection cancels the lobe weight);
    // colour comes from this volume term.
    var absorption = vec3<f32>(1.0);
    if !entering {
        absorption = exp(-ray_hit.material.extinction * ray_t);
    }

    payload.emitted = vec3<f32>(0.0);
    payload.attenuation = sample.throughput * absorption;
    // Offset along the GEOMETRIC normal (robust self-intersection avoidance),
    // toward the side the next ray leaves: refraction crosses to the far side,
    // reflection stays on the incoming (`wo`) side.
    let push_normal = select(safe_normal, -safe_normal, refracted);
    payload.next_origin = offset_ray_origin(ray_hit.world_position, push_normal);
    payload.next_direction = normalize(wi);
    // Delta lobe → no MIS against NEE at the next vertex (it can't be reached by
    // light sampling); the next hit takes full emissive.
    payload.p_bounce = 0.0;
    payload.bounce = 1u;
    payload.rng = rng;
}
