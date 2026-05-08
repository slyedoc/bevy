// Optional gbuffer-debug overwrite pass for `SolariLighting`.
//
// When `SolariLighting::debug_view` is `Some`, this compute pass runs as the
// last step of `solari_lighting` and overwrites `view_output` with a
// visualisation of the chosen channel (world normal, depth, base colour,
// motion vector, etc.). Mirrors aurora's compose-debug paths so users can
// compare the same channels across renderers.
//
// The bind group layout is shared with the rest of the solari lighting
// pipelines -- this shader only references a small subset (gbuffer, depth,
// motion, view, view_output). The remaining slots in the layout are
// inert here.

#import bevy_render::view::depth_ndc_to_view_z
#import bevy_solari::gbuffer_utils::gpixel_resolve
#import bevy_solari::realtime_bindings::{
    view_output,
    gbuffer,
    depth_buffer,
    motion_vectors,
    view,
}

// Discriminants must match `SolariDebugView` in mod.rs.
const DEBUG_WORLD_NORMAL:   u32 = 1u;
const DEBUG_WORLD_POSITION: u32 = 2u;
const DEBUG_BASE_COLOR:     u32 = 3u;
const DEBUG_MATERIAL:       u32 = 4u;
const DEBUG_DEPTH:          u32 = 5u;
const DEBUG_ROUGHNESS:      u32 = 6u;
const DEBUG_METALLIC:       u32 = 7u;
const DEBUG_EMISSIVE:       u32 = 8u;
const DEBUG_MOTION_VECTOR:  u32 = 9u;

const SKY_COLOR: vec3<f32> = vec3<f32>(0.04, 0.06, 0.10);

// The debug pass runs with its own immediate layout (4 bytes) -- the rest
// of `solari_lighting` uses the 8-byte `PushConstants` from
// `realtime_bindings`. Shared bind group, distinct push-constant slot.
struct DebugConstants { view_mode: u32 }
var<immediate> debug_constants: DebugConstants;

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn id_color(id: u32) -> vec3<f32> {
    let h = pcg_hash(id);
    return vec3<f32>(
        f32((h >> 0u)  & 0xffu) / 255.0,
        f32((h >> 8u)  & 0xffu) / 255.0,
        f32((h >> 16u) & 0xffu) / 255.0,
    );
}

@compute @workgroup_size(8, 8, 1)
fn debug_view(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_id = global_id.xy;
    if any(pixel_id >= vec2u(view.main_pass_viewport.zw)) {
        return;
    }

    let depth = textureLoad(depth_buffer, pixel_id, 0);
    var color: vec3<f32>;

    if depth == 0.0 {
        color = SKY_COLOR;
    } else {
        let gpixel_raw = textureLoad(gbuffer, pixel_id, 0);
        let surface = gpixel_resolve(
            gpixel_raw,
            depth,
            pixel_id,
            view.main_pass_viewport.zw,
            view.world_from_clip,
        );

        if debug_constants.view_mode == DEBUG_WORLD_NORMAL {
            color = surface.world_normal * 0.5 + 0.5;
        } else if debug_constants.view_mode == DEBUG_WORLD_POSITION {
            color = abs(fract(surface.world_position * 0.5));
        } else if debug_constants.view_mode == DEBUG_BASE_COLOR {
            color = surface.material.base_color;
        } else if debug_constants.view_mode == DEBUG_MATERIAL {
            // No explicit material slot in the deferred gbuffer -- hash the
            // packed base_color+roughness u32 so distinct materials read as
            // distinct hues.
            color = id_color(gpixel_raw.r);
        } else if debug_constants.view_mode == DEBUG_DEPTH {
            let view_z = -depth_ndc_to_view_z(depth, view.clip_from_view, view.view_from_clip);
            let d = clamp(view_z / 32.0, 0.0, 1.0);
            color = vec3<f32>(1.0 - d);
        } else if debug_constants.view_mode == DEBUG_ROUGHNESS {
            color = vec3<f32>(surface.material.perceptual_roughness);
        } else if debug_constants.view_mode == DEBUG_METALLIC {
            color = vec3<f32>(surface.material.metallic);
        } else if debug_constants.view_mode == DEBUG_EMISSIVE {
            color = surface.material.emissive;
        } else if debug_constants.view_mode == DEBUG_MOTION_VECTOR {
            // Per-frame UV motion is typically << 1; scale 50x and bias to
            // mid-grey so static frames read as (0.5, 0.5, 0).
            let motion = textureLoad(motion_vectors, pixel_id, 0).xy;
            let scaled = motion * 50.0;
            color = vec3<f32>(scaled.x * 0.5 + 0.5, scaled.y * 0.5 + 0.5, 0.5);
        } else {
            color = vec3<f32>(1.0, 0.0, 1.0); // unknown mode -> magenta
        }
    }

    textureStore(view_output, pixel_id, vec4<f32>(color, 1.0));
}
