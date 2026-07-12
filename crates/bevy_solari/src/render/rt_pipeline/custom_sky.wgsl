#define_import_path bevy_solari::custom_sky

// The built-in procedural sky (`SolariSky::Procedural`): a ground/horizon/zenith
// gradient. Colors are artistic (0-1), scaled to physical luminance (cd/m²) so the
// blit's exposure reads them like every other traced radiance.
//
// `SolariSky::Shader` replaces this whole module. A replacement must declare
// `#define_import_path bevy_solari::custom_sky` and define
// `sample_custom_sky(ray_direction) -> vec3<f32>` returning physical radiance.
// It composes into the primary-miss stage only; keep it self-contained (no
// scene-binding imports).
const SKY_GROUND_COLOR = vec3(0.1855, 0.159, 0.1855); // vec3(0.35, 0.3, 0.35) * 0.53
const SKY_HORIZON_COLOR = vec3(1.0, 1.0, 1.0);
const SKY_ZENITH_COLOR = vec3(0.08, 0.37, 0.73);
const SKY_LUMINANCE = 5000.0;

fn sample_custom_sky(ray_direction: vec3<f32>) -> vec3<f32> {
    let sky_gradient_t = pow(smoothstep(0.0, 0.4, ray_direction.y), 0.35);
    let sky_gradient = mix(SKY_HORIZON_COLOR, SKY_ZENITH_COLOR, sky_gradient_t);
    let ground_to_sky_t = smoothstep(-0.01, 0.0, ray_direction.y);
    return mix(SKY_GROUND_COLOR, sky_gradient, ground_to_sky_t) * SKY_LUMINANCE;
}
