// The built-in procedural sky (`SolariSky::Procedural`): a ground/horizon/zenith
// gradient. Colors are artistic (0-1), scaled to physical luminance (cd/m²) so the
// blit's exposure reads them like every other traced radiance.
//
// `SolariSky::Shader` replaces this whole module. A replacement is a Slang
// module defining `public float3 sample_custom_sky(float3 ray_direction)`
// returning physical radiance. It composes into the primary-miss stage only;
// keep it self-contained (no scene-binding imports).
module custom_sky;

static const float3 SKY_GROUND_COLOR = float3(0.1855, 0.159, 0.1855); // float3(0.35, 0.3, 0.35) * 0.53
static const float3 SKY_HORIZON_COLOR = float3(1.0, 1.0, 1.0);
static const float3 SKY_ZENITH_COLOR = float3(0.08, 0.37, 0.73);
static const float SKY_LUMINANCE = 5000.0;

public float3 sample_custom_sky(float3 ray_direction)
{
    let sky_gradient_t = pow(smoothstep(0.0, 0.4, ray_direction.y), 0.35);
    let sky_gradient = lerp(SKY_HORIZON_COLOR, SKY_ZENITH_COLOR, sky_gradient_t);
    let ground_to_sky_t = smoothstep(-0.01, 0.0, ray_direction.y);
    return lerp(SKY_GROUND_COLOR, sky_gradient, ground_to_sky_t) * SKY_LUMINANCE;
}
