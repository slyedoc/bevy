#![allow(
    clippy::doc_markdown,
    reason = "Aurora identifiers (PGBuffer, PgbufferTextures, etc.) and Vulkan format names need not be backticked."
)]
//! Aurora's primary G-buffer texture set.
//!
//! Five storage textures, one per channel of the per-pixel RT primary
//! visibility output. Aurora's compute shader writes them in M-B sub-3 and
//! downstream lighting / compose passes (M-C onward) read them the same way
//! solari's lighting reads its raster G-buffer.
//!
//! | Channel         | Format          | Meaning                                       |
//! | ---             | ---             | ---                                           |
//! | `world_position`| `Rgba32Float`   | xyz: world-space hit point. w: hit-flag (0 = miss, 1 = hit, NaN = sky). |
//! | `world_normal`  | `Rgba16Float`   | xy: octahedral-encoded shading normal. zw: padding. |
//! | `motion_vector` | `Rg32Float`     | screen-space motion (uv-space, like raster).  |
//! | `material_id`   | `R32Uint`       | 24-bit cluster id + 8-bit instance index.     |
//! | `depth`         | `R32Float`      | linear hit distance (`ray_t`), -1 on miss.    |
//!
//! Format choice notes:
//!
//! - All five formats are in wgpu's universally-storage-bindable list, so no
//!   extra adapter feature flags are needed.
//! - Normals use 16-bit float (octahedral xy in `xy`) instead of `Rg16Snorm`
//!   because the `Rg16Snorm` storage texture format requires
//!   `TEXTURE_FORMAT_16BIT_NORM` which isn't ubiquitous; `Rgba16Float` is.
//! - Depth is `R32Float` (linear distance) rather than `Depth32Float` because
//!   Aurora's consumer is a compute shader, not a raster depth test, and
//!   storage-binding a depth-format texture is restricted on most adapters.

use bevy_ecs::resource::Resource;
use bevy_math::UVec2;
use bevy_render::render_resource::{
    Extent3d, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};
use bevy_render::renderer::RenderDevice;

/// Default PGBuffer extent before per-camera viewport sizing lands. Sized to
/// 1080p so the bunny smoke test renders meaningfully on most monitors.
pub const DEFAULT_PGBUFFER_EXTENT: UVec2 = UVec2::new(1920, 1080);

/// The viewport extent Aurora's PGBuffer is allocated for. External systems
/// (eventually the camera-viewport tracker) mutate this to drive
/// reallocation in [`super::resize_pgbuffer`].
#[derive(Resource, Debug, Clone, Copy)]
pub struct PgbufferExtent(pub UVec2);

impl Default for PgbufferExtent {
    fn default() -> Self {
        Self(DEFAULT_PGBUFFER_EXTENT)
    }
}

/// All five PGBuffer storage textures + their default views. Inserted into the
/// render world by [`super::AuroraPrimaryVisibilityPlugin`].
#[derive(Resource)]
pub struct PgbufferTextures {
    pub extent: UVec2,

    pub world_position: Texture,
    pub world_position_view: TextureView,

    pub world_normal: Texture,
    pub world_normal_view: TextureView,

    pub motion_vector: Texture,
    pub motion_vector_view: TextureView,

    pub material_id: Texture,
    pub material_id_view: TextureView,

    pub depth: Texture,
    pub depth_view: TextureView,
}

impl PgbufferTextures {
    /// Allocate all five textures + default views sized to `extent`.
    pub fn allocate(device: &RenderDevice, extent: UVec2) -> Self {
        let make = |label: &'static str, format: TextureFormat| -> (Texture, TextureView) {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d {
                    width: extent.x,
                    height: extent.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor {
                label: Some(label),
                ..Default::default()
            });
            (texture, view)
        };

        let (world_position, world_position_view) =
            make("aurora.pgbuffer.world_position", TextureFormat::Rgba32Float);
        let (world_normal, world_normal_view) =
            make("aurora.pgbuffer.world_normal", TextureFormat::Rgba16Float);
        let (motion_vector, motion_vector_view) =
            make("aurora.pgbuffer.motion_vector", TextureFormat::Rg32Float);
        let (material_id, material_id_view) =
            make("aurora.pgbuffer.material_id", TextureFormat::R32Uint);
        let (depth, depth_view) = make("aurora.pgbuffer.depth", TextureFormat::R32Float);

        Self {
            extent,
            world_position,
            world_position_view,
            world_normal,
            world_normal_view,
            motion_vector,
            motion_vector_view,
            material_id,
            material_id_view,
            depth,
            depth_view,
        }
    }

    /// Total byte size on the GPU. Useful telemetry for resize decisions.
    pub fn approx_bytes(&self) -> u64 {
        let pixels = u64::from(self.extent.x) * u64::from(self.extent.y);
        pixels * (16 /* world_position Rgba32Float */
            + 8 /* world_normal Rgba16Float */
            + 8 /* motion_vector Rg32Float */
            + 4 /* material_id R32Uint */
            + 4 /* depth R32Float */)
    }
}
