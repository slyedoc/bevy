use bevy_camera::MainPassResolutionOverride;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    system::{Commands, Query, Res},
};
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        Buffer, BufferDescriptor, BufferUsages, TextureDescriptor, TextureDimension, TextureFormat,
        TextureUsages, TextureView, TextureViewDescriptor,
    },
    renderer::RenderDevice,
};

use crate::render::SolariCamera;

/// Byte size of the `PathReservoir` shader struct. MUST match the
/// `array<PathReservoir>` stride declared in `restir_bindings.wgsl`.
pub const PATH_RESERVOIR_STRUCT_SIZE: u64 = 80;

/// Byte size of one `LightTileSample` in the light-tile pool. MUST match
/// the struct in `restir_bindings.wgsl`.
const LIGHT_TILE_SAMPLE_STRUCT_SIZE: u64 = 32;

/// Byte size of the GI `GiReservoir` struct. MUST match `restir_bindings.wgsl`.
const GI_RESERVOIR_STRUCT_SIZE: u64 = 48;

pub const LIGHT_TILE_BLOCKS: u64 = 128;
pub const LIGHT_TILE_SAMPLES_PER_BLOCK: u64 = 1024;

/// ReGIR world-space light grid: hash-table cell count (power of two). Same
/// spatial-hash scheme as the old world cache: position-keyed cells with
/// distance-LOD sizing, marked alive on query and refilled each frame.
/// MUST match `REGIR_TABLE_SIZE` in `restir_bindings.wgsl`.
pub const REGIR_TABLE_SIZE: u64 = 65536;
/// Presampled light entries per ReGIR cell. MUST match
/// `REGIR_ENTRIES_PER_CELL` in `restir_bindings.wgsl`.
pub const REGIR_ENTRIES_PER_CELL: u64 = 32;

/// Per-view GPU resources for the full-RT ReSTIR path tracer.
///
/// The G-buffer is kept lean — position, normal, motion — with material data
/// refetched from the scene bindings at shade time rather than baked in.
/// Position and normal are double-buffered (`[current, previous]`, ping-ponged
/// by frame parity in [`super::node::restir`]) for temporal reprojection;
/// reservoirs are likewise ping-ponged.
#[derive(Component)]
pub struct RestirResources {
    /// World-space hit position (`xyz`) + packed material id (`w`).
    pub world_position: [TextureView; 2],
    /// World-space shading normal.
    pub world_normal: [TextureView; 2],
    /// Screen-space motion vectors (current frame).
    pub motion_vectors: TextureView,
    /// Primary-hit texture coordinates (current frame), for shade-time
    /// material re-resolve. `.xy` = uv; full-float so texture sampling is
    /// stable. Stays lean — uv + material id instead of baked albedo/pbr.
    pub uv: TextureView,
    /// First-hit distance of the specular reflection ray (written by the
    /// specular-GI pass; `RAY_T_MAX` on an environment miss). Drives the
    /// virtual-reflection reprojection for DLSS's specular motion guide.
    pub specular_hit_distance: TextureView,
    /// Per-pixel DI (direct-lighting) reservoirs. Two fixed roles (not
    /// frame-parity ping-pong): `[0]` = temporal history (persists across
    /// frames), `[1]` = this-frame intermediate (initial+temporal →
    /// spatial/shade handoff).
    pub reservoirs: [Buffer; 2],
    /// Per-pixel GI (indirect, one-bounce) reservoirs. Same `[history,
    /// intermediate]` roles as the DI pair.
    pub gi_reservoirs: [Buffer; 2],
    /// Presampled light-tile sample pool — the uniform candidate source the
    /// ReGIR grid selects from, and the out-of-grid fallback.
    pub light_tiles: Buffer,
    /// ReGIR cell identity hashes (0 = empty slot). Spatial hash keyed by
    /// quantized world position + distance LOD; linear-probed.
    pub regir_checksums: Buffer,
    /// ReGIR cell lifetimes: reset on query, decremented by the decay pass;
    /// a cell that hits 0 is freed.
    pub regir_life: Buffer,
    /// Per-cell representative point + cell size (`xyz` = cell center, `w` =
    /// cell size), written on insert — the fill pass's RIS target.
    pub regir_cell_data: Buffer,
    /// Per-cell light samples (`REGIR_TABLE_SIZE × REGIR_ENTRIES_PER_CELL`),
    /// RIS-selected from the tile pool with a distance-to-cell target — a
    /// pixel's initial candidates come from a pool already importance-reduced
    /// to lights that matter NEAR its surface point.
    pub regir_samples: Buffer,
    /// Self-owned previous-frame `clip_from_world`: two `mat4x4<f32>` slots,
    /// frame-parity ping-ponged on the GPU (one writer thread in the
    /// visibility pass). Replaces a prepass-fed previous-view uniform.
    pub view_clip_from_world: Buffer,
    pub view_size: UVec2,
}

pub fn prepare_restir_resources(
    query: Query<
        (
            Entity,
            &ExtractedCamera,
            Option<&RestirResources>,
            Option<&MainPassResolutionOverride>,
        ),
        With<SolariCamera>,
    >,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for (entity, camera, existing, resolution_override) in &query {
        let Some(mut view_size) = camera.physical_viewport_size else {
            continue;
        };
        // DLSS renders at a lower internal resolution and upscales; honor the
        // override so the G-buffer / reservoirs match the render resolution.
        if let Some(MainPassResolutionOverride(override_size)) = resolution_override {
            view_size = *override_size;
        }

        // Resources are sized to the (render) viewport; reuse until it changes.
        if existing.map(|r| r.view_size) == Some(view_size) {
            continue;
        }

        let storage_texture = |name: &str, format| {
            render_device
                .create_texture(&TextureDescriptor {
                    label: Some(name),
                    size: view_size.to_extents(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    // TEXTURE_BINDING so DLSS can read motion vectors as a guide.
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        };

        let reservoir_buffer = |name: &str| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(name),
                size: (view_size.x * view_size.y) as u64 * PATH_RESERVOIR_STRUCT_SIZE,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };

        let gi_reservoir_buffer = |name: &str| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(name),
                size: (view_size.x * view_size.y) as u64 * GI_RESERVOIR_STRUCT_SIZE,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };

        let light_tiles = render_device.create_buffer(&BufferDescriptor {
            label: Some("restir_light_tiles"),
            size: LIGHT_TILE_BLOCKS
                * LIGHT_TILE_SAMPLES_PER_BLOCK
                * LIGHT_TILE_SAMPLE_STRUCT_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // ReGIR hash grid (wgpu zero-initializes: checksum 0 = empty cell).
        let regir_buffer = |name: &str, size: u64| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(name),
                size,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let regir_checksums = regir_buffer("restir_regir_checksums", REGIR_TABLE_SIZE * 4);
        let regir_life = regir_buffer("restir_regir_life", REGIR_TABLE_SIZE * 4);
        let regir_cell_data = regir_buffer("restir_regir_cell_data", REGIR_TABLE_SIZE * 16);
        let regir_samples = regir_buffer(
            "restir_regir_samples",
            REGIR_TABLE_SIZE * REGIR_ENTRIES_PER_CELL * LIGHT_TILE_SAMPLE_STRUCT_SIZE,
        );

        // Two `mat4x4<f32>` (64 B each): current + previous clip_from_world.
        let view_clip_from_world = render_device.create_buffer(&BufferDescriptor {
            label: Some("restir_view_clip_from_world"),
            size: 2 * 64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        commands.entity(entity).insert(RestirResources {
            world_position: [
                storage_texture("restir_world_position_a", TextureFormat::Rgba32Float),
                storage_texture("restir_world_position_b", TextureFormat::Rgba32Float),
            ],
            world_normal: [
                storage_texture("restir_world_normal_a", TextureFormat::Rgba16Float),
                storage_texture("restir_world_normal_b", TextureFormat::Rgba16Float),
            ],
            motion_vectors: storage_texture("restir_motion_vectors", TextureFormat::Rgba16Float),
            uv: storage_texture("restir_uv", TextureFormat::Rgba32Float),
            specular_hit_distance: storage_texture(
                "restir_specular_hit_distance",
                TextureFormat::R32Float,
            ),
            reservoirs: [
                reservoir_buffer("restir_reservoirs_a"),
                reservoir_buffer("restir_reservoirs_b"),
            ],
            gi_reservoirs: [
                gi_reservoir_buffer("restir_gi_reservoirs_a"),
                gi_reservoir_buffer("restir_gi_reservoirs_b"),
            ],
            light_tiles,
            regir_checksums,
            regir_life,
            regir_cell_data,
            regir_samples,
            view_clip_from_world,
            view_size,
        });
    }
}
