//! On-disk asset types for clustered-LOD geometry, plus the
//! `.cluster_mesh` loader and saver. Mirrors `bevy_pbr::meshlet::asset`
//! in structure (single file holds asset + loader + saver + envelope
//! helpers, lz4-framed body, uncompressed sizing header).

use alloc::sync::Arc;
use bevy_asset::{
    io::{Reader, Writer},
    saver::{AssetSaver, SavedAsset},
    Asset, AssetLoader, AssetPath, AsyncReadExt, AsyncWriteExt, LoadContext,
};
use bevy_math::{Vec2, Vec3, Vec4};
use bevy_reflect::{Reflect, TypePath};
use bevy_render::render_resource::ShaderType;
use bevy_tasks::block_on;
use bytemuck::{Pod, Zeroable};
use lz4_flex::frame::{FrameDecoder, FrameEncoder};
use std::io::{Read, Write};
use thiserror::Error;

/// Unique identifier for the [`ClusterMesh`] asset format. ASCII
/// `"CLUSTERS"` interpreted little-endian.
const CLUSTER_MESH_ASSET_MAGIC: u64 = u64::from_le_bytes(*b"CLUSTERS");

/// Current version of the [`ClusterMesh`] asset format. Bump on
/// incompatible struct-layout changes. v3 appends optional opacity
/// micro-map (OMM) slices; v2 files still load (with no OMM).
pub const CLUSTER_MESH_ASSET_VERSION: u64 = 3;

/// Oldest format version this loader still accepts. Versions in
/// `[MIN, VERSION]` load; older/newer are rejected.
const CLUSTER_MESH_ASSET_MIN_VERSION: u64 = 2;

/// First version carrying the optional OMM slices.
const CLUSTER_MESH_ASSET_OMM_VERSION: u64 = 3;

/// One baked opacity micro-map's location/shape within
/// [`ClusterMesh::omm_array_data`]. Field layout matches
/// `VkMicromapTriangleEXT` (offset, subdivisionLevel, format) so it feeds
/// the micromap build's triangle array directly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct OmmDesc {
    /// Byte offset into `omm_array_data`.
    pub offset: u32,
    /// Micro-triangle count is `4^subdivision_level`.
    pub subdivision_level: u16,
    /// OMM format: 1 = OC1 2-state, 2 = OC1 4-state.
    pub format: u16,
}

/// Histogram entry: how many OMMs share a (subdivision, format). Field layout
/// matches `VkMicromapUsageEXT`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct OmmUsage {
    pub count: u32,
    pub subdivision_level: u16,
    pub format: u16,
}

/// A mesh pre-processed into a DAG of clusters for hardware
/// ray-traced LOD selection.
///
/// Vertex and index data are stored in the raw layout consumed
/// directly by NV cluster acceleration structures — `[f32; 3]`
/// positions, `u32` indices — so the runtime can feed cluster build
/// descriptors without an unpack step.
///
/// The selector walks `nodes` (optional interior tree) down to
/// `groups` (DAG-cut units) and emits the contiguous cluster range
/// each accepted group owns.
// Reflect-opaque: `ClusterMesh` is never authored inline / reflected field-by-field (it loads from
// the binary `.cluster_mesh` format), but it must be `Reflect` so `HandleTemplate<ClusterMesh>` is
// reflectable and `RaytracingMesh3d("path")` can resolve from a `.bsn` string. `Clone` is cheap
// (the heavy fields are `Arc`s).
#[derive(Asset, Clone, Reflect)]
#[reflect(opaque)]
pub struct ClusterMesh {
    /// `[f32; 3]` positions, one entry per vertex. 12-byte stride
    /// keeps the stream tight; shaders read via an `array<f32>`
    /// accessor function in `cluster_bindings.wgsl`.
    pub vertex_positions: Arc<[Vec3]>,
    /// Octahedral-encoded normals packed as 2x16snorm in a `u32`,
    /// parallel to `vertex_positions`. Shaders decode via
    /// `octahedral_decode_signed(unpack2x16snorm(packed))`.
    pub vertex_normals: Arc<[u32]>,
    /// `[f32; 4]` tangents (xyz tangent, w bitangent sign, mikktspace).
    pub vertex_tangents: Arc<[Vec4]>,
    /// `[f32; 2]` UVs, parallel to `vertex_positions`.
    pub vertex_uvs: Arc<[Vec2]>,
    /// Per-triangle vertex indices (three `u32`s per triangle).
    /// Indices are mesh-local: they point into `vertex_positions`,
    /// not into a cluster-local table.
    pub indices: Arc<[u32]>,
    /// All clusters across all LOD levels, flat. DAG connectivity
    /// lives in `groups` / `nodes`.
    pub clusters: Arc<[Cluster]>,
    /// DAG-cut units. The selector evaluates a group's screen-space
    /// error and either descends (emit child groups) or accepts
    /// (emit all clusters in the group).
    pub groups: Arc<[ClusterLodGroup]>,
    /// Optional interior node tree built spatially over groups.
    /// Empty for small meshes that root directly at a group.
    pub nodes: Arc<[ClusterBvhNode]>,
    /// Flat child-id table referenced by `groups` and `nodes` via
    /// `children_offset` / `children_count` — keeps the node and
    /// group structs fixed-size.
    pub child_table: Arc<[u32]>,
    /// Parallel to `clusters` — `cluster_to_group[i]` is the group
    /// id that owns `clusters[i]`. Moved out of [`Cluster`] so the
    /// GPU manager can queue clusters before groups (group ids are
    /// rebased separately during this slice's upload).
    pub cluster_to_group: Arc<[u32]>,
    /// World-space AABB of the mesh.
    pub aabb: ClusterMeshAabb,
    /// Root group's quadric error. Lets the selector reject distant
    /// instances without descending.
    pub mesh_max_error: f32,
    /// Root group id (entry point for the DAG cut).
    pub root_group_id: u32,
    /// Root node id, or `u32::MAX` if there is no interior node
    /// tree (selector enters at `root_group_id` directly).
    pub root_node_id: u32,
    /// Max LOD level present (== bake recursion depth).
    pub lod_levels: u32,
    /// Opacity micro-map array build data (`VkMicromapEXT` input), or empty
    /// when this mesh has no baked OMM. Baked offline from the material's
    /// alpha texture — see the `solari-omm-cluster-path` plan.
    pub omm_array_data: Arc<[u8]>,
    /// Per-OMM descriptors into `omm_array_data` (micromap triangle array).
    pub omm_descs: Arc<[OmmDesc]>,
    /// Per-triangle OMM index, parallel to `indices` / 3 in the SAME
    /// (post-cluster) triangle order, so each cluster's index range slices it
    /// directly at CLAS-attach time. Negative = special index (-1 transparent,
    /// -2 opaque, -3 unknown-transparent, -4 unknown-opaque).
    pub omm_index: Arc<[i32]>,
    /// Usage histogram for the micromap build.
    pub omm_usage: Arc<[OmmUsage]>,
    /// Usage histogram for the BLAS/CLAS OMM attach.
    pub omm_index_usage: Arc<[OmmUsage]>,
}

impl ClusterMesh {
    #[inline]
    pub fn vertex_positions(&self) -> &[Vec3] {
        &self.vertex_positions
    }
    #[inline]
    pub fn vertex_normals(&self) -> &[u32] {
        &self.vertex_normals
    }
    #[inline]
    pub fn vertex_tangents(&self) -> &[Vec4] {
        &self.vertex_tangents
    }
    #[inline]
    pub fn vertex_uvs(&self) -> &[Vec2] {
        &self.vertex_uvs
    }
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
    #[inline]
    pub fn clusters(&self) -> &[Cluster] {
        &self.clusters
    }
    #[inline]
    pub fn groups(&self) -> &[ClusterLodGroup] {
        &self.groups
    }
    #[inline]
    pub fn nodes(&self) -> &[ClusterBvhNode] {
        &self.nodes
    }
    #[inline]
    pub fn child_table(&self) -> &[u32] {
        &self.child_table
    }
    #[inline]
    pub fn cluster_to_group(&self) -> &[u32] {
        &self.cluster_to_group
    }
    #[inline]
    pub fn aabb(&self) -> &ClusterMeshAabb {
        &self.aabb
    }
    #[inline]
    pub fn mesh_max_error(&self) -> f32 {
        self.mesh_max_error
    }
    #[inline]
    pub fn root_group_id(&self) -> u32 {
        self.root_group_id
    }
    #[inline]
    pub fn root_node_id(&self) -> u32 {
        self.root_node_id
    }
    #[inline]
    pub fn lod_levels(&self) -> u32 {
        self.lod_levels
    }

    /// Whether this mesh carries a baked opacity micro-map.
    #[inline]
    pub fn has_opacity_micromap(&self) -> bool {
        !self.omm_array_data.is_empty()
    }
    #[inline]
    pub fn omm_array_data(&self) -> &[u8] {
        &self.omm_array_data
    }
    #[inline]
    pub fn omm_descs(&self) -> &[OmmDesc] {
        &self.omm_descs
    }
    #[inline]
    pub fn omm_index(&self) -> &[i32] {
        &self.omm_index
    }
    #[inline]
    pub fn omm_usage(&self) -> &[OmmUsage] {
        &self.omm_usage
    }
    #[inline]
    pub fn omm_index_usage(&self) -> &[OmmUsage] {
        &self.omm_index_usage
    }

    /// Attach an offline-baked opacity micro-map. `per_tri_index` must be
    /// parallel to the mesh's triangles in the SAME order as [`Self::indices`]
    /// (post-cluster), so cluster index ranges slice it directly. See the
    /// `solari-omm-cluster-path` plan.
    pub fn set_opacity_micromap(
        &mut self,
        array_data: Vec<u8>,
        descs: Vec<OmmDesc>,
        per_tri_index: Vec<i32>,
        usage: Vec<OmmUsage>,
        index_usage: Vec<OmmUsage>,
    ) {
        self.omm_array_data = array_data.into();
        self.omm_descs = descs.into();
        self.omm_index = per_tri_index.into();
        self.omm_usage = usage.into();
        self.omm_index_usage = index_usage.into();
    }
}

/// Interleaved (AoS) vertex record for the bindless RT-pipeline resolve: one
/// vertex's shading attributes in one contiguous 16-byte slot, so a closest-hit
/// touches one cache line per vertex instead of separate SoA pools. Reached by
/// buffer-device-address via `physical_load` (field-by-field, no std430 padding —
/// exact 16 B: normal@0, tangent@4, uv@8). Position is NOT here — the closest-hit
/// reads it straight from the acceleration structure via
/// `@builtin(hit_triangle_vertex_positions)` (`VK_KHR_ray_tracing_position_fetch`),
/// and the rare CPU/sampling path that resolves a non-hit triangle reads it from
/// the separate `vertex_positions` SoA pool. Dropping the 12-byte position head
/// (28→16 B) packs more vertices per cache line → higher L1 hit, and removes the
/// position duplication between this pool and `vertex_positions`. Built at upload,
/// parallel to `vertex_positions` (same global vertex index).
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct PackedVertex {
    /// Octahedral-encoded normal (2×16snorm packed, matches `vertex_normals`).
    pub normal: u32,
    /// Tangent direction + bitangent sign packed into one `u32`: xyz in 10 bits
    /// each (unorm of the [-1,1] direction) + the sign in bit 30. 10-bit tangent
    /// precision is ample for normal mapping; the shader renormalizes on decode.
    pub tangent: u32,
    /// Texture coordinates (matches `vertex_uvs`). Kept full `f32` — UVs tile past
    /// [0,1], so f16 would lose texel precision.
    pub uv: [f32; 2],
}

/// Pack a `(xyz, sign)` tangent into [`PackedVertex::tangent`]: 10 bits per
/// direction component (unorm of `[-1,1]`) + the bitangent sign in bit 30.
#[inline]
pub fn pack_tangent(tangent: Vec4) -> u32 {
    let dir = Vec3::new(tangent.x, tangent.y, tangent.z).normalize_or_zero();
    let q = |c: f32| -> u32 { (((c * 0.5 + 0.5).clamp(0.0, 1.0) * 1023.0) + 0.5) as u32 & 0x3ff };
    let sign = u32::from(tangent.w < 0.0);
    q(dir.x) | (q(dir.y) << 10) | (q(dir.z) << 20) | (sign << 30)
}

/// A single cluster — the unit of CLAS build. Mirrors what an NV
/// per-cluster build descriptor needs (vertex range + index range)
/// plus runtime selection / shading metadata.
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct Cluster {
    /// Start of this cluster's vertex range in
    /// [`ClusterMesh::vertex_positions`] (and the parallel
    /// normals / tangents / uvs streams).
    pub vertex_offset: u32,
    /// Vertex count (≤ 256, NV CLAS per-cluster cap).
    pub vertex_count: u32,
    /// Start of this cluster's index range in
    /// [`ClusterMesh::indices`] (in `u32`s).
    pub index_offset: u32,
    /// Triangle count. Three indices per triangle.
    pub triangle_count: u32,
    /// Tight bounding sphere covering the cluster's vertices —
    /// xyz center, w radius. Single `vec4<f32>` on the WGSL side.
    /// Per-cluster screen-space-error fallback when the group
    /// sphere is too coarse.
    pub bounds_sphere: [f32; 4],
    /// Per-cluster local material slot. All triangles in one
    /// cluster share this slot.
    pub local_material_id: u32,
    /// LOD level for debug overlays (0 = finest).
    pub lod_level: u32,
    /// Padding to bring the Rust struct size up to 48 bytes, matching
    /// WGSL's `array<Cluster>` stride. `bounds_sphere` is `[f32; 4]`
    /// here (alignment 4) but `vec4<f32>` in WGSL (alignment 16),
    /// which forces the WGSL struct stride to `roundUp(40, 16) = 48`.
    /// Without these 8 bytes, every `clusters[N]` past index 0 reads
    /// shifted bytes — vertex / index offsets land on the wrong
    /// fields and the rendered geometry tears.
    pub _pad: [u32; 2],
}

/// A DAG-cut unit. The selector evaluates this group's traversal
/// sphere and quadric error and either descends (enqueue child
/// groups) or accepts (emit all `cluster_count` clusters starting
/// at `cluster_start`).
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct ClusterLodGroup {
    /// Start of this group's clusters in [`ClusterMesh::clusters`].
    pub cluster_start: u32,
    pub cluster_count: u32,
    /// Pointer into [`ClusterMesh::child_table`] — variable-width
    /// list of child group ids enqueued on descent. `children_count
    /// == 0` marks a leaf group (DAG cut must accept).
    pub children_offset: u32,
    pub children_count: u32,
    /// Traversal sphere — xyz center, w radius. Packed as a 4-vec
    /// for single-load WGSL access.
    pub traversal_sphere: [f32; 4],
    /// Monotone-enlarged across the DAG: parent ≥ every child.
    pub max_quadric_error: f32,
    /// Parent group id, or `u32::MAX` for the root group.
    pub parent_group: u32,
    /// LOD level this group lives at (0 = finest).
    pub lod_level: u32,
    pub _pad: u32,
}

/// Interior DAG node — spatial subdivision over groups for the
/// selector's upper levels. Optional; small meshes set
/// [`ClusterMesh::root_node_id`] to `u32::MAX` and the selector
/// enters at `root_group_id` directly.
///
/// Variable-arity: the children list lives in
/// [`ClusterMesh::child_table`], and leaves can be either groups
/// or other nodes (see [`ClusterBvhNode::is_group_leaf`]).
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct ClusterBvhNode {
    pub traversal_sphere: [f32; 4],
    pub max_quadric_error: f32,
    /// Pointer into [`ClusterMesh::child_table`].
    pub children_offset: u32,
    /// Packed: low 16 bits = `children_count`, bit 16 =
    /// `is_group_leaf` (1 = children are group ids, 0 = node ids).
    pub children_packed: u32,
    pub _pad: u32,
}

impl ClusterBvhNode {
    #[inline]
    pub fn children_count(&self) -> u32 {
        self.children_packed & 0xFFFF
    }
    #[inline]
    pub fn is_group_leaf(&self) -> bool {
        (self.children_packed >> 16) & 1 == 1
    }
    #[inline]
    pub fn pack_children(count: u32, is_group_leaf: bool) -> u32 {
        (count & 0xFFFF) | (u32::from(is_group_leaf) << 16)
    }
}

/// Tight AABB used for instance-level visibility and debug.
/// `[f32; 4]` fields (xyz + unused `w = 0`) so the GPU reads
/// naturally as `vec4<f32>` without padding gymnastics.
#[derive(Copy, Clone, Default, Pod, Zeroable, ShaderType, Debug)]
#[repr(C)]
pub struct ClusterMeshAabb {
    pub center: [f32; 4],
    pub half_extent: [f32; 4],
}

/// Synchronous writer for offline CLI tools — same wire format as
/// [`ClusterMeshSaver`], but bypasses bevy_asset's async I/O so
/// bake binaries can stream directly to a [`std::io::Write`].
/// [`AssetSaver`] itself is documented as not suited for general
/// asset persistence (see bevy issue #11216).
pub fn write_cluster_mesh_sync<W: Write>(
    asset: &ClusterMesh,
    mut writer: W,
) -> Result<(), ClusterMeshSaveOrLoadError> {
    writer.write_all(&CLUSTER_MESH_ASSET_MAGIC.to_le_bytes())?;
    writer.write_all(&CLUSTER_MESH_ASSET_VERSION.to_le_bytes())?;
    writer.write_all(bytemuck::bytes_of(&asset.aabb))?;
    writer.write_all(&asset.mesh_max_error.to_le_bytes())?;
    writer.write_all(&asset.root_group_id.to_le_bytes())?;
    writer.write_all(&asset.root_node_id.to_le_bytes())?;
    writer.write_all(&asset.lod_levels.to_le_bytes())?;

    let mut encoder = FrameEncoder::new(writer);
    write_slice(&asset.vertex_positions, &mut encoder)?;
    write_slice(&asset.vertex_normals, &mut encoder)?;
    write_slice(&asset.vertex_tangents, &mut encoder)?;
    write_slice(&asset.vertex_uvs, &mut encoder)?;
    write_slice(&asset.indices, &mut encoder)?;
    write_slice(&asset.clusters, &mut encoder)?;
    write_slice(&asset.groups, &mut encoder)?;
    write_slice(&asset.nodes, &mut encoder)?;
    write_slice(&asset.child_table, &mut encoder)?;
    write_slice(&asset.cluster_to_group, &mut encoder)?;
    // v3: optional OMM slices (empty for non-cutout meshes).
    write_slice(&asset.omm_array_data, &mut encoder)?;
    write_slice(&asset.omm_descs, &mut encoder)?;
    write_slice(&asset.omm_index, &mut encoder)?;
    write_slice(&asset.omm_usage, &mut encoder)?;
    write_slice(&asset.omm_index_usage, &mut encoder)?;
    encoder.finish()?;
    Ok(())
}

/// An [`AssetSaver`] for `.cluster_mesh` [`ClusterMesh`] assets.
#[derive(TypePath)]
pub struct ClusterMeshSaver;

impl AssetSaver for ClusterMeshSaver {
    type Asset = ClusterMesh;
    type Settings = ();
    type OutputLoader = ClusterMeshLoader;
    type Error = ClusterMeshSaveOrLoadError;

    async fn save(
        &self,
        writer: &mut Writer,
        asset: SavedAsset<'_, '_, ClusterMesh>,
        _settings: &(),
        _asset_path: AssetPath<'_>,
    ) -> Result<(), ClusterMeshSaveOrLoadError> {
        writer
            .write_all(&CLUSTER_MESH_ASSET_MAGIC.to_le_bytes())
            .await?;
        writer
            .write_all(&CLUSTER_MESH_ASSET_VERSION.to_le_bytes())
            .await?;

        writer.write_all(bytemuck::bytes_of(&asset.aabb)).await?;
        writer
            .write_all(&asset.mesh_max_error.to_le_bytes())
            .await?;
        writer.write_all(&asset.root_group_id.to_le_bytes()).await?;
        writer.write_all(&asset.root_node_id.to_le_bytes()).await?;
        writer.write_all(&asset.lod_levels.to_le_bytes()).await?;

        let mut writer = FrameEncoder::new(AsyncWriteSyncAdapter(writer));
        write_slice(&asset.vertex_positions, &mut writer)?;
        write_slice(&asset.vertex_normals, &mut writer)?;
        write_slice(&asset.vertex_tangents, &mut writer)?;
        write_slice(&asset.vertex_uvs, &mut writer)?;
        write_slice(&asset.indices, &mut writer)?;
        write_slice(&asset.clusters, &mut writer)?;
        write_slice(&asset.groups, &mut writer)?;
        write_slice(&asset.nodes, &mut writer)?;
        write_slice(&asset.child_table, &mut writer)?;
        write_slice(&asset.cluster_to_group, &mut writer)?;
        // v3: optional OMM slices (empty for non-cutout meshes).
        write_slice(&asset.omm_array_data, &mut writer)?;
        write_slice(&asset.omm_descs, &mut writer)?;
        write_slice(&asset.omm_index, &mut writer)?;
        write_slice(&asset.omm_usage, &mut writer)?;
        write_slice(&asset.omm_index_usage, &mut writer)?;
        writer.finish()?;

        Ok(())
    }
}

/// An [`AssetLoader`] for `.cluster_mesh` [`ClusterMesh`] assets.
#[derive(TypePath, Default)]
pub struct ClusterMeshLoader;

impl AssetLoader for ClusterMeshLoader {
    type Asset = ClusterMesh;
    type Settings = ();
    type Error = ClusterMeshSaveOrLoadError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<ClusterMesh, ClusterMeshSaveOrLoadError> {
        let magic = async_read_u64(reader).await?;
        if magic != CLUSTER_MESH_ASSET_MAGIC {
            return Err(ClusterMeshSaveOrLoadError::WrongFileType);
        }
        let version = async_read_u64(reader).await?;
        if !(CLUSTER_MESH_ASSET_MIN_VERSION..=CLUSTER_MESH_ASSET_VERSION).contains(&version) {
            return Err(ClusterMeshSaveOrLoadError::WrongVersion { found: version });
        }
        let has_omm = version >= CLUSTER_MESH_ASSET_OMM_VERSION;

        let mut bytes = [0u8; size_of::<ClusterMeshAabb>()];
        reader.read_exact(&mut bytes).await?;
        let aabb = bytemuck::cast(bytes);

        let mesh_max_error = f32::from_le_bytes(async_read_4(reader).await?);
        let root_group_id = u32::from_le_bytes(async_read_4(reader).await?);
        let root_node_id = u32::from_le_bytes(async_read_4(reader).await?);
        let lod_levels = u32::from_le_bytes(async_read_4(reader).await?);

        let reader = &mut FrameDecoder::new(AsyncReadSyncAdapter(reader));
        let vertex_positions = read_slice(reader)?;
        let vertex_normals = read_slice(reader)?;
        let vertex_tangents = read_slice(reader)?;
        let vertex_uvs = read_slice(reader)?;
        let indices = read_slice(reader)?;
        let clusters = read_slice(reader)?;
        let groups = read_slice(reader)?;
        let nodes = read_slice(reader)?;
        let child_table = read_slice(reader)?;
        let cluster_to_group = read_slice(reader)?;
        // v3+ appends the optional OMM slices; v2 files have none.
        let (omm_array_data, omm_descs, omm_index, omm_usage, omm_index_usage) = if has_omm {
            (
                read_slice(reader)?,
                read_slice(reader)?,
                read_slice(reader)?,
                read_slice(reader)?,
                read_slice(reader)?,
            )
        } else {
            (
                Arc::from(&[][..]),
                Arc::from(&[][..]),
                Arc::from(&[][..]),
                Arc::from(&[][..]),
                Arc::from(&[][..]),
            )
        };

        Ok(ClusterMesh {
            vertex_positions,
            vertex_normals,
            vertex_tangents,
            vertex_uvs,
            indices,
            clusters,
            groups,
            nodes,
            child_table,
            cluster_to_group,
            aabb,
            mesh_max_error,
            root_group_id,
            root_node_id,
            lod_levels,
            omm_array_data,
            omm_descs,
            omm_index,
            omm_usage,
            omm_index_usage,
        })
    }

    fn extensions(&self) -> &[&str] {
        &["cluster_mesh"]
    }
}

#[derive(Error, Debug)]
pub enum ClusterMeshSaveOrLoadError {
    #[error("file was not a ClusterMesh asset")]
    WrongFileType,
    #[error("expected asset version {CLUSTER_MESH_ASSET_VERSION} but found version {found}")]
    WrongVersion { found: u64 },
    #[error("failed to compress or decompress asset data")]
    CompressionOrDecompression(#[from] lz4_flex::frame::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

async fn async_read_u64(reader: &mut dyn Reader) -> Result<u64, std::io::Error> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).await?;
    Ok(u64::from_le_bytes(bytes))
}

async fn async_read_4(reader: &mut dyn Reader) -> Result<[u8; 4], std::io::Error> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).await?;
    Ok(bytes)
}

fn read_u64(reader: &mut dyn Read) -> Result<u64, std::io::Error> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_slice<T: Pod>(
    field: &[T],
    writer: &mut dyn Write,
) -> Result<(), ClusterMeshSaveOrLoadError> {
    writer.write_all(&(field.len() as u64).to_le_bytes())?;
    writer.write_all(bytemuck::cast_slice(field))?;
    Ok(())
}

fn read_slice<T: Pod>(reader: &mut dyn Read) -> Result<Arc<[T]>, std::io::Error> {
    let len = read_u64(reader)? as usize;

    let mut data: Arc<[T]> = core::iter::repeat_with(T::zeroed).take(len).collect();
    let slice = Arc::get_mut(&mut data).unwrap();
    reader.read_exact(bytemuck::cast_slice_mut(slice))?;

    Ok(data)
}

// TODO: Use async for everything and get rid of this adapter
struct AsyncWriteSyncAdapter<'a>(&'a mut Writer);

impl Write for AsyncWriteSyncAdapter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        block_on(self.0.write(buf))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        block_on(self.0.flush())
    }
}

// TODO: Use async for everything and get rid of this adapter
struct AsyncReadSyncAdapter<'a>(&'a mut dyn Reader);

impl Read for AsyncReadSyncAdapter<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        block_on(self.0.read(buf))
    }
}
