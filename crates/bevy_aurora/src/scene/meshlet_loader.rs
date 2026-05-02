#![allow(
    clippy::doc_markdown,
    reason = "Vulkan / WGSL identifiers (vk::*, get_meshlet_vertex_position) would each need backticks otherwise."
)]
//! Convert a [`bevy_pbr::experimental::meshlet::MeshletMesh`] into the
//! flat, dequantized form Aurora uploads to GPU buffers consumed by the
//! cluster_AS build.
//!
//! Bevy's meshlet format stores vertex positions as a bitstream packed
//! per-meshlet with channel-specific bit widths and a per-meshlet
//! quantization factor. Vulkan's `vkCmdBuildClusterAccelerationStructureIndirectNV`
//! expects flat `R32G32B32_SFLOAT` positions (or one of a fixed list of formats),
//! so we dequantize on the CPU at upload time and produce one big concatenated
//! `Vec<[f32; 3]>` plus a per-meshlet metadata array.
//!
//! The dequant routine is a direct CPU port of
//! `bevy_pbr::meshlet::meshlet_bindings::get_meshlet_vertex_position` (see the
//! WGSL source for the bit-packing layout). Indices are copied verbatim — the
//! cluster_AS extension supports the same `u8` index type the meshlet format
//! already uses (`VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV = 1`),
//! and meshlet indices are already local to each meshlet's vertex range.

use bevy_pbr::experimental::meshlet::{Meshlet, MeshletMesh};

/// Per-meshlet entry in a [`DequantizedMeshletMesh`], indexing into the flat
/// position + index arrays.
#[derive(Debug, Clone, Copy)]
pub struct DequantizedMeshlet {
    /// Offset, in vertices, into [`DequantizedMeshletMesh::positions`].
    pub vertex_offset: u32,
    /// Number of vertices in this meshlet (= `Meshlet::vertex_count_minus_one + 1`).
    pub vertex_count: u32,
    /// Offset, in `u32`s, into [`DequantizedMeshletMesh::indices`]. Each
    /// triangle is 3 consecutive `u32`s; total = `3 * triangle_count` u32s
    /// (= `12 * triangle_count` bytes).
    pub index_offset: u32,
    /// Number of triangles in this meshlet.
    pub triangle_count: u32,
}

impl DequantizedMeshlet {
    /// `triangle_count * 3`. Convenience for sizing index reads.
    #[inline]
    pub fn index_count(&self) -> u32 {
        self.triangle_count * 3
    }
}

/// All meshlets of one [`MeshletMesh`], dequantized to the flat format Aurora
/// uploads to GPU buffers consumed by `vkCmdBuildClusterAccelerationStructureIndirectNV`.
#[derive(Debug, Clone)]
pub struct DequantizedMeshletMesh {
    /// Concatenated `f32x3` positions for every meshlet, in source order. A
    /// given meshlet's positions live at
    /// `[meshlet.vertex_offset .. meshlet.vertex_offset + meshlet.vertex_count]`.
    pub positions: Vec<[f32; 3]>,
    /// Per-meshlet index streams concatenated, widened from the meshlet
    /// asset's u8 storage to u32 because the cluster_AS extension's NVIDIA
    /// driver implementation does not appear to handle 8-bit indices
    /// correctly (Unreal's NaniteRayTracingDecodePageClusters.usf also uses
    /// `IndexFormat = 4` = 32-bit). Each index is local to its owning
    /// meshlet's vertex range (i.e. in `0 .. meshlet.vertex_count`).
    pub indices: Vec<u32>,
    /// One entry per meshlet, in source order.
    pub meshlets: Vec<DequantizedMeshlet>,
}

impl DequantizedMeshletMesh {
    /// Convenience: total triangle count across all meshlets.
    pub fn total_triangle_count(&self) -> u32 {
        self.meshlets.iter().map(|m| m.triangle_count).sum()
    }

    /// Convenience: total vertex count across all meshlets (with duplicates
    /// across meshlet boundaries).
    pub fn total_vertex_count(&self) -> u32 {
        self.meshlets.iter().map(|m| m.vertex_count).sum()
    }
}

/// Dequantize every meshlet in `mesh` into the flat layout described by
/// [`DequantizedMeshletMesh`]. Pure CPU work — no GPU contact.
///
/// Cost: O(total vertex count). For a ~50k-meshlet bunny that's a one-time
/// few-MB dequant on asset load; cheap enough that we don't need a GPU
/// compute path until the streaming milestone.
pub fn dequantize_meshlet_mesh(mesh: &MeshletMesh) -> DequantizedMeshletMesh {
    let meshlets_src = mesh.meshlets();
    let positions_src = mesh.vertex_positions();
    let indices_src = mesh.indices();

    // Pre-size: each meshlet contributes `vertex_count` positions + `3 * tri`
    // indices to the concatenated outputs.
    let mut total_verts = 0usize;
    let mut total_indices = 0usize;
    for m in meshlets_src {
        total_verts += usize::from(m.vertex_count_minus_one) + 1;
        total_indices += usize::from(m.triangle_count) * 3;
    }

    let mut positions = Vec::with_capacity(total_verts);
    let mut indices = Vec::with_capacity(total_indices);
    let mut meshlets = Vec::with_capacity(meshlets_src.len());

    for m in meshlets_src {
        let vertex_offset = u32::try_from(positions.len())
            .expect("dequantized vertex offset overflows u32 -- mesh too large for cluster_AS");
        let index_offset = u32::try_from(indices.len())
            .expect("dequantized index offset overflows u32 -- mesh too large for cluster_AS");
        let vertex_count = u32::from(m.vertex_count_minus_one) + 1;
        let triangle_count = u32::from(m.triangle_count);

        // Dequant this meshlet's vertices into the running tail of `positions`.
        dequantize_meshlet_into(m, positions_src, &mut positions);

        // Widen the meshlet's u8 indices to u32 (cluster_AS NV driver path
        // requires 32-bit indices despite the spec listing 8-bit as a
        // supported format).
        let idx_start = m.start_index_id as usize;
        let idx_end = idx_start + (triangle_count as usize) * 3;
        indices.extend(indices_src[idx_start..idx_end].iter().map(|&i| u32::from(i)));

        meshlets.push(DequantizedMeshlet {
            vertex_offset,
            vertex_count,
            index_offset,
            triangle_count,
        });
    }

    DequantizedMeshletMesh {
        positions,
        indices,
        meshlets,
    }
}

/// CPU port of `bevy_pbr::meshlet::meshlet_bindings::get_meshlet_vertex_position`
/// for one meshlet. Walks the bitstream from `meshlet.start_vertex_position_bit`,
/// extracts `bits_per_channel_*` per X/Y/Z, then applies the meshlet's stored
/// quantization params to produce world-space `f32x3` positions.
///
/// Appends to `out` rather than allocating a fresh vec so the per-mesh
/// dequantize can stream into a single concatenated buffer.
pub fn dequantize_meshlet_into(
    meshlet: &Meshlet,
    vertex_positions: &[u32],
    out: &mut Vec<[f32; 3]>,
) {
    let vertex_count = u32::from(meshlet.vertex_count_minus_one) + 1;
    let bits = [
        u32::from(meshlet.bits_per_vertex_position_channel_x),
        u32::from(meshlet.bits_per_vertex_position_channel_y),
        u32::from(meshlet.bits_per_vertex_position_channel_z),
    ];
    let bits_per_vertex = bits[0] + bits[1] + bits[2];

    // From `bevy_pbr::meshlet::from_mesh::CENTIMETERS_PER_METER` and the WGSL
    // dequant: `(1u << factor) * 100` is the multiplier applied at encode; we
    // divide by it on decode.
    let inv_quant = 1.0 / ((1u32 << meshlet.vertex_position_quantization_factor) as f32 * 100.0);
    let mins = [
        meshlet.min_vertex_position_channel_x,
        meshlet.min_vertex_position_channel_y,
        meshlet.min_vertex_position_channel_z,
    ];

    out.reserve(vertex_count as usize);
    for v in 0..vertex_count {
        let mut start_bit = meshlet.start_vertex_position_bit + v * bits_per_vertex;
        let mut packed = [0u32; 3];
        for (i, &b) in bits.iter().enumerate() {
            let lower_word = (start_bit / 32) as usize;
            let bit_offset = start_bit & 31;
            let mut next_32 = vertex_positions[lower_word] >> bit_offset;
            // Spans a u32 boundary. `bit_offset` is in `1..32` here so the
            // shift is well-defined.
            if bit_offset + b > 32 {
                next_32 |= vertex_positions[lower_word + 1] << (32 - bit_offset);
            }
            let mask = if b == 32 { u32::MAX } else { (1u32 << b) - 1 };
            packed[i] = next_32 & mask;
            start_bit += b;
        }
        out.push([
            (packed[0] as f32 + mins[0]) * inv_quant,
            (packed[1] as f32 + mins[1]) * inv_quant,
            (packed[2] as f32 + mins[2]) * inv_quant,
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_pbr::experimental::meshlet::Meshlet;

    /// Builds a hand-crafted single-meshlet bitstream + indices for a
    /// 1-triangle test case with vertices at `(0,0,0) (1,0,0) (0,1,0)`.
    /// 8 bits per channel, quantization factor 0 → packed `100 = 0x64`
    /// represents one world-space meter.
    fn synthetic_triangle() -> (Meshlet, Vec<u32>, Vec<u8>) {
        // 24 bits/vertex × 3 verts = 72 bits ≤ 3 × 32-bit words.
        //   word 0 = vert0(xyz=000) | vert1.x(0x64)        = 0x64000000
        //   word 1 = vert1(yz=00)   | vert2(x=0, y=0x64)   = 0x64000000
        //   word 2 = vert2.z(0)                            = 0x00000000
        let positions = vec![0x6400_0000_u32, 0x6400_0000_u32, 0x0000_0000_u32];
        let indices = vec![0u8, 1, 2];
        let meshlet = Meshlet {
            start_vertex_position_bit: 0,
            start_vertex_attribute_id: 0,
            start_index_id: 0,
            vertex_count_minus_one: 2,
            triangle_count: 1,
            padding: 0,
            bits_per_vertex_position_channel_x: 8,
            bits_per_vertex_position_channel_y: 8,
            bits_per_vertex_position_channel_z: 8,
            vertex_position_quantization_factor: 0,
            min_vertex_position_channel_x: 0.0,
            min_vertex_position_channel_y: 0.0,
            min_vertex_position_channel_z: 0.0,
        };
        (meshlet, positions, indices)
    }

    #[test]
    fn dequant_round_trips_synthetic_triangle() {
        let (meshlet, positions, _indices) = synthetic_triangle();
        let mut out = Vec::new();
        dequantize_meshlet_into(&meshlet, &positions, &mut out);
        let expected: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        for (got, exp) in out.iter().zip(expected.iter()) {
            for axis in 0..3 {
                let diff = (got[axis] - exp[axis]).abs();
                assert!(
                    diff < 1e-5,
                    "axis {axis}: got {got:?} expected {exp:?} (diff {diff})"
                );
            }
        }
    }

    #[test]
    fn dequant_appends_without_clearing() {
        let (meshlet, positions, _) = synthetic_triangle();
        let mut out = vec![[99.0_f32, 99.0, 99.0]];
        dequantize_meshlet_into(&meshlet, &positions, &mut out);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], [99.0, 99.0, 99.0], "prior data must be preserved");
        assert_eq!(out[1], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn dequant_handles_min_offset() {
        // Same triangle but shifted by min = -50 (in quantized units, so -0.5 m).
        // Encoded packed values: 0, 100, 0  -> +(-50) = -50, +50, -50 quantized.
        let positions = vec![0x6400_0000_u32, 0x6400_0000_u32, 0x0000_0000_u32];
        let meshlet = Meshlet {
            start_vertex_position_bit: 0,
            start_vertex_attribute_id: 0,
            start_index_id: 0,
            vertex_count_minus_one: 2,
            triangle_count: 1,
            padding: 0,
            bits_per_vertex_position_channel_x: 8,
            bits_per_vertex_position_channel_y: 8,
            bits_per_vertex_position_channel_z: 8,
            vertex_position_quantization_factor: 0,
            min_vertex_position_channel_x: -50.0,
            min_vertex_position_channel_y: -50.0,
            min_vertex_position_channel_z: -50.0,
        };
        let mut out = Vec::new();
        dequantize_meshlet_into(&meshlet, &positions, &mut out);
        // packed=0 + min=-50 = -50; /100 = -0.5
        // packed=100 + min=-50 = 50;  /100 = 0.5
        for (got, exp) in out.iter().zip(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
            ]
            .iter(),
        ) {
            for axis in 0..3 {
                assert!(
                    (got[axis] - exp[axis]).abs() < 1e-5,
                    "axis {axis}: got {got:?} expected {exp:?}"
                );
            }
        }
    }
}
