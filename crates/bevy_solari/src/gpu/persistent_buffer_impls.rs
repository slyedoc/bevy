use crate::geometry::asset::{Cluster, ClusterBloatAabb, ClusterBvhNode, ClusterLodGroup, PackedVertex};
use crate::geometry::ClusterIndex;
use super::persistent_buffer::PersistentGpuBufferable;
use alloc::sync::Arc;
use bevy_math::{Vec2, Vec3, Vec4};
use bevy_render::render_resource::BufferAddress;
use wgpu_types::WriteOnly;

impl PersistentGpuBufferable for Arc<[u32]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<u32>()
    }

    fn write_bytes_le(
        &self,
        _: Self::Metadata,
        mut buffer_slice: WriteOnly<[u8]>,
        _: BufferAddress,
    ) {
        buffer_slice.copy_from_slice(bytemuck::cast_slice(self));
    }
}

impl PersistentGpuBufferable for Arc<[PackedVertex]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<PackedVertex>()
    }

    fn write_bytes_le(
        &self,
        _: Self::Metadata,
        mut buffer_slice: WriteOnly<[u8]>,
        _: BufferAddress,
    ) {
        buffer_slice.copy_from_slice(bytemuck::cast_slice(self));
    }
}

impl PersistentGpuBufferable for Arc<[Vec2]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<Vec2>()
    }

    fn write_bytes_le(
        &self,
        _: Self::Metadata,
        mut buffer_slice: WriteOnly<[u8]>,
        _: BufferAddress,
    ) {
        buffer_slice.copy_from_slice(bytemuck::cast_slice(self));
    }
}

impl PersistentGpuBufferable for Arc<[Vec3]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<Vec3>()
    }

    fn write_bytes_le(
        &self,
        _: Self::Metadata,
        mut buffer_slice: WriteOnly<[u8]>,
        _: BufferAddress,
    ) {
        buffer_slice.copy_from_slice(bytemuck::cast_slice(self));
    }
}

impl PersistentGpuBufferable for Arc<[Vec4]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<Vec4>()
    }

    fn write_bytes_le(
        &self,
        _: Self::Metadata,
        mut buffer_slice: WriteOnly<[u8]>,
        _: BufferAddress,
    ) {
        buffer_slice.copy_from_slice(bytemuck::cast_slice(self));
    }
}

impl PersistentGpuBufferable for Arc<[[u16; 4]]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<[u16; 4]>()
    }

    fn write_bytes_le(&self, _: Self::Metadata, mut buffer_slice: WriteOnly<[u8]>, _: BufferAddress) {
        buffer_slice.copy_from_slice(bytemuck::cast_slice(self));
    }
}

impl PersistentGpuBufferable for Arc<[ClusterBloatAabb]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<ClusterBloatAabb>()
    }

    fn write_bytes_le(&self, _: Self::Metadata, mut buffer_slice: WriteOnly<[u8]>, _: BufferAddress) {
        buffer_slice.copy_from_slice(bytemuck::cast_slice(self));
    }
}

impl PersistentGpuBufferable for Arc<[Cluster]> {
    /// `(vertex_base, index_base)` — shifts each cluster's
    /// `vertex_offset` / `index_offset` from mesh-local to
    /// persistent-buffer-global slots during the write.
    type Metadata = (u32, u32);

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<Cluster>()
    }

    fn write_bytes_le(
        &self,
        (vertex_base, index_base): Self::Metadata,
        mut buffer_slice: WriteOnly<[u8]>,
        _: BufferAddress,
    ) {
        const SIZE: usize = size_of::<Cluster>();
        for (i, c) in self.iter().enumerate() {
            let shifted = Cluster {
                vertex_offset: c.vertex_offset + vertex_base,
                index_offset: c.index_offset + index_base,
                ..*c
            };
            let bytes: [u8; SIZE] = bytemuck::cast(shifted);
            buffer_slice
                .slice(i * SIZE..(i + 1) * SIZE)
                .copy_from_slice(&bytes);
        }
    }
}

impl PersistentGpuBufferable for Arc<[ClusterLodGroup]> {
    /// `(cluster_base, child_table_base)` — shifts `cluster_start`
    /// into the global cluster pool ([`ClusterIndex`]) and
    /// `children_offset` into the global child table (`u32` —
    /// anonymous flat-table offset, not a slot id).
    ///
    /// `parent_group` is stored mesh-local — the selector rebases
    /// it in-shader by adding `cluster_instance_group_bases[instance]`
    /// at read time. Threading `group_base` through here would
    /// require a peek-before-allocate API on `PersistentGpuBuffer`
    /// (group_base is only known *after* the queue_write reserves
    /// the range), which isn't worth it for a single u32 add on the
    /// shader side.
    type Metadata = (ClusterIndex, u32);

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<ClusterLodGroup>()
    }

    fn write_bytes_le(
        &self,
        (cluster_base, child_table_base): Self::Metadata,
        mut buffer_slice: WriteOnly<[u8]>,
        _: BufferAddress,
    ) {
        const SIZE: usize = size_of::<ClusterLodGroup>();
        for (i, g) in self.iter().enumerate() {
            let shifted = ClusterLodGroup {
                cluster_start: g.cluster_start + cluster_base.0,
                children_offset: if g.children_count > 0 {
                    g.children_offset + child_table_base
                } else {
                    g.children_offset
                },
                ..*g
            };
            let bytes: [u8; SIZE] = bytemuck::cast(shifted);
            buffer_slice
                .slice(i * SIZE..(i + 1) * SIZE)
                .copy_from_slice(&bytes);
        }
    }
}

impl PersistentGpuBufferable for Arc<[ClusterBvhNode]> {
    /// `child_table_base` — shifts `children_offset` into the
    /// global child table.
    type Metadata = u32;

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<ClusterBvhNode>()
    }

    fn write_bytes_le(
        &self,
        child_table_base: Self::Metadata,
        mut buffer_slice: WriteOnly<[u8]>,
        _: BufferAddress,
    ) {
        const SIZE: usize = size_of::<ClusterBvhNode>();
        for (i, n) in self.iter().enumerate() {
            let shifted = ClusterBvhNode {
                children_offset: if n.children_count() > 0 {
                    n.children_offset + child_table_base
                } else {
                    n.children_offset
                },
                ..*n
            };
            let bytes: [u8; SIZE] = bytemuck::cast(shifted);
            buffer_slice
                .slice(i * SIZE..(i + 1) * SIZE)
                .copy_from_slice(&bytes);
        }
    }
}
