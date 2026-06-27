//! Clean-room barycentric triangle subdivision for the tessellation
//! template path (Phase 2b).
//!
//! Produces the **topology** of a uniformly subdivided unit triangle —
//! barycentric micro-vertices plus micro-triangle indices — independent of
//! any base mesh. A CLAS *template* is built once per subdivision level from
//! this topology ([`ClasArena::upload_mesh_via_template`](super::clas_arena)
//! supplies the template/instantiate plumbing); instantiating that template
//! with displaced micro-vertex positions yields the dense displaced
//! geometry (Phase 2c).
//!
//! Why generate rather than embed Unreal Engine's tessellation table: the
//! Epic table ships under Epic's EULA, kept license-segregated from the
//! reference's Apache code. Bevy is Apache/MIT, so the pattern is generated
//! procedurally here instead. Uniform subdivision is crack-free on its own;
//! per-edge (adaptive, crack-free across LOD seams) factors are a later
//! extension of the same lattice — see [`SubdividedTriangle::new`].
//!
//! ## Lattice
//!
//! Micro-vertices sit on the regular barycentric lattice of a triangle wound
//! `A → B → C`. A vertex is addressed by integer weights `(i, j, k)` with
//! `i + j + k = level`; its barycentric coordinate is `(i, j, k) / level`,
//! i.e. weight `i/level` on `A`, `j/level` on `B`, `k/level` on `C`. Vertices
//! are emitted in `i`-major, then `j`-ascending order. Each lattice cell
//! contributes one upward micro-triangle and (away from the `A` corner row)
//! one downward one, for `level²` micro-triangles total, all wound CCW to
//! match the base triangle.

extern crate alloc;
use alloc::vec::Vec;

/// A uniformly subdivided unit triangle at `level` segments per edge.
///
/// `level == 1` is the trivial subdivision (the base triangle itself: 3
/// vertices, 1 triangle). The vertex/triangle counts grow as
/// `(level+1)(level+2)/2` vertices and `level²` triangles.
#[derive(Debug, Clone)]
pub struct SubdividedTriangle {
    /// Segments per edge. `level² ` micro-triangles.
    pub level: u32,
    /// Barycentric weight `(w_a, w_b, w_c)` of each micro-vertex, each
    /// component in `[0, 1]` and the three summing to `1`. Indexed by the
    /// values in [`Self::indices`].
    pub barycentrics: Vec<[f32; 3]>,
    /// Micro-triangle corner indices into [`Self::barycentrics`], three per
    /// triangle, wound CCW to match a base triangle wound `A → B → C`.
    pub indices: Vec<u32>,
}

impl SubdividedTriangle {
    /// Build the uniform subdivision at `level` segments per edge.
    ///
    /// # Panics
    ///
    /// Panics if `level == 0` (a triangle has at least one segment per edge).
    pub fn new(level: u32) -> Self {
        assert!(level >= 1, "subdivision level must be >= 1");

        let l = level;
        let vertex_count = ((l + 1) * (l + 2) / 2) as usize;
        let mut barycentrics = Vec::with_capacity(vertex_count);

        // `i`-major, `j`-ascending emission — must match `index_of`.
        for i in 0..=l {
            for j in 0..=(l - i) {
                let k = l - i - j;
                barycentrics.push([
                    i as f32 / l as f32,
                    j as f32 / l as f32,
                    k as f32 / l as f32,
                ]);
            }
        }
        debug_assert_eq!(barycentrics.len(), vertex_count);

        let mut indices = Vec::with_capacity((l * l * 3) as usize);
        for i in 0..l {
            for j in 0..(l - i) {
                // Upward triangle: valid wherever `i + j <= level - 1`.
                indices.push(index_of(l, i, j));
                indices.push(index_of(l, i + 1, j));
                indices.push(index_of(l, i, j + 1));

                // Downward triangle: valid wherever `i + j <= level - 2`.
                if i + j + 1 < l {
                    indices.push(index_of(l, i + 1, j));
                    indices.push(index_of(l, i + 1, j + 1));
                    indices.push(index_of(l, i, j + 1));
                }
            }
        }
        debug_assert_eq!(indices.len(), (l * l * 3) as usize);

        Self {
            level,
            barycentrics,
            indices,
        }
    }

    /// Number of micro-vertices: `(level+1)(level+2)/2`.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.barycentrics.len()
    }

    /// Number of micro-triangles: `level²`.
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// Linear index of the lattice vertex with integer weights `(i, j)` (the
/// third weight `k = level - i - j` is implied). Mirrors the `i`-major,
/// `j`-ascending emission order in [`SubdividedTriangle::new`]:
/// `offset(i) = i*(level+1) - i*(i-1)/2`, then `+ j`.
#[inline]
fn index_of(level: u32, i: u32, j: u32) -> u32 {
    // Sum over earlier rows `ii < i`, each holding `(level - ii + 1)`
    // vertices: `Σ (level + 1 - ii) = i*(level+1) - i*(i-1)/2`.
    let offset = i * (level + 1) - i * (i.wrapping_sub(1)) / 2;
    offset + j
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;

    /// 2D embedding used to check winding + watertightness: place the base
    /// triangle as `A=(0,0)`, `B=(1,0)`, `C=(0,1)` (CCW) and map each
    /// barycentric to its planar point.
    fn embed(bary: [f32; 3]) -> (f64, f64) {
        let (a, b, c) = (bary[0] as f64, bary[1] as f64, bary[2] as f64);
        // x = b*Bx + c*Cx = b ; y = b*By + c*Cy = c
        let _ = a;
        (b, c)
    }

    fn signed_area(p0: (f64, f64), p1: (f64, f64), p2: (f64, f64)) -> f64 {
        0.5 * ((p1.0 - p0.0) * (p2.1 - p0.1) - (p1.1 - p0.1) * (p2.0 - p0.0))
    }

    #[test]
    fn counts_match_closed_form() {
        for level in 1..=11u32 {
            let t = SubdividedTriangle::new(level);
            assert_eq!(
                t.vertex_count(),
                ((level + 1) * (level + 2) / 2) as usize,
                "vertex count at level {level}"
            );
            assert_eq!(
                t.triangle_count(),
                (level * level) as usize,
                "triangle count at level {level}"
            );
        }
    }

    #[test]
    fn level_one_is_base_triangle() {
        let t = SubdividedTriangle::new(1);
        assert_eq!(t.vertex_count(), 3);
        assert_eq!(t.triangle_count(), 1);

        // The single micro-triangle must be the base triangle: all three
        // corners, wound the same cyclic way as `A → B → C`. The emission
        // order doesn't guarantee index 0 == corner A, so resolve corners by
        // barycentric and check the triple is a rotation of `[A, B, C]`.
        let corner = |target: [f32; 3]| {
            t.barycentrics
                .iter()
                .position(|b| b.iter().zip(target).all(|(x, y)| (x - y).abs() < 1e-6))
                .unwrap() as u32
        };
        let (a, b, c) = (
            corner([1.0, 0.0, 0.0]),
            corner([0.0, 1.0, 0.0]),
            corner([0.0, 0.0, 1.0]),
        );
        let tri = [t.indices[0], t.indices[1], t.indices[2]];
        assert!(
            tri == [a, b, c] || tri == [b, c, a] || tri == [c, a, b],
            "level-1 triangle {tri:?} is not a CCW rotation of [A={a}, B={b}, C={c}]"
        );
    }

    #[test]
    fn barycentrics_sum_to_one_and_include_corners() {
        for level in 1..=11u32 {
            let t = SubdividedTriangle::new(level);
            for b in &t.barycentrics {
                let s = b[0] + b[1] + b[2];
                assert!((s - 1.0).abs() < 1e-5, "bary {b:?} sums to {s} at {level}");
                assert!(b.iter().all(|&w| (-1e-6..=1.0 + 1e-6).contains(&w)));
            }
            // The three pure corners must be present exactly.
            let has = |target: [f32; 3]| {
                t.barycentrics
                    .iter()
                    .any(|b| b.iter().zip(target).all(|(x, y)| (x - y).abs() < 1e-6))
            };
            assert!(has([1.0, 0.0, 0.0]) && has([0.0, 1.0, 0.0]) && has([0.0, 0.0, 1.0]));
        }
    }

    #[test]
    fn indices_in_range_and_nondegenerate() {
        for level in 1..=11u32 {
            let t = SubdividedTriangle::new(level);
            let n = t.vertex_count() as u32;
            for tri in t.indices.chunks_exact(3) {
                assert!(tri.iter().all(|&v| v < n));
                assert!(tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2]);
            }
        }
    }

    #[test]
    fn all_triangles_wound_ccw() {
        for level in 1..=11u32 {
            let t = SubdividedTriangle::new(level);
            for tri in t.indices.chunks_exact(3) {
                let p = |idx: u32| embed(t.barycentrics[idx as usize]);
                let area = signed_area(p(tri[0]), p(tri[1]), p(tri[2]));
                assert!(area > 0.0, "non-CCW micro-triangle {tri:?} at level {level}");
            }
        }
    }

    /// Every interior edge is shared by exactly two micro-triangles and every
    /// boundary edge by exactly one — i.e. the subdivision is watertight with
    /// no T-junctions.
    #[test]
    fn watertight_edge_manifold() {
        for level in 1..=11u32 {
            let t = SubdividedTriangle::new(level);
            let mut edge_uses: BTreeMap<(u32, u32), u32> = BTreeMap::new();
            for tri in t.indices.chunks_exact(3) {
                for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                    let key = if a < b { (a, b) } else { (b, a) };
                    *edge_uses.entry(key).or_insert(0) += 1;
                }
            }
            for (edge, count) in &edge_uses {
                assert!(
                    *count == 1 || *count == 2,
                    "edge {edge:?} used {count}x at level {level}"
                );
            }
            // Boundary edge count must equal the three sides at `level`
            // segments each: `3 * level`.
            let boundary = edge_uses.values().filter(|&&c| c == 1).count();
            assert_eq!(boundary, (3 * level) as usize, "boundary edges at {level}");
        }
    }

    #[test]
    fn total_micro_area_equals_unit() {
        // The micro-triangles tile the base triangle exactly: their summed
        // (embedded) area equals the base triangle's area (0.5).
        for level in 1..=11u32 {
            let t = SubdividedTriangle::new(level);
            let mut total = 0.0;
            for tri in t.indices.chunks_exact(3) {
                let p = |idx: u32| embed(t.barycentrics[idx as usize]);
                total += signed_area(p(tri[0]), p(tri[1]), p(tri[2]));
            }
            assert!((total - 0.5).abs() < 1e-6, "tiled area {total} at level {level}");
        }
    }
}
