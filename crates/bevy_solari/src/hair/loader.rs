//! Loader for Cem Yuksel's `.hair` strand format
//! (<https://www.cemyuksel.com/research/hairmodels/>) — the de-facto hair test
//! assets used by pbrt / Mitsuba / NVIDIA RTXCR (`straight`, `wavy`, `curly`,
//! `wCurly`, `natural`, `dark`). A `.hair` file is a 128-byte header followed by
//! a few flag-gated arrays; we map it straight onto [`HairAsset`].

use bevy_asset::{io::Reader, AssetLoader, LoadContext};
use bevy_math::Vec3;
use std::io::{Error, ErrorKind};

use super::asset::{HairAsset, HairStrand};

/// Loads `*.hair` files into [`HairAsset`].
#[derive(Default, bevy_reflect::TypePath)]
pub struct HairLoader;

impl AssetLoader for HairLoader {
    type Asset = HairAsset;
    type Settings = ();
    type Error = Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<HairAsset, Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        parse_hair(&bytes)
    }

    fn extensions(&self) -> &[&str] {
        &["hair"]
    }
}

const HEADER_SIZE: usize = 128;
const FLAG_SEGMENTS: u32 = 1 << 0;
const FLAG_POINTS: u32 = 1 << 1;
const FLAG_THICKNESS: u32 = 1 << 2;
// bits 3 (transparency) and 4 (color) are skipped over but unused.
const FLAG_TRANSPARENCY: u32 = 1 << 3;
const FLAG_COLOR: u32 = 1 << 4;

fn invalid(msg: &'static str) -> Error {
    Error::new(ErrorKind::InvalidData, msg)
}

fn parse_hair(bytes: &[u8]) -> Result<HairAsset, Error> {
    if bytes.len() < HEADER_SIZE || &bytes[0..4] != b"HAIR" {
        return Err(invalid("not a HAIR file (bad magic / too short)"));
    }
    let u32_at = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let f32_at = |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

    let num_strands = u32_at(4) as usize;
    let num_points = u32_at(8) as usize;
    let flags = u32_at(12);
    let default_segments = u32_at(16);
    let default_thickness = f32_at(20);

    if flags & FLAG_POINTS == 0 {
        return Err(invalid("HAIR file has no points array"));
    }

    // Arrays follow the header in flag order: segments, points, thickness,
    // transparency, color.
    let mut cursor = HEADER_SIZE;

    // Per-strand segment counts (u16), or the default for every strand.
    let segments: Vec<u32> = if flags & FLAG_SEGMENTS != 0 {
        let end = cursor + num_strands * 2;
        if end > bytes.len() {
            return Err(invalid("HAIR file truncated (segments)"));
        }
        let s = (cursor..end)
            .step_by(2)
            .map(|o| u16::from_le_bytes([bytes[o], bytes[o + 1]]) as u32)
            .collect();
        cursor = end;
        s
    } else {
        vec![default_segments; num_strands]
    };

    // Points (f32 ×3 each).
    let points_end = cursor + num_points * 12;
    if points_end > bytes.len() {
        return Err(invalid("HAIR file truncated (points)"));
    }
    let points_base = cursor;
    cursor = points_end;

    // Optional per-point thickness.
    let thickness_base = if flags & FLAG_THICKNESS != 0 {
        let end = cursor + num_points * 4;
        if end > bytes.len() {
            return Err(invalid("HAIR file truncated (thickness)"));
        }
        let base = cursor;
        cursor = end;
        Some(base)
    } else {
        None
    };
    // (transparency / color arrays are present in some files but unused here.)
    let _ = (FLAG_TRANSPARENCY, FLAG_COLOR, cursor);

    let point = |i: usize| {
        let o = points_base + i * 12;
        Vec3::new(
            f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]),
            f32::from_le_bytes([bytes[o + 4], bytes[o + 5], bytes[o + 6], bytes[o + 7]]),
            f32::from_le_bytes([bytes[o + 8], bytes[o + 9], bytes[o + 10], bytes[o + 11]]),
        )
    };
    // `.hair` thickness is the strand diameter → radius is half. No array ⇒
    // the file default; some files leave it 0 ("undefined"), so fall back to a
    // small visible diameter rather than degenerate zero-radius segments.
    let fallback_diameter = if default_thickness > 0.0 {
        default_thickness
    } else {
        0.05
    };
    let radius = |i: usize| match thickness_base {
        Some(base) => {
            let o = base + i * 4;
            0.5 * f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
        }
        None => 0.5 * fallback_diameter,
    };

    let mut strands = Vec::with_capacity(num_strands);
    let mut p = 0usize; // running point index
    for &seg in &segments {
        let n = seg as usize + 1; // points = segments + 1
        if p + n > num_points {
            return Err(invalid("HAIR file strand point count overruns points array"));
        }
        let mut pts = Vec::with_capacity(n);
        let mut radii = Vec::with_capacity(n);
        for k in 0..n {
            pts.push(point(p + k));
            radii.push(radius(p + k).max(1e-5));
        }
        p += n;
        strands.push(HairStrand { points: pts, radii });
    }

    Ok(HairAsset { strands })
}
