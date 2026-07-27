//! `.animclip` — a serialized [`AnimationClip`] of rigid TRS keyframes, bound onto a spawned
//! hierarchy by node name-path.
//!
//! [`AnimationClip`] is a bag of type-erased `Box<dyn AnimationCurve>` and does not round-trip
//! through reflection, so there is no way to write one into a scene file. This is the same
//! custom-binary + loader answer bevy uses for other unreflectable assets: an offline tool writes
//! the clip next to a scene, and the scene root names it with an [`AnimatedScene`] component.
//!
//! The pieces:
//!   1. [`AnimationClipLoader`] reads `.animclip` into an [`AnimationClip`], keyed by
//!      `AnimationTargetId::from_names(path)` where `path` is a node's `Name` chain. Curves are
//!      built the way `bevy_gltf` builds them — LINEAR keyframes over `Transform` T/R/S.
//!   2. [`wire_animated_scenes`] sees an [`AnimatedScene`] whose clip has loaded, builds an
//!      `AnimationGraph`/`AnimationPlayer` on that root, and walks the spawned tree inserting
//!      `(AnimationTargetId, AnimatedBy(root))` on every named descendant — the same convention
//!      `bevy_gltf`'s loader uses, so the ids match the clip's keys.
//!
//! From there stock animation writes each node's `Transform` and normal transform propagation
//! carries it to the node's children. An animated transform-only node drives its mesh children,
//! which is what makes this work for rigged props whose joints carry no geometry.
//!
//! Binding is by name-PATH, not by bare name, so sibling subtrees may reuse names. The root's own
//! name is excluded from the path: paths start at the root's direct children.
//!
//! Format (little-endian) — a writer must match byte-for-byte:
//! ```text
//!   magic  "ANIMCLP\x01"                     (8 bytes)
//!   u32    target_count
//!   per target:
//!     u16  path_len;  per component: u16 len + len UTF-8 bytes (Name chain, top-level..node)
//!     u8   channel_mask                       (bit0 = translation, bit1 = rotation, bit2 = scale)
//!     per present channel, in T,R,S order:
//!       u32  key_count
//!       key_count × f32        times (seconds)
//!       key_count × dim × f32  values         (dim: T=3, R=4 [x,y,z,w], S=3)
//! ```
//! Interpolation is not stored — every sampler is LINEAR.

use alloc::{string::ToString, vec::Vec};

use bevy_asset::{io::Reader, AssetLoader, Assets, Handle, LoadContext};
use bevy_ecs::{name::Name, prelude::*};
use bevy_math::{
    curve::{ConstantCurve, Interval, UnevenSampleAutoCurve},
    Quat, ToPrecision, Vec3,
};
use bevy_reflect::{prelude::ReflectDefault, Reflect, TypePath};
use bevy_transform::components::Transform;

use crate::{
    animated_field,
    animation_curves::AnimatableCurve,
    graph::{AnimationGraph, AnimationGraphHandle},
    AnimatedBy, AnimationClip, AnimationPlayer, AnimationTargetId, RepeatAnimation, VariableCurve,
};

/// Names an [`AnimationClip`] to load, bind and play across this entity's descendants.
///
/// Put it on the root of a spawned hierarchy — typically written into a scene file by an offline
/// baker, e.g. `bevy_animation::AnimatedScene("mech/DRV4_Drover.animclip")`. Everything else is
/// automatic: [`AnimationPlugin`](crate::AnimationPlugin) installs the loader and the wiring
/// system, so no per-asset code or extra plugin is required.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Default)]
pub struct AnimatedScene(pub Handle<AnimationClip>);

/// Once an [`AnimatedScene`] root's clip has loaded, build its graph/player and bind the tree.
///
/// Runs once per root — the `Without<AnimationPlayer>` filter drops it after wiring. Gating on the
/// clip being loaded doubles as a "the hierarchy has finished spawning" check, since a scene's
/// entities and its clip resolve through the same asset pipeline.
pub fn wire_animated_scenes(
    mut commands: Commands,
    clips: Res<Assets<AnimationClip>>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    roots: Query<(Entity, &AnimatedScene), Without<AnimationPlayer>>,
    children: Query<&Children>,
    names: Query<&Name>,
) {
    for (root, anim) in &roots {
        if clips.get(&anim.0).is_none() {
            continue; // clip still loading (and, by extension, the tree may still be spawning)
        }
        let (graph, node) = AnimationGraph::from_clip(anim.0.clone());
        let graph_handle = graphs.add(graph);
        let mut player = AnimationPlayer::default();
        player.play(node).set_repeat(RepeatAnimation::Forever);
        commands
            .entity(root)
            .insert((player, AnimationGraphHandle(graph_handle)));

        let mut path: Vec<Name> = Vec::new();
        wire_descendants(root, root, &mut path, &children, &names, &mut commands);
    }
}

fn wire_descendants(
    entity: Entity,
    root: Entity,
    path: &mut Vec<Name>,
    children: &Query<&Children>,
    names: &Query<&Name>,
    commands: &mut Commands,
) {
    let Ok(kids) = children.get(entity) else {
        return;
    };
    for &child in kids {
        let named = names.get(child).ok().cloned();
        if let Some(name) = &named {
            path.push(name.clone());
            commands
                .entity(child)
                .insert((AnimationTargetId::from_names(path.iter()), AnimatedBy(root)));
        }
        wire_descendants(child, root, path, children, names, commands);
        if named.is_some() {
            path.pop();
        }
    }
}

// ---------------------------------------------------------------------------------------------

/// Loads a `.animclip` into an [`AnimationClip`].
#[derive(Default, TypePath)]
pub struct AnimationClipLoader;

const MAGIC: &[u8; 8] = b"ANIMCLP\x01";
const CH_T: u8 = 1;
const CH_R: u8 = 2;
const CH_S: u8 = 4;

/// Errors that can occur loading a `.animclip`.
#[derive(Debug, thiserror::Error)]
pub enum AnimationClipLoaderError {
    /// The file could not be read.
    #[error("could not read .animclip: {0}")]
    Io(#[from] std::io::Error),
    /// The file did not begin with the `ANIMCLP\x01` magic.
    #[error("bad .animclip magic")]
    BadMagic,
    /// The file ended mid-record.
    #[error("truncated .animclip")]
    Truncated,
    /// A name-path component was not valid UTF-8.
    #[error("invalid UTF-8 in .animclip name path")]
    BadUtf8,
}

impl AssetLoader for AnimationClipLoader {
    type Asset = AnimationClip;
    type Settings = ();
    type Error = AnimationClipLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _ctx: &mut LoadContext<'_>,
    ) -> Result<AnimationClip, AnimationClipLoaderError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        parse_animclip(&bytes)
    }

    fn extensions(&self) -> &[&str] {
        &["animclip"]
    }
}

/// A tiny forward byte cursor (avoids a bytes/serde dep for this fixed layout).
struct Cur<'a>(&'a [u8]);

impl Cur<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], AnimationClipLoaderError> {
        if self.0.len() < n {
            return Err(AnimationClipLoaderError::Truncated);
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8, AnimationClipLoaderError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<usize, AnimationClipLoaderError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as usize)
    }
    fn u32(&mut self) -> Result<usize, AnimationClipLoaderError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }
    /// `n` little-endian f32s.
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, AnimationClipLoaderError> {
        let raw = self.take(n * 4)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

/// Parse `.animclip` bytes into an [`AnimationClip`]. Exposed for offline validation tools.
pub fn parse_animclip(bytes: &[u8]) -> Result<AnimationClip, AnimationClipLoaderError> {
    let mut c = Cur(bytes);
    if c.take(8)? != MAGIC {
        return Err(AnimationClipLoaderError::BadMagic);
    }
    let mut clip = AnimationClip::default();
    let target_count = c.u32()?;
    for _ in 0..target_count {
        // name-path -> AnimationTargetId
        let parts = c.u16()?;
        let mut path: Vec<Name> = Vec::with_capacity(parts);
        for _ in 0..parts {
            let len = c.u16()?;
            let s = core::str::from_utf8(c.take(len)?)
                .map_err(|_| AnimationClipLoaderError::BadUtf8)?;
            path.push(Name::new(s.to_string()));
        }
        let target = AnimationTargetId::from_names(path.iter());

        let mask = c.u8()?;
        if mask & CH_T != 0 {
            let (times, vals) = read_channel(&mut c, 3)?;
            let pts: Vec<_> = vals
                .chunks_exact(3)
                .map(|v| Vec3::new(v[0], v[1], v[2]).to_precision())
                .collect();
            if let Some(vc) = translation_curve(&times, pts) {
                clip.add_variable_curve_to_target(target, vc);
            }
        }
        if mask & CH_R != 0 {
            let (times, vals) = read_channel(&mut c, 4)?;
            let pts: Vec<_> = vals
                .chunks_exact(4)
                .map(|v| Quat::from_array([v[0], v[1], v[2], v[3]]).to_precision())
                .collect();
            if let Some(vc) = rotation_curve(&times, pts) {
                clip.add_variable_curve_to_target(target, vc);
            }
        }
        if mask & CH_S != 0 {
            let (times, vals) = read_channel(&mut c, 3)?;
            let pts: Vec<_> = vals
                .chunks_exact(3)
                .map(|v| Vec3::new(v[0], v[1], v[2]).to_precision())
                .collect();
            if let Some(vc) = scale_curve(&times, pts) {
                clip.add_variable_curve_to_target(target, vc);
            }
        }
    }
    Ok(clip)
}

/// Read one channel's `(times, flat values)` (values = key_count × `dim` f32s).
fn read_channel(c: &mut Cur, dim: usize) -> Result<(Vec<f32>, Vec<f32>), AnimationClipLoaderError> {
    let keys = c.u32()?;
    let times = c.f32s(keys)?;
    let values = c.f32s(keys * dim)?;
    Ok((times, values))
}

// One builder per field: `animated_field!` yields a distinct property type per T/R/S, and a single
// keyframe collapses to a `ConstantCurve` (matches bevy_gltf). LINEAR uses `UnevenSampleAutoCurve`,
// which slerps for the quaternion field via `TQuat`'s `Animatable` impl.
macro_rules! trs_curve {
    ($fn:ident, $field:ident, $ty:ty) => {
        fn $fn(times: &[f32], pts: Vec<$ty>) -> Option<VariableCurve> {
            if pts.is_empty() {
                return None;
            }
            if pts.len() == 1 {
                return Some(VariableCurve::new(AnimatableCurve::new(
                    animated_field!(Transform::$field),
                    ConstantCurve::new(Interval::EVERYWHERE, pts[0]),
                )));
            }
            UnevenSampleAutoCurve::new(times.iter().copied().zip(pts))
                .ok()
                .map(|curve| {
                    VariableCurve::new(AnimatableCurve::new(
                        animated_field!(Transform::$field),
                        curve,
                    ))
                })
        }
    };
}

// `TVec3`/`TQuat` are the precision aliases (`Vec3`/`Quat`, or `DVec3`/`DQuat` under the f64
// feature). Using them — and `.to_precision()` on the values — keeps this compiling at both
// `Transform` precisions.
trs_curve!(translation_curve, translation, bevy_math::TVec3);
trs_curve!(rotation_curve, rotation, bevy_math::TQuat);
trs_curve!(scale_curve, scale, bevy_math::TVec3);
