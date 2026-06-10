use bevy_ecs::{
    component::Component,
    reflect::ReflectComponent,
    system::{Query, Res}, template::FromTemplate,
};
use bevy_reflect::Reflect;
use bevy_render::{
    extract_component::ExtractComponent,
    renderer::CurrentView,
};
use derive_more::Display;

/// Unified debug visualization, selected as a camera component. Each view is
/// its own variant; the restir overlay maps the selected variant to a buffer +
/// how to display it. `Display` strings double as dropdown / label text.
///
/// `None` is the normal lit output. The DLSS-guide variants only exist with the
/// `dlss` feature. (Step 2 will fold in a `Pathtrace` reference view and the
/// cluster-debug views.)
#[derive(
    Component, Default, Reflect, FromTemplate, Display, Copy, Clone, Debug, PartialEq, Eq, Hash, ExtractComponent,
)]
#[reflect(Component)]
pub enum SolariOverlay {
    // TODO: will be Restir-pt
    #[display("none")]
    None,
    /// Reference path tracer (replaces restir for this view).
    #[default]
    #[display("pathtrace")]
    Pathtrace,
    /// First-hit cluster LOD level
    #[display("lod")]
    Lod,
    /// First-hit cluster id, hashed to a color.
    #[display("cluster")]
    Cluster,
    /// First-hit triangle id, hashed to a color.
    #[display("triangle")]
    Triangle,
    /// BLAS-sharing correctness (green = correct, red = mis-bucketed).
    #[display("geometry check")]
    GeometryCheck,
    #[display("world position")]
    WorldPosition,
    #[display("material id")]
    MaterialId,
    #[display("world normal")]
    WorldNormal,
    #[display("uv")]
    Uv,
    #[display("motion vectors")]
    MotionVectors,
    #[cfg(feature = "dlss")]
    #[display("dlss depth")]
    DlssDepth,
    #[cfg(feature = "dlss")]
    #[display("dlss normal+roughness")]
    DlssNormalRoughness,
    #[cfg(feature = "dlss")]
    #[display("dlss diffuse albedo")]
    DlssDiffuseAlbedo,
    #[cfg(feature = "dlss")]
    #[display("dlss specular albedo")]
    DlssSpecularAlbedo,
    #[cfg(feature = "dlss")]
    #[display("dlss specular motion")]
    DlssSpecularMotion,
}

impl SolariOverlay {
    /// All views, in cycle order (the in-engine UI cycles this list). Adding a
    /// view: a variant + an entry here + an `overlay_view!` marker in the restir
    /// overlay + a `#ifdef VIEW_*` block in `debug_overlay.wgsl`.
    pub const ALL: &'static [SolariOverlay] = &[
        Self::None,
        Self::Pathtrace,
        Self::Lod,
        Self::Cluster,
        Self::Triangle,
        Self::GeometryCheck,
        Self::WorldPosition,
        Self::MaterialId,
        Self::WorldNormal,
        Self::Uv,
        Self::MotionVectors,
        #[cfg(feature = "dlss")]
        Self::DlssDepth,
        #[cfg(feature = "dlss")]
        Self::DlssNormalRoughness,
        #[cfg(feature = "dlss")]
        Self::DlssDiffuseAlbedo,
        #[cfg(feature = "dlss")]
        Self::DlssSpecularAlbedo,
        #[cfg(feature = "dlss")]
        Self::DlssSpecularMotion,
    ];

    /// Next view in [`Self::ALL`] order (wraps).
    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|m| *m == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    /// Previous view in [`Self::ALL`] order (wraps).
    pub fn prev(self) -> Self {
        let i = Self::ALL.iter().position(|m| *m == self).unwrap_or(0);
        Self::ALL[(i + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

/// Run condition: true when the **current view's** [`SolariDebugView`] equals
/// `target`.
///
/// The `Core3d` schedule runs once per camera with the [`CurrentView`] resource
/// set (see `bevy_core_pipeline::schedule`), so a render system can be gated to
/// a single debug view at registration — `system.run_if(view_is(X))` — instead
/// of an in-body early-return. Returns `false` when no current view / the view
/// lacks the component.
pub fn overlay_is(
    target: SolariOverlay,
) -> impl Fn(Option<Res<CurrentView>>, Query<&SolariOverlay>) -> bool + Clone {
    move |current, views| {
        current.is_some_and(|current| views.get(current.0).copied() == Ok(target))
    }
}
