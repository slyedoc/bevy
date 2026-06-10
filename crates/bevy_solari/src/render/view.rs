use bevy_ecs::{reflect::ReflectResource, resource::Resource, system::Res};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::extract_resource::ExtractResource;
use derive_more::Display;

/// Which integrator lights solari views.
#[derive(Default, Reflect, Display, Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SolariLighting {
    /// Reference path tracer: unbiased, progressive accumulation. The ground
    /// truth the realtime path is validated against.
    #[default]
    #[display("pathtrace")]
    Pathtracer,
    /// Realtime ReSTIR path.    
    #[display("restir")]
    Restir,
}

/// Debug visualization, selected in [`SolariViewState`]. The overlay overwrites
/// the lit output with the chosen view (see `render::overlay`); `Display`
/// strings double as dropdown / label text.
#[derive(Reflect, Display, Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SolariDebugView {
    /// First-hit cluster LOD level.
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
    /// DI reservoir contribution weight (heatmap; NaN = magenta).
    #[display("di weight")]
    DiWeight,
    /// DI reservoir confidence (M) over its cap.
    #[display("di confidence")]
    DiConfidence,
    /// DI reservoir light identity, hashed to a color.
    #[display("di light")]
    DiLight,
    /// ReGIR cell of each surface point, hashed to a color (red = cold).
    #[display("regir cells")]
    RegirCells,
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

impl SolariDebugView {
    /// Every view, in menu order. Adding a view: a variant + an entry here + an
    /// `overlay_view!` marker in `render::overlay` + a `#ifdef VIEW_*` block in
    /// `debug_overlay.wgsl`.
    pub const ALL: &'static [SolariDebugView] = &[
        Self::Lod,
        Self::Cluster,
        Self::Triangle,
        Self::GeometryCheck,
        Self::WorldPosition,
        Self::MaterialId,
        Self::WorldNormal,
        Self::Uv,
        Self::MotionVectors,
        Self::DiWeight,
        Self::DiConfidence,
        Self::DiLight,
        Self::RegirCells,
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

    /// The `restir_debug` shader mode rendering this view, if it's one of the
    /// reservoir/grid visualizations the restir node draws itself (they read
    /// the reservoir + ReGIR buffers, which only the restir bind group sees).
    pub fn restir_debug_mode(self) -> Option<u32> {
        match self {
            Self::DiWeight => Some(1),
            Self::DiConfidence => Some(2),
            Self::DiLight => Some(3),
            Self::RegirCells => Some(4),
            _ => None,
        }
    }

    /// Whether this view reads the restir chain's output (its G-buffer or the
    /// DLSS guide textures derived from it), so the chain must run to produce
    /// it. Cluster-family views trace their own primary ray instead and need
    /// no integrator at all.
    pub fn needs_restir(self) -> bool {
        match self {
            Self::Lod | Self::Cluster | Self::Triangle | Self::GeometryCheck => false,
            Self::WorldPosition
            | Self::MaterialId
            | Self::WorldNormal
            | Self::Uv
            | Self::MotionVectors
            | Self::DiWeight
            | Self::DiConfidence
            | Self::DiLight
            | Self::RegirCells => true,
            #[cfg(feature = "dlss")]
            Self::DlssDepth
            | Self::DlssNormalRoughness
            | Self::DlssDiffuseAlbedo
            | Self::DlssSpecularAlbedo
            | Self::DlssSpecularMotion => true,
        }
    }
}

/// Global render state for every solari view: which integrator lights the
/// frame, and an optional debug visualization overwriting the output.
///
/// A resource, not a per-camera component: the integrator is an app-level
/// choice and the debug views are dev tooling, so all solari cameras render
/// the same way (cf. bevy's `UiDebugOptions` / gizmo configs). A debug view
/// fully overwrites the output, so selecting one skips whatever the view
/// doesn't read: cluster-family views trace their own primary ray and run
/// **no** integrator; buffer views keep the restir chain alive for its
/// G-buffer.
#[derive(Resource, ExtractResource, Reflect, Clone, Default, Debug, PartialEq)]
#[reflect(Resource, Default)]
pub struct SolariViewState {
    /// The integrator lighting the frame when no debug view is selected.
    pub lighting: SolariLighting,
    /// Debug visualization overwriting the lit output, if any.
    pub debug: Option<SolariDebugView>,
}

impl SolariViewState {
    /// Whether the restir chain runs this frame: it's the selected integrator
    /// with nothing overwriting its output, or the selected debug view samples
    /// its G-buffer.
    pub fn restir_runs(&self) -> bool {
        match self.debug {
            Some(view) => view.needs_restir(),
            None => self.lighting == SolariLighting::Restir,
        }
    }

    /// Whether the reference path tracer runs this frame: it's the selected
    /// integrator and no debug view overwrites its output.
    pub fn pathtracer_runs(&self) -> bool {
        self.debug.is_none() && self.lighting == SolariLighting::Pathtracer
    }
}

/// Run condition: the restir chain produces output this frame (lit color or
/// the G-buffer a debug view / the gizmo depth bridge reads).
pub fn restir_enabled(state: Res<SolariViewState>) -> bool {
    state.restir_runs()
}

/// Run condition: the reference path tracer lights the frame.
pub fn pathtracer_enabled(state: Res<SolariViewState>) -> bool {
    state.pathtracer_runs()
}

/// Run condition: `target` is the selected debug view. Gates each
/// `overlay::<V>` system at registration instead of an in-body early-return.
pub fn debug_is(target: SolariDebugView) -> impl Fn(Res<SolariViewState>) -> bool + Clone {
    move |state| state.debug == Some(target)
}
