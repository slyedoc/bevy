use bevy_ecs::{reflect::ReflectResource, resource::Resource, system::Res};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::extract_resource::ExtractResource;
use derive_more::Display;

/// Which integrator lights solari views. The ray-tracing-pipeline path is the
/// only integrator; the enum is retained as the selection point for future paths.
#[derive(Default, Reflect, Display, Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SolariLighting {
    /// Ray-tracing-pipeline (SBT, per-material closest-hit shaders) path.
    #[default]
    #[display("rt_pipeline")]
    RtPipeline,
}

/// Global render state for every solari view. A resource, not a per-camera
/// component: the integrator is an app-level choice, so all solari cameras render
/// the same way.
#[derive(Resource, ExtractResource, Reflect, Clone, Default, Debug, PartialEq)]
#[reflect(Resource, Default)]
pub struct SolariViewState {
    /// The integrator lighting the frame.
    pub lighting: SolariLighting,
}

impl SolariViewState {
    /// Whether the RT-pipeline path runs this frame.
    pub fn rt_pipeline_runs(&self) -> bool {
        self.lighting == SolariLighting::RtPipeline
    }
}

/// Run condition: the RT-pipeline path lights the frame.
pub fn rt_pipeline_enabled(state: Res<SolariViewState>) -> bool {
    state.rt_pipeline_runs()
}
