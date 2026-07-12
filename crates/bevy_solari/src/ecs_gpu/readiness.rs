//! One readiness gate for every solari pipeline.
//!
//! Every pipeline solari queues registers itself here at creation, and
//! [`solari_pipelines_ready`] is THE cold-start run condition — all of them
//! compiled, or nothing that consumes their output runs. Never gate on a
//! hand-listed subset: diverged subset gates are the recurring startup
//! black-screen class (a producer's delta consumed while its consumer's
//! pipeline is still compiling — dropped one-shot writes, statics collapsed
//! at origin, black accumulations).
//!
//! # Readiness is three axes, and this gate only covers the first
//!
//! 1. **Pipelines compiled** — this registry.
//! 2. **Bind groups built from buffers that exist** — schedule placement.
//!    `RenderSystems::PrepareBindGroups` is a chained SUB-set *inside*
//!    `Prepare`; a system added bare to `Prepare` has NO ordering against it,
//!    so its bind-group builder can run first on some frames (per-run
//!    nondeterminism). Rule: solari never adds a system bare to
//!    `RenderSystems::Prepare` — buffer-writing prepares go in
//!    `PrepareResources`, bind-group builders in `PrepareBindGroups`. Bind
//!    groups are rebuilt every frame unless a buffer-identity signature is
//!    checked (see `bindings/bind_groups.rs`) — "stable handle today" is not
//!    a caching contract.
//! 3. **The producer actually ran this frame** — a dispatch consuming another
//!    pass's per-frame output (indirect args, freshly-scattered columns) must
//!    verify that pass recorded (e.g. `TransformFrontier::ran`), not merely
//!    that its own pipeline exists.
//!
//! # One-shot data must be retained until consumed, or fail loudly
//!
//! Any delta/queue/latch that is cleared each frame trusts its consumer to
//! have run that frame. If the consumer has ANY bail path, that trust is a
//! silent session-long loss (invisible instances, identity camera). Either:
//! - **retain-until-consumed**: clear only after the consumer records
//!   (`GpuColumn::pending`, `TransformPropagate::needs_full_rebuild`,
//!   `TransformSubtract::dirty`, the reconcile journal's `mark_folded`), or
//! - **re-arm a recovery latch** when consumption was missed
//!   (`dispatch_transform_propagate`'s full-rebuild re-arm), or
//! - **warn loudly** when dropping is the only option
//!   (`prepare_column`'s no-column tripwire, `clas_arena`'s DROPPING warns).

use bevy_ecs::{
    resource::Resource,
    system::{Local, Res},
};
use bevy_render::render_resource::{
    CachedComputePipelineId, CachedRenderPipelineId, PipelineCache,
};

/// Registry of every solari-created pipeline (label + cache id). Empty means
/// "not a solari device / startup hasn't run" and reads as NOT ready, so gated
/// systems stay dormant.
#[derive(Resource, Default)]
pub struct SolariPipelineRegistry {
    compute: Vec<(&'static str, CachedComputePipelineId)>,
    render: Vec<(&'static str, CachedRenderPipelineId)>,
}

impl SolariPipelineRegistry {
    pub fn register(&mut self, label: &'static str, id: CachedComputePipelineId) {
        self.compute.push((label, id));
    }

    pub fn register_render(&mut self, label: &'static str, id: CachedRenderPipelineId) {
        self.render.push((label, id));
    }

    pub fn ready(&self, cache: &PipelineCache) -> bool {
        !self.compute.is_empty()
            && self
                .compute
                .iter()
                .all(|(_, id)| cache.get_compute_pipeline(*id).is_some())
            && self
                .render
                .iter()
                .all(|(_, id)| cache.get_render_pipeline(*id).is_some())
    }

    /// Labels still compiling — the startup-wait diagnostic.
    pub fn missing<'a>(&'a self, cache: &'a PipelineCache) -> Vec<&'static str> {
        self.compute
            .iter()
            .filter(|(_, id)| cache.get_compute_pipeline(*id).is_none())
            .map(|(label, _)| *label)
            .chain(
                self.render
                    .iter()
                    .filter(|(_, id)| cache.get_render_pipeline(*id).is_none())
                    .map(|(label, _)| *label),
            )
            .collect()
    }
}

/// THE solari cold-start run condition: every registered pipeline compiled.
pub fn solari_pipelines_ready(
    registry: Option<Res<SolariPipelineRegistry>>,
    cache: Res<PipelineCache>,
) -> bool {
    registry.is_some_and(|registry| registry.ready(&cache))
}

/// `Render`: while the gate is closed, periodically name what it's waiting on —
/// a pipeline stuck in `Err` would otherwise read as a silent hang.
pub fn log_pipeline_wait(
    registry: Option<Res<SolariPipelineRegistry>>,
    cache: Res<PipelineCache>,
    mut frames: Local<u32>,
) {
    let Some(registry) = registry else { return };
    if registry.ready(&cache) {
        *frames = 0;
        return;
    }
    *frames += 1;
    if *frames % 600 == 0 {
        bevy_log::warn!(
            "solari: pipelines still compiling after {} frames: {:?}",
            *frames,
            registry.missing(&cache)
        );
    }
}
