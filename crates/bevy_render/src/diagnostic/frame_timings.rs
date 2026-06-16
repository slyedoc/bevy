//! Per-frame CPU schedule-timing diagnostics.
//!
//! [`FrameTimeBreakdownPlugin`] records the wall time of the three CPU stages a
//! frame passes through, as `f64` milliseconds in the main-world
//! [`DiagnosticsStore`](bevy_diagnostic::DiagnosticsStore):
//!
//! - [`MAIN_CPU`] — the main-world schedule (`First`→`Last`), i.e. app/game
//!   logic.
//! - [`EXTRACT_CPU`] — the full extract step. Under pipelined rendering this is
//!   `renderer_extract`: waiting to receive the render world back from the
//!   render thread, entity sync, the `ExtractSchedule`, and the hand-off back
//!   (the "RenderExtractApp" span in Tracy). It all runs on the *main* thread,
//!   so it is part of the main thread's serial per-frame cost — and the recv
//!   wait is where a render-bound frame shows up as main-thread stall.
//! - [`RENDER_CPU`] — the render-world schedule (`Render`). Under pipelined
//!   rendering this runs on the render thread, in parallel with the *next*
//!   frame's main schedule.
//!
//! So the main thread's serial per-frame cost is `MAIN_CPU + EXTRACT_CPU`, while
//! `RENDER_CPU` overlaps it. Together with the per-pass GPU timings from
//! [`RenderDiagnosticsPlugin`](super::RenderDiagnosticsPlugin) and `frame_time`,
//! these decompose where a frame goes without needing a Tracy capture — e.g.
//! for a live perf overlay.
//!
//! The extract/render values are produced in the render world and read back in
//! the main world through a shared atomic pair, mirroring how
//! `RenderDiagnosticsMutex` hands GPU timings back. Under pipelined rendering
//! they lag one frame, which is fine for a live readout.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};

use bevy_app::{App, First, Last, Plugin, PreUpdate};
use bevy_diagnostic::{Diagnostic, DiagnosticPath, Diagnostics, RegisterDiagnostic};
use bevy_ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Res, ResMut},
};
use bevy_platform::time::Instant;

use crate::{pipelined_rendering::RenderExtractApp, Render, RenderApp, RenderSystems};

/// Main-world schedule wall time (`First`→`Last`), in milliseconds.
pub const MAIN_CPU: DiagnosticPath = DiagnosticPath::const_new("cpu/main");

/// Extract-step wall time (entity sync + `ExtractSchedule`), in milliseconds.
/// Runs on the main thread, so it adds to the main thread's serial frame cost.
pub const EXTRACT_CPU: DiagnosticPath = DiagnosticPath::const_new("cpu/extract");

/// Render-world schedule (`Render`) wall time, in milliseconds. Runs on the
/// render thread under pipelined rendering (parallel to the next main frame).
pub const RENDER_CPU: DiagnosticPath = DiagnosticPath::const_new("cpu/render");

/// Microsecond stage timings written by the render world and read back by the
/// main world's [`sync_frame_timings`]. Stored as `u64` micros in atomics so the
/// render thread can write them without locking; one-frame lag under pipelining.
#[derive(Resource, Clone, Default)]
struct FrameTimings(Arc<FrameTimingsInner>);

#[derive(Default)]
struct FrameTimingsInner {
    extract_us: AtomicU64,
    render_us: AtomicU64,
}

/// Start instant of the current main-world schedule, set in `First`.
#[derive(Resource)]
struct MainCpuStart(Instant);

/// Start instant of the current render-world schedule, set before
/// [`RenderSystems::ExtractCommands`].
#[derive(Resource)]
struct RenderCpuStart(Instant);

/// Records `cpu/main`, `cpu/extract` and `cpu/render` frame-timing diagnostics.
/// See the [module docs](self) for what each stage means.
#[derive(Default)]
pub struct FrameTimeBreakdownPlugin;

impl Plugin for FrameTimeBreakdownPlugin {
    fn build(&self, app: &mut App) {
        let timings = FrameTimings::default();
        app.register_diagnostic(Diagnostic::new(MAIN_CPU).with_suffix("ms"))
            .register_diagnostic(Diagnostic::new(EXTRACT_CPU).with_suffix("ms"))
            .register_diagnostic(Diagnostic::new(RENDER_CPU).with_suffix("ms"))
            .insert_resource(timings.clone())
            .insert_resource(MainCpuStart(Instant::now()))
            .add_systems(First, record_main_cpu_start)
            .add_systems(Last, record_main_cpu_end)
            .add_systems(PreUpdate, sync_frame_timings);

        // The extract/render stages run in the render world; give it the shared
        // handle so the bracket systems (added in `finish`) can write to it.
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(timings);
        }
    }

    fn finish(&self, app: &mut App) {
        // Bracket the whole render-world schedule (runs on the render thread
        // under pipelined rendering). `ExtractCommands`/`PostCleanup` are its
        // first/last sets, so this captures the full CPU cost of the frame's
        // render work. Done in a scope so the `RenderApp` borrow ends before we
        // reach for the extract sub-app below.
        let timings = {
            let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
                return;
            };
            render_app
                .insert_resource(RenderCpuStart(Instant::now()))
                .add_systems(
                    Render,
                    (
                        record_render_cpu_start.before(RenderSystems::ExtractCommands),
                        record_render_cpu_end.after(RenderSystems::PostCleanup),
                    ),
                );
            render_app.world().resource::<FrameTimings>().clone()
        };

        // Time the extract step by wrapping its extract fn — the pattern
        // documented on `SubApp::take_extract`. Prefer `RenderExtractApp`, whose
        // extract (`renderer_extract`) is the *full* main-thread step: recv-wait
        // + entity sync + `ExtractSchedule` + hand-off. Without pipelined
        // rendering there's no `RenderExtractApp`, so fall back to `RenderApp`'s
        // own extract (entity sync + `ExtractSchedule`, no hand-off).
        let extract_app = match app.get_sub_app_mut(RenderExtractApp) {
            Some(extract_app) => extract_app,
            None => app
                .get_sub_app_mut(RenderApp)
                .expect("RenderApp present — checked above"),
        };
        let mut inner = extract_app.take_extract();
        extract_app.set_extract(move |main_world, render_world| {
            let start = Instant::now();
            if let Some(extract) = inner.as_mut() {
                extract(main_world, render_world);
            }
            timings
                .0
                .extract_us
                .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        });
    }
}

/// `First`: stamp the start of the main-world schedule.
fn record_main_cpu_start(mut start: ResMut<MainCpuStart>) {
    start.0 = Instant::now();
}

/// `Last`: the main-world schedule is finishing — record its elapsed time.
fn record_main_cpu_end(start: Res<MainCpuStart>, mut diagnostics: Diagnostics) {
    let elapsed_ms = start.0.elapsed().as_secs_f64() * 1000.0;
    diagnostics.add_measurement(&MAIN_CPU, || elapsed_ms);
}

/// `Render`, before `ExtractCommands`: stamp the start of the render schedule.
fn record_render_cpu_start(mut start: ResMut<RenderCpuStart>) {
    start.0 = Instant::now();
}

/// `Render`, after `PostCleanup`: record the render schedule's elapsed time for
/// the main world to pick up next frame.
fn record_render_cpu_end(start: Res<RenderCpuStart>, timings: Res<FrameTimings>) {
    timings
        .0
        .render_us
        .store(start.0.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// `PreUpdate` (main world): publish the render world's extract/render timings
/// from the shared atomics into the diagnostics store.
fn sync_frame_timings(timings: Res<FrameTimings>, mut diagnostics: Diagnostics) {
    let extract_ms = timings.0.extract_us.load(Ordering::Relaxed) as f64 / 1000.0;
    let render_ms = timings.0.render_us.load(Ordering::Relaxed) as f64 / 1000.0;
    diagnostics.add_measurement(&EXTRACT_CPU, || extract_ms);
    diagnostics.add_measurement(&RENDER_CPU, || render_ms);
}
