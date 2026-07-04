//! Deferred GPU destruction for resources referenced by device address —
//! invisible to wgpu's lifetime tracker, so they must outlive their last submit.

use core::any::Any;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use bevy_ecs::prelude::*;
use bevy_render::renderer::RenderQueue;

/// Render-world reaper: parks retired bundles until the GPU signals the
/// submit that last references them, then drops them. THE disposal path for
/// any buffer whose device address escaped to a raw-VK op.
#[derive(Resource, Default)]
pub struct GpuRetire {
    in_flight: Vec<(Arc<AtomicBool>, &'static str, Box<dyn Any + Send + Sync>)>,
}

impl GpuRetire {
    /// Park `bundle` until everything submitted SO FAR completes on the GPU.
    /// Call after the last submit that references the bundle's addresses.
    pub fn retire(
        &mut self,
        queue: &RenderQueue,
        label: &'static str,
        bundle: impl Any + Send + Sync,
    ) {
        let done = Arc::new(AtomicBool::new(false));
        let signal = done.clone();
        queue.on_submitted_work_done(move || signal.store(true, Ordering::Release));
        tracing::debug!("gpu_retire: park {label}");
        self.in_flight.push((done, label, Box::new(bundle)));
    }

    /// Drop every bundle whose covering submit the GPU signaled complete.
    pub fn reap(&mut self) {
        self.in_flight.retain(|(done, label, _)| {
            let done = done.load(Ordering::Acquire);
            if done {
                tracing::debug!("gpu_retire: release {label}");
            }
            !done
        });
    }

    /// Parked bundle count — leak telemetry.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }
}

/// Per-frame reap; completion flags make timing a latency concern, not safety.
pub fn reap_retired(mut retire: ResMut<GpuRetire>) {
    retire.reap();
}

/// Guard for a buffer whose device address escaped to a raw-VK op: the ONLY
/// sanctioned disposal is [`GpuRetire::retire_guarded`]. Dropping it armed
/// panics in debug builds (it's a GPU use-after-free) and errors in release.
pub struct Retirable<T: Any + Send + Sync> {
    inner: Option<T>,
    label: &'static str,
}

impl<T: Any + Send + Sync> Retirable<T> {
    pub fn new(label: &'static str, value: T) -> Self {
        Self { inner: Some(value), label }
    }

    pub fn get(&self) -> &T {
        self.inner.as_ref().expect("Retirable already retired")
    }
}

impl<T: Any + Send + Sync> core::ops::Deref for Retirable<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.get()
    }
}

impl<T: Any + Send + Sync> Drop for Retirable<T> {
    fn drop(&mut self) {
        if self.inner.is_some() {
            debug_assert!(
                false,
                "Retirable<{}> dropped without GpuRetire::retire_guarded — GPU use-after-free",
                self.label,
            );
            tracing::error!(
                "Retirable {} dropped without retire (GPU use-after-free risk)",
                self.label,
            );
        }
    }
}

impl GpuRetire {
    /// Disarm `guarded` and park its value — the guard's only disposal path.
    pub fn retire_guarded<T: Any + Send + Sync>(
        &mut self,
        queue: &RenderQueue,
        mut guarded: Retirable<T>,
    ) {
        let value = guarded.inner.take().expect("Retirable already retired");
        self.retire(queue, guarded.label, value);
    }
}
