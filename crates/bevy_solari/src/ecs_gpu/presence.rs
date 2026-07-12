//! Presence columns — a per-slot `u32` flag column tracking whether a marker
//! component `Marker` is present on a slot table's member entities, maintained by
//! **observers** (zero per-frame cost).
//!
//! Use for render-only classification flags that are set post-spawn and (almost)
//! never change — e.g. [`TransformStatic`](crate::transform::TransformStatic),
//! whose presence routes an instance to a spatial PTLAS partition vs the global
//! one. The marker can't be sourced through a table's extract: an `Added<Marker>`
//! query re-scans the whole archetype every frame (the exact cost markers like
//! `TransformStatic` exist to avoid), and the transform extract deliberately
//! *excludes* such markers at archetype granularity. Observers fire only on the
//! actual add/remove, so steady state does no work.
//!
//! The flag defaults to `0` for any slot whose entity never carried `Marker` (the
//! sparse column buffer zero-inits on growth). `Add<Marker>` writes `1`,
//! `Remove<Marker>` writes `0`.
//!
//! A presence column reuses the full [`GpuColumn`] scatter machinery via the
//! [`Presence<C>`] wrapper, which carries the [`GpuColumnDesc`] impl. The wrapper
//! exists so that blanket impl can't overlap the concrete column descs that
//! [`gpu_table!`](crate::gpu_table) generates. Consumers read the flag buffer via
//! `Res<GpuColumn<Presence<C>>>`, indexed by `GpuSlot<C::SlotTable>`.

use core::marker::PhantomData;

use bevy_app::{App, Plugin};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    lifecycle::{Add, Remove},
    observer::On,
    resource::Resource,
    schedule::IntoScheduleConfigs as _,
    system::{Res, ResMut, SystemState},
};
use bevy_render::{
    render_resource::PipelineCache, ExtractSchedule, MainWorld, RenderApp,
};

use crate::SolariClusterSystems;

use super::{
    column::{GpuColumnDesc, GpuColumnPlugin, GpuTable},
    slot::{push_record, GpuSlot, GpuSlotAllocator, GpuSlotTable},
};

/// Describes a presence column: which slot table indexes it (and bounds its size),
/// which marker drives the flag, and its debug label. Consume the flag buffer via
/// `Res<GpuColumn<Presence<Self>>>`.
pub trait GpuPresenceColumn: Send + Sync + 'static {
    /// The slot table whose `GpuSlot<SlotTable>` indexes this column; its
    /// allocator high-water sizes the flag buffer so every slot is in-bounds
    /// (an unmarked slot reads the `0` default).
    type SlotTable: GpuSlotTable;
    /// The marker whose presence the flag tracks (`1` while present, else `0`).
    type Marker: Component;
    /// Debug label for the GPU buffer + pipeline.
    const LABEL: &'static str;
}

/// Adapts a [`GpuPresenceColumn`] into a [`GpuColumnDesc`] so it can reuse the
/// standard column scatter. A distinct type so the blanket impl below doesn't
/// overlap the concrete descs `gpu_table!` generates.
pub struct Presence<C: GpuPresenceColumn>(PhantomData<fn() -> C>);

/// Render-world delta storage + slot high-water for a presence column — its own
/// [`GpuTable`] so it drops straight into the column scatter. Filled by
/// [`drain_presence`] from observer events; cleared each frame in `Cleanup`.
#[derive(Resource)]
pub struct PresenceTable<C: GpuPresenceColumn> {
    high_water: u32,
    delta: Vec<u32>,
    _marker: PhantomData<fn() -> C>,
}

impl<C: GpuPresenceColumn> Default for PresenceTable<C> {
    fn default() -> Self {
        Self {
            high_water: 0,
            delta: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<C: GpuPresenceColumn> GpuTable for PresenceTable<C> {
    #[inline]
    fn high_water(&self) -> u32 {
        self.high_water
    }
}

impl<C: GpuPresenceColumn> GpuColumnDesc for Presence<C> {
    type Value = u32;
    type Table = PresenceTable<C>;
    const LABEL: &'static str = C::LABEL;
    fn delta_records(table: &PresenceTable<C>) -> &[u32] {
        &table.delta
    }
}

/// Main-world `(entity, present)` events accumulated by the marker observers,
/// drained each frame by [`drain_presence`]. An entity's slot isn't resolved here
/// (it may not be assigned yet at observe time) — the drain resolves it later.
#[derive(Resource)]
pub struct PresenceEvents<C: GpuPresenceColumn> {
    events: Vec<(Entity, bool)>,
    _marker: PhantomData<fn() -> C>,
}

impl<C: GpuPresenceColumn> Default for PresenceEvents<C> {
    fn default() -> Self {
        Self {
            events: Vec::new(),
            _marker: PhantomData,
        }
    }
}

/// Observer: the marker was added → flag the entity present.
fn on_add_marker<C: GpuPresenceColumn>(
    add: On<Add, C::Marker>,
    mut events: ResMut<PresenceEvents<C>>,
) {
    events.events.push((add.entity, true));
}

/// Observer: the marker was removed → flag the entity absent.
fn on_remove_marker<C: GpuPresenceColumn>(
    remove: On<Remove, C::Marker>,
    mut events: ResMut<PresenceEvents<C>>,
) {
    events.events.push((remove.entity, false));
}

/// `ExtractSchedule`: mirror the slot high-water (always — so the flag buffer
/// covers every slot and a consumer's `flags[slot]` read is never an uncommitted
/// page, just the `0` default), and drain the observer events into the column's
/// delta once the scatter pipeline has compiled. Until then events accumulate
/// (no delta loss), exactly like the cold-start guard in `prepare_column`.
///
/// Resolves each entity's `GpuSlot<C::SlotTable>` (assigned in `PostUpdate`, so
/// present by now). Mirrors the `flush_cluster_instances` `MainWorld` +
/// `SystemState` drain pattern.
fn drain_presence<C: GpuPresenceColumn>(
    mut main_world: ResMut<MainWorld>,
    mut table: ResMut<PresenceTable<C>>,
    registry: Option<Res<super::SolariPipelineRegistry>>,
    cache: Res<PipelineCache>,
    #[allow(clippy::type_complexity)] mut state: bevy_ecs::system::Local<
        Option<
            SystemState<(
                ResMut<'static, PresenceEvents<C>>,
                bevy_ecs::system::Query<'static, 'static, &'static GpuSlot<C::SlotTable>>,
                Res<'static, GpuSlotAllocator<C::SlotTable>>,
            )>,
        >,
    >,
) {
    if state.is_none() {
        *state = Some(SystemState::new(&mut main_world));
    }
    let state = state.as_mut().unwrap();
    let Ok((mut events, slots, allocator)) = state.get_mut(&mut main_world) else {
        return;
    };
    // Cover every handed-out slot so a read at any slot is in-bounds (0 default).
    table.high_water = allocator.high_water();
    // Hold events behind THE readiness gate, then drain them all — the consumers
    // of these flags (PTLAS placement) must also be live, not just this scatter.
    if !registry.is_some_and(|r| r.ready(&cache)) {
        return;
    }
    for (entity, present) in events.events.drain(..) {
        if let Ok(slot) = slots.get(entity) {
            push_record(&mut table.delta, slot.index(), present as u32);
        }
    }
}

/// `Cleanup`: clear this frame's presence delta after the scatter consumed it.
fn clear_presence<C: GpuPresenceColumn>(mut table: ResMut<PresenceTable<C>>) {
    table.delta.clear();
}

/// Registers presence column `C`: the GPU flag buffer + scatter (via
/// [`GpuColumnPlugin`]), the main-world events resource + marker observers, and
/// the render-world delta table + drain + clear.
pub struct GpuPresenceColumnPlugin<C: GpuPresenceColumn>(PhantomData<fn() -> C>);

impl<C: GpuPresenceColumn> Default for GpuPresenceColumnPlugin<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C: GpuPresenceColumn> Plugin for GpuPresenceColumnPlugin<C> {
    fn build(&self, app: &mut App) {
        // Main world: observer-fed event set (zero per-frame cost).
        app.init_resource::<PresenceEvents<C>>()
            .add_observer(on_add_marker::<C>)
            .add_observer(on_remove_marker::<C>);

        // The flag buffer + scatter pipeline reuse the standard column machinery.
        app.add_plugins(GpuColumnPlugin::<Presence<C>>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<PresenceTable<C>>()
            .add_systems(ExtractSchedule, drain_presence::<C>)
            .add_systems(
                bevy_render::renderer::RenderGraph,
                clear_presence::<C>.in_set(SolariClusterSystems::Cleanup),
            );
    }
}
