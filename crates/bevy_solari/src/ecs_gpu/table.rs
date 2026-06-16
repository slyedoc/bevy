//! [`gpu_table!`] — declare a component-indexed GPU table and its columns in one
//! place, instead of hand-wiring the resource, the slot lookup, each column's
//! `GpuColumnDesc`, and the plugin.
//!
//! The macro generates the per-table types and wiring; the slot machinery
//! ([`gpu_slot`](super::slot)) and column scatter ([`GpuColumn`](super::GpuColumn))
//! are generic infra it reuses. You hand-write only the **extract** (which
//! components map to which column values — irreducible domain logic), passed in
//! via `extract:`.
//!
//! What it generates for `gpu_table! { table T as TPlugin { ... } }`:
//! - the render-world delta-storage resource `T` (a `pub Vec<u32>` per column,
//!   `high_water`, `clear_deltas`, [`GpuTable`](super::GpuTable) +
//!   [`GpuSlotTable`](super::slot::GpuSlotTable) impls);
//! - a marker struct + [`GpuColumnDesc`](super::GpuColumnDesc) impl per column;
//! - `TPlugin`, which registers the columns' scatter, the main-world slot
//!   allocator + assign system + free observer (the `GpuSlot<T>` component is its
//!   own per-table type via the generic `GpuSlot<T>`), the high-water extract,
//!   the gated `extract`, and the `Cleanup` delta clear.

/// Declare a component-indexed GPU table. See the module docs.
#[macro_export]
macro_rules! gpu_table {
    (
        $(#[$meta:meta])*
        $vis:vis table $Table:ident as $Plugin:ident {
            members: $Members:ty,
            extract: $extract:expr,
            columns {
                $( $Col:ident => $field:ident : $Value:ty = $label:literal $(@ scene $scene:literal)? ),+ $(,)?
            } $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(bevy_ecs::prelude::Resource, Default)]
        $vis struct $Table {
            /// Slot high-water (extracted from the main-world allocator); bounds
            /// every column's buffer.
            high_water: u32,
            $(
                /// Raw `[slot, value-words…]` scatter delta for this column,
                /// filled by the table's extract, uploaded verbatim by the column.
                pub $field: Vec<u32>,
            )+
        }

        impl $Table {
            /// Clear this frame's column deltas after the scatter consumes them.
            pub fn clear_deltas(&mut self) {
                $( self.$field.clear(); )+
            }
        }

        impl $crate::ecs_gpu::GpuTable for $Table {
            #[inline]
            fn high_water(&self) -> u32 {
                self.high_water
            }
        }

        impl $crate::ecs_gpu::GpuSlotTable for $Table {
            type Members = $Members;
            #[inline]
            fn set_high_water(&mut self, high_water: u32) {
                self.high_water = high_water;
            }
        }

        $(
            #[doc = concat!("Column `", $label, "` of the `", stringify!($Table), "` table.")]
            $vis struct $Col;
            impl $crate::ecs_gpu::GpuColumnDesc for $Col {
                type Value = $Value;
                type Table = $Table;
                const LABEL: &'static str = $label;
                // Optional `@ scene N` → this column joins the shared scene-columns
                // bind group at binding N (the path tracer reads it).
                $( const SCENE_BINDING: Option<u32> = Some($scene); )?
                fn delta_records(table: &$Table) -> &[u32] {
                    &table.$field
                }
            }
        )+

        #[doc = concat!("Plugin for the `", stringify!($Table), "` GPU table: columns, slot index, extract, clear.")]
        $vis struct $Plugin;

        impl bevy_app::Plugin for $Plugin {
            fn build(&self, app: &mut bevy_app::App) {
                use bevy_ecs::schedule::IntoScheduleConfigs as _;

                // Column scatter pipelines (shared mechanism, one per column).
                app.add_plugins((
                    $( $crate::ecs_gpu::GpuColumnPlugin::<$Col>::default(), )+
                ));

                // Main world: the component-as-index slot allocator + assign +
                // free observer (no hashmap).
                app.init_resource::<$crate::ecs_gpu::GpuSlotAllocator<$Table>>();
                app.add_systems(
                    bevy_app::PostUpdate,
                    $crate::ecs_gpu::assign_gpu_slots::<$Table>,
                );
                app.add_observer($crate::ecs_gpu::free_gpu_slot::<$Table>);

                let Some(render_app) = app.get_sub_app_mut(bevy_render::RenderApp) else {
                    return;
                };
                render_app.init_resource::<$Table>();
                render_app.add_systems(
                    bevy_render::ExtractSchedule,
                    (
                        // Copy the main-world allocator high-water into the table
                        // so its columns size their buffers this frame.
                        |allocator: bevy_render::Extract<
                            bevy_ecs::system::Res<
                                $crate::ecs_gpu::GpuSlotAllocator<$Table>,
                            >,
                        >,
                         mut table: bevy_ecs::system::ResMut<$Table>| {
                            <$Table as $crate::ecs_gpu::GpuSlotTable>::set_high_water(
                                &mut table,
                                allocator.high_water(),
                            );
                        },
                        // Gate the delta extract until every column's scatter
                        // pipeline has compiled (cold-start delta-loss guard).
                        ($extract).run_if(
                            |$( $field: bevy_ecs::system::Res<$crate::ecs_gpu::GpuColumn<$Col>>, )+
                             __cache: bevy_ecs::system::Res<bevy_render::render_resource::PipelineCache>|
                                -> bool {
                                true $( && $field.scatter_pipeline_ready(&__cache) )+
                            }
                        ),
                    ),
                );
                render_app.add_systems(
                    bevy_render::renderer::RenderGraph,
                    (|mut __table: bevy_ecs::system::ResMut<$Table>| {
                        __table.clear_deltas();
                    })
                    .in_set($crate::SolariClusterSystems::Cleanup),
                );
            }
        }
    };
}
