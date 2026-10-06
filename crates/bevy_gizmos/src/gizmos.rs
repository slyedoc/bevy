//! A module for the [`Gizmos`] [`SystemParam`].

use core::{
    iter,
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
};

use bevy_color::{Color, LinearRgba};
use bevy_ecs::{
    change_detection::Tick,
    entity::Entity,
    resource::Resource,
    system::{
        Deferred, ReadOnlySystemParam, Res, SystemAccess, SystemBuffer, SystemMeta, SystemParam,
        SystemParamAccessConflict, SystemParamValidationError,
    },
    world::{unsafe_world_cell::UnsafeWorldCell, DeferredWorld, World},
};
use bevy_math::{Isometry2d, Isometry3d, Vec2, Vec3};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_shape::Aabb3d;
use bevy_transform::TransformPoint;
use bevy_utils::default;

use crate::{
    config::{DefaultGizmoConfigGroup, GizmoConfigGroup, GizmoConfigStore},
    prelude::GizmoConfig,
};

/// Storage of gizmo primitives.
#[derive(Resource)]
pub struct GizmoStorage<Config, Clear> {
    pub(crate) list_positions: Vec<Vec3>,
    pub(crate) list_colors: Vec<LinearRgba>,
    pub(crate) list_owners: OwnerRuns,
    pub(crate) strip_positions: Vec<Vec3>,
    pub(crate) strip_colors: Vec<LinearRgba>,
    pub(crate) strip_owners: OwnerRuns,
    marker: PhantomData<(Config, Clear)>,
}

/// The entity each stretch of a gizmo vertex stream is drawn for, as `(first vertex, owner)` in
/// order. Vertices before the first run have no owner.
///
/// A renderer treats an owned line like anything else its owner renders: it shows in the owner's
/// render layers and world, and not while the owner is hidden or gone. Unowned lines follow their
/// config group.
#[derive(Debug, Clone, Default, PartialEq, Eq, Reflect)]
#[reflect(Default, Clone)]
pub struct OwnerRuns(pub Vec<(u32, Option<Entity>)>);

impl OwnerRuns {
    /// From vertex `at` on, lines belong to `owner`.
    pub fn mark(&mut self, at: usize, owner: Option<Entity>) {
        let at = at as u32;
        if let Some((start, last)) = self.0.last_mut()
            && *start == at
        {
            *last = owner;
            // Overwriting may make it repeat the run before it.
            let len = self.0.len();
            if (len >= 2 && self.0[len - 2].1 == owner) || (len == 1 && owner.is_none()) {
                self.0.pop();
            }
            return;
        }
        if self.owner_at_end() != owner {
            self.0.push((at, owner));
        }
    }

    /// The owner the stream ends in.
    pub fn owner_at_end(&self) -> Option<Entity> {
        self.0.last().and_then(|(_, owner)| *owner)
    }

    /// Append `other`'s runs, for a stream appended at vertex `offset`.
    pub fn append(&mut self, other: &OwnerRuns, offset: usize) {
        self.mark(offset, None);
        for &(start, owner) in &other.0 {
            self.mark(offset + start as usize, owner);
        }
    }

    /// Clear all runs.
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// `(first vertex, end vertex, owner)` for a stream of `len` vertices, covering it all.
    pub fn ranges(&self, len: usize) -> impl Iterator<Item = (usize, usize, Option<Entity>)> + '_ {
        let first = self.0.first().map_or(len, |(start, _)| (*start as usize).min(len));
        let lead = (first > 0).then_some((0, first, None));
        let runs = self.0.iter().enumerate().map(move |(i, &(start, owner))| {
            let end = self.0.get(i + 1).map_or(len, |(next, _)| *next as usize);
            ((start as usize).min(len), end.min(len), owner)
        });
        lead.into_iter()
            .chain(runs)
            .filter(|(start, end, _)| start < end)
    }
}

impl<Config, Clear> Default for GizmoStorage<Config, Clear> {
    fn default() -> Self {
        Self {
            list_positions: default(),
            list_colors: default(),
            list_owners: default(),
            strip_positions: default(),
            strip_colors: default(),
            strip_owners: default(),
            marker: PhantomData,
        }
    }
}

impl<Config, Clear> GizmoStorage<Config, Clear>
where
    Config: GizmoConfigGroup,
    Clear: 'static + Send + Sync,
{
    /// Combine the other gizmo storage with this one.
    pub fn append_storage<OtherConfig, OtherClear>(
        &mut self,
        other: &GizmoStorage<OtherConfig, OtherClear>,
    ) {
        self.list_owners
            .append(&other.list_owners, self.list_positions.len());
        self.strip_owners
            .append(&other.strip_owners, self.strip_positions.len());
        self.list_positions.extend(other.list_positions.iter());
        self.list_colors.extend(other.list_colors.iter());
        self.strip_positions.extend(other.strip_positions.iter());
        self.strip_colors.extend(other.strip_colors.iter());
    }

    pub(crate) fn swap<OtherConfig, OtherClear>(
        &mut self,
        other: &mut GizmoStorage<OtherConfig, OtherClear>,
    ) {
        mem::swap(&mut self.list_positions, &mut other.list_positions);
        mem::swap(&mut self.list_colors, &mut other.list_colors);
        mem::swap(&mut self.strip_positions, &mut other.strip_positions);
        mem::swap(&mut self.strip_colors, &mut other.strip_colors);
        mem::swap(&mut self.list_owners, &mut other.list_owners);
        mem::swap(&mut self.strip_owners, &mut other.strip_owners);
    }

    /// Clear this gizmo storage of any requested gizmos.
    pub fn clear(&mut self) {
        self.list_positions.clear();
        self.list_colors.clear();
        self.list_owners.clear();
        self.strip_positions.clear();
        self.strip_colors.clear();
        self.strip_owners.clear();
    }
}

/// Swap buffer for a specific clearing context.
///
/// This is to stash/store the default/requested gizmos so another context can
/// be substituted for that duration.
pub struct Swap<Clear>(PhantomData<Clear>);

/// A [`SystemParam`] for drawing gizmos.
///
/// They are drawn in immediate mode, which means they will be rendered only for
/// the frames, or ticks when in [`FixedMain`](bevy_app::FixedMain), in which
/// they are spawned.
///
/// A system in [`Main`](bevy_app::Main) will be cleared each rendering
/// frame, while a system in [`FixedMain`](bevy_app::FixedMain) will be
/// cleared each time the [`RunFixedMainLoop`](bevy_app::RunFixedMainLoop)
/// schedule is run.
///
/// Gizmos should be spawned before the [`Last`](bevy_app::Last) schedule
/// to ensure they are drawn.
///
/// To set up your own clearing context (useful for custom scheduling similar
/// to [`FixedMain`](bevy_app::FixedMain)):
///
/// ```
/// use bevy_gizmos::{prelude::*, *, gizmos::GizmoStorage};
/// # use bevy_app::prelude::*;
/// # use bevy_ecs::{schedule::ScheduleLabel, prelude::*};
/// # #[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
/// # struct StartOfMyContext;
/// # #[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
/// # struct EndOfMyContext;
/// # #[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
/// # struct StartOfRun;
/// # #[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
/// # struct EndOfRun;
/// # struct MyContext;
/// struct ClearContextSetup;
/// impl Plugin for ClearContextSetup {
///     fn build(&self, app: &mut App) {
///         app.init_resource::<GizmoStorage<DefaultGizmoConfigGroup, MyContext>>()
///            // Make sure this context starts/ends cleanly if inside another context. E.g. it
///            // should start after the parent context starts and end after the parent context ends.
///            .add_systems(StartOfMyContext, start_gizmo_context::<DefaultGizmoConfigGroup, MyContext>)
///            // If not running multiple times, put this with [`start_gizmo_context`].
///            .add_systems(StartOfRun, clear_gizmo_context::<DefaultGizmoConfigGroup, MyContext>)
///            // If not running multiple times, put this with [`end_gizmo_context`].
///            .add_systems(EndOfRun, collect_requested_gizmos::<DefaultGizmoConfigGroup, MyContext>)
///            .add_systems(EndOfMyContext, end_gizmo_context::<DefaultGizmoConfigGroup, MyContext>)
///            .add_systems(
///                Last,
///                propagate_gizmos::<DefaultGizmoConfigGroup, MyContext>.before(GizmoMeshSystems),
///            );
///     }
/// }
/// ```
pub struct Gizmos<'w, 's, Config = DefaultGizmoConfigGroup, Clear = ()>
where
    Config: GizmoConfigGroup,
    Clear: 'static + Send + Sync,
{
    buffer: Deferred<'s, GizmoBuffer<Config, Clear>>,
    /// The currently used [`GizmoConfig`]
    pub config: &'w GizmoConfig,
    /// The currently used [`GizmoConfigGroup`]
    pub config_ext: &'w Config,
}

impl<'w, 's, Config, Clear> Gizmos<'w, 's, Config, Clear>
where
    Config: GizmoConfigGroup,
    Clear: 'static + Send + Sync,
{
    /// Draw with `draw` for `entity`, then go back to the owner before. See
    /// [`GizmoBuffer::set_owner`].
    pub fn for_entity<R>(&mut self, entity: Entity, draw: impl FnOnce(&mut Self) -> R) -> R {
        let before = self.owner();
        self.set_owner(Some(entity));
        let out = draw(self);
        self.set_owner(before);
        out
    }
}

impl<'w, 's, Config, Clear> Deref for Gizmos<'w, 's, Config, Clear>
where
    Config: GizmoConfigGroup,
    Clear: 'static + Send + Sync,
{
    type Target = GizmoBuffer<Config, Clear>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<'w, 's, Config, Clear> DerefMut for Gizmos<'w, 's, Config, Clear>
where
    Config: GizmoConfigGroup,
    Clear: 'static + Send + Sync,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

type GizmosState<Config, Clear> = (
    Deferred<'static, GizmoBuffer<Config, Clear>>,
    Res<'static, GizmoConfigStore>,
);
#[doc(hidden)]
pub struct GizmosFetchState<Config, Clear>
where
    Config: GizmoConfigGroup,
    Clear: 'static + Send + Sync,
{
    state: <GizmosState<Config, Clear> as SystemParam>::State,
}

#[expect(
    unsafe_code,
    reason = "We cannot implement SystemParam without using unsafe code."
)]
// SAFETY: All methods are delegated to existing `SystemParam` implementations
unsafe impl<Config, Clear> SystemParam for Gizmos<'_, '_, Config, Clear>
where
    Config: GizmoConfigGroup,
    Clear: 'static + Send + Sync,
{
    type State = GizmosFetchState<Config, Clear>;
    type Item<'w, 's> = Gizmos<'w, 's, Config, Clear>;

    fn init_state(world: &mut World) -> Self::State {
        GizmosFetchState {
            state: GizmosState::<Config, Clear>::init_state(world),
        }
    }

    fn init_access(
        state: &Self::State,
        system_meta: &mut SystemMeta,
        system_access: &mut SystemAccess,
    ) -> Result<(), SystemParamAccessConflict> {
        GizmosState::<Config, Clear>::init_access(&state.state, system_meta, system_access)
    }

    fn apply(state: &mut Self::State, system_meta: &SystemMeta, world: &mut World) {
        GizmosState::<Config, Clear>::apply(&mut state.state, system_meta, world);
    }

    fn queue(state: &mut Self::State, system_meta: &SystemMeta, world: DeferredWorld) {
        GizmosState::<Config, Clear>::queue(&mut state.state, system_meta, world);
    }

    #[inline]
    unsafe fn get_param<'w, 's>(
        state: &'s mut Self::State,
        system_meta: &SystemMeta,
        world: UnsafeWorldCell<'w>,
        change_tick: Tick,
    ) -> Result<Self::Item<'w, 's>, SystemParamValidationError> {
        // SAFETY: Delegated to existing `SystemParam` implementations.
        let (mut f0, f1) = unsafe {
            GizmosState::<Config, Clear>::get_param(
                &mut state.state,
                system_meta,
                world,
                change_tick,
            )?
        };

        // Accessing the GizmoConfigStore in every API call reduces performance significantly.
        // Implementing SystemParam manually allows us to cache whether the config is currently enabled.
        // Having this available allows for cheap early returns when gizmos are disabled.
        //
        // We use `get_config` instead of `config` to accommodate `Option<Gizmos>`:
        // the user may decide not to initialize a gizmo group, so its config will not exist.
        let (config, config_ext) = f1.into_inner().get_config::<Config>().ok_or_else(|| {
            SystemParamValidationError::invalid::<Self>(
                format!("Requested config {} does not exist in `GizmoConfigStore`! Did you forget to add it using `app.init_gizmo_group<T>()`?", 
                Config::type_path()))
        })?;
        f0.enabled = config.enabled;

        Ok(Gizmos {
            buffer: f0,
            config,
            config_ext,
        })
    }
}

#[expect(
    unsafe_code,
    reason = "We cannot implement ReadOnlySystemParam without using unsafe code."
)]
// Safety: Each field is `ReadOnlySystemParam`, and Gizmos SystemParam does not mutate world
unsafe impl<'w, 's, Config, Clear> ReadOnlySystemParam for Gizmos<'w, 's, Config, Clear>
where
    Config: GizmoConfigGroup,
    Clear: 'static + Send + Sync,
    Deferred<'s, GizmoBuffer<Config, Clear>>: ReadOnlySystemParam,
    Res<'w, GizmoConfigStore>: ReadOnlySystemParam,
{
}

/// Buffer for gizmo vertex data.
#[derive(Debug, Clone, Reflect)]
#[reflect(Default)]
pub struct GizmoBuffer<Config, Clear>
where
    Config: GizmoConfigGroup,
    Clear: 'static + Send + Sync,
{
    pub(crate) enabled: bool,
    /// The positions of line segment endpoints.
    pub list_positions: Vec<Vec3>,
    /// The colors of line segment endpoints.
    pub list_colors: Vec<LinearRgba>,
    /// The positions of line strip vertices.
    pub strip_positions: Vec<Vec3>,
    /// The colors of line strip vertices.
    pub strip_colors: Vec<LinearRgba>,
    /// Who the line segments are drawn for.
    pub list_owners: OwnerRuns,
    /// Who the line strips are drawn for.
    pub strip_owners: OwnerRuns,
    /// Who lines drawn from now on are for.
    pub(crate) owner: Option<Entity>,
    #[reflect(ignore, clone)]
    pub(crate) marker: PhantomData<(Config, Clear)>,
}

impl<Config, Clear> Default for GizmoBuffer<Config, Clear>
where
    Config: GizmoConfigGroup,
    Clear: 'static + Send + Sync,
{
    fn default() -> Self {
        GizmoBuffer::new()
    }
}

impl<Config, Clear> GizmoBuffer<Config, Clear>
where
    Config: GizmoConfigGroup,
    Clear: 'static + Send + Sync,
{
    /// Constructs an empty `GizmoBuffer`.
    pub const fn new() -> Self {
        GizmoBuffer {
            enabled: true,
            list_positions: Vec::new(),
            list_colors: Vec::new(),
            strip_positions: Vec::new(),
            strip_colors: Vec::new(),
            list_owners: OwnerRuns(Vec::new()),
            strip_owners: OwnerRuns(Vec::new()),
            owner: None,
            marker: PhantomData,
        }
    }

    /// Draw what follows for `owner`: a renderer shows those lines in the owner's layers and
    /// world, and hides them with it. `None` goes back to the config group's rules.
    pub fn set_owner(&mut self, owner: Option<Entity>) {
        self.owner = owner;
        self.list_owners.mark(self.list_positions.len(), owner);
        self.strip_owners.mark(self.strip_positions.len(), owner);
    }

    /// Who lines drawn now are for.
    pub fn owner(&self) -> Option<Entity> {
        self.owner
    }

    /// Draw with `draw` for `entity`, then go back to the owner before.
    pub fn for_entity<R>(&mut self, entity: Entity, draw: impl FnOnce(&mut Self) -> R) -> R {
        let before = self.owner;
        self.set_owner(Some(entity));
        let out = draw(self);
        self.set_owner(before);
        out
    }
}

/// Read-only view into [`GizmoBuffer`] data.
pub struct GizmoBufferView<'a> {
    /// Vertex positions for line-list topology.
    pub list_positions: &'a Vec<Vec3>,
    /// Vertex colors for line-list topology.
    pub list_colors: &'a Vec<LinearRgba>,
    /// Vertex positions for line-strip topology.
    pub strip_positions: &'a Vec<Vec3>,
    /// Vertex colors for line-strip topology.
    pub strip_colors: &'a Vec<LinearRgba>,
    /// Who the line-list vertices are drawn for.
    pub list_owners: &'a OwnerRuns,
    /// Who the line-strip vertices are drawn for.
    pub strip_owners: &'a OwnerRuns,
}

impl<Config, Clear> SystemBuffer for GizmoBuffer<Config, Clear>
where
    Config: GizmoConfigGroup,
    Clear: 'static + Send + Sync,
{
    fn queue(&mut self, _system_meta: &SystemMeta, mut world: DeferredWorld) {
        if let Some(mut storage) = world.get_resource_mut::<GizmoStorage<Config, Clear>>() {
            let (list_at, strip_at) = (storage.list_positions.len(), storage.strip_positions.len());
            storage.list_owners.append(&self.list_owners, list_at);
            storage.strip_owners.append(&self.strip_owners, strip_at);
            storage.list_positions.append(&mut self.list_positions);
            storage.list_colors.append(&mut self.list_colors);
            storage.strip_positions.append(&mut self.strip_positions);
            storage.strip_colors.append(&mut self.strip_colors);
        } else {
            // Prevent the buffer from growing indefinitely if GizmoStorage
            // for the config group has not been initialized
            self.list_positions.clear();
            self.list_colors.clear();
            self.strip_positions.clear();
            self.strip_colors.clear();
        }
        // Next run starts unowned.
        self.list_owners.clear();
        self.strip_owners.clear();
        self.owner = None;
    }
}

impl<Config, Clear> GizmoBuffer<Config, Clear>
where
    Config: GizmoConfigGroup,
    Clear: 'static + Send + Sync,
{
    /// Clear all data. The owner stays, for what is drawn next.
    pub fn clear(&mut self) {
        self.list_positions.clear();
        self.list_colors.clear();
        self.strip_positions.clear();
        self.strip_colors.clear();
        self.list_owners.clear();
        self.strip_owners.clear();
        self.set_owner(self.owner);
    }

    /// Read-only view into the buffers data.
    pub fn buffer(&self) -> GizmoBufferView<'_> {
        let GizmoBuffer {
            list_positions,
            list_colors,
            strip_positions,
            strip_colors,
            list_owners,
            strip_owners,
            ..
        } = self;
        GizmoBufferView {
            list_positions,
            list_colors,
            strip_positions,
            strip_colors,
            list_owners,
            strip_owners,
        }
    }
    /// Draw a line in 3D from `start` to `end`.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::GREEN;
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.line(Vec3::ZERO, Vec3::X, GREEN);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn line(&mut self, start: Vec3, end: Vec3, color: impl Into<Color>) {
        if !self.enabled {
            return;
        }
        self.extend_list_positions([start, end]);
        self.add_list_color(color, 2);
    }

    /// Draw a line in 3D with a color gradient from `start` to `end`.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::{RED, GREEN};
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.line_gradient(Vec3::ZERO, Vec3::X, GREEN, RED);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn line_gradient<C: Into<Color>>(
        &mut self,
        start: Vec3,
        end: Vec3,
        start_color: C,
        end_color: C,
    ) {
        if !self.enabled {
            return;
        }
        self.extend_list_positions([start, end]);
        self.extend_list_colors([start_color, end_color]);
    }

    /// Draw a line in 3D from `start` to `start + vector`.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::GREEN;
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.ray(Vec3::Y, Vec3::X, GREEN);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn ray(&mut self, start: Vec3, vector: Vec3, color: impl Into<Color>) {
        if !self.enabled {
            return;
        }
        self.line(start, start + vector, color);
    }

    /// Draw a line in 3D with a color gradient from `start` to `start + vector`.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::{RED, GREEN};
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.ray_gradient(Vec3::Y, Vec3::X, GREEN, RED);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn ray_gradient<C: Into<Color>>(
        &mut self,
        start: Vec3,
        vector: Vec3,
        start_color: C,
        end_color: C,
    ) {
        if !self.enabled {
            return;
        }
        self.line_gradient(start, start + vector, start_color, end_color);
    }

    /// Draw a line in 3D made of straight segments between the points.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::GREEN;
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.linestrip([Vec3::ZERO, Vec3::X, Vec3::Y], GREEN);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn linestrip(
        &mut self,
        positions: impl IntoIterator<Item = Vec3>,
        color: impl Into<Color>,
    ) {
        if !self.enabled {
            return;
        }
        self.extend_strip_positions(positions);
        let len = self.strip_positions.len();
        let linear_color = LinearRgba::from(color.into());
        self.strip_colors.resize(len - 1, linear_color);
        self.strip_colors.push(LinearRgba::NAN);
    }

    /// Draw a line in 3D made of straight segments between the points, with the first and last connected.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::GREEN;
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.lineloop([Vec3::ZERO, Vec3::X, Vec3::Y], GREEN);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn lineloop(&mut self, positions: impl IntoIterator<Item = Vec3>, color: impl Into<Color>) {
        if !self.enabled {
            return;
        }

        // Loop back to the start; second is needed to ensure that
        // the joint on the first corner is drawn.
        let mut positions = positions.into_iter();
        let first = positions.next();
        let second = positions.next();

        self.linestrip(
            first
                .into_iter()
                .chain(second)
                .chain(positions)
                .chain(first)
                .chain(second),
            color,
        );
    }

    /// Draw a line in 3D made of straight segments between the points, with a color gradient.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::{BLUE, GREEN, RED};
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.linestrip_gradient([
    ///         (Vec3::ZERO, GREEN),
    ///         (Vec3::X, RED),
    ///         (Vec3::Y, BLUE)
    ///     ]);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn linestrip_gradient<C: Into<Color>>(
        &mut self,
        points: impl IntoIterator<Item = (Vec3, C)>,
    ) {
        if !self.enabled {
            return;
        }
        let points = points.into_iter();

        let GizmoBuffer {
            strip_positions,
            strip_colors,
            ..
        } = self;

        let (min, _) = points.size_hint();
        strip_positions.reserve(min);
        strip_colors.reserve(min);

        for (position, color) in points {
            strip_positions.push(position);
            strip_colors.push(LinearRgba::from(color.into()));
        }

        strip_positions.push(Vec3::NAN);
        strip_colors.push(LinearRgba::NAN);
    }

    /// Draw a wireframe rectangle in 3D with the given `isometry` applied.
    ///
    /// If `isometry == Isometry3d::IDENTITY` then
    ///
    /// - the center is at `Vec3::ZERO`
    /// - the sizes are aligned with the `Vec3::X` and `Vec3::Y` axes.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::GREEN;
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.rect(Isometry3d::IDENTITY, Vec2::ONE, GREEN);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn rect(&mut self, isometry: impl Into<Isometry3d>, size: Vec2, color: impl Into<Color>) {
        if !self.enabled {
            return;
        }
        let isometry = isometry.into();
        let [tl, tr, br, bl] = rect_inner(size).map(|vec2| isometry * vec2.extend(0.));
        self.lineloop([tl, tr, br, bl], color);
    }

    /// Draw a wireframe cube in 3D.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_transform::prelude::*;
    /// # use bevy_color::palettes::basic::GREEN;
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.cube(Transform::IDENTITY, GREEN);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn cube(&mut self, transform: impl TransformPoint, color: impl Into<Color>) {
        let polymorphic_color: Color = color.into();
        if !self.enabled {
            return;
        }
        let rect = rect_inner(Vec2::ONE);
        // Front
        let [tlf, trf, brf, blf] = rect.map(|vec2| transform.transform_point(vec2.extend(0.5)));
        // Back
        let [tlb, trb, brb, blb] = rect.map(|vec2| transform.transform_point(vec2.extend(-0.5)));

        let strip_positions = [
            tlf, trf, brf, blf, tlf, // Front
            tlb, trb, brb, blb, tlb, // Back
        ];
        self.linestrip(strip_positions, polymorphic_color);

        let list_positions = [
            trf, trb, brf, brb, blf, blb, // Front to back
        ];
        self.extend_list_positions(list_positions);

        self.add_list_color(polymorphic_color, 6);
    }

    /// Draw a wireframe aabb in 3D.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_transform::prelude::*;
    /// # use bevy_math::Vec3;
    /// # use bevy_shape::Aabb3d;
    /// # use bevy_color::palettes::basic::GREEN;
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.aabb_3d(Aabb3d::new(Vec3::ZERO, Vec3::ONE), Transform::IDENTITY, GREEN);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn aabb_3d(
        &mut self,
        aabb: impl Into<Aabb3d>,
        transform: impl TransformPoint,
        color: impl Into<Color>,
    ) {
        let polymorphic_color: Color = color.into();
        if !self.enabled {
            return;
        }
        let aabb = aabb.into();
        let [tlf, trf, brf, blf, tlb, trb, brb, blb] = [
            Vec3::new(aabb.min.x, aabb.max.y, aabb.max.z),
            Vec3::new(aabb.max.x, aabb.max.y, aabb.max.z),
            Vec3::new(aabb.max.x, aabb.min.y, aabb.max.z),
            Vec3::new(aabb.min.x, aabb.min.y, aabb.max.z),
            Vec3::new(aabb.min.x, aabb.max.y, aabb.min.z),
            Vec3::new(aabb.max.x, aabb.max.y, aabb.min.z),
            Vec3::new(aabb.max.x, aabb.min.y, aabb.min.z),
            Vec3::new(aabb.min.x, aabb.min.y, aabb.min.z),
        ]
        .map(|v| transform.transform_point(v));

        let strip_positions = [
            tlf, trf, brf, blf, tlf, // Front
            tlb, trb, brb, blb, tlb, // Back
        ];
        self.linestrip(strip_positions, polymorphic_color);

        let list_positions = [
            trf, trb, brf, brb, blf, blb, // Front to back
        ];
        self.extend_list_positions(list_positions);

        self.add_list_color(polymorphic_color, 6);
    }

    /// Draw a line in 2D from `start` to `end`.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::GREEN;
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.line_2d(Vec2::ZERO, Vec2::X, GREEN);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn line_2d(&mut self, start: Vec2, end: Vec2, color: impl Into<Color>) {
        if !self.enabled {
            return;
        }
        self.line(start.extend(0.), end.extend(0.), color);
    }

    /// Draw a line in 2D with a color gradient from `start` to `end`.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::{RED, GREEN};
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.line_gradient_2d(Vec2::ZERO, Vec2::X, GREEN, RED);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn line_gradient_2d<C: Into<Color>>(
        &mut self,
        start: Vec2,
        end: Vec2,
        start_color: C,
        end_color: C,
    ) {
        if !self.enabled {
            return;
        }
        self.line_gradient(start.extend(0.), end.extend(0.), start_color, end_color);
    }

    /// Draw a line in 2D made of straight segments between the points.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::GREEN;
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.linestrip_2d([Vec2::ZERO, Vec2::X, Vec2::Y], GREEN);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn linestrip_2d(
        &mut self,
        positions: impl IntoIterator<Item = Vec2>,
        color: impl Into<Color>,
    ) {
        if !self.enabled {
            return;
        }
        self.linestrip(positions.into_iter().map(|vec2| vec2.extend(0.)), color);
    }

    /// Draw a line in 2D made of straight segments between the points, with the first and last connected.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::GREEN;
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.lineloop_2d([Vec2::ZERO, Vec2::X, Vec2::Y], GREEN);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn lineloop_2d(
        &mut self,
        positions: impl IntoIterator<Item = Vec2>,
        color: impl Into<Color>,
    ) {
        if !self.enabled {
            return;
        }
        self.lineloop(positions.into_iter().map(|vec2| vec2.extend(0.)), color);
    }

    /// Draw a line in 2D made of straight segments between the points, with a color gradient.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::{RED, GREEN, BLUE};
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.linestrip_gradient_2d([
    ///         (Vec2::ZERO, GREEN),
    ///         (Vec2::X, RED),
    ///         (Vec2::Y, BLUE)
    ///     ]);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn linestrip_gradient_2d<C: Into<Color>>(
        &mut self,
        positions: impl IntoIterator<Item = (Vec2, C)>,
    ) {
        if !self.enabled {
            return;
        }
        self.linestrip_gradient(
            positions
                .into_iter()
                .map(|(vec2, color)| (vec2.extend(0.), color)),
        );
    }

    /// Draw a line in 2D from `start` to `start + vector`.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::GREEN;
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.ray_2d(Vec2::Y, Vec2::X, GREEN);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn ray_2d(&mut self, start: Vec2, vector: Vec2, color: impl Into<Color>) {
        if !self.enabled {
            return;
        }
        self.line_2d(start, start + vector, color);
    }

    /// Draw a line in 2D with a color gradient from `start` to `start + vector`.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::{RED, GREEN};
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.line_gradient(Vec3::Y, Vec3::X, GREEN, RED);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn ray_gradient_2d<C: Into<Color>>(
        &mut self,
        start: Vec2,
        vector: Vec2,
        start_color: C,
        end_color: C,
    ) {
        if !self.enabled {
            return;
        }
        self.line_gradient_2d(start, start + vector, start_color, end_color);
    }

    /// Draw a wireframe rectangle in 2D with the given `isometry` applied.
    ///
    /// If `isometry == Isometry2d::IDENTITY` then
    ///
    /// - the center is at `Vec2::ZERO`
    /// - the sizes are aligned with the `Vec2::X` and `Vec2::Y` axes.
    ///
    /// # Example
    /// ```
    /// # use bevy_gizmos::prelude::*;
    /// # use bevy_math::prelude::*;
    /// # use bevy_color::palettes::basic::GREEN;
    /// fn system(mut gizmos: Gizmos) {
    ///     gizmos.rect_2d(Isometry2d::IDENTITY, Vec2::ONE, GREEN);
    /// }
    /// # bevy_ecs::system::assert_is_system(system);
    /// ```
    #[inline]
    pub fn rect_2d(
        &mut self,
        isometry: impl Into<Isometry2d>,
        size: Vec2,
        color: impl Into<Color>,
    ) {
        if !self.enabled {
            return;
        }
        let isometry = isometry.into();
        let [tl, tr, br, bl] = rect_inner(size).map(|vec2| isometry * vec2);
        self.lineloop_2d([tl, tr, br, bl], color);
    }

    #[inline]
    fn extend_list_positions(&mut self, positions: impl IntoIterator<Item = Vec3>) {
        self.list_positions.extend(positions);
    }

    #[inline]
    fn extend_list_colors(&mut self, colors: impl IntoIterator<Item = impl Into<Color>>) {
        self.list_colors.extend(
            colors
                .into_iter()
                .map(|color| LinearRgba::from(color.into())),
        );
    }

    #[inline]
    fn add_list_color(&mut self, color: impl Into<Color>, count: usize) {
        let polymorphic_color: Color = color.into();
        let linear_color = LinearRgba::from(polymorphic_color);

        self.list_colors.extend(iter::repeat_n(linear_color, count));
    }

    #[inline]
    fn extend_strip_positions(&mut self, positions: impl IntoIterator<Item = Vec3>) {
        self.strip_positions.extend(positions);
        self.strip_positions.push(Vec3::NAN);
    }
}

fn rect_inner(size: Vec2) -> [Vec2; 4] {
    let half_size = size / 2.;
    let tl = Vec2::new(-half_size.x, half_size.y);
    let tr = Vec2::new(half_size.x, half_size.y);
    let bl = Vec2::new(-half_size.x, -half_size.y);
    let br = Vec2::new(half_size.x, -half_size.y);
    [tl, tr, br, bl]
}

#[cfg(test)]
mod owner_tests {
    use super::*;
    use crate::config::DefaultGizmoConfigGroup;
    use bevy_color::palettes::basic::WHITE;

    fn entity(index: u32) -> Entity {
        Entity::from_raw_u32(index).unwrap()
    }

    #[test]
    fn owners_mark_runs_and_restore() {
        let mut buffer = GizmoBuffer::<DefaultGizmoConfigGroup, ()>::new();
        buffer.line(Vec3::ZERO, Vec3::X, WHITE);
        buffer.for_entity(entity(1), |g| {
            g.line(Vec3::ZERO, Vec3::Y, WHITE);
            g.for_entity(entity(2), |g| g.line(Vec3::ZERO, Vec3::Z, WHITE));
            g.line(Vec3::ZERO, Vec3::Y, WHITE);
        });
        buffer.line(Vec3::ZERO, Vec3::X, WHITE);
        let ranges: Vec<_> = buffer.list_owners.ranges(buffer.list_positions.len()).collect();
        assert_eq!(
            ranges,
            vec![
                (0, 2, None),
                (2, 4, Some(entity(1))),
                (4, 6, Some(entity(2))),
                (6, 8, Some(entity(1))),
                (8, 10, None),
            ]
        );
        // An owner set and unset with nothing drawn leaves no run.
        let mut empty = GizmoBuffer::<DefaultGizmoConfigGroup, ()>::new();
        empty.for_entity(entity(3), |_| {});
        assert!(empty.list_owners.0.is_empty());
    }

    #[test]
    fn appended_runs_shift_and_do_not_inherit_the_tail_owner() {
        let mut storage = GizmoStorage::<DefaultGizmoConfigGroup, ()>::default();
        let mut owned = GizmoStorage::<DefaultGizmoConfigGroup, ()>::default();
        owned.list_positions = vec![Vec3::ZERO; 2];
        owned.list_colors = vec![LinearRgba::WHITE; 2];
        owned.list_owners.mark(0, Some(entity(1)));
        let mut unowned = GizmoStorage::<DefaultGizmoConfigGroup, ()>::default();
        unowned.list_positions = vec![Vec3::ZERO; 4];
        unowned.list_colors = vec![LinearRgba::WHITE; 4];
        storage.append_storage(&owned);
        storage.append_storage(&unowned);
        storage.append_storage(&owned);
        let ranges: Vec<_> = storage.list_owners.ranges(storage.list_positions.len()).collect();
        assert_eq!(
            ranges,
            vec![(0, 2, Some(entity(1))), (2, 6, None), (6, 8, Some(entity(1)))]
        );
    }
}
