use bevy_app::{Plugin, PostUpdate};
use bevy_ecs::{
    bundle::Bundle, children, component::Component, entity::Entity, hierarchy::{ChildOf, Children},
    observer::On, prelude::ReflectComponent, spawn::SpawnableList, spawn::SpawnRelated,
    query::{Added, With},
    system::{Commands, Query},
};
use bevy_math::Vec2;
use bevy_picking::events::{Pointer, Scroll};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_ui::{AlignItems, ComputedNode, Display, JustifyContent, Node, Overflow, OverflowAxis, PositionType, ScrollPosition, UiRect, Val};
use bevy_ui_widgets::{observe, Scrollbar, ControlOrientation, CoreScrollbarThumb};

use crate::{rounded_corners::RoundedCorners, theme::{ThemeBackgroundColor, ThemeToken}, tokens};

/// Scrollbar styling constants
const SCROLLBAR_WIDTH: f32 = 8.0;
const SCROLLBAR_MIN_THUMB_SIZE: f32 = 10.0; // Minimum thumb size as percentage
const LINE_HEIGHT: f32 = 21.0;

/// Plugin that handles scrollbar creation and updates
pub struct ScrollbarPlugin;

impl Plugin for ScrollbarPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(PostUpdate, (spawn_scrollbars, update_scrollbars));
    }
}

/// Marker component for scroll containers that stores props
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct ScrollContainer {
    /// Whether to show scrollbars for this container
    pub show_scrollbars: bool,
}

/// Marker component for the scroll wrapper (parent of scroll container)
#[derive(Component, Debug, Default, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Component, Default, Debug, PartialEq, Clone)]
pub struct ScrollWrapper;

/// Marker component for vertical scrollbar (stores thumb entity)
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Component, Debug, PartialEq, Clone)]
pub struct VScrollbar(pub Entity);

/// Marker component for horizontal scrollbar (stores thumb entity)
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Component, Debug, PartialEq, Clone)]
pub struct HScrollbar(pub Entity);


/// Parameters for the scroll container template.
pub struct ScrollProps {
    /// Width of the scroll container
    pub width: Val,
    /// Height of the scroll container
    pub height: Val,
    /// Overflow settings (horizontal, vertical, or both)
    pub overflow: Overflow,
    /// Flex direction for content layout
    pub flex_direction: bevy_ui::FlexDirection,
    /// Show scrollbars
    pub show_scrollbars: bool,
    /// Rounded corners options
    pub corners: RoundedCorners,
    /// Background color token
    pub bg_token: ThemeToken,
    /// Align items (horizontal alignment for column layouts, vertical for row layouts)
    pub align_items: AlignItems,
}

impl Default for ScrollProps {
    fn default() -> Self {
        Self {
            width: Val::Percent(100.0),
            height: Val::Auto,
            overflow: Overflow::visible(),
            flex_direction: bevy_ui::FlexDirection::Column,
            show_scrollbars: true,
            corners: RoundedCorners::default(),
            bg_token: tokens::SCROLL_BG,
            align_items: AlignItems::Stretch,
        }
    }
}

/// Template function to spawn a scroll container.
///
/// This widget provides a styled scrollable container that responds to mouse wheel events.
/// The scroll observer is automatically attached to handle `Pointer<Scroll>` events.
///
/// # Arguments
/// * `props` - construction properties for the scroll container.
/// * `overrides` - a bundle of components that are merged in with the normal scroll container components.
/// * `children` - Either a [`SpawnableList`] (like SpawnIter) or the result of the `children!` macro.
///
/// # Examples
/// ```ignore
/// // Homogeneous list with SpawnIter
/// scroll(
///     ScrollProps::vertical(px(400)),
///     (),
///     SpawnIter((0..10).map(|i| (Node::default(), Text(format!("Item {}", i)))))
/// )
///
/// // Heterogeneous list with children! macro
/// scroll(
///     ScrollProps::vertical(px(400)),
///     (),
///     children![
///         some_function_returning_impl_bundle(),
///         another_function(),
///     ]
/// )
/// ```
pub fn scroll<C: Bundle, B: Bundle>(
    props: ScrollProps,
    overrides: B,
    children: C,
) -> impl Bundle {
    // Calculate padding based on which scrollbars will be shown
    let base_padding = 4.0;
    let padding = if props.show_scrollbars {
        UiRect {
            left: Val::Px(base_padding),
            top: Val::Px(base_padding),
            right: Val::Px(if props.overflow.y == OverflowAxis::Scroll {
                base_padding + SCROLLBAR_WIDTH
            } else {
                base_padding
            }),
            bottom: Val::Px(if props.overflow.x == OverflowAxis::Scroll {
                base_padding + SCROLLBAR_WIDTH
            } else {
                base_padding
            }),
        }
    } else {
        UiRect::all(Val::Px(base_padding))
    };

    (
        Node {
            width: props.width,
            height: props.height,
            position_type: PositionType::Relative,
            ..Default::default()
        },
        ScrollWrapper,
        children![
            (
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    flex_direction: props.flex_direction,
                    justify_content: JustifyContent::Start,
                    align_items: props.align_items,
                    overflow: props.overflow,
                    padding,
                    ..Default::default()
                },
                ScrollContainer {
                    show_scrollbars: props.show_scrollbars,
                },
                ScrollPosition::default(),
                props.corners.to_border_radius(4.0),
                ThemeBackgroundColor(props.bg_token),
                observe(scroll_observer),
                children,
                overrides,
            ),
        ],
    )
}

impl ScrollProps {
    /// Create a vertically scrolling container with default styling
    pub fn vertical(height: Val) -> Self {
        Self {
            width: Val::Percent(100.0),
            height,
            overflow: Overflow {
                x: OverflowAxis::Visible,
                y: OverflowAxis::Scroll,
            },
            flex_direction: bevy_ui::FlexDirection::Column,
            show_scrollbars: true,
            corners: RoundedCorners::default(),
            bg_token: tokens::SCROLL_BG,
            align_items: AlignItems::Stretch,
        }
    }

    /// Create a horizontally scrolling container with default styling
    pub fn horizontal(width: Val) -> Self {
        Self {
            width,
            height: Val::Auto,
            overflow: Overflow {
                x: OverflowAxis::Scroll,
                y: OverflowAxis::Visible,
            },
            flex_direction: bevy_ui::FlexDirection::Row,
            show_scrollbars: true,
            corners: RoundedCorners::default(),
            bg_token: tokens::SCROLL_BG,
            align_items: AlignItems::Stretch,
        }
    }

    /// Create a bidirectionally scrolling container with default styling
    pub fn both(width: Val, height: Val) -> Self {
        Self {
            width,
            height,
            overflow: Overflow {
                x: OverflowAxis::Scroll,
                y: OverflowAxis::Scroll,
            },
            flex_direction: bevy_ui::FlexDirection::Column,
            show_scrollbars: true,
            corners: RoundedCorners::default(),
            bg_token: tokens::SCROLL_BG,
            align_items: AlignItems::Stretch,
        }
    }
}

/// System that spawns scrollbars for newly created scroll containers
fn spawn_scrollbars(
    mut commands: Commands,
    scroll_containers: Query<(Entity, &Node, &ScrollContainer, &ChildOf), Added<ScrollContainer>>,
) {
    for (entity, node, container, child_of) in scroll_containers.iter() {
        if !container.show_scrollbars {
            continue;
        }

        let wrapper_entity = child_of.parent();

        // Spawn vertical scrollbar
        if node.overflow.y == OverflowAxis::Scroll {
            let mut thumb_id = Entity::PLACEHOLDER;
            let scrollbar = commands
                .spawn((
                    Node {
                        width: Val::Px(SCROLLBAR_WIDTH),
                        height: Val::Percent(100.0),
                        position_type: PositionType::Absolute,
                        right: Val::Px(0.0),
                        top: Val::Px(0.0),
                        bottom: Val::Px(0.0),
                        display: Display::Flex,
                        flex_direction: bevy_ui::FlexDirection::Column,
                        ..Default::default()
                    },
                    Scrollbar {
                        target: entity,
                        orientation: ControlOrientation::Vertical,
                        min_thumb_length: SCROLLBAR_MIN_THUMB_SIZE,
                    },
                    ThemeBackgroundColor(tokens::SCROLLBAR_TRACK),
                    bevy_ui::ZIndex(1),
                ))
                .with_children(|parent| {
                    thumb_id = parent.spawn((
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Percent(20.0),
                            ..Default::default()
                        },
                        CoreScrollbarThumb,
                        ThemeBackgroundColor(tokens::SCROLLBAR_THUMB),
                    )).id();
                })
                .id();
            commands.entity(entity).insert(VScrollbar(thumb_id));
            commands.entity(wrapper_entity).add_child(scrollbar);
        }

        // Spawn horizontal scrollbar
        if node.overflow.x == OverflowAxis::Scroll {
            let mut thumb_id = Entity::PLACEHOLDER;
            let scrollbar = commands
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(SCROLLBAR_WIDTH),
                        position_type: PositionType::Absolute,
                        bottom: Val::Px(0.0),
                        left: Val::Px(0.0),
                        right: Val::Px(0.0),
                        display: Display::Flex,
                        flex_direction: bevy_ui::FlexDirection::Row,
                        ..Default::default()
                    },
                    Scrollbar {
                        target: entity,
                        orientation: ControlOrientation::Horizontal,
                        min_thumb_length: SCROLLBAR_MIN_THUMB_SIZE,
                    },
                    ThemeBackgroundColor(tokens::SCROLLBAR_TRACK),
                    bevy_ui::ZIndex(1),
                ))
                .with_children(|parent| {
                    thumb_id = parent.spawn((
                        Node {
                            width: Val::Percent(20.0),
                            height: Val::Percent(100.0),
                            ..Default::default()
                        },
                        CoreScrollbarThumb,
                        ThemeBackgroundColor(tokens::SCROLLBAR_THUMB),
                    )).id();
                })
                .id();
            commands.entity(entity).insert(HScrollbar(thumb_id));
            commands.entity(wrapper_entity).add_child(scrollbar);
        }
    }
}

/// System that updates scrollbar thumb position and size based on scroll position
fn update_scrollbars(
    scroll_containers: Query<(&ScrollPosition, &ComputedNode, Option<&VScrollbar>, Option<&HScrollbar>), With<ScrollContainer>>,
    mut thumb_query: Query<&mut Node, With<CoreScrollbarThumb>>,
) {
    for (scroll_pos, computed, v_scrollbar, h_scrollbar) in scroll_containers.iter() {
        let content_size = computed.content_size();
        let container_size = computed.size();
        let scale = computed.inverse_scale_factor();

        // Update vertical scrollbar thumb
        if let Some(v_scrollbar) = v_scrollbar {
            if v_scrollbar.0 != Entity::PLACEHOLDER {
                if let Ok(mut thumb_node) = thumb_query.get_mut(v_scrollbar.0) {
                    let max_scroll = (content_size.y - container_size.y) * scale;
                    if max_scroll > 0.0 {
                        // Calculate thumb size as a percentage of visible area
                        let visible_ratio = (container_size.y / content_size.y).clamp(0.0, 1.0);
                        let thumb_height = (visible_ratio * 100.0).max(SCROLLBAR_MIN_THUMB_SIZE);

                        // Calculate thumb position as percentage
                        let scroll_ratio = (scroll_pos.y / max_scroll).clamp(0.0, 1.0);
                        let max_thumb_offset = 100.0 - thumb_height;
                        let thumb_top = scroll_ratio * max_thumb_offset;

                        thumb_node.height = Val::Percent(thumb_height);
                        thumb_node.top = Val::Percent(thumb_top);
                    }
                }
            }
        }

        // Update horizontal scrollbar thumb
        if let Some(h_scrollbar) = h_scrollbar {
            if h_scrollbar.0 != Entity::PLACEHOLDER {
                if let Ok(mut thumb_node) = thumb_query.get_mut(h_scrollbar.0) {
                    let max_scroll = (content_size.x - container_size.x) * scale;
                    if max_scroll > 0.0 {
                        // Calculate thumb size as a percentage of visible area
                        let visible_ratio = (container_size.x / content_size.x).clamp(0.0, 1.0);
                        let thumb_width = (visible_ratio * 100.0).max(SCROLLBAR_MIN_THUMB_SIZE);

                        // Calculate thumb position as percentage
                        let scroll_ratio = (scroll_pos.x / max_scroll).clamp(0.0, 1.0);
                        let max_thumb_offset = 100.0 - thumb_width;
                        let thumb_left = scroll_ratio * max_thumb_offset;

                        thumb_node.width = Val::Percent(thumb_width);
                        thumb_node.left = Val::Percent(thumb_left);
                    }
                }
            }
        }
    }
}

/// Observer that handles scroll events for UI containers with ScrollPosition.
///
/// This observer is automatically attached to scroll containers created with the
/// `scroll()` function. It handles `Pointer<Scroll>` events and updates the
/// `ScrollPosition` accordingly.
fn scroll_observer(
    scroll: On<Pointer<Scroll>>,
    mut query: Query<(&mut ScrollPosition, &Node, &ComputedNode)>,
) {
    let Ok((mut scroll_position, node, computed)) = query.get_mut(scroll.entity) else {
        return;
    };

    let event = scroll.event();
    let mut delta = -Vec2::new(event.x, event.y);

    // Convert line units to pixels (MouseScrollUnit is not public, so we check the magnitude)
    // Line scrolling typically has smaller values than pixel scrolling
    if delta.x.abs() < 10.0 && delta.y.abs() < 10.0 {
        delta *= LINE_HEIGHT;
    }

    // If only horizontal scrolling is enabled and we have vertical scroll input,
    // convert the vertical scroll to horizontal
    if node.overflow.x == OverflowAxis::Scroll
        && node.overflow.y != OverflowAxis::Scroll
        && delta.x == 0.
        && delta.y != 0. {
        delta.x = delta.y;
        delta.y = 0.;
    }

    let max_offset = (computed.content_size() - computed.size()) * computed.inverse_scale_factor();

    // Handle horizontal scrolling
    if node.overflow.x == OverflowAxis::Scroll && delta.x != 0. {
        let at_limit = if delta.x > 0. {
            scroll_position.x >= max_offset.x
        } else {
            scroll_position.x <= 0.
        };

        if !at_limit {
            scroll_position.x = (scroll_position.x + delta.x).clamp(0., max_offset.x);
        }
    }

    // Handle vertical scrolling
    if node.overflow.y == OverflowAxis::Scroll && delta.y != 0. {
        let at_limit = if delta.y > 0. {
            scroll_position.y >= max_offset.y
        } else {
            scroll_position.y <= 0.
        };

        if !at_limit {
            scroll_position.y = (scroll_position.y + delta.y).clamp(0., max_offset.y);
        }
    }
}
