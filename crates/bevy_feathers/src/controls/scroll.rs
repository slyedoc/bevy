use bevy_ecs::{
    bundle::Bundle, observer::On, relationship::RelatedSpawner,
    spawn::{SpawnRelated, SpawnWith}, system::Query,
};
use bevy_math::Vec2;
use bevy_picking::events::{Pointer, Scroll};
use bevy_ui::{AlignItems, ComputedNode, JustifyContent, Node, Overflow, OverflowAxis, PositionType, ScrollPosition, UiRect, Val};
use bevy_ui_widgets::{observe, Scrollbar, ControlOrientation, CoreScrollbarThumb};

use crate::{rounded_corners::RoundedCorners, theme::{ThemeBackgroundColor, ThemeToken}, tokens};

/// Scrollbar styling constants
const SCROLLBAR_WIDTH: f32 = 8.0;
const SCROLLBAR_MIN_THUMB_SIZE: f32 = 10.0;
const LINE_HEIGHT: f32 = 21.0;


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
/// * `children` - Either `Children::spawn(SpawnIter(...))` for homogeneous lists or the result of the `children!` macro for heterogeneous content.
///
/// # Examples
/// ```ignore
/// // Homogeneous list with SpawnIter (must be wrapped in Children::spawn)
/// scroll(
///     ScrollProps::vertical(px(400)),
///     (),
///     Children::spawn(SpawnIter((0..10).map(|i| (Node::default(), Text(format!("Item {}", i))))))
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
    let base_padding = 4.0;
    let has_vscroll = props.overflow.y == OverflowAxis::Scroll;
    let has_hscroll = props.overflow.x == OverflowAxis::Scroll;

    let padding = UiRect {
        left: Val::Px(base_padding),
        top: Val::Px(base_padding),
        right: Val::Px(if has_vscroll { base_padding + SCROLLBAR_WIDTH } else { base_padding }),
        bottom: Val::Px(if has_hscroll { base_padding + SCROLLBAR_WIDTH } else { base_padding }),
    };

    (
        Node {
            width: props.width,
            height: props.height,
            position_type: PositionType::Relative,
            ..Default::default()
        },
        bevy_ecs::hierarchy::Children::spawn(
            SpawnWith(move |parent: &mut RelatedSpawner<bevy_ecs::hierarchy::ChildOf>| {
                // Spawn the scroll area first
                let scroll_area_id = parent.spawn((
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
                    ScrollPosition::default(),
                    props.corners.to_border_radius(4.0),
                    ThemeBackgroundColor(props.bg_token),
                    observe(scroll_observer),
                    children,
                    overrides,
                )).id();

                // Spawn vertical scrollbar if needed
                if has_vscroll {
                    parent.spawn((
                        Node {
                            width: Val::Px(SCROLLBAR_WIDTH),
                            height: Val::Percent(100.0),
                            position_type: PositionType::Absolute,
                            right: Val::Px(0.0),
                            top: Val::Px(0.0),
                            ..Default::default()
                        },
                        Scrollbar::new(scroll_area_id, ControlOrientation::Vertical, SCROLLBAR_MIN_THUMB_SIZE),
                        ThemeBackgroundColor(tokens::SCROLLBAR_TRACK),
                        bevy_ui::ZIndex(1),
                        bevy_ecs::hierarchy::Children::spawn(bevy_ecs::spawn::Spawn((
                            Node {
                                width: Val::Percent(100.0),
                                position_type: PositionType::Absolute,
                                ..Default::default()
                            },
                            CoreScrollbarThumb,
                            props.corners.to_border_radius(2.0),
                            ThemeBackgroundColor(tokens::SCROLLBAR_THUMB),
                        ))),
                    ));
                }

                // Spawn horizontal scrollbar if needed
                if has_hscroll {
                    parent.spawn((
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Px(SCROLLBAR_WIDTH),
                            position_type: PositionType::Absolute,
                            bottom: Val::Px(0.0),
                            left: Val::Px(0.0),
                            ..Default::default()
                        },
                        Scrollbar::new(scroll_area_id, ControlOrientation::Horizontal, SCROLLBAR_MIN_THUMB_SIZE),
                        ThemeBackgroundColor(tokens::SCROLLBAR_TRACK),
                        bevy_ui::ZIndex(1),
                        bevy_ecs::hierarchy::Children::spawn(bevy_ecs::spawn::Spawn((
                            Node {
                                height: Val::Percent(100.0),
                                position_type: PositionType::Absolute,
                                ..Default::default()
                            },
                            CoreScrollbarThumb,
                            props.corners.to_border_radius(2.0),
                            ThemeBackgroundColor(tokens::SCROLLBAR_THUMB),
                        ))),
                    ));
                }
            })
        ),
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
            corners: RoundedCorners::default(),
            bg_token: tokens::SCROLL_BG,
            align_items: AlignItems::Stretch,
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
