//! `String` fields: a feathers text input that commits on Enter or when it loses focus, and puts
//! back what it showed on Escape.

use bevy_ecs::prelude::*;
use bevy_feathers::controls::{FeathersTextInput, FeathersTextInputContainer};
use bevy_input::{keyboard::KeyboardInput, ButtonState};
use bevy_input::keyboard::KeyCode;
use bevy_input_focus::{FocusLost, FocusedInput, InputFocus};
use bevy_reflect::{CreateTypeData, PartialReflect};
use bevy_scene::prelude::*;
use bevy_scene::Scene;
use bevy_text::{EditableText, TextEdit};
use bevy_ui_widgets::ValueChange;

use crate::attributes::FieldCtx;
use crate::binding::{queue_edit, InspectorBinding};
use crate::recurse::{parse_path, BuildCx};
use crate::widget::ReflectInspectorWidget;

/// The text a string field shows, written into its buffer by [`seed_text_inputs`] once the
/// input has spawned, and kept so Escape and the external sync know what is on screen.
#[derive(Component, Clone, Default)]
pub(crate) struct ShownText {
    pub(crate) text: String,
    pub(crate) seeded: bool,
}

/// An Enter or Escape settled the edit, so the focus loss that follows commits nothing.
#[derive(Component)]
struct Settled;

fn build_string(
    cx: &BuildCx,
    path: &str,
    value: &dyn PartialReflect,
    _field: &FieldCtx,
) -> Box<dyn Scene> {
    let text = value
        .try_as_reflect()
        .and_then(|r| r.downcast_ref::<String>())
        .cloned()
        .unwrap_or_default();
    let shown = ShownText {
        text,
        seeded: false,
    };
    let binding = InspectorBinding {
        root: Some(cx.root.clone()),
        path: parse_path(path),
    };
    Box::new(bsn! {
        @FeathersTextInputContainer
        Children [
            @FeathersTextInput
            shown
            binding
            on(on_enter_or_escape)
            on(on_focus_lost)
            on(writeback_string)
        ]
    })
}

impl CreateTypeData<String> for ReflectInspectorWidget {
    fn create_type_data(_: ()) -> Self {
        Self {
            build: build_string,
        }
    }
}

/// Write each new input's text into its buffer.
pub(crate) fn seed_text_inputs(mut inputs: Query<(&mut ShownText, &mut EditableText)>) {
    for (mut shown, mut editable) in &mut inputs {
        if shown.seeded {
            continue;
        }
        shown.seeded = true;
        replace_text(&mut editable, &shown.text);
    }
}

pub(crate) fn replace_text(editable: &mut EditableText, text: &str) {
    editable.queue_edit(TextEdit::SelectAll);
    editable.queue_edit(TextEdit::Insert(text.to_string().into()));
}

fn on_enter_or_escape(
    key: On<FocusedInput<KeyboardInput>>,
    mut inputs: Query<(&mut EditableText, &mut ShownText)>,
    mut focus: ResMut<InputFocus>,
    mut commands: Commands,
) {
    let code = key.input.key_code;
    if key.input.state != ButtonState::Pressed || !matches!(code, KeyCode::Enter | KeyCode::Escape)
    {
        return;
    }
    let input = key.event_target();
    let Ok((mut editable, mut shown)) = inputs.get_mut(input) else {
        return;
    };
    if code == KeyCode::Enter {
        let value = editable.value().to_string();
        shown.text = value.clone();
        commands.trigger(ValueChange {
            source: input,
            value,
            is_final: true,
        });
    } else {
        replace_text(&mut editable, &shown.text);
    }
    commands.entity(input).insert(Settled);
    if focus.get() == Some(input) {
        focus.clear();
    }
}

fn on_focus_lost(
    lost: On<FocusLost>,
    mut inputs: Query<(&EditableText, &mut ShownText, Has<Settled>)>,
    mut commands: Commands,
) {
    let input = lost.event_target();
    let Ok((editable, mut shown, settled)) = inputs.get_mut(input) else {
        return;
    };
    if settled {
        commands.entity(input).remove::<Settled>();
        return;
    }
    let value = editable.value().to_string();
    if value == shown.text {
        return;
    }
    shown.text = value.clone();
    commands.trigger(ValueChange {
        source: input,
        value,
        is_final: true,
    });
}

fn writeback_string(
    change: On<ValueChange<String>>,
    bindings: Query<&InspectorBinding>,
    mut commands: Commands,
) {
    if let Ok(binding) = bindings.get(change.source) {
        queue_edit(
            &mut commands,
            binding,
            Box::new(change.value.clone()),
            change.is_final,
        );
    }
}
