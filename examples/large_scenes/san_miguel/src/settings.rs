//! Light-control panel (top right): sun azimuth/elevation sliders and an
//! emissive boost slider.
//!
//! San Miguel is a daylit courtyard with little authored emissive, so the boost
//! defaults to 1.0 (no change). It is kept for parity with the other large-scene
//! examples and to crank any emissive materials should the scene gain them; the
//! slider scales every `StandardSolariMaterial`'s emissive against its as-loaded value.

use std::collections::HashMap;

use bevy::{
    asset::AssetId,
    camera_controller::free_camera::FreeCameraState,
    feathers::{
        self,
        controls::{FeathersCheckbox, FeathersSlider},
        theme::{ThemeBackgroundColor, ThemeTextColor, ThemedText},
    },
    light::light_consts::lux,
    prelude::*,
    solari::prelude::{CameraReset, SolariCamera, SolariDirectionLight, StandardSolariMaterial},
    ui::Checked,
    ui_widgets::{
        checkbox_self_update, slider_self_update, SliderPrecision, SliderStep, ValueChange,
    },
};

/// The sun entity the sliders steer.
#[derive(Component)]
pub struct Sun;

#[derive(Resource)]
pub struct LightSettings {
    /// Whether the sun emits at all.
    pub sun_enabled: bool,
    /// Sun heading in degrees (rotation about world Y).
    pub sun_azimuth: f32,
    /// Sun height above the horizon in degrees; negative is twilight.
    pub sun_elevation: f32,
    /// Multiplier applied to every material's as-loaded emissive color.
    pub emissive_boost: f32,
}

impl Default for LightSettings {
    fn default() -> Self {
        Self {
            sun_enabled: true,
            sun_azimuth: -23.0,
            sun_elevation: 63.0,
            emissive_boost: 1.0,
        }
    }
}

pub struct LightSettingsPlugin;

impl Plugin for LightSettingsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LightSettings>()
            .add_systems(Startup, (|| settings_ui()).spawn())
            .add_systems(Update, (apply_sun, apply_emissive, reset_on_change));
    }
}

/// Any lighting change invalidates accumulated/temporal history — request a
/// camera reset for the frame.
fn reset_on_change(
    settings: Res<LightSettings>,
    mut cameras: Query<&mut CameraReset, With<SolariCamera>>,
) {
    if !settings.is_changed() || settings.is_added() {
        return;
    }
    for mut reset in &mut cameras {
        reset.history = true;
    }
}

fn apply_sun(
    settings: Res<LightSettings>,
    mut sun: Query<(&mut Transform, &mut SolariDirectionLight), With<Sun>>,
) {
    if !settings.is_changed() {
        return;
    }
    for (mut transform, mut light) in &mut sun {
        transform.rotation = Quat::from_euler(
            EulerRot::YXZ,
            settings.sun_azimuth.to_radians(),
            -settings.sun_elevation.to_radians(),
            0.0,
        )
        .to_precision();
        let illuminance = if settings.sun_enabled {
            lux::FULL_DAYLIGHT
        } else {
            0.0
        };
        if light.illuminance != illuminance {
            light.illuminance = illuminance;
        }
    }
}

/// Scale every material's emissive by the boost, relative to its as-loaded
/// value. Materials stream in with the scene, so newly seen ones are recorded
/// and boosted every frame; existing ones are only rewritten when the slider
/// moves (a `get_mut` marks the asset modified, so avoid spurious writes).
fn apply_emissive(
    settings: Res<LightSettings>,
    mut materials: ResMut<Assets<StandardSolariMaterial>>,
    mut as_loaded: Local<HashMap<AssetId<StandardSolariMaterial>, LinearRgba>>,
) {
    let changed = settings.is_changed();
    let ids: Vec<_> = materials.ids().collect();
    for id in ids {
        let Some(material) = materials.get(id) else {
            continue;
        };
        if material.emissive == LinearRgba::BLACK && !as_loaded.contains_key(&id) {
            continue;
        }
        let known = as_loaded.contains_key(&id);
        let base = *as_loaded.entry(id).or_insert(material.emissive);
        if known && !changed {
            continue;
        }
        let mut want = base * settings.emissive_boost;
        want.alpha = base.alpha;
        if material.emissive != want {
            materials.get_mut(id).unwrap().emissive = want;
        }
    }
}

fn settings_ui() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            top: px(10),
            right: px(10),
            padding: px(8),
        }
        ThemeBackgroundColor(feathers::tokens::WINDOW_BG)
        on(|_: On<Pointer<Over>>, mut free_camera_state: Single<&mut FreeCameraState>| {
            free_camera_state.enabled = false;
        })
        on(|_: On<Pointer<Out>>, mut free_camera_state: Single<&mut FreeCameraState>| {
            free_camera_state.enabled = true;
        })
        Children [(
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Stretch,
                justify_content: JustifyContent::Start,
                row_gap: px(8),
                min_width: px(180),
            }
            Children [
                Text("Lights"),
                (
                    @FeathersCheckbox {
                        @caption: bsn! { Text("Sun") ThemedText }
                    }
                    Checked
                    on(checkbox_self_update)
                    on(|change: On<ValueChange<bool>>, mut settings: ResMut<LightSettings>| {
                        settings.sun_enabled = change.value;
                    })
                ),
                (
                    Text("Sun azimuth")
                    TextFont { font_size: FontSize::Px(14.0) }
                    ThemeTextColor(feathers::tokens::CHECKBOX_TEXT)
                ),
                (
                    @FeathersSlider {
                        @min: -180.0,
                        @max: 180.0,
                        @value: -23.0,
                    }
                    SliderStep(1.0)
                    SliderPrecision(0)
                    on(slider_self_update)
                    on(|change: On<ValueChange<f32>>, mut settings: ResMut<LightSettings>| {
                        settings.sun_azimuth = change.value;
                    })
                ),
                (
                    Text("Sun elevation")
                    TextFont { font_size: FontSize::Px(14.0) }
                    ThemeTextColor(feathers::tokens::CHECKBOX_TEXT)
                ),
                (
                    @FeathersSlider {
                        @min: -15.0,
                        @max: 89.0,
                        @value: 63.0,
                    }
                    SliderStep(1.0)
                    SliderPrecision(0)
                    on(slider_self_update)
                    on(|change: On<ValueChange<f32>>, mut settings: ResMut<LightSettings>| {
                        settings.sun_elevation = change.value;
                    })
                ),
                (
                    Text("Emissive boost")
                    TextFont { font_size: FontSize::Px(14.0) }
                    ThemeTextColor(feathers::tokens::CHECKBOX_TEXT)
                ),
                (
                    @FeathersSlider {
                        @min: 0.0,
                        @max: 100_000.0,
                        @value: 1.0,
                    }
                    SliderStep(500.0)
                    SliderPrecision(0)
                    on(slider_self_update)
                    on(|change: On<ValueChange<f32>>, mut settings: ResMut<LightSettings>| {
                        settings.emissive_boost = change.value;
                    })
                ),
            ]
        )]
    }
}
