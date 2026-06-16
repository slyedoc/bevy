use bevy::{
    camera::visibility::NoCpuCulling,
    camera_controller::free_camera::FreeCameraState,
    feathers::{
        self,
        controls::{FeathersButton, FeathersCheckbox, FeathersSlider},
        theme::{ThemeBackgroundColor, ThemeTextColor, ThemedText},
    },
    pbr::wireframe::WireframeConfig,
    prelude::*,
    ui::Checked,
    ui_widgets::{
        checkbox_self_update, slider_self_update, Activate, SliderPrecision, SliderStep,
        ValueChange,
    },
};
use rand::RngExt;

use crate::assets::CityAssets;
use crate::generate_city::{spawn_city, CityRoot, LampAssets};

/// Smallest / largest city the regenerate slider offers (blocks per side).
pub const CITY_SIZE_RANGE: (u32, u32) = (3, 150);

/// On the scene-info line under "Regenerate City"; [`update_city_info`] keeps
/// it current.
#[derive(Component, Default, Clone)]
pub struct CityInfoText;

/// Keep the scene counts current. Counting is cheap (dense-query size hints),
/// and skipping the write when unchanged avoids re-laying-out the text.
pub fn update_city_info(
    entities: Query<()>,
    cars: Query<(), With<crate::Car>>,
    mut text: Query<&mut Text, With<CityInfoText>>,
) {
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    let want = format!(
        "Entities: {}\nMoving: {}",
        entities.iter().count(),
        cars.iter().count(),
    );
    if text.0 != want {
        text.0 = want;
    }
}

#[derive(Resource)]
pub struct Settings {
    pub simulate_cars: bool,
    pub shadow_maps_enabled: bool,
    pub contact_shadows_enabled: bool,
    pub wireframe_enabled: bool,
    pub cpu_culling: bool,
    /// Blocks per side for "Regenerate City"; seeded from `--size` at startup.
    pub city_size: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            simulate_cars: true,
            shadow_maps_enabled: true,
            contact_shadows_enabled: true,
            wireframe_enabled: false,
            cpu_culling: true,
            city_size: 32,
        }
    }
}

pub fn settings_ui(city_size: u32) -> impl Scene {
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
            }
            Children [
                Text("Settings"),
                (
                    @FeathersCheckbox {
                        @caption: bsn! { Text("Simulate Cars") ThemedText }
                    }
                    Checked
                    on(checkbox_self_update)
                    on(|change: On<ValueChange<bool>>, mut settings: ResMut<Settings>| {
                        settings.simulate_cars = change.value;
                    })
                ),
                (
                    @FeathersCheckbox {
                        @caption: bsn! { Text("Shadow maps enabled") ThemedText }
                    }
                    Checked
                    on(checkbox_self_update)
                    on(
                        |change: On<ValueChange<bool>>,
                         mut settings: ResMut<Settings>,
                         mut directional_lights: Query<&mut DirectionalLight>| {
                            settings.shadow_maps_enabled = change.value;
                            for mut light in &mut directional_lights {
                                light.shadow_maps_enabled = change.value;

                            }
                        }
                    )
                ),
                (
                    @FeathersCheckbox {
                        @caption: bsn! { Text("Contact shadows enabled") ThemedText }
                    }
                    Checked
                    on(checkbox_self_update)
                    on(
                        |change: On<ValueChange<bool>>,
                         mut settings: ResMut<Settings>,
                         mut directional_lights: Query<&mut DirectionalLight>| {
                            settings.contact_shadows_enabled = change.value;
                            for mut light in &mut directional_lights {
                                light.contact_shadows_enabled = change.value;

                            }
                        }
                    )
                ),
                (
                    @FeathersCheckbox {
                        @caption: bsn! { Text("Wireframe Enabled") ThemedText }
                    }
                    on(checkbox_self_update)
                    on(
                        |change: On<ValueChange<bool>>,
                         mut settings: ResMut<Settings>,
                         mut wireframe_config: ResMut<WireframeConfig>| {
                            settings.wireframe_enabled = change.value;
                            wireframe_config.global = change.value;
                        }
                    )
                ),
                (
                    @FeathersCheckbox {
                        @caption: bsn! { Text("CPU culling") ThemedText }
                    }
                    Checked
                    on(checkbox_self_update)
                    on(
                        |change: On<ValueChange<bool>>,
                         mut settings: ResMut<Settings>,
                         mut commands: Commands,
                         meshes: Query<Entity, With<Mesh3d>>| {
                            settings.cpu_culling = change.value;

                            for entity in meshes.iter() {
                                if settings.cpu_culling {
                                    commands.entity(entity).remove::<NoCpuCulling>();
                                } else {
                                    commands.entity(entity).insert(NoCpuCulling);
                                }
                            }
                        }
                    )
                ),
                (
                    Text("Size")
                    TextFont { font_size: FontSize::Px(14.0) }
                    ThemeTextColor(feathers::tokens::CHECKBOX_TEXT)
                ),
                (
                    @FeathersSlider {
                        @min: {CITY_SIZE_RANGE.0 as f32},
                        @max: {CITY_SIZE_RANGE.1 as f32},
                        @value: {city_size as f32},
                    }
                    SliderStep(1.0)
                    SliderPrecision(0)
                    on(slider_self_update)
                    on(|change: On<ValueChange<f32>>, mut settings: ResMut<Settings>| {
                        settings.city_size = change.value.round() as u32;
                    })
                ),
                (
                    @FeathersButton {
                        @caption: bsn! { Text("Regenerate City") ThemedText }
                    }
                    on(
                        |_activate: On<Activate>,
                         mut commands: Commands,
                         city_root: Single<Entity, With<CityRoot>>,
                         assets: Res<CityAssets>,
                         lamps: Option<Res<LampAssets>>,
                         settings: Res<Settings>| {
                            commands.entity(*city_root).despawn();

                            let mut rng = rand::rng();
                            let seed = rng.random::<u64>();
                            println!("new seed: {seed}");
                            spawn_city(
                                &mut commands,
                                &assets,
                                lamps.as_deref(),
                                seed,
                                settings.city_size,
                            );
                        }
                    )
                ),
                (
                    CityInfoText
                    Text("")
                    TextFont { font_size: FontSize::Px(14.0) }
                    ThemeTextColor(feathers::tokens::CHECKBOX_TEXT)
                ),
            ]
        )]
    }
}
