// Under `solari` several raster-only plugins/systems (wireframe, atmosphere) are
// gated out, leaving their imports/fns conditionally unused.
#![cfg_attr(feature = "solari", allow(dead_code, unused_imports))]
//! A procedurally generated city.
//!
//! This scene is intended to be an attractive, fairly realistic stress test of Bevy's capacity
//! to model extremely large scenes.
//! As a result, the complexity is higher than in most examples or benchmarks —
//! we want to use a large number of features so that pathological paths
//! are caught during development, rather than by end users.

use argh::FromArgs;
use assets::{load_assets, CityAssets};
use bevy::{
    camera::{Exposure, Hdr, visibility::NoCpuCulling}, camera_controller::free_camera::{FreeCamera, FreeCameraPlugin}, color::palettes::css::WHITE, dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig}, diagnostic::FrameTimeDiagnosticsPlugin, feathers::{FeathersPlugins, dark_theme::create_dark_theme, theme::UiTheme}, light::{
        Atmosphere,
        atmosphere::{Falloff, PhaseFunction, ScatteringMedium, ScatteringTerm}
    }, pbr::wireframe::{WireframeConfig, WireframePlugin}, post_process::bloom::Bloom, prelude::*, window::{PresentMode, WindowResolution}, winit::WinitSettings, world_serialization::WorldInstanceReady
};

// Only used by the rasterized (non-Solari) camera.
#[cfg(not(feature = "solari"))]
use bevy::{
    pbr::{
        AtmosphereSettings
    },
    anti_alias::taa::TemporalAntiAliasing, light::AtmosphereEnvironmentMapLight, pbr::ContactShadows,
};

#[cfg(feature = "solari")]
use bevy::{
    camera::CameraMainTextureUsages,
    core_pipeline::Skybox,
    light::cluster::ClusterConfig,
    render::render_resource::TextureUsages,
    solari::prelude::*,
};

use bevy::render::view::screenshot::{save_to_disk, Screenshot};

use crate::generate_city::{spawn_city, LampAssets};
use crate::{
    assets::{merge_car_meshes, strip_base_url},
    settings::{settings_ui, Settings, CITY_SIZE_RANGE},
};

mod assets;
mod generate_city;
mod settings;

#[derive(FromArgs, Resource, Clone)]
/// Config
pub struct Args {
    /// seed
    #[argh(option, default = "42")]
    seed: u64,

    /// size
    #[argh(option, default = "100")]
    size: u32,

    /// adds NoCpuCulling to all meshes
    #[argh(switch)]
    no_cpu_culling: bool,

    /// spawn emissive street lamps along the roads (two per block — a
    /// many-lights stress source for the solari ReSTIR path); on by default,
    /// pass `--lights false` to disable
    #[argh(option, default = "false")]
    lights: bool,
}

fn main() {
    let args: Args = argh::from_env();

    let mut app = App::new();
        // DLSS needs its project id inserted before RenderPlugin (DlssInitPlugin
        // reads it during render init). `solari` enables `bevy/dlss`.
        #[cfg(feature = "dlss")]
        app.insert_resource(bevy::anti_alias::dlss::DlssProjectId(
            bevy::asset::uuid::uuid!("a0e6c8d2-1f3b-4c5a-9e7d-2b4f6a8c0e1d"),
        ));
        let default_plugins = DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_city".into(),
                resolution: WindowResolution::new(1920, 1080).with_scale_factor_override(1.0),
                present_mode: PresentMode::AutoNoVsync,
                position: WindowPosition::Centered(MonitorSelection::Primary),
                ..default()
            }),
            ..default()
        });
        // Under `solari` the GPU transform table drives the RT scene (transform
        // columns → Jacobi → gather into the instance transforms), so
        // bevy_transform's CPU propagation is pure overhead — disable it. CPU
        // `GlobalTransform` is restored below only for the small set that still
        // reads it CPU-side (camera/sun/atmosphere + the UI tree).
        // Under `solari` the full-RT path replaces the raster mesh/material stack,
        // so disable `PbrPlugin` (bevy_solari owns its material/lights + vendors the
        // DfgLut + pbr shader helpers) and the render-debug overlay (DefaultPlugins
        // adds it with the bevy_pbr feature). `TransformPlugin` is GPU-driven.
        #[cfg(feature = "solari")]
        let default_plugins = default_plugins
            .disable::<bevy::transform::TransformPlugin>()
            .disable::<bevy::pbr::PbrPlugin>()
            .disable::<bevy::dev_tools::render_debug::RenderDebugOverlayPlugin>();

        app.add_plugins((
            default_plugins,
            FreeCameraPlugin,
            FeathersPlugins,
            // Wireframe needs the raster mesh pipeline (gone with PbrPlugin).
            #[cfg(not(feature = "solari"))]
            WireframePlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
            FpsOverlayPlugin {
                config: FpsOverlayConfig {
                    frame_time_graph_config: FrameTimeGraphConfig {
                        enabled: true,
                        target_fps: 240.0,
                        min_fps: 60.0,
                    },
                    ..default()
                },
            },    
            #[cfg(feature = "solari")]  SolariPlugin     
        ));
    

        app.insert_resource(args.clone())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(WinitSettings::continuous())
        .insert_resource(Settings {
            city_size: args.size.clamp(CITY_SIZE_RANGE.0, CITY_SIZE_RANGE.1),
            ..default()
        })
        .init_resource::<CaptureReady>()
        .insert_resource(UiTheme(create_dark_theme()))
        .insert_resource(WireframeConfig {
            global: false,
            default_color: WHITE.into(),
            ..default()
        })
        // Like in many realistic large scenes, many of the objects don't move
        // We can accelerate transform propagation by optimizing for this case
        .insert_resource(StaticTransformOptimizations::Enabled)
        .add_message::<CityAssetsLoaded>()
        .add_message::<CityAssetsReady>()
        .add_message::<CitySpawned>()
        // `spawn_atmosphere` needs `Assets<ScatteringMedium>` (registered via
        // PbrPlugin's AtmospherePlugin) — gone under solari, where the RT path
        // doesn't sample the raster atmosphere anyway.
        .add_systems(Startup, (
            scene.spawn(),
            #[cfg(not(feature = "solari"))]
            spawn_atmosphere,
            load_assets,
            generate_city::setup_lamp_assets,
        ))
        .add_systems(
            Update,
            (
                simulate_cars,
                burst_screenshots,
                settings::update_city_info,
                update_loading_screen,
                process_assets.run_if(on_message::<CityAssetsLoaded>),
                on_city_assets_ready.run_if(on_message::<CityAssetsReady>),
                (
                    add_no_cpu_culling,
                    on_city_spawned,
                    {
                        let city_size = args.size.clamp(CITY_SIZE_RANGE.0, CITY_SIZE_RANGE.1);
                        (move || settings_ui(city_size)).spawn()
                    },
                    arm_capture_ready,
                )
                    .run_if(on_message::<CitySpawned>),
                signal_capture_ready,
                #[cfg(feature = "solari")]
                (
                    convert_meshes_to_raytracing,
                    convert_standard_materials_to_solari,
                    mark_city_static,
                    // Swap to `add_solari_environment_map` to test the pisa skybox
                    // instead of the baked atmosphere on the camera.
                    // add_solari_environment_map,
                ),
            ),
        )
        .add_observer(add_no_cpu_culling_on_scene_ready)
        .run();
}

fn scene() -> impl SceneList {
    bsn_list![camera(), sun(), loading_screen()]
}

/// Tag the static city bulk (every non-car ray-traced mesh) with
/// [`TransformStatic`], so the GPU transform extract skips them in its per-frame
/// `Changed<Transform>` scan — only the moving cars (and the camera) are scanned.
/// `Without<TransformStatic>` makes this archetype-filtered: each mesh is tagged
/// once (as it converts to `RaytracingMesh3d`) and the system idles to zero work
/// thereafter. Cars are excluded via `Without<Car>` so their motion still updates.
#[cfg(feature = "solari")]
fn mark_city_static(
    mut commands: Commands,
    query: Query<Entity, (With<RaytracingMesh3d>, Without<Car>, Without<TransformStatic>)>,
) {
    for entity in &query {
        commands.entity(entity).insert(TransformStatic);
    }
}

/// Give the solari camera a [`Skybox`] (sky) once it exists. The pathtracer
/// samples its cube in the ray direction on a miss instead of returning black.
/// `Without<Skybox>` makes this archetype-filtered — it runs once and then idles.
/// `brightness` is in cd/m² (solari single-exposes it like the rest of the scene),
/// so it's a calibrated value, not a finicky multiplier — tune to taste. (Only the
/// `Pathtrace` view samples it today; the realtime ReSTIR path doesn't yet.)
#[cfg(feature = "solari")]
#[expect(dead_code, reason = "toggled in for skybox testing vs the baked atmosphere")]
fn add_solari_environment_map(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    camera: Query<Entity, (With<SolariCamera>, Without<Skybox>)>,
) {
    for entity in &camera {
        commands.entity(entity).insert(Skybox {
            image: Some(asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2")),
            brightness: 3000.0,
            ..default()
        });
    }
}

#[cfg(not(feature = "solari"))]
fn camera() -> impl Scene {        
    bsn! {
        Camera3d
        Hdr
        template_value(Transform::from_xyz(15.0, 10.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y))
        FreeCamera
        AtmosphereSettings {
            // Reduce the default max distance in the aerial view LUT
            // to 16km to approximately fit the size of the city. This way the aerial perspective
            // gets more detail and has less banding artifacts compared to the 32km default.
            aerial_view_lut_max_distance: 1.6e4,
        }
        // The directional light illuminance used in this scene is
        // quite bright, so raising the exposure compensation helps
        // bring the scene to a nicer brightness range.
        Exposure::OVERCAST
        // Bloom gives the sun a much more natural look.
        Bloom::NATURAL
        // Enables the atmosphere to drive reflections and ambient lighting (IBL) for this view
        AtmosphereEnvironmentMapLight
        Msaa::Off
        TemporalAntiAliasing
        ContactShadows
    }
}

#[cfg(feature = "solari")]
fn camera() -> impl Scene {        
    bsn! {
        Camera3d
        Hdr
        template_value(Transform::from_xyz(15.0, 10.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y))
        FreeCamera
        Exposure::OVERCAST
        //Bloom::NATURAL
        Msaa::Off
        template_value(SolariCamera::default())
        // Self-contained single-scattering sky, baked to a cube each frame and
        // sampled on a ray miss (background + IBL). Sun = the `SolariDirectionLight`.
        // Low-lying fog matching the raster path's `spawn_atmosphere`: ~12 km
        template_value(SolariAtmosphere::default())
        // Global height fog + god rays are opt-in (and view-dependent cost: a
        // ground-level camera traces a sun shadow ray per march step). Uncomment
        // for ~660-unit ground visibility + a ~5.5-unit fog layer at y = 0.
        // template_value(SolariGlobalFog { visibility: 1200.0, fog_height: 5.5, fog_base: 0.0, ..default() })
        template_value(ClusterConfig::None)
        template_value(CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING))        
          
        //template_value(SolariDebugView::Pathtrace)        
    }
}

fn loading_screen() -> impl Scene {
    bsn! {
        LoadingScreen
        Node {
            position_type: PositionType::Absolute,
            width: percent(100),
            height: percent(100),
        }
        BackgroundColor(Color::BLACK)
        Children [
            Node {
                position_type: PositionType::Absolute,
                top: percent(50),
                left: percent(20),
                right: percent(20),
                height: vh(40),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::FlexStart,
                overflow: Overflow::scroll_y(),
            }
            Children [
                (
                    LoadingText
                    Text("Loading...")
                    TextFont {
                        font_size: FontSize::Px(24.0),
                    }
                ),
                (
                    LoadingPaths
                    Text
                    TextFont {
                        font_size: FontSize::Px(14.0),
                    }
                ),
            ]
        ]
    }
}

#[cfg(not(feature = "solari"))]
fn sun() -> impl Scene {
    bsn! {
        DirectionalLight {
            shadow_maps_enabled: {Settings::default().shadow_maps_enabled},
            contact_shadows_enabled: {Settings::default().contact_shadows_enabled},
            illuminance: light_consts::lux::RAW_SUNLIGHT,
        }
        template_value(Transform::from_xyz(1.0, 0.15, 1.0).looking_at(Vec3::ZERO, Vec3::Y))
    }
}

// Under solari the sun is a `SolariDirectionLight` (own light type, direction
// resolved from the GPU transform table — no bevy_pbr `DirectionalLight`).
#[cfg(feature = "solari")]
fn sun() -> impl Scene {
    bsn! {
        SolariDirectionLight {
            illuminance: light_consts::lux::RAW_SUNLIGHT,
        }
        template_value(Transform::from_xyz(1.0, 0.15, 1.0).looking_at(Vec3::ZERO, Vec3::Y))
    }
}

/// Spawns the earth atmosphere plus an extra near-ground fog term.
fn spawn_atmosphere(
    mut commands: Commands,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    let mut earth_medium = ScatteringMedium::default();

    // Same 60 km atmosphere height as `ScatteringMedium::earth`
    const ATMOSPHERE_REF_HEIGHT_KM: f32 = 60.0;

    // The scale height of haze is set to 100 meters providing a low-lying dense fog layer.
    const HAZE_SCALE_HEIGHT_KM: f32 = 0.1;

    // Fog has high albedo and very low absorption resulting in a white color.
    const HAZE_SINGLE_SCATTER_ALBEDO: f32 = 0.99;

    // Distance at which contrast falls low enough to be indistinguishable from the sky.
    // known as Meteorological Optical Range
    const HAZE_VISIBILITY_KM: f32 = 12.0;

    // Koschmieder relation to calculate the extinction coefficient for the medium in m^-1 units.
    let beta_ext = (3.912 / HAZE_VISIBILITY_KM) * 1e-3;

    // Add the fog to the earth medium as an additional scattering term.
    earth_medium.terms.push(ScatteringTerm {
        absorption: Vec3::splat(beta_ext * (1.0 - HAZE_SINGLE_SCATTER_ALBEDO)),
        scattering: Vec3::splat(beta_ext * HAZE_SINGLE_SCATTER_ALBEDO),
        falloff: Falloff::Exponential {
            scale: HAZE_SCALE_HEIGHT_KM / ATMOSPHERE_REF_HEIGHT_KM,
        },
        // Fog is approximated as a mie scatterer with this asymmetry factor
        phase: PhaseFunction::Mie { asymmetry: 0.76 },
    });
    let earth_atmosphere = Atmosphere::earth(scattering_mediums.add(earth_medium));

    // This scale means that 1 city block in this scene will be roughly 100 meters relative to the atmosphere.
    let scale = 1.0 / 20.0;
    commands.spawn((
        earth_atmosphere.clone(),
        Transform::from_scale(Vec3::splat(scale))
            .with_translation(-Vec3::Y * earth_atmosphere.inner_radius * scale),
    ));
}

#[derive(Component, Default, Clone)]
struct LoadingScreen;
#[derive(Component, Default, Clone)]
struct LoadingText;
#[derive(Component, Default, Clone)]
struct LoadingPaths;

/// Triggers when all the assets managed in [`CityAssets`] are loaded
#[derive(Message)]
struct CityAssetsLoaded;
/// Triggers when all the assets are done loading and have been processed
#[derive(Message)]
struct CityAssetsReady;
/// Triggers once all the city blocks have been spawned
#[derive(Message)]
struct CitySpawned;

#[allow(clippy::type_complexity)]
fn update_loading_screen(
    mut commands: Commands,
    assets: Res<CityAssets>,
    asset_server: Res<AssetServer>,
    mut loading_text: Query<&mut Text, With<LoadingText>>,
    mut loading_paths: Query<(Entity, &mut Text), (With<LoadingPaths>, Without<LoadingText>)>,
) {
    let Ok(mut text) = loading_text.single_mut() else {
        return;
    };
    let Ok((paths_entity, mut paths_text)) = loading_paths.single_mut() else {
        return;
    };
    let mut paths = vec![];
    for untyped in &assets.untyped_assets {
        if let Some(path) = asset_server.get_path(untyped) {
            let state = asset_server.is_loaded_with_dependencies(untyped);
            if !state {
                paths.push(strip_base_url(path.to_string()));
            }
        }
    }
    if paths.is_empty() {
        commands.entity(paths_entity).despawn();
        text.0 = "Processing assets...".into();
        // Use a Message instead of an Event so asset processing only starts on the next frame
        commands.write_message(CityAssetsLoaded);
    } else {
        text.0 = format!(
            "Loading assets: {}/{}",
            assets.untyped_assets.len() - paths.len(),
            assets.untyped_assets.len(),
        );
        paths.reverse();
        paths_text.0 = paths.join("\n");
    }
}

/// Runs after the assets are loaded. For now, this will merge all the meshes for each car gltf into
/// a single mesh. This is necessary because the tires are separate meshes and this increases the
/// amount of meshes bevy has to process every frame for no benefits.
///
/// Eventually, this will also be used for things like generating LODs
fn process_assets(
    mut commands: Commands,
    mut city_assets: ResMut<CityAssets>,
    mut world_assets: ResMut<Assets<WorldAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    merge_car_meshes(&mut city_assets, &mut world_assets, &mut meshes);

    // Use a Message instead of an Event so spawning the city happens in the next frame
    commands.write_message(CityAssetsReady);
}

fn on_city_assets_ready(
    mut commands: Commands,
    city_assets: Res<CityAssets>,
    lamps: Option<Res<LampAssets>>,
    args: Res<Args>,
    mut loading_text: Query<&mut Text, With<LoadingText>>,
) {
    let Ok(mut text) = loading_text.single_mut() else {
        return;
    };
    text.0 = "Spawning city...".into();

    spawn_city(
        &mut commands,
        &city_assets,
        lamps.as_deref(),
        args.seed,
        args.size,
    );
    commands.write_message(CitySpawned);
}

fn on_city_spawned(
    mut commands: Commands,
    loading_screen: Option<Single<Entity, With<LoadingScreen>>>,
) {
    let Some(loading_screen) = loading_screen else {
        return;
    };
    commands.entity(*loading_screen).despawn();
}

/// Detects when the scene has finished spawning + streaming so profiling
/// captures measure the settled frame, not the (very long, at size 100) spawn
/// spike. `capture.sh` waits for the `CAPTURE_READY` log line.
#[derive(Resource, Default)]
struct CaptureReady {
    /// Set once the city's spawn commands have been issued; entities + streamed
    /// scenes still settle over the following frames.
    armed: bool,
    last_count: usize,
    stable_frames: u32,
    done: bool,
}

/// Arm the detector when the city is spawned (commands issued).
fn arm_capture_ready(mut ready: ResMut<CaptureReady>) {
    ready.armed = true;
}

/// Once the spatial-entity count has plateaued for ~2s after spawn, log
/// `CAPTURE_READY` (once). Car movement changes transforms, not entity counts,
/// so the count is stable in steady state.
fn signal_capture_ready(
    spatial: Query<(), With<GlobalTransform>>,
    mut ready: ResMut<CaptureReady>,
) {
    if ready.done || !ready.armed {
        return;
    }
    let count = spatial.iter().count();
    if count > 0 && count == ready.last_count {
        ready.stable_frames += 1;
        if ready.stable_frames >= 120 {
            info!("CAPTURE_READY (settled at {count} spatial entities)");
            ready.done = true;
        }
    } else {
        ready.last_count = count;
        ready.stable_frames = 0;
    }
}

/// F10: capture 10 consecutive frames to `/tmp/bevy_city_burst/` — for
/// diffing temporal stability of a fixed view.
#[derive(Resource, Default)]
struct BurstCapture {
    remaining: u32,
    index: u32,
}

fn burst_screenshots(
    keys: Res<ButtonInput<KeyCode>>,
    mut burst: Local<BurstCapture>,
    mut commands: Commands,
) {
    if keys.just_pressed(KeyCode::F10) {
        let _ = std::fs::create_dir_all("/tmp/bevy_city_burst");
        burst.remaining = 10;
        burst.index = 0;
        info!("burst capture: 10 frames -> /tmp/bevy_city_burst/");
    }
    if burst.remaining > 0 {
        burst.remaining -= 1;
        let path = format!("/tmp/bevy_city_burst/frame_{:02}.png", burst.index);
        burst.index += 1;
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }
}

#[derive(Component)]
struct Road {
    start: Vec3,
    end: Vec3,
}

#[derive(Component)]
struct Car {
    offset: Vec3,
    distance_traveled: f32,
    dir: f32,
}


/// Do a very naive traffic simulation. This will only move the car to the end of the road then
/// spawn it back at the start.
///
/// Eventually this will be a more complex traffic simulation that should stress the ECS
fn simulate_cars(
    settings: Res<Settings>,
    roads: Query<&Road>,
    mut cars: Query<(&mut Car, &mut Transform, &ChildOf), Without<Road>>,
    time: Res<Time>,
) {
    if !settings.simulate_cars {
        return;
    }
    let speed = 1.5;
    let dt = time.delta_secs();

    // Parallel over cars (each mutated by exactly one thread): a car looks up its
    // parent road via a read-only `get` — safe to share across the `par_iter`.
    // Replaces the serial roads → children → `get_mut` nested loop (~2.7ms over
    // ~120k cars at size 100).
    cars.par_iter_mut()
        .for_each(|(mut car, mut car_transform, child_of)| {
            let Ok(road) = roads.get(child_of.parent()) else {
                return;
            };
            car.distance_traveled += speed * dt;
            let road_len = (road.end - road.start).length();
            if car.distance_traveled > road_len {
                car.distance_traveled = 0.0;
            }
            let direction = (road.end - road.start).normalize() * car.dir;
            let progress = car.distance_traveled / road_len;
            car_transform.translation =
                (road.start + car.offset) + direction * road_len * progress;
        });
}

/// Adds [`NoCpuCulling`] to all meshes in the scene after the city is done spawning
fn add_no_cpu_culling(
    mut commands: Commands,
    meshes: Query<Entity, (With<Mesh3d>, Without<NoCpuCulling>)>,
    args: Res<Args>,
) {
    if args.no_cpu_culling {
        for entity in meshes.iter() {
            commands.entity(entity).insert(NoCpuCulling);
        }
    }
}

/// Adds [`NoCpuCulling`] to all meshes in all scenes after the city is done spawning
///
/// This is required because a few assets are spawned using a [`WorldAssetRoot`] instead of directly
/// spawning a [`Mesh`]
fn add_no_cpu_culling_on_scene_ready(
    scene_ready: On<WorldInstanceReady>,
    mut commands: Commands,
    children: Query<&Children>,
    meshes: Query<(), (With<Mesh3d>, Without<NoCpuCulling>)>,
    args: Res<Args>,
) {
    if args.no_cpu_culling {
        for descendant in children.iter_descendants(scene_ready.entity) {
            if meshes.get(descendant).is_ok() {
                commands.entity(descendant).insert(NoCpuCulling);
            }
        }
    }
}
