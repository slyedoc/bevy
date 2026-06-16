//! Bevy Solari demo — ray-traced hair via linear swept spheres.
//!
//! Loads a Cem Yuksel `.hair` groom (the de-facto hair test asset used by pbrt /
//! Mitsuba / NVIDIA RTXCR) and renders it with the full ray-traced Solari path.
//! Each strand is built into an NV linear-swept-sphere BLAS
//! (`VK_NV_ray_tracing_linear_swept_spheres`) and shaded with a Chiang fiber BSDF.
//! A small Feathers panel edits the fiber color live.
//!
//! Get a model from <https://www.cemyuksel.com/research/hairmodels/> (e.g.
//! `wCurly.hair`) and drop it in `assets/models/hair/`. With no file, set
//! [`HAIR_MODEL`] to `None` for the procedural fur ball ([`fur_ball`]).
//!
//! Requires a Blackwell / RTX 50-series GPU + driver >= 572.63. On any other
//! adapter the extension isn't advertised, hair is skipped (a warning is
//! logged), and the rest of the scene renders normally.

use bevy::{
    camera::CameraMainTextureUsages,
    camera::{Exposure, Hdr},
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState},
    ecs::VariantDefaults,
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig},
    diagnostic::FrameTimeDiagnosticsPlugin,
    feathers::{
        self,
        controls::{ColorSwatchValue, FeathersColorSwatch, FeathersSlider},
        dark_theme::create_dark_theme,
        theme::{ThemeBackgroundColor, ThemeTextColor, ThemedText, UiTheme},
        FeathersPlugins,
    },
    prelude::*,
    render::render_resource::TextureUsages,
    solari::{
        hair::{hair_absorption_to_color, melanin_absorption},
        prelude::*,
    },
    ui_widgets::{slider_self_update, SliderPrecision, SliderValue},
};
use std::f32::consts::FRAC_PI_2;

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

/// Set to e.g. `Some("models/hair/wCurly.hair")` to load a Cem Yuksel groom; the
/// `.hair` asset loader builds it into an LSS BLAS. `None` ⇒ procedural fur ball.
const HAIR_MODEL: Option<&str> = Some("models/hair/wCurly.hair");

/// Shared edited hair appearance (melanin model), written by the picker controls
/// and applied to the groom + picker widgets by [`sync_hair_appearance`].
#[derive(Resource, Clone, Copy)]
struct HairAppearance {
    /// Overall melanin (0 = platinum → 1 = black).
    melanin: f32,
    /// Eumelanin↔pheomelanin ratio (0 = ashy/brown → 1 = ginger/red).
    redness: f32,
    /// Artistic dye tint (white = none).
    dye: Vec3,
    /// Longitudinal roughness βm (sheen).
    beta_m: f32,
    /// Azimuthal roughness βn (frizz).
    beta_n: f32,
}

impl Default for HairAppearance {
    fn default() -> Self {
        // A natural medium brown.
        Self {
            melanin: 0.55,
            redness: 0.1,
            dye: Vec3::ONE,
            beta_m: 0.3,
            beta_n: 0.3,
        }
    }
}

impl HairAppearance {
    fn material(&self) -> HairMaterial {
        HairMaterial {
            melanin: self.melanin,
            redness: self.redness,
            dye: self.dye,
            longitudinal_roughness: self.beta_m,
            azimuthal_roughness: self.beta_n,
            cuticle_tilt: 0.035,
            ior: 1.55,
        }
    }
}

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "5417916c-0291-4e3f-8f65-326c1858ab96" // Don't copy paste this - generate your own UUID!
    )));

    app.add_plugins((
        DefaultPlugins
            .set(bevy::log::LogPlugin {
                filter: "wgpu=error,naga=warn,bevy_solari=info".into(),
                ..default()
            })
            .disable::<TransformPlugin>()
            .disable::<bevy::pbr::PbrPlugin>()
            .disable::<bevy::dev_tools::render_debug::RenderDebugOverlayPlugin>(),
        SolariPlugin,
        FeathersPlugins,
        FreeCameraPlugin,
        FrameTimeDiagnosticsPlugin::default(),
        FpsOverlayPlugin {
            config: FpsOverlayConfig {
                frame_time_graph_config: FrameTimeGraphConfig {
                    enabled: true,
                    target_fps: 240.0,
                    min_fps: 30.0,
                },
                ..default()
            },
        },
    ))
    .insert_resource(UiTheme(create_dark_theme()))
    .init_resource::<HairAppearance>()
    // `PbrPlugin` is disabled, so re-register `Assets<StandardMaterial>` — the
    // ground plane's material still goes through `convert_standard_materials_to_solari`.
    .init_asset::<StandardMaterial>()
    .add_systems(
        Startup,
        (setup_scene, setup_camera_and_light, hair_color_ui.spawn()),
    )
    .add_systems(
        Update,
        (
            (convert_meshes_to_raytracing, convert_standard_materials_to_solari).chain(),
            (read_hair_sliders, sync_hair_appearance).chain(),
        ),
    )
    .run();
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut hair_assets: ResMut<Assets<HairAsset>>,
) {
    // Ground plane — a regular `Mesh3d`; the `convert_*` systems bake it to a
    // `ClusterMesh` / `SolariMaterial`. Gives the path tracer a surface for the
    // fur's shadow + bounce light, and keeps the cluster PTLAS non-empty (hair
    // instances are appended above the cluster slots).
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(400.0, 400.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.32),
            ..default()
        })),
    ));

    // A loaded `.hair` model, or the procedural fur ball as a fallback.
    let (hair, transform) = match HAIR_MODEL {
        // `wCurly.hair` is authored Z-up (so it lies on its side under Y-up) and
        // centered at (-24, 0.7, 8.5) spanning ~100 units. Stand it up (rotate Z→Y),
        // then recenter on the origin with its base on the floor.
        Some(path) => (
            asset_server.load(path),
            Transform::from_translation(Vec3::new(23.97, 48.41, 0.74))
                .with_rotation(Quat::from_rotation_x(-FRAC_PI_2)),
        ),
        None => (hair_assets.add(fur_ball(1.0, 16_000)), Transform::from_xyz(0.0, 1.2, 0.0)),
    };
    commands.spawn((
        Hair {
            asset: hair,
            material: HairAppearance::default().material(),
        },
        transform,
     ));
}

/// Which hair parameter a slider drives (read from `SliderValue` by
/// [`read_hair_sliders`]).
#[derive(Component, Clone, Copy, Default, VariantDefaults)]
enum HairParam {
    #[default]
    Melanin,
    Redness,
    Sheen,
    Frizz,
}

/// A physically-based hair picker (UE / NVIDIA RTXCR model), `bsn!` scene API.
/// Melanin and redness are independent (as in UE's `GetHairColorFromMelanin`), so
/// they get their own sliders, alongside the two roughness sliders and a swatch
/// previewing the resulting color. Hovering the panel disables the free camera so
/// the sliders are clickable. Each slider uses `slider_self_update` (the feathers
/// thumb-tracking observer); [`read_hair_sliders`] reads the resulting
/// `SliderValue` via its [`HairParam`] tag — the working gallery pattern.
fn hair_color_ui() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            right: px(10),
            top: px(10),
            width: px(240),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
            padding: px(10),
        }
        ThemeBackgroundColor(feathers::tokens::WINDOW_BG)
        on(|_: On<Pointer<Over>>, mut camera: Single<&mut FreeCameraState>| {
            camera.enabled = false;
        })
        on(|_: On<Pointer<Out>>, mut camera: Single<&mut FreeCameraState>| {
            camera.enabled = true;
        })
        Children [
            (Text("Hair color") ThemedText),
            (@FeathersColorSwatch),
            (Text("Melanin (dark)") TextFont { font_size: FontSize::Px(13.0) } ThemeTextColor(feathers::tokens::CHECKBOX_TEXT)),
            (
                @FeathersSlider { @min: 0.0, @max: 1.0, @value: 0.55 }
                SliderPrecision(2)
                HairParam::Melanin
                on(slider_self_update)
            ),
            (Text("Redness") TextFont { font_size: FontSize::Px(13.0) } ThemeTextColor(feathers::tokens::CHECKBOX_TEXT)),
            (
                @FeathersSlider { @min: 0.0, @max: 1.0, @value: 0.1 }
                SliderPrecision(2)
                HairParam::Redness
                on(slider_self_update)
            ),
            (Text("Sheen") TextFont { font_size: FontSize::Px(13.0) } ThemeTextColor(feathers::tokens::CHECKBOX_TEXT)),
            (
                @FeathersSlider { @min: 0.0, @max: 1.0, @value: 0.3 }
                SliderPrecision(2)
                HairParam::Sheen
                on(slider_self_update)
            ),
            (Text("Frizz") TextFont { font_size: FontSize::Px(13.0) } ThemeTextColor(feathers::tokens::CHECKBOX_TEXT)),
            (
                @FeathersSlider { @min: 0.0, @max: 1.0, @value: 0.3 }
                SliderPrecision(2)
                HairParam::Frizz
                on(slider_self_update)
            ),
        ]
    }
}

/// Read each slider's `SliderValue` (kept current by `slider_self_update`) into
/// the shared [`HairAppearance`], routed by the slider's [`HairParam`] tag.
fn read_hair_sliders(
    sliders: Query<(&SliderValue, &HairParam), Changed<SliderValue>>,
    mut appearance: ResMut<HairAppearance>,
) {
    for (value, param) in &sliders {
        match param {
            HairParam::Melanin => appearance.melanin = value.0,
            HairParam::Redness => appearance.redness = value.0,
            HairParam::Sheen => appearance.beta_m = value.0,
            HairParam::Frizz => appearance.beta_n = value.0,
        }
    }
}

/// When [`HairAppearance`] changes, apply it to the groom (resetting the path
/// tracer's accumulation) and refresh the swatch with the resulting fiber color
/// derived from the melanin absorption.
fn sync_hair_appearance(
    appearance: Res<HairAppearance>,
    mut swatches: Query<&mut ColorSwatchValue, With<FeathersColorSwatch>>,
    mut hair: Query<&mut Hair>,
    mut resets: Query<&mut CameraReset, With<SolariCamera>>,
) {
    if !appearance.is_changed() {
        return;
    }
    let a = *appearance;
    let sigma_a = melanin_absorption(a.melanin, a.redness, a.dye, a.beta_n);
    let rc = hair_absorption_to_color(sigma_a, a.beta_n);
    for mut swatch in &mut swatches {
        swatch.0 = Color::srgb(rc.x, rc.y, rc.z);
    }
    let material = a.material();
    for mut h in &mut hair {
        h.material = material;
    }
    for mut reset in &mut resets {
        reset.0 = true;
    }
}

/// Procedurally grow `strand_count` strands radially out of a unit sphere of
/// `radius`, each drooping under a little gravity and tapering toward the tip.
fn fur_ball(radius: f32, strand_count: usize) -> HairAsset {
    const POINTS_PER_STRAND: usize = 6;
    const LENGTH: f32 = 0.35;
    const GRAVITY: f32 = 0.18;

    let mut strands = Vec::with_capacity(strand_count);
    for i in 0..strand_count {
        // Fibonacci sphere — even coverage without an RNG dependency.
        let t = (i as f32 + 0.5) / strand_count as f32;
        let phi = (1.0 - 2.0 * t).acos();
        let theta = core::f32::consts::PI * (1.0 + 5f32.sqrt()) * i as f32;
        let dir = Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
        let root = dir * radius;

        let mut points = Vec::with_capacity(POINTS_PER_STRAND);
        for k in 0..POINTS_PER_STRAND {
            let s = k as f32 / (POINTS_PER_STRAND - 1) as f32;
            let mut p = root + dir * (LENGTH * s);
            p.y -= GRAVITY * s * s;
            points.push(p);
        }
        strands.push(HairStrand::tapered(points, 0.006, 0.0015));
    }
    HairAsset { strands }
}

fn setup_camera_and_light(mut commands: Commands) {
    commands.spawn((
        SolariDirectionLight {
            illuminance: light_consts::lux::RAW_SUNLIGHT,
            ..default()
        },
        Transform::from_xyz(1.0, 1.6, 1.2).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(0.02, 0.02, 0.05)),
            ..default()
        },
        FreeCamera {
            walk_speed: 60.0,
            run_speed: 160.0,
            ..default()
        },
        // Framed for the upright ~114-unit groom (base on the floor); fly with
        // the FreeCamera (WASD + mouse) to reframe.
        Transform::from_translation(Vec3::new(0.0, 60.0, 250.0))
            .looking_at(Vec3::new(0.0, 55.0, 0.0), Vec3::Y),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        Exposure::OVERCAST,
        Hdr,
        SolariCamera,
        SolariAtmosphere::default(),
    ));
}
