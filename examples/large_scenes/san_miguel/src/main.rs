// Press B for benchmark.
// Preferably after frame time is reading consistently, rust-analyzer has calmed down, and with locked gpu clocks.

// Under `solari` the raster camera/light components are gated out, leaving their
// imports conditionally unused.
#![cfg_attr(feature = "solari", allow(dead_code, unused_imports))]

use std::{
    f32::consts::PI,
    ops::{Add, Mul, Sub},
    path::PathBuf,
    time::Instant,
};

use argh::FromArgs;
use bevy::image::{ImageAddressMode, ImageSamplerDescriptor};
use bevy::log::LogPlugin;
use bevy::pbr::ContactShadows;
use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    camera::visibility::{NoCpuCulling, NoFrustumCulling},
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    core_pipeline::prepass::{DeferredPrepass, DepthPrepass},
    diagnostic::DiagnosticsStore,
    light::TransmittedShadowReceiver,
    pbr::{
        DefaultOpaqueRendererMethod, ScreenSpaceAmbientOcclusion, ScreenSpaceTransmission,
        ScreenSpaceTransmissionQuality,
    },
    post_process::bloom::Bloom,
    render::{
        batching::NoAutomaticBatching, occlusion_culling::OcclusionCulling, render_resource::Face,
        view::NoIndirectDrawing,
    },
    world_serialization::WorldInstanceReady,
};
use bevy::{
    camera::Hdr,
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig},
    diagnostic::FrameTimeDiagnosticsPlugin,
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    light::CascadeShadowConfigBuilder,
    prelude::*,
    window::{PresentMode, WindowResolution},
    winit::WinitSettings,
};
use mipmap_generator::{
    generate_mipmaps, MipmapGeneratorDebugTextPlugin, MipmapGeneratorPlugin,
    MipmapGeneratorSettings,
};

#[cfg(feature = "solari")]
use bevy::{
    camera::CameraMainTextureUsages,
    light::cluster::ClusterConfig,
    render::render_resource::TextureUsages,
    solari::prelude::*,
};

use crate::light_consts::lux;

#[cfg(feature = "solari")]
mod settings;

#[derive(FromArgs, Resource, Clone)]
/// Config
pub struct Args {
    /// disable glTF lights
    #[argh(switch)]
    no_gltf_lights: bool,

    /// disable bloom, AO, AA, shadows
    #[argh(switch)]
    minimal: bool,

    /// compress textures (if they are not already, requires compress feature)
    #[argh(switch)]
    compress: bool,

    /// if low_quality_compression is set, only 0.5 byte/px formats will be used (BC1, BC4) unless the alpha channel is in use, then BC3 will be used.
    /// When low quality is set, compression is generally faster than CompressionSpeed::UltraFast and CompressionSpeed is ignored.
    #[argh(switch)]
    low_quality_compression: bool,

    /// compressed texture cache (requires compress feature)
    #[argh(switch)]
    cache: bool,

    /// quantity of San Miguel copies (tiled in a grid to stress the scene)
    #[argh(option, default = "1")]
    count: u32,

    /// spin the scene and camera
    #[argh(switch)]
    spin: bool,

    /// don't show frame time
    #[argh(switch)]
    hide_frame_time: bool,

    /// use deferred shading
    #[argh(switch)]
    deferred: bool,

    /// disable all frustum culling. Stresses queuing and batching as all mesh material entities in the scene are always drawn.
    #[argh(switch)]
    no_frustum_culling: bool,

    /// disable automatic batching. Skips batching resulting in heavy stress on render pass draw command encoding.
    #[argh(switch)]
    no_automatic_batching: bool,

    /// disable gpu occlusion culling for the camera
    #[argh(switch)]
    no_view_occlusion_culling: bool,

    /// disable gpu occlusion culling for the directional light
    #[argh(switch)]
    no_shadow_occlusion_culling: bool,

    /// disable indirect drawing.
    #[argh(switch)]
    no_indirect_drawing: bool,

    /// disable CPU culling.
    #[argh(switch)]
    no_cpu_culling: bool,

    /// disable mip map generation.
    #[argh(switch)]
    no_mip_generation: bool,

    /// number of meshes to bake into ray-tracing clusters per frame (solari).
    /// Lower keeps the window responsive on heavy scenes; higher loads faster.
    #[cfg(feature = "solari")]
    #[argh(option, default = "16")]
    bake_per_frame: u32,
}

pub fn main() {
    let args: Args = argh::from_env();

    let mut app = App::new();

    // DLSS needs its project id inserted before RenderPlugin (DlssInitPlugin reads
    // it during render init). `solari` enables `bevy/dlss`.
    #[cfg(feature = "dlss")]
    app.insert_resource(bevy::anti_alias::dlss::DlssProjectId(
        bevy::asset::uuid::uuid!("b1f7d9e3-2a4c-4d6b-8f1e-3c5a7b9d0f2e"),
    ));

    let default_plugins = DefaultPlugins
        .set(WindowPlugin {
            primary_window: Some(Window {
                title: "San Miguel".into(),
                //resolution: WindowResolution::new(1920, 1080).with_scale_factor_override(1.0),
                //present_mode: PresentMode::AutoNoVsync,
                //position: WindowPosition::Centered(MonitorSelection::Primary),
                ..default()
            }),
            ..default()
        })
        // San Miguel's floors/walls/columns tile their textures (UVs run well past [0,1]). The
        // `.bsn` import path carries no per-texture wrap mode (unlike glTF, which sets a sampler per
        // texture), so the scene's images fall back to this default — make it REPEAT. Without it a
        // clamp-to-edge sampler shows one correct texture cell at [0,1] and smears the edge texel
        // across the rest of the surface. (glTF-loaded textures on the raster path set their own
        // sampler and are unaffected.)
        .set(ImagePlugin {
            default_sampler: ImageSamplerDescriptor {
                address_mode_u: ImageAddressMode::Repeat,
                address_mode_v: ImageAddressMode::Repeat,
                address_mode_w: ImageAddressMode::Repeat,
                ..ImageSamplerDescriptor::linear()
            },
            ..default()
        })
        // Drop a hand-maintained list of benign Vulkan validation VUIDs (see
        // `tess_log_filter::BENIGN`) so real errors aren't buried in the solari
        // RT path's known-noise. The `LogPlugin.filter` (EnvFilter) can't target a
        // VUID — they all share the `wgpu_hal::vulkan::instance` target — so this
        // goes through a custom `fmt_layer` that filters by message text.
        .set(LogPlugin {
            fmt_layer: tess_log_filter::fmt_layer,
            filter: [
                "bevy_camera_controller::free_camera",
            ].join(","),
            ..default()
        });
    // Under `solari` the full-RT path replaces the raster mesh/material stack, so
    // disable `PbrPlugin` (bevy_solari owns its material/lights + vendors the DfgLut
    // + pbr shader helpers) and `TransformPlugin` (the GPU transform table drives
    // the RT scene; CPU `GlobalTransform` is restored below only where still read).
    #[cfg(feature = "solari")]
    let default_plugins = default_plugins
        .disable::<bevy::transform::TransformPlugin>()
        .disable::<bevy::pbr::PbrPlugin>()
        .disable::<bevy::dev_tools::render_debug::RenderDebugOverlayPlugin>();

    app.add_plugins(default_plugins)
        .add_plugins((
            FrameTimeDiagnosticsPlugin {
                max_history_length: 1000,
                ..default()
            },
            FreeCameraPlugin,
            FeathersPlugins,
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
        ))
        .init_resource::<CameraPositions>()
        .init_resource::<FrameLowHigh>()
        .insert_resource(GlobalAmbientLight::NONE)
        .insert_resource(args.clone())
        .insert_resource(ClearColor(Color::srgb(1.75, 1.9, 1.99)))
        .insert_resource(WinitSettings::continuous())
        .insert_resource(UiTheme(create_dark_theme()))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (input, run_animation, spin, frame_time_system, benchmark).chain(),
        )
        // Headless iteration: when launched by Claude Code (CLAUDECODE=1), auto-exit after
        // 20s so the agent can run + grep the log without manual window-closing.
        .add_systems(Update, claude_auto_exit);

    #[cfg(feature = "solari")]
    app.add_plugins((SolariPlugin, settings::LightSettingsPlugin))
        // PbrPlugin no longer registers `Assets<StandardMaterial>`, but proc_scene /
        // benchmark / mipmap systems still reference it.
        .init_asset::<StandardMaterial>()
        // The `.bsn` already carries `RaytracingMesh3d` + `SolariMaterial3d` (the importer baked
        // them), so there's no mesh/material conversion to run. No `TransformStatic` tagging
        // either: these entities are *born* carrying `RaytracingMesh3d`, so tagging them static
        // the same frame would exclude them from the transform extract before their first-sight
        // scatter — leaving every instance at the zero default (degenerate, so nothing renders).
        // The `Changed<Transform>` extract filter already makes a static scene cost ~0/frame, so
        // the tag bought nothing here anyway (see bistro — it omits it too).
        .add_systems(
            Update,
            (ensure_visibility, update_loading_screen, debug_scene_info).chain(),
        );

    // Under `solari` the GLB ships KTX2 textures with their own mip chains and
    // its materials are `SolariMaterial`, so the `StandardMaterial` runtime
    // mip generator is pure overhead — skip it entirely.
    #[cfg(not(feature = "solari"))]
    if !args.no_mip_generation {
        app.add_plugins((MipmapGeneratorPlugin, MipmapGeneratorDebugTextPlugin))
            // Generating mipmaps takes a minute
            // Mipmap generation be skipped if ktx2 is used
            .insert_resource(MipmapGeneratorSettings {
                anisotropic_filtering: 16,
                compression: args.compress.then(Default::default),
                compressed_image_data_cache_path: if args.cache {
                    Some(PathBuf::from("compressed_texture_cache"))
                } else {
                    None
                },
                low_quality: args.low_quality_compression,
                ..default()
            })
            .add_systems(Update, generate_mipmaps::<StandardMaterial>);
    }

    if args.no_frustum_culling {
        app.add_systems(Update, add_no_frustum_culling);
    }

    if args.deferred {
        app.insert_resource(DefaultOpaqueRendererMethod::deferred());
    }

    app.run();
}

/// CLAUDECODE-gated auto-exit (see registration): exits ~20s after launch when run
/// headlessly by the agent, so it can grep the log without a human closing the window.
fn claude_auto_exit(
    time: Res<Time>,
    mut exit: MessageWriter<AppExit>,
    mut start: Local<Option<f32>>,
) {
    if std::env::var_os("CLAUDECODE").is_none() {
        return;
    }
    let now = time.elapsed_secs();
    let s = *start.get_or_insert(now);
    if now - s > 20.0 {
        info!("claude_auto_exit: 45s elapsed — exiting for log capture");
        exit.write(AppExit::Success);
    }
}

#[derive(Component)]
pub struct Spin;

#[derive(Component)]
struct FrameTimeText;

/// Fullscreen loading overlay (solari): shown during the cluster bake.
#[cfg(feature = "solari")]
#[derive(Component)]
struct LoadingScreen;
/// The text line inside [`LoadingScreen`] that reports bake progress.
#[cfg(feature = "solari")]
#[derive(Component)]
struct LoadingText;

/// Mark up to `bake_per_frame` not-yet-converted meshes for ray-tracing each
/// frame, so [`convert_marked_meshes_to_raytracing`] bakes the scene in batches
/// instead of one window-freezing burst.
#[cfg(feature = "solari")]
fn mark_meshes_for_raytracing(
    mut commands: Commands,
    query: Query<Entity, (With<Mesh3d>, Without<ConvertToRaytracing>, Without<RaytracingMesh3d>)>,
    args: Res<Args>,
) {
    for entity in query.iter().take(args.bake_per_frame as usize) {
        commands.entity(entity).insert(ConvertToRaytracing);
    }
}

/// A cluster mesh is "ready" if it loaded (asset-server-managed) or was added directly (the debug
/// spheres) — directly-added assets aren't managed, so `is_loaded_with_dependencies` is false.
#[cfg(feature = "solari")]
fn cluster_ready(asset_server: &AssetServer, handle: &Handle<ClusterMesh>) -> bool {
    !asset_server.is_managed(handle) || asset_server.is_loaded_with_dependencies(handle)
}

/// Report `.bsn` load progress on the [`LoadingScreen`], removing it once every
/// spawned [`RaytracingMesh3d`] instance has its [`ClusterMesh`] asset loaded.
///
/// The `.bsn` spawns all instances atomically (each already carrying
/// `RaytracingMesh3d`), so "spawned" is instant — the slow part is streaming in
/// the per-submesh `.cluster_mesh` assets, which is what this tracks.
#[cfg(feature = "solari")]
fn update_loading_screen(
    mut commands: Commands,
    screen: Query<Entity, With<LoadingScreen>>,
    mut text: Query<&mut Text, With<LoadingText>>,
    instances: Query<&RaytracingMesh3d>,
    asset_server: Res<AssetServer>,
) {
    let Ok(screen) = screen.single() else {
        return;
    };
    let total = instances.iter().count();
    // Track the asset-server load state, NOT `Assets<ClusterMesh>`: solari consumes each
    // `ClusterMesh` (`remove_untracked`) when it uploads to the GPU, so it vanishes from `Assets`
    // immediately. `is_loaded_with_dependencies` persists past the upload — and is exactly what
    // solari gates instance binding on. Directly-added meshes (the debug spheres) aren't
    // asset-server-managed, so treat "unmanaged" as ready.
    let loaded = instances
        .iter()
        .filter(|mesh| cluster_ready(&asset_server, &mesh.0))
        .count();

    // All instances spawned and every cluster mesh loaded — drop the overlay.
    if total > 0 && loaded == total {
        commands.entity(screen).despawn();
        return;
    }
    if let Ok(mut text) = text.single_mut() {
        text.0 = if total == 0 {
            "Loading San Miguel...".into()
        } else {
            format!("Loading clusters: {loaded}/{total}")
        };
    }
}

/// TEST: the `.bsn` entities arrive with no visibility components (`RaytracingMesh3d` doesn't
/// require them). The glb→RT helper path keeps `ViewVisibility`. Give the `.bsn` RT meshes the
/// visibility chain to see whether solari's instance extract needs it present.
#[cfg(feature = "solari")]
fn ensure_visibility(
    mut commands: Commands,
    query: Query<Entity, (With<RaytracingMesh3d>, Without<Visibility>)>,
) {
    for entity in &query {
        commands.entity(entity).insert(Visibility::Visible);
    }
}

/// One-shot diagnostic: once every instance's cluster mesh is loaded, log the instance count,
/// the camera transform, and a few instance world transforms — to sanity-check geometry placement.
#[cfg(feature = "solari")]
fn debug_scene_info(
    instances: Query<(&RaytracingMesh3d, Option<&SolariMaterial3d>)>,
    asset_server: Res<AssetServer>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    let total = instances.iter().count();
    if total == 0 {
        return;
    }
    let loaded = instances
        .iter()
        .filter(|(mesh, _)| cluster_ready(&asset_server, &mesh.0))
        .count();
    if loaded < total {
        return;
    }
    *done = true;

    let with_material = instances.iter().filter(|(_, m)| m.is_some()).count();
    info!("San Miguel: {total} RaytracingMesh3d instances, {loaded} cluster meshes loaded, {with_material} have SolariMaterial3d");
}

pub fn setup(mut commands: Commands, asset_server: Res<AssetServer>, args: Res<Args>) {
    println!("Loading models, generating mipmaps");

    // Fullscreen overlay shown while the cluster bake streams in frame-by-frame;
    // `update_loading_screen` reports progress and removes it once every mesh is
    // baked.
    #[cfg(feature = "solari")]
    commands
        .spawn((
            LoadingScreen,
            Node {
                position_type: PositionType::Absolute,
                width: percent(100),
                height: percent(100),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(Color::BLACK),
            GlobalZIndex(100),
        ))
        .with_children(|parent| {
            parent.spawn((
                LoadingText,
                Text::new("Loading San Miguel..."),
                TextColor(Color::WHITE),
            ));
        });

    // San Miguel (Guillermo M. Leal Llaguno), converted from the McGuire-archive
    // OBJ to a single `san_miguel/SanMiguel.glb` by `prepare_san_miguel.py`
    // (headless Blender), then texture-compressed to UASTC KTX2
    // (`KHR_texture_basisu`) with mip chains + alpha modes restored
    // (`reclassify_alpha.py`) — BC7 on the GPU instead of uncompressed RGBA8.
    // The dense alpha-tested foliage is the point: it stresses the any-hit
    // alpha-test path. See the README for the pipeline.
    // glTF-free under solari: load the importer-baked `.bsn` (entities arrive with
    // `RaytracingMesh3d` + inline `SolariMaterial` already — no runtime mesh conversion). Bake it
    // with `san_miguel_import` and run with `BEVY_ASSET_ROOT` pointing at the directory that holds
    // `assets/san_miguel/` (the `solari_files` tree).
    // No `Spin`: San Miguel is static, and the OBJ geometry is offset from the world origin —
    // spinning the root would orbit the whole scene around (0,0,0) and out of frame.
    // `Visibility` on the root so the child instances' `InheritedVisibility` has a parent (avoids
    // the B0004 hierarchy warning while we test whether the visibility chain matters).
    // `SOLARI_TESS_FLOOR=1` loads the minimal floor-only scene (`Floor.bsn`, baked
    // with `san_miguel_import --floor-only`) — one displacement-mapped surface,
    // isolated, so the in-situ tessellation showcase is trivial to find.
    #[cfg(feature = "solari")]
    {
        let scene = if std::env::var_os("SOLARI_TESS_FLOOR").is_some() {
            "san_miguel/Cutout.bsn"
        } else {
            "san_miguel/SanMiguel.bsn"
        };
        commands.spawn((
            ScenePatchInstance(asset_server.load(scene)),
            Visibility::Visible,
        ));
    }

    // Raster path keeps the glTF (runtime `Mesh3d` conversion); `count > 1` tiles copies.
    #[cfg(not(feature = "solari"))]
    {
        let scene = asset_server.load("san_miguel/SanMiguel_ktx2.glb#Scene0");
        commands
            .spawn((WorldAssetRoot(scene.clone()), Spin))
            .observe(proc_scene);

        let mut count = 0;
        if args.count > 1 {
            let quantity = args.count - 1;
            let side = (quantity as f32).sqrt().ceil() as i32 / 2;
            'outer: for x in -side..=side {
                for z in -side..=side {
                    if count >= quantity {
                        break 'outer;
                    }
                    if x == 0 && z == 0 {
                        continue;
                    }
                    commands
                        .spawn((
                            WorldAssetRoot(scene.clone()),
                            Transform::from_xyz(x as f32 * 60.0, 0.0, z as f32 * 60.0),
                            Spin,
                        ))
                        .observe(proc_scene);
                    count += 1;
                }
            }
        }
    }

    // Sun
    #[cfg(not(feature = "solari"))]
    commands
        .spawn((
            Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, PI * -0.35, PI * -0.13, 0.0)),
            DirectionalLight {
                color: Color::srgb(1.0, 0.87, 0.78),
                illuminance: lux::FULL_DAYLIGHT,
                shadow_maps_enabled: !args.minimal,
                contact_shadows_enabled: !args.minimal,
                shadow_depth_bias: 0.1,
                shadow_normal_bias: 0.2,
                ..default()
            },
            CascadeShadowConfigBuilder {
                num_cascades: 3,
                minimum_distance: 0.05,
                maximum_distance: 100.0,
                first_cascade_far_bound: 10.0,
                overlap_proportion: 0.2,
            }
            .build(),
        ))
        .insert_if(OcclusionCulling, || !args.no_shadow_occlusion_culling);

    // Under solari the sun is a `SolariDirectionLight` (own light type, direction
    // resolved from the GPU transform table). Shadows/cascades are raster concepts —
    // the ray tracer traces shadow rays directly.
    #[cfg(feature = "solari")]
    commands.spawn((
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, PI * -0.35, PI * -0.13, 0.0)),
        SolariDirectionLight {
            color: Color::srgb(1.0, 0.87, 0.78),
            illuminance: lux::FULL_DAYLIGHT,
            ..default()
        },
        settings::Sun,
    ));

    // Camera. Headless (CLAUDECODE): aim straight at the GPU-tess geometry (gen verts
    // cluster ~x12-23, y1.8-9.4, z~0 — a wall plane) so the tess-hit counter isn't a false
    // zero from the surfaces being off-screen.
    let cam_transform = if std::env::var_os("CLAUDECODE").is_some() {
        Transform::from_xyz(17.0, 5.0, 9.0).looking_at(Vec3::new(17.0, 4.5, 0.0), Vec3::Y)
    } else {
        Transform::from_xyz(12.0, 2.0, 12.0).looking_at(Vec3::new(0.0, 2.5, 0.0), Vec3::Y)
    };
    let mut cam = commands.spawn((
        Msaa::Off,
        Camera3d::default(),
        Hdr,
        cam_transform,
        Projection::Perspective(PerspectiveProjection {
            fov: std::f32::consts::PI / 3.0,
            near: 0.1,
            far: 1000.0,
            aspect_ratio: 1.0,
            ..Default::default()
        }),
        FreeCamera::default(),
        Spin,
    ));

    // Under solari the camera is driven by the ray tracer: no raster transmission /
    // IBL / prepass / postfx. `STORAGE_BINDING` lets the RT compute pass write the
    // view's main texture; `ClusterConfig::None` skips raster light clustering.
    #[cfg(feature = "solari")]
    cam.insert((
        SolariCamera::default(),
        // Self-contained single-scattering sky, baked to a cube and sampled on
        // ray miss (background + IBL); sun = the `SolariDirectionLight`. The
        // defaults are metres-scale (~12 km visibility, 100 m fog layer), which
        // matches this scene.
        SolariAtmosphere::default(),
        ClusterConfig::None,
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
    ));

    #[cfg(not(feature = "solari"))]
    {
        cam.insert((
            ScreenSpaceTransmission {
                steps: 0,
                quality: ScreenSpaceTransmissionQuality::Low,
            },
            EnvironmentMapLight {
                diffuse_map: asset_server
                    .load("environment_maps/san_giuseppe_bridge_4k_diffuse.ktx2"),
                specular_map: asset_server
                    .load("environment_maps/san_giuseppe_bridge_4k_specular.ktx2"),
                intensity: 600.0,
                ..default()
            },
            ContactShadows::default(),
        ));
        cam.insert_if(DepthPrepass, || args.deferred)
            .insert_if(DeferredPrepass, || args.deferred)
            .insert_if(OcclusionCulling, || !args.no_view_occlusion_culling)
            .insert_if(NoFrustumCulling, || args.no_frustum_culling)
            .insert_if(NoAutomaticBatching, || args.no_automatic_batching)
            .insert_if(NoIndirectDrawing, || args.no_indirect_drawing)
            .insert_if(NoCpuCulling, || args.no_cpu_culling);
        if !args.minimal {
            cam.insert((
                Bloom {
                    intensity: 0.02,
                    ..default()
                },
                TemporalAntiAliasing::default(),
            ))
            .insert(ScreenSpaceAmbientOcclusion::default());
        }
    }

    if !args.hide_frame_time {
        commands
            .spawn((
                Node {
                    left: px(1.5),
                    top: px(1.5),
                    ..default()
                },
                GlobalZIndex(-1),
            ))
            .with_children(|parent| {
                parent.spawn((Text::new(""), TextColor(Color::BLACK), FrameTimeText));
            });
        commands.spawn(Node::default()).with_children(|parent| {
            parent.spawn((Text::new(""), TextColor(Color::WHITE), FrameTimeText));
        });
    }
}

pub fn all_children<F: FnMut(Entity)>(
    children: &Children,
    children_query: &Query<&Children>,
    closure: &mut F,
) {
    for child in children {
        if let Ok(children) = children_query.get(*child) {
            all_children(children, children_query, closure);
        }
        closure(*child);
    }
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn proc_scene(
    scene_ready: On<WorldInstanceReady>,
    mut commands: Commands,
    children: Query<&Children>,
    has_std_mat: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    lights: Query<Entity, Or<(With<PointLight>, With<DirectionalLight>, With<SpotLight>)>>,
    cameras: Query<Entity, With<Camera>>,
    args: Res<Args>,
) {
    for entity in children.iter_descendants(scene_ready.entity) {
        // San Miguel's dense alpha-cutout foliage is the workload we want: tag
        // MASK materials as thin, double-sided leaves so they transmit light and
        // the ray tracer's any-hit alpha-test path is properly exercised.
        if let Ok(mat_h) = has_std_mat.get(entity)
            && let Some(mut mat) = materials.get_mut(mat_h)
        {
            match mat.alpha_mode {
                AlphaMode::Mask(_) => {
                    mat.diffuse_transmission = 0.6;
                    mat.double_sided = true;
                    mat.cull_mode = None;
                    mat.thickness = 0.2;
                    commands.entity(entity).insert(TransmittedShadowReceiver);
                }
                AlphaMode::Opaque => {
                    mat.double_sided = false;
                    mat.cull_mode = Some(Face::Back);
                }
                _ => (),
            }
        }

        if args.no_gltf_lights {
            // Has a bunch of lights by default
            if lights.get(entity).is_ok() {
                commands.entity(entity).despawn();
            }
        }

        // Has a bunch of cameras by default
        if cameras.get(entity).is_ok() {
            commands.entity(entity).despawn();
        }
    }
}

#[derive(Resource, Deref, DerefMut)]
struct CameraPositions([Transform; 3]);

impl Default for CameraPositions {
    fn default() -> Self {
        // Starting viewpoints for San Miguel's courtyard. These are reasonable
        // first guesses — fly with the free camera and press `I` to print the
        // current transform, then paste better values here.
        Self([
            Transform::from_xyz(12.0, 2.0, 12.0).looking_at(Vec3::new(0.0, 2.5, 0.0), Vec3::Y),
            Transform::from_xyz(0.0, 9.0, 16.0).looking_at(Vec3::new(0.0, 3.0, -2.0), Vec3::Y),
            Transform::from_xyz(-10.0, 1.7, -8.0).looking_at(Vec3::new(4.0, 2.0, 6.0), Vec3::Y),
        ])
    }
}

const ANIM_SPEED: f32 = 0.2;
const ANIM_HYSTERESIS: f32 = 0.1; // EMA/LPF

// Placeholder fly-through path (Space toggles it). `const` can't call
// `looking_at`, so these use identity rotation — record a real path with `I`
// and paste the printed transforms here.
const ANIM_CAM: [Transform; 3] = [
    Transform {
        translation: Vec3::new(12.0, 2.0, 12.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    },
    Transform {
        translation: Vec3::new(0.0, 9.0, 16.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    },
    Transform {
        translation: Vec3::new(-10.0, 1.7, -8.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    },
];

fn input(
    input: Res<ButtonInput<KeyCode>>,
    mut camera: Query<&mut Transform, With<Camera>>,
    positions: Res<CameraPositions>,
) {
    let Ok(mut transform) = camera.single_mut() else {
        return;
    };
    if input.just_pressed(KeyCode::KeyI) {
        info!("{:?}", transform);
    }
    if input.just_pressed(KeyCode::Digit1) {
        *transform = positions[0]
    }
    if input.just_pressed(KeyCode::Digit2) {
        *transform = positions[1]
    }
    if input.just_pressed(KeyCode::Digit3) {
        *transform = positions[2]
    }
}

fn lerp<T>(a: T, b: T, t: f32) -> T
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<f32, Output = T>,
{
    a + (b - a) * t
}

fn follow_path(points: &[Transform], progress: f32) -> Transform {
    let total_segments = (points.len() - 1) as f32;
    let progress = progress.clamp(0.0, 1.0);
    let mut segment_progress = progress * total_segments;
    let segment_index = segment_progress.floor() as usize;
    segment_progress -= segment_index as f32;
    let a = points[segment_index];
    let b = points[(segment_index + 1).min(points.len() - 1)];
    Transform {
        translation: lerp(a.translation, b.translation, segment_progress),
        rotation: lerp(a.rotation, b.rotation, segment_progress),
        scale: lerp(a.scale, b.scale, segment_progress),
    }
}

fn run_animation(
    time: Res<Time>,
    input: Res<ButtonInput<KeyCode>>,
    mut animation_active: Local<bool>,
    mut camera: Query<&mut Transform, With<Camera>>,
) {
    let Ok(mut cam_tr) = camera.single_mut() else {
        return;
    };
    if input.just_pressed(KeyCode::Space) {
        *animation_active = !*animation_active;
    }
    if !*animation_active {
        return;
    }
    let progress = (time.elapsed_secs() * ANIM_SPEED).fract();
    let cycle = 1.0 - (progress * 2.0 - 1.0).abs();
    let path_state = follow_path(&ANIM_CAM, cycle);
    cam_tr.translation = lerp(cam_tr.translation, path_state.translation, ANIM_HYSTERESIS);
    cam_tr.rotation = lerp(cam_tr.rotation, path_state.rotation, ANIM_HYSTERESIS).normalize();
}

fn spin(
    camera: Single<Entity, With<Camera>>,
    mut things_to_spin: Query<&mut Transform, With<Spin>>,
    time: Res<Time>,
    args: Res<Args>,
    mut positions: ResMut<CameraPositions>,
) {
    if args.spin {
        let camera_position = things_to_spin.get(*camera).unwrap().translation;
        let spin = |thing_to_spin: &mut Transform| {
            thing_to_spin.rotate_around(camera_position, Quat::from_rotation_y(time.delta_secs()));
        };
        things_to_spin.iter_mut().for_each(|mut s| spin(s.as_mut())); // WHY
        positions.iter_mut().for_each(spin);
    }
}

#[allow(clippy::too_many_arguments)]
fn benchmark(
    input: Res<ButtonInput<KeyCode>>,
    mut camera_transform: Single<&mut Transform, With<Camera>>,
    materials: Res<Assets<StandardMaterial>>,
    meshes: Res<Assets<Mesh>>,
    has_std_mat: Query<&MeshMaterial3d<StandardMaterial>>,
    has_mesh: Query<&Mesh3d>,
    mut bench_started: Local<Option<Instant>>,
    mut bench_frame: Local<u32>,
    mut count_per_step: Local<u32>,
    time: Res<Time>,
    positions: Res<CameraPositions>,
    mut low_high: ResMut<FrameLowHigh>,
) {
    if input.just_pressed(KeyCode::KeyB) && bench_started.is_none() {
        low_high.bench_reset();
        *bench_started = Some(Instant::now());
        *bench_frame = 0;
        // Try to render for around 3s or at least 60 frames per step
        *count_per_step = ((3.0 / time.delta_secs()) as u32).max(60);
        println!(
            "Starting Benchmark with {} frames per step",
            *count_per_step
        );
    }
    if bench_started.is_none() {
        return;
    }
    if *bench_frame == 0 {
        **camera_transform = positions[0]
    } else if *bench_frame == *count_per_step {
        **camera_transform = positions[1]
    } else if *bench_frame == *count_per_step * 2 {
        **camera_transform = positions[2]
    } else if *bench_frame == *count_per_step * 3 {
        let elapsed = bench_started.unwrap().elapsed().as_secs_f32();
        println!(
            "{:>7.2}ms Benchmark avg cpu frame time",
            (elapsed / *bench_frame as f32) * 1000.0
        );
        let r = 1.0 / *bench_frame as f64;
        println!("{:>7.2}ms avg 1% low", low_high.sum_one_percent_low * r);
        println!("{:>7.2}ms avg 1% high", low_high.sum_one_percent_high * r);
        println!(
            "{:>7} Meshes\n{:>7} Mesh Instances\n{:>7} Materials\n{:>7} Material Instances",
            meshes.len(),
            has_mesh.iter().len(),
            materials.len(),
            has_std_mat.iter().len(),
        );
        *bench_started = None;
        *bench_frame = 0;
        **camera_transform = positions[0];
    }
    *bench_frame += 1;
    low_high.bench_step();
}

pub fn add_no_frustum_culling(
    mut commands: Commands,
    convert_query: Query<
        Entity,
        (
            Without<NoFrustumCulling>,
            With<MeshMaterial3d<StandardMaterial>>,
        ),
    >,
) {
    for entity in convert_query.iter() {
        commands.entity(entity).insert(NoFrustumCulling);
    }
}

#[derive(Resource, Default)]
struct FrameLowHigh {
    one_percent_low: f64,
    one_percent_high: f64,
    sum_one_percent_low: f64,
    sum_one_percent_high: f64,
}

impl FrameLowHigh {
    fn bench_reset(&mut self) {
        self.sum_one_percent_high = 0.0;
        self.sum_one_percent_low = 0.0;
    }
    fn bench_step(&mut self) {
        self.sum_one_percent_high += self.one_percent_high;
        self.sum_one_percent_low += self.one_percent_low;
    }
}

fn frame_time_system(
    diagnostics: Res<DiagnosticsStore>,
    mut text: Query<&mut Text, With<FrameTimeText>>,
    mut measurements: Local<Vec<f64>>,
    mut low_high: ResMut<FrameLowHigh>,
) {
    if let Some(frame_time) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FRAME_TIME) {
        let mut string = format!(
            "\n{:>7.2}ms ema\n{:>7.2}ms sma\n",
            frame_time.smoothed().unwrap_or_default(),
            frame_time.average().unwrap_or_default()
        );

        if frame_time.history_len() >= 100 {
            measurements.clear();
            measurements.extend(frame_time.measurements().map(|t| t.value));
            measurements.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let count = measurements.len() / 100;
            low_high.one_percent_low = measurements.iter().take(count).sum::<f64>() / count as f64;
            low_high.one_percent_high =
                measurements.iter().rev().take(count).sum::<f64>() / count as f64;

            string.push_str(&format!(
                "{:>7.2}ms 1% low\n{:>7.2}ms 1% high\n",
                low_high.one_percent_low, low_high.one_percent_high
            ));
        }

        for mut t in &mut text {
            t.0 = string.clone();
        }
    };
}

/// Custom `LogPlugin` fmt layer that drops a maintained list of benign Vulkan
/// validation VUIDs by message text, so real errors aren't buried in the solari
/// RT path's known-noise. `LogPlugin.filter` (an `EnvFilter`) can't target a
/// single VUID — they all log under the `wgpu_hal::vulkan::instance` target — so
/// the suppression is done here by inspecting the event message.
/// Add VUIDs to `BENIGN` only once confirmed noise.
mod tess_log_filter {
    use bevy::app::App;
    use bevy::log::{
        tracing::{
            field::{Field, Visit},
            Event, Metadata,
        },
        tracing_subscriber::{
            layer::{Context, Filter},
            Layer,
        },
        BoxedFmtLayer,
    };
    use core::fmt::Debug;

    /// VUIDs confirmed benign for the solari RT path — each is either a
    /// validation-layer false positive or a place the naga fork knowingly diverges;
    /// the driver executes all of them correctly (the scene renders right).
    ///
    /// Baseline: **Vulkan SDK / validation layers 1.4.350+**. Entries that an older
    /// layer flagged but 1.4.350 already fixed (the sparse cluster-AS dst-address
    /// resolution, and the spec-legal `srcAccelerationStructureData = 0`) are NOT
    /// carried here — we require the newer layers instead of suppressing them.
    ///
    /// Add a VUID here ONLY after confirming it's noise; never park a *fixed* bug
    /// here, or a real regression would be silently hidden.
    const BENIGN: &[&str] = &[
        // ── naga-fork SPIR-V emission (driver accepts; spirv-val is stricter) ──
        // Bindless `textures: binding_array<texture_2d>` → a runtime array; legal
        // for the UniformConstant storage class, and the driver accepts it.
        "VUID-StandaloneSpirv-OpTypeRuntimeArray-04680",
        // The naga fork emits Device-scope atomics (see the atomic-scope note); the
        // driver runs them fine without `vulkanMemoryModelDeviceScope`.
        "VUID-RuntimeSpirv-vulkanMemoryModel-06265",
        // naga emits `ArrayStride` on `var<workgroup>` array types (illegal explicit
        // layout for the Workgroup class). Upstream naga bug, hit by bevy's own SPD
        // mipmap-downsample shader (`array<array<f32,16>,16>` tiles); driver runs it.
        "VUID-StandaloneSpirv-None-10684",
        // Closest-hit reads set0/binding0 (`cluster_indices`), GPU-AV claims it's
        // uninitialized. It's bound (wgpu's own scene set) and read on every hit —
        // GPU-AV's descriptor-indexing tracking trips on a raw pipeline binding a
        // wgpu `UPDATE_AFTER_BIND` set (the set holds bindless texture arrays).
        "VUID-vkCmdTraceRaysKHR-None-08114",
        // First presented swapchain image is in UNDEFINED layout for one frame
        // before the first render writes it. Cosmetic first-frame warning.
        "VUID-VkPresentInfoKHR-pImageIndices-01430",
        
    ];

    /// Per-layer filter: drop an event if any of its fields' text contains a
    /// `BENIGN` VUID. The VUID rides in the event's `message` field, recorded via
    /// `record_debug`; `record_str` is covered for robustness.
    struct DropBenign;

    impl<S> Filter<S> for DropBenign {
        fn enabled(&self, _meta: &Metadata<'_>, _cx: &Context<'_, S>) -> bool {
            true
        }

        fn event_enabled(&self, event: &Event<'_>, _cx: &Context<'_, S>) -> bool {
            struct Scan(bool);
            impl Visit for Scan {
                fn record_debug(&mut self, _f: &Field, value: &dyn Debug) {
                    if BENIGN.iter().any(|v| format!("{value:?}").contains(v)) {
                        self.0 = true;
                    }
                }
                fn record_str(&mut self, _f: &Field, value: &str) {
                    if BENIGN.iter().any(|v| value.contains(v)) {
                        self.0 = true;
                    }
                }
            }
            let mut scan = Scan(false);
            event.record(&mut scan);
            !scan.0
        }
    }

    pub fn fmt_layer(_app: &mut App) -> Option<BoxedFmtLayer> {
        Some(Box::new(
            bevy::log::tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_filter(DropBenign),
        ))
    }
}
