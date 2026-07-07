//! Furnace validation for the reference path tracer. Two scenes:
//! `furnace` (default): spheres in a uniform white environment — a white sphere of
//! ANY roughness must converge to exactly the sky's radiance (ratio 1.0) or the
//! BRDF creates/destroys energy. `room`: a closed grey box lit by an emissive
//! panel — run with and without `--no-nee`; both must converge to the same means.
//! Probe pixels are read back from the RT output buffer and logged as numbers.

use bevy::prelude::*;

use bevy::{
    window::PresentMode,
    asset::{uuid::uuid, RenderAssetUsages},
    camera::CameraMainTextureUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    core_pipeline::Skybox,
    dev_tools::render_debug::RenderDebugOverlayPlugin,
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    image::{Image, ImageAddressMode, ImageSamplerDescriptor},
    log::LogPlugin,
    math::{DQuat, DVec3},
    pbr::PbrPlugin,
    render::{
        camera::ExtractedCamera,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode,
            PollType, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
            TextureViewDimension,
        },
        renderer::{RenderDevice, RenderQueue},
        view::{
            screenshot::{save_to_disk, Screenshot},
            ExtractedView,
        },
        Render, RenderApp, RenderSystems,
    },
    solari::{
        prelude::*,
        render::rt_pipeline::{RtAccumulation, RtOutputBuffer, SolariClusterView},
    },
    transform::TransformPlugin,
};
use argh::FromArgs;

// The full-RT path supplies its own DLSS Ray Reconstruction internally; the SDK
// only needs the project id inserted before `RenderPlugin`.
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::DlssProjectId;

/// Furnace exam harness for the solari reference path tracer (see
/// zero/docs/restir_roadmap.md for the rung-by-rung curriculum this drives).
#[derive(FromArgs)]
struct Args {
    /// exit after this many seconds (agent runs default to 10)
    #[argh(option, short = 't')]
    timeout: Option<f32>,

    /// scene: `furnace` (uniform sky + spheres), `room` (emissive panel in a grey box),
    /// `lamps` (many lamps, power spanning 4 decades — the rung-1 exam), `yard`
    /// (sun + lamps), or `cell` (static boxes under lamps — the spatial-bias study)
    #[argh(option, default = "String::from(\"furnace\")")]
    scene: String,

    /// paths per pixel per frame
    #[argh(option, default = "16")]
    spp: u32,

    /// bsdf-only estimator (no next-event estimation) — the unbiasedness A/B
    #[argh(switch)]
    no_nee: bool,

    /// rung-0 self-test: freeze at 4s, dump PFM at 5s, diff view on at 6s
    #[argh(switch)]
    rung0: bool,

    /// force uniform light picking (rung-1 A/B: must converge to the same image
    /// as power-weighted, just noisier)
    #[argh(switch)]
    uniform_lights: bool,

    /// write one PFM the moment the accumulation crosses N spp (equal-sample RMSE)
    #[argh(option, default = "0")]
    dump_at_spp: u32,

    /// ris candidates per NEE sample (rung 2). 1 = plain NEE
    #[argh(option, default = "1")]
    ris: u32,

    /// reSTIR DI (rung 3): temporal reservoir reuse at the primary vertex.
    /// Key R toggles it live (freeze first, then A/B against the diff view)
    #[argh(switch)]
    restir: bool,

    /// debug view: `cluster` = flat per-cluster color from raygen (proves primary
    /// rays hit + payload flows, independent of all shading/DI code); `spatial` =
    /// the spatial pass's kill-stage paint; `gi-dead` = dead-canonical indicator
    /// (accumulated mean = dead-draw rate; needs --restir-gi)
    #[argh(option, default = "String::new()")]
    debug_view: String,

    /// direct illumination only (terminate at the primary vertex) — the standard
    /// ReSTIR evaluation image; needs its own truth capture
    #[argh(switch)]
    di_only: bool,

    /// indirect only (suffix energy past the primary vertex) — the rung-4a
    /// complement; di_only + gi_only must sum to the full image
    #[argh(switch)]
    gi_only: bool,

    /// reSTIR GI (rung 4a.1): shade GI from the stored canonical sample —
    /// the buffer round-trip gate (must equal plain PT per-path)
    #[argh(switch)]
    restir_gi: bool,

    /// with --restir-gi: reshade f·cos·L/pdf from the surface G-buffer instead of
    /// the stored exact a0 — the reconnection-shift shading path (gate: unbiased)
    #[argh(switch)]
    gi_recon: bool,

    /// with --restir-gi: temporal GI reservoir reuse (reprojected + validated,
    /// m-capped)
    #[argh(switch)]
    gi_temporal: bool,

    /// with --restir-gi: spatial GI reuse in the compute pass (jacobian-weighted
    /// neighbors, winner visibility; --taps/--radius apply)
    #[argh(switch)]
    gi_spatial: bool,

    /// disable accumulation: fresh frames each frame (restir history stays warm).
    /// With `--rung0`, the 5s dump captures ONE frame — per-frame variance metric
    #[argh(switch)]
    no_accum: bool,

    /// spatial reuse pass (rung-3 session 2; needs --restir). Naive biased
    /// combiner by default — the darkening study; add --zcount for unbiased
    #[argh(switch)]
    spatial: bool,

    /// visibility-aware 1/Z spatial combiner (drops shadowed contributors from the
    /// denominator). MEASURED to OVER-brighten ~2.6% here — the base pass is already
    /// unbiased (per-neighbor p-hat re-eval + depth/normal validation), so naive is the
    /// production default; this is a research lever + costs one shadow ray/contributor
    #[argh(switch)]
    zcount: bool,

    /// spatial neighbor taps (<=8). Default 3 = the equal-time sweet spot (5 taps
    /// over-invests: lower per-frame noise but a net equal-time loss vs
    /// temporal-only; 2-3 taps is the small net win)
    #[argh(option, default = "3")]
    taps: u32,

    /// spatial neighbor disk radius in pixels
    #[argh(option, default = "20.0")]
    radius: f32,

    /// soak: sway/yaw the camera for N seconds, then snap back to the start
    /// pose and hold — churns reservoir history with motion + disocclusion
    #[argh(option, default = "0.0")]
    orbit: f32,

    /// lamps-scene power law: `pilot` (4 floods + 60 pilots) or `equal`
    /// (64 identical lamps — the receiver-locality exam where global picking
    /// is uninformative and RIS must find the lamps overhead)
    #[argh(option, default = "String::from(\"pilot\")")]
    lamp_law: String,

    /// production path: SolariRestir (full per-frame ReSTIR stack) + DLSS RR
    /// instead of the reference accumulator — the estimator flags above are
    /// ignored; --taps/--radius apply. Probe expects will FAIL (per-frame
    /// noise); this is the visual/perf smoke test, not an exam
    #[argh(switch)]
    production: bool,
}

/// Uniform sky radiance (cd/m²) for the furnace scene.
const SKY_NITS: f32 = 1000.0;

fn main() {
    let args: Args = argh::from_env();
    let scene_room = args.scene == "room";
    let scene_lamps = args.scene == "lamps";
    let scene_yard = args.scene == "yard";
    let scene_cell = args.scene == "cell";
    let mut app = App::new();
    // Exams time frames from logs: never let the unfocused 60 Hz throttle
    // (WinitSettings::game default) poison the clock.
    app.insert_resource(bevy::winit::WinitSettings {
        focused_mode: bevy::winit::UpdateMode::Continuous,
        unfocused_mode: bevy::winit::UpdateMode::Continuous,
    });
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(uuid!("d5580e3b-691b-4fef-8dcd-58c0ae6df08e")));
    app.insert_resource(OrbitSoak { secs: args.orbit, start: None });
    app.add_systems(Update, orbit_soak);
    if args.production {
        app.add_systems(Update, apply_production);
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        app.insert_resource(SolariDlssMode::Dlaa);
    }
    app.add_timeout_exit(args.timeout, 10.0)
        .add_screenshot(KeyCode::F12)
        .insert_resource(SceneArgs {
            room: scene_room,
            lamps: scene_lamps,
            yard: scene_yard,
            cell: scene_cell,
            spp: args.spp,
            no_nee: args.no_nee,
            rung0: args.rung0,
            ris: args.ris.max(1),
            restir: args.restir,
            di_only: args.di_only,
            gi_only: args.gi_only,
            restir_gi: args.restir_gi,
            gi_recon: args.gi_recon,
            gi_temporal: args.gi_temporal,
            gi_spatial: args.gi_spatial,
            no_accum: args.no_accum,
            spatial: args.spatial,
            zcount: args.zcount,
            taps: args.taps,
            radius: args.radius,
            spatial_debug: args.debug_view == "spatial",
            gi_dead: args.debug_view == "gi-dead",
            equal_lamps: args.lamp_law == "equal",
        })
        .insert_resource(SolariUniformLights { enabled: args.uniform_lights })
        .insert_resource(SolariClusterView { enabled: args.debug_view == "cluster" })
        .insert_resource(SolariFreezeDiff { dump_at_spp: args.dump_at_spp, ..default() })
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Furnace".into(),
                        present_mode: PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin {
                    default_sampler: ImageSamplerDescriptor {
                        address_mode_u: ImageAddressMode::Repeat,
                        address_mode_v: ImageAddressMode::Repeat,
                        address_mode_w: ImageAddressMode::Repeat,
                        ..ImageSamplerDescriptor::linear()
                    },
                    ..default()
                })
                .set(LogPlugin {
                    filter: LOG_FILTER.into(),
                    ..default()
                })
                .disable::<TransformPlugin>()
                .disable::<PbrPlugin>()
                .disable::<RenderDebugOverlayPlugin>(),
            FreeCameraPlugin,
            SolariPlugin,
            FeathersPlugins,
        ))
        .insert_resource(UiTheme(create_dark_theme()))
        .init_resource::<Probes>()
        .add_plugins(ExtractResourcePlugin::<Probes>::default())
        .add_systems(Startup, setup_furnace.run_if(|a: Res<SceneArgs>| !a.room && !a.lamps && !a.yard && !a.cell))
        .add_systems(Startup, setup_room.run_if(|a: Res<SceneArgs>| a.room))
        .add_systems(Startup, setup_lamps.run_if(|a: Res<SceneArgs>| a.lamps))
        .add_systems(Startup, setup_yard.run_if(|a: Res<SceneArgs>| a.yard))
        .add_systems(Startup, setup_cell.run_if(|a: Res<SceneArgs>| a.cell))
        .add_systems(Update, (rung0_keys, rung0_selftest));
    app.sub_app_mut(RenderApp)
        .add_systems(Render, probe_readback.in_set(RenderSystems::Cleanup));
    app.run();
}

#[derive(Resource)]
struct SceneArgs {
    room: bool,
    lamps: bool,
    yard: bool,
    cell: bool,
    spp: u32,
    no_nee: bool,
    rung0: bool,
    ris: u32,
    restir: bool,
    di_only: bool,
    gi_only: bool,
    restir_gi: bool,
    gi_recon: bool,
    gi_temporal: bool,
    gi_spatial: bool,
    no_accum: bool,
    spatial: bool,
    zcount: bool,
    taps: u32,
    radius: f32,
    spatial_debug: bool,
    gi_dead: bool,
    equal_lamps: bool,
}

/// A named world-space point whose surrounding pixels get averaged each probe pass.
#[derive(Clone)]
struct Probe {
    name: &'static str,
    world: DVec3,
    /// Expected luminance ratio vs the `sky` probe (furnace scene); None = log only.
    expect: Option<f64>,
}

#[derive(Resource, Clone, Default, ExtractResource)]
struct Probes(Vec<Probe>);

fn spawn_ball(
    commands: &mut Commands,
    meshes: &mut Assets<ClusterMesh>,
    materials: &mut Assets<StandardSolariMaterial>,
    sphere: &Handle<ClusterMesh>,
    name: &'static str,
    pos: DVec3,
    mat: StandardSolariMaterial,
) {
    let _ = meshes; // one shared sphere mesh; kept for signature symmetry
    commands.spawn((
        Name::new(name),
        TransformStatic,
        NoGpuGlobalTransformReadback,
        RaytracingMesh3d(sphere.clone()),
        SolariMaterial3d(materials.add(mat)),
        Transform::from_translation(pos),
    ));
}

fn setup_furnace(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<ClusterMesh>>,
    mut materials: ResMut<Assets<StandardSolariMaterial>>,
    args: Res<SceneArgs>,
) {
    // Uniform white environment: a 1×1 all-white cubemap × SKY_NITS. The whole
    // point of the furnace: incoming radiance identical from every direction.
    let mut cube = Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        // f16 1.0 = 0x3C00; Rgba16Float — Rgba32Float isn't linear-filterable and
        // the raw-VK env sampler reads it as zero with no validation to say so.
        &[0x00u8, 0x3C, 0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    cube.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });

    let sphere_mesh = Sphere::new(1.4)
        .mesh()
        .ico(6)
        .unwrap()
        .with_generated_tangents()
        .unwrap();
    let sphere = meshes.add(ClusterMesh::try_from(&sphere_mesh).expect("sphere bake"));

    let white = Color::WHITE;
    let x = |i: f64| DVec3::new((i - 3.5) * 3.0, 0.0, 0.0);
    let balls: [(&'static str, StandardSolariMaterial, Option<f64>); 8] = [
        (
            "diffuse_white",
            StandardSolariMaterial {
                base_color: white,
                metallic: 0.0,
                perceptual_roughness: 1.0,
                ..default()
            },
            Some(1.0),
        ),
        (
            "metal_r1.0",
            StandardSolariMaterial {
                base_color: white,
                metallic: 1.0,
                perceptual_roughness: 1.0,
                ..default()
            },
            Some(1.0),
        ),
        (
            "metal_r0.9",
            StandardSolariMaterial {
                base_color: white,
                metallic: 1.0,
                perceptual_roughness: 0.9,
                ..default()
            },
            Some(1.0),
        ),
        (
            "metal_r0.8",
            StandardSolariMaterial {
                base_color: white,
                metallic: 1.0,
                perceptual_roughness: 0.8,
                ..default()
            },
            Some(1.0),
        ),
        (
            "metal_r0.6",
            StandardSolariMaterial {
                base_color: white,
                metallic: 1.0,
                perceptual_roughness: 0.6,
                ..default()
            },
            Some(1.0),
        ),
        (
            "metal_r0.3",
            StandardSolariMaterial {
                base_color: white,
                metallic: 1.0,
                perceptual_roughness: 0.3,
                ..default()
            },
            Some(1.0),
        ),
        (
            "metal_r0.15",
            StandardSolariMaterial {
                base_color: white,
                metallic: 1.0,
                perceptual_roughness: 0.15,
                ..default()
            },
            Some(1.0),
        ),
        (
            "diffuse_grey",
            StandardSolariMaterial {
                base_color: Color::srgb(0.5, 0.5, 0.5),
                metallic: 0.0,
                perceptual_roughness: 1.0,
                ..default()
            },
            None,
        ),
    ];
    let mut probes = vec![Probe {
        name: "sky",
        world: DVec3::new(0.0, 14.0, -14.0),
        expect: None,
    }];
    for (i, (name, mat, expect)) in balls.into_iter().enumerate() {
        let pos = x(i as f64);
        spawn_ball(
            &mut commands,
            &mut meshes,
            &mut materials,
            &sphere,
            name,
            pos,
            mat,
        );
        // Probe the sphere's screen center: facing the camera head-on, away from
        // the silhouette (where the normal-bend adaptation is deliberately biased).
        probes.push(Probe {
            name,
            world: pos + DVec3::new(0.0, 0.0, 1.4),
            expect,
        });
    }
    let cam = DVec3::new(0.0, 0.0, 42.0);

    // Rung 0.5: a lossless dielectric must also vanish (Fresnel split + TIR).
    let glass_pos = DVec3::new(-13.5, 0.0, 0.0);
    spawn_ball(
        &mut commands,
        &mut meshes,
        &mut materials,
        &sphere,
        "glass_clear",
        glass_pos,
        StandardSolariMaterial {
            specular_transmission: 1.0,
            ..default()
        },
    );
    probes.push(Probe {
        name: "glass_clear",
        world: glass_pos + DVec3::new(0.0, 0.0, 1.4),
        expect: Some(1.0),
    });

    // Sphere/parallel-slab chords stay under the critical angle, so a 45°-rotated
    // cube is what actually exercises TIR (side-face exits) — must still vanish.
    let cube_pos = DVec3::new(0.0, 7.0, 0.0);
    let cube_mesh = Cuboid::from_size(Vec3::splat(3.0))
        .mesh()
        .build()
        .with_generated_tangents()
        .unwrap();
    commands.spawn((
        Name::new("glass_tir"),
        TransformStatic,
        NoGpuGlobalTransformReadback,
        RaytracingMesh3d(meshes.add(ClusterMesh::try_from(&cube_mesh).expect("cube bake"))),
        SolariMaterial3d(materials.add(StandardSolariMaterial {
            specular_transmission: 1.0,
            ..default()
        })),
        Transform::from_translation(cube_pos)
            .with_rotation(DQuat::from_rotation_y(std::f64::consts::FRAC_PI_4)),
    ));
    probes.push(Probe {
        name: "glass_tir",
        world: cube_pos + DVec3::new(0.0, 0.0, 2.0),
        expect: Some(1.0),
    });

    // Tinted slab facing the camera: attenuation_distance = thickness makes the
    // straight-through Beer–Lambert transmittance exactly attenuation_color.
    let slab_pos = DVec3::new(0.0, -7.0, 0.0);
    let tint = [0.9, 0.6, 0.3];
    let thickness = 3.0;
    let slab_mesh = Cuboid::new(6.0, 4.0, thickness as f32)
        .mesh()
        .build()
        .with_generated_tangents()
        .unwrap();
    commands.spawn((
        Name::new("glass_tinted"),
        TransformStatic,
        NoGpuGlobalTransformReadback,
        RaytracingMesh3d(meshes.add(ClusterMesh::try_from(&slab_mesh).expect("slab bake"))),
        SolariMaterial3d(materials.add(StandardSolariMaterial {
            specular_transmission: 1.0,
            attenuation_distance: thickness as f32,
            attenuation_color: Color::linear_rgb(tint[0] as f32, tint[1] as f32, tint[2] as f32),
            ..default()
        })),
        Transform::from_translation(slab_pos).looking_at(cam, Dir3::Y),
    ));
    probes.push(Probe {
        name: "glass_tinted",
        world: slab_pos + (cam - slab_pos).normalize() * (thickness / 2.0),
        expect: Some(tinted_slab_expect(tint, 1.5)),
    });

    commands.insert_resource(Probes(probes));

    commands.spawn((
        Name::new("camera"),
        Camera3d::default(),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariCamera,
        Skybox {
            image: Some(images.add(cube)),
            brightness: SKY_NITS,
            ..default()
        },
        SolariReference {
            samples_per_frame: args.spp,
            nee_off: args.no_nee,
            ris_candidates: args.ris,
            restir: args.restir,
            di_only: args.di_only,
            gi_only: args.gi_only,
            restir_gi: args.restir_gi,
            gi_recon: args.gi_recon,
            gi_temporal: args.gi_temporal,
            gi_spatial: args.gi_spatial,
            accumulate: !args.no_accum,
            spatial: args.spatial,
            spatial_unbiased: args.zcount,
            spatial_taps: args.taps,
            spatial_radius: args.radius,
            spatial_debug: args.spatial_debug,
            gi_dead_view: args.gi_dead,
            ..default()
        },
        NoGpuGlobalTransformReadback,
        FreeCamera::default(),
        Transform::from_translation(cam),
    ));
}

fn setup_room(
    mut commands: Commands,
    mut meshes: ResMut<Assets<ClusterMesh>>,
    mut materials: ResMut<Assets<StandardSolariMaterial>>,
    args: Res<SceneArgs>,
) {
    let grey = materials.add(StandardSolariMaterial {
        base_color: Color::srgb(0.6, 0.6, 0.6),
        perceptual_roughness: 1.0,
        ..default()
    });
    // Emissive-mesh light ONLY: a directional light is invisible to BSDF rays, so
    // the NEE on/off A/B is only an identity when every light is hittable geometry.
    let lamp = materials.add(StandardSolariMaterial {
        base_color: Color::WHITE,
        emissive: LinearRgba::rgb(4000.0, 4000.0, 4000.0),
        ..default()
    });

    let mut slab =
        |name: &'static str, size: Vec3, pos: DVec3, mat: &Handle<StandardSolariMaterial>| {
            let mesh = Cuboid::from_size(size)
                .mesh()
                .build()
                .with_generated_tangents()
                .unwrap();
            let handle = meshes.add(ClusterMesh::try_from(&mesh).expect("slab bake"));
            commands.spawn((
                Name::new(name),
                TransformStatic,
                NoGpuGlobalTransformReadback,
                RaytracingMesh3d(handle),
                SolariMaterial3d(mat.clone()),
                Transform::from_translation(pos),
            ));
        };
    // Closed 12×6×12 box (inner faces), 0.2 thick.
    slab(
        "floor",
        Vec3::new(12.4, 0.2, 12.4),
        DVec3::new(0.0, -3.1, 0.0),
        &grey,
    );
    slab(
        "ceiling",
        Vec3::new(12.4, 0.2, 12.4),
        DVec3::new(0.0, 3.1, 0.0),
        &grey,
    );
    slab(
        "wall_x-",
        Vec3::new(0.2, 6.4, 12.4),
        DVec3::new(-6.1, 0.0, 0.0),
        &grey,
    );
    slab(
        "wall_x+",
        Vec3::new(0.2, 6.4, 12.4),
        DVec3::new(6.1, 0.0, 0.0),
        &grey,
    );
    slab(
        "wall_z-",
        Vec3::new(12.4, 6.4, 0.2),
        DVec3::new(0.0, 0.0, -6.1),
        &grey,
    );
    slab(
        "wall_z+",
        Vec3::new(12.4, 6.4, 0.2),
        DVec3::new(0.0, 0.0, 6.1),
        &grey,
    );
    slab(
        "lamp",
        Vec3::new(2.4, 0.1, 2.4),
        DVec3::new(0.0, 2.9, 0.0),
        &lamp,
    );

    let sphere_mesh = Sphere::new(1.2)
        .mesh()
        .ico(6)
        .unwrap()
        .with_generated_tangents()
        .unwrap();
    let sphere = meshes.add(ClusterMesh::try_from(&sphere_mesh).expect("sphere bake"));
    spawn_ball(
        &mut commands,
        &mut meshes,
        &mut materials,
        &sphere,
        "ball_diffuse",
        DVec3::new(-2.0, -1.8, -1.5),
        StandardSolariMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 1.0,
            ..default()
        },
    );
    spawn_ball(
        &mut commands,
        &mut meshes,
        &mut materials,
        &sphere,
        "ball_metal",
        DVec3::new(2.0, -1.8, -1.5),
        StandardSolariMaterial {
            base_color: Color::WHITE,
            metallic: 1.0,
            perceptual_roughness: 0.4,
            ..default()
        },
    );

    // Absolute means only — the A/B across runs is the test, not a ratio.
    commands.insert_resource(Probes(vec![
        Probe {
            name: "back_wall",
            world: DVec3::new(0.0, 0.0, -6.0),
            expect: None,
        },
        Probe {
            name: "floor",
            world: DVec3::new(0.0, -3.0, -3.5),
            expect: None,
        },
        Probe {
            name: "ball_diffuse",
            world: DVec3::new(-2.0, -1.8, -0.3),
            expect: None,
        },
        Probe {
            name: "ball_metal",
            world: DVec3::new(2.0, -1.8, -0.3),
            expect: None,
        },
    ]));

    commands.spawn((
        Name::new("camera"),
        Camera3d::default(),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariCamera,
        SolariReference {
            samples_per_frame: args.spp,
            nee_off: args.no_nee,
            ris_candidates: args.ris,
            restir: args.restir,
            di_only: args.di_only,
            gi_only: args.gi_only,
            restir_gi: args.restir_gi,
            gi_recon: args.gi_recon,
            gi_temporal: args.gi_temporal,
            gi_spatial: args.gi_spatial,
            accumulate: !args.no_accum,
            spatial: args.spatial,
            spatial_unbiased: args.zcount,
            spatial_taps: args.taps,
            spatial_radius: args.radius,
            spatial_debug: args.spatial_debug,
            gi_dead_view: args.gi_dead,
            ..default()
        },
        NoGpuGlobalTransformReadback,
        FreeCamera::default(),
        Transform::from_translation(DVec3::new(0.0, 0.0, 5.2)),
    ));
}

/// Rung-1 exam: a 24×6×24 grey hall lit by an 8×8 ceiling grid of lamps whose
/// power spans ~4 decades (k² law). Uniform picking wastes almost every sample
/// on dim lamps; power weighting must converge to the SAME image, far faster.
fn setup_lamps(
    mut commands: Commands,
    mut meshes: ResMut<Assets<ClusterMesh>>,
    mut materials: ResMut<Assets<StandardSolariMaterial>>,
    args: Res<SceneArgs>,
) {
    let grey = materials.add(StandardSolariMaterial {
        base_color: Color::srgb(0.6, 0.6, 0.6),
        perceptual_roughness: 1.0,
        ..default()
    });
    let mut slab = |name: String, size: Vec3, pos: DVec3, mat: &Handle<StandardSolariMaterial>| {
        let mesh = Cuboid::from_size(size).mesh().build().with_generated_tangents().unwrap();
        let handle = meshes.add(ClusterMesh::try_from(&mesh).expect("slab bake"));
        commands.spawn((
            Name::new(name),
            TransformStatic,
            NoGpuGlobalTransformReadback,
            RaytracingMesh3d(handle),
            SolariMaterial3d(mat.clone()),
            Transform::from_translation(pos),
        ));
    };
    slab("floor".into(), Vec3::new(24.4, 0.2, 24.4), DVec3::new(0.0, -3.1, 0.0), &grey);
    slab("ceiling".into(), Vec3::new(24.4, 0.2, 24.4), DVec3::new(0.0, 3.1, 0.0), &grey);
    slab("wall_x-".into(), Vec3::new(0.2, 6.4, 24.4), DVec3::new(-12.1, 0.0, 0.0), &grey);
    slab("wall_x+".into(), Vec3::new(0.2, 6.4, 24.4), DVec3::new(12.1, 0.0, 0.0), &grey);
    slab("wall_z-".into(), Vec3::new(24.4, 6.4, 0.2), DVec3::new(0.0, 0.0, -12.1), &grey);
    slab("wall_z+".into(), Vec3::new(24.4, 6.4, 0.2), DVec3::new(0.0, 0.0, 12.1), &grey);
    for k in 0..64u32 {
        let (i, j) = (k % 8, k / 8);
        // pilot: a few floodlights over a sea of pilot lights (rung-1 exam:
        // global flux weighting wins). equal: 64 identical lamps (rung-2 exam:
        // global picking is uninformative; only receiver-aware RIS helps).
        let power = if args.equal_lamps {
            800.0
        } else if k % 16 == 15 {
            12000.0
        } else {
            2.0
        };
        let lamp = materials.add(StandardSolariMaterial {
            base_color: Color::WHITE,
            emissive: LinearRgba::rgb(power, power, power),
            ..default()
        });
        slab(
            format!("lamp_{k}"),
            Vec3::new(0.9, 0.1, 0.9),
            DVec3::new((i as f64 - 3.5) * 2.9, 2.9, (j as f64 - 3.5) * 2.9),
            &lamp,
        );
    }
    commands.insert_resource(Probes(vec![
        Probe { name: "floor_dim", world: DVec3::new(-9.0, -3.0, -9.0), expect: None },
        Probe { name: "floor_mid", world: DVec3::new(0.0, -3.0, 0.0), expect: None },
        Probe { name: "floor_bright", world: DVec3::new(8.0, -3.0, 8.5), expect: None },
        Probe { name: "back_wall", world: DVec3::new(0.0, 0.5, -12.0), expect: None },
    ]));
    commands.spawn((
        Name::new("camera"),
        Camera3d::default(),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariCamera,
        SolariReference {
            samples_per_frame: args.spp,
            nee_off: args.no_nee,
            ris_candidates: args.ris,
            restir: args.restir,
            di_only: args.di_only,
            gi_only: args.gi_only,
            restir_gi: args.restir_gi,
            gi_recon: args.gi_recon,
            gi_temporal: args.gi_temporal,
            gi_spatial: args.gi_spatial,
            accumulate: !args.no_accum,
            spatial: args.spatial,
            spatial_unbiased: args.zcount,
            spatial_taps: args.taps,
            spatial_radius: args.radius,
            spatial_debug: args.spatial_debug,
            gi_dead_view: args.gi_dead,
            ..default()
        },
        NoGpuGlobalTransformReadback,
        FreeCamera::default(),
        Transform::from_translation(DVec3::new(0.0, 1.2, 11.0))
            .looking_at(DVec3::new(0.0, -2.0, 0.0), Vec3::Y),
    ));
}

/// Rung-3 spatial-bias fixture: a STATIC grey room under an equal-lamp ceiling,
/// with a cluster of upright boxes + tilted ramps on the floor. The boxes cast
/// shadows and break the floor into varied normals/depths, so a spatial neighbor's
/// domain genuinely differs from its target — the M-sum bias (naive darkens at
/// occlusion boundaries, Z-count recovers) finally has something to bite. Nothing
/// moves, so temporal reprojection stays clean (unlike the yard).
fn setup_cell(
    mut commands: Commands,
    mut meshes: ResMut<Assets<ClusterMesh>>,
    mut materials: ResMut<Assets<StandardSolariMaterial>>,
    args: Res<SceneArgs>,
) {
    let grey = materials.add(StandardSolariMaterial {
        base_color: Color::srgb(0.6, 0.6, 0.6),
        perceptual_roughness: 1.0,
        ..default()
    });
    let mut spawn = |name: String, size: Vec3, xf: Transform, mat: &Handle<StandardSolariMaterial>| {
        let mesh = Cuboid::from_size(size).mesh().build().with_generated_tangents().unwrap();
        let handle = meshes.add(ClusterMesh::try_from(&mesh).expect("cell bake"));
        commands.spawn((
            Name::new(name),
            TransformStatic,
            NoGpuGlobalTransformReadback,
            RaytracingMesh3d(handle),
            SolariMaterial3d(mat.clone()),
            xf,
        ));
    };
    let at = |p: DVec3| Transform::from_translation(p);
    // 16×6×16 grey box; floor TOP sits at y = -3.0 (slab center -3.1, half 0.1).
    let f = -3.0f64;
    spawn("floor".into(), Vec3::new(16.4, 0.2, 16.4), at(DVec3::new(0.0, -3.1, 0.0)), &grey);
    spawn("ceiling".into(), Vec3::new(16.4, 0.2, 16.4), at(DVec3::new(0.0, 3.1, 0.0)), &grey);
    spawn("wall_x-".into(), Vec3::new(0.2, 6.4, 16.4), at(DVec3::new(-8.1, 0.0, 0.0)), &grey);
    spawn("wall_x+".into(), Vec3::new(0.2, 6.4, 16.4), at(DVec3::new(8.1, 0.0, 0.0)), &grey);
    spawn("wall_z-".into(), Vec3::new(16.4, 6.4, 0.2), at(DVec3::new(0.0, 0.0, -8.1)), &grey);
    spawn("wall_z+".into(), Vec3::new(16.4, 6.4, 0.2), at(DVec3::new(0.0, 0.0, 8.1)), &grey);
    // Upright boxes (footprint, height) sitting on the floor — vertical faces +
    // flat tops (normal/depth breaks) that cast shadows (occlusion boundaries).
    let boxes = [
        (Vec3::new(1.2, 2.4, 1.2), DVec3::new(-3.0, 0.0, -2.0)),
        (Vec3::new(1.5, 1.0, 1.5), DVec3::new(2.6, 0.0, 1.2)),
        (Vec3::new(1.0, 3.0, 1.0), DVec3::new(0.2, 0.0, 3.0)),
        (Vec3::new(2.2, 0.6, 1.1), DVec3::new(-2.2, 0.0, 2.4)),
        (Vec3::new(0.8, 1.8, 0.8), DVec3::new(3.2, 0.0, -3.0)),
        (Vec3::new(1.3, 1.4, 1.3), DVec3::new(-4.2, 0.0, 3.6)),
    ];
    for (i, (size, pos)) in boxes.iter().enumerate() {
        spawn(format!("box_{i}"), *size, at(DVec3::new(pos.x, f + size.y as f64 * 0.5, pos.z)), &grey);
    }
    // Tilted ramps — thin slabs rotated off-axis for shading-normal variety.
    spawn(
        "ramp_0".into(),
        Vec3::new(2.6, 0.15, 1.6),
        at(DVec3::new(4.2, f + 1.1, 3.2)).with_rotation(DQuat::from_rotation_z(0.6)),
        &grey,
    );
    spawn(
        "ramp_1".into(),
        Vec3::new(1.8, 0.15, 2.2),
        at(DVec3::new(-1.2, f + 0.9, -3.8)).with_rotation(DQuat::from_rotation_x(0.55)),
        &grey,
    );
    // Equal-lamp 4×4 ceiling grid (uniform lighting isolates the spatial bias from
    // power variance); `--lamp-law pilot` keeps the flood/pilot mix if wanted.
    for k in 0..16u32 {
        let (i, j) = (k % 4, k / 4);
        let power = if args.equal_lamps { 800.0 } else if k % 5 == 4 { 12000.0 } else { 2.0 };
        let lamp = materials.add(StandardSolariMaterial {
            base_color: Color::WHITE,
            emissive: LinearRgba::rgb(power, power, power),
            ..default()
        });
        spawn(
            format!("lamp_{k}"),
            Vec3::new(0.9, 0.1, 0.9),
            at(DVec3::new((i as f64 - 1.5) * 3.8, 2.9, (j as f64 - 1.5) * 3.8)),
            &lamp,
        );
    }
    commands.insert_resource(Probes(vec![
        Probe { name: "floor_open", world: DVec3::new(5.5, -3.0, -5.5), expect: None },
        Probe { name: "floor_amid", world: DVec3::new(-1.2, -3.0, 0.5), expect: None },
        Probe { name: "box_top", world: DVec3::new(0.2, 0.0, 3.0), expect: None },
        Probe { name: "wall_z-", world: DVec3::new(0.0, 0.5, -8.0), expect: None },
    ]));
    commands.spawn((
        Name::new("camera"),
        Camera3d::default(),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariCamera,
        SolariReference {
            samples_per_frame: args.spp,
            nee_off: args.no_nee,
            ris_candidates: args.ris,
            restir: args.restir,
            di_only: args.di_only,
            gi_only: args.gi_only,
            restir_gi: args.restir_gi,
            gi_recon: args.gi_recon,
            gi_temporal: args.gi_temporal,
            gi_spatial: args.gi_spatial,
            accumulate: !args.no_accum,
            spatial: args.spatial,
            spatial_unbiased: args.zcount,
            spatial_taps: args.taps,
            spatial_radius: args.radius,
            spatial_debug: args.spatial_debug,
            gi_dead_view: args.gi_dead,
            ..default()
        },
        NoGpuGlobalTransformReadback,
        FreeCamera::default(),
        Transform::from_translation(DVec3::new(3.5, 1.6, 7.0))
            .looking_at(DVec3::new(-0.5, -2.2, 0.0), Vec3::Y),
    ));
}

/// Rung-3 directional exam: an OPEN yard — sun (directional, NEE-only technique)
/// + a lamp cluster (reservoir technique) lighting the same ground. Restir mode
/// shades the sun per-light and drops the ½ stratum from the emissive pdf; if
/// either side of that bookkeeping is wrong, restir-vs-stratified means drift.
fn setup_yard(
    mut commands: Commands,
    mut meshes: ResMut<Assets<ClusterMesh>>,
    mut materials: ResMut<Assets<StandardSolariMaterial>>,
    args: Res<SceneArgs>,
) {
    let grey = materials.add(StandardSolariMaterial {
        base_color: Color::srgb(0.6, 0.6, 0.6),
        perceptual_roughness: 1.0,
        ..default()
    });
    let mut slab = |name: String, size: Vec3, pos: DVec3, mat: &Handle<StandardSolariMaterial>| {
        let mesh = Cuboid::from_size(size).mesh().build().with_generated_tangents().unwrap();
        let handle = meshes.add(ClusterMesh::try_from(&mesh).expect("slab bake"));
        commands.spawn((
            Name::new(name),
            TransformStatic,
            NoGpuGlobalTransformReadback,
            RaytracingMesh3d(handle),
            SolariMaterial3d(mat.clone()),
            Transform::from_translation(pos),
        ));
    };
    slab("ground".into(), Vec3::new(40.0, 0.3, 40.0), DVec3::new(0.0, -0.15, 0.0), &grey);
    slab("box_a".into(), Vec3::new(2.0, 2.0, 2.0), DVec3::new(-4.0, 1.0, 0.0), &grey);
    slab("box_b".into(), Vec3::new(2.0, 2.0, 2.0), DVec3::new(0.0, 1.0, -3.0), &grey);
    slab("box_c".into(), Vec3::new(2.0, 2.0, 2.0), DVec3::new(5.0, 1.0, 2.0), &grey);
    // Lamp cluster over one corner — keeps the emissive reservoir busy while the
    // sun exercises the deterministic directional arm of the visibility loop.
    for k in 0..4u32 {
        let lamp = materials.add(StandardSolariMaterial {
            base_color: Color::WHITE,
            emissive: LinearRgba::rgb(800.0, 800.0, 800.0),
            ..default()
        });
        slab(
            format!("yard_lamp_{k}"),
            Vec3::new(0.9, 0.1, 0.9),
            DVec3::new(-9.0 + (k % 2) as f64 * 2.0, 4.0, 6.0 + (k / 2) as f64 * 2.0),
            &lamp,
        );
    }
    commands.spawn((
        Name::new("sun"),
        SolariDirectionLight { illuminance: 5_000.0, ..default() },
        Transform::from_translation(DVec3::new(8.0, 16.0, 8.0)).looking_at(DVec3::ZERO, Vec3::Y),
        NoGpuGlobalTransformReadback,
    ));
    commands.insert_resource(Probes(vec![
        Probe { name: "sun_ground", world: DVec3::new(8.0, 0.0, 8.0), expect: None },
        // In box_b's shadow (sun from +x/+y/+z → shadows stretch toward -x/-z).
        Probe { name: "shadow_ground", world: DVec3::new(-1.6, 0.0, -4.8), expect: None },
        Probe { name: "lamp_ground", world: DVec3::new(-8.0, 0.0, 7.0), expect: None },
        Probe { name: "box_face", world: DVec3::new(-4.0, 1.2, 1.01), expect: None },
    ]));
    commands.spawn((
        Name::new("camera"),
        Camera3d::default(),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariCamera,
        SolariReference {
            samples_per_frame: args.spp,
            nee_off: args.no_nee,
            ris_candidates: args.ris,
            restir: args.restir,
            di_only: args.di_only,
            gi_only: args.gi_only,
            restir_gi: args.restir_gi,
            gi_recon: args.gi_recon,
            gi_temporal: args.gi_temporal,
            gi_spatial: args.gi_spatial,
            accumulate: !args.no_accum,
            spatial: args.spatial,
            spatial_unbiased: args.zcount,
            spatial_taps: args.taps,
            spatial_radius: args.radius,
            spatial_debug: args.spatial_debug,
            gi_dead_view: args.gi_dead,
            ..default()
        },
        NoGpuGlobalTransformReadback,
        FreeCamera::default(),
        Transform::from_translation(DVec3::new(2.0, 7.0, 20.0))
            .looking_at(DVec3::new(-1.0, 0.0, 0.0), Vec3::Y),
    ));
}

/// `--production`: swap the reference accumulator for the per-frame ReSTIR
/// stack the moment a scene camera appears (DLSS RR denoises downstream).
fn apply_production(
    mut commands: Commands,
    cameras: Query<Entity, Added<SolariReference>>,
    args: Res<SceneArgs>,
) {
    for entity in &cameras {
        commands.entity(entity).remove::<SolariReference>().insert(SolariRestir {
            spatial_taps: args.taps,
            spatial_radius: args.radius,
            ..default()
        });
    }
}

/// F = freeze reference snapshot, V = toggle |current-frozen| diff view, P = dump PFM,
/// R = toggle ReSTIR temporal reuse (rung-3 live A/B against a frozen reference).
fn rung0_keys(
    input: Res<ButtonInput<KeyCode>>,
    mut fd: ResMut<SolariFreezeDiff>,
    mut cameras: Query<&mut SolariReference>,
) {
    if input.just_pressed(KeyCode::KeyF) {
        fd.freeze_epoch += 1;
    }
    if input.just_pressed(KeyCode::KeyV) {
        fd.diff = !fd.diff;
        info!("diff view: {}", fd.diff);
    }
    if input.just_pressed(KeyCode::KeyP) {
        fd.dump_epoch += 1;
    }
    if input.just_pressed(KeyCode::KeyR) {
        for mut r in &mut cameras {
            r.restir = !r.restir;
            info!("restir temporal: {}", r.restir);
        }
    }
}

/// `--rung0`: scripted freeze(4s) -> dump(5s) -> diff-on(6s) for headless validation.
fn rung0_selftest(
    time: Res<Time>,
    args: Res<SceneArgs>,
    mut fd: ResMut<SolariFreezeDiff>,
    mut stage: Local<u32>,
) {
    if !args.rung0 {
        return;
    }
    let t = time.elapsed_secs();
    if *stage == 0 && t > 4.0 {
        fd.freeze_epoch += 1;
        *stage = 1;
    } else if *stage == 1 && t > 5.0 {
        fd.dump_epoch += 1;
        *stage = 2;
    } else if *stage == 2 && t > 6.0 {
        fd.diff = true;
        info!("diff view: on (self-test)");
        *stage = 3;
    } else if *stage == 3 && t > 30.0 {
        fd.dump_epoch += 1; // late dump = converged ground truth for RMSE
        *stage = 4;
    }
}

fn luminance(c: [f32; 3]) -> f64 {
    0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64
}

/// Normal-incidence slab under a uniform sky: every interreflection order escapes
/// to the same sky, so E = R + (1−R)²·a/(1−R·a) per channel; luminance-weighted.
fn tinted_slab_expect(a: [f64; 3], ior: f64) -> f64 {
    let r = ((ior - 1.0) / (ior + 1.0)).powi(2);
    let e = a.map(|a| r + (1.0 - r).powi(2) * a / (1.0 - r * a));
    luminance([e[0] as f32, e[1] as f32, e[2] as f32])
}

/// Render-world probe: every ~2s copy the RT output buffer (the accumulated mean)
/// to a staging buffer, average a 9×9 patch around each probe's projected pixel,
/// and log means + ratios. PASS/FAIL on `expect` ratios once past 4096 spp.
fn probe_readback(
    views: Query<(
        &RtOutputBuffer,
        Option<&RtAccumulation>,
        &ExtractedCamera,
        &ExtractedView,
    )>,
    probes: Res<Probes>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut staging: Local<Option<Buffer>>,
    mut tick: Local<u32>,
) {
    *tick += 1;
    if *tick % 240 != 0 || probes.0.is_empty() {
        return;
    }
    for (output, accumulation, camera, view) in &views {
        let Some(viewport) = camera.physical_viewport_size else {
            continue;
        };
        let spp = accumulation.map_or(0, |a| a.n);

        let staging = staging.get_or_insert_with(|| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some("furnace_probe_staging"),
                size: output.size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        if staging.size() != output.size {
            *staging = render_device.create_buffer(&BufferDescriptor {
                label: Some("furnace_probe_staging"),
                size: output.size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("furnace_probe_copy"),
        });
        encoder.copy_buffer_to_buffer(&output.buffer, 0, staging, 0, output.size);
        render_queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = render_device.poll(PollType::wait_indefinitely());
        if rx.recv().map(|r| r.is_err()).unwrap_or(true) {
            return;
        }
        let pixels: Vec<[f32; 4]> = slice
            .get_mapped_range()
            .chunks_exact(16)
            .map(|c| [0, 4, 8, 12].map(|o| f32::from_le_bytes(c[o..o + 4].try_into().unwrap())))
            .collect();
        staging.unmap();

        // Project each probe through the view (camera-relative, matching the trace).
        let inv_view = view.world_from_view.affine().inverse();
        let mut means: Vec<(usize, f64, [f32; 3])> = Vec::new();
        for (pi, probe) in probes.0.iter().enumerate() {
            let vpos = inv_view.transform_point3(probe.world.as_vec3());
            let clip = view.clip_from_view * Vec4::new(vpos.x, vpos.y, vpos.z, 1.0);
            if clip.w <= 0.0 {
                continue;
            }
            let ndc = clip.truncate() / clip.w;
            let px = ((ndc.x * 0.5 + 0.5) * viewport.x as f32) as i64;
            let py = ((-ndc.y * 0.5 + 0.5) * viewport.y as f32) as i64;
            let mut sum = [0.0f64; 3];
            let mut count = 0u32;
            for dy in -4i64..=4 {
                for dx in -4i64..=4 {
                    let (x, y) = (px + dx, py + dy);
                    if x < 0 || y < 0 || x >= viewport.x as i64 || y >= viewport.y as i64 {
                        continue;
                    }
                    let p = pixels[(y as u32 * viewport.x + x as u32) as usize];
                    sum[0] += p[0] as f64;
                    sum[1] += p[1] as f64;
                    sum[2] += p[2] as f64;
                    count += 1;
                }
            }
            if count > 0 {
                let m = [
                    (sum[0] / count as f64) as f32,
                    (sum[1] / count as f64) as f32,
                    (sum[2] / count as f64) as f32,
                ];
                means.push((pi, luminance(m), m));
            }
        }

        let sky = means
            .iter()
            .find(|(pi, ..)| probes.0[*pi].name == "sky")
            .map(|&(_, lum, _)| lum);
        let mut line = format!("probes @ {spp} spp:");
        let mut failures = 0u32;
        for (pi, lum, m) in &means {
            let probe = &probes.0[*pi];
            match (sky, probe.expect) {
                (Some(sky), Some(expect)) if sky > 0.0 => {
                    let ratio = lum / sky;
                    let verdict = if spp >= 4096 {
                        failures += ((ratio - expect).abs() > 0.01) as u32;
                        if (ratio - expect).abs() > 0.01 {
                            " FAIL"
                        } else {
                            " ok"
                        }
                    } else {
                        ""
                    };
                    line += &format!(" | {} r={ratio:.4}{verdict}", probe.name);
                }
                _ => {
                    line += &format!(
                        " | {} mean=({:.4},{:.4},{:.4})",
                        probe.name, m[0], m[1], m[2]
                    );
                }
            }
        }
        info!("{line}");
        if spp >= 4096 {
            if failures == 0 {
                info!("furnace: PASS ({} probes)", means.len());
            } else {
                warn!("furnace: {failures} probe(s) FAILED");
            }
        }
    }
}

// ---- Exam-harness plumbing (self-contained; no deps outside bevy) ----

/// Log filter: silence noisy startup INFO lines (keep warn/error).
const LOG_FILTER: &str = concat!(
    "bevy_camera_controller=off",
    ",bevy_winit=warn",
    ",bevy_render=warn",
    ",wgpu_hal=warn",
    ",bevy_diagnostic::system_information_diagnostics_plugin=warn",
    ",bevy_solari::gpu::allocator=warn",
);

#[derive(Resource)]
struct Timeout(Timer);

trait ExamAppExt {
    fn add_timeout_exit(&mut self, duration: Option<f32>, agent_default: f32) -> &mut Self;
    fn add_screenshot(&mut self, trigger: KeyCode) -> &mut Self;
}

impl ExamAppExt for App {
    /// Bounded runs: `--timeout N`, else `CLAUDECODE` (agent/CI) defaults to
    /// `agent_default` seconds; interactive runs never time out.
    fn add_timeout_exit(&mut self, duration: Option<f32>, agent_default: f32) -> &mut Self {
        let secs = duration.or_else(|| std::env::var_os("CLAUDECODE").map(|_| agent_default));
        if let Some(secs) = secs {
            self.insert_resource(Timeout(Timer::from_seconds(secs, TimerMode::Once)))
                .add_systems(Update, |time: Res<Time>, mut t: ResMut<Timeout>| {
                    if t.0.tick(time.delta()).just_finished() {
                        info!("timeout reached, exiting");
                        // Hard exit: skip teardown (device-heavy shutdown otherwise
                        // segfaults after success and taints the exam's exit code).
                        use std::io::Write;
                        let _ = std::io::stdout().flush();
                        let _ = std::io::stderr().flush();
                        std::process::exit(0);
                    }
                });
        }
        self
    }

    /// Manual screenshots on `trigger`; headless/agent runs (no keyboard) can set
    /// `AUTO_SCREENSHOT_MS=8000` to capture once that long after startup.
    fn add_screenshot(&mut self, trigger: KeyCode) -> &mut Self {
        fn take(commands: &mut Commands) {
            let dir = std::path::Path::new("./target/tmp");
            let _ = std::fs::create_dir_all(dir);
            let ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            let path = dir.join(format!("screenshot-{ms}.png"));
            info!("Screenshot saving to {}", path.display());
            commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
        }
        self.add_systems(
            Update,
            move |mut commands: Commands, input: Res<ButtonInput<KeyCode>>| {
                if input.just_pressed(trigger) {
                    take(&mut commands);
                }
            },
        )
        .add_systems(
            Update,
            |mut commands: Commands, time: Res<Time>, mut taken: Local<bool>| {
                if *taken {
                    return;
                }
                let Some(ms) = std::env::var("AUTO_SCREENSHOT_MS")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                else {
                    *taken = true; // not requested — check the env var only once
                    return;
                };
                if time.elapsed_secs_f64() * 1000.0 >= ms as f64 {
                    *taken = true;
                    take(&mut commands);
                }
            },
        )
    }
}

/// Motion soak: sway + yaw around the spawn pose, snap back exactly at `secs`.
#[derive(Resource)]
struct OrbitSoak {
    secs: f32,
    start: Option<Transform>,
}

fn orbit_soak(
    time: Res<Time>,
    mut orbit: ResMut<OrbitSoak>,
    mut cams: Query<&mut Transform, With<SolariCamera>>,
) {
    if orbit.secs <= 0.0 {
        return;
    }
    let Ok(mut tf) = cams.single_mut() else {
        return;
    };
    if orbit.start.is_none() {
        orbit.start = Some(*tf);
    }
    let start = orbit.start.unwrap();
    let t = time.elapsed_secs();
    if t >= orbit.secs + 1.0 {
        *tf = start;
        return;
    }
    // Settle phase: near-final pose while motion-contaminated history heals
    // (m-cap frames), then the exact pose forces one clean accumulation restart.
    if t >= orbit.secs {
        *tf = start;
        tf.translation += start.rotation * DVec3::new(1.0e-4, 0.0, 0.0);
        return;
    }
    let sway = (t * 0.9).sin() as f64 * 1.0;
    let bob = (t * 1.3).sin() as f64 * 0.25;
    *tf = start;
    let offset = start.rotation * DVec3::new(sway, bob, 0.0);
    tf.translation += offset;
    tf.rotate_y(((t * 0.55).sin() * 0.15) as f64);
}
