//! Visual test for Solari alpha-test shadow ray support.
//!
//! Sets up a ground plane and a vertical "lattice" quad with AlphaMode::Mask,
//! lit by a directional light from above. With correct any-hit / alpha-test
//! shadow rays, the ground shadow should show a checkerboard pattern (light
//! leaks through alpha-rejected texels). Without, the shadow is a solid rect.

use bevy::{
    camera::CameraMainTextureUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    image::{Image, ImageSampler, ImageSamplerDescriptor},
    mesh::Indices,
    prelude::*,
    render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureUsages,
    },
    solari::prelude::{RaytracingMesh3d, SolariLighting, SolariPlugins},
};

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::{
    Dlss, DlssProjectId, DlssRayReconstructionFeature, DlssRayReconstructionSupported,
};

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy::asset::uuid::uuid!(
        "5417916c-0291-4e3f-8f65-326c1858ab96"
    )));

    app.add_plugins((DefaultPlugins, SolariPlugins, FreeCameraPlugin))
        .add_systems(Startup, setup);

    app.run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] dlss_rr_supported: Option<
        Res<DlssRayReconstructionSupported>,
    >,
) {
    // Ground plane — diffuse white, large so shadow is obvious
    let ground_mesh = meshes.add(make_plane(20.0));
    let ground_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.9, 0.9),
        perceptual_roughness: 1.0,
        ..default()
    });
    commands.spawn((
        Mesh3d(ground_mesh.clone()),
        RaytracingMesh3d(ground_mesh),
        MeshMaterial3d(ground_mat),
        Transform::default(),
    ));

    // Alpha-masked lattice quad — vertical, held above ground so the light
    // passes through its alpha-rejected texels onto the ground.
    let lattice_tex = images.add(make_checker_alpha_texture(16, 16));
    let lattice_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.8, 0.2),
        base_color_texture: Some(lattice_tex),
        alpha_mode: AlphaMode::Mask(0.5),
        double_sided: true,
        cull_mode: None,
        perceptual_roughness: 1.0,
        ..default()
    });
    let lattice_mesh = meshes.add(make_plane(4.0));
    commands.spawn((
        Mesh3d(lattice_mesh.clone()),
        RaytracingMesh3d(lattice_mesh),
        MeshMaterial3d(lattice_mat),
        Transform::from_xyz(0.0, 3.0, 0.0)
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
    ));

    // Directional light from overhead-right — casts lattice shadow down onto ground
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -std::f32::consts::FRAC_PI_3,
            std::f32::consts::FRAC_PI_6,
            0.0,
        )),
    ));

    // Camera — angled view of ground + lattice so shadow is visible
    #[allow(unused_mut)]
    let mut camera = commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(0.05, 0.05, 0.08)),
            ..default()
        },
        FreeCamera {
            walk_speed: 3.0,
            run_speed: 10.0,
            ..default()
        },
        Transform::from_xyz(6.0, 5.0, 8.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariLighting::default(),
    ));

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    if dlss_rr_supported.is_some() {
        camera.insert(Dlss::<DlssRayReconstructionFeature> {
            perf_quality_mode: default(),
            reset: default(),
            _phantom_data: default(),
        });
    }
    let _ = camera;

    // UI text describing expected behavior
    commands.spawn((
        Text::new(
            "SolariLighting alpha-test shadow ray test\n\
             Lattice renders transparent (rasterized primary view).\n\
             Ground shows checkerboard shadow — raytraced shadow/GI rays\n\
             alpha-test candidate intersections via ray queries.",
        ),
        Node {
            position_type: PositionType::Absolute,
            top: px(12.0),
            left: px(12.0),
            ..default()
        },
    ));
}

/// Build a flat plane (XZ) of given size, with UV_0, NORMAL, TANGENT, U32 indices
/// so it is Solari-compatible.
fn make_plane(size: f32) -> Mesh {
    let h = size * 0.5;
    let mut mesh = Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [-h, 0.0, -h],
            [h, 0.0, -h],
            [h, 0.0, h],
            [-h, 0.0, h],
        ],
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        vec![[0.0, 1.0, 0.0]; 4],
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
    );
    mesh.insert_indices(Indices::U32(vec![0, 2, 1, 0, 3, 2]));
    mesh.generate_tangents().unwrap();
    mesh
}

/// Generate an RGBA checkerboard where alpha alternates 0 / 255.
fn make_checker_alpha_texture(cols: u32, rows: u32) -> Image {
    let mut data = Vec::with_capacity((cols * rows * 4) as usize);
    for y in 0..rows {
        for x in 0..cols {
            let opaque = (x + y) % 2 == 0;
            data.extend_from_slice(&[255, 255, 255, if opaque { 255 } else { 0 }]);
        }
    }
    let mut image = Image::new(
        Extent3d {
            width: cols,
            height: rows,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::RENDER_WORLD | bevy::asset::RenderAssetUsages::MAIN_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mag_filter: bevy::image::ImageFilterMode::Nearest,
        min_filter: bevy::image::ImageFilterMode::Nearest,
        ..default()
    });
    image
}
