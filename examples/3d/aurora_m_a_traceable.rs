// Aurora is intrinsically unsafe (drives Vulkan via as_hal escapes).
#![allow(unsafe_code)]
//! M-A smoke test for `bevy_aurora`: load `bunny.meshlet_mesh`, attach an
//! [`AuroraMeshlet3d`] to a single entity, and verify Aurora's
//! `ClusterAsManager` builds CLAS + BLAS for the asset and `TlasManager`
//! builds a TLAS containing one instance pointing at it.
//!
//! Logs the resulting per-mesh CLAS count + BLAS device address + TLAS
//! device address. **No rendering** -- M-B adds RT primary visibility on top.
//!
//! ```text
//! cargo run --release \
//!     --features experimental_cluster_acceleration_structure,bevy_image/zstd_rust \
//!     --example aurora_m_a_traceable
//! ```
//!
//! Requires an NVIDIA GPU (Turing+) with a driver implementing
//! `VK_NV_cluster_acceleration_structure`.

use bevy::{
    pbr::experimental::meshlet::{MeshletMesh, MeshletPlugin},
    prelude::*,
    render::{
        render_resource::WgpuFeatures, settings::WgpuSettings, Render, RenderApp, RenderPlugin,
        RenderSystems,
    },
};
use bevy::aurora::{
    prelude::*,
    primary::AuroraCamera,
    scene::{AuroraMeshlet3d, ClusterAsManager, TlasManager, UploadedMeshes},
};
use bevy::ecs::schedule::IntoScheduleConfigs;

const ASSET_URL: &str =
    "https://github.com/bevyengine/bevy_asset_files/raw/6dccaef517bde74d1969734703709aead7211dbc/meshlet/bunny.meshlet_mesh";

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(RenderPlugin {
                render_creation: WgpuSettings {
                    features: WgpuFeatures::EXPERIMENTAL_RAY_QUERY
                        | WgpuFeatures::EXPERIMENTAL_CLUSTER_ACCELERATION_STRUCTURE,
                    ..default()
                }
                .into(),
                ..default()
            }),
            MeshletPlugin {
                cluster_buffer_slots: 1 << 14,
            },
            AuroraPlugins,
            AuroraDumpPlugin,
        ))
        .insert_resource(BunnyAsset(Handle::default()))
        .add_systems(Startup, (load_bunny, setup_camera))
        .add_systems(Update, spawn_when_ready)
        .run();
}

fn setup_camera(mut commands: Commands) {
    use bevy::render::render_resource::TextureUsages;
    commands.spawn((
        Camera3d::default(),
        bevy::camera::Hdr,
        Msaa::Off,
        bevy::camera::CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Transform::from_xyz(0.5, 0.7, 1.5).looking_at(Vec3::new(0.0, 0.4, 0.0), Vec3::Y),
        AuroraCamera,
    ));
}

#[derive(Resource)]
struct BunnyAsset(Handle<MeshletMesh>);

#[derive(Component)]
struct BunnySpawned;

fn load_bunny(asset_server: Res<AssetServer>, mut bunny: ResMut<BunnyAsset>) {
    bunny.0 = asset_server.load(ASSET_URL);
    info!("Loading bunny.meshlet_mesh from {ASSET_URL}");
}

fn spawn_when_ready(
    mut commands: Commands,
    bunny: Res<BunnyAsset>,
    meshes: Res<Assets<MeshletMesh>>,
    spawned: Query<&BunnySpawned>,
) {
    if !spawned.is_empty() {
        return;
    }
    if meshes.get(&bunny.0).is_none() {
        return;
    }
    info!("Bunny asset ready -- spawning AuroraMeshlet3d");
    commands.spawn((
        AuroraMeshlet3d(bunny.0.clone()),
        Transform::default(),
        GlobalTransform::default(),
        BunnySpawned,
    ));
}

/// Inline plugin that logs Aurora's AS state every frame after `rebuild_tlas`.
struct AuroraDumpPlugin;

impl Plugin for AuroraDumpPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(
            Render,
            dump_state.in_set(RenderSystems::Cleanup),
        );
    }
}

fn dump_state(
    manager: Option<Res<ClusterAsManager>>,
    uploaded: Option<Res<UploadedMeshes>>,
    tlas: Option<Res<TlasManager>>,
) {
    let (Some(manager), Some(uploaded), Some(tlas)) = (manager, uploaded, tlas) else {
        return;
    };
    let n_meshes = uploaded.0.len();
    if n_meshes == 0 || !tlas.is_built() {
        return;
    }
    // Print once per uploaded mesh (skip the no-op subsequent frames).
    static LOGGED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let prev = LOGGED.swap(n_meshes, std::sync::atomic::Ordering::Relaxed);
    if prev == n_meshes {
        return;
    }

    info!("aurora M-A summary:");
    info!("  uploaded meshes: {n_meshes}");
    for (handle, mc_handle) in &uploaded.0 {
        let mc = manager.mesh(*mc_handle);
        info!(
            "    mesh {:?}: {} CLASes, BLAS @ {:#018x}",
            handle.id(),
            mc.clas_addresses.len(),
            mc.blas_address,
        );
    }
    info!(
        "  TLAS: {} instances (wrapped as wgpu::Tlas)",
        tlas.last_built_instance_count(),
    );
}
