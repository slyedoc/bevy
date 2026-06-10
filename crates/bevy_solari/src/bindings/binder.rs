use super::extract::SolariMaterialAssets;
use crate::material::MaterialSlots;
use crate::accel::deform::Deform;
use crate::accel::ptlas::Ptlas;
use crate::instance::InstanceManager;
use crate::geometry::ClusterMeshManager;
use bevy_asset::Handle;
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
};
use bevy_math::Vec3;
use bevy_pbr::DfgLut;

use crate::lights::{ActiveDirectionalLights, EmissiveLights, GpuLightSource};
use bevy_platform::collections::HashMap;
use bevy_render::{
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::{FallbackImage, GpuImage},
};
use core::{hash::Hash, num::NonZeroU32, ops::Deref};

const MAX_TEXTURE_COUNT: NonZeroU32 = NonZeroU32::new(5_000).unwrap();

const TEXTURE_MAP_NONE: u32 = u32::MAX;

#[derive(Resource)]
pub struct RaytracingSceneBindings {
    pub bind_group: Option<BindGroup>,
    pub bind_group_layout: BindGroupLayoutDescriptor,
}

pub fn prepare_raytracing_scene_bindings(
    active_lights: Res<ActiveDirectionalLights>,
    emissive_lights: Res<EmissiveLights>,
    
    instance_manager: Res<InstanceManager>,
    cluster_mesh_manager: Res<ClusterMeshManager>,
    ptlas: Option<Res<Ptlas>>,
    deform: Option<Res<Deform>>,
    material_assets: Res<SolariMaterialAssets>,
    material_slots: Res<MaterialSlots>,
    texture_assets: Res<RenderAssets<GpuImage>>,
    fallback_texture: Res<FallbackImage>,
    dfg_lut: Res<DfgLut>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    render_queue: Res<RenderQueue>,
    mut raytracing_scene_bindings: ResMut<RaytracingSceneBindings>,
) {
    raytracing_scene_bindings.bind_group = None;

    let active_count = instance_manager.active_count();
    if active_count == 0 {
        return;
    }
    // Cluster mesh-pool buffers have no committed sparse pages until at least one
    // mesh has been uploaded; reading unbacked pages would fault. Bail for one
    // frame until `perform_pending_cluster_mesh_writes` has committed + filled them.
    if cluster_mesh_manager.vertex_positions.is_empty() {
        return;
    }
    // PTLAS must be built — it's the AS the bind-group's TLAS slot
    // references. Bail until ptlas.tlas is ready (first frame after
    // CLAS upload + BLAS build + PTLAS build complete).
    let Some(ptlas) = ptlas else {
        return;
    };
    let Some(tlas) = ptlas.current_tlas() else {
        return;
    };
    // The deform pool + animated table are required scene-group bindings (the
    // resolve shader always reads `instance_animated[instance_id]`). `Deform` is
    // created unconditionally in `RenderStartup`, so this is present whenever the
    // cluster pipeline is.
    let Some(deform) = deform else {
        return;
    };

    let mut textures = CachedBindingArray::new();
    let mut samplers = Vec::new();
    let mut materials = StorageBufferList::<GpuMaterial>::default();
    // Per-instance `transforms` / `previous_frame_transforms` /
    // `material_ids` are slot-indexed GPU columns now (`GpuInstances`,
    // scattered from a delta) — bound directly, not rebuilt here. Same
    // for `instance_cluster_ranges`, which reuses the slot-indexed
    // `instance_lod_inputs` column. Only the light buffers are still
    // built per frame.
    let mut light_sources = StorageBufferList::<GpuLightSource>::default();

    let mut process_texture = |texture_handle: &Option<Handle<_>>| -> Option<u32> {
        match texture_handle {
            Some(texture_handle) => match texture_assets.get(texture_handle.id()) {
                Some(texture) => {
                    let (texture_id, is_new) =
                        textures.push_if_absent(texture.texture_view.deref(), texture_handle.id());
                    if is_new {
                        samplers.push(texture.sampler.deref());
                    }
                    Some(texture_id)
                }
                None => None,
            },
            None => Some(TEXTURE_MAP_NONE),
        }
    };

    // `materials[]` is indexed by **stable** material slot
    // ([`MaterialSlots`]), so `material_id` is stable per instance and
    // the array order never churns. Freed slots leave a default (black)
    // hole. Sized to the slot high-water.
    let material_count = material_slots.len() as usize;
    if material_count == 0 {
        return;
    }
    materials.get_mut().resize(material_count, GpuMaterial::default());
    for (asset_id, material) in material_assets.iter() {
        let Some(slot) = material_slots.slot_of(*asset_id) else {
            continue;
        };
        let Some(base_color_texture_id) = process_texture(&material.base_color_texture) else {
            continue;
        };
        let Some(normal_map_texture_id) = process_texture(&material.normal_map_texture) else {
            continue;
        };
        let Some(emissive_texture_id) = process_texture(&material.emissive_texture) else {
            continue;
        };
        let Some(metallic_roughness_texture_id) =
            process_texture(&material.metallic_roughness_texture)
        else {
            continue;
        };

        let emissive_vec3 = material.emissive.to_vec3();

        materials.get_mut()[slot as usize] = GpuMaterial {
            normal_map_texture_id,
            base_color_texture_id,
            emissive_texture_id,
            metallic_roughness_texture_id,

            base_color: LinearRgba::from(material.base_color).to_vec3(),
            perceptual_roughness: material.perceptual_roughness,
            emissive: emissive_vec3,
            metallic: material.metallic,
            reflectance: material.reflectance,
            _padding: Default::default(),
        };
    }

    if textures.is_empty() {
        textures.vec.push(fallback_texture.d2.texture_view.deref());
        samplers.push(fallback_texture.d2.sampler.deref());
    }

    // Seed this frame's light list with the cached emissive-mesh lights
    // (`crate::lights::EmissiveLights`, rebuilt change-driven); directional lights
    // are appended below.
    *light_sources.get_mut() = emissive_lights.lights.clone();

    // Directional lights: the `directional_lights` column buffer (settings +
    // GPU-resolved direction) is owned by `crate::lights`; here we only append each
    // active light to the light-source index list — its `directional_light_id` is
    // its stable **table slot** (the column is slot-indexed).
    for &(_entity, slot) in &active_lights.0 {
        light_sources
            .get_mut()
            .push(GpuLightSource::new_directional_light(slot));
    }

    if light_sources.get().len() > u16::MAX as usize {
        panic!("Too many light sources in the scene, maximum is 65535.");
    }

    materials.write_buffer(&render_device, &render_queue);
    light_sources.write_buffer(&render_device, &render_queue);

    // PTLAS is built by `ptlas::dispatch_ptlas`; no TLAS build here.

    let (dfg_view, dfg_sampler) = texture_assets
        .get(&dfg_lut.texture)
        .map(|img| (&img.texture_view, &img.sampler))
        .unwrap_or((
            &fallback_texture.d2.texture_view,
            &fallback_texture.d2.sampler,
        ));

    // Per-instance `(cluster_base, cluster_count)` reuses the slot-indexed
    // `lod_inputs` GPU column (scattered by `gpu_instances`).
    // `transforms` / `previous_frame_transforms` / `material_ids` /
    // `directional_lights` / `instance_cluster_ranges` are GPU columns — bound from
    // the shared `ecs_gpu::SceneColumns` group (built generically from each column's
    // `SCENE_BINDING`), not assembled here.
    raytracing_scene_bindings.bind_group = Some(render_device.create_bind_group(
        "raytracing_scene_bind_group",
        &pipeline_cache.get_bind_group_layout(&raytracing_scene_bindings.bind_group_layout),
        &BindGroupEntries::sequential((
            cluster_mesh_manager.vertex_positions.binding(),
            cluster_mesh_manager.vertex_normals.binding(),
            cluster_mesh_manager.vertex_tangents.binding(),
            cluster_mesh_manager.vertex_uvs.binding(),
            cluster_mesh_manager.indices.binding(),
            cluster_mesh_manager.clusters.binding(),
            materials.binding().unwrap(),
            textures.as_slice(),
            samplers.as_slice(),
            tlas.as_binding(),
            light_sources.binding().unwrap(),
            dfg_view,
            dfg_sampler,
            // Skeletal animation: deform pool + slot-indexed animated table.
            deform.positions.as_entire_binding(),
            deform.normals.as_entire_binding(),
            deform.animated_table().as_entire_binding(),
            deform.tangents.as_entire_binding(),
        )),
    ));
}

impl RaytracingSceneBindings {
    pub fn new() -> Self {
        Self {
            bind_group: None,
            bind_group_layout: BindGroupLayoutDescriptor::new(
                "raytracing_scene_bind_group_layout",
                // `transforms` / `previous_frame_transforms` / `material_ids` /
                // `directional_lights` / `instance_cluster_ranges` are GPU columns,
                // now bound from the shared `ecs_gpu::SceneColumns` group — not here.
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        // Cluster mesh pool (shared across instances)
                        storage_buffer_read_only_sized(false, None), // 0: vertex_positions (array<f32>)
                        storage_buffer_read_only_sized(false, None), // 1: vertex_normals (octahedral u32)
                        storage_buffer_read_only_sized(false, None), // 2: vertex_tangents
                        storage_buffer_read_only_sized(false, None), // 3: vertex_uvs
                        storage_buffer_read_only_sized(false, None), // 4: cluster_indices
                        storage_buffer_read_only_sized(false, None), // 5: clusters
                        // Materials + textures
                        storage_buffer_read_only_sized(false, None), // 6: materials
                        texture_2d(TextureSampleType::Float { filterable: true })
                            .count(MAX_TEXTURE_COUNT),               // 7: textures
                        sampler(SamplerBindingType::Filtering).count(MAX_TEXTURE_COUNT), // 8: samplers
                        // Ray-tracing AS + lighting
                        acceleration_structure(),                    // 9: tlas
                        storage_buffer_read_only_sized(false, None), // 10: light_sources
                        // BRDF DFG LUT
                        texture_2d(TextureSampleType::Float { filterable: true }), // 11
                        sampler(SamplerBindingType::Filtering),      // 12
                        // Skeletal animation: deform pool + per-instance table
                        storage_buffer_read_only_sized(false, None), // 13: deform_positions
                        storage_buffer_read_only_sized(false, None), // 14: deform_normals
                        storage_buffer_read_only_sized(false, None), // 15: instance_animated
                        storage_buffer_read_only_sized(false, None), // 16: deform_tangents
                    ),
                ),
            ),
        }
    }
}

impl Default for RaytracingSceneBindings {
    fn default() -> Self {
        Self::new()
    }
}

struct CachedBindingArray<T, I: Eq + Hash> {
    map: HashMap<I, u32>,
    vec: Vec<T>,
}

impl<T, I: Eq + Hash> CachedBindingArray<T, I> {
    fn new() -> Self {
        Self {
            map: HashMap::default(),
            vec: Vec::default(),
        }
    }

    fn push_if_absent(&mut self, item: T, item_id: I) -> (u32, bool) {
        let mut is_new = false;
        let i = *self.map.entry(item_id).or_insert_with(|| {
            is_new = true;
            let i = self.vec.len() as u32;
            self.vec.push(item);
            i
        });
        (i, is_new)
    }

    fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    fn as_slice(&self) -> &[T] {
        self.vec.as_slice()
    }
}

type StorageBufferList<T> = StorageBuffer<Vec<T>>;

#[derive(ShaderType, Clone, Default)]
struct GpuMaterial {
    normal_map_texture_id: u32,
    base_color_texture_id: u32,
    emissive_texture_id: u32,
    metallic_roughness_texture_id: u32,

    base_color: Vec3,
    perceptual_roughness: f32,
    emissive: Vec3,
    metallic: f32,
    _padding: Vec3,
    reflectance: f32,
}

// `GpuLightSource` + `GpuDirectionalLight` now live in `crate::lights` (the lights
// table owns them; emissive + directional light-source lists are built there,
// direction resolved GPU-side); the binder only binds its buffer + builds the
// light-source index list.

