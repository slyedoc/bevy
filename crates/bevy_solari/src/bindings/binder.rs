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

use crate::lights::{GpuLightSource, LightSources};
use bevy_platform::collections::HashMap;
use bevy_render::{
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::{FallbackImage, GpuImage},
};
use core::{hash::Hash, num::NonZeroU32, ops::Deref};

pub(crate) const MAX_TEXTURE_COUNT: NonZeroU32 = NonZeroU32::new(5_000).unwrap();

const TEXTURE_MAP_NONE: u32 = u32::MAX;

#[derive(Resource)]
pub struct RaytracingSceneBindings {
    pub bind_group: Option<BindGroup>,
    pub bind_group_layout: BindGroupLayoutDescriptor,
}

/// Hair scene-group dependencies, bundled so the binder stays under the 16
/// system-param limit. Each is `Option` (absent on non-solari devices).
#[derive(bevy_ecs::system::SystemParam)]
pub struct HairSceneDeps<'w> {
    manager: Option<Res<'w, crate::hair::HairManager>>,
    instances: Option<Res<'w, crate::hair::HairInstances>>,
    propagate: Option<Res<'w, crate::transform::TransformPropagate>>,
}

pub fn prepare_raytracing_scene_bindings(
    lights: Res<LightSources>,
    
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
    hair: HairSceneDeps,
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
    // Hair scene data (segments + instance records + index range). Both are
    // created in `RenderStartup` whenever the cluster pipeline is, so this is
    // present on any solari-capable device; the buffers are empty when there's
    // no hair (the path tracer gates reads on the hair instance count).
    let (Some(hair_manager), Some(hair_instances), Some(transform_propagate)) =
        (hair.manager, hair.instances, hair.propagate)
    else {
        return;
    };
    let Some(hair_params) = hair_instances.params.binding() else {
        return;
    };
    let Some(hair_instance_buffer) = hair_instances.buffer.buffer() else {
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

        // Texture-size term of the ray-cone LOD, from the base-color texture
        // (representative of the material's maps). 0 ⇒ no base-color texture.
        let texel_lod_bias = material
            .base_color_texture
            .as_ref()
            .and_then(|handle| texture_assets.get(handle.id()))
            .map(|image| {
                let size = image.texture_descriptor.size;
                0.5 * ((size.width * size.height) as f32).log2()
            })
            .unwrap_or(0.0);

        let emissive_vec3 = material.emissive.to_vec3();

        // Beer–Lambert: attenuation_color remains after attenuation_distance,
        // so σ = -ln(color) / distance. Infinite distance ⇒ a clear volume.
        let extinction = if material.attenuation_distance.is_finite()
            && material.attenuation_distance > 0.0
        {
            let c = LinearRgba::from(material.attenuation_color).to_vec3();
            -Vec3::new(
                c.x.clamp(1e-4, 1.0).ln(),
                c.y.clamp(1e-4, 1.0).ln(),
                c.z.clamp(1e-4, 1.0).ln(),
            ) / material.attenuation_distance
        } else {
            Vec3::ZERO
        };

        materials.get_mut()[slot as usize] = GpuMaterial {
            normal_map_texture_id,
            base_color_texture_id,
            emissive_texture_id,
            metallic_roughness_texture_id,

            base_color: LinearRgba::from(material.base_color).to_vec3(),
            perceptual_roughness: material.perceptual_roughness,
            emissive: emissive_vec3,
            metallic: material.metallic,
            extinction,
            reflectance: material.reflectance,
            ior: material.ior,
            specular_transmission: material.specular_transmission,
            nested_priority: material.nested_priority,
            alpha_mask: material.traversal_alpha_cutoff(),
            dispersion: material.dispersion.max(0.0),
            texel_lod_bias,
        };
    }

    if textures.is_empty() {
        textures.vec.push(fallback_texture.d2.texture_view.deref());
        samplers.push(fallback_texture.d2.sampler.deref());
    }

    // The light-source table (`crate::lights::LightSources`, rebuilt
    // change-driven) is **stable-slot indexed** — reservoirs / light tiles
    // store the slot as their light identity — plus the dense active list the
    // uniform pick samples from. A `NONE` placeholder keeps the bindings valid
    // in a lightless scene (the active list's zero counts gate sampling).
    *light_sources.get_mut() = lights.table.clone();
    if light_sources.get().is_empty() {
        light_sources.get_mut().push(GpuLightSource::NONE);
    }
    let mut active_light_list = StorageBufferList::<u32>::default();
    *active_light_list.get_mut() = lights.active.clone();
    if active_light_list.get().is_empty() {
        active_light_list.get_mut().extend([0u32, 0u32]);
    }

    materials.write_buffer(&render_device, &render_queue);
    light_sources.write_buffer(&render_device, &render_queue);
    active_light_list.write_buffer(&render_device, &render_queue);

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
            active_light_list.binding().unwrap(),
            // Hair: per-segment `{p0,r0, p1,r1}` records, per-instance records, range,
            // and the transform-table world buffer (instance world matrices).
            hair_manager.segments.buffer().as_entire_binding(),
            hair_instance_buffer.as_entire_binding(),
            hair_params,
            transform_propagate.current_world().as_entire_binding(),
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
                    // COMPUTE for the megakernel/restir path; the RT-pipeline
                    // stages so the same bind group is visible to raygen + the
                    // hit/miss shaders when bound into the raw RT pipeline.
                    ShaderStages::COMPUTE
                        | ShaderStages::RAY_GENERATION
                        | ShaderStages::CLOSEST_HIT
                        | ShaderStages::ANY_HIT
                        | ShaderStages::MISS,
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
                        storage_buffer_read_only_sized(false, None), // 17: active_light_list
                        // Hair
                        storage_buffer_read_only_sized(false, None), // 18: hair_segments
                        storage_buffer_read_only_sized(false, None), // 19: hair_instances
                        storage_buffer_read_only_sized(false, None), // 20: hair_params
                        storage_buffer_read_only_sized(false, None), // 21: hair_world (transform table)
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

#[derive(ShaderType, Clone)]
struct GpuMaterial {
    normal_map_texture_id: u32,
    base_color_texture_id: u32,
    emissive_texture_id: u32,
    metallic_roughness_texture_id: u32,

    base_color: Vec3,
    perceptual_roughness: f32,
    emissive: Vec3,
    metallic: f32,
    // Beer–Lambert extinction σ (1/world-unit), precomputed from
    // attenuation color/distance.
    extinction: Vec3,
    reflectance: f32,
    ior: f32,
    specular_transmission: f32,
    nested_priority: u32,
    // Alpha-mask cutoff; negative = opaque (no alpha test during traversal).
    alpha_mask: f32,
    // Chromatic dispersion (20/Abbe, `KHR_materials_dispersion`); 0 = none.
    dispersion: f32,
    // `0.5·log2(width·height)` of the base-color texture — the texture-size
    // term of the ray-cone texture LOD, baked here so the path tracer doesn't
    // query `textureDimensions` per hit (a bindless descriptor fetch that
    // crushed warp occupancy). Shared across the material's maps (they're
    // near-always the same resolution).
    texel_lod_bias: f32,
}

impl Default for GpuMaterial {
    fn default() -> Self {
        Self {
            normal_map_texture_id: 0,
            base_color_texture_id: 0,
            emissive_texture_id: 0,
            metallic_roughness_texture_id: 0,
            base_color: Vec3::ZERO,
            perceptual_roughness: 0.0,
            emissive: Vec3::ZERO,
            metallic: 0.0,
            extinction: Vec3::ZERO,
            reflectance: 0.0,
            ior: 1.0,
            specular_transmission: 0.0,
            nested_priority: 0,
            // Opaque: freed material slots must not alpha-test in traversal.
            alpha_mask: -1.0,
            dispersion: 0.0,
            texel_lod_bias: 0.0,
        }
    }
}

// `GpuLightSource` + `GpuDirectionalLight` now live in `crate::lights` (the lights
// table owns them; emissive + directional light-source lists are built there,
// direction resolved GPU-side); the binder only binds its buffer + builds the
// light-source index list.

