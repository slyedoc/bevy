use super::extract::SolariMaterialAssets;
use crate::material::MaterialSlots;
use crate::accel::ptlas::Ptlas;
use crate::instance::InstanceManager;
use crate::geometry::ClusterMeshManager;
use bevy_asset::Handle;
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use crate::gpu::allocator::Allocator;
use crate::gpu::stable_storage_buffer::StableStorageBuffer;
use crate::gpu::RawTraceBindable;
use bevy_math::{UVec4, Vec3};
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

/// Size of the layered `texture_arrays` pool — must match the sized
/// `binding_array<texture_2d_array<f32>, 16>` in `raytracing_scene_bindings.wgsl`.
pub(crate) const MAX_TEXTURE_ARRAY_COUNT: NonZeroU32 = NonZeroU32::new(16).unwrap();

const TEXTURE_MAP_NONE: u32 = u32::MAX;

#[derive(Resource)]
pub struct RaytracingSceneBindings {
    pub bind_group: Option<BindGroup>,
    pub bind_group_layout: BindGroupLayoutDescriptor,
    /// Stable device address of the materials storage buffer, exposed so the
    /// RT-pipeline path can reach it by `physical_load<Material>` instead of the
    /// bound `materials` array. Captured from the stable-address
    /// [`StableStorageBuffer`](crate::gpu::stable_storage_buffer::StableStorageBuffer)
    /// via [`RawTraceBindable`](crate::gpu::RawTraceBindable), so it stays valid for
    /// any in-flight trace. `0` until the first bind-group build.
    pub materials_device_address: crate::gpu::allocator::StableAddr,
}

/// The scene's per-frame-rebuilt tables, on persistent **stable-address**
/// buffers. The RT trace reads them by device address (`materials`) or descriptor
/// (`light_sources` / `active_light_list`); a stable handle/address that's never
/// freed means an in-flight trace can never read a reallocated/freed buffer.
/// Built once in [`init_solari_scene_buffers`] and overwritten each frame.
/// (`directional_lights` is a `GpuColumn`.)
#[derive(Resource)]
pub struct SolariSceneBuffers {
    /// `array<Material>`, read bindlessly by `physical_load<Material>`.
    materials: StableStorageBuffer<Vec<GpuMaterial>>,
    /// Slot-indexed light table (scene bind group binding 5).
    light_sources: StableStorageBuffer<Vec<GpuLightSource>>,
    /// Uniform-pick active-light list (scene bind group binding 8).
    active_light_list: StableStorageBuffer<Vec<u32>>,
    /// Trilinear REPEAT sampler shared by every `texture_arrays` entry (binding 14)
    /// — tiling terrain layers; per-image samplers would be clamp-to-edge.
    array_sampler: Sampler,
}

/// `RenderStartup` (after `SolariSetup`): build the persistent scene buffers.
/// Skipped on a device without the cluster allocator (no solari RT support).
pub fn init_solari_scene_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
) {
    let Some(allocator) = allocator else {
        return;
    };
    commands.insert_resource(SolariSceneBuffers {
        materials: StableStorageBuffer::new(Vec::new(), &allocator, &render_device, "solari.materials"),
        light_sources: StableStorageBuffer::new(Vec::new(), &allocator, &render_device, "solari.light_sources"),
        active_light_list: StableStorageBuffer::new(Vec::new(), &allocator, &render_device, "solari.active_light_list"),
        array_sampler: render_device.create_sampler(&SamplerDescriptor {
            label: Some("solari.texture_arrays_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        }),
    });
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
    scene_buffers: Option<ResMut<SolariSceneBuffers>>,
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
    // Materials live in a persistent, stable-address `StableStorageBuffer` (built
    // once in `init_solari_scene_buffers`). The RT chit reads them by device
    // address (`physical_load<Material>`); a stable address that's never freed
    // means an in-flight trace can't read a reallocated/freed materials buffer.
    // `SHADER_DEVICE_ADDRESS` comes from the sparse buffer itself.
    let Some(mut scene_buffers) = scene_buffers else {
        return;
    };
    // Disjoint &mut to each persistent buffer for this frame's populate.
    let SolariSceneBuffers {
        materials,
        light_sources,
        active_light_list,
        array_sampler,
    } = &mut *scene_buffers;
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

    // Layered pool (`texture_arrays`, binding 13). Only D2Array views may enter —
    // a plain D2 image would fail bind-group validation. Unlike flat textures a
    // still-loading array degrades to NONE (not a material skip): array-painted
    // chits carry a procedural fallback, so they shade flat until the pop-in.
    let mut texture_arrays = CachedBindingArray::new();
    let mut process_texture_array = |texture_handle: &Option<Handle<_>>| -> u32 {
        match texture_handle {
            Some(texture_handle) => match texture_assets.get(texture_handle.id()) {
                Some(texture) if texture.texture.depth_or_array_layers() > 1 => {
                    let (texture_id, _) =
                        texture_arrays.push_if_absent(texture.texture_view.deref(), texture_handle.id());
                    texture_id
                }
                _ => TEXTURE_MAP_NONE,
            },
            None => TEXTURE_MAP_NONE,
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
    // Reset to all-default then overwrite the live slots — freed slots stay
    // default (black) holes.
    let materials_vec = materials.get_mut();
    materials_vec.clear();
    materials_vec.resize(material_count, GpuMaterial::default());
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
        let Some(displacement_texture_id) = process_texture(&material.depth_map) else {
            continue;
        };
        let texture_array_a_id = process_texture_array(&material.texture_array_a);
        let texture_array_b_id = process_texture_array(&material.texture_array_b);
        let texture_array_c_id = process_texture_array(&material.texture_array_c);

        // Texture-size term of the ray-cone LOD, from the base-color texture
        // (representative of the material's maps; texture_array_a stands in for
        // array-painted materials). 0 ⇒ untextured.
        let texel_lod_bias = material
            .base_color_texture
            .as_ref()
            .or(material.texture_array_a.as_ref())
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
            displacement_texture_id,
            displacement_scale: material.depth_scale,
            displacement_bias: material.depth_bias,
            texture_array_a_id,
            texture_array_b_id,
            texture_array_c_id,
            chit_data: UVec4::from_array(material.chit_data),
        };
    }

    if textures.is_empty() {
        textures.vec.push(fallback_texture.d2.texture_view.deref());
        samplers.push(fallback_texture.d2.sampler.deref());
    }
    if texture_arrays.is_empty() {
        texture_arrays.vec.push(fallback_texture.d2_array.texture_view.deref());
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
    *active_light_list.get_mut() = lights.active.clone();
    if active_light_list.get().is_empty() {
        active_light_list.get_mut().extend([0u32, 0u32]);
    }

    materials.write_buffer(&render_device, &render_queue);
    light_sources.write_buffer(&render_device, &render_queue);
    active_light_list.write_buffer(&render_device, &render_queue);

    // Expose the (stable-handle) materials buffer for the RT-pipeline's bindless
    // `physical_load`. `trace_device_address` is only callable on stable-address
    // buffers, so the trace can never capture a reallocating one (compile-time).
    raytracing_scene_bindings.materials_device_address = materials.trace_device_address();

    // The TLAS (PTLAS) is built by `ptlas::dispatch_ptlas`.

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
        // Vertex attributes + materials are reached bindlessly by
        // buffer-device-address (the RT pipeline's `geometry_addresses` uniform),
        // so only the cluster index/table, textures, TLAS, lights, and hair are
        // bound here. Bindings are CONTIGUOUS (0..14) — the raw RT pipeline reads
        // the layout's raw `VkDescriptorSetLayout`, and wgpu compacts sparse
        // (gappy) layouts to contiguous physical slots while the naga SPIR-V keeps
        // the logical numbers; a gap would desync the two. Order must match the
        // `@binding` order in `raytracing_scene_bindings.wgsl`.
        &BindGroupEntries::sequential((
            cluster_mesh_manager.indices.binding(),         // 0 cluster_indices
            cluster_mesh_manager.clusters.binding(),        // 1 clusters
            textures.as_slice(),                            // 2 textures
            samplers.as_slice(),                            // 3 samplers
            tlas.as_binding(),                              // 4 tlas
            light_sources.binding().unwrap(),               // 5 light_sources
            dfg_view,                                       // 6 brdf_dfg_lut
            dfg_sampler,                                    // 7 brdf_dfg_lut_sampler
            active_light_list.binding().unwrap(),           // 8 active_light_list
            // Hair: per-segment records, per-instance records, params, and the
            // transform-table world buffer.
            hair_manager.segments.buffer().as_entire_binding(), // 9 hair_segments
            hair_instance_buffer.as_entire_binding(),       // 10 hair_instances
            hair_params,                                    // 11 hair_params
            transform_propagate.current_world().as_entire_binding(), // 12 hair_world
            texture_arrays.as_slice(),                      // 13 texture_arrays
            &*array_sampler,                                // 14 texture_arrays_sampler
        )),
    ));
}

impl RaytracingSceneBindings {
    pub fn new() -> Self {
        Self {
            bind_group: None,
            materials_device_address: Default::default(),
            bind_group_layout: BindGroupLayoutDescriptor::new(
                "raytracing_scene_bind_group_layout",
                // `transforms` / `previous_frame_transforms` / `material_ids` /
                // `directional_lights` / `instance_cluster_ranges` are GPU columns,
                // bound from the shared `ecs_gpu::SceneColumns` group — not here.
                // Vertex attributes + materials are reached bindlessly by
                // buffer-device-address. Bindings are CONTIGUOUS (0..14) — a
                // sparse/gappy layout would be compacted by wgpu to contiguous
                // physical slots, desyncing it from the naga SPIR-V (which keeps
                // the logical numbers) in the raw RT pipeline. Order matches
                // `raytracing_scene_bindings.wgsl`.
                &BindGroupLayoutEntries::sequential(
                    // COMPUTE for any compute consumer; the RT-pipeline stages so the
                    // same bind group is visible to raygen + the hit/miss shaders.
                    ShaderStages::COMPUTE
                        | ShaderStages::RAY_GENERATION
                        | ShaderStages::CLOSEST_HIT
                        | ShaderStages::ANY_HIT
                        | ShaderStages::MISS,
                    (
                        storage_buffer_read_only_sized(false, None), // 0 cluster_indices
                        storage_buffer_read_only_sized(false, None), // 1 clusters
                        texture_2d(TextureSampleType::Float { filterable: true })
                            .count(MAX_TEXTURE_COUNT), // 2 textures
                        sampler(SamplerBindingType::Filtering).count(MAX_TEXTURE_COUNT), // 3 samplers
                        acceleration_structure(),                    // 4 tlas
                        storage_buffer_read_only_sized(false, None), // 5 light_sources
                        texture_2d(TextureSampleType::Float { filterable: true }), // 6 brdf_dfg_lut
                        sampler(SamplerBindingType::Filtering),      // 7 brdf_dfg_lut_sampler
                        storage_buffer_read_only_sized(false, None), // 8 active_light_list
                        storage_buffer_read_only_sized(false, None), // 9 hair_segments
                        storage_buffer_read_only_sized(false, None), // 10 hair_instances
                        storage_buffer_read_only_sized(false, None), // 11 hair_params
                        storage_buffer_read_only_sized(false, None), // 12 hair_world
                        texture_2d_array(TextureSampleType::Float { filterable: true })
                            .count(MAX_TEXTURE_ARRAY_COUNT), // 13 texture_arrays
                        sampler(SamplerBindingType::Filtering), // 14 texture_arrays_sampler
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

/// Byte stride of one [`GpuMaterial`] record — the RT-pipeline's
/// `physical_load<Material>` indexes the materials buffer by `material_id * this`.
/// This is the `ShaderType` (std430) size encase writes records at — NOT the Rust
/// `size_of`, which differs (the `vec3` fields pad to 16 in std430). Equals the
/// bound `array<Material>` element stride the megakernel reads with.
pub(crate) const GPU_MATERIAL_SIZE: u32 =
    <GpuMaterial as ShaderSize>::SHADER_SIZE.get() as u32;

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
    // query `textureDimensions` per hit. Shared across the material's maps
    // (they're near-always the same resolution).
    texel_lod_bias: f32,
    // Displacement (height) map: `TEXTURE_MAP_NONE` if absent. Consumed when the
    // surface is tessellated — each generated micro-vertex offsets along its
    // normal by `height * displacement_scale + displacement_bias` (world units).
    displacement_texture_id: u32,
    displacement_scale: f32,
    displacement_bias: f32,
    // Layered `texture_arrays` pool slots for custom closest-hits; `TEXTURE_MAP_NONE`
    // if absent. Field order matches the WGSL `Material` struct (std430).
    texture_array_a_id: u32,
    texture_array_b_id: u32,
    texture_array_c_id: u32,
    // Opaque per-material data for a custom closest-hit (StandardSolariMaterial::chit_data).
    chit_data: UVec4,
}

impl Default for GpuMaterial {
    fn default() -> Self {
        Self {
            // No map, not pool slot 0 — a slot whose textures haven't loaded
            // yet (or was freed) must not sample whatever lives at index 0.
            normal_map_texture_id: TEXTURE_MAP_NONE,
            base_color_texture_id: TEXTURE_MAP_NONE,
            emissive_texture_id: TEXTURE_MAP_NONE,
            metallic_roughness_texture_id: TEXTURE_MAP_NONE,
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
            displacement_texture_id: TEXTURE_MAP_NONE,
            displacement_scale: 0.0,
            displacement_bias: 0.0,
            texture_array_a_id: TEXTURE_MAP_NONE,
            texture_array_b_id: TEXTURE_MAP_NONE,
            texture_array_c_id: TEXTURE_MAP_NONE,
            chit_data: UVec4::ZERO,
        }
    }
}

