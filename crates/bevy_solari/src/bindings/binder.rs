use super::extract::SolariMaterialAssets;
use crate::material::MaterialSlots;
use crate::accel::ptlas::Ptlas;
use crate::gpu::binding_seam::{BindingSeam, HeapKind, HeapResource};
use crate::instance::InstanceManager;
use crate::geometry::ClusterMeshManager;
use ash::vk;
use bevy_asset::{AssetId, Handle};
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::{
    change_detection::DetectChanges,
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
use bevy_image::Image;
use core::{num::NonZeroU32, ops::Deref};
use wgpu::hal::api::Vulkan as VkApi;

pub(crate) const MAX_TEXTURE_COUNT: NonZeroU32 = NonZeroU32::new(5_000).unwrap();

/// Size of the layered `texture_arrays` pool — must match the sized
/// `binding_array<texture_2d_array<f32>, 16>` in `raytracing_scene_bindings.wgsl`.
pub(crate) const MAX_TEXTURE_ARRAY_COUNT: NonZeroU32 = NonZeroU32::new(16).unwrap();

/// Size of the deduplicated sampler-config array (`samplers[]`, binding 3) —
/// must match `SAMPLER_CONFIG_COUNT` in `scene_resolve.slang`. Bounded by
/// distinct sampler *configurations*, not texture count: the hardware sampler
/// heap holds ~4096 descriptors total, far under [`MAX_TEXTURE_COUNT`].
pub(crate) const SAMPLER_CONFIG_COUNT: NonZeroU32 = NonZeroU32::new(256).unwrap();

const TEXTURE_MAP_NONE: u32 = u32::MAX;

/// Material texture ids pack `sampler_config << 16 | texture_slot`
/// (`TEXTURE_MAP_NONE` is never a valid packed id: config indices stay far
/// below 0xFFFF). The shader unpacks in `scene_resolve.slang`'s samplers.
const SAMPLER_CFG_SHIFT: u32 = 16;

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
    /// The scene set as descriptor-heap slots. The TLAS (binding 4) is
    /// deliberately absent: its address rides push data — shader-side heap AS
    /// access device-losts even on R610.
    pub scene_heap: Option<SceneHeapSlots>,
    /// Stable slots for `textures[]` (binding 2) — see [`TextureSlotTable`].
    texture_slots: TextureSlotTable,
    /// Stable slots for `texture_arrays[]` (binding 13).
    texture_array_slots: TextureSlotTable,
    /// Deduplicated sampler configs for `samplers[]` (binding 3).
    sampler_configs: SamplerConfigs,
}

/// Stable-slot table for one bindless texture array: an asset takes a slot at
/// first sight and keeps it while the asset lives, so the ids baked into
/// material records — and the heap descriptor written at `block + slot` —
/// survive across frames. Steady-state frames register nothing and write zero
/// descriptors; eviction and view replacement are handled only on frames
/// where `RenderAssets<GpuImage>` changed. Values are
/// `(slot, sampler_config)`; freed slots recycle, and the wgpu binding array
/// pads holes with the fallback view.
#[derive(Default)]
struct TextureSlotTable {
    slots: HashMap<AssetId<Image>, (u32, u32)>,
    free: Vec<u32>,
    high_water: u32,
}

impl TextureSlotTable {
    fn alloc(&mut self, capacity: u32) -> u32 {
        let slot = self.free.pop().unwrap_or_else(|| {
            let slot = self.high_water;
            self.high_water += 1;
            slot
        });
        assert!(
            slot < capacity,
            "texture slot table overflow: more than {capacity} live textures"
        );
        slot
    }
}

/// Sampler descriptors deduplicated by create-info. Material texture ids
/// carry the config index in their high bits ([`SAMPLER_CFG_SHIFT`]), so
/// sampler-heap and array use is bounded by [`SAMPLER_CONFIG_COUNT`] distinct
/// configurations instead of texture count. Configs are never evicted (they
/// hold no texture memory — the clones keep the few `VkSampler`s alive).
#[derive(Default)]
struct SamplerConfigs {
    by_key: HashMap<[u32; 16], u32>,
    samplers: Vec<Sampler>,
}

/// Register `sampler`'s configuration, returning its config index; a new
/// config writes its heap descriptor once at `sampler_block + index`.
#[allow(unsafe_code)]
fn sampler_config(
    configs: &mut SamplerConfigs,
    sampler: &Sampler,
    heap: Option<(&BindingSeam, &SceneHeapSlots)>,
) -> u32 {
    // SAFETY: the sampler is a live wgpu resource on the Vulkan backend; the
    // guard is dropped before anything can destroy it.
    let Some(info) = unsafe { sampler.as_hal::<VkApi>() }.map(|s| s.create_info()) else {
        return 0;
    };
    let key = sampler_key(&info);
    if let Some(&index) = configs.by_key.get(&key) {
        return index;
    }
    let index = configs.samplers.len() as u32;
    assert!(
        index < SAMPLER_CONFIG_COUNT.get(),
        "more than {SAMPLER_CONFIG_COUNT} distinct sampler configurations"
    );
    if let Some((seam, slots)) = heap {
        seam.rewrite_heap_index(
            HeapKind::Sampler,
            slots.sampler_block + index,
            HeapResource::Sampler(&info),
        );
    }
    configs.by_key.insert(key, index);
    configs.samplers.push(sampler.clone());
    index
}

/// Dedup key: every `VkSamplerCreateInfo` field that shapes the descriptor.
fn sampler_key(info: &vk::SamplerCreateInfo) -> [u32; 16] {
    [
        info.flags.as_raw(),
        info.mag_filter.as_raw() as u32,
        info.min_filter.as_raw() as u32,
        info.mipmap_mode.as_raw() as u32,
        info.address_mode_u.as_raw() as u32,
        info.address_mode_v.as_raw() as u32,
        info.address_mode_w.as_raw() as u32,
        info.mip_lod_bias.to_bits(),
        info.anisotropy_enable,
        info.max_anisotropy.to_bits(),
        info.compare_enable,
        info.compare_op.as_raw() as u32,
        info.min_lod.to_bits(),
        info.max_lod.to_bits(),
        info.border_color.as_raw() as u32,
        info.unnormalized_coordinates,
    ]
}

/// Write one sampled-image heap descriptor from the view's recorded create
/// info (read-optimal by trace time — the wgpu bind group's usage keeps the
/// layout transitions happening).
#[allow(unsafe_code)]
pub(crate) fn write_image_descriptor(seam: &BindingSeam, slot: u32, view: &wgpu::TextureView) {
    // SAFETY: the view is a live wgpu resource on the Vulkan backend; the
    // guard is dropped before anything can destroy it.
    if let Some(hal_view) = unsafe { view.as_hal::<VkApi>() } {
        let info = hal_view.image_view_create_info();
        seam.rewrite_heap_index(
            HeapKind::Image,
            slot,
            HeapResource::SampledImage {
                view: &info,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        );
    }
}

/// Write one sampler heap descriptor from the sampler's recorded create info.
#[allow(unsafe_code)]
pub(crate) fn write_sampler_descriptor(seam: &BindingSeam, slot: u32, sampler: &wgpu::Sampler) {
    // SAFETY: as for the view above.
    if let Some(hal_sampler) = unsafe { sampler.as_hal::<VkApi>() } {
        let info = hal_sampler.create_info();
        seam.rewrite_heap_index(HeapKind::Sampler, slot, HeapResource::Sampler(&info));
    }
}

/// The scene set's resources as descriptor-heap slots. The bindless arrays
/// get one contiguous block each ([`BindingSeam::alloc_heap_block`]), dense
/// mirrors of the bind-group arrays: element `i`'s descriptor lives at
/// `block + i`, so the material table's texture ids index the heap unchanged.
///
/// [`BindingSeam::alloc_heap_block`]: crate::gpu::binding_seam::BindingSeam::alloc_heap_block
pub struct SceneHeapSlots {
    /// `(set-0 binding index, heap buffer-region slot)` pairs, in
    /// [`SCENE_BUFFER_BINDINGS`] order.
    pub buffers: Vec<(u32, u32)>,
    /// Last `(device address, size)` written per [`buffers`](Self::buffers)
    /// entry — a frame rewrites a buffer descriptor only when its backing
    /// grew or reallocated.
    pub(crate) buffer_written: Vec<(u64, u64)>,
    /// Image block backing `textures[]` (binding 2), `MAX_TEXTURE_COUNT` slots.
    pub texture_block: u32,
    /// Sampler block backing `samplers[]` (binding 3),
    /// [`SAMPLER_CONFIG_COUNT`] deduplicated config slots.
    pub sampler_block: u32,
    /// Image block backing `texture_arrays[]` (binding 13), `MAX_TEXTURE_ARRAY_COUNT` slots.
    pub texture_array_block: u32,
    /// DFG LUT image (binding 6).
    pub dfg_lut: u32,
    /// DFG LUT sampler (binding 7).
    pub dfg_sampler: u32,
    /// Shared `texture_arrays` sampler (binding 14).
    pub array_sampler: u32,
    /// Whether the single (non-array) descriptors above have been written;
    /// they rewrite only on `RenderAssets` change frames after that.
    pub(crate) singles_written: bool,
}

/// The set-0 storage-buffer bindings mirrored into the heap, in binding
/// order. The TLAS (4) rides push data; 6/7 and 13/14 are images/samplers;
/// 2/3 are the bindless arrays.
const SCENE_BUFFER_BINDINGS: [u32; 8] = [0, 1, 5, 8, 9, 10, 11, 12];

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
    seam: Option<Res<BindingSeam>>,
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
    let RaytracingSceneBindings {
        bind_group,
        bind_group_layout,
        materials_device_address,
        scene_heap,
        texture_slots,
        texture_array_slots,
        sampler_configs,
    } = &mut *raytracing_scene_bindings;
    let seam = seam.as_deref();

    // Heap blocks are allocated once — their bases are baked into the RT
    // pipeline's mapping tables, so they must never move. Descriptor writes
    // into them are diff-driven from here on: texture registration/eviction,
    // buffer reallocation, and `RenderAssets` change frames; a steady-state
    // frame writes none.
    if let Some(seam) = seam {
        scene_heap.get_or_insert_with(|| SceneHeapSlots {
            buffers: SCENE_BUFFER_BINDINGS
                .iter()
                .map(|&binding| (binding, seam.alloc_heap_block(HeapKind::Buffer, 1)))
                .collect(),
            buffer_written: vec![(0, 0); SCENE_BUFFER_BINDINGS.len()],
            texture_block: seam.alloc_heap_block(HeapKind::Image, MAX_TEXTURE_COUNT.get()),
            sampler_block: seam.alloc_heap_block(HeapKind::Sampler, SAMPLER_CONFIG_COUNT.get()),
            texture_array_block: seam
                .alloc_heap_block(HeapKind::Image, MAX_TEXTURE_ARRAY_COUNT.get()),
            dfg_lut: seam.alloc_heap_block(HeapKind::Image, 1),
            dfg_sampler: seam.alloc_heap_block(HeapKind::Sampler, 1),
            array_sampler: seam.alloc_heap_block(HeapKind::Sampler, 1),
            singles_written: false,
        });
    }
    let heap: Option<(&BindingSeam, &SceneHeapSlots)> = seam.zip(scene_heap.as_ref());

    // `RenderAssets` changed ⇒ textures may have loaded, been replaced, or
    // unloaded: evict dead entries (recycling their slots) and rewrite every
    // live descriptor — replacement swaps the underlying view/sampler, and
    // rewriting an unchanged descriptor is a benign host write. Quiet frames
    // (the steady state) skip this entirely.
    let assets_changed = texture_assets.is_changed();
    if assets_changed {
        let TextureSlotTable { slots, free, .. } = &mut *texture_slots;
        slots.retain(|id, (slot, cfg)| match texture_assets.get(*id) {
            Some(texture) => {
                if let Some((seam, heap_slots)) = heap {
                    write_image_descriptor(
                        seam,
                        heap_slots.texture_block + *slot,
                        &texture.texture_view,
                    );
                }
                *cfg = sampler_config(sampler_configs, &texture.sampler, heap);
                true
            }
            None => {
                free.push(*slot);
                false
            }
        });
        let TextureSlotTable { slots, free, .. } = &mut *texture_array_slots;
        slots.retain(|id, (slot, _)| match texture_assets.get(*id) {
            // A replaced asset may have collapsed to a plain D2 image — evict
            // it like an unload (D2 views can't enter the D2Array pool).
            Some(texture) if texture.texture.depth_or_array_layers() > 1 => {
                if let Some((seam, heap_slots)) = heap {
                    write_image_descriptor(
                        seam,
                        heap_slots.texture_array_block + *slot,
                        &texture.texture_view,
                    );
                }
                true
            }
            _ => {
                free.push(*slot);
                false
            }
        });
    }

    let mut process_texture = |texture_handle: &Option<Handle<Image>>| -> Option<u32> {
        let Some(texture_handle) = texture_handle else {
            return Some(TEXTURE_MAP_NONE);
        };
        let id = texture_handle.id();
        let texture = texture_assets.get(id)?;
        if let Some(&(slot, cfg)) = texture_slots.slots.get(&id) {
            return Some(cfg << SAMPLER_CFG_SHIFT | slot);
        }
        // First sight: take a stable slot, write its heap descriptor, dedup
        // the sampler into a config slot. Registration is the only frame this
        // texture costs descriptor writes.
        let slot = texture_slots.alloc(MAX_TEXTURE_COUNT.get());
        let cfg = sampler_config(sampler_configs, &texture.sampler, heap);
        if let Some((seam, heap_slots)) = heap {
            write_image_descriptor(seam, heap_slots.texture_block + slot, &texture.texture_view);
        }
        texture_slots.slots.insert(id, (slot, cfg));
        Some(cfg << SAMPLER_CFG_SHIFT | slot)
    };

    // Layered pool (`texture_arrays`, binding 13). Only D2Array views may enter —
    // a plain D2 image would fail bind-group validation. Unlike flat textures a
    // still-loading array degrades to NONE (not a material skip): array-painted
    // chits carry a procedural fallback, so they shade flat until the pop-in.
    let mut process_texture_array = |texture_handle: &Option<Handle<Image>>| -> u32 {
        let Some(texture_handle) = texture_handle else {
            return TEXTURE_MAP_NONE;
        };
        let id = texture_handle.id();
        let Some(texture) = texture_assets.get(id) else {
            return TEXTURE_MAP_NONE;
        };
        if texture.texture.depth_or_array_layers() <= 1 {
            return TEXTURE_MAP_NONE;
        }
        if let Some(&(slot, _)) = texture_array_slots.slots.get(&id) {
            return slot;
        }
        let slot = texture_array_slots.alloc(MAX_TEXTURE_ARRAY_COUNT.get());
        if let Some((seam, heap_slots)) = heap {
            write_image_descriptor(
                seam,
                heap_slots.texture_array_block + slot,
                &texture.texture_view,
            );
        }
        texture_array_slots.slots.insert(id, (slot, 0));
        slot
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

    // The wgpu binding arrays mirror the slot tables: holes (recycled slots)
    // and the zero-entry case pad with the fallback so every array element
    // stays valid. Lengths track the tables' high-water, not churn per frame.
    let mut textures_wgpu: Vec<&wgpu::TextureView> = vec![
        fallback_texture.d2.texture_view.deref();
        texture_slots.high_water.max(1) as usize
    ];
    for (id, &(slot, _)) in &texture_slots.slots {
        if let Some(texture) = texture_assets.get(*id) {
            textures_wgpu[slot as usize] = &texture.texture_view;
        }
    }
    let mut samplers_wgpu: Vec<&wgpu::Sampler> = sampler_configs
        .samplers
        .iter()
        .map(|sampler| sampler.deref())
        .collect();
    if samplers_wgpu.is_empty() {
        samplers_wgpu.push(fallback_texture.d2.sampler.deref());
    }
    let mut texture_arrays_wgpu: Vec<&wgpu::TextureView> = vec![
        fallback_texture.d2_array.texture_view.deref();
        texture_array_slots.high_water.max(1) as usize
    ];
    for (id, &(slot, _)) in &texture_array_slots.slots {
        if let Some(texture) = texture_assets.get(*id) {
            texture_arrays_wgpu[slot as usize] = &texture.texture_view;
        }
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
    *materials_device_address = materials.trace_device_address();

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
    *bind_group = Some(render_device.create_bind_group(
        "raytracing_scene_bind_group",
        &pipeline_cache.get_bind_group_layout(bind_group_layout),
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
            textures_wgpu.as_slice(),                       // 2 textures
            samplers_wgpu.as_slice(),                       // 3 samplers
            tlas.as_binding(),                              // 4 tlas
            light_sources.binding().unwrap(),               // 5 light_sources
            dfg_view,                                       // 6 brdf_dfg_lut
            dfg_sampler,                                    // 7 brdf_dfg_lut_sampler
            active_light_list.binding().unwrap(),           // 8 active_light_list
            // Hair: per-segment records, per-instance records, params, and the
            // transform-table world buffer.
            hair_manager.segments.buffer().as_entire_binding(), // 9 hair_segments
            hair_instance_buffer.as_entire_binding(),       // 10 hair_instances
            hair_params.clone(),                            // 11 hair_params
            transform_propagate.current_world().as_entire_binding(), // 12 hair_world
            texture_arrays_wgpu.as_slice(),                 // 13 texture_arrays
            &*array_sampler,                                // 14 texture_arrays_sampler
        )),
    ));

    // Buffer descriptors: rewritten only when the backing buffer grew or
    // reallocated (address/size change). The texture/sampler blocks were
    // handled above at registration/eviction/change time — a steady-state
    // frame reaches here having written nothing.
    if let (Some(seam), Some(slots)) = (seam, scene_heap.as_mut()) {
        let buffer_targets: [(u32, BindingResource); 8] = [
            (0, cluster_mesh_manager.indices.binding()),
            (1, cluster_mesh_manager.clusters.binding()),
            (5, light_sources.binding().unwrap()),
            (8, active_light_list.binding().unwrap()),
            (9, hair_manager.segments.buffer().as_entire_binding()),
            (10, hair_instance_buffer.as_entire_binding()),
            (11, hair_params),
            (12, transform_propagate.current_world().as_entire_binding()),
        ];
        for (i, (binding, res)) in buffer_targets.into_iter().enumerate() {
            let BindingResource::Buffer(b) = res else {
                continue;
            };
            let address = seam.device_address(b.buffer).get() + b.offset;
            let size = b.size.map(u64::from).unwrap_or(b.buffer.size() - b.offset);
            let (slot_binding, slot) = slots.buffers[i];
            debug_assert_eq!(binding, slot_binding);
            if slots.buffer_written[i] == (address, size) {
                continue;
            }
            slots.buffer_written[i] = (address, size);
            seam.rewrite_heap_index(HeapKind::Buffer, slot, HeapResource::Buffer { address, size });
        }
        // The single image/sampler descriptors: once at startup, then only on
        // `RenderAssets` change frames (the DFG LUT swaps from the fallback to
        // the baked texture when it loads).
        if !slots.singles_written || assets_changed {
            slots.singles_written = true;
            write_image_descriptor(seam, slots.dfg_lut, dfg_view);
            write_sampler_descriptor(seam, slots.dfg_sampler, dfg_sampler);
            write_sampler_descriptor(seam, slots.array_sampler, array_sampler);
        }
    }
}

impl RaytracingSceneBindings {
    pub fn new() -> Self {
        Self {
            bind_group: None,
            materials_device_address: Default::default(),
            scene_heap: None,
            texture_slots: Default::default(),
            texture_array_slots: Default::default(),
            sampler_configs: Default::default(),
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
                        sampler(SamplerBindingType::Filtering).count(SAMPLER_CONFIG_COUNT), // 3 samplers
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

