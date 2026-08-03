//! One layout-free compute kernel on the descriptor heap — the shared shape
//! of every Slang compute pass: Slang source → SPIR-V (`gpu/slang.rs`) →
//! heap-flagged pipeline (`gpu/binding_seam.rs`) plus the reflected
//! parameter table dispatches assemble their push-data slot arrays against.
//! Nothing about a kernel's bindings is hand-maintained: the mapping table
//! is derived from the compiled SPIR-V, and slot arrays are assembled by
//! parameter NAME against the kernel's reflected layout
//! ([`push_slots`](HeapKernel::push_slots)).
#![allow(unsafe_code)]

use ash::vk;

use super::binding_seam::BindingSeam;

/// One heap-flagged kernel plus its reflected set-0 parameter table.
pub struct HeapKernel {
    pub module: vk::ShaderModule,
    pub pipeline: vk::Pipeline,
    /// `(parameter name, binding)` from slang reflection of this kernel.
    pub bindings: Vec<(String, u32)>,
    /// Size of the kernel's `[[vk::push_constant]]` params block (0 if
    /// none); the slot array sits behind it in the push blob.
    push_params_size: u32,
}

impl HeapKernel {
    /// Compile `entry` from `source` (with `modules` importable and `defines`
    /// set) and create its heap pipeline. `push_params_size` is the size of
    /// the kernel's `[[vk::push_constant]]` params block (0 if none). `None`
    /// on compile or pipeline failure (logged under `label`).
    pub fn new(
        seam: &BindingSeam,
        file: &str,
        source: &str,
        entry: &str,
        modules: &[(&str, &str)],
        defines: &[(&str, &str)],
        label: &str,
        push_params_size: u32,
    ) -> Option<Self> {
        let shader = crate::gpu::slang::compile_rt_slang(file, source, entry, modules, defines, &[])
            .map_err(|e| bevy_log::error!("{label}: {e}"))
            .ok()?;
        let entry_c = std::ffi::CString::new(entry).expect("entry name has interior NUL");
        // Reflection lists every declared global — including parameters this
        // entry never touches (a module can serve several entries). The
        // dispatch contract is the bindings that SURVIVE in the compiled
        // SPIR-V (what the mapping table is built from), so intersect.
        let live = crate::gpu::binding_seam::spirv_descriptor_bindings(&shader.spirv);
        let (module, pipeline) =
            seam.create_heap_compute_pipeline(&shader.spirv, &entry_c, label, push_params_size)?;
        Some(Self {
            module,
            pipeline,
            bindings: shader
                .bindings
                .into_iter()
                .filter(|&(_, set, binding)| {
                    set == 0 && live.iter().any(|&(s, b, _)| (s, b) == (set, binding))
                })
                .map(|(name, _, binding)| (name, binding))
                .collect(),
            push_params_size,
        })
    }

    /// Multi-set variant of [`new`](Self::new), for kernels that read a
    /// shared heap surface (scene/columns/cluster-scene) and/or trace against
    /// the TLAS: `base_mappings` (caller-built — e.g.
    /// [`scene_heap_mappings`](crate::gpu::rt_pipeline::scene_heap_mappings),
    /// [`cluster_heap_mappings`](crate::gpu::rt_pipeline::cluster_heap_mappings),
    /// or a lone TLAS row) must cover every set-0/set-2 binding surviving in
    /// the SPIR-V. The kernel's OWN set-1 bindings are push-index-mapped
    /// behind the leading `push_params_size` bytes
    /// (`push_params_size + binding*4`), so the push blob is
    /// `[params @0 | set-1 slot array @push_params_size]`.
    ///
    /// `push_params_size` is whatever the kernel reads from the front of push
    /// data: a classic `[[vk::push_constant]]` params block for TLAS-less
    /// kernels (the cluster AS passes), or the 8-byte TLAS device address for
    /// tracing kernels (the TLAS rides `map_binding_push_address` at push
    /// offset 0, so a push-constant block would collide — their per-dispatch
    /// params ride a uniform-buffer slot instead, [`KernelSlots::uniform`]).
    /// Either way [`push_blob`](Self::push_blob) takes those leading bytes as
    /// its `params`. `capabilities` are target capability atoms
    /// (`spvRayQueryKHR` for inline ray queries).
    pub fn new_with_mappings(
        seam: &BindingSeam,
        file: &str,
        source: &str,
        entry: &str,
        modules: &[(&str, &str)],
        defines: &[(&str, &str)],
        capabilities: &[&str],
        label: &str,
        push_params_size: u32,
        base_mappings: &[vk::DescriptorSetAndBindingMappingEXT<'static>],
    ) -> Option<Self> {
        assert_eq!(push_params_size % 4, 0, "{label}: params size unaligned");
        let shader =
            crate::gpu::slang::compile_rt_slang(file, source, entry, modules, defines, capabilities)
                .map_err(|e| bevy_log::error!("{label}: {e}"))
                .ok()?;
        let entry_c = std::ffi::CString::new(entry).expect("entry name has interior NUL");
        let live = crate::gpu::binding_seam::spirv_descriptor_bindings(&shader.spirv);
        let mut mappings = base_mappings.to_vec();
        for &(set, binding, kind) in &live {
            match set {
                // The pass's own bindings: heap indices from the push blob's
                // slot array.
                1 => {
                    assert_ne!(
                        kind,
                        crate::gpu::binding_seam::SpirvBindingKind::AccelerationStructure,
                        "{label}: (set 1, binding {binding}): the TLAS is push-address-mapped \
                         at set 0 via the caller's base mappings"
                    );
                    mappings.push(seam.map_binding_push_index(
                        1,
                        binding,
                        kind.heap_kind(),
                        push_params_size + binding * 4,
                    ));
                }
                // Scene set 0 / columns set 2: must already be covered.
                _ => {
                    let covered = base_mappings.iter().any(|m| {
                        m.descriptor_set == set
                            && (m.first_binding..m.first_binding + m.binding_count)
                                .contains(&binding)
                    });
                    assert!(
                        covered,
                        "{label}: (set {set}, binding {binding}) not covered by the \
                         caller's base mappings"
                    );
                }
            }
        }
        let (module, pipeline) =
            seam.create_heap_compute_pipeline_with_mappings(&shader.spirv, &entry_c, label, &mappings)?;
        Some(Self {
            module,
            pipeline,
            bindings: shader
                .bindings
                .into_iter()
                .filter(|&(_, set, binding)| {
                    set == 1 && live.iter().any(|&(s, b, _)| (s, b) == (set, binding))
                })
                .map(|(name, _, binding)| (name, binding))
                .collect(),
            push_params_size,
        })
    }

    /// Assemble the full push blob — `params` bytes then the slot array from
    /// `(parameter name, heap slot)` pairs
    /// ([`push_slots`](Self::push_slots)). `params` must match the kernel's
    /// declared `[[vk::push_constant]]` block size.
    pub fn push_blob(&self, label: &str, params: &[u8], named: &[(&str, u32)]) -> Vec<u8> {
        assert_eq!(
            params.len() as u32,
            self.push_params_size,
            "{label}: params size doesn't match the kernel's push block"
        );
        let mut blob = params.to_vec();
        blob.extend(bytemuck::cast_slice::<u32, u8>(&self.push_slots(label, named)));
        blob
    }

    /// Assemble the push-data slot array from `(parameter name, heap slot)`
    /// pairs: `slots[binding] = slot`, with the binding read from the
    /// shader's own reflected layout. Any mismatch — a missing, misnamed,
    /// duplicated, or extra parameter — panics naming the kernel and the
    /// parameter, so a shader binding edit can't silently desync a dispatch.
    pub fn push_slots(&self, label: &str, named: &[(&str, u32)]) -> Vec<u32> {
        let len = self
            .bindings
            .iter()
            .map(|&(_, binding)| binding + 1)
            .max()
            .unwrap_or(0);
        let mut slots = vec![u32::MAX; len as usize];
        for &(name, slot) in named {
            let Some(&(_, binding)) = self.bindings.iter().find(|(n, _)| n == name) else {
                panic!(
                    "{label} has no parameter `{name}` (shader declares {:?})",
                    self.bindings
                );
            };
            assert!(
                slots[binding as usize] == u32::MAX,
                "{label}: parameter `{name}` supplied twice"
            );
            slots[binding as usize] = slot;
        }
        for (name, binding) in &self.bindings {
            assert!(
                slots[*binding as usize] != u32::MAX,
                "{label}: parameter `{name}` not supplied"
            );
        }
        slots
    }

    /// Destroy the pipeline + module.
    ///
    /// # Safety
    /// The caller must have drained in-flight GPU work referencing this
    /// kernel (`quiesce_before_raw_destroy`), and `device` must be the
    /// device it was created on.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: per the function contract.
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_shader_module(self.module, None);
        }
    }
}

/// Every heap-kernel Slang source in the crate, for the compile test AND a
/// single place to see the ported surface: `(label, file, source, entry,
/// modules)`. Grows with each WGSL→Slang port.
#[cfg(test)]
pub(crate) const HEAP_KERNELS: &[(&str, &str, &str, &[(&str, &str)])] = &[
    (
        "light_resolve.slang",
        include_str!("../lights/light_resolve.slang"),
        "resolve",
        &[],
    ),
    (
        "transform_gather.slang",
        include_str!("../transform/transform_gather.slang"),
        "gather",
        &[],
    ),
    (
        "transform_subtract.slang",
        include_str!("../transform/transform_subtract.slang"),
        "subtract",
        &[],
    ),
    (
        "transform_frontier.slang",
        include_str!("../transform/transform_frontier.slang"),
        "seed",
        &[],
    ),
    (
        "transform_frontier.slang",
        include_str!("../transform/transform_frontier.slang"),
        "finalize",
        &[],
    ),
    (
        "transform_frontier.slang",
        include_str!("../transform/transform_frontier.slang"),
        "expand",
        &[],
    ),
    (
        "transform_propagate.slang",
        include_str!("../transform/transform_propagate.slang"),
        "propagate",
        &[],
    ),
    (
        "transform_readback.slang",
        include_str!("../transform/transform_readback.slang"),
        "readback",
        &[],
    ),
    (
        "deform.slang",
        include_str!("../accel/deform.slang"),
        "deform",
        crate::bindings::OCTAHEDRAL_MODULES,
    ),
    (
        "reconcile.slang",
        include_str!("../ecs_gpu/reconcile.slang"),
        "reconcile_apply",
        &[],
    ),
    (
        "ptlas_hair_write.slang",
        include_str!("../hair/ptlas_hair_write.slang"),
        "hair_write",
        &[],
    ),
    (
        "instantiate.slang",
        include_str!("../accel/instantiate.slang"),
        "instantiate",
        &[],
    ),
    (
        "tess_ptlas_write.slang",
        include_str!("../geometry/tess_ptlas_write.slang"),
        "tess_write",
        &[],
    ),
    (
        "tess_instantiate.slang",
        include_str!("../geometry/tess_instantiate.slang"),
        "build_infos",
        &[],
    ),
    (
        "tess_classify.slang",
        include_str!("../geometry/tess_classify.slang"),
        "classify",
        &[],
    ),
    (
        "tess_classify.slang",
        include_str!("../geometry/tess_classify.slang"),
        "finalize",
        &[],
    ),
    (
        "tess_gen_attrs.slang",
        include_str!("../geometry/tess_gen_attrs.slang"),
        "gen_attrs_main",
        crate::bindings::OCTAHEDRAL_MODULES,
    ),
    (
        "atmosphere_lut_bake.slang",
        include_str!("../render/atmosphere_lut_bake.slang"),
        "bake",
        &[],
    ),
    (
        "gpu_instances_scatter.slang",
        include_str!("../ecs_gpu/gpu_instances_scatter.slang"),
        "scatter",
        &[],
    ),
    (
        "gpu_instances_scatter.slang",
        include_str!("../ecs_gpu/gpu_instances_scatter.slang"),
        "scatter_with_history",
        &[],
    ),
    (
        "rt_camera.slang",
        include_str!("../render/rt_pipeline/rt_camera.slang"),
        "rt_camera",
        &[(
            "rt_payload",
            include_str!("../render/rt_pipeline/rt_payload.slang"),
        )],
    ),
    (
        "blit.slang",
        include_str!("../render/rt_pipeline/blit.slang"),
        "blit",
        &[],
    ),
    (
        "dlss_resolve.slang",
        include_str!("../render/dlss_resolve.slang"),
        "resolve",
        &[],
    ),
    (
        "atmosphere_bake.slang",
        include_str!("../render/atmosphere_bake.slang"),
        "bake",
        &[("atmosphere", include_str!("../render/atmosphere.slang"))],
    ),
];

/// The MULTI-SET heap kernels — built via [`HeapKernel::new_with_mappings`]
/// because they read the shared scene heap surface and/or trace the TLAS:
/// `(label, file, source, entry, modules, capabilities)`. Kept out of
/// [`HEAP_KERNELS`], whose set-0-only / no-AS assertions they legitimately
/// violate.
#[cfg(test)]
pub(crate) const MULTI_SET_HEAP_KERNELS: &[(
    &str,
    &str,
    &str,
    &[(&str, &str)],
    &[&str],
)] = &[
    (
        "restir_spatial.slang",
        include_str!("../render/rt_pipeline/restir_spatial.slang"),
        "spatial",
        crate::render::rt_pipeline::RESTIR_SPATIAL_MODULES,
        crate::gpu::slang::RAY_QUERY_CAPABILITIES,
    ),
    (
        "ray_query.slang",
        include_str!("../ray_query/ray_query.slang"),
        "query_rays",
        &[],
        crate::gpu::slang::RAY_QUERY_CAPABILITIES,
    ),
    (
        "selector.slang",
        include_str!("../accel/selector.slang"),
        "select_reset",
        crate::accel::selector::SELECTOR_MODULES,
        &[],
    ),
    (
        "selector.slang",
        include_str!("../accel/selector.slang"),
        "select_main",
        crate::accel::selector::SELECTOR_MODULES,
        &[],
    ),
    (
        "blas_sharing.slang",
        include_str!("../accel/blas_sharing.slang"),
        "geom_reset",
        crate::accel::blas_sharing::BLAS_SHARING_MODULES,
        &[],
    ),
    (
        "blas_sharing.slang",
        include_str!("../accel/blas_sharing.slang"),
        "classify",
        crate::accel::blas_sharing::BLAS_SHARING_MODULES,
        &[],
    ),
    (
        "blas_sharing.slang",
        include_str!("../accel/blas_sharing.slang"),
        "elect_dirty",
        crate::accel::blas_sharing::BLAS_SHARING_MODULES,
        &[],
    ),
    (
        "blas_sharing.slang",
        include_str!("../accel/blas_sharing.slang"),
        "finalize_count",
        crate::accel::blas_sharing::BLAS_SHARING_MODULES,
        &[],
    ),
    (
        "blas_sharing.slang",
        include_str!("../accel/blas_sharing.slang"),
        "assign_address",
        crate::accel::blas_sharing::BLAS_SHARING_MODULES,
        &[],
    ),
    (
        "blas_sharing.slang",
        include_str!("../accel/blas_sharing.slang"),
        "commit_built",
        crate::accel::blas_sharing::BLAS_SHARING_MODULES,
        &[],
    ),
    (
        "ptlas_fill.slang",
        include_str!("../accel/ptlas_fill.slang"),
        "fill_seed",
        crate::accel::ptlas::PTLAS_FILL_MODULES,
        &[],
    ),
    (
        "ptlas_fill.slang",
        include_str!("../accel/ptlas_fill.slang"),
        "fill_incremental",
        crate::accel::ptlas::PTLAS_FILL_MODULES,
        &[],
    ),
    (
        "ptlas_fill.slang",
        include_str!("../accel/ptlas_fill.slang"),
        "finalize",
        crate::accel::ptlas::PTLAS_FILL_MODULES,
        &[],
    ),
    (
        "ptlas_fill.slang",
        include_str!("../accel/ptlas_fill.slang"),
        "validate",
        crate::accel::ptlas::PTLAS_FILL_MODULES,
        &[],
    ),
    (
        "tess_gen_verts.slang",
        include_str!("../geometry/tess_gen_verts.slang"),
        "gen_verts",
        crate::bindings::OCTAHEDRAL_MODULES,
        &[],
    ),
];

/// Persistent heap slots for a kernel's parameters, rewritten every
/// dispatch — a descriptor write is a host memcpy into the mapped heap, far
/// cheaper than tracking resource identity across reallocations. Slots are
/// app-lifetime (allocated once, never freed), matching the pass resources
/// that own them. Slot indices are region-local to their [`HeapKind`] — the
/// kind-aware mapping table routes image bindings to the image region.
pub struct KernelSlots {
    slots: Vec<(super::binding_seam::HeapKind, u32)>,
}

impl KernelSlots {
    /// Reserve `count` buffer-region slots (no descriptors written yet).
    pub fn new(seam: &BindingSeam, count: usize) -> Self {
        use super::binding_seam::HeapKind;
        Self::new_mixed(seam, &vec![HeapKind::Buffer; count])
    }

    /// Reserve one slot per entry of `kinds`, each in its kind's heap region
    /// (no descriptors written yet).
    pub fn new_mixed(seam: &BindingSeam, kinds: &[super::binding_seam::HeapKind]) -> Self {
        Self {
            slots: kinds
                .iter()
                .map(|&kind| (kind, seam.alloc_heap_block(kind, 1)))
                .collect(),
        }
    }

    fn slot(&self, i: usize, kind: super::binding_seam::HeapKind) -> u32 {
        let (allocated, slot) = self.slots[i];
        assert_eq!(allocated, kind, "KernelSlots: slot {i} was allocated as {allocated:?}");
        slot
    }

    /// Write a STORAGE_BUFFER descriptor over `buffer` into slot `i` and
    /// return the slot for the push array.
    pub fn buffer(&self, seam: &BindingSeam, i: usize, buffer: &wgpu::Buffer) -> u32 {
        use super::binding_seam::{HeapKind, HeapResource};
        let slot = self.slot(i, HeapKind::Buffer);
        seam.rewrite_heap_index(
            HeapKind::Buffer,
            slot,
            HeapResource::Buffer {
                address: seam.device_address(buffer).get(),
                size: buffer.size(),
            },
        );
        slot
    }

    /// As [`buffer`](Self::buffer), but the descriptor covers only the
    /// leading `size` bytes — for sparse-backed columns bound at their
    /// COMMITTED size, so an out-of-range index is bounds-checked instead of
    /// faulting an unbound page.
    pub fn buffer_sized(
        &self,
        seam: &BindingSeam,
        i: usize,
        buffer: &wgpu::Buffer,
        size: u64,
    ) -> u32 {
        use super::binding_seam::{HeapKind, HeapResource};
        let slot = self.slot(i, HeapKind::Buffer);
        seam.rewrite_heap_index(
            HeapKind::Buffer,
            slot,
            HeapResource::Buffer {
                address: seam.device_address(buffer).get(),
                size,
            },
        );
        slot
    }

    /// Write a UNIFORM_BUFFER descriptor over `buffer` into slot `i` and
    /// return the slot for the push array — for params blocks too large for
    /// push data (`ConstantBuffer<T>` shader-side).
    pub fn uniform(&self, seam: &BindingSeam, i: usize, buffer: &wgpu::Buffer) -> u32 {
        use super::binding_seam::{HeapKind, HeapResource};
        let slot = self.slot(i, HeapKind::Buffer);
        seam.rewrite_heap_index(
            HeapKind::Buffer,
            slot,
            HeapResource::UniformBuffer {
                address: seam.device_address(buffer).get(),
                size: buffer.size(),
            },
        );
        slot
    }

    /// Write a STORAGE_IMAGE descriptor over `view` (layout `GENERAL` — the
    /// layout raw storage writes require and wgpu tracks storage-use textures
    /// in) into image-region slot `i` and return the slot for the push array.
    ///
    /// Per-dispatch rewrites are safe ONLY for a view whose identity is
    /// frame-stable (descriptors are read at EXECUTION, so frame N+1's
    /// rewrite is visible to frame N's still-in-flight dispatch — harmless
    /// when the content is identical). A view that alternates across frames
    /// (the view target's ping-pong textures) must go through
    /// [`ImageSlotCache`] instead.
    pub fn storage_image(&self, seam: &BindingSeam, i: usize, view: &wgpu::TextureView) -> u32 {
        use super::binding_seam::{HeapKind, HeapResource};
        use wgpu::hal::api::Vulkan as VkApi;
        let slot = self.slot(i, HeapKind::Image);
        // SAFETY: the view is a live wgpu resource on the Vulkan backend; the
        // guard is dropped before anything can destroy it.
        let hal_view = unsafe { view.as_hal::<VkApi>() }
            .expect("bevy_solari requires the Vulkan backend");
        let info = hal_view.image_view_create_info();
        seam.rewrite_heap_index(
            HeapKind::Image,
            slot,
            HeapResource::StorageImage {
                view: &info,
                layout: vk::ImageLayout::GENERAL,
            },
        );
        slot
    }
}

/// STORAGE_IMAGE heap slots keyed by the view's raw handle, each written
/// ONCE — for views whose identity ALTERNATES across frames (the view
/// target's ping-pong textures). Heap descriptors are read at execution, so
/// rewriting one shared slot per frame corrupts the still-in-flight previous
/// frame (it writes the next frame's texture; the recorded display chain
/// reads the untouched one — a black frame). One slot per distinct view
/// makes every frame's push data point at an immutable descriptor.
///
/// Entries are never freed: a stale view costs one 32-B image slot until app
/// exit (a handful per resize), which is far cheaper than the quiesce a safe
/// free would need.
pub struct ImageSlotCache {
    entries: Vec<(vk::ImageView, u32)>,
}

impl ImageSlotCache {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// The slot holding `view`'s STORAGE_IMAGE descriptor (layout `GENERAL`),
    /// allocated + written on first sight of this view.
    pub fn storage_image(&mut self, seam: &BindingSeam, view: &wgpu::TextureView) -> u32 {
        use super::binding_seam::HeapResource;
        use wgpu::hal::api::Vulkan as VkApi;
        // SAFETY: the view is a live wgpu resource on the Vulkan backend; the
        // guard is dropped before anything can destroy it.
        let hal_view = unsafe { view.as_hal::<VkApi>() }
            .expect("bevy_solari requires the Vulkan backend");
        // SAFETY: handle used only as a cache key while the guard is live.
        let key = unsafe { hal_view.raw_handle() };
        if let Some(&(_, slot)) = self.entries.iter().find(|&&(k, _)| k == key) {
            return slot;
        }
        let info = hal_view.image_view_create_info();
        let slot = seam.alloc_heap_index(HeapResource::StorageImage {
            view: &info,
            layout: vk::ImageLayout::GENERAL,
        });
        self.entries.push((key, slot));
        slot
    }
}

impl Default for ImageSlotCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    /// The dispatch contract is the bindings that SURVIVE in each entry's
    /// SPIR-V, not everything the module declares — the plain `scatter`
    /// entry must NOT retain the `previous` binding it never touches (the
    /// non-history column dispatches don't supply it).
    #[test]
    fn scatter_drops_unused_previous() {
        let (file, source, entry, modules) = super::HEAP_KERNELS
            .iter()
            .find(|&&(f, _, e, _)| f == "gpu_instances_scatter.slang" && e == "scatter")
            .copied()
            .unwrap();
        let shader =
            crate::gpu::slang::compile_rt_slang(file, source, entry, modules, &[], &[]).unwrap();
        let bindings: Vec<u32> =
            crate::gpu::binding_seam::spirv_descriptor_bindings(&shader.spirv)
                .iter()
                .map(|&(_, b, _)| b)
                .collect();
        assert_eq!(bindings, [0, 1], "plain scatter must bind exactly delta + column");
    }

    /// Every ported compute kernel compiles from source and reflects its
    /// set-0 bindings — a Slang error surfaces here instead of at app
    /// startup.
    #[test]
    fn heap_kernels_compile() {
        for &(file, source, entry, modules) in super::HEAP_KERNELS {
            let shader = crate::gpu::slang::compile_rt_slang(file, source, entry, modules, &[], &[])
                .unwrap_or_else(|e| panic!("{file}: {e}"));
            for (set, binding, kind) in
                crate::gpu::binding_seam::spirv_descriptor_bindings(&shader.spirv)
            {
                assert_eq!(set, 0, "{file}: (set {set}, binding {binding}) — heap kernels bind set 0 only");
                assert_ne!(
                    kind,
                    crate::gpu::binding_seam::SpirvBindingKind::AccelerationStructure,
                    "{file}: TLAS bindings need an explicit mapping table"
                );
            }
        }
    }

    /// `tess_gen_verts`' set-0 surface must be EXACTLY the displacement-map
    /// array at (0,6) + its sampler at (0,7) — the two constant-offset rows
    /// `init_tess_classify` bakes over the heap block/slot. A drifted binding
    /// number or kind here would desync that hand-built mapping table.
    #[test]
    fn tess_gen_verts_set0_is_the_displacement_block() {
        use crate::gpu::binding_seam::SpirvBindingKind;
        let (file, source, entry, modules, capabilities) = super::MULTI_SET_HEAP_KERNELS
            .iter()
            .find(|&&(f, _, e, _, _)| f == "tess_gen_verts.slang" && e == "gen_verts")
            .copied()
            .unwrap();
        let shader =
            crate::gpu::slang::compile_rt_slang(file, source, entry, modules, &[], capabilities)
                .unwrap();
        let set0: Vec<(u32, SpirvBindingKind)> =
            crate::gpu::binding_seam::spirv_descriptor_bindings(&shader.spirv)
                .into_iter()
                .filter(|&(set, _, _)| set == 0)
                .map(|(_, binding, kind)| (binding, kind))
                .collect();
        assert_eq!(
            set0,
            [(6, SpirvBindingKind::Image), (7, SpirvBindingKind::Sampler)],
            "gen_verts set 0 must be the displacement array + sampler"
        );
    }

    /// The multi-set kernels compile and keep the binding surface
    /// `new_with_mappings` maps: sets 0/1/2 only, the TLAS solely at (0,4)
    /// (its push-address row), every set-1 binding present in reflection
    /// (the dispatch's name→slot contract), and — for `restir_spatial` — no
    /// (1,4): `scene_resolve`'s `geometry_addresses` must be DCE'd, since the
    /// dispatch pushes no slot for it.
    #[test]
    fn multi_set_heap_kernels_compile() {
        use crate::gpu::binding_seam::SpirvBindingKind;
        for &(file, source, entry, modules, capabilities) in super::MULTI_SET_HEAP_KERNELS {
            let shader = crate::gpu::slang::compile_rt_slang(
                file, source, entry, modules, &[], capabilities,
            )
            .unwrap_or_else(|e| panic!("{file}: {e}"));
            for (set, binding, kind) in
                crate::gpu::binding_seam::spirv_descriptor_bindings(&shader.spirv)
            {
                assert!(
                    set <= 2,
                    "{file}: (set {set}, binding {binding}) — outside the scene/own/columns sets"
                );
                if kind == SpirvBindingKind::AccelerationStructure {
                    assert_eq!(
                        (set, binding),
                        (0, 4),
                        "{file}: the TLAS must sit at (0,4), where the push-address row maps it"
                    );
                }
                if set == 1 {
                    assert!(
                        shader.bindings.iter().any(|&(_, s, b)| (s, b) == (set, binding)),
                        "{file}: (set 1, binding {binding}) missing from slang reflection"
                    );
                    assert!(
                        file != "restir_spatial.slang" || binding != 4,
                        "{file}: (1,4) survived — `geometry_addresses` reached from a \
                         resolve helper; the dispatch pushes no slot for it"
                    );
                }
            }
        }
    }
}
