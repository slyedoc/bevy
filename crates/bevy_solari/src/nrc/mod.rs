//! Neural Radiance Cache: a small MLP trained online to predict cached
//! radiance at GI path terminations.
//!
//! Training runs on the render graph as a short chain (encode records →
//! fused forward+loss+backward → per-layer dW layout convert → adam,
//! dispatched after the trace each frame); raygen writes one training record
//! per rotating pixel subset. The fused kernel is Slang-compiled SPIR-V
//! (`nrc_train.slang`, passthrough-loaded): one thread evaluates the whole
//! coopvec MLP, seeds the loss gradient, and back-propagates — each layer
//! outer-product-accumulating dW (TrainingOptimal layout) and
//! reduce-sum-accumulating db in place, which
//! `vkCmdConvertCooperativeVectorMatrixNV` then converts to the row-major
//! f32 gradients adam reads. GI paths terminate into
//! the cache by appending queries that the `nrc_query_infer` pass
//! batch-evaluates (coherent coopvec against the transposed f16 weight
//! mirror adam maintains) and composites into the output buffer; training
//! paths and [`SolariNrc::inline_coopvec`] query inline in raygen instead.

#![allow(unsafe_code)]

use bevy_ecs::prelude::*;
use bevy_render::{
    extract_resource::ExtractResource,
    render_resource::{Buffer, BufferUsages},
    renderer::{RenderDevice, RenderQueue},
};
use half::f16;

use crate::gpu::allocator::{Allocator, MemoryLocation};
use ash::vk;
use wgpu::hal::api::Vulkan as VkApi;

pub const NRC_WIDTH: usize = 64;
pub const NRC_LAYERS: usize = 6;
pub const NRC_W_ELEMS: usize = NRC_WIDTH * NRC_WIDTH;
pub const NRC_W_TOTAL: usize = NRC_LAYERS * NRC_W_ELEMS;
pub const NRC_B_TOTAL: usize = NRC_LAYERS * NRC_WIDTH;
/// Training records per frame; must match `NRC_RECORD_CAP` in raygen.slang.
pub const NRC_RECORD_CAP: usize = 16384;
/// Bytes per `NrcRecord` (5 × vec4<f32>), must match raygen.slang/nrc_mlp.slang.
pub const NRC_RECORD_SIZE: usize = 80;
/// Bytes per termination query (3 × vec4<u32>); the per-view ring is sized to
/// the viewport (one query per pixel per frame) plus a 16-byte count header.
/// Must match `NrcQueryGpu`/`NrcQueryBuf` in raygen.slang and nrc_mlp.slang.
pub const NRC_QUERY_SIZE: usize = 48;
/// Bytes per record in the loss-readback staging buffer: 4 loss + 16 target.
const LOSS_STAGING_BYTES_PER_RECORD: u64 = 20;
/// Inference-mirror EMA weight per adam step (~50-step time constant): the
/// rendered cache tracks a smoothed trajectory of the optimizer instead of
/// its per-step oscillation.
pub const NRC_EMA_ALPHA: f32 = 0.02;

/// Main-world NRC configuration; extracted into the render world each frame.
#[derive(Resource, ExtractResource, Clone)]
pub struct SolariNrc {
    /// Master switch: record writes + training + queryability.
    pub enabled: bool,
    /// Run the online training passes.
    pub training: bool,
    /// Train once every N frames (amortizes the training + trace GPU load;
    /// the cache converges fine at 2-4). 1 = every frame.
    pub train_interval: u32,
    pub lr: f32,
    pub loss_scale: f32,
    /// Meters spanned by the position encoding's unit cube, centered on the
    /// camera's world-snapped anchor (`RtCamera::nrc_anchor`) — the cache
    /// relearns only when the camera hops a scale/2 anchor cell, not as it
    /// travels within one.
    pub scene_scale: f32,
    /// Spread-termination threshold c (paper §5.1): terminate a GI path into
    /// the cache once its footprint area exceeds c × the primary hit's. Larger
    /// c = deeper paths, less bias, more cost. ~0.01 is the paper's default.
    pub spread_c: f32,
    /// Terminate GI paths via an inline coopvec MLP query in raygen instead
    /// of the batched query→infer→composite path. (The debug view, training
    /// paths, and ReSTIR GI terminations always query inline.)
    pub inline_coopvec: bool,
    /// Read back + log the training loss every ~120 frames (blocking; debug).
    pub log_loss: bool,
}

impl Default for SolariNrc {
    fn default() -> Self {
        Self {
            enabled: true,
            training: true,
            train_interval: 2,
            lr: 1e-3,
            loss_scale: 16.0,
            scene_scale: 64.0,
            spread_c: 0.01,
            inline_coopvec: false,
            log_loss: true,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamParams {
    count: u32,
    step: u32,
    mirror_mode: u32,
    lr: f32,
    beta_one: f32,
    beta_two: f32,
    eps: f32,
    inv_grad_scale: f32,
    ema_alpha: f32,
    pad_a: u32,
    pad_b: u32,
    pad_c: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NrcTrainParams {
    count: u32,
    inv_scene_scale: f32,
    pad_a: u32,
    pad_b: u32,
}

// MUST MATCH TrainParams in nrc_train.slang.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LearnParams {
    batch: u32,
    loss_scale: f32,
    /// Bytes per layer's TrainingOptimal dW block.
    opt_size: u32,
    pad_b: u32,
}

/// The training/inference kernels, compiled from Slang source at startup
/// (`compile_rt_slang`) and created eagerly — nothing queued on the
/// `PipelineCache`.
#[derive(Resource)]
pub struct NrcPipelines {
    /// Heap-flagged (layout-free) raw compute pipelines, one per kernel. Every
    /// binding is a push-indexed heap slot
    /// ([`BindingSeam::create_heap_compute_pipeline`]), so a dispatch is
    /// push-slots + bind + `vkCmdDispatch` — the same pipeline serves any
    /// buffer set (adam runs twice, weights then biases, from one pipeline).
    ///
    /// [`BindingSeam::create_heap_compute_pipeline`]: crate::gpu::binding_seam::BindingSeam::create_heap_compute_pipeline
    learn: NrcKernel,
    adam: NrcKernel,
    encode_records: NrcKernel,
    query_infer: NrcKernel,
    seam: crate::gpu::binding_seam::BindingSeam,
    /// `VK_NV_cooperative_vector` fn table + the raw device — the per-layer
    /// TrainingOptimal→RowMajor dW conversion is a raw device command.
    coopvec_fns: ash::nv::cooperative_vector::Device,
    raw_device: ash::Device,
    /// Bytes per layer's TrainingOptimal dW block (host convert size query).
    pub opt_size: u32,
    /// Pins the `VkDevice` across [`Drop`]'s raw destroys (teardown order).
    _device_keepalive: Allocator,
}

impl Drop for NrcPipelines {
    fn drop(&mut self) {
        // In-flight dispatches may still reference the pipelines; drain first.
        self._device_keepalive.quiesce_before_raw_destroy();
        for kernel in [
            &self.learn,
            &self.adam,
            &self.encode_records,
            &self.query_infer,
        ] {
            // SAFETY: quiesced; handles exclusively owned here.
            unsafe {
                self.raw_device.destroy_pipeline(kernel.pipeline, None);
                self.raw_device.destroy_shader_module(kernel.module, None);
            }
        }
    }
}

/// One heap-flagged kernel plus its reflected set-0 parameter table — the
/// contract [`push_slots`](Self::push_slots) assembles push data against.
struct NrcKernel {
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    /// `(parameter name, binding)` from slang reflection of this kernel.
    bindings: Vec<(String, u32)>,
}

impl NrcKernel {
    /// Assemble the push-data slot array from `(parameter name, heap slot)`
    /// pairs: `slots[binding] = slot`, with the binding read from the
    /// shader's own reflected layout. Any mismatch — a missing, misnamed,
    /// duplicated, or extra parameter — panics naming the kernel and the
    /// parameter, so a shader binding edit can't silently desync a dispatch.
    fn push_slots(&self, label: &str, named: &[(&str, u32)]) -> Vec<u32> {
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
                    "nrc: {label} has no parameter `{name}` (shader declares {:?})",
                    self.bindings
                );
            };
            assert!(
                slots[binding as usize] == u32::MAX,
                "nrc: {label}: parameter `{name}` supplied twice"
            );
            slots[binding as usize] = slot;
        }
        for (name, binding) in &self.bindings {
            assert!(
                slots[*binding as usize] != u32::MAX,
                "nrc: {label}: parameter `{name}` not supplied"
            );
        }
        slots
    }
}

/// All NRC GPU state. Weights/optimizer are global (one cache per app);
/// records rotate through a fixed-capacity ring written by raygen.
///
/// Many buffer fields are never read after init: they exist as OWNERSHIP —
/// their memory backs live heap descriptors ([`NrcHeapSlots`]) the raw
/// dispatches read through, and dropping one would free it under the GPU.
#[allow(dead_code)]
#[derive(Resource)]
pub struct NrcBuffers {
    // RT-visible (raw VK handles baked into set 1)
    pub weights_t: Buffer,
    pub weights_t_raw: vk::Buffer,
    pub weights_t_size: u64,
    pub bias16: Buffer,
    pub bias16_raw: vk::Buffer,
    pub bias16_size: u64,
    pub records: Buffer,
    pub records_raw: vk::Buffer,
    // training-only (plain wgpu)
    w16: Buffer,
    /// f16 `[batch][WIDTH]` encoded batch (the network input activations).
    acts: Buffer,
    /// f32 `[batch][WIDTH]` fused-forward predictions (debug visibility;
    /// the fused kernel writes them unconditionally).
    preds: Buffer,
    targets: Buffer,
    loss: Buffer,
    loss_staging: Buffer,
    zeros: Buffer,
    master_w: Buffer,
    master_b: Buffer,
    dw: Buffer,
    db: Buffer,
    m_w: Buffer,
    v_w: Buffer,
    m_b: Buffer,
    v_b: Buffer,
    ema_w: Buffer,
    ema_b: Buffer,
    weights_t_ema: Buffer,
    bias16_ema: Buffer,
    /// Per-layer TrainingOptimal-layout dW accumulator (`LAYERS` blocks of
    /// `NrcPipelines::opt_size` bytes); the fused kernel outer-product
    /// accumulates into it, the raw convert reads it out row-major into `dw`.
    dw_opt: Buffer,
    /// Buffer device addresses for the raw dW layout conversion (wgpu-hal
    /// creates every storage buffer with `SHADER_DEVICE_ADDRESS`).
    dw_opt_addr: u64,
    dw_addr: u64,
    learn_ubo: Buffer,
    adam_w_ubo: Buffer,
    adam_b_ubo: Buffer,
    train_ubo: Buffer,
    query_ubo: Buffer,
    /// Every buffer's descriptor-heap slot — the dispatch functions push the
    /// per-kernel slot arrays for the pipelines' push-indexed mappings.
    slots: NrcHeapSlots,
    pub step: u32,
    /// Non-blocking loss readback: 0 = idle, 1 = copied (map next frame),
    /// 2 = map requested (read when the shared flag flips).
    loss_state: u32,
    loss_mapped: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// Heap slots for the NRC buffers (allocated once in [`init_nrc_buffers`] —
/// the buffers never reallocate). Uniform param blocks get UNIFORM_BUFFER
/// descriptors; everything else STORAGE_BUFFER.
struct NrcHeapSlots {
    records: u32,
    acts: u32,
    targets: u32,
    preds: u32,
    loss: u32,
    dw_opt: u32,
    db: u32,
    zeros: u32,
    weights_t: u32,
    bias16: u32,
    w16: u32,
    master_w: u32,
    master_b: u32,
    dw: u32,
    m_w: u32,
    v_w: u32,
    m_b: u32,
    v_b: u32,
    ema_w: u32,
    ema_b: u32,
    weights_t_ema: u32,
    bias16_ema: u32,
    train_ubo: u32,
    learn_ubo: u32,
    adam_w_ubo: u32,
    adam_b_ubo: u32,
    query_ubo: u32,
}

/// `RenderStartup` (after `SolariSetup`): heap-flagged raw pipelines, one per
/// kernel, compiled from Slang source here (`nrc_mlp` importable, so the MLP
/// constants agree across kernels by construction). Every `[[vk::binding]]` in
/// the kernels is a push-indexed heap slot — the binding tables live in the
/// dispatch functions' slot arrays, which must match nrc_train.slang /
/// nrc_mlp.slang order.
#[allow(unsafe_code)]
pub fn init_nrc_pipelines(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    allocator: Option<Res<Allocator>>,
    seam: Option<Res<crate::gpu::binding_seam::BindingSeam>>,
) {
    let (Some(allocator), Some(seam)) = (allocator, seam) else {
        return;
    };

    // Nothing about the kernels' bindings is hand-maintained: the mapping
    // table per kernel is derived from its compiled SPIR-V
    // (`create_heap_compute_pipeline`), and the dispatch slot arrays are
    // assembled by parameter NAME against the kernel's reflected layout
    // ([`NrcKernel::push_slots`]).
    let mlp: &[(&str, &str)] = &[("nrc_mlp", include_str!("nrc_mlp.slang"))];
    let make = |label: &'static str, file: &'static str, source: &'static str, entry: &'static str| {
        let shader = crate::gpu::slang::compile_rt_slang(
            file,
            source,
            entry,
            mlp,
            &[],
            &[],
        )
        .map_err(|e| bevy_log::error!("nrc: {e}"))
        .ok()?;
        let entry_c = std::ffi::CString::new(entry).expect("entry name has interior NUL");
        let (module, pipeline) = seam.create_heap_compute_pipeline(&shader.spirv, &entry_c, label)?;
        Some(NrcKernel {
            module,
            pipeline,
            bindings: shader
                .bindings
                .into_iter()
                .filter(|&(_, set, _)| set == 0)
                .map(|(name, _, binding)| (name, binding))
                .collect(),
        })
    };
    let (Some(learn), Some(adam), Some(encode_records), Some(query_infer)) = (
        make(
            "nrc_learn",
            "nrc_train.slang",
            include_str!("nrc_train.slang"),
            "learn_gradient",
        ),
        make(
            "nrc_adam",
            "nrc_adam.slang",
            include_str!("nrc_adam.slang"),
            "adam",
        ),
        make(
            "nrc_encode_records",
            "nrc_encode_records.slang",
            include_str!("nrc_encode_records.slang"),
            "nrc_encode_records",
        ),
        make(
            "nrc_query_infer",
            "nrc_query_infer.slang",
            include_str!("nrc_query_infer.slang"),
            "nrc_query_infer",
        ),
    ) else {
        return;
    };

    // Raw handles for the dW layout conversion: the fn table once
    // (extension.rs pattern) plus the host-side size query for the opaque
    // TrainingOptimal block one layer's dW occupies (`dst_data` null = size
    // query only).
    // SAFETY: as_hal yields the raw Vulkan device only while the wgpu Device
    // is alive; we read function pointers and query a size.
    let (coopvec_fns, raw_device, opt_size) = unsafe {
        let hal_device = render_device
            .wgpu_device()
            .as_hal::<VkApi>()
            .expect("bevy_solari requires the Vulkan backend");
        let fns = ash::nv::cooperative_vector::Device::load(
            hal_device.shared_instance().raw_instance(),
            hal_device.raw_device(),
        );
        let width = NRC_WIDTH;
        let mut dst_size: usize = 0;
        let mut info = vk::ConvertCooperativeVectorMatrixInfoNV::default()
            .src_size(width * width * 4)
            .src_data(vk::DeviceOrHostAddressConstKHR {
                host_address: core::ptr::null(),
            })
            .dst_data(vk::DeviceOrHostAddressKHR {
                host_address: core::ptr::null_mut(),
            })
            .src_component_type(vk::ComponentTypeKHR::FLOAT32)
            .dst_component_type(vk::ComponentTypeKHR::FLOAT32)
            .num_rows(width as u32)
            .num_columns(width as u32)
            .src_layout(vk::CooperativeVectorMatrixLayoutNV::ROW_MAJOR)
            .src_stride(width * 4)
            .dst_layout(vk::CooperativeVectorMatrixLayoutNV::TRAINING_OPTIMAL)
            .dst_stride(0);
        info.p_dst_size = &mut dst_size;
        fns.convert_cooperative_vector_matrix(&info)
            .expect("vkConvertCooperativeVectorMatrixNV size query failed");
        // Round the per-layer stride to 64 B: `vkCmdConvertCooperativeVector-
        // MatrixNV` requires 64-B-aligned src/dst addresses (VUID 10084/
        // 10085), and every layer offset is a multiple of this stride. All
        // consumers — the dw_opt buffer size, the fused kernel's accumulate
        // offsets, the convert src addresses — inherit it from here.
        (fns, hal_device.raw_device().clone(), (dst_size as u32).next_multiple_of(64))
    };

    commands.insert_resource(NrcPipelines {
        learn,
        adam,
        encode_records,
        query_infer,
        seam: seam.clone(),
        coopvec_fns,
        raw_device,
        opt_size,
        _device_keepalive: allocator.clone(),
    });
}

/// `Render::Prepare`: create the buffers once the [`Allocator`] exists.
pub fn init_nrc_buffers(
    mut commands: Commands,
    existing: Option<Res<NrcBuffers>>,
    allocator: Option<Res<Allocator>>,
    seam: Option<Res<crate::gpu::binding_seam::BindingSeam>>,
    pipelines: Option<Res<NrcPipelines>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    settings: Res<crate::SolariSettings>,
) {
    if existing.is_some() {
        return;
    }
    let Some(allocator) = allocator else { return };
    let Some(seam) = seam else { return };
    let Some(pipelines) = pipelines else { return };

    let batch = NRC_RECORD_CAP as u64;
    let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
    // STORAGE rides along for `SHADER_DEVICE_ADDRESS` (the fork adds the
    // address bit to storage buffers) — the heap's UNIFORM_BUFFER descriptors
    // are written from device addresses.
    let uniform = BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST;
    let mk = |label: &'static str, size: u64, usage: BufferUsages| {
        render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let mk_raw = |label: &'static str, size: u64| {
        allocator.create_buffer_raw(
            &render_device,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            storage,
            size,
            MemoryLocation::GpuOnly,
            label,
        )
    };

    // +one layer of padding: coopVecMatMulAdd reads are outside
    // robustBufferAccess coverage, so an exact-fit buffer faults on any
    // hardware over-read at the last layer (Xid 31/32 MMU/pushbuffer).
    const NRC_W_PAD: u64 = (NRC_W_ELEMS * 2) as u64;
    const NRC_B_PAD: u64 = (NRC_WIDTH * 2) as u64;
    let (weights_t, weights_t_raw) = mk_raw("nrc_weights_t", (NRC_W_TOTAL * 2) as u64 + NRC_W_PAD);
    let (bias16, bias16_raw) = mk_raw("nrc_bias16", (NRC_B_TOTAL * 2) as u64 + NRC_B_PAD);
    let (records, records_raw) =
        mk_raw("nrc_records", (NRC_RECORD_CAP * NRC_RECORD_SIZE) as u64);
    let weights_t: Buffer = weights_t.into();
    let bias16: Buffer = bias16.into();
    let records: Buffer = records.into();

    // Deterministic He-uniform init, mirrored three ways;
    // `SolariSettings::nrc_seed` sets the seed.
    let mut state: u64 = settings.nrc_seed;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32
    };
    let bound = (6.0f32 / NRC_WIDTH as f32).sqrt();
    let master_w_data: Vec<f32> = (0..NRC_W_TOTAL)
        .map(|_| (next() * 2.0 - 1.0) * bound)
        .collect();
    let w16_data: Vec<f16> = master_w_data.iter().map(|&w| f16::from_f32(w)).collect();
    let mut w16_t_data = vec![f16::from_f32(0.0); NRC_W_TOTAL];
    for l in 0..NRC_LAYERS {
        for r in 0..NRC_WIDTH {
            for c in 0..NRC_WIDTH {
                w16_t_data[l * NRC_W_ELEMS + c * NRC_WIDTH + r] =
                    w16_data[l * NRC_W_ELEMS + r * NRC_WIDTH + c];
            }
        }
    }

    let w16 = mk("nrc_w16", (NRC_W_TOTAL * 2) as u64, storage);
    render_queue.write_buffer(&w16, 0, bytemuck::cast_slice(&w16_data));
    render_queue.write_buffer(&weights_t, 0, bytemuck::cast_slice(&w16_t_data));
    render_queue.write_buffer(
        &bias16,
        0,
        bytemuck::cast_slice(&vec![f16::from_f32(0.0); NRC_B_TOTAL]),
    );
    render_queue.write_buffer(
        &records,
        0,
        &vec![0u8; NRC_RECORD_CAP * NRC_RECORD_SIZE],
    );
    let master_w = mk("nrc_master_w", (NRC_W_TOTAL * 4) as u64, storage);
    render_queue.write_buffer(&master_w, 0, bytemuck::cast_slice(&master_w_data));
    let ema_w = mk("nrc_ema_w", (NRC_W_TOTAL * 4) as u64, storage);
    render_queue.write_buffer(&ema_w, 0, bytemuck::cast_slice(&master_w_data));
    let weights_t_ema = mk(
        "nrc_weights_t_ema",
        (NRC_W_TOTAL * 2) as u64 + NRC_W_PAD,
        storage,
    );
    render_queue.write_buffer(&weights_t_ema, 0, bytemuck::cast_slice(&w16_t_data));
    let bias16_ema = mk("nrc_bias16_ema", (NRC_B_TOTAL * 2) as u64 + NRC_B_PAD, storage);
    render_queue.write_buffer(
        &bias16_ema,
        0,
        bytemuck::cast_slice(&vec![f16::from_f32(0.0); NRC_B_TOTAL]),
    );

    let width = NRC_WIDTH as u64;
    let mk_ubo = |label: &'static str, size: u64| mk(label, size, uniform);

    // TrainingOptimal dW accumulator + the row-major dW the raw convert fills;
    // their device addresses feed vkCmdConvertCooperativeVectorMatrixNV.
    let dw_opt = mk(
        "nrc_dw_opt",
        NRC_LAYERS as u64 * pipelines.opt_size as u64,
        storage,
    );
    let dw = mk("nrc_dw", (NRC_W_TOTAL * 4) as u64, storage);
    // Mark dw initialized with a tracked write: wgpu lazily ZERO-INITIALIZES
    // a buffer at its first tracked use, and dw is only ever written by the
    // raw (untracked) layout convert — without this, adam's first read would
    // wipe the converted gradients.
    render_queue.write_buffer(&dw, 0, &vec![0u8; NRC_W_TOTAL * 4]);
    let dw_opt_addr = allocator.wgpu_buffer_device_address(&dw_opt).get();
    let dw_addr = allocator.wgpu_buffer_device_address(&dw).get();

    let acts = mk("nrc_acts", batch * width * 2, storage);
    let preds = mk("nrc_preds", batch * width * 4, storage);
    let targets = mk("nrc_targets", batch * 16, storage);
    let loss = mk("nrc_loss", batch * 4, storage);
    let zeros = mk("nrc_zeros", 1024, storage);
    let master_b = mk("nrc_master_b", (NRC_B_TOTAL * 4) as u64, storage);
    let db = mk("nrc_db", (NRC_B_TOTAL * 4) as u64, storage);
    let m_w = mk("nrc_m_w", (NRC_W_TOTAL * 4) as u64, storage);
    let v_w = mk("nrc_v_w", (NRC_W_TOTAL * 4) as u64, storage);
    let m_b = mk("nrc_m_b", (NRC_B_TOTAL * 4) as u64, storage);
    let v_b = mk("nrc_v_b", (NRC_B_TOTAL * 4) as u64, storage);
    let ema_b = mk("nrc_ema_b", (NRC_B_TOTAL * 4) as u64, storage);
    let learn_ubo = mk_ubo("nrc_learn_ubo", 16);
    let adam_w_ubo = mk_ubo("nrc_adam_w_ubo", 48);
    let adam_b_ubo = mk_ubo("nrc_adam_b_ubo", 48);
    let train_ubo = mk_ubo("nrc_train_ubo", 16);
    let query_ubo = mk_ubo("nrc_query_ubo", 16);

    // Mark every raw-only buffer initialized with a TRACKED write: wgpu lazily
    // zero-initializes a buffer at its first tracked use, and these are
    // otherwise touched only by untracked raw dispatches — so a later tracked
    // use (the loss/targets staging copies) would wipe raw-written contents,
    // and never-tracked buffers (adam moments, zeros) would start as garbage.
    let zero_init: [(&Buffer, usize); 9] = [
        (&zeros, 1024),
        (&master_b, NRC_B_TOTAL * 4),
        (&ema_b, NRC_B_TOTAL * 4),
        (&m_w, NRC_W_TOTAL * 4),
        (&v_w, NRC_W_TOTAL * 4),
        (&m_b, NRC_B_TOTAL * 4),
        (&v_b, NRC_B_TOTAL * 4),
        (&loss, batch as usize * 4),
        (&targets, batch as usize * 16),
    ];
    for (buf, bytes) in zero_init {
        render_queue.write_buffer(buf, 0, &vec![0u8; bytes]);
    }

    // Heap slots (the buffers are app-lifetime; the pipelines' push-indexed
    // mappings receive these per dispatch).
    use crate::gpu::binding_seam::HeapResource;
    let sslot = |b: &Buffer| {
        seam.alloc_heap_index(HeapResource::Buffer {
            address: seam.device_address(b).get(),
            size: b.size(),
        })
    };
    let uslot = |b: &Buffer| {
        seam.alloc_heap_index(HeapResource::UniformBuffer {
            address: seam.device_address(b).get(),
            size: b.size(),
        })
    };
    let slots = NrcHeapSlots {
        records: sslot(&records),
        acts: sslot(&acts),
        targets: sslot(&targets),
        preds: sslot(&preds),
        loss: sslot(&loss),
        dw_opt: sslot(&dw_opt),
        db: sslot(&db),
        zeros: sslot(&zeros),
        weights_t: sslot(&weights_t),
        bias16: sslot(&bias16),
        w16: sslot(&w16),
        master_w: sslot(&master_w),
        master_b: sslot(&master_b),
        dw: sslot(&dw),
        m_w: sslot(&m_w),
        v_w: sslot(&v_w),
        m_b: sslot(&m_b),
        v_b: sslot(&v_b),
        ema_w: sslot(&ema_w),
        ema_b: sslot(&ema_b),
        weights_t_ema: sslot(&weights_t_ema),
        bias16_ema: sslot(&bias16_ema),
        train_ubo: uslot(&train_ubo),
        learn_ubo: uslot(&learn_ubo),
        adam_w_ubo: uslot(&adam_w_ubo),
        adam_b_ubo: uslot(&adam_b_ubo),
        query_ubo: uslot(&query_ubo),
    };

    commands.insert_resource(NrcBuffers {
        weights_t,
        weights_t_raw,
        weights_t_size: (NRC_W_TOTAL * 2) as u64 + NRC_W_PAD,
        bias16,
        bias16_raw,
        bias16_size: (NRC_B_TOTAL * 2) as u64 + NRC_B_PAD,
        records,
        records_raw,
        w16,
        acts,
        preds,
        targets,
        loss,
        loss_staging: mk(
            "nrc_loss_staging",
            // 4 loss bytes + 16 target bytes per record.
            batch * LOSS_STAGING_BYTES_PER_RECORD,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        ),
        zeros,
        master_w,
        master_b,
        dw,
        db,
        m_w,
        v_w,
        m_b,
        v_b,
        ema_w,
        ema_b,
        weights_t_ema,
        bias16_ema,
        dw_opt,
        dw_opt_addr,
        dw_addr,
        learn_ubo,
        adam_w_ubo,
        adam_b_ubo,
        train_ubo,
        query_ubo,
        slots,
        step: 0,
        loss_state: 0,
        loss_mapped: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    });
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NrcQueryParams {
    scale: f32,
    inv_exposure: f32,
    cap: u32,
    pad: u32,
}

/// Record the batched termination-query inference + composite pass: evaluate
/// the MLP for every query raygen appended this frame and add the
/// de-factorized radiance into the per-pixel output buffer (`scale` pre-folds
/// the raygen accumulation blend). A raw heap dispatch in its own command
/// buffer, appended AFTER the trace's; `queries_slot`/`output_slot` are the
/// view's heap slots (the same ones the trace pushes).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_nrc_query_infer(
    ctx: &mut bevy_render::renderer::RenderContext,
    bufs: &NrcBuffers,
    pipelines: &NrcPipelines,
    device: &RenderDevice,
    queue: &RenderQueue,
    queries_slot: u32,
    output_slot: u32,
    cap: u32,
    scale: f32,
    exposure: f32,
) -> bool {
    queue.write_buffer(
        &bufs.query_ubo,
        0,
        bytemuck::bytes_of(&NrcQueryParams {
            scale,
            inv_exposure: 1.0 / exposure.max(1.0e-9),
            cap,
            pad: 0,
        }),
    );
    let slots = pipelines.query_infer.push_slots(
        "nrc_query_infer",
        &[
            ("weights_t", bufs.slots.weights_t_ema),
            ("biases", bufs.slots.bias16_ema),
            ("nrc_queries", queries_slot),
            ("out_radiance", output_slot),
            ("qparams", bufs.slots.query_ubo),
        ],
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("nrc_query_infer"),
    });
    // SAFETY: Vulkan backend; the slots reference live heap descriptors.
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &pipelines.raw_device;
            let seam = &pipelines.seam;
            // Queries were appended by the trace (its post-barrier already
            // covers RT write -> compute read on this queue).
            seam.bind_heaps(cb);
            seam.push_data(cb, bytemuck::cast_slice(&slots));
            dev.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.query_infer.pipeline,
            );
            // The count lives on the GPU; over-dispatch to the cap, threads
            // early-out.
            dev.cmd_dispatch(cb, cap.div_ceil(64), 1, 1);
            // Composite writes -> the wgpu blit's read of the output buffer.
            let post = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)];
            dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&post));
        });
    }
    ctx.add_command_buffer(encoder.finish());
    true
}

/// Record the per-frame training pass chain onto `encoder`. Called from the
/// `rt_pipeline` render-graph node after the trace (records are fresh).
/// Returns false when pipelines aren't compiled yet.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_training(
    ctx: &mut bevy_render::renderer::RenderContext,
    bufs: &mut NrcBuffers,
    pipelines: &NrcPipelines,
    device: &RenderDevice,
    queue: &RenderQueue,
    nrc: &SolariNrc,
) -> bool {

    bufs.step += 1;
    let batch = NRC_RECORD_CAP as u32;
    let inv_grad = 1.0 / (nrc.loss_scale * batch as f32);
    queue.write_buffer(
        &bufs.train_ubo,
        0,
        bytemuck::bytes_of(&NrcTrainParams {
            count: batch,
            inv_scene_scale: 1.0 / nrc.scene_scale,
            pad_a: 0,
            pad_b: 0,
        }),
    );
    queue.write_buffer(
        &bufs.learn_ubo,
        0,
        bytemuck::bytes_of(&LearnParams {
            batch,
            loss_scale: nrc.loss_scale,
            opt_size: pipelines.opt_size,
            pad_b: 0,
        }),
    );
    queue.write_buffer(
        &bufs.adam_w_ubo,
        0,
        bytemuck::bytes_of(&AdamParams {
            count: NRC_W_TOTAL as u32,
            step: bufs.step,
            mirror_mode: 1,
            lr: nrc.lr,
            beta_one: 0.9,
            beta_two: 0.999,
            eps: 1e-8,
            inv_grad_scale: inv_grad,
            ema_alpha: NRC_EMA_ALPHA,
            pad_a: 0,
            pad_b: 0,
            pad_c: 0,
        }),
    );
    queue.write_buffer(
        &bufs.adam_b_ubo,
        0,
        bytemuck::bytes_of(&AdamParams {
            count: NRC_B_TOTAL as u32,
            step: bufs.step,
            mirror_mode: 2,
            lr: nrc.lr,
            beta_one: 0.9,
            beta_two: 0.999,
            eps: 1e-8,
            inv_grad_scale: inv_grad,
            ema_alpha: NRC_EMA_ALPHA,
            pad_a: 0,
            pad_b: 0,
            pad_c: 0,
        }),
    );

    // Per-kernel heap-slot arrays, assembled by parameter name against each
    // kernel's reflected binding table — a shader binding rename/reorder
    // panics here instead of silently desyncing.
    let s = &bufs.slots;
    let encode_slots = pipelines.encode_records.push_slots(
        "nrc_encode_records",
        &[
            ("records", s.records),
            ("act_out", s.acts),
            ("targets_out", s.targets),
            ("train_params", s.train_ubo),
        ],
    );
    let learn_slots = pipelines.learn.push_slots(
        "nrc_learn",
        &[
            ("acts", s.acts),
            ("weights_t", s.weights_t),
            ("biases", s.bias16),
            ("targets", s.targets),
            ("preds", s.preds),
            ("loss_out", s.loss),
            ("dw_opt", s.dw_opt),
            ("db", s.db),
            ("params", s.learn_ubo),
            ("zeros_buf", s.zeros),
        ],
    );
    let adam_w_slots = pipelines.adam.push_slots(
        "nrc_adam (weights)",
        &[
            ("master", s.master_w),
            ("grad_in", s.dw),
            ("moment_m", s.m_w),
            ("moment_v", s.v_w),
            ("mirror_f16", s.w16),
            ("mirror_alt", s.weights_t),
            ("ema_master", s.ema_w),
            ("mirror_ema", s.weights_t_ema),
            ("adam_u", s.adam_w_ubo),
        ],
    );
    let adam_b_slots = pipelines.adam.push_slots(
        "nrc_adam (biases)",
        &[
            ("master", s.master_b),
            ("grad_in", s.db),
            ("moment_m", s.m_b),
            ("moment_v", s.v_b),
            ("mirror_f16", s.bias16),
            ("mirror_alt", s.weights_t),
            ("ema_master", s.ema_b),
            ("mirror_ema", s.bias16_ema),
            ("adam_u", s.adam_b_ubo),
        ],
    );

    // The whole chain — gradient clears, encode, fused train, dW layout
    // convert, adam ×2 — in ONE raw command buffer with sync2 barriers (wgpu
    // tracks none of it; the UBO writes above ride the pre-submit staging
    // belt).
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("nrc_training"),
    });
    // SAFETY: Vulkan backend; every slot references a live heap descriptor;
    // the convert addresses point at live SHADER_DEVICE_ADDRESS buffers.
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &pipelines.raw_device;
            let seam = &pipelines.seam;
            let barrier = |src_stage, src_access, dst_stage, dst_access| {
                vk::MemoryBarrier2::default()
                    .src_stage_mask(src_stage)
                    .src_access_mask(src_access)
                    .dst_stage_mask(dst_stage)
                    .dst_access_mask(dst_access)
            };
            let barrier2 = |cb, b: &[vk::MemoryBarrier2]| {
                dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(b));
            };
            let raw_buf = |b: &Buffer| -> vk::Buffer {
                b.as_hal::<VkApi>()
                    .map(|hb| hb.raw_handle())
                    .expect("bevy_solari requires the Vulkan backend")
            };

            // Zero the gradient accumulators (the fused kernel's accumulates
            // are additive). Last frame's convert read dw_opt and adam read
            // db/dw — fence those reads (and the trace's record writes) before
            // the clears and the reads below.
            let pre = [barrier(
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                    | vk::PipelineStageFlags2::CONVERT_COOPERATIVE_VECTOR_MATRIX_NV,
                vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::TRANSFER_READ,
                vk::PipelineStageFlags2::CLEAR | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::TRANSFER_WRITE
                    | vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::SHADER_WRITE,
            )];
            barrier2(cb, &pre);
            dev.cmd_fill_buffer(cb, raw_buf(&bufs.dw_opt), 0, vk::WHOLE_SIZE, 0);
            dev.cmd_fill_buffer(cb, raw_buf(&bufs.db), 0, vk::WHOLE_SIZE, 0);
            barrier2(
                cb,
                &[barrier(
                    vk::PipelineStageFlags2::CLEAR,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                )],
            );

            seam.bind_heaps(cb);
            let compute_to_compute = [barrier(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            )];

            // encode: records -> acts + targets
            seam.push_data(cb, bytemuck::cast_slice(&encode_slots));
            dev.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.encode_records.pipeline,
            );
            dev.cmd_dispatch(cb, batch.div_ceil(64), 1, 1);
            barrier2(cb, &compute_to_compute);

            // fused forward + loss + backward incl. the dW/db accumulates
            seam.push_data(cb, bytemuck::cast_slice(&learn_slots));
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipelines.learn.pipeline);
            dev.cmd_dispatch(cb, batch.div_ceil(64), 1, 1);

            // Per-layer TrainingOptimal -> RowMajor dW conversion.
            let to_convert = [barrier(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::CONVERT_COOPERATIVE_VECTOR_MATRIX_NV,
                vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
            )];
            barrier2(cb, &to_convert);
            let opt_size = pipelines.opt_size as usize;
            let row_bytes = NRC_W_ELEMS * 4;
            let mut dst_sizes = [row_bytes; NRC_LAYERS];
            let infos: Vec<vk::ConvertCooperativeVectorMatrixInfoNV> = (0..NRC_LAYERS)
                .map(|l| {
                    let mut info = vk::ConvertCooperativeVectorMatrixInfoNV::default()
                        .src_size(opt_size)
                        .src_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: bufs.dw_opt_addr + (l * opt_size) as u64,
                        })
                        .dst_data(vk::DeviceOrHostAddressKHR {
                            device_address: bufs.dw_addr + (l * row_bytes) as u64,
                        })
                        .src_component_type(vk::ComponentTypeKHR::FLOAT32)
                        .dst_component_type(vk::ComponentTypeKHR::FLOAT32)
                        .num_rows(NRC_WIDTH as u32)
                        .num_columns(NRC_WIDTH as u32)
                        .src_layout(vk::CooperativeVectorMatrixLayoutNV::TRAINING_OPTIMAL)
                        .src_stride(0)
                        .dst_layout(vk::CooperativeVectorMatrixLayoutNV::ROW_MAJOR)
                        .dst_stride(NRC_WIDTH * 4);
                    info.p_dst_size = &mut dst_sizes[l];
                    info
                })
                .collect();
            pipelines
                .coopvec_fns
                .cmd_convert_cooperative_vector_matrix(cb, &infos);
            barrier2(
                cb,
                &[barrier(
                    vk::PipelineStageFlags2::CONVERT_COOPERATIVE_VECTOR_MATRIX_NV,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_READ,
                )],
            );

            // adam: weights, then biases (both write the transposed weight
            // mirror -> fence between them).
            seam.push_data(cb, bytemuck::cast_slice(&adam_w_slots));
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipelines.adam.pipeline);
            dev.cmd_dispatch(cb, (NRC_W_TOTAL as u32).div_ceil(64), 1, 1);
            barrier2(cb, &compute_to_compute);
            seam.push_data(cb, bytemuck::cast_slice(&adam_b_slots));
            dev.cmd_dispatch(cb, (NRC_B_TOTAL as u32).div_ceil(64), 1, 1);

            // Updated mirrors -> next frame's raygen inference; loss/targets
            // -> the tracked staging copies below.
            let post = [barrier(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                    | vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::TRANSFER_READ,
            )];
            barrier2(cb, &post);
        });
    }
    ctx.add_command_buffer(encoder.finish());


    let encoder = ctx.command_encoder();
    if nrc.log_loss && bufs.step % 120 == 0 && bufs.loss_state == 0 {
        encoder.copy_buffer_to_buffer(&bufs.loss, 0, &bufs.loss_staging, 0, batch as u64 * 4);
        encoder.copy_buffer_to_buffer(
            &bufs.targets,
            0,
            &bufs.loss_staging,
            batch as u64 * 4,
            batch as u64 * 16,
        );
        bufs.loss_state = 1;
    }
    true
}

/// `Render::Cleanup`: fully non-blocking staged loss readback — a blocking
/// poll here races the raw-VK trace machinery and can lose the device.
/// Copy → map_async next frame → read once the callback fires.
pub fn log_nrc_loss(bufs: Option<ResMut<NrcBuffers>>, _device: Res<RenderDevice>) {
    use std::sync::atomic::Ordering;
    let Some(mut bufs) = bufs else { return };
    match bufs.loss_state {
        1 => {
            let flag = bufs.loss_mapped.clone();
            flag.store(false, Ordering::Release);
            bufs.loss_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    if r.is_ok() {
                        flag.store(true, Ordering::Release);
                    }
                });
            bufs.loss_state = 2;
        }
        2 if bufs.loss_mapped.load(Ordering::Acquire) => {
            {
                let data = bufs.loss_staging.slice(..).get_mapped_range();
                let all: &[f32] = bytemuck::cast_slice(&data);
                let batch = NRC_RECORD_CAP;
                let losses = &all[..batch];
                let targets = &all[batch..batch * 5];
                let nan_losses = losses.iter().filter(|l| !l.is_finite()).count();
                let finite_sum: f64 = losses
                    .iter()
                    .filter(|l| l.is_finite())
                    .map(|&l| l as f64)
                    .sum();
                let mean = finite_sum / (batch - nan_losses).max(1) as f64;
                let max_target = targets
                    .chunks(4)
                    .map(|c| c[0].max(c[1]).max(c[2]))
                    .fold(0.0f32, f32::max);
                let valid: usize = targets
                    .chunks(4)
                    .filter(|c| c[3] > 0.0)
                    .count();
                bevy_log::info!(
                    "nrc: step {} relative-L2 {mean:.5} (nan {nan_losses}, valid {valid}/{batch}, max target {max_target:.2})",
                    bufs.step
                );
            }
            bufs.loss_staging.unmap();
            bufs.loss_state = 0;
        }
        _ => {}
    }
}
