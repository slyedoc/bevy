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
    render_resource::{
        BindGroup, BindGroupEntries, Buffer, BufferUsages, ComputePassDescriptor, ShaderStages,
    },
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

/// Bind-group layouts + pipelines for the training kernels. Every kernel is
/// Slang-compiled passthrough SPIR-V (nrc_train.slang + nrc_mlp.slang — regen
/// commands in their headers) with an explicit layout, created eagerly: no
/// naga reflection, nothing queued on the `PipelineCache`.
#[derive(Resource)]
pub struct NrcPipelines {
    adam_layout: bevy_render::render_resource::BindGroupLayout,
    encode_layout: bevy_render::render_resource::BindGroupLayout,
    infer_layout: bevy_render::render_resource::BindGroupLayout,
    learn_layout: bevy_render::render_resource::BindGroupLayout,
    learn: bevy_render::render_resource::ComputePipeline,
    pub adam: bevy_render::render_resource::ComputePipeline,
    pub encode_records: bevy_render::render_resource::ComputePipeline,
    pub query_infer: bevy_render::render_resource::ComputePipeline,
    /// `VK_NV_cooperative_vector` fn table + the raw device — the per-layer
    /// TrainingOptimal→RowMajor dW conversion is a raw device command.
    coopvec_fns: ash::nv::cooperative_vector::Device,
    raw_device: ash::Device,
    /// Bytes per layer's TrainingOptimal dW block (host convert size query).
    pub opt_size: u32,
}

/// All NRC GPU state. Weights/optimizer are global (one cache per app);
/// records rotate through a fixed-capacity ring written by raygen.
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
    groups: Option<NrcBindGroups>,
    pub step: u32,
    /// Non-blocking loss readback: 0 = idle, 1 = copied (map next frame),
    /// 2 = map requested (read when the shared flag flips).
    loss_state: u32,
    loss_mapped: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

struct NrcBindGroups {
    learn: BindGroup,
    adam_w: BindGroup,
    adam_b: BindGroup,
    encode: BindGroup,
}

/// `RenderStartup`: explicit layouts + eager passthrough pipelines. Every
/// kernel is Slang-precompiled SPIR-V (no naga reflection), so each bind
/// group layout is spelled out to match the `[[vk::binding]]` tables in
/// nrc_train.slang / nrc_mlp.slang.
#[allow(unsafe_code)]
pub fn init_nrc_pipelines(mut commands: Commands, render_device: Res<RenderDevice>) {
    let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    // Passthrough SPIR-V bindings must be contiguous from 0 per kernel:
    // wgpu-hal's Vulkan backend numbers descriptor bindings sequentially by
    // layout-entry order, ignoring sparse wgpu binding numbers — a sparse
    // table silently desyncs the blob's [[vk::binding]] slots.
    let adam_layout = render_device.create_bind_group_layout(
        "nrc_adam_layout",
        &[
            storage_entry(0, false), // master
            storage_entry(1, true),  // grad_in
            storage_entry(2, false), // moment_m
            storage_entry(3, false), // moment_v
            storage_entry(4, false), // mirror_f16
            storage_entry(5, false), // mirror_alt
            storage_entry(6, false), // ema_master
            storage_entry(7, false), // mirror_ema
            uniform_entry(8),        // adam_u
        ],
    );
    let infer_layout = render_device.create_bind_group_layout(
        "nrc_infer_layout",
        &[
            storage_entry(0, true),  // weights_t
            storage_entry(1, true),  // biases
            storage_entry(2, true),  // queries
            storage_entry(3, false), // out_radiance
            uniform_entry(4),        // qparams
        ],
    );
    let encode_layout = render_device.create_bind_group_layout(
        "nrc_encode_layout",
        &[
            storage_entry(0, true),  // records
            storage_entry(1, false), // act_out
            storage_entry(2, false), // targets_out
            uniform_entry(3),        // train_params
        ],
    );
    let learn_layout = render_device.create_bind_group_layout(
        "nrc_learn_layout",
        &[
            storage_entry(0, true),  // acts (encoded batch)
            storage_entry(1, true),  // weights_t (live mirror)
            storage_entry(2, true),  // bias16 (live mirror)
            storage_entry(3, true),  // targets
            storage_entry(4, false), // preds
            storage_entry(5, false), // loss
            storage_entry(6, false), // dw_opt (TrainingOptimal accumulate)
            storage_entry(7, false), // db (f32 accumulate)
            uniform_entry(8),        // params
            storage_entry(9, true),  // zeros
        ],
    );

    // SAFETY (all blobs): compiled from the in-tree .slang sources and
    // spirv-val-validated (regen commands in their headers); each explicit
    // layout above matches its kernel's binding table.
    let make = |label: &'static str,
                spv_bytes: &'static [u8],
                layout: &bevy_render::render_resource::BindGroupLayout| {
        let spv = wgpu::util::make_spirv_raw(spv_bytes);
        let module = unsafe {
            render_device.wgpu_device().create_shader_module_passthrough(
                wgpu::ShaderModuleDescriptorPassthrough {
                    spirv: Some(spv),
                    ..Default::default()
                },
            )
        };
        let pl = render_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        });
        render_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    };

    let learn = make("nrc_learn", include_bytes!("nrc_train.spv"), &learn_layout);
    let adam = make("nrc_adam", include_bytes!("nrc_adam.spv"), &adam_layout);
    let encode_records = make(
        "nrc_encode_records",
        include_bytes!("nrc_encode_records.spv"),
        &encode_layout,
    );
    let query_infer = make(
        "nrc_query_infer",
        include_bytes!("nrc_query_infer.spv"),
        &infer_layout,
    );

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
        (fns, hal_device.raw_device().clone(), dst_size as u32)
    };

    commands.insert_resource(NrcPipelines {
        adam_layout,
        encode_layout,
        infer_layout,
        learn_layout,
        learn,
        adam,
        encode_records,
        query_infer,
        coopvec_fns,
        raw_device,
        opt_size,
    });
}

/// `Render::Prepare`: create the buffers once the [`Allocator`] exists.
pub fn init_nrc_buffers(
    mut commands: Commands,
    existing: Option<Res<NrcBuffers>>,
    allocator: Option<Res<Allocator>>,
    pipelines: Option<Res<NrcPipelines>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    settings: Res<crate::SolariSettings>,
) {
    if existing.is_some() {
        return;
    }
    let Some(allocator) = allocator else { return };
    let Some(pipelines) = pipelines else { return };

    let batch = NRC_RECORD_CAP as u64;
    let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
    let uniform = BufferUsages::UNIFORM | BufferUsages::COPY_DST;
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
        acts: mk("nrc_acts", batch * width * 2, storage),
        preds: mk("nrc_preds", batch * width * 4, storage),
        targets: mk("nrc_targets", batch * 16, storage),
        loss: mk("nrc_loss", batch * 4, storage),
        loss_staging: mk(
            "nrc_loss_staging",
            // 4 loss bytes + 16 target bytes per record.
            batch * LOSS_STAGING_BYTES_PER_RECORD,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        ),
        zeros: mk("nrc_zeros", 1024, storage),
        master_w,
        master_b: mk("nrc_master_b", (NRC_B_TOTAL * 4) as u64, storage),
        dw,
        db: mk("nrc_db", (NRC_B_TOTAL * 4) as u64, storage),
        m_w: mk("nrc_m_w", (NRC_W_TOTAL * 4) as u64, storage),
        v_w: mk("nrc_v_w", (NRC_W_TOTAL * 4) as u64, storage),
        m_b: mk("nrc_m_b", (NRC_B_TOTAL * 4) as u64, storage),
        v_b: mk("nrc_v_b", (NRC_B_TOTAL * 4) as u64, storage),
        ema_w,
        ema_b: mk("nrc_ema_b", (NRC_B_TOTAL * 4) as u64, storage),
        weights_t_ema,
        bias16_ema,
        dw_opt,
        dw_opt_addr,
        dw_addr,
        learn_ubo: mk_ubo("nrc_learn_ubo", 16),
        adam_w_ubo: mk_ubo("nrc_adam_w_ubo", 48),
        adam_b_ubo: mk_ubo("nrc_adam_b_ubo", 48),
        train_ubo: mk_ubo("nrc_train_ubo", 16),
        query_ubo: mk_ubo("nrc_query_ubo", 16),
        groups: None,
        step: 0,
        loss_state: 0,
        loss_mapped: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    });
}

fn build_bind_groups(
    bufs: &NrcBuffers,
    pipelines: &NrcPipelines,
    device: &RenderDevice,
) -> NrcBindGroups {
    NrcBindGroups {
        learn: device.create_bind_group(
            "nrc_learn",
            &pipelines.learn_layout,
            &BindGroupEntries::with_indices((
                (0, bufs.acts.as_entire_binding()),
                (1, bufs.weights_t.as_entire_binding()),
                (2, bufs.bias16.as_entire_binding()),
                (3, bufs.targets.as_entire_binding()),
                (4, bufs.preds.as_entire_binding()),
                (5, bufs.loss.as_entire_binding()),
                (6, bufs.dw_opt.as_entire_binding()),
                (7, bufs.db.as_entire_binding()),
                (8, bufs.learn_ubo.as_entire_binding()),
                // f32 zeros reinterpreted: all-zero bytes are all-zero f16s
                (9, bufs.zeros.as_entire_binding()),
            )),
        ),
        adam_w: device.create_bind_group(
            "nrc_adam_w",
            &pipelines.adam_layout,
            &BindGroupEntries::with_indices((
                (0, bufs.master_w.as_entire_binding()),
                (1, bufs.dw.as_entire_binding()),
                (2, bufs.m_w.as_entire_binding()),
                (3, bufs.v_w.as_entire_binding()),
                (4, bufs.w16.as_entire_binding()),
                (5, bufs.weights_t.as_entire_binding()),
                (6, bufs.ema_w.as_entire_binding()),
                (7, bufs.weights_t_ema.as_entire_binding()),
                (8, bufs.adam_w_ubo.as_entire_binding()),
            )),
        ),
        adam_b: device.create_bind_group(
            "nrc_adam_b",
            &pipelines.adam_layout,
            &BindGroupEntries::with_indices((
                (0, bufs.master_b.as_entire_binding()),
                (1, bufs.db.as_entire_binding()),
                (2, bufs.m_b.as_entire_binding()),
                (3, bufs.v_b.as_entire_binding()),
                (4, bufs.bias16.as_entire_binding()),
                (5, bufs.weights_t.as_entire_binding()),
                (6, bufs.ema_b.as_entire_binding()),
                (7, bufs.bias16_ema.as_entire_binding()),
                (8, bufs.adam_b_ubo.as_entire_binding()),
            )),
        ),
        encode: device.create_bind_group(
            "nrc_encode",
            &pipelines.encode_layout,
            &BindGroupEntries::with_indices((
                (0, bufs.records.as_entire_binding()),
                (1, bufs.acts.as_entire_binding()),
                (2, bufs.targets.as_entire_binding()),
                (3, bufs.train_ubo.as_entire_binding()),
            )),
        ),
    }
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
/// the raygen accumulation blend). Recorded on the ctx encoder AFTER the
/// trace.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_nrc_query_infer(
    encoder: &mut wgpu::CommandEncoder,
    bufs: &NrcBuffers,
    pipelines: &NrcPipelines,
    device: &RenderDevice,
    queue: &RenderQueue,
    queries: &Buffer,
    output: &Buffer,
    cap: u32,
    scale: f32,
    exposure: f32,
) -> bool {
    let infer = &pipelines.query_infer;
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
    let bg = device.create_bind_group(
        "nrc_query_infer",
        &pipelines.infer_layout,
        &BindGroupEntries::with_indices((
            (0, bufs.weights_t_ema.as_entire_binding()),
            (1, bufs.bias16_ema.as_entire_binding()),
            (2, queries.as_entire_binding()),
            (3, output.as_entire_binding()),
            (4, bufs.query_ubo.as_entire_binding()),
        )),
    );
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("nrc_query_infer"),
        timestamp_writes: None,
    });
    pass.set_pipeline(infer);
    pass.set_bind_group(0, &bg, &[]);
    // The count lives on the GPU; over-dispatch to the cap, threads early-out.
    pass.dispatch_workgroups(cap.div_ceil(64), 1, 1);
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
    let (adam, encode) = (&pipelines.adam, &pipelines.encode_records);
    if bufs.groups.is_none() {
        bufs.groups = Some(build_bind_groups(bufs, pipelines, device));
    }

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

    let groups = bufs.groups.as_ref().unwrap();
    {
        // The gradient accumulators start at zero every step (the fused
        // kernel's outer-product/reduce-sum accumulates are additive).
        let encoder = ctx.command_encoder();
        encoder.clear_buffer(&bufs.dw_opt, 0, None);
        encoder.clear_buffer(&bufs.db, 0, None);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("nrc_training"),
            timestamp_writes: None,
        });
        pass.set_pipeline(encode);
        pass.set_bind_group(0, &groups.encode, &[]);
        pass.dispatch_workgroups(batch.div_ceil(64), 1, 1);
        // fused forward + loss + backward incl. the dW/db accumulates
        // (nrc_train.slang)
        pass.set_pipeline(&pipelines.learn);
        pass.set_bind_group(0, &groups.learn, &[]);
        pass.dispatch_workgroups(batch.div_ceil(64), 1, 1);
    }

    // Per-layer TrainingOptimal→RowMajor dW conversion: a raw device command,
    // in its OWN encoder (wgpu-core panics if one encoder mixes wgpu passes
    // with raw `as_hal_mut`); `add_command_buffer` flushes the training pass
    // above first, so on the single queue the converts run after it. wgpu
    // tracks none of this, so both sides are fenced with raw sync2 barriers.
    let mut conv_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("nrc_dw_convert"),
    });
    // SAFETY: Vulkan backend; the addresses point at live storage buffers
    // created with SHADER_DEVICE_ADDRESS (wgpu-hal adds it to all storage
    // buffers); sizes come from the host convert query at init.
    unsafe {
        conv_encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.expect("bevy_solari requires the Vulkan backend");
            let cb = hal_encoder.raw_handle();
            let dev = &pipelines.raw_device;
            let barrier = |src_stage, src_access, dst_stage, dst_access| {
                vk::MemoryBarrier2::default()
                    .src_stage_mask(src_stage)
                    .src_access_mask(src_access)
                    .dst_stage_mask(dst_stage)
                    .dst_access_mask(dst_access)
            };
            let pre = [barrier(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::CONVERT_COOPERATIVE_VECTOR_MATRIX_NV,
                vk::AccessFlags2::TRANSFER_READ,
            )];
            dev.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().memory_barriers(&pre),
            );
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
            let post = [barrier(
                vk::PipelineStageFlags2::CONVERT_COOPERATIVE_VECTOR_MATRIX_NV,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            )];
            dev.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().memory_barriers(&post),
            );
        });
    }
    ctx.add_command_buffer(conv_encoder.finish());

    {
        let encoder = ctx.command_encoder();
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("nrc_adam"),
            timestamp_writes: None,
        });
        pass.set_pipeline(adam);
        pass.set_bind_group(0, &groups.adam_w, &[]);
        pass.dispatch_workgroups((NRC_W_TOTAL as u32).div_ceil(64), 1, 1);
        pass.set_bind_group(0, &groups.adam_b, &[]);
        pass.dispatch_workgroups((NRC_B_TOTAL as u32).div_ceil(64), 1, 1);
    }
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
