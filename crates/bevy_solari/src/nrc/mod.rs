//! Neural Radiance Cache: a small MLP trained online to predict cached
//! radiance at GI path terminations.
//!
//! Training runs on the render graph as a hybrid chain (encode records →
//! fused forward+loss+dZ → dW/db reductions → adam, dispatched after the
//! trace each frame); raygen writes one training record per rotating pixel
//! subset. The fused kernel is Slang-compiled SPIR-V (`nrc_train.slang`,
//! passthrough-loaded): one thread evaluates the whole coopvec MLP, seeds
//! the loss gradient, and back-propagates the per-layer dZ that the coopmat
//! `mm_tn`/`bias_grad` kernels reduce into dW/db. GI paths terminate into
//! the cache by appending queries that the `nrc_query_infer` pass
//! batch-evaluates (coherent coopvec against the transposed f16 weight
//! mirror adam maintains) and composites into the output buffer; training
//! paths and [`SolariNrc::inline_coopvec`] query inline in raygen instead.

use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_ecs::prelude::*;
use bevy_render::{
    extract_resource::ExtractResource,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        Buffer, BufferUsages, CachedComputePipelineId, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineCache, ShaderStages,
    },
    renderer::{RenderDevice, RenderQueue},
};
use half::f16;

use crate::gpu::allocator::{Allocator, MemoryLocation};
use ash::vk;

pub const NRC_WIDTH: usize = 64;
pub const NRC_LAYERS: usize = 6;
pub const NRC_W_ELEMS: usize = NRC_WIDTH * NRC_WIDTH;
pub const NRC_W_TOTAL: usize = NRC_LAYERS * NRC_W_ELEMS;
pub const NRC_B_TOTAL: usize = NRC_LAYERS * NRC_WIDTH;
/// Training records per frame; must match `NRC_RECORD_CAP` in raygen.wgsl.
pub const NRC_RECORD_CAP: usize = 16384;
/// Bytes per `NrcRecord` (5 × vec4<f32>), must match raygen.wgsl/nrc_mlp.wgsl.
pub const NRC_RECORD_SIZE: usize = 80;
/// Bytes per termination query (3 × vec4<u32>); the per-view ring is sized to
/// the viewport (one query per pixel per frame) plus a 16-byte count header.
/// Must match `NrcQueryGpu`/`NrcQueryBuf` in raygen.wgsl and nrc_mlp.wgsl.
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
struct MmDims {
    m: u32,
    n: u32,
    k: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    pad_a: u32,
    pad_b: u32,
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
    pad_a: u32,
    pad_b: u32,
}

/// Bind-group layouts + queued pipelines for the training kernels. The
/// fused `learn` kernel is passthrough SPIR-V with an explicit layout, so
/// it is created eagerly instead of queued on the [`PipelineCache`].
#[derive(Resource)]
pub struct NrcPipelines {
    mm_layout: BindGroupLayoutDescriptor,
    bias_grad_layout: BindGroupLayoutDescriptor,
    adam_layout: BindGroupLayoutDescriptor,
    encode_layout: BindGroupLayoutDescriptor,
    infer_layout: BindGroupLayoutDescriptor,
    learn_layout: bevy_render::render_resource::BindGroupLayout,
    learn: bevy_render::render_resource::ComputePipeline,
    pub mm_tn: CachedComputePipelineId,
    pub bias_grad: CachedComputePipelineId,
    pub adam: CachedComputePipelineId,
    pub encode_records: CachedComputePipelineId,
    pub query_infer: CachedComputePipelineId,
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
    /// f16 `[LAYERS][batch][WIDTH]` layer-input activations: `[0]` is the
    /// encoded batch, `[1..]` are recorded by the fused kernel for `mm_tn`.
    acts: Buffer,
    /// f16 `[LAYERS][batch][WIDTH]` dZ recorded by the fused kernel.
    dz: Buffer,
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
    dw_ubo: Vec<Buffer>,
    bg_ubo: Vec<Buffer>,
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
    dw: Vec<BindGroup>,
    bias_grad: Vec<BindGroup>,
    adam_w: BindGroup,
    adam_b: BindGroup,
    encode: BindGroup,
}

/// `RenderStartup`: layouts + pipeline queueing (no scene deps — queue now).
#[allow(unsafe_code)]
pub fn init_nrc_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
    render_device: Res<RenderDevice>,
    mut registry: ResMut<crate::ecs_gpu::SolariPipelineRegistry>,
) {
    let mm_layout = BindGroupLayoutDescriptor::new(
        "nrc_mm_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, storage_buffer_read_only_sized(false, None)),
                (1, storage_buffer_read_only_sized(false, None)),
                (2, storage_buffer_sized(false, None)),
                (3, storage_buffer_read_only_sized(false, None)),
                (4, uniform_buffer_sized(false, None)),
            ),
        ),
    );
    let bias_grad_layout = BindGroupLayoutDescriptor::new(
        "nrc_bias_grad_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (2, storage_buffer_sized(false, None)),
                (4, uniform_buffer_sized(false, None)),
                (12, storage_buffer_read_only_sized(false, None)),
            ),
        ),
    );
    let adam_layout = BindGroupLayoutDescriptor::new(
        "nrc_adam_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (13, storage_buffer_sized(false, None)),
                (14, storage_buffer_read_only_sized(false, None)),
                (15, storage_buffer_sized(false, None)),
                (16, storage_buffer_sized(false, None)),
                (17, storage_buffer_sized(false, None)),
                (18, uniform_buffer_sized(false, None)),
                (21, storage_buffer_sized(false, None)),
                (23, storage_buffer_sized(false, None)),
                (24, storage_buffer_sized(false, None)),
            ),
        ),
    );
    let infer_layout = BindGroupLayoutDescriptor::new(
        "nrc_infer_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (1, storage_buffer_read_only_sized(false, None)),  // weights_t (mat_b)
                (12, storage_buffer_read_only_sized(false, None)), // bias16 (act_mask)
                (26, storage_buffer_read_only_sized(false, None)), // queries
                (27, storage_buffer_sized(false, None)),           // out_radiance
                (28, uniform_buffer_sized(false, None)),           // qparams
            ),
        ),
    );
    let encode_layout = BindGroupLayoutDescriptor::new(
        "nrc_encode_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (6, storage_buffer_sized(false, None)),
                (19, uniform_buffer_sized(false, None)),
                (20, storage_buffer_sized(false, None)),
                (22, storage_buffer_read_only_sized(false, None)),
            ),
        ),
    );

    let shader = load_embedded_asset!(asset_server.as_ref(), "nrc_mlp.wgsl");
    let queue = |label: &'static str, entry: &'static str, layout: &BindGroupLayoutDescriptor| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(entry.into()),
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        })
    };
    let mm_tn = queue("nrc_mm_tn", "mm_tn", &mm_layout);
    let bias_grad = queue("nrc_bias_grad", "bias_grad", &bias_grad_layout);
    let adam = queue("nrc_adam", "adam", &adam_layout);
    let encode_records = queue("nrc_encode_records", "nrc_encode_records", &encode_layout);
    let query_infer = queue("nrc_query_infer", "nrc_query_infer", &infer_layout);
    for (name, id) in [
        ("nrc_mm_tn", mm_tn),
        ("nrc_bias_grad", bias_grad),
        ("nrc_adam", adam),
        ("nrc_encode_records", encode_records),
        ("nrc_query_infer", query_infer),
    ] {
        registry.register(name, id);
    }

    // The fused forward+loss+dZ kernel: Slang-compiled SPIR-V loaded through
    // the passthrough path (no naga reflection), so the bind group layout is
    // spelled out to match the [[vk::binding]] table in nrc_train.slang.
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
    let learn_layout = render_device.create_bind_group_layout(
        "nrc_learn_layout",
        &[
            storage_entry(0, false), // acts
            storage_entry(1, true),  // weights_t (live mirror)
            storage_entry(2, true),  // bias16 (live mirror)
            storage_entry(3, true),  // targets
            storage_entry(4, false), // preds
            storage_entry(5, false), // loss
            storage_entry(6, false), // dz
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_entry(8, true), // zeros (f16 view)
        ],
    );
    let spv = wgpu::util::make_spirv_raw(include_bytes!("nrc_train.spv"));
    // SAFETY: nrc_train.spv is compiled from nrc_train.slang in-tree and
    // spirv-val-validated (regen command in its header); the explicit layout
    // above matches its binding table.
    let learn_module = unsafe {
        render_device.wgpu_device().create_shader_module_passthrough(
            wgpu::ShaderModuleDescriptorPassthrough {
                spirv: Some(spv),
                ..Default::default()
            },
        )
    };
    let learn_pl = render_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("nrc_learn_pl"),
        bind_group_layouts: &[Some(&learn_layout)],
        immediate_size: 0,
    });
    let learn = render_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("nrc_learn"),
        layout: Some(&learn_pl),
        module: &learn_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    commands.insert_resource(NrcPipelines {
        mm_layout,
        bias_grad_layout,
        adam_layout,
        encode_layout,
        infer_layout,
        learn_layout,
        learn,
        mm_tn,
        bias_grad,
        adam,
        encode_records,
        query_infer,
    });
}

/// `Render::Prepare`: create the buffers once the [`Allocator`] exists.
pub fn init_nrc_buffers(
    mut commands: Commands,
    existing: Option<Res<NrcBuffers>>,
    allocator: Option<Res<Allocator>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    settings: Res<crate::SolariSettings>,
) {
    if existing.is_some() {
        return;
    }
    let Some(allocator) = allocator else { return };

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
    let layers = NRC_LAYERS as u64;
    let mk_ubo = |label: &'static str, size: u64| mk(label, size, uniform);
    let dw_ubo: Vec<Buffer> = (0..NRC_LAYERS).map(|_| mk_ubo("nrc_dw_ubo", 32)).collect();
    let bg_ubo: Vec<Buffer> = (0..NRC_LAYERS).map(|_| mk_ubo("nrc_bg_ubo", 32)).collect();

    let batch_u = NRC_RECORD_CAP as u32;
    let layer_elems = batch_u * NRC_WIDTH as u32;
    for l in 0..NRC_LAYERS {
        // dW_l = act_l^T · dZ_l: both operands live in the packed
        // [LAYERS][batch][WIDTH] buffers, selected by element offset.
        render_queue.write_buffer(
            &dw_ubo[l],
            0,
            bytemuck::bytes_of(&MmDims {
                m: NRC_WIDTH as u32,
                n: NRC_WIDTH as u32,
                k: batch_u,
                a_off: l as u32 * layer_elems,
                b_off: l as u32 * layer_elems,
                c_off: (l * NRC_W_ELEMS) as u32,
                ..bytemuck::Zeroable::zeroed()
            }),
        );
        render_queue.write_buffer(
            &bg_ubo[l],
            0,
            bytemuck::bytes_of(&MmDims {
                k: batch_u,
                c_off: (l * NRC_WIDTH) as u32,
                ..bytemuck::Zeroable::zeroed()
            }),
        );
    }

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
        acts: mk("nrc_acts", layers * batch * width * 2, storage),
        dz: mk("nrc_dz", layers * batch * width * 2, storage),
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
        dw: mk("nrc_dw", (NRC_W_TOTAL * 4) as u64, storage),
        db: mk("nrc_db", (NRC_B_TOTAL * 4) as u64, storage),
        m_w: mk("nrc_m_w", (NRC_W_TOTAL * 4) as u64, storage),
        v_w: mk("nrc_v_w", (NRC_W_TOTAL * 4) as u64, storage),
        m_b: mk("nrc_m_b", (NRC_B_TOTAL * 4) as u64, storage),
        v_b: mk("nrc_v_b", (NRC_B_TOTAL * 4) as u64, storage),
        ema_w,
        ema_b: mk("nrc_ema_b", (NRC_B_TOTAL * 4) as u64, storage),
        weights_t_ema,
        bias16_ema,
        dw_ubo,
        bg_ubo,
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
    cache: &PipelineCache,
) -> NrcBindGroups {
    let batch = NRC_RECORD_CAP as u64;
    let layer_bytes = batch * NRC_WIDTH as u64 * 2;
    let mut dw = vec![];
    let mut bias_grad = vec![];
    for l in 0..NRC_LAYERS {
        dw.push(device.create_bind_group(
            "nrc_dw",
            &cache.get_bind_group_layout(&pipelines.mm_layout),
            &BindGroupEntries::with_indices((
                (0, bufs.acts.as_entire_binding()),
                (1, bufs.dz.as_entire_binding()),
                (2, bufs.dw.as_entire_binding()),
                (3, bufs.zeros.as_entire_binding()),
                (4, bufs.dw_ubo[l].as_entire_binding()),
            )),
        ));
        // bias_grad has no element-offset field for act_mask — bind the
        // layer's dZ slice (the 2 MiB layer stride keeps any offset alignment).
        bias_grad.push(device.create_bind_group(
            "nrc_bias_grad",
            &cache.get_bind_group_layout(&pipelines.bias_grad_layout),
            &BindGroupEntries::with_indices((
                (2, bufs.db.as_entire_binding()),
                (4, bufs.bg_ubo[l].as_entire_binding()),
                (
                    12,
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &bufs.dz,
                        offset: l as u64 * layer_bytes,
                        size: Some(std::num::NonZeroU64::new(layer_bytes).unwrap()),
                    }),
                ),
            )),
        ));
    }
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
                (6, bufs.dz.as_entire_binding()),
                (7, bufs.learn_ubo.as_entire_binding()),
                // f32 zeros reinterpreted: all-zero bytes are all-zero f16s
                (8, bufs.zeros.as_entire_binding()),
            )),
        ),
        dw,
        bias_grad,
        adam_w: device.create_bind_group(
            "nrc_adam_w",
            &cache.get_bind_group_layout(&pipelines.adam_layout),
            &BindGroupEntries::with_indices((
                (13, bufs.master_w.as_entire_binding()),
                (14, bufs.dw.as_entire_binding()),
                (15, bufs.m_w.as_entire_binding()),
                (16, bufs.v_w.as_entire_binding()),
                (17, bufs.w16.as_entire_binding()),
                (18, bufs.adam_w_ubo.as_entire_binding()),
                (21, bufs.weights_t.as_entire_binding()),
                (23, bufs.ema_w.as_entire_binding()),
                (24, bufs.weights_t_ema.as_entire_binding()),
            )),
        ),
        adam_b: device.create_bind_group(
            "nrc_adam_b",
            &cache.get_bind_group_layout(&pipelines.adam_layout),
            &BindGroupEntries::with_indices((
                (13, bufs.master_b.as_entire_binding()),
                (14, bufs.db.as_entire_binding()),
                (15, bufs.m_b.as_entire_binding()),
                (16, bufs.v_b.as_entire_binding()),
                (17, bufs.bias16.as_entire_binding()),
                (18, bufs.adam_b_ubo.as_entire_binding()),
                (21, bufs.weights_t.as_entire_binding()),
                (23, bufs.ema_b.as_entire_binding()),
                (24, bufs.bias16_ema.as_entire_binding()),
            )),
        ),
        encode: device.create_bind_group(
            "nrc_encode",
            &cache.get_bind_group_layout(&pipelines.encode_layout),
            &BindGroupEntries::with_indices((
                (6, bufs.acts.as_entire_binding()),
                (19, bufs.train_ubo.as_entire_binding()),
                (20, bufs.targets.as_entire_binding()),
                (22, bufs.records.as_entire_binding()),
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
/// trace. Returns false when the pipeline isn't compiled yet.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_nrc_query_infer(
    encoder: &mut wgpu::CommandEncoder,
    bufs: &NrcBuffers,
    pipelines: &NrcPipelines,
    cache: &PipelineCache,
    device: &RenderDevice,
    queue: &RenderQueue,
    queries: &Buffer,
    output: &Buffer,
    cap: u32,
    scale: f32,
    exposure: f32,
) -> bool {
    let Some(infer) = cache.get_compute_pipeline(pipelines.query_infer) else {
        return false;
    };
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
        &cache.get_bind_group_layout(&pipelines.infer_layout),
        &BindGroupEntries::with_indices((
            (1, bufs.weights_t_ema.as_entire_binding()),
            (12, bufs.bias16_ema.as_entire_binding()),
            (26, queries.as_entire_binding()),
            (27, output.as_entire_binding()),
            (28, bufs.query_ubo.as_entire_binding()),
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
    encoder: &mut wgpu::CommandEncoder,
    bufs: &mut NrcBuffers,
    pipelines: &NrcPipelines,
    cache: &PipelineCache,
    device: &RenderDevice,
    queue: &RenderQueue,
    nrc: &SolariNrc,
) -> bool {
    let Some((mm_tn, bias_grad, adam, encode)) = (|| {
        Some((
            cache.get_compute_pipeline(pipelines.mm_tn)?,
            cache.get_compute_pipeline(pipelines.bias_grad)?,
            cache.get_compute_pipeline(pipelines.adam)?,
            cache.get_compute_pipeline(pipelines.encode_records)?,
        ))
    })() else {
        return false;
    };
    if bufs.groups.is_none() {
        bufs.groups = Some(build_bind_groups(bufs, pipelines, device, cache));
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
            pad_a: 0,
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
    let width = NRC_WIDTH as u32;
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("nrc_training"),
            timestamp_writes: None,
        });
        pass.set_pipeline(encode);
        pass.set_bind_group(0, &groups.encode, &[]);
        pass.dispatch_workgroups(batch.div_ceil(64), 1, 1);
        // fused forward + loss + dZ chain (nrc_train.slang)
        pass.set_pipeline(&pipelines.learn);
        pass.set_bind_group(0, &groups.learn, &[]);
        pass.dispatch_workgroups(batch.div_ceil(64), 1, 1);
        // dW/db reductions over the recorded activations/dZ
        for l in 0..NRC_LAYERS {
            pass.set_pipeline(mm_tn);
            pass.set_bind_group(0, &groups.dw[l], &[]);
            pass.dispatch_workgroups(width / 16, width / 16, 1);
            pass.set_pipeline(bias_grad);
            pass.set_bind_group(0, &groups.bias_grad[l], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        pass.set_pipeline(adam);
        pass.set_bind_group(0, &groups.adam_w, &[]);
        pass.dispatch_workgroups((NRC_W_TOTAL as u32).div_ceil(64), 1, 1);
        pass.set_bind_group(0, &groups.adam_b, &[]);
        pass.dispatch_workgroups((NRC_B_TOTAL as u32).div_ceil(64), 1, 1);
    }
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
