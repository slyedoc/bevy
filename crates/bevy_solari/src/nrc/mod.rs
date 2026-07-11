//! Neural Radiance Cache — zero/docs/nrc.md.
//!
//! Training transplants the gym-certified coopmat pipeline into the render
//! graph (encode records → fwd → loss → bwd → adam, dispatched after the
//! trace each frame); raygen writes one training record per rotating pixel
//! subset. GI paths terminate into the cache by appending queries that the
//! `nrc_query_infer` pass batch-evaluates (coherent coopvec against the
//! transposed f16 weight mirror adam maintains) and composites into the
//! output buffer; training paths and [`SolariNrc::inline_coopvec`] query
//! inline in raygen instead.

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
struct EwParams {
    batch: u32,
    layer_off: u32,
    relu: u32,
    loss_scale: f32,
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

/// Bind-group layouts + queued pipelines for the training kernels.
#[derive(Resource)]
pub struct NrcPipelines {
    mm_layout: BindGroupLayoutDescriptor,
    bias_layout: BindGroupLayoutDescriptor,
    loss_layout: BindGroupLayoutDescriptor,
    relu_layout: BindGroupLayoutDescriptor,
    bias_grad_layout: BindGroupLayoutDescriptor,
    adam_layout: BindGroupLayoutDescriptor,
    encode_layout: BindGroupLayoutDescriptor,
    infer_layout: BindGroupLayoutDescriptor,
    pub mm_nn: CachedComputePipelineId,
    pub mm_nt: CachedComputePipelineId,
    pub mm_tn: CachedComputePipelineId,
    pub bias_act: CachedComputePipelineId,
    pub loss_grad: CachedComputePipelineId,
    pub relu_bwd: CachedComputePipelineId,
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
    act: Vec<Buffer>,
    pre: Buffer,
    dz_a: Buffer,
    dz_b: Buffer,
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
    fwd_ubo: Vec<Buffer>,
    dw_ubo: Vec<Buffer>,
    dx_ubo: Vec<Buffer>,
    bg_ubo: Vec<Buffer>,
    bias_ubo: Vec<Buffer>,
    loss_ubo: Buffer,
    relu_ubo: Buffer,
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
    fwd: Vec<BindGroup>,
    bias: Vec<BindGroup>,
    loss: BindGroup,
    dw: Vec<BindGroup>,
    dx: Vec<BindGroup>,
    relu: Vec<BindGroup>,
    bias_grad: Vec<BindGroup>,
    adam_w: BindGroup,
    adam_b: BindGroup,
    encode: BindGroup,
}

/// `RenderStartup`: layouts + pipeline queueing (no scene deps — queue now).
pub fn init_nrc_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
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
    let bias_layout = BindGroupLayoutDescriptor::new(
        "nrc_bias_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (5, storage_buffer_sized(false, None)),
                (6, storage_buffer_sized(false, None)),
                (7, storage_buffer_read_only_sized(false, None)),
                (8, uniform_buffer_sized(false, None)),
            ),
        ),
    );
    let loss_layout = BindGroupLayoutDescriptor::new(
        "nrc_loss_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (5, storage_buffer_sized(false, None)),
                (8, uniform_buffer_sized(false, None)),
                (9, storage_buffer_read_only_sized(false, None)),
                (10, storage_buffer_sized(false, None)),
                (11, storage_buffer_sized(false, None)),
            ),
        ),
    );
    let relu_layout = BindGroupLayoutDescriptor::new(
        "nrc_relu_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (5, storage_buffer_sized(false, None)),
                (6, storage_buffer_sized(false, None)),
                (8, uniform_buffer_sized(false, None)),
                (12, storage_buffer_read_only_sized(false, None)),
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
    let mm_nn = queue("nrc_mm_nn", "mm_nn", &mm_layout);
    let mm_nt = queue("nrc_mm_nt", "mm_nt", &mm_layout);
    let mm_tn = queue("nrc_mm_tn", "mm_tn", &mm_layout);
    let bias_act = queue("nrc_bias_act", "bias_act", &bias_layout);
    let loss_grad = queue("nrc_loss_grad", "loss_grad", &loss_layout);
    let relu_bwd = queue("nrc_relu_bwd", "relu_bwd", &relu_layout);
    let bias_grad = queue("nrc_bias_grad", "bias_grad", &bias_grad_layout);
    let adam = queue("nrc_adam", "adam", &adam_layout);
    let encode_records = queue("nrc_encode_records", "nrc_encode_records", &encode_layout);
    let query_infer = queue("nrc_query_infer", "nrc_query_infer", &infer_layout);
    for (name, id) in [
        ("nrc_mm_nn", mm_nn),
        ("nrc_mm_nt", mm_nt),
        ("nrc_mm_tn", mm_tn),
        ("nrc_bias_act", bias_act),
        ("nrc_loss_grad", loss_grad),
        ("nrc_relu_bwd", relu_bwd),
        ("nrc_bias_grad", bias_grad),
        ("nrc_adam", adam),
        ("nrc_encode_records", encode_records),
        ("nrc_query_infer", query_infer),
    ] {
        registry.register(name, id);
    }
    commands.insert_resource(NrcPipelines {
        mm_layout,
        bias_layout,
        loss_layout,
        relu_layout,
        bias_grad_layout,
        adam_layout,
        encode_layout,
        infer_layout,
        mm_nn,
        mm_nt,
        mm_tn,
        bias_act,
        loss_grad,
        relu_bwd,
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

    // deterministic He-uniform init, mirrored three ways. `SOLARI_NRC_SEED`
    // overrides (A/B: does a trained-cache artifact follow the init?).
    let mut state: u64 = std::env::var("SOLARI_NRC_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x9e3779b97f4a7c15);
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
    let act = (0..NRC_LAYERS)
        .map(|_| mk("nrc_act", batch * width * 2, storage))
        .collect();
    let mk_ubo = |label: &'static str, size: u64| mk(label, size, uniform);
    let fwd_ubo: Vec<Buffer> = (0..NRC_LAYERS).map(|_| mk_ubo("nrc_fwd_ubo", 32)).collect();
    let dw_ubo: Vec<Buffer> = (0..NRC_LAYERS).map(|_| mk_ubo("nrc_dw_ubo", 32)).collect();
    let dx_ubo: Vec<Buffer> = (1..NRC_LAYERS).map(|_| mk_ubo("nrc_dx_ubo", 32)).collect();
    let bg_ubo: Vec<Buffer> = (0..NRC_LAYERS).map(|_| mk_ubo("nrc_bg_ubo", 32)).collect();
    let bias_ubo: Vec<Buffer> = (0..NRC_LAYERS).map(|_| mk_ubo("nrc_bias_ubo", 16)).collect();

    let batch_u = NRC_RECORD_CAP as u32;
    for l in 0..NRC_LAYERS {
        render_queue.write_buffer(
            &fwd_ubo[l],
            0,
            bytemuck::bytes_of(&MmDims {
                m: batch_u,
                n: NRC_WIDTH as u32,
                k: NRC_WIDTH as u32,
                b_off: (l * NRC_W_ELEMS) as u32,
                ..bytemuck::Zeroable::zeroed()
            }),
        );
        render_queue.write_buffer(
            &dw_ubo[l],
            0,
            bytemuck::bytes_of(&MmDims {
                m: NRC_WIDTH as u32,
                n: NRC_WIDTH as u32,
                k: batch_u,
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
        render_queue.write_buffer(
            &bias_ubo[l],
            0,
            bytemuck::bytes_of(&EwParams {
                batch: batch_u,
                layer_off: (l * NRC_WIDTH) as u32,
                relu: if l < NRC_LAYERS - 1 { 1 } else { 0 },
                loss_scale: 0.0,
            }),
        );
        if l >= 1 {
            render_queue.write_buffer(
                &dx_ubo[l - 1],
                0,
                bytemuck::bytes_of(&MmDims {
                    m: batch_u,
                    n: NRC_WIDTH as u32,
                    k: NRC_WIDTH as u32,
                    b_off: (l * NRC_W_ELEMS) as u32,
                    ..bytemuck::Zeroable::zeroed()
                }),
            );
        }
    }
    let relu_ubo = mk_ubo("nrc_relu_ubo", 16);
    render_queue.write_buffer(
        &relu_ubo,
        0,
        bytemuck::bytes_of(&EwParams {
            batch: batch_u,
            ..bytemuck::Zeroable::zeroed()
        }),
    );

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
        act,
        pre: mk("nrc_pre", batch * width * 4, storage),
        dz_a: mk("nrc_dz_a", batch * width * 2, storage),
        dz_b: mk("nrc_dz_b", batch * width * 2, storage),
        targets: mk("nrc_targets", batch * 16, storage),
        loss: mk("nrc_loss", batch * 4, storage),
        loss_staging: mk(
            "nrc_loss_staging",
            batch * 20,
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
        fwd_ubo,
        dw_ubo,
        dx_ubo,
        bg_ubo,
        bias_ubo,
        loss_ubo: mk_ubo("nrc_loss_ubo", 16),
        relu_ubo,
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

fn dz_cur(bufs: &NrcBuffers, layer: usize) -> &Buffer {
    if layer % 2 == 1 { &bufs.dz_a } else { &bufs.dz_b }
}
fn dz_next(bufs: &NrcBuffers, layer: usize) -> &Buffer {
    if layer % 2 == 1 { &bufs.dz_b } else { &bufs.dz_a }
}

fn build_bind_groups(
    bufs: &NrcBuffers,
    pipelines: &NrcPipelines,
    device: &RenderDevice,
    cache: &PipelineCache,
) -> NrcBindGroups {
    let mut fwd = vec![];
    let mut bias = vec![];
    let mut dw = vec![];
    let mut dx = vec![];
    let mut relu = vec![];
    let mut bias_grad = vec![];
    for l in 0..NRC_LAYERS {
        fwd.push(device.create_bind_group(
            "nrc_fwd",
            &cache.get_bind_group_layout(&pipelines.mm_layout),
            &BindGroupEntries::with_indices((
                (0, bufs.act[l].as_entire_binding()),
                (1, bufs.w16.as_entire_binding()),
                (2, bufs.pre.as_entire_binding()),
                (3, bufs.zeros.as_entire_binding()),
                (4, bufs.fwd_ubo[l].as_entire_binding()),
            )),
        ));
        // output layer's f16 activation write is dead — point it at dz scratch
        let act_target = if l < NRC_LAYERS - 1 { &bufs.act[l + 1] } else { &bufs.dz_b };
        bias.push(device.create_bind_group(
            "nrc_bias",
            &cache.get_bind_group_layout(&pipelines.bias_layout),
            &BindGroupEntries::with_indices((
                (5, bufs.pre.as_entire_binding()),
                (6, act_target.as_entire_binding()),
                (7, bufs.master_b.as_entire_binding()),
                (8, bufs.bias_ubo[l].as_entire_binding()),
            )),
        ));
        dw.push(device.create_bind_group(
            "nrc_dw",
            &cache.get_bind_group_layout(&pipelines.mm_layout),
            &BindGroupEntries::with_indices((
                (0, bufs.act[l].as_entire_binding()),
                (1, dz_cur(bufs, l).as_entire_binding()),
                (2, bufs.dw.as_entire_binding()),
                (3, bufs.zeros.as_entire_binding()),
                (4, bufs.dw_ubo[l].as_entire_binding()),
            )),
        ));
        bias_grad.push(device.create_bind_group(
            "nrc_bias_grad",
            &cache.get_bind_group_layout(&pipelines.bias_grad_layout),
            &BindGroupEntries::with_indices((
                (2, bufs.db.as_entire_binding()),
                (4, bufs.bg_ubo[l].as_entire_binding()),
                (12, dz_cur(bufs, l).as_entire_binding()),
            )),
        ));
        if l >= 1 {
            dx.push(device.create_bind_group(
                "nrc_dx",
                &cache.get_bind_group_layout(&pipelines.mm_layout),
                &BindGroupEntries::with_indices((
                    (0, dz_cur(bufs, l).as_entire_binding()),
                    (1, bufs.w16.as_entire_binding()),
                    (2, bufs.pre.as_entire_binding()),
                    (3, bufs.zeros.as_entire_binding()),
                    (4, bufs.dx_ubo[l - 1].as_entire_binding()),
                )),
            ));
            relu.push(device.create_bind_group(
                "nrc_relu",
                &cache.get_bind_group_layout(&pipelines.relu_layout),
                &BindGroupEntries::with_indices((
                    (5, bufs.pre.as_entire_binding()),
                    (6, dz_next(bufs, l).as_entire_binding()),
                    (8, bufs.relu_ubo.as_entire_binding()),
                    (12, bufs.act[l].as_entire_binding()),
                )),
            ));
        }
    }
    NrcBindGroups {
        fwd,
        bias,
        loss: device.create_bind_group(
            "nrc_loss",
            &cache.get_bind_group_layout(&pipelines.loss_layout),
            &BindGroupEntries::with_indices((
                (5, bufs.pre.as_entire_binding()),
                (8, bufs.loss_ubo.as_entire_binding()),
                (9, bufs.targets.as_entire_binding()),
                (10, bufs.loss.as_entire_binding()),
                (11, bufs.dz_a.as_entire_binding()),
            )),
        ),
        dw,
        dx,
        relu,
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
                (6, bufs.act[0].as_entire_binding()),
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
    let Some((mm_nn, mm_nt, mm_tn, bias_act, loss_grad, relu_bwd, bias_grad, adam, encode)) = (|| {
        Some((
            cache.get_compute_pipeline(pipelines.mm_nn)?,
            cache.get_compute_pipeline(pipelines.mm_nt)?,
            cache.get_compute_pipeline(pipelines.mm_tn)?,
            cache.get_compute_pipeline(pipelines.bias_act)?,
            cache.get_compute_pipeline(pipelines.loss_grad)?,
            cache.get_compute_pipeline(pipelines.relu_bwd)?,
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
        &bufs.loss_ubo,
        0,
        bytemuck::bytes_of(&EwParams {
            batch,
            layer_off: 0,
            relu: 0,
            loss_scale: nrc.loss_scale,
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
        for l in 0..NRC_LAYERS {
            pass.set_pipeline(mm_nn);
            pass.set_bind_group(0, &groups.fwd[l], &[]);
            pass.dispatch_workgroups(batch / 16, width / 16, 1);
            pass.set_pipeline(bias_act);
            pass.set_bind_group(0, &groups.bias[l], &[]);
            pass.dispatch_workgroups(batch, 1, 1);
        }
        pass.set_pipeline(loss_grad);
        pass.set_bind_group(0, &groups.loss, &[]);
        pass.dispatch_workgroups(batch.div_ceil(64), 1, 1);
        for l in (0..NRC_LAYERS).rev() {
            pass.set_pipeline(mm_tn);
            pass.set_bind_group(0, &groups.dw[l], &[]);
            pass.dispatch_workgroups(width / 16, width / 16, 1);
            pass.set_pipeline(bias_grad);
            pass.set_bind_group(0, &groups.bias_grad[l], &[]);
            pass.dispatch_workgroups(1, 1, 1);
            if l >= 1 {
                pass.set_pipeline(mm_nt);
                pass.set_bind_group(0, &groups.dx[l - 1], &[]);
                pass.dispatch_workgroups(batch / 16, width / 16, 1);
                pass.set_pipeline(relu_bwd);
                pass.set_bind_group(0, &groups.relu[l - 1], &[]);
                pass.dispatch_workgroups(batch, 1, 1);
            }
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
/// poll here races the raw-VK trace machinery (device-lost class, learned
/// the hard way). Copy → map_async next frame → read once the callback fires.
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
                let nan_targets = targets.iter().filter(|t| !t.is_finite()).count();
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
                let _ = nan_targets;
            }
            bufs.loss_staging.unmap();
            bufs.loss_state = 0;
        }
        _ => {}
    }
}
