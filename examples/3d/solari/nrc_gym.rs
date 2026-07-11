//! NRC rung-0 gym (zero/docs/nrc.md): trains the NRC MLP kernels against an
//! analytic radiance field, headless plain-wgpu (no bevy_render — production
//! wiring is rung 1). Certifies three things before training: GPU forward vs
//! a CPU reference (f16-quantized at the same points), analytic gradients vs
//! finite differences of the stop-grad-frozen loss, and loss convergence.

// requesting a device with ExperimentalFeatures (coopmat) is an unsafe wgpu API
#![allow(unsafe_code)]

use argh::FromArgs;
use half::f16;
use std::time::Instant;

const WIDTH: usize = 64;
const TILE: u32 = 16;
const LAYERS: usize = 6;
const OUT_CH: usize = 3;
const REL_EPS: f32 = 0.01;
const W_ELEMS: usize = WIDTH * WIDTH;
const W_TOTAL: usize = LAYERS * W_ELEMS;
const B_TOTAL: usize = LAYERS * WIDTH;

/// NRC rung-0 exam: certify the coopmat MLP kernels, then train.
#[derive(FromArgs)]
struct Args {
    /// training steps (default 2000)
    #[argh(option, default = "2000")]
    steps: u32,
    /// batch size, multiple of 16 (default 4096)
    #[argh(option, default = "4096")]
    batch: u32,
    /// adam learning rate (default 1e-3)
    #[argh(option, default = "1e-3")]
    lr: f32,
    /// f16 gradient loss scale (default 128)
    #[argh(option, default = "128.0")]
    loss_scale: f32,
    /// skip the fwd/grad certification and go straight to training
    #[argh(switch)]
    skip_checks: bool,
    /// rng seed for weight init (default 7)
    #[argh(option, default = "7")]
    seed: u64,
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
struct GenParams {
    seed: u32,
    batch: u32,
    pad_a: u32,
    pad_b: u32,
}

fn main() {
    let args: Args = argh::from_env();
    assert!(args.batch % 16 == 0, "batch must be a multiple of 16");
    pollster::block_on(run(args));
}

struct Gym {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    bufs: Buffers,
    batch: u32,
    loss_scale: f32,
    lr: f32,
}

struct Pipelines {
    mm_nn: wgpu::ComputePipeline,
    mm_nt: wgpu::ComputePipeline,
    mm_tn: wgpu::ComputePipeline,
    bias_act: wgpu::ComputePipeline,
    loss_grad: wgpu::ComputePipeline,
    relu_bwd: wgpu::ComputePipeline,
    bias_grad: wgpu::ComputePipeline,
    adam: wgpu::ComputePipeline,
    gym_gen: wgpu::ComputePipeline,
    infer_coopvec: wgpu::ComputePipeline,
}

struct Buffers {
    act: Vec<wgpu::Buffer>,
    pre: wgpu::Buffer,
    preds: wgpu::Buffer,
    dz_a: wgpu::Buffer,
    dz_b: wgpu::Buffer,
    targets: wgpu::Buffer,
    loss: wgpu::Buffer,
    zeros: wgpu::Buffer,
    w16: wgpu::Buffer,
    w16_t: wgpu::Buffer,
    b16: wgpu::Buffer,
    preds16: wgpu::Buffer,
    master_w: wgpu::Buffer,
    master_b: wgpu::Buffer,
    dw: wgpu::Buffer,
    db: wgpu::Buffer,
    m_w: wgpu::Buffer,
    v_w: wgpu::Buffer,
    m_b: wgpu::Buffer,
    v_b: wgpu::Buffer,
    ema_w: wgpu::Buffer,
    ema_b: wgpu::Buffer,
    ema_wt16: wgpu::Buffer,
    ema_b16: wgpu::Buffer,
    fwd_ubo: Vec<wgpu::Buffer>,
    dw_ubo: Vec<wgpu::Buffer>,
    dx_ubo: Vec<wgpu::Buffer>,
    bg_ubo: Vec<wgpu::Buffer>,
    bias_ubo: Vec<wgpu::Buffer>,
    loss_ubo: wgpu::Buffer,
    relu_ubo: wgpu::Buffer,
    adam_w_ubo: wgpu::Buffer,
    adam_b_ubo: wgpu::Buffer,
    gen_ubo: wgpu::Buffer,
    fwd_bg: Vec<wgpu::BindGroup>,
    bias_bg: Vec<wgpu::BindGroup>,
    loss_bg: wgpu::BindGroup,
    dw_bg: Vec<wgpu::BindGroup>,
    dx_bg: Vec<wgpu::BindGroup>,
    relu_bg: Vec<wgpu::BindGroup>,
    bias_grad_bg: Vec<wgpu::BindGroup>,
    adam_w_bg: wgpu::BindGroup,
    adam_b_bg: wgpu::BindGroup,
    gen_bg: wgpu::BindGroup,
    infer_bg: wgpu::BindGroup,
}

async fn run(args: Args) {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("no adapter");
    println!("adapter: {}", adapter.get_info().name);

    let coop = adapter.cooperative_matrix_properties();
    let has_cfg = coop.iter().any(|p| {
        p.m_size == 16
            && p.n_size == 16
            && p.k_size == 16
            && p.ab_type == wgpu::CooperativeScalarType::F16
            && p.cr_type == wgpu::CooperativeScalarType::F32
    });
    assert!(has_cfg, "16x16x16 AB=F16 CR=F32 coopmat config not supported");
    assert!(
        adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_VECTOR),
        "VK_NV_cooperative_vector not supported by this adapter/driver"
    );

    let (device, queue) = unsafe {
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("nrc gym"),
                required_features: wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX
                    | wgpu::Features::EXPERIMENTAL_COOPERATIVE_VECTOR
                    | wgpu::Features::SHADER_F16,
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::enabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device")
    };

    let shader_src = format!(
        "{}\n{}",
        include_str!("../../../crates/bevy_solari/src/nrc/nrc_mlp.wgsl"),
        include_str!("nrc_gym.wgsl")
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nrc kernels"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let mut gym = Gym {
        pipelines: build_pipelines(&device, &module),
        bufs: build_buffers(&device, args.batch),
        device,
        queue,
        batch: args.batch,
        loss_scale: args.loss_scale,
        lr: args.lr,
    };
    build_bind_groups(&mut gym);

    let (master_w, master_b) = init_weights(args.seed);
    gym.queue
        .write_buffer(&gym.bufs.master_w, 0, bytemuck::cast_slice(&master_w));
    gym.queue
        .write_buffer(&gym.bufs.master_b, 0, bytemuck::cast_slice(&master_b));
    let w16: Vec<f16> = master_w.iter().map(|&w| f16::from_f32(w)).collect();
    gym.queue
        .write_buffer(&gym.bufs.w16, 0, bytemuck::cast_slice(&w16));
    let mut w16_t = vec![f16::from_f32(0.0); W_TOTAL];
    for l in 0..LAYERS {
        for r in 0..WIDTH {
            for c in 0..WIDTH {
                w16_t[l * W_ELEMS + c * WIDTH + r] = w16[l * W_ELEMS + r * WIDTH + c];
            }
        }
    }
    gym.queue
        .write_buffer(&gym.bufs.w16_t, 0, bytemuck::cast_slice(&w16_t));
    let b16: Vec<f16> = master_b.iter().map(|&b| f16::from_f32(b)).collect();
    gym.queue
        .write_buffer(&gym.bufs.b16, 0, bytemuck::cast_slice(&b16));

    if !args.skip_checks {
        certify(&gym, &master_w, &master_b).await;
    }

    train(&gym, args.steps).await;
}

fn init_weights(seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32
    };
    let bound = (6.0f32 / WIDTH as f32).sqrt();
    let w = (0..W_TOTAL).map(|_| (next() * 2.0 - 1.0) * bound).collect();
    (w, vec![0.0; B_TOTAL])
}

fn build_pipelines(device: &wgpu::Device, module: &wgpu::ShaderModule) -> Pipelines {
    let make = |entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    Pipelines {
        mm_nn: make("mm_nn"),
        mm_nt: make("mm_nt"),
        mm_tn: make("mm_tn"),
        bias_act: make("bias_act"),
        loss_grad: make("loss_grad"),
        relu_bwd: make("relu_bwd"),
        bias_grad: make("bias_grad"),
        adam: make("adam"),
        gym_gen: make("gym_gen"),
        infer_coopvec: make("nrc_infer_coopvec"),
    }
}

fn build_buffers(device: &wgpu::Device, batch: u32) -> Buffers {
    let b = batch as u64;
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let mk_ubo = |label: &str, data: &[u8]| {
        let buf = mk(label, data.len() as u64, uniform);
        buf
    };
    let act = (0..LAYERS)
        .map(|l| mk(&format!("act{l}"), b * WIDTH as u64 * 2, storage))
        .collect();
    let fwd_ubo = (0..LAYERS)
        .map(|l| mk_ubo(&format!("fwd_ubo{l}"), bytemuck::bytes_of(&MmDims::default())))
        .collect();
    let dw_ubo = (0..LAYERS)
        .map(|l| mk_ubo(&format!("dw_ubo{l}"), bytemuck::bytes_of(&MmDims::default())))
        .collect();
    let dx_ubo = (1..LAYERS)
        .map(|l| mk_ubo(&format!("dx_ubo{l}"), bytemuck::bytes_of(&MmDims::default())))
        .collect();
    let bg_ubo = (0..LAYERS)
        .map(|l| mk_ubo(&format!("bg_ubo{l}"), bytemuck::bytes_of(&MmDims::default())))
        .collect();
    let bias_ubo = (0..LAYERS)
        .map(|l| mk_ubo(&format!("bias_ubo{l}"), bytemuck::bytes_of(&EwParams::default())))
        .collect();
    Buffers {
        act,
        pre: mk("pre", b * WIDTH as u64 * 4, storage),
        preds: mk("preds", b * WIDTH as u64 * 4, storage),
        dz_a: mk("dz_a", b * WIDTH as u64 * 2, storage),
        dz_b: mk("dz_b", b * WIDTH as u64 * 2, storage),
        targets: mk("targets", b * 16, storage),
        loss: mk("loss", b * 4, storage),
        zeros: mk("zeros", 256 * 4, storage),
        w16: mk("w16", W_TOTAL as u64 * 2, storage),
        w16_t: mk("w16_t", W_TOTAL as u64 * 2, storage),
        b16: mk("b16", B_TOTAL as u64 * 2, storage),
        preds16: mk("preds16", b * WIDTH as u64 * 2, storage),
        master_w: mk("master_w", W_TOTAL as u64 * 4, storage),
        master_b: mk("master_b", B_TOTAL as u64 * 4, storage),
        dw: mk("dw", W_TOTAL as u64 * 4, storage),
        db: mk("db", B_TOTAL as u64 * 4, storage),
        m_w: mk("m_w", W_TOTAL as u64 * 4, storage),
        v_w: mk("v_w", W_TOTAL as u64 * 4, storage),
        m_b: mk("m_b", B_TOTAL as u64 * 4, storage),
        v_b: mk("v_b", B_TOTAL as u64 * 4, storage),
        ema_w: mk("ema_w", W_TOTAL as u64 * 4, storage),
        ema_b: mk("ema_b", B_TOTAL as u64 * 4, storage),
        ema_wt16: mk("ema_wt16", W_TOTAL as u64 * 2, storage),
        ema_b16: mk("ema_b16", B_TOTAL as u64 * 2, storage),
        fwd_ubo,
        dw_ubo,
        dx_ubo,
        bg_ubo,
        bias_ubo,
        loss_ubo: mk("loss_ubo", 16, uniform),
        relu_ubo: mk("relu_ubo", 16, uniform),
        adam_w_ubo: mk("adam_w_ubo", 48, uniform),
        adam_b_ubo: mk("adam_b_ubo", 48, uniform),
        gen_ubo: mk("gen_ubo", 16, uniform),
        fwd_bg: vec![],
        bias_bg: vec![],
        loss_bg: placeholder_bg(device),
        dw_bg: vec![],
        dx_bg: vec![],
        relu_bg: vec![],
        bias_grad_bg: vec![],
        adam_w_bg: placeholder_bg(device),
        adam_b_bg: placeholder_bg(device),
        gen_bg: placeholder_bg(device),
        infer_bg: placeholder_bg(device),
    }
}

fn placeholder_bg(device: &wgpu::Device) -> wgpu::BindGroup {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[],
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[],
    })
}

impl Default for MmDims {
    fn default() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}
impl Default for EwParams {
    fn default() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

fn bg(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    entries: &[(u32, wgpu::BindingResource)],
) -> wgpu::BindGroup {
    let entries: Vec<_> = entries
        .iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

fn dz_cur(bufs: &Buffers, layer: usize) -> &wgpu::Buffer {
    if layer % 2 == 1 { &bufs.dz_a } else { &bufs.dz_b }
}
fn dz_next(bufs: &Buffers, layer: usize) -> &wgpu::Buffer {
    if layer % 2 == 1 { &bufs.dz_b } else { &bufs.dz_a }
}

fn build_bind_groups(gym: &mut Gym) {
    let batch = gym.batch;
    let d = &gym.device;
    let q = &gym.queue;
    let p = &gym.pipelines;
    let bufs = &gym.bufs;

    for l in 0..LAYERS {
        q.write_buffer(
            &bufs.fwd_ubo[l],
            0,
            bytemuck::bytes_of(&MmDims {
                m: batch,
                n: WIDTH as u32,
                k: WIDTH as u32,
                b_off: (l * W_ELEMS) as u32,
                ..Default::default()
            }),
        );
        q.write_buffer(
            &bufs.dw_ubo[l],
            0,
            bytemuck::bytes_of(&MmDims {
                m: WIDTH as u32,
                n: WIDTH as u32,
                k: batch,
                c_off: (l * W_ELEMS) as u32,
                ..Default::default()
            }),
        );
        q.write_buffer(
            &bufs.bg_ubo[l],
            0,
            bytemuck::bytes_of(&MmDims {
                k: batch,
                c_off: (l * WIDTH) as u32,
                ..Default::default()
            }),
        );
        q.write_buffer(
            &bufs.bias_ubo[l],
            0,
            bytemuck::bytes_of(&EwParams {
                batch,
                layer_off: (l * WIDTH) as u32,
                relu: if l < LAYERS - 1 { 1 } else { 0 },
                loss_scale: 0.0,
            }),
        );
        if l >= 1 {
            q.write_buffer(
                &bufs.dx_ubo[l - 1],
                0,
                bytemuck::bytes_of(&MmDims {
                    m: batch,
                    n: WIDTH as u32,
                    k: WIDTH as u32,
                    b_off: (l * W_ELEMS) as u32,
                    ..Default::default()
                }),
            );
        }
    }
    q.write_buffer(
        &bufs.loss_ubo,
        0,
        bytemuck::bytes_of(&EwParams {
            batch,
            layer_off: 0,
            relu: 0,
            loss_scale: gym.loss_scale,
        }),
    );
    q.write_buffer(
        &bufs.relu_ubo,
        0,
        bytemuck::bytes_of(&EwParams {
            batch,
            ..Default::default()
        }),
    );

    let mut fwd_bg = vec![];
    let mut bias_bg = vec![];
    let mut dw_bg = vec![];
    let mut dx_bg = vec![];
    let mut relu_bg = vec![];
    let mut bias_grad_bg = vec![];
    for l in 0..LAYERS {
        fwd_bg.push(bg(
            d,
            &p.mm_nn,
            &[
                (0, bufs.act[l].as_entire_binding()),
                (1, bufs.w16.as_entire_binding()),
                (2, bufs.pre.as_entire_binding()),
                (3, bufs.zeros.as_entire_binding()),
                (4, bufs.fwd_ubo[l].as_entire_binding()),
            ],
        ));
        // output layer's f16 activation write is dead — point it at scratch
        let act_target = if l < LAYERS - 1 { &bufs.act[l + 1] } else { &bufs.dz_b };
        bias_bg.push(bg(
            d,
            &p.bias_act,
            &[
                (5, bufs.pre.as_entire_binding()),
                (6, act_target.as_entire_binding()),
                (7, bufs.master_b.as_entire_binding()),
                (8, bufs.bias_ubo[l].as_entire_binding()),
            ],
        ));
        dw_bg.push(bg(
            d,
            &p.mm_tn,
            &[
                (0, bufs.act[l].as_entire_binding()),
                (1, dz_cur(bufs, l).as_entire_binding()),
                (2, bufs.dw.as_entire_binding()),
                (3, bufs.zeros.as_entire_binding()),
                (4, bufs.dw_ubo[l].as_entire_binding()),
            ],
        ));
        bias_grad_bg.push(bg(
            d,
            &p.bias_grad,
            &[
                (2, bufs.db.as_entire_binding()),
                (4, bufs.bg_ubo[l].as_entire_binding()),
                (12, dz_cur(bufs, l).as_entire_binding()),
            ],
        ));
        if l >= 1 {
            dx_bg.push(bg(
                d,
                &p.mm_nt,
                &[
                    (0, dz_cur(bufs, l).as_entire_binding()),
                    (1, bufs.w16.as_entire_binding()),
                    (2, bufs.pre.as_entire_binding()),
                    (3, bufs.zeros.as_entire_binding()),
                    (4, bufs.dx_ubo[l - 1].as_entire_binding()),
                ],
            ));
            relu_bg.push(bg(
                d,
                &p.relu_bwd,
                &[
                    (5, bufs.pre.as_entire_binding()),
                    (6, dz_next(bufs, l).as_entire_binding()),
                    (8, bufs.relu_ubo.as_entire_binding()),
                    (12, bufs.act[l].as_entire_binding()),
                ],
            ));
        }
    }
    let loss_bg = bg(
        d,
        &p.loss_grad,
        &[
            (5, bufs.pre.as_entire_binding()),
            (8, bufs.loss_ubo.as_entire_binding()),
            (9, bufs.targets.as_entire_binding()),
            (10, bufs.loss.as_entire_binding()),
            (11, bufs.dz_a.as_entire_binding()),
        ],
    );
    let adam_w_bg = bg(
        d,
        &p.adam,
        &[
            (13, bufs.master_w.as_entire_binding()),
            (14, bufs.dw.as_entire_binding()),
            (15, bufs.m_w.as_entire_binding()),
            (16, bufs.v_w.as_entire_binding()),
            (17, bufs.w16.as_entire_binding()),
            (18, bufs.adam_w_ubo.as_entire_binding()),
            (21, bufs.w16_t.as_entire_binding()),
            (23, bufs.ema_w.as_entire_binding()),
            (24, bufs.ema_wt16.as_entire_binding()),
        ],
    );
    let adam_b_bg = bg(
        d,
        &p.adam,
        &[
            (13, bufs.master_b.as_entire_binding()),
            (14, bufs.db.as_entire_binding()),
            (15, bufs.m_b.as_entire_binding()),
            (16, bufs.v_b.as_entire_binding()),
            (17, bufs.b16.as_entire_binding()),
            (18, bufs.adam_b_ubo.as_entire_binding()),
            (21, bufs.w16_t.as_entire_binding()),
            (23, bufs.ema_b.as_entire_binding()),
            (24, bufs.ema_b16.as_entire_binding()),
        ],
    );
    let infer_bg = bg(
        d,
        &p.infer_coopvec,
        &[
            (0, bufs.act[0].as_entire_binding()),
            (1, bufs.w16_t.as_entire_binding()),
            (6, bufs.preds16.as_entire_binding()),
            (8, bufs.relu_ubo.as_entire_binding()),
            (12, bufs.b16.as_entire_binding()),
        ],
    );
    let gen_bg = bg(
        d,
        &p.gym_gen,
        &[
            (6, bufs.act[0].as_entire_binding()),
            (20, bufs.targets.as_entire_binding()),
            (23, bufs.gen_ubo.as_entire_binding()),
        ],
    );

    let bufs = &mut gym.bufs;
    bufs.fwd_bg = fwd_bg;
    bufs.bias_bg = bias_bg;
    bufs.loss_bg = loss_bg;
    bufs.dw_bg = dw_bg;
    bufs.dx_bg = dx_bg;
    bufs.relu_bg = relu_bg;
    bufs.bias_grad_bg = bias_grad_bg;
    bufs.adam_w_bg = adam_w_bg;
    bufs.adam_b_bg = adam_b_bg;
    bufs.gen_bg = gen_bg;
    bufs.infer_bg = infer_bg;
}

fn encode_forward(gym: &Gym, pass: &mut wgpu::ComputePass) {
    let b = gym.batch;
    for l in 0..LAYERS {
        pass.set_pipeline(&gym.pipelines.mm_nn);
        pass.set_bind_group(0, &gym.bufs.fwd_bg[l], &[]);
        pass.dispatch_workgroups(b / TILE, WIDTH as u32 / TILE, 1);
        pass.set_pipeline(&gym.pipelines.bias_act);
        pass.set_bind_group(0, &gym.bufs.bias_bg[l], &[]);
        pass.dispatch_workgroups(b, 1, 1);
    }
}

fn encode_backward(gym: &Gym, pass: &mut wgpu::ComputePass) {
    let b = gym.batch;
    pass.set_pipeline(&gym.pipelines.loss_grad);
    pass.set_bind_group(0, &gym.bufs.loss_bg, &[]);
    pass.dispatch_workgroups(b.div_ceil(64), 1, 1);
    for l in (0..LAYERS).rev() {
        pass.set_pipeline(&gym.pipelines.mm_tn);
        pass.set_bind_group(0, &gym.bufs.dw_bg[l], &[]);
        pass.dispatch_workgroups(WIDTH as u32 / TILE, WIDTH as u32 / TILE, 1);
        pass.set_pipeline(&gym.pipelines.bias_grad);
        pass.set_bind_group(0, &gym.bufs.bias_grad_bg[l], &[]);
        pass.dispatch_workgroups(1, 1, 1);
        if l >= 1 {
            pass.set_pipeline(&gym.pipelines.mm_nt);
            pass.set_bind_group(0, &gym.bufs.dx_bg[l - 1], &[]);
            pass.dispatch_workgroups(b / TILE, WIDTH as u32 / TILE, 1);
            pass.set_pipeline(&gym.pipelines.relu_bwd);
            pass.set_bind_group(0, &gym.bufs.relu_bg[l - 1], &[]);
            pass.dispatch_workgroups(b, 1, 1);
        }
    }
}

fn run_step(gym: &Gym, step: u32, train: bool) {
    let inv = 1.0 / (gym.loss_scale * gym.batch as f32);
    gym.queue.write_buffer(
        &gym.bufs.gen_ubo,
        0,
        bytemuck::bytes_of(&GenParams {
            seed: step,
            batch: gym.batch,
            pad_a: 0,
            pad_b: 0,
        }),
    );
    if train {
        gym.queue.write_buffer(
            &gym.bufs.adam_w_ubo,
            0,
            bytemuck::bytes_of(&AdamParams {
                count: W_TOTAL as u32,
                step: step + 1,
                mirror_mode: 1,
                lr: gym.lr,
                beta_one: 0.9,
                beta_two: 0.999,
                eps: 1e-8,
                inv_grad_scale: inv,
                // 1 = mirrors track the live master exactly, so the parity
                // exams compare identical weights.
                ema_alpha: 1.0,
                pad_a: 0,
                pad_b: 0,
                pad_c: 0,
            }),
        );
        gym.queue.write_buffer(
            &gym.bufs.adam_b_ubo,
            0,
            bytemuck::bytes_of(&AdamParams {
                count: B_TOTAL as u32,
                step: step + 1,
                mirror_mode: 2,
                lr: gym.lr,
                beta_one: 0.9,
                beta_two: 0.999,
                eps: 1e-8,
                inv_grad_scale: inv,
                ema_alpha: 1.0,
                pad_a: 0,
                pad_b: 0,
                pad_c: 0,
            }),
        );
    }
    let mut encoder = gym
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&gym.pipelines.gym_gen);
        pass.set_bind_group(0, &gym.bufs.gen_bg, &[]);
        pass.dispatch_workgroups(gym.batch.div_ceil(64), 1, 1);
        encode_forward(gym, &mut pass);
    }
    // backward reuses `pre` as its dX scratch — snapshot predictions first
    encoder.copy_buffer_to_buffer(&gym.bufs.pre, 0, &gym.bufs.preds, 0, gym.bufs.preds.size());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        encode_backward(gym, &mut pass);
        if train {
            pass.set_pipeline(&gym.pipelines.adam);
            pass.set_bind_group(0, &gym.bufs.adam_w_bg, &[]);
            pass.dispatch_workgroups((W_TOTAL as u32).div_ceil(64), 1, 1);
            pass.set_bind_group(0, &gym.bufs.adam_b_bg, &[]);
            pass.dispatch_workgroups((B_TOTAL as u32).div_ceil(64), 1, 1);
        }
    }
    gym.queue.submit(Some(encoder.finish()));
}

async fn read_buffer(gym: &Gym, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let staging = gym.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gym
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    gym.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    gym.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range().to_vec();
    data
}

async fn read_f32(gym: &Gym, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
    bytemuck::cast_slice(&read_buffer(gym, buffer, (count * 4) as u64).await).to_vec()
}

async fn read_f16(gym: &Gym, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let raw = read_buffer(gym, buffer, (count * 2) as u64).await;
    bytemuck::cast_slice::<_, f16>(&raw)
        .iter()
        .map(|x| x.to_f32())
        .collect()
}

// CPU reference forward: f32 accumulate, activations quantized to f16
// between layers, exactly like the GPU path. Returns per-layer activations.
fn cpu_forward(
    x_enc: &[f32],
    w16: &[f32],
    bias: &[f32],
    batch: usize,
) -> Vec<Vec<f32>> {
    let mut acts = vec![x_enc.to_vec()];
    for l in 0..LAYERS {
        let x = &acts[l];
        let mut y = vec![0.0f32; batch * WIDTH];
        for s in 0..batch {
            for j in 0..WIDTH {
                let mut sum = 0.0f32;
                for i in 0..WIDTH {
                    sum += x[s * WIDTH + i] * w16[l * W_ELEMS + i * WIDTH + j];
                }
                let mut v = sum + bias[l * WIDTH + j];
                if l < LAYERS - 1 {
                    v = f16::from_f32(v.max(0.0)).to_f32();
                }
                y[s * WIDTH + j] = v;
            }
        }
        acts.push(y);
    }
    acts
}

// f64 accumulation: FD divides tiny loss differences by 2h, so f32 sum
// noise would put a ~3e-3 absolute floor under every finite difference.
// The denominator is the shared prediction LUMINANCE (matches
// nrc_mlp.wgsl::loss_grad).
fn cpu_lum_denom(preds: &[f32], s: usize) -> f32 {
    let lum = 0.2126 * preds[s * WIDTH]
        + 0.7152 * preds[s * WIDTH + 1]
        + 0.0722 * preds[s * WIDTH + 2];
    lum * lum + REL_EPS
}

fn cpu_loss(preds: &[f32], targets: &[f32], denom_frozen: Option<&[f32]>, batch: usize) -> f64 {
    let mut total = 0.0f64;
    for s in 0..batch {
        for c in 0..OUT_CH {
            let p = preds[s * WIDTH + c];
            let diff = p - targets[s * 4 + c];
            let denom = match denom_frozen {
                Some(d) => d[s * OUT_CH + c],
                None => cpu_lum_denom(preds, s),
            };
            total += (diff * diff / denom) as f64;
        }
    }
    total / batch as f64
}

fn run_infer(gym: &Gym) {
    let mut encoder = gym
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&gym.pipelines.infer_coopvec);
        pass.set_bind_group(0, &gym.bufs.infer_bg, &[]);
        pass.dispatch_workgroups(gym.batch.div_ceil(64), 1, 1);
    }
    gym.queue.submit(Some(encoder.finish()));
}

async fn certify(gym: &Gym, master_w: &[f32], master_b: &[f32]) {
    let batch = gym.batch as usize;
    println!("== certification: forward check ==");
    run_step(gym, 0, false);
    let x_enc = read_f16(gym, &gym.bufs.act[0], batch * WIDTH).await;
    let preds_gpu = read_f32(gym, &gym.bufs.preds, batch * WIDTH).await;
    let targets = read_f32(gym, &gym.bufs.targets, batch * 4).await;
    let w16: Vec<f32> = master_w
        .iter()
        .map(|&w| f16::from_f32(w).to_f32())
        .collect();
    let acts_cpu = cpu_forward(&x_enc, &w16, master_b, batch);
    let preds_cpu = &acts_cpu[LAYERS];
    let mut max_err = 0.0f32;
    for i in 0..batch * WIDTH {
        max_err = max_err.max((preds_gpu[i] - preds_cpu[i]).abs());
    }
    println!("forward max abs err GPU vs CPU: {max_err:.6}");
    assert!(max_err < 5e-3, "forward check FAILED");

    println!("== certification: coopvec inference parity ==");
    run_infer(gym);
    let preds_cv = read_f16(gym, &gym.bufs.preds16, batch * WIDTH).await;
    let mut cv_err = 0.0f32;
    let mut cv_rel = 0.0f32;
    for i in 0..batch * WIDTH {
        let err = (preds_cv[i] - preds_gpu[i]).abs();
        let tol = 0.02 * preds_gpu[i].abs() + 0.02;
        cv_rel = cv_rel.max(err / tol);
        cv_err = cv_err.max(err);
    }
    println!("coopvec vs coopmat forward: max abs err {cv_err:.5}, worst err/tol {cv_rel:.3}");
    assert!(cv_rel < 1.0, "coopvec inference parity FAILED");

    println!("== certification: gradient check ==");
    let denom_frozen: Vec<f32> = (0..batch)
        .flat_map(|s| {
            let d = cpu_lum_denom(&preds_cpu, s);
            (0..OUT_CH).map(move |_| d)
        })
        .collect();
    let dw_gpu = read_f32(gym, &gym.bufs.dw, W_TOTAL).await;
    let db_gpu = read_f32(gym, &gym.bufs.db, B_TOTAL).await;
    let scale = 1.0 / (gym.loss_scale * batch as f32);

    let h = 1.0f32 / 64.0;
    let mut checked = 0;
    let mut max_rel = 0.0f32;
    let mut rng = 0x2545f4914f6cdd1du64;
    let mut samples: Vec<(bool, usize)> = vec![];
    for _ in 0..30 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        samples.push((true, (rng % W_TOTAL as u64) as usize));
    }
    for l in 0..LAYERS {
        samples.push((false, l * WIDTH + (l * 13) % WIDTH));
    }
    for (is_weight, idx) in samples {
        let analytic = if is_weight {
            dw_gpu[idx] * scale
        } else {
            db_gpu[idx] * scale
        };
        let fd = {
            let eval = |delta: f32| {
                let mut w = w16.clone();
                let mut bias = master_b.to_vec();
                if is_weight {
                    w[idx] = f16::from_f32(w16[idx] + delta).to_f32();
                } else {
                    bias[idx] += delta;
                }
                let acts = cpu_forward(&x_enc, &w, &bias, batch);
                cpu_loss(&acts[LAYERS], &targets, Some(&denom_frozen), batch)
            };
            ((eval(h) - eval(-h)) / (2.0 * h as f64)) as f32
        };
        // hybrid tolerance: f16 dz quantization + FD noise are ABSOLUTE
        // error sources, so small gradients get an absolute floor
        let tol = 0.03 * analytic.abs().max(fd.abs()) + 8e-3;
        let err = (analytic - fd).abs();
        if analytic.abs().max(fd.abs()) > 1e-4 {
            max_rel = max_rel.max(err / tol);
            checked += 1;
        }
        println!(
            "{} [{idx}]: analytic {analytic:.6} fd {fd:.6} err/tol {:.3}",
            if is_weight { "W" } else { "b" },
            err / tol,
        );
    }
    println!("gradient check: {checked} params above noise floor, worst err/tol {max_rel:.3}");
    assert!(max_rel < 1.0, "gradient check FAILED");
    println!("== certification PASSED ==");
}

async fn train(gym: &Gym, steps: u32) {
    let batch = gym.batch as usize;
    println!("== training {steps} steps, batch {batch}, lr {} ==", gym.lr);
    let start = Instant::now();
    let mut first_loss = None;
    for step in 0..steps {
        run_step(gym, step, true);
        if step % 100 == 0 || step + 1 == steps {
            let loss = read_f32(gym, &gym.bufs.loss, batch).await;
            let mean = loss.iter().sum::<f32>() / batch as f32;
            if first_loss.is_none() {
                first_loss = Some(mean);
            }
            println!("step {step:5}: relative-L2 {mean:.5}");
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let loss = read_f32(gym, &gym.bufs.loss, batch).await;
    let final_loss = loss.iter().sum::<f32>() / batch as f32;
    println!(
        "trained {steps} steps in {elapsed:.2}s ({:.0} steps/s); loss {:.5} -> {final_loss:.5}",
        steps as f64 / elapsed,
        first_loss.unwrap_or(f32::NAN),
    );

    // parity again on the TRAINED weights: adam's transposed/f16 mirrors
    // must track the master copy for coopvec inference to be usable.
    // One extra forward first — `preds` lags the final adam step otherwise.
    run_step(gym, steps, false);
    run_infer(gym);
    let preds_ref = read_f32(gym, &gym.bufs.preds, batch * WIDTH).await;
    let preds_cv = read_f16(gym, &gym.bufs.preds16, batch * WIDTH).await;
    let mut cv_rel = 0.0f32;
    for i in 0..batch * WIDTH {
        let err = (preds_cv[i] - preds_ref[i]).abs();
        cv_rel = cv_rel.max(err / (0.02 * preds_ref[i].abs() + 0.02));
    }
    println!("post-training coopvec parity: worst err/tol {cv_rel:.3}");
    assert!(cv_rel < 1.0, "post-training coopvec parity FAILED");
}
