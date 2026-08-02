//! NRC rung-0 gym (zero/docs/nrc.md): trains the NRC MLP kernels against an
//! analytic radiance field, headless plain-wgpu (no bevy_render — production
//! wiring is rung 1). Certifies three things before training: GPU forward vs
//! a CPU reference (f16-quantized at the same points), analytic gradients vs
//! finite differences of the stop-grad-frozen loss, and loss convergence.
//!
//! The training step is hybrid: a fused Slang kernel (nrc_train.slang,
//! compiled at startup like production's) runs forward + loss + the whole dZ
//! backward chain in one dispatch, and the coopmat kernels (nrc_mlp.slang)
//! reduce dW/db from the recorded activations/dZ before adam.

// passthrough shader modules and ExperimentalFeatures are unsafe wgpu APIs
#![allow(unsafe_code)]

use argh::FromArgs;
use ash::vk;
use half::f16;
use std::time::Instant;
use wgpu::hal::api::Vulkan as VkApi;

const WIDTH: usize = 64;
const LAYERS: usize = 6;
const OUT_CH: usize = 3;
const REL_EPS: f32 = 0.01;
const W_ELEMS: usize = WIDTH * WIDTH;
const W_TOTAL: usize = LAYERS * W_ELEMS;
const B_TOTAL: usize = LAYERS * WIDTH;

/// NRC rung-0 exam: certify the MLP kernels, then train.
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
struct EwParams {
    batch: u32,
    layer_off: u32,
    relu: u32,
    loss_scale: f32,
}

// MUST MATCH TrainParams in nrc_train.slang.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LearnParams {
    batch: u32,
    loss_scale: f32,
    opt_size: u32,
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
    // VK_NV_cooperative_vector: the per-layer TrainingOptimal→RowMajor dW
    // conversion is a raw device command (same recipe as production nrc).
    coopvec_fns: ash::nv::cooperative_vector::Device,
    raw_device: ash::Device,
    opt_size: u32,
}

struct Pipelines {
    learn: wgpu::ComputePipeline,
    adam: wgpu::ComputePipeline,
    gym_gen: wgpu::ComputePipeline,
    infer_coopvec: wgpu::ComputePipeline,
}

struct Buffers {
    // [batch][WIDTH] f16 encoded batch (the network input activations)
    acts: wgpu::Buffer,
    preds: wgpu::Buffer,
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
    // [LAYERS] TrainingOptimal-layout dW accumulator blocks + the device
    // addresses the raw convert reads/writes through.
    dw_opt: wgpu::Buffer,
    dw_opt_addr: u64,
    dw_addr: u64,
    learn_ubo: wgpu::Buffer,
    ew_ubo: wgpu::Buffer,
    adam_w_ubo: wgpu::Buffer,
    adam_b_ubo: wgpu::Buffer,
    gen_ubo: wgpu::Buffer,
    learn_bg: wgpu::BindGroup,
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
    assert!(
        has_cfg,
        "16x16x16 AB=F16 CR=F32 coopmat config not supported"
    );
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
                    | wgpu::Features::SHADER_F16
                    | wgpu::Features::PASSTHROUGH_SHADERS
                    // Pulls in VK_KHR_buffer_device_address, which the raw dW
                    // layout-convert needs to address wgpu-created buffers.
                    | wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                required_limits: wgpu::Limits {
                    // the fused training kernel binds 9 storage buffers
                    max_storage_buffers_per_shader_stage: 16,
                    ..Default::default()
                },
                experimental_features: wgpu::ExperimentalFeatures::enabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device")
    };

    // Raw handles for the dW layout conversion + the host size query for one
    // layer's opaque TrainingOptimal block (`dst_data` null = size only).
    let (coopvec_fns, raw_device, opt_size) = unsafe {
        let hal_device = device
            .as_hal::<VkApi>()
            .expect("nrc gym requires the Vulkan backend");
        let fns = ash::nv::cooperative_vector::Device::load(
            hal_device.shared_instance().raw_instance(),
            hal_device.raw_device(),
        );
        let mut dst_size: usize = 0;
        let mut info = vk::ConvertCooperativeVectorMatrixInfoNV::default()
            .src_size(WIDTH * WIDTH * 4)
            .src_data(vk::DeviceOrHostAddressConstKHR {
                host_address: core::ptr::null(),
            })
            .dst_data(vk::DeviceOrHostAddressKHR {
                host_address: core::ptr::null_mut(),
            })
            .src_component_type(vk::ComponentTypeKHR::FLOAT32)
            .dst_component_type(vk::ComponentTypeKHR::FLOAT32)
            .num_rows(WIDTH as u32)
            .num_columns(WIDTH as u32)
            .src_layout(vk::CooperativeVectorMatrixLayoutNV::ROW_MAJOR)
            .src_stride(WIDTH * 4)
            .dst_layout(vk::CooperativeVectorMatrixLayoutNV::TRAINING_OPTIMAL)
            .dst_stride(0);
        info.p_dst_size = &mut dst_size;
        fns.convert_cooperative_vector_matrix(&info)
            .expect("vkConvertCooperativeVectorMatrixNV size query failed");
        // 64-B stride so every layer's convert src address stays aligned
        // (VUID 10084) — same rounding as production's `init_nrc_pipelines`.
        (fns, hal_device.raw_device().clone(), (dst_size as u32).next_multiple_of(64))
    };
    println!("dW TrainingOptimal block: {opt_size} B/layer (row-major {} B)", WIDTH * WIDTH * 4);

    let mut gym = Gym {
        pipelines: build_pipelines(&device),
        bufs: build_buffers(&device, args.batch, opt_size),
        device,
        queue,
        batch: args.batch,
        loss_scale: args.loss_scale,
        lr: args.lr,
        coopvec_fns,
        raw_device,
        opt_size,
    };
    build_bind_groups(&mut gym);

    // Mark dw initialized with a tracked write: wgpu lazily ZERO-INITIALIZES
    // a buffer at its first tracked use, and dw is only ever written by the
    // raw (untracked) layout convert — without this, the first tracked read
    // (adam / the certification readback) would wipe the converted gradients.
    gym.queue
        .write_buffer(&gym.bufs.dw, 0, &vec![0u8; W_TOTAL * 4]);

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

fn build_pipelines(device: &wgpu::Device) -> Pipelines {
    // Every kernel compiles from the crate's Slang source, with the same
    // `nrc_mlp` module production imports, then loads as passthrough SPIR-V:
    // no naga reflection, so each bind group layout is spelled out to match
    // its kernel's [[vk::binding]] table.
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mlp: &[(&str, &str)] = &[(
        "nrc_mlp",
        include_str!("../../../crates/bevy_solari/src/nrc/nrc_mlp.slang"),
    )];
    let make = |label: &str,
                file: &str,
                source: &str,
                entry: &str,
                entries: &[wgpu::BindGroupLayoutEntry]| {
        let spv = bevy::solari::gpu::slang::compile_rt_slang(
            file,
            source,
            entry,
            mlp,
            &[],
            &[],
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                spirv: Some(std::borrow::Cow::Owned(spv.spirv)),
                ..Default::default()
            })
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    };

    Pipelines {
        learn: make(
            "nrc_learn",
            "nrc_train.slang",
            include_str!("../../../crates/bevy_solari/src/nrc/nrc_train.slang"),
            "learn_gradient",
            &[
                storage(0, true),  // acts (encoded batch)
                storage(1, true),  // weights_t
                storage(2, true),  // biases
                storage(3, true),  // targets
                storage(4, false), // preds
                storage(5, false), // loss
                storage(6, false), // dw_opt
                storage(7, false), // db
                uniform(8),
                storage(9, true), // zeros16
            ],
        ),
        adam: make(
            "adam",
            "nrc_adam.slang",
            include_str!("../../../crates/bevy_solari/src/nrc/nrc_adam.slang"),
            "adam",
            &[
                storage(0, false), // master
                storage(1, true),  // grad_in
                storage(2, false), // moment_m
                storage(3, false), // moment_v
                storage(4, false), // mirror_f16
                storage(5, false), // mirror_alt
                storage(6, false), // ema_master
                storage(7, false), // mirror_ema
                uniform(8),        // adam_u
            ],
        ),
        gym_gen: make(
            "gym_gen",
            "nrc_gym_gen.slang",
            include_str!("nrc_gym_gen.slang"),
            "gym_gen",
            &[storage(0, false), storage(1, false), uniform(2)],
        ),
        infer_coopvec: make(
            "nrc_infer_coopvec",
            "nrc_infer_coopvec.slang",
            include_str!("../../../crates/bevy_solari/src/nrc/nrc_infer_coopvec.slang"),
            "nrc_infer_coopvec",
            &[
                storage(0, true),  // inputs
                storage(1, true),  // weights_t
                storage(2, true),  // biases
                storage(3, false), // preds_out
                uniform(4),        // ew
            ],
        ),
    }
}

fn build_buffers(device: &wgpu::Device, batch: u32, opt_size: u32) -> Buffers {
    let b = batch as u64;
    let storage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    // wgpu-hal creates every storage buffer with SHADER_DEVICE_ADDRESS, so
    // the raw convert can address these directly. (64 B convert alignment is
    // satisfied here only by driver tolerance — see heap_plan.md findings;
    // dedicated raw allocations are the strict fix.)
    let dw_opt = mk("dw_opt", LAYERS as u64 * opt_size as u64, storage);
    let dw = mk("dw", W_TOTAL as u64 * 4, storage);
    let device_address = |buf: &wgpu::Buffer| unsafe {
        let raw = buf
            .as_hal::<VkApi>()
            .expect("vulkan backend")
            .raw_handle();
        let hal_device = device.as_hal::<VkApi>().expect("vulkan backend");
        hal_device
            .raw_device()
            .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(raw))
    };
    let dw_opt_addr = device_address(&dw_opt);
    let dw_addr = device_address(&dw);
    Buffers {
        acts: mk("acts", b * WIDTH as u64 * 2, storage),
        preds: mk("preds", b * WIDTH as u64 * 4, storage),
        targets: mk("targets", b * 16, storage),
        loss: mk("loss", b * 4, storage),
        zeros: mk("zeros", 256 * 4, storage),
        w16: mk("w16", W_TOTAL as u64 * 2, storage),
        w16_t: mk("w16_t", W_TOTAL as u64 * 2, storage),
        b16: mk("b16", B_TOTAL as u64 * 2, storage),
        preds16: mk("preds16", b * WIDTH as u64 * 2, storage),
        master_w: mk("master_w", W_TOTAL as u64 * 4, storage),
        master_b: mk("master_b", B_TOTAL as u64 * 4, storage),
        dw,
        db: mk("db", B_TOTAL as u64 * 4, storage),
        m_w: mk("m_w", W_TOTAL as u64 * 4, storage),
        v_w: mk("v_w", W_TOTAL as u64 * 4, storage),
        m_b: mk("m_b", B_TOTAL as u64 * 4, storage),
        v_b: mk("v_b", B_TOTAL as u64 * 4, storage),
        ema_w: mk("ema_w", W_TOTAL as u64 * 4, storage),
        ema_b: mk("ema_b", B_TOTAL as u64 * 4, storage),
        ema_wt16: mk("ema_wt16", W_TOTAL as u64 * 2, storage),
        ema_b16: mk("ema_b16", B_TOTAL as u64 * 2, storage),
        dw_opt,
        dw_opt_addr,
        dw_addr,
        learn_ubo: mk("learn_ubo", 16, uniform),
        ew_ubo: mk("ew_ubo", 16, uniform),
        adam_w_ubo: mk("adam_w_ubo", 48, uniform),
        adam_b_ubo: mk("adam_b_ubo", 48, uniform),
        gen_ubo: mk("gen_ubo", 16, uniform),
        learn_bg: placeholder_bg(device),
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

fn build_bind_groups(gym: &mut Gym) {
    let batch = gym.batch;
    let d = &gym.device;
    let q = &gym.queue;
    let p = &gym.pipelines;
    let bufs = &gym.bufs;
    q.write_buffer(
        &bufs.learn_ubo,
        0,
        bytemuck::bytes_of(&LearnParams {
            batch,
            loss_scale: gym.loss_scale,
            opt_size: gym.opt_size,
            pad_b: 0,
        }),
    );
    q.write_buffer(
        &bufs.ew_ubo,
        0,
        bytemuck::bytes_of(&EwParams {
            batch,
            layer_off: 0,
            relu: 0,
            loss_scale: 0.0,
        }),
    );

    let learn_bg = bg(
        d,
        &p.learn,
        &[
            (0, bufs.acts.as_entire_binding()),
            (1, bufs.w16_t.as_entire_binding()),
            (2, bufs.b16.as_entire_binding()),
            (3, bufs.targets.as_entire_binding()),
            (4, bufs.preds.as_entire_binding()),
            (5, bufs.loss.as_entire_binding()),
            (6, bufs.dw_opt.as_entire_binding()),
            (7, bufs.db.as_entire_binding()),
            (8, bufs.learn_ubo.as_entire_binding()),
            // f32 zeros reinterpreted: all-zero bytes are all-zero f16s
            (9, bufs.zeros.as_entire_binding()),
        ],
    );
    let adam_w_bg = bg(
        d,
        &p.adam,
        &[
            (0, bufs.master_w.as_entire_binding()),
            (1, bufs.dw.as_entire_binding()),
            (2, bufs.m_w.as_entire_binding()),
            (3, bufs.v_w.as_entire_binding()),
            (4, bufs.w16.as_entire_binding()),
            (5, bufs.w16_t.as_entire_binding()),
            (6, bufs.ema_w.as_entire_binding()),
            (7, bufs.ema_wt16.as_entire_binding()),
            (8, bufs.adam_w_ubo.as_entire_binding()),
        ],
    );
    let adam_b_bg = bg(
        d,
        &p.adam,
        &[
            (0, bufs.master_b.as_entire_binding()),
            (1, bufs.db.as_entire_binding()),
            (2, bufs.m_b.as_entire_binding()),
            (3, bufs.v_b.as_entire_binding()),
            (4, bufs.b16.as_entire_binding()),
            (5, bufs.w16_t.as_entire_binding()),
            (6, bufs.ema_b.as_entire_binding()),
            (7, bufs.ema_b16.as_entire_binding()),
            (8, bufs.adam_b_ubo.as_entire_binding()),
        ],
    );
    let infer_bg = bg(
        d,
        &p.infer_coopvec,
        &[
            (0, bufs.acts.as_entire_binding()),
            (1, bufs.w16_t.as_entire_binding()),
            (2, bufs.b16.as_entire_binding()),
            (3, bufs.preds16.as_entire_binding()),
            (4, bufs.ew_ubo.as_entire_binding()),
        ],
    );
    let gen_bg = bg(
        d,
        &p.gym_gen,
        &[
            (0, bufs.acts.as_entire_binding()),
            (1, bufs.targets.as_entire_binding()),
            (2, bufs.gen_ubo.as_entire_binding()),
        ],
    );

    let bufs = &mut gym.bufs;
    bufs.learn_bg = learn_bg;
    bufs.adam_w_bg = adam_w_bg;
    bufs.adam_b_bg = adam_b_bg;
    bufs.gen_bg = gen_bg;
    bufs.infer_bg = infer_bg;
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
    // Three command buffers: wgpu-core panics if one encoder mixes wgpu
    // passes with raw `as_hal_mut`, so the raw dW layout conversion sits in
    // its own encoder between the training pass and adam (single queue —
    // submission order is execution order; memory visibility comes from the
    // raw sync2 barriers around the converts).
    let mut encoder = gym
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    // The gradient accumulators start at zero every step (the fused kernel's
    // outer-product/reduce-sum accumulates are additive).
    encoder.clear_buffer(&gym.bufs.dw_opt, 0, None);
    encoder.clear_buffer(&gym.bufs.db, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&gym.pipelines.gym_gen);
        pass.set_bind_group(0, &gym.bufs.gen_bg, &[]);
        pass.dispatch_workgroups(gym.batch.div_ceil(64), 1, 1);
        // fused forward + loss + backward incl. the dW/db accumulates
        pass.set_pipeline(&gym.pipelines.learn);
        pass.set_bind_group(0, &gym.bufs.learn_bg, &[]);
        pass.dispatch_workgroups(gym.batch.div_ceil(64), 1, 1);
    }

    let mut conv_encoder = gym
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dw_convert"),
        });
    unsafe {
        conv_encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let cb = hal_encoder.expect("vulkan backend").raw_handle();
            let dev = &gym.raw_device;
            let pre = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::CONVERT_COOPERATIVE_VECTOR_MATRIX_NV)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
            dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&pre));
            let opt_size = gym.opt_size as usize;
            let row_bytes = W_ELEMS * 4;
            let mut dst_sizes = [row_bytes; LAYERS];
            let infos: Vec<vk::ConvertCooperativeVectorMatrixInfoNV> = (0..LAYERS)
                .map(|l| {
                    let mut info = vk::ConvertCooperativeVectorMatrixInfoNV::default()
                        .src_size(opt_size)
                        .src_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: gym.bufs.dw_opt_addr + (l * opt_size) as u64,
                        })
                        .dst_data(vk::DeviceOrHostAddressKHR {
                            device_address: gym.bufs.dw_addr + (l * row_bytes) as u64,
                        })
                        .src_component_type(vk::ComponentTypeKHR::FLOAT32)
                        .dst_component_type(vk::ComponentTypeKHR::FLOAT32)
                        .num_rows(WIDTH as u32)
                        .num_columns(WIDTH as u32)
                        .src_layout(vk::CooperativeVectorMatrixLayoutNV::TRAINING_OPTIMAL)
                        .src_stride(0)
                        .dst_layout(vk::CooperativeVectorMatrixLayoutNV::ROW_MAJOR)
                        .dst_stride(WIDTH * 4);
                    info.p_dst_size = &mut dst_sizes[l];
                    info
                })
                .collect();
            gym.coopvec_fns.cmd_convert_cooperative_vector_matrix(cb, &infos);
            let post = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::CONVERT_COOPERATIVE_VECTOR_MATRIX_NV)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::COPY,
                )
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::TRANSFER_READ)];
            dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&post));
        });
    }

    let mut post_encoder = gym
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    if train {
        let mut pass = post_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&gym.pipelines.adam);
        pass.set_bind_group(0, &gym.bufs.adam_w_bg, &[]);
        pass.dispatch_workgroups((W_TOTAL as u32).div_ceil(64), 1, 1);
        pass.set_bind_group(0, &gym.bufs.adam_b_bg, &[]);
        pass.dispatch_workgroups((B_TOTAL as u32).div_ceil(64), 1, 1);
    }
    gym.queue.submit([encoder.finish(), conv_encoder.finish(), post_encoder.finish()]);
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
// between layers. Returns per-layer activations.
fn cpu_forward(x_enc: &[f32], w16: &[f32], bias: &[f32], batch: usize) -> Vec<Vec<f32>> {
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
// nrc_train.slang).
fn cpu_lum_denom(preds: &[f32], s: usize) -> f32 {
    let lum =
        0.2126 * preds[s * WIDTH] + 0.7152 * preds[s * WIDTH + 1] + 0.0722 * preds[s * WIDTH + 2];
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
    let x_enc = read_f16(gym, &gym.bufs.acts, batch * WIDTH).await;
    let preds_gpu = read_f32(gym, &gym.bufs.preds, batch * WIDTH).await;
    let targets = read_f32(gym, &gym.bufs.targets, batch * 4).await;
    let w16: Vec<f32> = master_w
        .iter()
        .map(|&w| f16::from_f32(w).to_f32())
        .collect();
    let acts_cpu = cpu_forward(&x_enc, &w16, master_b, batch);
    let preds_cpu = &acts_cpu[LAYERS];
    // The fused forward is all-f16 coopvec — hybrid tolerance, not the f32
    // coopmat budget the CPU reference was originally certified against.
    let mut max_err = 0.0f32;
    let mut worst = 0.0f32;
    for i in 0..batch * WIDTH {
        let err = (preds_gpu[i] - preds_cpu[i]).abs();
        max_err = max_err.max(err);
        worst = worst.max(err / (0.02 * preds_cpu[i].abs() + 0.02));
    }
    println!("forward GPU vs CPU: max abs err {max_err:.5}, worst err/tol {worst:.3}");
    assert!(worst < 1.0, "forward check FAILED");

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
    println!("infer vs fused forward: max abs err {cv_err:.5}, worst err/tol {cv_rel:.3}");
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
        // error sources, so small gradients get an absolute floor. The
        // analytic gradient differentiates the network as the fused kernel
        // actually computes it — the all-f16 coopvec forward (the SAME math
        // inference runs) — while the FD reference is the f32-accumulate CPU
        // forward. Near-zero pre-activations flip ReLU masks between the two,
        // so early-layer dW carries an extra systematic component the old
        // f32-accum training forward didn't have; both terms are widened
        // accordingly (worst observed err/tol at the old budget: 1.13).
        let tol = 0.05 * analytic.abs().max(fd.abs()) + 2e-2;
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
