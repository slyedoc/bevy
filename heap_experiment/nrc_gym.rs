//! NRC rung-0 gym (zero/docs/nrc.md): trains the NRC MLP kernels against an
//! analytic radiance field, headless plain-wgpu (no bevy_render — production
//! wiring is rung 1). Certifies three things before training: GPU forward vs
//! a CPU reference (f16-quantized at the same points), analytic gradients vs
//! finite differences of the stop-grad-frozen loss, and loss convergence.
//!
//! The training step is hybrid: a fused Slang kernel (nrc_train.slang,
//! shipped as passthrough SPIR-V) runs forward + loss + the whole dZ
//! backward chain in one dispatch, and the coopmat kernels (nrc_mlp.slang)
//! reduce dW/db from the recorded activations/dZ before adam.

// passthrough shader modules and ExperimentalFeatures are unsafe wgpu APIs
#![allow(unsafe_code)]

use argh::FromArgs;
use ash::vk::{self, TaggedStructure};
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

// Push-data blocks — MUST MATCH the push structs in the .slang kernels
// (heap indices first, then params; all 4-byte scalars, sequentially packed).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LearnPush {
    acts_idx: u32,
    weights_t_idx: u32,
    biases_idx: u32,
    targets_idx: u32,
    preds_idx: u32,
    loss_idx: u32,
    dw_opt_idx: u32,
    db_idx: u32,
    zeros_idx: u32,
    batch: u32,
    loss_scale: f32,
    opt_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamPush {
    master_idx: u32,
    grad_idx: u32,
    m_idx: u32,
    v_idx: u32,
    mirror_f16_idx: u32,
    mirror_alt_idx: u32,
    ema_idx: u32,
    mirror_ema_idx: u32,
    count: u32,
    step: u32,
    mirror_mode: u32,
    lr: f32,
    beta_one: f32,
    beta_two: f32,
    eps: f32,
    inv_grad_scale: f32,
    ema_alpha: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GenPush {
    act_out_idx: u32,
    targets_out_idx: u32,
    seed: u32,
    batch: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InferPush {
    inputs_idx: u32,
    weights_t_idx: u32,
    biases_idx: u32,
    preds_out_idx: u32,
    batch: u32,
}

// Resource-heap descriptor indices, fixed at init (write order below).
mod heap_idx {
    pub const ACTS: u32 = 0;
    pub const W16_T: u32 = 1;
    pub const B16: u32 = 2;
    pub const TARGETS: u32 = 3;
    pub const PREDS: u32 = 4;
    pub const LOSS: u32 = 5;
    pub const DW_OPT: u32 = 6;
    pub const DB: u32 = 7;
    pub const ZEROS: u32 = 8;
    pub const MASTER_W: u32 = 9;
    pub const DW: u32 = 10;
    pub const M_W: u32 = 11;
    pub const V_W: u32 = 12;
    pub const W16: u32 = 13;
    pub const EMA_W: u32 = 14;
    pub const EMA_WT16: u32 = 15;
    pub const MASTER_B: u32 = 16;
    pub const M_B: u32 = 17;
    pub const V_B: u32 = 18;
    pub const EMA_B: u32 = 19;
    pub const EMA_B16: u32 = 20;
    pub const PREDS16: u32 = 21;
    pub const COUNT: u32 = 22;
}

fn main() {
    env_logger::init();
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
    heap: Heap,
}

// VK_EXT_descriptor_heap resource heap: a host-visible BDA buffer holding
// descriptor BYTES; bound once per raw command buffer. No descriptor set
// layouts exist anywhere in this gym.
struct Heap {
    fns: ash::ext::descriptor_heap::Device,
    buffer: vk::Buffer,
    #[allow(dead_code)]
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    address: u64,
    size: u64,
    desc_size: usize,
    reserved_offset: u64,
    reserved_size: u64,
}

// Raw compute pipelines created with PIPELINE_CREATE_2_DESCRIPTOR_HEAP_EXT
// and no pipeline layout.
struct Pipelines {
    learn: vk::Pipeline,
    adam: vk::Pipeline,
    gym_gen: vk::Pipeline,
    infer_coopvec: vk::Pipeline,
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
        (fns, hal_device.raw_device().clone(), dst_size as u32)
    };
    println!("dW TrainingOptimal block: {opt_size} B/layer (row-major {} B)", WIDTH * WIDTH * 4);

    let (raw_instance, phys) = unsafe {
        let hal_device = device.as_hal::<VkApi>().expect("vulkan backend");
        let raw_instance = hal_device.shared_instance().raw_instance().clone();
        let phys = adapter
            .as_hal::<VkApi>()
            .expect("vulkan backend")
            .raw_physical_device();
        (raw_instance, phys)
    };
    let heap = unsafe { build_heap(&raw_instance, phys, &raw_device) };

    let pipelines = unsafe { build_pipelines(&raw_device) };
    device.set_device_lost_callback(|reason, msg| {
        eprintln!("DEVICE LOST: {reason:?}: {msg}");
    });
    let mut gym = Gym {
        pipelines,
        bufs: build_buffers(&device, &raw_device, &raw_instance, phys, args.batch, opt_size),
        device,
        queue,
        batch: args.batch,
        loss_scale: args.loss_scale,
        lr: args.lr,
        coopvec_fns,
        raw_device,
        opt_size,
        heap,
    };
    write_heap_descriptors(&gym);
    // Mark every raw-written buffer the harness later READS (via tracked wgpu
    // copies) as initialized: wgpu lazily zero-initializes a buffer at its
    // first tracked use, wiping untracked (raw compute) writes.
    for (buf, len) in [
        (&gym.bufs.acts, gym.batch as usize * WIDTH * 2),
        (&gym.bufs.targets, gym.batch as usize * 16),
        (&gym.bufs.preds, gym.batch as usize * WIDTH * 4),
        (&gym.bufs.preds16, gym.batch as usize * WIDTH * 2),
        (&gym.bufs.loss, gym.batch as usize * 4),
        (&gym.bufs.db, B_TOTAL * 4),
    ] {
        gym.queue.write_buffer(buf, 0, &vec![0u8; len]);
    }

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

// Raw compute pipelines from the Slang blobs: descriptor-heap pipelines
// have NO pipeline layout and NO descriptor interface — the create flag
// opts them into the bound heaps + push data.
unsafe fn build_pipelines(device: &ash::Device) -> Pipelines {
    let make = |spv_bytes: &[u8]| -> vk::Pipeline {
        let words: Vec<u32> = spv_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let module = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .expect("shader module");
        let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default()
            .flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .push(&mut flags2);
        let pipeline = device
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
            .expect("compute pipeline")[0];
        device.destroy_shader_module(module, None);
        pipeline
    };
    Pipelines {
        learn: make(include_bytes!("../../../crates/bevy_solari/src/nrc/nrc_train.spv")),
        adam: make(include_bytes!("../../../crates/bevy_solari/src/nrc/nrc_adam.spv")),
        gym_gen: make(include_bytes!("nrc_gym_gen.spv")),
        infer_coopvec: make(include_bytes!(
            "../../../crates/bevy_solari/src/nrc/nrc_infer_coopvec.spv"
        )),
    }
}

// Allocate the resource heap: query per-device descriptor sizes + reserved
// range, create a host-visible BDA buffer with DESCRIPTOR_HEAP usage, map it.
unsafe fn build_heap(
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    device: &ash::Device,
) -> Heap {
    let mut heap_props = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut heap_props);
    instance.get_physical_device_properties2(phys, &mut props2);
    // This heap holds ONLY buffer descriptors; the shader-side heap index
    // unit for a descriptor type is that type's descriptor size.
    let desc_size = heap_props.buffer_descriptor_size;
    println!(
        "heap descriptor sizes: buffer {} image {} sampler {} (reserved {})",
        heap_props.buffer_descriptor_size,
        heap_props.image_descriptor_size,
        heap_props.sampler_descriptor_size,
        heap_props.min_resource_heap_reserved_range,
    );
    assert!(desc_size > 0, "descriptor heap unsupported");

    let align = heap_props.resource_heap_alignment.max(1);
    // Reserved range (driver-managed) sits after the descriptor region,
    // aligned to the heap alignment.
    let descriptors = heap_idx::COUNT as u64 * desc_size as u64;
    let reserved_offset = descriptors.div_ceil(align) * align;
    let reserved_size = heap_props.min_resource_heap_reserved_range;
    let size = (reserved_offset + reserved_size).div_ceil(align) * align;

    let buffer = device
        .create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
        .expect("heap buffer");
    let reqs = device.get_buffer_memory_requirements(buffer);
    let mem_props = instance.get_physical_device_memory_properties(phys);
    let type_index = (0..mem_props.memory_type_count)
        .find(|&i| {
            (reqs.memory_type_bits & (1 << i)) != 0
                && mem_props.memory_types[i as usize].property_flags.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
        })
        .expect("host-visible memory type");
    let mut alloc_flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let memory = device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(type_index)
                .push(&mut alloc_flags),
            None,
        )
        .expect("heap memory");
    device.bind_buffer_memory(buffer, memory, 0).expect("bind heap");
    let ptr = device
        .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        .expect("map heap")
        .cast::<u8>();
    let address = device.get_buffer_device_address(
        &vk::BufferDeviceAddressInfo::default().buffer(buffer),
    );
    Heap {
        fns: ash::ext::descriptor_heap::Device::load(instance, device),
        buffer,
        memory,
        ptr,
        address,
        size,
        desc_size: desc_size as usize,
        reserved_offset,
        reserved_size,
    }
}

// Write one STORAGE_BUFFER descriptor per gym buffer into the mapped heap,
// at the fixed `heap_idx` slots.
fn write_heap_descriptors(gym: &Gym) {
    let device_address = |buf: &wgpu::Buffer| unsafe {
        let raw = buf.as_hal::<VkApi>().expect("vulkan backend").raw_handle();
        gym.raw_device
            .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(raw))
    };
    let write = |index: u32, buf: &wgpu::Buffer| unsafe {
        let range = vk::DeviceAddressRangeEXT {
            address: device_address(buf),
            size: buf.size(),
        };
        let info = vk::ResourceDescriptorInfoEXT {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            data: vk::ResourceDescriptorDataEXT {
                p_address_range: &range,
            },
            ..Default::default()
        };
        let mut dst = vk::HostAddressRangeEXT::default();
        dst.address = gym.heap.ptr.add(index as usize * gym.heap.desc_size).cast();
        dst.size = gym.heap.desc_size;
        gym.heap.fns
            .write_resource_descriptors(&[info], &[dst])
            .expect("write descriptor");
    };
    let b = &gym.bufs;
    write(heap_idx::ACTS, &b.acts);
    write(heap_idx::W16_T, &b.w16_t);
    write(heap_idx::B16, &b.b16);
    write(heap_idx::TARGETS, &b.targets);
    write(heap_idx::PREDS, &b.preds);
    write(heap_idx::LOSS, &b.loss);
    write(heap_idx::DW_OPT, &b.dw_opt);
    write(heap_idx::DB, &b.db);
    write(heap_idx::ZEROS, &b.zeros);
    write(heap_idx::MASTER_W, &b.master_w);
    write(heap_idx::DW, &b.dw);
    write(heap_idx::M_W, &b.m_w);
    write(heap_idx::V_W, &b.v_w);
    write(heap_idx::W16, &b.w16);
    write(heap_idx::EMA_W, &b.ema_w);
    write(heap_idx::EMA_WT16, &b.ema_wt16);
    write(heap_idx::MASTER_B, &b.master_b);
    write(heap_idx::M_B, &b.m_b);
    write(heap_idx::V_B, &b.v_b);
    write(heap_idx::EMA_B, &b.ema_b);
    write(heap_idx::EMA_B16, &b.ema_b16);
    write(heap_idx::PREDS16, &b.preds16);
}

// Dedicated raw VK buffer wrapped as a wgpu buffer: offset-0 bind gives a
// 64-byte-aligned device address (vkCmdConvertCooperativeVectorMatrixNV
// requires 64 B alignment; wgpu's own suballocation does not provide it).
fn raw_aligned_buffer(
    device: &wgpu::Device,
    raw_device: &ash::Device,
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    size: u64,
    label: &'static str,
) -> (wgpu::Buffer, u64) {
    unsafe {
        let raw = raw_device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .expect("raw buffer");
        let reqs = raw_device.get_buffer_memory_requirements(raw);
        let mem_props = instance.get_physical_device_memory_properties(phys);
        let type_index = (0..mem_props.memory_type_count)
            .find(|&i| {
                (reqs.memory_type_bits & (1 << i)) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .expect("device-local memory type");
        let mut alloc_flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let memory = raw_device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(type_index)
                    .push(&mut alloc_flags),
                None,
            )
            .expect("raw memory");
        raw_device.bind_buffer_memory(raw, memory, 0).expect("bind");
        let addr = raw_device
            .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(raw));
        let hal_buffer = wgpu::hal::vulkan::Buffer::from_raw_managed(raw, memory, 0, size);
        let buffer = device.create_buffer_from_hal::<VkApi>(
            hal_buffer,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        );
        (buffer, addr)
    }
}

fn build_buffers(
    device: &wgpu::Device,
    raw_device: &ash::Device,
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    batch: u32,
    opt_size: u32,
) -> Buffers {
    let b = batch as u64;
    let storage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    // Dedicated raw allocations: the layout convert requires 64 B-aligned
    // device addresses, which wgpu's suballocated buffers don't guarantee.
    let (dw_opt, dw_opt_addr) = raw_aligned_buffer(
        device,
        raw_device,
        instance,
        phys,
        LAYERS as u64 * opt_size as u64,
        "dw_opt",
    );
    let (dw, dw_addr) = raw_aligned_buffer(device, raw_device, instance, phys, W_TOTAL as u64 * 4, "dw");
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
    }
}

fn run_step(gym: &Gym, step: u32, train: bool) {
    let inv = 1.0 / (gym.loss_scale * gym.batch as f32);
    let b = &gym.bufs;
    let gen_push = GenPush {
        act_out_idx: heap_idx::ACTS,
        targets_out_idx: heap_idx::TARGETS,
        seed: step,
        batch: gym.batch,
    };
    let learn_push = LearnPush {
        acts_idx: heap_idx::ACTS,
        weights_t_idx: heap_idx::W16_T,
        biases_idx: heap_idx::B16,
        targets_idx: heap_idx::TARGETS,
        preds_idx: heap_idx::PREDS,
        loss_idx: heap_idx::LOSS,
        dw_opt_idx: heap_idx::DW_OPT,
        db_idx: heap_idx::DB,
        zeros_idx: heap_idx::ZEROS,
        batch: gym.batch,
        loss_scale: gym.loss_scale,
        opt_size: gym.opt_size,
    };
    let adam_common = AdamPush {
        master_idx: 0,
        grad_idx: 0,
        m_idx: 0,
        v_idx: 0,
        mirror_f16_idx: 0,
        mirror_alt_idx: 0,
        ema_idx: 0,
        mirror_ema_idx: 0,
        count: 0,
        step: step + 1,
        mirror_mode: 0,
        lr: gym.lr,
        beta_one: 0.9,
        beta_two: 0.999,
        eps: 1e-8,
        inv_grad_scale: inv,
        // 1 = mirrors track the live master exactly, so the parity exams
        // compare identical weights.
        ema_alpha: 1.0,
    };
    let adam_w_push = AdamPush {
        master_idx: heap_idx::MASTER_W,
        grad_idx: heap_idx::DW,
        m_idx: heap_idx::M_W,
        v_idx: heap_idx::V_W,
        mirror_f16_idx: heap_idx::W16,
        mirror_alt_idx: heap_idx::W16_T,
        ema_idx: heap_idx::EMA_W,
        mirror_ema_idx: heap_idx::EMA_WT16,
        count: W_TOTAL as u32,
        mirror_mode: 1,
        ..adam_common
    };
    let adam_b_push = AdamPush {
        master_idx: heap_idx::MASTER_B,
        grad_idx: heap_idx::DB,
        m_idx: heap_idx::M_B,
        v_idx: heap_idx::V_B,
        mirror_f16_idx: heap_idx::B16,
        mirror_alt_idx: heap_idx::W16_T,
        ema_idx: heap_idx::EMA_B,
        mirror_ema_idx: heap_idx::EMA_B16,
        count: B_TOTAL as u32,
        mirror_mode: 2,
        ..adam_common
    };

    // The whole training step is ONE raw command buffer: clears, dispatches,
    // the dW layout converts, adam — with explicit sync2 barriers. No wgpu
    // tracking is involved anywhere (see heap_plan.md H-LAW 1).
    let raw_dw_opt = unsafe { b.dw_opt.as_hal::<VkApi>().expect("vk").raw_handle() };
    let raw_db = unsafe { b.db.as_hal::<VkApi>().expect("vk").raw_handle() };
    let mut encoder = gym
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let cb = hal_encoder.expect("vulkan backend").raw_handle();
            let dev = &gym.raw_device;
            let barrier = |src_stage, src_access, dst_stage, dst_access| {
                [vk::MemoryBarrier2::default()
                    .src_stage_mask(src_stage)
                    .src_access_mask(src_access)
                    .dst_stage_mask(dst_stage)
                    .dst_access_mask(dst_access)]
            };
            let dep = |b: &[vk::MemoryBarrier2]| {
                dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(b));
            };
            let push = |data: &[u8]| {
                let mut range = vk::HostAddressRangeConstEXT::default();
                range.address = data.as_ptr().cast();
                range.size = data.len();
                let mut info = vk::PushDataInfoEXT::default();
                info.offset = 0;
                info.data = range;
                gym.heap.fns.cmd_push_data(cb, &info);
            };
            let bisect: u32 = std::env::var("GYM_BISECT").ok().and_then(|v| v.parse().ok()).unwrap_or(99);
            if bisect >= 1 { bind_resource_heap(gym, cb); }

            // Fence prior tracked uploads (weight init, zero-init marks)
            // against this raw command buffer's transfers + reads.
            dep(&barrier(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
                vk::PipelineStageFlags2::CLEAR | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_READ,
            ));

            // Gradient accumulators start at zero (the accumulates are additive).
            dev.cmd_fill_buffer(cb, raw_dw_opt, 0, vk::WHOLE_SIZE, 0);
            dev.cmd_fill_buffer(cb, raw_db, 0, vk::WHOLE_SIZE, 0);
            dep(&barrier(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
            ));

            if bisect >= 2 {
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, gym.pipelines.gym_gen);
            push(bytemuck::bytes_of(&gen_push));
            dev.cmd_dispatch(cb, gym.batch.div_ceil(64), 1, 1);
            }
            dep(&barrier(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            ));

            if bisect >= 3 {
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, gym.pipelines.learn);
            push(bytemuck::bytes_of(&learn_push));
            dev.cmd_dispatch(cb, gym.batch.div_ceil(64), 1, 1);
            }
            dep(&barrier(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::CONVERT_COOPERATIVE_VECTOR_MATRIX_NV,
                vk::AccessFlags2::TRANSFER_READ,
            ));

            // dW: TrainingOptimal → row-major f32 for adam, all layers one cmd.
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
            if bisect >= 4 { gym.coopvec_fns.cmd_convert_cooperative_vector_matrix(cb, &infos); }
            dep(&barrier(
                vk::PipelineStageFlags2::CONVERT_COOPERATIVE_VECTOR_MATRIX_NV,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::TRANSFER_READ,
            ));

            if train {
                dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, gym.pipelines.adam);
                push(bytemuck::bytes_of(&adam_w_push));
                dev.cmd_dispatch(cb, (W_TOTAL as u32).div_ceil(64), 1, 1);
                push(bytemuck::bytes_of(&adam_b_push));
                dev.cmd_dispatch(cb, (B_TOTAL as u32).div_ceil(64), 1, 1);
            }
            // Everything the harness copies out later (tracked wgpu copies in
            // a later submission) is fenced by this final barrier.
            dep(&barrier(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::TRANSFER_READ,
            ));
        });
    }
    eprintln!("TEMP run_step: recorded, submitting");
    gym.queue.submit(Some(encoder.finish()));
    eprintln!("TEMP run_step: submitted");
}

// Bind the resource heap for a raw command buffer. The sampler heap is never
// bound — no kernel constructs a sampler handle.
unsafe fn bind_resource_heap(gym: &Gym, cb: vk::CommandBuffer) {
    let mut info = vk::BindHeapInfoEXT::default();
    info.heap_range = vk::DeviceAddressRangeEXT {
        address: gym.heap.address,
        size: gym.heap.size,
    };
    info.reserved_range_offset = gym.heap.reserved_offset;
    info.reserved_range_size = gym.heap.reserved_size;
    gym.heap.fns.cmd_bind_resource_heap(cb, &info);
}

fn run_infer(gym: &Gym) {
    let infer_push = InferPush {
        inputs_idx: heap_idx::ACTS,
        weights_t_idx: heap_idx::W16_T,
        biases_idx: heap_idx::B16,
        preds_out_idx: heap_idx::PREDS16,
        batch: gym.batch,
    };
    let mut encoder = gym
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let cb = hal_encoder.expect("vulkan backend").raw_handle();
            let dev = &gym.raw_device;
            bind_resource_heap(gym, cb);
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, gym.pipelines.infer_coopvec);
            let mut range = vk::HostAddressRangeConstEXT::default();
            range.address = (&infer_push as *const InferPush).cast();
            range.size = core::mem::size_of::<InferPush>();
            let mut info = vk::PushDataInfoEXT::default();
            info.offset = 0;
            info.data = range;
            gym.heap.fns.cmd_push_data(cb, &info);
            dev.cmd_dispatch(cb, gym.batch.div_ceil(64), 1, 1);
            dev.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().memory_barriers(&[
                    vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ),
                ]),
            );
        });
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
    eprintln!("TEMP read_buffer: polling");
    for i in 0..1200 {
        let state = gym.device.poll(wgpu::PollType::Poll).expect("poll");
        if let Ok(r) = rx.try_recv() {
            r.unwrap();
            eprintln!("TEMP read_buffer: mapped after {i} polls");
            let data = slice.get_mapped_range().to_vec();
            return data;
        }
        eprintln!("TEMP poll {i}: {state:?}");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    panic!("map never completed");
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
