//! Minimal reproducer for the coopvec × descriptor-heap × dispatch-mode hang
//! (see heap_plan.md, H0 findings). Dispatches heap_probe/heap_probe.slang as
//! either a VK_EXT_shader_object or a heap-flagged compute pipeline and
//! reports COMPLETED / HANG, with result values for correctness.
//!
//!   cargo run --example solari_heap_probe -- --mode so
//!   cargo run --example solari_heap_probe -- --mode pipeline
//!   cargo run --example solari_heap_probe -- --mode so --plain   (no coopvec)
//!
//! `--mode mapped` is the M0 gate from moonshot_plan.md: the same coopvec
//! kernel with CLASSIC bindings (heap_probe_mapped.slang), descriptors
//! sourced from the heap host-side via a set→heap mapping
//! (VkShaderDescriptorSetAndBindingMappingInfoEXT, HEAP_WITH_CONSTANT_OFFSET).

#![allow(unsafe_code)]

use argh::FromArgs;
use ash::vk::{self, TaggedStructure};
use wgpu::hal::api::Vulkan as VkApi;

#[derive(FromArgs)]
/// coopvec-over-descriptor-heap dispatch probe
struct Args {
    /// dispatch mode: "so" (shader object), "pipeline", "ptr" (BDA), or
    /// "mapped" (classic bindings + host-side set→heap mapping)
    #[argh(option, default = "String::from(\"so\")")]
    mode: String,
    /// use the plain (non-coopvec) control path
    #[argh(switch)]
    plain: bool,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ProbePush {
    w_idx: u32,
    io_idx: u32,
    zeros_idx: u32,
    batch: u32,
    use_coopvec: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PtrPush {
    w_addr: u64,
    out_addr: u64,
    zeros_addr: u64,
}

const BATCH: u32 = 64;
const WIDTH: usize = 64;

fn main() {
    let args: Args = argh::from_env();
    pollster::block_on(run(args));
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

    let (device, queue) = unsafe {
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("heap probe"),
                required_features: wgpu::Features::EXPERIMENTAL_COOPERATIVE_VECTOR
                    | wgpu::Features::SHADER_F16
                    // Pulls in VK_KHR_buffer_device_address so storage buffers
                    // get SHADER_DEVICE_ADDRESS (heap descriptors need it).
                    | wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::enabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device")
    };
    device.set_device_lost_callback(|reason, msg| {
        eprintln!("DEVICE LOST: {reason:?}: {msg}");
    });

    let (raw_device, raw_instance, phys) = unsafe {
        let hal_device = device.as_hal::<VkApi>().expect("vulkan backend");
        (
            hal_device.raw_device().clone(),
            hal_device.shared_instance().raw_instance().clone(),
            adapter
                .as_hal::<VkApi>()
                .expect("vulkan backend")
                .raw_physical_device(),
        )
    };
    let heap_fns = ash::ext::descriptor_heap::Device::load(&raw_instance, &raw_device);
    let so_fns = ash::ext::shader_object::Device::load(&raw_instance, &raw_device);

    // ── resource heap (host-visible, buffer descriptors only) ───────────────
    let mut heap_props = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut heap_props);
    unsafe { raw_instance.get_physical_device_properties2(phys, &mut props2) };
    let desc_size = heap_props.buffer_descriptor_size as usize;
    let align = heap_props.resource_heap_alignment.max(1);
    let reserved_offset = (3 * desc_size as u64).div_ceil(align) * align;
    let reserved_size = heap_props.min_resource_heap_reserved_range;
    let heap_size = (reserved_offset + reserved_size).div_ceil(align) * align;
    let (heap_buffer, heap_ptr, heap_addr) = unsafe {
        let buffer = raw_device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(heap_size)
                    .usage(
                        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .expect("heap buffer");
        let reqs = raw_device.get_buffer_memory_requirements(buffer);
        let mem_props = raw_instance.get_physical_device_memory_properties(phys);
        let type_index = (0..mem_props.memory_type_count)
            .find(|&i| {
                (reqs.memory_type_bits & (1 << i)) != 0
                    && mem_props.memory_types[i as usize].property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .expect("host-visible type");
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let memory = raw_device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(type_index)
                    .push(&mut flags),
                None,
            )
            .expect("heap memory");
        raw_device.bind_buffer_memory(buffer, memory, 0).expect("bind");
        let ptr = raw_device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .expect("map")
            .cast::<u8>();
        let addr = raw_device
            .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
        (buffer, ptr, addr)
    };
    let _ = heap_buffer;
    println!(
        "heap: desc {desc_size} B, reserved {reserved_size} B @ {reserved_offset}, total {heap_size} B"
    );

    // ── data buffers (plain wgpu storage; fork gives them BDA usage) ────────
    let storage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let mk = |label: &str, size: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: storage,
            mapped_at_creation: false,
        })
    };
    let weights = mk("weights", (WIDTH * WIDTH * 2 + WIDTH * 2) as u64);
    let io = mk("io", (BATCH as usize * WIDTH * 4 + 65536) as u64);
    let zeros = mk("zeros", 1024);
    let w_init: Vec<half::f16> = (0..WIDTH * WIDTH + WIDTH)
        .map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.01))
        .collect();
    queue.write_buffer(&weights, 0, bytemuck::cast_slice(&w_init));
    queue.write_buffer(&io, 0, &vec![0u8; io.size() as usize]);
    queue.write_buffer(&zeros, 0, &[0u8; 1024]);

    let device_address = |buf: &wgpu::Buffer| unsafe {
        let raw = buf.as_hal::<VkApi>().expect("vk").raw_handle();
        raw_device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(raw))
    };
    for (idx, buf) in [(0u32, &weights), (1, &io), (2, &zeros)] {
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
        dst.address = unsafe { heap_ptr.add(idx as usize * desc_size).cast() };
        dst.size = desc_size;
        unsafe {
            heap_fns
                .write_resource_descriptors(&[info], &[dst])
                .expect("write descriptor");
        }
    }

    // ── the kernel, both ways ───────────────────────────────────────────────
    let spv_bytes = include_bytes!("heap_probe/heap_probe.spv");
    let words: Vec<u32> = spv_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let ptr_spv: &[u8] = include_bytes!("heap_probe/heap_probe_ptr.spv");
    let (active_spv, ptr_mode) = if args.mode == "ptr" {
        (ptr_spv, true)
    } else {
        (spv_bytes.as_slice(), false)
    };
    let shader_object = unsafe {
        let info = vk::ShaderCreateInfoEXT {
            flags: vk::ShaderCreateFlagsEXT::DESCRIPTOR_HEAP,
            stage: vk::ShaderStageFlags::COMPUTE,
            code_type: vk::ShaderCodeTypeEXT::SPIRV,
            code_size: active_spv.len(),
            p_code: active_spv.as_ptr().cast(),
            p_name: c"main".as_ptr(),
            ..Default::default()
        };
        so_fns.create_shaders(&[info], None).expect("shader object")[0]
    };
    let pipeline = unsafe {
        let module = raw_device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .expect("module");
        let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default()
            .flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default().stage(stage).push(&mut flags2);
        raw_device
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
            .expect("pipeline")[0]
    };

    // Mapped mode: classic-binding SPIR-V; every set-0 binding sources its
    // descriptor from the heap at a constant byte offset. Bindings 1/2/3 alias
    // the io descriptor, 4/5 alias zeros (offsets in bytes — per-type
    // descriptor sizes stay a host-side concern, exactly the seam's job).
    let sb_mask = vk::SpirvResourceTypeFlagsEXT::READ_ONLY_STORAGE_BUFFER
        | vk::SpirvResourceTypeFlagsEXT::READ_WRITE_STORAGE_BUFFER;
    let map_binding = |binding: u32, heap_slot: u32| {
        vk::DescriptorSetAndBindingMappingEXT::default()
            .descriptor_set(0)
            .first_binding(binding)
            .binding_count(1)
            .resource_mask(sb_mask)
            .source(vk::DescriptorMappingSourceEXT::HEAP_WITH_CONSTANT_OFFSET)
            .source_data(vk::DescriptorMappingSourceDataEXT {
                constant_offset: vk::DescriptorMappingSourceConstantOffsetEXT::default()
                    .heap_offset(heap_slot * desc_size as u32),
            })
    };
    let mappings = [
        map_binding(0, 0), // weights
        map_binding(1, 1), // io_f
        map_binding(2, 1), // io_h
        map_binding(3, 1), // io_bab
        map_binding(4, 2), // zeros_h
        map_binding(5, 2), // zeros_f
    ];
    let mapped_pipeline = unsafe {
        let mapped_spv = include_bytes!("heap_probe/heap_probe_mapped.spv");
        let mapped_words: Vec<u32> = mapped_spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let module = raw_device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&mapped_words), None)
            .expect("mapped module");
        let mut mapping_info =
            vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default().mappings(&mappings);
        let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default()
            .flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main")
            .push(&mut mapping_info);
        let info = vk::ComputePipelineCreateInfo::default().stage(stage).push(&mut flags2);
        raw_device
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
            .expect("mapped pipeline")[0]
    };

    let push = ProbePush {
        w_idx: 0,
        io_idx: 1,
        zeros_idx: 2,
        batch: BATCH,
        use_coopvec: if args.plain { 0 } else { 1 },
    };
    println!("mode {}, coopvec {}", args.mode, push.use_coopvec);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    unsafe {
        encoder.as_hal_mut::<VkApi, _, _>(|hal_encoder| {
            let cb = hal_encoder.expect("vk").raw_handle();
            let mut bind = vk::BindHeapInfoEXT::default();
            bind.heap_range = vk::DeviceAddressRangeEXT {
                address: heap_addr,
                size: heap_size,
            };
            bind.reserved_range_offset = reserved_offset;
            bind.reserved_range_size = reserved_size;
            heap_fns.cmd_bind_resource_heap(cb, &bind);

            let pre = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)];
            raw_device
                .cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&pre));

            match args.mode.as_str() {
                "so" | "ptr" => so_fns.cmd_bind_shaders(
                    cb,
                    &[vk::ShaderStageFlags::COMPUTE],
                    &[shader_object],
                ),
                "pipeline" => {
                    raw_device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline)
                }
                "mapped" => {
                    raw_device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, mapped_pipeline)
                }
                m => panic!("unknown mode {m}"),
            }
            let ptr_push = PtrPush {
                w_addr: device_address(&weights),
                out_addr: device_address(&io),
                zeros_addr: device_address(&zeros),
            };
            let mut range = vk::HostAddressRangeConstEXT::default();
            if ptr_mode {
                range.address = (&ptr_push as *const PtrPush).cast();
                range.size = core::mem::size_of::<PtrPush>();
            } else {
                range.address = (&push as *const ProbePush).cast();
                range.size = core::mem::size_of::<ProbePush>();
            }
            let mut pinfo = vk::PushDataInfoEXT::default();
            pinfo.offset = 0;
            pinfo.data = range;
            heap_fns.cmd_push_data(cb, &pinfo);
            raw_device.cmd_dispatch(cb, BATCH.div_ceil(64), 1, 1);

            let post = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
            raw_device
                .cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&post));
        });
    }
    queue.submit(Some(encoder.finish()));
    println!("submitted; waiting...");

    // Readback with a hang timeout.
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 512,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc2.copy_buffer_to_buffer(&io, 0, &staging, 0, 512);
    queue.submit(Some(enc2.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    for i in 0..200 {
        let _ = device.poll(wgpu::PollType::Poll);
        if let Ok(r) = rx.try_recv() {
            r.unwrap();
            let data = slice.get_mapped_range().to_vec();
            let vals: &[f32] = bytemuck::cast_slice(&data[..64]);
            println!("COMPLETED after {} polls; io[0..8] = {:?}", i, &vals[..8]);
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    println!("HANG: dispatch never completed (10 s)");
    std::process::exit(2);
}
