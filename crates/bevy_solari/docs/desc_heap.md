 Push data by itself is just bytes. What gives it meaning is that it's one of five storage locations the mapping table
  can source a binding from. Lay out DescriptorMappingSourceEXT as a matrix and the whole extension collapses into one
  idea:
 
  ┌───────────────────────┬────────────────────────┬────────────────────────────┬───────────────────────────────────┐
  │                       │      inline data       │       device address       │            heap index             │
  ├───────────────────────┼────────────────────────┼────────────────────────────┼───────────────────────────────────┤
  │ compile-time constant │ —                      │ —                          │ HEAP_WITH_CONSTANT_OFFSET (0)     │
  ├───────────────────────┼────────────────────────┼────────────────────────────┼───────────────────────────────────┤
  │ push data             │ PUSH_DATA (5)          │ PUSH_ADDRESS (6)           │ HEAP_WITH_PUSH_INDEX (1)          │
  ├───────────────────────┼────────────────────────┼────────────────────────────┼───────────────────────────────────┤
  │ indirect buffer       │ —                      │ INDIRECT_ADDRESS (7)       │ HEAP_WITH_INDIRECT_INDEX (2/3)    │
  ├───────────────────────┼────────────────────────┼────────────────────────────┼───────────────────────────────────┤
  │ SBT shader record     │ SHADER_RECORD_DATA (9) │ SHADER_RECORD_ADDRESS (10) │ HEAP_WITH_SHADER_RECORD_INDEX (8) │
  ├───────────────────────┼────────────────────────┼────────────────────────────┼───────────────────────────────────┤
  │ resource heap itself  │ RESOURCE_HEAP_DATA (4) │ —                          │ —                                 │
  └───────────────────────┴────────────────────────┴────────────────────────────┴───────────────────────────────────┘

 Three access modes × where the bytes live. That's the entire model. The shader-record row you asked about two messages ago is the 
  same three mechanisms pointed at the SBT instead of push data — which is why your plan's structure is right: global args go in the
  push-data row, per-material args go in the shader-record row, and neither requires different shader code.

  Your dispatch-layer bullet maps straight onto row two:

  TLAS address        -> PUSH_ADDRESS       (or push the u64 and use it directly in Slang)
  camera BDA pointer  -> PUSH_ADDRESS
  output BDA pointer  -> PUSH_ADDRESS
  heap bases          -> HEAP_WITH_PUSH_INDEX
  per-frame scalars   -> PUSH_DATA          (frame index, jitter, exposure)

  What a frame looks like

  let heap = ext::descriptor_heap::Device::new(&instance, &device);

  // once per command buffer
  heap.cmd_bind_resource_heap(cmd, &resource_heap_info);
  heap.cmd_bind_sampler_heap(cmd, &sampler_heap_info);
      
  // per dispatch — no layout, no stage mask, no descriptor sets
  let block = FrameArgs { tlas: tlas_addr, camera: cam_addr, out: out_addr, frame: n };
  heap.cmd_push_data(cmd, &vk::PushDataInfoEXT::default()
      .offset(0)
      .data(vk::HostAddressRangeConstEXT::default() /* ptr + size of block */));

  rt.cmd_trace_rays(cmd, &raygen_sbt, &miss_sbt, &hit_sbt, &call_sbt, w, h, 1);

  The full command set is small — write_sampler_descriptors, write_resource_descriptors, cmd_bind_sampler_heap,
  cmd_bind_resource_heap, cmd_push_data, plus capture/replay and border-color helpers. That's the whole binding API.


    Two things worth knowing
  
  RESOURCE_HEAP_DATA (4) is the escape hatch when you blow maxPushDataSize: stash the argument block in the resource heap
  and source bindings from there. Between that and PUSH_ADDRESS, push data itself should stay tiny — a handful of
  addresses and indices, not payloads.
  
  NVIDIA has a push-data bank extension. Your ash tree has PushConstantBankInfoNV { bank: u32 }, which extends
  PushDataInfoEXT, DescriptorSetAndBindingMappingEXT, PushConstantsInfo, and IndirectCommandsLayoutTokenEXT, with
  properties max_graphics_push_data_banks / max_compute_push_data_banks. The shape strongly implies you can keep several
  argument blocks resident and switch banks instead of re-pushing — potentially useful if you end up pushing per-pass
  blocks in a tight loop. I'm reading that off the ash signatures only; I haven't read that spec, so treat it as a lead
  rather than a description.