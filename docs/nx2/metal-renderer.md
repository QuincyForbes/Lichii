# Native Metal renderer (iOS and macOS)

Status: in progress on branch `feat-metal-renderer`. Vulkan (through MoltenVK on Apple) stays the
default; Metal is selected with `backend = Metal` (iOS: Settings > Graphics > Graphics API; macOS
Qt: Configure > Graphics > API "Metal (experimental)"). Only Apple builds compile it
(`ENABLE_METAL`, CMake, default ON on Apple and unavailable elsewhere).

## Why

IPA 152 on an iPhone18,2 (A19 Pro), Ocarina of Time in Kokiri Forest: 13-16 fps at 28-63 ms of
GPU time per frame. What MoltenVK costs there:

| Cost | Measured | Cause |
|---|---|---|
| ~1300 extra Metal passes per timed frame | `Perf passes:` | MoltenVK ends the render encoder at every barrier, copy, clear outside a pass and query reset, and opens a blit/compute encoder for each |
| 22-33 % of a core on the GPU back thread | `Perf threads:` | Vulkan recording, then MoltenVK re-encoding every command into Metal |
| 37-40 command buffers a frame | `Perf sites:` | Vulkan submits; each is a Metal command buffer with its own scheduling cost |
| ~75 % of uniform upload bytes are padding | `Perf:` uniform line | 64 KiB-aligned uniform ranges padded for MoltenVK's argument layout |
| 0.2-2.9 s stalls on first use of a pipeline | `Build shaders:` | SPIR-V -> MSL -> GPU code at draw time, no binary archive |

nxnative (seedds/nxnative, MIT; its Ocarina fork cheemu21/kid-with-sword-remake) drives Metal
directly with ahead-of-time shaders and a Metal 4 pipeline archive and reports 12-22 ms of GPU time
on an iPhone 18 Pro. Much of their gain is pass structure (#1430: a depth write that is not a read
keeps the pass open, 132 -> 66 passes), which a backend that owns its encoders gets by design.

## Prior art

- **nxnative** (MIT): NVN -> Metal; one render encoder per guest pass; argument buffers per
  stage; Metal 4 pipeline prebuild + offline `MTLBinaryArchive` per GPU family (`metal-tt`); the
  grass mesh pipeline replaced by a compute pre-pass writing a vertex slice plus indirect draws;
  per-frame uniform ring. Their shader chain is Ryujinx's recompiler -> SPIR-V -> SPIRV-Cross MSL,
  the same first step we take.
- **Ryujinx / Ryubing Metal backend** (MIT, C#, 1.2.80): Maxwell state -> Metal pipeline state
  (`MTLRenderPipelineDescriptor` holds blend, formats, sample count; depth/stencil a separate
  `MTLDepthStencilState`; rasterizer state set on the encoder). Encoder state is cached and
  re-applied on encoder change. It fails where Metal has no counterpart: geometry shaders,
  transform feedback, some vertex formats (emulated with compute or vertex pulling), and clip
  distances on older GPUs.
- **Hydra** (Switch 1, Metal): same split of pipeline/depth state, a texture cache with views
  for format reinterpretation, MSL generated directly.
- **Dolphin / PCSX2 Metal backends**: each keeps a "current encoder" and only ends it when the
  next command needs a different encoder type or a different attachment set; uploads are done
  with `setVertexBytes`/`setFragmentBytes` (< 4 KiB) or a per-frame ring; textures written by a
  previous pass are tracked with `MTLFence` (PCSX2) or rely on Metal's automatic hazard tracking
  for untracked heaps disabled.
- **MoltenVK** (`%TEMP%\lichii-bench\mvk-src`): keeps what is correct (format table,
  argument-buffer layouts, SPIRV-Cross options for Apple GPUs, depth-clip/viewport conversions),
  drops what costs (encoder per barrier, per-command re-encoding, 64 KiB padding,
  `VkPipelineCache` mutex).
- **Our backends**: `renderer_null` (the smallest `RasterizerInterface`) and `renderer_opengl`
  (~9 k lines, `TextureCache<P>` / `BufferCache<P>` runtimes without a scheduler) are the
  templates; `renderer_vulkan` provides the pieces we port one at a time.

## Architecture

```
VideoCore::CreateRenderer (video_core.cpp)  -- RendererBackend::Metal
  Metal::RendererMetal : RendererBase
    Device          MTLDevice, feature table (GPU family, mesh, argument buffer tier, memory)
    Scheduler       one MTLCommandQueue; the current command buffer + current encoder
    Swapchain       CAMetalLayer from WindowSystemInfo::render_surface, nextDrawable
    BlitScreen      layers -> drawable (filters, MetalFX spatial/temporal, frame gen)
    RasterizerMetal : RasterizerInterface, ChannelSetupCaches
      TextureCache<TextureCacheParams>  (VideoCommon, unchanged) + Metal::TextureCacheRuntime
      BufferCache<BufferCacheParams>    (VideoCommon, unchanged) + Metal::BufferCacheRuntime
      QueryCache                         visibility result buffer
      PipelineCache                      SPIR-V (shader_recompiler) -> MSL -> MTLLibrary
      FenceManager                       MTLSharedEvent ticks
```

Code layout: only `metal_*.mm` files touch Metal. The caches, rasterizer, staging pool and the
SPIR-V -> MSL translator are plain C++ that hold Metal objects as opaque retained handles
(`metal_object.h`) and call Metal through `metal_resources.h` (textures, views, buffers,
samplers, formats) and `metal_scheduler.h` (recording). With `-DLICHII_METAL_COMPILE_CHECK=ON`
that C++ part compiles on Linux and Windows (`video_core_metal_check`), so template errors
show before an Apple CI run.

Bindings: the buffer cache runs in its Vulkan shape (`IS_OPENGL=false`): uniform, storage and
texel buffers are appended in the order the recompiler declares them
(`BufferCacheRuntime::PendingBindings`). The translator numbers MSL `[[buffer]]`, `[[texture]]`
and `[[sampler]]` slots per kind in the same (set, binding) order and returns that table, so a
pipeline binds the list in sequence. Streamed uniforms come from a 32 MiB ring in 16 regions
(`StagingBufferPool::Stream`) with no padding past the used size.

The frontends already hand a `CAMetalLayer` to the renderer for MoltenVK
(`EmuWindow_SDL3_VK`, `QtCommon::GetWindowSystemInfo`), so Metal reuses those windows.

### Vulkan piece -> Metal piece

| Vulkan backend | Metal backend | Notes |
|---|---|---|
| `vk_instance`, `vulkan_device` | `metal_device` | `MTLCreateSystemDefaultDevice`; `supportsFamily(Apple9)`, `argumentBuffersSupport`, `recommendedMaxWorkingSetSize` (macOS) / `os_proc_available_memory` (iOS) feed the existing budgets |
| `vk_scheduler`, `vk_master_semaphore`, `vk_command_pool` | `metal_scheduler` | One command buffer per flush; the tick is an `MTLSharedEvent` value signalled at the end of each command buffer (`encodeSignalEvent`), waited with `waitUntilSignaledValue`/`notifyListener` |
| render pass begin/end, `vk_render_pass_cache` | `metal_scheduler` render state | One `MTLRenderCommandEncoder` per real attachment set; load/store actions from the clear state and whether the next pass reads the attachment |
| barriers (`vkCmdPipelineBarrier`) | own hazard tracking | Resources come from heaps with `MTLHazardTrackingModeUntracked`; a write -> read between encoders becomes one `MTLFence` update/wait (or `memoryBarrierWithScope` inside an encoder); no encoder ends for a barrier |
| `vk_staging_buffer_pool`, `vk_update_descriptor` | `metal_staging_buffer_pool`, ring buffer | Shared-storage `MTLBuffer` ring per frame; uniforms < 4 KiB use `setBytes`, larger ones a 256-byte aligned ring slice (no 64 KiB padding) |
| `vk_texture_cache` | `metal_texture_cache` (runtime) | `MTLTexture` + views (`newTextureViewWithPixelFormat:` with swizzle); format table from `maxwell_to_mtl.h`; ASTC native on Apple GPUs; BC on macOS only (iOS decodes BC on the CPU/GPU as today) |
| `vk_buffer_cache` | `metal_buffer_cache` (runtime) | Private `MTLBuffer` per cached buffer; index-format conversion (u8 -> u16) in a compute pass, as MoltenVK does but once per buffer range |
| `vk_pipeline_cache`, `vk_graphics_pipeline`, `vk_compute_pipeline` | `metal_pipeline_cache` | Key = the same `FixedPipelineState` hash; `MTLRenderPipelineState` + `MTLDepthStencilState` + encoder state (cull, winding, depth bias, viewport, scissor) |
| `vk_descriptor_pool`, descriptor sets | argument buffers (tier 2) / direct binds | First step: direct `setTexture/setBuffer/setSamplerState` with our own binding numbers (fewer calls than MoltenVK's per-set rebinds); then one argument buffer per stage when a stage binds more than ~8 resources |
| `vk_query_cache` | `metal_query_cache` | Occlusion: `setVisibilityResultMode` into a visibility buffer, resolved after the command buffer completes; timestamps from `MTLCounterSampleBuffer` at encoder boundaries (no extra passes) |
| `vk_fence_manager` | `metal_fence_manager` | Guest fences = scheduler ticks |
| `vk_swapchain`, `vk_present_manager` | `metal_swapchain` | `CAMetalLayer` (`maximumDrawableCount` 3, `displaySyncEnabled` from vsync on macOS), `presentDrawable` on the frame's command buffer; present thread not needed |
| `vk_blit_screen`, `present/*` | `metal_blit_screen` | Same layers and filters, MSL versions of the present shaders |
| `present/metalfx_*` | direct | MetalFX takes our `MTLTexture`s and our command buffer: no `VK_EXT_metal_objects`, no shared-event bridge |
| `metal_gpu_time.mm` | `MTLCommandBuffer GPUStartTime/GPUEndTime` | Same `Perf:` GPU ms; `Perf passes:` from counter samples per encoder |
| mesh pipelines (MoltenVK mesh fork / compute emulation) | Metal mesh pipelines on Apple9+/M3+ via `MTLMeshRenderPipelineDescriptor`, compute pre-pass otherwise | The compute pre-pass is preferred for the grass (nxnative: occupancy) |
| transform feedback | compute vertex capture | Only where a game needs it |

### Shaders

1. **First (M3):** the shader recompiler's SPIR-V backend (already the input to MoltenVK) ->
   SPIRV-Cross MSL (`spirv_msl.hpp`, CPM package `spirv-cross`, Apple only) with *our* resource
   binding layout: every SPIR-V `(set, binding)` gets an explicit `[[buffer(n)]]`/`[[texture(n)]]`/
   `[[sampler(n)]]` index from `Shader::Info`, so the rasterizer binds directly and no
   descriptor-set emulation is needed. MSL options: `msl_version` 3.0 (iOS 17), `ios` platform on
   iOS, `use_framebuffer_fetch_subpasses`, `enable_decoration_binding`, fast math off where the
   SPIR-V has `NoContraction`.
2. **Then:** `newLibraryWithSource:` on a worker pool, `MTLBinaryArchive` per game
   (`<cache>/<title>/metal.archive`): every pipeline built at run time is added to it and
   serialized at shutdown and every 60 s; boot loads it before the first draw.
3. **Ahead of time (M5):** a run with `EDEN_NX2_METAL_RECORD=1` writes the pipeline keys and MSL;
   `tools/nx2/metal_pack.py` builds them with `xcrun metal` + `metal-tt` into a pack for one GPU
   family (applegpu_g17p for the A19 Pro), loaded at boot so no draw waits for a compile.
4. **Later, if it pays:** a native MSL emitter in `shader_recompiler/backend/msl` (no SPIR-V
   round trip; native simdgroup ops, float atomics).

### Encoders and passes

- The scheduler owns one current encoder. A render encoder ends only when the attachment set
  changes, a blit/compute must run outside it, or the command buffer is flushed.
- Clears inside a pass become load actions when they are the first command, otherwise a
  full-screen clear draw (no pass split).
- Copies and uploads go to one blit encoder before the next render encoder (batched per pass
  boundary), not one encoder each.
- Write -> read hazards between encoders: `MTLFence` (render stage granularity); inside a
  render encoder, `memoryBarrierWithScope:afterStages:beforeStages:` (Apple GPU family 6+).
- The `32768x32768` targetless occlusion pass: a render encoder with
  `renderTargetWidth/Height` from the scissor reach (the clamp feat-ios-gpu-clamp made for Vulkan).

### Uniform data

Guest constant buffers are uploaded per draw today. Metal path: `set{Vertex,Fragment}Bytes` for
ranges <= 4 KiB, else a slice of a per-frame shared ring aligned to 256 B (Apple) bound with
`setBuffer:offset:` (offset-only rebinds are cheap). No 64 KiB padding.

### Memory, heaps and residency

- Textures and buffers from `MTLHeap`s (private, untracked hazards) grown in 64 MiB steps on
  macOS, 16 MiB on iOS; staging rings in shared storage.
- The existing budgets (`host_budget.h`, device profiles) read `Device::MemoryBudget()`.
- macOS 15 / iOS 18 residency sets (`MTLResidencySet`) when available, so per-encoder
  `useResource` calls go away.

### Synchronisation with the split GPU thread

The split GPU thread (`gpu_split.cpp`) is enabled only for Vulkan. The Metal rasterizer starts on
the single GPU thread; the split is turned on for Metal once the Metal scheduler records on the
back thread (M4). Command buffer completion handlers only advance the tick; all cache work stays
on the GPU thread.

### Presentation

`CAMetalLayer` from the frontend (`WindowSystemInfo::render_surface`): `device`, `pixelFormat`
BGRA8Unorm (BGRA10_XR for HDR later), `framebufferOnly` YES, `drawableSize` from the framebuffer
layout, `maximumDrawableCount` 3 (2 with frame pacing on iPhone). The present command buffer
draws the guest layers into the drawable and calls `presentDrawable:`.

## Milestones

| | What | What the owner sees |
|---|---|---|
| M0 | This document | - |
| M1 | `RendererBackend::Metal`, `renderer_metal/` skeleton (device, queue, CAMetalLayer swapchain, a rasterizer that accepts work like the null one, cleared frame presented), selectable on iOS and in Qt on macOS, macOS CI job + offscreen smoke test | Picking Metal boots the game with a dark-blue screen (the game runs, nothing is drawn); `Metal:` log lines name the GPU |
| M2 | Texture/buffer cache runtimes, uploads/downloads, blits, present of the guest framebuffer | Menus/title screens of games that draw with copies and CPU uploads; every framebuffer the game presents is shown |
| M3 | Graphics pipelines (SPIRV-Cross MSL), grouped render encoders, draws, state mapping | Ocarina of Time title screen and Kokiri Forest drawing |
| M4 | Compute, queries, transform feedback, mesh emulation, OoT upscale replacement, MetalFX direct, frame gen, `Perf:`/`Perf passes:` | Kokiri at full quality; GPU ms comparable with MoltenVK |
| M5 | Binary archive + ahead-of-time pack, MoltenVK-vs-Metal comparison on device | No compile stalls; the comparison table |

State (2026-11-15, d7bfa1971f): M0-M3 done (vertex + pixel pipelines); from M4: compute
dispatches, indirect draws, texel buffers (unaligned ones through aligned copies, written ones
copied back), index rewriting (quads, quad strips, fans, polygons, line loops, 8-bit indices; on
the GPU when the GPU wrote them), occlusion queries, task and mesh pipelines through Vulkan's
mesh emulation, all presented layers with blending, the GPU thread split, MetalFX spatial,
MetalFX temporal in place of a game's DLSS, Ocarina's upscale dispatch as Vulkan's bilinear pass,
resolution scaling (depth and stencil carried over), tessellation as MoltenVK runs it (vertex
program as a stage-in kernel, control program as a multi-patch kernel, evaluation program as the
post-tessellation vertex function), the `Metal perf:` line; from M5 a per-game MTLBinaryArchive
of the pipelines a run builds. The macOS CI smoke test runs the translator, a rewritten-index draw
and an indirect draw through the renderer's pipeline builder and render pass with occlusion
counts, a texel buffer read, the mesh emulation's draw side and distribution pass, the GPU index
rewrite against the CPU one, the Ocarina bilinear pass, the pipeline archive across a reopen, the
depth and depth-stencil blits, and a tessellated triangle through all three stages. Geometry
programs are refused as on MoltenVK. Open: indirect tessellated draws, Ocarina's live MetalFX
temporal path, frame generation, the M5 AOT pack.
Switches: `EDEN_NX2_METAL_DRAW=0` (build pipelines, draw nothing), `EDEN_NX2_METAL_DISPATCH=0`,
`EDEN_NX2_METAL_SPLIT=0` (one GPU thread), `EDEN_NX2_METAL_GPU_INDICES=1|0` (every indexed
rewrite on the GPU / none), `EDEN_NX2_METAL_ARCHIVE=0` (no pipeline archive).

## What can be checked where

- Windows/Linux CI: nothing of Metal is compiled; the enum value and frontend hooks are.
- macOS CI runner (Apple-silicon VM with a paravirtual Metal device): build, the offscreen smoke
  test (`lichii-metal-smoke`: clear + a test draw into a texture, read back, compare).
- Only a real Mac / the iPhone: presenting to a `CAMetalLayer`, game rendering, performance.
