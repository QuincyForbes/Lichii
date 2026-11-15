# Metal coverage matrix

What the Vulkan renderer does, and whether the native Metal renderer (`backend = Metal`, opt-in)
does it too. Kept current with the code: every change to `renderer_metal/` that closes or opens a
gap updates its row. `python -I tools/nx2/metal_coverage.py` counts the rows and checks every
file reference.

- Vulkan references are relative to `src/video_core/renderer_vulkan/`, Metal ones to
  `src/video_core/renderer_metal/` (`../` leaves the backend's folder for shared code). A line
  number names the function or block that holds the behaviour, not one statement in it.
- **Status** is one of: `native` (Metal's own form of the feature), `emulated` (the same result
  by other means; how is in brackets), `missing` (absent, wrong or approximated; what happens
  instead is in brackets), `n/a` (nothing to do on Metal; why is in brackets).
- **Tested**: a `lichii-metal-smoke` case (`metal_smoke.mm`, run on a Metal device by
  `metal.yml`), `probe:<name>` (a descriptor Metal's validation may abort on, run in a child
  process, `metal_smoke_probe.mm`), `replay` (the offline replay of the game's pipelines on the
  PC), or `untested`. Nothing here was run on an iPhone.
- **OoT**: whether The Legend of Zelda: Ocarina of Time (0400486024DEC000) uses it: `yes`, `no`,
  or `?` where no measurement says. Source: the feature census (`EDEN_NX2_FEATURE_CENSUS=1`,
  `tools/nx2/feature_census.py`) of a Vulkan run on the PC, and the device logs of earlier builds.
  The census walk went from boot through the title, file select, Kokiri Forest, a dialogue, the
  pause menu and a house to an item scene (26.5 million draws); it has no movie. `no` means not
  seen in that walk. The walk drew no tessellated patch, but a device log shows the game building
  a tessellation pipeline, so those rows stay `yes`.
- "100 %" means: no row with `missing` and OoT `yes`, and every other gap listed with its reason.
- Sections A to P are the capabilities. Section Q lists the `EDEN_NX2_*` switches that change
  what Vulkan renders or how, each with the capability row it belongs to; it is counted apart, as
  is section R (what only Metal needs).

## A. Rasterizer entry points (`RasterizerInterface`)

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| A01 | `Draw`: arrays, indexed (16/32-bit), instanced, base vertex, base instance | vk_rasterizer.cpp:1260 | native | metal_rasterizer.cpp:369 | RunRendererDraw, replay | yes |
| A02 | `DrawIndirect`: one draw, arguments in a buffer | vk_rasterizer.cpp:1630 | native | metal_rasterizer.cpp:1479 | untested | no |
| A03 | `DrawIndirect`: several draws from one buffer | vk_rasterizer.cpp:1630 | emulated (one Metal indirect draw per guest draw: Metal has no multi-draw) | metal_rasterizer.cpp:1545 | untested | no |
| A04 | `DrawIndirect`: draw count from a buffer | vk_rasterizer.cpp:1630 | missing (the count is read from guest memory on the CPU: Metal has no count buffer; stale when the GPU wrote it and it was not downloaded) | metal_rasterizer.cpp:1548 | untested | no |
| A05 | `DrawIndirect`: transform feedback byte count | vk_rasterizer.cpp:1630 | missing (draw refused: "transform feedback draw") | metal_rasterizer.cpp:1502 | untested | no |
| A06 | `DrawIndirect` of a mesh pipeline | vk_rasterizer.cpp:1630 | missing (draw refused) | metal_rasterizer.cpp:1509 | untested | no |
| A07 | `DrawIndirect` of a tessellated pipeline | vk_rasterizer.cpp:1630 | missing (draw refused) | metal_rasterizer.cpp:1514 | untested | no |
| A08 | `DrawIndirect` with 8-bit indices | vk_rasterizer.cpp:1630 | missing (draw refused) | metal_rasterizer.cpp:1522 | untested | no |
| A09 | `DrawIndirect` of quads, fans, loops and other rewritten topologies | vk_rasterizer.cpp:1630 | missing (draw refused) | metal_rasterizer.cpp:1522 | untested | no |
| A10 | `DrawMeshTasksFromGpu`: mesh draws whose task count the GPU wrote, drawn without reading it back | vk_rasterizer.cpp:1588 | missing (interface default `false`: the count is read on the CPU and the draw comes through `Draw`) | - | untested | ? |
| A11 | `DrawTexture` | vk_rasterizer.cpp:1909 | missing (draw refused) | metal_rasterizer.cpp:1572 | untested | ? |
| A12 | `Clear`: whole colour, depth and stencil attachments | vk_rasterizer.cpp:1962 | native (load action of the next pass) | metal_rasterizer.cpp:1609, metal_scheduler.mm:555 | replay | yes |
| A13 | `Clear` inside a scissor rectangle | vk_rasterizer.cpp:1962 | missing (whole target cleared, counted as refused; the game's clears have the scissor test on, whether any rectangle is smaller than its target the census does not say) | metal_rasterizer.cpp:1609 | untested | ? |
| A14 | `Clear` of some colour channels | vk_rasterizer.cpp:1962 | missing (clear skipped, counted as refused) | metal_rasterizer.cpp:1609 | untested | no |
| A15 | `Clear` of some stencil bits (write mask) | vk_rasterizer.cpp:1962 | missing (all bits cleared, counted as refused) | metal_rasterizer.cpp:1609 | untested | no |
| A16 | `Clear` of chosen layers (`layer_count`, `clear_surface.layer`) | vk_rasterizer.cpp:1962 | missing (the attachment's bound layer is cleared) | metal_rasterizer.cpp:1609 | untested | no |
| A17 | `Clear` values of integer targets (float register bits converted) | vk_rasterizer.cpp:1962 | missing (the float values are passed as they are) | metal_scheduler.mm:555 | untested | ? |
| A18 | `DispatchCompute` with a grid in the launch description | vk_rasterizer.cpp:2238 | native | metal_rasterizer.cpp:1685, metal_scheduler.mm:887 | RunTranslate, RunTexelBuffer, replay | yes |
| A19 | `DispatchCompute` with the grid in a buffer | vk_rasterizer.cpp:2238 | native | metal_rasterizer.cpp:1685, metal_scheduler.mm:896 | replay | yes |
| A20 | `BindGraphicsUniformBuffer`, `DisableGraphicsUniformBuffer` | vk_rasterizer.cpp:2542 | native | metal_rasterizer.cpp:1988 | RunRendererDraw | yes |
| A21 | `FlushAll` | vk_rasterizer.cpp:2551 | n/a (empty on Vulkan too) | metal_rasterizer.cpp:1998 | - | no |
| A22 | `FlushRegion`: texture and buffer downloads | vk_rasterizer.cpp:2587 | native | metal_rasterizer.cpp:2000 | untested | yes |
| A23 | `FlushRegion`: collected CPU writes applied before the download (`gpu.InvalidateGPUCache`) | vk_rasterizer.cpp:2587 | native (under `EDEN_NX2_METAL_CACHE_PATHS`) | metal_rasterizer.cpp:2000 | untested | yes |
| A24 | `FlushRegion` / `InvalidateRegion` of the query cache | vk_rasterizer.cpp:2587 | n/a (Metal's reports are written when their fence signals; no cached reports) | - | - | no |
| A25 | `MustFlushRegion` | vk_rasterizer.cpp:2623 | native | metal_rasterizer.cpp:2019 | untested | yes |
| A26 | `MustFlushRegion` / `GetFlushArea` skipping the caches' locks for pages nothing wrote | vk_rasterizer.cpp:2623, vk_rasterizer.cpp:2668 | native (under `EDEN_NX2_METAL_CACHE_PATHS`) | metal_rasterizer.cpp:2019, metal_rasterizer.cpp:2041 | untested | yes |
| A27 | `GetFlushArea` | vk_rasterizer.cpp:2668 | native | metal_rasterizer.cpp:2041 | untested | yes |
| A28 | `LandDownloadForCpuRead`: an in-flight download landed on the reading core | vk_rasterizer.cpp:2710 | native (under `EDEN_NX2_METAL_CACHE_PATHS`; `EDEN_NX2_GPU_READ_LAND` as on Vulkan) | metal_rasterizer.cpp:2081, metal_scheduler.mm:388 | untested | yes |
| A29 | `PrepareRegionWrite` | vk_rasterizer.cpp:2723 | native (under `EDEN_NX2_METAL_CACHE_PATHS`) | metal_rasterizer.cpp:2092 | untested | ? |
| A30 | `InvalidateRegion` | vk_rasterizer.cpp:2731 | native | metal_rasterizer.cpp:2101 | untested | yes |
| A31 | `InnerInvalidation` (the GPU's own cached writes, batched, page filter) | vk_rasterizer.cpp:2751 | native (under `EDEN_NX2_METAL_CACHE_PATHS`) | metal_rasterizer.cpp:2118 | untested | ? |
| A32 | `OnCPUWrite` | vk_rasterizer.cpp:2818 | native | metal_rasterizer.cpp:2154 | untested | yes |
| A33 | `OnCacheInvalidations` (batch, CPU write reclaim) | vk_rasterizer.cpp:2873 | native (under `EDEN_NX2_METAL_CACHE_PATHS`; `EDEN_NX2_CPU_WRITE_RECLAIM` as on Vulkan) | metal_rasterizer.cpp:2213 | untested | yes |
| A34 | `OnCacheInvalidation` | vk_rasterizer.cpp:2912 | native | metal_rasterizer.cpp:2286 | untested | yes |
| A35 | `TryCacheInvalidation` (invalidation on the writing core) | vk_rasterizer.cpp:2928 | native (under `EDEN_NX2_METAL_CACHE_PATHS`) | metal_rasterizer.cpp:2254 | untested | ? |
| A36 | `InvalidateGPUCache`, `UnmapMemory` | vk_rasterizer.cpp:2944 | native | metal_rasterizer.cpp:2301 | untested | yes |
| A37 | `ModifyGPUMemory`: texture cache | vk_rasterizer.cpp:2960 | native | metal_rasterizer.cpp:2317 | untested | yes |
| A38 | `ModifyGPUMemory`: buffer cache (`UnmapGPUMemory`) | vk_rasterizer.cpp:2960 | n/a (the shared buffer cache keeps virtual ranges only for runtimes with `BindMultiRangeStorageBuffer`, which is Vulkan's alone; the call is made and does nothing) | metal_rasterizer.cpp:2317 | - | no |
| A39 | `SignalFence`, `SyncOperation`, `SignalSyncPoint`, `SignalReference`, `ReleaseFences` | vk_rasterizer.cpp:2968 | native | metal_rasterizer.cpp:2326, metal_fence_manager.cpp:12 | untested | yes |
| A40 | `FencesQuiet`, `FenceWorkEpoch` (the puller skips fence releases when nothing is pending) | vk_rasterizer.cpp:2988 | native (under `EDEN_NX2_METAL_CACHE_PATHS`) | metal_rasterizer.cpp:2277 | untested | yes |
| A41 | `FlushQueryResults`, `HasPendingQueries`, `HasCachedReports` | vk_rasterizer.cpp:2996, vk_rasterizer.cpp:2660 | n/a (reports are written when their fence signals; nothing is cached) | - | - | no |
| A42 | `ReadGpuWrittenCommandList`, `NoteGpuWrittenCommandListSynced`, `FlushGpuWrittenCommandListPages` (command lists the GPU wrote, read from copies) | vk_rasterizer.cpp:2562 | missing (interface defaults: `Unknown`, the list is read after a full sync) | - | untested | ? |
| A43 | `FlushAndInvalidateRegion` | vk_rasterizer.cpp:3017 | native | metal_rasterizer.cpp:2346 | untested | ? |
| A44 | `WaitForIdle`: later work sees earlier writes | vk_rasterizer.cpp:3025 | native (Metal orders the encoders of a queue and tracks hazards itself) | metal_rasterizer.cpp:2354 | untested | yes |
| A45 | `FragmentBarrier`: the next draw may read what this pass wrote | vk_rasterizer.cpp:3078 | missing (nothing: a draw sampling its own target is not split from its pass) | metal_rasterizer.cpp:2358 | untested | yes |
| A46 | `TiledCacheBarrier` | vk_rasterizer.cpp:3084 | n/a (empty on Vulkan too) | metal_rasterizer.cpp:2360 | - | no |
| A47 | `FlushCommands` | vk_rasterizer.cpp:3088 | native | metal_rasterizer.cpp:2362 | untested | yes |
| A48 | `TickFrame` | vk_rasterizer.cpp:3831 | native | metal_rasterizer.cpp:2368 | untested | yes |
| A49 | `AccelerateConditionalRendering` | vk_rasterizer.cpp:4672 | missing (interface default `false`: the condition is read on the CPU) | - | untested | yes |
| A50 | `HasDrawTransformFeedback` | vk_rasterizer.cpp:4684 | missing (interface default `false`) | - | untested | no |
| A51 | `AccelerateSurfaceCopy` (Fermi 2D) | vk_rasterizer.cpp:4688 | native | metal_rasterizer.cpp:2389 | untested | ? |
| A52 | `AccessAccelerateDMA` | vk_rasterizer.cpp:4719 | native | metal_rasterizer.cpp:2405 | untested | ? |
| A53 | `AccelerateInlineToMemory` | vk_rasterizer.cpp:4723 | native | metal_rasterizer.cpp:2409 | untested | ? |
| A54 | `AccelerateInlineToMemory` fast paths (one-page writes without locks, deferred and merged invalidations) | vk_rasterizer.cpp:4723 | native (under `EDEN_NX2_METAL_CACHE_PATHS`) | metal_rasterizer.cpp:2461 | untested | ? |
| A55 | `AccelerateDisplay` | vk_rasterizer.cpp:4905 | native | metal_rasterizer.cpp:2610 | untested | yes |
| A56 | `LoadDiskResources`: pipelines of earlier runs built before play | vk_rasterizer.cpp:4965 | missing (only the binary archive is opened; programs are translated at first use) | metal_rasterizer.cpp:2579 | RunPipelineArchive | yes |
| A57 | `InitializeChannel`, `BindChannel`, `ReleaseChannel` | vk_rasterizer.cpp:6229 | native | metal_rasterizer.cpp:2585 | untested | yes |
| A58 | `SupportsGameDlssNetwork`, `UsesNvidiaDlss`, `CanProbeGameDlssOutput`, `ProbeGameDlssOutput` | vk_rasterizer.cpp:4695 | n/a (NVIDIA DLSS does not exist on Apple devices; the interface defaults say so) | - | - | no |
| A59 | `OotNetworkBypass` (the game's upscale network replaced, `oot_network_bypass`) | vk_rasterizer.cpp:1728 | missing (interface default: the game's dispatches run as they are, or O01 takes the one upscale dispatch) | - | untested | yes |
| A60 | GPU-sized dispatches: the size's words copied into the constant buffer on the GPU | vk_rasterizer.cpp:4723 | native (under `EDEN_NX2_METAL_CACHE_PATHS`) | metal_rasterizer.cpp:2409 | untested | yes |
| A61 | Draws whose vertex count GPU work wrote (`EDEN_NX2_INDIRECT_DRAWS=1`, off by default) | vk_rasterizer.cpp:1262 | emulated (the count is read back on the CPU, which waits for the GPU; Vulkan draws most of them with the count read on the GPU) | metal_rasterizer.cpp:249 | untested | no |
| A62 | Draws of more than 16 million vertices skipped | vk_rasterizer.cpp:1342 | native (under `EDEN_NX2_METAL_PIPELINE_CHECK`) | metal_rasterizer.cpp:249 | untested | no |
| A63 | Waiting sync requests of the guest threads served after every draw | vk_rasterizer.cpp:753 | native (under `EDEN_NX2_METAL_CACHE_PATHS`) | metal_rasterizer.cpp:369 | untested | yes |
| A64 | A draw whose native program failed marks the frame (`native_execution_failed`) | vk_rasterizer.cpp:759 | native | metal_rasterizer.cpp:1583 | untested | ? |
| A65 | Donkey Kong Bananza's reset program: its indirect records written on the host (`EDEN_NX2_RESETFIX`) | vk_rasterizer.cpp:677 | native (under `EDEN_NX2_METAL_CACHE_PATHS`) | metal_rasterizer.cpp:292 | untested | no |
| A66 | Caches invalidated before a draw or dispatch uses them (`EDEN_NX2_INVALIDATE_BEFORE_DISPATCH`, off by default) | vk_rasterizer.cpp:109 | missing (an experiment of the Vulkan backend; not present) | - | untested | no |

## B. Pipeline state

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| B01 | Colour attachment formats, up to 8 targets | vk_graphics_pipeline.cpp, fixed_pipeline_state.h | native | metal_rasterizer.cpp:1058, metal_graphics.mm:316 | RunRendererDraw, replay | yes |
| B02 | Depth and stencil attachment formats | vk_graphics_pipeline.cpp, fixed_pipeline_state.h | native | metal_rasterizer.cpp:1058 | RunDepthStencilBlit, replay | yes |
| B03 | Draws without any attachment (targetless, sized by the scissor) | vk_rasterizer.cpp:745, vk_graphics_pipeline.cpp:75 | native | metal_rasterizer.cpp:72, metal_scheduler.mm:682 | probe:no-attachments, probe:draw-without-attachments, replay | yes |
| B04 | Sample count of the pipeline (`msaa_mode`) | fixed_pipeline_state.h, maxwell_to_vk.cpp | native | metal_rasterizer.cpp:1058 | probe:samples-4 | no |
| B05 | Colour write masks, per target or common | vk_rasterizer.cpp:6067 | native | metal_rasterizer.cpp:1058 | replay | yes |
| B06 | Blend enable, equations and factors, per target or common | vk_rasterizer.cpp:5972, maxwell_to_vk.cpp | native | metal_rasterizer.cpp:1058, metal_graphics.mm:24 | replay | yes |
| B07 | Dual-source blend factors | maxwell_to_vk.cpp | native | metal_graphics.mm:43 | untested | ? |
| B08 | Blending switched off on integer targets | vk_graphics_pipeline.cpp | native | metal_graphics.mm:316 | probe:blending-on-rgba8uint | ? |
| B09 | Blend constants | vk_rasterizer.cpp:5499 | native | metal_rasterizer.cpp:1335 | untested | ? |
| B10 | Logic operations | vk_rasterizer.cpp:5837, vk_rasterizer.cpp:5962 | missing (Metal has no logic operation; the draw blends as if it were off) | - | untested | no |
| B11 | Alpha test | fixed_pipeline_state.h | native (in the fragment program, as on Vulkan) | metal_pipeline_translate.cpp:188 | untested | no |
| B12 | Alpha to coverage | vk_rasterizer.cpp:5866 | native | metal_rasterizer.cpp:1058 | probe:alpha-to-coverage-on-rgba8uint | ? |
| B13 | Alpha to one | vk_rasterizer.cpp:5885 | missing (not set) | - | untested | ? |
| B14 | Depth test, depth write, depth compare function | vk_rasterizer.cpp:5675, vk_rasterizer.cpp:5905 | native | metal_rasterizer.cpp:1140, metal_graphics.mm:458 | RunRendererDraw, replay | yes |
| B15 | Depth bounds test | vk_rasterizer.cpp:5509, vk_rasterizer.cpp:5657 | missing (Metal has no depth bounds test; not applied) | - | untested | ? |
| B16 | Depth bias (constant, slope, clamp) | vk_rasterizer.cpp:5454, vk_rasterizer.cpp:5803 | native | metal_rasterizer.cpp:1427 | untested | yes |
| B17 | Depth clamp (clip off) | vk_rasterizer.cpp:5849 | native (`setDepthClipMode`) | metal_rasterizer.cpp:1421 | untested | ? |
| B18 | Depth range -1..1 (`ndc_minus_one_to_one`) | fixed_pipeline_state.h | native (converted by the vertex program, as on Vulkan without the extension) | metal_pipeline_translate.cpp:179 | replay | yes |
| B19 | Stencil test: functions, operations, compare and write masks, two-sided | vk_rasterizer.cpp:5517, vk_rasterizer.cpp:5928, vk_rasterizer.cpp:6081 | native | metal_rasterizer.cpp:1140, metal_graphics.mm:458 | untested | no |
| B20 | Stencil reference values, front and back | vk_rasterizer.cpp:5517 | native | metal_rasterizer.cpp:1335 | untested | no |
| B21 | Cull mode front or back, front face (with the window origin flip) | vk_rasterizer.cpp:5642, vk_rasterizer.cpp:5914 | native | metal_rasterizer.cpp:1416 | untested | yes |
| B22 | Cull mode front and back | vk_rasterizer.cpp:5642 | emulated (triangles are not drawn at all; points and lines are) | metal_rasterizer.cpp:1407 | untested | ? |
| B23 | Polygon mode line | maxwell_to_vk.cpp | missing (drawn filled) | - | untested | no |
| B24 | Polygon mode point | maxwell_to_vk.cpp | missing (drawn filled; Metal has no point fill mode) | - | untested | no |
| B25 | Rasterizer discard | vk_rasterizer.cpp:5723 | native | metal_rasterizer.cpp:1058 | probe:no-attachments-no-rasterization | ? |
| B26 | Viewport 0: rectangle, depth range, y flip by window origin | vk_rasterizer.cpp:5295 | native | metal_rasterizer.cpp:1346 | probe:viewport-negative-height | yes |
| B27 | Viewports 1-15 | vk_rasterizer.cpp:5295 | missing (viewport 0 only; a program that writes the viewport index draws into viewport 0) | metal_rasterizer.cpp:1346 | untested | ? |
| B28 | Viewport swizzles other than a negated y | maxwell_to_vk.cpp | missing (ignored) | metal_rasterizer.cpp:1346 | untested | ? |
| B29 | Scissor 0 | vk_rasterizer.cpp:5379 | native | metal_rasterizer.cpp:1379 | probe:scissor-past-the-target, probe:scissor-empty | yes |
| B30 | Scissors 1-15 | vk_rasterizer.cpp:5379 | missing (scissor 0 only) | metal_rasterizer.cpp:1379 | untested | ? |
| B31 | Line width | vk_rasterizer.cpp:5630 | missing (Metal draws lines 1 pixel wide) | - | untested | no |
| B32 | Line stipple, smooth lines, line rasterization mode | vk_rasterizer.cpp:5752, vk_rasterizer.cpp:5766 | missing (Metal has neither; plain lines) | - | untested | no |
| B33 | Provoking vertex last | fixed_pipeline_state.h | missing (Metal's is always the first vertex; flat inputs take the wrong vertex. The game sets last on every draw; MoltenVK has no VK_EXT_provoking_vertex either, so the phone's Vulkan picture is the same. Candidate: rotated indices for programs with flat inputs) | - | untested | yes |
| B34 | Conservative rasterization | vk_rasterizer.cpp:5736 | missing (Metal has none; rasterized normally) | - | untested | no |
| B35 | Primitive restart enabled | vk_rasterizer.cpp:5701 | native (Metal always restarts strips at the all-ones index) | metal_scheduler.mm:951 | RunIndexRewrite | yes |
| B36 | Primitive restart disabled while the all-ones index is drawn | vk_rasterizer.cpp:5701 | missing (Metal restarts all the same) | - | untested | ? |
| B37 | Point size from the register, point size written by the program | fixed_pipeline_state.h | native (fixed state of the translated program) | metal_pipeline_translate.cpp:122 | replay | yes |
| B38 | y negate (`y_negate`), early depth test (`early_z`) | fixed_pipeline_state.h | native | metal_pipeline_translate.cpp:179, metal_pipeline_translate.cpp:188 | replay | yes |
| B39 | Transform feedback state (`xfb_state`) | vk_rasterizer.cpp:5266 | missing (see section K) | - | untested | no |
| B40 | Dynamic state (extended dynamic state 1-3) | vk_rasterizer.cpp:5157 | n/a (all of it is encoder state on Metal; that every draw sets it again is P11) | metal_scheduler.mm:746 | - | no |
| B41 | Pipelines built off the draw's thread (worker threads, pipeline libraries, translation threads) | vk_pipeline_cache.cpp:1112, vk_pipeline_translate_queue.cpp, vk_pipeline_library.cpp | missing (programs are translated and render states built on the draw's thread; the draw waits) | metal_pipeline_cache.cpp:214 | untested | yes |
| B42 | Pipeline binaries kept between runs | vk_pipeline_cache.cpp | native (`MTLBinaryArchive`) | metal_archive.mm:30 | RunPipelineArchive | yes |

## C. Vertex input

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| C01 | Attributes, streams, strides, offsets, per-instance streams with divisors | vk_rasterizer.cpp:6122 | native | metal_rasterizer.cpp:1058, metal_graphics.mm:316 | RunRendererDraw, probe:constant-step-with-rate-1, replay | yes |
| C02 | More than 31 attributes or streams in the guest's layout (Metal has 31) | vk_rasterizer.cpp:6122 | emulated (only what the vertex program reads is declared, renumbered from 0) | metal_pipeline_check.cpp:166 | RunPipelineCheck, probe:layout-index-31, replay | yes |
| C03 | More read streams than Metal has free buffer slots next to the program's buffers | vk_rasterizer.cpp:6122 | missing (draw refused) | metal_pipeline_check.cpp:166, metal_shader_translate.cpp:295 | RunPipelineCheck, replay | no |
| C04 | Attribute past its stream's stride | vk_rasterizer.cpp:6122 | emulated (the attribute moves into the stride and reads a copy of the stream bound that many bytes further on) | metal_pipeline_check.cpp:166 | RunPipelineCheck, probe:offset-past-stride | ? |
| C05 | Attribute of a stream without a buffer, constant attributes | vk_buffer_cache.cpp | emulated (a zeroed buffer read at a constant step) | metal_rasterizer.cpp:1058 | probe:attribute-without-layout | yes |
| C06 | Zero stride streams | vk_buffer_cache.cpp | emulated (constant step) | metal_pipeline_check.cpp:166 | probe:constant-step-with-rate-0 | yes |
| C07 | Float 16/32-bit 1-4 components, 11:11:10 float | maxwell_to_vk.cpp | native | metal_graphics.mm:169 | RunRendererDraw, replay | yes |
| C08 | UNORM / SNORM 8 and 16-bit 1-4 components | maxwell_to_vk.cpp | native | metal_graphics.mm:169 | replay | yes |
| C09 | UINT / SINT 8, 16 and 32-bit 1-4 components | maxwell_to_vk.cpp | native | metal_graphics.mm:169 | probe:attribute-format-uint-for-float-input, replay | yes |
| C10 | USCALED / SSCALED 8, 16 and 32-bit | maxwell_to_vk.cpp | emulated (read as integers and converted by the vertex program: `support_scaled_attributes` off) | metal_graphics.mm:169 | untested | ? |
| C11 | 10:10:10:2 UNORM / SNORM | maxwell_to_vk.cpp | native | metal_graphics.mm:169 | replay | yes |
| C12 | 10:10:10:2 UINT / SINT / scaled | maxwell_to_vk.cpp | missing (Metal has it normalized only) | metal_graphics.mm:169 | untested | ? |
| C13 | Attribute type the program reads differs from the format's (`attribute_types`) | fixed_pipeline_state.h | native (keyed and translated per type) | metal_pipeline_cache.cpp:214 | probe:attribute-format-uint-for-float-input | ? |
| C14 | Vertex repacking for strides the host can't take | vk_rasterizer.cpp:6098, vk_compute_pass.cpp | missing (strides go to Metal as they are, nothing is repacked; whether Metal takes one that is not a multiple of 4 is what the probe reports) | metal_pipeline_check.cpp:166 | probe:stride-not-multiple-of-4 | ? |

## D. Topologies and index buffers

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| D01 | Points, lines, line strips, triangles, triangle strips | maxwell_to_vk.cpp | native | metal_graphics.mm:233 | RunRendererDraw | yes |
| D02 | Triangle fans, line loops, polygons | maxwell_to_vk.cpp | emulated (indices rewritten to lists: Metal has no fans or loops) | metal_index_convert.cpp:53 | RunIndexRewrite | no |
| D03 | Quads, quad strips | vk_buffer_cache.cpp, vk_compute_pass.cpp | emulated (indices rewritten to triangles, on the CPU or by a kernel for GPU-written indices) | metal_index_convert.cpp:53, metal_rasterizer.cpp:633 | RunIndexRewrite | no |
| D04 | Lines and triangles with adjacency | maxwell_to_vk.cpp | missing (draw refused: "topology Metal can't draw"; only geometry programs read adjacency) | metal_rasterizer.cpp:489 | untested | no |
| D05 | Patches | maxwell_to_vk.cpp | native (see section J) | metal_scheduler.mm:924 | RunTessellation | yes |
| D06 | 16 and 32-bit indices | maxwell_to_vk.cpp | native | metal_rasterizer.cpp:537, metal_scheduler.mm:951 | RunRendererDraw | yes |
| D07 | 8-bit indices | vk_buffer_cache.cpp:898, vk_compute_pass.cpp | emulated (widened, on the CPU or by a kernel) | metal_index_convert.cpp:10, metal_rasterizer.cpp:633 | RunIndexRewrite, replay | yes |
| D08 | Inline index data | vk_rasterizer.cpp:1260 | native (copied into the stream ring) | metal_rasterizer.cpp:537 | untested | ? |

## E. Texture formats

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| E01 | R8, R8G8, A8B8G8R8, B8G8R8A8: UNORM, SNORM, UINT, SINT, sRGB | maxwell_to_vk.cpp | native | metal_formats.cpp:36 | RunRendererDraw | yes |
| E02 | R16, R16G16, R16G16B16A16: UNORM, SNORM, UINT, SINT, FLOAT; R16G16B16X16_FLOAT | maxwell_to_vk.cpp | native | metal_formats.cpp:36 | replay | yes |
| E03 | R32, R32G32, R32G32B32A32: UINT, SINT, FLOAT | maxwell_to_vk.cpp | native | metal_formats.cpp:36 | replay | yes |
| E04 | R32G32B32_FLOAT | maxwell_to_vk.cpp | missing (Metal has no 96-bit format: no texture is made; texel buffers of it are read as raw words) | metal_formats.cpp:36 | untested | ? |
| E05 | A2B10G10R10 UNORM / UINT, A2R10G10B10_UNORM, B10G11R11_FLOAT | maxwell_to_vk.cpp | native | metal_formats.cpp:36 | replay | yes |
| E06 | E5B9G9R9_FLOAT (sampled) | maxwell_to_vk.cpp | native | metal_formats.cpp:36 | RunOcarinaBilinear | yes |
| E07 | R5G6B5, A1R5G5B5 | maxwell_to_vk.cpp | native | metal_formats.cpp:36 | untested | ? |
| E08 | B5G6R5, A1B5G5R5, A5B5G5R1, A4B4G4R4, sampled | maxwell_to_vk.cpp | emulated (the nearest Metal format read through a view swizzle) | metal_formats.cpp:36 | untested | ? |
| E09 | B5G6R5, A5B5G5R1, A4B4G4R4 as render targets | maxwell_to_vk.cpp | missing (the table marks them not attachable) | metal_formats.cpp:36 | untested | ? |
| E10 | A1B5G5R5 as a render target | maxwell_to_vk.cpp | missing (attachable, but what is drawn is stored unswizzled and read back swizzled; not verified) | metal_formats.cpp:36 | untested | ? |
| E11 | G4R4_UNORM | maxwell_to_vk.cpp | missing (stored as R8: only the high nibble survives) | metal_formats.cpp:36 | untested | ? |
| E12 | BC1, BC2, BC3, BC4, BC5, BC6H, BC7 where the GPU reads them | maxwell_to_vk.cpp | native | metal_formats.cpp:36 | replay | yes |
| E13 | BC formats where the GPU does not read them | vk_texture_cache.cpp | emulated (decoded on the CPU at upload) | metal_resources.mm:187 | untested | ? |
| E14 | ASTC LDR on Apple GPUs | maxwell_to_vk.cpp | native | metal_formats.cpp:36 | replay | yes |
| E15 | ASTC on GPUs without it (Intel / AMD Macs) | vk_compute_pass.cpp | emulated (decoded on the CPU at upload; Vulkan decodes in a compute pass) | metal_resources.mm:187 | untested | no |
| E16 | ETC2, EAC | maxwell_to_vk.cpp | native | metal_formats.cpp:36 | untested | ? |
| E17 | D32_FLOAT, D16_UNORM | maxwell_to_vk.cpp | native | metal_formats.cpp:36 | RunDepthBlit | yes |
| E18 | X8_D24_UNORM | maxwell_to_vk.cpp | emulated (Depth32Float: Apple GPUs have no 24-bit depth, as MoltenVK) | metal_formats.cpp:36 | untested | ? |
| E19 | D24_UNORM_S8_UINT, S8_UINT_D24_UNORM | maxwell_to_vk.cpp | emulated (Depth32Float_Stencil8) | metal_formats.cpp:36 | RunDepthStencilBlit | ? |
| E20 | D32_FLOAT_S8_UINT, S8_UINT | maxwell_to_vk.cpp | native | metal_formats.cpp:36 | RunDepthStencilBlit | ? |
| E21 | Format fallbacks when a device lacks a format or usage (`GetSupportedFormat`) | vk_texture_cache.cpp | missing (one fixed mapping: ASTC and BC have CPU decoders, ETC2 / EAC and the 16-bit packed formats have none on Macs without an Apple GPU) | metal_resources.mm:187 | untested | no |
| E22 | ASTC kept compressed or transcoded to BC (`astc_recompression`) | vk_texture_cache.cpp:2977, vk_texture_cache.cpp:1293 | n/a (Apple GPUs read ASTC as it is) | - | - | no |

## F. Texture kinds, views, samplers

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| F01 | 1D, 2D, 3D, cube, array images with mip levels | vk_texture_cache.cpp | native | metal_resources.mm:31, metal_resources.mm:206 | RunRendererDraw | yes |
| F02 | Linear (pitch) images | vk_texture_cache.cpp | native | metal_texture_cache.cpp:519, metal_texture_cache.cpp:554 | untested | yes |
| F03 | Multisampled images | vk_texture_cache.cpp | native | metal_resources.mm:206 | probe:texture-4-samples | no |
| F04 | Views of every shader texture type (1D, 2D, 3D, cube, arrays, rectangle), level and layer ranges | vk_texture_cache.cpp | native | metal_texture_cache.cpp:673, metal_resources.mm:249 | untested | yes |
| F05 | Component swizzles | maxwell_to_vk.cpp | native | metal_resources.mm:249 | untested | yes |
| F06 | Depth-only and stencil-only views of depth-stencil images | vk_texture_cache.cpp | native | metal_texture_cache.cpp:673, metal_scheduler.mm:116 | RunDepthStencilBlit | ? |
| F07 | Views of another format of the same size class | vk_texture_cache.cpp | native (pixel format views) | metal_resources.mm:249 | untested | yes |
| F08 | Storage image views, written images | vk_texture_cache.cpp | native | metal_texture_cache.cpp:673 | untested | yes |
| F09 | Storage use of sRGB, packed or multisampled images | vk_texture_cache.cpp | missing (such textures are made without shader-write usage; a program writing one is not handled) | metal_texture_cache.cpp:386, metal_resources.mm:199 | untested | ? |
| F10 | Slices of 3D images as render targets | vk_texture_cache.cpp | native (depth plane) | metal_texture_cache.cpp:735, metal_scheduler.mm:129 | untested | ? |
| F11 | Layered render targets (layer chosen by the program) | vk_texture_cache.cpp | missing (one layer is bound; `renderTargetArrayLength` is not set) | metal_scheduler.mm:682 | untested | ? |
| F12 | Texel buffer views | vk_buffer_cache.cpp | native (a texture over the buffer; misaligned ranges through a copy) | metal_rasterizer.cpp:584, metal_resources.mm:301 | RunTexelBuffer | ? |
| F13 | Sampler filters, mip filter, address modes, compare function, LOD clamp, anisotropy | vk_texture_cache.cpp, maxwell_to_vk.cpp | native | metal_resources.mm:342 | probe:sampler-anisotropy-17 | yes |
| F14 | Sampler LOD bias | vk_texture_cache.cpp | missing (no bias is set; `MTLSamplerDescriptor.lodBias` is the candidate where the SDK has it) | metal_resources.mm:342 | untested | ? |
| F15 | Sampler border colours | vk_texture_cache.cpp | missing (approximated: the nearest of Metal's three border colours) | metal_resources.mm:342 | untested | ? |
| F16 | Sampler reduction filter (min / max) | maxwell_to_vk.cpp | missing (Metal has none; filtered normally) | - | untested | ? |
| F17 | Mirror once, clamp to edge | maxwell_to_vk.cpp | native (`MirrorClampToEdge`) | metal_resources.mm:342 | untested | ? |
| F18 | Mirror once with border, mirror once with the OpenGL clamp | maxwell_to_vk.cpp | missing (clamped to border or edge without the mirrored repeat) | metal_resources.mm:342 | untested | ? |
| F19 | Two samplers per guest sampler where anisotropy applies only to some views | vk_texture_cache.cpp | missing (one sampler state per guest sampler) | metal_resources.mm:342 | untested | ? |
| F20 | LOD bias of the Ocarina upscalers | vk_texture_cache.cpp:4950 | missing (see F14) | - | untested | ? |

## G. Uploads, downloads, copies, blits, conversions

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| G01 | Upload and download of colour images | vk_texture_cache.cpp | native | metal_texture_cache.cpp:554, metal_texture_cache.cpp:591 | RunRendererDraw | yes |
| G02 | Upload and download of D32_FLOAT and D16_UNORM | vk_texture_cache.cpp | native | metal_texture_cache.cpp:533 | RunDepthBlit | ? |
| G03 | Upload and download of 24-bit depth, stencil and depth-stencil images | vk_texture_cache.cpp | missing (skipped, logged once) | metal_texture_cache.cpp:533 | untested | ? |
| G04 | Upload and download of multisampled images | vk_texture_cache.cpp | missing (skipped) | metal_texture_cache.cpp:533 | untested | no |
| G05 | `CopyImage` between images of one host format and sample count | vk_texture_cache.cpp | native | metal_texture_cache.cpp:136 | untested | yes |
| G06 | `CopyImageMSAA` (copies between sample counts) | vk_texture_cache.cpp, blit_image.h | missing (skipped, logged once) | metal_texture_cache.cpp:168 | untested | no |
| G07 | `ReinterpretImage`: colour formats of equal texel size | vk_texture_cache.cpp | emulated (through a buffer) | metal_texture_cache.cpp:174 | untested | ? |
| G08 | `ReinterpretImage`: depth <-> colour, compressed, multisampled | vk_texture_cache.cpp | missing (skipped, logged once) | metal_texture_cache.cpp:174 | untested | ? |
| G09 | `ConvertImage`: D32F <-> R32F, D16 <-> R16 | blit_image.h | missing (goes to the blit, which draws depth into depth only, logged once) | metal_texture_cache.cpp:274 | untested | ? |
| G10 | `ConvertImage`: ABGR8 <-> D24S8 / S8D24 / X8D24 / D32F | blit_image.h | missing (goes to the blit, which draws depth into depth only, logged once) | metal_texture_cache.cpp:274 | untested | ? |
| G11 | `BlitImage`: colour, scaled, linear or nearest | vk_texture_cache.cpp, blit_image.h | native (a full-screen triangle) | metal_texture_cache.cpp:236, metal_scheduler.mm:592 | RunOcarinaBilinear | yes |
| G12 | `BlitImage`: depth and stencil | blit_image.h | native (depth into depth under `EDEN_NX2_METAL_COPIES`: nearest, the stencil comes along between images of one format; a stencil-only blit and one into a depth-stencil image of another format are skipped, logged once) | metal_texture_cache.cpp:284, metal_scheduler.mm:592 | RunDepthBlit, RunDepthStencilBlit | yes |
| G13 | `BlitImage`: Fermi 2D operations other than source copy | vk_texture_cache.cpp | missing (copied) | metal_texture_cache.cpp:236 | untested | ? |
| G14 | Multisample resolves (colour and depth-stencil) | blit_image.h, vk_texture_cache.cpp | missing (no resolve attachment or resolve pass) | - | untested | no |
| G15 | Rescaling: scale up, scale down, scaled copies of 2D colour and depth images | vk_texture_cache.cpp | native | metal_texture_cache.cpp:416, metal_texture_cache.cpp:460 | RunDepthBlit, RunDepthStencilBlit | ? |
| G16 | Rescaling of multisampled, 3D and stencil-only images | vk_texture_cache.cpp | missing (those images stay at native size) | metal_texture_cache.cpp:406 | untested | ? |
| G17 | `AccelerateImageUpload` (GPU ASTC decode, BC transcode, 3D unswizzle on the GPU) | vk_texture_cache.cpp, vk_compute_pass.cpp | n/a (never asked: the runtime reports no accelerated uploads; ASTC and BC are read natively, unswizzle is on the CPU) | metal_texture_cache.cpp:350 | - | no |
| G18 | Sparse images | vk_texture_cache.cpp:278 | missing (images are made whole) | - | untested | ? |
| G19 | Byte-addressed shadows of images bound as buffers | vk_texture_cache.cpp | missing (the runtime has no such shadow; what a program reading one gets is not verified) | - | untested | ? |
| G20 | Deferred clears kept across passes | vk_scheduler.cpp:151 | n/a (a Metal clear is the next pass's load action already) | metal_scheduler.mm:555 | - | no |
| G21 | Feedback loops (an image sampled while it is a target) | vk_texture_cache.cpp | missing (see A45) | - | untested | ? |
| G22 | Copies between an uncompressed image and a block-compressed one of the same block size (R16G16B16A16_UINT -> BC4_UNORM, 64 times in the walk) | vk_texture_cache.cpp:1382 | emulated (through a buffer, block for texel, under `EDEN_NX2_METAL_COPIES`: Metal copies texture to texture only within one format; a copy that starts inside a block is skipped, logged once) | metal_texture_cache.cpp:190 | RunBlockCopy | yes |

## H. Queries and conditional rendering

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| H01 | Occlusion counter (samples passed), reported at a fence | vk_query_cache.cpp | native (visibility result buffer, 16384 slots) | metal_rasterizer.cpp:1892, metal_scheduler.mm:724 | untested | yes |
| H02 | Occlusion reports to unmapped addresses dropped, waiting channels woken | vk_query_cache.cpp | native | metal_rasterizer.cpp:1936 | untested | yes |
| H03 | Payload reports (semaphores, timestamps) | vk_query_cache.cpp | native | metal_rasterizer.cpp:1905 | untested | yes |
| H04 | Primitives generated, transform feedback bytes and primitives counters | vk_query_cache.cpp | missing (read as 1) | metal_rasterizer.cpp:1962 | untested | no |
| H05 | Other counters (vertices, invocations, clipping) | vk_query_cache.cpp | missing (read as 1) | metal_rasterizer.cpp:1962 | untested | ? |
| H06 | Counter resets | vk_rasterizer.cpp:2523 | native (occlusion) | metal_rasterizer.cpp:1872 | untested | yes |
| H07 | Conditional rendering on a host query | vk_query_cache.cpp, vk_compute_pass.cpp | missing (the condition is read from guest memory, which waits for the report) | - | untested | yes |
| H08 | Query prefix sums on the GPU | vk_compute_pass.cpp | n/a (sums are made on the CPU when the fence signals) | metal_rasterizer.cpp:1905 | - | no |

## I. Compute, mesh and task programs

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| I01 | Compute programs: uniform and storage buffers, textures, storage images, texel buffers | vk_compute_pipeline.cpp | native | metal_rasterizer.cpp:770, metal_pipeline_cache.cpp:240 | RunTranslate, RunTexelBuffer | yes |
| I02 | Mesh programs without a task program | vk_rasterizer.cpp:1260, vk_graphics_pipeline.cpp | emulated (a compute kernel writes the vertices, a vertex function reads them: Metal mesh functions are not used yet) | metal_rasterizer.cpp:861, metal_pipeline_translate.cpp:449 | RunMeshEmulation, replay | yes |
| I03 | Task + mesh programs | vk_rasterizer.cpp:1260, vk_graphics_pipeline.cpp | emulated (task kernel, distribution kernel, mesh kernel, indirect draw) | metal_rasterizer.cpp:861, metal_pipeline_translate.cpp:449 | RunMeshDistribution, replay | yes |
| I04 | Mesh programs the emulation can't take | vk_pipeline_cache.cpp:3890 | missing (pipeline refused: "mesh program the emulation can't take"; none of the game's 241 stored mesh and task+mesh keys is one) | metal_pipeline_translate.cpp:449 | replay | no |
| I05 | Mesh kernels run ahead of the open render pass | vk_graphics_pipeline.cpp:92 | missing (the kernels' compute encoder ends the open render encoder; the pass is opened again with load and store for every emulated mesh draw) | metal_rasterizer.cpp:1152, metal_scheduler.mm:877 | untested | yes |
| I06 | Consecutive dispatches share one compute encoder | vk_rasterizer.cpp:316 | native | metal_scheduler.mm:877 | untested | yes |

## J. Tessellation

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| J01 | Control + evaluation programs: triangles and quads, integer and fractional spacing, winding | vk_graphics_pipeline.cpp | native (Metal's tessellator; the vertex and control programs run as compute kernels, as MoltenVK does) | metal_rasterizer.cpp:861, metal_pipeline_translate.cpp:605, metal_scheduler.mm:924 | RunTessellation | yes |
| J02 | Isolines | vk_graphics_pipeline.cpp | missing (translated with the isoline mode, but Metal's tessellator has triangle and quad patches only; what is drawn is not verified) | metal_pipeline_translate.cpp:605 | untested | ? |
| J03 | Point mode | vk_graphics_pipeline.cpp | missing (the mode reaches the translation, but Metal's tessellator has no point mode; what is drawn is not verified) | metal_pipeline_translate.cpp:605 | untested | ? |
| J04 | Indexed tessellated draws, 16 and 32-bit, 8-bit widened | vk_rasterizer.cpp:1260 | native | metal_rasterizer.cpp:1230 | RunTessellation | ? |
| J05 | Tessellated draws with a base vertex | vk_rasterizer.cpp:1260 | missing (the base vertex is ignored, logged once) | metal_rasterizer.cpp:462 | untested | ? |
| J06 | Evaluation program without a control program (factors from registers) | vk_graphics_pipeline.cpp | missing (pipeline refused: "only one of the tessellation programs") | metal_pipeline_translate.cpp:605 | untested | ? |
| J07 | Patches of 1-32 control points | fixed_pipeline_state.h | native | metal_rasterizer.cpp:1230 | RunTessellation | yes |
| J08 | Tessellation kernels run ahead of the open render pass | vk_graphics_pipeline.cpp | missing (as I05: the pass is ended and opened again for every tessellated draw; Vulkan tessellates inside the pass) | metal_rasterizer.cpp:1230, metal_scheduler.mm:877 | untested | yes |

## K. Geometry programs and transform feedback

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| K01 | Geometry programs | vk_graphics_pipeline.cpp | missing (pipeline refused: "geometry stage"; Metal has no geometry stage) | metal_pipeline_translate.cpp:339 | untested | no |
| K02 | Layer or viewport index written by the vertex program | vk_pipeline_cache.cpp | native | metal_pipeline_cache.cpp:197 | untested | ? |
| K03 | Geometry program passthrough | vk_pipeline_cache.cpp | missing (profile bit off: it stays a geometry program and is refused) | metal_pipeline_cache.cpp:197 | untested | no |
| K04 | Transform feedback capture | vk_rasterizer.cpp:5266, vk_buffer_cache.cpp | missing (Metal has no transform feedback; the buffers are recorded and nothing is captured) | metal_buffer_cache.cpp:128 | untested | no |
| K05 | Draw of captured vertices (byte count draw) | vk_rasterizer.cpp:1630 | missing (see A05) | metal_rasterizer.cpp:1502 | untested | no |

## L. Shader translation

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| L01 | Vertex, fragment and compute programs: guest code -> IR -> SPIR-V -> MSL (SPIRV-Cross) | vk_pipeline_cache.cpp | native | metal_shader_translate.cpp:110, metal_pipeline_translate.cpp:339 | RunTranslate, replay | yes |
| L02 | Uniform buffers, storage buffers, textures, samplers, storage images numbered into Metal's slots | vk_pipeline_cache.cpp | native | metal_shader_translate.cpp:110 | RunTranslate, replay | yes |
| L03 | Vertex programs with more buffers than the slots left next to the vertex streams | - | missing (program refused: "too many buffers for the vertex buffer slots") | metal_shader_translate.cpp:295 | replay | no |
| L04 | Rescaling words, render area and other push constants | vk_graphics_pipeline.cpp:64 | native (one buffer of bytes after the program's own) | metal_rasterizer.cpp:770 | RunRendererDraw | yes |
| L05 | Clip distances (up to 8) | vk_pipeline_cache.cpp | native | metal_pipeline_cache.cpp:197 | untested | ? |
| L06 | Cull distances | vk_pipeline_cache.cpp | missing (Metal has no cull distances; what SPIRV-Cross makes of a program writing them is not verified) | - | untested | ? |
| L07 | Sample mask output, sample shading, sample id | vk_pipeline_cache.cpp | native (not verified beyond the translation) | metal_shader_translate.cpp:110 | untested | ? |
| L08 | Fragment depth output, early depth tests, discard | vk_pipeline_cache.cpp | native | metal_shader_translate.cpp:110 | replay | yes |
| L09 | 64-bit floats | vk_pipeline_cache.cpp | n/a (lowered to 32-bit by the IR, as on Vulkan devices without them) | metal_pipeline_cache.cpp:197 | - | ? |
| L10 | 64-bit integer atomics, typeless image loads, viewport masks | vk_pipeline_cache.cpp | n/a (profile bits off: the IR takes the path Vulkan takes on devices without them) | metal_pipeline_cache.cpp:197 | - | ? |
| L11 | Texture handles computed by the program (bindless) | vk_compute_pipeline.cpp:444 | native | metal_rasterizer.cpp:925 | replay | yes |
| L12 | Programs translated with the draw's fixed state | vk_pipeline_cache.cpp | native | metal_pipeline_translate.cpp:122 | replay | yes |
| L13 | Compiled programs kept between runs | vk_pipeline_cache.cpp | native (binary archive; the SPIR-V -> MSL step is redone each run) | metal_archive.mm:30 | RunPipelineArchive | yes |
| L14 | Instance index, base instance and base vertex read by a vertex program on iOS (every program reading the instance index: 22 % of the game's stored pipelines) | - | native (under `EDEN_NX2_METAL_MSL_DEVICE`: SPIRV-Cross is told the GPU family, as MoltenVK tells it; without, the program is refused) | metal_shader_translate.cpp:110 | RunDeviceMsl, replay | yes |
| L15 | Subgroup operations on SIMD groups (what MoltenVK emits) rather than quad groups of 4 | - | native (under `EDEN_NX2_METAL_MSL_DEVICE`) | metal_shader_translate.cpp:110 | RunDeviceMsl, replay | yes |
| L16 | MoltenVK's translation options for Apple GPUs: cube gradient fix-up, stores of discarded fragments checked, recursive input structures | - | native (under `EDEN_NX2_METAL_MSL_DEVICE`; no program of the game was seen to depend on one) | metal_shader_translate.cpp:110 | replay | ? |
| L17 | MoltenVK's other translation options: fixed subgroup size, shadow samples with a LOD as gradients (before iOS 17), point size only in point draws | - | missing (not adopted: the first two change nothing on the A19; the point size is written by every vertex program that has one, which Metal accepts only for point topologies with the validation layer on) | - | untested | ? |

## M. Buffer cache and memory

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| M01 | Uniform buffers | vk_buffer_cache.cpp | native (the stream ring) | metal_buffer_cache.cpp:146, metal_staging_buffer_pool.cpp:73 | RunRendererDraw | yes |
| M02 | Storage buffers, written storage buffers tracked as GPU-modified | vk_buffer_cache.cpp | native | metal_rasterizer.cpp:770, metal_rasterizer.cpp:925 | RunTranslate | yes |
| M03 | Vertex, index and indirect buffers | vk_buffer_cache.cpp | native | metal_buffer_cache.cpp:82, metal_buffer_cache.cpp:105 | RunRendererDraw | yes |
| M04 | `CopyBuffer`, `ClearBuffer` | vk_buffer_cache.cpp | native | metal_buffer_cache.cpp:51, metal_buffer_cache.cpp:59 | untested | ? |
| M05 | Buffer downloads at fences, CPU reads of GPU-written pages | vk_buffer_cache.cpp | native (the shared buffer cache) | metal_rasterizer.cpp:2000 | untested | yes |
| M06 | Mesh output mirror (`Nx2MeshMirror`) | vk_buffer_cache.cpp | native | metal_buffer_cache.cpp:74 | RunMeshEmulation | ? |
| M07 | Storage buffers spanning several mappings | vk_buffer_cache.cpp, vk_multi_range_buffer.cpp | missing (the runtime has no such entry; what the shared cache does instead is not verified) | - | untested | ? |
| M08 | Staging and stream buffers bounded under memory pressure | vk_staging_buffer_pool.cpp:95, vk_staging_buffer_pool.cpp:157 | missing (a fixed stream ring; power-of-two staging buffers kept 300 frames; no cap) | metal_staging_buffer_pool.cpp:26, metal_staging_buffer_pool.cpp:103 | untested | yes |
| M09 | Out-of-memory and device-loss handling | vk_scheduler.cpp, vk_swapchain.cpp | missing (no recovery: command buffer errors are logged with the failing encoders, a refused render pass drops its draws, a staging buffer that can't be made asserts) | metal_scheduler.mm:400, metal_staging_buffer_pool.cpp:26 | untested | ? |
| M10 | Memory report lines ("Graphics memory") | vk_rasterizer.cpp:4547 | missing (allocated bytes are part of the 10 s `Metal:` line only) | renderer_metal.mm:520 | untested | no |

## N. Presentation, scalers, frame generation

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| N01 | Composite of the display layers into the window, blending per layer | renderer_vulkan.cpp, present/layer.cpp | native | renderer_metal.mm:411 | untested | yes |
| N02 | Layers from guest memory (no cached image) | present/layer.cpp | native | renderer_metal.mm:411 | untested | ? |
| N03 | Window scaling: bilinear | present/filters.cpp | native | renderer_metal.mm:411 | untested | yes |
| N04 | Window scaling: nearest, bicubic, Gaussian, Lanczos, ScaleForce, area and the other filters | present/filters.cpp | missing (bilinear) | renderer_metal.mm:411 | untested | no |
| N05 | FSR 1, SGSR | present/fsr.cpp, present/sgsr.cpp | missing (bilinear) | - | untested | no |
| N06 | MetalFX Spatial | present/metalfx_spatial.mm | native | renderer_metal.mm:465 | untested | ? |
| N07 | FXAA, SMAA | present/fxaa.cpp, present/smaa.cpp | missing (not applied) | - | untested | no |
| N08 | Post-processing chain | present/post_process.cpp | missing (not applied) | - | untested | ? |
| N09 | Frame generation: MetalFX | present/metalfx_frame_gen.mm, present/frame_gen.cpp | missing (no generated frames; every presented frame is a rendered one) | - | untested | ? |
| N10 | Frame generation: FSR 3, LSFG, NVIDIA optical flow | present/fsr3_frame_gen.cpp, present/lsfg_generate.cpp, present/nv_optical_flow.cpp | missing (FSR 3 and LSFG are compute and fragment programs that could run on Metal; NVIDIA optical flow does not exist on Apple devices) | - | untested | ? |
| N11 | HDR output and colour space | vk_swapchain.cpp | missing (BGRA8 drawable) | metal_resources.mm:202 | untested | no |
| N12 | Vsync modes | vk_swapchain.cpp | native (macOS: `displaySyncEnabled`; iOS presents on the display's refresh only, as it does under Vulkan) | metal_swapchain.mm:33 | untested | yes |
| N13 | Frame pacing (host pacer, present thread, turbo mode) | vk_present_manager.cpp, vk_host_pacer.cpp | missing (frames are presented when they are ready) | metal_swapchain.mm:43 | untested | ? |
| N14 | Screenshots | renderer_vulkan.cpp | missing (the request is answered with failure) | renderer_metal.mm:411 | untested | no |
| N15 | Applet capture buffer | renderer_vulkan.cpp | missing (an empty buffer) | renderer_metal.mm:582 | untested | ? |
| N16 | Performance report lines every 10 s | vk_rasterizer.cpp:4162 | native (own lines: `Metal: N frames presented ...`, `Perf Metal coverage:`) | renderer_metal.mm:520 | untested | yes |
| N17 | Frame dump, texture dump, flash watch | vk_frame_dump.cpp, vk_flash_watch.cpp | missing (debugging tools of the Vulkan backend) | - | untested | no |
| N18 | Debug server hooks (render target list, pass list, GPU profile) | vk_rasterizer.cpp:3777 | missing (the debug server's graphics commands answer nothing under Metal) | - | untested | no |

## O. Ocarina of Time paths

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| O01 | Upscale dispatch replaced by one bilinear pass (phones) | vk_rasterizer.cpp:3716, vk_ocarina_camera_motion.cpp:1116 | native | metal_ocarina.cpp:79, metal_rasterizer.cpp:1685 | RunOcarinaBilinear | yes |
| O02 | Upscale dispatch replaced by a temporal upscaler (`TryOcarinaUpscale`: camera tracking, jitter, motion, MetalFX Temporal / FSR 3 / DLSS) | vk_rasterizer.cpp:3488, vk_ocarina_camera_motion.cpp | missing (the camera tracking lives in the Vulkan backend) | - | untested | yes |
| O03 | `AccelerateTemporalUpscale` through MetalFX Temporal (the game's own network inputs) | vk_rasterizer.cpp:4710, present/metalfx_temporal.mm | native | metal_texture_cache.cpp:265, metal_temporal.mm:58 | untested | ? |
| O04 | Temporal upscale through FSR 3 or the built-in temporal filter | vk_texture_cache.cpp:1936, blit_image.h | missing (declined: the game's dispatch runs) | metal_texture_cache.cpp:265 | untested | ? |
| O05 | Resolution scale, frame rate and graphics presets (`oot_resolution`, `oot_frame_rate`, `oot_graphics_preset`) | - | n/a (applied to the game, not the renderer) | - | - | yes |
| O06 | Handheld temporal switch (lower scales to 1920x1080 through the temporal upscaler) | vk_rasterizer.cpp:3605 | missing (with O02) | - | untested | yes |
| O07 | Visual workaround | vk_pipeline_cache.cpp:582 | missing (opt-in on Vulkan; not present) | - | untested | ? |

## P. Performance work

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| P01 | Render pass kept open across draws to the same targets | vk_scheduler.cpp | native | metal_scheduler.mm:682 | untested | yes |
| P02 | Uploads, copies and dispatches recorded ahead of the open pass (upload command buffer, before-pass command buffer) | vk_scheduler.cpp:142 | missing (a blit or compute encoder ends the open render encoder; the pass is opened again with load and store) | metal_scheduler.mm:208, metal_scheduler.mm:877 | untested | yes |
| P03 | Fences without a submission of their own (no flush when nothing is pending, deferred and batched flushes) | vk_fence_manager.cpp:48, vk_scheduler.cpp:469 | missing (a command buffer is committed for every fence that is not stubbed) | metal_fence_manager.cpp:12 | untested | yes |
| P04 | CPU write reclaim | vk_rasterizer.cpp:2873 | native (A33, under `EDEN_NX2_METAL_CACHE_PATHS`) | metal_rasterizer.cpp:2213 | untested | yes |
| P05 | GPU read landing | vk_rasterizer.cpp:2710 | native (A28, under `EDEN_NX2_METAL_CACHE_PATHS`) | metal_rasterizer.cpp:2081 | untested | yes |
| P06 | Guest writes collected without the caches' locks | vk_rasterizer.cpp:2818 | native (under `EDEN_NX2_METAL_CACHE_PATHS`) | metal_rasterizer.cpp:2154 | untested | yes |
| P07 | Pass contents discarded on tile GPUs where nothing reads them (store and load actions) | vk_texture_cache.cpp:85 | missing (every attachment is loaded and stored) | metal_scheduler.mm:682 | untested | ? |
| P08 | GPU time of the frame | metal_gpu_time.mm:117 | native (GPU time of every command buffer) | metal_scheduler.mm:973, renderer_metal.mm:520 | untested | yes |
| P09 | GPU time per pass | vk_scheduler.cpp:293 | missing (only the total; no `Perf passes` lines) | - | untested | no |
| P10 | Pipeline eviction under memory pressure | vk_pipeline_cache.cpp:1129 | missing (no pipeline is ever released) | - | untested | ? |
| P11 | Unchanged draw state and bindings not set again | vk_rasterizer.cpp:342, pipeline_helper.h:467 | missing (every draw sets the pipeline, depth-stencil state, viewport, scissor, cull, bias, stencil references, blend colour and every buffer, texture and sampler again) | metal_scheduler.mm:746, metal_scheduler.mm:785 | untested | yes |
| P12 | Texture cache budget from the memory left | vk_texture_cache.cpp | native (`DeviceMemoryBudget`) | metal_resources.mm:183 | untested | yes |
| P13 | Uniform padding, push-constant reuse | vk_buffer_cache.cpp:442, vk_graphics_pipeline.cpp:64 | n/a (costs of MoltenVK's translation that Metal used directly does not have) | - | - | no |

## Q. Switches that change what Vulkan renders

One row per `EDEN_NX2_*` switch (or group read together) that changes rendering in
`renderer_vulkan/`. The Status says whether Metal has the behaviour the switch controls, not the
switch. Switches read in shared code apply to both backends where the backend has the hook; they
are the rows whose Vulkan cell starts with `../`.

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| Q01 | `EDEN_NX2_A2C_REFRESH`: alpha to coverage recorded on every draw | vk_rasterizer.cpp:5869 | n/a (B12: part of the render state built for each draw) | metal_rasterizer.cpp:1058 | - | no |
| Q02 | `EDEN_NX2_ASTC_BC1`, `EDEN_NX2_KEEP_ASTC` | vk_texture_cache.cpp:2977, vk_texture_cache.cpp:1293 | n/a (E22) | - | - | no |
| Q03 | `EDEN_NX2_ASYNC_DISCARD`: draws skipped or held while their pipeline builds | vk_pipeline_cache.cpp:3436 | missing (B41: every draw waits for its pipeline) | - | untested | yes |
| Q04 | `EDEN_NX2_ASYNC_TRANSLATE`, `EDEN_NX2_ASYNC_TRANSLATE_THREADS` | ../async_translate.h:20, vk_pipeline_cache.cpp:401 | missing (B41) | - | untested | yes |
| Q05 | `EDEN_NX2_BACKGROUND_PIPELINES`, `_THREADS`, `_LIMIT`, `EDEN_NX2_PRELOAD_ALL_PIPELINES`, `EDEN_NX2_BUILD_SHADERS_LIMIT`, `EDEN_NX2_REKEY_FOREIGN_PIPELINES` | vk_pipeline_cache.cpp:1112, vk_pipeline_cache.cpp:1268 | missing (A56) | - | untested | yes |
| Q06 | `EDEN_NX2_BEFORE_PASS` | vk_scheduler.cpp:142 | missing (P02) | - | untested | yes |
| Q07 | `EDEN_NX2_BINDMEMO`, `_BATCH`, `_PARTIAL` | pipeline_helper.h:467 | missing (P11) | - | untested | yes |
| Q08 | `EDEN_NX2_DISPATCH_CHAIN` | vk_rasterizer.cpp:316 | native (I06) | metal_scheduler.mm:877 | untested | yes |
| Q09 | `EDEN_NX2_DYNSTATE_CACHE`, `EDEN_NX2_VERTEX_INPUT_CACHE` | vk_rasterizer.cpp:342, vk_rasterizer.cpp:350 | missing (P11) | - | untested | yes |
| Q10 | `EDEN_NX2_FAIL_NATIVE_MESH`, `EDEN_NX2_MESH_EMULATE`, `EDEN_NX2_NATIVE_PER_PRIMITIVE`, `EDEN_NX2_FLAT_PER_PRIMITIVE`, `EDEN_NX2_MESH_DIRECT_OUTPUTS`: native mesh pipelines and the choice between them and the emulation | vk_graphics_pipeline.cpp:653, vk_pipeline_cache.cpp:1322, vk_pipeline_cache.cpp:1309, vk_pipeline_cache.cpp:1295, vk_pipeline_cache.cpp:1279 | missing (I02: Metal has the emulation only) | - | untested | yes |
| Q11 | `EDEN_NX2_NO_MESH`, `EDEN_NX2_NO_MESH_FILE`, `EDEN_NX2_NO_MESH_EMULATION`: mesh draws skipped for comparisons | vk_rasterizer.cpp:842, vk_pipeline_cache.cpp:3890 | missing (`EDEN_NX2_METAL_DRAW=0` skips every draw, not mesh draws alone) | metal_rasterizer.cpp:369 | untested | no |
| Q12 | `EDEN_NX2_MESH_BEFORE_PASS` | vk_graphics_pipeline.cpp:92 | missing (I05) | - | untested | yes |
| Q13 | `EDEN_NX2_FENCE_FLUSH_US`, `EDEN_NX2_FENCE_PATIENCE_US`, `EDEN_NX2_LAZY_WAKE` | vk_scheduler.cpp:469, vk_fence_manager.cpp:48, vk_scheduler.cpp:1103 | missing (P03) | - | untested | yes |
| Q14 | `EDEN_NX2_FENCE_BATCH`, `EDEN_NX2_FENCE_FOLD` | ../fence_manager.h:356, ../fence_manager.h:449 | native (shared fence manager) | metal_fence_manager.cpp:12 | untested | yes |
| Q15 | `EDEN_NX2_RELEASE_SUBMIT` | ../fence_manager.h:260 | n/a (Metal defers no submission; with P03 closed it applies) | - | - | no |
| Q16 | `EDEN_NX2_FLUSH_DRAWS`: a submission every so many draws | vk_rasterizer.cpp:4981 | missing (command buffers are committed at fences, waits and frame ends only) | metal_rasterizer.cpp:2362 | untested | ? |
| Q17 | `EDEN_NX2_FORCE_LOD0` | vk_texture_cache.cpp:4950 | missing (debug aid) | - | untested | no |
| Q18 | `EDEN_NX2_GPL`, `EDEN_NX2_GPL_OPTIMIZE`, `EDEN_NX2_MVK_PIPELINE_CACHE` | vk_pipeline_library.cpp:35, vk_pipeline_cache.cpp:559 | n/a (Vulkan pipeline library and cache objects; Metal's counterpart is B42) | - | - | no |
| Q19 | `EDEN_NX2_HANDLE_ALIGN` | vk_compute_pipeline.cpp:444 | native (L11: the shared `EvaluateTextureHandle`; aligned read not verified) | metal_rasterizer.cpp:925 | untested | yes |
| Q20 | `EDEN_NX2_HOST_PACE` | vk_host_pacer.cpp | missing (N13) | - | untested | ? |
| Q21 | `EDEN_NX2_INLINE_FAST`, `EDEN_NX2_INLINE_DEFER`, `EDEN_NX2_I2M_COALESCE`, `EDEN_NX2_I2M_LIST_PAGE` | vk_rasterizer.cpp:4795, vk_rasterizer.cpp:4836 | native (A54) | metal_rasterizer.cpp:2461 | untested | ? |
| Q22 | `EDEN_NX2_INDIRECT_GRID`, `EDEN_NX2_INDIRECT_DRAWS`, `EDEN_NX2_GPU_ARG_DRAW`: draws and dispatches whose arguments the GPU wrote, issued without reading them back | ../nx2_indirect_grid.h:56, vk_rasterizer.cpp:1323 | missing (dispatch grids are native, A19 and A60; draw counts are read back, A61; mesh task counts are A10) | - | untested | ? |
| Q23 | `EDEN_NX2_INVALIDATE_BEFORE_DISPATCH` | vk_rasterizer.cpp:111 | missing (A66: experiment, off by default) | - | untested | no |
| Q24 | `EDEN_NX2_KEEP_DEFERRED_CLEAR` | vk_scheduler.cpp:151 | n/a (G20) | - | - | no |
| Q25 | `EDEN_NX2_KEY_ALL_ATTRIBUTES` | ../shader_cache.cpp:43 | native (shared; C13) | metal_pipeline_cache.cpp:214 | untested | ? |
| Q26 | `EDEN_NX2_METALFX_SYNC` | present/metalfx_context.mm:25 | n/a (MetalFX under Vulkan needs a hand-over between two APIs; Metal encodes it into its own command buffer) | metal_temporal.mm:58 | - | no |
| Q27 | `EDEN_NX2_MSAA_DISCARD` | vk_texture_cache.cpp:85 | missing (P07) | - | untested | no |
| Q28 | `EDEN_NX2_OOT_BILINEAR_UPSCALE` | vk_ocarina_camera_motion.cpp:1116 | native (O01) | metal_ocarina.cpp:33 | RunOcarinaBilinear | yes |
| Q29 | `EDEN_NX2_OOT_TEMPORAL_HANDHELD`, `EDEN_NX2_DLSS_REPLACEMENT`, `EDEN_NX2_OOT_JITTER`, `EDEN_NX2_DLSS_JITTER`, `EDEN_NX2_TEMPORAL_SHARPEN` and the `EDEN_NX2_TEMPORAL_*` filter weights | vk_rasterizer.cpp:3605, vk_ocarina_camera_motion.cpp:1096, vk_texture_cache.cpp:1936 | missing (O02, O04, O06) | - | untested | yes |
| Q30 | `EDEN_NX2_OOT_VISUAL_WORKAROUND` | vk_pipeline_cache.cpp:582 | missing (O07) | - | untested | ? |
| Q31 | `EDEN_NX2_PIPELINE_EVICT_ABOVE`, `_LEFT_MIB`, `_MAX` | vk_pipeline_cache.cpp:1129 | missing (P10) | - | untested | ? |
| Q32 | `EDEN_NX2_PUSH_CONSTANT_REUSE`, `EDEN_NX2_UBO_PAD` | vk_graphics_pipeline.cpp:64, vk_buffer_cache.cpp:442 | n/a (P13) | - | - | no |
| Q33 | `EDEN_NX2_QUERY_LAZY_SYNC` | vk_query_cache.h:45 | n/a (A41: no query cache) | - | - | no |
| Q34 | `EDEN_NX2_READ_FASTPATH`, `EDEN_NX2_TEXTURE_READ_FILTER` | vk_rasterizer.cpp:2670, vk_rasterizer.cpp:2616 | native (A26) | metal_rasterizer.cpp:2019 | untested | yes |
| Q35 | `EDEN_NX2_WRITE_FASTPATH` | vk_rasterizer.cpp:2838 | native (P06) | metal_rasterizer.cpp:2154 | untested | yes |
| Q36 | `EDEN_NX2_CPU_WRITE_RECLAIM` | ../buffer_cache/cpu_write_stats.h:40 | native (A33) | metal_rasterizer.cpp:2213 | untested | yes |
| Q37 | `EDEN_NX2_GPU_READ_LAND` | ../buffer_cache/cpu_read_stats.h:48 | native (A28) | metal_rasterizer.cpp:2081 | untested | yes |
| Q38 | `EDEN_NX2_REBAR_UPLOAD`, `EDEN_NX2_STREAM_BUFFER_MIB`, `EDEN_NX2_STAGING_CAP_MIB`, `EDEN_NX2_STAGING_SUBMIT_MIB` | vk_staging_buffer_pool.cpp:86, vk_staging_buffer_pool.cpp:95, vk_staging_buffer_pool.cpp:157 | missing (M08; resizable BAR is n/a on shared memory) | - | untested | yes |
| Q39 | `EDEN_NX2_RESETFIX` (Donkey Kong Bananza: the reset program's indirect records written on the host) | vk_rasterizer.cpp:657 | native (A65) | metal_rasterizer.cpp:292 | untested | no |
| Q40 | `EDEN_NX2_SPARSE_ALIASED` | vk_texture_cache.cpp:278 | missing (G18) | - | untested | ? |
| Q41 | `EDEN_NX2_TARGETLESS_REACH` | vk_graphics_pipeline.cpp:75 | native (B03; the same switch) | metal_rasterizer.cpp:72 | probe:no-attachments | yes |
| Q42 | `EDEN_NX2_UINT8_INDICES`, `EDEN_NX2_UINT8_REUSE` | vk_buffer_cache.h:206, vk_buffer_cache.cpp:898 | emulated (D07: widened for each draw; the widened copy is not kept) | metal_index_convert.cpp:10 | RunIndexRewrite | ? |
| Q43 | `EDEN_NX2_WFI_DEFER` | vk_rasterizer.cpp:3053 | n/a (A44: nothing to defer) | - | - | no |
| Q44 | `EDEN_NX2_SPLIT`, `EDEN_NX2_METAL_SPLIT`: guest commands and backend work on two threads | ../gpu_split.cpp:114, ../gpu_split.cpp:129 | native (shared; on for Metal unless `EDEN_NX2_METAL_SPLIT=0`) | metal_scheduler.mm:158 | untested | ? |
| Q45 | `EDEN_NX2_OOT_MACROS`, `EDEN_NX2_OOT_MOVIE_NETWORK`, `EDEN_NX2_EXPERIMENTAL_GEOMETRY` and the buffer and texture cache tuning switches (`EDEN_NX2_BC_*`, `EDEN_NX2_BUFCACHE_*`, `EDEN_NX2_TC_*`, `EDEN_NX2_SSBO_MEMO`, `EDEN_NX2_COMPUTE_BINDCACHE`, `EDEN_NX2_RT_FAST`, `EDEN_NX2_RT_LOOKUP_MEMO`, `EDEN_NX2_CACHE_BUDGET`) | ../engines/maxwell_3d.cpp:1377, ../engines/kepler_compute.cpp:56, ../shader_environment.cpp:894 | native (read in shared code both backends run; not verified one by one on Metal) | - | untested | ? |

Not listed, because they draw nothing differently: the traces, dumps, probes and reports
(`EDEN_NX2_*_TRACE`, `*_DUMP*`, `*_PROBE`, `FLASH_WATCH*`, `FRAME_DUMP*`, `FG_DUMP*`,
`PASS_LIST`, `PASS_TIMES`, `GPU_PROFILE`, `PERF_REPORT`, `MEMORY_REPORT`, `SEM_WATCH`,
`DISPATCH_DEPS`, `PIPELINE_DIFF`, `PIPELINE_CLEAN`, `PRECACHE_*`), the debugging overrides
(`PIPE_PATCH`, `PIPE_SKIP`, `SKIP_VS`, `TRACE_PS`, `TRACE_VS`), the numbered experiments
(`EXP*`), the ASTC decoder pool tuning (`ASTC_ARRAY_MIB`, `ASTC_FLUSH`, `ASTC_ONE_SUBMIT`,
`ASTC_POOL_*`, `ASTC_SCRATCH_BUCKET`: no GPU decoder on Metal, E15), the NVIDIA DLSS options
(`DLSS_PRESET`, `DLSS_HDR`, `DLSS_AUTO_EXPOSURE`: A58) and `APPLE_WORKGROUP` (threads, not
rendering). The feature census (`EDEN_NX2_FEATURE_CENSUS`) and the shader dump
(`EDEN_NX2_SHADER_DUMP`) are shared code and work under both backends.

## R. Metal only

What Metal needs and Vulkan does not; not part of the parity count.

| ID | Capability | Vulkan | Status | Metal | Tested | OoT |
|---|---|---|---|---|---|---|
| R01 | Descriptors Metal's validation aborts on are refused and logged (`EDEN_NX2_METAL_PIPELINE_CHECK`) | - | native | metal_pipeline_check.cpp:270, metal_graphics.mm:316 | RunPipelineCheck, probe:control | yes |
| R02 | Texture descriptors Metal aborts on are refused | - | native | metal_pipeline_check.cpp:301, metal_resources.mm:206 | RunPipelineCheck, probe:texture-4-samples | yes |
| R03 | Refused draws, dispatches and clears counted by cause, logged once per cause; `Perf Metal coverage:` every 10 s | - | native | metal_coverage.cpp, renderer_metal.mm:543 | untested | yes |
| R04 | Abort trace: the last render state and the abort's stack in the log (`EDEN_NX2_METAL_ABORT_TRACE`) | - | native | renderer_metal.mm:270 | untested | yes |
| R05 | Command buffer errors logged with the failing encoders (`EDEN_NX2_METAL_ENCODER_STATUS`) | - | native | metal_scheduler.mm:400 | untested | yes |
| R06 | Draws, dispatches and GPU index rewriting switched off for comparisons (`EDEN_NX2_METAL_DRAW`, `EDEN_NX2_METAL_DISPATCH`, `EDEN_NX2_METAL_GPU_INDICES`) | - | native | metal_rasterizer.cpp:369, metal_rasterizer.cpp:1685 | untested | no |
| R07 | Programs translated with fixed state on or off (`EDEN_NX2_METAL_FIXED_STATE`) | - | native | metal_pipeline_translate.cpp:56 | untested | yes |
| R08 | Pipeline archive on or off (`EDEN_NX2_METAL_ARCHIVE`) | - | native | metal_archive.mm:30 | RunPipelineArchive | yes |
| R09 | Guest reads, guest writes, inline writes and fences on the Vulkan rasterizer's paths or the plain ones (`EDEN_NX2_METAL_CACHE_PATHS`) | - | native | metal_rasterizer.cpp:2154 | untested | yes |
| R10 | MSL for the device's GPU family or plain (`EDEN_NX2_METAL_MSL_DEVICE`) | - | native | metal_shader_translate.cpp:102 | RunDeviceMsl, replay | yes |
| R11 | Vertex buffer offset past the buffer's end, attachment level or slice past the texture, pipeline and pass disagreeing on attachment formats | - | missing (not checked: Metal aborts on them only with its validation layer on, which release builds do not have; what is drawn then is undefined) | - | untested | ? |
| R12 | Guest targets of 8 samples | - | missing (texture refused: Apple GPUs have 4 at most; not drawn at 4 instead) | metal_pipeline_check.cpp:301 | probe:texture-4-samples | no |
| R13 | Stencil views the scheduler's blits make of depth-stencil textures | - | missing (made outside the checked `CreateTextureView`; the formats are the two Metal defines for it) | metal_scheduler.mm:116 | RunDepthStencilBlit | no |
| R14 | A multisampled image bound where the program declares a plain texture | - | missing (not checked; Metal's behaviour is not verified) | - | untested | no |
| R15 | A refused pipeline, texture or view | - | missing (the draw is dropped or the texture left unbound: a hole in the picture, counted and logged once per cause, never a substitute) | metal_coverage.cpp | untested | ? |
| R16 | Depth blits and block-compressed copies done or skipped (`EDEN_NX2_METAL_COPIES`) | - | native | metal_texture_cache.cpp:55 | RunBlockCopy, RunDepthBlit | yes |
