# iOS platform layer

Engine-side iOS support: MoltenVK options, memory budgets and pressure, cache persistence,
lifecycle, thermal state and per-device defaults. The app talks to it through
`src/ios_frontend/lichii_bridge_platform.h` (C API, target `lichii_platform`). Everything here is
compile-verified only.

## Use from the app

1. `lichii_platform_init()` on the main thread at start, before Vulkan is opened and a game boots
   (`yuzu_cmd/yuzu.cpp` does it in `SDL_AppInit`).
2. First launch: `lichii_platform_apply_device_defaults(LICHII_DEFAULTS_FIRST_LAUNCH)`, then save
   the configuration when it returns 1. "Reset to device defaults": `LICHII_DEFAULTS_FORCE`. The
   SDL front end uses `LICHII_DEFAULTS_UNCHANGED` (settings still at their built-in default).
3. `lichii_platform_set_event_handler`: pause emulation on `WILL_RESIGN_ACTIVE`, resume on
   `DID_BECOME_ACTIVE` unless the user paused; show `lichii_platform_thermal_advice` on
   `THERMAL_STATE` / `LOW_POWER_MODE`. The SDL front end pauses on SDL's
   `WILL_ENTER_BACKGROUND` (sent on resign-active) and resumes on `DID_ENTER_FOREGROUND`.
4. `lichii_platform_set_playing(1/0)` around a session (idle timer, audio session).
5. Loading screen: forward the disk cache loader's progress with
   `lichii_platform_note_shader_progress` (done in yuzu.cpp) and read it with
   `lichii_platform_shader_progress` or a handler.
6. Game context menu: `lichii_platform_shader_cache_bytes` / `lichii_platform_clear_shader_cache`.

## MoltenVK

Set as environment variables before the library loads (`vulkan_library.cpp`, iOS part, and
`lichii_platform_init` for GPU dependent ones), never overwriting a value already set. Source:
MoltenVK's `Docs/MoltenVK_Configuration_Parameters.md` (main, fetched 2026-11-10); the bundled
iOS MoltenVK is 1.4.2.

| Option | Value | Why |
|---|---|---|
| `MVK_CONFIG_SYNCHRONOUS_QUEUE_SUBMITS` | 0 | Submits go to a GCD queue instead of the GPU thread (default 1). |
| `MVK_CONFIG_RESUME_LOST_DEVICE` | 1 | A command buffer failing (e.g. committed just as the app left the foreground: Metal refuses background GPU work) is logged instead of losing the device for good. MeloNX does the same. |
| `MVK_CONFIG_SHADER_COMPRESSION_ALGORITHM` | 1 (LZFSE) | MoltenVK keeps every shader's MSL in memory for pipeline cache export; thousands of shaders cost real memory under jetsam limits. |
| `MVK_CONFIG_MAX_ACTIVE_METAL_COMMAND_BUFFERS_PER_QUEUE` | 32 | As MeloNX; with prefill off one Metal command buffer per submit. |
| `MVK_CONFIG_LOG_LEVEL` | 2 | Warnings and errors only. |
| `MVK_CONFIG_PERFORMANCE_TRACKING` | 0 | Off (the default, pinned). |
| `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` | 0 only below tier 2 | Default 1 is right on every A13+ GPU; tier 1 devices get it off (MeloNX). |
| `MVK_CONFIG_SHOULD_MAXIMIZE_CONCURRENT_COMPILATION` | 1 (kept) | No effect on iOS per the docs; macOS only. |

Left at their defaults on purpose:

- `MVK_CONFIG_FAST_MATH_ENABLED` = 2: fast math, but a shader declaring float controls
  (`SignedZeroInfNanPreserve`, `ContractionOff`) keeps them. MeloNX forces 1; Switch shaders rely
  on NaN/Inf behaviour often enough that the per-shader opt-out matters.
- `MVK_CONFIG_PREFILL_METAL_COMMAND_BUFFERS` = 0: documented as the fastest.
- `MVK_CONFIG_USE_MTLHEAP` = 1: heaps on Apple GPUs; needed for `VK_EXT_image_2d_view_of_3d` and
  block-texel views.
- `MVK_CONFIG_VK_SEMAPHORE_SUPPORT_STYLE` = 1 (MTLEvent): MeloNX uses single-queue (0); we keep
  real semaphores.

## Shader and pipeline caches

- Our caches (`<Documents>/eden/cache/shader/<title>/vulkan-frizti*.bin`, `vulkan_used-frizti*`,
  `vulkan_pipelines*.bin`) already persist in the app container. The cache directory is excluded
  from iCloud backups (`NSURLIsExcludedFromBackupKey`).
- MoltenVK's `VkPipelineCache` holds MSL source, not Metal binaries (MoltenVK issue #1765:
  `MTLBinaryArchive` is not used). The Metal compiler's own per-app shader cache keeps compiled
  functions across launches, so a second launch skips SPIR-V to MSL (our file) and usually the
  Metal compile (the OS cache). Using `MTLBinaryArchive` would need MoltenVK changes.
- iOS may kill a suspended app without notice, so the Vulkan pipeline cache is written when the
  app enters the background (a `Common::PlatformEvents` background listener in
  `vk_pipeline_cache.cpp`, run in a UIKit background task) and the used-pipeline list at the next
  draw. Load and save log their size in KiB.

## Memory

- Entitlements in `dist/ios/Lichii.entitlements`: `increased-memory-limit`,
  `extended-virtual-addressing`, `get-task-allow`. At start the platform layer reads which ones
  the signature really carries (`SecTaskCopyValueForEntitlement`) and probes 4 / 16 GiB
  PROT_NONE reservations; all exposed in `LichiiDeviceInfo` and logged.
- Budgets from `os_proc_available_memory()` at start (texture cache critical
  clamp(avail/4, 1, 2 GiB), expected 512 MiB less; buffer cache collects at
  clamp(avail/16, 256, 768 MiB) (as MeloNX), aggressively at twice that; low-water mark
  clamp(avail/10, 384, 768 MiB)). `EDEN_NX2_*` environment overrides still win.
- Memory warnings (`UIApplicationDidReceiveMemoryWarningNotification`) and the low-water mark
  (checked every 250 ms from the texture cache's frame tick) raise
  `Common::PlatformEvents::SignalMemoryPressure`: the texture cache collects as if out of device
  memory and the buffer cache collects unused buffers for 30 frames. While the process stays
  below the mark without getting worse, the signals come 1, 2, 4 ... 30 s apart.
- Experiments (`lichii_platform_set_experiment`, next launch, off by default): `file_guest`
  keeps guest memory in a mapped file in tmp, outside the footprint; `chunked_guest` gives
  all-zero 2 MiB pieces of it back. `VM_LEDGER_FLAG_NO_FOOTPRINT` is not one of them: iOS refuses
  it without a private entitlement.
- The `.nup` loader keeps 96 MiB of decompressed files in memory on iOS (`EDEN_NUP_CACHE_MIB`,
  512 MiB elsewhere) and decompresses files over 48 MiB to `cache/nup/*.tmp`; files a closed
  game left there are removed at launch. Experiment `nup_in_memory` keeps the 512 MiB.
- Every 30 s, at a memory warning and when iOS is about to close the app, the log says what the
  footprint is made of (`Memory regions`, `Memory malloc`, `Storage`: `LogMemoryBreakdown`).

## Lifecycle, threads, display

- Thread QoS: `Common::SetCurrentThreadPriority` maps to QoS classes on iOS (Critical/VeryHigh:
  user-interactive for guest cores, GPU thread and scheduler; High: user-initiated; Low: utility
  for shader builders). Scheduling parameters opted threads out of QoS before.
- `Info.plist`: `CADisableMinimumFrameDurationOnPhone` (120 Hz on ProMotion iPhones), game
  category + `LSSupportsGameMode`/`GCSupportsGameMode` (iOS 18 Game Mode).
- `UIScreen.maximumFramesPerSecond` feeds the frame pacer's display refresh
  (`vk_host_pacer.cpp`).
- Thermal: `ProcessInfo.thermalState` and Low Power Mode changes are logged and reported; the
  advice (frame generation off and 30 fps at serious, also pause at critical) is for the UI to
  apply, since frame generation and resolution cannot change while a game runs.

## TXM detection

`IosSupport::HasTxmIoKit()` (ios_support.mm) reads `IODeviceTree:/chosen/memory-map` for a
`TXM` entry (as MeloNX); `IosSupport::HasTxm()` uses it first and falls back to the model table.
