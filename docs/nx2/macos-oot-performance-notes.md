# OoT on macOS: performance and rendering notes (Nov 2026)

What was measured and changed while taking Ocarina of Time (0400486024DEC000) on an
M4 Pro from ~26-33 fps (Metal) / ~41 fps (Vulkan) with broken lighting to the current
state. All figures are lab runs with `eden-cli` (hidden window, debug server, Link's house
from the same save, then the walk out into Kokiri Forest).

## Where things stand

| | Indoors | Outdoors |
|---|---|---|
| Vulkan (MoltenVK) | 57-60 fps | 45-48 fps |
| Metal (native) | ~41 fps | ~32 fps |

Both draw the full scene now (grass, foliage, lit props). Earlier Vulkan builds were
faster only because they skipped about a third of each frame's draws.

## How to measure

- `EDEN_NX2_PERF_REPORT=1`: 10 s "Perf" lines. "Perf sched:" (guest threads: on a core,
  ready without one, longest wait), "Perf threads:" (host threads), "Perf guest"
  (system calls by number, slow memory callbacks, yields). Both backends print them.
- `EDEN_NX2_SPLIT_STATS=1`: "GPU split stalls" (front waits by site, back idle) and the
  "nx2 critical path per frame" line: frame = guest + present, syncpoint waits, vsync.
- `EDEN_NX2_PASS_TIMES=10` (Metal): one frame's render passes timed on the GPU.
- `EDEN_NX2_METAL_DRAW_TRACE=1`: per-pipeline draw diagnostics, task/mesh readbacks.
- `EDEN_NX2_HANDLE_FALLBACK_TRACE=1`: shaders whose texture handle could not be resolved.
- The debug server (`EDEN_NX2_DEBUG_SERVER`) can reject shaders live on Vulkan; with
  `tools/nx2/lab_ctl.py ablate` and a pixel metric this bisects a visual bug to one
  program in a few minutes.

## The limit today

The GPU threads have slack on both backends. The game's own CPU frame is the limiter:
~17.5 ms on Vulkan, ~24-25 ms on Metal. Emulator overhead is only ~9% of guest-core time;
the rest is game code (JIT) and guest threads waiting for each other. "Perf sched" shows
game threads ready without an emulated core ~225% of the time on both backends (a thread
pinned to cores 4-5 waits behind priority-44 workers). Why the same game code takes
longer per frame under Metal is not settled: per frame Metal has ~13% more system calls
(extra yields from spin-waits) and ~35% more slow memory write callbacks; guest lock
waits are small (~0.75 ms/frame); GPU syncpoint waits are under 1 ms/frame.

The display does not cap it: vsync off changes nothing, the emulated vsync runs at
240 Hz, and the built-in display is 120 Hz.

## Big wins (and why)

- **MoltenVK NoContraction** (Vulkan 39-43 -> 57-60): every guest FADD/FMUL/FFMA is
  NoContraction, and SPIRV-Cross turns each into a `[[clang::optnone]]` helper call.
  Fragment/compute stages drop the decoration on MoltenVK; position stages write the
  exact form as plain `fma()` (`EDEN_NX2_MVK_PRECISE`).
- **Metal per-draw sampler-reduction variants** (scene pass 10 ms -> 2 ms): the min/max
  rewrite wrapped every texture sample; it is now compiled only for draws that bind such
  a sampler.
- **Metal inlined NoContraction helpers** (scene pass 13 -> 10 ms).
- **Metal async fences** (~38 -> 50 fps before grass): the back thread no longer waits
  for the GPU at command list ends.
- **Metal pipeline archive off** (cold-cache pipelines 20 s -> sub-second): archive
  additions compiled every pipeline twice under a lock all builders waited on.

## Correctness fixes that mattered

- **Command list memory reuse**: OoT recycles queued command list memory through its own
  allocator (frames ahead). Lists now run the copy taken at submit (`TrustSubmitted`,
  `EDEN_NX2_TRUST_SUBMIT`). This removed the garbled engine state behind many crashes,
  stalls and broken draws, including the Kokiri Forest transition.
- **Bindless handles from buffers**: clustered-light shaders read each light's shadow-map
  handle from a table in a storage buffer; the guessed fallback made lit foliage/props
  white (Vulkan) or black (Metal). Dynamic handle arrays (`EDEN_NX2_DYNAMIC_HANDLE_ARRAYS`).
- **Metal storage buffers through pointers** (`indirect_offset`): grass task programs read
  their lists through the cbuf0+0x520 pointer table and asked for no work.
- **MoltenVK descriptor aliasing in argument buffers**: such pipelines failed to build;
  they are rebuilt without aliasing (only those; aliasing elsewhere is worth ~10 fps).
- Masked MOV (quad lane mask), FFMA constant negate, header byte-offset handles only for
  Switch Sports, slope depth bias on rescaled targets, Xbox stick axis inversion.

## Tried and reverted (do not repeat blindly)

- Fence batching / list-end deferral on Metal: a batched fence hung once; no fps gain.
- OoT E2C macro forwarding (`EDEN_NX2_OOT_E2C_FORWARD=1`): front waits moved elsewhere,
  and Vulkan hung entering Kokiri Forest.
- Deciding per list from parse plausibility whether memory was recycled: the game also
  legitimately patches queued lists; loading broke.
- Native Metal mesh pipelines by default: wrong leaf lighting and slower than the compute
  emulation (now only a fallback).
- Guest lock handoff for the buffer cache on Metal: lock waits were too small to matter.

## Open

- Vulkan's quarter-resolution tree-leaf pass (pixel 4785dd005ea05332) is still white;
  Metal draws it correctly. Next suspect: MoltenVK's compute mesh emulation.
- Metal's longer guest frame (see above).
- One intermittent Metal hang during a cold shader rebuild (not reproduced).
- `EDEN_NX2_METAL_RECORDER=1` (Metal encoder thread) works and frees the back thread, but
  is off by default until the guest frame stops being the limit.
