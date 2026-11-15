# Offline Metal replay

`lichii-metal-replay` pushes a game's stored pipelines and the draws of a feature census through
the Metal renderer's own translation (`PipelineTranslator`: recompiler, SPIR-V to MSL) and its
descriptor checks (`SanitizeVertexLayout`, `CheckVertexLayout`, `CheckPipeline`), and reports what
the renderer would refuse and why. It needs no Metal device, starts no emulator and only reads
its inputs.

## Build

The target exists when the Metal renderer itself is not built (any machine but a Mac):

```sh
cmake -DLICHII_METAL_COMPILE_CHECK=ON <build dir>
cmake --build <build dir> --target lichii-metal-replay
```

## Inputs

- `--cache <file>`: a Vulkan pipeline cache of the game (`<user dir>/cache/shader/<title
  id>/vulkan*.bin`, versions 52 to 54). Repeat it for several files; later ones count as newer.
  Every record is one pipeline the game used, with the environment its programs were translated
  from.
- `--census <file>`: `nx2_feature_census.json`, written to the log directory by a run with
  `EDEN_NX2_FEATURE_CENSUS=1` (one row per distinct draw, clear and dispatch state, with how
  often it happened). Must come from the same build as the replay. Start that run with the
  game's pipeline cache moved aside and pass the cache it writes, so every program of the census
  has an environment.

Both hold game data: keep them and the replay's output on the PC.

```sh
lichii-metal-replay --cache vulkan.fork-v53.bin --census nx2_feature_census.json \
    --report replay.txt --msl-out msl
```

`--macos` translates as the macOS renderer does (default: iOS), `--apple-family <n>` names the
GPU the MSL is for (`MTLGPUFamilyApple<n>`, default 10 = A19; 0 = not an Apple GPU; the
environment's `EDEN_NX2_METAL_MSL_DEVICE=0` translates as if it were unknown), `--jobs <n>` sets
the worker threads, `--msl-out <dir>` writes every distinct MSL function (`<hash>_<role>.metal`, and
`index.txt` with each file's entry point).

`--spirv <file>` (any number of them) translates plain SPIR-V files instead of a cache: the
stage comes from the file's name (`.vert.spv`, `.frag.spv`, anything else is compute), and the
MSL or the reason it was refused is printed. For looking at what a translation option does to a
shader of one's own; the exit code is 1 when one was refused.

## What the report says

- Stored pipelines: each record's key becomes the Metal renderer's key and is translated. Counted
  by result: translated (plain, mesh, task and mesh, tessellation), refused with the cause the
  renderer would log, and what the failing step said. A stored key holds no blending, vertex
  layout or exact topology when Vulkan had them as dynamic state, so only targets, sample count
  and the stored topology are checked there.
- Census draws: the row's registers are put back, the key is built as the rasterizer builds it,
  and the draw goes down the rasterizer's decisions: refused (cause), native, or emulated (index
  rewrite, mesh kernels, tessellation kernels, indirect loop). The render pipeline description is
  filled by the renderer's own code and run through its checks.
- Clears, dispatches, images by format and the other rasterizer calls, as far as the census
  holds them.
- "With guessed reads": no stored environment had a value this translation read (a constant
  buffer word or texture type Vulkan's translation did not need), so `GuessMissing` supplied it.
  The program is translated, but not necessarily as it would be in the game.

The report ends with what the replay takes for granted. It does not judge queries, copies, image
views or samplers.

## MSL is not compiled here

The replay stops at MSL text. Apple's compiler is needed to know that Metal takes it; on a Mac
with Xcode, over the `--msl-out` directory:

```sh
for f in msl/*.metal; do
    xcrun -sdk iphoneos metal -std=metal3.0 -ffast-math -c "$f" -o /dev/null || echo "FAILED $f"
done
```

(`-sdk macosx` for `--macos` output.) The renderer compiles the same text at
run time with fast math on and MSL 3.0.
