# One app, guest programs as data

`runner.cpp` is a standalone CPU experiment for little-endian hosts. Build it
**once**, then point that same executable at different raw AArch64 programs:

```sh
clang++ -std=c++20 -O3 tools/ios/jitless/runner.cpp -o runner
./runner guest.bin cached 100000 0 1 0  # budget and initial X0, X1, X2
```

Guest PC starts at zero; the harness sets LR to the last word of the input
(typically an SVC). Output is `stop pc x0 svc retired decoded elapsed_ns`.
Runtime translation only constructs arrays of operation numbers, operands and
immediates in ordinary data memory. Instruction handlers are compiled into
the app. There is no per-title build/signing step, JIT fallback, guest-to-host
code pointer, or executable allocation. This supersedes the adjacent AOT
experiment for the single-app goal.

## What was tested

On Apple silicon with Xcode:

```sh
python3 -I tools/ios/jitless/check.py --ios --bench
```

The test assembles `reference.S` into a separate **macOS-only oracle**, calls
it on real hardware, then passes the *same instruction bytes as a file* to
the runner. It compares results for backward conditional branches (taken and
not taken), RET, wrapping arithmetic, and a 32-bit store/zero-extending load.
It checks a mid-block budget exit, unmapped memory, misaligned PC, and unknown
instructions. The second fixture stores a replacement ADD over a **previously
cached** block and jumps back to it: X0 must be 10, not the stale result 2.
For a second independent check of this fixture add `--unicorn` using a host
Python environment with `unicorn==2.1.4`. Unicorn is not linked to either runner.

Both host references passed. The generic runner alone also linked as an
unsigned arm64 iPhoneOS executable (minOS 16, SDK 27.1); no device execution
or signing was performed.

| M4 Pro, Apple clang 21, `-O3` | Median ns/guest instruction | Range |
|---|---:|---:|
| Re-decode each block visit | 3.38 | 3.33–3.47 |
| Reuse decoded blocks | 1.70 | 1.69–1.72 |

One warm-up plus six timed runs per mode in alternating order, cold cache at
process start. The internal timer excludes file I/O and process startup. Each
sample retires 30,000,004 instructions, dominated by ADD/SUB/CBNZ; decode counts
are 30,000,004 versus 7. The two modes share the executor. The roughly 2× gain
measures repeated decoding on this loop, **not** CPU coverage, phone throughput,
game FPS, or superiority to another engine.

## Engine choices for a real backend

* **Lichii/Dynarmic:** `src/dynarmic/src/dynarmic/frontend/{A64,A32}/translate`
  already translates guest bytes to `IR::Block`, including ARM and Thumb in
  A32. The `Translate`/`TranslateSingleInstruction` interfaces separate fetch
  from translation. `ir/opcodes.inc` and `ir/terminal.h` define operations
  and control-flow exits, not an interpreter. The NCE `InterpreterVisitor` in
  `src/core/arm/nce/` handles selected memory accesses, not full CPU execution.
  **Recommended integration experiment:** lower existing IR to compact typed
  records once per block and execute them with signed handlers. Compare that
  against direct decoded ops on the same workloads: SSA temporaries, flags
  and helper calls can expand one guest instruction to many dispatches.
  Preserve per-instruction fault PCs, retirement and polling before trying
  fused handlers. This prototype uses a small direct decoder, not Dynarmic IR.
* **[QEMU TCG interpreter](https://gitlab.com/qemu-project/qemu/-/tree/master/tcg/tci)
  / [UTM SE threaded interpreter](https://github.com/ktemkin/qemu/blob/with_tcti/tcg/aarch64-tcti/README.md):**
  credible broad ARM32/ARM64 comparison; UTM SE already ships a JITless
  interpreter on iOS. The latter uses prebuilt handlers on an AArch64 host.
  Neither is a drop-in `ArmInterface`: evaluate CPU-only embedding, QEMU
  memory/TLB callbacks, precise interrupt/SVC exits, build size and current
  ARM feature coverage. No comparative timing or port is claimed.
* **[VIXL A64 simulator](https://github.com/Linaro/vixl):** useful independent
  reference; its documented coverage is tied to its assembler, with limits on
  exclusive-access simulation. It does not solve A32 guest execution.
* **[Azahar FastInterp and DynCom](https://github.com/azahar-emu/azahar/tree/master/src/core/arm):**
  inspect for A32. FastInterp decodes ARM/Thumb16/Thumb32, caches decoded blocks
  and fuses common sequences. Its ARM64 handler file targets the *host*, not
  A64 guests. It assumes a 3DS CPU model and Azahar memory/timing services;
  validate Switch A32 ISA, VFP/NEON, atomics and thread/exception behavior
  rather than transplanting it as a complete Switch engine.

Eglantine upstream PR/issue and remote-history checks found no ready JITless
CPU backend. No third-party engine was imported.

## Limits and next milestone

The prototype has just register-only A64 ADD/SUB immediate, CBNZ, RET,
32-bit unsigned-offset STR/LDR and SVC in selected register forms. Blocks
hold up to 16 operations in a 64-entry direct-mapped cache. A store ends the
block and invalidates all overlapping entries immediately. That is conservative
single-thread coherence on one flat writable byte image; production needs page
permissions, executable-page tracking, aliases, remaps, external writers,
cross-core invalidation and guest cache-maintenance semantics.

Switch 1 needs A64 **and**, for some titles, A32 ARM/Thumb interworking, IT
state, CPSR and different syscall/context conventions. Switch 2 also needs
its actual A64 extension coverage. Neither is bootable here. There are no
flags, FP/SIMD, TLS, exclusives, barriers, MMU or process services. A real
cache key must include address space and translation-relevant CPU state.
Indirect branches and newly loaded modules should resolve via guest virtual
addresses and decode on demand; no offline target list or game-specific
signing is necessary.

The next useful milestone is a public multi-block module through real guest
memory and `ArmInterface` with a full thread context, bounded interrupt
latency, and SVC returning to `physical_core.cpp`/`Svc::Call`. Compare against
the current desktop CPU backend instruction-by-instruction, including A32
mode switches and faults, then benchmark memory-heavy, indirect, SIMD and
multi-thread workloads on a device. Renderer, services and app packaging
remain separate from the CPU experiment.
