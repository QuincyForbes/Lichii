# Offline guest CPU code for iOS: link experiment

`python3 -I tools/ios/aot/check.py` builds and runs a synthetic eight-instruction
guest with a host C11 compiler (`CC`, default `cc`). Add `--ios` on a Mac with
Xcode to cross-compile and link the same two translation units for iPhoneOS.
The script uses a temporary directory. It does not build an IPA, provision,
install or run a device app.

`lift.py guest.bin guest.c` takes a flat little-endian text section at guest
address zero. It decodes *at build time*, rejecting every unrecognised word,
and emits a C switch on guest PC with translated operations, not an opcode
decoder. The generated object is linked into the test executable's text. This
is the object one would link into an app target **before normal app signing**.
No guest bits are cast to host code pointers; no code cache or runtime executable
memory allocation is needed. This experiment has no dependency on Lichii's CPU
backend and does not select a new backend in the app.

The entire guest is the annotated `WORDS` array in `check.py`. It calls a
subroutine via `BLR X30`, adds 40, returns via LR, and yields at `SVC #0x42`.
The harness supplies a synthetic result, resumes and checks the next SVC sees
42. It also checks instruction-budget suspension, an unmapped high indirect
target, a misaligned PC, and rejection of incomplete/unsupported input.
Only 64-bit MOVZ, ADD immediate (general registers), BR/BLR/RET (general
registers) and SVC are accepted. There is no memory instruction or guest OS.

Verified on macOS: host execution passed; the device link produced an unsigned
arm64 executable (`LC_BUILD_VERSION`: IOS, minOS 16.0, SDK 27.1). The latter
tests target ABI and linking, not execution on iOS.

## What to reuse from Longshot

Source inspected: [MagicBeanDev/Longshot, b62932a9](https://git.magicbean.dev/MagicBeanDev/Longshot/src/commit/b62932a9607ff026b1c12eb9df168a9556b59264).
`nx2/lift.py` uses a64-kit and Switch-specific lowering to produce C chunks;
`image.py` models Eglantine's module layout and relocations; `discover.py`
walks functions and recovers jump tables. Those are better candidates for a
full lifter evaluation than extending the tiny decoder here. No Longshot code
or generated game code is included.

In particular, `nx2/rt/mkw_host.h` defines versioned attach/lookup, a CPU
context and callbacks for SVC, polls, traps, counter reads and indirect-call
misses. `nx2/test_dll.py` exercises the emulator ABI in a Windows DLL;
`nx2/rtgen.py` assumes a specific fastmem window. Static linking can replace
the DLL boundary, but not the memory assumptions or the JIT fallback suggested
by `call_miss`. Its standalone SDK hooks/stubs are a separate runtime, not
Lichii HLE. The prototype ABI intentionally only models the exits tested here.

Eglantine upstream check: fetched `upstream/nx2` (`54515b3368`), searched PRs
and issues (all states), branch names and history for Longshot/offline AOT/no-JIT,
and examined the `nx2-aarch64` NCE diff (visitor methods, not an offline
backend). No matching implementation found. The shared checkout was on `main`;
the two upstream-only commits changed release CI and README only.

## Remaining boot gates

1. **Identify every executable module.** Match the loaded bytes (including
   patches), build IDs, text ranges, ASLR bases, relocations and imports for
   rtld/main/subsdk/sdk against a manifest shipped with the signed code.
   `src/core/loader/{deconstructed_rom_directory,nso}.cpp` and
   `KProcess::LoadModule` own the initial layout. Dynamic NRO loads/unloads
   and invalidations must register/remove only matching precompiled ranges.
   Arbitrary additional titles/updates cannot gain new native text after signing.
2. **Dispatch by guest address.** Resolve indirect calls, jumps, returns,
   jump tables, callbacks, thread entries and arbitrary resume PCs, including
   across modules. Handle unknown targets, executable writes and unsupported
   opcodes explicitly: stop with a diagnostic or design a data-only interpreter.
   Never use the NCE/JIT on a lookup miss. Verify code discovery, relocations
   and instruction semantics against a reference on synthetic binaries first.
3. **Implement a real CPU and memory contract.** `src/core/arm/arm_interface.h`
   needs context transfer (SP, NZCV, SIMD/FP, TLS), stepping, interrupts,
   cache invalidation and faults. `KProcess::InitializeInterfaces` in
   `src/core/hle/kernel/k_process.cpp` is the backend-selection area.
   Lower loads/stores via guest mappings with permissions/faults, and implement
   atomics, exclusives, barriers, exceptions and timers. Measure supported iOS
   address-space and memory limits rather than assuming a 39-bit fastmem arena.
4. **Cross the actual HLE boundary.** Preserve per-thread state, expose the SVC
   number and argument/result registers, and return `HaltReason::SupervisorCall`
   to `src/core/hle/kernel/physical_core.cpp`, which invokes `Svc::Call`.
   Support block/wake, context switches, interrupts at bounded intervals and
   IPC/service memory access. A C callback returning 41 does not test any of this.
5. **Integrate and validate the app.** Link all matching generated objects and
   metadata into the normal app target; audit all CPU execution paths and
   helpers for runtime code generation. Boot a public synthetic module through
   the real loader/kernel before testing a game. GPU translation, rendering,
   input, audio and filesystem services remain separate boot dependencies.
