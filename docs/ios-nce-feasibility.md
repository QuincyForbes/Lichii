# NCE on iOS: feasibility notes (research only)

Status: not implemented, nothing here was tried on a device. NCE (`src/core/arm/nce/`) runs guest
AArch64 code natively: the patcher (`patcher.cpp`) rewrites SVC, `MRS/MSR TPIDR*_EL0` and
`CNTPCT` reads into branches to trampolines, guest memory is mapped at the guest's own virtual
addresses, and faults/preemption go through signals (`arm_nce.cpp`, `arm_nce.s`). Below is what
stands between that and iOS (iPhone/iPad, A15+/M, iOS 17-27), roughly from hardest to easiest.

## 1. Executable memory under TXM (iOS 26+): blocker on most devices

- Our JIT (`common/ios_jit.*`) gets executable memory once, from StikDebug's universal script:
  the debugger allocates R-X pages and touches each 16 KiB page through the debug connection;
  we write through a `vm_remap` RW alias. The pool is carved by a bump allocator and the
  debugger detaches afterwards (no more `brk` RPCs).
- NCE needs the guest's own `.text` executable *at the guest address* (next point). Pool pages
  live at addresses the debugger chose. Moving them would need `vm_remap` of prepared R-X pages to
  a fixed guest address; whether TXM accepts a remapped code page as prepared is unknown and is
  the first thing to probe (remap one prepared page to a fixed address, execute it, see if the
  process survives). If it is refused, every guest code page would need its own debugger
  preparation at its final address: possible only while StikDebug stays attached (each `brk`
  RPC costs a round trip; a 100 MB title is ~6,400 pages of 16 KiB), and every code change
  (self-patching, NRO loading, patcher trampolines) needs another.
- On iOS 17-25 (no TXM), a debugger having attached once (CS_DEBUGGED) allows RWX mappings
  anywhere, so this point disappears there; points 2-5 remain.

## 2. Guest code must run at the guest address

- Switch executables are position independent, but code reaches its data with PC-relative
  `ADRP/ADD/LDR` (and literal pools), so code and data must keep their relative layout: the code
  has to be mapped where the guest expects it, inside the guest address space (`host_memory.cpp`
  reserves it as one region; on iOS it is a Mach memory object mapped with `vm_map`).
- Relocating code elsewhere means rewriting every PC-relative instruction: that is a binary
  translator, not NCE.

## 3. 16 KiB host pages vs 4 KiB guest pages

- iOS on arm64 uses 16 KiB pages; Horizon maps and protects at 4 KiB. `host_memory.cpp` already
  refuses fastmem with NCE when the host page size differs ("does not support fastmem with NCE").
- Segment boundaries (`.text` R-X / `.rodata` R / `.data` RW) of a title are 4 KiB aligned, so a
  16 KiB host page can hold code and writable data: it cannot be both executable and writable
  under W^X / TXM. Workarounds (making mixed pages writable and trapping, or copying code into
  separate pages and patching references) bring back point 2.
- Memory tracking (write watches on 4 KiB pages for the GPU caches) loses precision 4x; dynarmic
  handles this today by not depending on host page protections at 4 KiB.

## 4. Platform register x18 and thread registers

- Apple's arm64 ABI reserves x18; XNU may zero or clobber it on context switches and exceptions
  (Apple documents it as reserved; it is not preserved for user code). Horizon code compiled by
  Nintendo's toolchain uses x18 as an ordinary register (it is not reserved there), so guest code
  running natively would see x18 change under it. Fixing it means patching every instruction that
  reads or writes x18 to use a spill slot via trampolines: a large, slow patcher change, and
  indirect uses (`ldp x17, x18`) make it worse. Must be confirmed with a probe (set x18 in a busy
  loop, yield, check) on current iOS versions.
- `TPIDRRO_EL0` holds XNU's per-thread data and is read only at EL0; NCE already patches guest
  reads of it, so this is fine. NCE also writes `TPIDR_EL0` (`msr tpidr_el0` in `arm_nce.s`) to
  switch between host and guest thread pointers; XNU keeps `TPIDR_EL0` per thread for user use,
  but this needs checking too.

## 5. The rest of NCE is Linux specific

- `arm_nce.s` uses ELF sections and `svc #0` with `__NR_tkill` to re-enter a signal handler;
  Darwin needs Mach-O symbol names, `pthread_kill`, and its `ucontext`/`__darwin_mcontext64`
  layout in the signal handlers (`arm_nce.cpp`).
- Guest addresses below 4 GiB fall into `__PAGEZERO` (4 GiB on arm64 iOS); guest address spaces
  of 39 bits (512 GiB) exceed what iOS gives an app unless extended virtual addressing grants
  enough (the platform layer logs whether 4 and 16 GiB reservations succeed, `va_reserve_*`;
  probing up to 512 GiB would settle it).
- A debugger attached while NCE runs (StikDebug during preparation) receives EXC_BAD_ACCESS /
  EXC_BREAKPOINT before our signal handlers: NCE must only start after it detaches.

## Conclusion

On iOS 26+ with TXM, NCE is not feasible with the current one-shot JIT protocol: guest code
must be executable at fixed guest addresses, which only a per-page debugger preparation could
provide, and 16 KiB pages mixing code and data cannot be W^X compliant. On iOS 17-25 the
executable memory problem goes away, but 16 KiB pages, x18 and the Linux specific entry code
still need large changes; the gain over dynarmic would also be capped by how often guest code
mixes code and data within 16 KiB.

Cheap probes worth running on a device before anyone revisits this:
1. `vm_remap` a debugger-prepared code page to a fixed address and execute it (TXM).
2. x18 survival across `sched_yield()` and a page fault.
3. Largest PROT_NONE reservation with extended-virtual-addressing (up to 512 GiB) and the
   lowest address mappable with `MAP_FIXED`.
