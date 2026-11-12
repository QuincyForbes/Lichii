// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <optional>
#include <string>

// iOS JIT substrate.
//
// A sideloaded app cannot create executable memory itself. On iOS 17-25 it is enough that a
// debugger has been attached once (CS_DEBUGGED): RWX mappings then work. On iOS 26+ with the
// Trusted Execution Monitor (TXM) that is no longer enough: every executable page has to be
// prepared through the debug connection. StikDebug (on-device debugger) attaches and runs its
// "universal" script, which services a breakpoint-based RPC: the app executes `brk #0xf00d` with
// x16 = command, x0/x1 = arguments; the debugger allocates an R-X region in the process, touches
// every 16 KiB page through the debugger (which is what makes the pages usable) and returns the
// address in x0. The app then makes a writable alias of the same pages with vm_remap and emits
// code through the alias while executing through the R-X view.
//
// Allocation is one-shot: after the pool is acquired the debugger is detached and no further
// `brk` may be issued (an unserviced `brk` is a fatal SIGTRAP, guarded here by a SIGTRAP handler
// that turns it into a null result). Every dynarmic code block is therefore carved out of this
// pool, prepared in ChunkBytes chunks before the emulated CPU starts.
namespace Common::IosJit {

/// dynarmic's code cache per guest core on iOS: a full cache is flushed and translated again.
/// The pool is prepared in chunks of this size, each one guest core's code block, so progress
/// shows and no block straddles two chunks.
inline constexpr std::size_t DefaultCodeCacheBytes = 32 * 1024 * 1024;
/// The device profile's size (Settings::DeviceProfile, jit_cache_mib); set before StartAcquire.
void SetCodeCacheBytes(std::size_t bytes);
std::size_t CodeCacheBytes();

enum class State {
    Waiting,   ///< no debugger yet
    Acquiring, ///< debugger attached, pool being prepared
    Ready,     ///< pool ready, code may be emitted
    Failed,    ///< see GetError()
};

/// True while a debugger is attached (P_TRACED).
bool DebuggerAttached();
/// True once the process has been debugged (CS_DEBUGGED).
bool Debugged();
/// True after the pool is ready.
bool PoolReady();

/// Start the background thread that waits for the debugger and acquires a pool of `bytes`.
/// `txm`: this device requires the breakpoint protocol (iOS 26+ with TXM); otherwise a plain
/// RWX mapping is used once CS_DEBUGGED is set.
void StartAcquire(std::size_t bytes, bool txm);
State GetState();
std::string GetError();
/// Bytes of the pool prepared so far (for progress while Acquiring).
std::size_t PreparedBytes();
/// Bytes of the pool code blocks have been carved from.
std::size_t HandedOutBytes();

/// Called on the acquiring thread whenever the state or the prepared bytes change.
using StateCallback = void (*)();
void SetStateCallback(StateCallback callback);

/// Carve a dual-mapped block out of the pool. *wmem is the writable alias, *xmem the executable
/// view of the same pages. Returns false when the pool is missing or exhausted.
bool Alloc(std::size_t bytes, void** wmem, void** xmem);

/// Hands the whole pool out again. Only after every code block is gone (the emulated process was
/// shut down), so a second game in the same app session reuses the prepared pages.
void Reset();

/// Drops duplicate pmap PTEs of the writable alias in [wmem, wmem + size) (page-aligned inward to
/// 16 KiB) on dual-mapped TXM chunks while keeping the executable view untouched and wmem valid.
void DropWritablePtes(void* wmem, std::size_t size);

struct PoolLocation {
    std::size_t chunk;
    std::size_t offset;
    bool writable_view; ///< In the writable alias, which is never executable
    bool handed_out;    ///< Below the chunk's allocation mark
};
/// Where `address` lies in the pool, for crash logs. Takes no lock (the chunk list does not change
/// once the pool is ready), so a signal handler may call it.
std::optional<PoolLocation> Locate(const void* address);

} // namespace Common::IosJit

extern "C" {
/// C entry point used by the patched oaknut::CodeBlock.
int lichii_ios_jit_alloc(std::size_t bytes, void** wmem, void** xmem);
}
