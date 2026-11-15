# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""JIT for Lichii from a Mac's lldb, without StikDebug (experimental, not yet run on a device).

Serves the breakpoint protocol of common/ios_jit.cpp the way StikDebug's universal.js does:
the app executes `brk #0xf00d` with x16 = 1 (x0 = address or 0, x1 = length) to have an R-X region
allocated (gdb-remote `_M<len>,rx`) and every 16 KiB page of it written once through the
debugger (`M<addr>,1:69`), which is what TXM needs, and x16 = 0 to have the debugger detach.

    xcrun lldb --batch \\
      -o "settings set platform.plugin.darwin.ignored-exceptions EXC_BAD_ACCESS|EXC_BAD_INSTRUCTION" \\
      -o "device select <udid>" -o "device process attach --pid <pid>" \\
      -o "command script import jit_lldb.py" -o "lichii_jit_serve --timeout 600"

bench_mac.py run --jit lldb does this (request "jit": {"provider": "external"}).
"""

import argparse
import shlex
import sys
import time

import lldb

BRK_F00D = 0xD43E01A0
PAGE = 0x4000
CMD_DETACH = 0
CMD_PREPARE_REGION = 1


def register(frame, name):
    return frame.FindRegister(name).GetValueAsUnsigned()


def set_register(frame, name, value):
    error = lldb.SBError()
    frame.FindRegister(name).SetValueFromCString(hex(value), error)
    return error.Success()


def prepare_region(process, address, length, out):
    if address == 0 and length == 0:
        return 0
    if address == 0:
        error = lldb.SBError()
        address = process.AllocateMemory(
            length, lldb.ePermissionsReadable | lldb.ePermissionsExecutable, error)
        if not error.Success() or address in (0, lldb.LLDB_INVALID_ADDRESS):
            print(f"lichii_jit: allocating {length >> 20} MiB R-X failed: {error}", file=out, flush=True)
            return 0
    start = time.monotonic()
    error = lldb.SBError()
    for page in range(address, address + length, PAGE):
        if process.WriteMemory(page, b"\x69", error) != 1 or not error.Success():
            print(f"lichii_jit: writing page {page:#x} failed: {error}", file=out, flush=True)
            return 0
    print(f"lichii_jit: prepared {length >> 20} MiB at {address:#x} in "
          f"{time.monotonic() - start:.1f} s", file=out, flush=True)
    return address


def stopped_thread(process):
    for thread in process:
        if thread.GetStopReason() not in (lldb.eStopReasonNone, lldb.eStopReasonInvalid):
            return thread
    return None


def serve(debugger, command, exe_ctx, result, internal_dict):
    parser = argparse.ArgumentParser(prog="lichii_jit_serve")
    parser.add_argument("--timeout", type=float, default=600.0)
    try:
        args = parser.parse_args(shlex.split(command))
    except SystemExit:
        result.SetError("usage: lichii_jit_serve [--timeout seconds]")
        return
    out = sys.stdout  # lldb shows the result object only when the command returns
    debugger.SetAsync(False)
    process = debugger.GetSelectedTarget().GetProcess()
    if not process.IsValid():
        result.SetError("no process: attach first (device process attach --pid <pid>)")
        return
    # Continue past a brk without handing the app its SIGTRAP.
    signals = process.GetUnixSignals()
    sigtrap = signals.GetSignalNumberFromName("SIGTRAP")
    signals.SetShouldSuppress(sigtrap, True)
    signals.SetShouldStop(sigtrap, True)
    deadline = time.monotonic() + args.timeout
    prepared = 0
    while time.monotonic() < deadline:
        error = process.Continue()
        state = process.GetState()
        if state in (lldb.eStateExited, lldb.eStateDetached, lldb.eStateCrashed):
            print(f"lichii_jit: process ended (state {state}) after {prepared >> 20} MiB", file=out, flush=True)
            return
        if not error.Success() and state != lldb.eStateStopped:
            print(f"lichii_jit: continue failed: {error}", file=out, flush=True)
            return
        thread = stopped_thread(process)
        if thread is None:
            continue
        frame = thread.GetFrameAtIndex(0)
        pc = frame.GetPC()
        read_error = lldb.SBError()
        word = process.ReadUnsignedFromMemory(pc, 4, read_error)
        if not read_error.Success() or word != BRK_F00D:
            # Something else stopped the app (a signal, an exception): let it go on.
            continue
        x16 = register(frame, "x16")
        x0 = register(frame, "x0")
        x1 = register(frame, "x1")
        frame.SetPC(pc + 4)
        if x16 == CMD_DETACH:
            print(f"lichii_jit: the app asked to detach after {prepared >> 20} MiB", file=out, flush=True)
            process.Detach()
            return
        if x16 == CMD_PREPARE_REGION:
            address = prepare_region(process, x0, x1, out)
            if address:
                prepared += x1
            set_register(frame, "x0", address)
            continue
        print(f"lichii_jit: unknown command {x16}", file=out, flush=True)
    print(f"lichii_jit: gave up after {args.timeout:.0f} s, detaching", file=out, flush=True)
    process.Detach()


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand(
        f"command script add -o -f {__name__}.serve lichii_jit_serve")
