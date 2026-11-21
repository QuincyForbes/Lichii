"""Apple silicon host differential test, optional iOS link and microbenchmark."""

import argparse
import ctypes as C
from pathlib import Path
import statistics
import struct
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent


def check(tmp, args):
    runner = tmp / "runner"
    image = tmp / "guest.bin"
    flags = ["-std=c++20", "-O3", "-Wall", "-Wextra", "-Werror", HERE / "runner.cpp"]
    subprocess.run(["clang++", *flags, "-o", runner], check=True)
    if args.ios:
        device = tmp / "ios"
        subprocess.run(["xcrun", "--sdk", "iphoneos", "clang++", "-target", "arm64-apple-ios16.0",
                        *flags, "-Wl,-no_adhoc_codesign", "-o", device], check=True)
        subprocess.run(["xcrun", "vtool", "-show-build", device], check=True)

    def run(mode, budget, x0=0, x1=0, x2=0):
        out = subprocess.check_output([runner, image, mode, *map(str, (budget, x0, x1, x2))],
                                      text=True).split()
        return [out[0], *map(int, out[1:])]

    lib = tmp / "reference.dylib"
    subprocess.run(["clang", "-dynamiclib", HERE / "reference.S", "-o", lib], check=True)
    oracle = C.CDLL(str(lib))
    oracle.reference.argtypes = [C.c_uint64, C.c_uint64, C.POINTER(C.c_uint32)]
    oracle.reference.restype = C.c_uint64
    first = C.addressof(C.c_char.in_dll(oracle, "reference"))
    last = C.addressof(C.c_char.in_dll(oracle, "reference_end"))
    text = C.string_at(first, last - first)
    svc = 0xd40000e1  # svc #7
    def words(*items):
        return struct.pack(f"<{len(items)}I", *items)

    # Separate data area after RET, followed by the top-level return sentinel.
    image.write_bytes(text + words(0, 0, svc))
    for mode in ("cached", "decode"):
        for seed, n in [(0, 1), (7, 19), (2**32 - 7, 3), (2**64 - 8, 257)]:
            data = (C.c_uint32 * 2)()
            expected = oracle.reference(seed, n, data)
            assert expected == data[1]
            assert run(mode, 10_000, seed, n, len(text))[:5] == [
                "svc", len(text) + 12, expected, 7, 3 * n + 4]
        assert run(mode, 2, 3, 2)[:5] == ["budget", 8, 8, 0, 2]
        assert run(mode, 10, 0, 1, 999)[:5] == ["data", 12, 5, 0, 3]

    # Visit block 0 twice. First iteration stores a replacement ADD #9 over
    # the already decoded ADD #1; RET X3 jumps back to address zero.
    image.write_bytes(words(0x91000400, 0xd1000421, 0xb5000061, svc, 0,
                            0xb9000062, 0xd65f0060))
    if args.unicorn:
        from unicorn import Uc, UC_ARCH_ARM64, UC_MODE_ARM
        from unicorn.arm64_const import UC_ARM64_REG_PC, UC_ARM64_REG_X0, UC_ARM64_REG_X1, UC_ARM64_REG_X2
        ref = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
        ref.mem_map(0, 4096)
        ref.mem_write(0, image.read_bytes())
        ref.reg_write(UC_ARM64_REG_X1, 2)
        ref.reg_write(UC_ARM64_REG_X2, 0x91002400)
        ref.emu_start(0, 12, count=20)  # Stop immediately before SVC.
        assert ref.reg_read(UC_ARM64_REG_PC) == 12 and ref.reg_read(UC_ARM64_REG_X0) == 10
        assert ref.mem_read(0, 4) == words(0x91002400)
        print("PASS: independent Unicorn self-modifying branch fixture")
    for mode in ("cached", "decode"):
        assert run(mode, 20, 0, 2, 0x91002400)[:5] == ["svc", 16, 10, 7, 9]
    image.write_bytes(words(0xd65f0000, svc))  # ret x0
    assert run("cached", 10, 2)[:5] == ["fetch", 2, 2, 0, 1]
    image.write_bytes(words(0))
    assert run("cached", 10)[:5] == ["unsupported", 0, 0, 0, 0]
    print("PASS: native differential, budget, store invalidation and faults")

    if args.bench:
        image.write_bytes(text + words(0, 0, svc))
        times = {"decode": [], "cached": []}
        n = 10_000_000
        for iteration in range(7):
            for mode in (list(times) if iteration % 2 == 0 else reversed(times)):
                stop, _, result, _, retired, decodes, ns = run(mode, 4 * n, 1, n, len(text))
                assert (stop, result, retired) == ("svc", 1 + 5*n, 3*n + 4)
                if iteration:
                    times[mode].append(ns / retired)
                counts = 3*n + 4 if mode == "decode" else 7
                assert decodes == counts
        for mode, samples in times.items():
            print(f"{mode}: median {statistics.median(samples):.2f} ns/guest instruction "
                  f"(range {min(samples):.2f}–{max(samples):.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ios", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--unicorn", action="store_true", help="also use host-only Unicorn 2.1.4 oracle")
    with tempfile.TemporaryDirectory() as directory:
        check(Path(directory), parser.parse_args())
