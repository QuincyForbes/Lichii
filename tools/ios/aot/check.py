"""Run host execution tests; --ios additionally links an unsigned device Mach-O."""

import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
# Original synthetic guest, in order of guest PC:
WORDS = [
    0xd280031e,  # movz x30, #24
    0xd63f03c0,  # blr x30 (read target before writing LR)
    0xd4000841,  # svc #0x42
    0x91000400,  # add x0, x0, #1
    0xd4000001,  # svc #0
    0xd61f0020,  # br x1
    0x9100a000,  # add x0, x0, #40
    0xd65f03c0,  # ret
]


def check(work, ios):
    text = work / "guest.bin"
    source = work / "guest.c"
    lift = [sys.executable, "-I", HERE / "lift.py", text, source]
    for data, message in [(b"x", "4-byte"), (bytes(4), "unsupported A64 word")]:
        text.write_bytes(data)
        bad = subprocess.run(lift, capture_output=True, text=True)
        if bad.returncode == 0 or message not in bad.stderr or source.exists():
            raise RuntimeError(f"invalid text not rejected as expected: {bad.stderr}")
    text.write_bytes(struct.pack(f"<{len(WORDS)}I", *WORDS))
    subprocess.run(lift, check=True)
    flags = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-O2", "-I", HERE]
    for device in ([False, True] if ios else [False]):
        cc = (["xcrun", "--sdk", "iphoneos", "clang", "-target", "arm64-apple-ios16.0"]
              if device else [os.environ.get("CC", "cc")])
        obj = work / "guest.o"
        exe = work / "smoke"
        subprocess.run([*cc, *flags, "-c", source, "-o", obj], check=True)
        link = ["-Wl,-no_adhoc_codesign"] if device else []
        subprocess.run([*cc, *flags, HERE / "smoke.c", obj, *link, "-o", exe], check=True)
        if device:
            subprocess.run(["xcrun", "vtool", "-show-build", exe], check=True)
        else:
            subprocess.run([exe], check=True)
            print("Host execution passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ios", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        check(Path(tmp), args.ios)
