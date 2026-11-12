#!/usr/bin/env python3
"""Expected results of oot_texture_mod.py on synthetic BNTX files, for the C++ port's unit test
(src/tests/core/texture_quality.cpp). Prints one line per case: length and FNV-1a 64 of the output.

  texture_quality_vectors.py              the table the test holds
  texture_quality_vectors.py --compare D  checks the real files the hidden test case
                                          "[.texture_quality_files]" wrote to D
"""

import importlib.util
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load("oot_texture_mod")
make_bntx = _load("test_oot_texture_mod").make_bntx

# name, make_bntx arguments, (brti offset, struct format, value) patches, levels (0 = drop_top_mip
# once without the size rule), min size
CASES = [
    ("bc4_512_l1", dict(), [], 1, 512),
    ("bc4_512_l2", dict(), [], 2, 512),
    ("bc4_512_drop", dict(), [], 0, 512),
    ("astc5x5_1000x600_l1", dict(width=1000, height=600, mips=8, fmt=0x2F01), [], 1, 512),
    ("astc5x5_1000x600_l2", dict(width=1000, height=600, mips=8, fmt=0x2F01), [], 2, 512),
    ("astc8x8_2048x64_l2", dict(width=2048, height=64, mips=6, fmt=0x3401, block_height_log2=1),
     [], 2, 512),
    ("r8_640x16_l1", dict(width=640, height=16, mips=4, fmt=0x0201, block_height_log2=0), [], 1, 512),
    ("bc1_300x200_drop", dict(width=300, height=200, mips=5, fmt=0x1A01), [], 0, 512),
    ("bc1_300x200_l1", dict(width=300, height=200, mips=5, fmt=0x1A01), [], 1, 512),
    ("two_mips_l2", dict(mips=2), [], 2, 512),
    ("one_mip_l1", dict(mips=1), [], 1, 512),
    ("cube_l1", dict(), [(0x30, "<i", 6)], 1, 512),
    ("dim3_l1", dict(), [(0x11, "<B", 3)], 1, 512),
    ("tiled_l1", dict(), [(0x12, "<H", 1)], 1, 512),
    ("unknown_format_l1", dict(fmt=0x7701), [], 1, 512),
]


def build(args, patches):
    data = bytearray(make_bntx(**args))
    brti, = struct.unpack_from("<Q", data, struct.unpack_from("<Q", data, 0x28)[0])
    for offset, fmt, value in patches:
        struct.pack_into(fmt, data, brti + offset, value)
    return bytes(data)


def reduce(data, levels, min_size):
    if levels == 0:
        return mod.drop_top_mip(data)
    tex = mod.parse_bntx(data)
    if len(tex) != 1 or max(tex[0].width, tex[0].height) < min_size:
        return None
    reduced = data
    for _ in range(levels):
        smaller = mod.drop_top_mip(reduced)
        if smaller is None:
            break
        reduced = smaller
    return None if reduced is data else reduced


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def compare(directory):
    """<name>.bntx originals and <name>.l1/.l2 written by the hidden C++ test case."""
    checked = bad = 0
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".bntx"):
            continue
        with open(os.path.join(directory, name), "rb") as f:
            data = f.read()
        for levels in (1, 2):
            try:
                want = reduce(data, levels, 512)
            except (struct.error, ValueError, IndexError):
                want = None
            path = os.path.join(directory, f"{name[:-5]}.l{levels}")
            got = None
            if os.path.exists(path):
                with open(path, "rb") as f:
                    got = f.read()
            checked += 1
            if got != want:
                bad += 1
                print(f"MISMATCH {name} levels {levels}: tool {None if want is None else len(want)}, "
                      f"C++ {None if got is None else len(got)}")
    print(f"{checked - bad} of {checked} equal")
    return 1 if bad else 0


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--compare":
        return compare(sys.argv[2])
    for name, args, patches, levels, min_size in CASES:
        data = build(args, patches)
        out = reduce(data, levels, min_size)
        if out is None:
            print(f"{name}: unchanged (input {len(data)} bytes, FNV {fnv1a64(data):#018x})")
        else:
            print(f"{name}: {len(out)} bytes, FNV {fnv1a64(out):#018x} (input {len(data)}, "
                  f"FNV {fnv1a64(data):#018x})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
