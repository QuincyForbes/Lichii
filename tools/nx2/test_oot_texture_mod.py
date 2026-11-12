#!/usr/bin/env python3
"""Tests for oot_texture_mod.drop_top_mip on a synthetic BNTX (no game data)."""

import importlib.util
import os
import struct
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("oot_texture_mod",
                                               os.path.join(HERE, "oot_texture_mod.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def make_bntx(width=512, height=512, mips=10, fmt=0x1D01, block_height_log2=4) -> bytes:
    """A one-texture BNTX laid out like the game's (BC4, data at 0x1000, RLT at the end)."""
    out = bytearray(0x1000)
    out[:4] = b"BNTX"
    struct.pack_into("<IHBB", out, 8, 0x00040101, 0xFEFF, 0x0C, 0x40)
    info_array, brti, mip_ptr, name_ptr, brtd = 0x198, 0x1F8, 0x498, 0x1B8, 0xFF0
    struct.pack_into("<4sIQQ", out, 0x20, b"Ounc", 1, info_array, brtd)
    struct.pack_into("<Q", out, info_array, brti)
    struct.pack_into("<H", out, name_ptr, 4)
    out[name_ptr + 2:name_ptr + 6] = b"Test"
    out[brti:brti + 4] = b"BRTI"
    struct.pack_into("<BBHHHIIIiiiiii", out, brti + 0x10, 9, 2, 0, 0, mips, 1, fmt, 0x21,
                     width, height, 1, 1, block_height_log2, 0x100C9)
    sizes = []
    w, h = width, height
    for _ in range(mips):
        sizes.append(max(512, ((w + 3) // 4) * ((h + 3) // 4) * 8))
        w, h = max(1, w // 2), max(1, h // 2)
    offsets = [0x1000]
    for size in sizes[:-1]:
        offsets.append(offsets[-1] + size)
    image_size = sum(sizes)
    struct.pack_into("<II", out, brti + 0x50, image_size, 512)
    struct.pack_into("<QQQ", out, brti + 0x60, name_ptr, 0, mip_ptr)
    struct.pack_into(f"<{mips}Q", out, mip_ptr, *offsets)
    out[brtd:brtd + 4] = b"BRTD"
    struct.pack_into("<Q", out, brtd + 8, 0x10 + image_size)
    data = bytes((i * 7) & 0xFF for i in range(image_size))
    out += data
    rlt = len(out)
    rel = bytearray(16 + 2 * 0x18 + 2 * 8)
    rel[:4] = b"_RLT"
    struct.pack_into("<III", rel, 4, rlt, 2, 0)
    struct.pack_into("<QIIII", rel, 16, 0, 0, mip_ptr + 8 * mips, 0, 1)
    struct.pack_into("<QIIII", rel, 16 + 0x18, 0, brtd, 0x10 + image_size, 1, 1)
    struct.pack_into("<IHBB", rel, 16 + 0x30, 0x30, 1, 1, 0)
    struct.pack_into("<IHBB", rel, 16 + 0x38, mip_ptr, 1, mips, 0)
    out += rel
    struct.pack_into("<II", out, 0x18, rlt, len(out))
    return bytes(out)


class DropTopMipTest(unittest.TestCase):
    def test_halves_the_texture_and_keeps_the_lower_mips(self):
        data = make_bntx()
        old = mod.parse_bntx(data)[0]
        reduced = mod.drop_top_mip(data)
        self.assertIsNotNone(reduced)
        new = mod.parse_bntx(reduced)[0]
        self.assertEqual((new.width, new.height, new.mips), (256, 256, 9))
        self.assertEqual(new.block_height_log2, 3)
        top = old.mip_offsets[1] - old.mip_offsets[0]
        self.assertEqual(len(reduced), len(data) - top)
        self.assertEqual(new.image_size, old.image_size - top)
        self.assertEqual(new.mip_offsets, [0x1000] + [o - top for o in old.mip_offsets[2:]])
        self.assertEqual(reduced[0x1000:0x1000 + new.image_size],
                         data[old.mip_offsets[1]:old.mip_offsets[1] + new.image_size])
        rlt, size = struct.unpack_from("<II", reduced, 0x18)
        self.assertEqual(size, len(reduced))
        self.assertEqual(reduced[rlt:rlt + 4], b"_RLT")
        self.assertEqual(struct.unpack_from("<I", reduced, rlt + 4)[0], rlt)
        self.assertEqual(struct.unpack_from("<I", reduced, rlt + 16 + 0x18 + 12)[0],
                         0x10 + new.image_size)
        self.assertEqual(reduced[rlt + 16 + 0x38 + 6], 9)

    def test_twice(self):
        reduced = mod.drop_top_mip(mod.drop_top_mip(make_bntx()))
        new = mod.parse_bntx(reduced)[0]
        self.assertEqual((new.width, new.height, new.mips, new.block_height_log2), (128, 128, 8, 2))

    def test_single_mip_is_left_alone(self):
        self.assertIsNone(mod.drop_top_mip(make_bntx(mips=1)))


if __name__ == "__main__":
    unittest.main()
