#!/usr/bin/env python3
"""Smaller textures for Ocarina of Time (Switch 2) as a LayeredFS mod, made from the owner's game.

The game keeps one texture per `Tex/<name>.bntx.zs` (zstd, no dictionary). Dropping the top mip
level of a texture cuts its memory by about 75 %. The mod goes into the title's mod folder, e.g.
`%APPDATA%/eden/load/0400486024DEC000/Reduced textures/romfs/Tex/...`, and can be switched off like
any other mod. Nothing here leaves the owner's machine: never commit or upload what it writes.

  oot_texture_mod.py survey <game.nup> [--list LOG] [--csv out.csv]
  oot_texture_mod.py make <game.nup> <mod dir> [--list LOG] [--min-size 512] [--levels 1]

--list takes a Lichii log made with EDEN_NX2_NUP_READ_LOG=1 and limits the work to the textures
the game read in that session (the `NUP first read: #<index> ...` lines).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import struct
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("nup_tool", os.path.join(HERE, "nup_tool.py"))
nup_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nup_tool)

try:
    import zstandard  # type: ignore
except ImportError:
    zstandard = None

# BNTX/NVN format type (high byte) -> (block width, block height, bytes per block, name).
FORMATS = {
    0x02: (1, 1, 1, "R8"), 0x07: (1, 1, 2, "R5G6B5"), 0x09: (1, 1, 2, "R8G8"),
    0x0A: (1, 1, 2, "R16"), 0x0B: (1, 1, 4, "R8G8B8A8"), 0x0C: (1, 1, 4, "B8G8R8A8"),
    0x0E: (1, 1, 4, "R10G10B10A2"), 0x0F: (1, 1, 4, "R11G11B10"), 0x12: (1, 1, 8, "R16G16B16A16"),
    0x14: (1, 1, 4, "R32"), 0x1A: (4, 4, 8, "BC1"), 0x1B: (4, 4, 16, "BC2"),
    0x1C: (4, 4, 16, "BC3"), 0x1D: (4, 4, 8, "BC4"), 0x1E: (4, 4, 16, "BC5"),
    0x1F: (4, 4, 16, "BC6H"), 0x20: (4, 4, 16, "BC7"),
    0x2D: (4, 4, 16, "ASTC4x4"), 0x2E: (5, 4, 16, "ASTC5x4"), 0x2F: (5, 5, 16, "ASTC5x5"),
    0x30: (6, 5, 16, "ASTC6x5"), 0x31: (6, 6, 16, "ASTC6x6"), 0x32: (8, 5, 16, "ASTC8x5"),
    0x33: (8, 6, 16, "ASTC8x6"), 0x34: (8, 8, 16, "ASTC8x8"), 0x35: (10, 5, 16, "ASTC10x5"),
    0x36: (10, 6, 16, "ASTC10x6"), 0x37: (10, 8, 16, "ASTC10x8"), 0x38: (10, 10, 16, "ASTC10x10"),
    0x39: (12, 10, 16, "ASTC12x10"), 0x3A: (12, 12, 16, "ASTC12x12"),
}


class Texture:
    """One BRTI of a BNTX file, with the offsets needed to rewrite it."""

    def __init__(self, data: bytes, brti: int):
        self.brti = brti
        (self.flags, self.dim, self.tile, self.swizzle, self.mips, _, self.format, _, self.width,
         self.height, self.depth, self.layers, self.block_height_log2, _) = struct.unpack_from(
            "<BBHHHIIIiiiiii", data, brti + 0x10)
        self.image_size, = struct.unpack_from("<I", data, brti + 0x50)
        self.mip_ptr, = struct.unpack_from("<Q", data, brti + 0x70)
        self.mip_offsets = list(struct.unpack_from(f"<{self.mips}Q", data, self.mip_ptr))
        name_ptr, = struct.unpack_from("<Q", data, brti + 0x60)
        length, = struct.unpack_from("<H", data, name_ptr)
        self.name = data[name_ptr + 2:name_ptr + 2 + length].decode("utf-8", "replace")

    @property
    def format_name(self) -> str:
        info = FORMATS.get(self.format >> 8)
        return info[3] if info else f"0x{self.format >> 8:02X}"

    @property
    def top_mip_bytes(self) -> int:
        if self.mips < 2:
            return self.image_size
        return self.mip_offsets[1] - self.mip_offsets[0]

    def reducible(self) -> str | None:
        if self.mips < 2:
            return "one mip"
        if self.dim != 2 or self.depth != 1 or self.layers != 1:
            return "not a single 2D texture"
        if self.tile != 0 or (self.format >> 8) not in FORMATS:
            return "unknown layout"
        return None


def parse_bntx(data: bytes) -> list[Texture]:
    if data[:4] != b"BNTX":
        raise ValueError("not a BNTX file")
    count, info_array = struct.unpack_from("<IQ", data, 0x24)
    return [Texture(data, struct.unpack_from("<Q", data, info_array + 8 * i)[0])
            for i in range(count)]


def block_height_for(height_in_gobs: int, block_height: int) -> int:
    # The Tegra block-linear rule each mip level follows: halve while the level fits in half.
    while block_height > 1 and height_in_gobs <= block_height // 2:
        block_height //= 2
    return block_height


def drop_top_mip(data: bytes) -> bytes | None:
    """The same BNTX with mip level 0 removed from its only texture, or None if not possible."""
    textures = parse_bntx(data)
    if len(textures) != 1 or textures[0].reducible():
        return None
    tex = textures[0]
    bw, bh, _, _ = FORMATS[tex.format >> 8]
    removed = tex.mip_offsets[1] - tex.mip_offsets[0]
    start = tex.mip_offsets[0]
    width = max(1, tex.width // 2)
    height = max(1, tex.height // 2)
    rows = (height + bh - 1) // bh
    new_block_height = block_height_for((rows + 7) // 8, 1 << tex.block_height_log2)

    out = bytearray(data[:start] + data[start + removed:])
    struct.pack_into("<H", out, tex.brti + 0x16, tex.mips - 1)
    struct.pack_into("<ii", out, tex.brti + 0x24, width, height)
    struct.pack_into("<i", out, tex.brti + 0x34, new_block_height.bit_length() - 1)
    struct.pack_into("<I", out, tex.brti + 0x50, tex.image_size - removed)
    new_offsets = [start] + [o - removed for o in tex.mip_offsets[2:]]
    struct.pack_into(f"<{len(new_offsets)}Q", out, tex.mip_ptr, *new_offsets)
    struct.pack_into("<Q", out, tex.mip_ptr + 8 * len(new_offsets), 0)

    brtd, = struct.unpack_from("<Q", data, 0x30)
    brtd_size, = struct.unpack_from("<Q", data, brtd + 8)
    if not brtd < start < brtd + brtd_size:
        return None
    struct.pack_into("<Q", out, brtd + 8, brtd_size - removed)
    rlt, file_size = struct.unpack_from("<II", data, 0x18)
    if rlt < start:
        return None
    rlt -= removed
    struct.pack_into("<II", out, 0x18, rlt, file_size - removed)
    struct.pack_into("<I", out, rlt + 4, rlt)
    sections, = struct.unpack_from("<I", out, rlt + 8)
    entries = 0
    for i in range(sections):
        pos = rlt + 16 + 0x18 * i
        offset, size, _, count = struct.unpack_from("<IIII", out, pos + 8)
        if offset <= start < offset + size:
            struct.pack_into("<I", out, pos + 12, size - removed)
        entries += count
    for i in range(entries):
        pos = rlt + 16 + 0x18 * sections + 8 * i
        offset, structs, count, _ = struct.unpack_from("<IHBB", out, pos)
        if offset == tex.mip_ptr and structs == 1 and count == tex.mips:
            struct.pack_into("<B", out, pos + 6, count - 1)
    return bytes(out)


def read_list(path: str) -> set[int]:
    rx = re.compile(r"NUP first read: #(\d+) ")
    with open(path, encoding="utf-8", errors="replace") as f:
        return {int(m.group(1)) for line in f if (m := rx.search(line))}


def textures(nup, package: int, only: set[int] | None):
    for path, index in nup.package_files(package):
        if not (path.startswith("Tex/") and path.endswith(".bntx.zs")):
            continue
        if only is not None and index not in only:
            continue
        yield path, index


def size_class(tex: Texture) -> str:
    edge = max(tex.width, tex.height)
    for limit in (128, 256, 512, 1024, 2048):
        if edge <= limit:
            return f"<={limit}"
    return ">2048"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["survey", "make"])
    ap.add_argument("nup")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--list", help="Lichii log with EDEN_NX2_NUP_READ_LOG=1 lines")
    ap.add_argument("--package", type=int)
    ap.add_argument("--csv")
    ap.add_argument("--min-size", type=int, default=512,
                    help="only textures whose larger edge is at least this (make)")
    ap.add_argument("--levels", type=int, default=1, help="mip levels to drop (make)")
    ap.add_argument("--level", type=int, default=3, help="zstd level of the written files")
    args = ap.parse_args()
    if zstandard is None:
        sys.exit("needs the zstandard package: pip install zstandard")

    nup = nup_tool.Nup(args.nup)
    package = nup_tool.pick_package(nup, args.package)
    only = read_list(args.list) if args.list else None
    dctx = zstandard.ZstdDecompressor()

    if args.command == "survey":
        by_format: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        by_size: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        csv = open(args.csv, "w", encoding="utf-8") if args.csv else None
        if csv:
            csv.write("path,format,width,height,depth,layers,mips,file_bytes,image_bytes,top_mip_bytes,reducible\n")
        total_file = 0
        for path, index in textures(nup, package, only):
            data = dctx.decompress(nup.read_file(index), max_output_size=1 << 31)
            total_file += len(data)
            for tex in parse_bntx(data):
                why = tex.reducible()
                for table, key in ((by_format, tex.format_name), (by_size, size_class(tex))):
                    row = table[key]
                    row[0] += 1
                    row[1] += tex.image_size
                    row[2] += tex.top_mip_bytes if why is None else 0
                if csv:
                    csv.write(f"{path},{tex.format_name},{tex.width},{tex.height},{tex.depth},"
                              f"{tex.layers},{tex.mips},{len(data)},{tex.image_size},"
                              f"{tex.top_mip_bytes},{why or 'yes'}\n")
        mib = lambda v: v / 2**20
        print(f"BNTX files: {mib(total_file):.1f} MiB decompressed")
        for title, table in (("format", by_format), ("larger edge", by_size)):
            print(f"\n{title:12} {'count':>7} {'image MiB':>10} {'top mip MiB (reducible)':>24}")
            for key, (n, image, top) in sorted(table.items(), key=lambda kv: -kv[1][1]):
                print(f"{key:12} {n:7} {mib(image):10.1f} {mib(top):24.1f}")
        return 0

    if not args.out:
        ap.error("make needs the mod directory")
    cctx = zstandard.ZstdCompressor(level=args.level)
    written = skipped = 0
    saved = 0
    for path, index in textures(nup, package, only):
        data = dctx.decompress(nup.read_file(index), max_output_size=1 << 31)
        tex = parse_bntx(data)
        if len(tex) != 1 or max(tex[0].width, tex[0].height) < args.min_size:
            continue
        reduced = data
        for _ in range(args.levels):
            smaller = drop_top_mip(reduced)
            if smaller is None:
                break
            reduced = smaller
        if reduced is data:
            skipped += 1
            continue
        dest = os.path.join(args.out, "romfs", *path.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(cctx.compress(reduced))
        saved += len(data) - len(reduced)
        written += 1
    print(f"{written} textures reduced ({saved / 2**20:.1f} MiB less decompressed data), "
          f"{skipped} skipped (one mip, arrays, 3D or unknown layout)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
