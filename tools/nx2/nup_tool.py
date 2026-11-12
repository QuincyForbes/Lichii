#!/usr/bin/env python3
"""List and extract the members of a NUP container (src/core/file_sys/nup_container.cpp).

  nup_tool.py list <file.nup> [--package N] [--match REGEX] [--summary]
  nup_tool.py extract <file.nup> <out dir> --match REGEX [--package N]

Paths are relative to the package's RomFS (or `exefs/...`). Extraction is for the owner's own
files; never commit what it writes.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
from collections import defaultdict


def zstd_decompress(data: bytes, size: int) -> bytes:
    try:
        import zstandard  # type: ignore

        return zstandard.ZstdDecompressor().decompress(data, max_output_size=size)
    except ImportError:
        pass
    try:
        from compression import zstd  # type: ignore  # Python 3.14+

        return zstd.decompress(data)
    except ImportError:
        sys.exit("zstd support needed: Python 3.14+, or `pip install zstandard`")


class Nup:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        sb = self.f.read(0x80)
        if sb[:4] != b"NUP\x01":
            raise ValueError("not a NUP container")
        toc_offset, toc_size = struct.unpack_from("<QQ", sb, 8)
        toc = self._read(toc_offset, toc_size)
        magic, package_count, dir_count, file_count, pool_offset, pool_size = struct.unpack_from(
            "<IIQQQQ", toc, 0)
        if magic != 0x01434F54:
            raise ValueError("bad TOC magic")
        pos = 0x30
        self.packages = []
        for _ in range(package_count):
            title_id, root, exefs, romfs, version, kind = struct.unpack_from("<QIIIIB", toc, pos)
            self.packages.append(dict(title_id=title_id, root=root, exefs=exefs, romfs=romfs,
                                      version=version, kind=kind))
            pos += 0x40
        pool = toc[pool_offset:pool_offset + pool_size]
        self.dirs = []
        for _ in range(dir_count):
            no, nl, first, count = struct.unpack_from("<IIII", toc, pos)
            self.dirs.append((pool[no:no + nl].decode("utf-8", "replace"), first, count))
            pos += 0x18
        self.files = []
        for _ in range(file_count):
            no, nl, size, off, ext = struct.unpack_from("<IIQQQ", toc, pos)
            codec = toc[pos + 0x40]
            self.files.append((pool[no:no + nl].decode("utf-8", "replace"), size, off, ext, codec))
            pos += 0x50
        children_bytes = pool_offset - pos
        self.children = struct.unpack_from(f"<{children_bytes // 4}I", toc, pos)
        self.dir_count = dir_count

    def _read(self, offset: int, size: int) -> bytes:
        self.f.seek(offset)
        return self.f.read(size)

    def walk(self, dir_index: int, prefix: str = ""):
        _, first, count = self.dirs[dir_index]
        for child in self.children[first:first + count]:
            if child < self.dir_count:
                yield from self.walk(child, f"{prefix}{self.dirs[child][0]}/")
            else:
                index = child - self.dir_count
                yield f"{prefix}{self.files[index][0]}", index

    def package_files(self, package: int):
        p = self.packages[package]
        if p["romfs"] != 0xFFFFFFFF:
            yield from self.walk(p["romfs"])
        if p["exefs"] != 0xFFFFFFFF:
            yield from self.walk(p["exefs"], "exefs/")

    def read_file(self, index: int) -> bytes:
        _, size, off, ext, codec = self.files[index]
        if size == 0:
            return b""
        data = self._read(off, ext)
        return zstd_decompress(data, size) if codec == 1 else data


def pick_package(nup: Nup, wanted: int | None) -> int:
    if wanted is not None:
        return wanted
    # Same choice as the loader: highest update with an ExeFS, else the base.
    updates = [i for i, p in enumerate(nup.packages) if p["kind"] == 1]
    if updates:
        return max(updates, key=lambda i: nup.packages[i]["version"])
    return 0


def ext_of(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".zs"):
        name = name[:-3]
    return name.rsplit(".", 1)[-1] if "." in name else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["list", "extract", "packages"])
    ap.add_argument("nup")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--package", type=int)
    ap.add_argument("--match", default="")
    ap.add_argument("--summary", action="store_true", help="bytes by extension")
    args = ap.parse_args()

    nup = Nup(args.nup)
    if args.command == "packages":
        for i, p in enumerate(nup.packages):
            print(i, f"{p['title_id']:016X}", "kind", p["kind"], "version", p["version"])
        return 0
    package = pick_package(nup, args.package)
    rx = re.compile(args.match) if args.match else None
    if args.command == "list":
        by_ext: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for path, index in nup.package_files(package):
            if rx and not rx.search(path):
                continue
            _, size, _, ext, codec = nup.files[index]
            if args.summary:
                row = by_ext[ext_of(path)]
                row[0] += 1
                row[1] += size
                row[2] += ext
            else:
                print(f"{index}\t{size}\t{ext}\t{codec}\t{path}")
        if args.summary:
            for e, (n, size, stored) in sorted(by_ext.items(), key=lambda kv: -kv[1][1]):
                print(f"{e or '-':12} {n:8} files {size / 2**20:10.1f} MiB  stored {stored / 2**20:10.1f} MiB")
        return 0
    if not args.out or not rx:
        ap.error("extract needs an output directory and --match")
    count = 0
    for path, index in nup.package_files(package):
        if not rx.search(path):
            continue
        dest = os.path.join(args.out, *path.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as out:
            out.write(nup.read_file(index))
        count += 1
    print(f"extracted {count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
