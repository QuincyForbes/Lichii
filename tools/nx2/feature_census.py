#!/usr/bin/env python3
"""Print what a game asked of the graphics backend, from nx2_feature_census.json.

The emulator writes the file to its log folder every 10 s with EDEN_NX2_FEATURE_CENSUS=1
(either renderer). Names of formats, topologies and stages come from the source tree.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
DRAW_FLAGS = ("indexed", "instanced", "indirect", "indirect-count", "byte-count", "mesh", "task",
              "tessellated", "geometry", "inline-index", "from-gpu-arguments")
COPIES = ("copy", "emulated copy", "reinterpret", "convert", "overlap copy", "overlap copy msaa",
          "blit", "blit framebuffer")
MISC = ("draw texture", "inline to memory", "conditional rendering", "transform feedback binding",
        "surface copy", "temporal upscale", "dma copy", "download memory", "fragment barrier",
        "tiled cache barrier", "counter reset", "layer from guest memory", "applet capture layer")
ATTRIBUTE_SIZES = {1: "32x4", 2: "32x3", 4: "32x2", 0x12: "32", 3: "16x4", 5: "16x3", 0xf: "16x2",
                   0x1b: "16", 0xa: "8x4", 0x13: "8x3", 0x18: "8x2", 0x1d: "8", 0x30: "10_10_10_2",
                   0x31: "11_11_10"}
ATTRIBUTE_TYPES = ("?0", "snorm", "unorm", "sint", "uint", "uscaled", "sscaled", "float")
SWIZZLES = ("0", "?1", "r", "g", "b", "a", "1i", "1f")


def enum_names(path, name):
    """Value -> name of one C++ enum (plain or 'Name = value' entries)."""
    text = (ROOT / path).read_text(encoding="utf-8", errors="replace")
    match = re.search(r"enum\s+class\s+" + name + r"\b[^{]*\{(.*?)\}", text, re.S)
    if not match:
        return {}
    names = {}
    value = 0
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", match.group(1), flags=re.S)
    for entry in body.split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, _, explicit = entry.partition("=")
        if explicit.strip():
            try:
                value = int(explicit.strip(), 0)
            except ValueError:
                continue
        names.setdefault(value, key.strip())
        value += 1
    return names


def pixel_formats():
    text = (ROOT / "src/video_core/surface.h").read_text(encoding="utf-8", errors="replace")
    return dict(enumerate(re.findall(r"PIXEL_FORMAT_ELEM\((\w+),\s*\d", text)))


class Names:
    def __init__(self):
        self.pixel = pixel_formats()
        self.target = enum_names("src/video_core/gpu.h", "RenderTargetFormat")
        self.depth = enum_names("src/video_core/gpu.h", "DepthFormat")
        self.topology = enum_names("src/video_core/engines/maxwell_3d.h", "PrimitiveTopology")
        self.index = enum_names("src/video_core/engines/maxwell_3d.h", "IndexFormat")
        self.stage = enum_names("src/shader_recompiler/stage.h", "Stage")
        self.image_type = enum_names("src/video_core/texture_cache/types.h", "ImageType")
        self.view_type = enum_names("src/video_core/texture_cache/types.h", "ImageViewType")

    @staticmethod
    def of(table, value):
        return table.get(value, f"{value:#x}")


class Category:
    def __init__(self, data):
        self.total = data.get("total", 0)
        self.dropped = data.get("dropped", 0)
        self.columns = data.get("columns", [])
        self.rows = data.get("rows", [])
        self.index = {name: at for at, name in enumerate(self.columns)}

    def column(self, name):
        """Counter of one column's values, weighted by how often each record was seen."""
        at = self.index.get(name)
        values = Counter()
        if at is not None:
            for row in self.rows:
                values[row[at]] += row[0]
        return values

    def group(self, prefix):
        return [name for name in self.columns if name == prefix or name.startswith(prefix + "[")]


def show(title, counter, name=str, limit=40):
    if not counter:
        return
    print(f"  {title}:")
    for value, count in counter.most_common(limit):
        print(f"    {count:>12}  {name(value)}")
    if len(counter) > limit:
        print(f"    ... {len(counter) - limit} more")


def flag_names(value, names):
    parts = [names[bit] for bit in range(len(names)) if value >> bit & 1]
    return "+".join(parts) if parts else "plain"


def attribute_name(value):
    if value >> 6 & 1:
        return "constant"
    size = ATTRIBUTE_SIZES.get(value >> 21 & 0x3f, f"size {value >> 21 & 0x3f:#x}")
    kind = ATTRIBUTE_TYPES[value >> 27 & 7]
    return f"{kind} {size}{' bgra' if value >> 31 else ''}"


def swizzle_name(value):
    return "".join(SWIZZLES[value >> shift & 7] for shift in (0, 8, 16, 24))


def draws(category, names):
    print(f"draws: {category.total} ({len(category.rows)} distinct states, "
          f"{category.dropped} dropped)")
    show("kind", category.column("flags"), lambda value: flag_names(value, DRAW_FLAGS))
    show("topology", category.column("topology"), lambda value: names.of(names.topology, value))
    show("index format", category.column("index_format"),
         lambda value: "none" if value == 0xffffffff else names.of(names.index, value))
    pipelines = Counter()
    hashes = [category.index[name] for name in category.columns
              if name.endswith(".hi") or name.endswith(".lo")]
    for row in category.rows:
        pipelines[tuple(row[at] for at in hashes)] += row[0]
    print(f"  program sets: {len(pipelines)}")
    for prefix, table in (("rt.format", names.target), ("zeta.format", names.depth)):
        merged = Counter()
        for name in category.group(prefix):
            merged.update(category.column(name))
        merged.pop(0, None)
        show(prefix, merged, lambda value, table=table: names.of(table, value))
    show("colour targets", category.column("color_targets"))
    show("samples mode", category.column("anti_alias_samples_mode"))
    attributes = Counter()
    for name in category.group("vertex_attrib_format"):
        attributes.update(category.column(name))
    show("vertex attribute", attributes, attribute_name)
    strides = Counter()
    for name in category.group("vertex_streams.stride"):
        for value, count in category.column(name).items():
            if value >> 12 & 1:
                strides[value & 0xfff] += count
    show("vertex stride (enabled streams)", strides)
    listed = ("rt.format", "zeta.format", "vertex_attrib_format", "vertex_streams.stride",
              "vertex_streams.frequency", "vertex_stream_instances", "blend_per_target",
              "blend", "color_mask", "scissor_test.enable")
    print("  other state (values seen, most drawn first):")
    for name in category.columns:
        if name in ("flags", "topology", "index_format", "color_targets",
                    "anti_alias_samples_mode") or name.endswith((".hi", ".lo")):
            continue
        if name.partition("[")[0] in listed:
            continue
        values = category.column(name)
        text = ", ".join(f"{value:#x} x{count}" for value, count in values.most_common(8))
        more = f", ... {len(values) - 8} more" if len(values) > 8 else ""
        print(f"    {name}: {text}{more}")
    for prefix in ("blend", "color_mask", "scissor_test.enable", "vertex_streams.frequency",
                   "vertex_stream_instances", "blend_per_target"):
        merged = defaultdict(Counter)
        for name in category.group(prefix):
            merged[name] = category.column(name)
        distinct = Counter()
        for row in category.rows:
            distinct[tuple(row[category.index[name]] for name in merged)] += row[0]
        print(f"    {prefix}[]: {len(distinct)} distinct sets")


def programs(category, names):
    print(f"programs: {len(category.rows)} translated")
    show("stage", category.column("stage"), lambda value: names.of(names.stage, value))
    print("  features (programs using each):")
    for name in category.columns:
        if name.endswith((".hi", ".lo")) or name in ("stage",):
            continue
        used = sum(1 for row in category.rows if row[category.index[name]] != 0)
        if used:
            print(f"    {used:>6}  {name}")


def simple(title, category, describe):
    print(f"{title}: {category.total} ({len(category.rows)} distinct, {category.dropped} dropped)")
    rows = sorted(category.rows, key=lambda row: -row[0])
    for row in rows[:60]:
        print(f"    {row[0]:>12}  {describe(dict(zip(category.columns[1:], row[1:])))}")
    if len(rows) > 60:
        print(f"    ... {len(rows) - 60} more")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("census", type=Path)
    arguments = parser.parse_args()
    data = json.loads(arguments.census.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        sys.exit(f"unsupported census version {data.get('version')}")
    names = Names()
    category = {name: Category(value) for name, value in data.items() if isinstance(value, dict)}
    pixel = lambda value: names.of(names.pixel, value)
    draws(category["draws"], names)
    simple("clears", category["clears"], lambda row: ", ".join(
        f"{key} {value:#x}" for key, value in row.items()))
    simple("dispatches", category["dispatches"], lambda row:
           f"{row['compute.hi']:08x}{row['compute.lo']:08x}"
           f"{' indirect' if row['indirect'] else ''}")
    simple("queries", category["queries"], lambda row:
           f"report {row['report']} query {row['query']:#x}"
           f"{'' if row['has_address'] else ' no address'}")
    programs(category["programs"], names)
    simple("images", category["images"], lambda row:
           f"{pixel(row['format'])} {names.of(names.image_type, row['type'])} "
           f"samples {row['samples']} levels {row['levels']} layers {row['layers']}"
           f"{' sparse' if row['flags'] & 1 else ''}")
    views = category["image_views"]
    print(f"image views: {views.total} ({len(views.rows)} distinct)")
    show("type", views.column("type"), lambda value: names.of(names.view_type, value))
    show("format", views.column("format"), pixel, 80)
    show("swizzle", views.column("swizzle"), swizzle_name)
    reinterpreted = Counter()
    for row in views.rows:
        record = dict(zip(views.columns, row))
        if record["format"] != record["image_format"]:
            reinterpreted[(record["image_format"], record["format"])] += row[0]
    show("view format differs from image", reinterpreted,
         lambda pair: f"{pixel(pair[0])} as {pixel(pair[1])}")
    samplers = category["samplers"]
    print(f"samplers: {samplers.total} ({len(samplers.rows)} distinct)")
    simple("copies", category["copies"], lambda row:
           f"{COPIES[row['kind']]}: {pixel(row['src_format'])} x{row['src_samples']} -> "
           f"{pixel(row['dst_format'])} x{row['dst_samples']} flags {row['flags']:#x}")
    simple("other calls", category["misc"], lambda row:
           f"{MISC[row['kind']]} {row['value']:#x}")


if __name__ == "__main__":
    main()
