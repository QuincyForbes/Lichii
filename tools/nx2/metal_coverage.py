#!/usr/bin/env python3
"""Counts the rows of docs/nx2/metal-coverage.md and checks its file references.

    python -I tools/nx2/metal_coverage.py            totals + reference check
    python -I tools/nx2/metal_coverage.py --lines    also print the source line of every reference
    python -I tools/nx2/metal_coverage.py --open     list the rows that are missing

Exit code 1 when a row is malformed or a reference names a file or line that does not exist.
"""

import argparse
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs" / "nx2" / "metal-coverage.md"
VULKAN = ROOT / "src" / "video_core" / "renderer_vulkan"
METAL = ROOT / "src" / "video_core" / "renderer_metal"

STATUSES = ("native", "emulated", "missing", "n/a")
USES = ("yes", "no", "?")
# Sections that are counted apart from the capability rows.
APART = ("Q", "R")

ROW = re.compile(r"^\|\s*([A-Z]\d{2})\s*\|")
REFERENCE = re.compile(r"^([\w./-]+?)(?::(\d+))?$")


def split_row(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def read_rows(path):
    rows, problems = [], []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not ROW.match(line):
            continue
        cells = split_row(line)
        if len(cells) != 7:
            problems.append(f"line {number}: {len(cells)} cells instead of 7")
            continue
        row = dict(zip(("id", "capability", "vulkan", "status", "metal", "tested", "oot"), cells))
        row["line"] = number
        row["kind"] = next((s for s in STATUSES if row["status"].startswith(s)), None)
        if row["kind"] is None:
            problems.append(f"{row['id']}: status '{row['status']}' is none of {STATUSES}")
            continue
        if row["oot"] not in USES:
            problems.append(f"{row['id']}: OoT '{row['oot']}' is none of {USES}")
            continue
        if row["id"] in seen:
            problems.append(f"{row['id']}: used twice")
        seen.add(row["id"])
        if row["kind"] in ("emulated", "missing", "n/a") and "(" not in row["status"]:
            problems.append(f"{row['id']}: '{row['kind']}' without a reason in brackets")
        rows.append(row)
    return rows, problems


def check_references(rows, show_lines):
    problems, shown = [], []
    lengths = {}
    for row in rows:
        for column, base in (("vulkan", VULKAN), ("metal", METAL)):
            cell = row[column]
            if cell == "-":
                continue
            for item in (part.strip() for part in cell.split(",")):
                match = REFERENCE.match(item)
                if not match:
                    problems.append(f"{row['id']} {column}: '{item}' is not file[:line]")
                    continue
                path = (base / match.group(1)).resolve()
                if not path.is_file():
                    problems.append(f"{row['id']} {column}: no file {match.group(1)}")
                    continue
                if match.group(2) is None:
                    continue
                if path not in lengths:
                    lengths[path] = path.read_text(encoding="utf-8", errors="replace").splitlines()
                line = int(match.group(2))
                if not 1 <= line <= len(lengths[path]):
                    problems.append(
                        f"{row['id']} {column}: {match.group(1)} has {len(lengths[path])} lines, not {line}")
                    continue
                if show_lines:
                    shown.append(f"{row['id']} {item}: {lengths[path][line - 1].strip()}")
    return problems, shown


def check_tests(rows):
    """Every smoke case and probe a row names exists in the smoke test's sources."""
    smoke = (METAL / "metal_smoke.mm").read_text(encoding="utf-8", errors="replace")
    probes = (METAL / "metal_smoke_probe.mm").read_text(encoding="utf-8", errors="replace")
    problems = []
    for row in rows:
        for item in (part.strip() for part in row["tested"].split(",")):
            if item in ("untested", "-", "replay"):
                continue
            if item.startswith("probe:"):
                if f'{{"{item[6:]}"' not in probes:
                    problems.append(f"{row['id']}: no probe '{item[6:]}'")
            elif not re.search(rf"\b{re.escape(item)}\(", smoke):
                problems.append(f"{row['id']}: no smoke case '{item}'")
    return problems


def count(rows):
    counts = collections.Counter(row["kind"] for row in rows)
    return "  ".join(f"{status} {counts[status]}" for status in STATUSES) + f"  (rows {len(rows)})"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=pathlib.Path, default=MATRIX)
    parser.add_argument("--lines", action="store_true", help="print the source line of every reference")
    parser.add_argument("--open", action="store_true", help="list the rows that are missing")
    args = parser.parse_args()

    rows, problems = read_rows(args.matrix)
    more, shown = check_references(rows, args.lines)
    problems += more + check_tests(rows)

    parity = [row for row in rows if row["id"][0] not in APART]
    print(f"capabilities (A-P): {count(parity)}")
    print(f"  the game uses:    {count([row for row in parity if row['oot'] == 'yes'])}")
    print(f"  not measured (?): {count([row for row in parity if row['oot'] == '?'])}")
    print(f"  the game does not use: {count([row for row in parity if row['oot'] == 'no'])}")
    for section in sorted({row["id"][0] for row in rows}):
        print(f"  {section}: {count([row for row in rows if row['id'][0] == section])}")
    untested = sum(1 for row in parity if row["tested"] == "untested")
    print(f"untested capability rows: {untested} of {len(parity)}")

    if args.open:
        for uses in USES:
            print(f"\nmissing, OoT {uses}:")
            for row in rows:
                if row["kind"] == "missing" and row["oot"] == uses and row["id"][0] not in APART:
                    print(f"  {row['id']} {row['capability']}")
    for line in shown:
        print(line)
    for problem in problems:
        print(f"PROBLEM {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
