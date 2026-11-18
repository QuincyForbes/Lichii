#!/usr/bin/env python3
"""Counts the rows of docs/nx2/metal-coverage.md and checks its file references.

    python -I tools/nx2/metal_coverage.py            totals + reference check
    python -I tools/nx2/metal_coverage.py --lines    also print the source line of every reference
    python -I tools/nx2/metal_coverage.py --open     list the rows that are missing
    python -I tools/nx2/metal_coverage.py --moved <commit>
                                                     move the line numbers for what the sources
                                                     changed since the commit the matrix was true for

Exit code 1 when a row is malformed or a reference names a file or line that does not exist.
"""

import argparse
import collections
import pathlib
import re
import subprocess
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
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


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
    smoke = "".join(path.read_text(encoding="utf-8", errors="replace")
                    for path in sorted(METAL.glob("metal_smoke*.mm")))
    units = "".join(path.read_text(encoding="utf-8", errors="replace")
                    for path in (ROOT / "src" / "tests").rglob("*.cpp"))
    probes = (METAL / "metal_smoke_probe.mm").read_text(encoding="utf-8", errors="replace")
    units = "".join(path.read_text(encoding="utf-8", errors="replace")
                    for path in (ROOT / "src" / "tests").rglob("*.cpp"))
    problems = []
    for row in rows:
        for item in (part.strip() for part in row["tested"].split(",")):
            if item in ("untested", "-", "replay"):
                continue
            if item.startswith("unit:"):
                if f'TEST_CASE("{item[5:]}"' not in units:
                    problems.append(f"{row['id']}: no unit test '{item[5:]}'")
            elif item.startswith("probe:"):
                if f'{{"{item[6:]}"' not in probes:
                    problems.append(f"{row['id']}: no probe '{item[6:]}'")
            elif not re.search(rf"\b{re.escape(item)}\(", smoke):
                problems.append(f"{row['id']}: no smoke case '{item}'")
    return problems


def line_map(commit, path):
    """Where the lines `path` had at `commit` are in the working tree."""
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", "-U0", commit, "--", str(path)],
                          capture_output=True, text=True, check=True).stdout
    hunks = []
    for match in HUNK.finditer(diff):
        old, old_count, new, new_count = (int(value) if value else 1 for value in match.groups())
        hunks.append((old, old_count, new, new_count))

    def moved(line):
        shift = 0
        for old, old_count, new, new_count in hunks:
            old_end = old + old_count if old_count else old + 1
            new_end = new + new_count if new_count else new + 1
            if line >= old_end:
                shift = new_end - old_end
            elif old_count and line >= old:
                # A changed line: the same place in what replaced it.
                return new + min(line - old, max(new_count - 1, 0))
        return line + shift

    return moved


def move_references(matrix, commit):
    """Rewrites the line numbers of the matrix for sources changed since `commit`."""
    maps, changed, lines = {}, 0, []
    for line in matrix.read_text(encoding="utf-8").splitlines():
        if ROW.match(line) and len(split_row(line)) == 7:
            cells = split_row(line)
            for index, base in ((2, VULKAN), (4, METAL)):
                items = []
                for item in (part.strip() for part in cells[index].split(",")):
                    match = REFERENCE.match(item)
                    path = (base / match.group(1)).resolve() if match else None
                    if match and match.group(2) and path.is_file():
                        if path not in maps:
                            maps[path] = line_map(commit, path)
                        number = maps[path](int(match.group(2)))
                        changed += number != int(match.group(2))
                        item = f"{match.group(1)}:{number}"
                    items.append(item)
                cells[index] = ", ".join(items)
            line = "| " + " | ".join(cells) + " |"
        lines.append(line)
    matrix.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{changed} references moved")


def count(rows):
    counts = collections.Counter(row["kind"] for row in rows)
    return "  ".join(f"{status} {counts[status]}" for status in STATUSES) + f"  (rows {len(rows)})"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=pathlib.Path, default=MATRIX)
    parser.add_argument("--lines", action="store_true", help="print the source line of every reference")
    parser.add_argument("--open", action="store_true", help="list the rows that are missing")
    parser.add_argument("--moved", metavar="COMMIT",
                        help="first move the line numbers for sources changed since COMMIT")
    args = parser.parse_args()

    if args.moved:
        move_references(args.matrix, args.moved)
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
