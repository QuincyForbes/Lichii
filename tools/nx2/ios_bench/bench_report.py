#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compares iOS bench results (Documents/bench/result-*.json from the app's bench mode).

    python3 -I bench_report.py <result.json or a directory with them> ... [--baseline baseline]
                               [--json summary.json] [--markdown summary.md]

Each result is one run; runs with the same "cell" (the toggle set, from the request) are repeats.
Prints a table per measured window (stand, pan) with the mean of the valid repeats, and judges
each cell against the baseline: faster or slower only when the difference is larger than both 3 %
and twice the combined standard error of the repeats.
"""

import argparse
import json
import math
import os
import re
import statistics
import sys

MIN_FRAMES = 100
MIN_CHANGE = 0.03

# "Perf threads: 41 threads, CPU 340%, 15% of it on performance cores, waiting for a CPU 65% ..."
THREADS_TOTAL = re.compile(
    r"Perf threads: (\d+) threads, CPU (\d+)%(?:, (\d+)% of it on performance cores)?"
    r"(?:, waiting for a CPU (\d+)%)?")
# "| CPUCore_0 (user-interactive/47): CPU 93%, 90% on performance cores, waiting 0%"
THREAD_ROW = re.compile(
    r"\| ([^|(]+?) \(([^)]*)\): CPU (\d+)%(?:, (\d+)% on performance cores)?(?:, waiting (\d+)%)?")
# "| guest core wakes 812/s, average 16 us, 99% within 197 us, longest 11600 us |"
WAKES = re.compile(
    r"guest core wakes (\d+)/s(?:, average (\d+) us, 99% within (\d+) us, longest (\d+) us)?")
# "Perf sched: game threads: 12 ran, on a core 54%, ready without one 19% ..."
GAME_THREADS = re.compile(r"game threads: (\d+) ran, on a core (\d+)%, ready without one (\d+)%")
# "Perf: 14.2 fps, frame 70.4 ms avg 133 max | GPU 28.1 ms/frame, 42% busy, ..."
PERF_LINE = re.compile(r"Perf: ([\d.]+) fps, frame ([\d.]+) ms avg (\d+) max \| GPU ([\d.]+) ms/frame")


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def parse_perf_lines(lines):
    """Averages of the figures in a window's Perf lines."""
    total_cpu, total_p, total_wait = [], [], []
    wakes_rate, wakes_avg, wakes_p99 = [], [], []
    ready = []
    perf_fps, perf_gpu = [], []
    cores = {}
    for line in lines:
        if "Perf threads:" in line:
            m = THREADS_TOTAL.search(line)
            if m:
                total_cpu.append(float(m.group(2)))
                if m.group(3):
                    total_p.append(float(m.group(3)))
                if m.group(4):
                    total_wait.append(float(m.group(4)))
            for row in THREAD_ROW.finditer(line):
                name = row.group(1).strip()
                if not name.startswith("CPUCore_"):
                    continue
                entry = cores.setdefault(name, {"qos": row.group(2), "cpu": [], "p": [], "wait": []})
                entry["cpu"].append(float(row.group(3)))
                if row.group(4):
                    entry["p"].append(float(row.group(4)))
                if row.group(5):
                    entry["wait"].append(float(row.group(5)))
        elif "Perf sched:" in line:
            m = WAKES.search(line)
            if m:
                wakes_rate.append(float(m.group(1)))
                if m.group(2):
                    wakes_avg.append(float(m.group(2)))
                    wakes_p99.append(float(m.group(3)))
            m = GAME_THREADS.search(line)
            if m:
                ready.append(float(m.group(3)))
        elif "Perf:" in line:
            m = PERF_LINE.search(line)
            if m:
                perf_fps.append(float(m.group(1)))
                perf_gpu.append(float(m.group(4)))
    return {
        "threads_cpu_pct": mean(total_cpu),
        "threads_p_core_pct": mean(total_p),
        "threads_waiting_pct": mean(total_wait),
        "guest_wakes_per_s": mean(wakes_rate),
        "guest_wake_avg_us": mean(wakes_avg),
        "guest_wake_p99_us": mean(wakes_p99),
        "game_threads_ready_pct": mean(ready),
        "perf_line_fps": mean(perf_fps),
        "perf_line_gpu_ms": mean(perf_gpu),
        "guest_cores": {
            name: {"qos": e["qos"], "cpu_pct": mean(e["cpu"]), "p_core_pct": mean(e["p"]),
                   "waiting_pct": mean(e["wait"])}
            for name, e in sorted(cores.items())
        },
    }


def load_results(paths):
    results = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for name in sorted(files):
                    if name.startswith("result") and name.endswith(".json"):
                        results.extend(load_results([os.path.join(root, name)]))
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "schema" not in data:
            continue
        data["_path"] = path
        results.append(data)
    # last.json repeats a result-<stamp>.json.
    seen = set()
    unique = []
    for r in results:
        key = (r.get("stamp"), r.get("run"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def cell_of(result):
    request = result.get("request") or {}
    if request.get("cell"):
        return request["cell"]
    run = result.get("run") or "run"
    return re.sub(r"[#-]\d+$", "", run)


def validity(result, window, max_thermal):
    """Why a window cannot be compared, or None."""
    if result.get("status") != "ok":
        return f"status {result.get('status')}"
    start = (result.get("thermal_measure_start") or {}).get("state")
    if start is not None and start > max_thermal:
        return f"thermal {start} at the start"
    if window.get("paused_or_inactive"):
        return "paused or in the background"
    if not window.get("completed", True):
        return "window did not complete"
    if (window.get("frames") or 0) < MIN_FRAMES:
        return f"{window.get('frames')} frames"
    return None


def window_figures(result, window):
    cpu = window.get("cpu") or {}
    perf = parse_perf_lines(window.get("perf_lines") or [])
    thermal_start = (window.get("thermal_start") or {}).get("state")
    thermal_end = (window.get("thermal_end") or {}).get("state")
    return {
        "fps": window.get("fps_avg"),
        "low1": window.get("fps_1pct_low"),
        "p99": (window.get("frametime_ms") or {}).get("p99"),
        "gpu": window.get("gpu_ms_per_frame"),
        "cpu": cpu.get("process_pct"),
        "pshare": cpu.get("performance_core_share_pct"),
        "watts": cpu.get("watts"),
        "runnable": cpu.get("runnable_not_running_pct"),
        "jit": window.get("jit_compile_pct"),
        "mem": window.get("footprint_peak_mib"),
        "thermal": f"{thermal_start}->{thermal_end}",
        "wait": perf["threads_waiting_pct"],
        "wake99": perf["guest_wake_p99_us"],
        "perf": perf,
    }


def summarize(results, baseline, max_thermal):
    cells = {}
    for r in results:
        cell = cell_of(r)
        entry = cells.setdefault(cell, {"runs": [], "windows": {}})
        entry["runs"].append({"run": r.get("run"), "status": r.get("status"),
                              "message": r.get("message"), "path": r.get("_path"),
                              "build": (r.get("build") or {}).get("sha"),
                              "experiments": r.get("experiments")})
        for window in r.get("windows") or []:
            name = window.get("name", "?")
            bucket = entry["windows"].setdefault(name, {"valid": [], "invalid": []})
            why = validity(r, window, max_thermal)
            figures = window_figures(r, window)
            figures["run"] = r.get("run")
            (bucket["invalid"] if why else bucket["valid"]).append(
                dict(figures, why=why) if why else figures)
    base = cells.get(baseline)
    for cell, entry in cells.items():
        for name, bucket in entry["windows"].items():
            fps = [f["fps"] for f in bucket["valid"] if f["fps"] is not None]
            bucket["mean"] = {key: mean([f[key] for f in bucket["valid"]])
                              for key in ("fps", "low1", "p99", "gpu", "cpu", "pshare", "watts",
                                          "runnable", "jit", "mem", "wait", "wake99")}
            bucket["fps_sd"] = statistics.stdev(fps) if len(fps) > 1 else None
            bucket["n"] = len(fps)
            bucket["verdict"] = None
            if base is None or cell == baseline or name not in base["windows"]:
                continue
            ref = base["windows"][name]
            ref_fps = [f["fps"] for f in ref["valid"] if f["fps"] is not None]
            bucket["verdict"] = judge(fps, ref_fps)
    return cells


def judge(fps, ref_fps):
    if not fps or not ref_fps:
        return {"verdict": "no valid runs", "change_pct": None}
    a, b = statistics.mean(fps), statistics.mean(ref_fps)
    change = (a - b) / b if b else 0.0
    se = math.sqrt(sum((statistics.variance(v) / len(v)) if len(v) > 1 else 0.0
                       for v in (fps, ref_fps)))
    threshold = max(MIN_CHANGE * b, 2.0 * se)
    if len(fps) < 2 or len(ref_fps) < 2:
        note = " (single runs: repeat to be sure)"
    else:
        note = ""
    if a - b > threshold:
        verdict = "faster" + note
    elif b - a > threshold:
        verdict = "slower" + note
    else:
        verdict = "no clear change" + note
    return {"verdict": verdict, "change_pct": round(100.0 * change, 1),
            "threshold_fps": round(threshold, 2)}


def fmt(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(cells, baseline):
    names = sorted({n for e in cells.values() for n in e["windows"]},
                   key=lambda n: (n != "stand", n))
    order = sorted(cells, key=lambda c: (c != baseline, c))
    lines = []
    for name in names:
        lines.append(f"## {name}")
        lines.append("")
        header = ["cell", "n", "fps", "sd", "1% low", "p99 ms", "GPU ms", "CPU %", "P-core %",
                  "W", "runnable %", "waiting %", "wake p99 us", "mem MiB", "vs baseline"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for cell in order:
            bucket = cells[cell]["windows"].get(name)
            if not bucket:
                continue
            m = bucket["mean"]
            verdict = bucket.get("verdict")
            vs = "baseline" if cell == baseline else (
                f"{fmt(verdict['change_pct'])} % {verdict['verdict']}" if verdict and
                verdict.get("change_pct") is not None else (verdict or {}).get("verdict", "-"))
            invalid = len(bucket["invalid"])
            lines.append("| " + " | ".join([
                cell + (f" ({invalid} invalid)" if invalid else ""), str(bucket["n"]),
                fmt(m["fps"]), fmt(bucket["fps_sd"], 2), fmt(m["low1"]), fmt(m["p99"]),
                fmt(m["gpu"]), fmt(m["cpu"], 0), fmt(m["pshare"], 0), fmt(m["watts"], 2),
                fmt(m["runnable"], 0), fmt(m["wait"], 0), fmt(m["wake99"], 0), fmt(m["mem"], 0),
                vs]) + " |")
        lines.append("")
    problems = []
    for cell in order:
        for run in cells[cell]["runs"]:
            if run["status"] != "ok":
                problems.append(f"- {cell} {run['run']}: {run['status']}: {run['message']}")
        for name, bucket in cells[cell]["windows"].items():
            for f in bucket["invalid"]:
                problems.append(f"- {cell} {f['run']} {name}: not compared ({f['why']})")
    if problems:
        lines.append("## Runs not compared")
        lines.append("")
        lines.extend(problems)
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--max-thermal", type=int, default=1)
    parser.add_argument("--json")
    parser.add_argument("--markdown")
    args = parser.parse_args(argv)
    results = load_results(args.paths)
    if not results:
        print("no bench results found", file=sys.stderr)
        return 1
    cells = summarize(results, args.baseline, args.max_thermal)
    text = table(cells, args.baseline)
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(cells, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
