# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import tempfile
import unittest

import bench_report

THREADS = ("[ 301.000000] Common.Memory <Info> common/perf_sched.cpp:Report:610: Perf threads: "
           "41 threads, CPU 340%, 15% of it on performance cores, waiting for a CPU 65% (200 "
           "samples) | CPUCore_0 (user-interactive/47): CPU 93%, 90% on performance cores, "
           "waiting 0% | GPU (user-interactive/47): CPU 60%, 20% on performance cores, waiting 9% "
           "| CPUCore_2 (user-initiated/37): CPU 40%, 30% on performance cores, waiting 12%")
SCHED = ("[ 301.000100] Common.Memory <Info> common/perf_sched.cpp:Report:613: Perf sched: game "
         "threads: 12 ran, on a core 54%, ready without one 19%, 800 switches/s | guest core "
         "wakes 812/s, average 16 us, 99% within 197 us, longest 11600 us | timer thread late x")
PERF = ("[ 300.000000] Common.Memory <Info> video_core/renderer_vulkan/vk_rasterizer.cpp:"
        "TickPerfReport:3975: Perf: 14.2 fps, frame 70.4 ms avg 133 max | GPU 28.1 ms/frame, 42% "
        "busy, 38.0 command buffers/frame | busy: ...")


def result(cell, repeat, fps, status="ok", thermal=0, frames=2000):
    window = {
        "name": "stand", "frames": frames, "fps_avg": fps, "fps_1pct_low": fps / 2,
        "frametime_ms": {"p50": 1000 / fps, "p99": 2000 / fps}, "gpu_ms_per_frame": 28.0,
        "cpu": {"process_pct": 340.0, "performance_core_share_pct": 15.0, "watts": 0.8,
                "runnable_not_running_pct": 800.0},
        "jit_compile_pct": 0.5, "footprint_peak_mib": 4900, "paused_or_inactive": False,
        "completed": True, "thermal_start": {"state": thermal}, "thermal_end": {"state": thermal},
        "perf_lines": [PERF, THREADS, SCHED],
    }
    return {"schema": 1, "run": f"{cell}#{repeat}", "stamp": f"{cell}{repeat}", "status": status,
            "message": "", "request": {"cell": cell, "repeat": repeat},
            "thermal_measure_start": {"state": thermal}, "build": {"sha": "abc"},
            "windows": [window]}


class PerfLines(unittest.TestCase):
    def test_figures(self):
        perf = bench_report.parse_perf_lines([PERF, THREADS, SCHED])
        self.assertEqual(perf["threads_cpu_pct"], 340)
        self.assertEqual(perf["threads_p_core_pct"], 15)
        self.assertEqual(perf["threads_waiting_pct"], 65)
        self.assertEqual(perf["guest_wake_p99_us"], 197)
        self.assertEqual(perf["game_threads_ready_pct"], 19)
        self.assertEqual(perf["perf_line_fps"], 14.2)
        self.assertEqual(perf["perf_line_gpu_ms"], 28.1)
        self.assertEqual(sorted(perf["guest_cores"]), ["CPUCore_0", "CPUCore_2"])
        self.assertEqual(perf["guest_cores"]["CPUCore_0"]["p_core_pct"], 90)
        self.assertEqual(perf["guest_cores"]["CPUCore_2"]["qos"], "user-initiated/37")


class Judge(unittest.TestCase):
    def summarize(self, results):
        with tempfile.TemporaryDirectory() as tmp:
            for r in results:
                with open(os.path.join(tmp, f"result-{r['stamp']}.json"), "w") as f:
                    json.dump(r, f)
            return bench_report.summarize(bench_report.load_results([tmp]), "baseline", 1)

    def test_faster_slower_noise(self):
        cells = self.summarize(
            [result("baseline", i, fps) for i, fps in enumerate([14.0, 14.2, 13.9])] +
            [result("guest_cores_first", i, fps) for i, fps in enumerate([17.5, 17.9, 17.2])] +
            [result("idle_spin", i, fps) for i, fps in enumerate([12.0, 12.3, 12.1])] +
            [result("clamp", i, fps) for i, fps in enumerate([14.1, 14.3, 13.8])])
        self.assertTrue(cells["guest_cores_first"]["windows"]["stand"]["verdict"]["verdict"]
                        .startswith("faster"))
        self.assertTrue(cells["idle_spin"]["windows"]["stand"]["verdict"]["verdict"]
                        .startswith("slower"))
        self.assertTrue(cells["clamp"]["windows"]["stand"]["verdict"]["verdict"]
                        .startswith("no clear change"))
        text = bench_report.table(cells, "baseline")
        self.assertIn("| baseline | 3 | 14.0", text)

    def test_invalid_runs_are_left_out(self):
        cells = self.summarize([result("baseline", 0, 14.0), result("baseline", 1, 30.0, thermal=2),
                                result("baseline", 2, 30.0, status="too_hot"),
                                result("baseline", 3, 30.0, frames=10)])
        stand = cells["baseline"]["windows"]["stand"]
        self.assertEqual(stand["n"], 1)
        self.assertEqual(len(stand["invalid"]), 3)
        text = bench_report.table(cells, "baseline")
        self.assertIn("not compared (thermal 2 at the start)", text)
        self.assertIn("too_hot", text)

    def test_cell_from_run_label(self):
        self.assertEqual(bench_report.cell_of({"run": "guest_cores_first#3"}), "guest_cores_first")
        self.assertEqual(bench_report.cell_of({"run": "baseline-2", "request": {}}), "baseline")


if __name__ == "__main__":
    unittest.main()
