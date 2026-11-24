#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""AI-driven iOS IPA test & performance pipeline for Lichii on a paired iPhone.

Wraps tools/nx2/ios_bench/bench_mac.py, bench_report.py, xcrun devicectl, gh, and
tools/nx2/debug_mcp.py so an agent can test IPAs, pull logs, stage saves/profiles,
run single benchmarks or experiment matrices, and inspect a live iPhone session over Wi-Fi.

Examples:
    python3 -I tools/nx2/ios_bench/ai_pipeline.py status
    python3 -I tools/nx2/ios_bench/ai_pipeline.py pull-logs --out /tmp/ios-logs
    python3 -I tools/nx2/ios_bench/ai_pipeline.py fetch-ipa --out ~/Desktop/lichii-latest-ios-arm64.ipa [--install]
    python3 -I tools/nx2/ios_bench/ai_pipeline.py stage [--hyrule-save ~/Desktop/ocarina-hyrule-field-save.zip]
    python3 -I tools/nx2/ios_bench/ai_pipeline.py bench --profile hyrule-field --out /tmp/ios-runs/hf
    python3 -I tools/nx2/ios_bench/ai_pipeline.py matrix --plan plans/mobile-experiments.json --out /tmp/ios-runs/exp
    python3 -I tools/nx2/ios_bench/ai_pipeline.py live --host 192.168.1.42:6545 status
    python3 -I tools/nx2/ios_bench/ai_pipeline.py live --host 192.168.1.42:6545 screenshot --out shot.png
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BENCH_MAC = os.path.join(HERE, "bench_mac.py")
BENCH_REPORT = os.path.join(HERE, "bench_report.py")
DEBUG_MCP = os.path.join(REPO_ROOT, "tools", "nx2", "debug_mcp.py")
CONFIG_PATH = os.path.expanduser("~/lichii-bench/config.json")


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_cmd(cmd, check=True, capture=False):
    if capture:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=check)
    return subprocess.run(cmd, check=check)


def bench_mac_base_args(args):
    cfg = load_config()
    cmd = [sys.executable, "-I", BENCH_MAC]
    device = args.device or cfg.get("device")
    bundle_id = args.bundle_id or cfg.get("bundle_id")
    if device:
        cmd += ["--device", device]
    if bundle_id:
        cmd += ["--bundle-id", bundle_id]
    return cmd, device, bundle_id


def cmd_status(args):
    cmd, device, bundle_id = bench_mac_base_args(args)
    print("=== Paired Devices ===")
    run_cmd([sys.executable, "-I", BENCH_MAC, "devices"], check=False)
    print("\n=== Bench Doctor ===")
    run_cmd(cmd + ["doctor"], check=False)
    print("\n=== Code Signing Identities ===")
    run_cmd(["security", "find-identity", "-v", "-p", "codesigning"], check=False)
    if args.repo:
        print(f"\n=== Latest CI Runs ({args.repo}) ===")
        run_cmd(["gh", "run", "list", "-R", args.repo, "--workflow", "release-binaries",
                 "--limit", "3"], check=False)


def cmd_pull_logs(args):
    _, device, bundle_id = bench_mac_base_args(args)
    if not device or not bundle_id:
        raise SystemExit("Configure ~/lichii-bench/config.json or pass --device and --bundle-id.")
    os.makedirs(args.out, exist_ok=True)
    files = [
        ("Documents/eden/log/eden_log.txt", "eden_log.txt"),
        ("Documents/eden/log/eden_log.txt.old.txt", "eden_log.txt.old.txt"),
        ("Documents/eden/log/lichii-startup.txt", "lichii-startup.txt"),
        ("Documents/eden/log/lichii-startup.old.txt", "lichii-startup.old.txt"),
        ("Documents/bench/last.json", "bench_last.json"),
        ("Documents/bench/status.json", "bench_status.json"),
    ]
    pulled = []
    for src, dst_name in files:
        dst = os.path.join(args.out, dst_name)
        proc = run_cmd([
            "xcrun", "devicectl", "device", "copy", "from",
            "--device", device,
            "--domain-type", "appDataContainer",
            "--domain-identifier", bundle_id,
            "--source", src,
            "--destination", dst,
        ], check=False, capture=True)
        if proc.returncode == 0 and os.path.exists(dst):
            pulled.append((dst_name, os.path.getsize(dst)))
    print(json.dumps({"out": args.out, "pulled": pulled}, indent=2))


def cmd_fetch_ipa(args):
    repo = args.repo
    run_id = args.run_id
    if not run_id:
        proc = run_cmd([
            "gh", "run", "list", "-R", repo, "--workflow", "release-binaries",
            "--limit", "10", "--json", "databaseId,status,conclusion,headSha",
        ], capture=True)
        runs = json.loads(proc.stdout)
        chosen = next((r for r in runs if r.get("status") == "completed"), None)
        if not chosen:
            raise SystemExit("No completed release-binaries run found.")
        run_id = str(chosen["databaseId"])
        print(f"Selected CI run {run_id} (sha {chosen['headSha'][:10]})")

    with tempfile.TemporaryDirectory() as tmp:
        run_cmd(["gh", "run", "download", run_id, "-R", repo, "-p", "lichii-ios-arm64-*", "-D", tmp])
        ipa = None
        for root, _, filenames in os.walk(tmp):
            for name in filenames:
                if name.endswith(".ipa"):
                    ipa = os.path.join(root, name)
                    break
        if not ipa:
            raise SystemExit(f"No .ipa artifact found in run {run_id}")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        shutil.copy2(ipa, args.out)
        print(f"Downloaded IPA -> {args.out} ({os.path.getsize(args.out) >> 20} MiB)")

    if args.install:
        cmd, _, _ = bench_mac_base_args(args)
        with tempfile.TemporaryDirectory() as sign_dir:
            run_cmd(cmd + ["sign", "--ipa", args.out, "--out", sign_dir])
            app = os.path.join(sign_dir, "Lichii.app")
            run_cmd(cmd + ["install", "--app", app])


def cmd_stage(args):
    cmd, _, _ = bench_mac_base_args(args)
    stage_src = os.path.join(HERE, "stage")
    with tempfile.TemporaryDirectory() as tmp:
        stage = os.path.join(tmp, "stage")
        shutil.copytree(stage_src, stage)
        if args.hyrule_save and os.path.exists(args.hyrule_save):
            save_dst = os.path.join(stage, "saves", "hyrule-field")
            os.makedirs(save_dst, exist_ok=True)
            with zipfile.ZipFile(args.hyrule_save) as zf:
                zf.extractall(tmp)
            extracted_save = os.path.join(tmp, "save")
            src_dir = extracted_save if os.path.isdir(extracted_save) else tmp
            for entry in os.listdir(src_dir):
                if entry in ("account.sav", "option.sav") or entry.startswith("slot_"):
                    s = os.path.join(src_dir, entry)
                    d = os.path.join(save_dst, entry)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
            print(f"Included hyrule-field bench save from {args.hyrule_save}")
        run_cmd(cmd + ["push", "--stage", stage])


def cmd_bench(args):
    cmd, _, _ = bench_mac_base_args(args)
    profile_path = os.path.join(HERE, "stage", "profiles", f"{args.profile}.json")
    if not os.path.exists(profile_path):
        profile_path = args.profile
    run_args = cmd + ["run", "--request", profile_path, "--out", args.out, "--jit", args.jit]
    if args.trace:
        run_args.append("--trace")
    run_cmd(run_args)
    if os.path.exists(BENCH_REPORT):
        run_cmd([sys.executable, "-I", BENCH_REPORT, args.out], check=False)


def cmd_matrix(args):
    cmd, _, _ = bench_mac_base_args(args)
    plan_path = args.plan
    if not os.path.exists(plan_path):
        plan_path = os.path.join(HERE, args.plan)
    run_cmd(cmd + ["matrix", "--plan", plan_path, "--out", args.out])
    if os.path.exists(BENCH_REPORT):
        run_cmd([sys.executable, "-I", BENCH_REPORT, args.out], check=False)


def cmd_live(args):
    payload = {"method": args.method}
    if args.params:
        payload["params"] = json.loads(args.params)
    call_cmd = [sys.executable, "-I", DEBUG_MCP, "--server", args.host, "--call", json.dumps(payload)]
    if args.out:
        call_cmd += ["--out", args.out]
    run_cmd(call_cmd)


def main():
    p = argparse.ArgumentParser(description="AI iPhone test & performance pipeline for Lichii")
    p.add_argument("--device", help="iPhone UDID/name (default: ~/lichii-bench/config.json)")
    p.add_argument("--bundle-id", help="Installed Lichii bundle ID")
    p.add_argument("--repo", default="QuincyForbes/Lichii", help="GitHub repo for CI IPAs")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Check iPhone pairing, bench doctor, signing, and latest CI")

    pl = sub.add_parser("pull-logs", help="Pull Lichii logs and bench results from the iPhone")
    pl.add_argument("--out", required=True)

    fi = sub.add_parser("fetch-ipa", help="Download newest ios-arm64 IPA from CI (and optionally sign+install)")
    fi.add_argument("--run-id", help="Specific GitHub Actions run ID")
    fi.add_argument("--out", default=os.path.expanduser("~/Desktop/lichii-latest-ios-arm64.ipa"))
    fi.add_argument("--install", action="store_true", help="Sign and install on the paired iPhone")

    st = sub.add_parser("stage", help="Push bench profiles, scripts, and optional Hyrule Field save to iPhone")
    st.add_argument("--hyrule-save", default=os.path.expanduser("~/Desktop/ocarina-hyrule-field-save.zip"))

    bn = sub.add_parser("bench", help="Run one automated benchmark profile on the iPhone")
    bn.add_argument("--profile", default="hyrule-field")
    bn.add_argument("--out", required=True)
    bn.add_argument("--jit", default="stikdebug", choices=("stikdebug", "lldb"))
    bn.add_argument("--trace", action="store_true")

    mx = sub.add_parser("matrix", help="Run an A/B experiment matrix on the iPhone and print verdicts")
    mx.add_argument("--plan", default="plans/mobile-experiments.json")
    mx.add_argument("--out", required=True)

    lv = sub.add_parser("live", help="Talk to Lichii's live Wi-Fi debug server on the iPhone")
    lv.add_argument("--host", required=True, help="iPhone Wi-Fi endpoint, e.g. 192.168.1.42:6545")
    lv.add_argument("method", help="Debug server method (status, screenshot, log_tail, input_sequence, ...)")
    lv.add_argument("--params", help="JSON params object")
    lv.add_argument("--out", help="Output PNG path for screenshot")

    args = p.parse_args()
    handlers = {
        "status": cmd_status,
        "pull-logs": cmd_pull_logs,
        "fetch-ipa": cmd_fetch_ipa,
        "stage": cmd_stage,
        "bench": cmd_bench,
        "matrix": cmd_matrix,
        "live": cmd_live,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
