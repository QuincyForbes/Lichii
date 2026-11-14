#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mac side of the iOS bench: signs and installs Lichii on the attached iPhone, runs bench
requests in the app's bench mode and brings the results back. Python 3.9 (the Command Line Tools'
python3), only xcrun/devicectl/codesign/security from Xcode.

    bench_mac.py devices
    bench_mac.py sign --ipa Lichii.ipa --out build/        # development signing for this device
    bench_mac.py install --app build/Lichii.app
    bench_mac.py push --stage stage/                       # Documents/bench/{scripts,saves,profiles}
    bench_mac.py probe [--via arg|env|file]                # does a launch reach bench mode?
    bench_mac.py run --request req.json --out runs/x/ [--jit stikdebug|lldb] [--trace]
    bench_mac.py matrix --plan plan.json --out runs/<session>/
    bench_mac.py pull --out dir/                           # the last result and its log
    bench_mac.py doctor                                    # what is still missing
    bench_mac.py make-profile                              # development profile via Xcode

Settings come from ~/lichii-bench/config.json (setup-mac.sh writes it): device, bundle_id,
team_id, identity, profile. Command line options override them.
"""

import argparse
import copy
import glob
import json
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.expanduser("~/lichii-bench/config.json")
BENCH = "Documents/bench"
LICHII_ENTITLEMENTS = (
    "com.apple.developer.kernel.increased-memory-limit",
    "com.apple.developer.kernel.increased-debugging-memory-limit",
    "com.apple.developer.kernel.extended-virtual-addressing",
)


def log(message):
    print(time.strftime("%H:%M:%S ") + message, flush=True)


def load_config(args):
    config = {}
    if os.path.exists(CONFIG):
        with open(CONFIG, encoding="utf-8") as f:
            config = json.load(f)
    for key in ("device", "bundle_id", "team_id", "identity", "profile"):
        value = getattr(args, key, None)
        if value:
            config[key] = value
    return config


def run(cmd, check=True, timeout=None, quiet=False):
    if not quiet:
        log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          timeout=timeout)
    if check and proc.returncode != 0:
        raise SystemExit(f"{cmd[0]} {cmd[1] if len(cmd) > 1 else ''} failed ({proc.returncode}):\n"
                         + proc.stdout[-4000:])
    return proc


def devicectl(args, check=True, timeout=300, quiet=False):
    """xcrun devicectl ... --json-output; returns (result dict, process)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    try:
        proc = run(["xcrun", "devicectl"] + args + ["--json-output", out], check=False,
                   timeout=timeout, quiet=quiet)
        data = {}
        if os.path.getsize(out) > 0:
            with open(out, encoding="utf-8") as f:
                data = json.load(f)
        if check and proc.returncode != 0:
            error = (data.get("error") or {}).get("userInfo") or data.get("error") or proc.stdout
            raise SystemExit(f"devicectl {' '.join(args[:3])} failed: {json.dumps(error)[:3000]}")
        return data.get("result") or {}, proc
    finally:
        os.unlink(out)


def list_devices():
    result, _ = devicectl(["list", "devices"], quiet=True)
    devices = []
    for d in result.get("devices", []):
        hw = d.get("hardwareProperties") or {}
        dp = d.get("deviceProperties") or {}
        cp = d.get("connectionProperties") or {}
        devices.append({
            "identifier": d.get("identifier"),
            "udid": hw.get("udid"),
            "name": dp.get("name"),
            "model": hw.get("productType"),
            "platform": hw.get("platform"),
            "os": dp.get("osVersionNumber"),
            "developer_mode": dp.get("developerModeStatus"),
            "ddi_mounted": dp.get("ddiServicesAvailable"),
            "paired": cp.get("pairingState"),
            "tunnel": cp.get("tunnelState"),
            "transport": cp.get("transportType"),
        })
    return devices


def pick_device(config):
    devices = list_devices()
    wanted = config.get("device")
    for d in devices:
        if wanted and wanted in (d["identifier"], d["udid"], d["name"]):
            return d
    if wanted:
        raise SystemExit(f"device {wanted} not found; devicectl sees: {json.dumps(devices)}")
    phones = [d for d in devices if d["platform"] == "iOS" and d["tunnel"] != "unavailable"]
    if not phones:
        raise SystemExit("no iPhone connected (xcrun devicectl list devices)")
    return phones[0]


# ------------------------------------------------------------------------------- signing

def decode_profile(path):
    proc = subprocess.run(["security", "cms", "-D", "-i", path], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        return None
    try:
        return plistlib.loads(proc.stdout)
    except Exception:
        return None


def profile_paths():
    paths = []
    for pattern in ("~/Library/MobileDevice/Provisioning Profiles/*.mobileprovision",
                    "~/Library/Developer/Xcode/UserData/Provisioning Profiles/*.mobileprovision"):
        paths.extend(glob.glob(os.path.expanduser(pattern)))
    return paths


def find_profile(config, udid):
    """The newest development profile for bundle_id that lists this device."""
    if config.get("profile"):
        profile = decode_profile(os.path.expanduser(config["profile"]))
        if profile is None:
            raise SystemExit(f"cannot read {config['profile']}")
        return os.path.expanduser(config["profile"]), profile
    bundle = config.get("bundle_id")
    best = None
    for path in profile_paths():
        profile = decode_profile(path)
        if not profile:
            continue
        ent = profile.get("Entitlements") or {}
        app_id = ent.get("application-identifier", "")
        team = (profile.get("TeamIdentifier") or [""])[0]
        if config.get("team_id") and team != config["team_id"]:
            continue
        if bundle and app_id not in (f"{team}.{bundle}", f"{team}.*"):
            continue
        if udid not in (profile.get("ProvisionedDevices") or []):
            continue
        if not ent.get("get-task-allow"):
            continue
        if profile.get("ExpirationDate") and profile["ExpirationDate"].timestamp() < time.time():
            continue
        score = (app_id.endswith(".*") is False,
                 sum(1 for k in LICHII_ENTITLEMENTS if ent.get(k)),
                 profile.get("CreationDate").timestamp() if profile.get("CreationDate") else 0)
        if best is None or score > best[0]:
            best = (score, path, profile)
    if best is None:
        raise SystemExit(
            "no development provisioning profile for bundle id "
            f"{bundle or '(any)'} that lists device {udid}. Run: bench_mac.py make-profile "
            "(or create one in Xcode, see README.md 'One-time steps').")
    return best[1], best[2]


def find_identity(config, profile):
    if config.get("identity"):
        return config["identity"]
    proc = run(["security", "find-identity", "-v", "-p", "codesigning"], quiet=True)
    import hashlib
    allowed = {hashlib.sha1(cert).hexdigest().upper()
               for cert in profile.get("DeveloperCertificates") or []}
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[1].upper() in allowed:
            return parts[1]
    raise SystemExit("no signing identity in the keychain matches the profile's certificates "
                     "(Xcode > Settings > Accounts > Manage Certificates > + Apple Development)")


def sign(config, ipa, out_dir, device):
    profile_path, profile = find_profile(config, device["udid"])
    identity = find_identity(config, profile)
    ent = dict(profile.get("Entitlements") or {})
    team = (profile.get("TeamIdentifier") or [""])[0]
    bundle = config.get("bundle_id") or ent["application-identifier"].split(".", 1)[1]
    if "*" in bundle:
        raise SystemExit("a wildcard profile needs bundle_id in the config")
    ent["application-identifier"] = f"{team}.{bundle}"
    missing = [k for k in LICHII_ENTITLEMENTS if not ent.get(k)]
    for key in missing:
        ent.pop(key, None)
    # Keychain groups and the like stay as the profile grants them; nothing beyond it.
    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="lichii-sign-")
    with zipfile.ZipFile(ipa) as z:
        z.extractall(work)
    apps = glob.glob(os.path.join(work, "Payload", "*.app"))
    if not apps:
        raise SystemExit(f"{ipa} has no Payload/*.app")
    app = os.path.join(out_dir, os.path.basename(apps[0]))
    if os.path.exists(app):
        shutil.rmtree(app)
    shutil.move(apps[0], app)
    shutil.rmtree(work, ignore_errors=True)
    # Unzip keeps no permissions bits for some archives.
    executable = plistlib.load(open(os.path.join(app, "Info.plist"), "rb"))["CFBundleExecutable"]
    os.chmod(os.path.join(app, executable), 0o755)
    info_path = os.path.join(app, "Info.plist")
    with open(info_path, "rb") as f:
        info = plistlib.load(f)
    info["CFBundleIdentifier"] = bundle
    for url_type in info.get("CFBundleURLTypes", []):
        url_type["CFBundleURLName"] = bundle
    with open(info_path, "wb") as f:
        plistlib.dump(info, f)
    shutil.copyfile(profile_path, os.path.join(app, "embedded.mobileprovision"))
    ent_path = os.path.join(out_dir, "bench.entitlements")
    with open(ent_path, "wb") as f:
        plistlib.dump(ent, f)
    for framework in sorted(glob.glob(os.path.join(app, "Frameworks", "*"))):
        run(["codesign", "--force", "--sign", identity, "--timestamp=none", framework])
    run(["codesign", "--force", "--sign", identity, "--timestamp=none", "--entitlements", ent_path,
         app])
    log(f"signed {app} as {bundle} (team {team}, profile {os.path.basename(profile_path)})")
    if missing:
        log("WARNING: the profile lacks " + ", ".join(missing) + ": enable Increased Memory "
            "Limit and Extended Virtual Addressing for the App ID, or iOS will close the game")
    return app, bundle


# ------------------------------------------------------------------------- device files

def copy_to(device, bundle, local, remote):
    devicectl(["device", "copy", "to", "--device", device["identifier"], "--domain-type",
               "appDataContainer", "--domain-identifier", bundle, "--source", local,
               "--destination", remote], quiet=True)


def copy_from(device, bundle, remote, local, check=True):
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
    _, proc = devicectl(["device", "copy", "from", "--device", device["identifier"],
                         "--domain-type", "appDataContainer", "--domain-identifier", bundle,
                         "--source", remote, "--destination", local], check=check, quiet=True,
                        timeout=900)
    return proc.returncode == 0 and os.path.exists(local)


def push_tree(device, bundle, stage):
    """Each file of `stage` to Documents/bench/<same path>."""
    count = 0
    for root, _, files in os.walk(stage):
        for name in files:
            if name.startswith("."):
                continue
            local = os.path.join(root, name)
            rel = os.path.relpath(local, stage).replace(os.sep, "/")
            copy_to(device, bundle, local, f"{BENCH}/{rel}")
            count += 1
    log(f"pushed {count} files from {stage} to {BENCH}")


def running_pids(device):
    result, _ = devicectl(["device", "info", "processes", "--device", device["identifier"]],
                          check=False, quiet=True)
    return {p.get("processIdentifier") for p in result.get("runningProcesses", [])}


def launch(device, bundle, trigger="arg", value="request"):
    args = ["device", "process", "launch", "--device", device["identifier"],
            "--terminate-existing"]
    if trigger == "env":
        args += ["--environment-variables", json.dumps({"LICHII_BENCH": value})]
    args.append(bundle)
    if trigger == "arg":
        args += ["--lichii-bench", value]
    result, _ = devicectl(args)
    pid = (result.get("process") or {}).get("processIdentifier")
    log(f"launched {bundle} ({trigger}), pid {pid}")
    return pid


def start_lldb_jit(device, pid, logfile):
    cmd = ["xcrun", "lldb", "--batch",
           "-o", "settings set platform.plugin.darwin.ignored-exceptions "
                 "EXC_BAD_ACCESS|EXC_BAD_INSTRUCTION|EXC_ARITHMETIC",
           "-o", f"device select {device['udid'] or device['identifier']}",
           "-o", f"device process attach --pid {pid}",
           "-o", f"command script import {os.path.join(HERE, 'jit_lldb.py')}",
           "-o", "lichii_jit_serve --timeout 900"]
    log("JIT through lldb: " + " ".join(cmd[2:]))
    return subprocess.Popen(cmd, stdout=open(logfile, "w"), stderr=subprocess.STDOUT)


def start_trace(device, pid, seconds, out):
    cmd = ["xcrun", "xctrace", "record", "--template", "Metal System Trace", "--device",
           device["udid"] or device["identifier"], "--attach", str(pid), "--time-limit",
           f"{int(seconds)}s", "--output", out, "--no-prompt"]
    log("trace: " + " ".join(cmd[2:]))
    return subprocess.Popen(cmd, stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT)


def read_remote_json(device, bundle, remote, scratch):
    local = os.path.join(scratch, os.path.basename(remote))
    if os.path.exists(local):
        os.unlink(local)
    if not copy_from(device, bundle, remote, local, check=False):
        return None
    try:
        with open(local, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def pull(device, bundle, out_dir):
    """Documents/bench/last.json (the newest result) as result.json, and its log as log.txt."""
    os.makedirs(out_dir, exist_ok=True)
    last = read_remote_json(device, bundle, f"{BENCH}/last.json", out_dir)
    if last is None:
        return None
    os.replace(os.path.join(out_dir, "last.json"), os.path.join(out_dir, "result.json"))
    if last.get("log"):
        copy_from(device, bundle, f"{BENCH}/{last['log']}", os.path.join(out_dir, "log.txt"),
                  check=False)
    return last


def run_request(config, device, request, out_dir, jit="stikdebug", trace=False, timeout=3600,
                trigger="arg"):
    bundle = config["bundle_id"]
    os.makedirs(out_dir, exist_ok=True)
    request = copy.deepcopy(request)
    if jit == "lldb":
        request.setdefault("jit", {})["provider"] = "external"
    request_path = os.path.join(out_dir, "request.json")
    with open(request_path, "w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)
    before = read_remote_json(device, bundle, f"{BENCH}/status.json", tempfile.mkdtemp())
    copy_to(device, bundle, request_path, f"{BENCH}/request.json")
    pid = launch(device, bundle, trigger=trigger, value="request")
    helpers = []
    if jit == "lldb" and pid:
        helpers.append(start_lldb_jit(device, pid, os.path.join(out_dir, "lldb-jit.log")))
    scratch = tempfile.mkdtemp(prefix="lichii-status-")
    start = time.monotonic()
    phase = None
    run_stamp = None
    traced = False
    gone_since = None
    while time.monotonic() - start < timeout:
        time.sleep(10)
        status = read_remote_json(device, bundle, f"{BENCH}/status.json", scratch)
        fresh = status and status.get("pid") == pid and status != before
        if fresh:
            run_stamp = status.get("stamp")
        if fresh and status.get("phase") != phase:
            phase = status.get("phase")
            log(f"phase {phase} at {status.get('t_s')} s")
            if trace and not traced and str(phase).startswith("stand"):
                traced = True
                stand = (request.get("measure") or {}).get("stand_s", 75)
                helpers.append(start_trace(device, pid, min(stand, 90),
                                           os.path.join(out_dir, "metal.trace")))
        if fresh and phase == "done":
            break
        if pid not in running_pids(device):
            gone_since = gone_since or time.monotonic()
            if time.monotonic() - gone_since > 20:
                log("the app is gone")
                break
    else:
        log(f"timed out after {timeout} s; stopping the app")
        devicectl(["device", "process", "terminate", "--device", device["identifier"], "--pid",
                   str(pid)], check=False)
    for helper in helpers:
        try:
            helper.wait(timeout=120)
        except subprocess.TimeoutExpired:
            helper.kill()
    result = pull(device, bundle, out_dir)
    if run_stamp is not None:
        ours = result is not None and result.get("stamp") == run_stamp
    else:
        ours = (result is not None and result.get("run") == request.get("run") and
                result.get("stamp") != (before or {}).get("stamp"))
    if not ours:
        crash = {"schema": 1, "status": "app_ended", "run": request.get("run"),
                 "message": f"no result for this run (last phase {phase}); the app crashed, was "
                            "closed by iOS or the run timed out", "request": request}
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(crash, f, indent=2)
        # The log of a run that ended early is still the app's current log.
        copy_from(device, bundle, "Documents/eden/log/eden_log.txt",
                  os.path.join(out_dir, "log.txt"), check=False)
        result = crash
    log(f"{request.get('run')}: {result.get('status')}: {result.get('message')}")
    return result


# ------------------------------------------------------------------------------ commands

def load_json(path_or_object):
    if isinstance(path_or_object, dict):
        return copy.deepcopy(path_or_object)
    with open(path_or_object, encoding="utf-8") as f:
        return json.load(f)


def merge(base, extra):
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def cmd_matrix(config, args):
    plan = load_json(args.plan)
    plan_dir = os.path.dirname(os.path.abspath(args.plan))
    template = plan["request"]
    if isinstance(template, str):
        template = load_json(os.path.join(plan_dir, template))
    cells = plan.get("cells") or [{"name": "baseline"}]
    repeats = int(plan.get("repeats", 3))
    cooldown = float(plan.get("cooldown_s", 240))
    hot_retries = int(plan.get("hot_retries", 3))
    device = pick_device(config)
    os.makedirs(args.out, exist_ok=True)
    order = [(rep, cell) for rep in range(1, repeats + 1) for cell in cells]
    summary = {"plan": plan, "device": device, "runs": []}
    for index, (rep, cell) in enumerate(order):
        request = merge(template, {k: v for k, v in cell.items() if k != "name"})
        request["run"] = f"{cell['name']}#{rep}"
        request["cell"] = cell["name"]
        request["repeat"] = rep
        out_dir = os.path.join(args.out, f"{cell['name']}-{rep}")
        for attempt in range(hot_retries + 1):
            result = run_request(config, device, request, out_dir, jit=plan.get("jit", "stikdebug"),
                                 trace=bool(plan.get("trace_first")) and index == 0,
                                 timeout=float(plan.get("timeout_s", 3600)))
            if result.get("status") != "too_hot" or attempt == hot_retries:
                break
            log(f"too hot; waiting {cooldown * 2:.0f} s before retrying")
            time.sleep(cooldown * 2)
        summary["runs"].append({"cell": cell["name"], "repeat": rep, "dir": out_dir,
                                "status": result.get("status")})
        with open(os.path.join(args.out, "matrix.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if index + 1 < len(order):
            log(f"cooling down {cooldown:.0f} s")
            time.sleep(cooldown)
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "bench_report.py"), args.out,
                    "--markdown", os.path.join(args.out, "summary.md"), "--json",
                    os.path.join(args.out, "summary.json")])
    return 0


def list_files(device, bundle, subdirectory):
    result, proc = devicectl(["device", "info", "files", "--device", device["identifier"],
                              "--domain-type", "appDataContainer", "--domain-identifier", bundle,
                              "--subdirectory", subdirectory, "--recurse"], check=False,
                             quiet=True, timeout=120)
    if proc.returncode != 0:
        return None
    return [f.get("relativePath") or f.get("name") or "" for f in result.get("files", [])]


def cmd_doctor(config, args):
    """What the bench still needs on this Mac and phone; exit 1 when something is missing."""
    missing = []

    def check(ok, what, fix):
        print(("  ok    " if ok else "  MISSING ") + what + ("" if ok else f"\n          -> {fix}"))
        if not ok:
            missing.append(what)
        return ok

    devices = list_devices()
    phones = [d for d in devices if d["platform"] == "iOS"]
    if not check(bool(phones), "an iPhone known to this Mac",
                 "connect the iPhone by USB, unlock it, tap Trust, then run setup-mac.sh again"):
        return 1
    try:
        device = pick_device(config)
    except SystemExit as error:
        check(False, "the configured device", str(error))
        return 1
    print(f"  device {device['name']} ({device['model']}, iOS {device['os']}, {device['udid']})")
    check(device["paired"] == "paired", "device paired", "unlock the phone and tap Trust")
    check(device["developer_mode"] == "enabled", "Developer Mode on",
          "iPhone Settings > Privacy & Security > Developer Mode > On (restarts the phone)")
    check(device["tunnel"] in ("connected", "disconnected"), "a CoreDevice connection",
          "keep the phone on USB (or the same network) and unlocked; tunnel is "
          f"{device['tunnel']}")
    apps_result, proc = devicectl(["device", "info", "apps", "--device", device["identifier"]],
                                  check=False, quiet=True)
    apps = {a.get("bundleIdentifier", ""): a.get("name") for a in apps_result.get("apps", [])}
    bundle = config.get("bundle_id")
    check(bool(bundle), "bundle_id in ~/lichii-bench/config.json", "run setup-mac.sh")
    installed = bool(bundle) and bundle in apps
    check(installed, f"Lichii installed as {bundle}",
          "bench_mac.py sign --ipa <ipa> --out build && bench_mac.py install --app build/Lichii.app")
    check(any("stik" in b.lower() for b in apps), "StikDebug on the phone (default JIT path)",
          "install StikDebug with its pairing file and LocalDevVPN, or use --jit lldb")
    try:
        profile_path, profile = find_profile(config, device["udid"])
        ent = profile.get("Entitlements") or {}
        check(True, f"provisioning profile {os.path.basename(profile_path)}", "")
        for key in LICHII_ENTITLEMENTS[0], LICHII_ENTITLEMENTS[2]:
            check(bool(ent.get(key)), f"profile grants {key}",
                  "developer.apple.com > Identifiers > the App ID > enable Increased Memory Limit "
                  "and Extended Virtual Addressing, then bench_mac.py make-profile")
        try:
            identity = find_identity(config, profile)
            check(True, f"signing identity {identity}", "")
        except SystemExit as error:
            check(False, "a signing identity for the profile", str(error))
    except SystemExit as error:
        check(False, "a development provisioning profile", str(error))
    if installed:
        files = list_files(device, bundle, "Documents")
        if files is None:
            print("  ?     cannot list the app's Documents (devicectl device info files)")
        else:
            keys = any(f.endswith("prod.keys") for f in files)
            games = [f for f in files if f.lower().endswith((".nup", ".nsp", ".xci", ".nsz", ".xcz"))]
            check(keys, "keys in Documents/eden/keys",
                  "copy prod.keys (and firmware) into the bench app's Documents/eden/keys")
            check(bool(games), "a game in Documents", "copy the game file into the app's Documents")
            saves = sorted({f.split("/")[2] for f in files
                            if f.startswith("bench/saves/") and f.count("/") >= 3})
            check(bool(saves), "a bench save in Documents/bench/saves",
                  "record one: bench_mac.py run with mode capture_save (README.md)")
            scripts = [f for f in files if f.startswith("bench/scripts/")]
            print(f"  info  games {games[:3]}, saves {saves}, scripts {len(scripts)}")
    print("  info  keep the phone unlocked with Auto-Lock set to Never (Settings > Display & "
          "Brightness > Auto-Lock): devicectl cannot launch apps on a locked phone")
    if missing:
        print(f"\n{len(missing)} missing: " + "; ".join(missing))
        return 1
    print("\neverything the bench needs is in place")
    return 0


def cmd_probe(config, args):
    device = pick_device(config)
    bundle = config["bundle_id"]
    scratch = tempfile.mkdtemp(prefix="lichii-probe-")
    vias = ["arg", "env", "file"] if args.via == "all" else [args.via]
    ok = True
    for via in vias:
        request = {"mode": "probe", "run": f"probe-{via}"}
        path = os.path.join(scratch, f"probe-{via}.json")
        with open(path, "w") as f:
            json.dump(request, f)
        if via == "file":
            copy_to(device, bundle, path, f"{BENCH}/request.json")
            pid = launch(device, bundle, trigger="none")
        else:
            copy_to(device, bundle, path, f"{BENCH}/profiles/probe-{via}.json")
            pid = launch(device, bundle, trigger=via, value=f"probe-{via}")
        result = None
        for _ in range(12):
            time.sleep(5)
            status = read_remote_json(device, bundle, f"{BENCH}/status.json", scratch)
            if status and status.get("pid") == pid and status.get("phase") == "done":
                result = read_remote_json(device, bundle, f"{BENCH}/last.json", scratch)
                break
        good = bool(result) and result.get("run") == f"probe-{via}" and result.get("status") == "ok"
        ok &= good
        log(f"probe via {via}: {'OK' if good else 'FAILED'}"
            + (f" (trigger {result.get('trigger')}, device {result.get('device')}, JIT "
               f"{(result.get('probe') or {})})" if result else " (no result: bench mode did not "
                                                              "start or the app did not exit)"))
    return 0 if ok else 1



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device")
    parser.add_argument("--bundle-id", dest="bundle_id")
    parser.add_argument("--team-id", dest="team_id")
    parser.add_argument("--identity")
    parser.add_argument("--profile")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("devices")
    p = sub.add_parser("sign")
    p.add_argument("--ipa", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("install")
    p.add_argument("--app", required=True)
    p = sub.add_parser("push")
    p.add_argument("--stage", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--via", choices=["arg", "env", "file", "all"], default="all")
    p = sub.add_parser("run")
    p.add_argument("--request", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jit", choices=["stikdebug", "lldb"], default="stikdebug")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--via", choices=["arg", "env", "file"], default="arg")
    p = sub.add_parser("matrix")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--out", required=True)
    sub.add_parser("make-profile")
    sub.add_parser("doctor")
    args = parser.parse_args(argv)
    config = load_config(args)

    if args.command == "devices":
        print(json.dumps(list_devices(), indent=2))
        return 0
    if args.command == "make-profile":
        return subprocess.call(["bash", os.path.join(HERE, "make-profile.sh")])
    if args.command == "doctor":
        return cmd_doctor(config, args)
    if not config.get("bundle_id") and args.command != "sign":
        raise SystemExit(f"no bundle_id: run setup-mac.sh or pass --bundle-id ({CONFIG})")
    if args.command == "sign":
        sign(config, args.ipa, args.out, pick_device(config))
        return 0
    if args.command == "install":
        devicectl(["device", "install", "app", "--device", pick_device(config)["identifier"],
                   args.app], timeout=1800)
        log("installed")
        return 0
    if args.command == "push":
        push_tree(pick_device(config), config["bundle_id"], args.stage)
        return 0
    if args.command == "probe":
        return cmd_probe(config, args)
    if args.command == "run":
        result = run_request(config, pick_device(config), load_json(args.request), args.out,
                             jit=args.jit, trace=args.trace, timeout=args.timeout,
                             trigger="none" if args.via == "file" else args.via)
        return 0 if result.get("status") == "ok" else 2
    if args.command == "matrix":
        return cmd_matrix(config, args)
    if args.command == "pull":
        last = pull(pick_device(config), config["bundle_id"], args.out)
        print(json.dumps({k: last.get(k) for k in ("run", "status", "message")} if last else None))
        return 0 if last else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
