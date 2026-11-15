# iOS bench: device performance runs without a person

An agent on the Windows PC builds or fetches an IPA, has the Mac sign and install it on the
iPhone, and runs Ocarina of Time (0400486024DEC000) in Kokiri Forest in the app's **bench mode**:
the app boots the game, plays a recorded input script to the measured place, waits for a steady
state, measures a standing window and a camera pan, writes `Documents/bench/result-<stamp>.json`
with the log beside it, stops the game and exits. The Mac copies the results back; the PC prints
a comparison table and a verdict per toggle set.

```
Windows PC                        Mac (admins-macbook-pro, 100.93.246.33)       iPhone18,2
bench.ps1 --ssh/scp-->  bench_mac.py: sign, install, push, launch --devicectl-->  Lichii bench mode
          <--results--             poll status.json, pull result + log   <--     Documents/bench/
bench_report.py (table, verdicts)
```

## Files

| File | Where it runs | What it does |
|---|---|---|
| `bench.ps1` | PC | Resolves the build (run id, artifact id, sha, .ipa or `installed`), copies the tools and plan to the Mac, runs the matrix there under `nohup`, copies `runs/` back to `%TEMP%\lichii-bench\device-runs\<date>\<session>\`, prints `summary.md`. `-DryRun` only writes the plan. |
| `bench_mac.py` | Mac | `devices`, `doctor`, `make-profile`, `sign`, `install`, `push`, `probe`, `run`, `matrix`, `pull` (see `--help`). Only Xcode tools: `devicectl`, `codesign`, `security`, `lldb`, `xctrace`. |
| `setup-mac.sh` | Mac | One-time check: Xcode 26, tools, signing identity, team id, writes `~/lichii-bench/config.json`, then `bench_mac.py doctor` (phone paired, Developer Mode, app installed, profile entitlements, StikDebug, keys/game/save in the app's Documents). Prints what is missing and how to fix it. |
| `make-profile.sh` | Mac | Builds an empty app with the bench bundle id and the memory entitlements once with Xcode's automatic signing, which registers the phone and the App ID capabilities and leaves a development profile in `~/lichii-bench/bench.mobileprovision`. |
| `build-ipa.sh` | Mac | Local build with the configure flags of `ios.yml` (alternative to the CI artifact). |
| `jit_lldb.py` | Mac (in lldb) | JIT without StikDebug: serves the app's `brk #0xf00d` protocol from the Mac's lldb (see below). |
| `bench_report.py` | PC or Mac | Table per window (fps, sd, 1 % low, p99, GPU ms, CPU %, P-core %, W, runnable %, waiting %, guest wake p99, memory) and verdicts. Tests: `test_bench_report.py`. |
| `stage/` | pushed to `Documents/bench/` | `profiles/*.json` (requests), `scripts/*.json` (input scripts), `saves/<name>/` (bench saves, not in git). |
| `plans/*.json` | Mac | Matrices for `bench.ps1 -Plan` / `bench_mac.py matrix`. |

## How a run starts (no touch on the screen)

Any of these makes the app a bench run; without them nothing changes:

- launch argument `--lichii-bench <profile|file>` (`xcrun devicectl device process launch ... <bundle id> --lichii-bench kokiri`),
- environment `LICHII_BENCH=<profile|file>` (`--environment-variables '{"LICHII_BENCH":"kokiri"}'`),
- the file `Documents/bench/request.json` present at launch (renamed to `request-used-<stamp>.json` when read, so the next normal launch is normal).

`<profile>` is `Documents/bench/profiles/<profile>.json`; `request` or `1` is `request.json`; a name
with `/` or `.json` is a file under `Documents/bench`. `--lichii-bench-experiments a,b` /
`LICHII_BENCH_EXPERIMENTS` add experiments, `--lichii-bench-run` / `LICHII_BENCH_RUN` set the
label. The bench arguments are removed before the launch parameters are read.
`bench_mac.py probe` checks all three triggers with a game-less probe request.

What the app does (`src/ios_frontend/lichii_bench.cpp`, logic in `bench_core.cpp`):

1. Before the platform layer starts: the request's `env` (overwriting), its `experiments` replace
   the saved Experiments for this process only (`keep_saved_experiments` keeps them too), its
   `settings` are set in memory and the config is not written while the bench runs; a game's own
   config cannot override them. `log_filter` is appended to the iOS log filter.
2. JIT: asks StikDebug (`jit.provider` `stikdebug`) or waits for a Mac debugger (`external`), up
   to `jit.timeout_s`; otherwise result `jit_not_granted` with what it saw, and exits.
3. Game: a program id is looked up in Documents, or `game` is a path under Documents. A `save`
   (`Documents/bench/saves/<name>/`) replaces the game's save folder for the run; the owner's save
   is moved to `Documents/bench/owner-save-backup/` and put back at the end (also at the next
   normal launch if a run was interrupted).
4. `script` plays through the controls bridge (the same path as the touch controls). Its clock
   stops while the game is paused.
5. Steady state from the Perf counters, not a fixed time: every second of the last `hold_s` had
   JIT compiling under `jit_percent_max` % of a core, texture + buffer uploads under
   `upload_mib_s_max` MiB/s and at least `min_fps` frames; `timeout_s` -> status `steady_timeout`
   (measured anyway, not compared).
6. Thermal gate: above `thermal.max_state` (0 nominal, 1 fair) the game waits paused up to
   `thermal.wait_s`, else status `too_hot` (the host waits and retries). At launch with
   `wait_s` 0 the run ends `too_hot` before booting.
7. Windows: `stand` (`measure.stand_s`) and `pan` (as long as the `measure.pan` script plays).
   Per window: frames, fps avg, 1 % low (1000 / mean of the slowest 1 % of frame times), frame
   time p50/p90/p95/p99/max (game frame timestamps), GPU ms/frame and busy %, command buffers,
   draws and render passes per frame, JIT %, uploads, guest core split, process CPU %, P-core share,
   W, runnable-not-running %, thermal state/pressure/Auto-Cool cap at start and end, footprint
   peak, memory left, and the raw `Perf:`/`Perf guest:`/`Perf threads:`/`Perf sched:`/`Perf passes:`
   lines of the window.
8. `result-<stamp>.json` + `result-<stamp>.log` (the full log) + `last.json` (same as the result),
   `status.json` (phase, polled by the Mac), then stops the game and exits (`"exit": false`
   keeps the app open).

## Request (`stage/profiles/kokiri.json`)

```json
{
  "mode": "run",                       // run | record | capture_save | probe
  "run": "label", "game": "0400486024DEC000",
  "save": {"name": "kokiri", "slot": 1, "restore_owner_save": true},
  "experiments": ["guest_cores_first"], "keep_saved_experiments": false,
  "env": {"EDEN_NX2_IDLE_SPIN_US": "20"}, "settings": {"oot_graphics_preset": "0"},
  "log_filter": "Render.Vulkan:Info",
  "jit": {"provider": "stikdebug", "timeout_s": 180},
  "thermal": {"max_state": 1, "wait_s": 600},
  "boot_timeout_s": 600, "script": "scripts/oot-title-to-kokiri.json", "script_timeout_s": 900,
  "steady": {"jit_percent_max": 5, "upload_mib_s_max": 8, "hold_s": 10, "timeout_s": 300, "min_fps": 1},
  "measure": {"stand_s": 75, "pan": "scripts/oot-pan.json", "pan_settle_s": 2},
  "record_s": 900, "exit": true
}
```

Script steps (a list, or `{"steps": [...]}`): `{"wait_ms": n}` / `{"wait_s": n}`,
`{"wait_frames": n}`, `{"wait_running": true}`, `{"wait_quiet_s": n}` (JIT quiet that long: the
end of a load), `{"press": "A" | ["ZL","A"], "hold_ms": 100 | "hold_frames": n, "after_ms": n}`,
`{"hold": "B"}`, `{"release": "B"}`, `{"stick": "left"|"right", "x": -1..1, "y": -1..1 (up +),
"ms"|"s"|"frames": n, "keep": false}`, `{"mark": "name"}` (time in the result), `{"log": "text"}`,
`{"repeat": n, "steps": [...]}`. Buttons: A B X Y LS RS L R ZL ZR PLUS MINUS UP DOWN LEFT RIGHT
(or DUP/DDOWN/...), SL SR HOME CAPTURE.

## One-time steps for the owner

1. On the Mac: System Settings > General > Sharing > Remote Login on (for this user), and add the
   PC's key: `cat lichii_mac.pub >> ~/.ssh/authorized_keys` (the PC has
   `%USERPROFILE%\.ssh\lichii_mac`). `bench.ps1 -Mac <user>@100.93.246.33` if the user is not `admin`.
2. Xcode 26 installed and opened once; Xcode > Settings > Accounts signed in with the Apple ID of
   the developer team (no password is stored anywhere by these scripts).
3. iPhone on USB, unlocked, Trust this computer; Settings > Privacy & Security > Developer Mode on;
   Settings > Display & Brightness > Auto-Lock > Never; charger connected.
4. On the Mac: `bash ~/lichii-bench/tools/setup-mac.sh` (after a first `bench.ps1 -DryRun` copy,
   or copy the folder by hand), then `python3 -I ~/lichii-bench/tools/bench_mac.py make-profile`.
   If make-profile fails: in Xcode make a new iOS App with the bundle id from
   `~/lichii-bench/config.json`, add the capabilities Increased Memory Limit and Extended Virtual
   Addressing, run it once on the phone.
5. The bench app is a separate app (its own bundle id, `dev.lichii.bench.<team>` by default) with
   its own Documents: copy the game (`.nup`), `prod.keys` and firmware into it once
   (Finder > iPhone > Files > the bench Lichii, or `devicectl device copy to`), as for a fresh
   install. Its settings start as a fresh install's (device recommended values).
6. StikDebug: allow it to open the bench Lichii (its pairing file and LocalDevVPN set up as for the
   normal install). The first `stikdebug://` request may ask to confirm on the phone.
7. Record once (below): the Kokiri save and the menu script. Time the pan.

## Recording the menu script and the save (once, on the device)

1. Save in Kokiri Forest in the bench app (play there once by hand, or copy a save in), then
   `bench_mac.py run --request stage/profiles/capture-save.json --out x` copies the game's save to
   `Documents/bench/saves/kokiri/`. Pull it with `devicectl device copy from` to keep it on the
   Mac (`stage/saves/kokiri/`, not committed).
2. `bench_mac.py run --request stage/profiles/record.json --out x --timeout 1200`: the app boots
   the game with that save and records every button and stick change of player 1 (touch or
   controller) until the game is stopped from the in-game menu. Play from the title screen to
   standing in Kokiri, wait a few seconds, stop the game. The script is
   `Documents/bench/scripts/recorded-<stamp>.json` (path in the result's `recorded_script`).
3. Copy it to `stage/scripts/oot-title-to-kokiri.json`; replace long waits in loads with
   `{"wait_quiet_s": 3}` so a slower build still lines up, and end with `{"mark": "kokiri"}`.
4. The pan: adjust `s` in `stage/scripts/oot-pan.json` until one turn comes back to the start.

## Running

```powershell
# PC: baseline + three toggle sets, 3 repeats each, interleaved, from a CI run
.\tools\nx2\ios_bench\bench.ps1 -Build 43611345294 -Toggles guest_cores_first, oot_ideal_affinity, guest_idle_spin
.\tools\nx2\ios_bench\bench.ps1 -Build installed -Plan tools\nx2\ios_bench\plans\cpu-sched.json
```

On the Mac directly: `bench_mac.py run --request stage/profiles/kokiri.json --out runs/one [--trace]`
(`--trace` records `metal.trace` with Instruments' Metal System Trace during the stand window).

A comparison counts a window only when the run's status is `ok`, the thermal state at the start
was within the request, the game was not paused, the window completed and had 100+ frames.
A cell is "faster"/"slower" than the baseline only past both 3 % and twice the combined standard
error of the repeats.

## JIT from the Mac without StikDebug

The app's JIT on iOS 26 (TXM) is the universal breakpoint protocol (`common/ios_jit.cpp`):
`brk #0xf00d` with x16 = 1 asks the debugger for an R-X region (StikDebug: gdb-remote
`_M<len>,rx`) and a one-byte debugger write to every 16 KiB page of it (`M<addr>,1:69`,
StikDebug's `handleJITPageWrite`), x16 = 0 asks it to detach. Nothing in that needs StikDebug
itself: Xcode 16+'s lldb attaches to an iOS 17+ device process through CoreDevice
(`device select`, `device process attach --pid`), and Flutter's tool does the same page-touching
from a Mac's lldb for its own iOS 26 debug JIT (`NOTIFY_DEBUGGER_ABOUT_RX_PAGES`). `jit_lldb.py`
serves the protocol with `SBProcess.AllocateMemory(rx)` and `WriteMemory` per page;
`bench_mac.py run --jit lldb` launches the app with `"jit": {"provider": "external"}` and attaches.
It has not run on the phone yet: StikDebug stays the default. If it works it removes StikDebug,
its pairing file and VPN from the loop; expect ~25 k page writes for a 384 MiB pool (tens of
seconds over USB).

## Not verified yet (no Mac access while this was written)

- Every `devicectl`/`lldb`/`xctrace` call and JSON key is from Apple's documentation and other
  projects' use, not from a run on this Mac; `bench_mac.py doctor` and `probe` are the first things
  to run and report the exact error if a key or option differs.
- `devicectl device copy to` into `Documents/bench/...` before the folder exists on the phone.
- The menu script and the pan timing (placeholders in `stage/scripts/`).
